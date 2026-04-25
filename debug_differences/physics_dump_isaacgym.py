"""Dump per-rigid-body / per-shape physics params from the Isaac Gym
SimToolReal env (8 envs, all DR off) to JSON for cross-backend comparison.

Mirrors `policy_eval_isaacgym.py` for env construction. Walks env_0's
robot, table, and object actors via `gym.get_actor_rigid_body_properties`
and `gym.get_actor_rigid_shape_properties` and emits a structured JSON:

    {
      "sim_cfg": {dt, substeps, solver iterations, ...},
      "robot":  {"actor_name": ..., "links": [{name, mass, inertia, com}, ...],
                  "shapes": [{rb_name, friction, restitution, thickness,
                              contact_offset, rest_offset, ...}, ...]},
      "table":  {...same...},
      "object": {...same...}
    }

    .venv/bin/python debug_differences/physics_dump_isaacgym.py
"""

from __future__ import annotations

# isort: off
from isaacgym import gymapi, gymtorch, gymutil  # noqa: F401
import torch  # noqa: F401
# isort: on

import argparse
import json
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from isaacgymenvs.tasks import isaacgym_task_map
from isaacgymenvs.utils.reformat import omegaconf_to_dict


REPO_ROOT = Path(__file__).resolve().parents[1]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--out", type=str,
                        default=str(REPO_ROOT / "debug_differences/data/physics/isaacgym.json"))
    return parser


def _build_cfg(args) -> dict:
    cfg_dir = str((REPO_ROOT / "isaacgymenvs" / "cfg").resolve())
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "task=SimToolReal",
                f"task.env.numEnvs={args.num_envs}",
                "task.env.numAssetsPerType=1",
                # All randomness off so the dumped values are deterministic.
                "task.env.resetPositionNoiseX=0.0",
                "task.env.resetPositionNoiseY=0.0",
                "task.env.resetPositionNoiseZ=0.0",
                "task.env.randomizeObjectRotation=False",
                "task.env.resetDofPosRandomIntervalFingers=0.0",
                "task.env.resetDofPosRandomIntervalArm=0.0",
                "task.env.resetDofVelRandomInterval=0.0",
                "task.env.tableResetZRange=0.0",
                "task.env.forceScale=0.0",
                "task.env.torqueScale=0.0",
                "task.env.useObsDelay=False",
                "task.env.useActionDelay=False",
                "task.env.useObjectStateDelayNoise=False",
                "task.env.objectScaleNoiseMultiplierRange=[1.0,1.0]",
                "task.env.capture_viewer=False",
                "task.env.capture_video=False",
                "task.env.episodeLength=600",
                "task.env.resetWhenDropped=False",
                "task.env.useFixedGoalStates=False",
                "task.env.fixedGoalStatesJsonPath=null",
            ],
        )
    return omegaconf_to_dict(cfg.task)


def _dump_actor(env, env_ptr, actor_handle: int, label: str) -> dict:
    """Pull per-link mass/inertia and per-shape contact properties for one actor."""
    rb_props = env.gym.get_actor_rigid_body_properties(env_ptr, actor_handle)
    shape_props = env.gym.get_actor_rigid_shape_properties(env_ptr, actor_handle)
    rb_names = env.gym.get_actor_rigid_body_names(env_ptr, actor_handle)

    links = []
    for name, p in zip(rb_names, rb_props):
        links.append({
            "name": name,
            "mass": float(p.mass),
            "com": [float(p.com.x), float(p.com.y), float(p.com.z)],
            "inertia_diag": [float(p.inertia.x.x), float(p.inertia.y.y), float(p.inertia.z.z)],
            "inertia_off": [float(p.inertia.x.y), float(p.inertia.x.z), float(p.inertia.y.z)],
        })

    shapes = []
    for i, sp in enumerate(shape_props):
        shapes.append({
            "shape_idx": i,
            "friction": float(sp.friction),
            "rolling_friction": float(getattr(sp, "rolling_friction", 0.0)),
            "torsion_friction": float(getattr(sp, "torsion_friction", 0.0)),
            "restitution": float(sp.restitution),
            "thickness": float(sp.thickness),
            "contact_offset": float(getattr(sp, "contact_offset", -1.0)),
            "rest_offset": float(getattr(sp, "rest_offset", -1.0)),
            "filter": int(getattr(sp, "filter", -1)),
        })

    return {"actor_name": label, "links": links, "shapes": shapes}


def main() -> None:
    args = _build_parser().parse_args()

    cfg_dict = _build_cfg(args)
    env = isaacgym_task_map["SimToolReal"](
        cfg=cfg_dict,
        rl_device="cuda:0",
        sim_device="cuda:0",
        graphics_device_id=0,
        headless=True,
        virtual_screen_capture=False,
        force_render=False,
    )
    # Trigger initial reset so actor handles are valid.
    obs_dict, _, _, _ = env.step(
        torch.zeros((env.num_envs, env.num_hand_arm_dofs), device="cuda:0")
    )

    env_ptr = env.envs[0]
    # Find actor handles by name in env_0.
    n_actors = env.gym.get_actor_count(env_ptr)
    handles_by_name: dict[str, int] = {}
    for i in range(n_actors):
        handle = env.gym.get_actor_handle(env_ptr, i)
        name = env.gym.get_actor_name(env_ptr, handle)
        handles_by_name[name] = handle

    print(f"[gym] env_0 actors: {sorted(handles_by_name.keys())}")

    role_to_actor = {}
    for role, candidates in [
        ("robot", ("robot",)),
        ("table", ("table", "table_object")),
        ("object", ("object",)),
    ]:
        for c in candidates:
            if c in handles_by_name:
                role_to_actor[role] = (c, handles_by_name[c])
                break
    missing = [r for r in ("robot", "table", "object") if r not in role_to_actor]
    if missing:
        print(f"[gym] WARN: missing actors for roles: {missing}")

    sim_params = env.gym.get_sim_params(env.sim)
    sim_cfg = {
        "dt": float(sim_params.dt),
        "substeps": int(sim_params.substeps),
        "physx_solver_position_iterations": int(getattr(sim_params.physx, "num_position_iterations", -1)),
        "physx_solver_velocity_iterations": int(getattr(sim_params.physx, "num_velocity_iterations", -1)),
        "physx_contact_offset_default": float(getattr(sim_params.physx, "contact_offset", -1.0)),
        "physx_rest_offset_default": float(getattr(sim_params.physx, "rest_offset", -1.0)),
        "physx_bounce_threshold_velocity": float(getattr(sim_params.physx, "bounce_threshold_velocity", -1.0)),
        "physx_friction_offset_threshold": float(getattr(sim_params.physx, "friction_offset_threshold", -1.0)),
    }

    out: dict = {"sim_cfg": sim_cfg}
    for role, (actor_name, handle) in role_to_actor.items():
        out[role] = _dump_actor(env, env_ptr, handle, actor_name)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[gym] wrote {out_path} — roles dumped: {list(role_to_actor.keys())}")


if __name__ == "__main__":
    main()
