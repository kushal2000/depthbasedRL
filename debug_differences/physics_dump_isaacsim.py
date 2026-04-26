"""Dump per-collider / per-rigid-body physics params from the Isaac Sim
SimToolReal env (8 envs, all DR off) to JSON for cross-backend comparison.

Mirrors `policy_eval_isaacsim.py` for env construction. After scene setup,
walks env_0's Robot, Table, Object subtrees on the live USD stage:

  - For each prim with `UsdPhysics.RigidBodyAPI` → dump mass + inertia + COM
    (via UsdPhysics.MassAPI / PhysxRigidBodyAPI).
  - For each prim with `UsdPhysics.CollisionAPI` → dump contact_offset,
    rest_offset, collision_enabled (via PhysxCollisionAPI), plus the
    friction/restitution of any bound `UsdPhysics.PhysicsMaterial`.

Output schema mirrors `physics_dump_isaacgym.py` so `compare_physics.py`
can do field-by-field diffs.

    .venv_isaacsim/bin/python debug_differences/physics_dump_isaacsim.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--out", type=str,
                        default=str(REPO_ROOT / "debug_differences/data/physics/isaacsim.json"))
    return parser


def _launch_app():
    from isaaclab.app import AppLauncher
    parser = _build_parser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    args.enable_cameras = False
    return AppLauncher(args).app, args


_app, _args = _launch_app()


def _bound_material_path(prim) -> str | None:
    """Return the path of a PhysicsMaterial bound to ``prim`` via the
    ``physics`` purpose binding, or None if no binding."""
    from pxr import UsdShade
    rel = prim.GetRelationship("material:binding:physics")
    if not rel.IsValid():
        return None
    targets = rel.GetTargets()
    if not targets:
        return None
    return targets[0].pathString


def _dump_material(stage, material_path: str) -> dict | None:
    from pxr import UsdPhysics
    prim = stage.GetPrimAtPath(material_path)
    if not prim.IsValid():
        return None
    api = UsdPhysics.MaterialAPI(prim)
    if not api:
        return {"path": material_path, "static_friction": None,
                "dynamic_friction": None, "restitution": None}
    static = api.GetStaticFrictionAttr()
    dynamic = api.GetDynamicFrictionAttr()
    restitution = api.GetRestitutionAttr()
    return {
        "path": material_path,
        "static_friction": float(static.Get()) if static and static.HasAuthoredValue() else None,
        "dynamic_friction": float(dynamic.Get()) if dynamic and dynamic.HasAuthoredValue() else None,
        "restitution": float(restitution.Get()) if restitution and restitution.HasAuthoredValue() else None,
    }


def _dump_subtree(stage, root_prim_path: str, label: str) -> dict:
    """Walk root_prim_path's subtree (after un-instancing) and pull rigid-body
    + collision-property values from the live USD stage."""
    from pxr import Usd, UsdPhysics, PhysxSchema
    # Re-use the same un-instance helper used by scene_utils.
    from isaacsimenvs.tasks.simtoolreal.utils.scene_utils import _uninstance_subtree
    _uninstance_subtree(root_prim_path)

    root = stage.GetPrimAtPath(root_prim_path)
    if not root.IsValid():
        return {"actor_name": label, "links": [], "shapes": []}

    links: list[dict] = []
    shapes: list[dict] = []
    queue = [root]
    visited_materials: dict[str, dict] = {}

    while queue:
        prim = queue.pop(0)
        path = prim.GetPath().pathString

        # Rigid body schema -> dump mass / inertia / COM.
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            mass_api = UsdPhysics.MassAPI(prim) if prim.HasAPI(UsdPhysics.MassAPI) else None
            mass_attr = mass_api.GetMassAttr() if mass_api else None
            inertia_attr = mass_api.GetDiagonalInertiaAttr() if mass_api else None
            com_attr = mass_api.GetCenterOfMassAttr() if mass_api else None
            links.append({
                "name": prim.GetName(),
                "path": path,
                "mass": float(mass_attr.Get()) if (mass_attr and mass_attr.HasAuthoredValue()) else None,
                "inertia_diag": (
                    [float(x) for x in inertia_attr.Get()]
                    if (inertia_attr and inertia_attr.HasAuthoredValue()) else None
                ),
                "com": (
                    [float(x) for x in com_attr.Get()]
                    if (com_attr and com_attr.HasAuthoredValue()) else None
                ),
            })

        # Collision schema -> dump contact_offset / rest_offset / friction.
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            col_api = UsdPhysics.CollisionAPI(prim)
            physx_col = PhysxSchema.PhysxCollisionAPI(prim) if prim.HasAPI(PhysxSchema.PhysxCollisionAPI) else None

            ce_attr = col_api.GetCollisionEnabledAttr()
            collision_enabled = (
                bool(ce_attr.Get()) if (ce_attr and ce_attr.HasAuthoredValue()) else None
            )

            co_attr = physx_col.GetContactOffsetAttr() if physx_col else None
            ro_attr = physx_col.GetRestOffsetAttr() if physx_col else None
            contact_offset = (
                float(co_attr.Get()) if (co_attr and co_attr.HasAuthoredValue()) else None
            )
            rest_offset = (
                float(ro_attr.Get()) if (ro_attr and ro_attr.HasAuthoredValue()) else None
            )

            mat_path = _bound_material_path(prim)
            if mat_path and mat_path not in visited_materials:
                visited_materials[mat_path] = _dump_material(stage, mat_path) or {}
            mat_info = visited_materials.get(mat_path) if mat_path else None

            shapes.append({
                "path": path,
                "name": prim.GetName(),
                "collision_enabled": collision_enabled,
                "contact_offset": contact_offset,
                "rest_offset": rest_offset,
                "material_path": mat_path,
                "material": mat_info,
            })

        queue.extend(prim.GetChildren())

    return {"actor_name": label, "links": links, "shapes": shapes}


def main() -> None:
    args = _args

    import gymnasium as gym
    import torch

    import isaacsimenvs  # noqa: F401  registers gym envs
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import SimToolRealEnvCfg
    from omni.usd import get_context

    cfg = SimToolRealEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.assets.num_assets_per_type = 1

    dr = cfg.domain_randomization
    dr.use_obs_delay = False
    dr.use_action_delay = False
    dr.use_object_state_delay_noise = False
    dr.object_scale_noise_multiplier_range = (1.0, 1.0)
    dr.joint_velocity_obs_noise_std = 0.0
    dr.force_scale = 0.0
    dr.torque_scale = 0.0
    dr.force_prob_range = (0.0001, 0.0001)
    dr.torque_prob_range = (0.0001, 0.0001)

    rs = cfg.reset
    rs.reset_position_noise_x = 0.0
    rs.reset_position_noise_y = 0.0
    rs.reset_position_noise_z = 0.0
    rs.reset_dof_pos_random_interval_arm = 0.0
    rs.reset_dof_pos_random_interval_fingers = 0.0
    rs.reset_dof_vel_random_interval = 0.0
    rs.table_reset_z_range = 0.0
    rs.fixed_start_pose = (0.0, 0.0, rs.table_reset_z + rs.table_object_z_offset, 1.0, 0.0, 0.0, 0.0)

    env = gym.make("Isaacsimenvs-SimToolReal-Direct-v0", cfg=cfg)
    inner = env.unwrapped
    inner._replay_target_lab_order = None

    obs, _ = env.reset()
    obs, _, _, _, _ = env.step(
        torch.zeros((args.num_envs, cfg.action_space), device=inner.device)
    )

    stage = get_context().get_stage()
    sim_cfg_dict = {
        "dt": float(cfg.sim.dt),
        "decimation": int(cfg.decimation),
        "physx_solver_position_iterations_default": int(getattr(cfg.sim.physx, "min_position_iteration_count", -1)),
        "physx_solver_velocity_iterations_default": int(getattr(cfg.sim.physx, "min_velocity_iteration_count", -1)),
        "physx_bounce_threshold_velocity": float(getattr(cfg.sim.physx, "bounce_threshold_velocity", -1.0)),
        "physx_friction_offset_threshold": float(getattr(cfg.sim.physx, "friction_offset_threshold", -1.0)),
        "gravity": list(cfg.sim.gravity),
    }

    out: dict = {"sim_cfg": sim_cfg_dict}
    out["robot"] = _dump_subtree(stage, "/World/envs/env_0/Robot", "Robot")
    out["table"] = _dump_subtree(stage, "/World/envs/env_0/Table", "Table")
    out["object"] = _dump_subtree(stage, "/World/envs/env_0/Object", "Object")
    out["goal_viz"] = _dump_subtree(stage, "/World/envs/env_0/GoalViz", "GoalViz")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[sim] wrote {out_path}")


if __name__ == "__main__":
    import os
    import sys
    main()
    del _app
    sys.stdout.flush()
    os._exit(0)
