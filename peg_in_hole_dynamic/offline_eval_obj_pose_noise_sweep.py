#!/usr/bin/env python3
"""Object-pose observation-noise sweep for a state-obs SAPG teacher.

Sweeps the noise applied to the *observed* object pose -- translational
(``object_state_xyz_noise_std``) and rotational
(``object_state_rotation_noise_degrees``) -- together, from 0 up to
``--xyz-max`` / ``--rot-max`` over ``--points`` evenly spaced settings, and
reports the success rate at each.

Obs/action/object-state delays stay ON for every point, so the pose-noise
magnitude is the only thing varying.

Everything else (problem, goal_mode, tolerances, retract, reset noise, the
rest of the DR block) is replayed from the checkpoint's own training
``overrides.yaml`` via --train-overrides.

One Kit boot for the whole sweep: the env is built once and
``env.cfg.domain_randomization`` is mutated between points, which takes effect
immediately because ``_apply_object_state_dr`` re-reads the cfg every step.

Metric matches offline_eval_teacher_robustness.py: ONE episode per env, outcome
latched from env._termination_reasons at the first termination (env._successes
is zeroed by the auto-reset inside step() and cannot be used).

Example:

    RUN=train_dir/teachers_final/play2win_peg_insertion/lpeg_tol0p5mm_finetune_rgf0_dr_2026-05-15_21-56-40
    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python -u \\
      peg_in_hole_dynamic/offline_eval_obj_pose_noise_sweep.py \\
      --teacher-checkpoint $RUN/*/last/model.pth \\
      --train-overrides $RUN/.hydra/overrides.yaml \\
      --num-envs 512 --points 10 --xyz-max 0.5 --rot-max 45.0 \\
      --output-json /tmp/sweep.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from peg_in_hole_dynamic.offline_eval_teacher_robustness import (  # noqa: E402
    _REASONS,
    _load_train_overrides,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="Isaacsimenvs-PegInHoleDepthStudent-Direct-v0")
    p.add_argument("--teacher-checkpoint", required=True)
    p.add_argument("--train-overrides", required=True,
                   help="path to the checkpoint's .hydra/overrides.yaml")
    p.add_argument("--override", action="append", default=[], metavar="KEY=VALUE",
                   help="extra env override applied after --train-overrides")
    p.add_argument("--points", type=int, default=10,
                   help="sweep points, evenly spaced from 0 to the maxima inclusive")
    p.add_argument("--xyz-max", type=float, default=0.5, help="metres")
    p.add_argument("--rot-max", type=float, default=45.0, help="degrees")
    p.add_argument("--hold-xyz", type=float, default=None,
                   help="freeze translational σ (m) and sweep rotation alone")
    p.add_argument("--hold-rot", type=float, default=None,
                   help="freeze rotational σ (deg) and sweep translation alone")
    p.add_argument("--num-envs", type=int, default=512)
    p.add_argument("--max-steps-per-episode", type=int, default=600)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--delay-max", type=int, default=3)
    p.add_argument("--early-drop-steps", type=int, default=100)
    p.add_argument("--rl-device", default="cuda:0")
    p.add_argument("--sim-device", default="cuda:0")
    p.add_argument("--headless", action="store_true", default=True)
    p.add_argument("--output-json", required=True)
    return p.parse_args()


def _reseed(seed: int) -> None:
    import numpy as np
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def _sweep_points(n: int, xyz_max: float, rot_max: float,
                  hold_xyz: float | None = None,
                  hold_rot: float | None = None) -> list[tuple[str, float, float]]:
    """n points from 0 to the maxima inclusive.

    By default both axes scale together. Pass --hold-rot to freeze rotation at a
    constant (usually the training σ) and sweep translation alone, or --hold-xyz
    for the mirror case -- that isolates which axis drives the collapse, which a
    joint sweep cannot attribute.
    """
    if n < 2:
        return [("xyz0.000_rot0.0", 0.0, 0.0)]
    out = []
    for i in range(n):
        f = i / (n - 1)
        xyz = hold_xyz if hold_xyz is not None else f * xyz_max
        rot = hold_rot if hold_rot is not None else f * rot_max
        out.append((f"xyz{xyz:.3f}m_rot{rot:.1f}deg", float(xyz), float(rot)))
    return out


def main() -> int:
    args = _parse_args()
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

    from isaaclab.app import AppLauncher
    launcher_parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(launcher_parser)
    launcher_args, _ = launcher_parser.parse_known_args([])
    launcher_args.headless = bool(args.headless)
    launcher_args.enable_cameras = True
    app = AppLauncher(launcher_args).app

    import torch
    import isaacsimenvs  # noqa: F401
    from isaacsimenvs.utils.rlgames_utils import register_rlgames_env, teacher_env_info
    from isaacsimenvs.dagger.teacher import Teacher

    from peg_in_hole_dynamic.eval_isaacsim import (
        _apply_env_overrides,
        _configure_agent,
        _instantiate_env,
        _load_env_cfg,
        _set_attr_path,
    )
    from peg_in_hole_dynamic.eval_student_isaacsim import _extract_teacher_obs

    cfg = _load_env_cfg(args.task)
    _apply_env_overrides(
        cfg,
        problem="Lpeg.tol0p5mm",       # overwritten by the training overrides
        goal_mode="finalGoalOnly",     # ditto
        random_goal_fraction=0.0,
        insertion_success_tolerance=0.010,
        retract_success_tolerance=0.005,
        num_envs=int(args.num_envs),
        sim_device=args.sim_device,
        sdf=False,
        keep_dr=True,                  # training DR block is authoritative
        extra_overrides={},
    )
    dmax = max(1, int(args.delay_max))
    cfg.domain_randomization.obs_delay_max = dmax
    cfg.domain_randomization.action_delay_max = dmax
    cfg.student_obs.camera_delay_max = dmax

    train_overrides = _load_train_overrides(args.train_overrides)
    for key, value in train_overrides.items():
        _set_attr_path(cfg, key, value)
    print(f"=> applied {len(train_overrides)} training overrides", flush=True)

    cli_overrides = {}
    if args.override:
        import yaml as _yaml
        for item in args.override:
            key, _, value = str(item).partition("=")
            if not _:
                raise SystemExit(f"--override must be KEY=VALUE, got {item!r}")
            cli_overrides[key] = _yaml.safe_load(value)
            _set_attr_path(cfg, key, cli_overrides[key])
        print(f"=> cli overrides: {cli_overrides}", flush=True)

    print(f"=> problem={cfg.peg_in_hole.problem}  goal_mode={cfg.peg_in_hole.goal_mode}  "
          f"retract={cfg.peg_in_hole.enable_retract}", flush=True)

    env = _instantiate_env(args.task, cfg)
    agent_cfg = _configure_agent(
        args.task, "rl_games_sapg_cfg_entry_point",
        rl_device=args.rl_device, num_envs=int(args.num_envs),
        deterministic=True, games=1, extra_overrides={},
    )
    clip_obs = float(agent_cfg["params"]["env"].get("clip_observations", math.inf))
    clip_actions = float(agent_cfg["params"]["env"].get("clip_actions", math.inf))
    wrapped = register_rlgames_env(env, rl_device=args.rl_device,
                                   clip_obs=clip_obs, clip_actions=clip_actions)

    env_info_teacher = teacher_env_info(wrapped)
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    student_agent_cfg = load_cfg_from_registry(args.task, "rl_games_dagger_sapg_cfg_entry_point")
    dagger_block = student_agent_cfg["params"]["config"].get("dagger", {})
    teacher = Teacher(
        task_id=str(dagger_block.get("teacher_task_id", "Isaacsimenvs-PegInHole-Direct-v0")),
        agent_key=str(dagger_block.get("teacher_agent_key", "rl_games_sapg_cfg_entry_point")),
        checkpoint_path=args.teacher_checkpoint,
        num_envs=int(args.num_envs),
        rl_device=args.rl_device,
        env_info=env_info_teacher,
    )

    def _run_point(name: str, xyz: float, rot: float) -> dict:
        dr = env.cfg.domain_randomization
        # Delays stay ON for every point; only the pose noise varies.
        dr.use_obs_delay = True
        dr.use_action_delay = True
        dr.use_object_state_delay_noise = True
        dr.object_state_xyz_noise_std = float(xyz)
        dr.object_state_rotation_noise_degrees = float(rot)

        _reseed(args.seed)
        teacher.reset()
        wrapped.reset()
        _reseed(args.seed)          # same init states at every point
        obs = wrapped.reset()

        n = int(args.num_envs)
        dev = args.rl_device
        finished = torch.zeros(n, dtype=torch.bool, device=dev)
        end_step = torch.full((n,), -1, dtype=torch.long, device=dev)
        flags = {k: torch.zeros(n, dtype=torch.bool, device=dev) for k in _REASONS}
        inserted = torch.zeros(n, dtype=torch.bool, device=dev)

        # +1: truncation fires when episode_length_buf >= max_episode_length, so
        # a timing-out env is only observed as done on the step AFTER the cap.
        # Looping to exactly max_steps leaves those envs `unfinished` and drops
        # them from the denominator, which silently inflates the success rate.
        for step in range(int(args.max_steps_per_episode) + 1):
            tobs = _extract_teacher_obs(obs)
            act = teacher.get_action(tobs)
            step_out = wrapped.step(act)
            obs = step_out[0] if isinstance(step_out, tuple) else step_out
            inserted |= env.retract_phase & ~finished
            done = env.reset_terminated | env.reset_time_outs
            newly = done & ~finished
            if bool(newly.any()):
                for k in _REASONS:
                    flags[k] |= env._termination_reasons[k] & newly
                end_step = torch.where(newly, torch.full_like(end_step, step), end_step)
                finished |= newly
            if bool(finished.all()):
                break

        retracted = flags["max_successes"]
        inserted |= retracted
        drop = flags["fall"] | flags["dropped"]
        early_drop = drop & (end_step >= 0) & (end_step < int(args.early_drop_steps))

        def _rates(mask):
            k = int(mask.sum().item())
            if k == 0:
                return {"n": 0, "inserted": 0, "retracted": 0,
                        "insertion_rate": float("nan"), "retract_rate": float("nan")}
            ins = int((inserted & mask).sum().item())
            ret = int((retracted & mask).sum().item())
            return {"n": k, "inserted": ins, "retracted": ret,
                    "insertion_rate": ins / k, "retract_rate": ret / k}

        return {
            "name": name,
            "obj_xyz_noise_std_m": float(xyz),
            "obj_rot_noise_deg": float(rot),
            "all_envs": _rates(finished),
            "early_drop_filtered": _rates(finished & ~early_drop),
            "termination_counts": {
                k: int((flags[k] & finished).sum().item()) for k in _REASONS
            },
            "n_dropped_early": int(early_drop.sum().item()),
            "unfinished_envs": int((~finished).sum().item()),
        }

    points = _sweep_points(int(args.points), float(args.xyz_max), float(args.rot_max),
                           hold_xyz=args.hold_xyz, hold_rot=args.hold_rot)
    print(f"=> {len(points)} sweep points, num_envs={args.num_envs}, seed={args.seed}",
          flush=True)

    results = []
    for name, xyz, rot in points:
        t0 = time.time()
        try:
            res = _run_point(name, xyz, rot)
        except Exception as exc:
            print(f"[point] {name}: FAILED -- {exc}", flush=True)
            traceback.print_exc()
            res = {"name": name, "obj_xyz_noise_std_m": xyz, "obj_rot_noise_deg": rot,
                   "error": str(exc), "early_drop_filtered": {}}
        results.append(res)
        e = res.get("early_drop_filtered", {}) or {}
        nan = float("nan")
        print(f"[SWEEP] xyz={xyz:6.3f}m rot={rot:5.1f}deg  "
              f"no_early_drop: retract={e.get('retract_rate', nan):.4f} "
              f"insertion={e.get('insertion_rate', nan):.4f} (n={e.get('n', 0)})  "
              f"({time.time() - t0:.1f}s)", flush=True)

    with open(args.output_json, "w") as fp:
        json.dump({
            "teacher_checkpoint": str(args.teacher_checkpoint),
            "num_envs": int(args.num_envs),
            "seed": int(args.seed),
            "points": int(args.points),
            "xyz_max_m": float(args.xyz_max),
            "rot_max_deg": float(args.rot_max),
            "hold_xyz_m": args.hold_xyz,
            "hold_rot_deg": args.hold_rot,
            "early_drop_steps": int(args.early_drop_steps),
            "train_overrides": train_overrides,
            "cli_overrides": cli_overrides,
            "results": results,
        }, fp, indent=2)
    print(f"=> wrote {args.output_json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
