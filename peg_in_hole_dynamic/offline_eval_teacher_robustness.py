#!/usr/bin/env python3
"""Teacher-only robustness eval: state-obs SAPG policy under the same
delays / cam-rand / depth-aug / table xy / table yaw / table scale grid as
offline_eval_robustness.py.

Cam-rand and depth-aug are no-ops for the teacher (state obs has no camera
channel), but kept in the matrix so rows align with the student eval.
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


SETTINGS = [
    ("baseline",        {"delays": False, "cam": False, "depth": False, "xy_m": 0.00, "yaw_deg": 0.0}),
    ("delays",          {"delays": True,  "cam": False, "depth": False, "xy_m": 0.00, "yaw_deg": 0.0}),
    ("cam_pose_rand",   {"delays": False, "cam": True,  "depth": False, "xy_m": 0.00, "yaw_deg": 0.0}),
    ("depth_aug",       {"delays": False, "cam": False, "depth": True,  "xy_m": 0.00, "yaw_deg": 0.0}),
    ("table_xy_3cm",    {"delays": False, "cam": False, "depth": False, "xy_m": 0.03, "yaw_deg": 0.0}),
    ("table_yaw_5deg",  {"delays": False, "cam": False, "depth": False, "xy_m": 0.00, "yaw_deg": 5.0}),
    ("table_xy_yaw",    {"delays": False, "cam": False, "depth": False, "xy_m": 0.03, "yaw_deg": 5.0}),
    ("all_realistic",   {"delays": True,  "cam": True,  "depth": True,  "xy_m": 0.03, "yaw_deg": 5.0}),
]


def _parse_pair(s: str) -> tuple[float, float]:
    a, b = s.split(",")
    return float(a), float(b)


# Training overrides we must NOT replay at eval time: `num_envs` is set from
# --num-envs (training used 12288), and agent./hydra. keys aren't env config.
_TRAIN_OVERRIDE_SKIP = {"env.scene.num_envs"}

# Keys of env._termination_reasons, recorded per env at its first termination.
_REASONS = ("max_successes", "fall", "dropped", "hand_far", "timeout")


def _load_train_overrides(path: str) -> dict:
    """Parse a Hydra ``overrides.yaml`` into an ``{env.a.b: value}`` dict.

    Used by --exact-train to reproduce the checkpoint's own training env
    config instead of the eval defaults, which silently differ (no object-
    state noise, no hole yaw, no retract, clean goal yaw).
    """
    import yaml

    with open(path) as fp:
        raw = yaml.safe_load(fp)

    out = {}
    for entry in raw:
        key, _, value = str(entry).partition("=")
        if not _ or key.startswith(("agent.", "hydra.")) or key in _TRAIN_OVERRIDE_SKIP:
            continue
        try:
            out[key] = yaml.safe_load(value)
        except yaml.YAMLError:
            out[key] = value
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="Isaacsimenvs-PegInHoleDepthStudent-Direct-v0")
    p.add_argument("--problem", default="Lpeg.tol0p5mm")
    p.add_argument("--goal-mode", default="finalGoalOnly")
    p.add_argument("--teacher-checkpoint", required=True)
    p.add_argument("--num-envs", type=int, default=10)
    p.add_argument("--max-steps-per-episode", type=int, default=600)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--delay-max", type=int, default=3)
    p.add_argument("--rl-device", default="cuda:0")
    p.add_argument("--sim-device", default="cuda:0")
    p.add_argument("--headless", action="store_true", default=True)
    p.add_argument("--table-scale-x", type=_parse_pair, default=(1.0, 1.0))
    p.add_argument("--table-scale-y", type=_parse_pair, default=(1.0, 1.0))
    p.add_argument("--table-scale-n", type=int, default=1)
    p.add_argument("--output-json", required=True)
    p.add_argument("--train-overrides", default=None,
                   help="path to the checkpoint's .hydra/overrides.yaml")
    p.add_argument("--override", action="append", default=[], metavar="KEY=VALUE",
                   help="extra env override applied AFTER --train-overrides, "
                        "e.g. env.domain_randomization.force_scale=20.0. Repeatable.")
    p.add_argument("--early-drop-steps", type=int, default=100,
                   help="a fall/drop before this step counts as an unstable "
                        "initial placement, not a policy failure")
    p.add_argument("--exact-train", action="store_true",
                   help="reproduce the training env config exactly: apply "
                        "--train-overrides on top of the eval config, keep DR "
                        "on, and run a single 'train_exact' setting instead of "
                        "the robustness grid.")
    args = p.parse_args()
    if args.exact_train and not args.train_overrides:
        p.error("--exact-train requires --train-overrides")
    return args


def _reseed(seed: int) -> None:
    import numpy as np
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def _apply_setting(env, t: dict) -> None:
    so = env.cfg.student_obs
    dr = env.cfg.domain_randomization
    so.use_camera_delay = bool(t["delays"])
    dr.use_obs_delay = bool(t["delays"])
    dr.use_action_delay = bool(t["delays"])
    so.use_camera_pose_rand = bool(t["cam"])
    so.use_depth_aug = bool(t["depth"])
    env.cfg.reset.table_reset_xy_range_m = (float(t["xy_m"]), float(t["xy_m"]))
    env.cfg.reset.table_reset_yaw_range_deg = float(t["yaw_deg"])


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
        problem=args.problem,
        goal_mode=args.goal_mode,
        random_goal_fraction=0.0,
        insertion_success_tolerance=0.010,
        retract_success_tolerance=0.005,
        num_envs=int(args.num_envs),
        sim_device=args.sim_device,
        sdf=False,
        keep_dr=bool(args.exact_train),
        extra_overrides={},
    )
    dmax = max(1, int(args.delay_max))
    cfg.student_obs.camera_delay_max = dmax
    cfg.domain_randomization.obs_delay_max = dmax
    cfg.domain_randomization.action_delay_max = dmax

    train_overrides = {}
    if args.exact_train:
        # Applied last so the checkpoint's own training values win over every
        # eval default above (problem, goal_mode, tolerances, retract, hole
        # yaw, goal obs noise, and the full DR block).
        train_overrides = _load_train_overrides(args.train_overrides)
        for key, value in train_overrides.items():
            _set_attr_path(cfg, key, value)
        print(f"=> exact-train: applied {len(train_overrides)} overrides from "
              f"{args.train_overrides}", flush=True)
    else:
        cfg.student_obs.use_camera_delay = False
        cfg.student_obs.use_camera_pose_rand = False
        cfg.student_obs.use_depth_aug = False
        cfg.domain_randomization.use_obs_delay = False
        cfg.domain_randomization.use_action_delay = False
        cfg.reset.table_reset_xy_range_m = (0.0, 0.0)
        cfg.reset.table_reset_yaw_range_deg = 0.0

    cfg.assets.table_scale_range_x = tuple(args.table_scale_x)
    cfg.assets.table_scale_range_y = tuple(args.table_scale_y)
    cfg.assets.table_scale_num_variants = int(args.table_scale_n)

    # Applied last of all, so an explicit --override beats the training config.
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

    print(f"=> TEACHER eval  scale_x={args.table_scale_x}  scale_y={args.table_scale_y}  "
          f"n_variants={args.table_scale_n}", flush=True)
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
    wrapped = register_rlgames_env(env, rl_device=args.rl_device, clip_obs=clip_obs, clip_actions=clip_actions)

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

    def _run_one_setting(name: str, toggles: dict | None) -> dict:
        if toggles is not None:
            _apply_setting(env, toggles)
        _reseed(args.seed)
        teacher.reset()
        wrapped.reset()
        _reseed(args.seed)
        obs = wrapped.reset()
        n = int(args.num_envs)
        dev = args.rl_device

        # ONE episode per env. A fixed *step* budget over-samples short episodes
        # (success terminates early, so it fits ~3x into the window) and censors
        # whichever episode is still running at the cap -- both biased upward.
        # Freezing each env's outcome at its FIRST termination makes every env
        # exactly one unweighted, uncensored sample.
        finished = torch.zeros(n, dtype=torch.bool, device=dev)
        end_step = torch.full((n,), -1, dtype=torch.long, device=dev)
        flags = {k: torch.zeros(n, dtype=torch.bool, device=dev) for k in _REASONS}
        # Insertion is a strictly weaker criterion than the env's success flag:
        # retract_phase latches when both insertion goals are hit, and only then
        # can retract_succeeded fire. Tracking it separately splits "peg went in"
        # from "peg stayed in after the hand let go".
        inserted = torch.zeros(n, dtype=torch.bool, device=dev)

        used_steps = int(args.max_steps_per_episode)
        # +1: truncation fires when episode_length_buf >= max_episode_length, so
        # a timing-out env is only observed as done on the step AFTER the cap.
        # Looping to exactly max_steps leaves those envs `unfinished` and drops
        # them from the denominator, which silently inflates the success rate.
        for step in range(int(args.max_steps_per_episode) + 1):
            tobs = _extract_teacher_obs(obs)
            act = teacher.get_action(tobs)
            step_out = wrapped.step(act)
            obs = step_out[0] if isinstance(step_out, tuple) else step_out
            # Latch before `finished` is updated: an env that terminates this
            # step has already had retract_phase cleared by the auto-reset, so
            # we rely on having seen it on the preceding steps (and on
            # max_successes below, which implies insertion).
            inserted |= env.retract_phase & ~finished
            # DirectRLEnv.step() resets terminated envs before returning, and
            # _reset_idx zeroes _successes -- so an env that just completed the
            # final goal reads back as 0 and _successes NEVER shows it. The
            # per-episode outcome must come from the termination flags, which
            # are computed in _get_dones (pre-reset) and survive the step.
            done = (env.reset_terminated | env.reset_time_outs)
            newly = done & ~finished
            if bool(newly.any()):
                for k in _REASONS:
                    flags[k] |= env._termination_reasons[k] & newly
                end_step = torch.where(newly, torch.full_like(end_step, step), end_step)
                finished |= newly
            # Envs keep stepping past their first episode; `finished` masks them.
            if bool(finished.all()):
                used_steps = step + 1
                break

        # The env's own success flag. With enable_retract=True and
        # random_goal_fraction=0 this is exactly retract_succeeded: peg inserted,
        # hand withdrawn past retract_distance_threshold, and the peg still
        # within retract_success_tolerance * keypoint_scale of the goal.
        retracted = flags["max_successes"]
        inserted |= retracted  # succeeding implies having inserted

        # Only *early* drops are treated as unstable initial placement. A drop
        # later in the episode is a genuine policy failure and must count.
        # NOTE: this still excludes a few real early failures -- it is a
        # heuristic on timing, not a reset-time stability check.
        drop = flags["fall"] | flags["dropped"]
        early = int(args.early_drop_steps)
        early_drop = drop & (end_step >= 0) & (end_step < early)

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
            "toggles": toggles,
            "num_envs": n,
            "episodes_per_env": 1,
            "steps_used": used_steps,
            "unfinished_envs": int((~finished).sum().item()),
            # Primary: every env that completed its one episode.
            "all_envs": _rates(finished),
            # Excludes drops in the first `early_drop_steps` steps only.
            "early_drop_filtered": _rates(finished & ~early_drop),
            "early_drop_steps": early,
            "termination_counts": {
                k: int((flags[k] & finished).sum().item()) for k in _REASONS
            },
            "n_dropped_any": int((drop & finished).sum().item()),
            "n_dropped_early": int(early_drop.sum().item()),
            "end_step_per_env": end_step.cpu().tolist(),
            "reasons_per_env": {k: flags[k].cpu().tolist() for k in _REASONS},
            "inserted_per_env": inserted.cpu().tolist(),
            "retracted_per_env": retracted.cpu().tolist(),
        }

    # toggles=None keeps the config exactly as built above (no _apply_setting).
    settings = [("train_exact", None)] if args.exact_train else SETTINGS

    print(f"=> teacher loaded from '{args.teacher_checkpoint}'", flush=True)
    print(f"=> {len(settings)} settings, num_envs={args.num_envs}, seed={args.seed}", flush=True)

    results = []
    for name, toggles in settings:
        t0 = time.time()
        try:
            res = _run_one_setting(name, toggles)
        except Exception as exc:
            print(f"[setting] {name}: FAILED -- {exc}", flush=True)
            traceback.print_exc()
            res = {"name": name, "toggles": toggles, "error": str(exc), "all_envs": {}}
        dt = time.time() - t0
        results.append(res)
        nan = float("nan")
        a = res.get("all_envs", {})
        e = res.get("early_drop_filtered", {})
        print(f"[TEACHER] {name:18s} ({dt:.1f}s)", flush=True)
        print(f"           all envs   (n={a.get('n',0):4d}): "
              f"insertion={a.get('insertion_rate', nan):.4f} ({a.get('inserted',0)})  "
              f"retract={a.get('retract_rate', nan):.4f} ({a.get('retracted',0)})", flush=True)
        print(f"           no early drop (n={e.get('n',0):4d}): "
              f"insertion={e.get('insertion_rate', nan):.4f} ({e.get('inserted',0)})  "
              f"retract={e.get('retract_rate', nan):.4f} ({e.get('retracted',0)})", flush=True)
        if res.get("termination_counts"):
            print(f"           terminations: {res['termination_counts']}  "
                  f"dropped_early={res.get('n_dropped_early',0)}/"
                  f"{res.get('n_dropped_any',0)}  "
                  f"unfinished={res.get('unfinished_envs', 0)}  "
                  f"steps={res.get('steps_used', 0)}", flush=True)

    with open(args.output_json, "w") as fp:
        json.dump({
            "teacher_checkpoint": str(args.teacher_checkpoint),
            "num_envs": int(args.num_envs),
            "seed": int(args.seed),
            "max_steps_per_episode": int(args.max_steps_per_episode),
            "table_scale_x": list(args.table_scale_x),
            "table_scale_y": list(args.table_scale_y),
            "table_scale_num_variants": int(args.table_scale_n),
            "exact_train": bool(args.exact_train),
            "train_overrides": train_overrides,
            "cli_overrides": cli_overrides,
            "results": results,
        }, fp, indent=2)
    print(f"=> wrote {args.output_json}", flush=True)

    try:
        env.close()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
