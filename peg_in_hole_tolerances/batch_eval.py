#!/usr/bin/env python3
"""Offline batch evaluation for peg-in-hole tolerance generalization.

Runs each policy on N init states from train and val splits, reports
per-tolerance success rates.

Usage:
    python peg_in_hole_tolerances/batch_eval.py \
        --policies-dir hardware_rollouts/Apr29_tolerance \
        --num-scenes 10 --split both
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "assets" / "urdf" / "peg_in_hole_tolerances"

SET_TO_SCENES_PATH = {
    "train": "assets/urdf/peg_in_hole_tolerances/scenes/scenes.npz",
    "val": "assets/urdf/peg_in_hole_tolerances/scenes_val/scenes_val.npz",
}

TABLE_Z = 0.38
N_ACT = 29
OBS_DIM = 140

BASE_OVERRIDES = {
    "task.env.resetPositionNoiseX": 0.0,
    "task.env.resetPositionNoiseY": 0.0,
    "task.env.resetPositionNoiseZ": 0.0,
    "task.env.randomizeObjectRotation": False,
    "task.env.resetDofPosRandomIntervalFingers": 0.0,
    "task.env.resetDofPosRandomIntervalArm": 0.0,
    "task.env.resetDofVelRandomInterval": 0.0,
    "task.env.tableResetZRange": 0.0,
    "task.env.numEnvs": 1,
    "task.env.envSpacing": 0.4,
    "task.env.capture_video": False,
    "task.env.useActionDelay": False,
    "task.env.useObsDelay": False,
    "task.env.useObjectStateDelayNoise": False,
    "task.env.objectScaleNoiseMultiplierRange": [1.0, 1.0],
    "task.env.resetWhenDropped": False,
    "task.env.armMovingAverage": 0.1,
    "task.env.forceConsecutiveNearGoalSteps": True,
    "task.env.fixedSizeKeypointReward": True,
    "task.env.useFixedInitObjectPose": True,
    "task.env.startArmHigher": True,
    "task.env.forceScale": 0.0,
    "task.env.torqueScale": 0.0,
}


def _create_env(config_path, headless, device, overrides):
    from deployment.rl_player_utils import read_cfg_omegaconf
    from deployment.isaac.isaac_env import merge_cfg_with_default_config, create_env_from_cfg
    from omegaconf import OmegaConf

    cfg = read_cfg_omegaconf(config_path=config_path, device=device)
    cfg = merge_cfg_with_default_config(cfg)
    OmegaConf.set_struct(cfg, False)
    cfg.task.name = "PegInHoleEnv"
    cfg.task_name = "PegInHoleEnv"
    pih_defaults = {
        "goalMode": "preInsertAndFinal",
        "objectName": "peg",
        "useFixedGoalStates": True,
        "useFixedInitObjectPose": True,
        "enableRetract": True,
        "retractRewardScale": 1.0,
        "retractDistanceThreshold": 0.1,
        "retractSuccessBonus": 1000.0,
        "retractSuccessTolerance": 0.005,
        "forceSceneTolCombo": None,
        "forcePegIdx": None,
        "goalXyObsNoise": 0.0,
        "tableForceResetThreshold": 100.0,
    }
    for k, v in pih_defaults.items():
        OmegaConf.update(cfg, f"task.env.{k}", v, force_add=True)
    return create_env_from_cfg(cfg=cfg, headless=headless, overrides=overrides)


def _sim_reset(env, device):
    import torch
    obs_dict = env.reset()
    return obs_dict["obs"]


def load_scenes(split):
    path = ASSETS_DIR / ("scenes" if split == "train" else "scenes_val")
    npz_name = "scenes.npz" if split == "train" else "scenes_val.npz"
    npz_path = path / npz_name
    if not npz_path.exists():
        return None
    data = np.load(str(npz_path))
    return {
        "start_poses": data["start_poses"],
        "goals": data["goals"],
        "traj_lengths": data["traj_lengths"],
        "hole_positions": data["hole_positions"],
        "tolerance_pool_m": data["tolerance_pool_m"],
        "scene_tolerance_indices": data["scene_tolerance_indices"],
    }


def eval_one_episode(conn, config_path, checkpoint_path, scenes_path,
                     scene_idx, tol_slot_idx, peg_idx, goal_mode,
                     extra_overrides):
    try:
        from isaacgym import gymapi  # noqa: F401
        import torch
        from deployment.rl_player import RlPlayer
        import peg_in_hole.objects  # noqa: F401

        device = "cuda" if torch.cuda.is_available() else "cpu"
        overrides = {
            **BASE_OVERRIDES,
            "task.env.scenesPath": scenes_path,
            "task.env.forceSceneTolCombo": [int(scene_idx), int(tol_slot_idx)],
            "task.env.forcePegIdx": int(peg_idx),
            "task.env.goalMode": goal_mode,
            **(extra_overrides or {}),
        }
        env = _create_env(config_path=str(config_path), headless=True,
                          device=device, overrides=overrides)

        env.set_env_state(torch.load(checkpoint_path, map_location=device)[0]["env_state"])
        policy = RlPlayer(OBS_DIM, N_ACT, config_path, checkpoint_path, device, env.num_envs)

        policy.reset()
        obs = _sim_reset(env, device)
        step = 0
        done = False
        retract_ok = False
        max_goals_seen = max(1, int(env.env_max_goals[0].item()) if hasattr(env, "env_max_goals") else env.max_consecutive_successes)
        peak_successes = 0

        while not done:
            action = policy.get_normalized_action(obs, deterministic_actions=True)
            obs_dict, _, done_tensor, _ = env.step(action)
            obs = obs_dict["obs"]
            done = done_tensor[0].item()
            step += 1
            cur_succ = int(env.successes[0].item())
            peak_successes = max(peak_successes, cur_succ)
            if hasattr(env, "extras") and env.extras.get("retract_success_ratio", 0) > 0.5:
                retract_ok = True

        goal_pct = 100.0 * peak_successes / max_goals_seen
        conn.send(("ok", goal_pct, step, retract_ok))

    except Exception:
        conn.send(("error", traceback.format_exc()))

    conn.close()


def run_one(config_path, checkpoint_path, scenes_path,
            scene_idx, tol_slot_idx, peg_idx, goal_mode,
            extra_overrides, timeout=300):
    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()

    proc = ctx.Process(
        target=eval_one_episode,
        args=(child_conn, str(config_path), str(checkpoint_path), scenes_path,
              scene_idx, tol_slot_idx, peg_idx, goal_mode, extra_overrides),
        daemon=True,
    )
    proc.start()
    child_conn.close()

    t0 = time.time()
    result = {"goal_pct": 0.0, "steps": 0, "retract_ok": False, "error": None}

    if parent_conn.poll(timeout):
        msg = parent_conn.recv()
        if msg[0] == "ok":
            result["goal_pct"] = msg[1]
            result["steps"] = msg[2]
            result["retract_ok"] = msg[3]
        elif msg[0] == "error":
            result["error"] = msg[1]
    else:
        result["error"] = f"Timeout after {timeout}s"

    result["wall_time_s"] = round(time.time() - t0, 1)

    proc.join(timeout=10)
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=5)
    parent_conn.close()
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Offline batch eval for peg-in-hole tolerance generalization")
    parser.add_argument("--policies-dir", type=str, required=True)
    parser.add_argument("--split", choices=["train", "val", "both"], default="both")
    parser.add_argument("--num-scenes", type=int, default=10)
    parser.add_argument("--goal-mode", default="preInsertAndFinal")
    parser.add_argument("--peg-idx", type=int, default=0)
    parser.add_argument("--tol-slot", type=int, default=None,
                        help="Single tol slot index (default: all slots)")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--override", nargs=2, action="append", default=[],
                        metavar=("KEY", "VALUE"), help="Extra config overrides")
    args = parser.parse_args()

    extra_overrides = {}
    for key, val in args.override:
        for cast in (int, float):
            try:
                val = cast(val)
                break
            except ValueError:
                continue
        if val == "True":
            val = True
        elif val == "False":
            val = False
        extra_overrides[key] = val

    # Discover policies
    pdir = Path(args.policies_dir)
    if not pdir.is_absolute():
        pdir = REPO_ROOT / pdir
    policies = {}
    for sub in sorted(pdir.iterdir()):
        cfg = sub / "config.yaml"
        ckpt = sub / "model.pth"
        if cfg.exists() and ckpt.exists():
            policies[sub.name] = (cfg, ckpt)
    if not policies:
        print(f"ERROR: No policies found in {pdir}")
        sys.exit(1)

    splits = ["train", "val"] if args.split == "both" else [args.split]

    # Load scenes
    scenes_by_split = {}
    for split in splits:
        sc = load_scenes(split)
        if sc is None:
            print(f"WARNING: No scenes for split={split}")
            continue
        scenes_by_split[split] = sc

    # Build task list: (split, scene_idx, tol_slot_idx)
    tasks = []
    for split in splits:
        if split not in scenes_by_split:
            continue
        sc = scenes_by_split[split]
        N = sc["start_poses"].shape[0]
        K = sc["scene_tolerance_indices"].shape[1]
        n_scenes = min(args.num_scenes, N)
        tol_slots = [args.tol_slot] if args.tol_slot is not None else list(range(K))
        for scene_idx in range(n_scenes):
            for tol_slot in tol_slots:
                tasks.append((split, scene_idx, tol_slot))

    total = len(policies) * len(tasks)
    print(f"\n{'='*70}")
    print(f"Peg-in-Hole Tolerance Batch Evaluation")
    print(f"  Policies: {list(policies.keys())}")
    print(f"  Splits: {splits}")
    print(f"  Tasks: {len(tasks)} (scenes × tol_slots)")
    print(f"  Total evals: {total}")
    print(f"  Goal mode: {args.goal_mode}")
    print(f"{'='*70}\n")

    results = {}
    completed = 0
    t_start = time.time()

    for policy_name, (config_path, checkpoint_path) in policies.items():
        print(f"\n--- Policy: {policy_name} ---")
        results[policy_name] = {}

        for split, scene_idx, tol_slot in tasks:
            completed += 1
            scenes = scenes_by_split[split]
            tol_pool_idx = scenes["scene_tolerance_indices"][scene_idx, tol_slot]
            tol_mm = float(scenes["tolerance_pool_m"][tol_pool_idx]) * 1000
            scenes_path = SET_TO_SCENES_PATH[split]

            print(f"  [{completed}/{total}] {split}/scene{scene_idx}/tol{tol_slot} "
                  f"({tol_mm:.2f}mm) ... ", end="", flush=True)

            r = run_one(
                config_path, checkpoint_path, scenes_path,
                scene_idx, tol_slot, args.peg_idx, args.goal_mode,
                extra_overrides, args.timeout,
            )

            results[policy_name].setdefault(split, []).append({
                "scene_idx": scene_idx,
                "tol_slot": tol_slot,
                "tol_mm": tol_mm,
                **r,
            })

            if r["error"]:
                err_line = r["error"].strip().split("\n")[-1][:60]
                print(f"ERROR: {err_line}")
            else:
                retract_str = "OK" if r["retract_ok"] else "FAIL"
                print(f"{r['goal_pct']:.0f}% goals, retract: {retract_str}")

    total_time = time.time() - t_start

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    for policy_name in policies:
        print(f"\n  Policy: {policy_name}")
        for split in splits:
            entries = results.get(policy_name, {}).get(split, [])
            if not entries:
                continue
            valid = [e for e in entries if e["error"] is None]
            if not valid:
                print(f"    {split}: all errors")
                continue

            success_rate = 100 * np.mean([e["goal_pct"] == 100.0 and e["retract_ok"] for e in valid])
            all_goals_rate = 100 * np.mean([e["goal_pct"] == 100.0 for e in valid])
            retract_rate = 100 * np.mean([e["retract_ok"] for e in valid])
            n_errors = len(entries) - len(valid)

            print(f"    {split}: success={success_rate:.0f}%, 100%_goals={all_goals_rate:.0f}%, "
                  f"retract={retract_rate:.0f}%, n={len(valid)}, errors={n_errors}")

            # Per-tolerance breakdown
            tol_groups: Dict[float, List] = {}
            for e in valid:
                tol_groups.setdefault(e["tol_mm"], []).append(e)
            for tol_mm in sorted(tol_groups.keys()):
                group = tol_groups[tol_mm]
                s = 100 * np.mean([e["goal_pct"] == 100.0 and e["retract_ok"] for e in group])
                g = 100 * np.mean([e["goal_pct"] == 100.0 for e in group])
                r = 100 * np.mean([e["retract_ok"] for e in group])
                print(f"      tol={tol_mm:.2f}mm: success={s:.0f}%, 100%_goals={g:.0f}%, retract={r:.0f}% (n={len(group)})")

    print(f"\nTotal wall time: {total_time/60:.1f} minutes")

    # Save JSON
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = args.output or str(
        REPO_ROOT / "peg_in_hole_tolerances" / "eval_outputs"
        / f"batch_eval_{timestamp}.json"
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "metadata": {
            "timestamp": timestamp,
            "splits": splits,
            "num_scenes": args.num_scenes,
            "goal_mode": args.goal_mode,
            "peg_idx": args.peg_idx,
            "total_wall_time_s": round(total_time, 1),
            "policies": {
                name: {"config": str(cfg), "checkpoint": str(ckpt)}
                for name, (cfg, ckpt) in policies.items()
            },
        },
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, default=str)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
