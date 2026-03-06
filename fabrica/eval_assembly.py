"""Sequential assembly evaluation.

For an assembly with order [p0, p1, ..., pN-1], runs N sub-problems.
Each sub-problem n uses:
  - A table URDF with parts p0..pn-1 baked in as static geometry
  - A pick_place trajectory for part pn
  - The canonical mesh URDF for part pn

Usage:
    python fabrica/eval_assembly.py --assembly beam \
        --config-path pretrained_policy/config.yaml \
        --checkpoint-path pretrained_policy/model.pth
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Workaround: PyTorch 2.4 hipify_python.py generates regexes that exceed
# Python 3.8's default sre_parse recursion limit.
sys.setrecursionlimit(max(sys.getrecursionlimit(), 10000))

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
TABLE_Z = 0.38
Z_OFFSET = 0.03
N_ACT = 29
OBS_DIM = 140
CONTROL_DT = 1.0 / 60.0

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
    "task.env.evalSuccessTolerance": 0.01,
    "task.env.successSteps": 1,
    "task.env.fixedSizeKeypointReward": True,
    "task.env.useFixedInitObjectPose": True,
    "task.env.startArmHigher": True,
    "task.env.forceScale": 0.0,
    "task.env.torqueScale": 0.0,
    "task.env.linVelImpulseScale": 0.0,
    "task.env.angVelImpulseScale": 0.0,
    "task.env.forceOnlyWhenLifted": True,
    "task.env.torqueOnlyWhenLifted": True,
    "task.env.linVelImpulseOnlyWhenLifted": True,
    "task.env.angVelImpulseOnlyWhenLifted": True,
    "task.env.forceProbRange": [0.0001, 0.0001],
    "task.env.torqueProbRange": [0.0001, 0.0001],
    "task.env.linVelImpulseProbRange": [0.0001, 0.0001],
    "task.env.angVelImpulseProbRange": [0.0001, 0.0001],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Sequential assembly evaluation")
    parser.add_argument("--assembly", type=str, required=True,
                        choices=["beam", "car", "cooling_manifold", "duct",
                                 "gamepad", "plumbers_block", "stool_circular"])
    parser.add_argument("--config-path", type=str, default="pretrained_policy/config.yaml")
    parser.add_argument("--checkpoint-path", type=str, default="pretrained_policy/model.pth")
    parser.add_argument("--no-headless", action="store_true", help="Show Isaac Gym viewer GUI")
    return parser.parse_args()


def resolve_path(path_str):
    path = Path(path_str)
    if path.exists():
        return path
    path = REPO_ROOT / path_str
    if path.exists():
        return path
    raise FileNotFoundError(path_str)


def load_trajectory(path):
    with open(path) as f:
        traj = json.load(f)
    traj["start_pose"][2] += Z_OFFSET
    return traj


def make_overrides(object_name, table_urdf, traj):
    return {
        **BASE_OVERRIDES,
        "task.env.objectName": object_name,
        "task.env.useFixedGoalStates": True,
        "task.env.fixedGoalStates": traj["goals"],
        "task.env.asset.table": str(table_urdf),
        "task.env.tableResetZ": TABLE_Z,
        "task.env.objectStartPose": traj["start_pose"],
    }


def load_assembly_order(assets_dir, assembly):
    """Load assembly order from JSON."""
    order_path = assets_dir / assembly / "assembly_order.json"
    if not order_path.exists():
        raise FileNotFoundError(f"No assembly_order.json for {assembly}")
    data = json.loads(order_path.read_text())
    return data["steps"]


def run_subproblem(assembly, part_id, config_path, checkpoint_path,
                   headless, torch, device):
    """Run a single sub-problem and return (success_pct, steps)."""
    from deployment.isaac.isaac_env import create_env
    from deployment.rl_player import RlPlayer
    import fabrica.objects  # noqa: F401

    object_name = f"{assembly}_{part_id}"
    table_urdf = f"urdf/fabrica/environments/{object_name}/pick_place.urdf"
    traj_path = REPO_ROOT / "fabrica" / "trajectories" / object_name / "pick_place.json"

    if not traj_path.exists():
        print(f"  ERROR: Trajectory not found: {traj_path}")
        return 0.0, 0

    traj = load_trajectory(traj_path)
    env = create_env(
        config_path=str(config_path),
        headless=headless,
        device=device,
        overrides=make_overrides(object_name, table_urdf, traj),
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    env.set_env_state(checkpoint[0]["env_state"])
    policy = RlPlayer(OBS_DIM, N_ACT, str(config_path), str(checkpoint_path),
                       device, env.num_envs)

    # Reset
    obs_dict, _, _, _ = env.step(torch.zeros((env.num_envs, N_ACT), device=device))
    obs = obs_dict["obs"]
    policy.reset()

    max_succ = int(env.max_consecutive_successes)
    step = 0
    done = False

    while not done:
        action = policy.get_normalized_action(obs, deterministic_actions=True)
        obs_dict, _, done_tensor, _ = env.step(action)
        obs = obs_dict["obs"]
        done = bool(done_tensor[0].item())
        step += 1

    successes = int(env.successes[0].item())
    success_pct = 100.0 * successes / max_succ if max_succ > 0 else 0.0

    # Cleanup
    del policy
    del env

    return success_pct, step


def main():
    from isaacgym import gymapi  # noqa: F401 isort:skip
    import torch

    args = parse_args()
    config_path = resolve_path(args.config_path)
    checkpoint_path = resolve_path(args.checkpoint_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    assets_dir = REPO_ROOT / "assets" / "urdf" / "fabrica"
    assembly_order = load_assembly_order(assets_dir, args.assembly)

    print(f"\n{'='*60}")
    print(f"Assembly: {args.assembly}")
    print(f"Order: {assembly_order}")
    print(f"{'='*60}\n")

    results = []
    for step_idx, pid in enumerate(assembly_order):
        name = f"{args.assembly}_{pid}"
        print(f"[Step {step_idx+1}/{len(assembly_order)}] Placing part {pid} ({name})")

        success_pct, steps = run_subproblem(
            args.assembly, pid, config_path, checkpoint_path,
            headless=not args.no_headless,
            torch=torch, device=device,
        )

        status = "PASS" if success_pct >= 100.0 else "FAIL"
        results.append({
            "part_id": pid,
            "success_pct": success_pct,
            "steps": steps,
            "status": status,
        })
        print(f"  -> {status}: {success_pct:.0f}% goals in {steps/60:.1f}s\n")

    # Summary
    print(f"\n{'='*60}")
    print(f"Results for {args.assembly}")
    print(f"{'='*60}")
    n_pass = sum(1 for r in results if r["status"] == "PASS")
    for r in results:
        print(f"  Part {r['part_id']}: {r['status']} ({r['success_pct']:.0f}%, {r['steps']/60:.1f}s)")
    print(f"\nOverall: {n_pass}/{len(results)} sub-problems passed")

    # Full assembly success = all sub-problems passed
    if n_pass == len(results):
        print("Assembly: COMPLETE")
    else:
        first_fail = next(r for r in results if r["status"] != "PASS")
        print(f"Assembly: FAILED at part {first_fail['part_id']}")


if __name__ == "__main__":
    main()
