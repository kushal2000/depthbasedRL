from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from deployment.isaac.isaac_env import create_env_from_cfg, merge_cfg_with_default_config
from deployment.rl_player_utils import read_cfg_omegaconf


N_ACT = 29


def build_env(config_path: str, scene_idx: int, peg_idx: int, tol_slot_idx: int, goal_mode: str, steps: int):
    import peg_in_hole.objects  # noqa: F401

    cfg = read_cfg_omegaconf(config_path=config_path, device="cuda")
    cfg = merge_cfg_with_default_config(cfg)
    OmegaConf.set_struct(cfg, False)
    cfg.task.name = "PegInHoleEnv"
    cfg.task_name = "PegInHoleEnv"
    overrides = {
        "task.env.numEnvs": 1,
        "task.env.envSpacing": 0.4,
        "task.env.capture_video": False,
        "task.env.viserViz": False,
        "task.env.episodeLength": max(steps + 5, 64),
        "task.env.forceSceneTolCombo": [int(scene_idx), int(tol_slot_idx)],
        "task.env.forcePegIdx": int(peg_idx),
        "task.env.goalMode": goal_mode,
        "task.env.useFixedGoalStates": True,
        "task.env.useFixedInitObjectPose": True,
        "task.env.resetPositionNoiseX": 0.0,
        "task.env.resetPositionNoiseY": 0.0,
        "task.env.resetPositionNoiseZ": 0.0,
        "task.env.randomizeObjectRotation": False,
        "task.env.resetDofPosRandomIntervalFingers": 0.0,
        "task.env.resetDofPosRandomIntervalArm": 0.0,
        "task.env.resetDofVelRandomInterval": 0.0,
        "task.env.tableResetZRange": 0.0,
        "task.env.useActionDelay": False,
        "task.env.useObsDelay": False,
        "task.env.useObjectStateDelayNoise": False,
        "task.env.objectScaleNoiseMultiplierRange": [1.0, 1.0],
        "task.env.goalXyObsNoise": 0.0,
        "task.env.startArmHigher": False,
        "task.env.forceScale": 0.0,
        "task.env.torqueScale": 0.0,
        "task.env.linVelImpulseScale": 0.0,
        "task.env.angVelImpulseScale": 0.0,
        "task.env.forceProbRange": [0.0001, 0.0001],
        "task.env.torqueProbRange": [0.0001, 0.0001],
        "task.env.linVelImpulseProbRange": [0.0001, 0.0001],
        "task.env.angVelImpulseProbRange": [0.0001, 0.0001],
        "task.env.resetWhenDropped": False,
    }
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value, force_add=True)
    return create_env_from_cfg(cfg=cfg, headless=True, overrides=None)


def capture_step(env) -> dict[str, np.ndarray]:
    return {
        "obs": env.obs_buf[0].detach().cpu().numpy().astype(np.float32).copy(),
        "q": env.arm_hand_dof_pos[0, :N_ACT].detach().cpu().numpy().astype(np.float32).copy(),
        "qd": env.arm_hand_dof_vel[0, :N_ACT].detach().cpu().numpy().astype(np.float32).copy(),
        "prev_targets": env.prev_targets[0, :N_ACT].detach().cpu().numpy().astype(np.float32).copy(),
        "palm_pos": env.palm_center_pos[0].detach().cpu().numpy().astype(np.float32).copy(),
        "palm_rot": env._palm_state[0, 3:7].detach().cpu().numpy().astype(np.float32).copy(),
        "object_rot": env.object_state[0, 3:7].detach().cpu().numpy().astype(np.float32).copy(),
        "fingertip_pos_rel_palm": env.fingertip_pos_rel_palm[0].detach().cpu().numpy().astype(np.float32).reshape(-1).copy(),
        "keypoints_rel_palm": env.keypoints_rel_palm[0].detach().cpu().numpy().astype(np.float32).reshape(-1).copy(),
        "keypoints_rel_goal": env.keypoints_rel_goal[0].detach().cpu().numpy().astype(np.float32).reshape(-1).copy(),
        "object_scales": (env.object_scales[0] * env.object_scale_noise_multiplier[0]).detach().cpu().numpy().astype(np.float32).copy(),
        "object_pose": env.object_pose[0, :7].detach().cpu().numpy().astype(np.float32).copy(),
        "goal_pose": env.goal_pose[0, :7].detach().cpu().numpy().astype(np.float32).copy(),
        "goal_idx": np.array([float(env.successes[0].item())], dtype=np.float32),
    }


def main():
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scene-idx", type=int, default=82)
    parser.add_argument("--peg-idx", type=int, default=5)
    parser.add_argument("--tol-slot-idx", type=int, default=5)
    parser.add_argument("--goal-mode", default="preInsertAndFinal")
    parser.add_argument("--steps", type=int, default=12)
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    env = build_env(args.config_path, args.scene_idx, args.peg_idx, args.tol_slot_idx, args.goal_mode, args.steps)
    zero_action = torch.zeros((env.num_envs, N_ACT), device=env.device, dtype=torch.float32)

    obs_dict = env.reset()
    _ = obs_dict
    obs_dict, _, _, _ = env.step(zero_action)
    _ = obs_dict

    per_step: dict[str, list[np.ndarray]] = {}
    for _step in range(args.steps):
        captured = capture_step(env)
        for key, value in captured.items():
            per_step.setdefault(key, []).append(value)
        obs_dict, _, _, _ = env.step(zero_action)
        _ = obs_dict

    arrays = {key: np.stack(values, axis=0) for key, values in per_step.items()}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    meta = {
        "backend": "isaacgym",
        "scene_idx": args.scene_idx,
        "peg_idx": args.peg_idx,
        "tol_slot_idx": args.tol_slot_idx,
        "goal_mode": args.goal_mode,
        "steps": args.steps,
    }
    output.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(output)


if __name__ == "__main__":
    main()
