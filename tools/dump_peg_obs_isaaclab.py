from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from isaacgymenvs.utils.observation_action_utils_sharpa import compute_joint_pos_targets
from isaacgymenvs.utils.torch_jit_utils import quat_rotate
from isaacsim_conversion.distill_env import IsaacSimDistillEnv
from isaacsim_conversion.task_utils import CameraIntrinsics, load_task_spec, load_yaml


N_ACT = 29


def capture_step(env: IsaacSimDistillEnv):
    sim_state = env.compute_sim_state()
    obs = env.build_teacher_obs(sim_state)[0].astype(np.float32).copy()
    body_state = env.robot.data.body_state_w
    palm_state = body_state[:, env.palm_body_idx]
    env_origins = env.env_origins
    palm_pos = palm_state[:, :3] - env_origins
    palm_rot = palm_state[:, 3:7][:, [1, 2, 3, 0]]
    palm_center = palm_pos + quat_rotate(palm_rot, env.palm_offset_t)
    fingertip_state = body_state[:, env.fingertip_body_indices]
    fingertip_pos = fingertip_state[:, :, :3] - env_origins.unsqueeze(1)
    fingertip_quat = fingertip_state[:, :, 3:7][:, :, [1, 2, 3, 0]]
    fingertip_pos_offset = fingertip_pos + quat_rotate(
        fingertip_quat.reshape(-1, 4),
        env.fingertip_offsets_t.reshape(-1, 3),
    ).reshape(env.num_envs, len(env.fingertip_body_indices), 3)
    fingertip_pos_rel_palm = fingertip_pos_offset - palm_center.unsqueeze(1)
    object_pos = env.object_rigid.data.root_pos_w - env_origins
    object_quat = env.object_rigid.data.root_quat_w[:, [1, 2, 3, 0]]
    goal_pose_t = torch.tensor(env.goal_pose, dtype=torch.float32, device=env.device)
    object_scales_t = torch.tensor(env.object_scales_batch, dtype=torch.float32, device=env.device)
    keypoint_offsets = env.object_keypoint_offsets_t.repeat(env.num_envs, 1, 1) * object_scales_t.unsqueeze(1)
    obj_keypoint_pos = object_pos.unsqueeze(1) + quat_rotate(
        object_quat.unsqueeze(1).repeat(1, keypoint_offsets.shape[1], 1).reshape(-1, 4),
        keypoint_offsets.reshape(-1, 3),
    ).reshape(env.num_envs, keypoint_offsets.shape[1], 3)
    goal_keypoint_pos = goal_pose_t[:, :3].unsqueeze(1) + quat_rotate(
        goal_pose_t[:, 3:7].unsqueeze(1).repeat(1, keypoint_offsets.shape[1], 1).reshape(-1, 4),
        keypoint_offsets.reshape(-1, 3),
    ).reshape(env.num_envs, keypoint_offsets.shape[1], 3)
    keypoints_rel_palm = obj_keypoint_pos - palm_center.unsqueeze(1)
    keypoints_rel_goal = obj_keypoint_pos - goal_keypoint_pos
    return {
        "obs": obs,
        "q": sim_state.q[0].astype(np.float32).copy(),
        "qd": sim_state.qd[0].astype(np.float32).copy(),
        "prev_targets": env.prev_targets[0].astype(np.float32).copy(),
        "palm_pos": palm_center[0].detach().cpu().numpy().astype(np.float32).copy(),
        "palm_rot": palm_rot[0].detach().cpu().numpy().astype(np.float32).copy(),
        "object_rot": sim_state.object_pose[0, 3:7].astype(np.float32).copy(),
        "fingertip_pos_rel_palm": fingertip_pos_rel_palm[0].detach().cpu().numpy().astype(np.float32).reshape(-1).copy(),
        "keypoints_rel_palm": keypoints_rel_palm[0].detach().cpu().numpy().astype(np.float32).reshape(-1).copy(),
        "keypoints_rel_goal": keypoints_rel_goal[0].detach().cpu().numpy().astype(np.float32).reshape(-1).copy(),
        "object_scales": env.object_scales_batch[0].astype(np.float32).copy(),
        "object_pose": sim_state.object_pose[0].astype(np.float32).copy(),
        "goal_pose": sim_state.goal_pose[0].astype(np.float32).copy(),
        "goal_idx": np.array([float(sim_state.goal_idx[0])], dtype=np.float32),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-config", required=True)
    parser.add_argument("--distill-config", default="isaacsim_conversion/configs/hammer_distill.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--scene-idx", type=int, default=82)
    parser.add_argument("--peg-idx", type=int, default=5)
    parser.add_argument("--tol-slot-idx", type=int, default=5)
    parser.add_argument("--goal-mode", default="preInsertAndFinal")
    parser.add_argument("--steps", type=int, default=12)
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=True, enable_cameras=False)
    app = launcher.app

    repo_root = Path(__file__).resolve().parent.parent
    distill_cfg = load_yaml(repo_root / args.distill_config)
    task_spec = load_task_spec(
        repo_root=repo_root,
        task_source="peg_in_hole",
        assembly="beam",
        part_id="2",
        collision_method="coacd",
        object_category="hammer",
        object_name="claw_hammer",
        task_name="swing_down",
        teacher_config_path=args.teacher_config,
        peg_scene_idx=args.scene_idx,
        peg_idx=args.peg_idx,
        peg_tol_slot_idx=args.tol_slot_idx,
        peg_goal_mode=args.goal_mode,
        peg_force_identity_start_quat=False,
    )
    env = IsaacSimDistillEnv(
        task_spec=task_spec,
        app=app,
        headless=True,
        camera_modality="depth",
        enable_camera=False,
        num_envs=1,
        env_spacing=0.4,
        object_start_mode="fixed",
        object_pos_noise_xyz=(0.0, 0.0, 0.0),
        object_yaw_noise_deg=0.0,
        camera_backend=distill_cfg.get("camera_backend", "tiled"),
        ground_plane_size=float(distill_cfg.get("ground_plane_size", 500.0)),
        episode_length=max(args.steps + 5, 64),
        reset_when_dropped=False,
        camera_intrinsics=CameraIntrinsics(),
    )
    env.reset()

    per_step: dict[str, list[np.ndarray]] = {}
    zero_actions = np.zeros((1, N_ACT), dtype=np.float32)
    for _step in range(args.steps):
        captured = capture_step(env)
        for key, value in captured.items():
            per_step.setdefault(key, []).append(value)
        sim_state = env.compute_sim_state()
        targets = compute_joint_pos_targets(
            actions=zero_actions,
            prev_targets=env.prev_targets,
            hand_moving_average=0.1,
            arm_moving_average=0.1,
            hand_dof_speed_scale=1.5,
            dt=1.0 / 60.0,
        )
        env.apply_action(targets)
        env.step(render=False)

    arrays = {key: np.stack(values, axis=0) for key, values in per_step.items()}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    meta = {
        "backend": "isaaclab",
        "scene_idx": args.scene_idx,
        "peg_idx": args.peg_idx,
        "tol_slot_idx": args.tol_slot_idx,
        "goal_mode": args.goal_mode,
        "steps": args.steps,
    }
    output.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(output)
    app.close()


if __name__ == "__main__":
    main()
