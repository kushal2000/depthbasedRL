"""Run a pretrained SimToolReal/peg-in-hole policy in Isaac Sim.

Bridges the isaacsimenvs task registry to the working Isaac Sim scene setup in
``isaacsim_conversion.isaacsim_env``. The rollout loop mirrors
``isaacsim_conversion/rollout.py`` (currently the canonical path that has
completed a full assembly task with the pretrained policy).

Usage:
    python isaacsimenvs/tasks/simtoolreal/play_simtoolreal.py \
        --task Isaacsimenvs-SimToolReal-Direct-v0 \
        --checkpoint pretrained_policy/model.pth \
        --config pretrained_policy/config.yaml \
        --assembly beam --part_id 2 --collision_method coacd \
        --max_steps 1200 \
        --enable_cameras

Notes:
- Single-env direct-sim loop; does NOT use ``gym.make`` because the pretrained
  policy predates our DirectRLEnv port, and this inference path is validated.
- Observation dim = 140, action dim = 29 (7-DOF IIWA + 22-DOF Sharpa).
- ``--enable_cameras`` is required if you want an mp4 output; without it the
  script completes but skips frame capture.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="Isaacsimenvs-SimToolReal-Direct-v0", help="Gym task id")
    parser.add_argument(
        "--checkpoint",
        default="pretrained_policy/model.pth",
        help="rl_games .pth checkpoint (path relative to repo root or absolute).",
    )
    parser.add_argument(
        "--config",
        default="pretrained_policy/config.yaml",
        help="Policy config YAML (path relative to repo root or absolute).",
    )
    parser.add_argument("--assembly", default="beam")
    parser.add_argument("--part_id", default="2")
    parser.add_argument("--collision_method", default="coacd")
    parser.add_argument(
        "--task_source",
        choices=["fabrica", "dextoolbench"],
        default="fabrica",
        help="Which asset/trajectory layout to use.",
    )
    parser.add_argument("--object_category", default="hammer")
    parser.add_argument("--object_name", default="claw_hammer")
    parser.add_argument("--object_task_name", default="swing_down", help="DexToolBench task name")
    parser.add_argument("--max_steps", type=int, default=6000)
    parser.add_argument("--video_dir", default="rollout_videos")
    parser.add_argument("--video_fps", type=int, default=30)
    parser.add_argument("--rl_device", default="cuda", help="Policy/rl_games device; avoid clashing with AppLauncher's --device.")
    return parser


def _launch_app():
    """AppLauncher must run BEFORE any isaaclab / omni imports."""
    from isaaclab.app import AppLauncher

    parser = _build_parser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app
    return app, args


# Must happen at module scope, before the other imports below.
_app, _args = _launch_app()


def _resolve_paths(args: argparse.Namespace):
    """Build the (robot, table, object, trajectory, object_info) tuple matching
    ``isaacsim_conversion/rollout.py``."""
    from dextoolbench.objects import NAME_TO_OBJECT

    robot_urdf = str(REPO_ROOT / "assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf")

    if args.task_source == "fabrica":
        import fabrica.objects  # noqa: F401  registers fabrica parts in NAME_TO_OBJECT
        table_urdf = str(
            REPO_ROOT
            / f"assets/urdf/fabrica/{args.assembly}/environments/{args.part_id}/scene_{args.collision_method}.urdf"
        )
        traj_path = REPO_ROOT / f"assets/urdf/fabrica/{args.assembly}/trajectories/{args.part_id}/pick_place.json"
        object_name = f"{args.assembly}_{args.part_id}_{args.collision_method}"
    else:
        table_urdf = str(
            REPO_ROOT
            / f"assets/urdf/dextoolbench/environments/{args.object_category}/{args.object_name}/{args.object_task_name}.urdf"
        )
        traj_path = REPO_ROOT / f"dextoolbench/trajectories/{args.object_category}/{args.object_name}/{args.object_task_name}.json"
        object_name = args.object_name

    obj_info = NAME_TO_OBJECT.get(object_name)
    if obj_info is None:
        raise KeyError(f"Object '{object_name}' not found in NAME_TO_OBJECT")

    return robot_urdf, table_urdf, obj_info.urdf_path, traj_path, obj_info


def _resolve_policy_paths(args: argparse.Namespace) -> tuple[str, str]:
    ckpt = args.checkpoint
    cfg = args.config
    if not Path(ckpt).is_absolute():
        ckpt = str(REPO_ROOT / ckpt)
    if not Path(cfg).is_absolute():
        cfg = str(REPO_ROOT / cfg)
    return ckpt, cfg


def main() -> None:
    args = _args

    # Imports deferred until after AppLauncher boot (CLAUDE.md rule).
    import isaaclab.sim as sim_utils  # noqa: F401

    from deployment.rl_player import RlPlayer
    from isaacgymenvs.utils.observation_action_utils_sharpa import (
        _compute_keypoint_positions,
        compute_joint_pos_targets,
        compute_observation,
        create_urdf_object,
    )
    from isaacsim_conversion.isaacsim_env import IsaacSimEnv, _log

    # Make sure isaacsimenvs side effects (gym.register) have fired so the
    # --task id is resolvable by load_cfg_from_registry users, even though this
    # script doesn't use gym.make directly.
    import isaacsimenvs  # noqa: F401
    import gymnasium as gym
    try:
        gym.spec(args.task)
    except gym.error.NameNotFound:
        _log(
            f"[warn] --task '{args.task}' not in gym registry; continuing with raw rollout."
            "  Is isaacsimenvs/tasks/simtoolreal/__init__.py importable?"
        )

    robot_urdf, table_urdf, object_urdf, traj_path, obj_info = _resolve_paths(args)
    _log(f"Robot URDF:  {robot_urdf}")
    _log(f"Table URDF:  {table_urdf}")
    _log(f"Object URDF: {object_urdf}")
    _log(f"Trajectory:  {traj_path}")

    with open(traj_path) as f:
        traj = json.load(f)
    goals = traj["goals"]
    start_pose = traj["start_pose"]
    _log(f"Trajectory: {len(goals)} goals, start_pose={start_pose}")

    env = IsaacSimEnv(
        robot_urdf=robot_urdf,
        table_urdf=table_urdf,
        object_urdf=object_urdf,
        headless=True,
        app=_app,
    )

    start_pos = np.array(start_pose[:3], dtype=np.float32)
    start_quat_xyzw = np.array(start_pose[3:7], dtype=np.float32)

    ckpt_path, cfg_path = _resolve_policy_paths(args)
    _log(f"Loading policy: checkpoint={ckpt_path}, config={cfg_path}")
    player = RlPlayer(
        num_observations=140,
        num_actions=29,
        config_path=cfg_path,
        checkpoint_path=ckpt_path,
        device=args.rl_device,
        num_envs=1,
    )

    urdf = create_urdf_object("iiwa14_left_sharpa_adjusted_restricted")

    obs_list = [
        "joint_pos", "joint_vel", "prev_action_targets", "palm_pos",
        "palm_rot", "object_rot", "fingertip_pos_rel_palm",
        "keypoints_rel_palm", "keypoints_rel_goal", "object_scales",
    ]
    import yaml
    with open(cfg_path) as f:
        policy_cfg = yaml.safe_load(f)
    if args.task_source == "fabrica":
        fixed_size = policy_cfg.get("task", {}).get("env", {}).get("fixedSize", [0.141, 0.03025, 0.0271])
        object_scales = np.array([fixed_size], dtype=np.float32)
        _log(f"Object scales (fixedSize): {object_scales[0]}")
    else:
        object_scales = np.array([obj_info.scale], dtype=np.float32)
        _log(f"Object scales (NAME_TO_OBJECT.scale): {object_scales[0]}")
    hand_moving_average = 0.1
    arm_moving_average = 0.1
    dof_speed_scale = 1.5
    dt = 1 / 60
    success_steps = 10
    keypoint_tolerance = 0.01 * 1.5  # targetSuccessTolerance * keypointScale

    video_dir = Path(args.video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / f"rollout_{args.assembly}_{args.part_id}.mp4"
    frames: list = []
    trajectory_log: list = []

    has_camera = False
    if getattr(args, "enable_cameras", False):
        try:
            from isaaclab.sensors import Camera, CameraCfg
            import isaaclab.sim as sim_utils_local

            camera_cfg = CameraCfg(
                prim_path="/World/RecordCamera",
                update_period=0,
                height=480,
                width=640,
                data_types=["rgb"],
                spawn=sim_utils_local.PinholeCameraCfg(
                    focal_length=24.0,
                    focus_distance=400.0,
                    horizontal_aperture=20.955,
                    clipping_range=(0.1, 100.0),
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=(0.0, -1.0, 1.03),
                    rot=(0.8507, 0.5257, 0.0, 0.0),
                    convention="opengl",
                ),
            )
            camera = Camera(cfg=camera_cfg)
            env.sim.reset()
            has_camera = True
            _log(f"Camera created for video recording to {video_dir}")
        except Exception as e:
            _log(f"Camera setup failed: {e}")
    else:
        _log("Video recording disabled (pass --enable_cameras to enable)")

    env.set_object_pose(start_pos, start_quat_xyzw)
    env.reset_robot_to_default_pose(render=False)
    q_init, _ = env.get_robot_state()
    _log(f"Robot default pose set. Arm joints: {q_init[:7]}")
    prev_targets = q_init[None].copy()

    current_goal_idx = 0
    goal_pose = np.array(goals[current_goal_idx], dtype=np.float32)[None]
    near_goal_steps = 0

    player.player.init_rnn()

    _log(f"\n=== Starting rollout: {args.max_steps} steps, {len(goals)} goals ===\n")

    for step_i in range(args.max_steps):
        q, qd = env.get_robot_state()
        object_pose = env.get_object_pose_xyzw()[None]

        obs = compute_observation(
            q=q[None], qd=qd[None], prev_action_targets=prev_targets,
            object_pose=object_pose, goal_object_pose=goal_pose,
            object_scales=object_scales, urdf=urdf, obs_list=obs_list,
        )

        obs_tensor = torch.from_numpy(obs).float().to(args.rl_device)
        action = player.get_normalized_action(obs_tensor, deterministic_actions=True)

        targets = compute_joint_pos_targets(
            actions=action.cpu().numpy(),
            prev_targets=prev_targets,
            hand_moving_average=hand_moving_average,
            arm_moving_average=arm_moving_average,
            hand_dof_speed_scale=dof_speed_scale,
            dt=dt,
        )
        prev_targets = targets

        env.set_joint_position_targets(targets[0])
        env.step(render=True)

        if has_camera and step_i % 2 == 0:
            camera.update(dt)
            rgb = camera.data.output["rgb"]
            if rgb is not None and rgb.shape[0] > 0:
                frames.append(rgb[0].cpu().numpy()[:, :, :3])

        object_kps = _compute_keypoint_positions(object_pose, object_scales)
        goal_kps = _compute_keypoint_positions(goal_pose, object_scales)
        keypoints_max_dist = np.max(np.linalg.norm(object_kps[0] - goal_kps[0], axis=-1))

        if keypoints_max_dist < keypoint_tolerance:
            near_goal_steps += 1
        else:
            near_goal_steps = 0

        if near_goal_steps >= success_steps:
            _log(f"[step {step_i}] Goal {current_goal_idx} REACHED! (dist={keypoints_max_dist:.4f})")
            current_goal_idx += 1
            near_goal_steps = 0
            if current_goal_idx >= len(goals):
                _log(f"\n=== ALL {len(goals)} GOALS REACHED at step {step_i}! ===")
                break
            goal_pose = np.array(goals[current_goal_idx], dtype=np.float32)[None]
            _log(f"  -> Advancing to goal {current_goal_idx}/{len(goals)}")

        trajectory_log.append(
            {
                "step": step_i,
                "q": q.copy(),
                "object_pose": object_pose[0].copy(),
                "kp_dist": keypoints_max_dist,
                "goal_idx": current_goal_idx,
            }
        )

        if step_i % 60 == 0:
            obj_z = object_pose[0, 2]
            _log(
                f"[step {step_i:5d}] goal={current_goal_idx}/{len(goals)}, "
                f"kp_dist={keypoints_max_dist:.4f}, obj_z={obj_z:.3f}, "
                f"near_goal={near_goal_steps}/{success_steps}"
            )

    _log(f"\nRollout complete. Final goal: {current_goal_idx}/{len(goals)}")

    if trajectory_log:
        traj_file = video_dir / "trajectory.npz"
        np.savez(
            str(traj_file),
            steps=np.array([t["step"] for t in trajectory_log]),
            q=np.array([t["q"] for t in trajectory_log]),
            object_poses=np.array([t["object_pose"] for t in trajectory_log]),
            kp_dists=np.array([t["kp_dist"] for t in trajectory_log]),
            goal_idxs=np.array([t["goal_idx"] for t in trajectory_log]),
        )
        _log(f"Trajectory saved: {traj_file} ({len(trajectory_log)} steps)")

    if frames:
        import imageio

        _log(f"Saving {len(frames)} frames to {video_path}")
        imageio.mimwrite(str(video_path), frames, fps=args.video_fps)
        _log(f"Video saved: {video_path}")
    else:
        _log("WARNING: No frames captured for video")

    env.close()


if __name__ == "__main__":
    main()
