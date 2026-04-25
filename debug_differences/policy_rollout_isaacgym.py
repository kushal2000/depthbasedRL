"""Pretrained simtoolreal policy rollout — Isaac Gym (legacy) side.

Self-contained driver modeled on ``sine_hand_isaacgym.py``: bootstraps
``isaacgymenvs.tasks.SimToolReal`` via hydra-compose with all DR/reset
randomness off, pins the procedural pool to a single hammer URDF, and
pins the goal to one env-local pose. Loads ``pretrained_policy/model.pth``
via ``RlPlayer`` and rolls it forward for ``--num_steps``. Dumps a per-step
trace (obs / action / joint_pos / joint_vel / joint_targets / object_state /
goal_pose / reward) to npz in canonical (DFS) joint order, plus an mp4
captured from a camera that matches the isaacsim Camera intrinsics so the
two videos are byte-comparable.

    .venv/bin/python debug_differences/policy_rollout_isaacgym.py
"""

from __future__ import annotations

# isort: off
# IMPORTANT: isaacgym must be imported before torch — VecTask init checks
# this and the C++ symbols load incorrectly otherwise.
from isaacgym import gymapi, gymtorch, gymutil  # noqa: F401
# isort: on

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from deployment.rl_player import RlPlayer
from isaacgymenvs.tasks import isaacgym_task_map
from isaacgymenvs.utils.reformat import omegaconf_to_dict


REPO_ROOT = Path(__file__).resolve().parents[1]

# Single fixed goal pose used by both backends. Format: (x, y, z, qx, qy, qz, qw)
# — xyzw quat (matches isaacgym's root_state_tensor convention). Position is
# inside target_volume_mins/maxs ([-0.35,-0.2,0.6]→[0.35,0.2,0.95]) so the
# pose lies in the trained policy's distribution. Identity quat = world-aligned.
# Keep in sync with debug_differences/policy_rollout_isaacsim.py.
FIXED_GOAL_POSE_XYZ_XYZW = (0.0, 0.0, 0.78, 0.0, 0.0, 0.0, 1.0)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_steps", type=int, default=240,
                        help="Policy steps (default 240 = 4s @ 60Hz).")
    parser.add_argument("--config", type=str,
                        default=str(REPO_ROOT / "pretrained_policy/config.yaml"))
    parser.add_argument("--checkpoint", type=str,
                        default=str(REPO_ROOT / "pretrained_policy/model.pth"))
    parser.add_argument("--sim_device", default="cuda:0")
    parser.add_argument("--rl_device", default="cuda:0")
    parser.add_argument("--graphics_device_id", type=int, default=0)
    parser.add_argument("--out", type=str,
                        default=str(REPO_ROOT / "debug_differences/data/isaacgym_policy_rollout.npz"))
    parser.add_argument("--video", type=str,
                        default=str(REPO_ROOT / "debug_differences/plots/isaacgym_policy_rollout.mp4"),
                        help="MP4 output path. Empty string disables video.")
    parser.add_argument("--video_fps", type=int, default=30)
    parser.add_argument("--frame_every", type=int, default=2,
                        help="Capture one frame every N policy steps "
                        "(matches isaacsim eval default).")
    return parser


def _build_cfg() -> dict:
    """Compose isaacgymenvs config with reset/DR overrides + fixed-goal +
    hammer-only. Returns the task-side dict that ``SimToolReal.__init__``
    expects. Mirrors ``sine_hand_isaacgym.py:_build_cfg`` plus the
    fixed-goal/hammer-only overrides used by ``dextoolbench/
    eval_simtoolreal_base.py``."""
    cfg_dir = str((REPO_ROOT / "isaacgymenvs" / "cfg").resolve())
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    fixed = list(FIXED_GOAL_POSE_XYZ_XYZW)
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "task=SimToolReal",
                "task.env.numEnvs=1",
                # All reset randomness off.
                "task.env.resetPositionNoiseX=0.0",
                "task.env.resetPositionNoiseY=0.0",
                "task.env.resetPositionNoiseZ=0.0",
                "task.env.randomizeObjectRotation=False",
                "task.env.resetDofPosRandomIntervalFingers=0.0",
                "task.env.resetDofPosRandomIntervalArm=0.0",
                "task.env.resetDofVelRandomInterval=0.0",
                "task.env.tableResetZRange=0.0",
                # All DR off.
                "task.env.forceScale=0.0",
                "task.env.torqueScale=0.0",
                "task.env.useObsDelay=False",
                "task.env.useActionDelay=False",
                "task.env.useObjectStateDelayNoise=False",
                "task.env.objectScaleNoiseMultiplierRange=[1.0,1.0]",
                # Capture-viewer / interactive rollouts off — they spam
                # logs and confuse the camera-sensor pipeline.
                "task.env.capture_viewer=False",
                "task.env.capture_video=False",
                # Episode long enough that we never hit a time-truncation
                # mid-trace; resetWhenDropped off so the trace doesn't end
                # if the hammer slips.
                "task.env.episodeLength=1200",
                "task.env.resetWhenDropped=False",
                # Single object asset: hammer URDF. numAssetsPerType=1 +
                # randomizeAssetOrder=False keeps pool[0] = first matching
                # ObjectSizeDistribution (the cuboid hammer entry); the sim
                # pair script uses identical knobs so both backends spawn
                # the same shape on env_0.
                "task.env.handleHeadTypes=[hammer]",
                "task.env.numAssetsPerType=1",
                "task.env.randomizeAssetOrder=False",
                # Pin goal to one fixed pose every reset. The legacy env
                # forces max_consecutive_successes = len(fixedGoalStates)
                # (env.py:1280), which would auto-reset the episode after
                # one goal hit — sim's reset path doesn't have that
                # coupling, so to make the two backends comparable we
                # replicate the same pose to keep the env running.
                "task.env.useFixedGoalStates=True",
                f"task.env.fixedGoalStates=[{', '.join([str(fixed)] * 100)}]",
                "task.env.fixedGoalStatesJsonPath=null",
            ],
        )
    return omegaconf_to_dict(cfg.task)


def main() -> None:
    args = _build_parser().parse_args()

    cfg_dict = _build_cfg()
    env = isaacgym_task_map["SimToolReal"](
        cfg=cfg_dict,
        rl_device=args.rl_device,
        sim_device=args.sim_device,
        graphics_device_id=args.graphics_device_id,
        headless=True,
        virtual_screen_capture=False,
        force_render=False,
    )

    print(f"[gym] asset pool size={len(env.object_asset_files)}, "
          f"env 0 picks pool[0] = {Path(env.object_asset_files[0]).name}")

    # Optional camera sensor — same pose + intrinsics as the isaacsim
    # Camera (PinholeCameraCfg focal=24mm, aperture=20.955mm ⇒ ~47.10° FoV)
    # so the two mp4s frame the scene the same way.
    camera_handle = None
    cam_props = None
    frames: list = []
    if args.video:
        cam_props = gymapi.CameraProperties()
        cam_props.width = 640
        cam_props.height = 480
        cam_props.horizontal_fov = math.degrees(
            2.0 * math.atan(20.955 / (2.0 * 24.0))
        )
        env_ptr = env.envs[0]
        camera_handle = env.gym.create_camera_sensor(env_ptr, cam_props)
        env.gym.set_camera_location(
            camera_handle, env_ptr,
            gymapi.Vec3(0.0, -1.0, 1.03),
            gymapi.Vec3(0.0, 0.0, 0.53),
        )
        # Diagnostic: dump the actual camera transform after set_camera_location.
        cam_xform = env.gym.get_camera_transform(env.sim, env_ptr, camera_handle)
        print(f"[diag] camera_pos    world={cam_xform.p.x:.3f}, {cam_xform.p.y:.3f}, {cam_xform.p.z:.3f}")
        print(f"[diag] camera_quat   xyzw={cam_xform.r.x:.3f}, {cam_xform.r.y:.3f}, {cam_xform.r.z:.3f}, {cam_xform.r.w:.3f}")
        print(f"[diag] camera_fov    h={cam_props.horizontal_fov:.2f}°  size={cam_props.width}x{cam_props.height}")

    n_obs = 140
    n_act = env.num_hand_arm_dofs  # 29
    player = RlPlayer(
        num_observations=n_obs,
        num_actions=n_act,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.rl_device,
        num_envs=env.num_envs,
    )
    player.player.init_rnn()

    # First env.step is the env's reset trigger after construction.
    obs_dict, _, _, _ = env.step(torch.zeros((env.num_envs, n_act), device=args.rl_device))
    obs = obs_dict["obs"]

    T = args.num_steps
    joint_names = list(env.joint_names)
    obs_log = np.zeros((T, n_obs), dtype=np.float32)
    action_log = np.zeros((T, n_act), dtype=np.float32)
    joint_pos_log = np.zeros((T, n_act), dtype=np.float32)
    joint_vel_log = np.zeros((T, n_act), dtype=np.float32)
    joint_targets_log = np.zeros((T, n_act), dtype=np.float32)
    object_state_log = np.zeros((T, 13), dtype=np.float32)
    goal_pose_log = np.zeros((T, 7), dtype=np.float32)
    reward_log = np.zeros(T, dtype=np.float32)

    for step in range(T):
        # Capture obs + raw state BEFORE action (matches what the policy sees).
        obs_log[step] = obs[0].detach().cpu().numpy()
        joint_pos_log[step] = env.arm_hand_dof_pos[0].detach().cpu().numpy()
        joint_vel_log[step] = env.arm_hand_dof_vel[0].detach().cpu().numpy()
        object_state_log[step] = env.object_state[0].detach().cpu().numpy()
        goal_pose_log[step] = env.goal_pose[0].detach().cpu().numpy()

        action = player.get_normalized_action(obs, deterministic_actions=True)
        action_log[step] = action[0].detach().cpu().numpy()

        obs_dict, reward, _done, _ = env.step(action)
        obs = obs_dict["obs"]
        reward_log[step] = float(reward[0].detach().cpu().item())

        # prev_targets is what the env pushed to PhysX during this step's
        # decimation ticks (legacy apply_actions sets prev_targets =
        # cur_targets at the end of the step).
        joint_targets_log[step] = env.prev_targets[0, :n_act].detach().cpu().numpy()

        if camera_handle is not None and step % args.frame_every == 0:
            # env.step() (vec_task.py:403) only fetches PhysX results on
            # CPU device; on GPU we must fetch ourselves before letting
            # the graphics layer sync. Without this, step_graphics reads
            # stale actor poses and the camera sensor renders a frame from
            # before reset_idx wrote the actor placements.
            env.gym.fetch_results(env.sim, True)
            env.gym.step_graphics(env.sim)
            env.gym.render_all_camera_sensors(env.sim)
            env.gym.start_access_image_tensors(env.sim)
            rgba = env.gym.get_camera_image(
                env.sim, env.envs[0], camera_handle, gymapi.IMAGE_COLOR
            )
            env.gym.end_access_image_tensors(env.sim)
            rgba = rgba.reshape(cam_props.height, cam_props.width, 4)
            frames.append(rgba[:, :, :3])

        if step % 60 == 0:
            print(f"[isaacgym] step {step:4d}  reward={reward_log[step]:+.3f}  "
                  f"frames={len(frames)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        joint_names=np.array(joint_names),
        obs=obs_log,
        action=action_log,
        joint_pos=joint_pos_log,
        joint_vel=joint_vel_log,
        joint_targets=joint_targets_log,
        object_state=object_state_log,
        goal_pose=goal_pose_log,
        reward=reward_log,
        side="isaacgym",
    )
    print(f"[isaacgym] Wrote {out_path} — {T} steps, {n_act} DOFs")

    if frames:
        import imageio
        video_path = Path(args.video)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(str(video_path), frames, fps=args.video_fps)
        print(f"[isaacgym] Wrote {video_path} — {len(frames)} frames @ {args.video_fps} fps")


if __name__ == "__main__":
    main()
