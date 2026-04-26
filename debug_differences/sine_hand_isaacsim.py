"""Sine-wave PD-target rollout on the SHARPA hand — Isaac Sim side.

Bootstraps the isaacsimenvs SimToolReal env, disables every reset/DR
randomness, and feeds an open-loop sine-wave target on the 22 hand DOFs
(arm targets pinned to default reset pose). The action pipeline is
bypassed via ``inner._replay_target_lab_order`` (see
``isaacsimenvs/tasks/simtoolreal/utils/action_utils.py:42-46``) so the
target written here lands verbatim in ``_cur_targets`` and goes straight
to ``robot.set_joint_position_target``.

Output: ``debug_differences/data/isaacsim_sine_hand.npz`` with arrays in
the env's *Lab* joint order. The pair script ``sine_hand_isaacgym.py``
dumps the legacy env's *canonical* order; the diff script aligns by
joint name.

    .venv_isaacsim/bin/python debug_differences/sine_hand_isaacsim.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_steps", type=int, default=120,
                        help="Policy steps (default 120 = 2s @ 60Hz).")
    parser.add_argument("--policy_dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--sine_freq_hz", type=float, default=1.0)
    parser.add_argument("--amplitude_frac", type=float, default=0.02,
                        help="Hand sine amplitude as fraction of (upper - lower) per joint. "
                        "Centered around the default reset pose; lowered to 0.02 because "
                        "5%% still drove fingers through the table at the folded-arm pose.")
    parser.add_argument("--arm_amplitude_frac", type=float, default=0.02,
                        help="Arm sine amplitude as fraction of (upper - lower) per joint. "
                        "Only applied when --joints=all.")
    parser.add_argument("--joints", choices=["hand", "all"], default="hand",
                        help="Which DOFs to drive with the sine sweep. "
                        "'hand' = 22 hand only (arm pinned at default). "
                        "'all' = hand + 7 arm (arm sweeps around desired_kuka_pos).")
    parser.add_argument("--num_assets_per_type", type=int, default=1,
                        help="Keep procedural object pool tiny for fast launch.")
    parser.add_argument("--out", type=str,
                        default=str(REPO_ROOT / "debug_differences/data/isaacsim_sine_hand.npz"))
    parser.add_argument("--video", type=str,
                        default=str(REPO_ROOT / "debug_differences/plots/isaacsim_sine_hand.mp4"),
                        help="MP4 output path. Empty string disables video.")
    parser.add_argument("--video_fps", type=int, default=30)
    return parser


def _launch_app():
    from isaaclab.app import AppLauncher
    parser = _build_parser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    # Camera sensor needs Kit launched with --enable_cameras (RTX renderer).
    args.enable_cameras = bool(args.video)
    app = AppLauncher(args).app
    return app, args


_app, _args = _launch_app()


def main() -> None:
    args = _args

    import math
    import gymnasium as gym
    import numpy as np
    import torch

    import isaacsimenvs  # noqa: F401  registers gym envs
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import SimToolRealEnvCfg

    cfg = SimToolRealEnvCfg()
    cfg.scene.num_envs = 1
    cfg.assets.num_assets_per_type = args.num_assets_per_type

    # All randomness off so the response is fully reproducible.
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
    rs.randomize_object_rotation = False
    rs.reset_dof_pos_random_interval_arm = 0.0
    rs.reset_dof_pos_random_interval_fingers = 0.0
    rs.reset_dof_vel_random_interval = 0.0
    rs.table_reset_z_range = 0.0

    env = gym.make("Isaacsimenvs-SimToolReal-Direct-v0", cfg=cfg)
    inner = env.unwrapped
    inner._replay_target_lab_order = None  # set per-step below

    # Optional camera sensor for video matching the legacy IsaacGym camera pose:
    # pos=(0,-1,1.03), target=(0,0,0.53).
    camera = None
    frames: list = []
    if args.video:
        import isaaclab.sim as sim_utils
        from isaaclab.sensors import Camera, CameraCfg
        camera_cfg = CameraCfg(
            prim_path="/World/RecordCamera",
            update_period=0,
            height=480, width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0,
                horizontal_aperture=20.955, clipping_range=(0.1, 100.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.0, -1.0, 1.03),
                rot=(0.8507, 0.5257, 0.0, 0.0),
                convention="opengl",
            ),
        )
        camera = Camera(cfg=camera_cfg)
        inner.sim.reset()  # finalize camera registration

    action_dim = cfg.action_space
    obs, _ = env.reset()
    # Don't take a warmup zero-action step: the action pipeline maps a
    # zero hand action to the joint mid-range (absolute scale), which
    # would yank the fingers off default before the sine even starts.
    # The first sine tick (sin(0) = 0) holds at default by construction.

    # --- Default pose + per-joint sine amplitudes (Lab order) ---
    default_pos_lab = inner.robot.data.default_joint_pos[0].clone()  # (29,)
    upper_lab = inner.robot.data.joint_pos_limits[0, :, 1].clone()
    lower_lab = inner.robot.data.joint_pos_limits[0, :, 0].clone()

    hand_ids = torch.as_tensor(inner._hand_joint_ids, device=inner.device, dtype=torch.long)
    sine_amplitude_lab = torch.zeros_like(default_pos_lab)
    sine_amplitude_lab[hand_ids] = (
        args.amplitude_frac * (upper_lab[hand_ids] - lower_lab[hand_ids])
    )
    if args.joints == "all":
        arm_ids = torch.as_tensor(inner._arm_joint_ids, device=inner.device, dtype=torch.long)
        sine_amplitude_lab[arm_ids] = (
            args.arm_amplitude_frac * (upper_lab[arm_ids] - lower_lab[arm_ids])
        )

    # --- Logging buffers ---
    T = args.num_steps
    joint_names_lab = list(inner.robot.data.joint_names)
    joint_pos_log = np.zeros((T, action_dim), dtype=np.float32)
    joint_vel_log = np.zeros((T, action_dim), dtype=np.float32)
    target_log = np.zeros((T, action_dim), dtype=np.float32)
    time_log = np.zeros(T, dtype=np.float32)

    zeros_action = torch.zeros((1, action_dim), device=inner.device)

    for step in range(T):
        t = step * args.policy_dt
        time_log[step] = t

        target_lab = default_pos_lab + sine_amplitude_lab * math.sin(
            2.0 * math.pi * args.sine_freq_hz * t
        )
        target_lab = torch.clamp(target_lab, lower_lab, upper_lab)
        inner._replay_target_lab_order = target_lab.unsqueeze(0).contiguous()

        env.step(zeros_action)

        joint_pos_log[step] = inner.robot.data.joint_pos[0].detach().cpu().numpy()
        joint_vel_log[step] = inner.robot.data.joint_vel[0].detach().cpu().numpy()
        target_log[step] = target_lab.detach().cpu().numpy()

        if camera is not None:
            camera.update(dt=inner.physics_dt)
            rgb = camera.data.output["rgb"][0].detach().cpu().numpy()
            frames.append(rgb[:, :, :3].astype(np.uint8))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        joint_names=np.array(joint_names_lab),
        joint_pos=joint_pos_log,
        joint_vel=joint_vel_log,
        target=target_log,
        time=time_log,
        policy_dt=args.policy_dt,
        sine_freq_hz=args.sine_freq_hz,
        amplitude_frac=args.amplitude_frac,
        side="isaacsim",
    )
    print(f"[isaacsim] Wrote {out_path} — {T} steps, {action_dim} DOFs")

    if frames:
        import imageio
        video_path = Path(args.video)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(str(video_path), frames, fps=args.video_fps)
        print(f"[isaacsim] Wrote {video_path} — {len(frames)} frames @ {args.video_fps} fps")

    env.close()


if __name__ == "__main__":
    import os
    import sys
    main()
    # Kit shutdown hangs; force-exit (matches train.py).
    del _app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
