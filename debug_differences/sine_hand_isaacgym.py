"""Sine-wave PD-target rollout on the SHARPA hand — Isaac Gym (legacy) side.

Bootstraps ``isaacgymenvs.tasks.SimToolReal`` directly (no rl_games stack),
disables every reset/DR randomness, and feeds an open-loop sine-wave
target on the 22 hand DOFs. Action pipeline is bypassed by skipping
``env.step()`` entirely — we write to ``env.cur_targets`` and step PhysX
ourselves via the gym API.

Output: ``debug_differences/data/isaacgym_sine_hand.npz`` with arrays in
the env's *canonical* joint order (DFS-per-finger; the legacy URDF +
isaacgym alphabetical sort yield this order natively).

    .venv/bin/python debug_differences/sine_hand_isaacgym.py
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

from isaacgymenvs.tasks import isaacgym_task_map
from isaacgymenvs.utils.reformat import omegaconf_to_dict


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
    parser.add_argument("--sim_device", default="cuda:0")
    parser.add_argument("--rl_device", default="cuda:0")
    parser.add_argument("--graphics_device_id", type=int, default=0)
    parser.add_argument("--out", type=str,
                        default=str(REPO_ROOT / "debug_differences/data/isaacgym_sine_hand.npz"))
    parser.add_argument("--video", type=str,
                        default=str(REPO_ROOT / "debug_differences/plots/isaacgym_sine_hand.mp4"),
                        help="MP4 output path. Empty string disables video.")
    parser.add_argument("--video_fps", type=int, default=30)
    return parser


def _build_cfg() -> dict:
    """Compose isaacgymenvs config with reset/DR overrides for deterministic
    open-loop replay. Returns the task-side dict that ``SimToolReal.__init__``
    expects."""
    cfg_dir = str((REPO_ROOT / "isaacgymenvs" / "cfg").resolve())
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
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

    # Reset to default pose. Two gotchas vs the standard training loop:
    #   1. `VecTask.reset()` is a no-op (vec_task.py:444-456) — the actual
    #      reset_idx fires inside `step()` based on `reset_buf`. We bypass
    #      step(), so we must call reset_idx directly.
    #   2. `reset_idx` only QUEUES DOF/root state writes
    #      (env.py:3793-3796 → `deferred_set_*`). The queue is normally
    #      flushed inside `pre_physics_step` (env.py:3994); we have to
    #      flush it manually here.
    all_envs = torch.arange(env.num_envs, device=env.device)
    env.reset_idx(all_envs)
    env.set_dof_state_tensor_indexed()
    env.set_actor_root_state_tensor_indexed()
    env.gym.set_dof_position_target_tensor(
        env.sim, gymtorch.unwrap_tensor(env.cur_targets)
    )
    env.gym.refresh_dof_state_tensor(env.sim)

    # Optional camera sensor for video — same pose as eval_simtoolreal.py /
    # legacy env.py:397 viewer (cam_pos=(0,-1,1.03), cam_target=(0,0,0.53)).
    camera_handle = None
    frames: list = []
    if args.video:
        cam_props = gymapi.CameraProperties()
        cam_props.width = 640
        cam_props.height = 480
        # Match the isaacsim Camera intrinsics so frames are byte-comparable:
        #   PinholeCameraCfg(focal_length=24mm, horizontal_aperture=20.955mm)
        #   ⇒ horizontal_fov = 2 * atan(aperture / (2 * focal)) ≈ 47.10°.
        # gymapi defaults to ~90° which makes the gym view look "zoomed out"
        # vs sim. Keeping aspect ratio (width/height) in sync with sim's
        # 640×480.
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

    num_dofs = env.num_hand_arm_dofs  # 29
    default_pos = env.hand_arm_default_dof_pos.to(env.device).clone()  # (29,)
    upper = env.arm_hand_dof_upper_limits[:num_dofs].to(env.device).clone()
    lower = env.arm_hand_dof_lower_limits[:num_dofs].to(env.device).clone()

    # Canonical order: arm = 0..6, hand = 7..28.
    sine_amplitude = torch.zeros_like(default_pos)
    sine_amplitude[7:] = args.amplitude_frac * (upper[7:] - lower[7:])
    if args.joints == "all":
        sine_amplitude[:7] = args.arm_amplitude_frac * (upper[:7] - lower[:7])

    T = args.num_steps
    joint_names = list(env.joint_names)  # canonical (DFS) order
    joint_pos_log = np.zeros((T, num_dofs), dtype=np.float32)
    joint_vel_log = np.zeros((T, num_dofs), dtype=np.float32)
    target_log = np.zeros((T, num_dofs), dtype=np.float32)
    time_log = np.zeros(T, dtype=np.float32)

    sim = env.sim
    gym = env.gym
    decimation = max(1, env.control_freq_inv)

    for step in range(T):
        t = step * args.policy_dt
        time_log[step] = t

        target = default_pos + sine_amplitude * math.sin(
            2.0 * math.pi * args.sine_freq_hz * t
        )
        target = torch.clamp(target, lower, upper)

        # Bypass action pipeline: write target directly into env.cur_targets
        # and push to PhysX. cur_targets is shape (num_envs, num_dofs).
        env.cur_targets[:, :num_dofs] = target.unsqueeze(0)
        gym.set_dof_position_target_tensor(
            sim, gymtorch.unwrap_tensor(env.cur_targets)
        )

        for _ in range(decimation):
            gym.simulate(sim)
            gym.fetch_results(sim, True)
        gym.refresh_dof_state_tensor(sim)

        joint_pos_log[step] = (
            env.arm_hand_dof_pos[0, :num_dofs].detach().cpu().numpy()
        )
        joint_vel_log[step] = (
            env.arm_hand_dof_vel[0, :num_dofs].detach().cpu().numpy()
        )
        target_log[step] = target.detach().cpu().numpy()

        if camera_handle is not None:
            gym.step_graphics(sim)
            gym.render_all_camera_sensors(sim)
            rgba = gym.get_camera_image(
                sim, env.envs[0], camera_handle, gymapi.IMAGE_COLOR
            )
            # IMAGE_COLOR returns flat uint8 array — reshape to (H, W, 4).
            rgba = rgba.reshape(cam_props.height, cam_props.width, 4)
            frames.append(rgba[:, :, :3])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        joint_names=np.array(joint_names),
        joint_pos=joint_pos_log,
        joint_vel=joint_vel_log,
        target=target_log,
        time=time_log,
        policy_dt=args.policy_dt,
        sine_freq_hz=args.sine_freq_hz,
        amplitude_frac=args.amplitude_frac,
        side="isaacgym",
    )
    print(f"[isaacgym] Wrote {out_path} — {T} steps, {num_dofs} DOFs")

    if frames:
        import imageio
        video_path = Path(args.video)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(str(video_path), frames, fps=args.video_fps)
        print(f"[isaacgym] Wrote {video_path} — {len(frames)} frames @ {args.video_fps} fps")


if __name__ == "__main__":
    main()
