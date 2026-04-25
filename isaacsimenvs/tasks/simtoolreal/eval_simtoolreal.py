"""Run the pretrained SimToolReal policy against the new gym env, save mp4.

Single-env rollout via gym.make → Isaacsimenvs-SimToolReal-Direct-v0,
attaches an Isaac Lab Camera sensor, steps the policy for N steps, writes
captured RGB frames to mp4. Used to eyeball whether a checkpoint produces
sensible motion against the refactored DirectRLEnv (vs. the legacy
isaacsim_conversion path that play_simtoolreal.py uses).

    .venv_isaacsim/bin/python isaacsimenvs/tasks/simtoolreal/eval_simtoolreal.py \\
        --max_steps 600 --enable_cameras

Notes:
- --enable_cameras is required to attach the Camera sensor (without it
  Isaac Sim runs without RTX rendering and the mp4 will be empty).
- num_envs=1 by default — single-env inference. The procedural object pool
  is set to one USD per type to minimise startup conversion time.
- The pretrained policy expects 140 obs (matches cfg.obs.obs_list default).
  RlPlayer raises if the configured shape disagrees.
"""

from __future__ import annotations

import argparse
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", default="pretrained_policy/model.pth",
        help="rl_games .pth checkpoint (relative to repo root or absolute).",
    )
    parser.add_argument(
        "--config", default="pretrained_policy/config.yaml",
        help="Policy config YAML (relative to repo root or absolute).",
    )
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num_assets_per_type", type=int, default=1)
    parser.add_argument("--video_dir", default="rollout_videos_eval")
    parser.add_argument("--video_name", default="eval_simtoolreal.mp4")
    parser.add_argument("--video_fps", type=int, default=30)
    parser.add_argument("--frame_every", type=int, default=2,
                        help="Capture one frame every N policy steps")
    parser.add_argument("--rl_device", default="cuda")
    parser.add_argument("--obs_dump_npz", default=None,
                        help="If set, write a per-step .npz mirroring "
                             "dextoolbench/eval_simtoolreal_base.py for "
                             "side-by-side obs comparison.")
    parser.add_argument("--hold", action="store_true",
                        help="Bypass policy: every step send the action that "
                             "maps targets to current joint positions (arm: "
                             "0; hand: inverse of absolute scaling). Used to "
                             "isolate Isaac Lab vs Isaac Gym physics drift "
                             "from policy variance.")
    parser.add_argument("--hammer_only", action="store_true",
                        help="Restrict procedural pool to handle_head_types="
                             "(\"hammer\",). Paired with the collapsed "
                             "OBJECT_SIZE_DISTRIBUTIONS hammer-cuboid entry "
                             "this makes our generator and the legacy one "
                             "emit byte-equivalent first-asset URDFs.")
    parser.add_argument("--replay_targets_npz", default=None,
                        help="If set, load a (T, 29) joint_targets array from "
                             "this .npz (produced by "
                             "dextoolbench/eval_simtoolreal_base.py) and "
                             "bypass our action pipeline: send the recorded "
                             "target directly each step. Isolates PhysX 5 vs "
                             "PhysX 4 physics response to identical commands.")
    parser.add_argument("--fixed_goal_pose", type=float, nargs=7, default=None,
                        metavar=("X", "Y", "Z", "QX", "QY", "QZ", "QW"),
                        help="Pin the goal to a single env-local pose. "
                             "CLI is xyzw quat (matches the gym CLI in "
                             "dextoolbench/eval_simtoolreal_base.py); we "
                             "convert to wxyz internally before assigning "
                             "cfg.reset.fixed_goal_pose.")
    return parser


def _launch_app():
    from isaaclab.app import AppLauncher
    parser = _build_parser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    args.enable_cameras = True
    app = AppLauncher(args).app
    return app, args


_app, _args = _launch_app()


def _resolve_path(p: str) -> str:
    return p if Path(p).is_absolute() else str(REPO_ROOT / p)


def main() -> None:
    args = _args

    import gymnasium as gym
    import imageio
    import numpy as np
    import torch

    import isaaclab.sim as sim_utils
    from isaaclab.sensors import Camera, CameraCfg

    import isaacsimenvs  # noqa: F401  registers gym envs
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import SimToolRealEnvCfg
    from deployment.rl_player import RlPlayer

    cfg = SimToolRealEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.assets.num_assets_per_type = args.num_assets_per_type
    if args.hammer_only:
        cfg.assets.handle_head_types = ("hammer",)

    # Mirror dextoolbench/eval_simtoolreal_base.py overrides — disable all
    # DR (delay queues, noise, force/torque impulses) and reset randomness
    # so the obs trace is reproducible and comparable to the legacy dump.
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

    if args.fixed_goal_pose is not None:
        x, y, z, qx, qy, qz, qw = args.fixed_goal_pose
        # CLI is xyzw; cfg.reset.fixed_goal_pose is wxyz (matches
        # write_root_pose_to_sim convention in reset_utils.py).
        rs.fixed_goal_pose = (x, y, z, qw, qx, qy, qz)
        print(f"[eval] fixed_goal_pose (wxyz) = {rs.fixed_goal_pose}")

    env = gym.make("Isaacsimenvs-SimToolReal-Direct-v0", cfg=cfg)
    inner = env.unwrapped
    inner._replay_target_lab_order = None  # set per-step only in replay mode

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

    ckpt = _resolve_path(args.checkpoint)
    cfgp = _resolve_path(args.config)
    print(f"[eval] Loading policy: ckpt={ckpt}, cfg={cfgp}")
    obs_dim = cfg.observation_space
    action_dim = cfg.action_space
    player = RlPlayer(
        num_observations=obs_dim,
        num_actions=action_dim,
        config_path=cfgp,
        checkpoint_path=ckpt,
        device=args.rl_device,
        num_envs=args.num_envs,
    )
    player.player.init_rnn()

    obs, _ = env.reset()
    # Align with legacy eval timing: legacy's `env.step(zeros)` serves as
    # both the reset trigger AND the first physics step, so legacy's step-0
    # obs is post-1-tick. Our env.reset() is pre-physics. Advance one tick
    # here so step-0 obs lines up (hand PD response, etc.).
    obs, _, _, _, _ = env.step(
        torch.zeros((args.num_envs, action_dim), device=inner.device)
    )
    print(
        f"[eval] reset + 1-tick → policy obs {obs['policy'].shape}, "
        f"critic obs {obs['critic'].shape}"
    )

    video_dir = Path(args.video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = video_dir / args.video_name
    frames: list = []

    dump_obs = args.obs_dump_npz is not None
    if dump_obs:
        obs_log = np.zeros((args.max_steps, obs_dim), dtype=np.float32)
        action_log = np.zeros((args.max_steps, action_dim), dtype=np.float32)
        joint_pos_log = np.zeros((args.max_steps, action_dim), dtype=np.float32)
        joint_vel_log = np.zeros((args.max_steps, action_dim), dtype=np.float32)
        joint_targets_log = np.zeros((args.max_steps, action_dim), dtype=np.float32)
        object_state_log = np.zeros((args.max_steps, 13), dtype=np.float32)
        goal_pose_log = np.zeros((args.max_steps, 7), dtype=np.float32)
        reward_log = np.zeros(args.max_steps, dtype=np.float32)

    replay_targets = None
    if args.replay_targets_npz is not None:
        npz = np.load(args.replay_targets_npz, allow_pickle=True)
        replay_targets = torch.from_numpy(
            npz["joint_targets"][:args.max_steps].astype("float32")
        ).to(inner.device)
        print(f"[eval] Replay mode: loaded {replay_targets.shape[0]} targets "
              f"from {args.replay_targets_npz}")

    for step in range(args.max_steps):
        policy_obs = obs["policy"].to(args.rl_device)

        if dump_obs:
            # Capture canonical-order joint state by reading the same
            # internal buffers obs_utils does, applying the same perm.
            obs_log[step] = policy_obs[0].detach().cpu().numpy()
            jpos_canon = (
                inner.robot.data.joint_pos[:, inner._perm_lab_to_canon]
            )[0].detach().cpu().numpy()
            jvel_canon = (
                inner.robot.data.joint_vel[:, inner._perm_lab_to_canon]
            )[0].detach().cpu().numpy()
            joint_pos_log[step] = jpos_canon
            joint_vel_log[step] = jvel_canon

            obj_pos = (
                inner.object.data.root_pos_w[0] - inner.scene.env_origins[0]
            )
            obj_quat = inner.object.data.root_quat_w[0]
            obj_lvel = inner.object.data.root_lin_vel_w[0]
            obj_avel = inner.object.data.root_ang_vel_w[0]
            object_state_log[step] = torch.cat(
                [obj_pos, obj_quat, obj_lvel, obj_avel]
            ).detach().cpu().numpy()

            goal_pos = (
                inner.goal_viz.data.root_pos_w[0] - inner.scene.env_origins[0]
            )
            goal_quat = inner.goal_viz.data.root_quat_w[0]
            goal_pose_log[step] = torch.cat([goal_pos, goal_quat]).detach().cpu().numpy()

        if replay_targets is not None:
            # Replay mode: write the recorded canonical-order target into a
            # per-env attribute that apply_action_pipeline reads as an
            # override. We reorder to Lab positions here since apply_action_pipeline
            # normally does canonical→Lab internally and we're skipping that path.
            tgt_canon = replay_targets[step]  # (29,) canonical
            tgt_lab = tgt_canon[inner._perm_canon_to_lab].unsqueeze(0)  # (1, 29)
            inner._replay_target_lab_order = tgt_lab
            action = torch.zeros(
                args.num_envs, action_dim, device=inner.device, dtype=torch.float32
            )
        elif args.hold:
            # Hold mode: arm action=0 (velocity-delta keeps arm at
            # prev_target = current pos); hand action = inverse of
            # absolute scaling so target = current hand q.
            jpos_canon = inner.robot.data.joint_pos[:, inner._perm_lab_to_canon]
            lower_canon = inner._joint_lower_canon
            upper_canon = inner._joint_upper_canon
            action = torch.zeros_like(jpos_canon)
            hand_q = jpos_canon[:, 7:]
            hand_lo = lower_canon[7:]
            hand_hi = upper_canon[7:]
            action[:, 7:] = 2.0 * (hand_q - hand_lo) / (hand_hi - hand_lo) - 1.0
        else:
            action = player.get_normalized_action(policy_obs, deterministic_actions=True)

        if dump_obs:
            action_log[step] = action[0].detach().cpu().numpy()
            joint_targets_log[step] = (
                inner._cur_targets[:, inner._perm_lab_to_canon]
            )[0].detach().cpu().numpy()

        obs, reward, terminated, truncated, info = env.step(
            action.to(inner.device)
        )

        if dump_obs:
            reward_log[step] = float(reward.mean().detach().cpu().item())

        if step % args.frame_every == 0:
            camera.update(inner.physics_dt)
            rgb = camera.data.output["rgb"]
            if rgb is not None and rgb.shape[0] > 0:
                frames.append(rgb[0].cpu().numpy()[:, :, :3])

        if step % 60 == 0:
            print(
                f"[eval] step {step:4d}  reward={reward.mean().item():+.3f}  "
                f"frames={len(frames)}"
            )

    print(f"[eval] Rollout complete. {len(frames)} frames captured.")
    if frames:
        imageio.mimwrite(str(mp4_path), frames, fps=args.video_fps)
        print(f"[eval] Saved {mp4_path} ({mp4_path.stat().st_size} bytes)")
    else:
        print("[eval] WARNING: no frames captured (camera disabled?)")

    if dump_obs:
        from isaacsimenvs.tasks.simtoolreal.utils.scene_utils import JOINT_NAMES_CANONICAL
        out_path = Path(args.obs_dump_npz)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            str(out_path),
            obs=obs_log,
            action=action_log,
            joint_pos=joint_pos_log,
            joint_vel=joint_vel_log,
            joint_targets=joint_targets_log,
            object_state=object_state_log,
            goal_pose=goal_pose_log,
            reward=reward_log,
            joint_names=np.array(list(JOINT_NAMES_CANONICAL)),
        )
        print(f"[eval] Saved obs dump to {out_path}")

    # env.close()
    # use os._exit(0) to exit the program since env.close() hangs on shutdown
    import os
    os._exit(0)


if __name__ == "__main__":
    main()
