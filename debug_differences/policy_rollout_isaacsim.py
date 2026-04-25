"""Pretrained simtoolreal policy rollout — Isaac Sim side.

Self-contained driver modeled on ``sine_hand_isaacsim.py``: bootstraps
``Isaacsimenvs-SimToolReal-Direct-v0`` with all DR/reset randomness off,
restricts the procedural pool to a single hammer URDF, and pins the goal
to one env-local pose. Loads ``pretrained_policy/model.pth`` via
``RlPlayer`` and rolls it forward for ``--max_steps``. Dumps the same
per-step trace schema as the gym pair script (canonical joint order,
quat-wxyz internally — converted at diff time) plus an mp4.

    .venv_isaacsim/bin/python debug_differences/policy_rollout_isaacsim.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Keep in sync with policy_rollout_isaacgym.py. CLI is xyzw quat (matches
# gym's root_state_tensor convention); we convert to wxyz internally
# before assigning cfg.reset.fixed_goal_pose.
FIXED_GOAL_POSE_XYZ_XYZW = (0.0, 0.0, 0.78, 0.0, 0.0, 0.0, 1.0)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=240,
                        help="Policy steps (default 240 = 4s @ 60Hz).")
    parser.add_argument("--config", type=str,
                        default=str(REPO_ROOT / "pretrained_policy/config.yaml"))
    parser.add_argument("--checkpoint", type=str,
                        default=str(REPO_ROOT / "pretrained_policy/model.pth"))
    parser.add_argument("--num_assets_per_type", type=int, default=1,
                        help="Keep procedural object pool tiny for fast launch.")
    parser.add_argument("--rl_device", default="cuda")
    parser.add_argument("--out", type=str,
                        default=str(REPO_ROOT / "debug_differences/data/isaacsim_policy_rollout.npz"))
    parser.add_argument("--video", type=str,
                        default=str(REPO_ROOT / "debug_differences/plots/isaacsim_policy_rollout.mp4"),
                        help="MP4 output path. Empty string disables video.")
    parser.add_argument("--video_fps", type=int, default=30)
    parser.add_argument("--frame_every", type=int, default=2,
                        help="Capture one frame every N policy steps.")
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

    import gymnasium as gym
    import numpy as np
    import torch

    import isaacsimenvs  # noqa: F401  registers gym envs
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import SimToolRealEnvCfg
    from deployment.rl_player import RlPlayer

    cfg = SimToolRealEnvCfg()
    cfg.scene.num_envs = 1
    cfg.assets.num_assets_per_type = args.num_assets_per_type
    cfg.assets.handle_head_types = ("hammer",)
    # Disable shuffle so pool[0] = first matching ObjectSizeDistribution
    # (cuboid hammer) — matches policy_rollout_isaacgym.py.
    cfg.assets.shuffle_assets = False

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

    # CLI is xyzw; cfg.reset.fixed_goal_pose is wxyz (matches
    # write_root_pose_to_sim convention in reset_utils.py).
    x, y, z, qx, qy, qz, qw = FIXED_GOAL_POSE_XYZ_XYZW
    rs.fixed_goal_pose = (x, y, z, qw, qx, qy, qz)

    env = gym.make("Isaacsimenvs-SimToolReal-Direct-v0", cfg=cfg)
    inner = env.unwrapped
    inner._replay_target_lab_order = None  # not in replay mode

    # ---- DIAG: dump the procedural USD pool actually loaded ----
    print("=" * 78)
    print("[diag-sim] handle_head_types (cfg) :", cfg.assets.handle_head_types)
    print(f"[diag-sim] num_envs={cfg.scene.num_envs}, num_assets_per_type={cfg.assets.num_assets_per_type}")
    spawn = inner.object.cfg.spawn
    pool = list(getattr(spawn, "usd_path", []) or [])
    print(f"[diag-sim] Object asset pool size = {len(pool)}")
    for i, p in enumerate(pool):
        print(f"[diag-sim]   pool[{i}] = {p}")
    chosen_idx = 0 % max(1, len(pool))
    if pool:
        print(f"[diag-sim] env 0 picks pool[{chosen_idx}] = {pool[chosen_idx]}")
        # MultiUsdFileCfg gets a converted USD; the SOURCE URDF is one
        # generated under cfg.assets.generated_assets_dir. Try to surface it.
        from pathlib import Path as _P
        gen_dir = getattr(cfg.assets, "generated_assets_dir", None)
        if gen_dir:
            urdfs = sorted(_P(gen_dir).glob("*.urdf"))
            print(f"[diag-sim] {len(urdfs)} URDF(s) in {gen_dir}:")
            for u in urdfs:
                print(f"[diag-sim]   urdf -> {u}")
            if urdfs:
                print(f"[diag-sim] head of urdfs[0] (first 40 lines):")
                for ln in urdfs[0].read_text().splitlines()[:40]:
                    print(f"[diag-sim]   | {ln}")
    print("=" * 78)
    # ---- end DIAG ----

    # Optional camera sensor — same pose used by sine_hand_isaacsim.py /
    # eval_simtoolreal.py (matches legacy isaacgym cam: pos=(0,-1,1.03),
    # target=(0,0,0.53)).
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

    n_obs = 140
    n_act = cfg.action_space  # 29

    player = RlPlayer(
        num_observations=n_obs,
        num_actions=n_act,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.rl_device,
        num_envs=cfg.scene.num_envs,
    )
    player.player.init_rnn()

    obs, _ = env.reset()
    # Align with the gym pair script: gym's first env.step(zeros) serves as
    # both reset trigger AND the first physics step, so its step-0 obs is
    # post-1-tick. Our env.reset() is pre-physics, so advance one tick here
    # so step-0 obs lines up.
    obs, _, _, _, _ = env.step(
        torch.zeros((cfg.scene.num_envs, n_act), device=inner.device)
    )

    T = args.max_steps
    joint_names_canonical_torch_perm = inner._perm_lab_to_canon
    obs_log = np.zeros((T, n_obs), dtype=np.float32)
    action_log = np.zeros((T, n_act), dtype=np.float32)
    joint_pos_log = np.zeros((T, n_act), dtype=np.float32)
    joint_vel_log = np.zeros((T, n_act), dtype=np.float32)
    joint_targets_log = np.zeros((T, n_act), dtype=np.float32)
    object_state_log = np.zeros((T, 13), dtype=np.float32)
    goal_pose_log = np.zeros((T, 7), dtype=np.float32)
    reward_log = np.zeros(T, dtype=np.float32)

    for step in range(T):
        policy_obs = obs["policy"].to(args.rl_device)
        obs_log[step] = policy_obs[0].detach().cpu().numpy()

        # Joint state in canonical (DFS) order to match gym pair script.
        jpos_canon = inner.robot.data.joint_pos[:, joint_names_canonical_torch_perm]
        jvel_canon = inner.robot.data.joint_vel[:, joint_names_canonical_torch_perm]
        joint_pos_log[step] = jpos_canon[0].detach().cpu().numpy()
        joint_vel_log[step] = jvel_canon[0].detach().cpu().numpy()

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

        action = player.get_normalized_action(policy_obs, deterministic_actions=True)
        action_log[step] = action[0].detach().cpu().numpy()

        # _cur_targets is in Lab order; permute to canonical to match gym.
        joint_targets_log[step] = (
            inner._cur_targets[:, joint_names_canonical_torch_perm]
        )[0].detach().cpu().numpy()

        obs, reward, terminated, truncated, info = env.step(action.to(inner.device))
        reward_log[step] = float(reward.mean().detach().cpu().item())

        if camera is not None and step % args.frame_every == 0:
            camera.update(inner.physics_dt)
            rgb = camera.data.output["rgb"]
            if rgb is not None and rgb.shape[0] > 0:
                frames.append(rgb[0].cpu().numpy()[:, :, :3])

        if step % 60 == 0:
            print(f"[isaacsim] step {step:4d}  reward={reward_log[step]:+.3f}  "
                  f"frames={len(frames)}")

    from isaacsimenvs.tasks.simtoolreal.utils.scene_utils import JOINT_NAMES_CANONICAL
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        joint_names=np.array(list(JOINT_NAMES_CANONICAL)),
        obs=obs_log,
        action=action_log,
        joint_pos=joint_pos_log,
        joint_vel=joint_vel_log,
        joint_targets=joint_targets_log,
        object_state=object_state_log,
        goal_pose=goal_pose_log,
        reward=reward_log,
        side="isaacsim",
    )
    print(f"[isaacsim] Wrote {out_path} — {T} steps, {n_act} DOFs")

    if frames:
        import imageio
        video_path = Path(args.video)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(str(video_path), frames, fps=args.video_fps)
        print(f"[isaacsim] Wrote {video_path} — {len(frames)} frames @ {args.video_fps} fps")


if __name__ == "__main__":
    import os
    import sys
    main()
    # Kit shutdown hangs; force-exit (matches train.py).
    del _app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
