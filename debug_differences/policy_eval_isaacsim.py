"""Pretrained simtoolreal policy distributional eval — Isaac Sim side.

Mirror of ``policy_eval_isaacgym.py`` on the new ``Isaacsimenvs-SimToolReal-
Direct-v0`` env. Many envs in parallel with the *full* procedural asset pool
and *natural* goal sampling. All DR / reset noise off. Dumps a per-step
(T, N) trace plus a per-env asset_type tag (parsed from the procedural
URDF filename, same as the gym side).

    .venv_isaacsim/bin/python debug_differences/policy_eval_isaacsim.py

Outputs ``debug_differences/data/isaacsim_policy_eval.npz`` matching the
gym dump's schema. Pairs with ``plot_policy_eval_diff.py`` for aggregate
overlay plots.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Type token sits between ``{idx:03d}_`` and the next ``_handle`` separator
# on both backends. Legacy emits ``..._handle_head_{scales}...`` while the
# sim port emits ``..._handle_{scales}_head_{head_scales}...`` — both share
# the leading ``\d{3}_{type}_handle`` prefix, so this regex covers both.
_TYPE_RE = re.compile(r"^\d{3}_([a-zA-Z]+)_handle")


def _asset_type_from_path(path: str) -> str:
    name = Path(path).name
    m = _TYPE_RE.match(name)
    return m.group(1) if m else "unknown"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--num_steps", type=int, default=600,
                        help="Policy steps (default 600 = one full episode "
                        "@ 60Hz).")
    parser.add_argument("--num_assets_per_type", type=int, default=10,
                        help="Pool size = 12 × this. Lower = faster startup "
                        "(URDF→USD conversion is the bottleneck).")
    parser.add_argument("--config", type=str,
                        default=str(REPO_ROOT / "pretrained_policy/config.yaml"))
    parser.add_argument("--checkpoint", type=str,
                        default=str(REPO_ROOT / "pretrained_policy/model.pth"))
    parser.add_argument("--rl_device", default="cuda")
    parser.add_argument("--out", type=str,
                        default=str(REPO_ROOT / "debug_differences/data/policy_eval_default_episode/isaacsim_policy_eval.npz"))
    parser.add_argument("--video_envs", type=str, default="0",
                        help="Comma-separated env ids to capture mp4 video for. "
                        "Empty disables video.")
    parser.add_argument("--video_dir", type=str,
                        default=str(REPO_ROOT / "debug_differences/plots/policy_eval_default_episode/videos"))
    parser.add_argument("--video_fps", type=int, default=30)
    parser.add_argument("--frame_every", type=int, default=2,
                        help="Capture one frame every N policy steps.")
    parser.add_argument("--success_tolerance", type=float, default=0.01,
                        help="Pin success_tolerance (meters); overrides the "
                        "curriculum. Default 0.01 m = 1 cm. Pass <0 to keep "
                        "the curriculum.")
    return parser


def _launch_app():
    from isaaclab.app import AppLauncher
    parser = _build_parser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    # Camera sensor needs Kit launched with --enable_cameras (RTX renderer);
    # only request it when at least one env is being captured.
    args.enable_cameras = bool(args.video_envs.strip())
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
    cfg.scene.num_envs = args.num_envs
    cfg.assets.num_assets_per_type = args.num_assets_per_type
    # Full asset pool: keep default cfg.assets.handle_head_types (all types)
    # and default cfg.assets.shuffle_assets=True for natural per-env type
    # variety. Both backends seed numpy with 42, so the shuffle is reproducible.

    # All DR off so the only source of variance is asset shape and goal pose.
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
    rs.reset_dof_pos_random_interval_arm = 0.0
    rs.reset_dof_pos_random_interval_fingers = 0.0
    rs.reset_dof_vel_random_interval = 0.0
    rs.table_reset_z_range = 0.0
    rs.fixed_start_pose = (0.0, 0.0, rs.table_reset_z + rs.table_object_z_offset, 1.0, 0.0, 0.0, 0.0)

    # Natural goal sampling: leave fixed_goal_pose=None so reset_utils samples
    # uniformly in the target_volume.

    # Pin success_tolerance when args.success_tolerance >= 0 (parity with
    # legacy evalSuccessTolerance override). The cfg field is wired into
    # update_tolerance_curriculum in termination_utils.py.
    if args.success_tolerance >= 0:
        cfg.termination.eval_success_tolerance = float(args.success_tolerance)

    env = gym.make("Isaacsimenvs-SimToolReal-Direct-v0", cfg=cfg)
    inner = env.unwrapped
    inner._replay_target_lab_order = None  # not in replay mode

    # Per-env asset_type tag — parse from the per-env Object USD path.
    # spawn.usd_path is the cycled pool; each env_K's Object spawns from
    # usd_path[K % len(pool)] (see scene_utils.py:746-753).
    spawn = inner.object.cfg.spawn
    pool = list(getattr(spawn, "usd_path", []) or [])
    asset_paths = [pool[i % len(pool)] for i in range(args.num_envs)]
    asset_types = [_asset_type_from_path(p) for p in asset_paths]
    type_counts: dict[str, int] = {}
    for t in asset_types:
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"[sim] num_envs={args.num_envs}, asset pool size={len(pool)}")
    print(f"[sim] per-env asset_type histogram: {type_counts}")

    # Optional per-env video capture. One Camera per selected env, anchored
    # at /World/RecordCamera_env{K} with pose = env_origins[K] + (exp 2's
    # offset). Same intrinsics as policy_rollout_isaacsim.py.
    video_env_ids: list[int] = (
        [int(x) for x in args.video_envs.split(",") if x.strip() != ""]
        if args.video_envs else []
    )
    cameras: dict[int, object] = {}
    cam_frames: dict[int, list] = {}
    if video_env_ids:
        import isaaclab.sim as sim_utils
        from isaaclab.sensors import Camera, CameraCfg
        env_origins_w = inner.scene.env_origins.detach().cpu().numpy()
        for eid in video_env_ids:
            if not (0 <= eid < args.num_envs):
                print(f"[sim] WARN: video_env id {eid} out of range, skipping")
                continue
            ox, oy, oz = float(env_origins_w[eid][0]), float(env_origins_w[eid][1]), float(env_origins_w[eid][2])
            camera_cfg = CameraCfg(
                prim_path=f"/World/RecordCamera_env{eid:04d}",
                update_period=0,
                height=480, width=640,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0, focus_distance=400.0,
                    horizontal_aperture=20.955, clipping_range=(0.1, 100.0),
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=(ox + 0.0, oy + -1.0, oz + 1.03),
                    rot=(0.8507, 0.5257, 0.0, 0.0),
                    convention="opengl",
                ),
            )
            cameras[eid] = Camera(cfg=camera_cfg)
            cam_frames[eid] = []
        inner.sim.reset()  # finalize camera registration
        print(f"[sim] video capture enabled for envs {sorted(cameras.keys())} "
              f"every {args.frame_every} steps")

    n_obs = 140
    n_act = cfg.action_space  # 29

    player = RlPlayer(
        num_observations=n_obs,
        num_actions=n_act,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.rl_device,
        num_envs=args.num_envs,
    )
    player.player.init_rnn()

    obs, _ = env.reset()
    # Match gym driver's reset timing: gym's first env.step(zeros) is both
    # the reset trigger AND the first physics step. Our env.reset() is
    # pre-physics, so advance one tick.
    obs, _, _, _, _ = env.step(
        torch.zeros((args.num_envs, n_act), device=inner.device)
    )

    T = args.num_steps
    N = args.num_envs
    reward_log = np.zeros((T, N), dtype=np.float32)
    is_success_log = np.zeros((T, N), dtype=np.bool_)
    reset_log = np.zeros((T, N), dtype=np.bool_)
    progress_log = np.zeros((T, N), dtype=np.int32)
    successes_log = np.zeros((T, N), dtype=np.int32)
    lifted_log = np.zeros((T, N), dtype=np.bool_)
    obj_pos_log = np.zeros((T, N, 3), dtype=np.float32)
    goal_pos_log = np.zeros((T, N, 3), dtype=np.float32)

    for step in range(T):
        # State BEFORE this step's action — matches what the policy sees.
        # Object/goal pos in world coords; subtract env_origins below for
        # comparability with gym (legacy reports world coords too).
        obj_pos_log[step] = inner.object.data.root_pos_w[:, 0:3].detach().cpu().numpy()
        goal_pos_log[step] = inner.goal_viz.data.root_pos_w[:, 0:3].detach().cpu().numpy()
        progress_log[step] = inner.episode_length_buf.detach().cpu().numpy()

        policy_obs = obs["policy"].to(args.rl_device)
        action = player.get_normalized_action(policy_obs, deterministic_actions=True)
        obs, reward, terminated, truncated, info = env.step(action.to(inner.device))

        reward_log[step] = reward.detach().cpu().numpy()
        # _is_success was just refreshed in compute_intermediate_values during
        # _get_dones. _successes was just incremented in compute_terminations.
        is_success_log[step] = inner._is_success.detach().cpu().numpy().astype(bool)
        # DirectRLEnv combines terminated|truncated and resets envs at the
        # end of step(); but the returned tensors here pre-reset describe
        # exactly which envs reset.
        reset_log[step] = (
            terminated | truncated
        ).detach().cpu().numpy().astype(bool)
        successes_log[step] = inner._successes.detach().cpu().numpy().astype(np.int32)
        lifted_log[step] = inner._lifted_object.detach().cpu().numpy().astype(bool)

        if cameras and step % args.frame_every == 0:
            for eid, cam in cameras.items():
                cam.update(inner.physics_dt)
                rgb = cam.data.output["rgb"]
                if rgb is not None and rgb.shape[0] > 0:
                    cam_frames[eid].append(rgb[0].cpu().numpy()[:, :, :3])

        if step % 60 == 0:
            print(f"[isaacsim] step {step:4d}  "
                  f"mean_reward={reward.mean().item():+.3f}  "
                  f"hits_so_far={int(successes_log[step].sum())}  "
                  f"resets_this_step={int(reset_log[step].sum())}")

    obj_to_goal_dist = np.linalg.norm(obj_pos_log - goal_pos_log, axis=-1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        reward=reward_log,
        is_success=is_success_log,
        reset=reset_log,
        progress=progress_log,
        successes=successes_log,
        lifted_object=lifted_log,
        obj_pos_world=obj_pos_log,
        goal_pos_world=goal_pos_log,
        obj_to_goal_dist=obj_to_goal_dist.astype(np.float32),
        asset_types=np.array(asset_types),
        asset_paths=np.array(asset_paths),
        side="isaacsim",
    )
    print(f"[isaacsim] Wrote {out_path} — T={T}, N={N}, "
          f"total goal hits={int(is_success_log.sum())}, "
          f"total resets={int(reset_log.sum())}")

    if cam_frames:
        import imageio
        video_dir = Path(args.video_dir)
        video_dir.mkdir(parents=True, exist_ok=True)
        for eid, frames in cam_frames.items():
            if not frames:
                continue
            atype = asset_types[eid] if eid < len(asset_types) else "unknown"
            mp4 = video_dir / f"isaacsim_env{eid:04d}_{atype}.mp4"
            imageio.mimwrite(str(mp4), frames, fps=args.video_fps)
            print(f"[isaacsim] Wrote {mp4} — {len(frames)} frames")


if __name__ == "__main__":
    import os
    import sys
    main()
    # Kit shutdown hangs; force-exit (matches train.py).
    del _app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
