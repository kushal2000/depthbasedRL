"""Pretrained simtoolreal policy distributional eval — Isaac Gym side.

Self-contained driver modeled on ``policy_rollout_isaacgym.py`` but scaled
out: many envs running in parallel with the *full* procedural asset pool
and *natural* goal sampling. All DR / reset noise still off so the only
unfixed dimensions are (asset_type-per-env, goal-pose-per-reset). Dumps a
per-step (T, N) trace plus a per-env asset_type tag so the diff script can
break aggregate stats down by asset type.

    .venv/bin/python debug_differences/policy_eval_isaacgym.py

Outputs ``debug_differences/data/isaacgym_policy_eval.npz`` with:
  reward, is_success, reset, progress, successes, lifted_object,
  obj_pos_world, goal_pos_world, obj_to_goal_dist : (T, N) or (T, N, 3)
  asset_types, asset_paths : (N,) string

Pairs with ``policy_eval_isaacsim.py``; aggregate-diff plots come from
``plot_policy_eval_diff.py``.
"""

from __future__ import annotations

# isort: off
# IMPORTANT: isaacgym must be imported before torch.
from isaacgym import gymapi, gymtorch, gymutil  # noqa: F401
# isort: on

import argparse
import math
import re
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from deployment.rl_player import RlPlayer
from isaacgymenvs.tasks import isaacgym_task_map
from isaacgymenvs.utils.reformat import omegaconf_to_dict


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
                        help="Policy steps per env (default 600 = one full "
                        "episode @ 60Hz).")
    parser.add_argument("--num_assets_per_type", type=int, default=10,
                        help="Pool size = 12 × this. 10 keeps startup quick "
                        "while still giving each type multiple shape variants.")
    parser.add_argument("--config", type=str,
                        default=str(REPO_ROOT / "pretrained_policy/config.yaml"))
    parser.add_argument("--checkpoint", type=str,
                        default=str(REPO_ROOT / "pretrained_policy/model.pth"))
    parser.add_argument("--sim_device", default="cuda:0")
    parser.add_argument("--rl_device", default="cuda:0")
    parser.add_argument("--graphics_device_id", type=int, default=0)
    parser.add_argument("--out", type=str,
                        default=str(REPO_ROOT / "debug_differences/data/policy_eval_default_episode/isaacgym_policy_eval.npz"))
    parser.add_argument("--video_envs", type=str, default="0",
                        help="Comma-separated env ids to capture mp4 video for "
                        "(e.g. '0' or '0,32,128'). Empty disables video.")
    parser.add_argument("--video_dir", type=str,
                        default=str(REPO_ROOT / "debug_differences/plots/policy_eval_default_episode/videos"))
    parser.add_argument("--video_fps", type=int, default=30)
    parser.add_argument("--frame_every", type=int, default=2,
                        help="Capture one frame every N policy steps (matches "
                        "exp 2 default for byte-comparable cadence).")
    parser.add_argument("--success_tolerance", type=float, default=0.01,
                        help="Pin success_tolerance to this value (meters); "
                        "overrides the curriculum so the success criterion "
                        "is reproducible across runs. Default 0.01 m = 1 cm "
                        "(curriculum floor). Pass <0 to keep the curriculum.")
    return parser


def _build_cfg(args) -> dict:
    """Compose the SimToolReal config: many envs, full asset pool, natural
    goal sampling, all DR/reset noise off. Mirrors the exp-2 driver minus
    fixed-goal / hammer-only pinning."""
    cfg_dir = str((REPO_ROOT / "isaacgymenvs" / "cfg").resolve())
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "task=SimToolReal",
                f"task.env.numEnvs={args.num_envs}",
                f"task.env.numAssetsPerType={args.num_assets_per_type}",
                # All reset randomness off (matches exp 2).
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
                # Capture-viewer / video off.
                "task.env.capture_viewer=False",
                "task.env.capture_video=False",
                # Use the cfg default episodeLength (600) — the natural reset
                # cadence is part of what we want to characterize across
                # backends. progress_buf zeros on every goal hit (env.py:2511),
                # so an env that's hitting goals ride out indefinitely.
                "task.env.resetWhenDropped=False",
                # Natural goal sampling — explicitly disable any fixed-goal
                # override that may have leaked from a sibling experiment's
                # global hydra cache.
                "task.env.useFixedGoalStates=False",
                "task.env.fixedGoalStatesJsonPath=null",
                # Pin success criterion when args.success_tolerance >= 0
                # (legacy `evalSuccessTolerance` override is wired in
                # env.py:1573-1575). Negative value falls through to the
                # curriculum default.
                *(
                    [f"task.env.evalSuccessTolerance={args.success_tolerance}"]
                    if args.success_tolerance >= 0
                    else []
                ),
            ],
        )
    return omegaconf_to_dict(cfg.task)


def main() -> None:
    args = _build_parser().parse_args()

    cfg_dict = _build_cfg(args)
    env = isaacgym_task_map["SimToolReal"](
        cfg=cfg_dict,
        rl_device=args.rl_device,
        sim_device=args.sim_device,
        graphics_device_id=args.graphics_device_id,
        headless=True,
        virtual_screen_capture=False,
        force_render=False,
    )

    pool = list(env.object_asset_files)
    print(f"[gym] num_envs={env.num_envs}, asset pool size={len(pool)}")
    asset_paths = [pool[i % len(pool)] for i in range(env.num_envs)]
    asset_types = [_asset_type_from_path(p) for p in asset_paths]
    type_counts: dict[str, int] = {}
    for t in asset_types:
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"[gym] per-env asset_type histogram: {type_counts}")

    # Optional per-env video capture. We attach one camera_sensor to each
    # selected env; same intrinsics as policy_rollout_isaacgym.py so frames
    # are comparable to the sim-side mp4s.
    video_env_ids: list[int] = (
        [int(x) for x in args.video_envs.split(",") if x.strip() != ""]
        if args.video_envs else []
    )
    cam_props = None
    cam_handles: dict[int, object] = {}
    cam_frames: dict[int, list] = {}
    if video_env_ids:
        cam_props = gymapi.CameraProperties()
        cam_props.width = 640
        cam_props.height = 480
        # PinholeCameraCfg(focal=24mm, aperture=20.955mm) ⇒ ~47.10° FoV.
        cam_props.horizontal_fov = math.degrees(
            2.0 * math.atan(20.955 / (2.0 * 24.0))
        )
        for eid in video_env_ids:
            if not (0 <= eid < env.num_envs):
                print(f"[gym] WARN: video_env id {eid} out of range, skipping")
                continue
            env_ptr = env.envs[eid]
            handle = env.gym.create_camera_sensor(env_ptr, cam_props)
            # Camera pose is *env-local* in isaacgym, so the same offset
            # works for any selected env (env grid handles the world shift).
            env.gym.set_camera_location(
                handle, env_ptr,
                gymapi.Vec3(0.0, -1.0, 1.03),
                gymapi.Vec3(0.0, 0.0, 0.53),
            )
            cam_handles[eid] = handle
            cam_frames[eid] = []
        print(f"[gym] video capture enabled for envs {sorted(cam_handles.keys())} "
              f"every {args.frame_every} steps")

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
    N = env.num_envs
    reward_log = np.zeros((T, N), dtype=np.float32)
    is_success_log = np.zeros((T, N), dtype=np.bool_)
    reset_log = np.zeros((T, N), dtype=np.bool_)
    progress_log = np.zeros((T, N), dtype=np.int32)
    successes_log = np.zeros((T, N), dtype=np.int32)
    lifted_log = np.zeros((T, N), dtype=np.bool_)
    obj_pos_log = np.zeros((T, N, 3), dtype=np.float32)
    goal_pos_log = np.zeros((T, N, 3), dtype=np.float32)

    for step in range(T):
        # State BEFORE the step (matches what the policy sees this step).
        obj_pos_log[step] = env.object_state[:, 0:3].detach().cpu().numpy()
        goal_pos_log[step] = env.goal_pose[:, 0:3].detach().cpu().numpy()
        progress_log[step] = env.progress_buf.detach().cpu().numpy()

        action = player.get_normalized_action(obs, deterministic_actions=True)
        obs_dict, reward, _done, _ = env.step(action)
        obs = obs_dict["obs"]

        reward_log[step] = reward.detach().cpu().numpy()
        # reset_goal_buf was set inside the reward computation this step;
        # it carries 1 where a goal was hit (env.py:2720). reset_buf carries
        # all reset causes (drop / max-succ / hand-far / time).
        is_success_log[step] = env.reset_goal_buf.detach().cpu().numpy().astype(bool)
        reset_log[step] = env.reset_buf.detach().cpu().numpy().astype(bool)
        successes_log[step] = env.successes.detach().cpu().numpy().astype(np.int32)
        lifted_log[step] = env.lifted_object.detach().cpu().numpy().astype(bool)

        if cam_handles and step % args.frame_every == 0:
            # GPU pipeline must explicitly fetch results before step_graphics —
            # see policy_rollout_isaacgym.py for the rationale.
            env.gym.fetch_results(env.sim, True)
            env.gym.step_graphics(env.sim)
            env.gym.render_all_camera_sensors(env.sim)
            env.gym.start_access_image_tensors(env.sim)
            for eid, handle in cam_handles.items():
                rgba = env.gym.get_camera_image(
                    env.sim, env.envs[eid], handle, gymapi.IMAGE_COLOR
                )
                rgba = rgba.reshape(cam_props.height, cam_props.width, 4)
                cam_frames[eid].append(rgba[:, :, :3])
            env.gym.end_access_image_tensors(env.sim)

        if step % 60 == 0:
            print(f"[isaacgym] step {step:4d}  "
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
        side="isaacgym",
    )
    print(f"[isaacgym] Wrote {out_path} — T={T}, N={N}, "
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
            mp4 = video_dir / f"isaacgym_env{eid:04d}_{atype}.mp4"
            imageio.mimwrite(str(mp4), frames, fps=args.video_fps)
            print(f"[isaacgym] Wrote {mp4} — {len(frames)} frames")


if __name__ == "__main__":
    main()
