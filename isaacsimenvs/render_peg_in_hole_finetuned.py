"""Render/evaluate a finetuned PegInHole policy on its task distribution.

This is intentionally separate from ``render_simtoolreal_pretrained.py``:
the SimToolReal video focuses on object/pretraining diversity, while this
script focuses on a single finetuned checkpoint/task such as FurnitureBench
200 mm leg screwing.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_CHECKPOINT = "/juno/u/kedia/depthbasedRL/train_dir/May26/screwing_newer/model.pth"
DEFAULT_TASK = "Isaacsimenvs-PegInHole-Direct-v0"
DEFAULT_AGENT = "rl_games_sapg_cfg_entry_point"
DEFAULT_PROBLEM = "furniture_bench.one_leg_leg4_200mm_matchedmass_sdf_hybrid_super_dense"


def _parse_step_list(value: str) -> list[int]:
    if value.strip() == "":
        return []
    return sorted({int(item) for item in value.split(",") if item.strip()})


def _apply_200mm_furniturebench_overrides(env_cfg, args) -> None:
    """Match the fig4 200 mm long-leg screwing finetune setup."""
    env_cfg.scene.num_envs = int(args.num_envs)
    env_cfg.peg_in_hole.problem = str(args.problem)
    env_cfg.peg_in_hole.goal_mode = str(args.goal_mode)

    # Goal/tolerance knobs from plot_figures/fig4/experiments/
    # furniture_bench_leg4_200mm_finetune_rgf10_dr.sub.
    env_cfg.peg_in_hole.goal_xy_obs_noise = 0.002
    env_cfg.peg_in_hole.goal_yaw_obs_noise_deg = 1.0
    env_cfg.peg_in_hole.random_goal_fraction = float(args.random_goal_fraction)
    env_cfg.peg_in_hole.random_goal_max_successes = 5
    env_cfg.peg_in_hole.insertion_success_tolerance = 0.005
    env_cfg.peg_in_hole.hole_yaw_range_deg = 10.0
    env_cfg.termination.success_tolerance = 0.01
    env_cfg.termination.target_success_tolerance = 0.002
    env_cfg.termination.success_steps = 1
    env_cfg.peg_in_hole.enable_retract = True
    env_cfg.peg_in_hole.retract_reward_scale = 1.0
    env_cfg.peg_in_hole.retract_distance_threshold = 0.1
    env_cfg.peg_in_hole.retract_success_tolerance = 0.005
    env_cfg.termination.force_consecutive_near_goal_steps = True

    # Reset/reward knobs.
    env_cfg.reset.reset_position_noise_x = float(args.reset_position_noise_x)
    env_cfg.reset.reset_position_noise_y = float(args.reset_position_noise_y)
    env_cfg.reset.reset_position_noise_z = float(args.reset_position_noise_z)
    env_cfg.reset.reset_dof_pos_random_interval_arm = float(args.reset_dof_pos_noise_arm)
    env_cfg.reset.reset_dof_pos_random_interval_fingers = float(args.reset_dof_pos_noise_fingers)
    env_cfg.reset.reset_dof_vel_random_interval = float(args.reset_dof_vel_noise)
    env_cfg.reset.table_reset_z_range = float(args.table_reset_z_range)
    env_cfg.reward.lifting_rew_scale = 20.0
    env_cfg.reward.lifting_bonus = 300.0

    dr = env_cfg.domain_randomization
    dr.force_scale = float(args.force_scale)
    dr.torque_scale = float(args.torque_scale)
    if hasattr(dr, "force_only_when_lifted"):
        dr.force_only_when_lifted = bool(args.force_only_when_lifted)
    if hasattr(dr, "torque_only_when_lifted"):
        dr.torque_only_when_lifted = bool(args.torque_only_when_lifted)

    if args.train_dr:
        dr.use_obs_delay = True
        dr.use_action_delay = True
        dr.use_object_state_delay_noise = True
        dr.object_state_xyz_noise_std = 0.01
        dr.object_state_rotation_noise_degrees = 5.0
        dr.joint_velocity_obs_noise_std = 0.1
        dr.object_scale_noise_multiplier_range = (0.9, 1.1)
    else:
        dr.use_obs_delay = False
        dr.use_action_delay = False
        dr.use_object_state_delay_noise = False
        dr.object_state_xyz_noise_std = 0.0
        dr.object_state_rotation_noise_degrees = 0.0
        dr.joint_velocity_obs_noise_std = 0.0
        dr.object_scale_noise_multiplier_range = (1.0, 1.0)


def _tensor_list(x) -> list[float]:
    return [float(v) for v in x.detach().cpu().reshape(-1).tolist()]


def _style_scene_for_video(env, args) -> dict[str, Any]:
    from isaacsimenvs.render_simtoolreal_pretrained import (
        DEFAULT_NVIDIA_PRECAST_CONCRETE_MDL,
        _apply_backdrop_walls,
        _apply_pbr_tile_floor,
        _apply_single_sun_lighting,
        _apply_sky_background,
        _set_default_world_light_intensity,
        _style_prim_trees,
    )

    summary: dict[str, Any] = {}
    summary["table"] = _style_prim_trees(
        "/World/envs/env_.*/Table",
        color=tuple(float(v) for v in args.table_color),
        opacity=1.0,
        visible=True,
        label="table",
    )
    summary["ground_base"] = _style_prim_trees(
        "/World/ground",
        color=(0.55, 0.56, 0.54),
        opacity=1.0,
        visible=True,
        label="floor_base",
    )
    summary["floor"] = _apply_pbr_tile_floor(
        base_color=tuple(float(v) for v in args.floor_color),
        tile_count=int(args.floor_tile_count),
        tile_size=float(args.floor_tile_size),
        tile_gap=float(args.floor_tile_gap),
        texture_path=str(DEFAULT_NVIDIA_PRECAST_CONCRETE_MDL),
        texture_scale=float(args.floor_texture_scale),
        material_name="FurnitureBenchPrecastConcrete",
        style_name="nvidia_precast_concrete_dark_gray",
        mdl_material_name="concrete_precast_dark_charcoal",
    )
    if args.hide_goal_viz:
        summary["goal_viz"] = _style_prim_trees(
            "/World/envs/env_.*/GoalViz",
            color=(0.25, 0.95, 0.35),
            opacity=0.0,
            visible=False,
            label="goal_viz",
        )
    summary["sky"] = _apply_sky_background(
        style=args.sky_style,
        color=tuple(float(v) for v in args.sky_color),
        dome_intensity=float(args.sky_dome_intensity),
        hdri_preset="stinson_beach",
        hdri_path=None,
    )
    summary["default_light"] = _set_default_world_light_intensity(float(args.default_light_intensity))
    summary["backdrop"] = _apply_backdrop_walls(
        env,
        style=args.backdrop_style,
        color=tuple(float(v) for v in args.backdrop_color),
        horizon_color=tuple(float(v) for v in args.backdrop_horizon_color),
        distance=float(args.backdrop_distance),
        height=float(args.backdrop_height),
        extent_margin=float(args.backdrop_extent_margin),
    )
    summary["sun"] = _apply_single_sun_lighting(
        exposure=float(args.single_sun_exposure),
        angle=float(args.single_sun_angle),
        color_temperature=float(args.single_sun_color_temperature),
        yaw_offset_deg=float(args.single_sun_yaw_offset_deg),
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--agent", default=DEFAULT_AGENT)
    parser.add_argument("--problem", default=DEFAULT_PROBLEM)
    parser.add_argument("--goal_mode", default="preInsertAndFinal")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--video_fps", type=int, default=30)
    parser.add_argument("--capture_png_steps", type=_parse_step_list, default=_parse_step_list("0"))
    parser.add_argument("--make_video", action="store_true")
    parser.add_argument("--no_render", action="store_true")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rl_device", default="cuda:0")
    parser.add_argument("--out_dir", type=Path, default=Path("local_logs/furniturebench_200mm_finetuned"))
    parser.add_argument("--no_timestamp_out_dir", action="store_true")

    # The finetune run used rgf=0.1. For videos, default to screwing episodes
    # only so a single-env rollout does not randomly become a free-space goal.
    parser.add_argument("--random_goal_fraction", type=float, default=0.0)
    parser.add_argument("--train_dr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force_scale", type=float, default=0.0)
    parser.add_argument("--torque_scale", type=float, default=0.0)
    parser.add_argument("--force_only_when_lifted", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--torque_only_when_lifted", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--reset_position_noise_x", type=float, default=0.1)
    parser.add_argument("--reset_position_noise_y", type=float, default=0.1)
    parser.add_argument("--reset_position_noise_z", type=float, default=0.02)
    parser.add_argument("--reset_dof_pos_noise_arm", type=float, default=0.1)
    parser.add_argument("--reset_dof_pos_noise_fingers", type=float, default=0.1)
    parser.add_argument("--reset_dof_vel_noise", type=float, default=0.5)
    parser.add_argument("--table_reset_z_range", type=float, default=0.01)

    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--render_quality_preset", choices=("default", "beauty"), default="beauty")
    parser.add_argument("--render_mode", choices=("auto", "default", "rt", "pt"), default="rt")
    parser.add_argument("--render_samples_per_pixel", type=int, default=64)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hold_open_s", type=float, default=0.0)

    # Fixed straight-on camera. Values are env-local, centered on env 0.
    parser.add_argument("--camera_eye", type=float, nargs=3, default=(0.12, -1.55, 0.98))
    parser.add_argument("--camera_target", type=float, nargs=3, default=(0.02, 0.02, 0.58))
    parser.add_argument("--camera_focal_length_cm", type=float, default=24.0)
    parser.add_argument("--camera_focus_distance_m", type=float, default=1.3)
    parser.add_argument("--camera_f_stop", type=float, default=0.0)
    parser.add_argument("--camera_horizontal_aperture_cm", type=float, default=20.955)
    parser.add_argument("--camera_render_warmup_frames", type=int, default=1)

    parser.add_argument("--table_color", type=float, nargs=3, default=(0.42, 0.27, 0.15))
    parser.add_argument("--floor_color", type=float, nargs=3, default=(0.44, 0.45, 0.43))
    parser.add_argument("--floor_tile_count", type=int, default=64)
    parser.add_argument("--floor_tile_size", type=float, default=0.9)
    parser.add_argument("--floor_tile_gap", type=float, default=0.006)
    parser.add_argument("--floor_texture_scale", type=float, default=4.0)
    parser.add_argument("--hide_goal_viz", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sky_style", choices=("default", "blue_color", "blue_dome", "hdri", "dynamic_clear_sky"), default="blue_dome")
    parser.add_argument("--sky_color", type=float, nargs=3, default=(0.50, 0.66, 0.86))
    parser.add_argument("--sky_dome_intensity", type=float, default=650.0)
    parser.add_argument("--backdrop_style", choices=("none", "blue_wall", "blue_walls", "gradient_sky"), default="gradient_sky")
    parser.add_argument("--backdrop_color", type=float, nargs=3, default=(0.25, 0.48, 0.76))
    parser.add_argument("--backdrop_horizon_color", type=float, nargs=3, default=(0.68, 0.78, 0.88))
    parser.add_argument("--backdrop_distance", type=float, default=5.0)
    parser.add_argument("--backdrop_height", type=float, default=18.0)
    parser.add_argument("--backdrop_extent_margin", type=float, default=30.0)
    parser.add_argument("--default_light_intensity", type=float, default=120.0)
    parser.add_argument("--single_sun_exposure", type=float, default=6.8)
    parser.add_argument("--single_sun_angle", type=float, default=0.45)
    parser.add_argument("--single_sun_color_temperature", type=float, default=5250.0)
    parser.add_argument("--single_sun_yaw_offset_deg", type=float, default=70.0)
    parser.add_argument("--image_exposure", type=float, default=0.72)
    parser.add_argument("--image_contrast", type=float, default=1.08)
    parser.add_argument("--image_saturation", type=float, default=1.03)
    parser.add_argument("--image_gamma", type=float, default=1.0)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = args.out_dir if args.no_timestamp_out_dir else args.out_dir.parent / f"{timestamp}_{args.out_dir.name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    from isaaclab.app import AppLauncher

    launcher_parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(launcher_parser)
    launcher_args, _ = launcher_parser.parse_known_args([])
    launcher_args.headless = bool(args.headless)
    launcher_args.enable_cameras = not bool(args.no_render)
    app = AppLauncher(launcher_args).app

    import gymnasium as gym
    import imageio
    import torch
    from isaaclab.sensors import Camera
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    import isaacsimenvs  # noqa: F401
    from isaacsimenvs.render_simtoolreal_pretrained import (
        _apply_beauty_render_settings,
        _apply_runtime_render_settings,
        _capture_rgb,
        _make_camera_cfg,
        _postprocess_rgb_frame,
        _save_manifest,
        _set_active_viewport_camera,
        _set_record_camera,
    )
    from isaacsimenvs.tasks.simtoolreal.utils.obs_utils import compute_intermediate_values
    from isaacsimenvs.utils.rlgames_utils import register_rlgames_env
    from rl_games.torch_runner import Runner

    torch.manual_seed(args.seed)

    env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    _apply_200mm_furniturebench_overrides(env_cfg, args)
    render_quality_summary = None
    runtime_render_summary = None
    if not args.no_render:
        render_quality_summary = _apply_beauty_render_settings(
            env_cfg,
            preset=args.render_quality_preset,
            samples_per_pixel=int(args.render_samples_per_pixel),
            dome_light_upper_lower_strategy=None,
        )
        runtime_render_summary = _apply_runtime_render_settings(
            preset=args.render_quality_preset,
            render_mode=args.render_mode,
            samples_per_pixel=int(args.render_samples_per_pixel),
            dome_light_upper_lower_strategy=None,
        )
    if hasattr(env_cfg, "seed"):
        env_cfg.seed = args.seed

    spec = gym.spec(args.task)
    mod_name, cls_name = spec.entry_point.split(":")
    env_cls = getattr(importlib.import_module(mod_name), cls_name)
    env = env_cls(cfg=env_cfg)

    camera = None
    if not args.no_render:
        camera = Camera(
            cfg=_make_camera_cfg(
                int(args.width),
                int(args.height),
                focal_length_cm=float(args.camera_focal_length_cm),
                focus_distance_m=float(args.camera_focus_distance_m),
                f_stop=float(args.camera_f_stop),
                horizontal_aperture_cm=float(args.camera_horizontal_aperture_cm),
            )
        )
    env.sim.reset()

    style_summary = None
    if not args.no_render:
        style_summary = _style_scene_for_video(env, args)

    agent_cfg = load_cfg_from_registry(args.task, args.agent)
    clip_obs = float(agent_cfg["params"]["env"].get("clip_observations", math.inf))
    clip_actions = float(agent_cfg["params"]["env"].get("clip_actions", math.inf))
    wrapped = register_rlgames_env(
        env,
        rl_device=args.rl_device,
        clip_obs=clip_obs,
        clip_actions=clip_actions,
    )
    agent_cfg["params"]["config"]["device"] = args.rl_device
    agent_cfg["params"]["config"]["device_name"] = args.rl_device

    runner = Runner()
    runner.load(agent_cfg)
    runner.reset()
    player = runner.create_player()
    player.restore(str(checkpoint))
    player.has_batch_dimension = True
    player.reset()
    obs = player.env_reset(wrapped)

    physics_dt = env.sim.get_physics_dt()
    decimation = int(getattr(env.cfg, "decimation", 1) or 1)
    policy_dt = physics_dt * decimation
    capture_every = max(1, round((1.0 / float(args.video_fps)) / policy_dt))
    capture_steps = set(args.capture_png_steps)
    frames = []
    completed: list[dict[str, Any]] = []

    manifest: dict[str, Any] = {
        "script": Path(__file__).name,
        "checkpoint": str(checkpoint),
        "task": args.task,
        "agent": args.agent,
        "problem": args.problem,
        "goal_mode": args.goal_mode,
        "seed": int(args.seed),
        "num_envs": int(args.num_envs),
        "steps": int(args.steps),
        "train_dr": bool(args.train_dr),
        "random_goal_fraction": float(args.random_goal_fraction),
        "camera_eye": list(args.camera_eye),
        "camera_target": list(args.camera_target),
        "render_quality_summary": render_quality_summary,
        "runtime_render_summary": runtime_render_summary,
        "style_summary": style_summary,
        "outputs": {"pngs": [], "video": None},
        "metrics": {},
    }

    def capture(step_i: int) -> None:
        if args.no_render or camera is None:
            return
        import torch

        env_origin = env.scene.env_origins[0]
        eye = env_origin + torch.tensor(args.camera_eye, device=env.device, dtype=torch.float32)
        target = env_origin + torch.tensor(args.camera_target, device=env.device, dtype=torch.float32)
        _set_record_camera(camera, eye, target)
        if not args.headless:
            _set_active_viewport_camera("/World/RecordCamera")
        for _ in range(max(1, int(args.camera_render_warmup_frames))):
            env.sim.render()
        frame = _capture_rgb(camera, dt=policy_dt)
        if frame is None:
            print(f"[warning] no RGB frame at step={step_i}")
            return
        frame = _postprocess_rgb_frame(
            frame,
            exposure=float(args.image_exposure),
            contrast=float(args.image_contrast),
            saturation=float(args.image_saturation),
            gamma=float(args.image_gamma),
        )
        if step_i in capture_steps:
            png_path = out_dir / f"step_{step_i:04d}.png"
            imageio.imwrite(str(png_path), frame)
            manifest["outputs"]["pngs"].append(str(png_path))
            print(f"[render_peg_in_hole_finetuned] wrote {png_path}")
        if args.make_video and step_i % capture_every == 0:
            frames.append(frame)

    def record_completed(done_mask) -> int:
        done = torch.as_tensor(done_mask, device=env.device).reshape(-1).bool()
        ids = done.nonzero(as_tuple=False).flatten()
        for env_id in ids.detach().cpu().tolist():
            successes = int(env._prev_episode_successes[env_id].item())
            max_goals = int(env.prev_episode_env_max_goals[env_id].item())
            completed.append(
                {
                    "step": current_step,
                    "env_id": int(env_id),
                    "successes": successes,
                    "max_goals": max_goals,
                    "all_goals_hit": bool(successes >= max_goals),
                }
            )
        return int(ids.numel())

    print(
        "[render_peg_in_hole_finetuned] rolling out "
        f"steps={args.steps} envs={args.num_envs} problem={args.problem}",
        flush=True,
    )
    compute_intermediate_values(env)
    current_step = 0
    capture(0)
    for current_step in range(1, int(args.steps) + 1):
        action = player.get_action(obs, is_deterministic=bool(args.deterministic))
        obs, rew, dones, infos = player.env_step(wrapped, action)
        num_just_completed = record_completed(dones)
        if current_step % 60 == 0 or num_just_completed:
            current_successes = int(env._successes.max().item())
            current_max = int(env.env_max_goals.max().item())
            print(
                "[metrics] "
                f"step={current_step} current_successes_max={current_successes}/{current_max} "
                f"completed={len(completed)} "
                f"completed_full_successes={sum(int(x['all_goals_hit']) for x in completed)}",
                flush=True,
            )
        if current_step in capture_steps or (args.make_video and current_step % capture_every == 0):
            capture(current_step)

    if args.make_video and frames:
        video_path = out_dir / "rollout.mp4"
        imageio.mimwrite(str(video_path), frames, fps=int(args.video_fps), macro_block_size=1)
        manifest["outputs"]["video"] = str(video_path)
        print(f"[render_peg_in_hole_finetuned] wrote {len(frames)} frames to {video_path}")

    current_successes = _tensor_list(env._successes.float())
    current_max_goals = _tensor_list(env.env_max_goals.float())
    manifest["metrics"] = {
        "completed": completed,
        "completed_count": len(completed),
        "completed_full_success_rate": (
            sum(int(x["all_goals_hit"]) for x in completed) / len(completed)
            if completed
            else None
        ),
        "current_successes": current_successes,
        "current_max_goals": current_max_goals,
        "current_all_goals_hit": [
            s >= m for s, m in zip(current_successes, current_max_goals, strict=False)
        ],
        "retract_succeeded": (
            [bool(v) for v in env.retract_succeeded.detach().cpu().tolist()]
            if hasattr(env, "retract_succeeded")
            else None
        ),
    }
    _save_manifest(out_dir, manifest)

    if float(args.hold_open_s) != 0.0 and not args.headless:
        if float(args.hold_open_s) < 0.0:
            print("[render_peg_in_hole_finetuned] holding non-headless window open until Ctrl-C")
            while True:
                app.update()
                time.sleep(0.02)
        else:
            deadline = time.time() + float(args.hold_open_s)
            while time.time() < deadline:
                app.update()
                time.sleep(0.02)

    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
