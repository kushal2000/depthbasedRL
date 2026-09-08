#!/usr/bin/env python3
"""Render one high-quality Isaac Sim teacher rollout for a rebuttal figure."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-overrides", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task", default="Isaacsimenvs-PegInHole-Direct-v0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Render several seeds in one simulator session; --output is then a directory.",
    )
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument(
        "--record-from-step",
        type=int,
        default=0,
        help="Run the policy from reset but begin recording at this policy step.",
    )
    parser.add_argument("--rl-device", default="cuda:0")
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument(
        "--capture-every",
        type=int,
        default=2,
        help="Capture every N 60 Hz policy steps; 2 produces real-time 30 fps video.",
    )
    parser.add_argument("--video-quality", type=int, default=9)
    parser.add_argument("--tail-hold-frames", type=int, default=15)
    parser.add_argument(
        "--samples-per-pixel",
        type=int,
        default=64,
        help="RTX samples per pixel. Use 8 for camera/seed previews and 64 for final renders.",
    )
    parser.add_argument(
        "--camera-eye",
        nargs=3,
        type=float,
        default=(-1.4, 0.0, 0.85),
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument(
        "--camera-target",
        nargs=3,
        type=float,
        default=(0.05, 0.0, 0.58),
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument("--focal-length", type=float, default=24.0)
    parser.add_argument("--horizontal-aperture", type=float, default=20.955)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional env override applied after the saved training config.",
    )
    return parser.parse_args()


def _reseed(seed: int) -> None:
    import numpy as np
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def _policy_obs(obs):
    """Extract the state-policy observation from either rl_games wrapper."""
    import torch

    if isinstance(obs, tuple):
        obs = obs[0]
    if isinstance(obs, torch.Tensor):
        return obs
    if isinstance(obs, dict):
        for key in ("teacher", "teacher_obs"):
            if key in obs:
                return obs[key]
        inner = obs.get("obs")
        if isinstance(inner, torch.Tensor):
            return inner
        if isinstance(inner, dict):
            for key in ("teacher", "teacher_obs", "policy"):
                if key in inner:
                    return inner[key]
    raise RuntimeError(f"Could not extract policy observation from {type(obs)}")


def _rgb_frame(camera):
    rgb = camera.data.output.get("rgb")
    if rgb is None or rgb.shape[0] == 0:
        raise RuntimeError("Record camera returned no RGB frame")
    return rgb[0, :, :, :3].detach().cpu().numpy().copy()


def _hide_goal_visualization() -> int:
    """Hide policy target meshes so the video contains only physical objects."""
    from isaaclab.sim.utils import find_matching_prim_paths, get_current_stage
    from pxr import UsdGeom

    stage = get_current_stage()
    hidden = 0
    for prim_path in find_matching_prim_paths("/World/envs/env_.*/GoalViz"):
        prim = stage.GetPrimAtPath(prim_path)
        if prim.IsValid():
            UsdGeom.Imageable(prim).MakeInvisible()
            hidden += 1
    return hidden


def _episode_state(env, step: int) -> dict[str, object]:
    """Record enough state to reproduce and select semantic figure frames."""
    origin = env.scene.env_origins[0]

    def local(values):
        return [float(value) for value in (values[0] - origin).detach().cpu().tolist()]

    def values(tensor):
        return [float(value) for value in tensor[0].detach().cpu().tolist()]

    return {
        "step": int(step),
        "object_position": local(env.object.data.root_pos_w),
        "object_quaternion_wxyz": values(env.object.data.root_quat_w),
        "fixture_position": local(env.hole.data.root_pos_w),
        "fixture_quaternion_wxyz": values(env.hole.data.root_quat_w),
        "goals_reached": int(env._successes[0].item()),
        "retract_phase": bool(env.retract_phase[0].item()),
    }


def main() -> None:
    args = _parse_args()
    output = Path(args.output).resolve()
    if args.seeds:
        output.mkdir(parents=True, exist_ok=True)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

    from isaaclab.app import AppLauncher

    launcher_parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(launcher_parser)
    launcher_args, _ = launcher_parser.parse_known_args([])
    launcher_args.headless = True
    launcher_args.enable_cameras = True
    app = AppLauncher(launcher_args).app

    import imageio.v2 as imageio
    import torch
    import yaml
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import Camera, CameraCfg

    import isaacsimenvs  # noqa: F401  Registers the gym task.
    from isaacsimenvs.dagger.teacher import Teacher
    from isaacsimenvs.utils.rlgames_utils import (
        register_rlgames_env,
        teacher_env_info,
    )
    from peg_in_hole_dynamic.eval_isaacsim import (
        _configure_agent,
        _instantiate_env,
        _load_env_cfg,
        _set_attr_path,
    )
    from peg_in_hole_dynamic.offline_eval_teacher_robustness import (
        _load_train_overrides,
    )

    checkpoint = Path(args.checkpoint).resolve()
    train_overrides_path = Path(args.train_overrides).resolve()
    cfg = _load_env_cfg(args.task)
    for key, value in _load_train_overrides(str(train_overrides_path)).items():
        _set_attr_path(cfg, key, value)

    cfg.scene.num_envs = 1
    cfg.sim.device = str(args.sim_device)

    # The reference play2perfect video used this quality overlay.
    cfg.sim.render.rendering_mode = "quality"
    cfg.sim.render.antialiasing_mode = "DLAA"
    cfg.sim.render.enable_translucency = True
    cfg.sim.render.enable_reflections = True
    cfg.sim.render.enable_global_illumination = True
    cfg.sim.render.enable_direct_lighting = True
    cfg.sim.render.enable_shadows = True
    cfg.sim.render.enable_ambient_occlusion = True
    cfg.sim.render.enable_dl_denoiser = True
    cfg.sim.render.samples_per_pixel = int(args.samples_per_pixel)

    # Rebuttal rollout protocol: preserve all saved training-time variation
    # except the external force/torque wrench explicitly disabled by the user.
    cli_overrides: dict[str, object] = {
        "env.domain_randomization.force_scale": 0.0,
        "env.domain_randomization.torque_scale": 0.0,
    }
    for item in args.override:
        key, separator, value = item.partition("=")
        if not separator:
            raise SystemExit(f"--override must be KEY=VALUE, got {item!r}")
        cli_overrides[key] = yaml.safe_load(value)
    for key, value in cli_overrides.items():
        _set_attr_path(cfg, key, value)

    print(
        f"=> problem={cfg.peg_in_hole.problem} goal_mode={cfg.peg_in_hole.goal_mode} "
        f"force_scale={cfg.domain_randomization.force_scale} "
        f"torque_scale={cfg.domain_randomization.torque_scale}",
        flush=True,
    )

    env = _instantiate_env(args.task, cfg)
    print(f"=> hid {_hide_goal_visualization()} GoalViz prims", flush=True)

    camera_cfg = CameraCfg(
        prim_path="/World/RebuttalRecordCamera",
        update_period=0,
        height=int(args.height),
        width=int(args.width),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=float(args.focal_length),
            focus_distance=400.0,
            horizontal_aperture=float(args.horizontal_aperture),
            clipping_range=(0.1, 100.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 10.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="opengl",
        ),
    )
    camera = Camera(cfg=camera_cfg)
    env.sim.reset()

    origin = env.scene.env_origins[0]
    eye = origin + torch.tensor(args.camera_eye, device=env.device)
    target = origin + torch.tensor(args.camera_target, device=env.device)
    camera.set_world_poses_from_view(eye.unsqueeze(0), target.unsqueeze(0))
    env.sim.step()
    camera.update(0.0)

    agent_cfg = _configure_agent(
        args.task,
        "rl_games_sapg_cfg_entry_point",
        rl_device=args.rl_device,
        num_envs=1,
        deterministic=True,
        games=1,
        extra_overrides={},
    )
    clip_obs = float(agent_cfg["params"]["env"].get("clip_observations", math.inf))
    clip_actions = float(agent_cfg["params"]["env"].get("clip_actions", math.inf))
    wrapped = register_rlgames_env(
        env,
        rl_device=args.rl_device,
        clip_obs=clip_obs,
        clip_actions=clip_actions,
    )
    teacher = Teacher(
        task_id=args.task,
        agent_key="rl_games_sapg_cfg_entry_point",
        checkpoint_path=checkpoint,
        num_envs=1,
        rl_device=args.rl_device,
        env_info=teacher_env_info(wrapped),
    )

    physics_dt = float(env.sim.get_physics_dt())
    policy_dt = physics_dt * int(getattr(env.cfg, "decimation", 1) or 1)
    capture_every = max(1, int(args.capture_every))

    def render_one(seed: int, video_path: Path) -> None:
        # Match the validated offline-eval reset sequence exactly.
        _reseed(seed)
        teacher.reset()
        wrapped.reset()
        _reseed(seed)
        obs = wrapped.reset()

        frames = []
        frame_states = []
        record_from_step = max(0, int(args.record_from_step))
        if record_from_step == 0:
            camera.update(policy_dt)
            frames.append(_rgb_frame(camera))
            frame_states.append(_episode_state(env, step=0))
        termination_reasons = {
            key: False
            for key in ("max_successes", "fall", "dropped", "hand_far", "timeout")
        }
        steps_used = 0
        for step in range(int(args.max_steps)):
            action = teacher.get_action(_policy_obs(obs))
            step_out = wrapped.step(action)
            obs = step_out[0] if isinstance(step_out, tuple) else step_out
            steps_used = step + 1

            done = bool((env.reset_terminated | env.reset_time_outs)[0].item())
            if done:
                for key in termination_reasons:
                    termination_reasons[key] = bool(
                        env._termination_reasons[key][0].item()
                    )
                break

            should_capture = (
                steps_used >= record_from_step
                and (steps_used - record_from_step) % capture_every == 0
            )
            if should_capture:
                camera.update(capture_every * policy_dt)
                frames.append(_rgb_frame(camera))
                frame_states.append(_episode_state(env, step=steps_used))

        if args.tail_hold_frames > 0 and frames:
            frames.extend([frames[-1].copy() for _ in range(int(args.tail_hold_frames))])
            frame_states.extend(
                [dict(frame_states[-1]) for _ in range(int(args.tail_hold_frames))]
            )

        first_frame = video_path.with_name(f"{video_path.stem}_first.png")
        last_frame = video_path.with_name(f"{video_path.stem}_last.png")
        metadata_path = video_path.with_suffix(".json")
        imageio.imwrite(first_frame, frames[0])
        imageio.imwrite(last_frame, frames[-1])
        imageio.mimwrite(
            video_path,
            frames,
            fps=int(args.video_fps),
            codec="libx264",
            quality=int(args.video_quality),
            macro_block_size=2,
            ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        )

        metadata = {
            "checkpoint": str(checkpoint),
            "checkpoint_epoch": None,
            "train_overrides": str(train_overrides_path),
            "task": args.task,
            "problem": str(cfg.peg_in_hole.problem),
            "goal_mode": str(cfg.peg_in_hole.goal_mode),
            "seed": int(seed),
            "wrench": {
                "force_scale": float(cfg.domain_randomization.force_scale),
                "torque_scale": float(cfg.domain_randomization.torque_scale),
            },
            "camera": {
                "eye": [float(value) for value in args.camera_eye],
                "target": [float(value) for value in args.camera_target],
                "focal_length": float(args.focal_length),
                "horizontal_aperture": float(args.horizontal_aperture),
                "resolution": [int(args.width), int(args.height)],
            },
            "video": {
                "path": str(video_path),
                "fps": int(args.video_fps),
                "capture_every": capture_every,
                "record_from_step": record_from_step,
                "frames": len(frames),
                "first_frame": str(first_frame),
                "last_frame": str(last_frame),
            },
            "episode": {
                "steps": steps_used,
                "terminated": any(termination_reasons.values()),
                "success": termination_reasons["max_successes"],
                "termination_reasons": termination_reasons,
                "initial_state": frame_states[0],
                "frame_states": frame_states,
            },
            "render": {
                "mode": "quality",
                "antialiasing": "DLAA",
                "samples_per_pixel": int(args.samples_per_pixel),
            },
        }
        with metadata_path.open("w") as file:
            json.dump(metadata, file, indent=2)

        print(
            f"=> seed={seed} steps={steps_used} "
            f"success={termination_reasons['max_successes']} reasons={termination_reasons}",
            flush=True,
        )
        print(f"=> wrote video {video_path} ({len(frames)} frames)", flush=True)
        print(f"=> wrote metadata {metadata_path}", flush=True)

    seeds = args.seeds if args.seeds else [args.seed]
    for seed in seeds:
        video_path = output / f"plug_seed_{seed:02d}.mp4" if args.seeds else output
        render_one(int(seed), video_path)
    sys.stdout.flush()
    sys.stderr.flush()
    del app
    os._exit(0)


if __name__ == "__main__":
    main()
