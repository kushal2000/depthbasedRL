"""Render SimToolReal pretrained policy review images/videos.

This script is intentionally focused on the pretrained-policy visualization
workflow.  It replays an rl_games checkpoint in the SimToolReal training
distribution, captures RGB from a single overview camera, and can recolor each
environment's object so the object pool diversity is easier to see.

Examples:
    python isaacsimenvs/render_simtoolreal_pretrained.py \
        --checkpoint /juno/u/kedia/depthbasedRL/train_dir/TrainingObjective/Play2Win/model.pth \
        --num_envs 16 --env_spacing 1.0 --quality low --capture_png_steps 0,60,180,360

    python isaacsimenvs/render_simtoolreal_pretrained.py \
        --checkpoint /juno/u/kedia/depthbasedRL/pretrained_policy/model.pth \
        --num_envs 16 --env_spacing 1.0 --quality medium --make_video
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


VIDEO_DIR = Path(__file__).resolve().parent / "videos" / "simtoolreal_pretrained_review"

DEFAULT_PLAY2WIN_CHECKPOINT = "/juno/u/kedia/depthbasedRL/train_dir/TrainingObjective/Play2Win/model.pth"
DEFAULT_PRETRAINED_POLICY_CHECKPOINT = "/juno/u/kedia/depthbasedRL/pretrained_policy/model.pth"

QUALITY_PRESETS = {
    # Quality currently controls capture resolution.  Keep render settings
    # conservative until we verify which Isaac/RTX quality knobs are stable in
    # this environment.
    "low": {"width": 960, "height": 540},
    "medium": {"width": 1280, "height": 720},
    "high": {"width": 1920, "height": 1080},
    "ultra": {"width": 2560, "height": 1440},
}

# Muted palette: diverse enough to read in a grid, but avoids bright goal-green.
DIVERSE_PALETTE = (
    (0.25, 0.47, 0.70),  # blue
    (0.86, 0.52, 0.18),  # orange
    (0.55, 0.38, 0.67),  # purple
    (0.77, 0.31, 0.32),  # red
    (0.45, 0.62, 0.30),  # olive
    (0.30, 0.58, 0.62),  # teal
    (0.72, 0.55, 0.28),  # ochre
    (0.50, 0.50, 0.50),  # gray
    (0.62, 0.45, 0.30),  # brown
    (0.36, 0.56, 0.78),  # light blue
    (0.70, 0.44, 0.55),  # rose
    (0.40, 0.64, 0.50),  # muted green
)


def _parse_step_list(value: str) -> list[int]:
    if value.strip() == "":
        return []
    steps = sorted({int(item) for item in value.split(",") if item.strip()})
    if any(step < 0 for step in steps):
        raise argparse.ArgumentTypeError("capture steps must be non-negative")
    return steps


def _parse_handle_head_types(value: str) -> tuple[str, ...]:
    out = tuple(item.strip() for item in value.split(",") if item.strip())
    if not out:
        raise argparse.ArgumentTypeError("at least one object type is required")
    return out


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _checkpoint_slug(checkpoint: str) -> str:
    path = Path(checkpoint)
    if path.parent.name in {"Play2Win", "1000_obj", "pretrained_policy"}:
        return path.parent.name
    if path.parent.parent.name:
        return f"{path.parent.parent.name}_{path.parent.name}"
    return path.stem


def _auto_camera_pose(env, *, z_offset: float = 0.75):
    """Choose a diagonal overview camera that covers the env-origin grid."""
    import torch

    origins = env.scene.env_origins
    mins = torch.min(origins, dim=0).values
    maxs = torch.max(origins, dim=0).values
    center = 0.5 * (mins + maxs)
    span_xy = max(float(maxs[0] - mins[0]), float(maxs[1] - mins[1]), 1.0)
    target = center + torch.tensor([0.0, 0.0, z_offset], device=env.device)

    distance = max(2.4, 1.25 * span_xy)
    height = max(1.5, 0.62 * span_xy)
    eye = target + torch.tensor([-distance, -0.82 * distance, height], device=env.device)
    return eye, target


def _set_record_camera(camera, eye, target) -> None:
    camera.set_world_poses_from_view(eye.unsqueeze(0), target.unsqueeze(0))
    # The caller will step/update the sim after this.  Keep this helper small so
    # camera motion can be extended later without changing capture semantics.


def _camera_eye_for_step(
    base_eye,
    target,
    *,
    progress: float,
    motion: str,
    orbit_deg: float,
    dolly_scale: float,
):
    if motion == "static":
        return base_eye

    import torch

    rel = base_eye - target
    if motion in {"orbit", "orbit_dolly_out"}:
        theta = math.radians(orbit_deg) * progress
        c = math.cos(theta)
        s = math.sin(theta)
        rot = torch.tensor(
            [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
            dtype=rel.dtype,
            device=rel.device,
        )
        rel = rot @ rel
    if motion in {"dolly_out", "orbit_dolly_out"}:
        scale = 1.0 + progress * (dolly_scale - 1.0)
        rel = rel * scale
    return target + rel


def _recolor_objects_by_env() -> dict[str, Any]:
    """Apply display colors to each env's Object prim and return a summary."""
    from pxr import Gf, Usd, UsdGeom, UsdShade
    from isaaclab.sim.utils import find_matching_prim_paths, get_current_stage

    stage = get_current_stage()
    object_paths = sorted(
        find_matching_prim_paths("/World/envs/env_.*/Object"),
        key=lambda p: int(p.rsplit("/", 2)[-2].removeprefix("env_")),
    )
    colored = []
    for idx, root_path in enumerate(object_paths):
        color = DIVERSE_PALETTE[idx % len(DIVERSE_PALETTE)]
        color_vec = Gf.Vec3f(*color)
        root_prim = stage.GetPrimAtPath(root_path)
        if not root_prim.IsValid():
            continue
        gprim_count = 0
        for prim in Usd.PrimRange(root_prim):
            try:
                UsdShade.MaterialBindingAPI(prim).UnbindAllBindings()
            except Exception:
                pass
            if prim.IsA(UsdGeom.Gprim):
                UsdGeom.Gprim(prim).GetDisplayColorAttr().Set([color_vec])
                gprim_count += 1
        colored.append({"object_path": root_path, "color": color, "gprim_count": gprim_count})
    return {"num_colored": len(colored), "objects": colored[:32]}


def _make_camera_cfg(width: int, height: int):
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg

    return CameraCfg(
        prim_path="/World/RecordCamera",
        update_period=0,
        height=height,
        width=width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 100.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 10.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="opengl",
        ),
    )


def _capture_rgb(camera, *, dt: float):
    camera.update(dt)
    rgb = camera.data.output.get("rgb")
    if rgb is None or rgb.shape[0] == 0:
        return None
    return rgb[0].detach().cpu().numpy()[:, :, :3]


def _save_manifest(out_dir: Path, manifest: dict[str, Any]) -> None:
    path = out_dir / "manifest.json"
    with path.open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[render_simtoolreal_pretrained] wrote {path}")


def _apply_training_distribution(env_cfg, args) -> None:
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.scene.env_spacing = args.env_spacing
    env_cfg.assets.num_assets_per_type = args.num_assets_per_type
    env_cfg.assets.handle_head_types = args.handle_head_types


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_PLAY2WIN_CHECKPOINT)
    parser.add_argument("--task", default="Isaacsimenvs-SimToolReal-Direct-v0")
    parser.add_argument("--agent", default="rl_games_sapg_cfg_entry_point")
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--env_spacing", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=360)
    parser.add_argument("--video_fps", type=int, default=30)
    parser.add_argument("--quality", choices=sorted(QUALITY_PRESETS), default="low")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--rl_device", default="cuda:0")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--capture_png_steps",
        type=_parse_step_list,
        default=_parse_step_list("0,60,180,360"),
    )
    parser.add_argument("--make_video", action="store_true")
    parser.add_argument("--camera_eye", type=float, nargs=3, default=None)
    parser.add_argument("--camera_target", type=float, nargs=3, default=None)
    parser.add_argument(
        "--camera_motion",
        choices=("static", "orbit", "dolly_out", "orbit_dolly_out"),
        default="static",
    )
    parser.add_argument("--camera_orbit_deg", type=float, default=10.0)
    parser.add_argument("--camera_dolly_scale", type=float, default=1.12)
    parser.add_argument("--no_recolor_objects", action="store_true")
    parser.add_argument("--num_assets_per_type", type=int, default=100)
    parser.add_argument(
        "--handle_head_types",
        type=_parse_handle_head_types,
        default=_parse_handle_head_types("hammer,screwdriver,marker,spatula,eraser,brush"),
    )
    my_args = parser.parse_args()

    checkpoint = Path(my_args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    quality = QUALITY_PRESETS[my_args.quality]
    width = my_args.width or int(quality["width"])
    height = my_args.height or int(quality["height"])
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = my_args.out_dir
    if out_dir is None:
        out_dir = (
            VIDEO_DIR
            / f"{timestamp}_{_checkpoint_slug(my_args.checkpoint)}"
            / f"n{my_args.num_envs}_spacing{my_args.env_spacing:g}_{my_args.quality}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    from isaaclab.app import AppLauncher

    launcher_parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(launcher_parser)
    launcher_args, _ = launcher_parser.parse_known_args([])
    launcher_args.headless = True
    launcher_args.enable_cameras = True
    app = AppLauncher(launcher_args).app

    import gymnasium as gym
    import imageio
    import torch
    from isaaclab.sensors import Camera
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    import isaacsimenvs  # noqa: F401  triggers gym.register
    from isaacsimenvs.utils.rlgames_utils import register_rlgames_env
    from rl_games.torch_runner import Runner

    torch.manual_seed(my_args.seed)

    env_cfg = load_cfg_from_registry(my_args.task, "env_cfg_entry_point")
    _apply_training_distribution(env_cfg, my_args)
    if hasattr(env_cfg, "seed"):
        env_cfg.seed = my_args.seed

    spec = gym.spec(my_args.task)
    mod_name, cls_name = spec.entry_point.split(":")
    env_cls = getattr(importlib.import_module(mod_name), cls_name)
    env = env_cls(cfg=env_cfg)

    camera = Camera(cfg=_make_camera_cfg(width=width, height=height))
    env.sim.reset()

    recolor_summary = None
    if not my_args.no_recolor_objects:
        recolor_summary = _recolor_objects_by_env()
        print(
            "[render_simtoolreal_pretrained] recolored "
            f"{recolor_summary['num_colored']} object prims"
        )

    if my_args.camera_eye is not None:
        eye = torch.tensor(my_args.camera_eye, device=env.device)
    else:
        eye, _ = _auto_camera_pose(env)
    if my_args.camera_target is not None:
        target = torch.tensor(my_args.camera_target, device=env.device)
    elif my_args.camera_eye is not None:
        _, target = _auto_camera_pose(env)
    else:
        _, target = _auto_camera_pose(env)
    _set_record_camera(camera, eye, target)
    env.sim.step()
    camera.update(0.0)

    print(f"[diag] checkpoint = {checkpoint}")
    print(f"[diag] num_envs = {my_args.num_envs}, env_spacing = {my_args.env_spacing}")
    print(f"[diag] quality = {my_args.quality}, width = {width}, height = {height}")
    print(f"[diag] camera eye = {eye.detach().cpu().tolist()}")
    print(f"[diag] camera target = {target.detach().cpu().tolist()}")
    print(f"[diag] camera pos_w actual = {camera.data.pos_w[0].detach().cpu().tolist()}")
    print(f"[diag] camera quat_w actual = {camera.data.quat_w_world[0].detach().cpu().tolist()}")

    agent_cfg = load_cfg_from_registry(my_args.task, my_args.agent)
    clip_obs = float(agent_cfg["params"]["env"].get("clip_observations", math.inf))
    clip_actions = float(agent_cfg["params"]["env"].get("clip_actions", math.inf))
    wrapped = register_rlgames_env(
        env,
        rl_device=my_args.rl_device,
        clip_obs=clip_obs,
        clip_actions=clip_actions,
    )
    agent_cfg["params"]["config"]["device"] = my_args.rl_device
    agent_cfg["params"]["config"]["device_name"] = my_args.rl_device
    agent_cfg["params"]["seed"] = my_args.seed

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
    capture_every = max(1, round((1.0 / my_args.video_fps) / policy_dt))
    capture_steps = set(my_args.capture_png_steps)

    manifest: dict[str, Any] = {
        "script": str(Path(__file__).resolve()),
        "git_commit": _git_commit(),
        "checkpoint": str(checkpoint),
        "task": my_args.task,
        "agent": my_args.agent,
        "seed": my_args.seed,
        "num_envs": my_args.num_envs,
        "env_spacing": my_args.env_spacing,
        "num_assets_per_type": my_args.num_assets_per_type,
        "handle_head_types": list(my_args.handle_head_types),
        "quality": my_args.quality,
        "width": width,
        "height": height,
        "steps": my_args.steps,
        "video_fps": my_args.video_fps,
        "capture_every": capture_every,
        "capture_png_steps": sorted(capture_steps),
        "make_video": my_args.make_video,
        "camera_eye": eye.detach().cpu().tolist(),
        "camera_target": target.detach().cpu().tolist(),
        "camera_motion": my_args.camera_motion,
        "camera_orbit_deg": my_args.camera_orbit_deg,
        "camera_dolly_scale": my_args.camera_dolly_scale,
        "recolor_summary": recolor_summary,
        "outputs": {"pngs": [], "video": None},
    }

    frames = []

    def capture(step_i: int, *, for_video: bool) -> None:
        progress = 0.0 if my_args.steps <= 0 else min(1.0, max(0.0, step_i / my_args.steps))
        capture_eye = _camera_eye_for_step(
            eye,
            target,
            progress=progress,
            motion=my_args.camera_motion,
            orbit_deg=my_args.camera_orbit_deg,
            dolly_scale=my_args.camera_dolly_scale,
        )
        _set_record_camera(camera, capture_eye, target)
        frame = _capture_rgb(camera, dt=policy_dt)
        if frame is None:
            print(f"[warning] no RGB frame at step {step_i}")
            return
        if step_i in capture_steps:
            png_path = out_dir / f"step_{step_i:04d}.png"
            imageio.imwrite(str(png_path), frame)
            manifest["outputs"]["pngs"].append(str(png_path))
            print(f"[render_simtoolreal_pretrained] wrote {png_path}")
        if for_video:
            frames.append(frame)

    print(
        "[render_simtoolreal_pretrained] rolling out "
        f"{my_args.steps} policy steps on {my_args.num_envs} envs"
    )
    capture(0, for_video=my_args.make_video)
    for step_i in range(1, my_args.steps + 1):
        action = player.get_action(obs, is_deterministic=my_args.deterministic)
        obs, rew, dones, infos = player.env_step(wrapped, action)
        if step_i in capture_steps or (my_args.make_video and step_i % capture_every == 0):
            capture(step_i, for_video=my_args.make_video and step_i % capture_every == 0)

    if my_args.make_video:
        video_path = out_dir / "rollout.mp4"
        imageio.mimwrite(str(video_path), frames, fps=my_args.video_fps)
        manifest["outputs"]["video"] = str(video_path)
        print(f"[render_simtoolreal_pretrained] wrote {len(frames)} frames to {video_path}")

    _save_manifest(out_dir, manifest)

    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
