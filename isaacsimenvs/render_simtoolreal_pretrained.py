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


def _rectangular_env_origins(
    num_envs: int,
    x_spacing: float,
    y_spacing: float,
    *,
    device,
    grid_cols: int | None = None,
):
    import torch

    cols = int(grid_cols or math.ceil(math.sqrt(num_envs)))
    if cols <= 0:
        raise ValueError(f"grid_cols must be positive, got {grid_cols}")
    rows = int(math.ceil(num_envs / cols))
    origins = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)
    for env_id in range(num_envs):
        row = env_id // cols
        col = env_id % cols
        origins[env_id, 0] = (col - 0.5 * (cols - 1)) * x_spacing
        origins[env_id, 1] = (row - 0.5 * (rows - 1)) * y_spacing
    return origins


def _apply_rectangular_env_layout(
    env,
    x_spacing: float,
    y_spacing: float,
    *,
    grid_cols: int | None,
) -> dict[str, Any]:
    """Move env root prims to a rectangular grid and keep env_origins consistent."""
    from pxr import Gf, UsdGeom
    from isaaclab.sim.utils import get_current_stage

    new_origins = _rectangular_env_origins(
        env.num_envs,
        x_spacing,
        y_spacing,
        device=env.device,
        grid_cols=grid_cols,
    )
    old_origins = env.scene.env_origins.clone()
    stage = get_current_stage()
    moved = 0
    for env_id, env_path in enumerate(env.scene.env_prim_paths):
        prim = stage.GetPrimAtPath(env_path)
        if not prim.IsValid():
            continue
        xformable = UsdGeom.Xformable(prim)
        xformable.ClearXformOpOrder()
        translate_op = xformable.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(*[float(v) for v in new_origins[env_id].detach().cpu().tolist()]))
        moved += 1
    env.scene.env_origins.copy_(new_origins)
    return {
        "requested_x_spacing": x_spacing,
        "requested_y_spacing": y_spacing,
        "requested_grid_cols": grid_cols,
        "actual_grid_cols": int(grid_cols or math.ceil(math.sqrt(env.num_envs))),
        "actual_grid_rows": int(math.ceil(env.num_envs / int(grid_cols or math.ceil(math.sqrt(env.num_envs))))),
        "moved_env_roots": moved,
        "old_min": old_origins.min(dim=0).values.detach().cpu().tolist(),
        "old_max": old_origins.max(dim=0).values.detach().cpu().tolist(),
        "new_min": new_origins.min(dim=0).values.detach().cpu().tolist(),
        "new_max": new_origins.max(dim=0).values.detach().cpu().tolist(),
    }


def _set_record_camera(camera, eye, target) -> None:
    camera.set_world_poses_from_view(eye.unsqueeze(0), target.unsqueeze(0))
    # The caller will step/update the sim after this.  Keep this helper small so
    # camera motion can be extended later without changing capture semantics.


def _camera_pose_for_step(
    env,
    base_eye,
    target,
    *,
    progress: float,
    motion: str,
    orbit_deg: float,
    dolly_scale: float,
    grid_cols: int | None,
    env_spacing_xy: list[float] | tuple[float, float] | None,
    sapg_ref_anchor_env: int,
    sapg_ref_start_target: list[float] | tuple[float, float, float],
    sapg_ref_start_eye_offset: list[float] | tuple[float, float, float],
    sapg_ref_end_target_grid_scale: list[float] | tuple[float, float, float],
    sapg_ref_end_eye_grid_scale: list[float] | tuple[float, float, float],
):
    if motion == "static":
        return base_eye, target

    import torch

    if motion == "sapg_ref_pan":
        cols = int(grid_cols or math.ceil(math.sqrt(env.num_envs)))
        rows = int(math.ceil(env.num_envs / cols))
        if env_spacing_xy is None:
            x_spacing = y_spacing = float(env.cfg.scene.env_spacing)
        else:
            x_spacing = float(env_spacing_xy[0])
            y_spacing = float(env_spacing_xy[1])

        anchor_env = env.num_envs + sapg_ref_anchor_env if sapg_ref_anchor_env < 0 else sapg_ref_anchor_env
        if not 0 <= anchor_env < env.num_envs:
            raise ValueError(f"sapg_ref_anchor_env={sapg_ref_anchor_env} resolves to invalid env {anchor_env}")

        # Defaults match simtoolreal_private/origin/2026-02-18_video_2:
        # viewer_camera_look_at(..., envs[num_envs - 1], local_pos, local_target).
        anchor = env.scene.env_origins[anchor_env]
        start_target = anchor + torch.tensor(sapg_ref_start_target, dtype=anchor.dtype, device=anchor.device)
        start_eye = start_target + torch.tensor(sapg_ref_start_eye_offset, dtype=anchor.dtype, device=anchor.device)

        grid_width = max(0.0, (cols - 1) * x_spacing)
        grid_depth = max(0.0, (rows - 1) * y_spacing)
        t = progress
        t_smooth = t * t * t * (t * (6.0 * t - 15.0) + 10.0)

        end_target_scale = torch.tensor(
            sapg_ref_end_target_grid_scale,
            dtype=anchor.dtype,
            device=anchor.device,
        )
        end_eye_scale = torch.tensor(
            sapg_ref_end_eye_grid_scale,
            dtype=anchor.dtype,
            device=anchor.device,
        )
        grid_delta = torch.tensor([grid_width, grid_depth, 1.0], dtype=anchor.dtype, device=anchor.device)
        end_target = start_target + end_target_scale * grid_delta
        end_eye = start_eye + end_eye_scale * grid_delta
        return start_eye + t_smooth * (end_eye - start_eye), start_target + t_smooth * (end_target - start_target)

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
    return target + rel, target


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


def _style_goal_viz(color: tuple[float, float, float], opacity: float) -> dict[str, Any]:
    """Style GoalViz as a translucent target distinct from the live object."""
    from pxr import Gf, Usd, UsdGeom, UsdShade
    from isaaclab.sim.utils import find_matching_prim_paths, get_current_stage

    stage = get_current_stage()
    goal_paths = sorted(
        find_matching_prim_paths("/World/envs/env_.*/GoalViz"),
        key=lambda p: int(p.rsplit("/", 2)[-2].removeprefix("env_")),
    )
    color_vec = Gf.Vec3f(*color)
    styled = []
    for root_path in goal_paths:
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
                gprim = UsdGeom.Gprim(prim)
                gprim.GetDisplayColorAttr().Set([color_vec])
                gprim.GetDisplayOpacityAttr().Set([float(opacity)])
                gprim_count += 1
        styled.append({"goal_path": root_path, "color": color, "opacity": opacity, "gprim_count": gprim_count})
    return {"num_styled": len(styled), "goals": styled[:32]}


def _style_prim_trees(
    pattern: str,
    *,
    color: tuple[float, float, float],
    opacity: float = 1.0,
    visible: bool = True,
    label: str,
) -> dict[str, Any]:
    """Set simple display styling on all Gprims under prims matching ``pattern``."""
    from pxr import Gf, Usd, UsdGeom, UsdShade
    from isaaclab.sim.utils import find_matching_prim_paths, get_current_stage

    stage = get_current_stage()
    paths = sorted(find_matching_prim_paths(pattern))
    color_vec = Gf.Vec3f(*color)
    styled = []
    for root_path in paths:
        root_prim = stage.GetPrimAtPath(root_path)
        if not root_prim.IsValid():
            continue
        imageable = UsdGeom.Imageable(root_prim)
        if visible:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()
        gprim_count = 0
        for prim in Usd.PrimRange(root_prim):
            try:
                UsdShade.MaterialBindingAPI(prim).UnbindAllBindings()
            except Exception:
                pass
            if prim.IsA(UsdGeom.Gprim):
                gprim = UsdGeom.Gprim(prim)
                gprim.GetDisplayColorAttr().Set([color_vec])
                gprim.GetDisplayOpacityAttr().Set([float(opacity)])
                gprim_count += 1
        styled.append(
            {
                "path": root_path,
                "color": color,
                "opacity": opacity,
                "visible": visible,
                "gprim_count": gprim_count,
            }
        )
    return {"label": label, "pattern": pattern, "num_styled": len(styled), "items": styled[:32]}


def _apply_marble_tile_floor(
    *,
    base_color: tuple[float, float, float],
    tile_count: int,
    tile_size: float,
    tile_gap: float,
    tile_thickness: float = 0.006,
) -> dict[str, Any]:
    """Add a render-only tiled stone/marble floor above the physics ground."""
    import random

    from pxr import Gf, UsdGeom, UsdShade
    from isaaclab.sim.utils import get_current_stage

    stage = get_current_stage()
    root_path = "/World/CinematicFloor"
    root = UsdGeom.Xform.Define(stage, root_path)

    rng = random.Random(17)
    half = 0.5 * (tile_count - 1)
    effective_size = max(0.01, float(tile_size) - float(tile_gap))
    top_z = 0.0015
    center_z = top_z - 0.5 * tile_thickness

    tiles = []
    for row in range(tile_count):
        for col in range(tile_count):
            prim_path = f"{root_path}/Tile_{row:02d}_{col:02d}"
            cube = UsdGeom.Cube.Define(stage, prim_path)
            cube.CreateSizeAttr(1.0)
            xform = UsdGeom.Xformable(cube.GetPrim())
            xform.AddTranslateOp().Set(
                (
                    (col - half) * float(tile_size),
                    (row - half) * float(tile_size),
                    center_z,
                )
            )
            # With Cube.size=1, xform scale is the full visual edge length.
            # Keep gaps small so the floor reads as continuous stone, not patches.
            xform.AddScaleOp().Set((effective_size, effective_size, tile_thickness))

            # Slight per-tile variation is enough to read as stone at cinematic distance.
            noise = rng.uniform(-0.045, 0.035)
            warm = rng.uniform(-0.012, 0.018)
            color = (
                min(1.0, max(0.0, base_color[0] + noise + warm)),
                min(1.0, max(0.0, base_color[1] + noise + warm)),
                min(1.0, max(0.0, base_color[2] + noise)),
            )
            gprim = UsdGeom.Gprim(cube.GetPrim())
            gprim.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
            gprim.GetDisplayOpacityAttr().Set([1.0])
            try:
                UsdShade.MaterialBindingAPI(cube.GetPrim()).UnbindAllBindings()
            except Exception:
                pass
            if len(tiles) < 16:
                tiles.append({"path": prim_path, "color": color})

    return {
        "style": "marble_tiles",
        "root_path": root_path,
        "tile_count": tile_count,
        "tile_size": tile_size,
        "tile_gap": tile_gap,
        "num_tiles": tile_count * tile_count,
        "sample_tiles": tiles,
    }


def _apply_beauty_render_settings(env_cfg, *, preset: str, samples_per_pixel: int) -> dict[str, Any]:
    """Apply render-only quality settings to the env config before construction."""
    if preset == "default":
        return {"preset": preset, "applied": False}

    render_cfg = getattr(getattr(env_cfg, "sim", None), "render", None)
    if render_cfg is None:
        return {"preset": preset, "applied": False, "reason": "env_cfg.sim.render missing"}

    settings = {
        "rendering_mode": "quality",
        "antialiasing_mode": "DLAA",
        "enable_translucency": True,
        "enable_reflections": True,
        "enable_global_illumination": True,
        "enable_direct_lighting": True,
        "enable_shadows": True,
        "enable_ambient_occlusion": True,
        "enable_dl_denoiser": True,
        "samples_per_pixel": int(samples_per_pixel),
    }
    applied = {}
    for key, value in settings.items():
        if hasattr(render_cfg, key):
            setattr(render_cfg, key, value)
            applied[key] = value
    return {"preset": preset, "applied": True, "settings": applied}


def _apply_beauty_lighting() -> dict[str, Any]:
    """Add soft global/key/fill lighting for presentation renders."""
    import isaaclab.sim as sim_utils

    spawned = []
    dome = sim_utils.DomeLightCfg(intensity=900.0, color=(0.95, 0.97, 1.0), exposure=0.0)
    dome.func("/World/BeautyDomeLight", dome)
    spawned.append({"path": "/World/BeautyDomeLight", "type": "DomeLight", "intensity": 900.0})

    key = sim_utils.SphereLightCfg(radius=3.0, intensity=11000.0, color=(1.0, 0.86, 0.68), exposure=0.0)
    key.func("/World/KeyLight", key, translation=(-3.0, -4.0, 6.0))
    spawned.append({"path": "/World/KeyLight", "type": "SphereLight", "translation": [-3.0, -4.0, 6.0]})

    fill = sim_utils.SphereLightCfg(radius=5.0, intensity=1800.0, color=(0.72, 0.82, 1.0), exposure=0.0)
    fill.func("/World/FillLight", fill, translation=(4.0, 3.0, 4.0))
    spawned.append({"path": "/World/FillLight", "type": "SphereLight", "translation": [4.0, 3.0, 4.0]})
    return {"spawned": spawned}


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
    if args.env_spacing_xy is not None:
        env_cfg.scene.env_spacing = max(args.env_spacing_xy)
    else:
        env_cfg.scene.env_spacing = args.env_spacing
    if args.robot_urdf is not None:
        robot_urdf = Path(args.robot_urdf).expanduser()
        if not robot_urdf.exists():
            raise FileNotFoundError(f"Robot URDF override does not exist: {robot_urdf}")
        env_cfg.assets.robot_urdf = str(robot_urdf.resolve())
    env_cfg.assets.num_assets_per_type = args.num_assets_per_type
    env_cfg.assets.handle_head_types = args.handle_head_types
    env_cfg.assets.object_distribution_mode = args.object_distribution_mode
    if args.reset_position_noise_m is not None:
        x_noise, y_noise, z_noise = args.reset_position_noise_m
        env_cfg.reset.reset_position_noise_x = float(x_noise)
        env_cfg.reset.reset_position_noise_y = float(y_noise)
        env_cfg.reset.reset_position_noise_z = float(z_noise)
    if args.reset_orientation_mode is not None:
        env_cfg.reset.reset_orientation_mode = args.reset_orientation_mode
    if args.reset_yaw_noise_deg is not None:
        env_cfg.reset.reset_orientation_yaw_range_deg = float(args.reset_yaw_noise_deg)
    if args.reset_axis_angle_noise_deg is not None:
        env_cfg.reset.reset_orientation_axis_angle_range_deg = float(args.reset_axis_angle_noise_deg)


def _tensor_to_list(value) -> list[float]:
    if value is None:
        return []
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().float().cpu().reshape(-1).tolist()
    except Exception:
        pass
    try:
        import numpy as np

        return np.asarray(value, dtype=float).reshape(-1).tolist()
    except Exception:
        return []


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _summarize_metric_values(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": float(sum(values) / len(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_PLAY2WIN_CHECKPOINT)
    parser.add_argument("--task", default="Isaacsimenvs-SimToolReal-Direct-v0")
    parser.add_argument("--agent", default="rl_games_sapg_cfg_entry_point")
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--env_spacing", type=float, default=1.2)
    parser.add_argument("--env_spacing_xy", type=float, nargs=2, default=None)
    parser.add_argument("--grid_cols", type=int, default=None)
    parser.add_argument("--steps", type=int, default=360)
    parser.add_argument("--video_fps", type=int, default=30)
    parser.add_argument("--quality", choices=sorted(QUALITY_PRESETS), default="low")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument(
        "--render_quality_preset",
        choices=("default", "beauty"),
        default="default",
        help="Render settings preset. 'beauty' enables quality mode, DLAA, GI, shadows, AO, reflections, denoiser.",
    )
    parser.add_argument("--render_samples_per_pixel", type=int, default=64)
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument(
        "--no_timestamp_out_dir",
        action="store_true",
        help="Write directly to --out_dir instead of creating a timestamped subdirectory.",
    )
    parser.add_argument("--rl_device", default="cuda:0")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--capture_png_steps",
        type=_parse_step_list,
        default=_parse_step_list("0,60,180,360"),
    )
    parser.add_argument("--make_video", action="store_true")
    parser.add_argument(
        "--metrics_interval",
        type=int,
        default=60,
        help="Print rollout metrics every N policy steps. Set <=0 to disable periodic prints.",
    )
    parser.add_argument("--camera_eye", type=float, nargs=3, default=None)
    parser.add_argument("--camera_target", type=float, nargs=3, default=None)
    parser.add_argument(
        "--camera_motion",
        choices=("static", "orbit", "dolly_out", "orbit_dolly_out", "sapg_ref_pan"),
        default="static",
    )
    parser.add_argument("--camera_orbit_deg", type=float, default=10.0)
    parser.add_argument("--camera_dolly_scale", type=float, default=1.12)
    parser.add_argument(
        "--camera_render_warmup_frames",
        type=int,
        default=1,
        help="Extra render calls after camera pose updates before RGB readback. Helps avoid stale/ghosted frames.",
    )
    parser.add_argument("--sapg_ref_anchor_env", type=int, default=-1)
    parser.add_argument("--sapg_ref_start_target", type=float, nargs=3, default=(0.0, 0.0, 0.63))
    parser.add_argument("--sapg_ref_start_eye_offset", type=float, nargs=3, default=(-0.25, -0.5, 0.35))
    parser.add_argument("--sapg_ref_end_target_grid_scale", type=float, nargs=3, default=(-0.5, -0.5, 0.0))
    parser.add_argument("--sapg_ref_end_eye_grid_scale", type=float, nargs=3, default=(-0.8, -1.2, 0.5))
    parser.add_argument("--no_recolor_objects", action="store_true")
    parser.add_argument("--no_style_goal_viz", action="store_true")
    parser.add_argument("--hide_goal_viz", action="store_true")
    parser.add_argument("--style_table", action="store_true")
    parser.add_argument("--style_floor", action="store_true")
    parser.add_argument(
        "--floor_style",
        choices=("display_color", "marble_tiles"),
        default="display_color",
        help="Floor visual style used when --style_floor is set.",
    )
    parser.add_argument("--beauty_lighting", action="store_true")
    parser.add_argument("--goal_color", type=float, nargs=3, default=(0.55, 1.0, 0.55))
    parser.add_argument("--goal_opacity", type=float, default=0.35)
    parser.add_argument("--table_color", type=float, nargs=3, default=(0.72, 0.50, 0.30))
    parser.add_argument("--floor_color", type=float, nargs=3, default=(0.60, 0.60, 0.56))
    parser.add_argument("--floor_tile_count", type=int, default=24)
    parser.add_argument("--floor_tile_size", type=float, default=0.9)
    parser.add_argument("--floor_tile_gap", type=float, default=0.012)
    parser.add_argument(
        "--robot_urdf",
        type=Path,
        default=None,
        help=(
            "Optional robot URDF override. Useful for cinematic visual-only URDFs "
            "that preserve the same kinematics/collisions."
        ),
    )
    parser.add_argument("--num_assets_per_type", type=int, default=100)
    parser.add_argument(
        "--object_distribution_mode",
        choices=("training", "mixed_training_simple_25_25_50"),
        default="training",
    )
    parser.add_argument(
        "--handle_head_types",
        type=_parse_handle_head_types,
        default=_parse_handle_head_types("hammer,screwdriver,marker,spatula,eraser,brush"),
    )
    parser.add_argument(
        "--reset_position_noise_m",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help=(
            "Override object reset position half-widths in meters. "
            "Default env training distribution is 0.1 0.1 0.02."
        ),
    )
    parser.add_argument(
        "--reset_orientation_mode",
        choices=("full", "identity", "yaw", "axis_angle"),
        default=None,
        help="Override object reset orientation distribution. Default env training distribution is full SO(3).",
    )
    parser.add_argument(
        "--reset_yaw_noise_deg",
        type=float,
        default=None,
        help="Yaw half-width for --reset_orientation_mode yaw.",
    )
    parser.add_argument(
        "--reset_axis_angle_noise_deg",
        type=float,
        default=None,
        help="Angle half-width for --reset_orientation_mode axis_angle.",
    )
    my_args = parser.parse_args()

    checkpoint = Path(my_args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    quality = QUALITY_PRESETS[my_args.quality]
    width = my_args.width or int(quality["width"])
    height = my_args.height or int(quality["height"])
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    requested_out_dir = my_args.out_dir
    if requested_out_dir is None:
        out_dir = (
            VIDEO_DIR
            / f"{timestamp}_{_checkpoint_slug(my_args.checkpoint)}"
            / f"n{my_args.num_envs}_spacing{my_args.env_spacing:g}_{my_args.quality}"
        )
    elif my_args.no_timestamp_out_dir:
        out_dir = requested_out_dir
    else:
        out_dir = requested_out_dir.parent / f"{timestamp}_{requested_out_dir.name}"
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
    from isaacsimenvs.tasks.simtoolreal.utils.obs_utils import compute_intermediate_values
    from isaacsimenvs.utils.rlgames_utils import register_rlgames_env
    from rl_games.torch_runner import Runner

    torch.manual_seed(my_args.seed)

    env_cfg = load_cfg_from_registry(my_args.task, "env_cfg_entry_point")
    _apply_training_distribution(env_cfg, my_args)
    render_quality_summary = _apply_beauty_render_settings(
        env_cfg,
        preset=my_args.render_quality_preset,
        samples_per_pixel=my_args.render_samples_per_pixel,
    )
    if hasattr(env_cfg, "seed"):
        env_cfg.seed = my_args.seed

    spec = gym.spec(my_args.task)
    mod_name, cls_name = spec.entry_point.split(":")
    env_cls = getattr(importlib.import_module(mod_name), cls_name)
    env = env_cls(cfg=env_cfg)
    rectangular_layout_summary = None
    if my_args.env_spacing_xy is not None:
        rectangular_layout_summary = _apply_rectangular_env_layout(
            env,
            x_spacing=float(my_args.env_spacing_xy[0]),
            y_spacing=float(my_args.env_spacing_xy[1]),
            grid_cols=my_args.grid_cols,
        )
        print(f"[diag] rectangular env layout = {rectangular_layout_summary}")

    camera = Camera(cfg=_make_camera_cfg(width=width, height=height))
    env.sim.reset()

    recolor_summary = None
    if not my_args.no_recolor_objects:
        recolor_summary = _recolor_objects_by_env()
        print(
            "[render_simtoolreal_pretrained] recolored "
            f"{recolor_summary['num_colored']} object prims"
        )
    table_style_summary = None
    if my_args.style_table:
        table_style_summary = _style_prim_trees(
            "/World/envs/env_.*/Table",
            color=tuple(float(v) for v in my_args.table_color),
            opacity=1.0,
            visible=True,
            label="table",
        )
        print(
            "[render_simtoolreal_pretrained] styled "
            f"{table_style_summary['num_styled']} table prims"
        )
    floor_style_summary = None
    if my_args.style_floor:
        if my_args.floor_style == "marble_tiles":
            # Keep the original physics ground, but make it a neutral base below
            # the render-only cinematic tile overlay.
            base_floor_summary = _style_prim_trees(
                "/World/ground",
                color=(0.55, 0.56, 0.54),
                opacity=1.0,
                visible=True,
                label="floor_base",
            )
            tile_floor_summary = _apply_marble_tile_floor(
                base_color=tuple(float(v) for v in my_args.floor_color),
                tile_count=int(my_args.floor_tile_count),
                tile_size=float(my_args.floor_tile_size),
                tile_gap=float(my_args.floor_tile_gap),
            )
            floor_style_summary = {
                "label": "floor",
                "style": "marble_tiles",
                "base": base_floor_summary,
                "tiles": tile_floor_summary,
            }
        else:
            floor_style_summary = _style_prim_trees(
                "/World/ground",
                color=tuple(float(v) for v in my_args.floor_color),
                opacity=1.0,
                visible=True,
                label="floor",
            )
        print(
            "[render_simtoolreal_pretrained] styled "
            f"floor using {my_args.floor_style}"
        )
    goal_style_summary = None
    if my_args.hide_goal_viz:
        goal_style_summary = _style_prim_trees(
            "/World/envs/env_.*/GoalViz",
            color=tuple(float(v) for v in my_args.goal_color),
            opacity=0.0,
            visible=False,
            label="goal_viz",
        )
        print(
            "[render_simtoolreal_pretrained] hid "
            f"{goal_style_summary['num_styled']} GoalViz prims"
        )
    elif not my_args.no_style_goal_viz:
        goal_style_summary = _style_goal_viz(tuple(float(v) for v in my_args.goal_color), my_args.goal_opacity)
        print(
            "[render_simtoolreal_pretrained] styled "
            f"{goal_style_summary['num_styled']} GoalViz prims"
        )
    lighting_summary = None
    if my_args.beauty_lighting:
        lighting_summary = _apply_beauty_lighting()
        print(f"[render_simtoolreal_pretrained] applied beauty lighting: {lighting_summary}")

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
    if my_args.env_spacing_xy is not None:
        print(f"[diag] env_spacing_xy = {my_args.env_spacing_xy}")
        print(f"[diag] grid_cols = {my_args.grid_cols}")
        print(f"[diag] scene env_spacing = {max(my_args.env_spacing_xy)}")
    print(f"[diag] quality = {my_args.quality}, width = {width}, height = {height}")
    print(f"[diag] render_quality = {render_quality_summary}")
    print(f"[diag] robot_urdf = {env_cfg.assets.robot_urdf}")
    print(f"[diag] camera eye = {eye.detach().cpu().tolist()}")
    print(f"[diag] camera target = {target.detach().cpu().tolist()}")
    print(f"[diag] camera pos_w actual = {camera.data.pos_w[0].detach().cpu().tolist()}")
    print(f"[diag] camera quat_w actual = {camera.data.quat_w_world[0].detach().cpu().tolist()}")
    print(
        "[diag] object reset noise = "
        f"xyz_m=({env_cfg.reset.reset_position_noise_x:g}, "
        f"{env_cfg.reset.reset_position_noise_y:g}, "
        f"{env_cfg.reset.reset_position_noise_z:g}) "
        f"orientation_mode={env_cfg.reset.reset_orientation_mode} "
        f"yaw_deg={env_cfg.reset.reset_orientation_yaw_range_deg:g} "
        f"axis_angle_deg={env_cfg.reset.reset_orientation_axis_angle_range_deg:g}"
    )

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
        "scene_env_spacing": max(my_args.env_spacing_xy) if my_args.env_spacing_xy is not None else my_args.env_spacing,
        "env_spacing_xy": my_args.env_spacing_xy,
        "grid_cols": my_args.grid_cols,
        "rectangular_layout_summary": rectangular_layout_summary,
        "robot_urdf": str(env_cfg.assets.robot_urdf),
        "num_assets_per_type": my_args.num_assets_per_type,
        "object_distribution_mode": my_args.object_distribution_mode,
        "handle_head_types": list(my_args.handle_head_types),
        "reset": {
            "reset_position_noise_m": [
                float(env_cfg.reset.reset_position_noise_x),
                float(env_cfg.reset.reset_position_noise_y),
                float(env_cfg.reset.reset_position_noise_z),
            ],
            "reset_orientation_mode": str(env_cfg.reset.reset_orientation_mode),
            "reset_orientation_yaw_range_deg": float(env_cfg.reset.reset_orientation_yaw_range_deg),
            "reset_orientation_axis_angle_range_deg": float(
                env_cfg.reset.reset_orientation_axis_angle_range_deg
            ),
        },
        "quality": my_args.quality,
        "width": width,
        "height": height,
        "render_quality_preset": my_args.render_quality_preset,
        "render_samples_per_pixel": my_args.render_samples_per_pixel,
        "render_quality_summary": render_quality_summary,
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
        "camera_render_warmup_frames": my_args.camera_render_warmup_frames,
        "sapg_ref_pan": {
            "source_repo": "/home/tylerlum/github_repos/simtoolreal_private",
            "source_branch": "origin/2026-02-18_video_2",
            "source_commit": "495fdeeaa10af7aefed3edc58c7428179899fcb9",
            "anchor_env": my_args.sapg_ref_anchor_env,
            "local_start_target": list(my_args.sapg_ref_start_target),
            "local_start_eye_offset": list(my_args.sapg_ref_start_eye_offset),
            "end_target_grid_scale": list(my_args.sapg_ref_end_target_grid_scale),
            "end_eye_grid_scale": list(my_args.sapg_ref_end_eye_grid_scale),
            "easing": "quintic_smoothstep",
        },
        "recolor_summary": recolor_summary,
        "table_style_summary": table_style_summary,
        "floor_style_summary": floor_style_summary,
        "goal_style_summary": goal_style_summary,
        "lighting_summary": lighting_summary,
        "hide_goal_viz": bool(my_args.hide_goal_viz),
        "goal_color": list(my_args.goal_color),
        "goal_opacity": float(my_args.goal_opacity),
        "table_color": list(my_args.table_color),
        "floor_style": my_args.floor_style,
        "floor_color": list(my_args.floor_color),
        "floor_tile_count": int(my_args.floor_tile_count),
        "floor_tile_size": float(my_args.floor_tile_size),
        "floor_tile_gap": float(my_args.floor_tile_gap),
        "requested_out_dir": str(requested_out_dir) if requested_out_dir is not None else None,
        "effective_out_dir": str(out_dir),
        "outputs": {"pngs": [], "video": None},
        "timings": {"captures": [], "video_write_ms": None},
        "metrics": None,
    }

    frames = []
    metric_history: list[dict[str, float | int | None]] = []
    episode_final_values: dict[str, list[float]] = {}
    done_reason_counts: dict[str, int] = {}
    completed_episode_count = 0
    previous_successes = None
    max_successes_seen = None
    cumulative_goal_hits = 0

    def record_metrics(step_i: int, *, dones=None, infos=None) -> None:
        nonlocal completed_episode_count, previous_successes, max_successes_seen, cumulative_goal_hits

        successes_long = env._successes.detach().long()
        successes = successes_long.float()
        if previous_successes is None:
            previous_successes = successes_long.clone()
            max_successes_seen = successes_long.clone()
        else:
            cumulative_goal_hits += int(torch.clamp(successes_long - previous_successes, min=0).sum().item())
            previous_successes = successes_long.clone()
            max_successes_seen = torch.maximum(max_successes_seen, successes_long)
        lifted = env._lifted_object.detach().float()
        keypoint = env._keypoints_max_dist.detach().float()
        fingertip = env._curr_fingertip_distances.detach().float()
        sample = {
            "step": step_i,
            "successes_mean": float(successes.mean().item()),
            "successes_min": float(successes.min().item()),
            "successes_max": float(successes.max().item()),
            "lifted_frac": float(lifted.mean().item()),
            "keypoint_max_dist_mean": float(keypoint.mean().item()),
            "keypoint_max_dist_min": float(keypoint.min().item()),
            "keypoint_max_dist_max": float(keypoint.max().item()),
            "fingertip_dist_mean": float(fingertip.mean().item()),
        }
        metric_history.append(sample)

        done_idx: list[int] = []
        if dones is not None:
            done_values = _tensor_to_list(dones)
            done_idx = [idx for idx, value in enumerate(done_values) if value > 0.0]
            completed_episode_count += len(done_idx)

        if infos and isinstance(infos, dict) and done_idx:
            episode_final = infos.get("episode_final") or {}
            for key, value in episode_final.items():
                values = _tensor_to_list(value)
                if not values:
                    continue
                if len(values) == env.num_envs:
                    selected = [values[idx] for idx in done_idx]
                else:
                    selected = values
                episode_final_values.setdefault(key, []).extend(selected)
                if key.startswith("done_"):
                    done_reason_counts[key.removeprefix("done_")] = (
                        done_reason_counts.get(key.removeprefix("done_"), 0)
                        + int(round(sum(selected)))
                    )

        if my_args.metrics_interval > 0 and (
            step_i == 0 or step_i == my_args.steps or step_i % my_args.metrics_interval == 0
        ):
            print(
                "[metrics] "
                f"step={step_i} "
                f"successes_mean/max={sample['successes_mean']:.2f}/{sample['successes_max']:.0f} "
                f"lifted={100.0 * sample['lifted_frac']:.1f}% "
                f"kp_dist_mean/min={sample['keypoint_max_dist_mean']:.4f}/{sample['keypoint_max_dist_min']:.4f} "
                f"completed_eps={completed_episode_count}",
                flush=True,
            )

    def finalize_metrics() -> dict[str, Any]:
        final = metric_history[-1] if metric_history else {}
        return {
            "completed_episode_count": completed_episode_count,
            "done_reason_counts": done_reason_counts,
            "episode_final": {
                key: _summarize_metric_values(values)
                for key, values in sorted(episode_final_values.items())
            },
            "final_step": final,
            "cumulative_goal_hits": cumulative_goal_hits,
            "per_env_final_successes": (
                env._successes.detach().long().cpu().tolist()
                if hasattr(env, "_successes")
                else None
            ),
            "per_env_max_successes_seen": (
                max_successes_seen.detach().long().cpu().tolist()
                if max_successes_seen is not None
                else None
            ),
            "history": metric_history,
        }

    def capture(step_i: int, *, for_video: bool) -> None:
        capture_t0 = time.perf_counter()
        progress = 0.0 if my_args.steps <= 0 else min(1.0, max(0.0, step_i / my_args.steps))
        capture_eye, capture_target = _camera_pose_for_step(
            env,
            eye,
            target,
            progress=progress,
            motion=my_args.camera_motion,
            orbit_deg=my_args.camera_orbit_deg,
            dolly_scale=my_args.camera_dolly_scale,
            grid_cols=my_args.grid_cols,
            env_spacing_xy=my_args.env_spacing_xy,
            sapg_ref_anchor_env=my_args.sapg_ref_anchor_env,
            sapg_ref_start_target=my_args.sapg_ref_start_target,
            sapg_ref_start_eye_offset=my_args.sapg_ref_start_eye_offset,
            sapg_ref_end_target_grid_scale=my_args.sapg_ref_end_target_grid_scale,
            sapg_ref_end_eye_grid_scale=my_args.sapg_ref_end_eye_grid_scale,
        )
        _set_record_camera(camera, capture_eye, capture_target)
        # Flush camera pose through Hydra before readback. Without this, the
        # first frame after a camera move can use the previous camera pose.
        for _ in range(max(1, my_args.camera_render_warmup_frames)):
            env.sim.render()
        frame = _capture_rgb(camera, dt=policy_dt)
        render_ms = (time.perf_counter() - capture_t0) * 1000.0
        if frame is None:
            print(f"[warning] no RGB frame at step {step_i}")
            return
        write_png_ms = None
        if step_i in capture_steps:
            png_path = out_dir / f"step_{step_i:04d}.png"
            write_t0 = time.perf_counter()
            imageio.imwrite(str(png_path), frame)
            write_png_ms = (time.perf_counter() - write_t0) * 1000.0
            manifest["outputs"]["pngs"].append(str(png_path))
            print(f"[render_simtoolreal_pretrained] wrote {png_path}")
        if for_video:
            frames.append(frame)
        manifest["timings"]["captures"].append(
            {
                "step": step_i,
                "for_video": for_video,
                "render_ms": render_ms,
                "write_png_ms": write_png_ms,
                "height": int(frame.shape[0]),
                "width": int(frame.shape[1]),
            }
        )
        timing_log_every = max(1, capture_every * my_args.video_fps)
        if write_png_ms is not None or step_i % timing_log_every == 0:
            print(
                "[timing] capture "
                f"step={step_i} render_ms={render_ms:.2f}"
                + (f" write_png_ms={write_png_ms:.2f}" if write_png_ms is not None else "")
            )

    print(
        "[render_simtoolreal_pretrained] rolling out "
        f"{my_args.steps} policy steps on {my_args.num_envs} envs"
    )
    # DirectRLEnv computes geometry caches during env.step(). For a zero-step
    # smoke render, force one cache update so step-0 metrics are meaningful.
    compute_intermediate_values(env)
    record_metrics(0)
    capture(0, for_video=my_args.make_video)
    for step_i in range(1, my_args.steps + 1):
        action = player.get_action(obs, is_deterministic=my_args.deterministic)
        obs, rew, dones, infos = player.env_step(wrapped, action)
        record_metrics(step_i, dones=dones, infos=infos)
        if step_i in capture_steps or (my_args.make_video and step_i % capture_every == 0):
            capture(step_i, for_video=my_args.make_video and step_i % capture_every == 0)

    if my_args.make_video:
        video_path = out_dir / "rollout.mp4"
        video_t0 = time.perf_counter()
        imageio.mimwrite(str(video_path), frames, fps=my_args.video_fps, macro_block_size=1)
        manifest["timings"]["video_write_ms"] = (time.perf_counter() - video_t0) * 1000.0
        manifest["outputs"]["video"] = str(video_path)
        print(f"[render_simtoolreal_pretrained] wrote {len(frames)} frames to {video_path}")

    manifest["metrics"] = finalize_metrics()
    _save_manifest(out_dir, manifest)

    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
