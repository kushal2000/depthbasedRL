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
import colorsys
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
REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PLAY2WIN_CHECKPOINT = "/juno/u/kedia/depthbasedRL/train_dir/TrainingObjective/Play2Win/model.pth"
DEFAULT_PRETRAINED_POLICY_CHECKPOINT = "/juno/u/kedia/depthbasedRL/pretrained_policy/model.pth"
DEFAULT_HDRI_STINSON_BEACH = (
    "/home/tylerlum/github_repos/depthbasedRL/.venv-isaacsim-py311/lib/python3.11/site-packages/"
    "isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311/bin/usd/hdx/resources/textures/"
    "StinsonBeach.hdr"
)
DEFAULT_HDRI_PHOTO_STUDIO = (
    "/home/tylerlum/github_repos/depthbasedRL/.venv-isaacsim-py311/lib/python3.11/site-packages/"
    "isaacsim/extscache/omni.kit.widget.material_preview-1.0.16/data/photo_studio_01_4k.hdr"
)
ROBOLAB_BACKGROUND_DIR = Path("/home/tylerlum/github_repos/RoboLab/assets/backgrounds")
ISAAC_SAMPLE_MARBLE_TEXTURE = "Isaac/Samples/DR/Materials/Textures/marble_tile.png"
DEFAULT_DYNAMIC_CLEAR_SKY = (
    "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Skies/Dynamic/ClearSky.usd"
)
DEFAULT_LOCAL_DYNAMIC_SIMPLE_SKY = (
    REPO_ROOT
    / ".venv-isaacsim-py311/lib/python3.11/site-packages/isaacsim/extscache/"
    / "omni.kit.environment.core-1.3.24/data/tests/Skies/Dynamic/simple.usd"
)
DEFAULT_LOCAL_DYNAMIC_SUNSTUDY_SKY = (
    REPO_ROOT
    / ".venv-isaacsim-py311/lib/python3.11/site-packages/isaacsim/extscache/"
    / "omni.kit.environment.core-1.3.24/data/tests/Skies/Dynamic/sunstudy.usd"
)
DEFAULT_WHITE_STONE_TEXTURE = REPO_ROOT / "assets/textures/cinematic_white_stone_slab.png"
DEFAULT_WARM_LIMESTONE_TEXTURE = REPO_ROOT / "assets/textures/cinematic_warm_limestone_slab.png"
DEFAULT_WARM_LIMESTONE_NORMAL = REPO_ROOT / "assets/textures/cinematic_warm_limestone_slab_normal.png"
DEFAULT_COOL_CONCRETE_TEXTURE = REPO_ROOT / "assets/textures/cinematic_cool_concrete_slab.png"
DEFAULT_COOL_CONCRETE_NORMAL = REPO_ROOT / "assets/textures/cinematic_cool_concrete_slab_normal.png"
DEFAULT_SOFT_CONCRETE_TEXTURE = REPO_ROOT / "assets/textures/cinematic_soft_concrete_slab.png"
DEFAULT_SOFT_CONCRETE_NORMAL = REPO_ROOT / "assets/textures/cinematic_soft_concrete_slab_normal.png"
DEFAULT_MATTE_WARM_GRAY_TEXTURE = REPO_ROOT / "assets/textures/cinematic_matte_warm_gray_slab.png"
DEFAULT_MATTE_WARM_GRAY_NORMAL = REPO_ROOT / "assets/textures/cinematic_matte_warm_gray_slab_normal.png"
DEFAULT_MATTE_SLATE_TEXTURE = REPO_ROOT / "assets/textures/cinematic_matte_slate_slab.png"
DEFAULT_MATTE_SLATE_NORMAL = REPO_ROOT / "assets/textures/cinematic_matte_slate_slab_normal.png"
DEFAULT_MATTE_GREIGE_TEXTURE = REPO_ROOT / "assets/textures/cinematic_matte_greige_slab.png"
DEFAULT_MATTE_GREIGE_NORMAL = REPO_ROOT / "assets/textures/cinematic_matte_greige_slab_normal.png"
DEFAULT_NVIDIA_PRECAST_CONCRETE_MDL = (
    REPO_ROOT
    / ".venv-isaacsim-py311/lib/python3.11/site-packages/isaacsim/extscache/"
    / "omni.kit.tool.collect-2.2.18+69cbf6ad/data/test_stages/OM_55150/1/Materials/"
    / "vMaterials_2/Concrete/Concrete_Precast.mdl"
)
DEFAULT_NVIDIA_PRECAST_CONCRETE_TEXTURE = (
    REPO_ROOT
    / ".venv-isaacsim-py311/lib/python3.11/site-packages/isaacsim/extscache/"
    / "omni.kit.tool.collect-2.2.18+69cbf6ad/data/test_stages/OM_55150/1/Materials/"
    / "vMaterials_2/Concrete/textures/precastconcrete_diff.png"
)
DEFAULT_NVIDIA_PRECAST_CONCRETE_NORMAL = (
    REPO_ROOT
    / ".venv-isaacsim-py311/lib/python3.11/site-packages/isaacsim/extscache/"
    / "omni.kit.tool.collect-2.2.18+69cbf6ad/data/test_stages/OM_55150/1/Materials/"
    / "vMaterials_2/Concrete/textures/precastconcrete_norm.jpg"
)
DEFAULT_FIELDSTONE_TEXTURE = (
    REPO_ROOT
    / ".venv-isaacsim-py311/lib/python3.11/site-packages/isaacsim/extscache/"
    / "omni.asset_validator.core-1.1.6/omni/asset_validator/core/tests/data/Materials/"
    / "Fieldstone/Fieldstone_BaseColor.png"
)
DEFAULT_FIELDSTONE_NORMAL = (
    REPO_ROOT
    / ".venv-isaacsim-py311/lib/python3.11/site-packages/isaacsim/extscache/"
    / "omni.asset_validator.core-1.1.6/omni/asset_validator/core/tests/data/Materials/"
    / "Fieldstone/Fieldstone_N.png"
)
ROBOLAB_OAK_DIR = Path("/home/tylerlum/github_repos/RoboLab/assets/materials/Base/Wood/Oak")
DEFAULT_OAK_BASE_COLOR = ROBOLAB_OAK_DIR / "Oak_BaseColor.png"
DEFAULT_OAK_NORMAL = ROBOLAB_OAK_DIR / "Oak_N.png"
DEFAULT_OAK_ORM = ROBOLAB_OAK_DIR / "Oak_ORM.png"

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


def _set_active_viewport_camera(camera_path: str) -> bool:
    try:
        from omni.kit.viewport.utility import get_active_viewport
        from pxr import Sdf

        viewport = get_active_viewport()
        if viewport is None:
            return False
        viewport.camera_path = Sdf.Path(camera_path)
        return True
    except Exception:
        return False


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


def _adjust_rgb_saturation_value(
    color: tuple[float, float, float],
    *,
    saturation: float,
    value_scale: float,
) -> tuple[float, float, float]:
    h, s, v = colorsys.rgb_to_hsv(*[max(0.0, min(1.0, float(c))) for c in color])
    s = max(0.0, min(1.0, s * float(saturation)))
    v = max(0.0, min(1.0, v * float(value_scale)))
    return tuple(float(c) for c in colorsys.hsv_to_rgb(h, s, v))


def _recolor_objects_by_env(*, saturation: float = 1.0, value_scale: float = 1.0) -> dict[str, Any]:
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
        raw_color = DIVERSE_PALETTE[idx % len(DIVERSE_PALETTE)]
        color = _adjust_rgb_saturation_value(
            raw_color,
            saturation=float(saturation),
            value_scale=float(value_scale),
        )
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
        colored.append(
            {
                "object_path": root_path,
                "raw_color": raw_color,
                "color": color,
                "gprim_count": gprim_count,
            }
        )
    return {
        "num_colored": len(colored),
        "saturation": float(saturation),
        "value_scale": float(value_scale),
        "objects": colored[:32],
    }


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


def _bind_material_to_prim_trees(pattern: str, material_prim, *, label: str) -> dict[str, Any]:
    """Bind a material to every Gprim under prims matching ``pattern``."""
    from pxr import Usd, UsdGeom, UsdShade
    from isaaclab.sim.utils import find_matching_prim_paths, get_current_stage

    stage = get_current_stage()
    material = UsdShade.Material(material_prim)
    paths = sorted(find_matching_prim_paths(pattern))
    styled = []
    for root_path in paths:
        root_prim = stage.GetPrimAtPath(root_path)
        if not root_prim.IsValid():
            continue
        gprim_count = 0
        for prim in Usd.PrimRange(root_prim):
            if prim.IsA(UsdGeom.Gprim):
                UsdShade.MaterialBindingAPI(prim).Bind(
                    material,
                    UsdShade.Tokens.strongerThanDescendants,
                )
                gprim_count += 1
        styled.append({"path": root_path, "gprim_count": gprim_count})
    return {
        "label": label,
        "pattern": pattern,
        "num_styled": len(styled),
        "material_path": str(material_prim.GetPath()),
        "items": styled[:32],
    }


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


def _resolve_hdri_path(preset: str, explicit_path: str | None) -> str | None:
    """Resolve a local HDRI/EXR path for a visible dome light."""
    if explicit_path:
        return str(Path(explicit_path).expanduser())

    candidates_by_preset = {
        "stinson_beach": [DEFAULT_HDRI_STINSON_BEACH],
        "photo_studio": [
            DEFAULT_HDRI_PHOTO_STUDIO,
            str(ROBOLAB_BACKGROUND_DIR / "indoors/photo_studio_01_2k.hdr"),
            str(ROBOLAB_BACKGROUND_DIR / "default/brown_photostudio.hdr"),
        ],
        "cape_hill": [str(ROBOLAB_BACKGROUND_DIR / "indoors/cape_hill_2k.hdr")],
        "old_outdoor_theater": [str(ROBOLAB_BACKGROUND_DIR / "indoors/old_outdoor_theater_2k.hdr")],
        "cloudy_vondelpark": [
            str(ROBOLAB_BACKGROUND_DIR / "outdoors/cloudy_vondelpark_2k.hdr"),
            str(ROBOLAB_BACKGROUND_DIR / "outdoors/cloudy_vondelpark_2k.png"),
        ],
        "wasteland_clouds": [
            str(ROBOLAB_BACKGROUND_DIR / "outdoors/wasteland_clouds_2k.hdr"),
            str(ROBOLAB_BACKGROUND_DIR / "outdoors/wasteland_clouds_2k.png"),
        ],
        "winter_sky": [
            str(ROBOLAB_BACKGROUND_DIR / "outdoors/winter_sky_2k.hdr"),
            str(ROBOLAB_BACKGROUND_DIR / "outdoors/winter_sky_2k.png"),
        ],
        "kloofendal_partly_cloudy": [
            str(ROBOLAB_BACKGROUND_DIR / "outdoors/kloofendal_48d_partly_cloudy_2k.hdr"),
            str(ROBOLAB_BACKGROUND_DIR / "outdoors/kloofendal_48d_partly_cloudy_2k.png"),
        ],
        "sunset_fairway": [
            str(ROBOLAB_BACKGROUND_DIR / "outdoors/sunset_fairway_2k.hdr"),
            str(ROBOLAB_BACKGROUND_DIR / "outdoors/sunset_fairway_2k.png"),
        ],
    }
    for candidate in candidates_by_preset.get(preset, []):
        if Path(candidate).is_file():
            return candidate
    return None


def _resolve_dynamic_sky_path(preset: str, explicit_path: str | None) -> str:
    if explicit_path:
        return str(Path(explicit_path).expanduser())
    candidates_by_preset = {
        "remote_clear": str(DEFAULT_DYNAMIC_CLEAR_SKY),
        "local_simple": str(DEFAULT_LOCAL_DYNAMIC_SIMPLE_SKY),
        "local_sunstudy": str(DEFAULT_LOCAL_DYNAMIC_SUNSTUDY_SKY),
    }
    if preset not in candidates_by_preset:
        raise ValueError(f"Unsupported dynamic sky preset: {preset}")
    return candidates_by_preset[preset]


def _resolve_isaac_asset(asset_relpath: str) -> str | None:
    """Resolve an Isaac sample asset under the configured Isaac asset root."""
    import carb

    settings = carb.settings.get_settings()
    root = settings.get("/persistent/isaac/asset_root/default")
    if not root:
        return None
    return f"{str(root).rstrip('/')}/{asset_relpath.lstrip('/')}"


def _create_omnipbr_material(
    *,
    material_name: str,
    diffuse_texture: str | None,
    diffuse_color: tuple[float, float, float],
    normal_texture: str | None = None,
    normal_texture_influence: float = 0.35,
    texture_scale: tuple[float, float] | None = None,
    roughness: float = 0.38,
    specular_level: float = 0.5,
) -> tuple[Any, dict[str, Any]]:
    """Create an OmniPBR material and set a small set of robust shader inputs."""
    from pxr import Gf, Sdf, UsdShade
    from isaaclab.sim.utils import get_current_stage

    stage = get_current_stage()
    material_path = f"/World/Looks/{material_name}"
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/Shader")
    shader_prim = shader.GetPrim()
    shader_prim.CreateAttribute("info:implementationSource", Sdf.ValueTypeNames.Token).Set("sourceAsset")
    shader_prim.CreateAttribute("info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset).Set("OmniPBR.mdl")
    shader_prim.CreateAttribute("info:mdl:sourceAsset:subIdentifier", Sdf.ValueTypeNames.Token).Set("OmniPBR")
    shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    material.CreateDisplacementOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    material.CreateVolumeOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")

    inputs = {
        "diffuse_color_constant": list(diffuse_color),
        "reflection_roughness_constant": float(roughness),
        "metallic_constant": 0.0,
        "specular_level": float(specular_level),
    }
    shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*diffuse_color))
    shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(float(roughness))
    shader.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("specular_level", Sdf.ValueTypeNames.Float).Set(float(specular_level))
    if diffuse_texture:
        shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(diffuse_texture)
        inputs["diffuse_texture"] = diffuse_texture
    if normal_texture:
        shader.CreateInput("normalmap_texture", Sdf.ValueTypeNames.Asset).Set(normal_texture)
        inputs["normalmap_texture"] = normal_texture
        inputs["normalmap_texture_influence"] = float(normal_texture_influence)
    if texture_scale is not None:
        shader.CreateInput("project_uvw", Sdf.ValueTypeNames.Bool).Set(True)
        shader.CreateInput("texture_scale", Sdf.ValueTypeNames.Float2).Set(
            Gf.Vec2f(float(texture_scale[0]), float(texture_scale[1]))
        )
        inputs["project_uvw"] = True
        inputs["texture_scale"] = [float(texture_scale[0]), float(texture_scale[1])]

    return material.GetPrim(), {"path": material_path, "inputs": inputs}


def _apply_pbr_table_material(
    *,
    base_color_texture: str,
    normal_texture: str | None,
    orm_texture: str | None,
    texture_scale: float,
) -> dict[str, Any]:
    """Bind a real wood PBR material to all table prims."""
    material_prim, material_summary = _create_omnipbr_material(
        material_name="CinematicOakTable",
        diffuse_texture=base_color_texture,
        normal_texture=normal_texture,
        diffuse_color=(0.76, 0.52, 0.32),
        texture_scale=(float(texture_scale), float(texture_scale)),
        roughness=0.72,
    )
    bind_summary = _bind_material_to_prim_trees(
        "/World/envs/env_.*/Table",
        material_prim,
        label="table",
    )
    bind_summary["style"] = "oak_pbr"
    bind_summary["material"] = material_summary
    bind_summary["orm_texture"] = orm_texture
    return bind_summary


def _create_mdl_file_material(
    *,
    material_name: str,
    mdl_path: str,
    mdl_material_name: str,
    project_uvw: bool | None = True,
    texture_scale: tuple[float, float] | None = None,
    albedo_brightness: float | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Create an MDL file material, allowing explicit exported material selection."""
    from omni.usd.commands import CreateMdlMaterialPrimCommand
    from isaaclab.sim.utils import get_current_stage

    stage = get_current_stage()
    material_path = f"/World/Looks/{material_name}"
    CreateMdlMaterialPrimCommand(
        mtl_url=str(mdl_path),
        mtl_name=mdl_material_name,
        mtl_path=material_path,
        stage=stage,
        select_new_prim=False,
    ).do()
    shader_prim = stage.GetPrimAtPath(f"{material_path}/Shader")
    inputs: dict[str, Any] = {"mdl_path": str(mdl_path), "mdl_material_name": mdl_material_name}
    if project_uvw is not None:
        from pxr import Sdf

        shader_prim.CreateAttribute("inputs:project_uvw", Sdf.ValueTypeNames.Bool).Set(bool(project_uvw))
        inputs["project_uvw"] = bool(project_uvw)
    if texture_scale is not None:
        from pxr import Gf, Sdf

        shader_prim.CreateAttribute("inputs:texture_scale", Sdf.ValueTypeNames.Float2).Set(
            Gf.Vec2f(float(texture_scale[0]), float(texture_scale[1]))
        )
        inputs["texture_scale"] = [float(texture_scale[0]), float(texture_scale[1])]
    if albedo_brightness is not None:
        from pxr import Sdf

        shader_prim.CreateAttribute("inputs:albedo_brightness", Sdf.ValueTypeNames.Float).Set(
            float(albedo_brightness)
        )
        inputs["albedo_brightness"] = float(albedo_brightness)
    return stage.GetPrimAtPath(material_path), {"path": material_path, "inputs": inputs}


def _apply_pbr_tile_floor(
    *,
    base_color: tuple[float, float, float],
    tile_count: int,
    tile_size: float,
    tile_gap: float,
    texture_path: str | None,
    normal_texture_path: str | None,
    texture_scale: float,
    roughness: float,
    normal_texture_influence: float,
    specular_level: float,
    material_name: str = "CinematicPbrFloorMaterial",
    style_name: str = "pbr_tiles",
    tile_thickness: float = 0.006,
    mdl_material_name: str | None = None,
) -> dict[str, Any]:
    """Add a render-only tiled floor using a real OmniPBR material."""
    import random

    from pxr import Gf, UsdGeom, UsdShade
    from isaaclab.sim.utils import get_current_stage

    stage = get_current_stage()
    root_path = "/World/CinematicPbrFloor"
    UsdGeom.Xform.Define(stage, root_path)

    if mdl_material_name is not None:
        material_prim, material_summary = _create_mdl_file_material(
            material_name=material_name,
            mdl_path=str(texture_path),
            mdl_material_name=mdl_material_name,
            project_uvw=None,
            texture_scale=(float(texture_scale), float(texture_scale)),
        )
    else:
        material_prim, material_summary = _create_omnipbr_material(
            material_name=material_name,
            diffuse_texture=texture_path,
            normal_texture=normal_texture_path,
            normal_texture_influence=float(normal_texture_influence),
            diffuse_color=base_color,
            texture_scale=(float(texture_scale), float(texture_scale)),
            roughness=float(roughness),
            specular_level=float(specular_level),
        )
    material = UsdShade.Material(material_prim)

    rng = random.Random(19)
    half = 0.5 * (tile_count - 1)
    effective_size = max(0.01, float(tile_size) - float(tile_gap))
    top_z = 0.0018
    center_z = top_z - 0.5 * tile_thickness

    sample_tiles = []
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
            xform.AddScaleOp().Set((effective_size, effective_size, tile_thickness))
            # Preserve slight tile-level variation even with texture so large
            # floor areas do not read as one repeated flat plane.
            color_scale = rng.uniform(0.92, 1.04)
            gprim = UsdGeom.Gprim(cube.GetPrim())
            gprim.GetDisplayColorAttr().Set(
                [
                    Gf.Vec3f(
                        min(1.0, base_color[0] * color_scale),
                        min(1.0, base_color[1] * color_scale),
                        min(1.0, base_color[2] * color_scale),
                    )
                ]
            )
            UsdShade.MaterialBindingAPI(cube.GetPrim()).Bind(
                material,
                UsdShade.Tokens.strongerThanDescendants,
            )
            if len(sample_tiles) < 16:
                sample_tiles.append({"path": prim_path})

    return {
        "style": style_name,
        "root_path": root_path,
        "tile_count": int(tile_count),
        "tile_size": float(tile_size),
        "tile_gap": float(tile_gap),
        "num_tiles": int(tile_count) * int(tile_count),
        "texture_path": texture_path,
        "normal_texture_path": normal_texture_path,
        "texture_scale": float(texture_scale),
        "roughness": float(roughness),
        "normal_texture_influence": float(normal_texture_influence),
        "specular_level": float(specular_level),
        "material": material_summary,
        "sample_tiles": sample_tiles,
    }


def _apply_beauty_render_settings(
    env_cfg,
    *,
    preset: str,
    samples_per_pixel: int,
    dome_light_upper_lower_strategy: int | None,
) -> dict[str, Any]:
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
    if dome_light_upper_lower_strategy is not None:
        settings["dome_light_upper_lower_strategy"] = int(dome_light_upper_lower_strategy)
    applied = {}
    for key, value in settings.items():
        if hasattr(render_cfg, key):
            setattr(render_cfg, key, value)
            applied[key] = value
    return {"preset": preset, "applied": True, "settings": applied}


def _apply_runtime_render_settings(
    *,
    preset: str,
    render_mode: str,
    samples_per_pixel: int,
    dome_light_upper_lower_strategy: int | None,
) -> dict[str, Any]:
    """Apply RTX settings that live in carb settings rather than env_cfg.sim.render."""
    import carb

    settings = carb.settings.get_settings()
    applied: dict[str, Any] = {"preset": preset, "requested_render_mode": render_mode}
    effective_mode = render_mode
    if effective_mode == "auto":
        effective_mode = "pt" if preset == "beauty" else "default"

    if effective_mode != "default":
        settings.set("/rtx/rendermode", effective_mode)
        applied["/rtx/rendermode"] = effective_mode

    if preset == "beauty":
        spp = int(samples_per_pixel)
        runtime_settings = {
            "/rtx/pathtracing/spp": spp,
            "/rtx/pathtracing/totalSpp": spp,
            "/rtx/pathtracing/maxBounces": 8,
            "/rtx/pathtracing/maxSpecularAndTransmissionBounces": 4,
            "/rtx/pathtracing/maxVolumeBounces": 2,
            "/rtx/pathtracing/optixDenoiser/enabled": True,
            "/rtx/pathtracing/clampSpp": 0,
        }
        if dome_light_upper_lower_strategy is not None:
            strategy = int(dome_light_upper_lower_strategy)
            runtime_settings["/rtx/domeLight/upperLowerStrategy"] = strategy
            # IsaacLab RenderCfg uses the dotted carb key name in examples.
            # Set both forms because Isaac/Kit settings APIs accept different
            # spellings in different versions.
            runtime_settings["rtx.domeLight.upperLowerStrategy"] = strategy
        for key, value in runtime_settings.items():
            try:
                settings.set(key, value)
                applied[key] = value
            except Exception as exc:
                applied[key] = f"failed: {exc}"
    return applied


def _apply_beauty_lighting() -> dict[str, Any]:
    """Add soft outdoor/studio lighting for presentation renders."""
    import isaaclab.sim as sim_utils

    spawned = []
    dome = sim_utils.DomeLightCfg(intensity=450.0, color=(0.92, 0.96, 1.0), exposure=0.0)
    dome.func("/World/BeautyDomeLight", dome)
    spawned.append({"path": "/World/BeautyDomeLight", "type": "DomeLight", "intensity": 450.0})

    # Sun-like key light: the reference IsaacLab/Newton render has directional
    # outdoor lighting, not just a local studio sphere light.
    sun = sim_utils.DistantLightCfg(
        intensity=1.0,
        exposure=9.0,
        angle=1.6,
        color=(1.0, 0.94, 0.84),
        enable_color_temperature=True,
        color_temperature=5600.0,
    )
    sun.func(
        "/World/BeautySunLight",
        sun,
        orientation=(0.70034, -0.27732, 0.62398, 0.20799),
    )
    spawned.append(
        {
            "path": "/World/BeautySunLight",
            "type": "DistantLight",
            "exposure": 9.0,
            "angle": 1.6,
            "orientation_wxyz": [0.70034, -0.27732, 0.62398, 0.20799],
        }
    )

    key = sim_utils.SphereLightCfg(radius=4.0, intensity=3200.0, color=(1.0, 0.90, 0.78), exposure=0.0)
    key.func("/World/KeyBounceLight", key, translation=(-2.4, -3.2, 4.0))
    spawned.append({"path": "/World/KeyBounceLight", "type": "SphereLight", "translation": [-2.4, -3.2, 4.0]})

    fill = sim_utils.SphereLightCfg(radius=6.0, intensity=1100.0, color=(0.70, 0.82, 1.0), exposure=0.0)
    fill.func("/World/FillLight", fill, translation=(4.0, 3.0, 3.5))
    spawned.append({"path": "/World/FillLight", "type": "SphereLight", "translation": [4.0, 3.0, 3.5]})
    return {"spawned": spawned}


def _apply_single_sun_lighting(
    *,
    exposure: float = 5.5,
    angle: float = 1.2,
    color_temperature: float = 5200.0,
    yaw_offset_deg: float = 0.0,
    color: tuple[float, float, float] = (1.0, 0.97, 0.90),
    elevation_deg: float | None = None,
) -> dict[str, Any]:
    """Add one directional sun light without the overexposing key/fill stack."""
    import isaaclab.sim as sim_utils

    def quat_mul_wxyz(a, b):
        aw, ax, ay, az = a
        bw, bx, by, bz = b
        return (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        )

    def quat_rotate_wxyz(q, v):
        qv = (0.0, float(v[0]), float(v[1]), float(v[2]))
        qc = (q[0], -q[1], -q[2], -q[3])
        rotated = quat_mul_wxyz(quat_mul_wxyz(q, qv), qc)
        return rotated[1:]

    def normalize(v):
        norm = math.sqrt(sum(float(x) * float(x) for x in v))
        if norm <= 1e-8:
            return (0.0, 0.0, 0.0)
        return tuple(float(x) / norm for x in v)

    def quat_from_vectors_wxyz(v_from, v_to):
        v_from = normalize(v_from)
        v_to = normalize(v_to)
        dot = max(-1.0, min(1.0, sum(a * b for a, b in zip(v_from, v_to))))
        if dot < -0.999999:
            # Pick any axis perpendicular to v_from.
            axis = normalize((1.0, 0.0, 0.0) if abs(v_from[0]) < 0.9 else (0.0, 1.0, 0.0))
            axis = normalize(
                (
                    v_from[1] * axis[2] - v_from[2] * axis[1],
                    v_from[2] * axis[0] - v_from[0] * axis[2],
                    v_from[0] * axis[1] - v_from[1] * axis[0],
                )
            )
            return (0.0, axis[0], axis[1], axis[2])
        cross = (
            v_from[1] * v_to[2] - v_from[2] * v_to[1],
            v_from[2] * v_to[0] - v_from[0] * v_to[2],
            v_from[0] * v_to[1] - v_from[1] * v_to[0],
        )
        return normalize((1.0 + dot, cross[0], cross[1], cross[2]))

    sun = sim_utils.DistantLightCfg(
        intensity=1.0,
        exposure=float(exposure),
        angle=float(angle),
        # Keep the key light warm enough to read as sunlight, but avoid the
        # orange/pink cast that made the white robot and object colors look flat.
        color=tuple(float(v) for v in color),
        enable_color_temperature=True,
        color_temperature=float(color_temperature),
    )
    base_orientation = (0.76041, -0.20648, 0.59052, 0.17299)
    yaw_rad = math.radians(float(yaw_offset_deg))
    yaw_orientation = (math.cos(0.5 * yaw_rad), 0.0, 0.0, math.sin(0.5 * yaw_rad))
    orientation = quat_mul_wxyz(yaw_orientation, base_orientation)
    if elevation_deg is not None:
        # Isaac distant lights emit along local -Z.  The old base orientation is
        # intentionally preserved unless this override is passed.
        old_direction = normalize(quat_rotate_wxyz(orientation, (0.0, 0.0, -1.0)))
        horizontal = normalize((old_direction[0], old_direction[1], 0.0))
        elevation_rad = math.radians(float(elevation_deg))
        desired_direction = normalize(
            (
                horizontal[0] * math.cos(elevation_rad),
                horizontal[1] * math.cos(elevation_rad),
                -abs(math.sin(elevation_rad)),
            )
        )
        orientation = quat_from_vectors_wxyz((0.0, 0.0, -1.0), desired_direction)
    sun.func("/World/CinematicSingleSun", sun, orientation=orientation)
    return {
        "style": "single_sun",
        "spawned": [
            {
                "path": "/World/CinematicSingleSun",
                "type": "DistantLight",
                "exposure": float(exposure),
                "angle": float(angle),
                "color_temperature": float(color_temperature),
                "color": [float(v) for v in color],
                "yaw_offset_deg": float(yaw_offset_deg),
                "elevation_deg": None if elevation_deg is None else float(elevation_deg),
                "orientation_wxyz": list(orientation),
            }
        ],
    }


def _set_default_world_light_intensity(intensity: float) -> dict[str, Any]:
    """Dim the env-created default dome light so sun/shadows can read."""
    from pxr import Sdf
    from isaaclab.sim.utils import get_current_stage

    stage = get_current_stage()
    paths = ["/World/Light", "/World/light"]
    updated = []
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            continue
        attr = prim.GetAttribute("inputs:intensity")
        if not attr.IsValid():
            attr = prim.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float)
        old_value = attr.Get()
        attr.Set(float(intensity))
        updated.append({"path": path, "old_intensity": old_value, "new_intensity": float(intensity)})
    return {"requested_intensity": float(intensity), "updated": updated}


def _apply_sky_background(
    *,
    style: str,
    color: tuple[float, float, float],
    dome_intensity: float,
    hdri_preset: str,
    hdri_path: str | None,
    dynamic_sky_preset: str,
    dynamic_sky_path: str | None,
) -> dict[str, Any]:
    """Set a presentation background without changing physics geometry."""
    if style == "default":
        return {"style": style, "applied": False}

    import carb

    settings = carb.settings.get_settings()
    rgb = tuple(float(v) for v in color)
    if style == "blue_color":
        settings.set("/rtx/background/source/type", "color")
        settings.set("/rtx/background/source/color", rgb)
        return {"style": style, "applied": True, "color": list(rgb)}

    if style == "blue_dome":
        import isaaclab.sim as sim_utils

        settings.set("/rtx/background/source/type", "domeLight")
        settings.set("/rtx/background/source/color", rgb)
        dome = sim_utils.DomeLightCfg(intensity=float(dome_intensity), color=rgb, exposure=0.0)
        dome.func("/World/BlueSkyDomeLight", dome)
        return {
            "style": style,
            "applied": True,
            "color": list(rgb),
            "dome_light": {"path": "/World/BlueSkyDomeLight", "intensity": float(dome_intensity)},
        }

    if style == "hdri":
        import isaaclab.sim as sim_utils

        resolved = _resolve_hdri_path(hdri_preset, hdri_path)
        if resolved is None:
            return {
                "style": style,
                "applied": False,
                "reason": f"could not resolve HDRI preset={hdri_preset!r} path={hdri_path!r}",
            }
        settings.set("/rtx/background/source/type", "domeLight")
        dome = sim_utils.DomeLightCfg(
            intensity=float(dome_intensity),
            exposure=0.0,
            texture_file=resolved,
            texture_format="latlong",
            visible_in_primary_ray=True,
        )
        dome.func("/World/HDRISkyDomeLight", dome)
        try:
            from pxr import Sdf
            from isaaclab.sim.utils import get_current_stage

            prim = get_current_stage().GetPrimAtPath("/World/HDRISkyDomeLight")
            prim.CreateAttribute("visibleInPrimaryRay", Sdf.ValueTypeNames.Bool).Set(True)
        except Exception:
            pass
        return {
            "style": style,
            "applied": True,
            "hdri_preset": hdri_preset,
            "hdri_path": resolved,
            "dome_light": {"path": "/World/HDRISkyDomeLight", "intensity": float(dome_intensity)},
        }

    if style == "dynamic_clear_sky":
        import omni.kit.app

        ext_manager = omni.kit.app.get_app().get_extension_manager()
        if not ext_manager.is_extension_enabled("omni.kit.environment.core"):
            ext_manager.set_extension_enabled_immediate("omni.kit.environment.core", True)
        from omni.kit.environment.core import EnvironmentSettings, SkyHelper, SkyType, import_environment

        url = _resolve_dynamic_sky_path(dynamic_sky_preset, dynamic_sky_path)
        settings.set(EnvironmentSettings.SHOW_LIGHT_WARNING, False)
        sky_type = SkyHelper.get_env_file_type(url) or SkyType.DYNAMIC
        import_environment(sky_type, url)
        return {
            "style": style,
            "applied": True,
            "dynamic_sky_preset": dynamic_sky_preset,
            "sky_type": sky_type,
            "sky_url": url,
        }

    raise ValueError(f"Unsupported sky background style: {style}")


def _apply_backdrop_walls(
    env,
    *,
    style: str,
    color: tuple[float, float, float],
    horizon_color: tuple[float, float, float],
    distance: float,
    height: float,
    extent_margin: float,
    gradient_bands: int,
) -> dict[str, Any]:
    """Add visual-only blue walls behind the grid to read like a clean sky backdrop."""
    if style == "none":
        return {"style": style, "applied": False}
    if style not in {"blue_wall", "blue_walls", "gradient_sky"}:
        raise ValueError(f"Unsupported backdrop style: {style}")

    from pxr import Gf, UsdGeom
    from isaaclab.sim.utils import get_current_stage

    stage = get_current_stage()
    origins = env.scene.env_origins.detach().cpu()
    min_x = float(origins[:, 0].min())
    max_x = float(origins[:, 0].max())
    min_y = float(origins[:, 1].min())
    max_y = float(origins[:, 1].max())
    z_center = 0.5 * float(height)
    thickness = 0.04
    rgb = Gf.Vec3f(*[float(v) for v in color])
    center_x = 0.5 * (min_x + max_x)
    center_y = 0.5 * (min_y + max_y)
    span_x = max_x - min_x
    span_y = max_y - min_y

    if style == "gradient_sky":
        root_path = "/World/GradientSkyBackdrop"
        root = UsdGeom.Xform.Define(stage, root_path)
        normal = 2.0 ** -0.5
        max_projection = 0.5 * (span_x + span_y) * normal
        wall_center = (
            center_x + normal * (max_projection + float(distance)),
            center_y + normal * (max_projection + float(distance)),
            z_center,
        )
        width = (span_x**2 + span_y**2) ** 0.5 + 2.0 * float(extent_margin)
        # Synthetic sky for camera sensors that do not show the DomeLight
        # texture reliably.  Use --backdrop_color as the top color, then fade
        # down to a pale horizon so the backdrop reads like atmosphere instead
        # of a flat blue wall.
        top = [float(v) for v in color]
        horizon = [float(v) for v in horizon_color]
        band_colors = []
        num_bands = max(2, int(gradient_bands))
        for band_idx in range(num_bands):
            alpha = band_idx / max(1, num_bands - 1)
            band_colors.append(tuple((1.0 - alpha) * h + alpha * t for h, t in zip(horizon, top)))
        bands = list(enumerate(band_colors))
        walls = []
        for band_idx, band_color in bands:
            band_height = float(height) / len(bands)
            band_center_z = band_height * (band_idx + 0.5)
            prim_path = f"{root_path}/Band_{band_idx:02d}"
            prim = stage.DefinePrim(prim_path, "Cube")
            cube = UsdGeom.Cube(prim)
            cube.CreateSizeAttr(1.0)
            xform = UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3d(wall_center[0], wall_center[1], band_center_z))
            xform.AddRotateZOp().Set(-45.0)
            xform.AddScaleOp().Set(Gf.Vec3d(width, thickness, band_height + 0.02))
            UsdGeom.Gprim(prim).GetDisplayColorAttr().Set([Gf.Vec3f(*band_color)])
            walls.append(
                {
                    "path": prim_path,
                    "translate": [wall_center[0], wall_center[1], band_center_z],
                    "rotate_z_deg": -45.0,
                    "scale": [width, thickness, band_height + 0.02],
                    "color": list(band_color),
                }
            )
        return {
            "style": style,
            "applied": True,
            "distance": float(distance),
            "height": float(height),
            "extent_margin": float(extent_margin),
            "gradient_bands": int(num_bands),
            "walls": walls,
        }

    # The SAPG reference pan looks mostly from -x/-y toward +x/+y.
    # A single diagonal wall avoids the obvious corner seam of two axis-aligned walls.
    if style == "blue_wall":
        normal = 2.0 ** -0.5
        max_projection = 0.5 * (span_x + span_y) * normal
        wall_center = (
            center_x + normal * (max_projection + float(distance)),
            center_y + normal * (max_projection + float(distance)),
            z_center,
        )
        walls = [
            {
                "path": "/World/BlueBackdropDiagonal",
                "translate": wall_center,
                "rotate_z_deg": -45.0,
                "scale": (
                    (span_x**2 + span_y**2) ** 0.5 + 2.0 * float(extent_margin),
                    thickness,
                    float(height),
                ),
            }
        ]
    else:
        walls = [
            {
                "path": "/World/BlueBackdropY",
                "translate": (
                    center_x,
                    max_y + float(distance),
                    z_center,
                ),
                "scale": (
                    span_x + 2.0 * float(extent_margin),
                    thickness,
                    float(height),
                ),
            },
            {
                "path": "/World/BlueBackdropX",
                "translate": (
                    max_x + float(distance),
                    center_y,
                    z_center,
                ),
                "scale": (
                    thickness,
                    span_y + 2.0 * float(extent_margin),
                    float(height),
                ),
            },
        ]
    for wall in walls:
        prim = stage.DefinePrim(wall["path"], "Cube")
        cube = UsdGeom.Cube(prim)
        cube.CreateSizeAttr(1.0)
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(*wall["translate"]))
        if "rotate_z_deg" in wall:
            xform.AddRotateZOp().Set(float(wall["rotate_z_deg"]))
        xform.AddScaleOp().Set(Gf.Vec3d(*wall["scale"]))
        UsdGeom.Gprim(prim).GetDisplayColorAttr().Set([rgb])

    return {
        "style": style,
        "applied": True,
        "color": [float(v) for v in color],
        "distance": float(distance),
        "height": float(height),
        "extent_margin": float(extent_margin),
        "walls": walls,
    }


def _make_camera_cfg(
    width: int,
    height: int,
    *,
    focal_length_cm: float = 24.0,
    focus_distance_m: float = 400.0,
    f_stop: float = 0.0,
    horizontal_aperture_cm: float = 20.955,
):
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg

    return CameraCfg(
        prim_path="/World/RecordCamera",
        update_period=0,
        height=height,
        width=width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=float(focal_length_cm),
            focus_distance=float(focus_distance_m),
            f_stop=float(f_stop),
            horizontal_aperture=float(horizontal_aperture_cm),
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


def _capture_viewport_rgb(
    *,
    camera_path: str,
    output_dir: Path,
    step_i: int,
    width: int,
    height: int,
    samples_per_pixel: int,
):
    """Capture through Kit's viewport capture extension instead of the camera sensor."""
    import time
    import imageio.v2 as imageio
    import omni.kit.app

    app = omni.kit.app.get_app()
    ext_manager = app.get_extension_manager()
    for ext_name in ("omni.kit.viewport.window", "omni.kit.capture.viewport"):
        if not ext_manager.is_extension_enabled(ext_name):
            ext_manager.set_extension_enabled_immediate(ext_name, True)

    from omni.kit.capture.viewport import CaptureExtension, CaptureOptions, CaptureRenderPreset

    try:
        from omni.kit.viewport.utility import get_active_viewport
        from pxr import Sdf

        viewport = get_active_viewport()
        if viewport is not None:
            viewport.camera_path = Sdf.Path(camera_path)
            try:
                viewport.resolution = (int(width), int(height))
            except Exception:
                pass
    except Exception:
        pass

    capture_dir = output_dir / "_viewport_capture_tmp"
    capture_dir.mkdir(parents=True, exist_ok=True)
    capture = CaptureExtension.get_instance()
    options = CaptureOptions()
    options.file_type = ".png"
    options.output_folder = str(capture_dir)
    options.file_name = f"viewport_step_{step_i:04d}"
    options.camera = camera_path
    options.res_width = int(width)
    options.res_height = int(height)
    options.render_preset = CaptureRenderPreset.PATH_TRACE
    options.path_trace_spp = int(samples_per_pixel)
    options.hdr_output = False
    options.overwrite_existing_frames = True
    capture.options = options
    if not capture.start():
        raise RuntimeError("Viewport capture failed to start")

    output_path = capture_dir / f"{options.file_name}1.png"
    timeout_s = 45.0
    deadline = time.time() + timeout_s
    while not capture.done and time.time() < deadline:
        if output_path.exists():
            break
        app.update()

    outputs = capture.get_outputs()
    if outputs:
        output_path = Path(outputs[0])
    for _ in range(50):
        if output_path.exists():
            break
        app.update()
        time.sleep(0.05)
    if not output_path.exists():
        raise FileNotFoundError(f"Viewport capture output was reported but not written: {output_path}")
    if not capture.done:
        try:
            capture.cancel()
        except Exception:
            pass
    return imageio.imread(output_path)[:, :, :3]


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
    if args.table_reset_z_range_m is not None:
        env_cfg.reset.table_reset_z_range = float(args.table_reset_z_range_m)
    if args.reset_dof_pos_noise_arm is not None:
        env_cfg.reset.reset_dof_pos_random_interval_arm = float(args.reset_dof_pos_noise_arm)
    if args.reset_dof_pos_noise_fingers is not None:
        env_cfg.reset.reset_dof_pos_random_interval_fingers = float(args.reset_dof_pos_noise_fingers)
    if args.reset_dof_vel_noise is not None:
        env_cfg.reset.reset_dof_vel_random_interval = float(args.reset_dof_vel_noise)
    if args.force_scale is not None:
        env_cfg.domain_randomization.force_scale = float(args.force_scale)
    if args.torque_scale is not None:
        env_cfg.domain_randomization.torque_scale = float(args.torque_scale)


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


def _postprocess_rgb_frame(
    frame,
    *,
    exposure: float,
    contrast: float,
    saturation: float,
    gamma: float,
):
    """Apply simple RGB-only output grading after sensor readback."""
    import numpy as np

    if (
        abs(float(exposure)) < 1e-6
        and abs(float(contrast) - 1.0) < 1e-6
        and abs(float(saturation) - 1.0) < 1e-6
        and abs(float(gamma) - 1.0) < 1e-6
    ):
        return frame

    x = frame.astype(np.float32) / 255.0
    x = x * (2.0 ** float(exposure))
    x = (x - 0.5) * float(contrast) + 0.5
    if abs(float(saturation) - 1.0) > 1e-6:
        gray = (
            0.2126 * x[..., 0:1]
            + 0.7152 * x[..., 1:2]
            + 0.0722 * x[..., 2:3]
        )
        x = gray + float(saturation) * (x - gray)
    if abs(float(gamma) - 1.0) > 1e-6:
        x = np.clip(x, 0.0, 1.0) ** (1.0 / max(1e-6, float(gamma)))
    return (np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


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
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run Isaac Sim headless. Use --no-headless to open the Kit window for visual debugging.",
    )
    parser.add_argument(
        "--hold_open_s",
        type=float,
        default=0.0,
        help="Keep the Isaac window alive after capture. Use a negative value to hold until Ctrl-C.",
    )
    parser.add_argument(
        "--render_quality_preset",
        choices=("default", "beauty"),
        default="default",
        help="Render settings preset. 'beauty' enables quality mode, DLAA, GI, shadows, AO, reflections, denoiser.",
    )
    parser.add_argument("--render_samples_per_pixel", type=int, default=64)
    parser.add_argument(
        "--dome_light_upper_lower_strategy",
        type=int,
        default=None,
        help=(
            "Optional RTX dome light upper/lower strategy. IsaacLab docs use 0 for full IBL "
            "and 4 for approximate sky-with-separate-sun."
        ),
    )
    parser.add_argument(
        "--render_mode",
        choices=("auto", "default", "rt", "pt"),
        default="auto",
        help="Runtime RTX render mode. auto uses path tracing for --render_quality_preset=beauty.",
    )
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
    parser.add_argument(
        "--capture_source",
        choices=("camera_sensor", "viewport"),
        default="camera_sensor",
        help="RGB capture path. 'viewport' uses Kit viewport capture and may show stage lighting/sky differently.",
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
        "--camera_focal_length_cm",
        type=float,
        default=24.0,
        help="USD camera focal length in cm. Larger values narrow FOV and read more cinematic.",
    )
    parser.add_argument(
        "--camera_focus_distance_m",
        type=float,
        default=400.0,
        help="Focus plane distance in meters. Use scene-scale values such as 8-25 for DOF hero frames.",
    )
    parser.add_argument(
        "--camera_f_stop",
        type=float,
        default=0.0,
        help="USD camera aperture. 0 disables DOF; smaller positive values create stronger blur.",
    )
    parser.add_argument(
        "--camera_horizontal_aperture_cm",
        type=float,
        default=20.955,
        help="USD horizontal aperture in cm.",
    )
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
        choices=(
            "display_color",
            "marble_tiles",
            "isaac_marble_pbr_tiles",
            "white_stone_slabs",
            "warm_limestone_pbr_tiles",
            "cool_concrete_pbr_tiles",
            "soft_concrete_pbr_tiles",
            "matte_warm_gray_pbr_tiles",
            "matte_slate_pbr_tiles",
            "matte_greige_pbr_tiles",
            "nvidia_precast_concrete_pbr_tiles",
            "fieldstone_pbr_tiles",
            "nvidia_precast_concrete_white",
            "nvidia_precast_concrete_ivory",
            "nvidia_precast_concrete_light_gray",
            "nvidia_precast_concrete_warm_gray",
            "nvidia_precast_concrete_gray",
            "nvidia_precast_concrete_dark_gray",
        ),
        default="display_color",
        help="Floor visual style used when --style_floor is set.",
    )
    parser.add_argument("--beauty_lighting", action="store_true")
    parser.add_argument(
        "--lighting_style",
        choices=("none", "beauty", "single_sun"),
        default="none",
        help="Optional extra lighting stack. 'beauty' is brighter studio fill; 'single_sun' keeps contrast/shadows.",
    )
    parser.add_argument("--single_sun_exposure", type=float, default=5.5)
    parser.add_argument("--single_sun_angle", type=float, default=1.2)
    parser.add_argument("--single_sun_color_temperature", type=float, default=5200.0)
    parser.add_argument("--single_sun_color", type=float, nargs=3, default=(1.0, 0.97, 0.90))
    parser.add_argument(
        "--single_sun_elevation_deg",
        type=float,
        default=None,
        help="Optional sun elevation above horizon in degrees. If unset, preserves the historical fixed orientation.",
    )
    parser.add_argument(
        "--single_sun_yaw_offset_deg",
        type=float,
        default=0.0,
        help="Rotate the cinematic sun around world Z while preserving its base elevation.",
    )
    parser.add_argument(
        "--default_light_intensity",
        type=float,
        default=None,
        help="If set, override the env-created /World/Light intensity before adding cinematic lights.",
    )
    parser.add_argument("--goal_color", type=float, nargs=3, default=(0.55, 1.0, 0.55))
    parser.add_argument("--goal_opacity", type=float, default=0.35)
    parser.add_argument(
        "--table_style",
        choices=("display_color", "oak_pbr"),
        default="display_color",
        help="Table visual style used when --style_table is set.",
    )
    parser.add_argument("--table_color", type=float, nargs=3, default=(0.72, 0.50, 0.30))
    parser.add_argument("--table_texture_scale", type=float, default=1.0)
    parser.add_argument("--table_texture_path", default=str(DEFAULT_OAK_BASE_COLOR))
    parser.add_argument("--table_normal_path", default=str(DEFAULT_OAK_NORMAL))
    parser.add_argument("--table_orm_path", default=str(DEFAULT_OAK_ORM))
    parser.add_argument("--floor_color", type=float, nargs=3, default=(0.60, 0.60, 0.56))
    parser.add_argument("--floor_tile_count", type=int, default=24)
    parser.add_argument("--floor_tile_size", type=float, default=0.9)
    parser.add_argument("--floor_tile_gap", type=float, default=0.012)
    parser.add_argument(
        "--floor_texture_path",
        default=None,
        help=(
            "Optional texture for --floor_style=isaac_marble_pbr_tiles. "
            "If omitted, resolves NVIDIA Isaac sample marble_tile.png from the Isaac asset root."
        ),
    )
    parser.add_argument("--floor_texture_scale", type=float, default=4.0)
    parser.add_argument(
        "--floor_normal_path",
        default=None,
        help="Optional normal map for PBR tile floor styles.",
    )
    parser.add_argument("--floor_roughness", type=float, default=0.42)
    parser.add_argument("--floor_normal_strength", type=float, default=0.35)
    parser.add_argument("--floor_specular_level", type=float, default=0.5)
    parser.add_argument(
        "--sky_style",
        choices=("default", "blue_color", "blue_dome", "hdri", "dynamic_clear_sky"),
        default="default",
        help="Presentation background style. Does not change physics ground.",
    )
    parser.add_argument("--sky_color", type=float, nargs=3, default=(0.58, 0.74, 0.98))
    parser.add_argument("--sky_dome_intensity", type=float, default=1200.0)
    parser.add_argument(
        "--sky_hdri_preset",
        choices=(
            "stinson_beach",
            "photo_studio",
            "cape_hill",
            "old_outdoor_theater",
            "cloudy_vondelpark",
            "wasteland_clouds",
            "winter_sky",
            "kloofendal_partly_cloudy",
            "sunset_fairway",
        ),
        default="stinson_beach",
    )
    parser.add_argument(
        "--sky_hdri_path",
        default=None,
        help="Explicit local .hdr/.exr path for --sky_style=hdri. Overrides --sky_hdri_preset.",
    )
    parser.add_argument(
        "--dynamic_sky_preset",
        choices=("remote_clear", "local_simple", "local_sunstudy"),
        default="remote_clear",
        help="Dynamic sky asset used when --sky_style=dynamic_clear_sky and --dynamic_sky_path is unset.",
    )
    parser.add_argument(
        "--dynamic_sky_path",
        default=None,
        help="Explicit dynamic sky USD path/URL for --sky_style=dynamic_clear_sky.",
    )
    parser.add_argument(
        "--backdrop_style",
        choices=("none", "blue_wall", "blue_walls", "gradient_sky"),
        default="none",
        help="Render-only background geometry for stronger sky contrast.",
    )
    parser.add_argument("--backdrop_color", type=float, nargs=3, default=(0.36, 0.58, 0.90))
    parser.add_argument("--backdrop_horizon_color", type=float, nargs=3, default=(0.84, 0.89, 0.92))
    parser.add_argument("--backdrop_distance", type=float, default=5.0)
    parser.add_argument("--backdrop_height", type=float, default=18.0)
    parser.add_argument("--backdrop_extent_margin", type=float, default=10.0)
    parser.add_argument("--backdrop_gradient_bands", type=int, default=4)
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
        "--object_color_saturation",
        type=float,
        default=1.0,
        help="HSV saturation multiplier for recolored objects. Values >1 make object colors pop more.",
    )
    parser.add_argument(
        "--object_color_value_scale",
        type=float,
        default=1.0,
        help="HSV value multiplier for recolored objects. Values <1 reduce pastel overexposure.",
    )
    parser.add_argument(
        "--image_exposure",
        type=float,
        default=0.0,
        help="Output-only exposure adjustment in stops applied to saved PNG/video frames.",
    )
    parser.add_argument(
        "--image_contrast",
        type=float,
        default=1.0,
        help="Output-only contrast multiplier applied to saved PNG/video frames.",
    )
    parser.add_argument(
        "--image_saturation",
        type=float,
        default=1.0,
        help="Output-only saturation multiplier applied to saved PNG/video frames.",
    )
    parser.add_argument(
        "--image_gamma",
        type=float,
        default=1.0,
        help="Output-only gamma adjustment applied to saved PNG/video frames.",
    )
    parser.add_argument(
        "--object_distribution_mode",
        choices=(
            "training",
            "mixed_training_simple_25_25_50",
            "mixed_training_easy_video_25_25_50",
        ),
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
    parser.add_argument(
        "--table_reset_z_range_m",
        type=float,
        default=None,
        help="Override table z reset half-width in meters. Use 0 for the default table height.",
    )
    parser.add_argument(
        "--reset_dof_pos_noise_arm",
        type=float,
        default=None,
        help="Override arm joint reset interpolation interval. Use 0 for default joint pose.",
    )
    parser.add_argument(
        "--reset_dof_pos_noise_fingers",
        type=float,
        default=None,
        help="Override hand joint reset interpolation interval. Use 0 for default joint pose.",
    )
    parser.add_argument(
        "--reset_dof_vel_noise",
        type=float,
        default=None,
        help="Override joint velocity reset half-width. Use 0 for zero reset velocities.",
    )
    parser.add_argument(
        "--force_scale",
        type=float,
        default=None,
        help="Override random object force scale. Use 0 to disable force perturbations.",
    )
    parser.add_argument(
        "--torque_scale",
        type=float,
        default=None,
        help="Override random object torque scale. Use 0 to disable torque perturbations.",
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
    launcher_args.headless = bool(my_args.headless)
    launcher_args.enable_cameras = True
    app = AppLauncher(launcher_args).app
    runtime_render_summary = _apply_runtime_render_settings(
        preset=my_args.render_quality_preset,
        render_mode=my_args.render_mode,
        samples_per_pixel=my_args.render_samples_per_pixel,
        dome_light_upper_lower_strategy=my_args.dome_light_upper_lower_strategy,
    )

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
        dome_light_upper_lower_strategy=my_args.dome_light_upper_lower_strategy,
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

    camera = Camera(
        cfg=_make_camera_cfg(
            width=width,
            height=height,
            focal_length_cm=float(my_args.camera_focal_length_cm),
            focus_distance_m=float(my_args.camera_focus_distance_m),
            f_stop=float(my_args.camera_f_stop),
            horizontal_aperture_cm=float(my_args.camera_horizontal_aperture_cm),
        )
    )
    env.sim.reset()

    recolor_summary = None
    if not my_args.no_recolor_objects:
        recolor_summary = _recolor_objects_by_env(
            saturation=float(my_args.object_color_saturation),
            value_scale=float(my_args.object_color_value_scale),
        )
        print(
            "[render_simtoolreal_pretrained] recolored "
            f"{recolor_summary['num_colored']} object prims"
        )
    table_style_summary = None
    if my_args.style_table:
        if my_args.table_style == "oak_pbr":
            table_texture_path = Path(my_args.table_texture_path).expanduser()
            table_normal_path = Path(my_args.table_normal_path).expanduser()
            table_orm_path = Path(my_args.table_orm_path).expanduser()
            if not table_texture_path.is_file():
                raise FileNotFoundError(f"Oak table base color texture does not exist: {table_texture_path}")
            table_style_summary = _apply_pbr_table_material(
                base_color_texture=str(table_texture_path),
                normal_texture=str(table_normal_path) if table_normal_path.is_file() else None,
                orm_texture=str(table_orm_path) if table_orm_path.is_file() else None,
                texture_scale=float(my_args.table_texture_scale),
            )
        else:
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
        if my_args.floor_style in {
            "marble_tiles",
            "isaac_marble_pbr_tiles",
            "white_stone_slabs",
            "warm_limestone_pbr_tiles",
            "cool_concrete_pbr_tiles",
            "soft_concrete_pbr_tiles",
            "matte_warm_gray_pbr_tiles",
            "matte_slate_pbr_tiles",
            "matte_greige_pbr_tiles",
            "nvidia_precast_concrete_pbr_tiles",
            "fieldstone_pbr_tiles",
            "nvidia_precast_concrete_white",
            "nvidia_precast_concrete_ivory",
            "nvidia_precast_concrete_light_gray",
            "nvidia_precast_concrete_warm_gray",
            "nvidia_precast_concrete_gray",
            "nvidia_precast_concrete_dark_gray",
        }:
            # Keep the original physics ground, but make it a neutral base below
            # the render-only cinematic tile overlay.
            base_floor_summary = _style_prim_trees(
                "/World/ground",
                color=tuple(float(v) for v in my_args.floor_color),
                opacity=1.0,
                visible=True,
                label="floor_base",
            )
            if my_args.floor_style in {
                "isaac_marble_pbr_tiles",
                "white_stone_slabs",
                "warm_limestone_pbr_tiles",
                "cool_concrete_pbr_tiles",
                "nvidia_precast_concrete_pbr_tiles",
                "fieldstone_pbr_tiles",
                "matte_warm_gray_pbr_tiles",
                "matte_slate_pbr_tiles",
                "matte_greige_pbr_tiles",
                "nvidia_precast_concrete_white",
                "nvidia_precast_concrete_ivory",
                "nvidia_precast_concrete_light_gray",
                "nvidia_precast_concrete_warm_gray",
                "nvidia_precast_concrete_gray",
                "nvidia_precast_concrete_dark_gray",
            }:
                mdl_material_name = None
                if my_args.floor_style == "white_stone_slabs":
                    floor_texture_path = my_args.floor_texture_path or str(DEFAULT_WHITE_STONE_TEXTURE)
                    floor_normal_path = my_args.floor_normal_path
                    material_name = "CinematicWhiteStone"
                    style_name = "white_stone_slabs"
                elif my_args.floor_style == "warm_limestone_pbr_tiles":
                    floor_texture_path = my_args.floor_texture_path or str(DEFAULT_WARM_LIMESTONE_TEXTURE)
                    floor_normal_path = my_args.floor_normal_path or str(DEFAULT_WARM_LIMESTONE_NORMAL)
                    material_name = "CinematicWarmLimestone"
                    style_name = "warm_limestone_pbr_tiles"
                elif my_args.floor_style == "cool_concrete_pbr_tiles":
                    floor_texture_path = my_args.floor_texture_path or str(DEFAULT_COOL_CONCRETE_TEXTURE)
                    floor_normal_path = my_args.floor_normal_path or str(DEFAULT_COOL_CONCRETE_NORMAL)
                    material_name = "CinematicCoolConcrete"
                    style_name = "cool_concrete_pbr_tiles"
                elif my_args.floor_style == "soft_concrete_pbr_tiles":
                    floor_texture_path = my_args.floor_texture_path or str(DEFAULT_SOFT_CONCRETE_TEXTURE)
                    floor_normal_path = my_args.floor_normal_path or str(DEFAULT_SOFT_CONCRETE_NORMAL)
                    material_name = "CinematicSoftConcrete"
                    style_name = "soft_concrete_pbr_tiles"
                elif my_args.floor_style == "matte_warm_gray_pbr_tiles":
                    floor_texture_path = my_args.floor_texture_path or str(DEFAULT_MATTE_WARM_GRAY_TEXTURE)
                    floor_normal_path = my_args.floor_normal_path or str(DEFAULT_MATTE_WARM_GRAY_NORMAL)
                    material_name = "CinematicMatteWarmGray"
                    style_name = "matte_warm_gray_pbr_tiles"
                elif my_args.floor_style == "matte_slate_pbr_tiles":
                    floor_texture_path = my_args.floor_texture_path or str(DEFAULT_MATTE_SLATE_TEXTURE)
                    floor_normal_path = my_args.floor_normal_path or str(DEFAULT_MATTE_SLATE_NORMAL)
                    material_name = "CinematicMatteSlate"
                    style_name = "matte_slate_pbr_tiles"
                elif my_args.floor_style == "matte_greige_pbr_tiles":
                    floor_texture_path = my_args.floor_texture_path or str(DEFAULT_MATTE_GREIGE_TEXTURE)
                    floor_normal_path = my_args.floor_normal_path or str(DEFAULT_MATTE_GREIGE_NORMAL)
                    material_name = "CinematicMatteGreige"
                    style_name = "matte_greige_pbr_tiles"
                elif my_args.floor_style == "nvidia_precast_concrete_pbr_tiles":
                    floor_texture_path = my_args.floor_texture_path or str(DEFAULT_NVIDIA_PRECAST_CONCRETE_TEXTURE)
                    floor_normal_path = my_args.floor_normal_path or str(DEFAULT_NVIDIA_PRECAST_CONCRETE_NORMAL)
                    material_name = "CinematicPrecastConcretePbr"
                    style_name = "nvidia_precast_concrete_pbr_tiles"
                elif my_args.floor_style == "fieldstone_pbr_tiles":
                    floor_texture_path = my_args.floor_texture_path or str(DEFAULT_FIELDSTONE_TEXTURE)
                    floor_normal_path = my_args.floor_normal_path or str(DEFAULT_FIELDSTONE_NORMAL)
                    material_name = "CinematicFieldstone"
                    style_name = "fieldstone_pbr_tiles"
                elif my_args.floor_style == "isaac_marble_pbr_tiles":
                    floor_texture_path = my_args.floor_texture_path or _resolve_isaac_asset(
                        ISAAC_SAMPLE_MARBLE_TEXTURE
                    )
                    floor_normal_path = my_args.floor_normal_path
                    material_name = "CinematicIsaacMarble"
                    style_name = "isaac_marble_pbr_tiles"
                else:
                    floor_texture_path = my_args.floor_texture_path or str(DEFAULT_NVIDIA_PRECAST_CONCRETE_MDL)
                    floor_normal_path = my_args.floor_normal_path
                    precast_styles = {
                        "nvidia_precast_concrete_white": ("Concrete_Precast", "CinematicPrecastConcreteWhite"),
                        "nvidia_precast_concrete_ivory": (
                            "concrete_precast_ivory",
                            "CinematicPrecastConcreteIvory",
                        ),
                        "nvidia_precast_concrete_light_gray": (
                            "concrete_precast_light_gray",
                            "CinematicPrecastConcreteLightGray",
                        ),
                        "nvidia_precast_concrete_warm_gray": (
                            "concrete_precast_light_warm_gray",
                            "CinematicPrecastConcreteWarmGray",
                        ),
                        "nvidia_precast_concrete_gray": (
                            "concrete_precast_dark_gray",
                            "CinematicPrecastConcreteGray",
                        ),
                        "nvidia_precast_concrete_dark_gray": (
                            "concrete_precast_dark_charcoal",
                            "CinematicPrecastConcreteDarkGray",
                        ),
                    }
                    mdl_material_name, material_name = precast_styles[my_args.floor_style]
                    style_name = my_args.floor_style
                tile_floor_summary = _apply_pbr_tile_floor(
                    base_color=tuple(float(v) for v in my_args.floor_color),
                    tile_count=int(my_args.floor_tile_count),
                    tile_size=float(my_args.floor_tile_size),
                    tile_gap=float(my_args.floor_tile_gap),
                    texture_path=floor_texture_path,
                    normal_texture_path=floor_normal_path,
                    texture_scale=float(my_args.floor_texture_scale),
                    roughness=float(my_args.floor_roughness),
                    normal_texture_influence=float(my_args.floor_normal_strength),
                    specular_level=float(my_args.floor_specular_level),
                    material_name=material_name,
                    style_name=style_name,
                    mdl_material_name=mdl_material_name,
                )
            else:
                tile_floor_summary = _apply_marble_tile_floor(
                    base_color=tuple(float(v) for v in my_args.floor_color),
                    tile_count=int(my_args.floor_tile_count),
                    tile_size=float(my_args.floor_tile_size),
                    tile_gap=float(my_args.floor_tile_gap),
                )
            floor_style_summary = {
                "label": "floor",
                "style": my_args.floor_style,
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
    sky_summary = _apply_sky_background(
        style=my_args.sky_style,
        color=tuple(float(v) for v in my_args.sky_color),
        dome_intensity=float(my_args.sky_dome_intensity),
        hdri_preset=my_args.sky_hdri_preset,
        hdri_path=my_args.sky_hdri_path,
        dynamic_sky_preset=my_args.dynamic_sky_preset,
        dynamic_sky_path=my_args.dynamic_sky_path,
    )
    if sky_summary["applied"]:
        print(f"[render_simtoolreal_pretrained] applied sky background: {sky_summary}")
    default_light_summary = None
    if my_args.default_light_intensity is not None:
        default_light_summary = _set_default_world_light_intensity(float(my_args.default_light_intensity))
        print(f"[render_simtoolreal_pretrained] set default light: {default_light_summary}")
    backdrop_summary = _apply_backdrop_walls(
        env,
        style=my_args.backdrop_style,
        color=tuple(float(v) for v in my_args.backdrop_color),
        horizon_color=tuple(float(v) for v in my_args.backdrop_horizon_color),
        distance=float(my_args.backdrop_distance),
        height=float(my_args.backdrop_height),
        extent_margin=float(my_args.backdrop_extent_margin),
        gradient_bands=int(my_args.backdrop_gradient_bands),
    )
    if backdrop_summary["applied"]:
        print(f"[render_simtoolreal_pretrained] applied backdrop: {backdrop_summary}")
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
    lighting_style = "beauty" if my_args.beauty_lighting else my_args.lighting_style
    if lighting_style == "beauty":
        lighting_summary = _apply_beauty_lighting()
        print(f"[render_simtoolreal_pretrained] applied beauty lighting: {lighting_summary}")
    elif lighting_style == "single_sun":
        lighting_summary = _apply_single_sun_lighting(
            exposure=float(my_args.single_sun_exposure),
            angle=float(my_args.single_sun_angle),
            color_temperature=float(my_args.single_sun_color_temperature),
            yaw_offset_deg=float(my_args.single_sun_yaw_offset_deg),
            color=tuple(float(v) for v in my_args.single_sun_color),
            elevation_deg=None
            if my_args.single_sun_elevation_deg is None
            else float(my_args.single_sun_elevation_deg),
        )
        print(f"[render_simtoolreal_pretrained] applied single-sun lighting: {lighting_summary}")

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
    else:
        print(f"[diag] scene env_spacing = {my_args.env_spacing}")
    print(f"[diag] quality = {my_args.quality}, width = {width}, height = {height}")
    print(f"[diag] render_quality = {render_quality_summary}")
    print(f"[diag] runtime_render = {runtime_render_summary}")
    print(f"[diag] sky = {sky_summary}")
    print(f"[diag] backdrop = {backdrop_summary}")
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
    print(f"[diag] table reset z range = {env_cfg.reset.table_reset_z_range:g} m")
    print(
        "[diag] robot reset noise = "
        f"arm={env_cfg.reset.reset_dof_pos_random_interval_arm:g} "
        f"fingers={env_cfg.reset.reset_dof_pos_random_interval_fingers:g} "
        f"vel={env_cfg.reset.reset_dof_vel_random_interval:g}"
    )
    print(
        "[diag] wrench randomization = "
        f"force_scale={env_cfg.domain_randomization.force_scale:g} "
        f"torque_scale={env_cfg.domain_randomization.torque_scale:g}"
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
            "table_reset_z_range_m": float(env_cfg.reset.table_reset_z_range),
            "reset_dof_pos_random_interval_arm": float(
                env_cfg.reset.reset_dof_pos_random_interval_arm
            ),
            "reset_dof_pos_random_interval_fingers": float(
                env_cfg.reset.reset_dof_pos_random_interval_fingers
            ),
            "reset_dof_vel_random_interval": float(env_cfg.reset.reset_dof_vel_random_interval),
        },
        "domain_randomization": {
            "force_scale": float(env_cfg.domain_randomization.force_scale),
            "torque_scale": float(env_cfg.domain_randomization.torque_scale),
        },
        "quality": my_args.quality,
        "width": width,
        "height": height,
        "render_quality_preset": my_args.render_quality_preset,
        "render_samples_per_pixel": my_args.render_samples_per_pixel,
        "dome_light_upper_lower_strategy": (
            int(my_args.dome_light_upper_lower_strategy)
            if my_args.dome_light_upper_lower_strategy is not None
            else None
        ),
        "render_mode": my_args.render_mode,
        "render_quality_summary": render_quality_summary,
        "runtime_render_summary": runtime_render_summary,
        "steps": my_args.steps,
        "video_fps": my_args.video_fps,
        "capture_every": capture_every,
        "capture_png_steps": sorted(capture_steps),
        "capture_source": my_args.capture_source,
        "make_video": my_args.make_video,
        "camera_eye": eye.detach().cpu().tolist(),
        "camera_target": target.detach().cpu().tolist(),
        "camera_focal_length_cm": my_args.camera_focal_length_cm,
        "camera_focus_distance_m": my_args.camera_focus_distance_m,
        "camera_f_stop": my_args.camera_f_stop,
        "camera_horizontal_aperture_cm": my_args.camera_horizontal_aperture_cm,
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
        "sky_summary": sky_summary,
        "default_light_summary": default_light_summary,
        "backdrop_summary": backdrop_summary,
        "goal_style_summary": goal_style_summary,
        "lighting_summary": lighting_summary,
        "lighting_style": lighting_style,
        "single_sun_exposure": float(my_args.single_sun_exposure),
        "single_sun_angle": float(my_args.single_sun_angle),
        "single_sun_color_temperature": float(my_args.single_sun_color_temperature),
        "single_sun_color": list(my_args.single_sun_color),
        "single_sun_elevation_deg": None
        if my_args.single_sun_elevation_deg is None
        else float(my_args.single_sun_elevation_deg),
        "single_sun_yaw_offset_deg": float(my_args.single_sun_yaw_offset_deg),
        "hide_goal_viz": bool(my_args.hide_goal_viz),
        "goal_color": list(my_args.goal_color),
        "goal_opacity": float(my_args.goal_opacity),
        "table_style": my_args.table_style,
        "table_color": list(my_args.table_color),
        "table_texture_scale": float(my_args.table_texture_scale),
        "table_texture_path": my_args.table_texture_path,
        "table_normal_path": my_args.table_normal_path,
        "table_orm_path": my_args.table_orm_path,
        "floor_style": my_args.floor_style,
        "floor_color": list(my_args.floor_color),
        "floor_tile_count": int(my_args.floor_tile_count),
        "floor_tile_size": float(my_args.floor_tile_size),
        "floor_tile_gap": float(my_args.floor_tile_gap),
        "floor_texture_path": my_args.floor_texture_path,
        "floor_normal_path": my_args.floor_normal_path,
        "floor_texture_scale": float(my_args.floor_texture_scale),
        "floor_roughness": float(my_args.floor_roughness),
        "floor_normal_strength": float(my_args.floor_normal_strength),
        "floor_specular_level": float(my_args.floor_specular_level),
        "sky_style": my_args.sky_style,
        "sky_color": list(my_args.sky_color),
        "sky_dome_intensity": float(my_args.sky_dome_intensity),
        "sky_hdri_preset": my_args.sky_hdri_preset,
        "sky_hdri_path": my_args.sky_hdri_path,
        "dynamic_sky_preset": my_args.dynamic_sky_preset,
        "dynamic_sky_path": my_args.dynamic_sky_path,
        "default_light_intensity": (
            float(my_args.default_light_intensity)
            if my_args.default_light_intensity is not None
            else None
        ),
        "backdrop_style": my_args.backdrop_style,
        "backdrop_color": list(my_args.backdrop_color),
        "backdrop_horizon_color": list(my_args.backdrop_horizon_color),
        "backdrop_distance": float(my_args.backdrop_distance),
        "backdrop_height": float(my_args.backdrop_height),
        "backdrop_extent_margin": float(my_args.backdrop_extent_margin),
        "backdrop_gradient_bands": int(my_args.backdrop_gradient_bands),
        "object_color_saturation": float(my_args.object_color_saturation),
        "object_color_value_scale": float(my_args.object_color_value_scale),
        "image_postprocess": {
            "exposure": float(my_args.image_exposure),
            "contrast": float(my_args.image_contrast),
            "saturation": float(my_args.image_saturation),
            "gamma": float(my_args.image_gamma),
        },
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
        if not my_args.headless:
            _set_active_viewport_camera("/World/RecordCamera")
        # Flush camera pose through Hydra before readback. Without this, the
        # first frame after a camera move can use the previous camera pose.
        for _ in range(max(1, my_args.camera_render_warmup_frames)):
            env.sim.render()
        if my_args.capture_source == "viewport":
            frame = _capture_viewport_rgb(
                camera_path="/World/RecordCamera",
                output_dir=out_dir,
                step_i=step_i,
                width=width,
                height=height,
                samples_per_pixel=int(my_args.render_samples_per_pixel),
            )
        else:
            frame = _capture_rgb(camera, dt=policy_dt)
        render_ms = (time.perf_counter() - capture_t0) * 1000.0
        if frame is None:
            print(f"[warning] no RGB frame at step {step_i}")
            return
        frame = _postprocess_rgb_frame(
            frame,
            exposure=float(my_args.image_exposure),
            contrast=float(my_args.image_contrast),
            saturation=float(my_args.image_saturation),
            gamma=float(my_args.image_gamma),
        )
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

    if float(my_args.hold_open_s) != 0.0:
        if my_args.headless:
            print("[render_simtoolreal_pretrained] --hold_open_s ignored in headless mode")
        else:
            if float(my_args.hold_open_s) < 0.0:
                print("[render_simtoolreal_pretrained] holding non-headless window open until Ctrl-C")
                while True:
                    app.update()
                    time.sleep(0.02)
            else:
                print(
                    "[render_simtoolreal_pretrained] holding non-headless window open for "
                    f"{float(my_args.hold_open_s):.1f}s"
                )
                deadline = time.time() + float(my_args.hold_open_s)
                while time.time() < deadline:
                    app.update()
                    time.sleep(0.02)

    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
