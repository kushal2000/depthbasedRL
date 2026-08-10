#!/usr/bin/env python3
"""Render isolated fork/plug task overviews with metric dimension callouts."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = REPO_ROOT / "assets" / "urdf" / "plug_fork"


TASKS = {
    "fork": {
        "object": ASSET_ROOT / "fork" / "fork_visual_oriented.obj",
        "receptacle": ASSET_ROOT / "holder" / "holder_tol1mm_visual.glb",
        "dimensions_mm": (197.6, 27.1, 15.9),
        "dimension_label_offsets": ((135.0, 0.0), (-160.0, 25.0), (-150.0, 0.0)),
        "dimension_arrow_offsets": ((166.0, 0.0), (-30.0, -105.0), (-125.0, 255.0)),
        "dimension_min_lengths": (0.0, 0.0, 64.0),
        "receptacle_transform": ("z", 18.0, (-0.030, 0.000, 0.0425)),
        "preinsert_z": 0.141277,
        "preinsert_gap": 0.025,
        "cross_section_fraction": 0.82,
        "camera_eye": (0.393, 0.408, 0.38),
        "camera_target": (-0.015, 0.000, 0.190),
        "yfov_deg": 37.0,
    },
    "plug": {
        "object": ASSET_ROOT / "plug" / "plug_visual.glb",
        "receptacle": ASSET_ROOT / "socket" / "socket_tol0p5mm_visual.glb",
        # Overall mesh dimensions, including the 15.9 mm blades.
        "dimensions_mm": (42.9, 42.5, 41.5),
        "dimension_label_offsets": ((-100.0, 0.0), (-30.0, -80.0), (125.0, 64.0)),
        "dimension_arrow_offsets": ((0.0, 0.0), (0.0, -90.0), (0.0, -90.0)),
        "split_length_dimensions": (
            (-0.0135, 0.0135, "27.0\nmm", (-100.0, 18.0), (35, 47, 55), "left", (0.0, 0.0), 18.0),
            (-0.0294, -0.0135, "15.9 mm", (125.0, 0.0), (35, 47, 55), "right", (130.0, 100.0), 0.0),
        ),
        "receptacle_transform": ("z", 12.0, (-0.0158, 0.0078, 0.0150)),
        "preinsert_z": 0.044400,
        "preinsert_gap": 0.020,
        "cross_section_fraction": 0.62,
        "camera_eye": (0.30, -0.36, 0.28),
        "camera_target": (-0.005, 0.000, 0.055),
        "yfov_deg": 20.0,
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(
            REPO_ROOT
            / "plot_figures/rebuttal/outputs/new_task_rollouts/final_selected/problem_overviews"
        ),
    )
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--height", type=int, default=1080)
    return parser.parse_args()


def _rotation(axis: str, degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    c, s = math.cos(angle), math.sin(angle)
    if axis == "x":
        rotation = np.array(((1, 0, 0), (0, c, -s), (0, s, c)))
    elif axis == "y":
        rotation = np.array(((c, 0, s), (0, 1, 0), (-s, 0, c)))
    elif axis == "z":
        rotation = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
    else:
        raise ValueError(f"Unknown rotation axis: {axis}")
    transform = np.eye(4)
    transform[:3, :3] = rotation
    return transform


def _translation(position: tuple[float, float, float]) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, 3] = position
    return transform


def _compose_transform(
    rotations: tuple[tuple[str, float], ...], position: tuple[float, float, float]
) -> np.ndarray:
    transform = _translation(position)
    for axis, degrees in rotations:
        transform = transform @ _rotation(axis, degrees)
    return transform


def _camera_pose(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return an OpenGL camera-to-world transform looking at target."""
    z_axis = eye - target
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross(np.array((0.0, 0.0, 1.0)), z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    pose = np.eye(4)
    pose[:3, 0] = x_axis
    pose[:3, 1] = y_axis
    pose[:3, 2] = z_axis
    pose[:3, 3] = eye
    return pose


def _add_trimesh_scene(render_scene, path: Path, transform: np.ndarray) -> np.ndarray:
    import pyrender
    import trimesh

    source = trimesh.load(path, force="scene", process=False)
    bounds = source.bounds.copy()
    for node_name in source.graph.nodes_geometry:
        node_transform, geometry_name = source.graph[node_name]
        geometry = source.geometry[geometry_name]
        mesh = pyrender.Mesh.from_trimesh(geometry, smooth=False)
        render_scene.add(mesh, pose=transform @ node_transform)
    return bounds


def _project(
    points_world: np.ndarray,
    camera_pose: np.ndarray,
    yfov: float,
    width: int,
    height: int,
) -> np.ndarray:
    points_h = np.column_stack((points_world, np.ones(len(points_world))))
    camera_points = (np.linalg.inv(camera_pose) @ points_h.T).T[:, :3]
    depth = -camera_points[:, 2]
    fy = 1.0 / math.tan(yfov / 2.0)
    fx = fy / (width / height)
    x_ndc = fx * camera_points[:, 0] / depth
    y_ndc = fy * camera_points[:, 1] / depth
    return np.column_stack(
        ((x_ndc + 1.0) * width / 2.0, (1.0 - y_ndc) * height / 2.0)
    )


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points_h = np.column_stack((points, np.ones(len(points))))
    return (transform @ points_h.T).T[:, :3]


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: np.ndarray,
    end: np.ndarray,
    label: str,
    font: ImageFont.FreeTypeFont,
    label_offset: tuple[float, float],
    color: tuple[int, int, int] = (35, 47, 55),
    endpoint_inset: float = 0.0,
    minimum_length: float = 0.0,
) -> None:
    width = 6
    p0 = np.asarray(start, dtype=float)
    p1 = np.asarray(end, dtype=float)
    direction = p1 - p0
    length = np.linalg.norm(direction)
    if length < 1.0:
        return
    unit = direction / length
    if minimum_length > length:
        extension = (minimum_length - length) / 2.0
        p0 -= unit * extension
        p1 += unit * extension
        direction = p1 - p0
        length = np.linalg.norm(direction)
        unit = direction / length
    if endpoint_inset > 0.0 and length > 2.0 * endpoint_inset:
        p0 += unit * endpoint_inset
        p1 -= unit * endpoint_inset
        direction = p1 - p0
        length = np.linalg.norm(direction)
        unit = direction / length
    normal = np.array((-unit[1], unit[0]))
    draw.line((tuple(p0), tuple(p1)), fill=color, width=width)
    arrow_length, arrow_width = 22.0, 11.0
    for tip, sign in ((p0, 1.0), (p1, -1.0)):
        base = tip + sign * unit * arrow_length
        polygon = (tuple(tip), tuple(base + normal * arrow_width), tuple(base - normal * arrow_width))
        draw.polygon(polygon, fill=color)

    midpoint = (p0 + p1) / 2.0 + np.asarray(label_offset, dtype=float)
    box = draw.multiline_textbbox((0, 0), label, font=font, spacing=0, align="center")
    text_width = box[2] - box[0]
    text_height = box[3] - box[1]
    xy = (midpoint[0] - text_width / 2.0, midpoint[1] - text_height / 2.0 - box[1])
    padding = 8
    background = (
        xy[0] - padding,
        xy[1] + box[1] - padding,
        xy[0] + text_width + padding,
        xy[1] + box[3] + padding,
    )
    draw.rounded_rectangle(background, radius=6, fill=(250, 250, 248, 235))
    draw.multiline_text(xy, label, font=font, fill=color, spacing=0, align="center")


def _annotate_dimensions(
    image: Image.Image,
    bounds: np.ndarray,
    object_transform: np.ndarray,
    dimensions_mm: tuple[float, float, float],
    label_offsets: tuple[tuple[float, float], ...],
    arrow_offsets: tuple[tuple[float, float], ...],
    minimum_lengths: tuple[float, float, float],
    cross_section_fraction: float,
    split_length_dimensions: tuple | None,
    camera_pose: np.ndarray,
    yfov: float,
) -> Image.Image:
    width, height = image.size
    lo, hi = bounds
    span = hi - lo
    margin = max(float(span.max()) * 0.10, 0.006)
    cross_x = lo[0] + float(cross_section_fraction) * span[0]

    # Three offset bounding-box edges, one for each object-local axis.
    segments = []
    if split_length_dimensions:
        for start_x, end_x, label, label_offset, color, edge, arrow_offset, endpoint_inset in split_length_dimensions:
            edge_y = lo[1] - margin if edge == "left" else hi[1] + margin
            segments.append(
                (
                    np.array((start_x, edge_y, hi[2] + margin)),
                    np.array((end_x, edge_y, hi[2] + margin)),
                    label,
                    label_offset,
                    color,
                    arrow_offset,
                    endpoint_inset,
                    0.0,
                )
            )
    else:
        segments.append(
            (
                np.array((lo[0], lo[1] - margin, hi[2] + margin)),
                np.array((hi[0], lo[1] - margin, hi[2] + margin)),
                f"{dimensions_mm[0]:g} mm",
                label_offsets[0],
                (35, 47, 55),
                arrow_offsets[0],
                0.0,
                minimum_lengths[0],
            )
        )
    segments.extend(
        (
        (
            np.array((cross_x, lo[1], hi[2] + margin)),
            np.array((cross_x, hi[1], hi[2] + margin)),
            f"{dimensions_mm[1]:g} mm",
            label_offsets[1],
            (35, 47, 55),
            arrow_offsets[1],
            0.0,
            minimum_lengths[1],
        ),
        (
            np.array((cross_x, hi[1] + margin, lo[2])),
            np.array((cross_x, hi[1] + margin, hi[2])),
            f"{dimensions_mm[2]:g} mm",
            label_offsets[2],
            (35, 47, 55),
            arrow_offsets[2],
            0.0,
            minimum_lengths[2],
        ),
        )
    )

    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    font = ImageFont.truetype(font_path, 42)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for start, end, label, label_offset, color, arrow_offset, endpoint_inset, minimum_length in segments:
        world = _transform_points(np.stack((start, end)), object_transform)
        screen = _project(world, camera_pose, yfov, width, height) + np.asarray(arrow_offset)
        _draw_arrow(
            draw,
            screen[0],
            screen[1],
            label,
            font,
            label_offset,
            color,
            endpoint_inset,
            minimum_length,
        )
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def _render_task(task: str, cfg: dict, output: Path, width: int, height: int) -> None:
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import pyrender
    import trimesh

    scene = pyrender.Scene(
        bg_color=(0.965, 0.965, 0.955, 1.0),
        ambient_light=(0.28, 0.28, 0.28),
    )

    axis, degrees, position = cfg["receptacle_transform"]
    receptacle_transform = _translation(position) @ _rotation(axis, degrees)
    _add_trimesh_scene(scene, cfg["receptacle"], receptacle_transform)

    # Registered insertion quaternion is -90 degrees about Y (xyzw =
    # [0, -sqrt(1/2), 0, sqrt(1/2)]). Add a small gap above the exact
    # pre-insert pose so both the entry feature and receptive opening read.
    object_transform = (
        receptacle_transform
        @ _translation((0.0, 0.0, cfg["preinsert_z"] + cfg["preinsert_gap"]))
        @ _rotation("y", -90.0)
    )
    object_bounds = _add_trimesh_scene(scene, cfg["object"], object_transform)

    floor = trimesh.creation.box(extents=(1.0, 1.0, 0.002))
    floor.visual.face_colors = np.array((238, 239, 237, 255), dtype=np.uint8)
    scene.add(pyrender.Mesh.from_trimesh(floor, smooth=False), pose=_translation((0, 0, -0.0015)))

    eye = np.asarray(cfg["camera_eye"], dtype=float)
    target = np.asarray(cfg["camera_target"], dtype=float)
    camera_pose = _camera_pose(eye, target)
    yfov = math.radians(float(cfg["yfov_deg"]))
    camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=width / height)
    scene.add(camera, pose=camera_pose)

    key_light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.1)
    scene.add(key_light, pose=_camera_pose(np.array((0.2, -0.3, 0.6)), target))
    fill_light = pyrender.DirectionalLight(color=np.array((0.82, 0.88, 1.0)), intensity=0.9)
    scene.add(fill_light, pose=_camera_pose(np.array((-0.4, 0.2, 0.35)), target))

    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    flags = pyrender.RenderFlags.RGBA
    color, _ = renderer.render(scene, flags=flags)
    renderer.delete()

    image = Image.fromarray(color).convert("RGB")
    image = _annotate_dimensions(
        image,
        object_bounds,
        object_transform,
        cfg["dimensions_mm"],
        cfg["dimension_label_offsets"],
        cfg.get("dimension_arrow_offsets", ((0.0, 0.0),) * 3),
        cfg.get("dimension_min_lengths", (0.0, 0.0, 0.0)),
        cfg["cross_section_fraction"],
        cfg.get("split_length_dimensions"),
        camera_pose,
        yfov,
    )
    image.save(output, quality=95)
    print(f"wrote {task}: {output} ({width}x{height})")


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for task, cfg in TASKS.items():
        _render_task(task, cfg, output_dir / f"{task}_problem_overview.png", args.width, args.height)


if __name__ == "__main__":
    main()
