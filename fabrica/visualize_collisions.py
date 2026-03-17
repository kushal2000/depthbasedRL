#!/usr/bin/env python3
"""Side-by-side viser viewer comparing collision representations.

Shows 4 columns: Original mesh, CoACD hulls, VHACD hulls, SDF zero-level surface.

Usage:
    python fabrica/visualize_collisions.py --part 2
    python fabrica/visualize_collisions.py --part 6 --port 8082
"""

import argparse
import time
from pathlib import Path

import numpy as np
import trimesh
import viser

from fabrica.viser_utils import ASSETS_DIR, COLORS, SceneManager


COLUMN_SPACING = 0.2
COLUMN_LABELS = ["Original", "CoACD", "VHACD", "SDF"]


def load_original_mesh(assets_dir: Path, assembly: str, part: str) -> trimesh.Trimesh:
    mesh_path = assets_dir / assembly / part / f"{part}_canonical.obj"
    return trimesh.load_mesh(str(mesh_path), process=False)


def load_coacd_hulls(assets_dir: Path, assembly: str, part: str) -> list:
    coacd_dir = assets_dir / assembly / part / "coacd"
    if not coacd_dir.exists():
        print(f"WARNING: CoACD dir not found: {coacd_dir}")
        print("  Run: python fabrica/run_coacd.py --part", part)
        return []
    decomp_files = sorted(coacd_dir.glob("decomp_*.obj"))
    return [trimesh.load_mesh(str(f), process=False) for f in decomp_files]


def compute_vhacd_hulls(mesh: trimesh.Trimesh) -> list:
    """Use trimesh's VHACD interface to decompose mesh into convex hulls."""
    try:
        hulls = trimesh.interfaces.vhacd.convex_decomposition(mesh)
        if isinstance(hulls, trimesh.Trimesh):
            return [hulls]
        return list(hulls)
    except Exception as e:
        print(f"WARNING: VHACD decomposition failed: {e}")
        print("  Make sure testVHACD is installed (apt install libbullet-extras-dev or similar)")
        return []


def compute_sdf_surface(mesh: trimesh.Trimesh, resolution: int = 128) -> trimesh.Trimesh:
    """Compute SDF zero-level surface via marching cubes."""
    try:
        from pysdf import SDF
        from skimage.measure import marching_cubes
    except ImportError as e:
        print(f"WARNING: Missing dependency for SDF visualization: {e}")
        print("  Install: uv pip install pysdf scikit-image")
        return None

    vertices = np.array(mesh.vertices, dtype=np.float64)
    faces = np.array(mesh.faces, dtype=np.uint32)

    sdf_func = SDF(vertices, faces)

    # Build grid covering mesh bounding box with padding
    bbox_min = vertices.min(axis=0) - 0.01
    bbox_max = vertices.max(axis=0) + 0.01

    x = np.linspace(bbox_min[0], bbox_max[0], resolution)
    y = np.linspace(bbox_min[1], bbox_max[1], resolution)
    z = np.linspace(bbox_min[2], bbox_max[2], resolution)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    query_points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

    sdf_values = -sdf_func(query_points)  # pysdf convention: negative inside
    sdf_grid = sdf_values.reshape(resolution, resolution, resolution)

    try:
        verts_mc, faces_mc, _, _ = marching_cubes(sdf_grid, level=0)
    except ValueError:
        print("WARNING: Marching cubes found no zero-crossing")
        return None

    # Scale marching cubes output back to original coordinates
    scale = (bbox_max - bbox_min) / (resolution - 1)
    verts_mc = verts_mc * scale + bbox_min

    return trimesh.Trimesh(vertices=verts_mc, faces=faces_mc)


def show_part(server, scene: SceneManager, assets_dir: Path, assembly: str, part: str):
    scene.clear()

    original = load_original_mesh(assets_dir, assembly, part)

    # Compute bounding box center for label placement
    center_z = original.vertices[:, 2].max() + 0.02

    # Column 0: Original mesh (gray)
    verts = np.array(original.vertices, dtype=np.float32)
    faces = np.array(original.faces, dtype=np.uint32)
    scene.add(server.scene.add_mesh_simple(
        "/col0/mesh", vertices=verts, faces=faces, color=(0.6, 0.6, 0.6),
    ))
    scene.add(server.scene.add_label(
        "/col0/label", text="Original", wxyz=(1, 0, 0, 0),
        position=(0, 0, center_z),
    ))

    # Column 1: CoACD hulls
    x_offset = COLUMN_SPACING
    coacd_hulls = load_coacd_hulls(assets_dir, assembly, part)
    for i, hull in enumerate(coacd_hulls):
        hv = np.array(hull.vertices, dtype=np.float32)
        hv[:, 0] += x_offset
        hf = np.array(hull.faces, dtype=np.uint32)
        color = COLORS[i % len(COLORS)]
        scene.add(server.scene.add_mesh_simple(
            f"/col1/hull_{i}", vertices=hv, faces=hf, color=color,
        ))
    scene.add(server.scene.add_label(
        "/col1/label", text=f"CoACD ({len(coacd_hulls)} hulls)", wxyz=(1, 0, 0, 0),
        position=(x_offset, 0, center_z),
    ))

    # Column 2: VHACD hulls
    x_offset = COLUMN_SPACING * 2
    print("  Computing VHACD decomposition...")
    vhacd_hulls = compute_vhacd_hulls(original)
    for i, hull in enumerate(vhacd_hulls):
        hv = np.array(hull.vertices, dtype=np.float32)
        hv[:, 0] += x_offset
        hf = np.array(hull.faces, dtype=np.uint32)
        color = COLORS[i % len(COLORS)]
        scene.add(server.scene.add_mesh_simple(
            f"/col2/hull_{i}", vertices=hv, faces=hf, color=color,
        ))
    scene.add(server.scene.add_label(
        "/col2/label", text=f"VHACD ({len(vhacd_hulls)} hulls)", wxyz=(1, 0, 0, 0),
        position=(x_offset, 0, center_z),
    ))

    # Column 3: SDF zero-level surface
    x_offset = COLUMN_SPACING * 3
    print("  Computing SDF surface (marching cubes)...")
    sdf_mesh = compute_sdf_surface(original)
    if sdf_mesh is not None:
        sv = np.array(sdf_mesh.vertices, dtype=np.float32)
        sv[:, 0] += x_offset
        sf = np.array(sdf_mesh.faces, dtype=np.uint32)
        scene.add(server.scene.add_mesh_simple(
            "/col3/mesh", vertices=sv, faces=sf, color=(0.2, 0.8, 0.5),
        ))
    scene.add(server.scene.add_label(
        "/col3/label", text="SDF", wxyz=(1, 0, 0, 0),
        position=(x_offset, 0, center_z),
    ))

    # Origin frame
    scene.add(server.scene.add_frame("/origin", axes_length=0.05, axes_radius=0.002))

    print(f"  Part {part}: original + {len(coacd_hulls)} CoACD + {len(vhacd_hulls)} VHACD + SDF")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=str, default="2")
    parser.add_argument("--assembly", type=str, default="beam")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--assets-dir", type=str, default=str(ASSETS_DIR))
    args = parser.parse_args()

    assets_dir = Path(args.assets_dir)
    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    scene = SceneManager()

    # Find available parts for dropdown
    assembly_dir = assets_dir / args.assembly
    part_dirs = sorted(
        [d.name for d in assembly_dir.iterdir()
         if d.is_dir() and d.name.isdigit()],
        key=int,
    )

    initial_part = args.part if args.part in part_dirs else part_dirs[0]

    dropdown = server.gui.add_dropdown(
        "Part", options=part_dirs, initial_value=initial_part,
    )

    def on_update(_):
        print(f"Switching to part {dropdown.value}...")
        show_part(server, scene, assets_dir, args.assembly, dropdown.value)

    dropdown.on_update(on_update)

    print(f"Loading part {initial_part}...")
    show_part(server, scene, assets_dir, args.assembly, initial_part)

    print(f"\nOpen http://localhost:{args.port}")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
