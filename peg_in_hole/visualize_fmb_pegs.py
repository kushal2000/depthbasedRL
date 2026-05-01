#!/usr/bin/env python3
"""Visualize FMB pegs and peg boards in viser for selection.

Displays all pegs in a grid layout and the 3 peg boards below.
Click on items to highlight them.

Usage:
    python peg_in_hole/visualize_fmb_pegs.py --port 8043
"""

import argparse
import math
from pathlib import Path

import numpy as np
import trimesh
import viser

REPO_ROOT = Path(__file__).resolve().parent.parent
PEGS_DIR = REPO_ROOT / "assets" / "urdf" / "fmb" / "pegs"
BOARDS_DIR = REPO_ROOT / "assets" / "urdf" / "fmb" / "boards"
FIXTURE_DIR = REPO_ROOT / "assets" / "urdf" / "fmb" / "fixtures"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8043)
    args = parser.parse_args()

    server = viser.ViserServer(host="0.0.0.0", port=args.port)

    @server.on_client_connect
    def _(client: viser.ClientHandle):
        client.camera.position = (0.5, -0.8, 0.4)
        client.camera.look_at = (0.5, 0.3, 0.0)

    server.scene.add_grid("/ground", width=3, height=3, cell_size=0.1)

    # Layout: pegs in a grid, 10 per row, spaced 0.12m apart
    peg_dirs = sorted(PEGS_DIR.iterdir())
    cols = 10
    spacing = 0.12

    server.gui.add_markdown("# FMB Peg & Board Viewer")
    md_info = server.gui.add_markdown("**Selected:** none")

    for i, peg_dir in enumerate(peg_dirs):
        obj_file = peg_dir / f"{peg_dir.name}.obj"
        if not obj_file.exists():
            continue

        mesh = trimesh.load(str(obj_file), force="mesh")
        # Center the mesh at origin
        centroid = mesh.centroid.copy()
        mesh.vertices -= centroid

        row = i // cols
        col = i % cols
        x = col * spacing
        y = row * spacing

        name = peg_dir.name
        server.scene.add_frame(
            f"/pegs/{name}", position=(x, y, 0), show_axes=False,
        )
        handle = server.scene.add_mesh_simple(
            f"/pegs/{name}/mesh",
            vertices=np.array(mesh.vertices, dtype=np.float32),
            faces=np.array(mesh.faces, dtype=np.uint32),
            color=(200, 80, 80),
        )

        # Add label
        server.scene.add_label(
            f"/pegs/{name}/label",
            text=name,
            position=(0, 0, float(mesh.extents[2] / 2) + 0.01),
        )

    # Boards below the pegs
    board_y_offset = (math.ceil(len(peg_dirs) / cols) + 1) * spacing
    board_dirs = sorted(BOARDS_DIR.iterdir())

    for i, board_dir in enumerate(board_dirs):
        obj_file = board_dir / f"{board_dir.name}.obj"
        if not obj_file.exists():
            continue

        mesh = trimesh.load(str(obj_file), force="mesh")
        centroid = mesh.centroid.copy()
        mesh.vertices -= centroid

        x = i * 0.3
        y = board_y_offset

        name = board_dir.name
        server.scene.add_frame(
            f"/boards/{name}", position=(x, y, 0), show_axes=False,
        )
        server.scene.add_mesh_simple(
            f"/boards/{name}/mesh",
            vertices=np.array(mesh.vertices, dtype=np.float32),
            faces=np.array(mesh.faces, dtype=np.uint32),
            color=(120, 120, 200),
        )
        server.scene.add_label(
            f"/boards/{name}/label",
            text=name,
            position=(0, 0, float(mesh.extents[2] / 2) + 0.01),
        )

    # Fixture
    fixture_obj = FIXTURE_DIR / "peg_fixture" / "peg_fixture.obj"
    if fixture_obj.exists():
        mesh = trimesh.load(str(fixture_obj), force="mesh")
        mesh.vertices -= mesh.centroid
        server.scene.add_frame(
            "/fixture", position=(len(board_dirs) * 0.3, board_y_offset, 0), show_axes=False,
        )
        server.scene.add_mesh_simple(
            "/fixture/mesh",
            vertices=np.array(mesh.vertices, dtype=np.float32),
            faces=np.array(mesh.faces, dtype=np.uint32),
            color=(160, 160, 160),
        )
        server.scene.add_label(
            "/fixture/label",
            text="peg_fixture",
            position=(0, 0, float(mesh.extents[2] / 2) + 0.01),
        )

    # Print summary
    print(f"\n  FMB Peg & Board Viewer   http://localhost:{args.port}")
    print(f"  {len(peg_dirs)} pegs, {len(board_dirs)} boards, 1 fixture")
    print()

    # Print peg dimensions for reference
    print("  Peg dimensions (centered):")
    for peg_dir in peg_dirs:
        obj_file = peg_dir / f"{peg_dir.name}.obj"
        if not obj_file.exists():
            continue
        mesh = trimesh.load(str(obj_file), force="mesh")
        ext = mesh.extents
        print(f"    {peg_dir.name}: {ext[0]*1000:.1f} x {ext[1]*1000:.1f} x {ext[2]*1000:.1f} mm")

    try:
        while True:
            import time
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    main()
