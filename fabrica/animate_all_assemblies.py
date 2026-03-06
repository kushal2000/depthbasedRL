#!/usr/bin/env python3
"""Animate all Fabrica assemblies simultaneously from exploded to assembled.

Uses per-part frame handles with position updates — no mesh recreation during animation.

Usage:
    python fabrica/animate_all_assemblies.py
    python fabrica/animate_all_assemblies.py --port 8082 --duration 3.0
"""

import argparse
import time

import numpy as np
import viser

from fabrica.viser_utils import (
    ASSETS_DIR, COLORS, compute_explode_offsets, load_all_assemblies,
)


def ease_in_out(t):
    return t * t * (3.0 - 2.0 * t)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--spread", type=float, default=0.10)
    parser.add_argument("--assets-dir", default=ASSETS_DIR)
    args = parser.parse_args()

    print("Loading meshes...")
    assembly_parts = load_all_assemblies(args.assets_dir)

    # Precompute explode offsets
    assembly_offsets = {
        name: compute_explode_offsets(parts, spread=args.spread)
        for name, parts in assembly_parts.items()
    }

    # Grid layout: 2 rows (4 on top, 3 on bottom)
    cols = 4
    x_spacing, y_spacing = 0.35, 0.35
    grid_positions = {}
    for idx, name in enumerate(assembly_parts):
        grid_positions[name] = (
            (idx % cols) * x_spacing,
            (idx // cols) * y_spacing,
        )

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    is_animating = False

    # Create all scene elements once. Each part gets a parent frame whose
    # position we update during animation (no mesh recreation needed).
    part_frames = {}  # (assembly_name, part_idx) -> frame handle
    for name, parts in assembly_parts.items():
        gx, gy = grid_positions[name]

        # Label
        server.scene.add_label(
            f"/{name}/label", text=name, wxyz=(1, 0, 0, 0), position=(gx, gy, -0.05),
        )

        for i, (part_id, mesh) in enumerate(parts):
            # Parent frame that we'll move during animation
            frame = server.scene.add_frame(
                f"/{name}/frame_{part_id}",
                wxyz=(1, 0, 0, 0),
                position=(gx, gy, 0),
                show_axes=False,
            )
            part_frames[(name, i)] = frame

            # Mesh as child of frame (static, never recreated)
            server.scene.add_mesh_simple(
                f"/{name}/frame_{part_id}/mesh",
                vertices=np.array(mesh.vertices, dtype=np.float32),
                faces=np.array(mesh.faces, dtype=np.uint32),
                color=COLORS[i % len(COLORS)],
            )

    server.scene.add_frame("/origin", axes_length=0.05, axes_radius=0.002)

    def set_positions(t):
        """Update frame positions. t=0: exploded, t=1: assembled."""
        alpha = ease_in_out(t)
        for name, parts in assembly_parts.items():
            offsets = assembly_offsets[name]
            gx, gy = grid_positions[name]
            for i, offset in enumerate(offsets):
                displaced = offset * (1.0 - alpha)
                part_frames[(name, i)].position = (
                    gx + displaced[0],
                    gy + displaced[1],
                    displaced[2],
                )

    def animate(forward=True):
        nonlocal is_animating
        if is_animating:
            return
        is_animating = True

        n_frames = int(args.duration * args.fps)
        for frame in range(n_frames + 1):
            t = frame / n_frames
            set_positions(t if forward else 1.0 - t)
            time.sleep(1.0 / args.fps)

        is_animating = False

    assemble_btn = server.gui.add_button("Assemble")
    explode_btn = server.gui.add_button("Explode")
    assemble_btn.on_click(lambda _: animate(forward=True))
    explode_btn.on_click(lambda _: animate(forward=False))

    # Start exploded
    set_positions(0.0)

    print(f"\nOpen http://localhost:{args.port}")
    print("Click Assemble/Explode to animate all assemblies. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
