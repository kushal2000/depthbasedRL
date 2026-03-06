#!/usr/bin/env python3
"""Animate a single Fabrica assembly from exploded to assembled state in viser.

Usage:
    python fabrica/animate_assembly.py
    python fabrica/animate_assembly.py --port 8081 --duration 3.0
"""

import argparse
import time

import numpy as np
import viser

from fabrica.viser_utils import (
    ASSETS_DIR, COLORS, SceneManager, compute_explode_offsets, load_all_assemblies,
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

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    mesh_scene = SceneManager()  # tracks per-assembly scene elements for switching
    part_frames = []  # frame handles for current assembly
    current_offsets = []
    is_animating = False

    def show_assembly(name):
        nonlocal part_frames, current_offsets
        mesh_scene.clear()
        part_frames = []
        parts = assembly_parts[name]
        current_offsets = compute_explode_offsets(parts, spread=args.spread)

        for i, (part_id, mesh) in enumerate(parts):
            frame = server.scene.add_frame(
                f"/assembly/frame_{part_id}",
                wxyz=(1, 0, 0, 0),
                position=(0, 0, 0),
                show_axes=False,
            )
            mesh_scene.add(frame)
            part_frames.append(frame)

            h = server.scene.add_mesh_simple(
                f"/assembly/frame_{part_id}/mesh",
                vertices=np.array(mesh.vertices, dtype=np.float32),
                faces=np.array(mesh.faces, dtype=np.uint32),
                color=COLORS[i % len(COLORS)],
            )
            mesh_scene.add(h)

        mesh_scene.add(server.scene.add_frame("/origin", axes_length=0.05, axes_radius=0.002))
        set_positions(0.0)

    def set_positions(t):
        alpha = ease_in_out(t)
        for frame, offset in zip(part_frames, current_offsets):
            displaced = offset * (1.0 - alpha)
            frame.position = (displaced[0], displaced[1], displaced[2])

    def animate(forward=True):
        nonlocal is_animating
        if is_animating:
            return
        is_animating = True

        n_frames = int(args.duration * args.fps)
        for frame_idx in range(n_frames + 1):
            t = frame_idx / n_frames
            set_positions(t if forward else 1.0 - t)
            time.sleep(1.0 / args.fps)

        is_animating = False

    dropdown = server.gui.add_dropdown("Assembly", options=list(assembly_parts.keys()), initial_value=list(assembly_parts.keys())[0])
    assemble_btn = server.gui.add_button("Assemble")
    explode_btn = server.gui.add_button("Explode")

    dropdown.on_update(lambda _: show_assembly(dropdown.value))
    assemble_btn.on_click(lambda _: animate(forward=True))
    explode_btn.on_click(lambda _: animate(forward=False))

    show_assembly(dropdown.value)

    print(f"\nOpen http://localhost:{args.port}")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
