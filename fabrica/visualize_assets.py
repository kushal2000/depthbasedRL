#!/usr/bin/env python3
"""Interactive viser viewer for Fabrica assemblies with dropdown selector.

Usage:
    python fabrica/visualize_assets.py
    python fabrica/visualize_assets.py --port 8081
"""

import argparse
import time

import numpy as np
import viser

from fabrica.viser_utils import (
    ASSETS_DIR, COLORS, SceneManager, compute_explode_offsets, load_all_assemblies,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--assets-dir", default=ASSETS_DIR)
    args = parser.parse_args()

    print("Loading meshes...")
    assembly_parts = load_all_assemblies(args.assets_dir)

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    scene = SceneManager()

    def show_assembly(name):
        scene.clear()
        parts = assembly_parts[name]
        offsets = compute_explode_offsets(parts)

        for i, (part_id, mesh) in enumerate(parts):
            verts = np.array(mesh.vertices, dtype=np.float32)
            faces = np.array(mesh.faces, dtype=np.uint32)
            color = COLORS[i % len(COLORS)]

            scene.add(server.scene.add_mesh_simple(
                f"/assembled/part_{part_id}", vertices=verts, faces=faces, color=color,
            ))

            exploded_verts = verts.copy()
            exploded_verts += offsets[i].astype(np.float32)
            exploded_verts[:, 0] += 0.30
            scene.add(server.scene.add_mesh_simple(
                f"/exploded/part_{part_id}", vertices=exploded_verts, faces=faces, color=color,
            ))

        scene.add(server.scene.add_label("/assembled/label", text=f"{name} (assembled)", wxyz=(1,0,0,0), position=(0,0,-0.05)))
        scene.add(server.scene.add_label("/exploded/label", text=f"{name} (exploded)", wxyz=(1,0,0,0), position=(0.30,0,-0.05)))
        scene.add(server.scene.add_frame("/origin", axes_length=0.05, axes_radius=0.002))
        print(f"  Showing {name}: {len(parts)} parts")

    dropdown = server.gui.add_dropdown("Assembly", options=list(assembly_parts.keys()), initial_value=list(assembly_parts.keys())[0])
    dropdown.on_update(lambda _: show_assembly(dropdown.value))
    show_assembly(dropdown.value)

    print(f"\nOpen http://localhost:{args.port}")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
