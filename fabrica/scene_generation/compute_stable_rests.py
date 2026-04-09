#!/usr/bin/env python3
"""Enumerate stable resting orientations for each part of an assembly.

For each part, loads its canonical OBJ mesh and uses trimesh's
compute_stable_poses to find the discrete set of physically stable
orientations the part can rest in on a flat surface.

The resulting rests + their settle probabilities are saved to:
  assets/urdf/fabrica/{assembly}/stable_rests/{part_id}.npz

Each .npz contains:
  transforms: (N, 4, 4) float64 — homogeneous transforms placing the part
                                  in canonical frame onto a z=0 plane in
                                  the i-th stable orientation.
  probabilities: (N,) float64    — quasi-static probability of settling
                                  into each rest under random drops.

Usage:
    python fabrica/scene_generation/compute_stable_rests.py --assembly beam
    python fabrica/scene_generation/compute_stable_rests.py --assembly beam --part 2
    python fabrica/scene_generation/compute_stable_rests.py --assembly beam --force
"""

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ASSETS_DIR = REPO_ROOT / "assets" / "urdf" / "fabrica"


def load_assembly_parts(assembly):
    """Return the list of part ids in assembly_order for the given assembly."""
    order_path = ASSETS_DIR / assembly / "assembly_order.json"
    with open(order_path) as f:
        order = json.load(f)
    return order["steps"]


def compute_part_stable_rests(assembly, part_id):
    """Run trimesh.compute_stable_poses on a single part's canonical mesh."""
    mesh_path = ASSETS_DIR / assembly / part_id / f"{part_id}_canonical.obj"
    mesh = trimesh.load_mesh(str(mesh_path), process=False)

    transforms, probabilities = trimesh.poses.compute_stable_poses(mesh)
    return transforms, probabilities, mesh


def save_stable_rests(assembly, part_id, transforms, probabilities):
    """Save the stable rest set to assets/.../stable_rests/{part_id}.npz."""
    out_dir = ASSETS_DIR / assembly / "stable_rests"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{part_id}.npz"
    np.savez(
        out_path,
        transforms=np.asarray(transforms, dtype=np.float64),
        probabilities=np.asarray(probabilities, dtype=np.float64),
    )
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Compute stable resting orientations for assembly parts.",
    )
    parser.add_argument("--assembly", type=str, required=True,
                        help="Assembly name (e.g. 'beam').")
    parser.add_argument("--part", type=str, default=None,
                        help="Optional single part id to process. "
                             "Defaults to all parts in assembly_order.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing stable_rests/{part_id}.npz files.")
    args = parser.parse_args()

    if args.part is not None:
        parts = [args.part]
    else:
        parts = load_assembly_parts(args.assembly)

    print(f"Assembly: {args.assembly}")
    print(f"Parts: {parts}\n")

    for pid in parts:
        out_path = ASSETS_DIR / args.assembly / "stable_rests" / f"{pid}.npz"
        if out_path.exists() and not args.force:
            print(f"  Skipping part {pid} (exists at {out_path}, use --force)")
            continue

        transforms, probabilities, mesh = compute_part_stable_rests(args.assembly, pid)
        n = len(transforms)
        if n == 0:
            print(f"  WARNING part {pid}: trimesh returned 0 stable poses")
            continue

        saved_to = save_stable_rests(args.assembly, pid, transforms, probabilities)
        bbox = mesh.bounding_box.extents
        print(f"  part {pid}: {n} stable rests "
              f"(bbox {bbox[0]:.3f}x{bbox[1]:.3f}x{bbox[2]:.3f} m)")
        for i, (t, p) in enumerate(zip(transforms, probabilities)):
            print(f"    rest {i}: prob={p:.3f}  z_lift={t[2, 3]:+.4f}")
        print(f"    saved → {saved_to.relative_to(REPO_ROOT)}\n")

    print("Done.")


if __name__ == "__main__":
    main()
