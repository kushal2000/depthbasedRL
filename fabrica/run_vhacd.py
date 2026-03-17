#!/usr/bin/env python3
"""Run VHACD convex decomposition on Fabrica assembly parts.

Generates hull OBJs and a URDF where each hull is both visual and collision,
so IsaacGym renders the actual collision geometry.

Usage:
    python fabrica/run_vhacd.py --assembly beam --part 2
    python fabrica/run_vhacd.py --assembly beam --part 6
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import trimesh
import tyro

from fabrica.viser_utils import ASSETS_DIR
from fabrica.run_coacd import COACD_URDF_TEMPLATE, HULL_LINK_TEMPLATE, HULL_COLORS


def run_vhacd(mesh_path: Path, output_dir: Path) -> List[trimesh.Trimesh]:
    """Run VHACD on a mesh via trimesh and return convex hulls."""
    assert mesh_path.exists(), f"Mesh file {mesh_path} does not exist"

    output_dir.mkdir(parents=True, exist_ok=True)

    input_mesh = trimesh.load(mesh_path, force="mesh")
    print(f"Input mesh: {len(input_mesh.vertices)} verts, {len(input_mesh.faces)} faces")

    print("Running VHACD decomposition...")
    hulls = trimesh.interfaces.vhacd.convex_decomposition(input_mesh)
    if isinstance(hulls, trimesh.Trimesh):
        hulls = [hulls]
    else:
        hulls = list(hulls)

    for i, hull in enumerate(hulls):
        filename = output_dir / f"decomp_{i}.obj"
        print(f"Saving hull {i} to {filename} ({len(hull.vertices)} verts)")
        hull.export(filename)

    print(f"VHACD complete. {len(hulls)} hulls saved to {output_dir}.")
    return hulls


def generate_vhacd_urdf(assembly: str, part: str, output_dir: Path) -> Path:
    """Generate a multi-body URDF where each VHACD hull is both visual and collision."""
    decomp_files = sorted(output_dir.glob("decomp_*.obj"))
    assert len(decomp_files) > 0, f"No decomp files found in {output_dir}"

    n = len(decomp_files)
    materials = "\n".join(
        f'  <material name="color_{i}">'
        f'<color rgba="{HULL_COLORS[i % len(HULL_COLORS)][0]} '
        f'{HULL_COLORS[i % len(HULL_COLORS)][1]} '
        f'{HULL_COLORS[i % len(HULL_COLORS)][2]} '
        f'{HULL_COLORS[i % len(HULL_COLORS)][3]}"/>'
        f'</material>'
        for i in range(n)
    )
    hull_links = "\n".join(
        HULL_LINK_TEMPLATE.format(i=i) for i in range(n)
    )

    name = f"{assembly}_{part}_vhacd"
    urdf_content = COACD_URDF_TEMPLATE.format(
        name=name,
        materials=materials,
        hull_links=hull_links,
    )

    urdf_path = output_dir / f"{assembly}_{part}_vhacd.urdf"
    urdf_path.write_text(urdf_content)
    print(f"Generated URDF: {urdf_path}")
    return urdf_path


@dataclass
class RunVhacdArgs:
    assembly: str = "beam"
    """Assembly name (e.g., beam)."""

    part: str = "2"
    """Part ID (e.g., 2 or 6)."""

    assets_dir: Path = ASSETS_DIR
    """Root assets directory."""


def main():
    args: RunVhacdArgs = tyro.cli(RunVhacdArgs)
    mesh_path = args.assets_dir / args.assembly / args.part / f"{args.part}_canonical.obj"
    output_dir = args.assets_dir / args.assembly / args.part / "vhacd"

    print(f"Input mesh: {mesh_path}")
    print(f"Output dir: {output_dir}")

    run_vhacd(mesh_path=mesh_path, output_dir=output_dir)
    generate_vhacd_urdf(args.assembly, args.part, output_dir)


if __name__ == "__main__":
    main()
