#!/usr/bin/env python3
"""Run CoACD convex decomposition on Fabrica assembly parts.

Usage:
    python fabrica/run_coacd.py --assembly beam --part 2
    python fabrica/run_coacd.py --assembly beam --part 6
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal

import trimesh
import tyro

from fabrica.viser_utils import ASSETS_DIR


COACD_URDF_TEMPLATE = """\
<?xml version="1.0"?>
<robot name="{name}">
{materials}
  <link name="base_link">
    <inertial>
      <density value="1250.0"/>
    </inertial>
  </link>
{hull_links}
</robot>
"""

HULL_COLORS = [
    (0.9, 0.2, 0.2, 1.0),
    (0.2, 0.7, 0.2, 1.0),
    (0.2, 0.3, 0.9, 1.0),
    (0.9, 0.7, 0.1, 1.0),
    (0.8, 0.3, 0.8, 1.0),
    (0.1, 0.8, 0.8, 1.0),
    (0.9, 0.5, 0.1, 1.0),
    (0.5, 0.5, 0.5, 1.0),
    (0.6, 0.2, 0.6, 1.0),
    (0.3, 0.9, 0.5, 1.0),
    (0.9, 0.3, 0.5, 1.0),
    (0.4, 0.6, 0.9, 1.0),
    (0.7, 0.9, 0.2, 1.0),
    (0.9, 0.4, 0.7, 1.0),
    (0.2, 0.5, 0.7, 1.0),
    (0.8, 0.6, 0.3, 1.0),
    (0.5, 0.3, 0.9, 1.0),
    (0.3, 0.8, 0.3, 1.0),
    (0.7, 0.2, 0.4, 1.0),
    (0.4, 0.7, 0.6, 1.0),
]

HULL_LINK_TEMPLATE = """\
  <link name="hull_{i}">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="decomp_{i}.obj" scale="1 1 1"/>
      </geometry>
      <material name="color_{i}"/>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="decomp_{i}.obj" scale="1 1 1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.001"/>
      <inertia ixx="1e-6" ixy="0" ixz="0" iyy="1e-6" iyz="0" izz="1e-6"/>
    </inertial>
  </link>
  <joint name="hull_joint_{i}" type="fixed">
    <parent link="base_link"/>
    <child link="hull_{i}"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>"""


def run_coacd(
    mesh_path: Path,
    output_dir: Path,
    max_convex_hull: int = -1,
    threshold: float = 0.03,
    preprocess_resolution: int = 50,
    resolution: int = 3000,
    mcts_nodes: int = 25,
    mcts_iterations: int = 250,
    mcts_max_depth: int = 4,
    mode: Literal["subprocess", "python"] = "python",
) -> List[trimesh.Trimesh]:
    """Run COACD on a mesh and return the list of convex hulls."""
    assert mesh_path.exists(), f"Mesh file {mesh_path} does not exist"
    assert mesh_path.suffix == ".obj", f"Mesh file {mesh_path} is not an OBJ file"

    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if mode == "subprocess":
        from subprocess import run as run_cmd

        output_mesh_path = output_dir / mesh_path.name
        cmd = f"coacd -i {mesh_path} -o {output_mesh_path} -c {max_convex_hull}"
        print(f"Running command: {cmd}")
        run_cmd(cmd, shell=True, check=True)

        assert output_mesh_path.exists(), (
            f"Output mesh file {output_mesh_path} does not exist"
        )
        output_mesh = trimesh.load(output_mesh_path)
        parts = output_mesh.split()
        for i, part in enumerate(parts):
            filename = output_dir / f"decomp_{i}.obj"
            print(f"Saving part {i} to {filename}")
            part.export(filename)
        print(
            f"Decomposition complete. {len(parts)} parts found and saved to {output_dir}."
        )
        return parts
    elif mode == "python":
        import coacd
        import numpy as np

        input_mesh = trimesh.load(mesh_path, force="mesh")
        coacd_mesh = coacd.Mesh(input_mesh.vertices, input_mesh.faces)
        convex_vs_fs_parts = coacd.run_coacd(
            coacd_mesh,
            threshold=threshold,
            max_convex_hull=max_convex_hull,
            preprocess_resolution=preprocess_resolution,
            resolution=resolution,
            mcts_nodes=mcts_nodes,
            mcts_iterations=mcts_iterations,
            mcts_max_depth=mcts_max_depth,
        )
        parts = []
        for vs, fs in convex_vs_fs_parts:
            parts.append(trimesh.Trimesh(vs, fs))

        np.random.seed(0)
        scene = trimesh.Scene()
        for part in parts:
            part.visual.vertex_colors[:, :3] = (np.random.rand(3) * 255).astype(
                np.uint8
            )
            scene.add_geometry(part)
        output_mesh_path = output_dir / mesh_path.name
        print(f"Saving scene to {output_mesh_path}")
        scene.export(output_mesh_path)

        for i, part in enumerate(parts):
            filename = output_dir / f"decomp_{i}.obj"
            print(f"Saving part {i} to {filename}")
            part.export(filename)
        print(
            f"Decomposition complete. {len(parts)} parts found and saved to {output_dir}."
        )
        return parts
    else:
        raise ValueError(f"Invalid mode: {mode}")


def generate_coacd_urdf(assembly: str, part: str, output_dir: Path) -> Path:
    """Generate a multi-body URDF where each hull is both visual and collision.

    Each hull link gets a distinct color so you can see the decomposition
    in IsaacGym's rendered output.
    """
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

    name = f"{assembly}_{part}_coacd"
    urdf_content = COACD_URDF_TEMPLATE.format(
        name=name,
        materials=materials,
        hull_links=hull_links,
    )

    urdf_path = output_dir / f"{assembly}_{part}_coacd.urdf"
    urdf_path.write_text(urdf_content)
    print(f"Generated URDF: {urdf_path}")
    return urdf_path


@dataclass
class RunCoacdArgs:
    assembly: str = "beam"
    """Assembly name (e.g., beam)."""

    part: str = "2"
    """Part ID (e.g., 2 or 6)."""

    max_convex_hull: int = -1
    """Maximum number of convex hulls (-1 for unlimited)."""

    threshold: float = 0.03
    """Concavity threshold (lower = tighter fit). Default 0.03 for thin parts."""

    preprocess_resolution: int = 50
    """Voxel resolution for manifold preprocessing."""

    resolution: int = 3000
    """Sampling resolution for Hausdorff distance calculation."""

    mcts_nodes: int = 25
    """MCTS child nodes per level."""

    mcts_iterations: int = 250
    """MCTS search iterations."""

    mcts_max_depth: int = 4
    """MCTS search tree depth."""

    mode: Literal["subprocess", "python"] = "python"
    """COACD execution mode."""

    assets_dir: Path = ASSETS_DIR
    """Root assets directory."""


def main():
    args: RunCoacdArgs = tyro.cli(RunCoacdArgs)
    mesh_path = args.assets_dir / args.assembly / args.part / f"{args.part}_canonical.obj"
    output_dir = args.assets_dir / args.assembly / args.part / "coacd"

    print(f"Input mesh: {mesh_path}")
    print(f"Output dir: {output_dir}")

    run_coacd(
        mesh_path=mesh_path,
        output_dir=output_dir,
        max_convex_hull=args.max_convex_hull,
        threshold=args.threshold,
        preprocess_resolution=args.preprocess_resolution,
        resolution=args.resolution,
        mcts_nodes=args.mcts_nodes,
        mcts_iterations=args.mcts_iterations,
        mcts_max_depth=args.mcts_max_depth,
        mode=args.mode,
    )

    generate_coacd_urdf(args.assembly, args.part, output_dir)


if __name__ == "__main__":
    main()
