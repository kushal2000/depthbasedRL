#!/usr/bin/env python3
"""Generate trajectory JSONs and simple environment URDFs for Fabrica assemblies.

For each part in each assembly, generates:
  - fabrica/trajectories/{assembly}_{part_id}/pose_reach.json
  - assets/urdf/fabrica/environments/{assembly}_{part_id}/pose_reach.urdf

Usage:
    python fabrica/generate_trajectories.py
    python fabrica/generate_trajectories.py --assemblies car duct
"""

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

ALL_ASSEMBLIES = [
    "car",
    "cooling_manifold",
    "duct",
    "gamepad",
    "plumbers_block",
    "stool_circular",
]

REPO_ROOT = Path(__file__).resolve().parent.parent
TABLE_Z = 0.38
Z_OFFSET = 0.03

# Simple table-only environment URDF (no fixture)
ENV_URDF_TEMPLATE = """\
<?xml version="1.0"?>
<robot name="table_{assembly}_{part_id}_pose_reach">
  <link name="box">
    <visual>
      <material name="wood">
        <color rgba="0.82 0.56 0.35 1.0"/>
      </material>
      <origin xyz="0 0 0"/>
      <geometry>
        <box size="0.475 0.4 0.3"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0"/>
      <geometry>
        <box size="0.475 0.4 0.3"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="500"/>
      <friction value="1.0"/>
      <inertia ixx="1000.0" ixy="0.0" ixz="0.0" iyy="1000.0" iyz="0.0" izz="1000.0"/>
    </inertial>
  </link>
</robot>
"""


def generate_for_assembly(
    assembly: str,
    assets_dir: Path,
    traj_dir: Path,
    env_dir: Path,
    force: bool,
):
    assembly_asset_dir = assets_dir / assembly
    if not assembly_asset_dir.exists():
        print(f"  ERROR: {assembly_asset_dir} does not exist")
        return 0

    part_dirs = sorted(
        [d for d in assembly_asset_dir.iterdir() if d.is_dir() and d.name != "fixture"],
        key=lambda d: int(d.name),
    )

    count = 0
    for i, part_dir in enumerate(part_dirs):
        part_id = part_dir.name
        name = f"{assembly}_{part_id}"

        # Load mesh to compute placement
        obj_path = part_dir / f"{part_id}.obj"
        if not obj_path.exists():
            print(f"  WARNING: Missing {obj_path}")
            continue

        mesh = trimesh.load_mesh(str(obj_path), process=False)
        extents = mesh.extents
        half_z = extents[2] / 2.0

        # Start pose: spread parts along x, sitting on table
        x_start = 0.05 + 0.04 * (i % 3)
        y_start = -0.05 + 0.04 * (i // 3)
        z_start = TABLE_Z + Z_OFFSET + half_z
        start_pose = [
            round(x_start, 4),
            round(y_start, 4),
            round(z_start, 4),
            0.0, 0.0, 0.0, 1.0,  # identity quaternion
        ]

        # Goal pose: centered above table
        z_goal = TABLE_Z + Z_OFFSET + half_z + 0.12
        goal_pose = [
            round(-0.08, 4),
            round(0.04, 4),
            round(z_goal, 4),
            0.0, 0.0, 0.0, 1.0,
        ]

        trajectory = {
            "start_pose": start_pose,
            "goals": [goal_pose],
        }

        # Write trajectory JSON
        traj_part_dir = traj_dir / name
        traj_file = traj_part_dir / "pose_reach.json"
        if traj_file.exists() and not force:
            print(f"  Skipping trajectory {name} (exists)")
        else:
            traj_part_dir.mkdir(parents=True, exist_ok=True)
            traj_file.write_text(json.dumps(trajectory, indent=4) + "\n")
            print(f"  Trajectory: {name} (start z={z_start:.4f}, goal z={z_goal:.4f})")

        # Write environment URDF
        env_part_dir = env_dir / name
        env_file = env_part_dir / "pose_reach.urdf"
        if env_file.exists() and not force:
            print(f"  Skipping env URDF {name} (exists)")
        else:
            env_part_dir.mkdir(parents=True, exist_ok=True)
            env_file.write_text(
                ENV_URDF_TEMPLATE.format(assembly=assembly, part_id=part_id)
            )

        count += 1

    return count


def main():
    parser = argparse.ArgumentParser(description="Generate Fabrica trajectories")
    parser.add_argument(
        "--assemblies",
        nargs="+",
        default=ALL_ASSEMBLIES,
        choices=ALL_ASSEMBLIES,
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=REPO_ROOT / "assets" / "urdf" / "fabrica",
    )
    parser.add_argument(
        "--traj-dir",
        type=Path,
        default=REPO_ROOT / "fabrica" / "trajectories",
    )
    parser.add_argument(
        "--env-dir",
        type=Path,
        default=REPO_ROOT / "assets" / "urdf" / "fabrica" / "environments",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    total = 0
    for assembly in args.assemblies:
        print(f"Generating trajectories for {assembly}...")
        n = generate_for_assembly(
            assembly, args.assets_dir, args.traj_dir, args.env_dir, args.force
        )
        total += n

    print(f"\nDone. Generated trajectories for {total} parts.")


if __name__ == "__main__":
    main()
