#!/usr/bin/env python3
"""Generate pick_place trajectory JSONs and per-sub-problem environment URDFs.

For each assembly step n (placing part pn), generates:
  - fabrica/trajectories/{assembly}_{part_id}/pick_place.json
    Multi-waypoint trajectory sampled from crane-style evaluate_trajectory().
  - assets/urdf/fabrica/environments/{assembly}_{part_id}/scene.urdf
    Table URDF with parts p0..pn-1 baked in as static geometry.

Usage:
    python fabrica/generate_trajectories.py --assemblies beam
    python fabrica/generate_trajectories.py --assemblies beam car duct
    python fabrica/generate_trajectories.py  # all assemblies
"""

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

from fabrica.viser_utils import ASSETS_DIR, load_assembly_parts, load_assembly_order
from fabrica.animate_sequential import (
    TABLE_SURFACE_Z,
    DEFAULT_CLEARANCE,
    DEFAULT_LIFT_FRAC,
    DEFAULT_LOWER_FRAC,
    IDENTITY_QUAT,
    compute_assembled_positions,
    compute_start_positions,
    evaluate_trajectory,
    load_canonical_data,
    _slerp,
)

ALL_ASSEMBLIES = [
    "beam",
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
NUM_WAYPOINTS = 12

# Table-only URDF header + footer (completed parts inserted between)
ENV_URDF_HEADER = """\
<?xml version="1.0"?>
<robot name="table_{name}">
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
    </collision>"""

ENV_URDF_FOOTER = """\
    <inertial>
      <mass value="500"/>
      <friction value="1.0"/>
      <inertia ixx="1000.0" ixy="0.0" ixz="0.0" iyy="1000.0" iyz="0.0" izz="1000.0"/>
    </inertial>
  </link>
</robot>
"""


def wxyz_to_xyzw(q):
    """Convert wxyz quaternion to xyzw (Isaac Gym convention)."""
    return [q[1], q[2], q[3], q[0]]


def compute_assembly_offset(parts):
    """Compute the global offset that places the assembly on the table.

    All assembled-frame meshes share the same reference frame, so one offset
    positions them all. Returns the offset vector [ox, oy, oz].
    """
    centroids = [mesh.centroid for _, mesh in parts]
    overall_centroid = np.mean(centroids, axis=0)

    min_bottom = min(
        mesh.centroid[2] - mesh.bounding_box.extents[2] / 2.0 for _, mesh in parts
    )

    target_xy = np.array([-0.08, 0.04])
    offset_xy = target_xy - overall_centroid[:2]
    offset_z = TABLE_SURFACE_Z - min_bottom

    return np.array([offset_xy[0], offset_xy[1], offset_z])


def generate_completed_part_xml(assembly, part_id, origin_xyz):
    """Generate URDF visual+collision XML for a completed part."""
    mesh_path = f"../../{assembly}/{part_id}/{part_id}.obj"
    origin = f"{origin_xyz[0]:.6f} {origin_xyz[1]:.6f} {origin_xyz[2]:.6f}"
    return f"""
    <!-- Completed part {part_id} -->
    <visual>
      <material name="part_{part_id}">
        <color rgba="0.4 0.4 0.6 1.0"/>
      </material>
      <origin xyz="{origin}" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{mesh_path}" scale="1 1 1"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="{origin}" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{mesh_path}" scale="1 1 1"/>
      </geometry>
    </collision>"""


def generate_env_urdf(assembly, step_name, completed_part_ids, offset):
    """Build environment URDF string with completed parts baked in."""
    # URDF origin is relative to table link placed at (0,0,TABLE_Z).
    # All assembled-frame meshes share the same offset from the origin.
    urdf_offset = np.array([offset[0], offset[1], offset[2] - TABLE_Z])

    parts_xml = ""
    for pid in completed_part_ids:
        parts_xml += generate_completed_part_xml(assembly, pid, urdf_offset)

    return (
        ENV_URDF_HEADER.format(name=step_name)
        + parts_xml
        + "\n"
        + ENV_URDF_FOOTER
    )


def generate_trajectory(
    start_pos, start_quat_wxyz,
    end_pos, end_quat_wxyz,
    insertion_dir=None,
    num_waypoints=NUM_WAYPOINTS,
):
    """Sample waypoints from evaluate_trajectory and return pick_place dict.

    Returns dict with start_pose and goals in xyzw quaternion format.
    """
    clearance_z = TABLE_SURFACE_Z + DEFAULT_CLEARANCE
    if insertion_dir is not None:
        clearance_z += 0.05

    # Start pose: canonical position on table, canonical orientation (identity for canonical meshes)
    start_pose = list(start_pos) + wxyz_to_xyzw(start_quat_wxyz)

    # Sample waypoints at uniform t values in (0, 1]
    goals = []
    for i in range(1, num_waypoints + 1):
        t = i / num_waypoints
        pos, quat_wxyz = evaluate_trajectory(
            start_pos, start_quat_wxyz,
            end_pos, end_quat_wxyz,
            t, clearance_z,
            lift_frac=DEFAULT_LIFT_FRAC,
            lower_frac=DEFAULT_LOWER_FRAC,
            insertion_dir=insertion_dir,
        )
        goal = list(pos) + wxyz_to_xyzw(quat_wxyz)
        goals.append([round(v, 6) for v in goal])

    return {
        "start_pose": [round(v, 6) for v in start_pose],
        "goals": goals,
    }


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

    # Load assembled-frame meshes
    parts = load_assembly_parts(assets_dir, assembly)
    if not parts:
        print(f"  ERROR: No parts found for {assembly}")
        return 0

    part_ids = [pid for pid, _ in parts]
    assembly_order, start_rotations, insertion_directions = load_assembly_order(
        assets_dir, assembly, parts
    )

    # Load canonical data
    canonical_meshes, canonical_transforms = load_canonical_data(
        assets_dir, assembly, part_ids
    )
    if canonical_meshes is None:
        print(f"  ERROR: No canonical data for {assembly}, skipping")
        return 0

    # Compute assembled positions (world frame)
    assembled_positions, _ = compute_assembled_positions(parts)

    # Compute assembled quaternions (canonical -> assembled)
    assembled_quats = {}
    for pid in part_ids:
        assembled_quats[pid] = np.array(
            canonical_transforms[pid]["canonical_to_assembled_wxyz"]
        )

    # Compute start positions using canonical meshes (same as viser animation)
    canonical_parts = [(pid, canonical_meshes[pid]) for pid in part_ids]
    start_positions, start_quats = compute_start_positions(
        canonical_parts, assembly_order, {}
    )

    # Compute global assembly offset for URDF placement
    offset = compute_assembly_offset(parts)

    # Insertion directions
    part_insertion_dirs = {}
    for pid in part_ids:
        if pid in insertion_directions:
            part_insertion_dirs[pid] = np.array(insertion_directions[pid], dtype=float)
        else:
            part_insertion_dirs[pid] = None

    count = 0
    for step_idx, pid in enumerate(assembly_order):
        name = f"{assembly}_{pid}"
        completed_parts = assembly_order[:step_idx]

        # --- Trajectory ---
        traj_part_dir = traj_dir / name
        traj_file = traj_part_dir / "pick_place.json"
        if traj_file.exists() and not force:
            print(f"  Skipping trajectory {name} (exists)")
        else:
            traj = generate_trajectory(
                start_positions[pid], start_quats[pid],
                assembled_positions[pid], assembled_quats[pid],
                insertion_dir=part_insertion_dirs[pid],
            )
            traj_part_dir.mkdir(parents=True, exist_ok=True)
            traj_file.write_text(json.dumps(traj, indent=4) + "\n")
            print(f"  Trajectory: {name} ({len(traj['goals'])} waypoints)")

        # --- Environment URDF ---
        env_part_dir = env_dir / name
        env_file = env_part_dir / "scene.urdf"
        if env_file.exists() and not force:
            print(f"  Skipping env URDF {name} (exists)")
        else:
            urdf_str = generate_env_urdf(assembly, name, completed_parts, offset)
            env_part_dir.mkdir(parents=True, exist_ok=True)
            env_file.write_text(urdf_str)
            n_completed = len(completed_parts)
            print(f"  Env URDF: {name} ({n_completed} completed parts)")

        count += 1

    return count


def main():
    parser = argparse.ArgumentParser(
        description="Generate pick_place trajectories and environment URDFs"
    )
    parser.add_argument(
        "--assemblies",
        nargs="+",
        default=ALL_ASSEMBLIES,
        choices=ALL_ASSEMBLIES,
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=ASSETS_DIR,
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
        print(f"Generating for {assembly}...")
        n = generate_for_assembly(
            assembly, args.assets_dir, args.traj_dir, args.env_dir, args.force
        )
        total += n

    print(f"\nDone. Generated for {total} parts.")


if __name__ == "__main__":
    main()
