#!/usr/bin/env python3
"""Precompute table URDFs with completed parts baked in for each assembly step.

For each assembly step and each collision method (vhacd, sdf, coacd), generates
a scene URDF with all previously-completed parts as fixed collision geometry.

Usage:
    python fabrica/generate_table_urdfs.py
    python fabrica/generate_table_urdfs.py --assembly car
"""

import argparse
import json
import math
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "assets" / "urdf" / "fabrica"
TABLE_Z = 0.38


def _load_assembly_order(assembly: str) -> List[str]:
    path = ASSETS_DIR / assembly / "assembly_order.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())["steps"]


def _load_trajectory(assembly: str, pid: str):
    path = REPO_ROOT / "fabrica" / "trajectories" / f"{assembly}_{pid}" / "pick_place.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _quat_xyzw_to_rpy(q) -> Tuple[float, float, float]:
    from scipy.spatial.transform import Rotation
    r = Rotation.from_quat([float(q[0]), float(q[1]), float(q[2]), float(q[3])])
    roll, pitch, yaw = r.as_euler('xyz')
    return roll, pitch, yaw


TABLE_LINK = """\
  <link name="box">
    <visual>
      <material name="wood"><color rgba="0.82 0.56 0.35 1.0"/></material>
      <origin xyz="0 0 0"/>
      <geometry><box size="0.475 0.4 0.3"/></geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0"/>
      <geometry><box size="0.475 0.4 0.3"/></geometry>
    </collision>
    <inertial>
      <mass value="500"/>
      <friction value="1.0"/>
      <inertia ixx="1000.0" ixy="0.0" ixz="0.0" iyy="1000.0" iyz="0.0" izz="1000.0"/>
    </inertial>
  </link>"""


def _part_goal_offset(assembly: str, pid: str):
    """Return (rx, ry, rz, roll, pitch, yaw) for a completed part relative to table."""
    traj = _load_trajectory(assembly, pid)
    if traj is None:
        return None
    goal_pose = traj["goals"][-1]
    rx, ry, rz = goal_pose[0], goal_pose[1], goal_pose[2] - TABLE_Z
    rpy = _quat_xyzw_to_rpy(goal_pose[3:7])
    return rx, ry, rz, rpy[0], rpy[1], rpy[2]


def _generate_vhacd_table(assembly: str, active_pid: str, completed_pids: List[str],
                          env_dir: Path):
    """VHACD table: single link per completed part with raw mesh (IsaacGym applies VHACD)."""
    lines = [
        '<?xml version="1.0"?>',
        f'<robot name="table_{assembly}_{active_pid}_scene">',
        TABLE_LINK,
    ]

    for pid in completed_pids:
        offset = _part_goal_offset(assembly, pid)
        if offset is None:
            continue
        rx, ry, rz, roll, pitch, yaw = offset
        mesh_rel = f"../../{assembly}/{pid}/{pid}_canonical.obj"

        lines.extend([
            f'  <link name="part_{pid}">',
            '    <visual>',
            '      <origin xyz="0 0 0" rpy="0 0 0"/>',
            f'      <geometry><mesh filename="{mesh_rel}" scale="1 1 1"/></geometry>',
            f'      <material name="placed_{pid}"><color rgba="0.6 0.6 0.6 1.0"/></material>',
            '    </visual>',
            '    <collision>',
            '      <origin xyz="0 0 0" rpy="0 0 0"/>',
            f'      <geometry><mesh filename="{mesh_rel}" scale="1 1 1"/></geometry>',
            '    </collision>',
            '  </link>',
            f'  <joint name="part_{pid}_joint" type="fixed">',
            '    <parent link="box"/>',
            f'    <child link="part_{pid}"/>',
            f'    <origin xyz="{rx} {ry} {rz}" rpy="{roll} {pitch} {yaw}"/>',
            '  </joint>',
        ])

    lines.append('</robot>')
    out_path = env_dir / "scene.urdf"
    out_path.write_text('\n'.join(lines))
    return out_path


def _generate_sdf_table(assembly: str, active_pid: str, completed_pids: List[str],
                        env_dir: Path, sdf_resolution: int = 512):
    """SDF table: single link per completed part with raw mesh + SDF tag."""
    lines = [
        '<?xml version="1.0"?>',
        f'<robot name="table_{assembly}_{active_pid}_scene_sdf">',
        TABLE_LINK,
    ]

    for pid in completed_pids:
        offset = _part_goal_offset(assembly, pid)
        if offset is None:
            continue
        rx, ry, rz, roll, pitch, yaw = offset
        mesh_rel = f"../../{assembly}/{pid}/{pid}_canonical.obj"

        lines.extend([
            f'  <link name="part_{pid}">',
            '    <visual>',
            '      <origin xyz="0 0 0" rpy="0 0 0"/>',
            f'      <geometry><mesh filename="{mesh_rel}" scale="1 1 1"/></geometry>',
            f'      <material name="placed_{pid}"><color rgba="0.6 0.6 0.6 1.0"/></material>',
            '    </visual>',
            '    <collision>',
            '      <origin xyz="0 0 0" rpy="0 0 0"/>',
            f'      <geometry><mesh filename="{mesh_rel}" scale="1 1 1"/></geometry>',
            f'      <sdf resolution="{sdf_resolution}"/>',
            '    </collision>',
            '  </link>',
            f'  <joint name="part_{pid}_joint" type="fixed">',
            '    <parent link="box"/>',
            f'    <child link="part_{pid}"/>',
            f'    <origin xyz="{rx} {ry} {rz}" rpy="{roll} {pitch} {yaw}"/>',
            '  </joint>',
        ])

    lines.append('</robot>')
    out_path = env_dir / "scene_sdf.urdf"
    out_path.write_text('\n'.join(lines))
    return out_path


def _generate_coacd_table(assembly: str, active_pid: str, completed_pids: List[str],
                          env_dir: Path):
    """CoACD table: multiple sub-links per completed part (one per decomp hull)."""
    lines = [
        '<?xml version="1.0"?>',
        f'<robot name="table_{assembly}_{active_pid}_scene_coacd">',
        TABLE_LINK,
    ]

    for pid in completed_pids:
        offset = _part_goal_offset(assembly, pid)
        if offset is None:
            continue
        rx, ry, rz, roll, pitch, yaw = offset

        coacd_dir = ASSETS_DIR / assembly / pid / "coacd"
        decomp_files = sorted(coacd_dir.glob("decomp_*.obj"))
        if not decomp_files:
            print(f"  Warning: no CoACD decomps for {assembly}/{pid}, skipping")
            continue

        for i, decomp_file in enumerate(decomp_files):
            link_name = f"part_{pid}_hull_{i}"
            mesh_rel = f"../../{assembly}/{pid}/coacd/{decomp_file.name}"

            lines.extend([
                f'  <link name="{link_name}">',
                '    <visual>',
                '      <origin xyz="0 0 0" rpy="0 0 0"/>',
                f'      <geometry><mesh filename="{mesh_rel}" scale="1 1 1"/></geometry>',
                f'      <material name="placed_{pid}_{i}"><color rgba="0.6 0.6 0.6 1.0"/></material>',
                '    </visual>',
                '    <collision>',
                '      <origin xyz="0 0 0" rpy="0 0 0"/>',
                f'      <geometry><mesh filename="{mesh_rel}" scale="1 1 1"/></geometry>',
                '    </collision>',
                '  </link>',
                f'  <joint name="{link_name}_joint" type="fixed">',
                '    <parent link="box"/>',
                f'    <child link="{link_name}"/>',
                f'    <origin xyz="{rx} {ry} {rz}" rpy="{roll} {pitch} {yaw}"/>',
                '  </joint>',
            ])

    lines.append('</robot>')
    out_path = env_dir / "scene_coacd.urdf"
    out_path.write_text('\n'.join(lines))
    return out_path


def _generate_empty_table_coacd():
    """Generate empty_table_coacd.urdf (just the box, no parts)."""
    lines = [
        '<?xml version="1.0"?>',
        '<robot name="empty_table_coacd">',
        TABLE_LINK,
        '</robot>',
    ]
    out_path = ASSETS_DIR / "environments" / "empty_table_coacd.urdf"
    out_path.write_text('\n'.join(lines))
    print(f"  Generated: {out_path}")
    return out_path


def generate_all_tables(assembly: str):
    """Generate table URDFs for every step in the assembly order."""
    order = _load_assembly_order(assembly)
    if not order:
        print(f"No assembly order found for {assembly}")
        return

    # Filter to parts with trajectories
    available = []
    for pid in order:
        traj_path = REPO_ROOT / "fabrica" / "trajectories" / f"{assembly}_{pid}" / "pick_place.json"
        if traj_path.exists():
            available.append(pid)

    if not available:
        print(f"No trajectories found for {assembly}")
        return

    print(f"Assembly: {assembly}")
    print(f"  Available parts (in order): {available}")

    # Generate empty table coacd variant
    _generate_empty_table_coacd()

    for step_idx, active_pid in enumerate(available):
        completed_pids = available[:step_idx]
        env_dir = ASSETS_DIR / "environments" / f"{assembly}_{active_pid}"
        env_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  Step {step_idx}: Part {active_pid} (completed: {completed_pids})")

        vhacd_path = _generate_vhacd_table(assembly, active_pid, completed_pids, env_dir)
        print(f"    VHACD: {vhacd_path.name}")

        sdf_path = _generate_sdf_table(assembly, active_pid, completed_pids, env_dir)
        print(f"    SDF:   {sdf_path.name}")

        coacd_path = _generate_coacd_table(assembly, active_pid, completed_pids, env_dir)
        print(f"    CoACD: {coacd_path.name}")

    print(f"\nDone. Generated URDFs for {len(available)} steps.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute table URDFs for assembly evaluation")
    parser.add_argument("--assembly", default="beam", help="Assembly name (default: beam)")
    args = parser.parse_args()
    generate_all_tables(args.assembly)
