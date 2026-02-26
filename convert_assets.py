#!/usr/bin/env python3
"""Convert URDF assets to USD format for Isaac Lab.

Usage:
    # From within Isaac Lab's Python environment:
    python convert_assets.py

    # Or with Isaac Lab's launcher:
    ./isaaclab.sh -p convert_assets.py

This script converts the robot, table, and DexToolBench object URDFs
to USD format that Isaac Lab/Isaac Sim can load natively.
"""

import argparse
import os
from glob import glob
from pathlib import Path

# Isaac Sim must be initialized before any Isaac Lab imports
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


def _make_joint_drive():
    """Create a default JointDriveCfg for position-controlled joints."""
    from isaaclab.sim.converters import UrdfConverterCfg
    JDC = UrdfConverterCfg.JointDriveCfg
    return JDC(
        drive_type="force",
        target_type="position",
        gains=JDC.PDGainsCfg(stiffness=0.0, damping=0.0),  # overridden by actuator cfg
    )


def convert_robot(asset_root: str, usd_root: str):
    """Convert the Kuka + Sharpa hand URDF to USD."""
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    robot_urdf = os.path.join(
        asset_root, "urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf"
    )
    if not os.path.exists(robot_urdf):
        print(f"[SKIP] Robot URDF not found: {robot_urdf}")
        return None

    cfg = UrdfConverterCfg(
        asset_path=robot_urdf,
        usd_dir=os.path.join(usd_root, "kuka_sharpa"),
        usd_file_name="robot.usd",
        fix_base=True,
        merge_fixed_joints=False,  # keep all links for camera mounting
        joint_drive=_make_joint_drive(),
    )
    converter = UrdfConverter(cfg)
    print(f"[OK] Robot USD: {converter.usd_path}")
    return converter.usd_path


def convert_table(asset_root: str, usd_root: str, table_name: str = "table_narrow"):
    """Convert the table URDF to USD."""
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    table_urdf = os.path.join(asset_root, f"urdf/{table_name}.urdf")
    if not os.path.exists(table_urdf):
        print(f"[SKIP] Table URDF not found: {table_urdf}")
        return None

    cfg = UrdfConverterCfg(
        asset_path=table_urdf,
        usd_dir=os.path.join(usd_root, "table"),
        usd_file_name=f"{table_name}.usd",
        fix_base=True,
        joint_drive=None,  # table has no joints
    )
    converter = UrdfConverter(cfg)
    print(f"[OK] Table USD: {converter.usd_path}")
    return converter.usd_path


def convert_dextoolbench_objects(asset_root: str, usd_root: str):
    """Convert all DexToolBench object URDFs to USD."""
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    urdf_pattern = os.path.join(asset_root, "urdf/dextoolbench/**/*.urdf")
    urdf_files = sorted(glob(urdf_pattern, recursive=True))

    if not urdf_files:
        print(f"[SKIP] No DexToolBench URDFs found matching: {urdf_pattern}")
        return []

    results = []
    for urdf_path in urdf_files:
        # Derive output directory from URDF path structure
        # e.g. urdf/dextoolbench/hammer/claw_hammer/claw_hammer.urdf
        #   -> usd/dextoolbench/hammer/claw_hammer/
        rel_path = os.path.relpath(urdf_path, os.path.join(asset_root, "urdf/dextoolbench"))
        obj_dir = os.path.dirname(rel_path)
        obj_name = Path(urdf_path).stem

        cfg = UrdfConverterCfg(
            asset_path=urdf_path,
            usd_dir=os.path.join(usd_root, "dextoolbench", obj_dir),
            usd_file_name=f"{obj_name}.usd",
            fix_base=False,
            joint_drive=None,  # objects are rigid bodies, no joints
        )
        try:
            converter = UrdfConverter(cfg)
            print(f"[OK] Object USD: {converter.usd_path}")
            results.append(converter.usd_path)
        except Exception as e:
            print(f"[FAIL] {urdf_path}: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Convert URDF assets to USD for Isaac Lab")
    parser.add_argument(
        "--asset-root",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets"),
        help="Root directory containing URDF assets",
    )
    parser.add_argument(
        "--usd-root",
        default=None,
        help="Output directory for USD assets (default: <asset-root>/usd)",
    )
    parser.add_argument(
        "--robot-only", action="store_true", help="Only convert the robot"
    )
    parser.add_argument(
        "--table-only", action="store_true", help="Only convert the table"
    )
    parser.add_argument(
        "--objects-only", action="store_true", help="Only convert DexToolBench objects"
    )
    args = parser.parse_args()

    if args.usd_root is None:
        args.usd_root = os.path.join(args.asset_root, "usd")

    os.makedirs(args.usd_root, exist_ok=True)

    convert_all = not (args.robot_only or args.table_only or args.objects_only)

    if convert_all or args.robot_only:
        print("\n=== Converting Robot ===")
        convert_robot(args.asset_root, args.usd_root)

    if convert_all or args.table_only:
        print("\n=== Converting Table ===")
        convert_table(args.asset_root, args.usd_root)

    if convert_all or args.objects_only:
        print("\n=== Converting DexToolBench Objects ===")
        convert_dextoolbench_objects(args.asset_root, args.usd_root)

    print("\nDone! Verify assets by loading them in Isaac Sim GUI.")


if __name__ == "__main__":
    main()
    simulation_app.close()
