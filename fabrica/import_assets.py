#!/usr/bin/env python3
"""Import Fabrica OBJ assets: scale vertices cm->m and generate URDFs.

Usage:
    python fabrica/import_assets.py
    python fabrica/import_assets.py --assemblies car duct
    python fabrica/import_assets.py --force  # overwrite existing
"""

import argparse
from pathlib import Path

import trimesh

ALL_ASSEMBLIES = [
    "car",
    "cooling_manifold",
    "duct",
    "gamepad",
    "plumbers_block",
    "stool_circular",
]

URDF_TEMPLATE = """\
<?xml version="1.0"?>
<robot name="{assembly}_{part_id}">
  <link name="{assembly}_{part_id}">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{part_id}.obj" scale="1 1 1"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{part_id}.obj" scale="1 1 1"/>
      </geometry>
    </collision>
    <inertial>
      <density value="1250.0"/>
    </inertial>
  </link>
</robot>
"""


def import_assembly(fabrica_dir: Path, output_dir: Path, assembly: str, force: bool):
    src_dir = fabrica_dir / assembly
    obj_files = sorted(src_dir.glob("*.obj"))
    if not obj_files:
        print(f"  WARNING: No OBJ files found in {src_dir}")
        return 0

    count = 0
    for obj_path in obj_files:
        part_id = obj_path.stem  # e.g. "0", "1", ...
        part_dir = output_dir / assembly / part_id
        out_obj = part_dir / f"{part_id}.obj"
        out_urdf = part_dir / f"{assembly}_{part_id}.urdf"

        if out_obj.exists() and not force:
            print(f"  Skipping {assembly}/{part_id} (exists, use --force to overwrite)")
            continue

        mesh = trimesh.load_mesh(str(obj_path), process=False, maintain_order=True)
        mesh.vertices *= 0.01  # cm -> m

        part_dir.mkdir(parents=True, exist_ok=True)
        mesh.export(str(out_obj))
        out_urdf.write_text(URDF_TEMPLATE.format(assembly=assembly, part_id=part_id))
        count += 1
        print(f"  {assembly}/{part_id}: OBJ ({len(mesh.vertices)} verts) + URDF")

    return count


def main():
    parser = argparse.ArgumentParser(description="Import Fabrica OBJ assets")
    parser.add_argument(
        "--fabrica-dir",
        type=Path,
        default=Path("/juno/u/kedia/Fabrica/assets/fabrica"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "assets" / "urdf" / "fabrica",
    )
    parser.add_argument(
        "--assemblies",
        nargs="+",
        default=ALL_ASSEMBLIES,
        choices=ALL_ASSEMBLIES,
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    total = 0
    for assembly in args.assemblies:
        print(f"Importing {assembly}...")
        n = import_assembly(args.fabrica_dir, args.output_dir, assembly, args.force)
        total += n

    print(f"\nDone. Imported {total} parts.")


if __name__ == "__main__":
    main()
