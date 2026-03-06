#!/usr/bin/env python3
"""Verify imported Fabrica assets and generate objects.py code.

Usage:
    python fabrica/verify_assets.py
    python fabrica/verify_assets.py --assemblies car duct
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

REPO_ROOT = Path(__file__).resolve().parent.parent


def verify_assembly(assets_dir: Path, assembly: str):
    assembly_dir = assets_dir / assembly
    if not assembly_dir.exists():
        print(f"  ERROR: {assembly_dir} does not exist")
        return []

    parts = sorted(
        [d for d in assembly_dir.iterdir() if d.is_dir() and d.name != "fixture"],
        key=lambda d: int(d.name),
    )
    results = []

    for part_dir in parts:
        part_id = part_dir.name
        obj_path = part_dir / f"{part_id}.obj"
        urdf_path = part_dir / f"{assembly}_{part_id}.urdf"

        if not obj_path.exists():
            print(f"  ERROR: Missing OBJ {obj_path}")
            continue
        if not urdf_path.exists():
            print(f"  ERROR: Missing URDF {urdf_path}")
            continue

        mesh = trimesh.load_mesh(str(obj_path), process=False)
        extents = mesh.extents
        is_watertight = mesh.is_watertight

        status = "OK" if is_watertight else "WARN (not watertight)"
        print(
            f"  {assembly}_{part_id}: "
            f"extents=({extents[0]:.4f}, {extents[1]:.4f}, {extents[2]:.4f}) "
            f"scale*25=({extents[0]*25:.3f}, {extents[1]*25:.3f}, {extents[2]*25:.3f}) "
            f"[{status}]"
        )
        results.append((assembly, part_id, extents))

    return results


def generate_objects_code(all_results, write_objects: bool = False):
    """Generate Python code for objects.py, optionally writing it directly."""
    assemblies = {}
    for assembly, part_id, extents in all_results:
        assemblies.setdefault(assembly, []).append((part_id, extents))

    lines = []
    for assembly, parts in assemblies.items():
        dict_name = assembly.upper() + "_NAME_TO_OBJECT"
        lines.append(f"{dict_name} = {{")
        for part_id, extents in parts:
            name = f"{assembly}_{part_id}"
            ext_str = f"({extents[0]:.4f}, {extents[1]:.4f}, {extents[2]:.4f})"
            lines.append(f'    "{name}": Object(')
            lines.append(f"        urdf_path=(")
            lines.append(f"            get_repo_root_dir()")
            lines.append(f'            / "assets/urdf/fabrica/{assembly}/{part_id}/{name}.urdf"')
            lines.append(f"        ),")
            lines.append(f"        scale=rescale_by_factor({ext_str}, factor=25),")
            lines.append(f"        need_vhacd=True,")
            lines.append(f"    ),")
        lines.append("}")
        lines.append("")

    generated_code = "\n".join(lines)

    # Build update lines
    update_lines = []
    for assembly in assemblies:
        dict_name = assembly.upper() + "_NAME_TO_OBJECT"
        update_lines.append(f"FABRICA_NAME_TO_OBJECT.update({dict_name})")
    update_code = "\n".join(update_lines)

    if write_objects:
        objects_py = REPO_ROOT / "fabrica" / "objects.py"
        content = objects_py.read_text()

        # Insert new dicts before the FABRICA_NAME_TO_OBJECT line
        marker = "FABRICA_NAME_TO_OBJECT = {}"
        if marker not in content:
            print("ERROR: Could not find marker in objects.py")
            return

        content = content.replace(
            marker,
            generated_code + "\n" + marker,
        )

        # Insert update calls before the NAME_TO_OBJECT line
        old_tail = (
            "FABRICA_NAME_TO_OBJECT.update(BEAM_NAME_TO_OBJECT)\n"
            "\n"
            "# Register into the global object registry so the sim can find them\n"
            "NAME_TO_OBJECT.update(FABRICA_NAME_TO_OBJECT)"
        )
        new_tail = (
            "FABRICA_NAME_TO_OBJECT.update(BEAM_NAME_TO_OBJECT)\n"
            + update_code + "\n"
            "\n"
            "# Register into the global object registry so the sim can find them\n"
            "NAME_TO_OBJECT.update(FABRICA_NAME_TO_OBJECT)"
        )
        content = content.replace(old_tail, new_tail)

        objects_py.write_text(content)
        print(f"\nWrote {len(all_results)} new Object entries to {objects_py}")
    else:
        print("\n" + "=" * 70)
        print("# Generated code (use --write-objects to write to fabrica/objects.py)")
        print("=" * 70 + "\n")
        print(generated_code)
        print(update_code)


def main():
    parser = argparse.ArgumentParser(description="Verify imported Fabrica assets")
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=REPO_ROOT / "assets" / "urdf" / "fabrica",
    )
    parser.add_argument(
        "--assemblies",
        nargs="+",
        default=ALL_ASSEMBLIES,
        choices=ALL_ASSEMBLIES,
    )
    parser.add_argument(
        "--write-objects",
        action="store_true",
        help="Write Object entries directly into fabrica/objects.py",
    )
    args = parser.parse_args()

    all_results = []
    for assembly in args.assemblies:
        print(f"Verifying {assembly}...")
        results = verify_assembly(args.assets_dir, assembly)
        all_results.extend(results)

    print(f"\nTotal parts verified: {len(all_results)}")
    generate_objects_code(all_results, write_objects=args.write_objects)


if __name__ == "__main__":
    main()
