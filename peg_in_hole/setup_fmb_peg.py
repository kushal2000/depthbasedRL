#!/usr/bin/env python3
"""Generate CoACD decompositions, URDFs, and SDF URDFs for an FMB peg + board.

Produces centered meshes + all URDF variants needed by PegInHoleDynamicEnv:
  - peg: centered canonical mesh + CoACD decomp + plain/SDF/CoACD URDFs
  - board: centered canonical mesh + CoACD decomp + plain/SDF/CoACD URDFs

Usage:
    python peg_in_hole/setup_fmb_peg.py --peg 1 --board 0
"""

import argparse
from pathlib import Path

import coacd
import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "assets" / "urdf" / "fmb"


def _canonicalize_mesh(mesh: trimesh.Trimesh) -> tuple:
    """Center mesh at origin and rotate so longest axis is X, second longest is Y.
    Returns (mesh, rotation_matrix) where rotation_matrix maps original → canonical."""
    center = (mesh.bounds[0] + mesh.bounds[1]) / 2
    mesh.vertices -= center

    extents = mesh.extents.copy()
    # Sort axes: longest → X, second → Y, shortest → Z
    axis_order = np.argsort(-extents)  # descending
    if np.array_equal(axis_order, [0, 1, 2]):
        return mesh, np.eye(3)

    # Build permutation matrix
    perm = np.zeros((3, 3))
    for new_ax, old_ax in enumerate(axis_order):
        perm[new_ax, old_ax] = 1.0
    # Ensure right-handed (det = +1)
    if np.linalg.det(perm) < 0:
        perm[2] = -perm[2]

    mesh.vertices = mesh.vertices @ perm.T
    return mesh, perm


def _run_coacd(mesh: trimesh.Trimesh, threshold: float = 0.05) -> list:
    """Run CoACD and return list of convex hull trimeshes."""
    tm = coacd.Mesh(mesh.vertices, mesh.faces)
    parts = coacd.run_coacd(tm, threshold=threshold)
    hulls = []
    for vs, fs in parts:
        hulls.append(trimesh.Trimesh(vertices=vs, faces=fs))
    return hulls


def _write_plain_urdf(name: str, mesh_filename: str, out_path: Path, density: float = 1250.0):
    """Single-link URDF with the original mesh for visual+collision."""
    out_path.write_text(f"""<?xml version="1.0"?>
<robot name="{name}">
  <link name="{name}">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="{mesh_filename}" scale="1 1 1"/></geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="{mesh_filename}" scale="1 1 1"/></geometry>
    </collision>
    <inertial><density value="{density}"/></inertial>
  </link>
</robot>
""")


def _write_sdf_urdf(name: str, mesh_filename: str, out_path: Path,
                    density: float = 1250.0, sdf_resolution: int = 512):
    """Single-link URDF with SDF collision."""
    out_path.write_text(f"""<?xml version="1.0"?>
<robot name="{name}_sdf">
  <link name="{name}_sdf">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="{mesh_filename}" scale="1 1 1"/></geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="{mesh_filename}" scale="1 1 1"/></geometry>
      <sdf resolution="{sdf_resolution}"/>
    </collision>
    <inertial><density value="{density}"/></inertial>
  </link>
</robot>
""")


def _write_coacd_urdf(name: str, hull_filenames: list, out_path: Path,
                      canonical_mesh: str = None, density: float = 1250.0):
    """Multi-link URDF: canonical mesh for visual, CoACD hulls for collision."""
    lines = [
        '<?xml version="1.0"?>',
        f'<robot name="{name}_coacd">',
        f'  <material name="color_0"><color rgba="0.9 0.2 0.2 1.0"/></material>',
        f'  <link name="base_link">',
        f'    <inertial><density value="{density}"/></inertial>',
    ]
    if canonical_mesh:
        lines += [
            f'    <visual>',
            f'      <origin xyz="0 0 0" rpy="0 0 0"/>',
            f'      <geometry><mesh filename="{canonical_mesh}" scale="1 1 1"/></geometry>',
            f'      <material name="color_0"/>',
            f'    </visual>',
        ]
    lines.append('  </link>')

    for i, hf in enumerate(hull_filenames):
        lines += [
            f'  <link name="hull_{i}">',
            f'    <visual>',
            f'      <origin xyz="0 0 0" rpy="0 0 0"/>',
            f'      <geometry><mesh filename="{hf}" scale="1 1 1"/></geometry>',
            f'      <material name="color_0"/>',
            f'    </visual>',
            f'    <collision>',
            f'      <origin xyz="0 0 0" rpy="0 0 0"/>',
            f'      <geometry><mesh filename="{hf}" scale="1 1 1"/></geometry>',
            f'    </collision>',
            f'    <inertial>',
            f'      <mass value="0.001"/>',
            f'      <inertia ixx="1e-6" ixy="0" ixz="0" iyy="1e-6" iyz="0" izz="1e-6"/>',
            f'    </inertial>',
            f'  </link>',
            f'  <joint name="hull_joint_{i}" type="fixed">',
            f'    <parent link="base_link"/>',
            f'    <child link="hull_{i}"/>',
            f'    <origin xyz="0 0 0" rpy="0 0 0"/>',
            f'  </joint>',
        ]

    lines.append('</robot>')
    lines.append('')
    out_path.write_text('\n'.join(lines))


def process_part(src_obj: Path, output_dir: Path, name: str, coacd_threshold: float = 0.05):
    """Full pipeline: center + rotate (longest→X) → canonical mesh → CoACD → URDFs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    coacd_dir = output_dir / "coacd"
    coacd_dir.mkdir(exist_ok=True)

    # Load, center, and rotate so longest axis is X
    mesh = trimesh.load(str(src_obj), force="mesh")
    original_bbox_center = ((mesh.bounds[0] + mesh.bounds[1]) / 2).copy()
    mesh, perm = _canonicalize_mesh(mesh)
    extents = mesh.extents.copy()

    # Save centered canonical mesh
    canonical_name = f"{name}_canonical.obj"
    canonical_path = output_dir / canonical_name
    mesh.export(str(canonical_path))
    print(f"  Saved canonical mesh: {canonical_path}")
    print(f"    extents (X≥Y≥Z): {extents * 1000} mm")
    print(f"    original bbox center: {original_bbox_center}")
    print(f"    axis permutation: {perm.tolist()}")

    # Also copy canonical into coacd dir for the coacd URDF visual
    coacd_canonical = coacd_dir / canonical_name
    mesh.export(str(coacd_canonical))

    # CoACD decomposition
    print(f"  Running CoACD (threshold={coacd_threshold})...")
    hulls = _run_coacd(mesh, threshold=coacd_threshold)
    print(f"  Got {len(hulls)} convex hulls")

    hull_filenames = []
    for i, hull in enumerate(hulls):
        hull_name = f"decomp_{i}.obj"
        hull.export(str(coacd_dir / hull_name))
        hull_filenames.append(hull_name)

    # Write URDFs
    _write_plain_urdf(name, canonical_name, output_dir / f"{name}.urdf")
    _write_sdf_urdf(name, canonical_name, output_dir / f"{name}_sdf.urdf")
    _write_coacd_urdf(name, hull_filenames, coacd_dir / f"{name}_coacd.urdf",
                      canonical_mesh=canonical_name)

    print(f"  URDFs: {name}.urdf, {name}_sdf.urdf, coacd/{name}_coacd.urdf")

    return {
        "original_bbox_center": original_bbox_center.tolist(),
        "canonical_extents": extents.tolist(),
        "permutation": perm.tolist(),
        "num_hulls": len(hulls),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--peg", type=int, required=True, help="Peg index (0-53)")
    parser.add_argument("--board", type=int, required=True, help="Board index (0-2)")
    parser.add_argument("--coacd-threshold", type=float, default=0.05,
                        help="CoACD threshold (lower = more hulls, tighter fit)")
    args = parser.parse_args()

    peg_name = f"peg_{args.peg}"
    board_name = f"peg_board_{args.board}"

    peg_src = ASSETS_DIR / "pegs" / peg_name / f"{peg_name}.obj"
    board_src = ASSETS_DIR / "boards" / board_name / f"{board_name}.obj"

    if not peg_src.exists():
        raise FileNotFoundError(f"Peg mesh not found: {peg_src}")
    if not board_src.exists():
        raise FileNotFoundError(f"Board mesh not found: {board_src}")

    # Output alongside the source meshes
    peg_out = ASSETS_DIR / "pegs" / peg_name
    board_out = ASSETS_DIR / "boards" / board_name

    print(f"\n=== Processing {peg_name} ===")
    peg_info = process_part(peg_src, peg_out, peg_name, args.coacd_threshold)

    print(f"\n=== Processing {board_name} ===")
    board_info = process_part(board_src, board_out, board_name, args.coacd_threshold)

    # Compute insertion pose in canonical frames.
    # Original frame: peg and board share the assembly coordinate system.
    # Each was centered + rotated independently to canonical form.
    #
    # canonical_point = perm @ (original_point - bbox_center)
    #
    # The insertion position of the peg in the board's canonical frame:
    #   1. peg bbox center in original frame = peg_bbox_center
    #   2. relative to board bbox center = peg_bbox_center - board_bbox_center
    #   3. rotated into board canonical frame = board_perm @ relative_pos
    peg_bbox = np.array(peg_info["original_bbox_center"])
    board_bbox = np.array(board_info["original_bbox_center"])
    peg_perm = np.array(peg_info["permutation"])
    board_perm = np.array(board_info["permutation"])

    # Position: peg center in board's canonical frame
    rel_original = peg_bbox - board_bbox
    insert_pos = board_perm @ rel_original

    # Orientation: peg canonical frame relative to board canonical frame
    # peg_canonical = peg_perm @ original
    # board_canonical = board_perm @ original
    # peg_canonical_in_board_canonical = peg_perm @ board_perm^-1
    # Since perm matrices are orthogonal: board_perm^-1 = board_perm.T
    rel_rot = peg_perm @ board_perm.T

    # Convert rotation matrix to quaternion (xyzw)
    from scipy.spatial.transform import Rotation
    quat_xyzw = Rotation.from_matrix(rel_rot).as_quat()  # [x, y, z, w]

    print(f"\n=== Insertion geometry ===")
    print(f"  Peg bbox center (original):  {peg_bbox}")
    print(f"  Board bbox center (original): {board_bbox}")
    print(f"  Peg permutation:  {peg_perm.tolist()}")
    print(f"  Board permutation: {board_perm.tolist()}")
    print(f"  Relative rotation: {rel_rot.tolist()}")
    print()
    print(f"  Insert pos (board canonical frame): {insert_pos}")
    print(f"  Insert pos (mm): {insert_pos * 1000}")
    print(f"  Insert quat (xyzw): {quat_xyzw}")
    print()
    print(f"  insertPoseRelHole: [{insert_pos[0]:.6f}, {insert_pos[1]:.6f}, {insert_pos[2]:.6f}, "
          f"{quat_xyzw[0]:.6f}, {quat_xyzw[1]:.6f}, {quat_xyzw[2]:.6f}, {quat_xyzw[3]:.6f}]")
    print()
    print(f"  Peg canonical extents (X≥Y≥Z): {np.array(peg_info['canonical_extents'])*1000} mm")
    print(f"  Board canonical extents (X≥Y≥Z): {np.array(board_info['canonical_extents'])*1000} mm")
    print(f"  fixedSize for peg: {peg_info['canonical_extents']}")
    print()
    print("  NOTE: This is a computed estimate. Use the alignment tool to verify/adjust:")
    print("    python peg_in_hole/visualize_fmb_pegs.py")
    print()
    print("Done!")


if __name__ == "__main__":
    main()
