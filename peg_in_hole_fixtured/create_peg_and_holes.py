#!/usr/bin/env python3
"""Generate peg + per-tolerance hole assets for the fixtured peg-in-hole task.

Mirrors ``peg_in_hole/create_peg_and_holes.py`` but writes into
``assets/urdf/peg_in_hole_fixtured/``. The scene URDFs (table + start fixture +
goal fixture) are produced separately by
``peg_in_hole_fixtured/scene_generation/generate_scenes.py``; this script only
emits the peg URDF + mesh and the per-tolerance hole URDFs (used as references
for visualization and tolerance-pool sanity checks).

Outputs to ``assets/urdf/peg_in_hole_fixtured/``:
  peg/{peg.obj, peg.mtl, peg_texture.png, peg.stl, peg.urdf}
  holes/hole_tol{T}mm/{hole.obj, hole.mtl, hole_texture.png, hole.stl,
                       hole_tol{T}mm.urdf}
"""

from pathlib import Path

import numpy as np
import trimesh
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "assets/urdf/peg_in_hole_fixtured"

# --- Peg geometry (must match peg_in_hole/create_peg_and_holes.py) ---
HANDLE_EXTENTS = (0.25, 0.03, 0.02)
HEAD_EXTENTS = (0.02, 0.10, 0.02)
HANDLE_CENTER = (0.0, 0.0, 0.0)
HEAD_CENTER = (0.115, 0.0, 0.0)

# --- Hole geometry ---
HOLE_FOOTPRINT_X = 0.08
HOLE_FOOTPRINT_Y = 0.08
HOLE_SLOT_CORE_X = 0.02
HOLE_SLOT_CORE_Y = 0.03
HOLE_FLOOR_THICKNESS = 0.01
HOLE_DEPTH = 0.05

# Tolerances we emit standalone URDFs for. The scene URDFs use the scene
# generator's per-scene tolerance pool, but these standalone ones cover the
# fixed start-fixture tolerance (0.5 mm) and a range of goal tolerances.
TOLERANCES_MM = [0.1, 0.5, 1.0, 5.0, 10.0, 20.0]

PEG_COLOR = (204, 40, 40)
HOLE_COLOR = (120, 120, 120)
DENSITY = 1250.0


def fmt_tol(tol_mm: float) -> str:
    if tol_mm == int(tol_mm):
        return str(int(tol_mm))
    return f"{tol_mm}".replace(".", "p")


def box_mesh(center, extents):
    m = trimesh.creation.box(extents=np.asarray(extents, dtype=float))
    m.apply_translation(np.asarray(center, dtype=float))
    return m


def peg_boxes():
    return [
        (HANDLE_CENTER, HANDLE_EXTENTS),
        (HEAD_CENTER, HEAD_EXTENTS),
    ]


def hole_boxes(tol_m):
    slot_x = HOLE_SLOT_CORE_X + 2 * tol_m
    slot_y = HOLE_SLOT_CORE_Y + 2 * tol_m
    t = HOLE_FLOOR_THICKNESS
    d = HOLE_DEPTH
    ox, oy = HOLE_FOOTPRINT_X, HOLE_FOOTPRINT_Y

    assert slot_x < ox, f"slot_x {slot_x} >= outer {ox} — increase HOLE_FOOTPRINT_X or reduce tol"
    assert slot_y <= oy, f"slot_y {slot_y} > outer {oy} — increase HOLE_FOOTPRINT_Y or reduce tol"

    boxes = [((0.0, 0.0, t / 2), (ox, oy, t))]
    zc = t + d / 2
    ew = (ox - slot_x) / 2
    if ew > 1e-6:
        boxes.append(((slot_x / 2 + ew / 2, 0.0, zc), (ew, oy, d)))
        boxes.append(((-(slot_x / 2 + ew / 2), 0.0, zc), (ew, oy, d)))
    nl = (oy - slot_y) / 2
    if nl > 1e-6:
        boxes.append(((0.0, slot_y / 2 + nl / 2, zc), (slot_x, nl, d)))
        boxes.append(((0.0, -(slot_y / 2 + nl / 2), zc), (slot_x, nl, d)))
    return boxes


def boxes_to_mesh(boxes):
    return trimesh.util.concatenate([box_mesh(c, e) for c, e in boxes])


def write_obj_mtl_png(out_dir: Path, name: str, mesh, rgb):
    out_dir.mkdir(parents=True, exist_ok=True)
    obj_path = out_dir / f"{name}.obj"
    mtl_path = out_dir / f"{name}.mtl"
    png_path = out_dir / f"{name}_texture.png"

    Image.new("RGB", (32, 32), rgb).save(png_path)

    mtl_lines = [
        f"newmtl {name}_mat",
        "Ka 0.2 0.2 0.2",
        f"Kd {rgb[0]/255:.4f} {rgb[1]/255:.4f} {rgb[2]/255:.4f}",
        "Ks 0.0 0.0 0.0",
        "d 1.0",
        "illum 2",
        f"map_Kd {name}_texture.png",
    ]
    mtl_path.write_text("\n".join(mtl_lines) + "\n")

    V = mesh.vertices
    F = mesh.faces
    FN = mesh.face_normals
    obj_lines = [f"mtllib {name}.mtl", f"o {name}", f"usemtl {name}_mat"]
    for v in V:
        obj_lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
    obj_lines.append("vt 0.0 0.0")
    for n in FN:
        obj_lines.append(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}")
    for fi, face in enumerate(F):
        v1, v2, v3 = face
        ni = fi + 1
        obj_lines.append(
            f"f {v1+1}/1/{ni} {v2+1}/1/{ni} {v3+1}/1/{ni}"
        )
    obj_path.write_text("\n".join(obj_lines) + "\n")


def write_stl(out_dir: Path, name: str, mesh):
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out_dir / f"{name}.stl"))


def box_xml(boxes, tag: str, material_name=None, indent="    "):
    parts = []
    for c, e in boxes:
        mat = f"\n{indent}  <material name=\"{material_name}\"/>" if (material_name and tag == "visual") else ""
        parts.append(
            f'{indent}<{tag}>\n'
            f'{indent}  <origin xyz="{c[0]:.6f} {c[1]:.6f} {c[2]:.6f}" rpy="0 0 0"/>\n'
            f'{indent}  <geometry><box size="{e[0]:.6f} {e[1]:.6f} {e[2]:.6f}"/></geometry>{mat}\n'
            f'{indent}</{tag}>'
        )
    return "\n".join(parts)


def write_peg_urdf(out_dir: Path):
    boxes = peg_boxes()
    h_c, h_e = boxes[0]
    d_c, d_e = boxes[1]
    vol_h = h_e[0] * h_e[1] * h_e[2]
    vol_d = d_e[0] * d_e[1] * d_e[2]
    vol = vol_h + vol_d
    mass = DENSITY * vol
    com_x = (h_c[0] * vol_h + d_c[0] * vol_d) / vol
    ixx = mass * (h_e[1] ** 2 + h_e[2] ** 2) / 12
    iyy = mass * (h_e[0] ** 2 + h_e[2] ** 2) / 12
    izz = mass * (h_e[0] ** 2 + h_e[1] ** 2) / 12

    vis = box_xml(boxes, "visual", material_name="peg_red")
    col = box_xml(boxes, "collision")
    r, g, b = PEG_COLOR
    xml = f'''<?xml version="1.0"?>
<robot name="peg">
  <material name="peg_red">
    <color rgba="{r/255:.4f} {g/255:.4f} {b/255:.4f} 1.0"/>
  </material>
  <link name="peg">
{vis}
{col}
    <inertial>
      <mass value="{mass:.6f}"/>
      <origin xyz="{com_x:.6f} 0 0" rpy="0 0 0"/>
      <inertia ixx="{ixx:.6e}" ixy="0" ixz="0" iyy="{iyy:.6e}" iyz="0" izz="{izz:.6e}"/>
    </inertial>
  </link>
</robot>
'''
    (out_dir / "peg.urdf").write_text(xml)


def write_hole_urdf(out_dir: Path, tol_m: float, urdf_name: str, robot_name: str):
    boxes = hole_boxes(tol_m)
    vis = box_xml(boxes, "visual", material_name="hole_grey")
    col = box_xml(boxes, "collision")
    r, g, b = HOLE_COLOR
    xml = f'''<?xml version="1.0"?>
<robot name="{robot_name}">
  <material name="hole_grey">
    <color rgba="{r/255:.4f} {g/255:.4f} {b/255:.4f} 1.0"/>
  </material>
  <link name="hole">
{vis}
{col}
    <inertial>
      <mass value="10.0"/>
      <origin xyz="0 0 0.055" rpy="0 0 0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>
</robot>
'''
    (out_dir / urdf_name).write_text(xml)


def main():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    peg_dir = ASSETS_DIR / "peg"
    peg_mesh = boxes_to_mesh(peg_boxes())
    write_obj_mtl_png(peg_dir, "peg", peg_mesh, PEG_COLOR)
    write_stl(peg_dir, "peg", peg_mesh)
    write_peg_urdf(peg_dir)
    print(f"[peg] extents={peg_mesh.extents}  -> {peg_dir}")

    for tol_mm in TOLERANCES_MM:
        tol_m = tol_mm * 1e-3
        tag = fmt_tol(tol_mm)
        hname = f"hole_tol{tag}mm"
        hdir = ASSETS_DIR / "holes" / hname
        hmesh = boxes_to_mesh(hole_boxes(tol_m))
        write_obj_mtl_png(hdir, "hole", hmesh, HOLE_COLOR)
        write_stl(hdir, "hole", hmesh)
        write_hole_urdf(hdir, tol_m, urdf_name=f"{hname}.urdf", robot_name=hname)
        slot = (HOLE_SLOT_CORE_X + 2 * tol_m, HOLE_SLOT_CORE_Y + 2 * tol_m)
        print(f"[{hname}] slot=({slot[0]:.4f}, {slot[1]:.4f}) -> {hdir}")

    print("Done.")


if __name__ == "__main__":
    main()
