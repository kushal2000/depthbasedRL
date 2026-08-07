#!/usr/bin/env python3
"""Generate the plug_fork URDFs from the source meshes and published specs.

Run this to (re)create everything under ``assets/urdf/plug_fork/``:

    python peg_in_hole_dynamic/plug_fork/create_assets.py [--tolerances 0.5 1 2]

Two tasks, both at real-world scale, added because reviewers noted the existing
suite uses enlarged geometry (beam_3x is literally 3x):

  fork  YCB 030_fork, a real scan   -> moulded flatware holder, handle first
  plug  Apple 20W adapter A2940     -> NEMA 5-15R socket, blades first

Design rules, each of which cost something to learn:

* VISUAL and COLLISION are separate. Visual is the real scan / the boolean
  solid; collision is CoACD hulls for the scan and exact boxes for anything
  procedural. Box collisions on procedural parts are not an approximation --
  they ARE the geometry.
* The fork's collision comes from CoACD, not from axis-aligned slabs. Slabs
  more than doubled its volume (46.5 vs 19.6 cm^3) by filling the dished head
  solid, which on a 1 mm-clearance task makes the simulated fork millimetres
  fatter than the rendered one. CoACD lands at +15%, the residual being tine
  gaps that no convex decomposition can represent.
* Geometry comes from ``fork_collision.ply``, never ``fork_visual.obj``. They
  are the same scan, but the OBJ splits vertices at UV seams, which reads as
  172 disconnected fragments and is not watertight.
* The slot is sized from the SWEPT envelope of the leading half, not from any
  single cross-section. A fork head is dished, so per-station measurements
  understate what a straight slot must clear.
* ``hole_z_offset`` is half the receptive's height so the block sits ON the
  table. The receptive origin is its centroid, so a zero offset sinks half the
  fixture through the table top.
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh

from peg_in_hole_dynamic.plug_fork import viz_insertion_spec as G

REPO = Path(__file__).resolve().parents[2]
ASSETS = REPO / "assets" / "urdf" / "plug_fork"

# Real dinner fork ~55 g. The scan is a solid 19.6 cm^3, so taking a stainless
# density would give 155 g -- the real part is thin-walled where the scan is
# filled. Set the mass from the real object and let density follow.
FORK_MASS_KG = 0.055
# Moulded polypropylene cutlery holder.
HOLDER_DENSITY = 940.0
# Apple 20W USB-C adapter A2940: 58.5 g measured (chargerlab teardown).
PLUG_MASS_KG = 0.0585
# Power board: aluminium sleeve over plastic chassis; effective bulk density.
SOCKET_DENSITY = 1500.0


def _scrub(urdf_text: str) -> str:
    """Scrub '--' inside XML comments only; it is illegal there and prose
    dashes in the generated headers hit it."""
    out, i = [], 0
    while True:
        a = urdf_text.find("<!--", i)
        if a < 0:
            out.append(urdf_text[i:]); break
        b = urdf_text.find("-->", a + 4)
        out.append(urdf_text[i:a + 4])
        out.append(urdf_text[a + 4:b].replace("--", "-"))
        out.append("-->")
        i = b + 3
    return "".join(out)


def _safe_comment(text: str) -> str:
    """XML comments may not contain '--'. Prose dashes in the generated
    headers tripped this, so scrub rather than rely on remembering."""
    return text.replace("--", "-")


def fmt_tol(mm: float) -> str:
    return f"{mm:g}".replace(".", "p")


def inertial_xml(mesh: trimesh.Trimesh, mass: float, indent="    ") -> str:
    """<inertial> from a watertight mesh, scaled to `mass`."""
    m = mesh.copy()
    if not m.is_watertight:
        m = m.convex_hull
    vol = float(abs(m.volume))
    density = mass / max(vol, 1e-12)
    I = np.asarray(m.moment_inertia, dtype=float) * density
    c = np.asarray(m.center_mass, dtype=float)
    return (
        f'{indent}<inertial>\n'
        f'{indent}  <mass value="{mass:.6f}"/>\n'
        f'{indent}  <origin xyz="{c[0]:.6f} {c[1]:.6f} {c[2]:.6f}" rpy="0 0 0"/>\n'
        f'{indent}  <inertia ixx="{I[0,0]:.6e}" ixy="{I[0,1]:.6e}" ixz="{I[0,2]:.6e}"'
        f' iyy="{I[1,1]:.6e}" iyz="{I[1,2]:.6e}" izz="{I[2,2]:.6e}"/>\n'
        f'{indent}</inertial>\n'
    )


# Objects in this repo are authored with their LONG AXIS ON X: the peg URDF
# lies along X and its insert pose carries the -90 deg Y rotation that stands it
# up. That is not cosmetic -- reward.fixed_size = (0.141, 0.030, 0.027) defines
# the success keypoints along X, so an object built along Z has its keypoint box
# rotated 90 deg against it and success is measured on the wrong axes.
#
# The tuner works in a Z-up frame (long axis Z, handle at -Z) because that is
# the natural frame for insertion maths. This maps that frame to the URDF
# convention: R_y(+90) sends z_hat -> x_hat, so the handle ends up at -X.
_TUNER_TO_URDF = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])


def write_fork_urdf() -> Path:
    """Fork: textured scan for visual, 9 CoACD hulls for collision."""
    d = ASSETS / "fork"
    raw = trimesh.load(d / "fork_collision.ply", process=False)
    T = trimesh.transformations.concatenate_matrices(
        _TUNER_TO_URDF, G._fork_orient_transform(raw))
    geom = raw.copy()
    geom.apply_transform(T)

    # Bake the oriented meshes so the URDF needs no origin transforms and the
    # object frame matches what the tuner showed. Only what a URDF references is
    # written -- the oriented collision PLY was an unreferenced 1.8 MB duplicate.
    vis = trimesh.load(d / "fork_visual.obj", process=False)
    vis.apply_transform(T)
    vis.export(d / "fork_visual_oriented.obj")

    hulls = G.load_coacd_hulls(T)
    for i, h in enumerate(hulls):
        h.export(d / f"coacd_hull_{i:02d}_oriented.ply")

    cols = "".join(
        f'    <collision>\n'
        f'      <origin xyz="0 0 0" rpy="0 0 0"/>\n'
        f'      <geometry><mesh filename="coacd_hull_{i:02d}_oriented.ply"/></geometry>\n'
        f'    </collision>\n'
        for i in range(len(hulls))
    )
    body = (
        '<?xml version="1.0"?>\n'
        '<!--\n'
        '  YCB 030_fork at true scale (197.6 x 27.1 x 15.9 mm).\n\n'
        '  visual    : the scan with its own texture.\n'
        f'  collision : {len(hulls)} CoACD hulls (threshold 0.03). Axis-aligned slab\n'
        '              boxes were tried first and more than DOUBLED the volume\n'
        '              (46.5 vs 19.6 cm^3) by filling the dished head solid,\n'
        '              unusable when the task clearance is 1 mm. CoACD is +15%,\n'
        '              and that residual is tine gaps, which is the floor for any\n'
        '              convex decomposition.\n\n'
        f'  mass      : {FORK_MASS_KG*1000:.0f} g, a real dinner fork. The scan is solid\n'
        '              where the real part is thin, so stainless density would\n'
        '              give 155 g; mass is set from the real object instead.\n\n'
        '  Frame: long axis on Z with the HANDLE pointing -Z, i.e. the entry\n'
        '  direction. Meshes are pre-oriented, so all origins here are identity.\n'
        '-->\n'
        '<robot name="ycb_fork">\n'
        '  <link name="ycb_fork">\n'
        '    <visual>\n'
        '      <origin xyz="0 0 0" rpy="0 0 0"/>\n'
        '      <geometry><mesh filename="fork_visual_oriented.obj"/></geometry>\n'
        '    </visual>\n'
        f'{cols}'
        f'{inertial_xml(geom, FORK_MASS_KG)}'
        '  </link>\n'
        '</robot>\n'
    )
    out = d / "fork.urdf"
    body = _scrub(body)
    ET.fromstring(body)                       # never write invalid XML
    out.write_text(body)
    return out


def write_plug_urdf() -> Path:
    """Apple 20 W adapter: boolean-cut shell for visual, 3 exact boxes for
    collision (body + two prongs).

    The collision boxes are not an approximation -- the part is procedural, so
    they ARE the geometry. Only the visual carries the corner rounds, the USB-C
    recess and the moulded seam.

    Authored long-axis-X with the PRONGS AT -X, matching the peg convention:
    reward.fixed_size and Object.scale both lay their keypoints along X, so a
    part built along another axis gets a keypoint box rotated against its mesh.
    The insertion quaternion is therefore the peg's -90 deg about Y.
    """
    d = ASSETS / "plug"
    d.mkdir(parents=True, exist_ok=True)

    # tuner frame has prongs along -Z; rotate so they point -X
    scene = G.build_plug_visual()
    vis = scene.copy()
    vis.apply_transform(_TUNER_TO_URDF)
    vis.export(d / "plug_visual.glb")

    # collision: body + two prongs, in the same frame
    boxes = [(G.BODY, (0.0, 0.0, 0.0))]
    for w, x in ((G.BLADE_W_HOT, +G.BLADE_PITCH / 2),
                 (G.BLADE_W_NEU, -G.BLADE_PITCH / 2)):
        boxes.append(((G.BLADE_T, w, G.BLADE_LEN),
                      (x, 0.0, -G.BODY[2] / 2 - G.BLADE_LEN / 2)))
    R = _TUNER_TO_URDF[:3, :3]
    rot_boxes = []
    for ext, ctr in boxes:
        e = np.abs(R @ np.asarray(ext))
        c = R @ np.asarray(ctr)
        rot_boxes.append((tuple(e), tuple(c)))

    col = "".join(
        f'    <collision>\n'
        f'      <origin xyz="{cx:.6f} {cy:.6f} {cz:.6f}" rpy="0 0 0"/>\n'
        f'      <geometry><box size="{ex:.6f} {ey:.6f} {ez:.6f}"/></geometry>\n'
        f'    </collision>\n'
        for (ex, ey, ez), (cx, cy, cz) in rot_boxes)
    solid = trimesh.util.concatenate([G.box(e, c) for e, c in rot_boxes])

    body = (
        '<?xml version="1.0"?>\n'
        '<!--\n'
        '  Apple 20 W USB-C power adapter, model A2940 (the iPhone 15-era\n'
        '  charger), built to published dimensions: 41.5 x 42.5 x 27 mm, 58.5 g,\n'
        '  per the chargerlab teardown. Prongs are NEMA 1-15P: 1.5 mm thick,\n'
        '  6.3 / 7.9 mm wide, 12.7 mm pitch, and FIXED (this model does not fold),\n'
        '  so a rigid single body is exact rather than an approximation.\n\n'
        '  This is the task reviewers asked for by name: a naturally sized\n'
        '  electric plug. Its 1.5 mm blades are an order of magnitude below any\n'
        '  feature in the existing suite, and unlike every other task here it is\n'
        '  a TWO-feature insertion -- both blades enter at once, so the contact\n'
        '  itself constrains yaw.\n\n'
        '  visual    : boolean-cut shell (corner rounds, USB-C recess, seam)\n'
        '  collision : 3 exact boxes, body + two prongs\n'
        '-->\n'
        '<robot name="apple_20w_plug">\n'
        '  <link name="apple_20w_plug">\n'
        '    <visual>\n'
        '      <origin xyz="0 0 0" rpy="0 0 0"/>\n'
        '      <geometry><mesh filename="plug_visual.glb"/></geometry>\n'
        '    </visual>\n'
        f'{col}'
        f'{inertial_xml(solid, PLUG_MASS_KG)}'
        '  </link>\n'
        '</robot>\n')
    out = d / "plug.urdf"
    body = _scrub(body)
    ET.fromstring(body)
    out.write_text(body)
    return out


def _socket_collision_boxes(tol_mm: float):
    """Exact box list for the socket, mirroring G.build_socket().

    Duplicated here rather than imported because build_socket() returns a
    concatenated mesh and the URDF needs the individual boxes. Any change to the
    layout there must be mirrored here; the assertion in write_socket_urdf
    catches divergence.
    """
    c = tol_mm / 1000.0
    bw, bh, bd = G.SOCKET_BLOCK
    pt = 0.004
    boxes = [((bw, bh, bd - pt), (0.0, 0.0, -pt / 2))]
    slots = [(+G.BLADE_PITCH / 2, G.SLOT_W + 2 * c, G.SLOT_H_HOT + 2 * c),
             (-G.BLADE_PITCH / 2, G.SLOT_W + 2 * c, G.SLOT_H_NEU + 2 * c)]
    lo, hi = sorted(slots, key=lambda t: t[0])
    zc = bd / 2 - pt / 2
    for x0, x1 in ((-bw / 2, lo[0] - lo[1] / 2),
                   (lo[0] + lo[1] / 2, hi[0] - hi[1] / 2),
                   (hi[0] + hi[1] / 2, bw / 2)):
        if x1 - x0 > 1e-6:
            boxes.append(((x1 - x0, bh, pt), ((x0 + x1) / 2, 0.0, zc)))
    for x, sw, sh in slots:
        seg = bh / 2 - sh / 2
        if seg > 1e-6:
            for sgn in (+1, -1):
                boxes.append(((sw, seg, pt), (x, sgn * (sh / 2 + seg / 2), zc)))
    return boxes


def write_socket_urdf(tol_mm: float) -> Path:
    """Power board receptacle at a given per-side clearance."""
    d = ASSETS / "socket"
    d.mkdir(parents=True, exist_ok=True)
    tag = f"socket_tol{fmt_tol(tol_mm)}mm"

    G.build_socket_visual(tol_mm).export(d / f"{tag}_visual.glb")

    boxes = _socket_collision_boxes(tol_mm)
    ref = G.build_socket(tol_mm)
    solid = trimesh.util.concatenate([G.box(e, c) for e, c in boxes])
    assert np.allclose(solid.extents, ref.extents, atol=1e-6), (
        f"socket collision drifted from build_socket(): "
        f"{solid.extents} vs {ref.extents}")

    col = "".join(
        f'    <collision>\n'
        f'      <origin xyz="{cx:.6f} {cy:.6f} {cz:.6f}" rpy="0 0 0"/>\n'
        f'      <geometry><box size="{ex:.6f} {ey:.6f} {ez:.6f}"/></geometry>\n'
        f'    </collision>\n'
        for (ex, ey, ez), (cx, cy, cz) in boxes)
    mass = SOCKET_DENSITY * float(sum(e[0]*e[1]*e[2] for e, _ in boxes))

    bw, bh, bd = G.SOCKET_BLOCK
    body = (
        '<?xml version="1.0"?>\n'
        '<!--\n'
        f'  Power board receptacle, {tol_mm:g} mm clearance per side.\n\n'
        '  A power strip rather than a wall socket, because the receptacle has to\n'
        '  sit on a table: a wall outlet mounted on a table top is incoherent, and\n'
        '  earlier attempts to model one (with a wall panel, then a faceplate over\n'
        '  a body) imitated an installation that does not exist in this scene.\n\n'
        '  Blade slots at x = +/-6.35 mm, 1.6 mm wide, opening into the body so a\n'
        '  plug can seat its full 15.9 mm blade depth. Collision is exact boxes;\n'
        '  the visual carries the aluminium sleeve, plastic chassis, switch and\n'
        '  cable. Only the ACTIVE outlet is opened in collision -- the companion\n'
        '  outlet is visual, so nothing can enter it.\n\n'
        f'  Block {bw*1000:.0f} x {bh*1000:.0f} x {bd*1000:.0f} mm, origin at its\n'
        f'  centroid, so the Problem uses hole_z_offset = {bd/2:.4f}.\n'
        '-->\n'
        f'<robot name="{tag}">\n'
        '  <link name="hole">\n'
        '    <visual>\n'
        '      <origin xyz="0 0 0" rpy="0 0 0"/>\n'
        f'      <geometry><mesh filename="{tag}_visual.glb"/></geometry>\n'
        '    </visual>\n'
        f'{col}'
        f'{inertial_xml(solid, mass)}'
        '  </link>\n'
        '</robot>\n')
    out = d / f"{tag}.urdf"
    body = _scrub(body)
    ET.fromstring(body)
    out.write_text(body)
    return out


def write_holder_urdf(tol_mm: float, sw: tuple[float, float]) -> Path:
    """Flatware holder at a given per-side clearance."""
    c = tol_mm / 1000.0
    open_w, open_t = sw[0] + 2 * c, sw[1] + 2 * c
    d = ASSETS / "holder"
    d.mkdir(parents=True, exist_ok=True)
    tag = f"holder_tol{fmt_tol(tol_mm)}mm"

    vis = G.build_slot_visual_perforated(open_w, open_t)
    vis.export(d / f"{tag}_visual.obj")

    pw, ph, h = G.SLOT_BLOCK
    floor = 0.005
    wall_h, zc = h - floor, floor / 2
    boxes = [((pw, ph, floor), (0.0, 0.0, -h / 2 + floor / 2))]
    for x0, x1 in ((-pw / 2, -open_w / 2), (open_w / 2, pw / 2)):
        boxes.append(((x1 - x0, ph, wall_h), ((x0 + x1) / 2, 0.0, zc)))
    seg = ph / 2 - open_t / 2
    for sgn in (+1, -1):
        boxes.append(((open_w, seg, wall_h), (0.0, sgn * (open_t / 2 + seg / 2), zc)))

    col = "".join(
        f'    <collision>\n'
        f'      <origin xyz="{cx:.6f} {cy:.6f} {cz:.6f}" rpy="0 0 0"/>\n'
        f'      <geometry><box size="{ex:.6f} {ey:.6f} {ez:.6f}"/></geometry>\n'
        f'    </collision>\n'
        for (ex, ey, ez), (cx, cy, cz) in boxes
    )
    solid = trimesh.util.concatenate([G.box(e, ctr) for e, ctr in boxes])
    mass = HOLDER_DENSITY * float(sum(e[0] * e[1] * e[2] for e, _ in boxes))

    body = (
        '<?xml version="1.0"?>\n'
        '<!--\n'
        f'  Flatware holder, {tol_mm:g} mm clearance per side.\n\n'
        f'  Slot {open_w*1000:.2f} x {open_t*1000:.2f} mm, sized from the fork\'s SWEPT\n'
        f'  envelope over its leading half ({sw[0]*1000:.2f} x {sw[1]*1000:.2f} mm),\n'
        '  NOT from any single cross-section. The head is dished, so per-station\n'
        '  measurements understate what a straight slot has to clear, and a slot\n'
        '  cut to them would not admit the fork at all.\n\n'
        '  Collision is the four walls + floor as exact boxes. This is procedural\n'
        '  geometry, so boxes are the ground truth rather than an approximation;\n'
        '  the visual mesh adds rounds, a chamfered mouth and perforations, none\n'
        '  of which the physics needs.\n\n'
        f'  Block {G.SLOT_BLOCK[0]*1000:.0f} x {G.SLOT_BLOCK[1]*1000:.0f} x'
        f' {G.SLOT_BLOCK[2]*1000:.0f} mm, origin at its centroid, so the Problem\n'
        f'  must use hole_z_offset = {G.SLOT_BLOCK[2]/2:.4f} to seat it on the table.\n'
        '-->\n'
        f'<robot name="{tag}">\n'
        '  <material name="holder_steel"><color rgba="0.69 0.71 0.74 1.0"/></material>\n'
        '  <link name="hole">\n'
        '    <visual>\n'
        '      <origin xyz="0 0 0" rpy="0 0 0"/>\n'
        f'      <geometry><mesh filename="{tag}_visual.obj"/></geometry>\n'
        '      <material name="holder_steel"/>\n'
        '    </visual>\n'
        f'{col}'
        f'{inertial_xml(solid, mass)}'
        '  </link>\n'
        '</robot>\n'
    )
    out = d / f"{tag}.urdf"
    body = _scrub(body)
    ET.fromstring(body)
    out.write_text(body)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tolerances", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    args = ap.parse_args()

    fork = G.load_fork()
    sw = G.swept_envelope(fork, 0.5)
    print(f"fork  {fork.extents[0]*1000:.1f} x {fork.extents[1]*1000:.1f} x "
          f"{fork.extents[2]*1000:.1f} mm, watertight={fork.is_watertight}")
    print(f"swept envelope (handle-first leading half): "
          f"{sw[0]*1000:.2f} x {sw[1]*1000:.2f} mm")

    p = write_fork_urdf()
    print(f"  wrote {p.relative_to(REPO)}")
    p = write_plug_urdf()
    print(f"  wrote {p.relative_to(REPO)}")
    for tol in args.tolerances:
        p = write_holder_urdf(tol, sw)
        print(f"  wrote {p.relative_to(REPO)}")
    for tol in (0.1, 0.3, 0.5, 1.0):
        p = write_socket_urdf(tol)
        print(f"  wrote {p.relative_to(REPO)}")
    print(f"\nhole_z_offset for problems.py: {G.SLOT_BLOCK[2]/2:.4f} m")


if __name__ == "__main__":
    main()
