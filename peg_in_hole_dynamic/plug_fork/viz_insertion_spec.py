#!/usr/bin/env python3
"""Interactive viser tuner for the plug_fork insertion specs.

Two real-world-scale tasks, added because reviewers noted the existing suite
uses enlarged geometry (beam_3x is literally 3x) rather than everyday sizes:

  fork  YCB 030_fork (real scan, 198 mm) -> moulded flatware slot, tines first
  plug  Apple 20W USB-C adapter A2940 (41.5 x 42.5 x 27 mm, 1.5 mm blades)
        -> NEMA 5-15R receptacle, blades first

Everything derived rather than hand-set, because hand-set numbers were wrong
three times running here:

  * slot size comes from the fork's SWEPT envelope, not any single station.
    A fork head is dished, so it sweeps 27.1 x 12.2 mm while its widest station
    reads 27.3 x 6.7 -- a slot cut to the station would not admit it at all.
  * seated depth comes from where the object's own profile jams the opening.
  * pre-insert defaults to the object's tip exactly touching the opening,
    zero penetration.
  * tines-down is chosen by cross-sectional area, not by assuming the scan's
    frame. Handle-first clears a head-sized slot and drops straight through.

The GUI prints the resulting `insert_pose_rel_receptive` tuple for problems.py.
All annotation lives in the GUI panel; floating 3D labels became unreadable
once meshes overlapped.

    python peg_in_hole_dynamic/plug_fork/viz_insertion_spec.py [--port 8086]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import trimesh
import viser

REPO = Path(__file__).resolve().parents[2]
# YCB 030_fork, used two ways:
#   fork_collision.ply  geometry -- watertight, ONE component, 197.6 x 27.1 x
#                       15.9 mm, 19.60 cm^3
#   fork_visual.obj     appearance -- the same scan with its real texture
#
# They are the same scan in the same frame. The OBJ splits vertices at UV
# seams, which trimesh counts as 172 disconnected fragments and which sent an
# earlier version of this file down a long detour into voxel remeshing. The PLY
# has no such splitting and needs no repair at all.
FORK_PLY = REPO / "assets/urdf/plug_fork/fork/fork_collision.ply"
FORK_OBJ = REPO / "assets/urdf/plug_fork/fork/fork_visual.obj"
# CoACD convex decomposition, baked once (24 s) and cached. Only the FORK needs
# it -- the holder, socket and plug are procedural, so their box collisions are
# exact by construction rather than an approximation.
FORK_COACD_DIR = REPO / "assets/urdf/plug_fork/fork"

SLOT_BLOCK = (0.090, 0.070, 0.085)   # holder block the flatware slot cuts through

# Apple 20W USB-C adapter A2940 -- the iPhone 15-era charger. Measured
# 41.5 x 42.5 x 27 mm, 58.5 g (chargerlab teardown of A2940). An earlier
# version here used 27.2 x 27.2 x 30.5 mm from memory: wrong on all three
# axes. The real part is a flat slab, wider in footprint and shallower in
# depth. Its prongs are FIXED, not folding, so a rigid single body is exact
# rather than an approximation.
# Note the iPhone 15 ships with no adapter at all (Apple dropped them with
# iPhone 12); A2940 is the separately sold charger for it.
BODY = (0.0415, 0.0425, 0.0270)
BLADE_T, BLADE_W_HOT, BLADE_W_NEU = 0.0015, 0.0063, 0.0079
BLADE_LEN, BLADE_PITCH = 0.0159, 0.0127
PLATE = (0.114, 0.070, 0.0022)       # NEMA 5-15R faceplate
SLOT_W, SLOT_H_HOT, SLOT_H_NEU = 0.0016, 0.0066, 0.0082
SOCKET_BODY_D = 0.020
# Deliberately ONE outlet on a compact block, matching the fork holder's
# visual language. An earlier version modelled a full duplex receptacle
# flush-mounted in a 320 x 240 mm wall panel: more faithful to a real
# installation, but the wall dominated the scene and the second outlet added
# nothing the task uses. The holder reads well because it is a single block
# with the opening cut into it; the socket now does the same.
OUTLET_DY = 0.0
GROUND_DROP = 0.0119
GROUND_R = 0.0032
# Brushed-aluminium POWER BOARD rather than a wall receptacle. A wall socket
# is never a tabletop object, so mounting one on the table was incoherent; a
# power strip genuinely sits there. It also separates the receptacle from the
# white plug visually, which a cream faceplate did not.
#
# 120 x 56 x 30 mm. The height is UNCHANGED (30 mm) because every derived pose
# hangs off SOCKET_Z_OFFSET = bd/2 and off the top plane; only the footprint
# moved.
#
# It got SHORTER, not longer. The instinct is a 250-300 mm strip, which is what
# a mains extension normally is, but the object that inserts into it is a 41.5 x
# 42.5 mm charger: at 180 mm the board was already 4x the plug's footprint and
# read as furniture the plug happened to land on. 120 mm is 2.9x the plug's
# length, so a seated plug covers a third of the board and the two obviously
# belong to the same scene.
#
# That length forces the product type, and the forcing is the useful part. The
# ACTIVE outlet must sit at x = 0 (the receptive origin), so the board's centre
# is an outlet and every other feature is off to one side. 60 mm of half-length
# minus a 12 mm end cap minus half a 34 mm outlet well leaves 31 mm per side --
# enough for a switch on one side and a USB bay on the other, not remotely
# enough for a second 34 mm mains outlet. So this is a compact 1-gang trailing
# block with USB (Anker 511 / Belkin BoostCharge class, ~110 x 55 x 40 real),
# not a strip -- and a centred outlet flanked by two smaller features composes
# better for a figure whose subject IS that outlet.
SOCKET_BLOCK = (0.120, 0.056, 0.030)
USB_X = 0.0325             # USB-A bay, visual only -- says "power board"
SWITCH_X = -0.0325

# Receptives are modelled centred on their own origin, but the env places them
# at z = table_top_z + hole_z_offset, so hole_z_offset must be HALF the block
# height or the fixture sinks through the table. Both blocks are drawn here
# sitting on a table plane at z = 0 so that is visible rather than assumed.
HOLDER_Z_OFFSET = SLOT_BLOCK[2] / 2      # 0.0425 m
SOCKET_Z_OFFSET = SOCKET_BLOCK[2] / 2    # 0.0225 m
TABLE = (0.40, 0.32, 0.012)

C_HOLDER = (176, 181, 188)   # brushed stainless
C_PLATE = (246, 245, 241)
C_BOARD = (188, 192, 198)      # brushed aluminium
C_POCKET = (58, 60, 64)        # recessed outlet face
C_SWITCH = (196, 66, 54)
C_SOCKET_BODY = (222, 217, 205)
C_SCREW = (176, 170, 158)
C_PLUG = (246, 246, 248)
C_BLADE = (198, 172, 104)
C_FORK = (203, 207, 213)     # stainless
C_GHOST = (96, 150, 226)
C_COLLISION = (232, 108, 52)


def box(extents, xyz):
    m = trimesh.creation.box(extents=extents)
    m.apply_translation(xyz)
    return m


def rounded_box(extents, xyz, radius, subdiv=2):
    """Box with rounded edges, as the convex hull of corner spheres.

    Outer dimensions are preserved exactly (spheres sit on a box inset by
    `radius`). Sharp primitive boxes read as CAD blocks; a 1-3 mm round is what
    moulded plastic actually looks like, and on a slot mouth it doubles as the
    lead-in chamfer a real part would have.
    """
    ex, ey, ez = extents
    r = min(radius, ex / 2 - 1e-5, ey / 2 - 1e-5, ez / 2 - 1e-5)
    if r <= 1e-5:
        return box(extents, xyz)
    pts = []
    unit = trimesh.creation.icosphere(subdivisions=subdiv, radius=r)
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                s = unit.copy()
                s.apply_translation((sx * (ex / 2 - r),
                                     sy * (ey / 2 - r),
                                     sz * (ez / 2 - r)))
                pts.append(s.vertices)
    hull = trimesh.Trimesh(vertices=np.vstack(pts)).convex_hull
    hull.apply_translation(xyz)
    return hull


def swept_envelope(mesh, lead_fraction=0.5):
    """(width, thickness) of the leading portion in the FIXED frame -- what a
    straight slot must clear. Per-station cross-sections understate this
    whenever the part is dished or twisted."""
    z_lo = mesh.bounds[0][2]
    v = mesh.vertices[mesh.vertices[:, 2] < z_lo + lead_fraction * mesh.extents[2]]
    return (float(v[:, 0].max() - v[:, 0].min()),
            float(v[:, 1].max() - v[:, 1].min()))


def seated_depth(mesh, open_w, open_t, max_depth):
    """How far the object descends before its own profile jams the opening."""
    z_lo, z_hi = mesh.bounds[0][2], mesh.bounds[1][2]
    v, depth = mesh.vertices, 0.0
    for u in np.linspace(0.002, min(z_hi - z_lo, max_depth), 160):
        sel = v[v[:, 2] < z_lo + u]
        if len(sel) < 4:
            continue
        if (sel[:, 0].max() - sel[:, 0].min()) > open_w or \
           (sel[:, 1].max() - sel[:, 1].min()) > open_t:
            break
        depth = u
    return depth


def touch_pose_z(mesh, surface_z):
    """Centre-Z putting the object's lowest point exactly on `surface_z`."""
    return surface_z - float(mesh.bounds[0][2])


def load_coacd_hulls(orient_transform=None):
    """Cached CoACD hulls for the fork, in the same frame as load_fork().

    Replaces an axis-aligned slab decomposition that more than DOUBLED the
    fork's volume (46.5 vs 19.6 cm^3, +137%): it filled the dished head solid
    and could not represent the tine gaps at all. On a task whose whole content
    is a 1 mm clearance, that made the collision fork millimetres wider than the
    one being rendered. CoACD at threshold 0.03 gives 9 hulls at +15%, and the
    residual is mostly tine gaps, which no convex decomposition can capture.
    """
    hulls = []
    # Anchored pattern: "coacd_hull_*.ply" also matches the
    # "coacd_hull_XX_oriented.ply" files the generator WRITES, so a second run
    # loaded 18 hulls -- 9 raw plus 9 already-oriented -- and rotated the
    # latter twice. That renders as a second fork at 90 degrees.
    for p in sorted(FORK_COACD_DIR.glob("coacd_hull_[0-9][0-9].ply")):
        h = trimesh.load(p, process=False)
        if orient_transform is not None:
            h.apply_transform(orient_transform)
        hulls.append(h)
    return hulls


def collision_boxes(mesh, n_segments=6):
    """Approximate a scanned mesh by a stack of axis-aligned boxes.

    Superseded by load_coacd_hulls() for the fork; kept because it is the right
    tool for anything procedural and axis-aligned.

    Visual and collision geometry want opposite things: the render wants the
    real 9.6k-triangle scan, the simulator wants something PhysX can resolve
    cheaply and stably. Slicing along the insertion axis and taking each
    slab's bounding box gives a handful of primitives that an URDF can carry as
    plain <box> collisions.

    Conservative by construction -- each box covers its slab, so the collision
    hull contains the visual mesh and the part can never pass somewhere the
    render says it should not. That costs realism at the dished fork head,
    where the box is fuller than the scan; `n_segments` trades that off.
    """
    z0, z1 = mesh.bounds[0][2], mesh.bounds[1][2]
    edges = np.linspace(z0, z1, n_segments + 1)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = mesh.vertices[(mesh.vertices[:, 2] >= a) & (mesh.vertices[:, 2] <= b)]
        if len(sel) < 4:
            continue
        lo, hi = sel.min(axis=0), sel.max(axis=0)
        ext = (float(hi[0] - lo[0]), float(hi[1] - lo[1]), float(b - a))
        ctr = (float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2), float((a + b) / 2))
        if min(ext) > 1e-5:
            out.append((ext, ctr))
    return out


def boxes_to_mesh(boxes):
    return trimesh.util.concatenate([box(e, c) for e, c in boxes])


def _frustum(w0, t0, w1, t1, h, zc):
    """Tapered box as the hull of a bottom and a top rectangle -- used to cut
    a lead-in chamfer at a slot mouth."""
    pts = []
    for w, t, z in ((w0, t0, zc - h / 2), (w1, t1, zc + h / 2)):
        for sx in (-1, 1):
            for sy in (-1, 1):
                pts.append((sx * w / 2, sy * t / 2, z))
    return trimesh.Trimesh(vertices=np.array(pts)).convex_hull


def pbr(mesh, rgb, metallic=0.0, roughness=0.6):
    """Attach a PBR material. viser's add_mesh_simple only exposes
    standard/toon shading, but add_mesh_trimesh carries a trimesh
    PBRMaterial through, which is how the stainless parts get metal."""
    m = mesh.copy()
    m.visual = trimesh.visual.TextureVisuals(
        material=trimesh.visual.material.PBRMaterial(
            baseColorFactor=[int(rgb[0]), int(rgb[1]), int(rgb[2]), 255],
            metallicFactor=float(metallic),
            roughnessFactor=float(roughness)))
    return m


def build_slot_visual_perforated(open_w, open_t, chamfer=0.003,
                                 hole_d=0.007):
    """Stainless cutlery holder: rounded body, slot cut through, and a grid of
    perforations like a real dishwasher utensil basket.

    Perforations are visual only -- collision stays the four wall strips, so
    the holes cost nothing in sim. They are what makes it read as a utensil
    basket rather than a machined block.
    """
    base = build_slot_visual(open_w, open_t, chamfer)
    pw, ph, h = SLOT_BLOCK
    cuts = []
    for z in (-0.026, -0.004, 0.018):
        for y in (-0.021, 0.0, 0.021):
            c = trimesh.creation.cylinder(radius=hole_d / 2, height=pw + 0.02,
                                          sections=48)
            c.apply_transform(
                trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
            c.apply_translation((0.0, y, z))
            cuts.append(c)
        for x in (-0.030, 0.0, 0.030):
            c = trimesh.creation.cylinder(radius=hole_d / 2, height=ph + 0.02,
                                          sections=48)
            c.apply_transform(
                trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
            c.apply_translation((x, 0.0, z))
            cuts.append(c)
    try:
        return trimesh.boolean.difference([base] + cuts)
    except Exception:
        return base


def build_slot_visual(open_w, open_t, chamfer=0.003):
    """Holder as ONE solid with the slot actually cut through it.

    The collision version stacks four wall strips, which is right for physics
    but reads as separate blocks with seams at every join. manifold3d gives a
    real boolean, so the visual can be a single rounded body with a crisp slot
    and a chamfered mouth -- which is also what a moulded part looks like.
    """
    pw, ph, h = SLOT_BLOCK
    floor = 0.005
    outer = rounded_box((pw, ph, h), (0, 0, 0), 0.005, subdiv=4)
    cut_h = h - floor + 0.02
    cut_z = (-h / 2 + floor) + cut_h / 2
    cut = box((open_w, open_t, cut_h), (0, 0, cut_z))
    lead = _frustum(open_w, open_t,
                    open_w + 2 * chamfer, open_t + 2 * chamfer,
                    chamfer, h / 2 - chamfer / 2 + 1e-4)
    try:
        return trimesh.boolean.difference([outer, cut, lead])
    except Exception:
        return build_slot(open_w, open_t)


def _outlet_cuts(cx, c, top, deep=0.06):
    """Slot pair + ground hole for one outlet, centred at x=cx."""
    cuts = []
    for dx, sh in ((+BLADE_PITCH / 2, SLOT_H_HOT), (-BLADE_PITCH / 2, SLOT_H_NEU)):
        w, t = SLOT_W + 2 * c, sh + 2 * c
        cuts.append(box((w, t, deep), (cx + dx, 0.0, top - deep / 2 + 1e-4)))
        # _frustum() builds on the origin, so this MUST be translated. Without
        # the translate both lead-ins landed on top of each other at x=0, which
        # put a stray 3.8 x 8.8 mm notch in the middle of the receptacle face
        # and left the slots themselves with no chamfer at all -- invisible in
        # the numbers, obvious in a top-down depth map.
        lead = _frustum(w, t, w + 0.0022, t + 0.0022, 0.0011,
                        top - 0.00055 + 1e-4)
        lead.apply_translation((cx + dx, 0.0, 0.0))
        cuts.append(lead)
    g = trimesh.creation.cylinder(radius=GROUND_R, height=deep, sections=48)
    g.apply_translation((cx, -GROUND_DROP, top - deep / 2 + 1e-4))
    cuts.append(g)
    return cuts


def build_socket_visual(clearance_mm):
    """Compact power board, built the way a real metal one is built: an EXTRUDED
    ALUMINIUM SLEEVE slid over a moulded plastic chassis, the chassis showing at
    both ends as caps and along the bottom as a base rail.

    Four earlier attempts were rejected, and each rejection is a constraint here:

      * a receptacle in a wall panel -- a wall socket is not a tabletop object;
      * a faceplate over a recessed body -- the seam imitated a wall install
        that does not exist for a tabletop part;
      * a plain rounded block with three holes -- dimensionally right, no
        character, reads as a lump;
      * a rounded block with two pockets and a switch stuck on -- the details
        were APPLIED to a box rather than implied by how the thing is made.

    So the governing idea is a manufacturing story rather than a decoration
    list, which is also why the fork holder works: one body, real process
    features. Everything visible here follows from sleeve-over-chassis.

      * the horizontal reveal 9.5 mm up the side is where the sleeve stops and
        the chassis rail continues -- a 0.6 mm step, not a drawn line;
      * the same 0.6 mm step wraps the two end caps, because the caps are the
        chassis and the sleeve laps over them;
      * the two half-round flutes per side are extrusion die features. They are
        the cheapest possible signal that the metal was extruded rather than
        milled, and they run the sleeve's length and STOP at the caps, which is
        what makes the sleeve read as a separate part;
      * the top edges carry the extrusion's 6 mm radius; the caps are moulded to
        the same radius so the silhouette is continuous across the joint.

    Materials are the four a real one has: anodised aluminium, dark ABS (caps,
    base rail, switch bezel, receptacle face), the red rocker, black PVC cable.
    The plug is white polycarbonate, so the board reads as metal against it,
    which was the point of choosing a metal board in the first place.

    Layout: the ACTIVE outlet is at x = 0 because that is the receptive origin,
    so it is also the visual centre, flanked by the switch and the USB bay. The
    USB bay is what says "power board" at this length -- a second mains outlet
    does not fit (see SOCKET_BLOCK) and a lone outlet reads as a wall socket
    lying down. Neither the switch nor the USB slots are cut through the
    collision geometry, so nothing can enter them.

    Nothing dips below -bd/2: that plane is the table top. The cable is clamped
    to lie ON it, and the rocker and LED-height details go UP instead.
    """
    c = clearance_mm / 1000.0
    bw, bh, bd = SOCKET_BLOCK
    top, bot = bd / 2, -bd / 2

    CAP_L = 0.012           # moulded end cap, each end
    BASE_H = 0.0095         # plastic base rail: sleeve stops here
    REVEAL = 0.0006         # how far the sleeve stands proud of the chassis
    R_EDGE = 0.006          # extrusion corner radius
    WELL = (0.034, 0.038)   # outlet recess in plan. 34 mm across the blades and
    WELL_Y = -0.002         # 38 mm along the ground axis is the real NEMA 5-15R
    WELL_R = 0.005          # face proportion -- the face is TALLER than it is
    WELL_D = 0.005          # wide, because the ground pin sits 11.9 mm below the
    FACE_GAP = 0.0005       # blades. Nudged -2 mm in y so the cluster is not
    FACE_DROP = 0.0008      # crammed against the bottom rim.
    SW = (0.019, 0.013)     # rocker aperture
    SW_D = 0.0045
    USB = (0.021, 0.030)    # USB-A bay: two 12.5 x 5 mm ports stacked in y
    EPS = 0.0003            # deliberate interpenetration; kills coplanar z-fight

    x_sleeve = bw / 2 - CAP_L
    base_top = bot + BASE_H
    well_floor = top - WELL_D
    face_top = top - FACE_DROP

    # ---- rounded-rectangle prisms -------------------------------------------
    # rounded_box() rounds all twelve edges, which is right for a moulded body
    # and wrong for a milled pocket: a real outlet well has VERTICAL walls with
    # a plan-view corner radius and a crisp rim. Hulling two rounded-rectangle
    # rings gives exactly that, and stays convex so convex_hull is exact.
    def _rr_ring(w, d, z, r, n=20):
        a = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
        rings = []
        for sx in (-1, 1):
            for sy in (-1, 1):
                rings.append(np.column_stack([
                    sx * (w / 2 - r) + r * np.cos(a),
                    sy * (d / 2 - r) + r * np.sin(a),
                    np.full(n, z)]))
        return np.vstack(rings)

    def _rr_solid(w0, d0, r0, w1, d1, r1, z0, z1, xy):
        v = np.vstack([_rr_ring(w0, d0, z0, r0), _rr_ring(w1, d1, z1, r1)])
        v[:, 0] += xy[0]
        v[:, 1] += xy[1]
        return trimesh.Trimesh(vertices=v).convex_hull

    def _rr_prism(w, d, r, z0, z1, xy):
        return _rr_solid(w, d, r, w, d, r, z0, z1, xy)

    # ---- swept tube ---------------------------------------------------------
    # A cable is the one part of this that is not a boolean of primitives.
    # Built as an explicit indexed tube with parallel-transported frames rather
    # than a union of capsules: watertight by construction (every quad shares
    # its vertices), no boolean cost, and the radius can vary along the path so
    # the strain-relief boot is the SAME solid as the cable rather than a cone
    # parked on the end cap.
    def _tube(pts, radii, n=18):
        pts = np.asarray(pts, float)
        tan = np.gradient(pts, axis=0)
        tan /= np.linalg.norm(tan, axis=1)[:, None]
        nrm = np.cross(tan[0], [0.0, 0.0, 1.0])
        if np.linalg.norm(nrm) < 1e-9:
            nrm = np.cross(tan[0], [0.0, 1.0, 0.0])
        nrm /= np.linalg.norm(nrm)
        ang = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
        cs, sn = np.cos(ang)[:, None], np.sin(ang)[:, None]
        rings = []
        for i, p in enumerate(pts):
            nrm = nrm - tan[i] * float(nrm @ tan[i])
            nrm /= np.linalg.norm(nrm)
            bin_ = np.cross(tan[i], nrm)
            rings.append(p + radii[i] * (cs * nrm + sn * bin_))
        m = len(pts)
        verts = np.vstack(rings + [pts[0][None, :], pts[-1][None, :]])
        faces = []
        for i in range(m - 1):
            for j in range(n):
                a0, b0 = i * n + j, i * n + (j + 1) % n
                c0, d0 = (i + 1) * n + j, (i + 1) * n + (j + 1) % n
                faces += [[a0, b0, d0], [a0, d0, c0]]
        s0, s1 = m * n, m * n + 1
        for j in range(n):
            faces.append([s0, (j + 1) % n, j])
            faces.append([s1, (m - 1) * n + j, (m - 1) * n + (j + 1) % n])
        tube = trimesh.Trimesh(vertices=verts, faces=np.array(faces),
                               process=False)
        tube.fix_normals()
        return tube

    def _catmull(waypoints, per=12):
        w = np.asarray(waypoints, float)
        w = np.vstack([2 * w[0] - w[1], w, 2 * w[-1] - w[-2]])
        u = np.linspace(0.0, 1.0, per, endpoint=False)[:, None]
        out = []
        for i in range(len(w) - 3):
            p0, p1, p2, p3 = w[i], w[i + 1], w[i + 2], w[i + 3]
            out.append(0.5 * (2 * p1 + (p2 - p0) * u
                              + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u ** 2
                              + (3 * p1 - 3 * p2 + p3 - p0) * u ** 3))
        return np.vstack(out + [w[-2][None, :]])

    # ---- aluminium sleeve ---------------------------------------------------
    # Built oversized and trimmed rather than sized directly, because the ends
    # must be FLAT where they lap the caps while the top edges keep the
    # extrusion radius. rounded_box would round the ends too and leave a gap at
    # the joint.
    sleeve = rounded_box((bw, bh, bd + 0.012), (0, 0, top - (bd + 0.012) / 2),
                         R_EDGE, subdiv=4)
    z0, z1 = base_top - EPS, top + 0.01
    sleeve = trimesh.boolean.intersection([
        sleeve, box((2 * x_sleeve, bh + 0.02, z1 - z0), (0, 0, (z0 + z1) / 2))])

    # extrusion flutes. Axis placed 0.4 mm OUTSIDE the flank so the groove meets
    # the surface at an angle: tangent to the flank it would be a knife edge,
    # which shades badly and is a poor boolean.
    flutes = []
    for sy in (-1, 1):
        for fz in (-0.001, 0.005):
            f = trimesh.creation.cylinder(radius=0.0016, height=bw, sections=40)
            f.apply_transform(
                trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
            f.apply_translation((0.0, sy * (bh / 2 + 0.0004), fz))
            flutes.append(f)

    # outlet well, plus a 0.8 mm lead-in flare at the rim
    wells = [
        _rr_prism(WELL[0], WELL[1], WELL_R, well_floor, top + 0.01, (0.0, WELL_Y)),
        _rr_solid(WELL[0], WELL[1], WELL_R,
                  WELL[0] + 0.0016, WELL[1] + 0.0016, WELL_R + 0.0008,
                  top - 0.0008, top + 0.001, (0.0, WELL_Y))]
    sw_well = _rr_prism(SW[0], SW[1], 0.0016, top - SW_D, top + 0.01, (SWITCH_X, 0.0))
    usb_well = _rr_prism(USB[0], USB[1], 0.004, top - 0.0025, top + 0.01,
                         (USB_X, WELL_Y))
    # The port apertures are driven 7 mm into the aluminium as well as through
    # the bay face. Stopping them at the face would back each port with a bright
    # machined floor; a real USB-A port is a dark cavity, and 7 mm of unlit slot
    # is the cheapest way to get one.
    usb_ports = [box((0.0125, 0.0050, 0.009), (USB_X, WELL_Y + dy, top - 0.0025))
                 for dy in (-0.0045, +0.0045)]

    cuts = _outlet_cuts(0.0, c, face_top)
    try:
        sleeve = trimesh.boolean.difference(
            [sleeve] + flutes + wells + [sw_well, usb_well] + usb_ports + cuts)
    except Exception:
        return build_socket(clearance_mm)

    # ---- plastic chassis: base rail + both end caps, one moulding -----------
    # Made as a full-size solid with the sleeve's volume subtracted, so the caps
    # and the rail are guaranteed to meet with no seam of their own. Inset
    # REVEAL on the flanks and the top, which is what produces the step the
    # sleeve reads against.
    chassis = rounded_box((bw, bh - 2 * REVEAL, bd - REVEAL),
                          (0, 0, bot + (bd - REVEAL) / 2), R_EDGE, subdiv=4)
    hz0, hz1 = base_top, top + 0.02
    chassis = trimesh.boolean.difference([
        chassis, box((2 * (x_sleeve - EPS), bh + 0.02, hz1 - hz0),
                     (0, 0, (hz0 + hz1) / 2))])

    # rocker bezel: a ring, so the rocker's low end can drop through it into
    # the switch cavity instead of floating on a flat face.
    bezel = trimesh.boolean.difference([
        _rr_prism(SW[0] - 0.001, SW[1] - 0.001, 0.0016,
                  top - SW_D - 0.0002, top + 0.0012, (SWITCH_X, 0.0)),
        _rr_prism(SW[0] - 0.0038, SW[1] - 0.0038, 0.0012,
                  top - SW_D - 0.01, top + 0.01, (SWITCH_X, 0.0))])
    # USB bay face, with the two port apertures cut through it
    usb_face = _rr_prism(USB[0] - 2 * FACE_GAP, USB[1] - 2 * FACE_GAP, 0.0035,
                         top - 0.0027, top - 0.0005, (USB_X, WELL_Y))
    usb_face = trimesh.boolean.difference(
        [usb_face] + [box((0.0125, 0.0050, 0.009),
                          (USB_X, WELL_Y + dy, top - 0.0025))
                      for dy in (-0.0045, +0.0045)])
    chassis = trimesh.util.concatenate([chassis, bezel, usb_face])

    # ---- receptacle face ----------------------------------------------------
    # A separate solid dropped into the well with a 0.5 mm gap all round and its
    # top 0.8 mm below the board's, so the rim throws a shadow line. Cutting the
    # slots into the aluminium alone would have made the outlet a hole in metal;
    # a real receptacle is a black moulding sitting in a metal aperture.
    face = _rr_prism(WELL[0] - 2 * FACE_GAP, WELL[1] - 2 * FACE_GAP,
                     WELL_R - FACE_GAP, well_floor, face_top, (0.0, WELL_Y))
    face = trimesh.boolean.difference([face] + _outlet_cuts(0.0, c, face_top))

    # ---- rocker -------------------------------------------------------------
    # Tilted 6.5 deg, which is the whole point: a flat rocker reads as a red
    # sticker. The low end buries itself in the switch cavity and the high end
    # stands 1.2 mm above the board.
    rocker = rounded_box((0.0138, 0.0080, 0.0038), (0, 0, 0), 0.0009, subdiv=3)
    rocker.apply_transform(
        trimesh.transformations.rotation_matrix(np.deg2rad(6.5), [0, 1, 0]))
    rocker.apply_translation((SWITCH_X, 0.0, top - 0.0016))

    # ---- cable --------------------------------------------------------------
    # 7.5 mm SJT flex leaving the +x cap, swelling to a 14 mm strain-relief boot
    # where it enters the moulding. It descends to the table over ~55 mm and
    # then runs flat: the z clamp below is what guarantees it never crosses the
    # table plane, since a Catmull-Rom through the waypoints can undershoot.
    path = _catmull([(0.053, 0.000, -0.0020),
                     (0.076, 0.000, -0.0026),
                     (0.096, 0.006, -0.0060),
                     (0.111, 0.020, -0.0098),
                     (0.120, 0.040, -0.0113),
                     (0.122, 0.062, -0.0113),
                     (0.117, 0.084, -0.0113),
                     (0.106, 0.104, -0.0113)])
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    radii = np.interp(arc, [0.0, 0.006, 0.020, 0.030],
                      [0.0072, 0.0068, 0.0042, 0.00375])
    path[:, 2] = np.maximum(path[:, 2], bot + radii + 1e-6)
    cable = _tube(path, radii)

    scene = trimesh.Scene()
    scene.add_geometry(pbr(sleeve, C_BOARD, 0.94, 0.32), geom_name="sleeve")
    scene.add_geometry(pbr(chassis, (44, 46, 50), 0.0, 0.62), geom_name="chassis")
    scene.add_geometry(pbr(face, C_POCKET, 0.0, 0.48), geom_name="receptacle")
    scene.add_geometry(pbr(rocker, C_SWITCH, 0.0, 0.42), geom_name="rocker")
    scene.add_geometry(pbr(cable, (26, 26, 28), 0.0, 0.86), geom_name="cable")
    return scene


def build_plug_visual():
    """Adapter shell: Apple's squircle round, the USB-C port recessed into the
    face opposite the blades, and the moulded seam where the two shell halves
    meet. Returned as a Scene so the shell and the prongs carry different
    materials -- white polycarbonate against brass."""
    shell = rounded_box(BODY, (0, 0, 0), 0.007, subdiv=4)
    port = rounded_box((0.00834, 0.00256, 0.006),
                       (0, 0, BODY[2] / 2 - 0.002), 0.0011, subdiv=2)
    # seam: a shallow groove around the body at mid height
    seam_o = rounded_box((BODY[0] + 0.001, BODY[1] + 0.001, 0.0009),
                         (0, 0, 0), 0.0004, subdiv=2)
    seam_i = rounded_box((BODY[0] - 0.0009, BODY[1] - 0.0009, 0.002),
                         (0, 0, 0), 0.0004, subdiv=2)
    try:
        shell = trimesh.boolean.difference([shell, port])
        groove = trimesh.boolean.difference([seam_o, seam_i])
        shell = trimesh.boolean.difference([shell, groove])
    except Exception:
        pass
    scene = trimesh.Scene()
    scene.add_geometry(pbr(shell, C_PLUG, 0.0, 0.35), geom_name="shell")
    scene.add_geometry(pbr(build_blades(), C_BLADE, 0.90, 0.28), geom_name="prongs")
    return scene


def build_slot(open_w, open_t, radius=0.0015):
    """Holder block with one rectangular through-slot: four rounded walls on a
    floor. Strips rather than boolean subtraction, so the opening is exact by
    arithmetic; the rounding gives the mouth a lead-in."""
    pw, ph, h = SLOT_BLOCK
    floor = 0.005
    wall_h, zc = h - floor, floor / 2
    parts = [rounded_box((pw, ph, floor), (0, 0, -h / 2 + floor / 2), radius)]
    for x0, x1 in ((-pw / 2, -open_w / 2), (open_w / 2, pw / 2)):
        if x1 - x0 > 1e-6:
            parts.append(rounded_box((x1 - x0, ph, wall_h),
                                     ((x0 + x1) / 2, 0, zc), radius))
    seg = ph / 2 - open_t / 2
    if seg > 1e-6:
        for sgn in (+1, -1):
            parts.append(rounded_box((open_w, seg, wall_h),
                                     (0, sgn * (open_t / 2 + seg / 2), zc), radius))
    return trimesh.util.concatenate(parts)


def build_plug():
    """Adapter body with Apple's ~4 mm corner round."""
    return rounded_box(BODY, (0, 0, 0), 0.004, subdiv=3)


def build_blades():
    return trimesh.util.concatenate([
        rounded_box((BLADE_T, BLADE_W_HOT, BLADE_LEN),
                    (+BLADE_PITCH / 2, 0, -BODY[2] / 2 - BLADE_LEN / 2), 0.0003),
        rounded_box((BLADE_T, BLADE_W_NEU, BLADE_LEN),
                    (-BLADE_PITCH / 2, 0, -BODY[2] / 2 - BLADE_LEN / 2), 0.0003)])


def socket_collision_boxes(clearance_mm):
    """Box list for the socket collision: floor, four cavity walls, slotted face.

    THE SINGLE SOURCE OF TRUTH. create_assets.py used to carry its own copy of
    this layout, so fixing one left the other stale -- the URDFs kept a solid
    body long after this function grew a cavity, and the extents-only assert
    that was supposed to catch drift passed because the OUTSIDE still matched.

    The bug that motivated the cavity: the body was one solid 120 x 56 x 26 mm
    box with slots cut only through the 4 mm top face, so blades entered 4 mm and
    hit solid geometry. They are 15.9 mm long, so the plug could never seat -- and
    it still scored as success, because a plug sitting 12 mm proud is inside the
    15 mm keypoint threshold.
    """
    c = clearance_mm / 1000.0
    bw, bh, bd = SOCKET_BLOCK
    pt = 0.004
    top = bd / 2
    face_bot = top - pt

    cav_w = 2 * (BLADE_PITCH / 2 + (SLOT_W + 2 * c) / 2) + 0.006
    cav_h = max(SLOT_H_HOT, SLOT_H_NEU) + 2 * c + 0.010
    cav_bot = face_bot - (BLADE_LEN + 0.004)
    assert cav_bot > -bd / 2, "cavity deeper than the block"

    boxes = []
    floor_t = cav_bot - (-bd / 2)
    boxes.append(((bw, bh, floor_t), (0.0, 0.0, -bd / 2 + floor_t / 2)))
    wall_h = face_bot - cav_bot
    zc = cav_bot + wall_h / 2
    for x0, x1 in ((-bw / 2, -cav_w / 2), (cav_w / 2, bw / 2)):
        boxes.append(((x1 - x0, bh, wall_h), ((x0 + x1) / 2, 0.0, zc)))
    seg = bh / 2 - cav_h / 2
    if seg > 1e-6:
        for sgn in (+1, -1):
            boxes.append(((cav_w, seg, wall_h),
                          (0.0, sgn * (cav_h / 2 + seg / 2), zc)))
    slots = [(+BLADE_PITCH / 2, SLOT_W + 2 * c, SLOT_H_HOT + 2 * c),
             (-BLADE_PITCH / 2, SLOT_W + 2 * c, SLOT_H_NEU + 2 * c)]
    lo, hi = sorted(slots, key=lambda t: t[0])
    fz = top - pt / 2
    for x0, x1 in ((-bw / 2, lo[0] - lo[1] / 2),
                   (lo[0] + lo[1] / 2, hi[0] - hi[1] / 2),
                   (hi[0] + hi[1] / 2, bw / 2)):
        if x1 - x0 > 1e-6:
            boxes.append(((x1 - x0, bh, pt), ((x0 + x1) / 2, 0.0, fz)))
    for x, sw, sh in slots:
        s2 = bh / 2 - sh / 2
        if s2 > 1e-6:
            for sgn in (+1, -1):
                boxes.append(((sw, s2, pt), (x, sgn * (sh / 2 + s2 / 2), fz)))
    return boxes


def build_socket(clearance_mm):
    """Collision mesh for the viewer; geometry comes from socket_collision_boxes."""
    return trimesh.util.concatenate(
        [box(e, c) for e, c in socket_collision_boxes(clearance_mm)])


def _fork_orient_transform(m):
    """Long axis on Z, HANDLE pointing -Z (the chosen entry direction).

    Which end lands down depends on the scan frame, so pick it by THICKNESS:
    a fork head is wider but thinner than the handle, so the two nearly cancel
    in cross-sectional area (measured 290 vs 304 mm^2 -- a 5% gap that picked
    the wrong end). Thickness separates cleanly, ~6.7 vs ~14.4 mm.
    """
    T = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
    w = m.copy()
    w.apply_translation(-w.bounds.mean(axis=0))
    w.apply_transform(T)
    w.apply_translation(-w.bounds.mean(axis=0))

    def end_thickness(low):
        z0, z1 = w.bounds[0][2], w.bounds[1][2]
        span = 0.2 * (z1 - z0)
        sel = (w.vertices[:, 2] < z0 + span) if low else (w.vertices[:, 2] > z1 - span)
        v = w.vertices[sel]
        return float(min(v[:, 0].max() - v[:, 0].min(),
                         v[:, 1].max() - v[:, 1].min()))

    # tines are the thin end; default entry is HANDLE first, so tines go UP
    flip = end_thickness(low=True) < end_thickness(low=False)
    full = trimesh.transformations.concatenate_matrices(
        T, trimesh.transformations.translation_matrix(-m.bounds.mean(axis=0)))
    if flip:
        full = trimesh.transformations.concatenate_matrices(
            trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]), full)
    probe = m.copy(); probe.apply_transform(full)
    full = trimesh.transformations.concatenate_matrices(
        trimesh.transformations.translation_matrix(-probe.bounds.mean(axis=0)), full)
    return full


def load_fork():
    """Watertight PLY geometry, oriented handle-down."""
    m = trimesh.load(FORK_PLY, process=False)
    m.apply_transform(_fork_orient_transform(m))
    return m


def load_fork_scene():
    """Textured OBJ for rendering, in the SAME frame as load_fork().

    Kept as a Scene and handed to viser as GLB so the texture image survives.
    Converting to vertex colours drops the UVs and the asset renders untextured.
    """
    raw = trimesh.load(FORK_OBJ, process=False)
    scene = raw if isinstance(raw, trimesh.Scene) else trimesh.Scene(raw)
    geom = trimesh.load(FORK_PLY, process=False)
    scene.apply_transform(_fork_orient_transform(geom))
    return scene


def tint(mesh, rgb):
    m = mesh.copy()
    m.visual = trimesh.visual.ColorVisuals(
        mesh=m, vertex_colors=np.tile(np.array([*rgb, 255], np.uint8),
                                      (len(m.vertices), 1)))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8086)
    args = ap.parse_args()

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/grid", width=0.5, height=0.5, cell_size=0.02)
    # Table top at z = 0; every receptive must sit ON this, never through it.
    _tbl = box(TABLE, (0, 0, -TABLE[2] / 2))
    server.scene.add_mesh_simple("/table", vertices=_tbl.vertices,
                                 faces=_tbl.faces, color=(126, 106, 84),
                                 flat_shading=True)

    fork = load_fork()
    fork_scene = load_fork_scene()
    _fork_raw = trimesh.load(FORK_PLY, process=False)
    fork_hulls = load_coacd_hulls(_fork_orient_transform(_fork_raw))
    fork_hulls_flipped = []
    for h in fork_hulls:
        hh = h.copy()
        hh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
        fork_hulls_flipped.append(hh)
    fork_len = float(fork.extents[2])
    fork_flipped = fork.copy()
    fork_flipped.apply_transform(
        trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
    fork_flipped.apply_translation(-fork_flipped.bounds.mean(axis=0))
    plug_full = trimesh.util.concatenate([build_plug(), build_blades()])

    handles: dict = {}

    def clear():
        for h in handles.values():
            h.remove()
        handles.clear()

    gui = server.gui
    task_dd = gui.add_dropdown("Task", ("fork", "plug"), initial_value="fork")
    with gui.add_folder("Pre-insert waypoint"):
        s_dz = gui.add_slider("lift above touching (mm)", 0, 150, 1, 0)
        s_dx = gui.add_slider("lateral x (mm)", -40, 40, 1, 0)
        s_dy = gui.add_slider("lateral y (mm)", -40, 40, 1, 0)
        s_tilt = gui.add_slider("tilt about y (deg)", -30, 30, 1, 0)
    with gui.add_folder("Receptive"):
        s_clear = gui.add_slider("clearance per side (mm)", 0.05, 6.0, 0.05, 1.00)
    with gui.add_folder("Fork orientation"):
        # Genuine task choice, not a correctness one: real dishwashers get
        # loaded both ways. Default is HANDLE first -- it reads better with the
        # tines up out of the holder, and it is the harder entry: the handle is
        # a blunt 15.5 x 18.8 mm end with no taper to self-align, so the slot
        # is sized to it and the approach has to be more accurate.
        c_tines = gui.add_checkbox("tines first (else handle first)", False)
    with gui.add_folder("Display"):
        mode_dd = gui.add_dropdown("geometry", ("visual", "collision", "both"),
                                   initial_value="visual")
        c_pre = gui.add_checkbox("pre-insert pose", True)
        c_final = gui.add_checkbox("final (seated) pose", True)
        c_recep = gui.add_checkbox("receptive", True)
        c_env = gui.add_checkbox("swept envelope", False)
    info = gui.add_markdown("")
    pose_out = gui.add_markdown("")

    def rebuild(_=None):
        clear()
        dz, dx, dy = s_dz.value / 1000, s_dx.value / 1000, s_dy.value / 1000
        q = trimesh.transformations.quaternion_about_axis(
            np.deg2rad(s_tilt.value), [0, 1, 0])          # wxyz
        c = s_clear.value / 1000
        mode = mode_dd.value
        show_vis = mode in ("visual", "both")
        show_col = mode in ("collision", "both")
        col_op = 0.40 if mode == "both" else 1.0

        def draw_vis(name, mesh, pos=(0, 0, 0), colour=None):
            if not show_vis:
                return
            m = mesh if colour is None else tint(mesh, colour)
            handles[name] = server.scene.add_mesh_trimesh(f"/{name}", m, position=pos)

        def draw_col(name, mesh, pos):
            if not show_col:
                return
            handles[name] = server.scene.add_mesh_simple(
                f"/{name}", vertices=mesh.vertices, faces=mesh.faces,
                color=C_COLLISION, opacity=col_op, flat_shading=True,
                position=pos)

        if task_dd.value == "fork":
            # load_fork() already returns the fork HANDLE-DOWN, so the
            # checkbox has to select the flipped copy for tines-first. The
            # inverse of this read handle-first as tines-first.
            fk = fork_flipped if c_tines.value else fork
            sw_w, sw_t = swept_envelope(fk, 0.5)
            open_w, open_t = sw_w + 2 * c, sw_t + 2 * c
            base_z = HOLDER_Z_OFFSET          # centre height with base on table
            top_z = base_z + SLOT_BLOCK[2] / 2
            floor_limit = SLOT_BLOCK[2] - 0.007
            depth = seated_depth(fk, open_w, open_t, floor_limit)
            stop = "floor-limited" if depth >= floor_limit - 1e-4 else "jam-limited"
            z_final = top_z - depth + fork_len / 2
            z_touch = touch_pose_z(fk, top_z)

            if c_recep.value:
                draw_vis("holder", pbr(build_slot_visual_perforated(open_w, open_t),
                                       C_HOLDER, metallic=0.95, roughness=0.28),
                         (0, 0, base_z))
                draw_col("holder_col", build_slot(open_w, open_t, radius=0.0),
                         (0, 0, base_z))
            hulls = fork_hulls_flipped if c_tines.value else fork_hulls
            if c_final.value:
                if show_vis:
                    # Export the textured scene to GLB so the real YCB texture
                    # reaches the browser; vertex-colour conversion drops UVs.
                    sc = fork_scene.copy()
                    sc.apply_translation((0, 0, z_final))
                    handles["fork"] = server.scene.add_glb(
                        "/fork", glb_data=sc.export(file_type="glb"))
                draw_col("fork_col", trimesh.util.concatenate(hulls),
                         (0, 0, z_final))
            if c_pre.value:
                handles["pre"] = server.scene.add_mesh_simple(
                    "/fork_pre", vertices=fk.vertices, faces=fk.faces,
                    color=C_GHOST, opacity=0.38, flat_shading=False,
                    position=(dx, dy, z_touch + dz), wxyz=q)
            if c_env.value:
                e = box((sw_w, sw_t, max(depth, 1e-3)), (0, 0, top_z - depth / 2))
                handles["env"] = server.scene.add_mesh_simple(
                    "/env", vertices=e.vertices, faces=e.faces,
                    color=(224, 126, 58), opacity=0.20, flat_shading=True)

            lead = "tines" if c_tines.value else "handle"
            info.content = (
                f"**Fork -> flatware slot**  (YCB 030_fork, real scan)\n\n"
                f"- entering **{lead} first**\n"
                f"- swept envelope of leading half "
                f"**{sw_w*1000:.1f} x {sw_t*1000:.1f} mm**\n"
                f"- slot {open_w*1000:.1f} x {open_t*1000:.1f} mm "
                f"-> **{s_clear.value:.2f} mm/side**\n"
                f"- seated depth **{depth*1000:.1f} mm** of a "
                f"{SLOT_BLOCK[2]*1000:.0f} mm block ({stop})\n"
                f"- collision: **{len(hulls)} CoACD hulls** "
                f"({sum(len(h.faces) for h in hulls)} tris, +15% volume) vs "
                f"{len(fk.faces)} visual triangles\n"
                f"- pre-insert = tip touching the mouth, +{s_dz.value:.0f} mm lift")
            rel_z = (z_touch + dz) - top_z
        else:
            base_z = SOCKET_Z_OFFSET
            top_z = base_z + SOCKET_BLOCK[2] / 2
            # Seated = body face FLUSH with the socket face, blades filling the
            # cavity. Subtracting BLADE_LEN here (an earlier bug) sank the body
            # 15.9 mm INTO the block: the blades already hang below the body, so
            # they need no extra allowance.
            z_seat = top_z + BODY[2] / 2
            z_touch = touch_pose_z(plug_full, top_z)
            plug_col = trimesh.util.concatenate([
                box(BODY, (0, 0, 0)),
                box((BLADE_T, BLADE_W_HOT, BLADE_LEN),
                    (+BLADE_PITCH / 2, 0, -BODY[2] / 2 - BLADE_LEN / 2)),
                box((BLADE_T, BLADE_W_NEU, BLADE_LEN),
                    (-BLADE_PITCH / 2, 0, -BODY[2] / 2 - BLADE_LEN / 2))])
            if c_recep.value:
                sock = build_socket_visual(s_clear.value)
                if show_vis:
                    if isinstance(sock, trimesh.Scene):
                        sc = sock.copy()
                        sc.apply_translation((0, 0, base_z))
                        handles["plate"] = server.scene.add_glb(
                            "/socket", glb_data=sc.export(file_type="glb"))
                    else:
                        draw_vis("plate", pbr(sock, C_PLATE, 0.0, 0.75),
                                 (0, 0, base_z))
                draw_col("plate_col", build_socket(s_clear.value), (0, 0, base_z))
            if c_final.value:
                if show_vis:
                    pl = build_plug_visual()
                    pl = pl.copy()
                    pl.apply_translation((0, OUTLET_DY, z_seat))
                    handles["plug"] = server.scene.add_glb(
                        "/plug", glb_data=pl.export(file_type="glb"))
                draw_col("plug_col", plug_col, (0, OUTLET_DY, z_seat))
            if c_pre.value:
                handles["pre"] = server.scene.add_mesh_simple(
                    "/plug_pre", vertices=plug_full.vertices,
                    faces=plug_full.faces, color=C_GHOST, opacity=0.38,
                    flat_shading=False, position=(dx, dy + OUTLET_DY, z_touch + dz), wxyz=q)
            info.content = (
                f"**Plug -> power board socket**  (Apple 20W, spec)\n\n"
                f"- body {BODY[0]*1000:.1f} x {BODY[1]*1000:.1f} x "
                f"{BODY[2]*1000:.1f} mm, blades {BLADE_T*1000:.1f} mm thick "
                f"at {BLADE_PITCH*1000:.1f} mm pitch\n"
                f"- slots widened **{s_clear.value:.2f} mm/side**\n"
                f"- blade insertion depth {BLADE_LEN*1000:.1f} mm\n"
                f"- collision: **3 boxes** (body + 2 blades); visual is the "
                f"rounded shell\n"
                f"- pre-insert = blade tips touching the plate, "
                f"+{s_dz.value:.0f} mm lift")
            rel_z = (z_touch + dz) - top_z

        pose_out.content = (
            "```python\n# pre-insert, receptive frame (x,y,z,qx,qy,qz,qw)\n"
            f"({dx:+.4f}, {dy:+.4f}, {rel_z:+.4f}, "
            f"{q[1]:+.6f}, {q[2]:+.6f}, {q[3]:+.6f}, {q[0]:+.6f})\n```")

    for w in (task_dd, s_dz, s_dx, s_dy, s_tilt, s_clear, c_tines,
              mode_dd, c_pre, c_final, c_recep, c_env):
        w.on_update(rebuild)
    rebuild()

    print(f"\n  plug_fork tuner -> http://localhost:{args.port}")
    for nm, m in (("handle first", fork), ("tines first", fork_flipped)):
        w, t = swept_envelope(m, 0.5)
        print(f"  {nm:13s} swept envelope {w*1000:5.1f} x {t*1000:5.1f} mm")
    print(f"  fork length {fork_len*1000:.0f} mm; default entry = handle first")
    print("  pre-insert defaults to tip-touching; lift slider raises it\n")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
