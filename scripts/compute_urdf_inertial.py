#!/usr/bin/env python3
"""Compute a URDF link's <inertial> block from its collision geometry.

The insertion-fixture URDFs were authored for *kinematic* bodies, where PhysX
ignores mass properties entirely -- so they carry either a placeholder inertial
(the peg hole had mass=10 kg, I=0.1, COM above the part) or none at all, in
which case the importer invents a default. Both are meaningless once
`peg_in_hole.fixture_bolted=False` makes the fixture dynamic.

This derives mass, COM and the full inertia tensor from the actual collision
geometry at a given material density, handling boxes (with rpy rotation) and
triangle meshes, combined via the parallel-axis theorem.

    python scripts/compute_urdf_inertial.py <urdf> [--density 1240] [--write]

Default density is solid PLA (1240 kg/m^3).
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def _rpy_to_R(rpy) -> np.ndarray:
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])


def _origin(el):
    o = el.find("origin")
    if o is None:
        return np.zeros(3), np.eye(3)
    xyz = np.array([float(v) for v in o.get("xyz", "0 0 0").split()])
    rpy = [float(v) for v in o.get("rpy", "0 0 0").split()]
    return xyz, _rpy_to_R(rpy)


def _solids(urdf_path: Path):
    """-> list of (volume, com_world, I_about_own_com_world_axes/density)."""
    root = ET.parse(urdf_path).getroot()
    out = []
    for link in root.iter("link"):
        for col in link.findall("collision"):
            xyz, R = _origin(col)
            geom = col.find("geometry")
            box, mesh = geom.find("box"), geom.find("mesh")
            if box is not None:
                sx, sy, sz = (float(v) for v in box.get("size").split())
                V = sx * sy * sz
                # unit-density inertia about the box centre, local axes
                Il = np.diag([(sy**2 + sz**2), (sx**2 + sz**2), (sx**2 + sy**2)]) * V / 12.0
                out.append((V, xyz, R @ Il @ R.T))
            elif mesh is not None:
                import trimesh
                mp = (urdf_path.parent / mesh.get("filename")).resolve()
                m = trimesh.load(mp, force="mesh")
                scale = mesh.get("scale")
                if scale:
                    m.apply_scale([float(v) for v in scale.split()])
                if not m.is_watertight:
                    m.fill_holes()
                V = float(abs(m.volume))
                # trimesh moment_inertia is unit-density about the mesh COM
                Il = np.asarray(m.moment_inertia, dtype=float)
                com_local = np.asarray(m.center_mass, dtype=float)
                out.append((V, xyz + R @ com_local, R @ Il @ R.T))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("urdf")
    ap.add_argument("--density", type=float, default=1240.0,
                    help="kg/m^3; default solid PLA")
    ap.add_argument("--mass", type=float, default=None,
                    help="target mass in kg; overrides --density by back-solving "
                         "the density that hits it (mass and inertia scale together, "
                         "so the COM and the tensor's shape are unchanged)")
    ap.add_argument("--com-z", type=float, default=None,
                    help="override the COM height (m), keeping mass and the "
                         "inertia tensor. SYNTHETIC: the collision geometry is "
                         "unchanged, so this models ballasting the base rather "
                         "than a real solid. Probe only.")
    ap.add_argument("--write", action="store_true",
                    help="insert/replace the <inertial> block in-place")
    args = ap.parse_args()

    path = Path(args.urdf)
    solids = _solids(path)
    if not solids:
        raise SystemExit(f"no collision geometry found in {path}")

    V = sum(s[0] for s in solids)
    com = sum(s[0] * s[1] for s in solids) / V

    density = args.density
    if args.mass is not None:
        if args.mass <= 0:
            raise SystemExit("--mass must be positive")
        density = args.mass / V

    M = density * V

    I = np.zeros((3, 3))
    for Vi, ci, Ii in solids:
        d = ci - com
        I += density * (Ii + Vi * (float(d @ d) * np.eye(3) - np.outer(d, d)))

    geom_com_z = float(com[2])
    if args.com_z is not None:
        # Tipping completes only once the body rotates past atan(d / z_com), so
        # COM height sets how recoverable a momentary tip is. Hold the tensor:
        # a real ballasted base would differ slightly, but this isolates the
        # single quantity under test.
        com = com.copy()
        com[2] = args.com_z

    print(f"{path.name}")
    print(f"  solids   : {len(solids)}")
    print(f"  volume   : {V * 1e6:.1f} cm^3")
    print(f"  mass     : {M:.4f} kg  @ {density:.0f} kg/m^3"
          + ("  (back-solved from --mass)" if args.mass is not None else ""))
    if args.com_z is not None:
        print(f"  com z    : {geom_com_z:.6f} -> {args.com_z:.6f}  "
              f"(SYNTHETIC ballast; inertia tensor held)")
    print(f"  com      : {com[0]:.6f} {com[1]:.6f} {com[2]:.6f}")
    print(f"  ixx={I[0,0]:.6e} iyy={I[1,1]:.6e} izz={I[2,2]:.6e}")
    print(f"  ixy={I[0,1]:.3e} ixz={I[0,2]:.3e} iyz={I[1,2]:.3e}")

    if not args.write:
        return

    block = (
        f'    <inertial>\n'
        f'      <mass value="{M:.4f}"/>\n'
        f'      <origin xyz="{com[0]:.6f} {com[1]:.6f} {com[2]:.6f}" rpy="0 0 0"/>\n'
        f'      <inertia ixx="{I[0,0]:.6e}" ixy="{I[0,1]:.6e}" ixz="{I[0,2]:.6e}"'
        f' iyy="{I[1,1]:.6e}" iyz="{I[1,2]:.6e}" izz="{I[2,2]:.6e}"/>\n'
        f'    </inertial>\n'
    )
    text = path.read_text()
    if "<inertial>" in text:
        # Replace exactly the <inertial>...</inertial> span and nothing else.
        # An earlier version trimmed back from the split point to the previous
        # newline when the preceding text ended in ">", which ate the preceding
        # element's closing tag and produced an unparseable URDF.
        start = text.index("<inertial>")
        end = text.index("</inertial>", start) + len("</inertial>")
        # Keep the original indentation of the opening tag.
        line_start = text.rfind("\n", 0, start) + 1
        indent = text[line_start:start]
        text = text[:line_start] + block.strip("\n").replace("    <inertial>", indent + "<inertial>", 1) + text[end:]
    else:
        idx = text.rindex("</link>")
        text = text[:idx] + block + text[idx:]
    # Never write a URDF we just broke.
    import xml.etree.ElementTree as _ET
    try:
        _ET.fromstring(text)
    except _ET.ParseError as exc:
        raise SystemExit(f"refusing to write {path}: result is not valid XML ({exc})")
    path.write_text(text)
    print(f"  -> wrote <inertial> into {path}")


if __name__ == "__main__":
    main()
