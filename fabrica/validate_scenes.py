#!/usr/bin/env python3
"""Validate scene URDFs and trajectories for all beam assembly parts.

Renders for each part:
- Table (brown box)
- Scene URDF placed parts (grey, from scene_coacd.urdf)
- Active part at start_pose (green, transparent)
- Active part at final goal (red, transparent)
- Trajectory waypoints (numbered spheres)
- Object URDF CoACD hulls at goal pose (wireframe overlay)

Usage:
    python fabrica/validate_scenes.py
    python fabrica/validate_scenes.py --port 8082 --assembly beam
"""

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
import viser

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "assets" / "urdf" / "fabrica"
TABLE_Z = 0.38


def quat_xyzw_to_wxyz(q):
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def rpy_to_quat_wxyz(roll, pitch, yaw):
    """Convert RPY (in radians) to wxyz quaternion."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z], dtype=np.float64)


def quat_multiply(q1, q2):
    """Multiply two wxyz quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float64)


def quat_rotate_point(q, p):
    """Rotate point p by quaternion q (wxyz format)."""
    p_quat = np.array([0, p[0], p[1], p[2]], dtype=np.float64)
    q_conj = np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)
    result = quat_multiply(quat_multiply(q, p_quat), q_conj)
    return result[1:4]


def load_trajectory(assembly, pid):
    path = REPO_ROOT / "fabrica" / "trajectories" / f"{assembly}_{pid}" / "pick_place.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_assembly_order(assembly):
    path = ASSETS_DIR / assembly / "assembly_order.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())["steps"]


def parse_scene_urdf(urdf_path):
    """Parse a scene URDF and return list of (mesh_path, xyz, rpy) for each collision link."""
    if not urdf_path.exists():
        return []

    tree = ET.parse(urdf_path)
    root = tree.getroot()
    urdf_dir = urdf_path.parent

    # Build joint origins: child_link -> (xyz, rpy)
    joint_origins = {}
    for joint in root.findall("joint"):
        child = joint.find("child").get("link")
        origin = joint.find("origin")
        xyz = [float(v) for v in origin.get("xyz", "0 0 0").split()]
        rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()]
        joint_origins[child] = (xyz, rpy)

    parts = []
    for link in root.findall("link"):
        name = link.get("name")
        if name == "box":
            continue

        visual = link.find("visual")
        if visual is None:
            continue
        geom = visual.find("geometry")
        if geom is None:
            continue
        mesh_elem = geom.find("mesh")
        if mesh_elem is None:
            continue

        mesh_rel = mesh_elem.get("filename")
        mesh_path = (urdf_dir / mesh_rel).resolve()

        xyz, rpy = joint_origins.get(name, ([0, 0, 0], [0, 0, 0]))
        parts.append((mesh_path, xyz, rpy, name))

    return parts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--assembly", default="beam")
    args = parser.parse_args()

    assembly = args.assembly
    order = load_assembly_order(assembly)

    # Filter to parts with trajectories
    available = []
    for pid in order:
        traj = load_trajectory(assembly, pid)
        if traj is not None:
            available.append(pid)

    print(f"Assembly: {assembly}")
    print(f"Order: {available}")

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    scene_handles = []

    @server.on_client_connect
    def _(client: viser.ClientHandle):
        client.camera.position = (0.3, -0.5, 0.9)
        client.camera.look_at = (0.0, 0.0, 0.55)

    # Static: ground grid
    server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)

    def clear_scene():
        for h in scene_handles:
            h.remove()
        scene_handles.clear()

    def show_part(pid):
        clear_scene()
        step_idx = available.index(pid)
        traj = load_trajectory(assembly, pid)

        # --- Table ---
        h = server.scene.add_box(
            "/validate/table",
            color=(180, 130, 70),
            dimensions=(0.475, 0.4, 0.3),
            position=(0, 0, TABLE_Z),
            side="double",
            opacity=0.7,
        )
        scene_handles.append(h)

        # --- Scene URDF parts (previously placed, grey) ---
        env_dir = ASSETS_DIR / "environments" / f"{assembly}_{pid}"
        scene_urdf = env_dir / "scene_coacd.urdf"
        scene_parts = parse_scene_urdf(scene_urdf)

        for mesh_path, xyz, rpy, link_name in scene_parts:
            if not mesh_path.exists():
                print(f"  WARNING: Missing mesh {mesh_path}")
                continue
            mesh = trimesh.load_mesh(str(mesh_path), process=False)

            # Position: joint origin is relative to table center
            world_pos = (xyz[0], xyz[1], xyz[2] + TABLE_Z)
            world_quat = rpy_to_quat_wxyz(rpy[0], rpy[1], rpy[2])

            h = server.scene.add_mesh_simple(
                f"/validate/scene/{link_name}",
                vertices=np.array(mesh.vertices, dtype=np.float32),
                faces=np.array(mesh.faces, dtype=np.uint32),
                color=(150, 150, 150),
                opacity=0.6,
                position=world_pos,
                wxyz=tuple(world_quat),
            )
            scene_handles.append(h)

        # --- Active part: canonical mesh ---
        canonical_path = ASSETS_DIR / assembly / pid / f"{pid}_canonical.obj"
        if not canonical_path.exists():
            print(f"  WARNING: No canonical mesh for part {pid}")
            return
        canonical_mesh = trimesh.load_mesh(str(canonical_path), process=False)
        verts = np.array(canonical_mesh.vertices, dtype=np.float32)
        faces = np.array(canonical_mesh.faces, dtype=np.uint32)

        # Start pose (green)
        sp = traj["start_pose"]
        start_pos = tuple(sp[:3])
        start_quat = tuple(quat_xyzw_to_wxyz(sp[3:7]))
        h = server.scene.add_mesh_simple(
            "/validate/active/start",
            vertices=verts, faces=faces,
            color=(50, 200, 50),
            opacity=0.5,
            position=start_pos,
            wxyz=start_quat,
        )
        scene_handles.append(h)
        h = server.scene.add_label(
            "/validate/active/start_label",
            text="START",
            position=(sp[0], sp[1], sp[2] + 0.02),
        )
        scene_handles.append(h)

        # Final goal (red)
        gp = traj["goals"][-1]
        goal_pos = tuple(gp[:3])
        goal_quat = tuple(quat_xyzw_to_wxyz(gp[3:7]))
        h = server.scene.add_mesh_simple(
            "/validate/active/goal",
            vertices=verts, faces=faces,
            color=(200, 50, 50),
            opacity=0.5,
            position=goal_pos,
            wxyz=goal_quat,
        )
        scene_handles.append(h)
        h = server.scene.add_label(
            "/validate/active/goal_label",
            text="GOAL",
            position=(gp[0], gp[1], gp[2] + 0.02),
        )
        scene_handles.append(h)

        # --- Waypoints (small spheres) ---
        all_waypoints = [traj["start_pose"]] + traj["goals"]
        for wi, wp in enumerate(all_waypoints):
            color = (50, 200, 50) if wi == 0 else (200, 50, 50) if wi == len(all_waypoints) - 1 else (50, 100, 200)
            h = server.scene.add_icosphere(
                f"/validate/waypoints/wp_{wi}",
                radius=0.003,
                color=color,
                position=(wp[0], wp[1], wp[2]),
            )
            scene_handles.append(h)
            h = server.scene.add_label(
                f"/validate/waypoints/wp_{wi}_label",
                text=str(wi),
                position=(wp[0], wp[1], wp[2] + 0.01),
            )
            scene_handles.append(h)

        # --- Trajectory line ---
        points = np.array([[wp[0], wp[1], wp[2]] for wp in all_waypoints], dtype=np.float32)
        if len(points) > 1:
            # Shape (N, 2, 3) for N line segments
            line_points = np.stack([points[:-1], points[1:]], axis=1)
            h = server.scene.add_line_segments(
                "/validate/trajectory_line",
                points=line_points,
                colors=(100, 100, 100),
                line_width=2.0,
            )
            scene_handles.append(h)

        # --- Info text ---
        completed = available[:step_idx]
        info = (
            f"**Part {pid}** (step {step_idx}/{len(available)})\n\n"
            f"Completed parts in scene: {completed if completed else 'none'}\n\n"
            f"Start: [{sp[0]:.4f}, {sp[1]:.4f}, {sp[2]:.4f}]\n\n"
            f"Goal:  [{gp[0]:.4f}, {gp[1]:.4f}, {gp[2]:.4f}]\n\n"
            f"Goal quat (xyzw): [{gp[3]:.4f}, {gp[4]:.4f}, {gp[5]:.4f}, {gp[6]:.4f}]\n\n"
            f"Waypoints: {len(traj['goals'])}\n\n"
            f"Scene URDF: {scene_urdf.name} ({len(scene_parts)} collision links)"
        )
        info_text.content = info

    # GUI
    part_options = [f"Part {pid} (step {i})" for i, pid in enumerate(available)]
    dropdown = server.gui.add_dropdown(
        "Part", options=part_options, initial_value=part_options[0]
    )
    info_text = server.gui.add_markdown("Loading...")

    def on_select(_):
        idx = part_options.index(dropdown.value)
        pid = available[idx]
        show_part(pid)

    dropdown.on_update(on_select)
    show_part(available[0])

    print(f"\nOpen http://localhost:{args.port}")
    print("Select parts from dropdown to validate scene + trajectory.")
    try:
        while True:
            import time
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
