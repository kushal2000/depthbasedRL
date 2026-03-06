#!/usr/bin/env python3
"""Animate Fabrica assembly parts one at a time in assembly order.

Parts start at their trajectory start_pose and move sequentially
to their assembled goal position (from pick_place.json trajectories),
matching the eval_assembly.py scene.

Usage:
    python fabrica/animate_sequential.py
    python fabrica/animate_sequential.py --port 8082
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import trimesh
import viser
from viser.extras import ViserUrdf

from fabrica.viser_utils import (
    ASSETS_DIR,
    COLORS,
    SceneManager,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TABLE_Z = 0.38
Z_OFFSET = 0.03
TABLE_SURFACE_Z = TABLE_Z + 0.15

ALL_ASSEMBLIES = [
    "beam", "car", "cooling_manifold", "duct",
    "gamepad", "plumbers_block", "stool_circular",
]


def _quat_xyzw_to_wxyz(q):
    return np.array([q[3], q[0], q[1], q[2]])


def _slerp(q0, q1, t):
    """Spherical linear interpolation between two wxyz quaternions."""
    dot = np.dot(q0, q1)
    if dot < 0:
        q1 = -q1
        dot = -dot
    dot = min(dot, 1.0)
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    return (np.sin((1 - t) * theta) / sin_theta) * q0 + (np.sin(t * theta) / sin_theta) * q1


def ease_in_out(t):
    return t * t * (3.0 - 2.0 * t)


def _load_trajectory(assembly, pid):
    path = REPO_ROOT / "fabrica" / "trajectories" / f"{assembly}_{pid}" / "pick_place.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _load_assembly_order(assembly):
    path = ASSETS_DIR / assembly / "assembly_order.json"
    if not path.exists():
        return [], {}
    data = json.loads(path.read_text())
    return data.get("steps", []), data.get("start_rotations", {})


def _get_available_parts(assembly):
    """Match eval_assembly.py by only using parts with trajectory and env URDF."""
    order, _ = _load_assembly_order(assembly)
    available = []
    for pid in order:
        name = f"{assembly}_{pid}"
        traj = REPO_ROOT / "fabrica" / "trajectories" / name / "pick_place.json"
        urdf = ASSETS_DIR / "environments" / name / "pick_place.urdf"
        if traj.exists() and urdf.exists():
            available.append(pid)
    return available


def _load_canonical_mesh(assembly, pid):
    path = ASSETS_DIR / assembly / pid / f"{pid}_canonical.obj"
    if not path.exists():
        return None
    return trimesh.load_mesh(str(path), process=False)


def _load_assembly_with_trajectories(assembly):
    """Load parts using the same availability rules as eval_assembly.py.

    Returns list of (part_id, mesh, mesh_offset, traj) in eval order for parts that have:
    - a trajectory JSON
    - an environment URDF
    - a canonical mesh for visualization
    """
    available = _get_available_parts(assembly)
    _, start_rotations = _load_assembly_order(assembly)

    parts = []
    for pid in available:
        mesh = _load_canonical_mesh(assembly, pid)
        traj = _load_trajectory(assembly, pid)
        if mesh is not None and traj is not None:
            mesh_offset = -np.array(mesh.centroid, dtype=np.float32)
            parts.append((pid, mesh, mesh_offset, traj))
    return parts, start_rotations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--assets-dir", default=ASSETS_DIR)
    args = parser.parse_args()

    print("Loading assemblies...")
    # Pre-load all assemblies
    all_assembly_data = {}
    for name in ALL_ASSEMBLIES:
        parts, start_rots = _load_assembly_with_trajectories(name)
        if parts:
            all_assembly_data[name] = (parts, start_rots)
            print(f"  {name}: {len(parts)} parts")

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    mesh_scene = SceneManager()

    # Scene setup (mirrors eval_assembly.py)
    @server.on_client_connect
    def _(client: viser.ClientHandle):
        client.camera.position = (0.0, -1.0, 1.0)
        client.camera.look_at = (0.0, 0.0, 0.5)

    server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)
    server.scene.add_frame(
        "/robot", position=(0, 0.8, 0), wxyz=(1, 0, 0, 0), show_axes=False
    )
    robot_urdf = (
        REPO_ROOT
        / "assets"
        / "urdf"
        / "kuka_sharpa_description"
        / "iiwa14_left_sharpa_adjusted_restricted.urdf"
    )
    if robot_urdf.exists():
        ViserUrdf(server, robot_urdf, root_node_name="/robot")

    server.scene.add_frame(
        "/table", position=(0, 0, TABLE_Z), wxyz=(1, 0, 0, 0), show_axes=False
    )
    server.scene.add_box(
        "/table/wood",
        color=(180, 130, 70),
        dimensions=(0.475, 0.4, 0.3),
        position=(0, 0, 0),
        side="double",
        opacity=0.9,
    )

    # Animation state
    part_frames = {}  # part_id -> frame handle
    part_labels = {}  # part_id -> label handle
    # Full waypoint paths: pid -> list of (pos, quat_wxyz) including start_pose + all goals
    waypoints = {}  # type: dict
    assembly_order = []
    current_step = 0
    is_animating = False

    def _apply_visibility():
        show_axes = cb_axes.value
        show_labels = cb_labels.value
        for frame in part_frames.values():
            frame.visible = True
            frame.show_axes = show_axes
        for label in part_labels.values():
            label.visible = show_labels

    def show_assembly(name):
        nonlocal part_frames, part_labels, waypoints, assembly_order, current_step
        mesh_scene.clear()
        part_frames = {}
        part_labels = {}

        parts, start_rots = all_assembly_data[name]
        assembly_order = [pid for pid, _, _, _ in parts]
        part_id_to_idx = {pid: i for i, pid in enumerate(assembly_order)}

        # Build full waypoint paths from trajectories
        waypoints = {}
        for pid, _, _, traj in parts:
            wps = []
            # Start pose
            sp = traj["start_pose"]
            wps.append((np.array(sp[:3]), _quat_xyzw_to_wxyz(sp[3:7])))
            # All goal waypoints
            for gp in traj["goals"]:
                wps.append((np.array(gp[:3]), _quat_xyzw_to_wxyz(gp[3:7])))
            waypoints[pid] = wps

        current_step = 0

        for pid, mesh, mesh_offset, traj in parts:
            frame_name = f"/assembly/frame_{pid}"
            idx = part_id_to_idx[pid]
            frame = server.scene.add_frame(
                frame_name,
                wxyz=(1.0, 0.0, 0.0, 0.0),
                position=(0.0, 0.0, 0.0),
                show_axes=True,
                axes_length=0.1,
                axes_radius=0.001,
            )
            mesh_scene.add(frame)
            part_frames[pid] = frame

            mesh_frame = server.scene.add_frame(
                f"{frame_name}/mesh_frame",
                position=tuple(mesh_offset),
                wxyz=(1, 0, 0, 0),
                show_axes=False,
            )
            mesh_scene.add(mesh_frame)

            h = server.scene.add_mesh_simple(
                f"{frame_name}/mesh_frame/mesh",
                vertices=np.array(mesh.vertices, dtype=np.float32),
                faces=np.array(mesh.faces, dtype=np.uint32),
                color=COLORS[idx % len(COLORS)],
            )
            mesh_scene.add(h)

            step_idx = assembly_order.index(pid)
            label = server.scene.add_label(
                f"{frame_name}/label",
                text=f"#{step_idx} (id:{pid})",
                position=(0, 0, mesh.bounding_box.extents[2] / 2.0 + 0.02),
            )
            mesh_scene.add(label)
            part_labels[pid] = label

        apply_positions(current_step, 0.0)
        _apply_visibility()
        update_status()

    def _interpolate_waypoints(wps, t):
        """Interpolate along a waypoint list. t in [0, 1] spans all segments."""
        n_segs = len(wps) - 1
        if n_segs <= 0 or t <= 0:
            return wps[0]
        if t >= 1:
            return wps[-1]
        # Map t to segment index and local t
        seg_f = t * n_segs
        seg_i = min(int(seg_f), n_segs - 1)
        local_t = seg_f - seg_i
        pos0, quat0 = wps[seg_i]
        pos1, quat1 = wps[seg_i + 1]
        alpha = ease_in_out(local_t)
        pos = (1 - alpha) * pos0 + alpha * pos1
        quat = _slerp(quat0, quat1, alpha)
        return pos, quat

    def apply_positions(step, t):
        """Set frame positions/rotations. t in [0,1] interpolates along full waypoint path."""
        for i, pid in enumerate(assembly_order):
            if pid not in part_frames:
                continue
            wps = waypoints[pid]
            if i < step:
                pos, quat = wps[-1]
            elif i == step and t > 0:
                pos, quat = _interpolate_waypoints(wps, t)
            else:
                pos, quat = wps[0]
            part_frames[pid].position = tuple(pos)
            part_frames[pid].wxyz = tuple(quat)

    def update_status():
        n = len(assembly_order)
        if current_step >= n:
            status_text.content = f"Step {n}/{n} - Assembly complete"
        else:
            pid = assembly_order[current_step]
            status_text.content = f"Step {current_step}/{n} - next part_id: {pid}"

    def animate_step_forward():
        nonlocal current_step, is_animating
        if is_animating or current_step >= len(assembly_order):
            return
        is_animating = True
        dur = duration_slider.value
        n_frames = max(1, int(dur * args.fps))
        for fi in range(n_frames + 1):
            t = fi / n_frames
            apply_positions(current_step, t)
            time.sleep(1.0 / args.fps)
        current_step += 1
        apply_positions(current_step, 0.0)
        update_status()
        is_animating = False

    def animate_step_back():
        nonlocal current_step, is_animating
        if is_animating or current_step <= 0:
            return
        is_animating = True
        current_step -= 1
        dur = duration_slider.value
        n_frames = max(1, int(dur * args.fps))
        for fi in range(n_frames + 1):
            t = 1.0 - fi / n_frames
            apply_positions(current_step, t)
            time.sleep(1.0 / args.fps)
        apply_positions(current_step, 0.0)
        update_status()
        is_animating = False

    def play_all():
        nonlocal is_animating
        if is_animating:
            return
        while current_step < len(assembly_order):
            animate_step_forward()
            if current_step < len(assembly_order):
                time.sleep(0.3)

    def reset():
        nonlocal current_step
        if is_animating:
            return
        current_step = 0
        apply_positions(0, 0.0)
        update_status()

    # GUI
    assembly_names = list(all_assembly_data.keys())
    dropdown = server.gui.add_dropdown(
        "Assembly", options=assembly_names, initial_value=assembly_names[0]
    )
    duration_slider = server.gui.add_slider(
        "Step Duration (s)", min=0.2, max=3.0, step=0.1, initial_value=1.0
    )
    cb_axes = server.gui.add_checkbox("Show Axes", initial_value=True)
    cb_labels = server.gui.add_checkbox("Show Part IDs", initial_value=True)
    status_text = server.gui.add_markdown("**Status:** Ready")
    play_btn = server.gui.add_button("Play All")
    fwd_btn = server.gui.add_button("Step Forward")
    back_btn = server.gui.add_button("Step Back")
    reset_btn = server.gui.add_button("Reset")

    dropdown.on_update(lambda _: show_assembly(dropdown.value))
    play_btn.on_click(lambda _: play_all())
    fwd_btn.on_click(lambda _: animate_step_forward())
    back_btn.on_click(lambda _: animate_step_back())
    reset_btn.on_click(lambda _: reset())
    cb_axes.on_update(lambda _: _apply_visibility())
    cb_labels.on_update(lambda _: _apply_visibility())

    show_assembly(dropdown.value)

    print(f"\nOpen http://localhost:{args.port}")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
