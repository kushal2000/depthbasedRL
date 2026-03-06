#!/usr/bin/env python3
"""Animate Fabrica assembly parts one at a time in assembly order.

Parts start spread on the left side of the table and move sequentially
to their assembled position on the right side, in an eval.py-style scene
with robot, table, and fixture.

Usage:
    python fabrica/animate_sequential.py
    python fabrica/animate_sequential.py --port 8082
"""

import argparse
import time

import numpy as np
import trimesh
import viser
from viser.extras import ViserUrdf

from fabrica.viser_utils import (
    ASSETS_DIR,
    COLORS,
    SceneManager,
    load_all_assemblies,
    load_assembly_order,
)

REPO_ROOT = ASSETS_DIR.parent.parent
TABLE_Z = 0.38
Z_OFFSET = 0.03
TABLE_SURFACE_Z = TABLE_Z + 0.15

def _axis_angle_to_quat(axis_x, axis_y, axis_z, angle_deg):
    """Convert axis-angle (degrees) to wxyz quaternion."""
    angle = np.radians(angle_deg) / 2.0
    s = np.sin(angle)
    norm = np.sqrt(axis_x**2 + axis_y**2 + axis_z**2)
    ax, ay, az = axis_x / norm, axis_y / norm, axis_z / norm
    return np.array([np.cos(angle), ax * s, ay * s, az * s])


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


IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])


def ease_in_out(t):
    return t * t * (3.0 - 2.0 * t)


def _rotated_extents(mesh, quat):
    """Get axis-aligned extents of a mesh after applying a rotation quaternion (wxyz)."""
    if np.allclose(quat, IDENTITY_QUAT):
        return mesh.bounding_box.extents
    w, x, y, z = quat
    # Rotation matrix from quaternion
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])
    verts = np.array(mesh.vertices) - mesh.centroid
    rotated = verts @ R.T
    return rotated.max(axis=0) - rotated.min(axis=0)


def compute_start_positions(parts, assembly_order, start_rotations, assembled_y):
    """Place parts in a grid (3 per row), first-assembled closest to robot.

    Row 0 is aligned to assembled_y so it shares the same y as the final assembly.
    Uses bounding box extents (accounting for start rotation) for spacing.
    start_rotations: dict of part_id -> [axis_x, axis_y, axis_z, angle_deg] from JSON.
    """
    cols = 3
    parts_dict = {pid: mesh for pid, mesh in parts}
    gap = 0.015

    # Precompute start quaternions and rotated extents
    start_quats = {}
    rotated_ext = {}
    for pid in assembly_order:
        if pid in start_rotations:
            start_quats[pid] = _axis_angle_to_quat(*start_rotations[pid])
        else:
            start_quats[pid] = IDENTITY_QUAT.copy()
        rotated_ext[pid] = _rotated_extents(parts_dict[pid], start_quats[pid])

    # Group ordered IDs into rows of 3
    ordered_ids = list(assembly_order)
    rows = [ordered_ids[i : i + cols] for i in range(0, len(ordered_ids), cols)]

    # Compute column widths (max x extent in each column across all rows)
    col_widths = [0.0] * cols
    for row in rows:
        for c, pid in enumerate(row):
            col_widths[c] = max(col_widths[c], rotated_ext[pid][0])

    # Compute row depths (max y extent in each row)
    row_depths = []
    for row in rows:
        row_depths.append(max(rotated_ext[pid][1] for pid in row))

    # Column x centers
    col_x = []
    cursor_x = 0.02
    for c in range(cols):
        cursor_x += col_widths[c] / 2.0
        col_x.append(cursor_x)
        cursor_x += col_widths[c] / 2.0 + gap

    # Row y centers: row 0 centered at assembled_y, later rows go further away (decreasing y)
    row_y = []
    row_y.append(assembled_y)
    cursor_y = assembled_y - row_depths[0] / 2.0
    for r in range(1, len(rows)):
        cursor_y -= gap + row_depths[r] / 2.0
        row_y.append(cursor_y)
        cursor_y -= row_depths[r] / 2.0

    positions = {}
    for r, row in enumerate(rows):
        for c, pid in enumerate(row):
            ext_z = rotated_ext[pid][2]
            z = TABLE_SURFACE_Z + ext_z / 2.0
            positions[pid] = np.array([col_x[c], row_y[r], z])
    return positions, start_quats


def compute_assembled_positions(parts):
    """Compute assembled positions on the right (negative x) side of the table.

    OBJ meshes are already in the assembled reference frame. We offset the
    whole assembly so its centroid sits above the table on the negative-x side.
    Returns (positions_dict, assembly_centroid_y) so start grid can align.
    """
    centroids = [mesh.centroid for _, mesh in parts]
    overall_centroid = np.mean(centroids, axis=0)

    # Place assembly centroid at this target location
    target = np.array([-0.08, 0.04, TABLE_SURFACE_Z + 0.05])
    offset = target - overall_centroid

    positions = {}
    for part_id, mesh in parts:
        positions[part_id] = mesh.centroid + offset
    return positions, target[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--assets-dir", default=ASSETS_DIR)
    args = parser.parse_args()

    print("Loading meshes...")
    assembly_parts = load_all_assemblies(args.assets_dir)

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    mesh_scene = SceneManager()

    # Scene setup (mirrors eval.py)
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
    server.scene.add_frame(
        "/table/fixture",
        position=(0.12, -0.152, 0.15),
        wxyz=(1, 0, 0, 0),
        show_axes=False,
    )
    fixture_urdf = REPO_ROOT / "assets" / "urdf" / "fabrica" / "beam" / "fixture" / "fixture.urdf"
    if fixture_urdf.exists():
        ViserUrdf(server, fixture_urdf, root_node_name="/table/fixture")

    # Animation state
    part_frames = {}  # part_id -> frame handle
    start_positions = {}
    assembled_positions = {}
    start_quats = {}
    assembly_order = []
    current_step = 0
    is_animating = False

    def show_assembly(name):
        nonlocal part_frames, start_positions, assembled_positions, start_quats, assembly_order, current_step
        mesh_scene.clear()
        part_frames = {}
        parts = assembly_parts[name]

        assembly_order, start_rotations = load_assembly_order(args.assets_dir, name, parts)
        assembled_positions, assembled_y = compute_assembled_positions(parts)
        start_positions, start_quats = compute_start_positions(parts, assembly_order, start_rotations, assembled_y)
        current_step = 0

        # Build a lookup for color by index
        part_id_to_idx = {pid: i for i, (pid, _) in enumerate(parts)}

        for part_id, mesh in parts:
            frame = server.scene.add_frame(
                f"/assembly/frame_{part_id}",
                wxyz=tuple(start_quats[part_id]),
                position=tuple(start_positions[part_id]),
                show_axes=False,
            )
            # Draw explicit XYZ axes as colored line segments
            ax_len = 0.05
            for axis_name, endpoint, color in [
                ("x", (ax_len, 0, 0), (255, 0, 0)),
                ("y", (0, ax_len, 0), (0, 255, 0)),
                ("z", (0, 0, ax_len), (0, 0, 255)),
            ]:
                seg = server.scene.add_line_segments(
                    f"/assembly/frame_{part_id}/axis_{axis_name}",
                    points=np.array([[0, 0, 0], endpoint], dtype=np.float32),
                    colors=np.array([color, color], dtype=np.uint8),
                    line_width=3.0,
                )
                mesh_scene.add(seg)
            mesh_scene.add(frame)
            part_frames[part_id] = frame

            # Render mesh relative to frame: vertices offset by -centroid
            # so the frame position controls where the centroid goes
            verts = np.array(mesh.vertices, dtype=np.float32) - mesh.centroid.astype(
                np.float32
            )
            idx = part_id_to_idx[part_id]
            h = server.scene.add_mesh_simple(
                f"/assembly/frame_{part_id}/mesh",
                vertices=verts,
                faces=np.array(mesh.faces, dtype=np.uint32),
                color=COLORS[idx % len(COLORS)],
            )
            mesh_scene.add(h)

            # Label above the part
            step_idx = assembly_order.index(part_id)
            label_z = mesh.bounding_box.extents[2] / 2.0 + 0.02
            label = server.scene.add_label(
                f"/assembly/frame_{part_id}/label",
                text=f"#{step_idx} (id:{part_id})",
                position=(0, 0, label_z),
            )
            mesh_scene.add(label)

        apply_positions(current_step, 0.0)
        update_status()

    def apply_positions(step, t):
        """Set frame positions and rotations based on current step and interpolation t."""
        for i, pid in enumerate(assembly_order):
            if pid not in part_frames:
                continue
            if i < step:
                pos = assembled_positions[pid]
                quat = IDENTITY_QUAT
            elif i == step and t > 0:
                alpha = ease_in_out(t)
                pos = (1 - alpha) * start_positions[pid] + alpha * assembled_positions[pid]
                quat = _slerp(start_quats[pid], IDENTITY_QUAT, alpha)
            else:
                pos = start_positions[pid]
                quat = start_quats[pid]
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
    assembly_names = list(assembly_parts.keys())
    dropdown = server.gui.add_dropdown(
        "Assembly", options=assembly_names, initial_value=assembly_names[0]
    )
    duration_slider = server.gui.add_slider(
        "Step Duration (s)", min=0.2, max=3.0, step=0.1, initial_value=1.0
    )
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

    show_assembly(dropdown.value)

    print(f"\nOpen http://localhost:{args.port}")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
