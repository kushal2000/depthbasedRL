#!/usr/bin/env python3
"""Viser viewer for inspecting generated peg-in-hole scenes.

Loads ``assets/urdf/peg_in_hole/scenes/scenes.npz`` and lets you:
  - scroll through the N hole-position scenes via a dropdown,
  - pick which of the M peg starts to animate,
  - pick which of the K tolerance slot URDFs to render (changes the slot
    geometry; the outer footprint is constant across tolerances).

All M peg starts are drawn faintly so you can see the init diversity at a
glance; the selected peg is drawn opaque. Play animates the selected peg
from its start pose along its cached trajectory to the insertion pose.

Usage:
    python peg_in_hole/scene_generation/visualize_scenes.py
    python peg_in_hole/scene_generation/visualize_scenes.py --port 8086
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import trimesh
import viser

from fabrica.viser_utils import SceneManager

from peg_in_hole.scene_generation.generate_scenes import (
    HANDLE_CENTER,
    HANDLE_EXTENTS,
    HEAD_CENTER,
    HEAD_EXTENTS,
    HOLE_COLOR,
    HOLE_SCENE_Z,
    TABLE_SIZE,
    TABLE_Z,
    hole_boxes,
    peg_canonical_mesh,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCENES_DIR = REPO_ROOT / "assets" / "urdf" / "peg_in_hole" / "scenes"

PEG_COLOR = (204, 40, 40)
PEG_GHOST_ALPHA = 0.2


# --- Pose helpers ---

def _qxyzw_to_wxyz(q):
    return np.array([q[3], q[0], q[1], q[2]])


def _slerp(q0, q1, t):
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        r = q0 + t * (q1 - q0)
        return r / np.linalg.norm(r)
    theta = np.arccos(min(dot, 1.0))
    return (np.sin((1 - t) * theta) * q0 + np.sin(t * theta) * q1) / np.sin(theta)


def _interp_waypoints(wps, t):
    """Smoothstep-paced interp across (pos, wxyz_quat) waypoints for t in [0,1]."""
    n = len(wps) - 1
    if n <= 0 or t <= 0:
        return wps[0]
    if t >= 1:
        return wps[-1]
    sf = t * n
    si = min(int(sf), n - 1)
    lt = sf - si
    a = lt * lt * (3.0 - 2.0 * lt)
    pos = (1 - a) * wps[si][0] + a * wps[si + 1][0]
    quat = _slerp(wps[si][1], wps[si + 1][1], a)
    return pos, quat


# --- Asset loaders ---

def load_scenes():
    path = SCENES_DIR / "scenes.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run generate_scenes.py first."
        )
    d = np.load(path)
    return {
        "start_poses": d["start_poses"],                 # (N, M, 7) xyzw
        "goals": d["goals"],                             # (N, M, T, 7) xyzw
        "traj_lengths": d["traj_lengths"],               # (N, M) int32
        "hole_positions": d["hole_positions"],           # (N, 3) world
        "tolerance_pool_m": d["tolerance_pool_m"],       # (pool_size,)
        "scene_tolerance_indices": d["scene_tolerance_indices"],  # (N, K)
    }


def build_hole_mesh_for_scene(hole_x, hole_y, tol_m):
    """Build the table-local hole block mesh (outer + inner slot walls) for a
    specific tolerance. Coordinates are in the table's scene-origin frame
    (i.e., will be placed at (0, 0, TABLE_Z) in world)."""
    meshes = []
    for (cx, cy, cz), ext in hole_boxes(tol_m):
        m = trimesh.creation.box(extents=np.asarray(ext, dtype=float))
        m.apply_translation((cx + hole_x, cy + hole_y, cz + HOLE_SCENE_Z))
        meshes.append(m)
    return trimesh.util.concatenate(meshes)


def waypoints_from_scene(scenes, scene_idx, peg_idx):
    """Build the (pos, wxyz) waypoint list for one peg in one scene: start + goals[:traj_len]."""
    wps = []
    sp = scenes["start_poses"][scene_idx, peg_idx]
    wps.append((np.array(sp[:3], dtype=np.float32), _qxyzw_to_wxyz(sp[3:7])))
    tl = int(scenes["traj_lengths"][scene_idx, peg_idx])
    for k in range(tl):
        gp = scenes["goals"][scene_idx, peg_idx, k]
        wps.append((np.array(gp[:3], dtype=np.float32), _qxyzw_to_wxyz(gp[3:7])))
    return wps


# --- Main viewer ---

def main():
    parser = argparse.ArgumentParser(
        description="Viser viewer for cached peg-in-hole scenes."
    )
    parser.add_argument("--port", type=int, default=8082)
    args = parser.parse_args()

    scenes = load_scenes()
    num_scenes, num_pegs = scenes["start_poses"].shape[:2]
    num_tols = scenes["scene_tolerance_indices"].shape[1]
    # Max waypoints = 1 (start) + max goals per peg.
    max_waypoints = 1 + int(scenes["goals"].shape[2])
    tol_pool = scenes["tolerance_pool_m"]
    print(
        f"Loaded {num_scenes} scenes × {num_pegs} pegs × "
        f"{num_tols} tolerance slots "
        f"(pool size {len(tol_pool)})."
    )

    peg_mesh = peg_canonical_mesh()
    peg_verts = np.array(peg_mesh.vertices, dtype=np.float32)
    peg_faces = np.array(peg_mesh.faces, dtype=np.uint32)

    # --- Viser setup ---

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    scene_mgr = SceneManager()

    @server.on_client_connect
    def _(client: viser.ClientHandle):
        client.camera.position = (0.0, -1.2, 1.1)
        client.camera.look_at = (0.0, 0.0, 0.55)

    # Static world: ground grid, table, robot marker.
    server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)
    server.scene.add_frame(
        "/table", position=(0.0, 0.0, TABLE_Z), wxyz=(1.0, 0.0, 0.0, 0.0), show_axes=False
    )
    server.scene.add_box(
        "/table/wood",
        color=(180, 130, 70),
        dimensions=TABLE_SIZE,
        position=(0.0, 0.0, 0.0),
        side="double",
        opacity=0.6,
    )
    server.scene.add_frame(
        "/robot", position=(0.0, 0.8, 0.0), wxyz=(1.0, 0.0, 0.0, 0.0), show_axes=True,
        axes_length=0.1, axes_radius=0.003,
    )
    robot_urdf = (
        REPO_ROOT
        / "assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf"
    )
    if robot_urdf.exists():
        try:
            from viser.extras import ViserUrdf
            ViserUrdf(server, robot_urdf, root_node_name="/robot")
        except Exception as e:
            print(f"(robot URDF couldn't be loaded: {e})")

    # State
    state = {
        "scene_idx": 0,
        "peg_idx": 0,
        "tol_slot_idx": 0,
        "animating": False,
    }
    active_peg_frame = {"handle": None}

    # --- Rebuild scene (hole + ghost pegs + active peg) ---

    def show_scene():
        scene_mgr.clear()

        s = state["scene_idx"]
        p = state["peg_idx"]
        ts = state["tol_slot_idx"]

        tol_idx = int(scenes["scene_tolerance_indices"][s, ts])
        tol_m = float(tol_pool[tol_idx])
        hole_x, hole_y, _ = scenes["hole_positions"][s]

        # Hole mesh (on the table frame so it lifts by TABLE_Z)
        hole_mesh = build_hole_mesh_for_scene(hole_x, hole_y, tol_m)
        scene_mgr.add(server.scene.add_mesh_simple(
            "/table/hole",
            vertices=np.array(hole_mesh.vertices, dtype=np.float32),
            faces=np.array(hole_mesh.faces, dtype=np.uint32),
            color=HOLE_COLOR,
        ))

        # Ghost pegs (all M starts, faint) — skip the active one.
        for m in range(num_pegs):
            if m == p:
                continue
            sp = scenes["start_poses"][s, m]
            scene_mgr.add(server.scene.add_mesh_simple(
                f"/scene/peg_ghost/{m}",
                vertices=peg_verts,
                faces=peg_faces,
                color=PEG_COLOR,
                position=tuple(float(v) for v in sp[:3]),
                wxyz=tuple(float(v) for v in _qxyzw_to_wxyz(sp[3:7])),
                opacity=PEG_GHOST_ALPHA,
            ))

        # Active peg (bright, gets animated).
        sp = scenes["start_poses"][s, p]
        h = server.scene.add_mesh_simple(
            "/scene/peg_active",
            vertices=peg_verts,
            faces=peg_faces,
            color=PEG_COLOR,
            position=tuple(float(v) for v in sp[:3]),
            wxyz=tuple(float(v) for v in _qxyzw_to_wxyz(sp[3:7])),
            opacity=1.0,
        )
        scene_mgr.add(h)
        active_peg_frame["handle"] = h

        # Trajectory polyline for the active peg (start + goals).
        wps = waypoints_from_scene(scenes, s, p)
        points = np.array([wp[0] for wp in wps], dtype=np.float32)
        scene_mgr.add(server.scene.add_spline_catmull_rom(
            "/scene/traj",
            positions=points,
            color=(0, 180, 255),
            line_width=2.0,
            tension=0.0,
        ))

        _update_status()

    def _set_active_pose(pos, wxyz):
        h = active_peg_frame["handle"]
        if h is None:
            return
        h.position = tuple(float(v) for v in pos)
        h.wxyz = tuple(float(v) for v in wxyz)

    def _update_status():
        s = state["scene_idx"]; p = state["peg_idx"]; ts = state["tol_slot_idx"]
        tol_idx = int(scenes["scene_tolerance_indices"][s, ts])
        tol_mm = float(tol_pool[tol_idx]) * 1000
        tl = int(scenes["traj_lengths"][s, p])
        hx, hy, _ = scenes["hole_positions"][s]
        sp = scenes["start_poses"][s, p]
        g_idx = 0
        try:
            g_idx = int(goal_slider.value)
        except NameError:
            pass  # slider not built yet on first call
        # Clamp for display only
        g_idx_disp = max(0, min(g_idx, tl))  # 0..tl waypoints (start + tl goals)
        status_panel.content = (
            f"## Scene {s} / {num_scenes - 1}\n\n"
            f"- Hole XY: ({hx:+.3f}, {hy:+.3f})\n"
            f"- Active peg {p}/{num_pegs - 1}: "
            f"start ({sp[0]:+.3f}, {sp[1]:+.3f}, {sp[2]:+.3f})\n"
            f"- Tolerance slot {ts}/{num_tols - 1}: "
            f"{tol_mm:.4f} mm (pool idx {tol_idx})\n"
            f"- Trajectory: waypoint **{g_idx_disp}** / {tl} "
            f"(0 = start, 1..{tl} = goals)\n"
        )

    # --- Animation ---

    def play():
        if state["animating"]:
            return
        state["animating"] = True
        wps = waypoints_from_scene(scenes, state["scene_idx"], state["peg_idx"])
        nf = max(1, int(dur_slider.value * 30))
        for fi in range(nf + 1):
            pos, q = _interp_waypoints(wps, fi / nf)
            _set_active_pose(pos, q)
            time.sleep(1.0 / 30)
        state["animating"] = False

    def reset():
        if state["animating"]:
            return
        sp = scenes["start_poses"][state["scene_idx"], state["peg_idx"]]
        _set_active_pose(sp[:3], _qxyzw_to_wxyz(sp[3:7]))

    # --- GUI ---

    scene_dropdown = server.gui.add_dropdown(
        "Scene (hole)",
        options=[str(i) for i in range(num_scenes)],
        initial_value="0",
    )
    peg_dropdown = server.gui.add_dropdown(
        "Peg start",
        options=[str(i) for i in range(num_pegs)],
        initial_value="0",
    )
    tol_dropdown = server.gui.add_dropdown(
        "Tolerance slot",
        options=[str(i) for i in range(num_tols)],
        initial_value="0",
    )
    goal_slider = server.gui.add_slider(
        "Goal idx (0 = start)", min=0, max=max_waypoints - 1, step=1,
        initial_value=0,
    )
    dur_slider = server.gui.add_slider(
        "Play duration (s)", min=0.3, max=4.0, step=0.1, initial_value=1.5
    )
    status_panel = server.gui.add_markdown("")
    server.gui.add_button("Play").on_click(lambda _: play())
    server.gui.add_button("Reset").on_click(lambda _: reset())

    def _goto_waypoint(idx):
        wps = waypoints_from_scene(scenes, state["scene_idx"], state["peg_idx"])
        idx = max(0, min(idx, len(wps) - 1))
        pos, q = wps[idx]
        _set_active_pose(pos, q)
        _update_status()

    def _on_scene(_):
        state["scene_idx"] = int(scene_dropdown.value)
        goal_slider.value = 0  # fires _on_goal which rebuilds the pose
        show_scene()

    def _on_peg(_):
        state["peg_idx"] = int(peg_dropdown.value)
        goal_slider.value = 0
        show_scene()

    def _on_tol(_):
        state["tol_slot_idx"] = int(tol_dropdown.value)
        show_scene()

    def _on_goal(_):
        if state["animating"]:
            return
        _goto_waypoint(int(goal_slider.value))

    scene_dropdown.on_update(_on_scene)
    peg_dropdown.on_update(_on_peg)
    tol_dropdown.on_update(_on_tol)
    goal_slider.on_update(_on_goal)

    show_scene()

    print(f"\nOpen http://localhost:{args.port}")
    print("Dropdowns: Scene (hole), Peg start, Tolerance slot. Press Play to animate.")
    print("Ctrl+C to quit.\n")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
