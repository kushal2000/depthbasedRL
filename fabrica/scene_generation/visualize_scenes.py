#!/usr/bin/env python3
"""Viser viewer to inspect generated scenes from ``scenes.npz``.

Loads ``assets/urdf/fabrica/{assembly}/scenes.npz`` and lets you scroll
through the cached scenes via a dropdown. For each scene, all 5 parts are
rendered at their sampled start poses on the table; pressing Step Forward
animates the next part in the assembly order from its start pose along its
cached trajectory to the fixture pose. Step Back rewinds; Play All cycles
through every part.

Use this to hand-validate the generated scenes before they're consumed by
training.

Usage:
    python fabrica/scene_generation/visualize_scenes.py --assembly beam
    python fabrica/scene_generation/visualize_scenes.py --assembly beam --port 8086
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import trimesh
import viser
from viser.extras import ViserUrdf

from fabrica.viser_utils import COLORS, SceneManager

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ASSETS_DIR = REPO_ROOT / "assets" / "urdf" / "fabrica"

# Scene constants (must match generate_scenes.py / step3 for visual alignment)
TABLE_Z = 0.38             # world center z of the table box
TABLE_DIM = (0.475, 0.4, 0.3)


def _qxyzw_to_wxyz(q):
    return np.array([q[3], q[0], q[1], q[2]])


def _slerp(q0, q1, t):
    """Quaternion slerp in wxyz convention."""
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        r = q0 + t * (q1 - q0)
        return r / np.linalg.norm(r)
    theta = np.arccos(min(dot, 1.0))
    return (np.sin((1 - t) * theta) * q0 + np.sin(t * theta) * q1) / np.sin(theta)


def _interp_waypoints(wps, t):
    """Linear-with-slerp interpolation across a list of (pos, wxyz) waypoints
    parameterized by t in [0, 1] over the whole sequence."""
    n = len(wps) - 1
    if n <= 0 or t <= 0:
        return wps[0]
    if t >= 1:
        return wps[-1]
    sf = t * n
    si = min(int(sf), n - 1)
    lt = sf - si
    a = lt * lt * (3.0 - 2.0 * lt)  # smoothstep
    pos = (1 - a) * wps[si][0] + a * wps[si + 1][0]
    quat = _slerp(wps[si][1], wps[si + 1][1], a)
    return pos, quat


def load_assembly_meta(assembly):
    with open(ASSETS_DIR / assembly / "assembly_order.json") as f:
        order = json.load(f)
    return order["steps"]


def load_canonical_meshes(assembly, part_ids):
    meshes = {}
    for pid in part_ids:
        path = ASSETS_DIR / assembly / pid / f"{pid}_canonical.obj"
        meshes[pid] = trimesh.load_mesh(str(path), process=False)
    return meshes


def load_scenes(assembly):
    path = ASSETS_DIR / assembly / "scenes.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run generate_scenes.py --assembly {assembly} first."
        )
    data = np.load(path)
    return {
        "start_poses": data["start_poses"],     # [num_scenes, num_parts, 7] xyzw
        "goals": data["goals"],                 # [num_scenes, num_parts, max_len, 7] xyzw
        "traj_lengths": data["traj_lengths"],   # [num_scenes, num_parts]
    }


def main():
    parser = argparse.ArgumentParser(
        description="Viser viewer for cached scenes (start poses + trajectories).",
    )
    parser.add_argument("--assembly", type=str, required=True)
    parser.add_argument("--port", type=int, default=8082)
    args = parser.parse_args()

    part_ids = load_assembly_meta(args.assembly)
    canonical_meshes = load_canonical_meshes(args.assembly, part_ids)
    scenes = load_scenes(args.assembly)

    num_scenes = scenes["start_poses"].shape[0]
    num_parts = scenes["start_poses"].shape[1]
    print(f"Loaded {num_scenes} scenes for {args.assembly} ({num_parts} parts)")

    # Build per-scene per-part waypoint lists ([(pos, wxyz_quat), ...]) that
    # start with the start_pose and continue through the cached goals.
    scene_waypoints = []
    for s in range(num_scenes):
        per_part = {}
        for p_idx, pid in enumerate(part_ids):
            wps = []
            sp = scenes["start_poses"][s, p_idx]
            wps.append((np.array(sp[:3]), _qxyzw_to_wxyz(sp[3:7])))
            traj_len = int(scenes["traj_lengths"][s, p_idx])
            for k in range(traj_len):
                gp = scenes["goals"][s, p_idx, k]
                wps.append((np.array(gp[:3]), _qxyzw_to_wxyz(gp[3:7])))
            per_part[pid] = wps
        scene_waypoints.append(per_part)

    # --- Viser setup ---

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    scene_mgr = SceneManager()

    @server.on_client_connect
    def _(client: viser.ClientHandle):
        client.camera.position = (0.0, -1.0, 1.0)
        client.camera.look_at = (0.0, 0.0, 0.5)

    # Static scene
    server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)
    server.scene.add_frame(
        "/table", position=(0.0, 0.0, TABLE_Z), wxyz=(1.0, 0.0, 0.0, 0.0), show_axes=False
    )
    server.scene.add_box(
        "/table/wood",
        color=(180, 130, 70),
        dimensions=TABLE_DIM,
        position=(0.0, 0.0, 0.0),
        side="double",
        opacity=0.7,
    )
    server.scene.add_frame(
        "/robot", position=(0.0, 0.8, 0.0), wxyz=(1.0, 0.0, 0.0, 0.0), show_axes=False
    )
    robot_urdf = (
        REPO_ROOT
        / "assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf"
    )
    if robot_urdf.exists():
        ViserUrdf(server, robot_urdf, root_node_name="/robot")

    # Dynamic per-part frames + meshes (rebuilt when scene changes)
    frames = {}      # pid → frame handle
    current_scene_idx = [0]
    step = [0]       # how many parts have completed their animation
    animating = [False]

    def _set_pose_for_part(pid, pos, wxyz):
        f = frames[pid]
        f.position = tuple(float(v) for v in pos)
        f.wxyz = tuple(float(v) for v in wxyz)

    def show_scene(scene_idx):
        scene_mgr.clear()
        frames.clear()
        step[0] = 0

        for i, pid in enumerate(part_ids):
            mesh = canonical_meshes[pid]
            offset = -np.array(mesh.centroid, dtype=np.float32)

            f = server.scene.add_frame(
                f"/scene/{pid}",
                wxyz=(1.0, 0.0, 0.0, 0.0),
                position=(0.0, 0.0, 0.0),
                show_axes=True,
                axes_length=0.05,
                axes_radius=0.0008,
            )
            scene_mgr.add(f)
            mf = server.scene.add_frame(
                f"/scene/{pid}/centered",
                position=tuple(float(v) for v in offset),
                wxyz=(1.0, 0.0, 0.0, 0.0),
                show_axes=False,
            )
            scene_mgr.add(mf)
            scene_mgr.add(server.scene.add_mesh_simple(
                f"/scene/{pid}/centered/mesh",
                vertices=np.array(mesh.vertices, dtype=np.float32),
                faces=np.array(mesh.faces, dtype=np.uint32),
                color=COLORS[i % len(COLORS)],
            ))
            frames[pid] = f

        # Initial pose: all parts at their start
        wps_for_scene = scene_waypoints[scene_idx]
        for pid in part_ids:
            sp = wps_for_scene[pid][0]
            _set_pose_for_part(pid, sp[0], sp[1])

        _update_status()
        print(f"Showing scene {scene_idx}.")

    def _update_status():
        n = len(part_ids)
        if step[0] >= n:
            status_panel.content = (
                f"## Scene {current_scene_idx[0]} / {num_scenes - 1}\n\n"
                f"Step {n}/{n} — assembly complete."
            )
        else:
            status_panel.content = (
                f"## Scene {current_scene_idx[0]} / {num_scenes - 1}\n\n"
                f"Step {step[0]}/{n} — next: part {part_ids[step[0]]}\n\n"
                f"Trajectory length: "
                f"{int(scenes['traj_lengths'][current_scene_idx[0], step[0]])}"
            )

    def _set_pose_during_animation(active_idx, t):
        """For parts before `active_idx` show them at their final pose; for the
        active part interpolate; for parts after, show them at the start."""
        wps_for_scene = scene_waypoints[current_scene_idx[0]]
        for i, pid in enumerate(part_ids):
            wps = wps_for_scene[pid]
            if i < active_idx:
                pos, q = wps[-1]
            elif i == active_idx and t > 0:
                pos, q = _interp_waypoints(wps, t)
            else:
                pos, q = wps[0]
            _set_pose_for_part(pid, pos, q)

    def step_fwd():
        if animating[0] or step[0] >= len(part_ids):
            return
        animating[0] = True
        nf = max(1, int(dur_slider.value * 30))
        for fi in range(nf + 1):
            _set_pose_during_animation(step[0], fi / nf)
            time.sleep(1.0 / 30)
        step[0] += 1
        _set_pose_during_animation(step[0], 0.0)
        _update_status()
        animating[0] = False

    def step_back():
        if animating[0] or step[0] <= 0:
            return
        animating[0] = True
        step[0] -= 1
        nf = max(1, int(dur_slider.value * 30))
        for fi in range(nf + 1):
            _set_pose_during_animation(step[0], 1.0 - fi / nf)
            time.sleep(1.0 / 30)
        _set_pose_during_animation(step[0], 0.0)
        _update_status()
        animating[0] = False

    def play_all():
        if animating[0]:
            return
        while step[0] < len(part_ids):
            step_fwd()
            if step[0] < len(part_ids):
                time.sleep(0.3)

    def reset():
        if animating[0]:
            return
        step[0] = 0
        wps_for_scene = scene_waypoints[current_scene_idx[0]]
        for pid in part_ids:
            sp = wps_for_scene[pid][0]
            _set_pose_for_part(pid, sp[0], sp[1])
        _update_status()

    # GUI
    dropdown = server.gui.add_dropdown(
        "Scene", options=[str(i) for i in range(num_scenes)], initial_value="0"
    )
    dur_slider = server.gui.add_slider(
        "Duration (s)", min=0.2, max=3.0, step=0.1, initial_value=1.0
    )
    status_panel = server.gui.add_markdown("")
    server.gui.add_button("Play All").on_click(lambda _: play_all())
    server.gui.add_button("Step Forward").on_click(lambda _: step_fwd())
    server.gui.add_button("Step Back").on_click(lambda _: step_back())
    server.gui.add_button("Reset").on_click(lambda _: reset())

    def _on_scene_change(_):
        current_scene_idx[0] = int(dropdown.value)
        show_scene(current_scene_idx[0])

    dropdown.on_update(_on_scene_change)

    show_scene(0)

    print(f"\nOpen http://localhost:{args.port}")
    print("Use the Scene dropdown to switch between cached scenes; Play All animates each part in assembly order.")
    print("Ctrl+C to quit.\n")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
