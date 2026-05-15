#!/usr/bin/env python
"""Visualize rollout recordings saved by student_depth_policy_node.py.

This is intentionally offline: the policy node records arrays in memory and
writes one compressed NPZ on shutdown, then this script lets us inspect the
actual policy inputs/actions/targets without adding file I/O to the rollout
loop.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import tyro
import viser
from scipy.spatial.transform import Rotation as R
from viser.extras import ViserUrdf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaacgymenvs.utils.utils import get_repo_root_dir

import peg_in_hole.objects  # noqa: F401 - registers peg_L/L_peg
from dextoolbench.objects import NAME_TO_OBJECT

DEFAULT_CAMERA_FRAME = "/student_depth_camera"
DEFAULT_CAMERA_POS_WORLD = (-0.5002050422666431, -0.6385715691360607, 1.0201893282998005)
DEFAULT_CAMERA_QUAT_WXYZ = (-0.5314110448277682, 0.833810802683381, -0.14035163049226862, 0.051606846267884886)
LIGHT_BLUE_RGBA = (80, 200, 255, 0.55)
ORANGE_RGB = (255, 145, 0)
WORLD_T_ROBOT_POS_M = np.asarray([0.0, 0.8, 0.0], dtype=np.float64)


def _policy_depth_to_rgb(policy_depth: np.ndarray, depth_format: str) -> np.ndarray:
    depth = np.asarray(policy_depth)
    if depth_format == "uint8":
        normalized = depth.astype(np.float32) / 255.0
    else:
        normalized = depth.astype(np.float32)
    normalized = np.clip(normalized, 0.0, 1.0)
    gray = (255.0 * (1.0 - normalized)).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=-1)


def _viser_frustum_image(rgb: np.ndarray, *, flip_y: bool) -> np.ndarray:
    image = np.asarray(rgb)
    return np.flipud(image) if flip_y else image


def _npz_get(data: np.lib.npyio.NpzFile, key: str):
    return data[key] if key in data.files else None


@dataclass
class Args:
    recording: Path
    """NPZ file written by --record_rollout_dir."""
    object_name: str = "peg_L"
    """Object mesh for predicted pose visualization."""
    port: int = 8080
    """Viser port."""
    fps: float = 30.0
    """Playback FPS."""
    start_paused: bool = False
    """Start paused instead of playing immediately."""
    camera_pos_world: tuple[float, float, float] = DEFAULT_CAMERA_POS_WORLD
    """Viser camera-frame position, matching visualization_node.py by default."""
    camera_quat_wxyz: tuple[float, float, float, float] = DEFAULT_CAMERA_QUAT_WXYZ
    """Viser camera-frame orientation as wxyz, matching visualization_node.py by default."""
    flip_depth_image_y: bool = False
    """Flip the policy depth image vertically before displaying it in the Viser frustum."""


def main() -> None:
    args = tyro.cli(Args)
    data = np.load(args.recording, allow_pickle=True)
    q = data["q"]
    qd = _npz_get(data, "qd")
    q_targets = data["q_targets"]
    prev_targets = _npz_get(data, "prev_targets")
    if prev_targets is None:
        prev_targets = q_targets
    actions = data["actions"]
    published = _npz_get(data, "published")
    policy_depth = data["policy_depth"]
    depth_format = str(np.asarray(data["policy_depth_format"]).item())
    time_s = data["time_s"]
    pred_pos = _npz_get(data, "predicted_object_pos")
    pred_quat = _npz_get(data, "predicted_object_quat_xyzw")
    actual_pos = _npz_get(data, "actual_object_pos")
    actual_quat = _npz_get(data, "actual_object_quat_xyzw")
    n = int(q.shape[0])

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    server.scene.add_grid("/ground", width=2.0, height=2.0, cell_size=0.1)

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.position = (0.7, -1.2, 0.9)
        client.camera.look_at = (0.0, 0.8, 0.45)

    robot_urdf_path = (
        get_repo_root_dir()
        / "assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf"
    )
    server.scene.add_frame("/robot/state", position=(0.0, 0.8, 0.0), show_axes=False)
    server.scene.add_frame("/robot/pd_target", position=(0.0, 0.8, 0.0), show_axes=False)
    server.scene.add_frame("/robot/prev_target", position=(0.0, 0.8, 0.0), show_axes=False)
    robot = ViserUrdf(server, robot_urdf_path, root_node_name="/robot/state")
    robot_pd_target = ViserUrdf(server, robot_urdf_path, root_node_name="/robot/pd_target", mesh_color_override=(0, 0, 255))
    robot_prev_target = ViserUrdf(
        server,
        robot_urdf_path,
        root_node_name="/robot/prev_target",
        mesh_color_override=ORANGE_RGB,
    )
    for mesh in robot_pd_target._meshes:
        if isinstance(mesh, viser.MeshHandle):
            mesh.opacity = 0.45
    for mesh in robot_prev_target._meshes:
        if isinstance(mesh, viser.MeshHandle):
            mesh.opacity = 0.30

    actual_frame = server.scene.add_frame(
        "/actual_object",
        position=(10.0, 10.0, 10.0),
        wxyz=(1.0, 0.0, 0.0, 0.0),
        show_axes=True,
        axes_length=0.08,
        axes_radius=0.001,
    )
    pred_frame = server.scene.add_frame(
        "/predicted_object",
        position=(10.0, 10.0, 10.0),
        wxyz=(1.0, 0.0, 0.0, 0.0),
        show_axes=True,
        axes_length=0.08,
        axes_radius=0.001,
    )
    if args.object_name in NAME_TO_OBJECT:
        object_urdf = NAME_TO_OBJECT[args.object_name].urdf_path
        ViserUrdf(server, object_urdf, root_node_name="/actual_object")
        ViserUrdf(
            server,
            object_urdf,
            root_node_name="/predicted_object",
            mesh_color_override=LIGHT_BLUE_RGBA,
        )

    server.scene.add_frame(
        DEFAULT_CAMERA_FRAME,
        position=tuple(args.camera_pos_world),
        wxyz=tuple(args.camera_quat_wxyz),
        show_axes=True,
        axes_length=0.08,
        axes_radius=0.002,
    )
    depth_frustum = server.scene.add_camera_frustum(
        f"{DEFAULT_CAMERA_FRAME}/policy_depth",
        fov=0.7,
        aspect=float(policy_depth.shape[2]) / float(policy_depth.shape[1]),
        scale=0.25,
        image=np.zeros((policy_depth.shape[1], policy_depth.shape[2], 3), dtype=np.uint8),
    )

    server.gui.add_markdown("# Student Depth Rollout Recording")
    server.gui.add_markdown(
        "Robot: black=current state, blue=output PD target, orange=previous target. "
        "Object: normal=actual pose, light blue=student predicted pose."
    )
    with server.gui.add_folder("Playback", expand_by_default=True):
        slider = server.gui.add_slider("Frame", min=0, max=max(0, n - 1), step=1, initial_value=0)
        pause_button = server.gui.add_button("Play" if args.start_paused else "Pause")
        status = server.gui.add_markdown("starting")

    paused = bool(args.start_paused)
    frame_idx = 0
    updating_slider_from_code = False

    def update(idx: int, *, sync_slider: bool = True) -> None:
        nonlocal frame_idx, updating_slider_from_code
        frame_idx = int(np.clip(idx, 0, n - 1))
        if sync_slider and int(slider.value) != frame_idx:
            updating_slider_from_code = True
            try:
                slider.value = frame_idx
            finally:
                updating_slider_from_code = False
        robot.update_cfg(q[frame_idx])
        robot_pd_target.update_cfg(q_targets[frame_idx])
        robot_prev_target.update_cfg(prev_targets[frame_idx])
        depth_frustum.image = _viser_frustum_image(
            _policy_depth_to_rgb(policy_depth[frame_idx], depth_format),
            flip_y=args.flip_depth_image_y,
        )
        if actual_pos is not None and actual_quat is not None and np.isfinite(actual_pos[frame_idx]).all():
            actual_frame.position = tuple(
                np.asarray(actual_pos[frame_idx], dtype=np.float64) + WORLD_T_ROBOT_POS_M
            )
            quat_xyzw = np.asarray(actual_quat[frame_idx], dtype=np.float64)
            if np.isfinite(quat_xyzw).all() and np.linalg.norm(quat_xyzw) > 1e-6:
                actual_frame.wxyz = tuple(R.from_quat(quat_xyzw).as_quat()[[3, 0, 1, 2]])
        if pred_pos is not None and pred_quat is not None and np.isfinite(pred_pos[frame_idx]).all():
            pred_frame.position = tuple(np.asarray(pred_pos[frame_idx], dtype=np.float64) + WORLD_T_ROBOT_POS_M)
            quat_xyzw = np.asarray(pred_quat[frame_idx], dtype=np.float64)
            if np.isfinite(quat_xyzw).all() and np.linalg.norm(quat_xyzw) > 1e-6:
                pred_frame.wxyz = tuple(R.from_quat(quat_xyzw).as_quat()[[3, 0, 1, 2]])
        action_abs_max = float(np.max(np.abs(actions[frame_idx])))
        qd_abs_max = float(np.nanmax(np.abs(qd[frame_idx]))) if qd is not None else float("nan")
        target_delta_abs_max = float(np.max(np.abs(q_targets[frame_idx] - prev_targets[frame_idx])))
        published_text = ""
        if published is not None:
            published_text = f" published={bool(published[frame_idx])}"
        status.content = (
            f"frame={frame_idx}/{n - 1} time={float(time_s[frame_idx]):.3f}s "
            f"action_abs_max={action_abs_max:.3f} "
            f"qd_abs_max={qd_abs_max:.3f} "
            f"target_delta_abs_max={target_delta_abs_max:.3f}"
            f"{published_text}"
        )

    @slider.on_update
    def _(_) -> None:
        if updating_slider_from_code:
            return
        update(int(slider.value), sync_slider=False)

    @pause_button.on_click
    def _(_) -> None:
        nonlocal paused
        paused = not paused
        pause_button.label = "Play" if paused else "Pause"

    update(0)
    dt = 1.0 / max(args.fps, 1e-6)
    while True:
        if not paused:
            update((frame_idx + 1) % n)
        time.sleep(dt)


if __name__ == "__main__":
    main()
