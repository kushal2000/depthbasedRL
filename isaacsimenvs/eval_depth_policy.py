"""Evaluate Isaac Lab depth student policies with metrics and visualization.

This script is intentionally evaluation-only.  It shares the same environment
configuration, student checkpoint loading, depth preprocessing, and metric
definitions as ``distill_depth.py`` so results are comparable to the W&B
training curves while adding per-episode CSV/JSON outputs and optional live
viser visualization.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEACHER_DIR = Path("/juno/u/kedia/depthbasedRL/train_dir/Apr28/isaacSim_PegInHole")


def _depth_tensor_to_nchw(depth: torch.Tensor) -> torch.Tensor:
    from isaacsimenvs.distillation.depth_debug import depth_tensor_to_nchw

    return depth_tensor_to_nchw(depth)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _gray_to_rgb_u8(image: np.ndarray) -> np.ndarray:
    gray = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
    gray = np.clip(gray, 0.0, 1.0)
    u8 = (gray * 255.0).round().astype(np.uint8)
    return np.repeat(u8[..., None], 3, axis=-1)


def _metric_depth_window(depth: np.ndarray, near: float, far: float) -> np.ndarray:
    window = np.nan_to_num((depth - near) / max(far - near, 1e-6), nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(window, 0.0, 1.0)


def _student_camera_k(cfg) -> np.ndarray:
    intrinsic = tuple(float(x) for x in getattr(cfg, "camera_intrinsic_matrix", ()))
    if intrinsic:
        if len(intrinsic) != 9:
            raise ValueError(f"camera_intrinsic_matrix must have 9 values, got {len(intrinsic)}")
        return np.asarray(intrinsic, dtype=np.float32).reshape(3, 3)

    width = float(cfg.image_width)
    height = float(cfg.image_height)
    focal_length = float(cfg.focal_length)
    horizontal_aperture = float(cfg.horizontal_aperture)
    fx = focal_length * width / horizontal_aperture
    # Isaac's pinhole cfg uses square pixels for this path; vertical aperture is
    # implied by the image aspect ratio, which leaves fy equal to fx in pixels.
    fy = fx
    return np.asarray(
        [
            [fx, 0.0, (width - 1.0) * 0.5],
            [0.0, fy, (height - 1.0) * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _crop_camera_k(k: np.ndarray, cfg) -> np.ndarray:
    if not bool(cfg.crop_enabled):
        return k.copy()
    x0, y0 = (int(v) for v in cfg.crop_top_left)
    cropped = k.copy()
    cropped[0, 2] -= float(x0)
    cropped[1, 2] -= float(y0)
    return cropped


def _points_from_depth(depth_m: np.ndarray, k: np.ndarray, stride: int) -> np.ndarray:
    height, width = depth_m.shape
    stride = max(1, int(stride))
    ys, xs = np.mgrid[0:height:stride, 0:width:stride]
    z = depth_m[ys, xs]
    x = (xs.astype(np.float32) - float(k[0, 2])) / float(k[0, 0]) * z
    y = (ys.astype(np.float32) - float(k[1, 2])) / float(k[1, 1]) * z
    return np.stack((x, y, z), axis=-1).reshape(-1, 3)


def _camera_pose_env_frame(env, env_id: int) -> tuple[np.ndarray, np.ndarray]:
    camera = env.student_camera
    origin = _to_numpy(env.scene.env_origins[env_id]).astype(np.float32)
    data = camera.data

    if hasattr(env, "_student_camera_current_pos_w") and hasattr(env, "_student_camera_current_quat_wxyz"):
        pos_w = _to_numpy(env._student_camera_current_pos_w[env_id]).astype(np.float32)
        quat = env._student_camera_current_quat_wxyz[env_id : env_id + 1]
        convention = str(env.cfg.student_obs.camera_convention).lower()
        if convention != "ros":
            from isaaclab.utils.math import convert_camera_frame_orientation_convention

            quat = convert_camera_frame_orientation_convention(quat, origin=convention, target="ros")
        quat_wxyz = _to_numpy(quat[0]).astype(np.float32)
    elif hasattr(data, "quat_w_ros"):
        pos_w = _to_numpy(data.pos_w[env_id]).astype(np.float32)
        quat_wxyz = _to_numpy(data.quat_w_ros[env_id]).astype(np.float32)
    elif hasattr(data, "quat_w_world"):
        pos_w = _to_numpy(data.pos_w[env_id]).astype(np.float32)
        quat_wxyz = _to_numpy(data.quat_w_world[env_id]).astype(np.float32)
    else:
        pos_w = _to_numpy(env._student_camera_current_pos_w[env_id]).astype(np.float32)
        quat_wxyz = _to_numpy(env._student_camera_current_quat_wxyz[env_id]).astype(np.float32)

    return pos_w - origin, quat_wxyz


def _viser_frustum_image(image: np.ndarray) -> np.ndarray:
    """Convert an image array into Viser's camera-frustum texture orientation."""

    # Keep the rendered image orientation exactly as the policy/debug videos see
    # it. Only Viser display uses this helper; policy tensors are untouched.
    return np.ascontiguousarray(image)


def _object_urdf_path_for_env(env, env_id: int) -> Path:
    urdf_paths = getattr(env, "_object_urdf_paths", None)
    asset_indices = getattr(env, "_object_asset_index_per_env", None)
    if not urdf_paths or asset_indices is None:
        raise RuntimeError("Env does not expose _object_urdf_paths/_object_asset_index_per_env")
    asset_index = int(asset_indices[env_id].detach().cpu().item())
    return Path(urdf_paths[asset_index])


def _table_urdf_path_for_env(env, env_id: int) -> Path:
    table_paths = getattr(env, "_table_urdf_paths", None)
    if table_paths:
        return Path(table_paths[env_id % len(table_paths)])
    return REPO_ROOT / "assets" / "urdf" / "table_narrow.urdf"


class DepthEvalViser:
    """Small live viser viewer for depth-policy eval."""

    def __init__(
        self,
        env,
        *,
        env_id: int,
        port: int,
        point_cloud: bool,
        point_stride: int,
        show_robot: bool,
        start_paused: bool,
    ) -> None:
        import viser
        from viser.extras import ViserUrdf

        self.viser = viser
        self.env = env
        self.env_id = int(env_id)
        self.point_cloud_enabled = bool(point_cloud)
        self.point_stride = max(1, int(point_stride))
        self.paused = bool(start_paused)
        self._single_step_requested = False
        self._restart_requested = False
        self._clear_stats_requested = False
        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        self.server.scene.add_grid("/ground", width=2.0, height=2.0, cell_size=0.1)

        @self.server.on_client_connect
        def _(client):
            client.camera.position = (0.9, -1.2, 0.9)
            client.camera.look_at = (0.0, 0.0, 0.45)

        self.server.gui.add_markdown("# Depth Policy Eval")
        with self.server.gui.add_folder("Episode Controls", expand_by_default=True):
            self._run_pause_button = self.server.gui.add_button("Run" if self.paused else "Pause")
            self._run_pause_button.on_click(lambda _: self._toggle_paused())
            self._step_button = self.server.gui.add_button("Step Once")
            self._step_button.on_click(lambda _: self._request_single_step())
            self._restart_button = self.server.gui.add_button("Restart Eval")
            self._restart_button.on_click(lambda _: self._request_restart())
            self._clear_stats_button = self.server.gui.add_button("Clear Stats")
            self._clear_stats_button.on_click(lambda _: self._request_clear_stats())
            self.control_status = self.server.gui.add_markdown("**Controls:** --")

        with self.server.gui.add_folder("Status", expand_by_default=True):
            self.status = self.server.gui.add_markdown("**Depth eval:** starting")
            self.metrics = self.server.gui.add_markdown("**Metrics:** --")

        self._update_control_status()

        self.robot = None
        if show_robot:
            robot_urdf = REPO_ROOT / "assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf"
            if robot_urdf.exists():
                self.robot_frame = self.server.scene.add_frame(
                    "/robot",
                    position=(0.0, 0.8, 0.0),
                    wxyz=(1.0, 0.0, 0.0, 0.0),
                    show_axes=False,
                )
                self.robot = ViserUrdf(self.server, robot_urdf, root_node_name="/robot")
            else:
                print(f"[eval_depth_policy] robot URDF missing, skipping viser robot: {robot_urdf}", flush=True)

        self.table_frame = self.server.scene.add_frame("/table", show_axes=False)
        self.object_frame = self.server.scene.add_frame("/object", show_axes=True, axes_length=0.07, axes_radius=0.001)
        self.goal_frame = self.server.scene.add_frame("/goal", show_axes=True, axes_length=0.07, axes_radius=0.001)
        self.pred_frame = self.server.scene.add_frame(
            "/object_pred", show_axes=True, axes_length=0.07, axes_radius=0.001
        )
        self.camera_frame = self.server.scene.add_frame(
            "/student_camera", show_axes=True, axes_length=0.08, axes_radius=0.001
        )

        ViserUrdf(self.server, _table_urdf_path_for_env(env, env_id), root_node_name="/table")
        object_urdf = _object_urdf_path_for_env(env, env_id)
        ViserUrdf(self.server, object_urdf, root_node_name="/object")
        ViserUrdf(self.server, object_urdf, root_node_name="/goal", mesh_color_override=(0, 180, 40))
        ViserUrdf(self.server, object_urdf, root_node_name="/object_pred", mesh_color_override=(25, 75, 255))

        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        self.full_frustum = self.server.scene.add_camera_frustum(
            "/student_camera/full_depth_window",
            fov=0.7,
            aspect=1.0,
            scale=0.12,
            line_width=2.0,
            color=(0, 0, 0),
            image=dummy,
        )
        self.policy_frustum = self.server.scene.add_camera_frustum(
            "/student_camera/policy_input",
            fov=0.7,
            aspect=1.0,
            scale=0.10,
            line_width=2.0,
            color=(25, 75, 255),
            image=dummy,
        )
        self.point_cloud = None

    def _update_control_status(self) -> None:
        state = "paused" if self.paused else "running"
        self.control_status.content = f"**Controls:** {state}"

    def _toggle_paused(self) -> None:
        self.paused = not self.paused
        self._run_pause_button.name = "Run" if self.paused else "Pause"
        self._update_control_status()

    def _request_single_step(self) -> None:
        self._single_step_requested = True
        self.paused = True
        self._run_pause_button.name = "Run"
        self._update_control_status()

    def _request_restart(self) -> None:
        self._restart_requested = True
        self._clear_stats_requested = True
        self.paused = True
        self._run_pause_button.name = "Run"
        self.control_status.content = "**Controls:** restart requested"

    def _request_clear_stats(self) -> None:
        self._clear_stats_requested = True
        self.control_status.content = "**Controls:** clear stats requested"

    def consume_restart_requested(self) -> bool:
        requested = self._restart_requested
        self._restart_requested = False
        if requested:
            self._update_control_status()
        return requested

    def consume_clear_stats_requested(self) -> bool:
        requested = self._clear_stats_requested
        self._clear_stats_requested = False
        return requested

    def should_step(self) -> bool:
        if not self.paused:
            return True
        if self._single_step_requested:
            self._single_step_requested = False
            return True
        return False

    def _joint_pos(self) -> np.ndarray:
        env = self.env
        if hasattr(env, "_perm_lab_to_canon"):
            return _to_numpy(env.robot.data.joint_pos[self.env_id, env._perm_lab_to_canon]).astype(np.float32)
        return _to_numpy(env.robot.data.joint_pos[self.env_id]).astype(np.float32)

    def _pose(self, pos_w: torch.Tensor, quat_wxyz: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        origin = self.env.scene.env_origins[self.env_id]
        pos = _to_numpy(pos_w[self.env_id] - origin).astype(np.float32)
        quat = _to_numpy(quat_wxyz[self.env_id]).astype(np.float32)
        return pos, quat

    def update(
        self,
        *,
        step: int,
        completed_episodes: int,
        current_goal_idx: float,
        recent_goal_idx: float,
        policy_depth: torch.Tensor | None,
        predicted_object_pose_wxyz: torch.Tensor | None,
    ) -> None:
        env = self.env
        env_id = self.env_id
        if self.robot is not None:
            self.robot.update_cfg(self._joint_pos())

        table_pos, table_quat = self._pose(env.table.data.root_pos_w, env.table.data.root_quat_w)
        object_pos, object_quat = self._pose(env.object.data.root_pos_w, env.object.data.root_quat_w)
        goal_pos, goal_quat = self._pose(env.goal_viz.data.root_pos_w, env.goal_viz.data.root_quat_w)
        self.table_frame.position = table_pos
        self.table_frame.wxyz = table_quat
        self.object_frame.position = object_pos
        self.object_frame.wxyz = object_quat
        self.goal_frame.position = goal_pos
        self.goal_frame.wxyz = goal_quat
        if predicted_object_pose_wxyz is not None:
            pred = predicted_object_pose_wxyz[env_id].detach()
            self.pred_frame.position = _to_numpy(pred[:3]).astype(np.float32)
            self.pred_frame.wxyz = _to_numpy(pred[3:7]).astype(np.float32)

        cam_pos, cam_quat = _camera_pose_env_frame(env, env_id)
        self.camera_frame.position = cam_pos
        self.camera_frame.wxyz = cam_quat

        cfg = env.cfg.student_obs
        near = float(cfg.depth_min_m)
        far = float(cfg.depth_max_m)
        full_k = _student_camera_k(cfg)
        crop_k = _crop_camera_k(full_k, cfg)
        raw_depth = getattr(env, "_student_depth_raw_m", None)
        policy_full = getattr(env, "_student_depth_policy_full", None)
        if raw_depth is not None:
            raw_np = _depth_tensor_to_nchw(raw_depth)[env_id, 0].detach().float().cpu().numpy()
            full_img = _gray_to_rgb_u8(_metric_depth_window(raw_np, near, far))
        elif policy_full is not None:
            raw_np = None
            full_img = _gray_to_rgb_u8(
                _depth_tensor_to_nchw(policy_full)[env_id, 0].detach().float().cpu().numpy()
            )
        else:
            raw_np = None
            full_img = np.zeros((int(cfg.image_height), int(cfg.image_width), 3), dtype=np.uint8)

        if policy_depth is not None:
            policy_img = _gray_to_rgb_u8(
                _depth_tensor_to_nchw(policy_depth)[env_id, 0].detach().float().cpu().numpy()
            )
        else:
            policy_img = full_img

        self._update_frustum(
            self.full_frustum,
            image=_viser_frustum_image(full_img),
            k=full_k,
            scale=0.14,
        )
        self._update_frustum(
            self.policy_frustum,
            image=_viser_frustum_image(policy_img),
            k=crop_k,
            scale=0.10,
        )
        if self.point_cloud_enabled and raw_np is not None:
            self._update_point_cloud(raw_np, full_k, full_img, near, far)

        self.status.content = f"**Depth eval:** step={step}, completed={completed_episodes}"
        self.metrics.content = (
            f"**Metrics:** current_goal_idx_avg={current_goal_idx:.3f}, "
            f"recent_reset_goal_idx_avg={recent_goal_idx:.3f}"
        )

    def _update_frustum(self, handle, *, image: np.ndarray, k: np.ndarray, scale: float):
        height, width = image.shape[:2]
        fy = float(k[1, 1])
        handle.image = image
        handle.fov = float(2.0 * np.arctan(height / (2.0 * fy)))
        handle.aspect = float(width / height)
        handle.scale = float(scale)
        # These frustums are children of /student_camera, whose pose is already
        # the actual rendered camera pose. Keep child transforms local-identity.
        handle.position = (0.0, 0.0, 0.0)
        handle.wxyz = (1.0, 0.0, 0.0, 0.0)

    def _update_point_cloud(
        self,
        depth_m: np.ndarray,
        k: np.ndarray,
        full_img: np.ndarray,
        near: float,
        far: float,
    ) -> None:
        # Points are in the same OpenCV/ROS camera frame used by /student_camera:
        # x right, y down, z forward. Keeping the cloud under /student_camera
        # avoids manually reapplying the camera transform and prevents
        # parent/child double-transform bugs.
        points_c = _points_from_depth(depth_m, k, self.point_stride)
        colors = full_img[:: self.point_stride, :: self.point_stride].reshape(-1, 3)
        finite = np.isfinite(points_c).all(axis=1)
        in_range = finite & (points_c[:, 2] >= max(0.01, near - 0.2)) & (points_c[:, 2] <= far + 0.5)
        points_c = points_c[in_range]
        colors = colors[in_range]
        if self.point_cloud is None:
            self.point_cloud = self.server.scene.add_point_cloud(
                "/student_camera/point_cloud",
                points=points_c,
                colors=colors,
                point_size=0.007,
            )
        else:
            self.point_cloud.points = points_c
            self.point_cloud.colors = colors


@dataclass
class EpisodeRecord:
    episode_index: int
    step: int
    env_id: int
    scene_idx: int
    tol_slot_idx: int
    peg_idx: int
    object_asset_idx: int
    start_object_x: float
    start_object_y: float
    start_object_z: float
    start_object_qw: float
    start_object_qx: float
    start_object_qy: float
    start_object_qz: float
    episode_length_steps: int
    goal_idx: float
    goal_completion_ratio: float
    max_goals: int
    done_fall: bool
    done_max_successes: bool
    done_hand_far: bool
    done_timeout: bool


def _write_episode_csv(path: Path, records: list[EpisodeRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(EpisodeRecord.__dataclass_fields__))
        writer.writeheader()
        for record in records:
            writer.writerow(record.__dict__)


def _episode_records_for_dones(
    *,
    env,
    dones: torch.Tensor,
    step: int,
    episode_lengths: torch.Tensor,
    episode_context: dict[str, torch.Tensor],
    start_index: int,
) -> list[EpisodeRecord]:
    done = dones.reshape(-1).bool()
    done_ids = done.nonzero(as_tuple=False).squeeze(-1)
    if done_ids.numel() == 0:
        return []

    successes = env._prev_episode_successes[done_ids].detach().float().cpu()
    max_goals = env.prev_episode_env_max_goals[done_ids].detach().long().cpu()
    lengths = episode_lengths[done_ids].detach().long().cpu()
    reasons = getattr(env, "_termination_reasons", {})
    scene_idx = episode_context["scene_idx"][done_ids].detach().long().cpu()
    tol_slot_idx = episode_context["tol_slot_idx"][done_ids].detach().long().cpu()
    peg_idx = episode_context["peg_idx"][done_ids].detach().long().cpu()
    object_asset_idx = episode_context["object_asset_idx"][done_ids].detach().long().cpu()
    start_object_pos = episode_context["start_object_pos"][done_ids].detach().float().cpu()
    start_object_quat = episode_context["start_object_quat"][done_ids].detach().float().cpu()

    records: list[EpisodeRecord] = []
    for i, env_id_t in enumerate(done_ids.detach().cpu()):
        env_id = int(env_id_t.item())
        max_goal = max(int(max_goals[i].item()), 1)
        goal_idx = float(successes[i].item())

        def reason(name: str) -> bool:
            value = reasons.get(name)
            if value is None:
                return False
            return bool(value[env_id].detach().cpu().item())

        records.append(
            EpisodeRecord(
                episode_index=start_index + i,
                step=int(step),
                env_id=env_id,
                scene_idx=int(scene_idx[i].item()),
                tol_slot_idx=int(tol_slot_idx[i].item()),
                peg_idx=int(peg_idx[i].item()),
                object_asset_idx=int(object_asset_idx[i].item()),
                start_object_x=float(start_object_pos[i, 0].item()),
                start_object_y=float(start_object_pos[i, 1].item()),
                start_object_z=float(start_object_pos[i, 2].item()),
                start_object_qw=float(start_object_quat[i, 0].item()),
                start_object_qx=float(start_object_quat[i, 1].item()),
                start_object_qy=float(start_object_quat[i, 2].item()),
                start_object_qz=float(start_object_quat[i, 3].item()),
                episode_length_steps=int(lengths[i].item()),
                goal_idx=goal_idx,
                goal_completion_ratio=goal_idx / float(max_goal),
                max_goals=max_goal,
                done_fall=reason("fall"),
                done_max_successes=reason("max_successes"),
                done_hand_far=reason("hand_far"),
                done_timeout=reason("timeout"),
            )
        )
    return records


def _make_episode_context(env) -> dict[str, torch.Tensor]:
    """Capture per-env context for the currently active episode."""

    def long_attr(name: str, default: int = -1) -> torch.Tensor:
        value = getattr(env, name, None)
        if value is None:
            return torch.full((env.num_envs,), default, device=env.device, dtype=torch.long)
        return value.detach().long().clone()

    object_asset_idx = getattr(env, "_object_asset_index_per_env", None)
    if object_asset_idx is None:
        object_asset_idx = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
    else:
        object_asset_idx = object_asset_idx.detach().long().clone()

    return {
        "scene_idx": long_attr("_pih_env_scene_idx_t"),
        "tol_slot_idx": long_attr("_pih_env_tol_slot_t"),
        "peg_idx": long_attr("env_peg_idx"),
        "object_asset_idx": object_asset_idx,
        "start_object_pos": (env.object.data.root_pos_w - env.scene.env_origins).detach().float().clone(),
        "start_object_quat": env.object.data.root_quat_w.detach().float().clone(),
    }


def _refresh_episode_context(env, context: dict[str, torch.Tensor], env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
        return
    latest = _make_episode_context(env)
    for key, value in latest.items():
        context[key][env_ids] = value[env_ids]


def _summary(records: list[EpisodeRecord], final_current_goal_idx: float, final_current_completion: float) -> dict[str, Any]:
    def group_summary(field: str) -> dict[str, dict[str, float | int]]:
        groups: dict[str, list[EpisodeRecord]] = {}
        for record in records:
            groups.setdefault(str(getattr(record, field)), []).append(record)

        out: dict[str, dict[str, float | int]] = {}
        for key, group in sorted(groups.items(), key=lambda item: item[0]):
            goal_idx = np.asarray([r.goal_idx for r in group], dtype=np.float32)
            completion = np.asarray([r.goal_completion_ratio for r in group], dtype=np.float32)
            max_goals = np.asarray([r.max_goals for r in group], dtype=np.float32)
            out[key] = {
                "count": int(len(group)),
                "goal_idx_avg": float(goal_idx.mean()),
                "goal_completion_ratio_avg": float(completion.mean()),
                "full_success_rate": float((goal_idx >= max_goals).mean()),
            }
        return out

    if records:
        goal_idx = np.asarray([r.goal_idx for r in records], dtype=np.float32)
        completion = np.asarray([r.goal_completion_ratio for r in records], dtype=np.float32)
        lengths = np.asarray([r.episode_length_steps for r in records], dtype=np.float32)
        max_goals = np.asarray([r.max_goals for r in records], dtype=np.float32)
        summary = {
            "completed_episode_count": int(len(records)),
            "completed_goal_idx_avg": float(goal_idx.mean()),
            "completed_goal_idx_std": float(goal_idx.std()),
            "completed_goal_idx_min": float(goal_idx.min()),
            "completed_goal_idx_max": float(goal_idx.max()),
            "completed_goal_completion_ratio_avg": float(completion.mean()),
            "completed_full_success_rate": float((goal_idx >= max_goals).mean()),
            "completed_episode_length_avg": float(lengths.mean()),
            "done_fall_count": int(sum(r.done_fall for r in records)),
            "done_max_successes_count": int(sum(r.done_max_successes for r in records)),
            "done_hand_far_count": int(sum(r.done_hand_far for r in records)),
            "done_timeout_count": int(sum(r.done_timeout for r in records)),
        }
        for value in sorted(set(float(x) for x in goal_idx)):
            summary[f"completed_goal_idx_eq_{value:g}_count"] = int((goal_idx == value).sum())
        summary["by_scene_idx"] = group_summary("scene_idx")
        summary["by_tol_slot_idx"] = group_summary("tol_slot_idx")
        summary["by_peg_idx"] = group_summary("peg_idx")
        summary["by_object_asset_idx"] = group_summary("object_asset_idx")
    else:
        summary = {
            "completed_episode_count": 0,
            "completed_goal_idx_avg": 0.0,
            "completed_goal_completion_ratio_avg": 0.0,
            "completed_full_success_rate": 0.0,
        }
    summary["final_current_goal_idx_avg"] = float(final_current_goal_idx)
    summary["final_current_goal_completion_ratio_avg"] = float(final_current_completion)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Isaacsimenvs-PegInHoleDepthStudent-Direct-v0")
    parser.add_argument("--teacher_agent", default="rl_games_sapg_cfg_entry_point")
    parser.add_argument("--teacher_checkpoint", type=Path, default=DEFAULT_TEACHER_DIR / "model.pth")
    parser.add_argument("--teacher_config", type=Path, default=DEFAULT_TEACHER_DIR / "config.yaml")
    parser.add_argument("--student_checkpoint", type=Path, required=True)
    parser.add_argument("--student_checkpoint_strict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--student_input", choices=("camera", "teacher_obs"), default="camera")
    parser.add_argument("--student_arch", choices=("mono_transformer_recurrent", "mlp_recurrent"), default=None)
    parser.add_argument(
        "--student_module_mode",
        choices=("eval", "train"),
        default="eval",
        help="Torch module mode for the student. Use 'train' to reproduce distill_depth.py's online rollout behavior.",
    )
    parser.add_argument("--policy_source", choices=("student", "teacher"), default="student")
    parser.add_argument("--run_dir", type=Path, default=None)
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--num_steps", type=int, default=2000)
    parser.add_argument("--num_completed_episodes", type=int, default=0)
    parser.add_argument(
        "--one_episode_per_env",
        action="store_true",
        help="Record only the first completed episode from each initial env and stop after all envs finish once.",
    )
    parser.add_argument(
        "--rolling_reset_window_size",
        type=int,
        default=1000,
        help="Number of most recent completed episodes to use for rolling reset metrics.",
    )
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument(
        "--aux_pose_mode",
        choices=("none", "position", "rot6d_keypoints"),
        default="rot6d_keypoints",
    )
    parser.add_argument("--aux_object_pos_weight", type=float, default=1.0)
    parser.add_argument("--aux_object_keypoint_weight", type=float, default=1.0)
    parser.add_argument("--deterministic_teacher", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rl_device", default="cuda:0")
    parser.add_argument("--sim_device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--force_scene_tol_combo", default=None)
    parser.add_argument("--force_peg_idx", type=int, default=None)
    parser.add_argument("--peg_urdf", default=None)
    parser.add_argument("--peg_goal_mode", choices=("dense", "preInsertAndFinal", "finalGoalOnly"), default=None)
    parser.add_argument("--peg_enable_retract", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--student_image_delay_queue_size", type=int, default=None)
    parser.add_argument("--depth_noise_profile", choices=("off", "weak", "medium", "strong", "custom"), default=None)
    parser.add_argument("--depth_noise_strength", type=float, default=None)
    parser.add_argument(
        "--student_camera_preset",
        choices=("default", "samrat_zed2i_debug", "samrat_zed2i_160x90_debug", "samrat_zed2i_half_debug"),
        default="default",
    )
    parser.add_argument("--camera_pose_randomization_profile", choices=("off", "weak", "medium", "strong", "custom"), default=None)
    parser.add_argument("--camera_pose_randomization_mode", choices=("startup", "reset"), default=None)
    parser.add_argument("--camera_pos_noise_m", type=float, nargs=3, default=None)
    parser.add_argument("--camera_rot_noise_deg", type=float, nargs=3, default=None)
    parser.add_argument("--peg_object_init_orientation_mode", choices=("scene", "yaw_only", "full"), default=None)
    parser.add_argument("--peg_object_init_position_noise_xy", type=float, nargs=2, default=None)
    parser.add_argument("--peg_object_init_position_noise_z", type=float, default=None)
    parser.add_argument("--capture_viewer", action="store_true")
    parser.add_argument("--capture_viewer_len", type=int, default=600)
    parser.add_argument("--depth_debug_interval", type=int, default=0)
    parser.add_argument("--depth_debug_env_ids", default="0,1,2,3")
    parser.add_argument("--depth_rollout_video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--depth_rollout_video_env_ids", default="0")
    parser.add_argument("--depth_rollout_video_len", type=int, default=600)
    parser.add_argument("--depth_rollout_video_fps", type=int, default=60)
    parser.add_argument("--depth_rollout_video_interval", type=int, default=600)
    parser.add_argument("--serve_viser", action="store_true")
    parser.add_argument("--viser_port", type=int, default=8080)
    parser.add_argument("--viser_env_id", type=int, default=0)
    parser.add_argument("--viser_update_interval", type=int, default=1)
    parser.add_argument("--viser_sleep_s", type=float, default=0.0)
    parser.add_argument("--viser_point_cloud", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--viser_point_stride", type=int, default=4)
    parser.add_argument("--viser_robot", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--viser_start_paused", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="depthbasedRL-isaacsim-distill")
    parser.add_argument("--wandb_group", default="")
    parser.add_argument("--wandb_entity", default="")
    parser.add_argument("--wandb_name", default="")
    parser.add_argument(
        "--force_exit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force os._exit(0) after successful Isaac cleanup to avoid Kit shutdown hangs.",
    )

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.student_checkpoint = args.student_checkpoint.expanduser()
    if not args.student_checkpoint.exists():
        raise FileNotFoundError(f"Student checkpoint not found: {args.student_checkpoint}")
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    if args.serve_viser:
        # Import before Isaac Sim mutates sys.path with its pip prebundle.
        # Otherwise viser may resolve Isaac's bundled websockets package and fail.
        import viser  # noqa: F401
        from viser.extras import ViserUrdf  # noqa: F401
    args.enable_cameras = args.student_input == "camera"
    app = AppLauncher(args).app

    import gymnasium as gym

    import isaacsimenvs  # noqa: F401
    from isaacsimenvs.distill_depth import (
        _apply_student_camera_preset,
        _aux_head_dims,
        _capture_depth_rollout_frame,
        _capture_viewer_if_needed,
        _compute_aux_losses,
        _done_success_values,
        _init_wandb,
        _load_env_cfg,
        _load_student_checkpoint,
        _load_teacher_player,
        _log_depth_rollout_video,
        _log_metrics,
        _parse_optional_pair,
        _reset_hidden_for_done,
        _student_image_channels,
        _teacher_obs_tensor,
    )
    from isaacsimenvs.distillation.depth_debug import save_depth_debug
    from isaacsimenvs.distillation.student_policy import MLPRecurrentPolicy, MonoTransformerRecurrentPolicy

    run_dir = args.run_dir or REPO_ROOT / "eval_runs" / f"depth_policy_eval_{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[eval_depth_policy] run_dir={run_dir}", flush=True)

    env_cfg = _load_env_cfg(args.task, args.teacher_config, args.num_envs, args.sim_device)
    if args.seed is not None and hasattr(env_cfg, "seed"):
        env_cfg.seed = int(args.seed)
    if args.student_input == "teacher_obs":
        env_cfg.student_obs.image_enabled = False
    if args.force_scene_tol_combo is not None:
        env_cfg.peg_in_hole.force_scene_tol_combo = _parse_optional_pair(args.force_scene_tol_combo)
    if args.force_peg_idx is not None:
        env_cfg.peg_in_hole.force_peg_idx = args.force_peg_idx
    if args.peg_urdf is not None:
        env_cfg.assets.peg_urdf = args.peg_urdf
        env_cfg.assets.object_name = Path(args.peg_urdf).stem
    if args.peg_goal_mode is not None:
        env_cfg.peg_in_hole.goal_mode = args.peg_goal_mode
    if args.peg_enable_retract is not None:
        env_cfg.peg_in_hole.enable_retract = bool(args.peg_enable_retract)
    if args.student_image_delay_queue_size is not None:
        queue_size = int(args.student_image_delay_queue_size)
        if queue_size < 1:
            raise ValueError("--student_image_delay_queue_size must be >= 1")
        env_cfg.student_obs.use_camera_delay = queue_size > 1
        env_cfg.student_obs.camera_delay_max = queue_size
    _apply_student_camera_preset(env_cfg, args.student_camera_preset)
    if args.depth_noise_profile is not None:
        env_cfg.student_obs.depth_noise_profile = args.depth_noise_profile
    if args.depth_noise_strength is not None:
        env_cfg.student_obs.depth_noise_strength = args.depth_noise_strength
    if args.camera_pose_randomization_profile is not None:
        env_cfg.student_obs.camera_pose_randomization_profile = args.camera_pose_randomization_profile
    if args.camera_pose_randomization_mode is not None:
        env_cfg.student_obs.camera_pose_randomization_mode = args.camera_pose_randomization_mode
    if args.camera_pos_noise_m is not None:
        env_cfg.student_obs.camera_pos_noise_m = tuple(args.camera_pos_noise_m)
    if args.camera_rot_noise_deg is not None:
        env_cfg.student_obs.camera_rot_noise_deg = tuple(args.camera_rot_noise_deg)
    if args.peg_object_init_orientation_mode is not None:
        env_cfg.peg_in_hole.object_init_orientation_mode = args.peg_object_init_orientation_mode
    if args.peg_object_init_position_noise_xy is not None:
        env_cfg.peg_in_hole.object_init_position_noise_xy = tuple(args.peg_object_init_position_noise_xy)
    if args.peg_object_init_position_noise_z is not None:
        env_cfg.peg_in_hole.object_init_position_noise_z = args.peg_object_init_position_noise_z

    env = gym.make(args.task, cfg=env_cfg)
    inner = env.unwrapped
    wrapped, teacher = _load_teacher_player(
        env,
        task=args.task,
        agent=args.teacher_agent,
        checkpoint=args.teacher_checkpoint,
        rl_device=args.rl_device,
    )
    obs = teacher.env_reset(wrapped)

    if args.student_input == "teacher_obs":
        arch = args.student_arch or "mlp_recurrent"
        if arch != "mlp_recurrent":
            raise ValueError("--student_input teacher_obs requires --student_arch mlp_recurrent")
        teacher_obs = _teacher_obs_tensor(obs)
        student = MLPRecurrentPolicy(
            obs_dim=teacher_obs.shape[-1],
            action_dim=int(env_cfg.action_space),
            aux_heads=_aux_head_dims(args.aux_pose_mode),
        ).to(inner.device)
        print(f"[eval_depth_policy] teacher_obs student obs_dim={teacher_obs.shape[-1]}", flush=True)
    else:
        arch = args.student_arch or "mono_transformer_recurrent"
        if arch != "mono_transformer_recurrent":
            raise ValueError("--student_input camera requires --student_arch mono_transformer_recurrent")
        student_obs = inner.get_student_obs()
        image = student_obs["image"]
        proprio = student_obs["proprio"]
        student = MonoTransformerRecurrentPolicy(
            image_channels=_student_image_channels(str(env_cfg.student_obs.image_modality).lower()),
            proprio_dim=proprio.shape[-1],
            action_dim=int(env_cfg.action_space),
            aux_heads=_aux_head_dims(args.aux_pose_mode),
        ).to(inner.device)
        print(
            "[eval_depth_policy] "
            f"camera image_shape={tuple(image.shape)} proprio_dim={proprio.shape[-1]} "
            f"action_dim={int(env_cfg.action_space)} aux_pose_mode={args.aux_pose_mode}",
            flush=True,
        )

    _load_student_checkpoint(args.student_checkpoint, student, optimizer=None, strict=args.student_checkpoint_strict)
    if args.student_module_mode == "eval":
        student.eval()
    else:
        student.train()
    hidden = student.initial_state(inner.num_envs, inner.device)
    wandb_run = _init_wandb(args, run_dir)

    depth_debug_env_ids = [int(item) for item in args.depth_debug_env_ids.replace(",", " ").split() if item]
    depth_debug_env_ids = [idx for idx in depth_debug_env_ids if 0 <= idx < inner.num_envs]
    depth_rollout_env_ids = [int(item) for item in args.depth_rollout_video_env_ids.replace(",", " ").split() if item]
    depth_rollout_env_ids = [idx for idx in depth_rollout_env_ids if 0 <= idx < inner.num_envs]

    viser_viewer = None
    if args.serve_viser:
        if args.viser_env_id < 0 or args.viser_env_id >= inner.num_envs:
            raise ValueError(f"--viser_env_id={args.viser_env_id} out of range for num_envs={inner.num_envs}")
        viser_viewer = DepthEvalViser(
            inner,
            env_id=args.viser_env_id,
            port=args.viser_port,
            point_cloud=args.viser_point_cloud,
            point_stride=args.viser_point_stride,
            show_robot=args.viser_robot,
            start_paused=args.viser_start_paused,
        )
        print(f"[eval_depth_policy] viser server: http://localhost:{args.viser_port}", flush=True)

    viewer_frames: list = []
    depth_rollout_video_frames: list = []
    episode_lengths = torch.zeros(inner.num_envs, dtype=torch.long, device=inner.device)
    episode_context = _make_episode_context(inner)
    first_episode_recorded = torch.zeros(inner.num_envs, dtype=torch.bool, device=inner.device)
    episode_records: list[EpisodeRecord] = []
    interval_action_loss = 0.0
    interval_aux_loss = 0.0
    interval_aux_pos_loss = 0.0
    interval_aux_keypoint_loss = 0.0
    interval_step_count = 0
    interval_done_goal_idx = 0.0
    interval_done_completion = 0.0
    interval_done_full_success = 0.0
    interval_done_count = 0
    rolling_reset_window = deque(maxlen=max(1, int(args.rolling_reset_window_size)))
    interval_start = time.perf_counter()
    last_step = 0

    def clear_eval_stats() -> None:
        nonlocal interval_action_loss
        nonlocal interval_aux_loss
        nonlocal interval_aux_pos_loss
        nonlocal interval_aux_keypoint_loss
        nonlocal interval_step_count
        nonlocal interval_done_goal_idx
        nonlocal interval_done_completion
        nonlocal interval_done_full_success
        nonlocal interval_done_count
        nonlocal interval_start

        episode_records.clear()
        first_episode_recorded.zero_()
        rolling_reset_window.clear()
        interval_action_loss = 0.0
        interval_aux_loss = 0.0
        interval_aux_pos_loss = 0.0
        interval_aux_keypoint_loss = 0.0
        interval_step_count = 0
        interval_done_goal_idx = 0.0
        interval_done_completion = 0.0
        interval_done_full_success = 0.0
        interval_done_count = 0
        interval_start = time.perf_counter()
        viewer_frames.clear()
        depth_rollout_video_frames.clear()

    def reset_all_envs() -> None:
        nonlocal obs
        nonlocal hidden
        nonlocal episode_context

        obs = teacher.env_reset(wrapped)
        hidden = student.initial_state(inner.num_envs, inner.device)
        episode_lengths.zero_()
        episode_context = _make_episode_context(inner)

    def update_live_viewer(policy_depth: torch.Tensor | None, predicted_pose: torch.Tensor | None) -> None:
        if viser_viewer is None:
            return
        current_goal_idx = float(inner._successes.float().mean().detach().cpu().item())
        recent_goal_idx = interval_done_goal_idx / max(interval_done_count, 1)
        viser_viewer.update(
            step=last_step,
            completed_episodes=len(episode_records),
            current_goal_idx=current_goal_idx,
            recent_goal_idx=recent_goal_idx,
            policy_depth=policy_depth,
            predicted_object_pose_wxyz=predicted_pose,
        )

    completed_ok = False
    try:
        for step in range(1, int(args.num_steps) + 1):
            if viser_viewer is not None:
                while not viser_viewer.should_step():
                    if viser_viewer.consume_clear_stats_requested():
                        clear_eval_stats()
                    if viser_viewer.consume_restart_requested():
                        reset_all_envs()
                    update_live_viewer(
                        policy_depth=getattr(inner, "_student_image_policy_input", None),
                        predicted_pose=None,
                    )
                    time.sleep(max(float(args.viser_sleep_s), 1.0 / 30.0))

                if viser_viewer.consume_clear_stats_requested():
                    clear_eval_stats()
                if viser_viewer.consume_restart_requested():
                    reset_all_envs()

            last_step = step
            episode_lengths += 1
            with torch.no_grad():
                teacher_action = teacher.get_action(obs, is_deterministic=args.deterministic_teacher)
                if args.student_input == "teacher_obs":
                    student_out, next_hidden = student(_teacher_obs_tensor(obs), hidden)
                    image = None
                else:
                    student_obs = inner.get_student_obs()
                    image = student_obs["image"]
                    proprio = student_obs["proprio"]
                    student_out, next_hidden = student(image, proprio, hidden)
                student_action = student_out.action
                aux_loss, aux_info = _compute_aux_losses(inner, student_out.aux, args)
                action_loss = F.mse_loss(student_action, teacher_action, reduction="none").mean(dim=1)

            hidden = next_hidden.detach()
            action_for_env = teacher_action if args.policy_source == "teacher" else student_action.detach()

            if args.capture_viewer:
                viewer_frames = _capture_viewer_if_needed(
                    env=inner,
                    frames=viewer_frames,
                    output_dir=run_dir / "interactive_viewer",
                    capture_len=args.capture_viewer_len,
                    step=step,
                    predicted_object_pose_wxyz=aux_info["pred_pose_wxyz"],
                    wandb_run=wandb_run,
                )

            if args.student_input == "camera" and args.depth_rollout_video and depth_rollout_env_ids:
                frame = _capture_depth_rollout_frame(
                    env=inner,
                    policy_depth=image,
                    env_ids=depth_rollout_env_ids,
                    near=float(env_cfg.student_obs.depth_min_m),
                    far=float(env_cfg.student_obs.depth_max_m),
                )
                depth_rollout_video_frames = _log_depth_rollout_video(
                    frame=frame,
                    frames=depth_rollout_video_frames,
                    wandb_run=wandb_run,
                    step=step,
                    key="eval_depth_policy/realtime_rollout_video",
                    fps=args.depth_rollout_video_fps,
                    max_frames=args.depth_rollout_video_len,
                    interval=args.depth_rollout_video_interval,
                    local_output_dir=run_dir / "depth_rollout_videos",
                )

            if args.student_input == "camera" and args.depth_debug_interval > 0 and (
                step == 1 or step % args.depth_debug_interval == 0
            ):
                save_depth_debug(
                    output_dir=run_dir / "depth_debug",
                    step=step,
                    env_ids=depth_debug_env_ids or [0],
                    raw_depth=inner.student_camera.data.output["distance_to_image_plane"],
                    noisy_depth=getattr(inner, "_student_depth_noisy_m", None),
                    policy_full_depth=getattr(inner, "_student_depth_policy_full", None),
                    latest_policy_depth=getattr(inner, "_student_image_latest", None),
                    policy_depth=image,
                    delay_indices=getattr(inner, "_student_camera_delay_indices", None),
                    delay_queue_size=getattr(inner, "_student_camera_delay_queue_size", None),
                    near=float(env_cfg.student_obs.depth_min_m),
                    far=float(env_cfg.student_obs.depth_max_m),
                )

            current_goal_idx = float(inner._successes.float().mean().detach().cpu().item())
            current_completion = float(
                (inner._successes.float() / inner.env_max_goals.clamp_min(1).float()).mean().detach().cpu().item()
            )
            recent_goal_idx = interval_done_goal_idx / max(interval_done_count, 1)

            if viser_viewer is not None and step % max(1, args.viser_update_interval) == 0:
                update_live_viewer(policy_depth=image, predicted_pose=aux_info["pred_pose_wxyz"])
                if args.viser_sleep_s > 0.0:
                    time.sleep(float(args.viser_sleep_s))

            obs, _, dones, _ = teacher.env_step(wrapped, action_for_env)
            hidden = _reset_hidden_for_done(hidden, dones)

            done_mask = dones.reshape(-1).bool()
            if args.one_episode_per_env:
                record_done_mask = done_mask & ~first_episode_recorded
                record_dones = record_done_mask.reshape_as(dones)
            else:
                record_done_mask = done_mask
                record_dones = dones
            new_records = _episode_records_for_dones(
                env=inner,
                dones=record_dones,
                step=step,
                episode_lengths=episode_lengths,
                episode_context=episode_context,
                start_index=len(episode_records),
            )
            episode_records.extend(new_records)
            if args.one_episode_per_env and record_done_mask.any():
                first_episode_recorded[record_done_mask] = True
            if done_mask.any():
                episode_lengths[done_mask] = 0
                _refresh_episode_context(inner, episode_context, done_mask.nonzero(as_tuple=False).squeeze(-1))

            done_goal_values, done_completion_values, done_full_success_values = _done_success_values(inner, record_dones)
            done_count = int(done_goal_values.numel())
            if done_count:
                interval_done_goal_idx += float(done_goal_values.sum().item())
                interval_done_completion += float(done_completion_values.sum().item())
                interval_done_full_success += float(done_full_success_values.sum().item())
                interval_done_count += done_count
                rolling_reset_window.extend(
                    (
                        float(goal_idx),
                        float(completion),
                        float(full_success),
                    )
                    for goal_idx, completion, full_success in zip(
                        done_goal_values.tolist(),
                        done_completion_values.tolist(),
                        done_full_success_values.tolist(),
                    )
                )

            interval_action_loss += float(action_loss.mean().detach().cpu().item())
            interval_aux_loss += float(aux_loss.mean().detach().cpu().item())
            interval_aux_pos_loss += float(aux_info["pos_loss"].mean().detach().cpu().item())
            interval_aux_keypoint_loss += float(aux_info["keypoint_loss"].mean().detach().cpu().item())
            interval_step_count += 1

            should_log = step == 1 or step % args.log_interval == 0
            if should_log:
                elapsed = max(time.perf_counter() - interval_start, 1e-6)
                current_goal_idx = float(inner._successes.float().mean().detach().cpu().item())
                current_completion = float(
                    (inner._successes.float() / inner.env_max_goals.clamp_min(1).float()).mean().detach().cpu().item()
                )
                recent_goal_idx = interval_done_goal_idx / max(interval_done_count, 1)
                recent_completion = interval_done_completion / max(interval_done_count, 1)
                recent_full_success = interval_done_full_success / max(interval_done_count, 1)
                rolling_count = len(rolling_reset_window)
                if rolling_count:
                    rolling_goal_idx = sum(item[0] for item in rolling_reset_window) / rolling_count
                    rolling_completion = sum(item[1] for item in rolling_reset_window) / rolling_count
                    rolling_full_success = sum(item[2] for item in rolling_reset_window) / rolling_count
                else:
                    rolling_goal_idx = 0.0
                    rolling_completion = 0.0
                    rolling_full_success = 0.0
                completed_summary = _summary(episode_records, current_goal_idx, current_completion)
                row = {
                    "step": step,
                    "mode": "eval",
                    "policy_source": args.policy_source,
                    "action_loss": interval_action_loss / max(interval_step_count, 1),
                    "action_rmse": math.sqrt(max(interval_action_loss / max(interval_step_count, 1), 0.0)),
                    "aux_loss": interval_aux_loss / max(interval_step_count, 1),
                    "aux_object_pos_loss": interval_aux_pos_loss / max(interval_step_count, 1),
                    "aux_object_pos_rmse_m": math.sqrt(max(interval_aux_pos_loss / max(interval_step_count, 1), 0.0)),
                    "aux_object_keypoint_loss": interval_aux_keypoint_loss / max(interval_step_count, 1),
                    "aux_object_keypoint_rmse_m": math.sqrt(
                        max(interval_aux_keypoint_loss / max(interval_step_count, 1), 0.0)
                    ),
                    "current_goal_idx_avg": current_goal_idx,
                    "current_goal_completion_ratio_avg": current_completion,
                    "recent_reset_goal_idx_avg": recent_goal_idx,
                    "recent_reset_goal_completion_ratio_avg": recent_completion,
                    "recent_reset_full_success_rate": recent_full_success,
                    "recent_reset_count": interval_done_count,
                    "rolling_reset_goal_idx_avg": rolling_goal_idx,
                    "rolling_reset_goal_completion_ratio_avg": rolling_completion,
                    "rolling_reset_full_success_rate": rolling_full_success,
                    "rolling_reset_count": rolling_count,
                    "rolling_reset_window_size": int(args.rolling_reset_window_size),
                    "completed_episode_count": completed_summary["completed_episode_count"],
                    "completed_goal_idx_avg": completed_summary["completed_goal_idx_avg"],
                    "completed_goal_completion_ratio_avg": completed_summary[
                        "completed_goal_completion_ratio_avg"
                    ],
                    "completed_full_success_rate": completed_summary["completed_full_success_rate"],
                    "env_steps_per_s": inner.num_envs * interval_step_count / elapsed,
                }
                print(
                    "[eval_depth_policy] "
                    f"step={step} completed={len(episode_records)} "
                    f"current_goal_idx={current_goal_idx:.3f} recent_reset_goal_idx={recent_goal_idx:.3f} "
                    f"rolling_reset_goal_idx={rolling_goal_idx:.3f} "
                    f"completed_goal_idx={row['completed_goal_idx_avg']:.3f} "
                    f"full_success={row['completed_full_success_rate']:.3f} "
                    f"action_rmse={row['action_rmse']:.4f}",
                    flush=True,
                )
                _log_metrics(run_dir, row, wandb_run=wandb_run, step=step)
                interval_action_loss = 0.0
                interval_aux_loss = 0.0
                interval_aux_pos_loss = 0.0
                interval_aux_keypoint_loss = 0.0
                interval_step_count = 0
                interval_done_goal_idx = 0.0
                interval_done_completion = 0.0
                interval_done_full_success = 0.0
                interval_done_count = 0
                interval_start = time.perf_counter()

            if args.num_completed_episodes > 0 and len(episode_records) >= args.num_completed_episodes:
                print(
                    f"[eval_depth_policy] reached num_completed_episodes={args.num_completed_episodes}",
                    flush=True,
                )
                break
            if args.one_episode_per_env and bool(first_episode_recorded.all().item()):
                print("[eval_depth_policy] recorded one episode for every env", flush=True)
                break
        completed_ok = True

    finally:
        if viewer_frames:
            _capture_viewer_if_needed(
                env=inner,
                frames=viewer_frames,
                output_dir=run_dir / "interactive_viewer",
                capture_len=len(viewer_frames),
                step=max(1, last_step),
                append_frame=False,
                wandb_run=wandb_run,
            )
        if depth_rollout_video_frames:
            _log_depth_rollout_video(
                frame=None,
                frames=depth_rollout_video_frames,
                wandb_run=wandb_run,
                step=max(1, last_step),
                key="eval_depth_policy/realtime_rollout_video",
                fps=args.depth_rollout_video_fps,
                max_frames=args.depth_rollout_video_len,
                interval=args.depth_rollout_video_interval,
                local_output_dir=run_dir / "depth_rollout_videos",
                force_log=True,
            )
        current_goal_idx = float(inner._successes.float().mean().detach().cpu().item())
        current_completion = float(
            (inner._successes.float() / inner.env_max_goals.clamp_min(1).float()).mean().detach().cpu().item()
        )
        summary = _summary(episode_records, current_goal_idx, current_completion)
        summary.update(
            {
                "student_checkpoint": str(args.student_checkpoint),
                "policy_source": args.policy_source,
                "num_envs": int(args.num_envs),
                "num_steps_requested": int(args.num_steps),
                "one_episode_per_env": bool(args.one_episode_per_env),
                "depth_noise_profile": env_cfg.student_obs.depth_noise_profile,
                "depth_noise_strength": float(env_cfg.student_obs.depth_noise_strength),
                "student_camera_preset": args.student_camera_preset,
                "camera_pose_randomization_profile": env_cfg.student_obs.camera_pose_randomization_profile,
                "camera_pose_randomization_mode": env_cfg.student_obs.camera_pose_randomization_mode,
                "camera_pos_noise_m": list(env_cfg.student_obs.camera_pos_noise_m),
                "camera_rot_noise_deg": list(env_cfg.student_obs.camera_rot_noise_deg),
                "student_image_delay_queue_size": int(env_cfg.student_obs.camera_delay_max)
                if bool(env_cfg.student_obs.use_camera_delay)
                else 1,
                "peg_urdf": str(env_cfg.assets.peg_urdf),
                "peg_goal_mode": str(env_cfg.peg_in_hole.goal_mode),
                "peg_enable_retract": bool(env_cfg.peg_in_hole.enable_retract),
            }
        )
        _write_episode_csv(run_dir / "episodes.csv", episode_records)
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[eval_depth_policy] wrote episodes: {run_dir / 'episodes.csv'}", flush=True)
        print(f"[eval_depth_policy] wrote summary: {run_dir / 'summary.json'}", flush=True)
        print("[eval_depth_policy] summary:\n" + json.dumps(summary, indent=2, sort_keys=True), flush=True)
        env.close()
        if wandb_run is not None:
            wandb_run.finish()
        del app
        sys.stdout.flush()
        sys.stderr.flush()
        if completed_ok and args.force_exit:
            os._exit(0)


if __name__ == "__main__":
    main()
