#!/usr/bin/env python
"""IsaacSim-backed ROS source for fake-real depth deployment debugging.

This node replaces ``fake_robot_node.py`` and ``fake_depth_image_node.py`` with
one Isaac Lab simulation process:

* subscribes to ``/iiwa/joint_cmd`` and ``/sharpa/joint_cmd``;
* drives the IsaacSim robot with those joint-position targets;
* publishes simulated joint states and object pose;
* optionally renders/publishes the student depth camera every N control steps;
* can run in benchmark mode without ROS to measure physics/render throughput.

It intentionally uses the existing SimToolReal/PegInHoleDepthStudent env rather
than inventing a second simulator path.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_TEACHER_DIR = Path("/juno/u/kedia/depthbasedRL/train_dir/Apr28/isaacSim_PegInHole")
DEPTH_TOPIC = "/zed/zed_node/depth/depth_registered"
RGB_TOPIC = "/zed/zed_node/rgb/image_rect_color"
CAMERA_INFO_TOPIC = "/zed/zed_node/rgb/camera_info"
IIWA_JOINT_STATE_TOPIC = "/iiwa/joint_states"
SHARPA_JOINT_STATE_TOPIC = "/sharpa/joint_states"
IIWA_JOINT_CMD_TOPIC = "/iiwa/joint_cmd"
SHARPA_JOINT_CMD_TOPIC = "/sharpa/joint_cmd"
OBJECT_POSE_TOPIC = "/robot_frame/current_object_pose"
ISAAC_GT_OBJECT_POSE_TOPIC = "/robot_frame/isaac_gt_object_pose"
SIM_WORLD_T_ROBOT_POS_M = np.array([0.0, 0.8, 0.0], dtype=np.float64)
N_ARM = 7
N_HAND = 22
N_ACTIONS = 29
IIWA_JOINT_NAMES = [f"iiwa_joint_{idx}" for idx in range(1, N_ARM + 1)]
SHARPA_JOINT_NAMES = [f"joint_{idx}.0" for idx in range(N_HAND)]
SIM_DEFAULT_ARM_Q = np.asarray([-1.571, 1.571, 0.0, 1.376, 0.0, 1.485, 1.308], dtype=np.float32)
DEPLOYMENT_HOME_ARM_Q = np.asarray(
    [-1.571, 1.571 - np.deg2rad(10.0), 0.0, 1.376 + np.deg2rad(10.0), 0.0, 1.485, 1.308],
    dtype=np.float32,
)
ZERO_HAND_Q = np.zeros(N_HAND, dtype=np.float32)
DEPLOYMENT_EPISODE_LENGTH_S = 1.0e6


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


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
    fy = fx
    return np.asarray(
        [[fx, 0.0, (width - 1.0) * 0.5], [0.0, fy, (height - 1.0) * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _centered_camera_k(k: np.ndarray, *, width: int, height: int) -> np.ndarray:
    """Use requested focal lengths but centered principal point.

    Isaac/Omniverse currently warns that aperture offsets are not supported for
    these camera render products, so this mode keeps ROS CameraInfo consistent
    with what is actually rendered when a non-centered K was requested.
    """
    centered = np.asarray(k, dtype=np.float32).copy()
    centered[0, 2] = (float(width) - 1.0) * 0.5
    centered[1, 2] = (float(height) - 1.0) * 0.5
    return centered


def _validate_camera_k_for_image(k: np.ndarray, *, width: int, height: int, source: str) -> None:
    cx, cy = float(k[0, 2]), float(k[1, 2])
    if not (0.0 <= cx <= float(width) and 0.0 <= cy <= float(height)):
        raise ValueError(
            f"{source} has principal point ({cx:.3f}, {cy:.3f}) outside image size "
            f"{width}x{height}. Use matching --camera_image_width/height or a scaled K."
        )


def _default_object_pose_wxyz(env_cfg) -> tuple[float, float, float, float, float, float, float]:
    """Default generic SimToolReal object pose in env-local IsaacSim coordinates."""
    return (
        0.0,
        0.0,
        float(env_cfg.reset.table_reset_z + env_cfg.reset.table_object_z_offset),
        1.0,
        0.0,
        0.0,
        0.0,
    )


def _zero_training_randomization(env_cfg) -> None:
    """Turn off reset/DR knobs that make the sim differ from a real continuous robot."""
    reset = env_cfg.reset
    reset.reset_dof_pos_random_interval_arm = 0.0
    reset.reset_dof_pos_random_interval_fingers = 0.0
    reset.reset_dof_vel_random_interval = 0.0
    reset.reset_position_noise_x = 0.0
    reset.reset_position_noise_y = 0.0
    reset.reset_position_noise_z = 0.0
    reset.table_reset_z_range = 0.0

    dr = env_cfg.domain_randomization
    dr.use_obs_delay = False
    dr.use_action_delay = False
    dr.use_object_state_delay_noise = False
    dr.object_state_xyz_noise_std = 0.0
    dr.object_state_rotation_noise_degrees = 0.0
    dr.joint_velocity_obs_noise_std = 0.0
    dr.force_scale = 0.0
    dr.torque_scale = 0.0
    dr.force_prob_range = (1.0e-12, 1.0e-12)
    dr.torque_prob_range = (1.0e-12, 1.0e-12)
    dr.object_scale_noise_multiplier_range = (1.0, 1.0)

    student_obs = env_cfg.student_obs
    student_obs.use_camera_delay = False
    student_obs.camera_delay_max = 0
    student_obs.use_student_obs_delay = False
    student_obs.student_obs_delay_max = 0
    student_obs.camera_pose_randomization_profile = "off"
    student_obs.camera_pose_randomization_mode = "startup"


def _apply_object_init_cfg(args: argparse.Namespace, env_cfg) -> tuple[float, ...] | None:
    """Configure deterministic or randomized object init.

    Returns a resolved pose for node-side manual writing when a fixed pose is
    requested. Peg-in-hole has its own scene-file reset path, so a fixed pose
    must be written after the env reset.
    """
    mode = str(args.object_init_mode).lower()
    if mode == "env_reset":
        return None

    has_peg_cfg = hasattr(env_cfg, "peg_in_hole")
    if has_peg_cfg:
        pih = env_cfg.peg_in_hole
        if mode == "default":
            pih.object_init_position_noise_xy = (0.0, 0.0)
            pih.object_init_position_noise_z = 0.0
            pih.object_init_orientation_mode = "scene"
        elif mode == "randomized":
            noise = tuple(float(v) for v in args.object_init_position_noise_m)
            pih.object_init_position_noise_xy = (noise[0], noise[1])
            pih.object_init_position_noise_z = noise[2]
            if args.object_init_orientation_mode is not None:
                pih.object_init_orientation_mode = args.object_init_orientation_mode
            elif args.object_init_yaw_noise_deg > 0.0:
                pih.object_init_orientation_mode = "yaw_only"
                pih.object_init_yaw_range_degrees = float(args.object_init_yaw_noise_deg)
        elif mode == "fixed":
            if args.object_init_pose_wxyz is None:
                raise ValueError(
                    "--object_init_mode fixed requires --object_init_pose_wxyz x y z qw qx qy qz"
                )
            return tuple(float(v) for v in args.object_init_pose_wxyz)
        else:
            raise ValueError(f"Unsupported --object_init_mode={args.object_init_mode!r}")
        return None

    if mode == "default":
        env_cfg.reset.fixed_start_pose = _default_object_pose_wxyz(env_cfg)
    elif mode == "randomized":
        noise = tuple(float(v) for v in args.object_init_position_noise_m)
        env_cfg.reset.fixed_start_pose = None
        env_cfg.reset.reset_position_noise_x = noise[0]
        env_cfg.reset.reset_position_noise_y = noise[1]
        env_cfg.reset.reset_position_noise_z = noise[2]
        if args.object_init_orientation_mode is not None:
            env_cfg.reset.object_orientation_mode = args.object_init_orientation_mode
        elif args.object_init_yaw_noise_deg > 0.0:
            env_cfg.reset.object_orientation_mode = "yaw_only"
            env_cfg.reset.object_yaw_range_degrees = float(args.object_init_yaw_noise_deg)
    elif mode == "fixed":
        if args.object_init_pose_wxyz is None:
            raise ValueError(
                "--object_init_mode fixed requires --object_init_pose_wxyz x y z qw qx qy qz"
            )
        env_cfg.reset.fixed_start_pose = tuple(float(v) for v in args.object_init_pose_wxyz)
    else:
        raise ValueError(f"Unsupported --object_init_mode={args.object_init_mode!r}")
    return None


def _install_no_reset_done_wrapper(inner) -> None:
    """Keep reward/success bookkeeping, but prevent DirectRLEnv auto-resets."""
    original_get_dones = inner._get_dones

    def _get_dones_no_reset():
        original_get_dones()
        zeros = torch.zeros(inner.num_envs, dtype=torch.bool, device=inner.device)
        inner._termination_reasons = getattr(inner, "_termination_reasons", {})
        inner._termination_reasons["deployment_no_reset"] = torch.ones_like(zeros)
        return zeros, zeros

    inner._deployment_original_get_dones = original_get_dones
    inner._get_dones = _get_dones_no_reset


def _fmt_stats(values: deque[float]) -> str:
    if not values:
        return "n/a"
    arr = np.asarray(values, dtype=np.float64)
    return f"mean/med/p95={arr.mean():.2f}/{np.median(arr):.2f}/{np.quantile(arr, 0.95):.2f}ms"


def _make_depth_msg(depth_m: np.ndarray, rospy, Image, *, frame_id: str):
    depth_m = np.ascontiguousarray(depth_m.astype(np.float32, copy=False))
    msg = Image()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = frame_id
    msg.height = int(depth_m.shape[0])
    msg.width = int(depth_m.shape[1])
    msg.encoding = "32FC1"
    msg.is_bigendian = 0
    msg.step = int(depth_m.shape[1] * np.dtype(np.float32).itemsize)
    msg.data = depth_m.tobytes()
    return msg


def _make_rgb_msg(rgb: np.ndarray, rospy, Image, *, frame_id: str):
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"Expected RGB image shape (H, W, >=3), got {rgb.shape}")
    rgb = np.ascontiguousarray(rgb[..., :3].astype(np.uint8, copy=False))
    msg = Image()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = frame_id
    msg.height = int(rgb.shape[0])
    msg.width = int(rgb.shape[1])
    msg.encoding = "rgb8"
    msg.is_bigendian = 0
    msg.step = int(rgb.shape[1] * 3)
    msg.data = rgb.tobytes()
    return msg


def _make_camera_info_msg(k: np.ndarray, height: int, width: int, rospy, CameraInfo, *, frame_id: str, stamp):
    msg = CameraInfo()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = int(height)
    msg.width = int(width)
    fx, fy, cx, cy = float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])
    msg.K = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    msg.P = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    msg.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    msg.distortion_model = "plumb_bob"
    msg.D = [0.0, 0.0, 0.0, 0.0, 0.0]
    return msg


@dataclass
class RosModules:
    rospy: Any
    JointState: Any
    Image: Any
    CameraInfo: Any
    PoseStamped: Any


class IsaacDepthEnvNode:
    def __init__(self, args: argparse.Namespace, env, inner, ros: RosModules | None) -> None:
        self.args = args
        self.env = env
        self.inner = inner
        self.ros = ros
        self.device = inner.device
        self.zero_action = torch.zeros((inner.num_envs, int(inner.cfg.action_space)), device=self.device)
        self.latest_target_canon: np.ndarray | None = None
        self.step_idx = 0
        self.depth_publish_count = 0
        self.physics_ms = deque(maxlen=512)
        self.camera_ms = deque(maxlen=512)
        self.loop_ms = deque(maxlen=512)
        self.last_status_time = time.time()
        self.camera_k = None
        if args.enable_depth or args.enable_rgb:
            self.camera_k = _student_camera_k(inner.cfg.student_obs)
            if args.camera_info_mode == "centered":
                self.camera_k = _centered_camera_k(
                    self.camera_k,
                    width=int(inner.cfg.student_obs.image_width),
                    height=int(inner.cfg.student_obs.image_height),
                )

        self.env.reset()
        if getattr(args, "resolved_object_init_pose_wxyz", None) is not None:
            self._write_object_pose_local_wxyz(args.resolved_object_init_pose_wxyz)
        q_canon = self._initial_pose_canon()
        if q_canon is not None:
            self._write_joint_state_canon(q_canon)
        else:
            q_canon, _ = self._joint_state_canon()
        self.latest_target_canon = q_canon.copy()
        self._write_replay_target(q_canon)

        self.iiwa_state_pub = None
        self.sharpa_state_pub = None
        self.depth_pub = None
        self.rgb_pub = None
        self.camera_info_pub = None
        self.object_pose_pub = None
        self.gt_object_pose_pub = None
        if ros is not None:
            ros.rospy.init_node("isaac_depth_env_node", anonymous=True)
            ros.rospy.Subscriber(args.iiwa_joint_cmd_topic, ros.JointState, self._iiwa_cmd_callback, queue_size=1)
            ros.rospy.Subscriber(args.sharpa_joint_cmd_topic, ros.JointState, self._sharpa_cmd_callback, queue_size=1)
            self.iiwa_state_pub = ros.rospy.Publisher(args.iiwa_joint_state_topic, ros.JointState, queue_size=1)
            self.sharpa_state_pub = ros.rospy.Publisher(args.sharpa_joint_state_topic, ros.JointState, queue_size=1)
            if args.publish_object_pose:
                self.object_pose_pub = ros.rospy.Publisher(args.object_pose_topic, ros.PoseStamped, queue_size=1)
            if args.publish_gt_object_pose_debug:
                self.gt_object_pose_pub = ros.rospy.Publisher(
                    args.gt_object_pose_topic, ros.PoseStamped, queue_size=1
                )
            if args.enable_depth and args.publish_depth:
                self.depth_pub = ros.rospy.Publisher(args.depth_topic, ros.Image, queue_size=1)
            if args.enable_rgb and args.publish_rgb:
                self.rgb_pub = ros.rospy.Publisher(args.rgb_topic, ros.Image, queue_size=1)
            if (args.enable_depth and args.publish_depth) or (args.enable_rgb and args.publish_rgb):
                self.camera_info_pub = ros.rospy.Publisher(args.camera_info_topic, ros.CameraInfo, queue_size=1)

    def _iiwa_cmd_callback(self, msg) -> None:
        target = self._target_or_current()
        values = np.asarray(msg.position, dtype=np.float32)
        if values.shape == (N_ARM,):
            target[:N_ARM] = values
            self.latest_target_canon = target

    def _sharpa_cmd_callback(self, msg) -> None:
        target = self._target_or_current()
        values = np.asarray(msg.position, dtype=np.float32)
        if values.shape == (N_HAND,):
            target[N_ARM:] = values
            self.latest_target_canon = target

    def _target_or_current(self) -> np.ndarray:
        if self.latest_target_canon is None:
            q, _ = self._joint_state_canon()
            return q.copy()
        return self.latest_target_canon.copy()

    def _joint_state_canon(self) -> tuple[np.ndarray, np.ndarray]:
        perm = self.inner._perm_lab_to_canon
        q = _to_numpy(self.inner.robot.data.joint_pos[0, perm]).astype(np.float32)
        qd = _to_numpy(self.inner.robot.data.joint_vel[0, perm]).astype(np.float32)
        return q, qd

    def _initial_pose_canon(self) -> np.ndarray | None:
        mode = str(self.args.initial_robot_pose).lower()
        if mode == "env_reset":
            return None
        if mode == "sim_default":
            return np.concatenate([SIM_DEFAULT_ARM_Q, ZERO_HAND_Q]).astype(np.float32)
        if mode == "deployment_home":
            return np.concatenate([DEPLOYMENT_HOME_ARM_Q, ZERO_HAND_Q]).astype(np.float32)
        raise ValueError(f"Unsupported --initial_robot_pose={self.args.initial_robot_pose!r}")

    def _write_joint_state_canon(self, q_canon: np.ndarray) -> None:
        q_canon = np.asarray(q_canon, dtype=np.float32)
        if q_canon.shape != (N_ACTIONS,):
            raise ValueError(f"Expected canonical q shape {(N_ACTIONS,)}, got {q_canon.shape}")
        q_lab = torch.as_tensor(q_canon, device=self.device, dtype=torch.float32).view(1, -1)[:, self.inner._perm_canon_to_lab]
        q_lab = q_lab.expand(self.inner.num_envs, -1).clone()
        qd_lab = torch.zeros_like(q_lab)
        env_ids = torch.arange(self.inner.num_envs, device=self.device, dtype=torch.long)
        self.inner.robot.write_joint_state_to_sim(q_lab, qd_lab, env_ids=env_ids)
        self.inner._prev_targets[:] = q_lab
        self.inner._cur_targets[:] = q_lab

    def _write_object_pose_local_wxyz(self, pose_wxyz: tuple[float, ...]) -> None:
        pose_wxyz_np = np.asarray(pose_wxyz, dtype=np.float32)
        if pose_wxyz_np.shape != (7,):
            raise ValueError(f"Expected object pose shape (7,), got {pose_wxyz_np.shape}")
        env_ids = torch.arange(self.inner.num_envs, device=self.device, dtype=torch.long)
        pose = torch.as_tensor(pose_wxyz_np, device=self.device, dtype=torch.float32).view(1, 7)
        pose = pose.expand(self.inner.num_envs, -1).clone()
        pose[:, :3] += self.inner.scene.env_origins[env_ids]
        self.inner.object.write_root_pose_to_sim(pose, env_ids=env_ids)
        self.inner.object.write_root_velocity_to_sim(
            torch.zeros(self.inner.num_envs, 6, device=self.device), env_ids=env_ids
        )
        self.inner._object_init_z[:] = pose_wxyz_np[2]

    def _write_replay_target(self, target_canon: np.ndarray) -> None:
        target = torch.as_tensor(target_canon, device=self.device, dtype=torch.float32).view(1, -1)
        target_lab = target[:, self.inner._perm_canon_to_lab]
        self.inner._replay_target_lab_order = target_lab.expand(self.inner.num_envs, -1).clone()

    def _publish_joint_states(self) -> None:
        if self.ros is None:
            return
        ros = self.ros
        stamp = ros.rospy.Time.now()
        q, qd = self._joint_state_canon()

        iiwa_msg = ros.JointState()
        iiwa_msg.header.stamp = stamp
        iiwa_msg.name = IIWA_JOINT_NAMES
        iiwa_msg.position = q[:N_ARM].tolist()
        iiwa_msg.velocity = qd[:N_ARM].tolist()
        self.iiwa_state_pub.publish(iiwa_msg)

        sharpa_msg = ros.JointState()
        sharpa_msg.header.stamp = stamp
        sharpa_msg.name = SHARPA_JOINT_NAMES
        sharpa_msg.position = q[N_ARM:].tolist()
        sharpa_msg.velocity = qd[N_ARM:].tolist()
        self.sharpa_state_pub.publish(sharpa_msg)

    def _make_object_pose_msg(self):
        from scipy.spatial.transform import Rotation as R

        ros = self.ros
        origin = self.inner.scene.env_origins[0]
        object_pos_env = _to_numpy(self.inner.object.data.root_pos_w[0] - origin).astype(np.float64)
        object_pos_robot = object_pos_env - SIM_WORLD_T_ROBOT_POS_M
        quat_wxyz = _to_numpy(self.inner.object.data.root_quat_w[0]).astype(np.float64)
        quat_xyzw = quat_wxyz[[1, 2, 3, 0]]
        # Translation-only robot frame offset, so orientation is unchanged.
        quat_xyzw = R.from_quat(quat_xyzw).as_quat()

        msg = ros.PoseStamped()
        msg.header.stamp = ros.rospy.Time.now()
        msg.header.frame_id = "robot_frame"
        msg.pose.position.x = float(object_pos_robot[0])
        msg.pose.position.y = float(object_pos_robot[1])
        msg.pose.position.z = float(object_pos_robot[2])
        msg.pose.orientation.x = float(quat_xyzw[0])
        msg.pose.orientation.y = float(quat_xyzw[1])
        msg.pose.orientation.z = float(quat_xyzw[2])
        msg.pose.orientation.w = float(quat_xyzw[3])
        return msg

    def _publish_object_pose(self) -> None:
        if self.ros is None or (self.object_pose_pub is None and self.gt_object_pose_pub is None):
            return
        msg = self._make_object_pose_msg()
        if self.object_pose_pub is not None:
            self.object_pose_pub.publish(msg)
        if self.gt_object_pose_pub is not None:
            self.gt_object_pose_pub.publish(msg)

    def _render_depth(self) -> np.ndarray:
        if not self.args.enable_depth:
            raise RuntimeError("Depth rendering requested with --no-enable_depth")
        source = str(self.args.published_depth_source).lower()
        if source == "raw":
            depth = self.inner._student_depth_raw_m[0, 0]
        elif source == "noisy":
            depth = self.inner._student_depth_noisy_m[0, 0]
        else:
            raise ValueError(f"Unsupported --published_depth_source={self.args.published_depth_source!r}")
        return _to_numpy(depth).astype(np.float32)

    def _render_rgb(self) -> np.ndarray:
        if not self.args.enable_rgb:
            raise RuntimeError("RGB rendering requested with --no-enable_rgb")
        rgb = self.inner.student_camera.data.output.get("rgb")
        if rgb is None:
            raise RuntimeError("Student camera has no RGB output. Check image_modality='rgb' or 'rgbd'.")
        return _to_numpy(rgb[0, ..., :3]).astype(np.uint8)

    def _publish_camera_if_due(self) -> None:
        if not (self.args.enable_depth or self.args.enable_rgb):
            return
        if self.args.depth_publish_every_n <= 0:
            return
        if self.step_idx % self.args.depth_publish_every_n != 0:
            return
        t0 = time.perf_counter()
        from isaacsimenvs.tasks.simtoolreal.utils.scene_utils import read_student_camera_image

        read_student_camera_image(self.inner)
        depth_m = self._render_depth() if self.args.enable_depth else None
        rgb = self._render_rgb() if self.args.enable_rgb else None
        self.camera_ms.append(1000.0 * (time.perf_counter() - t0))
        self.depth_publish_count += 1

        if self.ros is None:
            return
        ros = self.ros
        stamp = None
        height = width = None
        if depth_m is not None and self.depth_pub is not None and self.args.publish_depth:
            msg = _make_depth_msg(depth_m, ros.rospy, ros.Image, frame_id=self.args.depth_frame_id)
            self.depth_pub.publish(msg)
            stamp = msg.header.stamp
            height, width = depth_m.shape[:2]
        if rgb is not None and self.rgb_pub is not None and self.args.publish_rgb:
            msg = _make_rgb_msg(rgb, ros.rospy, ros.Image, frame_id=self.args.depth_frame_id)
            self.rgb_pub.publish(msg)
            stamp = msg.header.stamp if stamp is None else stamp
            height, width = rgb.shape[:2]
        if self.camera_info_pub is not None and self.camera_k is not None and stamp is not None:
            info_msg = _make_camera_info_msg(
                self.camera_k,
                height,
                width,
                ros.rospy,
                ros.CameraInfo,
                frame_id=self.args.depth_frame_id,
                stamp=stamp,
            )
            self.camera_info_pub.publish(info_msg)

    def _print_status(self, *, force: bool = False) -> None:
        now = time.time()
        if not force and now - self.last_status_time < self.args.status_interval_s:
            return
        self.last_status_time = now
        loop_hz = 1000.0 / np.median(np.asarray(self.loop_ms)) if self.loop_ms else float("nan")
        depth_hz = self.depth_publish_count / max(now - self.start_time, 1e-6)
        q, _ = self._joint_state_canon()
        print(
            "[isaac_depth_env_node] "
            f"step={self.step_idx} loop_hz_med={loop_hz:.1f} camera_pub_hz_avg={depth_hz:.1f} "
            f"physics_ms={_fmt_stats(self.physics_ms)} camera_ms={_fmt_stats(self.camera_ms)} "
            f"loop_ms={_fmt_stats(self.loop_ms)} q_abs_max={float(np.abs(q).max()):.3f}",
            flush=True,
        )

    def step(self) -> None:
        t_loop = time.perf_counter()
        target = self._target_or_current()
        self._write_replay_target(target)
        t0 = time.perf_counter()
        self.env.step(self.zero_action)
        self.physics_ms.append(1000.0 * (time.perf_counter() - t0))
        self._publish_joint_states()
        self._publish_object_pose()
        self._publish_camera_if_due()
        self.loop_ms.append(1000.0 * (time.perf_counter() - t_loop))
        self.step_idx += 1
        if self.args.benchmark_warmup_steps > 0 and self.step_idx == self.args.benchmark_warmup_steps:
            self.physics_ms.clear()
            self.camera_ms.clear()
            self.loop_ms.clear()
            self.depth_publish_count = 0
            self.start_time = time.time()
            print(
                f"[isaac_depth_env_node] cleared timing stats after "
                f"{self.args.benchmark_warmup_steps} warmup steps",
                flush=True,
            )
        self._print_status()

    def run(self) -> None:
        self.start_time = time.time()
        rate = self.ros.rospy.Rate(self.args.control_hz) if self.ros is not None and self.args.realtime else None
        while True:
            if self.ros is not None and self.ros.rospy.is_shutdown():
                break
            if self.args.num_steps >= 0 and self.step_idx >= self.args.num_steps:
                break
            if self.args.run_duration_s >= 0.0 and time.time() - self.start_time >= self.args.run_duration_s:
                break
            self.step()
            if rate is not None:
                rate.sleep()
        self._print_status(force=True)


def _import_ros() -> RosModules:
    import rospy
    from geometry_msgs.msg import PoseStamped
    from sensor_msgs.msg import CameraInfo, Image, JointState

    return RosModules(
        rospy=rospy,
        JointState=JointState,
        Image=Image,
        CameraInfo=CameraInfo,
        PoseStamped=PoseStamped,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Isaacsimenvs-PegInHoleDepthStudent-Direct-v0")
    parser.add_argument("--teacher_config", type=Path, default=DEFAULT_TEACHER_DIR / "config.yaml")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--sim_device", default="cuda:0")
    parser.add_argument("--control_hz", type=float, default=60.0)
    parser.add_argument("--run_duration_s", type=float, default=-1.0)
    parser.add_argument("--num_steps", type=int, default=-1)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark_warmup_steps", type=int, default=0)
    parser.add_argument("--realtime", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable_depth", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--publish_depth", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--enable_rgb",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable RGB rendering from the same student camera as depth.",
    )
    parser.add_argument(
        "--publish_rgb",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Publish RGB Image messages when --enable_rgb is set.",
    )
    parser.add_argument("--depth_publish_every_n", type=int, default=1)
    parser.add_argument("--student_camera_preset", default="default")
    parser.add_argument(
        "--camera_image_width",
        type=int,
        default=None,
        help="Override rendered camera image width for deployment/debug capture.",
    )
    parser.add_argument(
        "--camera_image_height",
        type=int,
        default=None,
        help="Override rendered camera image height for deployment/debug capture.",
    )
    parser.add_argument(
        "--camera_k_file",
        type=Path,
        default=None,
        help="Optional 3x3 pinhole intrinsics text file matching the rendered image size.",
    )
    parser.add_argument(
        "--camera_intrinsic_matrix",
        type=float,
        nargs=9,
        default=None,
        help="Optional row-major 3x3 pinhole intrinsics matching the rendered image size.",
    )
    parser.add_argument(
        "--camera_info_mode",
        choices=("requested", "centered"),
        default="requested",
        help=(
            "requested publishes the configured K. centered keeps fx/fy but centers cx/cy, "
            "matching Omniverse behavior when aperture offsets are ignored."
        ),
    )
    parser.add_argument("--depth_noise_profile", default="off")
    parser.add_argument("--depth_noise_strength", type=float, default=None)
    parser.add_argument(
        "--published_depth_source",
        choices=("raw", "noisy"),
        default="raw",
        help="Which metric depth image to publish on ROS. Use noisy to test the same pre-window metric noise used in training.",
    )
    parser.add_argument("--camera_pose_randomization_profile", default=None)
    parser.add_argument("--camera_pose_randomization_mode", default=None)
    parser.add_argument("--camera_pos_noise_m", type=float, nargs=3, default=None)
    parser.add_argument("--camera_rot_noise_deg", type=float, nargs=3, default=None)
    parser.add_argument("--peg_urdf", default=None)
    parser.add_argument("--peg_goal_mode", default=None)
    parser.add_argument(
        "--hide_hole_fixture",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Peg-in-hole debug mode: keep the wooden table but strip the grey "
            "hole fixture visual/collision boxes from scene URDFs before USD conversion."
        ),
    )
    parser.add_argument("--object_init_orientation_mode", default=None)
    parser.add_argument(
        "--deployment_mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply fake-real deployment defaults: deterministic resets, no training DR, and no auto-reset while stepping.",
    )
    parser.add_argument(
        "--disable_env_resets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prevent timeout/fall/success-driven automatic env resets after the initial construction reset.",
    )
    parser.add_argument(
        "--zero_training_randomization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Zero robot/object/table reset noise and dynamics/observation domain randomization.",
    )
    parser.add_argument(
        "--object_init_mode",
        choices=("default", "fixed", "randomized", "env_reset"),
        default="default",
        help=(
            "Object init source. default uses the deterministic task default/scene pose; fixed uses "
            "--object_init_pose_wxyz; randomized applies the provided init noise once at startup because resets are off."
        ),
    )
    parser.add_argument(
        "--object_init_pose_wxyz",
        type=float,
        nargs=7,
        default=None,
        metavar=("X", "Y", "Z", "QW", "QX", "QY", "QZ"),
        help="Fixed env-local IsaacSim object pose. This is not robot_frame; pose is before adding scene.env_origins.",
    )
    parser.add_argument(
        "--object_init_position_noise_m",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="Half-width position noise for --object_init_mode randomized.",
    )
    parser.add_argument(
        "--object_init_yaw_noise_deg",
        type=float,
        default=0.0,
        help="Yaw-only orientation noise for --object_init_mode randomized when no explicit orientation mode is provided.",
    )
    parser.add_argument(
        "--initial_robot_pose",
        choices=("deployment_home", "sim_default", "env_reset"),
        default="deployment_home",
        help=(
            "Initial robot joint pose. deployment_home matches deployment/home_robot.py; "
            "sim_default matches the IsaacSim training asset default; env_reset leaves the randomized env reset untouched."
        ),
    )
    parser.add_argument("--status_interval_s", type=float, default=2.0)
    parser.add_argument("--depth_topic", default=DEPTH_TOPIC)
    parser.add_argument("--rgb_topic", default=RGB_TOPIC)
    parser.add_argument("--camera_info_topic", default=CAMERA_INFO_TOPIC)
    parser.add_argument("--depth_frame_id", default="isaacsim_student_camera")
    parser.add_argument("--iiwa_joint_state_topic", default=IIWA_JOINT_STATE_TOPIC)
    parser.add_argument("--sharpa_joint_state_topic", default=SHARPA_JOINT_STATE_TOPIC)
    parser.add_argument("--iiwa_joint_cmd_topic", default=IIWA_JOINT_CMD_TOPIC)
    parser.add_argument("--sharpa_joint_cmd_topic", default=SHARPA_JOINT_CMD_TOPIC)
    parser.add_argument("--object_pose_topic", default=OBJECT_POSE_TOPIC)
    parser.add_argument(
        "--publish_object_pose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Publish Isaac ground-truth object pose on --object_pose_topic.",
    )
    parser.add_argument("--gt_object_pose_topic", default=ISAAC_GT_OBJECT_POSE_TOPIC)
    parser.add_argument(
        "--publish_gt_object_pose_debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Also publish Isaac ground-truth object pose on --gt_object_pose_topic. "
            "Use with --no-publish_object_pose when FoundationPose owns /robot_frame/current_object_pose."
        ),
    )

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = bool(args.enable_depth or args.enable_rgb)
    if args.benchmark:
        args.realtime = False
        args.publish_depth = False
        args.publish_rgb = False
    return args


def main() -> None:
    args = parse_args()

    from isaaclab.app import AppLauncher

    app = AppLauncher(args).app
    try:
        import gymnasium as gym

        import isaacsimenvs  # noqa: F401
        from isaacsimenvs.distill_depth import _apply_student_camera_preset, _load_env_cfg

        env_cfg = _load_env_cfg(args.task, args.teacher_config, args.num_envs, args.sim_device)
        env_cfg.scene.num_envs = int(args.num_envs)
        camera_enabled = bool(args.enable_depth or args.enable_rgb)
        env_cfg.student_obs.image_enabled = camera_enabled
        if camera_enabled:
            if args.enable_depth and args.enable_rgb:
                env_cfg.student_obs.image_modality = "rgbd"
            elif args.enable_rgb:
                env_cfg.student_obs.image_modality = "rgb"
            else:
                env_cfg.student_obs.image_modality = "depth"
        if args.deployment_mode and args.zero_training_randomization:
            _zero_training_randomization(env_cfg)
        if args.deployment_mode and args.disable_env_resets:
            env_cfg.episode_length_s = DEPLOYMENT_EPISODE_LENGTH_S
            env_cfg.termination.max_consecutive_successes = 0
        args.resolved_object_init_pose_wxyz = None
        if args.deployment_mode:
            args.resolved_object_init_pose_wxyz = _apply_object_init_cfg(args, env_cfg)
        elif args.initial_robot_pose != "env_reset":
            env_cfg.reset.reset_dof_pos_random_interval_arm = 0.0
            env_cfg.reset.reset_dof_pos_random_interval_fingers = 0.0
            env_cfg.reset.reset_dof_vel_random_interval = 0.0
        _apply_student_camera_preset(env_cfg, args.student_camera_preset)
        if args.camera_image_width is not None:
            env_cfg.student_obs.image_width = int(args.camera_image_width)
            env_cfg.student_obs.image_input_width = int(args.camera_image_width)
        if args.camera_image_height is not None:
            env_cfg.student_obs.image_height = int(args.camera_image_height)
            env_cfg.student_obs.image_input_height = int(args.camera_image_height)
        if args.camera_image_width is not None or args.camera_image_height is not None:
            env_cfg.student_obs.crop_enabled = False
            env_cfg.student_obs.crop_top_left = (0, 0)
            env_cfg.student_obs.crop_bottom_right = (
                int(env_cfg.student_obs.image_width),
                int(env_cfg.student_obs.image_height),
            )
        if args.camera_k_file is not None and args.camera_intrinsic_matrix is not None:
            raise ValueError("Use only one of --camera_k_file or --camera_intrinsic_matrix.")
        if args.camera_k_file is not None:
            k = np.loadtxt(args.camera_k_file, dtype=np.float32).reshape(3, 3)
            _validate_camera_k_for_image(
                k,
                width=int(env_cfg.student_obs.image_width),
                height=int(env_cfg.student_obs.image_height),
                source=str(args.camera_k_file),
            )
            env_cfg.student_obs.camera_intrinsic_matrix = tuple(float(v) for v in k.reshape(-1))
        if args.camera_intrinsic_matrix is not None:
            k = np.asarray(args.camera_intrinsic_matrix, dtype=np.float32).reshape(3, 3)
            _validate_camera_k_for_image(
                k,
                width=int(env_cfg.student_obs.image_width),
                height=int(env_cfg.student_obs.image_height),
                source="--camera_intrinsic_matrix",
            )
            env_cfg.student_obs.camera_intrinsic_matrix = tuple(float(v) for v in k.reshape(-1))
        env_cfg.student_obs.depth_noise_profile = args.depth_noise_profile
        if args.depth_noise_strength is not None:
            env_cfg.student_obs.depth_noise_strength = float(args.depth_noise_strength)
        if args.camera_pose_randomization_profile is not None:
            env_cfg.student_obs.camera_pose_randomization_profile = args.camera_pose_randomization_profile
        if args.camera_pose_randomization_mode is not None:
            env_cfg.student_obs.camera_pose_randomization_mode = args.camera_pose_randomization_mode
        if args.camera_pos_noise_m is not None:
            env_cfg.student_obs.camera_pos_noise_m = tuple(float(v) for v in args.camera_pos_noise_m)
        if args.camera_rot_noise_deg is not None:
            env_cfg.student_obs.camera_rot_noise_deg = tuple(float(v) for v in args.camera_rot_noise_deg)
        if args.peg_urdf is not None:
            env_cfg.assets.peg_urdf = args.peg_urdf
            env_cfg.assets.object_name = Path(args.peg_urdf).stem
        if args.peg_goal_mode is not None:
            env_cfg.peg_in_hole.goal_mode = args.peg_goal_mode
        if hasattr(env_cfg, "peg_in_hole"):
            env_cfg.peg_in_hole.hide_hole_fixture = bool(args.hide_hole_fixture)
        if args.object_init_orientation_mode is not None and hasattr(env_cfg, "peg_in_hole"):
            env_cfg.peg_in_hole.object_init_orientation_mode = args.object_init_orientation_mode
        if args.deployment_mode:
            object_pose_note = (
                "manual_fixed_pose"
                if args.resolved_object_init_pose_wxyz is not None
                else args.object_init_mode
            )
            print(
                "[isaac_depth_env_node] deployment config: "
                f"disable_env_resets={args.disable_env_resets} "
                f"zero_training_randomization={args.zero_training_randomization} "
                f"initial_robot_pose={args.initial_robot_pose} "
                f"object_init={object_pose_note} "
                f"enable_depth={args.enable_depth} "
                f"enable_rgb={args.enable_rgb} "
                f"camera_info_mode={args.camera_info_mode} "
                f"hide_hole_fixture={getattr(env_cfg.peg_in_hole, 'hide_hole_fixture', False)} "
                f"depth_noise_profile={env_cfg.student_obs.depth_noise_profile} "
                f"published_depth_source={args.published_depth_source} "
                f"camera_rand={env_cfg.student_obs.camera_pose_randomization_profile} "
                f"camera_pos_noise_m={env_cfg.student_obs.camera_pos_noise_m} "
                f"camera_rot_noise_deg={env_cfg.student_obs.camera_rot_noise_deg}",
                flush=True,
            )

        env = gym.make(args.task, cfg=env_cfg)
        inner = env.unwrapped
        if args.deployment_mode and args.disable_env_resets:
            _install_no_reset_done_wrapper(inner)
        ros = None if args.benchmark else _import_ros()
        node = IsaacDepthEnvNode(args, env, inner, ros)
        node.run()
        env.close()
    finally:
        app.close()


if __name__ == "__main__":
    main()
