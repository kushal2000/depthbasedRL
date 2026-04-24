#!/usr/bin/env python3

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pyzed.sl as sl
import rospy
import torch
import tyro
import yaml
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from termcolor import colored

from isaacgymenvs.utils.observation_action_utils_sharpa import (
    JOINT_NAMES_ISAACGYM,
    compute_joint_pos_targets,
)
from isaacsim_conversion.image_robustness import (
    PreprocessCfg,
    preprocess_policy_images,
)
from isaacsim_conversion.student_policy import MonoTransformerRecurrentPolicy


def warn(message: str):
    print(colored(message, "yellow"))


def info(message: str):
    print(colored(message, "green"))


def warn_every(message: str, n_seconds: float, key=None):
    if not hasattr(warn_every, "_last_times"):
        warn_every._last_times = {}
    key = key or message
    last = warn_every._last_times.get(key, 0.0)
    now = time.time()
    if now - last > n_seconds:
        warn(message)
        warn_every._last_times[key] = now


def load_yaml(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def load_student_checkpoint(path: Path, student: torch.nn.Module):
    ckpt = torch.load(path, map_location="cpu")
    state_dict = ckpt["student_state_dict"] if "student_state_dict" in ckpt else ckpt
    student.load_state_dict(state_dict)
    return ckpt


IIWA_JOINT_NAMES = [
    "iiwa_joint_1",
    "iiwa_joint_2",
    "iiwa_joint_3",
    "iiwa_joint_4",
    "iiwa_joint_5",
    "iiwa_joint_6",
    "iiwa_joint_7",
]

SHARPA_JOINT_NAMES = [f"joint_{i}.0" for i in range(22)]

T_W_R = np.eye(4, dtype=np.float32)
T_W_R[:3, 3] = np.array([0.0, 0.8, 0.0], dtype=np.float32)
T_R_W = np.linalg.inv(T_W_R)


def init_zed(serial_number: str, exposure: int, gain: int, resolution: str):
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.svo_real_time_mode = True
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_units = sl.UNIT.MILLIMETER
    init_params.set_from_serial_number(int(serial_number))
    resolution_map = {
        "HD1080": sl.RESOLUTION.HD1080,
        "HD720": sl.RESOLUTION.HD720,
        "VGA": sl.RESOLUTION.VGA,
    }
    init_params.camera_resolution = resolution_map[resolution]
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open ZED camera: {err}")
    zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, exposure)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, gain)
    runtime_parameters = sl.RuntimeParameters()
    depth_mat = sl.Mat()
    return zed, runtime_parameters, depth_mat


def reorder_joint_state(msg: JointState, expected_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    if len(msg.name) == len(expected_names) and list(msg.name) == expected_names:
        return np.asarray(msg.position, dtype=np.float32), np.asarray(msg.velocity, dtype=np.float32)

    if not msg.name:
        raise ValueError("JointState message has no names; cannot safely reorder.")

    name_to_index = {name: i for i, name in enumerate(msg.name)}
    missing = [name for name in expected_names if name not in name_to_index]
    if missing:
        raise ValueError(f"Missing expected joints in JointState: {missing}")

    pos = np.array([msg.position[name_to_index[name]] for name in expected_names], dtype=np.float32)
    vel = np.array([msg.velocity[name_to_index[name]] for name in expected_names], dtype=np.float32)
    return pos, vel


def read_depth_frame(
    zed,
    runtime_parameters,
    depth_mat,
    width: int,
    height: int,
    camera_upsidedown: bool = False,
) -> Optional[np.ndarray]:
    if zed.grab(runtime_parameters) != sl.ERROR_CODE.SUCCESS:
        return None
    zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
    depth_mm = depth_mat.get_data()
    if camera_upsidedown:
        depth_mm = np.flip(depth_mm, axis=(0, 1))
    depth_mm = depth_mm.astype(np.float32)
    import cv2

    depth_mm = cv2.resize(depth_mm, (width, height), interpolation=cv2.INTER_NEAREST)
    depth_m = depth_mm / 1000.0
    depth_m[(depth_m < 1e-3) | (~np.isfinite(depth_m))] = 0.0
    return depth_m


def preprocess_depth_metric(
    depth_m: np.ndarray,
    mode: str,
    depth_min_m: float,
    depth_max_m: float,
) -> np.ndarray:
    depth = depth_m.astype(np.float32, copy=True)
    if mode == "clip_divide":
        return np.clip(depth, depth_min_m, depth_max_m) / depth_max_m
    valid = (depth >= depth_min_m) & (depth <= depth_max_m)
    if mode == "metric":
        depth[~valid] = 0.0
        return depth
    if mode == "window_normalize":
        normalized = (depth - depth_min_m) / (depth_max_m - depth_min_m)
        normalized[~valid] = 0.0
        return normalized
    raise ValueError(f"Unsupported depth preprocess mode: {mode}")


@dataclass
class StudentDepthPolicyNodeArgs:
    student_checkpoint: Path = Path("distillation_runs/local_policy_rollouts_2026_04_21/checkpoints/img80_depth_third_best.pt")
    distill_config: Path = Path("isaacsim_conversion/configs/hammer_distill_depth_80x45_window_online_dagger_512.yaml")
    device: str = "cuda"
    iiwa_joint_state_topic: str = "/iiwa/joint_states"
    sharpa_joint_state_topic: str = "/sharpa/joint_states"
    iiwa_joint_cmd_topic: str = "/iiwa/joint_cmd"
    sharpa_joint_cmd_topic: str = "/sharpa/joint_cmd"
    serial_number: str = "15107"
    zed_resolution: str = "HD720"
    zed_exposure: int = 25
    zed_gain: int = 40
    camera_upsidedown: bool = False
    control_hz: float = 60.0
    task_progress: float = 0.0
    hand_moving_average: float = 0.1
    arm_moving_average: float = 0.1
    hand_dof_speed_scale: float = 1.5
    publish_joint_commands: bool = False
    publish_object_pos_topic: str = "/robot_frame/current_object_pose"
    object_pos_frame_id: str = "robot_frame"
    debug_print_proprio_every: int = 0
    debug_save_depth_dir: Optional[Path] = None
    debug_save_depth_every: int = 0
    debug_save_depth_video: Optional[Path] = None


class StudentDepthPolicyNode:
    def __init__(self, args: StudentDepthPolicyNodeArgs):
        self.args = args
        rospy.init_node("student_depth_policy_node")

        self.iiwa_joint_cmd_pub = rospy.Publisher(args.iiwa_joint_cmd_topic, JointState, queue_size=1)
        self.sharpa_joint_cmd_pub = rospy.Publisher(args.sharpa_joint_cmd_topic, JointState, queue_size=1)
        self.object_pos_pub = rospy.Publisher(args.publish_object_pos_topic, PoseStamped, queue_size=1)

        self.iiwa_joint_state_msg = None
        self.sharpa_joint_state_msg = None
        self.iiwa_joint_state_sub = rospy.Subscriber(args.iiwa_joint_state_topic, JointState, self.iiwa_joint_state_callback, queue_size=1)
        self.sharpa_joint_state_sub = rospy.Subscriber(args.sharpa_joint_state_topic, JointState, self.sharpa_joint_state_callback, queue_size=1)

        settings = load_yaml(args.distill_config)
        self.image_height = int(settings["image_height"])
        self.image_width = int(settings["image_width"])
        self.depth_preprocess_mode = str(settings["depth_preprocess_mode"])
        self.depth_min_m = float(settings["depth_min_m"])
        self.depth_max_m = float(settings["depth_max_m"])
        self.control_dt = float(settings.get("control_dt", 1.0 / args.control_hz))
        preprocess_settings = ((settings.get("image_robustness") or {}).get("preprocess") or {})
        self.preprocess_cfg = PreprocessCfg(
            resize_mode=str(preprocess_settings.get("resize_mode", "exact")),
            pad_to_size=bool(preprocess_settings.get("pad_to_size", False)),
            apply_rgb_float_scaling=bool(preprocess_settings.get("apply_rgb_float_scaling", True)),
            zero_out_invalid_depth=bool(preprocess_settings.get("zero_out_invalid_depth", True)),
        )

        requested_device = args.device
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            warn("CUDA requested but unavailable, falling back to CPU")
            requested_device = "cpu"
        self.device = torch.device(requested_device)

        self.student = MonoTransformerRecurrentPolicy(
            image_channels=1,
            proprio_dim=29 + 29 + 29 + 1,
            action_dim=29,
            aux_heads={"object_pos": 3},
        ).to(self.device)
        load_student_checkpoint(args.student_checkpoint, self.student)
        self.student.eval()
        self.hidden_state = self.student.initial_state(batch_size=1, device=self.device)

        self.prev_targets: Optional[np.ndarray] = None
        self._loop_counter = 0
        self._last_rate_log_time = time.time()
        self._last_rate_log_step = 0
        self._depth_video_writer = None

        if args.debug_save_depth_dir is not None:
            args.debug_save_depth_dir.mkdir(parents=True, exist_ok=True)

        self.zed, self.runtime_parameters, self.depth_mat = init_zed(
            serial_number=args.serial_number,
            exposure=args.zed_exposure,
            gain=args.zed_gain,
            resolution=args.zed_resolution,
        )
        camera_info = self.zed.get_camera_information()
        cam_cfg = camera_info.camera_configuration
        info(
            "Loaded student policy "
            f"(checkpoint={args.student_checkpoint}, image_channels=1, proprio_dim=88, action_dim=29, "
            f"trained_image={self.image_width}x{self.image_height}, depth_mode={self.depth_preprocess_mode})"
        )
        info(
            "Student inputs: depth image + proprio=[q(29), qd(29), prev_targets(29), task_progress(1)]"
        )
        info(
            f"ZED opened (serial={args.serial_number}, requested_resolution={args.zed_resolution}, "
            f"native_resolution={cam_cfg.resolution.width}x{cam_cfg.resolution.height}, "
            f"policy_resolution={self.image_width}x{self.image_height})"
        )
        info(
            f"ROS I/O: joint_states=({args.iiwa_joint_state_topic}, {args.sharpa_joint_state_topic}), "
            f"joint_cmd=({args.iiwa_joint_cmd_topic}, {args.sharpa_joint_cmd_topic}, enabled={args.publish_joint_commands}), "
            f"object_pose_topic={args.publish_object_pos_topic}"
        )
        expected_hz = 1.0 / self.control_dt if self.control_dt > 0 else float("nan")
        info(f"Configured control rate: requested={args.control_hz:.1f} Hz, distill_cfg={expected_hz:.1f} Hz")
        if abs(args.control_hz - expected_hz) > 1e-3:
            warn(
                f"control_hz ({args.control_hz}) does not match distill control_dt ({expected_hz:.3f} Hz equivalent)"
            )
        if args.debug_save_depth_dir is not None:
            info(f"Depth PNG debug enabled: dir={args.debug_save_depth_dir}, every={args.debug_save_depth_every} steps")
        if args.debug_save_depth_video is not None:
            info(f"Depth video debug enabled: path={args.debug_save_depth_video}")

    def iiwa_joint_state_callback(self, msg: JointState):
        self.iiwa_joint_state_msg = msg

    def sharpa_joint_state_callback(self, msg: JointState):
        self.sharpa_joint_state_msg = msg

    def create_student_inputs(
        self,
    ) -> tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        if self.iiwa_joint_state_msg is None or self.sharpa_joint_state_msg is None:
            warn_every(
                "Waiting for iiwa and sharpa joint states before student policy inference",
                n_seconds=1.0,
            )
            return None, None, None, None, None, None

        iiwa_msg = self.iiwa_joint_state_msg
        sharpa_msg = self.sharpa_joint_state_msg
        try:
            iiwa_position, iiwa_velocity = reorder_joint_state(iiwa_msg, IIWA_JOINT_NAMES)
            sharpa_position, sharpa_velocity = reorder_joint_state(sharpa_msg, SHARPA_JOINT_NAMES)
        except ValueError as exc:
            warn_every(f"JointState ordering error: {exc}", 1.0, key="joint_state_ordering")
            return None, None, None, None, None, None

        q = np.concatenate([iiwa_position, sharpa_position])
        qd = np.concatenate([iiwa_velocity, sharpa_velocity])
        if q.shape != (29,) or qd.shape != (29,):
            warn_every(f"Expected 29 joint positions/velocities, got q={q.shape}, qd={qd.shape}", 1.0)
            return None, None, None, None, None, None

        if self.prev_targets is None:
            self.prev_targets = q.copy()

        depth_m = read_depth_frame(
            self.zed,
            self.runtime_parameters,
            self.depth_mat,
            width=self.image_width,
            height=self.image_height,
            camera_upsidedown=self.args.camera_upsidedown,
        )
        if depth_m is None:
            warn_every("ZED grab failed, skipping control step", 1.0)
            return None, None, None, None, None, None

        depth_proc = preprocess_depth_metric(
            depth_m,
            mode=self.depth_preprocess_mode,
            depth_min_m=self.depth_min_m,
            depth_max_m=self.depth_max_m,
        )
        depth_tensor = torch.from_numpy(depth_proc).unsqueeze(0).unsqueeze(0).to(self.device)
        image_tensor, _ = preprocess_policy_images(
            {"depth": depth_tensor},
            modality="depth",
            out_height=self.image_height,
            out_width=self.image_width,
            preprocess_cfg=self.preprocess_cfg,
        )

        proprio = np.concatenate(
            [
                q,
                qd,
                self.prev_targets.astype(np.float32),
                np.array([self.args.task_progress], dtype=np.float32),
            ]
        )
        proprio_tensor = torch.from_numpy(proprio).unsqueeze(0).to(self.device)
        return image_tensor, proprio_tensor, q, qd, depth_m, depth_proc

    def _save_depth_debug(self, depth_m: np.ndarray, depth_proc: np.ndarray):
        if self.args.debug_save_depth_every <= 0:
            return
        if self._loop_counter % self.args.debug_save_depth_every != 0:
            return

        import cv2

        raw_vis = np.zeros_like(depth_m, dtype=np.float32)
        raw_valid = depth_m > 0.0
        if np.any(raw_valid):
            raw_vis[raw_valid] = np.clip(
                (depth_m[raw_valid] - self.depth_min_m) / max(self.depth_max_m - self.depth_min_m, 1e-6),
                0.0,
                1.0,
            )
        raw_vis_u8 = (raw_vis * 255.0).astype(np.uint8)

        policy_vis = np.clip(depth_proc, 0.0, 1.0)
        policy_vis_u8 = (policy_vis * 255.0).astype(np.uint8)

        if self.args.debug_save_depth_dir is not None:
            raw_png = self.args.debug_save_depth_dir / f"depth_raw_m_{self._loop_counter:06d}.png"
            raw_npy = self.args.debug_save_depth_dir / f"depth_raw_m_{self._loop_counter:06d}.npy"
            policy_png = self.args.debug_save_depth_dir / f"depth_policy_{self._loop_counter:06d}.png"
            policy_npy = self.args.debug_save_depth_dir / f"depth_policy_{self._loop_counter:06d}.npy"
            cv2.imwrite(str(raw_png), raw_vis_u8)
            cv2.imwrite(str(policy_png), policy_vis_u8)
            np.save(raw_npy, depth_m.astype(np.float32))
            np.save(policy_npy, depth_proc.astype(np.float32))

        if self.args.debug_save_depth_video is not None:
            if self._depth_video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self._depth_video_writer = cv2.VideoWriter(
                    str(self.args.debug_save_depth_video),
                    fourcc,
                    self.args.control_hz,
                    (self.image_width, self.image_height),
                    False,
                )
            self._depth_video_writer.write(policy_vis_u8)

    def _publish_object_pos(self, student_out):
        object_pos = student_out.aux.get("object_pos")
        if object_pos is None:
            return
        pos_world_like = object_pos.detach().cpu().numpy().reshape(-1)
        if pos_world_like.shape != (3,):
            warn_every(f"Unexpected object_pos shape: {pos_world_like.shape}", 1.0, key="object_pos_shape")
            return
        pos_h = np.ones(4, dtype=np.float32)
        pos_h[:3] = pos_world_like.astype(np.float32)
        pos_robot = (T_R_W @ pos_h)[:3]
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.args.object_pos_frame_id
        msg.pose.position.x = float(pos_robot[0])
        msg.pose.position.y = float(pos_robot[1])
        msg.pose.position.z = float(pos_robot[2])
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 1.0
        self.object_pos_pub.publish(msg)

    def _log_proprio_debug(self, q: np.ndarray, qd: np.ndarray):
        if self.args.debug_print_proprio_every <= 0:
            return
        if self._loop_counter % self.args.debug_print_proprio_every != 0:
            return
        info(
            "Proprio "
            f"q[:7]={np.array2string(q[:7], precision=3, suppress_small=True)} "
            f"qd[:7]={np.array2string(qd[:7], precision=3, suppress_small=True)} "
            f"prev[:7]={np.array2string(self.prev_targets[:7], precision=3, suppress_small=True)}"
        )

    def publish_targets(self, joint_pos_targets: np.ndarray):
        joint_pos_targets = joint_pos_targets[0]

        if not self.args.publish_joint_commands:
            return

        iiwa_msg = JointState()
        iiwa_msg.header.stamp = rospy.Time.now()
        iiwa_msg.name = [
            "iiwa_joint_1",
            "iiwa_joint_2",
            "iiwa_joint_3",
            "iiwa_joint_4",
            "iiwa_joint_5",
            "iiwa_joint_6",
            "iiwa_joint_7",
        ]
        iiwa_msg.position = joint_pos_targets[:7].tolist()
        self.iiwa_joint_cmd_pub.publish(iiwa_msg)

        sharpa_msg = JointState()
        sharpa_msg.header.stamp = rospy.Time.now()
        sharpa_msg.name = [f"joint_{i}.0" for i in range(22)]
        sharpa_msg.position = joint_pos_targets[7:].tolist()
        self.sharpa_joint_cmd_pub.publish(sharpa_msg)

    def run(self):
        rate = rospy.Rate(self.args.control_hz)
        info(
            f"Student depth policy node started "
            f"(checkpoint={self.args.student_checkpoint}, image={self.image_width}x{self.image_height}, "
            f"depth_mode={self.depth_preprocess_mode}, device={self.device})"
        )
        while not rospy.is_shutdown():
            loop_start = time.time()
            image_tensor, proprio_tensor, q, qd, depth_m, depth_proc = self.create_student_inputs()
            if (
                image_tensor is None
                or proprio_tensor is None
                or q is None
                or qd is None
                or depth_m is None
                or depth_proc is None
            ):
                rate.sleep()
                continue

            with torch.no_grad():
                student_out, self.hidden_state = self.student(image_tensor, proprio_tensor, self.hidden_state)
            self._save_depth_debug(depth_m, depth_proc)
            self._publish_object_pos(student_out)
            normalized_action = student_out.action.detach().cpu().numpy()
            joint_pos_targets = compute_joint_pos_targets(
                actions=normalized_action,
                prev_targets=self.prev_targets[None],
                hand_moving_average=self.args.hand_moving_average,
                arm_moving_average=self.args.arm_moving_average,
                hand_dof_speed_scale=self.args.hand_dof_speed_scale,
                dt=1.0 / self.args.control_hz,
            )
            self.prev_targets = joint_pos_targets[0].copy()
            self.publish_targets(joint_pos_targets)
            self._loop_counter += 1
            self._log_proprio_debug(q, qd)
            elapsed = time.time() - loop_start
            if elapsed > (1.0 / self.args.control_hz):
                warn_every(
                    f"Control loop overrun: step_ms={elapsed * 1000.0:.1f} > budget_ms={(1000.0 / self.args.control_hz):.1f}",
                    1.0,
                    key="loop_overrun",
                )

            now = time.time()
            if now - self._last_rate_log_time >= 2.0:
                dt = now - self._last_rate_log_time
                hz = (self._loop_counter - self._last_rate_log_step) / max(dt, 1e-6)
                info(
                    f"Loop rate={hz:.1f} Hz, step_ms={elapsed * 1000.0:.1f}, "
                    f"action_norm={float(np.linalg.norm(normalized_action)):.3f}"
                )
                self._last_rate_log_time = now
                self._last_rate_log_step = self._loop_counter
            rate.sleep()

        if self._depth_video_writer is not None:
            self._depth_video_writer.release()


def main():
    args = tyro.cli(StudentDepthPolicyNodeArgs)
    node = StudentDepthPolicyNode(args)
    try:
        node.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
