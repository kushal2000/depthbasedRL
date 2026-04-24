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
from sensor_msgs.msg import JointState
from termcolor import colored

from isaacgymenvs.utils.observation_action_utils_sharpa import (
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
    student_checkpoint: Path
    distill_config: Path
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


class StudentDepthPolicyNode:
    def __init__(self, args: StudentDepthPolicyNodeArgs):
        self.args = args
        rospy.init_node("student_depth_policy_node")

        self.iiwa_joint_cmd_pub = rospy.Publisher(args.iiwa_joint_cmd_topic, JointState, queue_size=1)
        self.sharpa_joint_cmd_pub = rospy.Publisher(args.sharpa_joint_cmd_topic, JointState, queue_size=1)

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

        self.zed, self.runtime_parameters, self.depth_mat = init_zed(
            serial_number=args.serial_number,
            exposure=args.zed_exposure,
            gain=args.zed_gain,
            resolution=args.zed_resolution,
        )

    def iiwa_joint_state_callback(self, msg: JointState):
        self.iiwa_joint_state_msg = msg

    def sharpa_joint_state_callback(self, msg: JointState):
        self.sharpa_joint_state_msg = msg

    def create_student_inputs(self) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[np.ndarray]]:
        if self.iiwa_joint_state_msg is None or self.sharpa_joint_state_msg is None:
            warn_every(
                "Waiting for iiwa and sharpa joint states before student policy inference",
                n_seconds=1.0,
            )
            return None, None, None

        iiwa_msg = self.iiwa_joint_state_msg
        sharpa_msg = self.sharpa_joint_state_msg
        q = np.concatenate([np.asarray(iiwa_msg.position, dtype=np.float32), np.asarray(sharpa_msg.position, dtype=np.float32)])
        qd = np.concatenate([np.asarray(iiwa_msg.velocity, dtype=np.float32), np.asarray(sharpa_msg.velocity, dtype=np.float32)])
        if q.shape != (29,) or qd.shape != (29,):
            warn_every(f"Expected 29 joint positions/velocities, got q={q.shape}, qd={qd.shape}", 1.0)
            return None, None, None

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
            return None, None, None

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
        return image_tensor, proprio_tensor, q

    def publish_targets(self, joint_pos_targets: np.ndarray):
        joint_pos_targets = joint_pos_targets[0]

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
            image_tensor, proprio_tensor, q = self.create_student_inputs()
            if image_tensor is None or proprio_tensor is None or q is None:
                rate.sleep()
                continue

            with torch.no_grad():
                student_out, self.hidden_state = self.student(image_tensor, proprio_tensor, self.hidden_state)
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

            if int(time.time() * 2) % 10 == 0:
                info(
                    f"Loop OK: action_norm={float(np.linalg.norm(normalized_action)):.3f}, "
                    f"step_ms={(time.time() - loop_start) * 1000.0:.1f}"
                )
            rate.sleep()


def main():
    args = tyro.cli(StudentDepthPolicyNodeArgs)
    node = StudentDepthPolicyNode(args)
    try:
        node.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
