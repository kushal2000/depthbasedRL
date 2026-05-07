#!/usr/bin/env python
"""ROS node for deploying Isaac Sim depth-student policies.

This node is for the new Isaac Lab-native depth student checkpoints trained by
``isaacsimenvs/distill_depth.py``. It is intentionally separate from
``deployment/rl_policy_node.py``, which serves the old privileged-state
``pretrained_policy`` rl_games model.
"""

from __future__ import annotations

import argparse
import atexit
import importlib.util
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import rospy
import torch
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Header

try:
    from termcolor import colored
except ModuleNotFoundError:
    def colored(message: str, *_args, **_kwargs) -> str:
        return message

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


N_ACTIONS = 29
N_ARM = 7
DEPTH_NEAR_M = 0.70
DEPTH_FAR_M = 1.10
RESIZED_WIDTH = 160
RESIZED_HEIGHT = 90
CROP_X0 = 90
CROP_Y0 = 0
CROP_X1 = 160
CROP_Y1 = 70
POLICY_WIDTH = CROP_X1 - CROP_X0
POLICY_HEIGHT = CROP_Y1 - CROP_Y0
DEFAULT_CONTROL_HZ = 60.0
# distill_depth.py trains object_pos in the env-local/world frame. The real
# robot ROS topics use robot_frame, whose origin is translated +0.8 m in sim.
SIM_WORLD_T_ROBOT_POS_M = np.array([0.0, 0.8, 0.0], dtype=np.float64)

# Keep these in sync with
# isaacgymenvs/utils/observation_action_utils_sharpa.py. The deployment node
# defines them locally to avoid importing IsaacGym/Hydra/IsaacLab packages in a
# ROS-only process.
Q_LOWER_LIMITS_np = np.array(
    [
        -2.9671,
        -2.0944,
        -2.9671,
        -2.0944,
        -2.9671,
        -2.0944,
        -3.0543,
        -0.1745,
        -0.3491,
        -0.5236,
        -0.3491,
        0.0000,
        -0.1745,
        -0.0349,
        0.0000,
        0.0000,
        -0.1745,
        -0.0349,
        0.0000,
        0.0000,
        -0.1745,
        -0.0349,
        0.0000,
        0.0000,
        0.0000,
        -0.1745,
        -0.0349,
        0.0000,
        0.0000,
    ],
    dtype=np.float32,
)
Q_UPPER_LIMITS_np = np.array(
    [
        2.9671,
        2.0944,
        2.9671,
        2.0944,
        2.9671,
        2.0944,
        3.0543,
        1.9199,
        0.1309,
        1.3963,
        0.3491,
        1.7453,
        1.5708,
        0.0349,
        1.7453,
        1.3963,
        1.5708,
        0.0349,
        1.7453,
        1.3963,
        1.5708,
        0.0349,
        1.7453,
        1.3963,
        0.2618,
        1.5708,
        0.0349,
        1.7453,
        1.3963,
    ],
    dtype=np.float32,
)
Q_LOWER_LIMITS_restricted_np = Q_LOWER_LIMITS_np.copy()
Q_LOWER_LIMITS_restricted_np[:N_ARM] += np.float32(np.deg2rad(10.0))
Q_UPPER_LIMITS_restricted_np = Q_UPPER_LIMITS_np.copy()
Q_UPPER_LIMITS_restricted_np[:N_ARM] -= np.float32(np.deg2rad(10.0))
assert Q_LOWER_LIMITS_restricted_np.shape == (N_ACTIONS,)
assert Q_UPPER_LIMITS_restricted_np.shape == (N_ACTIONS,)


def load_student_policy_class():
    """Load the policy class without importing the Isaac Sim task package."""

    module_path = REPO_ROOT / "isaacsimenvs" / "distillation" / "student_policy.py"
    spec = importlib.util.spec_from_file_location("depthbasedrl_student_policy", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load student policy module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.MonoTransformerRecurrentPolicy


def info(message: str) -> None:
    print(colored(message, "green"), flush=True)


def warn(message: str) -> None:
    print(colored(message, "yellow"), flush=True)


def warn_every(message: str, n_seconds: float, key: Optional[str] = None) -> None:
    if not hasattr(warn_every, "_last_times"):
        warn_every._last_times = {}
    key = key or message
    now = time.time()
    last = warn_every._last_times.get(key, 0.0)
    if now - last >= n_seconds:
        warn(message)
        warn_every._last_times[key] = now


def _safe_normalize(vec: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return vec / vec.norm(dim=-1, keepdim=True).clamp_min(eps)


def _orthogonal_fallback(unit_vec: torch.Tensor) -> torch.Tensor:
    x_axis = torch.zeros_like(unit_vec)
    y_axis = torch.zeros_like(unit_vec)
    x_axis[..., 0] = 1.0
    y_axis[..., 1] = 1.0
    axis = torch.where(unit_vec[..., 0:1].abs() < 0.9, x_axis, y_axis)
    return axis - (axis * unit_vec).sum(dim=-1, keepdim=True) * unit_vec


def rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    """Convert Zhou-style 6D rotation predictions to rotation matrices."""

    a1 = rot6d[..., 0:3]
    a2 = rot6d[..., 3:6]
    x_axis = torch.zeros_like(a1)
    x_axis[..., 0] = 1.0
    a1 = torch.where(a1.norm(dim=-1, keepdim=True) < 1e-5, x_axis, a1)
    b1 = _safe_normalize(a1)
    a2_orthogonal = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    degenerate = a2_orthogonal.norm(dim=-1, keepdim=True) < 1e-5
    a2_orthogonal = torch.where(degenerate, _orthogonal_fallback(b1), a2_orthogonal)
    b2 = _safe_normalize(a2_orthogonal)
    b3 = _safe_normalize(torch.cross(b1, b2, dim=-1))
    b2 = torch.cross(b3, b1, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def pose_stamped_msg(
    *,
    pos: np.ndarray,
    quat_xyzw: np.ndarray,
    frame_id: str,
    stamp: rospy.Time,
) -> PoseStamped:
    msg = PoseStamped()
    msg.header = Header(stamp=stamp, frame_id=frame_id)
    msg.pose.position.x = float(pos[0])
    msg.pose.position.y = float(pos[1])
    msg.pose.position.z = float(pos[2])
    msg.pose.orientation.x = float(quat_xyzw[0])
    msg.pose.orientation.y = float(quat_xyzw[1])
    msg.pose.orientation.z = float(quat_xyzw[2])
    msg.pose.orientation.w = float(quat_xyzw[3])
    return msg


def infer_policy_kwargs(state_dict: dict[str, torch.Tensor]) -> tuple[dict, dict[str, int]]:
    conv = state_dict["patch_embed.proj.weight"]
    proprio = state_dict["proprio_proj.0.weight"]
    rnn = state_dict["rnn.weight_hh"]
    actor_out = state_dict["actor.3.weight"]
    aux_heads: dict[str, int] = {}
    for key, value in state_dict.items():
        prefix = "aux_heads."
        suffix = ".3.weight"
        if key.startswith(prefix) and key.endswith(suffix):
            name = key[len(prefix) : -len(suffix)]
            aux_heads[name] = int(value.shape[0])
    kwargs = {
        "image_channels": int(conv.shape[1]),
        "proprio_dim": int(proprio.shape[1]),
        "action_dim": int(actor_out.shape[0]),
        "embed_dim": int(conv.shape[0]),
        "hidden_dim": int(rnn.shape[1]),
        "patch_size": int(conv.shape[-1]),
        "aux_heads": aux_heads,
    }
    return kwargs, aux_heads


def load_student_policy(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("student", checkpoint)
    kwargs, aux_heads = infer_policy_kwargs(state)
    MonoTransformerRecurrentPolicy = load_student_policy_class()
    policy = MonoTransformerRecurrentPolicy(**kwargs).to(device)
    policy.load_state_dict(state, strict=True)
    policy.eval()
    return policy, aux_heads, int(checkpoint.get("step", 0)), float(checkpoint.get("best_metric", -1.0))


def compute_restricted_joint_pos_targets(
    *,
    actions: np.ndarray,
    prev_targets: np.ndarray,
    hand_moving_average: float,
    arm_moving_average: float,
    dof_speed_scale: float,
    dt: float,
) -> np.ndarray:
    """Match Isaac Lab's current action pipeline with restricted robot limits."""

    if actions.ndim != 2 or actions.shape[1] != N_ACTIONS:
        raise RuntimeError(f"Expected actions shape (N, 29), got {actions.shape}")
    if prev_targets.shape != actions.shape:
        raise RuntimeError(f"prev_targets shape {prev_targets.shape} does not match actions {actions.shape}")
    lower = Q_LOWER_LIMITS_restricted_np.astype(np.float32)
    upper = Q_UPPER_LIMITS_restricted_np.astype(np.float32)
    targets = prev_targets.astype(np.float32, copy=True)

    arm_raw = prev_targets[:, :N_ARM] + dof_speed_scale * dt * actions[:, :N_ARM]
    arm_raw = np.clip(arm_raw, lower[:N_ARM], upper[:N_ARM])
    targets[:, :N_ARM] = (
        arm_moving_average * arm_raw
        + (1.0 - arm_moving_average) * prev_targets[:, :N_ARM]
    )
    targets[:, :N_ARM] = np.clip(targets[:, :N_ARM], lower[:N_ARM], upper[:N_ARM])

    hand_raw = lower[N_ARM:] + 0.5 * (actions[:, N_ARM:] + 1.0) * (upper[N_ARM:] - lower[N_ARM:])
    targets[:, N_ARM:] = (
        hand_moving_average * hand_raw
        + (1.0 - hand_moving_average) * prev_targets[:, N_ARM:]
    )
    targets[:, N_ARM:] = np.clip(targets[:, N_ARM:], lower[N_ARM:], upper[N_ARM:])
    return targets


@dataclass
class DepthPipelineOutput:
    raw_depth_m: np.ndarray
    resized_depth_m: np.ndarray
    policy_full_depth: np.ndarray
    policy_crop: np.ndarray


@dataclass
class DepthFrame:
    depth: np.ndarray
    encoding: str
    stamp: rospy.Time


class ZedDepthCamera:
    """Direct ZED SDK depth capture to avoid streaming depth images through ROS."""

    def __init__(self, args: argparse.Namespace) -> None:
        try:
            import pyzed.sl as sl
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "pyzed.sl is required for --depth_source zed_sdk. "
                "Use a shell/environment with the ZED SDK Python bindings, or pass --depth_source ros_topic."
            ) from exc

        self.sl = sl
        self.camera = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = self._enum_value(sl.RESOLUTION, args.zed_resolution)
        init_params.depth_mode = self._enum_value(sl.DEPTH_MODE, args.zed_depth_mode)
        init_params.coordinate_units = sl.UNIT.MILLIMETER
        if int(args.zed_camera_fps) > 0:
            init_params.camera_fps = int(args.zed_camera_fps)
        if args.zed_serial_number:
            init_params.set_from_serial_number(int(args.zed_serial_number))

        err = self.camera.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED camera serial={args.zed_serial_number!r}: {err}")

        self.runtime_parameters = sl.RuntimeParameters()
        self.depth_mat = sl.Mat()
        self.camera_upsidedown = bool(args.zed_camera_upsidedown)
        atexit.register(self.close)
        info(
            "Opened ZED SDK depth camera "
            f"serial={args.zed_serial_number or '<default>'} resolution={args.zed_resolution} "
            f"depth_mode={args.zed_depth_mode} units=millimeters"
        )

    @staticmethod
    def _enum_value(enum_cls, name: str):
        try:
            return getattr(enum_cls, str(name).upper())
        except AttributeError as exc:
            valid = [item for item in dir(enum_cls) if item.isupper()]
            raise ValueError(f"Invalid ZED enum value {name!r}; valid values include {valid}") from exc

    def close(self) -> None:
        if self.camera is not None:
            self.camera.close()
            self.camera = None

    def read(self) -> DepthFrame | None:
        if self.camera is None:
            return None
        err = self.camera.grab(self.runtime_parameters)
        if err != self.sl.ERROR_CODE.SUCCESS:
            warn_every(f"ZED grab failed: {err}", 1.0, key="zed_grab_failed")
            return None
        self.camera.retrieve_measure(self.depth_mat, self.sl.MEASURE.DEPTH)
        depth_mm = np.asarray(self.depth_mat.get_data())
        if self.camera_upsidedown:
            import cv2

            depth_mm = cv2.flip(depth_mm, -1)
        return DepthFrame(
            depth=depth_mm,
            encoding="zed_sdk_mm",
            stamp=rospy.Time.now(),
        )


class DepthPreprocessor:
    def __init__(
        self,
        *,
        depth_units: str,
        resize_interpolation: str,
        near_m: float = DEPTH_NEAR_M,
        far_m: float = DEPTH_FAR_M,
    ) -> None:
        self.depth_units = depth_units
        self.near_m = near_m
        self.far_m = far_m
        if far_m <= near_m:
            raise ValueError(f"far_m must be greater than near_m, got {near_m}, {far_m}")
        self._resize_interpolation = resize_interpolation

    def _convert_units(self, depth: np.ndarray, encoding: str) -> np.ndarray:
        depth = np.asarray(depth)
        if depth.ndim == 3:
            depth = depth[..., 0]
        raw_dtype = depth.dtype
        depth = depth.astype(np.float32, copy=False)
        finite = np.isfinite(depth)
        median_raw = float(np.median(depth[finite])) if finite.any() else float("nan")
        units = self.depth_units
        if units == "auto":
            if (
                np.issubdtype(raw_dtype, np.integer)
                or "16U" in encoding
                or "mm" in encoding.lower()
                or (finite.any() and median_raw > 10.0)
            ):
                units = "mm"
            else:
                units = "m"
        if units == "mm":
            depth_m = depth / 1000.0
        elif units == "m":
            depth_m = depth.copy()
        else:
            raise ValueError(f"depth_units must be auto, m, or mm; got {self.depth_units!r}")

        finite_m = np.isfinite(depth_m)
        if finite_m.any():
            median_m = float(np.median(depth_m[finite_m]))
            p95_m = float(np.quantile(depth_m[finite_m], 0.95))
            if median_m > 5.0 or p95_m > 10.0:
                warn_every(
                    f"Depth median/p95 after unit conversion is suspicious: median={median_m:.3f}m p95={p95_m:.3f}m. "
                    "If ZED publishes millimeters, use --depth_units mm or auto.",
                    2.0,
                    key="depth_units_large",
                )
            if median_m < 0.05:
                warn_every(
                    f"Depth median after unit conversion is very small: {median_m:.4f}m. "
                    "If ZED publishes meters, do not use --depth_units mm.",
                    2.0,
                    key="depth_units_small",
                )
        return depth_m

    def _resize(self, depth_m: np.ndarray) -> np.ndarray:
        import cv2

        interp_map = {
            "area": cv2.INTER_AREA,
            "nearest": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
        }
        if self._resize_interpolation not in interp_map:
            raise ValueError(
                f"resize_interpolation must be one of {sorted(interp_map)}, "
                f"got {self._resize_interpolation!r}"
            )
        return cv2.resize(
            depth_m,
            (RESIZED_WIDTH, RESIZED_HEIGHT),
            interpolation=interp_map[self._resize_interpolation],
        ).astype(np.float32, copy=False)

    def __call__(self, depth: np.ndarray, *, encoding: str) -> DepthPipelineOutput:
        depth_m = self._convert_units(depth, encoding=encoding)
        resized_m = self._resize(depth_m)
        safe = np.nan_to_num(resized_m, nan=self.far_m, posinf=self.far_m, neginf=self.near_m)
        policy_full = np.clip((safe - self.near_m) / (self.far_m - self.near_m), 0.0, 1.0).astype(np.float32)
        crop = policy_full[CROP_Y0:CROP_Y1, CROP_X0:CROP_X1]
        if crop.shape != (POLICY_HEIGHT, POLICY_WIDTH):
            raise RuntimeError(f"Policy crop has shape {crop.shape}, expected {(POLICY_HEIGHT, POLICY_WIDTH)}")
        self._warn_if_depth_distribution_suspicious(resized_m=resized_m, crop_raw_m=resized_m[CROP_Y0:CROP_Y1, CROP_X0:CROP_X1])
        return DepthPipelineOutput(
            raw_depth_m=depth_m,
            resized_depth_m=resized_m,
            policy_full_depth=policy_full,
            policy_crop=crop,
        )

    def _warn_if_depth_distribution_suspicious(self, *, resized_m: np.ndarray, crop_raw_m: np.ndarray) -> None:
        finite = np.isfinite(resized_m)
        if finite.any():
            full_in_window = float(((resized_m >= self.near_m) & (resized_m <= self.far_m) & finite).mean())
            if full_in_window < 0.01:
                warn_every(
                    f"Only {100.0 * full_in_window:.1f}% of resized depth is in [{self.near_m:.2f}, {self.far_m:.2f}]m. "
                    "Check camera pose, units, and depth topic.",
                    2.0,
                    key="full_window_frac_low",
                )
        crop_finite = np.isfinite(crop_raw_m)
        if crop_finite.any():
            crop_in_window = float(((crop_raw_m >= self.near_m) & (crop_raw_m <= self.far_m) & crop_finite).mean())
            crop_median = float(np.median(crop_raw_m[crop_finite]))
            if crop_in_window < 0.02:
                warn_every(
                    f"Only {100.0 * crop_in_window:.1f}% of policy crop is in [{self.near_m:.2f}, {self.far_m:.2f}]m "
                    f"(crop median={crop_median:.3f}m). The policy may be seeing saturated depth.",
                    2.0,
                    key="crop_window_frac_low",
                )


class DepthDebugSaver:
    def __init__(self, output_dir: Optional[Path], *, every_n: int, video_path: Optional[Path], fps: int) -> None:
        self.output_dir = output_dir
        self.every_n = max(1, int(every_n))
        self.video_path = video_path
        self.fps = fps
        self.counter = 0
        self._writer = None
        if self.video_path is not None:
            self.video_path.parent.mkdir(parents=True, exist_ok=True)
            import imageio.v2 as imageio

            self._writer = imageio.get_writer(str(self.video_path), fps=self.fps)
            atexit.register(self.close)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def maybe_save(self, pipeline: DepthPipelineOutput) -> None:
        self.counter += 1
        if self.output_dir is None and self._writer is None:
            return
        if self.counter % self.every_n != 0:
            return
        grid = self._make_grid(pipeline)
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            prefix = self.output_dir / f"depth_{self.counter:06d}"
            np.savez_compressed(
                prefix.with_suffix(".npz"),
                raw_depth_m=pipeline.raw_depth_m,
                resized_depth_m=pipeline.resized_depth_m,
                policy_full_depth=pipeline.policy_full_depth,
                policy_crop=pipeline.policy_crop,
            )
            from PIL import Image as PILImage

            PILImage.fromarray(grid).save(prefix.with_name(prefix.name + "_grid.png"))
            PILImage.fromarray(self._gray_to_u8(self._window_metric(pipeline.raw_depth_m))).save(
                prefix.with_name(prefix.name + "_raw_depth_window.png")
            )
            PILImage.fromarray(self._gray_to_u8(self._window_metric(pipeline.resized_depth_m))).save(
                prefix.with_name(prefix.name + "_resized_depth_window.png")
            )
            PILImage.fromarray(self._gray_to_u8(pipeline.policy_full_depth)).save(
                prefix.with_name(prefix.name + "_policy_full_160x90.png")
            )
            PILImage.fromarray(self._gray_to_u8(pipeline.policy_crop)).save(
                prefix.with_name(prefix.name + "_policy_crop.png")
            )
        if self._writer is not None:
            self._writer.append_data(grid)

    @staticmethod
    def _gray_to_u8(image: np.ndarray) -> np.ndarray:
        return (np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8)

    def _window_metric(self, depth_m: np.ndarray) -> np.ndarray:
        return np.clip((np.nan_to_num(depth_m, nan=DEPTH_FAR_M, posinf=DEPTH_FAR_M, neginf=DEPTH_NEAR_M) - DEPTH_NEAR_M) / (DEPTH_FAR_M - DEPTH_NEAR_M), 0.0, 1.0)

    def _panel(
        self,
        image: np.ndarray,
        label: str,
        *,
        size: tuple[int, int] | None = (160, 90),
        pad_to: tuple[int, int] | None = None,
    ) -> np.ndarray:
        from PIL import Image as PILImage
        from PIL import ImageDraw

        if size is None:
            img = PILImage.fromarray(self._gray_to_u8(image))
            size = (img.width, img.height)
        elif image.shape[:2] != (size[1], size[0]):
            img = PILImage.fromarray(self._gray_to_u8(image)).resize(size, resample=PILImage.Resampling.NEAREST)
        else:
            img = PILImage.fromarray(self._gray_to_u8(image))
        img = img.convert("RGB")
        label_h = 16
        canvas_w, canvas_h = size
        if pad_to is not None:
            canvas_w = max(canvas_w, pad_to[0])
            canvas_h = max(canvas_h, pad_to[1])
        canvas = PILImage.new("RGB", (canvas_w, canvas_h + label_h), color=(255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        draw.text((3, 2), label, fill=(0, 0, 0))
        canvas.paste(img, (0, label_h))
        return np.asarray(canvas)

    def _make_grid(self, pipeline: DepthPipelineOutput) -> np.ndarray:
        raw_window = self._window_metric(pipeline.raw_depth_m)
        resized_window = self._window_metric(pipeline.resized_depth_m)
        panels = [
            self._panel(raw_window, "raw ZED window"),
            self._panel(resized_window, "resized 160x90"),
            self._panel(pipeline.policy_full_depth, "window normalized"),
            self._panel(pipeline.policy_crop, "policy crop 70x70", size=None, pad_to=(90, 90)),
        ]
        return np.concatenate(panels, axis=1)


class StudentDepthPolicyNode:
    def __init__(self, args: argparse.Namespace) -> None:
        rospy.init_node("student_depth_policy_node")
        self.args = args
        self.device = torch.device(args.device)
        self.policy, self.aux_heads, step, best_metric = load_student_policy(args.checkpoint_path, self.device)
        self.hidden = self.policy.initial_state(1, self.device)
        self.preprocessor = DepthPreprocessor(
            depth_units=args.depth_units,
            resize_interpolation=args.resize_interpolation,
        )
        self.debug_saver = DepthDebugSaver(
            args.debug_depth_dir,
            every_n=args.debug_depth_every_n,
            video_path=args.debug_depth_video_path,
            fps=args.debug_depth_video_fps,
        )

        self.prev_targets: Optional[np.ndarray] = None
        self.latest_depth_msg: Optional[Image] = None
        self.latest_depth_receive_time: Optional[rospy.Time] = None
        self.zed_camera: Optional[ZedDepthCamera] = None
        self.bridge = None
        self.depth_sub = None
        self.latest_iiwa_joint_state: Optional[JointState] = None
        self.latest_sharpa_joint_state: Optional[JointState] = None
        self.loop_count = 0
        self.active_loop_start_time: Optional[rospy.Time] = None
        self.command_start_time: Optional[rospy.Time] = None
        self.last_status_time = time.time()
        self._warmup_completed = False

        if args.depth_source == "ros_topic":
            from cv_bridge import CvBridge

            self.bridge = CvBridge()
            self.depth_sub = rospy.Subscriber(args.depth_topic, Image, self.depth_callback, queue_size=1)
        elif args.depth_source == "zed_sdk":
            self.zed_camera = ZedDepthCamera(args)
        else:
            raise ValueError(f"Unsupported --depth_source {args.depth_source!r}")

        self.iiwa_sub = rospy.Subscriber(args.iiwa_joint_state_topic, JointState, self.iiwa_callback, queue_size=1)
        self.sharpa_sub = rospy.Subscriber(args.sharpa_joint_state_topic, JointState, self.sharpa_callback, queue_size=1)
        self.object_pose_pub = rospy.Publisher(args.object_pose_topic, PoseStamped, queue_size=1)
        self.iiwa_cmd_pub = rospy.Publisher(args.iiwa_joint_cmd_topic, JointState, queue_size=1)
        self.sharpa_cmd_pub = rospy.Publisher(args.sharpa_joint_cmd_topic, JointState, queue_size=1)

        info(
            "Loaded student depth policy "
            f"checkpoint={args.checkpoint_path} step={step} best_metric={best_metric:.3f} "
            f"aux_heads={self.aux_heads} device={self.device}"
        )
        info(
            "Depth pipeline: ZED -> meters -> resize 160x90 -> "
            "window_normalize [0.70,1.10]m -> crop x[90:160], y[0:70] -> 1x70x70"
        )
        if args.depth_source == "zed_sdk":
            info("Depth source: direct ZED SDK capture. No ROS depth image topic is subscribed.")
        else:
            warn(f"Depth source: ROS image topic {args.depth_topic}. This can add network/load overhead.")
        if args.debug_depth_dir is not None:
            info(f"Depth debug PNG/NPZ output enabled: {args.debug_depth_dir}")
        if args.debug_depth_video_path is not None:
            info(f"Depth debug mp4 output enabled: {args.debug_depth_video_path}")
        if not args.publish_joint_commands:
            warn("Joint command publishing is disabled. Use --publish_joint_commands to send targets.")
        elif args.publish_joint_commands_duration_s >= 0.0:
            warn(f"Joint commands will publish only for {args.publish_joint_commands_duration_s:.2f}s.")
        if "object_rot6d" in self.aux_heads:
            info(f"Object pose publishing mode: full pose from object_pos + object_rot6d to {args.object_pose_topic}")
        elif "object_pos" in self.aux_heads:
            info(f"Object pose publishing mode: object_pos with fallback quat {tuple(args.position_only_quat_xyzw)}")
        else:
            warn("Checkpoint has no object_pos aux head; object pose publishing will only warn.")
        if args.publish_object_pose:
            info(
                f"Predicted object pose frame conversion: model_frame={args.predicted_pose_model_frame} "
                f"publish_frame={args.object_pose_frame_id}"
            )

    def depth_callback(self, msg: Image) -> None:
        self.latest_depth_msg = msg
        self.latest_depth_receive_time = rospy.Time.now()

    def iiwa_callback(self, msg: JointState) -> None:
        self.latest_iiwa_joint_state = msg

    def sharpa_callback(self, msg: JointState) -> None:
        self.latest_sharpa_joint_state = msg

    def _ready(self) -> bool:
        missing = []
        if self.args.depth_source == "ros_topic" and self.latest_depth_msg is None:
            missing.append("depth_topic")
        if self.args.depth_source == "zed_sdk" and self.zed_camera is None:
            missing.append("zed_sdk_depth")
        if self.latest_iiwa_joint_state is None:
            missing.append("iiwa_joint_states")
        if self.latest_sharpa_joint_state is None:
            missing.append("sharpa_joint_states")
        if missing:
            warn_every(f"Waiting for inputs: {', '.join(missing)}", 1.0, key="waiting_inputs")
            return False
        return True

    def _joint_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        assert self.latest_iiwa_joint_state is not None
        assert self.latest_sharpa_joint_state is not None
        iiwa_pos = np.asarray(self.latest_iiwa_joint_state.position, dtype=np.float32)
        sharpa_pos = np.asarray(self.latest_sharpa_joint_state.position, dtype=np.float32)
        iiwa_vel = np.asarray(self.latest_iiwa_joint_state.velocity, dtype=np.float32)
        sharpa_vel = np.asarray(self.latest_sharpa_joint_state.velocity, dtype=np.float32)
        if iiwa_pos.shape[0] != N_ARM or sharpa_pos.shape[0] != N_ACTIONS - N_ARM:
            raise RuntimeError(
                f"Expected iiwa/sharpa joint lengths 7/22, got {iiwa_pos.shape[0]}/{sharpa_pos.shape[0]}"
            )
        if iiwa_vel.shape[0] != N_ARM:
            iiwa_vel = np.zeros_like(iiwa_pos)
        if sharpa_vel.shape[0] != N_ACTIONS - N_ARM:
            sharpa_vel = np.zeros_like(sharpa_pos)
        return np.concatenate([iiwa_pos, sharpa_pos]), np.concatenate([iiwa_vel, sharpa_vel])

    def _proprio_tensor(self, q: np.ndarray, qd: np.ndarray) -> torch.Tensor:
        lower = Q_LOWER_LIMITS_restricted_np.astype(np.float32)
        upper = Q_UPPER_LIMITS_restricted_np.astype(np.float32)
        q_norm = 2.0 * (q - lower) / (upper - lower) - 1.0
        if self.prev_targets is None:
            self.prev_targets = q.copy()
        proprio = np.concatenate([q_norm, qd, self.prev_targets]).astype(np.float32)
        if proprio.shape != (87,):
            raise RuntimeError(f"Expected proprio shape (87,), got {proprio.shape}")
        if np.any(np.abs(q_norm) > 1.25):
            warn_every(
                f"Some normalized joint positions are outside expected range: min={q_norm.min():.2f}, max={q_norm.max():.2f}",
                2.0,
                key="joint_norm_range",
            )
        return torch.from_numpy(proprio).to(self.device).unsqueeze(0)

    def _read_depth_frame(self) -> DepthFrame:
        if self.args.depth_source == "ros_topic":
            assert self.latest_depth_msg is not None
            assert self.bridge is not None
            msg = self.latest_depth_msg
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            stamp = msg.header.stamp if msg.header.stamp != rospy.Time(0) else (
                self.latest_depth_receive_time or rospy.Time.now()
            )
            return DepthFrame(depth=depth, encoding=msg.encoding, stamp=stamp)
        if self.args.depth_source == "zed_sdk":
            assert self.zed_camera is not None
            frame = self.zed_camera.read()
            if frame is None:
                raise RuntimeError("Failed to read a depth frame from the ZED SDK.")
            return frame
        raise ValueError(f"Unsupported --depth_source {self.args.depth_source!r}")

    def _image_tensor(self, *, save_debug: bool = True) -> tuple[torch.Tensor, DepthPipelineOutput, rospy.Time]:
        frame = self._read_depth_frame()
        pipeline = self.preprocessor(frame.depth, encoding=frame.encoding)
        if save_debug:
            self.debug_saver.maybe_save(pipeline)
        image = torch.from_numpy(pipeline.policy_crop).to(self.device).float().view(1, 1, POLICY_HEIGHT, POLICY_WIDTH)
        return image, pipeline, frame.stamp

    def _predicted_pose(self, aux: dict[str, torch.Tensor]) -> tuple[np.ndarray, np.ndarray] | None:
        if "object_pos" not in aux:
            return None
        pos = aux["object_pos"][0].detach().cpu().numpy().astype(np.float64)
        pos = self._convert_predicted_pos_to_publish_frame(pos)
        if "object_rot6d" in aux:
            rot_m = rot6d_to_matrix(aux["object_rot6d"])[0].detach().cpu().numpy()
            quat_xyzw = R.from_matrix(rot_m).as_quat()
        else:
            quat_xyzw = np.array(self.args.position_only_quat_xyzw, dtype=np.float64)
            norm = np.linalg.norm(quat_xyzw)
            quat_xyzw = quat_xyzw / norm if norm > 1e-8 else np.array([0.0, 0.0, 0.0, 1.0])
        return pos, quat_xyzw

    def _convert_predicted_pos_to_publish_frame(self, pos: np.ndarray) -> np.ndarray:
        """Convert the learned aux position to the requested ROS frame.

        The Isaac Sim distillation target is env-local/world-local position.
        Real SimToolReal ROS pose topics under /robot_frame are robot-frame
        positions. In this setup the robot frame differs by only a translation.
        """

        model_frame = self.args.predicted_pose_model_frame
        publish_frame = self.args.object_pose_frame_id
        if model_frame == publish_frame:
            return pos
        if model_frame == "env" and publish_frame == "robot_frame":
            return pos - SIM_WORLD_T_ROBOT_POS_M
        if model_frame == "robot_frame" and publish_frame in {"env", "world"}:
            return pos + SIM_WORLD_T_ROBOT_POS_M
        warn_every(
            f"No explicit object pose conversion from model_frame={model_frame!r} to publish_frame={publish_frame!r}; "
            "publishing raw predicted position.",
            5.0,
            key="unsupported_predicted_pose_frame_conversion",
        )
        return pos

    def _publish_predicted_pose(self, aux: dict[str, torch.Tensor], stamp: rospy.Time) -> None:
        if not self.args.publish_object_pose:
            return
        pose = self._predicted_pose(aux)
        if pose is None:
            warn_every("Policy checkpoint has no object_pos aux head; cannot publish object pose.", 5.0)
            return
        pos, quat_xyzw = pose
        self.object_pose_pub.publish(
            pose_stamped_msg(
                pos=pos,
                quat_xyzw=quat_xyzw,
                frame_id=self.args.object_pose_frame_id,
                stamp=stamp,
            )
        )

    def _should_publish_joints(self) -> bool:
        if not self.args.publish_joint_commands:
            return False
        duration = float(self.args.publish_joint_commands_duration_s)
        if duration < 0.0:
            return True
        if self.command_start_time is None:
            self.command_start_time = rospy.Time.now()
        elapsed = (rospy.Time.now() - self.command_start_time).to_sec()
        if elapsed <= duration:
            return True
        warn_every("Joint command publishing duration elapsed; holding publishes disabled.", 5.0, key="joint_duration_done")
        return False

    def _publish_joint_targets(self, targets: np.ndarray) -> None:
        targets = targets.reshape(N_ACTIONS)
        stamp = rospy.Time.now()
        iiwa_msg = JointState()
        iiwa_msg.header.stamp = stamp
        iiwa_msg.name = [
            "iiwa_joint_1",
            "iiwa_joint_2",
            "iiwa_joint_3",
            "iiwa_joint_4",
            "iiwa_joint_5",
            "iiwa_joint_6",
            "iiwa_joint_7",
        ]
        iiwa_msg.position = targets[:N_ARM].tolist()
        self.iiwa_cmd_pub.publish(iiwa_msg)

        sharpa_msg = JointState()
        sharpa_msg.header.stamp = stamp
        sharpa_msg.name = [f"joint_{i}.0" for i in range(N_ACTIONS - N_ARM)]
        sharpa_msg.position = targets[N_ARM:].tolist()
        self.sharpa_cmd_pub.publish(sharpa_msg)

    def _targets_safe_to_publish(self, q: np.ndarray, targets: np.ndarray) -> bool:
        max_arm_delta_deg = float(self.args.max_arm_target_delta_deg)
        if max_arm_delta_deg <= 0.0:
            return True
        q_arm_diff_deg = np.rad2deg(np.abs(targets[:N_ARM] - q[:N_ARM]))
        if float(q_arm_diff_deg.max()) <= max_arm_delta_deg:
            return True
        message = (
            "Joint target safety guard blocked publish: "
            f"arm_delta_deg={np.round(q_arm_diff_deg, 2).tolist()} "
            f"max_allowed={max_arm_delta_deg:.1f}"
        )
        if self.args.raise_on_large_target_delta:
            raise RuntimeError(message)
        warn_every(message, 1.0, key="joint_target_safety")
        return False

    def _set_prev_targets_after_step(self, *, q: np.ndarray, q_targets: np.ndarray, published: bool) -> None:
        if published or self.args.prev_targets_when_not_publishing == "computed":
            self.prev_targets = q_targets.copy()
        elif self.args.prev_targets_when_not_publishing == "current":
            self.prev_targets = q.copy()
        else:
            raise ValueError(
                "--prev_targets_when_not_publishing must be 'current' or 'computed', "
                f"got {self.args.prev_targets_when_not_publishing!r}"
            )

    def _print_status(
        self,
        pipeline: DepthPipelineOutput,
        action: np.ndarray,
        q_targets: np.ndarray,
        prev_targets: np.ndarray,
        published: bool,
    ) -> None:
        now = time.time()
        if now - self.last_status_time < self.args.status_interval_s:
            return
        self.last_status_time = now
        crop_raw = pipeline.resized_depth_m[CROP_Y0:CROP_Y1, CROP_X0:CROP_X1]
        finite = np.isfinite(crop_raw)
        if finite.any():
            crop_med = float(np.median(crop_raw[finite]))
            crop_in = float(((crop_raw >= DEPTH_NEAR_M) & (crop_raw <= DEPTH_FAR_M) & finite).mean())
        else:
            crop_med = float("nan")
            crop_in = 0.0
        info(
            f"[student_depth_policy_node] step={self.loop_count} "
            f"crop_median={crop_med:.3f}m crop_in_window={100.0 * crop_in:.1f}% "
            f"action_abs_max={np.abs(action).max():.3f} "
            f"target_delta_abs_max={np.abs(q_targets - prev_targets).max():.3f} "
            f"published={published}"
        )

    def _wait_for_first_inputs(self) -> None:
        depth_label = "ZED SDK depth" if self.args.depth_source == "zed_sdk" else "ROS depth topic"
        info(f"Waiting for first {depth_label} + iiwa + sharpa observations...")
        rate = rospy.Rate(self.args.control_hz)
        while not rospy.is_shutdown():
            if self._ready():
                q, _ = self._joint_arrays()
                try:
                    # For direct ZED capture, force one grab here so startup only
                    # completes after the camera is actually producing depth.
                    _ = self._read_depth_frame()
                except Exception as exc:
                    warn_every(f"Waiting for first usable depth frame: {type(exc).__name__}: {exc}", 1.0)
                    rate.sleep()
                    continue
                self.prev_targets = q.copy()
                info("=" * 100)
                info("First observations received; initialized prev_targets from current joints.")
                info("=" * 100)
                return
            rate.sleep()

    def _warmup_policy(self) -> None:
        assert not self._warmup_completed, "Warmup already completed"
        num_steps = max(0, int(self.args.warmup_steps))
        if num_steps == 0:
            self.hidden = self.policy.initial_state(1, self.device)
            self._warmup_completed = True
            return

        info("=" * 100)
        info(
            f"Warming up student policy for {num_steps} steps. "
            "Policy outputs are not used for actions during warmup."
        )
        if self.args.warmup_publish_current_targets:
            warn("Warmup will publish current joint positions as hold targets.")
        info("=" * 100)

        rate = rospy.Rate(self.args.control_hz)
        last_q = None
        for step_idx in range(num_steps):
            if rospy.is_shutdown():
                return
            if not self._ready():
                rate.sleep()
                continue
            q, qd = self._joint_arrays()
            last_q = q.copy()
            self.prev_targets = q.copy()
            image, _, _ = self._image_tensor(save_debug=self.args.debug_depth_during_warmup)
            proprio = self._proprio_tensor(q, qd)
            with torch.no_grad():
                _, _ = self.policy(image, proprio, self.hidden)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            if self.args.warmup_publish_current_targets:
                self._publish_joint_targets(np.clip(q, Q_LOWER_LIMITS_restricted_np, Q_UPPER_LIMITS_restricted_np))
            if step_idx == 0 or (step_idx + 1) % 10 == 0 or step_idx + 1 == num_steps:
                info(f"Warmup step {step_idx + 1}/{num_steps}")
            rate.sleep()

        # Keep recurrent state clean for the actual deployment run.
        self.hidden = self.policy.initial_state(1, self.device)
        if last_q is not None:
            self.prev_targets = last_q.copy()
        self._warmup_completed = True
        info("=" * 100)
        info("Warmup complete; recurrent hidden state reset for real run.")
        info("=" * 100)

    def _active_run_elapsed_s(self) -> float:
        if self.active_loop_start_time is None:
            return 0.0
        return (rospy.Time.now() - self.active_loop_start_time).to_sec()

    def step(self) -> None:
        if not self._ready():
            return
        if self.args.run_duration_s >= 0.0 and self._active_run_elapsed_s() >= self.args.run_duration_s:
            info(f"Reached --run_duration_s={self.args.run_duration_s:.2f}; shutting down.")
            rospy.signal_shutdown("student depth policy run duration elapsed")
            return
        q, qd = self._joint_arrays()
        image, pipeline, stamp = self._image_tensor()
        proprio = self._proprio_tensor(q, qd)
        with torch.no_grad():
            output, self.hidden = self.policy(image, proprio, self.hidden)
        action = output.action.detach().cpu().numpy()
        if action.shape != (1, N_ACTIONS):
            raise RuntimeError(f"Expected action shape (1, 29), got {action.shape}")
        prev_targets = self.prev_targets.copy()
        q_targets = compute_restricted_joint_pos_targets(
            actions=action,
            prev_targets=prev_targets[None],
            hand_moving_average=self.args.hand_moving_average,
            arm_moving_average=self.args.arm_moving_average,
            dof_speed_scale=self.args.hand_dof_speed_scale,
            dt=1.0 / self.args.control_hz,
        )[0].astype(np.float32)
        published = False
        if self._should_publish_joints() and self._targets_safe_to_publish(q, q_targets):
            self._publish_joint_targets(q_targets)
            published = True
        self._set_prev_targets_after_step(q=q, q_targets=q_targets, published=published)
        self._publish_predicted_pose(output.aux, stamp)
        self._print_status(pipeline, action[0], q_targets, prev_targets, published)
        self.loop_count += 1

    def run(self) -> None:
        self._wait_for_first_inputs()
        self._warmup_policy()
        self.active_loop_start_time = rospy.Time.now()
        self.command_start_time = None
        rate = rospy.Rate(self.args.control_hz)
        while not rospy.is_shutdown():
            t0 = time.time()
            try:
                self.step()
            except Exception as exc:
                warn_every(f"Policy step failed: {type(exc).__name__}: {exc}", 1.0, key="step_failed")
                if self.args.raise_on_step_error:
                    raise
            elapsed = time.time() - t0
            if elapsed > 1.0 / self.args.control_hz:
                warn_every(
                    f"Policy loop cannot keep up: step took {1000.0 * elapsed:.1f} ms for {self.args.control_hz:.1f} Hz.",
                    1.0,
                    key="loop_slow",
                )
            rate.sleep()
        self.debug_saver.close()
        if self.zed_camera is not None:
            self.zed_camera.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_path", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--depth_source",
        choices=("zed_sdk", "ros_topic"),
        default="zed_sdk",
        help="Use direct ZED SDK capture by default to avoid streaming depth images over ROS.",
    )
    parser.add_argument("--depth_topic", default="/zed/zed_node/depth/depth_registered")
    parser.add_argument("--depth_units", choices=("auto", "m", "mm"), default="auto")
    parser.add_argument("--resize_interpolation", choices=("area", "nearest", "linear"), default="area")
    parser.add_argument("--zed_serial_number", default="15107")
    parser.add_argument("--zed_resolution", default="HD1080")
    parser.add_argument("--zed_depth_mode", default="NEURAL")
    parser.add_argument("--zed_camera_fps", type=int, default=30)
    parser.add_argument("--zed_camera_upsidedown", action="store_true")
    parser.add_argument("--iiwa_joint_state_topic", default="/iiwa/joint_states")
    parser.add_argument("--sharpa_joint_state_topic", default="/sharpa/joint_states")
    parser.add_argument("--iiwa_joint_cmd_topic", default="/iiwa/joint_cmd")
    parser.add_argument("--sharpa_joint_cmd_topic", default="/sharpa/joint_cmd")
    parser.add_argument("--publish_joint_commands", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--publish_joint_commands_duration_s", type=float, default=-1.0)
    parser.add_argument(
        "--prev_targets_when_not_publishing",
        choices=("current", "computed"),
        default="current",
        help=(
            "What to feed as prev_action_targets when no joint target was actually published. "
            "'current' is safest for dry-runs; 'computed' simulates the command history the policy would have sent."
        ),
    )
    parser.add_argument("--max_arm_target_delta_deg", type=float, default=10.0)
    parser.add_argument("--raise_on_large_target_delta", action="store_true")
    parser.add_argument("--warmup_steps", type=int, default=30)
    parser.add_argument(
        "--warmup_publish_current_targets",
        action="store_true",
        help="During warmup only, publish current sensed joint positions as hold targets instead of policy targets.",
    )
    parser.add_argument("--debug_depth_during_warmup", action="store_true")
    parser.add_argument(
        "--run_duration_s",
        type=float,
        default=-1.0,
        help="If non-negative, stop the policy loop after this many seconds after warmup.",
    )
    parser.add_argument("--publish_object_pose", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--object_pose_topic", default="/robot_frame/current_pose")
    parser.add_argument("--object_pose_frame_id", default="robot_frame")
    parser.add_argument(
        "--predicted_pose_model_frame",
        choices=("env", "robot_frame"),
        default="env",
        help="Frame used by the checkpoint's aux object_pos head. distill_depth.py trains env-local positions.",
    )
    parser.add_argument("--position_only_quat_xyzw", type=float, nargs=4, default=(0.0, 0.0, 0.0, 1.0))
    parser.add_argument("--control_hz", type=float, default=DEFAULT_CONTROL_HZ)
    parser.add_argument("--hand_moving_average", type=float, default=0.1)
    parser.add_argument("--arm_moving_average", type=float, default=0.1)
    parser.add_argument("--hand_dof_speed_scale", type=float, default=1.5)
    parser.add_argument("--debug_depth_dir", type=Path, default=None)
    parser.add_argument("--debug_depth_every_n", type=int, default=30)
    parser.add_argument("--debug_depth_video_path", type=Path, default=None)
    parser.add_argument("--debug_depth_video_fps", type=int, default=10)
    parser.add_argument("--status_interval_s", type=float, default=1.0)
    parser.add_argument("--raise_on_step_error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    node = StudentDepthPolicyNode(args)
    node.run()


if __name__ == "__main__":
    main()
