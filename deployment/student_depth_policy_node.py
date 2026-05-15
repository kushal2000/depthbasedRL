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
import datetime
import importlib.util
import multiprocessing as mp
import signal
import struct
import sys
import time
from dataclasses import dataclass
from multiprocessing import shared_memory
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
DEPTH_INVALID_M = 10.0
RESIZED_WIDTH = 160
RESIZED_HEIGHT = 90
CROP_X0 = 90
CROP_Y0 = 0
CROP_X1 = 160
CROP_Y1 = 70
POLICY_WIDTH = CROP_X1 - CROP_X0
POLICY_HEIGHT = CROP_Y1 - CROP_Y0
DEFAULT_CONTROL_HZ = 60.0
STARTUP_POLICY_BENCHMARK_STEPS = 30
_ZED_META_FMT = "<qdqddddddqii"
_ZED_META_SIZE = struct.calcsize(_ZED_META_FMT)
_ZED_TS_NONE = -1
# distill_depth.py trains object_pos in the env-local/world frame. The real
# robot ROS topics use robot_frame, whose origin is translated +0.8 m in sim.
SIM_WORLD_T_ROBOT_POS_M = np.array([0.0, 0.8, 0.0], dtype=np.float64)

# Fixed real-robot deployment defaults. Keep routine run commands focused on
# operational choices rather than wiring details.
DEPTH_TOPIC = "/zed/zed_node/depth/depth_registered"
DEPTH_UNITS = "auto"
RESIZE_INTERPOLATION = "nearest"
ZED_SERIAL_NUMBER = "15107"
ZED_RESOLUTION = "HD1080"
ZED_DEPTH_MODE = "NEURAL"
ZED_CAMERA_FPS = 30
ZED_GRAB_HZ = 30.0
ZED_EXPOSURE = 25
ZED_GAIN = 40
MAX_ARM_TARGET_DELTA_DEG = 0.0
HAND_MOVING_AVERAGE = 0.1
ARM_MOVING_AVERAGE = 0.1
DOF_SPEED_SCALE = 1.5
IIWA_JOINT_STATE_TOPIC = "/iiwa/joint_states"
SHARPA_JOINT_STATE_TOPIC = "/sharpa/joint_states"
IIWA_JOINT_CMD_TOPIC = "/iiwa/joint_cmd"
SHARPA_JOINT_CMD_TOPIC = "/sharpa/joint_cmd"
OBJECT_POSE_TOPIC = "/robot_frame/current_object_pose"
OBJECT_POSE_FRAME_ID = "robot_frame"
PREDICTED_POSE_MODEL_FRAME = "env"

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
assert Q_LOWER_LIMITS_np.shape == (N_ACTIONS,)
assert Q_UPPER_LIMITS_np.shape == (N_ACTIONS,)


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


def compute_joint_pos_targets(
    *,
    actions: np.ndarray,
    prev_targets: np.ndarray,
    hand_moving_average: float,
    arm_moving_average: float,
    dof_speed_scale: float,
    dt: float,
) -> np.ndarray:
    """Match Isaac Lab's current action pipeline with full URDF joint limits."""

    if actions.ndim != 2 or actions.shape[1] != N_ACTIONS:
        raise RuntimeError(f"Expected actions shape (N, 29), got {actions.shape}")
    if prev_targets.shape != actions.shape:
        raise RuntimeError(f"prev_targets shape {prev_targets.shape} does not match actions {actions.shape}")
    lower = Q_LOWER_LIMITS_np.astype(np.float32)
    upper = Q_UPPER_LIMITS_np.astype(np.float32)
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
    frame_id: int = -1
    receive_time: rospy.Time | None = None
    grab_ms: float | None = None
    retrieve_ms: float | None = None
    copy_ms: float | None = None
    resize_ms: float | None = None
    total_ms: float | None = None
    inter_frame_ms: float | None = None
    failures: int = 0


def _zed_enum_value(enum_cls, name: str):
    try:
        return getattr(enum_cls, str(name).upper())
    except AttributeError as exc:
        valid = [item for item in dir(enum_cls) if item.isupper()]
        raise ValueError(f"Invalid ZED enum value {name!r}; valid values include {valid}") from exc


def _format_zed_camera_info(camera) -> str:
    """Return SDK-confirmed resolution/FPS/intrinsics after camera.open()."""
    try:
        camera_info = camera.get_camera_information()
        camera_cfg = camera_info.camera_configuration
        resolution = camera_cfg.resolution
        calib = camera_cfg.calibration_parameters
        left = calib.left_cam
        return (
            f"opened={int(resolution.width)}x{int(resolution.height)}@{int(camera_cfg.fps)}Hz "
            f"K_left=[[{float(left.fx):.3f},0,{float(left.cx):.3f}],"
            f"[0,{float(left.fy):.3f},{float(left.cy):.3f}],[0,0,1]] "
            f"dist_left={[float(x) for x in left.disto]}"
        )
    except Exception as exc:
        return f"opened_camera_info_unavailable={type(exc).__name__}: {exc}"


def _resize_zed_depth_for_policy_cache(depth_mm: np.ndarray) -> np.ndarray:
    if depth_mm.ndim == 3:
        depth_mm = depth_mm[..., 0]
    depth_mm = np.asarray(depth_mm, dtype=np.float32)
    if depth_mm.shape == (RESIZED_HEIGHT, RESIZED_WIDTH):
        return depth_mm
    import cv2

    return cv2.resize(
        depth_mm,
        (RESIZED_WIDTH, RESIZED_HEIGHT),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.float32, copy=False)


def _zed_shared_memory_producer_main(
    args_dict: dict,
    meta_shm_name: str,
    depth_shm_name: str,
    lock,
    stop_event,
) -> None:
    try:
        import pyzed.sl as sl
    except ModuleNotFoundError as exc:
        raise RuntimeError("pyzed.sl is required in the ZED producer process.") from exc

    args = argparse.Namespace(**args_dict)
    meta_shm = shared_memory.SharedMemory(name=meta_shm_name)
    depth_shm = shared_memory.SharedMemory(name=depth_shm_name)
    camera = sl.Camera()
    depth_mat = sl.Mat()
    retrieve_resolution = None
    low_res_retrieve_failed = False
    last_frame_wall_time_s = None
    runtime_parameters = sl.RuntimeParameters()
    grab_period_s = 1.0 / float(args.zed_grab_hz) if float(args.zed_grab_hz) > 0.0 else 0.0

    try:
        init_params = sl.InitParameters(input_t=sl.InputType())
        init_params.svo_real_time_mode = True
        init_params.camera_resolution = _zed_enum_value(sl.RESOLUTION, args.zed_resolution)
        init_params.depth_mode = _zed_enum_value(sl.DEPTH_MODE, args.zed_depth_mode)
        init_params.coordinate_units = sl.UNIT.MILLIMETER
        if int(args.zed_camera_fps) > 0:
            init_params.camera_fps = int(args.zed_camera_fps)
        if args.zed_serial_number:
            init_params.set_from_serial_number(int(args.zed_serial_number))

        err = camera.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED camera serial={args.zed_serial_number!r}: {err}")
        if int(args.zed_exposure) >= 0:
            camera.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, int(args.zed_exposure))
        if int(args.zed_gain) >= 0:
            camera.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, int(args.zed_gain))
        if int(args.zed_retrieve_width) > 0 and int(args.zed_retrieve_height) > 0:
            retrieve_resolution = sl.Resolution(int(args.zed_retrieve_width), int(args.zed_retrieve_height))

        info(
            "ZED producer process started "
            f"serial={args.zed_serial_number or '<default>'} resolution={args.zed_resolution} "
            f"depth_mode={args.zed_depth_mode} camera_fps={args.zed_camera_fps} "
            f"grab_hz_cap={args.zed_grab_hz if float(args.zed_grab_hz) > 0.0 else 'none'} "
            f"retrieve={args.zed_retrieve_width}x{args.zed_retrieve_height} "
            f"{_format_zed_camera_info(camera)}"
        )

        frame_id = 0
        fail_count = 0
        while not stop_event.is_set():
            t0 = time.time()
            err = camera.grab(runtime_parameters)
            t_grab = time.time()
            if err != sl.ERROR_CODE.SUCCESS:
                fail_count += 1
                with lock:
                    meta = struct.unpack_from(_ZED_META_FMT, meta_shm.buf)
                    struct.pack_into(
                        _ZED_META_FMT,
                        meta_shm.buf,
                        0,
                        *meta[:9],
                        fail_count,
                        *meta[10:],
                    )
                stop_event.wait(min(grab_period_s, 0.01) if grab_period_s > 0.0 else 0.001)
                continue

            if retrieve_resolution is None or low_res_retrieve_failed:
                camera.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
            else:
                try:
                    camera.retrieve_measure(depth_mat, sl.MEASURE.DEPTH, sl.MEM.CPU, retrieve_resolution)
                except TypeError:
                    low_res_retrieve_failed = True
                    info("ZED low-resolution retrieve_measure overload failed; falling back to full-frame retrieve.")
                    camera.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
            t_retrieve = time.time()

            depth_mm = np.array(depth_mat.get_data(), copy=True)
            if args.zed_camera_upsidedown:
                import cv2

                depth_mm = cv2.flip(depth_mm, -1)
            t_copy = time.time()
            depth_mm = _resize_zed_depth_for_policy_cache(depth_mm)
            t_resize = time.time()

            zed_ts_ms = _ZED_TS_NONE
            try:
                zed_ts_ms = int(camera.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds())
            except Exception:
                pass

            camera_period_ms = float("nan")
            if last_frame_wall_time_s is not None:
                camera_period_ms = 1000.0 * (t_resize - last_frame_wall_time_s)
            last_frame_wall_time_s = t_resize

            rows, cols = depth_mm.shape[:2]
            depth_bytes = depth_mm.astype(np.float32, copy=False).tobytes()
            with lock:
                depth_shm.buf[: len(depth_bytes)] = depth_bytes
                struct.pack_into(
                    _ZED_META_FMT,
                    meta_shm.buf,
                    0,
                    frame_id,
                    time.time(),
                    zed_ts_ms,
                    1000.0 * (t_grab - t0),
                    1000.0 * (t_retrieve - t_grab),
                    1000.0 * (t_copy - t_retrieve),
                    1000.0 * (t_resize - t_copy),
                    1000.0 * (t_resize - t0),
                    camera_period_ms,
                    fail_count,
                    rows,
                    cols,
                )
            frame_id += 1

            elapsed_s = time.time() - t0
            if grab_period_s > elapsed_s:
                stop_event.wait(grab_period_s - elapsed_s)
    finally:
        camera.close()
        meta_shm.close()
        depth_shm.close()


class ZedDepthCamera:
    """Direct ZED SDK depth capture without streaming depth images through ROS."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.nonblocking = bool(args.zed_nonblocking)
        self.max_cached_depth_age_s = float(args.zed_max_cached_depth_age_s)
        self.camera = None
        self.sl = None
        self.runtime_parameters = None
        self.depth_mat = None
        self.camera_upsidedown = bool(args.zed_camera_upsidedown)
        self.retrieve_resolution = None
        self._frame_id = 0
        self._last_frame_wall_time: float | None = None
        self._mp_ctx = None
        self._meta_shm = None
        self._depth_shm = None
        self._lock = None
        self._stop_event = None
        self._process = None

        if self.nonblocking:
            self._start_shared_memory_process(args)
        else:
            self._open_direct_camera(args)
        atexit.register(self.close)

    def _open_direct_camera(self, args: argparse.Namespace) -> None:
        try:
            import pyzed.sl as sl
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "pyzed.sl is required for --depth_source zed_sdk. "
                "Use a shell/environment with the ZED SDK Python bindings, or pass --depth_source ros_topic."
            ) from exc

        self.sl = sl
        self.camera = sl.Camera()
        init_params = sl.InitParameters(input_t=sl.InputType())
        init_params.svo_real_time_mode = True
        init_params.camera_resolution = _zed_enum_value(sl.RESOLUTION, args.zed_resolution)
        init_params.depth_mode = _zed_enum_value(sl.DEPTH_MODE, args.zed_depth_mode)
        init_params.coordinate_units = sl.UNIT.MILLIMETER
        if int(args.zed_camera_fps) > 0:
            init_params.camera_fps = int(args.zed_camera_fps)
        if args.zed_serial_number:
            init_params.set_from_serial_number(int(args.zed_serial_number))

        err = self.camera.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED camera serial={args.zed_serial_number!r}: {err}")
        if int(args.zed_exposure) >= 0:
            self.camera.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, int(args.zed_exposure))
        if int(args.zed_gain) >= 0:
            self.camera.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, int(args.zed_gain))

        self.runtime_parameters = sl.RuntimeParameters()
        self.depth_mat = sl.Mat()
        if int(args.zed_retrieve_width) > 0 and int(args.zed_retrieve_height) > 0:
            self.retrieve_resolution = sl.Resolution(int(args.zed_retrieve_width), int(args.zed_retrieve_height))
        self._log_camera_config(args, mode="blocking direct", camera=self.camera)

    def _start_shared_memory_process(self, args: argparse.Namespace) -> None:
        self._mp_ctx = mp.get_context("spawn")
        self._lock = self._mp_ctx.Lock()
        self._stop_event = self._mp_ctx.Event()
        self._meta_shm = shared_memory.SharedMemory(create=True, size=_ZED_META_SIZE)
        self._meta_shm.buf[:_ZED_META_SIZE] = b"\x00" * _ZED_META_SIZE
        self._depth_shm = shared_memory.SharedMemory(
            create=True,
            size=RESIZED_HEIGHT * RESIZED_WIDTH * np.dtype(np.float32).itemsize,
        )
        args_dict = {
            "zed_serial_number": args.zed_serial_number,
            "zed_resolution": args.zed_resolution,
            "zed_depth_mode": args.zed_depth_mode,
            "zed_camera_fps": args.zed_camera_fps,
            "zed_grab_hz": args.zed_grab_hz,
            "zed_exposure": args.zed_exposure,
            "zed_gain": args.zed_gain,
            "zed_camera_upsidedown": args.zed_camera_upsidedown,
            "zed_retrieve_width": args.zed_retrieve_width,
            "zed_retrieve_height": args.zed_retrieve_height,
        }
        self._process = self._mp_ctx.Process(
            target=_zed_shared_memory_producer_main,
            args=(args_dict, self._meta_shm.name, self._depth_shm.name, self._lock, self._stop_event),
            daemon=True,
            name="zed_depth_producer",
        )
        self._process.start()
        self._log_camera_config(args, mode="nonblocking shared-memory subprocess")

    @staticmethod
    def _log_camera_config(args: argparse.Namespace, *, mode: str, camera=None) -> None:
        info(
            "Opened ZED SDK depth camera "
            f"mode={mode} serial={args.zed_serial_number or '<default>'} resolution={args.zed_resolution} "
            f"depth_mode={args.zed_depth_mode} units=millimeters "
            f"exposure={args.zed_exposure} gain={args.zed_gain}"
        )
        if camera is not None:
            info(f"ZED actual camera info: {_format_zed_camera_info(camera)}")
        if bool(args.zed_nonblocking) and float(args.zed_grab_hz) > 0.0:
            info(f"ZED producer rate cap: {float(args.zed_grab_hz):.1f} Hz")
        if int(args.zed_retrieve_width) > 0 and int(args.zed_retrieve_height) > 0:
            info(f"ZED retrieve resolution: {int(args.zed_retrieve_width)}x{int(args.zed_retrieve_height)}")
        info(f"ZED cached depth frame shape for policy preprocessing: {RESIZED_WIDTH}x{RESIZED_HEIGHT}")

    def close(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._process is not None and self._process.is_alive():
            self._process.join(timeout=2.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1.0)
        if self.camera is not None:
            self.camera.close()
            self.camera = None
        if self._meta_shm is not None:
            self._meta_shm.close()
            try:
                self._meta_shm.unlink()
            except FileNotFoundError:
                pass
            self._meta_shm = None
        if self._depth_shm is not None:
            self._depth_shm.close()
            try:
                self._depth_shm.unlink()
            except FileNotFoundError:
                pass
            self._depth_shm = None

    def _grab_once(self) -> DepthFrame | None:
        assert self.camera is not None
        assert self.sl is not None
        assert self.runtime_parameters is not None
        assert self.depth_mat is not None
        t0 = time.time()
        err = self.camera.grab(self.runtime_parameters)
        t_grab = time.time()
        if err != self.sl.ERROR_CODE.SUCCESS:
            warn_every(f"ZED grab failed: {err}", 1.0, key="zed_grab_failed")
            return None
        if self.retrieve_resolution is None:
            self.camera.retrieve_measure(self.depth_mat, self.sl.MEASURE.DEPTH)
        else:
            try:
                self.camera.retrieve_measure(
                    self.depth_mat,
                    self.sl.MEASURE.DEPTH,
                    self.sl.MEM.CPU,
                    self.retrieve_resolution,
                )
            except TypeError:
                warn_every(
                    "ZED Python API did not accept low-resolution retrieve_measure; falling back to full frame.",
                    5.0,
                    key="zed_retrieve_resolution_unsupported",
                )
                self.camera.retrieve_measure(self.depth_mat, self.sl.MEASURE.DEPTH)
        t_retrieve = time.time()
        depth_mm = np.array(self.depth_mat.get_data(), copy=True)
        if self.camera_upsidedown:
            import cv2

            depth_mm = cv2.flip(depth_mm, -1)
        t_copy = time.time()
        depth_mm = _resize_zed_depth_for_policy_cache(depth_mm)
        now = time.time()
        inter_frame_ms = None
        if self._last_frame_wall_time is not None:
            inter_frame_ms = 1000.0 * (now - self._last_frame_wall_time)
        self._last_frame_wall_time = now
        frame_id = self._frame_id
        self._frame_id += 1
        return DepthFrame(
            depth=depth_mm,
            encoding="zed_sdk_mm",
            stamp=rospy.Time.now(),
            frame_id=frame_id,
            grab_ms=1000.0 * (t_grab - t0),
            retrieve_ms=1000.0 * (t_retrieve - t_grab),
            copy_ms=1000.0 * (t_copy - t_retrieve),
            resize_ms=1000.0 * (now - t_copy),
            total_ms=1000.0 * (now - t0),
            inter_frame_ms=inter_frame_ms,
        )

    def _read_shared_memory_frame(self) -> DepthFrame | None:
        assert self._meta_shm is not None
        assert self._depth_shm is not None
        assert self._lock is not None
        with self._lock:
            meta = struct.unpack_from(_ZED_META_FMT, self._meta_shm.buf)
            (
                frame_id,
                wall_time_s,
                zed_ts_ms,
                grab_ms,
                retrieve_ms,
                copy_ms,
                resize_ms,
                total_ms,
                camera_period_ms,
                failures,
                rows,
                cols,
            ) = meta
            if int(rows) <= 0 or int(cols) <= 0 or int(frame_id) < 0:
                return None
            depth_mm = np.ndarray(
                (int(rows), int(cols)),
                dtype=np.float32,
                buffer=self._depth_shm.buf,
            ).copy()
        stamp = rospy.Time.from_sec(float(wall_time_s)) if wall_time_s > 0.0 else rospy.Time.now()
        inter_frame_ms = None if np.isnan(camera_period_ms) else float(camera_period_ms)
        return DepthFrame(
            depth=depth_mm,
            encoding="zed_sdk_mm",
            stamp=stamp,
            frame_id=int(frame_id),
            grab_ms=float(grab_ms),
            retrieve_ms=float(retrieve_ms),
            copy_ms=float(copy_ms),
            resize_ms=float(resize_ms),
            total_ms=float(total_ms),
            inter_frame_ms=inter_frame_ms,
            failures=int(failures),
        )

    def read(self) -> DepthFrame | None:
        if not self.nonblocking:
            return self._grab_once()
        frame = self._read_shared_memory_frame()
        if frame is None:
            if self._process is not None and not self._process.is_alive():
                warn_every(
                    f"ZED producer process is not running; exitcode={self._process.exitcode}.",
                    1.0,
                    key="zed_process_dead",
                )
            warn_every("Waiting for first cached ZED depth frame from producer process.", 1.0, key="zed_waiting_cached_frame")
            return None
        age_s = max(0.0, (rospy.Time.now() - frame.stamp).to_sec())
        if self.max_cached_depth_age_s > 0.0 and age_s > self.max_cached_depth_age_s:
            warn_every(
                f"Cached ZED depth frame is stale: age={1000.0 * age_s:.1f} ms "
                f"> {1000.0 * self.max_cached_depth_age_s:.1f} ms.",
                1.0,
                key="zed_cached_frame_stale",
            )
        if frame.failures > 0:
            warn_every(f"ZED producer process has reported {frame.failures} grab failures.", 2.0, key="zed_failures")
        return frame


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

    @staticmethod
    def _sample_for_stats(depth: np.ndarray) -> np.ndarray:
        if depth.ndim != 2:
            return depth.reshape(-1)
        stride_y = max(1, depth.shape[0] // 120)
        stride_x = max(1, depth.shape[1] // 160)
        return depth[::stride_y, ::stride_x].reshape(-1)

    def _convert_units(self, depth: np.ndarray, encoding: str) -> np.ndarray:
        depth = np.asarray(depth)
        if depth.ndim == 3:
            depth = depth[..., 0]
        raw_dtype = depth.dtype
        depth = depth.astype(np.float32, copy=False)
        sample = self._sample_for_stats(depth)
        finite_sample = np.isfinite(sample)
        units = self.depth_units
        if units == "auto":
            if (
                np.issubdtype(raw_dtype, np.integer)
                or "16U" in encoding
                or "mm" in encoding.lower()
                or (finite_sample.any() and float(np.median(sample[finite_sample])) > 10.0)
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
        finite_depth = np.isfinite(depth_m)
        depth_m[finite_depth & (depth_m < 0.001)] = 0.0
        depth_m[~finite_depth] = DEPTH_INVALID_M

        sample_m = self._sample_for_stats(depth_m)
        valid_sample = np.isfinite(sample_m) & (sample_m > 0.0)
        if valid_sample.any():
            sample_finite_m = sample_m[valid_sample]
            median_m = float(np.median(sample_finite_m))
            p95_m = float(np.quantile(sample_finite_m, 0.95))
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
        if depth_m.shape == (RESIZED_HEIGHT, RESIZED_WIDTH):
            return depth_m.astype(np.float32, copy=False)
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


class StudentRolloutLogger:
    """In-memory rollout logger that writes once on shutdown."""

    def __init__(
        self,
        output_dir: Optional[Path],
        *,
        name: Optional[str],
        every_n: int,
        depth_format: str,
    ) -> None:
        self.output_dir = output_dir
        self.name = name
        self.every_n = max(1, int(every_n))
        self.depth_format = depth_format
        self.enabled = output_dir is not None
        self._saved = False
        self.step_indices: list[int] = []
        self.time_s: list[float] = []
        self.q: list[np.ndarray] = []
        self.qd: list[np.ndarray] = []
        self.proprio: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.q_targets: list[np.ndarray] = []
        self.prev_targets: list[np.ndarray] = []
        self.published: list[bool] = []
        self.policy_depth: list[np.ndarray] = []
        self.crop_raw_depth_m: list[np.ndarray] = []
        self.depth_stamp_s: list[float] = []
        self.depth_age_s: list[float] = []
        self.depth_pub_to_callback_s: list[float] = []
        self.depth_reused: list[bool] = []
        self.predicted_object_pos: list[np.ndarray] = []
        self.predicted_object_quat_xyzw: list[np.ndarray] = []

    def _encode_depth(self, depth: np.ndarray) -> np.ndarray:
        if self.depth_format == "none":
            return np.empty((0,), dtype=np.uint8)
        if self.depth_format == "float16":
            return np.asarray(depth, dtype=np.float16)
        if self.depth_format == "uint8":
            return (np.clip(depth, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        raise ValueError(f"Unsupported depth_format={self.depth_format!r}")

    def maybe_record(
        self,
        *,
        step: int,
        run_start_time: rospy.Time | None,
        q: np.ndarray,
        qd: np.ndarray,
        proprio: np.ndarray,
        pipeline: DepthPipelineOutput,
        action: np.ndarray,
        q_targets: np.ndarray,
        prev_targets: np.ndarray,
        published: bool,
        depth_stamp: rospy.Time,
        depth_age_s: float | None,
        depth_pub_to_callback_s: float | None,
        depth_reused: bool,
        predicted_pose: tuple[np.ndarray, np.ndarray] | None,
    ) -> None:
        if not self.enabled or step % self.every_n != 0:
            return
        if run_start_time is None:
            elapsed = 0.0
        else:
            elapsed = max(0.0, (rospy.Time.now() - run_start_time).to_sec())
        self.step_indices.append(int(step))
        self.time_s.append(float(elapsed))
        self.q.append(q.astype(np.float32, copy=True))
        self.qd.append(qd.astype(np.float32, copy=True))
        self.proprio.append(proprio.astype(np.float32, copy=True))
        self.actions.append(action.astype(np.float32, copy=True))
        self.q_targets.append(q_targets.astype(np.float32, copy=True))
        self.prev_targets.append(prev_targets.astype(np.float32, copy=True))
        self.published.append(bool(published))
        self.policy_depth.append(self._encode_depth(pipeline.policy_crop))
        crop_raw = pipeline.resized_depth_m[CROP_Y0:CROP_Y1, CROP_X0:CROP_X1]
        self.crop_raw_depth_m.append(crop_raw.astype(np.float16, copy=True))
        self.depth_stamp_s.append(float(depth_stamp.to_sec()))
        self.depth_age_s.append(float("nan") if depth_age_s is None else float(depth_age_s))
        self.depth_pub_to_callback_s.append(
            float("nan") if depth_pub_to_callback_s is None else float(depth_pub_to_callback_s)
        )
        self.depth_reused.append(bool(depth_reused))
        if predicted_pose is None:
            self.predicted_object_pos.append(np.full(3, np.nan, dtype=np.float32))
            self.predicted_object_quat_xyzw.append(np.full(4, np.nan, dtype=np.float32))
        else:
            pos, quat = predicted_pose
            self.predicted_object_pos.append(pos.astype(np.float32, copy=True))
            self.predicted_object_quat_xyzw.append(quat.astype(np.float32, copy=True))

    def save(self, *, checkpoint_path: Path | None = None) -> Path | None:
        if not self.enabled or self._saved:
            return None
        self._saved = True
        if not self.step_indices:
            warn("Rollout recording enabled but no steps were recorded.")
            return None
        assert self.output_dir is not None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        stem = self.name or "student_depth_rollout"
        path = self.output_dir / f"{stamp}_{stem}.npz"
        np.savez_compressed(
            path,
            step_indices=np.asarray(self.step_indices, dtype=np.int64),
            time_s=np.asarray(self.time_s, dtype=np.float64),
            q=np.stack(self.q),
            qd=np.stack(self.qd),
            proprio=np.stack(self.proprio),
            actions=np.stack(self.actions),
            q_targets=np.stack(self.q_targets),
            prev_targets=np.stack(self.prev_targets),
            published=np.asarray(self.published, dtype=bool),
            policy_depth=np.stack(self.policy_depth),
            policy_depth_format=np.asarray(self.depth_format),
            crop_raw_depth_m=np.stack(self.crop_raw_depth_m),
            depth_stamp_s=np.asarray(self.depth_stamp_s, dtype=np.float64),
            depth_age_s=np.asarray(self.depth_age_s, dtype=np.float64),
            depth_pub_to_callback_s=np.asarray(self.depth_pub_to_callback_s, dtype=np.float64),
            depth_reused=np.asarray(self.depth_reused, dtype=bool),
            predicted_object_pos=np.stack(self.predicted_object_pos),
            predicted_object_quat_xyzw=np.stack(self.predicted_object_quat_xyzw),
            checkpoint_path=np.asarray(str(checkpoint_path) if checkpoint_path is not None else ""),
            depth_near_m=np.asarray(DEPTH_NEAR_M, dtype=np.float32),
            depth_far_m=np.asarray(DEPTH_FAR_M, dtype=np.float32),
            crop_xyxy=np.asarray([CROP_X0, CROP_Y0, CROP_X1, CROP_Y1], dtype=np.int64),
        )
        info(f"Saved student rollout recording: {path}")
        return path


class StudentDepthPolicyNode:
    def __init__(self, args: argparse.Namespace) -> None:
        rospy.init_node("student_depth_policy_node")
        self.args = args
        self.device = torch.device(args.device)
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except AttributeError:
                pass
            info(f"CUDA policy device: {torch.cuda.get_device_name(self.device)}")
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
        self.rollout_logger = StudentRolloutLogger(
            args.record_rollout_dir,
            name=args.record_rollout_name,
            every_n=args.record_every_n,
            depth_format=args.record_depth_format,
        )
        if self.rollout_logger.enabled:
            atexit.register(self._save_rollout_recording)
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

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
        self.last_depth_age_s: float | None = None
        self.last_depth_pub_to_callback_s: float | None = None
        self.last_step_timing_ms: dict[str, float] = {}
        self.last_depth_timing_ms: dict[str, float] = {}
        self.last_depth_stamp: rospy.Time | None = None
        self.last_depth_frame_id: int = -1
        self.last_zed_grab_ms: float | None = None
        self.last_zed_retrieve_ms: float | None = None
        self.last_zed_copy_ms: float | None = None
        self.last_zed_resize_ms: float | None = None
        self.last_zed_total_ms: float | None = None
        self.last_zed_inter_frame_ms: float | None = None
        self.last_zed_failures: int = 0
        self.depth_frame_reused = False
        self.cached_depth_stamp: rospy.Time | None = None
        self.cached_depth_frame_id: int = -1
        self.cached_depth_pipeline: DepthPipelineOutput | None = None
        self.cached_depth_image: torch.Tensor | None = None
        self.last_proprio_np: np.ndarray | None = None
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
            if args.zed_nonblocking:
                info(
                    "ZED timing: non-blocking shared-memory subprocess enabled. "
                    "Policy loop may reuse a 30 Hz depth frame while running faster than camera FPS."
                )
        else:
            warn(f"Depth source: ROS image topic {args.depth_topic}. This can add network/load overhead.")
        if args.debug_depth_dir is not None:
            info(f"Depth debug PNG/NPZ output enabled: {args.debug_depth_dir}")
        if args.debug_depth_video_path is not None:
            info(f"Depth debug mp4 output enabled: {args.debug_depth_video_path}")
            warn(
                "Depth debug mp4 writing can add control-loop jitter. "
                "Disable --debug_depth_video_path for timing-critical joint publishing."
            )
        if self.rollout_logger.enabled:
            info(
                "Rollout recording enabled: "
                f"dir={args.record_rollout_dir} every_n={args.record_every_n} "
                f"depth_format={args.record_depth_format}. Data writes once on shutdown."
            )
        if not args.publish_joint_commands:
            warn("Joint command publishing is disabled. Use --publish_joint_commands to send targets.")
        elif args.publish_joint_commands_duration_s >= 0.0:
            warn(f"Joint commands will publish only for {args.publish_joint_commands_duration_s:.2f}s.")
        if args.publish_joint_commands:
            if args.max_arm_target_delta_deg > 0.0:
                info(
                    "Joint safety: targets clipped to full URDF limits; "
                    f"publishes blocked if any arm target is >{args.max_arm_target_delta_deg:.1f} deg from current joint."
                )
            else:
                info("Joint safety: targets clipped to full URDF limits; arm delta guard is disabled.")
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

    def _save_rollout_recording(self) -> None:
        self.rollout_logger.save(checkpoint_path=self.args.checkpoint_path)

    def _signal_handler(self, signum, _frame) -> None:
        info(f"Received signal {signum}; saving rollout recording before shutdown.")
        self._save_rollout_recording()
        rospy.signal_shutdown(f"signal {signum}")

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
        # Training normalizes joint_pos with the full URDF joint limits.
        # Deployment command safety is handled separately before publishing.
        lower = Q_LOWER_LIMITS_np.astype(np.float32)
        upper = Q_UPPER_LIMITS_np.astype(np.float32)
        q_norm = 2.0 * (q - lower) / (upper - lower) - 1.0
        if self.prev_targets is None:
            self.prev_targets = q.copy()
        proprio = np.concatenate([q_norm, qd, self.prev_targets]).astype(np.float32)
        self.last_proprio_np = proprio.copy()
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
            return DepthFrame(
                depth=depth,
                encoding=msg.encoding,
                stamp=stamp,
                receive_time=self.latest_depth_receive_time,
                # ROS Header.seq is not reliable across all publishers. Use
                # stamps only for cache invalidation on the ROS-topic fallback.
                frame_id=-1,
            )
        if self.args.depth_source == "zed_sdk":
            assert self.zed_camera is not None
            frame = self.zed_camera.read()
            if frame is None:
                raise RuntimeError("Failed to read a depth frame from the ZED SDK.")
            return frame
        raise ValueError(f"Unsupported --depth_source {self.args.depth_source!r}")

    def _image_tensor(self, *, save_debug: bool = True) -> tuple[torch.Tensor, DepthPipelineOutput, rospy.Time]:
        t_read_start = time.time()
        frame = self._read_depth_frame()
        t_read_done = time.time()
        same_frame_id = frame.frame_id >= 0 and self.cached_depth_frame_id == frame.frame_id
        same_stamp = self.cached_depth_stamp == frame.stamp
        cache_hit = (
            (same_frame_id or same_stamp)
            and self.cached_depth_pipeline is not None
            and self.cached_depth_image is not None
        )
        if cache_hit:
            pipeline = self.cached_depth_pipeline
            image = self.cached_depth_image
        else:
            pipeline = self.preprocessor(frame.depth, encoding=frame.encoding)
            image = torch.from_numpy(pipeline.policy_crop).to(self.device).float().view(
                1, 1, POLICY_HEIGHT, POLICY_WIDTH
            )
            self.cached_depth_stamp = frame.stamp
            self.cached_depth_frame_id = frame.frame_id
            self.cached_depth_pipeline = pipeline
            self.cached_depth_image = image
        t_preprocess_done = time.time()
        if save_debug and not cache_hit:
            self.debug_saver.maybe_save(pipeline)
        t_debug_done = time.time()
        self.last_depth_age_s = max(0.0, (rospy.Time.now() - frame.stamp).to_sec())
        if frame.receive_time is not None:
            self.last_depth_pub_to_callback_s = max(0.0, (frame.receive_time - frame.stamp).to_sec())
        else:
            self.last_depth_pub_to_callback_s = None
        self.depth_frame_reused = cache_hit or self.last_depth_stamp == frame.stamp or (
            frame.frame_id >= 0 and self.last_depth_frame_id == frame.frame_id
        )
        self.last_depth_stamp = frame.stamp
        self.last_depth_frame_id = frame.frame_id
        self.last_zed_grab_ms = frame.grab_ms
        self.last_zed_retrieve_ms = frame.retrieve_ms
        self.last_zed_copy_ms = frame.copy_ms
        self.last_zed_resize_ms = frame.resize_ms
        self.last_zed_total_ms = frame.total_ms
        self.last_zed_inter_frame_ms = frame.inter_frame_ms
        self.last_zed_failures = frame.failures
        self.last_depth_timing_ms = {
            "read": 1000.0 * (t_read_done - t_read_start),
            "preprocess": 1000.0 * (t_preprocess_done - t_read_done),
            "debug": 1000.0 * (t_debug_done - t_preprocess_done),
        }
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

    def _publish_predicted_pose(
        self,
        aux: dict[str, torch.Tensor],
        stamp: rospy.Time,
        predicted_pose: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        if not self.args.publish_object_pose:
            return
        pose = predicted_pose if predicted_pose is not None else self._predicted_pose(aux)
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
        depth_age_text = ""
        if self.last_depth_age_s is not None:
            depth_age_text = f" depth_age_ms={1000.0 * self.last_depth_age_s:.1f}"
        if self.last_depth_pub_to_callback_s is not None:
            depth_age_text += f" depth_pub_to_callback_ms={1000.0 * self.last_depth_pub_to_callback_s:.1f}"
        zed_timing_text = ""
        if self.last_zed_grab_ms is not None:
            zed_timing_text += f" zed_grab_ms={self.last_zed_grab_ms:.1f}"
        if self.last_zed_retrieve_ms is not None:
            zed_timing_text += f" zed_retrieve_ms={self.last_zed_retrieve_ms:.1f}"
        if self.last_zed_copy_ms is not None:
            zed_timing_text += f" zed_copy_ms={self.last_zed_copy_ms:.1f}"
        if self.last_zed_resize_ms is not None:
            zed_timing_text += f" zed_resize_ms={self.last_zed_resize_ms:.1f}"
        if self.last_zed_total_ms is not None:
            zed_timing_text += f" zed_total_ms={self.last_zed_total_ms:.1f}"
        if self.last_zed_inter_frame_ms is not None:
            zed_timing_text += f" zed_period_ms={self.last_zed_inter_frame_ms:.1f}"
        if self.last_zed_failures:
            zed_timing_text += f" zed_failures={self.last_zed_failures}"
        timing_text = ""
        if self.last_step_timing_ms:
            timing_text = (
                f" depth_ms={self.last_step_timing_ms.get('depth', 0.0):.1f}"
                f" policy_ms={self.last_step_timing_ms.get('policy', 0.0):.1f}"
                f" policy_submit_ms={self.last_step_timing_ms.get('policy_submit', 0.0):.1f}"
                f" action_sync_ms={self.last_step_timing_ms.get('action_sync', 0.0):.1f}"
                f" target_ms={self.last_step_timing_ms.get('targets', 0.0):.1f}"
                f" pose_ms={self.last_step_timing_ms.get('pose', 0.0):.1f}"
                f" total_ms={self.last_step_timing_ms.get('total', 0.0):.1f}"
            )
        if self.last_depth_timing_ms:
            timing_text += (
                f" depth_read_ms={self.last_depth_timing_ms.get('read', 0.0):.1f}"
                f" depth_pre_ms={self.last_depth_timing_ms.get('preprocess', 0.0):.1f}"
                f" depth_debug_ms={self.last_depth_timing_ms.get('debug', 0.0):.1f}"
            )
        depth_reuse_text = f" depth_reused={self.depth_frame_reused}"
        info(
            f"[student_depth_policy_node] step={self.loop_count} "
            f"crop_median={crop_med:.3f}m crop_in_window={100.0 * crop_in:.1f}% "
            f"action_abs_max={np.abs(action).max():.3f} "
            f"target_delta_abs_max={np.abs(q_targets - prev_targets).max():.3f} "
            f"published={published}"
            f" depth_frame_id={self.last_depth_frame_id}"
            f"{depth_age_text}"
            f"{depth_reuse_text}"
            f"{zed_timing_text}"
            f"{timing_text}"
        )

    def _wait_for_first_inputs(self) -> None:
        depth_label = "ZED SDK depth" if self.args.depth_source == "zed_sdk" else "ROS depth topic"
        info(f"Waiting for first {depth_label} + iiwa + sharpa observations...")
        rate = rospy.Rate(self.args.control_hz)
        while not rospy.is_shutdown():
            if self._ready():
                q, _ = self._joint_arrays()
                try:
                    # For direct ZED capture, this either grabs one frame
                    # directly or waits until the background reader has cached
                    # one frame in non-blocking mode.
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
            with torch.inference_mode():
                _, _ = self.policy(image, proprio, self.hidden)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            if self.args.warmup_publish_current_targets:
                self._publish_joint_targets(np.clip(q, Q_LOWER_LIMITS_np, Q_UPPER_LIMITS_np))
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

    def _benchmark_policy_latency(self) -> None:
        num_steps = int(self.args.startup_policy_benchmark_steps)
        if num_steps <= 0:
            return
        if self.device.type != "cuda":
            warn("Startup policy benchmark is most useful on CUDA; running on CPU.")
        q, qd = self._joint_arrays()
        image, _, _ = self._image_tensor(save_debug=False)
        proprio = self._proprio_tensor(q, qd)
        hidden = self.policy.initial_state(1, self.device)
        timings_ms = []
        with torch.inference_mode():
            for _ in range(num_steps):
                t0 = time.time()
                output, hidden = self.policy(image, proprio, hidden)
                _ = output.action.detach().cpu().numpy()
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                timings_ms.append(1000.0 * (time.time() - t0))
        median_ms = float(np.median(timings_ms))
        p95_ms = float(np.quantile(timings_ms, 0.95))
        budget_ms = 1000.0 / float(self.args.control_hz)
        message = (
            f"Startup policy latency benchmark over {num_steps} cached-frame steps: "
            f"median={median_ms:.1f}ms p95={p95_ms:.1f}ms budget={budget_ms:.1f}ms "
            f"for {self.args.control_hz:.1f}Hz."
        )
        if p95_ms > budget_ms:
            warn(message + " Pure policy inference is too slow for the requested control rate on this setup.")
        else:
            info(message)
        self.hidden = self.policy.initial_state(1, self.device)

    def _active_run_elapsed_s(self) -> float:
        if self.active_loop_start_time is None:
            return 0.0
        return (rospy.Time.now() - self.active_loop_start_time).to_sec()

    def step(self) -> None:
        t_step_start = time.time()
        if not self._ready():
            return
        if self.args.run_duration_s >= 0.0 and self._active_run_elapsed_s() >= self.args.run_duration_s:
            info(f"Reached --run_duration_s={self.args.run_duration_s:.2f}; shutting down.")
            rospy.signal_shutdown("student depth policy run duration elapsed")
            return
        q, qd = self._joint_arrays()
        t_depth_start = time.time()
        image, pipeline, stamp = self._image_tensor()
        t_depth_done = time.time()
        proprio = self._proprio_tensor(q, qd)
        t_policy_start = time.time()
        with torch.inference_mode():
            output, self.hidden = self.policy(image, proprio, self.hidden)
        t_policy_submit_done = time.time()
        action = output.action.detach().cpu().numpy()
        t_policy_done = time.time()
        if action.shape != (1, N_ACTIONS):
            raise RuntimeError(f"Expected action shape (1, 29), got {action.shape}")
        predicted_pose = self._predicted_pose(output.aux) if (self.args.publish_object_pose or self.rollout_logger.enabled) else None
        t_targets_start = time.time()
        prev_targets = self.prev_targets.copy()
        q_targets = compute_joint_pos_targets(
            actions=action,
            prev_targets=prev_targets[None],
            hand_moving_average=self.args.hand_moving_average,
            arm_moving_average=self.args.arm_moving_average,
            dof_speed_scale=self.args.dof_speed_scale,
            dt=1.0 / self.args.control_hz,
        )[0].astype(np.float32)
        published = False
        if self._should_publish_joints() and self._targets_safe_to_publish(q, q_targets):
            self._publish_joint_targets(q_targets)
            published = True
        self._set_prev_targets_after_step(q=q, q_targets=q_targets, published=published)
        t_targets_done = time.time()
        self._publish_predicted_pose(output.aux, stamp, predicted_pose=predicted_pose)
        t_pose_done = time.time()
        t_step_done = time.time()
        self.last_step_timing_ms = {
            "depth": 1000.0 * (t_depth_done - t_depth_start),
            "policy": 1000.0 * (t_policy_done - t_policy_start),
            "policy_submit": 1000.0 * (t_policy_submit_done - t_policy_start),
            "action_sync": 1000.0 * (t_policy_done - t_policy_submit_done),
            "targets": 1000.0 * (t_targets_done - t_targets_start),
            "pose": 1000.0 * (t_pose_done - t_targets_done),
            "total": 1000.0 * (t_step_done - t_step_start),
        }
        if self.last_proprio_np is None:
            raise RuntimeError("Internal error: last_proprio_np was not populated.")
        self.rollout_logger.maybe_record(
            step=self.loop_count,
            run_start_time=self.active_loop_start_time,
            q=q,
            qd=qd,
            proprio=self.last_proprio_np,
            pipeline=pipeline,
            action=action[0],
            q_targets=q_targets,
            prev_targets=prev_targets,
            published=published,
            depth_stamp=stamp,
            depth_age_s=self.last_depth_age_s,
            depth_pub_to_callback_s=self.last_depth_pub_to_callback_s,
            depth_reused=self.depth_frame_reused,
            predicted_pose=predicted_pose,
        )
        self._print_status(pipeline, action[0], q_targets, prev_targets, published)
        self.loop_count += 1

    def run(self) -> None:
        self._wait_for_first_inputs()
        self._warmup_policy()
        self._benchmark_policy_latency()
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
                breakdown = ""
                if self.last_step_timing_ms:
                    breakdown = (
                        " breakdown="
                        f"depth:{self.last_step_timing_ms.get('depth', 0.0):.1f}ms,"
                        f"policy:{self.last_step_timing_ms.get('policy', 0.0):.1f}ms,"
                        f"target:{self.last_step_timing_ms.get('targets', 0.0):.1f}ms,"
                        f"pose:{self.last_step_timing_ms.get('pose', 0.0):.1f}ms"
                    )
                warn_every(
                    f"Policy loop cannot keep up: step took {1000.0 * elapsed:.1f} ms for "
                    f"{self.args.control_hz:.1f} Hz.{breakdown}",
                    1.0,
                    key="loop_slow",
                )
            rate.sleep()
        self._save_rollout_recording()
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
    parser.add_argument("--depth_topic", default=DEPTH_TOPIC, help=argparse.SUPPRESS)
    parser.add_argument("--depth_units", choices=("auto", "m", "mm"), default=DEPTH_UNITS, help=argparse.SUPPRESS)
    parser.add_argument(
        "--resize_interpolation",
        choices=("area", "nearest", "linear"),
        default=RESIZE_INTERPOLATION,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--zed_serial_number", default=ZED_SERIAL_NUMBER, help=argparse.SUPPRESS)
    parser.add_argument("--zed_resolution", default=ZED_RESOLUTION, help=argparse.SUPPRESS)
    parser.add_argument("--zed_depth_mode", default=ZED_DEPTH_MODE, help=argparse.SUPPRESS)
    parser.add_argument("--zed_camera_fps", type=int, default=ZED_CAMERA_FPS, help=argparse.SUPPRESS)
    parser.add_argument("--zed_grab_hz", type=float, default=ZED_GRAB_HZ, help=argparse.SUPPRESS)
    parser.add_argument("--zed_camera_upsidedown", action="store_true")
    parser.add_argument(
        "--zed_exposure",
        type=int,
        default=ZED_EXPOSURE,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--zed_gain",
        type=int,
        default=ZED_GAIN,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--zed_retrieve_width",
        type=int,
        default=RESIZED_WIDTH,
        help="Ask the ZED SDK to retrieve depth at this width. Use <=0 with --zed_retrieve_height <=0 for full frame.",
    )
    parser.add_argument(
        "--zed_retrieve_height",
        type=int,
        default=RESIZED_HEIGHT,
        help="Ask the ZED SDK to retrieve depth at this height. Use <=0 with --zed_retrieve_width <=0 for full frame.",
    )
    parser.add_argument(
        "--zed_nonblocking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Grab ZED depth in a subprocess and reuse the latest shared-memory frame in the policy loop.",
    )
    parser.add_argument(
        "--zed_max_cached_depth_age_s",
        type=float,
        default=0.20,
        help="Warn if non-blocking cached ZED depth is older than this. Set <=0 to disable stale-frame warnings.",
    )
    parser.add_argument("--iiwa_joint_state_topic", default=IIWA_JOINT_STATE_TOPIC, help=argparse.SUPPRESS)
    parser.add_argument("--sharpa_joint_state_topic", default=SHARPA_JOINT_STATE_TOPIC, help=argparse.SUPPRESS)
    parser.add_argument("--iiwa_joint_cmd_topic", default=IIWA_JOINT_CMD_TOPIC, help=argparse.SUPPRESS)
    parser.add_argument("--sharpa_joint_cmd_topic", default=SHARPA_JOINT_CMD_TOPIC, help=argparse.SUPPRESS)
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
    parser.add_argument("--max_arm_target_delta_deg", type=float, default=MAX_ARM_TARGET_DELTA_DEG, help=argparse.SUPPRESS)
    parser.add_argument("--raise_on_large_target_delta", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--warmup_steps", type=int, default=30)
    parser.add_argument(
        "--startup_policy_benchmark_steps",
        type=int,
        default=STARTUP_POLICY_BENCHMARK_STEPS,
        help="Run this many cached-frame policy forwards after warmup to verify the requested control rate is feasible.",
    )
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
    parser.add_argument("--object_pose_topic", default=OBJECT_POSE_TOPIC, help=argparse.SUPPRESS)
    parser.add_argument("--object_pose_frame_id", default=OBJECT_POSE_FRAME_ID, help=argparse.SUPPRESS)
    parser.add_argument(
        "--predicted_pose_model_frame",
        choices=("env", "robot_frame"),
        default=PREDICTED_POSE_MODEL_FRAME,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--position_only_quat_xyzw", type=float, nargs=4, default=(0.0, 0.0, 0.0, 1.0))
    parser.add_argument("--control_hz", type=float, default=DEFAULT_CONTROL_HZ)
    parser.add_argument("--hand_moving_average", type=float, default=HAND_MOVING_AVERAGE, help=argparse.SUPPRESS)
    parser.add_argument("--arm_moving_average", type=float, default=ARM_MOVING_AVERAGE, help=argparse.SUPPRESS)
    parser.add_argument("--dof_speed_scale", type=float, default=DOF_SPEED_SCALE, help=argparse.SUPPRESS)
    parser.add_argument("--debug_depth_dir", type=Path, default=None)
    parser.add_argument("--debug_depth_every_n", type=int, default=30)
    parser.add_argument("--debug_depth_video_path", type=Path, default=None)
    parser.add_argument("--debug_depth_video_fps", type=int, default=10)
    parser.add_argument("--record_rollout_dir", type=Path, default=None)
    parser.add_argument("--record_rollout_name", default=None)
    parser.add_argument("--record_every_n", type=int, default=1)
    parser.add_argument("--record_depth_format", choices=("uint8", "float16", "none"), default="uint8")
    parser.add_argument("--status_interval_s", type=float, default=1.0)
    parser.add_argument("--raise_on_step_error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    node = StudentDepthPolicyNode(args)
    node.run()


if __name__ == "__main__":
    main()
