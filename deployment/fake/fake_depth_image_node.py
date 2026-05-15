#!/usr/bin/env python
"""Publish a fixed or replayed metric depth image as a ROS Image topic.

This is intentionally small: it lets the depth student node exercise its
existing ``--depth_source ros_topic`` path without requiring ZED hardware.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import rospy
from sensor_msgs.msg import CameraInfo, Image


DEPTH_TOPIC = "/zed/zed_node/depth/depth_registered"
DEPTH_NEAR_M = 0.70
DEPTH_FAR_M = 1.10
RESIZED_WIDTH = 160
RESIZED_HEIGHT = 90
CROP_X0 = 90
CROP_Y0 = 0
CROP_X1 = 160
CROP_Y1 = 70


def _to_metric_depth(
    array: np.ndarray,
    *,
    input_units: str,
    near_m: float,
    far_m: float,
) -> np.ndarray:
    depth = np.asarray(array)
    if depth.ndim == 3 and depth.shape[-1] in (3, 4):
        depth = depth[..., 0]
    depth = depth.astype(np.float32, copy=False)

    units = input_units
    finite = np.isfinite(depth)
    if units == "auto":
        if np.issubdtype(array.dtype, np.integer) and finite.any() and float(np.nanmax(depth)) > 255.0:
            units = "mm"
        elif finite.any() and 0.0 <= float(np.nanmin(depth)) and float(np.nanmax(depth)) <= 1.0:
            units = "normalized"
        elif finite.any() and float(np.nanmedian(depth[finite])) > 10.0:
            units = "mm"
        else:
            units = "m"

    if units == "m":
        depth_m = depth.copy()
    elif units == "mm":
        depth_m = depth / 1000.0
    elif units == "normalized":
        depth_m = near_m + np.clip(depth, 0.0, 1.0) * (far_m - near_m)
    else:
        raise ValueError(f"input_units must be auto, m, mm, or normalized; got {input_units!r}")

    depth_m[~np.isfinite(depth_m)] = far_m
    depth_m[depth_m < 0.0] = 0.0
    return depth_m.astype(np.float32, copy=False)


def _load_npz_depth(path: Path, args: argparse.Namespace) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    preferred_keys = [
        "raw_depth_m",
        "resized_depth_m",
        "noisy_depth_m",
        "policy_full_depth",
        "policy_crop",
    ]
    key = args.npz_key
    if key is None:
        key = next((candidate for candidate in preferred_keys if candidate in data), None)
    if key is None:
        raise KeyError(f"No supported depth key found in {path}. Available keys: {list(data.keys())}")

    array = np.asarray(data[key])
    if array.ndim >= 4:
        array = np.squeeze(array)
    if key in ("policy_full_depth", "policy_crop") and args.input_units == "auto":
        input_units = "normalized"
    else:
        input_units = args.input_units
    depth_m = _to_metric_depth(array, input_units=input_units, near_m=args.near_m, far_m=args.far_m)

    if key == "policy_crop" and depth_m.shape == (CROP_Y1 - CROP_Y0, CROP_X1 - CROP_X0):
        full = np.full((RESIZED_HEIGHT, RESIZED_WIDTH), args.far_m, dtype=np.float32)
        full[CROP_Y0:CROP_Y1, CROP_X0:CROP_X1] = depth_m
        depth_m = full
    return depth_m


def _load_depth(path: Path, args: argparse.Namespace) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        depth = _load_npz_depth(path, args)
    elif suffix == ".npy":
        depth = _to_metric_depth(np.load(path), input_units=args.input_units, near_m=args.near_m, far_m=args.far_m)
    elif suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        from PIL import Image as PILImage

        depth = _to_metric_depth(
            np.asarray(PILImage.open(path)),
            input_units=args.input_units,
            near_m=args.near_m,
            far_m=args.far_m,
        )
    elif suffix in (".mp4", ".avi", ".mov"):
        import imageio.v2 as imageio

        frames = []
        for frame in imageio.get_reader(str(path)):
            frames.append(_to_metric_depth(frame, input_units=args.input_units, near_m=args.near_m, far_m=args.far_m))
        if not frames:
            raise ValueError(f"No frames found in {path}")
        depth = np.stack(frames, axis=0)
    else:
        raise ValueError(f"Unsupported depth file suffix {suffix!r}: {path}")

    if depth.ndim == 2:
        depth = depth[None]
    if depth.ndim != 3:
        raise ValueError(f"Expected depth shape (H,W) or (T,H,W), got {depth.shape}")
    return depth.astype(np.float32, copy=False)


def _depth_msg(depth_m: np.ndarray, *, frame_id: str) -> Image:
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


def _camera_info_msg(depth_m: np.ndarray, *, frame_id: str, fx: float, fy: float, cx: float, cy: float) -> CameraInfo:
    msg = CameraInfo()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = frame_id
    msg.height = int(depth_m.shape[0])
    msg.width = int(depth_m.shape[1])
    msg.K = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    msg.P = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    msg.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    msg.distortion_model = "plumb_bob"
    msg.D = [0.0, 0.0, 0.0, 0.0, 0.0]
    return msg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth_path", type=Path, required=True)
    parser.add_argument("--topic", default=DEPTH_TOPIC)
    parser.add_argument("--rate_hz", type=float, default=30.0)
    parser.add_argument("--run_duration_s", type=float, default=-1.0)
    parser.add_argument("--frame_id", default="zed_left_camera_frame")
    parser.add_argument("--input_units", choices=("auto", "m", "mm", "normalized"), default="auto")
    parser.add_argument("--npz_key", default=None)
    parser.add_argument("--near_m", type=float, default=DEPTH_NEAR_M)
    parser.add_argument("--far_m", type=float, default=DEPTH_FAR_M)
    loop_group = parser.add_mutually_exclusive_group()
    loop_group.add_argument("--loop", dest="loop", action="store_true")
    loop_group.add_argument("--no-loop", dest="loop", action="store_false")
    parser.set_defaults(loop=True)
    parser.add_argument("--publish_camera_info", action="store_true")
    parser.add_argument("--camera_info_topic", default="/zed/zed_node/rgb/camera_info")
    parser.add_argument("--fx", type=float, default=80.0)
    parser.add_argument("--fy", type=float, default=80.0)
    parser.add_argument("--cx", type=float, default=(RESIZED_WIDTH - 1) / 2.0)
    parser.add_argument("--cy", type=float, default=(RESIZED_HEIGHT - 1) / 2.0)
    parser.add_argument("--status_interval_s", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    depth_seq = _load_depth(args.depth_path, args)
    rospy.init_node("fake_depth_image_node")
    pub = rospy.Publisher(args.topic, Image, queue_size=1)
    cam_info_pub = None
    if args.publish_camera_info:
        cam_info_pub = rospy.Publisher(args.camera_info_topic, CameraInfo, queue_size=1)
    rate = rospy.Rate(args.rate_hz)
    start = time.time()
    last_status = 0.0
    idx = 0
    print(
        f"[fake_depth_image_node] publishing {args.depth_path} shape={depth_seq.shape} "
        f"topic={args.topic} rate={args.rate_hz}Hz",
        flush=True,
    )
    while not rospy.is_shutdown():
        if args.run_duration_s >= 0.0 and time.time() - start >= args.run_duration_s:
            break
        depth_msg = _depth_msg(depth_seq[idx], frame_id=args.frame_id)
        pub.publish(depth_msg)
        if cam_info_pub is not None:
            cam_info_msg = _camera_info_msg(depth_seq[idx], frame_id=args.frame_id, fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy)
            cam_info_msg.header.stamp = depth_msg.header.stamp
            cam_info_pub.publish(cam_info_msg)
        now = time.time()
        if now - last_status >= args.status_interval_s:
            frame = depth_seq[idx]
            finite = np.isfinite(frame)
            med = float(np.median(frame[finite])) if finite.any() else float("nan")
            print(f"[fake_depth_image_node] frame={idx} median_depth_m={med:.3f}", flush=True)
            last_status = now
        idx += 1
        if idx >= depth_seq.shape[0]:
            if args.loop:
                idx = 0
            else:
                break
        rate.sleep()


if __name__ == "__main__":
    main()
