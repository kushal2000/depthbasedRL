#!/usr/bin/env python
"""Diagnose non-blocking ZED depth capture independently of the policy node.

The script runs a producer thread that owns all ZED SDK calls and a consumer
loop that repeatedly reads the latest cached frame at a requested rate. It is
intended to answer whether the camera path is blocking, over-polling, producing
stale frames, or spending unexpected time in preprocessing/debug image writes.
"""

from __future__ import annotations

import argparse
import signal
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


DEPTH_NEAR_M = 0.70
DEPTH_FAR_M = 1.10
RESIZED_WIDTH = 160
RESIZED_HEIGHT = 90
CROP_X0 = 90
CROP_Y0 = 0
CROP_X1 = 160
CROP_Y1 = 70


def info(message: str) -> None:
    print(message, flush=True)


def now_s() -> float:
    return time.perf_counter()


def resize_depth_nearest(depth: np.ndarray, width: int, height: int) -> np.ndarray:
    if depth.shape == (height, width):
        return depth.astype(np.float32, copy=False)
    import cv2

    return cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST).astype(np.float32, copy=False)


@dataclass
class PolicyDepth:
    raw_m: np.ndarray
    resized_m: np.ndarray
    window_160x90: np.ndarray
    crop_70x70: np.ndarray


def preprocess_depth(depth_mm: np.ndarray, mode: str) -> PolicyDepth | np.ndarray | None:
    """Optionally run the policy-style metric conversion/window/crop path."""

    if mode == "none":
        return None
    if depth_mm.ndim == 3:
        depth_mm = depth_mm[..., 0]
    depth_mm = depth_mm.astype(np.float32, copy=False)
    if mode == "resize":
        return resize_depth_nearest(depth_mm, RESIZED_WIDTH, RESIZED_HEIGHT)
    if mode != "policy":
        raise ValueError(f"preprocess mode must be none, resize, or policy; got {mode!r}")

    raw_m = depth_mm / 1000.0
    raw_m[(raw_m < 0.001) | (~np.isfinite(raw_m))] = 0.0
    resized_m = resize_depth_nearest(raw_m, RESIZED_WIDTH, RESIZED_HEIGHT)
    safe = np.nan_to_num(resized_m, nan=DEPTH_FAR_M, posinf=DEPTH_FAR_M, neginf=DEPTH_NEAR_M)
    window = np.clip((safe - DEPTH_NEAR_M) / (DEPTH_FAR_M - DEPTH_NEAR_M), 0.0, 1.0).astype(np.float32)
    crop = window[CROP_Y0:CROP_Y1, CROP_X0:CROP_X1]
    return PolicyDepth(raw_m=raw_m, resized_m=resized_m, window_160x90=window, crop_70x70=crop)


@dataclass
class ZedFrame:
    frame_id: int
    depth_mm: np.ndarray
    wall_time_s: float
    zed_timestamp_ms: int | None
    preprocessed: PolicyDepth | np.ndarray | None
    grab_ms: float
    retrieve_ms: float
    copy_ms: float
    preprocess_ms: float
    total_ms: float
    camera_period_ms: float | None


@dataclass
class CounterWindow:
    values: deque[float] = field(default_factory=lambda: deque(maxlen=512))

    def add(self, value: float) -> None:
        self.values.append(float(value))

    def summary(self) -> tuple[float, float, float]:
        if not self.values:
            return 0.0, 0.0, 0.0
        arr = np.asarray(self.values, dtype=np.float64)
        return float(np.median(arr)), float(np.quantile(arr, 0.95)), float(arr.max())


class ZedProducer:
    def __init__(self, args: argparse.Namespace) -> None:
        try:
            import pyzed.sl as sl
        except ModuleNotFoundError as exc:
            raise RuntimeError("pyzed.sl is required. Run in a shell with the ZED SDK Python bindings.") from exc

        self.args = args
        self.sl = sl
        self.camera = sl.Camera()
        self.depth_mat = sl.Mat()
        self.latest_lock = threading.Lock()
        self.latest_frame: ZedFrame | None = None
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.frame_id = 0
        self.fail_count = 0
        self.retrieve_resolution = None
        self.low_res_retrieve_failed = False
        self.last_frame_wall_time_s: float | None = None

        init_params = sl.InitParameters(input_t=sl.InputType())
        init_params.svo_real_time_mode = True
        init_params.camera_resolution = self._enum_value(sl.RESOLUTION, args.zed_resolution)
        init_params.depth_mode = self._enum_value(sl.DEPTH_MODE, args.zed_depth_mode)
        init_params.coordinate_units = sl.UNIT.MILLIMETER
        if args.zed_camera_fps > 0:
            init_params.camera_fps = int(args.zed_camera_fps)
        if args.zed_serial_number:
            init_params.set_from_serial_number(int(args.zed_serial_number))

        err = self.camera.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED camera serial={args.zed_serial_number!r}: {err}")
        if args.zed_exposure >= 0:
            self.camera.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, int(args.zed_exposure))
        if args.zed_gain >= 0:
            self.camera.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, int(args.zed_gain))
        if args.zed_retrieve_width > 0 and args.zed_retrieve_height > 0:
            self.retrieve_resolution = sl.Resolution(int(args.zed_retrieve_width), int(args.zed_retrieve_height))

        self.runtime_parameters = sl.RuntimeParameters()
        self.grab_period_s = 1.0 / args.zed_grab_hz if args.zed_grab_hz > 0.0 else 0.0
        info(
            "Opened ZED "
            f"serial={args.zed_serial_number or '<default>'} resolution={args.zed_resolution} "
            f"depth_mode={args.zed_depth_mode} camera_fps={args.zed_camera_fps} "
            f"grab_hz_cap={args.zed_grab_hz if args.zed_grab_hz > 0 else 'none'} "
            f"retrieve={args.zed_retrieve_width}x{args.zed_retrieve_height} "
            f"producer_preprocess={args.producer_preprocess}"
        )

    @staticmethod
    def _enum_value(enum_cls, name: str):
        try:
            return getattr(enum_cls, str(name).upper())
        except AttributeError as exc:
            valid = [item for item in dir(enum_cls) if item.isupper()]
            raise ValueError(f"Invalid ZED enum value {name!r}; valid values include {valid}") from exc

    def start(self) -> None:
        self.thread = threading.Thread(target=self._loop, name="zed_debug_producer", daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.stop_event.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.camera is not None:
            self.camera.close()
            self.camera = None

    def get_latest(self) -> ZedFrame | None:
        with self.latest_lock:
            return self.latest_frame

    def _retrieve_depth(self) -> None:
        if self.retrieve_resolution is None or self.low_res_retrieve_failed:
            self.camera.retrieve_measure(self.depth_mat, self.sl.MEASURE.DEPTH)
            return
        try:
            self.camera.retrieve_measure(
                self.depth_mat,
                self.sl.MEASURE.DEPTH,
                self.sl.MEM.CPU,
                self.retrieve_resolution,
            )
        except TypeError:
            self.low_res_retrieve_failed = True
            info("Low-resolution retrieve_measure overload failed; falling back to full-frame retrieve.")
            self.camera.retrieve_measure(self.depth_mat, self.sl.MEASURE.DEPTH)

    def _grab_once(self) -> ZedFrame | None:
        t0 = now_s()
        err = self.camera.grab(self.runtime_parameters)
        t_grab = now_s()
        if err != self.sl.ERROR_CODE.SUCCESS:
            self.fail_count += 1
            return None

        self._retrieve_depth()
        t_retrieve = now_s()
        depth_mm = np.array(self.depth_mat.get_data(), copy=True)
        if self.args.zed_camera_upsidedown:
            import cv2

            depth_mm = cv2.flip(depth_mm, -1)
        t_copy = now_s()

        preprocessed = preprocess_depth(depth_mm, self.args.producer_preprocess)
        t_pre = now_s()

        zed_timestamp_ms = None
        try:
            zed_timestamp_ms = int(self.camera.get_timestamp(self.sl.TIME_REFERENCE.IMAGE).get_milliseconds())
        except Exception:
            pass

        camera_period_ms = None
        if self.last_frame_wall_time_s is not None:
            camera_period_ms = 1000.0 * (t_pre - self.last_frame_wall_time_s)
        self.last_frame_wall_time_s = t_pre

        frame = ZedFrame(
            frame_id=self.frame_id,
            depth_mm=depth_mm,
            wall_time_s=t_pre,
            zed_timestamp_ms=zed_timestamp_ms,
            preprocessed=preprocessed,
            grab_ms=1000.0 * (t_grab - t0),
            retrieve_ms=1000.0 * (t_retrieve - t_grab),
            copy_ms=1000.0 * (t_copy - t_retrieve),
            preprocess_ms=1000.0 * (t_pre - t_copy),
            total_ms=1000.0 * (t_pre - t0),
            camera_period_ms=camera_period_ms,
        )
        self.frame_id += 1
        return frame

    def _loop(self) -> None:
        while not self.stop_event.is_set():
            t0 = now_s()
            frame = self._grab_once()
            if frame is not None:
                with self.latest_lock:
                    self.latest_frame = frame
            elapsed_s = now_s() - t0
            if self.grab_period_s > elapsed_s:
                self.stop_event.wait(self.grab_period_s - elapsed_s)
            elif frame is None:
                self.stop_event.wait(0.001)


class DebugImageSaver:
    def __init__(self, save_dir: Path | None, every_n_new_frames: int) -> None:
        self.save_dir = save_dir
        self.every_n_new_frames = max(1, int(every_n_new_frames))
        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def maybe_save(self, frame: ZedFrame) -> None:
        if self.save_dir is None or frame.frame_id % self.every_n_new_frames != 0:
            return
        from PIL import Image

        policy = frame.preprocessed
        if not isinstance(policy, PolicyDepth):
            policy = preprocess_depth(frame.depth_mm, "policy")
        assert isinstance(policy, PolicyDepth)

        prefix = self.save_dir / f"zed_frame_{frame.frame_id:06d}"
        np.savez_compressed(
            prefix.with_suffix(".npz"),
            depth_mm=frame.depth_mm,
            raw_m=policy.raw_m,
            resized_m=policy.resized_m,
            window_160x90=policy.window_160x90,
            crop_70x70=policy.crop_70x70,
        )
        Image.fromarray(self._to_u8(policy.window_160x90)).save(prefix.with_name(prefix.name + "_window_160x90.png"))
        Image.fromarray(self._to_u8(policy.crop_70x70)).save(prefix.with_name(prefix.name + "_crop_70x70.png"))

    @staticmethod
    def _to_u8(image: np.ndarray) -> np.ndarray:
        return (np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def run_consumer(args: argparse.Namespace, producer: ZedProducer) -> None:
    saver = DebugImageSaver(args.save_dir, args.save_every_n_new_frames)
    consumer_period_s = 1.0 / args.consumer_hz if args.consumer_hz > 0 else 0.0
    start_s = now_s()
    next_status_s = start_s + args.status_interval_s
    last_frame_id = -1
    last_saved_frame_id = -1
    consumer_count = 0
    reused_count = 0
    none_count = 0
    new_frame_count = 0
    slow_count = 0
    loop_ms = CounterWindow()
    consumer_pre_ms = CounterWindow()
    latest_age_ms = 0.0

    info(
        "Consumer loop "
        f"target_hz={args.consumer_hz} consumer_preprocess={args.consumer_preprocess} "
        f"duration_s={args.duration_s if args.duration_s > 0 else 'until Ctrl-C'}"
    )
    while not producer.stop_event.is_set():
        tick_start_s = now_s()
        if args.duration_s > 0.0 and tick_start_s - start_s >= args.duration_s:
            break

        frame = producer.get_latest()
        if frame is None:
            none_count += 1
            time.sleep(min(0.001, consumer_period_s) if consumer_period_s > 0 else 0.001)
            continue

        consumer_count += 1
        reused = frame.frame_id == last_frame_id
        if reused:
            reused_count += 1
        else:
            new_frame_count += 1
            last_frame_id = frame.frame_id
            if frame.frame_id != last_saved_frame_id:
                saver.maybe_save(frame)
                last_saved_frame_id = frame.frame_id

        t_pre0 = now_s()
        if args.consumer_preprocess != "none":
            _ = preprocess_depth(frame.depth_mm, args.consumer_preprocess)
        consumer_pre_ms.add(1000.0 * (now_s() - t_pre0))

        latest_age_ms = 1000.0 * (now_s() - frame.wall_time_s)
        loop_elapsed_ms = 1000.0 * (now_s() - tick_start_s)
        loop_ms.add(loop_elapsed_ms)
        if consumer_period_s > 0.0 and loop_elapsed_ms > 1000.0 * consumer_period_s:
            slow_count += 1

        status_now_s = now_s()
        if status_now_s >= next_status_s:
            elapsed_s = status_now_s - start_s
            loop_med, loop_p95, loop_max = loop_ms.summary()
            pre_med, pre_p95, pre_max = consumer_pre_ms.summary()
            reuse_pct = 100.0 * reused_count / max(1, consumer_count)
            producer_fps = producer.frame_id / max(1e-6, elapsed_s)
            consumer_hz = consumer_count / max(1e-6, elapsed_s)
            frame_period = frame.camera_period_ms if frame.camera_period_ms is not None else float("nan")
            info(
                "[zed_nonblocking_debug] "
                f"elapsed={elapsed_s:.1f}s producer_fps={producer_fps:.1f} consumer_hz={consumer_hz:.1f} "
                f"latest_id={frame.frame_id} reused={reuse_pct:.1f}% none={none_count} "
                f"slow_loops={slow_count} age_ms={latest_age_ms:.1f} "
                f"zed_grab={frame.grab_ms:.1f}ms retrieve={frame.retrieve_ms:.1f}ms "
                f"copy={frame.copy_ms:.1f}ms prod_pre={frame.preprocess_ms:.1f}ms "
                f"prod_total={frame.total_ms:.1f}ms zed_period={frame_period:.1f}ms "
                f"consumer_loop_med/p95/max={loop_med:.2f}/{loop_p95:.2f}/{loop_max:.2f}ms "
                f"consumer_pre_med/p95/max={pre_med:.2f}/{pre_p95:.2f}/{pre_max:.2f}ms "
                f"depth_shape={tuple(frame.depth_mm.shape)} failures={producer.fail_count}"
            )
            next_status_s = status_now_s + args.status_interval_s

        elapsed_s = now_s() - tick_start_s
        if consumer_period_s > elapsed_s:
            time.sleep(consumer_period_s - elapsed_s)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zed_serial_number", default="15107")
    parser.add_argument("--zed_resolution", default="HD1080")
    parser.add_argument("--zed_depth_mode", default="NEURAL")
    parser.add_argument("--zed_camera_fps", type=int, default=30)
    parser.add_argument("--zed_grab_hz", type=float, default=30.0, help="Producer rate cap. Use 0 for no cap.")
    parser.add_argument("--zed_exposure", type=int, default=25)
    parser.add_argument("--zed_gain", type=int, default=40)
    parser.add_argument("--zed_camera_upsidedown", action="store_true")
    parser.add_argument("--zed_retrieve_width", type=int, default=RESIZED_WIDTH)
    parser.add_argument("--zed_retrieve_height", type=int, default=RESIZED_HEIGHT)
    parser.add_argument("--producer_preprocess", choices=("none", "resize", "policy"), default="none")
    parser.add_argument("--consumer_hz", type=float, default=60.0)
    parser.add_argument("--consumer_preprocess", choices=("none", "resize", "policy"), default="none")
    parser.add_argument("--status_interval_s", type=float, default=1.0)
    parser.add_argument("--duration_s", type=float, default=-1.0)
    parser.add_argument("--save_dir", type=Path, default=None)
    parser.add_argument("--save_every_n_new_frames", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    producer = ZedProducer(args)

    def _handle_signal(_signum, _frame) -> None:
        producer.stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        producer.start()
        run_consumer(args, producer)
    finally:
        producer.close()


if __name__ == "__main__":
    main()
