#!/usr/bin/env python
"""Diagnose ZED capture using a separate process plus shared memory.

This isolates the policy/control loop from PyZED's Python thread scheduling.
The child process owns all ZED SDK calls and writes the latest depth frame into
double-buffered shared memory. The parent process reads metadata and copies the
latest cached frame at a requested consumer rate.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import signal
import time
from collections import deque
from dataclasses import dataclass, field
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np


DEPTH_NEAR_M = 0.70
DEPTH_FAR_M = 1.10
DEPTH_INVALID_M = 10.0
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


def preprocess_depth_mm(depth_mm: np.ndarray, mode: str) -> np.ndarray:
    if depth_mm.ndim == 3:
        depth_mm = depth_mm[..., 0]
    depth_mm = depth_mm.astype(np.float32, copy=False)
    if mode == "none":
        return resize_depth_nearest(depth_mm, RESIZED_WIDTH, RESIZED_HEIGHT)
    if mode == "resize":
        return resize_depth_nearest(depth_mm, RESIZED_WIDTH, RESIZED_HEIGHT)
    if mode != "policy":
        raise ValueError(f"preprocess mode must be none, resize, or policy; got {mode!r}")

    depth_m = depth_mm / 1000.0
    finite = np.isfinite(depth_m)
    depth_m[finite & (depth_m < 0.001)] = 0.0
    depth_m[~finite] = DEPTH_INVALID_M
    resized_m = resize_depth_nearest(depth_m, RESIZED_WIDTH, RESIZED_HEIGHT)
    safe = np.nan_to_num(resized_m, nan=DEPTH_FAR_M, posinf=DEPTH_FAR_M, neginf=DEPTH_NEAR_M)
    return np.clip((safe - DEPTH_NEAR_M) / (DEPTH_FAR_M - DEPTH_NEAR_M), 0.0, 1.0).astype(np.float32)


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


@dataclass
class SharedZedState:
    shm_name: str
    shape: tuple[int, int, int]
    active_idx: mp.Value
    frame_id: mp.Value
    wall_time_s: mp.Value
    zed_timestamp_ms: mp.Value
    grab_ms: mp.Value
    retrieve_ms: mp.Value
    copy_ms: mp.Value
    preprocess_ms: mp.Value
    total_ms: mp.Value
    camera_period_ms: mp.Value
    failures: mp.Value
    lock: mp.Lock


def _enum_value(enum_cls, name: str):
    try:
        return getattr(enum_cls, str(name).upper())
    except AttributeError as exc:
        valid = [item for item in dir(enum_cls) if item.isupper()]
        raise ValueError(f"Invalid ZED enum value {name!r}; valid values include {valid}") from exc


def zed_producer_process(args: argparse.Namespace, state: SharedZedState, stop_event: mp.Event) -> None:
    try:
        import pyzed.sl as sl
    except ModuleNotFoundError as exc:
        raise RuntimeError("pyzed.sl is required in the ZED producer process.") from exc

    shm = shared_memory.SharedMemory(name=state.shm_name)
    shared = np.ndarray(state.shape, dtype=np.float32, buffer=shm.buf)
    camera = sl.Camera()
    depth_mat = sl.Mat()
    retrieve_resolution = None
    low_res_retrieve_failed = False
    last_frame_wall_time_s = None
    runtime_parameters = sl.RuntimeParameters()
    grab_period_s = 1.0 / args.zed_grab_hz if args.zed_grab_hz > 0.0 else 0.0

    try:
        init_params = sl.InitParameters(input_t=sl.InputType())
        init_params.svo_real_time_mode = True
        init_params.camera_resolution = _enum_value(sl.RESOLUTION, args.zed_resolution)
        init_params.depth_mode = _enum_value(sl.DEPTH_MODE, args.zed_depth_mode)
        init_params.coordinate_units = sl.UNIT.MILLIMETER
        if args.zed_camera_fps > 0:
            init_params.camera_fps = int(args.zed_camera_fps)
        if args.zed_serial_number:
            init_params.set_from_serial_number(int(args.zed_serial_number))

        err = camera.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED camera serial={args.zed_serial_number!r}: {err}")
        if args.zed_exposure >= 0:
            camera.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, int(args.zed_exposure))
        if args.zed_gain >= 0:
            camera.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, int(args.zed_gain))
        if args.zed_retrieve_width > 0 and args.zed_retrieve_height > 0:
            retrieve_resolution = sl.Resolution(int(args.zed_retrieve_width), int(args.zed_retrieve_height))

        info(
            "ZED producer process started "
            f"resolution={args.zed_resolution} depth_mode={args.zed_depth_mode} "
            f"camera_fps={args.zed_camera_fps} grab_hz_cap={args.zed_grab_hz if args.zed_grab_hz > 0 else 'none'} "
            f"producer_preprocess={args.producer_preprocess}"
        )

        local_frame_id = 0
        while not stop_event.is_set():
            tick_start = now_s()
            err = camera.grab(runtime_parameters)
            t_grab = now_s()
            if err != sl.ERROR_CODE.SUCCESS:
                with state.failures.get_lock():
                    state.failures.value += 1
                if grab_period_s > 0.0:
                    stop_event.wait(min(grab_period_s, 0.01))
                else:
                    stop_event.wait(0.001)
                continue

            if retrieve_resolution is None or low_res_retrieve_failed:
                camera.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
            else:
                try:
                    camera.retrieve_measure(depth_mat, sl.MEASURE.DEPTH, sl.MEM.CPU, retrieve_resolution)
                except TypeError:
                    low_res_retrieve_failed = True
                    info("Low-resolution retrieve_measure overload failed; falling back to full-frame retrieve.")
                    camera.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
            t_retrieve = now_s()

            depth_mm = np.array(depth_mat.get_data(), copy=True)
            if args.zed_camera_upsidedown:
                import cv2

                depth_mm = cv2.flip(depth_mm, -1)
            t_copy = now_s()

            processed = preprocess_depth_mm(depth_mm, args.producer_preprocess)
            t_pre = now_s()

            zed_timestamp_ms = -1
            try:
                zed_timestamp_ms = int(camera.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds())
            except Exception:
                pass

            camera_period_ms = -1.0
            if last_frame_wall_time_s is not None:
                camera_period_ms = 1000.0 * (t_pre - last_frame_wall_time_s)
            last_frame_wall_time_s = t_pre

            with state.lock:
                next_idx = 1 - int(state.active_idx.value)
                shared[next_idx, :, :] = processed
                state.active_idx.value = next_idx
                state.frame_id.value = local_frame_id
                state.wall_time_s.value = t_pre
                state.zed_timestamp_ms.value = zed_timestamp_ms
                state.grab_ms.value = 1000.0 * (t_grab - tick_start)
                state.retrieve_ms.value = 1000.0 * (t_retrieve - t_grab)
                state.copy_ms.value = 1000.0 * (t_copy - t_retrieve)
                state.preprocess_ms.value = 1000.0 * (t_pre - t_copy)
                state.total_ms.value = 1000.0 * (t_pre - tick_start)
                state.camera_period_ms.value = camera_period_ms
            local_frame_id += 1

            elapsed_s = now_s() - tick_start
            if grab_period_s > elapsed_s:
                stop_event.wait(grab_period_s - elapsed_s)
    finally:
        camera.close()
        shm.close()


def save_debug_frame(save_dir: Path | None, frame_id: int, image: np.ndarray, every_n_new_frames: int) -> None:
    if save_dir is None or frame_id % max(1, every_n_new_frames) != 0:
        return
    save_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    prefix = save_dir / f"zed_mp_frame_{frame_id:06d}"
    np.savez_compressed(prefix.with_suffix(".npz"), image=image)
    Image.fromarray((np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8)).save(
        prefix.with_name(prefix.name + "_image.png")
    )


def run_consumer(args: argparse.Namespace, state: SharedZedState, shm: shared_memory.SharedMemory, stop_event: mp.Event) -> None:
    shared = np.ndarray(state.shape, dtype=np.float32, buffer=shm.buf)
    consumer_period_s = 1.0 / args.consumer_hz if args.consumer_hz > 0.0 else 0.0
    start_s = now_s()
    next_status_s = start_s + args.status_interval_s
    last_frame_id = -1
    last_saved_frame_id = -1
    consumer_count = 0
    reused_count = 0
    none_count = 0
    slow_count = 0
    latest_age_ms = 0.0
    loop_ms = CounterWindow()
    tick_period_ms = CounterWindow()
    sleep_actual_ms = CounterWindow()
    copy_ms = CounterWindow()
    last_tick_start_s = None

    info(
        "Multiprocess consumer loop "
        f"target_hz={args.consumer_hz} duration_s={args.duration_s if args.duration_s > 0 else 'until Ctrl-C'}"
    )
    while not stop_event.is_set():
        tick_start_s = now_s()
        if last_tick_start_s is not None:
            tick_period_ms.add(1000.0 * (tick_start_s - last_tick_start_s))
        last_tick_start_s = tick_start_s
        if args.duration_s > 0.0 and tick_start_s - start_s >= args.duration_s:
            break

        with state.lock:
            frame_id = int(state.frame_id.value)
            active_idx = int(state.active_idx.value)
            frame_wall_time_s = float(state.wall_time_s.value)
            zed_grab_ms = float(state.grab_ms.value)
            zed_retrieve_ms = float(state.retrieve_ms.value)
            zed_copy_ms = float(state.copy_ms.value)
            zed_pre_ms = float(state.preprocess_ms.value)
            zed_total_ms = float(state.total_ms.value)
            zed_period_ms = float(state.camera_period_ms.value)
            failures = int(state.failures.value)
        if frame_id < 0:
            none_count += 1
            sleep_request_s = min(0.001, consumer_period_s) if consumer_period_s > 0 else 0.001
            sleep_start_s = now_s()
            time.sleep(sleep_request_s)
            sleep_actual_ms.add(1000.0 * (now_s() - sleep_start_s))
            continue

        t_copy0 = now_s()
        image = shared[active_idx].copy()
        copy_ms.add(1000.0 * (now_s() - t_copy0))

        consumer_count += 1
        if frame_id == last_frame_id:
            reused_count += 1
        else:
            last_frame_id = frame_id
            if frame_id != last_saved_frame_id:
                save_debug_frame(args.save_dir, frame_id, image, args.save_every_n_new_frames)
                last_saved_frame_id = frame_id

        latest_age_ms = 1000.0 * (now_s() - frame_wall_time_s)
        loop_elapsed_ms = 1000.0 * (now_s() - tick_start_s)
        loop_ms.add(loop_elapsed_ms)
        if consumer_period_s > 0.0 and loop_elapsed_ms > 1000.0 * consumer_period_s:
            slow_count += 1

        status_now_s = now_s()
        if status_now_s >= next_status_s:
            elapsed_s = status_now_s - start_s
            loop_med, loop_p95, loop_max = loop_ms.summary()
            tick_med, tick_p95, tick_max = tick_period_ms.summary()
            sleep_med, sleep_p95, sleep_max = sleep_actual_ms.summary()
            copy_med, copy_p95, copy_max = copy_ms.summary()
            reuse_pct = 100.0 * reused_count / max(1, consumer_count)
            producer_fps = (frame_id + 1) / max(1e-6, elapsed_s)
            consumer_hz = consumer_count / max(1e-6, elapsed_s)
            info(
                "[zed_multiprocess_debug] "
                f"elapsed={elapsed_s:.1f}s producer_fps={producer_fps:.1f} consumer_hz={consumer_hz:.1f} "
                f"latest_id={frame_id} reused={reuse_pct:.1f}% none={none_count} "
                f"slow_loops={slow_count} age_ms={latest_age_ms:.1f} "
                f"zed_grab={zed_grab_ms:.1f}ms retrieve={zed_retrieve_ms:.1f}ms "
                f"copy={zed_copy_ms:.1f}ms prod_pre={zed_pre_ms:.1f}ms "
                f"prod_total={zed_total_ms:.1f}ms zed_period={zed_period_ms:.1f}ms "
                f"tick_period_med/p95/max={tick_med:.2f}/{tick_p95:.2f}/{tick_max:.2f}ms "
                f"sleep_actual_med/p95/max={sleep_med:.2f}/{sleep_p95:.2f}/{sleep_max:.2f}ms "
                f"consumer_loop_med/p95/max={loop_med:.2f}/{loop_p95:.2f}/{loop_max:.2f}ms "
                f"shm_copy_med/p95/max={copy_med:.3f}/{copy_p95:.3f}/{copy_max:.3f}ms "
                f"image_shape={tuple(image.shape)} failures={failures}"
            )
            next_status_s = status_now_s + args.status_interval_s

        elapsed_s = now_s() - tick_start_s
        if consumer_period_s > elapsed_s:
            sleep_request_s = consumer_period_s - elapsed_s
            sleep_start_s = now_s()
            time.sleep(sleep_request_s)
            sleep_actual_ms.add(1000.0 * (now_s() - sleep_start_s))


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
    parser.add_argument("--status_interval_s", type=float, default=1.0)
    parser.add_argument("--duration_s", type=float, default=-1.0)
    parser.add_argument("--save_dir", type=Path, default=None)
    parser.add_argument("--save_every_n_new_frames", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mp.set_start_method("spawn", force=True)
    shape = (2, RESIZED_HEIGHT, RESIZED_WIDTH)
    shm = shared_memory.SharedMemory(create=True, size=int(np.prod(shape)) * np.dtype(np.float32).itemsize)
    shared = np.ndarray(shape, dtype=np.float32, buffer=shm.buf)
    shared.fill(0.0)
    state = SharedZedState(
        shm_name=shm.name,
        shape=shape,
        active_idx=mp.Value("i", 0),
        frame_id=mp.Value("i", -1),
        wall_time_s=mp.Value("d", 0.0),
        zed_timestamp_ms=mp.Value("q", -1),
        grab_ms=mp.Value("d", 0.0),
        retrieve_ms=mp.Value("d", 0.0),
        copy_ms=mp.Value("d", 0.0),
        preprocess_ms=mp.Value("d", 0.0),
        total_ms=mp.Value("d", 0.0),
        camera_period_ms=mp.Value("d", -1.0),
        failures=mp.Value("i", 0),
        lock=mp.Lock(),
    )
    stop_event = mp.Event()
    process = mp.Process(target=zed_producer_process, args=(args, state, stop_event), daemon=True)

    def _handle_signal(_signum, _frame) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        process.start()
        run_consumer(args, state, shm, stop_event)
    finally:
        stop_event.set()
        process.join(timeout=2.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
        shm.close()
        shm.unlink()


if __name__ == "__main__":
    main()
