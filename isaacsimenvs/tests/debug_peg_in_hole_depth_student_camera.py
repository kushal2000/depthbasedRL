"""Debug PegInHole depth-student camera framing.

This script is intentionally artifact-oriented, not a strict unit test. It
creates a tiny depth-student PegInHole env, captures the student depth images,
saves PNG grids, and prints per-env depth statistics for visual inspection.

Fast fixed-scene check:
    .venv_isaacsim/bin/python isaacsimenvs/tests/debug_peg_in_hole_depth_student_camera.py

Broader scene/tolerance coverage:
    .venv_isaacsim/bin/python isaacsimenvs/tests/debug_peg_in_hole_depth_student_camera.py \\
      --random_scene_tol --num_envs 16
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


OUTPUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "videos"
    / "test_videos"
    / "peg_in_hole_depth_student"
)


def _depth_tensor_to_nhw(depth):
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    elif depth.ndim == 4 and depth.shape[1] == 1:
        depth = depth[:, 0]
    elif depth.ndim != 3:
        raise RuntimeError(f"Unsupported depth tensor shape: {tuple(depth.shape)}")
    return depth.detach().float().cpu().numpy()


def _make_grid(images, *, pad: int = 2):
    import numpy as np

    images = np.clip(images, 0.0, 1.0)
    n, h, w = images.shape
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    grid = np.full(
        (rows * h + (rows - 1) * pad, cols * w + (cols - 1) * pad),
        255,
        dtype=np.uint8,
    )
    images_u8 = (images * 255.0).round().astype(np.uint8)
    for i in range(n):
        row = i // cols
        col = i % cols
        y0 = row * (h + pad)
        x0 = col * (w + pad)
        grid[y0 : y0 + h, x0 : x0 + w] = images_u8[i]
    return grid


def _save_depth_grid(path: Path, images) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_make_grid(images)).save(path)


def _rgb_tensor_to_nhwc(rgb):
    if rgb.ndim != 4:
        raise RuntimeError(f"Unsupported RGB tensor shape: {tuple(rgb.shape)}")
    if rgb.shape[-1] >= 3:
        out = rgb[..., :3]
    elif rgb.shape[1] >= 3:
        out = rgb[:, :3].permute(0, 2, 3, 1)
    else:
        raise RuntimeError(f"Unsupported RGB tensor shape: {tuple(rgb.shape)}")
    return out.detach().float().cpu().numpy()


def _make_rgb_grid(images, *, pad: int = 2):
    import numpy as np

    images = np.clip(images, 0.0, 1.0)
    n, h, w, c = images.shape
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    grid = np.full(
        (rows * h + (rows - 1) * pad, cols * w + (cols - 1) * pad, c),
        255,
        dtype=np.uint8,
    )
    images_u8 = (images * 255.0).round().astype(np.uint8)
    for i in range(n):
        row = i // cols
        col = i % cols
        y0 = row * (h + pad)
        x0 = col * (w + pad)
        grid[y0 : y0 + h, x0 : x0 + w] = images_u8[i]
    return grid


def _save_rgb_grid(path: Path, images) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_make_rgb_grid(images)).save(path)


def _gray_to_rgb(image):
    import numpy as np

    return np.repeat(image[..., None], 3, axis=-1)


def _write_mp4(path: Path, frames, fps: int) -> None:
    import imageio.v3 as iio

    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, frames, fps=fps)


def _print_depth_stats(prefix: str, raw_depth_m, student_depth, near: float, far: float) -> None:
    import numpy as np

    for env_id in range(raw_depth_m.shape[0]):
        raw = raw_depth_m[env_id]
        student = student_depth[env_id]
        finite = np.isfinite(raw)
        if finite.any():
            raw_valid = raw[finite]
            raw_min = float(raw_valid.min())
            raw_max = float(raw_valid.max())
            raw_mean = float(raw_valid.mean())
            in_window = float(((raw >= near) & (raw <= far) & finite).mean())
        else:
            raw_min = raw_max = raw_mean = float("nan")
            in_window = 0.0
        nonzero = float((student > 0.0).mean())
        sat_low = float((student <= 1e-6).mean())
        sat_high = float((student >= 1.0 - 1e-6).mean())
        print(
            f"{prefix} env={env_id:02d} "
            f"raw_m[min={raw_min:.3f}, mean={raw_mean:.3f}, max={raw_max:.3f}, "
            f"in_window={in_window:.3f}] "
            f"student[nonzero={nonzero:.3f}, sat_low={sat_low:.3f}, "
            f"sat_high={sat_high:.3f}]"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--resets", type=int, default=1)
    parser.add_argument("--image_width", type=int, default=160)
    parser.add_argument("--image_height", type=int, default=90)
    parser.add_argument("--image_modality", choices=("depth", "rgb", "rgbd"), default="depth")
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--video_fps", type=int, default=30)
    parser.add_argument("--random_scene_tol", action="store_true")
    parser.add_argument("--force_scene_id", type=int, default=0)
    parser.add_argument("--force_tol_slot", type=int, default=0)
    parser.add_argument("--random_peg", action="store_true")
    parser.add_argument("--force_peg_idx", type=int, default=0)
    parser.add_argument("--object_init_orientation_mode", choices=("scene", "yaw_only", "full"), default=None)
    parser.add_argument("--object_init_position_noise_xy", type=float, nargs=2, default=None)
    parser.add_argument("--object_init_position_noise_z", type=float, default=None)
    parser.add_argument("--depth_noise_profile", choices=("off", "weak", "medium", "strong", "custom"), default="off")
    parser.add_argument("--depth_noise_strength", type=float, default=1.0)
    parser.add_argument(
        "--camera_pose_randomization_profile",
        choices=("off", "weak", "medium", "strong", "custom"),
        default="off",
    )
    parser.add_argument("--camera_pose_randomization_mode", choices=("startup", "reset"), default="startup")
    parser.add_argument("--depth_min_m", type=float, default=None)
    parser.add_argument("--depth_max_m", type=float, default=None)
    parser.add_argument(
        "--crop_top_left",
        type=int,
        nargs=2,
        default=(90, 0),
        metavar=("X", "Y"),
    )
    parser.add_argument(
        "--crop_bottom_right",
        type=int,
        nargs=2,
        default=(160, 70),
        metavar=("X", "Y"),
    )
    parser.add_argument("--no_crop", action="store_true")
    parser.add_argument("--camera_delay_max", type=int, default=0)
    parser.add_argument("--student_obs_delay_max", type=int, default=0)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    args.enable_cameras = True

    app_launcher = AppLauncher(args)
    app = app_launcher.app

    import gymnasium as gym
    import numpy as np
    import torch
    import yaml

    import isaacsimenvs  # noqa: F401  (registers gym envs)
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from isaacsimenvs.utils.config_utils import apply_env_cfg_dict

    task = "Isaacsimenvs-PegInHoleDepthStudent-Direct-v0"
    spec = gym.spec(task)
    cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    with Path(spec.kwargs["env_cfg_yaml_entry_point"]).open() as f:
        apply_env_cfg_dict(cfg, yaml.safe_load(f) or {})

    cfg.scene.num_envs = args.num_envs
    if args.random_scene_tol:
        cfg.peg_in_hole.force_scene_tol_combo = None
    else:
        cfg.peg_in_hole.force_scene_tol_combo = (
            args.force_scene_id,
            args.force_tol_slot,
        )
    cfg.peg_in_hole.force_peg_idx = None if args.random_peg else args.force_peg_idx
    if args.object_init_orientation_mode is not None:
        cfg.peg_in_hole.object_init_orientation_mode = args.object_init_orientation_mode
    if args.object_init_position_noise_xy is not None:
        cfg.peg_in_hole.object_init_position_noise_xy = tuple(args.object_init_position_noise_xy)
    if args.object_init_position_noise_z is not None:
        cfg.peg_in_hole.object_init_position_noise_z = args.object_init_position_noise_z

    cfg.student_obs.image_modality = args.image_modality
    cfg.student_obs.image_width = args.image_width
    cfg.student_obs.image_height = args.image_height
    cfg.student_obs.crop_enabled = not args.no_crop
    cfg.student_obs.crop_top_left = tuple(args.crop_top_left)
    cfg.student_obs.crop_bottom_right = tuple(args.crop_bottom_right)
    if cfg.student_obs.crop_enabled:
        x0, y0 = cfg.student_obs.crop_top_left
        x1, y1 = cfg.student_obs.crop_bottom_right
        cfg.student_obs.image_input_width = x1 - x0
        cfg.student_obs.image_input_height = y1 - y0
    else:
        cfg.student_obs.image_input_width = args.image_width
        cfg.student_obs.image_input_height = args.image_height
    cfg.student_obs.use_camera_delay = args.camera_delay_max > 0
    cfg.student_obs.camera_delay_max = args.camera_delay_max
    cfg.student_obs.use_student_obs_delay = args.student_obs_delay_max > 0
    cfg.student_obs.student_obs_delay_max = args.student_obs_delay_max
    cfg.student_obs.depth_noise_profile = args.depth_noise_profile
    cfg.student_obs.depth_noise_strength = args.depth_noise_strength
    cfg.student_obs.camera_pose_randomization_profile = args.camera_pose_randomization_profile
    cfg.student_obs.camera_pose_randomization_mode = args.camera_pose_randomization_mode
    if args.depth_min_m is not None:
        cfg.student_obs.depth_min_m = args.depth_min_m
    if args.depth_max_m is not None:
        cfg.student_obs.depth_max_m = args.depth_max_m

    env = gym.make(task, cfg=cfg)
    inner = env.unwrapped
    actions = torch.zeros(
        (inner.num_envs, cfg.action_space),
        device=inner.device,
        dtype=torch.float32,
    )

    out_dir = Path(args.output_dir)
    print(f"[debug] writing depth grids to {out_dir}")
    print(
        "[debug] camera cfg "
        f"pos={cfg.student_obs.camera_pos} quat_wxyz={cfg.student_obs.camera_quat_wxyz} "
        f"window=[{cfg.student_obs.depth_min_m}, {cfg.student_obs.depth_max_m}] "
        f"crop_enabled={cfg.student_obs.crop_enabled} "
        f"crop_top_left={cfg.student_obs.crop_top_left} "
        f"crop_bottom_right={cfg.student_obs.crop_bottom_right} "
        f"student_input={cfg.student_obs.image_input_width}x{cfg.student_obs.image_input_height} "
        f"depth_noise={cfg.student_obs.depth_noise_profile}x{cfg.student_obs.depth_noise_strength} "
        f"camera_rand={cfg.student_obs.camera_pose_randomization_profile}/"
        f"{cfg.student_obs.camera_pose_randomization_mode} "
        f"object_init={cfg.peg_in_hole.object_init_orientation_mode}"
    )

    depth_video_frames: list = []
    raw_video_frames: list = []
    noisy_video_frames: list = []
    rgb_video_frames: list = []
    has_depth = args.image_modality in ("depth", "rgbd")
    has_rgb = args.image_modality in ("rgb", "rgbd")
    near = float(cfg.student_obs.depth_min_m)
    far = float(cfg.student_obs.depth_max_m)

    for reset_idx in range(args.resets):
        env.reset()
        last_student_obs = None
        for _ in range(args.steps):
            env.step(actions)
            last_student_obs = inner.get_student_obs()

            if args.save_video and has_depth:
                student_depth_frame = last_student_obs["image"][:, -1].detach().float().cpu().numpy()
                depth_video_frames.append(_gray_to_rgb(_make_grid(student_depth_frame)))

                raw_depth = getattr(inner, "_student_depth_raw_m", None)
                if raw_depth is not None:
                    raw_depth_m_frame = _depth_tensor_to_nhw(raw_depth)
                    raw_window_frame = np.nan_to_num(
                        (raw_depth_m_frame - near) / (far - near),
                        nan=0.0,
                        posinf=1.0,
                        neginf=0.0,
                    )
                    raw_video_frames.append(_gray_to_rgb(_make_grid(raw_window_frame)))

                noisy_depth = getattr(inner, "_student_depth_noisy_m", None)
                if noisy_depth is not None:
                    noisy_depth_m_frame = _depth_tensor_to_nhw(noisy_depth)
                    noisy_window_frame = np.nan_to_num(
                        (noisy_depth_m_frame - near) / (far - near),
                        nan=0.0,
                        posinf=1.0,
                        neginf=0.0,
                    )
                    noisy_video_frames.append(_gray_to_rgb(_make_grid(noisy_window_frame)))

            if args.save_video and has_rgb:
                rgb = inner.student_camera.data.output["rgb"]
                rgb_video_frames.append(_make_rgb_grid(_rgb_tensor_to_nhwc(rgb) / 255.0))

        student_obs = last_student_obs or inner.get_student_obs()
        if has_depth:
            student_depth = student_obs["image"][:, -1].detach().float().cpu().numpy()

            raw_depth = getattr(inner, "_student_depth_raw_m", inner.student_camera.data.output["distance_to_image_plane"])
            raw_depth_m = _depth_tensor_to_nhw(raw_depth)
            raw_window = np.nan_to_num((raw_depth_m - near) / (far - near), nan=0.0, posinf=1.0, neginf=0.0)
            raw_window = np.clip(raw_window, 0.0, 1.0)

            noisy_depth = getattr(inner, "_student_depth_noisy_m", None)
            noisy_window = None
            if noisy_depth is not None:
                noisy_depth_m = _depth_tensor_to_nhw(noisy_depth)
                noisy_window = np.nan_to_num(
                    (noisy_depth_m - near) / (far - near),
                    nan=0.0,
                    posinf=1.0,
                    neginf=0.0,
                )
                noisy_window = np.clip(noisy_window, 0.0, 1.0)
                finite_pair = np.isfinite(noisy_depth_m) & np.isfinite(raw_depth_m)
                diff = noisy_depth_m[finite_pair] - raw_depth_m[finite_pair]
                finite_diff = diff[np.isfinite(diff)]
                if finite_diff.size:
                    print(
                        f"[debug reset={reset_idx:02d}] noisy_minus_raw_m "
                        f"mean={finite_diff.mean():.6f} std={finite_diff.std():.6f} "
                        f"min={finite_diff.min():.6f} max={finite_diff.max():.6f}"
                    )

            student_path = out_dir / f"reset_{reset_idx:02d}_student_depth.png"
            raw_path = out_dir / f"reset_{reset_idx:02d}_raw_depth_window.png"
            _save_depth_grid(student_path, student_depth)
            _save_depth_grid(raw_path, raw_window)
            print(f"[debug] wrote {student_path}")
            print(f"[debug] wrote {raw_path}")
            if noisy_window is not None:
                noisy_path = out_dir / f"reset_{reset_idx:02d}_noisy_depth_window.png"
                _save_depth_grid(noisy_path, noisy_window)
                print(f"[debug] wrote {noisy_path}")
            _print_depth_stats(
                f"[debug reset={reset_idx:02d}]",
                raw_depth_m,
                student_depth,
                near,
                far,
            )

        if has_rgb:
            rgb = inner.student_camera.data.output["rgb"]
            rgb_path = out_dir / f"reset_{reset_idx:02d}_rgb.png"
            _save_rgb_grid(rgb_path, _rgb_tensor_to_nhwc(rgb) / 255.0)
            print(f"[debug] wrote {rgb_path}")

    if args.save_video:
        _write_mp4(out_dir / "student_depth.mp4", depth_video_frames, args.video_fps)
        _write_mp4(out_dir / "raw_depth_window.mp4", raw_video_frames, args.video_fps)
        _write_mp4(out_dir / "noisy_depth_window.mp4", noisy_video_frames, args.video_fps)
        _write_mp4(out_dir / "rgb.mp4", rgb_video_frames, args.video_fps)
        print(f"[debug] wrote videos under {out_dir}")

    camera_data = inner.student_camera.data
    if hasattr(inner, "_student_camera_current_pos_w"):
        current = inner._student_camera_current_pos_w.detach().cpu()
        nominal = (inner.scene.env_origins + inner._student_camera_base_pos_local).detach().cpu()
        delta = current - nominal
        print(f"[debug] intended camera pos_w[0]={current[0].tolist()}")
        print(
            "[debug] intended camera pos delta range="
            f"min={delta.min(dim=0).values.tolist()} max={delta.max(dim=0).values.tolist()}"
        )
    if hasattr(camera_data, "pos_w"):
        print(f"[debug] camera pos_w[0]={camera_data.pos_w[0].detach().cpu().tolist()}")
    if hasattr(camera_data, "quat_w_ros"):
        print(
            f"[debug] camera quat_w_ros[0]="
            f"{camera_data.quat_w_ros[0].detach().cpu().tolist()}"
        )

    env.close()
    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
