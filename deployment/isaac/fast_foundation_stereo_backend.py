"""Thin runtime wrapper for Fast-FoundationStereo.

This module intentionally keeps Fast-FoundationStereo imports lazy and local so
the normal Isaac depth ROS node path does not depend on the FFS repository or
its Python dependencies.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf


class FastFoundationStereoDepth:
    """Load a Fast-FoundationStereo checkpoint and infer metric depth from RGB stereo."""

    def __init__(
        self,
        *,
        repo_root: str | Path,
        model_path: str | Path,
        baseline_m: float,
        valid_iters: int = 4,
        max_disp: int = 192,
        engine_dir: str | Path | None = None,
        device: str = "cuda",
        hiera: bool = False,
        optimize_build_volume: str = "pytorch1",
        remove_invisible: bool = True,
        zfar_m: float = 10.0,
    ) -> None:
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.model_path = Path(model_path).expanduser().resolve()
        self.baseline_m = float(baseline_m)
        self.valid_iters = int(valid_iters)
        self.max_disp = int(max_disp)
        self.engine_dir = Path(engine_dir).expanduser().resolve() if engine_dir else None
        self.device = torch.device(device)
        self.hiera = bool(hiera)
        self.optimize_build_volume = str(optimize_build_volume)
        self.remove_invisible = bool(remove_invisible)
        self.zfar_m = float(zfar_m)
        self.backend = "pytorch"
        self._trt_image_size_hw: tuple[int, int] | None = None

        if not self.repo_root.exists():
            raise FileNotFoundError(f"Fast-FoundationStereo repo not found: {self.repo_root}")
        if self.engine_dir is None and not self.model_path.exists():
            raise FileNotFoundError(f"Fast-FoundationStereo checkpoint not found: {self.model_path}")
        if self.baseline_m <= 0.0:
            raise ValueError(f"baseline_m must be positive, got {self.baseline_m}")

        if str(self.repo_root) not in sys.path:
            sys.path.insert(0, str(self.repo_root))

        # FFS decorates a helper with torch.compile. In this mixed IsaacSim/ROS
        # process, Inductor can spend minutes compiling on the first small
        # rollout frame. Keep the debug backend eager unless explicitly
        # overridden before process start.
        os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
        os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

        from core.utils.utils import InputPadder
        from Utils import AMP_DTYPE

        self._InputPadder = InputPadder
        self._amp_dtype = AMP_DTYPE
        if self.engine_dir is not None:
            self.model = self._load_trt_model()
            self.backend = "tensorrt"
        else:
            self.model = self._load_model()
        self.last_disparity: np.ndarray | None = None

    @staticmethod
    def _resolve_onnx_cfg_path(engine_dir: Path) -> Path:
        candidates = (engine_dir / "onnx.yaml", engine_dir.parent / "onnx.yaml")
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(f"onnx.yaml not found for FFS TensorRT engine. Looked in: {candidates}")

    def _load_trt_model(self):
        if self.engine_dir is None:
            raise RuntimeError("engine_dir is not set")
        feature_engine = self.engine_dir / "feature_runner.engine"
        post_engine = self.engine_dir / "post_runner.engine"
        if not feature_engine.exists() or not post_engine.exists():
            raise FileNotFoundError(
                f"Expected TensorRT engines at {feature_engine} and {post_engine}. "
                "Build them with Fast-FoundationStereo scripts/make_onnx.py and build_trt_engine.py."
            )
        cfg_path = self._resolve_onnx_cfg_path(self.engine_dir)
        with cfg_path.open("r") as f:
            cfg = yaml.safe_load(f)
        cfg["valid_iters"] = self.valid_iters
        cfg["max_disp"] = self.max_disp
        args = OmegaConf.create(cfg)
        image_size = tuple(int(v) for v in args.image_size)
        if len(image_size) != 2:
            raise ValueError(f"FFS TensorRT onnx.yaml image_size must be [H, W], got {image_size}")
        self._trt_image_size_hw = image_size

        from core.foundation_stereo import TrtRunner

        return TrtRunner(args, str(feature_engine), str(post_engine))

    def _load_model(self):
        cfg_path = self.model_path.parent / "cfg.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Expected FFS cfg.yaml next to checkpoint: {cfg_path}")
        with cfg_path.open("r") as f:
            cfg = yaml.safe_load(f)
        cfg["model_dir"] = str(self.model_path)
        cfg["valid_iters"] = self.valid_iters
        cfg["max_disp"] = self.max_disp
        args = OmegaConf.create(cfg)

        model = torch.load(str(self.model_path), map_location="cpu", weights_only=False)
        model.args.valid_iters = args.valid_iters
        model.args.max_disp = args.max_disp
        model.to(self.device).eval()
        return model

    @torch.no_grad()
    def infer_depth(self, left_rgb: np.ndarray, right_rgb: np.ndarray, *, fx_px: float) -> np.ndarray:
        """Return metric depth in meters from rectified left/right RGB images.

        Args:
            left_rgb: ``H x W x 3`` uint8/float RGB image from the left camera.
            right_rgb: ``H x W x 3`` uint8/float RGB image from the right camera.
            fx_px: Left-camera focal length in pixels at this render resolution.
        """
        left_rgb = self._validate_rgb(left_rgb, "left_rgb")
        right_rgb = self._validate_rgb(right_rgb, "right_rgb")
        if left_rgb.shape != right_rgb.shape:
            raise ValueError(f"Stereo images must match shape, got {left_rgb.shape} and {right_rgb.shape}")
        if fx_px <= 0.0:
            raise ValueError(f"fx_px must be positive, got {fx_px}")

        height, width = left_rgb.shape[:2]
        left = torch.as_tensor(left_rgb, device=self.device).float()[None].permute(0, 3, 1, 2)
        right = torch.as_tensor(right_rgb, device=self.device).float()[None].permute(0, 3, 1, 2)

        if self.backend == "tensorrt":
            if self._trt_image_size_hw != (height, width):
                raise ValueError(
                    "FFS TensorRT engine image size does not match rendered stereo images: "
                    f"engine={self._trt_image_size_hw}, images={(height, width)}. "
                    "Use an engine built for the capture resolution."
                )
            disp = self.model.forward(left.contiguous(), right.contiguous()).float()
        else:
            padder = self._InputPadder(left.shape, divis_by=32, force_square=False)
            left, right = padder.pad(left, right)

            with torch.amp.autocast("cuda", enabled=self.device.type == "cuda", dtype=self._amp_dtype):
                if self.hiera:
                    disp = self.model.run_hierachical(
                        left, right, iters=self.valid_iters, test_mode=True, small_ratio=0.5
                    )
                else:
                    disp = self.model.forward(
                        left,
                        right,
                        iters=self.valid_iters,
                        test_mode=True,
                        optimize_build_volume=self.optimize_build_volume,
                    )
            disp = padder.unpad(disp.float())
        disp_np = disp.detach().cpu().numpy().reshape(height, width).clip(0.0, None)

        if self.remove_invisible:
            yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
            disp_np = disp_np.copy()
            disp_np[(xx - disp_np) < 0] = np.inf

        self.last_disparity = disp_np.astype(np.float32, copy=False)
        safe_disp = np.where(np.isfinite(disp_np) & (disp_np > 1.0e-6), disp_np, np.nan)
        depth_m = (float(fx_px) * self.baseline_m / safe_disp).astype(np.float32)
        depth_m = np.nan_to_num(depth_m, nan=self.zfar_m, posinf=self.zfar_m, neginf=0.0)
        return np.clip(depth_m, 0.0, self.zfar_m).astype(np.float32, copy=False)

    @staticmethod
    def _validate_rgb(image: np.ndarray, name: str) -> np.ndarray:
        arr = np.asarray(image)
        if arr.ndim != 3 or arr.shape[2] < 3:
            raise ValueError(f"{name} must have shape HxWx3, got {arr.shape}")
        arr = arr[..., :3]
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(arr)


def write_stereo_debug(
    out_dir: str | Path,
    *,
    step_idx: int,
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
    disparity: np.ndarray | None,
    depth_m: np.ndarray,
) -> None:
    """Save lightweight FFS stereo debug artifacts for visual inspection."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / f"step_{step_idx:08d}"
    np.savez_compressed(
        f"{prefix}.npz",
        left_rgb=left_rgb,
        right_rgb=right_rgb,
        disparity=disparity,
        depth_m=depth_m,
    )
    cv2.imwrite(f"{prefix}_left_rgb.png", cv2.cvtColor(left_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{prefix}_right_rgb.png", cv2.cvtColor(right_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{prefix}_depth.png", cv2.cvtColor(_colorize(depth_m, low=0.4, high=1.2), cv2.COLOR_RGB2BGR))
    if disparity is not None:
        finite = disparity[np.isfinite(disparity)]
        if finite.size:
            lo, hi = float(np.quantile(finite, 0.02)), float(np.quantile(finite, 0.98))
        else:
            lo, hi = 0.0, 1.0
        cv2.imwrite(
            f"{prefix}_disparity.png",
            cv2.cvtColor(_colorize(disparity, low=lo, high=max(hi, lo + 1.0e-6)), cv2.COLOR_RGB2BGR),
        )


def _colorize(values: np.ndarray, *, low: float, high: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    safe = np.nan_to_num(values, nan=high, posinf=high, neginf=low)
    norm = np.clip((safe - low) / max(high - low, 1.0e-6), 0.0, 1.0)
    bgr = cv2.applyColorMap((norm * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
