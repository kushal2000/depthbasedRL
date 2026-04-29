from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF


@dataclass
class RgbAugCfg:
    enabled: bool = False
    background_prob: float = 0.0
    color_jitter_prob: float = 0.0
    motion_blur_prob: float = 0.0
    saturation_range: tuple[float, float] = (0.5, 1.5)
    contrast_range: tuple[float, float] = (0.5, 1.5)
    brightness_range: tuple[float, float] = (0.5, 1.5)
    hue_range: tuple[float, float] = (-0.15, 0.15)
    motion_blur_kernel_sizes: tuple[int, ...] = (9, 11, 13, 15, 17)
    background_dir: str | None = None


@dataclass
class DepthAugCfg:
    enabled: bool = False
    correlated_noise_std_px: float = 0.5
    correlated_noise_std_depth: float = 1.0 / 6.0
    normal_noise_std_m: float = 0.01
    pixel_dropout_prob: float = 0.003
    random_blob_prob: float = 0.003
    stick_prob: float = 0.0025
    max_stick_len_px: int = 18
    max_stick_width_px: int = 3
    random_artifact_min_m: float = -1.3
    random_artifact_max_m: float = -0.5


@dataclass
class ImageDelayCfg:
    enabled: bool = False
    queue_length_frames: int | None = None
    fixed_delay_frames: int | None = None
    max_random_delay_frames: int = 0
    resample_on_reset: bool = True


@dataclass
class VisualDRCfg:
    enabled: bool = False
    camera_pos_jitter_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    camera_rot_jitter_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    dome_light_randomization: bool = False
    dome_light_intensity_range: tuple[float, float] = (1500.0, 4000.0)
    dome_light_color_jitter: float = 0.25
    material_color_randomization: bool = False
    object_color_range: tuple[float, float] = (0.25, 1.0)
    table_color_range: tuple[float, float] = (0.2, 0.8)
    robot_color_range: tuple[float, float] = (0.4, 0.95)


@dataclass
class PreprocessCfg:
    resize_mode: str = "exact"
    pad_to_size: bool = False
    apply_rgb_float_scaling: bool = True
    zero_out_invalid_depth: bool = True


@dataclass
class ImageRobustnessCfg:
    enabled: bool = False
    visual_dr: VisualDRCfg = field(default_factory=VisualDRCfg)
    rgb_aug: RgbAugCfg = field(default_factory=RgbAugCfg)
    depth_aug: DepthAugCfg = field(default_factory=DepthAugCfg)
    image_delay: ImageDelayCfg = field(default_factory=ImageDelayCfg)
    preprocess: PreprocessCfg = field(default_factory=PreprocessCfg)


def _tuple_of_floats(value, length: int) -> tuple[float, ...]:
    if isinstance(value, (list, tuple)):
        if len(value) != length:
            raise ValueError(f"Expected length {length}, got {len(value)}")
        return tuple(float(x) for x in value)
    return tuple(float(value) for _ in range(length))


def image_robustness_cfg_from_dict(data: dict | None) -> ImageRobustnessCfg:
    data = {} if data is None else dict(data)
    visual_dr_raw = dict(data.get("visual_dr", {}))
    rgb_aug_raw = dict(data.get("rgb_aug", {}))
    depth_aug_raw = dict(data.get("depth_aug", {}))
    image_delay_raw = dict(data.get("image_delay", {}))
    preprocess_raw = dict(data.get("preprocess", {}))

    visual_dr = VisualDRCfg(
        enabled=bool(visual_dr_raw.get("enabled", False)),
        camera_pos_jitter_m=_tuple_of_floats(visual_dr_raw.get("camera_pos_jitter_m", (0.0, 0.0, 0.0)), 3),
        camera_rot_jitter_deg=_tuple_of_floats(visual_dr_raw.get("camera_rot_jitter_deg", (0.0, 0.0, 0.0)), 3),
        dome_light_randomization=bool(visual_dr_raw.get("dome_light_randomization", False)),
        dome_light_intensity_range=_tuple_of_floats(visual_dr_raw.get("dome_light_intensity_range", (1500.0, 4000.0)), 2),
        dome_light_color_jitter=float(visual_dr_raw.get("dome_light_color_jitter", 0.25)),
        material_color_randomization=bool(visual_dr_raw.get("material_color_randomization", False)),
        object_color_range=_tuple_of_floats(visual_dr_raw.get("object_color_range", (0.25, 1.0)), 2),
        table_color_range=_tuple_of_floats(visual_dr_raw.get("table_color_range", (0.2, 0.8)), 2),
        robot_color_range=_tuple_of_floats(visual_dr_raw.get("robot_color_range", (0.4, 0.95)), 2),
    )
    rgb_aug = RgbAugCfg(
        enabled=bool(rgb_aug_raw.get("enabled", False)),
        background_prob=float(rgb_aug_raw.get("background_prob", 0.0)),
        color_jitter_prob=float(rgb_aug_raw.get("color_jitter_prob", 0.0)),
        motion_blur_prob=float(rgb_aug_raw.get("motion_blur_prob", 0.0)),
        saturation_range=_tuple_of_floats(rgb_aug_raw.get("saturation_range", (0.5, 1.5)), 2),
        contrast_range=_tuple_of_floats(rgb_aug_raw.get("contrast_range", (0.5, 1.5)), 2),
        brightness_range=_tuple_of_floats(rgb_aug_raw.get("brightness_range", (0.5, 1.5)), 2),
        hue_range=_tuple_of_floats(rgb_aug_raw.get("hue_range", (-0.15, 0.15)), 2),
        motion_blur_kernel_sizes=tuple(int(x) for x in rgb_aug_raw.get("motion_blur_kernel_sizes", (9, 11, 13, 15, 17))),
        background_dir=rgb_aug_raw.get("background_dir"),
    )
    depth_aug = DepthAugCfg(
        enabled=bool(depth_aug_raw.get("enabled", False)),
        correlated_noise_std_px=float(depth_aug_raw.get("correlated_noise_std_px", 0.5)),
        correlated_noise_std_depth=float(depth_aug_raw.get("correlated_noise_std_depth", 1.0 / 6.0)),
        normal_noise_std_m=float(depth_aug_raw.get("normal_noise_std_m", 0.01)),
        pixel_dropout_prob=float(depth_aug_raw.get("pixel_dropout_prob", 0.003)),
        random_blob_prob=float(depth_aug_raw.get("random_blob_prob", 0.003)),
        stick_prob=float(depth_aug_raw.get("stick_prob", 0.0025)),
        max_stick_len_px=int(depth_aug_raw.get("max_stick_len_px", 18)),
        max_stick_width_px=int(depth_aug_raw.get("max_stick_width_px", 3)),
        random_artifact_min_m=float(depth_aug_raw.get("random_artifact_min_m", -1.3)),
        random_artifact_max_m=float(depth_aug_raw.get("random_artifact_max_m", -0.5)),
    )
    image_delay = ImageDelayCfg(
        enabled=bool(image_delay_raw.get("enabled", False)),
        queue_length_frames=(
            None if image_delay_raw.get("queue_length_frames") is None else int(image_delay_raw.get("queue_length_frames"))
        ),
        fixed_delay_frames=(
            None if image_delay_raw.get("fixed_delay_frames") is None else int(image_delay_raw.get("fixed_delay_frames"))
        ),
        max_random_delay_frames=int(image_delay_raw.get("max_random_delay_frames", 0)),
        resample_on_reset=bool(image_delay_raw.get("resample_on_reset", True)),
    )
    preprocess = PreprocessCfg(
        resize_mode=str(preprocess_raw.get("resize_mode", "exact")),
        pad_to_size=bool(preprocess_raw.get("pad_to_size", False)),
        apply_rgb_float_scaling=bool(preprocess_raw.get("apply_rgb_float_scaling", True)),
        zero_out_invalid_depth=bool(preprocess_raw.get("zero_out_invalid_depth", True)),
    )
    return ImageRobustnessCfg(
        enabled=bool(data.get("enabled", any((
            visual_dr.enabled,
            rgb_aug.enabled,
            depth_aug.enabled,
            image_delay.enabled,
        )))),
        visual_dr=visual_dr,
        rgb_aug=rgb_aug,
        depth_aug=depth_aug,
        image_delay=image_delay,
        preprocess=preprocess,
    )


def _resize_image_exact(image: torch.Tensor, height: int, width: int, pad_to_size: bool) -> torch.Tensor:
    _, _, src_h, src_w = image.shape
    if src_h == height and src_w == width:
        return image
    if not pad_to_size:
        mode = "bilinear" if image.shape[1] > 1 else "bilinear"
        return F.interpolate(image, size=(height, width), mode=mode, align_corners=False)
    scale = min(height / src_h, width / src_w)
    resized_h = max(1, int(round(src_h * scale)))
    resized_w = max(1, int(round(src_w * scale)))
    resized = F.interpolate(image, size=(resized_h, resized_w), mode="bilinear", align_corners=False)
    pad_h = height - resized_h
    pad_w = width - resized_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    return F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom))


def preprocess_policy_images(
    images: dict[str, torch.Tensor],
    modality: str,
    out_height: int,
    out_width: int,
    preprocess_cfg: PreprocessCfg,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    processed: dict[str, torch.Tensor] = {}
    if "rgb" in images:
        rgb = images["rgb"].float()
        if preprocess_cfg.apply_rgb_float_scaling and rgb.max() > 1.5:
            rgb = rgb / 255.0
        rgb = _resize_image_exact(rgb, out_height, out_width, preprocess_cfg.pad_to_size)
        processed["rgb"] = torch.clamp(rgb, 0.0, 1.0)
    if "depth" in images:
        depth = images["depth"].float()
        depth = _resize_image_exact(depth, out_height, out_width, preprocess_cfg.pad_to_size)
        processed["depth"] = depth
    if "depth_metric" in images:
        depth_metric = images["depth_metric"].float()
        depth_metric = _resize_image_exact(depth_metric, out_height, out_width, preprocess_cfg.pad_to_size)
        processed["depth_metric"] = depth_metric

    if modality == "depth":
        return processed["depth"], processed
    if modality == "rgb":
        return processed["rgb"], processed
    if modality == "rgbd":
        return torch.cat([processed["rgb"], processed["depth"]], dim=1), processed
    raise ValueError(f"Unsupported modality: {modality}")


class TrainImageRobustifier:
    def __init__(
        self,
        cfg: ImageRobustnessCfg,
        num_envs: int,
        device: torch.device,
        depth_preprocess_mode: str = "window_normalize",
        depth_min_m: float = 0.0,
        depth_max_m: float = 1.0,
    ):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device
        self.depth_preprocess_mode = depth_preprocess_mode
        self.depth_min_m = float(depth_min_m)
        self.depth_max_m = float(depth_max_m)
        self._background_images = self._load_background_images()
        self._delay_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._delay_queue: list[torch.Tensor] = []

    def preprocess_depth_metric(self, depth_m: torch.Tensor) -> torch.Tensor:
        depth_m = depth_m.float()
        if self.depth_preprocess_mode == "window_normalize":
            valid = (depth_m >= self.depth_min_m) & (depth_m <= self.depth_max_m)
            normalized = (depth_m - self.depth_min_m) / max(self.depth_max_m - self.depth_min_m, 1e-6)
            return torch.where(valid, normalized, torch.zeros_like(depth_m))
        if self.depth_preprocess_mode == "clip_divide":
            return torch.clamp(depth_m, self.depth_min_m, self.depth_max_m) / max(self.depth_max_m, 1e-6)
        if self.depth_preprocess_mode == "metric":
            valid = (depth_m >= self.depth_min_m) & (depth_m <= self.depth_max_m)
            return torch.where(valid, depth_m, torch.zeros_like(depth_m))
        raise ValueError(f"Unsupported depth_preprocess_mode={self.depth_preprocess_mode!r}")

    def _max_random_delay_frames(self) -> int:
        if self.cfg.image_delay.queue_length_frames is not None:
            # Queue length M samples delays in [0, M - 1]. M=1 is latest/no delay.
            return max(int(self.cfg.image_delay.queue_length_frames) - 1, 0)
        return max(int(self.cfg.image_delay.max_random_delay_frames), 0)

    def _load_background_images(self) -> list[torch.Tensor]:
        bg_dir = self.cfg.rgb_aug.background_dir
        if not bg_dir:
            return []
        bg_path = Path(bg_dir)
        if not bg_path.exists():
            return []
        import imageio.v3 as iio

        images: list[torch.Tensor] = []
        for path in sorted(bg_path.glob("*.jpg"))[:256]:
            img = iio.imread(path)
            if img.ndim != 3 or img.shape[2] < 3:
                continue
            images.append(torch.from_numpy(img[..., :3]).permute(2, 0, 1).float() / 255.0)
        return images

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
            self._delay_queue = []
        elif self._delay_queue:
            for queue_frame in self._delay_queue:
                queue_frame[env_ids] = 0.0
        if not self.cfg.image_delay.enabled:
            return
        if self.cfg.image_delay.fixed_delay_frames is not None:
            self._delay_steps[env_ids] = int(max(self.cfg.image_delay.fixed_delay_frames, 0))
            return
        max_delay = self._max_random_delay_frames()
        if max_delay <= 0:
            self._delay_steps[env_ids] = 0
            return
        self._delay_steps[env_ids] = torch.randint(0, max_delay + 1, (env_ids.numel(),), device=self.device)

    def _sample_background(self, height: int, width: int) -> torch.Tensor | None:
        if not self._background_images:
            return None
        bg = random.choice(self._background_images).to(self.device)
        if bg.shape[-2] < height or bg.shape[-1] < width:
            bg = F.interpolate(bg.unsqueeze(0), size=(max(height, bg.shape[-2]), max(width, bg.shape[-1])), mode="bilinear", align_corners=False)[0]
        top = 0 if bg.shape[-2] == height else random.randint(0, bg.shape[-2] - height)
        left = 0 if bg.shape[-1] == width else random.randint(0, bg.shape[-1] - width)
        return bg[:, top : top + height, left : left + width]

    def _apply_rgb_background(self, rgb: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None or self.cfg.rgb_aug.background_prob <= 0.0:
            return rgb
        out = rgb.clone()
        for env_id in range(rgb.shape[0]):
            if random.random() >= self.cfg.rgb_aug.background_prob:
                continue
            bg = self._sample_background(rgb.shape[-2], rgb.shape[-1])
            if bg is None:
                continue
            env_mask = mask[env_id]
            if env_mask.dim() == 3:
                env_mask = env_mask[0]
            env_mask = env_mask.bool().unsqueeze(0)
            out[env_id] = torch.where(env_mask, out[env_id], bg)
        return out

    def _apply_rgb_color_jitter(self, rgb: torch.Tensor) -> torch.Tensor:
        if self.cfg.rgb_aug.color_jitter_prob <= 0.0:
            return rgb
        out = rgb.clone()
        for env_id in range(rgb.shape[0]):
            if random.random() >= self.cfg.rgb_aug.color_jitter_prob:
                continue
            img = out[env_id]
            sat = random.uniform(*self.cfg.rgb_aug.saturation_range)
            con = random.uniform(*self.cfg.rgb_aug.contrast_range)
            bri = random.uniform(*self.cfg.rgb_aug.brightness_range)
            hue = random.uniform(*self.cfg.rgb_aug.hue_range)
            img = TF.adjust_saturation(img, sat)
            img = TF.adjust_contrast(img, con)
            img = TF.adjust_brightness(img, bri)
            img = TF.adjust_hue(img, hue)
            out[env_id] = torch.clamp(img, 0.0, 1.0)
        return out

    def _motion_blur_kernel(self, kernel_size: int, angle_rad: float) -> torch.Tensor:
        base = torch.zeros((kernel_size, kernel_size), dtype=torch.float32)
        base[kernel_size // 2] = 1.0
        base = base / base.sum()
        theta = torch.tensor(
            [
                [np.cos(angle_rad), -np.sin(angle_rad), 0.0],
                [np.sin(angle_rad), np.cos(angle_rad), 0.0],
            ],
            dtype=torch.float32,
        )
        grid = F.affine_grid(theta.unsqueeze(0), size=(1, 1, kernel_size, kernel_size), align_corners=False)
        return F.grid_sample(base.view(1, 1, kernel_size, kernel_size), grid, align_corners=False)[0, 0]

    def _apply_rgb_motion_blur(self, rgb: torch.Tensor) -> torch.Tensor:
        if self.cfg.rgb_aug.motion_blur_prob <= 0.0:
            return rgb
        out = rgb.clone()
        for env_id in range(rgb.shape[0]):
            if random.random() >= self.cfg.rgb_aug.motion_blur_prob:
                continue
            kernel_size = random.choice(self.cfg.rgb_aug.motion_blur_kernel_sizes)
            angle = random.uniform(0.0, 2.0 * np.pi)
            kernel = self._motion_blur_kernel(kernel_size, angle).to(self.device)
            kernel = kernel / torch.clamp(kernel.sum(), min=1e-6)
            weight = kernel.view(1, 1, kernel_size, kernel_size).repeat(3, 1, 1, 1)
            blurred = F.conv2d(
                out[env_id : env_id + 1],
                weight,
                padding=kernel_size // 2,
                groups=3,
            )
            out[env_id] = torch.clamp(0.7 * blurred[0] + 0.3 * out[env_id], 0.0, 1.0)
        return out

    def _apply_depth_aug_metric(self, depth_m: torch.Tensor) -> torch.Tensor:
        if not self.cfg.depth_aug.enabled:
            return depth_m
        out = depth_m.clone()
        cfg = self.cfg.depth_aug
        if cfg.correlated_noise_std_px > 0.0:
            small = F.interpolate(
                torch.randn_like(out),
                scale_factor=0.25,
                mode="bilinear",
                align_corners=False,
                recompute_scale_factor=False,
            )
            correlated = F.interpolate(small, size=out.shape[-2:], mode="bilinear", align_corners=False)
            out = out + correlated * cfg.correlated_noise_std_depth
        if cfg.normal_noise_std_m > 0.0:
            out = out + torch.randn_like(out) * cfg.normal_noise_std_m
        if cfg.pixel_dropout_prob > 0.0:
            dropout = torch.rand_like(out) < cfg.pixel_dropout_prob
            out = torch.where(dropout, torch.zeros_like(out), out)
        if cfg.random_blob_prob > 0.0:
            seeds = torch.rand_like(out) < cfg.random_blob_prob
            if torch.any(seeds):
                artifacts = torch.empty_like(out).uniform_(cfg.random_artifact_min_m, cfg.random_artifact_max_m)
                out = torch.where(seeds, artifacts, out)
        if cfg.stick_prob > 0.0:
            for env_id in range(out.shape[0]):
                if random.random() >= cfg.stick_prob:
                    continue
                length = random.randint(1, max(cfg.max_stick_len_px, 1))
                width = random.randint(1, max(cfg.max_stick_width_px, 1))
                value = random.uniform(cfg.random_artifact_min_m, cfg.random_artifact_max_m)
                y = random.randint(0, out.shape[-2] - 1)
                x = random.randint(0, out.shape[-1] - 1)
                horizontal = random.random() < 0.5
                if horizontal:
                    out[env_id, 0, y : min(y + width, out.shape[-2]), x : min(x + length, out.shape[-1])] = value
                else:
                    out[env_id, 0, y : min(y + length, out.shape[-2]), x : min(x + width, out.shape[-1])] = value
        return out

    def _apply_delay(self, policy_image: torch.Tensor, reset_env_ids: torch.Tensor | None) -> torch.Tensor:
        if not self.cfg.image_delay.enabled:
            return policy_image
        if reset_env_ids is not None and reset_env_ids.numel() > 0 and self.cfg.image_delay.resample_on_reset:
            self.reset(reset_env_ids)
        if not self._delay_queue:
            max_delay = max(int(self._delay_steps.max().item()), 0)
            self._delay_queue = [policy_image.clone() for _ in range(max_delay + 1)]
            return policy_image
        self._delay_queue.insert(0, policy_image.clone())
        max_keep = max(int(self._delay_steps.max().item()), 0) + 1
        self._delay_queue = self._delay_queue[:max_keep]
        delayed = policy_image.clone()
        for env_id in range(policy_image.shape[0]):
            delay = int(self._delay_steps[env_id].item())
            delay = min(delay, len(self._delay_queue) - 1)
            delayed[env_id] = self._delay_queue[delay][env_id]
        return delayed

    def apply_train_time(
        self,
        processed_images: dict[str, torch.Tensor],
        policy_image: torch.Tensor,
        reset_env_ids: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        images = {key: value.clone() for key, value in processed_images.items()}
        if "rgb" in images and self.cfg.rgb_aug.enabled:
            mask = None
            if "depth" in images:
                mask = images["depth"] > 0
            images["rgb"] = self._apply_rgb_background(images["rgb"], mask)
            images["rgb"] = self._apply_rgb_color_jitter(images["rgb"])
            images["rgb"] = self._apply_rgb_motion_blur(images["rgb"])
        if "depth_metric" in images:
            images["depth_metric_noisy"] = self._apply_depth_aug_metric(images["depth_metric"])
            images["depth"] = self.preprocess_depth_metric(images["depth_metric_noisy"])
        elif "depth" in images:
            images["depth"] = images["depth"].clone()

        if policy_image.shape[1] == 1 and "depth" in images:
            policy_image = images["depth"]
        elif policy_image.shape[1] == 3 and "rgb" in images:
            policy_image = images["rgb"]
        elif policy_image.shape[1] == 4 and "rgb" in images and "depth" in images:
            policy_image = torch.cat([images["rgb"], images["depth"]], dim=1)

        delayed = self._apply_delay(policy_image, reset_env_ids)
        return images, delayed
