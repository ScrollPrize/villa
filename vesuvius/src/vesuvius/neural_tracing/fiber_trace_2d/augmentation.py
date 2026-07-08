from __future__ import annotations

import math
import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class FiberStripAugmentConfig:
    enabled: bool = False
    seed: int = 1
    device: str = "auto"
    shift_x: float = 32.0
    shift_y: float = 32.0
    rotation_degrees: float = 180.0
    shear_x: float = 1.0
    shear_y: float = 1.0
    scale_min: float = math.sqrt(0.5)
    scale_max: float = math.sqrt(2.0)
    smooth_offset: float = 8.0
    smooth_offset_stride: float = 16.0
    brightness: float = 0.25
    contrast_min: float = 0.5
    contrast_max: float = 2.0
    gamma_min: float = 0.5
    gamma_max: float = 2.0
    noise_std: float = 0.125
    blur_sigma: float = 2.0


@dataclass(frozen=True)
class AugmentPadding:
    y: int
    x: int


def _stable_seed(*parts: Any) -> int:
    digest = hashlib.blake2b(digest_size=16)
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return int.from_bytes(digest.digest(), "little", signed=False)


@dataclass(frozen=True)
class FiberStripAugmentParams:
    shift_x: float = 0.0
    shift_y: float = 0.0
    rotation_degrees: float = 0.0
    shear_x: float = 0.0
    shear_y: float = 0.0
    scale: float = 1.0
    smooth_offset: float = 0.0
    smooth_offset_stride: float = 16.0
    smooth_offset_seed: int = 0
    flip_x: bool = False
    flip_y: bool = False
    brightness: float = 0.0
    contrast: float = 1.0
    gamma: float = 1.0
    noise_std: float = 0.0
    blur_sigma: float = 0.0
    noise_seed: int = 0


def augment_config_from_mapping(raw: dict[str, Any]) -> FiberStripAugmentConfig:
    if "augment_contrast_min" in raw or "augment_contrast_max" in raw:
        contrast_min = float(raw.get("augment_contrast_min", 0.5))
        contrast_max = float(raw.get("augment_contrast_max", 2.0))
    else:
        legacy = raw.get("augment_contrast")
        if legacy is None:
            contrast_min = 0.5
            contrast_max = 2.0
        else:
            legacy_value = float(legacy)
            contrast_min = max(0.0, 1.0 - legacy_value)
            contrast_max = 1.0 + legacy_value
    patch_shape_hw = raw.get("patch_shape_hw", (128, 128))
    patch_height = float(patch_shape_hw[0])
    patch_width = float(patch_shape_hw[1])
    return FiberStripAugmentConfig(
        enabled=bool(raw.get("augment_enabled", False)),
        seed=int(raw.get("augment_seed", raw.get("seed", 1))),
        device=str(raw.get("augment_device", "auto")),
        shift_x=float(raw.get("augment_shift_x", patch_width * 0.25)),
        shift_y=float(raw.get("augment_shift_y", patch_height * 0.25)),
        rotation_degrees=float(raw.get("augment_rotation_degrees", 180.0)),
        shear_x=float(raw.get("augment_shear_x", 1.0)),
        shear_y=float(raw.get("augment_shear_y", 1.0)),
        scale_min=float(raw.get("augment_scale_min", math.sqrt(0.5))),
        scale_max=float(raw.get("augment_scale_max", math.sqrt(2.0))),
        smooth_offset=float(raw.get("augment_smooth_offset", 8.0)),
        smooth_offset_stride=float(raw.get("augment_smooth_offset_stride", 16.0)),
        brightness=float(raw.get("augment_brightness", 0.25)),
        contrast_min=contrast_min,
        contrast_max=contrast_max,
        gamma_min=float(raw.get("augment_gamma_min", 0.5)),
        gamma_max=float(raw.get("augment_gamma_max", 2.0)),
        noise_std=float(raw.get("augment_noise_std", 0.125)),
        blur_sigma=float(raw.get("augment_blur_sigma", 2.0)),
    )


def resolve_torch_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def limit_augmentation_rows(
    config: FiberStripAugmentConfig, sample_index: int
) -> tuple[list[tuple[str, FiberStripAugmentParams]], list[tuple[str, FiberStripAugmentParams]]]:
    seed = int(_stable_seed(config.seed, int(sample_index), "limits") & 0x7FFFFFFF)
    smooth_seed = int(_stable_seed(config.seed, int(sample_index), "smooth", "limits") & 0x7FFFFFFF)
    lower = [
        ("unaugmented", FiberStripAugmentParams(noise_seed=seed)),
        ("shift_x_min", FiberStripAugmentParams(shift_x=-config.shift_x, noise_seed=seed)),
        ("shift_y_min", FiberStripAugmentParams(shift_y=-config.shift_y, noise_seed=seed)),
        ("rotate_min", FiberStripAugmentParams(rotation_degrees=-config.rotation_degrees, noise_seed=seed)),
        ("shear_x_min", FiberStripAugmentParams(shear_x=-config.shear_x, noise_seed=seed)),
        ("shear_y_min", FiberStripAugmentParams(shear_y=-config.shear_y, noise_seed=seed)),
        ("scale_min", FiberStripAugmentParams(scale=config.scale_min, noise_seed=seed)),
        (
            "smooth_min",
            FiberStripAugmentParams(
                smooth_offset=-config.smooth_offset,
                smooth_offset_stride=config.smooth_offset_stride,
                smooth_offset_seed=smooth_seed,
                noise_seed=seed,
            ),
        ),
        ("flip_x_off", FiberStripAugmentParams(flip_x=False, noise_seed=seed)),
        ("flip_y_off", FiberStripAugmentParams(flip_y=False, noise_seed=seed)),
        ("brightness_min", FiberStripAugmentParams(brightness=-config.brightness, noise_seed=seed)),
        ("contrast_min", FiberStripAugmentParams(contrast=config.contrast_min, noise_seed=seed)),
        ("gamma_min", FiberStripAugmentParams(gamma=config.gamma_min, noise_seed=seed)),
        ("noise_min", FiberStripAugmentParams(noise_std=0.0, noise_seed=seed)),
        ("blur_min", FiberStripAugmentParams(blur_sigma=0.0, noise_seed=seed)),
    ]
    upper = [
        ("unaugmented", FiberStripAugmentParams(noise_seed=seed)),
        ("shift_x_max", FiberStripAugmentParams(shift_x=config.shift_x, noise_seed=seed)),
        ("shift_y_max", FiberStripAugmentParams(shift_y=config.shift_y, noise_seed=seed)),
        ("rotate_max", FiberStripAugmentParams(rotation_degrees=config.rotation_degrees, noise_seed=seed)),
        ("shear_x_max", FiberStripAugmentParams(shear_x=config.shear_x, noise_seed=seed)),
        ("shear_y_max", FiberStripAugmentParams(shear_y=config.shear_y, noise_seed=seed)),
        ("scale_max", FiberStripAugmentParams(scale=config.scale_max, noise_seed=seed)),
        (
            "smooth_max",
            FiberStripAugmentParams(
                smooth_offset=config.smooth_offset,
                smooth_offset_stride=config.smooth_offset_stride,
                smooth_offset_seed=smooth_seed,
                noise_seed=seed,
            ),
        ),
        ("flip_x_on", FiberStripAugmentParams(flip_x=True, noise_seed=seed)),
        ("flip_y_on", FiberStripAugmentParams(flip_y=True, noise_seed=seed)),
        ("brightness_max", FiberStripAugmentParams(brightness=config.brightness, noise_seed=seed)),
        ("contrast_max", FiberStripAugmentParams(contrast=config.contrast_max, noise_seed=seed)),
        ("gamma_max", FiberStripAugmentParams(gamma=config.gamma_max, noise_seed=seed)),
        ("noise_max", FiberStripAugmentParams(noise_std=config.noise_std, noise_seed=seed)),
        ("blur_max", FiberStripAugmentParams(blur_sigma=config.blur_sigma, noise_seed=seed)),
    ]
    return lower, upper


def random_combined_augmentation(
    config: FiberStripAugmentConfig, sample_index: int, variant_index: int = 0
) -> FiberStripAugmentParams:
    rng = np.random.default_rng(_stable_seed(config.seed, int(sample_index), "combined", int(variant_index)))
    return FiberStripAugmentParams(
        shift_x=float(rng.uniform(-config.shift_x, config.shift_x)),
        shift_y=float(rng.uniform(-config.shift_y, config.shift_y)),
        rotation_degrees=float(rng.uniform(-config.rotation_degrees, config.rotation_degrees)),
        shear_x=float(rng.uniform(-config.shear_x, config.shear_x)),
        shear_y=float(rng.uniform(-config.shear_y, config.shear_y)),
        scale=float(rng.uniform(config.scale_min, config.scale_max)),
        smooth_offset=float(rng.uniform(-config.smooth_offset, config.smooth_offset)),
        smooth_offset_stride=float(config.smooth_offset_stride),
        smooth_offset_seed=int(_stable_seed(config.seed, int(sample_index), "smooth", int(variant_index)) & 0x7FFFFFFF),
        flip_x=bool(rng.random() < 0.5),
        flip_y=bool(rng.random() < 0.5),
        brightness=float(rng.uniform(-config.brightness, config.brightness)),
        contrast=float(rng.uniform(config.contrast_min, config.contrast_max)),
        gamma=float(rng.uniform(config.gamma_min, config.gamma_max)),
        noise_std=float(config.noise_std * rng.random()),
        blur_sigma=float(config.blur_sigma * rng.random()),
        noise_seed=int(_stable_seed(config.seed, int(sample_index), "noise", int(variant_index)) & 0x7FFFFFFF),
    )


def _pixel_grid(height: int, width: int, *, device: torch.device) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return torch.stack([xx, yy], dim=-1)


def _cubic_interp_1d(values: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    n = int(values.numel())
    if n <= 1:
        return values.new_full(positions.shape, float(values[0]) if n == 1 else 0.0)
    p1 = torch.floor(positions).to(torch.long)
    t = positions - p1.to(torch.float32)
    p0 = torch.clamp(p1 - 1, 0, n - 1)
    p1c = torch.clamp(p1, 0, n - 1)
    p2 = torch.clamp(p1 + 1, 0, n - 1)
    p3 = torch.clamp(p1 + 2, 0, n - 1)
    v0 = values[p0]
    v1 = values[p1c]
    v2 = values[p2]
    v3 = values[p3]
    return 0.5 * (
        (2.0 * v1)
        + (-v0 + v2) * t
        + (2.0 * v0 - 5.0 * v1 + 4.0 * v2 - v3) * t * t
        + (-v0 + 3.0 * v1 - 3.0 * v2 + v3) * t * t * t
    )


def smooth_offset_field(
    height: int,
    width: int,
    *,
    amplitude: float,
    stride: float,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    amplitude = float(amplitude)
    if amplitude == 0.0:
        return torch.zeros((int(height), int(width)), dtype=torch.float32, device=device)
    stride = max(float(stride), 1.0)
    control_count = max(2, int(math.ceil(max(int(width) - 1, 1) / stride)) + 3)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    controls = (torch.rand(control_count, dtype=torch.float32, device=device, generator=generator) * 2.0 - 1.0) * abs(
        amplitude
    )
    if amplitude < 0.0:
        controls = -controls
    x = torch.arange(int(width), dtype=torch.float32, device=device) / stride + 1.0
    offsets = _cubic_interp_1d(controls, x)
    return offsets.view(1, int(width)).expand(int(height), int(width))


def source_coordinate_grid(
    height: int, width: int, params: FiberStripAugmentParams, *, device: torch.device
) -> torch.Tensor:
    return source_coordinate_grid_for_output(height, width, height, width, params, device=device)


def source_coordinate_grid_for_output(
    output_height: int,
    output_width: int,
    source_height: int,
    source_width: int,
    params: FiberStripAugmentParams,
    *,
    device: torch.device,
) -> torch.Tensor:
    output_center_x = (float(output_width) - 1.0) * 0.5
    output_center_y = (float(output_height) - 1.0) * 0.5
    source_center_x = (float(source_width) - 1.0) * 0.5
    source_center_y = (float(source_height) - 1.0) * 0.5
    coords = _pixel_grid(output_height, output_width, device=device)
    x = coords[..., 0] - output_center_x
    y = coords[..., 1] - output_center_y

    if params.flip_x:
        x = -x
    if params.flip_y:
        y = -y
    scale = max(float(params.scale), 1.0e-6)
    x = x / scale
    y = y / scale

    x_sheared = x + float(params.shear_x) * y
    y_sheared = y + float(params.shear_y) * x

    angle = math.radians(float(params.rotation_degrees))
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    src_x = cos_a * x_sheared - sin_a * y_sheared + source_center_x - float(params.shift_x)
    src_y = sin_a * x_sheared + cos_a * y_sheared + source_center_y - float(params.shift_y)
    if float(params.smooth_offset) != 0.0:
        src_y = src_y + smooth_offset_field(
            output_height,
            output_width,
            amplitude=float(params.smooth_offset),
            stride=float(params.smooth_offset_stride),
            seed=int(params.smooth_offset_seed),
            device=device,
        )
    return torch.stack([src_x, src_y], dim=-1)


def augmentation_padding(config: FiberStripAugmentConfig, patch_shape_hw: tuple[int, int]) -> AugmentPadding:
    height, width = (int(v) for v in patch_shape_hw)
    radius = 0.5 * math.hypot(float(height), float(width))
    rotation_extra = radius if abs(float(config.rotation_degrees)) > 0.0 else 0.0
    scale_extra = radius * max(0.0, (1.0 / max(float(config.scale_min), 1.0e-6)) - 1.0)
    smooth_extra = abs(float(config.smooth_offset))
    pad_x = math.ceil(
        abs(float(config.shift_x)) + abs(float(config.shear_x)) * height + rotation_extra + scale_extra + 2.0
    )
    pad_y = math.ceil(
        abs(float(config.shift_y))
        + abs(float(config.shear_y)) * width
        + rotation_extra
        + scale_extra
        + smooth_extra
        + 2.0
    )
    return AugmentPadding(y=max(0, int(pad_y)), x=max(0, int(pad_x)))


def value_only_params(params: FiberStripAugmentParams) -> FiberStripAugmentParams:
    return FiberStripAugmentParams(
        brightness=params.brightness,
        contrast=params.contrast,
        gamma=params.gamma,
        noise_std=params.noise_std,
        blur_sigma=params.blur_sigma,
        noise_seed=params.noise_seed,
    )


def _grid_sample_coords(pixel_coords_xy: torch.Tensor, height: int, width: int) -> torch.Tensor:
    x = pixel_coords_xy[..., 0]
    y = pixel_coords_xy[..., 1]
    if width > 1:
        x = x * (2.0 / float(width - 1)) - 1.0
    else:
        x = torch.zeros_like(x)
    if height > 1:
        y = y * (2.0 / float(height - 1)) - 1.0
    else:
        y = torch.zeros_like(y)
    return torch.stack([x, y], dim=-1).unsqueeze(0)


def _line_mask(height: int, width: int, *, device: torch.device) -> torch.Tensor:
    y = torch.arange(height, device=device, dtype=torch.float32).view(height, 1)
    center_y = (float(height) - 1.0) * 0.5
    return (torch.abs(y - center_y) <= 0.75).expand(height, width).to(torch.float32)


def _valid_range_and_center(image: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if bool(valid.any().item()):
        values = image[valid]
        lo = values.min()
        hi = values.max()
        value_range = torch.clamp(hi - lo, min=1.0e-6)
        center = (hi + lo) * 0.5
        return value_range, center
    return torch.tensor(1.0, dtype=image.dtype, device=image.device), torch.tensor(
        0.0, dtype=image.dtype, device=image.device
    )


def _gaussian_blur_2d(image: torch.Tensor, sigma: float) -> torch.Tensor:
    sigma = float(sigma)
    if sigma <= 0.0:
        return image
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, dtype=image.dtype, device=image.device)
    kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum().clamp_min(1.0e-12)
    kernel_y = kernel_1d.view(1, 1, -1, 1)
    kernel_x = kernel_1d.view(1, 1, 1, -1)
    image_4d = image.view(1, 1, int(image.shape[0]), int(image.shape[1]))
    pad_mode_y = "reflect" if radius < int(image.shape[0]) else "replicate"
    pad_mode_x = "reflect" if radius < int(image.shape[1]) else "replicate"
    padded = F.pad(image_4d, (0, 0, radius, radius), mode=pad_mode_y)
    blurred = F.conv2d(padded, kernel_y)
    padded = F.pad(blurred, (radius, radius, 0, 0), mode=pad_mode_x)
    return F.conv2d(padded, kernel_x)[0, 0]


def _apply_gamma(image: torch.Tensor, valid: torch.Tensor, gamma: float, value_range: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    gamma = float(gamma)
    if abs(gamma - 1.0) <= 1.0e-6:
        return image
    lo = center - 0.5 * value_range
    norm = torch.clamp((image - lo) / value_range.clamp_min(1.0e-6), 0.0, 1.0)
    corrected = torch.pow(norm, gamma) * value_range + lo
    return torch.where(valid, corrected, image)


def augmented_line_mask(
    output_shape_hw: tuple[int, int],
    source_shape_hw: tuple[int, int],
    params: FiberStripAugmentParams,
    *,
    device: torch.device,
) -> torch.Tensor:
    output_height, output_width = (int(v) for v in output_shape_hw)
    source_height, source_width = (int(v) for v in source_shape_hw)
    coords = source_coordinate_grid_for_output(
        output_height,
        output_width,
        source_height,
        source_width,
        params,
        device=device,
    )
    grid = _grid_sample_coords(coords, source_height, source_width)
    line = _line_mask(source_height, source_width, device=device)
    return torch.clamp(
        F.grid_sample(
            line.view(1, 1, source_height, source_width),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[0, 0],
        0.0,
        1.0,
    )


def apply_strip_augmentation(
    image: np.ndarray | torch.Tensor,
    valid_mask: np.ndarray | torch.Tensor,
    params: FiberStripAugmentParams,
    *,
    device: torch.device,
    return_line_mask: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image_t = torch.as_tensor(image, dtype=torch.float32, device=device)
    valid_t = torch.as_tensor(valid_mask, dtype=torch.float32, device=device)
    if image_t.ndim != 2 or valid_t.ndim != 2:
        raise ValueError("image and valid_mask must be 2D")
    height, width = int(image_t.shape[0]), int(image_t.shape[1])
    coords = source_coordinate_grid(height, width, params, device=device)
    grid = _grid_sample_coords(coords, height, width)

    sampled = F.grid_sample(
        image_t.view(1, 1, height, width),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[0, 0]
    valid = F.grid_sample(
        valid_t.view(1, 1, height, width),
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=True,
    )[0, 0] > 0.5

    value_range, center = _valid_range_and_center(sampled, valid)
    sampled = (sampled - center) * float(params.contrast) + center + float(params.brightness) * value_range
    sampled = _apply_gamma(sampled, valid, params.gamma, value_range, center)
    if params.noise_std > 0.0:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(params.noise_seed))
        sampled = sampled + torch.randn(sampled.shape, generator=generator, device=device) * float(params.noise_std) * value_range
    sampled = _gaussian_blur_2d(sampled, params.blur_sigma)
    sampled = torch.where(valid, sampled, torch.zeros_like(sampled))

    if not return_line_mask:
        return sampled, valid

    line = _line_mask(height, width, device=device)
    sampled_line = F.grid_sample(
        line.view(1, 1, height, width),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[0, 0]
    sampled_line = torch.clamp(sampled_line, 0.0, 1.0) * valid.to(torch.float32)
    return sampled, valid, sampled_line


def apply_value_augmentation(
    image: np.ndarray | torch.Tensor,
    valid_mask: np.ndarray | torch.Tensor,
    params: FiberStripAugmentParams,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    image_t = torch.as_tensor(image, dtype=torch.float32, device=device)
    valid = torch.as_tensor(valid_mask, dtype=torch.bool, device=device)
    if image_t.ndim != 2 or valid.ndim != 2:
        raise ValueError("image and valid_mask must be 2D")
    value_range, center = _valid_range_and_center(image_t, valid)
    out = (image_t - center) * float(params.contrast) + center + float(params.brightness) * value_range
    out = _apply_gamma(out, valid, params.gamma, value_range, center)
    if params.noise_std > 0.0:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(params.noise_seed))
        out = out + torch.randn(out.shape, generator=generator, device=device) * float(params.noise_std) * value_range
    out = _gaussian_blur_2d(out, params.blur_sigma)
    return torch.where(valid, out, torch.zeros_like(out)), valid


def overlay_line_rgb(image_u8: np.ndarray, line_mask: np.ndarray, *, opacity: float = 0.5) -> np.ndarray:
    gray = np.asarray(image_u8, dtype=np.uint8)
    rgb = np.repeat(gray[..., None], 3, axis=-1).astype(np.float32)
    alpha = np.clip(np.asarray(line_mask, dtype=np.float32), 0.0, 1.0)[..., None] * float(opacity)
    color = np.zeros_like(rgb)
    color[..., 0] = 255.0
    rgb = rgb * (1.0 - alpha) + color * alpha
    return np.clip(rgb, 0.0, 255.0).astype(np.uint8)
