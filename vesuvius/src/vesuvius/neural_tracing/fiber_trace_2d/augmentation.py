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


def transformed_centerline_coords(
    output_shape_hw: tuple[int, int],
    source_shape_hw: tuple[int, int],
    params: FiberStripAugmentParams,
    *,
    device: torch.device,
) -> np.ndarray:
    output_height, output_width = (int(v) for v in output_shape_hw)
    source_height, source_width = (int(v) for v in source_shape_hw)
    if float(params.smooth_offset) == 0.0:
        return _transformed_centerline_coords_affine(
            output_shape_hw,
            source_shape_hw,
            params,
            device=device,
        )
    source_coords = source_coordinate_grid_for_output(
        output_height,
        output_width,
        source_height,
        source_width,
        params,
        device=device,
    )
    source_center_y = (float(source_height) - 1.0) * 0.5
    target_x = torch.linspace(0.0, float(source_width - 1), max(source_width, output_width), device=device)
    flat = source_coords.reshape(-1, 2)
    out_pixels = _pixel_grid(output_height, output_width, device=device).reshape(-1, 2)
    targets = torch.stack([target_x, torch.full_like(target_x, source_center_y)], dim=1)
    coords = _nearest_output_pixels_for_source_points(flat, out_pixels, targets)
    if coords.numel() == 0:
        return np.zeros((0, 2), dtype=np.float32)
    coords = coords[
        (coords[:, 0] >= 0.0)
        & (coords[:, 0] <= float(output_width - 1))
        & (coords[:, 1] >= 0.0)
        & (coords[:, 1] <= float(output_height - 1))
    ]
    if coords.numel() == 0:
        return np.zeros((0, 2), dtype=np.float32)
    # Remove adjacent duplicates from nearest-pixel inversion while preserving order.
    rounded = torch.round(coords)
    keep = torch.ones((rounded.shape[0],), dtype=torch.bool, device=device)
    keep[1:] = torch.any(rounded[1:] != rounded[:-1], dim=1)
    return coords[keep].detach().cpu().numpy().astype(np.float32)


def _nearest_output_pixels_for_source_points(
    source_flat_xy: torch.Tensor,
    output_flat_xy: torch.Tensor,
    target_source_xy: torch.Tensor,
    *,
    max_distance_sq: float = 4.0,
    chunk_size: int = 256,
) -> torch.Tensor:
    kept: list[torch.Tensor] = []
    for start in range(0, int(target_source_xy.shape[0]), int(chunk_size)):
        target = target_source_xy[start : start + int(chunk_size)]
        delta = source_flat_xy[:, None, :] - target[None, :, :]
        dist2 = torch.sum(delta * delta, dim=2)
        best_dist, best_index = torch.min(dist2, dim=0)
        valid = best_dist <= float(max_distance_sq)
        if bool(valid.any().item()):
            kept.append(output_flat_xy[best_index[valid]])
    if not kept:
        return torch.zeros((0, 2), dtype=output_flat_xy.dtype, device=output_flat_xy.device)
    return torch.cat(kept, dim=0)


def _transformed_centerline_coords_affine(
    output_shape_hw: tuple[int, int],
    source_shape_hw: tuple[int, int],
    params: FiberStripAugmentParams,
    *,
    device: torch.device,
) -> np.ndarray:
    output_height, output_width = (int(v) for v in output_shape_hw)
    source_height, source_width = (int(v) for v in source_shape_hw)
    output_center_x = (float(output_width) - 1.0) * 0.5
    output_center_y = (float(output_height) - 1.0) * 0.5
    source_center_x = (float(source_width) - 1.0) * 0.5
    source_center_y = (float(source_height) - 1.0) * 0.5

    count = max(source_width, output_width)
    src_x = torch.linspace(0.0, float(source_width - 1), count, device=device, dtype=torch.float32)
    src_y = torch.full_like(src_x, source_center_y)
    u = src_x - source_center_x + float(params.shift_x)
    v = src_y - source_center_y + float(params.shift_y)

    angle = math.radians(float(params.rotation_degrees))
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    x_sheared = cos_a * u + sin_a * v
    y_sheared = -sin_a * u + cos_a * v

    shear_x = float(params.shear_x)
    shear_y = float(params.shear_y)
    det = 1.0 - shear_x * shear_y
    if abs(det) <= 1.0e-6:
        return np.zeros((0, 2), dtype=np.float32)
    x = (x_sheared - shear_x * y_sheared) / det
    y = (-shear_y * x_sheared + y_sheared) / det

    scale = max(float(params.scale), 1.0e-6)
    x = x * scale
    y = y * scale
    if params.flip_x:
        x = -x
    if params.flip_y:
        y = -y
    coords = torch.stack([x + output_center_x, y + output_center_y], dim=1)
    coords = coords[
        (coords[:, 0] >= 0.0)
        & (coords[:, 0] <= float(output_width - 1))
        & (coords[:, 1] >= 0.0)
        & (coords[:, 1] <= float(output_height - 1))
    ]
    if coords.numel() == 0:
        return np.zeros((0, 2), dtype=np.float32)
    rounded = torch.round(coords)
    keep = torch.ones((rounded.shape[0],), dtype=torch.bool, device=device)
    keep[1:] = torch.any(rounded[1:] != rounded[:-1], dim=1)
    return coords[keep].detach().cpu().numpy().astype(np.float32)


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


def overlay_line_coords_rgb(
    image_u8: np.ndarray,
    line_xy: np.ndarray,
    *,
    opacity: float = 0.5,
    thickness: int = 1,
) -> np.ndarray:
    from PIL import Image, ImageDraw

    gray = np.asarray(image_u8, dtype=np.uint8)
    if gray.ndim == 2:
        rgb = np.repeat(gray[..., None], 3, axis=-1)
    else:
        rgb = gray.copy()
    coords = np.asarray(line_xy, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 2 or coords.shape[0] == 0:
        return rgb
    pil = Image.fromarray(rgb, mode="RGB")
    overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, mode="RGBA")
    points = [(float(x), float(y)) for x, y in coords.tolist()]
    alpha = int(np.clip(float(opacity), 0.0, 1.0) * 255.0)
    if len(points) == 1:
        x, y = points[0]
        r = max(1, int(thickness))
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 0, 0, alpha))
    else:
        draw.line(points, fill=(255, 0, 0, alpha), width=max(1, int(thickness)), joint="curve")
    return np.asarray(Image.alpha_composite(pil.convert("RGBA"), overlay).convert("RGB"), dtype=np.uint8)
