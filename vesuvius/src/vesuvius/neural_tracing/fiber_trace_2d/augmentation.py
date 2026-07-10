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


def sample_xy_maps_bilinear(
    maps_xy: torch.Tensor,
    points_xy: torch.Tensor,
    *,
    valid_lengths: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    maps = torch.as_tensor(maps_xy, dtype=torch.float32)
    points = torch.as_tensor(points_xy, dtype=torch.float32, device=maps.device)
    if maps.ndim != 4 or maps.shape[-1] != 2:
        raise ValueError("maps_xy must have shape B,H,W,2")
    if points.ndim != 3 or points.shape[-1] != 2:
        raise ValueError("points_xy must have shape B,N,2")
    if int(points.shape[0]) != int(maps.shape[0]):
        raise ValueError("maps_xy and points_xy batch dimensions must match")

    batch, height, width = int(maps.shape[0]), int(maps.shape[1]), int(maps.shape[2])
    x = points[..., 0]
    y = points[..., 1]
    valid = (
        torch.isfinite(points).all(dim=-1)
        & (x >= 0.0)
        & (x <= float(width - 1))
        & (y >= 0.0)
        & (y <= float(height - 1))
    )
    if valid_lengths is not None:
        lengths = torch.as_tensor(valid_lengths, dtype=torch.long, device=maps.device)
        if lengths.shape != (batch,):
            raise ValueError("valid_lengths must have shape B")
        index = torch.arange(int(points.shape[1]), dtype=torch.long, device=maps.device).view(1, -1)
        valid = valid & (index < lengths.view(-1, 1))

    x0 = torch.floor(torch.clamp(x, 0.0, float(width - 1))).to(torch.long)
    y0 = torch.floor(torch.clamp(y, 0.0, float(height - 1))).to(torch.long)
    x1 = torch.clamp(x0 + 1, max=width - 1)
    y1 = torch.clamp(y0 + 1, max=height - 1)
    wx = torch.clamp(x - x0.to(dtype=torch.float32), 0.0, 1.0)
    wy = torch.clamp(y - y0.to(dtype=torch.float32), 0.0, 1.0)

    batch_index = torch.arange(batch, dtype=torch.long, device=maps.device).view(batch, 1)
    v00 = maps[batch_index, y0, x0]
    v10 = maps[batch_index, y0, x1]
    v01 = maps[batch_index, y1, x0]
    v11 = maps[batch_index, y1, x1]
    top = v00 * (1.0 - wx)[..., None] + v10 * wx[..., None]
    bottom = v01 * (1.0 - wx)[..., None] + v11 * wx[..., None]
    sampled = top * (1.0 - wy)[..., None] + bottom * wy[..., None]
    sampled = torch.where(valid[..., None], sampled, torch.full_like(sampled, float("nan")))
    return sampled.to(dtype=torch.float32), valid


def _sample_xy_map(map_xy: torch.Tensor, points_xy: torch.Tensor) -> torch.Tensor:
    points = torch.as_tensor(points_xy, dtype=torch.float32, device=map_xy.device)
    shape = points.shape
    sampled, _ = sample_xy_maps_bilinear(
        map_xy.unsqueeze(0),
        points.reshape(1, -1, 2),
    )
    sampled = sampled[0]
    return sampled.reshape(*shape[:-1], 2)


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


def _smooth_offset_controls(
    width: int,
    *,
    amplitude: float,
    stride: float,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    amplitude = float(amplitude)
    stride = max(float(stride), 1.0)
    control_count = max(2, int(math.ceil(max(int(width) - 1, 1) / stride)) + 3)
    if amplitude == 0.0:
        return torch.zeros((control_count,), dtype=torch.float32, device=device)
    idx = torch.arange(control_count, dtype=torch.float64, device=device)
    seed_f = float(int(seed) & 0x7FFFFFFF)
    # Stateless deterministic LCG-style hash in tensor form. This avoids a
    # torch.Generator allocation in hot augmentation paths.
    hashed = torch.remainder(idx * 1103515245.0 + seed_f * 12345.0 + 67890.0, 2147483647.0)
    hashed = torch.remainder(hashed * 1103515245.0 + 12345.0, 2147483647.0)
    controls = (hashed / 2147483647.0 * 2.0 - 1.0) * abs(amplitude)
    if amplitude < 0.0:
        controls = -controls
    return controls.to(dtype=torch.float32)


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
    stride = max(float(stride), 1.0)
    controls = _smooth_offset_controls(
        int(width),
        amplitude=amplitude,
        stride=stride,
        seed=int(seed),
        device=device,
    )
    x = torch.arange(int(width), dtype=torch.float32, device=device) / stride + 1.0
    offsets = _cubic_interp_1d(controls, x)
    return offsets.view(1, int(width)).expand(int(height), int(width))


def _smooth_offset_values(
    output_x: torch.Tensor,
    output_width: int,
    *,
    amplitude: float,
    stride: float,
    seed: int,
) -> torch.Tensor:
    amplitude = float(amplitude)
    stride = max(float(stride), 1.0)
    controls = _smooth_offset_controls(
        int(output_width),
        amplitude=amplitude,
        stride=stride,
        seed=int(seed),
        device=output_x.device,
    )
    positions = output_x.to(dtype=torch.float32) / stride + 1.0
    return _cubic_interp_1d(controls, positions)


@dataclass(frozen=True)
class StripAugmentTransform:
    output_shape_hw: tuple[int, int]
    source_shape_hw: tuple[int, int]
    params: FiberStripAugmentParams
    device: torch.device

    def __post_init__(self) -> None:
        source_width = int(self.source_shape_hw[1])
        output_width = int(self.output_shape_hw[1])
        source_height = int(self.source_shape_hw[0])
        output_height = int(self.output_shape_hw[0])
        source_center = ((float(source_width) - 1.0) * 0.5, (float(source_height) - 1.0) * 0.5)
        output_center = ((float(output_width) - 1.0) * 0.5, (float(output_height) - 1.0) * 0.5)
        angle = math.radians(float(self.params.rotation_degrees))
        shear_x = float(self.params.shear_x)
        shear_y = float(self.params.shear_y)
        object.__setattr__(self, "_source_center", source_center)
        object.__setattr__(self, "_output_center", output_center)
        object.__setattr__(self, "_cos_a", math.cos(angle))
        object.__setattr__(self, "_sin_a", math.sin(angle))
        object.__setattr__(self, "_scale", max(float(self.params.scale), 1.0e-6))
        object.__setattr__(self, "_shear_x", shear_x)
        object.__setattr__(self, "_shear_y", shear_y)
        object.__setattr__(self, "_shear_det", 1.0 - shear_x * shear_y)
        object.__setattr__(
            self,
            "_smooth_controls",
            _smooth_offset_controls(
                source_width,
                amplitude=float(self.params.smooth_offset),
                stride=float(self.params.smooth_offset_stride),
                seed=int(self.params.smooth_offset_seed),
                device=self.device,
            ),
        )
        output_grid = _pixel_grid(output_height, output_width, device=self.device)
        source_grid = _pixel_grid(source_height, source_width, device=self.device)
        object.__setattr__(self, "_backward_map_xy", self._output_to_source_formula(output_grid))
        object.__setattr__(self, "_forward_map_xy", self._source_to_output_formula(source_grid))

    @property
    def output_height(self) -> int:
        return int(self.output_shape_hw[0])

    @property
    def output_width(self) -> int:
        return int(self.output_shape_hw[1])

    @property
    def source_height(self) -> int:
        return int(self.source_shape_hw[0])

    @property
    def source_width(self) -> int:
        return int(self.source_shape_hw[1])

    @property
    def output_center_x(self) -> float:
        return float(self._output_center[0])

    @property
    def output_center_y(self) -> float:
        return float(self._output_center[1])

    @property
    def source_center_x(self) -> float:
        return float(self._source_center[0])

    @property
    def source_center_y(self) -> float:
        return float(self._source_center[1])

    def _smooth_offsets_at_source_x(self, source_x: torch.Tensor) -> torch.Tensor:
        if float(self.params.smooth_offset) == 0.0:
            return torch.zeros_like(source_x, dtype=torch.float32)
        positions = source_x.to(dtype=torch.float32) / max(float(self.params.smooth_offset_stride), 1.0) + 1.0
        return _cubic_interp_1d(self._smooth_controls, positions)

    @property
    def backward_map_xy(self) -> torch.Tensor:
        return self._backward_map_xy

    @property
    def forward_map_xy(self) -> torch.Tensor:
        return self._forward_map_xy

    def _output_to_source_formula(self, output_xy: torch.Tensor) -> torch.Tensor:
        points = torch.as_tensor(output_xy, dtype=torch.float32, device=self.device)
        x = points[..., 0] - self.output_center_x - float(self.params.shift_x)
        y = points[..., 1] - self.output_center_y - float(self.params.shift_y)

        if self.params.flip_x:
            x = -x
        if self.params.flip_y:
            y = -y
        x = x / self._scale
        y = y / self._scale

        x_sheared = x + self._shear_x * y
        y_sheared = y + self._shear_y * x

        src_x = self._cos_a * x_sheared - self._sin_a * y_sheared + self.source_center_x
        src_y = self._sin_a * x_sheared + self._cos_a * y_sheared + self.source_center_y
        if float(self.params.smooth_offset) != 0.0:
            src_y = src_y + self._smooth_offsets_at_source_x(src_x)
        return torch.stack([src_x, src_y], dim=-1).to(dtype=torch.float32)

    def _source_to_output_formula(self, source_xy: torch.Tensor) -> torch.Tensor:
        points = torch.as_tensor(source_xy, dtype=torch.float32, device=self.device)
        src_x = points[..., 0]
        src_y = points[..., 1]
        if float(self.params.smooth_offset) != 0.0:
            src_y = src_y - self._smooth_offsets_at_source_x(src_x)
        u = src_x - self.source_center_x
        v = src_y - self.source_center_y

        x_sheared = self._cos_a * u + self._sin_a * v
        y_sheared = -self._sin_a * u + self._cos_a * v

        shear_x = self._shear_x
        shear_y = self._shear_y
        det = self._shear_det
        if abs(det) <= 1.0e-6:
            return torch.full(points.shape, float("nan"), dtype=torch.float32, device=self.device)
        x = (x_sheared - shear_x * y_sheared) / det
        y = (-shear_y * x_sheared + y_sheared) / det

        x = x * self._scale
        y = y * self._scale
        if self.params.flip_x:
            x = -x
        if self.params.flip_y:
            y = -y
        return torch.stack(
            [x + self.output_center_x + float(self.params.shift_x), y + self.output_center_y + float(self.params.shift_y)],
            dim=-1,
        ).to(dtype=torch.float32)

    def output_to_source_points(self, output_xy: torch.Tensor) -> torch.Tensor:
        return _sample_xy_map(self.backward_map_xy, torch.as_tensor(output_xy, dtype=torch.float32, device=self.device))

    def output_to_source_grid(self) -> torch.Tensor:
        return self.backward_map_xy

    def source_to_output_points(self, source_xy: torch.Tensor) -> torch.Tensor:
        return _sample_xy_map(self.forward_map_xy, torch.as_tensor(source_xy, dtype=torch.float32, device=self.device))


def strip_augment_transform(
    output_shape_hw: tuple[int, int],
    source_shape_hw: tuple[int, int],
    params: FiberStripAugmentParams,
    *,
    device: torch.device,
) -> StripAugmentTransform:
    return StripAugmentTransform(
        output_shape_hw=(int(output_shape_hw[0]), int(output_shape_hw[1])),
        source_shape_hw=(int(source_shape_hw[0]), int(source_shape_hw[1])),
        params=params,
        device=device,
    )


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
    return strip_augment_transform(
        (int(output_height), int(output_width)),
        (int(source_height), int(source_width)),
        params,
        device=device,
    ).output_to_source_grid()


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


def _gaussian_blur_2d_batch(images: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
    if images.ndim != 3 or sigmas.ndim != 1:
        raise ValueError("images must be B,H,W and sigmas must be B")
    batch, height, width = int(images.shape[0]), int(images.shape[1]), int(images.shape[2])
    if batch == 0 or not bool((sigmas > 0.0).any().item()):
        return images
    max_sigma = float(sigmas.max().item())
    max_radius = max(1, int(math.ceil(3.0 * max_sigma)))
    x = torch.arange(-max_radius, max_radius + 1, dtype=images.dtype, device=images.device)
    sigma_safe = sigmas.clamp_min(1.0e-6).view(batch, 1)
    radii = torch.ceil(3.0 * sigmas.clamp_min(0.0)).view(batch, 1)
    offsets = x.abs().view(1, -1)
    active = sigmas.view(batch, 1) > 0.0
    support = offsets <= radii
    kernels = torch.exp(-0.5 * (x.view(1, -1) / sigma_safe) ** 2)
    kernels = torch.where(active & support, kernels, torch.zeros_like(kernels))
    center = max_radius
    if bool((~active).any().item()):
        kernels = kernels.clone()
        kernels[~active.squeeze(1)] = 0.0
        kernels[~active.squeeze(1), center] = 1.0
    kernels = kernels / kernels.sum(dim=1, keepdim=True).clamp_min(1.0e-12)

    grouped = images.unsqueeze(0)
    kernel_y = kernels.view(batch, 1, -1, 1)
    kernel_x = kernels.view(batch, 1, 1, -1)
    pad_mode_y = "reflect" if max_radius < height else "replicate"
    pad_mode_x = "reflect" if max_radius < width else "replicate"
    padded = F.pad(grouped, (0, 0, max_radius, max_radius), mode=pad_mode_y)
    blurred = F.conv2d(padded, kernel_y, groups=batch)
    padded = F.pad(blurred, (max_radius, max_radius, 0, 0), mode=pad_mode_x)
    return F.conv2d(padded, kernel_x, groups=batch)[0]


def _apply_gamma(image: torch.Tensor, valid: torch.Tensor, gamma: float, value_range: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    gamma = float(gamma)
    if abs(gamma - 1.0) <= 1.0e-6:
        return image
    lo = center - 0.5 * value_range
    norm = torch.clamp((image - lo) / value_range.clamp_min(1.0e-6), 0.0, 1.0)
    corrected = torch.pow(norm, gamma) * value_range + lo
    return torch.where(valid, corrected, image)


def transformed_centerline_coords_torch(
    output_shape_hw: tuple[int, int],
    source_shape_hw: tuple[int, int],
    params: FiberStripAugmentParams,
    *,
    device: torch.device,
) -> torch.Tensor:
    output_height, output_width = (int(v) for v in output_shape_hw)
    source_height, source_width = (int(v) for v in source_shape_hw)
    source_center_y = (float(source_height) - 1.0) * 0.5
    target_x = torch.linspace(0.0, float(source_width - 1), max(source_width, output_width), device=device)
    source_line = torch.stack([target_x, torch.full_like(target_x, source_center_y)], dim=1)
    coords = strip_augment_transform(
        (output_height, output_width),
        (source_height, source_width),
        params,
        device=device,
    ).source_to_output_points(source_line)
    coords = coords[
        torch.isfinite(coords).all(dim=1)
        &
        (coords[:, 0] >= 0.0)
        & (coords[:, 0] <= float(output_width - 1))
        & (coords[:, 1] >= 0.0)
        & (coords[:, 1] <= float(output_height - 1))
    ]
    if coords.numel() == 0:
        return torch.zeros((0, 2), dtype=torch.float32, device=device)
    # Remove adjacent duplicates from nearest-pixel inversion while preserving order.
    rounded = torch.round(coords)
    keep = torch.ones((rounded.shape[0],), dtype=torch.bool, device=device)
    keep[1:] = torch.any(rounded[1:] != rounded[:-1], dim=1)
    return coords[keep].to(dtype=torch.float32)


def transformed_centerline_coords(
    output_shape_hw: tuple[int, int],
    source_shape_hw: tuple[int, int],
    params: FiberStripAugmentParams,
    *,
    device: torch.device,
) -> np.ndarray:
    return (
        transformed_centerline_coords_torch(
            output_shape_hw,
            source_shape_hw,
            params,
            device=device,
        )
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )


def transformed_source_point_coords_torch(
    output_shape_hw: tuple[int, int],
    source_shape_hw: tuple[int, int],
    params: FiberStripAugmentParams,
    source_xy: tuple[float, float],
    *,
    device: torch.device,
) -> torch.Tensor:
    source_point = torch.tensor([[float(source_xy[0]), float(source_xy[1])]], dtype=torch.float32, device=device)
    return strip_augment_transform(
        output_shape_hw,
        source_shape_hw,
        params,
        device=device,
    ).source_to_output_points(source_point)[0]


def transformed_source_point_coords(
    output_shape_hw: tuple[int, int],
    source_shape_hw: tuple[int, int],
    params: FiberStripAugmentParams,
    source_xy: tuple[float, float],
    *,
    device: torch.device,
) -> np.ndarray:
    return (
        transformed_source_point_coords_torch(
            output_shape_hw,
            source_shape_hw,
            params,
            source_xy,
            device=device,
        )
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )


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


def _transformed_source_point_coords_affine(
    output_shape_hw: tuple[int, int],
    source_shape_hw: tuple[int, int],
    params: FiberStripAugmentParams,
    source_points_xy: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    output_height, output_width = (int(v) for v in output_shape_hw)
    source_height, source_width = (int(v) for v in source_shape_hw)
    output_center_x = (float(output_width) - 1.0) * 0.5
    output_center_y = (float(output_height) - 1.0) * 0.5
    source_center_x = (float(source_width) - 1.0) * 0.5
    source_center_y = (float(source_height) - 1.0) * 0.5

    points = torch.as_tensor(source_points_xy, dtype=torch.float32, device=device)
    src_x = points[:, 0]
    src_y = points[:, 1]
    u = src_x - source_center_x
    v = src_y - source_center_y

    angle = math.radians(float(params.rotation_degrees))
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    x_sheared = cos_a * u + sin_a * v
    y_sheared = -sin_a * u + cos_a * v

    shear_x = float(params.shear_x)
    shear_y = float(params.shear_y)
    det = 1.0 - shear_x * shear_y
    if abs(det) <= 1.0e-6:
        return torch.full((points.shape[0], 2), float("nan"), dtype=torch.float32, device=device)
    x = (x_sheared - shear_x * y_sheared) / det
    y = (-shear_y * x_sheared + y_sheared) / det

    scale = max(float(params.scale), 1.0e-6)
    x = x * scale
    y = y * scale
    if params.flip_x:
        x = -x
    if params.flip_y:
        y = -y
    coords = torch.stack(
        [x + output_center_x + float(params.shift_x), y + output_center_y + float(params.shift_y)],
        dim=1,
    )
    return coords.to(dtype=torch.float32)


def _transformed_centerline_coords_affine(
    output_shape_hw: tuple[int, int],
    source_shape_hw: tuple[int, int],
    params: FiberStripAugmentParams,
    *,
    device: torch.device,
) -> torch.Tensor:
    output_height, output_width = (int(v) for v in output_shape_hw)
    source_height, source_width = (int(v) for v in source_shape_hw)
    output_center_x = (float(output_width) - 1.0) * 0.5
    output_center_y = (float(output_height) - 1.0) * 0.5
    source_center_x = (float(source_width) - 1.0) * 0.5
    source_center_y = (float(source_height) - 1.0) * 0.5

    count = max(source_width, output_width)
    src_x = torch.linspace(0.0, float(source_width - 1), count, device=device, dtype=torch.float32)
    src_y = torch.full_like(src_x, source_center_y)
    u = src_x - source_center_x
    v = src_y - source_center_y

    angle = math.radians(float(params.rotation_degrees))
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    x_sheared = cos_a * u + sin_a * v
    y_sheared = -sin_a * u + cos_a * v

    shear_x = float(params.shear_x)
    shear_y = float(params.shear_y)
    det = 1.0 - shear_x * shear_y
    if abs(det) <= 1.0e-6:
        return torch.zeros((0, 2), dtype=torch.float32, device=device)
    x = (x_sheared - shear_x * y_sheared) / det
    y = (-shear_y * x_sheared + y_sheared) / det

    scale = max(float(params.scale), 1.0e-6)
    x = x * scale
    y = y * scale
    if params.flip_x:
        x = -x
    if params.flip_y:
        y = -y
    coords = torch.stack(
        [x + output_center_x + float(params.shift_x), y + output_center_y + float(params.shift_y)],
        dim=1,
    )
    coords = coords[
        (coords[:, 0] >= 0.0)
        & (coords[:, 0] <= float(output_width - 1))
        & (coords[:, 1] >= 0.0)
        & (coords[:, 1] <= float(output_height - 1))
    ]
    if coords.numel() == 0:
        return torch.zeros((0, 2), dtype=torch.float32, device=device)
    rounded = torch.round(coords)
    keep = torch.ones((rounded.shape[0],), dtype=torch.bool, device=device)
    keep[1:] = torch.any(rounded[1:] != rounded[:-1], dim=1)
    return coords[keep].to(dtype=torch.float32)


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


def apply_value_augmentation_batch(
    images: np.ndarray | torch.Tensor,
    valid_masks: np.ndarray | torch.Tensor,
    params: list[FiberStripAugmentParams] | tuple[FiberStripAugmentParams, ...],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    image_t = torch.as_tensor(images, dtype=torch.float32, device=device)
    valid = torch.as_tensor(valid_masks, dtype=torch.bool, device=device)
    if image_t.ndim != 3 or valid.ndim != 3:
        raise ValueError("images and valid_masks must have shape B,H,W")
    batch = int(image_t.shape[0])
    if len(params) != batch:
        raise ValueError("params length must match image batch")

    inf = torch.tensor(float("inf"), dtype=image_t.dtype, device=device)
    neg_inf = torch.tensor(float("-inf"), dtype=image_t.dtype, device=device)
    valid_any = valid.view(batch, -1).any(dim=1)
    lo = torch.where(valid, image_t, inf).view(batch, -1).amin(dim=1)
    hi = torch.where(valid, image_t, neg_inf).view(batch, -1).amax(dim=1)
    lo = torch.where(valid_any, lo, torch.zeros_like(lo))
    hi = torch.where(valid_any, hi, torch.zeros_like(hi))
    value_range = torch.clamp(hi - lo, min=1.0e-6)
    center = (hi + lo) * 0.5

    contrast = torch.tensor([float(param.contrast) for param in params], dtype=image_t.dtype, device=device)
    brightness = torch.tensor([float(param.brightness) for param in params], dtype=image_t.dtype, device=device)
    gamma = torch.tensor([float(param.gamma) for param in params], dtype=image_t.dtype, device=device)
    view = (batch, 1, 1)
    out = (image_t - center.view(view)) * contrast.view(view) + center.view(view) + brightness.view(view) * value_range.view(view)

    gamma_mask = torch.abs(gamma - 1.0) > 1.0e-6
    if bool(gamma_mask.any().item()):
        norm = torch.clamp((out - lo.view(view)) / value_range.view(view), 0.0, 1.0)
        corrected = torch.pow(norm, gamma.view(view)) * value_range.view(view) + lo.view(view)
        out = torch.where((valid & gamma_mask.view(view)), corrected, out)

    for index, param in enumerate(params):
        if float(param.noise_std) > 0.0:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(param.noise_seed))
            out[index] = out[index] + torch.randn(
                out[index].shape,
                generator=generator,
                device=device,
            ) * float(param.noise_std) * value_range[index]
    blur_sigmas = torch.tensor([float(param.blur_sigma) for param in params], dtype=image_t.dtype, device=device)
    out = _gaussian_blur_2d_batch(out, blur_sigmas)
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
