from __future__ import annotations

import glob
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from lasagna.omezarr_pyramid import _decode_normals as _lasagna_decode_normals

from vesuvius.neural_tracing.datasets.common import (
    begin_zarr_cache_trace,
    end_zarr_cache_trace,
    open_zarr,
)
from vesuvius.neural_tracing.fiber_trace_2d.fiber_json import Vc3dFiber, load_vc3d_fiber
from vesuvius.neural_tracing.fiber_trace_2d.strip_geometry import (
    FiberStripFrame,
    FiberStripLineWindow,
    build_side_strip_patch_grid_from_line_window,
    side_strip_line_window,
)
from vesuvius.neural_tracing.fiber_trace.dataset import (
    _load_lasagna_volume,
    _omezarr_level_shape,
    _open_manifest_channel,
    _validate_shape,
    _volume_shape_zyx,
)
from vesuvius.neural_tracing.fiber_trace_2d.augmentation import (
    FiberStripAugmentConfig,
    apply_value_augmentation,
    augmentation_padding,
    augment_config_from_mapping,
    random_combined_augmentation,
    resolve_torch_device,
    source_coordinate_grid_for_output,
    transformed_centerline_coords,
    value_only_params,
)
from vesuvius.neural_tracing.fiber_trace_2d.loader_support import ZarrChunkRequest
from vesuvius.neural_tracing.fiber_trace_2d.sampling import CoordinateSampler, make_coordinate_sampler


_REMOTE_PREFIXES = ("http://", "https://", "s3://")


@dataclass(frozen=True)
class FiberStrip2DConfig:
    datasets: tuple[dict[str, Any], ...]
    batch_size: int = 1
    patch_shape_hw: tuple[int, int] = (21, 21)
    strip_z_offset_count: int = 16
    strip_z_offset_step: float = 1.0
    seed: int = 1
    prefetch_workers: int = 16
    volume_cache_dir: str | None = None
    volume_cache_offline: bool = False
    volume_cache_retry_seconds: float = 0.0
    augment: FiberStripAugmentConfig = FiberStripAugmentConfig()
    config_dir: Path | None = None


@dataclass(frozen=True)
class FiberStripSample:
    record_index: int
    fiber_path: str
    control_point_index: int
    control_point_xyz: np.ndarray
    strip_z_offset: float
    coords_zyx: np.ndarray
    valid_mask: np.ndarray
    frame: FiberStripFrame
    line_xy: np.ndarray


@dataclass(frozen=True)
class FiberStrip2DBatch:
    images: np.ndarray
    coords_zyx: np.ndarray
    valid_mask: np.ndarray
    strip_z_offsets: np.ndarray
    control_point_indices: np.ndarray
    record_indices: np.ndarray
    fiber_paths: tuple[str, ...]
    samples: tuple[FiberStripSample, ...]
    cache_stats: Any | None = None


@dataclass(frozen=True)
class _Record:
    fiber: Vc3dFiber
    volume: Any
    volume_path: str
    volume_scale: int
    volume_spacing_base: float
    sampler: CoordinateSampler
    grad_mag: Any
    grad_mag_spacing_base: float
    nx: Any | None
    ny: Any | None
    nx_spacing_base: float | None
    ny_spacing_base: float | None
    dataset_config: dict[str, Any]

def _is_remote_path(path: str | Path) -> bool:
    return str(path).startswith(_REMOTE_PREFIXES)


def _resolve_path(path: str | Path, config_dir: Path | None) -> str:
    path_s = str(path)
    if _is_remote_path(path_s):
        return path_s
    expanded = Path(path_s).expanduser()
    if expanded.is_absolute():
        return str(expanded)
    root = config_dir if config_dir is not None else Path.cwd()
    return str((root / expanded).resolve())


def _as_hw(value: Any, *, key: str) -> tuple[int, int]:
    if isinstance(value, int):
        size = int(value)
        return size, size
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{key} must be an int or length-2 sequence")
    return int(value[0]), int(value[1])


def strip_z_offsets_from_count_step(count: int, step: float) -> tuple[float, ...]:
    count = int(count)
    step = float(step)
    if count <= 0:
        raise ValueError("strip_z_offset_count must be > 0")
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("strip_z_offset_step must be positive and finite")
    start = -(count // 2)
    if count % 2 == 0:
        start += 1
    return tuple((start + i) * step for i in range(count))


def load_config(path: str | Path) -> FiberStrip2DConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"{config_path} must contain a JSON object")
    datasets = raw.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("config must contain a non-empty 'datasets' list")
    if "strip_z_offsets" in raw:
        raise ValueError("strip_z_offsets was removed; use strip_z_offset_count and strip_z_offset_step")
    cache_dir = raw.get("volume_cache_dir")
    return FiberStrip2DConfig(
        datasets=tuple(dict(entry) for entry in datasets),
        batch_size=int(raw.get("batch_size", 1)),
        patch_shape_hw=_as_hw(raw.get("patch_shape_hw", [21, 21]), key="patch_shape_hw"),
        strip_z_offset_count=int(raw.get("strip_z_offset_count", 16)),
        strip_z_offset_step=float(raw.get("strip_z_offset_step", 1.0)),
        seed=int(raw.get("seed", 1)),
        prefetch_workers=min(16, max(1, int(raw.get("prefetch_workers", 16)))),
        volume_cache_dir=None if cache_dir is None else str(cache_dir),
        volume_cache_offline=bool(raw.get("volume_cache_offline", False)),
        volume_cache_retry_seconds=float(raw.get("volume_cache_retry_seconds", 0.0)),
        augment=augment_config_from_mapping(raw),
        config_dir=config_path.parent,
    )


def _stable_seed(*parts: Any) -> int:
    import hashlib

    digest = hashlib.blake2b(digest_size=16)
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return int.from_bytes(digest.digest(), "little", signed=False)


def _resolve_fiber_paths(dataset_config: dict[str, Any], config: FiberStrip2DConfig) -> list[Path]:
    raw_paths = dataset_config.get("fiber_paths", dataset_config.get("fiber_glob"))
    if raw_paths is None:
        raise ValueError("dataset entry must define 'fiber_paths' or 'fiber_glob'")
    if isinstance(raw_paths, (str, Path)):
        raw_items = [raw_paths]
    elif isinstance(raw_paths, list):
        raw_items = raw_paths
    else:
        raise ValueError("fiber_paths/fiber_glob must be a string or list")

    paths: list[Path] = []
    for raw in raw_items:
        resolved = _resolve_path(raw, config.config_dir)
        matches = sorted(glob.glob(resolved))
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(resolved))
    if not paths:
        raise ValueError("dataset entry did not resolve any fiber paths")
    return paths


def _control_points_in_base_shape(fiber: Vc3dFiber, base_shape_zyx: tuple[int, int, int]) -> bool:
    cps = np.asarray(fiber.control_points_zyx, dtype=np.float64)
    if cps.ndim != 2 or cps.shape[1] != 3 or cps.shape[0] == 0:
        return False
    shape = np.asarray(base_shape_zyx, dtype=np.float64)
    return bool(np.isfinite(cps).all() and np.all(cps >= 0.0) and np.all(cps <= (shape - 1.0)))


def _first_out_of_bounds_control_point(
    fiber: Vc3dFiber, base_shape_zyx: tuple[int, int, int]
) -> tuple[int, np.ndarray] | None:
    cps = np.asarray(fiber.control_points_zyx, dtype=np.float64)
    shape = np.asarray(base_shape_zyx, dtype=np.float64)
    valid = np.isfinite(cps).all(axis=1)
    valid &= np.all(cps >= 0.0, axis=1)
    valid &= np.all(cps <= (shape - 1.0), axis=1)
    bad = np.flatnonzero(~valid)
    if bad.size == 0:
        return None
    index = int(bad[0])
    return index, cps[index]


def _open_dataset_volume(dataset_config: dict[str, Any], config: FiberStrip2DConfig) -> Any:
    volume_path = dataset_config.get("base_volume_path", dataset_config.get("volume_path"))
    if volume_path is None:
        raise ValueError("dataset entry must define 'base_volume_path'")
    volume_path = _resolve_path(volume_path, config.config_dir)
    scale = int(dataset_config.get("base_volume_scale", dataset_config.get("volume_scale", 0)))
    common_config = {
        "volume_cache_dir": config.volume_cache_dir,
        "volume_cache_offline": config.volume_cache_offline,
        "volume_cache_retry_seconds": config.volume_cache_retry_seconds,
        "_config_dir": str(config.config_dir) if config.config_dir is not None else None,
    }
    return open_zarr(
        volume_path,
        scale=scale,
        auth_json_path=dataset_config.get("base_volume_auth_json"),
        config=common_config,
    )


def _open_manifest_channels(
    dataset_config: dict[str, Any], config: FiberStrip2DConfig, *, volume: Any, volume_path: str
) -> tuple[tuple[int, int, int], Any, float, Any, Any, float, float]:
    manifest_path = dataset_config.get("lasagna_manifest_path")
    if not manifest_path:
        raise ValueError("dataset entry missing lasagna_manifest_path")

    common_config = {
        "volume_cache_dir": config.volume_cache_dir,
        "volume_cache_offline": config.volume_cache_offline,
        "volume_cache_retry_seconds": config.volume_cache_retry_seconds,
        "_config_dir": str(config.config_dir) if config.config_dir is not None else None,
    }
    lasagna_volume = _load_lasagna_volume(manifest_path, common_config)
    if lasagna_volume.base_shape_zyx is None:
        raise ValueError("Lasagna manifest must provide base_shape_zyx")

    base0 = open_zarr(
        volume_path,
        scale=0,
        auth_json_path=dataset_config.get("base_volume_auth_json", dataset_config.get("volume_auth_json")),
        config=common_config,
    )
    _validate_shape(
        _volume_shape_zyx(base0, label="base volume level 0"),
        tuple(lasagna_volume.base_shape_zyx),
        label="base volume level 0",
        context=f"Lasagna manifest {manifest_path}",
    )
    scale = int(dataset_config.get("base_volume_scale", dataset_config.get("volume_scale", 0)))
    _validate_shape(
        _volume_shape_zyx(volume, label="base volume selected level"),
        _omezarr_level_shape(lasagna_volume.base_shape_zyx, scale),
        label="base volume selected level",
        context=f"base_shape_zyx={lasagna_volume.base_shape_zyx} base_volume_scale={scale}",
    )
    try:
        grad_mag, grad_mag_spacing_base = _open_manifest_channel(
            lasagna_volume,
            "grad_mag",
            auth_json_path=dataset_config.get("lasagna_auth_json"),
            config=common_config,
        )
        nx, nx_spacing_base = _open_manifest_channel(
            lasagna_volume,
            "nx",
            auth_json_path=dataset_config.get("lasagna_auth_json"),
            config=common_config,
        )
        ny, ny_spacing_base = _open_manifest_channel(
            lasagna_volume,
            "ny",
            auth_json_path=dataset_config.get("lasagna_auth_json"),
            config=common_config,
        )
    except KeyError as exc:
        raise ValueError("Lasagna manifest missing required grad_mag/nx/ny channels") from exc
    if abs(float(nx_spacing_base) - float(ny_spacing_base)) > 1e-6:
        raise ValueError(
            f"Lasagna nx and ny normal channel spacing mismatch: nx={nx_spacing_base} ny={ny_spacing_base}"
        )
    return (
        tuple(int(v) for v in lasagna_volume.base_shape_zyx),
        grad_mag,
        float(grad_mag_spacing_base),
        nx,
        ny,
        float(nx_spacing_base),
        float(ny_spacing_base),
    )


def _decode_normal_components(nx_raw: np.ndarray, ny_raw: np.ndarray) -> np.ndarray:
    nx, ny, nz = _lasagna_decode_normals(np.asarray(nx_raw), np.asarray(ny_raw))
    nx = np.asarray(nx, dtype=np.float64)
    ny = np.asarray(ny, dtype=np.float64)
    nz = np.asarray(nz, dtype=np.float64)
    normal = np.stack([nx, ny, nz], axis=-1)
    norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    return np.divide(normal, np.maximum(norm, 1.0e-12), out=np.zeros_like(normal), where=norm > 1.0e-12)


def _principal_tensor_axis(tensor: np.ndarray, hint: np.ndarray) -> np.ndarray:
    axis = np.asarray(hint, dtype=np.float64)
    axis_norm = float(np.linalg.norm(axis))
    if not np.isfinite(axis_norm) or axis_norm <= 1.0e-12:
        diag = np.diag(tensor)
        axis = np.zeros(3, dtype=np.float64)
        axis[int(np.argmax(diag))] = 1.0
    else:
        axis = axis / axis_norm
    for _ in range(16):
        next_axis = tensor @ axis
        norm = float(np.linalg.norm(next_axis))
        if not np.isfinite(norm) or norm <= 1.0e-12:
            break
        axis = next_axis / norm
    norm = float(np.linalg.norm(axis))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        return np.zeros(3, dtype=np.float64)
    axis = axis / norm
    hint_norm = float(np.linalg.norm(hint))
    if np.isfinite(hint_norm) and hint_norm > 1.0e-12:
        if float(np.dot(axis, hint / hint_norm)) < 0.0:
            axis *= -1.0
    elif axis[2] < 0.0:
        axis *= -1.0
    return axis


def _resample_coords_like_augmentation(
    coords_zyx: np.ndarray,
    valid_mask: np.ndarray,
    params: Any,
    *,
    output_shape_hw: tuple[int, int],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    coords_t = torch.as_tensor(coords_zyx, dtype=torch.float32, device=device)
    valid_t = torch.as_tensor(valid_mask, dtype=torch.float32, device=device)
    if coords_t.ndim != 3 or coords_t.shape[-1] != 3:
        raise ValueError("coords_zyx must have shape H,W,3")
    source_height, source_width = int(coords_t.shape[0]), int(coords_t.shape[1])
    height, width = (int(v) for v in output_shape_hw)
    pixel_coords = source_coordinate_grid_for_output(
        height,
        width,
        source_height,
        source_width,
        params,
        device=device,
    )
    x = pixel_coords[..., 0]
    y = pixel_coords[..., 1]
    if source_width > 1:
        x = x * (2.0 / float(source_width - 1)) - 1.0
    else:
        x = torch.zeros_like(x)
    if source_height > 1:
        y = y * (2.0 / float(source_height - 1)) - 1.0
    else:
        y = torch.zeros_like(y)
    grid = torch.stack([x, y], dim=-1).unsqueeze(0)
    sampled_coords = F.grid_sample(
        coords_t.permute(2, 0, 1).unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[0].permute(1, 2, 0)
    sampled_valid = F.grid_sample(
        valid_t.view(1, 1, source_height, source_width),
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=True,
    )[0, 0] > 0.5
    sampled_coords = torch.where(sampled_valid[..., None], sampled_coords, torch.zeros_like(sampled_coords))
    return sampled_coords.cpu().numpy().astype(np.float32), sampled_valid.cpu().numpy().astype(bool)


class _ProfileBlock:
    def __init__(self, profile: dict[str, float] | None, name: str, device: torch.device | None = None) -> None:
        self.profile = profile
        self.name = str(name)
        self.device = device
        self.start = 0.0

    def __enter__(self) -> None:
        if self.profile is None:
            return None
        _sync_if_needed(self.device)
        self.start = time.perf_counter()
        return None

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.profile is None:
            return None
        _sync_if_needed(self.device)
        self.profile[self.name] = self.profile.get(self.name, 0.0) + (time.perf_counter() - self.start) * 1000.0
        return None


def _sync_if_needed(device: torch.device | None) -> None:
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)


SamplerFactory = Callable[..., CoordinateSampler]


class FiberStrip2DLoader:
    def __init__(self, config: FiberStrip2DConfig, *, sampler_factory: SamplerFactory | None = None) -> None:
        if config.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        self.strip_z_offsets = strip_z_offsets_from_count_step(
            config.strip_z_offset_count, config.strip_z_offset_step
        )
        self.config = config
        self._sampler_factory = make_coordinate_sampler if sampler_factory is None else sampler_factory
        self.records = self._load_records()
        self._flat_sample_count = sum(record.fiber.control_points_xyz.shape[0] for record in self.records)
        if self._flat_sample_count <= 0:
            raise ValueError("no control points found in configured fibers")

    def _load_records(self) -> list[_Record]:
        records: list[_Record] = []
        for dataset_config in self.config.datasets:
            volume = _open_dataset_volume(dataset_config, self.config)
            volume_path = _resolve_path(
                dataset_config.get("base_volume_path", dataset_config.get("volume_path")),
                self.config.config_dir,
            )
            volume_scale = int(dataset_config.get("base_volume_scale", dataset_config.get("volume_scale", 0)))
            volume_spacing_base = float(1 << volume_scale)
            sampler = self._sampler_factory(
                volume_path=volume_path,
                array=volume,
                level=volume_scale,
                level_spacing_base=volume_spacing_base,
                cache_root=self.config.volume_cache_dir,
            )
            (
                base_shape_zyx,
                grad_mag,
                grad_mag_spacing_base,
                nx,
                ny,
                nx_spacing_base,
                ny_spacing_base,
            ) = _open_manifest_channels(
                dataset_config,
                self.config,
                volume=volume,
                volume_path=volume_path,
            )
            for fiber_path in _resolve_fiber_paths(dataset_config, self.config):
                fiber = load_vc3d_fiber(fiber_path)
                bad_control = _first_out_of_bounds_control_point(fiber, base_shape_zyx)
                if bad_control is not None:
                    bad_index, bad_zyx = bad_control
                    print(
                        "fiber_trace_2d: skipping fiber with out-of-volume control point "
                        f"fiber_path='{fiber_path}' control_point_index={bad_index} "
                        f"control_point_zyx=({bad_zyx[0]:.3f}, {bad_zyx[1]:.3f}, {bad_zyx[2]:.3f}) "
                        f"base_shape_zyx={base_shape_zyx}",
                        flush=True,
                    )
                    continue
                records.append(
                    _Record(
                        fiber=fiber,
                        volume=volume,
                        volume_path=volume_path,
                        volume_scale=volume_scale,
                        volume_spacing_base=volume_spacing_base,
                        sampler=sampler,
                        grad_mag=grad_mag,
                        grad_mag_spacing_base=grad_mag_spacing_base,
                        nx=nx,
                        ny=ny,
                        nx_spacing_base=nx_spacing_base,
                        ny_spacing_base=ny_spacing_base,
                        dataset_config=dict(dataset_config),
                    )
                )
        if not records:
            raise ValueError("no fiber records loaded")
        return records

    @property
    def sample_count(self) -> int:
        return self._flat_sample_count

    def _random_flat_index(self, sample_index: int) -> int:
        rng = np.random.default_rng(_stable_seed(self.config.seed, "cp", int(sample_index)))
        return int(rng.integers(0, self._flat_sample_count))

    def _locate_flat_index(self, flat_index: int) -> tuple[int, int]:
        remaining = int(flat_index)
        for record_index, record in enumerate(self.records):
            count = int(record.fiber.control_points_xyz.shape[0])
            if remaining < count:
                return record_index, remaining
            remaining -= count
        raise IndexError(flat_index)

    def descriptor_for_sample_index(self, sample_index: int) -> tuple[_Record, int, int]:
        flat = self._random_flat_index(sample_index)
        record_index, control_index = self._locate_flat_index(flat)
        return self.records[record_index], record_index, control_index

    def _interpolation_cube(
        self, array: Any, point_zyx_base: np.ndarray, *, array_spacing_base: float
    ) -> tuple[np.ndarray, np.ndarray] | None:
        pos = np.asarray(point_zyx_base, dtype=np.float64) / float(array_spacing_base)
        shape = np.asarray(array.shape, dtype=np.int64)
        if (
            not bool(np.isfinite(pos).all())
            or bool(np.any(pos < 0.0))
            or bool(np.any(pos > (shape - 1).astype(np.float64)))
        ):
            return None
        base = np.floor(pos).astype(np.int64)
        frac = pos - base.astype(np.float64)
        hi = np.minimum(base + 1, shape - 1)
        block = np.asarray(
            array[
                base[0] : hi[0] + 1,
                base[1] : hi[1] + 1,
                base[2] : hi[2] + 1,
            ]
        )
        if block.size == 0:
            return None
        cube = np.empty((2, 2, 2), dtype=np.float64)
        for dz in (0, 1):
            src_z = min(dz, block.shape[0] - 1)
            for dy in (0, 1):
                src_y = min(dy, block.shape[1] - 1)
                for dx in (0, 1):
                    src_x = min(dx, block.shape[2] - 1)
                    cube[dz, dy, dx] = float(block[src_z, src_y, src_x])
        return cube, frac

    def _trilinear_cube(self, cube: np.ndarray, frac: np.ndarray) -> float:
        fz, fy, fx = (float(v) for v in frac)
        c00 = cube[0, 0, 0] * (1.0 - fx) + cube[0, 0, 1] * fx
        c01 = cube[0, 1, 0] * (1.0 - fx) + cube[0, 1, 1] * fx
        c10 = cube[1, 0, 0] * (1.0 - fx) + cube[1, 0, 1] * fx
        c11 = cube[1, 1, 0] * (1.0 - fx) + cube[1, 1, 1] * fx
        c0 = c00 * (1.0 - fy) + c01 * fy
        c1 = c10 * (1.0 - fy) + c11 * fy
        return float(c0 * (1.0 - fz) + c1 * fz)

    def _normal_sample_context(
        self,
        record: _Record,
        point_zyx_base: np.ndarray,
        *,
        channel: str,
        spacing_base: float,
        line_point_index: int | None = None,
        control_point_index: int | None = None,
    ) -> str:
        point = np.asarray(point_zyx_base, dtype=np.float64)
        channel_pos = point / float(spacing_base)
        array = getattr(record, channel)
        shape = tuple(int(v) for v in getattr(array, "shape", ()))
        in_bounds = (
            bool(np.isfinite(channel_pos).all())
            and len(shape) == 3
            and bool(np.all(channel_pos >= 0.0))
            and bool(np.all(channel_pos <= (np.asarray(shape, dtype=np.float64) - 1.0)))
        )
        fiber_path = str(record.fiber.path) if record.fiber.path is not None else ""
        return (
            f"channel={channel} spacing_base={float(spacing_base):.6g} "
            f"shape_zyx={shape} point_zyx_base=({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}) "
            f"point_xyz_base=({point[2]:.3f}, {point[1]:.3f}, {point[0]:.3f}) "
            f"channel_pos_zyx=({channel_pos[0]:.3f}, {channel_pos[1]:.3f}, {channel_pos[2]:.3f}) "
            f"in_bounds={in_bounds} base_volume_scale={record.volume_scale} "
            f"volume_spacing_base={record.volume_spacing_base:.6g} "
            f"line_point_index={line_point_index} control_point_index={control_point_index} "
            f"fiber_path='{fiber_path}' volume_path='{record.volume_path}'"
        )

    def _lasagna_normal_for_control_point(self, record: _Record, cp_index: int) -> np.ndarray:
        cp_xyz = np.asarray(record.fiber.control_points_xyz[cp_index], dtype=np.float64)
        cp_zyx = cp_xyz[[2, 1, 0]]
        return self._lasagna_normal_at_zyx(record, cp_zyx, control_point_index=cp_index)

    def _lasagna_normal_at_zyx(
        self,
        record: _Record,
        point_zyx_base: np.ndarray,
        *,
        line_point_index: int | None = None,
        control_point_index: int | None = None,
    ) -> np.ndarray:
        grad_request = self._interpolation_cube(
            record.grad_mag,
            point_zyx_base,
            array_spacing_base=record.grad_mag_spacing_base,
        )
        if grad_request is None:
            raise ValueError(
                "missing Lasagna grad_mag sample at fiber line point: "
                + self._normal_sample_context(
                    record,
                    point_zyx_base,
                    channel="grad_mag",
                    spacing_base=record.grad_mag_spacing_base,
                    line_point_index=line_point_index,
                    control_point_index=control_point_index,
                )
            )
        grad_value = self._trilinear_cube(*grad_request)
        if grad_value <= 0.0:
            raise ValueError(
                f"Lasagna grad_mag sample is zero at fiber line point value={grad_value:.6g}: "
                + self._normal_sample_context(
                    record,
                    point_zyx_base,
                    channel="grad_mag",
                    spacing_base=record.grad_mag_spacing_base,
                    line_point_index=line_point_index,
                    control_point_index=control_point_index,
                )
            )

        assert record.nx_spacing_base is not None
        assert record.ny_spacing_base is not None
        nx_request = self._interpolation_cube(record.nx, point_zyx_base, array_spacing_base=record.nx_spacing_base)
        ny_request = self._interpolation_cube(record.ny, point_zyx_base, array_spacing_base=record.ny_spacing_base)
        if nx_request is None or ny_request is None:
            missing = "nx" if nx_request is None else "ny"
            spacing = record.nx_spacing_base if nx_request is None else record.ny_spacing_base
            raise ValueError(
                "missing Lasagna nx/ny sample at fiber line point: "
                + self._normal_sample_context(
                    record,
                    point_zyx_base,
                    channel=missing,
                    spacing_base=float(spacing),
                    line_point_index=line_point_index,
                    control_point_index=control_point_index,
                )
            )
        nx_cube, frac = nx_request
        ny_cube, _ = ny_request

        tensor = np.zeros((3, 3), dtype=np.float64)
        hint = np.zeros(3, dtype=np.float64)
        total_weight = 0.0
        fz, fy, fx = (float(v) for v in frac)
        for dz in (0, 1):
            wz = (1.0 - fz) if dz == 0 else fz
            for dy in (0, 1):
                wy = (1.0 - fy) if dy == 0 else fy
                for dx in (0, 1):
                    wx = (1.0 - fx) if dx == 0 else fx
                    weight = wx * wy * wz
                    if weight <= 0.0:
                        continue
                    normal = _decode_normal_components(
                        np.asarray(nx_cube[dz, dy, dx]),
                        np.asarray(ny_cube[dz, dy, dx]),
                    ).reshape(3)
                    norm = float(np.linalg.norm(normal))
                    if not np.isfinite(norm) or norm <= 1.0e-12:
                        continue
                    tensor += weight * np.outer(normal, normal)
                    hint += normal * weight
                    total_weight += weight
        if total_weight <= 1.0e-12:
            raise ValueError(
                "degenerate Lasagna normal sample at fiber line point: "
                + self._normal_sample_context(
                    record,
                    point_zyx_base,
                    channel="nx",
                    spacing_base=record.nx_spacing_base,
                    line_point_index=line_point_index,
                    control_point_index=control_point_index,
                )
            )
        normal = _principal_tensor_axis(tensor, hint)
        if float(np.linalg.norm(normal)) <= 1.0e-12:
            raise ValueError(
                "degenerate Lasagna normal sample at fiber line point after principal-axis solve: "
                + self._normal_sample_context(
                    record,
                    point_zyx_base,
                    channel="nx",
                    spacing_base=record.nx_spacing_base,
                    line_point_index=line_point_index,
                    control_point_index=control_point_index,
                )
            )
        return normal.astype(np.float32)

    def _line_window_for_patch(
        self,
        record: _Record,
        *,
        control_point_index: int,
        patch_shape_hw: tuple[int, int],
    ) -> FiberStripLineWindow:
        return side_strip_line_window(
            record.fiber,
            control_point_index=control_point_index,
            patch_shape_hw=patch_shape_hw,
            pixel_spacing_base=record.volume_spacing_base,
        )

    def _lasagna_normals_for_line_window(
        self,
        record: _Record,
        line_window: FiberStripLineWindow,
        *,
        control_point_index: int | None = None,
    ) -> np.ndarray:
        normals = []
        for local_index, point_xyz in enumerate(np.asarray(line_window.line_points_xyz, dtype=np.float64)):
            line_point_index = int(line_window.original_line_indices[local_index])
            normals.append(
                self._lasagna_normal_at_zyx(
                    record,
                    point_xyz[[2, 1, 0]],
                    line_point_index=line_point_index,
                    control_point_index=control_point_index,
                )
            )
        return np.stack(normals, axis=0).astype(np.float32)

    def _unaugmented_centerline_coords(self, shape_hw: tuple[int, int]) -> np.ndarray:
        height, width = (int(v) for v in shape_hw)
        x = np.arange(width, dtype=np.float32)
        y = np.full((width,), (float(height) - 1.0) * 0.5, dtype=np.float32)
        return np.stack([x, y], axis=1)

    def build_sample(
        self, sample_index: int
    ) -> tuple[list[FiberStripSample], np.ndarray, np.ndarray, np.ndarray]:
        record, record_index, cp_index = self.descriptor_for_sample_index(sample_index)
        patch_shape_hw = self.config.patch_shape_hw
        if self.config.augment.enabled:
            pad = augmentation_padding(self.config.augment, self.config.patch_shape_hw)
            patch_shape_hw = (
                int(self.config.patch_shape_hw[0]) + 2 * pad.y,
                int(self.config.patch_shape_hw[1]) + 2 * pad.x,
            )
        line_window = self._line_window_for_patch(
            record,
            control_point_index=cp_index,
            patch_shape_hw=patch_shape_hw,
        )
        sampled_normals = self._lasagna_normals_for_line_window(
            record,
            line_window,
            control_point_index=cp_index,
        )
        images: list[np.ndarray] = []
        coords: list[np.ndarray] = []
        valids: list[np.ndarray] = []
        samples: list[FiberStripSample] = []
        augment_device = resolve_torch_device(self.config.augment.device) if self.config.augment.enabled else None
        for offset_index, offset in enumerate(self.strip_z_offsets):
            params = (
                random_combined_augmentation(self.config.augment, sample_index, offset_index)
                if self.config.augment.enabled
                else None
            )
            grid = build_side_strip_patch_grid_from_line_window(
                line_window,
                patch_shape_hw=patch_shape_hw,
                strip_z_offset=float(offset),
                sampled_normals=sampled_normals,
                pixel_spacing_base=record.volume_spacing_base,
            )
            coords_zyx = grid.coords_zyx.astype(np.float32, copy=False)
            valid_mask = grid.valid_mask.astype(bool, copy=False)
            line_xy = self._unaugmented_centerline_coords(patch_shape_hw)
            if self.config.augment.enabled:
                assert augment_device is not None
                assert params is not None
                coords_zyx, valid_mask = _resample_coords_like_augmentation(
                    coords_zyx,
                    valid_mask,
                    params,
                    output_shape_hw=self.config.patch_shape_hw,
                    device=augment_device,
                )
                line_xy = transformed_centerline_coords(
                    self.config.patch_shape_hw,
                    patch_shape_hw,
                    params,
                    device=augment_device,
                )
            result = record.sampler.sample_coords(coords_zyx, valid_mask)
            image = result.image
            valid_mask = result.valid_mask
            if self.config.augment.enabled:
                assert augment_device is not None
                assert params is not None
                image_t, valid_t = apply_value_augmentation(
                    image,
                    valid_mask,
                    value_only_params(params),
                    device=augment_device,
                )
                image = image_t.cpu().numpy().astype(np.float32)
                valid_mask = valid_t.cpu().numpy().astype(bool)
            sample = FiberStripSample(
                record_index=record_index,
                fiber_path=str(record.fiber.path) if record.fiber.path is not None else "",
                control_point_index=cp_index,
                control_point_xyz=np.asarray(record.fiber.control_points_xyz[cp_index], dtype=np.float32),
                strip_z_offset=float(offset),
                coords_zyx=coords_zyx,
                valid_mask=valid_mask,
                frame=grid.frame,
                line_xy=line_xy,
            )
            images.append(image)
            coords.append(coords_zyx)
            valids.append(valid_mask)
            samples.append(sample)
        return (
            samples,
            np.stack(images, axis=0),
            np.stack(coords, axis=0),
            np.stack(valids, axis=0),
        )

    def build_center_strip_patch(self, sample_index: int) -> tuple[FiberStripSample, np.ndarray, np.ndarray]:
        record, record_index, cp_index = self.descriptor_for_sample_index(sample_index)
        center_offset = min(self.strip_z_offsets, key=lambda value: abs(float(value)))
        line_window = self._line_window_for_patch(
            record,
            control_point_index=cp_index,
            patch_shape_hw=self.config.patch_shape_hw,
        )
        sampled_normals = self._lasagna_normals_for_line_window(
            record,
            line_window,
            control_point_index=cp_index,
        )
        grid = build_side_strip_patch_grid_from_line_window(
            line_window,
            patch_shape_hw=self.config.patch_shape_hw,
            strip_z_offset=float(center_offset),
            sampled_normals=sampled_normals,
            pixel_spacing_base=record.volume_spacing_base,
        )
        result = record.sampler.sample_coords(grid.coords_zyx, grid.valid_mask)
        image = result.image
        sample = FiberStripSample(
            record_index=record_index,
            fiber_path=str(record.fiber.path) if record.fiber.path is not None else "",
            control_point_index=cp_index,
            control_point_xyz=np.asarray(record.fiber.control_points_xyz[cp_index], dtype=np.float32),
            strip_z_offset=float(center_offset),
            coords_zyx=grid.coords_zyx.astype(np.float32, copy=False),
            valid_mask=result.valid_mask.astype(bool, copy=False),
            frame=grid.frame,
            line_xy=self._unaugmented_centerline_coords(self.config.patch_shape_hw),
        )
        return sample, image.astype(np.float32, copy=False), result.valid_mask.astype(bool, copy=False)

    def build_augmented_center_strip_patch(
        self,
        sample_index: int,
        params: Any,
        *,
        device: torch.device,
        profile: dict[str, float] | None = None,
    ) -> tuple[FiberStripSample, np.ndarray, np.ndarray, np.ndarray]:
        with _ProfileBlock(profile, "descriptor"):
            record, record_index, cp_index = self.descriptor_for_sample_index(sample_index)
            center_offset = min(self.strip_z_offsets, key=lambda value: abs(float(value)))
            pad = augmentation_padding(self.config.augment, self.config.patch_shape_hw)
            source_shape_hw = (
                int(self.config.patch_shape_hw[0]) + 2 * pad.y,
                int(self.config.patch_shape_hw[1]) + 2 * pad.x,
            )
        with _ProfileBlock(profile, "line_window"):
            line_window = self._line_window_for_patch(
                record,
                control_point_index=cp_index,
                patch_shape_hw=source_shape_hw,
            )
        with _ProfileBlock(profile, "lasagna_normals"):
            sampled_normals = self._lasagna_normals_for_line_window(
                record,
                line_window,
                control_point_index=cp_index,
            )
        with _ProfileBlock(profile, "strip_coords"):
            grid = build_side_strip_patch_grid_from_line_window(
                line_window,
                patch_shape_hw=source_shape_hw,
                strip_z_offset=float(center_offset),
                sampled_normals=sampled_normals,
                pixel_spacing_base=record.volume_spacing_base,
            )
        with _ProfileBlock(profile, "coord_augmentation", device):
            coords_zyx, valid_mask = _resample_coords_like_augmentation(
                grid.coords_zyx.astype(np.float32, copy=False),
                grid.valid_mask.astype(bool, copy=False),
                params,
                output_shape_hw=self.config.patch_shape_hw,
                device=device,
            )
        with _ProfileBlock(profile, "volume_sample"):
            result = record.sampler.sample_coords(coords_zyx, valid_mask)
            image = result.image
            valid_mask = result.valid_mask
            if profile is not None:
                for key, value in result.stats.items():
                    if isinstance(value, (int, float)):
                        profile[f"volume_stat_{key}"] = profile.get(f"volume_stat_{key}", 0.0) + float(value)
        with _ProfileBlock(profile, "value_augmentation", device):
            image_t, valid_t = apply_value_augmentation(
                image,
                valid_mask,
                value_only_params(params),
                device=device,
            )
        with _ProfileBlock(profile, "line_coords", device):
            line_xy = transformed_centerline_coords(
                self.config.patch_shape_hw,
                source_shape_hw,
                params,
                device=device,
            )
        sample = FiberStripSample(
            record_index=record_index,
            fiber_path=str(record.fiber.path) if record.fiber.path is not None else "",
            control_point_index=cp_index,
            control_point_xyz=np.asarray(record.fiber.control_points_xyz[cp_index], dtype=np.float32),
            strip_z_offset=float(center_offset),
            coords_zyx=coords_zyx,
            valid_mask=valid_t.cpu().numpy().astype(bool),
            frame=grid.frame,
            line_xy=line_xy,
        )
        return (
            sample,
            image_t.cpu().numpy().astype(np.float32),
            valid_t.cpu().numpy().astype(bool),
            line_xy,
        )

    def load_batch(self, start_sample_index: int = 0, batch_size: int | None = None) -> FiberStrip2DBatch:
        batch_size = self.config.batch_size if batch_size is None else int(batch_size)
        begin_zarr_cache_trace()
        all_samples: list[FiberStripSample] = []
        images: list[np.ndarray] = []
        coords: list[np.ndarray] = []
        valids: list[np.ndarray] = []
        record_indices: list[int] = []
        cp_indices: list[int] = []
        fiber_paths: list[str] = []
        try:
            for batch_pos in range(batch_size):
                sample_index = int(start_sample_index) + batch_pos
                (
                    sample_records,
                    sample_images,
                    sample_coords,
                    sample_valids,
                ) = self.build_sample(sample_index)
                first = sample_records[0]
                all_samples.extend(sample_records)
                images.append(sample_images)
                coords.append(sample_coords)
                valids.append(sample_valids)
                record_indices.append(first.record_index)
                cp_indices.append(first.control_point_index)
                fiber_paths.append(first.fiber_path)
        finally:
            cache_stats = end_zarr_cache_trace()
        return FiberStrip2DBatch(
            images=np.stack(images, axis=0)[:, :, None, :, :],
            coords_zyx=np.stack(coords, axis=0),
            valid_mask=np.stack(valids, axis=0),
            strip_z_offsets=np.asarray(self.strip_z_offsets, dtype=np.float32),
            control_point_indices=np.asarray(cp_indices, dtype=np.int32),
            record_indices=np.asarray(record_indices, dtype=np.int32),
            fiber_paths=tuple(fiber_paths),
            samples=tuple(all_samples),
            cache_stats=cache_stats,
        )

    def chunk_requests_for_sample_index(self, sample_index: int) -> list[ZarrChunkRequest]:
        record, _, cp_index = self.descriptor_for_sample_index(sample_index)
        requests: list[ZarrChunkRequest] = []
        augment_device = resolve_torch_device(self.config.augment.device) if self.config.augment.enabled else None
        patch_shape_hw = self.config.patch_shape_hw
        if self.config.augment.enabled:
            pad = augmentation_padding(self.config.augment, self.config.patch_shape_hw)
            patch_shape_hw = (
                int(self.config.patch_shape_hw[0]) + 2 * pad.y,
                int(self.config.patch_shape_hw[1]) + 2 * pad.x,
            )
        line_window = self._line_window_for_patch(
            record,
            control_point_index=cp_index,
            patch_shape_hw=patch_shape_hw,
        )
        sampled_normals = self._lasagna_normals_for_line_window(
            record,
            line_window,
            control_point_index=cp_index,
        )
        for offset_index, offset in enumerate(self.strip_z_offsets):
            params = (
                random_combined_augmentation(self.config.augment, sample_index, offset_index)
                if self.config.augment.enabled
                else None
            )
            grid = build_side_strip_patch_grid_from_line_window(
                line_window,
                patch_shape_hw=patch_shape_hw,
                strip_z_offset=float(offset),
                sampled_normals=sampled_normals,
                pixel_spacing_base=record.volume_spacing_base,
            )
            coords_zyx = grid.coords_zyx.astype(np.float32, copy=False)
            valid_mask = grid.valid_mask.astype(bool, copy=False)
            if self.config.augment.enabled:
                assert augment_device is not None
                assert params is not None
                coords_zyx, valid_mask = _resample_coords_like_augmentation(
                    coords_zyx,
                    valid_mask,
                    params,
                    output_shape_hw=self.config.patch_shape_hw,
                    device=augment_device,
                )
            requests.extend(record.sampler.chunk_requests_for_coords(coords_zyx, valid_mask))
        return requests

    def chunk_requests_for_samples(self, start_sample_index: int, sample_count: int) -> list[ZarrChunkRequest]:
        unique: dict[tuple[str, str], ZarrChunkRequest] = {}
        for sample_index in range(int(start_sample_index), int(start_sample_index) + int(sample_count)):
            for request in self.chunk_requests_for_sample_index(sample_index):
                unique[(request.store_identity, request.key)] = request
        return list(unique.values())

    def prefetch(self, start_sample_index: int, sample_count: int, *, workers: int | None = None) -> dict[str, Any]:
        raw_requests = self.chunk_requests_for_samples(start_sample_index, sample_count)
        deduped: dict[tuple[str, str], ZarrChunkRequest] = {}
        for request in raw_requests:
            deduped[(request.store_identity, request.key)] = request
        requests = list(deduped.values())
        pending: list[ZarrChunkRequest] = []
        for request in requests:
            cache_dir = getattr(request.store, "_cache_dir", None)
            if cache_dir is None:
                pending.append(request)
                continue
            cached = os.path.join(cache_dir, request.key)
            marker = cached + getattr(request.store, "_NEGATIVE_MARKER_SUFFIX", ".__notfound__")
            if not os.path.isfile(cached) and not os.path.isfile(marker):
                pending.append(request)

        total = len(pending)
        workers = min(16, max(1, int(self.config.prefetch_workers if workers is None else workers)))
        downloaded = 0
        bytes_read = 0
        errors = 0
        start = time.perf_counter()
        print(
            f"prefetch chunks: generated={len(requests)} missing={total} "
            f"samples={int(sample_count)} workers={workers}",
            flush=True,
        )

        def fetch(request: ZarrChunkRequest) -> int:
            data = request.store[request.key]
            return len(data) if isinstance(data, (bytes, bytearray, memoryview)) else len(bytes(data))

        if total:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(fetch, request) for request in pending]
                for future in as_completed(futures):
                    downloaded += 1
                    try:
                        bytes_read += int(future.result())
                    except Exception:
                        errors += 1
                    elapsed = max(time.perf_counter() - start, 1.0e-9)
                    mib_s = (bytes_read / (1024.0 * 1024.0)) / elapsed
                    remaining = total - downloaded
                    eta = remaining / max(downloaded / elapsed, 1.0e-9)
                    bar_width = 24
                    filled = int(math.floor(bar_width * downloaded / max(total, 1)))
                    bar = "#" * filled + "-" * (bar_width - filled)
                    print(
                        f"prefetch [{bar}] {downloaded}/{total} chunks "
                        f"{bytes_read / (1024.0 * 1024.0):.1f} MiB "
                        f"{mib_s:.1f} MiB/s eta={eta:.1f}s errors={errors}",
                        flush=True,
                    )

        return {
            "generated": len(requests),
            "missing": total,
            "downloaded": downloaded,
            "bytes": bytes_read,
            "errors": errors,
            "workers": workers,
        }


def iter_batches(loader: FiberStrip2DLoader, *, start_sample_index: int = 0) -> Iterable[FiberStrip2DBatch]:
    sample_index = int(start_sample_index)
    while True:
        yield loader.load_batch(sample_index)
        sample_index += loader.config.batch_size
