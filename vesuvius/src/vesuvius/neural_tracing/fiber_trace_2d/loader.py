from __future__ import annotations

import glob
import heapq
import hashlib
import json
import math
import os
import threading
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
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
    FiberStripGridTorch,
    FiberStripLineWindow,
    build_side_strip_patch_grid_tensor_from_line_window,
    build_top_strip_patch_grid_tensor_from_line_window,
    control_point_line_index,
    side_strip_segment_line_window,
    side_strip_line_window,
    source_line_xy_from_line_window,
    source_point_xy_for_line_index,
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
    FiberStripAugmentParams,
    apply_value_augmentation,
    apply_value_augmentation_batch,
    augmentation_padding,
    augment_config_from_mapping,
    random_combined_augmentation,
    resolve_torch_device,
    sample_xy_maps_bilinear,
    source_coordinate_grid_for_output,
    strip_augment_transform,
    value_only_params,
)
from vesuvius.neural_tracing.fiber_trace_2d.loader_support import ZarrChunkRequest
from vesuvius.neural_tracing.fiber_trace_2d.sampling import CoordinateSampler, make_coordinate_sampler


_REMOTE_PREFIXES = ("http://", "https://", "s3://")
_STRIP_COORD_CACHE_VERSION = "fiber_strip_2d_source_v3"
_STRIP_COORD_CACHE_KEY_VERSION = "fiber_strip_2d_source_v2"
_SUPPORTED_STRIP_COORD_CACHE_VERSIONS = {
    _STRIP_COORD_CACHE_KEY_VERSION,
    _STRIP_COORD_CACHE_VERSION,
}


@dataclass(frozen=True)
class FiberStrip2DConfig:
    datasets: tuple[dict[str, Any], ...]
    batch_size: int = 1
    patch_shape_hw: tuple[int, int] = (21, 21)
    strip_z_offset_count: int = 16
    strip_z_offset_step: float = 1.0
    seed: int = 1
    prefetch_workers: int = 16
    prefetch_sampler_workers: int = 2
    loader_workers: int = max(1, os.cpu_count() or 1)
    volume_cache_dir: str | None = None
    volume_cache_memory_mib: float | None = None
    volume_io_threads: int | None = None
    volume_cache_offline: bool = False
    volume_cache_retry_seconds: float = 0.0
    strip_coord_cache_dir: str | None = None
    augment: FiberStripAugmentConfig = FiberStripAugmentConfig()
    config_dir: Path | None = None
    suppress_record_warnings: bool = False

    @property
    def volume_cache_memory_bytes(self) -> int | None:
        if self.volume_cache_memory_mib is None:
            return None
        return int(float(self.volume_cache_memory_mib) * 1024.0 * 1024.0)


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
    control_point_xy: np.ndarray


@dataclass(frozen=True)
class FiberStripSegmentSample:
    record_index: int
    fiber_path: str
    start_control_point_index: int
    target_control_point_index: int
    start_control_point_xyz: np.ndarray
    target_control_point_xyz: np.ndarray
    strip_z_offset: float
    coords_zyx: np.ndarray
    valid_mask: np.ndarray
    frame: FiberStripFrame
    line_xy: np.ndarray
    start_control_point_xy: np.ndarray
    target_control_point_xy: np.ndarray
    line_point_indices: np.ndarray
    line_normals_xyz: np.ndarray
    start_row_axis_xyz: np.ndarray
    target_row_axis_xyz: np.ndarray


@dataclass(frozen=True)
class FiberStripTtaPatch:
    sample: FiberStripSample | FiberStripSegmentSample
    image: np.ndarray
    valid_mask: np.ndarray
    source_xy_grid: np.ndarray
    reference_to_tta_xy_grid: np.ndarray
    base_corners_xy: np.ndarray


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
    augmentation_params: tuple[FiberStripAugmentParams | None, ...] = ()


@dataclass(frozen=True)
class _Record:
    fiber: Vc3dFiber
    volume: Any
    volume_path: str
    volume_scale: int
    volume_spacing_base: float
    fiber_identity: str
    sampler: CoordinateSampler
    grad_mag: Any
    grad_mag_spacing_base: float
    nx: Any | None
    ny: Any | None
    nx_spacing_base: float | None
    ny_spacing_base: float | None
    dataset_config: dict[str, Any]


@dataclass(frozen=True)
class _StripSource:
    record: _Record
    record_index: int
    control_point_index: int
    center_offset: float
    source_shape_hw: tuple[int, int]
    grid: FiberStripGridTorch
    source_line_xy: torch.Tensor
    source_control_point_xy: torch.Tensor


@dataclass(frozen=True)
class _Trace2CpSegmentSource:
    record: _Record
    record_index: int
    start_control_point_index: int
    target_control_point_index: int
    center_offset: float
    source_shape_hw: tuple[int, int]
    grid: FiberStripGridTorch
    line_window: FiberStripLineWindow
    anchor_column_px: float
    line_xy: np.ndarray
    start_control_point_xy: np.ndarray
    target_control_point_xy: np.ndarray
    line_point_indices: np.ndarray
    line_normals_xyz: np.ndarray
    start_row_axis_xyz: np.ndarray
    target_row_axis_xyz: np.ndarray


@dataclass(frozen=True)
class _PreparedStripSample:
    source: _StripSource
    params_by_offset: list[FiberStripAugmentParams | None]
    offset_grids: list[FiberStripGridTorch]
    coords_zyx: np.ndarray
    valid_mask: np.ndarray
    line_xy_by_offset: list[np.ndarray]
    control_point_xy_by_offset: list[np.ndarray]


@dataclass(frozen=True)
class _PreparedTopStripSample:
    source: _StripSource
    params: FiberStripAugmentParams | None
    grid: FiberStripGridTorch
    coords_zyx: np.ndarray
    valid_mask: np.ndarray
    line_xy: np.ndarray
    control_point_xy: np.ndarray


@dataclass
class _PrefetchCounters:
    samples_done: int = 0
    samples_skipped: int = 0
    patches_done: int = 0
    unique_chunks_seen: int = 0
    queued_for_download: int = 0
    download_done: int = 0
    cache_hits: int = 0
    known_missing: int = 0
    downloaded: int = 0
    newly_missing: int = 0
    download_errors: int = 0
    bytes_downloaded: int = 0
    valid_pixels: int = 0
    first_error: str = ""
    first_sample_skip: str = ""
    queued_download_futures: int = 0
    max_exclusive_sample_index: int = 0


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


def _path_match_key(path: str | Path, config_dir: Path | None = None) -> str:
    resolved = _resolve_path(path, config_dir)
    if _is_remote_path(resolved):
        return resolved
    return str(Path(resolved).expanduser().resolve())


def _as_hw(value: Any, *, key: str) -> tuple[int, int]:
    if isinstance(value, int):
        size = int(value)
        return size, size
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{key} must be an int or length-2 sequence")
    return int(value[0]), int(value[1])


def _trace_xyz_at_x_for_top_strip(trace_xyz: np.ndarray, target_x: float) -> np.ndarray | None:
    trace = np.asarray(trace_xyz, dtype=np.float32)
    if trace.ndim != 2 or trace.shape[1] != 3 or trace.shape[0] == 0:
        raise ValueError("trace_xyz must have shape N,3")
    x_target = float(target_x)
    for p0, p1 in zip(trace[:-1], trace[1:]):
        x0 = float(p0[0])
        x1 = float(p1[0])
        if (x0 - x_target) == 0.0:
            return p0.astype(np.float32, copy=True)
        if (x0 - x_target) * (x1 - x_target) <= 0.0 and x0 != x1:
            alpha = np.float32((x_target - x0) / (x1 - x0))
            return (p0 + alpha * (p1 - p0)).astype(np.float32, copy=False)
    if float(trace[-1, 0]) == x_target:
        return trace[-1].astype(np.float32, copy=True)
    return None


def _trace_columns_xyz_for_top_strip(trace_xyz: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    columns = np.full((int(width), 3), np.nan, dtype=np.float32)
    valid = np.zeros((int(width),), dtype=bool)
    for x in range(int(width)):
        point = _trace_xyz_at_x_for_top_strip(trace_xyz, float(x))
        if point is None or not bool(np.isfinite(point).all()):
            continue
        columns[x] = point
        valid[x] = True
    return columns, valid


def _unit_vector_or_zero(vector: np.ndarray) -> np.ndarray:
    vec = np.asarray(vector, dtype=np.float32).reshape(-1)
    if vec.shape != (3,) or not bool(np.isfinite(vec).all()):
        return np.zeros(3, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        return np.zeros(3, dtype=np.float32)
    return (vec / np.float32(norm)).astype(np.float32, copy=False)


def _sample_grid_points_hwc(
    values_hwc: torch.Tensor,
    valid_hw: torch.Tensor,
    points_xy: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.as_tensor(values_hwc, dtype=torch.float32)
    valid = torch.as_tensor(valid_hw, dtype=torch.bool, device=values.device)
    points = torch.as_tensor(points_xy, dtype=torch.float32, device=values.device)
    if values.ndim != 3:
        raise ValueError("values_hwc must have shape H,W,C")
    if valid.shape != values.shape[:2]:
        raise ValueError("valid_hw shape must match values_hwc")
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points_xy must have shape N,2")
    height, width, _channels = (int(v) for v in values.shape)
    x = points[:, 0]
    y = points[:, 1]
    in_bounds = (
        torch.isfinite(points).all(dim=1)
        & (x >= 0.0)
        & (x <= float(width - 1))
        & (y >= 0.0)
        & (y <= float(height - 1))
    )
    if width > 1:
        grid_x = x * (2.0 / float(width - 1)) - 1.0
    else:
        grid_x = torch.zeros_like(x)
    if height > 1:
        grid_y = y * (2.0 / float(height - 1)) - 1.0
    else:
        grid_y = torch.zeros_like(y)
    grid_x = torch.where(in_bounds, grid_x, torch.zeros_like(grid_x))
    grid_y = torch.where(in_bounds, grid_y, torch.zeros_like(grid_y))
    grid = torch.stack([grid_x, grid_y], dim=-1).view(1, int(points.shape[0]), 1, 2)
    sampled = F.grid_sample(
        values.permute(2, 0, 1).unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[0, :, :, 0].permute(1, 0)
    sampled_valid = (
        F.grid_sample(
            valid.to(dtype=torch.float32).view(1, 1, height, width),
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=True,
        )[0, 0, :, 0]
        > 0.5
    ) & in_bounds
    sampled = torch.where(sampled_valid[:, None], sampled, torch.zeros_like(sampled))
    return sampled, sampled_valid


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
    cache_memory_mib = raw.get("volume_cache_memory_mib")
    if cache_memory_mib is not None:
        cache_memory_mib = float(cache_memory_mib)
        if not math.isfinite(cache_memory_mib) or cache_memory_mib <= 0.0:
            raise ValueError("volume_cache_memory_mib must be positive and finite when provided")
    volume_io_threads = raw.get("volume_io_threads")
    if volume_io_threads is not None:
        volume_io_threads = int(volume_io_threads)
        if volume_io_threads <= 0:
            raise ValueError("volume_io_threads must be positive when provided")
    strip_coord_cache_dir = raw.get("strip_coord_cache_dir")
    return FiberStrip2DConfig(
        datasets=tuple(dict(entry) for entry in datasets),
        batch_size=int(raw.get("batch_size", 1)),
        patch_shape_hw=_as_hw(raw.get("patch_shape_hw", [21, 21]), key="patch_shape_hw"),
        strip_z_offset_count=int(raw.get("strip_z_offset_count", 16)),
        strip_z_offset_step=float(raw.get("strip_z_offset_step", 1.0)),
        seed=int(raw.get("seed", 1)),
        prefetch_workers=max(1, int(raw.get("prefetch_workers", 16))),
        prefetch_sampler_workers=max(1, int(raw.get("prefetch_sampler_workers", 2))),
        loader_workers=max(1, int(raw.get("loader_workers", max(1, os.cpu_count() or 1)))),
        volume_cache_dir=None if cache_dir is None else str(cache_dir),
        volume_cache_memory_mib=cache_memory_mib,
        volume_io_threads=volume_io_threads,
        volume_cache_offline=bool(raw.get("volume_cache_offline", False)),
        volume_cache_retry_seconds=float(raw.get("volume_cache_retry_seconds", 0.0)),
        strip_coord_cache_dir=(
            None
            if strip_coord_cache_dir is None
            else _resolve_path(strip_coord_cache_dir, config_path.parent)
        ),
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


def _stable_digest(*parts: Any, digest_size: int = 24) -> str:
    digest = hashlib.blake2b(digest_size=digest_size)
    for part in parts:
        if isinstance(part, bytes):
            payload = part
        else:
            payload = str(part).encode("utf-8")
        digest.update(payload)
        digest.update(b"\0")
    return digest.hexdigest()


def _round_json_array(value: np.ndarray, *, decimals: int = 6) -> str:
    return json.dumps(np.round(np.asarray(value, dtype=np.float64), decimals).tolist(), separators=(",", ":"))


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
        "volume_cache_memory_mib": config.volume_cache_memory_mib,
        "volume_io_threads": config.volume_io_threads,
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
        "volume_cache_memory_mib": config.volume_cache_memory_mib,
        "volume_io_threads": config.volume_io_threads,
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


def _as_numpy_float32(array: torch.Tensor) -> np.ndarray:
    return array.detach().cpu().numpy().astype(np.float32, copy=False)


def _as_numpy_bool(array: torch.Tensor) -> np.ndarray:
    return array.detach().cpu().numpy().astype(bool, copy=False)


def _resample_coord_tensors_like_augmentation(
    coords_zyx: torch.Tensor,
    valid_mask: torch.Tensor,
    params: Any,
    *,
    output_shape_hw: tuple[int, int],
    device: torch.device,
    transform: Any | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    coords_t = coords_zyx.to(device=device, dtype=torch.float32)
    valid_bool = valid_mask.to(device=device, dtype=torch.bool)
    valid_t = valid_bool.to(dtype=torch.float32)
    if coords_t.ndim != 3 or coords_t.shape[-1] != 3:
        raise ValueError("coords_zyx must have shape H,W,3")
    source_height, source_width = int(coords_t.shape[0]), int(coords_t.shape[1])
    height, width = (int(v) for v in output_shape_hw)
    if transform is None:
        pixel_coords = source_coordinate_grid_for_output(
            height,
            width,
            source_height,
            source_width,
            params,
            device=device,
        )
    else:
        pixel_coords = transform.output_to_source_grid()
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
    return sampled_coords, sampled_valid


def _resample_coord_tensor_batch_like_augmentation(
    coords_zyx: torch.Tensor,
    valid_mask: torch.Tensor,
    backward_maps_xy: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    coords_t = torch.as_tensor(coords_zyx, dtype=torch.float32, device=backward_maps_xy.device)
    valid_bool = torch.as_tensor(valid_mask, dtype=torch.bool, device=backward_maps_xy.device)
    maps = torch.as_tensor(backward_maps_xy, dtype=torch.float32, device=backward_maps_xy.device)
    if coords_t.ndim != 4 or coords_t.shape[-1] != 3:
        raise ValueError("coords_zyx must have shape B,H,W,3")
    if valid_bool.shape != coords_t.shape[:3]:
        raise ValueError("valid_mask must have shape B,H,W")
    if maps.ndim != 4 or maps.shape[-1] != 2 or maps.shape[0] != coords_t.shape[0]:
        raise ValueError("backward_maps_xy must have shape B,Hout,Wout,2")
    batch, source_height, source_width = int(coords_t.shape[0]), int(coords_t.shape[1]), int(coords_t.shape[2])
    x = maps[..., 0]
    y = maps[..., 1]
    if source_width > 1:
        x = x * (2.0 / float(source_width - 1)) - 1.0
    else:
        x = torch.zeros_like(x)
    if source_height > 1:
        y = y * (2.0 / float(source_height - 1)) - 1.0
    else:
        y = torch.zeros_like(y)
    grid = torch.stack([x, y], dim=-1)
    sampled_coords = F.grid_sample(
        coords_t.permute(0, 3, 1, 2),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).permute(0, 2, 3, 1)
    sampled_valid = (
        F.grid_sample(
            valid_bool.to(dtype=torch.float32).view(batch, 1, source_height, source_width),
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=True,
        )[:, 0]
        > 0.5
    )
    sampled_coords = torch.where(sampled_valid[..., None], sampled_coords, torch.zeros_like(sampled_coords))
    return sampled_coords, sampled_valid


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


def _merge_profile(dst: dict[str, float] | None, src: dict[str, float]) -> None:
    if dst is None:
        return
    for key, value in src.items():
        if isinstance(value, (int, float)):
            dst[key] = dst.get(key, 0.0) + float(value)


def _sync_if_needed(device: torch.device | None) -> None:
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)


def _existing_data_path(request: ZarrChunkRequest) -> Path | None:
    store_paths = getattr(request.store, "data_paths_for_key", None)
    if callable(store_paths):
        paths = tuple(Path(path) for path in store_paths(request.key))
    elif request.cache_path is not None:
        cache_path = Path(request.cache_path)
        paths = (cache_path,) if cache_path.suffix else tuple(cache_path.parent.glob(f"{cache_path.name}.*"))
    else:
        paths = ()
    for path in paths:
        if path.name.endswith(".empty") or path.name.endswith(".tmp"):
            continue
        if path.is_file():
            return path
    return None


def _empty_marker_path(request: ZarrChunkRequest) -> Path | None:
    if request.empty_path is not None:
        return Path(request.empty_path)
    store_empty = getattr(request.store, "empty_path_for_key", None)
    if callable(store_empty):
        value = store_empty(request.key)
        return None if value is None else Path(value)
    return None


def _write_empty_marker(request: ZarrChunkRequest) -> None:
    path = _empty_marker_path(request)
    if path is None:
        raise RuntimeError(f"missing .empty marker path for chunk {request.store_identity}:{request.key}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb"):
        pass


class _PrefetchDefinitiveMissing(Exception):
    pass


class _PrefetchPermanentError(Exception):
    pass


def _fetch_prefetch_payload(request: ZarrChunkRequest) -> bytes:
    if request.cache_payload_format != "source_bytes":
        raise _PrefetchPermanentError(
            "unsupported prefetch cache payload format "
            f"{request.cache_payload_format!r} for chunk {request.store_identity}:{request.key}; "
            "only uncompressed direct-source zarr chunks are supported for Python prefetch"
        )
    if request.downloader is not None:
        payload = request.downloader(request)
        if payload is None:
            raise _PrefetchDefinitiveMissing()
        return bytes(payload)
    if not request.remote_url:
        raise _PrefetchPermanentError(f"missing remote URL for chunk {request.store_identity}:{request.key}")
    try:
        with urllib.request.urlopen(request.remote_url, timeout=60.0) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        if exc.code in {404, 410}:
            raise _PrefetchDefinitiveMissing() from exc
        raise


def _write_prefetch_payload_atomic(request: ZarrChunkRequest, payload: bytes) -> int:
    if request.cache_path is None:
        raise _PrefetchPermanentError(f"missing cache path for chunk {request.store_identity}:{request.key}")
    path = Path(request.cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (
        f".{path.name}.tmp.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}"
    )
    try:
        with tmp.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise
    return len(payload)


def _download_prefetch_request(request: ZarrChunkRequest, *, retry_seconds: float) -> dict[str, Any]:
    if _existing_data_path(request) is not None:
        return {"status": "cache_hit", "bytes": 0}
    empty_path = _empty_marker_path(request)
    if empty_path is not None and empty_path.is_file():
        return {"status": "known_missing", "bytes": 0}

    deadline = time.monotonic() + max(0.0, float(retry_seconds))
    delay = 0.25
    last_error = ""
    while True:
        try:
            payload = _fetch_prefetch_payload(request)
            byte_count = _write_prefetch_payload_atomic(request, payload)
            if _existing_data_path(request) is None:
                raise RuntimeError(f"downloaded chunk did not produce cache file for {request.key}")
            return {"status": "downloaded", "bytes": int(byte_count)}
        except _PrefetchDefinitiveMissing:
            _write_empty_marker(request)
            return {"status": "new_missing", "bytes": 0}
        except _PrefetchPermanentError as exc:
            return {"status": "error", "bytes": 0, "error": str(exc)}
        except Exception as exc:  # pragma: no cover - exact network exceptions are backend-specific.
            last_error = str(exc)
            if time.monotonic() >= deadline:
                return {"status": "error", "bytes": 0, "error": last_error}
            time.sleep(delay)
            delay = min(delay * 1.5, 5.0)


def _format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)}m{int(sec):02d}s"
    hours, minutes = divmod(minutes, 60.0)
    return f"{int(hours)}h{int(minutes):02d}m"


def _sync_value_augmentation_params(
    geometric_params: FiberStripAugmentParams,
    value_params: FiberStripAugmentParams,
) -> FiberStripAugmentParams:
    return replace(
        geometric_params,
        brightness=float(value_params.brightness),
        contrast=float(value_params.contrast),
        gamma=float(value_params.gamma),
        noise_std=float(value_params.noise_std),
        blur_sigma=float(value_params.blur_sigma),
        noise_seed=int(value_params.noise_seed),
    )


SamplerFactory = Callable[..., CoordinateSampler]


class FiberStrip2DLoader:
    def __init__(
        self,
        config: FiberStrip2DConfig,
        *,
        sampler_factory: SamplerFactory | None = None,
        records: tuple[_Record, ...] | list[_Record] | None = None,
        sample_identity_keys: tuple[str, ...] | None = None,
        random_pass_cache: dict[int, np.ndarray] | None = None,
        random_pass_cache_lock: threading.Lock | None = None,
    ) -> None:
        if config.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        self.strip_z_offsets = strip_z_offsets_from_count_step(
            config.strip_z_offset_count, config.strip_z_offset_step
        )
        self.config = config
        self._sampler_factory = make_coordinate_sampler if sampler_factory is None else sampler_factory
        self.records = self._load_records() if records is None else list(records)
        self._flat_sample_count = sum(record.fiber.control_points_xyz.shape[0] for record in self.records)
        if self._flat_sample_count <= 0:
            raise ValueError("no control points found in configured fibers")
        flat_offsets: list[int] = []
        flat_total = 0
        for record in self.records:
            flat_offsets.append(flat_total)
            flat_total += int(record.fiber.control_points_xyz.shape[0])
        self._record_flat_offsets = tuple(flat_offsets)
        self._sample_identity_keys = (
            self._build_sample_identity_keys()
            if sample_identity_keys is None
            else tuple(sample_identity_keys)
        )
        if len(self._sample_identity_keys) != int(self._flat_sample_count):
            raise ValueError("sample_identity_keys count does not match loaded control points")
        self._random_pass_cache = {} if random_pass_cache is None else random_pass_cache
        self._random_pass_cache_lock = (
            threading.Lock() if random_pass_cache_lock is None else random_pass_cache_lock
        )
        self._fiber_group_pass_cache: dict[tuple[int, int, int], tuple[tuple[int, tuple[int, ...]], ...]] = {}
        self._fiber_group_pass_cache_lock = threading.Lock()
        self._load_batch_skipped_samples = 0
        self._loader_executor: ThreadPoolExecutor | None = None
        self._loader_executor_workers = 0
        self._loader_executor_lock = threading.Lock()

    def close(self) -> None:
        with self._loader_executor_lock:
            executor = self._loader_executor
            self._loader_executor = None
            self._loader_executor_workers = 0
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self) -> "FiberStrip2DLoader":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        del exc_type, exc, tb
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def clone(self) -> "FiberStrip2DLoader":
        clone_config = replace(self.config, suppress_record_warnings=True)
        return FiberStrip2DLoader(
            clone_config,
            sampler_factory=self._sampler_factory,
            records=self._clone_records_with_local_samplers(clone_config),
            sample_identity_keys=self._sample_identity_keys,
            random_pass_cache=self._random_pass_cache,
            random_pass_cache_lock=self._random_pass_cache_lock,
        )

    def _make_sampler_for_record(self, record: _Record, config: FiberStrip2DConfig) -> CoordinateSampler:
        return self._sampler_factory(
            volume_path=record.volume_path,
            array=record.volume,
            level=record.volume_scale,
            level_spacing_base=record.volume_spacing_base,
            cache_root=config.volume_cache_dir,
            cache_budget_bytes=config.volume_cache_memory_bytes,
            io_threads=config.volume_io_threads,
        )

    def _clone_records_with_local_samplers(self, config: FiberStrip2DConfig) -> list[_Record]:
        samplers: dict[tuple[str, int, float, int], CoordinateSampler] = {}
        records: list[_Record] = []
        for record in self.records:
            key = (
                record.volume_path,
                int(record.volume_scale),
                float(record.volume_spacing_base),
                id(record.volume),
            )
            sampler = samplers.get(key)
            if sampler is None:
                sampler = self._make_sampler_for_record(record, config)
                samplers[key] = sampler
            records.append(replace(record, sampler=sampler))
        return records

    def _get_loader_executor(self, worker_count: int) -> ThreadPoolExecutor:
        worker_count = max(1, int(worker_count))
        with self._loader_executor_lock:
            if (
                self._loader_executor is None
                or self._loader_executor_workers != worker_count
            ):
                old_executor = self._loader_executor
                self._loader_executor = ThreadPoolExecutor(max_workers=worker_count)
                self._loader_executor_workers = worker_count
            else:
                old_executor = None
            executor = self._loader_executor
        if old_executor is not None:
            old_executor.shutdown(wait=True, cancel_futures=True)
        return executor

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
                cache_budget_bytes=self.config.volume_cache_memory_bytes,
                io_threads=self.config.volume_io_threads,
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
                fiber_identity = _stable_digest(
                    "fiber",
                    str(fiber_path),
                    _round_json_array(np.asarray(fiber.line_points_xyz, dtype=np.float64)),
                )
                bad_control = _first_out_of_bounds_control_point(fiber, base_shape_zyx)
                if bad_control is not None:
                    bad_index, bad_zyx = bad_control
                    if not self.config.suppress_record_warnings:
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
                        fiber_identity=fiber_identity,
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

    def flat_sample_indices_for_fiber_json(self, fiber_json: str | Path) -> tuple[int, ...]:
        query_key = _path_match_key(fiber_json, self.config.config_dir)
        flat_offset = 0
        for record in self.records:
            record_path = "" if record.fiber.path is None else str(record.fiber.path)
            record_count = int(record.fiber.control_points_xyz.shape[0])
            if record_path and _path_match_key(record_path, None) == query_key:
                return tuple(flat_offset + index for index in range(record_count))
            flat_offset += record_count
        raise ValueError(
            "fiber_json is not present in configured datasets: "
            f"fiber_json='{_resolve_path(fiber_json, self.config.config_dir)}'"
        )

    def _build_sample_identity_keys(self) -> tuple[str, ...]:
        keys: list[str] = []
        for record_index, record in enumerate(self.records):
            line = np.asarray(record.fiber.line_points_xyz, dtype=np.float64)
            line_key = json.dumps(np.round(line, 6).tolist(), separators=(",", ":"))
            cps = np.asarray(record.fiber.control_points_xyz, dtype=np.float64)
            for control_index, cp_xyz in enumerate(cps):
                cp_key = json.dumps(np.round(cp_xyz, 6).tolist(), separators=(",", ":"))
                keys.append(
                    f"line={line_key}|cp={cp_key}|record={record_index}|control={control_index}"
                )
        if len(keys) != int(self._flat_sample_count):
            raise ValueError("internal sample identity count mismatch")
        return tuple(keys)

    def _ensure_random_pass_order(self, pass_index: int) -> np.ndarray:
        pass_index = int(pass_index)
        order = self._random_pass_cache.get(pass_index)
        if order is not None:
            return order
        sample_count = int(self._flat_sample_count)
        with self._random_pass_cache_lock:
            order = self._random_pass_cache.get(pass_index)
            if order is None:
                order_list = sorted(
                    range(sample_count),
                    key=lambda flat: (
                        _stable_seed(
                            self.config.seed,
                            "cp_permutation",
                            pass_index,
                            self._sample_identity_keys[flat],
                        ),
                        self._sample_identity_keys[flat],
                        flat,
                    ),
                )
                order = np.asarray(order_list, dtype=np.int64)
                self._random_pass_cache[pass_index] = order
        return order

    def _ensure_random_pass_orders_for_indices(self, sample_indices: Iterable[int]) -> None:
        if int(self._flat_sample_count) <= 0:
            return
        passes = {
            math.floor(int(sample_index) / int(self._flat_sample_count))
            for sample_index in sample_indices
        }
        for pass_index in sorted(passes):
            self._ensure_random_pass_order(pass_index)

    def _random_flat_index(self, sample_index: int) -> int:
        sample_count = int(self._flat_sample_count)
        position = int(sample_index)
        pass_index = math.floor(position / sample_count)
        offset = position % sample_count
        order = self._ensure_random_pass_order(pass_index)
        return int(order[offset])

    def _flat_sample_index(self, sample_index: int) -> int:
        return int(sample_index) % int(self._flat_sample_count)

    @staticmethod
    def _bounded_sample_index(sample_index: int, sample_index_limit: int | None) -> int:
        if sample_index_limit is None:
            return int(sample_index)
        limit = int(sample_index_limit)
        if limit <= 0:
            return int(sample_index)
        return int(sample_index) % limit

    def _locate_flat_index(self, flat_index: int) -> tuple[int, int]:
        remaining = int(flat_index)
        for record_index, record in enumerate(self.records):
            count = int(record.fiber.control_points_xyz.shape[0])
            if remaining < count:
                return record_index, remaining
            remaining -= count
        raise IndexError(flat_index)

    def _descriptor_for_flat_index(self, flat_index: int) -> tuple[_Record, int, int]:
        record_index, control_index = self._locate_flat_index(int(flat_index))
        return self.records[record_index], record_index, control_index

    def descriptor_for_sample_index(
        self, sample_index: int, *, sample_mode: str = "random"
    ) -> tuple[_Record, int, int]:
        if sample_mode == "random":
            flat = self._random_flat_index(sample_index)
        elif sample_mode == "flat":
            flat = self._flat_sample_index(sample_index)
        else:
            raise ValueError("sample_mode must be 'random' or 'flat'")
        return self._descriptor_for_flat_index(flat)

    def _effective_flat_sample_count(self, sample_index_limit: int | None) -> int:
        if sample_index_limit is None or int(sample_index_limit) <= 0:
            return int(self._flat_sample_count)
        return max(0, min(int(sample_index_limit), int(self._flat_sample_count)))

    def _fiber_group_entries_for_pass(
        self,
        pass_index: int,
        *,
        control_points_per_group: int,
        sample_index_limit: int | None,
    ) -> tuple[tuple[int, tuple[int, ...]], ...]:
        group_size = max(1, int(control_points_per_group))
        effective_count = self._effective_flat_sample_count(sample_index_limit)
        if effective_count <= 0:
            raise ValueError("fiber group sampling has no effective control points")
        key = (int(pass_index), group_size, effective_count)
        cached = self._fiber_group_pass_cache.get(key)
        if cached is not None:
            return cached
        with self._fiber_group_pass_cache_lock:
            cached = self._fiber_group_pass_cache.get(key)
            if cached is not None:
                return cached
            entries: list[tuple[int, tuple[int, ...]]] = []
            for record_index, record in enumerate(self.records):
                record_start = int(self._record_flat_offsets[record_index])
                record_count = int(record.fiber.control_points_xyz.shape[0])
                allowed = max(0, min(record_count, effective_count - record_start))
                if allowed <= 0:
                    continue
                cp_order = sorted(
                    range(allowed),
                    key=lambda control_index: (
                        _stable_seed(
                            self.config.seed,
                            "fiber_group_cp",
                            int(pass_index),
                            record.fiber_identity,
                            control_index,
                        ),
                        self._sample_identity_keys[record_start + control_index],
                        control_index,
                    ),
                )
                for start in range(0, len(cp_order), group_size):
                    group = list(cp_order[start : start + group_size])
                    fill = 0
                    while len(group) < group_size:
                        group.append(cp_order[fill % len(cp_order)])
                        fill += 1
                    entries.append((record_index, tuple(int(v) for v in group)))
            if not entries:
                raise ValueError("fiber group sampling could not build any same-fiber CP groups")
            entries = sorted(
                entries,
                key=lambda item: (
                    _stable_seed(
                        self.config.seed,
                        "fiber_group_order",
                        int(pass_index),
                        item[0],
                        ",".join(str(v) for v in item[1]),
                    ),
                    item[0],
                    item[1],
                ),
            )
            cached = tuple(entries)
            self._fiber_group_pass_cache[key] = cached
        return cached

    def fiber_group_flat_indices_for_group(
        self,
        group_index: int,
        *,
        control_points_per_group: int,
        batch_size: int,
        sample_index_limit: int | None = None,
    ) -> tuple[int, ...]:
        batch_size = max(1, int(batch_size))
        group_size = max(1, int(control_points_per_group))
        groups_per_batch = max(1, int(math.ceil(batch_size / float(group_size))))
        first_entries = self._fiber_group_entries_for_pass(
            0,
            control_points_per_group=group_size,
            sample_index_limit=sample_index_limit,
        )
        groups_per_pass = len(first_entries)
        flat: list[int] = []
        for batch_group_offset in range(groups_per_batch):
            current_group_index = int(group_index) + int(batch_group_offset)
            pass_index = math.floor(current_group_index / groups_per_pass)
            offset = current_group_index % groups_per_pass
            entries = (
                first_entries
                if pass_index == 0
                else self._fiber_group_entries_for_pass(
                    pass_index,
                    control_points_per_group=group_size,
                    sample_index_limit=sample_index_limit,
                )
            )
            record_index, control_indices = entries[offset]
            record_start = int(self._record_flat_offsets[record_index])
            flat.extend(record_start + int(control_index) for control_index in control_indices)
        return tuple(int(v) for v in flat[:batch_size])

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

    @staticmethod
    def _prealign_lasagna_normals_to_row_axis_reference(
        sampled_normals: np.ndarray,
        line_window: FiberStripLineWindow,
        *,
        row_axis_alignment_line_index: int | None,
        row_axis_alignment_xyz: np.ndarray | None,
    ) -> np.ndarray:
        if row_axis_alignment_line_index is None or row_axis_alignment_xyz is None:
            return sampled_normals
        normals = np.asarray(sampled_normals, dtype=np.float32)
        reference = np.asarray(row_axis_alignment_xyz, dtype=np.float64).reshape(-1)
        if reference.shape != (3,):
            return normals
        reference_norm = float(np.linalg.norm(reference))
        if not np.isfinite(reference_norm) or reference_norm <= 1.0e-12:
            return normals
        reference = reference / reference_norm
        indices = np.asarray(line_window.original_line_indices, dtype=np.int64)
        matches = np.flatnonzero(indices == int(row_axis_alignment_line_index))
        if matches.size == 0:
            return normals
        local = np.asarray(normals[int(matches[0])], dtype=np.float64)
        local_norm = float(np.linalg.norm(local))
        if not np.isfinite(local_norm) or local_norm <= 1.0e-12:
            return normals
        if float(np.dot(local / local_norm, reference)) < 0.0:
            return (-normals).astype(np.float32, copy=False)
        return normals

    @staticmethod
    def _row_axis_at_xy(grid: FiberStripGridTorch, point_xy: np.ndarray) -> np.ndarray:
        axes = _as_numpy_float32(grid.offset_axis_xyz)
        point = np.asarray(point_xy, dtype=np.float32)
        if point.shape != (2,):
            return np.zeros(3, dtype=np.float32)
        height, width = axes.shape[:2]
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        if x < 0 or y < 0 or x >= width or y >= height:
            return np.zeros(3, dtype=np.float32)
        axis = np.asarray(axes[y, x], dtype=np.float32)
        norm = float(np.linalg.norm(axis))
        if not np.isfinite(norm) or norm <= 1.0e-12:
            return np.zeros(3, dtype=np.float32)
        return (axis / np.float32(norm)).astype(np.float32, copy=False)

    @classmethod
    def _grid_row_axis_aligns_reference(
        cls,
        grid: FiberStripGridTorch,
        line_window: FiberStripLineWindow,
        *,
        line_index: int,
        reference_xyz: np.ndarray,
        patch_shape_hw: tuple[int, int],
        anchor_column_px: float,
        pixel_spacing_base: float,
    ) -> bool:
        reference = np.asarray(reference_xyz, dtype=np.float64).reshape(-1)
        if reference.shape != (3,):
            return True
        reference_norm = float(np.linalg.norm(reference))
        if not np.isfinite(reference_norm) or reference_norm <= 1.0e-12:
            return True
        try:
            point_xy = source_point_xy_for_line_index(
                line_window,
                original_line_index=int(line_index),
                patch_shape_hw=patch_shape_hw,
                anchor_column_px=anchor_column_px,
                pixel_spacing_base=pixel_spacing_base,
            )
        except (IndexError, ValueError):
            return True
        axis = cls._row_axis_at_xy(grid, point_xy)
        axis_norm = float(np.linalg.norm(axis))
        if not np.isfinite(axis_norm) or axis_norm <= 1.0e-12:
            return True
        return float(np.dot(axis / axis_norm, reference / reference_norm)) >= 0.0

    def _unaugmented_centerline_coords(self, shape_hw: tuple[int, int]) -> np.ndarray:
        height, width = (int(v) for v in shape_hw)
        x = np.arange(width, dtype=np.float32)
        y = np.full((width,), (float(height) - 1.0) * 0.5, dtype=np.float32)
        return np.stack([x, y], axis=1)

    def _unaugmented_control_point_xy(self, shape_hw: tuple[int, int]) -> np.ndarray:
        height, width = (int(v) for v in shape_hw)
        return np.asarray([(float(width) - 1.0) * 0.5, (float(height) - 1.0) * 0.5], dtype=np.float32)

    @staticmethod
    def _source_line_and_cp_tensors(shape_hw: tuple[int, int], *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = (int(v) for v in shape_hw)
        x = torch.arange(width, dtype=torch.float32, device=device)
        y = torch.full((width,), (float(height) - 1.0) * 0.5, dtype=torch.float32, device=device)
        return (
            torch.stack([x, y], dim=1),
            torch.tensor(
                [(float(width) - 1.0) * 0.5, (float(height) - 1.0) * 0.5],
                dtype=torch.float32,
                device=device,
            ),
        )

    def _source_shape_hw(self, *, use_augmentation_envelope: bool | None = None) -> tuple[int, int]:
        use_envelope = self.config.augment.enabled if use_augmentation_envelope is None else bool(use_augmentation_envelope)
        if not use_envelope:
            return self.config.patch_shape_hw
        pad = augmentation_padding(self.config.augment, self.config.patch_shape_hw)
        return (
            int(self.config.patch_shape_hw[0]) + 2 * pad.y,
            int(self.config.patch_shape_hw[1]) + 2 * pad.x,
        )

    def _strip_coord_cache_path(
        self,
        record: _Record,
        *,
        control_point_index: int,
        center_offset: float,
        view: str = "side",
    ) -> Path | None:
        if self.config.strip_coord_cache_dir is None:
            return None
        cp_xyz = np.asarray(record.fiber.control_points_xyz[control_point_index], dtype=np.float64)
        view_key = str(view)
        if view_key == "side":
            family_key = _stable_digest(
                _STRIP_COORD_CACHE_KEY_VERSION,
                record.volume_path,
                record.volume_scale,
                f"{record.volume_spacing_base:.12g}",
                f"{float(center_offset):.12g}",
                f"{float(self.config.strip_z_offset_step):.12g}",
                record.fiber_identity,
                control_point_index,
                _round_json_array(cp_xyz),
            )
        else:
            family_key = _stable_digest(
                _STRIP_COORD_CACHE_KEY_VERSION,
                f"view:{view_key}",
                record.volume_path,
                record.volume_scale,
                f"{record.volume_spacing_base:.12g}",
                f"{float(center_offset):.12g}",
                record.fiber_identity,
                control_point_index,
                _round_json_array(cp_xyz),
            )
        root = Path(self.config.strip_coord_cache_dir)
        return root / family_key[:2] / f"{family_key}.npz"

    @staticmethod
    def _crop_cached_source(
        grid: FiberStripGridTorch,
        source_line_xy: torch.Tensor,
        source_control_point_xy: torch.Tensor,
        cached_shape_hw: tuple[int, int],
        shape_hw: tuple[int, int],
    ) -> tuple[FiberStripGridTorch, torch.Tensor, torch.Tensor]:
        cached_h, cached_w = (int(v) for v in cached_shape_hw)
        height, width = (int(v) for v in shape_hw)
        if cached_h < height or cached_w < width:
            raise ValueError("cached source shape is smaller than requested source shape")
        y0 = (cached_h - height) // 2
        x0 = (cached_w - width) // 2
        y1 = y0 + height
        x1 = x0 + width

        def crop_tensor(value: torch.Tensor | None) -> torch.Tensor | None:
            if value is None:
                return None
            return value[y0:y1, x0:x1].contiguous()

        cropped = FiberStripGridTorch(
            coords_xyz=crop_tensor(grid.coords_xyz),
            coords_zyx=crop_tensor(grid.coords_zyx),
            valid_mask=crop_tensor(grid.valid_mask),
            frame=grid.frame,
            offset_axis_xyz=crop_tensor(grid.offset_axis_xyz),
            offset_axis_zyx=crop_tensor(grid.offset_axis_zyx),
            side_axis_xyz=crop_tensor(grid.side_axis_xyz),
            side_axis_zyx=crop_tensor(grid.side_axis_zyx),
        )
        line = source_line_xy[x0:x1].contiguous()
        shift = torch.tensor([float(x0), float(y0)], dtype=line.dtype, device=line.device)
        line = line - shift.view(1, 2)
        cp = source_control_point_xy - shift
        return cropped, line, cp

    def _load_strip_coord_cache(
        self,
        path: Path | None,
        *,
        source_shape_hw: tuple[int, int],
        device: torch.device,
        center_offset: float,
        require_offset_axis: bool = True,
    ) -> tuple[FiberStripGridTorch, torch.Tensor, torch.Tensor] | None:
        if path is None or not path.is_file():
            return None
        try:
            with np.load(path, allow_pickle=False) as data:
                if str(data["version"].item()) not in _SUPPORTED_STRIP_COORD_CACHE_VERSIONS:
                    return None
                cached_shape = tuple(int(v) for v in data["source_shape_hw"].tolist())
                if cached_shape[0] < int(source_shape_hw[0]) or cached_shape[1] < int(source_shape_hw[1]):
                    return None
                if abs(float(data["center_offset"].item()) - float(center_offset)) > 1.0e-6:
                    return None
                frame = FiberStripFrame(
                    tangent_xyz=np.asarray(data["frame_tangent_xyz"], dtype=np.float32),
                        side_xyz=np.asarray(data["frame_side_xyz"], dtype=np.float32),
                        mesh_normal_xyz=np.asarray(data["frame_mesh_normal_xyz"], dtype=np.float32),
                    )
                coords_zyx = torch.as_tensor(np.asarray(data["coords_zyx"], dtype=np.float32), device=device)
                offset_axis_zyx = (
                    torch.as_tensor(
                        np.asarray(data["offset_axis_zyx"], dtype=np.float32),
                        device=device,
                    )
                    if bool(require_offset_axis)
                    else None
                )
                side_axis_zyx_np = (
                    np.asarray(data["side_axis_zyx"], dtype=np.float32)
                    if "side_axis_zyx" in data
                    else None
                )
                side_axis_zyx = (
                    torch.as_tensor(side_axis_zyx_np, dtype=torch.float32, device=device)
                    if side_axis_zyx_np is not None
                    and side_axis_zyx_np.shape == tuple(coords_zyx.shape)
                    else None
                )
                grid = FiberStripGridTorch(
                    coords_xyz=coords_zyx[..., (2, 1, 0)].contiguous(),
                    coords_zyx=coords_zyx,
                    valid_mask=torch.as_tensor(np.asarray(data["valid_mask"], dtype=bool), device=device),
                    frame=frame,
                    offset_axis_xyz=(
                        None
                        if offset_axis_zyx is None
                        else offset_axis_zyx[..., (2, 1, 0)].contiguous()
                    ),
                    offset_axis_zyx=offset_axis_zyx,
                    side_axis_xyz=(
                        None
                        if side_axis_zyx is None
                        else side_axis_zyx[..., (2, 1, 0)].contiguous()
                    ),
                    side_axis_zyx=side_axis_zyx,
                )
                source_line_xy = torch.as_tensor(np.asarray(data["source_line_xy"], dtype=np.float32), device=device)
                source_control_point_xy = torch.as_tensor(
                    np.asarray(data["source_control_point_xy"], dtype=np.float32),
                    device=device,
                )
        except Exception:
            return None
        grid, source_line_xy, source_control_point_xy = self._crop_cached_source(
            grid,
            source_line_xy,
            source_control_point_xy,
            cached_shape,
            source_shape_hw,
        )
        return grid, source_line_xy, source_control_point_xy

    def _store_strip_coord_cache(
        self,
        path: Path | None,
        grid: FiberStripGridTorch,
        *,
        source_shape_hw: tuple[int, int],
        center_offset: float,
        source_line_xy: torch.Tensor,
        source_control_point_xy: torch.Tensor,
    ) -> None:
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.parent / f".{path.name}.tmp.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}"
        frame = grid.frame
        try:
            with tmp.open("wb") as handle:
                payload = {
                    "version": np.asarray(_STRIP_COORD_CACHE_VERSION),
                    "source_shape_hw": np.asarray(source_shape_hw, dtype=np.int64),
                    "center_offset": np.asarray(float(center_offset), dtype=np.float64),
                    "coords_zyx": _as_numpy_float32(grid.coords_zyx),
                    "valid_mask": _as_numpy_bool(grid.valid_mask),
                    "offset_axis_zyx": _as_numpy_float32(grid.offset_axis_zyx),
                    "source_line_xy": _as_numpy_float32(source_line_xy),
                    "source_control_point_xy": _as_numpy_float32(source_control_point_xy),
                    "frame_tangent_xyz": np.asarray(frame.tangent_xyz, dtype=np.float32),
                    "frame_side_xyz": np.asarray(frame.side_xyz, dtype=np.float32),
                    "frame_mesh_normal_xyz": np.asarray(frame.mesh_normal_xyz, dtype=np.float32),
                }
                if grid.side_axis_zyx is not None:
                    payload["side_axis_zyx"] = _as_numpy_float32(grid.side_axis_zyx)
                np.savez(handle, **payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        except Exception:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass
            raise

    def build_strip_source(
        self,
        sample_index: int,
        *,
        device: torch.device,
        sample_mode: str = "random",
        descriptor: tuple[_Record, int, int] | None = None,
        profile: dict[str, float] | None = None,
        use_augmentation_envelope: bool | None = None,
        debug_label: str | None = None,
    ) -> _StripSource:
        if debug_label is not None:
            print(f"{debug_label} descriptor start", flush=True)
        with _ProfileBlock(profile, "descriptor"):
            if descriptor is None:
                record, record_index, cp_index = self.descriptor_for_sample_index(
                    sample_index, sample_mode=sample_mode
                )
            else:
                record, record_index, cp_index = descriptor
            center_offset = min(self.strip_z_offsets, key=lambda value: abs(float(value)))
            source_shape_hw = self._source_shape_hw(use_augmentation_envelope=use_augmentation_envelope)
        if debug_label is not None:
            print(
                f"{debug_label} descriptor done rec={record_index} cp={cp_index} "
                f"source_hw={source_shape_hw}",
                flush=True,
            )
            print(f"{debug_label} line_window start", flush=True)
        cache_path = self._strip_coord_cache_path(
            record,
            control_point_index=cp_index,
            center_offset=float(center_offset),
        )
        with _ProfileBlock(profile, "strip_coord_cache"):
            cached_source = self._load_strip_coord_cache(
                cache_path,
                source_shape_hw=source_shape_hw,
                device=device,
                center_offset=float(center_offset),
                require_offset_axis=not (
                    len(self.strip_z_offsets) == 1
                    and abs(float(self.strip_z_offsets[0]) - float(center_offset)) <= 1.0e-6
                ),
            )
        if cached_source is not None:
            cached_grid, source_line_xy, source_control_point_xy = cached_source
            if debug_label is not None:
                print(
                    f"{debug_label} strip_coord_cache hit valid={int(torch.count_nonzero(cached_grid.valid_mask).item())}",
                    flush=True,
                )
            return _StripSource(
                record=record,
                record_index=record_index,
                control_point_index=cp_index,
                center_offset=float(center_offset),
                source_shape_hw=source_shape_hw,
                grid=cached_grid,
                source_line_xy=source_line_xy,
                source_control_point_xy=source_control_point_xy,
            )
        with _ProfileBlock(profile, "line_window"):
            line_window = self._line_window_for_patch(
                record,
                control_point_index=cp_index,
                patch_shape_hw=source_shape_hw,
            )
        if debug_label is not None:
            print(
                f"{debug_label} line_window done points={len(line_window.line_points_xyz)} "
                f"cp_local={line_window.local_control_point_index}",
                flush=True,
            )
            print(f"{debug_label} lasagna_normals start", flush=True)
        with _ProfileBlock(profile, "lasagna_normals"):
            sampled_normals = self._lasagna_normals_for_line_window(
                record,
                line_window,
                control_point_index=cp_index,
            )
        if debug_label is not None:
            print(f"{debug_label} lasagna_normals done count={len(sampled_normals)}", flush=True)
            print(f"{debug_label} strip_coords start device={device}", flush=True)
        with _ProfileBlock(profile, "strip_coords", device):
            grid = build_side_strip_patch_grid_tensor_from_line_window(
                line_window,
                patch_shape_hw=source_shape_hw,
                strip_z_offset=float(center_offset),
                sampled_normals=sampled_normals,
                pixel_spacing_base=record.volume_spacing_base,
                device=device,
            )
        source_line_xy, source_control_point_xy = self._source_line_and_cp_tensors(source_shape_hw, device=device)
        self._store_strip_coord_cache(
            cache_path,
            grid,
            source_shape_hw=source_shape_hw,
            center_offset=float(center_offset),
            source_line_xy=source_line_xy,
            source_control_point_xy=source_control_point_xy,
        )
        if debug_label is not None:
            print(
                f"{debug_label} strip_coords done valid={int(torch.count_nonzero(grid.valid_mask).item())}",
                flush=True,
            )
        return _StripSource(
            record=record,
            record_index=record_index,
            control_point_index=cp_index,
            center_offset=float(center_offset),
            source_shape_hw=source_shape_hw,
            grid=grid,
            source_line_xy=source_line_xy,
            source_control_point_xy=source_control_point_xy,
        )

    def build_top_strip_source(
        self,
        sample_index: int,
        *,
        device: torch.device,
        sample_mode: str = "random",
        descriptor: tuple[_Record, int, int] | None = None,
        profile: dict[str, float] | None = None,
        use_augmentation_envelope: bool | None = None,
    ) -> _StripSource:
        with _ProfileBlock(profile, "descriptor"):
            if descriptor is None:
                record, record_index, cp_index = self.descriptor_for_sample_index(
                    sample_index, sample_mode=sample_mode
                )
            else:
                record, record_index, cp_index = descriptor
            source_shape_hw = self._source_shape_hw(use_augmentation_envelope=use_augmentation_envelope)
        cache_path = self._strip_coord_cache_path(
            record,
            control_point_index=cp_index,
            center_offset=0.0,
            view="top",
        )
        with _ProfileBlock(profile, "strip_coord_cache"):
            cached_source = self._load_strip_coord_cache(
                cache_path,
                source_shape_hw=source_shape_hw,
                device=device,
                center_offset=0.0,
                require_offset_axis=True,
            )
        if cached_source is not None:
            cached_grid, source_line_xy, source_control_point_xy = cached_source
            return _StripSource(
                record=record,
                record_index=record_index,
                control_point_index=cp_index,
                center_offset=0.0,
                source_shape_hw=source_shape_hw,
                grid=cached_grid,
                source_line_xy=source_line_xy,
                source_control_point_xy=source_control_point_xy,
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
        with _ProfileBlock(profile, "strip_coords", device):
            grid = build_top_strip_patch_grid_tensor_from_line_window(
                line_window,
                patch_shape_hw=source_shape_hw,
                sampled_normals=sampled_normals,
                pixel_spacing_base=record.volume_spacing_base,
                device=device,
            )
        source_line_xy, source_control_point_xy = self._source_line_and_cp_tensors(source_shape_hw, device=device)
        self._store_strip_coord_cache(
            cache_path,
            grid,
            source_shape_hw=source_shape_hw,
            center_offset=0.0,
            source_line_xy=source_line_xy,
            source_control_point_xy=source_control_point_xy,
        )
        return _StripSource(
            record=record,
            record_index=record_index,
            control_point_index=cp_index,
            center_offset=0.0,
            source_shape_hw=source_shape_hw,
            grid=grid,
            source_line_xy=source_line_xy,
            source_control_point_xy=source_control_point_xy,
        )

    def _offset_grid_from_source(
        self,
        source: _StripSource | _Trace2CpSegmentSource,
        offset: float,
    ) -> FiberStripGridTorch:
        delta_base = (float(offset) - float(source.center_offset)) * float(source.record.volume_spacing_base)
        if abs(float(delta_base)) <= 1.0e-6:
            return source.grid
        axis_zyx = source.grid.offset_axis_zyx
        axis_xyz = source.grid.offset_axis_xyz
        if axis_zyx is None or axis_xyz is None:
            raise ValueError("strip source grid is missing offset-axis data")
        delta = torch.as_tensor(delta_base, dtype=source.grid.coords_zyx.dtype, device=source.grid.coords_zyx.device)
        coords_zyx = source.grid.coords_zyx + axis_zyx.to(device=source.grid.coords_zyx.device) * delta
        coords_xyz = source.grid.coords_xyz + axis_xyz.to(device=source.grid.coords_xyz.device) * delta
        return FiberStripGridTorch(
            coords_xyz=coords_xyz,
            coords_zyx=coords_zyx,
            valid_mask=source.grid.valid_mask,
            frame=source.grid.frame,
            offset_axis_xyz=axis_xyz,
            offset_axis_zyx=axis_zyx,
            side_axis_xyz=source.grid.side_axis_xyz,
            side_axis_zyx=source.grid.side_axis_zyx,
        )

    def _trace2cp_side_z_grid_from_source(
        self,
        source: _Trace2CpSegmentSource,
        side_z_offset_voxels: float,
    ) -> FiberStripGridTorch:
        delta_base = float(side_z_offset_voxels) * float(source.record.volume_spacing_base)
        side_axis_zyx = source.grid.side_axis_zyx
        side_axis_xyz = source.grid.side_axis_xyz
        if side_axis_zyx is None or side_axis_xyz is None:
            raise ValueError("Trace2CP side z-layer source is missing side-axis data")
        if abs(float(delta_base)) <= 1.0e-6:
            return source.grid
        delta = torch.as_tensor(
            delta_base,
            dtype=source.grid.coords_zyx.dtype,
            device=source.grid.coords_zyx.device,
        )
        coords_zyx = source.grid.coords_zyx + side_axis_zyx.to(device=source.grid.coords_zyx.device) * delta
        coords_xyz = source.grid.coords_xyz + side_axis_xyz.to(device=source.grid.coords_xyz.device) * delta
        valid_mask = source.grid.valid_mask & torch.isfinite(coords_zyx).all(dim=-1)
        return FiberStripGridTorch(
            coords_xyz=coords_xyz,
            coords_zyx=coords_zyx,
            valid_mask=valid_mask,
            frame=source.grid.frame,
            offset_axis_xyz=source.grid.offset_axis_xyz,
            offset_axis_zyx=source.grid.offset_axis_zyx,
            side_axis_xyz=side_axis_xyz,
            side_axis_zyx=side_axis_zyx,
        )

    def _line_and_cp_xy_for_params(
        self,
        source: _StripSource,
        params: FiberStripAugmentParams | None,
        *,
        device: torch.device,
        transform: Any | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if params is None:
            return (
                source.source_line_xy.to(device=device, dtype=torch.float32),
                source.source_control_point_xy.to(device=device, dtype=torch.float32),
            )
        if transform is None:
            transform = strip_augment_transform(
                self.config.patch_shape_hw,
                source.source_shape_hw,
                params,
                device=device,
            )
        source_line_xy = source.source_line_xy.to(device=device, dtype=torch.float32)
        source_control_point_xy = source.source_control_point_xy.to(device=device, dtype=torch.float32).view(1, 2)
        mapped = transform.source_to_output_points(torch.cat([source_line_xy, source_control_point_xy], dim=0))
        line_xy = mapped[:-1]
        control_point_xy = mapped[-1]
        if line_xy.numel() != 0:
            height, width = self.config.patch_shape_hw
            line_xy = line_xy[
                torch.isfinite(line_xy).all(dim=1)
                & (line_xy[:, 0] >= 0.0)
                & (line_xy[:, 0] <= float(int(width) - 1))
                & (line_xy[:, 1] >= 0.0)
                & (line_xy[:, 1] <= float(int(height) - 1))
            ]
        return line_xy, control_point_xy

    def build_strip_patch_from_source(
        self,
        source: _StripSource,
        offset_index: int,
        params: FiberStripAugmentParams | None,
        *,
        device: torch.device,
        profile: dict[str, float] | None = None,
        load_image: bool = True,
        apply_image_augmentation: bool = True,
        line_and_cp_xy: tuple[torch.Tensor, torch.Tensor] | None = None,
        augment_transform: Any | None = None,
    ) -> tuple[FiberStripSample, np.ndarray | None, np.ndarray | None, np.ndarray]:
        offset = float(self.strip_z_offsets[int(offset_index)])
        grid = self._offset_grid_from_source(source, offset)
        coords_zyx_t = grid.coords_zyx
        valid_mask_t = grid.valid_mask
        if params is not None:
            if augment_transform is None:
                augment_transform = strip_augment_transform(
                    self.config.patch_shape_hw,
                    source.source_shape_hw,
                    params,
                    device=device,
                )
            with _ProfileBlock(profile, "coord_augmentation", device):
                coords_zyx_t, valid_mask_t = _resample_coord_tensors_like_augmentation(
                    coords_zyx_t,
                    valid_mask_t,
                    params,
                    output_shape_hw=self.config.patch_shape_hw,
                    device=device,
                    transform=augment_transform,
                )
        if line_and_cp_xy is None:
            with _ProfileBlock(profile, "line_coords", device):
                line_xy_t, control_point_xy_t = self._line_and_cp_xy_for_params(
                    source,
                    params,
                    device=device,
                    transform=augment_transform,
                )
        else:
            line_xy_t, control_point_xy_t = line_and_cp_xy
        coords_zyx = _as_numpy_float32(coords_zyx_t)
        valid_mask = _as_numpy_bool(valid_mask_t)
        line_xy = _as_numpy_float32(line_xy_t)
        control_point_xy = _as_numpy_float32(control_point_xy_t)
        image: np.ndarray | None = None
        sampled_valid: np.ndarray | None = None
        if load_image:
            with _ProfileBlock(profile, "volume_sample"):
                result = source.record.sampler.sample_coords(coords_zyx, valid_mask)
                image = result.image
                sampled_valid = result.valid_mask
                if profile is not None:
                    for key, value in result.stats.items():
                        if isinstance(value, (int, float)):
                            profile[f"volume_stat_{key}"] = profile.get(f"volume_stat_{key}", 0.0) + float(value)
            if params is not None and apply_image_augmentation:
                with _ProfileBlock(profile, "value_augmentation", device):
                    image_t, valid_t = apply_value_augmentation(
                        image,
                        sampled_valid,
                        value_only_params(params),
                        device=device,
                    )
                    image = image_t.cpu().numpy().astype(np.float32)
                    sampled_valid = valid_t.cpu().numpy().astype(bool)
        sample_valid = valid_mask if sampled_valid is None else sampled_valid
        sample = FiberStripSample(
            record_index=source.record_index,
            fiber_path=str(source.record.fiber.path) if source.record.fiber.path is not None else "",
            control_point_index=source.control_point_index,
            control_point_xyz=np.asarray(
                source.record.fiber.control_points_xyz[source.control_point_index], dtype=np.float32
            ),
            strip_z_offset=offset,
            coords_zyx=coords_zyx,
            valid_mask=sample_valid,
            frame=grid.frame,
            line_xy=line_xy,
            control_point_xy=control_point_xy,
        )
        return sample, image, sampled_valid, line_xy

    def _prefetch_envelope_coords_from_source(
        self,
        source: _StripSource,
        offset_index: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        offset = float(self.strip_z_offsets[int(offset_index)])
        grid = self._offset_grid_from_source(source, offset)
        return _as_numpy_float32(grid.coords_zyx), _as_numpy_bool(grid.valid_mask)

    def _prepare_sample(
        self,
        sample_index: int,
        *,
        sample_mode: str = "random",
        descriptor: tuple[_Record, int, int] | None = None,
        augmentation_sample_index: int | None = None,
        value_augmentation_sample_index: int | None = None,
        profile: dict[str, float] | None = None,
        include_line_xy: bool = True,
    ) -> _PreparedStripSample:
        device = resolve_torch_device(self.config.augment.device)
        source = self.build_strip_source(
            sample_index,
            device=device,
            sample_mode=sample_mode,
            descriptor=descriptor,
            profile=profile,
        )
        augment_index = int(sample_index) if augmentation_sample_index is None else int(augmentation_sample_index)
        def augmentation_params_for_offset(offset_index: int) -> FiberStripAugmentParams | None:
            if not self.config.augment.enabled:
                return None
            geometric = random_combined_augmentation(self.config.augment, augment_index, offset_index)
            if value_augmentation_sample_index is None:
                return geometric
            value = random_combined_augmentation(
                self.config.augment,
                int(value_augmentation_sample_index),
                offset_index,
            )
            return _sync_value_augmentation_params(geometric, value)

        params_by_offset: list[FiberStripAugmentParams | None] = [
            augmentation_params_for_offset(offset_index)
            for offset_index, _ in enumerate(self.strip_z_offsets)
        ]
        transforms_by_params: dict[FiberStripAugmentParams, Any] = {}
        if self.config.augment.enabled:
            with _ProfileBlock(profile, "map_build", device):
                for params in dict.fromkeys(params for params in params_by_offset if params is not None):
                    assert params is not None
                    transforms_by_params[params] = strip_augment_transform(
                        self.config.patch_shape_hw,
                        source.source_shape_hw,
                        params,
                        device=device,
                    )

        line_and_cp_cache: dict[FiberStripAugmentParams | None, tuple[torch.Tensor, torch.Tensor]] = {}
        if not self.config.augment.enabled:
            source_cp_xy = source.source_control_point_xy.to(device=device, dtype=torch.float32)
            if include_line_xy:
                line_xy = source.source_line_xy.to(device=device, dtype=torch.float32)
            else:
                unit = torch.tensor([1.0, 0.0], dtype=torch.float32, device=device)
                line_xy = torch.stack([source_cp_xy - unit, source_cp_xy + unit], dim=0)
            line_and_cp_cache[None] = (line_xy, source_cp_xy)
        else:
            unique_params = [params for params in dict.fromkeys(params_by_offset) if params is not None]
            if unique_params:
                source_cp_xy = source.source_control_point_xy.to(device=device, dtype=torch.float32).view(1, 2)
                if include_line_xy:
                    source_line_xy = source.source_line_xy.to(device=device, dtype=torch.float32)
                    source_points = torch.cat([source_line_xy, source_cp_xy], dim=0)
                    cp_point_index = int(source_points.shape[0] - 1)
                else:
                    unit = torch.tensor([[1.0, 0.0]], dtype=torch.float32, device=device)
                    source_points = torch.cat([source_cp_xy - unit, source_cp_xy, source_cp_xy + unit], dim=0)
                    cp_point_index = 1
                point_batch = source_points.unsqueeze(0).expand(len(unique_params), -1, -1)
                forward_maps = torch.stack([transforms_by_params[params].forward_map_xy for params in unique_params], dim=0)
                line_lookup_before = 0.0 if profile is None else profile.get("line_lookup", 0.0)
                with _ProfileBlock(profile, "line_lookup", device):
                    mapped_batch, _ = sample_xy_maps_bilinear(forward_maps, point_batch)
                height, width = self.config.patch_shape_hw
                line_filter_before = 0.0 if profile is None else profile.get("line_filter", 0.0)
                with _ProfileBlock(profile, "line_filter", device):
                    for index, params in enumerate(unique_params):
                        mapped = mapped_batch[index]
                        if include_line_xy:
                            line_xy = mapped[:cp_point_index]
                            cp_xy = mapped[cp_point_index]
                            line_xy = line_xy[
                                torch.isfinite(line_xy).all(dim=1)
                                & (line_xy[:, 0] >= 0.0)
                                & (line_xy[:, 0] <= float(int(width) - 1))
                                & (line_xy[:, 1] >= 0.0)
                                & (line_xy[:, 1] <= float(int(height) - 1))
                            ]
                        else:
                            cp_xy = mapped[cp_point_index]
                            line_xy = torch.stack([mapped[0], mapped[2]], dim=0)
                        line_and_cp_cache[params] = (line_xy, cp_xy)
                if profile is not None:
                    profile["line_coords"] = profile.get("line_coords", 0.0) + (
                        profile.get("line_lookup", 0.0)
                        - line_lookup_before
                        + profile.get("line_filter", 0.0)
                        - line_filter_before
                    )

        offset_grids = [self._offset_grid_from_source(source, offset) for offset in self.strip_z_offsets]
        coords_zyx_t = torch.stack([grid.coords_zyx.to(device=device, dtype=torch.float32) for grid in offset_grids], dim=0)
        valid_mask_t = torch.stack([grid.valid_mask.to(device=device, dtype=torch.bool) for grid in offset_grids], dim=0)
        if self.config.augment.enabled:
            backward_maps = torch.stack(
                [transforms_by_params[params].backward_map_xy for params in params_by_offset if params is not None],
                dim=0,
            )
            coord_aug_before = 0.0 if profile is None else profile.get("coord_aug_batch", 0.0)
            with _ProfileBlock(profile, "coord_aug_batch", device):
                coords_zyx_t, valid_mask_t = _resample_coord_tensor_batch_like_augmentation(
                    coords_zyx_t,
                    valid_mask_t,
                    backward_maps,
                )
            if profile is not None:
                profile["coord_augmentation"] = profile.get("coord_augmentation", 0.0) + (
                    profile.get("coord_aug_batch", 0.0) - coord_aug_before
                )

        coords_zyx_np = _as_numpy_float32(coords_zyx_t)
        valid_mask_np = _as_numpy_bool(valid_mask_t)
        line_xy_by_offset: list[np.ndarray] = []
        control_point_xy_by_offset: list[np.ndarray] = []
        for offset_index, _ in enumerate(self.strip_z_offsets):
            params = params_by_offset[offset_index]
            line_xy_t, control_point_xy_t = line_and_cp_cache[params]
            line_xy_by_offset.append(_as_numpy_float32(line_xy_t))
            control_point_xy_by_offset.append(_as_numpy_float32(control_point_xy_t))
        return _PreparedStripSample(
            source=source,
            params_by_offset=params_by_offset,
            offset_grids=offset_grids,
            coords_zyx=coords_zyx_np,
            valid_mask=valid_mask_np,
            line_xy_by_offset=line_xy_by_offset,
            control_point_xy_by_offset=control_point_xy_by_offset,
        )

    def _prepare_top_sample_from_side_sample(
        self,
        side_sample: FiberStripSample,
        params: FiberStripAugmentParams | None,
        *,
        device: torch.device,
        profile: dict[str, float] | None = None,
        include_line_xy: bool = True,
    ) -> _PreparedTopStripSample:
        record_index = int(side_sample.record_index)
        cp_index = int(side_sample.control_point_index)
        if record_index < 0 or record_index >= len(self.records):
            raise IndexError(f"record_index {record_index} out of range for {len(self.records)} records")
        record = self.records[record_index]
        source = self.build_top_strip_source(
            0,
            device=device,
            descriptor=(record, record_index, cp_index),
            profile=profile,
        )
        grid = source.grid

        transform = None
        if params is not None:
            with _ProfileBlock(profile, "map_build", device):
                transform = strip_augment_transform(
                    self.config.patch_shape_hw,
                    source.source_shape_hw,
                    params,
                    device=device,
                )

        side_cp_xy = torch.as_tensor(side_sample.control_point_xy, dtype=torch.float32, device=device)
        if include_line_xy:
            line_xy_t = torch.as_tensor(side_sample.line_xy, dtype=torch.float32, device=device)
            control_point_xy_t = side_cp_xy
        else:
            unit = torch.tensor([1.0, 0.0], dtype=torch.float32, device=device)
            line_xy_t = torch.stack([side_cp_xy - unit, side_cp_xy + unit], dim=0)
            control_point_xy_t = side_cp_xy

        coords_zyx_t = grid.coords_zyx.to(device=device, dtype=torch.float32).unsqueeze(0)
        valid_mask_t = grid.valid_mask.to(device=device, dtype=torch.bool).unsqueeze(0)
        if params is not None:
            assert transform is not None
            coord_aug_before = 0.0 if profile is None else profile.get("coord_aug_batch", 0.0)
            with _ProfileBlock(profile, "coord_aug_batch", device):
                coords_zyx_t, valid_mask_t = _resample_coord_tensor_batch_like_augmentation(
                    coords_zyx_t,
                    valid_mask_t,
                    transform.backward_map_xy.unsqueeze(0),
                )
            if profile is not None:
                profile["coord_augmentation"] = profile.get("coord_augmentation", 0.0) + (
                    profile.get("coord_aug_batch", 0.0) - coord_aug_before
                )

        return _PreparedTopStripSample(
            source=source,
            params=params,
            grid=grid,
            coords_zyx=_as_numpy_float32(coords_zyx_t[0]),
            valid_mask=_as_numpy_bool(valid_mask_t[0]),
            line_xy=_as_numpy_float32(line_xy_t),
            control_point_xy=_as_numpy_float32(control_point_xy_t),
        )

    def _finish_prepared_top_sample(
        self,
        prepared: _PreparedTopStripSample,
        image_np: np.ndarray,
        valid_np: np.ndarray,
        *,
        include_coords: bool,
    ) -> tuple[FiberStripSample, np.ndarray, np.ndarray, np.ndarray]:
        source = prepared.source
        image = np.asarray(image_np, dtype=np.float32)
        valid = np.asarray(valid_np, dtype=bool)
        sample = FiberStripSample(
            record_index=source.record_index,
            fiber_path=str(source.record.fiber.path) if source.record.fiber.path is not None else "",
            control_point_index=source.control_point_index,
            control_point_xyz=np.asarray(
                source.record.fiber.control_points_xyz[source.control_point_index], dtype=np.float32
            ),
            strip_z_offset=0.0,
            coords_zyx=(
                np.asarray(prepared.coords_zyx, dtype=np.float32)
                if include_coords
                else np.empty((0, 0, 3), dtype=np.float32)
            ),
            valid_mask=valid,
            frame=prepared.grid.frame,
            line_xy=np.asarray(prepared.line_xy, dtype=np.float32),
            control_point_xy=np.asarray(prepared.control_point_xy, dtype=np.float32),
        )
        coords = (
            np.asarray(prepared.coords_zyx, dtype=np.float32)
            if include_coords
            else np.empty((0, 0, 3), dtype=np.float32)
        )
        return sample, image, coords, valid

    def load_top_batch_for_batch(
        self,
        batch: FiberStrip2DBatch,
        *,
        profile: dict[str, float] | None = None,
        include_line_xy: bool = True,
        include_coords: bool = True,
    ) -> FiberStrip2DBatch:
        images_np = np.asarray(batch.images)
        if images_np.ndim != 5 or images_np.shape[2] != 1:
            raise ValueError("batch.images must have shape B,Z,1,H,W")
        control_point_count = int(images_np.shape[0])
        offset_count = int(images_np.shape[1])
        if len(batch.samples) != control_point_count * offset_count:
            raise ValueError("batch.samples length does not match batch image shape")
        if control_point_count <= 0 or offset_count <= 0:
            raise ValueError("batch must contain at least one control point and offset")
        center_offset_index = int(np.argmin(np.abs(np.asarray(batch.strip_z_offsets, dtype=np.float32)[:offset_count])))
        params = tuple(batch.augmentation_params)
        if params and len(params) != control_point_count * offset_count:
            raise ValueError("batch augmentation_params length does not match batch image shape")
        selected_samples: list[FiberStripSample] = []
        selected_params: list[FiberStripAugmentParams | None] = []
        for cp_row in range(control_point_count):
            flat_index = cp_row * offset_count + center_offset_index
            selected_samples.append(batch.samples[flat_index])
            selected_params.append(params[flat_index] if params else None)

        device = resolve_torch_device(self.config.augment.device)
        worker_count = min(max(1, int(self.config.loader_workers)), max(1, control_point_count))

        def build_one(item_index: int) -> tuple[int, dict[str, float], _PreparedTopStripSample]:
            local_profile: dict[str, float] = {} if profile is not None else {}
            worker_start = time.perf_counter()
            prepared = self._prepare_top_sample_from_side_sample(
                selected_samples[item_index],
                selected_params[item_index],
                device=device,
                profile=local_profile if profile is not None else None,
                include_line_xy=include_line_xy,
            )
            if profile is not None:
                local_profile["load_batch_worker"] = (time.perf_counter() - worker_start) * 1000.0
            return item_index, local_profile, prepared

        begin_zarr_cache_trace()
        top_wall_start = time.perf_counter()
        try:
            prepared_items: list[tuple[_PreparedTopStripSample, dict[str, float]]] = []
            if worker_count == 1:
                for item_index in range(control_point_count):
                    returned_index, local_profile, prepared = build_one(item_index)
                    if returned_index != item_index:
                        raise RuntimeError("internal top batch ordering mismatch")
                    prepared_items.append((prepared, local_profile))
            else:
                executor = self._get_loader_executor(worker_count)
                futures = [executor.submit(build_one, item_index) for item_index in range(control_point_count)]
                ordered: list[tuple[_PreparedTopStripSample, dict[str, float]] | None] = [None] * control_point_count
                for future in futures:
                    item_index, local_profile, prepared = future.result()
                    ordered[int(item_index)] = (prepared, local_profile)
                prepared_items = [item for item in ordered if item is not None]
            if len(prepared_items) != control_point_count:
                raise RuntimeError("internal top batch preparation count mismatch")

            images_by_item: list[np.ndarray | None] = [None] * control_point_count
            valids_by_item: list[np.ndarray | None] = [None] * control_point_count
            groups: dict[int, list[int]] = {}
            samplers: dict[int, CoordinateSampler] = {}
            for item_index, (prepared, _) in enumerate(prepared_items):
                sampler = prepared.source.record.sampler
                key = id(sampler)
                groups.setdefault(key, []).append(item_index)
                samplers[key] = sampler
            for key, item_indices in groups.items():
                coords = np.stack(
                    [
                        np.asarray(prepared_items[item_index][0].coords_zyx, dtype=np.float32)
                        for item_index in item_indices
                    ],
                    axis=0,
                )
                valid = np.stack(
                    [
                        np.asarray(prepared_items[item_index][0].valid_mask, dtype=bool)
                        for item_index in item_indices
                    ],
                    axis=0,
                )
                with _ProfileBlock(profile, "volume_sample"):
                    result = samplers[key].sample_coord_batch(coords, valid)
                images = np.asarray(result.image, dtype=np.float32)
                valids = np.asarray(result.valid_mask, dtype=bool)
                if images.shape != valid.shape:
                    raise ValueError(
                        "top batched coordinate sampler returned incompatible image shape: "
                        f"shape={images.shape} expected={valid.shape}"
                    )
                if valids.shape != valid.shape:
                    raise ValueError(
                        "top batched coordinate sampler returned incompatible valid-mask shape: "
                        f"shape={valids.shape} expected={valid.shape}"
                    )
                if profile is not None:
                    for stat_key, value in result.stats.items():
                        if isinstance(value, (int, float)):
                            profile[f"volume_stat_{stat_key}"] = (
                                profile.get(f"volume_stat_{stat_key}", 0.0) + float(value)
                            )
                for local_offset, item_index in enumerate(item_indices):
                    images_by_item[item_index] = images[local_offset]
                    valids_by_item[item_index] = valids[local_offset]

            samples: list[FiberStripSample] = []
            images: list[np.ndarray] = []
            coords: list[np.ndarray] = []
            valids: list[np.ndarray] = []
            record_indices: list[int] = []
            cp_indices: list[int] = []
            fiber_paths: list[str] = []
            for item_index, (prepared, local_profile) in enumerate(prepared_items):
                image = images_by_item[item_index]
                valid = valids_by_item[item_index]
                if image is None or valid is None:
                    raise RuntimeError("internal missing top sampled image")
                sample, sample_image, sample_coords, sample_valid = self._finish_prepared_top_sample(
                    prepared,
                    image,
                    valid,
                    include_coords=include_coords,
                )
                _merge_profile(profile, local_profile)
                samples.append(sample)
                images.append(sample_image)
                coords.append(sample_coords)
                valids.append(sample_valid)
                record_indices.append(sample.record_index)
                cp_indices.append(sample.control_point_index)
                fiber_paths.append(sample.fiber_path)
        finally:
            cache_stats = end_zarr_cache_trace()
            if profile is not None:
                profile["load_batch_wall"] = profile.get("load_batch_wall", 0.0) + (
                    time.perf_counter() - top_wall_start
                ) * 1000.0
        return FiberStrip2DBatch(
            images=np.stack(images, axis=0)[:, None, None, :, :],
            coords_zyx=np.stack(coords, axis=0)[:, None, :, :, :],
            valid_mask=np.stack(valids, axis=0)[:, None, :, :],
            strip_z_offsets=np.asarray([0.0], dtype=np.float32),
            control_point_indices=np.asarray(cp_indices, dtype=np.int32),
            record_indices=np.asarray(record_indices, dtype=np.int32),
            fiber_paths=tuple(fiber_paths),
            samples=tuple(samples),
            cache_stats=cache_stats,
            augmentation_params=tuple(selected_params),
        )

    def _finish_prepared_sample(
        self,
        prepared: _PreparedStripSample,
        images_np: np.ndarray,
        valids_np: np.ndarray,
        *,
        apply_image_augmentation: bool,
        include_coords: bool = True,
        profile: dict[str, float] | None = None,
    ) -> tuple[
        list[FiberStripSample],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        tuple[FiberStripAugmentParams | None, ...],
    ]:
        device = resolve_torch_device(self.config.augment.device)
        source = prepared.source
        coords_zyx_np = np.asarray(prepared.coords_zyx, dtype=np.float32)
        images_np = np.asarray(images_np, dtype=np.float32)
        valids_np = np.asarray(valids_np, dtype=bool)
        if images_np.shape != prepared.valid_mask.shape:
            raise ValueError(
                "batched sampler returned incompatible image shape: "
                f"shape={images_np.shape} expected={prepared.valid_mask.shape}"
            )
        if valids_np.shape != prepared.valid_mask.shape:
            raise ValueError(
                "batched sampler returned incompatible valid-mask shape: "
                f"shape={valids_np.shape} expected={prepared.valid_mask.shape}"
            )
        images: list[np.ndarray] = []
        valids: list[np.ndarray] = []
        samples: list[FiberStripSample] = []
        for offset_index, offset in enumerate(self.strip_z_offsets):
            line_xy = prepared.line_xy_by_offset[offset_index]
            control_point_xy = prepared.control_point_xy_by_offset[offset_index]
            coords_zyx = (
                coords_zyx_np[offset_index]
                if include_coords
                else np.empty((0, 0, 3), dtype=np.float32)
            )
            image = images_np[offset_index]
            sampled_valid = valids_np[offset_index]
            sample = FiberStripSample(
                record_index=source.record_index,
                fiber_path=str(source.record.fiber.path) if source.record.fiber.path is not None else "",
                control_point_index=source.control_point_index,
                control_point_xyz=np.asarray(
                    source.record.fiber.control_points_xyz[source.control_point_index], dtype=np.float32
                ),
                strip_z_offset=float(offset),
                coords_zyx=coords_zyx,
                valid_mask=sampled_valid,
                frame=prepared.offset_grids[offset_index].frame,
                line_xy=line_xy,
                control_point_xy=control_point_xy,
            )
            images.append(image)
            valids.append(sampled_valid)
            samples.append(sample)
        if self.config.augment.enabled and apply_image_augmentation:
            value_params = [value_only_params(params) for params in prepared.params_by_offset if params is not None]
            value_aug_before = 0.0 if profile is None else profile.get("value_aug_batch", 0.0)
            with _ProfileBlock(profile, "value_aug_batch", device):
                image_t, valid_t = apply_value_augmentation_batch(
                    np.stack(images, axis=0),
                    np.stack(valids, axis=0),
                    value_params,
                    device=device,
                )
                images = list(image_t.cpu().numpy().astype(np.float32))
                valids = list(valid_t.cpu().numpy().astype(bool))
            if profile is not None:
                profile["value_augmentation"] = profile.get("value_augmentation", 0.0) + (
                    profile.get("value_aug_batch", 0.0) - value_aug_before
                )
        return (
            samples,
            np.stack(images, axis=0),
            coords_zyx_np if include_coords else np.empty((len(self.strip_z_offsets), 0, 0, 3), dtype=np.float32),
            np.stack(valids, axis=0),
            tuple(prepared.params_by_offset),
        )

    def _build_sample_with_params(
        self,
        sample_index: int,
        *,
        sample_mode: str = "random",
        profile: dict[str, float] | None = None,
        apply_image_augmentation: bool = True,
        include_line_xy: bool = True,
        include_coords: bool = True,
    ) -> tuple[
        list[FiberStripSample],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        tuple[FiberStripAugmentParams | None, ...],
    ]:
        try:
            prepared = self._prepare_sample(
                sample_index,
                sample_mode=sample_mode,
                profile=profile,
                include_line_xy=include_line_xy,
            )
        except TypeError as exc:
            if "include_line_xy" not in str(exc):
                raise
            prepared = self._prepare_sample(
                sample_index,
                sample_mode=sample_mode,
                profile=profile,
            )
        with _ProfileBlock(profile, "volume_sample"):
            result = prepared.source.record.sampler.sample_coord_batch(prepared.coords_zyx, prepared.valid_mask)
            images_np = np.asarray(result.image, dtype=np.float32)
            valids_np = np.asarray(result.valid_mask, dtype=bool)
            if profile is not None:
                for key, value in result.stats.items():
                    if isinstance(value, (int, float)):
                        profile[f"volume_stat_{key}"] = profile.get(f"volume_stat_{key}", 0.0) + float(value)
        return self._finish_prepared_sample(
            prepared,
            images_np,
            valids_np,
            apply_image_augmentation=apply_image_augmentation,
            include_coords=include_coords,
            profile=profile,
        )

    def _finish_prepared_batch_samples(
        self,
        prepared_items: list[tuple[_PreparedStripSample, dict[str, float]]],
        *,
        apply_image_augmentation: bool,
        include_coords: bool,
        profile: dict[str, float] | None = None,
    ) -> list[
        tuple[
            dict[str, float],
            tuple[
                list[FiberStripSample],
                np.ndarray,
                np.ndarray,
                np.ndarray,
                tuple[FiberStripAugmentParams | None, ...],
            ],
        ]
    ]:
        if not prepared_items:
            return []

        images_by_item: list[np.ndarray | None] = [None] * len(prepared_items)
        valids_by_item: list[np.ndarray | None] = [None] * len(prepared_items)
        groups: dict[int, list[int]] = {}
        samplers: dict[int, CoordinateSampler] = {}
        for item_index, (prepared, _) in enumerate(prepared_items):
            sampler = prepared.source.record.sampler
            key = id(sampler)
            groups.setdefault(key, []).append(item_index)
            samplers[key] = sampler

        for key, item_indices in groups.items():
            coords = np.concatenate(
                [
                    np.asarray(prepared_items[item_index][0].coords_zyx, dtype=np.float32)
                    for item_index in item_indices
                ],
                axis=0,
            )
            valid = np.concatenate(
                [
                    np.asarray(prepared_items[item_index][0].valid_mask, dtype=bool)
                    for item_index in item_indices
                ],
                axis=0,
            )
            with _ProfileBlock(profile, "volume_sample"):
                result = samplers[key].sample_coord_batch(coords, valid)
            images = np.asarray(result.image, dtype=np.float32)
            valids = np.asarray(result.valid_mask, dtype=bool)
            if images.shape != valid.shape:
                raise ValueError(
                    "batched coordinate sampler returned incompatible image shape: "
                    f"shape={images.shape} expected={valid.shape}"
                )
            if valids.shape != valid.shape:
                raise ValueError(
                    "batched coordinate sampler returned incompatible valid-mask shape: "
                    f"shape={valids.shape} expected={valid.shape}"
                )
            if profile is not None:
                for stat_key, value in result.stats.items():
                    if isinstance(value, (int, float)):
                        profile[f"volume_stat_{stat_key}"] = (
                            profile.get(f"volume_stat_{stat_key}", 0.0) + float(value)
                        )
            offset = 0
            for item_index in item_indices:
                prepared = prepared_items[item_index][0]
                patch_count = int(prepared.coords_zyx.shape[0])
                images_by_item[item_index] = images[offset : offset + patch_count]
                valids_by_item[item_index] = valids[offset : offset + patch_count]
                offset += patch_count
            if offset != int(images.shape[0]):
                raise RuntimeError("internal batched coordinate sampler dispatch mismatch")

        finished: list[
            tuple[
                dict[str, float],
                tuple[
                    list[FiberStripSample],
                    np.ndarray,
                    np.ndarray,
                    np.ndarray,
                    tuple[FiberStripAugmentParams | None, ...],
                ],
            ]
        ] = []
        for item_index, (prepared, local_profile) in enumerate(prepared_items):
            item_images = images_by_item[item_index]
            item_valids = valids_by_item[item_index]
            if item_images is None or item_valids is None:
                raise RuntimeError("internal missing sampled image batch item")
            result = self._finish_prepared_sample(
                prepared,
                item_images,
                item_valids,
                apply_image_augmentation=apply_image_augmentation,
                include_coords=include_coords,
                profile=local_profile,
            )
            finished.append((local_profile, result))
        return finished

    def build_sample(
        self,
        sample_index: int,
        *,
        sample_mode: str = "random",
        profile: dict[str, float] | None = None,
        apply_image_augmentation: bool = True,
    ) -> tuple[list[FiberStripSample], np.ndarray, np.ndarray, np.ndarray]:
        samples, images, coords, valids, _ = self._build_sample_with_params(
            sample_index,
            sample_mode=sample_mode,
            profile=profile,
            apply_image_augmentation=apply_image_augmentation,
            include_line_xy=True,
        )
        return samples, images, coords, valids

    def apply_batch_image_augmentation(
        self,
        batch: FiberStrip2DBatch,
        *,
        profile: dict[str, float] | None = None,
    ) -> FiberStrip2DBatch:
        if not self.config.augment.enabled:
            return batch
        images_np = np.asarray(batch.images, dtype=np.float32)
        valids_np = np.asarray(batch.valid_mask, dtype=bool)
        if images_np.ndim != 5 or images_np.shape[2] != 1:
            raise ValueError("batch.images must have shape B,Z,1,H,W")
        expected_valid_shape = (images_np.shape[0], images_np.shape[1], images_np.shape[3], images_np.shape[4])
        if valids_np.shape != expected_valid_shape:
            raise ValueError("batch.valid_mask shape must match batch.images")
        flat_count = int(images_np.shape[0]) * int(images_np.shape[1])
        params = tuple(batch.augmentation_params)
        if len(params) != flat_count:
            raise ValueError(
                "batch does not carry image augmentation parameters: "
                f"params={len(params)} expected={flat_count}"
            )
        if all(param is None for param in params):
            return batch
        value_params = [value_only_params(param) for param in params if param is not None]
        if len(value_params) != flat_count:
            raise ValueError("partially missing image augmentation parameters in batch")

        flat_images = images_np.reshape(flat_count, int(images_np.shape[3]), int(images_np.shape[4]))
        flat_valids = valids_np.reshape(flat_count, int(valids_np.shape[2]), int(valids_np.shape[3]))
        device = resolve_torch_device(self.config.augment.device)
        value_aug_before = 0.0 if profile is None else profile.get("value_aug_batch", 0.0)
        with _ProfileBlock(profile, "value_aug_batch", device):
            image_t, valid_t = apply_value_augmentation_batch(
                flat_images,
                flat_valids,
                value_params,
                device=device,
            )
            images = image_t.cpu().numpy().astype(np.float32).reshape(images_np.shape)
            valids = valid_t.cpu().numpy().astype(bool).reshape(valids_np.shape)
        if profile is not None:
            profile["value_augmentation"] = profile.get("value_augmentation", 0.0) + (
                profile.get("value_aug_batch", 0.0) - value_aug_before
            )
        return replace(batch, images=images, valid_mask=valids)

    def build_center_strip_patch(
        self,
        sample_index: int,
        *,
        device: torch.device | None = None,
    ) -> tuple[FiberStripSample, np.ndarray, np.ndarray]:
        resolved_device = resolve_torch_device(self.config.augment.device) if device is None else device
        source = self.build_strip_source(sample_index, device=resolved_device)
        center_index = min(range(len(self.strip_z_offsets)), key=lambda index: abs(float(self.strip_z_offsets[index])))
        sample, image, valid_mask, _ = self.build_strip_patch_from_source(
            source,
            center_index,
            None,
            device=resolved_device,
            load_image=True,
        )
        assert image is not None
        assert valid_mask is not None
        return sample, image.astype(np.float32, copy=False), valid_mask.astype(bool, copy=False)

    @staticmethod
    def _line_arc_lengths(points_xyz: np.ndarray) -> np.ndarray:
        points = np.asarray(points_xyz, dtype=np.float64)
        if points.shape[0] <= 1:
            return np.zeros(points.shape[0], dtype=np.float64)
        segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        return np.concatenate([[0.0], np.cumsum(segment_lengths)])

    @staticmethod
    def _source_patch_corners_xy(source_shape_hw: tuple[int, int], *, device: torch.device) -> torch.Tensor:
        source_height, source_width = (int(v) for v in source_shape_hw)
        return torch.tensor(
            [
                [0.0, 0.0],
                [float(source_width - 1), 0.0],
                [float(source_width - 1), float(source_height - 1)],
                [0.0, float(source_height - 1)],
            ],
            dtype=torch.float32,
            device=device,
        )

    @staticmethod
    def _finite_source_xy_grid(transform: Any, source_shape_hw: tuple[int, int]) -> np.ndarray:
        source_height, source_width = (int(v) for v in source_shape_hw)
        source_xy_t = transform.output_to_source_grid().detach()
        inside = (
            torch.isfinite(source_xy_t).all(dim=-1)
            & (source_xy_t[..., 0] >= 0.0)
            & (source_xy_t[..., 0] <= float(source_width - 1))
            & (source_xy_t[..., 1] >= 0.0)
            & (source_xy_t[..., 1] <= float(source_height - 1))
        )
        source_xy_t = torch.where(
            inside[..., None],
            source_xy_t,
            torch.full_like(source_xy_t, float("nan")),
        )
        return _as_numpy_float32(source_xy_t)

    @staticmethod
    def _finite_reference_to_output_xy_grid(transform: Any, output_shape_hw: tuple[int, int]) -> np.ndarray:
        output_height, output_width = (int(v) for v in output_shape_hw)
        output_xy_t = transform.forward_map_xy.detach()
        inside = (
            torch.isfinite(output_xy_t).all(dim=-1)
            & (output_xy_t[..., 0] >= 0.0)
            & (output_xy_t[..., 0] <= float(output_width - 1))
            & (output_xy_t[..., 1] >= 0.0)
            & (output_xy_t[..., 1] <= float(output_height - 1))
        )
        output_xy_t = torch.where(
            inside[..., None],
            output_xy_t,
            torch.full_like(output_xy_t, float("nan")),
        )
        return _as_numpy_float32(output_xy_t)

    @staticmethod
    def _filter_line_xy_t(line_xy: torch.Tensor, output_shape_hw: tuple[int, int]) -> torch.Tensor:
        if line_xy.numel() == 0:
            return line_xy.to(dtype=torch.float32)
        output_height, output_width = (int(v) for v in output_shape_hw)
        return line_xy[
            torch.isfinite(line_xy).all(dim=1)
            & (line_xy[:, 0] >= 0.0)
            & (line_xy[:, 0] <= float(output_width - 1))
            & (line_xy[:, 1] >= 0.0)
            & (line_xy[:, 1] <= float(output_height - 1))
        ].to(dtype=torch.float32)

    def _tta_transform_for_source_shape(
        self,
        source_shape_hw: tuple[int, int],
        params: FiberStripAugmentParams,
        *,
        rf_margin_px: float,
        device: torch.device,
    ) -> tuple[tuple[int, int], FiberStripAugmentParams, Any, np.ndarray]:
        source_height, source_width = (int(v) for v in source_shape_hw)
        if source_height <= 0 or source_width <= 0:
            raise ValueError(f"invalid TTA source shape {source_shape_hw}")
        corners_source = self._source_patch_corners_xy(source_shape_hw, device=device)
        pad = max(1, int(math.ceil(max(0.0, float(rf_margin_px)))))
        output_shape = (source_height, source_width)
        transform = strip_augment_transform(output_shape, source_shape_hw, params, device=device)
        corners_out = transform.source_to_output_points(corners_source)
        for _ in range(4):
            transform = strip_augment_transform(output_shape, source_shape_hw, params, device=device)
            corners_out = transform.source_to_output_points(corners_source)
            if not bool(torch.isfinite(corners_out).all().item()):
                raise ValueError("TTA transform produced non-finite output corners")
            min_xy = torch.min(corners_out, dim=0).values
            max_xy = torch.max(corners_out, dim=0).values
            grow_x = max(
                0.0,
                float(pad) - float(min_xy[0]),
                float(max_xy[0]) - (float(output_shape[1] - 1) - float(pad)),
            )
            grow_y = max(
                0.0,
                float(pad) - float(min_xy[1]),
                float(max_xy[1]) - (float(output_shape[0] - 1) - float(pad)),
            )
            if grow_x <= 1.0e-5 and grow_y <= 1.0e-5:
                break
            output_shape = (
                int(output_shape[0]) + int(math.ceil(2.0 * grow_y)),
                int(output_shape[1]) + int(math.ceil(2.0 * grow_x)),
            )
        transform = strip_augment_transform(output_shape, source_shape_hw, params, device=device)
        corners_out = transform.source_to_output_points(corners_source)
        if not bool(torch.isfinite(corners_out).all().item()):
            raise ValueError("TTA transform produced non-finite final corners")
        return output_shape, params, transform, _as_numpy_float32(corners_out)

    def build_center_tta_patch_from_sample(
        self,
        sample: FiberStripSample,
        params: FiberStripAugmentParams,
        *,
        rf_margin_px: float = 0.0,
        device: torch.device | None = None,
    ) -> FiberStripTtaPatch:
        resolved_device = resolve_torch_device(self.config.augment.device) if device is None else device
        record = self.records[int(sample.record_index)]
        source_shape_hw = tuple(int(v) for v in np.asarray(sample.coords_zyx).shape[:2])
        output_shape, _, transform, corners_xy = self._tta_transform_for_source_shape(
            source_shape_hw,
            params,
            rf_margin_px=rf_margin_px,
            device=resolved_device,
        )
        coords_t, valid_t = _resample_coord_tensors_like_augmentation(
            torch.as_tensor(sample.coords_zyx, dtype=torch.float32, device=resolved_device),
            torch.as_tensor(sample.valid_mask, dtype=torch.bool, device=resolved_device),
            params,
            output_shape_hw=output_shape,
            device=resolved_device,
            transform=transform,
        )
        coords_zyx = _as_numpy_float32(coords_t)
        coord_valid = _as_numpy_bool(valid_t)
        result = record.sampler.sample_coords(coords_zyx, coord_valid)
        image = np.asarray(result.image, dtype=np.float32)
        valid_mask = np.asarray(result.valid_mask, dtype=bool)
        source_line = torch.as_tensor(sample.line_xy, dtype=torch.float32, device=resolved_device)
        source_cp = torch.as_tensor(sample.control_point_xy, dtype=torch.float32, device=resolved_device).view(1, 2)
        mapped = transform.source_to_output_points(torch.cat([source_line, source_cp], dim=0))
        line_xy = self._filter_line_xy_t(mapped[:-1], output_shape)
        cp_xy = mapped[-1]
        tta_sample = FiberStripSample(
            record_index=int(sample.record_index),
            fiber_path=sample.fiber_path,
            control_point_index=int(sample.control_point_index),
            control_point_xyz=np.asarray(sample.control_point_xyz, dtype=np.float32),
            strip_z_offset=float(sample.strip_z_offset),
            coords_zyx=coords_zyx,
            valid_mask=valid_mask,
            frame=sample.frame,
            line_xy=_as_numpy_float32(line_xy),
            control_point_xy=_as_numpy_float32(cp_xy),
        )
        return FiberStripTtaPatch(
            sample=tta_sample,
            image=image,
            valid_mask=valid_mask,
            source_xy_grid=self._finite_source_xy_grid(transform, source_shape_hw),
            reference_to_tta_xy_grid=self._finite_reference_to_output_xy_grid(transform, output_shape),
            base_corners_xy=corners_xy,
        )

    def build_trace2cp_segment_source(
        self,
        sample_index: int,
        *,
        target_control_point_index: int | None = None,
        target_offset: int = 1,
        rf_margin_px: float = 0.0,
        strip_z_offset: float | None = None,
        row_axis_alignment_line_index: int | None = None,
        row_axis_alignment_xyz: np.ndarray | None = None,
        device: torch.device | None = None,
        sample_mode: str = "random",
    ) -> _Trace2CpSegmentSource:
        resolved_device = resolve_torch_device(self.config.augment.device) if device is None else device
        record, record_index, start_cp_index = self.descriptor_for_sample_index(
            sample_index,
            sample_mode=sample_mode,
        )
        cp_count = int(record.fiber.control_points_xyz.shape[0])
        if target_control_point_index is None:
            target_cp_index = int(start_cp_index) + int(target_offset)
        else:
            target_cp_index = int(target_control_point_index)
        if target_cp_index < 0 or target_cp_index >= cp_count:
            raise ValueError(
                f"trace2cp target_control_point_index {target_cp_index} out of range for {cp_count} control points"
            )
        if target_cp_index == int(start_cp_index):
            raise ValueError("trace2cp target control point must differ from start control point")

        line_points = np.asarray(record.fiber.line_points_xyz, dtype=np.float64)
        cumulative = self._line_arc_lengths(line_points)
        start_line_index = control_point_line_index(record.fiber, int(start_cp_index))
        target_line_index = control_point_line_index(record.fiber, target_cp_index)
        start_arc = float(cumulative[start_line_index])
        target_arc = float(cumulative[target_line_index])
        signed_distance_px = (target_arc - start_arc) / float(record.volume_spacing_base)
        distance_px = abs(float(signed_distance_px))
        margin = max(float(rf_margin_px), 0.0) + 4.0
        width = max(
            int(self.config.patch_shape_hw[1]),
            int(math.ceil(distance_px + 2.0 * margin + 1.0)),
        )
        trace2cp_height_multiplier = 8
        height = int(self.config.patch_shape_hw[0]) * trace2cp_height_multiplier
        if height <= 0 or width <= 0:
            raise ValueError(f"invalid trace2cp segment shape {(height, width)}")
        start_col = margin if signed_distance_px >= 0.0 else float(width - 1) - margin
        target_col = start_col + float(signed_distance_px)
        if target_col < 0.0 or target_col > float(width - 1):
            raise ValueError(
                "trace2cp target column is outside the generated strip: "
                f"target_col={target_col:.3f} width={width} start_col={start_col:.3f}"
            )

        line_window = side_strip_segment_line_window(
            record.fiber,
            start_control_point_index=int(start_cp_index),
            target_control_point_index=target_cp_index,
            margin_px=margin,
            pixel_spacing_base=record.volume_spacing_base,
        )
        line_xy = source_line_xy_from_line_window(
            line_window,
            patch_shape_hw=(height, width),
            anchor_column_px=start_col,
            pixel_spacing_base=record.volume_spacing_base,
        )
        start_xy = source_point_xy_for_line_index(
            line_window,
            original_line_index=start_line_index,
            patch_shape_hw=(height, width),
            anchor_column_px=start_col,
            pixel_spacing_base=record.volume_spacing_base,
        )
        target_xy = source_point_xy_for_line_index(
            line_window,
            original_line_index=target_line_index,
            patch_shape_hw=(height, width),
            anchor_column_px=start_col,
            pixel_spacing_base=record.volume_spacing_base,
        )
        sampled_normals = self._lasagna_normals_for_line_window(
            record,
            line_window,
            control_point_index=int(start_cp_index),
        )
        sampled_normals = self._prealign_lasagna_normals_to_row_axis_reference(
            sampled_normals,
            line_window,
            row_axis_alignment_line_index=row_axis_alignment_line_index,
            row_axis_alignment_xyz=row_axis_alignment_xyz,
        )
        center_offset = (
            min(self.strip_z_offsets, key=lambda value: abs(float(value)))
            if strip_z_offset is None
            else float(strip_z_offset)
        )
        def build_grid(normals: np.ndarray) -> FiberStripGridTorch:
            return build_side_strip_patch_grid_tensor_from_line_window(
                line_window,
                patch_shape_hw=(height, width),
                strip_z_offset=float(center_offset),
                sampled_normals=normals,
                pixel_spacing_base=record.volume_spacing_base,
                anchor_column_px=start_col,
                device=resolved_device,
            )

        grid = build_grid(sampled_normals)
        if (
            row_axis_alignment_line_index is not None
            and row_axis_alignment_xyz is not None
            and not self._grid_row_axis_aligns_reference(
                grid,
                line_window,
                line_index=int(row_axis_alignment_line_index),
                reference_xyz=row_axis_alignment_xyz,
                patch_shape_hw=(height, width),
                anchor_column_px=start_col,
                pixel_spacing_base=record.volume_spacing_base,
            )
        ):
            sampled_normals = (-np.asarray(sampled_normals, dtype=np.float32)).astype(np.float32, copy=False)
            grid = build_grid(sampled_normals)
        start_row_axis = self._row_axis_at_xy(grid, start_xy)
        target_row_axis = self._row_axis_at_xy(grid, target_xy)
        return _Trace2CpSegmentSource(
            record=record,
            record_index=int(record_index),
            start_control_point_index=int(start_cp_index),
            target_control_point_index=int(target_cp_index),
            center_offset=float(center_offset),
            source_shape_hw=(height, width),
            grid=grid,
            line_window=line_window,
            anchor_column_px=float(start_col),
            line_xy=np.asarray(line_xy, dtype=np.float32),
            start_control_point_xy=start_xy.astype(np.float32, copy=False),
            target_control_point_xy=target_xy.astype(np.float32, copy=False),
            line_point_indices=np.asarray(line_window.original_line_indices, dtype=np.int64),
            line_normals_xyz=np.asarray(sampled_normals, dtype=np.float32),
            start_row_axis_xyz=start_row_axis,
            target_row_axis_xyz=target_row_axis,
        )

    def build_trace2cp_refined_segment_source(
        self,
        source: _Trace2CpSegmentSource,
        trace_xyz: np.ndarray,
        *,
        strip_z_offset: float | None = None,
        device: torch.device | None = None,
    ) -> _Trace2CpSegmentSource:
        resolved_device = (
            resolve_torch_device(self.config.augment.device) if device is None else device
        )
        trace = np.asarray(trace_xyz, dtype=np.float32)
        if trace.ndim != 2 or trace.shape[0] < 2 or trace.shape[1] not in (2, 3):
            raise ValueError("refined trace must have shape [N,2] or [N,3] with N >= 2")
        if trace.shape[1] == 2:
            trace = np.concatenate(
                [trace, np.zeros((int(trace.shape[0]), 1), dtype=np.float32)],
                axis=1,
            )
        if not bool(np.isfinite(trace).all()):
            raise ValueError("refined trace contains non-finite coordinates")
        if source.grid.offset_axis_xyz is None or source.grid.side_axis_xyz is None:
            raise ValueError("refined Trace2CP source requires row-axis and side-axis data")

        previous_start = np.asarray(source.start_control_point_xy, dtype=np.float32)
        previous_target = np.asarray(source.target_control_point_xy, dtype=np.float32)
        previous_height, previous_width = (int(v) for v in source.source_shape_hw)
        target_is_right = float(previous_target[0]) >= float(previous_start[0])
        if target_is_right:
            before_start_margin = max(0.0, float(previous_start[0]))
            after_target_margin = max(0.0, float(previous_width - 1) - float(previous_target[0]))
        else:
            before_start_margin = max(0.0, float(previous_width - 1) - float(previous_start[0]))
            after_target_margin = max(0.0, float(previous_target[0]))

        def extension_length(available: float) -> float:
            available_f = max(0.0, float(available))
            if available_f <= 1.0e-3:
                return 0.0
            preferred = max(2.0, min(8.0, available_f))
            return max(0.0, min(preferred, available_f - 1.0e-3))

        points_xy = torch.as_tensor(
            trace[:, :2],
            dtype=torch.float32,
            device=source.grid.coords_xyz.device,
        )
        base_xyz_t, base_valid_t = _sample_grid_points_hwc(
            source.grid.coords_xyz,
            source.grid.valid_mask,
            points_xy,
        )
        row_axis_xyz_t, row_axis_valid_t = _sample_grid_points_hwc(
            source.grid.offset_axis_xyz,
            source.grid.valid_mask,
            points_xy,
        )
        side_axis_xyz_t, side_axis_valid_t = _sample_grid_points_hwc(
            source.grid.side_axis_xyz,
            source.grid.valid_mask,
            points_xy,
        )
        row_axis_xyz_t = F.normalize(row_axis_xyz_t, p=2.0, dim=1, eps=1.0e-12)
        side_axis_xyz_t = F.normalize(side_axis_xyz_t, p=2.0, dim=1, eps=1.0e-12)
        z_offsets_t = torch.as_tensor(
            trace[:, 2],
            dtype=torch.float32,
            device=source.grid.coords_xyz.device,
        )
        center_xyz_t = base_xyz_t + side_axis_xyz_t * (
            z_offsets_t * np.float32(source.record.volume_spacing_base)
        )[:, None]
        valid_t = (
            base_valid_t
            & row_axis_valid_t
            & side_axis_valid_t
            & torch.isfinite(center_xyz_t).all(dim=1)
            & torch.isfinite(row_axis_xyz_t).all(dim=1)
            & torch.isfinite(side_axis_xyz_t).all(dim=1)
        )
        valid = valid_t.detach().cpu().numpy().astype(bool, copy=False)
        if not bool(np.all(valid)):
            bad = np.flatnonzero(~valid)
            first = int(bad[0]) if bad.size else -1
            raise ValueError(
                "refined Trace2CP trace leaves the source strip valid area: "
                f"invalid_points={int(bad.size)} first_invalid={first}"
            )
        center_xyz = center_xyz_t.detach().cpu().numpy().astype(np.float32, copy=False)
        sampled_normals = row_axis_xyz_t.detach().cpu().numpy().astype(np.float32, copy=False)

        keep = [0]
        for idx in range(1, int(center_xyz.shape[0])):
            if float(np.linalg.norm(center_xyz[idx] - center_xyz[keep[-1]])) > 1.0e-4:
                keep.append(idx)
        if len(keep) < 2:
            raise ValueError("refined Trace2CP trace degenerates to fewer than two volume points")
        keep_array = np.asarray(keep, dtype=np.int64)
        kept_start_matches = np.flatnonzero(keep_array == 0)
        kept_target_matches = np.flatnonzero(keep_array == int(trace.shape[0]) - 1)
        if kept_start_matches.size == 0 or kept_target_matches.size == 0:
            raise ValueError("refined Trace2CP trace lost a CP endpoint while removing duplicate points")
        kept_start_index = int(kept_start_matches[0])
        kept_target_index = int(kept_target_matches[0])
        center_xyz = center_xyz[keep_array]
        sampled_normals = sampled_normals[keep_array]

        def extrapolated_volume_endpoint(
            anchor: np.ndarray,
            neighbor: np.ndarray,
            length_px: float,
        ) -> np.ndarray | None:
            if length_px <= 1.0e-6:
                return None
            tangent = np.asarray(anchor, dtype=np.float32) - np.asarray(neighbor, dtype=np.float32)
            norm = float(np.linalg.norm(tangent))
            if not np.isfinite(norm) or norm <= 1.0e-6:
                return None
            length_base = np.float32(length_px * float(source.record.volume_spacing_base))
            return (
                np.asarray(anchor, dtype=np.float32)
                + tangent * np.float32(length_base / norm)
            ).astype(np.float32)

        start_extension = extrapolated_volume_endpoint(
            center_xyz[kept_start_index],
            center_xyz[min(int(center_xyz.shape[0]) - 1, kept_start_index + 1)],
            extension_length(before_start_margin),
        )
        if start_extension is not None:
            center_xyz = np.concatenate([start_extension[None, :], center_xyz], axis=0)
            sampled_normals = np.concatenate(
                [sampled_normals[kept_start_index][None, :], sampled_normals],
                axis=0,
            )
            kept_start_index += 1
            kept_target_index += 1

        target_extension = extrapolated_volume_endpoint(
            center_xyz[kept_target_index],
            center_xyz[max(0, kept_target_index - 1)],
            extension_length(after_target_margin),
        )
        if target_extension is not None:
            center_xyz = np.concatenate([center_xyz, target_extension[None, :]], axis=0)
            sampled_normals = np.concatenate(
                [sampled_normals, sampled_normals[kept_target_index][None, :]],
                axis=0,
            )

        if target_is_right:
            ordered_points = center_xyz
            ordered_normals = sampled_normals
            local_start = kept_start_index
            local_target = kept_target_index
            anchor_column = before_start_margin
        else:
            ordered_points = center_xyz[::-1].copy()
            ordered_normals = sampled_normals[::-1].copy()
            local_start = int(center_xyz.shape[0]) - 1 - kept_start_index
            local_target = int(center_xyz.shape[0]) - 1 - kept_target_index
            anchor_column = 0.0
        anchor_column = float(anchor_column)
        cumulative = self._line_arc_lengths(ordered_points)
        distance_px = abs(float(cumulative[int(local_target)] - cumulative[int(local_start)])) / float(
            source.record.volume_spacing_base
        )
        width = max(
            previous_width,
            int(math.ceil(distance_px + before_start_margin + after_target_margin + 1.0)),
        )
        height = previous_height
        if not target_is_right:
            anchor_column = float(width - 1) - before_start_margin
        original_indices = np.arange(int(ordered_points.shape[0]), dtype=np.int64)
        line_window = FiberStripLineWindow(
            line_points_xyz=np.asarray(ordered_points, dtype=np.float64),
            original_line_indices=original_indices,
            local_control_point_index=int(local_start),
        )
        line_xy = source_line_xy_from_line_window(
            line_window,
            patch_shape_hw=(height, width),
            anchor_column_px=anchor_column,
            pixel_spacing_base=source.record.volume_spacing_base,
        )
        start_xy = np.asarray(line_xy[int(local_start)], dtype=np.float32)
        target_xy = np.asarray(line_xy[int(local_target)], dtype=np.float32)
        center_offset = (
            float(source.center_offset) if strip_z_offset is None else float(strip_z_offset)
        )
        grid = build_side_strip_patch_grid_tensor_from_line_window(
            line_window,
            patch_shape_hw=(height, width),
            strip_z_offset=center_offset,
            sampled_normals=ordered_normals,
            pixel_spacing_base=source.record.volume_spacing_base,
            anchor_column_px=anchor_column,
            device=resolved_device,
        )
        start_row_axis = self._row_axis_at_xy(grid, start_xy)
        target_row_axis = self._row_axis_at_xy(grid, target_xy)
        return _Trace2CpSegmentSource(
            record=source.record,
            record_index=int(source.record_index),
            start_control_point_index=int(source.start_control_point_index),
            target_control_point_index=int(source.target_control_point_index),
            center_offset=float(center_offset),
            source_shape_hw=(height, width),
            grid=grid,
            line_window=line_window,
            anchor_column_px=float(anchor_column),
            line_xy=np.asarray(line_xy, dtype=np.float32),
            start_control_point_xy=start_xy.astype(np.float32, copy=False),
            target_control_point_xy=target_xy.astype(np.float32, copy=False),
            line_point_indices=original_indices,
            line_normals_xyz=np.asarray(ordered_normals, dtype=np.float32),
            start_row_axis_xyz=start_row_axis,
            target_row_axis_xyz=target_row_axis,
        )

    def sample_trace2cp_top_strip_source(
        self,
        source: _Trace2CpSegmentSource,
        *,
        normal_offsets_by_column: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        coords_xyz, grid_valid = self.trace2cp_top_strip_coords_xyz(
            source,
            normal_offsets_by_column=normal_offsets_by_column,
        )
        coords_zyx = coords_xyz[..., (2, 1, 0)].astype(np.float32, copy=False)
        result = source.record.sampler.sample_coords(coords_zyx, grid_valid)
        image = np.asarray(result.image, dtype=np.float32)
        valid_mask = np.asarray(result.valid_mask, dtype=bool)
        return image, valid_mask

    def trace2cp_segment_coords_xyz(
        self,
        source: _Trace2CpSegmentSource,
        *,
        strip_z_offset: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        offset = float(source.center_offset) if strip_z_offset is None else float(strip_z_offset)
        grid = self._offset_grid_from_source(source, offset)
        coords_xyz = _as_numpy_float32(grid.coords_xyz)
        grid_valid = _as_numpy_bool(grid.valid_mask)
        return coords_xyz, grid_valid

    def trace2cp_segment_side_z_coords_xyz(
        self,
        source: _Trace2CpSegmentSource,
        *,
        side_z_offset_voxels: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        grid = self._trace2cp_side_z_grid_from_source(source, float(side_z_offset_voxels))
        coords_xyz = _as_numpy_float32(grid.coords_xyz)
        grid_valid = _as_numpy_bool(grid.valid_mask)
        return coords_xyz, grid_valid

    def trace2cp_top_strip_coords_xyz(
        self,
        source: _Trace2CpSegmentSource,
        *,
        normal_offsets_by_column: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        grid = build_top_strip_patch_grid_tensor_from_line_window(
            source.line_window,
            patch_shape_hw=source.source_shape_hw,
            sampled_normals=np.asarray(source.line_normals_xyz, dtype=np.float32),
            pixel_spacing_base=source.record.volume_spacing_base,
            anchor_column_px=float(source.anchor_column_px),
            normal_offsets_by_column=normal_offsets_by_column,
            device=source.grid.coords_zyx.device,
        )
        coords_xyz = _as_numpy_float32(grid.coords_xyz)
        grid_valid = _as_numpy_bool(grid.valid_mask)
        return coords_xyz, grid_valid

    def sample_trace2cp_traced_top_strip_source(
        self,
        source: _Trace2CpSegmentSource,
        trace_xyz: np.ndarray,
        *,
        top_offsets_by_column: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        coords_xyz, grid_valid = self.trace2cp_traced_top_strip_coords_xyz(
            source,
            trace_xyz,
            top_offsets_by_column=top_offsets_by_column,
        )
        coords_zyx = coords_xyz[..., (2, 1, 0)].astype(np.float32, copy=False)
        result = source.record.sampler.sample_coords(coords_zyx, grid_valid)
        image = np.asarray(result.image, dtype=np.float32)
        valid_mask = np.asarray(result.valid_mask, dtype=bool)
        return image, valid_mask

    def trace2cp_traced_top_strip_coords_xyz(
        self,
        source: _Trace2CpSegmentSource,
        trace_xyz: np.ndarray,
        *,
        top_offsets_by_column: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        height, width = (int(v) for v in source.source_shape_hw)
        columns_xyz, columns_valid = _trace_columns_xyz_for_top_strip(trace_xyz, width)
        if top_offsets_by_column is None:
            top_offsets = np.zeros((width,), dtype=np.float32)
        else:
            top_offsets = np.asarray(top_offsets_by_column, dtype=np.float32).reshape(-1)
            if int(top_offsets.shape[0]) != width:
                raise ValueError(
                    "top_offsets_by_column length must match trace2cp strip width: "
                    f"got {int(top_offsets.shape[0])}, expected {width}"
                )
        if source.grid.offset_axis_xyz is None or source.grid.side_axis_xyz is None:
            raise ValueError("trace2cp traced top strip requires source row-axis and side-axis data")
        device = source.grid.coords_xyz.device
        points_xy = torch.as_tensor(columns_xyz[:, :2], dtype=torch.float32, device=device)
        base_xyz_t, base_valid_t = _sample_grid_points_hwc(
            source.grid.coords_xyz,
            source.grid.valid_mask,
            points_xy,
        )
        normal_xyz_t, normal_valid_t = _sample_grid_points_hwc(
            source.grid.offset_axis_xyz,
            source.grid.valid_mask,
            points_xy,
        )
        side_z_axis_t, side_z_valid_t = _sample_grid_points_hwc(
            source.grid.side_axis_xyz,
            source.grid.valid_mask,
            points_xy,
        )
        columns_valid_t = torch.as_tensor(columns_valid, dtype=torch.bool, device=device)
        z_offsets_t = torch.as_tensor(columns_xyz[:, 2], dtype=torch.float32, device=device)
        normal_xyz_t = F.normalize(normal_xyz_t, p=2.0, dim=1, eps=1.0e-12)
        side_z_axis_t = F.normalize(side_z_axis_t, p=2.0, dim=1, eps=1.0e-12)
        base_xyz_t = base_xyz_t + side_z_axis_t * (
            z_offsets_t * np.float32(source.record.volume_spacing_base)
        )[:, None]
        base_xyz = base_xyz_t.detach().cpu().numpy().astype(np.float32, copy=False)
        normal_xyz = normal_xyz_t.detach().cpu().numpy().astype(np.float32, copy=False)
        valid_columns = (
            columns_valid_t & base_valid_t & normal_valid_t & torch.isfinite(base_xyz_t).all(dim=1)
            & side_z_valid_t
        ).detach().cpu().numpy().astype(bool, copy=False)
        valid_columns &= np.isfinite(top_offsets)

        side_axes = np.zeros((width, 3), dtype=np.float32)
        for col in range(width):
            if not bool(valid_columns[col]):
                continue
            prev_col = col - 1
            while prev_col >= 0 and not bool(valid_columns[prev_col]):
                prev_col -= 1
            next_col = col + 1
            while next_col < width and not bool(valid_columns[next_col]):
                next_col += 1
            if prev_col >= 0 and next_col < width:
                tangent = base_xyz[next_col] - base_xyz[prev_col]
            elif next_col < width:
                tangent = base_xyz[next_col] - base_xyz[col]
            elif prev_col >= 0:
                tangent = base_xyz[col] - base_xyz[prev_col]
            else:
                continue
            tangent = _unit_vector_or_zero(tangent)
            normal = _unit_vector_or_zero(normal_xyz[col])
            side = _unit_vector_or_zero(np.cross(normal, tangent))
            if not bool(np.any(side)):
                valid_columns[col] = False
                continue
            side_axes[col] = side

        row_offsets = (
            np.arange(height, dtype=np.float32)[:, None]
            - np.float32((float(height) - 1.0) * 0.5)
            + top_offsets[None, :]
        ) * np.float32(source.record.volume_spacing_base)
        coords_xyz = base_xyz[None, :, :] + row_offsets[:, :, None] * side_axes[None, :, :]
        grid_valid = (
            valid_columns[None, :]
            & np.isfinite(coords_xyz).all(axis=2)
            & np.isfinite(side_axes).all(axis=1)[None, :]
        )
        return coords_xyz.astype(np.float32, copy=False), grid_valid.astype(bool, copy=False)

    def sample_trace2cp_segment_source(
        self,
        source: _Trace2CpSegmentSource,
        *,
        strip_z_offset: float | None = None,
    ) -> tuple[FiberStripSegmentSample, np.ndarray, np.ndarray]:
        offset = float(source.center_offset) if strip_z_offset is None else float(strip_z_offset)
        grid = self._offset_grid_from_source(source, offset)
        coords_zyx = _as_numpy_float32(grid.coords_zyx)
        grid_valid = _as_numpy_bool(grid.valid_mask)
        result = source.record.sampler.sample_coords(coords_zyx, grid_valid)
        image = np.asarray(result.image, dtype=np.float32)
        valid_mask = np.asarray(result.valid_mask, dtype=bool)
        sample = FiberStripSegmentSample(
            record_index=int(source.record_index),
            fiber_path=str(source.record.fiber.path) if source.record.fiber.path is not None else "",
            start_control_point_index=int(source.start_control_point_index),
            target_control_point_index=int(source.target_control_point_index),
            start_control_point_xyz=np.asarray(
                source.record.fiber.control_points_xyz[int(source.start_control_point_index)], dtype=np.float32
            ),
            target_control_point_xyz=np.asarray(
                source.record.fiber.control_points_xyz[int(source.target_control_point_index)], dtype=np.float32
            ),
            strip_z_offset=offset,
            coords_zyx=coords_zyx,
            valid_mask=valid_mask,
            frame=grid.frame,
            line_xy=np.asarray(source.line_xy, dtype=np.float32),
            start_control_point_xy=np.asarray(source.start_control_point_xy, dtype=np.float32),
            target_control_point_xy=np.asarray(source.target_control_point_xy, dtype=np.float32),
            line_point_indices=np.asarray(source.line_point_indices, dtype=np.int64),
            line_normals_xyz=np.asarray(source.line_normals_xyz, dtype=np.float32),
            start_row_axis_xyz=np.asarray(source.start_row_axis_xyz, dtype=np.float32),
            target_row_axis_xyz=np.asarray(source.target_row_axis_xyz, dtype=np.float32),
        )
        return sample, image, valid_mask

    def sample_trace2cp_segment_side_z_source(
        self,
        source: _Trace2CpSegmentSource,
        *,
        side_z_offset_voxels: float = 0.0,
    ) -> tuple[FiberStripSegmentSample, np.ndarray, np.ndarray]:
        grid = self._trace2cp_side_z_grid_from_source(source, float(side_z_offset_voxels))
        coords_zyx = _as_numpy_float32(grid.coords_zyx)
        grid_valid = _as_numpy_bool(grid.valid_mask)
        result = source.record.sampler.sample_coords(coords_zyx, grid_valid)
        image = np.asarray(result.image, dtype=np.float32)
        valid_mask = np.asarray(result.valid_mask, dtype=bool)
        sample = FiberStripSegmentSample(
            record_index=int(source.record_index),
            fiber_path=str(source.record.fiber.path) if source.record.fiber.path is not None else "",
            start_control_point_index=int(source.start_control_point_index),
            target_control_point_index=int(source.target_control_point_index),
            start_control_point_xyz=np.asarray(
                source.record.fiber.control_points_xyz[int(source.start_control_point_index)], dtype=np.float32
            ),
            target_control_point_xyz=np.asarray(
                source.record.fiber.control_points_xyz[int(source.target_control_point_index)], dtype=np.float32
            ),
            strip_z_offset=float(source.center_offset),
            coords_zyx=coords_zyx,
            valid_mask=valid_mask,
            frame=grid.frame,
            line_xy=np.asarray(source.line_xy, dtype=np.float32),
            start_control_point_xy=np.asarray(source.start_control_point_xy, dtype=np.float32),
            target_control_point_xy=np.asarray(source.target_control_point_xy, dtype=np.float32),
            line_point_indices=np.asarray(source.line_point_indices, dtype=np.int64),
            line_normals_xyz=np.asarray(source.line_normals_xyz, dtype=np.float32),
            start_row_axis_xyz=np.asarray(source.start_row_axis_xyz, dtype=np.float32),
            target_row_axis_xyz=np.asarray(source.target_row_axis_xyz, dtype=np.float32),
        )
        return sample, image, valid_mask

    def build_trace2cp_segment_patch(
        self,
        sample_index: int,
        *,
        target_control_point_index: int | None = None,
        target_offset: int = 1,
        rf_margin_px: float = 0.0,
        strip_z_offset: float | None = None,
        row_axis_alignment_line_index: int | None = None,
        row_axis_alignment_xyz: np.ndarray | None = None,
        device: torch.device | None = None,
        sample_mode: str = "random",
    ) -> tuple[FiberStripSegmentSample, np.ndarray, np.ndarray]:
        source = self.build_trace2cp_segment_source(
            sample_index,
            target_control_point_index=target_control_point_index,
            target_offset=target_offset,
            rf_margin_px=rf_margin_px,
            row_axis_alignment_line_index=row_axis_alignment_line_index,
            row_axis_alignment_xyz=row_axis_alignment_xyz,
            device=device,
            sample_mode=sample_mode,
        )
        return self.sample_trace2cp_segment_source(source, strip_z_offset=strip_z_offset)

    def build_trace2cp_tta_patch_from_sample(
        self,
        sample: FiberStripSegmentSample,
        params: FiberStripAugmentParams,
        *,
        rf_margin_px: float = 0.0,
        device: torch.device | None = None,
    ) -> FiberStripTtaPatch:
        resolved_device = resolve_torch_device(self.config.augment.device) if device is None else device
        record = self.records[int(sample.record_index)]
        source_shape_hw = tuple(int(v) for v in np.asarray(sample.coords_zyx).shape[:2])
        output_shape, _, transform, corners_xy = self._tta_transform_for_source_shape(
            source_shape_hw,
            params,
            rf_margin_px=rf_margin_px,
            device=resolved_device,
        )
        coords_t, valid_t = _resample_coord_tensors_like_augmentation(
            torch.as_tensor(sample.coords_zyx, dtype=torch.float32, device=resolved_device),
            torch.as_tensor(sample.valid_mask, dtype=torch.bool, device=resolved_device),
            params,
            output_shape_hw=output_shape,
            device=resolved_device,
            transform=transform,
        )
        coords_zyx = _as_numpy_float32(coords_t)
        coord_valid = _as_numpy_bool(valid_t)
        result = record.sampler.sample_coords(coords_zyx, coord_valid)
        image = np.asarray(result.image, dtype=np.float32)
        valid_mask = np.asarray(result.valid_mask, dtype=bool)
        source_line = torch.as_tensor(sample.line_xy, dtype=torch.float32, device=resolved_device)
        start = torch.as_tensor(sample.start_control_point_xy, dtype=torch.float32, device=resolved_device).view(1, 2)
        target = torch.as_tensor(sample.target_control_point_xy, dtype=torch.float32, device=resolved_device).view(1, 2)
        mapped = transform.source_to_output_points(torch.cat([source_line, start, target], dim=0))
        line_xy = self._filter_line_xy_t(mapped[:-2], output_shape)
        start_xy = mapped[-2]
        target_xy = mapped[-1]
        tta_sample = FiberStripSegmentSample(
            record_index=int(sample.record_index),
            fiber_path=sample.fiber_path,
            start_control_point_index=int(sample.start_control_point_index),
            target_control_point_index=int(sample.target_control_point_index),
            start_control_point_xyz=np.asarray(sample.start_control_point_xyz, dtype=np.float32),
            target_control_point_xyz=np.asarray(sample.target_control_point_xyz, dtype=np.float32),
            strip_z_offset=float(sample.strip_z_offset),
            coords_zyx=coords_zyx,
            valid_mask=valid_mask,
            frame=sample.frame,
            line_xy=_as_numpy_float32(line_xy),
            start_control_point_xy=_as_numpy_float32(start_xy),
            target_control_point_xy=_as_numpy_float32(target_xy),
            line_point_indices=np.asarray(sample.line_point_indices, dtype=np.int64),
            line_normals_xyz=np.asarray(sample.line_normals_xyz, dtype=np.float32),
            start_row_axis_xyz=np.asarray(sample.start_row_axis_xyz, dtype=np.float32),
            target_row_axis_xyz=np.asarray(sample.target_row_axis_xyz, dtype=np.float32),
        )
        return FiberStripTtaPatch(
            sample=tta_sample,
            image=image,
            valid_mask=valid_mask,
            source_xy_grid=self._finite_source_xy_grid(transform, source_shape_hw),
            reference_to_tta_xy_grid=self._finite_reference_to_output_xy_grid(transform, output_shape),
            base_corners_xy=corners_xy,
        )

    def build_augmented_center_strip_source(
        self,
        sample_index: int,
        *,
        device: torch.device,
        profile: dict[str, float] | None = None,
    ) -> _StripSource:
        return self.build_strip_source(
            sample_index,
            device=device,
            profile=profile,
            use_augmentation_envelope=True,
        )

    def build_augmented_center_strip_patch(
        self,
        sample_index: int,
        params: FiberStripAugmentParams,
        *,
        device: torch.device,
        profile: dict[str, float] | None = None,
        source: _StripSource | None = None,
    ) -> tuple[FiberStripSample, np.ndarray, np.ndarray, np.ndarray]:
        if source is None:
            source = self.build_augmented_center_strip_source(sample_index, device=device, profile=profile)
        center_index = min(range(len(self.strip_z_offsets)), key=lambda index: abs(float(self.strip_z_offsets[index])))
        sample, image, valid_mask, line_xy = self.build_strip_patch_from_source(
            source,
            center_index,
            params,
            device=device,
            profile=profile,
            load_image=True,
        )
        assert image is not None
        assert valid_mask is not None
        return sample, image.astype(np.float32, copy=False), valid_mask.astype(bool, copy=False), line_xy

    def load_batch(
        self,
        start_sample_index: int = 0,
        batch_size: int | None = None,
        *,
        sample_mode: str = "random",
        sample_index_limit: int | None = None,
        flat_indices: tuple[int, ...] | list[int] | None = None,
        augmentation_sample_start_index: int | None = None,
        value_augmentation_sync_group_size: int | None = None,
        profile: dict[str, float] | None = None,
        apply_image_augmentation: bool = True,
        include_line_xy: bool = True,
        include_coords: bool = True,
    ) -> FiberStrip2DBatch:
        batch_size = self.config.batch_size if batch_size is None else int(batch_size)
        flat_index_sequence = None if flat_indices is None else tuple(int(v) for v in flat_indices)
        if flat_index_sequence is not None and not flat_index_sequence:
            raise ValueError("flat_indices must be non-empty when provided")
        if flat_index_sequence is not None and batch_size > len(flat_index_sequence):
            raise ValueError("batch_size cannot exceed flat_indices length")
        aug_start_index = (
            int(start_sample_index)
            if augmentation_sample_start_index is None
            else int(augmentation_sample_start_index)
        )
        value_sync_size = (
            None
            if value_augmentation_sync_group_size is None
            else max(1, int(value_augmentation_sync_group_size))
        )
        begin_zarr_cache_trace()
        all_samples: list[FiberStripSample] = []
        images: list[np.ndarray] = []
        coords: list[np.ndarray] = []
        valids: list[np.ndarray] = []
        record_indices: list[int] = []
        cp_indices: list[int] = []
        fiber_paths: list[str] = []
        augmentation_params: list[FiberStripAugmentParams | None] = []
        batch_wall_start = time.perf_counter()

        def report_skip(current_sample_index: int, exc: ValueError) -> None:
            self._load_batch_skipped_samples += 1

        def build_candidate(raw_sample_index: int, attempt_index: int) -> tuple[
            int,
            int,
            dict[str, float],
            _PreparedStripSample | None,
            ValueError | None,
        ]:
            current_sample_index = self._bounded_sample_index(raw_sample_index, sample_index_limit)
            descriptor = None
            if flat_index_sequence is not None:
                current_sample_index = int(flat_index_sequence[int(attempt_index)])
                descriptor = self._descriptor_for_flat_index(current_sample_index)
            value_augmentation_sample_index = (
                None
                if value_sync_size is None
                else math.floor(int(raw_sample_index) / value_sync_size)
            )
            local_profile: dict[str, float] = {} if profile is not None else {}
            worker_start = time.perf_counter()
            try:
                try:
                    result = self._prepare_sample(
                        current_sample_index,
                        sample_mode=sample_mode,
                        descriptor=descriptor,
                        augmentation_sample_index=raw_sample_index,
                        value_augmentation_sample_index=value_augmentation_sample_index,
                        profile=local_profile if profile is not None else None,
                        include_line_xy=include_line_xy,
                    )
                except TypeError as exc:
                    message = str(exc)
                    if (
                        "include_line_xy" not in message
                        and "descriptor" not in message
                        and "augmentation_sample_index" not in message
                        and "value_augmentation_sample_index" not in message
                    ):
                        raise
                    result = self._prepare_sample(
                        current_sample_index,
                        sample_mode=sample_mode,
                        profile=local_profile if profile is not None else None,
                    )
            except ValueError as exc:
                if profile is not None:
                    local_profile["load_batch_worker"] = (
                        time.perf_counter() - worker_start
                    ) * 1000.0
                return raw_sample_index, current_sample_index, local_profile, None, exc
            if profile is not None:
                local_profile["load_batch_worker"] = (time.perf_counter() - worker_start) * 1000.0
            return raw_sample_index, current_sample_index, local_profile, result, None

        try:
            next_attempt_to_submit = 0
            next_attempt_to_consume = 0
            max_attempts = (
                len(flat_index_sequence)
                if flat_index_sequence is not None
                else max(batch_size * 4, batch_size + 1000)
            )
            attempts = 0
            worker_count = min(max(1, int(self.config.loader_workers)), max(1, batch_size), max_attempts)
            if sample_mode == "random" and flat_index_sequence is None:
                with _ProfileBlock(profile, "random_order"):
                    self._ensure_random_pass_orders_for_indices(
                        self._bounded_sample_index(raw, sample_index_limit)
                        for raw in range(int(start_sample_index), int(start_sample_index) + max_attempts)
                    )

            prepared_items: list[tuple[_PreparedStripSample, dict[str, float]]] = []

            def consume_candidate(
                current_sample_index: int,
                local_profile: dict[str, float],
                result: _PreparedStripSample | None,
                skip_exc: ValueError | None,
            ) -> None:
                if skip_exc is not None:
                    report_skip(current_sample_index, skip_exc)
                    _merge_profile(profile, local_profile)
                    return
                if result is None:
                    raise RuntimeError("internal load_batch missing sample result")
                prepared_items.append((result, local_profile))

            def consume_finished(
                local_profile: dict[str, float],
                result: tuple[
                    list[FiberStripSample],
                    np.ndarray,
                    np.ndarray,
                    np.ndarray,
                    tuple[FiberStripAugmentParams | None, ...],
                ],
            ) -> None:
                _merge_profile(profile, local_profile)
                (
                    sample_records,
                    sample_images,
                    sample_coords,
                    sample_valids,
                    sample_params,
                ) = result
                first = sample_records[0]
                all_samples.extend(sample_records)
                images.append(sample_images)
                coords.append(sample_coords)
                valids.append(sample_valids)
                record_indices.append(first.record_index)
                cp_indices.append(first.control_point_index)
                fiber_paths.append(first.fiber_path)
                augmentation_params.extend(sample_params)

            if worker_count == 1:
                while len(prepared_items) < batch_size and attempts < max_attempts:
                    attempt_index = next_attempt_to_submit
                    raw_sample_index = aug_start_index + attempt_index
                    next_attempt_to_submit += 1
                    attempts += 1
                    (
                        returned_raw,
                        current_sample_index,
                        local_profile,
                        result,
                        skip_exc,
                    ) = build_candidate(raw_sample_index, attempt_index)
                    if returned_raw != raw_sample_index:
                        raise RuntimeError("internal load_batch raw sample index mismatch")
                    consume_candidate(current_sample_index, local_profile, result, skip_exc)
            else:
                pending: dict[Future[Any], int] = {}
                outcomes: dict[
                    int,
                    tuple[
                        int,
                        dict[str, float],
                        _PreparedStripSample | None,
                        ValueError | None,
                    ],
                ] = {}
                executor = self._get_loader_executor(worker_count)
                while len(prepared_items) < batch_size and (pending or attempts < max_attempts):
                    needed_candidates = batch_size - len(prepared_items)
                    while (
                        len(pending) < worker_count
                        and len(pending) + len(outcomes) < needed_candidates
                        and attempts < max_attempts
                    ):
                        attempt_index = next_attempt_to_submit
                        raw_sample_index = aug_start_index + attempt_index
                        next_attempt_to_submit += 1
                        attempts += 1
                        future = executor.submit(build_candidate, raw_sample_index, attempt_index)
                        pending[future] = raw_sample_index
                    if not pending:
                        break
                    done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
                    for future in done:
                        raw_sample_index = pending.pop(future)
                        (
                            returned_raw,
                            current_sample_index,
                            local_profile,
                            result,
                            skip_exc,
                        ) = future.result()
                        if returned_raw != raw_sample_index:
                            raise RuntimeError("internal load_batch raw sample index mismatch")
                        outcomes[raw_sample_index] = (
                            current_sample_index,
                            local_profile,
                            result,
                            skip_exc,
                        )
                    while len(prepared_items) < batch_size and (aug_start_index + next_attempt_to_consume) in outcomes:
                        (
                            current_sample_index,
                            local_profile,
                            result,
                            skip_exc,
                        ) = outcomes.pop(aug_start_index + next_attempt_to_consume)
                        next_attempt_to_consume += 1
                        consume_candidate(current_sample_index, local_profile, result, skip_exc)
                for future in pending:
                    future.cancel()
            if len(prepared_items) < batch_size:
                raise ValueError(
                    "could not assemble a full fiber_trace_2d batch after "
                    f"{attempts} deterministic sample attempts; requested={batch_size} "
                    f"loaded={len(prepared_items)} skipped={attempts - len(prepared_items)}"
                )
            for local_profile, result in self._finish_prepared_batch_samples(
                prepared_items,
                apply_image_augmentation=apply_image_augmentation,
                include_coords=include_coords,
                profile=profile,
            ):
                consume_finished(local_profile, result)
        finally:
            cache_stats = end_zarr_cache_trace()
            if profile is not None:
                profile["load_batch_wall"] = profile.get("load_batch_wall", 0.0) + (
                    time.perf_counter() - batch_wall_start
                ) * 1000.0
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
            augmentation_params=tuple(augmentation_params),
        )

    def load_fiber_group_batch(
        self,
        group_index: int,
        *,
        batch_size: int,
        control_points_per_group: int,
        sample_index_limit: int | None = None,
        profile: dict[str, float] | None = None,
        apply_image_augmentation: bool = True,
        include_line_xy: bool = True,
        include_coords: bool = True,
    ) -> FiberStrip2DBatch:
        batch_size = max(1, int(batch_size))
        attempts = max(batch_size * 4, batch_size + 1000)
        group_size = max(1, int(control_points_per_group))
        groups_per_batch = max(1, int(math.ceil(batch_size / float(group_size))))
        group_span = max(1, int(math.ceil(attempts / float(batch_size))))
        flat_indices: list[int] = []
        for local_group in range(group_span):
            flat_indices.extend(
                self.fiber_group_flat_indices_for_group(
                    int(group_index) + local_group * groups_per_batch,
                    control_points_per_group=group_size,
                    batch_size=batch_size,
                    sample_index_limit=sample_index_limit,
                )
            )
        augmentation_start_index = int(group_index) * group_size
        return self.load_batch(
            augmentation_start_index,
            batch_size=batch_size,
            sample_mode="flat",
            flat_indices=tuple(flat_indices),
            augmentation_sample_start_index=augmentation_start_index,
            value_augmentation_sync_group_size=group_size,
            profile=profile,
            apply_image_augmentation=apply_image_augmentation,
            include_line_xy=include_line_xy,
            include_coords=include_coords,
        )

    def chunk_requests_for_sample_index(
        self,
        sample_index: int,
        *,
        sample_mode: str = "random",
        sample_index_limit: int | None = None,
        include_top_view: bool = False,
    ) -> list[ZarrChunkRequest]:
        device = resolve_torch_device(self.config.augment.device)
        bounded_sample_index = self._bounded_sample_index(sample_index, sample_index_limit)
        source = self.build_strip_source(bounded_sample_index, device=device, sample_mode=sample_mode)
        requests: list[ZarrChunkRequest] = []
        for offset_index, _ in enumerate(self.strip_z_offsets):
            coords_zyx, valid_mask = self._prefetch_envelope_coords_from_source(source, offset_index)
            requests.extend(source.record.sampler.chunk_requests_for_coords(coords_zyx, valid_mask))
        if include_top_view:
            top_source = self.build_top_strip_source(
                bounded_sample_index,
                device=device,
                sample_mode=sample_mode,
                use_augmentation_envelope=True,
            )
            coords_zyx = _as_numpy_float32(top_source.grid.coords_zyx)
            valid_mask = _as_numpy_bool(top_source.grid.valid_mask)
            requests.extend(top_source.record.sampler.chunk_requests_for_coords(coords_zyx, valid_mask))
        return requests

    def chunk_requests_for_samples(
        self,
        start_sample_index: int,
        sample_count: int,
        *,
        sample_mode: str = "random",
        sample_index_limit: int | None = None,
        include_top_view: bool = False,
    ) -> list[ZarrChunkRequest]:
        unique: dict[tuple[str, str], ZarrChunkRequest] = {}
        for sample_index in range(int(start_sample_index), int(start_sample_index) + int(sample_count)):
            for request in self.chunk_requests_for_sample_index(
                sample_index,
                sample_mode=sample_mode,
                sample_index_limit=sample_index_limit,
                include_top_view=include_top_view,
            ):
                unique[(request.store_identity, request.key)] = request
        return list(unique.values())

    def prefetch(
        self,
        start_sample_index: int,
        sample_count: int,
        *,
        workers: int | None = None,
        sample_mode: str = "random",
        sample_index_limit: int | None = None,
        include_top_view: bool = False,
    ) -> dict[str, Any]:
        total_samples = int(sample_count)
        total_patches = total_samples * len(self.strip_z_offsets)
        worker_count = max(1, int(self.config.prefetch_workers if workers is None else workers))
        producer_count = min(max(1, int(self.config.prefetch_sampler_workers)), max(1, total_samples))
        retry_seconds = self.config.volume_cache_retry_seconds
        if retry_seconds <= 0.0:
            retry_seconds = 600.0
        counters = _PrefetchCounters()
        start_sample = int(start_sample_index)
        seen: set[tuple[str, str]] = set()
        completed_chunks: set[tuple[str, str]] = set()
        chunk_waiters: dict[tuple[str, str], set[int]] = {}
        sample_pending: dict[int, set[tuple[str, str]]] = {}
        sample_requests_closed: set[int] = set()
        complete_raw_samples: set[int] = set()
        next_safe_raw_sample = start_sample
        start = time.perf_counter()
        print(
            "prefetch: "
            f"samples={total_samples} patches={total_patches} "
            f"workers={worker_count} sampler_workers={producer_count} "
            f"sample_mode={sample_mode} sample_index_limit={int(sample_index_limit or 0)} "
            f"include_top_view={bool(include_top_view)} "
            "mode=dependency_chunks",
            flush=True,
        )

        def print_progress(*, final: bool = False) -> None:
            elapsed = max(time.perf_counter() - start, 1.0e-9)
            sample_rate = counters.samples_done / elapsed
            remaining_samples = total_samples - counters.samples_done
            sample_eta = remaining_samples / sample_rate if remaining_samples > 0 and sample_rate > 0.0 else 0.0
            download_rate = counters.download_done / elapsed
            if counters.samples_done > 0 and counters.samples_done < total_samples:
                observed_chunks_per_sample = counters.unique_chunks_seen / max(counters.samples_done, 1)
                observed_download_ratio = counters.queued_for_download / max(counters.unique_chunks_seen, 1)
                estimated_total_downloads = counters.queued_for_download + int(
                    math.ceil(remaining_samples * observed_chunks_per_sample * observed_download_ratio)
                )
            else:
                estimated_total_downloads = counters.queued_for_download
            remaining_downloads = estimated_total_downloads - counters.download_done
            download_eta = (
                remaining_downloads / download_rate
                if remaining_downloads > 0 and download_rate > 0.0
                else 0.0
            )
            mib = counters.bytes_downloaded / (1024.0 * 1024.0)
            mib_s = mib / elapsed
            bar_width = 24
            sample_filled = int(math.floor(bar_width * counters.samples_done / max(total_samples, 1)))
            sample_bar = "#" * sample_filled + "-" * (bar_width - sample_filled)
            if counters.queued_for_download > 0:
                download_filled = int(
                    math.floor(bar_width * counters.download_done / max(counters.queued_for_download, 1))
                )
            else:
                download_filled = 0
            download_bar = "#" * download_filled + "-" * (bar_width - download_filled)
            sample_prefix = ""
            if counters.samples_done < total_samples:
                sample_prefix = (
                    f"prefetch samples[{sample_bar}] {counters.samples_done}/{total_samples} "
                    f"eta={_format_seconds(sample_eta)} "
                )
            else:
                sample_prefix = "prefetch "
            line = (
                sample_prefix
                + f"downloads[{download_bar}] {counters.download_done}/{counters.queued_for_download} "
                f"eta={_format_seconds(download_eta)} "
                f"idx={counters.max_exclusive_sample_index} "
                f"chunks={counters.unique_chunks_seen} hits={counters.cache_hits} "
                f"queued={counters.queued_download_futures} transfers={worker_count} "
                f"samplers={producer_count} "
                f"skipped={counters.samples_skipped} "
                f"missing={counters.known_missing + counters.newly_missing} "
                f"downloaded={counters.downloaded} errors={counters.download_errors} "
                f"{mib:.1f} MiB {mib_s:.1f} MiB/s"
            )
            if counters.first_error:
                error = counters.first_error
                if len(error) > 160:
                    error = error[:157] + "..."
                line += f" first_error={error!r}"
            print(line, end="\n" if final else "\r", flush=True)

        def build_requests(sample_index: int) -> tuple[int, int, list[ZarrChunkRequest]]:
            bounded_sample_index = self._bounded_sample_index(sample_index, sample_index_limit)
            source = self.build_strip_source(
                bounded_sample_index,
                device=torch.device("cpu"),
                sample_mode=sample_mode,
                use_augmentation_envelope=True,
            )
            requests: list[ZarrChunkRequest] = []
            valid_pixels = 0
            for offset_index, _ in enumerate(self.strip_z_offsets):
                coords_zyx, valid_mask = self._prefetch_envelope_coords_from_source(source, offset_index)
                valid_pixels += int(np.count_nonzero(valid_mask))
                requests.extend(source.record.sampler.chunk_requests_for_coords(coords_zyx, valid_mask))
            if include_top_view:
                top_source = self.build_top_strip_source(
                    bounded_sample_index,
                    device=torch.device("cpu"),
                    sample_mode=sample_mode,
                    use_augmentation_envelope=True,
                )
                top_coords_zyx = _as_numpy_float32(top_source.grid.coords_zyx)
                top_valid_mask = _as_numpy_bool(top_source.grid.valid_mask)
                valid_pixels += int(np.count_nonzero(top_valid_mask))
                requests.extend(top_source.record.sampler.chunk_requests_for_coords(top_coords_zyx, top_valid_mask))
            return bounded_sample_index, valid_pixels, requests

        def classify_or_submit(
            request: ZarrChunkRequest,
            raw_sample_index: int,
            download_queue: list[tuple[int, int, tuple[str, str]]],
            queued_downloads: dict[tuple[str, str], tuple[int, int, ZarrChunkRequest]],
        ) -> None:
            nonlocal download_sequence
            identity = (request.store_identity, request.key)
            if identity in completed_chunks:
                return
            sample_pending.setdefault(raw_sample_index, set()).add(identity)
            chunk_waiters.setdefault(identity, set()).add(raw_sample_index)
            if identity in seen:
                queued = queued_downloads.get(identity)
                if queued is not None and int(raw_sample_index) < queued[0]:
                    download_sequence += 1
                    queued_downloads[identity] = (int(raw_sample_index), download_sequence, queued[2])
                    heapq.heappush(download_queue, (int(raw_sample_index), download_sequence, identity))
                return
            seen.add(identity)
            counters.unique_chunks_seen += 1
            if _existing_data_path(request) is not None:
                counters.cache_hits += 1
                mark_chunk_complete(identity)
                return
            empty_path = _empty_marker_path(request)
            if empty_path is not None and empty_path.is_file():
                counters.known_missing += 1
                mark_chunk_complete(identity)
                return
            counters.queued_for_download += 1
            download_sequence += 1
            queued_downloads[identity] = (int(raw_sample_index), download_sequence, request)
            heapq.heappush(download_queue, (int(raw_sample_index), download_sequence, identity))

        def advance_safe_prefix() -> None:
            nonlocal next_safe_raw_sample
            while next_safe_raw_sample < end_sample and next_safe_raw_sample in complete_raw_samples:
                # This is the exclusive deterministic training-stream prefix.
                # The raw stream position is intentionally reported before the
                # seeded random CP permutation maps it to a flat fiber/CP id.
                counters.max_exclusive_sample_index = max(
                    counters.max_exclusive_sample_index,
                    int(next_safe_raw_sample) + 1,
                )
                next_safe_raw_sample += 1

        def mark_sample_complete(raw_sample_index: int) -> None:
            if raw_sample_index not in sample_requests_closed:
                return
            if sample_pending.get(raw_sample_index):
                return
            complete_raw_samples.add(raw_sample_index)
            advance_safe_prefix()

        def close_sample_requests(raw_sample_index: int) -> None:
            sample_requests_closed.add(raw_sample_index)
            mark_sample_complete(raw_sample_index)

        def mark_chunk_complete(identity: tuple[str, str]) -> None:
            if identity in completed_chunks:
                return
            completed_chunks.add(identity)
            for raw_sample_index in chunk_waiters.pop(identity, set()):
                pending = sample_pending.get(raw_sample_index)
                if pending is None:
                    continue
                pending.discard(identity)
                mark_sample_complete(raw_sample_index)

        def consume_download_result(request: ZarrChunkRequest, result: dict[str, Any]) -> None:
            counters.download_done += 1
            status = str(result.get("status", "error"))
            if status == "cache_hit":
                counters.cache_hits += 1
                mark_chunk_complete((request.store_identity, request.key))
            elif status == "known_missing":
                counters.known_missing += 1
                mark_chunk_complete((request.store_identity, request.key))
            elif status == "new_missing":
                counters.newly_missing += 1
                mark_chunk_complete((request.store_identity, request.key))
            elif status == "downloaded":
                counters.downloaded += 1
                counters.bytes_downloaded += int(result.get("bytes", 0))
                mark_chunk_complete((request.store_identity, request.key))
            else:
                counters.download_errors += 1
                if not counters.first_error:
                    counters.first_error = str(result.get("error", "prefetch download failed"))

        next_sample = start_sample
        end_sample = next_sample + total_samples
        last_progress = 0.0
        download_sequence = 0
        previous_torch_threads = int(torch.get_num_threads())
        torch_threads_changed = False
        if previous_torch_threads != 1:
            torch.set_num_threads(1)
            torch_threads_changed = True
        producer_executor = ThreadPoolExecutor(max_workers=producer_count)
        download_executor = ThreadPoolExecutor(max_workers=worker_count)
        normal_shutdown = False
        try:
            producer_futures: dict[Future[tuple[int, int, list[ZarrChunkRequest]]], int] = {}
            download_futures: dict[Future[dict[str, Any]], tuple[ZarrChunkRequest, float]] = {}
            queued_downloads: dict[tuple[str, str], tuple[int, int, ZarrChunkRequest]] = {}
            download_queue: list[tuple[int, int, tuple[str, str]]] = []
            producer_results: dict[int, tuple[str, Any]] = {}
            next_producer_result = start_sample

            def update_download_queue_counters() -> None:
                counters.queued_download_futures = len(download_futures) + len(queued_downloads)

            def submit_downloads() -> None:
                while len(download_futures) < worker_count:
                    request: ZarrChunkRequest | None = None
                    while download_queue:
                        priority, sequence, identity = heapq.heappop(download_queue)
                        queued = queued_downloads.get(identity)
                        if queued is None:
                            continue
                        queued_priority, queued_sequence, queued_request = queued
                        if priority != queued_priority or sequence != queued_sequence:
                            continue
                        queued_downloads.pop(identity, None)
                        request = queued_request
                        break
                    if request is None:
                        return
                    future = download_executor.submit(
                        _download_prefetch_request,
                        request,
                        retry_seconds=retry_seconds,
                    )
                    download_futures[future] = (request, time.perf_counter())

            def submit_producers() -> None:
                nonlocal next_sample
                while next_sample < end_sample and len(producer_futures) < producer_count:
                    future = producer_executor.submit(build_requests, next_sample)
                    producer_futures[future] = next_sample
                    next_sample += 1

            def consume_ready_producer_results() -> None:
                nonlocal next_producer_result
                while next_producer_result in producer_results:
                    status, payload = producer_results.pop(next_producer_result)
                    sample_index_for_error = next_producer_result
                    next_producer_result += 1
                    if status == "error":
                        exc = payload
                        counters.samples_done += 1
                        counters.samples_skipped += 1
                        if not counters.first_sample_skip:
                            counters.first_sample_skip = f"sample={sample_index_for_error}: {exc}"
                        sample_pending.setdefault(sample_index_for_error, set())
                        close_sample_requests(sample_index_for_error)
                        submit_producers()
                        continue

                    _bounded_sample_index, sample_valid_pixels, requests = payload
                    counters.samples_done += 1
                    sample_pending.setdefault(sample_index_for_error, set())
                    counters.patches_done += len(self.strip_z_offsets)
                    counters.valid_pixels += int(sample_valid_pixels)
                    for request in requests:
                        classify_or_submit(request, sample_index_for_error, download_queue, queued_downloads)
                    close_sample_requests(sample_index_for_error)
                    submit_downloads()
                    submit_producers()

            submit_producers()
            submit_downloads()
            update_download_queue_counters()
            print_progress()
            while producer_futures or download_futures or queued_downloads:
                consume_ready_producer_results()
                submit_downloads()
                active = set(producer_futures) | set(download_futures)
                if not active:
                    break
                done, _ = wait(active, timeout=0.25, return_when=FIRST_COMPLETED)
                for future in done:
                    if future in producer_futures:
                        sample_index_for_error = producer_futures.pop(future)
                        try:
                            producer_results[sample_index_for_error] = ("ok", future.result())
                        except ValueError as exc:
                            producer_results[sample_index_for_error] = ("error", exc)
                        consume_ready_producer_results()
                    else:
                        request, _submitted_at = download_futures.pop(future)
                        consume_download_result(request, future.result())
                        submit_downloads()
                now = time.perf_counter()
                if now - last_progress >= 0.5:
                    update_download_queue_counters()
                    print_progress()
                    last_progress = now
            update_download_queue_counters()
            print_progress(final=True)
            normal_shutdown = True
        except BaseException:
            for future in producer_futures:
                future.cancel()
            for future in download_futures:
                future.cancel()
            producer_executor.shutdown(wait=False, cancel_futures=True)
            download_executor.shutdown(wait=False, cancel_futures=True)
            raise
        finally:
            if normal_shutdown:
                producer_executor.shutdown(wait=True)
                download_executor.shutdown(wait=True)
            if torch_threads_changed:
                torch.set_num_threads(previous_torch_threads)

        return {
            "generated": counters.unique_chunks_seen,
            "cache_hits": counters.cache_hits,
            "missing": counters.known_missing + counters.newly_missing,
            "known_missing": counters.known_missing,
            "newly_missing": counters.newly_missing,
            "downloaded": counters.downloaded,
            "queued_for_download": counters.queued_for_download,
            "download_done": counters.download_done,
            "bytes": counters.bytes_downloaded,
            "errors": counters.download_errors,
            "workers": worker_count,
            "producer_workers": producer_count,
            "patches": counters.patches_done,
            "samples": counters.samples_done,
            "skipped_samples": counters.samples_skipped,
            "valid_pixels": counters.valid_pixels,
            "first_error": counters.first_error,
            "first_sample_skip": counters.first_sample_skip,
            "max_exclusive_sample_index": counters.max_exclusive_sample_index,
        }


def iter_batches(
    loader: FiberStrip2DLoader,
    *,
    start_sample_index: int = 0,
    sample_mode: str = "random",
) -> Iterable[FiberStrip2DBatch]:
    sample_index = int(start_sample_index)
    while True:
        yield loader.load_batch(sample_index, sample_mode=sample_mode)
        sample_index += loader.config.batch_size
