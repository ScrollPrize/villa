from __future__ import annotations

import glob
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
    FiberStripGridTorch,
    FiberStripLineWindow,
    build_side_strip_patch_grid_tensor_from_line_window,
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
    volume_cache_offline: bool = False
    volume_cache_retry_seconds: float = 0.0
    strip_coord_cache_dir: str | None = None
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
    control_point_xy: np.ndarray


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
class _PreparedStripSample:
    source: _StripSource
    params_by_offset: list[FiberStripAugmentParams | None]
    offset_grids: list[FiberStripGridTorch]
    coords_zyx: np.ndarray
    valid_mask: np.ndarray
    line_xy_by_offset: list[np.ndarray]
    control_point_xy_by_offset: list[np.ndarray]


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
        self._sample_identity_keys = self._build_sample_identity_keys()
        self._random_pass_cache: dict[int, np.ndarray] = {}
        self._random_pass_cache_lock = threading.Lock()
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

    def descriptor_for_sample_index(
        self, sample_index: int, *, sample_mode: str = "random"
    ) -> tuple[_Record, int, int]:
        if sample_mode == "random":
            flat = self._random_flat_index(sample_index)
        elif sample_mode == "flat":
            flat = self._flat_sample_index(sample_index)
        else:
            raise ValueError("sample_mode must be 'random' or 'flat'")
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
    ) -> Path | None:
        if self.config.strip_coord_cache_dir is None:
            return None
        cp_xyz = np.asarray(record.fiber.control_points_xyz[control_point_index], dtype=np.float64)
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
                offset_axis_zyx = torch.as_tensor(
                    np.asarray(data["offset_axis_zyx"], dtype=np.float32),
                    device=device,
                )
                grid = FiberStripGridTorch(
                    coords_xyz=coords_zyx[..., (2, 1, 0)].contiguous(),
                    coords_zyx=coords_zyx,
                    valid_mask=torch.as_tensor(np.asarray(data["valid_mask"], dtype=bool), device=device),
                    frame=frame,
                    offset_axis_xyz=offset_axis_zyx[..., (2, 1, 0)].contiguous(),
                    offset_axis_zyx=offset_axis_zyx,
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
                np.savez(
                    handle,
                    version=np.asarray(_STRIP_COORD_CACHE_VERSION),
                    source_shape_hw=np.asarray(source_shape_hw, dtype=np.int64),
                    center_offset=np.asarray(float(center_offset), dtype=np.float64),
                    coords_zyx=_as_numpy_float32(grid.coords_zyx),
                    valid_mask=_as_numpy_bool(grid.valid_mask),
                    offset_axis_zyx=_as_numpy_float32(grid.offset_axis_zyx),
                    source_line_xy=_as_numpy_float32(source_line_xy),
                    source_control_point_xy=_as_numpy_float32(source_control_point_xy),
                    frame_tangent_xyz=np.asarray(frame.tangent_xyz, dtype=np.float32),
                    frame_side_xyz=np.asarray(frame.side_xyz, dtype=np.float32),
                    frame_mesh_normal_xyz=np.asarray(frame.mesh_normal_xyz, dtype=np.float32),
                )
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
        profile: dict[str, float] | None = None,
        use_augmentation_envelope: bool | None = None,
        debug_label: str | None = None,
    ) -> _StripSource:
        if debug_label is not None:
            print(f"{debug_label} descriptor start", flush=True)
        with _ProfileBlock(profile, "descriptor"):
            record, record_index, cp_index = self.descriptor_for_sample_index(
                sample_index, sample_mode=sample_mode
            )
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

    def _offset_grid_from_source(self, source: _StripSource, offset: float) -> FiberStripGridTorch:
        axis_zyx = source.grid.offset_axis_zyx
        axis_xyz = source.grid.offset_axis_xyz
        if axis_zyx is None or axis_xyz is None:
            raise ValueError("strip source grid is missing offset-axis data")
        delta_base = (float(offset) - float(source.center_offset)) * float(source.record.volume_spacing_base)
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
        profile: dict[str, float] | None = None,
    ) -> _PreparedStripSample:
        device = resolve_torch_device(self.config.augment.device)
        source = self.build_strip_source(sample_index, device=device, sample_mode=sample_mode, profile=profile)
        params_by_offset: list[FiberStripAugmentParams | None] = [
            (
                random_combined_augmentation(self.config.augment, sample_index, offset_index)
                if self.config.augment.enabled
                else None
            )
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
            line_and_cp_cache[None] = (
                source.source_line_xy.to(device=device, dtype=torch.float32),
                source.source_control_point_xy.to(device=device, dtype=torch.float32),
            )
        else:
            unique_params = [params for params in dict.fromkeys(params_by_offset) if params is not None]
            if unique_params:
                source_line_xy = source.source_line_xy.to(device=device, dtype=torch.float32)
                source_cp_xy = source.source_control_point_xy.to(device=device, dtype=torch.float32).view(1, 2)
                source_points = torch.cat([source_line_xy, source_cp_xy], dim=0)
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
                        line_xy = mapped[:-1]
                        cp_xy = mapped[-1]
                        if line_xy.numel() != 0:
                            line_xy = line_xy[
                                torch.isfinite(line_xy).all(dim=1)
                                & (line_xy[:, 0] >= 0.0)
                                & (line_xy[:, 0] <= float(int(width) - 1))
                                & (line_xy[:, 1] >= 0.0)
                                & (line_xy[:, 1] <= float(int(height) - 1))
                            ]
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

    def _finish_prepared_sample(
        self,
        prepared: _PreparedStripSample,
        images_np: np.ndarray,
        valids_np: np.ndarray,
        *,
        apply_image_augmentation: bool,
        profile: dict[str, float] | None = None,
    ) -> tuple[list[FiberStripSample], np.ndarray, np.ndarray, np.ndarray]:
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
            coords_zyx = coords_zyx_np[offset_index]
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
            coords_zyx_np,
            np.stack(valids, axis=0),
        )

    def build_sample(
        self,
        sample_index: int,
        *,
        sample_mode: str = "random",
        profile: dict[str, float] | None = None,
        apply_image_augmentation: bool = True,
    ) -> tuple[list[FiberStripSample], np.ndarray, np.ndarray, np.ndarray]:
        prepared = self._prepare_sample(sample_index, sample_mode=sample_mode, profile=profile)
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
            profile=profile,
        )

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
        profile: dict[str, float] | None = None,
        apply_image_augmentation: bool = True,
    ) -> FiberStrip2DBatch:
        batch_size = self.config.batch_size if batch_size is None else int(batch_size)
        begin_zarr_cache_trace()
        all_samples: list[FiberStripSample] = []
        images: list[np.ndarray] = []
        coords: list[np.ndarray] = []
        valids: list[np.ndarray] = []
        record_indices: list[int] = []
        cp_indices: list[int] = []
        fiber_paths: list[str] = []
        batch_wall_start = time.perf_counter()

        def report_skip(current_sample_index: int, exc: ValueError) -> None:
            self._load_batch_skipped_samples += 1
            if self._load_batch_skipped_samples <= 10:
                print(
                    "fiber_trace_2d: skipping invalid training sample "
                    f"sample_index={current_sample_index} reason={exc}",
                    flush=True,
                )
            elif self._load_batch_skipped_samples == 11:
                print(
                    "fiber_trace_2d: further invalid training sample skip messages suppressed",
                    flush=True,
                )

        def build_candidate(raw_sample_index: int) -> tuple[
            int,
            int,
            dict[str, float],
            tuple[list[FiberStripSample], np.ndarray, np.ndarray, np.ndarray] | None,
            ValueError | None,
        ]:
            current_sample_index = self._bounded_sample_index(raw_sample_index, sample_index_limit)
            local_profile: dict[str, float] = {} if profile is not None else {}
            worker_start = time.perf_counter()
            try:
                result = self.build_sample(
                    current_sample_index,
                    sample_mode=sample_mode,
                    profile=local_profile if profile is not None else None,
                    apply_image_augmentation=apply_image_augmentation,
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
            next_raw_to_submit = int(start_sample_index)
            next_raw_to_consume = int(start_sample_index)
            max_attempts = max(batch_size * 100, batch_size + 1000)
            attempts = 0
            worker_count = min(max(1, int(self.config.loader_workers)), max(1, batch_size), max_attempts)
            if sample_mode == "random":
                with _ProfileBlock(profile, "random_order"):
                    self._ensure_random_pass_orders_for_indices(
                        self._bounded_sample_index(raw, sample_index_limit)
                        for raw in range(int(start_sample_index), int(start_sample_index) + max_attempts)
                    )

            def consume_result(
                current_sample_index: int,
                local_profile: dict[str, float],
                result: tuple[list[FiberStripSample], np.ndarray, np.ndarray, np.ndarray] | None,
                skip_exc: ValueError | None,
            ) -> None:
                if skip_exc is not None:
                    report_skip(current_sample_index, skip_exc)
                    _merge_profile(profile, local_profile)
                    return
                if result is None:
                    raise RuntimeError("internal load_batch missing sample result")
                _merge_profile(profile, local_profile)
                (
                    sample_records,
                    sample_images,
                    sample_coords,
                    sample_valids,
                ) = result
                first = sample_records[0]
                all_samples.extend(sample_records)
                images.append(sample_images)
                coords.append(sample_coords)
                valids.append(sample_valids)
                record_indices.append(first.record_index)
                cp_indices.append(first.control_point_index)
                fiber_paths.append(first.fiber_path)

            if worker_count == 1:
                while len(images) < batch_size and attempts < max_attempts:
                    raw_sample_index = next_raw_to_submit
                    next_raw_to_submit += 1
                    attempts += 1
                    (
                        returned_raw,
                        current_sample_index,
                        local_profile,
                        result,
                        skip_exc,
                    ) = build_candidate(raw_sample_index)
                    if returned_raw != raw_sample_index:
                        raise RuntimeError("internal load_batch raw sample index mismatch")
                    consume_result(current_sample_index, local_profile, result, skip_exc)
            else:
                pending: dict[Future[Any], int] = {}
                outcomes: dict[
                    int,
                    tuple[
                        int,
                        dict[str, float],
                        tuple[list[FiberStripSample], np.ndarray, np.ndarray, np.ndarray] | None,
                        ValueError | None,
                    ],
                ] = {}
                executor = self._get_loader_executor(worker_count)
                while len(images) < batch_size and (pending or attempts < max_attempts):
                    needed_candidates = batch_size - len(images)
                    while (
                        len(pending) < worker_count
                        and len(pending) + len(outcomes) < needed_candidates
                        and attempts < max_attempts
                    ):
                        raw_sample_index = next_raw_to_submit
                        next_raw_to_submit += 1
                        attempts += 1
                        future = executor.submit(build_candidate, raw_sample_index)
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
                    while len(images) < batch_size and next_raw_to_consume in outcomes:
                        (
                            current_sample_index,
                            local_profile,
                            result,
                            skip_exc,
                        ) = outcomes.pop(next_raw_to_consume)
                        next_raw_to_consume += 1
                        consume_result(current_sample_index, local_profile, result, skip_exc)
                for future in pending:
                    future.cancel()
            if len(images) < batch_size:
                raise ValueError(
                    "could not assemble a full fiber_trace_2d batch after "
                    f"{attempts} deterministic sample attempts; requested={batch_size} "
                    f"loaded={len(images)} skipped={attempts - len(images)}"
                )
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
        )

    def chunk_requests_for_sample_index(
        self, sample_index: int, *, sample_mode: str = "random", sample_index_limit: int | None = None
    ) -> list[ZarrChunkRequest]:
        device = resolve_torch_device(self.config.augment.device)
        bounded_sample_index = self._bounded_sample_index(sample_index, sample_index_limit)
        source = self.build_strip_source(bounded_sample_index, device=device, sample_mode=sample_mode)
        requests: list[ZarrChunkRequest] = []
        for offset_index, _ in enumerate(self.strip_z_offsets):
            coords_zyx, valid_mask = self._prefetch_envelope_coords_from_source(source, offset_index)
            requests.extend(source.record.sampler.chunk_requests_for_coords(coords_zyx, valid_mask))
        return requests

    def chunk_requests_for_samples(
        self,
        start_sample_index: int,
        sample_count: int,
        *,
        sample_mode: str = "random",
        sample_index_limit: int | None = None,
    ) -> list[ZarrChunkRequest]:
        unique: dict[tuple[str, str], ZarrChunkRequest] = {}
        for sample_index in range(int(start_sample_index), int(start_sample_index) + int(sample_count)):
            for request in self.chunk_requests_for_sample_index(
                sample_index,
                sample_mode=sample_mode,
                sample_index_limit=sample_index_limit,
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
        sample_bounded_index: dict[int, int] = {}
        complete_raw_samples: set[int] = set()
        next_safe_raw_sample = start_sample
        start = time.perf_counter()
        print(
            "prefetch: "
            f"samples={total_samples} patches={total_patches} "
            f"workers={worker_count} sampler_workers={producer_count} "
            f"sample_mode={sample_mode} sample_index_limit={int(sample_index_limit or 0)} "
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
            return bounded_sample_index, valid_pixels, requests

        def classify_or_submit(
            request: ZarrChunkRequest,
            raw_sample_index: int,
            download_executor: ThreadPoolExecutor,
            download_futures: dict[Future[dict[str, Any]], tuple[ZarrChunkRequest, float]],
        ) -> None:
            identity = (request.store_identity, request.key)
            if identity in completed_chunks:
                return
            sample_pending.setdefault(raw_sample_index, set()).add(identity)
            chunk_waiters.setdefault(identity, set()).add(raw_sample_index)
            if identity in seen:
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
            future = download_executor.submit(_download_prefetch_request, request, retry_seconds=retry_seconds)
            download_futures[future] = (request, time.perf_counter())

        def advance_safe_prefix() -> None:
            nonlocal next_safe_raw_sample
            while next_safe_raw_sample < end_sample and next_safe_raw_sample in complete_raw_samples:
                bounded_index = sample_bounded_index.get(
                    next_safe_raw_sample,
                    self._bounded_sample_index(next_safe_raw_sample, sample_index_limit),
                )
                counters.max_exclusive_sample_index = max(
                    counters.max_exclusive_sample_index,
                    int(bounded_index) + 1,
                )
                next_safe_raw_sample += 1

        def mark_sample_complete(raw_sample_index: int) -> None:
            if sample_pending.get(raw_sample_index):
                return
            complete_raw_samples.add(raw_sample_index)
            advance_safe_prefix()

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
        producer_executor = ThreadPoolExecutor(max_workers=producer_count)
        download_executor = ThreadPoolExecutor(max_workers=worker_count)
        normal_shutdown = False
        try:
            producer_futures: dict[Future[tuple[int, int, list[ZarrChunkRequest]]], int] = {}
            download_futures: dict[Future[dict[str, Any]], tuple[ZarrChunkRequest, float]] = {}

            def update_download_queue_counters() -> None:
                counters.queued_download_futures = len(download_futures)

            def submit_producers() -> None:
                nonlocal next_sample
                while next_sample < end_sample and len(producer_futures) < producer_count:
                    future = producer_executor.submit(build_requests, next_sample)
                    producer_futures[future] = next_sample
                    next_sample += 1

            submit_producers()
            update_download_queue_counters()
            print_progress()
            while producer_futures or download_futures:
                active = set(producer_futures) | set(download_futures)
                done, _ = wait(active, timeout=0.25, return_when=FIRST_COMPLETED)
                for future in done:
                    if future in producer_futures:
                        sample_index_for_error = producer_futures.pop(future)
                        try:
                            bounded_sample_index, sample_valid_pixels, requests = future.result()
                        except ValueError as exc:
                            counters.samples_done += 1
                            bounded_index = self._bounded_sample_index(sample_index_for_error, sample_index_limit)
                            sample_bounded_index[sample_index_for_error] = bounded_index
                            counters.samples_skipped += 1
                            if not counters.first_sample_skip:
                                counters.first_sample_skip = f"sample={sample_index_for_error}: {exc}"
                            mark_sample_complete(sample_index_for_error)
                            submit_producers()
                            continue
                        counters.samples_done += 1
                        sample_bounded_index[sample_index_for_error] = int(bounded_sample_index)
                        sample_pending.setdefault(sample_index_for_error, set())
                        counters.patches_done += len(self.strip_z_offsets)
                        counters.valid_pixels += int(sample_valid_pixels)
                        for request in requests:
                            classify_or_submit(request, sample_index_for_error, download_executor, download_futures)
                        mark_sample_complete(sample_index_for_error)
                        submit_producers()
                    else:
                        request, _submitted_at = download_futures.pop(future)
                        consume_download_result(request, future.result())
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
