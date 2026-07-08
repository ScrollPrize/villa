from __future__ import annotations

import glob
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from lasagna.omezarr_pyramid import _decode_normals as _lasagna_decode_normals

from vesuvius.neural_tracing.datasets.common import (
    begin_zarr_cache_trace,
    end_zarr_cache_trace,
    open_zarr,
)
from vesuvius.neural_tracing.fiber_trace_2d.fiber_json import Vc3dFiber, load_vc3d_fiber
from vesuvius.neural_tracing.fiber_trace_2d.strip_geometry import (
    FiberStripFrame,
    build_planar_side_strip_patch_grid,
    build_side_strip_patch_grid,
)
from vesuvius.neural_tracing.fiber_trace.dataset import (
    _SpatialChannelView,
    _load_lasagna_volume,
    _omezarr_level_shape,
    _open_manifest_channel,
    _validate_shape,
    _volume_shape_zyx,
)


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


@dataclass(frozen=True)
class FiberStrip2DBatch:
    images: np.ndarray
    coords_zyx: np.ndarray
    valid_mask: np.ndarray
    planar_images: np.ndarray
    planar_coords_zyx: np.ndarray
    planar_valid_mask: np.ndarray
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
    grad_mag: Any
    grad_mag_spacing_base: float
    nx: Any | None
    ny: Any | None
    nx_spacing_base: float | None
    ny_spacing_base: float | None
    dataset_config: dict[str, Any]


@dataclass(frozen=True)
class ZarrChunkRequest:
    store: Any
    store_identity: str
    key: str


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
) -> tuple[Any, float, Any, Any, float, float]:
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
    return grad_mag, float(grad_mag_spacing_base), nx, ny, float(nx_spacing_base), float(ny_spacing_base)


def _store_identity(store: Any) -> str:
    url = getattr(store, "_url", None)
    cache_dir = getattr(store, "_cache_dir", None)
    if url is not None:
        return f"{type(store).__module__}.{type(store).__name__}:{url}:{cache_dir}"
    path = getattr(store, "path", None)
    if path is not None:
        return f"{type(store).__module__}.{type(store).__name__}:{path}"
    return f"{type(store).__module__}.{type(store).__name__}:{repr(store)}"


def _array_chunks_zyx(array: Any) -> tuple[int, int, int] | None:
    chunks = getattr(array, "chunks", None)
    if chunks is None:
        return None
    chunks_tuple = tuple(int(v) for v in chunks)
    if isinstance(array, _SpatialChannelView):
        return array.chunks
    if len(chunks_tuple) == 4:
        return chunks_tuple[1:]
    if len(chunks_tuple) == 3:
        return chunks_tuple
    return None


def _is_remote_cached_store(store: Any) -> bool:
    return getattr(store, "_url", None) is not None and getattr(store, "_cache_dir", None) is not None


def _chunk_key(array: Any, chunk_zyx: tuple[int, int, int]) -> str | None:
    key_fn = getattr(array, "_chunk_key", None)
    if not callable(key_fn):
        return None
    shape = tuple(int(v) for v in getattr(array, "shape", ()))
    chunks = tuple(int(v) for v in getattr(array, "chunks", ()))
    if len(shape) == 3:
        return str(key_fn(chunk_zyx))
    if len(shape) == 4 and len(chunks) == 4:
        return str(key_fn((0,) + chunk_zyx))
    return None


def chunk_requests_for_coords(array: Any, coords_zyx: np.ndarray, valid_mask: np.ndarray) -> list[ZarrChunkRequest]:
    store = getattr(array, "store", None)
    chunks = getattr(array, "chunks", None)
    if store is None or chunks is None or not _is_remote_cached_store(store):
        return []
    chunks_tuple = tuple(int(v) for v in chunks)
    if len(chunks_tuple) == 4:
        chunks_zyx = chunks_tuple[1:]
    elif len(chunks_tuple) == 3:
        chunks_zyx = chunks_tuple
    else:
        return []
    if min(chunks_zyx) <= 0:
        return []

    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(coords_zyx).all(axis=-1)
    if not bool(valid.any()):
        return []
    shape = tuple(int(v) for v in getattr(array, "shape", ()))
    spatial_shape = shape[-3:]
    coords = np.asarray(coords_zyx, dtype=np.float64)
    valid &= (coords[..., 0] >= 0.0) & (coords[..., 0] <= float(spatial_shape[0] - 1))
    valid &= (coords[..., 1] >= 0.0) & (coords[..., 1] <= float(spatial_shape[1] - 1))
    valid &= (coords[..., 2] >= 0.0) & (coords[..., 2] <= float(spatial_shape[2] - 1))
    if not bool(valid.any()):
        return []

    base = np.floor(coords[valid]).astype(np.int64)
    high = np.minimum(base + 1, np.asarray(spatial_shape, dtype=np.int64) - 1)
    corners = []
    for dz in (0, 1):
        z = base[:, 0] if dz == 0 else high[:, 0]
        for dy in (0, 1):
            y = base[:, 1] if dy == 0 else high[:, 1]
            for dx in (0, 1):
                x = base[:, 2] if dx == 0 else high[:, 2]
                corners.append(np.stack([z, y, x], axis=1))
    corner_coords = np.concatenate(corners, axis=0)
    chunk_idx = corner_coords // np.asarray(chunks_zyx, dtype=np.int64)
    unique = np.unique(chunk_idx, axis=0)
    identity = _store_identity(store)
    requests: list[ZarrChunkRequest] = []
    for chunk in unique:
        key = _chunk_key(array, tuple(int(v) for v in chunk))
        if key is None:
            continue
        requests.append(ZarrChunkRequest(store=store, store_identity=identity, key=key))
    return requests


def _read_array_points(array: Any, z: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    shape = tuple(int(v) for v in getattr(array, "shape", ()))
    if len(shape) == 4:
        values = array[(0, z, y, x)]
    else:
        values = array[(z, y, x)]
    return np.asarray(values, dtype=np.float32)


def _sample_array_trilinear(array: Any, coords_zyx: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    shape = tuple(int(v) for v in getattr(array, "shape", ()))
    spatial_shape = shape[-3:]
    out = np.zeros(coords_zyx.shape[:2], dtype=np.float32)
    coords = np.asarray(coords_zyx, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool).copy()
    valid &= np.isfinite(coords).all(axis=-1)
    valid &= (coords[..., 0] >= 0.0) & (coords[..., 0] <= float(spatial_shape[0] - 1))
    valid &= (coords[..., 1] >= 0.0) & (coords[..., 1] <= float(spatial_shape[1] - 1))
    valid &= (coords[..., 2] >= 0.0) & (coords[..., 2] <= float(spatial_shape[2] - 1))
    if not bool(valid.any()):
        return out

    sample_coords = coords[valid]
    base = np.floor(sample_coords).astype(np.int64)
    high = np.minimum(base + 1, np.asarray(spatial_shape, dtype=np.int64) - 1)
    frac = (sample_coords - base.astype(np.float64)).astype(np.float32)
    fz, fy, fx = frac[:, 0], frac[:, 1], frac[:, 2]

    z0, y0, x0 = base[:, 0], base[:, 1], base[:, 2]
    z1, y1, x1 = high[:, 0], high[:, 1], high[:, 2]
    c000 = _read_array_points(array, z0, y0, x0)
    c001 = _read_array_points(array, z0, y0, x1)
    c010 = _read_array_points(array, z0, y1, x0)
    c011 = _read_array_points(array, z0, y1, x1)
    c100 = _read_array_points(array, z1, y0, x0)
    c101 = _read_array_points(array, z1, y0, x1)
    c110 = _read_array_points(array, z1, y1, x0)
    c111 = _read_array_points(array, z1, y1, x1)

    c00 = c000 * (1.0 - fx) + c001 * fx
    c01 = c010 * (1.0 - fx) + c011 * fx
    c10 = c100 * (1.0 - fx) + c101 * fx
    c11 = c110 * (1.0 - fx) + c111 * fx
    c0 = c00 * (1.0 - fy) + c01 * fy
    c1 = c10 * (1.0 - fy) + c11 * fy
    out[valid] = c0 * (1.0 - fz) + c1 * fz
    return out


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


class FiberStrip2DLoader:
    def __init__(self, config: FiberStrip2DConfig) -> None:
        if config.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        self.strip_z_offsets = strip_z_offsets_from_count_step(
            config.strip_z_offset_count, config.strip_z_offset_step
        )
        self.config = config
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
            grad_mag, grad_mag_spacing_base, nx, ny, nx_spacing_base, ny_spacing_base = _open_manifest_channels(
                dataset_config,
                self.config,
                volume=volume,
                volume_path=volume_path,
            )
            for fiber_path in _resolve_fiber_paths(dataset_config, self.config):
                fiber = load_vc3d_fiber(fiber_path)
                records.append(
                    _Record(
                        fiber=fiber,
                        volume=volume,
                        volume_path=volume_path,
                        volume_scale=volume_scale,
                        volume_spacing_base=volume_spacing_base,
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

    def _lasagna_normal_for_control_point(self, record: _Record, cp_index: int) -> np.ndarray:
        cp_xyz = np.asarray(record.fiber.control_points_xyz[cp_index], dtype=np.float64)
        cp_zyx = cp_xyz[[2, 1, 0]]
        grad_request = self._interpolation_cube(
            record.grad_mag,
            cp_zyx,
            array_spacing_base=record.grad_mag_spacing_base,
        )
        if grad_request is None:
            raise ValueError("missing Lasagna grad_mag sample at control point")
        grad_value = self._trilinear_cube(*grad_request)
        if grad_value <= 0.0:
            raise ValueError("Lasagna grad_mag sample is zero at control point")

        assert record.nx_spacing_base is not None
        assert record.ny_spacing_base is not None
        nx_request = self._interpolation_cube(record.nx, cp_zyx, array_spacing_base=record.nx_spacing_base)
        ny_request = self._interpolation_cube(record.ny, cp_zyx, array_spacing_base=record.ny_spacing_base)
        if nx_request is None or ny_request is None:
            raise ValueError("missing Lasagna nx/ny sample at control point")
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
            raise ValueError("degenerate Lasagna normal sample at control point")
        normal = _principal_tensor_axis(tensor, hint)
        if float(np.linalg.norm(normal)) <= 1.0e-12:
            raise ValueError("degenerate Lasagna normal sample at control point")
        return normal.astype(np.float32)

    def build_sample(
        self, sample_index: int
    ) -> tuple[list[FiberStripSample], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        record, record_index, cp_index = self.descriptor_for_sample_index(sample_index)
        sampled_normal = self._lasagna_normal_for_control_point(record, cp_index)
        images: list[np.ndarray] = []
        coords: list[np.ndarray] = []
        valids: list[np.ndarray] = []
        planar_images: list[np.ndarray] = []
        planar_coords: list[np.ndarray] = []
        planar_valids: list[np.ndarray] = []
        samples: list[FiberStripSample] = []
        for offset in self.strip_z_offsets:
            grid = build_side_strip_patch_grid(
                record.fiber,
                control_point_index=cp_index,
                patch_shape_hw=self.config.patch_shape_hw,
                strip_z_offset=float(offset),
                sampled_normal=sampled_normal,
                pixel_spacing_base=record.volume_spacing_base,
            )
            planar_grid = build_planar_side_strip_patch_grid(
                record.fiber,
                control_point_index=cp_index,
                patch_shape_hw=self.config.patch_shape_hw,
                strip_z_offset=float(offset),
                sampled_normal=sampled_normal,
                pixel_spacing_base=record.volume_spacing_base,
            )
            read_coords_zyx = grid.coords_zyx / float(record.volume_spacing_base)
            image = _sample_array_trilinear(record.volume, read_coords_zyx, grid.valid_mask)
            planar_read_coords_zyx = planar_grid.coords_zyx / float(record.volume_spacing_base)
            planar_image = _sample_array_trilinear(
                record.volume, planar_read_coords_zyx, planar_grid.valid_mask
            )
            sample = FiberStripSample(
                record_index=record_index,
                fiber_path=str(record.fiber.path) if record.fiber.path is not None else "",
                control_point_index=cp_index,
                control_point_xyz=np.asarray(record.fiber.control_points_xyz[cp_index], dtype=np.float32),
                strip_z_offset=float(offset),
                coords_zyx=grid.coords_zyx,
                valid_mask=grid.valid_mask,
                frame=grid.frame,
            )
            images.append(image)
            coords.append(grid.coords_zyx)
            valids.append(grid.valid_mask)
            planar_images.append(planar_image)
            planar_coords.append(planar_grid.coords_zyx)
            planar_valids.append(planar_grid.valid_mask)
            samples.append(sample)
        return (
            samples,
            np.stack(images, axis=0),
            np.stack(coords, axis=0),
            np.stack(valids, axis=0),
            np.stack(planar_images, axis=0),
            np.stack(planar_coords, axis=0),
            np.stack(planar_valids, axis=0),
        )

    def load_batch(self, start_sample_index: int = 0, batch_size: int | None = None) -> FiberStrip2DBatch:
        batch_size = self.config.batch_size if batch_size is None else int(batch_size)
        begin_zarr_cache_trace()
        all_samples: list[FiberStripSample] = []
        images: list[np.ndarray] = []
        coords: list[np.ndarray] = []
        valids: list[np.ndarray] = []
        planar_images: list[np.ndarray] = []
        planar_coords: list[np.ndarray] = []
        planar_valids: list[np.ndarray] = []
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
                    sample_planar_images,
                    sample_planar_coords,
                    sample_planar_valids,
                ) = self.build_sample(sample_index)
                first = sample_records[0]
                all_samples.extend(sample_records)
                images.append(sample_images)
                coords.append(sample_coords)
                valids.append(sample_valids)
                planar_images.append(sample_planar_images)
                planar_coords.append(sample_planar_coords)
                planar_valids.append(sample_planar_valids)
                record_indices.append(first.record_index)
                cp_indices.append(first.control_point_index)
                fiber_paths.append(first.fiber_path)
        finally:
            cache_stats = end_zarr_cache_trace()
        return FiberStrip2DBatch(
            images=np.stack(images, axis=0)[:, :, None, :, :],
            coords_zyx=np.stack(coords, axis=0),
            valid_mask=np.stack(valids, axis=0),
            planar_images=np.stack(planar_images, axis=0)[:, :, None, :, :],
            planar_coords_zyx=np.stack(planar_coords, axis=0),
            planar_valid_mask=np.stack(planar_valids, axis=0),
            strip_z_offsets=np.asarray(self.strip_z_offsets, dtype=np.float32),
            control_point_indices=np.asarray(cp_indices, dtype=np.int32),
            record_indices=np.asarray(record_indices, dtype=np.int32),
            fiber_paths=tuple(fiber_paths),
            samples=tuple(all_samples),
            cache_stats=cache_stats,
        )

    def chunk_requests_for_sample_index(self, sample_index: int) -> list[ZarrChunkRequest]:
        record, _, cp_index = self.descriptor_for_sample_index(sample_index)
        sampled_normal = self._lasagna_normal_for_control_point(record, cp_index)
        requests: list[ZarrChunkRequest] = []
        for offset in self.strip_z_offsets:
            grid = build_side_strip_patch_grid(
                record.fiber,
                control_point_index=cp_index,
                patch_shape_hw=self.config.patch_shape_hw,
                strip_z_offset=float(offset),
                sampled_normal=sampled_normal,
                pixel_spacing_base=record.volume_spacing_base,
            )
            read_coords_zyx = grid.coords_zyx / float(record.volume_spacing_base)
            requests.extend(chunk_requests_for_coords(record.volume, read_coords_zyx, grid.valid_mask))
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
