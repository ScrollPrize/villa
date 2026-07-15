from __future__ import annotations

import glob
import hashlib
import itertools
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from vesuvius.data.affine import read_transform_json
from vesuvius.neural_tracing.datasets.common import open_zarr
from vesuvius.neural_tracing.fiber_trace.dataset import (
    ZarrChunkRequest,
    _chunk_requests_for_bbox,
    _load_lasagna_volume,
    _omezarr_level_shape,
    _resolve_config_relative_path,
    _validate_shape,
    _volume_shape_zyx,
)
from vesuvius.neural_tracing.fiber_trace_2d.fiber_json import (
    Vc3dFiber,
    load_fiber_file,
)
from vesuvius.neural_tracing.fiber_trace_2d.strip_geometry import (
    control_point_line_index,
)
from vesuvius.neural_tracing.fiber_trace_3d.direction import (
    encode_lasagna_direction_3x2,
    projection_magnitude_weights_3x2,
)


_REMOTE_PREFIXES = ("http://", "https://", "s3://")


@dataclass(frozen=True)
class FiberTrace3DConfig:
    datasets: tuple[dict[str, Any], ...]
    batch_size: int = 192
    patch_shape_zyx: tuple[int, int, int] = (32, 32, 32)
    seed: int = 1
    cp_margin_voxels: int = 4
    presence_radius_voxels: float = 2.0
    presence_negative_edge_margin_voxels: int | None = None
    image_normalization: str = "zscore"
    augment_enabled: bool = True
    augment_shift_zyx: tuple[float, float, float] = (0.0, 0.0, 0.0)
    augment_rotation_degrees: float = 0.0
    augment_scale_min: float = 1.0
    augment_scale_max: float = 1.0
    augment_flip_probability: float = 0.0
    augment_brightness: float = 0.0
    augment_contrast_min: float = 1.0
    augment_contrast_max: float = 1.0
    augment_gamma_min: float = 1.0
    augment_gamma_max: float = 1.0
    augment_noise_std: float = 0.0
    augment_blur_sigma: float = 0.0
    round_source_to_chunk_boundaries: bool = True
    prefetch_workers: int = 16
    volume_cache_dir: str | None = None
    volume_cache_memory_mib: float | None = None
    volume_io_threads: int | None = None
    volume_cache_offline: bool = False
    volume_cache_retry_seconds: float = 0.0
    config_dir: Path | None = None

    @property
    def volume_cache_memory_bytes(self) -> int | None:
        if self.volume_cache_memory_mib is None:
            return None
        return int(float(self.volume_cache_memory_mib) * 1024.0 * 1024.0)


@dataclass(frozen=True)
class _Record:
    fiber: Vc3dFiber
    volume: Any
    volume_path: str
    volume_scale: int
    volume_spacing_base: float
    base_shape_zyx: tuple[int, int, int]
    fiber_identity: str
    dataset_config: dict[str, Any]


@dataclass(frozen=True)
class _Augment3DParams:
    sample_index: int
    cp_local_zyx: np.ndarray
    cp_volume_zyx: np.ndarray
    source_to_output_zyx: np.ndarray
    output_to_source_zyx: np.ndarray
    brightness: float
    contrast: float
    gamma: float
    noise_std: float
    blur_sigma: float


@dataclass(frozen=True)
class FiberTrace3DSample:
    sample_index: int
    record_index: int
    fiber_path: str
    control_point_index: int
    volume: torch.Tensor
    valid_mask: torch.Tensor
    direction_target: torch.Tensor
    direction_weight: torch.Tensor
    direction_mask: torch.Tensor
    presence_target: torch.Tensor
    presence_mask: torch.Tensor
    cp_local_zyx: torch.Tensor
    crop_origin_zyx: torch.Tensor


@dataclass(frozen=True)
class FiberTrace3DBatch:
    volume: torch.Tensor
    valid_mask: torch.Tensor
    direction_target: torch.Tensor
    direction_weight: torch.Tensor
    direction_mask: torch.Tensor
    presence_target: torch.Tensor
    presence_mask: torch.Tensor
    cp_local_zyx: torch.Tensor
    crop_origin_zyx: torch.Tensor
    sample_indices: torch.Tensor
    record_indices: torch.Tensor
    control_point_indices: torch.Tensor
    fiber_paths: tuple[str, ...]

    def to(self, device: torch.device | str) -> "FiberTrace3DBatch":
        return FiberTrace3DBatch(
            volume=self.volume.to(device),
            valid_mask=self.valid_mask.to(device),
            direction_target=self.direction_target.to(device),
            direction_weight=self.direction_weight.to(device),
            direction_mask=self.direction_mask.to(device),
            presence_target=self.presence_target.to(device),
            presence_mask=self.presence_mask.to(device),
            cp_local_zyx=self.cp_local_zyx.to(device),
            crop_origin_zyx=self.crop_origin_zyx.to(device),
            sample_indices=self.sample_indices.to(device),
            record_indices=self.record_indices.to(device),
            control_point_indices=self.control_point_indices.to(device),
            fiber_paths=self.fiber_paths,
        )


def _as_zyx3(value: Any, *, key: str) -> tuple[int, int, int]:
    if isinstance(value, int):
        size = int(value)
        return size, size, size
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{key} must be an int or length-3 sequence, got {value!r}")
    result = tuple(int(v) for v in value)
    if any(v <= 0 for v in result):
        raise ValueError(f"{key} values must be positive, got {value!r}")
    return result


def _as_float_zyx3(value: Any, *, key: str) -> tuple[float, float, float]:
    if isinstance(value, (int, float)):
        raw = (float(value),) * 3
    elif isinstance(value, (list, tuple)) and len(value) == 3:
        raw = tuple(float(v) for v in value)
    else:
        raise ValueError(f"{key} must be a number or length-3 sequence, got {value!r}")
    if any((not math.isfinite(v)) or v < 0.0 for v in raw):
        raise ValueError(f"{key} values must be finite and >= 0, got {value!r}")
    return raw


def _stable_seed(*parts: Any) -> int:
    digest = hashlib.blake2b(digest_size=16)
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return int.from_bytes(digest.digest(), "little", signed=False)


def _stable_digest(*parts: Any, digest_size: int = 24) -> str:
    digest = hashlib.blake2b(digest_size=digest_size)
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _round_json_array(value: np.ndarray, *, decimals: int = 6) -> str:
    return json.dumps(
        np.round(np.asarray(value, dtype=np.float64), decimals).tolist(),
        separators=(",", ":"),
    )


def _resolve_path(path: str | Path, config_dir: Path | None) -> str:
    path_s = str(path)
    if path_s.startswith(_REMOTE_PREFIXES):
        return path_s
    path_obj = Path(path_s).expanduser()
    if path_obj.is_absolute():
        return str(path_obj)
    if config_dir is not None:
        return str((config_dir / path_obj).resolve())
    return str((Path.cwd() / path_obj).resolve())


def _parse_worker_count(value: Any, *, key: str) -> int:
    count = int(value)
    if count <= 0:
        raise ValueError(f"{key} must be > 0")
    return count


def load_config(path: str | Path) -> FiberTrace3DConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"{config_path} must contain a JSON object")
    datasets = raw.get("datasets")
    if raw.get("_array_records") is not None:
        datasets = raw.get("datasets", [{"array_records": True}])
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("config must contain a non-empty 'datasets' list")
    cache_memory_mib = raw.get("volume_cache_memory_mib")
    if cache_memory_mib is not None:
        cache_memory_mib = float(cache_memory_mib)
        if not math.isfinite(cache_memory_mib) or cache_memory_mib <= 0.0:
            raise ValueError("volume_cache_memory_mib must be positive when provided")
    for unsupported in (
        "augment_shear_x",
        "augment_shear_y",
        "augment_shear_z",
        "augment_ringing",
        "augment_ringing_strength",
    ):
        if unsupported in raw and float(raw[unsupported]) != 0.0:
            raise ValueError(
                f"{unsupported} is not supported for 3D CP training yet; "
                "3D skew/ringing semantics must be specified explicitly"
            )
    return FiberTrace3DConfig(
        datasets=tuple(dict(entry) for entry in datasets),
        batch_size=int(raw.get("batch_size", 192)),
        patch_shape_zyx=_as_zyx3(
            raw.get("patch_shape_zyx", raw.get("crop_size", [32, 32, 32])),
            key="patch_shape_zyx",
        ),
        seed=int(raw.get("seed", 1)),
        cp_margin_voxels=int(raw.get("cp_margin_voxels", raw.get("control_point_margin_voxels", 4))),
        presence_radius_voxels=float(raw.get("presence_radius_voxels", 2.0)),
        presence_negative_edge_margin_voxels=(
            None
            if raw.get("presence_negative_edge_margin_voxels") is None
            else int(raw.get("presence_negative_edge_margin_voxels"))
        ),
        image_normalization=str(raw.get("image_normalization", "zscore")),
        augment_enabled=bool(raw.get("augment_enabled", True)),
        augment_shift_zyx=_as_float_zyx3(
            raw.get("augment_shift_zyx", raw.get("augment_shift_xyz", [0.0, 0.0, 0.0])),
            key="augment_shift_zyx",
        ),
        augment_rotation_degrees=float(raw.get("augment_rotation_degrees", 0.0)),
        augment_scale_min=float(raw.get("augment_scale_min", 1.0)),
        augment_scale_max=float(raw.get("augment_scale_max", 1.0)),
        augment_flip_probability=float(raw.get("augment_flip_probability", 0.0)),
        augment_brightness=float(raw.get("augment_brightness", 0.0)),
        augment_contrast_min=float(raw.get("augment_contrast_min", 1.0)),
        augment_contrast_max=float(raw.get("augment_contrast_max", 1.0)),
        augment_gamma_min=float(raw.get("augment_gamma_min", 1.0)),
        augment_gamma_max=float(raw.get("augment_gamma_max", 1.0)),
        augment_noise_std=float(raw.get("augment_noise_std", 0.0)),
        augment_blur_sigma=float(raw.get("augment_blur_sigma", 0.0)),
        round_source_to_chunk_boundaries=bool(raw.get("round_source_to_chunk_boundaries", True)),
        prefetch_workers=_parse_worker_count(raw.get("prefetch_workers", 16), key="prefetch_workers"),
        volume_cache_dir=None if raw.get("volume_cache_dir") is None else str(raw.get("volume_cache_dir")),
        volume_cache_memory_mib=cache_memory_mib,
        volume_io_threads=(
            None if raw.get("volume_io_threads") is None else int(raw.get("volume_io_threads"))
        ),
        volume_cache_offline=bool(raw.get("volume_cache_offline", False)),
        volume_cache_retry_seconds=float(raw.get("volume_cache_retry_seconds", 0.0)),
        config_dir=config_path.parent,
    )


def _common_zarr_config(config: FiberTrace3DConfig) -> dict[str, Any]:
    return {
        "volume_cache_dir": config.volume_cache_dir,
        "volume_cache_memory_mib": config.volume_cache_memory_mib,
        "volume_io_threads": config.volume_io_threads,
        "volume_cache_offline": config.volume_cache_offline,
        "volume_cache_retry_seconds": config.volume_cache_retry_seconds,
        "_config_dir": str(config.config_dir) if config.config_dir is not None else None,
    }


def _open_dataset_volume(dataset_config: dict[str, Any], config: FiberTrace3DConfig) -> Any:
    volume_path = dataset_config.get("base_volume_path", dataset_config.get("volume_path"))
    if volume_path is None:
        raise ValueError("dataset entry must define base_volume_path")
    scale = int(dataset_config.get("base_volume_scale", dataset_config.get("volume_scale", 0)))
    return open_zarr(
        _resolve_path(volume_path, config.config_dir),
        scale=scale,
        auth_json_path=dataset_config.get("base_volume_auth_json", dataset_config.get("volume_auth_json")),
        config=_common_zarr_config(config),
    )


def _validate_dataset_manifest(
    dataset_config: dict[str, Any],
    config: FiberTrace3DConfig,
    *,
    volume: Any,
    volume_path: str,
) -> tuple[int, int, int]:
    manifest_path = dataset_config.get("lasagna_manifest_path")
    if not manifest_path:
        if dataset_config.get("_array_records"):
            return tuple(int(v) for v in volume.shape)
        raise ValueError("dataset entry missing lasagna_manifest_path")
    common_config = _common_zarr_config(config)
    resolved_manifest = _resolve_config_relative_path(manifest_path, common_config)
    lasagna_volume = _load_lasagna_volume(resolved_manifest, common_config)
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
        context=f"Lasagna manifest {resolved_manifest}",
    )
    volume_scale = int(dataset_config.get("base_volume_scale", dataset_config.get("volume_scale", 0)))
    _validate_shape(
        _volume_shape_zyx(volume, label="base volume selected level"),
        _omezarr_level_shape(tuple(lasagna_volume.base_shape_zyx), volume_scale),
        label="base volume selected level",
        context=f"base_shape_zyx={lasagna_volume.base_shape_zyx} base_volume_scale={volume_scale}",
    )
    return tuple(int(v) for v in lasagna_volume.base_shape_zyx)


def _resolve_fiber_paths(dataset_config: dict[str, Any], config: FiberTrace3DConfig) -> list[Path]:
    paths: list[Path] = []
    raw_paths = dataset_config.get("fiber_paths")
    if raw_paths is not None:
        if isinstance(raw_paths, (str, Path)):
            raw_paths = [raw_paths]
        for raw in raw_paths:
            paths.append(Path(_resolve_path(raw, config.config_dir)))
    raw_glob = dataset_config.get("fiber_glob")
    if raw_glob:
        resolved = _resolve_path(raw_glob, config.config_dir)
        paths.extend(Path(path) for path in sorted(glob.glob(resolved)))
    if not paths:
        raise ValueError("dataset entry must define fiber_paths or fiber_glob")
    return paths


def _homogeneous_matrix_from_config(raw_matrix: Any, *, key: str) -> np.ndarray:
    matrix = np.asarray(raw_matrix, dtype=np.float64)
    if matrix.shape == (3, 4):
        matrix = np.vstack([matrix, [0.0, 0.0, 0.0, 1.0]])
    if matrix.shape != (4, 4):
        raise ValueError(f"{key} must be a 3x4 or 4x4 XYZ affine matrix")
    if not bool(np.isfinite(matrix).all()):
        raise ValueError(f"{key} contains non-finite values")
    if not np.allclose(matrix[3], np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)):
        raise ValueError(f"{key} must have homogeneous last row [0, 0, 0, 1]")
    return matrix


def _dataset_fiber_transform(
    dataset_config: dict[str, Any],
    config: FiberTrace3DConfig,
) -> tuple[np.ndarray | None, str]:
    json_sources = [
        (key, dataset_config[key])
        for key in ("fiber_transform_json", "fiber_transform_json_path")
        if dataset_config.get(key) is not None
    ]
    inline_sources = [
        (key, dataset_config[key])
        for key in ("fiber_transform", "transform")
        if dataset_config.get(key) is not None
    ]
    if len(json_sources) + len(inline_sources) == 0:
        return None, "none"
    if len(json_sources) + len(inline_sources) > 1:
        raise ValueError("set only one fiber transform source")
    invert = bool(
        dataset_config.get(
            "fiber_transform_invert",
            dataset_config.get("transform_invert", False),
        )
    )
    if json_sources:
        key, raw_path = json_sources[0]
        resolved = _resolve_path(raw_path, config.config_dir)
        matrix = np.asarray(read_transform_json(resolved).matrix_xyz, dtype=np.float64)
        source_label = f"{key}:{resolved}"
    else:
        key, raw_matrix = inline_sources[0]
        matrix = _homogeneous_matrix_from_config(raw_matrix, key=key)
        source_label = key
    if invert:
        matrix = np.linalg.inv(matrix)
    identity = _stable_digest(
        "fiber_transform",
        source_label,
        f"invert={invert}",
        _round_json_array(matrix, decimals=9),
    )
    return matrix, identity


def _first_out_of_bounds_control_point(
    fiber: Vc3dFiber,
    base_shape_zyx: tuple[int, int, int],
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


def _axis_angle_rotation(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if not math.isfinite(norm) or norm <= 1.0e-12:
        return np.eye(3, dtype=np.float64)
    axis = axis / norm
    z, y, x = axis
    skew = np.asarray(
        [[0.0, -x, y], [x, 0.0, -z], [-y, z, 0.0]],
        dtype=np.float64,
    )
    ident = np.eye(3, dtype=np.float64)
    return ident + math.sin(angle_rad) * skew + (1.0 - math.cos(angle_rad)) * (skew @ skew)


def _read_raw_block(array: Any, start_zyx: np.ndarray, end_zyx: np.ndarray) -> np.ndarray:
    start = np.asarray(start_zyx, dtype=np.int64)
    end = np.asarray(end_zyx, dtype=np.int64)
    shape = tuple(int(v) for v in getattr(array, "shape", ()))
    if len(shape) != 3:
        raise ValueError(f"3D loader expects a 3D ZYX base volume, got shape={shape}")
    block_shape = tuple(int(v) for v in np.maximum(end - start, 0))
    sample = np.asarray(array[0:1, 0:1, 0:1])
    out = np.zeros(block_shape, dtype=sample.dtype)
    src_start = np.maximum(start, 0)
    src_end = np.minimum(end, np.asarray(shape, dtype=np.int64))
    if not bool(np.all(src_end > src_start)):
        return out
    dst_start = src_start - start
    dst_end = dst_start + (src_end - src_start)
    out[
        int(dst_start[0]) : int(dst_end[0]),
        int(dst_start[1]) : int(dst_end[1]),
        int(dst_start[2]) : int(dst_end[2]),
    ] = array[
        int(src_start[0]) : int(src_end[0]),
        int(src_start[1]) : int(src_end[1]),
        int(src_start[2]) : int(src_end[2]),
    ]
    return out


def _chunks_zyx(array: Any) -> tuple[int, int, int] | None:
    chunks = getattr(array, "chunks", None)
    if chunks is None:
        return None
    chunks_tuple = tuple(int(v) for v in chunks)
    if len(chunks_tuple) == 3:
        return chunks_tuple
    if len(chunks_tuple) == 4:
        return chunks_tuple[1:]
    return None


def _round_bbox_to_chunks(
    array: Any,
    start_zyx: np.ndarray,
    end_zyx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    chunks = _chunks_zyx(array)
    if chunks is None:
        return start_zyx.astype(np.int64), end_zyx.astype(np.int64)
    chunk = np.asarray(chunks, dtype=np.int64)
    start = (np.asarray(start_zyx, dtype=np.int64) // chunk) * chunk
    end = ((np.asarray(end_zyx, dtype=np.int64) + chunk - 1) // chunk) * chunk
    return start.astype(np.int64), end.astype(np.int64)


def _normalize_image(volume: torch.Tensor, valid_mask: torch.Tensor, mode: str) -> torch.Tensor:
    mode_l = str(mode).lower()
    if mode_l in {"none", "raw", "identity"}:
        return volume
    valid = valid_mask.to(dtype=torch.bool)
    if not bool(valid.any()):
        return volume
    values = volume[valid]
    if mode_l == "zscore":
        mean = values.mean()
        std = values.std(unbiased=False).clamp_min(1.0e-6)
        return (volume - mean) / std
    if mode_l == "minmax":
        lo = values.min()
        hi = values.max()
        return (volume - lo) / (hi - lo).clamp_min(1.0e-6)
    raise ValueError(f"unsupported image_normalization {mode!r}")


def _format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(int(seconds + 0.5), 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


class FiberTrace3DLoader:
    def __init__(self, config: FiberTrace3DConfig) -> None:
        if config.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if config.cp_margin_voxels < 0:
            raise ValueError("cp_margin_voxels must be >= 0")
        if min(config.patch_shape_zyx) <= 2 * int(config.cp_margin_voxels):
            raise ValueError(
                "cp_margin_voxels must leave at least one possible CP voxel "
                f"inside patch_shape_zyx={config.patch_shape_zyx}"
            )
        if config.presence_radius_voxels <= 0.0:
            raise ValueError("presence_radius_voxels must be > 0")
        if config.augment_scale_min <= 0.0 or config.augment_scale_max <= 0.0:
            raise ValueError("augment_scale_min/max must be > 0")
        if config.augment_scale_min > config.augment_scale_max:
            raise ValueError("augment_scale_min must be <= augment_scale_max")
        if not (0.0 <= config.augment_flip_probability <= 1.0):
            raise ValueError("augment_flip_probability must be in [0,1]")
        self.config = config
        self.records = self._load_records()
        self._flat_offsets: list[int] = []
        flat_total = 0
        for record in self.records:
            self._flat_offsets.append(flat_total)
            flat_total += int(record.fiber.control_points_xyz.shape[0])
        self._flat_sample_count = int(flat_total)
        if self._flat_sample_count <= 0:
            raise ValueError("no control points found in configured fibers")
        self._random_pass_cache: dict[int, np.ndarray] = {}

    @property
    def sample_count(self) -> int:
        return int(self._flat_sample_count)

    def _load_records(self) -> list[_Record]:
        array_records = getattr(self.config, "_array_records", None)
        del array_records
        records: list[_Record] = []
        raw_array_records = None
        # Tests can attach _array_records after load_config by using from_mapping.
        if hasattr(self.config, "__dict__"):
            raw_array_records = getattr(self.config, "_array_records", None)
        if raw_array_records is not None:
            for index, raw in enumerate(raw_array_records):
                record = dict(raw)
                volume = np.asarray(record["volume"])
                if volume.ndim != 3:
                    raise ValueError("_array_records volume must be 3D")
                fiber = record["fiber"]
                if not isinstance(fiber, Vc3dFiber):
                    fibers = load_fiber_file(record["fiber_path"])
                    if len(fibers) != 1:
                        raise ValueError("_array_records fiber_path must resolve one fiber")
                    fiber = fibers[0]
                records.append(
                    _Record(
                        fiber=fiber,
                        volume=volume,
                        volume_path=str(record.get("volume_path", f"array:{index}")),
                        volume_scale=0,
                        volume_spacing_base=1.0,
                        base_shape_zyx=tuple(int(v) for v in volume.shape),
                        fiber_identity=_stable_digest("array", index, id(volume)),
                        dataset_config=record,
                    )
                )
            return records

        for dataset_config in self.config.datasets:
            volume_path_raw = dataset_config.get("base_volume_path", dataset_config.get("volume_path"))
            if volume_path_raw is None:
                raise ValueError("dataset entry must define base_volume_path")
            volume_path = _resolve_path(volume_path_raw, self.config.config_dir)
            volume_scale = int(dataset_config.get("base_volume_scale", dataset_config.get("volume_scale", 0)))
            volume_spacing_base = float(1 << volume_scale)
            volume = _open_dataset_volume(dataset_config, self.config)
            base_shape_zyx = _validate_dataset_manifest(
                dataset_config,
                self.config,
                volume=volume,
                volume_path=volume_path,
            )
            transform_xyz, transform_identity = _dataset_fiber_transform(
                dataset_config,
                self.config,
            )
            for fiber_path in _resolve_fiber_paths(dataset_config, self.config):
                fibers = load_fiber_file(
                    fiber_path,
                    transform_xyz=transform_xyz,
                    transform_identity=transform_identity,
                )
                for source_fiber_index, fiber in enumerate(fibers):
                    bad_control = _first_out_of_bounds_control_point(fiber, base_shape_zyx)
                    if bad_control is not None:
                        bad_index, bad_zyx = bad_control
                        print(
                            "fiber_trace_3d: skipping fiber with out-of-volume control point "
                            f"fiber_path='{fiber_path}' source_fiber_index={source_fiber_index} "
                            f"control_point_index={bad_index} "
                            f"control_point_zyx=({bad_zyx[0]:.3f}, {bad_zyx[1]:.3f}, {bad_zyx[2]:.3f}) "
                            f"base_shape_zyx={base_shape_zyx}",
                            flush=True,
                        )
                        continue
                    fiber_identity = _stable_digest(
                        "fiber3d",
                        str(fiber_path),
                        source_fiber_index,
                        transform_identity,
                        _round_json_array(np.asarray(fiber.line_points_xyz, dtype=np.float64)),
                    )
                    records.append(
                        _Record(
                            fiber=fiber,
                            volume=volume,
                            volume_path=volume_path,
                            volume_scale=volume_scale,
                            volume_spacing_base=volume_spacing_base,
                            base_shape_zyx=base_shape_zyx,
                            fiber_identity=fiber_identity,
                            dataset_config=dict(dataset_config),
                        )
                    )
        if not records:
            raise ValueError("no fiber records loaded")
        return records

    def _descriptor_for_flat_index(self, flat_index: int) -> tuple[_Record, int, int]:
        flat = int(flat_index)
        if flat < 0 or flat >= self._flat_sample_count:
            raise IndexError(f"flat sample index {flat} out of range")
        record_index = int(np.searchsorted(self._flat_offsets, flat, side="right") - 1)
        record_offset = int(self._flat_offsets[record_index])
        cp_index = flat - record_offset
        return self.records[record_index], record_index, int(cp_index)

    def _random_flat_index(self, sample_index: int) -> int:
        sample_count = int(self._flat_sample_count)
        position = int(sample_index)
        pass_index = math.floor(position / sample_count)
        offset = position % sample_count
        if pass_index not in self._random_pass_cache:
            rng = np.random.default_rng(_stable_seed(self.config.seed, "sample_order", pass_index))
            self._random_pass_cache[pass_index] = rng.permutation(sample_count).astype(np.int64)
        return int(self._random_pass_cache[pass_index][offset])

    def descriptor_for_sample_index(
        self,
        sample_index: int,
        *,
        sample_mode: str = "random",
    ) -> tuple[_Record, int, int]:
        if sample_mode == "random":
            return self._descriptor_for_flat_index(self._random_flat_index(sample_index))
        if sample_mode == "flat":
            return self._descriptor_for_flat_index(int(sample_index) % int(self._flat_sample_count))
        raise ValueError("sample_mode must be 'random' or 'flat'")

    def _rng(self, sample_index: int, purpose: str) -> np.random.Generator:
        return np.random.default_rng(_stable_seed(self.config.seed, purpose, int(sample_index)))

    def _control_point_volume_zyx(self, record: _Record, cp_index: int) -> np.ndarray:
        cp_zyx_base = np.asarray(record.fiber.control_points_zyx[int(cp_index)], dtype=np.float64)
        return cp_zyx_base / float(record.volume_spacing_base)

    def _sample_augment_params(
        self,
        record: _Record,
        cp_index: int,
        sample_index: int,
    ) -> _Augment3DParams:
        patch = np.asarray(self.config.patch_shape_zyx, dtype=np.int64)
        margin = int(self.config.cp_margin_voxels)
        center = (patch.astype(np.float64) - 1.0) * 0.5
        cp_volume = self._control_point_volume_zyx(record, cp_index)
        rng = self._rng(sample_index, "augment3d")

        if self.config.augment_enabled:
            shift = np.asarray(self.config.augment_shift_zyx, dtype=np.float64)
            low = np.maximum(np.floor(center - shift), margin)
            high = np.minimum(np.ceil(center + shift), patch - margin - 1)
            cp_local = np.asarray(
                [
                    rng.integers(int(lo), int(hi) + 1)
                    if int(hi) >= int(lo)
                    else int(round(center_axis))
                    for lo, hi, center_axis in zip(low, high, center, strict=True)
                ],
                dtype=np.float64,
            )
            scale = float(rng.uniform(self.config.augment_scale_min, self.config.augment_scale_max))
            flip = np.ones(3, dtype=np.float64)
            if self.config.augment_flip_probability > 0.0:
                flip[rng.random(3) < self.config.augment_flip_probability] = -1.0
            max_angle = math.radians(float(self.config.augment_rotation_degrees))
            if max_angle > 0.0:
                axis = rng.normal(size=3)
                angle = float(rng.uniform(-max_angle, max_angle))
                rotation = _axis_angle_rotation(axis, angle)
            else:
                rotation = np.eye(3, dtype=np.float64)
            source_to_output = rotation @ np.diag(flip * scale)
            brightness = (
                float(rng.uniform(-self.config.augment_brightness, self.config.augment_brightness))
                if self.config.augment_brightness > 0.0
                else 0.0
            )
            contrast = float(rng.uniform(self.config.augment_contrast_min, self.config.augment_contrast_max))
            gamma = float(rng.uniform(self.config.augment_gamma_min, self.config.augment_gamma_max))
            noise_std = float(self.config.augment_noise_std)
            blur_sigma = float(self.config.augment_blur_sigma)
        else:
            cp_local = center
            source_to_output = np.eye(3, dtype=np.float64)
            brightness = 0.0
            contrast = 1.0
            gamma = 1.0
            noise_std = 0.0
            blur_sigma = 0.0

        return _Augment3DParams(
            sample_index=int(sample_index),
            cp_local_zyx=cp_local.astype(np.float64),
            cp_volume_zyx=cp_volume.astype(np.float64),
            source_to_output_zyx=source_to_output.astype(np.float64),
            output_to_source_zyx=np.linalg.inv(source_to_output).astype(np.float64),
            brightness=brightness,
            contrast=contrast,
            gamma=gamma,
            noise_std=noise_std,
            blur_sigma=blur_sigma,
        )

    def _actual_source_bbox(
        self,
        record: _Record,
        params: _Augment3DParams,
    ) -> tuple[np.ndarray, np.ndarray]:
        patch = np.asarray(self.config.patch_shape_zyx, dtype=np.float64)
        corners = np.asarray(
            list(itertools.product(*[(0.0, float(size - 1.0)) for size in patch])),
            dtype=np.float64,
        )
        rel_output = corners - params.cp_local_zyx.reshape(1, 3)
        rel_source = rel_output @ params.output_to_source_zyx.T
        coords = params.cp_volume_zyx.reshape(1, 3) + rel_source
        start = np.floor(np.min(coords, axis=0) - 2.0).astype(np.int64)
        end = np.ceil(np.max(coords, axis=0) + 3.0).astype(np.int64)
        if self.config.round_source_to_chunk_boundaries:
            start, end = _round_bbox_to_chunks(record.volume, start, end)
        return start, end

    def _prefetch_source_bbox(
        self,
        record: _Record,
        cp_index: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        cp = self._control_point_volume_zyx(record, cp_index)
        patch = np.asarray(self.config.patch_shape_zyx, dtype=np.float64)
        min_scale = max(float(self.config.augment_scale_min), 1.0e-6)
        shift = np.asarray(self.config.augment_shift_zyx, dtype=np.float64)
        radius = float(np.linalg.norm(patch - 1.0)) / min_scale + float(np.max(shift)) + 4.0
        start = np.floor(cp - radius).astype(np.int64)
        end = np.ceil(cp + radius + 1.0).astype(np.int64)
        if self.config.round_source_to_chunk_boundaries:
            start, end = _round_bbox_to_chunks(record.volume, start, end)
        return start, end

    def _sample_volume_patch(
        self,
        record: _Record,
        block: np.ndarray,
        source_start_zyx: np.ndarray,
        params: _Augment3DParams,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        patch_shape = tuple(int(v) for v in self.config.patch_shape_zyx)
        block_t = torch.as_tensor(block, dtype=torch.float32, device=device).view(
            1, 1, *block.shape
        )
        z = torch.arange(patch_shape[0], dtype=torch.float32, device=device)
        y = torch.arange(patch_shape[1], dtype=torch.float32, device=device)
        x = torch.arange(patch_shape[2], dtype=torch.float32, device=device)
        zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
        local = torch.stack([zz, yy, xx], dim=-1)
        cp_local = torch.as_tensor(params.cp_local_zyx, dtype=torch.float32, device=device)
        cp_volume = torch.as_tensor(params.cp_volume_zyx, dtype=torch.float32, device=device)
        inv_matrix = torch.as_tensor(params.output_to_source_zyx, dtype=torch.float32, device=device)
        source_start = torch.as_tensor(source_start_zyx, dtype=torch.float32, device=device)
        rel_output = local - cp_local.view(1, 1, 1, 3)
        rel_source = torch.einsum("...i,ji->...j", rel_output, inv_matrix)
        source_local = cp_volume.view(1, 1, 1, 3) + rel_source - source_start.view(1, 1, 1, 3)

        d, h, w = (int(v) for v in block.shape)
        valid = (
            torch.isfinite(source_local).all(dim=-1)
            & (source_local[..., 0] >= 0.0)
            & (source_local[..., 0] <= float(d - 1))
            & (source_local[..., 1] >= 0.0)
            & (source_local[..., 1] <= float(h - 1))
            & (source_local[..., 2] >= 0.0)
            & (source_local[..., 2] <= float(w - 1))
        )
        gx = source_local[..., 2] * (2.0 / float(max(w - 1, 1))) - 1.0 if w > 1 else torch.zeros_like(source_local[..., 2])
        gy = source_local[..., 1] * (2.0 / float(max(h - 1, 1))) - 1.0 if h > 1 else torch.zeros_like(source_local[..., 1])
        gz = source_local[..., 0] * (2.0 / float(max(d - 1, 1))) - 1.0 if d > 1 else torch.zeros_like(source_local[..., 0])
        grid = torch.stack([gx, gy, gz], dim=-1).unsqueeze(0)
        sampled = F.grid_sample(
            block_t,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[0, 0]
        sampled = torch.where(valid, sampled, torch.zeros_like(sampled))
        del record
        return sampled, valid

    def _apply_value_augmentation(
        self,
        volume: torch.Tensor,
        valid_mask: torch.Tensor,
        params: _Augment3DParams,
    ) -> torch.Tensor:
        out = _normalize_image(volume, valid_mask, self.config.image_normalization)
        if not self.config.augment_enabled:
            return out
        if params.contrast != 1.0:
            if bool(valid_mask.any()):
                center = out[valid_mask].mean()
            else:
                center = out.mean()
            out = (out - center) * float(params.contrast) + center
        if params.brightness != 0.0:
            out = out + float(params.brightness)
        if params.gamma != 1.0:
            valid_values = out[valid_mask] if bool(valid_mask.any()) else out.reshape(-1)
            lo = valid_values.min()
            hi = valid_values.max()
            scaled = ((out - lo) / (hi - lo).clamp_min(1.0e-6)).clamp(0.0, 1.0)
            out = torch.pow(scaled, float(params.gamma)) * (hi - lo) + lo
        if params.noise_std > 0.0:
            gen = torch.Generator(device=out.device)
            gen.manual_seed(_stable_seed(self.config.seed, "value_noise", params.sample_index) % (2**63 - 1))
            out = out + torch.randn(out.shape, generator=gen, device=out.device) * float(params.noise_std)
        if params.blur_sigma > 0.0:
            # Conservative separable 3D blur. Kept simple for the first 3D path.
            sigma = float(params.blur_sigma)
            radius = max(1, int(math.ceil(3.0 * sigma)))
            coords = torch.arange(-radius, radius + 1, dtype=torch.float32, device=out.device)
            kernel = torch.exp(-0.5 * (coords / sigma) ** 2)
            kernel = kernel / kernel.sum().clamp_min(1.0e-12)
            data = out.view(1, 1, *out.shape)
            for axis in range(3):
                shape = [1, 1, 1, 1, 1]
                shape[2 + axis] = int(kernel.numel())
                weight = kernel.view(*shape)
                padding = [0, 0, 0, 0, 0, 0]
                padding[(2 - axis) * 2] = radius
                padding[(2 - axis) * 2 + 1] = radius
                data = F.conv3d(F.pad(data, padding, mode="replicate"), weight)
            out = data[0, 0]
        return torch.where(valid_mask, out, torch.zeros_like(out))

    def _line_window_for_labels(
        self,
        record: _Record,
        cp_index: int,
        params: _Augment3DParams,
    ) -> tuple[np.ndarray, int]:
        line_points = np.asarray(record.fiber.line_points_zyx, dtype=np.float64) / float(
            record.volume_spacing_base
        )
        line_index = control_point_line_index(record.fiber, int(cp_index))
        if line_points.shape[0] < 2:
            raise ValueError("fiber line must contain at least two points")
        cumulative = np.concatenate(
            [
                np.zeros((1,), dtype=np.float64),
                np.cumsum(np.linalg.norm(np.diff(line_points, axis=0), axis=1)),
            ]
        )
        patch_diag = float(np.linalg.norm(np.asarray(self.config.patch_shape_zyx, dtype=np.float64)))
        anchor = float(cumulative[line_index])
        start_arc = anchor - patch_diag
        end_arc = anchor + patch_diag
        start = max(0, int(np.searchsorted(cumulative, start_arc, side="right") - 2))
        end = min(
            int(line_points.shape[0]),
            int(np.searchsorted(cumulative, end_arc, side="left") + 3),
        )
        if end - start < 2:
            start = max(0, line_index - 1)
            end = min(int(line_points.shape[0]), line_index + 2)
        del params
        return line_points[start:end], line_index - start

    def _build_targets(
        self,
        record: _Record,
        cp_index: int,
        params: _Augment3DParams,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        patch_shape = tuple(int(v) for v in self.config.patch_shape_zyx)
        line_points, _local_line_index = self._line_window_for_labels(
            record,
            cp_index,
            params,
        )
        rel_source = line_points - params.cp_volume_zyx.reshape(1, 3)
        points_out = params.cp_local_zyx.reshape(1, 3) + rel_source @ params.source_to_output_zyx.T
        if points_out.shape[0] < 2:
            raise ValueError("label line window must contain at least two points")
        segment_start = points_out[:-1].astype(np.float32)
        segment_vec = (points_out[1:] - points_out[:-1]).astype(np.float32)
        segment_len2 = np.sum(segment_vec * segment_vec, axis=1)
        valid_segments = np.isfinite(segment_len2) & (segment_len2 > 1.0e-12)
        if not bool(valid_segments.any()):
            raise ValueError("label line window has no valid segments")
        segment_start = segment_start[valid_segments]
        segment_vec = segment_vec[valid_segments]
        segment_len2 = segment_len2[valid_segments]

        grid = np.stack(
            np.meshgrid(
                np.arange(patch_shape[0], dtype=np.float32),
                np.arange(patch_shape[1], dtype=np.float32),
                np.arange(patch_shape[2], dtype=np.float32),
                indexing="ij",
            ),
            axis=-1,
        ).reshape(-1, 3)
        min_dist2 = np.full((grid.shape[0],), np.inf, dtype=np.float32)
        nearest_tangent = np.zeros((grid.shape[0], 3), dtype=np.float32)
        for start in range(0, int(segment_start.shape[0]), 64):
            stop = min(start + 64, int(segment_start.shape[0]))
            p0 = segment_start[start:stop]
            vec = segment_vec[start:stop]
            length2 = segment_len2[start:stop]
            rel = grid[:, None, :] - p0[None, :, :]
            alpha = np.sum(rel * vec[None, :, :], axis=2) / length2[None, :]
            alpha = np.clip(alpha, 0.0, 1.0)
            closest = p0[None, :, :] + alpha[..., None] * vec[None, :, :]
            dist2 = np.sum((grid[:, None, :] - closest) ** 2, axis=2)
            local_nearest = np.argmin(dist2, axis=1)
            local_dist2 = dist2[np.arange(grid.shape[0]), local_nearest]
            replace_mask = local_dist2 < min_dist2
            if bool(np.any(replace_mask)):
                min_dist2[replace_mask] = local_dist2[replace_mask]
                nearest_tangent[replace_mask] = vec[local_nearest[replace_mask]]

        radius = float(self.config.presence_radius_voxels)
        positive = min_dist2.reshape(patch_shape) <= np.float32(radius * radius)
        tangent_zyx = nearest_tangent.reshape(*patch_shape, 3)
        tangent_xyz = tangent_zyx[..., [2, 1, 0]]
        direction_target_np = encode_lasagna_direction_3x2(tangent_xyz).astype(np.float32)
        direction_weight_np = projection_magnitude_weights_3x2(tangent_xyz).astype(np.float32)
        direction_target = torch.as_tensor(
            direction_target_np.transpose(3, 0, 1, 2),
            dtype=torch.float32,
            device=valid_mask.device,
        )
        direction_weight = torch.as_tensor(
            direction_weight_np.transpose(3, 0, 1, 2),
            dtype=torch.float32,
            device=valid_mask.device,
        )
        positive_t = torch.as_tensor(positive, dtype=torch.bool, device=valid_mask.device)
        direction_mask = valid_mask & positive_t

        presence_target = positive_t.to(dtype=torch.float32).view(1, *patch_shape)
        presence_mask = valid_mask.clone()
        edge_margin = self.config.presence_negative_edge_margin_voxels
        if edge_margin is None:
            edge_margin = int(math.ceil(max(self.config.augment_shift_zyx)))
        if edge_margin > 0:
            z = torch.arange(patch_shape[0], device=valid_mask.device)
            y = torch.arange(patch_shape[1], device=valid_mask.device)
            x = torch.arange(patch_shape[2], device=valid_mask.device)
            zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
            interior = (
                (zz >= edge_margin)
                & (zz < patch_shape[0] - edge_margin)
                & (yy >= edge_margin)
                & (yy < patch_shape[1] - edge_margin)
                & (xx >= edge_margin)
                & (xx < patch_shape[2] - edge_margin)
            )
            presence_mask &= interior
        presence_mask = presence_mask.view(1, *patch_shape)
        return (
            direction_target,
            direction_weight,
            direction_mask.view(1, *patch_shape),
            presence_target,
            presence_mask,
        )

    def load_sample(
        self,
        sample_index: int,
        *,
        sample_mode: str = "random",
        device: torch.device | str = "cpu",
    ) -> FiberTrace3DSample:
        resolved_device = torch.device(device)
        record, record_index, cp_index = self.descriptor_for_sample_index(
            sample_index,
            sample_mode=sample_mode,
        )
        params = self._sample_augment_params(record, cp_index, int(sample_index))
        source_start, source_end = self._actual_source_bbox(record, params)
        block = _read_raw_block(record.volume, source_start, source_end)
        volume, valid = self._sample_volume_patch(
            record,
            block,
            source_start,
            params,
            device=resolved_device,
        )
        volume = self._apply_value_augmentation(volume, valid, params)
        (
            direction_target,
            direction_weight,
            direction_mask,
            presence_target,
            presence_mask,
        ) = self._build_targets(record, cp_index, params, valid)
        return FiberTrace3DSample(
            sample_index=int(sample_index),
            record_index=int(record_index),
            fiber_path="" if record.fiber.path is None else str(record.fiber.path),
            control_point_index=int(cp_index),
            volume=volume.view(1, *self.config.patch_shape_zyx),
            valid_mask=valid.view(1, *self.config.patch_shape_zyx),
            direction_target=direction_target,
            direction_weight=direction_weight,
            direction_mask=direction_mask,
            presence_target=presence_target,
            presence_mask=presence_mask,
            cp_local_zyx=torch.as_tensor(params.cp_local_zyx, dtype=torch.float32, device=resolved_device),
            crop_origin_zyx=torch.as_tensor(source_start, dtype=torch.float32, device=resolved_device),
        )

    def load_batch(
        self,
        start_sample_index: int,
        *,
        sample_mode: str = "random",
        device: torch.device | str = "cpu",
    ) -> FiberTrace3DBatch:
        samples = [
            self.load_sample(
                int(start_sample_index) + offset,
                sample_mode=sample_mode,
                device=device,
            )
            for offset in range(int(self.config.batch_size))
        ]
        return FiberTrace3DBatch(
            volume=torch.stack([sample.volume for sample in samples], dim=0),
            valid_mask=torch.stack([sample.valid_mask for sample in samples], dim=0),
            direction_target=torch.stack([sample.direction_target for sample in samples], dim=0),
            direction_weight=torch.stack([sample.direction_weight for sample in samples], dim=0),
            direction_mask=torch.stack([sample.direction_mask for sample in samples], dim=0),
            presence_target=torch.stack([sample.presence_target for sample in samples], dim=0),
            presence_mask=torch.stack([sample.presence_mask for sample in samples], dim=0),
            cp_local_zyx=torch.stack([sample.cp_local_zyx for sample in samples], dim=0),
            crop_origin_zyx=torch.stack([sample.crop_origin_zyx for sample in samples], dim=0),
            sample_indices=torch.as_tensor([sample.sample_index for sample in samples], dtype=torch.long, device=samples[0].volume.device),
            record_indices=torch.as_tensor([sample.record_index for sample in samples], dtype=torch.long, device=samples[0].volume.device),
            control_point_indices=torch.as_tensor([sample.control_point_index for sample in samples], dtype=torch.long, device=samples[0].volume.device),
            fiber_paths=tuple(sample.fiber_path for sample in samples),
        )

    def chunk_requests_for_sample_index(
        self,
        sample_index: int,
        *,
        sample_mode: str = "random",
    ) -> list[ZarrChunkRequest]:
        record, _record_index, cp_index = self.descriptor_for_sample_index(
            sample_index,
            sample_mode=sample_mode,
        )
        start, end = self._prefetch_source_bbox(record, cp_index)
        return _chunk_requests_for_bbox(record.volume, (start, end), label="base")

    def prefetch(
        self,
        start_sample_index: int,
        sample_count: int,
        *,
        workers: int | None = None,
        sample_mode: str = "random",
    ) -> dict[str, Any]:
        total = int(sample_count)
        if total < 0:
            raise ValueError("sample_count must be >= 0")
        worker_count = max(1, int(self.config.prefetch_workers if workers is None else workers))
        request_by_identity: dict[tuple[str, str], ZarrChunkRequest] = {}
        print(
            f"fiber_trace_3d prefetch: generating chunk requests samples={total} "
            f"workers={worker_count} sample_mode={sample_mode}",
            flush=True,
        )
        for sample_index in range(int(start_sample_index), int(start_sample_index) + total):
            for request in self.chunk_requests_for_sample_index(
                sample_index,
                sample_mode=sample_mode,
            ):
                request_by_identity[(request.store_identity, request.key)] = request
        requests = list(request_by_identity.values())
        if not requests:
            print("fiber_trace_3d prefetch: no remote cached zarr chunk requests", flush=True)
            return {
                "samples": total,
                "chunks": 0,
                "downloaded": 0,
                "errors": 0,
                "elapsed_s": 0.0,
            }

        start_time = time.perf_counter()
        done = 0
        errors = 0
        bytes_read = 0
        bar_width = 24

        def fetch(request: ZarrChunkRequest) -> int:
            data = request.store[request.key]
            return len(data)

        def print_progress(*, force: bool = False) -> None:
            if not force and done < len(requests) and done % 32 != 0:
                return
            elapsed = max(time.perf_counter() - start_time, 1.0e-9)
            rate = done / elapsed
            eta = (len(requests) - done) / rate if rate > 0.0 else 0.0
            filled = int(bar_width * done / max(len(requests), 1))
            bar = "#" * filled + "-" * (bar_width - filled)
            mib = bytes_read / (1024.0 * 1024.0)
            mib_s = mib / elapsed
            print(
                "\r"
                f"fiber_trace_3d prefetch[{bar}] {done}/{len(requests)} "
                f"eta={_format_seconds(eta)} errors={errors} {mib:.1f}MiB {mib_s:.1f}MiB/s",
                end="" if done < len(requests) else "\n",
                flush=True,
            )

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(fetch, request) for request in requests]
            for future in as_completed(futures):
                try:
                    bytes_read += int(future.result())
                except Exception:
                    errors += 1
                done += 1
                print_progress()
        print_progress(force=True)
        elapsed = time.perf_counter() - start_time
        return {
            "samples": total,
            "chunks": len(requests),
            "downloaded": done - errors,
            "errors": errors,
            "elapsed_s": elapsed,
            "mib": bytes_read / (1024.0 * 1024.0),
        }


def config_from_mapping(raw: dict[str, Any], *, config_dir: Path | None = None) -> FiberTrace3DConfig:
    path = Path(config_dir or ".") / "_fiber_trace_3d_inline.json"
    del path
    datasets = raw.get("datasets")
    if raw.get("_array_records") is not None and datasets is None:
        datasets = [{"array_records": True}]
    cfg = FiberTrace3DConfig(
        datasets=tuple(dict(entry) for entry in datasets),
        batch_size=int(raw.get("batch_size", 192)),
        patch_shape_zyx=_as_zyx3(raw.get("patch_shape_zyx", [32, 32, 32]), key="patch_shape_zyx"),
        seed=int(raw.get("seed", 1)),
        cp_margin_voxels=int(raw.get("cp_margin_voxels", 4)),
        presence_radius_voxels=float(raw.get("presence_radius_voxels", 2.0)),
        augment_enabled=bool(raw.get("augment_enabled", True)),
        augment_shift_zyx=_as_float_zyx3(raw.get("augment_shift_zyx", [0.0, 0.0, 0.0]), key="augment_shift_zyx"),
        augment_rotation_degrees=float(raw.get("augment_rotation_degrees", 0.0)),
        augment_scale_min=float(raw.get("augment_scale_min", 1.0)),
        augment_scale_max=float(raw.get("augment_scale_max", 1.0)),
        augment_flip_probability=float(raw.get("augment_flip_probability", 0.0)),
        image_normalization=str(raw.get("image_normalization", "zscore")),
        round_source_to_chunk_boundaries=bool(raw.get("round_source_to_chunk_boundaries", True)),
        prefetch_workers=int(raw.get("prefetch_workers", 16)),
        config_dir=config_dir,
    )
    if "_array_records" in raw:
        object.__setattr__(cfg, "_array_records", raw["_array_records"])
    return cfg
