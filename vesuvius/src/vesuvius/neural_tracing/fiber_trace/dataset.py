from __future__ import annotations

import glob
import hashlib
import importlib.util
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from vesuvius.neural_tracing.fiber_trace.fiber_json import Vc3dFiber, load_vc3d_fiber
from vesuvius.neural_tracing.fiber_trace.geometry import (
    decode_lasagna_normals_xyz,
    perturb_direction,
    random_unit_vector,
    tangent_at_point,
    xyz_to_zyx,
    zyx_to_xyz,
)
from vesuvius.neural_tracing.fiber_trace.labels import IGNORE_ID, IGNORE_INDEX
from vesuvius.neural_tracing.datasets.common import (
    _read_volume_crop_from_patch as _common_read_volume_crop_from_patch,
    begin_zarr_cache_trace,
    end_zarr_cache_trace,
    open_zarr as _common_open_zarr,
)


@dataclass(frozen=True)
class FiberTraceBatch:
    volume: torch.Tensor
    mask_values: torch.Tensor
    nx_values: torch.Tensor | None
    ny_values: torch.Tensor | None
    valid_mask: torch.Tensor
    labels: torch.Tensor
    target_id: torch.Tensor
    target_fw_xyz: torch.Tensor
    center_normal_xyz: torch.Tensor
    cond_fw_xyz: torch.Tensor
    crop_origin_zyx: torch.Tensor
    sample_local_zyx: torch.Tensor
    line_points_xyz: torch.Tensor
    positive_target_id: torch.Tensor
    crop_kinds: tuple[str, ...]
    fiber_paths: tuple[str, ...]
    direction_kinds: tuple[str, ...]

    def to(self, device: torch.device | str) -> "FiberTraceBatch":
        return FiberTraceBatch(
            volume=self.volume.to(device),
            mask_values=self.mask_values.to(device),
            nx_values=None if self.nx_values is None else self.nx_values.to(device),
            ny_values=None if self.ny_values is None else self.ny_values.to(device),
            valid_mask=self.valid_mask.to(device),
            labels=self.labels.to(device),
            target_id=self.target_id.to(device),
            target_fw_xyz=self.target_fw_xyz.to(device),
            center_normal_xyz=self.center_normal_xyz.to(device),
            cond_fw_xyz=self.cond_fw_xyz.to(device),
            crop_origin_zyx=self.crop_origin_zyx.to(device),
            sample_local_zyx=self.sample_local_zyx.to(device),
            line_points_xyz=self.line_points_xyz.to(device),
            positive_target_id=self.positive_target_id.to(device),
            crop_kinds=self.crop_kinds,
            fiber_paths=self.fiber_paths,
            direction_kinds=self.direction_kinds,
        )

    def with_classification(
        self,
        *,
        valid_mask: torch.Tensor,
        labels: torch.Tensor,
        target_id: torch.Tensor,
        target_fw_xyz: torch.Tensor,
    ) -> "FiberTraceBatch":
        return FiberTraceBatch(
            volume=self.volume,
            mask_values=self.mask_values,
            nx_values=self.nx_values,
            ny_values=self.ny_values,
            valid_mask=valid_mask,
            labels=labels,
            target_id=target_id,
            target_fw_xyz=target_fw_xyz,
            center_normal_xyz=self.center_normal_xyz,
            cond_fw_xyz=self.cond_fw_xyz,
            crop_origin_zyx=self.crop_origin_zyx,
            sample_local_zyx=self.sample_local_zyx,
            line_points_xyz=self.line_points_xyz,
            positive_target_id=self.positive_target_id,
            crop_kinds=self.crop_kinds,
            fiber_paths=self.fiber_paths,
            direction_kinds=self.direction_kinds,
        )


@dataclass
class FiberTraceDebugTableState:
    rows: int = 0
    legend_printed: bool = False


@dataclass(frozen=True)
class ZarrChunkRequest:
    store: Any
    store_identity: str
    key: str
    label: str


@dataclass(frozen=True)
class _FiberRecord:
    fiber: Vc3dFiber
    volume: Any
    mask: Any
    nx: Any | None
    ny: Any | None
    volume_scale: int
    volume_spacing_base: float
    mask_spacing_base: float
    nx_spacing_base: float | None
    ny_spacing_base: float | None
    dataset_config: dict[str, Any]


_REMOTE_PREFIXES = ("http://", "https://", "s3://")
_MANIFESTLESS_DATASET_KEYS = frozenset(
    {
        "volume_path",
        "mask_path",
        "grad_mag_path",
        "mask_scale",
        "grad_mag_scale",
        "mask_auth_json",
        "nx_path",
        "ny_path",
        "normal_scale",
        "normal_auth_json",
        "nx_scale",
        "ny_scale",
        "nx_auth_json",
        "ny_auth_json",
    }
)


def _reject_valid_mask_threshold_key(config: dict[str, Any], *, context: str) -> None:
    if "valid_mask_threshold" not in config:
        return
    raise ValueError(
        "valid_mask_threshold was removed; mask/grad-mag validity is always "
        f"binary value > 0 in {context}"
    )


def _reject_manifestless_dataset_keys(dataset_config: dict[str, Any]) -> None:
    present = sorted(_MANIFESTLESS_DATASET_KEYS.intersection(dataset_config))
    if not present:
        return
    keys = ", ".join(present)
    raise ValueError(
        "manifest-less Lasagna channel configuration is not supported. "
        f"Remove {keys} and use lasagna_manifest_path with base_volume_path."
    )


def _is_remote_path(path: str | Path) -> bool:
    return str(path).startswith(_REMOTE_PREFIXES)


def _resolve_config_relative_path(path: str | Path, config: dict[str, Any]) -> str:
    path_s = str(path)
    if _is_remote_path(path_s):
        return path_s
    path_obj = Path(path_s).expanduser()
    if path_obj.is_absolute():
        return str(path_obj)
    config_dir = config.get("_config_dir") or config.get("config_dir")
    if config_dir:
        return str((Path(config_dir).expanduser() / path_obj).resolve())
    return str((Path.cwd() / path_obj).resolve())


def _scale_level_to_spacing_base(scale_level: int) -> float:
    scale_level = int(scale_level)
    if scale_level < 0:
        raise ValueError(f"scale level must be >= 0, got {scale_level}")
    return float(1 << scale_level)


def _omezarr_level_shape(
    base_shape_zyx: tuple[int, int, int], scale_level: int
) -> tuple[int, int, int]:
    z, y, x = (int(v) for v in base_shape_zyx)
    for _ in range(max(0, int(scale_level))):
        z = max(1, (z + 1) // 2)
        y = max(1, (y + 1) // 2)
        x = max(1, (x + 1) // 2)
    return z, y, x


class _SpatialChannelView:
    """3D view over a 3D zarr or one channel of a CZYX zarr."""

    def __init__(self, array: Any, *, channel_index: int, label: str) -> None:
        self.array = array
        self.channel_index = int(channel_index)
        self.label = str(label)
        shape = tuple(int(v) for v in getattr(array, "shape", ()))
        if len(shape) == 3:
            self._shape = shape
            self._is_czyx = False
        elif len(shape) == 4:
            if self.channel_index < 0 or self.channel_index >= shape[0]:
                raise ValueError(
                    f"{self.label} channel index {self.channel_index} is out of "
                    f"bounds for CZYX shape {shape}"
                )
            self._shape = shape[1:]
            self._is_czyx = True
        else:
            raise ValueError(f"{self.label} must be a 3D or CZYX array, got {shape}")

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._shape

    @property
    def chunks(self) -> tuple[int, int, int] | None:
        chunks = getattr(self.array, "chunks", None)
        if chunks is None:
            return None
        chunks_tuple = tuple(int(v) for v in chunks)
        if self._is_czyx and len(chunks_tuple) == 4:
            return chunks_tuple[1:]
        if not self._is_czyx and len(chunks_tuple) == 3:
            return chunks_tuple
        return None

    def __getitem__(self, key: Any) -> Any:
        if self._is_czyx:
            if not isinstance(key, tuple):
                key = (key,)
            return self.array[(self.channel_index,) + key]
        return self.array[key]


def _as_zyx3(value: Any, *, key: str) -> tuple[int, int, int]:
    if isinstance(value, int):
        size = int(value)
        return size, size, size
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{key} must be an int or length-3 sequence, got {value!r}")
    return tuple(int(v) for v in value)


def _open_zarr_array(
    path: str | Path, *, scale: int, auth_json_path: str | None, config: dict[str, Any]
):
    return _common_open_zarr(
        path, scale=scale, auth_json_path=auth_json_path, config=config
    )


def _open_zarr_array_resolved(
    path: str | Path, *, scale: int, auth_json_path: str | None, config: dict[str, Any]
):
    return _open_zarr_array(
        _resolve_config_relative_path(path, config),
        scale=scale,
        auth_json_path=auth_json_path,
        config=config,
    )


def _load_lasagna_volume(path: str | Path, config: dict[str, Any]) -> Any:
    resolved = _resolve_config_relative_path(path, config)
    if _is_remote_path(resolved):
        raise ValueError("lasagna_manifest_path must be a local .lasagna.json file")
    try:
        from lasagna.lasagna_volume import LasagnaVolume
    except ImportError:
        try:
            from lasagna_volume import LasagnaVolume
        except ImportError:
            repo_root = Path(__file__).resolve().parents[5]
            module_path = repo_root / "lasagna" / "lasagna_volume.py"
            if not module_path.exists():
                raise
            spec = importlib.util.spec_from_file_location(
                "_fiber_trace_lasagna_volume", module_path
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"could not load LasagnaVolume from {module_path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            LasagnaVolume = module.LasagnaVolume

    return LasagnaVolume.load(resolved)


def _manifest_group_root_and_level(volume: Any, group: Any) -> tuple[str, int]:
    group_path = Path(group.zarr_path)
    abs_path = group_path if group_path.is_absolute() else volume.path.parent / group_path
    data_level = int(group.scaledown)
    if abs_path.name.isdigit():
        data_level = int(abs_path.name)
        abs_path = abs_path.parent
    return str(abs_path.resolve()), data_level


def _spatial_shape(array: Any, *, label: str) -> tuple[int, int, int]:
    return _SpatialChannelView(array, channel_index=0, label=label).shape


def _volume_shape_zyx(array: Any, *, label: str) -> tuple[int, int, int]:
    shape = tuple(int(v) for v in getattr(array, "shape", ()))
    if len(shape) != 3:
        raise ValueError(f"{label} must be a 3D ZYX array, got shape={shape}")
    return shape


def _validate_shape(
    shape: tuple[int, int, int],
    expected: tuple[int, int, int],
    *,
    label: str,
    context: str,
) -> None:
    if tuple(shape) != tuple(expected):
        raise ValueError(
            f"{label} shape mismatch: shape={tuple(shape)} expected={tuple(expected)} "
            f"from {context}"
        )


def _open_manifest_channel(
    volume: Any,
    channel_name: str,
    *,
    auth_json_path: str | None,
    config: dict[str, Any],
) -> tuple[_SpatialChannelView, float]:
    group, channel_index = volume.channel_group(channel_name)
    root_path, data_level = _manifest_group_root_and_level(volume, group)
    array = _open_zarr_array(
        root_path, scale=data_level, auth_json_path=auth_json_path, config=config
    )
    view = _SpatialChannelView(array, channel_index=channel_index, label=channel_name)
    if volume.base_shape_zyx is not None:
        expected = _omezarr_level_shape(volume.base_shape_zyx, int(group.scaledown))
        _validate_shape(
            view.shape,
            expected,
            label=f"Lasagna {channel_name}",
            context=(
                f"base_shape_zyx={volume.base_shape_zyx} "
                f"scaledown={group.scaledown}"
            ),
        )
    spacing_base = float(group.sd_fac) * float(volume.source_to_base)
    return view, spacing_base


def _read_raw_crop(
    array: Any, crop_shape: tuple[int, int, int], min_corner: np.ndarray
) -> np.ndarray:
    max_corner = min_corner + np.asarray(crop_shape, dtype=np.int64)
    out = np.zeros(crop_shape, dtype=np.asarray(array[0:1, 0:1, 0:1]).dtype)
    shape = np.asarray(array.shape, dtype=np.int64)
    src_starts = np.maximum(min_corner, 0)
    src_ends = np.minimum(max_corner, shape)
    dst_starts = src_starts - min_corner
    dst_ends = dst_starts + (src_ends - src_starts)
    if np.all(src_ends > src_starts):
        out[
            dst_starts[0] : dst_ends[0],
            dst_starts[1] : dst_ends[1],
            dst_starts[2] : dst_ends[2],
        ] = array[
            src_starts[0] : src_ends[0],
            src_starts[1] : src_ends[1],
            src_starts[2] : src_ends[2],
        ]
    return out


def _read_scaled_channel_crop(
    array: Any,
    crop_shape: tuple[int, int, int],
    min_corner: np.ndarray,
    *,
    dst_spacing_base: float,
    src_spacing_base: float,
) -> np.ndarray:
    dst_spacing_base = float(dst_spacing_base)
    src_spacing_base = float(src_spacing_base)
    if dst_spacing_base <= 0.0 or src_spacing_base <= 0.0:
        raise ValueError(
            "channel spacing must be positive: "
            f"dst={dst_spacing_base} src={src_spacing_base}"
        )
    if abs(dst_spacing_base - src_spacing_base) < 1e-6:
        return _read_raw_crop(array, crop_shape, min_corner)

    sample_axes = [
        np.arange(int(crop_shape[axis]), dtype=np.int64) + int(min_corner[axis])
        for axis in range(3)
    ]
    src_axes = [
        np.floor(axis.astype(np.float64) * dst_spacing_base / src_spacing_base).astype(
            np.int64
        )
        for axis in sample_axes
    ]
    src_shape = np.asarray(array.shape, dtype=np.int64)
    valid_axes = [
        (src_axes[axis] >= 0) & (src_axes[axis] < src_shape[axis])
        for axis in range(3)
    ]
    sample_dtype = np.asarray(array[0:1, 0:1, 0:1]).dtype
    out = np.zeros(crop_shape, dtype=sample_dtype)
    if not all(bool(valid.any()) for valid in valid_axes):
        return out

    z_idx = src_axes[0][valid_axes[0]]
    y_idx = src_axes[1][valid_axes[1]]
    x_idx = src_axes[2][valid_axes[2]]
    z0, z1 = int(z_idx.min()), int(z_idx.max()) + 1
    y0, y1 = int(y_idx.min()), int(y_idx.max()) + 1
    x0, x1 = int(x_idx.min()), int(x_idx.max()) + 1
    block = np.asarray(array[z0:z1, y0:y1, x0:x1])
    data = block[np.ix_(z_idx - z0, y_idx - y0, x_idx - x0)]
    out[np.ix_(valid_axes[0], valid_axes[1], valid_axes[2])] = data
    return out


def _value_at_sample_zyx(
    array: Any,
    sample_zyx: np.ndarray,
    *,
    sample_spacing_base: float,
    array_spacing_base: float,
) -> tuple[np.ndarray, bool]:
    sample_zyx = np.asarray(sample_zyx, dtype=np.float64)
    array_zyx = np.floor(
        sample_zyx * float(sample_spacing_base) / float(array_spacing_base)
    ).astype(np.int64)
    shape = np.asarray(array.shape, dtype=np.int64)
    if np.any(array_zyx < 0) or np.any(array_zyx >= shape):
        return np.zeros((), dtype=np.asarray(array[0:1, 0:1, 0:1]).dtype), False
    value = np.asarray(
        array[
            array_zyx[0] : array_zyx[0] + 1,
            array_zyx[1] : array_zyx[1] + 1,
            array_zyx[2] : array_zyx[2] + 1,
        ]
    )
    if value.size == 0:
        return np.zeros((), dtype=np.asarray(array[0:1, 0:1, 0:1]).dtype), False
    return value.reshape(-1)[0], True


def _array_chunks_zyx(array: Any) -> tuple[int, int, int] | None:
    chunks = getattr(array, "chunks", None)
    if chunks is None:
        return None
    chunks_tuple = tuple(int(v) for v in chunks)
    if len(chunks_tuple) == 3:
        return chunks_tuple
    if len(chunks_tuple) == 4:
        return chunks_tuple[1:]
    return None


def _raw_source_bbox_zyx(
    array: Any, crop_shape: tuple[int, int, int], min_corner: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    shape = np.asarray(array.shape, dtype=np.int64)
    max_corner = np.asarray(min_corner, dtype=np.int64) + np.asarray(
        crop_shape, dtype=np.int64
    )
    starts = np.maximum(np.asarray(min_corner, dtype=np.int64), 0)
    ends = np.minimum(max_corner, shape)
    if not bool(np.all(ends > starts)):
        return None
    return starts, ends


def _scaled_source_bbox_zyx(
    array: Any,
    crop_shape: tuple[int, int, int],
    min_corner: np.ndarray,
    *,
    dst_spacing_base: float,
    src_spacing_base: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    sample_axes = [
        np.arange(int(crop_shape[axis]), dtype=np.int64) + int(min_corner[axis])
        for axis in range(3)
    ]
    src_axes = [
        np.floor(
            axis.astype(np.float64) * float(dst_spacing_base) / float(src_spacing_base)
        ).astype(np.int64)
        for axis in sample_axes
    ]
    src_shape = np.asarray(array.shape, dtype=np.int64)
    valid_axes = [
        (src_axes[axis] >= 0) & (src_axes[axis] < src_shape[axis])
        for axis in range(3)
    ]
    if not all(bool(valid.any()) for valid in valid_axes):
        return None
    starts = np.asarray(
        [int(src_axes[axis][valid_axes[axis]].min()) for axis in range(3)],
        dtype=np.int64,
    )
    ends = np.asarray(
        [int(src_axes[axis][valid_axes[axis]].max()) + 1 for axis in range(3)],
        dtype=np.int64,
    )
    return starts, ends


def _chunk_ranges_for_bbox(
    array: Any, bbox: tuple[np.ndarray, np.ndarray] | None
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None:
    if bbox is None:
        return None
    chunks = _array_chunks_zyx(array)
    if chunks is None:
        return None
    starts, ends = bbox
    chunk_np = np.asarray(chunks, dtype=np.int64)
    first = starts // chunk_np
    last = (ends - 1) // chunk_np
    return tuple((int(first[axis]), int(last[axis])) for axis in range(3))  # type: ignore[return-value]


def _zarr_store_identity(store: Any) -> str:
    url = getattr(store, "_url", None)
    cache_dir = getattr(store, "_cache_dir", None)
    if url is not None:
        return f"{type(store).__module__}.{type(store).__name__}:{url}:{cache_dir}"
    path = getattr(store, "path", None)
    if path is not None:
        return f"{type(store).__module__}.{type(store).__name__}:{path}"
    return f"{type(store).__module__}.{type(store).__name__}:{repr(store)}"


def _is_remote_cached_store(store: Any) -> bool:
    return getattr(store, "_url", None) is not None and getattr(
        store, "_cache_dir", None
    ) is not None


def _chunk_requests_for_bbox(
    array: Any,
    bbox: tuple[np.ndarray, np.ndarray] | None,
    *,
    label: str,
) -> list[ZarrChunkRequest]:
    if bbox is None:
        return []

    channel_index: int | None = None
    spatial_array = array
    zarr_array = array
    if isinstance(array, _SpatialChannelView):
        spatial_array = array
        zarr_array = array.array
        channel_index = int(array.channel_index)

    chunks = _array_chunks_zyx(spatial_array)
    store = getattr(zarr_array, "store", None)
    chunk_key_fn = getattr(zarr_array, "_chunk_key", None)
    if chunks is None or store is None or not callable(chunk_key_fn):
        return []
    if not _is_remote_cached_store(store):
        return []

    starts, ends = bbox
    chunk_np = np.asarray(chunks, dtype=np.int64)
    first = np.asarray(starts, dtype=np.int64) // chunk_np
    last = (np.asarray(ends, dtype=np.int64) - 1) // chunk_np
    if np.any(last < first):
        return []

    zarr_chunks = tuple(int(v) for v in getattr(zarr_array, "chunks", ()))
    zarr_shape = tuple(int(v) for v in getattr(zarr_array, "shape", ()))
    store_identity = _zarr_store_identity(store)
    requests: list[ZarrChunkRequest] = []
    for chunk_z in range(int(first[0]), int(last[0]) + 1):
        for chunk_y in range(int(first[1]), int(last[1]) + 1):
            for chunk_x in range(int(first[2]), int(last[2]) + 1):
                spatial_coords = (chunk_z, chunk_y, chunk_x)
                if len(zarr_shape) == 4:
                    if channel_index is None:
                        channel_index = 0
                    if len(zarr_chunks) != 4 or int(zarr_chunks[0]) <= 0:
                        continue
                    coords = (int(channel_index) // int(zarr_chunks[0]),) + spatial_coords
                elif len(zarr_shape) == 3:
                    coords = spatial_coords
                else:
                    continue
                requests.append(
                    ZarrChunkRequest(
                        store=store,
                        store_identity=store_identity,
                        key=str(chunk_key_fn(coords)),
                        label=str(label),
                    )
                )
    return requests


def _chunk_index_for_zyx(
    array: Any, array_zyx: np.ndarray
) -> tuple[int, int, int] | None:
    chunks = _array_chunks_zyx(array)
    if chunks is None:
        return None
    chunk_np = np.asarray(chunks, dtype=np.int64)
    return tuple(int(v) for v in (np.asarray(array_zyx, dtype=np.int64) // chunk_np))


def _format_optional_array(value: np.ndarray | None) -> str:
    if value is None:
        return "None"
    return repr(tuple(int(v) for v in np.asarray(value, dtype=np.int64)))


def _format_bbox(bbox: tuple[np.ndarray, np.ndarray] | None) -> str:
    if bbox is None:
        return "None"
    starts, ends = bbox
    return f"{_format_optional_array(starts)}:{_format_optional_array(ends)}"


def _read_image_crop_from_patch(
    patch: Any,
    crop_shape: tuple[int, int, int],
    min_corner: np.ndarray,
    max_corner: np.ndarray,
    *,
    image_normalization: str,
) -> np.ndarray:
    return _common_read_volume_crop_from_patch(
        patch,
        crop_shape,
        min_corner,
        max_corner,
        image_normalization=image_normalization,
    )


def _center_crop_array(
    array: np.ndarray | None,
    crop_shape: tuple[int, int, int],
    offset_zyx: np.ndarray,
) -> np.ndarray | None:
    if array is None:
        return None
    offset = np.asarray(offset_zyx, dtype=np.int64)
    z0, y0, x0 = (int(v) for v in offset)
    z_size, y_size, x_size = (int(v) for v in crop_shape)
    return array[z0 : z0 + z_size, y0 : y0 + y_size, x0 : x0 + x_size]


def _stable_seed_from_parts(*parts: Any) -> int:
    digest = hashlib.blake2b(digest_size=16)
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return int.from_bytes(digest.digest(), byteorder="little", signed=False)


def _resolve_fiber_paths(
    dataset_config: dict[str, Any], config: dict[str, Any]
) -> list[Path]:
    paths: list[Path] = []
    for item in dataset_config.get("fiber_paths", []) or []:
        paths.append(Path(_resolve_config_relative_path(item, config)))
    fiber_glob = dataset_config.get("fiber_glob")
    if fiber_glob:
        resolved_glob = _resolve_config_relative_path(fiber_glob, config)
        paths.extend(Path(path) for path in sorted(glob.glob(resolved_glob)))
    if not paths:
        raise ValueError("dataset entry must provide fiber_paths or fiber_glob")
    return paths


class FiberTraceBatchBuilder:
    """Sample single-fiber training batches from VC3D fiber JSON records."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        debug_table_state: FiberTraceDebugTableState | None = None,
    ) -> None:
        self.config = dict(config)
        _reject_valid_mask_threshold_key(self.config, context="top-level config")
        seed_value = self.config.get("seed", 0)
        if seed_value is None:
            seed_value = 0
        self._base_seed = int(seed_value)
        self.crop_size = _as_zyx3(self.config.get("crop_size", 64), key="crop_size")
        self.augmentation_crop_size = _as_zyx3(
            self.config.get("augmentation_crop_size", self.crop_size),
            key="augmentation_crop_size",
        )
        crop_np = np.asarray(self.crop_size, dtype=np.int64)
        aug_crop_np = np.asarray(self.augmentation_crop_size, dtype=np.int64)
        if np.any(aug_crop_np < crop_np):
            raise ValueError(
                "augmentation_crop_size must be greater than or equal to crop_size, "
                f"got augmentation_crop_size={self.augmentation_crop_size} "
                f"crop_size={self.crop_size}"
            )
        if np.any((aug_crop_np - crop_np) % 2 != 0):
            raise ValueError(
                "augmentation_crop_size and crop_size must differ by an even number "
                "on each axis so the post-augmentation center crop remains centered"
            )
        self._post_augmentation_crop_offset = (aug_crop_np - crop_np) // 2
        margin_values = [
            key
            for key in ("control_point_margin_voxels", "cp_margin_voxels")
            if key in self.config
        ]
        if len(margin_values) == 2 and (
            float(self.config[margin_values[0]]) != float(self.config[margin_values[1]])
        ):
            raise ValueError(
                "control_point_margin_voxels and cp_margin_voxels must match "
                "when both are provided"
            )
        configured_margin = (
            self.config[margin_values[0]] if margin_values else None
        )
        max_supported_margin = min((int(size) - 1) // 2 for size in self.crop_size)
        if configured_margin is None:
            self.control_point_margin_voxels = int(
                min(40, max(0, max_supported_margin))
            )
        else:
            margin = int(configured_margin)
            if margin < 0:
                raise ValueError(
                    "control_point_margin_voxels must be >= 0, "
                    f"got {configured_margin}"
                )
            if margin > max_supported_margin:
                raise ValueError(
                    "control_point_margin_voxels must leave at least that many "
                    "voxels on both sides of every crop axis: "
                    f"got {margin} for crop_size={self.crop_size}"
                )
            self.control_point_margin_voxels = margin
        self.batch_size = int(self.config.get("batch_size", 2))
        if self.batch_size <= 0 or self.batch_size % 2 != 0:
            raise ValueError(
                f"batch_size must be a positive even integer, got {self.batch_size}"
            )

        self.image_normalization = str(self.config.get("image_normalization", "zscore"))
        self.positive_direction_jitter_degrees = float(
            self.config.get("positive_direction_jitter_degrees", 30.0)
        )
        self.positive_radius = float(self.config.get("positive_radius", 1.5))
        self.ignore_radius = float(
            self.config.get("ignore_radius", max(3.0, self.positive_radius))
        )
        self.normal_plane_jitter_voxels = float(
            self.config.get("normal_plane_jitter_voxels", 40.0)
        )
        self.normal_perpendicular_jitter_voxels = float(
            self.config.get("normal_perpendicular_jitter_voxels", 10.0)
        )
        self.negative_cone_distance_voxels = float(
            self.config.get("negative_cone_distance_voxels", 30.0)
        )
        for removed_key in (
            "positive_direction_probability",
            "negative_direction_min_degrees",
            "negative_direction_max_degrees",
            "positive_cosine",
            "negative_cosine",
        ):
            if removed_key in self.config:
                raise ValueError(
                    f"{removed_key} was removed; labels are geometry-derived and "
                    "direction conditioning only uses positive_direction_jitter_degrees"
                )
        if not np.isfinite(self.positive_direction_jitter_degrees):
            raise ValueError("positive_direction_jitter_degrees must be finite")
        if self.positive_direction_jitter_degrees < 0.0:
            raise ValueError("positive_direction_jitter_degrees must be >= 0")
        if self.positive_direction_jitter_degrees > 90.0:
            raise ValueError(
                "positive_direction_jitter_degrees is a forward-angle jitter and "
                "must be <= 90 degrees"
            )
        self.random_valid_max_attempts = int(
            self.config.get("random_valid_max_attempts", 256)
        )
        self.random_negative_pool_size = int(
            self.config.get("random_negative_pool_size", 1000)
        )
        if self.random_negative_pool_size <= 0:
            raise ValueError(
                "random_negative_pool_size must be a positive integer, "
                f"got {self.random_negative_pool_size}"
            )
        raw_sample_limit = self.config.get("sample_limit")
        if raw_sample_limit is None:
            self.sample_limit: int | None = None
        else:
            self.sample_limit = int(raw_sample_limit)
            if self.sample_limit <= 0:
                raise ValueError(
                    f"sample_limit must be a positive integer when set, got {raw_sample_limit}"
                )
        self.debug_sampling = bool(
            self.config.get("debug_sampling", False)
            or self.config.get("debug_cache", False)
        )
        self._debug_sampling_limit = int(self.config.get("debug_sampling_limit", 0))
        self._debug_table_state = (
            debug_table_state
            if debug_table_state is not None
            else FiberTraceDebugTableState()
        )
        self._debug_batch_header_every = int(
            self.config.get("debug_batch_header_every", 20)
        )

        datasets = self.config.get("datasets")
        self.records: list[_FiberRecord] = []
        array_records = self.config.get("_array_records")
        if array_records is not None:
            if not isinstance(array_records, list) or not array_records:
                raise ValueError(
                    "_array_records must be a non-empty list when provided"
                )
            for record_raw in array_records:
                record_config = dict(record_raw)
                _reject_valid_mask_threshold_key(
                    record_config, context="_array_records entry"
                )
                volume = np.asarray(record_config["volume"])
                mask = np.asarray(record_config["mask"])
                if volume.ndim != 3 or mask.ndim != 3:
                    raise ValueError("_array_records volume and mask must be 3D arrays")
                if tuple(volume.shape) != tuple(mask.shape):
                    raise ValueError(
                        "_array_records mask shape must match volume shape: "
                        f"volume={tuple(volume.shape)} mask={tuple(mask.shape)}"
                    )
                nx = record_config.get("nx")
                ny = record_config.get("ny")
                if nx is None or ny is None:
                    raise ValueError(
                        "_array_records entries must provide Lasagna nx and ny "
                        "normal channels"
                    )
                else:
                    nx_arr = np.asarray(nx)
                    ny_arr = np.asarray(ny)
                    if nx_arr.ndim != 3 or ny_arr.ndim != 3:
                        raise ValueError("_array_records nx and ny must be 3D arrays")
                    if tuple(nx_arr.shape) != tuple(volume.shape) or tuple(
                        ny_arr.shape
                    ) != tuple(volume.shape):
                        raise ValueError(
                            "_array_records normal channel shapes must match volume shape: "
                            f"volume={tuple(volume.shape)} nx={tuple(nx_arr.shape)} "
                            f"ny={tuple(ny_arr.shape)}"
                        )
                fiber = record_config.get("fiber")
                if fiber is None:
                    fiber = load_vc3d_fiber(record_config["fiber_path"])
                self.records.append(
                    _FiberRecord(
                        fiber=fiber,
                        volume=volume,
                        mask=mask,
                        nx=nx_arr,
                        ny=ny_arr,
                        volume_scale=0,
                        volume_spacing_base=1.0,
                        mask_spacing_base=1.0,
                        nx_spacing_base=None if nx_arr is None else 1.0,
                        ny_spacing_base=None if ny_arr is None else 1.0,
                        dataset_config=record_config,
                    )
                )
        else:
            if not isinstance(datasets, list) or not datasets:
                raise ValueError("config must contain a non-empty datasets list")

        for dataset_config_raw in datasets or []:
            if array_records is not None:
                break
            dataset_config = dict(dataset_config_raw)
            _reject_valid_mask_threshold_key(dataset_config, context="dataset entry")
            _reject_manifestless_dataset_keys(dataset_config)
            lasagna_manifest_path = dataset_config.get("lasagna_manifest_path")
            volume_path = dataset_config.get("base_volume_path")
            if not volume_path:
                raise ValueError("dataset entry missing base_volume_path")
            if not lasagna_manifest_path:
                raise ValueError("dataset entry missing lasagna_manifest_path")

            volume_scale = int(dataset_config.get("base_volume_scale", 0))
            volume_spacing_base = _scale_level_to_spacing_base(volume_scale)
            volume = _open_zarr_array_resolved(
                volume_path,
                scale=volume_scale,
                auth_json_path=dataset_config.get(
                    "base_volume_auth_json", dataset_config.get("volume_auth_json")
                ),
                config=self.config,
            )

            lasagna_volume = _load_lasagna_volume(lasagna_manifest_path, self.config)
            if lasagna_volume.base_shape_zyx is None:
                raise ValueError(
                    "Lasagna manifest must provide base_shape_zyx for fiber "
                    "trace training"
                )
            base_level0 = (
                volume
                if volume_scale == 0
                else _open_zarr_array_resolved(
                    volume_path,
                    scale=0,
                    auth_json_path=dataset_config.get(
                        "base_volume_auth_json",
                        dataset_config.get("volume_auth_json"),
                    ),
                    config=self.config,
                )
            )
            _validate_shape(
                _volume_shape_zyx(base_level0, label="base volume level 0"),
                tuple(lasagna_volume.base_shape_zyx),
                label="base volume level 0",
                context=f"Lasagna manifest {lasagna_manifest_path}",
            )
            _validate_shape(
                _volume_shape_zyx(volume, label="base volume selected level"),
                _omezarr_level_shape(lasagna_volume.base_shape_zyx, volume_scale),
                label="base volume selected level",
                context=(
                    f"base_shape_zyx={lasagna_volume.base_shape_zyx} "
                    f"base_volume_scale={volume_scale}"
                ),
            )
            try:
                mask, mask_spacing_base = _open_manifest_channel(
                    lasagna_volume,
                    "grad_mag",
                    auth_json_path=dataset_config.get("lasagna_auth_json"),
                    config=self.config,
                )
            except KeyError as exc:
                raise ValueError(
                    "Lasagna manifest missing required grad_mag channel"
                ) from exc
            try:
                nx, nx_spacing_base = _open_manifest_channel(
                    lasagna_volume,
                    "nx",
                    auth_json_path=dataset_config.get("lasagna_auth_json"),
                    config=self.config,
                )
                ny, ny_spacing_base = _open_manifest_channel(
                    lasagna_volume,
                    "ny",
                    auth_json_path=dataset_config.get("lasagna_auth_json"),
                    config=self.config,
                )
            except KeyError as exc:
                raise ValueError(
                    "Lasagna manifest missing required nx/ny normal channels"
                ) from exc
            if abs(float(nx_spacing_base) - float(ny_spacing_base)) > 1e-6:
                raise ValueError(
                    "Lasagna nx and ny normal channel spacing mismatch: "
                    f"nx={nx_spacing_base} ny={ny_spacing_base}"
                )

            for fiber_path in _resolve_fiber_paths(dataset_config, self.config):
                self.records.append(
                    _FiberRecord(
                        fiber=load_vc3d_fiber(fiber_path),
                        volume=volume,
                        mask=mask,
                        nx=nx,
                        ny=ny,
                        volume_scale=volume_scale,
                        volume_spacing_base=volume_spacing_base,
                        mask_spacing_base=mask_spacing_base,
                        nx_spacing_base=nx_spacing_base,
                        ny_spacing_base=ny_spacing_base,
                        dataset_config=dataset_config,
                    )
                )
        if not self.records:
            raise ValueError("no fiber records were loaded")
        self._random_negative_center_pools: dict[tuple[Any, ...], np.ndarray] = {}
        self._record_random_negative_pool_keys: list[tuple[Any, ...]] = []
        self._build_random_negative_center_pools()
        self._sample_batch_count = 0
        self.last_cache_stats: Any | None = None
        self.last_sample_ms = 0.0

    def __len__(self) -> int:
        return len(self.records)

    def _rng_for(
        self,
        purpose: str,
        *,
        iteration: int,
        batch_slot: int = 0,
        record_index: int = 0,
        extra: int = 0,
    ) -> np.random.Generator:
        return np.random.default_rng(
            _stable_seed_from_parts(
                self._base_seed,
                purpose,
                int(iteration),
                int(batch_slot),
                int(record_index),
                int(extra),
            )
        )

    def _random_negative_pool_key(self, record: _FiberRecord) -> tuple[Any, ...]:
        return (
            id(record.volume),
            id(record.mask),
            id(record.nx),
            id(record.ny),
            float(record.volume_spacing_base),
            float(record.mask_spacing_base),
            None if record.nx_spacing_base is None else float(record.nx_spacing_base),
            None if record.ny_spacing_base is None else float(record.ny_spacing_base),
        )

    def _build_random_negative_center_pools(self) -> None:
        raw_seed = self.config.get("random_negative_seed", self._base_seed)
        negative_seed = 0 if raw_seed is None else int(raw_seed)
        for record_index, record in enumerate(self.records):
            key = self._random_negative_pool_key(record)
            self._record_random_negative_pool_keys.append(key)
            if key in self._random_negative_center_pools:
                continue
            pool_index = len(self._random_negative_center_pools)
            rng = np.random.default_rng(
                _stable_seed_from_parts(
                    negative_seed,
                    "random_negative_pool",
                    pool_index,
                    record_index,
                )
            )
            centers = [
                self._sample_random_valid_center(record, rng=rng)
                for _ in range(self.random_negative_pool_size)
            ]
            self._random_negative_center_pools[key] = np.stack(centers, axis=0)

    def _random_negative_center_for_patch(
        self,
        *,
        record_index: int,
        batch_ordinal: int,
        negative_index: int,
    ) -> np.ndarray:
        key = self._record_random_negative_pool_keys[int(record_index)]
        pool = self._random_negative_center_pools[key]
        pool_index = (
            int(record_index) * 1_000_003
            + int(batch_ordinal) * 65_537
            + int(negative_index)
        ) % int(pool.shape[0])
        return pool[pool_index].astype(np.int64, copy=True)

    def _limited_batch_ordinal(self, batch_ordinal: int) -> int:
        batch_ordinal = int(batch_ordinal)
        if self.sample_limit is None:
            return batch_ordinal
        if batch_ordinal <= 0:
            return batch_ordinal % int(self.sample_limit)
        return ((batch_ordinal - 1) % int(self.sample_limit)) + 1

    def _control_point_local_zyx(
        self,
        *,
        batch_ordinal: int,
        patch_index: int,
        record_index: int,
        control_index: int,
    ) -> np.ndarray:
        crop_np = np.asarray(self.crop_size, dtype=np.int64)
        margin = int(self.control_point_margin_voxels)
        low = np.full(3, margin, dtype=np.int64)
        high = crop_np - margin - 1
        rng = self._rng_for(
            "control_point_crop_offset",
            iteration=batch_ordinal,
            batch_slot=patch_index,
            record_index=record_index,
            extra=control_index,
        )
        choose_high = rng.integers(0, 2, size=3, dtype=np.int64).astype(bool)
        return np.where(choose_high, high, low).astype(np.int64, copy=False)

    def _control_center_zyx(
        self, record: _FiberRecord, control_index: int
    ) -> np.ndarray:
        center_xyz = record.fiber.control_points_xyz[int(control_index)]
        return np.rint(
            xyz_to_zyx(center_xyz) / float(record.volume_spacing_base)
        ).astype(np.int64)

    def _batch_crop_specs(
        self,
        *,
        record_index: int | None,
        batch_ordinal: int,
    ) -> list[dict[str, Any]]:
        if record_index is None:
            record_rng = self._rng_for("record", iteration=batch_ordinal)
            selected_record_index = int(record_rng.integers(0, len(self.records)))
        else:
            selected_record_index = int(record_index)
        record = self.records[selected_record_index]
        gt_count = self.batch_size // 2
        random_negative_count = self.batch_size - gt_count
        specs: list[dict[str, Any]] = []

        for patch_index in range(gt_count):
            control_count = int(record.fiber.control_points_xyz.shape[0])
            control_index = int(
                self._rng_for(
                    "control_index",
                    iteration=batch_ordinal,
                    batch_slot=patch_index,
                    record_index=selected_record_index,
                ).integers(0, control_count)
            )
            local_zyx = self._control_point_local_zyx(
                batch_ordinal=batch_ordinal,
                patch_index=patch_index,
                record_index=selected_record_index,
                control_index=control_index,
            )
            specs.append(
                {
                    "record": record,
                    "record_index": selected_record_index,
                    "patch_index": patch_index,
                    "crop_kind": "gt_control",
                    "control_index": control_index,
                    "center_zyx": self._control_center_zyx(record, control_index),
                    "sample_local_zyx": local_zyx,
                }
            )

        for negative_index in range(random_negative_count):
            patch_index = gt_count + negative_index
            specs.append(
                {
                    "record": record,
                    "record_index": selected_record_index,
                    "patch_index": patch_index,
                    "crop_kind": "random_negative",
                    "control_index": None,
                    "center_zyx": self._random_negative_center_for_patch(
                        record_index=selected_record_index,
                        batch_ordinal=batch_ordinal,
                        negative_index=negative_index,
                    ),
                    "sample_local_zyx": np.asarray(self.crop_size, dtype=np.int64)
                    // 2,
                }
            )
        return specs

    def _debug_batch_row(
        self,
        *,
        iteration: int | None,
        total_ms: float,
        cache_stats: Any,
    ) -> None:
        if not self.debug_sampling:
            return
        state = self._debug_table_state
        if self._debug_sampling_limit > 0 and state.rows >= self._debug_sampling_limit:
            return
        every = max(1, int(self._debug_batch_header_every))
        if not state.legend_printed:
            print(
                "fiber_trace batch columns: it=iteration data=batch data-loading ms "
                "hit/dl/mis=cache events hms/dms=cache-hit/download ms "
                "hMiB/s/dMiB/s=cache-hit/download throughput",
                flush=True,
            )
            state.legend_printed = True
        if state.rows % every == 0:
            print(
                "   it      data  hit  dl mis      hms      dms   hMiB/s   dMiB/s",
                flush=True,
            )
        state.rows += 1
        cache_mib = float(getattr(cache_stats, "cache_hit_bytes", 0)) / (1024.0 * 1024.0)
        download_mib = float(getattr(cache_stats, "download_bytes", 0)) / (1024.0 * 1024.0)
        cache_ms = float(getattr(cache_stats, "cache_hit_ms", 0.0))
        download_ms = float(getattr(cache_stats, "download_ms", 0.0))
        cache_mib_s = cache_mib / max(cache_ms / 1000.0, 1e-9) if cache_mib > 0.0 else 0.0
        download_mib_s = (
            download_mib / max(download_ms / 1000.0, 1e-9)
            if download_mib > 0.0
            else 0.0
        )

        print(
            f"{'-' if iteration is None else int(iteration):>5} "
            f"{total_ms:9.1f} "
            f"{int(getattr(cache_stats, 'cache_hits', 0)):3d} "
            f"{int(getattr(cache_stats, 'downloads', 0)):2d} "
            f"{int(getattr(cache_stats, 'missing', 0)) + int(getattr(cache_stats, 'negative_hits', 0)):3d} "
            f"{cache_ms:8.1f} {download_ms:8.1f} "
            f"{cache_mib_s:8.1f} {download_mib_s:8.1f}",
            flush=True,
        )

    def _sample_record(
        self,
        record_index: int | None,
        *,
        batch_ordinal: int = 0,
    ) -> _FiberRecord:
        if record_index is not None:
            return self.records[int(record_index)]
        rng = self._rng_for("record", iteration=batch_ordinal)
        return self.records[int(rng.integers(0, len(self.records)))]

    def _channel_value_info(
        self,
        array: Any,
        sample_zyx: np.ndarray,
        *,
        sample_spacing_base: float,
        array_spacing_base: float,
    ) -> dict[str, Any]:
        sample_zyx_f = np.asarray(sample_zyx, dtype=np.float64)
        array_zyx = np.floor(
            sample_zyx_f * float(sample_spacing_base) / float(array_spacing_base)
        ).astype(np.int64)
        shape = np.asarray(array.shape, dtype=np.int64)
        in_bounds = bool(np.all(array_zyx >= 0) and np.all(array_zyx < shape))
        info: dict[str, Any] = {
            "array_zyx": array_zyx,
            "shape": tuple(int(v) for v in shape),
            "chunks": _array_chunks_zyx(array),
            "chunk_index": _chunk_index_for_zyx(array, array_zyx)
            if in_bounds
            else None,
            "in_bounds": in_bounds,
            "value": None,
            "read_ok": False,
        }
        if not in_bounds:
            return info
        value = np.asarray(
            array[
                array_zyx[0] : array_zyx[0] + 1,
                array_zyx[1] : array_zyx[1] + 1,
                array_zyx[2] : array_zyx[2] + 1,
            ]
        )
        if value.size == 0:
            return info
        info["value"] = value.reshape(-1)[0]
        info["read_ok"] = True
        return info

    def _normal_sample_info(
        self, record: _FiberRecord, center_zyx: np.ndarray
    ) -> dict[str, Any]:
        if record.nx is None or record.ny is None:
            raise ValueError("Lasagna nx/ny normal channels are required")
        if record.nx_spacing_base is None or record.ny_spacing_base is None:
            raise ValueError("Lasagna nx/ny spacing metadata is required")

        center = np.asarray(center_zyx, dtype=np.int64)
        nx_info = self._channel_value_info(
            record.nx,
            center,
            sample_spacing_base=record.volume_spacing_base,
            array_spacing_base=record.nx_spacing_base,
        )
        ny_info = self._channel_value_info(
            record.ny,
            center,
            sample_spacing_base=record.volume_spacing_base,
            array_spacing_base=record.ny_spacing_base,
        )
        reason = ""
        normal_xyz = np.zeros(3, dtype=np.float32)
        normal_valid = False
        if not bool(nx_info["read_ok"]) or not bool(ny_info["read_ok"]):
            reason = "mapped outside nx/ny volume"
        else:
            nx_vox = np.asarray([nx_info["value"]])
            ny_vox = np.asarray([ny_info["value"]])
            normal_arr, valid_arr = decode_lasagna_normals_xyz(nx_vox, ny_vox)
            normal_xyz = normal_arr.reshape(-1, 3)[0]
            normal_valid = bool(valid_arr.reshape(-1)[0])
            if not normal_valid:
                reason = "Lasagna normal decode produced non-finite or zero vector"

        return {
            "center_zyx": center,
            "center_xyz": zyx_to_xyz(
                center.astype(np.float32) * float(record.volume_spacing_base)
            ),
            "volume_spacing_base": record.volume_spacing_base,
            "nx_spacing_base": record.nx_spacing_base,
            "ny_spacing_base": record.ny_spacing_base,
            "nx": nx_info,
            "ny": ny_info,
            "normal_xyz": normal_xyz,
            "normal_valid": normal_valid,
            "reason": reason,
        }

    def _format_normal_sample_info(self, info: dict[str, Any]) -> str:
        nx = info.get("nx", {})
        ny = info.get("ny", {})
        return (
            f"center_zyx={_format_optional_array(info.get('center_zyx'))} "
            f"center_xyz={tuple(float(v) for v in np.asarray(info.get('center_xyz', []), dtype=np.float32))} "
            f"volume_spacing_base={info.get('volume_spacing_base')} "
            f"nx_zyx={_format_optional_array(nx.get('array_zyx'))} "
            f"ny_zyx={_format_optional_array(ny.get('array_zyx'))} "
            f"nx_value={nx.get('value')} ny_value={ny.get('value')} "
            f"nx_shape={nx.get('shape')} ny_shape={ny.get('shape')} "
            f"nx_chunks={nx.get('chunks')} ny_chunks={ny.get('chunks')} "
            f"nx_chunk={nx.get('chunk_index')} ny_chunk={ny.get('chunk_index')} "
            f"normal_xyz={tuple(float(v) for v in np.asarray(info.get('normal_xyz', []), dtype=np.float32))} "
            f"reason={info.get('reason')!r}"
        )

    def _normal_at_zyx(
        self, record: _FiberRecord, center_zyx: np.ndarray
    ) -> tuple[np.ndarray | None, bool]:
        info = self._normal_sample_info(record, center_zyx)
        return info["normal_xyz"], bool(info["normal_valid"])

    def _sample_random_valid_center(
        self, record: _FiberRecord, *, rng: np.random.Generator
    ) -> np.ndarray:
        shape = np.asarray(record.volume.shape, dtype=np.int64)
        for _ in range(self.random_valid_max_attempts):
            center = np.asarray(
                [rng.integers(0, int(size)) for size in shape], dtype=np.int64
            )
            value, mask_ok = _value_at_sample_zyx(
                record.mask,
                center,
                sample_spacing_base=record.volume_spacing_base,
                array_spacing_base=record.mask_spacing_base,
            )
            if mask_ok and float(value) > 0.0:
                _, normal_valid = self._normal_at_zyx(record, center)
                if normal_valid:
                    return center

        raise ValueError(
            "could not sample a random valid center from mask/grad-mag and nx/ny "
            f"after {self.random_valid_max_attempts} attempts"
        )

    def _sample_jittered_conditioning(
        self,
        tangent_xyz: np.ndarray,
        *,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, str]:
        fw = perturb_direction(
            tangent_xyz,
            min_angle_degrees=0.0,
            max_angle_degrees=self.positive_direction_jitter_degrees,
            rng=rng,
        )
        return (
            fw.astype(np.float32, copy=False),
            "gt_jitter",
        )

    def _sample_random_conditioning(
        self, *, rng: np.random.Generator
    ) -> tuple[np.ndarray, str]:
        return random_unit_vector(rng).astype(np.float32, copy=False), "random"

    def _sample_crop_base(
        self,
        record: _FiberRecord,
        *,
        crop_kind: str,
        control_index: int | None = None,
        center_zyx: np.ndarray | None = None,
        control_point_local_zyx: np.ndarray | None = None,
    ) -> dict[str, Any]:
        fiber = record.fiber
        if center_zyx is not None:
            center = np.asarray(center_zyx, dtype=np.int64)
            if control_point_local_zyx is None:
                final_local_zyx = np.asarray(self.crop_size, dtype=np.int64) // 2
            else:
                final_local_zyx = np.asarray(control_point_local_zyx, dtype=np.int64)
        elif crop_kind == "gt_control":
            if control_index is None:
                control_index = 0
            center = self._control_center_zyx(record, int(control_index))
            if control_point_local_zyx is None:
                final_local_zyx = np.asarray(self.crop_size, dtype=np.int64) // 2
            else:
                final_local_zyx = np.asarray(control_point_local_zyx, dtype=np.int64)
        elif crop_kind == "random_negative":
            raise ValueError("random_negative center must be selected before loading")
        else:
            raise ValueError(f"unsupported crop_kind {crop_kind!r}")
        center_normal_info = self._normal_sample_info(record, center)
        if not bool(center_normal_info["normal_valid"]):
            raise ValueError(
                f"sampled {crop_kind} center has unusable Lasagna nx/ny: "
                f"{self._format_normal_sample_info(center_normal_info)}"
            )
        center_normal_xyz = center_normal_info["normal_xyz"]

        read_crop_size_np = np.asarray(self.augmentation_crop_size, dtype=np.int64)
        crop_offset = self._post_augmentation_crop_offset
        read_min_corner = center - final_local_zyx - crop_offset
        read_max_corner = read_min_corner + read_crop_size_np
        min_corner = read_min_corner + crop_offset
        patch = SimpleNamespace(volume=record.volume, scale=record.volume_scale)
        volume_crop = _read_image_crop_from_patch(
            patch,
            self.augmentation_crop_size,
            read_min_corner,
            read_max_corner,
            image_normalization=self.image_normalization,
        )
        mask_crop = _read_scaled_channel_crop(
            record.mask,
            self.augmentation_crop_size,
            read_min_corner,
            dst_spacing_base=record.volume_spacing_base,
            src_spacing_base=record.mask_spacing_base,
        )
        if record.nx is None or record.ny is None:
            raise ValueError("Lasagna nx/ny normal channels are required")
        else:
            if record.nx_spacing_base is None or record.ny_spacing_base is None:
                raise ValueError("Lasagna nx/ny spacing metadata is required")
            nx_crop = _read_scaled_channel_crop(
                record.nx,
                self.augmentation_crop_size,
                read_min_corner,
                dst_spacing_base=record.volume_spacing_base,
                src_spacing_base=record.nx_spacing_base,
            )
            ny_crop = _read_scaled_channel_crop(
                record.ny,
                self.augmentation_crop_size,
                read_min_corner,
                dst_spacing_base=record.volume_spacing_base,
                src_spacing_base=record.ny_spacing_base,
            )

        volume_crop = _center_crop_array(volume_crop, self.crop_size, crop_offset)
        mask_crop = _center_crop_array(mask_crop, self.crop_size, crop_offset)
        nx_crop = _center_crop_array(nx_crop, self.crop_size, crop_offset)
        ny_crop = _center_crop_array(ny_crop, self.crop_size, crop_offset)

        tangent_xyz = tangent_at_point(
            fiber.line_points_xyz,
            zyx_to_xyz(center.astype(np.float32) * float(record.volume_spacing_base)),
        )

        return {
            "volume": volume_crop.astype(np.float32, copy=False),
            "mask_values": mask_crop,
            "nx_values": nx_crop,
            "ny_values": ny_crop,
            "center_normal_xyz": center_normal_xyz,
            "tangent_xyz": tangent_xyz,
            "origin": min_corner.astype(np.int64, copy=False),
            "sample_local_zyx": (center - min_corner).astype(np.int64, copy=False),
            "crop_kind": crop_kind,
        }

    def _make_conditioned_sample(
        self,
        record: _FiberRecord,
        crop_base: dict[str, Any],
        *,
        positive_target_id: int,
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        if crop_base["crop_kind"] == "gt_control":
            cond_fw_xyz, direction_kind = self._sample_jittered_conditioning(
                crop_base["tangent_xyz"],
                rng=rng,
            )
        else:
            cond_fw_xyz, direction_kind = self._sample_random_conditioning(rng=rng)

        return {
            "volume": crop_base["volume"],
            "mask_values": crop_base["mask_values"],
            "nx_values": crop_base["nx_values"],
            "ny_values": crop_base["ny_values"],
            "center_normal_xyz": crop_base["center_normal_xyz"],
            "cond_fw_xyz": cond_fw_xyz,
            "origin": crop_base["origin"],
            "sample_local_zyx": crop_base["sample_local_zyx"],
            "line_points_xyz": record.fiber.line_points_xyz
            / float(record.volume_spacing_base),
            "positive_target_id": positive_target_id,
            "crop_kind": crop_base["crop_kind"],
            "direction_kind": direction_kind,
        }

    def prefetch_chunk_requests_for_iteration(
        self,
        *,
        iteration: int,
        record_index: int | None = None,
    ) -> list[ZarrChunkRequest]:
        batch_ordinal = self._limited_batch_ordinal(int(iteration))
        specs = self._batch_crop_specs(
            record_index=record_index,
            batch_ordinal=batch_ordinal,
        )
        requests: list[ZarrChunkRequest] = []
        read_crop_size_np = np.asarray(self.augmentation_crop_size, dtype=np.int64)
        crop_offset = self._post_augmentation_crop_offset
        for spec in specs:
            record = spec["record"]
            center = np.asarray(spec["center_zyx"], dtype=np.int64)
            sample_local_zyx = np.asarray(spec["sample_local_zyx"], dtype=np.int64)
            read_min_corner = center - sample_local_zyx - crop_offset
            read_shape = tuple(int(v) for v in read_crop_size_np)

            requests.extend(
                _chunk_requests_for_bbox(
                    record.volume,
                    _raw_source_bbox_zyx(
                        record.volume,
                        read_shape,
                        read_min_corner,
                    ),
                    label="base",
                )
            )
            requests.extend(
                _chunk_requests_for_bbox(
                    record.mask,
                    _scaled_source_bbox_zyx(
                        record.mask,
                        read_shape,
                        read_min_corner,
                        dst_spacing_base=record.volume_spacing_base,
                        src_spacing_base=record.mask_spacing_base,
                    ),
                    label="grad_mag",
                )
            )
            if record.nx is not None and record.nx_spacing_base is not None:
                requests.extend(
                    _chunk_requests_for_bbox(
                        record.nx,
                        _scaled_source_bbox_zyx(
                            record.nx,
                            read_shape,
                            read_min_corner,
                            dst_spacing_base=record.volume_spacing_base,
                            src_spacing_base=record.nx_spacing_base,
                        ),
                        label="nx",
                    )
                )
            if record.ny is not None and record.ny_spacing_base is not None:
                requests.extend(
                    _chunk_requests_for_bbox(
                        record.ny,
                        _scaled_source_bbox_zyx(
                            record.ny,
                            read_shape,
                            read_min_corner,
                            dst_spacing_base=record.volume_spacing_base,
                            src_spacing_base=record.ny_spacing_base,
                        ),
                        label="ny",
                    )
                )
        return requests

    def sample_batch(
        self,
        *,
        record_index: int | None = None,
        iteration: int | None = None,
        debug: bool | None = None,
        emit_debug_row: bool = True,
    ) -> FiberTraceBatch:
        trace_enabled = bool(
            self.debug_sampling if debug is None else self.debug_sampling and debug
        )
        self.last_cache_stats = None
        self.last_sample_ms = 0.0
        if trace_enabled:
            begin_zarr_cache_trace()
        batch_t0 = time.perf_counter()
        requested_batch_ordinal = (
            int(iteration) if iteration is not None else self._sample_batch_count
        )
        batch_ordinal = self._limited_batch_ordinal(requested_batch_ordinal)
        specs = self._batch_crop_specs(
            record_index=record_index,
            batch_ordinal=batch_ordinal,
        )
        selected_record_index = int(specs[0]["record_index"])
        record = specs[0]["record"]

        samples: list[dict[str, Any]] = []
        for spec in specs:
            patch_index = int(spec["patch_index"])
            control_index = spec["control_index"]
            crop_base = self._sample_crop_base(
                record,
                crop_kind=str(spec["crop_kind"]),
                control_index=control_index,
                center_zyx=np.asarray(spec["center_zyx"], dtype=np.int64),
                control_point_local_zyx=np.asarray(
                    spec["sample_local_zyx"], dtype=np.int64
                ),
            )
            direction_rng = self._rng_for(
                "direction_conditioning",
                iteration=batch_ordinal,
                batch_slot=patch_index,
                record_index=selected_record_index,
                extra=-1 if control_index is None else int(control_index),
            )
            samples.append(
                self._make_conditioned_sample(
                    record,
                    crop_base,
                    positive_target_id=selected_record_index,
                    rng=direction_rng,
                )
            )
        fiber_path = str(record.fiber.path) if record.fiber.path is not None else ""
        sample_count = len(samples)
        crop_shape = self.crop_size
        nx_values = None
        ny_values = None
        if all(s["nx_values"] is not None for s in samples):
            nx_values = torch.from_numpy(
                np.stack([s["nx_values"] for s in samples], axis=0)
            )
            ny_values = torch.from_numpy(
                np.stack([s["ny_values"] for s in samples], axis=0)
            )

        batch = FiberTraceBatch(
            volume=torch.from_numpy(
                np.stack([s["volume"] for s in samples], axis=0)[:, None]
            ).to(torch.float32),
            mask_values=torch.from_numpy(
                np.stack([s["mask_values"] for s in samples], axis=0)
            ),
            nx_values=nx_values,
            ny_values=ny_values,
            valid_mask=torch.zeros((sample_count,) + crop_shape, dtype=torch.bool),
            labels=torch.full(
                (sample_count,) + crop_shape, int(IGNORE_INDEX), dtype=torch.long
            ),
            target_id=torch.full(
                (sample_count,) + crop_shape, int(IGNORE_ID), dtype=torch.long
            ),
            target_fw_xyz=torch.zeros(
                (sample_count, 3) + crop_shape, dtype=torch.float32
            ),
            center_normal_xyz=torch.from_numpy(
                np.stack([s["center_normal_xyz"] for s in samples], axis=0)
            ).to(torch.float32),
            cond_fw_xyz=torch.from_numpy(
                np.stack([s["cond_fw_xyz"] for s in samples], axis=0)
            ).to(torch.float32),
            crop_origin_zyx=torch.from_numpy(
                np.stack([s["origin"] for s in samples], axis=0)
            ).to(torch.long),
            sample_local_zyx=torch.from_numpy(
                np.stack([s["sample_local_zyx"] for s in samples], axis=0)
            ).to(torch.long),
            line_points_xyz=torch.from_numpy(
                np.stack([s["line_points_xyz"] for s in samples], axis=0)
            ).to(torch.float32),
            positive_target_id=torch.tensor(
                [int(s["positive_target_id"]) for s in samples], dtype=torch.long
            ),
            crop_kinds=tuple(str(s["crop_kind"]) for s in samples),
            fiber_paths=tuple(fiber_path for _ in samples),
            direction_kinds=tuple(str(s["direction_kind"]) for s in samples),
        )
        cache_stats = end_zarr_cache_trace() if trace_enabled else None
        self.last_cache_stats = cache_stats
        self.last_sample_ms = (time.perf_counter() - batch_t0) * 1000.0
        if trace_enabled and emit_debug_row:
            self._debug_batch_row(
                iteration=iteration,
                total_ms=self.last_sample_ms,
                cache_stats=cache_stats,
            )
        self._sample_batch_count += 1
        return batch
