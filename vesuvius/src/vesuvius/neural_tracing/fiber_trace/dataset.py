from __future__ import annotations

import glob
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
    construct_up_vector,
    decode_lasagna_normals_xyz,
    perturb_direction,
    tangent_at_point,
    xyz_to_zyx,
    zyx_to_xyz,
)
from vesuvius.neural_tracing.fiber_trace.labels import IGNORE_ID, IGNORE_INDEX
from vesuvius.neural_tracing.datasets.common import (
    _read_volume_crop_from_patch as _common_read_volume_crop_from_patch,
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
    target_up_xyz: torch.Tensor
    target_up_valid: torch.Tensor
    cond_fw_xyz: torch.Tensor
    cond_up_xyz: torch.Tensor
    crop_origin_zyx: torch.Tensor
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
            target_up_xyz=self.target_up_xyz.to(device),
            target_up_valid=self.target_up_valid.to(device),
            cond_fw_xyz=self.cond_fw_xyz.to(device),
            cond_up_xyz=self.cond_up_xyz.to(device),
            crop_origin_zyx=self.crop_origin_zyx.to(device),
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
        target_up_xyz: torch.Tensor,
        target_up_valid: torch.Tensor,
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
            target_up_xyz=target_up_xyz,
            target_up_valid=target_up_valid,
            cond_fw_xyz=self.cond_fw_xyz,
            cond_up_xyz=self.cond_up_xyz,
            crop_origin_zyx=self.crop_origin_zyx,
            line_points_xyz=self.line_points_xyz,
            positive_target_id=self.positive_target_id,
            crop_kinds=self.crop_kinds,
            fiber_paths=self.fiber_paths,
            direction_kinds=self.direction_kinds,
        )


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
        self, config: dict[str, Any], *, rng: np.random.Generator | None = None
    ) -> None:
        self.config = dict(config)
        _reject_valid_mask_threshold_key(self.config, context="top-level config")
        self.rng = (
            rng
            if rng is not None
            else np.random.default_rng(self.config.get("seed", None))
        )
        self.crop_size = _as_zyx3(self.config.get("crop_size", 64), key="crop_size")
        self.batch_size = int(self.config.get("batch_size", 2))
        if self.batch_size <= 0 or self.batch_size % 2 != 0:
            raise ValueError(
                f"batch_size must be a positive even integer, got {self.batch_size}"
            )

        self.image_normalization = str(self.config.get("image_normalization", "zscore"))
        self.positive_direction_probability = float(
            self.config.get("positive_direction_probability", 0.5)
        )
        self.positive_direction_jitter_degrees = float(
            self.config.get("positive_direction_jitter_degrees", 30.0)
        )
        self.negative_direction_min_degrees = float(
            self.config.get("negative_direction_min_degrees", 60.0)
        )
        self.negative_direction_max_degrees = float(
            self.config.get("negative_direction_max_degrees", 90.0)
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
        self.positive_cosine = float(
            self.config.get("positive_cosine", np.cos(np.deg2rad(30.0)))
        )
        self.negative_cosine = float(
            self.config.get("negative_cosine", np.cos(np.deg2rad(60.0)))
        )
        if not np.isfinite(self.positive_direction_jitter_degrees):
            raise ValueError("positive_direction_jitter_degrees must be finite")
        if self.positive_direction_jitter_degrees < 0.0:
            raise ValueError("positive_direction_jitter_degrees must be >= 0")
        if self.positive_direction_jitter_degrees > 90.0:
            raise ValueError(
                "positive_direction_jitter_degrees is a folded-frame angle and "
                "must be <= 90"
            )
        if (
            not np.isfinite(self.negative_direction_min_degrees)
            or not np.isfinite(self.negative_direction_max_degrees)
        ):
            raise ValueError("negative direction folded-frame degrees must be finite")
        if self.negative_direction_min_degrees < 0.0:
            raise ValueError("negative_direction_min_degrees must be >= 0")
        if self.negative_direction_max_degrees > 90.0:
            raise ValueError(
                "negative_direction_max_degrees is a folded-frame maximum and "
                "must be <= 90"
            )
        if self.negative_direction_min_degrees > self.negative_direction_max_degrees:
            raise ValueError(
                "negative_direction_min_degrees must be <= "
                "negative_direction_max_degrees"
            )
        self.random_valid_max_attempts = int(
            self.config.get("random_valid_max_attempts", 256)
        )
        self.allow_arbitrary_up_fallback = bool(
            self.config.get("allow_arbitrary_up_fallback", False)
        )
        self.degenerate_up_policy = str(
            self.config.get("degenerate_up_policy", "invalid")
        )
        if self.degenerate_up_policy not in {"invalid", "raise"}:
            raise ValueError(
                "degenerate_up_policy must be 'invalid' or 'raise', "
                f"got {self.degenerate_up_policy!r}"
            )
        self.debug_sampling = bool(self.config.get("debug_sampling", False))
        self._debug_sampling_count = 0
        self._debug_sampling_limit = int(self.config.get("debug_sampling_limit", 0))

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
                    if not self.allow_arbitrary_up_fallback:
                        raise ValueError(
                            "_array_records entries must provide Lasagna nx and ny "
                            "normal channels"
                        )
                    nx_arr = None
                    ny_arr = None
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
            if self.allow_arbitrary_up_fallback:
                nx = None
                ny = None
                nx_spacing_base = None
                ny_spacing_base = None
            else:
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

    def __len__(self) -> int:
        return len(self.records)

    def _debug_sampling_print(self, message: str) -> None:
        if not self.debug_sampling:
            return
        if self._debug_sampling_limit > 0:
            if self._debug_sampling_count >= self._debug_sampling_limit:
                return
            self._debug_sampling_count += 1
        print(f"[fiber_trace:sample] {message}", flush=True)

    def _sample_record(self, record_index: int | None) -> _FiberRecord:
        if record_index is not None:
            return self.records[int(record_index)]
        return self.records[int(self.rng.integers(0, len(self.records)))]

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
            if self.allow_arbitrary_up_fallback:
                return {
                    "normal_xyz": None,
                    "normal_valid": True,
                    "reason": "arbitrary_up_fallback",
                }
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

    def _sample_random_valid_center(self, record: _FiberRecord) -> np.ndarray:
        shape = np.asarray(record.volume.shape, dtype=np.int64)
        for _ in range(self.random_valid_max_attempts):
            center = np.asarray(
                [self.rng.integers(0, int(size)) for size in shape], dtype=np.int64
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
        normal_xyz: np.ndarray | None,
        *,
        min_angle_degrees: float,
        max_angle_degrees: float,
        direction_kind: str,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        fw = None
        for _ in range(32):
            candidate = perturb_direction(
                tangent_xyz,
                min_angle_degrees=min_angle_degrees,
                max_angle_degrees=max_angle_degrees,
                rng=self.rng,
            )
            try:
                construct_up_vector(
                    candidate,
                    normal_xyz,
                    allow_arbitrary_up_fallback=self.allow_arbitrary_up_fallback,
                )
            except ValueError:
                continue
            fw = candidate
            break
        if fw is None:
            raise ValueError(
                f"could not sample a {direction_kind} conditioning direction with "
                "a valid Lasagna normal up vector"
            )
        up = construct_up_vector(
            fw,
            normal_xyz,
            allow_arbitrary_up_fallback=self.allow_arbitrary_up_fallback,
        )
        return (
            fw.astype(np.float32, copy=False),
            up.astype(np.float32, copy=False),
            direction_kind,
        )

    def _sample_crop_base(
        self,
        record: _FiberRecord,
        *,
        crop_kind: str,
        control_index: int | None = None,
    ) -> dict[str, Any]:
        fiber = record.fiber
        if crop_kind == "gt_control":
            controls_xyz = fiber.control_points_xyz
            if control_index is None:
                control_index = int(self.rng.integers(0, controls_xyz.shape[0]))
            center_xyz = controls_xyz[int(control_index)]
            center = np.rint(
                xyz_to_zyx(center_xyz) / float(record.volume_spacing_base)
            ).astype(np.int64)
        elif crop_kind == "random_valid":
            center = self._sample_random_valid_center(record)
        else:
            raise ValueError(f"unsupported crop_kind {crop_kind!r}")
        center_normal_info = self._normal_sample_info(record, center)
        self._debug_sampling_print(
            f"sampled center crop_kind={crop_kind!r} control_index={control_index} "
            f"fiber_path={record.fiber.path} "
            f"{self._format_normal_sample_info(center_normal_info)}"
        )
        if not bool(center_normal_info["normal_valid"]):
            raise ValueError(
                f"sampled {crop_kind} center has unusable Lasagna nx/ny: "
                f"{self._format_normal_sample_info(center_normal_info)}"
            )
        center_normal_xyz = center_normal_info["normal_xyz"]

        crop_size_np = np.asarray(self.crop_size, dtype=np.int64)
        min_corner = center - (crop_size_np // 2)
        max_corner = min_corner + crop_size_np
        patch = SimpleNamespace(volume=record.volume, scale=record.volume_scale)
        volume_bbox = _raw_source_bbox_zyx(record.volume, self.crop_size, min_corner)
        volume_t0 = time.perf_counter()
        volume_crop = _read_image_crop_from_patch(
            patch,
            self.crop_size,
            min_corner,
            max_corner,
            image_normalization=self.image_normalization,
        )
        self._debug_sampling_print(
            f"read crop field='base' crop_kind={crop_kind!r} "
            f"origin_zyx={_format_optional_array(min_corner)} "
            f"shape={self.crop_size} src_bbox={_format_bbox(volume_bbox)} "
            f"chunks={_chunk_ranges_for_bbox(record.volume, volume_bbox)} "
            f"elapsed_ms={(time.perf_counter() - volume_t0) * 1000.0:.2f}"
        )
        mask_bbox = _scaled_source_bbox_zyx(
            record.mask,
            self.crop_size,
            min_corner,
            dst_spacing_base=record.volume_spacing_base,
            src_spacing_base=record.mask_spacing_base,
        )
        mask_t0 = time.perf_counter()
        mask_crop = _read_scaled_channel_crop(
            record.mask,
            self.crop_size,
            min_corner,
            dst_spacing_base=record.volume_spacing_base,
            src_spacing_base=record.mask_spacing_base,
        )
        valid_count = int((mask_crop > 0.0).sum()) if self.debug_sampling else -1
        self._debug_sampling_print(
            f"read crop field='grad_mag' crop_kind={crop_kind!r} "
            f"origin_zyx={_format_optional_array(min_corner)} "
            f"shape={self.crop_size} src_bbox={_format_bbox(mask_bbox)} "
            f"chunks={_chunk_ranges_for_bbox(record.mask, mask_bbox)} "
            f"valid_voxels={valid_count} "
            f"elapsed_ms={(time.perf_counter() - mask_t0) * 1000.0:.2f}"
        )
        if record.nx is None or record.ny is None:
            if not self.allow_arbitrary_up_fallback:
                raise ValueError("Lasagna nx/ny normal channels are required")
            nx_crop = None
            ny_crop = None
        else:
            if record.nx_spacing_base is None or record.ny_spacing_base is None:
                raise ValueError("Lasagna nx/ny spacing metadata is required")
            nx_bbox = _scaled_source_bbox_zyx(
                record.nx,
                self.crop_size,
                min_corner,
                dst_spacing_base=record.volume_spacing_base,
                src_spacing_base=record.nx_spacing_base,
            )
            nx_t0 = time.perf_counter()
            nx_crop = _read_scaled_channel_crop(
                record.nx,
                self.crop_size,
                min_corner,
                dst_spacing_base=record.volume_spacing_base,
                src_spacing_base=record.nx_spacing_base,
            )
            self._debug_sampling_print(
                f"read crop field='nx' crop_kind={crop_kind!r} "
                f"origin_zyx={_format_optional_array(min_corner)} "
                f"shape={self.crop_size} src_bbox={_format_bbox(nx_bbox)} "
                f"chunks={_chunk_ranges_for_bbox(record.nx, nx_bbox)} "
                f"elapsed_ms={(time.perf_counter() - nx_t0) * 1000.0:.2f}"
            )
            ny_bbox = _scaled_source_bbox_zyx(
                record.ny,
                self.crop_size,
                min_corner,
                dst_spacing_base=record.volume_spacing_base,
                src_spacing_base=record.ny_spacing_base,
            )
            ny_t0 = time.perf_counter()
            ny_crop = _read_scaled_channel_crop(
                record.ny,
                self.crop_size,
                min_corner,
                dst_spacing_base=record.volume_spacing_base,
                src_spacing_base=record.ny_spacing_base,
            )
            self._debug_sampling_print(
                f"read crop field='ny' crop_kind={crop_kind!r} "
                f"origin_zyx={_format_optional_array(min_corner)} "
                f"shape={self.crop_size} src_bbox={_format_bbox(ny_bbox)} "
                f"chunks={_chunk_ranges_for_bbox(record.ny, ny_bbox)} "
                f"elapsed_ms={(time.perf_counter() - ny_t0) * 1000.0:.2f}"
            )

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
            "crop_kind": crop_kind,
        }

    def _make_conditioned_sample(
        self,
        record: _FiberRecord,
        crop_base: dict[str, Any],
        *,
        direction_kind: str,
        positive_target_id: int,
    ) -> dict[str, Any]:
        if direction_kind == "positive":
            cond_fw_xyz, cond_up_xyz, direction_kind = self._sample_jittered_conditioning(
                crop_base["tangent_xyz"],
                crop_base["center_normal_xyz"],
                min_angle_degrees=0.0,
                max_angle_degrees=self.positive_direction_jitter_degrees,
                direction_kind=direction_kind,
            )
        elif direction_kind == "negative":
            cond_fw_xyz, cond_up_xyz, direction_kind = self._sample_jittered_conditioning(
                crop_base["tangent_xyz"],
                crop_base["center_normal_xyz"],
                min_angle_degrees=self.negative_direction_min_degrees,
                max_angle_degrees=self.negative_direction_max_degrees,
                direction_kind=direction_kind,
            )
        else:
            raise ValueError(f"unsupported direction_kind {direction_kind!r}")

        return {
            "volume": crop_base["volume"],
            "mask_values": crop_base["mask_values"],
            "nx_values": crop_base["nx_values"],
            "ny_values": crop_base["ny_values"],
            "cond_fw_xyz": cond_fw_xyz,
            "cond_up_xyz": cond_up_xyz,
            "origin": crop_base["origin"],
            "line_points_xyz": record.fiber.line_points_xyz
            / float(record.volume_spacing_base),
            "positive_target_id": positive_target_id,
            "crop_kind": crop_base["crop_kind"],
            "direction_kind": direction_kind,
        }

    def sample_batch(self, *, record_index: int | None = None) -> FiberTraceBatch:
        if record_index is None:
            selected_record_index = int(self.rng.integers(0, len(self.records)))
        else:
            selected_record_index = int(record_index)
        record = self.records[selected_record_index]
        pair_count = self.batch_size // 2
        gt_pair_count = (pair_count + 1) // 2
        random_pair_count = pair_count - gt_pair_count
        crop_kinds = ("gt_control",) * gt_pair_count + (
            "random_valid",
        ) * random_pair_count

        control_indices: list[int | None] = []
        if gt_pair_count:
            control_count = int(record.fiber.control_points_xyz.shape[0])
            replace = gt_pair_count > control_count
            choices = self.rng.choice(
                control_count, size=gt_pair_count, replace=replace
            )
            control_indices = [int(choice) for choice in choices]

        samples: list[dict[str, Any]] = []
        gt_index = 0
        for kind in crop_kinds:
            control_index = None
            if kind == "gt_control":
                control_index = control_indices[gt_index]
                gt_index += 1
            crop_base = self._sample_crop_base(
                record, crop_kind=kind, control_index=control_index
            )
            samples.append(
                self._make_conditioned_sample(
                    record,
                    crop_base,
                    direction_kind="positive",
                    positive_target_id=selected_record_index,
                )
            )
            samples.append(
                self._make_conditioned_sample(
                    record,
                    crop_base,
                    direction_kind="negative",
                    positive_target_id=selected_record_index,
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

        return FiberTraceBatch(
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
            target_up_xyz=torch.zeros(
                (sample_count, 3) + crop_shape, dtype=torch.float32
            ),
            target_up_valid=torch.zeros((sample_count,) + crop_shape, dtype=torch.bool),
            cond_fw_xyz=torch.from_numpy(
                np.stack([s["cond_fw_xyz"] for s in samples], axis=0)
            ).to(torch.float32),
            cond_up_xyz=torch.from_numpy(
                np.stack([s["cond_up_xyz"] for s in samples], axis=0)
            ).to(torch.float32),
            crop_origin_zyx=torch.from_numpy(
                np.stack([s["origin"] for s in samples], axis=0)
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
