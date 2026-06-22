from __future__ import annotations

import glob
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from vesuvius.neural_tracing.fiber_trace.fiber_json import Vc3dFiber, load_vc3d_fiber
from vesuvius.neural_tracing.fiber_trace.geometry import (
    classify_voxels,
    construct_up_vector,
    decode_lasagna_normals_xyz,
    perturb_direction,
    tangent_at_point,
    xyz_to_zyx,
    zyx_to_xyz,
)
from vesuvius.neural_tracing.fiber_trace.labels import POSITIVE_LABEL
from vesuvius.neural_tracing.datasets.common import (
    _read_volume_crop_from_patch as _common_read_volume_crop_from_patch,
    open_zarr as _common_open_zarr,
    open_zarr_group as _common_open_zarr_group,
)


@dataclass(frozen=True)
class FiberTraceBatch:
    volume: torch.Tensor
    valid_mask: torch.Tensor
    labels: torch.Tensor
    target_id: torch.Tensor
    target_fw_xyz: torch.Tensor
    target_up_xyz: torch.Tensor
    target_up_valid: torch.Tensor
    cond_fw_xyz: torch.Tensor
    cond_up_xyz: torch.Tensor
    crop_origin_zyx: torch.Tensor
    crop_kinds: tuple[str, ...]
    fiber_paths: tuple[str, ...]
    direction_kinds: tuple[str, ...]

    def to(self, device: torch.device | str) -> "FiberTraceBatch":
        return FiberTraceBatch(
            volume=self.volume.to(device),
            valid_mask=self.valid_mask.to(device),
            labels=self.labels.to(device),
            target_id=self.target_id.to(device),
            target_fw_xyz=self.target_fw_xyz.to(device),
            target_up_xyz=self.target_up_xyz.to(device),
            target_up_valid=self.target_up_valid.to(device),
            cond_fw_xyz=self.cond_fw_xyz.to(device),
            cond_up_xyz=self.cond_up_xyz.to(device),
            crop_origin_zyx=self.crop_origin_zyx.to(device),
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


def _reject_valid_mask_threshold_key(config: dict[str, Any], *, context: str) -> None:
    if "valid_mask_threshold" not in config:
        return
    raise ValueError(
        "valid_mask_threshold was removed; mask/grad-mag validity is always "
        f"binary value > 0 in {context}"
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


def _open_zarr_group(
    path: str | Path, *, auth_json_path: str | None, config: dict[str, Any]
):
    return _common_open_zarr_group(path, auth_json_path=auth_json_path, config=config)


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
            lasagna_manifest_path = (
                dataset_config.get("lasagna_manifest_path")
                or dataset_config.get("lasagna_json")
                or dataset_config.get("manifest_path")
            )
            volume_path = dataset_config.get("base_volume_path") or dataset_config.get(
                "volume_path"
            )
            if not volume_path:
                raise ValueError("dataset entry missing base_volume_path")
            if not lasagna_manifest_path:
                mask_path_raw = dataset_config.get("mask_path") or dataset_config.get(
                    "grad_mag_path"
                )
                if not mask_path_raw:
                    raise ValueError(
                        "dataset entry must provide lasagna_manifest_path, "
                        "mask_path or grad_mag_path"
                    )
                if (
                    not dataset_config.get("nx_path")
                    or not dataset_config.get("ny_path")
                ) and not self.allow_arbitrary_up_fallback:
                    raise ValueError(
                        "dataset entry must provide lasagna_manifest_path or explicit "
                        "Lasagna nx_path and ny_path normal channels"
                    )

            volume_scale = int(
                dataset_config.get(
                    "base_volume_scale", dataset_config.get("volume_scale", 0)
                )
            )
            volume_spacing_base = _scale_level_to_spacing_base(volume_scale)
            volume = _open_zarr_array_resolved(
                volume_path,
                scale=volume_scale,
                auth_json_path=dataset_config.get(
                    "base_volume_auth_json", dataset_config.get("volume_auth_json")
                ),
                config=self.config,
            )

            if lasagna_manifest_path:
                lasagna_volume = _load_lasagna_volume(
                    lasagna_manifest_path, self.config
                )
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
            else:
                mask_path = dataset_config.get("mask_path") or dataset_config.get(
                    "grad_mag_path"
                )
                if not mask_path:
                    raise ValueError(
                        "dataset entry must provide lasagna_manifest_path or "
                        "mask_path/grad_mag_path"
                    )
                nx_path = dataset_config.get("nx_path")
                ny_path = dataset_config.get("ny_path")
                if (
                    not nx_path or not ny_path
                ) and not self.allow_arbitrary_up_fallback:
                    raise ValueError(
                        "dataset entry must provide lasagna_manifest_path or explicit "
                        "Lasagna nx_path and ny_path normal channels"
                    )
                mask_scale = int(
                    dataset_config.get(
                        "mask_scale",
                        dataset_config.get("grad_mag_scale", volume_scale),
                    )
                )
                normal_scale = int(dataset_config.get("normal_scale", volume_scale))
                mask = _open_zarr_array_resolved(
                    mask_path,
                    scale=mask_scale,
                    auth_json_path=dataset_config.get(
                        "mask_auth_json", dataset_config.get("volume_auth_json")
                    ),
                    config=self.config,
                )
                mask_spacing_base = _scale_level_to_spacing_base(mask_scale)
                if nx_path and ny_path:
                    nx_scale = int(dataset_config.get("nx_scale", normal_scale))
                    ny_scale = int(dataset_config.get("ny_scale", normal_scale))
                    nx = _open_zarr_array_resolved(
                        nx_path,
                        scale=nx_scale,
                        auth_json_path=dataset_config.get(
                            "nx_auth_json",
                            dataset_config.get(
                                "normal_auth_json",
                                dataset_config.get("volume_auth_json"),
                            ),
                        ),
                        config=self.config,
                    )
                    ny = _open_zarr_array_resolved(
                        ny_path,
                        scale=ny_scale,
                        auth_json_path=dataset_config.get(
                            "ny_auth_json",
                            dataset_config.get(
                                "normal_auth_json",
                                dataset_config.get("volume_auth_json"),
                            ),
                        ),
                        config=self.config,
                    )
                    nx_spacing_base = _scale_level_to_spacing_base(nx_scale)
                    ny_spacing_base = _scale_level_to_spacing_base(ny_scale)
                    if abs(float(nx_spacing_base) - float(ny_spacing_base)) > 1e-6:
                        raise ValueError(
                            "Lasagna nx_path and ny_path scale mismatch: "
                            f"nx_scale={nx_scale} ny_scale={ny_scale}"
                        )
                else:
                    nx = None
                    ny = None
                    nx_spacing_base = None
                    ny_spacing_base = None

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

    def _sample_record(self, record_index: int | None) -> _FiberRecord:
        if record_index is not None:
            return self.records[int(record_index)]
        return self.records[int(self.rng.integers(0, len(self.records)))]

    def _normal_at_zyx(
        self, record: _FiberRecord, center_zyx: np.ndarray
    ) -> tuple[np.ndarray | None, bool]:
        if record.nx is None or record.ny is None:
            if self.allow_arbitrary_up_fallback:
                return None, True
            raise ValueError("Lasagna nx/ny normal channels are required")
        if record.nx_spacing_base is None or record.ny_spacing_base is None:
            raise ValueError("Lasagna nx/ny spacing metadata is required")
        nx_value, nx_ok = _value_at_sample_zyx(
            record.nx,
            center_zyx,
            sample_spacing_base=record.volume_spacing_base,
            array_spacing_base=record.nx_spacing_base,
        )
        ny_value, ny_ok = _value_at_sample_zyx(
            record.ny,
            center_zyx,
            sample_spacing_base=record.volume_spacing_base,
            array_spacing_base=record.ny_spacing_base,
        )
        if not nx_ok or not ny_ok:
            return np.zeros(3, dtype=np.float32), False
        nx_vox = np.asarray([nx_value])
        ny_vox = np.asarray([ny_value])
        normal_xyz, valid = decode_lasagna_normals_xyz(nx_vox, ny_vox)
        return normal_xyz.reshape(-1, 3)[0], bool(valid.reshape(-1)[0])

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

        mask_np = np.asarray(record.mask[:])
        valid_mask = mask_np > 0.0
        valid_mask_coords = np.argwhere(valid_mask)
        if valid_mask_coords.size == 0:
            raise ValueError("mask/grad-mag and nx/ny volumes contain no valid voxels")
        order = self.rng.permutation(valid_mask_coords.shape[0])
        for idx in order:
            mask_center = valid_mask_coords[int(idx)]
            center = np.rint(
                mask_center.astype(np.float64)
                * float(record.mask_spacing_base)
                / float(record.volume_spacing_base)
            ).astype(np.int64)
            if np.any(center < 0) or np.any(center >= shape):
                continue
            _, normal_valid = self._normal_at_zyx(record, center)
            if normal_valid:
                return center
        raise ValueError("mask/grad-mag and nx/ny volumes contain no valid voxels")

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
        center_normal_xyz, center_normal_valid = self._normal_at_zyx(record, center)
        if not center_normal_valid:
            raise ValueError(f"sampled {crop_kind} center has invalid Lasagna nx/ny")

        crop_size_np = np.asarray(self.crop_size, dtype=np.int64)
        min_corner = center - (crop_size_np // 2)
        max_corner = min_corner + crop_size_np
        patch = SimpleNamespace(volume=record.volume, scale=record.volume_scale)
        volume_crop = _read_image_crop_from_patch(
            patch,
            self.crop_size,
            min_corner,
            max_corner,
            image_normalization=self.image_normalization,
        )
        valid_crop = (
            _read_scaled_channel_crop(
                record.mask,
                self.crop_size,
                min_corner,
                dst_spacing_base=record.volume_spacing_base,
                src_spacing_base=record.mask_spacing_base,
            )
            > 0.0
        )
        if record.nx is None or record.ny is None:
            if not self.allow_arbitrary_up_fallback:
                raise ValueError("Lasagna nx/ny normal channels are required")
            normal_xyz_crop = None
            normal_valid_crop = np.ones(self.crop_size, dtype=bool)
        else:
            if record.nx_spacing_base is None or record.ny_spacing_base is None:
                raise ValueError("Lasagna nx/ny spacing metadata is required")
            nx_crop = _read_scaled_channel_crop(
                record.nx,
                self.crop_size,
                min_corner,
                dst_spacing_base=record.volume_spacing_base,
                src_spacing_base=record.nx_spacing_base,
            )
            ny_crop = _read_scaled_channel_crop(
                record.ny,
                self.crop_size,
                min_corner,
                dst_spacing_base=record.volume_spacing_base,
                src_spacing_base=record.ny_spacing_base,
            )
            normal_xyz_crop, normal_valid_crop = decode_lasagna_normals_xyz(
                nx_crop, ny_crop
            )
            valid_crop = valid_crop & normal_valid_crop
        if not bool(valid_crop.any()):
            raise ValueError(
                f"sampled {crop_kind} crop has no valid mask/grad-mag and nx/ny voxels"
            )

        tangent_xyz = tangent_at_point(
            fiber.line_points_xyz,
            zyx_to_xyz(center.astype(np.float32) * float(record.volume_spacing_base)),
        )

        return {
            "volume": volume_crop.astype(np.float32, copy=False),
            "valid_mask": valid_crop,
            "normal_xyz": normal_xyz_crop,
            "normal_valid_mask": normal_valid_crop,
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

        classified = classify_voxels(
            crop_origin_zyx=crop_base["origin"],
            crop_shape=self.crop_size,
            line_points_xyz=record.fiber.line_points_xyz
            / float(record.volume_spacing_base),
            cond_fw_xyz=cond_fw_xyz,
            cond_up_xyz=cond_up_xyz,
            valid_mask=crop_base["valid_mask"],
            normal_xyz=crop_base["normal_xyz"],
            normal_valid_mask=crop_base["normal_valid_mask"],
            allow_arbitrary_up_fallback=self.allow_arbitrary_up_fallback,
            degenerate_up_policy=self.degenerate_up_policy,
            positive_radius=self.positive_radius,
            ignore_radius=self.ignore_radius,
            normal_plane_jitter_voxels=self.normal_plane_jitter_voxels,
            normal_perpendicular_jitter_voxels=self.normal_perpendicular_jitter_voxels,
            negative_cone_distance_voxels=self.negative_cone_distance_voxels,
            positive_cosine=self.positive_cosine,
            negative_cosine=self.negative_cosine,
            positive_target_id=positive_target_id,
        )
        positive_labels = classified["labels"] == POSITIVE_LABEL
        if bool(positive_labels.any()) and not bool(
            (positive_labels & classified["target_up_valid"]).any()
        ):
            raise ValueError(
                f"sampled {crop_base['crop_kind']} crop has no valid up-vector supervision"
            )
        return {
            "volume": crop_base["volume"],
            "valid_mask": crop_base["valid_mask"],
            "labels": classified["labels"],
            "target_id": classified["target_id"],
            "target_fw_xyz": classified["target_fw_xyz"],
            "target_up_xyz": classified["target_up_xyz"],
            "target_up_valid": classified["target_up_valid"],
            "cond_fw_xyz": cond_fw_xyz,
            "cond_up_xyz": cond_up_xyz,
            "origin": crop_base["origin"],
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

        return FiberTraceBatch(
            volume=torch.from_numpy(
                np.stack([s["volume"] for s in samples], axis=0)[:, None]
            ).to(torch.float32),
            valid_mask=torch.from_numpy(
                np.stack([s["valid_mask"] for s in samples], axis=0)
            ).to(torch.bool),
            labels=torch.from_numpy(
                np.stack([s["labels"] for s in samples], axis=0)
            ).to(torch.long),
            target_id=torch.from_numpy(
                np.stack([s["target_id"] for s in samples], axis=0)
            ).to(torch.long),
            target_fw_xyz=torch.from_numpy(
                np.stack([s["target_fw_xyz"] for s in samples], axis=0)
            ).to(torch.float32),
            target_up_xyz=torch.from_numpy(
                np.stack([s["target_up_xyz"] for s in samples], axis=0)
            ).to(torch.float32),
            target_up_valid=torch.from_numpy(
                np.stack([s["target_up_valid"] for s in samples], axis=0)
            ).to(torch.bool),
            cond_fw_xyz=torch.from_numpy(
                np.stack([s["cond_fw_xyz"] for s in samples], axis=0)
            ).to(torch.float32),
            cond_up_xyz=torch.from_numpy(
                np.stack([s["cond_up_xyz"] for s in samples], axis=0)
            ).to(torch.float32),
            crop_origin_zyx=torch.from_numpy(
                np.stack([s["origin"] for s in samples], axis=0)
            ).to(torch.long),
            crop_kinds=tuple(str(s["crop_kind"]) for s in samples),
            fiber_paths=tuple(fiber_path for _ in samples),
            direction_kinds=tuple(str(s["direction_kind"]) for s in samples),
        )
