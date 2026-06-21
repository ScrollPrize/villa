from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import zarr

from vesuvius.neural_tracing.fiber_trace.fiber_json import Vc3dFiber, load_vc3d_fiber
from vesuvius.neural_tracing.fiber_trace.geometry import (
    classify_voxels,
    construct_up_vector,
    perturb_direction,
    random_unit_vector,
    tangent_at_point,
)


try:
    from vesuvius.neural_tracing.datasets.common import (
        _read_volume_crop_from_patch as _common_read_volume_crop_from_patch,
        open_zarr as _common_open_zarr,
    )
except Exception:  # pragma: no cover - exercised only with incompatible optional deps
    _common_read_volume_crop_from_patch = None
    _common_open_zarr = None


try:
    from zarr.errors import PathNotFoundError
except Exception:  # pragma: no cover - compatibility with older zarr

    class PathNotFoundError(Exception):
        pass


@dataclass(frozen=True)
class FiberTraceBatch:
    volume: torch.Tensor
    valid_mask: torch.Tensor
    labels: torch.Tensor
    target_fw: torch.Tensor
    target_up: torch.Tensor
    cond_fw: torch.Tensor
    cond_up: torch.Tensor
    crop_origin_zyx: torch.Tensor
    crop_kinds: tuple[str, ...]
    fiber_paths: tuple[str, ...]
    direction_kinds: tuple[str, ...]

    def to(self, device: torch.device | str) -> "FiberTraceBatch":
        return FiberTraceBatch(
            volume=self.volume.to(device),
            valid_mask=self.valid_mask.to(device),
            labels=self.labels.to(device),
            target_fw=self.target_fw.to(device),
            target_up=self.target_up.to(device),
            cond_fw=self.cond_fw.to(device),
            cond_up=self.cond_up.to(device),
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
    volume_scale: int
    dataset_config: dict[str, Any]
    valid_threshold: float


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
    if _common_open_zarr is not None:
        try:
            return _common_open_zarr(
                path, scale=scale, auth_json_path=auth_json_path, config=config
            )
        except (KeyError, PathNotFoundError):
            pass
    arr = zarr.open(str(path), mode="r")
    if hasattr(arr, "keys") and str(scale) in arr:
        return arr[str(scale)]
    return arr


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


def _normalize_zscore(crop: np.ndarray) -> np.ndarray:
    arr = crop.astype(np.float32, copy=False)
    mean = float(arr.mean())
    std = float(arr.std())
    if not np.isfinite(std) or std <= 1e-6:
        return arr - mean
    return (arr - mean) / std


def _read_image_crop_from_patch(
    patch: Any,
    crop_shape: tuple[int, int, int],
    min_corner: np.ndarray,
    max_corner: np.ndarray,
    *,
    image_normalization: str,
) -> np.ndarray:
    if _common_read_volume_crop_from_patch is not None:
        return _common_read_volume_crop_from_patch(
            patch,
            crop_shape,
            min_corner,
            max_corner,
            image_normalization=image_normalization,
        )
    crop = _read_raw_crop(patch.volume, crop_shape, min_corner)
    if image_normalization == "unit":
        return crop.astype(np.float32, copy=False) / 255.0
    if image_normalization == "zscore":
        return _normalize_zscore(crop)
    raise ValueError(
        f"Unknown image_normalization '{image_normalization}' "
        f"(expected 'zscore' or 'unit')."
    )


def _resolve_fiber_paths(dataset_config: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for item in dataset_config.get("fiber_paths", []) or []:
        paths.append(Path(item))
    fiber_glob = dataset_config.get("fiber_glob")
    if fiber_glob:
        paths.extend(Path(path) for path in sorted(glob.glob(str(fiber_glob))))
    if not paths:
        raise ValueError("dataset entry must provide fiber_paths or fiber_glob")
    return paths


class FiberTraceBatchBuilder:
    """Sample single-fiber training batches from VC3D fiber JSON records."""

    def __init__(
        self, config: dict[str, Any], *, rng: np.random.Generator | None = None
    ) -> None:
        self.config = dict(config)
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
            self.config.get("positive_direction_jitter_degrees", 10.0)
        )
        self.positive_radius = float(self.config.get("positive_radius", 1.5))
        self.ignore_radius = float(
            self.config.get("ignore_radius", max(3.0, self.positive_radius))
        )
        self.positive_cosine = float(
            self.config.get("positive_cosine", np.cos(np.deg2rad(30.0)))
        )
        self.negative_cosine = float(
            self.config.get("negative_cosine", np.cos(np.deg2rad(75.0)))
        )
        self.random_valid_max_attempts = int(
            self.config.get("random_valid_max_attempts", 256)
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
                volume = np.asarray(record_config["volume"])
                mask = np.asarray(record_config["mask"])
                if volume.ndim != 3 or mask.ndim != 3:
                    raise ValueError("_array_records volume and mask must be 3D arrays")
                if tuple(volume.shape) != tuple(mask.shape):
                    raise ValueError(
                        "_array_records mask shape must match volume shape: "
                        f"volume={tuple(volume.shape)} mask={tuple(mask.shape)}"
                    )
                fiber = record_config.get("fiber")
                if fiber is None:
                    fiber = load_vc3d_fiber(record_config["fiber_path"])
                self.records.append(
                    _FiberRecord(
                        fiber=fiber,
                        volume=volume,
                        mask=mask,
                        volume_scale=0,
                        dataset_config=record_config,
                        valid_threshold=float(
                            record_config.get("valid_mask_threshold", 0.0)
                        ),
                    )
                )
        else:
            if not isinstance(datasets, list) or not datasets:
                raise ValueError("config must contain a non-empty datasets list")

        for dataset_config_raw in datasets or []:
            if array_records is not None:
                break
            dataset_config = dict(dataset_config_raw)
            volume_path = dataset_config.get("volume_path")
            if not volume_path:
                raise ValueError("dataset entry missing volume_path")
            mask_path = dataset_config.get("mask_path") or dataset_config.get(
                "grad_mag_path"
            )
            if not mask_path:
                raise ValueError(
                    "dataset entry must provide mask_path or grad_mag_path"
                )

            volume_scale = int(dataset_config.get("volume_scale", 0))
            mask_scale = int(
                dataset_config.get(
                    "mask_scale", dataset_config.get("grad_mag_scale", volume_scale)
                )
            )
            volume = _open_zarr_array(
                volume_path,
                scale=volume_scale,
                auth_json_path=dataset_config.get("volume_auth_json"),
                config=self.config,
            )
            mask = _open_zarr_array(
                mask_path,
                scale=mask_scale,
                auth_json_path=dataset_config.get(
                    "mask_auth_json", dataset_config.get("volume_auth_json")
                ),
                config=self.config,
            )
            if tuple(volume.shape) != tuple(mask.shape):
                raise ValueError(
                    "mask/grad-mag shape must match volume shape at selected levels: "
                    f"volume={tuple(volume.shape)} mask={tuple(mask.shape)}"
                )

            valid_threshold = float(
                dataset_config.get(
                    "valid_mask_threshold", self.config.get("valid_mask_threshold", 0.0)
                )
            )
            for fiber_path in _resolve_fiber_paths(dataset_config):
                self.records.append(
                    _FiberRecord(
                        fiber=load_vc3d_fiber(fiber_path),
                        volume=volume,
                        mask=mask,
                        volume_scale=volume_scale,
                        dataset_config=dataset_config,
                        valid_threshold=valid_threshold,
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

    def _sample_random_valid_center(self, record: _FiberRecord) -> np.ndarray:
        shape = np.asarray(record.mask.shape, dtype=np.int64)
        for _ in range(self.random_valid_max_attempts):
            center = np.asarray(
                [self.rng.integers(0, int(size)) for size in shape], dtype=np.int64
            )
            value = np.asarray(
                record.mask[
                    center[0] : center[0] + 1,
                    center[1] : center[1] + 1,
                    center[2] : center[2] + 1,
                ]
            )
            if value.size and float(value.reshape(-1)[0]) > record.valid_threshold:
                return center

        mask_np = np.asarray(record.mask[:])
        valid_coords = np.argwhere(mask_np > record.valid_threshold)
        if valid_coords.size == 0:
            raise ValueError("mask/grad-mag volume contains no valid voxels")
        return valid_coords[int(self.rng.integers(0, valid_coords.shape[0]))].astype(
            np.int64, copy=False
        )

    def _sample_conditioning(
        self, tangent_zyx: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, str]:
        if float(self.rng.random()) < self.positive_direction_probability:
            fw = perturb_direction(
                tangent_zyx,
                max_angle_degrees=self.positive_direction_jitter_degrees,
                rng=self.rng,
            )
            direction_kind = "gt_tangent"
        else:
            fw = random_unit_vector(self.rng)
            direction_kind = "random"
        up = construct_up_vector(fw)
        return (
            fw.astype(np.float32, copy=False),
            up.astype(np.float32, copy=False),
            direction_kind,
        )

    def _sample_one_crop(
        self, record: _FiberRecord, *, crop_kind: str
    ) -> dict[str, Any]:
        fiber = record.fiber
        if crop_kind == "gt_control":
            controls = fiber.control_points_zyx
            center = np.rint(
                controls[int(self.rng.integers(0, controls.shape[0]))]
            ).astype(np.int64)
        elif crop_kind == "random_valid":
            center = self._sample_random_valid_center(record)
        else:
            raise ValueError(f"unsupported crop_kind {crop_kind!r}")

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
            _read_raw_crop(record.mask, self.crop_size, min_corner)
            > record.valid_threshold
        )
        if not bool(valid_crop.any()):
            raise ValueError(
                f"sampled {crop_kind} crop has no valid mask/grad-mag voxels"
            )

        tangent = tangent_at_point(fiber.line_points_zyx, center.astype(np.float32))
        cond_fw, cond_up, direction_kind = self._sample_conditioning(tangent)
        classified = classify_voxels(
            crop_origin_zyx=min_corner,
            crop_shape=self.crop_size,
            line_points_zyx=fiber.line_points_zyx,
            cond_fw_zyx=cond_fw,
            valid_mask=valid_crop,
            positive_radius=self.positive_radius,
            ignore_radius=self.ignore_radius,
            positive_cosine=self.positive_cosine,
            negative_cosine=self.negative_cosine,
        )
        return {
            "volume": volume_crop.astype(np.float32, copy=False),
            "valid_mask": valid_crop,
            "labels": classified["labels"],
            "target_fw": classified["target_fw"],
            "target_up": classified["target_up"],
            "cond_fw": cond_fw,
            "cond_up": cond_up,
            "origin": min_corner.astype(np.int64, copy=False),
            "crop_kind": crop_kind,
            "direction_kind": direction_kind,
        }

    def sample_batch(self, *, record_index: int | None = None) -> FiberTraceBatch:
        record = self._sample_record(record_index)
        half = self.batch_size // 2
        crop_kinds = ("gt_control",) * half + ("random_valid",) * half
        samples = [self._sample_one_crop(record, crop_kind=kind) for kind in crop_kinds]
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
            target_fw=torch.from_numpy(
                np.stack([s["target_fw"] for s in samples], axis=0)
            ).to(torch.float32),
            target_up=torch.from_numpy(
                np.stack([s["target_up"] for s in samples], axis=0)
            ).to(torch.float32),
            cond_fw=torch.from_numpy(
                np.stack([s["cond_fw"] for s in samples], axis=0)
            ).to(torch.float32),
            cond_up=torch.from_numpy(
                np.stack([s["cond_up"] for s in samples], axis=0)
            ).to(torch.float32),
            crop_origin_zyx=torch.from_numpy(
                np.stack([s["origin"] for s in samples], axis=0)
            ).to(torch.long),
            crop_kinds=tuple(str(s["crop_kind"]) for s in samples),
            fiber_paths=tuple(fiber_path for _ in samples),
            direction_kinds=tuple(str(s["direction_kind"]) for s in samples),
        )
