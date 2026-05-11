from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from botocore.exceptions import (
    ConnectTimeoutError,
    EndpointConnectionError,
    ReadTimeoutError,
)
from torch.utils.data import Dataset

from vesuvius.data.utils import open_zarr

_TRANSIENT_VOLUME_READ_ERRORS: tuple[type[BaseException], ...] = (
    EndpointConnectionError,
    ReadTimeoutError,
    ConnectTimeoutError,
    OSError,
)
from vesuvius.neural_tracing.autoreg_fiber.fiber_geometry import load_fiber_cache
from vesuvius.neural_tracing.autoreg_fiber.serialization import IGNORE_INDEX, serialize_fiber_example
from vesuvius.neural_tracing.autoreg_mesh.dataset import (
    _pad_1d_bool,
    _pad_1d_long,
    _pad_2d_float,
    _pad_2d_long,
)


@dataclass(frozen=True)
class FiberSamplePlan:
    fiber_index: int
    fiber_id: str
    point_start: int
    min_corner: tuple[int, int, int]
    target_volume: str = "__single__"


def _as_3tuple(value, *, name: str) -> tuple[int, int, int]:
    if isinstance(value, int):
        return (int(value), int(value), int(value))
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must be an int or length-3 sequence, got {value!r}")
    return tuple(int(v) for v in value)


def _read_npz_points(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    points, metadata = load_fiber_cache(path)
    return np.asarray(points, dtype=np.float32), metadata


def _fiber_cache_paths_from_manifest(path: str | Path) -> list[Path]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    raw_paths = payload.get("fiber_cache_paths", payload.get("cache_paths", []))
    manifest_dir = Path(path).expanduser().resolve().parent
    paths = []
    for raw_path in raw_paths:
        candidate = Path(str(raw_path)).expanduser()
        if not candidate.is_absolute():
            candidate = manifest_dir / candidate
        paths.append(candidate)
    return paths


def _crop_min_corner_for_points(
    points_zyx: np.ndarray,
    *,
    crop_size: tuple[int, int, int],
    volume_shape: tuple[int, int, int],
) -> tuple[int, int, int]:
    center = 0.5 * (points_zyx.min(axis=0) + points_zyx.max(axis=0))
    min_corner = np.floor(center - 0.5 * np.asarray(crop_size, dtype=np.float32)).astype(np.int64)
    max_min_corner = np.maximum(0, np.asarray(volume_shape, dtype=np.int64) - np.asarray(crop_size, dtype=np.int64))
    min_corner = np.clip(min_corner, 0, max_min_corner)
    return tuple(int(v) for v in min_corner)


def _apply_cubic_spatial_augmentation(
    *,
    volume_zyx: np.ndarray,
    points_local_zyx: np.ndarray,
    crop_size_zyx: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    if not (crop_size_zyx[0] == crop_size_zyx[1] == crop_size_zyx[2]):
        raise ValueError(
            f"spatial_augmentation_enabled requires isotropic crop_size; got {crop_size_zyx!r}"
        )
    perm = np.random.permutation(3)
    flips = np.random.randint(0, 2, size=3).astype(bool)
    volume_aug = np.transpose(volume_zyx, perm)
    points_aug = points_local_zyx[:, perm].astype(np.float32, copy=True)
    extent = float(crop_size_zyx[0] - 1)
    for axis in range(3):
        if flips[axis]:
            volume_aug = np.flip(volume_aug, axis=axis)
            points_aug[:, axis] = extent - points_aug[:, axis]
    volume_aug = np.ascontiguousarray(volume_aug, dtype=np.float32)
    return volume_aug, points_aug


def _points_fit_crop(points_zyx: np.ndarray, *, min_corner: tuple[int, int, int], crop_size: tuple[int, int, int]) -> bool:
    local = points_zyx - np.asarray(min_corner, dtype=np.float32)
    return bool(np.all(local >= 0.0) and np.all(local < np.asarray(crop_size, dtype=np.float32)))


def split_indices_by_fiber_id(
    sample_plans: Sequence[FiberSamplePlan],
    *,
    val_fraction: float,
    seed: int = 0,
) -> tuple[list[int], list[int]]:
    by_id: dict[str, list[int]] = {}
    for idx, plan in enumerate(sample_plans):
        by_id.setdefault(plan.fiber_id, []).append(idx)
    ids = sorted(by_id)
    rng = random.Random(int(seed))
    rng.shuffle(ids)
    val_count = int(round(len(ids) * float(val_fraction)))
    if val_fraction > 0.0 and val_count == 0 and len(ids) > 1:
        val_count = 1
    val_ids = set(ids[:val_count])
    train_indices: list[int] = []
    val_indices: list[int] = []
    for fiber_id, indices in by_id.items():
        if fiber_id in val_ids:
            val_indices.extend(indices)
        else:
            train_indices.extend(indices)
    return sorted(train_indices), sorted(val_indices)


class AutoregFiberDataset(Dataset):
    def __init__(
        self,
        config: dict,
        *,
        volume_array: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.config = dict(config or {})
        self.crop_size = _as_3tuple(self.config.get("crop_size", (128, 128, 128)), name="crop_size")
        self.patch_size = _as_3tuple(self.config.get("patch_size", (8, 8, 8)), name="patch_size")
        self.offset_num_bins = _as_3tuple(self.config.get("offset_num_bins", (16, 16, 16)), name="offset_num_bins")
        self.prompt_length = int(self.config.get("prompt_length", 8))
        self.target_length = int(self.config.get("target_length", 32))
        self.point_stride = int(self.config.get("point_stride", 1))
        if self.prompt_length <= 0 or self.target_length <= 0:
            raise ValueError("prompt_length and target_length must be positive")
        if self.point_stride <= 0:
            raise ValueError("point_stride must be positive")

        self.spatial_augmentation_enabled = bool(self.config.get("spatial_augmentation_enabled", False))

        self.fiber_cache_paths = [Path(p) for p in self.config.get("fiber_cache_paths", [])]
        if self.config.get("fiber_cache_manifest_json") is not None:
            self.fiber_cache_paths.extend(_fiber_cache_paths_from_manifest(self.config["fiber_cache_manifest_json"]))
        if not self.fiber_cache_paths:
            raise ValueError("config.fiber_cache_paths must contain at least one fiber cache npz")

        self.volume_array = None if volume_array is None else np.asarray(volume_array)
        self.storage_options = dict(self.config.get("storage_options") or {})
        self._zarr_arrays: dict[str, Any] = {}
        self.volume_specs = self._normalize_volume_specs()
        self.volume_shape = self.volume_specs["__single__"]["shape"] if "__single__" in self.volume_specs else next(iter(self.volume_specs.values()))["shape"]

        self.fibers: list[tuple[np.ndarray, dict[str, Any]]] = []
        for path in self.fiber_cache_paths:
            points, metadata = _read_npz_points(path)
            self.fibers.append((points, metadata))

        self.sample_plans = self._build_sample_index()
        if not self.sample_plans:
            raise RuntimeError("AutoregFiberDataset found no valid fiber windows")

    def _normalize_volume_specs(self) -> dict[str, dict[str, Any]]:
        volumes_cfg = dict(self.config.get("volumes") or {})
        if volumes_cfg:
            specs: dict[str, dict[str, Any]] = {}
            for name, raw_spec in volumes_cfg.items():
                spec = dict(raw_spec or {})
                raw_shape = spec.get("volume_shape", spec.get("shape"))
                if raw_shape is None:
                    raise ValueError(f"volume spec {name!r} is missing volume_shape")
                storage_options = dict(self.storage_options)
                storage_options.update(dict(spec.get("storage_options") or {}))
                specs[str(name)] = {
                    "url": spec.get("volume_zarr_url", spec.get("url")),
                    "shape": _as_3tuple(raw_shape, name=f"volumes[{name!r}].volume_shape"),
                    "storage_options": storage_options,
                }
            return specs

        if self.volume_array is not None:
            shape = tuple(int(v) for v in self.volume_array.shape[-3:])
        else:
            raw_shape = self.config.get("volume_shape")
            if raw_shape is None:
                raise ValueError("volume_shape is required when volume_array is not supplied")
            shape = _as_3tuple(raw_shape, name="volume_shape")
        return {
            "__single__": {
                "url": self.config.get("volume_zarr_url"),
                "shape": shape,
                "storage_options": self.storage_options,
            }
        }

    def _target_volume_key(self, metadata: dict[str, Any]) -> str:
        if "__single__" in self.volume_specs:
            return "__single__"
        key = str(metadata.get("target_volume") or "")
        if key not in self.volume_specs:
            raise ValueError(
                f"fiber metadata target_volume={key!r} does not match configured volumes "
                f"{sorted(self.volume_specs)!r}"
            )
        return key

    def _build_sample_index(self) -> list[FiberSamplePlan]:
        plans: list[FiberSamplePlan] = []
        window_len = self.prompt_length + self.target_length
        for fiber_idx, (points, metadata) in enumerate(self.fibers):
            fiber_id = f"{metadata.get('annotation_id', fiber_idx)}:{metadata.get('tree_id', fiber_idx)}"
            target_volume = self._target_volume_key(metadata)
            volume_shape = self.volume_specs[target_volume]["shape"]
            if points.shape[0] < window_len:
                continue
            for start in range(0, points.shape[0] - window_len + 1, self.point_stride):
                window = points[start:start + window_len]
                min_corner = _crop_min_corner_for_points(
                    window,
                    crop_size=self.crop_size,
                    volume_shape=volume_shape,
                )
                if _points_fit_crop(window, min_corner=min_corner, crop_size=self.crop_size):
                    plans.append(
                        FiberSamplePlan(
                            fiber_index=fiber_idx,
                            fiber_id=fiber_id,
                            point_start=int(start),
                            min_corner=min_corner,
                            target_volume=target_volume,
                        )
                    )
        return plans

    def __len__(self) -> int:
        return len(self.sample_plans)

    def _ensure_zarr_array(self, target_volume: str):
        if target_volume in self._zarr_arrays:
            return self._zarr_arrays[target_volume]
        spec = self.volume_specs[target_volume]
        if spec["url"] is None:
            raise ValueError("volume_zarr_url is required when volume_array is not supplied")
        arr = open_zarr(str(spec["url"]), mode="r", storage_options=spec["storage_options"])
        if hasattr(arr, "keys"):
            arr = arr["0"]
        self._zarr_arrays[target_volume] = arr
        return arr

    def _read_volume_crop(self, min_corner: tuple[int, int, int], *, target_volume: str) -> np.ndarray:
        z, y, x = min_corner
        dz, dy, dx = self.crop_size
        source = self.volume_array if self.volume_array is not None else self._ensure_zarr_array(target_volume)
        max_attempts = max(1, int(self.config.get("s3_read_max_attempts", 8)))
        crop: np.ndarray | None = None
        for attempt in range(max_attempts):
            try:
                crop = np.asarray(source[z:z + dz, y:y + dy, x:x + dx], dtype=np.float32)
                break
            except _TRANSIENT_VOLUME_READ_ERRORS as exc:
                if attempt + 1 == max_attempts:
                    raise
                time.sleep(min(2.0 ** attempt, 30.0))
                if self.volume_array is None:
                    self._zarr_arrays.pop(target_volume, None)
                    source = self._ensure_zarr_array(target_volume)
        assert crop is not None
        if crop.shape != self.crop_size:
            out = np.zeros(self.crop_size, dtype=np.float32)
            slices = tuple(slice(0, min(crop.shape[axis], self.crop_size[axis])) for axis in range(3))
            out[slices] = crop[slices]
            crop = out
        return crop.astype(np.float32, copy=False)

    def __getitem__(self, index: int) -> dict:
        plan = self.sample_plans[int(index)]
        points, metadata = self.fibers[plan.fiber_index]
        window_len = self.prompt_length + self.target_length
        window_world = points[plan.point_start:plan.point_start + window_len]
        local = window_world - np.asarray(plan.min_corner, dtype=np.float32)
        volume = self._read_volume_crop(plan.min_corner, target_volume=plan.target_volume)
        if self.spatial_augmentation_enabled:
            volume, local = _apply_cubic_spatial_augmentation(
                volume_zyx=volume,
                points_local_zyx=local.astype(np.float32, copy=False),
                crop_size_zyx=self.crop_size,
            )
        serialized = serialize_fiber_example(
            local,
            prompt_length=self.prompt_length,
            target_length=self.target_length,
            volume_shape=self.crop_size,
            patch_size=self.patch_size,
            offset_num_bins=self.offset_num_bins,
        )
        result = {
            "volume": torch.from_numpy(volume[None, ...]).to(torch.float32),
            "prompt_tokens": {
                "coarse_ids": torch.from_numpy(serialized["prompt_tokens"]["coarse_ids"]).to(torch.long),
                "offset_bins": torch.from_numpy(serialized["prompt_tokens"]["offset_bins"]).to(torch.long),
                "xyz": torch.from_numpy(serialized["prompt_tokens"]["xyz"]).to(torch.float32),
                "positions": torch.from_numpy(serialized["prompt_tokens"]["positions"]).to(torch.long),
                "valid_mask": torch.from_numpy(serialized["prompt_tokens"]["valid_mask"]).to(torch.bool),
            },
            "prompt_anchor_xyz": torch.from_numpy(serialized["prompt_anchor_xyz"]).to(torch.float32),
            "prompt_anchor_valid": torch.tensor(bool(serialized["prompt_anchor_valid"]), dtype=torch.bool),
            "target_coarse_ids": torch.from_numpy(serialized["target_coarse_ids"]).to(torch.long),
            "target_offset_bins": torch.from_numpy(serialized["target_offset_bins"]).to(torch.long),
            "target_valid_mask": torch.from_numpy(serialized["target_valid_mask"]).to(torch.bool),
            "target_stop": torch.from_numpy(serialized["target_stop"]).to(torch.float32),
            "target_xyz": torch.from_numpy(serialized["target_xyz"]).to(torch.float32),
            "target_bin_center_xyz": torch.from_numpy(serialized["target_bin_center_xyz"]).to(torch.float32),
            "target_positions": torch.from_numpy(serialized["target_positions"]).to(torch.long),
            "target_mask": torch.ones((self.target_length,), dtype=torch.bool),
            "target_supervision_mask": torch.from_numpy(serialized["target_valid_mask"]).to(torch.bool),
            "target_lengths": torch.tensor(int(serialized["target_length"]), dtype=torch.long),
            "min_corner": torch.tensor(plan.min_corner, dtype=torch.float32),
            "fiber_metadata": {
                **metadata,
                "fiber_id": plan.fiber_id,
                "point_start": int(plan.point_start),
                "min_corner": tuple(int(v) for v in plan.min_corner),
                "target_volume": plan.target_volume if plan.target_volume != "__single__" else metadata.get("target_volume"),
            },
        }
        return result


def autoreg_fiber_collate(batch: list[dict]) -> dict:
    result = {
        "volume": torch.stack([item["volume"] for item in batch], dim=0),
        "prompt_anchor_xyz": torch.stack([item["prompt_anchor_xyz"] for item in batch], dim=0),
        "prompt_anchor_valid": torch.stack([item["prompt_anchor_valid"] for item in batch], dim=0),
        "target_lengths": torch.stack([item["target_lengths"] for item in batch], dim=0),
        "min_corner": torch.stack([item["min_corner"] for item in batch], dim=0),
        "fiber_metadata": [item["fiber_metadata"] for item in batch],
    }
    prompt_coarse, prompt_mask = _pad_1d_long(
        [item["prompt_tokens"]["coarse_ids"] for item in batch],
        pad_value=IGNORE_INDEX,
    )
    prompt_offset, _ = _pad_2d_long(
        [item["prompt_tokens"]["offset_bins"] for item in batch],
        pad_value=IGNORE_INDEX,
    )
    prompt_xyz, _ = _pad_2d_float([item["prompt_tokens"]["xyz"] for item in batch])
    prompt_positions, _ = _pad_1d_long(
        [item["prompt_tokens"]["positions"] for item in batch],
        pad_value=0,
    )
    prompt_valid, _ = _pad_1d_bool([item["prompt_tokens"]["valid_mask"] for item in batch])
    result["prompt_tokens"] = {
        "coarse_ids": prompt_coarse,
        "offset_bins": prompt_offset,
        "xyz": prompt_xyz,
        "positions": prompt_positions,
        "mask": prompt_mask,
        "valid_mask": prompt_valid & prompt_mask,
    }

    target_coarse, target_mask = _pad_1d_long(
        [item["target_coarse_ids"] for item in batch],
        pad_value=IGNORE_INDEX,
    )
    target_offset, _ = _pad_2d_long(
        [item["target_offset_bins"] for item in batch],
        pad_value=IGNORE_INDEX,
    )
    target_xyz, _ = _pad_2d_float([item["target_xyz"] for item in batch])
    target_bin_center, _ = _pad_2d_float([item["target_bin_center_xyz"] for item in batch])
    target_positions, _ = _pad_1d_long([item["target_positions"] for item in batch], pad_value=0)
    target_stop, _ = _pad_2d_float([item["target_stop"].unsqueeze(-1) for item in batch])
    target_valid, _ = _pad_1d_bool([item["target_valid_mask"] for item in batch])
    result["target_coarse_ids"] = target_coarse
    result["target_offset_bins"] = target_offset
    result["target_xyz"] = target_xyz
    result["target_bin_center_xyz"] = target_bin_center
    result["target_positions"] = target_positions
    result["target_stop"] = target_stop.squeeze(-1)
    result["target_mask"] = target_mask
    result["target_valid_mask"] = target_valid & target_mask
    result["target_supervision_mask"] = result["target_valid_mask"]
    return result


def load_dataset_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


__all__ = [
    "AutoregFiberDataset",
    "FiberSamplePlan",
    "autoreg_fiber_collate",
    "load_dataset_config",
    "split_indices_by_fiber_id",
]
