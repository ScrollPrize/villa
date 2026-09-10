"""Coordinate-driven full-volume patches for the v3 self-distillation phases."""

from __future__ import annotations

import random
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from vesuvius.ink_detection.config import NormalizationConfig
from vesuvius.ink_detection.data.normalization import normalize_image
from vesuvius.ink_detection.volume_io import open_volume, read_bbox_with_padding


class CoordPatchDataset(Dataset):
    """Read normalized cubic patches centered on authored XYZ coordinates."""

    def __init__(
        self,
        *,
        volume_path: str,
        resolution: int,
        coords_xyz: Iterable[Iterable[int]],
        jitter: int,
        length: int,
        patch_size: tuple[int, int, int],
        normalization: NormalizationConfig,
        input_mask_threshold: float,
        volume_auth_json=None,
        volume_cache_dir=None,
        volume_cache_max_gb: float | None = None,
    ) -> None:
        coords = []
        for index, coord in enumerate(coords_xyz):
            values = tuple(int(item) for item in coord)
            if len(values) != 3:
                raise ValueError(f"coords_xyz[{index}] must be one XYZ triple")
            coords.append(values)
        if not coords:
            raise ValueError("coords_xyz cannot be empty")
        if any(int(size) <= 0 for size in patch_size):
            raise ValueError("patch_size dimensions must be positive")
        if int(jitter) < 0:
            raise ValueError("jitter must be nonnegative")
        self.volume_path = str(volume_path)
        self.resolution = int(resolution)
        self.coords_xyz = tuple(coords)
        self.jitter = int(jitter)
        self.length = max(1, int(length))
        self.patch_size = tuple(int(size) for size in patch_size)
        self.normalization = normalization
        self.input_mask_threshold = float(input_mask_threshold)
        self.volume_auth_json = volume_auth_json
        self.volume_cache_dir = volume_cache_dir
        self.volume_cache_max_gb = volume_cache_max_gb
        self._volume = None

    def __len__(self) -> int:
        return self.length

    def _open(self):
        if self._volume is None:
            self._volume = open_volume(
                self.volume_path,
                self.resolution,
                self.volume_auth_json,
                cache_dir=self.volume_cache_dir,
                cache_max_gb=self.volume_cache_max_gb,
            )
        return self._volume

    def _center_zyx(self) -> tuple[int, int, int]:
        x, y, z = random.choice(self.coords_xyz)
        if self.jitter:
            x += random.randint(-self.jitter, self.jitter)
            y += random.randint(-self.jitter, self.jitter)
            z += random.randint(-self.jitter, self.jitter)
        return z, y, x

    def __getitem__(self, _index: int) -> dict[str, torch.Tensor]:
        center = self._center_zyx()
        starts = tuple(
            value - size // 2 for value, size in zip(center, self.patch_size)
        )
        bbox = (*starts, *(start + size for start, size in zip(starts, self.patch_size)))
        raw, valid_slices = read_bbox_with_padding(
            self._open(), bbox, fill_value=0
        )
        raw = np.asarray(raw)
        raw_mean = float(raw.mean())
        raw_std = float(raw.std())
        mask = (raw > self.input_mask_threshold).astype(np.float32)
        image = raw.astype(np.float32, copy=False)
        if valid_slices is not None:
            image[valid_slices] = normalize_image(
                image[valid_slices], self.normalization
            )
        image_tensor = torch.from_numpy(np.ascontiguousarray(image)).float().unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)
        return {
            "image": image_tensor,
            "image_for_label": image_tensor.clone(),
            "image_mask_for_label": mask_tensor,
            "image_raw_mean": torch.tensor(raw_mean, dtype=torch.float32),
            "image_raw_std": torch.tensor(raw_std, dtype=torch.float32),
            "inklabels": torch.zeros_like(image_tensor),
            "supervision_mask": mask_tensor.clone(),
            "is_unlabeled": torch.tensor(False, dtype=torch.bool),
        }
