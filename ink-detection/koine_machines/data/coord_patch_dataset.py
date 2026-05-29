"""Coord-driven patch dataset for v3 finetuning.

Returns synthetic 256^3 cubes centered on user-supplied XYZ coordinates (with
optional jitter), matching the dict schema produced by InkDataset in
mode="full_3d" so the trainer can mix these patches into the normal training
stream via ConcatDataset + WeightedRandomSampler.

This dataset has no associated ground-truth ink labels — it is meant to be used
together with the v3 SelfDistillLabelGenerator, which replaces batch['inklabels']
with model-driven pseudo-labels in the trainer step.
"""
from __future__ import annotations

import random
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from koine_machines.data.ink_dataset import _normalize_image_crop
from koine_machines.inference.infer_full3d_tifxyz import (
    _read_bbox_with_padding,
    open_zarr_array,
)


_CHUNK_SIZE = 256
_HALF = _CHUNK_SIZE // 2


class CoordPatchDataset(Dataset):
    """Synthetic 256^3 patches around a list of XYZ coordinates.

    Args:
        volume_uri: S3/local URI of the OME-Zarr root (slicing is ZYX).
        resolution: zarr resolution level (e.g. "0").
        coords_xyz: list of [X, Y, Z] (XYZ — will be swapped to ZYX internally).
        jitter: per-axis uniform jitter in voxels. Cube center = coord ± jitter.
        length: conceptual length for the sampler. Each __getitem__ re-randomizes,
                so the actual identity of any index is irrelevant.
        normalize_config: full training config (used by `_normalize_image_crop`).
        input_mask_threshold: raw uint8 voxels at or below this become background.
    """

    def __init__(
        self,
        *,
        volume_uri: str,
        resolution: str,
        coords_xyz: Iterable[Iterable[int]],
        jitter: int,
        length: int,
        normalize_config: dict,
        input_mask_threshold: float = 50.0,
    ):
        self._volume_uri = str(volume_uri)
        self._resolution = str(resolution)
        coords = []
        for c in coords_xyz:
            xs = list(c)
            if len(xs) != 3:
                raise ValueError(f"coord {c!r} is not XYZ")
            coords.append((int(xs[0]), int(xs[1]), int(xs[2])))
        if not coords:
            raise ValueError("coords_xyz must contain at least one XYZ triple")
        self._coords_xyz = coords
        self._jitter = int(jitter)
        self._length = max(1, int(length))
        self._normalize_config = normalize_config
        self._input_mask_threshold = float(input_mask_threshold)
        self._array = None  # opened lazily per worker

    def __len__(self) -> int:
        return self._length

    def _ensure_array(self):
        if self._array is None:
            self._array = open_zarr_array(self._volume_uri, self._resolution)
        return self._array

    def _pick_center_zyx(self) -> tuple[int, int, int]:
        x, y, z = random.choice(self._coords_xyz)
        if self._jitter > 0:
            jx = random.randint(-self._jitter, self._jitter)
            jy = random.randint(-self._jitter, self._jitter)
            jz = random.randint(-self._jitter, self._jitter)
            x += jx
            y += jy
            z += jz
        return int(z), int(y), int(x)

    def __getitem__(self, idx: int) -> dict:
        arr = self._ensure_array()
        cz, cy, cx = self._pick_center_zyx()
        bbox = (
            cz - _HALF, cy - _HALF, cx - _HALF,
            cz - _HALF + _CHUNK_SIZE, cy - _HALF + _CHUNK_SIZE, cx - _HALF + _CHUNK_SIZE,
        )
        crop_uint8, valid_slices = _read_bbox_with_padding(arr, bbox, fill_value=0)

        crop = crop_uint8.astype(np.float32, copy=False)
        raw_mean = float(crop.mean())
        raw_std = float(crop.std())

        mask = (crop > self._input_mask_threshold).astype(np.float32)

        if valid_slices is not None:
            crop[valid_slices] = _normalize_image_crop(
                crop[valid_slices], self._normalize_config
            )
        else:
            # Whole crop is padding — leave at 0.0 (model will see all-background).
            pass

        image = torch.from_numpy(np.ascontiguousarray(crop)).float().unsqueeze(0)
        mask_t = torch.from_numpy(mask).float().unsqueeze(0)
        zeros = torch.zeros_like(image)

        return {
            "image": image,
            "image_for_label": image.clone(),
            "image_mask_for_label": mask_t,
            "image_raw_mean": torch.tensor(raw_mean, dtype=torch.float32),
            "image_raw_std": torch.tensor(raw_std, dtype=torch.float32),
            "inklabels": zeros,
            "supervision_mask": mask_t.clone(),
            "is_unlabeled": torch.tensor(False, dtype=torch.bool),
        }
