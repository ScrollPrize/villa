from __future__ import annotations

from typing import Any

import numpy as np
from torch.utils.data import Dataset

from ink.recipes.data.transforms import apply_eval_sample_transforms, build_joint_transform
from ink.recipes.data.zarr_io import resolve_segment_volume


class ZarrInferDataset(Dataset):
    def __init__(
        self,
        samples,
        *,
        layout,
        segments,
        augment,
        normalization,
        patch_size: int,
        in_channels: int,
        volume_cache: dict[Any, Any] | None = None,
        segment_ids=(),
    ):
        self.layout = layout
        self.segments = segments
        self.augment = augment
        self.normalization = normalization
        self.patch_size = int(patch_size)
        self.in_channels = int(in_channels)
        self.segment_ids = tuple(str(segment_id) for segment_id in (segment_ids or ()))
        self._samples = [
            (
                str(segment_id),
                tuple(int(value) for value in xyxy),
            )
            for segment_id, xyxy in samples
        ]
        self._volume_cache = {} if volume_cache is None else volume_cache
        self.transform = build_joint_transform(
            "valid",
            augment=self.augment,
            normalization=self.normalization,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx):
        idx = int(idx)
        segment_id, (x1, y1, x2, y2) = self._samples[idx]
        volume = resolve_segment_volume(
            layout=self.layout,
            segments=self.segments,
            segment_id=segment_id,
            in_channels=int(self.in_channels),
            volume_cache=self._volume_cache,
        )
        image = volume.read_patch(y1, y2, x1, x2)
        label = np.zeros((int(y2 - y1), int(x2 - x1), 1), dtype=np.uint8)
        valid_mask = np.ones_like(label, dtype=np.uint8)
        image, label, valid_mask = apply_eval_sample_transforms(
            image,
            label,
            patch_size=self.patch_size,
            transform=self.transform,
            valid_mask=valid_mask,
        )
        return image, label, valid_mask, np.asarray((x1, y1, x2, y2), dtype=np.int64), segment_id


__all__ = ["ZarrInferDataset"]
