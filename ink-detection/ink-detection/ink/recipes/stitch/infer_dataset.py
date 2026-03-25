from __future__ import annotations

from typing import Any

import numpy as np
from torch.utils.data import Dataset

from ink.recipes.data.patching import candidate_patch_starts
from ink.recipes.data.transforms import apply_infer_sample_transforms, build_image_transform
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
        self.transform = build_image_transform(
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
        image = apply_infer_sample_transforms(
            image,
            transform=self.transform,
        )
        return image, np.asarray((x1, y1, x2, y2), dtype=np.int64), segment_id


class ZarrInferGridDataset(Dataset):
    def __init__(
        self,
        *,
        layout,
        segments,
        augment,
        normalization,
        patch_size: int,
        tile_size: int,
        stride: int,
        in_channels: int,
        segment_id: str,
        volume_cache: dict[Any, Any] | None = None,
        skip_empty: bool = False,
    ):
        self.layout = layout
        self.segments = segments
        self.augment = augment
        self.normalization = normalization
        self.patch_size = int(patch_size)
        self.tile_size = int(tile_size)
        self.stride = int(stride)
        self.in_channels = int(in_channels)
        self.segment_id = str(segment_id)
        self.segment_ids = (self.segment_id,)
        self.skip_empty = bool(skip_empty)
        self._volume_cache = {} if volume_cache is None else volume_cache
        self._volume = resolve_segment_volume(
            layout=self.layout,
            segments=self.segments,
            segment_id=self.segment_id,
            in_channels=int(self.in_channels),
            volume_cache=self._volume_cache,
        )
        image_h, image_w = [int(value) for value in self._volume.image_shape_hw]
        self._x_starts = candidate_patch_starts(
            image_w,
            size=self.patch_size,
            tile_size=self.tile_size,
            stride=self.stride,
        )
        self._y_starts = candidate_patch_starts(
            image_h,
            size=self.patch_size,
            tile_size=self.tile_size,
            stride=self.stride,
        )
        self._num_x = int(self._x_starts.size)
        self._num_y = int(self._y_starts.size)
        self.transform = build_image_transform(
            "valid",
            augment=self.augment,
            normalization=self.normalization,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
        )

    def __len__(self) -> int:
        return int(self._num_x * self._num_y)

    def __getitem__(self, idx):
        idx = int(idx)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        x_count = int(self._num_x)
        y_index, x_index = divmod(idx, x_count)
        x1 = int(self._x_starts[x_index])
        y1 = int(self._y_starts[y_index])
        x2 = int(x1 + self.patch_size)
        y2 = int(y1 + self.patch_size)
        image = self._volume.read_patch(y1, y2, x1, x2)
        if self.skip_empty and not bool(np.asarray(image).any()):
            return None
        image = apply_infer_sample_transforms(
            image,
            transform=self.transform,
        )
        return image, np.asarray((x1, y1, x2, y2), dtype=np.int64), self.segment_id


__all__ = ["ZarrInferDataset", "ZarrInferGridDataset"]
