from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from ink.core.types import Batch, BatchMeta, DataBundle
from ink.recipes.data.layout import NestedZarrLayout
from ink.recipes.data.normalization import ClipMaxDiv255Normalization
from ink.recipes.data.samplers import GroupBalancedSampler, GroupStratifiedSampler, ShuffleSampler
from ink.recipes.data.zarr_data import ZarrPatchDataset, _build_samples_from_segments
from ink.recipes.data.zarr_io import ZarrSegmentVolume


def _collate_grouped_batch(samples) -> Batch:
    images, labels, valid_masks, xyxys, segment_ids, group_idxs = zip(*samples)

    meta = BatchMeta(
        segment_ids=[str(segment_id) for segment_id in segment_ids],
        valid_mask=torch.stack([torch.as_tensor(mask) for mask in valid_masks], dim=0),
        patch_xyxy=torch.as_tensor(np.asarray(xyxys), dtype=torch.long),
        group_idx=torch.as_tensor(group_idxs, dtype=torch.long),
    )

    return Batch(
        x=torch.stack([torch.as_tensor(image) for image in images], dim=0),
        y=torch.stack([torch.as_tensor(label) for label in labels], dim=0),
        meta=meta,
    )


class GroupedZarrPatchDataset(ZarrPatchDataset):
    def __init__(self, samples, **kwargs):
        self._group_idxs = [int(group_idx) for _, _, _, _, group_idx in samples]
        super().__init__([(segment_id, layer_range, reverse_layers, xyxy) for segment_id, layer_range, reverse_layers, xyxy, _group_idx in samples], **kwargs)

    @property
    def sample_groups(self) -> list[int]:
        return list(self._group_idxs)

    def __getitem__(self, idx):
        idx = int(idx)
        image, label, valid_mask, xyxy, segment_id = super().__getitem__(idx)
        return image, label, valid_mask, xyxy, segment_id, self._group_idxs[idx]


@dataclass(frozen=True)
class GroupedZarrPatchDataRecipe:
    dataset_root: str
    in_channels: int
    patch_size: int
    segments: Mapping[str, Any]
    train_segment_ids: tuple[str, ...]
    val_segment_ids: tuple[str, ...]
    tile_size: int | None = None
    stride: int | None = None
    label_suffix: str = ""
    mask_suffix: str = ""
    train_batch_size: int = 1
    valid_batch_size: int | None = None
    num_workers: int = 0
    shuffle: bool = True
    sampler: ShuffleSampler | GroupBalancedSampler | GroupStratifiedSampler = field(default_factory=ShuffleSampler)
    normalization: Any = field(default_factory=ClipMaxDiv255Normalization)
    extras: dict[str, Any] = field(default_factory=dict)

    def build(self, *, runtime=None, augment=None) -> DataBundle:
        assert augment is not None
        augment_recipe = augment
        segments = self.segments
        patch_size = int(self.patch_size)
        in_channels = int(self.in_channels)
        tile_size = patch_size if self.tile_size is None else int(self.tile_size)
        stride = patch_size if self.stride is None else int(self.stride)
        train_segment_ids = tuple(str(segment_id).strip() for segment_id in self.train_segment_ids)
        val_segment_ids = tuple(str(segment_id).strip() for segment_id in self.val_segment_ids)
        layout = NestedZarrLayout(self.dataset_root)
        group_name_to_idx = {
            group_name: idx
            for idx, group_name in enumerate(
                sorted({layout.resolve_group_name(segment_id) for segment_id in train_segment_ids + val_segment_ids})
            )
        }
        volume_cache: dict[Any, ZarrSegmentVolume] = {}
        mask_cache: dict[Any, tuple[np.ndarray, np.ndarray]] = {}
        split_segment_ids = {
            "train": train_segment_ids,
            "valid": val_segment_ids,
        }
        base_samples_by_split = {
            split: _build_samples_from_segments(
                layout=layout,
                segments=segments,
                segment_ids=segment_ids,
                in_channels=in_channels,
                patch_size=patch_size,
                tile_size=tile_size,
                stride=stride,
                label_suffix=self.label_suffix,
                mask_suffix=self.mask_suffix,
                volume_cache=volume_cache,
                mask_cache=mask_cache,
            )
            for split, segment_ids in split_segment_ids.items()
        }
        samples_by_split = {
            split: [
                (
                    segment_id,
                    layer_range,
                    reverse_layers,
                    xyxy,
                    int(group_name_to_idx[layout.resolve_group_name(segment_id)]),
                )
                for segment_id, layer_range, reverse_layers, xyxy in samples
            ]
            for split, samples in base_samples_by_split.items()
        }
        train_samples = samples_by_split["train"]

        valid_batch_size = self.train_batch_size if self.valid_batch_size is None else self.valid_batch_size
        extras = dict(self.extras or {})
        extras["patch_size"] = patch_size
        if "group_counts" not in extras:
            group_counts = [0] * len(group_name_to_idx)
            for _, _, _, _, group_idx in train_samples:
                group_counts[int(group_idx)] += 1
            if group_counts:
                extras["group_counts"] = group_counts

        bind_context = DataBundle(
            train_loader=None,
            val_loader=None,
            in_channels=in_channels,
            extras=dict(extras),
        )
        normalization = self.normalization.build(data=bind_context)
        bind_context = DataBundle(
            train_loader=None,
            val_loader=None,
            in_channels=in_channels,
            extras={**extras, "normalization": normalization},
        )
        augment_recipe = augment_recipe.build(data=bind_context, runtime=runtime)
        bundle_extras = {**bind_context.extras, "augment": augment_recipe}

        dataset_kwargs = {
            "layout": layout,
            "augment": augment_recipe,
            "normalization": normalization,
            "patch_size": patch_size,
            "in_channels": in_channels,
            "label_suffix": self.label_suffix,
            "mask_suffix": self.mask_suffix,
            "volume_cache": volume_cache,
            "mask_cache": mask_cache,
        }
        datasets_by_split = {
            split: GroupedZarrPatchDataset(samples, split=split, **dataset_kwargs)
            for split, samples in samples_by_split.items()
        }
        train_dataset = datasets_by_split["train"]
        val_dataset = datasets_by_split["valid"]
        num_workers = max(0, int(self.num_workers))
        train_batch_size = int(self.train_batch_size)
        valid_batch_size = int(valid_batch_size)
        train_loader = self.sampler.build_loader(
            train_dataset,
            batch_size=train_batch_size,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=_collate_grouped_batch,
            shuffle=bool(self.shuffle),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=valid_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=_collate_grouped_batch,
        )
        return DataBundle(
            train_loader=train_loader,
            val_loader=val_loader,
            in_channels=in_channels,
            extras=bundle_extras,
        )


__all__ = ["GroupedZarrPatchDataRecipe"]
