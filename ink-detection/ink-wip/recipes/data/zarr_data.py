from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ink.core.types import Batch, BatchMeta, DataBundle
from ink.recipes.data.layout import NestedZarrLayout
from ink.recipes.data.normalization import ClipMaxDiv255Normalization
from ink.recipes.data.patching import extract_patch_coordinates
from ink.recipes.data.samplers import (
    GroupBalancedSampler,
    GroupStratifiedSampler,
    ShuffleSampler,
)
from ink.recipes.data.transforms import (
    apply_eval_sample_transforms,
    apply_train_sample_transforms,
    build_joint_transform,
)
from ink.recipes.data.zarr import (
    ZarrSegmentVolume,
    parse_layer_range_value,
    read_label_and_supervision_mask_for_shape,
)


def _read_mask_patch(mask, *, y1: int, y2: int, x1: int, x2: int) -> np.ndarray:
    y1 = int(y1)
    y2 = int(y2)
    x1 = int(x1)
    x2 = int(x2)
    if y2 <= y1 or x2 <= x1:
        raise ValueError(f"invalid patch coords: {(x1, y1, x2, y2)}")

    mask = np.asarray(mask)
    out = np.zeros((int(y2 - y1), int(x2 - x1)), dtype=mask.dtype)
    yy1 = max(0, int(y1))
    yy2 = min(int(mask.shape[0]), int(y2))
    xx1 = max(0, int(x1))
    xx2 = min(int(mask.shape[1]), int(x2))
    if yy2 > yy1 and xx2 > xx1:
        out[yy1 - int(y1):yy2 - int(y1), xx1 - int(x1):xx2 - int(x1)] = mask[yy1:yy2, xx1:xx2]
    return out


def _collate_batch(samples) -> Batch:
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

def _build_samples_from_segments(
    *,
    layout: NestedZarrLayout,
    segments,
    segment_ids: tuple[str, ...],
    group_name_to_idx: dict[str, int],
    in_channels: int,
    patch_size: int,
    tile_size: int,
    stride: int,
    label_suffix: str,
    mask_suffix: str,
    volume_cache: dict[Any, ZarrSegmentVolume],
    mask_cache: dict[Any, tuple[np.ndarray, np.ndarray]],
) -> list[tuple[str, tuple[int, int], bool, tuple[int, int, int, int], int]]:
    split_samples = []
    for segment_id in segment_ids:
        segment_spec = segments[segment_id]
        layer_range = parse_layer_range_value(
            segment_spec.get("layer_range"),
            context=f"segments[{segment_id!r}].layer_range",
        )
        reverse_layers = bool(segment_spec.get("reverse_layers", False))

        volume_key = (segment_id, layer_range, reverse_layers, in_channels)
        volume = volume_cache.get(volume_key)
        if volume is None:
            volume = ZarrSegmentVolume(
                layout,
                segment_id,
                layer_range=layer_range,
                reverse_layers=reverse_layers,
                in_channels=in_channels,
            )
            volume_cache[volume_key] = volume

        mask_key = (segment_id, label_suffix, mask_suffix)
        masks = mask_cache.get(mask_key)
        if masks is None:
            masks = read_label_and_supervision_mask_for_shape(
                layout,
                segment_id,
                volume.image_shape_hw,
                label_suffix=label_suffix,
                mask_suffix=mask_suffix,
            )
            mask_cache[mask_key] = masks
        label_mask, supervision_mask = masks

        group_idx = int(group_name_to_idx[layout.resolve_group_name(segment_id)])
        split_samples.extend(
            (
                segment_id,
                layer_range,
                reverse_layers,
                tuple(int(value) for value in xyxy),
                group_idx,
            )
            for xyxy in extract_patch_coordinates(
                label_mask,
                supervision_mask,
                size=patch_size,
                tile_size=tile_size,
                stride=stride,
                filter_empty_tile=False,
            )
        )
    return split_samples


class ZarrPatchDataset(Dataset):
    def __init__(
        self,
        samples,
        *,
        layout: NestedZarrLayout,
        split: str,
        augment,
        normalization,
        patch_size: int,
        in_channels: int,
        label_suffix: str = "",
        mask_suffix: str = "",
        volume_cache: dict[Any, ZarrSegmentVolume] | None = None,
        mask_cache: dict[Any, tuple[np.ndarray, np.ndarray]] | None = None,
    ):
        self.layout = layout
        self.split = str(split).strip().lower()
        self.augment = augment
        self.normalization = normalization
        self.patch_size = int(patch_size)
        self.in_channels = int(in_channels)
        self.label_suffix = str(label_suffix)
        self.mask_suffix = str(mask_suffix)
        if self.split not in {"train", "valid"}:
            raise ValueError(f"unknown split: {split!r}")

        self._samples = [
            (
                str(segment_id),
                tuple(int(value) for value in layer_range),
                bool(reverse_layers),
                tuple(int(value) for value in xyxy),
                int(group_idx),
            )
            for segment_id, layer_range, reverse_layers, xyxy, group_idx in samples
        ]

        self._volume_cache = {} if volume_cache is None else volume_cache
        self._mask_cache = {} if mask_cache is None else mask_cache
        self.transform = build_joint_transform(
            self.split,
            augment=self.augment,
            normalization=self.normalization,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
        )

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def sample_groups(self) -> list[int]:
        return [group_idx for _, _, _, _, group_idx in self._samples]

    def _segment_volume(self, segment_id: str, layer_range, reverse_layers) -> ZarrSegmentVolume:
        segment_id = str(segment_id)
        cache_key = (segment_id, layer_range, reverse_layers, int(self.in_channels))
        volume = self._volume_cache.get(cache_key)
        if volume is None:
            volume = ZarrSegmentVolume(
                self.layout,
                segment_id,
                layer_range=layer_range,
                reverse_layers=reverse_layers,
                in_channels=self.in_channels,
            )
            self._volume_cache[cache_key] = volume
        return volume

    def _segment_masks(self, segment_id: str, *, image_shape_hw) -> tuple[np.ndarray, np.ndarray]:
        segment_id = str(segment_id)
        cache_key = (segment_id, self.label_suffix, self.mask_suffix)
        masks = self._mask_cache.get(cache_key)
        if masks is not None:
            return masks

        masks = read_label_and_supervision_mask_for_shape(
            self.layout,
            segment_id,
            image_shape_hw,
            label_suffix=self.label_suffix,
            mask_suffix=self.mask_suffix,
        )
        self._mask_cache[cache_key] = masks
        return masks

    def _load_item(self, idx):
        idx = int(idx)
        segment_id, layer_range, reverse_layers, (x1, y1, x2, y2), _group_idx = self._samples[idx]

        volume = self._segment_volume(segment_id, layer_range, reverse_layers)
        label_mask, supervision_mask = self._segment_masks(segment_id, image_shape_hw=volume.image_shape_hw)
        image = volume.read_patch(y1, y2, x1, x2)
        label = _read_mask_patch(label_mask, y1=y1, y2=y2, x1=x1, x2=x2)[..., None]
        valid_mask = _read_mask_patch(supervision_mask, y1=y1, y2=y2, x1=x1, x2=x2)[..., None]

        if self.split == "train":
            return apply_train_sample_transforms(
                image,
                label,
                augment=self.augment,
                patch_size=self.patch_size,
                transform=self.transform,
                valid_mask=valid_mask,
            )
        return apply_eval_sample_transforms(
            image,
            label,
            patch_size=self.patch_size,
            transform=self.transform,
            valid_mask=valid_mask,
        )

    def __getitem__(self, idx):
        idx = int(idx)
        image, label, valid_mask = self._load_item(idx)
        segment_id, _layer_range, _reverse_layers, (x1, y1, x2, y2), group_idx = self._samples[idx]
        xyxy = np.asarray((x1, y1, x2, y2), dtype=np.int64)
        return image, label, valid_mask, xyxy, segment_id, group_idx


@dataclass(frozen=True)
class ZarrPatchDataRecipe:
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
        samples_by_split = {
            split: _build_samples_from_segments(
                layout=layout,
                segments=segments,
                segment_ids=segment_ids,
                group_name_to_idx=group_name_to_idx,
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

        normalization = self.normalization
        bind_context = DataBundle(
            train_loader=None,
            val_loader=None,
            in_channels=in_channels,
            extras=dict(extras),
        )
        normalization = normalization.build(data=bind_context)
        bind_context = DataBundle(
            train_loader=None,
            val_loader=None,
            in_channels=in_channels,
            extras={**extras, "normalization": normalization},
        )
        augment_recipe = augment_recipe.build(data=bind_context, runtime=runtime)

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
            split: ZarrPatchDataset(samples, split=split, **dataset_kwargs)
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
            collate_fn=_collate_batch,
            shuffle=bool(self.shuffle),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=valid_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=_collate_batch,
        )
        return DataBundle(
            train_loader=train_loader,
            val_loader=val_loader,
            in_channels=in_channels,
            extras=dict(bind_context.extras),
        )


__all__ = ["ZarrPatchDataRecipe"]
