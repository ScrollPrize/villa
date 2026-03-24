from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ink.core.types import Batch, BatchMeta, DataBundle
from ink.recipes.data.normalization import ClipMaxDiv255Normalization
from ink.recipes.data.samplers import GroupBalancedSampler, GroupStratifiedSampler, ShuffleSampler
from ink.recipes.data.transforms import (
    apply_eval_sample_transforms,
    apply_train_sample_transforms,
    build_joint_transform,
)


def _default_valid_mask_like(label):
    return np.ones_like(label, dtype=np.uint8)


@dataclass(frozen=True)
class InMemoryPatchSamples:
    images: tuple[Any, ...]
    labels: tuple[Any, ...]
    groups: tuple[int, ...] = field(default_factory=tuple)
    xyxys: tuple[Any, ...] | None = None
    valid_masks: tuple[Any, ...] | None = None

    def __post_init__(self) -> None:
        images = tuple(self.images)
        labels = tuple(self.labels)
        raw_groups = tuple(self.groups)
        xyxys = None if self.xyxys is None else tuple(self.xyxys)
        valid_masks = None if self.valid_masks is None else tuple(self.valid_masks)

        if len(labels) != len(images):
            raise ValueError("labels must have the same length as images")
        if xyxys is not None and len(xyxys) != len(images):
            raise ValueError("xyxys must have the same length as images")
        if valid_masks is not None and len(valid_masks) != len(images):
            raise ValueError("valid_masks must have the same length as images")

        if raw_groups and len(raw_groups) != len(images):
            raise ValueError("groups must have the same length as images")
        normalized_groups = tuple(int(group_idx) for group_idx in raw_groups)
        if any(group_idx < 0 for group_idx in normalized_groups):
            raise ValueError("groups must be non-negative")

        object.__setattr__(self, "images", images)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "groups", normalized_groups)
        object.__setattr__(self, "xyxys", xyxys)
        object.__setattr__(self, "valid_masks", valid_masks)

    def __len__(self) -> int:
        return len(self.images)

    def group_counts(self, *, default_if_missing: bool = False) -> list[int]:
        groups = self.groups
        if not groups and default_if_missing:
            groups = tuple(0 for _ in range(len(self.images)))
        if not groups:
            return []
        counts = [0] * (max(groups) + 1)
        for group_idx in groups:
            counts[int(group_idx)] += 1
        return counts


class InMemoryPatchDataset(Dataset):
    def __init__(
        self,
        samples: InMemoryPatchSamples,
        *,
        split: str,
        augment,
        normalization,
        patch_size: int,
        in_channels: int,
    ):
        self.samples = samples
        self.split = str(split).strip().lower()
        self.augment = augment
        self.normalization = normalization
        self.patch_size = int(patch_size)
        self.in_channels = int(in_channels)
        if self.split not in {"train", "valid"}:
            raise ValueError(f"unknown split: {split!r}")

        self._xyxys = self.samples.xyxys
        if self._xyxys is None:
            self._xyxys = tuple((0, 0, self.patch_size, self.patch_size) for _ in range(len(self.samples)))

        self.transform = build_joint_transform(
            self.split,
            augment=self.augment,
            normalization=self.normalization,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        idx = int(idx)
        image = self.samples.images[idx]
        label = self.samples.labels[idx]
        valid_mask = _default_valid_mask_like(label) if self.samples.valid_masks is None else self.samples.valid_masks[idx]
        if self.split == "train":
            image, label, valid_mask = apply_train_sample_transforms(
                image,
                label,
                augment=self.augment,
                patch_size=self.patch_size,
                transform=self.transform,
                valid_mask=valid_mask,
            )
        else:
            image, label, valid_mask = apply_eval_sample_transforms(
                image,
                label,
                patch_size=self.patch_size,
                transform=self.transform,
                valid_mask=valid_mask,
            )
        xyxy = np.asarray(self._xyxys[idx], dtype=np.int64)
        return image, label, valid_mask, xyxy


class GroupedInMemoryPatchDataset(InMemoryPatchDataset):
    def __init__(self, samples: InMemoryPatchSamples, **kwargs):
        super().__init__(samples, **kwargs)
        self._group_idxs = self.samples.groups or tuple(0 for _ in range(len(self.samples)))

    @property
    def sample_groups(self) -> list[int]:
        return list(self._group_idxs)

    def __getitem__(self, idx):
        idx = int(idx)
        image, label, valid_mask, xyxy = super().__getitem__(idx)
        return image, label, valid_mask, xyxy, self._group_idxs[idx]


def _collate_batch(samples) -> Batch:
    images, labels, valid_masks, xyxys = zip(*samples)
    meta = BatchMeta(
        segment_ids=[""] * len(samples),
        valid_mask=torch.stack([torch.as_tensor(mask) for mask in valid_masks], dim=0),
        patch_xyxy=torch.as_tensor(np.asarray(xyxys), dtype=torch.long),
    )
    return Batch(
        x=torch.stack([torch.as_tensor(image) for image in images], dim=0),
        y=torch.stack([torch.as_tensor(label) for label in labels], dim=0),
        meta=meta,
    )


def _collate_grouped_batch(samples) -> Batch:
    images, labels, valid_masks, xyxys, groups = zip(*samples)
    meta = BatchMeta(
        segment_ids=[""] * len(samples),
        valid_mask=torch.stack([torch.as_tensor(mask) for mask in valid_masks], dim=0),
        patch_xyxy=torch.as_tensor(np.asarray(xyxys), dtype=torch.long),
        group_idx=torch.as_tensor(groups, dtype=torch.long),
    )
    return Batch(
        x=torch.stack([torch.as_tensor(image) for image in images], dim=0),
        y=torch.stack([torch.as_tensor(label) for label in labels], dim=0),
        meta=meta,
    )
@dataclass(frozen=True)
class InMemoryPatchDataRecipe:
    train: InMemoryPatchSamples
    val: InMemoryPatchSamples
    in_channels: int
    patch_size: int
    train_batch_size: int = 1
    valid_batch_size: int | None = None
    num_workers: int = 0
    shuffle: bool = True
    sampler: ShuffleSampler = field(default_factory=ShuffleSampler)
    normalization: Any = field(default_factory=ClipMaxDiv255Normalization)
    extras: dict[str, Any] = field(default_factory=dict)

    def build(self, *, runtime=None, augment=None) -> DataBundle:
        assert augment is not None
        patch_size = int(self.patch_size)
        in_channels = int(self.in_channels)
        valid_batch_size = self.train_batch_size if self.valid_batch_size is None else self.valid_batch_size

        extras = dict(self.extras)
        normalization_stats = extras.pop("normalization_stats", None)
        normalization = self.normalization.build(
            normalization_stats=normalization_stats,
        )
        augment = augment.build(
            patch_size=patch_size,
            runtime=runtime,
        )
        group_counts = extras.pop("group_counts", None)
        bundle_extras = dict(extras)

        dataset_kwargs = {
            "augment": augment,
            "normalization": normalization,
            "patch_size": patch_size,
            "in_channels": in_channels,
        }
        train_dataset = InMemoryPatchDataset(self.train, split="train", **dataset_kwargs)
        val_dataset = InMemoryPatchDataset(self.val, split="valid", **dataset_kwargs)
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
        eval_loader = DataLoader(
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
            eval_loader=eval_loader,
            in_channels=in_channels,
            augment=augment,
            group_counts=group_counts,
            extras=bundle_extras,
        )


@dataclass(frozen=True)
class GroupedInMemoryPatchDataRecipe:
    train: InMemoryPatchSamples
    val: InMemoryPatchSamples
    in_channels: int
    patch_size: int
    train_batch_size: int = 1
    valid_batch_size: int | None = None
    num_workers: int = 0
    shuffle: bool = True
    sampler: ShuffleSampler | GroupBalancedSampler | GroupStratifiedSampler = field(default_factory=ShuffleSampler)
    normalization: Any = field(default_factory=ClipMaxDiv255Normalization)
    extras: dict[str, Any] = field(default_factory=dict)

    def build(self, *, runtime=None, augment=None) -> DataBundle:
        assert augment is not None
        patch_size = int(self.patch_size)
        in_channels = int(self.in_channels)
        valid_batch_size = self.train_batch_size if self.valid_batch_size is None else self.valid_batch_size

        extras = dict(self.extras)
        normalization_stats = extras.pop("normalization_stats", None)
        group_counts = extras.pop("group_counts", None)
        if group_counts is None:
            group_counts = self.train.group_counts(default_if_missing=True)
            if not group_counts:
                group_counts = None

        normalization = self.normalization.build(
            normalization_stats=normalization_stats,
        )
        augment = augment.build(
            patch_size=patch_size,
            runtime=runtime,
        )
        bundle_extras = dict(extras)

        dataset_kwargs = {
            "augment": augment,
            "normalization": normalization,
            "patch_size": patch_size,
            "in_channels": in_channels,
        }
        train_dataset = GroupedInMemoryPatchDataset(self.train, split="train", **dataset_kwargs)
        val_dataset = GroupedInMemoryPatchDataset(self.val, split="valid", **dataset_kwargs)
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
        eval_loader = DataLoader(
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
            eval_loader=eval_loader,
            in_channels=in_channels,
            augment=augment,
            group_counts=group_counts,
            extras=bundle_extras,
        )


__all__ = [
    "GroupedInMemoryPatchDataRecipe",
    "InMemoryPatchDataRecipe",
    "InMemoryPatchSamples",
]
