from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import logging
import multiprocessing as mp
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ink.core.types import Batch, BatchMeta, DataBundle
from ink.recipes.data.layout import NestedZarrLayout
from ink.recipes.data.normalization import ClipMaxDiv255Normalization
from ink.recipes.data.patch_index_cache import load_cached_patch_index, save_cached_patch_index
from ink.recipes.data.patching import build_patch_index
from ink.recipes.data.samplers import ShuffleSampler
from ink.recipes.data.transforms import (
    apply_eval_sample_transforms,
    apply_train_sample_transforms,
    build_joint_transform,
)
from ink.recipes.data.zarr_io import (
    ZarrSegmentLabelMaskStore,
    ZarrSegmentVolume,
    read_supervision_mask_for_shape,
    resolve_segment_label_mask_store,
    resolve_segment_volume,
)

_PROGRESS_LOG = logging.getLogger("ink.progress")


def log_data_progress(message: str) -> None:
    _PROGRESS_LOG.info(str(message))


@dataclass
class ZarrDataContext:
    layout: NestedZarrLayout
    segments: Mapping[str, Any]
    in_channels: int
    label_suffix: str = ""
    mask_suffix: str = ""
    patch_index_cache_dir: str | None = None
    volume_cache: dict[Any, ZarrSegmentVolume] = field(default_factory=dict)
    label_mask_store_cache: dict[Any, ZarrSegmentLabelMaskStore] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.in_channels = int(self.in_channels)
        self.label_suffix = str(self.label_suffix)
        self.mask_suffix = str(self.mask_suffix)
        self.patch_index_cache_dir = None if self.patch_index_cache_dir is None else str(self.patch_index_cache_dir)

    @classmethod
    def from_recipe(cls, recipe) -> "ZarrDataContext":
        return cls(
            layout=NestedZarrLayout(recipe.dataset_root),
            segments=recipe.segments,
            in_channels=int(recipe.in_channels),
            label_suffix=str(recipe.label_suffix),
            mask_suffix=str(recipe.mask_suffix),
            patch_index_cache_dir=recipe.patch_index_cache_dir,
        )

    @classmethod
    def from_dataset(cls, dataset) -> "ZarrDataContext":
        return cls(
            layout=dataset.layout,
            segments=dataset.segments,
            in_channels=int(dataset.in_channels),
            label_suffix=str(dataset.label_suffix),
            mask_suffix=str(dataset.mask_suffix),
            patch_index_cache_dir=dataset.patch_index_cache_dir,
            volume_cache=dataset._volume_cache,
            label_mask_store_cache=dataset._label_mask_store_cache,
        )


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
    sampler: ShuffleSampler = field(default_factory=ShuffleSampler)
    normalization: Any = field(default_factory=ClipMaxDiv255Normalization)
    patch_index_cache_dir: str | None = None
    cache_train_patches_in_memory: bool = False
    include_train_valid_mask: bool = True
    extras: dict[str, Any] = field(default_factory=dict)

    def split_segment_ids(self) -> dict[str, tuple[str, ...]]:
        return {
            "train": tuple(str(segment_id).strip() for segment_id in self.train_segment_ids),
            "valid": tuple(str(segment_id).strip() for segment_id in self.val_segment_ids),
        }

    def build(self, *, runtime=None, augment=None) -> DataBundle:
        assert augment is not None
        patch_size = int(self.patch_size)
        in_channels = int(self.in_channels)
        tile_size = patch_size if self.tile_size is None else int(self.tile_size)
        stride = patch_size if self.stride is None else int(self.stride)
        context = ZarrDataContext.from_recipe(self)
        log_data_progress(f"[data] dataset_root={self.dataset_root}")

        split_segment_ids = self.split_segment_ids()
        samples_by_split = {
            split: build_zarr_split_samples(
                context,
                segment_ids=segment_ids,
                patch_size=patch_size,
                tile_size=tile_size,
                stride=stride,
                split_name=split,
                build_workers=max(0, int(self.num_workers)),
            )
            for split, segment_ids in split_segment_ids.items()
        }

        log_data_progress(f"[data] train segments={len(self.train_segment_ids)} patches={len(samples_by_split['train'])}")
        log_data_progress(f"[data] eval segments={len(self.val_segment_ids)} patches={len(samples_by_split['valid'])}")

        extras = dict(self.extras or {})
        group_counts = extras.pop("group_counts", None)
        normalization_stats = extras.pop("normalization_stats", None)
        normalization = self.normalization.build(normalization_stats=normalization_stats)
        augment_recipe = augment.build(patch_size=patch_size, runtime=runtime)
        valid_batch_size = self.train_batch_size if self.valid_batch_size is None else self.valid_batch_size

        dataset_kwargs = {
            "layout": context.layout,
            "segments": context.segments,
            "augment": augment_recipe,
            "normalization": normalization,
            "patch_size": patch_size,
            "tile_size": tile_size,
            "stride": stride,
            "in_channels": in_channels,
            "label_suffix": context.label_suffix,
            "mask_suffix": context.mask_suffix,
            "volume_cache": context.volume_cache,
            "label_mask_store_cache": context.label_mask_store_cache,
            "patch_index_cache_dir": context.patch_index_cache_dir,
        }
        train_dataset = ZarrPatchDataset(
            samples_by_split["train"],
            split="train",
            segment_ids=split_segment_ids["train"],
            cache_patches_in_memory=bool(self.cache_train_patches_in_memory),
            include_valid_mask=bool(self.include_train_valid_mask),
            **dataset_kwargs,
        )
        eval_dataset = ZarrPatchDataset(
            samples_by_split["valid"],
            split="valid",
            segment_ids=split_segment_ids["valid"],
            cache_patches_in_memory=False,
            include_valid_mask=True,
            **dataset_kwargs,
        )

        train_patch_ram_cache = "disabled"
        if train_dataset.cache_patches_in_memory:
            train_patch_ram_cache = (
                "preload_then_fork_share" if int(self.num_workers) > 0 else "lazy_single_process"
            )
        log_data_progress(
            "[data] caches "
            f"volume_entries={len(context.volume_cache)} "
            f"label_mask_store_entries={len(context.label_mask_store_cache)} "
            f"volume_cache_scope=in_process "
            f"label_mask_store_cache_scope=in_process "
            f"train_patch_ram_cache={train_patch_ram_cache}"
        )
        return build_patch_data_bundle(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            train_batch_size=int(self.train_batch_size),
            valid_batch_size=int(valid_batch_size),
            num_workers=max(0, int(self.num_workers)),
            shuffle=bool(self.shuffle),
            sampler=self.sampler,
            collate_fn=collate_patch_batch,
            in_channels=in_channels,
            augment_recipe=augment_recipe,
            group_counts=group_counts,
            extras=extras,
        )


class ZarrPatchDataset(Dataset):
    def __init__(
        self,
        samples,
        *,
        layout: NestedZarrLayout,
        segments,
        split: str,
        augment,
        normalization,
        patch_size: int,
        tile_size: int,
        stride: int,
        in_channels: int,
        label_suffix: str = "",
        mask_suffix: str = "",
        volume_cache: dict[Any, ZarrSegmentVolume] | None = None,
        label_mask_store_cache: dict[Any, ZarrSegmentLabelMaskStore] | None = None,
        patch_index_cache_dir: str | None = None,
        cache_patches_in_memory: bool = False,
        include_valid_mask: bool = True,
        segment_ids=(),
    ):
        self.layout = layout
        self.segments = segments
        self.split = str(split).strip().lower()
        self.augment = augment
        self.normalization = normalization
        self.patch_size = int(patch_size)
        self.tile_size = int(tile_size)
        self.stride = int(stride)
        self.in_channels = int(in_channels)
        self.label_suffix = str(label_suffix)
        self.mask_suffix = str(mask_suffix)
        self.patch_index_cache_dir = None if patch_index_cache_dir is None else str(patch_index_cache_dir)
        self.cache_patches_in_memory = bool(cache_patches_in_memory)
        if self.split not in {"train", "valid"}:
            raise ValueError(f"unknown split: {split!r}")
        self.include_valid_mask = bool(include_valid_mask) if self.split == "train" else True

        self._samples = []
        for segment_id, xyxy, bbox_index in samples:
            self._samples.append(
                (
                    str(segment_id),
                    tuple(int(value) for value in xyxy),
                    int(bbox_index),
                )
            )
        if segment_ids:
            self.segment_ids = tuple(str(segment_id) for segment_id in segment_ids)
        else:
            seen = set()
            ordered_segment_ids = []
            for segment_id, *_rest in self._samples:
                if segment_id in seen:
                    continue
                seen.add(segment_id)
                ordered_segment_ids.append(segment_id)
            self.segment_ids = tuple(ordered_segment_ids)

        self._volume_cache = {} if volume_cache is None else volume_cache
        self._label_mask_store_cache = {} if label_mask_store_cache is None else label_mask_store_cache
        self._patch_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, str]] = {}
        self.transform = build_joint_transform(
            self.split,
            augment=self.augment,
            normalization=self.normalization,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
        )

    def data_context(self) -> ZarrDataContext:
        return ZarrDataContext.from_dataset(self)

    @property
    def sample_rows(self) -> tuple[tuple[str, tuple[int, int, int, int], int], ...]:
        return tuple(self._samples)

    def __len__(self) -> int:
        return len(self._samples)

    def _cache_raw_sample(self, idx: int):
        idx = int(idx)
        cached = self._patch_cache.get(idx)
        if cached is not None:
            return cached, 0
        raw_sample = load_raw_zarr_patch(
            self.data_context(),
            sample=self._samples[idx],
            include_valid_mask=self.include_valid_mask,
        )
        image, label, valid_mask, xyxy, segment_id = raw_sample
        cached = (
            _cache_ready_array(image),
            _cache_ready_array(label),
            _cache_ready_array(valid_mask),
            _cache_ready_array(xyxy),
            str(segment_id),
        )
        self._patch_cache[idx] = cached
        return cached, _cached_patch_nbytes(*cached[:4])

    def _prepare_worker_processes(self, *, num_workers: int):
        if int(num_workers) <= 0 or not self.cache_patches_in_memory:
            return None
        if "fork" not in mp.get_all_start_methods():
            log_data_progress("[data] train patch cache mode=worker_local shared_with_workers=False")
            return None
        warmed_bytes = 0
        for idx in range(len(self._samples)):
            _cached_sample, loaded_bytes = self._cache_raw_sample(idx)
            warmed_bytes += int(loaded_bytes)
        log_data_progress(
            "[data] train patch cache "
            f"samples={len(self._patch_cache)} "
            f"bytes={warmed_bytes} "
            "shared_with_workers=True"
        )
        return "fork"

    def _load_item(self, idx):
        idx = int(idx)
        if self.cache_patches_in_memory:
            image, label, valid_mask, _xyxy, _segment_id = self._cache_raw_sample(idx)[0]
        else:
            image, label, valid_mask, _xyxy, _segment_id = load_raw_zarr_patch(
                self.data_context(),
                sample=self._samples[idx],
                include_valid_mask=self.include_valid_mask,
            )
        if self.split == "train":
            transformed = apply_train_sample_transforms(
                image,
                label,
                augment=self.augment,
                patch_size=self.patch_size,
                transform=self.transform,
                valid_mask=valid_mask,
            )
            if valid_mask is None:
                image, label = transformed
                return image, label, None
            return transformed
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
        segment_id, (x1, y1, x2, y2), _bbox_index = self._samples[idx]
        xyxy = np.asarray((x1, y1, x2, y2), dtype=np.int64)
        return image, label, valid_mask, xyxy, segment_id


def _cache_ready_array(array) -> np.ndarray | None:
    if array is None:
        return None
    return np.ascontiguousarray(np.asarray(array))


def _cached_patch_nbytes(image, label, valid_mask, xyxy) -> int:
    return sum(int(np.asarray(part).nbytes) for part in (image, label, valid_mask, xyxy) if part is not None)


def _prepare_train_loader_multiprocessing_context(dataset, *, num_workers: int):
    if int(num_workers) <= 0:
        return None
    prepare = getattr(dataset, "_prepare_worker_processes", None)
    if not callable(prepare):
        return None
    return prepare(num_workers=int(num_workers))

def build_zarr_split_samples(
    context: ZarrDataContext,
    *,
    segment_ids: tuple[str, ...],
    patch_size: int,
    tile_size: int,
    stride: int,
    split_name: str = "",
    build_workers: int = 0,
) -> list[tuple[str, tuple[int, int, int, int], int]]:
    split_samples = []

    def _build_one(segment_id: str):
        log_data_progress(f"[data] {split_name} loading segment={segment_id}")
        volume = resolve_segment_volume(
            layout=context.layout,
            segments=context.segments,
            segment_id=segment_id,
            in_channels=context.in_channels,
            volume_cache=context.volume_cache,
        )
        cached_index = load_cached_patch_index(
            cache_dir=context.patch_index_cache_dir,
            segment_id=segment_id,
            split_name=split_name,
            patch_size=patch_size,
            tile_size=tile_size,
            stride=stride,
            label_suffix=context.label_suffix,
            mask_suffix=context.mask_suffix,
            log=log_data_progress,
        )
        if cached_index is None:
            supervision_mask = read_supervision_mask_for_shape(
                context.layout,
                segment_id,
                volume.image_shape_hw,
                mask_suffix=context.mask_suffix,
            )
            bbox_rows, xyxys, sample_bbox_indices = build_patch_index(
                None,
                supervision_mask,
                size=patch_size,
                tile_size=tile_size,
                stride=stride,
                filter_empty_tile=False,
            )
            save_cached_patch_index(
                cache_dir=context.patch_index_cache_dir,
                segment_id=segment_id,
                split_name=split_name,
                patch_size=patch_size,
                tile_size=tile_size,
                stride=stride,
                label_suffix=context.label_suffix,
                mask_suffix=context.mask_suffix,
                xyxys=xyxys,
                bbox_rows=bbox_rows,
                sample_bbox_indices=sample_bbox_indices,
            )
        else:
            xyxys = np.asarray(cached_index.xyxys, dtype=np.int64)
            bbox_rows = np.asarray(cached_index.bbox_rows, dtype=np.int32)
            sample_bbox_indices = np.asarray(cached_index.sample_bbox_indices, dtype=np.int32)

        resolve_segment_label_mask_store(
            layout=context.layout,
            segment_id=segment_id,
            image_shape_hw=volume.image_shape_hw,
            label_suffix=context.label_suffix,
            mask_suffix=context.mask_suffix,
            label_mask_store_cache=context.label_mask_store_cache,
            bbox_rows=bbox_rows,
        )

        patch_rows = [
            (
                segment_id,
                tuple(int(value) for value in xyxy),
                int(sample_bbox_indices[row_idx]),
            )
            for row_idx, xyxy in enumerate(np.asarray(xyxys, dtype=np.int64))
        ]
        log_data_progress(f"[data] {split_name} segment={segment_id} patches={len(patch_rows)}")
        return patch_rows

    build_workers = max(0, int(build_workers))
    if build_workers > 1 and len(segment_ids) > 1:
        with ThreadPoolExecutor(max_workers=min(build_workers, len(segment_ids))) as executor:
            for patch_rows in executor.map(_build_one, segment_ids):
                split_samples.extend(patch_rows)
        return split_samples

    for segment_id in segment_ids:
        split_samples.extend(_build_one(segment_id))
    return split_samples


def load_raw_zarr_patch(
    context: ZarrDataContext,
    *,
    sample,
    include_valid_mask: bool,
):
    segment_id, xyxy, bbox_index = sample[:3]
    x1, y1, x2, y2 = [int(value) for value in xyxy]
    volume = resolve_segment_volume(
        layout=context.layout,
        segments=context.segments,
        segment_id=segment_id,
        in_channels=context.in_channels,
        volume_cache=context.volume_cache,
    )

    label_mask_store = resolve_segment_label_mask_store(
        layout=context.layout,
        segment_id=segment_id,
        image_shape_hw=volume.image_shape_hw,
        label_suffix=context.label_suffix,
        mask_suffix=context.mask_suffix,
        label_mask_store_cache=context.label_mask_store_cache,
    )
    if include_valid_mask:
        label_mask, supervision_mask = label_mask_store.read_patch(
            y1=y1,
            y2=y2,
            x1=x1,
            x2=x2,
            bbox_index=int(bbox_index),
        )
    else:
        label_mask = label_mask_store.read_label_patch(
            y1=y1,
            y2=y2,
            x1=x1,
            x2=x2,
            bbox_index=int(bbox_index),
        )
        supervision_mask = None
    image = volume.read_patch(y1, y2, x1, x2)
    label = np.asarray(label_mask, dtype=np.uint8)[..., None]
    valid_mask = None if supervision_mask is None else np.asarray(supervision_mask, dtype=np.uint8)[..., None]
    return image, label, valid_mask, np.asarray((x1, y1, x2, y2), dtype=np.int64), str(segment_id)


def build_patch_data_bundle(
    *,
    train_dataset,
    eval_dataset,
    train_batch_size: int,
    valid_batch_size: int,
    num_workers: int,
    shuffle: bool,
    sampler,
    collate_fn,
    in_channels: int,
    augment_recipe,
    group_counts,
    extras,
) -> DataBundle:
    train_loader_multiprocessing_context = _prepare_train_loader_multiprocessing_context(
        train_dataset,
        num_workers=int(num_workers),
    )
    train_loader = sampler.build_loader(
        train_dataset,
        batch_size=int(train_batch_size),
        num_workers=int(num_workers),
        pin_memory=True,
        collate_fn=collate_fn,
        shuffle=bool(shuffle),
        multiprocessing_context=train_loader_multiprocessing_context,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=int(valid_batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        persistent_workers=bool(int(num_workers) > 0),
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )
    return DataBundle(
        train_loader=train_loader,
        eval_loader=eval_loader,
        in_channels=int(in_channels),
        augment=augment_recipe,
        group_counts=group_counts,
        extras=dict(extras or {}),
    )


def batch_from_parts(
    *,
    images,
    labels,
    valid_masks,
    xyxys,
    segment_ids,
    group_idxs=None,
) -> Batch:
    valid_mask_batch = None
    if len(valid_masks) > 0 and valid_masks[0] is not None:
        valid_mask_batch = _stack_batch_tensors(valid_masks)
    meta = BatchMeta(
        segment_ids=[str(segment_id) for segment_id in segment_ids],
        valid_mask=valid_mask_batch,
        patch_xyxy=torch.as_tensor(np.asarray(xyxys), dtype=torch.long),
        group_idx=None if group_idxs is None else torch.as_tensor(group_idxs, dtype=torch.long),
    )
    return Batch(
        x=_stack_batch_tensors(images),
        y=_stack_batch_tensors(labels),
        meta=meta,
    )


def collate_patch_batch(samples) -> Batch:
    images, labels, valid_masks, xyxys, segment_ids = zip(*samples)
    return batch_from_parts(
        images=images,
        labels=labels,
        valid_masks=valid_masks,
        xyxys=xyxys,
        segment_ids=segment_ids,
    )


def collate_grouped_batch(samples) -> Batch:
    images, labels, valid_masks, xyxys, segment_ids, group_idxs = zip(*samples)
    return batch_from_parts(
        images=images,
        labels=labels,
        valid_masks=valid_masks,
        xyxys=xyxys,
        segment_ids=segment_ids,
        group_idxs=group_idxs,
    )


def count_group_idxs(group_idxs) -> list[int] | None:
    group_counts = [0]
    for group_idx in group_idxs:
        while int(group_idx) >= len(group_counts):
            group_counts.append(0)
        group_counts[int(group_idx)] += 1
    if not any(group_counts):
        return None
    return group_counts


def _stack_batch_tensors(values):
    first = values[0]
    if isinstance(first, torch.Tensor):
        return torch.stack(list(values), dim=0)
    return torch.stack([torch.as_tensor(value) for value in values], dim=0)


__all__ = [
    "ZarrDataContext",
    "ZarrPatchDataRecipe",
    "ZarrPatchDataset",
    "batch_from_parts",
    "build_patch_data_bundle",
    "build_zarr_split_samples",
    "collate_grouped_batch",
    "collate_patch_batch",
    "count_group_idxs",
    "load_raw_zarr_patch",
    "log_data_progress",
]
