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
from ink.recipes.data.layout import NestedZarrLayout, resolve_layout_mask_names_for_segment
from ink.recipes.data.masks import (
    SUPERVISION_MASK_NAME,
    default_mask_name_for_split,
)
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
    split_name: str = "train"
    mask_split_name: str = ""
    mask_name: str = SUPERVISION_MASK_NAME
    train_segment_ids: frozenset[str] = field(default_factory=frozenset)
    patch_index_cache_dir: str | None = None
    volume_cache: dict[Any, ZarrSegmentVolume] = field(default_factory=dict)
    label_mask_store_cache: dict[Any, ZarrSegmentLabelMaskStore] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.in_channels = int(self.in_channels)
        self.label_suffix = str(self.label_suffix)
        self.mask_suffix = str(self.mask_suffix)
        self.split_name = str(self.split_name).strip().lower()
        self.mask_split_name = str(self.mask_split_name).strip().lower() or self.split_name
        self.mask_name = str(self.mask_name).strip()
        if not self.mask_name:
            raise ValueError("mask_name must be a non-empty string")
        if self.mask_split_name not in {"train", "valid"}:
            raise ValueError(f"unknown mask split: {self.mask_split_name!r}")
        self.train_segment_ids = frozenset(str(segment_id) for segment_id in self.train_segment_ids)
        self.patch_index_cache_dir = None if self.patch_index_cache_dir is None else str(self.patch_index_cache_dir)

    def mask_names_for_segment(self, segment_id: str) -> tuple[str, ...]:
        return resolve_layout_mask_names_for_segment(
            layout=self.layout,
            segment_id=segment_id,
            split_name=self.mask_split_name,
            train_segment_ids=self.train_segment_ids,
            default_mask_name=self.mask_name,
            mask_suffix=self.mask_suffix,
        )

    @classmethod
    def from_recipe_split(
        cls,
        recipe,
        *,
        split_name: str,
        layout: NestedZarrLayout | None = None,
        volume_cache: dict[Any, ZarrSegmentVolume] | None = None,
        label_mask_store_cache: dict[Any, ZarrSegmentLabelMaskStore] | None = None,
    ) -> "ZarrDataContext":
        return cls(
            layout=NestedZarrLayout(recipe.dataset_root) if layout is None else layout,
            segments=recipe.segments,
            in_channels=int(recipe.in_channels),
            label_suffix=str(recipe.label_suffix),
            mask_suffix=str(recipe.mask_suffix),
            split_name=str(split_name),
            mask_split_name=str(split_name),
            mask_name=default_mask_name_for_split(split_name),
            train_segment_ids=frozenset(str(segment_id) for segment_id in getattr(recipe, "train_segment_ids", ())),
            patch_index_cache_dir=recipe.patch_index_cache_dir,
            volume_cache={} if volume_cache is None else volume_cache,
            label_mask_store_cache={} if label_mask_store_cache is None else label_mask_store_cache,
        )

    @classmethod
    def from_dataset(cls, dataset) -> "ZarrDataContext":
        return cls(
            layout=dataset.layout,
            segments=dataset.segments,
            in_channels=int(dataset.in_channels),
            label_suffix=str(dataset.label_suffix),
            mask_suffix=str(dataset.mask_suffix),
            split_name=str(dataset.split),
            mask_split_name=str(getattr(dataset, "mask_split_name", dataset.split)),
            mask_name=str(dataset.mask_name),
            train_segment_ids=frozenset(str(segment_id) for segment_id in getattr(dataset, "train_segment_ids", ())),
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
    dataset_version: str = ""
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

    def __post_init__(self) -> None:
        dataset_version = _normalize_dataset_version(self.dataset_version)
        object.__setattr__(self, "dataset_version", dataset_version[1:] if dataset_version else "")
        if not str(self.label_suffix).strip():
            object.__setattr__(self, "label_suffix", dataset_version)
        if not str(self.mask_suffix).strip():
            object.__setattr__(self, "mask_suffix", dataset_version)

    def split_segment_ids(self) -> dict[str, tuple[str, ...]]:
        return {
            "train": tuple(str(segment_id).strip() for segment_id in self.train_segment_ids),
            "valid": tuple(str(segment_id).strip() for segment_id in self.val_segment_ids),
        }

    def build(self, *, runtime=None, augment=None) -> DataBundle:
        assert augment is not None

        patch_size = int(self.patch_size)
        tile_size = patch_size if self.tile_size is None else int(self.tile_size)
        stride = patch_size if self.stride is None else int(self.stride)
        in_channels = int(self.in_channels)
        build_workers = max(0, int(self.num_workers))
        valid_batch_size = int(self.train_batch_size if self.valid_batch_size is None else self.valid_batch_size)

        log_data_progress(f"[data] dataset_root={self.dataset_root}")
        split_segment_ids = self.split_segment_ids()
        contexts = self._build_contexts()
        samples_by_split = self._build_samples_by_split(
            contexts=contexts,
            split_segment_ids=split_segment_ids,
            patch_size=patch_size,
            tile_size=tile_size,
            stride=stride,
            build_workers=build_workers,
        )
        log_data_progress(
            f"[data] train segments={len(split_segment_ids['train'])} patches={len(samples_by_split['train'])}"
        )
        log_data_progress(
            f"[data] eval segments={len(split_segment_ids['valid'])} patches={len(samples_by_split['valid'])}"
        )

        extras = dict(self.extras or {})
        group_counts = extras.pop("group_counts", None)
        normalization_stats = extras.pop("normalization_stats", None)
        normalization = self.normalization.build(normalization_stats=normalization_stats)
        augment_recipe = augment.build(patch_size=patch_size, runtime=runtime)

        train_dataset = self._build_dataset(
            split_name="train",
            context=contexts["train"],
            samples=samples_by_split["train"],
            segment_ids=split_segment_ids["train"],
            train_segment_ids=split_segment_ids["train"],
            augment_recipe=augment_recipe,
            normalization=normalization,
            patch_size=patch_size,
            tile_size=tile_size,
            stride=stride,
            in_channels=in_channels,
            cache_patches_in_memory=bool(self.cache_train_patches_in_memory),
            include_valid_mask=bool(self.include_train_valid_mask),
        )
        valid_dataset = self._build_dataset(
            split_name="valid",
            context=contexts["valid"],
            samples=samples_by_split["valid"],
            segment_ids=split_segment_ids["valid"],
            train_segment_ids=split_segment_ids["train"],
            augment_recipe=augment_recipe,
            normalization=normalization,
            patch_size=patch_size,
            tile_size=tile_size,
            stride=stride,
            in_channels=in_channels,
            cache_patches_in_memory=False,
            include_valid_mask=True,
        )

        self._log_cache_summary(
            context=contexts["train"],
            train_dataset=train_dataset,
            build_workers=build_workers,
        )
        return build_patch_data_bundle(
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            train_batch_size=int(self.train_batch_size),
            valid_batch_size=valid_batch_size,
            num_workers=build_workers,
            shuffle=bool(self.shuffle),
            sampler=self.sampler,
            collate_fn=collate_patch_batch,
            in_channels=in_channels,
            augment_recipe=augment_recipe,
            group_counts=group_counts,
            extras=extras,
        )

    def _build_contexts(self) -> dict[str, ZarrDataContext]:
        layout = NestedZarrLayout(self.dataset_root)
        shared_volume_cache: dict[Any, ZarrSegmentVolume] = {}
        shared_label_mask_store_cache: dict[Any, ZarrSegmentLabelMaskStore] = {}
        return {
            split_name: ZarrDataContext.from_recipe_split(
                self,
                split_name=split_name,
                layout=layout,
                volume_cache=shared_volume_cache,
                label_mask_store_cache=shared_label_mask_store_cache,
            )
            for split_name in ("train", "valid")
        }

    def _build_samples_by_split(
        self,
        *,
        contexts: Mapping[str, ZarrDataContext],
        split_segment_ids: Mapping[str, tuple[str, ...]],
        patch_size: int,
        tile_size: int,
        stride: int,
        build_workers: int,
    ) -> dict[str, list[tuple[Any, ...]]]:
        all_segment_ids = tuple(
            str(segment_id)
            for segment_ids in split_segment_ids.values()
            for segment_id in segment_ids
        )
        return {
            split_name: build_zarr_split_samples(
                contexts[split_name],
                segment_ids=split_segment_ids[split_name],
                patch_size=patch_size,
                tile_size=tile_size,
                stride=stride,
                split_name=split_name,
                build_workers=build_workers,
                group_segment_ids=all_segment_ids,
            )
            for split_name in ("train", "valid")
        }

    def _build_dataset(
        self,
        *,
        split_name: str,
        context: ZarrDataContext,
        samples,
        segment_ids: tuple[str, ...],
        train_segment_ids: tuple[str, ...],
        augment_recipe,
        normalization,
        patch_size: int,
        tile_size: int,
        stride: int,
        in_channels: int,
        cache_patches_in_memory: bool,
        include_valid_mask: bool,
    ) -> "ZarrPatchDataset":
        return ZarrPatchDataset(
            samples,
            layout=context.layout,
            segments=context.segments,
            split=split_name,
            segment_ids=segment_ids,
            augment=augment_recipe,
            normalization=normalization,
            patch_size=patch_size,
            tile_size=tile_size,
            stride=stride,
            in_channels=in_channels,
            label_suffix=context.label_suffix,
            mask_suffix=context.mask_suffix,
            mask_name=context.mask_name,
            mask_split_name=context.mask_split_name,
            train_segment_ids=train_segment_ids,
            volume_cache=context.volume_cache,
            label_mask_store_cache=context.label_mask_store_cache,
            patch_index_cache_dir=context.patch_index_cache_dir,
            cache_patches_in_memory=cache_patches_in_memory,
            include_valid_mask=include_valid_mask,
        )

    def _log_cache_summary(
        self,
        *,
        context: ZarrDataContext,
        train_dataset: "ZarrPatchDataset",
        build_workers: int,
    ) -> None:
        train_patch_ram_cache = "disabled"
        if train_dataset.cache_patches_in_memory:
            train_patch_ram_cache = "preload_then_fork_share" if int(build_workers) > 0 else "lazy_single_process"
        log_data_progress(
            "[data] caches "
            f"volume_entries={len(context.volume_cache)} "
            f"label_mask_store_entries={len(context.label_mask_store_cache)} "
            "volume_cache_scope=in_process "
            "label_mask_store_cache_scope=in_process "
            f"train_patch_ram_cache={train_patch_ram_cache}"
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
        mask_name: str = SUPERVISION_MASK_NAME,
        mask_split_name: str | None = None,
        train_segment_ids=(),
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
        self.mask_name = str(mask_name).strip()
        if not self.mask_name:
            raise ValueError("mask_name must be a non-empty string")
        self.mask_split_name = str(mask_split_name).strip().lower() if mask_split_name is not None else self.split
        if self.mask_split_name not in {"train", "valid"}:
            raise ValueError(f"unknown mask split: {mask_split_name!r}")
        self.train_segment_ids = tuple(str(segment_id) for segment_id in train_segment_ids)
        self.patch_index_cache_dir = None if patch_index_cache_dir is None else str(patch_index_cache_dir)
        self.cache_patches_in_memory = bool(cache_patches_in_memory)
        if self.split not in {"train", "valid"}:
            raise ValueError(f"unknown split: {split!r}")
        self.include_valid_mask = bool(include_valid_mask) if self.split == "train" else True

        self._samples = []
        self._group_idxs = []
        for sample in samples:
            segment_id, xyxy, bbox_index = sample[:3]
            group_idx = None if len(sample) < 4 else int(sample[3])
            self._samples.append(
                (
                    str(segment_id),
                    tuple(int(value) for value in xyxy),
                    int(bbox_index),
                )
            )
            self._group_idxs.append(group_idx)
        if segment_ids:
            self.segment_ids = tuple(str(segment_id) for segment_id in segment_ids)
        else:
            self.segment_ids = tuple(dict.fromkeys(segment_id for segment_id, *_rest in self._samples))
        self._volume_cache = {} if volume_cache is None else volume_cache
        self._label_mask_store_cache = {} if label_mask_store_cache is None else label_mask_store_cache
        self._patch_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, str]] = {}
        self._data_context = ZarrDataContext.from_dataset(self)
        self.transform = build_joint_transform(
            self.split,
            augment=self.augment,
            normalization=self.normalization,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
        )

    def data_context(self) -> ZarrDataContext:
        return self._data_context

    def mask_names_for_segment(self, segment_id: str) -> tuple[str, ...]:
        return self.data_context().mask_names_for_segment(segment_id)

    @property
    def sample_rows(self) -> tuple[tuple[Any, ...], ...]:
        return tuple(
            sample if group_idx is None else sample + (int(group_idx),)
            for sample, group_idx in zip(self._samples, self._group_idxs, strict=False)
        )

    @property
    def sample_groups(self) -> list[int]:
        if any(group_idx is None for group_idx in self._group_idxs):
            raise ValueError("group_idx is unavailable for at least one sample")
        return [int(group_idx) for group_idx in self._group_idxs]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx):
        idx = int(idx)
        image, label, valid_mask = self._load_item(idx)
        segment_id, (x1, y1, x2, y2), _bbox_index = self._samples[idx]
        xyxy = np.asarray((x1, y1, x2, y2), dtype=np.int64)
        group_idx = self._group_idxs[idx]
        if group_idx is None:
            return image, label, valid_mask, xyxy, segment_id
        return image, label, valid_mask, xyxy, segment_id, int(group_idx)

    def _load_item(self, idx):
        image, label, valid_mask, _xyxy, _segment_id = self._load_raw_sample(idx)
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

    def _load_raw_sample(self, idx: int):
        idx = int(idx)
        if self.cache_patches_in_memory:
            return self._cache_raw_sample(idx)[0]
        return load_raw_zarr_patch(
            self.data_context(),
            sample=self._samples[idx],
            include_valid_mask=self.include_valid_mask,
        )

    def _cache_raw_sample(self, idx: int):
        idx = int(idx)
        cached = self._patch_cache.get(idx)
        if cached is not None:
            return cached, 0

        image, label, valid_mask, xyxy, segment_id = load_raw_zarr_patch(
            self.data_context(),
            sample=self._samples[idx],
            include_valid_mask=self.include_valid_mask,
        )
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


def build_patch_data_bundle(
    *,
    train_dataset,
    valid_dataset,
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
        valid_dataset,
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


def build_zarr_split_samples(
    context: ZarrDataContext,
    *,
    segment_ids: tuple[str, ...],
    patch_size: int,
    tile_size: int,
    stride: int,
    split_name: str = "",
    build_workers: int = 0,
    group_segment_ids: tuple[str, ...] | None = None,
) -> list[tuple[Any, ...]]:
    segment_ids = tuple(str(segment_id) for segment_id in segment_ids)
    group_idx_by_segment = None
    if group_segment_ids:
        group_idx_by_segment = _group_idx_by_segment_ids(context.layout, group_segment_ids)

    build_workers = max(0, int(build_workers))
    if build_workers > 1 and len(segment_ids) > 1:
        with ThreadPoolExecutor(max_workers=min(build_workers, len(segment_ids))) as executor:
            futures = [
                executor.submit(
                    _build_zarr_segment_samples,
                    context,
                    segment_id=str(segment_id),
                    patch_size=patch_size,
                    tile_size=tile_size,
                    stride=stride,
                    split_name=split_name,
                    group_idx_by_segment=group_idx_by_segment,
                )
                for segment_id in segment_ids
            ]
            split_samples: list[tuple[Any, ...]] = []
            for future in futures:
                split_samples.extend(future.result())
            return split_samples

    split_samples: list[tuple[Any, ...]] = []
    for segment_id in segment_ids:
        split_samples.extend(
            _build_zarr_segment_samples(
                context,
                segment_id=str(segment_id),
                patch_size=patch_size,
                tile_size=tile_size,
                stride=stride,
                split_name=split_name,
                group_idx_by_segment=group_idx_by_segment,
            )
        )
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
    mask_names = context.mask_names_for_segment(segment_id)
    label_mask_store = resolve_segment_label_mask_store(
        layout=context.layout,
        segment_id=segment_id,
        image_shape_hw=volume.image_shape_hw,
        label_suffix=context.label_suffix,
        mask_suffix=context.mask_suffix,
        mask_names=mask_names,
        label_mask_store_cache=context.label_mask_store_cache,
    )
    if include_valid_mask:
        label_mask, split_mask = label_mask_store.read_patch(
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
        split_mask = None

    image = volume.read_patch(y1, y2, x1, x2)
    label = np.asarray(label_mask, dtype=np.uint8)[..., None]
    valid_mask = None if split_mask is None else np.asarray(split_mask, dtype=np.uint8)[..., None]
    return image, label, valid_mask, np.asarray((x1, y1, x2, y2), dtype=np.int64), str(segment_id)


def collate_patch_batch(samples) -> Batch:
    if len(samples[0]) == 5:
        images, labels, valid_masks, xyxys, segment_ids = zip(*samples)
        group_idxs = None
    elif len(samples[0]) == 6:
        images, labels, valid_masks, xyxys, segment_ids, group_idxs = zip(*samples)
    else:
        raise ValueError("patch batch samples must be (x, y, valid_mask, xyxy, segment_id[, group_idx])")

    valid_mask_batch = None
    if len(valid_masks) > 0 and valid_masks[0] is not None:
        valid_mask_batch = _stack_batch_tensors(valid_masks)
    return Batch(
        x=_stack_batch_tensors(images),
        y=_stack_batch_tensors(labels),
        meta=BatchMeta(
            segment_ids=[str(segment_id) for segment_id in segment_ids],
            valid_mask=valid_mask_batch,
            patch_xyxy=torch.as_tensor(np.asarray(xyxys), dtype=torch.long),
            group_idx=None if group_idxs is None else torch.as_tensor(group_idxs, dtype=torch.long),
        ),
    )


def collate_infer_batch(samples) -> Batch | None:
    filtered_samples = [sample for sample in samples if sample is not None]
    if not filtered_samples:
        return None
    images, xyxys, segment_ids = zip(*filtered_samples)
    return Batch(
        x=_stack_batch_tensors(images),
        y=None,
        meta=BatchMeta(
            segment_ids=[str(segment_id) for segment_id in segment_ids],
            valid_mask=None,
            patch_xyxy=torch.as_tensor(np.asarray(xyxys), dtype=torch.long),
            group_idx=None,
        ),
    )


def _normalize_dataset_version(dataset_version: str) -> str:
    version = str(dataset_version).strip()
    if not version:
        return ""
    version = version.lstrip("_")
    if not version:
        return ""
    return f"_{version}"


def _prepare_train_loader_multiprocessing_context(dataset, *, num_workers: int):
    if int(num_workers) <= 0:
        return None
    prepare = getattr(dataset, "_prepare_worker_processes", None)
    if not callable(prepare):
        return None
    return prepare(num_workers=int(num_workers))


def _build_zarr_segment_samples(
    context: ZarrDataContext,
    *,
    segment_id: str,
    patch_size: int,
    tile_size: int,
    stride: int,
    split_name: str,
    group_idx_by_segment: Mapping[str, int] | None,
) -> list[tuple[Any, ...]]:
    log_data_progress(f"[data] {split_name} loading segment={segment_id}")
    mask_names = context.mask_names_for_segment(segment_id)
    resolved_paths, mask_artifacts = _resolve_segment_patch_artifacts(
        context,
        segment_id=segment_id,
        mask_names=mask_names,
    )
    volume = resolve_segment_volume(
        layout=context.layout,
        segments=context.segments,
        segment_id=segment_id,
        in_channels=context.in_channels,
        volume_cache=context.volume_cache,
    )
    bbox_rows, xyxys, sample_bbox_indices = _load_or_build_segment_patch_index(
        context,
        segment_id=segment_id,
        split_name=split_name,
        patch_size=patch_size,
        tile_size=tile_size,
        stride=stride,
        volume_image_shape_hw=volume.image_shape_hw,
        mask_names=mask_names,
        label_artifact=str(resolved_paths.inklabels_path.name),
        mask_artifacts=mask_artifacts,
    )
    resolve_segment_label_mask_store(
        layout=context.layout,
        segment_id=segment_id,
        image_shape_hw=volume.image_shape_hw,
        label_suffix=context.label_suffix,
        mask_suffix=context.mask_suffix,
        mask_names=mask_names,
        label_mask_store_cache=context.label_mask_store_cache,
        bbox_rows=bbox_rows,
    )
    segment_group_idx = None if group_idx_by_segment is None else int(group_idx_by_segment[segment_id])
    patch_rows = _build_segment_patch_rows(
        segment_id=segment_id,
        xyxys=xyxys,
        sample_bbox_indices=sample_bbox_indices,
        group_idx=segment_group_idx,
    )
    log_data_progress(f"[data] {split_name} segment={segment_id} patches={len(patch_rows)}")
    return patch_rows


def _resolve_segment_patch_artifacts(
    context: ZarrDataContext,
    *,
    segment_id: str,
    mask_names: tuple[str, ...],
):
    resolved_paths = context.layout.resolve_paths(
        segment_id,
        label_suffix=context.label_suffix,
        mask_suffix=context.mask_suffix,
        mask_name=mask_names[0],
    )
    mask_artifacts = tuple(
        str(
            context.layout.resolve_paths(
                segment_id,
                label_suffix=context.label_suffix,
                mask_suffix=context.mask_suffix,
                mask_name=current_mask_name,
            ).mask_path.name
        )
        for current_mask_name in mask_names
    )
    return resolved_paths, mask_artifacts


def _load_or_build_segment_patch_index(
    context: ZarrDataContext,
    *,
    segment_id: str,
    split_name: str,
    patch_size: int,
    tile_size: int,
    stride: int,
    volume_image_shape_hw,
    mask_names: tuple[str, ...],
    label_artifact: str,
    mask_artifacts: tuple[str, ...],
):
    cached_index = load_cached_patch_index(
        cache_dir=context.patch_index_cache_dir,
        segment_id=segment_id,
        split_name=split_name,
        patch_size=patch_size,
        tile_size=tile_size,
        stride=stride,
        label_suffix=context.label_suffix,
        mask_suffix=context.mask_suffix,
        mask_names=mask_names,
        label_artifact=str(label_artifact),
        mask_artifacts=mask_artifacts,
        log=log_data_progress,
    )
    if cached_index is not None:
        return (
            np.asarray(cached_index.bbox_rows, dtype=np.int32),
            np.asarray(cached_index.xyxys, dtype=np.int64),
            np.asarray(cached_index.sample_bbox_indices, dtype=np.int32),
        )

    split_mask = read_supervision_mask_for_shape(
        context.layout,
        segment_id,
        volume_image_shape_hw,
        mask_suffix=context.mask_suffix,
        mask_names=mask_names,
    )
    bbox_rows, xyxys, sample_bbox_indices = build_patch_index(
        None,
        split_mask,
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
        mask_names=mask_names,
        label_artifact=str(label_artifact),
        mask_artifacts=mask_artifacts,
        xyxys=xyxys,
        bbox_rows=bbox_rows,
        sample_bbox_indices=sample_bbox_indices,
    )
    return bbox_rows, xyxys, sample_bbox_indices


def _build_segment_patch_rows(
    *,
    segment_id: str,
    xyxys,
    sample_bbox_indices,
    group_idx: int | None,
) -> list[tuple[Any, ...]]:
    xyxys = np.asarray(xyxys, dtype=np.int64)
    sample_bbox_indices = np.asarray(sample_bbox_indices, dtype=np.int32)
    patch_rows: list[tuple[Any, ...]] = []
    for row_idx, xyxy in enumerate(xyxys):
        patch_row: tuple[Any, ...] = (
            segment_id,
            tuple(int(value) for value in xyxy),
            int(sample_bbox_indices[row_idx]),
        )
        if group_idx is not None:
            patch_row = patch_row + (int(group_idx),)
        patch_rows.append(patch_row)
    return patch_rows


def _group_idx_by_segment_ids(layout, segment_ids) -> dict[str, int]:
    segment_ids = tuple(dict.fromkeys(str(segment_id) for segment_id in segment_ids))
    group_name_to_idx = {
        group_name: idx
        for idx, group_name in enumerate(
            sorted({layout.resolve_group_name(segment_id) for segment_id in segment_ids})
        )
    }
    return {
        segment_id: int(group_name_to_idx[layout.resolve_group_name(segment_id)])
        for segment_id in segment_ids
    }


def _cache_ready_array(array) -> np.ndarray | None:
    if array is None:
        return None
    return np.ascontiguousarray(np.asarray(array))


def _cached_patch_nbytes(image, label, valid_mask, xyxy) -> int:
    return sum(int(np.asarray(part).nbytes) for part in (image, label, valid_mask, xyxy) if part is not None)


def _stack_batch_tensors(values):
    first = values[0]
    if isinstance(first, torch.Tensor):
        return torch.stack(list(values), dim=0)
    return torch.stack([torch.as_tensor(value) for value in values], dim=0)


__all__ = [
    "ZarrDataContext",
    "ZarrPatchDataRecipe",
    "ZarrPatchDataset",
    "build_patch_data_bundle",
    "build_zarr_split_samples",
    "collate_patch_batch",
    "load_raw_zarr_patch",
    "log_data_progress",
]
