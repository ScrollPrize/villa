from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import logging
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ink.core.types import Batch, BatchMeta, DataBundle
from ink.recipes.data.layout import NestedZarrLayout
from ink.recipes.data.normalization import ClipMaxDiv255Normalization
from ink.recipes.data.patch_index_cache import load_cached_patch_xyxys, save_cached_patch_xyxys
from ink.recipes.data.patching import extract_patch_coordinates
from ink.recipes.data.samplers import ShuffleSampler
from ink.recipes.data.transforms import (
    apply_eval_sample_transforms,
    apply_train_sample_transforms,
    build_joint_transform,
)
from ink.recipes.data.zarr_io import (
    ZarrSegmentVolume,
    parse_layer_range_value,
    read_label_and_supervision_mask_for_shape,
)

_PROGRESS_LOG = logging.getLogger("ink.progress")


def _log(message: str) -> None:
    _PROGRESS_LOG.info(str(message))


def _read_mask_patch(mask, *, y1: int, y2: int, x1: int, x2: int) -> np.ndarray:
    """Slice a mask patch and zero-pad any area that falls outside the source image."""
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


def _batch_from_parts(
    *,
    images,
    labels,
    valid_masks,
    xyxys,
    segment_ids,
    group_idxs=None,
) -> Batch:
    meta = BatchMeta(
        segment_ids=[str(segment_id) for segment_id in segment_ids],
        valid_mask=torch.stack([torch.as_tensor(mask) for mask in valid_masks], dim=0),
        patch_xyxy=torch.as_tensor(np.asarray(xyxys), dtype=torch.long),
        group_idx=None if group_idxs is None else torch.as_tensor(group_idxs, dtype=torch.long),
    )

    return Batch(
        x=torch.stack([torch.as_tensor(image) for image in images], dim=0),
        y=torch.stack([torch.as_tensor(label) for label in labels], dim=0),
        meta=meta,
    )


def _collate_batch(samples) -> Batch:
    images, labels, valid_masks, xyxys, segment_ids = zip(*samples)
    return _batch_from_parts(
        images=images,
        labels=labels,
        valid_masks=valid_masks,
        xyxys=xyxys,
        segment_ids=segment_ids,
    )


def _resolve_segment_volume(
    *,
    layout: NestedZarrLayout,
    segments,
    segment_id: str,
    in_channels: int,
    volume_cache: dict[Any, ZarrSegmentVolume],
) -> tuple[tuple[int, int], bool, ZarrSegmentVolume]:
    """Reuse segment volumes across samples so zarr handles and metadata stay cached."""
    segment_spec = segments[segment_id]
    raw_layer_range = segment_spec.get("layer_range")
    reverse_layers = bool(segment_spec.get("reverse_layers", False))
    layer_range = None
    if raw_layer_range is not None:
        layer_range = parse_layer_range_value(
            raw_layer_range,
            context=f"segments[{segment_id!r}].layer_range",
        )

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
    return volume.layer_range, reverse_layers, volume


def _build_samples_from_segments(
    *,
    layout: NestedZarrLayout,
    segments,
    segment_ids: tuple[str, ...],
    in_channels: int,
    patch_size: int,
    tile_size: int,
    stride: int,
    label_suffix: str,
    mask_suffix: str,
    volume_cache: dict[Any, ZarrSegmentVolume],
    mask_cache: dict[Any, tuple[np.ndarray, np.ndarray]],
    split_name: str = "",
    build_workers: int = 0,
    patch_index_cache_dir: str | None = None,
) -> list[tuple[str, tuple[int, int], bool, tuple[int, int, int, int]]]:
    """Expand segment configs into patch-level samples for a dataset split."""
    split_samples = []

    def _build_one(segment_id: str):
        _log(f"[data] {split_name} loading segment={segment_id}")
        layer_range, reverse_layers, volume = _resolve_segment_volume(
            layout=layout,
            segments=segments,
            segment_id=segment_id,
            in_channels=in_channels,
            volume_cache={},
        )
        masks = read_label_and_supervision_mask_for_shape(
            layout,
            segment_id,
            volume.image_shape_hw,
            label_suffix=label_suffix,
            mask_suffix=mask_suffix,
        )
        label_mask, supervision_mask = masks
        xyxys = load_cached_patch_xyxys(
            cache_dir=patch_index_cache_dir,
            layout=layout,
            segment_id=segment_id,
            split_name=split_name,
            patch_size=patch_size,
            tile_size=tile_size,
            stride=stride,
            label_suffix=label_suffix,
            mask_suffix=mask_suffix,
            log=_log,
        )
        if xyxys is None:
            xyxys = np.asarray(
                extract_patch_coordinates(
                    label_mask,
                    supervision_mask,
                    size=patch_size,
                    tile_size=tile_size,
                    stride=stride,
                    filter_empty_tile=False,
                ),
                dtype=np.int64,
            )
            save_cached_patch_xyxys(
                cache_dir=patch_index_cache_dir,
                layout=layout,
                segment_id=segment_id,
                split_name=split_name,
                patch_size=patch_size,
                tile_size=tile_size,
                stride=stride,
                label_suffix=label_suffix,
                mask_suffix=mask_suffix,
                xyxys=xyxys,
            )

        patch_rows = [
            (
                segment_id,
                layer_range,
                reverse_layers,
                tuple(int(value) for value in xyxy),
            )
            for xyxy in np.asarray(xyxys, dtype=np.int64)
        ]
        _log(f"[data] {split_name} segment={segment_id} patches={len(patch_rows)}")
        return str(segment_id), volume, masks, patch_rows

    def _cache_result(
        segment_id: str,
        volume: ZarrSegmentVolume,
        masks: tuple[np.ndarray, np.ndarray],
        patch_rows: list[tuple[str, tuple[int, int], bool, tuple[int, int, int, int]]],
    ) -> None:
        volume_key = (str(segment_id), volume.layer_range, bool(volume.reverse_layers), int(in_channels))
        mask_key = (str(segment_id), label_suffix, mask_suffix)
        volume_cache.setdefault(volume_key, volume)
        mask_cache.setdefault(mask_key, masks)
        split_samples.extend(patch_rows)

    build_workers = max(0, int(build_workers))
    if build_workers > 1 and len(segment_ids) > 1:
        with ThreadPoolExecutor(max_workers=min(build_workers, len(segment_ids))) as executor:
            for segment_id, volume, masks, patch_rows in executor.map(_build_one, segment_ids):
                _cache_result(segment_id, volume, masks, patch_rows)
        return split_samples

    for segment_id in segment_ids:
        segment_id, volume, masks, patch_rows = _build_one(segment_id)
        _cache_result(segment_id, volume, masks, patch_rows)
    return split_samples


def _load_raw_patch_sample(
    *,
    layout: NestedZarrLayout,
    segments,
    sample,
    in_channels: int,
    label_suffix: str,
    mask_suffix: str,
    volume_cache: dict[Any, ZarrSegmentVolume],
    mask_cache: dict[Any, tuple[np.ndarray, np.ndarray]],
):
    segment_id, layer_range, reverse_layers, xyxy = sample[:4]
    x1, y1, x2, y2 = [int(value) for value in xyxy]
    volume = _resolve_segment_volume(
        layout=layout,
        segments=segments,
        segment_id=segment_id,
        in_channels=in_channels,
        volume_cache=volume_cache,
    )[2]

    cache_key = (str(segment_id), str(label_suffix), str(mask_suffix))
    masks = mask_cache.get(cache_key)
    if masks is None:
        masks = read_label_and_supervision_mask_for_shape(
            layout,
            segment_id,
            volume.image_shape_hw,
            label_suffix=label_suffix,
            mask_suffix=mask_suffix,
        )
        mask_cache[cache_key] = masks
    label_mask, supervision_mask = masks
    image = volume.read_patch(y1, y2, x1, x2)
    label = _read_mask_patch(label_mask, y1=y1, y2=y2, x1=x1, x2=x2)[..., None]
    valid_mask = _read_mask_patch(supervision_mask, y1=y1, y2=y2, x1=x1, x2=x2)[..., None]
    return image, label, valid_mask, np.asarray((x1, y1, x2, y2), dtype=np.int64), str(segment_id)


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
        mask_cache: dict[Any, tuple[np.ndarray, np.ndarray]] | None = None,
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
        if self.split not in {"train", "valid"}:
            raise ValueError(f"unknown split: {split!r}")

        self._samples = [
            (
                str(segment_id),
                tuple(int(value) for value in layer_range),
                bool(reverse_layers),
                tuple(int(value) for value in xyxy),
            )
            for segment_id, layer_range, reverse_layers, xyxy in samples
        ]
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

    def _load_item(self, idx):
        """Load one raw patch and run the split-specific transform pipeline."""
        idx = int(idx)
        image, label, valid_mask, _xyxy, _segment_id = _load_raw_patch_sample(
            layout=self.layout,
            segments=self.segments,
            sample=self._samples[idx],
            in_channels=self.in_channels,
            label_suffix=self.label_suffix,
            mask_suffix=self.mask_suffix,
            volume_cache=self._volume_cache,
            mask_cache=self._mask_cache,
        )

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
        segment_id, _layer_range, _reverse_layers, (x1, y1, x2, y2) = self._samples[idx]
        xyxy = np.asarray((x1, y1, x2, y2), dtype=np.int64)
        return image, label, valid_mask, xyxy, segment_id


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
    extras: dict[str, Any] = field(default_factory=dict)

    def _split_segment_ids(self) -> dict[str, tuple[str, ...]]:
        return {
            "train": tuple(str(segment_id).strip() for segment_id in self.train_segment_ids),
            "valid": tuple(str(segment_id).strip() for segment_id in self.val_segment_ids),
        }

    def _build_layout(self) -> NestedZarrLayout:
        return NestedZarrLayout(self.dataset_root)

    def _build_samples_by_split(
        self,
        *,
        layout: NestedZarrLayout,
        segments,
        in_channels: int,
        patch_size: int,
        tile_size: int,
        stride: int,
        volume_cache: dict[Any, ZarrSegmentVolume],
        mask_cache: dict[Any, tuple[np.ndarray, np.ndarray]],
    ) -> dict[str, list]:
        return {
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
                split_name=split,
                build_workers=max(0, int(self.num_workers)),
                patch_index_cache_dir=self.patch_index_cache_dir,
            )
            for split, segment_ids in self._split_segment_ids().items()
        }

    def _build_bundle_extras(
        self,
    ) -> dict[str, Any]:
        return dict(self.extras or {})

    def _build_group_counts(
        self,
        *,
        samples_by_split: dict[str, list],
    ) -> list[int] | None:
        del samples_by_split
        return None

    def _dataset_class(self):
        return ZarrPatchDataset

    def _collate_fn(self):
        return _collate_batch

    def build(self, *, runtime=None, augment=None) -> DataBundle:
        """Build loaders, caches, and bundle extras from the configured zarr dataset."""
        assert augment is not None
        augment_recipe = augment
        segments = self.segments
        patch_size = int(self.patch_size)
        in_channels = int(self.in_channels)
        tile_size = patch_size if self.tile_size is None else int(self.tile_size)
        stride = patch_size if self.stride is None else int(self.stride)
        layout = self._build_layout()
        _log(f"[data] dataset_root={self.dataset_root}")
        volume_cache: dict[Any, ZarrSegmentVolume] = {}
        mask_cache: dict[Any, tuple[np.ndarray, np.ndarray]] = {}
        samples_by_split = self._build_samples_by_split(
            layout=layout,
            segments=segments,
            in_channels=in_channels,
            patch_size=patch_size,
            tile_size=tile_size,
            stride=stride,
            volume_cache=volume_cache,
            mask_cache=mask_cache,
        )
        train_segment_count = len(self.train_segment_ids)
        eval_segment_count = len(self.val_segment_ids)
        _log(f"[data] train segments={train_segment_count} patches={len(samples_by_split['train'])}")
        _log(f"[data] eval segments={eval_segment_count} patches={len(samples_by_split['valid'])}")

        valid_batch_size = self.train_batch_size if self.valid_batch_size is None else self.valid_batch_size
        extras = self._build_bundle_extras()
        group_counts = self._build_group_counts(samples_by_split=samples_by_split)
        if group_counts is None:
            group_counts = extras.pop("group_counts", None)
        normalization_stats = extras.pop("normalization_stats", None)

        normalization = self.normalization.build(
            normalization_stats=normalization_stats,
        )
        augment_recipe = augment_recipe.build(
            patch_size=patch_size,
            runtime=runtime,
        )
        bundle_extras = dict(extras)

        dataset_kwargs = {
            "layout": layout,
            "segments": segments,
            "augment": augment_recipe,
            "normalization": normalization,
            "patch_size": patch_size,
            "tile_size": tile_size,
            "stride": stride,
            "in_channels": in_channels,
            "label_suffix": self.label_suffix,
            "mask_suffix": self.mask_suffix,
            "volume_cache": volume_cache,
            "mask_cache": mask_cache,
        }
        dataset_class = self._dataset_class()
        collate_fn = self._collate_fn()
        split_segment_ids = self._split_segment_ids()
        train_dataset = dataset_class(
            samples_by_split["train"],
            split="train",
            segment_ids=split_segment_ids["train"],
            **dataset_kwargs,
        )
        val_dataset = dataset_class(
            samples_by_split["valid"],
            split="valid",
            segment_ids=split_segment_ids["valid"],
            **dataset_kwargs,
        )
        num_workers = max(0, int(self.num_workers))
        train_batch_size = int(self.train_batch_size)
        valid_batch_size = int(valid_batch_size)
        train_loader = self.sampler.build_loader(
            train_dataset,
            batch_size=train_batch_size,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            shuffle=bool(self.shuffle),
        )
        eval_loader = DataLoader(
            val_dataset,
            batch_size=valid_batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=bool(num_workers > 0),
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )
        _log(
            "[data] caches "
            f"volume_entries={len(volume_cache)} "
            f"mask_entries={len(mask_cache)} "
            f"shared_volume_cache=True "
            f"shared_mask_cache=True"
        )
        return DataBundle(
            train_loader=train_loader,
            eval_loader=eval_loader,
            in_channels=in_channels,
            augment=augment_recipe,
            group_counts=group_counts,
            extras=bundle_extras,
        )


__all__ = ["ZarrPatchDataRecipe"]
