from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from torch.utils.data import DataLoader, Dataset

from ink.core.types import DataBundle
from ink.recipes.components import component_bboxes
from ink.recipes.data.transforms import apply_eval_sample_transforms, build_joint_transform
from ink.recipes.data.zarr_data import (
    ZarrPatchDataset,
    _build_samples_from_segments,
    _collate_batch,
    _resolve_segment_volume,
)
from ink.recipes.data.zarr_io import (
    read_label_and_supervision_mask_for_shape,
    read_optional_supervision_mask_for_shape,
)
from ink.recipes.stitch.data import StitchData, StitchSegmentSpec


# Derive stitch segment specs from zarr-backed segment metadata and masks.

def _downsample_bool_mask_any(mask: np.ndarray, *, downsample: int) -> np.ndarray:
    ds = max(1, int(downsample))
    mask_bool = np.asarray(mask) > 0
    h = int(mask_bool.shape[0])
    w = int(mask_bool.shape[1])
    ds_h = (h + ds - 1) // ds
    ds_w = (w + ds - 1) // ds
    pad_h = int(ds_h * ds - h)
    pad_w = int(ds_w * ds - w)
    if pad_h or pad_w:
        mask_bool = np.pad(mask_bool, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=False)
    mask_bool = mask_bool.reshape(ds_h, ds, ds_w, ds)
    return mask_bool.any(axis=(1, 3))


def _mask_component_bbox_rows(mask: np.ndarray, *, downsample: int) -> tuple[tuple[int, int, int, int], ...] | None:
    mask_ds = _downsample_bool_mask_any(mask, downsample=int(downsample))
    if not bool(mask_ds.any()):
        return None
    return _bbox_rows_or_none(component_bboxes(mask_ds, connectivity=2))


def _bbox_rows_or_none(bboxes) -> tuple[tuple[int, int, int, int], ...] | None:
    bboxes_arr = np.asarray(bboxes, dtype=np.int32)
    if bboxes_arr.ndim != 2 or int(bboxes_arr.shape[1]) != 4 or int(bboxes_arr.shape[0]) <= 0:
        return None
    return tuple(tuple(int(value) for value in row) for row in bboxes_arr.tolist())


def _cfg_has_key(stitch_cfg, key: str) -> bool:
    if isinstance(stitch_cfg, StitchData):
        return True
    return isinstance(stitch_cfg, Mapping) and key in stitch_cfg


def _cfg_has_segment_specs(stitch_cfg, split_name: str) -> bool:
    if isinstance(stitch_cfg, StitchData):
        return bool(getattr(stitch_cfg, split_name).segments)
    if not isinstance(stitch_cfg, Mapping):
        return False
    split_cfg = stitch_cfg.get(split_name)
    return isinstance(split_cfg, Mapping) and "segments" in split_cfg


def _ordered_segment_ids(dataset: ZarrPatchDataset) -> tuple[str, ...]:
    segment_ids = tuple(str(segment_id) for segment_id in getattr(dataset, "segment_ids", ()) or ())
    if segment_ids:
        return segment_ids
    seen = set()
    ordered = []
    for sample in getattr(dataset, "_samples", ()):
        segment_id = str(sample[0])
        if segment_id in seen:
            continue
        seen.add(segment_id)
        ordered.append(segment_id)
    return tuple(ordered)


def _segment_spec_ids(segment_specs) -> tuple[str, ...]:
    return tuple(str(spec.segment_id) for spec in (segment_specs or ()))


def _optional_supervision_mask(dataset: ZarrPatchDataset, segment_id: str, image_shape_hw) -> np.ndarray | None:
    return read_optional_supervision_mask_for_shape(
        dataset.layout,
        str(segment_id),
        image_shape_hw,
        mask_suffix=dataset.mask_suffix,
    )


def _stitch_segment_bbox_rows(
    dataset: ZarrPatchDataset,
    *,
    segment_id: str,
    image_shape_hw: tuple[int, int],
    downsample: int,
    require_supervision_mask: bool,
    fallback_to_full_segment: bool,
) -> tuple[tuple[int, int, int, int], ...] | None:
    supervision_mask = None
    cache_key = (str(segment_id), dataset.label_suffix, dataset.mask_suffix)
    cached = dataset._mask_cache.get(cache_key)
    if cached is not None:
        supervision_mask = cached[1]

    if supervision_mask is None and require_supervision_mask:
        masks = read_label_and_supervision_mask_for_shape(
            dataset.layout,
            str(segment_id),
            image_shape_hw,
            label_suffix=dataset.label_suffix,
            mask_suffix=dataset.mask_suffix,
        )
        dataset._mask_cache[cache_key] = masks
        supervision_mask = masks[1]

    if supervision_mask is None:
        supervision_mask = _optional_supervision_mask(dataset, str(segment_id), image_shape_hw)

    if supervision_mask is None:
        if fallback_to_full_segment:
            return None
        raise FileNotFoundError(f"{segment_id}: stitch requires supervision_mask-derived ROI geometry")

    bbox_rows = _mask_component_bbox_rows(supervision_mask, downsample=int(downsample))
    if bbox_rows is not None:
        return bbox_rows
    if fallback_to_full_segment:
        return None
    raise ValueError(f"{segment_id}: stitch requires at least one supervision-mask ROI component")


def _derive_segment_specs(
    dataset: ZarrPatchDataset,
    *,
    segment_ids,
    downsample: int,
    require_supervision_mask: bool,
    fallback_to_full_segment: bool,
    use_roi: bool,
) -> list[StitchSegmentSpec]:
    segment_specs = []
    for segment_id in segment_ids:
        volume = _resolve_segment_volume(
            layout=dataset.layout,
            segments=dataset.segments,
            segment_id=str(segment_id),
            in_channels=int(dataset.in_channels),
            volume_cache=dataset._volume_cache,
        )[2]
        bbox_rows = None
        if bool(use_roi):
            bbox_rows = _stitch_segment_bbox_rows(
                dataset,
                segment_id=str(segment_id),
                image_shape_hw=volume.image_shape_hw,
                downsample=int(downsample),
                require_supervision_mask=bool(require_supervision_mask),
                fallback_to_full_segment=bool(fallback_to_full_segment),
            )
        segment_specs.append(
            StitchSegmentSpec(
                segment_id=str(segment_id),
                shape=tuple(int(value) for value in volume.image_shape_hw),
                bbox=bbox_rows,
            )
        )
    return segment_specs


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
                tuple(int(value) for value in layer_range),
                bool(reverse_layers),
                tuple(int(value) for value in xyxy),
            )
            for segment_id, layer_range, reverse_layers, xyxy in samples
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
        segment_id, layer_range, reverse_layers, (x1, y1, x2, y2) = self._samples[idx]
        volume = _resolve_segment_volume(
            layout=self.layout,
            segments=self.segments,
            segment_id=segment_id,
            in_channels=int(self.in_channels),
            volume_cache=self._volume_cache,
        )[2]
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


# Build per-segment stitch loaders without changing the main patch-data path.

def _build_infer_samples_from_segments(
    dataset: ZarrPatchDataset,
    *,
    segment_ids: tuple[str, ...],
) -> list[tuple[str, tuple[int, int], bool, tuple[int, int, int, int]]]:
    split_samples = []
    for segment_id in segment_ids:
        layer_range, reverse_layers, volume = _resolve_segment_volume(
            layout=dataset.layout,
            segments=dataset.segments,
            segment_id=segment_id,
            in_channels=int(dataset.in_channels),
            volume_cache=dataset._volume_cache,
        )
        supervision_mask = _optional_supervision_mask(dataset, segment_id, volume.image_shape_hw)
        if supervision_mask is None or not bool(np.asarray(supervision_mask).any()):
            supervision_mask = np.full(volume.image_shape_hw, 255, dtype=np.uint8)
        dummy_label = np.zeros(volume.image_shape_hw, dtype=np.uint8)
        split_samples.extend(
            (
                segment_id,
                layer_range,
                reverse_layers,
                tuple(int(value) for value in xyxy),
            )
            for xyxy in _build_samples_from_segments(
                layout=dataset.layout,
                segments=dataset.segments,
                segment_ids=(segment_id,),
                in_channels=int(dataset.in_channels),
                patch_size=int(dataset.patch_size),
                tile_size=int(dataset.tile_size),
                stride=int(dataset.stride),
                label_suffix=dataset.label_suffix,
                mask_suffix=dataset.mask_suffix,
                volume_cache=dataset._volume_cache,
                mask_cache={(segment_id, dataset.label_suffix, dataset.mask_suffix): (dummy_label, supervision_mask)},
            )
        )
    return split_samples


def _samples_by_segment(samples, *, segment_ids: tuple[str, ...]) -> dict[str, list]:
    grouped = {str(segment_id): [] for segment_id in segment_ids}
    for sample in samples:
        grouped[str(sample[0])].append(sample)
    return grouped


def _build_segment_loaders(
    *,
    dataset_cls,
    dataset: ZarrPatchDataset,
    segment_ids: tuple[str, ...],
    samples_by_segment: Mapping[str, list],
    batch_size: int,
    num_workers: int,
    include_tile_config: bool,
) -> list[DataLoader]:
    common_kwargs = {
        "layout": dataset.layout,
        "segments": dataset.segments,
        "augment": dataset.augment,
        "normalization": dataset.normalization,
        "patch_size": int(dataset.patch_size),
        "in_channels": int(dataset.in_channels),
        "volume_cache": dataset._volume_cache,
    }
    if include_tile_config:
        common_kwargs.update(
            {
                "split": "valid",
                "tile_size": int(dataset.tile_size),
                "stride": int(dataset.stride),
                "label_suffix": dataset.label_suffix,
                "mask_suffix": dataset.mask_suffix,
                "mask_cache": dataset._mask_cache,
            }
        )

    return [
        DataLoader(
            dataset_cls(
                samples_by_segment.get(str(segment_id), ()),
                segment_ids=(str(segment_id),),
                **common_kwargs,
            ),
            batch_size=max(1, int(batch_size)),
            shuffle=False,
            num_workers=max(0, int(num_workers)),
            pin_memory=True,
            drop_last=False,
            collate_fn=_collate_batch,
        )
        for segment_id in segment_ids
    ]


def _resolve_loader_build_settings(*, train_loader, eval_loader) -> tuple[int, int]:
    eval_batch_size = getattr(eval_loader, "batch_size", None)
    train_batch_size = getattr(train_loader, "batch_size", None)
    batch_size = int(eval_batch_size if eval_batch_size is not None else train_batch_size or 1)

    eval_num_workers = getattr(eval_loader, "num_workers", None)
    train_num_workers = getattr(train_loader, "num_workers", None)
    num_workers = int(eval_num_workers if eval_num_workers is not None else train_num_workers or 0)
    return batch_size, num_workers


def build_zarr_segment_eval_loaders(
    dataset,
    *,
    segment_ids,
    batch_size: int,
    num_workers: int = 0,
) -> list[DataLoader]:
    if not isinstance(dataset, ZarrPatchDataset):
        return []
    requested_segment_ids = tuple(str(segment_id) for segment_id in segment_ids)
    split_samples = _build_samples_from_segments(
        layout=dataset.layout,
        segments=dataset.segments,
        segment_ids=requested_segment_ids,
        in_channels=int(dataset.in_channels),
        patch_size=int(dataset.patch_size),
        tile_size=int(dataset.tile_size),
        stride=int(dataset.stride),
        label_suffix=dataset.label_suffix,
        mask_suffix=dataset.mask_suffix,
        volume_cache=dataset._volume_cache,
        mask_cache=dataset._mask_cache,
    )
    return _build_segment_loaders(
        dataset_cls=ZarrPatchDataset,
        dataset=dataset,
        segment_ids=requested_segment_ids,
        samples_by_segment=_samples_by_segment(split_samples, segment_ids=requested_segment_ids),
        batch_size=batch_size,
        num_workers=num_workers,
        include_tile_config=True,
    )


def build_zarr_segment_infer_loaders(
    dataset,
    *,
    segment_ids,
    batch_size: int,
    num_workers: int = 0,
) -> list[DataLoader]:
    if not isinstance(dataset, ZarrPatchDataset):
        return []
    requested_segment_ids = tuple(str(segment_id) for segment_id in segment_ids)
    split_samples = _build_infer_samples_from_segments(dataset, segment_ids=requested_segment_ids)
    return _build_segment_loaders(
        dataset_cls=ZarrInferDataset,
        dataset=dataset,
        segment_ids=requested_segment_ids,
        samples_by_segment=_samples_by_segment(split_samples, segment_ids=requested_segment_ids),
        batch_size=batch_size,
        num_workers=num_workers,
        include_tile_config=False,
    )


def configure_zarr_stitch_training_loaders(stitch_runtime, *, train_loader, eval_loader) -> None:
    stitch_train = getattr(stitch_runtime, "train", None)
    stitch_data = getattr(stitch_runtime, "data", None)
    train_dataset = getattr(train_loader, "dataset", None)
    if stitch_train is None or stitch_data is None or not isinstance(train_dataset, ZarrPatchDataset):
        return

    batch_size, num_workers = _resolve_loader_build_settings(train_loader=train_loader, eval_loader=eval_loader)

    train_segment_ids = _segment_spec_ids(getattr(getattr(stitch_data, "train", None), "segments", ()))
    train_viz = getattr(getattr(stitch_data, "train", None), "viz", None)
    if bool(getattr(train_viz, "enabled", False)) and train_segment_ids:
        loaders = build_zarr_segment_eval_loaders(
            train_dataset,
            segment_ids=train_segment_ids,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        if loaders and callable(getattr(stitch_train, "set_loaders", None)):
            stitch_train.set_loaders(loaders)

    log_only_segment_ids = tuple(str(segment_id) for segment_id in getattr(getattr(stitch_data, "log_only", None), "segment_ids", ()) or ())
    if log_only_segment_ids:
        loaders = build_zarr_segment_infer_loaders(
            train_dataset,
            segment_ids=log_only_segment_ids,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        if loaders and callable(getattr(stitch_train, "set_log_only_loaders", None)):
            stitch_train.set_log_only_loaders(loaders)


# Resolve stitch runtime config from the explicit recipe, then derive missing segment specs.

def prepare_stitch_data_for_bundle(bundle: DataBundle, *, config=None) -> StitchData:
    stitch_cfg = {} if config is None else config
    stitch_data = StitchData.from_config(stitch_cfg)

    train_dataset = getattr(getattr(bundle, "train_loader", None), "dataset", None)
    eval_dataset = getattr(getattr(bundle, "eval_loader", None), "dataset", None)
    if not isinstance(train_dataset, ZarrPatchDataset) or not isinstance(eval_dataset, ZarrPatchDataset):
        return stitch_data

    if not _cfg_has_key(stitch_cfg, "use_roi"):
        stitch_data.layout.use_roi = True

    if not _cfg_has_segment_specs(stitch_cfg, "train"):
        train_segment_ids = tuple(str(segment_id) for segment_id in (stitch_data.train.segment_ids or ()))
        if not train_segment_ids:
            train_segment_ids = _ordered_segment_ids(train_dataset)
        stitch_data.train.segments = _derive_segment_specs(
            train_dataset,
            segment_ids=train_segment_ids,
            downsample=int(stitch_data.layout.downsample),
            require_supervision_mask=bool(stitch_data.layout.use_roi),
            fallback_to_full_segment=not bool(stitch_data.layout.use_roi),
            use_roi=bool(stitch_data.layout.use_roi),
        )

    if not _cfg_has_segment_specs(stitch_cfg, "eval"):
        eval_segment_ids = tuple(str(segment_id) for segment_id in (stitch_data.eval.segment_ids or ()))
        if not eval_segment_ids:
            eval_segment_ids = _ordered_segment_ids(eval_dataset)
        stitch_data.eval.segments = _derive_segment_specs(
            eval_dataset,
            segment_ids=eval_segment_ids,
            downsample=int(stitch_data.layout.downsample),
            require_supervision_mask=bool(stitch_data.layout.use_roi),
            fallback_to_full_segment=not bool(stitch_data.layout.use_roi),
            use_roi=bool(stitch_data.layout.use_roi),
        )

    if (
        not _cfg_has_segment_specs(stitch_cfg, "log_only")
        and stitch_data.log_only.segment_ids
    ):
        stitch_data.log_only.segments = _derive_segment_specs(
            train_dataset,
            segment_ids=tuple(str(segment_id) for segment_id in stitch_data.log_only.segment_ids),
            downsample=int(stitch_data.layout.downsample),
            require_supervision_mask=False,
            fallback_to_full_segment=True,
            use_roi=bool(stitch_data.layout.use_roi),
        )

    return stitch_data


__all__ = [
    "configure_zarr_stitch_training_loaders",
    "build_zarr_segment_eval_loaders",
    "build_zarr_segment_infer_loaders",
    "prepare_stitch_data_for_bundle",
]
