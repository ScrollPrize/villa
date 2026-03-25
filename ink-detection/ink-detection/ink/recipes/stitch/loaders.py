from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from torch.utils.data import DataLoader

from ink.recipes.data.patching import build_patch_index
from ink.recipes.data.layout import resolve_layout_stitch_roi_mask_names
from ink.recipes.data.zarr_data import (
    ZarrPatchDataset,
    build_zarr_split_samples,
    collate_infer_batch,
    collate_patch_batch,
)
from ink.recipes.data.zarr_io import (
    ZarrSegmentLabelMaskStore,
    read_optional_supervision_mask_for_shape,
    resolve_segment_volume,
)
from ink.recipes.stitch.config import StitchData
from ink.recipes.stitch.infer_dataset import ZarrInferDataset, ZarrInferGridDataset


def build_stitch_runtime_loaders(*, stitch_data: StitchData, train_loader, eval_loader) -> tuple[list[DataLoader], list[DataLoader]]:
    train_dataset = getattr(train_loader, "dataset", None)
    if not isinstance(train_dataset, ZarrPatchDataset):
        return [], []

    batch_size, num_workers = _resolve_loader_build_settings(train_loader=train_loader, eval_loader=eval_loader)
    train_viz = getattr(getattr(stitch_data, "train", None), "viz", None)
    train_segment_ids = _segment_spec_ids(getattr(getattr(stitch_data, "train", None), "segments", ()))
    train_viz_loaders: list[DataLoader] = []
    if bool(getattr(train_viz, "enabled", False)) and train_segment_ids:
        train_viz_loaders = build_zarr_segment_eval_loaders(
            train_dataset,
            segment_ids=train_segment_ids,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    log_only_segment_ids = tuple(
        str(segment_id)
        for segment_id in getattr(getattr(stitch_data, "log_only", None), "segment_ids", ()) or ()
    )
    log_only_loaders: list[DataLoader] = []
    if log_only_segment_ids:
        log_only_loaders = build_zarr_segment_infer_loaders(
            train_dataset,
            segment_ids=log_only_segment_ids,
            batch_size=batch_size,
            num_workers=num_workers,
            use_roi=bool(stitch_data.layout.use_roi),
        )
    return train_viz_loaders, log_only_loaders


def build_stitch_inference_loaders(*, stitch_data: StitchData, eval_loader) -> list[DataLoader]:
    eval_dataset = getattr(eval_loader, "dataset", None)
    if not isinstance(eval_dataset, ZarrPatchDataset):
        return [] if eval_loader is None else [eval_loader]

    eval_segment_ids = _segment_spec_ids(getattr(getattr(stitch_data, "eval", None), "segments", ()))
    if not eval_segment_ids:
        eval_segment_ids = tuple(
            str(segment_id)
            for segment_id in getattr(getattr(stitch_data, "eval", None), "segment_ids", ()) or ()
        )
    if not eval_segment_ids:
        return [eval_loader]

    batch_size = int(getattr(eval_loader, "batch_size", None) or 1)
    num_workers = int(getattr(eval_loader, "num_workers", None) or 0)
    return build_zarr_segment_infer_loaders(
        eval_dataset,
        segment_ids=eval_segment_ids,
        batch_size=batch_size,
        num_workers=num_workers,
        use_roi=bool(stitch_data.layout.use_roi),
    )


def build_zarr_segment_eval_loaders(
    dataset,
    *,
    segment_ids,
    batch_size: int,
    num_workers: int = 0,
) -> list[DataLoader]:
    if not isinstance(dataset, ZarrPatchDataset):
        return []
    requested_segment_ids = tuple(str(segment_id) for segment_id in (segment_ids or ()))
    samples_by_segment, missing_segment_ids = _split_existing_and_missing_segment_samples(
        dataset,
        segment_ids=requested_segment_ids,
    )
    if missing_segment_ids:
        rebuilt_samples = build_zarr_split_samples(
            dataset.data_context(),
            segment_ids=missing_segment_ids,
            patch_size=int(dataset.patch_size),
            tile_size=int(dataset.tile_size),
            stride=int(dataset.stride),
            group_segment_ids=tuple(
                str(segment_id)
                for segment_id in (*getattr(dataset, "segment_ids", ()), *missing_segment_ids)
            ),
        )
        rebuilt_samples_by_segment = _samples_by_segment(rebuilt_samples, segment_ids=missing_segment_ids)
        for segment_id in missing_segment_ids:
            samples_by_segment[str(segment_id)] = rebuilt_samples_by_segment[str(segment_id)]
    return _build_segment_loaders(
        dataset_cls=ZarrPatchDataset,
        dataset=dataset,
        segment_ids=requested_segment_ids,
        samples_by_segment=samples_by_segment,
        batch_size=batch_size,
        num_workers=num_workers,
        include_tile_config=True,
        collate_fn=collate_patch_batch,
    )


def build_zarr_segment_infer_loaders(
    dataset,
    *,
    segment_ids,
    batch_size: int,
    num_workers: int = 0,
    use_roi: bool = True,
) -> list[DataLoader]:
    if not isinstance(dataset, ZarrPatchDataset):
        return []
    requested_segment_ids = tuple(str(segment_id) for segment_id in (segment_ids or ()))
    if not bool(use_roi):
        return _build_segment_grid_infer_loaders(
            dataset,
            segment_ids=requested_segment_ids,
            batch_size=batch_size,
            num_workers=num_workers,
            skip_empty=True,
        )
    infer_samples = _build_roi_infer_samples_from_segments(
        dataset,
        segment_ids=requested_segment_ids,
    )
    return _build_segment_loaders(
        dataset_cls=ZarrInferDataset,
        dataset=dataset,
        segment_ids=requested_segment_ids,
        samples_by_segment=_samples_by_segment(infer_samples, segment_ids=requested_segment_ids),
        batch_size=batch_size,
        num_workers=num_workers,
        include_tile_config=False,
        collate_fn=collate_infer_batch,
    )


def _resolve_loader_build_settings(*, train_loader, eval_loader) -> tuple[int, int]:
    eval_batch_size = getattr(eval_loader, "batch_size", None)
    train_batch_size = getattr(train_loader, "batch_size", None)
    batch_size = int(eval_batch_size if eval_batch_size is not None else train_batch_size or 1)

    eval_num_workers = getattr(eval_loader, "num_workers", None)
    train_num_workers = getattr(train_loader, "num_workers", None)
    num_workers = int(eval_num_workers if eval_num_workers is not None else train_num_workers or 0)
    return batch_size, num_workers


def _build_segment_loaders(
    *,
    dataset_cls,
    dataset: ZarrPatchDataset,
    segment_ids: tuple[str, ...],
    samples_by_segment: Mapping[str, list],
    batch_size: int,
    num_workers: int,
    include_tile_config: bool,
    collate_fn,
) -> list[DataLoader]:
    context = dataset.data_context()
    if include_tile_config:
        dataset_kwargs = {
            "layout": context.layout,
            "segments": context.segments,
            "split": "valid",
            "augment": dataset.augment,
            "normalization": dataset.normalization,
            "patch_size": int(dataset.patch_size),
            "tile_size": int(dataset.tile_size),
            "stride": int(dataset.stride),
            "in_channels": int(dataset.in_channels),
            "label_suffix": context.label_suffix,
            "mask_suffix": context.mask_suffix,
            "mask_name": context.mask_name,
            "mask_split_name": getattr(dataset, "mask_split_name", dataset.split),
            "train_segment_ids": tuple(str(segment_id) for segment_id in getattr(dataset, "train_segment_ids", ())),
            "volume_cache": context.volume_cache,
            "label_mask_store_cache": context.label_mask_store_cache,
            "patch_index_cache_dir": context.patch_index_cache_dir,
        }
    else:
        dataset_kwargs = {
            "layout": context.layout,
            "segments": context.segments,
            "augment": dataset.augment,
            "normalization": dataset.normalization,
            "patch_size": int(dataset.patch_size),
            "in_channels": int(dataset.in_channels),
            "volume_cache": context.volume_cache,
        }

    return [
        _build_loader(
            dataset_cls(
                samples_by_segment.get(str(segment_id), ()),
                segment_ids=(str(segment_id),),
                **dataset_kwargs,
            ),
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
        for segment_id in segment_ids
    ]


def _build_segment_grid_infer_loaders(
    dataset: ZarrPatchDataset,
    *,
    segment_ids: tuple[str, ...],
    batch_size: int,
    num_workers: int,
    skip_empty: bool,
) -> list[DataLoader]:
    context = dataset.data_context()
    return [
        _build_loader(
            ZarrInferGridDataset(
                layout=context.layout,
                segments=context.segments,
                augment=dataset.augment,
                normalization=dataset.normalization,
                patch_size=int(dataset.patch_size),
                tile_size=int(dataset.tile_size),
                stride=int(dataset.stride),
                in_channels=int(dataset.in_channels),
                segment_id=str(segment_id),
                volume_cache=context.volume_cache,
                skip_empty=bool(skip_empty),
            ),
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_infer_batch,
        )
        for segment_id in segment_ids
    ]


def _build_loader(dataset, *, batch_size: int, num_workers: int, collate_fn) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=max(0, int(num_workers)),
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )


def _build_roi_infer_samples_from_segments(
    dataset: ZarrPatchDataset,
    *,
    segment_ids: tuple[str, ...],
) -> list[tuple[str, tuple[int, int, int, int]]]:
    context = dataset.data_context()
    split_samples = []
    for segment_id in segment_ids:
        volume = resolve_segment_volume(
            layout=context.layout,
            segments=context.segments,
            segment_id=segment_id,
            in_channels=int(context.in_channels),
            volume_cache=context.volume_cache,
        )
        supervision_mask = _optional_supervision_mask(
            dataset,
            segment_id=segment_id,
            image_shape_hw=volume.image_shape_hw,
        )
        if supervision_mask is None or not bool(np.asarray(supervision_mask).any()):
            supervision_mask = np.full(volume.image_shape_hw, 255, dtype=np.uint8)
        bbox_rows, xyxys, _sample_bbox_indices = build_patch_index(
            None,
            supervision_mask,
            size=int(dataset.patch_size),
            tile_size=int(dataset.tile_size),
            stride=int(dataset.stride),
            filter_empty_tile=False,
        )
        if int(bbox_rows.shape[0]) > 0:
            mask_names = _stitch_roi_mask_names(dataset, segment_id=segment_id)
            context.label_mask_store_cache[
                (segment_id, context.label_suffix, context.mask_suffix, mask_names)
            ] = ZarrSegmentLabelMaskStore(
                layout=context.layout,
                segment_id=str(segment_id),
                image_shape_hw=volume.image_shape_hw,
                label_suffix=context.label_suffix,
                mask_suffix=context.mask_suffix,
                mask_names=mask_names,
                bbox_rows=bbox_rows,
            )
        split_samples.extend(
            (
                segment_id,
                tuple(int(value) for value in xyxy),
            )
            for xyxy in np.asarray(xyxys, dtype=np.int64)
        )
    return split_samples


def _samples_by_segment(samples, *, segment_ids: tuple[str, ...]) -> dict[str, list]:
    grouped = {str(segment_id): [] for segment_id in segment_ids}
    for sample in samples:
        grouped[str(sample[0])].append(sample)
    return grouped


def _split_existing_and_missing_segment_samples(
    dataset: ZarrPatchDataset,
    *,
    segment_ids: tuple[str, ...],
) -> tuple[dict[str, list], tuple[str, ...]]:
    requested_segment_ids = tuple(str(segment_id) for segment_id in segment_ids)
    if not requested_segment_ids:
        return {}, ()
    available_segment_ids = set(_ordered_segment_ids(dataset))
    requested_segment_id_set = set(requested_segment_ids)
    samples_by_segment = _samples_by_segment(
        [
            sample
            for sample in dataset.sample_rows
            if str(sample[0]) in requested_segment_id_set
        ],
        segment_ids=requested_segment_ids,
    )
    missing_segment_ids = tuple(
        segment_id
        for segment_id in requested_segment_ids
        if segment_id not in available_segment_ids
    )
    return samples_by_segment, missing_segment_ids


def _ordered_segment_ids(dataset: ZarrPatchDataset) -> tuple[str, ...]:
    return tuple(str(segment_id) for segment_id in getattr(dataset, "segment_ids", ()) or ())


def _optional_supervision_mask(dataset: ZarrPatchDataset, *, segment_id: str, image_shape_hw) -> np.ndarray | None:
    return read_optional_supervision_mask_for_shape(
        dataset.layout,
        str(segment_id),
        image_shape_hw,
        mask_suffix=dataset.mask_suffix,
        mask_names=_stitch_roi_mask_names(dataset, segment_id=segment_id),
    )


def _stitch_roi_mask_names(dataset: ZarrPatchDataset, *, segment_id: str) -> tuple[str, ...]:
    return resolve_layout_stitch_roi_mask_names(
        layout=dataset.layout,
        segment_id=str(segment_id),
        mask_suffix=dataset.mask_suffix,
    )


def _segment_spec_ids(segment_specs) -> tuple[str, ...]:
    return tuple(str(spec.segment_id) for spec in (segment_specs or ()))


__all__ = [
    "build_stitch_inference_loaders",
    "build_stitch_runtime_loaders",
    "build_zarr_segment_eval_loaders",
    "build_zarr_segment_infer_loaders",
]
