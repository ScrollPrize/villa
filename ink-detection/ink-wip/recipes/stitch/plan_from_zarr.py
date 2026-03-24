from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ink.core.types import DataBundle
from ink.recipes.components import component_bboxes
from ink.recipes.data.zarr_data import ZarrPatchDataset
from ink.recipes.data.zarr_io import read_optional_supervision_mask_for_shape, read_supervision_mask_for_shape, resolve_segment_volume
from ink.recipes.stitch.config import StitchData, StitchSegmentSpec


def derive_stitch_data_from_bundle(
    bundle: DataBundle,
    *,
    authored_config,
    stitch_data: StitchData,
) -> StitchData:
    train_dataset = getattr(getattr(bundle, "train_loader", None), "dataset", None)
    eval_dataset = getattr(getattr(bundle, "eval_loader", None), "dataset", None)
    if not isinstance(train_dataset, ZarrPatchDataset) or not isinstance(eval_dataset, ZarrPatchDataset):
        return stitch_data

    if not _cfg_has_key(authored_config, "use_roi"):
        stitch_data.layout.use_roi = True

    if not _cfg_has_segment_specs(authored_config, "train"):
        train_segment_ids = tuple(str(segment_id) for segment_id in (stitch_data.train.segment_ids or ()))
        if not train_segment_ids:
            train_segment_ids = _ordered_segment_ids(train_dataset)
        if train_segment_ids:
            stitch_data.train.segments = _derive_segment_specs(
                train_dataset,
                segment_ids=train_segment_ids,
                downsample=int(stitch_data.layout.downsample),
                require_supervision_mask=bool(stitch_data.layout.use_roi),
                fallback_to_full_segment=not bool(stitch_data.layout.use_roi),
                use_roi=bool(stitch_data.layout.use_roi),
            )

    if not _cfg_has_segment_specs(authored_config, "eval"):
        eval_segment_ids = tuple(str(segment_id) for segment_id in (stitch_data.eval.segment_ids or ()))
        if not eval_segment_ids:
            eval_segment_ids = _ordered_segment_ids(eval_dataset)
        if eval_segment_ids:
            stitch_data.eval.segments = _derive_segment_specs(
                eval_dataset,
                segment_ids=eval_segment_ids,
                downsample=int(stitch_data.layout.downsample),
                require_supervision_mask=bool(stitch_data.layout.use_roi),
                fallback_to_full_segment=not bool(stitch_data.layout.use_roi),
                use_roi=bool(stitch_data.layout.use_roi),
            )

    if not _cfg_has_segment_specs(authored_config, "log_only") and stitch_data.log_only.segment_ids:
        log_only_segment_ids = tuple(str(segment_id) for segment_id in stitch_data.log_only.segment_ids)
        stitch_data.log_only.segments = _derive_segment_specs(
            train_dataset,
            segment_ids=log_only_segment_ids,
            downsample=int(stitch_data.layout.downsample),
            require_supervision_mask=False,
            fallback_to_full_segment=True,
            use_roi=bool(stitch_data.layout.use_roi),
        )

    return stitch_data


def _derive_segment_specs(
    dataset: ZarrPatchDataset,
    *,
    segment_ids,
    downsample: int,
    require_supervision_mask: bool,
    fallback_to_full_segment: bool,
    use_roi: bool,
) -> list[StitchSegmentSpec]:
    context = dataset.data_context()
    segment_specs = []
    for segment_id in segment_ids:
        volume = resolve_segment_volume(
            layout=context.layout,
            segments=context.segments,
            segment_id=str(segment_id),
            in_channels=int(context.in_channels),
            volume_cache=context.volume_cache,
        )
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


def _stitch_segment_bbox_rows(
    dataset: ZarrPatchDataset,
    *,
    segment_id: str,
    image_shape_hw: tuple[int, int],
    downsample: int,
    require_supervision_mask: bool,
    fallback_to_full_segment: bool,
) -> tuple[tuple[int, int, int, int], ...] | None:
    context = dataset.data_context()
    cache_key = (str(segment_id), context.label_suffix, context.mask_suffix)
    cached = context.label_mask_store_cache.get(cache_key)
    if int(downsample) == 1 and cached is not None and getattr(cached, "bbox_rows", None):
        return tuple(tuple(int(value) for value in row) for row in cached.bbox_rows)

    supervision_mask = None
    if require_supervision_mask:
        supervision_mask = read_supervision_mask_for_shape(
            context.layout,
            str(segment_id),
            image_shape_hw,
            mask_suffix=context.mask_suffix,
        )

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


def _optional_supervision_mask(dataset: ZarrPatchDataset, segment_id: str, image_shape_hw) -> np.ndarray | None:
    return read_optional_supervision_mask_for_shape(
        dataset.layout,
        str(segment_id),
        image_shape_hw,
        mask_suffix=dataset.mask_suffix,
    )


def _mask_component_bbox_rows(mask: np.ndarray, *, downsample: int) -> tuple[tuple[int, int, int, int], ...] | None:
    mask_ds = _downsample_bool_mask_any(mask, downsample=int(downsample))
    if not bool(mask_ds.any()):
        return None
    return _bbox_rows_or_none(component_bboxes(mask_ds, connectivity=2))


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


def _bbox_rows_or_none(bboxes) -> tuple[tuple[int, int, int, int], ...] | None:
    bboxes_arr = np.asarray(bboxes, dtype=np.int32)
    if bboxes_arr.ndim != 2 or int(bboxes_arr.shape[1]) != 4 or int(bboxes_arr.shape[0]) <= 0:
        return None
    return tuple(tuple(int(value) for value in row) for row in bboxes_arr.tolist())


def _ordered_segment_ids(dataset: ZarrPatchDataset) -> tuple[str, ...]:
    return tuple(str(segment_id) for segment_id in getattr(dataset, "segment_ids", ()) or ())


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


__all__ = ["derive_stitch_data_from_bundle"]
