import time

import numpy as np

from train_resnet3d_lib.config import CFG, log
from train_resnet3d_lib.data.dataloaders import build_eval_loader
from train_resnet3d_lib.data.datasets_runtime import (
    CustomDataset,
    CustomDatasetTest,
    LazyZarrXyLabelDataset,
    LazyZarrXyOnlyDataset,
)
from train_resnet3d_lib.data.patching import _mask_component_bboxes_downsample, _mask_store_shape, extract_patches_infer
from train_resnet3d_lib.data.image_readers import (
    read_fragment_mask_for_shape,
    read_image_fragment_mask,
    read_image_layers,
)
from train_resnet3d_lib.data.zarr_volume import ZarrSegmentVolume
from train_resnet3d_lib.data.patch_index_cache import extract_infer_patch_coordinates_cached
from train_resnet3d_lib.data.segment_metadata import (
    get_segment_layer_range as _segment_layer_range,
    get_segment_meta as _segment_meta,
    get_segment_reverse_layers as _segment_reverse_layers,
)

_SUPPORTED_DATA_BACKENDS = ("zarr", "tiff")


def _normalize_data_backend(data_backend):
    backend = str(data_backend).strip().lower()
    if backend not in _SUPPORTED_DATA_BACKENDS:
        raise ValueError(f"Unknown training.data_backend: {data_backend!r}. Expected 'zarr' or 'tiff'.")
    return backend


def _resolve_requested_train_stitch_ids(train_fragment_ids, available_segment_ids, stitch_segment_id):
    available = {str(x) for x in available_segment_ids}
    if bool(getattr(CFG, "stitch_all_val", False)):
        return [str(fid) for fid in train_fragment_ids if str(fid) in available]

    requested_ids = []
    if stitch_segment_id is not None and str(stitch_segment_id) in available:
        requested_ids = [str(stitch_segment_id)]
    else:
        for fid in train_fragment_ids:
            if str(fid) in available:
                requested_ids = [str(fid)]
                break
    if not requested_ids:
        log(
            "WARNING: stitch_train is enabled but no train segments had stitch candidates. "
            "No train visualization stitch will be produced."
        )
    return requested_ids


def _collect_train_stitch_loaders(requested_ids, *, build_segment_loader):
    train_stitch_loaders = []
    train_stitch_shapes = []
    train_stitch_segment_ids = []
    for segment_id in requested_ids:
        built = build_segment_loader(str(segment_id))
        if built is None:
            continue
        loader, shape, sid = built
        train_stitch_loaders.append(loader)
        train_stitch_shapes.append(tuple(shape))
        train_stitch_segment_ids.append(str(sid))
    return train_stitch_loaders, train_stitch_shapes, train_stitch_segment_ids


def _build_train_stitch_common(
    train_fragment_ids,
    available_segment_ids,
    stitch_segment_id,
    *,
    build_segment_loader,
):
    if not bool(getattr(CFG, "stitch_train", False)):
        return [], [], []
    requested_ids = _resolve_requested_train_stitch_ids(
        train_fragment_ids,
        available_segment_ids,
        stitch_segment_id,
    )
    return _collect_train_stitch_loaders(
        requested_ids,
        build_segment_loader=build_segment_loader,
    )


def build_train_stitch_loaders(train_fragment_ids, train_stitch_candidates, stitch_segment_id, *, valid_transform):
    def _build_segment_loader(segment_id):
        entry = train_stitch_candidates.get(str(segment_id))
        if entry is None:
            return None
        seg_images, seg_masks, seg_xyxys, group_idx, seg_shape = entry
        seg_groups = [int(group_idx)] * len(seg_images)
        train_dataset_viz = CustomDataset(
            seg_images,
            CFG,
            xyxys=seg_xyxys,
            labels=seg_masks,
            groups=seg_groups,
            transform=valid_transform,
        )
        return build_eval_loader(train_dataset_viz), tuple(seg_shape), str(segment_id)

    return _build_train_stitch_common(
        train_fragment_ids,
        train_stitch_candidates.keys(),
        stitch_segment_id,
        build_segment_loader=_build_segment_loader,
    )


def build_train_stitch_loaders_lazy(
    train_fragment_ids,
    train_volumes_by_segment,
    train_masks_by_segment,
    train_xyxys_by_segment,
    train_sample_bbox_indices_by_segment,
    train_groups_by_segment,
    stitch_segment_id,
    *,
    valid_transform,
):
    def _build_segment_loader(segment_id):
        sid = str(segment_id)
        xy = train_xyxys_by_segment.get(sid)
        if xy is None or int(len(xy)) == 0:
            return None
        if sid not in train_volumes_by_segment or sid not in train_masks_by_segment:
            return None
        bbox_idx = train_sample_bbox_indices_by_segment.get(sid)
        if bbox_idx is None:
            bbox_idx = np.full((int(len(xy)),), -1, dtype=np.int32)

        dataset = LazyZarrXyLabelDataset(
            {sid: train_volumes_by_segment[sid]},
            {sid: train_masks_by_segment[sid]},
            {sid: xy},
            {sid: int(train_groups_by_segment.get(sid, 0))},
            CFG,
            transform=valid_transform,
            sample_bbox_indices_by_segment={sid: bbox_idx},
        )
        return build_eval_loader(dataset), tuple(_mask_store_shape(train_masks_by_segment[sid])), sid

    return _build_train_stitch_common(
        train_fragment_ids,
        train_xyxys_by_segment.keys(),
        stitch_segment_id,
        build_segment_loader=_build_segment_loader,
    )


def build_train_stitch_outputs(
    *,
    data_backend,
    train_fragment_ids,
    stitch_segment_id,
    valid_transform,
    train_stitch_candidates=None,
    train_volumes_by_segment=None,
    train_masks_by_segment=None,
    train_xyxys_by_segment=None,
    train_sample_bbox_indices_by_segment=None,
    train_groups_by_segment=None,
):
    backend = _normalize_data_backend(data_backend)
    train_stitch_builder = _train_stitch_builder_for_backend(backend=backend)
    return train_stitch_builder(
        train_fragment_ids=train_fragment_ids,
        stitch_segment_id=stitch_segment_id,
        valid_transform=valid_transform,
        train_stitch_candidates=train_stitch_candidates,
        train_volumes_by_segment=train_volumes_by_segment,
        train_masks_by_segment=train_masks_by_segment,
        train_xyxys_by_segment=train_xyxys_by_segment,
        train_sample_bbox_indices_by_segment=train_sample_bbox_indices_by_segment,
        train_groups_by_segment=train_groups_by_segment,
    )


def _collect_log_only_outputs(log_only_segments, *, build_segment_output):
    log_only_loaders = []
    log_only_shapes = []
    log_only_segment_ids = []
    log_only_bboxes = {}

    for fragment_id in log_only_segments:
        built = build_segment_output(str(fragment_id))
        if built is None:
            continue
        loader, mask_shape, segment_id, bboxes = built
        sid = str(segment_id)
        log_only_loaders.append(loader)
        log_only_shapes.append(tuple(mask_shape))
        log_only_segment_ids.append(sid)
        if bboxes is not None and int(bboxes.shape[0]) > 0:
            log_only_bboxes[sid] = bboxes

    return log_only_loaders, log_only_shapes, log_only_segment_ids, log_only_bboxes


def build_log_only_stitch_loaders(
    log_only_segments,
    *,
    segments_metadata,
    layers_cache,
    valid_transform,
    mask_suffix,
    log_only_downsample,
):
    def _build_segment_output(fragment_id):
        seg_meta = _segment_meta(segments_metadata, fragment_id)
        layer_range = _segment_layer_range(seg_meta, fragment_id)
        reverse_layers = _segment_reverse_layers(seg_meta, fragment_id)
        layers = layers_cache.get(fragment_id)
        if layers is None:
            layers = read_image_layers(
                fragment_id,
                layer_range=layer_range,
            )
        else:
            log(f"reuse layers cache for log-only segment={fragment_id}")

        image, fragment_mask = read_image_fragment_mask(
            fragment_id,
            reverse_layers=reverse_layers,
            mask_suffix=mask_suffix,
            images=layers,
        )

        log(f"extract log-only patches segment={fragment_id}")
        t0 = time.time()
        images, xyxys = extract_patches_infer(image, fragment_mask, include_xyxys=True)
        log(f"patches log-only segment={fragment_id} n={len(images)} in {time.time() - t0:.1f}s")
        if len(images) == 0:
            return None

        xyxys = np.stack(xyxys) if len(xyxys) > 0 else np.zeros((0, 4), dtype=np.int64)
        dataset = CustomDatasetTest(images, xyxys, CFG, transform=valid_transform)
        loader = build_eval_loader(dataset)
        bboxes = _mask_component_bboxes_downsample(fragment_mask, int(log_only_downsample))
        return loader, tuple(fragment_mask.shape), str(fragment_id), bboxes

    return _collect_log_only_outputs(
        log_only_segments,
        build_segment_output=_build_segment_output,
    )


def build_log_only_stitch_loaders_lazy(
    log_only_segments,
    *,
    segments_metadata,
    volume_cache,
    valid_transform,
    mask_suffix,
    log_only_downsample,
):
    def _build_segment_output(fragment_id):
        sid = str(fragment_id)
        seg_meta = _segment_meta(segments_metadata, fragment_id)
        layer_range = _segment_layer_range(seg_meta, fragment_id)
        reverse_layers = _segment_reverse_layers(seg_meta, fragment_id)
        volume = volume_cache.get(sid)
        if volume is None:
            volume = ZarrSegmentVolume(
                sid,
                seg_meta,
                layer_range=layer_range,
                reverse_layers=reverse_layers,
            )
            volume_cache[sid] = volume
        else:
            log(f"reuse zarr volume cache for log-only segment={sid}")

        fragment_mask = read_fragment_mask_for_shape(
            sid,
            volume.shape[:2],
            mask_suffix=mask_suffix,
        )
        xyxys = extract_infer_patch_coordinates_cached(
            fragment_mask=fragment_mask,
            fragment_id=sid,
            mask_suffix=mask_suffix,
            split_name="val",
        )
        log(f"patches log-only segment={sid} n={int(len(xyxys))}")
        if int(len(xyxys)) == 0:
            return None

        dataset = LazyZarrXyOnlyDataset(
            {sid: volume},
            {sid: xyxys},
            CFG,
            transform=valid_transform,
        )
        loader = build_eval_loader(dataset)
        bboxes = _mask_component_bboxes_downsample(fragment_mask, int(log_only_downsample))
        return loader, tuple(fragment_mask.shape), sid, bboxes

    return _collect_log_only_outputs(
        log_only_segments,
        build_segment_output=_build_segment_output,
    )


def build_log_only_outputs(
    *,
    data_backend,
    log_only_segments,
    segments_metadata,
    valid_transform,
    mask_suffix,
    log_only_downsample,
    layers_cache=None,
    volume_cache=None,
):
    requested_segments = list(log_only_segments or [])
    if not requested_segments:
        return [], [], [], {}

    backend = _normalize_data_backend(data_backend)
    log_only_builder = _log_only_builder_for_backend(backend=backend)
    return log_only_builder(
        log_only_segments=requested_segments,
        segments_metadata=segments_metadata,
        valid_transform=valid_transform,
        mask_suffix=mask_suffix,
        log_only_downsample=log_only_downsample,
        layers_cache=layers_cache,
        volume_cache=volume_cache,
    )


def _build_train_stitch_outputs_zarr(
    *,
    train_fragment_ids,
    stitch_segment_id,
    valid_transform,
    train_volumes_by_segment,
    train_masks_by_segment,
    train_xyxys_by_segment,
    train_sample_bbox_indices_by_segment,
    train_groups_by_segment,
    **_,
):
    return build_train_stitch_loaders_lazy(
        train_fragment_ids,
        train_volumes_by_segment,
        train_masks_by_segment,
        train_xyxys_by_segment,
        train_sample_bbox_indices_by_segment,
        train_groups_by_segment,
        stitch_segment_id,
        valid_transform=valid_transform,
    )


def _build_train_stitch_outputs_tiff(
    *,
    train_fragment_ids,
    stitch_segment_id,
    valid_transform,
    train_stitch_candidates,
    **_,
):
    return build_train_stitch_loaders(
        train_fragment_ids,
        train_stitch_candidates,
        stitch_segment_id,
        valid_transform=valid_transform,
    )


def _train_stitch_builder_for_backend(*, backend):
    if backend == "zarr":
        return _build_train_stitch_outputs_zarr
    return _build_train_stitch_outputs_tiff


def _build_log_only_outputs_zarr(
    *,
    log_only_segments,
    segments_metadata,
    valid_transform,
    mask_suffix,
    log_only_downsample,
    volume_cache,
    **_,
):
    return build_log_only_stitch_loaders_lazy(
        log_only_segments,
        segments_metadata=segments_metadata,
        volume_cache=volume_cache,
        valid_transform=valid_transform,
        mask_suffix=mask_suffix,
        log_only_downsample=log_only_downsample,
    )


def _build_log_only_outputs_tiff(
    *,
    log_only_segments,
    segments_metadata,
    valid_transform,
    mask_suffix,
    log_only_downsample,
    layers_cache,
    **_,
):
    return build_log_only_stitch_loaders(
        log_only_segments,
        segments_metadata=segments_metadata,
        layers_cache=layers_cache,
        valid_transform=valid_transform,
        mask_suffix=mask_suffix,
        log_only_downsample=log_only_downsample,
    )


def _log_only_builder_for_backend(*, backend):
    if backend == "zarr":
        return _build_log_only_outputs_zarr
    return _build_log_only_outputs_tiff
