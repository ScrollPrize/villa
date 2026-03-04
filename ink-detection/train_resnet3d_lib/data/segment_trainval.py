import time

import numpy as np

from train_resnet3d_lib.config import CFG, log
from train_resnet3d_lib.data.dataloaders import build_eval_loader
from train_resnet3d_lib.data.datasets_runtime import CustomDataset, LazyZarrXyLabelDataset
from train_resnet3d_lib.data.patching import (
    _downsample_bool_mask_any,
    _mask_border,
    _mask_component_bboxes_downsample,
    _mask_store_shape,
    extract_patches,
)
from train_resnet3d_lib.data.image_readers import (
    read_image_layers,
    read_image_mask,
    read_label_and_fragment_mask_for_shape,
)
from train_resnet3d_lib.data.zarr_volume import ZarrSegmentVolume
from train_resnet3d_lib.data.patch_index_cache import build_mask_store_and_patch_index_cached
from train_resnet3d_lib.data.segment_metadata import (
    get_segment_layer_range as _segment_layer_range,
    get_segment_reverse_layers as _segment_reverse_layers,
)


def _stitch_mask_geometry(fragment_mask, *, include_train_xyxys):
    mask_border = None
    if include_train_xyxys:
        mask_border = _mask_border(
            _downsample_bool_mask_any(fragment_mask, int(getattr(CFG, "stitch_downsample", 1)))
        )

    mask_bbox = None
    if bool(getattr(CFG, "stitch_use_roi", False)):
        bboxes = _mask_component_bboxes_downsample(
            fragment_mask,
            int(getattr(CFG, "stitch_downsample", 1)),
        )
        if int(bboxes.shape[0]) > 0:
            mask_bbox = bboxes

    return mask_border, mask_bbox


def load_train_segment(
    fragment_id,
    seg_meta,
    group_idx,
    group_name,
    *,
    overlap_segments,
    layers_cache,
    include_train_xyxys,
    label_suffix,
    mask_suffix,
):
    t0 = time.time()
    log(f"load train segment={fragment_id} group={group_name}")
    layer_range = _segment_layer_range(seg_meta, fragment_id)
    reverse_layers = _segment_reverse_layers(seg_meta, fragment_id)
    layers = read_image_layers(
        fragment_id,
        layer_range=layer_range,
    )
    if fragment_id in overlap_segments:
        layers_cache[fragment_id] = layers

    image, mask, fragment_mask = read_image_mask(
        fragment_id,
        reverse_layers=reverse_layers,
        label_suffix=label_suffix,
        mask_suffix=mask_suffix,
        images=layers,
    )
    log(
        f"loaded train segment={fragment_id} "
        f"image={tuple(image.shape)} label={tuple(mask.shape)} mask={tuple(fragment_mask.shape)} "
        f"in {time.time() - t0:.1f}s"
    )
    log(f"extract train patches segment={fragment_id}")
    t1 = time.time()
    frag_train_images, frag_train_masks, frag_train_xyxys = extract_patches(
        image,
        mask,
        fragment_mask,
        include_xyxys=include_train_xyxys,
        filter_empty_tile=True,
    )
    log(f"patches train segment={fragment_id} n={len(frag_train_images)} in {time.time() - t1:.1f}s")
    patch_count = int(len(frag_train_images))

    stitch_candidate = None
    if include_train_xyxys and patch_count > 0:
        frag_train_xyxys = (
            np.stack(frag_train_xyxys) if len(frag_train_xyxys) > 0 else np.zeros((0, 4), dtype=np.int64)
        )
        stitch_candidate = (
            frag_train_images,
            frag_train_masks,
            frag_train_xyxys,
            group_idx,
            tuple(mask.shape),
        )

    mask_border, mask_bbox = _stitch_mask_geometry(
        fragment_mask,
        include_train_xyxys=include_train_xyxys,
    )

    return {
        "patch_count": patch_count,
        "images": frag_train_images,
        "masks": frag_train_masks,
        "group_idx": group_idx,
        "stitch_candidate": stitch_candidate,
        "mask_border": mask_border,
        "mask_bbox": mask_bbox,
    }


def load_train_segment_lazy(
    fragment_id,
    seg_meta,
    group_idx,
    group_name,
    *,
    volume_cache,
    include_train_xyxys,
    label_suffix,
    mask_suffix,
):
    sid = str(fragment_id)
    layer_range = _segment_layer_range(seg_meta, fragment_id)
    reverse_layers = _segment_reverse_layers(seg_meta, fragment_id)

    t0 = time.time()
    log(f"load train segment={sid} group={group_name} (zarr)")
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
        log(f"reuse zarr volume cache for train segment={sid}")

    mask, fragment_mask = read_label_and_fragment_mask_for_shape(
        sid,
        volume.shape[:2],
        label_suffix=label_suffix,
        mask_suffix=mask_suffix,
    )
    mask_store, xyxys, sample_bbox_indices = build_mask_store_and_patch_index_cached(
        mask,
        fragment_mask,
        fragment_id=sid,
        split_name="train",
        filter_empty_tile=True,
        label_suffix=label_suffix,
        mask_suffix=mask_suffix,
    )
    patch_count = int(len(xyxys))
    log(
        f"loaded train segment={sid} image={tuple(volume.shape)} label={tuple(mask.shape)} "
        f"mask={tuple(fragment_mask.shape)} patches={patch_count} in {time.time() - t0:.1f}s"
    )

    mask_border, mask_bbox = _stitch_mask_geometry(
        fragment_mask,
        include_train_xyxys=include_train_xyxys,
    )

    return {
        "sid": sid,
        "group_idx": group_idx,
        "patch_count": patch_count,
        "volume": volume,
        "mask_store": mask_store,
        "xyxys": xyxys,
        "sample_bbox_indices": sample_bbox_indices,
        "mask_border": mask_border,
        "mask_bbox": mask_bbox,
    }


def load_val_segment(
    fragment_id,
    seg_meta,
    group_idx,
    group_name,
    *,
    layers_cache,
    include_train_xyxys,
    valid_transform,
    label_suffix,
    mask_suffix,
):
    t0 = time.time()
    log(f"load val segment={fragment_id} group={group_name}")
    layer_range = _segment_layer_range(seg_meta, fragment_id)
    reverse_layers = _segment_reverse_layers(seg_meta, fragment_id)
    layers = layers_cache.get(fragment_id)
    if layers is None:
        layers = read_image_layers(
            fragment_id,
            layer_range=layer_range,
        )
    else:
        log(f"reuse layers cache for val segment={fragment_id}")

    image_val, mask_val, fragment_mask_val = read_image_mask(
        fragment_id,
        reverse_layers=reverse_layers,
        label_suffix=label_suffix,
        mask_suffix=mask_suffix,
        images=layers,
    )
    log(
        f"loaded val segment={fragment_id} "
        f"image={tuple(image_val.shape)} label={tuple(mask_val.shape)} mask={tuple(fragment_mask_val.shape)} "
        f"in {time.time() - t0:.1f}s"
    )
    log(f"extract val patches segment={fragment_id}")
    t1 = time.time()
    frag_val_images, frag_val_masks, frag_val_xyxys = extract_patches(
        image_val,
        mask_val,
        fragment_mask_val,
        include_xyxys=True,
        filter_empty_tile=False,
    )
    log(f"patches val segment={fragment_id} n={len(frag_val_images)} in {time.time() - t1:.1f}s")

    patch_count = int(len(frag_val_images))
    if patch_count == 0:
        return {
            "patch_count": patch_count,
            "val_loader": None,
            "mask_shape": None,
            "mask_border": None,
        }

    frag_val_xyxys = np.stack(frag_val_xyxys) if len(frag_val_xyxys) > 0 else np.zeros((0, 4), dtype=np.int64)
    frag_val_groups = [group_idx] * len(frag_val_images)
    val_dataset = CustomDataset(
        frag_val_images,
        CFG,
        xyxys=frag_val_xyxys,
        labels=frag_val_masks,
        groups=frag_val_groups,
        transform=valid_transform,
    )
    val_loader = build_eval_loader(val_dataset)

    mask_border, mask_bbox = _stitch_mask_geometry(
        fragment_mask_val,
        include_train_xyxys=include_train_xyxys,
    )

    return {
        "patch_count": patch_count,
        "val_loader": val_loader,
        "mask_shape": tuple(mask_val.shape),
        "mask_border": mask_border,
        "mask_bbox": mask_bbox,
    }


def load_val_segment_lazy(
    fragment_id,
    seg_meta,
    group_idx,
    group_name,
    *,
    volume_cache,
    include_train_xyxys,
    valid_transform,
    label_suffix,
    mask_suffix,
):
    sid = str(fragment_id)
    layer_range = _segment_layer_range(seg_meta, fragment_id)
    reverse_layers = _segment_reverse_layers(seg_meta, fragment_id)

    t0 = time.time()
    log(f"load val segment={sid} group={group_name} (zarr)")
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
        log(f"reuse zarr volume cache for val segment={sid}")

    mask_val, fragment_mask_val = read_label_and_fragment_mask_for_shape(
        sid,
        volume.shape[:2],
        label_suffix=label_suffix,
        mask_suffix=mask_suffix,
    )
    mask_store_val, val_xyxys, val_sample_bbox_indices = build_mask_store_and_patch_index_cached(
        mask_val,
        fragment_mask_val,
        fragment_id=sid,
        split_name="val",
        filter_empty_tile=False,
        label_suffix=label_suffix,
        mask_suffix=mask_suffix,
    )
    patch_count = int(len(val_xyxys))
    log(
        f"loaded val segment={sid} image={tuple(volume.shape)} label={tuple(mask_val.shape)} "
        f"mask={tuple(fragment_mask_val.shape)} patches={patch_count} in {time.time() - t0:.1f}s"
    )
    if patch_count == 0:
        return {
            "patch_count": patch_count,
            "val_loader": None,
            "mask_shape": None,
            "mask_border": None,
        }

    val_dataset = LazyZarrXyLabelDataset(
        {sid: volume},
        {sid: mask_store_val},
        {sid: val_xyxys},
        {sid: group_idx},
        CFG,
        transform=valid_transform,
        sample_bbox_indices_by_segment={sid: val_sample_bbox_indices},
    )
    val_loader = build_eval_loader(val_dataset)
    mask_shape = tuple(_mask_store_shape(mask_store_val))

    mask_border, mask_bbox = _stitch_mask_geometry(
        fragment_mask_val,
        include_train_xyxys=include_train_xyxys,
    )

    return {
        "patch_count": patch_count,
        "val_loader": val_loader,
        "mask_shape": mask_shape,
        "mask_border": mask_border,
        "mask_bbox": mask_bbox,
    }


__all__ = [
    "load_train_segment",
    "load_train_segment_lazy",
    "load_val_segment",
    "load_val_segment_lazy",
]
