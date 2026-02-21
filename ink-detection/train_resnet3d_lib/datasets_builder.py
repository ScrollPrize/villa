import hashlib
import json
import math
import os
import os.path as osp
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from samplers import GroupStratifiedBatchSampler

from train_resnet3d_lib.config import CFG, log
from train_resnet3d_lib.data_ops import (
    build_group_mappings,
    read_image_layers,
    read_image_mask,
    read_image_fragment_mask,
    read_label_and_fragment_mask_for_shape,
    read_fragment_mask_for_shape,
    ZarrSegmentVolume,
    extract_patches,
    extract_patches_infer,
    extract_patch_coordinates,
    get_transforms,
    CustomDataset,
    CustomDatasetTest,
    LazyZarrTrainDataset,
    LazyZarrXyLabelDataset,
    LazyZarrXyOnlyDataset,
    _build_mask_store_and_patch_index,
    _mask_border,
    _downsample_bool_mask_any,
    _mask_component_bboxes_downsample,
    _mask_store_shape,
)

_PATCH_INDEX_CACHE_SCHEMA_VERSION = 1
_PATCH_INDEX_CACHE_REQUIRED_META_KEYS = {
    "schema_version",
    "fragment_id",
    "split",
    "filter_empty_tile",
    "size",
    "tile_size",
    "stride",
    "label_suffix",
    "mask_suffix",
    "mask_shape",
    "mask_store_mode",
}
_PATCH_INDEX_CACHE_HASH_META_KEYS = {
    "label_sha256",
    "fragment_mask_sha256",
}


def _segment_meta(segments_metadata, fragment_id):
    if fragment_id not in segments_metadata:
        raise KeyError(f"segments metadata missing fragment id: {fragment_id!r}")
    seg_meta = segments_metadata[fragment_id]
    if not isinstance(seg_meta, dict):
        raise TypeError(f"segments[{fragment_id!r}] must be an object, got {type(seg_meta).__name__}")
    return seg_meta


def _segment_layer_range(seg_meta, fragment_id):
    if "layer_range" not in seg_meta:
        raise KeyError(f"segments[{fragment_id!r}] missing required key: 'layer_range'")
    layer_range = seg_meta["layer_range"]
    if not isinstance(layer_range, (list, tuple)) or len(layer_range) != 2:
        raise TypeError(
            f"segments[{fragment_id!r}].layer_range must be [start_idx, end_idx], got {layer_range!r}"
        )
    start_idx = int(layer_range[0])
    end_idx = int(layer_range[1])
    if end_idx <= start_idx:
        raise ValueError(
            f"segments[{fragment_id!r}].layer_range must satisfy end_idx > start_idx, got {layer_range!r}"
        )
    return start_idx, end_idx


def _segment_reverse_layers(seg_meta, fragment_id):
    if "reverse_layers" not in seg_meta:
        raise KeyError(f"segments[{fragment_id!r}] missing required key: 'reverse_layers'")
    reverse_layers = seg_meta["reverse_layers"]
    if not isinstance(reverse_layers, bool):
        raise TypeError(
            f"segments[{fragment_id!r}].reverse_layers must be boolean, got {type(reverse_layers).__name__}"
        )
    return reverse_layers


def _require_2d_array(value, *, context):
    arr = np.asarray(value)
    if arr.ndim != 2:
        raise ValueError(f"{context}: expected 2D array, got shape={tuple(arr.shape)}")
    return arr


def _label_mask_uint8(mask, *, context):
    mask_arr = _require_2d_array(mask, context=context)
    if mask_arr.dtype == np.uint8:
        return mask_arr
    if np.issubdtype(mask_arr.dtype, np.integer) or np.issubdtype(mask_arr.dtype, np.floating):
        return np.clip(mask_arr, 0, 255).astype(np.uint8, copy=False)
    raise TypeError(f"{context}: expected integer/float label mask dtype, got {mask_arr.dtype}")


def _sha256_array(arr, *, context):
    arr_2d = _require_2d_array(arr, context=context)
    arr_c = np.ascontiguousarray(arr_2d)
    hasher = hashlib.sha256()
    hasher.update(np.asarray(arr_c.shape, dtype=np.int64).tobytes())
    hasher.update(str(arr_c.dtype).encode("utf-8"))
    hasher.update(arr_c.tobytes(order="C"))
    return hasher.hexdigest()


def _cache_segment_slug(segment_id, *, max_len=80):
    raw = str(segment_id)
    normalized = "".join(ch if (ch.isalnum() or ch in {"-", "_", "."}) else "_" for ch in raw)
    normalized = normalized.strip("._-")
    if not normalized:
        normalized = "segment"
    return normalized[:max_len]


def _resolve_patch_index_cache_dir():
    cache_dir_value = getattr(CFG, "dataset_cache_dir", "./dataset_cache/train_resnet3d_patch_index")
    if not isinstance(cache_dir_value, str):
        raise TypeError(f"CFG.dataset_cache_dir must be a string, got {type(cache_dir_value).__name__}")
    cache_dir = cache_dir_value.strip()
    if not cache_dir:
        raise ValueError("CFG.dataset_cache_dir must be a non-empty string")
    if not osp.isabs(cache_dir):
        cache_dir = osp.abspath(cache_dir)
    return cache_dir


def _build_patch_index_cache_params(
    *,
    fragment_id,
    split_name,
    filter_empty_tile,
    label_suffix,
    mask_suffix,
):
    split = str(split_name)
    if split not in {"train", "val"}:
        raise ValueError(f"split_name must be 'train' or 'val', got {split_name!r}")
    return {
        "schema_version": int(_PATCH_INDEX_CACHE_SCHEMA_VERSION),
        "fragment_id": str(fragment_id),
        "split": split,
        "filter_empty_tile": bool(filter_empty_tile),
        "size": int(CFG.size),
        "tile_size": int(CFG.tile_size),
        "stride": int(CFG.stride),
        "label_suffix": str(label_suffix),
        "mask_suffix": str(mask_suffix),
    }


def _build_patch_index_expected_metadata(params, mask, fragment_mask, *, check_hash):
    mask_arr = _require_2d_array(mask, context="patch-index cache label")
    fragment_mask_arr = _require_2d_array(fragment_mask, context="patch-index cache fragment mask")
    if mask_arr.shape != fragment_mask_arr.shape:
        raise ValueError(
            "patch-index cache label/mask shape mismatch: "
            f"{tuple(mask_arr.shape)} vs {tuple(fragment_mask_arr.shape)}"
        )
    metadata = dict(params)
    metadata["mask_shape"] = [int(mask_arr.shape[0]), int(mask_arr.shape[1])]
    if bool(check_hash):
        metadata["label_sha256"] = _sha256_array(mask_arr, context="patch-index cache label")
        metadata["fragment_mask_sha256"] = _sha256_array(
            fragment_mask_arr,
            context="patch-index cache fragment mask",
        )
    return metadata


def _patch_index_cache_file_path(cache_dir, params):
    params_json = json.dumps(params, sort_keys=True, separators=(",", ":"))
    params_hash = hashlib.sha256(params_json.encode("utf-8")).hexdigest()[:16]
    segment_slug = _cache_segment_slug(params["fragment_id"])
    filename = f"{segment_slug}__{params['split']}__{params_hash}.npz"
    return osp.join(cache_dir, filename)


def _atomic_save_npz(path, payload):
    directory = osp.dirname(path)
    os.makedirs(directory, exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}.{time.time_ns()}"
    try:
        with open(tmp_path, "wb") as f:
            np.savez_compressed(f, **payload)
        os.replace(tmp_path, path)
    finally:
        if osp.exists(tmp_path):
            os.remove(tmp_path)


def _serialize_patch_index_cache_payload(
    *,
    metadata,
    mask_store,
    xyxys,
    sample_bbox_indices,
):
    if not isinstance(mask_store, dict):
        raise TypeError(f"mask_store must be a dict, got {type(mask_store).__name__}")

    xy = np.asarray(xyxys, dtype=np.int64)
    if xy.ndim != 2 or xy.shape[1] != 4:
        raise ValueError(f"xyxys must have shape (N, 4), got {tuple(xy.shape)}")
    bbox_idx = np.asarray(sample_bbox_indices, dtype=np.int32).reshape(-1)
    if bbox_idx.shape[0] != xy.shape[0]:
        raise ValueError(
            f"sample_bbox_indices length {bbox_idx.shape[0]} does not match xyxys length {xy.shape[0]}"
        )

    mode = str(mask_store.get("mode"))
    payload_meta = dict(metadata)
    payload_meta["mask_store_mode"] = mode
    if mode == "full":
        if xy.shape[0] > 0 and np.any(bbox_idx != -1):
            raise ValueError("mask_store mode='full' requires all sample_bbox_indices to be -1")
        bboxes = np.zeros((0, 4), dtype=np.int32)
    elif mode == "bboxes":
        bboxes = np.asarray(mask_store.get("bboxes"), dtype=np.int32)
        if bboxes.ndim != 2 or bboxes.shape[1] != 4:
            raise ValueError(f"mask_store mode='bboxes' requires bboxes shape (N, 4), got {tuple(bboxes.shape)}")
        if xy.shape[0] > 0:
            if int(bboxes.shape[0]) <= 0:
                raise ValueError("mask_store mode='bboxes' has patches but no bboxes")
            if np.any(bbox_idx < 0):
                raise ValueError("mask_store mode='bboxes' requires non-negative sample_bbox_indices")
            max_idx = int(bbox_idx.max())
            if max_idx >= int(bboxes.shape[0]):
                raise ValueError(
                    f"sample_bbox_indices max={max_idx} is out of range for {int(bboxes.shape[0])} bboxes"
                )
    else:
        raise ValueError(f"unsupported mask_store mode: {mode!r}")

    return {
        "metadata_json": np.asarray(json.dumps(payload_meta, sort_keys=True)),
        "xyxys": xy,
        "sample_bbox_indices": bbox_idx,
        "bboxes": bboxes.astype(np.int32, copy=False),
    }


def _save_patch_index_cache(
    *,
    cache_path,
    metadata,
    mask_store,
    xyxys,
    sample_bbox_indices,
):
    payload = _serialize_patch_index_cache_payload(
        metadata=metadata,
        mask_store=mask_store,
        xyxys=xyxys,
        sample_bbox_indices=sample_bbox_indices,
    )
    _atomic_save_npz(cache_path, payload)


def _rebuild_mask_store_from_cached_bboxes(mask, bboxes):
    mask_u8 = _label_mask_uint8(mask, context="patch-index cache label")
    bboxes_i32 = np.asarray(bboxes, dtype=np.int32)
    if bboxes_i32.ndim != 2 or bboxes_i32.shape[1] != 4:
        raise ValueError(f"cached bboxes must have shape (N, 4), got {tuple(bboxes_i32.shape)}")

    mask_h = int(mask_u8.shape[0])
    mask_w = int(mask_u8.shape[1])
    mask_crops = []
    for bbox in bboxes_i32.tolist():
        y0, y1, x0, x1 = [int(v) for v in bbox]
        if y1 <= y0 or x1 <= x0:
            raise ValueError(f"invalid cached bbox: {(y0, y1, x0, x1)}")
        if y0 < 0 or x0 < 0 or y1 > mask_h or x1 > mask_w:
            raise ValueError(f"cached bbox out of bounds for mask shape {tuple(mask_u8.shape)}: {(y0, y1, x0, x1)}")
        mask_crops.append(np.asarray(mask_u8[y0:y1, x0:x1], dtype=np.uint8).copy())

    return {
        "mode": "bboxes",
        "shape": tuple(mask_u8.shape),
        "bboxes": bboxes_i32,
        "mask_crops": mask_crops,
    }


def _load_patch_index_cache(
    *,
    cache_path,
    expected_metadata,
    mask,
):
    if not osp.exists(cache_path):
        return None

    with np.load(cache_path, allow_pickle=False) as npz_data:
        required_keys = {"metadata_json", "xyxys", "sample_bbox_indices", "bboxes"}
        missing_keys = required_keys - set(npz_data.files)
        if missing_keys:
            raise ValueError(f"cache file {cache_path!r} is missing keys: {sorted(missing_keys)!r}")

        raw_metadata = npz_data["metadata_json"]
        metadata_text = str(np.asarray(raw_metadata).item())
        metadata = json.loads(metadata_text)
        if not isinstance(metadata, dict):
            raise TypeError(f"cache metadata must be an object, got {type(metadata).__name__}")

        missing_meta_keys = _PATCH_INDEX_CACHE_REQUIRED_META_KEYS - set(metadata.keys())
        if missing_meta_keys:
            raise ValueError(
                f"cache metadata {cache_path!r} is missing keys: {sorted(missing_meta_keys)!r}"
            )
        if all(key in expected_metadata for key in _PATCH_INDEX_CACHE_HASH_META_KEYS):
            missing_hash_keys = _PATCH_INDEX_CACHE_HASH_META_KEYS - set(metadata.keys())
            if missing_hash_keys:
                return None

        for key, expected_value in expected_metadata.items():
            if key not in metadata:
                return None
            if metadata[key] != expected_value:
                return None

        mode = str(metadata["mask_store_mode"])
        xyxys = np.asarray(npz_data["xyxys"], dtype=np.int64)
        if xyxys.ndim != 2 or xyxys.shape[1] != 4:
            raise ValueError(f"cached xyxys must have shape (N, 4), got {tuple(xyxys.shape)}")

        bbox_idx = np.asarray(npz_data["sample_bbox_indices"], dtype=np.int32).reshape(-1)
        if bbox_idx.shape[0] != xyxys.shape[0]:
            raise ValueError(
                f"cached sample_bbox_indices length {bbox_idx.shape[0]} does not match xyxys length {xyxys.shape[0]}"
            )

        if mode == "full":
            if xyxys.shape[0] > 0 and np.any(bbox_idx != -1):
                raise ValueError("cached mode='full' requires all sample_bbox_indices to be -1")
            mask_u8 = _label_mask_uint8(mask, context="patch-index cache label")
            mask_store = {"mode": "full", "shape": tuple(mask_u8.shape), "mask": mask_u8}
        elif mode == "bboxes":
            bboxes = np.asarray(npz_data["bboxes"], dtype=np.int32)
            if bboxes.ndim != 2 or bboxes.shape[1] != 4:
                raise ValueError(f"cached bboxes must have shape (N, 4), got {tuple(bboxes.shape)}")
            if xyxys.shape[0] > 0:
                if int(bboxes.shape[0]) <= 0:
                    raise ValueError("cached mode='bboxes' has patches but no bboxes")
                if np.any(bbox_idx < 0):
                    raise ValueError("cached mode='bboxes' requires non-negative sample_bbox_indices")
                max_idx = int(bbox_idx.max())
                if max_idx >= int(bboxes.shape[0]):
                    raise ValueError(
                        f"cached sample_bbox_indices max={max_idx} is out of range for {int(bboxes.shape[0])} bboxes"
                    )
            mask_store = _rebuild_mask_store_from_cached_bboxes(mask, bboxes)
        else:
            raise ValueError(f"unsupported cached mask_store_mode: {mode!r}")

    return mask_store, xyxys, bbox_idx


def _build_mask_store_and_patch_index_cached(
    mask,
    fragment_mask,
    *,
    fragment_id,
    split_name,
    filter_empty_tile,
    label_suffix,
    mask_suffix,
):
    if not bool(getattr(CFG, "dataset_cache_enabled", True)):
        return _build_mask_store_and_patch_index(
            mask,
            fragment_mask,
            filter_empty_tile=bool(filter_empty_tile),
        )

    params = _build_patch_index_cache_params(
        fragment_id=fragment_id,
        split_name=split_name,
        filter_empty_tile=filter_empty_tile,
        label_suffix=label_suffix,
        mask_suffix=mask_suffix,
    )
    check_hash = bool(getattr(CFG, "dataset_cache_check_hash", True))
    expected_metadata = _build_patch_index_expected_metadata(
        params,
        mask,
        fragment_mask,
        check_hash=check_hash,
    )
    cache_dir = _resolve_patch_index_cache_dir()
    cache_path = _patch_index_cache_file_path(cache_dir, params)
    cached = _load_patch_index_cache(
        cache_path=cache_path,
        expected_metadata=expected_metadata,
        mask=mask,
    )
    if cached is not None:
        log(f"patch-index cache hit split={split_name} segment={fragment_id} path={cache_path}")
        return cached

    log(f"patch-index cache miss split={split_name} segment={fragment_id} path={cache_path}")
    mask_store, xyxys, sample_bbox_indices = _build_mask_store_and_patch_index(
        mask,
        fragment_mask,
        filter_empty_tile=bool(filter_empty_tile),
    )
    _save_patch_index_cache(
        cache_path=cache_path,
        metadata=expected_metadata,
        mask_store=mask_store,
        xyxys=xyxys,
        sample_bbox_indices=sample_bbox_indices,
    )
    return mask_store, xyxys, sample_bbox_indices


def _label_foreground_mask(mask):
    mask_arr = np.asarray(mask)
    if mask_arr.ndim != 2:
        raise ValueError(f"expected 2D label mask, got shape={tuple(mask_arr.shape)}")
    if np.issubdtype(mask_arr.dtype, np.floating):
        return mask_arr > 0.0
    return mask_arr >= 3


def _init_uint8_stats_accumulator():
    return {
        "hist": np.zeros((256,), dtype=np.int64),
        "count": 0,
        "sum": 0.0,
        "sum_sq": 0.0,
    }


def _accumulate_uint8_values(stats_accumulator, values, *, context):
    values_arr = np.asarray(values)
    if values_arr.size == 0:
        return
    if values_arr.dtype != np.uint8:
        if np.issubdtype(values_arr.dtype, np.integer):
            values_min = int(values_arr.min())
            values_max = int(values_arr.max())
            if values_min < 0 or values_max > 255:
                raise ValueError(
                    f"{context}: expected integer values in [0, 255], "
                    f"got min={values_min}, max={values_max}"
                )
            values_arr = values_arr.astype(np.uint8, copy=False)
        else:
            raise TypeError(f"{context}: expected uint8/integer values, got {values_arr.dtype}")

    flat_u8 = values_arr.reshape(-1)
    stats_accumulator["hist"] += np.bincount(flat_u8, minlength=256).astype(np.int64, copy=False)

    flat_f64 = flat_u8.astype(np.float64, copy=False)
    stats_accumulator["count"] += int(flat_f64.size)
    stats_accumulator["sum"] += float(flat_f64.sum(dtype=np.float64))
    stats_accumulator["sum_sq"] += float(np.square(flat_f64, dtype=np.float64).sum(dtype=np.float64))


def _accumulate_image_foreground_uint8_stats(stats_accumulator, image, foreground_mask, *, context):
    image_arr = np.asarray(image)
    foreground = np.asarray(foreground_mask, dtype=bool)
    if image_arr.ndim != 3:
        raise ValueError(f"{context}: expected image shape (H, W, C), got {tuple(image_arr.shape)}")
    if foreground.ndim != 2:
        raise ValueError(f"{context}: expected foreground mask shape (H, W), got {tuple(foreground.shape)}")
    if image_arr.shape[:2] != foreground.shape:
        raise ValueError(
            f"{context}: image/mask shape mismatch image={tuple(image_arr.shape[:2])} "
            f"mask={tuple(foreground.shape)}"
        )

    chunk_h = max(1, int(getattr(CFG, "size", 256)))
    h = int(image_arr.shape[0])
    for y0 in range(0, h, chunk_h):
        y1 = min(h, y0 + chunk_h)
        fg_chunk = foreground[y0:y1]
        if not bool(fg_chunk.any()):
            continue
        values = image_arr[y0:y1][fg_chunk]
        _accumulate_uint8_values(stats_accumulator, values, context=context)


def _accumulate_volume_foreground_uint8_stats(stats_accumulator, volume, foreground_mask, *, context):
    foreground = np.asarray(foreground_mask, dtype=bool)
    if foreground.ndim != 2:
        raise ValueError(f"{context}: expected foreground mask shape (H, W), got {tuple(foreground.shape)}")

    h = int(foreground.shape[0])
    w = int(foreground.shape[1])
    chunk_size = max(1, int(getattr(CFG, "tile_size", 1024)))
    for y0 in range(0, h, chunk_size):
        y1 = min(h, y0 + chunk_size)
        for x0 in range(0, w, chunk_size):
            x1 = min(w, x0 + chunk_size)
            fg_chunk = foreground[y0:y1, x0:x1]
            if not bool(fg_chunk.any()):
                continue
            patch = volume.read_patch(y0, y1, x0, x1)
            if patch.shape[0] != fg_chunk.shape[0] or patch.shape[1] != fg_chunk.shape[1]:
                raise ValueError(
                    f"{context}: patch/foreground shape mismatch patch={tuple(patch.shape)} "
                    f"mask={tuple(fg_chunk.shape)}"
                )
            values = patch[fg_chunk]
            _accumulate_uint8_values(stats_accumulator, values, context=context)


def _percentile_from_histogram_uint8(histogram, percentile):
    q = float(percentile)
    if not (0.0 <= q <= 100.0):
        raise ValueError(f"percentile must be in [0, 100], got {q}")

    hist = np.asarray(histogram, dtype=np.int64).reshape(-1)
    if hist.shape[0] != 256:
        raise ValueError(f"expected histogram with 256 bins, got shape={tuple(hist.shape)}")
    total = int(hist.sum())
    if total <= 0:
        raise ValueError("cannot compute percentile from empty histogram")

    rank = (q / 100.0) * float(total - 1)
    lower_rank = int(math.floor(rank))
    upper_rank = int(math.ceil(rank))
    cdf = np.cumsum(hist, dtype=np.int64)
    lower_idx = int(np.searchsorted(cdf, lower_rank + 1, side="left"))
    upper_idx = int(np.searchsorted(cdf, upper_rank + 1, side="left"))
    if lower_rank == upper_rank:
        return float(lower_idx)
    alpha = float(rank - lower_rank)
    return float((1.0 - alpha) * lower_idx + alpha * upper_idx)


def _finalize_uint8_stats(stats_accumulator):
    total_count = int(stats_accumulator["count"])
    if total_count <= 0:
        raise ValueError("normalization stats require at least one foreground voxel")

    total_sum = float(stats_accumulator["sum"])
    total_sum_sq = float(stats_accumulator["sum_sq"])
    mean = total_sum / float(total_count)
    variance = (total_sum_sq / float(total_count)) - (mean * mean)
    if variance < 0 and abs(variance) < 1e-12:
        variance = 0.0
    if variance < 0:
        raise ValueError(f"computed negative variance: {variance}")
    std = float(np.sqrt(variance))
    if std <= 0:
        raise ValueError(f"computed non-positive std: {std}")

    p005 = _percentile_from_histogram_uint8(stats_accumulator["hist"], 0.5)
    p995 = _percentile_from_histogram_uint8(stats_accumulator["hist"], 99.5)
    if p995 < p005:
        raise ValueError(f"invalid percentile bounds: p0.5={p005}, p99.5={p995}")

    return {
        "percentile_00_5": float(p005),
        "percentile_99_5": float(p995),
        "mean": float(mean),
        "std": float(std),
        "num_voxels": int(total_count),
    }


def _maybe_prepare_fold_label_foreground_percentile_clip_zscore_stats(
    *,
    segments_metadata,
    train_fragment_ids,
    data_backend,
    train_label_suffix,
    train_mask_suffix,
    volume_cache,
):
    normalization_mode = str(getattr(CFG, "normalization_mode", "clip_max_div255")).strip().lower()
    if normalization_mode != "train_fold_fg_clip_zscore":
        CFG.fold_label_foreground_percentile_clip_zscore_stats = None
        return

    stats_accumulator = _init_uint8_stats_accumulator()
    segments_with_foreground = 0

    if data_backend == "zarr":
        for fragment_id in train_fragment_ids:
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

            label_mask, fragment_mask = read_label_and_fragment_mask_for_shape(
                sid,
                volume.shape[:2],
                label_suffix=train_label_suffix,
                mask_suffix=train_mask_suffix,
            )
            foreground = _label_foreground_mask(label_mask) & (np.asarray(fragment_mask) > 0)
            if not bool(foreground.any()):
                log(f"normalization stats: segment={sid} has no foreground label voxels; skipping")
                continue
            segments_with_foreground += 1
            _accumulate_volume_foreground_uint8_stats(
                stats_accumulator,
                volume,
                foreground,
                context=f"normalization stats segment={sid}",
            )
    elif data_backend == "tiff":
        for fragment_id in train_fragment_ids:
            seg_meta = _segment_meta(segments_metadata, fragment_id)
            layer_range = _segment_layer_range(seg_meta, fragment_id)
            reverse_layers = _segment_reverse_layers(seg_meta, fragment_id)
            image, label_mask, fragment_mask = read_image_mask(
                fragment_id,
                layer_range=layer_range,
                reverse_layers=reverse_layers,
                label_suffix=train_label_suffix,
                mask_suffix=train_mask_suffix,
            )
            foreground = _label_foreground_mask(label_mask) & (np.asarray(fragment_mask) > 0)
            if not bool(foreground.any()):
                log(f"normalization stats: segment={fragment_id} has no foreground label voxels; skipping")
                continue
            segments_with_foreground += 1
            _accumulate_image_foreground_uint8_stats(
                stats_accumulator,
                image,
                foreground,
                context=f"normalization stats segment={fragment_id}",
            )
    else:
        raise ValueError(f"Unknown training.data_backend: {data_backend!r}. Expected 'zarr' or 'tiff'.")

    if segments_with_foreground <= 0:
        raise ValueError(
            "normalization_mode='train_fold_fg_clip_zscore' requires at least one "
            "training segment with foreground label voxels"
        )

    stats = _finalize_uint8_stats(stats_accumulator)
    CFG.fold_label_foreground_percentile_clip_zscore_stats = stats
    log(
        "normalization stats "
        f"mode={normalization_mode} train_segments={len(train_fragment_ids)} "
        f"segments_with_foreground={segments_with_foreground} "
        f"num_voxels={stats['num_voxels']} "
        f"p0.5={stats['percentile_00_5']:.6f} "
        f"p99.5={stats['percentile_99_5']:.6f} "
        f"mean={stats['mean']:.6f} std={stats['std']:.6f}"
    )


def build_group_metadata(fragment_ids, segments_metadata, group_key):
    group_names, _group_name_to_idx, fragment_to_group_idx = build_group_mappings(
        fragment_ids,
        segments_metadata,
        group_key=group_key,
    )
    return group_names, fragment_to_group_idx


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

    return {
        "patch_count": patch_count,
        "images": frag_train_images,
        "masks": frag_train_masks,
        "group_idx": group_idx,
        "stitch_candidate": stitch_candidate,
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
    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.valid_batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    mask_border = None
    if include_train_xyxys:
        mask_border = _mask_border(
            _downsample_bool_mask_any(fragment_mask_val, int(getattr(CFG, "stitch_downsample", 1)))
        )
    mask_bbox = None
    if bool(getattr(CFG, "stitch_use_roi", False)):
        bboxes = _mask_component_bboxes_downsample(
            fragment_mask_val,
            int(getattr(CFG, "stitch_downsample", 1)),
        )
        if int(bboxes.shape[0]) > 0:
            mask_bbox = bboxes

    return {
        "patch_count": patch_count,
        "val_loader": val_loader,
        "mask_shape": tuple(mask_val.shape),
        "mask_border": mask_border,
        "mask_bbox": mask_bbox,
    }


def summarize_patch_counts(split_name, fragment_ids_list, counts_by_segment, *, group_names, fragment_to_group_idx):
    total = int(sum(int(counts_by_segment.get(fid, 0)) for fid in fragment_ids_list))
    counts_by_group = {name: 0 for name in group_names}
    for fid in fragment_ids_list:
        n = int(counts_by_segment.get(fid, 0))
        gidx = fragment_to_group_idx.get(fid, 0)
        gname = group_names[gidx] if gidx < len(group_names) else str(gidx)
        counts_by_group[gname] = int(counts_by_group.get(gname, 0)) + n

    log(f"{split_name} patch counts total={total}")
    for fid in fragment_ids_list:
        n = int(counts_by_segment.get(fid, 0))
        gidx = fragment_to_group_idx.get(fid, 0)
        gname = group_names[gidx] if gidx < len(group_names) else str(gidx)
        log(f"  {split_name} segment={fid} group={gname} patches={n}")
    log(f"{split_name} patch counts by group {counts_by_group}")


def build_train_loader(train_images, train_masks, train_groups, group_names, *, train_transform):
    train_dataset = CustomDataset(
        train_images,
        CFG,
        labels=train_masks,
        groups=train_groups,
        transform=train_transform,
    )
    return _build_train_loader_from_dataset(train_dataset, train_groups, group_names)


def _build_train_loader_from_dataset(train_dataset, train_groups, group_names):
    group_array = torch.as_tensor(train_groups, dtype=torch.long)
    group_counts = torch.bincount(group_array, minlength=len(group_names)).float()
    train_group_counts = [int(x) for x in group_counts.tolist()]
    log(f"train group counts {dict(zip(group_names, train_group_counts))}")

    if CFG.sampler == "shuffle":
        train_sampler = None
        train_shuffle = True
        train_batch_sampler = None
    elif CFG.sampler == "group_balanced":
        group_weights = len(train_dataset) / group_counts.clamp_min(1)
        weights = group_weights[group_array]
        train_sampler = WeightedRandomSampler(weights, len(train_dataset), replacement=True)
        train_shuffle = False
        train_batch_sampler = None
    elif CFG.sampler == "group_stratified":
        train_sampler = None
        train_shuffle = False
        epoch_size_mode = str(getattr(CFG, "group_stratified_epoch_size_mode", "dataset")).strip().lower()
        log(f"group_stratified sampler epoch_size_mode={epoch_size_mode!r}")
        train_batch_sampler = GroupStratifiedBatchSampler(
            train_groups,
            batch_size=CFG.train_batch_size,
            seed=getattr(CFG, "seed", 0),
            drop_last=True,
            epoch_size_mode=epoch_size_mode,
        )
    else:
        raise ValueError(f"Unknown training.sampler: {CFG.sampler!r}")

    if train_batch_sampler is not None:
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            num_workers=CFG.num_workers,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=CFG.train_batch_size,
            shuffle=train_shuffle,
            sampler=train_sampler,
            num_workers=CFG.num_workers,
            pin_memory=True,
            drop_last=True,
        )
    return train_loader, train_group_counts


def build_train_loader_lazy(
    train_volumes_by_segment,
    train_masks_by_segment,
    train_xyxys_by_segment,
    train_sample_bbox_indices_by_segment,
    train_groups_by_segment,
    group_names,
    *,
    train_transform,
):
    train_dataset = LazyZarrTrainDataset(
        train_volumes_by_segment,
        train_masks_by_segment,
        train_xyxys_by_segment,
        train_groups_by_segment,
        CFG,
        transform=train_transform,
        sample_bbox_indices_by_segment=train_sample_bbox_indices_by_segment,
    )
    train_groups = [int(x) for x in train_dataset.sample_groups.tolist()]
    return _build_train_loader_from_dataset(train_dataset, train_groups, group_names)


def log_training_budget(train_loader):
    steps_per_epoch = len(train_loader)
    accum = int(getattr(CFG, "accumulate_grad_batches", 1) or 1)
    if accum > 1:
        steps_per_epoch = int(math.ceil(steps_per_epoch / accum))

    micro_steps_per_epoch = int(len(train_loader))
    optimizer_steps_per_epoch = int(steps_per_epoch)
    total_optimizer_steps = int(optimizer_steps_per_epoch * int(CFG.epochs))
    effective_batch_size = int(int(CFG.train_batch_size) * int(accum))

    log(
        "train budget "
        f"len(train_loader)={micro_steps_per_epoch} accumulate_grad_batches={accum} "
        f"optimizer_steps_per_epoch={optimizer_steps_per_epoch} epochs={int(CFG.epochs)} "
        f"total_optimizer_steps={total_optimizer_steps} effective_batch_size={effective_batch_size}"
    )
    log(
        "scheduler budget "
        f"scheduler={getattr(CFG, 'scheduler', None)!r} "
        f"onecycle steps_per_epoch={optimizer_steps_per_epoch} epochs={int(CFG.epochs)} "
        f"max_lr={float(CFG.lr)} div_factor={float(getattr(CFG, 'onecycle_div_factor', 25.0))} "
        f"pct_start={float(getattr(CFG, 'onecycle_pct_start', 0.15))}"
    )
    return steps_per_epoch


def build_train_stitch_loaders(train_fragment_ids, train_stitch_candidates, stitch_segment_id, *, valid_transform):
    train_stitch_loaders = []
    train_stitch_shapes = []
    train_stitch_segment_ids = []
    if not bool(getattr(CFG, "stitch_train", False)):
        return train_stitch_loaders, train_stitch_shapes, train_stitch_segment_ids

    requested_ids = _resolve_requested_train_stitch_ids(
        train_fragment_ids,
        train_stitch_candidates.keys(),
        stitch_segment_id,
    )

    for segment_id in requested_ids:
        entry = train_stitch_candidates.get(str(segment_id))
        if entry is None:
            continue
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
        train_loader_viz = DataLoader(
            train_dataset_viz,
            batch_size=CFG.valid_batch_size,
            shuffle=False,
            num_workers=CFG.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        train_stitch_loaders.append(train_loader_viz)
        train_stitch_shapes.append(tuple(seg_shape))
        train_stitch_segment_ids.append(str(segment_id))

    return train_stitch_loaders, train_stitch_shapes, train_stitch_segment_ids


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
    train_stitch_loaders = []
    train_stitch_shapes = []
    train_stitch_segment_ids = []
    if not bool(getattr(CFG, "stitch_train", False)):
        return train_stitch_loaders, train_stitch_shapes, train_stitch_segment_ids

    requested_ids = _resolve_requested_train_stitch_ids(
        train_fragment_ids,
        train_xyxys_by_segment.keys(),
        stitch_segment_id,
    )

    for segment_id in requested_ids:
        sid = str(segment_id)
        xy = train_xyxys_by_segment.get(sid)
        if xy is None or int(len(xy)) == 0:
            continue
        if sid not in train_volumes_by_segment or sid not in train_masks_by_segment:
            continue
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
        loader = DataLoader(
            dataset,
            batch_size=CFG.valid_batch_size,
            shuffle=False,
            num_workers=CFG.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        train_stitch_loaders.append(loader)
        train_stitch_shapes.append(tuple(_mask_store_shape(train_masks_by_segment[sid])))
        train_stitch_segment_ids.append(sid)

    return train_stitch_loaders, train_stitch_shapes, train_stitch_segment_ids


def build_log_only_stitch_loaders(
    log_only_segments,
    *,
    segments_metadata,
    layers_cache,
    valid_transform,
    mask_suffix,
    log_only_downsample,
):
    log_only_loaders = []
    log_only_shapes = []
    log_only_segment_ids = []
    log_only_bboxes = {}

    for fragment_id in log_only_segments:
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
            continue

        xyxys = np.stack(xyxys) if len(xyxys) > 0 else np.zeros((0, 4), dtype=np.int64)
        dataset = CustomDatasetTest(images, xyxys, CFG, transform=valid_transform)
        loader = DataLoader(
            dataset,
            batch_size=CFG.valid_batch_size,
            shuffle=False,
            num_workers=CFG.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        log_only_loaders.append(loader)
        log_only_shapes.append(tuple(fragment_mask.shape))
        log_only_segment_ids.append(str(fragment_id))

        bboxes = _mask_component_bboxes_downsample(fragment_mask, int(log_only_downsample))
        if int(bboxes.shape[0]) > 0:
            log_only_bboxes[str(fragment_id)] = bboxes

    return log_only_loaders, log_only_shapes, log_only_segment_ids, log_only_bboxes


def build_log_only_stitch_loaders_lazy(
    log_only_segments,
    *,
    segments_metadata,
    volume_cache,
    valid_transform,
    mask_suffix,
    log_only_downsample,
):
    log_only_loaders = []
    log_only_shapes = []
    log_only_segment_ids = []
    log_only_bboxes = {}

    for fragment_id in log_only_segments:
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
        xyxys = extract_patch_coordinates(None, fragment_mask, filter_empty_tile=False)
        log(f"patches log-only segment={sid} n={int(len(xyxys))}")
        if int(len(xyxys)) == 0:
            continue

        dataset = LazyZarrXyOnlyDataset(
            {sid: volume},
            {sid: xyxys},
            CFG,
            transform=valid_transform,
        )
        loader = DataLoader(
            dataset,
            batch_size=CFG.valid_batch_size,
            shuffle=False,
            num_workers=CFG.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        log_only_loaders.append(loader)
        log_only_shapes.append(tuple(fragment_mask.shape))
        log_only_segment_ids.append(sid)

        bboxes = _mask_component_bboxes_downsample(fragment_mask, int(log_only_downsample))
        if int(bboxes.shape[0]) > 0:
            log_only_bboxes[sid] = bboxes

    return log_only_loaders, log_only_shapes, log_only_segment_ids, log_only_bboxes


def build_datasets(run_state):
    segments_metadata = run_state["segments_metadata"]
    fragment_ids = run_state["fragment_ids"]
    train_fragment_ids = run_state["train_fragment_ids"]
    val_fragment_ids = run_state["val_fragment_ids"]
    group_key = run_state["group_key"]

    group_names, fragment_to_group_idx = build_group_metadata(
        fragment_ids,
        segments_metadata,
        group_key,
    )
    group_idx_by_segment = {str(fragment_id): int(group_idx) for fragment_id, group_idx in fragment_to_group_idx.items()}

    train_label_suffix = getattr(CFG, "train_label_suffix", "")
    train_mask_suffix = getattr(CFG, "train_mask_suffix", "")
    val_label_suffix = getattr(CFG, "val_label_suffix", "_val")
    val_mask_suffix = getattr(CFG, "val_mask_suffix", "_val")
    cv_fold = getattr(CFG, "cv_fold", None)
    log(
        "label/mask suffixes "
        f"cv_fold={cv_fold!r} "
        f"train=(label={train_label_suffix!r}, mask={train_mask_suffix!r}) "
        f"val=(label={val_label_suffix!r}, mask={val_mask_suffix!r})"
    )

    data_backend = str(getattr(CFG, "data_backend", "zarr")).strip().lower()
    if data_backend not in {"zarr", "tiff"}:
        raise ValueError(f"Unknown training.data_backend: {data_backend!r}. Expected 'zarr' or 'tiff'.")
    log(f"data backend={data_backend}")
    if bool(getattr(CFG, "dataset_cache_enabled", True)) and not bool(getattr(CFG, "dataset_cache_check_hash", True)):
        log("WARNING: dataset cache hash validation is disabled (metadata.training.dataset_cache_check_hash=false)")
    shared_volume_cache = {}
    _maybe_prepare_fold_label_foreground_percentile_clip_zscore_stats(
        segments_metadata=segments_metadata,
        train_fragment_ids=train_fragment_ids,
        data_backend=data_backend,
        train_label_suffix=train_label_suffix,
        train_mask_suffix=train_mask_suffix,
        volume_cache=shared_volume_cache,
    )

    train_transform = get_transforms(data="train", cfg=CFG)
    valid_transform = get_transforms(data="valid", cfg=CFG)

    if data_backend == "zarr":
        train_patch_counts_by_segment = {}
        train_mask_borders = {}
        train_mask_bboxes = {}
        train_groups_by_segment = {str(fid): int(fragment_to_group_idx[fid]) for fid in train_fragment_ids}
        train_volumes_by_segment = {}
        train_masks_by_segment = {}
        train_xyxys_by_segment = {}
        train_sample_bbox_indices_by_segment = {}

        val_loaders = []
        val_stitch_shapes = []
        val_stitch_segment_ids = []
        val_patch_counts_by_segment = {}
        val_mask_borders = {}
        val_mask_bboxes = {}
        stitch_val_dataloader_idx = None
        stitch_pred_shape = None
        stitch_segment_id = None

        log("building datasets (zarr lazy)")
        include_train_xyxys = bool(getattr(CFG, "stitch_train", False))
        volume_cache = shared_volume_cache

        for fragment_id in train_fragment_ids:
            sid = str(fragment_id)
            seg_meta = _segment_meta(segments_metadata, fragment_id)
            layer_range = _segment_layer_range(seg_meta, fragment_id)
            reverse_layers = _segment_reverse_layers(seg_meta, fragment_id)
            group_idx = int(fragment_to_group_idx[fragment_id])
            group_name = group_names[group_idx] if group_idx < len(group_names) else str(group_idx)

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
                label_suffix=train_label_suffix,
                mask_suffix=train_mask_suffix,
            )
            mask_store, xyxys, sample_bbox_indices = _build_mask_store_and_patch_index_cached(
                mask,
                fragment_mask,
                fragment_id=sid,
                split_name="train",
                filter_empty_tile=True,
                label_suffix=train_label_suffix,
                mask_suffix=train_mask_suffix,
            )
            patch_count = int(len(xyxys))
            train_patch_counts_by_segment[fragment_id] = patch_count
            log(
                f"loaded train segment={sid} image={tuple(volume.shape)} label={tuple(mask.shape)} "
                f"mask={tuple(fragment_mask.shape)} patches={patch_count} in {time.time() - t0:.1f}s"
            )

            if patch_count > 0:
                train_volumes_by_segment[sid] = volume
                train_masks_by_segment[sid] = mask_store
                train_xyxys_by_segment[sid] = xyxys
                train_sample_bbox_indices_by_segment[sid] = sample_bbox_indices

            if include_train_xyxys:
                train_mask_borders[sid] = _mask_border(
                    _downsample_bool_mask_any(fragment_mask, int(getattr(CFG, "stitch_downsample", 1)))
                )
            if bool(getattr(CFG, "stitch_use_roi", False)):
                bboxes = _mask_component_bboxes_downsample(
                    fragment_mask,
                    int(getattr(CFG, "stitch_downsample", 1)),
                )
                if int(bboxes.shape[0]) > 0:
                    train_mask_bboxes[sid] = bboxes

        for fragment_id in val_fragment_ids:
            sid = str(fragment_id)
            seg_meta = _segment_meta(segments_metadata, fragment_id)
            layer_range = _segment_layer_range(seg_meta, fragment_id)
            reverse_layers = _segment_reverse_layers(seg_meta, fragment_id)
            group_idx = int(fragment_to_group_idx[fragment_id])
            group_name = group_names[group_idx] if group_idx < len(group_names) else str(group_idx)

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
                label_suffix=val_label_suffix,
                mask_suffix=val_mask_suffix,
            )
            mask_store_val, val_xyxys, val_sample_bbox_indices = _build_mask_store_and_patch_index_cached(
                mask_val,
                fragment_mask_val,
                fragment_id=sid,
                split_name="val",
                filter_empty_tile=False,
                label_suffix=val_label_suffix,
                mask_suffix=val_mask_suffix,
            )
            patch_count = int(len(val_xyxys))
            val_patch_counts_by_segment[fragment_id] = patch_count
            log(
                f"loaded val segment={sid} image={tuple(volume.shape)} label={tuple(mask_val.shape)} "
                f"mask={tuple(fragment_mask_val.shape)} patches={patch_count} in {time.time() - t0:.1f}s"
            )
            if patch_count == 0:
                continue

            val_dataset = LazyZarrXyLabelDataset(
                {sid: volume},
                {sid: mask_store_val},
                {sid: val_xyxys},
                {sid: group_idx},
                CFG,
                transform=valid_transform,
                sample_bbox_indices_by_segment={sid: val_sample_bbox_indices},
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=CFG.valid_batch_size,
                shuffle=False,
                num_workers=CFG.num_workers,
                pin_memory=True,
                drop_last=False,
            )
            val_loaders.append(val_loader)
            mask_val_shape = tuple(_mask_store_shape(mask_store_val))
            val_stitch_shapes.append(mask_val_shape)
            val_stitch_segment_ids.append(fragment_id)

            if include_train_xyxys:
                val_mask_borders[sid] = _mask_border(
                    _downsample_bool_mask_any(fragment_mask_val, int(getattr(CFG, "stitch_downsample", 1)))
                )
            if bool(getattr(CFG, "stitch_use_roi", False)):
                bboxes = _mask_component_bboxes_downsample(
                    fragment_mask_val,
                    int(getattr(CFG, "stitch_downsample", 1)),
                )
                if int(bboxes.shape[0]) > 0:
                    val_mask_bboxes[sid] = bboxes

            if fragment_id == CFG.valid_id:
                stitch_val_dataloader_idx = len(val_loaders) - 1
                stitch_pred_shape = mask_val_shape
                stitch_segment_id = fragment_id

        summarize_patch_counts(
            "train",
            train_fragment_ids,
            train_patch_counts_by_segment,
            group_names=group_names,
            fragment_to_group_idx=fragment_to_group_idx,
        )
        summarize_patch_counts(
            "val",
            val_fragment_ids,
            val_patch_counts_by_segment,
            group_names=group_names,
            fragment_to_group_idx=fragment_to_group_idx,
        )

        train_patches_total = int(sum(int(v) for v in train_patch_counts_by_segment.values()))
        log(f"dataset built (zarr) train_patches={train_patches_total} val_loaders={len(val_loaders)}")
        if train_patches_total == 0:
            raise ValueError("No training data was built (all segments produced 0 training patches).")
        if len(val_loaders) == 0:
            raise ValueError("No validation data was built (all segments produced 0 validation patches).")

        train_loader, train_group_counts = build_train_loader_lazy(
            train_volumes_by_segment,
            train_masks_by_segment,
            train_xyxys_by_segment,
            train_sample_bbox_indices_by_segment,
            train_groups_by_segment,
            group_names,
            train_transform=train_transform,
        )
        steps_per_epoch = log_training_budget(train_loader)

        train_stitch_loaders, train_stitch_shapes, train_stitch_segment_ids = build_train_stitch_loaders_lazy(
            train_fragment_ids,
            train_volumes_by_segment,
            train_masks_by_segment,
            train_xyxys_by_segment,
            train_sample_bbox_indices_by_segment,
            train_groups_by_segment,
            stitch_segment_id,
            valid_transform=valid_transform,
        )

        log_only_segments = list(getattr(CFG, "stitch_log_only_segments", []) or [])
        log_only_loaders = []
        log_only_shapes = []
        log_only_segment_ids = []
        log_only_bboxes = {}
        if log_only_segments:
            log_only_loaders, log_only_shapes, log_only_segment_ids, log_only_bboxes = build_log_only_stitch_loaders_lazy(
                log_only_segments,
                segments_metadata=segments_metadata,
                volume_cache=volume_cache,
                valid_transform=valid_transform,
                mask_suffix=val_mask_suffix,
                log_only_downsample=int(getattr(CFG, "stitch_log_only_downsample", getattr(CFG, "stitch_downsample", 1))),
            )

        return {
            "train_loader": train_loader,
            "val_loaders": val_loaders,
            "group_names": group_names,
            "group_idx_by_segment": group_idx_by_segment,
            "train_group_counts": train_group_counts,
            "steps_per_epoch": steps_per_epoch,
            "train_stitch_loaders": train_stitch_loaders,
            "train_stitch_shapes": train_stitch_shapes,
            "train_stitch_segment_ids": train_stitch_segment_ids,
            "train_mask_borders": train_mask_borders,
            "train_mask_bboxes": train_mask_bboxes,
            "val_mask_borders": val_mask_borders,
            "val_mask_bboxes": val_mask_bboxes,
            "log_only_stitch_loaders": log_only_loaders,
            "log_only_stitch_shapes": log_only_shapes,
            "log_only_stitch_segment_ids": log_only_segment_ids,
            "log_only_mask_bboxes": log_only_bboxes,
            "include_train_xyxys": include_train_xyxys,
            "stitch_val_dataloader_idx": stitch_val_dataloader_idx,
            "stitch_pred_shape": stitch_pred_shape,
            "stitch_segment_id": stitch_segment_id,
            "val_stitch_shapes": val_stitch_shapes,
            "val_stitch_segment_ids": val_stitch_segment_ids,
        }

    train_images = []
    train_masks = []
    train_groups = []
    train_patch_counts_by_segment = {}
    train_stitch_candidates = {}
    train_mask_borders = {}
    train_mask_bboxes = {}

    val_loaders = []
    val_stitch_shapes = []
    val_stitch_segment_ids = []
    val_patch_counts_by_segment = {}
    val_mask_borders = {}
    val_mask_bboxes = {}
    stitch_val_dataloader_idx = None
    stitch_pred_shape = None
    stitch_segment_id = None

    log("building datasets")
    train_set = set(train_fragment_ids)
    val_set = set(val_fragment_ids)
    overlap_segments = train_set & val_set
    layers_cache = {}
    include_train_xyxys = bool(getattr(CFG, "stitch_train", False))

    for fragment_id in train_fragment_ids:
        seg_meta = _segment_meta(segments_metadata, fragment_id)
        group_idx = fragment_to_group_idx[fragment_id]
        group_name = group_names[group_idx] if group_idx < len(group_names) else str(group_idx)

        result = load_train_segment(
            fragment_id,
            seg_meta,
            group_idx,
            group_name,
            overlap_segments=overlap_segments,
            layers_cache=layers_cache,
            include_train_xyxys=include_train_xyxys,
            label_suffix=train_label_suffix,
            mask_suffix=train_mask_suffix,
        )

        train_patch_counts_by_segment[fragment_id] = result["patch_count"]
        if result["stitch_candidate"] is not None:
            train_stitch_candidates[str(fragment_id)] = result["stitch_candidate"]
        if result["mask_border"] is not None:
            train_mask_borders[str(fragment_id)] = result["mask_border"]
        if result.get("mask_bbox") is not None:
            train_mask_bboxes[str(fragment_id)] = result["mask_bbox"]
        train_images.extend(result["images"])
        train_masks.extend(result["masks"])
        train_groups.extend([group_idx] * len(result["images"]))

    for fragment_id in val_fragment_ids:
        seg_meta = _segment_meta(segments_metadata, fragment_id)
        group_idx = fragment_to_group_idx[fragment_id]
        group_name = group_names[group_idx] if group_idx < len(group_names) else str(group_idx)

        result = load_val_segment(
            fragment_id,
            seg_meta,
            group_idx,
            group_name,
            layers_cache=layers_cache,
            include_train_xyxys=include_train_xyxys,
            valid_transform=valid_transform,
            label_suffix=val_label_suffix,
            mask_suffix=val_mask_suffix,
        )

        val_patch_counts_by_segment[fragment_id] = result["patch_count"]
        if result["val_loader"] is None:
            continue

        val_loaders.append(result["val_loader"])
        val_stitch_shapes.append(result["mask_shape"])
        val_stitch_segment_ids.append(fragment_id)
        if result["mask_border"] is not None:
            val_mask_borders[str(fragment_id)] = result["mask_border"]
        if result.get("mask_bbox") is not None:
            val_mask_bboxes[str(fragment_id)] = result["mask_bbox"]
        if fragment_id == CFG.valid_id:
            stitch_val_dataloader_idx = len(val_loaders) - 1
            stitch_pred_shape = result["mask_shape"]
            stitch_segment_id = fragment_id

    summarize_patch_counts(
        "train",
        train_fragment_ids,
        train_patch_counts_by_segment,
        group_names=group_names,
        fragment_to_group_idx=fragment_to_group_idx,
    )
    summarize_patch_counts(
        "val",
        val_fragment_ids,
        val_patch_counts_by_segment,
        group_names=group_names,
        fragment_to_group_idx=fragment_to_group_idx,
    )

    log(f"dataset built train_patches={len(train_images)} val_loaders={len(val_loaders)}")
    if len(val_loaders) == 0:
        raise ValueError("No validation data was built (all segments produced 0 validation patches).")

    train_loader, train_group_counts = build_train_loader(
        train_images,
        train_masks,
        train_groups,
        group_names,
        train_transform=train_transform,
    )
    steps_per_epoch = log_training_budget(train_loader)

    train_stitch_loaders, train_stitch_shapes, train_stitch_segment_ids = build_train_stitch_loaders(
        train_fragment_ids,
        train_stitch_candidates,
        stitch_segment_id,
        valid_transform=valid_transform,
    )

    log_only_segments = list(getattr(CFG, "stitch_log_only_segments", []) or [])
    log_only_loaders = []
    log_only_shapes = []
    log_only_segment_ids = []
    log_only_bboxes = {}
    if log_only_segments:
        log_only_loaders, log_only_shapes, log_only_segment_ids, log_only_bboxes = build_log_only_stitch_loaders(
            log_only_segments,
            segments_metadata=segments_metadata,
            layers_cache=layers_cache,
            valid_transform=valid_transform,
            mask_suffix=val_mask_suffix,
            log_only_downsample=int(getattr(CFG, "stitch_log_only_downsample", getattr(CFG, "stitch_downsample", 1))),
        )

    return {
        "train_loader": train_loader,
        "val_loaders": val_loaders,
        "group_names": group_names,
        "group_idx_by_segment": group_idx_by_segment,
        "train_group_counts": train_group_counts,
        "steps_per_epoch": steps_per_epoch,
        "train_stitch_loaders": train_stitch_loaders,
        "train_stitch_shapes": train_stitch_shapes,
        "train_stitch_segment_ids": train_stitch_segment_ids,
        "train_mask_borders": train_mask_borders,
        "train_mask_bboxes": train_mask_bboxes,
        "val_mask_borders": val_mask_borders,
        "val_mask_bboxes": val_mask_bboxes,
        "log_only_stitch_loaders": log_only_loaders,
        "log_only_stitch_shapes": log_only_shapes,
        "log_only_stitch_segment_ids": log_only_segment_ids,
        "log_only_mask_bboxes": log_only_bboxes,
        "include_train_xyxys": include_train_xyxys,
        "stitch_val_dataloader_idx": stitch_val_dataloader_idx,
        "stitch_pred_shape": stitch_pred_shape,
        "stitch_segment_id": stitch_segment_id,
        "val_stitch_shapes": val_stitch_shapes,
        "val_stitch_segment_ids": val_stitch_segment_ids,
    }
