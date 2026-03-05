import math

import numpy as np

from train_resnet3d_lib.config import CFG, log
from train_resnet3d_lib.data.image_readers import (
    get_segment_layer_range as _segment_layer_range,
    get_segment_meta as _segment_meta,
    get_segment_reverse_layers as _segment_reverse_layers,
    read_image_mask,
    read_label_and_fragment_mask_for_shape,
)
from train_resnet3d_lib.data.zarr_volume import ZarrSegmentVolume


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


def _weighted_percentile(values, weights, percentile):
    q = float(percentile)
    if not (0.0 <= q <= 100.0):
        raise ValueError(f"percentile must be in [0, 100], got {q}")

    values_arr = np.asarray(values, dtype=np.float64).reshape(-1)
    weights_arr = np.asarray(weights, dtype=np.int64).reshape(-1)
    if values_arr.shape != weights_arr.shape:
        raise ValueError(
            f"values/weights shape mismatch values={tuple(values_arr.shape)} "
            f"weights={tuple(weights_arr.shape)}"
        )
    if values_arr.size <= 0:
        raise ValueError("cannot compute percentile from empty values")
    if np.any(weights_arr < 0):
        raise ValueError("weights must be non-negative")
    total = int(weights_arr.sum())
    if total <= 0:
        raise ValueError("cannot compute percentile from empty histogram")

    order = np.argsort(values_arr, kind="mergesort")
    sorted_values = values_arr[order]
    sorted_weights = weights_arr[order]
    cdf = np.cumsum(sorted_weights, dtype=np.int64)
    rank = (q / 100.0) * float(total - 1)
    lower_rank = int(math.floor(rank))
    upper_rank = int(math.ceil(rank))
    lower_idx = int(np.searchsorted(cdf, lower_rank + 1, side="left"))
    upper_idx = int(np.searchsorted(cdf, upper_rank + 1, side="left"))
    lower_value = float(sorted_values[lower_idx])
    upper_value = float(sorted_values[upper_idx])
    if lower_rank == upper_rank:
        return lower_value
    alpha = float(rank - lower_rank)
    return float((1.0 - alpha) * lower_value + alpha * upper_value)


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
    intensity_values = np.arange(256, dtype=np.float64)
    return _weighted_percentile(intensity_values, hist, q)


def _median_and_mad_from_histogram_uint8(histogram):
    hist = np.asarray(histogram, dtype=np.int64).reshape(-1)
    if hist.shape[0] != 256:
        raise ValueError(f"expected histogram with 256 bins, got shape={tuple(hist.shape)}")
    total = int(hist.sum())
    if total <= 0:
        raise ValueError("cannot compute median/MAD from empty histogram")

    intensity_values = np.arange(256, dtype=np.float64)
    median = _weighted_percentile(intensity_values, hist, 50.0)
    abs_deviation_values = np.abs(intensity_values - median)
    mad = _weighted_percentile(abs_deviation_values, hist, 50.0)
    return float(median), float(mad)


def _finalize_uint8_stats(stats_accumulator, *, normalization_mode):
    total_count = int(stats_accumulator["count"])
    if total_count <= 0:
        raise ValueError("normalization stats require at least one foreground voxel")

    p005 = _percentile_from_histogram_uint8(stats_accumulator["hist"], 0.5)
    p995 = _percentile_from_histogram_uint8(stats_accumulator["hist"], 99.5)
    if p995 < p005:
        raise ValueError(f"invalid percentile bounds: p0.5={p005}, p99.5={p995}")

    if normalization_mode == "train_fold_fg_clip_zscore":
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
        return {
            "percentile_00_5": float(p005),
            "percentile_99_5": float(p995),
            "mean": float(mean),
            "std": float(std),
            "num_voxels": int(total_count),
        }

    if normalization_mode == "train_fold_fg_clip_robust_zscore":
        median, mad = _median_and_mad_from_histogram_uint8(stats_accumulator["hist"])
        robust_scale = 1.4826 * mad
        if robust_scale <= 0:
            raise ValueError(f"computed non-positive robust scale from MAD: {robust_scale}")
        return {
            "percentile_00_5": float(p005),
            "percentile_99_5": float(p995),
            "median": float(median),
            "mad": float(mad),
            "robust_scale": float(robust_scale),
            "num_voxels": int(total_count),
        }

    raise ValueError(f"Unsupported normalization_mode for stats finalization: {normalization_mode!r}")


def prepare_fold_label_foreground_percentile_clip_zscore_stats(
    *,
    segments_metadata,
    train_fragment_ids,
    data_backend,
    train_label_suffix,
    train_mask_suffix,
    volume_cache,
):
    normalization_mode = str(getattr(CFG, "normalization_mode", "clip_max_div255")).strip().lower()
    if normalization_mode not in {
        "train_fold_fg_clip_zscore",
        "train_fold_fg_clip_robust_zscore",
    }:
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
            f"normalization_mode={normalization_mode!r} requires at least one "
            "training segment with foreground label voxels"
        )

    stats = _finalize_uint8_stats(stats_accumulator, normalization_mode=normalization_mode)
    CFG.fold_label_foreground_percentile_clip_zscore_stats = stats
    if normalization_mode == "train_fold_fg_clip_zscore":
        stats_summary = f"mean={stats['mean']:.6f} std={stats['std']:.6f}"
    elif normalization_mode == "train_fold_fg_clip_robust_zscore":
        stats_summary = (
            f"median={stats['median']:.6f} mad={stats['mad']:.6f} "
            f"robust_scale={stats['robust_scale']:.6f}"
        )
    else:
        raise ValueError(f"Unsupported normalization_mode: {normalization_mode!r}")
    log(
        "normalization stats "
        f"mode={normalization_mode} train_segments={len(train_fragment_ids)} "
        f"segments_with_foreground={segments_with_foreground} "
        f"num_voxels={stats['num_voxels']} "
        f"p0.5={stats['percentile_00_5']:.6f} "
        f"p99.5={stats['percentile_99_5']:.6f} "
        f"{stats_summary}"
    )
