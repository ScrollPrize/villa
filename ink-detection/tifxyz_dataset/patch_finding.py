import hashlib
import json
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product

import numpy as np
from tqdm.auto import tqdm

import vesuvius.tifxyz as tifxyz
from .common import load_segment_label_masks, open_zarr
from vesuvius.neural_tracing.datasets.common import (
    _parse_z_range,
    _segment_overlaps_z_range,
)


DEFAULT_STORED_GRID_PAD = 40


def _empty_points():
    return np.zeros((0, 3), dtype=np.float32)


def _scale_pair_or_default(segment):
    segment.use_stored_resolution()
    scale_y, scale_x = getattr(segment, "_scale", (1.0, 1.0))
    scale_y = float(scale_y) if np.isfinite(scale_y) and float(scale_y) > 0.0 else 1.0
    scale_x = float(scale_x) if np.isfinite(scale_x) and float(scale_x) > 0.0 else 1.0
    return scale_y, scale_x


def _points_from_mask(z_grid, y_grid, x_grid, mask, retarget_factor):
    if not bool(np.any(mask)):
        return _empty_points()
    points = np.stack(
        [z_grid[mask], y_grid[mask], x_grid[mask]],
        axis=-1,
    ).astype(np.float32, copy=False)
    if retarget_factor != 1:
        points /= float(retarget_factor)
    return points


def _inclusive_bbox_from_point_sets(point_sets):
    mins = []
    maxs = []
    for points in point_sets:
        if points.size == 0:
            continue
        mins.append(np.min(points, axis=0))
        maxs.append(np.max(points, axis=0))
    if not mins:
        return None

    overall_min = np.min(np.stack(mins, axis=0), axis=0)
    overall_max = np.max(np.stack(maxs, axis=0), axis=0)
    min_zyx = np.floor(overall_min).astype(np.int64, copy=False)
    max_zyx = np.floor(overall_max).astype(np.int64, copy=False)
    max_zyx = np.maximum(max_zyx, min_zyx)
    return (
        int(min_zyx[0]),
        int(max_zyx[0]),
        int(min_zyx[1]),
        int(max_zyx[1]),
        int(min_zyx[2]),
        int(max_zyx[2]),
    )


def _axis_patch_starts(axis_min, axis_max, patch_size, overlap_fraction):
    axis_min = int(axis_min)
    axis_max = int(axis_max)
    patch_size = int(patch_size)
    overlap_fraction = float(overlap_fraction)
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}")

    last_start = axis_max - patch_size + 1
    if last_start <= axis_min:
        return [axis_min]

    stride = max(1, int(round(float(patch_size) * (1.0 - overlap_fraction))))
    starts = []
    current = axis_min
    while current < last_start:
        starts.append(int(current))
        current += stride
    if not starts or starts[-1] != int(last_start):
        starts.append(int(last_start))
    return starts


def _generate_sliding_bboxes(world_bbox, patch_size_zyx, overlap_fraction):
    z0, z1, y0, y1, x0, x1 = (int(v) for v in world_bbox)
    patch_size_zyx = np.asarray(patch_size_zyx, dtype=np.int64).reshape(3)

    z_starts = _axis_patch_starts(z0, z1, int(patch_size_zyx[0]), overlap_fraction)
    y_starts = _axis_patch_starts(y0, y1, int(patch_size_zyx[1]), overlap_fraction)
    x_starts = _axis_patch_starts(x0, x1, int(patch_size_zyx[2]), overlap_fraction)

    bboxes = []
    for start_z, start_y, start_x in product(z_starts, y_starts, x_starts):
        bboxes.append(
            (
                int(start_z),
                int(start_z + int(patch_size_zyx[0]) - 1),
                int(start_y),
                int(start_y + int(patch_size_zyx[1]) - 1),
                int(start_x),
                int(start_x + int(patch_size_zyx[2]) - 1),
            )
        )
    return bboxes


def _points_in_world_bbox(points_zyx, world_bbox):
    if points_zyx.size == 0:
        return np.zeros((0,), dtype=bool)
    z0, z1, y0, y1, x0, x1 = (float(v) for v in world_bbox)
    return (
        (points_zyx[:, 0] >= z0)
        & (points_zyx[:, 0] < z1 + 1.0)
        & (points_zyx[:, 1] >= y0)
        & (points_zyx[:, 1] < y1 + 1.0)
        & (points_zyx[:, 2] >= x0)
        & (points_zyx[:, 2] < x1 + 1.0)
    )


def _slice_bounds_from_rows_cols(rows, cols, shape, pad):
    row_count, col_count = (int(shape[0]), int(shape[1]))
    pad = max(0, int(pad))
    row_start = max(0, int(np.min(rows)) - pad)
    row_stop = min(row_count, int(np.max(rows)) + pad + 1)
    col_start = max(0, int(np.min(cols)) - pad)
    col_stop = min(col_count, int(np.max(cols)) + pad + 1)
    return (row_start, row_stop, col_start, col_stop)


def _build_patch_segment_entry(segment_record, world_bbox, stored_grid_pad):
    valid_in_bbox = _points_in_world_bbox(segment_record["valid_points_zyx"], world_bbox)
    valid_count = int(np.count_nonzero(valid_in_bbox))
    if valid_count == 0:
        return None

    rows = segment_record["valid_rows"][valid_in_bbox]
    cols = segment_record["valid_cols"][valid_in_bbox]
    stored_rowcol_bounds = _slice_bounds_from_rows_cols(
        rows,
        cols,
        segment_record["grid_shape"],
        stored_grid_pad,
    )

    positive_in_bbox = _points_in_world_bbox(segment_record["positive_points_zyx"], world_bbox)
    positive_count = int(np.count_nonzero(positive_in_bbox))
    return {
        "segment_idx": int(segment_record["segment_idx"]),
        "segment_uuid": str(segment_record["segment_uuid"]),
        "segment": segment_record["segment"],
        "ink_label_path": segment_record["ink_label_path"],
        "stored_rowcol_bounds": tuple(int(v) for v in stored_rowcol_bounds),
        "valid_point_count": int(valid_count),
        "positive_point_count": int(positive_count),
        "has_positive_points": bool(positive_count > 0),
    }


def _build_patch_record_for_bbox(
    *,
    dataset_idx,
    volume,
    volume_scale,
    world_bbox,
    segment_records,
    stored_grid_pad,
):
    patch_segments = []
    supervised_segment_indices = []
    for segment_record in segment_records:
        patch_segment = _build_patch_segment_entry(
            segment_record,
            world_bbox,
            stored_grid_pad,
        )
        if patch_segment is None:
            continue
        if patch_segment["has_positive_points"]:
            supervised_segment_indices.append(len(patch_segments))
        patch_segments.append(patch_segment)

    if not patch_segments:
        return None, "without_points"
    if not supervised_segment_indices:
        return None, "without_positive"

    return (
        {
            "dataset_idx": int(dataset_idx),
            "volume": volume,
            "scale": int(volume_scale),
            "world_bbox": tuple(int(v) for v in world_bbox),
            "segments": patch_segments,
            "supervised_segment_indices": tuple(int(v) for v in supervised_segment_indices),
        },
        None,
    )


def _evaluate_bbox_chunk(
    *,
    dataset_idx,
    volume,
    volume_scale,
    world_bboxes,
    segment_records,
    stored_grid_pad,
):
    chunk_patches = []
    chunk_stats = {
        "rejected_without_points": 0,
        "rejected_without_positive": 0,
    }
    for world_bbox in world_bboxes:
        patch_record, rejection_reason = _build_patch_record_for_bbox(
            dataset_idx=dataset_idx,
            volume=volume,
            volume_scale=volume_scale,
            world_bbox=world_bbox,
            segment_records=segment_records,
            stored_grid_pad=stored_grid_pad,
        )
        if patch_record is not None:
            chunk_patches.append(patch_record)
            continue
        if rejection_reason == "without_points":
            chunk_stats["rejected_without_points"] += 1
        elif rejection_reason == "without_positive":
            chunk_stats["rejected_without_positive"] += 1
        else:
            raise ValueError(f"Unexpected bbox rejection reason: {rejection_reason!r}")

    return chunk_patches, chunk_stats


def _build_dataset_patch_records(
    *,
    dataset_idx,
    volume,
    volume_scale,
    segment_records,
    patch_size_zyx,
    overlap_fraction,
    patch_finding_workers,
    stored_grid_pad,
):
    stats = {
        "candidate_bboxes": 0,
        "rejected_without_points": 0,
        "rejected_without_positive": 0,
        "kept_patches": 0,
    }
    union_bbox = _inclusive_bbox_from_point_sets(
        record["labeled_points_zyx"] for record in segment_records
    )
    if union_bbox is None:
        return [], stats

    candidate_bboxes = _generate_sliding_bboxes(
        union_bbox,
        patch_size_zyx,
        overlap_fraction,
    )
    stats["candidate_bboxes"] = int(len(candidate_bboxes))

    dataset_patches = []
    worker_count = max(1, int(patch_finding_workers))
    if worker_count == 1 or len(candidate_bboxes) <= 1:
        bbox_iter = tqdm(
            candidate_bboxes,
            total=len(candidate_bboxes),
            desc=f"Filtering bboxes (dataset {dataset_idx + 1})",
            unit="bbox",
            leave=False,
        )
        for world_bbox in bbox_iter:
            patch_record, rejection_reason = _build_patch_record_for_bbox(
                dataset_idx=dataset_idx,
                volume=volume,
                volume_scale=volume_scale,
                world_bbox=world_bbox,
                segment_records=segment_records,
                stored_grid_pad=stored_grid_pad,
            )
            if patch_record is not None:
                dataset_patches.append(patch_record)
                continue
            if rejection_reason == "without_points":
                stats["rejected_without_points"] += 1
            elif rejection_reason == "without_positive":
                stats["rejected_without_positive"] += 1
            else:
                raise ValueError(f"Unexpected bbox rejection reason: {rejection_reason!r}")
    else:
        chunk_size = max(1, len(candidate_bboxes) // (worker_count * 4))
        bbox_chunks = [
            candidate_bboxes[start : start + chunk_size]
            for start in range(0, len(candidate_bboxes), chunk_size)
        ]
        completed_chunks = {}
        bbox_progress = tqdm(
            total=len(candidate_bboxes),
            desc=f"Filtering bboxes (dataset {dataset_idx + 1})",
            unit="bbox",
            leave=False,
        )
        try:
            with ThreadPoolExecutor(max_workers=min(worker_count, len(bbox_chunks))) as executor:
                future_to_chunk = {
                    executor.submit(
                        _evaluate_bbox_chunk,
                        dataset_idx=dataset_idx,
                        volume=volume,
                        volume_scale=volume_scale,
                        world_bboxes=tuple(world_bboxes),
                        segment_records=segment_records,
                        stored_grid_pad=stored_grid_pad,
                    ): (chunk_idx, len(world_bboxes))
                    for chunk_idx, world_bboxes in enumerate(bbox_chunks)
                }
                for future in as_completed(future_to_chunk):
                    chunk_idx, processed_count = future_to_chunk[future]
                    completed_chunks[chunk_idx] = future.result()
                    bbox_progress.update(processed_count)
        finally:
            bbox_progress.close()

        for chunk_idx in range(len(bbox_chunks)):
            chunk_patches, chunk_stats = completed_chunks[chunk_idx]
            dataset_patches.extend(chunk_patches)
            stats["rejected_without_points"] += int(chunk_stats["rejected_without_points"])
            stats["rejected_without_positive"] += int(chunk_stats["rejected_without_positive"])

    stats["kept_patches"] = int(len(dataset_patches))
    return dataset_patches, stats


def _prepare_segment_records(
    *,
    dataset_idx,
    dataset,
    volume,
    volume_scale,
    patch_generation_stats,
):
    segments_path = dataset["segments_path"]
    z_range = _parse_z_range(dataset.get("z_range", None))
    dataset_segments = list(tifxyz.load_folder(segments_path))

    retarget_factor = 2 ** int(volume_scale)
    segment_records = []
    segment_iter = tqdm(
        enumerate(dataset_segments),
        total=len(dataset_segments),
        desc=f"Preparing segments (dataset {dataset_idx + 1})",
        unit="segment",
    )
    for segment_idx, original_seg in segment_iter:
        seg_scaled = original_seg.retarget(retarget_factor)
        if not _segment_overlaps_z_range(seg_scaled, z_range):
            continue

        patch_generation_stats["segments_considered"] += 1
        patch_generation_stats["segments_tried"] += 1
        seg_scaled.volume = volume
        segment_uuid = str(seg_scaled.uuid)

        original_seg.use_stored_resolution()
        x_stored, y_stored, z_stored, valid_stored = original_seg[:, :]
        x_stored = np.asarray(x_stored, dtype=np.float32)
        y_stored = np.asarray(y_stored, dtype=np.float32)
        z_stored = np.asarray(z_stored, dtype=np.float32)
        valid_mask = np.asarray(valid_stored, dtype=bool)
        valid_mask &= np.isfinite(x_stored)
        valid_mask &= np.isfinite(y_stored)
        valid_mask &= np.isfinite(z_stored)
        if not bool(np.any(valid_mask)):
            patch_generation_stats["segments_without_points"] += 1
            continue

        grid_shape = (int(x_stored.shape[0]), int(x_stored.shape[1]))
        valid_rows, valid_cols = np.where(valid_mask)
        valid_points_zyx = _points_from_mask(
            z_stored,
            y_stored,
            x_stored,
            valid_mask,
            retarget_factor,
        )

        positive_mask = np.zeros_like(valid_mask, dtype=bool)
        labeled_mask = np.zeros_like(valid_mask, dtype=bool)
        ink_label_path = None
        try:
            ink_mask, supervision_mask, ink_label_path = load_segment_label_masks(
                original_seg,
                grid_shape,
            )
            positive_mask = valid_mask & ink_mask
            labeled_mask = valid_mask & (ink_mask | supervision_mask)
        except AssertionError as exc:
            patch_generation_stats["segments_missing_ink"] += 1
            warnings.warn(f"Segment {segment_uuid!r} labels unavailable: {exc}")

        labeled_points_zyx = _points_from_mask(
            z_stored,
            y_stored,
            x_stored,
            labeled_mask,
            retarget_factor,
        )
        positive_points_zyx = _points_from_mask(
            z_stored,
            y_stored,
            x_stored,
            positive_mask,
            retarget_factor,
        )

        if labeled_points_zyx.size == 0:
            patch_generation_stats["segments_without_labels"] += 1
        if positive_points_zyx.size == 0:
            patch_generation_stats["segments_without_positive_points"] += 1

        scale_y, scale_x = _scale_pair_or_default(seg_scaled)
        segment_records.append(
            {
                "segment_idx": int(segment_idx),
                "segment_uuid": segment_uuid,
                "segment": seg_scaled,
                "volume": volume,
                "scale": int(volume_scale),
                "grid_shape": grid_shape,
                "scale_yx": (float(scale_y), float(scale_x)),
                "ink_label_path": str(ink_label_path) if ink_label_path else None,
                "valid_rows": np.asarray(valid_rows, dtype=np.int32),
                "valid_cols": np.asarray(valid_cols, dtype=np.int32),
                "valid_points_zyx": valid_points_zyx,
                "labeled_points_zyx": labeled_points_zyx,
                "positive_points_zyx": positive_points_zyx,
            }
        )

    return segment_records


def _serialize_patch_record(patch):
    return {
        "world_bbox": [int(v) for v in patch["world_bbox"]],
        "supervised_segment_indices": [
            int(v) for v in patch["supervised_segment_indices"]
        ],
        "segments": [
            {
                "segment_uuid": str(segment["segment_uuid"]),
                "segment_idx": int(segment["segment_idx"]),
                "stored_rowcol_bounds": [
                    int(v) for v in segment["stored_rowcol_bounds"]
                ],
                "valid_point_count": int(segment["valid_point_count"]),
                "positive_point_count": int(segment["positive_point_count"]),
                "has_positive_points": bool(segment["has_positive_points"]),
                "ink_label_path": segment["ink_label_path"],
            }
            for segment in patch["segments"]
        ],
    }


def _load_cached_patches(cache_entry, *, dataset_idx, volume, volume_scale, segment_by_uuid):
    cached_patches = []
    for record in cache_entry.get("patches", []):
        patch_segments = []
        supervised_segment_indices = []
        for cached_segment in record.get("segments", []):
            segment_uuid = str(cached_segment["segment_uuid"])
            runtime_segment = segment_by_uuid.get(segment_uuid)
            if runtime_segment is None:
                continue
            patch_segment = {
                "segment_idx": int(runtime_segment["segment_idx"]),
                "segment_uuid": segment_uuid,
                "segment": runtime_segment["segment"],
                "ink_label_path": cached_segment.get("ink_label_path")
                or runtime_segment.get("ink_label_path"),
                "stored_rowcol_bounds": tuple(
                    int(v) for v in cached_segment["stored_rowcol_bounds"]
                ),
                "valid_point_count": int(cached_segment["valid_point_count"]),
                "positive_point_count": int(cached_segment["positive_point_count"]),
                "has_positive_points": bool(cached_segment["has_positive_points"]),
            }
            if patch_segment["has_positive_points"]:
                supervised_segment_indices.append(len(patch_segments))
            patch_segments.append(patch_segment)

        if not patch_segments or not supervised_segment_indices:
            continue

        cached_patches.append(
            {
                "dataset_idx": int(dataset_idx),
                "volume": volume,
                "scale": int(volume_scale),
                "world_bbox": tuple(int(v) for v in record["world_bbox"]),
                "segments": patch_segments,
                "supervised_segment_indices": tuple(int(v) for v in supervised_segment_indices),
            }
        )
    return cached_patches


def find_patches(
    config,
    *,
    patch_size_zyx,
    overlap_fraction,
    patch_finding_workers,
    patch_cache_force_recompute,
    patch_cache_filename,
):
    stored_grid_pad = int(config.get("stored_grid_pad", DEFAULT_STORED_GRID_PAD))
    patches = []
    patch_generation_stats = {
        "segments_considered": 0,
        "segments_tried": 0,
        "segments_missing_ink": 0,
        "segments_without_points": 0,
        "segments_without_labels": 0,
        "segments_without_positive_points": 0,
        "candidate_bboxes": 0,
        "rejected_without_points": 0,
        "rejected_without_positive": 0,
        "kept_patches": 0,
        "cache_hits": 0,
    }

    datasets = config["datasets"]
    for dataset_idx, dataset in enumerate(datasets):
        volume_path = dataset["volume_path"]
        volume_scale = int(dataset["volume_scale"])
        volume_auth_json = dataset.get("volume_auth_json", config.get("volume_auth_json"))
        volume = open_zarr(
            volume_path,
            volume_scale,
            auth=volume_auth_json,
        )

        segment_records = _prepare_segment_records(
            dataset_idx=dataset_idx,
            dataset=dataset,
            volume=volume,
            volume_scale=volume_scale,
            patch_generation_stats=patch_generation_stats,
        )
        segment_by_uuid = {
            str(record["segment_uuid"]): record
            for record in segment_records
        }

        cache_path = os.path.join(
            str(dataset["segments_path"]),
            str(patch_cache_filename),
        )
        cache_keys = {
            "dataset": {
                "volume_path": str(dataset["volume_path"]),
                "volume_scale": int(dataset["volume_scale"]),
                "segments_path": str(dataset["segments_path"]),
                "z_range": dataset.get("z_range"),
            },
            "patch_size_zyx": [int(v) for v in patch_size_zyx],
            "overlap_fraction": float(overlap_fraction),
            "stored_grid_pad": int(stored_grid_pad),
        }
        cache_key = hashlib.sha256(
            json.dumps(cache_keys, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

        cache_entries = {}
        if not patch_cache_force_recompute and os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                cache_json = json.load(f)
            if isinstance(cache_json, dict) and isinstance(cache_json.get("entries"), dict):
                cache_entries = cache_json["entries"]

            cache_entry = cache_entries.get(cache_key)
            if isinstance(cache_entry, dict):
                cache_patches = _load_cached_patches(
                    cache_entry,
                    dataset_idx=dataset_idx,
                    volume=volume,
                    volume_scale=volume_scale,
                    segment_by_uuid=segment_by_uuid,
                )
                patches.extend(cache_patches)
                patch_generation_stats["kept_patches"] += int(len(cache_patches))
                patch_generation_stats["cache_hits"] += 1
                continue

        dataset_patches, dataset_stats = _build_dataset_patch_records(
            dataset_idx=dataset_idx,
            volume=volume,
            volume_scale=volume_scale,
            segment_records=segment_records,
            patch_size_zyx=patch_size_zyx,
            overlap_fraction=overlap_fraction,
            patch_finding_workers=patch_finding_workers,
            stored_grid_pad=stored_grid_pad,
        )
        patches.extend(dataset_patches)
        patch_generation_stats["candidate_bboxes"] += int(dataset_stats["candidate_bboxes"])
        patch_generation_stats["rejected_without_points"] += int(dataset_stats["rejected_without_points"])
        patch_generation_stats["rejected_without_positive"] += int(dataset_stats["rejected_without_positive"])
        patch_generation_stats["kept_patches"] += int(dataset_stats["kept_patches"])

        cache_entries[cache_key] = {
            "patches": [_serialize_patch_record(patch) for patch in dataset_patches]
        }
        cache_dir = os.path.dirname(cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"entries": cache_entries}, f, separators=(",", ":"))

    return patches, patch_generation_stats
