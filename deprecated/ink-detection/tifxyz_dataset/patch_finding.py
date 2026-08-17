import hashlib
import json
import multiprocessing as mp
import os
import warnings
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from tqdm.auto import tqdm

import vesuvius.tifxyz as tifxyz
from .common import (
    _empty_patch_generation_stats,
    _fix_known_bottom_right_padding,
)
from vesuvius.neural_tracing.datasets.common import (
    _parse_z_range,
    _segment_overlaps_z_range,
    open_zarr,
)
from vesuvius.neural_tracing.inference.generate_segment_cover_bboxes import (
    _generate_segment_cover_records,
)

_PATCH_CACHE_DEFAULT_FILENAME = ".tifxyz_patch_cache.json"
_PATCH_EVAL_CONTEXT = None


def _step_from_scale(scale_value):
    scale_value = float(scale_value)
    if not np.isfinite(scale_value) or scale_value <= 0.0:
        return 1.0
    return max(1.0, 1.0 / scale_value)


def _prepare_patch_candidates(initial_patches):
    parsed = []
    for patch_idx, patch in enumerate(initial_patches):
        bbox = tuple(int(v) for v in patch["bbox"])
        parsed.append(
            {
                "patch_idx": int(patch_idx),
                "bbox": bbox,
                "z_key": (int(bbox[0]), int(bbox[1])),
                "bbox_id": int(patch["bbox_id"]),
                "z_band": int(patch["z_band"]),
                "grid_index": tuple(int(v) for v in patch["grid_index"]),
            }
        )
    return parsed


def _group_patches_by_z_key(parsed_patches):
    if not parsed_patches:
        return []
    sorted_patches = sorted(
        parsed_patches,
        key=lambda patch: (
            int(patch["z_key"][0]),
            int(patch["z_key"][1]),
            int(patch["patch_idx"]),
        ),
    )
    grouped = []
    current_group = [sorted_patches[0]]
    current_z_key = sorted_patches[0]["z_key"]
    for patch in sorted_patches[1:]:
        if patch["z_key"] == current_z_key:
            current_group.append(patch)
            continue
        grouped.append(current_group)
        current_group = [patch]
        current_z_key = patch["z_key"]
    grouped.append(current_group)
    return grouped


def _evaluate_patch_chunk(patch_chunk):
    context = _PATCH_EVAL_CONTEXT
    if not context:
        raise RuntimeError("patch evaluation context is not initialized")

    all_valid_z = context["all_valid_z"]
    all_valid_y = context["all_valid_y"]
    all_valid_x = context["all_valid_x"]
    all_positive_z = context["all_positive_z"]
    all_positive_y = context["all_positive_y"]
    all_positive_x = context["all_positive_x"]
    min_positive_fraction = float(context["min_positive_fraction"])
    min_span_ratio = float(context["min_span_ratio"])
    patch_size_zyx = context["patch_size_zyx"]
    sample_step_zyx = np.asarray(context["sample_step_zyx"], dtype=np.float32).reshape(3)
    # Spans are measured from sampled points (max-min), so allow one sampling step of slack.
    required_span_z = float(min_span_ratio) * max(
        0.0, (float(patch_size_zyx[0]) - 1.0) - float(sample_step_zyx[0])
    )
    required_span_y = float(min_span_ratio) * max(
        0.0, (float(patch_size_zyx[1]) - 1.0) - float(sample_step_zyx[1])
    )
    required_span_x = float(min_span_ratio) * max(
        0.0, (float(patch_size_zyx[2]) - 1.0) - float(sample_step_zyx[2])
    )

    kept = []
    rejected_positive_fraction = 0
    rejected_span = 0
    z_band_cache = {}

    for patch in patch_chunk:
        z_key = patch["z_key"]
        z_cache_entry = z_band_cache.get(z_key)
        if z_cache_entry is None:
            z_min, z_max = z_key
            valid_in_band = (
                (all_valid_z >= float(z_min))
                & (all_valid_z < float(z_max) + 1.0)
            )
            positive_in_band = (
                (all_positive_z >= float(z_min))
                & (all_positive_z < float(z_max) + 1.0)
            )
            z_cache_entry = {
                "valid_z": all_valid_z[valid_in_band],
                "valid_y": all_valid_y[valid_in_band],
                "valid_x": all_valid_x[valid_in_band],
                "positive_y": all_positive_y[positive_in_band],
                "positive_x": all_positive_x[positive_in_band],
            }
            z_band_cache[z_key] = z_cache_entry

        _, _, y_min, y_max, x_min, x_max = patch["bbox"]
        valid_y = z_cache_entry["valid_y"]
        valid_x = z_cache_entry["valid_x"]
        in_bbox_valid = (
            (valid_y >= float(y_min))
            & (valid_y < float(y_max) + 1.0)
            & (valid_x >= float(x_min))
            & (valid_x < float(x_max) + 1.0)
        )
        valid_count = int(np.count_nonzero(in_bbox_valid))
        if valid_count == 0:
            continue

        positive_y = z_cache_entry["positive_y"]
        positive_x = z_cache_entry["positive_x"]
        in_bbox_positive = (
            (positive_y >= float(y_min))
            & (positive_y < float(y_max) + 1.0)
            & (positive_x >= float(x_min))
            & (positive_x < float(x_max) + 1.0)
        )
        positive_count = int(np.count_nonzero(in_bbox_positive))
        positive_fraction = float(positive_count) / float(valid_count)
        if positive_fraction < min_positive_fraction:
            rejected_positive_fraction += 1
            continue

        z_values = z_cache_entry["valid_z"][in_bbox_valid]
        y_values = valid_y[in_bbox_valid]
        x_values = valid_x[in_bbox_valid]
        z_span = float(np.max(z_values) - np.min(z_values))
        y_span = float(np.max(y_values) - np.min(y_values))
        x_span = float(np.max(x_values) - np.min(x_values))
        if y_span >= x_span:
            dominant_span = y_span
            required_dominant_span = required_span_y
        else:
            dominant_span = x_span
            required_dominant_span = required_span_x

        if z_span < required_span_z or dominant_span < required_dominant_span:
            rejected_span += 1
            continue

        kept.append(
            {
                "patch_idx": int(patch["patch_idx"]),
                "world_bbox": patch["bbox"],
                "bbox_id": int(patch["bbox_id"]),
                "z_band": int(patch["z_band"]),
                "grid_index": patch["grid_index"],
                "valid_point_count": valid_count,
                "positive_point_count": positive_count,
                "positive_fraction": positive_fraction,
                "span_zyx": (z_span, y_span, x_span),
            }
        )

    return {
        "processed": int(len(patch_chunk)),
        "rejected_positive_fraction": int(rejected_positive_fraction),
        "rejected_span": int(rejected_span),
        "kept": kept,
    }


def find_patches(
    config,
    *,
    patch_size_zyx,
    overlap_fraction,
    min_positive_fraction,
    min_span_ratio,
    patch_finding_workers,
    patch_cache_force_recompute,
    patch_cache_filename,
    auto_fix_padding_multiples,
):
    patches = []
    patch_generation_stats = _empty_patch_generation_stats()

    datasets = config["datasets"]
    for dataset_idx, dataset in enumerate(datasets):
        volume_path = dataset["volume_path"]
        volume_scale = dataset["volume_scale"]

        volume_auth_json = dataset.get("volume_auth_json", config.get("volume_auth_json"))
        volume = open_zarr(
            volume_path,
            scale=volume_scale,
            auth_json_path=volume_auth_json,
            config=config,
        )
        segments_path = dataset["segments_path"]
        z_range = _parse_z_range(dataset.get("z_range", None))
        dataset_segments = list(tifxyz.load_folder(segments_path))

        retarget_factor = 2 ** volume_scale
        segment_pairs = []
        for i, seg in enumerate(dataset_segments):
            seg_scaled = seg.retarget(retarget_factor)
            if not _segment_overlaps_z_range(seg_scaled, z_range):
                continue
            seg_scaled.volume = volume
            segment_pairs.append((i, seg, seg_scaled))

        patch_generation_stats["segments_considered"] += int(len(segment_pairs))
        segment_by_uuid = {
            str(seg_scaled.uuid): (int(segment_idx), seg_scaled)
            for segment_idx, _, seg_scaled in segment_pairs
        }
        segment_ink_label_path_by_uuid = {}
        for _, original_seg, seg_scaled in segment_pairs:
            ink_meta = next(
                (label for label in original_seg.list_labels() if label.get("name") == "inklabels"),
                None,
            )
            if ink_meta is None:
                continue
            ink_path = ink_meta.get("path")
            if ink_path is None:
                continue
            segment_ink_label_path_by_uuid[str(seg_scaled.uuid)] = str(ink_path)

        cache_path = os.path.join(
            str(dataset["segments_path"]),
            str(patch_cache_filename or _PATCH_CACHE_DEFAULT_FILENAME),
        )
        cache_keys = {
            "dataset": {
                "volume_path": str(dataset["volume_path"]),
                "volume_scale": int(dataset["volume_scale"]),
                "segments_path": str(dataset["segments_path"]),
                "z_range": dataset.get("z_range"),
            },
            "patch_size_zyx": [int(v) for v in patch_size_zyx],
            "min_positive_fraction": float(min_positive_fraction),
            "min_span_ratio": float(min_span_ratio),
            "overlap_fraction": float(overlap_fraction),
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
                cache_patches = []
                for record in cache_entry.get("patches", []):
                    segment_uuid = str(record["segment_uuid"])
                    seg_payload = segment_by_uuid.get(segment_uuid)
                    if seg_payload is None:
                        continue
                    ink_label_path = record.get("ink_label_path")
                    if not ink_label_path:
                        ink_label_path = segment_ink_label_path_by_uuid.get(segment_uuid)
                    segment_idx_cached, seg_scaled_cached = seg_payload
                    cache_patches.append(
                        {
                            "dataset_idx": int(dataset_idx),
                            "segment_idx": int(segment_idx_cached),
                            "segment_uuid": segment_uuid,
                            "segment": seg_scaled_cached,
                            "volume": volume,
                            "scale": int(volume_scale),
                            "world_bbox": tuple(int(v) for v in record["world_bbox"]),
                            "bbox_id": int(record["bbox_id"]),
                            "z_band": int(record["z_band"]),
                            "grid_index": tuple(int(v) for v in record["grid_index"]),
                            "valid_point_count": int(record["valid_point_count"]),
                            "positive_point_count": int(record["positive_point_count"]),
                            "positive_fraction": float(record["positive_fraction"]),
                            "span_zyx": tuple(float(v) for v in record["span_zyx"]),
                            "ink_label_path": str(ink_label_path) if ink_label_path else None,
                        }
                    )
                patches.extend(cache_patches)
                patch_generation_stats["kept_patches"] += int(len(cache_patches))
                continue

        dataset_patches = []

        segment_pairs_iter = tqdm(
            segment_pairs,
            total=len(segment_pairs),
            desc=f"Finding patches (dataset {dataset_idx + 1}/{len(datasets)})",
            unit="segment",
        )

        for segment_ordinal, (segment_idx, original_seg, seg_scaled) in enumerate(
            segment_pairs_iter,
            start=1,
        ):
            patch_generation_stats["segments_tried"] += 1

            ink_meta = next(
                (label for label in original_seg.list_labels() if label.get("name") == "inklabels"),
                None,
            )
            if ink_meta is None:
                patch_generation_stats["segments_missing_ink"] += 1
                warnings.warn(
                    f"Skipping segment {original_seg.uuid!r}: unable to find 'inklabels'."
                )
                continue
            segment_ink_label_path = str(ink_meta["path"])

            ink_label = cv2.imread(str(ink_meta["path"]), cv2.IMREAD_UNCHANGED)
            if ink_label is None:
                patch_generation_stats["segments_missing_ink"] += 1
                warnings.warn(
                    f"Skipping segment {original_seg.uuid!r}: unable to read 'inklabels'."
                )
                continue
            if ink_label.ndim == 3:
                if ink_label.shape[2] == 4:
                    ink_label = cv2.cvtColor(ink_label, cv2.COLOR_BGRA2GRAY)
                else:
                    ink_label = cv2.cvtColor(ink_label, cv2.COLOR_BGR2GRAY)

            stored_h, stored_w = original_seg.use_stored_resolution().shape
            scale_y, scale_x = original_seg._scale
            expected_label_shape = (
                int(stored_h / scale_y),
                int(stored_w / scale_x),
            )
            if tuple(int(v) for v in ink_label.shape) != expected_label_shape:
                fixed_label, matched_multiple = _fix_known_bottom_right_padding(
                    ink_label,
                    expected_label_shape,
                    auto_fix_padding_multiples,
                )
                if fixed_label is not None:
                    ink_label = fixed_label
                    patch_generation_stats["segments_autofixed_padding"] += 1
                    label_path = str(ink_meta["path"])
                    if cv2.imwrite(label_path, ink_label):
                        warnings.warn(
                            f"Auto-fixed {original_seg.uuid!r} label shape mismatch "
                            f"{tuple(int(v) for v in ink_label.shape)} using multiple={matched_multiple}; "
                            f"saved to {label_path!r}."
                        )
                    else:
                        warnings.warn(
                            f"Auto-fixed {original_seg.uuid!r} in memory but failed to save {label_path!r}; "
                            "continuing."
                        )
                else:
                    patch_generation_stats["segments_missing_ink"] += 1
                    warnings.warn(
                        f"Skipping segment {original_seg.uuid!r}: label shape {ink_label.shape} "
                        f"does not match segment full-resolution shape {expected_label_shape}, "
                        "and does not match known bottom/right 256/64 padding."
                    )
                    continue

            original_seg.use_stored_resolution()
            x_stored, y_stored, z_stored, valid_stored = original_seg[:, :]
            valid_mask = np.asarray(valid_stored, dtype=bool).copy()
            valid_mask &= np.isfinite(z_stored)
            valid_mask &= np.isfinite(y_stored)
            valid_mask &= np.isfinite(x_stored)

            if tuple(int(v) for v in ink_label.shape) == (int(stored_h), int(stored_w)):
                positive_label_mask = (ink_label > 0)
            else:
                # Downsample full-res labels onto the stored tifxyz grid and keep any-positive cells.
                positive_label_mask = (
                    cv2.resize(
                        (ink_label > 0).astype(np.uint8),
                        (int(stored_w), int(stored_h)),
                        interpolation=cv2.INTER_AREA,
                    )
                    > 0
                )

            positive_mask = valid_mask & positive_label_mask
            if not np.any(positive_mask):
                patch_generation_stats["segments_without_positive_points"] += 1
                warnings.warn(
                    f"Skipping segment {original_seg.uuid!r}: No positive labels found"
                )
                continue

            all_valid_points_zyx = np.stack(
                [z_stored[valid_mask], y_stored[valid_mask], x_stored[valid_mask]],
                axis=-1,
            ).astype(np.float32, copy=False)

            positive_points_zyx = np.stack(
                [z_stored[positive_mask], y_stored[positive_mask], x_stored[positive_mask]],
                axis=-1,
            ).astype(np.float32, copy=False)

            if retarget_factor != 1:
                all_valid_points_zyx /= float(retarget_factor)
                positive_points_zyx /= float(retarget_factor)

            sample_step_y = _step_from_scale(scale_y)
            sample_step_x = _step_from_scale(scale_x)
            sample_step_zyx = np.asarray(
                [max(sample_step_y, sample_step_x), sample_step_y, sample_step_x],
                dtype=np.float32,
            )
            if retarget_factor != 1:
                sample_step_zyx /= float(retarget_factor)

            result = _generate_segment_cover_records(
                points_zyx=positive_points_zyx,
                crop_size_zyx=patch_size_zyx,
                overlap=overlap_fraction,
                prune_bboxes=True,
                band_workers=patch_finding_workers,
                show_progress=True,
                progress_desc=(
                    f"Optimizing z-bands (segment {segment_ordinal}/{len(segment_pairs)})"
                ),
            )

            initial_patches = result["final_records"]
            patch_generation_stats["candidate_bboxes"] += int(len(initial_patches))
            parsed_patches = _prepare_patch_candidates(initial_patches)
            if parsed_patches:
                patch_chunks = _group_patches_by_z_key(parsed_patches)
                chunk_results = []

                global _PATCH_EVAL_CONTEXT
                _PATCH_EVAL_CONTEXT = {
                    "all_valid_z": all_valid_points_zyx[:, 0],
                    "all_valid_y": all_valid_points_zyx[:, 1],
                    "all_valid_x": all_valid_points_zyx[:, 2],
                    "all_positive_z": positive_points_zyx[:, 0],
                    "all_positive_y": positive_points_zyx[:, 1],
                    "all_positive_x": positive_points_zyx[:, 2],
                    "min_positive_fraction": float(min_positive_fraction),
                    "min_span_ratio": float(min_span_ratio),
                    "patch_size_zyx": np.asarray(patch_size_zyx, dtype=np.float32).reshape(3),
                    "sample_step_zyx": sample_step_zyx,
                }

                bbox_eval_bar = tqdm(
                    total=len(parsed_patches),
                    desc=f"Filtering bboxes (segment {segment_ordinal}/{len(segment_pairs)})",
                    unit="bbox",
                    leave=False,
                )
                try:
                    process_ctx = mp.get_context("fork")
                    with ProcessPoolExecutor(
                        max_workers=int(patch_finding_workers),
                        mp_context=process_ctx,
                    ) as pool:
                        for chunk_result in pool.map(_evaluate_patch_chunk, patch_chunks):
                            chunk_results.append(chunk_result)
                            bbox_eval_bar.update(int(chunk_result["processed"]))
                finally:
                    bbox_eval_bar.close()
                    _PATCH_EVAL_CONTEXT = None

                kept_records = []
                for chunk_result in chunk_results:
                    patch_generation_stats["rejected_positive_fraction"] += int(
                        chunk_result["rejected_positive_fraction"]
                    )
                    patch_generation_stats["rejected_span"] += int(
                        chunk_result["rejected_span"]
                    )
                    kept_records.extend(chunk_result["kept"])

                kept_records.sort(key=lambda record: int(record["patch_idx"]))
                for kept in kept_records:
                    dataset_patches.append(
                        {
                            "dataset_idx": int(dataset_idx),
                            "segment_idx": int(segment_idx),
                            "segment_uuid": str(seg_scaled.uuid),
                            "segment": seg_scaled,
                            "volume": volume,
                            "scale": int(volume_scale),
                            "world_bbox": tuple(int(v) for v in kept["world_bbox"]),
                            "bbox_id": int(kept["bbox_id"]),
                            "z_band": int(kept["z_band"]),
                            "grid_index": tuple(int(v) for v in kept["grid_index"]),
                            "valid_point_count": int(kept["valid_point_count"]),
                            "positive_point_count": int(kept["positive_point_count"]),
                            "positive_fraction": float(kept["positive_fraction"]),
                            "span_zyx": tuple(float(v) for v in kept["span_zyx"]),
                            "ink_label_path": segment_ink_label_path,
                        }
                    )
                patch_generation_stats["kept_patches"] += int(len(kept_records))

        patches.extend(dataset_patches)
        cache_entries[cache_key] = {
            "patches": [
                {
                    "segment_uuid": str(p["segment_uuid"]),
                    "world_bbox": list(p["world_bbox"]),
                    "bbox_id": int(p["bbox_id"]),
                    "z_band": int(p["z_band"]),
                    "grid_index": list(p["grid_index"]),
                    "valid_point_count": int(p["valid_point_count"]),
                    "positive_point_count": int(p["positive_point_count"]),
                    "positive_fraction": float(p["positive_fraction"]),
                    "span_zyx": list(p["span_zyx"]),
                    "ink_label_path": p.get("ink_label_path"),
                }
                for p in dataset_patches
            ]
        }

        cache_dir = os.path.dirname(cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"entries": cache_entries}, f, separators=(",", ":"))

    return patches, patch_generation_stats
