from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from vesuvius.tifxyz import Tifxyz


PATCH_FINDER_VERSION = 2


@dataclass(frozen=True)
class SegmentRecord:
    segment_idx: int
    segment: Tifxyz
    path: str
    uuid: str
    wrap_label: int


def parse_wrap_label(path_or_name: Any) -> int:
    text = Path(str(path_or_name)).name
    labels = sorted({int(m.group(1)) for m in re.finditer(r"(?:^|[^A-Za-z0-9])w(\d+)(?=$|[^A-Za-z0-9])", text)})
    if not labels:
        labels = sorted({int(m.group(1)) for m in re.finditer(r"w(\d+)", text)})
    if not labels:
        raise ValueError(f"Could not parse required wrap label w<number> from segment path/name: {text}")
    if len(labels) > 1:
        raise ValueError(f"Ambiguous wrap labels {labels} in segment path/name: {text}")
    return int(labels[0])


def _volume_array_shape(volume: Any, scale: int) -> Tuple[int, int, int]:
    arr = volume
    if hasattr(volume, "__contains__") and str(scale) in volume:
        arr = volume[str(scale)]
    shape = tuple(int(v) for v in getattr(arr, "shape", ()))
    if len(shape) != 3:
        raise ValueError(f"Expected a 3D zarr array/group at scale {scale}, got shape={shape!r}")
    return shape


def _chunk_starts(axis_size: int, crop: int, stride: int) -> np.ndarray:
    if axis_size <= 0:
        return np.array([], dtype=np.int64)
    if axis_size <= crop:
        return np.array([0], dtype=np.int64)
    starts = list(range(0, max(axis_size - crop, 0) + 1, stride))
    final = axis_size - crop
    if starts[-1] != final:
        starts.append(final)
    return np.asarray(starts, dtype=np.int64)


def _build_chunk_grid(
    volume_shape: Tuple[int, int, int],
    crop_size: Tuple[int, int, int],
    overlap_fraction: float,
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[int, int, int]]:
    if not (0.0 <= float(overlap_fraction) < 1.0):
        raise ValueError(f"overlap_fraction must be in [0, 1), got {overlap_fraction}")
    strides = tuple(max(1, int(round(c * (1.0 - float(overlap_fraction))))) for c in crop_size)
    starts = tuple(_chunk_starts(s, c, st) for s, c, st in zip(volume_shape, crop_size, strides))
    if any(len(axis) == 0 for axis in starts):
        raise ValueError(f"Cannot build chunk grid for volume shape={volume_shape}")
    return starts, strides


def _stored_step_voxels(seg: Tifxyz) -> float:
    scale = getattr(seg, "_scale", None)
    if scale is None:
        return 20.0
    values = np.asarray(scale, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        return 20.0
    return float(np.max(values))


def _chunk_id_ranges_for_points(
    coords: np.ndarray,
    *,
    starts: np.ndarray,
    size: int,
    pad: float,
) -> Tuple[np.ndarray, np.ndarray]:
    lo = np.asarray(coords, dtype=np.float64) - float(size) - float(pad)
    hi = np.asarray(coords, dtype=np.float64) + float(pad)
    start_idx = np.searchsorted(starts, lo, side="right")
    stop_idx = np.searchsorted(starts, hi, side="right")
    return start_idx.astype(np.int64, copy=False), stop_idx.astype(np.int64, copy=False)


def _chunk_id_ranges_for_guarded_points(
    coords: np.ndarray,
    *,
    starts: np.ndarray,
    size: int,
    guard_voxels: float,
) -> Tuple[np.ndarray, np.ndarray]:
    lo = np.asarray(coords, dtype=np.float64) - float(guard_voxels) - float(size)
    hi = np.asarray(coords, dtype=np.float64) + float(guard_voxels)
    start_idx = np.searchsorted(starts, lo, side="right")
    stop_idx = np.searchsorted(starts, hi, side="right")
    return start_idx.astype(np.int64, copy=False), stop_idx.astype(np.int64, copy=False)


def _merge_exclusion_side(prev: Optional[str], side: str) -> str:
    return side if prev in (None, side) else "both"


def _compute_terminal_exclusions(
    *,
    segment_idx: int,
    seg: Tifxyz,
    chunk_starts: Tuple[np.ndarray, np.ndarray, np.ndarray],
    crop_size: Tuple[int, int, int],
    guard_voxels: float,
) -> Dict[Tuple[int, int, int], str]:
    valid = np.asarray(seg._valid_mask, dtype=bool)
    if not bool(valid.any()):
        raise ValueError(f"Segment has no valid stored cells: {getattr(seg, 'path', '') or seg.uuid}")

    rows, cols = np.nonzero(valid)
    row_extent = int(rows.max() - rows.min())
    col_extent = int(cols.max() - cols.min())
    long_axis_rows = row_extent >= col_extent
    coords = rows if long_axis_rows else cols
    lo = int(coords.min())
    hi = int(coords.max())
    band_width = max(1, int(round(float(getattr(seg, "_scale", (1.0, 1.0))[0 if long_axis_rows else 1]) / 20.0)))

    start_mask = valid & ((np.arange(valid.shape[0])[:, None] <= lo + band_width - 1) if long_axis_rows else (np.arange(valid.shape[1])[None, :] <= lo + band_width - 1))
    end_mask = valid & ((np.arange(valid.shape[0])[:, None] >= hi - band_width + 1) if long_axis_rows else (np.arange(valid.shape[1])[None, :] >= hi - band_width + 1))

    exclusions: Dict[Tuple[int, int, int], str] = {}
    for mask, side in ((start_mask, "start"), (end_mask, "end")):
        rr, cc = np.nonzero(mask)
        if rr.size == 0:
            raise ValueError(
                f"Could not compute {side} terminal cells for segment: {getattr(seg, 'path', '') or seg.uuid}"
            )
        z = np.asarray(seg._z[rr, cc], dtype=np.float64)
        y = np.asarray(seg._y[rr, cc], dtype=np.float64)
        x = np.asarray(seg._x[rr, cc], dtype=np.float64)
        finite = np.isfinite(z) & np.isfinite(y) & np.isfinite(x)
        if not bool(finite.any()):
            raise ValueError(
                f"{side} terminal has no finite world points for segment: {getattr(seg, 'path', '') or seg.uuid}"
            )
        ranges_by_axis = [
            _chunk_id_ranges_for_guarded_points(
                coord[finite],
                starts=starts,
                size=size,
                guard_voxels=guard_voxels,
            )
            for coord, starts, size in zip((z, y, x), chunk_starts, crop_size)
        ]
        range_rows = np.column_stack((
            ranges_by_axis[0][0],
            ranges_by_axis[0][1],
            ranges_by_axis[1][0],
            ranges_by_axis[1][1],
            ranges_by_axis[2][0],
            ranges_by_axis[2][1],
        ))
        for z0, z1, y0, y1, x0, x1 in np.unique(range_rows, axis=0):
            for cz in range(int(z0), int(z1)):
                for cy in range(int(y0), int(y1)):
                    for cx in range(int(x0), int(x1)):
                        key = (int(cz), int(cy), int(cx))
                        exclusions[key] = _merge_exclusion_side(exclusions.get(key), side)
    return exclusions


def _assign_segment_to_chunks(
    *,
    segment_idx: int,
    seg: Tifxyz,
    chunk_starts: Tuple[np.ndarray, np.ndarray, np.ndarray],
    crop_size: Tuple[int, int, int],
    min_points_per_wrap: int,
    bbox_pad_2d: int,
    terminal_exclusions: Dict[Tuple[int, int, int], str],
    chunk_pad: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    valid = np.asarray(seg._valid_mask, dtype=bool)
    rows, cols = np.nonzero(valid)
    if rows.size == 0:
        return [], {"empty": 1, "terminal_start": 0, "terminal_end": 0, "terminal_both": 0, "terminal_chunk_ids": set()}

    z = np.asarray(seg._z[rows, cols], dtype=np.float64)
    y = np.asarray(seg._y[rows, cols], dtype=np.float64)
    x = np.asarray(seg._x[rows, cols], dtype=np.float64)
    finite = np.isfinite(z) & np.isfinite(y) & np.isfinite(x)
    rows, cols, z, y, x = rows[finite], cols[finite], z[finite], y[finite], x[finite]
    if rows.size == 0:
        return [], {"empty": 1, "terminal_start": 0, "terminal_end": 0, "terminal_both": 0, "terminal_chunk_ids": set()}

    ranges_by_axis = [
        _chunk_id_ranges_for_points(
            coords,
            starts=starts,
            size=size,
            pad=float(chunk_pad),
        )
        for coords, starts, size in zip((z, y, x), chunk_starts, crop_size)
    ]

    axis_lengths = [stop - start for start, stop in ranges_by_axis]
    if all(bool(np.all(length == 1)) for length in axis_lengths):
        chunk_ids = np.column_stack([start for start, _ in ranges_by_axis]).astype(np.int64, copy=False)
        order = np.lexsort((chunk_ids[:, 2], chunk_ids[:, 1], chunk_ids[:, 0]))
        sorted_ids = chunk_ids[order]
        boundaries = np.flatnonzero(np.any(sorted_ids[1:] != sorted_ids[:-1], axis=1)) + 1
        groups = np.split(order, boundaries)
        grouped_items = [(tuple(int(v) for v in chunk_ids[group[0]]), group) for group in groups]
    else:
        grouped: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
        for point_idx in range(rows.size):
            for cz in range(int(ranges_by_axis[0][0][point_idx]), int(ranges_by_axis[0][1][point_idx])):
                for cy in range(int(ranges_by_axis[1][0][point_idx]), int(ranges_by_axis[1][1][point_idx])):
                    for cx in range(int(ranges_by_axis[2][0][point_idx]), int(ranges_by_axis[2][1][point_idx])):
                        grouped[(cz, cy, cx)].append(point_idx)
        grouped_items = [
            (chunk_id, np.asarray(point_indices, dtype=np.int64))
            for chunk_id, point_indices in grouped.items()
        ]

    records = []
    stats = {"empty": 0, "terminal_start": 0, "terminal_end": 0, "terminal_both": 0, "terminal_chunk_ids": set()}
    for chunk_id, idx in grouped_items:
        exclusion = terminal_exclusions.get(chunk_id)
        if exclusion is not None:
            stats[f"terminal_{exclusion}"] += 1
            stats["terminal_chunk_ids"].add(tuple(int(v) for v in chunk_id))
            continue
        if idx.size < int(min_points_per_wrap):
            continue
        rr = rows[idx]
        cc = cols[idx]
        zz = z[idx]
        yy = y[idx]
        xx = x[idx]
        records.append({
            "chunk_id": tuple(int(v) for v in chunk_id),
            "segment_idx": int(segment_idx),
            "bbox_2d": (
                int(rr.min()) - int(bbox_pad_2d),
                int(rr.max()) + int(bbox_pad_2d),
                int(cc.min()) - int(bbox_pad_2d),
                int(cc.max()) + int(bbox_pad_2d),
            ),
            "world_bbox": (
                float(zz.min()), float(zz.max()),
                float(yy.min()), float(yy.max()),
                float(xx.min()), float(xx.max()),
            ),
            "point_count": int(idx.size),
        })
    return records, stats


def _neighbor_indices(wraps: List[Dict[str, Any]], source_idx: int) -> Tuple[int, ...]:
    source = wraps[source_idx]
    source_label = int(source["wrap_label"])
    candidates = []
    for idx, wrap in enumerate(wraps):
        if idx == source_idx:
            continue
        if int(wrap["segment_idx"]) == int(source["segment_idx"]):
            continue
        label = int(wrap["wrap_label"])
        if abs(label - source_label) != 1:
            continue
        candidates.append((abs(label - source_label), label, int(wrap["segment_idx"]), int(wrap["wrap_idx"]), idx))
    candidates.sort()
    return tuple(int(v[-1]) for v in candidates)


def _has_lower_and_upper_neighbors(wraps: List[Dict[str, Any]], source_idx: int, neighbor_indices: Iterable[int]) -> bool:
    source_label = int(wraps[source_idx]["wrap_label"])
    labels = [int(wraps[int(idx)]["wrap_label"]) for idx in neighbor_indices]
    return any(label < source_label for label in labels) and any(label > source_label for label in labels)


def find_training_chunks(
    *,
    segments: List[Tifxyz],
    volume: Any,
    scale: int,
    target_size: Tuple[int, int, int],
    overlap_fraction: float = 0.0,
    min_points_per_wrap: int = 100,
    bbox_pad_2d: int = 0,
    cache_dir: Optional[Path] = None,
    force_recompute: bool = False,
    verbose: bool = False,
    chunk_pad: float = 0.0,
    terminal_chunk_guard_voxels: Optional[float] = None,
    training_mode: str = "rowcol_hidden",
) -> List[Dict[str, Any]]:
    if not segments:
        return []

    segment_records = []
    for segment_idx, seg in enumerate(segments):
        path = str(getattr(seg, "path", "") or getattr(seg, "uuid", "") or segment_idx)
        segment_records.append(SegmentRecord(
            segment_idx=int(segment_idx),
            segment=seg,
            path=path,
            uuid=str(getattr(seg, "uuid", "")),
            wrap_label=parse_wrap_label(path),
        ))

    volume_shape = _volume_array_shape(volume, int(scale))
    target_size = tuple(int(v) for v in target_size)
    chunk_starts, strides = _build_chunk_grid(volume_shape, target_size, overlap_fraction)
    stored_step_voxels = max(_stored_step_voxels(record.segment) for record in segment_records)
    configured_guard = float(terminal_chunk_guard_voxels) if terminal_chunk_guard_voxels is not None else float(chunk_pad)
    guard_voxels = max(configured_guard, float(stored_step_voxels), 20.0)

    cache_key_data = {
        "version": PATCH_FINDER_VERSION,
        "volume_shape": list(volume_shape),
        "scale": int(scale),
        "target_size": list(target_size),
        "overlap_fraction": float(overlap_fraction),
        "min_points_per_wrap": int(min_points_per_wrap),
        "bbox_pad_2d": int(bbox_pad_2d),
        "chunk_pad": float(chunk_pad),
        "terminal_chunk_guard_voxels": float(guard_voxels),
        "training_mode": str(training_mode),
        "segments": [
            {
                "uuid": s.uuid,
                "path": s.path,
                "scale": list(getattr(s.segment, "_scale", ())),
                "wrap_label": int(s.wrap_label),
            }
            for s in segment_records
        ],
    }
    cache_key = hashlib.md5(json.dumps(cache_key_data, sort_keys=True).encode("utf8")).hexdigest()
    cache_file = Path(cache_dir) / f"training_chunks_{cache_key}.json" if cache_dir is not None else None
    if cache_file is not None and cache_file.exists() and not force_recompute:
        tqdm.write(f"Loading cached training chunks: {cache_file}")
        with open(cache_file, "r", encoding="utf8") as f:
            return json.load(f)["chunks"]

    chunk_wraps: Dict[Tuple[int, int, int], List[Dict[str, Any]]] = defaultdict(list)
    terminal_dropped_chunk_ids = set()
    stats = {
        "wraps_dropped_terminal_start": 0,
        "wraps_dropped_terminal_end": 0,
        "wraps_dropped_terminal_both": 0,
        "chunks_dropped_after_terminal_filter": 0,
        "chunks_without_neighboring_wraps": 0,
    }

    terminal_exclusions_by_segment: Dict[int, Dict[Tuple[int, int, int], str]] = {}
    for record in tqdm(segment_records, desc="Building terminal chunk exclusions"):
        terminal_exclusions_by_segment[int(record.segment_idx)] = _compute_terminal_exclusions(
            segment_idx=record.segment_idx,
            seg=record.segment,
            chunk_starts=chunk_starts,
            crop_size=target_size,
            guard_voxels=guard_voxels,
        )

    iterator = tqdm(segment_records, desc="Finding chunk intersections")
    for record in iterator:
        wraps, segment_stats = _assign_segment_to_chunks(
            segment_idx=record.segment_idx,
            seg=record.segment,
            chunk_starts=chunk_starts,
            crop_size=target_size,
            min_points_per_wrap=int(min_points_per_wrap),
            bbox_pad_2d=int(bbox_pad_2d),
            terminal_exclusions=terminal_exclusions_by_segment[int(record.segment_idx)],
            chunk_pad=float(chunk_pad),
        )
        stats["wraps_dropped_terminal_start"] += segment_stats["terminal_start"]
        stats["wraps_dropped_terminal_end"] += segment_stats["terminal_end"]
        stats["wraps_dropped_terminal_both"] += segment_stats["terminal_both"]
        terminal_dropped_chunk_ids.update(segment_stats.get("terminal_chunk_ids", set()))
        for wrap in wraps:
            wrap["wrap_label"] = int(record.wrap_label)
            chunk_wraps[tuple(wrap["chunk_id"])].append(wrap)

    stats["chunks_dropped_after_terminal_filter"] = int(
        len(terminal_dropped_chunk_ids.difference(set(chunk_wraps.keys())))
    )
    chunks = []
    for chunk_id in tqdm(sorted(chunk_wraps), desc="Building neighbor sets"):
        cz, cy, cx = chunk_id
        z0 = int(chunk_starts[0][cz])
        y0 = int(chunk_starts[1][cy])
        x0 = int(chunk_starts[2][cx])
        wraps = sorted(
            chunk_wraps[chunk_id],
            key=lambda w: (int(w["wrap_label"]), int(w["segment_idx"]), tuple(int(v) for v in w["bbox_2d"])),
        )
        for wrap_idx, wrap in enumerate(wraps):
            wrap["wrap_idx"] = int(wrap_idx)
            wrap["wrap_id"] = int(wrap["wrap_label"])
        neighbor_sets = {}
        for wrap_idx in range(len(wraps)):
            neighbors = _neighbor_indices(wraps, wrap_idx)
            if neighbors:
                neighbor_sets[str(wrap_idx)] = [int(v) for v in neighbors]

        if str(training_mode) == "copy_neighbors":
            eligible_source_wrap_indices = sorted(int(k) for k in neighbor_sets)
        else:
            eligible_source_wrap_indices = sorted(
                int(k)
                for k, values in neighbor_sets.items()
                if _has_lower_and_upper_neighbors(wraps, int(k), values)
            )

        has_neighboring_wraps = bool(eligible_source_wrap_indices)
        if not has_neighboring_wraps:
            stats["chunks_without_neighboring_wraps"] += 1
        chunks.append({
            "chunk_id": [int(cz), int(cy), int(cx)],
            "bbox_3d": [
                float(z0), float(z0 + target_size[0]),
                float(y0), float(y0 + target_size[1]),
                float(x0), float(x0 + target_size[2]),
            ],
            "wrap_count": int(len(wraps)),
            "has_multiple_wraps": len(wraps) > 1,
            "segment_ids": sorted({segment_records[int(w["segment_idx"])].uuid for w in wraps}),
            "wraps": [
                {
                    "wrap_idx": int(w["wrap_idx"]),
                    "wrap_id": int(w["wrap_id"]),
                    "wrap_label": int(w["wrap_label"]),
                    "segment_id": segment_records[int(w["segment_idx"])].uuid,
                    "segment_idx": int(w["segment_idx"]),
                    "bbox_2d": [int(v) for v in w["bbox_2d"]],
                    "world_bbox": [float(v) for v in w["world_bbox"]],
                    "point_count": int(w["point_count"]),
                }
                for w in wraps
            ],
            "neighbor_sets": neighbor_sets,
            "eligible_source_wrap_indices": [int(v) for v in eligible_source_wrap_indices],
            "has_neighboring_wraps": bool(has_neighboring_wraps),
            "no_neighboring_wraps": bool(not has_neighboring_wraps),
            "stats": {
                "source_wraps_with_neighbors": int(len(neighbor_sets)),
                "source_wraps_with_lower_and_upper_neighbors": int(
                    sum(_has_lower_and_upper_neighbors(wraps, int(k), v) for k, v in neighbor_sets.items())
                ),
                "source_wraps_with_at_least_2_neighbors": int(sum(len(v) >= 2 for v in neighbor_sets.values())),
                "source_wraps_with_at_least_3_neighbors": int(sum(len(v) >= 3 for v in neighbor_sets.values())),
            },
        })

    if cache_file is not None:
        for _ in tqdm(range(1), desc="Serializing chunk cache"):
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "w", encoding="utf8") as f:
                json.dump({
                    "patch_finder_version": PATCH_FINDER_VERSION,
                    "cache_key": cache_key,
                    "stats": stats,
                    "chunks": chunks,
                }, f, indent=2)
    return chunks
