import hashlib
import json
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product

import numpy as np
from tqdm.auto import tqdm

import vesuvius.tifxyz as tifxyz
from common import load_segment_label_masks, open_zarr
from vesuvius.neural_tracing.datasets.common import (
    _parse_z_range,
    _segment_overlaps_z_range,
)


DEFAULT_STORED_GRID_PAD = 40


class _LazyRetargetedTifxyzSegment:
    """Load and retarget a tifxyz segment only when a sample touches it."""

    def __init__(
        self,
        *,
        path,
        uuid,
        scale_yx,
        bbox,
        retarget_factor,
        volume,
    ):
        self.path = path
        self.uuid = str(uuid)
        self.bbox = None if bbox is None else tuple(float(v) for v in bbox)
        self._scale = (
            float(scale_yx[0]) * float(retarget_factor),
            float(scale_yx[1]) * float(retarget_factor),
        )
        self.resolution = "stored"
        self._retarget_factor = float(retarget_factor)
        self._volume = volume
        self._loaded = None

    def _ensure_loaded(self):
        try:
            loaded = object.__getattribute__(self, "_loaded")
        except AttributeError:
            loaded = None
            object.__setattr__(self, "_loaded", None)
        if loaded is not None:
            return loaded

        path = object.__getattribute__(self, "path")
        volume = object.__getattribute__(self, "_volume")
        retarget_factor = object.__getattribute__(self, "_retarget_factor")
        resolution = object.__getattribute__(self, "resolution")
        scale = object.__getattribute__(self, "_scale")

        loaded = tifxyz.read_tifxyz(path)
        loaded.volume = volume
        if retarget_factor != 1.0:
            loaded = loaded.retarget(retarget_factor)
        elif getattr(loaded, "volume", None) is None:
            loaded.volume = volume

        if resolution == "full":
            loaded.use_full_resolution()
        else:
            loaded.use_stored_resolution()

        object.__setattr__(self, "_loaded", loaded)
        object.__setattr__(self, "bbox", loaded.bbox)
        object.__setattr__(self, "_scale", getattr(loaded, "_scale", scale))
        return loaded

    def use_stored_resolution(self):
        object.__setattr__(self, "resolution", "stored")
        try:
            loaded = object.__getattribute__(self, "_loaded")
        except AttributeError:
            loaded = None
            object.__setattr__(self, "_loaded", None)
        if loaded is not None:
            loaded.use_stored_resolution()
        return self

    def use_full_resolution(self):
        object.__setattr__(self, "resolution", "full")
        try:
            loaded = object.__getattribute__(self, "_loaded")
        except AttributeError:
            loaded = None
            object.__setattr__(self, "_loaded", None)
        if loaded is not None:
            loaded.use_full_resolution()
        return self

    def __getstate__(self):
        state = dict(object.__getattribute__(self, "__dict__"))
        state["_loaded"] = None
        return state

    def __setstate__(self, state):
        object.__getattribute__(self, "__dict__").update(state)

    def __getitem__(self, key):
        return self._ensure_loaded()[key]

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(
                f"{type(self).__name__!s} object has no attribute {name!r}"
            )
        return getattr(self._ensure_loaded(), name)


def _empty_points():
    return np.zeros((0, 3), dtype=np.float32)


def _dataset_progress_desc(prefix, dataset):
    return f"{prefix} ({dataset['segments_path']})"


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


def _inclusive_bbox_from_bboxes(world_bboxes):
    mins = []
    maxs = []
    for world_bbox in world_bboxes:
        if world_bbox is None:
            continue
        z0, z1, y0, y1, x0, x1 = (int(v) for v in world_bbox)
        mins.append((z0, y0, x0))
        maxs.append((z1, y1, x1))
    if not mins:
        return None

    overall_min = np.min(np.asarray(mins, dtype=np.int64), axis=0)
    overall_max = np.max(np.asarray(maxs, dtype=np.int64), axis=0)
    return (
        int(overall_min[0]),
        int(overall_max[0]),
        int(overall_min[1]),
        int(overall_max[1]),
        int(overall_min[2]),
        int(overall_max[2]),
    )


def _bbox_intersects(lhs_bbox, rhs_bbox):
    if lhs_bbox is None or rhs_bbox is None:
        return False

    lhs_z0, lhs_z1, lhs_y0, lhs_y1, lhs_x0, lhs_x1 = (int(v) for v in lhs_bbox)
    rhs_z0, rhs_z1, rhs_y0, rhs_y1, rhs_x0, rhs_x1 = (int(v) for v in rhs_bbox)
    return not (
        lhs_z1 < rhs_z0
        or rhs_z1 < lhs_z0
        or lhs_y1 < rhs_y0
        or rhs_y1 < lhs_y0
        or lhs_x1 < rhs_x0
        or rhs_x1 < lhs_x0
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


def _candidate_bbox_start_grid(world_bbox, patch_size_zyx, overlap_fraction):
    z0, z1, y0, y1, x0, x1 = (int(v) for v in world_bbox)
    patch_size_zyx = np.asarray(patch_size_zyx, dtype=np.int64).reshape(3)

    z_starts = np.asarray(
        _axis_patch_starts(z0, z1, int(patch_size_zyx[0]), overlap_fraction),
        dtype=np.int64,
    )
    y_starts = np.asarray(
        _axis_patch_starts(y0, y1, int(patch_size_zyx[1]), overlap_fraction),
        dtype=np.int64,
    )
    x_starts = np.asarray(
        _axis_patch_starts(x0, x1, int(patch_size_zyx[2]), overlap_fraction),
        dtype=np.int64,
    )
    return z_starts, y_starts, x_starts


def _generate_sliding_bboxes(world_bbox, patch_size_zyx, overlap_fraction):
    patch_size_zyx = np.asarray(patch_size_zyx, dtype=np.int64).reshape(3)
    z_starts, y_starts, x_starts = _candidate_bbox_start_grid(
        world_bbox,
        patch_size_zyx,
        overlap_fraction,
    )

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


def _axis_candidate_index_range(axis_starts, axis_bbox_min, axis_bbox_max, patch_size):
    if axis_starts.size == 0:
        return None

    lower_bound = int(axis_bbox_min) - int(patch_size) + 1
    start_idx = int(np.searchsorted(axis_starts, lower_bound, side="left"))
    stop_idx = int(np.searchsorted(axis_starts, int(axis_bbox_max), side="right"))
    if start_idx >= stop_idx:
        return None
    return start_idx, stop_idx


def _candidate_bbox_overlap_ranges(
    world_bbox,
    *,
    z_starts,
    y_starts,
    x_starts,
    patch_size_zyx,
):
    if world_bbox is None:
        return None

    z_range = _axis_candidate_index_range(
        z_starts,
        world_bbox[0],
        world_bbox[1],
        int(patch_size_zyx[0]),
    )
    y_range = _axis_candidate_index_range(
        y_starts,
        world_bbox[2],
        world_bbox[3],
        int(patch_size_zyx[1]),
    )
    x_range = _axis_candidate_index_range(
        x_starts,
        world_bbox[4],
        world_bbox[5],
        int(patch_size_zyx[2]),
    )
    if z_range is None or y_range is None or x_range is None:
        return None
    return z_range, y_range, x_range


def _add_segment_to_candidate_index(
    candidate_segment_indices,
    overlap_ranges,
    *,
    segment_list_idx,
    y_count,
    x_count,
):
    if overlap_ranges is None:
        return

    (z_start, z_stop), (y_start, y_stop), (x_start, x_stop) = overlap_ranges
    plane_stride = int(y_count) * int(x_count)
    for z_idx in range(z_start, z_stop):
        z_offset = z_idx * plane_stride
        for y_idx in range(y_start, y_stop):
            row_offset = z_offset + (y_idx * int(x_count))
            for x_idx in range(x_start, x_stop):
                flat_idx = row_offset + x_idx
                bbox_segments = candidate_segment_indices[flat_idx]
                if bbox_segments is None:
                    candidate_segment_indices[flat_idx] = [int(segment_list_idx)]
                else:
                    bbox_segments.append(int(segment_list_idx))


def _mark_candidate_positive_mask(
    candidate_has_positive,
    overlap_ranges,
    *,
    y_count,
    x_count,
):
    if overlap_ranges is None:
        return

    (z_start, z_stop), (y_start, y_stop), (x_start, x_stop) = overlap_ranges
    plane_stride = int(y_count) * int(x_count)
    x_width = x_stop - x_start
    for z_idx in range(z_start, z_stop):
        z_offset = z_idx * plane_stride
        for y_idx in range(y_start, y_stop):
            row_offset = z_offset + (y_idx * int(x_count)) + x_start
            candidate_has_positive[row_offset : row_offset + x_width] = True


def _build_candidate_bbox_segment_index(
    segment_records,
    *,
    z_starts,
    y_starts,
    x_starts,
    patch_size_zyx,
):
    candidate_count = int(z_starts.size * y_starts.size * x_starts.size)
    candidate_segment_indices = [None] * candidate_count
    candidate_has_positive = np.zeros(candidate_count, dtype=bool)
    y_count = int(y_starts.size)
    x_count = int(x_starts.size)

    for segment_list_idx, segment_record in enumerate(segment_records):
        valid_overlap_ranges = _candidate_bbox_overlap_ranges(
            segment_record["valid_world_bbox"],
            z_starts=z_starts,
            y_starts=y_starts,
            x_starts=x_starts,
            patch_size_zyx=patch_size_zyx,
        )
        _add_segment_to_candidate_index(
            candidate_segment_indices,
            valid_overlap_ranges,
            segment_list_idx=segment_list_idx,
            y_count=y_count,
            x_count=x_count,
        )

        positive_overlap_ranges = _candidate_bbox_overlap_ranges(
            segment_record["positive_world_bbox"],
            z_starts=z_starts,
            y_starts=y_starts,
            x_starts=x_starts,
            patch_size_zyx=patch_size_zyx,
        )
        _mark_candidate_positive_mask(
            candidate_has_positive,
            positive_overlap_ranges,
            y_count=y_count,
            x_count=x_count,
        )

    return candidate_segment_indices, candidate_has_positive


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


def _slice_bounds_from_minmax_rows_cols(row_min, row_max, col_min, col_max, shape, pad):
    row_count, col_count = (int(shape[0]), int(shape[1]))
    pad = max(0, int(pad))
    row_start = max(0, int(row_min) - pad)
    row_stop = min(row_count, int(row_max) + pad + 1)
    col_start = max(0, int(col_min) - pad)
    col_stop = min(col_count, int(col_max) + pad + 1)
    return (row_start, row_stop, col_start, col_stop)


def _axis_point_candidate_index_ranges(axis_starts, axis_points, patch_size):
    axis_points = np.asarray(axis_points, dtype=np.float32)
    start_idx = np.searchsorted(
        axis_starts,
        axis_points - float(patch_size),
        side="right",
    ).astype(np.int32, copy=False)
    stop_idx = np.searchsorted(
        axis_starts,
        axis_points,
        side="right",
    ).astype(np.int32, copy=False)
    return start_idx, stop_idx


def _accumulate_segment_patch_hits(
    segment_record,
    *,
    z_starts,
    y_starts,
    x_starts,
    patch_size_zyx,
    stored_grid_pad,
):
    valid_points_zyx = segment_record["valid_points_zyx"]
    if valid_points_zyx.size == 0:
        return {}

    patch_size_zyx = np.asarray(patch_size_zyx, dtype=np.int64).reshape(3)
    valid_rows = segment_record["valid_rows"]
    valid_cols = segment_record["valid_cols"]
    plane_stride = int(y_starts.size) * int(x_starts.size)
    x_count = int(x_starts.size)
    segment_hits = {}

    valid_z_start_idx, valid_z_stop_idx = _axis_point_candidate_index_ranges(
        z_starts,
        valid_points_zyx[:, 0],
        int(patch_size_zyx[0]),
    )
    valid_y_start_idx, valid_y_stop_idx = _axis_point_candidate_index_ranges(
        y_starts,
        valid_points_zyx[:, 1],
        int(patch_size_zyx[1]),
    )
    valid_x_start_idx, valid_x_stop_idx = _axis_point_candidate_index_ranges(
        x_starts,
        valid_points_zyx[:, 2],
        int(patch_size_zyx[2]),
    )

    for point_idx in range(valid_points_zyx.shape[0]):
        z_start = int(valid_z_start_idx[point_idx])
        z_stop = int(valid_z_stop_idx[point_idx])
        y_start = int(valid_y_start_idx[point_idx])
        y_stop = int(valid_y_stop_idx[point_idx])
        x_start = int(valid_x_start_idx[point_idx])
        x_stop = int(valid_x_stop_idx[point_idx])
        if z_start >= z_stop or y_start >= y_stop or x_start >= x_stop:
            continue

        row = int(valid_rows[point_idx])
        col = int(valid_cols[point_idx])
        for z_idx in range(z_start, z_stop):
            z_offset = z_idx * plane_stride
            for y_idx in range(y_start, y_stop):
                row_offset = z_offset + (y_idx * x_count)
                for x_idx in range(x_start, x_stop):
                    flat_idx = row_offset + x_idx
                    hit = segment_hits.get(flat_idx)
                    if hit is None:
                        segment_hits[flat_idx] = [1, row, row, col, col, 0]
                        continue
                    hit[0] += 1
                    if row < hit[1]:
                        hit[1] = row
                    if row > hit[2]:
                        hit[2] = row
                    if col < hit[3]:
                        hit[3] = col
                    if col > hit[4]:
                        hit[4] = col

    positive_points_zyx = segment_record["positive_points_zyx"]
    if positive_points_zyx.size != 0:
        positive_z_start_idx, positive_z_stop_idx = _axis_point_candidate_index_ranges(
            z_starts,
            positive_points_zyx[:, 0],
            int(patch_size_zyx[0]),
        )
        positive_y_start_idx, positive_y_stop_idx = _axis_point_candidate_index_ranges(
            y_starts,
            positive_points_zyx[:, 1],
            int(patch_size_zyx[1]),
        )
        positive_x_start_idx, positive_x_stop_idx = _axis_point_candidate_index_ranges(
            x_starts,
            positive_points_zyx[:, 2],
            int(patch_size_zyx[2]),
        )

        for point_idx in range(positive_points_zyx.shape[0]):
            z_start = int(positive_z_start_idx[point_idx])
            z_stop = int(positive_z_stop_idx[point_idx])
            y_start = int(positive_y_start_idx[point_idx])
            y_stop = int(positive_y_stop_idx[point_idx])
            x_start = int(positive_x_start_idx[point_idx])
            x_stop = int(positive_x_stop_idx[point_idx])
            if z_start >= z_stop or y_start >= y_stop or x_start >= x_stop:
                continue

            for z_idx in range(z_start, z_stop):
                z_offset = z_idx * plane_stride
                for y_idx in range(y_start, y_stop):
                    row_offset = z_offset + (y_idx * x_count)
                    for x_idx in range(x_start, x_stop):
                        flat_idx = row_offset + x_idx
                        hit = segment_hits.get(flat_idx)
                        if hit is not None:
                            hit[5] += 1

    patch_segments = {}
    for flat_idx, (
        valid_count,
        row_min,
        row_max,
        col_min,
        col_max,
        positive_count,
    ) in segment_hits.items():
        patch_segments[flat_idx] = {
            "segment_idx": int(segment_record["segment_idx"]),
            "segment_uuid": str(segment_record["segment_uuid"]),
            "segment": segment_record["segment"],
            "ink_label_path": segment_record["ink_label_path"],
            "stored_rowcol_bounds": _slice_bounds_from_minmax_rows_cols(
                row_min,
                row_max,
                col_min,
                col_max,
                segment_record["grid_shape"],
                stored_grid_pad,
            ),
            "valid_point_count": int(valid_count),
            "positive_point_count": int(positive_count),
            "has_positive_points": bool(positive_count > 0),
        }
    return patch_segments


def _world_bbox_from_flat_candidate_idx(flat_idx, *, z_starts, y_starts, x_starts, patch_size_zyx):
    x_count = int(x_starts.size)
    yz_stride = int(y_starts.size) * x_count
    z_idx = int(flat_idx // yz_stride)
    yz_remainder = int(flat_idx % yz_stride)
    y_idx = int(yz_remainder // x_count)
    x_idx = int(yz_remainder % x_count)
    start_z = int(z_starts[z_idx])
    start_y = int(y_starts[y_idx])
    start_x = int(x_starts[x_idx])
    return (
        start_z,
        int(start_z + int(patch_size_zyx[0]) - 1),
        start_y,
        int(start_y + int(patch_size_zyx[1]) - 1),
        start_x,
        int(start_x + int(patch_size_zyx[2]) - 1),
    )


def _build_patch_segment_entry(segment_record, world_bbox, stored_grid_pad):
    valid_world_bbox = segment_record["valid_world_bbox"]
    if not _bbox_intersects(world_bbox, valid_world_bbox):
        return None

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

    positive_world_bbox = segment_record["positive_world_bbox"]
    if _bbox_intersects(world_bbox, positive_world_bbox):
        positive_in_bbox = _points_in_world_bbox(segment_record["positive_points_zyx"], world_bbox)
        positive_count = int(np.count_nonzero(positive_in_bbox))
    else:
        positive_count = 0

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


def _resolve_min_positive_point_count(config):
    value = config.get("min_positive_point_count", 1)
    min_positive_point_count = int(value)
    if min_positive_point_count < 1:
        raise ValueError(
            f"min_positive_point_count must be >= 1, got {value!r}"
        )
    return min_positive_point_count


def _supervised_segment_indices_for_patch_segments(
    patch_segments,
    *,
    min_positive_point_count,
):
    min_positive_point_count = int(min_positive_point_count)
    return tuple(
        patch_segment_idx
        for patch_segment_idx, patch_segment in enumerate(patch_segments)
        if int(patch_segment.get("positive_point_count", 0)) >= min_positive_point_count
    )


def _build_patch_record_for_bbox(
    *,
    dataset_idx,
    volume,
    volume_scale,
    world_bbox,
    segment_records,
    stored_grid_pad,
    min_positive_point_count,
    segment_indices=None,
    has_positive_candidate=True,
):
    if segment_indices is None:
        candidate_segment_records = segment_records
    elif not segment_indices:
        return None, "without_points"
    elif not has_positive_candidate:
        return None, "without_positive"
    else:
        candidate_segment_records = (
            segment_records[int(segment_idx)] for segment_idx in segment_indices
        )

    patch_segments = []
    for segment_record in candidate_segment_records:
        patch_segment = _build_patch_segment_entry(
            segment_record,
            world_bbox,
            stored_grid_pad,
        )
        if patch_segment is None:
            continue
        patch_segments.append(patch_segment)

    if not patch_segments:
        return None, "without_points"
    supervised_segment_indices = _supervised_segment_indices_for_patch_segments(
        patch_segments,
        min_positive_point_count=min_positive_point_count,
    )
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
    min_positive_point_count,
    segment_indices_by_bbox,
    has_positive_candidates,
):
    chunk_patches = []
    chunk_stats = {
        "rejected_without_points": 0,
        "rejected_without_positive": 0,
    }
    for world_bbox, segment_indices, has_positive_candidate in zip(
        world_bboxes,
        segment_indices_by_bbox,
        has_positive_candidates,
    ):
        patch_record, rejection_reason = _build_patch_record_for_bbox(
            dataset_idx=dataset_idx,
            volume=volume,
            volume_scale=volume_scale,
            world_bbox=world_bbox,
            segment_records=segment_records,
            stored_grid_pad=stored_grid_pad,
            min_positive_point_count=min_positive_point_count,
            segment_indices=segment_indices,
            has_positive_candidate=has_positive_candidate,
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
    dataset,
    volume,
    volume_scale,
    segment_records,
    patch_size_zyx,
    overlap_fraction,
    patch_finding_workers,
    stored_grid_pad,
    min_positive_point_count,
):
    stats = {
        "candidate_bboxes": 0,
        "rejected_without_points": 0,
        "rejected_without_positive": 0,
        "kept_patches": 0,
    }
    union_bbox = _inclusive_bbox_from_bboxes(
        record["labeled_world_bbox"] for record in segment_records
    )
    if union_bbox is None:
        return [], stats

    z_starts, y_starts, x_starts = _candidate_bbox_start_grid(
        union_bbox,
        patch_size_zyx,
        overlap_fraction,
    )
    patch_size_zyx = np.asarray(patch_size_zyx, dtype=np.int64).reshape(3)
    stats["candidate_bboxes"] = int(z_starts.size * y_starts.size * x_starts.size)
    bbox_to_segments = {}
    segment_iter = tqdm(
        segment_records,
        total=len(segment_records),
        desc=_dataset_progress_desc("Accumulating patches", dataset),
        unit="segment",
        leave=False,
    )
    for segment_record in segment_iter:
        segment_patch_hits = _accumulate_segment_patch_hits(
            segment_record,
            z_starts=z_starts,
            y_starts=y_starts,
            x_starts=x_starts,
            patch_size_zyx=patch_size_zyx,
            stored_grid_pad=stored_grid_pad,
        )
        for flat_idx, patch_segment in segment_patch_hits.items():
            patch_segments = bbox_to_segments.get(flat_idx)
            if patch_segments is None:
                bbox_to_segments[flat_idx] = [patch_segment]
            else:
                patch_segments.append(patch_segment)

    dataset_patches = []
    stats["rejected_without_points"] = int(stats["candidate_bboxes"] - len(bbox_to_segments))
    for flat_idx in sorted(bbox_to_segments):
        patch_segments = bbox_to_segments[flat_idx]
        supervised_segment_indices = _supervised_segment_indices_for_patch_segments(
            patch_segments,
            min_positive_point_count=min_positive_point_count,
        )
        if not supervised_segment_indices:
            stats["rejected_without_positive"] += 1
            continue

        dataset_patches.append(
            {
                "dataset_idx": int(dataset_idx),
                "volume": volume,
                "scale": int(volume_scale),
                "world_bbox": _world_bbox_from_flat_candidate_idx(
                    flat_idx,
                    z_starts=z_starts,
                    y_starts=y_starts,
                    x_starts=x_starts,
                    patch_size_zyx=patch_size_zyx,
                ),
                "segments": patch_segments,
                "supervised_segment_indices": supervised_segment_indices,
            }
        )

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
        desc=_dataset_progress_desc("Preparing segments", dataset),
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
        supervision_only_mask = np.zeros_like(valid_mask, dtype=bool)
        labeled_mask = np.zeros_like(valid_mask, dtype=bool)
        ink_label_path = None
        try:
            label_scale_y, label_scale_x = _scale_pair_or_default(original_seg)
            ink_mask, supervision_mask, ink_label_path = load_segment_label_masks(
                original_seg,
                tuple(int(v) for v in original_seg.full_resolution_shape),
            )
            row_idx = np.rint(
                np.arange(grid_shape[0], dtype=np.float32) / label_scale_y
            ).astype(np.int64)
            col_idx = np.rint(
                np.arange(grid_shape[1], dtype=np.float32) / label_scale_x
            ).astype(np.int64)
            assert np.all((row_idx >= 0) & (row_idx < ink_mask.shape[0])), (
                f"Segment {segment_uuid!r} stored rows map outside full-resolution ink labels: "
                f"grid_shape={grid_shape}, label_shape={ink_mask.shape}, scale_y={label_scale_y}"
            )
            assert np.all((col_idx >= 0) & (col_idx < ink_mask.shape[1])), (
                f"Segment {segment_uuid!r} stored cols map outside full-resolution ink labels: "
                f"grid_shape={grid_shape}, label_shape={ink_mask.shape}, scale_x={label_scale_x}"
            )
            ink_mask = ink_mask[row_idx[:, None], col_idx[None, :]]
            supervision_mask = supervision_mask[row_idx[:, None], col_idx[None, :]]
            positive_mask = valid_mask & ink_mask
            supervision_only_mask = valid_mask & supervision_mask
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
        supervision_points_zyx = _points_from_mask(
            z_stored,
            y_stored,
            x_stored,
            supervision_only_mask,
            retarget_factor,
        )
        valid_world_bbox = _inclusive_bbox_from_point_sets((valid_points_zyx,))
        labeled_world_bbox = _inclusive_bbox_from_point_sets((labeled_points_zyx,))
        positive_world_bbox = _inclusive_bbox_from_point_sets((positive_points_zyx,))
        supervision_world_bbox = _inclusive_bbox_from_point_sets((supervision_points_zyx,))

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
                "valid_world_bbox": valid_world_bbox,
                "labeled_world_bbox": labeled_world_bbox,
                "positive_world_bbox": positive_world_bbox,
                "supervision_world_bbox": supervision_world_bbox,
                "valid_points_zyx": valid_points_zyx,
                "positive_points_zyx": positive_points_zyx,
                "supervision_points_zyx": supervision_points_zyx,
            }
        )

    return segment_records


def _cached_segment_uuids(cache_entry):
    segment_uuids = set()
    for patch in cache_entry.get("patches", []):
        for segment in patch.get("segments", []):
            segment_uuids.add(str(segment["segment_uuid"]))
    return segment_uuids


def _build_cached_segment_records(
    *,
    dataset,
    volume,
    volume_scale,
    cache_entry,
):
    required_segment_uuids = _cached_segment_uuids(cache_entry)
    if not required_segment_uuids:
        return {}

    segment_infos = tifxyz.list_tifxyz(
        dataset["segments_path"],
        z_range=_parse_z_range(dataset.get("z_range", None)),
    )
    retarget_factor = 2 ** int(volume_scale)
    segment_by_uuid = {}
    for segment_idx, segment_info in enumerate(segment_infos):
        segment_uuid = str(segment_info.uuid)
        if segment_uuid not in required_segment_uuids:
            continue

        segment_by_uuid[segment_uuid] = {
            "segment_idx": int(segment_idx),
            "segment_uuid": segment_uuid,
            "segment": _LazyRetargetedTifxyzSegment(
                path=segment_info.path,
                uuid=segment_uuid,
                scale_yx=segment_info.scale,
                bbox=segment_info.bbox,
                retarget_factor=retarget_factor,
                volume=volume,
            ),
            "ink_label_path": None,
        }
    return segment_by_uuid


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


def _load_cached_patches(
    cache_entry,
    *,
    dataset_idx,
    volume,
    volume_scale,
    segment_by_uuid,
    min_positive_point_count,
):
    cached_patches = []
    for record in cache_entry.get("patches", []):
        patch_segments = []
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
            patch_segments.append(patch_segment)

        supervised_segment_indices = _supervised_segment_indices_for_patch_segments(
            patch_segments,
            min_positive_point_count=min_positive_point_count,
        )
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
    min_positive_point_count = _resolve_min_positive_point_count(config)
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
                segment_by_uuid = _build_cached_segment_records(
                    dataset=dataset,
                    volume=volume,
                    volume_scale=volume_scale,
                    cache_entry=cache_entry,
                )
                cache_patches = _load_cached_patches(
                    cache_entry,
                    dataset_idx=dataset_idx,
                    volume=volume,
                    volume_scale=volume_scale,
                    segment_by_uuid=segment_by_uuid,
                    min_positive_point_count=min_positive_point_count,
                )
                patches.extend(cache_patches)
                patch_generation_stats["kept_patches"] += int(len(cache_patches))
                patch_generation_stats["cache_hits"] += 1
                continue

        segment_records = _prepare_segment_records(
            dataset_idx=dataset_idx,
            dataset=dataset,
            volume=volume,
            volume_scale=volume_scale,
            patch_generation_stats=patch_generation_stats,
        )

        dataset_patches, dataset_stats = _build_dataset_patch_records(
            dataset_idx=dataset_idx,
            dataset=dataset,
            volume=volume,
            volume_scale=volume_scale,
            segment_records=segment_records,
            patch_size_zyx=patch_size_zyx,
            overlap_fraction=overlap_fraction,
            patch_finding_workers=patch_finding_workers,
            stored_grid_pad=stored_grid_pad,
            min_positive_point_count=min_positive_point_count,
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
