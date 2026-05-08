"""
World-Chunk Patch Tiling

A world-first patch tiling method that avoids "strip" crops for thin,
curvilinear surfaces by binning surface points into fixed 3D chunks, then
deriving row/col patches from the points that actually fall inside each chunk.

This method operates at the dataset level: the world chunk grid is defined by
the bbox of all segments assigned to the same volume.
"""

from __future__ import annotations

import json
import hashlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numba import njit
from scipy import ndimage
from tqdm import tqdm

from vesuvius.tifxyz import Tifxyz


# =============================================================================
# Helper functions
# =============================================================================


def compute_dataset_bbox(segments: List[Tifxyz]) -> Tuple[float, ...]:
    """
    Compute the union bounding box of all segments in ZYX order.

    Parameters
    ----------
    segments : List[Tifxyz]
        List of segments (should already be retargeted to volume scale).

    Returns
    -------
    Tuple[float, ...]
        (z_min, z_max, y_min, y_max, x_min, x_max) in world coordinates.
    """
    z_mins, z_maxs = [], []
    y_mins, y_maxs = [], []
    x_mins, x_maxs = [], []

    for seg in segments:
        valid = seg._valid_mask
        if not valid.any():
            continue
        # Segment bbox is in XYZ order: (x_min, y_min, z_min, x_max, y_max, z_max)
        if seg.bbox is not None:
            x_min, y_min, z_min, x_max, y_max, z_max = seg.bbox
        else:
            # Compute from coordinates
            x_min = float(seg._x[valid].min())
            y_min = float(seg._y[valid].min())
            z_min = float(seg._z[valid].min())
            x_max = float(seg._x[valid].max())
            y_max = float(seg._y[valid].max())
            z_max = float(seg._z[valid].max())

        z_mins.append(z_min)
        z_maxs.append(z_max)
        y_mins.append(y_min)
        y_maxs.append(y_max)
        x_mins.append(x_min)
        x_maxs.append(x_max)

    if not z_mins:
        raise ValueError("No valid segments provided")

    return (
        min(z_mins),
        max(z_maxs),
        min(y_mins),
        max(y_maxs),
        min(x_mins),
        max(x_maxs),
    )


def build_chunk_grid(
    dataset_bbox_zyx: Tuple[float, ...],
    target_size: Tuple[int, int, int],
    overlap_fraction: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int, int], Tuple[int, int, int]]:
    """
    Build the 3D chunk grid covering the dataset bbox.

    Parameters
    ----------
    dataset_bbox_zyx : Tuple[float, ...]
        (z_min, z_max, y_min, y_max, x_min, x_max)
    target_size : Tuple[int, int, int]
        (chunk_d, chunk_h, chunk_w) - chunk dimensions in each axis
    overlap_fraction : float
        Overlap fraction between chunks (0.0 to <1.0)

    Returns
    -------
    Tuple containing:
        - chunk_starts_z: array of z start coordinates
        - chunk_starts_y: array of y start coordinates
        - chunk_starts_x: array of x start coordinates
        - strides: (stride_d, stride_h, stride_w)
        - n_chunks: (n_z, n_y, n_x)
    """
    z_min, z_max, y_min, y_max, x_min, x_max = dataset_bbox_zyx
    chunk_d, chunk_h, chunk_w = target_size

    # Compute strides
    stride_d = max(1, int(chunk_d * (1 - overlap_fraction)))
    stride_h = max(1, int(chunk_h * (1 - overlap_fraction)))
    stride_w = max(1, int(chunk_w * (1 - overlap_fraction)))

    # Build chunk start positions
    # Ensure coverage of entire bbox by adding final chunk if needed
    def make_starts(origin, extent, chunk_size, stride):
        starts = []
        pos = origin
        while pos < extent:
            starts.append(pos)
            pos += stride
        # Ensure the last chunk covers the end
        if len(starts) > 0:
            last_end = starts[-1] + chunk_size
            if last_end < extent:
                # Add a final chunk starting at extent - chunk_size
                final_start = max(origin, extent - chunk_size)
                if final_start > starts[-1]:
                    starts.append(final_start)
        elif extent > origin:
            # Edge case: bbox smaller than chunk_size
            starts.append(origin)
        return np.array(starts, dtype=np.float64)

    chunk_starts_z = make_starts(z_min, z_max, chunk_d, stride_d)
    chunk_starts_y = make_starts(y_min, y_max, chunk_h, stride_h)
    chunk_starts_x = make_starts(x_min, x_max, chunk_w, stride_w)

    n_z = len(chunk_starts_z)
    n_y = len(chunk_starts_y)
    n_x = len(chunk_starts_x)

    return (
        chunk_starts_z,
        chunk_starts_y,
        chunk_starts_x,
        (stride_d, stride_h, stride_w),
        (n_z, n_y, n_x),
    )


def get_chunks_containing_point(
    z: float, y: float, x: float,
    chunk_starts_z: np.ndarray,
    chunk_starts_y: np.ndarray,
    chunk_starts_x: np.ndarray,
    target_size: Tuple[int, int, int],
) -> List[Tuple[int, int, int]]:
    """
    Get all chunk indices that contain the given point.

    A point is inside a chunk if its coordinate is in [start, start + size).

    Parameters
    ----------
    z, y, x : float
        Point coordinates
    chunk_starts_z/y/x : np.ndarray
        Arrays of chunk start positions
    target_size : Tuple[int, int, int]
        (chunk_d, chunk_h, chunk_w)

    Returns
    -------
    List[Tuple[int, int, int]]
        List of (iz, iy, ix) chunk indices containing the point.
    """
    chunk_d, chunk_h, chunk_w = target_size

    # Find all chunks where point is in [start, start+size)
    iz_valid = np.where((z >= chunk_starts_z) & (z < chunk_starts_z + chunk_d))[0]
    iy_valid = np.where((y >= chunk_starts_y) & (y < chunk_starts_y + chunk_h))[0]
    ix_valid = np.where((x >= chunk_starts_x) & (x < chunk_starts_x + chunk_w))[0]

    # Return all combinations
    return [(int(iz), int(iy), int(ix))
            for iz in iz_valid for iy in iy_valid for ix in ix_valid]


@njit
def _searchsorted_right(arr: np.ndarray, value: float) -> int:
    """Numba-compatible searchsorted(..., side="right")."""
    lo = 0
    hi = len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if value < arr[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo


@njit
def _find_containing_chunk_bounds_1d(
    coord: float,
    chunk_starts: np.ndarray,
    chunk_size: int,
    pad: float = 0.0,
) -> Tuple[int, int]:
    """Return [start, stop) chunk-index bounds containing coord on one axis."""
    lower_exclusive = coord - chunk_size - pad
    upper_inclusive = coord + pad
    start = _searchsorted_right(chunk_starts, lower_exclusive)
    stop = _searchsorted_right(chunk_starts, upper_inclusive)
    return start, stop


@njit
def _count_chunk_indices_for_points(
    z_flat: np.ndarray,
    y_flat: np.ndarray,
    x_flat: np.ndarray,
    chunk_starts_z: np.ndarray,
    chunk_starts_y: np.ndarray,
    chunk_starts_x: np.ndarray,
    chunk_d: int,
    chunk_h: int,
    chunk_w: int,
    chunk_pad: float = 0.0,
) -> int:
    pair_count = 0

    for pt_idx in range(len(z_flat)):
        z = z_flat[pt_idx]
        y = y_flat[pt_idx]
        x = x_flat[pt_idx]

        if not (np.isfinite(z) and np.isfinite(y) and np.isfinite(x)):
            continue

        iz_start, iz_stop = _find_containing_chunk_bounds_1d(
            z, chunk_starts_z, chunk_d, chunk_pad
        )
        iy_start, iy_stop = _find_containing_chunk_bounds_1d(
            y, chunk_starts_y, chunk_h, chunk_pad
        )
        ix_start, ix_stop = _find_containing_chunk_bounds_1d(
            x, chunk_starts_x, chunk_w, chunk_pad
        )

        pair_count += (
            (iz_stop - iz_start) *
            (iy_stop - iy_start) *
            (ix_stop - ix_start)
        )

    return pair_count


@njit
def _get_chunk_indices_for_points(
    z_flat: np.ndarray,
    y_flat: np.ndarray,
    x_flat: np.ndarray,
    chunk_starts_z: np.ndarray,
    chunk_starts_y: np.ndarray,
    chunk_starts_x: np.ndarray,
    chunk_d: int,
    chunk_h: int,
    chunk_w: int,
    chunk_pad: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized chunk assignment for all points.

    Returns arrays of (point_idx, iz, iy, ix) for all point-chunk pairs.
    A point can appear multiple times if it belongs to overlapping chunks.

    With chunk_pad > 0, points within `pad` distance of chunk boundaries
    are also assigned to that chunk.
    """
    max_pairs = _count_chunk_indices_for_points(
        z_flat, y_flat, x_flat,
        chunk_starts_z, chunk_starts_y, chunk_starts_x,
        chunk_d, chunk_h, chunk_w,
        chunk_pad,
    )
    point_indices = np.empty(max_pairs, dtype=np.int64)
    iz_arr = np.empty(max_pairs, dtype=np.int32)
    iy_arr = np.empty(max_pairs, dtype=np.int32)
    ix_arr = np.empty(max_pairs, dtype=np.int32)

    pair_count = 0

    for pt_idx in range(len(z_flat)):
        z = z_flat[pt_idx]
        y = y_flat[pt_idx]
        x = x_flat[pt_idx]

        # Skip non-finite
        if not (np.isfinite(z) and np.isfinite(y) and np.isfinite(x)):
            continue

        iz_start, iz_stop = _find_containing_chunk_bounds_1d(
            z, chunk_starts_z, chunk_d, chunk_pad
        )
        iy_start, iy_stop = _find_containing_chunk_bounds_1d(
            y, chunk_starts_y, chunk_h, chunk_pad
        )
        ix_start, ix_stop = _find_containing_chunk_bounds_1d(
            x, chunk_starts_x, chunk_w, chunk_pad
        )

        # Generate all combinations (cartesian product)
        for iz in range(iz_start, iz_stop):
            for iy in range(iy_start, iy_stop):
                for ix in range(ix_start, ix_stop):
                    point_indices[pair_count] = pt_idx
                    iz_arr[pair_count] = iz
                    iy_arr[pair_count] = iy
                    ix_arr[pair_count] = ix
                    pair_count += 1

    return (
        point_indices[:pair_count],
        iz_arr[:pair_count],
        iy_arr[:pair_count],
        ix_arr[:pair_count],
    )


def _linear_chunk_ids(
    iz_arr: np.ndarray,
    iy_arr: np.ndarray,
    ix_arr: np.ndarray,
    n_y: int,
    n_x: int,
) -> np.ndarray:
    return (
        (iz_arr.astype(np.int64, copy=False) * int(n_y) +
         iy_arr.astype(np.int64, copy=False)) * int(n_x) +
        ix_arr.astype(np.int64, copy=False)
    )


def _chunk_id_from_linear(linear_id: int, n_y: int, n_x: int) -> Tuple[int, int, int]:
    yz = int(n_y) * int(n_x)
    iz = int(linear_id // yz)
    rem = int(linear_id - iz * yz)
    iy = int(rem // int(n_x))
    ix = int(rem - iy * int(n_x))
    return iz, iy, ix


def _store_valid_point_indices_by_chunk(
    chunk_to_valid_points: Dict[Tuple[int, int, int], Dict[int, np.ndarray]],
    *,
    seg_idx: int,
    point_indices: np.ndarray,
    iz_arr: np.ndarray,
    iy_arr: np.ndarray,
    ix_arr: np.ndarray,
    valid_pair_mask: np.ndarray,
    n_y: int,
    n_x: int,
) -> None:
    if not np.any(valid_pair_mask):
        return

    linear_ids = _linear_chunk_ids(
        iz_arr[valid_pair_mask],
        iy_arr[valid_pair_mask],
        ix_arr[valid_pair_mask],
        n_y,
        n_x,
    )
    valid_point_indices = point_indices[valid_pair_mask]
    order = np.argsort(linear_ids, kind="mergesort")
    linear_sorted = linear_ids[order]
    point_sorted = np.asarray(valid_point_indices[order], dtype=np.int64)

    boundaries = np.flatnonzero(linear_sorted[1:] != linear_sorted[:-1]) + 1
    starts = np.concatenate((np.array([0], dtype=np.int64), boundaries))
    stops = np.concatenate((boundaries, np.array([len(linear_sorted)], dtype=np.int64)))

    for start, stop in zip(starts, stops):
        chunk_id = _chunk_id_from_linear(int(linear_sorted[start]), n_y, n_x)
        if chunk_id not in chunk_to_valid_points:
            chunk_to_valid_points[chunk_id] = {}
        chunk_to_valid_points[chunk_id][seg_idx] = point_sorted[start:stop]


def _store_invalid_chunk_memberships(
    chunk_to_invalid_segments: Dict[Tuple[int, int, int], set],
    *,
    seg_idx: int,
    iz_arr: np.ndarray,
    iy_arr: np.ndarray,
    ix_arr: np.ndarray,
    invalid_pair_mask: np.ndarray,
    n_y: int,
    n_x: int,
) -> None:
    if not np.any(invalid_pair_mask):
        return

    linear_ids = _linear_chunk_ids(
        iz_arr[invalid_pair_mask],
        iy_arr[invalid_pair_mask],
        ix_arr[invalid_pair_mask],
        n_y,
        n_x,
    )
    for linear_id in np.unique(linear_ids):
        chunk_id = _chunk_id_from_linear(int(linear_id), n_y, n_x)
        if chunk_id not in chunk_to_invalid_segments:
            chunk_to_invalid_segments[chunk_id] = set()
        chunk_to_invalid_segments[chunk_id].add(seg_idx)


def assign_points_to_chunks(
    segments: List[Tifxyz],
    chunk_starts_z: np.ndarray,
    chunk_starts_y: np.ndarray,
    chunk_starts_x: np.ndarray,
    target_size: Tuple[int, int, int],
    verbose: bool = False,
    chunk_pad: float = 0.0,
) -> Tuple[Dict, Dict]:
    """
    Assign all grid cells from all segments to chunks.

    Tracks both valid and invalid cells to support rejection checks.
    Uses vectorized operations with numba for performance.

    Parameters
    ----------
    segments : List[Tifxyz]
        List of segments
    chunk_starts_z/y/x : np.ndarray
        Arrays of chunk start positions
    target_size : Tuple[int, int, int]
        (chunk_d, chunk_h, chunk_w)
    verbose : bool
        If True, show progress bar
    chunk_pad : float
        Padding to expand chunk boundaries when assigning points.
        Points within `chunk_pad` of the boundary are included.

    Returns
    -------
    Tuple[Dict, Dict]
        - chunk_to_valid_points: {chunk_id: {seg_idx: flat_point_indices}}
        - chunk_to_invalid_segments: {chunk_id: {seg_idx, ...}}
    """
    chunk_d, chunk_h, chunk_w = target_size
    chunk_to_valid_points: Dict[Tuple[int, int, int], Dict[int, np.ndarray]] = {}
    chunk_to_invalid_segments: Dict[Tuple[int, int, int], set] = {}
    n_y = len(chunk_starts_y)
    n_x = len(chunk_starts_x)

    # Ensure chunk_starts are float64 for numba compatibility
    chunk_starts_z = np.asarray(chunk_starts_z, dtype=np.float64)
    chunk_starts_y = np.asarray(chunk_starts_y, dtype=np.float64)
    chunk_starts_x = np.asarray(chunk_starts_x, dtype=np.float64)

    seg_iter = tqdm(
        enumerate(segments),
        total=len(segments),
        desc="Assigning points to chunks",
        disable=not verbose,
    )

    for seg_idx, seg in seg_iter:
        z_arr = seg._z
        y_arr = seg._y
        x_arr = seg._x
        valid_mask = seg._valid_mask

        # Flatten arrays
        z_flat = z_arr.ravel().astype(np.float64)
        y_flat = y_arr.ravel().astype(np.float64)
        x_flat = x_arr.ravel().astype(np.float64)
        valid_flat = valid_mask.ravel()

        # Get all (point_idx, chunk_idx) pairs using numba
        point_indices, iz_arr, iy_arr, ix_arr = _get_chunk_indices_for_points(
            z_flat, y_flat, x_flat,
            chunk_starts_z, chunk_starts_y, chunk_starts_x,
            chunk_d, chunk_h, chunk_w,
            chunk_pad,
        )

        if len(point_indices) == 0:
            continue

        valid_pair_mask = valid_flat[point_indices]
        _store_valid_point_indices_by_chunk(
            chunk_to_valid_points,
            seg_idx=seg_idx,
            point_indices=point_indices,
            iz_arr=iz_arr,
            iy_arr=iy_arr,
            ix_arr=ix_arr,
            valid_pair_mask=valid_pair_mask,
            n_y=n_y,
            n_x=n_x,
        )
        _store_invalid_chunk_memberships(
            chunk_to_invalid_segments,
            seg_idx=seg_idx,
            iz_arr=iz_arr,
            iy_arr=iy_arr,
            ix_arr=ix_arr,
            invalid_pair_mask=~valid_pair_mask,
            n_y=n_y,
            n_x=n_x,
        )

    return chunk_to_valid_points, chunk_to_invalid_segments


@njit
def _build_local_mask(
    rows: np.ndarray,
    cols: np.ndarray,
    r_min_all: int,
    c_min_all: int,
    local_h: int,
    local_w: int,
) -> np.ndarray:
    """JIT-compiled mask building for connected components."""
    mask = np.zeros((local_h, local_w), dtype=np.uint8)
    for i in range(len(rows)):
        local_r = rows[i] - r_min_all
        local_c = cols[i] - c_min_all
        mask[local_r, local_c] = 1
    return mask


def _passes_span_check_arrays(
    z: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    chunk_bbox: Tuple[float, ...],
    target_size: Tuple[int, int, int],
    min_span_ratio: float,
    edge_touch_frac: float,
    edge_touch_min_count: int,
    edge_touch_pad: int = 0,
) -> bool:
    if len(z) == 0:
        return False

    z_min, z_max, y_min, y_max, x_min, x_max = chunk_bbox
    chunk_d, chunk_h, chunk_w = target_size

    z_min_t = z_min - edge_touch_pad
    z_max_t = z_max + edge_touch_pad
    y_min_t = y_min - edge_touch_pad
    y_max_t = y_max + edge_touch_pad
    x_min_t = x_min - edge_touch_pad
    x_max_t = x_max + edge_touch_pad

    z_span = z.max() - z.min()
    y_span = y.max() - y.min()
    x_span = x.max() - x.min()

    if z_span < min_span_ratio * chunk_d:
        return False

    if y_span >= x_span:
        second_span = y_span
        second_chunk_size = chunk_h
        second_coords = y
        second_min = y_min_t
        second_max = y_max_t
    else:
        second_span = x_span
        second_chunk_size = chunk_w
        second_coords = x
        second_min = x_min_t
        second_max = x_max_t

    if second_span < min_span_ratio * second_chunk_size:
        return False

    def edge_touch_ok(
        coords: np.ndarray,
        bbox_min: float,
        bbox_max: float,
        chunk_size: int,
    ) -> bool:
        band = edge_touch_frac * chunk_size
        low_count = np.count_nonzero(coords <= bbox_min + band)
        high_count = np.count_nonzero(coords >= bbox_max - band)
        return low_count >= edge_touch_min_count and high_count >= edge_touch_min_count

    return (
        edge_touch_ok(z, z_min_t, z_max_t, chunk_d) and
        edge_touch_ok(second_coords, second_min, second_max, second_chunk_size)
    )


def _passes_inner_bbox_check_arrays(
    y: np.ndarray,
    x: np.ndarray,
    chunk_bbox: Tuple[float, ...],
    inner_bbox_fraction: float,
) -> bool:
    if len(y) == 0:
        return False
    if inner_bbox_fraction >= 1.0:
        return True
    if inner_bbox_fraction <= 0.0:
        return False

    _, _, y_min, y_max, x_min, x_max = chunk_bbox
    margin_frac = (1.0 - inner_bbox_fraction) / 2.0
    y_margin = (y_max - y_min) * margin_frac
    x_margin = (x_max - x_min) * margin_frac

    center_y = float(y.mean())
    center_x = float(x.mean())

    return (
        center_y >= y_min + y_margin and center_y <= y_max - y_margin and
        center_x >= x_min + x_margin and center_x <= x_max - x_margin
    )


def find_wraps_in_chunk(
    point_indices: np.ndarray,
    seg: Tifxyz,
    min_points_per_wrap: int,
    bbox_pad_2d: int,
    require_all_valid_in_bbox: bool,
    chunk_bbox: Tuple[float, ...],
    target_size: Tuple[int, int, int],
    min_span_ratio: float,
    edge_touch_frac: float,
    edge_touch_min_count: int,
    edge_touch_pad: int,
    inner_bbox_fraction: float,
) -> Tuple[List[Dict], Dict[str, int]]:
    """
    Find connected components (wraps) from the given points for a single segment.

    Parameters
    ----------
    point_indices : np.ndarray
        Flat segment-grid indices for this segment in this chunk
    seg : Tifxyz
        The segment
    min_points_per_wrap : int
        Minimum number of points required for a wrap
    bbox_pad_2d : int
        Padding to add to wrap bbox
    require_all_valid_in_bbox : bool
        If True, reject wraps that have invalid cells inside their padded bbox

    Returns
    -------
    Tuple[List[Dict], Dict[str, int]]
        Wrap dicts and rejection counts for span/inner-bbox filters.
    """
    reject_stats = {"span": 0, "inner_bbox": 0}
    if len(point_indices) < min_points_per_wrap:
        return [], reject_stats

    flat_indices = np.asarray(point_indices, dtype=np.int64)
    seg_h, seg_w = seg._valid_mask.shape
    rows = (flat_indices // seg_w).astype(np.int32)
    cols = (flat_indices % seg_w).astype(np.int32)
    z_arr = seg._z.ravel()[flat_indices].astype(np.float32)
    y_arr = seg._y.ravel()[flat_indices].astype(np.float32)
    x_arr = seg._x.ravel()[flat_indices].astype(np.float32)

    r_min_all, r_max_all = rows.min(), rows.max()
    c_min_all, c_max_all = cols.min(), cols.max()
    local_rows = rows - r_min_all
    local_cols = cols - c_min_all

    # Create local mask using numba helper
    local_h = r_max_all - r_min_all + 1
    local_w = c_max_all - c_min_all + 1
    mask = _build_local_mask(rows, cols, r_min_all, c_min_all, local_h, local_w)

    # Find connected components
    labeled, num_components = ndimage.label(mask)
    if num_components <= 0:
        return [], reject_stats

    labels_for_points = labeled[local_rows, local_cols]
    order = np.argsort(labels_for_points, kind="mergesort")
    labels_sorted = labels_for_points[order]
    boundaries = np.flatnonzero(labels_sorted[1:] != labels_sorted[:-1]) + 1
    starts = np.concatenate((np.array([0], dtype=np.int64), boundaries))
    stops = np.concatenate((boundaries, np.array([len(labels_sorted)], dtype=np.int64)))

    wraps = []
    valid_mask = seg._valid_mask
    next_wrap_id = 0

    for start, stop in zip(starts, stops):
        comp_id = int(labels_sorted[start])
        if comp_id <= 0:
            continue

        comp_point_idx = order[start:stop]

        if len(comp_point_idx) < min_points_per_wrap:
            continue

        # Get bbox in local coordinates
        comp_rows = rows[comp_point_idx]
        comp_cols = cols[comp_point_idx]
        r_min = int(comp_rows.min())
        r_max = int(comp_rows.max())
        c_min = int(comp_cols.min())
        c_max = int(comp_cols.max())

        comp_z = z_arr[comp_point_idx]
        comp_y = y_arr[comp_point_idx]
        comp_x = x_arr[comp_point_idx]

        # Apply padding
        r_min_p = r_min - bbox_pad_2d
        r_max_p = r_max + bbox_pad_2d
        c_min_p = c_min - bbox_pad_2d
        c_max_p = c_max + bbox_pad_2d

        # Check validity in padded bbox
        if require_all_valid_in_bbox:
            # Clamp to segment bounds for validity check
            r0 = max(r_min_p, 0)
            r1 = min(r_max_p, seg_h - 1)
            c0 = max(c_min_p, 0)
            c1 = min(c_max_p, seg_w - 1)
            if not valid_mask[r0:r1 + 1, c0:c1 + 1].all():
                continue

        wrap_id = next_wrap_id
        next_wrap_id += 1

        if not _passes_span_check_arrays(
            comp_z,
            comp_y,
            comp_x,
            chunk_bbox,
            target_size,
            min_span_ratio,
            edge_touch_frac,
            edge_touch_min_count,
            edge_touch_pad,
        ):
            reject_stats["span"] += 1
            continue

        if not _passes_inner_bbox_check_arrays(
            comp_y,
            comp_x,
            chunk_bbox,
            inner_bbox_fraction,
        ):
            reject_stats["inner_bbox"] += 1
            continue

        wraps.append({
            "wrap_id": wrap_id,
            "bbox_2d": (r_min_p, r_max_p, c_min_p, c_max_p),
        })

    return wraps, reject_stats


def get_required_axes(points_zyx: np.ndarray) -> Tuple[str, str]:
    """
    Get the axes that must pass the span check.

    For scroll papyrus surfaces, we require:
    - Z axis to span the full crop (always)
    - One of X or Y to span the full crop (whichever is larger)

    Parameters
    ----------
    points_zyx : np.ndarray
        Array of shape (N, 3) with [z, y, x] coordinates

    Returns
    -------
    Tuple[str, str]
        ("z", "y") or ("z", "x") depending on which horizontal axis has larger span
    """
    y_span = points_zyx[:, 1].max() - points_zyx[:, 1].min()
    x_span = points_zyx[:, 2].max() - points_zyx[:, 2].min()

    second_axis = "y" if y_span >= x_span else "x"
    return ("z", second_axis)


def passes_span_check_axis_aligned(
    points_zyx: np.ndarray,
    chunk_bbox: Tuple[float, ...],
    target_size: Tuple[int, int, int],
    min_span_ratio: float,
    edge_touch_frac: float,
    edge_touch_min_count: int,
    edge_touch_pad: int = 0,
) -> bool:
    """
    Check if the wrap has sufficient span and edge coverage along tangential axes.

    Uses axis-aligned span checks (no PCA).

    Parameters
    ----------
    points_zyx : np.ndarray
        Array of shape (N, 3) with [z, y, x] coordinates
    chunk_bbox : Tuple[float, ...]
        (z_min, z_max, y_min, y_max, x_min, x_max)
    target_size : Tuple[int, int, int]
        (chunk_d, chunk_h, chunk_w)
    min_span_ratio : float
        Minimum required span as fraction of chunk size (0 to 1)
    edge_touch_frac : float
        Fraction of chunk size for edge bands
    edge_touch_min_count : int
        Minimum points required in each edge band
    edge_touch_pad : int
        Optional padding to expand chunk bbox for edge test

    Returns
    -------
    bool
        True if wrap passes span and edge-touch checks
    """
    if len(points_zyx) == 0:
        return False
    return _passes_span_check_arrays(
        points_zyx[:, 0],
        points_zyx[:, 1],
        points_zyx[:, 2],
        chunk_bbox,
        target_size,
        min_span_ratio,
        edge_touch_frac,
        edge_touch_min_count,
        edge_touch_pad,
    )


def passes_inner_bbox_check(
    points_zyx: np.ndarray,
    chunk_bbox: Tuple[float, ...],
    inner_bbox_fraction: float,
) -> bool:
    """
    Check if the wrap center lies within the inner fraction of the chunk bbox.

    Parameters
    ----------
    points_zyx : np.ndarray
        Array of shape (N, 3) with [z, y, x] coordinates
    chunk_bbox : Tuple[float, ...]
        (z_min, z_max, y_min, y_max, x_min, x_max)
    inner_bbox_fraction : float
        Fraction of chunk size to keep (0 to 1). 1.0 means no filtering.

    Returns
    -------
    bool
        True if wrap center (Y/X only) is inside inner bbox
    """
    if len(points_zyx) == 0:
        return False
    if inner_bbox_fraction >= 1.0:
        return True
    if inner_bbox_fraction <= 0.0:
        return False

    return _passes_inner_bbox_check_arrays(
        points_zyx[:, 1],
        points_zyx[:, 2],
        chunk_bbox,
        inner_bbox_fraction,
    )


def find_world_chunk_patches(
    segments: List[Tifxyz],
    target_size: Tuple[int, int, int],
    overlap_fraction: float = 0.0,
    min_span_ratio: float = 1.0,
    edge_touch_frac: float = 0.1,
    edge_touch_min_count: int = 10,
    edge_touch_pad: int = 0,
    min_points_per_wrap: int = 100,
    bbox_pad_2d: int = 0,
    require_all_valid_in_bbox: bool = True,
    skip_chunk_if_any_invalid: bool = False,
    inner_bbox_fraction: float = 0.7,
    use_pca_for_span: bool = False,
    cache_dir: Optional[Path] = None,
    force_recompute: bool = False,
    verbose: bool = False,
    n_workers: Optional[int] = None,
    chunk_pad: float = 0.0,
) -> List[Dict]:
    """
    Find world-chunk patches across all segments.

    This method defines a 3D chunk grid covering the union bbox of all segments,
    then finds surface wraps within each chunk.

    Parameters
    ----------
    segments : List[Tifxyz]
        List of segments (should already be retargeted to volume scale)
    target_size : Tuple[int, int, int]
        (depth, height, width) of each chunk in world coordinates
    overlap_fraction : float
        Fraction of overlap between adjacent chunks (0.0 to <1.0)
    min_span_ratio : float
        Minimum required span as fraction of target_size for tangent axes (0 to 1)
    edge_touch_frac : float
        Fraction of target_size for edge bands in edge-touch test
    edge_touch_min_count : int
        Minimum points required in each edge band
    edge_touch_pad : int
        Padding to expand chunk bbox for edge-touch test
    min_points_per_wrap : int
        Minimum number of points for a wrap to be valid
    bbox_pad_2d : int
        Padding to add to wrap 2D bbox
    require_all_valid_in_bbox : bool
        If True, reject wraps with invalid cells in padded bbox
    skip_chunk_if_any_invalid : bool
        If True, reject entire chunk if ANY segment has invalid cells;
        If False (default), only skip that segment's wraps
    inner_bbox_fraction : float
        Fraction of chunk size to keep when filtering wraps by center (Y/X only).
        1.0 disables this filter.
    use_pca_for_span : bool
        If True, use PCA for span check (not implemented, use False)
    cache_dir : Optional[Path]
        Directory for caching results
    force_recompute : bool
        If True, ignore cache and recompute
    verbose : bool
        Print progress information and show progress bars
    n_workers : Optional[int]
        Number of parallel workers for chunk processing.
        None or 0 means sequential processing (default).
        Currently reserved for future implementation.
    chunk_pad : float
        Padding to expand chunk boundaries when assigning points.
        Points within `chunk_pad` of the boundary are included.
        Use ~20.0 to capture one extra row/col for discrete sampling.

    Returns
    -------
    List[Dict]
        List of chunk dicts, each containing:
        - "chunk_id": (cz, cy, cx) tuple
        - "bbox_3d": (z_min, z_max, y_min, y_max, x_min, x_max)
        - "wrap_count": int
        - "has_multiple_wraps": bool
        - "segment_ids": list of segment UUIDs
        - "wraps": list of wrap dicts with "wrap_id", "segment_id", "segment_idx", "bbox_2d"
    """
    if use_pca_for_span:
        raise NotImplementedError("PCA span check not yet implemented; use use_pca_for_span=False")

    if not segments:
        return []

    # Build cache key
    cache_key_data = {
        "method": "world_chunks",
        "target_size": list(target_size),
        "overlap_fraction": overlap_fraction,
        "min_span_ratio": min_span_ratio,
        "edge_touch_frac": edge_touch_frac,
        "edge_touch_min_count": edge_touch_min_count,
        "edge_touch_pad": edge_touch_pad,
        "min_points_per_wrap": min_points_per_wrap,
        "bbox_pad_2d": bbox_pad_2d,
        "require_all_valid_in_bbox": require_all_valid_in_bbox,
        "skip_chunk_if_any_invalid": skip_chunk_if_any_invalid,
        "inner_bbox_fraction": inner_bbox_fraction,
        "chunk_pad": chunk_pad,
        "segment_scales": [list(seg._scale) for seg in segments],
        "segment_uuids": [seg.uuid for seg in segments],
    }
    cache_key = hashlib.md5(
        json.dumps(cache_key_data, sort_keys=True).encode()
    ).hexdigest()

    # Try loading from cache and always report whether we hit cache or recomputed.
    cache_file = None
    cache_miss_reason = None
    if cache_dir is None:
        cache_miss_reason = "cache disabled (cache_dir=None)"
    else:
        cache_file = cache_dir / f"world_chunks_{cache_key}.json"
        if force_recompute:
            cache_miss_reason = "force_recompute=True"
        elif cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    cached = json.load(f)
                print(
                    f"[find_world_chunk_patches] cache hit key={cache_key} "
                    f"chunks={len(cached)} path={cache_file}"
                )
                return cached
            except (json.JSONDecodeError, KeyError):
                cache_miss_reason = f"cache unreadable path={cache_file}"
        else:
            cache_miss_reason = f"cache miss path={cache_file}"

    if cache_miss_reason is None:
        cache_miss_reason = "cache unavailable"
    print(
        f"[find_world_chunk_patches] recompute key={cache_key} reason={cache_miss_reason}"
    )

    # Compute dataset bbox
    if verbose:
        print("Computing dataset bbox...")
    dataset_bbox_zyx = compute_dataset_bbox(segments)
    if verbose:
        print(f"  Dataset bbox (ZYX): {dataset_bbox_zyx}")

    # Build chunk grid
    if verbose:
        print("Building chunk grid...")
    chunk_starts_z, chunk_starts_y, chunk_starts_x, strides, n_chunks = build_chunk_grid(
        dataset_bbox_zyx, target_size, overlap_fraction
    )
    if verbose:
        print(f"  Grid size: {n_chunks[0]} x {n_chunks[1]} x {n_chunks[2]} = {np.prod(n_chunks)} chunks")
        print(f"  Strides: {strides}")

    # Assign points to chunks (with tqdm progress bar when verbose)
    chunk_to_valid_points, chunk_to_invalid_segments = assign_points_to_chunks(
        segments, chunk_starts_z, chunk_starts_y, chunk_starts_x, target_size,
        verbose=verbose,
        chunk_pad=chunk_pad,
    )
    if verbose:
        print(f"  {len(chunk_to_valid_points)} chunks have valid points")

    # Process each chunk
    chunk_d, chunk_h, chunk_w = target_size
    results = []

    stats = {
        "chunks_examined": 0,
        "chunks_rejected_all_invalid": 0,
        "chunks_no_valid_wraps": 0,
        "chunks_accepted": 0,
        "wraps_rejected_size": 0,
        "wraps_rejected_validity": 0,
        "wraps_rejected_span": 0,
        "wraps_rejected_inner_bbox": 0,
        "wraps_accepted": 0,
    }

    # Create progress bar for chunk processing
    chunk_iter = tqdm(
        chunk_to_valid_points.keys(),
        desc="Processing chunks",
        disable=not verbose,
    )

    for chunk_id in chunk_iter:
        stats["chunks_examined"] += 1
        iz, iy, ix = chunk_id

        # Compute chunk bbox
        z_start = chunk_starts_z[iz]
        y_start = chunk_starts_y[iy]
        x_start = chunk_starts_x[ix]
        chunk_bbox = (
            z_start, z_start + chunk_d,
            y_start, y_start + chunk_h,
            x_start, x_start + chunk_w,
        )

        # Check for invalid cells in chunk
        if skip_chunk_if_any_invalid:
            # Reject entire chunk if ANY segment has invalid cells
            if chunk_id in chunk_to_invalid_segments:
                stats["chunks_rejected_all_invalid"] += 1
                continue

        # Collect wraps from all segments
        all_wraps = []
        segment_ids = set()

        for seg_idx, points in chunk_to_valid_points[chunk_id].items():
            seg = segments[seg_idx]

            # Per-segment invalid check (when not skip_chunk_if_any_invalid)
            if not skip_chunk_if_any_invalid:
                if seg_idx in chunk_to_invalid_segments.get(chunk_id, set()):
                    # Skip this segment's wraps for this chunk.
                    continue

            # Find wraps in this segment
            segment_wraps, reject_counts = find_wraps_in_chunk(
                points,
                seg,
                min_points_per_wrap,
                bbox_pad_2d,
                require_all_valid_in_bbox,
                chunk_bbox,
                target_size,
                min_span_ratio,
                edge_touch_frac,
                edge_touch_min_count,
                edge_touch_pad,
                inner_bbox_fraction,
            )
            stats["wraps_rejected_span"] += reject_counts["span"]
            stats["wraps_rejected_inner_bbox"] += reject_counts["inner_bbox"]

            for wrap in segment_wraps:
                # Wrap is valid
                stats["wraps_accepted"] += 1
                segment_ids.add(seg.uuid)

                all_wraps.append({
                    "wrap_id": wrap["wrap_id"],
                    "segment_id": seg.uuid,
                    "segment_idx": seg_idx,
                    "bbox_2d": wrap["bbox_2d"],
                })

        if not all_wraps:
            stats["chunks_no_valid_wraps"] += 1
            continue

        stats["chunks_accepted"] += 1

        results.append({
            "chunk_id": chunk_id,
            "bbox_3d": chunk_bbox,
            "wrap_count": len(all_wraps),
            "has_multiple_wraps": len(all_wraps) > 1,
            "segment_ids": list(segment_ids),
            "wraps": all_wraps,
        })

    if verbose:
        print("\n=== Statistics ===")
        print(f"  Chunks examined: {stats['chunks_examined']}")
        print(f"  Chunks rejected (all invalid): {stats['chunks_rejected_all_invalid']}")
        print(f"  Chunks rejected (no valid wraps): {stats['chunks_no_valid_wraps']}")
        print(f"  Chunks accepted: {stats['chunks_accepted']}")
        print(f"  Wraps rejected (span check): {stats['wraps_rejected_span']}")
        print(f"  Wraps rejected (inner bbox): {stats['wraps_rejected_inner_bbox']}")
        print(f"  Wraps accepted: {stats['wraps_accepted']}")

    # Save to cache
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"world_chunks_{cache_key}.json"
        try:
            # Convert tuples to lists for JSON serialization
            serializable = []
            for chunk in results:
                serializable.append({
                    "chunk_id": [int(x) for x in chunk["chunk_id"]],
                    "bbox_3d": [float(x) for x in chunk["bbox_3d"]],
                    "wrap_count": chunk["wrap_count"],
                    "has_multiple_wraps": chunk["has_multiple_wraps"],
                    "segment_ids": chunk["segment_ids"],
                    "wraps": [
                        {
                            "wrap_id": w["wrap_id"],
                            "segment_id": w["segment_id"],
                            "segment_idx": w["segment_idx"],
                            "bbox_2d": [int(x) for x in w["bbox_2d"]],
                        }
                        for w in chunk["wraps"]
                    ],
                })
            with open(cache_file, "w") as f:
                json.dump(serializable, f, indent=2)
            if verbose:
                print(f"Saved {len(results)} chunks to cache: {cache_file}")
        except Exception as e:
            if verbose:
                print(f"Warning: Could not save cache: {e}")

    return results
