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
def _find_containing_chunks_1d(
    coord: float,
    chunk_starts: np.ndarray,
    chunk_size: int,
    pad: float = 0.0,
) -> np.ndarray:
    """Find all chunk indices along one axis that contain the coordinate.

    With pad > 0, the effective range is [start - pad, start + size + pad).
    """
    n_chunks = len(chunk_starts)
    result = np.empty(n_chunks, dtype=np.int32)
    count = 0
    for i in range(n_chunks):
        if chunk_starts[i] - pad <= coord < chunk_starts[i] + chunk_size + pad:
            result[count] = i
            count += 1
    return result[:count]


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
    # Pre-allocate with upper bound estimate
    # Worst case: each point in ~8 chunks (2x2x2 overlap)
    max_pairs = len(z_flat) * 8
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

        # Find containing chunks along each axis (with padding)
        iz_valid = _find_containing_chunks_1d(z, chunk_starts_z, chunk_d, chunk_pad)
        iy_valid = _find_containing_chunks_1d(y, chunk_starts_y, chunk_h, chunk_pad)
        ix_valid = _find_containing_chunks_1d(x, chunk_starts_x, chunk_w, chunk_pad)

        # Generate all combinations (cartesian product)
        for i in range(len(iz_valid)):
            for j in range(len(iy_valid)):
                for k in range(len(ix_valid)):
                    if pair_count >= max_pairs:
                        # Reallocate (shouldn't happen often)
                        new_size = max_pairs * 2
                        new_point_indices = np.empty(new_size, dtype=np.int64)
                        new_iz = np.empty(new_size, dtype=np.int32)
                        new_iy = np.empty(new_size, dtype=np.int32)
                        new_ix = np.empty(new_size, dtype=np.int32)
                        new_point_indices[:pair_count] = point_indices[:pair_count]
                        new_iz[:pair_count] = iz_arr[:pair_count]
                        new_iy[:pair_count] = iy_arr[:pair_count]
                        new_ix[:pair_count] = ix_arr[:pair_count]
                        point_indices = new_point_indices
                        iz_arr = new_iz
                        iy_arr = new_iy
                        ix_arr = new_ix
                        max_pairs = new_size

                    point_indices[pair_count] = pt_idx
                    iz_arr[pair_count] = iz_valid[i]
                    iy_arr[pair_count] = iy_valid[j]
                    ix_arr[pair_count] = ix_valid[k]
                    pair_count += 1

    return (
        point_indices[:pair_count],
        iz_arr[:pair_count],
        iy_arr[:pair_count],
        ix_arr[:pair_count],
    )


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
        - chunk_to_valid_points: {chunk_id: {seg_idx: [(r, c, z, y, x), ...]}}
        - chunk_to_invalid_points: {chunk_id: {seg_idx: [(r, c, z, y, x), ...]}}
    """
    chunk_d, chunk_h, chunk_w = target_size
    chunk_to_valid_points: Dict[Tuple[int, int, int], Dict[int, List]] = {}
    chunk_to_invalid_points: Dict[Tuple[int, int, int], Dict[int, List]] = {}

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
        H, W = z_arr.shape

        # DEBUG: print segment info once
        if seg_idx == 0 and not hasattr(assign_points_to_chunks, '_debug_printed'):
            assign_points_to_chunks._debug_printed = True
            print(f"  [DEBUG] seg._scale={seg._scale}, array shape=({H}, {W})")
            valid_z = z_arr[valid_mask]
            valid_y = y_arr[valid_mask]
            valid_x = x_arr[valid_mask]
            print(f"  [DEBUG] z range: {valid_z.min():.2f} to {valid_z.max():.2f}, span={valid_z.max()-valid_z.min():.2f}")
            print(f"  [DEBUG] y range: {valid_y.min():.2f} to {valid_y.max():.2f}, span={valid_y.max()-valid_y.min():.2f}")
            print(f"  [DEBUG] x range: {valid_x.min():.2f} to {valid_x.max():.2f}, span={valid_x.max()-valid_x.min():.2f}")

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

        # Group results into dicts (Python loop but over much smaller result set)
        for i in range(len(point_indices)):
            pt_idx = point_indices[i]
            chunk_id = (int(iz_arr[i]), int(iy_arr[i]), int(ix_arr[i]))

            # Convert flat index back to (r, c)
            r = int(pt_idx // W)
            c = int(pt_idx % W)
            z_val = z_flat[pt_idx]
            y_val = y_flat[pt_idx]
            x_val = x_flat[pt_idx]
            is_valid = valid_flat[pt_idx]

            target_dict = chunk_to_valid_points if is_valid else chunk_to_invalid_points
            if chunk_id not in target_dict:
                target_dict[chunk_id] = {}
            if seg_idx not in target_dict[chunk_id]:
                target_dict[chunk_id][seg_idx] = []
            target_dict[chunk_id][seg_idx].append((r, c, z_val, y_val, x_val))

    return chunk_to_valid_points, chunk_to_invalid_points


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


@njit
def _collect_points_for_component(
    comp_indices_r: np.ndarray,
    comp_indices_c: np.ndarray,
    r_min_all: int,
    c_min_all: int,
    rows: np.ndarray,
    cols: np.ndarray,
    z_arr: np.ndarray,
    y_arr: np.ndarray,
    x_arr: np.ndarray,
) -> np.ndarray:
    """JIT-compiled collection of 3D coordinates for a component."""
    # Build lookup: create a mapping from (local_r, local_c) to point index
    # First pass: count matches
    n_comp = len(comp_indices_r)
    result = np.empty((n_comp, 3), dtype=np.float32)
    count = 0

    for i in range(n_comp):
        target_r = comp_indices_r[i] + r_min_all
        target_c = comp_indices_c[i] + c_min_all

        # Find matching point
        for j in range(len(rows)):
            if rows[j] == target_r and cols[j] == target_c:
                result[count, 0] = z_arr[j]
                result[count, 1] = y_arr[j]
                result[count, 2] = x_arr[j]
                count += 1
                break

    return result[:count]


def find_wraps_in_chunk(
    points: List[Tuple],
    seg: Tifxyz,
    min_points_per_wrap: int,
    bbox_pad_2d: int,
    require_all_valid_in_bbox: bool,
) -> List[Dict]:
    """
    Find connected components (wraps) from the given points for a single segment.

    Parameters
    ----------
    points : List[Tuple]
        List of (r, c, z, y, x) tuples for this segment in this chunk
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
    List[Dict]
        List of wrap dicts: {"wrap_id": int, "bbox_2d": tuple, "points_zyx": array}
    """
    if len(points) < min_points_per_wrap:
        return []

    # Extract arrays for numba processing
    rows = np.array([p[0] for p in points], dtype=np.int32)
    cols = np.array([p[1] for p in points], dtype=np.int32)
    z_arr = np.array([p[2] for p in points], dtype=np.float32)
    y_arr = np.array([p[3] for p in points], dtype=np.float32)
    x_arr = np.array([p[4] for p in points], dtype=np.float32)

    r_min_all, r_max_all = rows.min(), rows.max()
    c_min_all, c_max_all = cols.min(), cols.max()

    # Create local mask using numba helper
    local_h = r_max_all - r_min_all + 1
    local_w = c_max_all - c_min_all + 1
    mask = _build_local_mask(rows, cols, r_min_all, c_min_all, local_h, local_w)

    # Find connected components
    labeled, num_components = ndimage.label(mask)

    wraps = []
    valid_mask = seg._valid_mask
    seg_h, seg_w = valid_mask.shape

    for comp_id in range(1, num_components + 1):
        comp_mask = labeled == comp_id
        comp_indices = np.argwhere(comp_mask)

        if len(comp_indices) < min_points_per_wrap:
            continue

        # Get bbox in local coordinates
        local_r_min = comp_indices[:, 0].min()
        local_r_max = comp_indices[:, 0].max()
        local_c_min = comp_indices[:, 1].min()
        local_c_max = comp_indices[:, 1].max()

        # Convert to global segment coordinates
        r_min = local_r_min + r_min_all
        r_max = local_r_max + r_min_all
        c_min = local_c_min + c_min_all
        c_max = local_c_max + c_min_all

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

        # Collect 3D coordinates for this wrap using numba helper
        comp_indices_r = comp_indices[:, 0].astype(np.int32)
        comp_indices_c = comp_indices[:, 1].astype(np.int32)
        points_zyx = _collect_points_for_component(
            comp_indices_r, comp_indices_c,
            r_min_all, c_min_all,
            rows, cols, z_arr, y_arr, x_arr,
        )

        wraps.append({
            "wrap_id": len(wraps),
            "bbox_2d": (r_min_p, r_max_p, c_min_p, c_max_p),
            "points_zyx": points_zyx,
        })

    return wraps


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

    z_min, z_max, y_min, y_max, x_min, x_max = chunk_bbox
    chunk_d, chunk_h, chunk_w = target_size

    # Expand bbox by edge_touch_pad
    z_min_t = z_min - edge_touch_pad
    z_max_t = z_max + edge_touch_pad
    y_min_t = y_min - edge_touch_pad
    y_max_t = y_max + edge_touch_pad
    x_min_t = x_min - edge_touch_pad
    x_max_t = x_max + edge_touch_pad

    # Get required axes: Z (always) + larger of X/Y
    tangent_axes = get_required_axes(points_zyx)

    z = points_zyx[:, 0]
    y = points_zyx[:, 1]
    x = points_zyx[:, 2]

    # Compute spans
    z_span = z.max() - z.min()
    y_span = y.max() - y.min()
    x_span = x.max() - x.min()

    # Check span ratio for tangent axes
    axis_info = {
        "z": (z_span, chunk_d, z, z_min_t, z_max_t),
        "y": (y_span, chunk_h, y, y_min_t, y_max_t),
        "x": (x_span, chunk_w, x, x_min_t, x_max_t),
    }

    for ax in tangent_axes:
        span, chunk_size, coords, bbox_min, bbox_max = axis_info[ax]
        if span < min_span_ratio * chunk_size:
            # DEBUG
            if not hasattr(passes_span_check_axis_aligned, '_debug_count'):
                passes_span_check_axis_aligned._debug_count = 0
            if passes_span_check_axis_aligned._debug_count < 10:
                print(f"  [DEBUG] axis={ax}, span={span:.2f}, chunk_size={chunk_size}, threshold={min_span_ratio * chunk_size:.2f}")
                passes_span_check_axis_aligned._debug_count += 1
            return False

    # Edge-touch check for tangent axes
    for ax in tangent_axes:
        span, chunk_size, coords, bbox_min, bbox_max = axis_info[ax]
        band = edge_touch_frac * chunk_size

        # Count points near low edge
        low_count = (coords <= bbox_min + band).sum()
        # Count points near high edge
        high_count = (coords >= bbox_max - band).sum()

        if low_count < edge_touch_min_count or high_count < edge_touch_min_count:
            return False

    return True


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

    _, _, y_min, y_max, x_min, x_max = chunk_bbox
    margin_frac = (1.0 - inner_bbox_fraction) / 2.0
    y_margin = (y_max - y_min) * margin_frac
    x_margin = (x_max - x_min) * margin_frac

    inner_y_min = y_min + y_margin
    inner_y_max = y_max - y_margin
    inner_x_min = x_min + x_margin
    inner_x_max = x_max - x_margin

    center = points_zyx.mean(axis=0)

    return (
        center[1] >= inner_y_min and center[1] <= inner_y_max and
        center[2] >= inner_x_min and center[2] <= inner_x_max
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

    # Try loading from cache
    if cache_dir is not None and not force_recompute:
        cache_file = cache_dir / f"world_chunks_{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    cached = json.load(f)
                if verbose:
                    print(f"Loaded {len(cached)} chunks from cache: {cache_file}")
                return cached
            except (json.JSONDecodeError, KeyError):
                pass  # Cache corrupted, recompute

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
    chunk_to_valid_points, chunk_to_invalid_points = assign_points_to_chunks(
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
            has_invalid = False
            if chunk_id in chunk_to_invalid_points:
                for seg_idx in chunk_to_invalid_points[chunk_id]:
                    if len(chunk_to_invalid_points[chunk_id][seg_idx]) > 0:
                        has_invalid = True
                        break
            if has_invalid:
                stats["chunks_rejected_all_invalid"] += 1
                continue

        # Collect wraps from all segments
        all_wraps = []
        segment_ids = set()

        for seg_idx, points in chunk_to_valid_points[chunk_id].items():
            seg = segments[seg_idx]

            # Per-segment invalid check (when not skip_chunk_if_any_invalid)
            if not skip_chunk_if_any_invalid:
                if chunk_id in chunk_to_invalid_points:
                    if seg_idx in chunk_to_invalid_points[chunk_id]:
                        if len(chunk_to_invalid_points[chunk_id][seg_idx]) > 0:
                            # Skip this segment's wraps for this chunk
                            continue

            # Find wraps in this segment
            segment_wraps = find_wraps_in_chunk(
                points, seg,
                min_points_per_wrap, bbox_pad_2d, require_all_valid_in_bbox
            )

            for wrap in segment_wraps:
                # Apply span check
                if not passes_span_check_axis_aligned(
                    wrap["points_zyx"],
                    chunk_bbox,
                    target_size,
                    min_span_ratio,
                    edge_touch_frac,
                    edge_touch_min_count,
                    edge_touch_pad,
                ):
                    stats["wraps_rejected_span"] += 1
                    continue

                if not passes_inner_bbox_check(
                    wrap["points_zyx"],
                    chunk_bbox,
                    inner_bbox_fraction,
                ):
                    stats["wraps_rejected_inner_bbox"] += 1
                    continue

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
