"""
Hierarchical Coarse-to-Fine Tiling

Strategy:
1. Find global minimum tile size, double it for coarse tiles
2. Tile entire grid with coarse tiles
3. Recursively subdivide: measure local min, create children at local_min size
4. Stop when tiles are at their local minimum
5. Return flat list of valid patches

Key features:
- Adaptive subdivision based on local geometry (not fixed halving)
- Numba-accelerated measurement for performance
- Returns only valid patches that meet 3D coverage requirements
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import edt
import numpy as np
from numba import njit


# =============================================================================
# Numba helper functions
# =============================================================================

@njit
def _detect_axes(z, y, x, start_r, start_c, H, W):
    """
    Detect which 3D axis rows and columns primarily progress in.
    Returns (row_axis, col_axis) where 0=z, 1=y, 2=x
    """
    # Check column axis by looking at difference between adjacent columns
    if W == 1:
        dz_col = 0.0
        dy_col = 0.0
        dx_col = 0.0
    elif start_c + 1 < W:
        dz_col = abs(z[start_r, start_c + 1] - z[start_r, start_c])
        dy_col = abs(y[start_r, start_c + 1] - y[start_r, start_c])
        dx_col = abs(x[start_r, start_c + 1] - x[start_r, start_c])
    else:
        dz_col = abs(z[start_r, start_c] - z[start_r, start_c - 1])
        dy_col = abs(y[start_r, start_c] - y[start_r, start_c - 1])
        dx_col = abs(x[start_r, start_c] - x[start_r, start_c - 1])

    # Check row axis by looking at difference between adjacent rows
    if H == 1:
        dz_row = 0.0
        dy_row = 0.0
        dx_row = 0.0
    elif start_r + 1 < H:
        dz_row = abs(z[start_r + 1, start_c] - z[start_r, start_c])
        dy_row = abs(y[start_r + 1, start_c] - y[start_r, start_c])
        dx_row = abs(x[start_r + 1, start_c] - x[start_r, start_c])
    else:
        dz_row = abs(z[start_r, start_c] - z[start_r - 1, start_c])
        dy_row = abs(y[start_r, start_c] - y[start_r - 1, start_c])
        dx_row = abs(x[start_r, start_c] - x[start_r - 1, start_c])

    # Determine col axis (0=z, 1=y, 2=x) and second-best
    if dx_col >= dy_col and dx_col >= dz_col:
        col_axis = 2  # x
        col_max = dx_col
        col_second = 1 if dy_col >= dz_col else 0
    elif dy_col >= dz_col:
        col_axis = 1  # y
        col_max = dy_col
        col_second = 2 if dx_col >= dz_col else 0
    else:
        col_axis = 0  # z
        col_max = dz_col
        col_second = 2 if dx_col >= dy_col else 1

    if col_second == 0:
        col_second_max = dz_col
    elif col_second == 1:
        col_second_max = dy_col
    else:
        col_second_max = dx_col

    # Determine row axis and second-best
    if dz_row >= dy_row and dz_row >= dx_row:
        row_axis = 0  # z
        row_max = dz_row
        row_second = 2 if dx_row >= dy_row else 1
    elif dy_row >= dx_row:
        row_axis = 1  # y
        row_max = dy_row
        row_second = 2 if dx_row >= dz_row else 0
    else:
        row_axis = 2  # x
        row_max = dx_row
        row_second = 1 if dy_row >= dz_row else 0

    if row_second == 0:
        row_second_max = dz_row
    elif row_second == 1:
        row_second_max = dy_row
    else:
        row_second_max = dx_row

    if row_axis == col_axis:
        axis_ratio = 0.35
        if row_max <= col_max:
            if row_second_max >= row_max * axis_ratio:
                row_axis = row_second
        else:
            if col_second_max >= col_max * axis_ratio:
                col_axis = col_second

    return row_axis, col_axis


@njit
def _get_edge_minmax(z, y, x, r_min, r_max, c_min, c_max, axis, edge):
    """
    Get min and max of specified axis along an edge.
    edge: 0=first_row, 1=last_row, 2=first_col, 3=last_col
    axis: 0=z, 1=y, 2=x
    Returns (min_val, max_val)
    """
    if axis == 0:
        arr = z
    elif axis == 1:
        arr = y
    else:
        arr = x

    if edge == 0:  # first row
        min_val = arr[r_min, c_min]
        max_val = arr[r_min, c_min]
        for c in range(c_min, c_max + 1):
            v = arr[r_min, c]
            min_val = min(min_val, v)
            max_val = max(max_val, v)
    elif edge == 1:  # last row
        min_val = arr[r_max, c_min]
        max_val = arr[r_max, c_min]
        for c in range(c_min, c_max + 1):
            v = arr[r_max, c]
            min_val = min(min_val, v)
            max_val = max(max_val, v)
    elif edge == 2:  # first col
        min_val = arr[r_min, c_min]
        max_val = arr[r_min, c_min]
        for r in range(r_min, r_max + 1):
            v = arr[r, c_min]
            min_val = min(min_val, v)
            max_val = max(max_val, v)
    else:  # last col
        min_val = arr[r_min, c_max]
        max_val = arr[r_min, c_max]
        for r in range(r_min, r_max + 1):
            v = arr[r, c_max]
            min_val = min(min_val, v)
            max_val = max(max_val, v)

    return min_val, max_val


@njit
def _check_coverage(z, y, x, r_min, r_max, c_min, c_max,
                    row_axis, col_axis, target_d, target_h, target_w):
    """
    Check if the current rectangle provides full coverage for target volume.
    Returns True if the gap between opposite edges meets the target size.
    """
    # Get target size for each axis
    if row_axis == 0:
        row_target = target_d
    elif row_axis == 1:
        row_target = target_h
    else:
        row_target = target_w

    if col_axis == 0:
        col_target = target_d
    elif col_axis == 1:
        col_target = target_h
    else:
        col_target = target_w

    # Check row coverage: edge separation must meet target size, regardless of direction
    min_first_row, max_first_row = _get_edge_minmax(z, y, x, r_min, r_max, c_min, c_max, row_axis, 0)
    min_last_row, max_last_row = _get_edge_minmax(z, y, x, r_min, r_max, c_min, c_max, row_axis, 1)
    row_forward = min_last_row - max_first_row
    row_backward = min_first_row - max_last_row
    row_coverage_ok = (row_forward >= row_target) or (row_backward >= row_target)

    # Check col coverage: edge separation must meet target size, regardless of direction
    min_first_col, max_first_col = _get_edge_minmax(z, y, x, r_min, r_max, c_min, c_max, col_axis, 2)
    min_last_col, max_last_col = _get_edge_minmax(z, y, x, r_min, r_max, c_min, c_max, col_axis, 3)
    col_forward = min_last_col - max_first_col
    col_backward = min_first_col - max_last_col
    col_coverage_ok = (col_forward >= col_target) or (col_backward >= col_target)

    return row_coverage_ok and col_coverage_ok


@njit
def _compute_3d_bbox(z, y, x, r_min, r_max, c_min, c_max):
    """Compute 3D bounding box for a rectangular region."""
    z_min = z_max = z[r_min, c_min]
    y_min = y_max = y[r_min, c_min]
    x_min = x_max = x[r_min, c_min]

    for r in range(r_min, r_max + 1):
        for c in range(c_min, c_max + 1):
            zv, yv, xv = z[r, c], y[r, c], x[r, c]
            z_min = min(z_min, zv)
            z_max = max(z_max, zv)
            y_min = min(y_min, yv)
            y_max = max(y_max, yv)
            x_min = min(x_min, xv)
            x_max = max(x_max, xv)

    return z_min, z_max, y_min, y_max, x_min, x_max


@njit
def _compute_centered_3d_bbox(z, y, x, r_min, r_max, c_min, c_max, target_d, target_h, target_w):
    """Compute 3D bounding box centered on surface centroid with target size.

    Instead of returning the actual coordinate extent of the surface,
    this returns a fixed-size bbox centered on the surface centroid.
    """
    z_sum = 0.0
    y_sum = 0.0
    x_sum = 0.0
    count = 0

    for r in range(r_min, r_max + 1):
        for c in range(c_min, c_max + 1):
            z_sum += z[r, c]
            y_sum += y[r, c]
            x_sum += x[r, c]
            count += 1

    if count == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    z_center = z_sum / count
    y_center = y_sum / count
    x_center = x_sum / count

    return (
        z_center - target_d / 2.0, z_center + target_d / 2.0,
        y_center - target_h / 2.0, y_center + target_h / 2.0,
        x_center - target_w / 2.0, x_center + target_w / 2.0,
    )


@njit
def _measure_min_tile_size_at_point(z, y, x, valid, start_r, start_c, target_d, target_h, target_w):
    """
    Measure the minimum 2D tile size needed at a point to cover the target 3D volume.

    Strategy:
    1. Symmetric expansion to find a working size (original approach)
    2. Then shrink each dimension independently while checking FULL coverage

    Returns (tile_height, tile_width) or (0, 0) if coverage can't be achieved.
    """
    H, W = z.shape

    if not valid[start_r, start_c]:
        return 0, 0

    # Detect axes
    row_axis, col_axis = _detect_axes(z, y, x, start_r, start_c, H, W)

    # Phase 1: Symmetric expansion to find a WORKING size
    working_half = 0
    max_expand_r = max(start_r, H - 1 - start_r)
    max_expand_c = max(start_c, W - 1 - start_c)
    max_expand = max(max_expand_r, max_expand_c)

    for half in range(1, max_expand + 1):
        r_min = max(0, start_r - half)
        r_max = min(H - 1, start_r + half)
        c_min = max(0, start_c - half)
        c_max = min(W - 1, start_c + half)

        # Check validity of full rectangle
        all_valid = True
        for r in range(r_min, r_max + 1):
            for c in range(c_min, c_max + 1):
                if not valid[r, c]:
                    all_valid = False
                    break
            if not all_valid:
                break

        if not all_valid:
            break  # Hit invalid region

        # Check full coverage
        if _check_coverage(z, y, x, r_min, r_max, c_min, c_max,
                          row_axis, col_axis, target_d, target_h, target_w):
            working_half = half
            break

    if working_half == 0:
        return 0, 0

    # We have a working rectangle - now try to shrink each dimension
    r_min = max(0, start_r - working_half)
    r_max = min(H - 1, start_r + working_half)
    c_min = max(0, start_c - working_half)
    c_max = min(W - 1, start_c + working_half)

    # Phase 2: Try to shrink HEIGHT while keeping full coverage
    best_half_h = working_half
    for half_h in range(working_half - 1, 0, -1):
        test_r_min = max(0, start_r - half_h)
        test_r_max = min(H - 1, start_r + half_h)

        # Check if smaller height still provides coverage (with current width)
        if _check_coverage(z, y, x, test_r_min, test_r_max, c_min, c_max,
                          row_axis, col_axis, target_d, target_h, target_w):
            best_half_h = half_h
            r_min, r_max = test_r_min, test_r_max
        else:
            break  # Can't shrink further

    # Phase 3: Try to shrink WIDTH while keeping full coverage
    best_half_w = working_half
    for half_w in range(working_half - 1, 0, -1):
        test_c_min = max(0, start_c - half_w)
        test_c_max = min(W - 1, start_c + half_w)

        # Check if smaller width still provides coverage (with current height)
        if _check_coverage(z, y, x, r_min, r_max, test_c_min, test_c_max,
                          row_axis, col_axis, target_d, target_h, target_w):
            best_half_w = half_w
            c_min, c_max = test_c_min, test_c_max
        else:
            break  # Can't shrink further

    return r_max - r_min + 1, c_max - c_min + 1


# =============================================================================
# Tile size estimation
# =============================================================================

def estimate_tile_size(segment, target_size, num_samples=100, margin=1.1):
    """
    Sample points across the segment to find the tile size needed for coverage.

    Parameters
    ----------
    segment : Tifxyz
        The segment to analyze
    target_size : tuple of (depth, height, width)
        Target 3D volume size
    num_samples : int
        Number of points to sample
    margin : float
        Safety margin multiplier (default 1.1 = 10% extra)

    Returns
    -------
    tile_size : tuple of (tile_height, tile_width)
    """
    target_d, target_h, target_w = target_size
    z, y, x = segment._z, segment._y, segment._x
    valid = segment._valid_mask
    H, W = segment.shape

    # Use EDT to find points far from invalid regions (good candidates for measurement)
    distance = edt.edt(valid.astype(np.uint8))
    min_dist = min(target_d, target_h, target_w) // 2

    # Find valid interior points
    valid_indices = np.argwhere((valid) & (distance >= min_dist))
    if len(valid_indices) == 0:
        # Fall back to any valid points
        valid_indices = np.argwhere(valid)

    if len(valid_indices) == 0:
        return None

    # Use deterministic grid sampling instead of random
    # This ensures we don't miss worst-case regions
    grid_step_r = max(1, H // int(np.sqrt(num_samples)))
    grid_step_c = max(1, W // int(np.sqrt(num_samples)))

    sample_indices = []
    for r in range(grid_step_r // 2, H, grid_step_r):
        for c in range(grid_step_c // 2, W, grid_step_c):
            if valid[r, c] and distance[r, c] >= min_dist:
                sample_indices.append((r, c))

    if not sample_indices and len(valid_indices) > 0:
        step = max(1, len(valid_indices) // max(1, num_samples))
        for i in range(0, len(valid_indices), step):
            r, c = valid_indices[i]
            sample_indices.append((r, c))

    sample_indices = np.array(sample_indices) if sample_indices else np.array([]).reshape(0, 2)
    sample_count = len(sample_indices)

    max_tile_h = 0
    max_tile_w = 0
    successful_samples = 0

    for r, c in sample_indices:
        tile_h, tile_w = _measure_min_tile_size_at_point(
            z, y, x, valid, r, c,
            float(target_d), float(target_h), float(target_w)
        )
        if tile_h > 0 and tile_w > 0:
            max_tile_h = max(max_tile_h, tile_h)
            max_tile_w = max(max_tile_w, tile_w)
            successful_samples += 1

    if successful_samples == 0:
        return None

    # Add safety margin
    tile_h = int(max_tile_h * margin)
    tile_w = int(max_tile_w * margin)

    print(f"Sampled {successful_samples}/{sample_count} points, max tile size: {max_tile_h}x{max_tile_w}, with margin: {tile_h}x{tile_w}")

    return tile_h, tile_w


# =============================================================================
# Hierarchical tiling data structures and functions
# =============================================================================

@dataclass
class Tile:
    """A tile for subdivision processing."""
    bbox_2d: Tuple[int, int, int, int]  # (r_min, r_max, c_min, c_max)
    bbox_3d: Optional[Tuple[float, ...]] = None  # (z_min, z_max, y_min, y_max, x_min, x_max)
    tile_size: Tuple[int, int] = (0, 0)  # (rows, cols)
    is_valid: bool = True  # False if tile can't achieve 3D coverage
    local_min_size: Optional[Tuple[int, int]] = None  # Measured local minimum


def create_coarse_grid(segment, coarse_size: Tuple[int, int],
                       overlap_fraction: float = 0.0,
                       grid_offset: Tuple[int, int] = (0, 0)) -> List[Tile]:
    """
    Create initial coarse grid tiling.

    Parameters
    ----------
    segment : Tifxyz
        The segment to tile
    coarse_size : tuple of (tile_height, tile_width)
        Size of coarse tiles
    overlap_fraction : float
        Fraction of overlap between adjacent tiles (0.0 = no overlap, 0.25 = 25% overlap)
    grid_offset : tuple of (row_offset, col_offset)
        Offset for the grid starting position (useful for multi-pass tiling)

    Returns
    -------
    tiles : list of Tile
    """
    tile_h, tile_w = coarse_size
    H, W = segment.shape
    valid = segment._valid_mask

    # Compute stride based on overlap fraction
    stride_h = max(1, int(tile_h * (1 - overlap_fraction)))
    stride_w = max(1, int(tile_w * (1 - overlap_fraction)))

    # Apply grid offset
    r_offset, c_offset = grid_offset

    tiles = []

    # Grid iteration with configurable stride and offset
    for r_start in range(r_offset, H - tile_h + 1, stride_h):
        for c_start in range(c_offset, W - tile_w + 1, stride_w):
            r_end = r_start + tile_h - 1
            c_end = c_start + tile_w - 1

            # Check if tile region has any valid cells
            tile_valid = valid[r_start:r_end+1, c_start:c_end+1]
            if not tile_valid.any():
                continue  # Skip completely invalid tiles

            # Create tile (validation happens later during subdivision)
            tile = Tile(
                bbox_2d=(r_start, r_end, c_start, c_end),
                tile_size=(tile_h, tile_w),
            )
            tiles.append(tile)

    return tiles


def create_child_tiles(
    parent: Tile,
    child_size: Tuple[int, int],
    segment,
    target_size: Tuple[int, int, int],
    overlap_fraction: float = 0.0,
) -> List[Tile]:
    """
    Create child tiles within a parent tile region.

    Parameters
    ----------
    parent : Tile
        The parent tile to subdivide
    child_size : tuple of (tile_height, tile_width)
        Size for child tiles
    segment : Tifxyz
        The segment
    target_size : tuple of (depth, height, width)
        Target 3D volume size
    overlap_fraction : float
        Fraction of overlap between adjacent tiles (0.0 = no overlap, 0.25 = 25% overlap)

    Returns
    -------
    children : list of Tile
    """
    r_min, r_max, c_min, c_max = parent.bbox_2d
    child_h, child_w = child_size
    target_d, target_h, target_w = target_size

    z, y, x = segment._z, segment._y, segment._x
    valid = segment._valid_mask

    # Compute stride based on overlap fraction
    stride_h = max(1, int(child_h * (1 - overlap_fraction)))
    stride_w = max(1, int(child_w * (1 - overlap_fraction)))

    children = []

    # Grid within parent bounds with configurable stride
    for child_r_start in range(r_min, r_max - child_h + 2, stride_h):
        for child_c_start in range(c_min, c_max - child_w + 2, stride_w):
            child_r_end = min(child_r_start + child_h - 1, r_max)
            child_c_end = min(child_c_start + child_w - 1, c_max)

            # Skip partial tiles at edges
            actual_h = child_r_end - child_r_start + 1
            actual_w = child_c_end - child_c_start + 1
            if actual_h < child_h or actual_w < child_w:
                continue

            # Check if child is fully valid
            child_valid = valid[child_r_start:child_r_end+1, child_c_start:child_c_end+1]
            if not child_valid.all():
                continue

            # Compute 3D bbox centered on surface centroid with target size
            z_min, z_max, y_min, y_max, x_min, x_max = _compute_centered_3d_bbox(
                z, y, x, child_r_start, child_r_end, child_c_start, child_c_end,
                target_d, target_h, target_w
            )

            # Use edge-based coverage check (stricter than extent check)
            # Detect which axes rows/cols correspond to
            H, W = z.shape
            row_axis, col_axis = _detect_axes(z, y, x, child_r_start, child_c_start, H, W)
            has_coverage = _check_coverage(
                z, y, x, child_r_start, child_r_end, child_c_start, child_c_end,
                row_axis, col_axis, target_d, target_h, target_w
            )

            child = Tile(
                bbox_2d=(child_r_start, child_r_end, child_c_start, child_c_end),
                bbox_3d=(z_min, z_max, y_min, y_max, x_min, x_max),
                tile_size=child_size,
                is_valid=has_coverage,
            )
            children.append(child)

    return children


def _test_child_size_validity(
    parent: Tile,
    child_h: int,
    child_w: int,
    segment,
    target_size: Tuple[int, int, int],
    overlap_fraction: float,
    min_valid_fraction: float,
) -> bool:
    """Test if a given child size produces enough valid children."""
    children = create_child_tiles(
        parent, (child_h, child_w), segment, target_size, overlap_fraction
    )
    if not children:
        return False
    valid_count = sum(1 for c in children if c.is_valid)
    return valid_count / len(children) >= min_valid_fraction


def _binary_search_min_size(lo: int, hi: int, test_fn) -> int:
    """
    Binary search to find minimum size where test_fn returns True.
    Returns hi if no valid size found.
    """
    if lo >= hi:
        return hi

    # First check if any size works
    if not test_fn(hi):
        return 0  # Even max size doesn't work

    # Binary search for minimum
    while hi - lo > 2:
        mid = (lo + hi) // 2
        if test_fn(mid):
            hi = mid
        else:
            lo = mid + 1

    # Fine-tune in final range
    for size in range(lo, hi + 1):
        if test_fn(size):
            return size
    return hi


def find_valid_child_size(
    parent: Tile,
    segment,
    target_size: Tuple[int, int, int],
    overlap_fraction: float,
    global_min_size: Tuple[int, int],
    min_valid_fraction: float = 0.5,
) -> Tuple[int, int]:
    """
    Find smallest child size where grid-placed children are valid.

    Uses binary search to find minimum valid height and width INDEPENDENTLY,
    allowing rectangular tiles for better efficiency.

    Parameters
    ----------
    parent : Tile
        The parent tile to subdivide
    segment : Tifxyz
        The segment
    target_size : tuple of (depth, height, width)
        Target 3D volume size
    overlap_fraction : float
        Fraction of overlap between adjacent tiles
    global_min_size : tuple of (min_h, min_w)
        Global minimum tile size (hint for search start)
    min_valid_fraction : float
        Minimum fraction of children that must be valid (default 0.5)

    Returns
    -------
    child_size : tuple of (child_h, child_w)
        The smallest valid child size, or (0, 0) if none found
    """
    parent_h, parent_w = parent.tile_size

    # Search bounds
    min_h_start = max(30, global_min_size[0] // 2)
    min_w_start = max(30, global_min_size[1] // 2)
    max_h = parent_h
    max_w = parent_w

    if min_h_start >= max_h or min_w_start >= max_w:
        return (0, 0)

    # Phase 1: Binary search for minimum valid height (using max width)
    def test_height(h):
        return _test_child_size_validity(
            parent, h, max_w, segment, target_size, overlap_fraction, min_valid_fraction
        )

    min_h = _binary_search_min_size(min_h_start, max_h, test_height)
    if min_h == 0:
        return (0, 0)

    # Phase 2: Binary search for minimum valid width (using found height)
    def test_width(w):
        return _test_child_size_validity(
            parent, min_h, w, segment, target_size, overlap_fraction, min_valid_fraction
        )

    min_w = _binary_search_min_size(min_w_start, max_w, test_width)
    if min_w == 0:
        return (0, 0)

    # Phase 3: Try to shrink height further with the found width
    # (the optimal width might allow smaller height than max_w did)
    def test_height_refined(h):
        return _test_child_size_validity(
            parent, h, min_w, segment, target_size, overlap_fraction, min_valid_fraction
        )

    min_h_refined = _binary_search_min_size(min_h_start, min_h, test_height_refined)
    if min_h_refined > 0:
        min_h = min_h_refined

    return (min_h, min_w)


def validate_tile_as_leaf(tile: Tile, segment, target_size: Tuple[int, int, int]):
    """
    Validate a tile as a leaf using edge-based coverage check.

    Sets tile.bbox_3d and tile.is_valid based on strict coverage criteria.
    """
    target_d, target_h, target_w = target_size
    z, y, x = segment._z, segment._y, segment._x
    valid = segment._valid_mask
    H, W = segment.shape

    r_min, r_max, c_min, c_max = tile.bbox_2d

    # Check if all pixels are valid
    if not valid[r_min:r_max+1, c_min:c_max+1].all():
        tile.is_valid = False
        return

    # Compute 3D bbox centered on surface centroid with target size
    z_min, z_max, y_min, y_max, x_min, x_max = _compute_centered_3d_bbox(
        z, y, x, r_min, r_max, c_min, c_max, target_d, target_h, target_w
    )
    tile.bbox_3d = (z_min, z_max, y_min, y_max, x_min, x_max)

    # Use edge-based coverage check (stricter than extent check)
    row_axis, col_axis = _detect_axes(z, y, x, r_min, c_min, H, W)
    tile.is_valid = _check_coverage(
        z, y, x, r_min, r_max, c_min, c_max,
        row_axis, col_axis, target_d, target_h, target_w
    )


def subdivide_tiles(
    segment,
    initial_tiles: List[Tile],
    target_size: Tuple[int, int, int],
    overlap_fraction: float = 0.0,
    global_min_size: Tuple[int, int] = None,
    verbose: bool = True,
) -> List[Tuple[Tuple[int, int, int, int], Tuple[float, ...]]]:
    """
    Recursively subdivide tiles until each is at its local minimum size.

    Uses grid-aware sizing: tries actual child sizes and validates that
    children placed on the grid are valid, instead of sampling arbitrary points.

    Parameters
    ----------
    segment : Tifxyz
        The segment
    initial_tiles : list of Tile
        Initial coarse tiles
    target_size : tuple of (depth, height, width)
        Target 3D volume size
    overlap_fraction : float
        Fraction of overlap between adjacent tiles (0.0 = no overlap, 0.25 = 25% overlap)
    global_min_size : tuple of (min_h, min_w)
        Global minimum tile size (starting point for child size search)
    verbose : bool
        Print progress information

    Returns
    -------
    valid_patches : list of (bbox_2d, bbox_3d)
        Flat list of valid patches that meet 3D coverage requirements
    """
    # Default global_min_size if not provided
    if global_min_size is None:
        global_min_size = (50, 50)  # Reasonable default

    valid_patches = []
    queue = list(initial_tiles)
    iteration = 0

    while queue:
        iteration += 1
        if verbose:
            print(f"  Iteration {iteration}: processing {len(queue)} tiles")

        next_queue = []
        subdivided_count = 0
        leaf_count = 0
        invalid_count = 0

        for tile in queue:
            tile_h, tile_w = tile.tile_size

            # Find valid child size using grid-aware search
            child_h, child_w = find_valid_child_size(
                tile, segment, target_size, overlap_fraction, global_min_size
            )
            tile.local_min_size = (child_h, child_w)

            # No valid child size found - validate tile as leaf
            if child_h == 0 or child_w == 0:
                validate_tile_as_leaf(tile, segment, target_size)
                if tile.is_valid:
                    valid_patches.append((tile.bbox_2d, tile.bbox_3d))
                    leaf_count += 1
                else:
                    invalid_count += 1
                continue

            # Can't subdivide meaningfully (child would be >= 90% of parent)
            if child_h >= tile_h * 0.9 and child_w >= tile_w * 0.9:
                validate_tile_as_leaf(tile, segment, target_size)
                if tile.is_valid:
                    valid_patches.append((tile.bbox_2d, tile.bbox_3d))
                    leaf_count += 1
                else:
                    invalid_count += 1
                continue

            # Can subdivide - create children at the validated child size
            children = create_child_tiles(
                tile, (child_h, child_w), segment, target_size, overlap_fraction
            )

            if children:
                valid_children = [c for c in children if c.is_valid]
                if valid_children:
                    # Has valid children - add to queue for further subdivision
                    next_queue.extend(valid_children)
                    subdivided_count += 1
                else:
                    # All children invalid - keep parent as leaf (fallback)
                    validate_tile_as_leaf(tile, segment, target_size)
                    if tile.is_valid:
                        valid_patches.append((tile.bbox_2d, tile.bbox_3d))
                        leaf_count += 1
                    else:
                        invalid_count += 1
            else:
                # No children could be created - validate tile as leaf
                validate_tile_as_leaf(tile, segment, target_size)
                if tile.is_valid:
                    valid_patches.append((tile.bbox_2d, tile.bbox_3d))
                    leaf_count += 1
                else:
                    invalid_count += 1

        if verbose:
            print(f"    -> {subdivided_count} subdivided, {leaf_count} leaves, {invalid_count} invalid, {len(next_queue)} queued")

        queue = next_queue

    return valid_patches


def hierarchical_tile_patches(
    segment,
    target_size: Tuple[int, int, int],
    num_calibration_samples: int = 200,
    coarse_multiplier: float = 2.0,
    overlap_fraction: float = 0.0,
    grid_offset: Tuple[int, int] = (0, 0),
    verbose: bool = True,
) -> List[Tuple[Tuple[int, int, int, int], Tuple[float, ...]]]:
    """
    Hierarchical coarse-to-fine tiling.

    Strategy:
    1. Find global minimum tile size via sampling
    2. Multiply by coarse_multiplier to get coarse tile size
    3. Tile entire grid with coarse tiles (with optional overlap)
    4. Recursively subdivide based on local minimum measurements
    5. Return flat list of valid patches

    Parameters
    ----------
    segment : Tifxyz
        The segment to tile
    target_size : tuple of (depth, height, width)
        Target 3D volume size in voxels
    num_calibration_samples : int
        Number of samples for global calibration
    coarse_multiplier : float
        Multiplier for coarse tile size (default 2.0 = double the minimum)
    overlap_fraction : float
        Fraction of overlap between adjacent tiles (0.0 = no overlap, 0.25 = 25% overlap)
    grid_offset : tuple of (row_offset, col_offset)
        Offset for the coarse grid starting position (useful for multi-pass tiling)
    verbose : bool
        Print progress information

    Returns
    -------
    patches : list of (bbox_2d, bbox_3d)
        Flat list of valid patches that meet 3D coverage requirements
    """
    import time
    t_start = time.time()

    # Phase 1: Global calibration
    if verbose:
        print("Phase 1: Global calibration...")

    global_min = estimate_tile_size(segment, target_size, num_samples=num_calibration_samples, margin=1.0)
    if global_min is None:
        print("  Could not estimate global minimum tile size")
        return []

    global_min_h, global_min_w = global_min
    coarse_h = int(global_min_h * coarse_multiplier)
    coarse_w = int(global_min_w * coarse_multiplier)
    coarse_size = (coarse_h, coarse_w)

    if verbose:
        print(f"  Global minimum: {global_min_h}x{global_min_w}")
        print(f"  Coarse tile size: {coarse_h}x{coarse_w}")
        print(f"  Overlap fraction: {overlap_fraction:.1%}")

    # Phase 2: Create coarse grid
    if verbose:
        print("\nPhase 2: Creating coarse grid...")

    coarse_tiles = create_coarse_grid(segment, coarse_size, overlap_fraction, grid_offset)

    if verbose:
        print(f"  Created {len(coarse_tiles)} coarse tiles")

    # Phase 3: Recursive subdivision
    if verbose:
        print("\nPhase 3: Recursive subdivision...")

    patches = subdivide_tiles(segment, coarse_tiles, target_size, overlap_fraction, global_min, verbose=verbose)

    t_elapsed = time.time() - t_start

    if verbose:
        valid_area = segment._valid_mask.sum()
        coverage = sum((p[0][1]-p[0][0]+1)*(p[0][3]-p[0][2]+1) for p in patches)
        coverage_pct = coverage / valid_area * 100 if valid_area > 0 else 0
        print(f"\nResult: {len(patches)} patches, {coverage:,} px ({coverage_pct:.1f}%), {t_elapsed:.2f}s")

    return patches


def multipass_hierarchical_tiling(
    segment,
    target_size: Tuple[int, int, int],
    num_calibration_samples: int = 200,
    coarse_multiplier: float = 2.0,
    overlap_fraction: float = 0.0,
    min_new_coverage: float = 0.5,
    verbose: bool = True,
) -> List[Tuple[Tuple[int, int, int, int], Tuple[float, ...]]]:
    """
    Multi-pass hierarchical tiling with grid offsets for better coverage.

    Runs hierarchical tiling multiple times with different grid offsets.
    Second-pass patches are filtered to ensure they contribute meaningful
    new coverage (not just overlapping existing patches).

    Parameters
    ----------
    segment : Tifxyz
        The segment to tile
    target_size : tuple of (depth, height, width)
        Target 3D volume size in voxels
    num_calibration_samples : int
        Number of samples for global calibration
    coarse_multiplier : float
        Multiplier for coarse tile size
    overlap_fraction : float
        Fraction of overlap between adjacent tiles
    min_new_coverage : float
        Minimum fraction of new (uncovered) pixels required for a patch
        to be accepted in subsequent passes (default 0.5 = 50%)
    verbose : bool
        Print progress information

    Returns
    -------
    all_patches : list of (bbox_2d, bbox_3d)
        Combined patches from all passes
    """
    import time
    t_start = time.time()

    H, W = segment.shape
    valid_area = segment._valid_mask.sum()

    # Estimate global minimum tile size for offset calculation
    global_min = estimate_tile_size(segment, target_size, num_samples=num_calibration_samples, margin=1.0)
    if global_min is None:
        print("  Could not estimate global minimum tile size")
        return []

    global_min_h, global_min_w = global_min
    coarse_h = int(global_min_h * coarse_multiplier)
    coarse_w = int(global_min_w * coarse_multiplier)

    # Compute stride for offset calculation
    stride_h = max(1, int(coarse_h * (1 - overlap_fraction)))
    stride_w = max(1, int(coarse_w * (1 - overlap_fraction)))

    # Build coverage mask (tracks which pixels are covered)
    coverage_mask = np.zeros((H, W), dtype=np.bool_)

    all_patches = []

    # Pass 1: No offset
    if verbose:
        print("=" * 60)
        print("Pass 1: Grid offset (0, 0)")
        print("=" * 60)

    patches_1 = hierarchical_tile_patches(
        segment, target_size,
        num_calibration_samples=num_calibration_samples,
        coarse_multiplier=coarse_multiplier,
        overlap_fraction=overlap_fraction,
        grid_offset=(0, 0),
        verbose=verbose,
    )

    # Add pass 1 patches and update coverage mask
    for bbox_2d, bbox_3d in patches_1:
        r_min, r_max, c_min, c_max = bbox_2d
        coverage_mask[r_min:r_max+1, c_min:c_max+1] = True
        all_patches.append((bbox_2d, bbox_3d))

    if verbose:
        covered_1 = coverage_mask.sum()
        print(f"\nPass 1 coverage: {covered_1:,} px ({covered_1/valid_area*100:.1f}%)")

    # Define 4 offset patterns to cover gaps:
    # Pass 2: vertical offset only (fills horizontal gaps)
    # Pass 3: horizontal offset only
    # Pass 4: diagonal offset
    additional_offsets = [
        (stride_h // 2, 0),           # vertical only - fills horizontal bands
        (0, stride_w // 2),           # horizontal only
        (stride_h // 2, stride_w // 2),  # diagonal
    ]

    total_accepted = 0
    total_rejected = 0

    for pass_num, (offset_h, offset_w) in enumerate(additional_offsets, start=2):
        if verbose:
            print("\n" + "=" * 60)
            print(f"Pass {pass_num}: Grid offset ({offset_h}, {offset_w})")
            print("=" * 60)

        patches_n = hierarchical_tile_patches(
            segment, target_size,
            num_calibration_samples=num_calibration_samples,
            coarse_multiplier=coarse_multiplier,
            overlap_fraction=overlap_fraction,
            grid_offset=(offset_h, offset_w),
            verbose=verbose,
        )

        # Filter patches: only keep those with significant new coverage
        accepted_n = 0
        rejected_n = 0
        for bbox_2d, bbox_3d in patches_n:
            r_min, r_max, c_min, c_max = bbox_2d
            patch_area = (r_max - r_min + 1) * (c_max - c_min + 1)

            # Count how many pixels in this patch are NOT already covered
            patch_coverage = coverage_mask[r_min:r_max+1, c_min:c_max+1]
            already_covered = patch_coverage.sum()
            new_pixels = patch_area - already_covered

            # Accept if enough new coverage
            if new_pixels / patch_area >= min_new_coverage:
                coverage_mask[r_min:r_max+1, c_min:c_max+1] = True
                all_patches.append((bbox_2d, bbox_3d))
                accepted_n += 1
            else:
                rejected_n += 1

        total_accepted += accepted_n
        total_rejected += rejected_n

        if verbose:
            current_coverage = coverage_mask.sum()
            print(f"\nPass {pass_num} filtering: {accepted_n} accepted, {rejected_n} rejected")
            print(f"Coverage after pass {pass_num}: {current_coverage:,} px ({current_coverage/valid_area*100:.1f}%)")

    t_elapsed = time.time() - t_start

    if verbose:
        unique_coverage = coverage_mask.sum()
        coverage_pct = unique_coverage / valid_area * 100 if valid_area > 0 else 0
        print("\n" + "=" * 60)
        print("Multi-pass Summary")
        print("=" * 60)
        print(f"Pass 1: {len(patches_1)} patches")
        print(f"Passes 2-4: {total_accepted} patches accepted, {total_rejected} rejected")
        print(f"Total: {len(all_patches)} patches")
        print(f"Coverage: {unique_coverage:,} px ({coverage_pct:.1f}% of valid area)")
        print(f"Time: {t_elapsed:.2f}s")

    return all_patches


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse
    import zarr
    from tqdm.auto import tqdm

    # Import tifxyz utilities - adjust path as needed
    import sys
    from vesuvius.tifxyz import read_tifxyz, list_tifxyz

    def load_tifxyz_segments(segments_path, z_range, ome_zarr):
        """Load tifxyz segments and associate them with an OME-zarr volume."""
        segment_infos = list_tifxyz(segments_path, z_range=z_range, recursive=False)
        print(f'Found {len(segment_infos)} tifxyz segments')

        all_segments = []
        for info in tqdm(segment_infos, desc='Loading tifxyz segments'):
            try:
                tifxyz = info.load()
                tifxyz.volume = ome_zarr
                all_segments.append(tifxyz)
            except Exception as e:
                print(f'Error loading {info.path}: {e}')
                continue

        print(f'Loaded {len(all_segments)} tifxyz segments from {segments_path}')
        return all_segments

    # Parse arguments
    parser = argparse.ArgumentParser(description='Test hierarchical tiling on a segment')
    parser.add_argument('--volume', type=str,
                        default='/mnt/raid_nvme/volpkgs/PHercParis4.volpkg/volumes/Scroll1_8um_uint8.zarr',
                        help='Path to OME-zarr volume')
    parser.add_argument('--segments', type=str,
                        default='/mnt/raid_nvme/datasets/neural-tracer-data_2025-11-10_still-paper-43/segments/PHercParis4',
                        help='Path to tifxyz segments')
    parser.add_argument('--segment-idx', type=int, default=1,
                        help='Index of segment to use')
    parser.add_argument('--target-size', type=int, nargs=3, default=[256, 256, 256],
                        help='Target 3D volume size (d h w)')
    parser.add_argument('--coarse-multiplier', type=float, default=1.0,
                        help='Multiplier for coarse tile size')
    parser.add_argument('--overlap', type=float, default=0.25,
                        help='Overlap fraction between tiles (0.0-1.0, default 0.25 = 25%% overlap)')
    parser.add_argument('--single-pass', action='store_true',
                        help='Use single-pass tiling instead of multi-pass (multi-pass is default)')
    parser.add_argument('--min-new-coverage', type=float, default=0.5,
                        help='For multipass: min fraction of new pixels required (default 0.5)')
    parser.add_argument('--save', type=str, default="patch_viz.jpg",
                        help='Path to save visualization')
    args = parser.parse_args()

    # Load volume and segments
    print(f"Loading volume from {args.volume}")
    ome_zarr = zarr.open(args.volume, mode='r')

    print(f"Loading segments from {args.segments}")
    segments = load_tifxyz_segments(args.segments, None, ome_zarr)

    if not segments:
        print("No segments found!")
        sys.exit(1)

    seg = segments[args.segment_idx]
    print(f"\nUsing segment {args.segment_idx}: {seg.uuid}")
    print(f"  Shape: {seg.shape}")
    print(f"  Valid area: {seg._valid_mask.sum():,} pixels")

    # Run hierarchical tiling
    target_size = tuple(args.target_size)

    if not args.single_pass:
        print(f"\nRunning MULTIPASS hierarchical tiling with target_size={target_size}, coarse_multiplier={args.coarse_multiplier}, overlap={args.overlap}")

        patches = multipass_hierarchical_tiling(
            seg,
            target_size=target_size,
            coarse_multiplier=args.coarse_multiplier,
            overlap_fraction=args.overlap,
            min_new_coverage=args.min_new_coverage,
            verbose=True,
        )

    else:
        print(f"\nRunning hierarchical tiling with target_size={target_size}, coarse_multiplier={args.coarse_multiplier}, overlap={args.overlap}")

        patches = hierarchical_tile_patches(
            seg,
            target_size=target_size,
            coarse_multiplier=args.coarse_multiplier,
            overlap_fraction=args.overlap,
            verbose=True,
        )

    # Visualize
    import matplotlib.pyplot as plt
    H, W = seg.shape
    valid_area = seg._valid_mask.sum()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(seg._valid_mask, cmap='gray')
    axes[0].set_title(f'Valid mask ({valid_area:,} pixels)')

    coverage_map = np.zeros((H, W), dtype=np.uint8)
    for i, (bbox_2d, bbox_3d) in enumerate(patches):
        r_min, r_max, c_min, c_max = bbox_2d
        coverage_map[r_min:r_max+1, c_min:c_max+1] = (i % 255) + 1
    axes[1].imshow(coverage_map, cmap='tab20')
    cov_px = coverage_map.astype(bool).sum()
    axes[1].set_title(f'{len(patches)} patches, {cov_px/valid_area*100:.1f}%')

    plt.tight_layout()
    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {args.save}")
    plt.show()
