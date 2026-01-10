"""
Surface extrapolation methods for neural tracing datasets.

Supports multiple extrapolation methods via the `method` parameter.
"""
import numpy as np
import torch
from typing import Callable


# Registry of extrapolation methods
_EXTRAPOLATION_METHODS: dict[str, Callable] = {}


def register_method(name: str):
    """Decorator to register an extrapolation method."""
    def decorator(fn):
        _EXTRAPOLATION_METHODS[name] = fn
        return fn
    return decorator


@register_method('rbf')
def _extrapolate_rbf(
    uv_cond: np.ndarray,
    zyx_cond: np.ndarray,
    uv_query: np.ndarray,
    downsample_factor: int = 40,
    **kwargs,
) -> np.ndarray:
    """
    RBF (Radial Basis Function) extrapolation using thin plate splines.

    Args:
        uv_cond: (N, 2) flattened UV coordinates of conditioning points
        zyx_cond: (N, 3) flattened ZYX coordinates of conditioning points
        uv_query: (M, 2) flattened UV coordinates to extrapolate to
        downsample_factor: downsample conditioning points for efficiency

    Returns:
        (M, 3) extrapolated ZYX coordinates
    """
    from vesuvius.neural_tracing.datasets.interpolation.torch_rbf import RBFInterpolator

    # Downsample for RBF fitting
    uv_cond_ds = uv_cond[::downsample_factor]
    zyx_cond_ds = zyx_cond[::downsample_factor]

    # Fit RBF interpolator
    rbf = RBFInterpolator(
        y=torch.from_numpy(uv_cond_ds).float(),   # input: (N, 2) UV
        d=torch.from_numpy(zyx_cond_ds).float(),  # output: (N, 3) ZYX
        kernel='thin_plate_spline'
    )

    # Extrapolate
    zyx_extrapolated = rbf(torch.from_numpy(uv_query).float()).numpy()
    return zyx_extrapolated


@register_method('linear_edge')
def _extrapolate_linear_edge(
    uv_cond: np.ndarray,
    zyx_cond: np.ndarray,
    uv_query: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """
    Linear extrapolation using gradients at the conditioning edge.

    Detects the direction of extrapolation, extracts the edge of the
    conditioning region, computes local gradients, and extrapolates linearly.

    Args:
        uv_cond: (N, 2) flattened UV coordinates of conditioning points
        zyx_cond: (N, 3) flattened ZYX coordinates of conditioning points
        uv_query: (M, 2) flattened UV coordinates to extrapolate to

    Returns:
        (M, 3) extrapolated ZYX coordinates
    """
    # Detect extrapolation direction by comparing UV centers
    cond_center = uv_cond.mean(axis=0)
    query_center = uv_query.mean(axis=0)

    delta_row = query_center[0] - cond_center[0]
    delta_col = query_center[1] - cond_center[1]

    # Determine primary direction (row vs col)
    if abs(delta_col) > abs(delta_row):
        # Horizontal extrapolation (left/right)
        is_horizontal = True
        direction = 1 if delta_col > 0 else -1  # +1 = rightward, -1 = leftward
    else:
        # Vertical extrapolation (up/down)
        is_horizontal = False
        direction = 1 if delta_row > 0 else -1  # +1 = downward, -1 = upward

    # Reconstruct 2D grid from flattened conditioning data
    rows_unique = np.unique(uv_cond[:, 0])
    cols_unique = np.unique(uv_cond[:, 1])
    n_rows, n_cols = len(rows_unique), len(cols_unique)

    # Create lookup from (row, col) -> index
    row_to_idx = {r: i for i, r in enumerate(rows_unique)}
    col_to_idx = {c: i for i, c in enumerate(cols_unique)}

    # Build 2D grids for ZYX
    zyx_grid = np.zeros((n_rows, n_cols, 3), dtype=np.float64)
    for i, (uv, zyx) in enumerate(zip(uv_cond, zyx_cond)):
        ri, ci = row_to_idx[uv[0]], col_to_idx[uv[1]]
        zyx_grid[ri, ci] = zyx

    if is_horizontal:
        # Extract edge column and compute gradient
        if direction > 0:
            # Rightward: use right edge (last column)
            edge_col_idx = -1
            prev_col_idx = -2 if n_cols > 1 else -1
        else:
            # Leftward: use left edge (first column)
            edge_col_idx = 0
            prev_col_idx = 1 if n_cols > 1 else 0

        edge_zyx = zyx_grid[:, edge_col_idx, :]  # (n_rows, 3)
        prev_zyx = zyx_grid[:, prev_col_idx, :]  # (n_rows, 3)

        edge_col = cols_unique[edge_col_idx]
        prev_col = cols_unique[prev_col_idx]
        col_step = edge_col - prev_col if edge_col != prev_col else 1.0

        # Gradient per row: dZYX / dcol
        gradient = (edge_zyx - prev_zyx) / col_step  # (n_rows, 3)

        # For each query point, find matching row and extrapolate
        zyx_extrapolated = np.zeros((len(uv_query), 3), dtype=np.float64)
        for i, uv in enumerate(uv_query):
            query_row, query_col = uv[0], uv[1]

            # Find closest row in conditioning data
            row_idx = np.argmin(np.abs(rows_unique - query_row))

            # Distance from edge in col direction
            delta = query_col - edge_col

            # Linear extrapolation
            zyx_extrapolated[i] = edge_zyx[row_idx] + gradient[row_idx] * delta

    else:
        # Vertical extrapolation
        if direction > 0:
            # Downward: use bottom edge (last row)
            edge_row_idx = -1
            prev_row_idx = -2 if n_rows > 1 else -1
        else:
            # Upward: use top edge (first row)
            edge_row_idx = 0
            prev_row_idx = 1 if n_rows > 1 else 0

        edge_zyx = zyx_grid[edge_row_idx, :, :]  # (n_cols, 3)
        prev_zyx = zyx_grid[prev_row_idx, :, :]  # (n_cols, 3)

        edge_row = rows_unique[edge_row_idx]
        prev_row = rows_unique[prev_row_idx]
        row_step = edge_row - prev_row if edge_row != prev_row else 1.0

        # Gradient per col: dZYX / drow
        gradient = (edge_zyx - prev_zyx) / row_step  # (n_cols, 3)

        # For each query point, find matching col and extrapolate
        zyx_extrapolated = np.zeros((len(uv_query), 3), dtype=np.float64)
        for i, uv in enumerate(uv_query):
            query_row, query_col = uv[0], uv[1]

            # Find closest col in conditioning data
            col_idx = np.argmin(np.abs(cols_unique - query_col))

            # Distance from edge in row direction
            delta = query_row - edge_row

            # Linear extrapolation
            zyx_extrapolated[i] = edge_zyx[col_idx] + gradient[col_idx] * delta

    return zyx_extrapolated


@register_method('rbf_clamped')
def _extrapolate_rbf_clamped(
    uv_cond: np.ndarray,
    zyx_cond: np.ndarray,
    uv_query: np.ndarray,
    min_corner: np.ndarray = None,
    crop_size: tuple = None,
    margin_factor: float = 0.5,
    downsample_factor: int = 40,
    **kwargs,
) -> np.ndarray:
    """
    RBF extrapolation with output clamping to prevent extreme values.

    Same as RBF but clamps extrapolated coordinates to crop bounds + margin.
    Points outside the actual crop bounds are filtered out later by
    compute_extrapolation's in-bounds check.

    Args:
        uv_cond: (N, 2) flattened UV coordinates of conditioning points
        zyx_cond: (N, 3) flattened ZYX coordinates of conditioning points
        uv_query: (M, 2) flattened UV coordinates to extrapolate to
        min_corner: (3,) origin of crop in world coords (z, y, x)
        crop_size: (D, H, W) size of crop
        margin_factor: extra margin as fraction of crop size (default 0.5 = 50%)
        downsample_factor: downsample conditioning points for efficiency

    Returns:
        (M, 3) extrapolated ZYX coordinates, clamped to generous bounds
    """
    # Run standard RBF extrapolation
    zyx_extrapolated = _extrapolate_rbf(
        uv_cond=uv_cond,
        zyx_cond=zyx_cond,
        uv_query=uv_query,
        downsample_factor=downsample_factor,
    )

    # Clamp to crop bounds + margin (in world coordinates)
    # The margin allows some flexibility; downstream filtering removes OOB points
    if min_corner is not None and crop_size is not None:
        crop_size_arr = np.asarray(crop_size)
        margin = crop_size_arr * margin_factor
        zyx_min = np.asarray(min_corner) - margin
        zyx_max = np.asarray(min_corner) + crop_size_arr + margin
        zyx_extrapolated = np.clip(zyx_extrapolated, zyx_min, zyx_max)

    return zyx_extrapolated


def compute_extrapolation(
    uv_cond: np.ndarray,
    zyx_cond: np.ndarray,
    uv_mask: np.ndarray,
    zyx_mask: np.ndarray,
    min_corner: np.ndarray,
    crop_size: tuple,
    method: str = 'rbf',
    **method_kwargs,
) -> dict:
    """
    Compute surface extrapolation from conditioning region to masked region.

    Args:
        uv_cond: (R, C, 2) UV coordinates of conditioning region
        zyx_cond: (R, C, 3) ZYX world coordinates of conditioning region
        uv_mask: (R', C', 2) UV coordinates of masked region
        zyx_mask: (R', C', 3) ground truth ZYX world coordinates of masked region
        min_corner: (3,) origin of crop in world coords (z, y, x)
        crop_size: (D, H, W) size of crop
        method: extrapolation method to use (default: 'rbf')
        **method_kwargs: additional kwargs passed to the extrapolation method

    Returns:
        dict with:
            - extrap_coords_local: (N, 3) local coords of extrapolated points
            - gt_displacement: (N, 3) displacement vectors (ground truth - extrapolated)
            - extrap_surface: (D, H, W) voxelized extrapolated surface

    Raises:
        ValueError: if method is unknown or no extrapolated points are within crop bounds
    """
    if method not in _EXTRAPOLATION_METHODS:
        available = list(_EXTRAPOLATION_METHODS.keys())
        raise ValueError(f"Unknown extrapolation method '{method}'. Available: {available}")

    # Flatten inputs
    uv_cond_flat = uv_cond.reshape(-1, 2)
    zyx_cond_flat = zyx_cond.reshape(-1, 3)
    uv_mask_flat = uv_mask.reshape(-1, 2)
    zyx_mask_flat = zyx_mask.reshape(-1, 3)

    # Run extrapolation method
    extrapolate_fn = _EXTRAPOLATION_METHODS[method]
    zyx_extrapolated = extrapolate_fn(
        uv_cond=uv_cond_flat,
        zyx_cond=zyx_cond_flat,
        uv_query=uv_mask_flat,
        min_corner=min_corner,
        crop_size=crop_size,
        **method_kwargs,
    )

    # Unpack extrapolated coordinates
    z_extrap = zyx_extrapolated[:, 0]
    y_extrap = zyx_extrapolated[:, 1]
    x_extrap = zyx_extrapolated[:, 2]

    # Ground truth masked coords
    z_gt = zyx_mask_flat[:, 0]
    y_gt = zyx_mask_flat[:, 1]
    x_gt = zyx_mask_flat[:, 2]

    # Displacement = ground truth - extrapolated
    dz = z_gt - z_extrap
    dy = y_gt - y_extrap
    dx = x_gt - x_extrap

    # Convert to local (crop) coordinates
    z_extrap_local = z_extrap - min_corner[0]
    y_extrap_local = y_extrap - min_corner[1]
    x_extrap_local = x_extrap - min_corner[2]

    # Filter to in-bounds points
    in_bounds = (
        (z_extrap_local >= 0) & (z_extrap_local < crop_size[0]) &
        (y_extrap_local >= 0) & (y_extrap_local < crop_size[1]) &
        (x_extrap_local >= 0) & (x_extrap_local < crop_size[2])
    )

    if in_bounds.sum() == 0:
        print(f"DEBUG: No extrapolated points in bounds")
        print(f"  crop_size: {crop_size}")
        print(f"  min_corner: {min_corner}")
        print(f"  uv_cond range: rows [{uv_cond_flat[:, 0].min():.0f}, {uv_cond_flat[:, 0].max():.0f}], cols [{uv_cond_flat[:, 1].min():.0f}, {uv_cond_flat[:, 1].max():.0f}]")
        print(f"  uv_query range: rows [{uv_mask_flat[:, 0].min():.0f}, {uv_mask_flat[:, 0].max():.0f}], cols [{uv_mask_flat[:, 1].min():.0f}, {uv_mask_flat[:, 1].max():.0f}]")
        print(f"  zyx_cond (training) range: z [{zyx_cond_flat[:, 0].min():.1f}, {zyx_cond_flat[:, 0].max():.1f}], y [{zyx_cond_flat[:, 1].min():.1f}, {zyx_cond_flat[:, 1].max():.1f}], x [{zyx_cond_flat[:, 2].min():.1f}, {zyx_cond_flat[:, 2].max():.1f}]")
        print(f"  zyx_extrapolated range: z [{z_extrap.min():.1f}, {z_extrap.max():.1f}], y [{y_extrap.min():.1f}, {y_extrap.max():.1f}], x [{x_extrap.min():.1f}, {x_extrap.max():.1f}]")
        print(f"  local coords range: z [{z_extrap_local.min():.1f}, {z_extrap_local.max():.1f}], y [{y_extrap_local.min():.1f}, {y_extrap_local.max():.1f}], x [{x_extrap_local.min():.1f}, {x_extrap_local.max():.1f}]")
        return None  # Let caller handle retry

    # Build outputs for in-bounds points only
    extrap_coords_local = np.stack([
        z_extrap_local[in_bounds],
        y_extrap_local[in_bounds],
        x_extrap_local[in_bounds]
    ], axis=-1)  # (N, 3)

    gt_displacement = np.stack([
        dz[in_bounds],
        dy[in_bounds],
        dx[in_bounds]
    ], axis=-1)  # (N, 3)

    # Voxelize extrapolated surface
    z_vox = np.round(z_extrap_local).astype(np.int64)
    y_vox = np.round(y_extrap_local).astype(np.int64)
    x_vox = np.round(x_extrap_local).astype(np.int64)

    z_vox_valid = np.clip(z_vox[in_bounds], 0, crop_size[0] - 1)
    y_vox_valid = np.clip(y_vox[in_bounds], 0, crop_size[1] - 1)
    x_vox_valid = np.clip(x_vox[in_bounds], 0, crop_size[2] - 1)

    extrap_surface = np.zeros(crop_size, dtype=np.float32)
    extrap_surface[z_vox_valid, y_vox_valid, x_vox_valid] = 1.0

    return {
        'extrap_coords_local': extrap_coords_local,
        'gt_displacement': gt_displacement,
        'extrap_surface': extrap_surface,
    }
