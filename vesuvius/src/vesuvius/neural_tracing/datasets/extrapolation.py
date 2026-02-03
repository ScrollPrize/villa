"""
Surface extrapolation methods for neural tracing datasets.

Supports multiple extrapolation methods via the `method` parameter.
"""
import numpy as np
import torch
import random
from typing import Callable, Optional

from .common import voxelize_surface_grid


def apply_degradation(
    zyx_local: np.ndarray,
    uv_shape: tuple,
    cond_direction: str,
    degrade_prob: float = 0.0,
    curvature_range: tuple = (0.001, 0.01),
    gradient_range: tuple = (0.05, 0.2),
) -> tuple[np.ndarray, bool]:
    """
    Apply degradation to extrapolated coordinates (full grid, before filtering).

    Args:
        zyx_local: (N, 3) local z,y,x coordinates (full UV grid flattened)
        uv_shape: (R, C) shape of UV grid for reshaping
        cond_direction: "left", "right", "up", or "down"
        degrade_prob: probability of applying degradation
        curvature_range: (min, max) for quadratic curvature coefficient
        gradient_range: (min, max) for linear gradient magnitude

    Returns:
        tuple: (degraded_coords, was_applied)
    """
    if degrade_prob <= 0.0 or random.random() > degrade_prob:
        return zyx_local, False

    # Reshape to grid to compute distance from conditioning edge
    zyx_grid = zyx_local.reshape(uv_shape + (3,))
    R, C = uv_shape

    # Compute distance from conditioning edge
    if cond_direction == "left":
        # Conditioning on left, distance increases with column
        distance = np.arange(C)[None, :].repeat(R, axis=0)
    elif cond_direction == "right":
        # Conditioning on right, distance increases as column decreases
        distance = (C - 1 - np.arange(C))[None, :].repeat(R, axis=0)
    elif cond_direction == "up":
        # Conditioning on top, distance increases with row
        distance = np.arange(R)[:, None].repeat(C, axis=1)
    else:  # down
        # Conditioning on bottom, distance increases as row decreases
        distance = (R - 1 - np.arange(R))[:, None].repeat(C, axis=1)

    distance = distance.astype(np.float64)

    # Avoid issues if all same distance
    if distance.max() < 1e-6:
        return zyx_local, False

    # Choose degradation type randomly
    if random.random() < 0.5:
        # Curvature bias: error = k * distance^2
        k = random.uniform(curvature_range[0], curvature_range[1])
        direction = np.random.randn(3)
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        error = k * (distance[:, :, None] ** 2) * direction
    else:
        # Gradient perturbation: linear tilt
        magnitude = random.uniform(gradient_range[0], gradient_range[1])
        tilt = np.random.randn(3) * magnitude
        error = distance[:, :, None] * tilt

    degraded_grid = zyx_grid + error
    return degraded_grid.reshape(-1, 3), True


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
    cond_direction: Optional[str] = None,
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
        cond_direction: optional "left", "right", "up", or "down" to override
            UV-center-based direction inference

    Returns:
        (M, 3) extrapolated ZYX coordinates
    """
    if cond_direction is not None:
        # Use explicit direction instead of UV center inference
        is_horizontal = cond_direction in ("left", "right")
        if is_horizontal:
            direction = 1 if cond_direction == "left" else -1
        else:
            direction = 1 if cond_direction == "up" else -1
    else:
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

    # Build 2D grids for ZYX (NaN for missing entries to avoid zero corruption)
    zyx_grid = np.full((n_rows, n_cols, 3), np.nan, dtype=np.float64)
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

        # Identify rows where both edge and prev are valid (not NaN)
        valid_rows_mask = ~(np.isnan(edge_zyx).any(axis=1) | np.isnan(prev_zyx).any(axis=1))
        if valid_rows_mask.any():
            median_gradient = np.nanmedian(gradient[valid_rows_mask], axis=0)
        else:
            median_gradient = np.zeros(3, dtype=np.float64)

        # For rows with NaN edge or gradient, use median gradient fallback
        for ri in range(n_rows):
            if np.isnan(edge_zyx[ri]).any():
                # Use nearest valid edge row
                valid_indices = np.where(~np.isnan(edge_zyx[:, 0]))[0]
                if len(valid_indices) > 0:
                    nearest = valid_indices[np.argmin(np.abs(valid_indices - ri))]
                    edge_zyx[ri] = edge_zyx[nearest]
                    gradient[ri] = median_gradient
            elif np.isnan(gradient[ri]).any():
                gradient[ri] = median_gradient

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

        # Identify cols where both edge and prev are valid (not NaN)
        valid_cols_mask = ~(np.isnan(edge_zyx).any(axis=1) | np.isnan(prev_zyx).any(axis=1))
        if valid_cols_mask.any():
            median_gradient = np.nanmedian(gradient[valid_cols_mask], axis=0)
        else:
            median_gradient = np.zeros(3, dtype=np.float64)

        # For cols with NaN edge or gradient, use median gradient fallback
        for ci in range(n_cols):
            if np.isnan(edge_zyx[ci]).any():
                # Use nearest valid edge col
                valid_indices = np.where(~np.isnan(edge_zyx[:, 0]))[0]
                if len(valid_indices) > 0:
                    nearest = valid_indices[np.argmin(np.abs(valid_indices - ci))]
                    edge_zyx[ci] = edge_zyx[nearest]
                    gradient[ci] = median_gradient
            elif np.isnan(gradient[ci]).any():
                gradient[ci] = median_gradient

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


@register_method('linear_rowcol')
def _extrapolate_linear_rowcol(
    uv_cond: np.ndarray,
    zyx_cond: np.ndarray,
    uv_query: np.ndarray,
    cond_direction: Optional[str] = None,
    **kwargs,
) -> np.ndarray:
    """
    Linear extrapolation fitting a line per-row or per-column.

    Detects extrapolation direction, then:
    - Horizontal: fit linear model per-row (z,y,x as function of col)
    - Vertical: fit linear model per-col (z,y,x as function of row)

    More robust than linear_edge since it uses all points in each row/col.

    Args:
        uv_cond: (N, 2) flattened UV coordinates of conditioning points
        zyx_cond: (N, 3) flattened ZYX coordinates of conditioning points
        uv_query: (M, 2) flattened UV coordinates to extrapolate to
        cond_direction: optional "left", "right", "up", or "down" to override
            UV-center-based direction inference

    Returns:
        (M, 3) extrapolated ZYX coordinates
    """
    if cond_direction is not None:
        is_horizontal = cond_direction in ("left", "right")
    else:
        # Detect extrapolation direction
        cond_center = uv_cond.mean(axis=0)
        query_center = uv_query.mean(axis=0)
        delta_row = query_center[0] - cond_center[0]
        delta_col = query_center[1] - cond_center[1]

        is_horizontal = abs(delta_col) > abs(delta_row)

    # Get unique rows and cols
    rows_unique = np.unique(uv_cond[:, 0])
    cols_unique = np.unique(uv_cond[:, 1])

    # Build lookup structures
    row_to_idx = {r: i for i, r in enumerate(rows_unique)}
    col_to_idx = {c: i for i, c in enumerate(cols_unique)}

    zyx_extrapolated = np.zeros((len(uv_query), 3), dtype=np.float64)

    if is_horizontal:
        # Fit linear model per row: zyx = a * col + b
        # Store coefficients for each row
        row_coeffs = {}  # row -> (slope, intercept) for each of z,y,x

        for row in rows_unique:
            # Get all points in this row
            mask = uv_cond[:, 0] == row
            cols_in_row = uv_cond[mask, 1]
            zyx_in_row = zyx_cond[mask]

            if len(cols_in_row) >= 2:
                # Fit linear: [col, 1] @ [a, b].T = zyx
                A = np.column_stack([cols_in_row, np.ones(len(cols_in_row))])
                coeffs, *_ = np.linalg.lstsq(A, zyx_in_row, rcond=None)
                row_coeffs[row] = coeffs  # (2, 3): [slope; intercept] for z,y,x
            else:
                # Single point: use constant extrapolation
                row_coeffs[row] = np.array([[0, 0, 0], zyx_in_row[0]])

        # Extrapolate query points
        for i, (query_row, query_col) in enumerate(uv_query):
            # Find closest row
            closest_row_idx = np.argmin(np.abs(rows_unique - query_row))
            closest_row = rows_unique[closest_row_idx]

            coeffs = row_coeffs[closest_row]
            # zyx = slope * col + intercept
            zyx_extrapolated[i] = coeffs[0] * query_col + coeffs[1]

    else:
        # Fit linear model per column: zyx = a * row + b
        col_coeffs = {}

        for col in cols_unique:
            # Get all points in this column
            mask = uv_cond[:, 1] == col
            rows_in_col = uv_cond[mask, 0]
            zyx_in_col = zyx_cond[mask]

            if len(rows_in_col) >= 2:
                A = np.column_stack([rows_in_col, np.ones(len(rows_in_col))])
                coeffs, *_ = np.linalg.lstsq(A, zyx_in_col, rcond=None)
                col_coeffs[col] = coeffs
            else:
                col_coeffs[col] = np.array([[0, 0, 0], zyx_in_col[0]])

        # Extrapolate query points
        for i, (query_row, query_col) in enumerate(uv_query):
            # Find closest column
            closest_col_idx = np.argmin(np.abs(cols_unique - query_col))
            closest_col = cols_unique[closest_col_idx]

            coeffs = col_coeffs[closest_col]
            zyx_extrapolated[i] = coeffs[0] * query_row + coeffs[1]

    return zyx_extrapolated


@register_method('polynomial')
def _extrapolate_polynomial(
    uv_cond: np.ndarray,
    zyx_cond: np.ndarray,
    uv_query: np.ndarray,
    degree: int = 2,
    **kwargs,
) -> np.ndarray:
    """
    Polynomial surface extrapolation using 2D polynomial fitting.

    Fits a polynomial f(u,v) = Σ c_ij * u^i * v^j for each output dimension.

    Args:
        uv_cond: (N, 2) flattened UV coordinates of conditioning points
        zyx_cond: (N, 3) flattened ZYX coordinates of conditioning points
        uv_query: (M, 2) flattened UV coordinates to extrapolate to
        degree: polynomial degree (2=quadratic, 3=cubic)

    Returns:
        (M, 3) extrapolated ZYX coordinates
    """
    from numpy.polynomial.polynomial import polyvander2d

    u_cond, v_cond = uv_cond[:, 0], uv_cond[:, 1]
    u_query, v_query = uv_query[:, 0], uv_query[:, 1]

    # Build Vandermonde matrix for conditioning points
    # For degree=2: columns are [1, u, v, u², uv, v², ...]
    vander_cond = polyvander2d(u_cond, v_cond, [degree, degree])

    # Fit coefficients for each output dimension (z, y, x) using least squares
    coeffs, *_ = np.linalg.lstsq(vander_cond, zyx_cond, rcond=None)

    # Evaluate at query points
    vander_query = polyvander2d(u_query, v_query, [degree, degree])
    zyx_extrapolated = vander_query @ coeffs

    return zyx_extrapolated


@register_method('bspline')
def _extrapolate_bspline(
    uv_cond: np.ndarray,
    zyx_cond: np.ndarray,
    uv_query: np.ndarray,
    degree: int = 3,
    smoothing: float = 0,
    **kwargs,
) -> np.ndarray:
    """
    B-spline surface extrapolation using scipy's bivariate spline fitting.

    Args:
        uv_cond: (N, 2) flattened UV coordinates of conditioning points
        zyx_cond: (N, 3) flattened ZYX coordinates of conditioning points
        uv_query: (M, 2) flattened UV coordinates to extrapolate to
        degree: spline degree (1=linear, 3=cubic)
        smoothing: smoothing factor (0=interpolate exactly, higher=smoother)

    Returns:
        (M, 3) extrapolated ZYX coordinates
    """
    from scipy.interpolate import bisplrep, bisplev

    u_cond, v_cond = uv_cond[:, 0], uv_cond[:, 1]
    u_query, v_query = uv_query[:, 0], uv_query[:, 1]

    zyx_extrapolated = np.zeros((len(uv_query), 3), dtype=np.float64)

    for dim in range(3):  # z, y, x
        # Fit spline to scattered data
        tck = bisplrep(u_cond, v_cond, zyx_cond[:, dim],
                       kx=degree, ky=degree, s=smoothing)
        # Evaluate at query points
        zyx_extrapolated[:, dim] = bisplev(u_query, v_query, tck, grid=False)

    return zyx_extrapolated


def compute_extrapolation(
    uv_cond: np.ndarray,
    zyx_cond: np.ndarray,
    uv_mask: np.ndarray,
    zyx_mask: np.ndarray,
    min_corner: np.ndarray,
    crop_size: tuple,
    method: str = 'rbf',
    cond_direction: Optional[str] = None,
    degrade_prob: float = 0.0,
    degrade_curvature_range: tuple = (0.001, 0.01),
    degrade_gradient_range: tuple = (0.05, 0.2),
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
        cond_direction: "left", "right", "up", or "down" (required if degrade_prob > 0)
        degrade_prob: probability of applying degradation to extrapolated coords
        degrade_curvature_range: (min, max) for quadratic curvature coefficient
        degrade_gradient_range: (min, max) for linear gradient magnitude
        **method_kwargs: additional kwargs passed to the extrapolation method

    Returns:
        dict with:
            - extrap_coords_local: (N, 3) local coords of extrapolated points
            - gt_coords_local: (N, 3) local coords of ground truth points
            - gt_displacement: (N, 3) displacement vectors (deprecated, use gt_coords_local)
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
    if uv_cond_flat.size == 0 or uv_mask_flat.size == 0:
        return None

    # Run extrapolation method
    extrapolate_fn = _EXTRAPOLATION_METHODS[method]
    zyx_extrapolated = extrapolate_fn(
        uv_cond=uv_cond_flat,
        zyx_cond=zyx_cond_flat,
        uv_query=uv_mask_flat,
        min_corner=min_corner,
        crop_size=crop_size,
        cond_direction=cond_direction,
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

    # Apply optional degradation before filtering/voxelization
    if degrade_prob > 0.0 and cond_direction is not None:
        zyx_extrap_local_full = np.stack([z_extrap_local, y_extrap_local, x_extrap_local], axis=-1)
        uv_mask_shape = uv_mask.shape[:2]
        zyx_extrap_local_full, _ = apply_degradation(
            zyx_extrap_local_full,
            uv_mask_shape,
            cond_direction,
            degrade_prob=degrade_prob,
            curvature_range=degrade_curvature_range,
            gradient_range=degrade_gradient_range,
        )
        z_extrap_local = zyx_extrap_local_full[:, 0]
        y_extrap_local = zyx_extrap_local_full[:, 1]
        x_extrap_local = zyx_extrap_local_full[:, 2]

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

    # Ground truth coords in local (crop) coordinates
    z_gt_local = z_gt - min_corner[0]
    y_gt_local = y_gt - min_corner[1]
    x_gt_local = x_gt - min_corner[2]

    gt_coords_local = np.stack([
        z_gt_local[in_bounds],
        y_gt_local[in_bounds],
        x_gt_local[in_bounds]
    ], axis=-1)  # (N, 3)

    # Displacement = ground truth - extrapolated (kept for backward compatibility)
    gt_displacement = np.stack([
        dz[in_bounds],
        dy[in_bounds],
        dx[in_bounds]
    ], axis=-1)  # (N, 3)

    # Voxelize extrapolated surface with line interpolation
    # Reshape local coords back to original UV grid shape for line drawing
    zyx_extrap_local = np.stack([z_extrap_local, y_extrap_local, x_extrap_local], axis=-1)
    uv_mask_shape = uv_mask.shape[:2]  # (R', C')
    zyx_grid_local = zyx_extrap_local.reshape(uv_mask_shape + (3,))
    extrap_surface = voxelize_surface_grid(zyx_grid_local, crop_size)

    return {
        'extrap_coords_local': extrap_coords_local,
        'gt_coords_local': gt_coords_local,  # Ground truth coords for post-augmentation displacement
        'gt_displacement': gt_displacement,  # Pre-computed displacement (deprecated, for backward compat)
        'extrap_surface': extrap_surface,
    }
