"""
Surface extrapolation methods for neural tracing datasets.

Supports multiple extrapolation methods via the `method` parameter.
"""
import numpy as np
import torch
import random
from contextlib import nullcontext
from typing import Callable, Optional

from .common import voxelize_surface_grid


def _resolve_torch_precision(value) -> torch.dtype:
    if value is None:
        return torch.float64
    if isinstance(value, torch.dtype):
        if value in (torch.float32, torch.float64):
            return value
        raise ValueError(f"Unsupported precision dtype: {value}")
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"float32", "fp32", "single", "f32"}:
            return torch.float32
        if token in {"float64", "fp64", "double", "f64"}:
            return torch.float64
    if value in (np.float32, np.dtype(np.float32)):
        return torch.float32
    if value in (np.float64, np.dtype(np.float64)):
        return torch.float64
    raise ValueError(
        "Unsupported precision value for RBF extrapolation. "
        "Use float32/fp32 or float64/fp64."
    )


def _compute_surface_normals_grid(surface_zyx: np.ndarray) -> np.ndarray:
    """Estimate per-point unit normals from local row/col tangents."""
    h, w, _ = surface_zyx.shape
    if h < 2 or w < 2:
        return np.zeros_like(surface_zyx, dtype=np.float32)

    surface = surface_zyx.astype(np.float32, copy=False)
    row_tangent = np.empty_like(surface)
    col_tangent = np.empty_like(surface)

    row_tangent[1:-1] = surface[2:] - surface[:-2]
    row_tangent[0] = surface[1] - surface[0]
    row_tangent[-1] = surface[-1] - surface[-2]

    col_tangent[:, 1:-1] = surface[:, 2:] - surface[:, :-2]
    col_tangent[:, 0] = surface[:, 1] - surface[:, 0]
    col_tangent[:, -1] = surface[:, -1] - surface[:, -2]

    normals = np.cross(col_tangent, row_tangent)
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / np.maximum(norms, 1e-6)
    normals[norms[..., 0] <= 1e-6] = 0.0
    return normals.astype(np.float32, copy=False)


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


def _run_fallback_method(
    fallback_method: Optional[str],
    uv_cond: np.ndarray,
    zyx_cond: np.ndarray,
    uv_query: np.ndarray,
    cond_direction: Optional[str],
    default: str = 'linear_edge',
    disallow: Optional[tuple[str, ...]] = None,
) -> np.ndarray:
    """Run a registered fallback method for extrapolation."""
    chosen = fallback_method or default
    if disallow and chosen in disallow:
        chosen = default
    if chosen == 'rbf':
        chosen = default

    fallback_fn = _EXTRAPOLATION_METHODS.get(chosen)
    if fallback_fn is None:
        available = list(_EXTRAPOLATION_METHODS.keys())
        raise ValueError(f"Unknown fallback extrapolation method '{chosen}'. Available: {available}")

    return fallback_fn(
        uv_cond=uv_cond,
        zyx_cond=zyx_cond,
        uv_query=uv_query,
        cond_direction=cond_direction,
    )


def _profile_section(profiler, name: str):
    if profiler is None:
        return nullcontext()
    return profiler.section(name)


def _downsample_uv_cond_directional(
    uv_cond: np.ndarray,
    zyx_cond: np.ndarray,
    stride: int,
    cond_direction: Optional[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Downsample conditioning points in a direction-aware way.

    Why: `uv_cond[::stride]` is order-dependent. For row-major edge bands this
    creates phase aliasing where small changes in edge-band width (e.g., 4->5)
    radically change which columns are selected per row.
    """
    stride = max(1, int(stride))
    if stride <= 1 or len(uv_cond) == 0:
        return uv_cond, zyx_cond

    uv = np.asarray(uv_cond)
    zyx = np.asarray(zyx_cond)

    keep_mask = np.zeros((uv.shape[0],), dtype=bool)
    if cond_direction in {"left", "right"}:
        line_vals = uv[:, 0]
        boundary_vals = -uv[:, 1] if cond_direction == "left" else uv[:, 1]
    elif cond_direction in {"up", "down"}:
        line_vals = uv[:, 1]
        boundary_vals = -uv[:, 0] if cond_direction == "up" else uv[:, 0]
    else:
        # Unknown direction: preserve legacy behavior.
        keep_mask[::stride] = True
        return uv[keep_mask], zyx[keep_mask]

    # Sort once by line, then by edge-distance ordering within each line.
    sorted_idx = np.lexsort((boundary_vals, line_vals))
    sorted_lines = line_vals[sorted_idx]
    group_starts = np.flatnonzero(np.r_[True, sorted_lines[1:] != sorted_lines[:-1]])
    group_ends = np.r_[group_starts[1:], sorted_idx.size]

    for start, end in zip(group_starts, group_ends):
        keep_mask[sorted_idx[start:end:stride]] = True

    if not keep_mask.any():
        keep_mask[::stride] = True

    return uv[keep_mask], zyx[keep_mask]


def _infer_cond_direction(
    uv_cond: np.ndarray,
    uv_query: np.ndarray,
    cond_direction: Optional[str],
) -> Optional[str]:
    """Infer conditioning-side direction if not explicitly provided."""
    if cond_direction in {"left", "right", "up", "down"}:
        return cond_direction
    if uv_cond.size == 0 or uv_query.size == 0:
        return None

    cond_center = uv_cond.mean(axis=0)
    query_center = uv_query.mean(axis=0)
    delta_row = query_center[0] - cond_center[0]
    delta_col = query_center[1] - cond_center[1]

    if abs(delta_col) > abs(delta_row):
        return "left" if delta_col > 0 else "right"
    return "up" if delta_row > 0 else "down"


@register_method('rbf')
def _extrapolate_rbf(
    uv_cond: np.ndarray,
    zyx_cond: np.ndarray,
    uv_query: np.ndarray,
    downsample_factor: int = 40,
    rbf_max_points: Optional[int] = None,
    fallback_method: str = 'linear_edge',
    singular_smoothing: float = 1e-4,
    **kwargs,
) -> np.ndarray:
    """
    RBF (Radial Basis Function) extrapolation using thin plate splines.

    Args:
        uv_cond: (N, 2) flattened UV coordinates of conditioning points
        zyx_cond: (N, 3) flattened ZYX coordinates of conditioning points
        uv_query: (M, 2) flattened UV coordinates to extrapolate to
        downsample_factor: downsample conditioning points for efficiency
        rbf_max_points: optional cap on post-downsample conditioning points.
            If set and exceeded, points are uniformly subselected.
        fallback_method: fallback method if RBF system remains singular
        singular_smoothing: smoothing floor used for singular retry

    Returns:
        (M, 3) extrapolated ZYX coordinates
    """
    from vesuvius.neural_tracing.datasets.interpolation.torch_rbf import RBFInterpolator
    profiler = kwargs.get("profiler")

    with _profile_section(profiler, "iter_edge_rbf_downsample"):
        # Downsample for RBF fitting.
        stride = max(1, int(downsample_factor))
        cond_direction = kwargs.get("cond_direction")
        uv_cond_ds, zyx_cond_ds = _downsample_uv_cond_directional(
            uv_cond,
            zyx_cond,
            stride,
            cond_direction=cond_direction,
        )

        # Optional cap on the number of control points to reduce solve cost on large crops.
        if rbf_max_points is not None:
            max_pts = int(rbf_max_points)
            if max_pts > 0 and len(uv_cond_ds) > max_pts:
                keep_idx = np.linspace(0, len(uv_cond_ds) - 1, num=max_pts, dtype=np.int64)
                uv_cond_ds = uv_cond_ds[keep_idx]
                zyx_cond_ds = zyx_cond_ds[keep_idx]

        rbf_precision = _resolve_torch_precision(kwargs.get("precision"))
        np_precision = np.float64 if rbf_precision == torch.float64 else np.float32
        uv_cond_t = torch.from_numpy(np.asarray(uv_cond_ds, dtype=np_precision))
        zyx_cond_t = torch.from_numpy(np.asarray(zyx_cond_ds, dtype=np_precision))
        uv_query_t = torch.from_numpy(np.asarray(uv_query, dtype=np_precision))

    smoothing = kwargs.get('smoothing', 0.0)
    rbf_kwargs = dict(
        y=uv_cond_t,   # input: (N, 2) UV
        d=zyx_cond_t,  # output: (N, 3) ZYX
        kernel='thin_plate_spline',
        precision=rbf_precision,
    )

    with _profile_section(profiler, "iter_edge_rbf_solve_eval"):
        try:
            rbf = RBFInterpolator(
                smoothing=smoothing,
                **rbf_kwargs,
            )
            return rbf(uv_query_t).cpu().numpy().astype(np_precision, copy=False)
        except ValueError as exc:
            # Degenerate UV geometry can make the RBF system singular for some crops.
            if "Singular matrix" not in str(exc):
                raise

            try:
                rbf = RBFInterpolator(
                    smoothing=max(float(smoothing), float(singular_smoothing)),
                    **rbf_kwargs,
                )
                return rbf(uv_query_t).cpu().numpy().astype(np_precision, copy=False)
            except ValueError as retry_exc:
                if "Singular matrix" not in str(retry_exc):
                    raise

    # Last-resort fallback for singular systems; this avoids killing dataloader workers.
    return _run_fallback_method(
        fallback_method=fallback_method,
        uv_cond=uv_cond,
        zyx_cond=zyx_cond,
        uv_query=uv_query,
        cond_direction=kwargs.get('cond_direction'),
        default='linear_edge',
        disallow=('rbf',),
    )


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


@register_method('rbf_edge_only')
def _extrapolate_rbf_edge_only(
    uv_cond: np.ndarray,
    zyx_cond: np.ndarray,
    uv_query: np.ndarray,
    cond_direction: Optional[str] = None,
    downsample_factor: int = 40,
    rbf_max_points: Optional[int] = None,
    edge_band_frac: float = 0.35,
    edge_band_cells: Optional[int] = None,
    edge_min_points: int = 128,
    fallback_method: str = 'linear_edge',
    singular_smoothing: float = 1e-4,
    smoothing: float = 0.0,
    **kwargs,
) -> np.ndarray:
    """
    RBF extrapolation using only conditioning points near the mask-adjacent edge.

    This selects an edge strip from the conditioning region in UV space, then
    runs the standard RBF solver on that subset.
    """
    if len(uv_cond) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    direction = _infer_cond_direction(uv_cond, uv_query, cond_direction)
    if direction is None:
        # If direction is unknown, fall back to full conditioning set.
        direction = cond_direction
        uv_fit = uv_cond
        zyx_fit = zyx_cond
    else:
        rows_unique = np.unique(uv_cond[:, 0])
        cols_unique = np.unique(uv_cond[:, 1])

        axis_vals = cols_unique if direction in {"left", "right"} else rows_unique
        axis = uv_cond[:, 1] if direction in {"left", "right"} else uv_cond[:, 0]
        n_axis = len(axis_vals)

        if n_axis == 0:
            uv_fit = uv_cond
            zyx_fit = zyx_cond
        else:
            if edge_band_cells is not None:
                band = int(edge_band_cells)
            else:
                frac = float(edge_band_frac)
                frac = min(max(frac, 0.0), 1.0)
                band = int(np.ceil(frac * n_axis))
            band = min(max(band, 1), n_axis)

            if direction in {"left", "up"}:
                threshold = axis_vals[-band]
                edge_mask = axis >= threshold
                edge_distance = axis_vals[-1] - axis
            else:
                threshold = axis_vals[band - 1]
                edge_mask = axis <= threshold
                edge_distance = axis - axis_vals[0]

            uv_edge = uv_cond[edge_mask]
            zyx_edge = zyx_cond[edge_mask]

            min_points = max(3, int(edge_min_points))
            if len(uv_edge) >= min_points or len(uv_edge) == len(uv_cond):
                uv_fit = uv_edge
                zyx_fit = zyx_edge
            else:
                # Expand by nearest-to-edge points until minimum count is met.
                keep = min(len(uv_cond), min_points)
                order = np.argsort(edge_distance, kind='stable')
                keep_idx = order[:keep]
                uv_fit = uv_cond[keep_idx]
                zyx_fit = zyx_cond[keep_idx]

    fb = fallback_method
    if fb == 'rbf_edge_only':
        fb = 'linear_edge'

    return _extrapolate_rbf(
        uv_cond=uv_fit,
        zyx_cond=zyx_fit,
        uv_query=uv_query,
        downsample_factor=downsample_factor,
        rbf_max_points=rbf_max_points,
        fallback_method=fb,
        singular_smoothing=singular_smoothing,
        smoothing=smoothing,
        cond_direction=direction,
        **kwargs,
    )


def generate_extended_uv(
    uv_cond: np.ndarray,
    growth_direction: str,
    extension_size: int,
) -> np.ndarray:
    """
    Generate UV coordinates extending beyond the boundary of the conditioning region.

    Args:
        uv_cond: (H, W, 2) UV coordinates of the conditioning region (the full segment)
        growth_direction: "up", "down", "left", or "right" - direction to extend
        extension_size: number of rows/columns to extend

    Returns:
        (H_e, W_e, 2) UV coordinates for the extension region only
            - up/down: (extension_size, W, 2)
            - left/right: (H, extension_size, 2)
    """
    H, W = uv_cond.shape[:2]

    # Get the row/col values from the conditioning UV grid
    rows = uv_cond[:, 0, 0]  # (H,) - row values along first column
    cols = uv_cond[0, :, 1]  # (W,) - col values along first row

    if growth_direction == 'up':
        # Extend above: rows -1, -2, ..., -extension_size (prepend)
        # Compute row step from existing grid
        row_step = rows[1] - rows[0] if H > 1 else 1.0
        new_rows = rows[0] - row_step * np.arange(extension_size, 0, -1)  # ascending order
        # Create extended UV grid
        extrap_uv = np.zeros((extension_size, W, 2), dtype=uv_cond.dtype)
        extrap_uv[:, :, 0] = new_rows[:, None]  # broadcast rows
        extrap_uv[:, :, 1] = cols[None, :]  # broadcast cols

    elif growth_direction == 'down':
        # Extend below: rows H, H+1, ..., H+extension_size-1 (append)
        row_step = rows[-1] - rows[-2] if H > 1 else 1.0
        new_rows = rows[-1] + row_step * np.arange(1, extension_size + 1)
        extrap_uv = np.zeros((extension_size, W, 2), dtype=uv_cond.dtype)
        extrap_uv[:, :, 0] = new_rows[:, None]
        extrap_uv[:, :, 1] = cols[None, :]

    elif growth_direction == 'left':
        # Extend left: cols -1, -2, ..., -extension_size (prepend)
        col_step = cols[1] - cols[0] if W > 1 else 1.0
        new_cols = cols[0] - col_step * np.arange(extension_size, 0, -1)  # ascending order
        extrap_uv = np.zeros((H, extension_size, 2), dtype=uv_cond.dtype)
        extrap_uv[:, :, 0] = rows[:, None]
        extrap_uv[:, :, 1] = new_cols[None, :]

    elif growth_direction == 'right':
        # Extend right: cols W, W+1, ..., W+extension_size-1 (append)
        col_step = cols[-1] - cols[-2] if W > 1 else 1.0
        new_cols = cols[-1] + col_step * np.arange(1, extension_size + 1)
        extrap_uv = np.zeros((H, extension_size, 2), dtype=uv_cond.dtype)
        extrap_uv[:, :, 0] = rows[:, None]
        extrap_uv[:, :, 1] = new_cols[None, :]

    else:
        raise ValueError(f"Unknown growth_direction: {growth_direction}. "
                         f"Expected 'up', 'down', 'left', or 'right'.")

    return extrap_uv


def compute_boundary_extrapolation(
    uv_cond: np.ndarray,
    zyx_cond: np.ndarray,
    growth_direction: str,
    extension_size: int,
    method: str = 'linear_edge',
    **method_kwargs,
) -> dict:
    """
    Extrapolate beyond the segment boundary to generate an extended grid.

    This function is designed for use with `split_segment_by_growth_direction`
    when `at_boundary=True`, where the entire segment becomes the conditioning
    region and we want to predict new points beyond the edge.

    Args:
        uv_cond: (H, W, 2) UV coordinates of conditioning region (the full segment)
        zyx_cond: (H, W, 3) ZYX world coordinates of conditioning region
        growth_direction: "up", "down", "left", or "right" - direction to extend
        extension_size: number of rows/columns to extend beyond the boundary
        method: extrapolation method ('linear_edge', 'rbf', etc.)
        **method_kwargs: additional kwargs passed to the extrapolation method

    Returns:
        dict with:
            - extended_uv: (H_ext, W_ext, 2) full UV grid (cond + extrapolated)
            - extended_zyx: (H_ext, W_ext, 3) full ZYX grid (cond + extrapolated)
            - extension_mask: (H_ext, W_ext) bool, True for extrapolated points
            - extrap_uv: (H_e, W_e, 2) just the extension region UV
            - extrap_zyx: (H_e, W_e, 3) just the extension region ZYX

    Raises:
        ValueError: if method is unknown or growth_direction is invalid
    """
    if method not in _EXTRAPOLATION_METHODS:
        available = list(_EXTRAPOLATION_METHODS.keys())
        raise ValueError(f"Unknown extrapolation method '{method}'. Available: {available}")

    H, W = uv_cond.shape[:2]

    # Generate extended UV coordinates
    extrap_uv = generate_extended_uv(uv_cond, growth_direction, extension_size)
    H_e, W_e = extrap_uv.shape[:2]

    # Flatten for extrapolation
    uv_cond_flat = uv_cond.reshape(-1, 2)
    zyx_cond_flat = zyx_cond.reshape(-1, 3)
    uv_query_flat = extrap_uv.reshape(-1, 2)

    # Map growth_direction to cond_direction for the extrapolation method
    # growth_direction is where we want to GO, cond_direction is where conditioning IS
    cond_direction_map = {
        'up': 'down',     # extending upward means conditioning is below
        'down': 'up',     # extending downward means conditioning is above
        'left': 'right',  # extending left means conditioning is to the right
        'right': 'left',  # extending right means conditioning is to the left
    }
    cond_direction = cond_direction_map[growth_direction]

    # Run extrapolation method
    extrapolate_fn = _EXTRAPOLATION_METHODS[method]
    zyx_extrapolated_flat = extrapolate_fn(
        uv_cond=uv_cond_flat,
        zyx_cond=zyx_cond_flat,
        uv_query=uv_query_flat,
        cond_direction=cond_direction,
        **method_kwargs,
    )

    # Reshape extrapolated ZYX back to grid form
    extrap_zyx = zyx_extrapolated_flat.reshape(H_e, W_e, 3)

    # Concatenate conditioning and extrapolated grids based on direction
    if growth_direction == 'up':
        # extrap goes above cond: [extrap, cond] along axis 0
        extended_uv = np.concatenate([extrap_uv, uv_cond], axis=0)
        extended_zyx = np.concatenate([extrap_zyx, zyx_cond], axis=0)
        # Mask: first extension_size rows are extrapolated
        extension_mask = np.zeros((H_e + H, W), dtype=bool)
        extension_mask[:H_e, :] = True

    elif growth_direction == 'down':
        # extrap goes below cond: [cond, extrap] along axis 0
        extended_uv = np.concatenate([uv_cond, extrap_uv], axis=0)
        extended_zyx = np.concatenate([zyx_cond, extrap_zyx], axis=0)
        # Mask: last extension_size rows are extrapolated
        extension_mask = np.zeros((H + H_e, W), dtype=bool)
        extension_mask[H:, :] = True

    elif growth_direction == 'left':
        # extrap goes left of cond: [extrap, cond] along axis 1
        extended_uv = np.concatenate([extrap_uv, uv_cond], axis=1)
        extended_zyx = np.concatenate([extrap_zyx, zyx_cond], axis=1)
        # Mask: first extension_size cols are extrapolated
        extension_mask = np.zeros((H, W_e + W), dtype=bool)
        extension_mask[:, :W_e] = True

    elif growth_direction == 'right':
        # extrap goes right of cond: [cond, extrap] along axis 1
        extended_uv = np.concatenate([uv_cond, extrap_uv], axis=1)
        extended_zyx = np.concatenate([zyx_cond, extrap_zyx], axis=1)
        # Mask: last extension_size cols are extrapolated
        extension_mask = np.zeros((H, W + W_e), dtype=bool)
        extension_mask[:, W:] = True

    return {
        'extended_uv': extended_uv,
        'extended_zyx': extended_zyx,
        'extension_mask': extension_mask,
        'extrap_uv': extrap_uv,
        'extrap_zyx': extrap_zyx,
    }


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
    debug_no_in_bounds: bool = False,
    debug_no_in_bounds_every: int = 1,
    return_gt_normals: bool = False,
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

    gt_normals_flat = None
    if return_gt_normals:
        gt_normals_grid = _compute_surface_normals_grid(zyx_mask)
        gt_normals_flat = gt_normals_grid.reshape(-1, 3)

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
        if debug_no_in_bounds:
            every = max(1, int(debug_no_in_bounds_every))
            count = getattr(compute_extrapolation, "_debug_no_in_bounds_count", 0) + 1
            compute_extrapolation._debug_no_in_bounds_count = count

            if (count % every) == 0:
                print(
                    f"DEBUG: No extrapolated points in bounds "
                    f"(count={count}, every={every})"
                )
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

    gt_normals_local = None
    if gt_normals_flat is not None:
        gt_normals_local = gt_normals_flat[in_bounds].astype(np.float32, copy=False)

    # Voxelize extrapolated surface with line interpolation
    # Reshape local coords back to original UV grid shape for line drawing
    zyx_extrap_local = np.stack([z_extrap_local, y_extrap_local, x_extrap_local], axis=-1)
    uv_mask_shape = uv_mask.shape[:2]  # (R', C')
    zyx_grid_local = zyx_extrap_local.reshape(uv_mask_shape + (3,))
    extrap_surface = voxelize_surface_grid(zyx_grid_local, crop_size)

    result = {
        'extrap_coords_local': extrap_coords_local,
        'gt_coords_local': gt_coords_local,  # Ground truth coords for post-augmentation displacement
        'gt_displacement': gt_displacement,  # Pre-computed displacement (deprecated, for backward compat)
        'extrap_surface': extrap_surface,
    }
    if gt_normals_local is not None:
        result['gt_normals_local'] = gt_normals_local
    return result
