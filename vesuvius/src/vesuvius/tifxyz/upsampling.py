"""Upsampling and interpolation utilities for tifxyz surfaces."""

from __future__ import annotations

from typing import Literal, Tuple

import cv2
import numpy as np
from numba import njit, prange
from numpy.typing import NDArray
from scipy import ndimage


@njit(fastmath=True)
def _catmull_rom_smooth_1d_kernel(
    ext_x: NDArray[np.float64],
    ext_y: NDArray[np.float64],
    ext_z: NDArray[np.float64],
    out_x: NDArray[np.float64],
    out_y: NDArray[np.float64],
    out_z: NDArray[np.float64],
    n: int,
) -> None:
    """Apply 1D Catmull-Rom smoothing kernel to extended coordinate sequence.

    Uses CR weights at t=0.5: [-1/16, 9/16, 9/16, -1/16]
    Expects extended arrays with phantom points already filled.
    """
    w0, w1, w2, w3 = -0.0625, 0.5625, 0.5625, -0.0625

    for i in range(n):
        # Extended array has 1 phantom point at start, so offset by 0
        # ext[i] = P_{i-1}, ext[i+1] = P_i, ext[i+2] = P_{i+1}, ext[i+3] = P_{i+2}
        out_x[i] = w0 * ext_x[i] + w1 * ext_x[i + 1] + w2 * ext_x[i + 2] + w3 * ext_x[i + 3]
        out_y[i] = w0 * ext_y[i] + w1 * ext_y[i + 1] + w2 * ext_y[i + 2] + w3 * ext_y[i + 3]
        out_z[i] = w0 * ext_z[i] + w1 * ext_z[i + 1] + w2 * ext_z[i + 2] + w3 * ext_z[i + 3]


def catmull_rom_smooth_1d(
    x: NDArray[np.float32],
    y: NDArray[np.float32],
    z: NDArray[np.float32],
) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Apply 1D Catmull-Rom smoothing to coordinate arrays.

    Applies a 4-tap smoothing filter using Catmull-Rom weights at t=0.5:
    [-1/16, 9/16, 9/16, -1/16]. This provides local smoothing while
    preserving the general shape of the curve.

    Edge points are handled by linearly extrapolating phantom control points
    beyond the boundaries, ensuring proper smoothing all the way to the edges.

    Parameters
    ----------
    x, y, z : NDArray[np.float32]
        1D coordinate arrays of the same length.

    Returns
    -------
    Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]
        Smoothed (x, y, z) coordinate arrays.
        Returns empty arrays if input has < 2 points (need 2 for extrapolation).
    """
    n = len(x)
    if n < 2:
        return (
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
        )

    # Create extended arrays with phantom points:
    # ext[0] = phantom before start
    # ext[1..n] = original points
    # ext[n+1], ext[n+2] = phantoms after end
    ext_x = np.empty(n + 3, dtype=np.float64)
    ext_y = np.empty(n + 3, dtype=np.float64)
    ext_z = np.empty(n + 3, dtype=np.float64)

    # Copy original points (offset by 1)
    ext_x[1 : n + 1] = x.astype(np.float64)
    ext_y[1 : n + 1] = y.astype(np.float64)
    ext_z[1 : n + 1] = z.astype(np.float64)

    # Linear extrapolation for phantom points
    # Before: P_{-1} = 2*P_0 - P_1
    ext_x[0] = 2.0 * ext_x[1] - ext_x[2]
    ext_y[0] = 2.0 * ext_y[1] - ext_y[2]
    ext_z[0] = 2.0 * ext_z[1] - ext_z[2]

    # After: P_n = 2*P_{n-1} - P_{n-2}, P_{n+1} = 2*P_n - P_{n-1}
    ext_x[n + 1] = 2.0 * ext_x[n] - ext_x[n - 1]
    ext_y[n + 1] = 2.0 * ext_y[n] - ext_y[n - 1]
    ext_z[n + 1] = 2.0 * ext_z[n] - ext_z[n - 1]

    ext_x[n + 2] = 2.0 * ext_x[n + 1] - ext_x[n]
    ext_y[n + 2] = 2.0 * ext_y[n + 1] - ext_y[n]
    ext_z[n + 2] = 2.0 * ext_z[n + 1] - ext_z[n]

    out_x = np.empty(n, dtype=np.float64)
    out_y = np.empty(n, dtype=np.float64)
    out_z = np.empty(n, dtype=np.float64)

    _catmull_rom_smooth_1d_kernel(ext_x, ext_y, ext_z, out_x, out_y, out_z, n)

    return (
        out_x.astype(np.float32),
        out_y.astype(np.float32),
        out_z.astype(np.float32),
    )


@njit(parallel=True, fastmath=True)
def _catmull_rom_core(data, qy, qx, out, h, w, invalid_value, mask):
    """Numba-accelerated Catmull-Rom interpolation core.

    Only interpolates if ALL 16 pixels in the 4x4 kernel are valid.
    """
    for idx in prange(len(qy)):
        y, x = qy[idx], qx[idx]
        if y < 0 or y >= h or x < 0 or x >= w:
            out[idx] = invalid_value
            continue

        iy, ix = int(y), int(x)

        # Check that all 16 pixels in the 4x4 kernel are valid
        all_valid = True
        for j in range(4):
            ny = min(max(iy + j - 1, 0), h - 1)
            for i in range(4):
                nx = min(max(ix + i - 1, 0), w - 1)
                if not mask[ny, nx]:
                    all_valid = False
                    break
            if not all_valid:
                break

        if not all_valid:
            out[idx] = invalid_value
            continue

        ty, tx = y - iy, x - ix
        ty2, ty3 = ty * ty, ty * ty * ty
        tx2, tx3 = tx * tx, tx * tx * tx

        # Catmull-Rom weights
        wy0 = -0.5 * ty3 + ty2 - 0.5 * ty
        wy1 = 1.5 * ty3 - 2.5 * ty2 + 1.0
        wy2 = -1.5 * ty3 + 2.0 * ty2 + 0.5 * ty
        wy3 = 0.5 * ty3 - 0.5 * ty2

        wx0 = -0.5 * tx3 + tx2 - 0.5 * tx
        wx1 = 1.5 * tx3 - 2.5 * tx2 + 1.0
        wx2 = -1.5 * tx3 + 2.0 * tx2 + 0.5 * tx
        wx3 = 0.5 * tx3 - 0.5 * tx2

        wy = (wy0, wy1, wy2, wy3)
        wx = (wx0, wx1, wx2, wx3)

        val = 0.0
        for j in range(4):
            ny = min(max(iy + j - 1, 0), h - 1)
            for i in range(4):
                nx = min(max(ix + i - 1, 0), w - 1)
                val += wy[j] * wx[i] * data[ny, nx]
        out[idx] = val


def _interpolate_catmull_rom(
    data: NDArray[np.float32],
    query_y: NDArray[np.float32],
    query_x: NDArray[np.float32],
    mask: NDArray[np.bool_],
    invalid_value: float = -1.0,
) -> NDArray[np.float32]:
    """Numba-accelerated 2D Catmull-Rom interpolation.

    Only interpolates points where the entire 4x4 kernel is valid.
    """
    h, w = data.shape
    output_shape = query_y.shape

    qy = query_y.ravel().astype(np.float64)
    qx = query_x.ravel().astype(np.float64)
    out = np.empty(len(qy), dtype=np.float64)

    # Keep data as float32 - weights are float64, accumulator is float64,
    # so the multiply-accumulate auto-promotes. Avoids copying the entire grid.
    _catmull_rom_core(data, qy, qx, out, h, w, invalid_value, mask)

    return out.astype(np.float32).reshape(output_shape)


@njit(parallel=True)
def _check_bilinear_validity(mask, qy, qx, h, w):
    """Check if all 4 pixels in the bilinear kernel are valid.

    Returns a boolean array indicating validity for each query point.
    """
    n = len(qy)
    valid = np.empty(n, dtype=np.bool_)

    for idx in prange(n):
        y, x = qy[idx], qx[idx]

        # Out of bounds
        if y < 0 or y >= h - 1 or x < 0 or x >= w - 1:
            valid[idx] = False
            continue

        iy, ix = int(y), int(x)

        # Check all 4 corners of bilinear kernel
        if (mask[iy, ix] and mask[iy, ix + 1] and
            mask[iy + 1, ix] and mask[iy + 1, ix + 1]):
            valid[idx] = True
        else:
            valid[idx] = False

    return valid


@njit(parallel=True)
def _check_bspline_validity(mask, qy, qx, h, w, order):
    """Check if all pixels in the B-spline kernel are valid.

    B-spline of order n uses n+1 pixels in each dimension.
    """
    n = len(qy)
    valid = np.empty(n, dtype=np.bool_)
    # B-spline of order n uses (n+1) points per dimension, spanning indices
    # from floor(x) - (n-1)//2 to floor(x) + n//2 + 1. We check a symmetric
    # region of radius = (order + 1) // 2, which is conservative but safe.
    # order=2: radius=1 (3 pts), order=3: radius=2 (4 pts), order=5: radius=3 (6 pts)
    radius = (order + 1) // 2

    for idx in prange(n):
        y, x = qy[idx], qx[idx]

        # Out of bounds (with kernel margin)
        if y < radius or y >= h - radius or x < radius or x >= w - radius:
            valid[idx] = False
            continue

        iy, ix = int(y), int(x)

        # Check kernel neighborhood
        all_valid = True
        for j in range(-radius, radius + 1):
            for i in range(-radius, radius + 1):
                ny = min(max(iy + j, 0), h - 1)
                nx = min(max(ix + i, 0), w - 1)
                if not mask[ny, nx]:
                    all_valid = False
                    break
            if not all_valid:
                break

        valid[idx] = all_valid

    return valid


def upsample_coordinates(
    x: NDArray[np.float32],
    y: NDArray[np.float32],
    z: NDArray[np.float32],
    mask: NDArray[np.bool_],
    source_scale: Tuple[float, float],
    target_scale: float = 1.0,
    order: int = 1,
) -> Tuple[
    NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_]
]:
    """Upsample coordinate arrays to target scale.

    This implements the Python equivalent of C++ QuadSurface::gen() using
    OpenCV for fast interpolation (matching cv::warpAffine with INTER_LINEAR).

    Only interpolates where all pixels in the interpolation kernel are valid,
    to avoid contamination from invalid sentinel values (-1, -1, -1).

    Parameters
    ----------
    x, y, z : NDArray[np.float32]
        Coordinate arrays at source scale, shape (H, W).
    mask : NDArray[np.bool_]
        Validity mask at source scale.
    source_scale : Tuple[float, float]
        Current scale (scale_y, scale_x), typically (20.0, 20.0).
    target_scale : float
        Target scale factor. 1.0 = full resolution.
    order : int
        Interpolation order (0=nearest, 1=bilinear, 3=bicubic).

    Returns
    -------
    Tuple[NDArray, NDArray, NDArray, NDArray]
        Upsampled (x, y, z, mask) arrays.
    """
    # Calculate output size
    zoom_y = source_scale[0] / target_scale
    zoom_x = source_scale[1] / target_scale

    h, w = x.shape
    new_h = int(round(h * zoom_y))
    new_w = int(round(w * zoom_x))

    # Map order to OpenCV interpolation flag and erode mask appropriately
    # Erosion ensures interpolation kernels never touch invalid pixels
    if order == 0:
        interp = cv2.INTER_NEAREST
        # Nearest neighbor uses 1 pixel - no erosion needed
        mask_eroded = mask
    elif order == 1:
        interp = cv2.INTER_LINEAR
        # Bilinear uses 2x2 kernel: pixels at (y,x), (y,x+1), (y+1,x), (y+1,x+1)
        # Erode with 2x2 kernel anchored at top-left to check exactly those 4 pixels
        erosion_kernel = np.ones((2, 2), dtype=np.uint8)
        mask_eroded = cv2.erode(
            mask.astype(np.uint8), erosion_kernel, anchor=(0, 0)
        ).astype(bool)
    elif order == 3:
        interp = cv2.INTER_CUBIC
        # Bicubic uses 4x4 kernel: pixels from (y-1,x-1) to (y+2,x+2)
        # Erode with 4x4 kernel anchored at (1,1) to check that neighborhood
        erosion_kernel = np.ones((4, 4), dtype=np.uint8)
        mask_eroded = cv2.erode(
            mask.astype(np.uint8), erosion_kernel, anchor=(1, 1)
        ).astype(bool)
    else:
        interp = cv2.INTER_LINEAR
        erosion_kernel = np.ones((2, 2), dtype=np.uint8)
        mask_eroded = cv2.erode(
            mask.astype(np.uint8), erosion_kernel, anchor=(0, 0)
        ).astype(bool)

    # Use OpenCV resize - much faster than scipy.ndimage.zoom
    # cv2.resize takes (width, height) not (height, width)
    x_up = cv2.resize(x, (new_w, new_h), interpolation=interp)
    y_up = cv2.resize(y, (new_w, new_h), interpolation=interp)
    z_up = cv2.resize(z, (new_w, new_h), interpolation=interp)

    # Upsample the eroded mask to determine output validity
    # Use nearest neighbor for mask to get clean boundaries
    mask_up = cv2.resize(
        mask_eroded.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST
    ).astype(bool)

    # Set invalid points to -1 in-place (matching C++ convention)
    invalid = ~mask_up
    x_up[invalid] = -1.0
    y_up[invalid] = -1.0
    z_up[invalid] = -1.0

    return x_up, y_up, z_up, mask_up


def interpolate_at_points(
    x: NDArray[np.float32],
    y: NDArray[np.float32],
    z: NDArray[np.float32],
    mask: NDArray[np.bool_],
    query_y: NDArray[np.floating],
    query_x: NDArray[np.floating],
    scale: Tuple[float, float],
    order: int = 1,
    method: Literal["linear", "bspline", "catmull_rom"] = "catmull_rom",
    invalid_value: float = -1.0,
) -> Tuple[
    NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_]
]:
    """Interpolate coordinates at arbitrary query points.

    Parameters
    ----------
    x, y, z : NDArray[np.float32]
        Source coordinate grids, shape (H, W).
    mask : NDArray[np.bool_]
        Source validity mask.
    query_y, query_x : NDArray
        Query points in nominal (voxel) coordinates.
    scale : Tuple[float, float]
        Grid scale (scale_y, scale_x).
    order : int
        Interpolation order for bspline method (2-5). Ignored for other methods.
    method : str
        Interpolation method:
        - "catmull_rom": Catmull-Rom spline (default). Fast and accurate.
        - "linear": Bilinear interpolation (fast, via OpenCV).
        - "bspline": B-spline interpolation (via scipy, order parameter controls degree).
    invalid_value : float
        Value to use for invalid points. Default -1.0.

    Returns
    -------
    Tuple[NDArray, NDArray, NDArray, NDArray]
        Interpolated (x, y, z, valid) at query points.
    """
    # Convert nominal coordinates to grid coordinates
    grid_y = np.asarray(query_y, dtype=np.float32) / scale[0]
    grid_x = np.asarray(query_x, dtype=np.float32) / scale[1]

    output_shape = grid_y.shape

    h, w = x.shape

    if method == "catmull_rom":
        # Catmull-Rom internally checks 4x4 kernel validity
        x_interp = _interpolate_catmull_rom(x, grid_y, grid_x, mask, invalid_value)
        y_interp = _interpolate_catmull_rom(y, grid_y, grid_x, mask, invalid_value)
        z_interp = _interpolate_catmull_rom(z, grid_y, grid_x, mask, invalid_value)
        # Valid where z was successfully interpolated (not invalid_value)
        valid = z_interp != invalid_value

    elif method == "bspline":
        # Use scipy's map_coordinates with spline interpolation.
        # We interpolate all points first, then check validity and mask invalid results.
        # This is simpler than conditional interpolation and the perf difference is
        # negligible since map_coordinates is already optimized and the masking is O(n).
        coords = np.array([grid_y.ravel(), grid_x.ravel()])
        spline_order = max(2, min(order, 5))  # Clamp to valid range

        x_interp = ndimage.map_coordinates(
            x, coords, order=spline_order, mode="constant", cval=invalid_value
        ).reshape(output_shape).astype(np.float32)
        y_interp = ndimage.map_coordinates(
            y, coords, order=spline_order, mode="constant", cval=invalid_value
        ).reshape(output_shape).astype(np.float32)
        z_interp = ndimage.map_coordinates(
            z, coords, order=spline_order, mode="constant", cval=invalid_value
        ).reshape(output_shape).astype(np.float32)

        # Check kernel validity explicitly
        valid_flat = _check_bspline_validity(
            mask, grid_y.ravel().astype(np.float64),
            grid_x.ravel().astype(np.float64), h, w, spline_order
        )
        valid = valid_flat.reshape(output_shape)

        # Set invalid points
        x_interp[~valid] = invalid_value
        y_interp[~valid] = invalid_value
        z_interp[~valid] = invalid_value

    else:
        # Use OpenCV for bilinear (fast path).
        # Like bspline, we interpolate all points then mask invalid results.
        # cv2.remap doesn't support conditional/masked interpolation anyway.
        map_x = grid_x.reshape(output_shape).astype(np.float32)
        map_y = grid_y.reshape(output_shape).astype(np.float32)

        x_interp = cv2.remap(x, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=invalid_value)
        y_interp = cv2.remap(y, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=invalid_value)
        z_interp = cv2.remap(z, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=invalid_value)

        # Check 2x2 kernel validity explicitly
        valid_flat = _check_bilinear_validity(
            mask, grid_y.ravel().astype(np.float64),
            grid_x.ravel().astype(np.float64), h, w
        )
        valid = valid_flat.reshape(output_shape)

        # Set invalid points
        x_interp[~valid] = invalid_value
        y_interp[~valid] = invalid_value
        z_interp[~valid] = invalid_value

    return x_interp, y_interp, z_interp, valid


def compute_grid_bounds(
    x: NDArray[np.float32],
    y: NDArray[np.float32],
    z: NDArray[np.float32],
    mask: NDArray[np.bool_],
) -> Tuple[float, float, float, float, float, float]:
    """Compute the bounding box of valid points.

    Parameters
    ----------
    x, y, z : NDArray[np.float32]
        Coordinate arrays.
    mask : NDArray[np.bool_]
        Validity mask.

    Returns
    -------
    Tuple[float, float, float, float, float, float]
        (x_min, y_min, z_min, x_max, y_max, z_max)
    """
    if not mask.any():
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    x_valid = x[mask]
    y_valid = y[mask]
    z_valid = z[mask]

    return (
        float(x_valid.min()),
        float(y_valid.min()),
        float(z_valid.min()),
        float(x_valid.max()),
        float(y_valid.max()),
        float(z_valid.max()),
    )
