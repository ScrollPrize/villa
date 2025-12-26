"""Upsampling and interpolation utilities for tifxyz surfaces."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
from numpy.typing import NDArray


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

    # Map order to OpenCV interpolation flag
    if order == 0:
        interp = cv2.INTER_NEAREST
    elif order == 1:
        interp = cv2.INTER_LINEAR
    elif order == 3:
        interp = cv2.INTER_CUBIC
    else:
        interp = cv2.INTER_LINEAR

    # Use OpenCV resize - much faster than scipy.ndimage.zoom
    # cv2.resize takes (width, height) not (height, width)
    x_up = cv2.resize(x, (new_w, new_h), interpolation=interp)
    y_up = cv2.resize(y, (new_w, new_h), interpolation=interp)
    z_up = cv2.resize(z, (new_w, new_h), interpolation=interp)

    # Derive mask from z > 0 (matching C++ behavior - no separate mask upsampling)
    mask_up = z_up > 0

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
    invalid_value: float = -1.0,
) -> Tuple[
    NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_]
]:
    """Interpolate coordinates at arbitrary query points using OpenCV.

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
        Interpolation order (0=nearest, 1=bilinear, 3=bicubic).
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

    # Record original shape for reshaping output
    output_shape = grid_y.shape

    # Map order to OpenCV interpolation flag
    if order == 0:
        interp = cv2.INTER_NEAREST
    elif order == 1:
        interp = cv2.INTER_LINEAR
    elif order == 3:
        interp = cv2.INTER_CUBIC
    else:
        interp = cv2.INTER_LINEAR

    # cv2.remap needs 2D map arrays - reshape if needed
    map_x = grid_x.reshape(output_shape).astype(np.float32)
    map_y = grid_y.reshape(output_shape).astype(np.float32)

    # Use cv2.remap for interpolation (much faster than scipy for 2D grids)
    x_interp = cv2.remap(x, map_x, map_y, interp, borderMode=cv2.BORDER_CONSTANT, borderValue=invalid_value)
    y_interp = cv2.remap(y, map_x, map_y, interp, borderMode=cv2.BORDER_CONSTANT, borderValue=invalid_value)
    z_interp = cv2.remap(z, map_x, map_y, interp, borderMode=cv2.BORDER_CONSTANT, borderValue=invalid_value)

    # Interpolate validity mask to detect partial interpolation across boundaries.
    # When bilinear interpolation uses a mix of valid and invalid source pixels,
    # the coordinates get corrupted (blended toward -1). We detect this by
    # interpolating the mask: if the result is < 1.0, some invalid pixels were used.
    mask_float = mask.astype(np.float32)
    mask_interp = cv2.remap(mask_float, map_x, map_y, interp,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

    # A point is only valid if:
    # 1. z > 0 (not a sentinel value)
    # 2. mask interpolation is ~1.0 (all source pixels in interpolation were valid)
    valid = (z_interp > 0) & (mask_interp > 0.99)

    # Set invalid points to invalid_value
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
