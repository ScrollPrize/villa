"""NumPy forms of the local float32 CT geometry (``vcr::localPoints``, ``vcr::localOrigin``), shared by CPU/CUDA."""
import operator

import numpy as np

from .. import _reader


def shape3(shape):
    try:
        shape = [operator.index(n) for n in shape]
    except TypeError:
        raise ValueError("shape must have three nonnegative integers") from None
    return tuple(_reader._check_crop_shape(shape))


def origin3(origin):
    return tuple(_reader._check_coordinate_origin([operator.index(n) for n in origin]))


def _given_origin(coordinate_origin):
    return None if coordinate_origin is None else [operator.index(n) for n in coordinate_origin]


def points32(points, coordinate_origin=None):
    pts = np.asarray(points)
    if pts.ndim == 0 or pts.shape[-1] != 3 or pts.dtype.kind not in "iuf":
        raise ValueError("points must be a numeric array of shape (..., 3)")
    flat = np.ascontiguousarray(pts.reshape(-1, 3), dtype=np.float32 if pts.dtype == np.float32 else np.float64)
    local = _reader._local_points(flat, _given_origin(coordinate_origin))
    if local is None:
        # Dispersed global coordinates cannot share a float32 local frame without
        # losing voxel positions. Only coordinates need float64; uint8 interpolation
        # and its output remain float32.
        return np.ascontiguousarray(flat, dtype=np.float64), (0, 0, 0), pts.shape[:-1]
    return local[0], tuple(local[1]), pts.shape[:-1]


def affine32(shape, origin, axes, coordinate_origin=None):
    shape = shape3(shape)
    origin = np.asarray(origin)
    axes = np.asarray(axes, dtype=np.float32)
    if origin.shape != (3,) or origin.dtype.kind not in "iuf" or axes.shape != (3, 3) or not np.isfinite(axes).all():
        raise ValueError("origin and axes must be finite with shapes (3,) and (3, 3)")
    flat = np.ascontiguousarray(origin, dtype=np.float32 if origin.dtype == np.float32 else np.float64)
    base, local = _reader._local_origin(flat, _given_origin(coordinate_origin))
    return shape, tuple(base), np.asarray(local, dtype=np.float32), axes
