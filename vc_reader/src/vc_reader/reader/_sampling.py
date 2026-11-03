"""Local float32 CT geometry with precise dispersed points, shared by CPU/CUDA."""
import math
import operator
import numpy as np


def shape3(shape):
    try:
        shape = tuple(operator.index(n) for n in shape)
    except TypeError:
        raise ValueError("shape must have three nonnegative integers") from None
    if len(shape) != 3 or any(n < 0 or n > 2**24 for n in shape) or math.prod(shape) > np.iinfo(np.intp).max // 4:
        raise ValueError("shape must have three nonnegative dimensions <= 2**24")
    return shape


def origin3(origin):
    origin = tuple(operator.index(n) for n in origin)
    if len(origin) != 3 or any(n < 0 or n > 2**30 for n in origin):
        raise ValueError("coordinate_origin must have three integers in [0, 2**30]")
    return origin


def points32(points, coordinate_origin=None):
    pts = np.asarray(points)
    if pts.ndim == 0 or pts.shape[-1] != 3 or pts.dtype.kind not in "iuf":
        raise ValueError("points must be a numeric array of shape (..., 3)")
    flat = pts.reshape(-1, 3)
    if coordinate_origin is None:
        from .. import _reader

        flat = np.ascontiguousarray(flat, dtype=np.float32 if pts.dtype == np.float32 else np.float64)
        base = _reader._point_origin(flat)
        if base is None:
            # Dispersed global coordinates cannot share a float32 local frame
            # without losing voxel positions. Only coordinates need float64;
            # uint8 interpolation and its output remain float32.
            return np.ascontiguousarray(flat, dtype=np.float64), (0, 0, 0), pts.shape[:-1]
        with np.errstate(over="ignore", invalid="ignore"):
            local = np.ascontiguousarray(flat - np.asarray(base, dtype=flat.dtype), dtype=np.float32)
    else:
        base = origin3(coordinate_origin)
        with np.errstate(over="ignore"):
            local = np.ascontiguousarray(flat, dtype=np.float32)
    return local, base, pts.shape[:-1]


def affine32(shape, origin, axes, coordinate_origin=None):
    shape = shape3(shape)
    origin = np.asarray(origin)
    axes = np.asarray(axes, dtype=np.float32)
    if origin.shape != (3,) or origin.dtype.kind not in "iuf" or axes.shape != (3, 3) or not np.isfinite(origin).all() or not np.isfinite(axes).all():
        raise ValueError("origin and axes must be finite with shapes (3,) and (3, 3)")
    if coordinate_origin is None:
        base = tuple(max(0, min(2**30, math.floor(float(v)))) for v in origin)
        difference = origin - np.asarray(base, dtype=origin.dtype) if origin.dtype.kind == "f" else origin.astype(np.float64) - base
        local = np.asarray(difference, dtype=np.float32)
    else:
        base = origin3(coordinate_origin)
        local = np.asarray(origin, dtype=np.float32)
    if not np.isfinite(local).all():
        raise ValueError("origin overflows float32")
    return shape, base, local, axes
