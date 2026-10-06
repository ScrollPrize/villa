"""Fused, CPU trilinear maximum projection over an already loaded CT block.

Matches villa's RawChunkSampler.sample_block boundary and scale conventions.
Coordinates, weights and sums use float64, each depth sample rounds to float32
before the maximum, and the maximum starts at zero as in _render_values.
Numba compilation is deliberately without fastmath.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(inline='always')
def _position(point, direction, offset, scale, shape, origin, blockshape,
              direction32, addition32):
    displacement = direction*offset
    if direction32:
        displacement = float(np.float32(np.float32(direction)*np.float32(offset)))
    value = point+displacement
    if addition32:
        value = float(np.float32(value))
    value = min(max(value/scale, 0.0), shape-1.0)-origin
    value = min(max(value, 0.0), blockshape-1.0)
    lower = int(np.floor(value))
    return lower, value-lower


@njit(parallel=True, cache=True)
def _sample_max(block, points, directions, offsets, scale, shape, origin, result,
                direction32, addition32):
    for pixel in prange(points.shape[0]):
        maximum = np.float32(0.0)
        for layer in range(offsets.size):
            # Input fields are XYZ; raw CT and metadata are ZYX. Preserve
            # NumPy's float32 weak-scalar arithmetic when callers supply it.
            z0, fz = _position(points[pixel,2],directions[pixel,2],offsets[layer],
                               scale[0],shape[0],origin[0],block.shape[0],direction32,addition32)
            y0, fy = _position(points[pixel,1],directions[pixel,1],offsets[layer],
                               scale[1],shape[1],origin[1],block.shape[1],direction32,addition32)
            x0, fx = _position(points[pixel,0],directions[pixel,0],offsets[layer],
                               scale[2],shape[2],origin[2],block.shape[2],direction32,addition32)
            total = 0.0
            for dz in range(2):
                iz = min(z0+dz, block.shape[0]-1)
                wz = fz if dz else 1.0-fz
                for dy in range(2):
                    iy = min(y0+dy, block.shape[1]-1)
                    wy = fy if dy else 1.0-fy
                    for dx in range(2):
                        ix = min(x0+dx, block.shape[2]-1)
                        wx = fx if dx else 1.0-fx
                        # ndimage multiplies the voxel by each axis weight.
                        contribution = float(block[iz, iy, ix])
                        contribution *= wz
                        contribution *= wy
                        contribution *= wx
                        total += contribution
            value32 = np.float32(total)
            maximum = max(maximum, value32)
        result[pixel] = maximum


def render_values(xyz, normals, valid, offsets, sampler, chunks):
    """Drop-in complete _render_values call for finite XYZ and normal fields.

    Uses the existing sampler's load_block, including its chunk cache and sparse
    fill handling. No remote transport, geometry, or normal convention changes.
    Geometry fields must be float32 or float64. Supports real integer/float CT
    blocks; nonfinite geometry is rejected. CT samples must be finite.
    """
    xyz = np.asarray(xyz)
    normals = np.asarray(normals)
    valid = np.asarray(valid, dtype=np.bool_)
    if xyz.shape != normals.shape or xyz.shape != valid.shape+(3,):
        raise ValueError('XYZ and normals must have shape valid.shape+(3,)')
    if any(a.dtype not in (np.dtype('float32'),np.dtype('float64')) for a in (xyz,normals)):
        raise ValueError('XYZ and normals must have float32 or float64 dtype')
    offsets = np.asarray(offsets, dtype=np.float64)
    if offsets.ndim != 1 or not np.isfinite(offsets).all():
        raise ValueError('Offsets must be a finite one-dimensional array')
    output = np.zeros(valid.shape, dtype=np.float32)
    selected = valid.ravel()
    if not selected.any():
        return output
    points = np.asarray(xyz.reshape(-1, 3)[selected], dtype=np.float64)
    directions = np.asarray(normals.reshape(-1, 3)[selected], dtype=np.float64)
    if not np.isfinite(points).all() or not np.isfinite(directions).all():
        raise ValueError('Valid XYZ and normals must be finite')
    scale = np.asarray(sampler.info.scale_zyx, dtype=np.float64)
    shape = np.asarray(sampler.info.shape, dtype=np.int64)
    if scale.shape != (3,) or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('CT scale must contain three finite positive values')
    if shape.shape != (3,) or (shape < 1).any():
        raise ValueError('CT shape must contain three positive dimensions')
    block, origin = sampler.load_block(chunks)
    if block.ndim != 3 or min(block.shape) < 1:
        raise ValueError('Loaded CT block must be nonempty and three-dimensional')
    result = np.empty(points.shape[0], dtype=np.float32)
    _sample_max(block, points, directions, offsets, scale, shape,
                np.asarray(origin, dtype=np.int64), result,
                normals.dtype == np.dtype('float32'),
                xyz.dtype == normals.dtype == np.dtype('float32'))
    output.ravel()[selected] = result
    return output
