"""Unsigned fiber orientations in each observation's own (u, v, forward) frame.

The stored Lasagna nx/ny bytes encode world xyz with nz >= 0. Byte zero is
reserved for missing/masked data (valid component endpoints encode as 1/255).
Decode and normalize each source voxel *before* interpolation. Sample the six
entries of R.T @ (n n.T) @ R, not signed vectors or encoded bytes. Since R is
constant within a crop, rotating source vectors before forming/interpolating
their second moments is equivalent to rotating the interpolated tensor.

Channels: uu, vv, ff, uv, uf, vf. Off-diagonals are signed; never clip to [0,1].
Trilinear mixtures remain positive semidefinite. Their trace is interpolated
validity (zero outside support); do not normalize away that information.
"""
import numba
import numpy as np
import torch
from functools import lru_cache

from .data import _grid_flat, tight_block
from .fast_sample import trilinear_weight
from .crop_sampling import unique_crop_indices

DIRECTION_CHANNELS = ('uu', 'vv', 'ff', 'uv', 'uf', 'vf')


def decode_direction_bytes(nx, ny):
    """Lasagna compact nx/ny format -> unit world xyz vectors; missing -> zero.

    Local implementation of the file-format equation, because the training
    environment has no importable Lasagna dependency. Byte conversion matches
    lasagna.omezarr_pyramid._decode_normals; normalization handles quantization
    just outside the unit disk. Encoding uses round(component * 127 + 128).
    """
    nx, ny = np.asarray(nx), np.asarray(ny)
    if nx.shape != ny.shape or nx.dtype != np.uint8 or ny.dtype != np.uint8:
        raise ValueError('nx/ny must be matching uint8 arrays')
    valid = (nx != 0) & (ny != 0)
    xy = (np.stack((nx, ny), axis=-1).astype(np.float32) - 128.) / 127.
    z = np.sqrt(np.maximum(0., 1. - xy[..., 0]**2 - xy[..., 1]**2))
    vectors = np.concatenate((xy, z[..., None]), axis=-1)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-12)
    vectors[~valid] = 0.
    return vectors


def local_direction_moments(vectors, frame):
    """World row vectors -> local unsigned moments, trailing dimension six."""
    frame = np.asarray(frame, dtype=np.float64)
    if frame.shape != (3, 3) or not np.isfinite(frame).all() or not np.allclose(
            frame.T @ frame, np.eye(3), rtol=0, atol=1e-5):
        raise ValueError('Direction crop frame must be orthonormal world-xyz columns')
    local = np.asarray(vectors, dtype=np.float32) @ frame.astype(np.float32)
    u, v, f = (local[..., j] for j in range(3))
    return np.stack((u*u, v*v, f*f, u*v, u*f, v*f), axis=-1)


@lru_cache(maxsize=1)
def _decoded_directions():
    """All 65536 byte pairs, decoded exactly once (768 KiB per process)."""
    nx, ny = np.indices((256, 256), dtype=np.uint8)
    values = decode_direction_bytes(nx, ny).reshape(-1, 3)
    values.setflags(write=False)
    return values


@numba.njit(cache=True, fastmath=False, nogil=True)
def _sample_moments(raw, table, start, pos, frame, grid, out):
    """Six-channel trilinear sampling without per-voxel arrays or BLAS calls."""
    for p in range(len(grid)):
        a, b, c = grid[p, 0], grid[p, 1], grid[p, 2]
        qx = pos[0] + (frame[0, 0]*a + frame[0, 1]*b + frame[0, 2]*c) - start[2]
        qy = pos[1] + (frame[1, 0]*a + frame[1, 1]*b + frame[1, 2]*c) - start[1]
        qz = pos[2] + (frame[2, 0]*a + frame[2, 1]*b + frame[2, 2]*c) - start[0]
        x0, y0, z0 = int(np.floor(qx)), int(np.floor(qy)), int(np.floor(qz))
        fx, fy, fz = qx-x0, qy-y0, qz-z0
        uu = vv = ff = uv = uf = vf = 0.0
        # tight_block includes every interpolation corner, even when the crop
        # crosses a volume boundary (the reader already zero-fills those cells).
        for dz in numba.literal_unroll((0, 1)):
            z = z0+dz
            for dy in numba.literal_unroll((0, 1)):
                y = y0+dy
                for dx in numba.literal_unroll((0, 1)):
                    x = x0+dx
                    weight = trilinear_weight(fz, fy, fx, dz, dy, dx)
                    code = raw[z, y, x]
                    uu += weight * table[code, 0]
                    vv += weight * table[code, 1]
                    ff += weight * table[code, 2]
                    uv += weight * table[code, 3]
                    uf += weight * table[code, 4]
                    vf += weight * table[code, 5]
        out[0, p], out[1, p], out[2, p] = uu, vv, ff
        out[3, p], out[4, p], out[5, p] = uv, uf, vf


def direction_crops(items, vol, crop, pool=None, *, out=None):
    """Use the level shared by presence/nx/ny, independent of the CT scale.

    Process one tight source block at a time per thread, keeping decoded fields
    out of long-lived caches. Main crops, history patches and seed patches all
    call this with their own position and frame. No augmentation or presence
    weighting/dropout is applied to directions.
    """
    nx, ny = vol.direction_fields()
    grid = _grid_flat(crop)
    shape = (6, crop.depth, crop.width, crop.width)
    result = np.empty((len(items), *shape), dtype=np.float32) if out is None else out
    if result.shape != (len(items), *shape) or result.dtype != np.float32:
        raise ValueError('Direction crop destination must have the expected shape and float32 dtype')
    if any(not result[j].flags.c_contiguous for j in range(len(items))):
        raise ValueError('Each direction crop destination must be contiguous')
    decoded = _decoded_directions()

    def sample(entry):
        j, item = entry
        pos, frame = np.asarray(item['pos'], np.float64), np.asarray(item['frame'], np.float64)
        if pos.shape != (3,) or not np.isfinite(pos).all():
            raise ValueError('Direction crop position must be finite xyz')
        start, size = tight_block(pos, frame, crop)
        # Exact lookup, not quantization: the source already consists of bytes.
        indices = nx.read(start, size).astype(np.uint16)*256 + ny.read(start, size)
        # Rotate the byte-pair dictionary, rather than expanding the entire
        # source block to vectors and six moments. Lookup preserves the exact
        # float32 rotation/products and the interpolation summation order.
        moments = local_direction_moments(decoded, frame)
        _sample_moments(indices, moments, start, pos, frame, grid, result[j].reshape(6, -1))

    sources = unique_crop_indices(items)
    unique = [(j, items[j]) for j, source in enumerate(sources) if j == source]
    if pool is None:
        for entry in unique:
            sample(entry)
    else:
        # Exhaust map so exceptions propagate and all output slices are ready.
        for _ in pool.map(sample, unique):
            pass
    for j, source in enumerate(sources):
        if j != source:
            result[j] = result[source]
    return torch.from_numpy(result)
