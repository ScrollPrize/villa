"""CPU scalar crop interpolation for data workers."""

from __future__ import annotations

import numba
import numpy as np


@numba.njit(cache=True, fastmath=False, inline="always")
def trilinear_weight(fz, fy, fx, dz, dy, dx):
    return (fz if dz else 1-fz)*(fy if dy else 1-fy)*(fx if dx else 1-fx)


@numba.njit(cache=True, fastmath=False, inline="always")
def _scalar_point(raw, qz, qy, qx):
    """Trilinear CT/255 at block coordinates (zeros outside the block)."""
    z0 = int(np.floor(qz))
    y0 = int(np.floor(qy))
    x0 = int(np.floor(qx))
    fz = qz - z0
    fy = qy - y0
    fx = qx - x0
    S0, S1, S2 = raw.shape[1:]
    ctv = 0.0
    # Most samples lie strictly inside the tight source block. Unroll the
    # eight neighbours there; retain the exact summation order and weights.
    if 0 <= z0 < S0-1 and 0 <= y0 < S1-1 and 0 <= x0 < S2-1:
        for dz in numba.literal_unroll((0, 1)):
            for dy in numba.literal_unroll((0, 1)):
                for dx in numba.literal_unroll((0, 1)):
                    ctv += trilinear_weight(fz, fy, fx, dz, dy, dx)*raw[0,z0+dz,y0+dy,x0+dx]
        return ctv*(1.0/255.0)
    for dz in range(2):
        zz = z0 + dz
        if zz < 0 or zz >= S0:
            continue
        for dy in range(2):
            yy = y0 + dy
            if yy < 0 or yy >= S1:
                continue
            for dx in range(2):
                xx = x0 + dx
                if xx < 0 or xx >= S2:
                    continue
                w = trilinear_weight(fz, fy, fx, dz, dy, dx)
                ctv += w * raw[0, zz, yy, xx]
    return ctv * (1.0 / 255.0)


@numba.njit(cache=True, fastmath=False, nogil=True)
def _sample_scalar(raw, start, pos, frame, lc, vc, fc, out):
    """Crop point (fc[i], vc[j], lc[k]) at world pos + frame @ (lc[k], vc[j], fc[i]), summed in that order. Each
    frame*coordinate product depends on one axis, so it is formed once per axis value rather than per point (the
    same products and sums: identical values)."""
    A = np.empty((len(lc), 3))
    B = np.empty((len(vc), 3))
    C = np.empty((len(fc), 3))
    for r in range(3):
        for k in range(len(lc)):
            A[k, r] = frame[r, 0] * lc[k]
        for j in range(len(vc)):
            B[j, r] = frame[r, 1] * vc[j]
        for i in range(len(fc)):
            C[i, r] = frame[r, 2] * fc[i]
    p = 0
    for i in range(len(fc)):
        for j in range(len(vc)):
            for k in range(len(lc)):
                wx = pos[0] + A[k, 0] + B[j, 0] + C[i, 0]
                wy = pos[1] + A[k, 1] + B[j, 1] + C[i, 1]
                wz = pos[2] + A[k, 2] + B[j, 2] + C[i, 2]
                out[p] = _scalar_point(raw, wz - start[0], wy - start[1], wx - start[2])
                p += 1


def sample_scalar_crop(raw, start_zyx, pos, frame, crop, out):
    """Write the scalar channel of ``crop`` (a CropSpec, local grid ``geometry.crop_local_grid``) directly, with
    identical interpolation and no history."""
    if raw.shape[0] != 1:
        raise ValueError('The fused sampler reads one scalar channel')
    lc, vc, fc = _crop_axes(crop)
    _sample_scalar(raw, np.asarray(start_zyx, np.float64), np.asarray(pos, np.float64),
                   np.asarray(frame, np.float64), lc, vc, fc, out.reshape(-1))


_AXES_CACHE: dict = {}


def _crop_axes(crop):
    """The crop's lateral, vertical and forward coordinates (float64), as ``crop_local_grid`` places them."""
    axes = _AXES_CACHE.get(crop)
    if axes is None:
        axes = _AXES_CACHE[crop] = tuple(np.ascontiguousarray(a, np.float64) for a in
                                         (crop.lateral_coords, crop.vertical_coords, crop.forward_coords))
    return axes
