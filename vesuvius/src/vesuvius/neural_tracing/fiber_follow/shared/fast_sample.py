"""CPU scalar crop interpolation and observed-history distances for data workers."""

from __future__ import annotations

import numba
import numpy as np


@numba.njit(cache=True, fastmath=False, nogil=True)
def segment_history_distances(hist, mask, grid):
    """CPU float32 equivalent of geometry.render_history's segment distances.

    Keep each float32 operation and reduction order, including masked gaps and
    the implicit origin. The caller retains torch.exp for identical rounding.
    """
    out = np.full((len(hist), len(grid)), np.inf, np.float32)
    for row in range(len(hist)):
        for k in range(hist.shape[1]):
            if mask[row, k] <= 0:
                continue
            start = np.zeros(3, np.float32) if k == 0 else hist[row, k-1]
            if k and not mask[row, k-1] > 0:
                start = hist[row, k]
            vx = hist[row, k, 0]-start[0]
            vy = hist[row, k, 1]-start[1]
            vz = hist[row, k, 2]-start[2]
            vv = max((vx*vx+vy*vy)+vz*vz, np.float32(1e-12))
            for p in range(len(grid)):
                qx = grid[p, 0]-start[0]
                qy = grid[p, 1]-start[1]
                qz = grid[p, 2]-start[2]
                t = min(max(((qx*vx+qy*vy)+qz*vz)/vv, np.float32(0)), np.float32(1))
                dx, dy, dz = qx-t*vx, qy-t*vy, qz-t*vz
                candidate = (dx*dx+dy*dy)+dz*dz
                out[row, p] = np.minimum(out[row, p], candidate)
    return out


@numba.njit(cache=True, fastmath=False, inline="always")
def trilinear_weight(fz, fy, fx, dz, dy, dx):
    return (fz if dz else 1-fz)*(fy if dy else 1-fy)*(fx if dx else 1-fx)


@numba.njit(cache=True, fastmath=False, inline="always")
def _scalar_point(p, raw, start_zyx, pos, frame, grid):
    ga, gb, gc = grid[p, 0], grid[p, 1], grid[p, 2]
    wx = pos[0] + frame[0, 0] * ga + frame[0, 1] * gb + frame[0, 2] * gc
    wy = pos[1] + frame[1, 0] * ga + frame[1, 1] * gb + frame[1, 2] * gc
    wz = pos[2] + frame[2, 0] * ga + frame[2, 1] * gb + frame[2, 2] * gc
    qz = wz - start_zyx[0]
    qy = wy - start_zyx[1]
    qx = wx - start_zyx[2]
    # trilinear (zeros outside)
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
        wz_ = fz if dz else 1.0 - fz
        for dy in range(2):
            yy = y0 + dy
            if yy < 0 or yy >= S1:
                continue
            wy_ = fy if dy else 1.0 - fy
            for dx in range(2):
                xx = x0 + dx
                if xx < 0 or xx >= S2:
                    continue
                w = trilinear_weight(fz, fy, fx, dz, dy, dx)
                ctv += w * raw[0, zz, yy, xx]
    return ctv * (1.0 / 255.0)


@numba.njit(cache=True, fastmath=False, nogil=True)
def _sample_scalar(raw, start, pos, frame, grid, out):
    for p in range(len(grid)):
        out[p] = _scalar_point(p, raw, start, pos, frame, grid)


def sample_scalar_crop(raw, start_zyx, pos, frame, grid_flat, out):
    """Write the scalar channel directly, with identical interpolation and no history."""
    if raw.shape[0] != 1:
        raise ValueError('The fused sampler reads one scalar channel')
    _sample_scalar(raw, np.asarray(start_zyx, np.float64), np.asarray(pos, np.float64),
                   np.asarray(frame, np.float64), grid_flat, out.reshape(-1))
