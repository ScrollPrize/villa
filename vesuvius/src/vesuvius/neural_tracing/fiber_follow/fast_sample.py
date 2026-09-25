"""Fused CPU crop sampler (numba) for DataLoader workers.

Same output as ``geometry.sample_oriented_fast`` + ``geometry.render_history``
for one sample, without converting the whole block to float: presence/CT
trilinear (zero padding), fiber axis from the nearest voxel (border clamp),
decoded, rotated into the local frame, optionally gated by presence.
"""

from __future__ import annotations

import numba
import numpy as np


@numba.njit(cache=True, fastmath=False)
def _sample(raw, start_zyx, pos, frame, grid, gate, hist, hmask, out, sigma, segments):
    C = raw.shape[0]
    S0, S1, S2 = raw.shape[1], raw.shape[2], raw.shape[3]
    P = grid.shape[0]
    has_dir = C >= 3
    ct_ch = 3 if C == 4 else (0 if C == 1 else -1)
    lin_out_ct = 7 if C == 4 else 0
    H = hist.shape[0]
    hist_ch = out.shape[0] - 1
    # history lies behind/near the current point: crop points further than
    # 7.1 sigma ahead of every valid history point get exp(-25) ~ 1e-11 -> 0
    hmax_c = -1e30
    for k in range(H):
        if hmask[k] > 0 and hist[k, 2] > hmax_c:
            hmax_c = hist[k, 2]
    if segments and H > 0 and hmask[0] > 0:
        hmax_c = max(hmax_c, 0.0)
    # Outside this box even exp(-distance² / (2*sigma²)) rounds to
    # zero in float32: exp(-128) is below half its smallest subnormal.
    # Include the origin only when the first history segment connects to it.
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for k in range(H):
        if hmask[k] > 0:
            for axis in range(3):
                lo[axis] = min(lo[axis], hist[k, axis])
                hi[axis] = max(hi[axis], hist[k, axis])
    if segments and H > 0 and hmask[0] > 0:
        for axis in range(3):
            lo[axis] = min(lo[axis], 0.0)
            hi[axis] = max(hi[axis], 0.0)
    lo -= 16.0*sigma
    hi += 16.0*sigma
    for p in range(P):
        ga, gb, gc = grid[p, 0], grid[p, 1], grid[p, 2]
        wx = pos[0] + frame[0, 0] * ga + frame[0, 1] * gb + frame[0, 2] * gc
        wy = pos[1] + frame[1, 0] * ga + frame[1, 1] * gb + frame[1, 2] * gc
        wz = pos[2] + frame[2, 0] * ga + frame[2, 1] * gb + frame[2, 2] * gc
        qz = wz - start_zyx[0]
        qy = wy - start_zyx[1]
        qx = wx - start_zyx[2]
        # trilinear (zeros outside) for presence / CT
        z0 = int(np.floor(qz))
        y0 = int(np.floor(qy))
        x0 = int(np.floor(qx))
        fz = qz - z0
        fy = qy - y0
        fx = qx - x0
        pres = 0.0
        ctv = 0.0
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
                    w = wz_ * wy_ * (fx if dx else 1.0 - fx)
                    if has_dir:
                        pres += w * raw[0, zz, yy, xx]
                    if ct_ch >= 0:
                        ctv += w * raw[ct_ch, zz, yy, xx]
        if has_dir:
            pres *= 1.0 / 255.0
            out[0, p] = pres
            # nearest voxel (round half to even, clamped) for the axis
            iz = min(max(int(np.rint(qz)), 0), S0 - 1)
            iy = min(max(int(np.rint(qy)), 0), S1 - 1)
            ix = min(max(int(np.rint(qx)), 0), S2 - 1)
            nx = (raw[1, iz, iy, ix] - 128.0) * (1.0 / 127.0)
            ny = (raw[2, iz, iy, ix] - 128.0) * (1.0 / 127.0)
            nz2 = 1.0 - nx * nx - ny * ny
            nz = np.sqrt(nz2) if nz2 > 0.0 else 0.0
            nn = nx * nx + ny * ny + nz * nz
            inv = 1.0 / np.sqrt(nn if nn > 1e-12 else 1e-12)
            nx *= inv
            ny *= inv
            nz *= inv
            a = nx * frame[0, 0] + ny * frame[1, 0] + nz * frame[2, 0]
            b = nx * frame[0, 1] + ny * frame[1, 1] + nz * frame[2, 1]
            c = nx * frame[0, 2] + ny * frame[1, 2] + nz * frame[2, 2]
            g = pres if gate else 1.0
            out[1, p] = a * a * g
            out[2, p] = b * b * g
            out[3, p] = c * c * g
            out[4, p] = a * b * g
            out[5, p] = a * c * g
            out[6, p] = b * c * g
            if ct_ch >= 0:
                out[lin_out_ct, p] = ctv * (1.0 / 255.0)
        else:
            out[0, p] = ctv * (1.0 / 255.0)
        # Own history: nearest point or connected segment, including the current origin.
        best = 1e30
        if (gc - hmax_c > 7.1*sigma or ga < lo[0] or ga > hi[0]
                or gb < lo[1] or gb > hi[1] or gc < lo[2]):
            out[hist_ch, p] = 0.0
            continue
        for k in range(H):
            if hmask[k] > 0:
                d0 = ga - hist[k, 0]
                d1 = gb - hist[k, 1]
                d2 = gc - hist[k, 2]
                if segments and (k == 0 or hmask[k-1] > 0):
                    # Project from this endpoint toward the preceding point (origin for k=0).
                    v0 = (hist[k-1, 0] if k else 0.0) - hist[k, 0]
                    v1 = (hist[k-1, 1] if k else 0.0) - hist[k, 1]
                    v2 = (hist[k-1, 2] if k else 0.0) - hist[k, 2]
                    vv = v0*v0 + v1*v1 + v2*v2
                    t = min(max((d0*v0 + d1*v1 + d2*v2)/max(vv, 1e-12), 0.0), 1.0)
                    d0 -= t*v0
                    d1 -= t*v1
                    d2 -= t*v2
                dd = d0 * d0 + d1 * d1 + d2 * d2
                if dd < best:
                    best = dd
        out[hist_ch, p] = np.exp(-best / (2.0*sigma*sigma)) if best < 1e29 else 0.0


def sample_crop(raw, start_zyx, pos, frame, grid_flat, gate, hist, hmask, n_out, history_sigma=1.0, history_render='points'):
    """raw (C, S, S, S) uint8; grid_flat (P, 3) float64 local (a, b, c);
    hist (N, 3) local, hmask (N,). Returns (n_out, P) float32."""
    if history_render not in ('points', 'segments') or not np.isfinite(history_sigma) or history_sigma <= 0:
        raise ValueError('Invalid history rendering mode or sigma')
    out = np.empty((n_out, grid_flat.shape[0]), np.float32)
    _sample(raw, np.asarray(start_zyx, np.float64), np.asarray(pos, np.float64), np.asarray(frame, np.float64),
            grid_flat, bool(gate), np.asarray(hist, np.float64), np.asarray(hmask, np.float64), out, float(history_sigma), history_render == 'segments')
    return out
