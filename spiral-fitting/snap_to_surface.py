#!/usr/bin/env python3
"""Snap every winding of a finished spiral fit onto a surface prediction at once, without two windings landing on the
same sheet.

A spiral fit gives all windings of a scroll, but each winding sits a few voxels off the papyrus, so its renders cut
through the dark gaps between sheets. Snapping each winding to its nearest predicted sheet on its own is easy, but where
the fit's winding spacing does not match the scroll, two windings snap onto the same sheet and the sheet between them is
left empty. This tool assigns windings to sheets jointly, so they keep their radial order and each sheet takes at most
one winding.

Method, per z slice (every --dz voxels) on --rays rays from the umbilicus:
  - the fitted windings cross the ray at radii t_1 < ... < t_n; the surface prediction (value > --threshold) gives
    sheets at radii rho_1 < ... < rho_m (centre of each run);
  - an exact order-preserving one-to-one alignment (dynamic programming) minimises
        sum over matched pairs ((t_i - rho_j) / (sigma * p_i))^2 + C_w per unmatched winding + C_s per unused sheet,
    where p_i is the local winding spacing on the ray; a match needs |t_i - rho_j| <= dmax * p_i;
  - pass 1 gives a radial move per winding, slice and ray; per winding these moves are smoothed (NaN-aware median,
    then a normalised Gaussian that fills gaps); pass 2 aligns again around the smoothed moves with a tighter dmax.
Each node of a winding (grid densified --densify times) moves radially by its pass-2 match (snap_flag 0), else by the
smoothed field (flag 1); nodes with neither keep their place (flag 3). By default only flag-0 nodes are written
(the others become -1 holes); --keep-unmatched writes every node.

Inputs: a fit_spiral meshes folder (wNNN_<tag>/ tifxyz), a surface prediction zarr array, and the dataset umbilicus.json.
--surface-scale converts mesh voxels to surface-array voxels (0.5 when the array is one pyramid level coarser than the
mesh coordinates). Outputs: one tifxyz per winding with the same folder name, plus snap_flag.tif and snap_d.tif (radial
move in mesh voxels), and snap_report.json. CPU only; needs numba.

Example (PHerc Paris 4, the gen-19 fit, level-2 coordinates):
    snap_to_surface.py meshes/fitted_gen19 snapped/ \\
        --surface .../20260411134726-surface-20260413141734-surface-recto-2um-ps256-L0-th0.45.zarr --level 2 \\
        --umbilicus dataset/umbilicus.json --z-range 9984 11520
"""
from __future__ import annotations

import json
import os
import re
import time

import click
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from surface_orientation import METADATA_KEY as LAYOUT_METADATA_KEY
from tifxyz import load_tifxyz, save_tifxyz
from umbilicus import json_umbilicus_z_to_yx

try:
    from numba import njit, prange
    HAVE_NUMBA = True
except ImportError:  # spiral-fitting installs numba on Linux x86-64 only
    HAVE_NUMBA = False
    prange = range

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f


WINDING_DIR = re.compile(r'^w(\d+)(_.*)?$')


# ------------------------------------------------------------------------------------------------ mesh I/O


def load_windings(meshes_dir, spliced):
    """{winding: (folder name, grid (H, W, 3) as x, y, z with NaN for invalid, meta.json)}.

    spliced=True takes the `_spliced` folders (render_ink's input), otherwise the plain ones."""
    out = {}
    for name in sorted(os.listdir(meshes_dir)):
        m = WINDING_DIR.match(name)
        path = os.path.join(meshes_dir, name)
        if m is None or not os.path.exists(os.path.join(path, 'x.tif')):
            continue
        if ('spliced' in name) != spliced:
            continue
        w = int(m.group(1))
        if w in out:
            raise click.ClickException(f'two folders for winding {w}: {out[w][0]} and {name}')
        patch = load_tifxyz(path)
        P = patch.zyxs.numpy()[..., ::-1].astype(np.float32).copy()
        P[~patch.valid_vertex_mask.numpy()] = np.nan
        with open(os.path.join(path, 'meta.json')) as f:
            out[w] = (name, P, json.load(f))
    return out


def write_winding(out_dir, name, P, flag, move, meta, densify_f):
    """Write the snapped winding as a new tifxyz (spiral-fitting's save_tifxyz) plus snap_flag.tif and snap_d.tif."""
    step = 1.0 / meta['scale'][0] / densify_f
    area_vx2, area_cm2 = meta.get('area_vx2'), meta.get('area_cm2')
    voxel_um = (area_cm2 * 1e8 / area_vx2) ** 0.5 if area_vx2 and area_cm2 else 1.0
    zyxs = np.where(np.isfinite(P).all(-1, keepdims=True), P[..., ::-1], -1.0).astype(np.float32)
    save_tifxyz(zyxs, out_dir, name, step, voxel_um, f'snap_to_surface of {meta.get("source", "fit_spiral")}',
                layout_metadata=meta.get(LAYOUT_METADATA_KEY))
    path = os.path.join(out_dir, name)
    Image.fromarray(flag).save(os.path.join(path, 'snap_flag.tif'))
    Image.fromarray(move.astype(np.float32)).save(os.path.join(path, 'snap_d.tif'))


def densify(P, f):
    """Bilinear upsampling of a grid by f (corner-aligned); valid only where all four parent nodes are valid."""
    if f == 1:
        return P
    H, W, _ = P.shape
    v = np.isfinite(P).all(-1).astype(np.float32)
    Pz = np.nan_to_num(P)
    ys = np.arange((H - 1) * f + 1, dtype=np.float32) / f
    xs = np.arange((W - 1) * f + 1, dtype=np.float32) / f
    mx, my = np.meshgrid(xs, ys)
    out = np.stack([cv2.remap(Pz[..., c], mx, my, cv2.INTER_LINEAR) for c in range(3)], -1)
    vv = cv2.remap(v, mx, my, cv2.INTER_LINEAR)
    out[vv < 0.999] = np.nan
    return out


# ------------------------------------------------------------------------------------------------ per-ray kernels


@njit(cache=True)
def slice_poly(P, z0):
    """Polyline (W, 2) x, y of a winding grid at exact z0 (per column, between the two valid rows bracketing z0)."""
    H, W, _ = P.shape
    out = np.full((W, 2), np.nan, np.float32)
    for j in range(W):
        for i in range(H - 1):
            za = P[i, j, 2]
            zb = P[i + 1, j, 2]
            if np.isnan(za) or np.isnan(zb):
                continue
            if (za - z0) * (zb - z0) <= 0.0:
                t = 0.0 if zb == za else (z0 - za) / (zb - za)
                out[j, 0] = P[i, j, 0] * (1 - t) + P[i + 1, j, 0] * t
                out[j, 1] = P[i, j, 1] * (1 - t) + P[i + 1, j, 1] * t
                break
    return out


@njit(cache=True)
def ray_hits(poly, wid, cx, cy, nray, hit_r, hit_w, hit_n, maxseg):
    """Intersect a polyline with nray rays from (cx, cy); append (radius, winding) per ray."""
    two_pi = 2.0 * np.pi
    W = poly.shape[0]
    kmax = hit_r.shape[1]
    for j in range(W - 1):
        x0, y0, x1, y1 = poly[j, 0], poly[j, 1], poly[j + 1, 0], poly[j + 1, 1]
        if np.isnan(x0) or np.isnan(x1):
            continue
        if (x1 - x0) ** 2 + (y1 - y0) ** 2 > maxseg * maxseg:
            continue
        a0 = np.arctan2(y0 - cy, x0 - cx)
        a1 = np.arctan2(y1 - cy, x1 - cx)
        r0 = np.hypot(x0 - cx, y0 - cy)
        r1 = np.hypot(x1 - cx, y1 - cy)
        da = a1 - a0
        if da > np.pi:
            da -= two_pi
        elif da < -np.pi:
            da += two_pi
        if abs(da) < 1e-9:
            continue
        lo = min(a0, a0 + da)
        hi = max(a0, a0 + da)
        m0 = int(np.ceil(lo / two_pi * nray))
        m1 = int(np.floor(hi / two_pi * nray))
        for m in range(m0, m1 + 1):
            th = m * two_pi / nray
            t = (th - a0) / da
            if t < 0.0 or t > 1.0:
                continue
            mm = m % nray
            n = hit_n[mm]
            if n < kmax:
                hit_r[mm, n] = r0 * (1 - t) + r1 * t
                hit_w[mm, n] = wid
                hit_n[mm] = n + 1


@njit(cache=True)
def sheet_candidates(prof, step, thr, cand, cand_n):
    """Runs of prof > thr -> value-weighted centre radius of each run, per ray."""
    nray, T = prof.shape
    cmax = cand.shape[1]
    for m in range(nray):
        n = 0
        t = 0
        while t < T:
            if prof[m, t] > thr:
                acc = 0.0
                wsum = 0.0
                while t < T and prof[m, t] > thr:
                    acc += prof[m, t] * t
                    wsum += prof[m, t]
                    t += 1
                if n < cmax:
                    cand[m, n] = (acc / wsum) * step
                    n += 1
            else:
                t += 1
        cand_n[m] = n


@njit(cache=True)
def align_ray(t, rho, sigma_f, dmax_f, cw, cs, out_rho):
    """Order-preserving one-to-one alignment of target radii t (sorted) to sheet radii rho (sorted).
    out_rho[i] = matched sheet radius or NaN."""
    n = t.shape[0]
    m = rho.shape[0]
    for i in range(n):
        out_rho[i] = np.nan
    if n == 0 or m == 0:
        return
    p = np.empty(n)  # local winding spacing: mean of the two neighbour gaps (one at the ends)
    for i in range(n):
        g1 = t[i] - t[i - 1] if i > 0 else -1.0
        g2 = t[i + 1] - t[i] if i < n - 1 else -1.0
        if g1 > 0 and g2 > 0:
            p[i] = 0.5 * (g1 + g2)
        elif g1 > 0:
            p[i] = g1
        elif g2 > 0:
            p[i] = g2
        else:
            p[i] = 20.0
        if p[i] < 4.0:
            p[i] = 4.0
    INF = 1e30
    D = np.full((n + 1, m + 1), INF)
    B = np.zeros((n + 1, m + 1), np.int8)  # 1 match, 2 skip winding, 3 skip sheet
    D[0, 0] = 0.0
    for j in range(1, m + 1):
        D[0, j] = D[0, j - 1] + cs
        B[0, j] = 3
    for i in range(1, n + 1):
        D[i, 0] = D[i - 1, 0] + cw
        B[i, 0] = 2
        for j in range(1, m + 1):
            best = D[i - 1, j] + cw
            b = 2
            v = D[i, j - 1] + cs
            if v < best:
                best = v
                b = 3
            d = abs(t[i - 1] - rho[j - 1])
            if d <= dmax_f * p[i - 1]:
                v = D[i - 1, j - 1] + (d / (sigma_f * p[i - 1])) ** 2
                if v < best:
                    best = v
                    b = 1
            D[i, j] = best
            B[i, j] = b
    i, j = n, m
    while i > 0 or j > 0:
        b = B[i, j]
        if b == 1:
            out_rho[i - 1] = rho[j - 1]
            i -= 1
            j -= 1
        elif b == 2:
            i -= 1
        else:
            j -= 1


@njit(cache=True, parallel=True)
def solve_slice(hit_r, hit_w, hit_o, hit_n, cand, cand_n, sigma_f, dmax_f, cw, cs, res_w, res_d, res_n):
    """Per ray: targets = fitted radius + prior move, sorted; align; move relative to the fitted radius."""
    nray = hit_r.shape[0]
    for m in prange(nray):
        n = hit_n[m]
        res_n[m] = n
        if n == 0:
            continue
        tgt = hit_r[m, :n] + hit_o[m, :n]
        order = np.argsort(tgt)
        t = tgt[order]
        rho = cand[m, :cand_n[m]]
        out = np.empty(n)
        align_ray(t, rho, sigma_f, dmax_f, cw, cs, out)
        for i in range(n):
            k = order[i]
            res_w[m, i] = hit_w[m, k]
            res_d[m, i] = out[i] - hit_r[m, k]


@njit(cache=True, parallel=True)
def nan_median_filter(F, hk, hm, min_frac):
    """Median over the finite values of a (2hk+1) x (2hm+1) window (slice axis clamped, ray axis wrapped);
    NaN where fewer than min_frac of the window is finite. A median of matched moves picks one sheet, where a mean
    would land between two sheets."""
    K, M = F.shape
    out = np.full((K, M), np.nan, np.float32)
    need = int(min_frac * (2 * hk + 1) * (2 * hm + 1))
    for k in prange(K):
        buf = np.empty((2 * hk + 1) * (2 * hm + 1), np.float32)
        for m in range(M):
            n = 0
            for dk in range(-hk, hk + 1):
                kk = min(max(k + dk, 0), K - 1)
                for dm in range(-hm, hm + 1):
                    v = F[kk, (m + dm) % M]
                    if not np.isnan(v):
                        buf[n] = v
                        n += 1
            if n >= max(need, 1):
                s = np.sort(buf[:n])
                out[k, m] = s[n // 2]
    return out


def smooth_field(F, med, sig):
    """Move prior for one winding over (slices, rays): NaN-aware median of the pass-1 moves where the window has
    support, gaps filled by a normalised Gaussian of that median field (rays wrap around)."""
    if not np.isfinite(F).any():
        return F.copy()
    Mf = nan_median_filter(F.astype(np.float32), med[0] // 2, med[1] // 2, 0.3)
    vm = np.isfinite(Mf)
    sup = gaussian_filter(vm.astype(np.float32), sig, mode=('nearest', 'wrap'))
    num = gaussian_filter(np.where(vm, Mf, 0.0).astype(np.float32), sig, mode=('nearest', 'wrap'))
    fill = np.where(sup > 0.05, num / np.maximum(sup, 1e-6), np.nan)
    return np.where(vm, Mf, fill).astype(np.float32)


def feather_field(F, sig, s0):
    """Complete move field for one winding over (slices, rays): matched moves where the second pass matched, elsewhere
    the Gaussian-normalised mean of the matched moves scaled by min(support / s0, 1), so the fill fades to no move
    away from matched rays. Unlike the median-field fill, a winding that lost its sheet to a neighbour is not pushed
    the whole way onto that sheet."""
    vm = np.isfinite(F)
    if not vm.any():
        return np.zeros_like(F, dtype=np.float32)
    sup = gaussian_filter(vm.astype(np.float32), sig, mode=('nearest', 'wrap'))
    num = gaussian_filter(np.where(vm, F, 0.0).astype(np.float32), sig, mode=('nearest', 'wrap'))
    fill = num / np.maximum(sup, 1e-6) * np.clip(sup / s0, 0.0, 1.0)
    return np.where(vm, F, np.where(sup > 1e-6, fill, 0.0)).astype(np.float32)


def _grid_normals(P, valid):
    gi, gj = np.gradient(np.where(valid[..., None], P, 0.0), axis=0), np.gradient(np.where(valid[..., None], P, 0.0), axis=1)
    n = np.cross(gi, gj)
    ln = np.linalg.norm(n, axis=-1)
    ok = valid & np.isfinite(ln) & (ln > 0)
    return n / np.where(ok, ln, 1.0)[..., None], ok


def drop_stacked_fill(res, perp=3.13, lateral=25.0, k=64, reach=2, workers=8):
    """In complete (--feather) output, a filled node (flag 4) that sits on a sheet already occupied by a nearby winding
    (w +- 1 .. w +- reach has a node within `lateral` voxels whose normal plane is within `perp` voxels) becomes NaN.
    Matched nodes are never removed. Modifies res in place; returns the number of nodes removed."""
    from scipy.spatial import cKDTree
    wids = sorted(res)
    geo = {}
    for w in wids:
        P = res[w][0].astype(np.float64)
        v = np.isfinite(P).all(-1)
        n, ok = _grid_normals(P, v)
        geo[w] = (P, n, ok)
    removed = 0
    drops = {}
    for i, w in enumerate(wids):
        P, _, _ = geo[w]
        q = np.isfinite(P).all(-1) & (res[w][1] == 4)
        nb = [geo[wids[j]] for j in range(i - reach, i + reach + 1) if j != i and 0 <= j < len(wids)]
        drop = np.zeros(q.shape, bool)
        if q.any() and nb:
            X = np.concatenate([g[0][g[2]] for g in nb])
            N = np.concatenate([g[1][g[2]] for g in nb])
            rr, cc = np.nonzero(q)
            Q = P[rr, cc]
            dist, idx = cKDTree(X).query(Q, k=k, distance_upper_bound=lateral, workers=workers)
            j = np.minimum(idx, len(X) - 1)
            hit = (np.isfinite(dist) & (np.abs(((X[j] - Q[:, None, :]) * N[j]).sum(-1)) < perp)).any(1)
            drop[rr[hit], cc[hit]] = True
        drops[w] = drop
    for w in wids:  # decided on the unmodified output, applied together
        Pn, flag, move = res[w]
        Pn = Pn.copy()
        Pn[drops[w]] = np.nan
        removed += int(drops[w].sum())
        res[w] = (Pn, flag, move)
    return removed


# ------------------------------------------------------------------------------------------------ driver


def open_surface(surface, level):
    import zarr
    kw = {'storage_options': {'anon': True}} if surface.startswith('s3://') else {}
    z = zarr.open(surface, mode='r', **kw)
    if hasattr(z, 'shape'):
        return z
    return z[str(level)]


def snap(windings, S, umb_yx, z_range, *, dz=4, rays=2048, step=0.5, rmax=4000.0, threshold=127.0, densify_f=2,
         sigma=0.35, dmax=0.6, dmax2=0.45, cs=1.0, med=(3, 9), sig=(1.5, 6.0), bbox=None, feather=None, log=print):
    """windings: {w: (name, P (H, W, 3) x, y, z in surface-array voxels, meta)}. Returns {w: (P_densified_moved,
    flag, move)} plus run statistics. umb_yx(z) -> (y, x) in surface-array voxels."""
    t0 = time.time()
    wids = sorted(windings)
    idx = {w: i for i, w in enumerate(wids)}
    zs = np.arange(z_range[0], z_range[1], dz)
    K, M, NW = len(zs), rays, len(wids)
    T = int(rmax / step)
    tt = np.arange(T, dtype=np.float32) * step
    th = np.arange(M) * 2 * np.pi / M
    cos, sin = np.cos(th).astype(np.float32), np.sin(th).astype(np.float32)
    y0, y1, x0, x1 = bbox if bbox else (0, S.shape[1], 0, S.shape[2])
    czc = S.chunks[0]
    slab = {'c': None, 'a': None}
    centres = np.array([umb_yx(float(z)) for z in zs], np.float64)  # (K, 2) y, x
    cw1 = (dmax / sigma) ** 2  # leaving a winding unmatched costs as much as the largest allowed move
    cw2 = (dmax2 / sigma) ** 2

    def get_slice(z):
        c = int(z) // czc
        if slab['c'] != c:  # one chunk row of z at a time: each chunk is decompressed once per pass
            slab['a'] = None
            slab['a'] = np.asarray(S[c * czc:min((c + 1) * czc, S.shape[0]), y0:y1, x0:x1])
            slab['c'] = c
        return slab['a'][int(z) - c * czc].astype(np.float32)

    def run_pass(prior, dmax_f, cwx, tag):
        F = np.full((NW, K, M), np.nan, np.float32)
        seen = np.zeros((NW, K, M), bool)
        st = dict(rays=0, hits=0, matched=0, sheets=0, fold_hits=0)
        slab['c'] = None
        kmax = 2 * NW + 16
        for k, z in enumerate(zs):
            sl = get_slice(z)
            cy, cx = centres[k]
            mx = (cx - x0 + tt[None, :] * cos[:, None]).astype(np.float32)
            my = (cy - y0 + tt[None, :] * sin[:, None]).astype(np.float32)
            prof = cv2.remap(sl, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            cand = np.zeros((M, 1024), np.float64)
            cand_n = np.zeros(M, np.int64)
            sheet_candidates(prof, step, threshold, cand, cand_n)
            hit_r = np.zeros((M, kmax), np.float64)
            hit_w = np.zeros((M, kmax), np.int64)
            hit_n = np.zeros(M, np.int64)
            for w in wids:
                ray_hits(slice_poly(windings[w][1], float(z)), w, cx, cy, M, hit_r, hit_w, hit_n, 120.0)
            hit_o = np.zeros((M, kmax), np.float64)
            if prior is not None:
                wi = np.vectorize(lambda q: idx.get(int(q), 0))(hit_w)
                pv = prior[wi, k, np.repeat(np.arange(M)[:, None], kmax, 1)]
                hit_o = np.where(np.isfinite(pv), pv, 0.0).astype(np.float64)
            res_w = np.zeros((M, kmax), np.int64)
            res_d = np.full((M, kmax), np.nan)
            res_n = np.zeros(M, np.int64)
            solve_slice(hit_r, hit_w, hit_o, hit_n, cand, cand_n, sigma, dmax_f, cwx, cs, res_w, res_d, res_n)
            for m in range(M):
                n = res_n[m]
                for i in range(n):
                    wi = idx[int(res_w[m, i])]
                    if seen[wi, k, m]:  # a winding crossing one ray twice (a fold): keep neither
                        F[wi, k, m] = np.nan
                        st['fold_hits'] += 1
                        continue
                    seen[wi, k, m] = True
                    F[wi, k, m] = res_d[m, i]
                st['rays'] += 1
                st['hits'] += int(n)
                st['matched'] += int(np.isfinite(res_d[m, :n]).sum())
                st['sheets'] += int(cand_n[m])
            if k % 50 == 0:
                log(f'{tag}: slice {k + 1}/{K} (z {z}), {time.time() - t0:.0f} s')
        return F, st

    F1, st1 = run_pass(None, dmax, cw1, 'pass 1')
    Fs = np.stack([smooth_field(F1[i], med, sig) for i in range(NW)])
    F2, st2 = run_pass(Fs, dmax2, cw2, 'pass 2')
    if feather is None:
        Ffin = np.where(np.isfinite(F2), F2, Fs)
        Fsrc = np.where(np.isfinite(F2), 0, np.where(np.isfinite(Fs), 1, 3)).astype(np.uint8)
    else:
        Ffin = np.stack([feather_field(F2[i], sig, feather) for i in range(NW)])
        Fsrc = np.where(np.isfinite(F2), 0, 4).astype(np.uint8)

    out = {}
    for w in wids:
        _, P0, _ = windings[w]
        P = densify(P0, densify_f)
        wi = idx[w]
        H, W, _ = P.shape
        flag = np.full((H, W), 3, np.uint8)
        move = np.zeros((H, W), np.float32)
        ok = np.isfinite(P).all(-1)
        Z = P[..., 2]
        inz = ok & (Z >= zs[0] - dz / 2) & (Z <= zs[-1] + dz / 2)
        rr, cc = np.nonzero(inz)
        k = np.clip(np.rint((Z[rr, cc] - zs[0]) / dz).astype(int), 0, K - 1)
        cy, cx = centres[k, 0], centres[k, 1]
        dx, dy = P[rr, cc, 0] - cx, P[rr, cc, 1] - cy
        m = np.rint(np.mod(np.arctan2(dy, dx), 2 * np.pi) / (2 * np.pi) * M).astype(int) % M
        d = Ffin[wi, k, m]
        good = np.isfinite(d)
        flag[rr, cc] = np.where(good, Fsrc[wi, k, m], 3)
        r = np.hypot(dx, dy)
        scale = np.where(r > 1e-6, (r + np.nan_to_num(d)) / np.maximum(r, 1e-6), 1.0)
        Pn = P.copy()
        Pn[rr[good], cc[good], 0] = (cx + dx * scale)[good]
        Pn[rr[good], cc[good], 1] = (cy + dy * scale)[good]
        move[rr[good], cc[good]] = d[good]
        out[w] = (Pn, flag, move)
    return out, dict(pass1=st1, pass2=st2, seconds=round(time.time() - t0, 1))


@click.command(context_settings={'show_default': True})
@click.argument('meshes_dir', type=click.Path(exists=True, file_okay=False))
@click.argument('out_dir', type=click.Path(file_okay=False))
@click.option('--surface', required=True, help='Surface prediction zarr (local path or s3:// URL); a group or an array.')
@click.option('--level', type=int, default=0, help='Pyramid level to read when --surface is a group.')
@click.option('--surface-scale', type=float, default=1.0,
              help='Surface-array voxels per mesh voxel (0.5 if the array is one level coarser than the meshes).')
@click.option('--umbilicus', 'umbilicus_path', required=True, type=click.Path(exists=True, dir_okay=False),
              help='The dataset umbilicus.json.')
@click.option('--umbilicus-scale', type=float, default=1.0, help='Mesh voxels per umbilicus.json unit.')
@click.option('--threshold', type=float, default=127.0, help='Surface values above this are sheet.')
@click.option('--z-range', type=int, nargs=2, default=None,
              help='z range to snap, in mesh voxels (default: the meshes\' z extent).')
@click.option('--bbox', type=int, nargs=4, default=None,
              help='y0 y1 x0 x1 of the surface array to read, in surface-array voxels (default: the meshes\' extent '
                   'plus the umbilicus, with a margin). Must contain the umbilicus.')
@click.option('--spliced', is_flag=True, help='Snap the `_spliced` meshes (the ones render_ink.py renders).')
@click.option('--keep-unmatched', is_flag=True,
              help='Write every node (matched, moved by the smoothed field, or unmoved) instead of matched nodes only.')
@click.option('--feather', type=float, default=None,
              help='Write complete windings: nodes without a one-to-one match get the mean of nearby matched moves, '
                   'faded to no move where the matched support is below this value (flag 4).')
@click.option('--densify', 'densify_f', type=int, default=2, help='Grid densification before snapping.')
@click.option('--dz', type=int, default=4, help='Slice spacing, in surface-array voxels.')
@click.option('--rays', type=int, default=2048)
@click.option('--threads', type=int, default=8)
def main(meshes_dir, out_dir, surface, level, surface_scale, umbilicus_path, umbilicus_scale, threshold, z_range,
         bbox, spliced, keep_unmatched, feather, densify_f, dz, rays, threads):
    """Snap all windings in MESHES_DIR onto the surface prediction; write them to OUT_DIR."""
    if not HAVE_NUMBA:
        raise click.ClickException('snap_to_surface.py needs numba (installed with spiral-fitting on Linux x86-64; '
                                   'elsewhere run `uv pip install numba`).')
    import numba
    numba.set_num_threads(threads)
    S = open_surface(surface, level)
    windings = load_windings(meshes_dir, spliced)
    if not windings:
        raise click.ClickException(f'no {"_spliced " if spliced else ""}winding meshes in {meshes_dir}')
    s = surface_scale
    win_s = {w: (n, P * s, meta) for w, (n, P, meta) in windings.items()}
    umb = json_umbilicus_z_to_yx(umbilicus_path, coordinate_scale=umbilicus_scale * s)
    allp = np.concatenate([P[np.isfinite(P).all(-1)] for _, P, _ in win_s.values()])
    if z_range is None:
        zr = (int(np.ceil(allp[:, 2].min())), int(np.floor(allp[:, 2].max())) + 1)
    else:
        zr = (int(round(z_range[0] * s)), int(round(z_range[1] * s)))
    if bbox is None:
        uyx = np.array([umb(float(z)) for z in np.linspace(zr[0], zr[1], 16)])
        lo = np.minimum(allp[:, [1, 0]].min(0), uyx.min(0)) - 16
        hi = np.maximum(allp[:, [1, 0]].max(0), uyx.max(0)) + 16
        bbox = (max(0, int(lo[0])), min(S.shape[1], int(hi[0]) + 1), max(0, int(lo[1])), min(S.shape[2], int(hi[1]) + 1))
    gb = S.chunks[0] * (bbox[1] - bbox[0]) * (bbox[3] - bbox[2]) * np.dtype(S.dtype).itemsize / 1e9
    click.echo(f'{len(windings)} windings, z {zr[0]}-{zr[1]} (surface voxels), bbox y {bbox[0]}-{bbox[1]} '
               f'x {bbox[2]}-{bbox[3]}, {gb:.1f} GB per z chunk row held in memory')
    res, stats = snap(win_s, S, umb, zr, dz=dz, rays=rays, threshold=threshold, densify_f=densify_f, bbox=bbox,
                      feather=feather, log=click.echo)
    if feather is not None:
        stats['stacked_fill_removed'] = drop_stacked_fill(res, workers=threads)
        click.echo(f'{stats["stacked_fill_removed"]} filled nodes removed: they sat on a sheet a neighbouring winding occupies')
    os.makedirs(out_dir, exist_ok=True)
    report = dict(args=dict(meshes_dir=meshes_dir, surface=surface, level=level, surface_scale=s,
                            umbilicus=umbilicus_path, umbilicus_scale=umbilicus_scale, threshold=threshold,
                            z_range=list(zr), bbox=list(bbox), spliced=spliced, keep_unmatched=keep_unmatched,
                            densify=densify_f, dz=dz, rays=rays), **stats, windings={})
    for w, (name, _, meta) in windings.items():
        Pn, flag, move = res[w]
        Pn = Pn / s
        move = move / s
        if not keep_unmatched and feather is None:
            Pn[flag != 0] = np.nan
        write_winding(out_dir, name, Pn, flag, move, meta, densify_f)
        valid = flag != 3
        report['windings'][name] = dict(
            nodes=int(np.isfinite(densify(windings[w][1], densify_f)).all(-1).sum()),
            matched=int((flag == 0).sum()), smoothed=int((flag == 1).sum()),
            median_abs_move=float(np.median(np.abs(move[valid]))) if valid.any() else None)
    tot = {k: sum(v[k] for v in report['windings'].values()) for k in ('nodes', 'matched', 'smoothed')}
    report['total'] = tot
    with open(os.path.join(out_dir, 'snap_report.json'), 'w') as f:
        json.dump(report, f, indent=1)
    click.echo(f'done: {tot["matched"]} of {tot["nodes"]} nodes matched, {tot["smoothed"]} moved by the smoothed '
               f'field, {stats["seconds"]} s')


if __name__ == '__main__':
    main()
