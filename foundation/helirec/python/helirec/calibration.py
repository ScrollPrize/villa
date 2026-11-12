"""Calibration of an extended-field-of-view helical scan pair from the raw Bliss data.

Reproduces the reference (nabu-helical / nabuxx) calibration:
  * darks / flats reduction (nx2etf), weight map + horizontal apodisation (prepare_weights_double),
  * rotation axis of the main (HA) scan (nabuxx-calibrate extract_cor),
  * HA -> TA overlap vector (nabuxx-calibrate get_overlap),
and, beyond the reference, measures the helical pitch (rows of travel per recorded row: measure_pitch), and derives the placement of both scans in the merged frame and the rotation axis there.
Heavy kernels are C++ (helirec._core); the bookkeeping is numpy.
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import time
import types
from dataclasses import asdict, dataclass, field

import numpy as np

from . import _core
from .bliss import BlissScan, FrameRefs

NOMINAL_CURRENT_A = 0.2
FLT_MIN = np.float32(np.finfo(np.float32).tiny)


# ============================================================================ raw access
class FrameSource:
    """Fast row-window access to one FrameRefs sequence (C++ pread reader)."""

    def __init__(self, refs: FrameRefs, shape):
        self.refs = refs
        self.rows, self.cols = shape
        self.reader = _core.RawReader(list(refs.files), refs.dataset, self.rows, self.cols)

    def frames(self, idx) -> np.ndarray:
        idx = np.asarray(idx)
        return self.reader.read_frames(self.refs.file_index[idx], self.refs.frame_index[idx])

    def rows_window(self, idx, row0, nrows) -> np.ndarray:
        idx = np.asarray(idx)
        return self.reader.read_rows(self.refs.file_index[idx], self.refs.frame_index[idx],
                                     np.ascontiguousarray(row0, np.int32), int(nrows))


def _mean_frames(src: FrameSource, transform=None, batch=40, progress=None) -> np.ndarray:
    """numpy.mean(stack.astype(float32), axis=0) semantics: sequential float32 sum, then / n.
    progress(k): called as k more frames are done."""
    n = len(src.refs)
    acc = None
    for s in range(0, n, batch):
        e = min(n, s + batch)
        fr = src.frames(np.arange(s, e)).astype(np.float32)
        for k in range(e - s):
            f = fr[k] if transform is None else transform(fr[k], s + k)
            if acc is None:
                acc = np.array(f, np.float32, copy=True)
            else:
                acc += f
        if progress:
            progress(e - s)
    return acc / n


def _bar(show, total, desc, unit):
    """a helirec.ProgressBar, or a stand-in that shows nothing"""
    if show:
        return _core.ProgressBar(int(total), desc, unit)
    return contextlib.nullcontext(types.SimpleNamespace(update=lambda n=1: None))


def reduce_darks_flats(scan: BlissScan, progress=None):
    """dark (rows, cols) and flats (n_series, rows, cols) exactly as nx2etf stores them.
    progress(k): called as k more of the len(scan.darks) + sum(len(series)) frames are done."""
    dark = _mean_frames(FrameSource(scan.darks, scan.frame_shape), progress=progress)
    flats = []
    for refs, ctl in zip(scan.flats, scan.flats_control_A):
        scale = np.float32(NOMINAL_CURRENT_A) / np.asarray(ctl, np.float32)

        def tr(F, i, scale=scale):
            return dark + (F - dark) * scale[i]
        flats.append(_mean_frames(FrameSource(refs, scan.frame_shape), tr, progress=progress))
    return dark.astype(np.float32), np.stack(flats).astype(np.float32)


# ============================================================================ weights
def weight_map(flat_stacks, transition_width_vertical=15.0, sigma=10.0, crop_top=0, crop_bottom=0) -> np.ndarray:
    """prepare_weights_double: gaussian_filter(mean(all flats) / max, sigma) * vertical erf border."""
    prof = 0
    n = 0
    for st in flat_stacks:
        for i in range(st.shape[0]):
            prof = prof + st[i]
        n += st.shape[0]
    prof = prof / n
    prof = (prof / prof.max()).astype(np.float32)
    g = _core.gaussian_filter1d(np.ascontiguousarray(prof), 0, sigma, "reflect")
    g = _core.gaussian_filter1d(g, 1, sigma, "reflect")
    border = _core.weight_border_v(prof.shape[0], transition_width_vertical, crop_top, crop_bottom)
    return (g * border[:, None]).astype(np.float32)


def apodisation_widths(n_cols, est_cor_main_offset, companion_dx, transition_width_horizontal=1000.0,
                       transition_width_borders=50.0):
    """(w_left, w_right) for (main, companion) as prepare_weights_double --is_tetra writes them.
    Main scan: 'pente' beyond the rotation axis (min(2 tw, distance axis->edge)) on the axis side and a
    border taper towards the companion; companion: border taper on the side facing the main scan."""
    r = (n_cols - 1) / 2 + est_cor_main_offset
    if r < (n_cols - 1) / 2:
        raise NotImplementedError("rotation axis in the left half of the main scan is not supported yet")
    pente = min(2 * transition_width_horizontal, n_cols - r)
    if companion_dx < 0:   # companion on the left of the main scan
        return (transition_width_borders, pente), (0.0, transition_width_borders)
    raise NotImplementedError("companion on the right of the main scan is not supported yet")


# ============================================================================ geometry helpers
class Span:
    """Span info of one standalone scan (nabuxx SpanInfo) + chunk helpers."""

    def __init__(self, scan: BlissScan, tol_steps: float, z_readjustment: float = 0.0):
        self.scan = scan
        self.angles = np.ascontiguousarray(scan.unwrapped_angles_deg, np.float64)
        d = _core.span_info(self.angles, np.ascontiguousarray(scan.z_translation_m, np.float64), scan.pixel_size_m,
                            scan.frame_shape[0], z_readjustment, tol_steps)
        self.ishift = d["integer_shift_v"]
        self.fract = d["fract_complement"]
        self.z_pix = d["z_pix_per_proj"]
        self.z_pix_int = self.z_pix.astype(np.int64)  # the nabuxx binding casts z_pix_per_proj to int
        self.h_first = d["h_first"]
        self.ss, self.se = d["sunshine_start"], d["sunshine_end"]
        self.doable = d["doable"]                     # (first, last) valid heights, inclusive

    def chunk(self, h0, h1):
        """projection span [p0, p1) of heights h0..h1 and the chunk angles in degrees (via radians, as nabuxx)."""
        i0, i1 = h0 - self.h_first, h1 - self.h_first
        p0, p1 = int(self.ss[i0:i1 + 1].min()), int(self.se[i0:i1 + 1].max())
        ang = np.rad2deg(self.angles[p0:p1] * (2 * math.pi / 360.0))
        return p0, p1, ang


def choose_slices(start, end, m):
    N = end - start
    if N <= 0:
        return np.array([], int)
    step = N / m
    idx = np.unique(np.linspace(step / 2, N - 1 - step / 2, m, dtype=int))
    return np.arange(start, end)[idx.tolist()]


def _bracket(avail, angle):
    pos1 = int(np.clip(np.searchsorted(avail, angle, side="right"), 1, len(avail) - 1))
    pos0 = pos1 - 1
    f = (angle - avail[pos0]) / (avail[pos1] - avail[pos0])
    return pos0, pos1, f


class Extractor:
    """get_interpolated_frames_and_weights: flat-fielded intensity + plain weight map, bilinear in z and angle."""

    def __init__(self, scan: BlissScan, dark, flats, wmap):
        self.scan, self.dark, self.wmap = scan, dark, wmap
        self.F = (flats - dark[None]).astype(np.float32)          # control 0.2 -> divisor 1
        self.fa = np.asarray(scan.flats_unwrapped_angles_deg, np.float32)
        self.src = FrameSource(scan.projections, scan.frame_shape)
        self.cd = (np.asarray(scan.control_A, np.float64) / NOMINAL_CURRENT_A).astype(np.float32)

    def _flat(self, p):
        ang = np.float32(self.scan.unwrapped_angles_deg[p])
        i1 = int(np.searchsorted(self.fa, ang, side="left"))
        if i1 == 0:
            return self.F[0]
        if i1 == len(self.fa):
            return self.F[-1]
        f = (ang - self.fa[i1 - 1]) / (self.fa[i1] - self.fa[i1 - 1])
        return (np.float32(1) - f) * self.F[i1 - 1] + f * self.F[i1]

    def extract(self, p0, p1, f, zs0, zs1, zf0, zf1, S, progress=None):
        """progress(1): called after each of the len(p0) requests"""
        n = len(p0)
        rows, cols = self.scan.frame_shape
        out_s = np.zeros((n, S, cols), np.float32)
        out_w = np.zeros((n, S, cols), np.float32)
        side = []
        for p, zs in ((p0, zs0), (p1, zs1)):
            raw = self.src.rows_window(p, zs, S + 1).astype(np.float32)
            side.append(raw)
        for i in range(n):
            vals, wts, lows, fr = [], [], [], []
            ok_rows = True
            for k, (p, zs, zf) in enumerate(((p0[i], zs0[i], zf0[i]), (p1[i], zs1[i], zf1[i]))):
                a, b = max(0, zs), min(rows, zs + S + 1)
                void = a >= b
                zoff = a - zs if not void else 0
                raw = side[k][i]
                if not void:
                    lo, hi = a - zs, b - zs
                    sig = np.zeros_like(raw)
                    sig[lo:hi] = ((raw[lo:hi] - self.dark[a:b]) / self.cd[p]) / self._flat(p)[a:b]
                    w = np.zeros_like(raw)
                    w[lo:hi] = self.wmap[a:b]
                    vals.append((sig[zoff:hi], w[zoff:hi], b - a))
                else:
                    vals.append((None, None, 0))
                lows.append(zoff)
                fr.append(np.float32(zf))
            fi = np.float32(f[i])
            for j in range(S):
                zl = [j - lows[0], j - lows[1]]
                if zl[0] < 0 or zl[1] < 0 or zl[0] >= rows - 1 or zl[1] >= rows - 1:
                    out_w[i, j] = FLT_MIN
                    continue
                v = [np.zeros(cols, np.float32), np.zeros(cols, np.float32)]
                w = [np.zeros(cols, np.float32), np.zeros(cols, np.float32)]
                for k in range(2):
                    sig, wt, nr = vals[k]
                    if sig is None or zl[k] + 1 >= nr:
                        continue
                    frac = np.float32(np.float32(j + fr[k] - lows[k]) - np.float32(zl[k]))
                    v[k] = (np.float32(1) - frac) * sig[zl[k]] + frac * sig[zl[k] + 1]
                    w[k] = (np.float32(1) - frac) * wt[zl[k]] + frac * wt[zl[k] + 1]
                out_s[i, j] = (np.float32(1) - fi) * v[0] + fi * v[1]
                out_w[i, j] = (np.float32(1) - fi) * w[0] + fi * w[1]
            if progress:
                progress(1)
        return out_s, out_w


# ============================================================================ COR
def savgol_detrend(y, window=60, polyorder=3):
    """y - scipy.signal.savgol_filter(y, window, polyorder, axis=-1, mode='interp'), float32 like scipy.

    savgol_coeffs: fit position pos = window//2 (odd) or window//2 - 0.5 (even); convolve1d(mode='constant')
    (correlation with the reversed kernel, origin -1 for even sizes); edges ('interp'): cubic polyfit of
    the first / last `window` samples evaluated on the first / last window//2 positions."""
    y = np.asarray(y)
    if y.dtype not in (np.float32, np.float64):
        y = y.astype(np.float64)
    n = y.shape[-1]
    half, rem = divmod(window, 2)
    pos = half if rem else half - 0.5
    x = np.arange(-pos, window - pos, dtype=float)[::-1]           # use='conv'
    A = x[None, :] ** np.arange(polyorder + 1)[:, None]
    e0 = np.zeros(polyorder + 1); e0[0] = 1.0
    coeffs = np.linalg.lstsq(A, e0, rcond=None)[0]
    w = coeffs[::-1]                                                 # convolve1d -> correlate1d(reversed)
    center = window // 2 + (-1 if window % 2 == 0 else 0)            # size//2 + origin
    yy = y.reshape(-1, n).astype(np.float64)
    out = np.zeros_like(yy)
    for j in range(window):
        off = j - center
        lo, hi = max(0, -off), min(n, n - off)
        out[:, lo:hi] += w[j] * yy[:, lo + off:hi + off]
    out = out.astype(y.dtype)
    t = np.arange(window, dtype=float)
    for ws, we, i0, i1 in ((0, window, 0, half), (n - window, n, n - half, n)):
        P = np.polyfit(t, yy[:, ws:we].T, polyorder)                 # (polyorder+1, rows)
        idx = np.arange(i0 - ws, i1 - ws, dtype=float)
        vals = sum(P[k][:, None] * idx[None, :] ** (polyorder - k) for k in range(polyorder + 1))
        out[:, i0:i1] = vals.astype(y.dtype)
    return (y - out.reshape(y.shape)).astype(y.dtype)


def line_error_map(merits, z):
    za, zb = float(np.nanmin(z)), float(np.nanmax(z))
    if abs(zb - za) < 1.0:
        za, zb = za - 0.5, zb + 0.5
    n, S = merits.shape
    t = z - za
    E = np.empty((S, S), np.float64)
    rows = np.arange(n)
    b = np.arange(S, dtype=np.float64)
    for a in range(S):
        idx = ((b - a) / (zb - za))[:, None] * t[None, :] + a
        idx[:, z == zb] = b[:, None]
        idx = np.rint(idx).astype(int)
        E[a] = merits[rows[None, :], idx].astype(np.float32).mean(axis=1, dtype=np.float32)
    return E, za, zb


def find_cor(ext: Extractor, span: Span, near_pos, near_width=40, n_slices=20, step_deg=5.0, ovs=4,
             sigma_wide=10.0, sigma_narrow=0.4, log=print, progress=False):
    """Rotation axis of a half-acquisition scan (detector frame of that scan)."""
    scan = ext.scan
    W = scan.frame_shape[1]
    d0, d1 = span.doable
    slices = choose_slices(d0, d1, min(n_slices, d1 - d0))
    req = {k: [] for k in ("p0", "p1", "f", "zs0", "zs1", "zf0", "zf1")}
    req180 = {k: [] for k in req}
    zl = []
    for h in slices:
        p0c, p1c, avail = span.chunk(h, h)
        th = avail.min() + np.arange(0, 179.999, step_deg)
        th180 = th + 180.0
        if th180[-1] > avail.max() + 1e-7:
            continue
        zz = []
        for angles, r in ((th, req), (th180, req180)):
            zp = []
            for a in angles:
                q0, q1, f = _bracket(avail, a)
                P0, P1 = q0 + p0c, q1 + p0c
                r["p0"].append(P0); r["p1"].append(P1); r["f"].append(f)
                r["zs0"].append(h - span.ishift[P0]); r["zs1"].append(h - span.ishift[P1])
                r["zf0"].append(span.fract[P0]); r["zf1"].append(span.fract[P1])
                zp.append(span.z_pix_int[P0] * (1 - f) + span.z_pix_int[P1] * f)
            zz.append(np.array(zp))
        zl.extend(((zz[0] + zz[1]) * 0.5).tolist())
    zl = np.asarray(zl)
    sig, wgt = [], []
    with _bar(progress, len(req["p0"]) + len(req180["p0"]), "rotation axis", "line") as bar:
        for r in (req, req180):
            s, w = ext.extract(*[np.asarray(r[k]) for k in ("p0", "p1", "f", "zs0", "zs1", "zf0", "zf1")], S=1,
                               progress=bar.update)
            sig.append(s[:, 0]); wgt.append(w[:, 0])
    sig = np.ascontiguousarray(np.stack(sig), np.float32)
    wgt = np.ascontiguousarray(np.stack(wgt), np.float32)
    ovl_min = int(round(max(4, W - 2 * (near_pos + near_width))))
    ovl_max = int(round(min(2 * W - 6, W - 2 * (near_pos - near_width))))
    merits_raw, energies = _core.build_all_merits(sig, wgt, ovl_min, ovl_max, ovs, sigma_wide, sigma_narrow)
    wl = min(60, merits_raw.shape[-1])
    merits = savgol_detrend(merits_raw, wl, 3) if wl > 20 else merits_raw
    E, za, zb = line_error_map(merits, zl)
    ia, ib = np.unravel_index(np.argmin(E), E.shape)
    cor_of = lambda i: (W - 1) / 2 + (W - (ovl_min + i / ovs)) / 2
    ca, cb = cor_of(ia), cor_of(ib)
    cor0 = ca if za == zb else ca + (cb - ca) * (0 - za) / (zb - za)
    log(f"  COR search: {len(zl)} line pairs, overlap [{ovl_min}, {ovl_max}], best index ({ia}, {ib}) -> "
        f"COR {ca:.3f} .. {cb:.3f} (detector frame)")
    return dict(cor_begin_end=(ca, cb), cor0=cor0, z_begin_end=(za, zb), best_index=(int(ia), int(ib)),
                slices=slices, merits_raw=merits_raw, merits=merits, E=E, lines_signal=sig, lines_weight=wgt)


# ============================================================================ overlap
def _filter_overlap(sig, w, low_pass=0.4, high_pass=10.0):
    g = lambda a, axis, s: _core.gaussian_filter1d(np.ascontiguousarray(a, np.float64), axis, s, "reflect")
    out = []
    for R0, Wt in zip(sig, w):
        R0 = R0.astype(np.float64); Wt = Wt.astype(np.float64)
        Wl = g(Wt, 2, low_pass)
        with np.errstate(divide="ignore", invalid="ignore"):
            R = g(Wt * R0, 2, low_pass) / Wl
        R[~np.isfinite(R)] = 0
        wr = Wl * R
        with np.errstate(divide="ignore", invalid="ignore"):
            Rx = R - g(wr, 2, high_pass) / g(Wl, 2, high_pass)
            Rx[~np.isfinite(Rx)] = 0
            Ry = R - g(wr, 1, high_pass) / g(Wl, 1, high_pass)
            Ry[~np.isfinite(Ry)] = 0
        out.append((np.ascontiguousarray(Rx + Ry), np.ascontiguousarray(Wl)))
    return out


def overlap_cross_check(img0, w0, img1, w1, res, sigma=2.0, tolerance=0.375, log=print):
    """Guard of the sub-pixel overlap search. find_offset compares img0 with img1 interpolated bilinearly; at
    half-pixel offsets the interpolation averages img1's noise (up to 4 x less variance), so on noise-dominated
    stripes the L1 loss is lowest there whatever the true offset. The search is repeated around the same integer
    offset on copies smoothed by `sigma` pixels, whose noise interpolation can hardly lower; a warning is logged if
    the two sub-pixel offsets differ by more than `tolerance` pixels. The calibration keeps the reference estimate.
    Returns dict(sub=(dy, dx) of the smoothed search, differs=bool)."""
    g = lambda a: _core.gaussian_filter1d(_core.gaussian_filter1d(np.ascontiguousarray(a, np.float64), 1, sigma, "reflect"),
                                          2, sigma, "reflect")
    r = _core.find_offset(g(img0), np.ascontiguousarray(w0, np.float64), g(img1), np.ascontiguousarray(w1, np.float64),
                          int(res["best_int"][0]), int(res["best_int"][1]), 1, 1)
    sub = tuple(float(v) for v in r["best_sub"])
    differs = max(abs(a - b) for a, b in zip(sub, res["best_sub"])) > tolerance
    if differs:
        log(f"  WARNING: sub-pixel overlap {tuple(res['best_sub'])} is not confirmed on smoothed stripes ({sub}): the "
            f"stripes may be too noisy for the half-pixel estimate; check the vector (vector_override / --vector)")
    return dict(sub=sub, differs=bool(differs))


def find_overlap(ext_main: Extractor, span_main: Span, ext_comp: Extractor, span_comp: Span, init_dx,
                 radius_x=40, radius_z=10, n_slices=80, low_pass=0.4, high_pass=10.0, log=print, progress=False):
    """(dy, dx): main[y, x] == companion[y - dy, x - dx] at equal angle and view height."""
    zm, zc = ext_main.scan.z_translation_m, ext_comp.scan.z_translation_m
    same_dir = (zm[-1] >= zm[0]) == (zc[-1] >= zc[0])
    if not same_dir:
        raise NotImplementedError("scans with opposite z directions are not supported yet")
    a0, a1 = span_main.doable
    b0, b1 = span_comp.doable
    d0, d1 = max(a0, b0) + 3 * radius_z, min(a1, b1) - 3 * radius_z
    slices = choose_slices(d0, d1, n_slices)
    sr = 4 * radius_z
    S = 2 * sr + 1
    reqs = []
    for ext, span in ((ext_main, span_main), (ext_comp, span_comp)):
        reqs.append({k: [] for k in ("p0", "p1", "f", "zs0", "zs1", "zf0", "zf1")})
    for h in slices:
        angle = None
        for r, span in zip(reqs, (span_main, span_comp)):
            p0c, p1c, avail = span.chunk(h - sr, h + sr)
            if angle is None:
                angle = avail.mean()
            else:
                angle = angle + 360 * int(round((avail.mean() - angle) / 360))
            q0, q1, f = _bracket(avail, angle)
            P0, P1 = q0 + p0c, q1 + p0c
            base = h - sr
            r["p0"].append(P0); r["p1"].append(P1); r["f"].append(f)
            r["zs0"].append(base - span.ishift[P0]); r["zs1"].append(base - span.ishift[P1])
            r["zf0"].append(span.fract[P0]); r["zf1"].append(span.fract[P1])
    stripes = []
    with _bar(progress, 2 * len(slices), "HA -> TA overlap", "stripe") as bar:
        for ext, r in zip((ext_main, ext_comp), reqs):
            stripes.append(ext.extract(*[np.asarray(r[k]) for k in ("p0", "p1", "f", "zs0", "zs1", "zf0", "zf1")], S=S,
                                       progress=bar.update))
    (img0, w0), (img1, w1) = _filter_overlap([stripes[0][0], stripes[1][0]], [stripes[0][1], stripes[1][1]],
                                             low_pass, high_pass)
    res = _core.find_offset(img0, w0, img1, w1, 0, int(init_dx), radius_z, radius_x)
    log(f"  overlap: {len(slices)} stripes x {S} rows, integer {res['best_int']} -> sub-pixel {res['best_sub']}")
    res["cross_check"] = overlap_cross_check(img0, w0, img1, w1, res, log=log)
    res["slices"] = slices
    return res


# ============================================================================ top level
# ============================================================================ helical pitch
PITCH_MIN_TRAVEL_ROWS = 20.0     # least travel in half a turn for the pitch to be measurable
PITCH_MAX_ERROR = 0.02           # a larger |scale - 1| is taken for a failed measurement


def pitch_scale_from_pairs(line, angles_deg, z_pix, axis, shape, n_pairs=40, min_half_width=64, log=print,
                           progress=None):
    """Detector rows the sample really moves per row of recorded z translation, from the frames themselves.

    A parallel-beam projection and the one half a turn later are mirror images about the rotation axis, displaced
    along the rows by the sample's travel in between: what is at row v of projection p is at row v + (z_pix[p] -
    z_pix[q]) x scale of projection q. The displacement of n_pairs such pairs spread over the scan is measured
    (_core.mirror_row_shift) and compared with the recorded one. Only pairs with the full travel of half a turn count
    (those reaching into a stationary turn or a ramp also see where the stage came to rest).

    line(p): the (rows, cols) float32 line integrals of projection p; axis: the rotation axis in its columns.
    Returns a dict: scale (1.0 unless a significant error was measured), measured (the raw ratio, or None),
    stderr, n_pairs, significant, reason (when nothing could be measured), pairs [[p, q, recorded, measured,
    correlation], ...]."""
    rows, cols = shape
    out = dict(scale=1.0, measured=None, stderr=None, n_pairs=0, significant=False, pairs=[])
    half = int(min(axis, cols - 1 - axis)) - 4
    if half < min_half_width:
        return dict(out, reason=f"rotation axis at column {axis:.1f}: only {max(half, 0)} mirrored columns")
    ang = np.asarray(angles_deg, np.float64)
    z = np.asarray(z_pix, np.float64)
    n = len(ang)
    step = float(np.median(np.abs(np.diff(ang))))
    turn = 180.0 if ang[-1] >= ang[0] else -180.0
    q_of = np.searchsorted(ang * np.sign(turn), (ang + turn) * np.sign(turn))
    q_of = np.clip(q_of, 1, n - 1)
    q_of = np.where(np.abs(ang[q_of - 1] - ang - turn) < np.abs(ang[q_of] - ang - turn), q_of - 1, q_of)
    ok = np.abs(ang[q_of] - ang - turn) <= 0.5 * step + 1e-9
    travel = z - z[q_of]
    most = float(np.abs(travel[ok]).max()) if ok.any() else 0.0
    if most < PITCH_MIN_TRAVEL_ROWS:
        return dict(out, reason=f"travel of {most:.1f} rows in half a turn: not a helical scan")
    # (the two projections must share enough rows beside the search range and the kernel's margins)
    full = np.flatnonzero(ok & (np.abs(travel) >= 0.98 * most) & (rows - np.abs(travel) >= max(64, 0.1 * rows)))
    if len(full) < 8:
        return dict(out, reason="too few projection pairs half a turn apart see the same rows")
    chosen = full[np.unique(np.linspace(0, len(full) - 1, n_pairs).round().astype(int))]
    pairs = []
    for p in chosen:
        q = int(q_of[p])
        d = float(travel[p])
        radius = max(8, int(math.ceil(0.01 * abs(d))) + 2)
        shift, corr, edge = _core.mirror_row_shift(line(int(p)), line(q), float(axis), half, d, radius)
        if not edge and corr > 0.2:
            pairs.append([int(p), q, d, float(shift), float(corr)])
        if progress:
            progress(1)
    out["pairs"] = pairs
    if len(pairs) < 8:
        return dict(out, reason=f"only {len(pairs)} of {len(chosen)} pairs correlate")
    r = np.array([m / d for _, _, d, m, _ in pairs])
    med = float(np.median(r))
    mad = float(np.median(np.abs(r - med)))
    keep = np.abs(r - med) <= 4 * 1.4826 * mad + 1e-5
    measured = float(r[keep].mean())
    stderr = float(r[keep].std(ddof=1) / math.sqrt(keep.sum())) if keep.sum() > 1 else float("inf")
    out.update(measured=measured, stderr=stderr, n_pairs=int(keep.sum()))
    if abs(measured - 1) > PITCH_MAX_ERROR:
        log(f"  WARNING: helical pitch: the frames give {measured:.5f} x the recorded travel: taken for a failed "
            f"measurement, not applied (set z_translation_scale to use a value)")
        return dict(out, reason="implausible ratio")
    out["significant"] = bool(abs(measured - 1) > 3 * stderr)
    if out["significant"]:
        out["scale"] = measured
    return out


def find_pitch_scale(ext: "Extractor", span: "Span", axis, n_pairs=40, log=print, progress=False):
    """pitch_scale_from_pairs on a scan: its flat-fielded projections, its recorded travel (span), its rotation axis
    (detector columns)."""
    def line(p):
        raw = ext.src.frames(np.array([p]))[0].astype(np.float32)
        sig = ((raw - ext.dark) / ext.cd[p]) / np.maximum(ext._flat(p), FLT_MIN)
        return np.ascontiguousarray(-np.log(np.clip(sig, 1e-6, None)), np.float32)
    with _bar(progress, n_pairs, "helical pitch", "pair") as bar:
        return pitch_scale_from_pairs(line, span.angles, span.z_pix, axis, ext.scan.frame_shape, n_pairs, log=log,
                                      progress=bar.update)


def measure_pitch(main, companion, arrays, cal_cor_main, placement, rot_center, log=print, progress=False):
    """The helical pitch of the pair, from whichever scan's detector holds the rotation axis with the more mirrored
    columns (both scans ride the same stage): the dict of pitch_scale_from_pairs, with the scan's name."""
    wm = arrays["weight_map"]
    cands = []
    for tag, sc in (("main", main), ("companion", companion)):
        pl = placement[sc.name]
        axis = cal_cor_main if tag == "main" else rot_center - (pl.shift_x - pl.x_readjustment)
        cands.append((min(axis, sc.frame_shape[1] - 1 - axis), tag, sc, axis))
    _, tag, sc, axis = max(cands, key=lambda c: c[0])
    ext = Extractor(sc, arrays[f"{tag}_dark"], arrays[f"{tag}_flats"], wm)
    res = find_pitch_scale(ext, Span(sc, 50.0), float(axis), log=log, progress=progress)
    res["scan"] = sc.name
    if res["measured"] is None:
        log(f"helical pitch: not measured ({res.get('reason')}): the recorded z translation is used")
    else:
        log(f"helical pitch: the frames of {sc.name} show {res['measured']:.5f} +- {res['stderr']:.5f} x the recorded z "
            f"travel ({res['n_pairs']} pairs half a turn apart): "
            + (f"z_translation_scale {res['scale']:.5f}" if res["significant"] else "no significant error, scale 1"))
    return res


@dataclass
class ScanPlacement:
    shift_x: int
    x_readjustment: float
    z_readjustment: float


# bump when anything that changes the cached results changes (reduction, COR / overlap search, cache layout)
CALIBRATION_VERSION = 2

# The rotation-axis search fits the axis at both ends of the scan (in steps of 1/8 pixel); the reconstruction uses one
# axis for every slice. A larger difference than this needs the per-projection lateral correction.
COR_DRIFT_TOLERANCE_PX = 0.25


def check_cor_drift(cor_begin_end, tolerance=COR_DRIFT_TOLERANCE_PX, log=print):
    """The measured rotation axis at the two ends of the scan must agree: slices are reconstructed with one axis
    (the fitted line at z = 0), so a drift along the scan would misplace it by up to that drift."""
    a, b = (float(v) for v in cor_begin_end)
    if abs(b - a) > tolerance:
        raise NotImplementedError(
            f"the rotation axis drifts by {b - a:+.3f} px along the scan ({a:.3f} -> {b:.3f}); per-projection lateral "
            f"correction is not supported yet. To reconstruct with one fixed axis, give it explicitly "
            f"(cor_override / --cor)")
    if b != a:
        log(f"  rotation axis differs by {b - a:+.3f} px between the ends of the scan (within {tolerance} px: ignored)")


@dataclass
class Calibration:
    main: str
    companion: str
    cor_main_detector: float          # rotation axis in the main scan's own columns
    vector: tuple                     # (dy, dx) main -> companion
    rot_center: float                 # rotation axis in the merged frame
    merged_width: int
    placement: dict                   # name -> ScanPlacement (as dict)
    apodisation: dict                 # name -> (w_left, w_right)
    estimated_cor_from_motor_pix: dict
    details: dict = field(default_factory=dict)
    # Detector rows the sample really moves per row of recorded z translation (measure_pitch; details["pitch"]).
    # None: not measured yet (a cache written before the measurement existed: measured and added on loading).
    z_translation_scale: float | None = None
    # The detector's blur field (measure_detector_blur; details["detector_blur"]): [a11, a12, a21, a22, c1, c2], a uniform blur of
    # A r + c pixels at r = (column, row) from the detector's centre in units of 1000 pixels, from the noise of the
    # flats. []: measured, nothing significant. None: not measured yet (an older cache: measured on loading).
    detector_blur_field: list | None = None
    # Registration of the companion's frames on the main scan's (measure_registration; details["registration"]):
    # dict(projection, row_offset, row_slope, col_slope: knots along the companion's projections, segments: the
    # stretches they are interpolated within, slope_col, slope_row). {}: measured, nothing usable.
    # None: not measured yet (an older cache; one without the lateral knots is measured again).
    companion_registration: dict | None = None
    # Refinement of the rotation axis (axis_from_pairs; details["axis"]): pixels to add to cor_main_detector and
    # rot_center, which keep the search's value (on its grid of a quarter pixel). None: not measured yet.
    axis_refinement: float | None = None
    # The turns recorded with the z stage at rest against the helix (stationary_from_pairs; details["stationary"]):
    # dict(segments=[[first projection, end, rows], ...]): rows to add to the vertical position of those projections
    # of the main scan (and of the companion's of the same angles). {}: measured, nothing to apply. None: not measured.
    stationary_registration: dict | None = None
    # Rolling shutter (recon.measure_rolling_shutter; details["rolling_shutter"]): the time from the read-out of the
    # detector's first row to its last, in frame periods (pipeline.hpp). None: not measured yet.
    rolling_shutter: float | None = None
    source: str = "measured"          # measured | override (cor and / or vector given by the caller; never cached)
    fingerprint: str = ""             # acquisition_fingerprint() of the scan pair it was computed from
    version: int = CALIBRATION_VERSION

    def to_json(self):
        return json.dumps(asdict(self), indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def acquisition_fingerprint(main: BlissScan, companion: BlissScan) -> str:
    """sha256 over everything the calibration depends on: calibration version, scan identities, metadata, frame
    geometry, trajectories and the referenced raw files (path, size, mtime). A replaced acquisition (even under the
    same name) or a different pair invalidates the cache."""
    import hashlib
    h = hashlib.sha256(f"helirec-calibration-v{CALIBRATION_VERSION}".encode())
    for sc in (main, companion):
        meta = dict(name=sc.name, master=os.path.abspath(sc.master), shape=list(sc.frame_shape), dtype=str(sc.dtype),
                    n_proj=len(sc.projections), n_darks=len(sc.darks), n_flats=[len(f) for f in sc.flats],
                    energy_kev=sc.energy_kev, distance_m=sc.distance_m, pixel_size_m=sc.pixel_size_m,
                    est_cor=sc.estimated_cor_from_motor_pix)
        h.update(json.dumps(meta, sort_keys=True).encode())
        for arr in (sc.unwrapped_angles_deg, sc.z_translation_m, sc.control_A, sc.flats_unwrapped_angles_deg):
            h.update(np.ascontiguousarray(arr, np.float64).tobytes())
        for fn in [sc.master, *sc.referenced_files()]:
            st = os.stat(fn)
            h.update(f"{os.path.abspath(fn)}|{st.st_size}|{st.st_mtime_ns}".encode())
    return h.hexdigest()


def _atomic_write(path, write):
    """write(tmp_path) then rename over `path` (readers never see a partial file)."""
    tmp = f"{path}.tmp{os.getpid()}"
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def placement_from_vector(vector, n_cols):
    """nabuxx check_companion: outermost companion anchored at x = 0 of the merged frame."""
    dy, dx = vector
    cum = [(0.0, 0.0), (float(dy), float(dx))]
    outer_x = cum[-1][1]
    anchor = -outer_x if outer_x < 0 else 0.0
    out = []
    for j, (z, x) in enumerate(cum):
        d = anchor + x
        sx = int(math.floor(d))
        out.append(ScanPlacement(shift_x=sx, x_readjustment=float(np.float32(sx - d)),
                                 z_readjustment=float(z) if j > 0 else 0.0))
    width = n_cols + int(math.ceil(abs(outer_x)))
    return out, width, anchor


REDUCTION_KEYS = ("main_dark", "main_flats", "companion_dark", "companion_flats", "weight_map")


def _load_cache(cache_dir, fp, log):
    """(calibration dict or None, reduction arrays or None) from cache_dir, each only if its fingerprint matches."""
    js, npz = os.path.join(cache_dir, "calibration.json"), os.path.join(cache_dir, "calibration_arrays.npz")
    arrays = None
    if os.path.exists(npz):
        with np.load(npz) as z:
            if "_fingerprint" in z.files and str(z["_fingerprint"]) == fp and all(k in z.files for k in REDUCTION_KEYS):
                arrays = {k: z[k] for k in REDUCTION_KEYS}
            else:
                log(f"calibration cache {npz}: other acquisition / version, ignored")
    cal = None
    if arrays is not None and os.path.exists(js):
        d = json.load(open(js))
        if d.get("fingerprint") == fp and d.get("version") == CALIBRATION_VERSION and d.get("source") == "measured":
            cal = d
        else:
            log(f"calibration cache {js}: other acquisition / version or not a measured calibration, ignored")
    return cal, arrays


def calibrate(main: BlissScan, companion: BlissScan, cache_dir=None, log=print, cor_override=None,
              vector_override=None, progress=None):
    """Full calibration from raw data. Returns (Calibration, arrays dict).

    cache_dir: the darks/flats reduction and the *measured* calibration are cached there, keyed by
    acquisition_fingerprint(); a cache of another acquisition or calibration version is ignored and replaced.
    cor_override / vector_override replace the measured values for this call only (Calibration.source =
    "override"); they are never written to the cache, so a later run without overrides gets the measured values."""
    if progress is None:   # progress bars for the long steps: whenever the log is shown
        progress = log is print
    fp = acquisition_fingerprint(main, companion)
    override = cor_override is not None or vector_override is not None
    cached_cal, arrays = _load_cache(cache_dir, fp, log) if cache_dir else (None, None)
    if cached_cal is not None:
        cached_cal = _current_names(cached_cal)
    if cached_cal is not None and not override:
        d = dict(cached_cal)
        d["placement"] = {k: ScanPlacement(**v) for k, v in d["placement"].items()}
        d["vector"] = tuple(d["vector"])
        d["apodisation"] = {k: tuple(v) for k, v in d["apodisation"].items()}
        cal = Calibration(**d)
        if "cor" in cal.details:
            check_cor_drift(cal.details["cor"]["cor_begin_end"], log=log)
        log(f"calibration loaded from {cache_dir}")
        if cal.z_translation_scale is None:   # (a cache from before the pitch measurement)
            _add_pitch(cal, main, companion, arrays, log, progress)
            _save_calibration(cache_dir, cal)
        if cal.detector_blur_field is None:        # (a cache from before the blur measurement)
            _add_detector_blur(cal, main, arrays, log)
            _save_calibration(cache_dir, cal)
        if cal.companion_registration is None or (cal.companion_registration and "col_slope" not in cal.companion_registration):
            # (a cache from before the frame registration, or from before its lateral part)
            _add_registration(cal, main, companion, arrays, log, progress)
            _save_calibration(cache_dir, cal)
        if cal.axis_refinement is None:             # (a cache from before the axis refinement)
            _add_axis(cal, main, arrays, log, progress)
            _save_calibration(cache_dir, cal)
        if cal.stationary_registration is None:     # (a cache from before the stationary turns' registration)
            _add_stationary(cal, main, arrays, log, progress)
            _save_calibration(cache_dir, cal)
        return cal, {**arrays, **_scan_weights(arrays["weight_map"], main, cal)}
    t0 = time.time()
    if arrays is None:
        arrays = {}
        for tag, sc in (("main", main), ("companion", companion)):
            log(f"[{tag}] reducing darks/flats of {sc.name} ...")
            with _bar(progress, len(sc.darks) + sum(len(refs) for refs in sc.flats), f"darks/flats ({tag})", "frame") as bar:
                dark, flats = reduce_darks_flats(sc, bar.update)
            arrays[f"{tag}_dark"], arrays[f"{tag}_flats"] = dark, flats
        # weight map from all flats (companion first, as the reference --scan_files order)
        arrays["weight_map"] = weight_map([arrays["companion_flats"], arrays["main_flats"]])
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            def save(t):
                with open(t, "wb") as fh:
                    np.savez(fh, _fingerprint=np.array(fp), **arrays)
            _atomic_write(os.path.join(cache_dir, "calibration_arrays.npz"), save)
    wm = arrays["weight_map"]
    W = main.frame_shape[1]
    details = {}
    ext = {}
    def extractors():
        if not ext:
            ext["m"] = Extractor(main, arrays["main_dark"], arrays["main_flats"], wm)
            ext["c"] = Extractor(companion, arrays["companion_dark"], arrays["companion_flats"], wm)
            ext["sm"], ext["sc"] = Span(main, 50.0), Span(companion, 50.0)
        return ext
    if cor_override is None:
        log("rotation axis ...")
        e = extractors()
        cr = find_cor(e["m"], e["sm"], main.estimated_cor_from_motor_pix, log=log, progress=progress)
        check_cor_drift(cr["cor_begin_end"], log=log)
        cor_main = float(cr["cor0"])
        details["cor"] = dict(cor_begin_end=cr["cor_begin_end"], z_begin_end=cr["z_begin_end"],
                              best_index=cr["best_index"])
    else:
        cor_main = float(cor_override)
        details["cor_override"] = cor_main
    if vector_override is None:
        log("HA -> TA overlap ...")
        e = extractors()
        init_dx = round(main.estimated_cor_from_motor_pix - companion.estimated_cor_from_motor_pix)
        ov = find_overlap(e["m"], e["sm"], e["c"], e["sc"], init_dx, log=log, progress=progress)
        vector = tuple(float(v) for v in ov["best_sub"])
        details["overlap"] = dict(best_int=ov["best_int"], best_int_loss=ov["best_int_loss"],
                                  best_sub_loss=ov["best_sub_loss"], smoothed_sub=ov["cross_check"]["sub"],
                                  smoothed_differs=ov["cross_check"]["differs"])
    else:
        vector = tuple(float(v) for v in vector_override)
        details["vector_override"] = list(vector)
    pl, width, anchor = placement_from_vector(vector, W)
    rot_center = cor_main + anchor
    (aw_m, aw_c) = apodisation_widths(W, main.estimated_cor_from_motor_pix, vector[1])
    cal = Calibration(main=main.name, companion=companion.name, cor_main_detector=cor_main, vector=vector,
                      rot_center=float(rot_center), merged_width=width,
                      placement={main.name: pl[0], companion.name: pl[1]},
                      apodisation={main.name: aw_m, companion.name: aw_c},
                      estimated_cor_from_motor_pix={main.name: main.estimated_cor_from_motor_pix,
                                                    companion.name: companion.estimated_cor_from_motor_pix},
                      details=details, source="override" if override else "measured", fingerprint=fp)
    _add_pitch(cal, main, companion, arrays, log, progress)
    if override and cached_cal is not None and cached_cal.get("detector_blur_field") is not None:
        # (the detector does not change with the caller's axis or vector: take the cached measurement)
        cal.detector_blur_field = list(cached_cal["detector_blur_field"])
        cal.details["detector_blur"] = cached_cal.get("details", {}).get("detector_blur")
    else:
        _add_detector_blur(cal, main, arrays, log)
    _add_registration(cal, main, companion, arrays, log, progress)
    if cor_override is not None:
        cal.axis_refinement = 0.0                    # (the caller's axis is used as given)
    else:
        _add_axis(cal, main, arrays, log, progress)
    _add_stationary(cal, main, arrays, log, progress)
    if override and cached_cal is not None and cached_cal.get("rolling_shutter") is not None:
        cal.rolling_shutter = float(cached_cal["rolling_shutter"])     # (the camera's: not the caller's geometry's)
        cal.details["rolling_shutter"] = cached_cal.get("details", {}).get("rolling_shutter")
    log(f"calibration ({cal.source}) done in {time.time() - t0:.0f}s: COR {cor_main:.3f} (detector), vector {vector}, "
        f"rotation axis {rot_center:.3f} in a {width}-px merged frame")
    if cache_dir and not override:
        _save_calibration(cache_dir, cal)
    return cal, {**arrays, **_scan_weights(wm, main, cal)}


def _save_calibration(cache_dir, cal):
    def save_json(t):
        with open(t, "w") as fh:
            fh.write(cal.to_json())
    _atomic_write(os.path.join(cache_dir, "calibration.json"), save_json)


def _add_pitch(cal, main, companion, arrays, log, progress):
    """measure the helical pitch and record it in the calibration"""
    res = measure_pitch(main, companion, arrays, cal.cor_main_detector, cal.placement, cal.rot_center, log=log,
                        progress=progress)
    cal.z_translation_scale = float(res["scale"])
    cal.details["pitch"] = res


def _add_registration(cal, main, companion, arrays, log, progress):
    """register the companion's frames on the main scan's and record it in the calibration"""
    res = measure_registration(main, companion, arrays, cal, log=log, progress=progress)
    cal.companion_registration = res.pop("registration")
    cal.details["registration"] = res


# ============================================================================ companion frames against the main scan's
REGISTRATION_MIN_TILES = 20
REGISTRATION_MAX_SCATTER = 0.35    # rows: scatter of the pairs about the smoothed offset above which nothing is applied


def _smooth_knots(y, median, mean):
    """running median, then running mean (edge values repeated)"""
    y = np.asarray(y, np.float64)
    def run(v, n, f):
        if n <= 1 or len(v) < 2:
            return v
        h = n // 2
        pad = np.concatenate([np.full(h, v[0]), v, np.full(h, v[-1])])
        return np.array([f(pad[i:i + n]) for i in range(len(v))])
    return run(run(y, min(median, 2 * (len(y) // 2) + 1), np.median), min(mean, 2 * (len(y) // 2) + 1), np.mean)


def registration_arrays(reg, n_proj):
    """Calibration.companion_registration -> the companion's registration per projection, as make_scan_data takes
    it: dict(row_offset, row_slope, slope_col[, col_slope, slope_row][, col_offset]) (float32 arrays; None when
    there is no registration)"""
    if not reg:
        return None
    p = np.asarray(reg["projection"], np.float64)
    keys = [k for k in ("row_offset", "row_slope", "col_offset", "col_slope") if k in reg]
    out = {k: np.zeros(n_proj, np.float32) for k in keys}
    for s0, s1 in reg["segments"]:
        m = (p >= s0) & (p < s1)
        if not m.any():
            continue
        x = np.arange(s0, min(s1, n_proj))
        for k in keys:
            out[k][x] = np.interp(x, p[m], np.asarray(reg[k])[m])
    out["slope_col"] = float(reg["slope_col"])
    if "col_slope" in reg or "col_offset" in reg:
        out["slope_row"] = float(reg["slope_row"])
    return out


def measure_registration(main, companion, arrays, cal, n_pairs=240, tile=240, log=print, progress=False):
    """How the companion's frames lie on the main scan's, along the scan, from the frames themselves.

    The calibrated vector places the companion by one (dy, dx) for the whole scan, found where the two detectors
    overlap. For n_pairs projections spread over the scan, the main frame and the companion frame of the same angle
    are compared where they see the same rays: their high-passed line integrals are registered on tiles
    (_core.tile_shifts), and per pair  dv = a + phi u,  du = b + psi v  are fitted (u: columns from the rotation
    axis, v: rows from the detector's centre): the row displacement at the axis and its slope along the columns, the
    column displacement at the axis and its slope along the rows. phi and -psi are a roll of the companion's frames
    against the main scan's about the rotation axis. With a roll the one dy is right only at the middle of the
    overlap.
    b is measured and kept with the pairs but not applied: where the detector's blur is lopsided and differs between
    the two places a ray is recorded at, the frames' features are displaced by the blur itself, and b is that as much
    as a misplacement (PHerc0343P: applying it moved the companion's content 0.3-0.4 voxel radially off the main
    scan's and lowered the correlation with a finer scan; the slopes are free of it).
    Against what the geometry assumes (the recorded z of both scans, the pitch scale, dy) the residual row offset at
    the axis and the two slopes are smoothed along the scan (running median, then mean), within the
    stretches where the recorded difference of the two scans' z is continuous, and returned as knots for
    registration_arrays().

    Returns a dict: registration ({} when too few pairs register or they scatter by more than
    REGISTRATION_MAX_SCATTER rows), pairs [[main projection, companion projection, tiles, a, phi (rows per 1000
    columns), b, psi (columns per 1000 rows), residual], ...], scatter, reason."""
    t0 = time.time()
    rows, cols = main.frame_shape
    out = dict(registration={}, pairs=[], scatter=None, reason=None)
    if tuple(companion.frame_shape) != (rows, cols):
        return dict(out, reason="the scans' frames differ in shape")
    pa, pb = cal.placement[main.name], cal.placement[companion.name]
    off = (pa.shift_x - pa.x_readjustment) - (pb.shift_x - pb.x_readjustment)     # companion column = main column + off
    lo, hi = max(0.0, -off), min(cols - 1.0, cols - 1.0 - off)                      # main columns both scans hold
    c0, n_ov = int(math.ceil(lo)), int(math.floor(hi)) - int(math.ceil(lo))
    if n_ov < 2 * tile:
        return dict(out, reason=f"the scans overlap over {max(n_ov, 0)} columns only")
    scale = cal.z_translation_scale or 1.0
    dy = pb.z_readjustment - pa.z_readjustment
    am, ac = np.asarray(main.unwrapped_angles_deg, np.float64), np.asarray(companion.unwrapped_angles_deg, np.float64)
    zm = (np.asarray(main.z_translation_m, np.float64) - main.z_translation_m[0]) / main.pixel_size_m * scale
    zc = (np.asarray(companion.z_translation_m, np.float64) - companion.z_translation_m[0]) / companion.pixel_size_m * scale
    step = float(np.median(np.abs(np.diff(am))))
    order = np.argsort(ac)
    def partner(p):
        i = int(np.clip(np.searchsorted(ac[order], am[p]), 1, len(ac) - 1))
        q = order[i - 1] if abs(ac[order[i - 1]] - am[p]) <= abs(ac[order[i]] - am[p]) else order[i]
        return int(q) if abs(ac[q] - am[p]) <= 0.5 * step + 1e-9 else -1
    wm = arrays["weight_map"]
    ext = {"m": Extractor(main, arrays["main_dark"], arrays["main_flats"], wm),
           "c": Extractor(companion, arrays["companion_dark"], arrays["companion_flats"], wm)}
    def line(e, p):
        raw = e.src.frames(np.array([p]))[0].astype(np.float32)
        sig = ((raw - e.dark) / e.cd[p]) / np.maximum(e._flat(p), FLT_MIN)
        return -np.log(np.clip(sig, 1e-6, None))
    def high_pass(im):
        im = np.ascontiguousarray(im, np.float32)
        return im - _core.gaussian_filter1d(_core.gaussian_filter1d(im, 0, 6.0, "reflect"), 1, 6.0, "reflect")
    io, fo = int(math.floor(c0 + off)), (c0 + off) - math.floor(c0 + off)
    picks = np.unique(np.linspace(0, len(am) - 1, n_pairs + 2)[1:-1].round().astype(int))
    pairs = []
    with _bar(progress, len(picks), "frame registration", "pair") as bar:
        for p in picks:
            q = partner(int(p))
            if q >= 0:
                A = high_pass(line(ext["m"], int(p))[:, c0:c0 + n_ov])
                Bf = line(ext["c"], q)
                B = high_pass(Bf[:, io:io + n_ov] * (1 - fo) + Bf[:, io + 1:io + 1 + n_ov] * fo if fo > 0 else Bf[:, io:io + n_ov])
                t = _core.tile_shifts(np.ascontiguousarray(A), np.ascontiguousarray(B), tile, tile // 2, 20, 6)
                t = t[np.isfinite(t[:, 2]) & (t[:, 4] > 0.3)]
                if len(t) >= REGISTRATION_MIN_TILES:
                    u = (t[:, 1] + c0 - cal.cor_main_detector) / 1000.0
                    v = (t[:, 0] - 0.5 * (rows - 1)) / 1000.0
                    M = np.stack([np.ones(len(t)), u, v], -1)
                    cv, cu = np.linalg.lstsq(M, t[:, 2], rcond=None)[0], np.linalg.lstsq(M, t[:, 3], rcond=None)[0]
                    keep = np.abs(t[:, 2] - M @ cv) <= 3 * 1.4826 * np.median(np.abs(t[:, 2] - M @ cv)) + 0.05
                    if keep.sum() >= REGISTRATION_MIN_TILES:
                        cv = np.linalg.lstsq(M[keep], t[keep, 2], rcond=None)[0]
                    keep = np.abs(t[:, 3] - M @ cu) <= 3 * 1.4826 * np.median(np.abs(t[:, 3] - M @ cu)) + 0.05
                    if keep.sum() >= REGISTRATION_MIN_TILES:
                        cu = np.linalg.lstsq(M[keep], t[keep, 3], rcond=None)[0]
                    # what the geometry assumes: a row of the companion = the main scan's - (z difference) - dy
                    residual = cv[0] + dy + (zc[q] - zm[p])
                    pairs.append([int(p), q, int(len(t)), float(cv[0]), float(cv[1]), float(cu[0]), float(cu[2]), float(residual)])
            bar.update(1)
    out["pairs"] = pairs
    if len(pairs) < max(8, len(picks) // 2):
        log(f"frame registration: only {len(pairs)} of {len(picks)} projection pairs register: none applied")
        return dict(out, reason="too few projection pairs register")
    P = np.array(pairs)
    q, d = P[:, 1], zc[P[:, 1].astype(int)] - zm[P[:, 0].astype(int)]
    # stretches of continuous recorded difference (e.g. the stationary turns and the helix), in companion projections
    cuts = [0]
    for i in np.flatnonzero(np.abs(np.diff(d)) > 0.25):
        qa, qb = int(q[i]), int(q[i + 1])
        cut = (qa + qb + 1) // 2
        if len(am) == len(ac) and np.allclose(am[qa:qb + 1], ac[qa:qb + 1], atol=0.5 * step):
            full = zc[qa:qb + 1] - zm[qa:qb + 1]            # (the same projections in both scans: the exact place)
            cut = qa + int(np.argmax(np.abs(full - d[i]) > 0.5 * abs(d[i + 1] - d[i])))
        cuts.append(cut)
    cuts.append(len(ac))
    segments = [[int(a), int(b)] for a, b in zip(cuts[:-1], cuts[1:]) if b > a]
    row_offset, row_slope, dev = np.zeros(len(P)), np.zeros(len(P)), np.zeros(len(P))
    col_offset, col_slope = np.zeros(len(P)), np.zeros(len(P))
    for s0, s1 in segments:
        m = (q >= s0) & (q < s1)
        if m.any():
            smooth = _smooth_knots(P[m, 7], 9, 5)
            row_offset[m] = -smooth
            dev[m] = P[m, 7] - smooth
            row_slope[m] = _smooth_knots(P[m, 4], 21, 11) / 1000.0
            col_offset[m] = _smooth_knots(P[m, 5], 9, 5)
            col_slope[m] = _smooth_knots(P[m, 6], 21, 11) / 1000.0
    scatter = float(1.4826 * np.median(np.abs(dev)))
    out["scatter"] = scatter
    summary = (f"frame registration of {companion.name} on {main.name} ({len(pairs)} projection pairs): rows at the "
               f"rotation axis {(-row_offset).min():+.2f} .. {(-row_offset).max():+.2f} from the calibrated vector, roll "
               f"{np.median(row_slope) * 1e3:+.2f} mrad ({row_slope.min() * 1e3:+.2f} .. {row_slope.max() * 1e3:+.2f}), "
               f"scatter {scatter:.2f} rows; columns {np.median(col_slope) * 1e3:+.2f} per 1000 rows (and "
               f"{col_offset.min():+.2f} .. {col_offset.max():+.2f} at the axis, not applied)  [{time.time() - t0:.0f}s]")
    if scatter > REGISTRATION_MAX_SCATTER:
        log(summary + ": too scattered, none applied")
        return dict(out, reason="the pairs scatter too much")
    log(summary)
    out["registration"] = dict(projection=[int(v) for v in q], row_offset=[float(v) for v in row_offset],
                               row_slope=[float(v) for v in row_slope], col_slope=[float(v) for v in col_slope],
                               segments=segments,
                               slope_col=float(cal.rot_center - (pb.shift_x - pb.x_readjustment)),
                               slope_row=0.5 * (rows - 1))
    return out


# ============================================================================ one scan's frames against themselves
AXIS_MAX_REFINEMENT = 1.0          # pixels: a larger correction of the searched axis is taken for a failed measurement
STATIONARY_MIN_ROWS = 0.05         # rows: a smaller offset of a stationary turn is not applied
STATIONARY_MAX_SCATTER = 0.35      # rows: scatter of the pairs above which nothing is applied
STATIONARY_RAMP = 40               # projections next to a change of the stage's motion that are not used as partners


def _line_integrals(ext, p):
    """flat-fielded line integrals of projection p, float32 (rows, cols)"""
    raw = ext.src.frames(np.array([p]))[0].astype(np.float32)
    sig = ((raw - ext.dark) / ext.cd[p]) / np.maximum(ext._flat(p), FLT_MIN)
    return np.ascontiguousarray(-np.log(np.clip(sig, 1e-6, None)), np.float32)


def _high_pass(im, sigma=6.0):
    im = np.ascontiguousarray(im, np.float32)
    return im - _core.gaussian_filter1d(_core.gaussian_filter1d(im, 0, sigma, "reflect"), 1, sigma, "reflect")


def _partner(ang, p, turn_deg):
    """the projection turn_deg (signed, in the scan's sense of rotation) from projection p, or -1"""
    step = float(np.median(np.abs(np.diff(ang))))
    sense = 1.0 if ang[-1] >= ang[0] else -1.0
    target = ang[p] + sense * turn_deg
    order = np.argsort(ang)
    i = int(np.clip(np.searchsorted(ang[order], target), 1, len(ang) - 1))
    q = order[i - 1] if abs(ang[order[i - 1]] - target) <= abs(ang[order[i]] - target) else order[i]
    return int(q) if abs(ang[q] - target) <= 0.5 * step + 1e-9 else -1


def _row_registered(a, b, s, tile):
    """Tiles of frame a on frame b whose content lies s rows further down (the whole rows of s are cropped):
    _core.tile_shifts rows [row, col, dv - (s - round(s)), du, corr] of the tiles that correlate, or None when the
    frames share too few rows."""
    rows = a.shape[0]
    si = int(np.rint(s))
    n = rows - abs(si)
    if n < tile + 16:
        return None
    a, b = (a[:n], b[si:]) if si >= 0 else (a[-si:], b[:n])
    t = _core.tile_shifts(np.ascontiguousarray(a), np.ascontiguousarray(b), tile, tile // 2, 4, 6)
    t = t[np.isfinite(t[:, 2]) & (t[:, 4] > 0.3)]
    t[:, 2] -= s - si
    return t


def axis_from_pairs(line, angles_deg, z_pix, axis, shape, n_pairs=48, tile=240, progress=None):
    """Refinement of the rotation axis from a scan's own opposite views.

    A projection and the one half a turn on are mirror images about the rotation axis (displaced along the rows by
    the travel in between). For n_pairs pairs spread over the scan the high-passed line integrals of the one are
    registered on the mirrored other on tiles (_core.tile_shifts) over the columns that have a mirror partner; the
    median lateral displacement is twice the error of `axis` (the scan's columns). The axis search works on a grid
    of a quarter pixel; this measures the remainder.

    line(p): the (rows, cols) float32 line integrals of projection p; z_pix: the rows the sample has travelled.
    Returns a dict: refinement (pixels to add to the axis; 0.0 when nothing could be measured or the result is
    implausible), measured, scatter, n_pairs, range (the least and largest refinement along the scan, over runs of
    pairs), reason, pairs [[p, q, tiles, refinement of the pair], ...]."""
    out = dict(refinement=0.0, measured=None, scatter=None, n_pairs=0, range=None, reason=None, pairs=[])
    rows, cols = shape
    flip = int(math.floor(2 * axis))               # column c of the mirrored frame = column flip - c of the frame
    frac = 2 * axis - flip
    c0, c1 = max(0, flip - (cols - 1)), min(cols - 1, flip)     # columns with a mirror partner on the detector
    if c1 - c0 + 1 < tile + 40:
        return dict(out, reason=f"rotation axis at column {axis:.1f}: only {max(c1 - c0 + 1, 0)} mirrored columns")
    ang = np.asarray(angles_deg, np.float64)
    z = np.asarray(z_pix, np.float64)
    picks = np.unique(np.linspace(0, len(ang) - 1, n_pairs + 2)[1:-1].round().astype(int))
    pairs = []
    for p in picks:
        p = int(p)
        q = _partner(ang, p, 180.0)
        if q < 0:
            q = _partner(ang, p, -180.0)
        if q >= 0:
            a = _high_pass(line(p)[:, c0:c1 + 1])
            b = _high_pass(line(q)[:, flip - c1:flip - c0 + 1][:, ::-1])
            t = _row_registered(a, b, z[p] - z[q], tile)
            if t is not None and len(t) >= 6:
                # a feature at column x of the frame is at flip - (2 axis_true - x) of the mirrored one:
                # du = flip - 2 axis_true
                pairs.append([p, q, int(len(t)), float(-0.5 * (np.median(t[:, 3]) + frac))])
        if progress:
            progress(1)
    out["pairs"] = pairs
    if len(pairs) < max(8, len(picks) // 3):
        return dict(out, reason=f"only {len(pairs)} of {len(picks)} pairs of opposite views register")
    d = np.array([r[3] for r in pairs])
    measured = float(np.median(d))
    scatter = float(1.4826 * np.median(np.abs(d - measured)))
    runs = [float(np.median(c)) for c in np.array_split(d, min(6, len(d) // 4))]
    out.update(measured=measured, scatter=scatter, n_pairs=len(pairs), range=[min(runs), max(runs)])
    if abs(measured) > AXIS_MAX_REFINEMENT:
        return dict(out, reason="implausible refinement")
    out["refinement"] = measured
    return out


def _scan_rows(scan, z_scale):
    """rows the sample has travelled at each projection (recorded z x the pitch scale)"""
    return (np.asarray(scan.z_translation_m, np.float64) - scan.z_translation_m[0]) / scan.pixel_size_m * z_scale


def _add_axis(cal, main, arrays, log, progress):
    """refine the rotation axis from the main scan's opposite views and record it in the calibration"""
    ext = Extractor(main, arrays["main_dark"], arrays["main_flats"], arrays["weight_map"])
    with _bar(progress, 48, "rotation axis (opposite views)", "pair") as bar:
        res = axis_from_pairs(lambda p: _line_integrals(ext, p), main.unwrapped_angles_deg,
                              _scan_rows(main, cal.z_translation_scale or 1.0), cal.cor_main_detector, main.frame_shape,
                              progress=bar.update)
    cal.axis_refinement = float(res["refinement"])
    cal.details["axis"] = res
    if res["measured"] is None:
        log(f"rotation axis: not refined ({res['reason']})")
    else:
        log(f"rotation axis: the opposite views of {main.name} put it {res['measured']:+.3f} px from the search's "
            f"{cal.cor_main_detector:.3f} ({res['n_pairs']} pairs, scatter {res['scatter']:.3f}; "
            f"{res['range'][0]:+.3f} .. {res['range'][1]:+.3f} along the scan)"
            + ("" if res["reason"] is None else f": {res['reason']}, not applied"))


def stationary_from_pairs(line, angles_deg, z_pix, shape, n_pairs=24, tile=240, progress=None):
    """Where the turns recorded with the z stage at rest lie against the helix, from a scan's own frames.

    A scan may begin and end with a turn at rest. Its recorded z joins the helix's by what the motor record says,
    which need not be where the sample was while the stage started or stopped. A projection of such a turn and the
    one a whole turn away in the helix see the same rays, displaced along the rows by the travel in between: the
    high-passed line integrals of n_pairs such pairs per turn are registered on tiles (_core.tile_shifts) and the
    median residual against the recorded travel is the turn's offset: rows to add to the vertical position of its
    projections. Partners within STATIONARY_RAMP projections of the helix's ends are not used.

    line(p): the (rows, cols) float32 line integrals of projection p; z_pix: the rows the sample has travelled, as
    recorded (x the pitch scale).
    Returns a dict: segments [[first projection, end, rows], ...] (only offsets that are significant, at least
    STATIONARY_MIN_ROWS and whose pairs scatter by less than STATIONARY_MAX_SCATTER), turns [dict(first, end,
    offset, stderr, scatter, n_pairs, applied)], pairs [[p, q, lag, tiles, residual], ...], reason."""
    out = dict(segments=[], turns=[], pairs=[], reason=None)
    rows, cols = shape
    ang = np.asarray(angles_deg, np.float64)
    z = np.asarray(z_pix, np.float64)
    n = len(ang)
    moving = np.flatnonzero(np.abs(np.diff(z)) > 0.01)
    if len(moving) < 2 * STATIONARY_RAMP + 8:
        return dict(out, reason="the scan has no helix")
    h0, h1 = int(moving[0]) + 1, int(moving[-1]) + 1          # the helix: projections [h0, h1)
    step = float(np.median(np.abs(np.diff(ang))))
    per_turn = 360.0 / step
    stretches = [(0, h0, 360.0), (h1, n, -360.0)]             # (first, end, the turn to the partner in the helix)
    for s0, s1, turn in stretches:
        if s1 - s0 < 0.25 * per_turn:
            continue
        cand = []
        for p in range(s0, s1):
            q = _partner(ang, p, turn)
            if q < h0 + STATIONARY_RAMP or q >= h1 - STATIONARY_RAMP:
                continue
            if rows - abs(z[p] - z[q]) >= tile + 60:
                cand.append((p, q))
        if len(cand) < 8:
            continue
        picks = [cand[i] for i in np.unique(np.linspace(0, len(cand) - 1, n_pairs).round().astype(int))]
        res = []
        for p, q in picks:
            t = _row_registered(_high_pass(line(p)), _high_pass(line(q)), z[p] - z[q], tile)
            if t is not None and len(t) >= 6:
                # what is at row v of p is at v + (z[p] - z[q]) + dv of q: the turn really lies dv further
                res.append([p, q, float(z[p] - z[q]), int(len(t)), float(np.median(t[:, 2]))])
            if progress:
                progress(1)
        out["pairs"] += res
        if len(res) < 6:
            continue
        d = np.array([r[4] for r in res])
        off = float(np.median(d))
        scatter = float(1.4826 * np.median(np.abs(d - off)))
        stderr = scatter / math.sqrt(len(d))
        applied = bool(abs(off) >= max(STATIONARY_MIN_ROWS, 3 * stderr) and scatter <= STATIONARY_MAX_SCATTER)
        out["turns"].append(dict(first=s0, end=s1, offset=off, stderr=stderr, scatter=scatter, n_pairs=len(res), applied=applied))
        if applied:
            out["segments"].append([s0, s1, off])
    if not out["turns"]:
        out["reason"] = "no turn at rest that shares rows with the helix a turn away"
    return out


def _add_stationary(cal, main, arrays, log, progress):
    """place the main scan's stationary turns against its helix and record it in the calibration"""
    ext = Extractor(main, arrays["main_dark"], arrays["main_flats"], arrays["weight_map"])
    with _bar(progress, 48, "stationary turns against the helix", "pair") as bar:
        res = stationary_from_pairs(lambda p: _line_integrals(ext, p), main.unwrapped_angles_deg,
                                    _scan_rows(main, cal.z_translation_scale or 1.0), main.frame_shape, progress=bar.update)
    cal.stationary_registration = dict(segments=res["segments"]) if res["segments"] else {}
    cal.details["stationary"] = res
    if not res["turns"]:
        log(f"stationary turns: none measured ({res['reason']})")
    for t in res["turns"]:
        log(f"stationary turn (projections {t['first']} .. {t['end'] - 1}) of {main.name}: {t['offset']:+.3f} rows from "
            f"where the recorded z puts it against the helix ({t['n_pairs']} pairs, scatter {t['scatter']:.3f})"
            + ("" if t["applied"] else ": not applied"))


def stationary_arrays(reg, n_proj):
    """Calibration.stationary_registration -> rows to add to the vertical position of each of the main scan's
    projections (float32 (n_proj,)), or None when there is nothing to apply"""
    if not reg or not reg.get("segments"):
        return None
    out = np.zeros(n_proj, np.float32)
    for s0, s1, off in reg["segments"]:
        out[int(s0):int(s1)] = off
    return out


def _current_names(cached: dict) -> dict:
    """a cached calibration under today's names (until 2026-11-11 the detector's blur field was "detector_smear",
    its details "smear" with "max_smear_px")"""
    d = dict(cached)
    if "detector_smear" in d:
        d.setdefault("detector_blur_field", d.pop("detector_smear"))
    details = dict(d.get("details") or {})
    if "smear" in details:
        blur = details.pop("smear")
        if isinstance(blur, dict) and "max_smear_px" in blur:
            blur = {("max_blur_px" if k == "max_smear_px" else k): v for k, v in blur.items()}
        details.setdefault("detector_blur", blur)
        d["details"] = details
    return d


def _add_detector_blur(cal, main, arrays, log):
    """measure the detector's blur field from the main scan's flats and record it in the calibration"""
    res = measure_detector_blur(main, arrays["main_dark"], log=log)
    cal.detector_blur_field = [float(v) for v in res["field"]]
    cal.details["detector_blur"] = res


# ============================================================================ detector blur from the flats' noise
BLUR_TILE = 128
BLUR_MIN_FLATS = 20      # fewer flats in the series do not carry the noise spectra
BLUR_MIN_PX = 0.5        # a field whose blur nowhere reaches this many pixels is not used
BLUR_MIN_GAIN = 0.97     # ... nor one that does not bring the residual of the noise spectra below this share


def flat_noise_spectra(scan: BlissScan, dark, n_flats=200, tile=BLUR_TILE):
    """Noise power spectra of the scan's first flat series on a lattice of tiles (_core.tile_power_spectra): of the
    differences of consecutive flats, each less the dark and scaled to its mean (what is the same in both, the beam's
    profile and the fixed pattern, drops out; the noise stays). Returns (power (ny, nx, tile, tile / 2 + 1), tile
    origins ys, xs, flats used)."""
    refs = scan.flats[0]
    n = min(n_flats, len(refs)) // 2 * 2
    if n < 2:
        raise ValueError("the detector's blur needs at least two flats")
    rows, cols = scan.frame_shape
    ys = list(range((rows % tile) // 2, rows - tile + 1, tile))
    xs = list(range((cols % tile) // 2, cols - tile + 1, tile))
    dark = np.asarray(dark, np.float32)
    power = np.zeros((len(ys), len(xs), tile, tile // 2 + 1))
    for k0 in range(0, n, 40):
        fl = scan.read_frames(refs, slice(k0, min(n, k0 + 40))).astype(np.float32) - dark
        fl /= fl.mean((1, 2), keepdims=True)
        power += _core.tile_power_spectra(np.ascontiguousarray(fl[0::2] - fl[1::2]), tile, ys, xs)
    return power, ys, xs, n


def measure_detector_blur(scan: BlissScan, dark, n_flats=200, log=print):
    """The detector's blur field of a scan, from the noise of its flats (_core.detector_blur_field_fit: calib.hpp). The field is kept
    only if it explains the noise spectra (residual below BLUR_MIN_GAIN of the residual without it) and reaches
    BLUR_MIN_PX somewhere on the detector; otherwise "field" is [] and nothing changes in the reconstruction."""
    t0 = time.time()
    rows, cols = scan.frame_shape
    n_have = min(n_flats, len(scan.flats[0])) if scan.flats else 0
    if n_have < BLUR_MIN_FLATS or (rows // BLUR_TILE) * (cols // BLUR_TILE) < 6:
        reason = (f"{n_have} flats in the first series" if n_have < BLUR_MIN_FLATS else
                  f"a detector of {rows} x {cols} holds too few tiles of {BLUR_TILE}")
        log(f"detector blur: not measured ({reason})")
        return dict(field=[], measured=None, reason=reason, scan=scan.name)
    power, ys, xs, n = flat_noise_spectra(scan, dark, n_flats)
    field, rms_none, rms_fit = _core.detector_blur_field_fit(power, ys, xs, rows, cols)
    th = np.array(field)
    try:
        zero = (-np.linalg.solve(th[:4].reshape(2, 2), th[4:]) * 1000 + [0.5 * (cols - 1), 0.5 * (rows - 1)]).tolist()
    except np.linalg.LinAlgError:
        zero = [float("nan"), float("nan")]
    res = dict(field=[float(v) for v in field], rms_none=float(rms_none), rms_fit=float(rms_fit), zero=zero,
               tiles=[len(ys), len(xs)])
    corners = [(x, y) for x in (-0.5 * cols / 1000, 0.5 * cols / 1000) for y in (-0.5 * rows / 1000, 0.5 * rows / 1000)]
    reach = max(abs(th[0] * x + th[1] * y + th[4]) for x, y in corners)
    res.update(n_flats=int(n), max_blur_px=float(reach), scan=scan.name)
    ok = res["rms_fit"] < BLUR_MIN_GAIN * res["rms_none"] and reach >= BLUR_MIN_PX
    res["measured"] = res["field"]
    if not ok:
        res["field"] = []
    log(f"detector blur from the noise of {n} flats of {scan.name}: "
        + (f"{reach:.1f} px along the columns at the worst corner, none at column {res['zero'][0]:.0f} "
           f"(residual of the noise spectra {res['rms_none']:.3f} -> {res['rms_fit']:.3f})" if ok else
           f"nothing significant (residual {res['rms_none']:.3f} -> {res['rms_fit']:.3f}, reach {reach:.2f} px)")
        + f"  [{time.time() - t0:.0f}s]")
    return res


def _scan_weights(wm, main, cal):
    """per-scan weight maps: the common weight map x each scan's horizontal apodisation"""
    W = main.frame_shape[1]
    out = {}
    for tag, name in (("main", cal.main), ("companion", cal.companion)):
        a = cal.apodisation[name]
        out[f"{tag}_weight"] = (wm * _core.apodisation_mask(W, a[0], a[1], 1e-8)[None, :]).astype(np.float32)
    return out
