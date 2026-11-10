"""Calibration of an extended-field-of-view helical scan pair from the raw Bliss data.

Reproduces the reference (nabu-helical / nabuxx) calibration:
  * darks / flats reduction (nx2etf), weight map + horizontal apodisation (prepare_weights_double),
  * rotation axis of the main (HA) scan (nabuxx-calibrate extract_cor),
  * HA -> TA overlap vector (nabuxx-calibrate get_overlap),
and derives the placement of both scans in the merged frame and the rotation axis there.
Heavy kernels are C++ (helirec._core); the bookkeeping is numpy.
"""
from __future__ import annotations

import json
import math
import os
import time
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


def _mean_frames(src: FrameSource, transform=None, batch=40) -> np.ndarray:
    """numpy.mean(stack.astype(float32), axis=0) semantics: sequential float32 sum, then / n."""
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
    return acc / n


def reduce_darks_flats(scan: BlissScan):
    """dark (rows, cols) and flats (n_series, rows, cols) exactly as nx2etf stores them."""
    dark = _mean_frames(FrameSource(scan.darks, scan.frame_shape))
    flats = []
    for refs, ctl in zip(scan.flats, scan.flats_control_A):
        scale = np.float32(NOMINAL_CURRENT_A) / np.asarray(ctl, np.float32)

        def tr(F, i, scale=scale):
            return dark + (F - dark) * scale[i]
        flats.append(_mean_frames(FrameSource(refs, scan.frame_shape), tr))
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

    def extract(self, p0, p1, f, zs0, zs1, zf0, zf1, S):
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
             sigma_wide=10.0, sigma_narrow=0.4, log=print):
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
    for r in (req, req180):
        s, w = ext.extract(*[np.asarray(r[k]) for k in ("p0", "p1", "f", "zs0", "zs1", "zf0", "zf1")], S=1)
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
                 radius_x=40, radius_z=10, n_slices=80, low_pass=0.4, high_pass=10.0, log=print):
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
    for ext, r in zip((ext_main, ext_comp), reqs):
        stripes.append(ext.extract(*[np.asarray(r[k]) for k in ("p0", "p1", "f", "zs0", "zs1", "zf0", "zf1")], S=S))
    (img0, w0), (img1, w1) = _filter_overlap([stripes[0][0], stripes[1][0]], [stripes[0][1], stripes[1][1]],
                                             low_pass, high_pass)
    res = _core.find_offset(img0, w0, img1, w1, 0, int(init_dx), radius_z, radius_x)
    log(f"  overlap: {len(slices)} stripes x {S} rows, integer {res['best_int']} -> sub-pixel {res['best_sub']}")
    res["cross_check"] = overlap_cross_check(img0, w0, img1, w1, res, log=log)
    res["slices"] = slices
    return res


# ============================================================================ top level
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
              vector_override=None):
    """Full calibration from raw data. Returns (Calibration, arrays dict).

    cache_dir: the darks/flats reduction and the *measured* calibration are cached there, keyed by
    acquisition_fingerprint(); a cache of another acquisition or calibration version is ignored and replaced.
    cor_override / vector_override replace the measured values for this call only (Calibration.source =
    "override"); they are never written to the cache, so a later run without overrides gets the measured values."""
    fp = acquisition_fingerprint(main, companion)
    override = cor_override is not None or vector_override is not None
    cached_cal, arrays = _load_cache(cache_dir, fp, log) if cache_dir else (None, None)
    if cached_cal is not None and not override:
        d = dict(cached_cal)
        d["placement"] = {k: ScanPlacement(**v) for k, v in d["placement"].items()}
        d["vector"] = tuple(d["vector"])
        d["apodisation"] = {k: tuple(v) for k, v in d["apodisation"].items()}
        cal = Calibration(**d)
        if "cor" in cal.details:
            check_cor_drift(cal.details["cor"]["cor_begin_end"], log=log)
        log(f"calibration loaded from {cache_dir}")
        return cal, {**arrays, **_scan_weights(arrays["weight_map"], main, cal)}
    t0 = time.time()
    if arrays is None:
        arrays = {}
        for tag, sc in (("main", main), ("companion", companion)):
            log(f"[{tag}] reducing darks/flats of {sc.name} ...")
            dark, flats = reduce_darks_flats(sc)
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
        cr = find_cor(e["m"], e["sm"], main.estimated_cor_from_motor_pix, log=log)
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
        ov = find_overlap(e["m"], e["sm"], e["c"], e["sc"], init_dx, log=log)
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
    log(f"calibration ({cal.source}) done in {time.time() - t0:.0f}s: COR {cor_main:.3f} (detector), vector {vector}, "
        f"rotation axis {rot_center:.3f} in a {width}-px merged frame")
    if cache_dir and not override:
        def save_json(t):
            with open(t, "w") as fh:
                fh.write(cal.to_json())
        _atomic_write(os.path.join(cache_dir, "calibration.json"), save_json)
    return cal, {**arrays, **_scan_weights(wm, main, cal)}


def _scan_weights(wm, main, cal):
    """per-scan weight maps: the common weight map x each scan's horizontal apodisation"""
    W = main.frame_shape[1]
    out = {}
    for tag, name in (("main", cal.main), ("companion", cal.companion)):
        a = cal.apodisation[name]
        out[f"{tag}_weight"] = (wm * _core.apodisation_mask(W, a[0], a[1], 1e-8)[None, :]).astype(np.float32)
    return out
