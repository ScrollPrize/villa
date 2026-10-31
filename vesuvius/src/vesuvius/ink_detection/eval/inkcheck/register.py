"""Resampling a reference onto the prediction grid and masked-NCC translation search."""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import fft as sfft


def resample_to_grid(a: np.ndarray, src_um: float, dst_um: float, out_shape: tuple[int, int],
                     origin: tuple[float, float] = (0.0, 0.0), valid: np.ndarray | None = None):
    """Resample image `a` (pixel size src_um) onto a grid of pixel size dst_um.

    `origin` = (y, x) position, in *source* pixels, of the destination pixel (0, 0)'s
    top-left corner. Area-averaging (BOX) is used when downsampling, bilinear when
    upsampling. Returns (resampled, valid_mask) where valid_mask marks destination
    pixels fully covered by in-bounds (and `valid`) source pixels.
    """
    H, W = out_shape
    f = dst_um / src_um  # source pixels per destination pixel
    y0, x0 = origin
    box = (x0, y0, x0 + W * f, y0 + H * f)
    resample = Image.BOX if f >= 1 else Image.BILINEAR
    src_valid = np.ones(a.shape, np.float32) if valid is None else valid.astype(np.float32)
    # pad so that the requested box always lies inside the (padded) source;
    # padded pixels are marked invalid and excluded via the returned mask
    import math
    pt = int(max(1, math.ceil(-y0) + 1))
    pl = int(max(1, math.ceil(-x0) + 1))
    pb = int(max(1, math.ceil(box[3] - a.shape[0]) + 1))
    pr = int(max(1, math.ceil(box[2] - a.shape[1]) + 1))
    ap = np.pad(a.astype(np.float32), ((pt, pb), (pl, pr)))
    vp = np.pad(src_valid, ((pt, pb), (pl, pr)))
    bx = (box[0] + pl, box[1] + pt, box[2] + pl, box[3] + pt)
    out = np.asarray(Image.fromarray(ap, mode="F").resize((W, H), resample, box=bx), np.float32)
    v = np.asarray(Image.fromarray(vp, mode="F").resize((W, H), resample, box=bx), np.float32)
    ok = v > 0.999
    # renormalise partially-valid boxes is unnecessary: they are excluded via `ok`
    return out, ok


def downsample(a: np.ndarray, factor: float, mask: np.ndarray | None = None):
    """Area-downsample a map by `factor` (>=1); mask pixels count as invalid."""
    if factor <= 1.0 + 1e-9:
        return a.copy(), (np.ones(a.shape, bool) if mask is None else mask.copy())
    H = max(1, int(round(a.shape[0] / factor)))
    W = max(1, int(round(a.shape[1] / factor)))
    m = np.ones(a.shape, bool) if mask is None else mask
    num = np.asarray(Image.fromarray(np.where(m, a, 0).astype(np.float32), mode="F").resize((W, H), Image.BOX))
    den = np.asarray(Image.fromarray(m.astype(np.float32), mode="F").resize((W, H), Image.BOX))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0, num / np.maximum(den, 1e-12), 0).astype(np.float32)
    return out, den > 0.5


def masked_ncc_map(a: np.ndarray, ma: np.ndarray, b: np.ndarray, mb: np.ndarray, radius: int,
                   min_overlap: float = 0.5, workers: int = 2):
    """Exact masked Pearson correlation between a(x) and b(x+s) for all integer
    shifts |s_y|,|s_x| <= radius (Padfield-style, via 6 FFT cross-correlations).

    Returns (ncc, overlap) arrays of shape (2R+1, 2R+1); entry [R+dy, R+dx]
    corresponds to comparing a(y, x) with b(y+dy, x+dx). Shifts whose overlap is
    below `min_overlap` x the zero-shift overlap are set to NaN.
    """
    R = int(radius)
    H, W = a.shape
    shape = (H + R, W + R)
    sh = (sfft.next_fast_len(shape[0], real=True), sfft.next_fast_len(shape[1], real=True))
    A = np.where(ma, a, 0).astype(np.float64)
    B = np.where(mb, b, 0).astype(np.float64)
    MA = ma.astype(np.float64)
    MB = mb.astype(np.float64)

    def F(x):
        return sfft.rfft2(x, s=sh, workers=workers)

    def corr(Fx, Fy):  # c(s) = sum_x x(x) y(x+s)
        c = sfft.irfft2(np.conj(Fx) * Fy, s=sh, workers=workers)
        # gather shifts -R..R (negative shifts wrap to the end)
        idy = np.r_[np.arange(sh[0] - R, sh[0]), np.arange(0, R + 1)]
        idx = np.r_[np.arange(sh[1] - R, sh[1]), np.arange(0, R + 1)]
        return c[np.ix_(idy, idx)]

    FA, FA2, FMA = F(A), F(A * A), F(MA)
    FB, FB2, FMB = F(B), F(B * B), F(MB)
    N = corr(FMA, FMB)
    Sa = corr(FA, FMB)
    Sb = corr(FMA, FB)
    Saa = corr(FA2, FMB)
    Sbb = corr(FMA, FB2)
    Sab = corr(FA, FB)
    N = np.round(N)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = Sab - Sa * Sb / N
        va = Saa - Sa * Sa / N
        vb = Sbb - Sb * Sb / N
        r = cov / np.sqrt(np.maximum(va, 1e-12) * np.maximum(vb, 1e-12))
    n0 = N[R, R]
    r[(N < min_overlap * max(n0, 1)) | (N < 100)] = np.nan
    return r.astype(np.float32), N


def best_shift(a, ma, b, mb, radius: int, workers: int = 2):
    """Best integer shift (dy, dx) and its masked Pearson r within +-radius."""
    if radius <= 0:
        m = ma & mb
        return (0, 0), pearson(a, b, m)
    r, _ = masked_ncc_map(a, ma, b, mb, radius, workers=workers)
    if np.all(np.isnan(r)):
        return (0, 0), float("nan")
    k = np.nanargmax(r)
    iy, ix = np.unravel_index(k, r.shape)
    return (int(iy - radius), int(ix - radius)), float(r[iy, ix])


def pearson(a, b, m):
    if m.sum() < 3:
        return float("nan")
    x = a[m].astype(np.float64)
    y = b[m].astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else float("nan")


def shift_image(b: np.ndarray, mb: np.ndarray, dy: int, dx: int):
    """Return b', mb' with b'(y, x) = b(y+dy, x+dx) (zero/invalid outside)."""
    H, W = b.shape
    out = np.zeros_like(b)
    om = np.zeros_like(mb)
    ys, yd = (slice(dy, H), slice(0, H - dy)) if dy >= 0 else (slice(0, H + dy), slice(-dy, H))
    xs, xd = (slice(dx, W), slice(0, W - dx)) if dx >= 0 else (slice(0, W + dx), slice(-dx, W))
    out[yd, xd] = b[ys, xs]
    om[yd, xd] = mb[ys, xs]
    return out, om
