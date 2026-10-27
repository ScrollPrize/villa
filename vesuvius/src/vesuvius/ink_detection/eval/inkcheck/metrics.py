"""Band-pass filtering and pixel / structure-level agreement metrics."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def masked_lowpass(a: np.ndarray, m: np.ndarray, sigma: float) -> np.ndarray:
    """Normalised (masked) Gaussian convolution: G(a*m)/G(m)."""
    if sigma <= 0:
        return np.where(m, a, 0).astype(np.float32)
    mf = m.astype(np.float32)
    num = ndi.gaussian_filter(np.where(m, a, 0).astype(np.float32), sigma, mode="constant")
    den = ndi.gaussian_filter(mf, sigma, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 1e-3, num / np.maximum(den, 1e-6), 0)
    return out.astype(np.float32)


def bandpass(a: np.ndarray, m: np.ndarray, px_um: float, fine_um: float, coarse_um: float) -> np.ndarray:
    """Masked difference-of-Gaussians keeping structures between ~fine_um and
    ~coarse_um (sigmas given in micrometres). Removes slow brightness trends
    (papyrus density, rendering artefacts, sheet curvature) that inflate global r."""
    f = masked_lowpass(a, m, fine_um / px_um) if fine_um > 0 else np.where(m, a, 0)
    c = masked_lowpass(a, m, coarse_um / px_um)
    return np.where(m, f - c, 0).astype(np.float32)


def otsu_threshold(x: np.ndarray, bins: int = 256) -> float:
    x = x[np.isfinite(x)]
    h, e = np.histogram(x, bins=bins)
    c = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(h)
    w1 = w0[-1] - w0
    s0 = np.cumsum(h * c)
    m0 = s0 / np.maximum(w0, 1)
    m1 = (s0[-1] - s0) / np.maximum(w1, 1)
    var = w0 * w1 * (m0 - m1) ** 2
    return float(c[int(np.argmax(var[:-1]))])


def rank_scores(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties averaged) — compute once, reuse for many AUCs."""
    from scipy.stats import rankdata

    return rankdata(x).astype(np.float64)


def auc_from_ranks(ranks: np.ndarray, pos: np.ndarray) -> float:
    n1 = int(pos.sum())
    n0 = pos.size - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def best_f1(score: np.ndarray, pos: np.ndarray, n_thr: int = 64):
    """Best F1 over thresholds at score quantiles 50%..99.8%. Returns (f1, thr, prec, rec)."""
    qs = np.quantile(score, np.linspace(0.5, 0.998, n_thr))
    P = pos.sum()
    best = (0.0, float(qs[0]), 0.0, 0.0)
    for t in np.unique(qs):
        pr = score >= t
        tp = np.count_nonzero(pr & pos)
        npr = np.count_nonzero(pr)
        if npr == 0 or P == 0:
            continue
        p, r = tp / npr, tp / P
        f = 2 * p * r / (p + r) if p + r > 0 else 0.0
        if f > best[0]:
            best = (float(f), float(t), float(p), float(r))
    return best


def topk_mask(score_img: np.ndarray, m: np.ndarray, k: int) -> np.ndarray:
    """Binary mask of the k highest-scoring valid pixels."""
    out = np.zeros(score_img.shape, bool)
    if k <= 0:
        return out
    v = score_img[m]
    k = min(k, v.size)
    thr = np.partition(v, v.size - k)[v.size - k]
    out[m] = v >= thr
    return out


def stroke_overlap(pred_bin: np.ndarray, ref_bin: np.ndarray, m: np.ndarray, min_area_px: int,
                   cover: float = 0.25, tol_px: int = 1):
    """Structure-level agreement.

    Reference "strokes" = 8-connected components of ref_bin (inside mask) with
    area >= min_area_px. A stroke is *recalled* if >= `cover` of its pixels are
    predicted ink after dilating the prediction by tol_px. Symmetrically, a
    predicted blob is *supported* if >= `cover` of it lies on (dilated) reference
    ink. Returns dict with recall, precision, counts.
    """
    st = np.ones((3, 3), bool)
    pb = pred_bin & m
    rb = ref_bin & m
    pd = ndi.binary_dilation(pb, st, iterations=tol_px) if tol_px > 0 else pb
    rd = ndi.binary_dilation(rb, st, iterations=tol_px) if tol_px > 0 else rb

    def frac_covered(bin_img, other):
        lab, n = ndi.label(bin_img, st)
        if n == 0:
            return np.zeros(0), np.zeros(0)
        area = np.bincount(lab.ravel(), minlength=n + 1)[1:]
        hit = np.bincount(lab.ravel(), weights=other.ravel().astype(np.float64), minlength=n + 1)[1:]
        keep = area >= min_area_px
        return (hit[keep] / area[keep]), area[keep]

    fr, ar = frac_covered(rb, pd)
    fp, ap = frac_covered(pb, rd)
    rec = float((fr >= cover).mean()) if fr.size else float("nan")
    prec = float((fp >= cover).mean()) if fp.size else float("nan")
    # area-weighted variants are less sensitive to many tiny fragments
    rec_w = float(ar[fr >= cover].sum() / ar.sum()) if fr.size else float("nan")
    return {"stroke_recall": rec, "stroke_precision": prec, "stroke_recall_area_weighted": rec_w,
            "n_ref_strokes": int(fr.size), "n_pred_blobs": int(fp.size)}
