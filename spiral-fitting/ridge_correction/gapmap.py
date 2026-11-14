"""Gap-logit geometry helpers: stored logits <-> gap widths, strip layout and
sheet radii exactly as the gap-expanding transform computes them.
"""

import numpy as np


def interp_cols(A, coord, Wc):
    p = np.clip(coord * (Wc - 1) / Wc, 0, Wc - 1)
    i0 = np.minimum(np.floor(p).astype(np.int64), Wc - 2)
    f = p - i0
    return A[:, i0] * (1 - f) + A[:, i0 + 1] * f


class GapMap:
    """Stored gap logits <-> gap widths (voxels), as the gap-expanding transform maps them."""

    def __init__(self, model, cfg):
        self.dr = float(model.get_dr_per_winding().detach())
        self.min_gap = float(model.gap_min_gap)
        self.bias = float(model.gap_softplus_bias)
        self.k = float(model.gap_softplus_scale)
        self.lr = float(cfg["model_gap_expander_lr_scale"])
        self.den = float(np.log1p(np.exp(-abs(self.bias))) + max(self.bias, 0.0))

    def gaps(self, L):
        e = L.copy()
        e[:, 0] = 0.0
        e = e * self.lr
        x = self.bias + self.k * e
        sp = np.where(x > 30, x, np.log1p(np.exp(np.minimum(x, 30))))
        return self.min_gap + (self.dr - self.min_gap) * sp / self.den

    def gapf(self, L):  # elementwise stored logit -> gap (no pinning)
        x = self.bias + self.k * self.lr * L
        return (
            self.min_gap
            + (self.dr - self.min_gap)
            * np.where(x > 30, x, np.log1p(np.exp(np.minimum(x, 30))))
            / self.den
        )

    def gapinv(self, g):
        ratio = np.maximum((g - self.min_gap) / (self.dr - self.min_gap) * self.den, 1e-300)
        x = np.where(ratio > 30, ratio, np.log(np.expm1(np.minimum(ratio, 30))))
        return (x - self.bias) / self.k / self.lr


def strip_geometry(idx):
    Wc = int(idx[-1])
    num = np.diff(idx)
    coord = np.arange(Wc) * Wc / (Wc - 1)
    j = np.clip(np.searchsorted(idx, coord, "right") - 1, 0, len(num) - 1)
    th = (coord - idx[j]) / num[j]
    return Wc, num, j, th


def radii_at(L, gm, idx, dr, k, th):
    """R_k at fractional turn th for every row, EXACTLY as the transform: bilinear interpolation of the (pinned) LOGITS, then the
    softplus gap map, then the cumulative sum (READ transforms.py get_logits_by_winding / get_transformed_winding_radii)."""
    Wc = int(idx[-1])
    num = np.diff(idx)
    R = np.broadcast_to(dr * th, (L.shape[0], len(th))).copy()
    for jj in range(k):
        R += gm.gapf(interp_cols(L, idx[jj] + th * num[jj], Wc))
    return R
