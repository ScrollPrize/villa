"""CT intensity profiles across each fitted sheet at every node of the gap-logit grid.

Geometry: sheet k (integer spiral winding W = k) owns the columns of strip
segment k - 1 of the gap logits; node (row r, column c) describes that sheet
at (theta(c), z(r)). Around each node a fine lattice at 1/3-cell steps is
mapped into the CT through the checkpoint's own inverse transform at
W = k + t, t in {-1, -0.5, 0, 0.5, 1}, and sampled as a 41-sample profile
in u in [-1, 1] windings (piecewise linear between those knots). The box
kernel averages the 3 x 3 lattice points around the node.
"""

import time

import numpy as np

TK = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])  # knots in windings about the sheet
U = np.linspace(-1, 1, 41)  # profile samples
NU = 41
_K = np.array([1, 1, 1.0]) / 3
BOX = np.outer(_K, _K)  # 3 x 3 lattice points around each node, equal weights

G = {}  # transform, geometry and CT store; set by stage._install, inherited by forked workers


def trilinear(X):
    """X (m,3) zyx float64 in CT L2 vx -> values (float32); z outside the resident rows -> nan (counted)."""
    f = np.floor(X).astype(np.int64)
    t = (X - f).astype(np.float32)
    v = np.zeros(len(X), np.float32)
    for dz in (0, 1):
        wz = t[:, 0] if dz else 1 - t[:, 0]
        for dy in (0, 1):
            wy = t[:, 1] if dy else 1 - t[:, 1]
            for dx in (0, 1):
                wx = t[:, 2] if dx else 1 - t[:, 2]
                v += wz * wy * wx * sample_int(f[:, 0] + dz, f[:, 1] + dy, f[:, 2] + dx)
    return v


def inv(tag, z, th, w):
    import torch

    tr, dr = G["tr"][tag], G["dr"][tag]
    rr = (w + th / (2 * np.pi)) * dr
    out = []
    for i in range(0, len(z), 200000):
        sp = torch.tensor(
            np.stack(
                [
                    z[i : i + 200000],
                    rr[i : i + 200000] * np.sin(th[i : i + 200000]),
                    rr[i : i + 200000] * np.cos(th[i : i + 200000]),
                ],
                1,
            ),
            dtype=torch.float32,
        )
        with torch.inference_mode():
            out.append(tr.inv(sp).numpy().astype(np.float64))
    return np.concatenate(out) if out else np.zeros((0, 3))


def profiles(Xk):
    """Xk (n, 5, 3) knots -> (n, 41) profile, piecewise linear in u between knots."""
    seg = np.clip(np.searchsorted(TK, U, "right") - 1, 0, 3)
    f = (U - TK[seg]) / 0.5
    P = Xk[:, seg, :] * (1 - f)[None, :, None] + Xk[:, seg + 1, :] * f[None, :, None]
    P = P.reshape(-1, 3)
    return trilinear(P).reshape(len(Xk), NU)


def conv_nodes(A, nr, nc):
    """A (Fr, Fc, ...) fine grid (node m at fine index 2+3m) -> (nr, nc, ...) box-weighted stacks (nan-aware)."""
    W2 = BOX
    h = W2.shape[0] // 2
    ok = np.isfinite(A).all(-1) if A.ndim == 3 else np.isfinite(A)
    A0 = np.where(ok[..., None], A, 0.0)
    num = np.zeros((nr, nc) + A.shape[2:])
    den = np.zeros((nr, nc))
    ri = 2 + 3 * np.arange(nr)
    ci = 2 + 3 * np.arange(nc)
    for a in range(-h, h + 1):
        for b in range(-h, h + 1):
            w = W2[a + h, b + h]
            num += w * A0[np.ix_(ri + a, ci + b)]
            den += w * ok[np.ix_(ri + a, ci + b)]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den[..., None]
    out[den < 0.5] = np.nan
    return out.astype(np.float32)


def work(item):
    """Box profile stacks of winding k for gap rows [ra, rb): (tag, k, ra, rb, c0, c1, stacks, nan count, seconds)."""
    tag, k, ra, rb = item
    t0 = time.time()
    g = G["geo"][tag]
    s = k - 1
    idx, num, Wc, Z, zmin, zmax = g["idx"], g["num"], g["Wc"], g["Z"], g["zmin"], g["zmax"]
    c0, c1 = int(idx[s]), int(idx[s + 1])  # node columns of this sheet
    nr, nc = rb - ra, c1 - c0
    fr = ra - 2 / 3 + np.arange(3 * nr + 2) / 3.0  # fine rows: node m at index 2 + 3m
    fc = c0 - 2 / 3 + np.arange(3 * nc + 2) / 3.0
    z = zmin + fr * (zmax - zmin) / (Z - 1)
    th = (fc * Wc / (Wc - 1) - idx[s]) / num[s] * 2 * np.pi  # may leave [0, 2pi) at the margins: the spiral continues
    ZZ = np.repeat(z, len(fc))
    TT = np.tile(th, len(fr))
    n = len(ZZ)
    w = np.repeat(float(k) + TK[None, :], n, 0)
    knots = inv(tag, np.repeat(ZZ, 5), np.repeat(TT, 5), w.ravel()).reshape(n, 5, 3)
    P = profiles(knots)
    nnan = int(np.isnan(P).any(1).sum())
    stacks = conv_nodes(P.reshape((len(fr), len(fc), NU)), nr, nc)
    return tag, k, ra, rb, c0, c1, stacks, nnan, time.time() - t0


def sample_int(zi, yi, xi):
    """Integer voxel lookups -> float32 through the CT store in ``G['ct']``: a z
    outside the volume gives NaN; y or x outside gives 0; an absent chunk gives
    the array's fill value."""
    return G["ct"].sample_int(zi, yi, xi)
