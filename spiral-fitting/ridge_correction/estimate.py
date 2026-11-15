"""Per-node ridge offset estimator: template cross-correlation per sheet.

For each node profile p(u) (u in [-1, 1], 41 samples, windings about the
fitted sheet) the offset u* is the argmax over tau in [-tmax, tmax] of the
normalised cross-correlation of p(u + tau) with the sheet template T(u) on
u in [-win, win], refined parabolically; confidence = the peak NCC, 0 when
the maximum sits on the search edge. The template of a sheet is the mean of
its profiles, re-aligned three times on its confident nodes, with its zero
at its own smoothed peak (kept inside +-0.3 windings, see peak_zero_info).
No labels are used.
"""


import numpy as np

U = np.linspace(-1, 1, 41)
DU = U[1] - U[0]


def shift_matrix(taus, win):
    """M (ntau, nwin, 41): p @ M[i].T = p sampled at uw + taus[i] (linear interp), uw = window grid."""
    uw = np.arange(-win, win + 1e-9, DU)
    M = np.zeros((len(taus), len(uw), len(U)))
    for i, t in enumerate(taus):
        x = (uw + t - U[0]) / DU
        i0 = np.clip(np.floor(x).astype(int), 0, len(U) - 2)
        f = x - i0
        M[i, np.arange(len(uw)), i0] = 1 - f
        M[i, np.arange(len(uw)), i0 + 1] += f
    return M, uw


class Est:
    def __init__(self, tmax=0.40, win=0.55, step=0.01):
        self.taus = np.arange(-tmax, tmax + 1e-9, step)
        self.tmax = tmax
        self.win = win
        self.M, self.uw = shift_matrix(self.taus, win)

    def ncc(self, P, T):
        """P (n, 41), T (nwin,) -> NCC (n, ntau)."""
        Tz = T - T.mean()
        Tz /= np.linalg.norm(Tz) + 1e-12
        out = np.empty((len(P), len(self.taus)), np.float32)
        for i in range(len(self.taus)):
            S = P @ self.M[i].T
            S -= S.mean(1, keepdims=True)
            nrm = np.linalg.norm(S, axis=1)
            nrm[nrm < 1e-9] = np.inf
            out[:, i] = (S @ Tz) / nrm
        return out

    def estimate(self, P, T):
        """-> u* (n,), conf (n,) (nan rows -> u 0, conf 0)."""
        ok = np.isfinite(P).all(1)
        u = np.zeros(len(P))
        c = np.zeros(len(P))
        if not ok.any():
            return u, c
        R = self.ncc(P[ok].astype(np.float64), T)
        j = R.argmax(1)
        r = R[np.arange(len(R)), j]
        jj = np.clip(j, 1, len(self.taus) - 2)
        y0, y1, y2 = (
            R[np.arange(len(R)), jj - 1],
            R[np.arange(len(R)), jj],
            R[np.arange(len(R)), jj + 1],
        )
        den = y0 - 2 * y1 + y2
        off = np.where(np.abs(den) > 1e-9, 0.5 * (y0 - y2) / den, 0.0)
        tau = self.taus[jj] + np.clip(off, -1, 1) * (self.taus[1] - self.taus[0])
        edge = (j == 0) | (j == len(self.taus) - 1)
        r = np.where(edge, 0.0, r)
        # profile coordinate: the ridge sits at u = +tau (p(u + tau) matches T(u) whose peak is at 0)
        u[ok] = tau
        c[ok] = np.maximum(r, 0)
        return u, c


def align_mean_fast(P, u, w, nb=81):
    """binned version: group by u (step 0.01), shift each group's weighted mean once."""
    out = np.zeros(len(U))
    den = np.zeros(len(U))
    b = np.round(u / 0.01).astype(int)
    for bi in np.unique(b):
        m = b == bi
        ww = w[m]
        if ww.sum() <= 0:
            continue
        q = (P[m] * ww[:, None]).sum(0) / ww.sum()
        v = np.interp(U + bi * 0.01, U, q, left=np.nan, right=np.nan)
        f = np.isfinite(v)
        out[f] += ww.sum() * v[f]
        den[f] += ww.sum()
    return out / np.maximum(den, 1e-12)


CENTRE_WINDOW = 0.3  # the template centre is searched within |u| <= CENTRE_WINDOW windings


def peak_zero_info(Tfull, sm=1.0, clip=True):
    """Template centre of the smoothed full-grid template within |u| <= CENTRE_WINDOW.

    Returns (centre used, parabolic vertex, grid maximum, clipped). The centre
    is the parabolic vertex around the grid maximum; with ``clip`` it falls back
    to the grid maximum when the vertex lies more than one grid step from it or
    outside the window."""
    from scipy.ndimage import gaussian_filter1d

    s = gaussian_filter1d(np.nan_to_num(Tfull, nan=np.nanmean(Tfull)), sm)
    m = np.abs(U) <= CENTRE_WINDOW + 1e-9
    ig = np.flatnonzero(m)[s[m].argmax()]
    i = int(np.clip(ig, 1, len(U) - 2))
    y0, y1, y2 = s[i - 1], s[i], s[i + 1]
    den = y0 - 2 * y1 + y2
    z = U[i] + (0.5 * (y0 - y2) / den * DU if abs(den) > 1e-9 else 0.0)
    bad = bool(abs(z - U[ig]) > DU * (1 + 1e-9) or abs(z) > CENTRE_WINDOW + 1e-9)
    return (U[ig] if (clip and bad) else z), float(z), float(U[ig]), bool(clip and bad)


def peak_zero(Tfull, sm=1.0, clip=True):
    """Template centre (windings); see peak_zero_info."""
    return peak_zero_info(Tfull, sm, clip)[0]


def sheet_template(E, P, iters=3, cmin=0.3, maxn=20000, rng=None):
    """P (n, 41) stacks of one sheet -> template on E's window, centred on its own peak; returns (T, history)."""
    rng = rng or np.random.default_rng(0)
    ok = np.flatnonzero(np.isfinite(P).all(1))
    if len(ok) < 50:
        return None, []
    sel = ok if len(ok) <= maxn else rng.choice(ok, maxn, replace=False)
    Ps = P[sel].astype(np.float64)
    Tf = Ps.mean(0)
    hist = []
    for it in range(iters + 1):
        z0, zr, zg, cl = peak_zero_info(Tf)
        T = np.interp(E.uw + z0, U, Tf)
        if it == iters:
            break
        u, c = E.estimate(Ps, T)
        w = (c >= cmin).astype(float)
        hist.append(
            {
                "it": it,
                "z0": float(z0),
                "z0_vertex": zr,
                "z0_grid": zg,
                "clipped": cl,
                "conf_mean": float(c.mean()),
                "share_conf": float(w.mean()),
                "u_sd": float(u[w > 0].std()) if w.any() else None,
            }
        )
        Tf = align_mean_fast(Ps, u, w if w.sum() > 10 else np.ones(len(u)))
    hist.append({"final_z0": float(z0), "z0_vertex": zr, "z0_grid": zg, "clipped": cl})
    return T, hist
