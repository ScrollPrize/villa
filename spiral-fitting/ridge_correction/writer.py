"""Write a per-node winding displacement field into the gap logits.

For sheet k the gap g_(k-1)(theta) is solved per z row so that the sheet's
radius follows R_k + u g (u > 0: the gap outside, u < 0: the gap inside)
on four samples per column plus every node (Gauss-Newton on the softplus gap
map, tridiagonal normal equations). Three Gauss-Seidel sweeps over the
segments make each sheet's last node exact as well (the inner segment's
interpolation reaches into the next segment's first column). A row-segment
whose target equals its current gap is not solved, so a zero field leaves
the logits bit-identical. A move that would take a gap below
min(old gap, floor) is halved locally (up to eight times, then dropped)
instead of clipped. Column 0 stays pinned.
"""

# Kept in its original form so that it can be compared line by line with the
# implementation the published results were made with; style-only lint
# findings are therefore not fixed here.
# ruff: noqa: C408, RUF059

import numpy as np

from . import gapmap as WB


def dgap(gm, Lv):
    x = gm.bias + gm.k * gm.lr * Lv
    return (
        (gm.dr - gm.min_gap) / gm.den * (1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))) * gm.k * gm.lr
    )


def thomas(a, b, c, d):
    """batched tridiagonal solve over rows: a sub (n), b diag (n), c super (n), d rhs (n); all (rows, n)."""
    n = b.shape[1]
    cp = np.zeros_like(b)
    dp = np.zeros_like(b)
    cp[:, 0] = c[:, 0] / b[:, 0]
    dp[:, 0] = d[:, 0] / b[:, 0]
    for i in range(1, n):
        m = b[:, i] - a[:, i] * cp[:, i - 1]
        cp[:, i] = c[:, i] / m
        dp[:, i] = (d[:, i] - a[:, i] * dp[:, i - 1]) / m
    x = np.zeros_like(b)
    x[:, -1] = dp[:, -1]
    for i in range(n - 2, -1, -1):
        x[:, i] = dp[:, i] - cp[:, i] * x[:, i + 1]
    return x


class Seg:
    """sample geometry of strip segment s."""

    def __init__(self, idx, s, Wc, per=4):
        num = np.diff(idx)
        self.s = s
        pa, pb = idx[s] * (Wc - 1) / Wc, idx[s + 1] * (Wc - 1) / Wc
        M = per * int(num[s])
        p = np.concatenate(
            [
                pa + (pb - pa) * (np.arange(M) + 0.5) / M,
                np.arange(int(idx[s]), int(idx[s + 1])).astype(float),
            ]
        )
        p = np.clip(p, 0, Wc - 1)
        self.p = p
        self.th = (p * Wc / (Wc - 1) - idx[s]) / num[s]
        self.i0 = np.minimum(np.floor(p).astype(int), Wc - 2)
        self.f = p - self.i0
        self.c0, self.c1 = int(idx[s]), int(idx[s + 1])  # own (free) columns [c0, c1)
        if s == 0:
            self.c0 = 1  # column 0 is pinned


def interp_at(A, seg):
    return A[:, seg.i0] * (1 - seg.f) + A[:, seg.i0 + 1] * seg.f


def solve_rows(Lold, gm, idx, dr, U, gfloor=3.0, sweeps=3, gn_iter=4):
    """Lold (rows, Wc) float64 stored logits; U (rows, Wc). Returns Lnew (float64, col 0 = stored), info, viol (rows, Wc) bool."""
    Wc, num, j, th = WB.strip_geometry(idx)
    nw = len(num)
    L = Lold.copy()
    L[:, 0] = 0.0
    out = L.copy()
    segs = [Seg(idx, s, Wc) for s in range(nw)]
    R = Lold.shape[0]
    # targets depend only on the OLD model and U: precompute per segment
    tgt = []
    for s, sg in enumerate(segs):
        k = s + 1
        Rold = WB.radii_at(L, gm, idx, dr, k, sg.th)
        g_in = gm.gapf(interp_at(L, sg))
        g_out = gm.gapf(WB.interp_cols(L, idx[k] + sg.th * num[k], Wc)) if k < nw else g_in
        ub = interp_at(U, sg)
        tgt.append((Rold + np.where(ub > 0, ub * g_out, ub * g_in), np.minimum(g_in, gfloor)))
    viol = np.zeros(Lold.shape, bool)
    solved = np.zeros((R, nw), bool)
    nviol = 0
    hist = []
    for sw in range(sweeps):
        maxchg = 0.0
        for s, sg in enumerate(segs):
            k = s + 1
            Rt, fl = tgt[s]
            Rin = WB.radii_at(out, gm, idx, dr, k - 1, sg.th) if k > 1 else dr * sg.th
            G = Rt - Rin
            if sw == sweeps - 1:
                low = G < fl - 1e-9
                if low.any():
                    rr, mm = np.nonzero(low)
                    viol[rr, sg.i0[mm]] = True
                    viol[rr, np.minimum(sg.i0[mm] + 1, Wc - 1)] = True
                    nviol += int(low.sum())
            G = np.maximum(G, fl)
            gcur = gm.gapf(interp_at(out, sg))
            act = np.abs(G - gcur).max(1) > 1e-9
            if not act.any():
                continue
            solved[act, s] = True
            c0, c1 = sg.c0, sg.c1
            n = c1 - c0
            # sample -> local column weights (free columns only); fixed contributions stay in b
            Lr = out[act]
            Gr = G[act]
            for it in range(gn_iter):
                lv = interp_at(Lr, sg)
                g = gm.gapf(lv)
                J = dgap(gm, lv)
                res = Gr - g
                A = np.zeros((Lr.shape[0], n))
                Bd = np.zeros((Lr.shape[0], n))
                Cd = np.zeros((Lr.shape[0], n))
                D = np.zeros((Lr.shape[0], n))
                for col, wgt in ((sg.i0, 1 - sg.f), (sg.i0 + 1, sg.f)):
                    ok = (col >= c0) & (col < c1)
                    lc = col[ok] - c0
                    w_ = wgt[ok]
                    Jw = J[:, ok] * w_
                    np.add.at(Bd.T, lc, (Jw**2).T)
                    np.add.at(D.T, lc, (Jw * res[:, ok]).T)
                # off-diagonals: samples touching (i0, i0+1) both free
                ok = (sg.i0 >= c0) & (sg.i0 + 1 < c1)
                lc = sg.i0[ok] - c0
                off = (J[:, ok] ** 2) * ((1 - sg.f[ok]) * sg.f[ok])
                np.add.at(Cd.T, lc, off.T)
                np.add.at(A.T, lc + 1, off.T)
                Bd += 1e-12 * (Bd.max(1, keepdims=True) + 1e-30)
                dx = thomas(A, Bd, Cd, D)
                Lr[:, c0:c1] += dx
            old = out[act, c0:c1]
            maxchg = max(maxchg, float(np.abs(Lr[:, c0:c1] - old).max()))
            out[act, c0:c1] = Lr[:, c0:c1]
        hist.append(dict(sweep=sw, max_logit_change=maxchg))
    out[:, 0] = Lold[:, 0]
    return (
        out,
        dict(sweeps=hist, floor_violations_samples=nviol, row_segments_solved=int(solved.sum())),
        viol,
    )


def write_field(Lold_rows, gm, idx, dr, U, gfloor=3.0, sweeps=3, max_shrink=8):
    """shrink-on-violation loop. Returns Lnew (float64), U_eff (the field actually written), info."""
    U = U.copy()
    Wc, num, j, th = WB.strip_geometry(idx)
    Lnew, info, viol = solve_rows(Lold_rows, gm, idx, dr, U, gfloor, sweeps)
    shr = []
    for it in range(max_shrink + 1):
        if not viol.any():
            break
        rows = np.flatnonzero(viol.any(1))
        cols = viol.copy()
        # also the inner sheet's two nodes at the same angle
        rr, cc = np.nonzero(viol)
        s = j[cc]
        ok = s >= 1
        p = (idx[s[ok] - 1] + th[cc[ok]] * num[s[ok] - 1]) * (Wc - 1) / Wc
        c0 = np.floor(p).astype(int)
        cols[rr[ok], c0] = True
        cols[rr[ok], np.minimum(c0 + 1, Wc - 1)] = True
        U[cols] = U[cols] * (0.5 if it < max_shrink else 0.0)
        shr.append(dict(iter=it, rows=len(rows), nodes_scaled=int(cols.sum())))
        Ln, inf2, v2 = solve_rows(Lold_rows[rows], gm, idx, dr, U[rows], gfloor, sweeps)
        Lnew[rows] = Ln
        viol = np.zeros_like(viol)
        viol[rows] = v2
        info["floor_violations_samples_final"] = inf2["floor_violations_samples"]
    info["shrink"] = shr
    info["floor_violations_left"] = int(viol.sum())
    return Lnew, U, info
