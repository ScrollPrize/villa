"""Write a per-node winding displacement field into the gap logits.

For winding k the gap g_(k-1)(theta) is solved per z row so that the winding's
radius follows R_k + u g (u > 0: the gap outside, u < 0: the gap inside) on
four samples per column plus every node. Rows are independent; three
Gauss-Seidel sweeps over the strip segments make each winding's last node
exact as well. Built for fits where many gap cells sit at the minimum gap
(the gap map is flat there and a gap cannot be inverted to a logit):
- inside a re-solved row-segment only the columns next to a sample whose
  target gap changes are free; every other column keeps its stored logit;
- free columns start node-exact (the target gap at the node, inverted) and
  are refined by Levenberg-Marquardt with per-row damping, a step being kept
  only if the row's dense residual falls;
- a move whose INTENDED gap falls more than FLOOR_TOL voxels below
  min(old gap, floor) is halved locally (up to eight times, then dropped).
A zero field leaves the logits bit-identical; column 0 stays pinned.
"""


import numpy as np

from . import gapmap as WB

TOL = 1e-9

STEP_CLIP = 0.05  # largest logit step per iteration (~15 units of softplus argument)

LM_ITER = 8  # Levenberg-Marquardt iterations per segment and sweep

FLOOR_TOL = 0.02  # vx: a sample counts as a floor violation (-> shrink) only if its target gap is this far below the floor;


def dgap(gm, Lv):
    x = gm.bias + gm.k * gm.lr * Lv
    return (
        (gm.dr - gm.min_gap) / gm.den * (1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))) * gm.k * gm.lr
    )


def thomas(a, b, c, d):
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
        self.c0, self.c1 = int(idx[s]), int(idx[s + 1])
        if s == 0:
            self.c0 = 1
        self.M = M  # samples [M:] are the integer node positions idx[s] .. idx[s+1]-1


def interp_at(A, sg):
    return A[:, sg.i0] * (1 - sg.f) + A[:, sg.i0 + 1] * sg.f


def _free_mask(chg, sg, n):
    """chg (rows, samples) bool: sample's target differs. -> (rows, n) bool: local column influenced by a changed sample."""
    F = np.zeros((chg.shape[0], n), bool)
    for col, w in ((sg.i0, 1 - sg.f), (sg.i0 + 1, sg.f)):
        ok = (col >= sg.c0) & (col < sg.c0 + n) & (w > 0)
        lc = col[ok] - sg.c0
        rr, mm = np.nonzero(chg[:, ok])
        F[rr, lc[mm]] = True
    return F


def _lm(Lr, Gr, F, sg, gm, n_iter):
    """Levenberg-Marquardt on the free columns of each row; returns the refined rows and the final per-row cost."""
    c0 = sg.c0
    n = F.shape[1]
    R = Lr.shape[0]
    cost = lambda L: ((Gr - gm.gapf(interp_at(L, sg))) ** 2).sum(1)
    mu = np.full(R, 1e-3)
    cur = cost(Lr)
    okA = (sg.i0 >= c0) & (sg.i0 + 1 < c0 + n)
    lcA = sg.i0[okA] - c0
    for it in range(n_iter):
        lv = interp_at(Lr, sg)
        g = gm.gapf(lv)
        J = dgap(gm, lv)
        res = Gr - g
        A = np.zeros((R, n))
        Bd = np.zeros((R, n))
        Cd = np.zeros((R, n))
        D = np.zeros((R, n))
        for col, wgt in ((sg.i0, 1 - sg.f), (sg.i0 + 1, sg.f)):
            ok = (col >= c0) & (col < c0 + n)
            lc = col[ok] - c0
            Jw = J[:, ok] * wgt[ok]
            np.add.at(Bd.T, lc, (Jw**2).T)
            np.add.at(D.T, lc, (Jw * res[:, ok]).T)
        off = (J[:, okA] ** 2) * ((1 - sg.f[okA]) * sg.f[okA])
        np.add.at(Cd.T, lcA, off.T)
        np.add.at(A.T, lcA + 1, off.T)
        # fixed columns: identity rows, no coupling
        A[:, 1:] *= F[:, 1:] & F[:, :-1]
        A[:, 0] = 0
        Cd[:, :-1] *= F[:, :-1] & F[:, 1:]
        Cd[:, -1] = 0
        tiny = 1e-9 * (Bd.max(1, keepdims=True) + 1e-30)
        Bdd = Bd + mu[:, None] * (Bd + tiny)
        Bdd = np.where(F, Bdd, 1.0)
        D = np.where(F, D, 0.0)
        dx = thomas(A, Bdd, Cd, D)
        dx = np.clip(dx, -STEP_CLIP, STEP_CLIP)  # at most ~15 units of softplus argument per step
        Ltry = Lr.copy()
        Ltry[:, c0 : c0 + n] += dx
        new = cost(Ltry)
        acc = new < cur
        Lr[acc] = Ltry[acc]
        cur = np.where(acc, new, cur)
        mu = np.where(acc, np.maximum(mu / 3, 1e-9), mu * 10)
    return Lr, cur


def solve_rows(Lold, gm, idx, dr, U, gfloor=3.0, sweeps=3, lm_iter=None):
    lm_iter = lm_iter or LM_ITER
    Wc, num, _j, _th = WB.strip_geometry(idx)
    nw = len(num)
    L = Lold.copy()
    L[:, 0] = 0.0
    out = L.copy()
    segs = [Seg(idx, s, Wc) for s in range(nw)]
    R = Lold.shape[0]
    tgt = []
    for s, sg in enumerate(segs):
        k = s + 1
        Rold = WB.radii_at(L, gm, idx, dr, k, sg.th)
        g_in = gm.gapf(interp_at(L, sg))
        g_out = gm.gapf(WB.interp_cols(L, idx[k] + sg.th * num[k], Wc)) if k < nw else g_in
        ub = interp_at(U, sg)
        Rt = Rold + np.where(ub > 0, ub * g_out, ub * g_in)
        # intended radius of the inner sheet k-1 at the same angles (for the floor test on the INTENT, not on writer residuals)
        if k > 1:
            coord = idx[s - 1] + sg.th * num[s - 1]
            ub_in = WB.interp_cols(U, coord, Wc)
            gi_in = gm.gapf(WB.interp_cols(L, coord, Wc))
            Rt_in = WB.radii_at(L, gm, idx, dr, k - 1, sg.th) + np.where(
                ub_in > 0, ub_in * g_in, ub_in * gi_in
            )
        else:
            Rt_in = dr * sg.th
        tgt.append((Rt, np.minimum(g_in, gfloor), Rt - Rt_in))
    viol = np.zeros(Lold.shape, bool)
    solved = np.zeros((R, nw), bool)
    nviol = 0
    nfree = 0
    hist = []
    for sw in range(sweeps):
        maxchg = 0.0
        for s, sg in enumerate(segs):
            k = s + 1
            Rt, fl, Gint = tgt[s]
            Rin = WB.radii_at(out, gm, idx, dr, k - 1, sg.th) if k > 1 else dr * sg.th
            G = Rt - Rin
            if sw == sweeps - 1:
                low = Gint < fl - FLOOR_TOL
                if low.any():
                    rr, mm = np.nonzero(low)
                    viol[rr, sg.i0[mm]] = True
                    viol[rr, np.minimum(sg.i0[mm] + 1, Wc - 1)] = True
                    nviol += int(low.sum())
            G = np.maximum(G, fl)
            gcur = gm.gapf(interp_at(out, sg))
            chg = np.abs(G - gcur) > TOL
            act = chg.any(1)
            if not act.any():
                continue
            solved[act, s] = True
            c0, c1 = sg.c0, sg.c1
            n = c1 - c0
            Lr = out[act].copy()
            Gr = G[act]
            F = _free_mask(chg[act], sg, n)
            if sw == 0:
                nfree += int(F.sum())
                # node-exact start on the free columns: gap AT node column c = target at that node sample
                Gn = Gr[:, sg.M :]
                cols = np.arange(int(idx[s]), int(idx[s + 1]))
                okc = cols >= c0
                lc = cols[okc] - c0
                init = gm.gapinv(np.maximum(Gn[:, okc], gm.min_gap + 1e-9))
                nodechg = (
                    np.abs(Gn[:, okc] - gcur[act][:, sg.M :][:, okc]) > TOL
                )  # node gap unchanged -> keep stored logit
                blk = Lr[:, c0:c1]
                blk[:, lc] = np.where(F[:, lc] & nodechg, init, blk[:, lc])
                Lr[:, c0:c1] = blk
            Lr, _ = _lm(Lr, Gr, F, sg, gm, lm_iter)
            maxchg = max(maxchg, float(np.abs(Lr[:, c0:c1] - out[act, c0:c1]).max()))
            out[act, c0:c1] = Lr[:, c0:c1]
        hist.append({"sweep": sw, "max_logit_change": maxchg})
    out[:, 0] = Lold[:, 0]
    return (
        out,
        {
            "sweeps": hist,
            "floor_violations_samples": nviol,
            "row_segments_solved": int(solved.sum()),
            "free_cells_sweep0": nfree,
        },
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
        rr, cc = np.nonzero(viol)
        s = j[cc]
        ok = s >= 1
        p = (idx[s[ok] - 1] + th[cc[ok]] * num[s[ok] - 1]) * (Wc - 1) / Wc
        c0 = np.floor(p).astype(int)
        cols[rr[ok], c0] = True
        cols[rr[ok], np.minimum(c0 + 1, Wc - 1)] = True
        U[cols] = U[cols] * (0.5 if it < max_shrink else 0.0)
        shr.append({"iter": it, "rows": len(rows), "nodes_scaled": int(cols.sum())})
        Ln, inf2, v2 = solve_rows(Lold_rows[rows], gm, idx, dr, U[rows], gfloor, sweeps)
        Lnew[rows] = Ln
        viol = np.zeros_like(viol)
        viol[rows] = v2
        info["floor_violations_samples_final"] = inf2["floor_violations_samples"]
    info["shrink"] = shr
    info["floor_violations_left"] = int(viol.sum())
    return Lnew, U, info
