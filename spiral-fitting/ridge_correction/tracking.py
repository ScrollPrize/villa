"""Ridge tracking: the default way of turning CT profiles into a move field.

For every node it lists up to three candidate ridge positions (local maxima of
the normalised cross-correlation with the winding's template over a
1.3-winding search range), then chooses among them by quality-guided region
growing from confident anchors, so that a node takes the candidate that agrees
with its already-assigned neighbours; nodes where two candidates fit about
equally well, or none fits, are left unmoved. The move field is the chosen
offset, faded to zero within three nodes of any unassigned node and by the
z taper at the band ends.

Tracking is quality-guided region growing (path-following phase unwrapping); it uses no labels.

Candidates: for each node profile p(u) (41 samples, u in [-1, 1] windings) and the winding's template (the
plain estimator's template, cut to its central +-0.35 winding), R(tau) = NCC of p(u + tau) with the template
for tau in [-0.65, 0.65] (step 0.01); candidates = interior local maxima of R with R >= RMIN, parabolic
refinement, non-max suppression (MINSEP), the best NC by R. The plain estimate (u1, c1) is kept as well.

Grid: the gap expander's node grid restricted to sheets 1..KMAX, i.e. rows r (z) x strip columns c in [idx[0], idx[KMAX]). The strip is
one continuous spiral, so column c and c+1 are arc neighbours also across a sheet boundary. Optional cross-winding edges: node of sheet
k at turn fraction th <-> the nearest node of sheet k+1 / k-1 at the same th and row (weight WX in the prediction).

Why region growing (not BP / MST): it is the standard quality-guided unwrapper; deterministic (heap ordered by node quality, ties by
flat index); never assigns a node without a chain of accepted neighbours back to an anchor; the ambiguity rule is local and explicit;
O(N log N) in pure numpy/heapq. BP with a truncated-quadratic cost would need tuning of a temperature-like weight and can settle on
mixed labelings in loops; MST unwrapping needs a global edge-quality definition (we have node quality only).

Algorithm (params P):
 1. candidates per node (cu, cr) with cr >= CMIN; invalid nodes (mask) have none.
 2. anchors: plain estimate conf c1 >= CA, |u1| <= UA; a candidate within 0.10 of u1 (anchor value = that candidate); no other
    candidate with cr >= cr_anchor - 0.10; and >= 2 of the 4 in-sheet-grid neighbours have a candidate within 0.10 of the anchor value.
 3. heap of frontier nodes keyed by -max(cr). Pop n: prediction p = weighted mean of assigned neighbours' u (4-nbr weight 1, cross
    weight WX). Distances d_j = |cu_j - p|. Accept the nearest if d1 <= DMAX and d2 - d1 >= MARGIN (d2 = inf if alone). Otherwise
    leave it (it is re-pushed when another neighbour gets assigned; final state ambiguous / no-fit). Accepted -> push neighbours.
 4. seams: assigned 4-neighbour pairs with |du| > JMAX -> both unassigned (2 passes).
States: 0 invalid (no candidate), 1 unreached, 2 ambiguous, 3 no-fit, 4 assigned, 5 anchor.
"""

# Kept in its original form so that it can be compared line by line with the
# implementation the published results were made with; style-only lint
# findings are therefore not fixed here.
# ruff: noqa: C408, F841, RUF059

import heapq

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

from .estimate import Est, sheet_template

U41 = np.linspace(-1, 1, 41); DU = U41[1] - U41[0]
TAUS = np.round(np.arange(-0.65, 0.65 + 1e-9, 0.01), 6)
E = Est(tmax=0.40, win=0.55)
HW = 4                                   # T35 = T55[HW:-HW]
UW = E.uw[HW:len(E.uw) - HW]
RMIN, NC, MINSEP = 0.30, 3, 0.30
_G = []
for t in TAUS:
    x = UW + t; ok = (x >= -1 - 1e-9) & (x <= 1 + 1e-9)
    assert ok.all()
    xi = (x[ok] - U41[0]) / DU; i0 = np.clip(np.floor(xi).astype(int), 0, 39); f = xi - i0
    _G.append((ok, i0, f))


def ext_ncc(P, T):
    """P (n, 41) finite, T (len(UW),) -> R (n, len(TAUS)) float32."""
    out = np.empty((len(P), len(TAUS)), np.float32)
    for i, (ok, i0, f) in enumerate(_G):
        S = P[:, i0] * (1 - f) + P[:, i0 + 1] * f
        S = S - S.mean(1, keepdims=True)
        Tz = T[ok] - T[ok].mean(); Tz = Tz / (np.linalg.norm(Tz) + 1e-12)
        nrm = np.linalg.norm(S, axis=1); nrm[nrm < 1e-9] = np.inf
        out[:, i] = (S @ Tz) / nrm
    return out


def candidates(R, rmin=RMIN, nc=NC):
    """R (n, ntau) -> cu (n, nc), cr (n, nc) sorted by cr desc (nan padded)."""
    n, m = R.shape
    lm = np.zeros(R.shape, bool)
    lm[:, 1:-1] = (R[:, 1:-1] >= R[:, :-2]) & (R[:, 1:-1] > R[:, 2:]) & (R[:, 1:-1] >= rmin)
    y0 = np.empty_like(R); y2 = np.empty_like(R); y0[:, 1:] = R[:, :-1]; y0[:, 0] = R[:, 0]; y2[:, :-1] = R[:, 1:]; y2[:, -1] = R[:, -1]
    den = y0 - 2 * R + y2
    off = np.where(np.abs(den) > 1e-9, 0.5 * (y0 - y2) / np.where(np.abs(den) > 1e-9, den, 1), 0.0)
    off = np.clip(off, -0.5, 0.5)
    pos = TAUS[None, :] + off * 0.01
    val = R - 0.25 * (y0 - y2) * off
    score = np.where(lm, val, -np.inf)
    order = np.argsort(-score, axis=1)[:, :2 * nc + 2]
    ar = np.arange(n)[:, None]
    cr0 = score[ar, order]; cu0 = pos[ar, order]
    cr = np.full((n, nc), -np.inf); cu = np.full((n, nc), np.nan); cnt = np.zeros(n, int)
    for j in range(cr0.shape[1]):                     # greedy non-max suppression, best first
        v = cr0[:, j]; p = cu0[:, j]
        keep = np.isfinite(v) & (cnt < nc)
        for q in range(nc):
            keep &= ~(np.abs(cu[:, q] - p) < MINSEP)
        r = np.flatnonzero(keep); cr[r, cnt[r]] = v[r]; cu[r, cnt[r]] = p[r]; cnt[r] += 1
    bad = ~np.isfinite(cr)
    cr = np.where(bad, np.nan, cr); cu = np.where(bad, np.nan, cu)
    return cu.astype(np.float32), cr.astype(np.float32)


def winding_candidates(P):
    """Candidates of one winding's node profiles P (n, 41) float64.

    Returns cu, cr (n, NC) float32 (candidate offsets in windings and their
    correlation, best first, NaN padded) and u1, c1 (n,) float32 (the single
    best offset and its confidence, as the plain estimator gives them)."""
    ok = np.isfinite(P).all(1)
    T, _ = sheet_template(E, P)
    cu = np.full((len(P), NC), np.nan, np.float32)
    cr = np.full((len(P), NC), np.nan, np.float32)
    u1 = np.zeros(len(P), np.float32)
    c1v = np.zeros(len(P), np.float32)
    if T is not None and ok.any():
        R = ext_ncc(P[ok], T[HW:len(T) - HW])
        a, b = candidates(R)
        cu[ok] = a
        cr[ok] = b
        uu, cc = E.estimate(P[ok], T)
        u1[ok] = uu
        c1v[ok] = cc
    return cu, cr, u1, c1v


DEF = dict(CMIN=0.60, CA=0.80, UA=0.25, DMAX=0.20, MARGIN=0.30, JMAX=0.30, WX=0.0)


def cross_cols(idx, kmax):
    """for each column c of sheets 1..kmax: column of the same-angle node in sheet k+1 (up) and k-1 (dn); -1 if none."""
    idx = np.asarray(idx).astype(int); Wc = int(idx[kmax])
    up = -np.ones(Wc, int); dn = -np.ones(Wc, int)
    for k in range(1, kmax + 1):
        c0, c1 = idx[k - 1], idx[k]; n = c1 - c0; th = (np.arange(n) + 0.0) / n
        if k + 1 <= kmax:
            n2 = idx[k + 1] - idx[k]; up[c0:c1] = idx[k] + np.clip(np.round(th * n2).astype(int), 0, n2 - 1)
        if k - 1 >= 1:
            n0 = idx[k - 1] - idx[k - 2]; dn[c0:c1] = idx[k - 2] + np.clip(np.round(th * n0).astype(int), 0, n0 - 1)
    return up, dn


def track(cu, cr, u1, c1, valid, idx, kmax=67, P=None):
    """cu, cr (Nr, Wc, nc); u1, c1 (Nr, Wc); valid (Nr, Wc) bool. Returns dict(u (Nr, Wc) nan where unassigned, r, state, info)."""
    P = dict(DEF, **(P or {}))
    idx = np.asarray(idx).astype(int); W = int(idx[kmax]); c0 = int(idx[0])
    Nr = cu.shape[0]
    CU = np.array(cu[:, :W], np.float64); CR = np.array(cr[:, :W], np.float64)
    ok = valid[:, :W, None] & np.isfinite(CR) & (CR >= P['CMIN'])
    CU = np.where(ok, CU, np.nan); CR = np.where(ok, CR, np.nan)
    has = np.isfinite(CU).any(-1); has[:, :c0] = False
    q = np.where(has, np.nanmax(np.where(np.isfinite(CR), CR, -1), -1), -1.0)
    # anchors
    U1 = np.array(u1[:, :W], np.float64); C1 = np.array(c1[:, :W], np.float64)
    dist = np.abs(CU - U1[..., None]); dist = np.where(np.isfinite(dist), dist, 9.0)
    ja = dist.argmin(-1); da = dist.min(-1)
    av = np.take_along_axis(CU, ja[..., None], -1)[..., 0]; ar_ = np.take_along_axis(CR, ja[..., None], -1)[..., 0]
    other = np.where(np.arange(CU.shape[-1])[None, None, :] == ja[..., None], np.nan, CR)
    oth_max = np.nanmax(np.where(np.isfinite(other), other, -1), -1)
    anc = has & valid[:, :W] & (C1 >= P['CA']) & (np.abs(U1) <= P['UA']) & (da <= 0.10) & (oth_max < ar_ - 0.10)
    # neighbour support for anchors
    def nb_has_near(val):
        cnt = np.zeros(val.shape, int)
        for dr_, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            sh = np.full(CU.shape, np.nan)
            rs = slice(max(dr_, 0), Nr + min(dr_, 0)); rd = slice(max(-dr_, 0), Nr + min(-dr_, 0))
            cs = slice(max(dc, 0), W + min(dc, 0)); cd = slice(max(-dc, 0), W + min(-dc, 0))
            sh[rd, cd] = CU[rs, cs]
            cnt += (np.nanmin(np.where(np.isfinite(sh), np.abs(sh - val[..., None]), 9.0), -1) <= 0.10)
        return cnt
    anc &= nb_has_near(av) >= 2
    u = np.full((Nr, W), np.nan); r = np.full((Nr, W), np.nan); state = np.where(has, 1, 0).astype(np.int8)
    u[anc] = av[anc]; r[anc] = ar_[anc]; state[anc] = 5
    if P['WX'] > 0:
        xup, xdn = cross_cols(idx, kmax)
    uf = u.reshape(-1); rf = r.reshape(-1); stf = state.reshape(-1); qf = q.reshape(-1)
    CUf = CU.reshape(-1, CU.shape[-1]); CRf = CR.reshape(-1, CR.shape[-1])
    N = Nr * W
    heap = []

    def nbrs(n):
        rr, cc = divmod(n, W); out = []
        if rr > 0: out.append((n - W, 1.0))
        if rr < Nr - 1: out.append((n + W, 1.0))
        if cc > c0: out.append((n - 1, 1.0))
        if cc < W - 1: out.append((n + 1, 1.0))
        if P['WX'] > 0:
            if xup[cc] >= 0: out.append((rr * W + xup[cc], P['WX']))
            if xdn[cc] >= 0: out.append((rr * W + xdn[cc], P['WX']))
        return out

    for n in np.flatnonzero(anc.reshape(-1)):
        for m, _ in nbrs(n):
            if stf[m] in (1, 2, 3):
                heapq.heappush(heap, (-qf[m], m))
    npop = 0
    while heap:
        _, n = heapq.heappop(heap); npop += 1
        if stf[n] >= 4 or stf[n] == 0:
            continue
        sw = 0.0; su = 0.0
        nb = nbrs(n)
        for m, wt in nb:
            if stf[m] >= 4:
                sw += wt; su += wt * uf[m]
        if sw <= 0:
            continue
        p = su / sw
        cands = CUf[n]; d = np.abs(cands - p); d = np.where(np.isfinite(d), d, 9.0)
        o = np.argsort(d); d1 = d[o[0]]; d2 = d[o[1]] if len(o) > 1 else 9.0
        if d1 > P['DMAX']:
            stf[n] = 3; continue
        if d2 - d1 < P['MARGIN']:
            stf[n] = 2; continue
        uf[n] = cands[o[0]]; rf[n] = CRf[n, o[0]]; stf[n] = 4
        for m, _ in nb:
            if 1 <= stf[m] <= 3:
                heapq.heappush(heap, (-qf[m], m))
    # seams
    nseam = 0
    for _ in range(2):
        A = np.isfinite(u) & (state >= 4); bad = np.zeros_like(A)
        dz = np.abs(np.diff(u, axis=0)) > P['JMAX']; dz &= A[1:] & A[:-1]
        dc = np.abs(np.diff(u, axis=1)) > P['JMAX']; dc &= A[:, 1:] & A[:, :-1]
        bad[1:] |= dz; bad[:-1] |= dz; bad[:, 1:] |= dc; bad[:, :-1] |= dc
        nseam += int(bad.sum())
        u[bad] = np.nan; state[bad] = 3
    ass = state >= 4
    nsheet = has[:, c0:].size
    info = dict(P=P, nodes=int(nsheet), with_cand=int(has.sum()), anchors=int((state == 5).sum()), assigned=int(ass.sum()),
                ambiguous=int((state == 2).sum()), nofit=int((state == 3).sum()), unreached=int((state == 1).sum()),
                seam_unassigned=nseam, pops=npop)
    return dict(u=u, r=r, state=state, info=info)


def field(T, sigma=0.0, D=3.0, tw=None):
    """tracked u -> move field F (Nr, W): normalised Gaussian (sigma nodes, weights = chosen NCC) over assigned nodes, kept only at
    assigned nodes, times a distance taper clip(d / D, 0, 1) to the nearest unassigned node, times the z taper tw (Nr,)."""
    u, r = T['u'], T['r']; A = np.isfinite(u)
    w = np.where(A, r, 0.0); uu = np.where(A, u, 0.0)
    if sigma > 0:
        num = gaussian_filter(w * uu, sigma, mode='constant'); den = gaussian_filter(w, sigma, mode='constant')
        Fs = np.where(den > 1e-9, num / np.maximum(den, 1e-12), 0.0)
    else:
        Fs = uu
    F = np.where(A, Fs, 0.0)
    if D > 0:
        d = distance_transform_edt(A)
        F = F * np.clip(d / D, 0, 1)
    if tw is not None:
        F = F * tw[:, None]
    return F
