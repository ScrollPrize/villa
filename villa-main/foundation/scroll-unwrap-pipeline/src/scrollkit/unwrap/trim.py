"""Data-driven junk-tail trim for wrap charts (animation deliverables only).

Segmentation traces can end in low-confidence tails where the UV chart is
squeezed or warped (e.g. a double-digit denormalization residual concentrated
in the inner tail). Consequences: per-vertex arcs disagree with flat positions
by thousands of voxels (material transported through already-settled sheet),
warp-seam slivers spike at the roll top, and the squeezed chart renders its
texture content as a compressed, seam-bounded column at the flat sheet's edge
— a faithful sampling of the input, but a geometrically broken presentation.

Rule: per-u-bin median of the relative isometry residual of the global denorm
fit; a CONTIGUOUS run of bad bins touching either chart end is junk and its
faces are dropped (cap per end: max_trim_frac of the u-range). Interior junk is
never trimmed (it would split the sheet). M1/M2 mesh deliverables stay FULL —
only the unwrap animation consumes the trimmed mesh, and the trim is recorded
in metrics.json.
"""

from __future__ import annotations

import numpy as np

__all__ = ["junk_tail_trim", "enforce_chart_injectivity"]


def enforce_chart_injectivity(V: np.ndarray, F: np.ndarray, uv: np.ndarray, *,
                              r_uv: float = 5e-4, d3_min_frac: float = 0.15) -> dict:
    """Drop duplicated chart patches: a chart must be injective (one UV point ==
    one 3D point). Segmentation traces occasionally duplicate a patch (the trace
    rides the same physical surface twice); both layers carry the SAME texture
    region, so they render as offset double-exposed text and flicker as the
    layers cross while landing (typically a few hundred vertices at a trace's
    inner tail).

    Detection: vertex pairs within r_uv in UV but > d3_min_frac*median_edge apart
    in 3D. Resolution: per connected component of flagged vertices, keep the
    layer consistent with the LOCAL CONSENSUS — the median 3D position of
    unflagged UV-neighbours — and drop faces touching the deviant layer."""
    from scipy.spatial import cKDTree

    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F)
    uv = np.asarray(uv, dtype=np.float64)
    e = V[F[:, 1]] - V[F[:, 0]]
    med_edge = float(np.median(np.linalg.norm(e, axis=1)))
    d3_min = d3_min_frac * med_edge

    tree = cKDTree(uv)
    pairs = tree.query_pairs(r=r_uv, output_type="ndarray")
    info = {"n_pairs_uv": int(len(pairs)), "n_pairs_far": 0, "n_verts_flagged": 0,
            "n_verts_dropped": 0, "n_faces_dropped": 0, "med_edge": med_edge}
    if len(pairs) == 0:
        return {"V": V, "F": F, "uv": uv, "vert_keep": np.ones(len(V), bool), "info": info}
    d3 = np.linalg.norm(V[pairs[:, 0]] - V[pairs[:, 1]], axis=1)
    far = pairs[d3 > d3_min]
    info["n_pairs_far"] = int(len(far))
    if len(far) == 0:
        return {"V": V, "F": F, "uv": uv, "vert_keep": np.ones(len(V), bool), "info": info}

    flagged = np.zeros(len(V), bool)
    flagged[far.ravel()] = True
    info["n_verts_flagged"] = int(flagged.sum())

    # consensus position per flagged vertex from UNFLAGGED uv-neighbours
    ball = tree.query_ball_point(uv[flagged], r=3.0 * r_uv)
    flag_idx = np.flatnonzero(flagged)
    consensus = np.full((len(flag_idx), 3), np.nan)
    for k, nb in enumerate(ball):
        nb = np.asarray(nb)
        nb = nb[~flagged[nb]]
        if len(nb) >= 3:
            consensus[k] = np.median(V[nb], axis=0)
    dev = np.linalg.norm(V[flag_idx] - consensus, axis=1)

    # connected components of flagged vertices over mesh edges
    drop = np.zeros(len(V), bool)
    f_any = flagged[F].any(axis=1)
    adj: dict[int, list[int]] = {}
    for tri in F[f_any]:
        vs = [v for v in tri if flagged[v]]
        for a in vs:
            for b in vs:
                if a != b:
                    adj.setdefault(int(a), []).append(int(b))
    seen = set()
    pos_in_flag = {int(v): i for i, v in enumerate(flag_idx)}
    for start in flag_idx:
        start = int(start)
        if start in seen:
            continue
        comp = [start]
        seen.add(start)
        stack = [start]
        while stack:
            cur = stack.pop()
            for nxt in adj.get(cur, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    comp.append(nxt)
                    stack.append(nxt)
        devs = [dev[pos_in_flag[v]] for v in comp if not np.isnan(dev[pos_in_flag[v]]).any()]
        med_dev = float(np.median(devs)) if devs else np.inf
        # a duplicate layer sits far from the local consensus sheet; the genuine
        # layer IS the consensus (median dev ~ 0)
        if med_dev > d3_min:
            for v in comp:
                drop[v] = True
    info["n_verts_dropped"] = int(drop.sum())
    if not drop.any():
        return {"V": V, "F": F, "uv": uv, "vert_keep": np.ones(len(V), bool), "info": info}

    keep_f = ~drop[F].any(axis=1)
    info["n_faces_dropped"] = int((~keep_f).sum())
    F2 = F[keep_f]
    used = np.zeros(len(V), bool)
    used[F2] = True
    remap = -np.ones(len(V), np.int64)
    remap[used] = np.arange(int(used.sum()))
    return {"V": V[used], "F": remap[F2], "uv": uv[used], "vert_keep": used, "info": info}


def _denorm_fit(V: np.ndarray, F: np.ndarray, uv: np.ndarray) -> tuple[float, float]:
    P = [V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 1]], V[F[:, 0]] - V[F[:, 2]]]
    Q = [uv[F[:, 1]] - uv[F[:, 0]], uv[F[:, 2]] - uv[F[:, 1]], uv[F[:, 0]] - uv[F[:, 2]]]
    d2 = np.concatenate([np.sum(p * p, axis=1) for p in P])
    du2 = np.concatenate([q[:, 0] ** 2 for q in Q])
    dv2 = np.concatenate([q[:, 1] ** 2 for q in Q])
    A = np.stack([du2, dv2], axis=1)
    coef, *_ = np.linalg.lstsq(A, d2, rcond=None)
    return float(np.sqrt(max(coef[0], 1e-12))), float(np.sqrt(max(coef[1], 1e-12)))


def junk_tail_trim(V: np.ndarray, F: np.ndarray, uv: np.ndarray, *,
                   n_bins: int = 200, bad_ratio: float = 3.0,
                   min_bad_bins: int = 2, max_trim_frac: float = 0.10) -> dict:
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F)
    uv = np.asarray(uv, dtype=np.float64)
    s_u, s_v = _denorm_fit(V, F, uv)

    # per-face relative isometry residual (worst edge of the face)
    res_face = np.zeros(len(F))
    for a, b in ((0, 1), (1, 2), (2, 0)):
        e = V[F[:, b]] - V[F[:, a]]
        q = uv[F[:, b]] - uv[F[:, a]]
        d2 = np.sum(e * e, axis=1)
        m2 = (s_u * q[:, 0]) ** 2 + (s_v * q[:, 1]) ** 2
        r = np.abs(d2 - m2) / np.maximum(d2, 1e-12)
        res_face = np.maximum(res_face, r)

    cu = uv[F].mean(axis=1)[:, 0]                     # face centroid u
    u_lo, u_hi = float(cu.min()), float(cu.max())
    edges = np.linspace(u_lo, u_hi, n_bins + 1)
    which = np.clip(np.searchsorted(edges, cu) - 1, 0, n_bins - 1)
    med_bin = np.full(n_bins, np.nan)
    for b in range(n_bins):
        m = which == b
        if m.any():
            med_bin[b] = np.median(res_face[m])
    global_med = float(np.median(res_face))
    thresh = max(bad_ratio * global_med, 0.02)
    bad = med_bin > thresh                              # NaN bins -> False (empty)

    def tail_len(order: np.ndarray) -> int:
        n = 0
        for b in order:
            if np.isnan(med_bin[b]) or bad[b]:
                n += 1
            else:
                break
        return n

    cap = int(max_trim_frac * n_bins)
    lo_n = min(tail_len(np.arange(n_bins)), cap)
    hi_n = min(tail_len(np.arange(n_bins)[::-1]), cap)
    if lo_n < min_bad_bins:
        lo_n = 0
    if hi_n < min_bad_bins:
        hi_n = 0

    u_min_keep = edges[lo_n] if lo_n else -np.inf
    u_max_keep = edges[n_bins - hi_n] if hi_n else np.inf
    keep_f = (cu >= u_min_keep) & (cu <= u_max_keep)
    info = {
        "s_u": s_u, "s_v": s_v,
        "global_med_residual": global_med, "threshold": thresh,
        "trim_lo_bins": int(lo_n), "trim_hi_bins": int(hi_n),
        "u_keep": [float(u_min_keep), float(u_max_keep)],
        "n_faces_dropped": int((~keep_f).sum()),
        "frac_faces_dropped": float((~keep_f).mean()),
    }
    if not (~keep_f).any():
        return {"V": V, "F": F, "uv": uv, "vert_keep": np.ones(len(V), bool),
                "info": info}

    F2 = F[keep_f]
    used = np.zeros(len(V), bool)
    used[F2] = True
    remap = -np.ones(len(V), np.int64)
    remap[used] = np.arange(int(used.sum()))
    return {"V": V[used], "F": remap[F2], "uv": uv[used], "vert_keep": used,
            "info": info}
