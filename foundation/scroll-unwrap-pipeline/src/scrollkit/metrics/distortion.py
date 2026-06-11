"""Distortion metrics.

`flatboi_stretch_metrics` mirrors volume-cartographer/libs/flatboi/flatboi.cpp
`stretch_metrics`/`triangle_stretch` exactly (area-weighted Sander L2 mean, weighted
median, Linf = max largest singular value, signed per-triangle area error) so reports
use the team's established numbers.

`frame_metrics` evaluates one animation frame against the unwrap-math-spec gates.
"""

from __future__ import annotations

import numpy as np


def _triangle_stretch_terms(V3: np.ndarray, UV2: np.ndarray, F: np.ndarray):
    """Vectorized port of flatboi's triangle_stretch. Returns (L2², Linf(G), area3d, area2d_abs)."""
    q1, q2, q3 = (V3[F[:, k]].astype(np.float64) for k in range(3))
    p1, p2, p3 = (UV2[F[:, k]].astype(np.float64) for k in range(3))
    s1, t1 = p1[:, 0], p1[:, 1]
    s2, t2 = p2[:, 0], p2[:, 1]
    s3, t3 = p3[:, 0], p3[:, 1]
    A = ((s2 - s1) * (t3 - t1) - (s3 - s1) * (t2 - t1)) / 2.0
    Aabs = np.abs(A)
    ok = Aabs >= 1e-30
    Asafe = np.where(ok, A, 1.0)
    Ss = (q1 * (t2 - t3)[:, None] + q2 * (t3 - t1)[:, None] + q3 * (t1 - t2)[:, None]) / (2.0 * Asafe[:, None])
    St = (q1 * (s3 - s2)[:, None] + q2 * (s1 - s3)[:, None] + q3 * (s2 - s1)[:, None]) / (2.0 * Asafe[:, None])
    a = np.einsum("ij,ij->i", Ss, Ss)
    b = np.einsum("ij,ij->i", Ss, St)
    c = np.einsum("ij,ij->i", St, St)
    G = np.sqrt(np.maximum(((a + c) + np.sqrt((a - c) ** 2 + 4.0 * b * b)) / 2.0, 0.0))
    L2sq = (a + c) / 2.0
    ab = np.linalg.norm(q2 - q1, axis=1)
    bc = np.linalg.norm(q3 - q2, axis=1)
    ca = np.linalg.norm(q1 - q3, axis=1)
    s = (ab + bc + ca) / 2.0
    area3d = np.sqrt(np.maximum(s * (s - ab) * (s - bc) * (s - ca), 0.0))
    L2sq = np.where(ok, L2sq, 0.0)
    G = np.where(ok, G, 0.0)
    area3d = np.where(ok, area3d, 0.0)
    Aabs = np.where(ok, Aabs, 0.0)
    return L2sq, G, area3d, Aabs


def _weighted_median(data: np.ndarray, weights: np.ndarray) -> float:
    idx = np.argsort(data)
    acc = np.cumsum(weights[idx])
    cut = 0.5 * weights.sum()
    k = int(np.searchsorted(acc, cut))
    return float(data[idx[min(k, len(idx) - 1)]])


def flatboi_stretch_metrics(V3: np.ndarray, UV2: np.ndarray, F: np.ndarray) -> dict:
    """flatboi stretch_metrics verbatim: (L2_mean, L2_median, Linf, area_error)."""
    L2sq, G, area3d, area2d = _triangle_stretch_terms(V3, UV2, F)
    sum_a3 = area3d.sum()
    l2_mean = float(np.sqrt((L2sq * area3d).sum() / sum_a3)) if sum_a3 > 0 else 0.0
    l2_median = _weighted_median(L2sq, area3d)
    linf = float(G.max()) if len(G) else 0.0
    sum_a2 = area2d.sum()
    alpha = area3d / sum_a3 if sum_a3 > 0 else np.zeros_like(area3d)
    beta = area2d / sum_a2 if sum_a2 > 0 else np.zeros_like(area2d)
    per_tri = np.where(alpha > beta, 1.0 - beta / (alpha + 1e-30), 1.0 - alpha / (beta + 1e-30))
    area_err = float(per_tri.sum() / max(1, len(F)))
    return {"l2_mean": l2_mean, "l2_median": l2_median, "linf": linf, "area_error": area_err}


def symmetric_dirichlet(V_from: np.ndarray, V_to: np.ndarray, F: np.ndarray,
                        face_mask: np.ndarray | None = None) -> float:
    """Area-weighted mean symmetric Dirichlet energy of the per-face map V_from→V_to
    (singular values σ of the 2D-restricted differential): mean_A(σ1²+σ2²+σ1⁻²+σ2⁻²)/4, 1=isometry."""
    if face_mask is not None:
        F = F[face_mask]
    e1f = V_from[F[:, 1]] - V_from[F[:, 0]]
    e2f = V_from[F[:, 2]] - V_from[F[:, 0]]
    e1t = V_to[F[:, 1]] - V_to[F[:, 0]]
    e2t = V_to[F[:, 2]] - V_to[F[:, 0]]

    def to_local2d(e1, e2):
        x1 = np.linalg.norm(e1, axis=1)
        x1 = np.where(x1 < 1e-300, 1e-300, x1)
        u = e1 / x1[:, None]
        proj = np.einsum("ij,ij->i", e2, u)
        h = e2 - proj[:, None] * u
        y2 = np.linalg.norm(h, axis=1)
        return np.stack([np.stack([x1, proj], -1), np.stack([np.zeros_like(y2), y2], -1)], axis=2)

    Sf = to_local2d(e1f, e2f)  # (m,2,2) columns = local coords of e1,e2
    St = to_local2d(e1t, e2t)
    ok = np.abs(np.linalg.det(Sf)) > 1e-24
    J = np.einsum("mij,mjk->mik", St[ok], np.linalg.inv(Sf[ok]))
    # closed form for 2x2: σ1²+σ2² = ‖J‖²_F, σ1²σ2² = det(J)² ⇒ Σσ⁻² = ‖J‖²_F/det²
    fro2 = np.einsum("mij,mij->m", J, J)
    det2 = np.maximum(np.linalg.det(J) ** 2, 1e-24)
    e = (fro2 + fro2 / det2) / 4.0
    A = 0.5 * np.linalg.norm(np.cross(e1f, e2f), axis=1)
    A = A[ok]
    return float((e * A).sum() / max(A.sum(), 1e-300))


def frame_metrics(
    Vt: np.ndarray,
    t: float,
    V0: np.ndarray,
    V1: np.ndarray,
    F: np.ndarray,
    A0: np.ndarray,
    A1: np.ndarray,
    N_blend_ref: np.ndarray | None = None,
    visible: np.ndarray | None = None,
    ref_mode: str = "blend",
    flip_exclude: np.ndarray | None = None,
) -> dict:
    """Per-frame gate metrics (see unwrap-math-spec §metrics).

    ref_mode: 'blend' compares per-face areas/edges against the linear endpoint blend
    (right for fields that deform everywhere, e.g. the DG path). 'step' compares against
    the NEAREST endpoint — right for rolling-contact kinematics whose per-face schedule is
    a step (rigid at A0 until the contact passes, exactly A1 after); the blend reference
    would penalize that step for being correct."""
    finite = bool(np.isfinite(Vt).all())
    e1 = Vt[F[:, 1]] - Vt[F[:, 0]]
    e2 = Vt[F[:, 2]] - Vt[F[:, 0]]
    n = np.cross(e1, e2)
    At = 0.5 * np.linalg.norm(n, axis=1)
    degenerate = int((At < 1e-12).sum())

    if ref_mode == "step":
        rel = np.minimum(np.abs(At - A0), np.abs(At - A1)) / np.maximum(
            np.minimum(A0, A1), 1e-300)
    else:
        blend = (1.0 - t) * A0 + t * A1
        rel = np.abs(At - blend) / np.maximum(blend, 1e-300)
    area_p95 = float(np.quantile(rel, 0.95))
    area_p999 = float(np.quantile(rel, 0.999))
    area_max = float(rel.max())
    rel_vis = rel[visible] if visible is not None and visible.any() else rel
    area_p95_vis = float(np.quantile(rel_vis, 0.95))
    area_p999_vis = float(np.quantile(rel_vis, 0.999))

    def edge_lengths(V):
        return np.stack([
            np.linalg.norm(V[F[:, 1]] - V[F[:, 0]], axis=1),
            np.linalg.norm(V[F[:, 2]] - V[F[:, 1]], axis=1),
            np.linalg.norm(V[F[:, 0]] - V[F[:, 2]], axis=1),
        ])

    Lt = edge_lengths(Vt)
    L0, L1 = edge_lengths(V0), edge_lengths(V1)
    if ref_mode == "step":
        erel = np.minimum(np.abs(Lt - L0), np.abs(Lt - L1)) / np.maximum(
            np.minimum(L0, L1), 1e-300)
    else:
        Lb = (1.0 - t) * L0 + t * L1
        erel = np.abs(Lt - Lb) / np.maximum(Lb, 1e-300)
    edge_p95 = float(np.quantile(erel, 0.95))
    edge_max = float(erel.max())
    erel_vis = erel[:, visible] if visible is not None and visible.any() else erel
    edge_p95_vis = float(np.quantile(erel_vis, 0.95))

    flipped = 0
    if N_blend_ref is not None:
        nn = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)
        flip_mask = np.einsum("ij,ij->i", nn, N_blend_ref) < 0
        if flip_exclude is not None:
            flip_mask = flip_mask & ~flip_exclude   # takeoff-band swing is by-design
        flipped = int(flip_mask.sum())

    sd = symmetric_dirichlet(V0, Vt, F, face_mask=visible)
    return {
        "t": float(t),
        "finite": finite,
        "degenerate_tris": degenerate,
        "flipped_tris": flipped,
        "area_rel_p95": area_p95,
        "area_rel_p999": area_p999,
        "area_rel_max": area_max,
        "area_rel_p95_vis": area_p95_vis,
        "area_rel_p999_vis": area_p999_vis,
        "edge_rel_p95": edge_p95,
        "edge_rel_max": edge_max,
        "edge_rel_p95_vis": edge_p95_vis,
        "sym_dirichlet_vs_source": sd,
    }
