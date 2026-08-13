"""Recover the physical scale of normalized UVs.

The wrap UVs are SLIM output in voxel units, shifted to origin and normalized per-axis
to [0,1] at texturing time (see flatboi.cpp). For edge e: ||dp_e||^2 ~= a*du_e^2 + b*dv_e^2
with a=s_u^2, b=s_v^2. Linear least squares; one robust refit dropping the worst 1% residuals.
"""

from __future__ import annotations

import numpy as np


def mesh_edges(faces: np.ndarray) -> np.ndarray:
    """Unique undirected edges (k,2) int64 from (m,3) faces."""
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0).astype(np.int64)
    e.sort(axis=1)
    return np.unique(e, axis=0)


def fit_uv_scale(
    vertices: np.ndarray,
    faces: np.ndarray,
    uv: np.ndarray,
    *,
    robust_trim: float = 0.01,
) -> dict:
    """Fit per-axis de-normalization scales (s_u, s_v).

    Returns dict with s_u, s_v, anisotropy (s_u/s_v), rel_residual (RMS relative error of
    edge lengths reconstructed from scaled UVs), n_edges.
    """
    V = np.asarray(vertices, dtype=np.float64)
    UV = np.asarray(uv, dtype=np.float64)
    E = mesh_edges(np.asarray(faces))
    dp2 = np.sum((V[E[:, 0]] - V[E[:, 1]]) ** 2, axis=1)
    du2 = (UV[E[:, 0], 0] - UV[E[:, 1], 0]) ** 2
    dv2 = (UV[E[:, 0], 1] - UV[E[:, 1], 1]) ** 2

    def lsq(mask: np.ndarray) -> np.ndarray:
        A = np.stack([du2[mask], dv2[mask]], axis=1)
        coef, *_ = np.linalg.lstsq(A, dp2[mask], rcond=None)
        return np.maximum(coef, 0.0)

    mask = np.ones(len(dp2), dtype=bool)
    coef = lsq(mask)
    pred = du2 * coef[0] + dv2 * coef[1]
    resid = np.abs(pred - dp2)
    if robust_trim > 0 and len(resid) > 100:
        cut = np.quantile(resid, 1.0 - robust_trim)
        coef = lsq(resid <= cut)
        pred = du2 * coef[0] + dv2 * coef[1]

    s_u, s_v = float(np.sqrt(coef[0])), float(np.sqrt(coef[1]))
    len_true = np.sqrt(dp2)
    len_pred = np.sqrt(np.maximum(pred, 0.0))
    ok = len_true > 0
    rel = (len_pred[ok] - len_true[ok]) / len_true[ok]
    return {
        "s_u": s_u,
        "s_v": s_v,
        "anisotropy": s_u / s_v if s_v > 0 else float("inf"),
        "rel_residual_rms": float(np.sqrt(np.mean(rel**2))),
        "rel_residual_p95": float(np.quantile(np.abs(rel), 0.95)),
        "n_edges": int(len(dp2)),
    }
