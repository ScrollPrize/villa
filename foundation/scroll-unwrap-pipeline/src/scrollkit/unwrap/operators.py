"""Differential operators for the Poisson reconstruction.

Primary: libigl wheel (igl.grad). Fallback: scipy-only construction (same row layout).
Layout contract: G is (3m, n); rows [0:m] are d/dx-ish components in igl's per-face frame —
we only ever use G together with the matching area weights and per-face target gradients,
so the internal layout just has to be self-consistent (asserted in tests on a known mesh).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def face_areas(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    e1 = V[F[:, 1]] - V[F[:, 0]]
    e2 = V[F[:, 2]] - V[F[:, 0]]
    return 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)


def grad_operator(V: np.ndarray, F: np.ndarray) -> sp.csr_matrix:
    """Per-face gradient operator (3m × n), igl layout: rows stacked [m dx; m dy; m dz]
    in WORLD coordinates (igl.grad uses the 3D embedding, giving world-frame gradients)."""
    try:
        import igl

        G = igl.grad(np.asarray(V, dtype=np.float64), np.asarray(F, dtype=np.int64))
        return sp.csr_matrix(G)
    except Exception:
        return _grad_scipy(np.asarray(V, dtype=np.float64), np.asarray(F))


def _grad_scipy(V: np.ndarray, F: np.ndarray) -> sp.csr_matrix:
    """World-frame per-face gradient: ∇φ_i are the standard linear FEM hat-gradients.

    For triangle (i,j,k) with area A and unit normal n̂:
      ∇φ_i = (n̂ × e_jk) / (2A),  e_jk = x_k − x_j   (rotated opposite edge)
    """
    m, n = len(F), len(V)
    i, j, k = F[:, 0], F[:, 1], F[:, 2]
    e_i = V[k] - V[j]
    e_j = V[i] - V[k]
    e_k = V[j] - V[i]
    nrm = np.cross(V[j] - V[i], V[k] - V[i])
    dblA = np.linalg.norm(nrm, axis=1)
    dblA = np.where(dblA < 1e-300, 1e-300, dblA)
    n_hat = nrm / dblA[:, None]
    gi = np.cross(n_hat, e_i) / dblA[:, None]
    gj = np.cross(n_hat, e_j) / dblA[:, None]
    gk = np.cross(n_hat, e_k) / dblA[:, None]

    rows, cols, vals = [], [], []
    for d in range(3):  # world axis
        base = d * m
        r = np.arange(m) + base
        for g, idx in ((gi, i), (gj, j), (gk, k)):
            rows.append(r)
            cols.append(idx)
            vals.append(g[:, d])
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)
    return sp.csr_matrix((vals, (rows, cols)), shape=(3 * m, n))


def poisson_operator(V: np.ndarray, F: np.ndarray) -> tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray]:
    """Return (L, G, area3): L = Gᵀ diag(area⊗3) G (n×n, PSD with 1-dim null space), G (3m×n)."""
    G = grad_operator(V, F)
    A = face_areas(np.asarray(V, np.float64), F)
    W = sp.diags(np.tile(A, 3))
    L = (G.T @ W @ G).tocsr()
    return L, G, A


def gradient_rhs(G: sp.csr_matrix, A: np.ndarray, target_grads: np.ndarray) -> np.ndarray:
    """RHS = Gᵀ diag(area⊗3) f for per-face target gradient matrices.

    target_grads: (m, 3, 3) where target_grads[f] @ (per-face source gradient of x) ...
    Practically we pass f = stacked per-face world gradients of the target coordinates,
    shaped (3m, 3) matching G's row layout: rows [d*m + f] = d-th world component of ∇(coord)
    at face f, columns = x,y,z target coordinate functions.
    """
    W = np.tile(A, 3)[:, None]
    return G.T @ (W * target_grads)
