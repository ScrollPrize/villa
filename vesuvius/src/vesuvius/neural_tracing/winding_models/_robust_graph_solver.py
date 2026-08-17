"""Shared robust graph-offset solver for winding reconstruction."""

from __future__ import annotations

import numpy as np


def solve_robust_graph_offsets(
    prior,
    edge_u,
    edge_v,
    edge_delta,
    edge_weight,
    *,
    iterations=5,
    huber=0.25,
    prior_weight=0.02,
    prior_huber=0.5,
    max_correction=4.0,
    error_context="graph offset",
):
    """Solve ``offset[v] - offset[u] = delta`` with matrix-free IRLS.

    Returns the solved offsets, their correction from ``prior``, final edge
    residuals, and per-node edge degree. Invalid edges are discarded.
    """
    from scipy.sparse.linalg import LinearOperator, cg

    prior = np.asarray(prior, dtype=np.float64)
    edge_u = np.asarray(edge_u, dtype=np.int64)
    edge_v = np.asarray(edge_v, dtype=np.int64)
    edge_delta = np.asarray(edge_delta, dtype=np.float64)
    base_weight = np.asarray(edge_weight, dtype=np.float64)
    nodes = len(prior)
    finite = (
        np.isfinite(edge_delta)
        & np.isfinite(base_weight)
        & (base_weight > 0)
        & (edge_u >= 0)
        & (edge_u < nodes)
        & (edge_v >= 0)
        & (edge_v < nodes)
        & (edge_u != edge_v)
    )
    edge_u, edge_v, edge_delta, base_weight = (
        value[finite] for value in (edge_u, edge_v, edge_delta, base_weight)
    )
    degree = np.bincount(edge_u, minlength=nodes) + np.bincount(edge_v, minlength=nodes)
    if not len(edge_u):
        return prior.copy(), np.zeros(nodes, dtype=np.float64), edge_delta, degree

    target = edge_delta - (prior[edge_v] - prior[edge_u])
    correction = np.zeros(nodes, dtype=np.float64)
    robust_edge_weight = base_weight.copy()
    robust_prior_weight = np.full(nodes, float(prior_weight), dtype=np.float64)

    for _ in range(max(1, int(iterations))):

        def matvec(
            values,
            edge_weights=robust_edge_weight,
            prior_weights=robust_prior_weight,
        ):
            return (
                np.bincount(
                    edge_u,
                    edge_weights * (values[edge_u] - values[edge_v]),
                    minlength=nodes,
                )
                + np.bincount(
                    edge_v,
                    edge_weights * (values[edge_v] - values[edge_u]),
                    minlength=nodes,
                )
                + prior_weights * values
            )

        operator = LinearOperator((nodes, nodes), matvec=matvec, dtype=np.float64)
        diagonal = (
            np.bincount(edge_u, robust_edge_weight, minlength=nodes)
            + np.bincount(edge_v, robust_edge_weight, minlength=nodes)
            + robust_prior_weight
        )
        preconditioner = LinearOperator(
            (nodes, nodes),
            matvec=lambda values, diagonal=diagonal: (
                values / np.maximum(diagonal, 1e-12)
            ),
            dtype=np.float64,
        )
        weighted_target = robust_edge_weight * target
        rhs = np.bincount(edge_v, weighted_target, minlength=nodes) - np.bincount(
            edge_u, weighted_target, minlength=nodes
        )
        correction, info = cg(
            operator,
            rhs,
            x0=correction,
            M=preconditioner,
            rtol=1e-5,
            atol=1e-8,
            maxiter=200,
        )
        if info < 0:  # pragma: no cover - scipy input/internal failure
            raise RuntimeError(f"{error_context} solve failed: cg={info}")
        np.clip(
            correction,
            -float(max_correction),
            float(max_correction),
            out=correction,
        )
        residual = correction[edge_v] - correction[edge_u] - target
        robust_edge_weight = base_weight * np.minimum(
            1.0, float(huber) / np.maximum(np.abs(residual), 1e-12)
        )
        robust_prior_weight = float(prior_weight) * np.minimum(
            1.0,
            float(prior_huber) / np.maximum(np.abs(correction), 1e-12),
        )

    solved = prior + correction
    residual = solved[edge_v] - solved[edge_u] - edge_delta
    return solved, correction, residual, degree
