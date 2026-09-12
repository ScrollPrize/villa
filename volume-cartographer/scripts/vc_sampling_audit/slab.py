"""Local continuous-depth diagnostic, not anatomical or global certification."""
import numpy as np

from .geometry import DIAGONALS, area_coefficients, evaluate, quadratic_minimum, triangulate

CONTRACT = "piecewise-affine-triangle-sweep"


def report_grid(q, n, reference, low, high, *, contract, units, witness_limit=12):
    if contract != CONTRACT:
        raise ValueError("UNSUPPORTED_INTERPOLATION_CONTRACT")
    if not units or not np.isfinite([low, high]).all() or low > high:
        raise ValueError("Require declared units and finite ordered interval")
    q = np.asarray(q, dtype=np.float64); n = np.asarray(n, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if q.shape != n.shape or reference.shape != (3,) or not np.isfinite(reference).all():
        raise ValueError("Position/direction/reference shape or finiteness mismatch")
    if np.linalg.norm(reference) == 0:
        raise ValueError("Reference direction is zero")
    results = {}
    for diagonal in DIAGONALS:
        triangles = triangulate(q, diagonal)
        vectors = triangulate(n, diagonal)
        a, b, c = area_coefficients(triangles, vectors)
        base_jac = np.einsum("ni,nvi->nv", c, vectors)
        # A global normal reversal only reverses the depth coordinate. Preserve
        # the physical sample set and map witness depths back to the input sign.
        sign = -1 if np.median(base_jac) < 0 else 1
        vectors = sign * vectors
        lo, hi = (low, high) if sign == 1 else (-high, -low)
        summary, projected_bad, volume_bad = evaluate(triangles, vectors, reference, lo, hi)
        a, b, c = area_coefficients(triangles, vectors)
        pc = (a @ reference, b @ reference, c @ reference)
        vc = tuple(np.einsum("ni,nvi->nv", x, vectors) for x in (a,b,c))
        pmin, pd = quadratic_minimum(*pc, lo, hi)
        vmin, vd = quadratic_minimum(*vc, lo, hi)
        endpoint_p = np.minimum(pc[0]*lo**2+pc[1]*lo+pc[2], pc[0]*hi**2+pc[1]*hi+pc[2])
        witnesses = []
        # Deterministic topology order, never rank by an ink/detector score.
        for idx in np.flatnonzero(projected_bad | volume_bad)[:max(0, witness_limit)]:
            tri, row, col = np.unravel_index(idx, (2, q.shape[0]-1, q.shape[1]-1))
            vertex = int(np.argmin(vmin[idx] / vc[2][idx]))
            witnesses.append({"cell_row": int(row), "cell_col": int(col), "triangle_in_cell": int(tri),
                              "projected_min_depth": float(sign*pd[idx]),
                              "projected_min_ratio": float(pmin[idx]/pc[2][idx]),
                              "volume_min_depth": float(sign*vd[idx, vertex]),
                              "volume_vertex": vertex,
                              "volume_min_ratio": float(vmin[idx,vertex]/vc[2][idx,vertex]),
                              "base_xyz": triangles[idx].tolist()})
        summary.pop("interval_native")
        summary.update(depth_interval_input_units=[low,high], audit_direction_sign=sign,
                       endpoint_only_projected_failures=int((endpoint_p <= 0).sum()),
                       interior_projected_failures_missed_by_endpoints=int((projected_bad & (endpoint_p > 0)).sum()),
                       failure_witnesses=witnesses, witnesses_truncated=int((projected_bad | volume_bad).sum())>len(witnesses))
        results[diagonal] = summary
    return {"status": "LOCAL_SAMPLING_DIAGNOSTIC_ONLY", "interpolation_contract": contract,
            "units": units, "zero_threshold": 0.0, "floating_point": "float64; numerical diagnostic, not interval-arithmetic certification",
            "normal_interpolation": "Affine inside each triangle; never renormalized between supplied vertices",
            "position_shape": list(q.shape), "reference_direction": reference.tolist(),
            "corner_definition": "A=(r,c), B=(r+1,c), C=(r+1,c+1), D=(r,c+1)",
            "triangle_orders": DIAGONALS, "diagonals": results,
            "global_injectivity": "NOT_ASSESSED", "anatomical_sheet_identity": "NOT_ASSESSED",
            "CT_support": "NOT_ASSESSED", "ink_or_letters": "NOT_ASSESSED",
            "renderer_interpolation_match": "CALLER_MUST_ESTABLISH; no automatic VC renderer certification"}
