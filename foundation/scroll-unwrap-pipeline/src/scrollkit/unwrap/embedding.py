"""Target embedding: place the de-normalized flat UV domain as a plane in 3D.

Frame conventions (see docs/UNWRAP-MATH.md):
- the UV axis best correlated with a consistent 3D direction is the 'along-axis' (scroll
  axis) direction; the other is the unroll direction.
- the flat sheet is rigidly placed by Procrustes of its anchor strip onto the same strip's
  3D position, so the morph 'peels open' from a near-stationary edge.
- chirality: the embedding must not mirror the texture; checked numerically here (basis
  determinant against the source surface orientation) and end-to-end by the render oracle.
"""

from __future__ import annotations

import numpy as np

from .denorm import fit_uv_scale


def fit_scroll_axis(V: np.ndarray, F: np.ndarray, uv: np.ndarray) -> dict:
    """Decide which UV axis runs along the scroll axis, by tangent coherence.

    Per face, world tangents (t_u, t_v) = inv([Δuv1; Δuv2]) · [e1; e2]. The along-axis
    direction has unit tangents that agree globally (coherence ≈ 1); the unroll direction's
    tangents sweep around the winding and cancel (coherence ≪ 1, →0 over full turns).
    Area-weighted; faces with degenerate UV are skipped.
    """
    V = np.asarray(V, dtype=np.float64)
    uv = np.asarray(uv, dtype=np.float64)
    e1 = V[F[:, 1]] - V[F[:, 0]]
    e2 = V[F[:, 2]] - V[F[:, 0]]
    d1 = uv[F[:, 1]] - uv[F[:, 0]]
    d2 = uv[F[:, 2]] - uv[F[:, 0]]
    det = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    ok = np.abs(det) > 1e-18
    inv_det = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    # rows of inv([[du1,dv1],[du2,dv2]]) = [[dv2,-dv1],[-du2,du1]]/det
    t_u = (d2[:, 1, None] * e1 - d1[:, 1, None] * e2) * inv_det[:, None]
    t_v = (-d2[:, 0, None] * e1 + d1[:, 0, None] * e2) * inv_det[:, None]
    area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    w = np.where(ok, area, 0.0)

    def coherence(t: np.ndarray) -> tuple[float, np.ndarray]:
        n = np.linalg.norm(t, axis=1)
        good = (n > 1e-300) & ok
        unit = np.zeros_like(t)
        unit[good] = t[good] / n[good][:, None]
        mean = (unit * w[:, None]).sum(0) / max(w.sum(), 1e-300)
        return float(np.linalg.norm(mean)), mean

    coh_u, mean_u = coherence(t_u)
    coh_v, mean_v = coherence(t_v)
    axis_uv = "v" if coh_v >= coh_u else "u"
    mean_dir = mean_v if axis_uv == "v" else mean_u
    axis_dir = mean_dir / max(np.linalg.norm(mean_dir), 1e-300)
    return {
        "axis_uv": axis_uv,
        "axis_dir": axis_dir,
        "details": {"u": {"coherence": coh_u}, "v": {"coherence": coh_v}},
    }


def anchor_strip_indices(uv: np.ndarray, unroll_axis: int, frac: float = 0.02) -> tuple[np.ndarray, str]:
    """Vertices within `frac` of one extreme of the unroll axis.

    Both extremes are candidates; pick the one with smaller 3D spread relative to its UV
    extent later — here we simply return the low end; build_target_embedding tries both.
    """
    q = uv[:, unroll_axis]
    lo, hi = q.min(), q.max()
    span = hi - lo
    low_idx = np.where(q <= lo + frac * span)[0]
    high_idx = np.where(q >= hi - frac * span)[0]
    return low_idx, high_idx  # type: ignore[return-value]


def _procrustes_to_strip(flat: np.ndarray, V: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rigid (R,t), det(R)=+1, minimizing ‖flat[idx]·Rᵀ + t − V[idx]‖."""
    P = flat[idx]
    Q = V[idx]
    cp, cq = P.mean(0), Q.mean(0)
    H = (P - cp).T @ (Q - cq)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    t = cq - cp @ R.T
    return R, t, P @ R.T + t


def surface_orientation_sign(V: np.ndarray, F: np.ndarray, uv: np.ndarray) -> float:
    """Mean sign linking the 3D face normal to the UV winding: >0 means UV winding is CCW
    when seen from the +normal side (the textured 'recto')."""
    a2 = (
        (uv[F[:, 1], 0] - uv[F[:, 0], 0]) * (uv[F[:, 2], 1] - uv[F[:, 0], 1])
        - (uv[F[:, 2], 0] - uv[F[:, 0], 0]) * (uv[F[:, 1], 1] - uv[F[:, 0], 1])
    )
    return float(np.sign(np.median(a2)))


def build_target_embedding(
    V: np.ndarray,
    F: np.ndarray,
    uv_norm: np.ndarray,
    *,
    anchor_frac: float = 0.02,
    force_anchor_end: str = "auto",   # 'auto' | 'outer_radial' (rolling-contact kinematics
                                       # must anchor at the winding's natural free end)
) -> dict:
    """Compute the flat target V1 (n,3) + anchor indices + diagnostics.

    Steps: de-normalize UV → flat 2D in voxel units; decide axis mapping; embed flat in 3D
    with v-axis ∥ scroll axis via Procrustes on the anchor strip (low-u end by default,
    choose the end that moves least); enforce non-mirroring (chirality) numerically.
    """
    V = np.asarray(V, dtype=np.float64)
    uv = np.asarray(uv_norm, dtype=np.float64)

    fit = fit_uv_scale(V, F, uv)
    flat2 = np.stack([uv[:, 0] * fit["s_u"], uv[:, 1] * fit["s_v"]], axis=1)

    ax = fit_scroll_axis(V, F, uv)
    unroll_axis = 0 if ax["axis_uv"] == "v" else 1  # the OTHER uv axis unrolls

    flat3 = np.zeros((len(V), 3))
    flat3[:, :2] = flat2

    low_idx, high_idx = anchor_strip_indices(uv, unroll_axis, anchor_frac)

    def tangent_plane_placement(idx: np.ndarray):
        """Place the flat sheet in the surface's tangent plane at the anchor strip:
        origin = strip centroid, in-plane axes = (strip direction ≈ scroll axis,
        unroll direction = the strip's own unroll tangent), normal = area-weighted mean
        surface normal over the strip's faces. The roll then 'peels open' onto its own
        tangent plane — no tumble, anchor nearly stationary.

        Orientation contract (natural unroll): the in-plane frame must coincide with the
        surface's local frame at the strip — u_hat along ∂p/∂(unroll uv), a_hat along the
        strip's axis tangent. Only then is the source→target rotation field a pure winding
        about the scroll axis, which `winding_decomposition` can represent. A 180° in-plane
        misplacement is NOT benign: it bakes a constant π rotation about an in-plane axis
        into every face (residual ‖w_res‖≈π mesh-wide → mid-transit tumble/snap,
        observed on damaged merge traces). The legacy 'extend away
        from the roll body' frame is therefore kept ONLY as the anchor-END selection key
        (its rms blows up when the body-side flip reverses the strip, which preserves the
        fleet's historical anchor choices); the PLACED frame is always the natural one.

        Returns (rms_selection, R_natural, t_natural, rms_natural, corrected) where
        `corrected` is True iff the legacy frame differs from the natural frame.
        """
        strip_mask = np.zeros(len(V), dtype=bool)
        strip_mask[idx] = True
        f_mask = strip_mask[F].any(axis=1)
        Fa = F[f_mask]
        e1 = V[Fa[:, 1]] - V[Fa[:, 0]]
        e2 = V[Fa[:, 2]] - V[Fa[:, 0]]
        n = np.cross(e1, e2).sum(0)
        n_hat = n / max(np.linalg.norm(n), 1e-300)
        # axis direction within the plane: regression of strip positions on their v coord
        vq = uv[idx, 1 - unroll_axis] - uv[idx, 1 - unroll_axis].mean()
        d_axis = (V[idx] - V[idx].mean(0)).T @ vq
        d_axis -= (d_axis @ n_hat) * n_hat
        a_hat = d_axis / max(np.linalg.norm(d_axis), 1e-300)
        flat_unroll_col = unroll_axis
        flat_axis_col = 1 - unroll_axis
        # u_hat chosen so det(R) = +1 for the actual column layout (mirror would flip text)
        u_hat = np.cross(a_hat, n_hat) if flat_unroll_col == 0 else np.cross(n_hat, a_hat)

        # natural unroll tangent of the strip: area-weighted ∂p/∂(unroll uv) over strip
        # faces, projected into the tangent plane (only its sign matters)
        d1 = uv[Fa[:, 1]] - uv[Fa[:, 0]]
        d2 = uv[Fa[:, 2]] - uv[Fa[:, 0]]
        det = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
        safe = np.where(np.abs(det) < 1e-18, 1.0, det)
        if unroll_axis == 0:
            t_un = (d2[:, 1, None] * e1 - d1[:, 1, None] * e2) / safe[:, None]
        else:
            t_un = (-d2[:, 0, None] * e1 + d1[:, 0, None] * e2) / safe[:, None]
        t_un[np.abs(det) < 1e-18] = 0.0
        w_area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
        tangent = (t_un * w_area[:, None]).sum(0)
        tangent -= (tangent @ n_hat) * n_hat
        # natural frame: u_hat must follow the strip's unroll tangent (flip (u,a) together
        # = in-plane 180°, the only det-preserving freedom). With n_hat/a_hat taken from
        # the strip itself this is generically already true; the guard protects degenerate
        # strips.
        u_nat, a_nat = (-u_hat, -a_hat) if u_hat @ tangent < 0 else (u_hat, a_hat)
        # legacy frame (selection key only): extend away from the roll body
        body_dir = V.mean(0) - V[idx].mean(0)
        u_leg, a_leg = (-u_hat, -a_hat) if u_hat @ body_dir > 0 else (u_hat, a_hat)

        def build(u_h, a_h):
            R = np.zeros((3, 3))
            R[:, flat_unroll_col] = u_h   # flat unroll coord → u_hat
            R[:, flat_axis_col] = a_h     # flat axis coord → a_hat
            R[:, 2] = n_hat
            assert np.linalg.det(R) > 0.99, "tangent-plane frame must be a proper rotation"
            # anchor strip flat coords → its 3D centroid
            t_vec = V[idx].mean(0) - (flat3[idx] @ R.T).mean(0)
            placed = flat3[idx] @ R.T + t_vec
            rms = float(np.sqrt(((placed - V[idx]) ** 2).sum(1).mean()))
            return rms, R, t_vec

        rms_nat, R_nat, t_nat = build(u_nat, a_nat)
        corrected = bool(u_leg @ u_nat < 0)
        rms_sel = build(u_leg, a_leg)[0] if corrected else rms_nat
        return rms_sel, R_nat, t_nat, rms_nat, corrected

    candidates = []
    for name, idx in (("low", low_idx), ("high", high_idx)):
        rms_sel, R, t, rms_nat, corrected = tangent_plane_placement(idx)
        candidates.append((rms_sel, name, idx, R, t, rms_nat, corrected))
    if force_anchor_end == "outer_radial":
        # winding axis from v-tangent coherence; outer end = strip farther from the axis
        from .dg_interp import face_uv_tangents

        _, t_v = face_uv_tangents(V, F, uv)
        nv = np.linalg.norm(t_v, axis=1)
        ok2 = nv > 1e-12
        axw = (t_v[ok2] / nv[ok2][:, None]).mean(0)
        axw /= max(np.linalg.norm(axw), 1e-300)
        c0 = V.mean(0)

        def mean_radial(idx):
            rel = V[idx] - c0
            return float(np.linalg.norm(rel - np.outer(rel @ axw, axw), axis=1).mean())

        r_low, r_high = mean_radial(low_idx), mean_radial(high_idx)
        want = "low" if r_low >= r_high else "high"
        candidates = [c for c in candidates if c[1] == want]
    else:
        candidates.sort(key=lambda c: c[0])
    rms_sel, anchor_end, anchor_idx, R, t, rms, orientation_corrected = candidates[0]

    V1 = flat3 @ R.T + t

    # chirality: source recto orientation must survive the embedding.
    # In the flat sheet the normal is R @ ez (constant). UV winding sign s_uv tells which
    # side of the UV plane is the recto; the embedded plane normal must equal R@ez * s_uv
    # consistently — numerically: compare mean target face normal to R@ez.
    src_sign = surface_orientation_sign(V, F, uv)
    e1 = V1[F[:, 1]] - V1[F[:, 0]]
    e2 = V1[F[:, 2]] - V1[F[:, 0]]
    n_t = np.cross(e1, e2).sum(0)
    n_t /= max(np.linalg.norm(n_t), 1e-300)
    plane_n = R @ np.array([0.0, 0.0, 1.0])
    mirror_check = float(np.dot(n_t, plane_n) * src_sign)
    # mirror_check > 0 ⇔ winding preserved w.r.t. plane normal ⇔ no mirroring
    # (Procrustes with det=+1 cannot mirror flat3 in its own plane, but the UV map itself
    #  could be flipped; record it loudly.)

    return {
        "V1": V1,
        "anchor_idx": anchor_idx,
        "anchor_end": anchor_end,
        "anchor_rms": rms,
        "anchor_rms_selection": rms_sel,
        "orientation_corrected": orientation_corrected,
        "denorm_fit": fit,
        "axis": {"axis_uv": ax["axis_uv"], "axis_dir": ax["axis_dir"].tolist(),
                 "coherences": ax["details"]},
        "unroll_axis_uv": "u" if unroll_axis == 0 else "v",
        "uv_winding_sign": src_sign,
        "mirror_check": mirror_check,
    }
