"""Roll-vs-settled-sheet clearance certificate (anti z-fight / anti pierce-through).

Z-fighting or pop-through needs a second surface within ~depth precision of the
settled flat sheet at the same plan position. Per frame:
  1. fit the unroll plane from the final (flat) frame,
  2. mark PLAN-RESIDENT vertices (current plan position ~= final plan position;
     height deliberately ignored — the settled sheet legitimately rides the
     clearance ramp far above the plane near the contact),
  3. classify by FACE: sheet = faces with all 3 vertices resident; hover = faces
     with all 3 non-resident. Face-level classification is what kills both false
     positive families: a roll vertex coincidentally revisiting its final (u,v)
     mid-flight has non-resident neighbours (not sheet), and ANY height-threshold
     split would slice a smooth surface (roll flank or clearance ramp) into a
     fake sheet/hover pair within one cell. Mixed faces are the landing flap —
     single material mid-blend — and count as neither,
  4. bin on a plan grid (cell ~ 2 median edges): margin per cell =
     min(hover-vertex height) - max(sheet-vertex height).
Cells within band_pad of the contact line are exempt (flap territory). Ahead of
the contact there is no settled sheet, so nothing pairs. Intra-roll winding gaps
are rigid-transported source geometry (intersection-free at source) and are
deliberately not measured.

Pass: every frame's worst margin >= margin_frac * median_edge (default 0.05 —
an order of magnitude above 24-bit depth resolution at our camera ranges).
"""

from __future__ import annotations

import numpy as np

__all__ = ["clearance_certificate"]


def clearance_certificate(T: np.ndarray, F: np.ndarray, margin_frac: float = 0.05,
                          per_frame: bool = False,
                          s: np.ndarray | None = None,
                          c_arc: np.ndarray | None = None,
                          pad_arc: float | None = None) -> dict:
    """T: (n_frames, n_verts, 3) trajectory whose LAST frame is the flat sheet.

    EXACT-STATE MODE (s, c_arc, pad_arc given — rolling kinematics): per-vertex
    landing state is known, not inferred: settled <=> s < c - pad, rolled <=>
    s > c + pad, |s - c| <= pad = the landing flap (single material, exempt).
    Plan-residency inference CANNOT classify near the contact — roll faces
    descending onto their landing slots are plan-resident whole faces and read
    as 'sheet' at roll height, displacing the contact estimate.
    Heuristic mode (state omitted) keeps face-residency classification and is
    only used for non-rolling (DG fallback) trajectories."""
    n_frames = len(T)
    flat = T[-1]
    ctr = flat.mean(0)
    _, _, Vt = np.linalg.svd(flat - ctr, full_matrices=False)
    n_pl = Vt[2] / np.linalg.norm(Vt[2])
    if (T[0].mean(0) - ctr) @ n_pl < 0:
        n_pl = -n_pl                       # toward the roll side
    u_pl, v_pl = Vt[0], Vt[1]

    e = T[0][F[:, 1]] - T[0][F[:, 0]]
    med_edge = float(np.median(np.linalg.norm(e, axis=1)))
    cell = 2.0 * med_edge
    margin_min = margin_frac * med_edge

    pu_fin = (flat - ctr) @ u_pl
    pv_fin = (flat - ctr) @ v_pl
    S_est = float(pu_fin.max() - pu_fin.min())
    band_pad = 0.02 * S_est + 6.0 * med_edge

    # payout sign along pu: the sheet grows AWAY from the anchor (the material already
    # plan-resident earliest). The SVD axis sign is arbitrary — assuming +pu measured the
    # contact at the ANCHOR end and exempted the wrong end (payout can run toward -pu).
    anchor_sign = 0.0
    for k_ref in range(n_frames):
        P = T[k_ref]
        res0 = ((np.abs((P - ctr) @ u_pl - pu_fin) < med_edge)
                & (np.abs((P - ctr) @ v_pl - pv_fin) < med_edge))
        if res0.sum() > 0.005 * len(P):
            anchor_sign = float(np.sign(pu_fin[res0].mean() - pu_fin.mean()))
            break
    if anchor_sign == 0.0:
        anchor_sign = 1.0
    payout = -anchor_sign          # direction (in pu) in which the contact line travels

    exact = s is not None and c_arc is not None and pad_arc is not None
    if exact:
        s = np.asarray(s, dtype=np.float64)
        c_arc = np.asarray(c_arc, dtype=np.float64)
        state_pad = float(pad_arc)

    rows: list[dict] = []
    worst = {"frame": -1, "margin": float("inf")}
    overlap_frames = 0
    for k in range(n_frames):
        P = T[k]
        h = (P - ctr) @ n_pl
        pu = (P - ctr) @ u_pl
        pv = (P - ctr) @ v_pl
        if exact:
            c = float(c_arc[k])
            sheet_eval = s < c - state_pad
            hover_v = s > c + state_pad
        else:
            resident = ((np.abs(pu - pu_fin) < 0.5 * cell)
                        & (np.abs(pv - pv_fin) < 0.5 * cell))
            res_f = resident[F]
            sheet_v = np.zeros(len(P), bool)
            sheet_v[F[res_f.all(axis=1)]] = True
            hover_v = np.zeros(len(P), bool)
            hover_v[F[(~res_f).all(axis=1)]] = True
            if not sheet_v.any() or not hover_v.any():
                if per_frame:
                    rows.append({"frame": k, "margin": None, "cells": 0})
                continue
            contact_u = float((pu_fin[sheet_v] * payout).max()) * payout
            sheet_eval = sheet_v & ((contact_u - pu) * payout > band_pad)
        if not sheet_eval.any() or not hover_v.any():
            if per_frame:
                rows.append({"frame": k, "margin": None, "cells": 0})
            continue
        iu = np.floor(pu / cell).astype(np.int64)
        iv = np.floor(pv / cell).astype(np.int64)
        width = np.int64(iv.max() - iv.min()) + 2
        key = (iu - iu.min()) * width + (iv - iv.min())

        def cell_reduce(mask: np.ndarray, vals: np.ndarray, op):
            kk, vv = key[mask], vals[mask]
            order = np.argsort(kk, kind="stable")
            kk, vv = kk[order], vv[order]
            starts = np.r_[0, np.flatnonzero(np.diff(kk)) + 1]
            return kk[starts], op.reduceat(vv, starts)

        sheet_cells, sheet_top = cell_reduce(sheet_eval, h, np.maximum)
        hover_cells, hover_bot = cell_reduce(hover_v, h, np.minimum)
        common, si, hi = np.intersect1d(sheet_cells, hover_cells, return_indices=True)
        if len(common) == 0:
            if per_frame:
                rows.append({"frame": k, "margin": None, "cells": 0})
            continue
        overlap_frames += 1
        margins = hover_bot[hi] - sheet_top[si]
        m = float(margins.min())
        if per_frame:
            rows.append({"frame": k, "margin": m, "cells": int(len(common))})
        if m < worst["margin"]:
            worst = {"frame": k, "margin": m}

    out = {
        "n_frames": n_frames,
        "med_edge": med_edge,
        "margin_min_required": margin_min,
        "worst_frame": worst["frame"],
        "worst_margin": (None if worst["frame"] < 0 else worst["margin"]),
        "frames_with_overlap": overlap_frames,
        "pass": bool(worst["frame"] < 0 or worst["margin"] >= margin_min),
    }
    if per_frame:
        out["per_frame"] = rows
    return out
