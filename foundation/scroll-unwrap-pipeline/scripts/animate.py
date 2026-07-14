#!/usr/bin/env python
"""M3: compute the unwrap trajectory for one mesh, with full metric evaluation.

Outputs under outputs/anim/<stem>/:
  frames.npz      float32 vertex positions per frame (compressed) + faces + uv + meta
  metrics.json    per-frame gate metrics + embedding/unwrap diagnostics + gate verdicts
  metrics.png     plots (area/edge deviation envelopes, SD, displacement)

Usage: uv run python scripts/animate.py <mesh.ply|mesh.obj> [--frames 240] [--substeps 1]
       [--solver auto] [--out outputs/anim]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from scrollkit.io import read_obj, read_ply  # noqa: E402
from scrollkit.metrics import flatboi_stretch_metrics, frame_metrics  # noqa: E402
from scrollkit.metrics.visibility import visible_face_mask  # noqa: E402
from scrollkit.unwrap.dg_interp import build_unwrap_path  # noqa: E402
from scrollkit.unwrap.embedding import build_target_embedding  # noqa: E402
from scrollkit.unwrap.operators import face_areas  # noqa: E402
from scrollkit.unwrap.timeline import timeline_t  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--hold-start", type=int, default=20)
    ap.add_argument("--hold-end", type=int, default=25)
    ap.add_argument("--substeps", type=int, default=1)
    ap.add_argument("--solver", default="auto")
    ap.add_argument("--axis-mode", default="auto", choices=["auto", "global", "smooth", "local"])
    ap.add_argument("--kinematics", default="rolling", choices=["rolling", "dg"])
    ap.add_argument("--roll-smooth", type=int, default=21)
    ap.add_argument("--out", default="outputs/anim")
    ap.add_argument("--full-metrics-stride", type=int, default=5)
    args = ap.parse_args()

    src = Path(args.mesh)
    mesh = read_ply(src) if src.suffix.lower() == ".ply" else read_obj(src)
    if mesh.vertex_uv is None:
        raise SystemExit("mesh has no per-vertex UV — unwrap needs the A/B layout")
    stem = src.stem.replace("_decimated", "")
    out_dir = ROOT / args.out / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # Junk-tail trim (animation inputs only — conversion deliverables stay full).
    # Low-confidence trace tails have squeezed/warped charts: they transport
    # material through the settled sheet, produce warp-seam slivers, and render
    # border-unreliable content as a compressed column at the flat sheet's edge.
    # See scrollkit.unwrap.trim.
    from dataclasses import replace as _dc_replace

    from scrollkit.unwrap.trim import enforce_chart_injectivity, junk_tail_trim

    # 1) chart injectivity: duplicated trace patches render the same texture region
    #    twice (offset double-printed text, layer-crossing flicker while landing,
    #    detached flying patches) — drop the layer deviating from local consensus.
    _inj = enforce_chart_injectivity(mesh.vertices.astype(np.float64), mesh.faces,
                                     mesh.vertex_uv.astype(np.float64))
    inj_info = _inj["info"]
    if inj_info["n_faces_dropped"]:
        keep = _inj["vert_keep"]
        mesh = _dc_replace(
            mesh,
            vertices=_inj["V"].astype(np.float32),
            faces=_inj["F"].astype(np.int32),
            vertex_uv=_inj["uv"].astype(np.float32),
            normals=mesh.normals[keep] if mesh.normals is not None else None,
            wedge_uv=None, face_texnumber=None,
        )
        print(f"[{stem}] chart-injectivity: dropped {inj_info['n_verts_dropped']} verts / "
              f"{inj_info['n_faces_dropped']} faces (duplicated patch)")

    # 2) junk-tail trim (data-driven; no-op on healthy charts)
    _tr = junk_tail_trim(mesh.vertices.astype(np.float64), mesh.faces,
                         mesh.vertex_uv.astype(np.float64))
    trim_info = _tr["info"]
    trim_info["injectivity"] = inj_info
    if trim_info["n_faces_dropped"]:
        keep = _tr["vert_keep"]
        mesh = _dc_replace(
            mesh,
            vertices=_tr["V"].astype(np.float32),
            faces=_tr["F"].astype(np.int32),
            vertex_uv=_tr["uv"].astype(np.float32),
            normals=mesh.normals[keep] if mesh.normals is not None else None,
            wedge_uv=None, face_texnumber=None,
        )
        print(f"[{stem}] junk-tail trim: dropped {trim_info['n_faces_dropped']} faces "
              f"({100 * trim_info['frac_faces_dropped']:.2f}%), keep u in "
              f"[{trim_info['u_keep'][0]:.4f}, {trim_info['u_keep'][1]:.4f}] "
              f"(bins lo/hi {trim_info['trim_lo_bins']}/{trim_info['trim_hi_bins']})")

    t0 = time.time()
    V0 = mesh.vertices.astype(np.float64)
    F = mesh.faces
    emb = build_target_embedding(V0, F, mesh.vertex_uv)
    print(f"[{stem}] embedding: axis_uv={emb['axis']['axis_uv']} "
          f"coh={emb['axis']['coherences']} anchor={emb['anchor_end']} rms={emb['anchor_rms']:.3f} "
          f"mirror={emb['mirror_check']:.3f} denorm_resid={emb['denorm_fit']['rel_residual_rms']:.4f}")
    if emb["mirror_check"] <= 0:
        raise SystemExit("CHIRALITY FAILURE: embedding mirrors the texture — aborting")

    # visibility mask: texture alpha at face UV centroids, audit pixel convention per group
    visible = visible_face_mask(mesh, src, ROOT)
    vert_visible = np.zeros(len(V0), dtype=bool)
    vert_visible[F[visible].ravel()] = True
    print(f"[{stem}] visible faces: {100 * visible.mean():.1f}%")

    A0 = face_areas(V0, F)
    A1 = face_areas(emb["V1"], F)

    from scrollkit.metrics import symmetric_dirichlet

    resid = float(emb["denorm_fit"]["rel_residual_rms"])
    area_gate = max(0.02, 3.0 * resid)
    area_p95_gate = area_gate  # qa-gates M3: P95 ≤ max(2%, 3 × de-norm rel_residual_rms)
    sd1 = symmetric_dirichlet(V0, emb["V1"], F, face_mask=visible)
    sd_gate = max(1.0, sd1) * 1.15 + 0.05

    ts = timeline_t(args.frames, args.hold_start, args.hold_end)

    def build_path(junk_weight: float, axis_from_conf: bool):
        if args.axis_mode == "auto":
            # gate-aware A/B: a mode must keep BOTH the area quantile and the SD envelope
            # inside gates on sampled frames; among passers pick lower area, else lower SD.
            cands = []
            for mode in ("global", "smooth"):
                cand = build_unwrap_path(V0, emb["V1"], F, mesh.vertex_uv.astype(np.float64),
                                         emb["anchor_idx"], solver_backend=args.solver,
                                         axis_mode=mode, face_conf=visible,
                                         junk_weight=junk_weight, axis_from_conf=axis_from_conf,
                                         repo_root=ROOT)
                worst_area, worst_sd = 0.0, 0.0
                for tt in (0.35, 0.5, 0.65, 0.8):
                    Vt = cand.eval_frame(tt)
                    At = face_areas(Vt, F)
                    blend = (1 - tt) * A0 + tt * A1
                    rel = (np.abs(At - blend) / np.maximum(blend, 1e-300))[visible]
                    worst_area = max(worst_area, float(np.quantile(rel, 0.95)))
                    worst_sd = max(worst_sd, symmetric_dirichlet(V0, Vt, F, face_mask=visible))
                passes = worst_area <= area_gate and worst_sd <= sd_gate
                print(f"[{stem}] axis-mode A/B {mode}: area_p95 {worst_area:.4f} (gate {area_gate:.3f}) "
                      f"sd {worst_sd:.3f} (gate {sd_gate:.3f}) -> {'pass' if passes else 'fail'}")
                cands.append((cand, mode, worst_area, worst_sd, passes))
            passing = [c for c in cands if c[4]]
            pick = min(passing, key=lambda c: c[2]) if passing else min(cands, key=lambda c: c[3])
            return pick[0], pick[1]
        cand = build_unwrap_path(V0, emb["V1"], F, mesh.vertex_uv.astype(np.float64),
                                 emb["anchor_idx"], solver_backend=args.solver,
                                 axis_mode=args.axis_mode, face_conf=visible,
                                 junk_weight=junk_weight, axis_from_conf=axis_from_conf,
                                 repo_root=ROOT)
        return cand, args.axis_mode

    def evaluate(path, chosen_mode, ref_mode="blend"):
        """Full frame loop + M3 gates for one configured path."""
        uniq_t = np.unique(ts)
        solved: dict[float, np.ndarray] = {}
        if args.substeps > 1:
            from scrollkit.unwrap.dg_interp import SubsteppedPath

            path = SubsteppedPath(path, args.substeps)
            print(f"[{stem}] substepped integration: K={args.substeps}")

        prev_normals = None
        prev_V = None
        per_frame = []
        for k, t in enumerate(uniq_t):
            Vt = solved.get(float(t))
            if Vt is None:
                Vt = path.eval_frame(float(t))
                solved[float(t)] = Vt
            excl = path.transition_face_mask(float(t)) if hasattr(path, "transition_face_mask") else None
            fm = frame_metrics(Vt, float(t), V0, emb["V1"], F, A0, A1,
                               N_blend_ref=prev_normals, visible=visible, ref_mode=ref_mode,
                               flip_exclude=excl)
            full = (k % args.full_metrics_stride == 0) or t in (0.0, 0.25, 0.5, 0.75, 1.0)
            if full:
                uvphys = np.stack([mesh.vertex_uv[:, 0] * emb["denorm_fit"]["s_u"],
                                   mesh.vertex_uv[:, 1] * emb["denorm_fit"]["s_v"]], axis=1)
                fm["flatboi_vs_uv"] = flatboi_stretch_metrics(Vt, uvphys, F)
            if prev_V is not None:
                d = np.linalg.norm(Vt - prev_V, axis=1)
                fm["disp_p99"] = float(np.quantile(d[vert_visible], 0.99))
                fm["disp_max"] = float(d.max())
            e1 = Vt[F[:, 1]] - Vt[F[:, 0]]
            e2 = Vt[F[:, 2]] - Vt[F[:, 0]]
            n = np.cross(e1, e2)
            prev_normals = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)
            prev_V = Vt
            per_frame.append(fm)
            if k % 10 == 0:
                print(f"  t={t:.3f} area_p95={fm['area_rel_p95']:.4f} edge_p95={fm['edge_rel_p95']:.4f} "
                      f"SD={fm['sym_dirichlet_vs_source']:.4f}")

        # assemble full frame array following the timeline (holds reuse solved endpoints)
        frames = np.stack([solved[float(t)].astype(np.float32) for t in ts])

        # gates (M3, residual-scaled — see docs/QUALITY-GATES.md for thresholds)
        morph = [f for f in per_frame if 0.0 < f["t"] < 1.0]
        sd_endpoints = max(per_frame[0]["sym_dirichlet_vs_source"], per_frame[-1]["sym_dirichlet_vs_source"])
        gates = {
            "finite_all": all(f["finite"] for f in per_frame),
            "degenerate_max": max(f["degenerate_tris"] for f in per_frame),
            "flipped_frac_max": max(f["flipped_tris"] for f in per_frame) / len(F),
            "area_p95_vis_max": max(f["area_rel_p95_vis"] for f in morph),
            "edge_p95_vis_max": max(f["edge_rel_p95_vis"] for f in morph),
            "area_p95_all_max": max(f["area_rel_p95"] for f in morph),
            "soft_area_p999_vis_max": max(f["area_rel_p999_vis"] for f in morph),
            "soft_area_max_max": max(f["area_rel_max"] for f in morph),
            "sd_peak": max(f["sym_dirichlet_vs_source"] for f in per_frame),
            "sd_peak_sustained": float(max(
                min(per_frame[j]["sym_dirichlet_vs_source"]
                    for j in range(max(0, i - 1), min(len(per_frame), i + 2)))
                for i in range(len(per_frame)))),
            "area_p95_gate": area_p95_gate,
            # envelope on the SUSTAINED level (3-frame rolling minimum): metric transients
            # shorter than ~100 ms with clean flips/areas are imperceptible; the failure
            # modes of record (rotation seams etc.) violate for dozens of frames.
            "sd_envelope_ok": float(max(
                min(per_frame[j]["sym_dirichlet_vs_source"]
                    for j in range(max(0, i - 1), min(len(per_frame), i + 2)))
                for i in range(len(per_frame)))) <= sd_endpoints * 1.15 + 0.05,
            "disp_jump_ok": True,
            "visible_frac": float(visible.mean()),
            "axis_mode": chosen_mode,
        }
        # popping = a DISCONTINUITY in the displacement sequence, not the easing's designed
        # speed variation (quintic peak/median ≈ 3 by construction). Smooth runs have
        # consecutive deltas ≪ median; a snap is O(median).
        disps = [f.get("disp_p99") for f in per_frame if f.get("disp_p99") is not None]
        if len(disps) > 3:
            med = float(np.median(disps))
            peak = float(np.max(disps))
            deltas = np.abs(np.diff(disps))
            gates["disp_delta_max_ratio"] = float(deltas.max() / max(med, 1e-9))
            gates["disp_delta_peak_ratio"] = float(deltas.max() / max(peak, 1e-9))
            # median under-normalizes motion-concentrated kinematics (rolling parks
            # most vertices). Thresholds are calibrated for the flap-bridge kinematics:
            # the bridge adds legitimate smooth flap motion that inflates deltas —
            # visually-clean reference runs measure 0.33-0.56 med-ratio / <=0.26
            # peak-ratio, while true junction snaps measure 2.9-3.8 / 0.35-0.47.
            # The thresholds sit in that gap with margin on both sides.
            gates["disp_jump_ok"] = bool(deltas.max() <= max(0.7 * med, 0.30 * peak) + 1e-9)
        # clearance certificate: nothing may hover/pierce within depth-precision of
        # the settled sheet. Quantile gates are blind to sub-percent vertex
        # populations (a few hundred vertices can travel through the sheet while
        # every P95/flip/SD metric stays green), so this is a per-frame certificate.
        # Rolling paths certify in EXACT-STATE mode (per-vertex arc + contact
        # schedule); DG fallback paths use the residency heuristic.
        from scrollkit.metrics.clearance import clearance_certificate
        base_path = getattr(path, "base", path)
        cl_kw = {}
        if hasattr(base_path, "pad_arc") and hasattr(base_path, "_contact_arc"):
            cl_kw = {"s": base_path.s, "pad_arc": base_path.pad_arc,
                     "c_arc": np.array([base_path._contact_arc(float(t)) for t in ts])}
        cl = clearance_certificate(frames.astype(np.float64), F, **cl_kw)
        gates["clearance_worst_margin"] = cl["worst_margin"]
        gates["clearance_worst_frame"] = cl["worst_frame"]
        gates["clearance_ok"] = cl["pass"]
        gates["n_hard_failures"] = (
            int(not gates["finite_all"]) + int(gates["flipped_frac_max"] > 1e-4)
            + int(gates["area_p95_vis_max"] > area_p95_gate)
            + int(gates["edge_p95_vis_max"] > area_p95_gate)
            + int(not gates["sd_envelope_ok"]) + int(not gates["disp_jump_ok"])
            + int(not gates["clearance_ok"])
        )
        gates["pass"] = gates["n_hard_failures"] == 0
        return path, frames, per_frame, gates

    # Kinematics rung -1: rolling-contact ("carpet") unroll — self-intersection-free by
    # construction (the loosening-spiral DG transit lets inner windings pop
    # through outer layers). The rolled part moves RIGIDLY (zero transit distortion); falls
    # through to the DG ladder only if its gates fail.
    attempts = []
    best = None
    if args.kinematics == "rolling":
        from scrollkit.unwrap.embedding import build_target_embedding as _bte
        from scrollkit.unwrap.rolling import build_rolling_path

        emb_roll = _bte(V0, F, mesh.vertex_uv, force_anchor_end="outer_radial")
        if emb_roll["mirror_check"] <= 0:
            raise SystemExit("CHIRALITY FAILURE on outer-anchored embedding")
        unroll_axis = 0 if emb_roll["unroll_axis_uv"] == "u" else 1
        s_scale = emb_roll["denorm_fit"]["s_u" if unroll_axis == 0 else "s_v"]
        rpath = build_rolling_path(V0, emb_roll["V1"], F,
                                   mesh.vertex_uv.astype(np.float64), s_scale,
                                   unroll_axis=unroll_axis,
                                   outer_is_low_u=(emb_roll["anchor_end"] == "low"),
                                   smooth_sigma=args.roll_smooth)
        print(f"[{stem}] rolling-contact path: {rpath.info}")
        emb = emb_roll  # metrics/gates measure against the outer-anchored target
        A0 = face_areas(V0, F)
        A1 = face_areas(emb["V1"], F)
        _, frames, per_frame, gates = evaluate(rpath, "rolling", ref_mode="step")
        attempts.append({"rung": -1, "kinematics": "rolling",
                         "pass": gates["pass"], "n_hard_failures": gates["n_hard_failures"],
                         "sd_peak": gates["sd_peak"], "sd_envelope_ok": gates["sd_envelope_ok"],
                         "clearance_ok": gates["clearance_ok"],
                         "clearance_worst_margin": gates["clearance_worst_margin"],
                         "area_p95_vis_max": gates["area_p95_vis_max"],
                         "edge_p95_vis_max": gates["edge_p95_vis_max"],
                         "flipped_frac_max": gates["flipped_frac_max"],
                         "disp_jump_ok": gates["disp_jump_ok"],
                         "disp_delta_max_ratio": gates.get("disp_delta_max_ratio"),
                         "disp_delta_peak_ratio": gates.get("disp_delta_peak_ratio"),
                         **{f"roll_{k}": v for k, v in rpath.info.items() if k != "kinematics"}})
        # Best-attempt rank: CLEARANCE DOMINATES — interpenetration (through-surface
        # ghosting) is the never-acceptable artifact class; a pacing blemish is not.
        # Without this, a poppy-but-clean rolling (4 hard) would lose to a DG fallback
        # that pierces the sheet (1 hard) and the worst artifact would ship.
        if gates["pass"]:
            best = ((0, 0, gates["sd_peak"]), {"kinematics": "rolling"}, rpath, "rolling",
                    frames, per_frame, gates)
        else:
            print(f"[{stem}] rolling kinematics FAILED gates "
                  f"({gates['n_hard_failures']} hard) -> falling back to DG ladder")
            best = ((int(not gates["clearance_ok"]), gates["n_hard_failures"],
                     gates["sd_peak"]), {"kinematics": "rolling"},
                    rpath, "rolling", frames, per_frame, gates)

    # Escalation ladder (gate-driven): rung 0 is the production default — meshes that
    # pass it are untouched by the ladder. Damaged merged traces escalate to harder
    # junk suppression (the knobs only change how UNTRUSTED faces are treated; gates and
    # thresholds are identical on every rung).
    rungs = [
        {"junk_weight": 0.05, "axis_from_conf": False},
        {"junk_weight": 0.005, "axis_from_conf": False},
        {"junk_weight": 0.05, "axis_from_conf": True},
        {"junk_weight": 0.005, "axis_from_conf": True},
    ]
    if best is not None and best[6]["pass"]:
        rungs = []  # rolling kinematics passed — DG ladder not needed
    for r_i, knobs in enumerate(rungs):
        print(f"[{stem}] rung {r_i}: {knobs}")
        path, chosen_mode = build_path(**knobs)
        print(f"[{stem}] axis_mode={chosen_mode} path: {path.info}")
        path, frames, per_frame, gates = evaluate(path, chosen_mode)
        attempts.append({"rung": r_i, **knobs, "axis_mode": chosen_mode,
                         "pass": gates["pass"], "n_hard_failures": gates["n_hard_failures"],
                         "sd_peak": gates["sd_peak"],
                         "clearance_ok": gates["clearance_ok"],
                         "clearance_worst_margin": gates["clearance_worst_margin"],
                         "area_p95_vis_max": gates["area_p95_vis_max"],
                         "disp_delta_max_ratio": gates.get("disp_delta_max_ratio")})
        rank = (int(not gates["clearance_ok"]), gates["n_hard_failures"], gates["sd_peak"])
        if best is None or rank < best[0]:
            best = (rank, knobs, path, chosen_mode, frames, per_frame, gates)
        if gates["pass"]:
            break
        print(f"[{stem}] rung {r_i} FAILED ({gates['n_hard_failures']} hard failures, "
              f"sd_peak {gates['sd_peak']:.3f}) -> escalating")
    _, knobs, path, chosen_mode, frames, per_frame, gates = best

    meta = {
        "mesh": str(src), "stem": stem, "n_vertices": int(len(V0)), "n_faces": int(len(F)),
        "trim": trim_info,
        "frames": int(args.frames), "embedding": {k: v for k, v in emb.items() if k not in ("V1", "anchor_idx")},
        "anchor_n": int(len(emb["anchor_idx"])),
        "path_info": (path.base if hasattr(path, "base") else path).info,
        "solver": getattr(getattr(path, "base", path), "solver", None).backend
                  if getattr(getattr(path, "base", path), "solver", None) else "rigid-kinematics",
        "escalation": {"chosen": knobs, "attempts": attempts},
        "gates": gates, "seconds": round(time.time() - t0, 1),
    }
    state_kw = {}
    bp = getattr(path, "base", path)
    if hasattr(bp, "pad_arc") and hasattr(bp, "_contact_arc"):
        state_kw = {"s_arc": bp.s.astype(np.float32),
                    "c_arc": np.array([bp._contact_arc(float(t)) for t in ts], np.float32),
                    "pad_arc": np.float32(bp.pad_arc)}
    np.savez_compressed(out_dir / "frames.npz", frames=frames, faces=F,
                        uv=mesh.vertex_uv, t=ts.astype(np.float32), **state_kw)
    (out_dir / "metrics.json").write_text(json.dumps(
        {"meta": meta, "per_frame": per_frame}, indent=1, default=float))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        tt = [f["t"] for f in per_frame]
        fig, axs = plt.subplots(2, 2, figsize=(11, 7))
        axs[0, 0].plot(tt, [f["area_rel_p95"] for f in per_frame], label="P95")
        axs[0, 0].plot(tt, [f["area_rel_max"] for f in per_frame], label="max", alpha=0.6)
        axs[0, 0].axhline(0.02, ls="--", c="g"); axs[0, 0].axhline(0.05, ls="--", c="r")
        axs[0, 0].set_title("area deviation vs endpoint blend"); axs[0, 0].legend()
        axs[0, 1].plot(tt, [f["edge_rel_p95"] for f in per_frame])
        axs[0, 1].axhline(0.02, ls="--", c="g"); axs[0, 1].set_title("edge length deviation P95")
        axs[1, 0].plot(tt, [f["sym_dirichlet_vs_source"] for f in per_frame])
        axs[1, 0].set_title("symmetric Dirichlet vs source (1=isometric)")
        axs[1, 1].plot(tt[1:], [f.get("disp_p99", np.nan) for f in per_frame][1:])
        axs[1, 1].set_title("per-frame displacement P99 (voxels)")
        for ax in axs.flat:
            ax.grid(alpha=0.3)
        fig.suptitle(f"{stem} — unwrap transit metrics ({'PASS' if gates['pass'] else 'FAIL'})")
        fig.tight_layout()
        fig.savefig(out_dir / "metrics.png", dpi=110)
    except Exception as e:  # plots are best-effort
        print("plot failed:", e)

    print(f"[{stem}] GATES: {'PASS' if gates['pass'] else 'FAIL'} {gates}")
    print(f"[{stem}] wrote {out_dir} in {meta['seconds']}s")
    return 0 if gates["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
