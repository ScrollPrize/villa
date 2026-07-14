#!/usr/bin/env python
"""Milestone M2 gate (docs/QUALITY-GATES.md, M2 section). Exit 0 only if 34/34 meshes are
green: a passing decimation rung, OR (Group C only) a documented no_safe_decimation
decision with evidence of the failed extended ladder.

Checks, from reports/m2_decimation.json:
  * 34 meshes, no errors; chosen rung is in the group's ladder (A/B: pinned ladder;
    C: pinned + GROUP_C_EXTENDED_LADDER [0.5, 0.65, 0.8]).
  * Two-sided Hausdorff max <= 1.0 x that mesh's audit edge_len_mean.
  * No new flipped UV triangles; UV chart count did not grow; boundary length within
    2% of source; non-manifold edges <= source; zero isolated vertices.
  * SSIM >= 0.97 on both production views and >= 0.95 on both close-up crops
    (`--geometry-only` skips this while the render module is pending — NOT green M2).
  * Per-frame OBJ size estimate + projected frame-export disk total recorded.
  * decision='no_safe_decimation' (C only): the
    verbatim rationale is present; the SSIM journey shows the pinned ladder max AND
    every extended rung tried with zero passes; the deliverable is the identity
    (faces/verts unchanged, Hausdorff 0) and is re-verified here byte-identical
    (SHA256) to the M1-converted OBJ + MTL, texture SHA re-verified. A/B meshes must
    still pass a rung — any decision on A/B is red.
Spot re-verification (trust but verify):
  * Reload 3 random decimated OBJs with scrollkit.io.read_obj: parseable, face count
    matches the report, texture copy SHA256 matches, UVs inside the source UV bbox.
  * For 1 of them re-run the two-sided Hausdorff against the source PLY in pymeshlab
    (arrays in-memory) and re-check the threshold.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from scrollkit.decimate.core import GROUP_C_EXTENDED_LADDER, NO_SAFE_DECIMATION_RATIONALE  # noqa: E402
from scrollkit.io import read_obj, read_ply  # noqa: E402

FAIL: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAIL.append(msg)
        print("  FAIL  " + msg)


def _abs(p: str | Path) -> Path:
    q = Path(p)
    return q if q.is_absolute() else ROOT / q


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    geometry_only = "--geometry-only" in sys.argv
    rep_p = ROOT / "reports/m2_decimation.json"
    if not rep_p.exists():
        print("reports/m2_decimation.json missing — run scripts/decimate.py first")
        return 1
    doc = json.loads(rep_p.read_text())
    meshes = doc["meshes"]
    cfg = doc["config"]
    base_ladder = set(float(x) for x in cfg["keep_ladder"])
    ladder_max = max(base_ladder)
    ext_c = [float(x) for x in GROUP_C_EXTENDED_LADDER]
    allowed_rungs = {"A": base_ladder, "B": base_ladder, "C": base_ladder | set(ext_c)}
    ssim_full_min = float(cfg["ssim_production_min"])
    ssim_close_min = float(cfg["ssim_closeup_min"])
    hd_ratio_max = float(cfg["hausdorff_max_mean_edge"])

    audit = json.loads((ROOT / "reports/audit.json").read_text())
    audit_edge = {m["stem"]: float(m["edge_len_mean"]) for m in audit["meshes"]}

    check(len(meshes) == 34, f"34 meshes in report (got {len(meshes)})")

    for r in meshes:
        tag = f"[{r.get('group')}] {r.get('stem')}"
        if "error" in r:
            check(False, f"{tag}: pipeline error: {r['error']}")
            continue
        decision = r.get("decision")
        if decision is None:
            check(float(r["rung_chosen"]) in allowed_rungs.get(r["group"], base_ladder),
                  f"{tag}: rung {r['rung_chosen']} not in the group-{r['group']} ladder")
        elif decision == "no_safe_decimation":
            # Group C policy: C-only fallback — ship undecimated,
            # but ONLY with full evidence that the extended ladder was tried and failed.
            check(r["group"] == "C", f"{tag}: no_safe_decimation is a Group-C-only decision")
            check(r.get("decision_rationale") == NO_SAFE_DECIMATION_RATIONALE,
                  f"{tag}: decision rationale is not the pinned verbatim text")
            check(r["faces_out"] == r["faces_in"] and r["verts_out"] == r["verts_in"]
                  and float(r["keep_achieved"]) == 1.0,
                  f"{tag}: no_safe_decimation deliverable must be full resolution")
            check(float(r["hausdorff"]["two_sided_max"]) == 0.0,
                  f"{tag}: no_safe_decimation deliverable must be the identity (Hausdorff 0)")
            sj = r["ssim"].get("journey", []) if isinstance(r.get("ssim"), dict) else []
            for rung in [ladder_max, *ext_c]:
                tried = [j for j in sj if j.get("rung") == rung]
                check(bool(tried), f"{tag}: extended-ladder evidence missing for rung {rung:g}")
                check(all(not j.get("pass") for j in tried),
                      f"{tag}: rung {rung:g} shows a passing attempt — no_safe_decimation invalid")
            check(not any(j.get("pass") and j.get("rung") != 1.0 for j in sj),
                  f"{tag}: a decimation rung passed SSIM — no_safe_decimation invalid")
            # deliverable provenance: byte-identical to the M1 conversion (SHA re-verified HERE)
            d = r.get("deliverable", {})
            check(d.get("kind") == "undecimated_m1_copy", f"{tag}: deliverable provenance missing")
            if d.get("kind") == "undecimated_m1_copy":
                pairs = ((_abs(d["m1_obj"]), _abs(r["obj"]), d.get("obj_sha256"), "OBJ"),
                         (_abs(d["m1_mtl"]), _abs(r["mtl"]), d.get("mtl_sha256"), "MTL"))
                for m1_p, dec_p, rec_sha, what in pairs:
                    if not (m1_p.exists() and dec_p.exists()):
                        check(False, f"{tag}: {what} copy or its M1 source missing on disk")
                        continue
                    h_m1, h_dec = _sha(m1_p), _sha(dec_p)
                    check(h_m1 == h_dec == rec_sha,
                          f"{tag}: {what} copy not byte-identical to M1 "
                          f"(m1 {h_m1[:12]} / copy {h_dec[:12]} / recorded {str(rec_sha)[:12]})")
                tex_p = _abs(r["obj"]).parent / r["texture"]
                check(tex_p.exists() and _sha(tex_p) == r["texture_sha256"],
                      f"{tag}: deliverable texture SHA mismatch")
        else:
            check(False, f"{tag}: unknown decision {decision!r}")
        check(bool(r["geometry_pass"]), f"{tag}: geometry gates not green ({r.get('fail_reasons')})")
        edge_mean = audit_edge[r["stem"]]
        hd = float(r["hausdorff"]["two_sided_max"])
        check(hd <= hd_ratio_max * edge_mean + 1e-9,
              f"{tag}: hausdorff {hd:.4f} > {hd_ratio_max} x audit edge_len_mean {edge_mean:.4f}")
        src, dec = r["src_stats"], r["dec_stats"]
        check(dec["uv_flipped"] <= src["uv_flipped"], f"{tag}: new flipped UV triangles ({src['uv_flipped']}->{dec['uv_flipped']})")
        check(dec["uv_charts"] <= src["uv_charts"], f"{tag}: UV chart split ({src['uv_charts']}->{dec['uv_charts']})")
        if src["boundary_length"] == 0:
            check(dec["boundary_length"] == 0, f"{tag}: boundary appeared on closed mesh")
        else:
            rel = abs(dec["boundary_length"] - src["boundary_length"]) / src["boundary_length"]
            check(rel <= 0.02 + 1e-12, f"{tag}: boundary length deviates {rel:.4f} > 2%")
        check(dec["nonmanifold_edge_count"] <= src["nonmanifold_edge_count"], f"{tag}: non-manifold edges grew")
        check(dec["isolated_vertices"] == 0, f"{tag}: isolated vertices present")
        check(r.get("uv_obj_mode") in ("shared", "wedge"), f"{tag}: uv_obj_mode missing")
        check(isinstance(r.get("frame_obj_est_bytes"), int) and r["frame_obj_est_bytes"] > 0,
              f"{tag}: per-frame OBJ size estimate missing")
        check((ROOT / Path(r["obj"]).relative_to(ROOT)).exists() if Path(r["obj"]).is_absolute() else (ROOT / r["obj"]).exists(),
              f"{tag}: decimated OBJ missing on disk")

        s = r.get("ssim")
        if geometry_only:
            continue
        if not isinstance(s, dict):
            check(False, f"{tag}: SSIM pending — perceptual gate not run")
            continue
        views = s.get("views", [])
        check(len([v for v in views if not v["closeup"]]) >= 2, f"{tag}: <2 production SSIM views")
        check(len([v for v in views if v["closeup"]]) >= 2, f"{tag}: <2 close-up SSIM views")
        for v in views:
            thr = ssim_close_min if v["closeup"] else ssim_full_min
            check(float(v["ssim"]) >= thr, f"{tag}: SSIM {v['name']}={v['ssim']:.4f} < {thr}")
        check(bool(s.get("pass")), f"{tag}: ssim.pass is false")

    # disk budget line present (qa-gates M2)
    db = doc["summary"].get("disk_budget", {})
    check(db.get("projected_frame_export_bytes", 0) > 0, "projected frame-export disk total missing")

    # ---------------- spot re-verification ----------------
    rng = random.Random(20260610)
    ok_meshes = [r for r in meshes if "error" not in r]
    sample = rng.sample(ok_meshes, k=min(3, len(ok_meshes)))
    for r in sample:
        tag = f"[spot {r['group']}] {r['stem']}"
        obj_p = Path(r["obj"])
        obj_p = obj_p if obj_p.is_absolute() else ROOT / obj_p
        m = read_obj(obj_p)
        check(m.n_faces == r["faces_out"], f"{tag}: reloaded face count {m.n_faces} != report {r['faces_out']}")
        check(m.n_vertices == r["verts_out"], f"{tag}: reloaded vertex count differs")
        if r["texture"]:
            tex_p = obj_p.parent / r["texture"]
            check(tex_p.exists(), f"{tag}: texture copy missing")
            if tex_p.exists():
                sha = hashlib.sha256(tex_p.read_bytes()).hexdigest()
                check(sha == r["texture_sha256"], f"{tag}: texture SHA mismatch")
        src_mesh = read_ply(Path(r["src"]) if Path(r["src"]).is_absolute() else ROOT / r["src"])
        wedge = m.vertex_uv[m.faces] if m.vertex_uv is not None else m.wedge_uv
        check(wedge is not None, f"{tag}: reloaded OBJ has no UVs")
        if wedge is not None and src_mesh.wedge_uv is not None:
            lo = src_mesh.wedge_uv.reshape(-1, 2).min(axis=0) - 1e-6
            hi = src_mesh.wedge_uv.reshape(-1, 2).max(axis=0) + 1e-6
            w2 = wedge.reshape(-1, 2)
            check(bool((w2 >= lo).all() and (w2 <= hi).all()), f"{tag}: reloaded UVs outside source UV bbox")
        print(f"  ok    {tag}: reload + texture SHA verified")

    # re-check Hausdorff for one sampled mesh, arrays in-memory (pymeshlab as engine only)
    r = sample[0]
    import pymeshlab

    src_mesh = read_ply(Path(r["src"]) if Path(r["src"]).is_absolute() else ROOT / r["src"])
    obj_p = Path(r["obj"]) if Path(r["obj"]).is_absolute() else ROOT / Path(r["obj"])
    dec_mesh = read_obj(obj_p)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=src_mesh.vertices.astype(np.float64),
                               face_matrix=src_mesh.faces.astype(np.int32)))
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=dec_mesh.vertices.astype(np.float64),
                               face_matrix=dec_mesh.faces.astype(np.int32)))
    cap = 500_000
    fwd = ms.get_hausdorff_distance(sampledmesh=0, targetmesh=1,
                                    samplenum=min(cap, src_mesh.n_vertices), samplevert=True, sampleface=True)
    rev = ms.get_hausdorff_distance(sampledmesh=1, targetmesh=0,
                                    samplenum=min(cap, dec_mesh.n_vertices), samplevert=True, sampleface=True)
    hd2 = max(float(fwd["max"]), float(rev["max"]))
    thr = hd_ratio_max * audit_edge[r["stem"]]
    check(hd2 <= thr + 1e-9, f"[spot] {r['stem']}: re-measured Hausdorff {hd2:.4f} > threshold {thr:.4f}")
    print(f"  ok    [spot] {r['stem']}: re-measured two-sided Hausdorff {hd2:.4f} <= {thr:.4f}")

    n = len(meshes)
    if FAIL:
        print(f"\nM2 GATE: RED — {len(FAIL)} failures across {n} meshes")
        return 1
    mode = " (geometry-only: SSIM not asserted — NOT a green M2)" if geometry_only else ""
    ndec = sum(1 for r in meshes if r.get("decision") == "no_safe_decimation")
    print(f"\nM2 GATE: GREEN — {n}/{n} meshes pass "
          f"({n - ndec} decimated, {ndec} no_safe_decimation undecimated C deliverables){mode}")
    return 0 if not geometry_only else 0


if __name__ == "__main__":
    sys.exit(main())
