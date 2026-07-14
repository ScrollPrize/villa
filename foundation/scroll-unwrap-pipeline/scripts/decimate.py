#!/usr/bin/env python
"""M2: perceptually-gated decimation ladder for all 34 meshes.

Phases:
  geometry  parallel (multiprocessing, pymeshlab per worker): per mesh walk the keep
            ladder [0.05 0.08 0.12 0.20 0.35] from the ORIGINAL, accept the most
            aggressive rung passing all geometry gates, write
            outputs/decimated/<group>/<stem>/<stem>_decimated.obj/.mtl/<texture>,
            write reports/m2_decimation.json (ssim: pending) + .md.
  ssim      SERIAL (one GPU render at a time): per mesh render source vs decimated
            (2 production views + 2 close-ups), SSIM gates 0.97/0.95; on failure
            retry the rung with extratcoordw=3 then move up the ladder. Group C climbs
            the extended ladder ([0.5 0.65 0.8], Group C policy);
            if nothing passes, the mesh ships UNDECIMATED (byte-identical M1 copy,
            SHA-verified) with decision='no_safe_decimation'. Updates the report in
            place.
  report    rebuild reports/m2_decimation.md from the JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

REPORT_JSON = ROOT / "reports/m2_decimation.json"
REPORT_MD = ROOT / "reports/m2_decimation.md"
FRAMES_PER_MESH = 240  # unwrap animation frames (configs/global.yaml: unwrap.frames)


def load_cfg() -> dict:
    import yaml

    return yaml.safe_load((ROOT / "configs/global.yaml").read_text())["decimation"]


def load_audit() -> dict:
    return json.loads((ROOT / "reports/audit.json").read_text())


def build_tasks(audit: dict) -> list[dict]:
    tasks = []
    for m in audit["meshes"]:
        tasks.append(
            {
                "group": m["group"],
                "stem": m["stem"],
                "src": str(ROOT / m["path"]),
                "edge_len_mean": float(m["edge_len_mean"]),
                "n_faces": int(m["n_faces"]),
                "out_dir": str(ROOT / "outputs/decimated" / m["group"] / m["stem"]),
            }
        )
    return tasks


def task_for(rec: dict, audit: dict) -> dict:
    for t in build_tasks(audit):
        if t["stem"] == rec["stem"] and t["group"] == rec["group"]:
            return t
    raise KeyError(rec["stem"])


def mesh_orientation(audit: dict, group: str, stem: str) -> str:
    to = audit["tex_orientation"]
    return to.get("per_mesh_overrides", {}).get(stem, to[group])


# --------------------------------------------------------------------------- geometry
def _worker(payload: tuple[dict, dict]) -> dict:
    task, cfg = payload
    from scrollkit.decimate.core import run_ladder

    try:
        rec = run_ladder(task, cfg)
    except Exception as exc:  # surfaced in the report; gate goes red
        import traceback

        rec = {
            "group": task["group"],
            "stem": task["stem"],
            "src": task["src"],
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "geometry_pass": False,
            "ssim": "pending",
        }
    return rec


def _g(x) -> str:
    """%g float formatting that tolerates the None ('n/a') of no_safe_decimation records."""
    return "—" if x is None else f"{float(x):g}"


def summarize(meshes: list[dict], cfg: dict) -> dict:
    ladder = list(cfg["keep_ladder"])
    no_safe = sorted(r["stem"] for r in meshes if r.get("decision") == "no_safe_decimation")
    hist: dict[str, dict[str, int]] = {}
    for rec in meshes:
        if "rung_chosen" not in rec or rec.get("decision") == "no_safe_decimation":
            continue
        g = rec["group"]
        key = f"{rec['rung_chosen']:g}"
        hist.setdefault(g, {})
        hist[g][key] = hist[g].get(key, 0) + 1
    total_hist: dict[str, int] = {}
    for g in hist:
        for k, v in hist[g].items():
            total_hist[k] = total_hist.get(k, 0) + v

    faces_in = sum(r.get("faces_in", 0) for r in meshes)
    faces_out = sum(r.get("faces_out", 0) for r in meshes)
    ratios = [r["hausdorff_ratio"] for r in meshes if r.get("hausdorff_ratio") is not None]
    wrap_meshes = [r for r in meshes if r.get("group") in ("A", "B")]
    projected = sum(r.get("frame_obj_est_bytes", 0) * FRAMES_PER_MESH for r in wrap_meshes)

    ssim_vals_full: list[float] = []
    ssim_vals_close: list[float] = []
    ssim_state = "complete"
    for r in meshes:
        s = r.get("ssim")
        if not isinstance(s, dict):
            ssim_state = "pending"
            continue
        for v in s.get("views", []):
            (ssim_vals_close if v.get("closeup") else ssim_vals_full).append(v["ssim"])

    return {
        "ladder": ladder,
        "no_safe_decimation_count": len(no_safe),
        "no_safe_decimation_meshes": no_safe,
        "rung_histogram_by_group": hist,
        "rung_histogram_total": total_hist,
        "faces_in_total": faces_in,
        "faces_out_total": faces_out,
        "overall_keep": (faces_out / faces_in) if faces_in else None,
        "worst_hausdorff_ratio": max(ratios) if ratios else None,
        "ssim_status": ssim_state,
        "ssim_worst_full": min(ssim_vals_full) if ssim_vals_full else None,
        "ssim_worst_closeup": min(ssim_vals_close) if ssim_vals_close else None,
        "geometry_pass_count": sum(1 for r in meshes if r.get("geometry_pass")),
        "mesh_count": len(meshes),
        "disk_budget": {
            "frames_per_mesh": FRAMES_PER_MESH,
            "wrap_meshes": len(wrap_meshes),
            "projected_frame_export_bytes": projected,
            "projected_frame_export_gib": round(projected / 2**30, 2),
            "basis": "per-mesh single-frame OBJ byte size (ObjFrameWriter content) x 240 frames, summed over A+B wraps",
        },
    }


def save_report(meshes: list[dict], cfg: dict, extra: dict | None = None) -> None:
    import pymeshlab  # noqa: F401  (version stamp)

    from scrollkit.decimate.core import GROUP_C_EXTENDED_LADDER, NO_SAFE_DECIMATION_RATIONALE

    meshes = sorted(meshes, key=lambda r: (r["group"], r["stem"]))
    doc = {
        "milestone": "M2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pymeshlab_version": "2025.7.post1",
        "filter": "meshing_decimation_quadric_edge_collapse_with_texture",
        "filter_params_base": {
            "targetperc": "<keep fraction per rung>",
            "qualitythr": cfg["qualitythr"],
            "preserveboundary": cfg["preserveboundary"],
            "boundaryweight": cfg["boundaryweight"],
            "extratcoordw": f"{cfg['extratcoordw']} (3.0 retry on UV-gate failure)",
            "optimalplacement": True,
            "preservenormal": True,
        },
        "gates": {
            "hausdorff_two_sided_max_vs_edge_len_mean": cfg["hausdorff_max_mean_edge"],
            "boundary_length_tol": 0.02,
            "uv": "no new flipped UV triangles, no UV chart growth",
            "topology": "nonmanifold edges <= source, isolated vertices == 0",
            "ssim_production_min": cfg["ssim_production_min"],
            "ssim_closeup_min": cfg["ssim_closeup_min"],
        },
        "config": cfg,
        # Policy (decimate aggressively ONLY
        # when visuals are unaffected): Group C walks extra conservative rungs; if none
        # passes, the mesh ships undecimated with decision='no_safe_decimation'.
        "group_ladder_extensions": {"C": list(GROUP_C_EXTENDED_LADDER)},
        "no_safe_decimation_policy": {
            "groups": ["C"],
            "deliverable": "byte-identical copy of the M1-converted OBJ (+ MTL + texture), SHA-verified",
            "rationale_verbatim": NO_SAFE_DECIMATION_RATIONALE,
        },
        "summary": summarize(meshes, cfg),
        "meshes": meshes,
    }
    if extra:
        doc.update(extra)
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    tmp = REPORT_JSON.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=1))
    tmp.replace(REPORT_JSON)
    write_md(doc)


def cmd_geometry(args: argparse.Namespace) -> int:
    import multiprocessing as mp

    cfg = load_cfg()
    audit = load_audit()
    tasks = build_tasks(audit)
    if args.only:
        tasks = [t for t in tasks if t["stem"] in args.only]
    tasks.sort(key=lambda t: -t["n_faces"])  # big meshes first: better pool packing
    print(f"[geometry] {len(tasks)} meshes, {args.workers} workers")

    t0 = time.time()
    results: list[dict] = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.workers, maxtasksperchild=2) as pool:
        for rec in pool.imap_unordered(_worker, [(t, cfg) for t in tasks]):
            results.append(rec)
            if "error" in rec:
                print(f"  ERROR [{rec['group']}] {rec['stem']}: {rec['error']}")
            else:
                print(
                    f"  [{rec['group']}] {rec['stem']}: keep={rec['rung_chosen']:g} "
                    f"faces {rec['faces_in']}->{rec['faces_out']} "
                    f"hd_ratio={rec['hausdorff_ratio']:.2f} uv={rec['uv_obj_mode']} "
                    f"w={rec['extratcoordw']:g} {'PASS' if rec['geometry_pass'] else 'FAIL'} "
                    f"({rec['seconds_geometry']}s, {len(results)}/{len(tasks)})"
                )

    if args.only and REPORT_JSON.exists():
        # merge re-run meshes into the existing report instead of clobbering it
        old = {(r["group"], r["stem"]): r for r in json.loads(REPORT_JSON.read_text())["meshes"]}
        for rec in results:
            old[(rec["group"], rec["stem"])] = rec
        results = list(old.values())
    save_report(results, cfg, extra={"geometry_wall_seconds": round(time.time() - t0, 1)})
    npass = sum(1 for r in results if r.get("geometry_pass"))
    print(f"[geometry] done in {time.time() - t0:.0f}s — {npass}/{len(results)} geometry-green; report: {REPORT_JSON}")
    return 0 if npass == len(results) else 1


# --------------------------------------------------------------------------- ssim
def cmd_ssim(args: argparse.Namespace) -> int:
    from scrollkit.decimate.perceptual import evaluate_with_ladder

    cfg = load_cfg()
    audit = load_audit()
    doc = json.loads(REPORT_JSON.read_text())
    meshes = doc["meshes"]
    t0 = time.time()
    for i, rec in enumerate(meshes):
        if "error" in rec:
            continue
        if isinstance(rec.get("ssim"), dict) and rec["ssim"].get("pass") and not args.force:
            continue
        task = task_for(rec, audit)
        orientation = mesh_orientation(audit, rec["group"], rec["stem"])
        print(f"[ssim {i + 1}/{len(meshes)}] [{rec['group']}] {rec['stem']} (rung {_g(rec['rung_chosen'])})")
        evaluate_with_ladder(rec, task, cfg, orientation, ROOT)
        s = rec["ssim"]
        if isinstance(s, dict):
            vals = ", ".join(f"{v['name']}={v['ssim']:.4f}" for v in s["views"])
            decided = " decision=no_safe_decimation (undecimated M1 copy)" if rec.get("decision") == "no_safe_decimation" else ""
            print(f"    -> {'PASS' if s['pass'] else 'FAIL'} rung={_g(rec['rung_chosen'])} w={_g(rec['extratcoordw'])}{decided} [{vals}]")
        # checkpoint after every mesh (serial GPU pass can be interrupted/resumed)
        save_report(meshes, cfg, extra={"geometry_wall_seconds": doc.get("geometry_wall_seconds")})
    doc2 = json.loads(REPORT_JSON.read_text())
    save_report(doc2["meshes"], cfg, extra={
        "geometry_wall_seconds": doc.get("geometry_wall_seconds"),
        "ssim_wall_seconds": round(time.time() - t0, 1),
    })
    ok = all(isinstance(r.get("ssim"), dict) and r["ssim"].get("pass") for r in doc2["meshes"])
    print(f"[ssim] done in {time.time() - t0:.0f}s — {'all pass' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


# --------------------------------------------------------------------------- report
def write_md(doc: dict) -> None:
    s = doc["summary"]
    lines = [
        "# M2 — Decimation report",
        "",
        f"Generated {doc['generated_at']} | pymeshlab {doc['pymeshlab_version']} | filter `{doc['filter']}`",
        "",
        f"- Ladder (keep fraction): {s['ladder']}"
        + (f" | group extensions: {doc['group_ladder_extensions']}" if doc.get("group_ladder_extensions") else ""),
        f"- Meshes: {s['mesh_count']} | geometry-green: {s['geometry_pass_count']}",
        f"- Faces: {s['faces_in_total']:,} -> {s['faces_out_total']:,} (overall keep {s['overall_keep']:.3f})"
        if s.get("overall_keep") else "- Faces: n/a",
        f"- Worst two-sided Hausdorff / edge_len_mean: {s['worst_hausdorff_ratio']:.3f}" if s.get("worst_hausdorff_ratio") else "",
        f"- Rung histogram (all): {s['rung_histogram_total']}",
        f"- SSIM status: {s['ssim_status']}"
        + (
            f" | worst full {s['ssim_worst_full']:.4f} | worst close-up {s['ssim_worst_closeup']:.4f}"
            if s.get("ssim_worst_full") is not None
            else ""
        ),
        f"- Projected frame-export disk: {s['disk_budget']['projected_frame_export_gib']} GiB "
        f"({s['disk_budget']['wrap_meshes']} wraps x {s['disk_budget']['frames_per_mesh']} frames; {s['disk_budget']['basis']})",
        (
            f"- no_safe_decimation: {s['no_safe_decimation_count']} Group-C meshes ship UNDECIMATED "
            f"(byte-identical M1 copies, SHA-verified): {', '.join(s['no_safe_decimation_meshes'])}. "
            f"Rationale (pinned, verbatim): "
            f"'{doc.get('no_safe_decimation_policy', {}).get('rationale_verbatim', '')}'"
            if s.get("no_safe_decimation_count")
            else ""
        ),
        "",
        "| grp | mesh | rung | w_tc | faces in -> out | keep | Hd_max/edge | bndry Δ | charts | flips | UV path | SSIM full | SSIM close | pass | notes |",
        "|-----|------|------|------|-----------------|------|-------------|---------|--------|-------|---------|-----------|------------|------|-------|",
    ]
    for r in doc["meshes"]:
        if "error" in r:
            lines.append(f"| {r['group']} | {r['stem']} | — | — | — | — | — | — | — | — | — | — | — | ERROR | {r['error']} |")
            continue
        s_ = r.get("ssim")
        if isinstance(s_, dict):
            fulls = [v["ssim"] for v in s_["views"] if not v["closeup"]]
            closes = [v["ssim"] for v in s_["views"] if v["closeup"]]
            ssim_full = f"{min(fulls):.4f}" if fulls else "—"
            ssim_close = f"{min(closes):.4f}" if closes else "—"
            ok = "PASS" if (r.get("geometry_pass") and s_.get("pass")) else "FAIL"
        else:
            ssim_full = ssim_close = "pending"
            ok = "geom-PASS" if r.get("geometry_pass") else "geom-FAIL"
        notes = "; ".join(r.get("notes", []) + r.get("fail_reasons", []))
        journey = [
            f"{a['keep']:g}{'+w3' if a['extratcoordw'] == 3.0 else ''}"
            f"{'+pq' if a.get('planarquadric') else ''}:{'ok' if a['pass'] else 'x'}"
            for a in r.get("journey", [])
        ]
        if len(journey) > 1:
            notes = (notes + "; " if notes else "") + "ladder " + " ".join(journey)
        if r.get("decision") == "no_safe_decimation":
            rung_s = "copy(1.0)"
            notes = f"DECISION no_safe_decimation — {r.get('decision_rationale', '')}" + ("; " + notes if notes else "")
        else:
            rung_s = f"{r['rung_chosen']:g}"
        lines.append(
            f"| {r['group']} | {r['stem']} | {rung_s} | {_g(r.get('extratcoordw'))} "
            f"| {r['faces_in']:,} -> {r['faces_out']:,} | {r['keep_achieved']:.3f} "
            f"| {r['hausdorff_ratio']:.3f} | {r['boundary_rel_diff']:.4f} "
            f"| {r['src_stats']['uv_charts']}->{r['dec_stats']['uv_charts']} "
            f"| {r['src_stats']['uv_flipped']}->{r['dec_stats']['uv_flipped']} "
            f"| {r['uv_obj_mode']} | {ssim_full} | {ssim_close} | {ok} | {notes} |"
        )
    REPORT_MD.write_text("\n".join(lines) + "\n")


def cmd_report(args: argparse.Namespace) -> int:
    doc = json.loads(REPORT_JSON.read_text())
    write_md(doc)
    print(f"wrote {REPORT_MD}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("geometry")
    g.add_argument("--workers", type=int, default=5)
    g.add_argument("--only", nargs="*", default=None)
    g.set_defaults(fn=cmd_geometry)
    ss = sub.add_parser("ssim")
    ss.add_argument("--force", action="store_true", help="re-evaluate even meshes already SSIM-green")
    ss.set_defaults(fn=cmd_ssim)
    rp = sub.add_parser("report")
    rp.set_defaults(fn=cmd_report)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
