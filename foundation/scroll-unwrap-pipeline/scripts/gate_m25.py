#!/usr/bin/env python
"""Milestone M2.5 gate — see docs/QUALITY-GATES.md. Exit 0 = green.

Checks (independently of the bake's own records where it matters):
  1. Matching table complete: every Group A/B mesh classified (matched or
     UNMATCHED+documented), auto_trace UNMATCHED documented, w010 orphan
     documented.
  2. 30/30 overlay PNGs exist, dims == base texture dims, alpha channel
     byte-identical to base (np.array_equal on freshly decoded pixels).
  3. Polarity sanity: in-mask ink coverage in [0.5%, 35%]; out-of-band entries
     stay green only with an explicit explanation sourced from audit
     alpha_frac data (damaged sparse wraps).
  4. Registration: outside-mask fraction < 3% or a documented, decisively
     resolved anomaly.
  5. reports/m25_ink.json complete (params, per-mesh records, anomalies).
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FAIL: list[str] = []

COVERAGE_BAND = (0.005, 0.35)
REG_OUTSIDE_MAX = 0.03


def check(cond: bool, msg: str) -> None:
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        FAIL.append(msg)


def _alpha_check(args: tuple[str, str, list[int]]) -> tuple[str, bool, bool, str]:
    """worker: (stem, ok_dims, ok_alpha, detail) for one overlay vs its base."""
    import numpy as np
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    tex_path, out_path, tex_wh = args
    op = ROOT / out_path
    if not op.exists():
        return out_path, False, False, "overlay missing"
    with Image.open(op) as oi:
        if oi.mode != "RGBA":
            return out_path, False, False, f"mode {oi.mode} != RGBA"
        ok_dims = list(oi.size) == list(tex_wh)
        over_a = np.asarray(oi)[..., 3]
    with Image.open(ROOT / tex_path) as bi:
        base_a = np.asarray(bi)[..., 3]
    ok_alpha = bool(over_a.shape == base_a.shape and np.array_equal(over_a, base_a))
    return out_path, ok_dims, ok_alpha, "ok" if (ok_dims and ok_alpha) else "dims/alpha mismatch"


def main() -> int:
    print("== M2.5 gate ==")
    audit = json.loads((ROOT / "reports/audit.json").read_text())
    ink = audit["ink"]

    # --- 1. matching table complete -------------------------------------------------
    march, may = ink["march_matches"], ink["may_matches"]
    check(len(march) == 19, f"19 march matches (got {len(march)})")
    check(len(may) == 11, f"11 may matches (got {len(may)})")
    ab_meshes = {m["path"] for m in audit["meshes"] if m["group"] in ("A", "B")}
    classified = {m["matched_mesh"] for m in march + may} | set(ink["unmatched_meshes"])
    check(classified == ab_meshes,
          f"every A/B mesh classified matched|UNMATCHED (diff: {sorted(ab_meshes ^ classified)})")
    kinds = {a["kind"]: a for a in audit["anomalies"]}
    doc_unmatched = kinds.get("mesh_without_ink", {})
    check("auto_trace" in str(ink["unmatched_meshes"]) and bool(doc_unmatched.get("explanation")),
          "auto_trace UNMATCHED documented with explanation (audit anomaly 'mesh_without_ink')")
    doc_orphan = kinds.get("orphan_ink", {})
    orphans = [o for o in ink["orphans"] if o["matched_mesh"] is None]
    check(len(orphans) == 1 and "w010" in orphans[0]["tif"] and bool(doc_orphan.get("explanation")),
          "w010 orphan ink documented with explanation (audit anomaly 'orphan_ink')")

    # --- 5a. report exists / complete ----------------------------------------------
    rep_p = ROOT / "reports/m25_ink.json"
    check(rep_p.exists(), "reports/m25_ink.json exists")
    if not rep_p.exists():
        print("== M2.5: RED (no report) =="); return 1
    rep = json.loads(rep_p.read_text())
    recs = rep.get("records", [])
    check(len(recs) == 30 and not rep.get("failures"), f"30/30 records baked, 0 failures "
          f"(got {len(recs)}, failures={rep.get('failures')})")
    need = {"stem", "kind", "tex_path", "ink_path", "tex_wh", "resample_steps", "polarity",
            "registration", "coverage", "output", "runtime_s"}
    incomplete = [r.get("stem", "?") for r in recs if not need.issubset(r)]
    check(not incomplete, f"per-mesh records complete ({sorted(need)}) (incomplete: {incomplete})")
    check(bool(rep.get("params")) and bool(rep.get("generated_at")) and "anomalies" in rep,
          "report carries params, generated_at, anomalies")
    expected = {(m["matched_mesh"]) for m in march + may}
    got = {r["mesh_path"] for r in recs}
    check(got == expected, f"records cover exactly the 30 matched meshes (diff: {sorted(expected ^ got)})")
    sha_missing = [r["stem"] for r in recs if not r.get("output", {}).get("sha256")]
    check(not sha_missing, f"sha256 recorded for every overlay (missing: {sha_missing})")

    # --- 2. overlays exist, dims == base, alpha byte-identical (independent decode) -
    jobs = [(r["tex_path"], r["output"]["path"], r["tex_wh"]) for r in recs]
    bad_dims, bad_alpha = [], []
    with ProcessPoolExecutor(max_workers=6) as ex:
        for path, ok_d, ok_a, detail in ex.map(_alpha_check, jobs):
            if not ok_d:
                bad_dims.append(f"{path}: {detail}")
            if not ok_a:
                bad_alpha.append(f"{path}: {detail}")
    check(not bad_dims, f"30/30 overlays exist with dims == base texture dims (bad: {bad_dims})")
    check(not bad_alpha, f"30/30 alpha channels byte-identical to base (np.array_equal) (bad: {bad_alpha})")

    # --- 3. polarity / coverage band ------------------------------------------------
    out_of_band = [(r["stem"], r["polarity"]["coverage_used"]) for r in recs
                   if not (COVERAGE_BAND[0] <= r["polarity"]["coverage_used"] <= COVERAGE_BAND[1])]
    unexplained = [s for s, _ in out_of_band
                   if not next(r for r in recs if r["stem"] == s)["coverage"].get("explanation")]
    print(f"        coverage range: {min(r['polarity']['coverage_used'] for r in recs):.3%}"
          f" .. {max(r['polarity']['coverage_used'] for r in recs):.3%};"
          f" out-of-band: {[(s, f'{c:.3%}') for s, c in out_of_band]}")
    check(not unexplained,
          f"ink-in-mask coverage in [0.5%,35%] or flagged with explanation (unexplained: {unexplained})")
    for s, c in out_of_band:
        r = next(r for r in recs if r["stem"] == s)
        expl = r["coverage"].get("explanation", "")
        if c < COVERAGE_BAND[0]:  # below-band stays green only when sourced from audit alpha_frac
            check("alpha_frac" in expl, f"below-band {s}: explanation cites audit alpha_frac data")

    # --- 4. registration ------------------------------------------------------------
    reg_bad = []
    for r in recs:
        reg = r["registration"]
        if reg["outside_frac"] < REG_OUTSIDE_MAX and reg["variant_used"] == "identity":
            continue  # clean pass, no flip
        an = reg.get("anomaly")
        if not an:  # any violation or flip must be documented as an anomaly with evidence
            reg_bad.append((r["stem"], "violation/flip without anomaly record"))
            continue
        if reg["variant_used"] != "identity" and not an.get("decisive"):
            reg_bad.append((r["stem"], f"non-decisive flip {reg['variant_used']} applied"))
        if reg["variant_used"] == "identity" and "halo_identity" not in an:
            reg_bad.append((r["stem"], "identity kept on violation without halo evidence"))
    n_flip = sum(1 for r in recs if r["registration"]["variant_used"] != "identity")
    n_halo = sum(1 for r in recs
                 if r["registration"]["variant_used"] == "identity" and r["registration"].get("anomaly"))
    print(f"        registration: {30 - n_flip - n_halo} clean, {n_flip} decisive flips, "
          f"{n_halo} documented edge-halo violations")
    check(not reg_bad,
          f"registration: <3% outside-mask, or decisive documented flip, or documented halo (bad: {reg_bad})")

    # --- previews exist for the gate's vision review --------------------------------
    prevs = rep.get("previews", [])
    pv_files_ok = all((ROOT / p["sbs"]).exists() and (ROOT / p["zoom"]).exists() for p in prevs)
    kinds_cov = {p["kind"] for p in prevs}
    check(len(prevs) >= 6 and pv_files_ok and kinds_cov == {"march", "may"},
          f"6 preview pairs on disk incl. TIF- and JPG-source ({len(prevs)} listed, kinds {sorted(kinds_cov)})")

    print("== M2.5:", "GREEN ==" if not FAIL else f"RED ({len(FAIL)} failures) ==")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
