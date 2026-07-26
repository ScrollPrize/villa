#!/usr/bin/env python
"""Milestone M0 gate — see docs/QUALITY-GATES.md. Exit 0 = green."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FAIL: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        FAIL.append(msg)


def main() -> int:
    print("== M0 gate ==")

    # 1. tests
    r = subprocess.run(["uv", "run", "pytest", "-q"], cwd=ROOT, capture_output=True, text=True)
    check(r.returncode == 0, f"pytest green ({r.stdout.strip().splitlines()[-1] if r.stdout else 'no output'})")

    # 2. audit completeness
    audit_p = ROOT / "reports/audit.json"
    check(audit_p.exists(), "reports/audit.json exists")
    if audit_p.exists():
        audit = json.loads(audit_p.read_text())
        meshes = audit.get("meshes", [])
        check(len(meshes) == 34, f"34 meshes audited (got {len(meshes)})")
        groups = {}
        for m in meshes:
            groups[m["group"]] = groups.get(m["group"], 0) + 1
        check(groups.get("A") == 20 and groups.get("B") == 11 and groups.get("C") == 3,
              f"group counts A=20 B=11 C=3 (got {groups})")
        anomalies = audit.get("anomalies", [])
        unexplained = [a for a in anomalies if not a.get("explanation")]
        check(not unexplained, f"all anomalies explained ({len(anomalies)} total, {len(unexplained)} unexplained)")
        oracle = audit.get("orientation_oracle", {})
        consensus = oracle.get("group_consensus", {})
        check(bool(consensus.get("A")) and bool(consensus.get("B")),
              f"orientation oracle has A/B consensus (got {consensus})")
        ink = audit.get("ink", {})
        check(len(ink.get("march_matches", [])) == 19, f"19 march ink matches (got {len(ink.get('march_matches', []))})")
        check(len(ink.get("may_matches", [])) == 11, f"11 may ink matches (got {len(ink.get('may_matches', []))})")
        zipc = ink.get("zip_completeness", {})
        check(bool(zipc) and all(v.get("complete") for v in zipc.values()) if isinstance(zipc, dict) else bool(zipc),
              "zip extraction completeness verified")
        # texture presence: every mesh has at least one declared texture that exists on disk
        missing_tex = [m["path"] for m in meshes
                       if not any(t.get("exists") for t in m.get("textures", []))]
        check(not missing_tex, f"primary texture exists for every mesh (missing: {missing_tex})")
        # wedge==vertex holds for all A/B (conversion fast path assumption)
        ab_bad = [m["path"] for m in meshes if m["group"] in ("A", "B") and m.get("wedge_equals_vertex_uv") is not True]
        check(not ab_bad, f"wedge≡vertex UV bit-exact for all A/B (violations: {ab_bad})")
        # denorm fit sane for all A/B: the *pixel-scale* ratio (s_u/tex_W)/(s_v/tex_H) must be
        # ≈1 (the compositor rasterized the flat sheet at one uniform px-per-voxel scale);
        # the raw s_u/s_v ratio is just the sheet aspect and can be anything.
        fit_bad = []
        for m in meshes:
            if m["group"] not in ("A", "B"):
                continue
            fit = m.get("denorm") or {}
            ratio = fit.get("pixel_scale_anisotropy")
            if ratio is None or not (0.8 < ratio < 1.25):
                fit_bad.append((m["path"], ratio))
        check(not fit_bad, f"denorm pixel-scale isotropy for all A/B (bad: {fit_bad})")

    # 3. EGL smoke
    smoke = ROOT / "reports/smoke/egl_w030.png"
    check(smoke.exists(), "EGL smoke render exists")
    probe = ROOT / "reports/smoke/uv_probe.json"
    check(probe.exists() and json.loads(probe.read_text()).get("t0_maps_to") in ("top_row", "bottom_row"),
          "UV probe pinned VTK t-coordinate convention")

    # 4. solver bench
    bench_p = ROOT / "reports/solver_bench.json"
    check(bench_p.exists(), "reports/solver_bench.json exists")
    if bench_p.exists():
        bench = json.loads(bench_p.read_text())
        chosen = bench.get("chosen")
        ok = chosen and bench.get("backends", {}).get(chosen, {}).get("available")
        res = bench.get("backends", {}).get(chosen, {}).get("rel_residual", 1.0)
        check(bool(ok) and res < 1e-6, f"chosen solver backend valid: {chosen} (residual {res})")

    print("== M0:", "GREEN ==" if not FAIL else f"RED ({len(FAIL)} failures) ==")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
