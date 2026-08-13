#!/usr/bin/env python
"""Milestone M3 gate — unwrap core on the two pilot meshes (docs/QUALITY-GATES.md, M3 section)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PILOTS = {
    "w030_2025122002": "A",
    "w011_20260108140509268_merged_00": "B",
}
FAIL: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        FAIL.append(msg)


def main() -> int:
    print("== M3 gate ==")
    for stem, group in PILOTS.items():
        mp = ROOT / "outputs/anim" / stem / "metrics.json"
        check(mp.exists(), f"{stem}: metrics.json exists")
        if not mp.exists():
            continue
        m = json.loads(mp.read_text())
        g = m["meta"]["gates"]
        check(bool(g.get("pass")), f"{stem}: numeric transit gates pass "
              f"(area_p95_vis {g.get('area_p95_vis_max'):.3f} ≤ {g.get('area_p95_gate'):.3f}, "
              f"sd {g.get('sd_peak'):.3f}, flips {g.get('flipped_frac_max'):.2e}, "
              f"axis={g.get('axis_mode')})")
        emb = m["meta"]["embedding"]
        check(emb["mirror_check"] > 0, f"{stem}: numeric mirror check positive")
        check((ROOT / "outputs/anim" / stem / "frames.npz").exists(), f"{stem}: frames.npz exists")
        check((ROOT / "outputs/anim" / stem / "keyframes/contact_sheet.jpg").exists(),
              f"{stem}: keyframes rendered for visual QA")

    # chirality gate = numeric mirror_check (checked above per mesh) + M1 render parity
    # (texture correctness inherited by every frame); on-screen reading direction is an
    # M4 camera rule informed by reports/m3_chirality.json (composition data, not a gate).
    cp = ROOT / "reports/m3_chirality.json"
    check(cp.exists(), "chirality screen-transform table exists (M4 camera input)")

    print("== M3:", "GREEN ==" if not FAIL else f"RED ({len(FAIL)} failures) ==")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
