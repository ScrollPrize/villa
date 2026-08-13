#!/usr/bin/env python
"""Post-fleet objective visual checks on rendered masters (no GPU needed).

Per mesh (plain_h + ink_h 4K):
  1. early-flicker: A->B->A texel-crawl metric over frames 0-70, left half
     (regression guard for the nearest-sampling shimmer; bilinear should hold
     totals < ~3000 px-windows vs ~23000 on the r7 baseline).
  2. late-flicker: same metric over frames 110-215 full frame (z-fight /
     layer-crossing guard; duplicated-patch hotspots measured 80k-120k).
  3. watermark presence: bottom-right corner luminance signature on frame 10.

Usage: uv run python scripts/verify_fleet_visuals.py <stem> [<stem> ...]
Writes reports/fleet_visual_checks.json (merge per stem).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EARLY_MAX = 4000      # px-window total, frames 0-70 left half (r7 baseline 23060)
LATE_MAX = 25000      # px-window total, frames 110-215 (dup-patch era 86k-120k)


def frames(path: Path, n0: int, n1: int, w: int, h: int) -> np.ndarray:
    cmd = ["ffmpeg", "-loglevel", "error", "-i", str(path),
           "-vf", f"select='gte(n\\,{n0})*lte(n\\,{n1})',scale={w}:{h}",
           "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w).astype(np.int16)


def flicker_total(F: np.ndarray, thresh: int = 40) -> int:
    D1 = np.abs(F[1:-1] - F[:-2])
    D2 = np.abs(F[2:] - F[1:-1])
    D02 = np.abs(F[2:] - F[:-2])
    fl = np.minimum(D1, D2) * (D02 < 0.3 * np.minimum(D1, D2))
    return int((fl.reshape(len(fl), -1) > thresh).sum())


def check_mesh(stem: str) -> dict:
    vdir = ROOT / "outputs/anim" / stem / "video"
    out: dict = {"stem": stem}
    for variant in ("plain", "ink"):
        p = vdir / f"{stem}_{variant}_h_4k.mp4"
        if not p.exists():
            out[variant] = {"missing": True}
            continue
        Fe = frames(p, 0, 70, 1920, 1080)[:, :, :960]
        Fl = frames(p, 110, 215, 1920, 1080)
        early = flicker_total(Fe)
        late = flicker_total(Fl, 50)
        f10 = frames(p, 10, 10, 1920, 1080)[0]
        corner = f10[1020:1060, 1500:1900]
        wm_present = bool(corner.max() > 90)
        out[variant] = {"early_flicker": early, "early_ok": early <= EARLY_MAX,
                        "late_flicker": late, "late_ok": late <= LATE_MAX,
                        "watermark_present": wm_present}
    return out


def main() -> int:
    stems = sys.argv[1:]
    rp = ROOT / "reports/fleet_visual_checks.json"
    acc = json.loads(rp.read_text()) if rp.exists() else {}
    rc = 0
    for stem in stems:
        r = check_mesh(stem)
        acc[stem] = r
        ok = all(v.get("early_ok") and v.get("late_ok") and v.get("watermark_present")
                 for k, v in r.items() if isinstance(v, dict) and not v.get("missing"))
        print(f"{stem}: {json.dumps(r)}" )
        if not ok:
            rc = 1
    rp.write_text(json.dumps(acc, indent=1))
    return rc


if __name__ == "__main__":
    sys.exit(main())
