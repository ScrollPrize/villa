#!/usr/bin/env python
"""CLI wrapper around scrollkit.metrics.clearance over a saved trajectory.

Usage: uv run python scripts/clearance_cert.py outputs/anim/<stem> [--json out]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scrollkit.metrics.clearance import clearance_certificate  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("anim_dir", type=Path)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--margin-frac", type=float, default=0.05)
    args = ap.parse_args()
    d = np.load(args.anim_dir / "frames.npz")
    kw = {}
    if "s_arc" in d and "c_arc" in d and "pad_arc" in d:
        kw = {"s": d["s_arc"], "c_arc": d["c_arc"], "pad_arc": float(d["pad_arc"])}
    res = clearance_certificate(d["frames"], d["faces"], args.margin_frac,
                                per_frame=True, **kw)
    res["anim_dir"] = str(args.anim_dir)
    res["mode"] = "exact-state" if kw else "residency-heuristic"
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(res, indent=1))
    wm = res["worst_margin"]
    print(f"clearance: worst margin {wm if wm is None else f'{wm:.4f}'} "
          f"(need >= {res['margin_min_required']:.4f}, med_edge {res['med_edge']:.3f}) "
          f"at frame {res['worst_frame']} over {res['frames_with_overlap']} overlap frames "
          f"-> {'PASS' if res['pass'] else 'FAIL'}")
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
