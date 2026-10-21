#!/usr/bin/env python3
"""Decide whether an automatic split alignment may be merged, and with which thin-axis shift.

Reads ALIGN_DIR/align.log and silhouette.log (split_align.py align / silhouette output). Accepts when
  - the best flip hypothesis scores at least twice the best hypothesis with different axis signs (one clear flip;
    a nearly symmetric slab can fit both ways, as sncass94 did before its blocks were regrouped);
  - the rim rms is below 0.1 mm and the rim scale (B relative to A) is within 1% of 1 (the marker scales agree);
  - the silhouette optimum is between -0.3 and +1.2 mm.
The shift is the silhouette optimum when it is above +0.2 mm, else 0. The optimum is biased towards a thicker object
on bright captures, where the soft contact shadow under the object enters the masks (1756 +1.0 and sncass94 +0.6 mm,
against +0.3 and +0.15 on the darker 1482/1585 captures), while the face overlap in split_align.py check pointed to
+0.1..0.4 mm and about 0 respectively. Erring thicker gives a looser case, the choice made for 1482, 1756 and
sncass94; an optimum above +0.3 mm is marked for review: merge the unshifted variant as well and compare the two
meshes with split_align.py check (INSTRUCTIONS, "Orientations that do not link").
Prints the decision, writes it to ALIGN_DIR/auto_decision.json and exits 1 when rejected.
Usage: split_gate.py ALIGN_DIR"""
import json
import os
import re
import sys

d = sys.argv[1]
a = open(os.path.join(d, "align.log")).read()
s = open(os.path.join(d, "silhouette.log")).read()
rms = float(re.search(r"best: .*rms ([\d.]+) mm", a).group(1))
rim = float(re.search(r"best fits the rim: ([\d.na]+)", a).group(1))
rank = re.findall(r"axes ([+-]1[+-]1) shift [+-]\d score ([\d.]+)", re.search(r"ranking: (.*)", a).group(1))
best_axes, best = rank[0][0], float(rank[0][1])
other = max([float(v) for ax, v in rank if ax != best_axes], default=0.0)
opt = float(re.search(r"smallest at a shift of ([+-][\d.]+) mm", s).group(1))
checks = {"clear flip (best/other >= 2)": best >= 2 * other, "rim rms < 0.1 mm": rms < 0.1,
          "rim scale within 1%": abs(rim - 1) <= 0.01, "silhouette optimum within -0.3..+1.2 mm": -0.3 <= opt <= 1.2}
shift = round(opt, 2) if opt > 0.2 else 0.0
ok = all(checks.values())
review = ok and opt > 0.3
res = {"rms_mm": rms, "rim_scale": rim, "best_score": best, "other_flip_score": other, "silhouette_optimum_mm": opt,
       "checks": checks, "accepted": ok, "thin_shift_mm": shift, "review_unshifted_variant": review}
json.dump(res, open(os.path.join(d, "auto_decision.json"), "w"), indent=1)
for k, v in checks.items():
    print(f"  {'ok  ' if v else 'FAIL'} {k}")
print(f"rms {rms} mm, rim scale {rim}, flip scores {best} vs {other}, silhouette optimum {opt:+.3f} mm -> "
      + (f"merge with --thin-shift {shift}" + ("; REVIEW: also build the unshifted variant" if review else "")
         if ok else "REJECTED, not merged"))
sys.exit(0 if ok else 1)
