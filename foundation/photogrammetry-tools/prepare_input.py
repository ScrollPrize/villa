#!/usr/bin/env python3
"""Prepare the PGS-Recon inputs of one object (run as the data owner, no sudo; used by batch/batch_recon.sh).

From the masked acquisition tree WORK/<id>/<id>/<orientation>_camera_<height>/{JPGEnhanced,ScrollMasks}/ it builds
  - WORK/<id>/pgs-input/: every photograph with its mask beside it (hard links), the joint PGS-Recon input;
  - WORK/<id>/recon/scale-lists/keep-o<n>-<height>.txt and keep-o<n>-all.txt: the scale candidate lists;
  - WORK/<id>/split/pgs-input-o<n>/ and split/run-record/keep-o<n>-<height>.txt: the per-orientation inputs that
    split_recon.sh needs if the orientations do not link (cheap hard links, so built for every object).
It refuses to run if any photograph lacks its ScrollMasks/ mask.
Usage: prepare_input.py WORK ID
"""
import os, re, sys
work, oid = sys.argv[1], sys.argv[2]
base = os.path.join(work, oid)
tree = os.path.join(base, oid)
blocks = sorted(d for d in os.listdir(tree) if re.match(r"orientation\d+_camera_\w+$", d))
lists, per_o = {}, {}
for b in blocks:
    o, h = re.match(r"orientation(\d+)_camera_(\w+)$", b).groups()
    jdir, mdir = os.path.join(tree, b, "JPGEnhanced"), os.path.join(tree, b, "ScrollMasks")
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(jdir) if f.endswith(".jpg"))
    missing = [s for s in stems if not os.path.exists(os.path.join(mdir, s + "_mask.png"))]
    if missing:
        raise SystemExit(f"{b}: {len(missing)} photographs without ScrollMasks mask, e.g. {missing[:3]}")
    lists[f"o{o}-{h}"] = (b, stems)
    per_o.setdefault(o, []).append(b)

def link_into(dst, names):
    os.makedirs(dst, exist_ok=True)
    for b, stems in names:
        for s in stems:
            for src, n in ((os.path.join(tree, b, "JPGEnhanced", s + ".jpg"), s + ".jpg"),
                           (os.path.join(tree, b, "ScrollMasks", s + "_mask.png"), s + "_mask.png")):
                if not os.path.exists(os.path.join(dst, n)):
                    os.link(src, os.path.join(dst, n))

link_into(os.path.join(base, "pgs-input"), lists.values())
sl = os.path.join(base, "recon", "scale-lists")
os.makedirs(sl, exist_ok=True)
rr = os.path.join(base, "split", "run-record")
os.makedirs(rr, exist_ok=True)
for o in sorted(per_o):
    sel = [v for k, v in lists.items() if k.startswith(f"o{o}-")]
    link_into(os.path.join(base, "split", f"pgs-input-o{o}"), sel)
    allstems = sorted(s for _, st in sel for s in st)
    for d in (sl, rr):
        open(os.path.join(d, f"keep-o{o}-all.txt"), "w").write("".join(s + ".jpg\n" for s in allstems))
for k, (b, stems) in lists.items():
    for d in (sl, rr):
        open(os.path.join(d, f"keep-{k}.txt"), "w").write("".join(s + ".jpg\n" for s in stems))
n = len(os.listdir(os.path.join(base, "pgs-input")))
print(f"{oid}: {len(blocks)} blocks, pgs-input {n // 2} photographs + {n - n // 2} masks; "
      f"orientations {', '.join(f'o{o}: {len(v)} blocks' for o, v in sorted(per_o.items()))}")
