#!/usr/bin/env python3
"""Exit 0 if a joint SfM posed (nearly) every orientation, 1 if one orientation is missing or mostly unposed.
Usage: linked.py SFM_JSON SCALE_LISTS_DIR [--min-frac 0.9]   (SFM_JSON: openMVG_main_ConvertSfM_DataFormat export)"""
import json, os, sys
scene, lists = sys.argv[1], sys.argv[2]
min_frac = float(sys.argv[sys.argv.index("--min-frac") + 1]) if "--min-frac" in sys.argv else 0.9
d = json.load(open(scene))
poses = {e["key"] for e in d["extrinsics"]}
posed = {v["value"]["ptr_wrapper"]["data"]["filename"] for v in d["views"]
         if v["value"]["ptr_wrapper"]["data"]["id_pose"] in poses}
ok = True
for f in sorted(os.listdir(lists)):
    if f.endswith("-all.txt"):
        names = open(os.path.join(lists, f)).read().split()
        frac = sum(n in posed for n in names) / len(names)
        print(f"{f[5:-8]}: {sum(n in posed for n in names)}/{len(names)} posed ({frac:.0%})")
        ok &= frac >= min_frac
print("linked" if ok else "NOT linked")
sys.exit(0 if ok else 1)
