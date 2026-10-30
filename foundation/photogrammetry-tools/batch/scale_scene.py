#!/usr/bin/env python3
"""Multiply an OpenMVG JSON scene (camera centres, landmarks, control points) by a factor about the origin.
Usage: scale_scene.py IN.json OUT.json FACTOR"""
import json, sys
d = json.load(open(sys.argv[1]))
k = float(sys.argv[3])
for e in d["extrinsics"]:
    e["value"]["center"] = [k * x for x in e["value"]["center"]]
for key in ("structure", "control_points"):
    for l in d.get(key, []):
        l["value"]["X"] = [k * x for x in l["value"]["X"]]
json.dump(d, open(sys.argv[2], "w"))
print(f"scaled {sys.argv[1]} by {k} -> {sys.argv[2]}")
