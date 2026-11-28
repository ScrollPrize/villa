#!/usr/bin/env python3
# usage: compare_track_stores.py PER_TRACK_STORE BULK_STORE N   (needs the vc_object_store Python module)
# equivalence of a per-track store and a bulk (overlay) store of the same packed tracks: same names, metadata, detail, geometry bytes
import json, random, sys
from vc_object_store.locations import open_store
A, B, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open_store(A, read_only=True) as a, open_store(B, read_only=True) as b:
    ra = {r["name"]: r for r in a.objects_by_kind("track")}; rb = {r["name"]: r for r in b.objects_by_kind("track")}
    print("track count:", len(ra), len(rb), "| same set of source ids:", set(ra) == set(rb))
    random.seed(1); sample = random.sample(sorted(ra), min(N, len(ra)))
    def meta(r): m = r["meta"]; return json.loads(m) if isinstance(m, str) else m
    bad = dict(meta=0, kind_tags=0, geometry_attrs=0, points=0)
    for n in sample:
        if meta(ra[n]) != meta(rb[n]): bad["meta"] += 1
        if (ra[n]["kind"], ra[n]["tags"], ra[n]["retired"]) != (rb[n]["kind"], rb[n]["tags"], rb[n]["retired"]): bad["kind_tags"] += 1
        ga, gb = a.current_geometry(ra[n]["uuid"]), b.current_geometry(rb[n]["uuid"])
        if (ga["kind"], ga["format_version"], ga["coordinate_system"], ga["attributes"]) != (gb["kind"], gb["format_version"], gb["coordinate_system"], gb["attributes"]): bad["geometry_attrs"] += 1
        pa, pb = a.read_geometry_array(ga["uuid"], "points"), b.read_geometry_array(gb["uuid"], "points")
        if pa != pb: bad["points"] += 1
    print(f"sampled {len(sample)} tracks; mismatches {bad}")
