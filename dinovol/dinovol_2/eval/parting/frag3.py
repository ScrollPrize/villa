"""The frozen Frag3 test of Parting, pinned: data, site rule, sites and the published result (29 Oct 2026).

Object: PHercParis1Fr34 (Frag3 in the open data), fragment 34 of PHerc. Paris 1. Truth: the six layer segments the
Vesuvius Challenge team traced in October 2024 (`paths/` in its volpkg), read as ordered pointsets, scaled from
3.24 um to 7.91 um voxels, every 12th row and every 2nd column kept, and ordered by geometry (low y to high y).
Every vertex whose next layer is not above it is a crossing point, and no site window comes within 16 voxels of
one. Scan: the organizers' standardized 54 keV, 3.24 um volume, box-averaged to 7.91 um. Public record:
https://github.com/ChaseHendrick/Scrolls/blob/main/docs/logs/2026-10-29-parting-fragment-contact.md
"""
from pathlib import Path

import numpy as np

from . import geometry

VOLPKG = "https://dl.ash2txt.org/fragments/Frag3/PHercParis1Fr34.volpkg"
VOLUME = f"{VOLPKG}/volumes_standardized/54keV_3.24um_.zarr/0"
RESAMPLE = "3.24:7.91"
SRC_UM, DST_UM = 3.24, 7.91
SHAPE_ZYX = (2727, 590, 2586)          # the 7.91 um frame of the 6656 x 1440 x 6312 level 0
ROW_STEP, COL_STEP = 12, 2
NEED = 0.995                           # share of matched vertices whose next layer is above
# The six layer segments' pointset.vcps files by sha256, in stack order (L0 lowest y to L5 highest). The segments are
# found in the volpkg's paths/ listing at run time and matched by these hashes, so no segment is named here.
POINTSET_SHA256 = (
    "cdc35f9c2ab5d7d2629bec7df6cf8ebba3f800c7a01ef3a967619f4ac08b687b",
    "703288d7a43ebc3cb34a2e0d85b878e1f27e7f2ca55a6f60bf0c0592fe0d25f9",
    "443dbc513c0e29254556ed5c2d742b2a55d3f66c96677f501168d5986e0fdb7a",
    "984e0a800f8f6d664088d602a5f56d5a0c9f2f664456d12c158781a959f60fa2",
    "58604758a9b61cc257905a9d779bd5ee505b13a0a00ab7b1e312f7f277e98281",
    "40deae642728d5cc2e8894f7715e0112e468565481f487d41fe2287221a4b6b1")
CROSSINGS = {"points": 719, "sha256": "4679962df83c1e1ae2f1d106a95037bff3b469695dc1e7f8f446f62fcffa6562"}
SITE_RULE = {"seed": 20261030, "dense": 1, "spacing": 48.0,
             "caps": {"touching": 30, "apart": 15, "fused": 8, "mixed": 4}, "stop_when_full": True,
             "exclude_vox": 16.0}
# canonical sha256 of the frozen site list (geometry.sites_sha256): 283 sites, 150 touching over 5 pairs. No site file
# ships with this tool; rebuild_sites makes the list again from the public pointsets and must give this hash.
SITES_SHA = "c67ee0de3e40aace19030a16f4a851027fb5c7240b389359959be528a4b92c27"
SITES_TALLY = {"sites": 283, "strata": {"touching": 150, "apart": 75, "fused": 38, "mixed": 20}, "touching_pairs": 5}
TOP_PAIR = "L4-L5"                     # the pair with the crossing zone; reported without it, never judged
MODEL_SHA = "f0684ab2ec5725ad9050e08bcdf73df22983960ba906d222cb9c8068a0fdc601"
EXPECTED = {    # the frozen run (rtx-pro-6000, bf16), read once about 21:10 UTC 29 Oct 2026: Model output
    "touching": {"sites": 150, "pairs": 5, "parting": 0.8952, "dino_base": 0.8392},
    "apart": {"sites": 75, "pairs": 5, "parting": 0.9623, "dino_base": 0.9438},
    "fused": {"sites": 38, "pairs": 5, "parting": 0.633, "dino_base": 0.6108},
    "primary": {"parting": 0.8952, "dino_base": 0.8392, "delta": 0.056, "parting_bound": 0.8452,
                "delta_bound": 0.0368},
    "permutation_mean": 0.50039,
    "verdict": "PASS"}


def segment_names():
    """The segment folders listed under the volpkg's paths/ (a public directory index)."""
    import re
    import urllib.request
    with urllib.request.urlopen(f"{VOLPKG}/paths/", timeout=60) as r:
        page = r.read().decode("utf-8", "replace")
    return sorted(set(re.findall(r'href="(\d{14})/"', page)))


def layer_grids(folder):
    """({L0..L5: (H, W, 3) float32 xyz grids in the 7.91 um frame}, record) from FOLDER/SEGMENT/pointset.vcps, each
    file matched to its layer by its pinned sha256."""
    found = {}
    for p in sorted(Path(folder).glob("*/pointset.vcps")):
        got = geometry.sha256_file(p)
        if got in POINTSET_SHA256:
            found[POINTSET_SHA256.index(got)] = p
    missing = [i for i in range(len(POINTSET_SHA256)) if i not in found]
    if missing:
        raise SystemExit(f"{folder}: no pointset.vcps with the pinned sha256 of layer(s) "
                         f"{', '.join(f'L{i}' for i in missing)}; run frag3 without --no-download")
    out, rec = {}, {}
    for i in range(len(POINTSET_SHA256)):
        p = found[i]
        g = geometry.read_vcps(p)[::ROW_STEP, ::COL_STEP] * (SRC_UM / DST_UM)
        out[f"L{i}"] = g.astype(np.float32)
        rec[f"L{i}"] = {"segment": p.parent.name, "vcps_sha256": POINTSET_SHA256[i], "grid": list(g.shape)}
    return out, rec


def fetch_pointsets(folder):
    """Download every segment's pointset.vcps listed under paths/ (six files, about 160 MB each) into
    FOLDER/SEGMENT/, skipping files already there with a pinned sha256."""
    import urllib.request
    for name in segment_names():
        p = Path(folder) / name / "pointset.vcps"
        if p.exists() and geometry.sha256_file(p) in POINTSET_SHA256:
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".part")
        with urllib.request.urlopen(f"{VOLPKG}/paths/{name}/pointset.vcps", timeout=120) as r, open(tmp, "wb") as f:
            while True:
                b = r.read(1 << 22)
                if not b:
                    break
                f.write(b)
        tmp.replace(p)
        print(f"fetched {name}/pointset.vcps", flush=True)


def rebuild_sites(folder):
    """The Frag3 sites made again from the pointsets: (sites, record). Checks the stack order and the crossings."""
    grids, rec = layer_grids(folder)
    order, cross = geometry.stack_layers(grids, NEED)
    got = geometry.npy_sha256(cross)
    if len(cross) != CROSSINGS["points"] or got != CROSSINGS["sha256"]:
        raise SystemExit(f"crossings: {len(cross)} points, sha256 {got}; the frozen run had {CROSSINGS['points']}, "
                         f"{CROSSINGS['sha256']}")
    r = SITE_RULE
    rule = geometry.Rule(seed=r["seed"], dense=r["dense"], spacing=r["spacing"], caps=r["caps"],
                         stop_when_full=r["stop_when_full"], exclude=cross, exclude_vox=r["exclude_vox"])
    sites = geometry.make_sites(grids, SHAPE_ZYX, rule)
    return sites, {"layers": rec, "stack": order, "crossings": {"points": len(cross), "sha256": got},
                   "rule": rule.record(), "sites_sha256": geometry.sites_sha256(sites)}
