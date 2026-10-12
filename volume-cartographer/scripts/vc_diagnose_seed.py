"""Seed triage reporter for vc_grow_seg_from_seed.

Given a seed voxel (x, y, z) and a surface-prediction zarr volume, diagnoses WHY
a growth run from that seed is likely to fail or trigger the #1906
on-prediction support warning -- BEFORE running the (expensive) growth.

Failure-mode taxonomy (primary class + edge flag):
  SEED_VOID      seed voxel has no prediction and the neighborhood is empty:
                 nothing to grow on.
  SPARSE_REGION  too little prediction in the seed's local patch (< 8 vx):
                 growth wanders off immediately.
  TRAPPED        seed patch is dense but prediction does not extend into the
                 growth horizon (8-32 vx): the seed sits on an isolated blob,
                 growth is trapped and on-prediction support collapses.
  EDGE           seed within EDGE_DIST of a volume face (reported as a flag):
                 growth runs into the boundary.
  HEALTHY        dense prediction at the seed and through the growth horizon:
                 if growth still warns, the cause is drift during growth,
                 not the seed.

Thresholds INNER_DENSITY_MIN / OUTER_DENSITY_MIN were set by 5-fold
cross-validation on the 576^3 PHerc1203 L2 crop (165 seeds; mean CV
precision 0.953 / recall 0.950) and confirmed on 200 fresh seeds grown
with vc_grow_seg_from_seed (precision 0.952 / recall 0.922 against
support < 40%). Starting points, not universal constants.

Usage:
  python vc_diagnose_seed.py --zarr crop.zarr --seed 470,227,87
  python vc_diagnose_seed.py --zarr crop.zarr --seeds seeds.tsv --out report.tsv
"""

import argparse
import math
import sys

import numpy as np
import zarr

NEIGHBORHOOD_HALF = 32   # growth horizon: generations(8) * step_size(4)
EDGE_DIST = 24           # seeds closer than this to a face get the EDGE flag
INNER_R = 8              # inner shell radius (voxels): the seed's local patch
# Calibrated by 5-fold cross-validation on the 576^3 PHerc1203 L2 crop
# (165 seeds, 2026-10-03 sweep): CV-selected thresholds were stable at
# inner=0.15, outer in {0.40, 0.45} across all folds
# (mean CV precision 0.953 / recall 0.950 / F1 0.950).
# Treat these as starting points, not universal constants.
INNER_DENSITY_MIN = 0.15
OUTER_DENSITY_MIN = 0.40


def load_volume(zarr_path):
    store = zarr.open(zarr_path, mode="r")
    # Multiscales layout (vc prediction zarrs): the data lives under "0".
    # A flat zarr array at the path root is also accepted.
    try:
        return store["0"]
    except (KeyError, IndexError, ValueError, TypeError):
        return store


def shell_profile(arr, x, y, z):
    """Nonzero-prediction density in concentric shells around the seed.

    Shells (voxel radii): [0,8), [8,16), [16,24), [24,65). The inner shell
    covers the seed's local patch; the outer shells cover the growth horizon
    (generations * step_size). Returns (inner, outer_mean)."""
    sz, sy, sx = arr.shape  # (z, y, x)
    h = NEIGHBORHOOD_HALF
    z0, z1 = max(0, z - h), min(sz, z + h + 1)
    y0, y1 = max(0, y - h), min(sy, y + h + 1)
    x0, x1 = max(0, x - h), min(sx, x + h + 1)
    nb = np.asarray(arr[z0:z1, y0:y1, x0:x1]) > 0
    gz, gy, gx = np.mgrid[z0:z1, y0:y1, x0:x1]
    d2 = (gz - z) ** 2 + (gy - y) ** 2 + (gx - x) ** 2
    shells = []
    for r0, r1 in ((0, INNER_R), (INNER_R, 16), (16, 24), (24, 65)):
        m = (d2 >= r0 * r0) & (d2 < r1 * r1)
        shells.append(float(nb[m].mean()) if m.sum() > 0 else 0.0)
    return shells[0], sum(shells[1:]) / 3.0


def triage_seed(arr, x, y, z):
    """Classify one seed. Coordinates are volume voxel indices (x, y, z)."""
    shape = arr.shape  # (z, y, x)
    sz, sy, sx = shape
    for name, v, lim in (("x", x, sx), ("y", y, sy), ("z", z, sz)):
        if not (0 <= v < lim):
            return {
                "class": "SEED_OUT_OF_BOUNDS",
                "edge": True,
                "seed_val": -1,
                "inner": 0.0,
                "outer": 0.0,
                "dist_edge": 0,
                "action": f"seed {name}={v} is outside the volume (size {lim}); fix the seed coordinates.",
            }

    seed_val = int(arr[z, y, x])
    inner, outer = shell_profile(arr, x, y, z)
    dist_edge = min(x, y, z, sx - 1 - x, sy - 1 - y, sz - 1 - z)
    edge = dist_edge < EDGE_DIST

    if seed_val == 0 and inner < 0.02 and outer < 0.02:
        cls = "SEED_VOID"
        action = "seed sits in empty prediction; move it onto a lit region."
    elif inner < INNER_DENSITY_MIN and outer < OUTER_DENSITY_MIN:
        cls = "SPARSE_REGION"
        action = ("too little prediction near the seed; growth will wander. "
                  "Move the seed to a denser patch of the sheet.")
    elif outer < OUTER_DENSITY_MIN:
        cls = "TRAPPED"
        action = ("seed sits on an isolated blob: prediction is dense at the "
                  "seed but does not extend, so growth is trapped and support "
                  "collapses. Move the seed to the nearest extended sheet.")
    else:
        cls = "HEALTHY"
        action = ("seed looks growable; if growth still warns, the surface "
                  "drifted during growth rather than starting badly.")

    return {
        "class": cls,
        "edge": edge,
        "seed_val": seed_val,
        "inner": round(inner, 4),
        "outer": round(outer, 4),
        "dist_edge": dist_edge,
        "action": action,
    }


def main():
    ap = argparse.ArgumentParser(description="Triage a grow_seg seed against a prediction volume.")
    ap.add_argument("--zarr", required=True, help="prediction zarr volume path")
    ap.add_argument("--seed", help="single seed as x,y,z")
    ap.add_argument("--seeds", help="TSV with x,y,z tab-separated columns (with header)")
    ap.add_argument("--out", help="output TSV path for batch mode")
    args = ap.parse_args()

    arr = load_volume(args.zarr)

    if args.seed:
        try:
            x, y, z = (int(v) for v in args.seed.split(","))
        except ValueError:
            ap.error("--seed must be three integers as x,y,z")
        r = triage_seed(arr, x, y, z)
        print(f"seed=({x},{y},{z}) class={r['class']} edge={r['edge']} "
              f"seed_val={r['seed_val']} inner={r['inner']} outer={r['outer']} "
              f"dist_edge={r['dist_edge']}")
        print(f"action: {r['action']}")
        return 0

    if args.seeds:
        import csv
        seeds = []
        with open(args.seeds) as f:
            first = f.readline()
            f.seek(0)
            # TSV with x/y/z header, or bare "x y z" / "x,y,z" lines.
            if first.lstrip().startswith("x"):
                for row in csv.DictReader(f, delimiter="\t"):
                    seeds.append((int(row["x"]), int(row["y"]), int(row["z"])))
            else:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.replace(",", " ").split()
                    seeds.append((int(parts[0]), int(parts[1]), int(parts[2])))
        rows = [(x, y, z, triage_seed(arr, x, y, z)) for x, y, z in seeds]
        out = open(args.out, "w") if args.out else sys.stdout
        print("x\ty\tz\tclass\tedge\tseed_val\tinner\touter\tdist_edge\taction", file=out)
        for x, y, z, r in rows:
            print(f"{x}\t{y}\t{z}\t{r['class']}\t{r['edge']}\t{r['seed_val']}\t"
                  f"{r['inner']}\t{r['outer']}\t{r['dist_edge']}\t{r['action']}", file=out)
        if args.out:
            out.close()
        return 0

    ap.error("provide --seed or --seeds")


if __name__ == "__main__":
    sys.exit(main())
