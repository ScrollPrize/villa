"""Calibrate the scintillator light diffusion correction (sld=fraction=F,length=L) of a scan from its own slices.

Air must reconstruct to zero. Long-range detector glare spreads the object's shadow over the rays next to it, so the
air around (and inside) the object reads positive, falling off over millimetres. This script reconstructs one or more
slices for a grid of (fraction, length), measures the air level against the distance to the object, and reports the
pair that brings it closest to zero.

    python evaluation/calibrate_sld.py --projections PROJ --cache_dir CAL --out OUT --z 2698
    python evaluation/calibrate_sld.py --projections PROJ --cache_dir CAL --out OUT --z 1798 2698 3598 \
        --fractions 0.05 0.1 0.2 --lengths 50 100 200 400

The air mask comes from a reference slice without the correction (FBP, Paganin): pixels below --object_threshold,
at least --guard pixels from the object and --guard pixels inside the reconstruction disk. Slices are box-binned by
--bin first. Air is split into the open air around the object and the air it encloses (not reachable from the disk's
edge: inside a case, between its walls and the sample), each in bands of distance to the object. The air level is the
median of a band (object fragments below the threshold do not move it); the cost of a pair is the RMS of the band
levels, open and enclosed weighted equally, over the slices. Output: OUT/sld_calibration.json (every pair's
bands and cost, the best pair) and a table on stdout. Use the result as  --set sld=fraction=F,length=L.

The fit only sees the glare the model can express (K = (delta + f t) / (1 + f), t ~ exp(-r / l) / r): check the
remaining band levels of the best pair, not only its rank.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from helirec.filters import parse_chain
from helirec.recon import HelicalRecon

try:
    from .levels import BANDS, air_bands, band_levels, box_bin
except ImportError:   # run as a script
    from levels import BANDS, air_bands, band_levels, box_bin


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--projections", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--z", type=int, nargs="+", required=True, help="volume slice indices (view height - first doable)")
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.05, 0.1, 0.15, 0.2, 0.3])
    ap.add_argument("--lengths", type=float, nargs="+", default=[50, 100, 200, 400, 800])
    ap.add_argument("--filters", default="paganin(delta_beta=1000)", help="filter chain of the test slices")
    ap.add_argument("--bin", type=int, default=4)
    ap.add_argument("--object_threshold", type=float, default=0.04, help="cm^-1, on the binned reference slice")
    ap.add_argument("--guard", type=int, default=24, help="pixels kept clear of the object and of the disk's edge")
    ap.add_argument("--sld_bin", type=int, default=8)
    ap.add_argument("--sld_stride", type=int, default=10)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    r = HelicalRecon(a.projections, cache_dir=a.cache_dir, device=a.device,
                     params=dict(backprojector="fbp", filters=parse_chain(a.filters)))
    heights = [z + r.lo for z in a.z]
    for z, h in zip(a.z, heights):
        if not r.lo <= h < r.hi:
            sys.exit(f"slice z {z} outside the doable range [0, {r.hi - r.lo})")

    def levels(labels=None):
        out, lab = [], []
        for i, h in enumerate(heights):
            img = box_bin(r.reconstruct_heights(h, h + 1)[0], a.bin)
            lab.append(air_bands(img, a.bin, a.object_threshold, a.guard) if labels is None else labels[i])
            out.append(band_levels(img, lab[-1]))
        return np.array(out), lab

    nb = len(BANDS) - 1

    def cost(lv):   # open and enclosed air weighted equally (enclosed alone if there is no open air, and vice versa)
        parts = [np.mean(lv[:, s][np.isfinite(lv[:, s])] ** 2) for s in (slice(0, nb), slice(nb, 2 * nb)) if np.isfinite(lv[:, s]).any()]
        return float(np.sqrt(np.mean(parts)))
    t0 = time.perf_counter()
    base, labels = levels()
    print(f"bands (px from the object, beyond the {a.guard} px guard): " + " ".join(f"{BANDS[i]}-{BANDS[i + 1]}" for i in range(nb))
          + "   | open air, then enclosed air (cm^-1)")

    def show(name, lv):
        ok = np.isfinite(lv)
        m = np.where(ok.any(0), np.where(ok, lv, 0).sum(0) / np.maximum(ok.sum(0), 1), np.nan)   # per band, over the slices
        fmt = lambda v: " ".join(f"{x:+.4f}" for x in v if np.isfinite(x))
        print(f"{name:26s} cost {cost(lv):.5f}  {fmt(m[:nb])} | {fmt(m[nb:])}", flush=True)
    show("no correction", base)
    grid = []
    for length in a.lengths:
        for fraction in a.fractions:
            r.set_params(sld=dict(fraction=fraction, length=length, bin=a.sld_bin, stride=a.sld_stride))
            lv, _ = levels(labels)
            grid.append(dict(fraction=fraction, length=length, cost=cost(lv), bands=lv.tolist()))
            show(f"fraction={fraction:g},length={length:g}", lv)
    best = min(grid, key=lambda g: g["cost"])
    result = dict(projections=os.path.abspath(a.projections), z=a.z, filters=a.filters, bands_px=list(BANDS), guard=a.guard,
                  bin=a.bin, object_threshold=a.object_threshold, none=dict(cost=cost(base), bands=base.tolist()),
                  grid=grid, best=dict(fraction=best["fraction"], length=best["length"], cost=best["cost"]),
                  sld=f"fraction={best['fraction']:g},length={best['length']:g}", seconds=time.perf_counter() - t0)
    with open(os.path.join(a.out, "sld_calibration.json"), "w") as f:
        json.dump(result, f, indent=1)
    print(f"best: --set sld={result['sld']}   (cost {best['cost']:.5f}, none {cost(base):.5f})")
    edge = best["fraction"] in (min(a.fractions), max(a.fractions)) or best["length"] in (min(a.lengths), max(a.lengths))
    if edge and len(grid) > 1:
        print("the best pair is on the edge of the grid: extend --fractions / --lengths")
    return result


if __name__ == "__main__":
    main()
