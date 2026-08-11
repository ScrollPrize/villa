#!/usr/bin/env python3
"""The side arms' un-nesting must not move the real or conditioned arms.

Adding the _full control variants meant reordering two filters: the old
side_stats took near-the-prediction first and the coherence floor second,
the current one takes coherence first so the whole coherent centerline set
stays available for the unconditioned arms. Set intersection says those
select the same points, which is the kind of claim worth handing to a
machine. This holds the pre-change implementation next to the shipped one
and runs both over synthetic stacked sheets.

  python3 test_side_arms_equivalence.py

Exit 0 means the real and conditioned arms are bit-identical and the
unconditioned arm is strictly wider.
"""
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_surface_pred as E


def old_side_stats(acc, tb, pb, pbs, recto_k, ys, xs, cd):
    """Verbatim pre-change implementation."""
    near = cd <= 3
    if near.sum() < 100:
        return
    yn, xn = ys[near], xs[near]
    nyf, nxf, cohf = E.normal_field(
        ndi.gaussian_filter(tb.astype(np.float32), 1.5))
    ny, nx = nyf[yn, xn], nxf[yn, xn]
    m = np.nonzero(tb)
    cy, cx = m[0].mean(), m[1].mean()
    flip = (ny * (yn - cy) + nx * (xn - cx)) < 0
    ny = np.where(flip, -ny, ny)
    nx = np.where(flip, -nx, nx)
    ok = cohf[yn, xn] > E.SIDE_COH_MIN
    if ok.sum() < 100:
        return
    yn, xn, ny, nx = yn[ok], xn[ok], ny[ok], nx[ok]
    dpred = ndi.distance_transform_edt(~pb)
    i, o = E._inward_counts(dpred, yn, xn, ny, nx)
    acc["side_in"] += i
    acc["side_out"] += o
    for band, key in ((pbs, "null"), (recto_k, "ideal")):
        if not band.any():
            continue
        d = ndi.distance_transform_edt(~band)
        sel = d[yn, xn] <= 3
        if sel.sum() <= 100:
            continue
        i, o = E._inward_counts(d, yn[sel], xn[sel], ny[sel], nx[sel])
        acc[f"side_{key}_in"] += i
        acc[f"side_{key}_out"] += o


def blank():
    return dict(side_in=0, side_out=0, side_null_in=0, side_null_out=0,
                side_ideal_in=0, side_ideal_out=0,
                side_null_full_in=0, side_null_full_out=0,
                side_ideal_full_in=0, side_ideal_full_out=0)


def make_slice(seed, H=420, W=420):
    """Curved stacked sheets, a prediction band offset to one side."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cy, cx = H / 2, W / 2
    r = np.hypot(yy - cy, xx - cx)
    pitch = 14.0 + rng.uniform(-2, 2)
    phase = rng.uniform(0, pitch)
    band = np.abs(((r + phase) % pitch) - pitch / 2)
    tb = band < 2.2
    # centerline: local ridge of the distance transform inside material
    dt = ndi.distance_transform_edt(tb)
    ctr = tb & (dt >= ndi.maximum_filter(dt, 3)) & (dt >= 1)
    # prediction: the material shifted radially outward by ~1.5 voxels,
    # with holes punched in, so it lands on one side of the centerline
    shift = 1.5
    rr = r - shift
    pb = np.abs(((rr + phase) % pitch) - pitch / 2) < 1.4
    holes = rng.random((H, W)) < 0.12
    pb = pb & ~ndi.binary_dilation(holes, iterations=2)
    # ideal recto band: inward-facing material boundary
    recto = tb & (np.abs(((r + phase + 1.6) % pitch) - pitch / 2) >= 2.2)
    pbs = np.zeros_like(pb)
    pbs[E.NULL_SHIFT:] = pb[:-E.NULL_SHIFT]
    return tb, pb, pbs, recto, ctr


def main():
    tot_old, tot_new = blank(), blank()
    checked = 0
    for seed in range(12):
        tb, pb, pbs, recto, ctr = make_slice(seed)
        if ctr.sum() < 500 or pb.sum() < 100:
            continue
        dpred = ndi.distance_transform_edt(~pb)
        ys, xs = np.nonzero(ctr)
        cd = dpred[ys, xs]
        a_old, a_new = blank(), blank()
        old_side_stats(a_old, tb, pb, pbs, recto, ys, xs, cd)
        E.side_stats(a_new, tb, pb, pbs, recto, ys, xs, cd)
        for k in a_old:
            tot_old[k] += a_old[k]
            tot_new[k] += a_new[k]
        checked += 1

    print(f"slices exercised: {checked}")
    shared = ["side_in", "side_out", "side_null_in", "side_null_out",
              "side_ideal_in", "side_ideal_out"]
    bad = [k for k in shared if tot_old[k] != tot_new[k]]
    for k in shared:
        flag = "MISMATCH" if tot_old[k] != tot_new[k] else "ok"
        print(f"  {k:22s} old {tot_old[k]:9d}  new {tot_new[k]:9d}  {flag}")
    print("  --- new-only arms ---")
    for k in ["side_null_full_in", "side_null_full_out",
              "side_ideal_full_in", "side_ideal_full_out"]:
        print(f"  {k:22s} {tot_new[k]:9d}")

    dec = tot_new["side_in"] + tot_new["side_out"]
    dn = tot_new["side_null_in"] + tot_new["side_null_out"]
    dnf = tot_new["side_null_full_in"] + tot_new["side_null_full_out"]
    if dec and dn and dnf:
        print(f"\n  inward          {tot_new['side_in']/dec:.4f}")
        print(f"  null cond       {tot_new['side_null_in']/dn:.4f}"
              f"   (n={dn})")
        print(f"  null full       {tot_new['side_null_full_in']/dnf:.4f}"
              f"   (n={dnf})")

    if checked < 3:
        print("\nINCONCLUSIVE: too few slices exercised the code path")
        return 2
    if bad:
        print(f"\nFAIL: real/conditioned arms moved: {bad}")
        return 1
    if dnf <= dn:
        print("\nSUSPECT: unconditioned null is not larger than conditioned;"
              " the new arm may not be selecting from the full set")
        return 1
    print("\nPASS: real and conditioned arms bit-identical, full arm wider")
    return 0


if __name__ == "__main__":
    sys.exit(main())
