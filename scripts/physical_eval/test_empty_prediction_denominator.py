#!/usr/bin/env python3
"""A slice is scored on its truth, whatever the prediction does.

The evaluator used to skip a slice whose prediction held fewer than 100
voxels, which took that slice's truth out of the denominator along with its
failures. A sparse prediction could then score well by disappearing from the
hard slices rather than by covering them. Raised by TAUIL-Abd-Elilah on the
villa PR; his own regression is at
https://github.com/TAUIL-Abd-Elilah/vesuvius-repro (test_physical_normalization_ab.py).

This builds three small zarr volumes over one truth and runs the shipped
entry point on each:

  empty    no predicted voxel at all      must score 0, not vanish
  sparse   a handful of voxels            must keep the same denominator
  dense    covers the sheets              must score well

The denominator has to be identical across all three, since the truth is.

  python3 test_empty_prediction_denominator.py

Exit 0 on success.
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numcodecs
import numpy as np

HERE = Path(__file__).resolve().parent
EVAL = HERE / "eval_surface_pred.py"
CH = 64
SHAPE = (64, 256, 256)


def write_zarr(root, arr, chunk=CH, compress=True):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    comp = numcodecs.Blosc() if compress else None
    meta = {
        "zarr_format": 2, "shape": list(arr.shape),
        "chunks": [chunk] * 3, "dtype": "|u1", "fill_value": 0,
        "order": "C", "dimension_separator": "/", "filters": None,
        "compressor": (comp.get_config() if comp else None),
    }
    (root / ".zarray").write_text(json.dumps(meta))
    nz, ny, nx = [-(-s // chunk) for s in arr.shape]
    for cz in range(nz):
        for cy in range(ny):
            for cx in range(nx):
                blk = np.zeros((chunk, chunk, chunk), np.uint8)
                sub = arr[cz * chunk:(cz + 1) * chunk,
                          cy * chunk:(cy + 1) * chunk,
                          cx * chunk:(cx + 1) * chunk]
                blk[:sub.shape[0], :sub.shape[1], :sub.shape[2]] = sub
                d = root / str(cz) / str(cy)
                d.mkdir(parents=True, exist_ok=True)
                raw = blk.tobytes()
                (d / str(cx)).write_bytes(comp.encode(raw) if comp else raw)


def build_truth():
    """Flat stacked sheets: material, its centerline, a recto band."""
    z, y, x = SHAPE
    lab = np.zeros(SHAPE, np.uint8)
    lab |= 1                                    # valid everywhere
    for y0 in range(20, y - 20, 16):
        lab[:, y0:y0 + 5, :] |= 2               # material, 5 voxels thick
        lab[:, y0 + 2, :] |= 4                  # centerline, the middle one
        lab[:, y0, :] |= 8                      # recto band, inward face
    return lab


def build_pred(kind):
    p = np.zeros(SHAPE, np.uint8)
    if kind == "empty":
        return p
    if kind == "sparse":
        p[:, 22, 10:40] = 1                     # a few voxels on one sheet
        return p
    for y0 in range(20, SHAPE[1] - 20, 16):     # dense: sits on every sheet
        p[:, y0 + 1, :] = 1
    return p


def run(labels, pred):
    r = subprocess.run(
        [sys.executable, str(EVAL), str(labels), str(pred), "1"],
        capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        raise SystemExit(f"evaluator failed on {pred}")
    return json.loads(r.stdout)


def main():
    tmp = Path(tempfile.mkdtemp(prefix="evaltest_"))
    try:
        lab = build_truth()
        labels = tmp / "labels.zarr"
        write_zarr(labels, lab)
        (labels / ".zattrs").write_text(json.dumps(
            {"origin_l1": [0, 0, 0], "bits": {
                "valid": 1, "material": 2, "centerline": 4,
                "recto_band": 8}}))

        out = {}
        for kind in ("empty", "sparse", "dense"):
            p = tmp / f"pred_{kind}.zarr"
            write_zarr(p, build_pred(kind))
            out[kind] = run(labels, p)
            o = out[kind]
            print(f"{kind:7s} n_centerline {o['n_centerline']:8d}  "
                  f"n_arcs {o['n_arcs']:5d}  recall_37um {o['recall_37um']:.4f}"
                  f"  arc_recall {o['arc_recall']:.4f}"
                  f"  fully_missed {o['arc_fully_missed']:.4f}")

        fail = []
        base = out["dense"]["n_centerline"]
        if base <= 0:
            fail.append("dense run scored no centerline points at all")
        for kind in ("empty", "sparse"):
            if out[kind]["n_centerline"] != base:
                fail.append(
                    f"{kind} denominator {out[kind]['n_centerline']} differs "
                    f"from dense {base}: the truth left with the prediction")
            if out[kind]["n_arcs"] != out["dense"]["n_arcs"]:
                fail.append(f"{kind} arc denominator moved")
        e = out["empty"]
        for f in ("recall_19um", "recall_37um", "recall_56um", "arc_recall"):
            if e[f] != 0.0:
                fail.append(f"empty prediction scored {f}={e[f]}, expected 0")
        if e["arc_fully_missed"] != 1.0:
            fail.append(f"empty prediction fully_missed "
                        f"{e['arc_fully_missed']}, expected 1.0")
        if not (out["sparse"]["recall_37um"] < out["dense"]["recall_37um"]):
            fail.append("sparse did not score below dense")

        print()
        if fail:
            for f in fail:
                print("FAIL:", f)
            return 1
        print("PASS: the denominator is set by the truth alone; an empty "
              "prediction scores zero and keeps every arc as fully missed")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
