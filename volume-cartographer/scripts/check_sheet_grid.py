#!/usr/bin/env python3
"""Measure local grid folds and spacing in a level-0 tifxyz surface (NumPy/tifffile)."""
import argparse
import json
from pathlib import Path

import numpy as np
import tifffile


def analyze(folder):
    points = np.stack([tifffile.imread(folder / f"{axis}.tif") for axis in "xyz"], axis=-1).astype(float)
    valid = np.isfinite(points).all(axis=-1) & (points >= 0).all(axis=-1)
    dr, dc = np.diff(points, axis=0), np.diff(points, axis=1)
    vr, vc = valid[:-1] & valid[1:], valid[:, :-1] & valid[:, 1:]
    quads = valid[:-1, :-1] & valid[1:, :-1] & valid[:-1, 1:] & valid[1:, 1:]
    t1 = np.cross(points[:-1, 1:] - points[:-1, :-1], points[1:, :-1] - points[:-1, :-1])
    t2 = np.cross(points[1:, :-1] - points[1:, 1:], points[:-1, 1:] - points[1:, 1:])
    normals = t1 + t2

    def reversed_vectors(a, b, mask):
        return int(np.count_nonzero(mask & (np.sum(a * b, axis=-1) < 0)))

    def spacing(steps, mask):
        values = np.linalg.norm(steps, axis=-1)[mask]
        return dict(zip(["min", "p01", "median", "p99", "max"],
                        np.percentile(values, [0, 1, 50, 99, 100]).tolist())) if values.size else {}

    return {
        "shape": list(points.shape[:2]),
        "row_spacing": spacing(dr, vr),
        "column_spacing": spacing(dc, vc),
        "row_order_reversals": reversed_vectors(dr[:-1], dr[1:], vr[:-1] & vr[1:]),
        "column_order_reversals": reversed_vectors(dc[:, :-1], dc[:, 1:], vc[:, :-1] & vc[:, 1:]),
        "crossed_quads": reversed_vectors(t1, t2, quads),
        "flipped_row_neighbors": reversed_vectors(normals[:-1], normals[1:], quads[:-1] & quads[1:]),
        "flipped_column_neighbors": reversed_vectors(normals[:, :-1], normals[:, 1:], quads[:, :-1] & quads[:, 1:]),
        "degenerate_quads": int(np.count_nonzero(quads & ((np.linalg.norm(t1, axis=-1) < 1e-6) |
                                                                        (np.linalg.norm(t2, axis=-1) < 1e-6)))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("surface", type=Path)
    parser.add_argument("--require-no-folds", action="store_true",
                        help="Fail if any local reversal/degeneracy is found; use only on suitable regression patches.")
    args = parser.parse_args()
    result = analyze(args.surface)
    print(json.dumps(result, indent=2))
    if args.require_no_folds and any(result[key] for key in (
        "row_order_reversals", "column_order_reversals", "crossed_quads",
        "flipped_row_neighbors", "flipped_column_neighbors", "degenerate_quads",
    )):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
