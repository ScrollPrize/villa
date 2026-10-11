#!/usr/bin/env python
"""
Compare pgs-global-scaler candidates by the geometry of their scaled marker corners.

For each candidate directory under CANDIDATES (one per include list, holding scaler.log and
landmarks-scaled.ply as written by the handover README's scaler loop), reports the scale,
detection and triangulation counts, and on the scaled corners: marker side lengths (should
all be the marker size), off-plane residual (the markers lie on the flat turntable) and the
radius of the marker ring (the same turntable gives the same radius in every candidate and
every capture session). A missing or badly triangulated corner shows up as a wide side-length
spread or a large off-plane residual; prefer candidates whose corners are complete and tight,
and check that independent good candidates agree.

Usage:
    python scale_review.py CANDIDATES [--marker-size 31.5]
"""

import argparse
import os
import re
from pathlib import Path

import numpy as np


def read_ascii_ply_vertices(path):
    text = Path(path).read_text()
    head, body = text.split("end_header", 1)
    n = int(re.search(r"element vertex (\d+)", head).group(1))
    values = np.array(body.split(), float)
    return values.reshape(n, -1)[:, :3]


def corner_geometry(P, complete):
    """Off-plane rms/max and ring radius from all corners; side lengths only when every marker has 4 corners."""
    c = P.mean(0)
    _, _, Vt = np.linalg.svd(P - c)
    off = np.abs((P - c) @ Vt[2])
    out = dict(off_rms=np.sqrt((off ** 2).mean()), off_max=off.max())
    centres = P.reshape(-1, 4, 3).mean(1) if complete else P
    X = (centres - c) @ Vt[:2].T
    sol = np.linalg.lstsq(np.c_[2 * X, np.ones(len(X))], (X ** 2).sum(1), rcond=None)[0]
    out["ring"] = np.sqrt(sol[2] + (sol[:2] ** 2).sum())
    if complete:
        Q = P.reshape(-1, 4, 3)  # the scaler writes each marker's four corners consecutively
        out["sides"] = np.linalg.norm(Q - np.roll(Q, 1, axis=1), axis=2).ravel()
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("candidates", help="directory with one subdirectory per scale candidate")
    parser.add_argument("--marker-size", type=float, default=31.5, help="marker side passed to the scaler (mm)")
    args = parser.parse_args()
    print(f"{'candidate':14s} {'scale':>9s}  views  corners   side mean / sd / range (mm)     off-plane rms / max   ring radius")
    for name in sorted(os.listdir(args.candidates)):
        d = os.path.join(args.candidates, name)
        log_path = os.path.join(d, "scaler.log")
        if not os.path.isdir(d) or not os.path.exists(log_path):
            continue
        log = Path(log_path).read_text()
        scale = re.search(r"median scale:\s*([\d.]+)", log)
        if not scale or not os.path.exists(os.path.join(d, "landmarks-scaled.ply")):
            print(f"{name:14s} no result (see {log_path})")
            continue
        views = re.search(r"Detected landmarks in (\d+) of (\d+) views", log)
        tri = re.search(r"Triangulated (\d+) of (\d+) landmarks", log)
        P = read_ascii_ply_vertices(os.path.join(d, "landmarks-scaled.ply"))
        complete = tri and tri.group(1) == tri.group(2) and len(P) % 4 == 0
        g = corner_geometry(P, complete)
        sides = (f"{g['sides'].mean():6.2f} / {g['sides'].std():4.2f} / {g['sides'].min():5.1f}-{g['sides'].max():5.1f}"
                 if complete else "   (corners missing)        ")
        flag = "" if complete and abs(g["sides"].mean() - args.marker_size) < 1 and g["off_rms"] < 1 else "  <- check"
        print(f"{name:14s} {float(scale.group(1)):9.3f}  {views.group(1) if views else '?':>5s}  "
              f"{tri.group(1) + '/' + tri.group(2) if tri else '?':>7s}   {sides}     {g['off_rms']:5.2f} / {g['off_max']:5.2f}"
              f"        {g['ring']:6.1f}{flag}")


if __name__ == "__main__":
    main()
