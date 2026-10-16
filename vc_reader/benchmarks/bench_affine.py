"""Oriented crops on real CT: explicit coordinate grids versus native affine reads.

    python benchmarks/bench_affine.py --reps 7 --cache-state both --output /tmp/affine.json

``points`` includes coordinate construction; ``points-prebuilt`` excludes it.
``affine-out`` reuses the final output. All methods return raw float32 intensity,
without normalization. This does not benchmark the tracers' Numba/GPU paths.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from _common import chunk_files, environment, evict_files, filesystem_type
from bench_sample import DEFAULT_ARRAY
from vc_reader.reader import ZarrReader


def coordinates(shape, origin, axes):
    # Sparse axes avoid allocating three extra full meshgrid arrays.
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    return origin + z[..., None] * axes[:, 0] + y[..., None] * axes[:, 1] + x[..., None] * axes[:, 2]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--array", default=DEFAULT_ARRAY)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--cache-state", choices=["warm", "cold", "both"], default="warm")
    ap.add_argument("--methods", default="points,points-prebuilt,affine,affine-out")
    ap.add_argument("--only", default="", help="comma-separated crop names: cube64,cube128,plane")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    methods = args.methods.split(",")
    if args.reps < 1 or not set(methods) <= {"points", "points-prebuilt", "affine", "affine-out"}:
        ap.error("positive reps and valid methods required")
    if args.cache_state != "warm" and filesystem_type(Path(args.array)) in ("tmpfs", "ramfs"):
        ap.error("cold reads require a disk-backed array")
    # Same populated region used by bench_sample; a rotated anisotropic grid.
    u = np.array([0.1, 0.2, 0.97]); u /= np.linalg.norm(u)
    w = np.array([0.95, 0.3, 0.0]); w -= u * (u @ w); w /= np.linalg.norm(w)
    v = np.cross(w, u)
    axes = np.column_stack([0.8*w, 0.9*v, 0.73*u])
    origin = np.array([9300.25, 4000.5, 4000.75])
    rows = []
    for name, shape in [("cube64", (64, 64, 64)), ("cube128", (128, 128, 128)), ("plane", (1, 512, 512))]:
        if args.only and name not in args.only.split(","):
            continue
        points = coordinates(shape, origin, axes)
        corners = np.array(list(itertools.product(*[(0, n-1) for n in shape]))) @ axes.T + origin
        lo = np.floor(corners.min(0)).astype(int)
        hi = np.ceil(corners.max(0)).astype(int) + 1
        files = chunk_files(args.array, [(lo, hi-lo)])
        if not files or not all(Path(f).is_file() for f in files):
            raise RuntimeError("benchmark region must have all source chunks stored")
        with ZarrReader.open(args.array, threads=args.threads) as reader:
            reference = reader.sample(points)
        states = ["cold", "warm"] if args.cache_state == "both" else [args.cache_state]
        for state in states:
            timings = {method: [] for method in methods}
            output = np.empty(shape, np.float32)
            reader = ZarrReader.open(args.array, threads=args.threads)
            try:
                for rep in range(args.reps + 1):  # first repetition is untimed warm-up
                    for method in methods[::(-1 if rep % 2 else 1)]:
                        if state == "cold":
                            reader.close()
                            evict_files(files)
                            reader = ZarrReader.open(args.array, threads=args.threads)
                        start = time.perf_counter()
                        if method == "points":
                            got = reader.sample(coordinates(shape, origin, axes))
                        elif method == "points-prebuilt":
                            got = reader.sample(points)
                        else:
                            got = reader.read_affine(shape, origin, axes, out=output if method == "affine-out" else None)
                        elapsed = time.perf_counter() - start
                        # Allow the floating-point contraction differences supported on other platforms.
                        np.testing.assert_allclose(got, reference, rtol=1e-6, atol=1e-5)
                        if rep:
                            timings[method].append(elapsed * 1000)
                        del got
            finally:
                reader.close()
            for method, times in timings.items():
                row = dict(crop=name, shape=shape, origin=origin.tolist(), axes=axes.tolist(), cache=state,
                           method=method, ms=times, mean=statistics.mean(times), median=statistics.median(times),
                           minimum=min(times), maximum=max(times), coordinate_bytes=points.nbytes)
                rows.append(row)
                print(f"{name:8} {state:4} {method:15} mean/min/median/max ms: "
                      f"{row['mean']:.3f}/{row['minimum']:.3f}/{row['median']:.3f}/{row['maximum']:.3f}", flush=True)
    args.output.write_text(json.dumps(dict(command=sys.argv, environment=environment(), array=args.array,
                                         threads=args.threads, reps=args.reps, rows=rows), indent=2))


if __name__ == "__main__":
    main()
