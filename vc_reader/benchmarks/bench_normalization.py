"""Real CT crop reads: explicit NumPy casting/normalization versus native output.

Run via run_native.py with --methods numpy on a saved baseline extension,
then with --methods numpy,native on the new extension. Every output is checked.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from _common import environment, chunk_files, evict_files, filesystem_type
from bench_sample import DEFAULT_ARRAY
from vc_reader.reader import ZarrReader


def numpy_normalize(raw, dtype, mode):
    out = raw.astype(dtype, copy=False)
    if mode == "cast" or not out.size:
        return out
    mask = out != 0 if mode == "zscore" else None
    values = out[mask] if mask is not None else out
    if not values.size:
        out.fill(0)
        return out
    if mode == "minmax":
        center = float(values.min())
        divisor = float(values.max()) - center
    else:
        center = float(values.mean(dtype=np.float64))
        divisor = float(values.std(dtype=np.float64))
    if not divisor:
        out.fill(0)
    else:
        out -= center
        out /= divisor
        if mask is not None:
            out[~mask] = 0
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--array", default=DEFAULT_ARRAY)
    ap.add_argument("--baseline-reader", type=Path, help="saved zarr_reader.py matching a baseline extension")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--edge", type=int, default=128)
    ap.add_argument("--methods", default="numpy,native")
    ap.add_argument("--dtypes", default="float32,float64")
    ap.add_argument("--modes", default="cast,minmax,zscore")
    ap.add_argument("--paths", default="block,affine")
    ap.add_argument("--cache-state", choices=["warm", "cold"], default="warm")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    reader_class = ZarrReader
    if args.baseline_reader:
        spec = importlib.util.spec_from_file_location("vc_reader.reader._benchmark_baseline", args.baseline_reader)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        reader_class = module.ZarrReader
    methods = args.methods.split(",")
    if args.reps < 1 or args.edge < 1 or not set(methods) <= {"numpy", "native"}:
        ap.error("positive reps/edge and valid methods required")
    if args.cache_state == "cold" and filesystem_type(Path(args.array)) in ("tmpfs", "ramfs"):
        ap.error("cold reads require a disk-backed array")
    shape = (args.edge,) * 3
    origin = np.array([9300, 4000, 4000])
    axes = np.array([[.9, -.15, .1], [.1, .8, -.2], [.05, .2, .7]])
    corners = np.array(list(np.ndindex(2, 2, 2))) * (args.edge - 1)
    corners = np.concatenate([corners + origin, corners @ axes.T + origin])
    lo = np.floor(corners.min(0)).astype(int)
    hi = np.ceil(corners.max(0)).astype(int) + 1
    files = chunk_files(args.array, [(lo, hi-lo)])
    if not all(Path(f).is_file() for f in files):
        raise RuntimeError("all enclosing chunks must be stored")
    rows = []
    for path in args.paths.split(","):
        for dtype_name in args.dtypes.split(","):
            dtype = np.dtype(dtype_name)
            for mode in args.modes.split(","):
                timings = {m: [] for m in methods}
                r = reader_class.open(args.array, threads=args.threads)
                def read(method):
                    if method == "numpy":
                        raw = r.read(origin, shape) if path == "block" else r.read_affine(shape, origin, axes, dtype=dtype)
                        return numpy_normalize(raw, dtype, mode)
                    kwargs = dict(dtype=dtype, normalization=None if mode == "cast" else mode)
                    return r.read(origin, shape, **kwargs) if path == "block" else r.read_affine(shape, origin, axes, **kwargs)
                try:
                    expected = read("numpy")
                    for rep in range(args.reps + 1):
                        for method in methods[::(-1 if rep % 2 else 1)]:
                            if args.cache_state == "cold":
                                r.close(); evict_files(files)
                                r = reader_class.open(args.array, threads=args.threads)
                            start = time.perf_counter()
                            got = read(method)
                            elapsed = (time.perf_counter() - start) * 1000
                            np.testing.assert_allclose(got, expected, rtol=3e-6, atol=3e-6)
                            if rep: timings[method].append(elapsed)
                            del got
                finally:
                    r.close()
                for method, times in timings.items():
                    row = dict(path=path, dtype=dtype_name, mode=mode, method=method,
                               mean=statistics.mean(times), median=statistics.median(times),
                               minimum=min(times), maximum=max(times), ms=times)
                    rows.append(row)
                    print(f"{path:6} {dtype_name} {mode:6} {method:6} mean/min/median/max ms: "
                          f"{row['mean']:.3f}/{row['minimum']:.3f}/{row['median']:.3f}/{row['maximum']:.3f}", flush=True)
    args.output.write_text(json.dumps(dict(command=sys.argv, environment=environment(), array=args.array,
        shape=shape, origin=origin.tolist(), axes=axes.tolist(), threads=args.threads, cache=args.cache_state,
        reps=args.reps, rows=rows), indent=2))


if __name__ == "__main__":
    main()
