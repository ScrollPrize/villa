"""Benchmark the CuPy vs NumPy backends for foundation/datasets/fibers-dataset/tools.py.

Usage (from the fibers-dataset directory):

    python bench/bench_tools.py
    python bench/bench_tools.py --sizes 64 128 256 --repeats 3

Outputs a Markdown table on stdout that can be pasted into a PR description.
Skips CuPy timings if cupy is not installed.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import numpy as np
from scipy import ndimage as scipy_ndimage

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import tools  # noqa: E402

try:
    import cupy as cp
    import cupyx.scipy.ndimage as cupy_ndimage

    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False


def _time_call(fn, *args, repeats=3, sync_cupy=False, **kwargs):
    """Return median wall-clock seconds for fn(*args, **kwargs)."""
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        if sync_cupy:
            cp.cuda.Stream.null.synchronize()
        timings.append(time.perf_counter() - start)
        del result
    return statistics.median(timings)


def _set_numpy_backend():
    tools.xp = np
    tools.xndimage = scipy_ndimage


def _set_cupy_backend():
    tools.xp = cp
    tools.xndimage = cupy_ndimage


def _build_volume(size, rng):
    return rng.random((size, size, size)).astype(np.float32)


def _build_grad(size, rng):
    return rng.standard_normal((3, size, size, size)).astype(np.float32) * 0.3


def _try_numpy(fn, args, repeats):
    try:
        _set_numpy_backend()
        return _time_call(fn, *args, repeats=repeats)
    except Exception as exc:  # noqa: BLE001
        return f"error: {type(exc).__name__}"


def _try_cupy(fn, args, repeats):
    if not _CUPY_AVAILABLE:
        return None
    try:
        _set_cupy_backend()
        cp_args = tuple(cp.asarray(a) if isinstance(a, np.ndarray) else a for a in args)
        return _time_call(fn, *cp_args, repeats=repeats, sync_cupy=True)
    except Exception as exc:  # noqa: BLE001
        return f"error: {type(exc).__name__}"


def bench_normalize(size, repeats, rng):
    vol = _build_volume(size, rng)
    return _try_numpy(tools.normalize, (vol.copy(),), repeats), _try_cupy(tools.normalize, (vol.copy(),), repeats)


def bench_nms_3d(size, repeats, rng):
    magnitude = _build_volume(size, rng)
    grad = _build_grad(size, rng)
    return (
        _try_numpy(tools.nms_3d, (magnitude, grad, np.float32), repeats),
        _try_cupy(tools.nms_3d, (magnitude, grad, np.float32), repeats),
    )


def bench_hessian(size, repeats, rng):
    vol = _build_volume(size, rng)
    return _try_numpy(tools.hessian, (vol,), repeats), _try_cupy(tools.hessian, (vol,), repeats)


def bench_detect_ridges(size, repeats, rng):
    vol = _build_volume(size, rng)
    return _try_numpy(tools.detect_ridges, (vol,), repeats), _try_cupy(tools.detect_ridges, (vol,), repeats)


_BENCHES = [
    ("normalize", bench_normalize),
    ("nms_3d", bench_nms_3d),
    ("hessian", bench_hessian),
    ("detect_ridges", bench_detect_ridges),
]


def _format_cell(value):
    if isinstance(value, str):
        return value
    return f"{value * 1000:.1f}"


def _format_row(name, size, np_time, cp_time):
    np_str = _format_cell(np_time)
    cp_str = _format_cell(cp_time) if cp_time is not None else "n/a"
    if isinstance(np_time, (int, float)) and isinstance(cp_time, (int, float)) and cp_time > 0:
        speedup = f"{np_time / cp_time:.1f}x"
    else:
        speedup = "n/a"
    return f"| {name} | {size}³ | {np_str} | {cp_str} | {speedup} |"


def _safe_run(fn, *args, **kwargs):
    """Run fn(...) and return its result, or the string 'error: <reason>' on failure."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - intentional broad catch for bench resilience
        return f"error: {type(exc).__name__}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    if not _CUPY_AVAILABLE:
        print("# Note: cupy is not installed; CuPy columns will be reported as n/a.", file=sys.stderr)

    print("| function | volume | NumPy (ms) | CuPy (ms) | speedup |")
    print("| --- | --- | --- | --- | --- |")
    for size in args.sizes:
        for name, fn in _BENCHES:
            result = _safe_run(fn, size, args.repeats, rng)
            if isinstance(result, tuple):
                np_time, cp_time = result
            else:
                np_time, cp_time = result, result
            print(_format_row(name, size, np_time, cp_time), flush=True)


if __name__ == "__main__":
    main()
