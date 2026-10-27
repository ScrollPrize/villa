#!/usr/bin/env python3
"""Check bit-identity and measure the speedup of the histogram fast path.

`normalize_robust_histogram` is only acceptable if it is *exact* for integer-valued
patches, so this tool does two things on the data the flat reader actually sees --
uint8 CT surface renders held as zarr arrays:

1. **Correctness.** For every sampled patch, compare the fast path against
   `normalize_robust` bit for bit, and count how many took the delegated
   (degenerate-MAD / non-integer) route.
2. **Speed.** Time both per patch, and report the end-to-end factor implied by the
   measured share of a screen depth spent in normalisation (the flat inference
   contract normalises one patch, once, per block).

Usage:

    python benchmark_robust_normalization.py /path/to/render.zarr [more.zarr ...]
    python benchmark_robust_normalization.py --patches 400 --repeats 3 render.zarr

With no path it synthesises render-like uint8 patches, so the tool still runs
without any scroll data.
"""
from __future__ import annotations

import argparse
import time
from typing import Iterator

import numpy as np

from vesuvius.image_proc.intensity.normalization import (
    _histogram_robust_statistics,
    _is_integer_valued_uint8,
    normalize_robust,
    normalize_robust_histogram,
)

# The patch `FlatBlockDataset` normalises: depth 17 x patch 128 x patch 128.
PATCH = 128
DEPTH = 17
# Share of a screen depth's wall time measured to be spent inside `normalize_robust`
# (project profile, journal 2026-10-15 §88).  Used only to translate the per-patch
# factor into an end-to-end estimate; the per-patch numbers stand on their own.
NORMALIZE_TIME_SHARE = 0.68


def _decode(array, depth: int) -> np.ndarray:
    if hasattr(array, "dtype") and array.dtype == np.uint8:
        return np.asarray(array)
    return np.asarray(array, dtype=np.uint8)


def _iter_patches(paths: list[str], rng: np.random.Generator, count: int) -> Iterator[np.ndarray]:
    if not paths:
        for index in range(count):
            base = rng.normal(118.0, 11.0, size=(DEPTH, PATCH, PATCH))
            base[..., : PATCH // 8] = 0.0
            base[rng.random(base.shape) < 0.01] = 255.0
            yield np.clip(np.rint(base), 0, 255).astype(np.float32)
            del index
        return

    import zarr

    for path in paths:
        try:
            array = zarr.open_array(path, mode="r")
        except Exception:
            # Render zarrs are groups whose pyramid level "0" holds the data.
            group = zarr.open_group(path, mode="r")
            try:
                array = group["0"]
            except Exception:
                print(f"skip (no level '0') {path}")
                continue
        if array.ndim == 4:
            array = array[0]
        height, width = array.shape[-2], array.shape[-1]
        if height < PATCH or width < PATCH:
            print(f"skip (too small) {path} shape={array.shape}")
            continue
        layers = min(DEPTH, array.shape[0])
        for _ in range(count):
            y0 = int(rng.integers(0, height - PATCH + 1))
            x0 = int(rng.integers(0, width - PATCH + 1))
            patch = _decode(array[:layers, y0 : y0 + PATCH, x0 : x0 + PATCH], layers)
            if patch.shape[0] < DEPTH:
                pad = np.zeros((DEPTH - patch.shape[0],) + patch.shape[1:], patch.dtype)
                patch = np.concatenate([patch, pad], axis=0)
            yield patch.astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="uint8 render zarr arrays to sample")
    parser.add_argument("--patches", type=int, default=200, help="patches per input")
    parser.add_argument("--repeats", type=int, default=3, help="timed repeats per patch")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    identical = 0
    mismatched = 0
    delegated = 0
    worst = 0.0
    total = 0
    fast_seconds = 0.0
    slow_seconds = 0.0

    for patch in _iter_patches(args.paths, rng, args.patches):
        total += 1
        reference = np.ascontiguousarray(normalize_robust(patch))
        fast = normalize_robust_histogram(patch)
        if np.array_equal(reference, fast) and reference.dtype == fast.dtype:
            identical += 1
        else:
            mismatched += 1
            worst = max(worst, float(np.abs(reference - fast).max()))
        # Report how often the fast path had to hand back to `normalize_robust`
        # (degenerate MAD), which is expected for near-flat / two-level patches.
        if _is_integer_valued_uint8(patch):
            if _histogram_robust_statistics(np.rint(patch).astype(np.int64).ravel())[4]:
                delegated += 1
        else:
            delegated += 1

        for _ in range(args.repeats):
            start = time.perf_counter()
            normalize_robust(patch)
            slow_seconds += time.perf_counter() - start
            start = time.perf_counter()
            normalize_robust_histogram(patch)
            fast_seconds += time.perf_counter() - start

    if total == 0:
        print("no patches sampled")
        return 1

    slow_ms = 1000.0 * slow_seconds / (total * args.repeats)
    fast_ms = 1000.0 * fast_seconds / (total * args.repeats)
    print(f"patches                 : {total}")
    print(f"bit-identical           : {identical}")
    print(f"mismatched              : {mismatched} (worst |diff| {worst:.3e})")
    print(f"normalize_robust        : {slow_ms:.2f} ms/patch")
    print(f"normalize_robust_hist   : {fast_ms:.2f} ms/patch")
    print(f"per-patch speedup       : {slow_ms / fast_ms:.2f}x")
    share = NORMALIZE_TIME_SHARE
    depth_factor = 1.0 / ((1.0 - share) + share * (fast_ms / slow_ms))
    print(f"implied per-depth factor: {depth_factor:.2f}x (at {share:.0%} normalisation share)")
    return 0 if mismatched == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
