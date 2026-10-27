#!/usr/bin/env python3
"""Bit-identity and speed of `normalize_image` after routing robust_mad through the fast path.

`normalize_image` is the per-crop normaliser of ink training (`ink_detection/data/dataset.py`,
`coord_patch_dataset.py`) and native 3D inference (`infer_full3d_tifxyz.py`); all three hand it a
uint8 CT crop widened to float32 with the default `NormalizationConfig` (robust_mad, 1.0/99.0).
This script cuts crops out of a real render volume, checks the new path against the
`normalize_robust` call it replaced element for element, and times both.

    python vesuvius/scripts/benchmark_normalize_image.py /path/to/render.zarr [n_crops]
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import zarr

from vesuvius.image_proc.intensity.normalization import normalize_robust
from vesuvius.ink_detection.config import NormalizationConfig
from vesuvius.ink_detection.data.normalization import normalize_image

PATCH = (17, 128, 128)


def _crops(volume, n, seed=0):
    # One bounded band, cut into crops: never materialise a whole volume.
    band = np.asarray(volume[0:PATCH[0], 0:min(768, volume.shape[1]), :])
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        y = int(rng.integers(0, band.shape[1] - PATCH[1] + 1))
        x = int(rng.integers(0, band.shape[2] - PATCH[2] + 1))
        out.append(np.ascontiguousarray(band[:, y:y + PATCH[1], x:x + PATCH[2]]))
    return band, out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("volume", help="a uint8 render zarr array (or a group, uses level 0)")
    parser.add_argument("crops", nargs="?", type=int, default=400)
    args = parser.parse_args()

    volume = zarr.open(args.volume, mode="r")
    if not hasattr(volume, "shape"):
        volume = volume["0"]
    band, crops = _crops(volume, args.crops)
    config = NormalizationConfig()
    print(f"volume {volume.shape} -> band {band.shape}; {len(crops)} crops of {PATCH} {band.dtype}")

    def old(crop):
        return normalize_robust(
            crop.astype(np.float32, copy=False),
            percentile_lower=config.percentile_lower,
            percentile_upper=config.percentile_upper,
        )

    mismatch = sum(0 if np.array_equal(normalize_image(c, config), old(c)) else 1 for c in crops)
    print(f"bit-identical to normalize_robust: {len(crops) - mismatch}/{len(crops)}")

    def run(fn):
        for crop in crops:
            fn(crop)

    run(old)
    run(lambda c: normalize_image(c, config))  # warm-up, discarded
    times = {"old": [], "new": []}
    for _ in range(3):
        t = time.perf_counter(); run(old); times["old"].append(time.perf_counter() - t)
        t = time.perf_counter(); run(lambda c: normalize_image(c, config))
        times["new"].append(time.perf_counter() - t)
    med = {k: sorted(v)[len(v) // 2] for k, v in times.items()}
    for name, label in (("old", "normalize_robust (as called before)"),
                        ("new", "normalize_image  (fast path)")):
        print(f"{label:32s} {med[name]:7.3f} s  {med[name] / len(crops) * 1e3:6.3f} ms/crop")
    print(f"per-crop speedup: {med['old'] / med['new']:.2f}x")
    return 0 if mismatch == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
