#!/usr/bin/env python3
"""Throughput of the *training* patch path with and without the normalisation fast path.

`normalize_image` is called once per sample by ink training through
`ink_detection.data.coord_patch_dataset.CoordPatchDataset` (the self-distillation
coordinate sampler) and `ink_detection.data.dataset.InkDataset`.  This script drives the real
`CoordPatchDataset` over a real render volume and times three arms of the *same* read path:

  * `none`        -- normalisation disabled, to show its share of the sample;
  * `robust_mad`  -- the pre-change `normalize_image` (a direct `normalize_robust` call), i.e. the
                     code this branch replaces;
  * `fast`        -- `normalize_image` as this branch ships it (histogram fast path).

It also asserts `robust_mad` and `fast` produce the same tensor bit for bit.

    PYTHONPATH=vesuvius/src python vesuvius/scripts/benchmark_training_patch_normalization.py \
        /path/to/render.zarr [n_samples]
"""
from __future__ import annotations

import argparse
import random
import sys
import time

import numpy as np

from vesuvius.image_proc.intensity.normalization import normalize_robust
from vesuvius.ink_detection.config import NormalizationConfig
from vesuvius.ink_detection.data import coord_patch_dataset as cpd
from vesuvius.ink_detection.data.normalization import normalize_image as fast_normalize_image

PATCH = (17, 128, 128)


def _old_normalize_image(image, config):
    """The `normalize_image` this branch replaced (its pre-change `robust_mad` branch)."""
    if config.mode == "robust_mad":
        return normalize_robust(
            image,
            percentile_lower=config.percentile_lower,
            percentile_upper=config.percentile_upper,
        )
    return fast_normalize_image(image, config)


def _noop_normalize_image(image, config):
    return image


def _dataset(volume, coords, jitter, length):
    return cpd.CoordPatchDataset(
        volume_path=volume,
        resolution=0,
        coords_xyz=coords,
        jitter=jitter,
        length=length,
        patch_size=PATCH,
        normalization=NormalizationConfig(),
        input_mask_threshold=0.0,
    )


def _run_arm(dataset, normalize_impl, n):
    cpd.normalize_image = normalize_impl
    start = time.perf_counter()
    for index in range(n):
        dataset[index]
    return time.perf_counter() - start


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("volume", help="a real render zarr (a group is opened at level 0)")
    parser.add_argument("samples", nargs="?", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    import zarr

    root = zarr.open(args.volume, mode="r")
    shape = root.shape if hasattr(root, "shape") else root["0"].shape
    _, height, width = shape
    py, px = PATCH[1], PATCH[2]
    rng = random.Random(0)
    coords = [
        (rng.randint(px, width - px), rng.randint(py, height - py), (shape[0] - 1) // 2)
        for _ in range(max(8, args.samples))
    ]

    # bit-identity: deterministic sampler (no jitter, one coord) -> identical read each sample
    random.seed(0)
    frozen = [(width // 2, height // 2, (shape[0] - 1) // 2)]
    ds_new = _dataset(args.volume, frozen, 0, 1)
    ds_old = _dataset(args.volume, frozen, 0, 1)
    cpd.normalize_image = fast_normalize_image
    new_sample = ds_new[0]["image"]
    cpd.normalize_image = _old_normalize_image
    old_sample = ds_old[0]["image"]
    identical = new_sample.shape == old_sample.shape and np.array_equal(
        new_sample.numpy().view(np.uint8), old_sample.numpy().view(np.uint8)
    )
    print(f"volume {shape}; patch {PATCH}; {args.samples} samples/arm x {args.repeats} repeats")
    print(f"bit-identical training sample (robust_mad vs fast): {identical}")

    arms = [
        ("none (normalisation disabled)", _noop_normalize_image),
        ("robust_mad (pre-change)", _old_normalize_image),
        ("fast (this branch)", fast_normalize_image),
    ]
    datasets = {name: _dataset(args.volume, coords, 32, args.samples) for name, _ in arms}
    # warm-up, discarded
    _run_arm(datasets[arms[0][0]], arms[0][1], max(5, args.samples // 4))
    times = {name: [] for name, _ in arms}
    for _ in range(args.repeats):
        for name, impl in arms:  # arms interleaved within each repeat
            times[name].append(_run_arm(datasets[name], impl, args.samples))

    per_sample = {}
    for name, _ in arms:
        median = sorted(times[name])[len(times[name]) // 2]
        per_sample[name] = median / args.samples * 1e3
        print(
            f"{name:32s} {median:7.3f} s  {per_sample[name]:6.3f} ms/sample  "
            f"{args.samples / median:7.1f} samples/s"
        )

    slow = per_sample["robust_mad (pre-change)"]
    fast = per_sample["fast (this branch)"]
    none = per_sample["none (normalisation disabled)"]
    cpd.normalize_image = fast_normalize_image  # leave the module as imported
    print(f"\nnormalisation share of the sample (pre-change): {(slow - none) / slow * 100:.0f} %")
    print(f"per-sample speedup of the training path: {slow / fast:.2f}x")
    return 0 if identical else 1


if __name__ == "__main__":
    sys.exit(main())
