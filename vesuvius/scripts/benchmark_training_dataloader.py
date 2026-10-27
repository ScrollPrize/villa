#!/usr/bin/env python3
"""Throughput of the training patch path through a real multi-worker `DataLoader`.

`benchmark_training_patch_normalization.py` times the sampler in *one* process, which can only
swap `normalize_image` by monkeypatching. A real training run does not work that way: it pulls
from a `torch.utils.data.DataLoader` whose workers are separate processes, so the normalisation
cost sits in the workers and cannot be switched in-process. This script therefore measures the
deployment shape directly and is meant to be run twice -- once against the pre-change package and
once against the branch -- with the same arguments:

    PYTHONPATH=<tree>/vesuvius/src python benchmark_training_dataloader.py <render.zarr> \
        --workers 4 --samples 200

It reports samples/s cold (the loader's first pass, which includes worker start-up) and warm
(workers kept alive across passes via `persistent_workers`), plus a sha256 over the sample stream
so the two arms can be shown to emit identical samples.

The volume is a real flattened render (`zarr` group with a `0` array of shape (depth, rows, cols),
uint8). Samples are deterministic -- fixed coordinates, no jitter, `shuffle=False` -- so the
sample hash is comparable between the two packages.
"""
from __future__ import annotations

import argparse
import hashlib
import random
import sys
import time

from vesuvius.ink_detection.config import NormalizationConfig
from vesuvius.ink_detection.data import coord_patch_dataset as cpd

PATCH = (17, 128, 128)


def _coords(shape, n):
    _, height, width = shape
    py, px = PATCH[1], PATCH[2]
    rng = random.Random(0)
    return [
        (rng.randint(px, width - px), rng.randint(py, height - py), (shape[0] - 1) // 2)
        for _ in range(n)
    ]


def _dataset(volume, coords):
    return cpd.CoordPatchDataset(
        volume_path=volume,
        resolution=0,
        coords_xyz=coords,
        jitter=0,
        length=len(coords),
        patch_size=PATCH,
        normalization=NormalizationConfig(),
        input_mask_threshold=0.0,
    )


class _IndexSeeded:
    """Make the sampler's coordinate draw a function of the sample index.

    `CoordPatchDataset.__getitem__` takes a coordinate from the global `random` module, so with
    several loader workers the patch each index reads depends on which worker happened to get it,
    and the sample stream stops being comparable between runs. Seeding the module per index makes
    the read a deterministic function of the index, so the hash is a valid equality check between
    the two packages at any worker count, while the coords stay diverse.
    """

    def __init__(self, dataset, seed):
        self.dataset = dataset
        self.seed = seed

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        random.seed(f"{self.seed}:{index}")
        return self.dataset[index]


def _pass(loader):
    """Iterate one full pass; return (seconds, sha256 over every sample's image bytes)."""
    digest = hashlib.sha256()
    seen = 0
    start = time.perf_counter()
    for batch in loader:
        image = batch["image"]
        digest.update(image.numpy().view("uint8").tobytes())
        seen += int(image.shape[0])
    return time.perf_counter() - start, digest.hexdigest(), seen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("volume", help="a real render zarr (group with a `0` array)")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--prefetch", type=int, default=2)
    parser.add_argument("--label", default="", help="printed with every line, e.g. the arm name")
    parser.add_argument("--seed", type=int, default=0,
                        help="seeds the sampler's coordinate draw, so the two arms read the same "
                             "patches; the dataset picks a coordinate with the `random` module")
    args = parser.parse_args()

    import torch
    import zarr

    root = zarr.open(args.volume, mode="r")
    shape = root.shape if hasattr(root, "shape") else root["0"].shape
    dataset = _IndexSeeded(_dataset(args.volume, _coords(shape, args.samples)), args.seed)
    print(f"volume {shape} uint8; patch {PATCH}; {args.samples} samples; "
          f"workers={args.workers} prefetch={args.prefetch} torch={torch.__version__}")

    def loader():
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            prefetch_factor=args.prefetch if args.workers else None,
            persistent_workers=bool(args.workers),
        )

    # `_IndexSeeded` makes each index's coordinate draw deterministic, so the sample hash below is
    # comparable between the two packages (and between worker counts).
    cold = loader()
    cold_s, cold_hash, cold_n = _pass(cold)
    warm_times = []
    warm_hashes = []
    for _ in range(args.repeats):
        seconds, warm_hash, n = _pass(cold)  # same loader: persistent workers are reused
        warm_times.append(seconds)
        warm_hashes.append(warm_hash)
    del cold
    warm = sorted(warm_times)[len(warm_times) // 2]

    tag = f"{args.label}: " if args.label else ""
    print(f"{tag}cold  {cold_s:7.3f} s  {cold_n / cold_s:7.2f} samples/s  (includes worker start-up)")
    print(f"{tag}warm  {warm:7.3f} s  {args.samples / warm:7.2f} samples/s  "
          f"(median of {args.repeats}, workers reused)")
    # Only the cold pass is comparable across arms: with persistent workers the sampler's RNG
    # state carries on, so later passes legitimately read a different patch order.
    print(f"{tag}cold sample sha256 {cold_hash}")
    print(f"{tag}warm sample sha256 {warm_hashes[-1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
