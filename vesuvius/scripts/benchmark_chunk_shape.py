#!/usr/bin/env python3
"""Show how a surface volume's chunk shape sets the cost of flat inference reads.

Flat inference reads one patch of ``input_depth x patch x patch`` from the surface
volume for every scheduled window.  A zarr read decompresses every chunk it touches,
so a chunk much larger than the patch makes each read pay for a whole chunk -- the
inference then runs at ~0 % GPU utilisation because it is waiting on CPU.  This is
the mechanism behind the measured 3.2x wall-time difference between the same render
chunked ``(34, 768, 4096)`` and ``(34, 64, 256)``.

This script measures the effect directly and cheaply (CPU only, no checkpoint): the
same voxels are written to two zarrs with different chunk shapes, then the same
number of random inference-shaped patch reads are timed from each.

Usage:

    python benchmark_chunk_shape.py /path/to/render.zarr
    python benchmark_chunk_shape.py --patch 128 --depth 17 --reads 400 render.zarr
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
import time

import numpy as np
import zarr

# Chunk shapes compared: a single monolithic chunk (the pathological case) against
# the patch-aligned tiling the surface renderer should write.
DEFAULT_CHUNKS = [(None, None, None), (None, 64, 256), (None, 128, 128)]


def _load(path: str) -> np.ndarray:
    try:
        array = zarr.open_array(path, mode="r")
    except Exception:
        array = zarr.open_group(path, mode="r")["0"]
    if array.ndim == 4:
        array = array[0]
    return np.asarray(array)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="surface-render zarr (uint8) to sample")
    parser.add_argument("--patch", type=int, default=128)
    parser.add_argument("--depth", type=int, default=17)
    parser.add_argument("--reads", type=int, default=300, help="patch reads per chunk shape")
    parser.add_argument("--max-rows", type=int, default=768, help="crop the working volume to this many rows")
    parser.add_argument("--max-cols", type=int, default=4096, help="crop the working volume to this many columns")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    volume = _load(args.path)
    # Bound the working volume: a monolithic chunk of a full render is >1 GB and reading
    # hundreds of patches from it takes minutes. The ratio is what matters, not the size.
    if volume.shape[1] > args.max_rows or volume.shape[2] > args.max_cols:
        volume = volume[:, : args.max_rows, : args.max_cols].copy()
    depth_total, height, width = volume.shape
    depth = min(args.depth, depth_total)
    if height < args.patch or width < args.patch:
        print(f"volume too small: {volume.shape}")
        return 1
    print(f"working volume {volume.shape} {volume.dtype} (cropped to {args.max_rows}x{args.max_cols} max)")

    rng = np.random.default_rng(args.seed)
    offsets = [
        (int(rng.integers(0, height - args.patch + 1)), int(rng.integers(0, width - args.patch + 1)))
        for _ in range(args.reads)
    ]

    workdir = tempfile.mkdtemp(prefix="chunk-shape-")
    try:
        results = []
        for chunks in DEFAULT_CHUNKS:
            shape = (depth_total, height, width)
            actual = tuple(c if c is not None else s for c, s in zip(chunks, shape))
            path = f"{workdir}/chunks_{'_'.join(str(c) for c in actual)}.zarr"
            group = zarr.open_group(path, mode="w")
            array = group.create_dataset("0", shape=shape, chunks=actual, dtype=volume.dtype)
            array[...] = volume

            reader = zarr.open_group(path, mode="r")["0"]
            started = time.perf_counter()
            for y0, x0 in offsets:
                reader[0:depth, y0 : y0 + args.patch, x0 : x0 + args.patch]
            elapsed = time.perf_counter() - started
            chunk_voxels = int(np.prod(actual))
            patch_voxels = depth * args.patch * args.patch
            results.append((actual, elapsed, chunk_voxels / patch_voxels))
            print(
                f"chunks={str(actual):22s} chunk/patch={chunk_voxels / patch_voxels:8.1f}x "
                f"reads={args.reads} time={elapsed:.2f}s "
                f"per_read={1000.0 * elapsed / args.reads:.2f}ms"
            )

        if len(results) >= 2:
            best = min(r[1] for r in results)
            worst = max(r[1] for r in results)
            print(f"\nmis-chunked / patch-aligned = {worst / best:.2f}x")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
