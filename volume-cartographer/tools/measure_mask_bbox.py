#!/usr/bin/env python3
"""Measure the non-zero-voxel bounding box of a masked zarr v2 level.

Walks every on-disk chunk, finds the tightest (z, y, x) bounding box of
non-zero voxels. Reports the result in both the scanned level's resolution
and the native (level 0) resolution, given a fixed downscale factor.

The "padding" numbers tell you how much of each axis is pure mask (can be
skipped when fetching from S3 at level 0).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "level_dir",
        nargs="?",
        default=str(Path.home() / ".VC3D/remote_cache/2.4um_PHerc-Paris4_masked.zarr/5"),
        help="Path to the level directory (containing .zarray + chunks).",
    )
    parser.add_argument(
        "--downscale",
        type=int,
        default=32,
        help="Downscale factor of this level relative to native (default 32).",
    )
    args = parser.parse_args()

    level_dir = Path(args.level_dir)
    zarray_path = level_dir / ".zarray"
    if not zarray_path.exists():
        print(f"error: {zarray_path} not found", file=sys.stderr)
        return 1

    with zarray_path.open() as f:
        meta = json.load(f)

    shape = tuple(meta["shape"])  # (Z, Y, X)
    chunks = tuple(meta["chunks"])  # (CZ, CY, CX)
    dtype = np.dtype(meta["dtype"])
    fill = meta.get("fill_value", 0)
    sep = meta.get("dimension_separator", ".")
    if meta.get("compressor") is not None or meta.get("filters"):
        print(
            f"warning: chunks use compressor/filters — this script only handles raw uncompressed zarr",
            file=sys.stderr,
        )

    print(f"level dir:  {level_dir}")
    print(f"shape:      {shape}  (Z, Y, X)")
    print(f"chunks:     {chunks}")
    print(f"dtype:      {dtype}  fill={fill}  sep={sep!r}")
    print(f"downscale:  {args.downscale}x")
    print()

    expected_chunk_bytes = int(np.prod(chunks) * dtype.itemsize)

    # Global bbox of non-zero voxels, in this level's voxel coordinates.
    # (min is inclusive, max is inclusive; start at sentinels.)
    bbox_min = [shape[0], shape[1], shape[2]]
    bbox_max = [-1, -1, -1]

    total_chunks_present = 0
    total_chunks_with_data = 0
    nonzero_voxels = 0

    t_start = time.time()
    # Chunk grid size per axis (ceil).
    grid = tuple((s + c - 1) // c for s, c in zip(shape, chunks))
    for cz in range(grid[0]):
        for cy in range(grid[1]):
            for cx in range(grid[2]):
                path = level_dir / sep.join([str(cz), str(cy), str(cx)])
                if not path.exists():
                    continue
                total_chunks_present += 1
                raw = path.read_bytes()
                if len(raw) != expected_chunk_bytes:
                    print(
                        f"warning: chunk {cz}/{cy}/{cx} has {len(raw)} bytes, "
                        f"expected {expected_chunk_bytes}",
                        file=sys.stderr,
                    )
                    continue
                arr = np.frombuffer(raw, dtype=dtype).reshape(chunks)

                # Effective voxel extent within this chunk (may be partial at edges).
                z0, y0, x0 = cz * chunks[0], cy * chunks[1], cx * chunks[2]
                z1 = min(shape[0], z0 + chunks[0])
                y1 = min(shape[1], y0 + chunks[1])
                x1 = min(shape[2], x0 + chunks[2])
                arr = arr[: z1 - z0, : y1 - y0, : x1 - x0]

                nz_mask = arr != fill
                if not nz_mask.any():
                    continue
                total_chunks_with_data += 1

                # Per-axis any-reduce gives the local nonzero coord range.
                z_any = nz_mask.any(axis=(1, 2))
                y_any = nz_mask.any(axis=(0, 2))
                x_any = nz_mask.any(axis=(0, 1))
                lz0 = int(np.argmax(z_any))
                lz1 = int(len(z_any) - 1 - np.argmax(z_any[::-1]))
                ly0 = int(np.argmax(y_any))
                ly1 = int(len(y_any) - 1 - np.argmax(y_any[::-1]))
                lx0 = int(np.argmax(x_any))
                lx1 = int(len(x_any) - 1 - np.argmax(x_any[::-1]))

                bbox_min[0] = min(bbox_min[0], z0 + lz0)
                bbox_min[1] = min(bbox_min[1], y0 + ly0)
                bbox_min[2] = min(bbox_min[2], x0 + lx0)
                bbox_max[0] = max(bbox_max[0], z0 + lz1)
                bbox_max[1] = max(bbox_max[1], y0 + ly1)
                bbox_max[2] = max(bbox_max[2], x0 + lx1)
                nonzero_voxels += int(nz_mask.sum())

    t_end = time.time()
    print(f"scanned in {t_end - t_start:.2f} s")
    print(f"chunks present:   {total_chunks_present}")
    print(f"chunks with data: {total_chunks_with_data}")
    print(f"non-zero voxels:  {nonzero_voxels:,}")
    print()

    if bbox_max[0] < 0:
        print("No non-zero voxels found.")
        return 0

    # Size including the max voxel.
    ds = args.downscale
    print(f"Bounding box at this level (voxels):")
    for axis, name in enumerate("ZYX"):
        lo, hi = bbox_min[axis], bbox_max[axis]
        extent = hi - lo + 1
        total = shape[axis]
        pad_lo = lo
        pad_hi = total - 1 - hi
        print(f"  {name}: [{lo:5d}, {hi:5d}] extent={extent:5d}   "
              f"pad_lo={pad_lo:4d}  pad_hi={pad_hi:4d}  "
              f"({100.0 * extent / total:5.1f}% of {total})")

    print()
    print(f"Bounding box at native resolution (level 0, {ds}x):")
    for axis, name in enumerate("ZYX"):
        lo = bbox_min[axis] * ds
        hi = (bbox_max[axis] + 1) * ds - 1  # inclusive in L0
        extent = hi - lo + 1
        total = shape[axis] * ds
        pad_lo = lo
        pad_hi = total - 1 - hi
        print(f"  {name}: [{lo:7d}, {hi:7d}] extent={extent:7d}  "
              f"pad_lo={pad_lo:6d}  pad_hi={pad_hi:6d}")

    bbox_voxels = np.prod([bbox_max[a] - bbox_min[a] + 1 for a in range(3)])
    total_voxels = np.prod(shape)
    print()
    print(f"Volume: {bbox_voxels:,} voxels in bbox / {total_voxels:,} total")
    print(f"        {100.0 * bbox_voxels / total_voxels:.1f}% of full volume in bbox")
    print(f"        {100.0 * nonzero_voxels / bbox_voxels:.1f}% of bbox voxels are non-zero (scroll fill)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
