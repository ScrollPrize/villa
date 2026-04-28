#!/usr/bin/env python3
"""Rechunk an OME-Zarr to a different chunk size (default 128).

Uses multiprocessing to copy 3D blocks aligned to both input and output
chunk grids, keeping memory per worker to a few MB.
"""
from __future__ import annotations

import argparse
import math
import multiprocessing
import shutil
import time
import zarr
import zarr.storage
from pathlib import Path


def _copy_block(args):
    src_path, dst_path, z0, z1, y0, y1, x0, x1 = args
    z_in = zarr.open(src_path, mode="r", zarr_format=2)
    z_out = zarr.open(zarr.storage.LocalStore(dst_path), mode="r+", zarr_format=2)
    z_out[z0:z1, y0:y1, x0:x1] = z_in[z0:z1, y0:y1, x0:x1]
    return z0, z1, y0, y1, x0, x1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("-c", "--chunk-size", type=int, default=128)
    parser.add_argument("-j", "--workers", type=int, default=16,
                        help="Number of parallel worker processes (default: 16)")
    parser.add_argument("--compressor", type=str, default=None,
                        choices=["none", "lz4", "zstd", "blosc"],
                        help="Output compressor (default: none)")
    args = parser.parse_args()

    src, dst, cs = args.input, args.output, args.chunk_size

    import numcodecs
    compressor = None
    if args.compressor == "lz4":
        compressor = numcodecs.LZ4()
    elif args.compressor == "zstd":
        compressor = numcodecs.Zstd(level=1)
    elif args.compressor == "blosc":
        compressor = numcodecs.Blosc(cname="lz4", clevel=1)
    dst.mkdir(parents=True, exist_ok=True)

    for f in [".zgroup", ".zattrs"]:
        if (src / f).exists():
            shutil.copy2(src / f, dst / f)

    scales = sorted(
        int(d.name) for d in src.iterdir()
        if d.is_dir() and d.name.isdigit() and (d / ".zarray").exists()
    )

    for s in scales:
        z_in = zarr.open(str(src / str(s)), mode="r", zarr_format=2)
        shape = z_in.shape
        in_cs = z_in.chunks[0] if hasattr(z_in, "chunks") else cs
        lcm = (in_cs * cs) // math.gcd(in_cs, cs)

        # Create output array
        out_path = str(dst / str(s))
        store = zarr.storage.LocalStore(out_path)
        zarr.open(
            store, mode="w", zarr_format=2,
            shape=shape, chunks=(cs, cs, cs),
            dtype=z_in.dtype, dimension_separator="/", fill_value=0,
            compressor=compressor,
        )

        # Build block list aligned to LCM in all dimensions
        blocks = []
        for z0 in range(0, shape[0], lcm):
            for y0 in range(0, shape[1], lcm):
                for x0 in range(0, shape[2], lcm):
                    blocks.append((
                        str(src / str(s)), out_path,
                        z0, min(z0 + lcm, shape[0]),
                        y0, min(y0 + lcm, shape[1]),
                        x0, min(x0 + lcm, shape[2]),
                    ))

        mem_per_block = lcm * lcm * lcm / 1e6
        print(f"Scale {s}: {shape} {z_in.chunks} -> ({cs},)³  "
              f"{len(blocks)} blocks of {lcm}³ (~{mem_per_block:.0f} MB each), "
              f"{args.workers} workers")

        done = 0
        t0 = time.time()
        with multiprocessing.Pool(args.workers) as pool:
            for result in pool.imap_unordered(_copy_block, blocks):
                done += 1
                if done % 50 == 0 or done == len(blocks):
                    elapsed = time.time() - t0
                    pct = done / len(blocks) * 100
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (len(blocks) - done) / rate if rate > 0 else 0
                    print(f"  [{pct:5.1f}%] {done}/{len(blocks)} blocks  "
                          f"({rate:.1f} blk/s, ETA {eta:.0f}s)")

        dt = time.time() - t0
        print(f"  Scale {s} done in {dt:.1f}s")

    print("Done:", dst)


if __name__ == "__main__":
    main()
