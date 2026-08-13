#!/usr/bin/env python3
"""Analyze per-channel statistics of a preprocessed zarr volume.

Usage:
    python analyze_zarr_channels.py <path-to-zarr>

Reads the zarr array (expected shape: C×Z×Y×X, uint8), prints per-channel
min, max, mean, median (from histogram), nonzero%, and a small histogram
sketch.  Single pass over Z, parallelized with processes, progress + ETA.

If the zarr has crop_xyzwhd + scaledown in preprocess_params, only the crop
region is scanned (the rest is zero padding).
"""

import sys
import time
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import zarr


def hist_sparkline(hist, bins=16):
    """Compact histogram visualization with 16 bins."""
    rebinned = hist.reshape(bins, 256 // bins).sum(axis=1).astype(float)
    peak = rebinned.max()
    if peak == 0:
        return "." * bins
    bars = " ▁▂▃▄▅▆▇█"
    scaled = (rebinned / peak * (len(bars) - 1)).astype(int)
    return "".join(bars[v] for v in scaled)


def _process_slab(zarr_path, n_ch, z0, z1, y0, y1, x0, x1):
    """Read one Z-slab (cropped in Y/X) and return partial stats per channel."""
    arr = zarr.open(str(zarr_path), mode="r")
    if not hasattr(arr, "shape"):
        arr = arr[list(arr.keys())[0]]
    data = np.asarray(arr[:, z0:z1, y0:y1, x0:x1])
    nvox = data.shape[1] * data.shape[2] * data.shape[3]
    hists = np.zeros((n_ch, 256), dtype=np.int64)
    vmins = np.empty(n_ch, dtype=np.int32)
    vmaxs = np.empty(n_ch, dtype=np.int32)
    sums = np.zeros(n_ch, dtype=np.int64)
    for ch in range(n_ch):
        ch_data = data[ch]
        hists[ch] = np.bincount(ch_data.ravel(), minlength=256).astype(np.int64)
        vmins[ch] = int(ch_data.min())
        vmaxs[ch] = int(ch_data.max())
        sums[ch] = ch_data.astype(np.int64).sum()
    return hists, vmins, vmaxs, sums, nvox


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <zarr-path>", file=sys.stderr)
        sys.exit(1)

    path = sys.argv[1].rstrip("/")
    n_workers = os.cpu_count()

    a = zarr.open(str(path), mode="r")

    if hasattr(a, "shape"):
        arr = a
    else:
        keys = list(a.keys())
        if not keys:
            print(f"Empty zarr group: {path}", file=sys.stderr)
            sys.exit(1)
        arr = a[keys[0]]
        print(f"Opened group, using array '{keys[0]}'")

    shape = tuple(int(v) for v in arr.shape)
    dtype = arr.dtype
    chunks = tuple(int(v) for v in arr.chunks) if hasattr(arr, "chunks") else "?"

    params = {}
    attrs_src = a if hasattr(a, "attrs") else arr
    if hasattr(attrs_src, "attrs"):
        params = dict(attrs_src.attrs.get("preprocess_params", {}))

    channel_names = params.get("channels", None)
    print(f"Path:     {path}")
    print(f"Shape:    {shape}  dtype={dtype}  chunks={chunks}")
    if params:
        for k, v in params.items():
            if k != "channels":
                print(f"  {k}: {v}")

    if len(shape) != 4:
        print(f"Expected 4D (C,Z,Y,X), got {len(shape)}D — aborting", file=sys.stderr)
        sys.exit(1)

    n_ch, Z, Y, X = shape
    if channel_names and len(channel_names) != n_ch:
        print(f"WARNING: {len(channel_names)} channel names but {n_ch} channels in array")
        channel_names = None

    # Determine scan region (crop or full)
    crop_param = params.get("crop_xyzwhd", None)
    scaledown = int(params.get("scaledown", 1))
    output_full = params.get("output_full_scaled", False)

    if crop_param is not None and output_full and scaledown >= 1:
        cx, cy, cz, cw, ch, cd = (int(v) for v in crop_param)
        sz0 = cz // scaledown
        sy0 = cy // scaledown
        sx0 = cx // scaledown
        sz1 = min(Z, sz0 + max(1, cd // scaledown))
        sy1 = min(Y, sy0 + max(1, ch // scaledown))
        sx1 = min(X, sx0 + max(1, cw // scaledown))
        print(f"Crop:     z=[{sz0}:{sz1}) y=[{sy0}:{sy1}) x=[{sx0}:{sx1})  (from crop_xyzwhd, sd={scaledown})")
    else:
        sz0, sz1 = 0, Z
        sy0, sy1 = 0, Y
        sx0, sx1 = 0, X

    # Align chunk_z to zarr's Z chunk size for efficient reads
    zarr_chunk_z = chunks[1] if isinstance(chunks, tuple) and len(chunks) > 1 else 64
    chunk_z = max(1, zarr_chunk_z)

    # Build slab ranges over the crop Z range
    slabs = [(z0, min(z0 + chunk_z, sz1)) for z0 in range(sz0, sz1, chunk_z)]
    n_slabs = len(slabs)

    # Global accumulators
    hists = np.zeros((n_ch, 256), dtype=np.int64)
    vmins = np.full(n_ch, 255, dtype=np.int32)
    vmaxs = np.zeros(n_ch, dtype=np.int32)
    sums = np.zeros(n_ch, dtype=np.int64)
    total = 0

    done_count = 0
    t0 = time.time()
    print(f"\nScanning {n_slabs} Z-slabs (chunk_z={chunk_z}, workers={n_workers}) ...", flush=True)

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_process_slab, path, n_ch, z0, z1, sy0, sy1, sx0, sx1): i
            for i, (z0, z1) in enumerate(slabs)
        }
        for fut in as_completed(futures):
            s_hists, s_vmins, s_vmaxs, s_sums, s_nvox = fut.result()
            hists += s_hists
            np.minimum(vmins, s_vmins, out=vmins)
            np.maximum(vmaxs, s_vmaxs, out=vmaxs)
            sums += s_sums
            total += s_nvox
            done_count += 1
            elapsed = time.time() - t0
            progress = done_count / n_slabs
            eta = elapsed / progress * (1 - progress) if progress > 0 else 0
            print(
                f"\r  slab {done_count}/{n_slabs}  "
                f"{100 * progress:.0f}%  "
                f"elapsed {int(elapsed)}s  "
                f"ETA {int(eta)}s",
                end="", flush=True,
            )

    elapsed = time.time() - t0
    throughput = total * n_ch / max(elapsed, 1e-6) / 1e6
    print(f"\r  done in {elapsed:.1f}s  ({throughput:.0f} Mvox/s)                    ")

    # Print results
    print()
    hdr = f"{'Ch':>3}  {'Name':<12} {'Min':>4} {'Max':>4} {'Mean':>7} {'Median':>6} {'NZ%':>7}  Histogram"
    print(hdr)
    print("-" * len(hdr) + "----------------")

    for ch in range(n_ch):
        name = channel_names[ch] if channel_names else f"ch{ch}"
        mean = sums[ch] / max(total, 1)
        cumsum = np.cumsum(hists[ch])
        median = int(np.searchsorted(cumsum, total / 2))
        nonzero_pct = 100.0 * (1.0 - hists[ch, 0] / max(total, 1))
        spark = hist_sparkline(hists[ch])
        print(
            f"{ch:3d}  {name:<12} {vmins[ch]:4d} {vmaxs[ch]:4d} "
            f"{mean:7.2f} {median:6d} {nonzero_pct:6.1f}%  {spark}"
        )

    print(f"\nTotal voxels per channel: {total:,}")


if __name__ == "__main__":
    main()
