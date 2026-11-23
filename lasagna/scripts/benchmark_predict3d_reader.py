#!/usr/bin/env python3
"""Measure predict3d bbox read windows through a vc_reader array."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import statistics
import sys
import time

import numpy as np
from vc_reader.reader import open_zarr

LASAGNA_ROOT = Path(__file__).resolve().parents[1]
if str(LASAGNA_ROOT) not in sys.path:
	sys.path.insert(0, str(LASAGNA_ROOT))

from tiled_predict3d import _read_tile_zarr


def _coords(origin: tuple[int, int, int], count: int, stride: int):
	z0, y0, x0 = origin
	width = max(1, int(np.ceil(np.sqrt(count))))
	return tuple(
		(z0, y0 + (index // width) * stride, x0 + (index % width) * stride)
		for index in range(count)
	)


def _summary(values: list[float]) -> str:
	ordered = sorted(values)
	p95 = ordered[min(len(ordered) - 1, int(np.ceil(0.95 * len(ordered))) - 1)]
	return (
		f"mean={statistics.mean(values):.3f}s p50={statistics.median(values):.3f}s "
		f"p95={p95:.3f}s min={min(values):.3f}s max={max(values):.3f}s"
	)


def main(argv: list[str] | None = None) -> int:
	p = argparse.ArgumentParser(description=__doc__)
	p.add_argument("input")
	p.add_argument("--origin-zyx", type=int, nargs=3, required=True)
	p.add_argument("--tile-size", type=int, default=256)
	p.add_argument("--stride", type=int, default=160)
	p.add_argument("--tiles", type=int, default=32)
	p.add_argument("--warmups", type=int, default=1)
	p.add_argument("--iterations", type=int, default=3)
	p.add_argument("--window", type=int, default=8, help="Tile reads at once")
	p.add_argument("--cache-gib", type=float, default=4.0)
	p.add_argument("--threads", type=int, default=16)
	args = p.parse_args(argv)
	arr = open_zarr(args.input, memory_cache=int(args.cache_gib * (1 << 30)), threads=args.threads)
	shape = tuple(int(v) for v in arr.shape)
	coords = _coords(tuple(args.origin_zyx), int(args.tiles), int(args.stride))

	def run() -> int:
		with ThreadPoolExecutor(max_workers=args.window) as pool:
			arrays = tuple(pool.map(
				lambda coord: _read_tile_zarr(arr, shape, (0, 0, 0), *coord, args.tile_size, 0),
				coords,
			))
		return sum(value.nbytes for value in arrays)

	print(
		f"input={args.input} shape={shape} chunks={arr.chunks} "
		f"dtype={arr.dtype} tile={args.tile_size} stride={args.stride} tiles={args.tiles} "
		f"warmups={args.warmups} iterations={args.iterations}", flush=True,
	)
	timings = []
	byte_count = 0
	for iteration in range(args.warmups + args.iterations):
		started = time.perf_counter()
		byte_count = run()
		elapsed = time.perf_counter() - started
		if iteration >= args.warmups:
			timings.append(elapsed)
	gib = byte_count / float(1 << 30)
	print(f"{_summary(timings)} throughput_mean={gib/statistics.mean(timings):.2f}GiB/s", flush=True)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
