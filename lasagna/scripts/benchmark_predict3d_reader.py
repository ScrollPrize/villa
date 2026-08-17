#!/usr/bin/env python3
"""Compare predict3d Python-Zarr and TensorStore bbox read windows."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import zarr

LASAGNA_ROOT = Path(__file__).resolve().parents[1]
if str(LASAGNA_ROOT) not in sys.path:
	sys.path.insert(0, str(LASAGNA_ROOT))

from tiled_predict3d import _TensorStoreTileReader, _read_tile_zarr


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
	p.add_argument("--zarr-threads", type=int, default=16)
	p.add_argument("--cache-gib", type=float, default=4.0)
	p.add_argument("--io-threads", type=int, default=16)
	p.add_argument("--copy-threads", type=int, default=4)
	args = p.parse_args(argv)
	arr = zarr.open(args.input, mode="r")
	shape = tuple(int(v) for v in arr.shape)
	coords = _coords(tuple(args.origin_zyx), int(args.tiles), int(args.stride))
	ts_reader = _TensorStoreTileReader(
		args.input, cache_bytes=int(args.cache_gib * (1 << 30)),
		file_io_threads=args.io_threads, data_copy_threads=args.copy_threads,
	)

	def run_zarr() -> int:
		with ThreadPoolExecutor(max_workers=args.zarr_threads) as pool:
			arrays = tuple(pool.map(
				lambda coord: _read_tile_zarr(arr, shape, (0, 0, 0), *coord, args.tile_size, 0),
				coords,
			))
		return sum(value.nbytes for value in arrays)

	def run_tensorstore() -> int:
		tasks = tuple(ts_reader.submit(
			volume_shape=shape, crop_offset=(0, 0, 0), coord=coord,
			tile_size=args.tile_size, border=0,
		) for coord in coords)
		arrays = tuple(ts_reader.result(task)[0] for task in tasks)
		return sum(value.nbytes for value in arrays)

	print(
		f"input={args.input} shape={shape} chunks={tuple(int(v) for v in arr.chunks)} "
		f"dtype={arr.dtype} tile={args.tile_size} stride={args.stride} tiles={args.tiles} "
		f"warmups={args.warmups} iterations={args.iterations}", flush=True,
	)
	for name, runner in (("python-zarr", run_zarr), ("tensorstore", run_tensorstore)):
		timings = []
		byte_count = 0
		for iteration in range(args.warmups + args.iterations):
			started = time.perf_counter()
			byte_count = runner()
			elapsed = time.perf_counter() - started
			if iteration >= args.warmups:
				timings.append(elapsed)
		gib = byte_count / float(1 << 30)
		print(
			f"{name}: {_summary(timings)} throughput_mean={gib/statistics.mean(timings):.2f}GiB/s",
			flush=True,
		)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
