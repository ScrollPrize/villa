#!/usr/bin/env python3
"""Rechunk an OME-Zarr and rebuild its scales."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from vc_reader.reader import ZarrWriter, create_group, open_group, open_zarr

_LASAGNA_DIR = Path(__file__).resolve().parents[1]
if str(_LASAGNA_DIR) not in sys.path:
	sys.path.insert(0, str(_LASAGNA_DIR))

from omezarr_pyramid import (
	build_normal_omezarr_pyramid,
	build_scalar_omezarr_pyramid,
	print_progress,
	shape_div2,
)


def _parse_chunk(value: str) -> tuple[int, int, int]:
	parts = [p.strip() for p in value.split(",")]
	if len(parts) == 1:
		c = int(parts[0])
		if c <= 0:
			raise argparse.ArgumentTypeError("chunk size must be positive")
		return (c, c, c)
	if len(parts) != 3:
		raise argparse.ArgumentTypeError("chunk size must be N or Z,Y,X")
	try:
		chunk = (int(parts[0]), int(parts[1]), int(parts[2]))
	except ValueError as exc:
		raise argparse.ArgumentTypeError("chunk size values must be integers") from exc
	if min(chunk) <= 0:
		raise argparse.ArgumentTypeError("chunk size values must be positive")
	return chunk


def _cache_bytes(value: str) -> int:
	text = value.strip().lower()
	scale = 1
	for suffix, mult in (("gib", 1024**3), ("gb", 1000**3), ("mib", 1024**2), ("mb", 1000**2), ("kib", 1024), ("kb", 1000)):
		if text.endswith(suffix):
			text = text[: -len(suffix)]
			scale = mult
			break
	try:
		return int(float(text) * scale)
	except ValueError as exc:
		raise argparse.ArgumentTypeError(f"invalid byte size: {value}") from exc


def infer_n_levels(root: Path) -> int:
	levels = [int(name) for name in open_group(root).array_keys() if name.isdigit()]
	if not levels:
		raise ValueError(f"no numeric OME-Zarr levels found in {root}")
	return max(levels) + 1


def _multiscales(source_attrs: dict, data_level: int, n_levels: int, name: str) -> dict:
	"""The source's multiscales entry (its axes and name), with this pyramid's levels."""
	axes = [{"name": axis, "type": "space", "unit": "pixel"} for axis in "zyx"]
	source = source_attrs.get("multiscales")
	if isinstance(source, list) and source:
		axes = source[0].get("axes", axes)
		name = str(source[0].get("name", name))
	return {
		"version": "0.4",
		"name": name,
		"axes": axes,
		"datasets": [
			{"path": str(lv), "coordinateTransformations": [{"type": "scale", "scale": [float(2 ** lv)] * 3}]}
			for lv in range(data_level, n_levels)
		],
	}


def create_omezarr_like(*, src_root: Path, dst_root: Path, data_level: int, n_levels: int,
						chunk: tuple[int, int, int], name: str, overwrite: bool) -> None:
	if dst_root.exists():
		if not overwrite:
			raise FileExistsError(f"output already exists: {dst_root}")
		shutil.rmtree(dst_root)
	dst_root.parent.mkdir(parents=True, exist_ok=True)
	group = open_group(src_root)
	source = group.array(str(data_level))
	if source.dtype != np.dtype("uint8"):
		raise ValueError(f"only uint8 OME-Zarr arrays are supported, got dtype={source.dtype}")
	create_group(dst_root, attrs={**group.attrs, "multiscales": [_multiscales(group.attrs, data_level, n_levels, name)]})
	for lv in range(data_level, n_levels):
		shape = shape_div2(source.shape, lv - data_level)
		ZarrWriter.create_like(dst_root / str(lv), source, shape=shape, chunks=[min(n, c) for n, c in zip(shape, chunk)], separator="/")


def copy_data_level(*, src_root: Path, dst_root: Path, data_level: int, workers: int, cache_bytes: int, label: str) -> None:
	"""Every chunk of the output level from the source's voxels. Empty chunks are not stored."""
	source = open_zarr(src_root / str(data_level), threads=1, memory_cache=cache_bytes)
	target = ZarrWriter.open(dst_root / str(data_level), skip_empty=True)
	starts = [
		(z, y, x)
		for z in range(0, target.shape[0], target.chunks[0])
		for y in range(0, target.shape[1], target.chunks[1])
		for x in range(0, target.shape[2], target.chunks[2])
	]

	def copy(start) -> None:
		region = tuple(slice(a, min(n, a + c)) for a, n, c in zip(start, target.shape, target.chunks))
		target[region] = source[region]

	tag = f"[rechunk {label} L{data_level}]"
	print(f"{tag} jobs={len(starts)} workers={workers} chunk={target.chunks}", flush=True)
	t0 = time.time()
	with ThreadPoolExecutor(max_workers=workers) as pool:
		for done, _ in enumerate(pool.map(copy, starts), 1):
			print_progress(prefix=tag, done=done, total=len(starts), t0=t0)
	print("", flush=True)


def rechunk_scalar(*, src_root: Path, dst_root: Path, chunk, data_level: int, n_levels: int | None, workers: int,
				   cache_bytes: int, zero_overrides: bool, overwrite: bool, label: str) -> None:
	if n_levels is None:
		n_levels = infer_n_levels(src_root)
	create_omezarr_like(src_root=src_root, dst_root=dst_root, data_level=data_level, n_levels=n_levels,
						chunk=chunk, name=label, overwrite=overwrite)
	copy_data_level(src_root=src_root, dst_root=dst_root, data_level=data_level, workers=workers, cache_bytes=cache_bytes, label=label)
	build_scalar_omezarr_pyramid(dst_root, data_level, n_levels, workers=workers, label=label, zero_overrides=zero_overrides,
								 scan_existing_source_chunks=True)


def rechunk_normal_pair(*, nx_src: Path, ny_src: Path, nx_dst: Path, ny_dst: Path, chunk, data_level: int,
						n_levels: int | None, workers: int, cache_bytes: int, overwrite: bool) -> None:
	if n_levels is None:
		n_levels = max(infer_n_levels(nx_src), infer_n_levels(ny_src))
	if open_zarr(nx_src / str(data_level)).shape != open_zarr(ny_src / str(data_level)).shape:
		raise ValueError("normal data-level shape mismatch")
	for src, dst, name in ((nx_src, nx_dst, "nx"), (ny_src, ny_dst, "ny")):
		create_omezarr_like(src_root=src, dst_root=dst, data_level=data_level, n_levels=n_levels, chunk=chunk, name=name, overwrite=overwrite)
		copy_data_level(src_root=src, dst_root=dst, data_level=data_level, workers=workers, cache_bytes=cache_bytes, label=name)
	build_normal_omezarr_pyramid(nx_dst, ny_dst, data_level, n_levels, workers=workers, scan_existing_source_chunks=True)


def _temporary_sibling(root: Path) -> Path:
	tmp = root.with_name(f"{root.name}.tmp-rechunk-{os.getpid()}")
	if tmp.exists():
		shutil.rmtree(tmp)
	return tmp


def _swap_dir(tmp: Path, target: Path) -> None:
	backup = target.with_name(f"{target.name}.bak-{os.getpid()}")
	if backup.exists():
		raise FileExistsError(f"backup path already exists: {backup}")
	if not tmp.exists():
		raise FileNotFoundError(f"temporary output does not exist: {tmp}")
	target.rename(backup)
	try:
		tmp.rename(target)
	except Exception:
		backup.rename(target)
		raise
	shutil.rmtree(backup)


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("input", type=Path, help="Input scalar OME-Zarr, or nx OME-Zarr when --normal-pair is used.")
	parser.add_argument("output", type=Path, nargs="?", help="Output OME-Zarr for out-of-place operation.")
	parser.add_argument("-c", "--chunk-size", type=_parse_chunk, default=_parse_chunk("128"), help="Output chunk size as N or Z,Y,X.")
	parser.add_argument("--in-place", action="store_true", help="Rewrite the input path through a temporary OME-Zarr and rename it into place.")
	parser.add_argument("--rebuild-scales-only", action="store_true", help="Only remove and rebuild coarser scales; do not rechunk the data level.")
	parser.add_argument("--data-level", type=int, default=0, help="Numeric level containing the source data.")
	parser.add_argument("--levels", type=int, default=None, help="Number of numeric OME-Zarr levels. Default: infer from input.")
	parser.add_argument("--zero-overrides", action="store_true", help="Use zero-overrides mean pooling for scalar data such as grad_mag.")
	parser.add_argument("--normal-pair", type=Path, default=None, help="ny OME-Zarr path; input is treated as nx and scales are rebuilt as a pair.")
	parser.add_argument("--normal-output", type=Path, default=None, help="Output ny OME-Zarr path for out-of-place paired normal processing.")
	parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing out-of-place output path.")
	parser.add_argument("-j", "--workers", type=int, default=0, help="Worker threads. Default: CPU count.")
	parser.add_argument("--cache-bytes", type=_cache_bytes, default=4 << 30, help="Memory budget for decoded source chunks while copying the data level.")
	args = parser.parse_args()

	if args.rebuild_scales_only and not args.in_place:
		raise SystemExit("--rebuild-scales-only currently operates in place; pass --in-place")
	if args.in_place and args.output is not None:
		raise SystemExit("do not pass an output path with --in-place")
	if not args.in_place and args.output is None:
		raise SystemExit("out-of-place operation requires an output path; pass --in-place for in-place operation")
	if args.normal_pair is None and args.normal_output is not None:
		raise SystemExit("--normal-output requires --normal-pair")
	if args.normal_pair is not None and not args.in_place and args.normal_output is None:
		raise SystemExit("out-of-place normal-pair operation requires --normal-output")

	workers = int(args.workers) if args.workers > 0 else max(1, multiprocessing.cpu_count())
	data_level = int(args.data_level)
	common = dict(chunk=args.chunk_size, data_level=data_level, n_levels=args.levels, workers=workers, cache_bytes=int(args.cache_bytes))

	if args.normal_pair is not None:
		if args.rebuild_scales_only:
			n_levels = args.levels or max(infer_n_levels(args.input), infer_n_levels(args.normal_pair))
			build_normal_omezarr_pyramid(args.input, args.normal_pair, data_level, n_levels, workers=workers, force=True,
										 scan_existing_source_chunks=True)
		elif args.in_place:
			nx_tmp, ny_tmp = _temporary_sibling(args.input), _temporary_sibling(args.normal_pair)
			rechunk_normal_pair(nx_src=args.input, ny_src=args.normal_pair, nx_dst=nx_tmp, ny_dst=ny_tmp, overwrite=True, **common)
			_swap_dir(nx_tmp, args.input)
			_swap_dir(ny_tmp, args.normal_pair)
		else:
			rechunk_normal_pair(nx_src=args.input, ny_src=args.normal_pair, nx_dst=args.output, ny_dst=args.normal_output,
								overwrite=bool(args.overwrite), **common)
		return

	if args.rebuild_scales_only:
		build_scalar_omezarr_pyramid(args.input, data_level, args.levels or infer_n_levels(args.input), workers=workers, label=args.input.name,
									 force=True, zero_overrides=bool(args.zero_overrides), scan_existing_source_chunks=True)
	elif args.in_place:
		tmp = _temporary_sibling(args.input)
		rechunk_scalar(src_root=args.input, dst_root=tmp, zero_overrides=bool(args.zero_overrides), overwrite=True, label=args.input.name, **common)
		_swap_dir(tmp, args.input)
	else:
		rechunk_scalar(src_root=args.input, dst_root=args.output, zero_overrides=bool(args.zero_overrides), overwrite=bool(args.overwrite),
					   label=args.output.name, **common)


if __name__ == "__main__":
	main()
