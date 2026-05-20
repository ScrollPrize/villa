#!/usr/bin/env python3
"""Rechunk an OME-Zarr and rebuild scales with TensorStore-backed I/O."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_LASAGNA_DIR = Path(__file__).resolve().parents[1]
if str(_LASAGNA_DIR) not in sys.path:
	sys.path.insert(0, str(_LASAGNA_DIR))

from tensorstore_omezarr import (
	TensorStoreConfig,
	normalize_chunk,
	rebuild_normal_pair_in_place,
	rebuild_scalar_in_place,
	rechunk_normal_pair_in_place,
	rechunk_normal_pair_out_of_place_changed,
	rechunk_normal_pair_out_of_place,
	rechunk_scalar_in_place,
	rechunk_scalar_out_of_place_changed,
	rechunk_scalar_out_of_place,
)


def _parse_chunk(value: str) -> tuple[int, int, int]:
	parts = [p.strip() for p in value.split(",")]
	if len(parts) == 1:
		return normalize_chunk(int(parts[0]))
	if len(parts) != 3:
		raise argparse.ArgumentTypeError("chunk size must be N or Z,Y,X")
	try:
		return normalize_chunk((int(parts[0]), int(parts[1]), int(parts[2])))
	except ValueError as exc:
		raise argparse.ArgumentTypeError(str(exc)) from exc


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
	parser.add_argument("--overwrite-changed", action="store_true", help="Update an existing out-of-place output by writing only chunks whose values changed.")
	parser.add_argument("-j", "--workers", type=int, default=0, help="Worker processes. Default: CPU count.")
	parser.add_argument("--cache-bytes", type=_cache_bytes, default=4 << 30, help="Total TensorStore cache budget split across worker processes.")
	parser.add_argument("--file-io-threads", type=int, default=2, help="TensorStore file I/O concurrency per worker process.")
	parser.add_argument("--data-copy-threads", type=int, default=1, help="TensorStore data copy concurrency per worker process.")
	args = parser.parse_args()

	cfg = TensorStoreConfig(
		cache_pool_bytes=int(args.cache_bytes),
		file_io_threads=int(args.file_io_threads),
		data_copy_threads=int(args.data_copy_threads),
		workers=int(args.workers),
	)

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
	if args.overwrite and args.overwrite_changed:
		raise SystemExit("use either --overwrite or --overwrite-changed, not both")
	if args.overwrite_changed and args.in_place:
		raise SystemExit("--overwrite-changed is an out-of-place mode; do not pass --in-place")
	if args.overwrite_changed and args.rebuild_scales_only:
		raise SystemExit("--overwrite-changed cannot be combined with --rebuild-scales-only")

	if args.normal_pair is not None:
		if args.rebuild_scales_only:
			rebuild_normal_pair_in_place(
				nx_root=args.input,
				ny_root=args.normal_pair,
				data_level=int(args.data_level),
				n_levels=args.levels,
				cfg=cfg,
			)
		elif args.in_place:
			rechunk_normal_pair_in_place(
				nx_root=args.input,
				ny_root=args.normal_pair,
				chunk=args.chunk_size,
				data_level=int(args.data_level),
				n_levels=args.levels,
				cfg=cfg,
			)
		elif args.overwrite_changed:
			rechunk_normal_pair_out_of_place_changed(
				nx_src=args.input,
				ny_src=args.normal_pair,
				nx_dst=args.output,
				ny_dst=args.normal_output,
				chunk=args.chunk_size,
				data_level=int(args.data_level),
				n_levels=args.levels,
				cfg=cfg,
			)
		else:
			rechunk_normal_pair_out_of_place(
				nx_src=args.input,
				ny_src=args.normal_pair,
				nx_dst=args.output,
				ny_dst=args.normal_output,
				chunk=args.chunk_size,
				data_level=int(args.data_level),
				n_levels=args.levels,
				cfg=cfg,
				overwrite=bool(args.overwrite),
			)
		return

	if args.rebuild_scales_only:
		rebuild_scalar_in_place(
			root=args.input,
			data_level=int(args.data_level),
			n_levels=args.levels,
			cfg=cfg,
			zero_overrides=bool(args.zero_overrides),
			label=args.input.name,
		)
	elif args.overwrite_changed:
		rechunk_scalar_out_of_place_changed(
			src_root=args.input,
			dst_root=args.output,
			chunk=args.chunk_size,
			data_level=int(args.data_level),
			n_levels=args.levels,
			cfg=cfg,
			zero_overrides=bool(args.zero_overrides),
			label=args.output.name,
		)
	elif args.in_place:
		rechunk_scalar_in_place(
			root=args.input,
			chunk=args.chunk_size,
			data_level=int(args.data_level),
			n_levels=args.levels,
			cfg=cfg,
			zero_overrides=bool(args.zero_overrides),
			label=args.input.name,
		)
	else:
		rechunk_scalar_out_of_place(
			src_root=args.input,
			dst_root=args.output,
			chunk=args.chunk_size,
			data_level=int(args.data_level),
			n_levels=args.levels,
			cfg=cfg,
			zero_overrides=bool(args.zero_overrides),
			overwrite=bool(args.overwrite),
			label=args.output.name,
		)


if __name__ == "__main__":
	main()
