#!/usr/bin/env python3
"""
Binarize 3D TIFF volumes in a directory using tifffile.

For each input TIFF, any non-zero value becomes 1 (binary). Supports
multiprocessing for speed and shows a tqdm progress bar.

Examples
--------
- Write outputs to a new folder next to input (default):
    python binarize_tifs.py /path/to/tifs

- Overwrite in place safely (via temp file + replace):
    python binarize_tifs.py /path/to/tifs --in-place

- Keep original dtype instead of uint8 (still 0/1 values):
    python binarize_tifs.py /path/to/tifs --preserve-dtype

Requirements
------------
pip install tifffile tqdm
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from tifffile import imread, imwrite
from tqdm import tqdm


def find_tifs(input_dir: Path) -> List[Path]:
    exts = {".tif", ".tiff"}
    return sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])


def binarize_array(arr: np.ndarray, preserve_dtype: bool) -> np.ndarray:
    mask = arr != 0
    if preserve_dtype:
        return mask.astype(arr.dtype, copy=False)
    return mask.astype(np.uint8, copy=False)


def process_one(
    in_path: Path,
    out_path: Path,
    preserve_dtype: bool,
    compression: str | None,
) -> Tuple[Path, Path, bool, str | None]:
    try:
        vol = imread(in_path)
        out = binarize_array(vol, preserve_dtype)

        if out_path == in_path:
            tmp_path = in_path.with_suffix(in_path.suffix + ".tmp")
            imwrite(tmp_path, out, compression=compression)
            os.replace(tmp_path, in_path)
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            imwrite(out_path, out, compression=compression)

        return (in_path, out_path, True, None)
    except Exception as e:  # pragma: no cover - defensive
        return (in_path, out_path, False, str(e))


def make_out_path(in_path: Path, input_dir: Path, output_dir: Path | None, in_place: bool) -> Path:
    if in_place:
        return in_path
    if output_dir is None:
        output_dir = input_dir.with_name(input_dir.name + "_bin")
    return output_dir / in_path.name


def run(
    input_dir: Path,
    output_dir: Path | None,
    in_place: bool,
    workers: int,
    preserve_dtype: bool,
    compression: str | None,
) -> None:
    files = find_tifs(input_dir)
    if not files:
        raise SystemExit(f"No TIFF files found in {input_dir}")

    tasks = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for f in files:
            out_f = make_out_path(f, input_dir, output_dir, in_place)
            tasks.append(
                ex.submit(process_one, f, out_f, preserve_dtype, compression)
            )

        errors = []
        for fut in tqdm(as_completed(tasks), total=len(tasks), desc="Binarizing"):
            in_f, out_f, ok, err = fut.result()
            if not ok:
                errors.append((in_f, err))

    if errors:
        msg = "\n".join([f"- {p}: {e}" for p, e in errors])
        raise SystemExit(f"Encountered {len(errors)} errors:\n{msg}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Binarize 3D TIFFs (non-zero -> 1)")
    p.add_argument("input_dir", type=Path, help="Directory containing .tif/.tiff files")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write outputs (default: <input_dir>_bin). Ignored if --in-place.",
    )
    p.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite input files in place (uses atomic replace).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Number of worker processes (default: CPU count - 1)",
    )
    p.add_argument(
        "--preserve-dtype",
        action="store_true",
        help="Keep original dtype (values 0/1). Default writes uint8.",
    )
    p.add_argument(
        "--compression",
        type=str,
        default="zlib",
        choices=["none", "zlib", "lzma", "jpeg", "webp", "zstd"],
        help="TIFF compression for outputs (default: zlib).",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    if not args.input_dir.is_dir():
        p.error(f"input_dir is not a directory: {args.input_dir}")

    if args.in_place and args.output_dir is not None:
        p.error("--output-dir cannot be used with --in-place")

    # Map "none" to no compression
    if args.compression == "none":
        args.compression = None

    return args


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    run(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        in_place=args.in_place,
        workers=args.workers,
        preserve_dtype=args.preserve_dtype,
        compression=args.compression,
    )


if __name__ == "__main__":
    main()

