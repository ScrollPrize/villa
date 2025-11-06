#!/usr/bin/env python3
"""Upsample or downsample a directory of 3D TIFF volumes by a uniform factor."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import tifffile
from scipy.ndimage import zoom

from vesuvius.image_proc.run import gather_inputs, run_workflow


def _build_worker(
    output_dir: Path,
    scale: float,
    order: int,
    compression: str | None,
    prefilter: bool,
):
    def _worker(path: Path) -> Path:
        volume = tifffile.imread(path)
        zoom_factors: Iterable[float] = (scale,) * volume.ndim
        scaled = zoom(volume, zoom=tuple(zoom_factors), order=order, prefilter=prefilter)
        scaled = _cast_dtype(scaled, volume.dtype)

        output_path = output_dir / path.name
        tifffile.imwrite(output_path, scaled, compression=compression)
        return output_path

    return _worker


def _cast_dtype(array: np.ndarray, target_dtype: np.dtype) -> np.ndarray:
    """Cast the scaled array back to the input dtype while keeping values in-range."""
    target_dtype = np.dtype(target_dtype)

    if np.issubdtype(target_dtype, np.integer):
        info = np.iinfo(target_dtype)
        array = np.clip(np.rint(array), info.min, info.max)
        return array.astype(target_dtype)

    if np.issubdtype(target_dtype, np.floating):
        return array.astype(target_dtype)

    if target_dtype == np.bool_:
        return array > 0.5

    return array.astype(target_dtype, copy=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scale 3D TIFF volumes by a uniform factor.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_dir", help="Directory containing 3D .tif/.tiff volumes.")
    parser.add_argument("output_dir", help="Directory where scaled volumes will be written.")
    parser.add_argument(
        "--scale",
        type=float,
        required=True,
        help="Uniform scale factor (>1 upscales, <1 downscales).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of worker processes.",
    )
    parser.add_argument(
        "--order",
        type=int,
        default=1,
        choices=range(0, 6),
        help="Spline interpolation order used by scipy.ndimage.zoom.",
    )
    parser.add_argument(
        "--prefilter",
        action="store_true",
        help="Apply spline prefilter before interpolation (useful when order > 1).",
    )
    parser.add_argument(
        "--no-prefilter",
        dest="prefilter",
        action="store_false",
        help="Disable spline prefiltering.",
    )
    parser.add_argument(
        "--compression",
        default="zlib",
        help="Compression to use when saving TIFFs (pass '' to disable).",
    )
    parser.set_defaults(prefilter=False)
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for TIFF volumes inside the input directory.",
    )

    args = parser.parse_args()

    if args.scale <= 0:
        parser.error("--scale must be greater than 0.")

    if args.num_workers <= 0:
        parser.error("--num-workers must be at least 1.")

    if args.compression == "":
        args.compression = None

    return args


def main() -> int:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    items = gather_inputs(
        input_dir,
        image_extensions=(".tif", ".tiff"),
        include_zarr=False,
        recursive=args.recursive,
    )

    if not items:
        print(f"No TIFF volumes found in {input_dir}")
        return 1

    worker = _build_worker(
        output_dir=output_dir,
        scale=args.scale,
        order=args.order,
        compression=args.compression,
        prefilter=args.prefilter,
    )

    desc = f"Scaling volumes ({args.scale:g}x)"
    results = run_workflow(
        worker_fn=worker,
        inputs=items,
        num_workers=args.num_workers,
        progress_desc=desc,
    )

    print(f"Wrote {len(results)} volume(s) to {output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
