#!/usr/bin/env python3
"""Convert transferred full-resolution label TIFFs into OME-Zarr pyramids.

Output layout follows the ink-detection preprocessing convention
(``koine_machines/preprocessing/create_label_zarrs.py``): each 2D label is
embedded at slice 32 of a 65-deep z/y/x volume, written as a 6-level
OME-NGFF 0.4 pyramid with nearest-neighbour downsampling, chunks of
(65, 128, 128), Blosc zstd level-3 bitshuffle compression, ``/`` dimension
separators, and ``_ARRAY_DIMENSIONS`` on every level. Only the label slice
is ever populated, so levels are computed in 2D and streamed block-wise —
the 65-deep volume is never materialised (at 9k x 30k+ full resolution it
would not fit in memory).

Unlike the raw annotation Zarrs this pipeline consumes, the outputs carry
provenance attributes (source image, canvas size, transfer report) so a
future canvas-offset hunt does not start from empty ``.zattrs``.

Requires ``zarr<3`` (the v2 API, as used by the koine scripts).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import tifffile
import zarr
from numcodecs import Blosc

AXES = [
    {"name": "z", "type": "space"},
    {"name": "y", "type": "space"},
    {"name": "x", "type": "space"},
]
ARRAY_DIMENSIONS = ["z", "y", "x"]
DEFAULT_LEVELS = 6
VOLUME_DEPTH = 65
LABEL_SLICE = 32
CHUNK_SHAPE = (VOLUME_DEPTH, 128, 128)
BLOCK_SIZE = 1024
LABEL_COMPRESSOR = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)


def _multiscales_metadata(name: str, levels: int) -> Dict[str, object]:
    datasets = []
    for level in range(levels):
        scale_factor = float(2**level)
        datasets.append(
            {
                "path": str(level),
                "coordinateTransformations": [
                    {"type": "scale", "scale": [1.0, scale_factor, scale_factor]}
                ],
            }
        )
    return {
        "multiscales": [
            {
                "name": name,
                "version": "0.4",
                "axes": AXES,
                "datasets": datasets,
            }
        ]
    }


def _pyramid_shapes(image_shape: Sequence[int], levels: int) -> List[tuple[int, int, int]]:
    if levels < 1:
        raise ValueError("levels must be at least 1")
    height, width = int(image_shape[0]), int(image_shape[1])
    shapes: List[tuple[int, int, int]] = []
    for _ in range(levels):
        shapes.append((VOLUME_DEPTH, height, width))
        height = (height + 1) // 2
        width = (width + 1) // 2
    return shapes


def _load_label_2d(path: Path) -> np.ndarray:
    image = np.squeeze(np.asarray(tifffile.imread(path)))
    if image.ndim == 3:
        image = image[..., 0]
    if image.ndim != 2:
        raise ValueError(f"expected a 2D label image at {path}, got shape {image.shape}")
    return np.ascontiguousarray(image)


def write_label_zarr(
    image_2d: np.ndarray,
    output_path: Path,
    *,
    levels: int = DEFAULT_LEVELS,
    overwrite: bool = False,
    extra_attributes: Optional[Dict[str, object]] = None,
) -> None:
    """Write ``image_2d`` as a koine-convention OME-Zarr label pyramid.

    Every level holds a 65-deep volume whose only non-zero slice is
    ``LABEL_SLICE``; nearest downsampling of that volume equals nearest
    downsampling of the 2D image, so levels are derived purely in 2D.
    """

    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists: {output_path}")
        shutil.rmtree(output_path)

    shapes = _pyramid_shapes(image_2d.shape, levels)
    group = zarr.open_group(str(output_path), mode="w")
    group.attrs.update(_multiscales_metadata(output_path.stem, levels))
    if extra_attributes:
        group.attrs.update(extra_attributes)

    level_image = image_2d
    for level, shape in enumerate(shapes):
        dataset = group.create_dataset(
            str(level),
            shape=shape,
            chunks=CHUNK_SHAPE,
            dtype=image_2d.dtype,
            compressor=LABEL_COMPRESSOR,
            fill_value=0,
            overwrite=True,
            dimension_separator="/",
            write_empty_chunks=False,
        )
        dataset.attrs["_ARRAY_DIMENSIONS"] = ARRAY_DIMENSIONS

        height, width = level_image.shape
        for y0 in range(0, height, BLOCK_SIZE):
            y1 = min(height, y0 + BLOCK_SIZE)
            for x0 in range(0, width, BLOCK_SIZE):
                x1 = min(width, x0 + BLOCK_SIZE)
                block = level_image[y0:y1, x0:x1]
                if not block.any():
                    continue
                dataset[LABEL_SLICE, y0:y1, x0:x1] = block

        level_image = np.ascontiguousarray(level_image[::2, ::2])


def _provenance_attributes(input_path: Path) -> Dict[str, object]:
    attributes: Dict[str, object] = {"source_image": str(input_path)}
    report_path = input_path.with_name(input_path.stem + ".report.json")
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text())
        except (OSError, json.JSONDecodeError):
            report = None
        if isinstance(report, dict):
            attributes["transfer_report"] = str(report_path)
            shape = report.get("output_shape")
            if isinstance(shape, (list, tuple)) and len(shape) == 2:
                attributes["canvas_size"] = [int(shape[0]), int(shape[1])]
    return attributes


def convert_label_tiff(
    input_path: Path,
    output_path: Optional[Path] = None,
    *,
    levels: int = DEFAULT_LEVELS,
    overwrite: bool = False,
) -> Path:
    resolved_output = output_path or input_path.with_suffix(".zarr")
    image = _load_label_2d(input_path)
    attributes = _provenance_attributes(input_path)
    attributes.setdefault("canvas_size", [int(image.shape[0]), int(image.shape[1])])
    write_label_zarr(
        image,
        resolved_output,
        levels=levels,
        overwrite=overwrite,
        extra_attributes=attributes,
    )
    return resolved_output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("inputs", nargs="+", type=Path, help="label TIFF(s) to convert")
    parser.add_argument(
        "--levels",
        type=int,
        default=DEFAULT_LEVELS,
        help=f"pyramid levels to write (default: {DEFAULT_LEVELS})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output .zarr path (single input only; default: input with .zarr suffix)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output .zarr",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output is not None and len(args.inputs) != 1:
        raise SystemExit("--output requires exactly one input")
    for input_path in args.inputs:
        if not input_path.is_file():
            raise SystemExit(f"input not found: {input_path}")
        output = convert_label_tiff(
            input_path,
            args.output,
            levels=args.levels,
            overwrite=args.overwrite,
        )
        print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
