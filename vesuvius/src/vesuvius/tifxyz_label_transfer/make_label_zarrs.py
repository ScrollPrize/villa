#!/usr/bin/env python3
"""Convert transferred full-resolution label TIFFs to provenance-aware label Zarrs."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import tifffile

from vc_reader.reader import write_pyramid_level
from vesuvius.label_zarr import (
    DEFAULT_LEVELS,
    LABEL_SLICE,
    create_label_array,
    create_label_group,
    pyramid_shapes,
)
from vesuvius.utils.cli import HyphenUnderscoreParser

BLOCK_SIZE = 1024


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
    ``LABEL_SLICE``.
    """

    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists: {output_path}")
        shutil.rmtree(output_path)

    shapes = pyramid_shapes(image_2d.shape, levels)
    group = create_label_group(output_path, levels=levels)
    if extra_attributes:
        group.attrs.update(extra_attributes)

    datasets = [
        create_label_array(group, str(level), shape=shape, dtype=image_2d.dtype)
        for level, shape in enumerate(shapes)
    ]
    height, width = image_2d.shape
    for y0 in range(0, height, BLOCK_SIZE):
        y1 = min(height, y0 + BLOCK_SIZE)
        for x0 in range(0, width, BLOCK_SIZE):
            x1 = min(width, x0 + BLOCK_SIZE)
            block = image_2d[y0:y1, x0:x1]
            if block.any():
                datasets[0][LABEL_SLICE, y0:y1, x0:x1] = block
    for level in range(1, levels):
        write_pyramid_level(output_path, level, factors=(1, 2, 2), reduction="nearest")


def _provenance_attributes(input_path: Path) -> Dict[str, object]:
    attributes: Dict[str, object] = {"source_image": str(input_path)}
    report_path = input_path.with_name(input_path.stem + ".report.json")
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Could not read provenance report {report_path}: {exc}"
            ) from exc
        if not isinstance(report, dict):
            raise ValueError(
                f"Provenance report {report_path} must contain a JSON object"
            )
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
    parser = HyphenUnderscoreParser(description=__doc__.splitlines()[0])
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
