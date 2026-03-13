#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,63).__str__()

import click
import cv2


SUPPORTED_SUFFIXES = {".jpeg", ".jpg", ".png", ".tif", ".tiff",}
NONZERO_PADDING = 16


def iter_image_paths(source_root: Path) -> list[Path]:
    image_paths: list[Path] = []
    for path in sorted(source_root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        image_paths.append(path)
    return image_paths


def destination_path_for(source_root: Path, dest_root: Path, image_path: Path) -> Path:
    relative_path = image_path.relative_to(source_root)
    parts = relative_path.parts

    if len(parts) == 1:
        top_level_dir = dest_root
        flattened_name = Path(parts[0]).stem
    else:
        top_level_dir = dest_root / parts[0]
        flattened_name = "_".join([*parts[1:-1], Path(parts[-1]).stem])

    return top_level_dir / f"{flattened_name}.png"


def crop_to_nonzero_region(image: cv2.typing.MatLike, padding: int) -> cv2.typing.MatLike:
    nonzero_points = cv2.findNonZero(image)
    if nonzero_points is None:
        return image

    x, y, width, height = cv2.boundingRect(nonzero_points)
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(image.shape[1], x + width + padding)
    y1 = min(image.shape[0], y + height + padding)
    return image[y0:y1, x0:x1]


def downsample_to_png(source_path: Path, dest_path: Path, downsampling_factor: int) -> None:
    image = cv2.imread(str(source_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image data from {source_path}")

    resized = cv2.resize(
        image,
        (
            max(1, image.shape[1] // downsampling_factor),
            max(1, image.shape[0] // downsampling_factor),
        ),
        interpolation=cv2.INTER_AREA,
    )
    resized = crop_to_nonzero_region(resized, padding=NONZERO_PADDING)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(dest_path), resized):
        raise ValueError(f"Failed to write image data to {dest_path}")


@click.command()
@click.argument("source", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("dest", type=click.Path(file_okay=False, path_type=Path))
@click.argument("downsampling_factor", type=click.IntRange(min=1))
def main(source: Path, dest: Path, downsampling_factor: int) -> None:
    """Downsample images from SOURCE and write flattened 1-channel PNGs into DEST."""
    source = source.resolve()
    dest = dest.resolve()

    image_paths = iter_image_paths(source)
    if not image_paths:
        raise click.ClickException(f"No supported image files found under {source}")

    written = 0
    skipped = 0

    for image_path in image_paths:
        print(image_path)
        dest_path = destination_path_for(source, dest, image_path)
        try:
            downsample_to_png(image_path, dest_path, downsampling_factor)
            written += 1
        except ValueError:
            skipped += 1

    click.echo(f"Wrote {written} PNG files to {dest}")
    if skipped:
        click.echo(f"Skipped {skipped} unreadable image files")


if __name__ == "__main__":
    main()
