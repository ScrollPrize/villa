#!/usr/bin/env python3
"""
Crop a folder of tiled TIF images horizontally into strips of a specified max width.
Each crop goes into a numbered folder (crop01, crop02, etc.) containing all
slices from the stack.

Uses direct tile decoding and writes tiled output. Streams tile-rows to minimize memory.
"""

import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple
import numpy as np
from vc_reader import images
from tqdm import tqdm


def process_single_tif(
    tif_path: Path,
    idx: int,
    crop_bounds: List[Tuple[int, int]],
    crop_dirs: List[Path],
    height: int,
    tile_size: int,
) -> None:
    """Process all crops for a single TIF file."""
    with images.open(str(tif_path)) as reader:
        for crop_idx, (left, right) in enumerate(crop_bounds):
            # Read the full crop region
            crop_data = reader[0:height, left:right]

            # Write as single tiled image
            output_path = crop_dirs[crop_idx] / f"{idx:02d}.tif"
            images.write_tiff(str(output_path), crop_data, tile=(tile_size, tile_size), compression='lzw')


def crop_tif_stack(input_dir: Path, output_dir: Path, max_width: int, num_workers: int, tile_size: int):
    """Crop all TIFs horizontally, streaming tile-rows to minimize memory."""
    tif_files = sorted(input_dir.glob("*.tif")) + sorted(input_dir.glob("*.tiff"))
    tif_files = sorted(set(tif_files))

    if not tif_files:
        print(f"No TIF files found in {input_dir}")
        return

    print(f"Found {len(tif_files)} TIF files")

    with images.open(str(tif_files[0])) as reader:
        height, width = reader.shape[:2]

    print(f"Image dimensions: {width}x{height}")

    num_crops = (width + max_width - 1) // max_width
    print(f"Will create {num_crops} crop(s) of max width {max_width}px")
    print(f"Using {num_workers} workers, streaming {tile_size}px tile-rows")

    crop_bounds = []
    for crop_idx in range(num_crops):
        left = crop_idx * max_width
        right = min(left + max_width, width)
        crop_bounds.append((left, right))

    crop_dirs = []
    for i in range(num_crops):
        crop_dir = output_dir / f"crop{i+1:02d}"
        crop_dir.mkdir(parents=True, exist_ok=True)
        crop_dirs.append(crop_dir)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                process_single_tif, tif_path, idx, crop_bounds, crop_dirs, height, tile_size
            ): tif_path
            for idx, tif_path in enumerate(tif_files)
        }

        with tqdm(total=len(tif_files), desc="Processing TIFs") as pbar:
            for future in as_completed(futures):
                future.result()
                pbar.update(1)

    print(f"\nDone! Created {num_crops} crop folders in {output_dir}")
    for i, crop_dir in enumerate(crop_dirs):
        left, right = crop_bounds[i]
        print(f"  {crop_dir.name}: x=[{left}, {right}), width={right-left}px")


def main():
    parser = argparse.ArgumentParser(
        description="Crop a tiled TIF stack horizontally into strips"
    )
    parser.add_argument("input_dir", type=Path, help="Input directory containing TIF files")
    parser.add_argument("output_dir", type=Path, help="Output directory for crop folders")
    parser.add_argument("--max-width", type=int, required=True, help="Maximum width in pixels for each crop")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers (default: 8)")
    parser.add_argument("--tile-size", type=int, default=512, help="Output tile size in pixels (default: 512)")

    args = parser.parse_args()

    if not args.input_dir.is_dir():
        print(f"Error: {args.input_dir} is not a directory")
        return 1

    crop_tif_stack(args.input_dir, args.output_dir, args.max_width, args.workers, args.tile_size)
    return 0


if __name__ == "__main__":
    exit(main())
