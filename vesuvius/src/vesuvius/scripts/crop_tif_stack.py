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
import tifffile
from tqdm import tqdm


def read_tile_region(
    tif: tifffile.TiffFile,
    page,
    y_start: int,
    y_end: int,
    x_start: int,
    x_end: int,
) -> np.ndarray:
    """Read a region from an open TIFF using direct tile decoding."""
    img_height, img_width = page.shape[:2]
    y_end = min(y_end, img_height)
    x_end = min(x_end, img_width)

    if not page.is_tiled:
        full = page.asarray()
        return full[y_start:y_end, x_start:x_end]

    tile_height, tile_width = page.chunks[:2]
    dtype = page.dtype

    tile_row_start = y_start // tile_height
    tile_row_end = (y_end + tile_height - 1) // tile_height
    tile_col_start = x_start // tile_width
    tile_col_end = (x_end + tile_width - 1) // tile_width

    tiles_per_row = (img_width + tile_width - 1) // tile_width

    out_height = y_end - y_start
    out_width = x_end - x_start
    result = np.zeros((out_height, out_width), dtype=dtype)

    offsets = page.dataoffsets
    bytecounts = page.databytecounts
    fh = tif.filehandle

    for tile_row in range(tile_row_start, tile_row_end):
        for tile_col in range(tile_col_start, tile_col_end):
            tile_idx = tile_row * tiles_per_row + tile_col
            if tile_idx >= len(offsets):
                continue

            fh.seek(offsets[tile_idx])
            raw = fh.read(bytecounts[tile_idx])
            tile_data = page.decode(raw, tile_idx)[0]
            tile_data = np.squeeze(tile_data)

            tile_y = tile_row * tile_height
            tile_x = tile_col * tile_width

            src_y_start = max(0, y_start - tile_y)
            src_y_end = min(tile_data.shape[0], y_end - tile_y)
            src_x_start = max(0, x_start - tile_x)
            src_x_end = min(tile_data.shape[1], x_end - tile_x)

            dst_y_start = tile_y + src_y_start - y_start
            dst_y_end = tile_y + src_y_end - y_start
            dst_x_start = tile_x + src_x_start - x_start
            dst_x_end = tile_x + src_x_end - x_start

            result[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                tile_data[src_y_start:src_y_end, src_x_start:src_x_end]

    return result


def process_single_tif(
    tif_path: Path,
    idx: int,
    crop_bounds: List[Tuple[int, int]],
    crop_dirs: List[Path],
    height: int,
    tile_size: int,
) -> None:
    """Process all crops for a single TIF file."""
    with tifffile.TiffFile(str(tif_path)) as tif:
        page = tif.pages[0]

        for crop_idx, (left, right) in enumerate(crop_bounds):
            # Read the full crop region
            crop_data = read_tile_region(tif, page, 0, height, left, right)

            # Write as single tiled image
            output_path = crop_dirs[crop_idx] / f"{idx:02d}.tif"
            tifffile.imwrite(
                str(output_path),
                crop_data,
                tile=(tile_size, tile_size),
                compression='lzw',
            )


def crop_tif_stack(input_dir: Path, output_dir: Path, max_width: int, num_workers: int, tile_size: int):
    """Crop all TIFs horizontally, streaming tile-rows to minimize memory."""
    tif_files = sorted(input_dir.glob("*.tif")) + sorted(input_dir.glob("*.tiff"))
    tif_files = sorted(set(tif_files))

    if not tif_files:
        print(f"No TIF files found in {input_dir}")
        return

    print(f"Found {len(tif_files)} TIF files")

    with tifffile.TiffFile(str(tif_files[0])) as tif:
        page = tif.pages[0]
        height, width = page.shape[:2]
        is_tiled = page.is_tiled
        if is_tiled:
            print(f"Tiled TIF: tile size {page.tilewidth}x{page.tilelength}")
        else:
            print("Warning: TIF is not tiled, will read full image each time")

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
