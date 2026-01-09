#!/usr/bin/env python3
"""
Extract non-overlapping chunks from an OME-Zarr and save as 3D TIFFs with LZW compression.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import tifffile
import zarr
from tqdm import tqdm


def save_chunk(args):
    """Save a single chunk as a TIFF file."""
    zarr_path, zarr2_path, resolution, z_start, y_start, x_start, chunk_size, output_dir, output_dir2, ignore_label, bg_dir = args

    # Open zarr in each worker process
    store = zarr.open(zarr_path, mode='r')
    data = store[resolution]

    z_end = min(z_start + chunk_size, data.shape[0])
    y_end = min(y_start + chunk_size, data.shape[1])
    x_end = min(x_start + chunk_size, data.shape[2])

    chunk_data = data[z_start:z_end, y_start:y_end, x_start:x_end]

    # Skip empty chunks
    if chunk_data.size == 0:
        return None

    # Skip chunks that contain only zeros
    if (chunk_data == 0).all():
        return None

    # Skip chunks that contain only the ignore label
    if ignore_label is not None and (chunk_data == ignore_label).all():
        return None

    # Check if chunk is background (contains only zeros and ignore label, but both present)
    is_background = False
    if ignore_label is not None:
        mask = (chunk_data == 0) | (chunk_data == ignore_label)
        has_zeros = (chunk_data == 0).any()
        has_ignore = (chunk_data == ignore_label).any()
        is_background = mask.all() and has_zeros and has_ignore

    if is_background:
        if bg_dir is None:
            return None
        target_dir = bg_dir
    else:
        target_dir = output_dir

    output_filename = f"{chunk_size}_z{z_start}_y{y_start}_x{x_start}.tif"
    output_path = Path(target_dir) / output_filename

    tifffile.imwrite(
        output_path,
        chunk_data,
        compression='lzw',
        photometric='minisblack',
    )

    # Process second zarr if provided
    if zarr2_path is not None:
        store2 = zarr.open(zarr2_path, mode='r')
        data2 = store2[resolution]
        chunk_data2 = data2[z_start:z_end, y_start:y_end, x_start:x_end]

        output_path2 = Path(output_dir2) / output_filename
        tifffile.imwrite(
            output_path2,
            chunk_data2,
            compression='lzw',
            photometric='minisblack',
        )

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Extract non-overlapping chunks from OME-Zarr and save as 3D TIFFs"
    )
    parser.add_argument(
        "--zarr-path",
        type=str,
        required=True,
        help="Path to the OME-Zarr file"
    )
    parser.add_argument(
        "--resolution",
        type=str,
        required=True,
        help="Resolution level to extract from (e.g., '0', '1', '2')"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        required=True,
        help="Chunk size (single int applied to z, y, x)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for TIFF files"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker processes (default: 4)"
    )
    parser.add_argument(
        "--zarr2",
        type=str,
        default=None,
        help="Path to a second OME-Zarr file to extract chunks from the same positions"
    )
    parser.add_argument(
        "--out2",
        type=str,
        default=None,
        help="Output directory for second zarr's TIFF files (required if --zarr2 is provided)"
    )
    parser.add_argument(
        "--ignore-label",
        type=int,
        default=None,
        help="Skip chunks where the first zarr contains only this label value"
    )
    parser.add_argument(
        "--bg-dir",
        type=str,
        default=None,
        help="Output directory for background chunks (containing only zeros and ignore label)"
    )

    args = parser.parse_args()

    # Validate that --out2 is provided if --zarr2 is provided
    if args.zarr2 is not None and args.out2 is None:
        parser.error("--out2 is required when --zarr2 is provided")

    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_dir2 = None
    if args.out2 is not None:
        output_dir2 = Path(args.out2)
        output_dir2.mkdir(parents=True, exist_ok=True)

    bg_dir = None
    if args.bg_dir is not None:
        bg_dir = Path(args.bg_dir)
        bg_dir.mkdir(parents=True, exist_ok=True)

    # Open zarr to get dimensions
    store = zarr.open(args.zarr_path, mode='r')
    data = store[args.resolution]

    z_size, y_size, x_size = data.shape[:3]
    chunk_size = args.chunk_size

    print(f"Data shape at resolution '{args.resolution}': {data.shape}")
    print(f"Chunk size: {chunk_size}")
    print(f"Output directory: {output_dir}")
    if args.zarr2 is not None:
        print(f"Second zarr: {args.zarr2}")
        print(f"Second output directory: {output_dir2}")
    if args.ignore_label is not None:
        print(f"Ignoring chunks containing only label: {args.ignore_label}")
    if bg_dir is not None:
        print(f"Background directory: {bg_dir}")

    # Generate all chunk coordinates
    chunk_coords = []
    for z_start in range(0, z_size, chunk_size):
        for y_start in range(0, y_size, chunk_size):
            for x_start in range(0, x_size, chunk_size):
                chunk_coords.append((
                    args.zarr_path,
                    args.zarr2,
                    args.resolution,
                    z_start,
                    y_start,
                    x_start,
                    chunk_size,
                    str(output_dir),
                    str(output_dir2) if output_dir2 is not None else None,
                    args.ignore_label,
                    str(bg_dir) if bg_dir is not None else None
                ))

    total_chunks = len(chunk_coords)
    print(f"Total chunks to process: {total_chunks}")

    # Process chunks in parallel
    saved_count = 0
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(save_chunk, coord): coord for coord in chunk_coords}

        with tqdm(total=total_chunks, desc="Processing chunks") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    saved_count += 1
                pbar.update(1)

    print(f"Done! Saved {saved_count}/{total_chunks} chunks to {output_dir}")


if __name__ == "__main__":
    main()
