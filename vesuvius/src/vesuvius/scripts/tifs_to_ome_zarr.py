#!/usr/bin/env python3
"""Convert a folder of 2D TIFFs to a 6-level OME-Zarr pyramid.

Uses XY-tile-major processing for efficiency: opens all TIFF files once at the
start, then iterates over XY tile positions, reading each position from all
z-slices and writing complete 3D chunks to zarr.

This avoids the massive overhead of opening/closing files per tile read
(e.g., 3.4M file opens reduced to 65 opens for a typical 65-slice volume),
providing 10-50x speedup for large volumes.

Usage:
    python tifs_to_ome_zarr.py /path/to/tiff/folder /path/to/output.zarr
    python tifs_to_ome_zarr.py /path/to/tiff/folder /path/to/output.zarr --workers 16
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from vc_reader import images
from vc_reader.reader import create_group, ome_multiscales, write_pyramid
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_chunk_size(value: str) -> Tuple[int, int, int]:
    """Parse chunk size: either single int or comma-separated z,y,x."""
    parts = value.split(",")
    if len(parts) == 1:
        size = int(parts[0])
        return (size, size, size)
    elif len(parts) == 3:
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    else:
        raise argparse.ArgumentTypeError(
            f"chunk-size must be single int or z,y,x (got {value})"
        )


def natural_sort_key(path: Path) -> Tuple:
    """Sort key for natural ordering of numbered files."""
    parts = re.split(r"(\d+)", path.stem)
    return tuple(int(p) if p.isdigit() else p.lower() for p in parts)


def discover_tiffs(folder: Path) -> List[Path]:
    """Discover and sort TIFF files in folder."""
    patterns = ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]
    tiffs = []
    for pattern in patterns:
        tiffs.extend(folder.glob(pattern))
    tiffs = sorted(set(tiffs), key=natural_sort_key)
    if not tiffs:
        raise ValueError(f"No TIFF files found in {folder}")
    return tiffs


def get_tiff_info(tiff_path: Path) -> Tuple[Tuple[int, int], np.dtype]:
    """Get shape and dtype of a TIFF's first page."""
    with images.open(str(tiff_path)) as reader:
        shape = reader.shape[1:] if reader.axes[0] == "z" else reader.shape
        return tuple(shape), np.dtype(reader.dtype)


def scan_tiff_minmax(tiff_path: str) -> Tuple[float, float]:
    """Scan a single TIFF to find min/max values."""
    data = images.read(tiff_path, page=0)
    return float(data.min()), float(data.max())


def scan_global_minmax(tiff_paths: List[str], num_workers: int) -> Tuple[float, float]:
    """Parallel scan of all TIFFs to find global min/max."""
    global_min = float('inf')
    global_max = float('-inf')

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(scan_tiff_minmax, p) for p in tiff_paths]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Scanning min/max"):
            file_min, file_max = future.result()
            global_min = min(global_min, file_min)
            global_max = max(global_max, file_max)

    return global_min, global_max


def rescale_to_uint8(data: np.ndarray, global_min: float, global_max: float) -> np.ndarray:
    """Rescale data from [global_min, global_max] to uint8 [0, 255]."""
    if global_max == global_min:
        return np.zeros(data.shape, dtype=np.uint8)
    scaled = (data.astype(np.float32) - global_min) / (global_max - global_min) * 255.0
    return np.clip(scaled, 0, 255).astype(np.uint8)


# Global reference for worker functions (set before parallel execution)
_global_readers: Optional[list] = None
_global_zarr_arr = None
_global_rescale: bool = False
_global_min: float = 0.0
_global_max: float = 0.0


def process_xy_tile(args: Tuple) -> Tuple[int, int, int, int]:
    """Process one XY tile position across all z-slices.

    Reads the tile from all z-slices, stacks into a 3D chunk, and writes
    to zarr. Uses the global pool and zarr array references.

    Parameters
    ----------
    args : Tuple
        (tile_idx, y_start, x_start, y_end, x_end, num_z)

    Returns
    -------
    Tuple[int, int, int, int]
        (y_start, x_start, actual_height, actual_width) for verification.
    """
    tile_idx, y_start, x_start, y_end, x_end, num_z = args

    # Read tile from each z-slice
    slices = []
    for z in range(num_z):
        tile_2d = _global_readers[z][y_start:y_end, x_start:x_end]
        slices.append(tile_2d)

    # Stack into 3D chunk (z, y, x)
    chunk_3d = np.stack(slices, axis=0)

    # Apply uint16→uint8 rescaling if needed
    if _global_rescale:
        chunk_3d = rescale_to_uint8(chunk_3d, _global_min, _global_max)

    # Write to zarr (complete chunk in one operation)
    _global_zarr_arr[:, y_start:y_start + chunk_3d.shape[1], x_start:x_start + chunk_3d.shape[2]] = chunk_3d

    return (y_start, x_start, chunk_3d.shape[1], chunk_3d.shape[2])


def convert_tifs_to_ome_zarr(
    input_folder: Path,
    output_path: Path,
    num_levels: int = 6,
    chunk_shape: Tuple[int, int, int] = (64, 256, 256),
    num_workers: int = 8,
    compressor_level: int = 3,
) -> None:
    """Convert folder of 2D TIFFs to OME-Zarr pyramid.

    Uses XY-tile-major processing for efficiency: opens all TIFF files once at
    the start, then iterates over XY tile positions, reading each position from
    all z-slices and writing complete 3D chunks to zarr.

    This avoids the massive overhead of opening/closing files per tile read
    (3.4M file opens reduced to 65 opens for a typical 65-slice volume).

    Parameters
    ----------
    input_folder : Path
        Folder containing 2D TIFF files (one per z-slice).
    output_path : Path
        Output .zarr path.
    num_levels : int
        Number of pyramid levels (default 6).
    chunk_shape : Tuple[int, int, int]
        Chunk shape for zarr (z, y, x). Default (64, 256, 256).
    num_workers : int
        Number of threads for parallel tile reading within each z-slice.
    compressor_level : int
        Blosc compression level (1-9).
    """
    # Discover TIFFs
    logger.info(f"Discovering TIFFs in {input_folder}")
    tiff_paths = discover_tiffs(input_folder)
    num_z = len(tiff_paths)
    logger.info(f"Found {num_z} TIFF files")

    # Convert to strings for thread safety
    tiff_path_strs = [str(p) for p in tiff_paths]

    # Get image dimensions from first TIFF
    shape_2d, dtype = get_tiff_info(tiff_paths[0])
    height, width = shape_2d
    logger.info(f"TIFF shape: {shape_2d}, dtype: {dtype}")

    # Detect if we need uint16→uint8 conversion
    global _global_rescale, _global_min, _global_max
    output_dtype = dtype
    if dtype == np.uint16:
        logger.info("Detected uint16 input, will convert to uint8 with full-range rescaling")
        logger.info("Scanning all TIFFs to find global min/max...")
        global_min, global_max = scan_global_minmax(tiff_path_strs, num_workers)
        logger.info(f"Global range: [{global_min}, {global_max}]")
        output_dtype = np.dtype(np.uint8)
        _global_rescale = True
        _global_min = global_min
        _global_max = global_max
    else:
        _global_rescale = False

    # Volume shape (z, y, x)
    volume_shape = (num_z, height, width)
    logger.info(f"Volume shape: {volume_shape}")

    # Calculate number of XY tiles
    _, chunk_y, chunk_x = chunk_shape
    num_y_tiles = (height + chunk_y - 1) // chunk_y
    num_x_tiles = (width + chunk_x - 1) // chunk_x
    total_xy_tiles = num_y_tiles * num_x_tiles

    # Memory per tile
    bytes_per_tile = num_z * chunk_y * chunk_x * output_dtype.itemsize
    logger.info(f"XY tiles: {num_y_tiles} x {num_x_tiles} = {total_xy_tiles} tiles")
    logger.info(f"Memory per tile: {bytes_per_tile / 1024**2:.1f} MB")

    # Create output zarr
    logger.info(f"Creating OME-Zarr at {output_path}")
    if output_path.exists():
        import shutil
        logger.warning(f"Removing existing {output_path}")
        shutil.rmtree(output_path)

    root = create_group(output_path, overwrite=True, auth=None)

    # Create level 0 array
    arr_0 = root.create_array(
        "0",
        shape=volume_shape,
        chunks=chunk_shape,
        dtype=output_dtype,
        cname="zstd",
        level=compressor_level,
        shuffle=2,
        skip_empty=True,
    )

    # Write OME-Zarr metadata
    root.attrs["multiscales"] = [
        ome_multiscales(name="volume", levels=num_levels - 1, unit="micrometer", reduction="nearest")
    ]
    logger.info("Wrote OME-Zarr metadata")

    # Process level 0 using XY-tile-major order (much faster!)
    # Instead of iterating z-slices and reading each XY tile (opens same file many times),
    # we iterate XY tiles and read from all z-slices (opens each file once).
    logger.info(f"Processing level 0: {total_xy_tiles} XY tiles, {num_z} z-slices each, {num_workers} workers")
    logger.info("Using XY-tile-major processing (files opened once, not per-tile)")

    # Pre-generate tile coordinates
    tile_coords = []
    for y_start in range(0, height, chunk_y):
        y_end = min(y_start + chunk_y, height)
        for x_start in range(0, width, chunk_x):
            x_end = min(x_start + chunk_x, width)
            tile_coords.append((y_start, x_start, y_end, x_end))

    # Set up global references for worker function
    global _global_readers, _global_zarr_arr

    # Open all TIFF files once (not once per tile!)
    logger.info(f"Opening {num_z} TIFF files...")
    readers = [images.open(path, decoded_cache_bytes=0) for path in tiff_path_strs]
    try:
        _global_readers = readers
        _global_zarr_arr = arr_0

        # Generate tasks: one per XY tile position
        tasks = [
            (i, y_start, x_start, y_end, x_end, num_z)
            for i, (y_start, x_start, y_end, x_end) in enumerate(tile_coords)
        ]

        if num_workers > 1:
            # Process XY tiles in parallel
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(process_xy_tile, task): task for task in tasks}
                for future in tqdm(as_completed(futures), total=len(tasks), desc="XY tiles"):
                    future.result()  # Raises exception if worker failed
        else:
            for task in tqdm(tasks, desc="XY tiles"):
                process_xy_tile(task)

        _global_readers = None
        _global_zarr_arr = None
    finally:
        for reader in readers:
            reader.close()

    logger.info("Level 0 complete")

    # Generate pyramid levels
    if num_levels > 1:
        logger.info(f"Generating {num_levels - 1} pyramid levels")
        write_pyramid(output_path, num_levels - 1, reduction="nearest", threads=num_workers, overwrite=True)

    logger.info(f"Conversion complete: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert 2D TIFFs to OME-Zarr pyramid",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_folder",
        type=Path,
        help="Folder containing 2D TIFF files (one per z-slice)",
    )
    parser.add_argument(
        "output_path",
        type=Path,
        help="Output .zarr path",
    )
    parser.add_argument(
        "--num-levels",
        type=int,
        default=6,
        help="Number of pyramid levels",
    )
    parser.add_argument(
        "--chunk-size",
        type=parse_chunk_size,
        default="64,256,256",
        help="Chunk size: single int (e.g. 128) or z,y,x (e.g. 64,256,256)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of worker processes for parallel tile processing",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=3,
        choices=range(1, 10),
        help="Blosc compression level (1-9)",
    )

    args = parser.parse_args()

    convert_tifs_to_ome_zarr(
        input_folder=args.input_folder,
        output_path=args.output_path,
        num_levels=args.num_levels,
        chunk_shape=args.chunk_size,
        num_workers=args.workers,
        compressor_level=args.compression_level,
    )


if __name__ == "__main__":
    main()
