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
from threading import Lock
from typing import List, Optional, Tuple

import numpy as np
import tifffile
import zarr
from numcodecs import Blosc
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


def get_tiff_info(tiff_path: Path) -> Tuple[Tuple[int, int], np.dtype, bool, Tuple[int, int]]:
    """Get shape, dtype, tiled status, and tile shape from a TIFF."""
    with tifffile.TiffFile(str(tiff_path)) as tif:
        page = tif.pages[0]
        shape = page.shape
        dtype = page.dtype
        is_tiled = page.is_tiled
        tile_shape = page.chunks if is_tiled else (min(256, shape[0]), min(256, shape[1]))
    return shape, dtype, is_tiled, tile_shape


class TiffHandlePool:
    """Thread-safe pool of open TIFF file handles.

    Keeps all TIFF files open throughout processing to avoid the massive overhead
    of opening/closing files for each tile read. Each file has its own lock to
    allow parallel reads from different files while preventing concurrent access
    to the same file handle.
    """

    def __init__(self, paths: List[str]):
        """Open all TIFF files and create per-file locks.

        Parameters
        ----------
        paths : List[str]
            List of TIFF file paths (as strings).
        """
        self.handles: List[tifffile.TiffFile] = []
        self.locks: List[Lock] = []
        self.pages: List = []  # Cache first page for each file

        for path in paths:
            tif = tifffile.TiffFile(path)
            self.handles.append(tif)
            self.locks.append(Lock())
            self.pages.append(tif.pages[0])

    def close(self):
        """Close all open file handles."""
        for handle in self.handles:
            handle.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def read_tile(
        self,
        z_idx: int,
        y_start: int,
        y_end: int,
        x_start: int,
        x_end: int,
    ) -> np.ndarray:
        """Read a tile from a specific z-slice (TIFF file).

        Thread-safe: uses per-file locking to prevent concurrent access.

        Parameters
        ----------
        z_idx : int
            Z-slice index (which TIFF file).
        y_start, y_end : int
            Y (row) range to read.
        x_start, x_end : int
            X (column) range to read.

        Returns
        -------
        np.ndarray
            2D array of shape (y_end - y_start, x_end - x_start).
        """
        with self.locks[z_idx]:
            tif = self.handles[z_idx]
            page = self.pages[z_idx]
            img_height, img_width = page.shape[:2]

            # Clamp to image bounds
            y_end = min(y_end, img_height)
            x_end = min(x_end, img_width)

            if not page.is_tiled:
                # Non-tiled: must read full page and slice
                full = page.asarray()
                return full[y_start:y_end, x_start:x_end]

            # Tiled reading: decode only needed tiles
            tile_height, tile_width = page.chunks[:2]
            dtype = page.dtype

            # Calculate which tiles we need
            tile_row_start = y_start // tile_height
            tile_row_end = (y_end + tile_height - 1) // tile_height
            tile_col_start = x_start // tile_width
            tile_col_end = (x_end + tile_width - 1) // tile_width

            # Number of tile columns in the image
            tiles_per_row = (img_width + tile_width - 1) // tile_width

            # Output array
            out_height = y_end - y_start
            out_width = x_end - x_start
            result = np.zeros((out_height, out_width), dtype=dtype)

            # Get tile data
            offsets = page.dataoffsets
            bytecounts = page.databytecounts
            fh = tif.filehandle

            for tile_row in range(tile_row_start, tile_row_end):
                for tile_col in range(tile_col_start, tile_col_end):
                    tile_idx = tile_row * tiles_per_row + tile_col

                    if tile_idx >= len(offsets):
                        continue

                    # Read and decode tile
                    fh.seek(offsets[tile_idx])
                    raw = fh.read(bytecounts[tile_idx])
                    tile_data = page.decode(raw, tile_idx)[0]
                    tile_data = np.squeeze(tile_data)

                    # Calculate tile position in image coordinates
                    tile_y = tile_row * tile_height
                    tile_x = tile_col * tile_width

                    # Calculate overlap between tile and requested region
                    src_y_start = max(0, y_start - tile_y)
                    src_y_end = min(tile_data.shape[0], y_end - tile_y)
                    src_x_start = max(0, x_start - tile_x)
                    src_x_end = min(tile_data.shape[1], x_end - tile_x)

                    # Calculate destination in output array
                    dst_y_start = tile_y + src_y_start - y_start
                    dst_y_end = tile_y + src_y_end - y_start
                    dst_x_start = tile_x + src_x_start - x_start
                    dst_x_end = tile_x + src_x_end - x_start

                    result[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                        tile_data[src_y_start:src_y_end, src_x_start:src_x_end]

            return result


def read_tiff_region(
    tiff_path: str,
    y_start: int,
    y_end: int,
    x_start: int,
    x_end: int,
) -> np.ndarray:
    """Read a region from a TIFF. Opens file, reads region, closes file.

    For tiled TIFFs, reads only the tiles that overlap with the requested region.
    For non-tiled TIFFs, reads full image and slices (unavoidable).
    """
    with tifffile.TiffFile(tiff_path) as tif:
        page = tif.pages[0]
        img_height, img_width = page.shape[:2]

        # Clamp to image bounds
        y_end = min(y_end, img_height)
        x_end = min(x_end, img_width)

        if not page.is_tiled:
            # Non-tiled: must read full page and slice
            full = page.asarray()
            return full[y_start:y_end, x_start:x_end]

        # Tiled reading: decode only needed tiles
        tile_height, tile_width = page.chunks[:2]
        dtype = page.dtype

        # Calculate which tiles we need
        tile_row_start = y_start // tile_height
        tile_row_end = (y_end + tile_height - 1) // tile_height
        tile_col_start = x_start // tile_width
        tile_col_end = (x_end + tile_width - 1) // tile_width

        # Number of tile columns in the image
        tiles_per_row = (img_width + tile_width - 1) // tile_width

        # Output array
        out_height = y_end - y_start
        out_width = x_end - x_start
        result = np.zeros((out_height, out_width), dtype=dtype)

        # Get tile data
        offsets = page.dataoffsets
        bytecounts = page.databytecounts
        fh = tif.filehandle

        for tile_row in range(tile_row_start, tile_row_end):
            for tile_col in range(tile_col_start, tile_col_end):
                tile_idx = tile_row * tiles_per_row + tile_col

                if tile_idx >= len(offsets):
                    continue

                # Read and decode tile
                fh.seek(offsets[tile_idx])
                raw = fh.read(bytecounts[tile_idx])
                tile_data = page.decode(raw, tile_idx)[0]

                # tile_data shape is typically (1, tile_h, tile_w, 1) or (tile_h, tile_w)
                tile_data = np.squeeze(tile_data)

                # Calculate tile position in image coordinates
                tile_y = tile_row * tile_height
                tile_x = tile_col * tile_width

                # Calculate overlap between tile and requested region
                src_y_start = max(0, y_start - tile_y)
                src_y_end = min(tile_data.shape[0], y_end - tile_y)
                src_x_start = max(0, x_start - tile_x)
                src_x_end = min(tile_data.shape[1], x_end - tile_x)

                # Calculate destination in output array
                dst_y_start = tile_y + src_y_start - y_start
                dst_y_end = tile_y + src_y_end - y_start
                dst_x_start = tile_x + src_x_start - x_start
                dst_x_end = tile_x + src_x_end - x_start

                result[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                    tile_data[src_y_start:src_y_end, src_x_start:src_x_end]

        return result


def read_tile_from_tiff(args: Tuple) -> Tuple[int, int, np.ndarray]:
    """Read one XY tile from a single TIFF.

    Called by ThreadPoolExecutor to parallelize tile reads from one TIFF.

    Parameters
    ----------
    args : Tuple
        (y_start, x_start, tiff_path, y_end, x_end)

    Returns
    -------
    Tuple[int, int, np.ndarray]
        (y_start, x_start, tile_data) where tile_data has shape (tile_h, tile_w).
    """
    y_start, x_start, tiff_path, y_end, x_end = args
    tile = read_tiff_region(tiff_path, y_start, y_end, x_start, x_end)
    return (y_start, x_start, tile)


def convert_to_tiled_tiff(tiff_path: str, tile_size: int = 256) -> None:
    """Convert a non-tiled TIFF to tiled format in place."""
    with tifffile.TiffFile(tiff_path) as tif:
        page = tif.pages[0]
        if page.is_tiled:
            return  # Already tiled
        data = page.asarray()
        dtype = page.dtype

    # Write tiled version to temp file, then replace
    temp_path = tiff_path + ".tiled.tmp"
    tifffile.imwrite(
        temp_path,
        data,
        tile=(tile_size, tile_size),
        compression="zstd",
        compressionargs={"level": 3},
    )

    # Replace original with tiled version
    import os
    os.replace(temp_path, tiff_path)


def convert_tiffs_to_tiled(tiff_paths: List[str], num_workers: int, tile_size: int = 256) -> int:
    """Convert all non-tiled TIFFs to tiled format in parallel. Returns count converted."""
    # First check which ones need conversion
    non_tiled = []
    for p in tiff_paths:
        with tifffile.TiffFile(p) as tif:
            if not tif.pages[0].is_tiled:
                non_tiled.append(p)

    if not non_tiled:
        return 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(convert_to_tiled_tiff, p, tile_size) for p in non_tiled]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Converting to tiled"):
            future.result()

    return len(non_tiled)


def scan_tiff_minmax(tiff_path: str) -> Tuple[float, float]:
    """Scan a single TIFF to find min/max values."""
    with tifffile.TiffFile(tiff_path) as tif:
        data = tif.pages[0].asarray()
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
_global_pool: Optional[TiffHandlePool] = None
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
        tile_2d = _global_pool.read_tile(z, y_start, y_end, x_start, x_end)
        slices.append(tile_2d)

    # Stack into 3D chunk (z, y, x)
    chunk_3d = np.stack(slices, axis=0)

    # Apply uint16→uint8 rescaling if needed
    if _global_rescale:
        chunk_3d = rescale_to_uint8(chunk_3d, _global_min, _global_max)

    # Write to zarr (complete chunk in one operation)
    _global_zarr_arr[:, y_start:y_start + chunk_3d.shape[1], x_start:x_start + chunk_3d.shape[2]] = chunk_3d

    return (y_start, x_start, chunk_3d.shape[1], chunk_3d.shape[2])


def create_ome_zarr_metadata(
    shape: Tuple[int, int, int],
    num_levels: int,
    dtype: str,
    name: str = "volume",
) -> dict:
    """Create OME-Zarr multiscales metadata."""
    datasets = []
    for level in range(num_levels):
        scale = 2 ** level
        datasets.append({
            "path": str(level),
            "coordinateTransformations": [
                {
                    "type": "scale",
                    "scale": [float(scale), float(scale), float(scale)]
                }
            ]
        })

    return {
        "multiscales": [{
            "version": "0.4",
            "name": name,
            "axes": [
                {"name": "z", "type": "space", "unit": "micrometer"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"},
            ],
            "datasets": datasets,
            "type": "gaussian",
            "metadata": {
                "method": "strided 2x downsampling",
            }
        }]
    }


def downsample_chunk(args: Tuple) -> Optional[Tuple[Tuple[int, int, int], np.ndarray]]:
    """Downsample a single chunk from the previous level."""
    out_chunk_idx, zarr_path, prev_level, prev_shape, out_shape, out_chunks = args
    cz, cy, cx = out_chunk_idx

    # Calculate output chunk bounds
    z_out_start = cz * out_chunks[0]
    y_out_start = cy * out_chunks[1]
    x_out_start = cx * out_chunks[2]
    z_out_end = min(z_out_start + out_chunks[0], out_shape[0])
    y_out_end = min(y_out_start + out_chunks[1], out_shape[1])
    x_out_end = min(x_out_start + out_chunks[2], out_shape[2])

    # Calculate corresponding input bounds (2x)
    z_in_start = z_out_start * 2
    y_in_start = y_out_start * 2
    x_in_start = x_out_start * 2
    z_in_end = min(z_out_end * 2, prev_shape[0])
    y_in_end = min(y_out_end * 2, prev_shape[1])
    x_in_end = min(x_out_end * 2, prev_shape[2])

    # Read only the required input region
    root = zarr.open(zarr_path, mode="r")
    prev_arr = root[str(prev_level)]
    input_chunk = prev_arr[z_in_start:z_in_end, y_in_start:y_in_end, x_in_start:x_in_end]

    # Skip empty chunks
    if input_chunk.max() == 0:
        return None

    # Output dimensions for this chunk
    out_z = z_out_end - z_out_start
    out_y = y_out_end - y_out_start
    out_x = x_out_end - x_out_start

    # Downsample: take every other voxel (strided 2x)
    downsampled = input_chunk[::2, ::2, ::2]

    # Ensure correct output size (handle edge cases)
    if downsampled.shape != (out_z, out_y, out_x):
        result = np.zeros((out_z, out_y, out_x), dtype=input_chunk.dtype)
        sz = min(downsampled.shape[0], out_z)
        sy = min(downsampled.shape[1], out_y)
        sx = min(downsampled.shape[2], out_x)
        result[:sz, :sy, :sx] = downsampled[:sz, :sy, :sx]
        downsampled = result

    return (out_chunk_idx, downsampled)


def generate_pyramid_levels(
    zarr_path: str,
    num_levels: int,
    chunks: Tuple[int, int, int],
    compressor,
    dtype: np.dtype,
    num_workers: int = 1,
) -> None:
    """Generate downsampled pyramid levels from level 0."""
    root = zarr.open(zarr_path, mode="r+")

    for level in range(1, num_levels):
        prev_level = level - 1
        prev_arr = root[str(prev_level)]
        prev_shape = prev_arr.shape

        # Output shape is halved
        out_shape = (
            (prev_shape[0] + 1) // 2,
            (prev_shape[1] + 1) // 2,
            (prev_shape[2] + 1) // 2,
        )

        logger.info(f"Level {level}: {prev_shape} -> {out_shape}")

        # Create output array
        out_arr = root.create_dataset(
            str(level),
            shape=out_shape,
            chunks=chunks,
            dtype=dtype,
            compressor=compressor,
            overwrite=True,
        )

        # Generate chunk tasks
        tasks = []
        for cz in range((out_shape[0] + chunks[0] - 1) // chunks[0]):
            for cy in range((out_shape[1] + chunks[1] - 1) // chunks[1]):
                for cx in range((out_shape[2] + chunks[2] - 1) // chunks[2]):
                    tasks.append((
                        (cz, cy, cx),
                        zarr_path,
                        prev_level,
                        prev_shape,
                        out_shape,
                        chunks,
                    ))

        # Process chunks in parallel
        if num_workers > 1:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(downsample_chunk, task): task for task in tasks}
                for future in tqdm(as_completed(futures), total=len(tasks), desc=f"Level {level}"):
                    result = future.result()
                    if result is not None:
                        (cz, cy, cx), data = result
                        z_start = cz * chunks[0]
                        y_start = cy * chunks[1]
                        x_start = cx * chunks[2]
                        out_arr[
                            z_start:z_start + data.shape[0],
                            y_start:y_start + data.shape[1],
                            x_start:x_start + data.shape[2],
                        ] = data
        else:
            for task in tqdm(tasks, desc=f"Level {level}"):
                result = downsample_chunk(task)
                if result is not None:
                    (cz, cy, cx), data = result
                    z_start = cz * chunks[0]
                    y_start = cy * chunks[1]
                    x_start = cx * chunks[2]
                    out_arr[
                        z_start:z_start + data.shape[0],
                        y_start:y_start + data.shape[1],
                        x_start:x_start + data.shape[2],
                    ] = data


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
    shape_2d, dtype, is_tiled, tile_shape = get_tiff_info(tiff_paths[0])
    height, width = shape_2d
    logger.info(f"TIFF shape: {shape_2d}, dtype: {dtype}, tiled: {is_tiled}, tile_shape: {tile_shape}")

    # Convert any non-tiled TIFFs to tiled format for efficient random access
    logger.info("Checking all TIFFs for tiled format...")
    num_converted = convert_tiffs_to_tiled(tiff_path_strs, num_workers)
    if num_converted > 0:
        logger.info(f"Converted {num_converted} non-tiled TIFFs to tiled format")
        # Re-read info after conversion
        _, _, is_tiled, tile_shape = get_tiff_info(tiff_paths[0])
        logger.info(f"Now tiled: {is_tiled}, tile_shape: {tile_shape}")
    else:
        logger.info("All TIFFs already tiled")

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

    # Create compressor
    compressor = Blosc(cname="zstd", clevel=compressor_level, shuffle=Blosc.BITSHUFFLE)

    # Create output zarr
    logger.info(f"Creating OME-Zarr at {output_path}")
    if output_path.exists():
        import shutil
        logger.warning(f"Removing existing {output_path}")
        shutil.rmtree(output_path)

    root = zarr.open(str(output_path), mode="w")

    # Create level 0 array
    arr_0 = root.create_dataset(
        "0",
        shape=volume_shape,
        chunks=chunk_shape,
        dtype=output_dtype,
        compressor=compressor,
    )

    # Write OME-Zarr metadata
    metadata = create_ome_zarr_metadata(volume_shape, num_levels, str(output_dtype))
    attrs_path = output_path / ".zattrs"
    with open(attrs_path, "w") as f:
        json.dump(metadata, f, indent=2)
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
    global _global_pool, _global_zarr_arr

    # Open all TIFF files once (not once per tile!)
    logger.info(f"Opening {num_z} TIFF files...")
    with TiffHandlePool(tiff_path_strs) as pool:
        _global_pool = pool
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

        _global_pool = None
        _global_zarr_arr = None

    logger.info("Level 0 complete")

    # Generate pyramid levels
    if num_levels > 1:
        logger.info(f"Generating {num_levels - 1} pyramid levels")
        generate_pyramid_levels(
            str(output_path),
            num_levels,
            chunk_shape,
            compressor,
            output_dtype,
            num_workers,
        )

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
