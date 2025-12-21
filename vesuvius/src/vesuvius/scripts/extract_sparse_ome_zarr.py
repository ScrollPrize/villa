#!/usr/bin/env python3
"""Extract sparse OME-Zarr containing only chunks near tifxyz surfaces.

This tool creates a sparse copy of an OME-Zarr volume, including only
chunks that are within a specified distance (default 64 voxels) of
tifxyz surface points. This provides a tight fit around the surface,
not a simple bounding box.

Usage:
    python extract_sparse_ome_zarr.py source.zarr output.zarr \
        --tifxyz /path/to/surfaces_folder/ --margin 64

    python extract_sparse_ome_zarr.py source.zarr output.zarr \
        --tifxyz /path/to/surfaces_folder/ --margin 128 --workers 16
"""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import zarr
from numcodecs import Blosc
from tqdm import tqdm

from vesuvius.tifxyz import read_tifxyz

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def discover_tifxyz_dirs(folder: Path) -> List[Path]:
    """Find all tifxyz directories by looking for meta.json files.

    Parameters
    ----------
    folder : Path
        Root folder to search.

    Returns
    -------
    List[Path]
        List of directories containing meta.json (tifxyz directories).
    """
    tifxyz_dirs = [p.parent for p in folder.rglob("meta.json")]
    if not tifxyz_dirs:
        raise ValueError(f"No tifxyz directories found in {folder}")
    return sorted(tifxyz_dirs)


def load_surface_points(tifxyz_paths: List[Path]) -> Tuple[np.ndarray, List[str]]:
    """Load all valid surface points from tifxyz directories.

    Parameters
    ----------
    tifxyz_paths : List[Path]
        List of tifxyz directory paths.

    Returns
    -------
    Tuple[np.ndarray, List[str]]
        - Array of shape (N, 3) containing (x, y, z) voxel coordinates
        - List of surface names
    """
    all_points = []
    surface_names = []

    for path in tqdm(tifxyz_paths, desc="Loading surfaces"):
        surface = read_tifxyz(path)
        mask = surface._valid_mask
        points = np.column_stack([
            surface._x[mask],
            surface._y[mask],
            surface._z[mask]
        ])
        all_points.append(points)
        surface_names.append(path.name)
        logger.info(f"  {path.name}: {len(points):,} valid points")

    return np.vstack(all_points), surface_names


def compute_relevant_chunks(
    points: np.ndarray,
    chunk_size: Tuple[int, int, int],
    volume_shape: Tuple[int, int, int],
    margin_voxels: int = 64,
) -> Set[Tuple[int, int, int]]:
    """Compute chunk indices within margin of surface points.

    For each surface point, directly computes which chunks have any voxel
    within margin_voxels distance. This is exact - no conservative buffer.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (N, 3) containing (x, y, z) voxel coordinates.
    chunk_size : Tuple[int, int, int]
        Chunk size as (z, y, x).
    volume_shape : Tuple[int, int, int]
        Volume shape as (z, y, x).
    margin_voxels : int
        Distance in voxels from surface to include.

    Returns
    -------
    Set[Tuple[int, int, int]]
        Set of chunk indices (cz, cy, cx) that are within margin of surface.
    """
    cz_size, cy_size, cx_size = chunk_size

    # Max valid chunk indices
    max_cz = (volume_shape[0] - 1) // cz_size
    max_cy = (volume_shape[1] - 1) // cy_size
    max_cx = (volume_shape[2] - 1) // cx_size

    # Points are (x, y, z) - extract coordinates
    px = points[:, 0]  # x
    py = points[:, 1]  # y
    pz = points[:, 2]  # z

    # For each point, find the range of chunks it could affect
    # A point at (x,y,z) with margin m affects chunks where:
    #   chunk contains any voxel within distance m of the point
    #
    # Conservatively, this is chunks in range:
    #   z: floor((pz - m) / cz_size) to floor((pz + m) / cz_size)
    #   y: floor((py - m) / cy_size) to floor((py + m) / cy_size)
    #   x: floor((px - m) / cx_size) to floor((px + m) / cx_size)

    m = margin_voxels

    # Compute chunk ranges for all points at once
    cz_min = np.floor((pz - m) / cz_size).astype(np.int32)
    cz_max = np.floor((pz + m) / cz_size).astype(np.int32)
    cy_min = np.floor((py - m) / cy_size).astype(np.int32)
    cy_max = np.floor((py + m) / cy_size).astype(np.int32)
    cx_min = np.floor((px - m) / cx_size).astype(np.int32)
    cx_max = np.floor((px + m) / cx_size).astype(np.int32)

    # Clip to valid range
    cz_min = np.clip(cz_min, 0, max_cz)
    cz_max = np.clip(cz_max, 0, max_cz)
    cy_min = np.clip(cy_min, 0, max_cy)
    cy_max = np.clip(cy_max, 0, max_cy)
    cx_min = np.clip(cx_min, 0, max_cx)
    cx_max = np.clip(cx_max, 0, max_cx)

    logger.info(f"Computing affected chunks for {len(points):,} points with {m} voxel margin...")

    # Collect all affected chunks
    relevant_chunks: Set[Tuple[int, int, int]] = set()

    # Process in batches to show progress
    batch_size = 100000
    for start in tqdm(range(0, len(points), batch_size), desc="Processing points"):
        end = min(start + batch_size, len(points))
        for i in range(start, end):
            for cz in range(cz_min[i], cz_max[i] + 1):
                for cy in range(cy_min[i], cy_max[i] + 1):
                    for cx in range(cx_min[i], cx_max[i] + 1):
                        relevant_chunks.add((int(cz), int(cy), int(cx)))

    total_chunks = (max_cz + 1) * (max_cy + 1) * (max_cx + 1)
    logger.info(f"Found {len(relevant_chunks):,} relevant chunks (from {total_chunks:,} total)")
    return relevant_chunks


def compute_chunks_for_level(
    level0_chunks: Set[Tuple[int, int, int]],
    level: int,
) -> Set[Tuple[int, int, int]]:
    """Map level-0 chunks to corresponding chunks at lower resolution.

    Parameters
    ----------
    level0_chunks : Set[Tuple[int, int, int]]
        Set of level-0 chunk indices.
    level : int
        Target resolution level (1, 2, 3, ...).

    Returns
    -------
    Set[Tuple[int, int, int]]
        Set of chunk indices at the target level.
    """
    factor = 2 ** level
    level_chunks = set()
    for cz, cy, cx in level0_chunks:
        level_chunks.add((cz // factor, cy // factor, cx // factor))
    return level_chunks


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
                "method": "sparse extraction from source zarr",
            }
        }]
    }


def copy_chunk_worker(args: Tuple) -> Tuple[Tuple[int, int, int], bool]:
    """Copy a single chunk from source to destination.

    Parameters
    ----------
    args : Tuple
        (chunk_idx, source_path, dest_path, level, chunk_size, level_shape)

    Returns
    -------
    Tuple[Tuple[int, int, int], bool]
        (chunk_idx, success) where success indicates if chunk was non-empty.
    """
    chunk_idx, source_path, dest_path, level, chunk_size, level_shape = args
    cz, cy, cx = chunk_idx

    # Calculate bounds
    z_start = cz * chunk_size[0]
    y_start = cy * chunk_size[1]
    x_start = cx * chunk_size[2]
    z_end = min(z_start + chunk_size[0], level_shape[0])
    y_end = min(y_start + chunk_size[1], level_shape[1])
    x_end = min(x_start + chunk_size[2], level_shape[2])

    try:
        # Read from source
        source = zarr.open(source_path, mode="r")
        src_arr = source[str(level)]
        chunk_data = src_arr[z_start:z_end, y_start:y_end, x_start:x_end]

        # Skip empty chunks
        if chunk_data.max() == 0:
            return (chunk_idx, False)

        # Write to destination
        dest = zarr.open(dest_path, mode="r+")
        dest_arr = dest[str(level)]
        dest_arr[z_start:z_end, y_start:y_end, x_start:x_end] = chunk_data

        return (chunk_idx, True)
    except Exception as e:
        logger.warning(f"Failed to copy chunk {chunk_idx} at level {level}: {e}")
        return (chunk_idx, False)


def copy_sparse_chunks(
    source_path: str,
    dest_path: str,
    chunks_by_level: Dict[int, Set[Tuple[int, int, int]]],
    num_workers: int = 8,
) -> Dict[int, int]:
    """Copy relevant chunks from source to destination in parallel.

    Parameters
    ----------
    source_path : str
        Path to source OME-Zarr.
    dest_path : str
        Path to destination OME-Zarr.
    chunks_by_level : Dict[int, Set[Tuple[int, int, int]]]
        Chunks to copy for each level.
    num_workers : int
        Number of parallel workers.

    Returns
    -------
    Dict[int, int]
        Number of non-empty chunks copied per level.
    """
    source = zarr.open(source_path, mode="r")
    copied_counts = {}

    for level in sorted(chunks_by_level.keys()):
        chunks_to_copy = chunks_by_level[level]
        if not chunks_to_copy:
            copied_counts[level] = 0
            continue

        src_arr = source[str(level)]
        level_shape = src_arr.shape
        chunk_size = src_arr.chunks

        tasks = [
            (chunk_idx, source_path, dest_path, level, chunk_size, level_shape)
            for chunk_idx in chunks_to_copy
        ]

        logger.info(f"Level {level}: copying {len(tasks)} chunks")

        copied = 0
        if num_workers > 1:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(copy_chunk_worker, task): task for task in tasks}
                for future in tqdm(as_completed(futures), total=len(tasks), desc=f"Level {level}"):
                    _, success = future.result()
                    if success:
                        copied += 1
        else:
            for task in tqdm(tasks, desc=f"Level {level}"):
                _, success = copy_chunk_worker(task)
                if success:
                    copied += 1

        copied_counts[level] = copied
        logger.info(f"  Copied {copied} non-empty chunks")

    return copied_counts


def extract_sparse_ome_zarr(
    source_zarr: Path,
    output_zarr: Path,
    tifxyz_folder: Path,
    margin_voxels: int = 64,
    num_levels: int = 6,
    num_workers: int = 8,
) -> None:
    """Extract sparse OME-Zarr containing only chunks near tifxyz surfaces.

    Parameters
    ----------
    source_zarr : Path
        Path to source OME-Zarr.
    output_zarr : Path
        Path to output sparse OME-Zarr.
    tifxyz_folder : Path
        Folder containing tifxyz directories.
    margin_voxels : int
        Distance in voxels from surface to include.
    num_levels : int
        Number of pyramid levels.
    num_workers : int
        Number of parallel workers.
    """
    # Discover tifxyz directories
    logger.info(f"Discovering tifxyz directories in {tifxyz_folder}")
    tifxyz_dirs = discover_tifxyz_dirs(tifxyz_folder)
    logger.info(f"Found {len(tifxyz_dirs)} tifxyz directories")

    # Load surface points
    logger.info("Loading surface points...")
    points, surface_names = load_surface_points(tifxyz_dirs)
    logger.info(f"Loaded {len(points):,} total points from {len(surface_names)} surfaces")

    # Open source zarr to get metadata
    logger.info(f"Opening source zarr: {source_zarr}")
    source = zarr.open(str(source_zarr), mode="r")
    src_arr = source["0"]
    volume_shape = src_arr.shape
    chunk_size = src_arr.chunks
    dtype = src_arr.dtype
    compressor = src_arr.compressor

    logger.info(f"Source volume: shape={volume_shape}, chunks={chunk_size}, dtype={dtype}")

    # Compute relevant chunks at level 0
    logger.info(f"Computing relevant chunks (margin={margin_voxels} voxels)...")
    level0_chunks = compute_relevant_chunks(points, chunk_size, volume_shape, margin_voxels)

    # Compute chunks for all levels
    chunks_by_level: Dict[int, Set[Tuple[int, int, int]]] = {0: level0_chunks}
    for level in range(1, num_levels):
        chunks_by_level[level] = compute_chunks_for_level(level0_chunks, level)
        logger.info(f"Level {level}: {len(chunks_by_level[level])} chunks")

    # Create output zarr
    logger.info(f"Creating output zarr: {output_zarr}")
    if output_zarr.exists():
        import shutil
        logger.warning(f"Removing existing {output_zarr}")
        shutil.rmtree(output_zarr)

    root = zarr.open(str(output_zarr), mode="w")

    # Create arrays for all levels
    for level in range(num_levels):
        scale = 2 ** level
        level_shape = tuple((s + scale - 1) // scale for s in volume_shape)
        level_chunks = tuple(min(c, s) for c, s in zip(chunk_size, level_shape))

        root.create_dataset(
            str(level),
            shape=level_shape,
            chunks=level_chunks,
            dtype=dtype,
            compressor=compressor,
            fill_value=0,
            write_empty_chunks=False,
        )
        logger.info(f"Created level {level}: shape={level_shape}")

    # Write OME-Zarr metadata
    metadata = create_ome_zarr_metadata(volume_shape, num_levels, str(dtype))
    attrs_path = output_zarr / ".zattrs"
    with open(attrs_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Wrote OME-Zarr metadata")

    # Copy chunks
    logger.info("Copying chunks from source...")
    copied_counts = copy_sparse_chunks(
        str(source_zarr),
        str(output_zarr),
        chunks_by_level,
        num_workers,
    )

    # Write chunks manifest
    total_chunks = sum(len(chunks) for chunks in chunks_by_level.values())
    total_copied = sum(copied_counts.values())

    chunks_manifest = {
        "surfaces": surface_names,
        "margin_voxels": margin_voxels,
        "chunks_by_level": {
            str(level): [list(chunk) for chunk in sorted(chunks)]
            for level, chunks in chunks_by_level.items()
        },
        "total_chunks": total_chunks,
        "total_copied": total_copied,
        "source_zarr": str(source_zarr),
        "volume_shape": list(volume_shape),
        "chunk_size": list(chunk_size),
    }

    manifest_path = output_zarr / "chunks.json"
    with open(manifest_path, "w") as f:
        json.dump(chunks_manifest, f, indent=2)
    logger.info(f"Wrote chunks manifest: {manifest_path}")

    logger.info(f"Complete! Extracted {total_copied} non-empty chunks to {output_zarr}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract sparse OME-Zarr containing only chunks near tifxyz surfaces",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "source_zarr",
        type=Path,
        help="Path to source OME-Zarr volume",
    )
    parser.add_argument(
        "output_zarr",
        type=Path,
        help="Path to output sparse OME-Zarr",
    )
    parser.add_argument(
        "--tifxyz",
        type=Path,
        required=True,
        help="Folder containing one or more tifxyz directories",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=64,
        help="Distance in voxels from surface to include",
    )
    parser.add_argument(
        "--num-levels",
        type=int,
        default=6,
        help="Number of pyramid levels",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers",
    )

    args = parser.parse_args()

    extract_sparse_ome_zarr(
        source_zarr=args.source_zarr,
        output_zarr=args.output_zarr,
        tifxyz_folder=args.tifxyz,
        margin_voxels=args.margin,
        num_levels=args.num_levels,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    main()
