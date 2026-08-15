"""Shared utilities for zarr processing tasks.

Common functions used across multiple tasks including:
- Chunk coordinate generation
- OME-Zarr pyramid building
- Metadata writing
"""

from __future__ import annotations

import itertools
import json
import shutil
from multiprocessing import Pool
from pathlib import Path
from typing import List, Tuple

import numpy as np
import zarr
from numcodecs import Blosc
from tqdm import tqdm


def get_chunk_coords(
    shape: Tuple[int, ...], chunks: Tuple[int, ...]
) -> List[Tuple[Tuple[int, int], ...]]:
    """Generate coordinates for all chunks in the array.

    Args:
        shape: Array shape
        chunks: Chunk sizes

    Returns:
        List of chunk coordinates, where each coordinate is a tuple of
        (start, stop) pairs for each dimension
    """
    chunk_ranges = []
    for dim_size, chunk_size in zip(shape, chunks):
        ranges = []
        for start in range(0, dim_size, chunk_size):
            stop = min(start + chunk_size, dim_size)
            ranges.append((start, stop))
        chunk_ranges.append(ranges)

    return list(itertools.product(*chunk_ranges))


def get_chunk_slices(
    shape: Tuple[int, ...], chunks: Tuple[int, ...]
) -> List[Tuple[slice, ...]]:
    """Generate slice tuples for each chunk in the array.

    Args:
        shape: Array shape
        chunks: Chunk sizes

    Returns:
        List of slice tuples for each chunk
    """
    slices_per_dim = []
    for dim_size, chunk_size in zip(shape, chunks):
        dim_slices = []
        for start in range(0, dim_size, chunk_size):
            end = min(start + chunk_size, dim_size)
            dim_slices.append(slice(start, end))
        slices_per_dim.append(dim_slices)

    chunk_slices = []

    def recurse(current_slices, dim_idx):
        if dim_idx == len(slices_per_dim):
            chunk_slices.append(tuple(current_slices))
            return
        for s in slices_per_dim[dim_idx]:
            recurse(current_slices + [s], dim_idx + 1)

    recurse([], 0)
    return chunk_slices


def create_level_dataset(
    root_group_path: str,
    level_name: str,
    shape: Tuple[int, ...],
    chunks: Tuple[int, ...],
    dtype: np.dtype,
    compressor,
    overwrite: bool = True,
) -> zarr.Array:
    """Create a dataset using NestedDirectoryStore for nested chunk directories.

    Args:
        root_group_path: Path to the root zarr group
        level_name: Name of the level (e.g., "0", "1")
        shape: Array shape
        chunks: Chunk sizes
        dtype: Data type
        compressor: Compressor instance
        overwrite: Whether to overwrite existing level

    Returns:
        Created zarr array
    """
    root_path = Path(root_group_path)
    root_path.mkdir(parents=True, exist_ok=True)
    zgroup_path = root_path / ".zgroup"
    if not zgroup_path.exists():
        with open(zgroup_path, "w") as f:
            json.dump({"zarr_format": 2}, f, indent=4)

    level_path = root_path / level_name
    if overwrite and level_path.exists():
        shutil.rmtree(level_path)

    store = zarr.NestedDirectoryStore(str(level_path))
    return zarr.open(
        store=store,
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        compressor=compressor,
        mode="w",
        write_empty_chunks=False,
        fill_value=0,
    )


def write_multiscales_metadata(
    root_group_path: str, axes_names: List[str], num_levels: int
) -> None:
    """Write OME-Zarr multiscales metadata to .zattrs.

    Args:
        root_group_path: Path to the root zarr group
        axes_names: List of axis names (e.g., ["z", "y", "x"])
        num_levels: Number of pyramid levels
    """
    root = zarr.open_group(root_group_path, mode="a")
    axes = []
    for a in axes_names:
        axes.append(
            {"name": a, "type": "space" if a in ("x", "y", "z") else "unknown"}
        )
    datasets = [{"path": str(i)} for i in range(num_levels)]
    root.attrs["multiscales"] = [
        {
            "version": "0.4",
            "name": "image",
            "axes": axes,
            "datasets": datasets,
        }
    ]


def _scale_per_axis_worker(args):
    """Multiprocessing worker for per-axis scaling chunks."""
    input_path, output_path, scales, out_chunk_coords = args
    in_z = zarr.open(input_path, mode="r")
    out_z = zarr.open(output_path, mode="r+")

    out_slices = tuple(slice(start, stop) for start, stop in out_chunk_coords)

    in_slices = []
    for (o_start, o_stop), dim in zip(out_chunk_coords, range(in_z.ndim)):
        s = float(scales[dim])
        in_start = int(np.floor(o_start / s))
        in_stop = int(np.ceil(o_stop / s))
        in_start = max(0, in_start)
        in_stop = min(in_z.shape[dim], max(in_start + 1, in_stop))
        in_slices.append(slice(in_start, in_stop))

    in_block = in_z[tuple(in_slices)]

    idx_list = []
    for ax, (o_start, o_stop), s in zip(
        range(in_z.ndim), out_chunk_coords, scales
    ):
        o_coords = np.arange(o_start, o_stop, dtype=np.float64)
        in_idx = (
            np.floor(o_coords / float(s)).astype(np.int64) - in_slices[ax].start
        )
        in_idx[in_idx < 0] = 0
        max_valid = in_block.shape[ax] - 1
        if max_valid >= 0:
            in_idx[in_idx > max_valid] = max_valid
        idx_list.append(in_idx)

    ix = np.ix_(*idx_list)
    out_block = in_block[ix]

    out_z[out_slices] = out_block

    return out_chunk_coords


def build_pyramid(
    base_group_path: str,
    axes_names: List[str],
    num_levels: int,
    num_workers: int,
) -> None:
    """Build pyramid levels 1..num_levels-1 from level 0.

    Uses nearest-neighbor downsampling with uniform 2x downsample per level.

    Args:
        base_group_path: Path to the base zarr group
        axes_names: List of axis names
        num_levels: Total number of levels to build
        num_workers: Number of worker processes
    """
    per_axis_scales = [0.5] * len(axes_names)

    for level in range(1, num_levels):
        in_path = f"{base_group_path}/{level - 1}"
        out_path = f"{base_group_path}/{level}"

        in_z = zarr.open(in_path, mode="r")
        out_shape = tuple(
            max(1, int(round(s * dim)))
            for dim, s in zip(in_z.shape, per_axis_scales)
        )
        out_chunks = tuple(min(c, s) for c, s in zip(in_z.chunks, out_shape))

        create_level_dataset(
            base_group_path,
            str(level),
            out_shape,
            out_chunks,
            in_z.dtype,
            in_z.compressor,
        )

        out_chunk_coords = get_chunk_coords(out_shape, out_chunks)

        work_items = [
            (in_path, out_path, per_axis_scales, coords)
            for coords in out_chunk_coords
        ]

        with Pool(processes=num_workers) as pool:
            for _ in tqdm(
                pool.imap_unordered(_scale_per_axis_worker, work_items),
                total=len(work_items),
                desc=f"Building level {level}",
            ):
                pass


def inverse_permutation(perm: List[int]) -> List[int]:
    """Compute the inverse of a permutation.

    Args:
        perm: Permutation as a list of indices

    Returns:
        Inverse permutation
    """
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return inv


def delete_chunk_file(
    zarr_path: str,
    chunk_coords: Tuple[Tuple[int, int], ...],
    chunks: Tuple[int, ...],
) -> None:
    """Delete the chunk file for given coordinates (nested directory structure).

    Args:
        zarr_path: Path to zarr array
        chunk_coords: Chunk coordinates as (start, stop) pairs
        chunks: Chunk sizes
    """
    chunk_indices = tuple(
        start // chunk_size
        for (start, _), chunk_size in zip(chunk_coords, chunks)
    )
    chunk_path = Path(zarr_path) / "/".join(str(i) for i in chunk_indices)
    if chunk_path.exists():
        chunk_path.unlink()


def update_compressor_metadata(zarr_path: str, compressor) -> None:
    """Update the compressor in .zarray metadata file.

    Args:
        zarr_path: Path to zarr array
        compressor: Compressor instance
    """
    zarray_path = Path(zarr_path) / ".zarray"
    with open(zarray_path, "r") as f:
        metadata = json.load(f)
    metadata["compressor"] = compressor.get_config()
    with open(zarray_path, "w") as f:
        json.dump(metadata, f, indent=4)


def confirm_overwrite(path: Path) -> bool:
    """Prompt user to confirm overwriting existing path.

    Args:
        path: Path to check

    Returns:
        True if user confirms or path doesn't exist, False otherwise
    """
    if path.exists():
        response = input(f"Output path {path} already exists. Overwrite? (y/n): ")
        return response.lower() == "y"
    return True


def get_default_compressor(level: int = 1) -> Blosc:
    """Get a default zstd compressor.

    Args:
        level: Compression level (1-9)

    Returns:
        Blosc compressor instance
    """
    return Blosc(cname="zstd", clevel=level, shuffle=Blosc.BITSHUFFLE)
