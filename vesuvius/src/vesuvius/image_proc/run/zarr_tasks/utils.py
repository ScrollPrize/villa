"""Shared utilities for zarr processing tasks.

Common functions used across multiple tasks including:
- Chunk coordinate generation
- Opening arrays in workers
- Finishing an OME-Zarr output
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import List, Tuple

from vc_reader.reader import ZarrReader, ZarrWriter, create_group, ome_multiscales, open_zarr, write_pyramid


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


def open_input(path: str) -> ZarrReader:
    """An array to read in a pool worker: the pool already reads chunks in parallel."""
    return open_zarr(path, threads=1)


def open_output(path: str, *, skip_empty: bool = True) -> ZarrWriter:
    """An array to write in a pool worker; chunks holding only the fill value stay off disk."""
    return ZarrWriter.open(path, skip_empty=skip_empty, auth=None)


def create_level(root_group_path: str, level_name: str, like: ZarrReader, **overrides) -> ZarrWriter:
    """Replace a level with an empty one like ``like``: nested chunk keys, fill 0."""
    return ZarrWriter.create_like(
        f"{root_group_path}/{level_name}", like, fill_value=0, separator="/", overwrite=True, auth=None, **overrides
    )


def finish_pyramid(
    root_group_path: str, axes_names: List[str], num_levels: int, num_workers: int
) -> None:
    """Build levels 1..num_levels-1 from level 0 (uniform 2x, nearest) and write the multiscales metadata."""
    levels = num_levels - 1
    write_pyramid(root_group_path, levels, reduction="nearest", threads=num_workers, overwrite=True)
    create_group(root_group_path, auth=None).attrs["multiscales"] = [
        ome_multiscales(levels=levels, reduction="nearest", axes=axes_names)
    ]


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
