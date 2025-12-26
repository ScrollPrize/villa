#!/usr/bin/env python3
"""
Remap values in an OME-Zarr array chunk-wise using multiprocessing.

Usage:
    python remap_ome_zarr.py /path/to/input.zarr /path/to/output.zarr
"""

import argparse
import multiprocessing as mp
from pathlib import Path

import fastremap
import zarr
from tqdm import tqdm

# =============================================================================
# VALUE REMAPPING DICTIONARY
# Format: original_value: remapped_value
# =============================================================================
REMAP_VALUES = {
    255: 1,
    100: 0,
    0: 2
    # Add more mappings as needed
}

# =============================================================================
# CONFIGURATION
# =============================================================================
NUM_WORKERS = mp.cpu_count()  # Number of parallel workers


def get_chunk_slices(shape: tuple, chunks: tuple) -> list[tuple[slice, ...]]:
    """Generate slice tuples for each chunk in the array."""
    slices_per_dim = []
    for dim_size, chunk_size in zip(shape, chunks):
        dim_slices = []
        for start in range(0, dim_size, chunk_size):
            end = min(start + chunk_size, dim_size)
            dim_slices.append(slice(start, end))
        slices_per_dim.append(dim_slices)

    # Generate all combinations of slices (cartesian product)
    chunk_slices = []

    def recurse(current_slices, dim_idx):
        if dim_idx == len(slices_per_dim):
            chunk_slices.append(tuple(current_slices))
            return
        for s in slices_per_dim[dim_idx]:
            recurse(current_slices + [s], dim_idx + 1)

    recurse([], 0)
    return chunk_slices


def remap_chunk(
    chunk_idx: int,
    chunk_slice: tuple[slice, ...],
    input_path: str,
    output_path: str,
    remap_dict: dict,
    resolution_level: int,
) -> int:
    """Process a single chunk: read, remap values, write."""
    # Open stores (each worker opens its own handle)
    input_store = zarr.open(input_path, mode="r")
    output_store = zarr.open(output_path, mode="r+")

    input_arr = input_store[str(resolution_level)]
    output_arr = output_store[str(resolution_level)]

    # Read chunk data
    data = input_arr[chunk_slice]

    # Apply remapping using fastremap (in-place, preserves unmapped values)
    remapped = fastremap.remap(data, remap_dict, preserve_missing_labels=True, in_place=False)

    # Write to output
    output_arr[chunk_slice] = remapped

    return chunk_idx


def remap_chunk_wrapper(args):
    """Wrapper for multiprocessing that unpacks arguments."""
    return remap_chunk(*args)


def main():
    parser = argparse.ArgumentParser(
        description="Remap values in an OME-Zarr array chunk-wise"
    )
    parser.add_argument("input_zarr", type=str, help="Path to input OME-Zarr")
    parser.add_argument("output_zarr", type=str, help="Path to output OME-Zarr")
    parser.add_argument(
        "--workers", type=int, default=NUM_WORKERS, help="Number of parallel workers"
    )
    parser.add_argument(
        "--level", type=int, default=None, help="Resolution level to process (default: all levels)"
    )
    args = parser.parse_args()

    input_path = Path(args.input_zarr)
    output_path = Path(args.output_zarr)

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Workers: {args.workers}")
    print(f"Remap values: {REMAP_VALUES}")

    # Open input to get metadata
    input_store = zarr.open(str(input_path), mode="r")

    # Determine which levels to process
    if args.level is not None:
        levels = [args.level]
    else:
        # Find all numeric keys (resolution levels)
        levels = sorted([int(k) for k in input_store.keys() if k.isdigit()])
        print(f"Found {len(levels)} resolution levels: {levels}")

    # Create output zarr
    if output_path.exists():
        print("Output already exists, opening in r+ mode")
        output_store = zarr.open(str(output_path), mode="r+")
    else:
        print("Creating output zarr")
        output_store = zarr.open(str(output_path), mode="w")

    # Process each level
    for level in levels:
        print(f"\n{'='*60}")
        print(f"Processing level {level}")
        print(f"{'='*60}")

        input_arr = input_store[str(level)]

        print(f"Array shape: {input_arr.shape}")
        print(f"Array chunks: {input_arr.chunks}")
        print(f"Array dtype: {input_arr.dtype}")

        # Create output array with same properties
        if str(level) not in output_store:
            output_store.create_dataset(
                str(level),
                shape=input_arr.shape,
                chunks=input_arr.chunks,
                dtype=input_arr.dtype,
                compressor=input_arr.compressor,
            )

        # Generate chunk slices
        chunk_slices = get_chunk_slices(input_arr.shape, input_arr.chunks)
        print(f"Total chunks to process: {len(chunk_slices)}")

        # Prepare arguments for each chunk (in order)
        task_args = [
            (idx, chunk_slice, str(input_path), str(output_path), REMAP_VALUES, level)
            for idx, chunk_slice in enumerate(chunk_slices)
        ]

        # Process chunks using multiprocessing
        with mp.Pool(processes=args.workers) as pool:
            # Use imap to maintain order and show progress
            results = list(
                tqdm(
                    pool.imap(remap_chunk_wrapper, task_args),
                    total=len(task_args),
                    desc=f"Level {level}",
                )
            )

        print(f"Completed processing {len(results)} chunks for level {level}")

    # Copy metadata if present
    if ".zattrs" in input_store:
        output_store.attrs.update(input_store.attrs)

    # Copy metadata.json if it exists
    metadata_src = input_path / "metadata.json"
    if metadata_src.exists():
        import shutil
        metadata_dst = output_path / "metadata.json"
        shutil.copy2(metadata_src, metadata_dst)
        print("Copied metadata.json")

    print("\nDone!")


if __name__ == "__main__":
    main()
