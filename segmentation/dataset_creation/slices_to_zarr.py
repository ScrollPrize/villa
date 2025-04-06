#!/usr/bin/env python
import os
import re
import argparse
import numpy as np
import imageio

import dask
from dask import delayed
import dask.array as da
from dask.distributed import Client, LocalCluster, progress

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Load image files (tif, tiff, jpg, png) from a folder into a 3D (or 4D) volume "
            "based on the slice number extracted from each filename. If slices are missing, "
            "the volume will be padded with blank (zero) slices so that the slice index matches "
            "the filename number. Finally, the volume is written to a Zarr store."
        )
    )
    parser.add_argument(
        "input_folder",
        help="Path to the folder containing image files."
    )
    parser.add_argument(
        "output_zarr",
        help="Path (or directory) where the Zarr store will be created."
    )
    parser.add_argument(
        "--chunks",
        type=int,
        default=None,
        help=(
            "Chunk size to be applied to the spatial dimensions. For a 3D volume (n_slices, H, W), "
            "this will result in chunks of shape (1, chunks, chunks). For a 4D volume (n_slices, H, W, C), "
            "the channel dimension is left intact (i.e. chunks of shape (1, chunks, chunks, -1))."
        )
    )
    parser.add_argument(
        "--memory-limit",
        type=str,
        default="auto",
        help="Maximum memory for each Dask worker (e.g., '2GB' or '1024MB')."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of Dask workers to launch. If not provided, Dask will use the default (usually one per CPU core)."
    )
    args = parser.parse_args()

    input_folder = args.input_folder
    output_zarr = args.output_zarr
    memory_limit = args.memory_limit
    num_workers = args.num_workers

    # Define valid file extensions (case-insensitive)
    valid_extensions = ('.tif', '.tiff', '.jpg', '.png')

    # List all files in the folder that have one of the valid extensions.
    files = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if os.path.isfile(os.path.join(input_folder, f)) and f.lower().endswith(valid_extensions)
    ]

    if not files:
        print(f"No valid image files found in folder: {input_folder}")
        return

    # Build a mapping from slice index to file.
    # Using a regex to capture the last occurrence of digits before the extension.
    index_to_file = {}
    for file in files:
        base = os.path.basename(file)
        m = re.search(r'(\d+)(?!.*\d)', base)
        if m:
            idx = int(m.group(1))
            if idx in index_to_file:
                print(f"Warning: duplicate slice index {idx} found in file {file}; ignoring duplicate.")
            else:
                index_to_file[idx] = file
        else:
            print(f"Warning: no numeric slice index found in file {file}; skipping.")

    if not index_to_file:
        print("No files with numeric slice indices were found; exiting.")
        return

    # Determine the volume range.
    max_index = max(index_to_file.keys())
    total_slices = max_index + 1
    print(f"Found slice indices: {sorted(index_to_file.keys())}")
    print(f"Volume will have {total_slices} slices (indices 0 to {max_index}).")

    # Load a sample file to determine the image shape and dtype.
    sample_index = sorted(index_to_file.keys())[0]
    sample_file = index_to_file[sample_index]
    sample = imageio.imread(sample_file)
    shape = sample.shape
    dtype = sample.dtype
    print(f"Sample image shape: {shape}, dtype: {dtype}")

    # Build a list of lazy arrays for each slice.
    lazy_arrays = []
    for i in range(total_slices):
        if i in index_to_file:
            file = index_to_file[i]
            # Create a delayed object to read the image.
            lazy_im = delayed(imageio.imread)(file)
        else:
            # For missing slices, create a delayed object that returns an array of zeros.
            lazy_im = delayed(np.zeros)(shape, dtype=dtype)
        # Wrap the delayed object as a Dask array with the proper shape and dtype.
        arr = da.from_delayed(lazy_im, shape=shape, dtype=dtype)
        lazy_arrays.append(arr)

    # Stack all slices along a new axis (the z-dimension).
    volume = da.stack(lazy_arrays, axis=0)
    print(f"Constructed volume shape: {volume.shape}")

    # Optionally rechunk the volume using a more memory-efficient scheme.
    if args.chunks is not None:
        chunk_size = args.chunks
        if len(shape) == 2:
            # For 2D images: volume shape is (n_slices, H, W)
            new_chunks = (1, chunk_size, chunk_size)
        elif len(shape) == 3:
            # For 3D images (e.g., RGB): volume shape is (n_slices, H, W, C)
            new_chunks = (1, chunk_size, chunk_size, -1)  # Leave the channel dimension unchunked
        else:
            # Fallback: uniform chunking
            new_chunks = (chunk_size,) * volume.ndim
        print(f"Rechunking volume to chunks: {new_chunks}")
        volume = volume.rechunk(new_chunks)

    # Start a local Dask distributed cluster.
    cluster = LocalCluster(n_workers=num_workers,
                           threads_per_worker=2,
                           memory_limit=memory_limit,
                           local_directory="/mnt/raid_nvme/dask_scratch")
    client = Client(cluster)
    print("Dask client created:", client)
    print("Dask dashboard available at:", client.dashboard_link)

    # Write the volume to a Zarr store.
    print(f"Writing volume to Zarr store at: {output_zarr}")
    # Using compute=False builds a lazy graph.
    write_graph = volume.to_zarr(output_zarr, overwrite=True, compute=False)
    future = client.compute(write_graph)
    progress(future)  # Shows a progress bar in the terminal.
    client.gather(future)

    print("Conversion complete.")

if __name__ == "__main__":
    main()
