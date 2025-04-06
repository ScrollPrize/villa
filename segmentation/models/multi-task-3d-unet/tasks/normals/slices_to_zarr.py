import os
import numpy as np
import zarr
import tifffile
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from concurrent.futures import ProcessPoolExecutor
import warnings
import math


def read_tiff_slice(filepath, expected_shape=None, binary=False):
    """Read a single TIFF file, optionally converting to binary mask."""
    try:
        img = tifffile.imread(str(filepath))
        print(f"Read image shape: {img.shape}")
        if expected_shape and not binary and img.shape != expected_shape:
            raise ValueError(f"Inconsistent dimensions in {filepath}. Expected {expected_shape}, got {img.shape}")

        if binary:
            # Convert to binary mask: any nonzero pixel becomes 255
            binary_mask = np.any(img > 0, axis=-1).astype(np.uint8) * 255
            print(f"Converted to binary mask shape: {binary_mask.shape}")
            return binary_mask
        return img
    except Exception as e:
        warnings.warn(f"Error reading {filepath}: {str(e)}")
        return None


def process_z_chunk(file_chunk, expected_shape, z_start_idx, binary=False):
    """Process a chunk of Z slices and return as a single numpy array."""
    if binary:
        # For binary mode, remove channel dimension from expected shape
        chunk_shape = (len(file_chunk), expected_shape[0], expected_shape[1])
        chunk_data = np.zeros(chunk_shape, dtype=np.uint8)
    else:
        chunk_shape = (len(file_chunk),) + expected_shape
        chunk_data = np.zeros(chunk_shape, dtype=np.uint16)

    valid_slices = []

    for i, filepath in enumerate(file_chunk):
        img = read_tiff_slice(filepath, expected_shape, binary)
        if img is not None:
            chunk_data[i] = img
            valid_slices.append(i + z_start_idx)

    return chunk_data, valid_slices


def get_tiff_shape(filepath):
    """Get the shape of a TIFF file without loading it entirely into memory."""
    with tifffile.TiffFile(filepath) as tiff:
        return tiff.pages[0].shape


def convert_tiffs_to_zarr(input_dir, output_path, min_z=0, dtype=np.uint16, chunk_size=(128, 128, 128, 3),
                          max_workers=None, binary=False):
    """
    Convert a series of multichannel TIFF files to a Zarr volume using z-chunk aligned processing.

    Parameters:
    -----------
    input_dir : str
        Directory containing TIFF files named by their z-index
    output_path : str
        Path where the Zarr volume will be saved
    min_z : int
        Minimum z-slice number to include in the volume (default: 0)
    dtype : numpy.dtype
        Data type for the Zarr array (default: np.uint16)
    chunk_size : tuple
        Size of chunks for the Zarr array (default: (128, 128, 128, 3))
    max_workers : int
        Maximum number of worker processes (default: number of CPU cores)
    """
    if max_workers is None:
        max_workers = mp.cpu_count()

    input_dir = Path(input_dir)
    print("Discovering TIFF files...")
    tiff_files = sorted(
        input_dir.glob("*.tif*"),
        key=lambda x: int(x.stem)
    )

    if not tiff_files:
        raise ValueError("No TIFF files found in the input directory")

    # Get dimensions from first file without loading it
    first_shape = get_tiff_shape(tiff_files[0])
    if len(first_shape) != 3:
        raise ValueError(f"Expected 3D array (Y,X,C) for multichannel TIFF, got shape {first_shape}")

    height, width, n_channels = first_shape

    # Calculate volume dimensions
    z_indices = [int(f.stem) for f in tiff_files]
    max_z = max(z_indices)
    depth = max_z - min_z + 1

    # Create zarr array
    print(f"Creating zarr store at {output_path}")
    store = zarr.DirectoryStore(output_path)
    root = zarr.group(store=store, overwrite=True)
    print("Created zarr group")

    # Verify chunk size matches dimensions
    if len(chunk_size) != 4:
        raise ValueError("Chunk size must be 4D (Z,Y,X,C)")

    # Create dataset with appropriate shape and dtype for binary/multichannel
    if binary:
        final_shape = (depth, height, width)
        final_chunks = chunk_size[:3]  # Remove channel dimension from chunks
        final_dtype = np.uint8
        print(f"Creating binary Zarr array with shape: {final_shape}")
    else:
        final_shape = (depth, height, width, n_channels)
        final_chunks = chunk_size
        final_dtype = dtype
        print(f"Creating multichannel Zarr array with shape: {final_shape}")

    zarr_array = root.create_dataset(
        'volume',
        shape=final_shape,
        chunks=final_chunks,
        dtype=final_dtype
    )

    # Calculate number of z-chunks
    z_chunk_size = chunk_size[0]
    n_chunks = math.ceil(depth / z_chunk_size)

    # Process files in z-aligned chunks
    processed_count = 0
    total_slices = depth

    with tqdm(total=total_slices, desc="Processing Z chunks") as pbar:
        for chunk_idx in range(n_chunks):
            z_start = chunk_idx * z_chunk_size
            z_end = min((chunk_idx + 1) * z_chunk_size, depth)
            chunk_size_actual = z_end - z_start

            # Get files for this z-chunk
            chunk_z_indices = range(min_z + z_start, min_z + z_end)
            chunk_files = []
            for z in chunk_z_indices:
                matching_files = [f for f in tiff_files if int(f.stem) == z]
                if matching_files:
                    chunk_files.append(matching_files[0])
                else:
                    # Add None as placeholder for missing files
                    chunk_files.append(None)

            # Remove None entries and process valid files
            valid_files = [f for f in chunk_files if f is not None]
            if valid_files:
                # Process this z-chunk
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = []
                    # Split the chunk into smaller groups for parallel processing
                    group_size = max(1, len(valid_files) // max_workers)
                    for i in range(0, len(valid_files), group_size):
                        file_group = valid_files[i:i + group_size]
                        futures.append(
                            executor.submit(
                                process_z_chunk,
                                file_group,
                                (height, width, n_channels),
                                i,
                                binary
                            )
                        )

                    # Collect results and write to zarr
                    if binary:
                        chunk_data = np.zeros((chunk_size_actual, height, width), dtype=np.uint8)
                    else:
                        chunk_data = np.zeros((chunk_size_actual, height, width, n_channels), dtype=dtype)

                    for future in futures:
                        sub_chunk_data, valid_indices = future.result()
                        for idx, valid_idx in enumerate(valid_indices):
                            if valid_idx < chunk_size_actual:
                                chunk_data[valid_idx] = sub_chunk_data[idx]
                                processed_count += 1

                    # Write the entire z-chunk at once
                    print(f"Writing chunk with shape {chunk_data.shape} to zarr array at indices {z_start}:{z_end}")
                    print(f"Chunk data min: {chunk_data.min()}, max: {chunk_data.max()}")
                    zarr_array[z_start:z_end] = chunk_data
                    print(f"Successfully wrote chunk to zarr")

            pbar.update(chunk_size_actual)

    print(f"Successfully processed {processed_count} out of {total_slices} slices")

    # Save metadata
    root.attrs['n_channels'] = n_channels
    root.attrs['dimensions'] = ['z', 'y', 'x', 'c']
    root.attrs['min_z'] = min_z
    root.attrs['max_z'] = max_z
    root.attrs['processed_files'] = processed_count
    root.attrs['total_slices'] = total_slices
    root.attrs['chunk_size'] = chunk_size

    print(f"Conversion complete. Zarr volume shape: {zarr_array.shape}")
    return zarr_array


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert multichannel TIFF slices to Zarr volume")
    parser.add_argument("input_dir", help="Directory containing TIFF files")
    parser.add_argument("output_path", help="Output path for Zarr volume")
    parser.add_argument("--min-z", type=int, default=0, help="Minimum z-slice to include (default: 0)")
    parser.add_argument("--max-workers", type=int, default=None,
                        help="Maximum number of worker processes (default: number of CPU cores)")
    parser.add_argument("--binary", action="store_true",
                        help="Convert to binary mask (255 for any nonzero pixel)")
    args = parser.parse_args()

    convert_tiffs_to_zarr(
        args.input_dir,
        args.output_path,
        min_z=args.min_z,
        max_workers=args.max_workers,
        binary=args.binary
    )