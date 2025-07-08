import webknossos as wk
import zarr
import numpy as np
from tqdm import tqdm
from numcodecs import Blosc
import math
import os
import time
import argparse
import re


def extract_uuid_from_url(url):
    """Extract UUID from URL pattern like .../UUID/surface_volume/1 or .../UUID/ink_labels/1"""
    # Match pattern for UUID followed by either surface_volume or ink_labels
    pattern = r'/([^/]+)/(surface_volume|ink_labels)/\d+/?$'
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"Could not extract UUID from URL: {url}")


def download_zarr_with_conversion(remote_url, local_path, fetch_chunk_size=(64, 512, 512)):
    if os.path.exists(local_path):
        print(f"Output already exists at {local_path}, skipping download")
        return

    print(f"downloading {remote_url}")

    remote_store = zarr.storage.FSStore(remote_url)
    remote_array = zarr.open_array(store=remote_store, mode='r')
    orig_shape = remote_array.shape
    print(f"Original shape (batch, x, y, z): {orig_shape}")

    batch_size, x_size, y_size, z_size = orig_shape
    print(f"remote dtype {remote_array.dtype}")
    z_size_capped = min(z_size, 64)

    def pad_to_multiple(size, multiple=64):
        return math.ceil(size / multiple) * multiple

    z_padded = pad_to_multiple(z_size_capped)
    y_padded = pad_to_multiple(y_size)
    x_padded = pad_to_multiple(x_size)

    print(f"Capped z size: {z_size_capped}")
    print(f"Padded shape (z, y, x): ({z_padded}, {y_padded}, {x_padded})")

    compressor = Blosc(cname='blosclz', clevel=9)

    local_array = zarr.open_array(
        local_path,
        mode='w',
        shape=(z_padded, y_padded, x_padded),
        chunks=(64, 64, 64),
        dtype=remote_array.dtype,
        compressor=compressor
    )

    print(f"Created local array with shape {local_array.shape}, chunks {local_array.chunks}")
    print(f"Fetching in chunks of {fetch_chunk_size} (z, y, x)")

    def copy_and_transform_chunk(chunk_coords, max_retries=5):
        z_start, y_start, x_start = chunk_coords

        for attempt in range(max_retries):
            time.sleep(.1)
            try:
                z_fetch_size, y_fetch_size, x_fetch_size = fetch_chunk_size

                z_end = min(z_start + z_fetch_size, z_padded)
                y_end = min(y_start + y_fetch_size, y_padded)
                x_end = min(x_start + x_fetch_size, x_padded)

                z_data_end = min(z_end, z_size_capped)
                y_data_end = min(y_end, y_size)
                x_data_end = min(x_end, x_size)

                chunk_data = np.zeros((z_end - z_start, y_end - y_start, x_end - x_start),
                                      dtype=remote_array.dtype)

                if z_start < z_size_capped and y_start < y_size and x_start < x_size:
                    z_copy_size = z_data_end - z_start
                    y_copy_size = y_data_end - y_start
                    x_copy_size = x_data_end - x_start

                    remote_data = remote_array[0,
                                  x_start:x_data_end,
                                  y_start:y_data_end,
                                  z_start:z_data_end]

                    transposed_data = np.transpose(remote_data, (2, 1, 0))
                    chunk_data[:z_copy_size, :y_copy_size, :x_copy_size] = transposed_data

                local_slice = (
                    slice(z_start, z_end),
                    slice(y_start, y_end),
                    slice(x_start, x_end)
                )
                local_array[local_slice] = chunk_data

                return chunk_coords

            except Exception as e:
                error_msg = str(e)
                if attempt < max_retries - 1:
                    if "502" in error_msg or "Bad Gateway" in error_msg or "ConnectionError" in error_msg:
                        print(
                            f"\nRetrying chunk {chunk_coords} (attempt {attempt + 2}/{max_retries}) after error: {error_msg}")
                        time.sleep(attempt + 1)
                        continue
                raise Exception(f"Failed to fetch chunk {chunk_coords} after {max_retries} attempts: {error_msg}")

    chunk_coords = []
    for z in range(0, z_padded, fetch_chunk_size[0]):
        for y in range(0, y_padded, fetch_chunk_size[1]):
            for x in range(0, x_padded, fetch_chunk_size[2]):
                chunk_coords.append((z, y, x))

    total_chunks = len(chunk_coords)
    failed_chunks = []

    with tqdm(total=total_chunks, desc="Fetching and converting chunks") as pbar:
        for coords in chunk_coords:
            try:
                copy_and_transform_chunk(coords)
                pbar.update(1)
            except Exception as e:
                print(f"\nError copying chunk {coords}: {e}")
                failed_chunks.append(coords)
                pbar.update(1)

    if failed_chunks:
        print(f"\nWarning: {len(failed_chunks)} chunks failed after all retries:")
        for chunk in failed_chunks[:10]:
            print(f"  - Chunk at {chunk}")
        if len(failed_chunks) > 10:
            print(f"  ... and {len(failed_chunks) - 10} more")
        print("\nThe data in these regions will be zeros.")
    else:
        print("\nAll chunks successfully downloaded!")

    if hasattr(remote_array, 'attrs'):
        local_array.attrs.update(remote_array.attrs)

    print(f"Successfully downloaded and converted to: {local_path}")
    print(f"Final shape (z, y, x): {local_array.shape}")
    print(f"Chunks: {local_array.chunks}")
    print(f"Compressor: {local_array.compressor}")


def main():
    parser = argparse.ArgumentParser(description='Download and convert WebKnossos zarr data')
    parser.add_argument('--surface_volume', required=True,
                        help='URL to surface volume data (e.g., http://path.com/to/UUID/surface_volume/1)')
    parser.add_argument('--ink_labels', required=True,
                        help='URL to ink labels data (e.g., http://path.com/to/UUID/ink_labels/1)')

    args = parser.parse_args()

    # Extract UUID from URLs
    try:
        surface_uuid = extract_uuid_from_url(args.surface_volume)
        ink_uuid = extract_uuid_from_url(args.ink_labels)

        # Verify both URLs have the same UUID
        if surface_uuid != ink_uuid:
            print(f"Warning: Different UUIDs detected in URLs!")
            print(f"Surface volume UUID: {surface_uuid}")
            print(f"Ink labels UUID: {ink_uuid}")
            print("Using surface volume UUID for output paths...")

        uuid = surface_uuid

    except ValueError as e:
        print(f"Error: {e}")
        return

    # Create output directories if they don't exist
    fragments_dir = "/vesuvius/fragments"
    inklabels_dir = "/vesuvius/inklabels"

    os.makedirs(fragments_dir, exist_ok=True)
    os.makedirs(inklabels_dir, exist_ok=True)

    # Construct output paths
    surface_output = os.path.join(fragments_dir, f"{uuid}.zarr")
    ink_output = os.path.join(inklabels_dir, f"{uuid}.zarr")

    print(f"\nExtracted UUID: {uuid}")
    print(f"Surface volume will be saved to: {surface_output}")
    print(f"Ink labels will be saved to: {ink_output}")
    print("")

    # Download surface volume
    download_zarr_with_conversion(args.surface_volume, surface_output)

    # Download ink labels
    download_zarr_with_conversion(args.ink_labels, ink_output)


if __name__ == '__main__':
    main()