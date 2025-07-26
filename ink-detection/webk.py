import zarr
import numpy as np
from tqdm import tqdm
import time
import argparse
import os
import re
from numcodecs import Blosc


def extract_uuid_from_url(url):
    pattern = r'/([^/]+)/(surface_volume|ink_labels)/\d+/?$'
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"Could not extract UUID from URL: {url}")


def download_zarr(remote_url, local_path, is_surface=True):
    # Open remote array
    remote_store = zarr.storage.FSStore(remote_url)
    remote_array = zarr.open_array(store=remote_store, mode='r')

    # Get dimensions
    _, x_size, y_size, z_size = remote_array.shape
    z_size_out = min(z_size, 64)

    print(f"Remote shape: {remote_array.shape}")
    print(f"Downloading all {z_size_out} layers")

    if is_surface:
        # Create output zarr array
        compressor = Blosc(cname='blosclz', clevel=9)
        local_array = zarr.open_array(
            local_path,
            mode='w',
            shape=(z_size_out, y_size, x_size),
            chunks=(64, 512, 512),
            dtype=remote_array.dtype,
            compressor=compressor
        )
    else:
        # For labels, we'll accumulate in memory then save as PNG
        labels_volume = np.zeros((z_size_out, y_size, x_size), dtype=bool)

    # Generate chunks (z, y, x)
    chunk_size = (64, 512, 512)
    chunks = []
    for z in range(0, z_size_out, chunk_size[0]):
        for y in range(0, y_size, chunk_size[1]):
            for x in range(0, x_size, chunk_size[2]):
                chunks.append((z, y, x))

    print(f"Processing {len(chunks)} chunks...")

    # Process chunks
    for z_start, y_start, x_start in tqdm(chunks):
        z_end = min(z_start + chunk_size[0], z_size_out)
        y_end = min(y_start + chunk_size[1], y_size)
        x_end = min(x_start + chunk_size[2], x_size)

        # Retry logic
        max_retries = 10
        for attempt in range(max_retries):
            time.sleep(0.1)
            try:
                # Fetch data - same for both surface and labels
                data = remote_array[0, x_start:x_end, y_start:y_end, z_start:z_end]
                # Transpose (x,y,z) -> (z,y,x)
                data_transposed = np.transpose(data, (2, 1, 0))

                if is_surface:
                    # Clip and save
                    data_transposed = np.clip(data_transposed, 0, 200)
                    local_array[z_start:z_end, y_start:y_end, x_start:x_end] = data_transposed
                else:
                    # Store in memory for later OR operation
                    labels_volume[z_start:z_end, y_start:y_end, x_start:x_end] = data_transposed > 0

                break
            except Exception as e:
                error_msg = str(e)
                if attempt < max_retries - 1:
                    if "502" in error_msg or "Bad Gateway" in error_msg or "ConnectionError" in error_msg:
                        print(
                            f"\nRetrying chunk ({z_start},{y_start},{x_start}) (attempt {attempt + 2}/{max_retries}) after error: {error_msg}")
                        time.sleep(attempt + 1)
                        continue
                print(f"\nFailed chunk ({z_start},{y_start},{x_start}) after {max_retries} attempts: {error_msg}")

    if not is_surface:
        # Collapse labels to 2D and save as PNG
        import cv2
        labels_2d = np.any(labels_volume, axis=0).astype(np.uint8) * 255
        cv2.imwrite(local_path, labels_2d)

    print(f"Download complete: {local_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--surface_volume', required=True)
    parser.add_argument('--ink_labels', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()

    uuid = extract_uuid_from_url(args.surface_volume)
    os.makedirs(args.output_dir, exist_ok=True)

    surface_output = os.path.join(args.output_dir, f"{uuid}.zarr")
    if os.path.exists(surface_output):
        print(f"surface volume already exists at {surface_output}, skipping download")
    else:
        download_zarr(args.surface_volume, surface_output, is_surface=True)

    ink_output = os.path.join(args.output_dir, f"{uuid}_inklabels.png")
    if os.path.exists(ink_output):
        print(f"inklabels already exists at {ink_output}, skipping download")
    else:
        download_zarr(args.ink_labels, ink_output, is_surface=False)

if __name__ == '__main__':
    main()