import os
import cv2
import numpy as np
import skimage.exposure
import zarr
from tqdm import tqdm
import glob
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, as_completed
from numcodecs import Blosc
import time
from config import *


def load_and_convert_layer(path, target_shape=None):
    """Load and process a single layer"""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if img.dtype == np.uint16:
        img = (img >> 8).astype(np.uint8)

    if target_shape is not None:
        pad_y = target_shape[0] - img.shape[0]
        pad_x = target_shape[1] - img.shape[1]
        if pad_y > 0 or pad_x > 0:
            img = np.pad(img, [(0, max(0, pad_y)), (0, max(0, pad_x))], constant_values=0)

    return img


def get_padded_shape(height, width, chunk_size=256):
    padded_h = ((height + chunk_size - 1) // chunk_size) * chunk_size
    padded_w = ((width + chunk_size - 1) // chunk_size) * chunk_size
    return padded_h, padded_w


def export_fragment_to_zarr(fragment_path, zarr_path, batch_size=16):
    """Export a fragment - process layers in batches to reduce memory usage"""
    fragment_id = os.path.basename(fragment_path)

    # Find layer files
    tif_files = sorted(glob.glob(os.path.join(fragment_path, "layers", "*.tif")))
    jpg_files = sorted(glob.glob(os.path.join(fragment_path, "layers", "*.jpg")))
    layer_files = tif_files if tif_files else jpg_files

    if not layer_files:
        print(f"No layer files found for {fragment_id}")
        return

    layer_files = layer_files[:64]

    # Get dimensions from first image
    first_img = load_and_convert_layer(layer_files[0])
    h, w = first_img.shape
    padded_h, padded_w = get_padded_shape(h, w)

    # Create zarr array
    compressor = Blosc(cname='blosclz', clevel=9)
    z_array = zarr.open_array(
        zarr_path,
        mode='w',
        shape=(64, padded_h, padded_w),
        chunks=(64, 256, 256),
        dtype='uint8',
        compressor=compressor)

    # Process layers in batches
    num_layers = len(layer_files)
    num_batches = (64 + batch_size - 1) // batch_size  # Ceiling division

    for batch_idx in range(num_batches):
        # Calculate batch range
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, 64)
        batch_layers = end_idx - start_idx

        # Allocate array for this batch
        batch_array = np.zeros((batch_layers, padded_h, padded_w), dtype=np.uint8)

        # Determine which layer files to load for this batch
        layer_start = start_idx
        layer_end = min(end_idx, num_layers)

        if layer_start < num_layers:
            # Load batch layers in parallel
            with ThreadPoolExecutor(max_workers=min(48, batch_layers)) as executor:
                future_to_idx = {}
                for idx in range(layer_start, layer_end):
                    future = executor.submit(load_and_convert_layer, layer_files[idx], (padded_h, padded_w))
                    future_to_idx[future] = idx - start_idx  # Relative index within batch

                for future in as_completed(future_to_idx):
                    batch_relative_idx = future_to_idx[future]
                    try:
                        batch_array[batch_relative_idx] = future.result()
                    except Exception as e:
                        print(f"Error loading layer {start_idx + batch_relative_idx}: {e}")

        # Write this batch to zarr
        z_array[start_idx:end_idx] = batch_array

        # Free memory
        del batch_array

    print(f"Exported {fragment_id}: shape={z_array.shape}, batch_size={batch_size}")


def process_fragment(args):
    """Process a single fragment with configurable batch size"""
    fragment_path, batch_size = args
    fragment_id = os.path.basename(fragment_path)
    fragment_zarr_path = os.path.join(ZARRS_PATH, f"{fragment_id}.zarr")

    if os.path.exists(fragment_zarr_path):
        return fragment_id, True, "Already exists"

    try:
        export_fragment_to_zarr(fragment_path, fragment_zarr_path, batch_size)
        return fragment_id, True, None
    except Exception as e:
        return fragment_id, False, str(e)


def main(batch_size=32):
    train_scrolls_dir = f"{VESUVIUS_ROOT}/train_scrolls"
    os.makedirs(ZARRS_PATH, exist_ok=True)

    fragment_dirs = [d for d in glob.glob(os.path.join(train_scrolls_dir, "*"))
                     if os.path.isdir(d) and os.path.exists(os.path.join(d, "layers"))]

    print(f"Found {len(fragment_dirs)} fragments to export")
    print(f"Output directory: {ZARRS_PATH}")
    print(f"Batch size: {batch_size} layers")

    num_workers = 4
    print(f"Processing {num_workers} fragments in parallel")

    # Prepare arguments for each fragment
    fragment_args = [(fragment_path, batch_size) for fragment_path in sorted(fragment_dirs)]

    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_fragment, fragment_args),
            total=len(fragment_dirs),
            desc="Exporting fragments"
        ))

    # Report results
    successful = sum(1 for _, success, _ in results if success)
    failed = [(fid, err) for fid, success, err in results if not success and err != "Already exists"]
    skipped = sum(1 for _, success, err in results if success and err == "Already exists")

    print(f"\nExport complete:")
    print(f"  - {successful - skipped} fragments newly exported")
    print(f"  - {skipped} fragments skipped (already exist)")
    print(f"  - {len(failed)} fragments failed")

    if failed:
        print("\nFailed fragments:")
        for fid, err in failed:
            print(f"  {fid}: {err}")


if __name__ == "__main__":
    # You can change the batch_size here
    main(batch_size=32)