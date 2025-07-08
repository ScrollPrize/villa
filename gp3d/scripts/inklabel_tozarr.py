import os
import cv2
import numpy as np
import zarr
from tqdm import tqdm
import glob
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, as_completed
from numcodecs import Blosc
from config import *

from config import *


INK_LABELS_IMG_PATH = f"{os.path.abspath(os.path.dirname(__file__))}/../all_labels/"


def get_fragment_depth(fragment_id):
    """Get the depth (number of layers) for a fragment by checking its zarr volume"""
    volume_zarr_path = os.path.join(ZARRS_PATH, f"{fragment_id}.zarr")

    if not os.path.exists(volume_zarr_path):
        print(f"Warning: Volume zarr not found for {fragment_id}")
        return None

    try:
        z_array = zarr.open_array(volume_zarr_path, mode='r')
        depth, height, width = z_array.shape
        return depth, height, width
    except Exception as e:
        print(f"Error reading volume zarr for {fragment_id}: {e}")
        return None


def load_ink_label_2d(path):
    """Load a 2D ink label from PNG or TIFF"""
    if path.endswith('.png'):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    elif path.endswith('.tiff') or path.endswith('.tif'):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        raise ValueError(f"Unsupported file format: {path}")

    if img is None:
        raise ValueError(f"Failed to load image: {path}")

    return img


def create_3d_ink_label_zarr(fragment_id, ink_label_2d, volume_shape, output_path):
    """Create a 3D zarr from a 2D ink label by broadcasting to all depths"""
    depth, height, width = volume_shape

    # Resize 2D label if needed to match volume dimensions
    if ink_label_2d.shape != (height, width):
        ink_label_2d = cv2.resize(ink_label_2d, (width, height), interpolation=cv2.INTER_NEAREST)

    # Create zarr array with same chunking strategy as volume
    compressor = Blosc(cname='blosclz', clevel=9)
    z_array = zarr.open_array(
        output_path,
        mode='w',
        shape=(depth, height, width),
        chunks=(64, 256, 256),  # Match the volume chunking
        dtype='uint8',
        compressor=compressor
    )

    # Process in batches to reduce memory usage
    batch_size = 16
    num_batches = (depth + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, depth)
        batch_depth = end_idx - start_idx

        # Create batch by broadcasting the 2D label
        batch_array = np.broadcast_to(
            ink_label_2d[np.newaxis, :, :],
            (batch_depth, height, width)
        ).copy()  # Copy to ensure it's writable

        # Write batch to zarr
        z_array[start_idx:end_idx] = batch_array

        # Free memory
        del batch_array

    return z_array.shape


def process_fragment_ink_labels(args):
    """Process ink labels for a single fragment"""
    fragment_id, input_dir, output_dir = args

    # Check if zarr already exists
    output_zarr_path = os.path.join(output_dir, f"{fragment_id}.zarr")

    if os.path.exists(output_zarr_path):
        return fragment_id, True, "Already exists"

    # Get volume dimensions
    volume_info = get_fragment_depth(fragment_id)
    if volume_info is None:
        return fragment_id, False, "No volume zarr found"

    depth, height, width = volume_info

    # Look for existing 2D ink label files in input directory
    png_path = os.path.join(input_dir, f"{fragment_id}_inklabels.png")
    tiff_path = os.path.join(input_dir, f"{fragment_id}_inklabels.tiff")
    tif_path = os.path.join(input_dir, f"{fragment_id}_inklabels.tif")

    ink_label_path = None
    if os.path.exists(png_path):
        ink_label_path = png_path
    elif os.path.exists(tiff_path):
        ink_label_path = tiff_path
    elif os.path.exists(tif_path):
        ink_label_path = tif_path
    else:
        return fragment_id, False, "No ink label file found"

    try:
        # Load 2D ink label
        ink_label_2d = load_ink_label_2d(ink_label_path)

        # Create 3D zarr
        shape = create_3d_ink_label_zarr(
            fragment_id,
            ink_label_2d,
            (depth, height, width),
            output_zarr_path
        )

        return fragment_id, True, f"Created {shape}"

    except Exception as e:
        # Clean up partial file if it exists
        if os.path.exists(output_zarr_path):
            import shutil
            shutil.rmtree(output_zarr_path)
        return fragment_id, False, str(e)


def get_fragments_with_ink_labels(input_dir):
    """Get list of fragment IDs that have ink label files"""
    fragment_ids = set()

    # Look for PNG files
    png_files = glob.glob(os.path.join(input_dir, "*_inklabels.png"))
    for f in png_files:
        basename = os.path.basename(f)
        frag_id = basename.replace("_inklabels.png", "")
        fragment_ids.add(frag_id)

    # Look for TIFF files
    tiff_files = glob.glob(os.path.join(input_dir, "*_inklabels.tiff"))
    tiff_files.extend(glob.glob(os.path.join(input_dir, "*_inklabels.tif")))
    for f in tiff_files:
        basename = os.path.basename(f)
        frag_id = basename.replace("_inklabels.tiff", "").replace("_inklabels.tif", "")
        fragment_ids.add(frag_id)

    return sorted(list(fragment_ids))


def main():
    """Convert all PNG/TIFF ink labels to 3D zarr format"""
    print(f"Input directory (PNG/TIFF): {INK_LABELS_IMG_PATH}")
    print(f"Output directory (zarr): {INK_LABELS_PATH}")
    print(f"Volume zarrs path: {ZARRS_PATH}")

    # Create output directory if it doesn't exist
    os.makedirs(INK_LABELS_PATH, exist_ok=True)

    # Get fragments with ink labels
    fragments = get_fragments_with_ink_labels(INK_LABELS_IMG_PATH)
    print(f"Found {len(fragments)} fragments with ink labels")

    if not fragments:
        print("No ink label files found!")
        return

    # Prepare arguments for multiprocessing
    fragment_args = [(frag_id, INK_LABELS_IMG_PATH, INK_LABELS_PATH) for frag_id in fragments]

    # Process fragments in parallel
    num_workers = 4
    print(f"Processing {num_workers} fragments in parallel")

    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_fragment_ink_labels, fragment_args),
            total=len(fragments),
            desc="Converting ink labels to 3D zarr"
        ))

    # Report results
    successful = sum(1 for _, success, _ in results if success)
    failed = [(fid, err) for fid, success, err in results if not success and err not in ["Already exists"]]
    skipped = sum(1 for _, success, err in results if success and err == "Already exists")
    no_volume = sum(1 for _, success, err in results if not success and err == "No volume zarr found")

    print(f"\nConversion complete:")
    print(f"  - {successful - skipped} ink labels newly converted")
    print(f"  - {skipped} ink labels skipped (already exist)")
    print(f"  - {no_volume} fragments skipped (no volume zarr)")
    print(f"  - {len(failed)} fragments failed")

    if failed:
        print("\nFailed fragments:")
        for fid, err in failed[:10]:  # Show first 10 failures
            print(f"  {fid}: {err}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")


if __name__ == "__main__":
    main()