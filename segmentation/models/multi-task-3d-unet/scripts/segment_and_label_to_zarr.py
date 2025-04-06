import os
import numpy as np
import zarr
import cv2
import glob
import re
from tqdm import tqdm

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'([0-9]+)', s)]

def extract_first_int(fname):
    """
    Extract the first integer found in the given string (filename).
    Returns None if no integer is found.
    """
    match = re.search(r'(\d+)', fname)
    if match:
        return int(match.group(1))
    return None

def process_all_segments(parent_folder, start, stop, layers_only=False, erode=False):
    """
    Process all segment folders in the parent directory.
    Each segment folder should contain 'layers' and possibly 'inklabels' subdirectories.
    """
    segment_folders = [
        f.path for f in os.scandir(parent_folder)
        if f.is_dir() and os.path.exists(os.path.join(f.path, 'layers'))
    ]  # Must have layers subdir

    if not segment_folders:
        raise ValueError(f"No valid segment folders found in {parent_folder}")

    print(f"Found {len(segment_folders)} segment folders to process")

    # Process each segment folder
    for folder in tqdm(segment_folders, desc="Processing segments"):
        try:
            print(f"\nProcessing segment folder: {os.path.basename(folder)}")
            stack_images_to_zarr(folder, start, stop, layers_only, erode)
        except Exception as e:
            print(f"Error processing {folder}: {str(e)}")
            continue

def stack_images_to_zarr(input_folder, start, stop, layers_only=False, erode=False):
    """
    - Layers are found by matching the integer in the filename (e.g., 'layer_25.tif' -> 25).
    - Inklabels remain as in the original code: sorted by natural sort, then indexed by [start..stop].
    """
    # ------------------------------------------------------------------------
    # STEP 1) LAYERS: BUILD A DICTIONARY { integer_in_filename: full_path }
    # ------------------------------------------------------------------------
    image_extensions = ('*.tif', '*.TIF', '*.png', '*.PNG', '*.jpg', '*.JPG', '*.jpeg', '*.JPEG')
    layer_files = []
    for ext in image_extensions:
        layer_files.extend(glob.glob(os.path.join(input_folder, 'layers', ext)))
    if not layer_files:
        raise ValueError(f"No .tif files found in {input_folder}/layers")

    # Sort with the same natural sort key (for consistent logging/order)
    layer_files.sort(key=natural_sort_key)

    layer_dict = {}
    for f in layer_files:
        fname = os.path.basename(f)
        idx = extract_first_int(fname)
        if idx is not None:
            layer_dict[idx] = f

    # We need to confirm that at least the `start` index is present for shape reading:
    if start not in layer_dict:
        raise ValueError(
            f"Cannot find a layer file whose filename contains index {start} "
            f"in {input_folder}/layers"
        )

    # Read the first requested layer to get shape
    first_layer_path = layer_dict[start]
    first_layer = cv2.imread(first_layer_path, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
    if first_layer is None:
        raise ValueError(f"Could not read layer image: {first_layer_path}")
    layer_h, layer_w = first_layer.shape

    # -------------------------------------------------------------------
    # STEP 2) INKLABELS (UNCHANGED) - old indexing approach by sorted list
    # -------------------------------------------------------------------
    if not layers_only:
        inklabels = glob.glob(os.path.join(input_folder, 'inklabels', '*.png'))
        inklabels.sort(key=natural_sort_key)
        if not inklabels:
            raise ValueError(f"No inklabels found in {input_folder}/inklabels")
        print(f"Found {len(layer_dict)} unique layer indices and {len(inklabels)} inklabels")

        # Check shape of the *first* inklabel in the sorted list
        first_inklabel = cv2.imread(inklabels[0], cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
        if first_inklabel is not None:
            ink_h, ink_w = first_inklabel.shape
            print(f"First inklabel shape: {(ink_h, ink_w)}")
    else:
        inklabels = []
        print(f"Found {len(layer_dict)} unique layer indices (layers only mode)")

    # -----------------------------------------------------------------
    # STEP 3) CREATE ZARR GROUP
    # -----------------------------------------------------------------
    parent_dir = os.path.dirname(input_folder)
    z_name = os.path.basename(input_folder)
    zarr_path = os.path.join(parent_dir, f"{z_name}.zarr")
    print(f"Creating zarr array at {zarr_path}...")

    store = zarr.DirectoryStore(zarr_path)
    z_root = zarr.group(store=store, overwrite=True)

    num_slices = stop - start + 1

    # Create dataset for layers (same chunking as before)
    z_root.create_dataset(
        'layers.zarr',
        shape=(num_slices, layer_h, layer_w),
        chunks=(num_slices, 128, 128),
        dtype=np.uint8
    )

    # Create dataset for inklabels if needed
    if not layers_only:
        z_root.create_dataset(
            'inklabels.zarr',
            shape=(num_slices, ink_h, ink_w),
            chunks=(num_slices, 128, 128),
            dtype=np.uint8
        )

    # -----------------------------------------------------------------
    # STEP 4) WRITE LAYERS (dictionary-based by integer in filename)
    # -----------------------------------------------------------------
    print(f"Writing layer slices for indices [{start}..{stop}]...")
    for idx in tqdm(range(num_slices), desc="Loading layers"):
        actual_idx = start + idx  # the integer we want
        if actual_idx not in layer_dict:
            raise ValueError(
                f"No layer file found containing index {actual_idx} in folder {input_folder}/layers"
            )

        layer_path = layer_dict[actual_idx]
        layer_img = cv2.imread(layer_path, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
        if layer_img is None:
            raise ValueError(f"Could not read layer image: {layer_path}")

        # Convert from 16-bit to 8-bit if needed
        if layer_img.dtype == np.uint16:
            scaled_layer = (layer_img / 65535.0 * 255).astype(np.uint8)
        else:
            scaled_layer = layer_img.astype(np.uint8)

        z_root['layers.zarr'][idx] = scaled_layer

    # ----------------------------------------------------------------------
    # STEP 5) WRITE INKLABELS (old approach) - unmodified from your original
    # ----------------------------------------------------------------------
    if not layers_only:
        print("Writing inklabel slices (old indexing approach)...")
        middle_idx = num_slices // 2

        for idx in tqdm(range(num_slices), desc="Loading and writing inklabels"):
            file_index = start + idx
            if file_index >= len(inklabels):
                file_index = len(inklabels) - 1  # fallback to last

            inklabel_path = inklabels[file_index]
            inklabel_img = cv2.imread(inklabel_path, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
            if inklabel_img is None:
                raise ValueError(f"Could not read inklabel image: {inklabel_path}")

            # Convert from 16-bit to 8-bit if needed
            if inklabel_img.dtype == np.uint16:
                inklabel_img = (inklabel_img / 65535.0 * 255).astype(np.uint8)
            else:
                inklabel_img = inklabel_img.astype(np.uint8)

            # Optionally apply erosion
            if erode:
                erosion_iterations = abs(idx - middle_idx)
                if erosion_iterations > 0:
                    kernel = np.ones((7, 7), np.uint8)
                    inklabel_img = cv2.erode(inklabel_img, kernel, iterations=erosion_iterations)

            z_root['inklabels.zarr'][idx] = inklabel_img

    print("Done!")

if __name__ == "__main__":
    parent_folder = "/home/sean/Desktop/process"
    process_all_segments(parent_folder, start=26, stop=39, layers_only=True, erode=False)
