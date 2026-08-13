import os
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import tifffile
from datetime import datetime
from scipy.ndimage import uniform_filter


def adaptive_merge(predictions_list, folder_weights=None, window_size=9, detail_scale=1.0, smoothing=1.0):
    """
    Enhanced adaptive weighted merging with adjustable parameters and folder-specific weights.
    """
    preds = [pred.astype(np.float32) for pred in predictions_list]

    # Calculate local means
    local_means = [uniform_filter(pred, window_size) for pred in preds]

    # Calculate local variance with detail scaling
    variances = [
        uniform_filter((pred - mean) ** 2, window_size) * detail_scale
        for pred, mean in zip(preds, local_means)
    ]

    # Apply smoothing to variances
    if smoothing > 0:
        variances = [uniform_filter(var, size=int(window_size * smoothing)) for var in variances]

    # Calculate weights from variances
    weights = [var / (var.max() + 1e-6) for var in variances]

    # Apply folder-specific weights if provided
    if folder_weights is not None:
        weights = [w * fw for w, fw in zip(weights, folder_weights)]

    weight_sum = sum(weights)
    weight_sum[weight_sum == 0] = 1

    # Merge using weights
    merged = sum(pred * weight / weight_sum for pred, weight in zip(preds, weights))

    return np.clip(merged, 0, 255)


def pad_if_needed(image, expected_size):
    """Pad image if it's smaller than expected size."""
    if not isinstance(expected_size, (tuple, list)):
        expected_size = (expected_size,) * 3

    current_size = image.shape
    if any(c < e for c, e in zip(current_size, expected_size)):
        pad_width = []
        for c, e in zip(current_size, expected_size):
            if c < e:
                diff = e - c
                pad_before = diff // 2
                pad_after = diff - pad_before
                pad_width.append((pad_before, pad_after))
            else:
                pad_width.append((0, 0))
        return np.pad(image, pad_width, mode='constant', constant_values=0)
    return image


def process_single_file(args):
    """Process a single file with given parameters."""
    filename, input_folders, output_folder, params = args
    try:
        # Read predictions
        predictions = []
        for folder in input_folders:
            file_path = os.path.join(folder, filename)
            if os.path.exists(file_path):
                with tifffile.TiffFile(file_path) as tif:
                    img = tifffile.imread(file_path)
                    if params['pad_to_size']:
                        img = pad_if_needed(img, params['expected_size'])
                    predictions.append(img)
            else:
                print(f"Warning: {file_path} not found")
                return False

        # Process each page/slice
        merged_pages = []
        for page_idx in range(predictions[0].shape[0]):
            page_predictions = [pred[page_idx] for pred in predictions]
            merged_page = adaptive_merge(
                page_predictions,
                folder_weights=params['folder_weights'],
                window_size=params['window_size'],
                detail_scale=params['detail_scale'],
                smoothing=params['smoothing']
            )
            merged_pages.append(merged_page)

        # Save merged result
        merged_array = np.stack(merged_pages)
        output_path = os.path.join(output_folder, filename)
        tifffile.imwrite(output_path, merged_array.astype(np.uint8))
        return True

    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return False


def process_folder(input_folders, output_folder, params):
    """Process all files in the input folders."""
    os.makedirs(output_folder, exist_ok=True)

    # Get list of files (using just the basename)
    files = [os.path.basename(f) for f in os.listdir(input_folders[0])
             if f.endswith('.tif') or f.endswith('.tiff')]

    if not files:
        print(f"No TIFF files found in {input_folders[0]}")
        return

    print(f"Found {len(files)} TIFF files")
    print(f"First few files: {files[:5]}")

    # Prepare arguments for each file
    args_list = [(f, input_folders, output_folder, params) for f in files]

    # Process files using multiprocessing
    with mp.Pool(params['num_processes'] or mp.cpu_count()) as pool:
        results = list(tqdm(
            pool.imap(process_single_file, args_list),
            total=len(files),
            desc="Processing files"
        ))

    # Print summary
    successful = sum(1 for r in results if r)
    print(f"\nProcessing complete:")
    print(f"Successfully processed: {successful}/{len(files)} files")
    print(f"Failed: {len(files) - successful} files")


if __name__ == "__main__":
    input_folders = [
        "/mnt/raid_nvme/datasets/ensemble/aug_erode_sm_tiffs",
        "/mnt/raid_nvme/datasets/ensemble/044_sm_pred_z_up_tifs",
        "/mnt/raid_nvme/datasets/ensemble/s1-gp-skeleton-sm",
        "/mnt/raid_nvme/datasets/ensemble/050_resencextraDAdistloss_sm_grids"
    ]

    #  "/mnt/raid_nvme/datasets/ensemble/036_skel_sm"
    # "/mnt/raid_nvme/datasets/ensemble/medial_thresholded",

    # All parameters in one dictionary
    params = {
        # Processing parameters
        'window_size': 9,
        'detail_scale': 0.67,
        'smoothing': 0.29,
        'num_processes': 6,

        # Size handling
        'expected_size': 600,  # or specify (600,600,600) for different dimensions
        'pad_to_size': True,  # whether to pad smaller images

        # Folder weights (must match number of input folders)
        'folder_weights': [1.0, 1.0, 1.0, 1.0]  # Example: third folder has half weight
    }

    output_folder = "/mnt/raid_nvme/datasets/ensemble/043_044_050_random_ensemble"

    process_folder(
        input_folders=input_folders,
        output_folder=output_folder,
        params=params
    )