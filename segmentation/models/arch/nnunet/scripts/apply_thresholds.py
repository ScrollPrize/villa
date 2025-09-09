import os
from PIL import Image
import numpy as np
from skimage import filters
from pathlib import Path
import tifffile
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from functools import partial

def process_tiff_with_threshold(input_path, output_path, threshold_method):
    """
    Process a multi-page TIFF file with the specified threshold method.
    
    Args:
        input_path: Path to input TIFF file
        output_path: Path to save thresholded TIFF file
        threshold_method: Function to calculate threshold (e.g., filters.threshold_otsu)
    """
    # Read the multi-page TIFF
    tiff = tifffile.imread(input_path)
    
    # Handle both single and multi-page TIFFs
    if tiff.ndim == 2:
        tiff = tiff[np.newaxis, ...]
    
    # Process each page
    thresholded_pages = []
    for page in tiff:
        # Calculate threshold
        thresh_value = threshold_method(page)
        # Apply threshold
        binary = (page > thresh_value).astype(np.uint8) * 255
        thresholded_pages.append(binary)
    
    # Save the processed pages as a new TIFF
    tifffile.imwrite(output_path, np.array(thresholded_pages))
    return input_path

def process_single_file(args):
    """
    Process a single file with remaining threshold methods.
    Wrapper function for multiprocessing.
    """
    input_file, methods, output_folders = args
    results = []
    
    # Check which methods still need to be processed for this file
    remaining_methods = {}
    for method_name, threshold_func in methods.items():
        output_path = os.path.join(output_folders[method_name], input_file.name)
        if not os.path.exists(output_path):
            remaining_methods[method_name] = threshold_func
    
    # If all methods are completed for this file, skip it
    if not remaining_methods:
        return []
    
    for method_name, threshold_func in remaining_methods.items():
        output_path = os.path.join(output_folders[method_name], input_file.name)
        try:
            process_tiff_with_threshold(str(input_file), output_path, threshold_func)
            results.append((True, method_name, input_file.name))
        except Exception as e:
            results.append((False, method_name, str(e)))
    
    return results

def process_folder(input_folder, output_base_folder, num_processes=None):
    """
    Process all TIFF files in the input folder using different thresholding methods.
    
    Args:
        input_folder: Path to folder containing input TIFF files
        output_base_folder: Base path for output folders
        num_processes: Number of processes to use (defaults to CPU count - 1)
    """
    if num_processes is None:
        num_processes = max(1, cpu_count() - 1)
    
    # Create output folders for each threshold method
    methods = {
        'otsu': filters.threshold_otsu
        #'maxentropy': filters.threshold_li,  # Li's method is equivalent to max entropy
    }
    
    output_folders = {}
    for method in methods.keys():
        folder_path = os.path.join(output_base_folder, method)
        os.makedirs(folder_path, exist_ok=True)
        output_folders[method] = folder_path
    
    # Get list of input files
    input_files = list(Path(input_folder).glob('*.tif*'))
    total_files = len(input_files)
    
    if total_files == 0:
        print("No TIFF files found in the input folder!")
        return
    
    # Count remaining files to process by checking which files don't have all outputs
    remaining_files = []
    for input_file in input_files:
        needs_processing = False
        for method in methods.keys():
            output_path = os.path.join(output_folders[method], input_file.name)
            if not os.path.exists(output_path):
                needs_processing = True
                break
        if needs_processing:
            remaining_files.append(input_file)
    
    print(f"Found {total_files} TIFF files")
    print(f"Remaining files to process: {len(remaining_files)}")
    print(f"Using {num_processes} processes")
    
    if not remaining_files:
        print("All files have been processed!")
        return
    
    # Prepare arguments for multiprocessing
    process_args = [(f, methods, output_folders) for f in remaining_files]
    
    # Process files with progress bar
    try:
        with Pool(num_processes) as pool:
            # Using tqdm to show progress
            results = list(tqdm(
                pool.imap(process_single_file, process_args),
                total=len(remaining_files),
                desc="Processing files",
                unit="file"
            ))
        
        # Print summary
        print("\nProcessing Summary:")
        for file_results in results:
            for success, method, message in file_results:
                if success:
                    print(f"✓ {method}: {message}")
                else:
                    print(f"✗ {method}: Error - {message}")
    
    except KeyboardInterrupt:
        print("\nProcessing interrupted!")
        print("Run the script again to resume from where it left off.")
        return

# Example usage
if __name__ == "__main__":
    input_folder = "/mnt/raid_nvme/datasets/ensemble/043_044_050_random_ensemble_MEAN" # Replace with your input folder path
    output_base_folder = "/mnt/raid_nvme/datasets/ensemble/043_044_050_random_ensemble_MEAN_otsu"  # Replace with your output base folder path
    
    # Optional: specify number of processes (defaults to CPU count - 1)
    process_folder(input_folder, output_base_folder, num_processes=4)