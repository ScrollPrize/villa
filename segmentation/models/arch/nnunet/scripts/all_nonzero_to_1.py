import os
from pathlib import Path
import tifffile
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from functools import partial

def normalize_16bit_to_8bit(data):
    """
    Normalize 16-bit data to 8-bit range while preserving relative values
    
    Parameters:
    data (numpy.ndarray): 16-bit input data
    
    Returns:
    numpy.ndarray: 8-bit normalized data
    """
    if data.dtype == np.uint16:
        # Convert to float for calculations
        data_float = data.astype(np.float32)
        
        # Normalize to 0-255 range
        data_norm = ((data_float - data_float.min()) * 255 / 
                    (data_float.max() - data_float.min()))
        
        # Convert to uint8
        return data_norm.astype(np.uint8)
    return data

def check_missing_values(data, values_to_check):
    """
    Check if specific values are missing from the image data
    
    Parameters:
    data (numpy.ndarray): Image data
    values_to_check (list): List of values to check for
    
    Returns:
    list: List of values that are missing from the data
    """
    unique_values = np.unique(data)
    missing_values = [val for val in values_to_check if val not in unique_values]
    return missing_values

def apply_value_mapping(data, value_mapping=None):
    """
    Set any non-zero value to 1
    
    Parameters:
    data (numpy.ndarray): Input data
    value_mapping (dict): Not used in this version, kept for compatibility
    
    Returns:
    numpy.ndarray: Processed data with all non-zero values set to 1
    """
    processed = data.copy()
    
    # Add debug prints
    print(f"Before mapping - unique values: {np.unique(processed)}")
    print(f"Data type: {processed.dtype}")
    
    # Simply set all non-zero values to 1
    processed = (processed > 0).astype(data.dtype)
    
    # Add debug print
    print(f"After mapping - unique values: {np.unique(processed)}")
    
    return processed

def process_tiff(input_path, output_path, value_mapping, check_values=None):
    """
    Process a multipage TIFF file using a custom value mapping and check for missing values
    
    Parameters:
    input_path (str): Path to input TIFF file
    output_path (str): Path to save processed TIFF file
    value_mapping (dict): Dictionary specifying {old_value: new_value} mappings
    check_values (list): List of values to check for in the image
    """
    try:
        # Read the TIFF file
        with tifffile.TiffFile(input_path) as tif:
            data = tif.asarray()
            
            # Check if input is 16-bit
            is_16bit = data.dtype == np.uint16
            
            # Check for missing values if specified
            missing_values = None
            if check_values:
                missing_values = check_missing_values(data, check_values)
            
            # Process according to value mapping
            processed = apply_value_mapping(data, value_mapping)
            
            # Normalize to 8-bit if input was 16-bit
            if is_16bit:
                processed = normalize_16bit_to_8bit(processed)
                print(f"Normalized 16-bit image to 8-bit: {os.path.basename(input_path)}")
            
            # Save with appropriate configuration
            tifffile.imwrite(
                output_path,
                processed,
                photometric='minisblack',
                compression='lzw',
                metadata={'axes': tif.series[0].axes}
            )
        
        return True, input_path, missing_values
    except Exception as e:
        return False, f"{input_path}: {str(e)}", None

def process_file_wrapper(args):
    """Wrapper function for multiprocessing"""
    return process_tiff(*args)

def batch_process_folder(input_folder, output_folder, value_mapping, check_values=None, num_workers=None):
    """
    Process all TIFF files in a folder using multiprocessing
    
    Parameters:
    input_folder (str): Path to folder containing input TIFF files
    output_folder (str): Path to folder where processed files will be saved
    value_mapping (dict): Dictionary specifying {old_value: new_value} mappings
    check_values (list): List of values to check for in each image
    num_workers (int): Number of worker processes to use (defaults to CPU count - 1)
    """
    # Validate value mapping
    if len(set(value_mapping.values())) != len(value_mapping):
        raise ValueError("Multiple source values map to the same target value. This will cause data loss!")
        
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Get all TIFF files in input folder
    tiff_files = [f for f in os.listdir(input_folder) 
                  if f.lower().endswith(('.tif', '.tiff'))]
    
    # Prepare arguments for multiprocessing
    args_list = [
        (
            os.path.join(input_folder, filename),
            os.path.join(output_folder, filename),
            value_mapping,
            check_values
        )
        for filename in tiff_files
    ]
    
    # Set number of workers
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)  # Leave one CPU free
    
    # Process files in parallel with progress bar
    print(f"Processing {len(tiff_files)} files using {num_workers} workers...")
    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_file_wrapper, args_list),
            total=len(args_list),
            desc="Processing TIFF files"
        ))
    
    # Report results
    successes = [r for r in results if r[0]]
    failures = [r for r in results if not r[0]]
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {len(successes)} files")
    
    # Report files with missing values
    if check_values:
        files_with_missing_values = [(os.path.basename(path), missing) 
                                   for success, path, missing in successes 
                                   if missing]
        if files_with_missing_values:
            print("\nFiles missing specified values:")
            for filename, missing_values in files_with_missing_values:
                print(f"- {filename}: missing values {missing_values}")
    
    if failures:
        print("\nErrors occurred while processing the following files:")
        for _, error_msg, _ in failures:
            print(f"- {error_msg}")

if __name__ == "__main__":
    # Define your input and output folders
    input_folder = r"E:\nnunet\train_results\addtl_train_data\fix"
    output_folder = r"E:\nnunet\train_results\addtl_train_data"
    num_workers = 16
    
    # Value mapping is no longer needed but we'll pass an empty dict for compatibility
    value_mapping = {}
    
    # Run the processing
    batch_process_folder(input_folder, output_folder, value_mapping, check_values=None, num_workers=num_workers)