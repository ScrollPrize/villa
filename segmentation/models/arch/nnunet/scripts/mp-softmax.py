import numpy as np
import pickle
from typing import Union
import os
from PIL import Image
from pathlib import Path
import glob
from multiprocessing import Pool, cpu_count, Value
from tqdm import tqdm
import warnings
import tifffile 

def normalize_to_uint8(arr):
    """Normalize array to 0-255 range for 8-bit image saving"""
    arr_min, arr_max = arr.min(), arr.max()
    return np.zeros_like(arr, dtype=np.uint8) if arr_max == arr_min else \
           ((arr - arr_min) * 255 / (arr_max - arr_min)).astype(np.uint8)

def save_softmax_tiff_from_softmax(segmentation_softmax: Union[str, np.ndarray], 
                                 out_fname: str,
                                 channel: int = 1,
                                 postprocess_fn: callable = None,
                                 postprocess_args: tuple = None,
                                 non_postprocessed_fname: str = None):
    """
    Saves softmax segmentation as an 8-bit multipage TIFF file using PIL.
    
    Args:
        segmentation_softmax: Path to .npz file or numpy array containing softmax probabilities
        out_fname: Output path for the TIFF file
        channel: Which channel to save (default=1 for foreground)
        postprocess_fn: Optional function to post-process the data
        postprocess_args: Arguments for post-processing function
        non_postprocessed_fname: Optional path to save non-postprocessed version
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(out_fname), exist_ok=True)
        if non_postprocessed_fname:
            os.makedirs(os.path.dirname(non_postprocessed_fname), exist_ok=True)

        # Load the segmentation data
        if isinstance(segmentation_softmax, str):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                npz_data = np.load(segmentation_softmax)
                if 'probabilities' in npz_data:
                    seg_old_spacing = npz_data['probabilities']
                else:
                    raise KeyError("Could not find probabilities data in npz file")
        else:
            seg_old_spacing = segmentation_softmax

        # Validate channel selection
        if channel >= seg_old_spacing.shape[0]:
            raise ValueError(f"Selected channel {channel} is out of range. Array has {seg_old_spacing.shape[0]} channels.")

        # Load properties dictionary
        if isinstance(segmentation_softmax, str):
            pkl_path = segmentation_softmax[:-4] + ".pkl"
            if os.path.exists(pkl_path):
                with open(pkl_path, "rb") as file_to_read:
                    properties_dict = pickle.load(file_to_read)
            else:
                properties_dict = {}
        else:
            properties_dict = {}

        # Handle bounding box
        bbox = properties_dict.get('crop_bbox')
        if bbox is not None:
            bbox = bbox.copy()
            bbox.insert(0, [0, seg_old_spacing.shape[0] + 1])
            
            shape_original_before_cropping = list(properties_dict.get('original_size_of_raw_data', []))
            shape_original_before_cropping.insert(0, seg_old_spacing.shape[0])
            shape_original_before_cropping = np.array(shape_original_before_cropping)
            
            seg_old_size = np.zeros(shape_original_before_cropping)
            for c in range(3):
                bbox[c][1] = np.min((bbox[c][0] + seg_old_spacing.shape[c], shape_original_before_cropping[c]))
            
            seg_old_size[
                bbox[0][0]:bbox[0][1],
                bbox[1][0]:bbox[1][1],
                bbox[2][0]:bbox[2][1],
                bbox[3][0]:bbox[3][1]
            ] = seg_old_spacing
        else:
            seg_old_size = seg_old_spacing

        if postprocess_fn is not None:
            seg_old_size_postprocessed = postprocess_fn(np.copy(seg_old_size), *postprocess_args)
        else:
            seg_old_size_postprocessed = seg_old_size

        # Vectorized normalization on the whole channel at once
        channel_data = seg_old_size_postprocessed[channel]
        normalized = normalize_to_uint8(channel_data)
        tifffile.imwrite(out_fname, normalized, photometric='minisblack')

        # Same for non-postprocessed version
        if non_postprocessed_fname is not None and postprocess_fn is not None:
            channel_data_raw = seg_old_size[channel]
            normalized_raw = normalize_to_uint8(channel_data_raw)
            tifffile.imwrite(non_postprocessed_fname, normalized_raw, photometric='minisblack')

        return True

    except Exception as e:
        print(f"\nError processing {out_fname}:")
        print(f"  {str(e)}")
        return False

def process_single_file(args):
    """Process a single file and return success status"""
    try:
        npz_file, output_dir, channel, postprocess_fn, postprocess_args = args
        base_name = Path(npz_file).stem
        out_fname = output_dir / f"{base_name}.tif"
        non_postprocessed_fname = output_dir / f"{base_name}_raw.tif"
        
        # Skip if both output files already exist
        if out_fname.exists():
            if postprocess_fn is None or (non_postprocessed_fname.exists()):
                return True
        
        result = save_softmax_tiff_from_softmax(
            str(npz_file),
            str(out_fname),
            channel=channel,
            postprocess_fn=postprocess_fn,
            postprocess_args=postprocess_args,
            non_postprocessed_fname=str(non_postprocessed_fname)
        )
        
        return result
    except Exception as e:
        print(f"Error processing {npz_file}: {str(e)}")
        return False

def process_directory(input_dir: str, output_dir: str, 
                     channel: int = 1,
                     postprocess_fn=None, 
                     postprocess_args=None, 
                     num_processes: int = None):
    """
    Process all .npz files in the input directory and save corresponding TIFF files.
    Only processes files that don't have corresponding TIFF outputs.
    
    Args:
        input_dir: Input directory containing .npz files
        output_dir: Output directory for TIFF files
        channel: Which channel to save (default=1 for foreground)
        postprocess_fn: Optional function to post-process the data
        postprocess_args: Arguments for post-processing function
        num_processes: Number of processes to use for parallel processing
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all .npz files and check their corresponding outputs
    npz_files = []
    already_processed = 0
    missing_pkl = 0
    needs_processing = 0

    for npz_file in input_dir.glob("*.npz"):
        base_name = npz_file.stem
        out_tiff = output_dir / f"{base_name}.tif"
        out_raw_tiff = output_dir / f"{base_name}_raw.tif"
        
        # Check for required .pkl file
        if not npz_file.with_suffix('.pkl').exists():
            print(f"Skipping {npz_file.name} - no matching .pkl file")
            missing_pkl += 1
            continue

        # Check if processing is needed
        if postprocess_fn is not None:
            # If postprocessing is enabled, we need both files
            if out_tiff.exists() and out_raw_tiff.exists():
                print(f"Skipping {npz_file.name} - both processed and raw outputs exist")
                already_processed += 1
                continue
        else:
            # If no postprocessing, we only need the main output
            if out_tiff.exists():
                print(f"Skipping {npz_file.name} - output exists")
                already_processed += 1
                continue

        print(f"Will process {npz_file.name} - output missing")
        npz_files.append(npz_file)
        needs_processing += 1

    # Summary before processing
    print(f"\nFound {needs_processing + already_processed + missing_pkl} total .npz files")
    print(f"  - {needs_processing} files need processing")
    print(f"  - {already_processed} files already processed")
    print(f"  - {missing_pkl} files skipped due to missing .pkl files")

    if not npz_files:
        print("No files need processing. Done!")
        return

    # Determine number of processes
    if num_processes is None:
        num_processes = max(1, cpu_count() - 1)
    num_processes = min(num_processes, len(npz_files))
    print(f"\nUsing {num_processes} processes")

    # Prepare arguments for multiprocessing
    process_args = [(npz_file, output_dir, channel, postprocess_fn, postprocess_args) 
                   for npz_file in npz_files]

    # Process files in parallel with progress bar
    with Pool(num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_single_file, process_args),
            total=len(npz_files),
            desc="Processing files",
            unit="file"
        ))

    # Report final summary
    successful = sum(1 for r in results if r)
    failed = len(npz_files) - successful
    print(f"\nProcessing complete!")
    print(f"  Successfully processed: {successful}/{len(npz_files)} files")
    if failed > 0:
        print(f"  Failed to process: {failed} files")
    print(f"  Previously processed: {already_processed} files")
    if missing_pkl > 0:
        print(f"  Skipped (missing .pkl): {missing_pkl} files")

if __name__ == "__main__":
    # Directory paths - update these to your specific paths
    input_dir = "/mnt/raid_hdd/scrolls/s1/man-sfc-skeleton_withprobs"
    output_dir = "/mnt/raid_nvme/datasets/ensemble/036_skel_sm"
    
    print("Starting batch processing...")
    
    try:
        # Process all files in the directory
        process_directory(
            input_dir=input_dir,
            output_dir=output_dir,
            channel=1,  # Specify which channel to save (0 for background, 1 for foreground)
            postprocess_fn=None,  # Remove postprocessing if not needed
            postprocess_args=None,
            num_processes=4
        )
        
    except Exception as e:
        print(f"\nAn error occurred during batch processing:")
        print(f"  {str(e)}")