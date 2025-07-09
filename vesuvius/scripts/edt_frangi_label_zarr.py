#!/usr/bin/env python
import os
import argparse
import multiprocessing
import itertools
import json
from datetime import datetime

import numpy as np
import zarr
import tifffile
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt, gaussian_filter, convolve, map_coordinates, label
from scipy import ndimage
from numpy import linalg as LA
from concurrent.futures import ProcessPoolExecutor, as_completed
import random


# ---------------------------
# Progress Tracking
# ---------------------------
class ProgressTracker:
    def __init__(self, output_zarr_path, total_chunks):
        self.progress_file = output_zarr_path + '.progress'
        self.total_chunks = total_chunks
        self.processed_chunks = set()
        self.failed_chunks = {}
        self.start_time = datetime.now()
        
        # Load existing progress if available
        if os.path.exists(self.progress_file):
            self.load_progress()
    
    def load_progress(self):
        try:
            with open(self.progress_file, 'r') as f:
                data = json.load(f)
                self.processed_chunks = set(tuple(tuple(x) for x in chunk) for chunk in data.get('processed', []))
                self.failed_chunks = {tuple(tuple(x) for x in eval(k)): v for k, v in data.get('failed', {}).items()}
                print(f"Resuming from checkpoint: {len(self.processed_chunks)}/{self.total_chunks} chunks completed")
        except Exception as e:
            print(f"Warning: Could not load progress file: {e}")
    
    def save_progress(self):
        data = {
            'processed': [[[s.start, s.stop] for s in chunk] for chunk in self.processed_chunks],
            'failed': {str(k): v for k, v in self.failed_chunks.items()},
            'total_chunks': self.total_chunks,
            'start_time': self.start_time.isoformat(),
            'last_update': datetime.now().isoformat()
        }
        
        # Write to temporary file first, then rename (atomic operation)
        temp_file = self.progress_file + '.tmp'
        with open(temp_file, 'w') as f:
            json.dump(data, f, indent=2)
        os.rename(temp_file, self.progress_file)
    
    def mark_processed(self, chunk_slices):
        chunk_key = tuple(chunk_slices)
        self.processed_chunks.add(chunk_key)
        # Save every 10 chunks or so
        if len(self.processed_chunks) % 10 == 0:
            self.save_progress()
    
    def mark_failed(self, chunk_slices, error):
        chunk_key = tuple(chunk_slices)
        self.failed_chunks[chunk_key] = {
            'error': str(error),
            'timestamp': datetime.now().isoformat()
        }
    
    def is_processed(self, chunk_slices):
        chunk_key = tuple(chunk_slices)
        return chunk_key in self.processed_chunks
    
    def get_summary(self):
        elapsed = datetime.now() - self.start_time
        total_attempted = len(self.processed_chunks) + len(self.failed_chunks)
        return {
            'processed': len(self.processed_chunks),
            'failed': len(self.failed_chunks),
            'remaining': self.total_chunks - len(self.processed_chunks),
            'elapsed_time': str(elapsed),
            'success_rate': len(self.processed_chunks) / max(total_attempted, 1)
        }
    
    def cleanup(self):
        """Remove progress file after successful completion"""
        if os.path.exists(self.progress_file):
            os.remove(self.progress_file)
            print("Progress file removed after successful completion")


# ---------------------------
# Utility and Processing Functions (from original script)
# ---------------------------
def divide_nonzero(array1, array2, eps=1e-10):
    denominator = np.copy(array2)
    denominator[denominator == 0] = eps
    return np.divide(array1, denominator)


def normalize(volume):
    minim = np.min(volume)
    maxim = np.max(volume)
    if maxim - minim == 0:
        return volume
    volume = volume - minim
    volume = volume / (maxim - minim)
    return volume


def hessian(volume, gauss_sigma=2, sigma=6):
    volume_smoothed = gaussian_filter(volume, sigma=gauss_sigma)
    volume_smoothed = normalize(volume_smoothed)

    joint_hessian = np.zeros((volume.shape[0], volume.shape[1], volume.shape[2], 3, 3), dtype=float)

    Dz = np.gradient(volume_smoothed, axis=0, edge_order=2)
    joint_hessian[:, :, :, 2, 2] = np.gradient(Dz, axis=0, edge_order=2)
    del Dz

    Dy = np.gradient(volume_smoothed, axis=1, edge_order=2)
    joint_hessian[:, :, :, 1, 1] = np.gradient(Dy, axis=1, edge_order=2)
    joint_hessian[:, :, :, 1, 2] = np.gradient(Dy, axis=0, edge_order=2)
    del Dy

    Dx = np.gradient(volume_smoothed, axis=2, edge_order=2)
    joint_hessian[:, :, :, 0, 0] = np.gradient(Dx, axis=2, edge_order=2)
    joint_hessian[:, :, :, 0, 1] = np.gradient(Dx, axis=1, edge_order=2)
    joint_hessian[:, :, :, 0, 2] = np.gradient(Dx, axis=0, edge_order=2)
    del Dx

    joint_hessian = joint_hessian * (sigma ** 2)
    zero_mask = np.trace(joint_hessian, axis1=3, axis2=4) == 0
    return joint_hessian, zero_mask


def detect_ridges(volume, gamma=1.5, beta1=0.5, beta2=0.5, gauss_sigma=2, sigma=6):
    joint_hessian, zero_mask = hessian(volume, gauss_sigma, sigma)
    eigvals = LA.eigvalsh(joint_hessian, 'U')
    idxs = np.argsort(np.abs(eigvals), axis=-1)
    eigvals = np.take_along_axis(eigvals, idxs, axis=-1)
    eigvals[zero_mask, :] = 0

    L1 = np.abs(eigvals[:, :, :, 0])
    L2 = np.abs(eigvals[:, :, :, 1])
    L3 = eigvals[:, :, :, 2]
    L3abs = np.abs(L3)

    S = np.sqrt(np.square(eigvals).sum(axis=-1))
    background_term = 1 - np.exp(-0.5 * np.square(S / gamma))

    Ra = divide_nonzero(L2, L3abs)
    planar_term = np.exp(-0.5 * np.square(Ra / beta1))

    Rb = divide_nonzero(L1, np.sqrt(L2 * L3abs))
    blob_term = np.exp(-0.5 * np.square(Rb / beta2))

    ridges = background_term * planar_term * blob_term
    ridges[L3 > 0] = 0
    return ridges


def dilate_by_inverse_edt(binary_volume, dilation_distance):
    eps = 1e-6
    edt = distance_transform_edt(1 - binary_volume)
    inv_edt = 1.0 / (edt + eps)
    threshold = 1.0 / dilation_distance
    dilated = (inv_edt > threshold).astype(np.uint8)
    return dilated


def remove_small_components(binary_volume, min_size):
    """Remove connected components smaller than min_size voxels."""
    if min_size <= 0:
        return binary_volume
    
    # Label connected components
    labeled_array, num_features = label(binary_volume)
    
    # Count voxels in each component
    component_sizes = np.bincount(labeled_array.ravel())
    
    # Create mask for components to keep (larger than min_size)
    # Note: component 0 is the background, so we skip it
    keep_mask = np.zeros_like(labeled_array, dtype=bool)
    for i in range(1, len(component_sizes)):
        if component_sizes[i] >= min_size:
            keep_mask[labeled_array == i] = True
    
    return keep_mask.astype(np.uint8)


# ---------------------------
# Zarr-specific Functions
# ---------------------------
def get_chunk_slices(shape, chunks):
    """Generate slices for all chunks in the array."""
    ndim = len(shape)
    
    # Handle single chunk size or tuple of chunk sizes
    if isinstance(chunks, int):
        chunks = (chunks,) * ndim
    elif len(chunks) != ndim:
        raise ValueError(f"Chunk size must be an integer or tuple of length {ndim}")
    
    ranges = []
    for i in range(ndim):
        ranges.append(range(0, shape[i], chunks[i]))
    
    for indices in itertools.product(*ranges):
        slices = []
        for i, start in enumerate(indices):
            stop = min(start + chunks[i], shape[i])
            slices.append(slice(start, stop))
        yield tuple(slices)


def get_expanded_slices(chunk_slices, shape, expansion_factor=2):
    """Get expanded slices for loading overlapping regions."""
    if expansion_factor == 1:
        return chunk_slices
    
    expanded_slices = []
    for i, s in enumerate(chunk_slices):
        chunk_size = s.stop - s.start
        expanded_size = int(chunk_size * expansion_factor)
        
        # Calculate expansion on each side
        expansion = (expanded_size - chunk_size) // 2
        
        # Compute new bounds with clipping to array boundaries
        new_start = max(0, s.start - expansion)
        new_stop = min(shape[i], s.stop + expansion)
        
        expanded_slices.append(slice(new_start, new_stop))
    
    return tuple(expanded_slices)


def get_trim_slices(chunk_slices, expanded_slices):
    """Get slices to trim the processed region back to original chunk size."""
    if chunk_slices == expanded_slices:
        # No expansion, return full slices
        return tuple(slice(None) for _ in chunk_slices)
    
    trim_slices = []
    for chunk_s, expanded_s in zip(chunk_slices, expanded_slices):
        # Calculate where the original chunk starts within the expanded region
        offset = chunk_s.start - expanded_s.start
        size = chunk_s.stop - chunk_s.start
        trim_slices.append(slice(offset, offset + size))
    return tuple(trim_slices)


def process_chunk(input_zarr_path, chunk_slices, shape, chunks, dilation_distance, 
                  ridge_threshold, ridge_params, output_mode='binary', 
                  expansion_factor=2, min_component_size=0, return_input=False):
    """Process a single chunk with optional overlapping regions."""
    try:
        # Open the zarr array for reading
        input_zarr = zarr.open(input_zarr_path, mode='r')
        
        # Get expanded slices
        expanded_slices = get_expanded_slices(chunk_slices, shape, expansion_factor)
        
        # Load the expanded region
        volume = input_zarr[expanded_slices]
        
        # Also get the original input if requested (for saving samples)
        input_chunk = None
        if return_input:
            input_chunk = input_zarr[chunk_slices]
        
        # Process the volume
        binary_volume = (volume > 0).astype(np.uint8)
        
        # Apply connected component filtering to input if requested
        if min_component_size > 0:
            binary_volume = remove_small_components(binary_volume, min_component_size)
        
        dilated_volume = dilate_by_inverse_edt(binary_volume, dilation_distance)
        dilated_float = dilated_volume.astype(np.float32)
        
        # Use ridge parameters
        ridges = detect_ridges(dilated_float, 
                             gamma=ridge_params['gamma'],
                             beta1=ridge_params['beta1'],
                             beta2=ridge_params['beta2'],
                             gauss_sigma=ridge_params['gauss_sigma'],
                             sigma=ridge_params['hessian_sigma'])
        
        # Handle output mode
        if output_mode == 'continuous':
            output_data = ridges.astype(np.float32)
        else:
            output_data = (ridges > ridge_threshold).astype(np.uint8)
        
        # Get trim slices to extract the original chunk size
        trim_slices = get_trim_slices(chunk_slices, expanded_slices)
        trimmed_result = output_data[trim_slices]
        
        if return_input:
            return chunk_slices, trimmed_result, input_chunk, True
        else:
            return chunk_slices, trimmed_result, True
    except Exception as e:
        if return_input:
            return chunk_slices, None, None, False, str(e)
        else:
            return chunk_slices, None, False, str(e)


def get_center_chunks(chunk_slices_list, shape, chunks, center_fraction=0.75):
    """Get chunks from the center portion of the volume."""
    center_chunks = []
    
    for chunk_slices in chunk_slices_list:
        is_center = True
        for i, s in enumerate(chunk_slices):
            # Calculate center bounds
            dim_size = shape[i]
            margin = (1 - center_fraction) * dim_size / 2
            center_start = margin
            center_end = dim_size - margin
            
            # Check if chunk is within center bounds
            chunk_center = (s.start + s.stop) / 2
            if chunk_center < center_start or chunk_center > center_end:
                is_center = False
                break
        
        if is_center:
            center_chunks.append(chunk_slices)
    
    return center_chunks


def sample_and_save_results(args):
    """Process a sample of chunks and save as TIFF files."""
    # Open input zarr to get metadata
    input_zarr = zarr.open(args.input_zarr, mode='r')
    shape = input_zarr.shape
    
    # Use custom chunk size if provided, otherwise use input zarr's chunks
    if args.chunk_size:
        chunks = args.chunk_size
        print(f"Using custom chunk size: {chunks}")
    else:
        chunks = input_zarr.chunks
    
    print(f"Input zarr shape: {shape}")
    print(f"Input zarr chunks: {input_zarr.chunks}")
    print(f"Processing chunk size: {chunks}")
    print(f"Expansion factor: {args.expansion_factor}")
    
    # Generate all chunk slices
    chunk_slices_list = list(get_chunk_slices(shape, chunks))
    
    # Get chunks from center 75%
    center_chunks = get_center_chunks(chunk_slices_list, shape, chunks)
    print(f"Total chunks in center 75%: {len(center_chunks)}")
    
    # Sample random chunks
    num_samples = min(args.sample_result, len(center_chunks))
    sampled_chunks = random.sample(center_chunks, num_samples)
    print(f"Processing {num_samples} sample chunks...")
    
    # Create output directory
    output_dir = args.output_zarr + "_samples"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create ridge parameters dict
    ridge_params = {
        'gamma': args.gamma,
        'beta1': args.beta1,
        'beta2': args.beta2,
        'gauss_sigma': args.gauss_sigma,
        'hessian_sigma': args.hessian_sigma
    }
    
    # Process samples
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = []
        for idx, chunk_slices in enumerate(sampled_chunks):
            future = executor.submit(
                process_chunk,
                args.input_zarr,
                chunk_slices,
                shape,
                chunks,
                args.dilation_distance,
                args.ridge_threshold,
                ridge_params,
                args.output_mode,
                args.expansion_factor,
                args.min_component_size,
                return_input=True
            )
            futures.append((idx, chunk_slices, future))
        
        # Process results
        with tqdm(total=num_samples, desc="Processing samples") as pbar:
            for idx, chunk_slices, future in futures:
                result = future.result()
                
                if result[3]:  # Success
                    _, output_chunk, input_chunk, _ = result
                    
                    # Generate filename with chunk coordinates
                    coords = "_".join([f"{s.start}-{s.stop}" for s in chunk_slices])
                    
                    # Save input chunk
                    input_path = os.path.join(output_dir, f"sample_{idx:03d}_input_{coords}.tif")
                    tifffile.imwrite(input_path, input_chunk)
                    
                    # Save output chunk
                    output_path = os.path.join(output_dir, f"sample_{idx:03d}_output_{coords}.tif")
                    tifffile.imwrite(output_path, output_chunk)
                    
                    print(f"\nSaved sample {idx}: {coords}")
                else:
                    _, _, _, _, error = result
                    print(f"\nError processing sample {idx}: {error}")
                
                pbar.update(1)
    
    print(f"\nSample results saved to: {output_dir}")


# ---------------------------
# Main Routine for Zarr Processing
# ---------------------------
def main():
    """
    Main function for processing zarr arrays with EDT dilation and Frangi ridge detection.
    
    Ridge Detection Parameters Explained:
    ------------------------------------
    
    --gamma (default: 1.5)
        Controls sensitivity to structure strength/contrast. Lower values make the filter
        more sensitive to low-contrast structures, while higher values suppress weak structures.
        - Decrease (<1.5): Detects fainter/weaker curvilinear structures
        - Increase (>1.5): Only detects high-contrast, well-defined structures
        
    --beta1 (default: 0.5)  
        Controls sensitivity to sheet-like vs. tube-like structures. This parameter affects
        the "planar term" that distinguishes between different geometric shapes.
        - Decrease (<0.5): More selective for ideal tube-like structures
        - Increase (>0.5): More permissive, accepts sheet-like structures
        
    --beta2 (default: 0.5)
        Controls blob suppression. This parameter helps distinguish elongated structures
        from blob-like (spherical) structures.
        - Decrease (<0.5): Stronger suppression of blob-like structures
        - Increase (>0.5): More permissive to blob-like structures
        
    --gauss-sigma (default: 2)
        Pre-smoothing sigma applied before Hessian computation. Controls noise suppression
        and small detail preservation.
        - Decrease (<2): Less smoothing, preserves fine details but more sensitive to noise
        - Increase (>2): More smoothing, reduces noise but may lose fine details
        
    --hessian-sigma (default: 6)
        Scale parameter for Hessian computation. Should match the expected width/scale of
        the curvilinear structures you want to detect.
        - Decrease (<6): Optimized for thinner structures
        - Increase (>6): Optimized for thicker structures
        
    Tips for Parameter Tuning:
    --------------------------
    1. Start by adjusting hessian-sigma to match your structure width
    2. Use --sample-result to quickly test parameters on representative chunks
    3. If getting too much noise, increase gauss-sigma or gamma
    4. If missing thin structures, decrease hessian-sigma
    5. If getting unwanted blob-like detections, decrease beta2
    6. For continuous output mode, examine the raw response values to better
       understand how parameters affect detection strength
    7. If getting speckle noise (small isolated voxels), use --min-component-size
       to filter out connected components smaller than the specified number of voxels
    """
    parser = argparse.ArgumentParser(
        description="Process zarr arrays chunkwise with inverse EDT dilation and custom ridge detection."
    )
    parser.add_argument("input_zarr", help="Path to input zarr array.")
    parser.add_argument("output_zarr", help="Path to output zarr array.")
    parser.add_argument("--dilation-distance", type=float, default=3, help="Dilation distance (in voxels).")
    parser.add_argument("--ridge-threshold", type=float, default=0.5,
                        help="Threshold for binarizing the ridge detection.")
    
    # Ridge detection parameters
    parser.add_argument("--gamma", type=float, default=1.5,
                        help="Background term parameter for ridge detection (controls sensitivity to structure strength).")
    parser.add_argument("--beta1", type=float, default=0.5,
                        help="Planar term parameter (controls sheet-like structure detection).")
    parser.add_argument("--beta2", type=float, default=0.5,
                        help="Blob term parameter (controls blob-like structure suppression).")
    parser.add_argument("--gauss-sigma", type=float, default=2,
                        help="Gaussian smoothing sigma before Hessian computation.")
    parser.add_argument("--hessian-sigma", type=float, default=6,
                        help="Scale parameter for Hessian computation.")
    
    # Output options
    parser.add_argument("--output-mode", choices=['binary', 'continuous'], default='binary',
                        help="Output mode: 'binary' for thresholded output, 'continuous' for ridge response values.")
    parser.add_argument("--min-component-size", type=int, default=0,
                        help="Minimum size (in voxels) for connected components in the input labels. Components smaller than this will be removed BEFORE dilation and ridge detection. Default: 0 (no filtering)")
    
    # Processing options
    parser.add_argument("--expansion-factor", type=float, default=2,
                        help="Expansion factor for chunk overlap. 1 = no overlap, 2 = double size. Default: 2")
    parser.add_argument("--num-workers", type=int, default=multiprocessing.cpu_count(),
                        help="Number of worker processes.")
    parser.add_argument("--sample-result", type=int, default=None,
                        help="Process and save only a sample of chunks (default: 8). This option will not process the entire zarr.")
    parser.add_argument("--chunk-size", type=str, default=None,
                        help="Chunk size for processing. Can be a single integer (e.g., 256) or comma-separated values for each dimension (e.g., 128,256,256). If not specified, uses input zarr's chunk size.")
    
    # Progress tracking
    parser.add_argument("--retry-failed", action="store_true",
                        help="Retry previously failed chunks")
    
    args = parser.parse_args()
    
    # Parse chunk size if provided
    if args.chunk_size:
        if ',' in args.chunk_size:
            args.chunk_size = tuple(int(x) for x in args.chunk_size.split(','))
        else:
            args.chunk_size = int(args.chunk_size)
    
    # If sample-result is requested, only process samples
    if args.sample_result is not None:
        # Default to 8 samples if 0 or no value specified
        if args.sample_result == 0:
            args.sample_result = 8
        sample_and_save_results(args)
        return

    # Open input zarr to get metadata
    input_zarr = zarr.open(args.input_zarr, mode='r')
    shape = input_zarr.shape
    
    # Determine output dtype based on mode
    if args.output_mode == 'continuous':
        dtype = np.float32
        print("Output mode: continuous (float32 ridge response values)")
    else:
        dtype = np.uint8
        print("Output mode: binary (thresholded)")
    
    # Use custom chunk size if provided, otherwise use input zarr's chunks
    if args.chunk_size:
        chunks = args.chunk_size
        # Convert single value to tuple if needed
        if isinstance(chunks, int):
            chunks = (chunks,) * len(shape)
        print(f"Using custom chunk size: {chunks}")
    else:
        chunks = input_zarr.chunks
    
    print(f"Input zarr shape: {shape}")
    print(f"Input zarr chunks: {input_zarr.chunks}")
    print(f"Processing/output chunk size: {chunks}")
    print(f"Expansion factor: {args.expansion_factor}")
    
    # Print ridge detection parameters
    print(f"\nRidge detection parameters:")
    print(f"  gamma: {args.gamma}")
    print(f"  beta1: {args.beta1}")
    print(f"  beta2: {args.beta2}")
    print(f"  gauss_sigma: {args.gauss_sigma}")
    print(f"  hessian_sigma: {args.hessian_sigma}")
    
    # Print component filtering info if enabled
    if args.min_component_size > 0:
        print(f"\nInput filtering enabled: removing components < {args.min_component_size} voxels from input labels")
    
    # Create output zarr with specified chunks
    output_zarr = zarr.open(args.output_zarr, mode='w', shape=shape, chunks=chunks, dtype=dtype)
    
    # Generate all chunk slices
    chunk_slices_list = list(get_chunk_slices(shape, chunks))
    total_chunks = len(chunk_slices_list)
    
    # Initialize progress tracker
    tracker = ProgressTracker(args.output_zarr, total_chunks)
    
    # Filter out already processed chunks
    chunks_to_process = []
    for chunk_slices in chunk_slices_list:
        if not tracker.is_processed(chunk_slices):
            chunks_to_process.append(chunk_slices)
    
    # Handle retry of failed chunks
    if args.retry_failed and tracker.failed_chunks:
        print(f"Retrying {len(tracker.failed_chunks)} previously failed chunks")
        for chunk_key in list(tracker.failed_chunks.keys()):
            # Convert back to slices
            chunk_slices = tuple(slice(start, stop) for start, stop in chunk_key)
            if chunk_slices not in chunks_to_process:
                chunks_to_process.append(chunk_slices)
        # Clear failed chunks from tracker
        tracker.failed_chunks.clear()
    
    print(f"Chunks to process: {len(chunks_to_process)} (skipping {len(chunk_slices_list) - len(chunks_to_process)} already completed)")
    
    # Create ridge parameters dict
    ridge_params = {
        'gamma': args.gamma,
        'beta1': args.beta1,
        'beta2': args.beta2,
        'gauss_sigma': args.gauss_sigma,
        'hessian_sigma': args.hessian_sigma
    }
    
    # Process chunks in parallel
    success_count = 0
    fail_count = 0
    
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        # Submit all tasks
        futures = []
        for chunk_slices in chunks_to_process:
            future = executor.submit(
                process_chunk,
                args.input_zarr,
                chunk_slices,
                shape,
                chunks,
                args.dilation_distance,
                args.ridge_threshold,
                ridge_params,
                args.output_mode,
                args.expansion_factor,
                args.min_component_size
            )
            futures.append((chunk_slices, future))
        
        # Process results as they complete
        with tqdm(total=len(chunks_to_process), desc="Processing chunks") as pbar:
            for chunk_slices, future in futures:
                try:
                    result = future.result()
                    
                    if result[2]:  # Success
                        _, trimmed_result, _ = result
                        output_zarr[chunk_slices] = trimmed_result
                        tracker.mark_processed(chunk_slices)
                        success_count += 1
                    else:  # Failure
                        _, _, _, error = result
                        tracker.mark_failed(chunk_slices, error)
                        print(f"\nError processing chunk {chunk_slices}: {error}")
                        fail_count += 1
                except Exception as e:
                    tracker.mark_failed(chunk_slices, str(e))
                    print(f"\nUnexpected error processing chunk {chunk_slices}: {e}")
                    fail_count += 1
                
                pbar.update(1)
    
    # Final progress save and summary
    tracker.save_progress()
    summary = tracker.get_summary()
    
    print(f"\nProcessing complete:")
    print(f"  Successful chunks: {summary['processed']}")
    print(f"  Failed chunks: {summary['failed']}")
    print(f"  Success rate: {summary['success_rate']:.1%}")
    print(f"  Total time: {summary['elapsed_time']}")
    
    # If all chunks processed successfully, clean up progress file
    if summary['failed'] == 0 and summary['remaining'] == 0:
        tracker.cleanup()
    else:
        print(f"\nProgress saved to: {tracker.progress_file}")
        print("Run the script again to retry failed chunks or continue processing")
    
    # Ensure all data is written
    if hasattr(output_zarr, 'store'):
        output_zarr.store.close()


if __name__ == "__main__":
    main()
