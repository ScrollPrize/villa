import os
import zarr
import numpy as np
import tifffile
import concurrent.futures
from tqdm import tqdm
import multiprocessing
from functools import partial
import json
import signal
import sys

class GracefulExit:
    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, *args):
        print("\nReceived termination signal. Finishing current batch before stopping...")
        self.kill_now = True

def get_progress_file(output_dir):
    """Return the path to the progress tracking file."""
    return os.path.join(output_dir, '.processing_progress.json')

def load_progress(output_dir):
    """Load progress from the progress file."""
    progress_file = get_progress_file(output_dir)
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return json.load(f)
    return {'completed_batches': [], 'total_slices': 0}

def save_progress(output_dir, completed_batches, total_slices):
    """Save progress to the progress file."""
    progress_file = get_progress_file(output_dir)
    with open(progress_file, 'w') as f:
        json.dump({
            'completed_batches': completed_batches,
            'total_slices': total_slices
        }, f)

def process_batch(args):
    """Process a batch of slices at once for better efficiency."""
    z0, start_idx, end_idx, axis, output_dir, kill_signal = args
    
    try:
        # Check if any files in this batch already exist
        all_exist = True
        for idx in range(start_idx, end_idx):
            output_path = os.path.join(output_dir, f'{str(idx).zfill(5)}.tif')
            if not os.path.exists(output_path):
                all_exist = False
                break
                
        if all_exist:
            return start_idx, end_idx, True  # Skip this batch
            
        # Read the entire batch at once
        if axis == 'z':
            batch = z0[start_idx:end_idx, :, :]
        elif axis == 'y':
            batch = z0[:, start_idx:end_idx, :]
        elif axis == 'x':
            batch = z0[:, :, start_idx:end_idx]
            
        # Process and save each slice in the batch
        for i, idx in enumerate(range(start_idx, end_idx)):
            if kill_signal.kill_now:
                return start_idx, end_idx, False
                
            output_path = os.path.join(output_dir, f'{str(idx).zfill(5)}.tif')
            if not os.path.exists(output_path):
                slice_data = batch[i] if axis == 'z' else (
                    batch[:, i, :] if axis == 'y' else batch[:, :, i]
                )
                
                # Save directly with compression - no scaling needed as data is already uint8
                tifffile.imwrite(
                    output_path,
                    slice_data,
                    compression='zlib'
                )
                
        return start_idx, end_idx, True
        
    except Exception as e:
        print(f"Error processing batch {start_idx}-{end_idx}: {str(e)}")
        return start_idx, end_idx, False

def process_zarr(zarr_path, output_dir, axis='z', batch_size=50, n_workers=None):
    """Main processing function with batching, multiprocessing, and resumability."""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize graceful exit handler
    kill_signal = GracefulExit()
    
    # Open the Zarr array
    z = zarr.open(zarr_path, mode='r')
    print(f"Input data type: {z.dtype}")
    
    # Determine total slices based on axis
    total_slices = z.shape[0] if axis == 'z' else (
        z.shape[1] if axis == 'y' else z.shape[2]
    )
    
    # Load progress from previous run
    progress = load_progress(output_dir)
    completed_batches = progress['completed_batches']
    
    # Create batches
    batch_starts = range(0, total_slices, batch_size)
    batch_ends = [min(start + batch_size, total_slices) for start in batch_starts]
    batches = []
    
    # Only include batches that weren't completed in previous runs
    for start, end in zip(batch_starts, batch_ends):
        if not any(cb['start'] == start and cb['end'] == end for cb in completed_batches):
            batches.append((z, start, end, axis, output_dir, kill_signal))
    
    if not batches:
        print("All batches have been processed. Nothing to do.")
        return
    
    # Use provided worker count or default to CPU count minus 1
    if n_workers is None:
        n_workers = max(1, multiprocessing.cpu_count() - 1)
    
    # Process batches using multiprocessing
    with multiprocessing.Pool(n_workers) as pool:
        try:
            for result in tqdm(
                pool.imap_unordered(process_batch, batches),
                total=len(batches),
                desc="Processing batches"
            ):
                start_idx, end_idx, success = result
                if success:
                    completed_batches.append({
                        'start': start_idx,
                        'end': end_idx
                    })
                    # Save progress after each successful batch
                    save_progress(output_dir, completed_batches, total_slices)
                
                if kill_signal.kill_now:
                    print("\nGracefully shutting down...")
                    pool.terminate()
                    break
                    
        except KeyboardInterrupt:
            print("\nInterrupt received. Gracefully shutting down...")
            pool.terminate()
    
    # Print array information
    print(f"\nZarr array info for: {zarr_path}")
    print(f"Shape: {z.shape}")
    print(f"Chunks: {z.chunks}")
    print(f"Data type: {z.dtype}")
    print(f"Number of dimensions: {z.ndim}")
    print(f"Read-only: {z.read_only}")
    
    # Print progress
    total_processed = sum(1 for batch in completed_batches for _ in range(batch['start'], batch['end']))
    print(f"\nProgress: {total_processed}/{total_slices} slices processed")

if __name__ == '__main__':
    # Configuration
    zarr_path = r"E:\scroll1.volpkg\036_043_044_ensemble.zarr"
    output_dir = r"E:\scroll1.volpkg\036_043_044_ensemble_otsu_sliced"
    
    # Run processing with 16 workers
    process_zarr(zarr_path, output_dir, axis='z', batch_size=10, n_workers=12)