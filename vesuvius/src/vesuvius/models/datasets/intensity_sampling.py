"""
Multiprocessing-enabled intensity property computation for normalization.
"""
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial


def sample_volume_task(task):
    """
    Sample from a single volume (used by multiprocessing pool).
    
    Parameters
    ----------
    task : tuple
        (vol_idx, img_data, shape, num_samples)
        
    Returns
    -------
    tuple
        (vol_idx, sampled_values)
    """
    vol_idx, img_data, shape, num_samples = task
    
    # Memory-efficient sampling
    if hasattr(img_data, 'chunks'):  # Zarr array
        # Sample from zarr array without loading everything
        sampled_values = sample_from_zarr_with_progress(img_data, num_samples, vol_idx)
    else:  # Numpy array
        # For numpy arrays, we can flatten and sample
        flat_data = img_data.flatten()
        indices = np.random.choice(len(flat_data), size=min(num_samples, len(flat_data)), replace=False)
        sampled_values = flat_data[indices].tolist()
    
    return vol_idx, sampled_values


def sample_from_zarr_with_progress(zarr_array, num_samples, vol_idx):
    """
    Sample from zarr array with progress tracking for large arrays.
    
    Parameters
    ----------
    zarr_array : zarr.Array
        The zarr array to sample from
    num_samples : int
        Number of samples to collect
    vol_idx : int
        Volume index for progress display
        
    Returns
    -------
    list
        List of sampled values
    """
    shape = zarr_array.shape
    ndim = len(shape)
    sampled_values = []
    
    # Show progress for large sample counts
    use_progress = num_samples > 10000
    if use_progress:
        pbar = tqdm(total=num_samples, desc=f"Sampling zarr volume {vol_idx}", leave=False)
    
    # Generate random indices in batches for efficiency
    batch_size = min(1000, num_samples)
    
    if ndim == 2:
        h, w = shape
        for i in range(0, num_samples, batch_size):
            batch_count = min(batch_size, num_samples - i)
            ys = np.random.randint(0, h, size=batch_count)
            xs = np.random.randint(0, w, size=batch_count)
            
            for y, x in zip(ys, xs):
                value = zarr_array[int(y), int(x)]
                sampled_values.append(float(value))
            
            if use_progress:
                pbar.update(batch_count)
                
    elif ndim == 3:
        d, h, w = shape
        for i in range(0, num_samples, batch_size):
            batch_count = min(batch_size, num_samples - i)
            zs = np.random.randint(0, d, size=batch_count)
            ys = np.random.randint(0, h, size=batch_count)
            xs = np.random.randint(0, w, size=batch_count)
            
            for z, y, x in zip(zs, ys, xs):
                try:
                    value = zarr_array[int(z), int(y), int(x)]
                    sampled_values.append(float(value))
                except (ValueError, IndexError) as e:
                    # Skip this sample if chunk is missing or corrupted
                    print(f"Warning: Skipping sample at ({z}, {y}, {x}) due to error: {e}")
                    continue
            
            if use_progress:
                pbar.update(batch_count)
    else:
        raise ValueError(f"Unsupported array dimensionality: {ndim}")
    
    if use_progress:
        pbar.close()
    
    # Check if we collected any valid samples
    if len(sampled_values) == 0:
        raise ValueError(
            f"Failed to collect any valid samples from zarr array with shape {shape}. "
            f"This may indicate corrupted data, missing chunks, or incorrect array boundaries. "
            f"Attempted to sample {num_samples} values from volume {vol_idx}."
        )
    
    return sampled_values


def compute_intensity_properties_parallel(target_volumes, sample_ratio=0.01, max_samples=1000000):
    """
    Compute intensity properties from dataset with multiprocessing.
    
    Parameters
    ----------
    target_volumes : dict
        The target volumes from the dataset
    sample_ratio : float
        Ratio of data to sample (default: 0.01 for 1%)
    max_samples : int
        Maximum number of samples to collect (default: 1,000,000)
        
    Returns
    -------
    dict
        Computed intensity properties
    """
    # Get the first target (all targets share the same image)
    first_target_name = list(target_volumes.keys())[0]
    volumes_list = target_volumes[first_target_name]
    
    total_voxels = 0
    volume_tasks = []
    
    # First pass: calculate total voxels and prepare tasks
    for vol_idx, volume_info in enumerate(volumes_list):
        img_data = volume_info['data']['data']
        shape = img_data.shape
        vol_size = np.prod(shape)
        total_voxels += vol_size
        volume_tasks.append((vol_idx, img_data, shape, vol_size))
    
    # Calculate target sample size with max_samples cap
    target_samples_from_ratio = int(total_voxels * sample_ratio)
    target_samples = min(target_samples_from_ratio, max_samples)
    effective_ratio = target_samples / total_voxels
    
    print(f"Total voxels: {total_voxels:,}")
    print(f"Target samples from ratio ({sample_ratio*100:.1f}%): {target_samples_from_ratio:,}")
    if target_samples_from_ratio > max_samples:
        print(f"Capping samples at maximum: {max_samples:,} ({effective_ratio*100:.2f}% effective ratio)")
    else:
        print(f"Using all target samples: {target_samples:,}")
    
    # Prepare sampling tasks with proportional sample counts
    sampling_tasks = []
    for vol_idx, img_data, shape, vol_size in volume_tasks:
        vol_ratio = vol_size / total_voxels
        vol_samples = int(target_samples * vol_ratio)
        if vol_samples > 0:
            sampling_tasks.append((vol_idx, img_data, shape, vol_samples))
    
    # Use multiprocessing to sample from volumes in parallel
    import os
    num_workers = os.cpu_count() // 2
    print(f"\nSampling from {len(sampling_tasks)} volumes using {num_workers} workers...")
    
    with Pool(num_workers) as pool:
        # Process with progress bar
        results = []
        with tqdm(total=len(sampling_tasks), desc="Sampling volumes") as pbar:
            for result in pool.imap_unordered(sample_volume_task, sampling_tasks):
                results.append(result)
                pbar.update(1)
    
    # Combine all samples
    all_values = []
    for vol_idx, sampled_values in results:
        all_values.extend(sampled_values)
        print(f"Volume {vol_idx}: collected {len(sampled_values):,} samples")
    
    # Convert to numpy array
    all_values = np.array(all_values, dtype=np.float32)
    print(f"\nTotal samples collected: {len(all_values):,}")
    
    # Compute statistics with progress
    print("\nComputing statistics...")
    with tqdm(total=7, desc="Computing properties") as pbar:
        mean_val = float(np.mean(all_values))
        pbar.update(1)
        
        std_val = float(np.std(all_values))
        pbar.update(1)
        
        min_val = float(np.min(all_values))
        pbar.update(1)
        
        max_val = float(np.max(all_values))
        pbar.update(1)
        
        median_val = float(np.median(all_values))
        pbar.update(1)
        
        percentile_00_5 = float(np.percentile(all_values, 0.5))
        pbar.update(1)
        
        percentile_99_5 = float(np.percentile(all_values, 99.5))
        pbar.update(1)
    
    # Store intensity properties
    intensity_properties = {
        'mean': mean_val,
        'std': std_val,
        'percentile_00_5': percentile_00_5,
        'percentile_99_5': percentile_99_5,
        'min': min_val,
        'max': max_val,
        'median': median_val
    }
    
    print(f"\nComputed intensity properties:")
    print(f"  Mean: {mean_val:.4f}")
    print(f"  Std: {std_val:.4f}")
    print(f"  Min: {min_val:.4f}")
    print(f"  Max: {max_val:.4f}")
    print(f"  Median: {median_val:.4f}")
    print(f"  0.5 percentile: {percentile_00_5:.4f}")
    print(f"  99.5 percentile: {percentile_99_5:.4f}")
    print()
    
    return intensity_properties
