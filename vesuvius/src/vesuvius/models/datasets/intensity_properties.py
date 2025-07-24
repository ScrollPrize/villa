"""
Intensity property computation and management for dataset normalization.

This module handles:
- Computing intensity statistics from dataset volumes
- Saving/loading intensity properties to/from cache
- Multiprocessing-enabled sampling for large datasets
"""
import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
from typing import Dict, List, Any, Optional, Tuple


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

    if hasattr(img_data, 'chunks'):  # Zarr array
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

    use_progress = num_samples > 10000
    if use_progress:
        pbar = tqdm(total=num_samples, desc=f"Sampling zarr volume {vol_idx}", leave=False)

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
                    print(f"Warning: Skipping sample at ({z}, {y}, {x}) due to error: {e}")
                    continue
            
            if use_progress:
                pbar.update(batch_count)
    else:
        raise ValueError(f"Unsupported array dimensionality: {ndim}")
    
    if use_progress:
        pbar.close()

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


def get_intensity_properties_filename(cache_dir: Path) -> Path:
    """
    Get filename for intensity properties JSON file.
    
    Parameters
    ----------
    cache_dir : Path
        Directory to store cache files
        
    Returns
    -------
    Path
        Full path to the intensity properties JSON file
    """
    # Create cache directory if it doesn't exist
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "intensity_properties.json"


def save_intensity_properties(cache_dir: Path, intensity_properties: Dict[str, float], normalization_scheme: str) -> bool:
    """
    Save intensity properties to a separate JSON file.
    
    Parameters
    ----------
    cache_dir : Path
        Directory to store cache files
    intensity_properties : dict
        Computed intensity properties
    normalization_scheme : str
        Normalization scheme used
        
    Returns
    -------
    bool
        True if save was successful
    """
    filename = get_intensity_properties_filename(cache_dir)
    
    data = {
        'intensity_properties': intensity_properties,
        'normalization_scheme': normalization_scheme,
        'timestamp': datetime.now().isoformat()
    }
    
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved intensity properties to: {filename}")
        return True
    except Exception as e:
        print(f"Warning: Failed to save intensity properties: {e}")
        return False


def load_intensity_properties(cache_dir: Path) -> Optional[Tuple[Dict[str, float], str]]:
    """
    Load intensity properties from JSON file.
    
    Parameters
    ----------
    cache_dir : Path
        Directory containing cache files
        
    Returns
    -------
    tuple or None
        (intensity_properties, normalization_scheme) if successful, None otherwise
    """
    filename = get_intensity_properties_filename(cache_dir)
    
    if not filename.exists():
        print(f"No intensity properties file found at: {filename}")
        return None
    
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        intensity_properties = data.get('intensity_properties')
        normalization_scheme = data.get('normalization_scheme')
        timestamp = data.get('timestamp', 'unknown')
        
        if intensity_properties and normalization_scheme:
            print(f"Loaded intensity properties from: {filename} (saved at {timestamp})")
            return intensity_properties, normalization_scheme
        else:
            print(f"Invalid intensity properties file: {filename}")
            return None
            
    except Exception as e:
        print(f"Warning: Failed to load intensity properties from {filename}: {e}")
        return None


def initialize_intensity_properties(target_volumes, 
                                  normalization_scheme,
                                  existing_properties=None,
                                  cache_enabled=True,
                                  cache_dir=None,
                                  mgr=None,
                                  sample_ratio=0.001,
                                  max_samples=1000000):
    """
    Initialize intensity properties for dataset normalization.
    
    This function handles the complete workflow of:
    1. Using existing properties if provided
    2. Loading from cache if available
    3. Computing if necessary
    4. Saving to cache
    5. Updating the config manager
    
    Parameters
    ----------
    target_volumes : dict
        The target volumes from the dataset
    normalization_scheme : str
        Normalization scheme ('zscore', 'ct', etc.)
    existing_properties : dict, optional
        Pre-computed intensity properties
    cache_enabled : bool
        Whether to use caching
    cache_dir : Path, optional
        Directory for cache files
    mgr : object, optional
        Config manager to update with properties
    sample_ratio : float
        Ratio of data to sample (default: 0.01 for 1%)
    max_samples : int
        Maximum number of samples to collect (default: 1,000,000)
        
    Returns
    -------
    dict
        Intensity properties for normalization
    """
    # If properties already exist, use them
    if existing_properties:
        return existing_properties
    
    # Only compute/load for schemes that need it
    if normalization_scheme not in ['zscore', 'ct']:
        return {}
    
    loaded_from_cache = False
    intensity_properties = {}
    
    # Try to load from cache first
    if cache_enabled and cache_dir is not None:
        print("\nChecking for cached intensity properties...")
        intensity_result = load_intensity_properties(cache_dir)
        
        if intensity_result is not None:
            cached_properties, cached_scheme = intensity_result
            if cached_scheme == normalization_scheme:
                intensity_properties = cached_properties
                loaded_from_cache = True
                
                # Update the config manager if provided
                if mgr is not None:
                    mgr.intensity_properties = cached_properties
                    if hasattr(mgr, 'dataset_config'):
                        mgr.dataset_config['intensity_properties'] = cached_properties
                
                print("\nLoaded intensity properties from JSON cache - skipping computation")
                print("Cached intensity properties:")
                for key, value in cached_properties.items():
                    print(f"  {key}: {value:.4f}")
            else:
                print(f"Cached normalization scheme '{cached_scheme}' doesn't match current '{normalization_scheme}'")
    
    # Compute if not loaded from cache
    if not loaded_from_cache:
        print(f"\nComputing intensity properties for {normalization_scheme} normalization...")
        intensity_properties = compute_intensity_properties_parallel(
            target_volumes, 
            sample_ratio=sample_ratio, 
            max_samples=max_samples
        )
        
        # Update the config manager if provided
        if mgr is not None and hasattr(mgr, 'intensity_properties'):
            mgr.intensity_properties = intensity_properties
            if hasattr(mgr, 'dataset_config'):
                mgr.dataset_config['intensity_properties'] = intensity_properties
        
        # Save to cache for future use
        if cache_enabled and cache_dir is not None:
            save_intensity_properties(cache_dir, intensity_properties, normalization_scheme)
    
    return intensity_properties