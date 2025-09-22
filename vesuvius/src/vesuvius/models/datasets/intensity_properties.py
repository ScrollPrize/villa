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


def _sample_foreground_from_numpy(img_data: np.ndarray, lbl_data: np.ndarray, num_samples: int) -> list:
    """
    Sample foreground intensities (label > 0) from numpy arrays.

    If there are fewer than num_samples foreground voxels, samples with replacement.
    """
    if img_data.shape != lbl_data.shape:
        raise ValueError(f"Image and label shape mismatch: {img_data.shape} vs {lbl_data.shape}")

    fg_idx = np.argwhere(lbl_data > 0)
    if fg_idx.size == 0:
        return []
    rs = np.random.RandomState(1234)
    if len(fg_idx) >= num_samples:
        sel = rs.choice(len(fg_idx), size=num_samples, replace=False)
    else:
        sel = rs.choice(len(fg_idx), size=num_samples, replace=True)
    coords = fg_idx[sel]
    # Support 2D and 3D
    if img_data.ndim == 2:
        vals = [float(img_data[y, x]) for (y, x) in coords]
    elif img_data.ndim == 3:
        vals = [float(img_data[z, y, x]) for (z, y, x) in coords]
    else:
        raise ValueError(f"Unsupported image dimensionality: {img_data.ndim}")
    return vals


def _sample_foreground_from_zarr(img_zarr, lbl_zarr, num_samples: int, vol_idx: int) -> list:
    """
    Sample foreground intensities (label > 0) from zarr arrays without loading entire arrays.

    Strategy: random coordinate proposals with acceptance if lbl>0, up to a capped number of trials.
    Falls back to loading the full label if acceptance is too low.
    """
    shape = lbl_zarr.shape
    ndim = len(shape)
    if ndim not in (2, 3):
        raise ValueError(f"Unsupported label dimensionality: {ndim}")

    rs = np.random.RandomState(1234)
    sampled = []
    max_trials = max(num_samples * 20, 20000)
    trials = 0
    use_progress = num_samples > 10000
    pbar = tqdm(total=num_samples, desc=f"Sampling fg zarr vol {vol_idx}", leave=False) if use_progress else None

    while len(sampled) < num_samples and trials < max_trials:
        trials += 1
        if ndim == 2:
            y = int(rs.randint(0, shape[0]))
            x = int(rs.randint(0, shape[1]))
            try:
                if float(lbl_zarr[y, x]) > 0:
                    sampled.append(float(img_zarr[y, x]))
                    if pbar: pbar.update(1)
            except Exception:
                continue
        else:
            z = int(rs.randint(0, shape[0]))
            y = int(rs.randint(0, shape[1]))
            x = int(rs.randint(0, shape[2]))
            try:
                if float(lbl_zarr[z, y, x]) > 0:
                    sampled.append(float(img_zarr[z, y, x]))
                    if pbar: pbar.update(1)
            except Exception:
                continue

    if pbar:
        pbar.close()

    # If acceptance was too low, fall back to loading label fully (best effort)
    if len(sampled) < max(100, int(0.1 * num_samples)):
        try:
            lbl = lbl_zarr[:]
            img = img_zarr[:]
            sampled = _sample_foreground_from_numpy(img, lbl, num_samples)
        except Exception:
            # Return whatever we have
            pass
    return sampled


def _fg_worker(args):
    """Top-level worker for multiprocessing: must be picklable.

    Parameters
    ----------
    args : tuple
        (v_idx, img, lbl, ns)

    Returns
    -------
    tuple
        (v_idx, sampled_values_list)
    """
    v_idx, img, lbl, ns = args
    try:
        if hasattr(img, 'chunks') or hasattr(lbl, 'chunks'):
            return v_idx, _sample_foreground_from_zarr(img, lbl, ns, v_idx)
        else:
            return v_idx, _sample_foreground_from_numpy(np.asarray(img), np.asarray(lbl), ns)
    except Exception as e:
        print(f"Warning: failed sampling foreground from volume {v_idx}: {e}")
        return v_idx, []


def compute_foreground_intensity_properties_parallel(target_volumes: Dict[str, List[Dict[str, Any]]],
                                                     sample_ratio: float = 0.01,
                                                     max_samples: int = 1000000) -> Dict[str, float]:
    """
    Compute intensity properties using only foreground voxels (label > 0), mimicking nnU-Net.

    - Aggregates foreground samples across all volumes
    - Computes mean, std, min, max, median, 0.5th and 99.5th percentiles
    """
    first_target = list(target_volumes.keys())[0]
    volumes_list = target_volumes[first_target]

    # Determine total image voxels for proportional sampling
    total_voxels = 0
    volume_entries = []
    for vol_idx, vinfo in enumerate(volumes_list):
        img = vinfo['data']['data']
        lbl = vinfo['data'].get('label')
        if lbl is None:
            continue  # skip unlabeled volumes
        vol_size = int(np.prod(img.shape))
        total_voxels += vol_size
        volume_entries.append((vol_idx, img, lbl, vol_size))

    if total_voxels == 0 or not volume_entries:
        raise ValueError("No labeled volumes available to compute foreground intensity properties")

    target_samples_from_ratio = int(total_voxels * sample_ratio)
    target_samples = min(target_samples_from_ratio, max_samples)
    if target_samples == 0:
        target_samples = min(10000, max_samples)

    # Proportionally assign samples per volume based on volume size
    per_volume_tasks = []
    for vol_idx, img, lbl, vol_size in volume_entries:
        vol_samples = int(target_samples * (vol_size / total_voxels))
        vol_samples = max(vol_samples, 1)
        per_volume_tasks.append((vol_idx, img, lbl, vol_samples))

    print(f"Sampling foreground intensities from {len(per_volume_tasks)} volumes ...")
    all_values = []
    # Parallel over volumes (use top-level worker to avoid pickling issues)
    with Pool(min(len(per_volume_tasks), os.cpu_count() or 2)) as pool:
        for v_idx, vals in tqdm(
            pool.imap_unordered(_fg_worker, per_volume_tasks),
            total=len(per_volume_tasks),
            desc="Sampling fg"
        ):
            all_values.extend(vals)

    if len(all_values) == 0:
        raise ValueError("Failed to collect any foreground intensity samples. Check your labels (must be >0 for foreground).")

    all_values = np.asarray(all_values, dtype=np.float32)
    percentiles = np.percentile(all_values, [0.5, 99.5])
    intensity_properties = {
        'mean': float(np.mean(all_values)),
        'std': float(np.std(all_values)),
        'percentile_00_5': float(percentiles[0]),
        'percentile_99_5': float(percentiles[1]),
        'min': float(np.min(all_values)),
        'max': float(np.max(all_values)),
        'median': float(np.median(all_values)),
    }
    print("Computed foreground intensity properties (nnU-Net style):")
    for k, v in intensity_properties.items():
        print(f"  {k}: {v:.6f}")
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


def save_intensity_props_formatted(output_path: Path, intensity_properties: Dict[str, float], channel: int = 0) -> bool:
    """
    Save intensity properties in the specific format requested for CT normalization.
    
    Parameters
    ----------
    output_path : Path
        Path to save the intensity_props.json file
    intensity_properties : dict
        Computed intensity properties
    channel : int
        Channel index (default is 0 for single channel data)
        
    Returns
    -------
    bool
        True if save was successful
    """
    # Format the data in the requested structure
    formatted_data = {
        "foreground_intensity_properties_per_channel": {
            str(channel): {
                "max": intensity_properties.get('max', 0.0),
                "mean": intensity_properties.get('mean', 0.0),
                "median": intensity_properties.get('median', 0.0),
                "min": intensity_properties.get('min', 0.0),
                "percentile_00_5": intensity_properties.get('percentile_00_5', 0.0),
                "percentile_99_5": intensity_properties.get('percentile_99_5', 0.0),
                "std": intensity_properties.get('std', 0.0)
            }
        }
    }
    
    try:
        with open(output_path, 'w') as f:
            json.dump(formatted_data, f, indent=4)
        print(f"Saved formatted intensity properties to: {output_path}")
        return True
    except Exception as e:
        print(f"Warning: Failed to save formatted intensity properties: {e}")
        return False


def load_intensity_props_formatted(file_path: Path, channel: int = 0) -> Optional[Dict[str, float]]:
    """
    Load intensity properties from the formatted JSON file for CT normalization.
    Supports both simple format and nnUNet format.
    
    Parameters
    ----------
    file_path : Path
        Path to the intensity_props.json file
    channel : int
        Channel index to load (default is 0)
        
    Returns
    -------
    dict or None
        Intensity properties if successful, None otherwise
    """
    if not file_path.exists():
        print(f"No intensity properties file found at: {file_path}")
        return None
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Check for our cached wrapper format first
        if 'intensity_properties' in data and isinstance(data['intensity_properties'], dict):
            props = data['intensity_properties']
            return {
                'mean': props.get('mean', 0.0),
                'std': props.get('std', 0.0),
                'min': props.get('min', 0.0),
                'max': props.get('max', 0.0),
                'median': props.get('median', 0.0),
                'percentile_00_5': props.get('percentile_00_5', 0.0),
                'percentile_99_5': props.get('percentile_99_5', 0.0)
            }

        # Check for simple format next (direct properties at root level)
        if 'mean' in data and 'std' in data:
            # Simple format - return directly
            return {
                'mean': data.get('mean', 0.0),
                'std': data.get('std', 0.0),
                'min': data.get('min', 0.0),
                'max': data.get('max', 0.0),
                'median': data.get('median', 0.0),
                'percentile_00_5': data.get('percentile_00_5', 0.0),
                'percentile_99_5': data.get('percentile_99_5', 0.0)
            }
        
        # Check for nnUNet format with channels
        channel_key = str(channel)
        if 'foreground_intensity_properties_per_channel' in data:
            if channel_key in data['foreground_intensity_properties_per_channel']:
                props = data['foreground_intensity_properties_per_channel'][channel_key]
                # Return in the format expected by CTNormalization
                return {
                    'mean': props.get('mean', 0.0),
                    'std': props.get('std', 0.0),
                    'min': props.get('min', 0.0),
                    'max': props.get('max', 0.0),
                    'median': props.get('median', 0.0),
                    'percentile_00_5': props.get('percentile_00_5', 0.0),
                    'percentile_99_5': props.get('percentile_99_5', 0.0)
                }
            else:
                print(f"Channel {channel} not found in intensity properties file")
                return None
        else:
            print(f"Unrecognized format in intensity properties file: {file_path}")
            return None
            
    except Exception as e:
        print(f"Warning: Failed to load intensity properties from {file_path}: {e}")
        return None


def load_intensity_properties(cache_dir: Path) -> Optional[Tuple[Dict[str, float], str]]:
    """
    Load intensity properties from JSON file.
    Checks both the standard format and simple intensity_props.json format.
    
    Parameters
    ----------
    cache_dir : Path
        Directory containing cache files
        
    Returns
    -------
    tuple or None
        (intensity_properties, normalization_scheme) if successful, None otherwise
    """
    # First try the standard filename
    filename = get_intensity_properties_filename(cache_dir)
    
    if filename.exists():
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            intensity_properties = data.get('intensity_properties')
            normalization_scheme = data.get('normalization_scheme')
            timestamp = data.get('timestamp', 'unknown')
            
            if intensity_properties and normalization_scheme:
                print(f"Loaded intensity properties from: {filename} (saved at {timestamp})")
                return intensity_properties, normalization_scheme
        except Exception as e:
            print(f"Warning: Failed to load from {filename}: {e}")
    
    # If standard file doesn't exist or failed, try the simple format
    simple_filename = cache_dir / "intensity_props.json"
    if simple_filename.exists():
        print(f"Trying simple format intensity properties file: {simple_filename}")
        props = load_intensity_props_formatted(simple_filename, channel=0)
        if props:
            # For simple format, we assume CT normalization since that's what uses these properties
            print(f"Loaded simple format intensity properties from: {simple_filename}")
            return props, 'ct'
    
    print(f"No valid intensity properties file found in: {cache_dir}")
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
        print(f"\nComputing intensity properties for {normalization_scheme} normalization (foreground only)...")
        try:
            # Match nnU-Net: compute on foreground voxels as defined by labels (>0)
            intensity_properties = compute_foreground_intensity_properties_parallel(
                target_volumes,
                sample_ratio=sample_ratio,
                max_samples=max_samples
            )
        except Exception as e:
            # Safe fallback: for zscore we can skip global props and use per-instance zscore
            if normalization_scheme == 'zscore':
                print(f"Warning: Failed to compute intensity properties ({e}). "
                      f"Falling back to instance z-score (no global props).")
                intensity_properties = {}
            else:
                # For schemes that require global stats (e.g., 'ct'), re-raise
                raise
        
        # Update the config manager if provided
        if mgr is not None and hasattr(mgr, 'intensity_properties'):
            mgr.intensity_properties = intensity_properties
            if hasattr(mgr, 'dataset_config'):
                mgr.dataset_config['intensity_properties'] = intensity_properties
        
        # Save to cache for future use
        if cache_enabled and cache_dir is not None:
            save_intensity_properties(cache_dir, intensity_properties, normalization_scheme)
            # Also save the formatted version for CT normalization
            formatted_path = cache_dir / "intensity_props.json"
            save_intensity_props_formatted(formatted_path, intensity_properties, channel=0)
    
    return intensity_properties
