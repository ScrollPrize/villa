"""
Utilities for patch caching functionality.
"""
import os
from pathlib import Path
import hashlib
import json
import pickle
from datetime import datetime
from typing import Dict, List, Any, Optional


def get_data_checksums(data_path: Path) -> Dict[str, float]:
    """
    Get modification times for all data files in a dataset directory.
    
    This function scans the standard dataset directory structure:
    - images/ (*.tif, *.tiff, *.zarr)
    - labels/ (*.tif, *.tiff, *.zarr)
    - masks/ (*.tif, *.tiff, *.zarr)
    - *.zarr directories (in root)
    
    Parameters
    ----------
    data_path : Path
        Root path of the dataset
        
    Returns
    -------
    dict
        Dictionary mapping file paths to modification times
    """
    checksums = {}
    
    # Check for TIF/TIFF files and Zarr directories in standard subdirectories
    for subdir in ["images", "labels", "masks"]:
        dir_path = data_path / subdir
        if dir_path.exists():
            # Check for TIF/TIFF files
            for pattern in ["*.tif", "*.tiff"]:
                for file_path in dir_path.glob(pattern):
                    checksums[str(file_path)] = os.path.getmtime(file_path)
            
            # Check for Zarr directories within subdirectories
            for zarr_path in dir_path.glob("*.zarr"):
                if zarr_path.is_dir():
                    # For Zarr, check the .zarray files which indicate structure
                    for zarray_file in zarr_path.rglob(".zarray"):
                        checksums[str(zarray_file)] = os.path.getmtime(zarray_file)
                    # Also check the root directory modification time
                    checksums[str(zarr_path)] = os.path.getmtime(zarr_path)
    
    # Also check for Zarr directories in the root (backward compatibility)
    for zarr_path in data_path.glob("*.zarr"):
        if zarr_path.is_dir():
            # For Zarr, check the .zarray files which indicate structure
            for zarray_file in zarr_path.rglob(".zarray"):
                checksums[str(zarray_file)] = os.path.getmtime(zarray_file)
            # Also check the root directory modification time
            checksums[str(zarr_path)] = os.path.getmtime(zarr_path)
    
    # Check for napari project files
    for napari_file in data_path.glob("*.napari"):
        checksums[str(napari_file)] = os.path.getmtime(napari_file)
    
    return checksums


def compute_cache_key(config_params: Dict[str, Any]) -> str:
    """
    Compute a cache key based on configuration parameters.
    
    Parameters
    ----------
    config_params : dict
        Dictionary of configuration parameters
        
    Returns
    -------
    str
        MD5 hash of the configuration
    """
    # Ensure stable ordering for consistent hashing
    config_str = json.dumps(config_params, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()


def get_cache_filename(cache_dir: Path, config_params: Dict[str, Any]) -> Path:
    """
    Generate cache filename based on configuration.
    
    Parameters
    ----------
    cache_dir : Path
        Directory to store cache files
    config_params : dict
        Configuration parameters for generating unique filename
        
    Returns
    -------
    Path
        Full path to the cache file
    """
    # Create cache directory if it doesn't exist
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename based on config hash
    cache_key = compute_cache_key(config_params)
    return cache_dir / f"patches_cache_{cache_key}.pkl"



def save_patches_cache(
    cache_file: Path,
    valid_patches: List[Dict],
    config_params: Dict[str, Any],
    data_checksums: Dict[str, float]
) -> bool:
    """
    Save valid patches to cache file.
    
    Parameters
    ----------
    cache_file : Path
        Path to save the cache file
    valid_patches : list
        List of valid patch positions
    config_params : dict
        Configuration parameters used for patch computation
    data_checksums : dict
        File modification times for data freshness checking
        
    Returns
    -------
    bool
        True if save was successful, False otherwise
    """
    cache_data = {
        **config_params,
        'creation_time': datetime.now().isoformat(),
        'data_checksums': data_checksums,
        'valid_patches': valid_patches,
        'num_patches': len(valid_patches)
    }
    
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"Saved {len(valid_patches)} patches to cache: {cache_file.name}")
        return True
    except Exception as e:
        print(f"Warning: Failed to save cache: {e}")
        return False


def load_patches_cache(cache_file: Path) -> Optional[Dict[str, Any]]:
    """
    Load cache data from file.
    
    Parameters
    ----------
    cache_file : Path
        Path to the cache file
        
    Returns
    -------
    dict or None
        Cache data if successful, None otherwise
    """
    if not cache_file.exists():
        return None
        
    try:
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        return cache_data
    except Exception as e:
        print(f"Warning: Failed to load cache from {cache_file}: {e}")
        return None


def is_cache_valid(
    cache_data: Dict[str, Any],
    current_config: Dict[str, Any],
    current_checksums: Dict[str, float]
) -> bool:
    """
    Validate if cached data is still valid.
    
    Parameters
    ----------
    cache_data : dict
        Loaded cache data
    current_config : dict
        Current configuration parameters
    current_checksums : dict
        Current file modification times
        
    Returns
    -------
    bool
        True if cache is valid, False otherwise
    """
    # Define core configuration keys that must always be checked
    core_config_keys = [
        'patch_size', 'min_labeled_ratio', 'min_bbox_percent',
        'skip_patch_validation', 'targets', 'is_2d_dataset'
    ]
    
    # Check core configuration parameters
    for key in core_config_keys:
        if key not in cache_data or cache_data.get(key) != current_config.get(key):
            print(f"Cache invalid: {key} mismatch")
            return False
    
    # Check additional configuration parameters that might be present
    # This handles dataset-specific parameters
    additional_keys = [
        'dataset_type', 'normalization_scheme', 'use_bounding_box', 'nonzero_validated'
    ]
    
    for key in additional_keys:
        # Only check if the key exists in current config
        if key in current_config:
            if key not in cache_data or cache_data.get(key) != current_config.get(key):
                print(f"Cache invalid: {key} mismatch (expected {current_config.get(key)}, got {cache_data.get(key)})")
                return False
    
    # Check data freshness
    cached_checksums = cache_data.get('data_checksums', {})
    
    # If we have current checksums, validate them
    if current_checksums:
        for file_path, current_mtime in current_checksums.items():
            cached_mtime = cached_checksums.get(file_path)
            if cached_mtime is None:
                print(f"Cache invalid: new file {file_path}")
                return False
            if cached_mtime != current_mtime:
                print(f"Cache invalid: {file_path} was modified")
                return False
    
    # Final check: ensure all keys in current_config match those in cache_data
    # This catches any dataset-specific parameters we might have missed
    for key, value in current_config.items():
        if key in cache_data and cache_data[key] != value:
            print(f"Cache invalid: config parameter '{key}' mismatch (expected {value}, got {cache_data[key]})")
            return False
    
    return True


def load_cached_patches(
    cache_dir: Path,
    config_params: Dict[str, Any],
    data_path: Path
) -> Optional[List[Dict]]:
    """
    High-level function to load cached patches if available and valid.
    
    Parameters
    ----------
    cache_dir : Path
        Directory containing cache files
    config_params : dict
        Current configuration parameters
    data_path : Path
        Path to the dataset for checksum validation
        
    Returns
    -------
    list or None
        valid_patches if cache is valid, None otherwise
    """
    # Get cache filename
    cache_file = get_cache_filename(cache_dir, config_params)
    
    # Try to load cache
    cache_data = load_patches_cache(cache_file)
    if cache_data is None:
        return None
    
    # Get current checksums
    current_checksums = get_data_checksums(data_path)
    
    # Validate cache
    if is_cache_valid(cache_data, config_params, current_checksums):
        valid_patches = cache_data.get('valid_patches', [])
        print(f"Loaded {len(valid_patches)} patches from cache (created {cache_data.get('creation_time', 'unknown')})")
        return valid_patches
    else:
        print("Cache is invalid or outdated, will recompute patches")
        return None


def save_computed_patches(
    valid_patches: List[Dict],
    cache_dir: Path,
    config_params: Dict[str, Any],
    data_path: Path
) -> bool:
    """
    High-level function to save computed patches to cache.
    
    Parameters
    ----------
    valid_patches : list
        List of valid patch positions
    cache_dir : Path
        Directory to save cache files
    config_params : dict
        Configuration parameters used for patch computation
    data_path : Path
        Path to the dataset for checksum generation
        
    Returns
    -------
    bool
        True if save was successful
    """
    # Get checksums
    data_checksums = get_data_checksums(data_path)
    
    # Get cache filename
    cache_file = get_cache_filename(cache_dir, config_params)
    
    # Save cache
    return save_patches_cache(cache_file, valid_patches, config_params, data_checksums)
