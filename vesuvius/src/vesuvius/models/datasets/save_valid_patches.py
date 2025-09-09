import json
import os
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

def generate_cache_filename(train_data_paths: List, 
                          label_paths: List,
                          patch_size: Tuple[int, int, int],
                          min_labeled_ratio: float,
                          bbox_threshold: float = 0.97,
                          downsample_level: int = 1) -> str:
    """
    Generate a unique cache filename based on dataset configuration.
    
    Args:
        train_data_paths: List of training data paths
        label_paths: List of label paths
        patch_size: Tuple of patch dimensions
        min_labeled_ratio: Minimum labeled ratio threshold
        bbox_threshold: Bounding box threshold
        downsample_level: Downsample level used for patch finding
        
    Returns:
        Unique filename for the cache
    """
    # Convert paths to strings for hashing
    train_paths_str = [str(path) for path in train_data_paths]
    label_paths_str = [str(path) for path in label_paths]
    
    # Create a string representation of the configuration
    config_str = (
        f"train_paths:{sorted(train_paths_str)}"
        f"label_paths:{sorted(label_paths_str)}"
        f"patch_size:{patch_size}"
        f"min_labeled_ratio:{min_labeled_ratio}"
        f"bbox_threshold:{bbox_threshold}"
        f"downsample_level:{downsample_level}"
    )
    
    # Generate hash
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:12]
    
    return f"valid_patches_{config_hash}.json"

def save_valid_patches(valid_patches: List[Dict],
                      train_data_paths: List,
                      label_paths: List,
                      patch_size: Tuple[int, int, int],
                      min_labeled_ratio: float,
                      bbox_threshold: float = 0.97,
                      downsample_level: int = 1,
                      cache_path: Optional[str] = None) -> str:
    """
    Save valid patches to a JSON file with metadata.
    
    Args:
        valid_patches: List of valid patch dictionaries
        train_data_paths: List of training data paths
        label_paths: List of label paths
        patch_size: Tuple of patch dimensions
        min_labeled_ratio: Minimum labeled ratio threshold
        bbox_threshold: Bounding box threshold
        downsample_level: Downsample level used for patch finding
        cache_path: Optional path to save cache file
        
    Returns:
        Path to the saved JSON file
    """
    # Determine cache directory
    if cache_path is None:
        cache_dir = Path("patch_caches")
    else:
        cache_dir = Path(cache_path)
    
    # Create cache directory if it doesn't exist
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate cache filename
    cache_filename = generate_cache_filename(
        train_data_paths, label_paths, patch_size, 
        min_labeled_ratio, bbox_threshold, downsample_level
    )
    
    cache_file_path = cache_dir / cache_filename
    
    # Prepare data to save
    cache_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "train_data_paths": [str(path) for path in train_data_paths],
            "label_paths": [str(path) for path in label_paths],
            "patch_size": list(patch_size),
            "min_labeled_ratio": min_labeled_ratio,
            "bbox_threshold": bbox_threshold,
            "downsample_level": downsample_level,
            "num_valid_patches": len(valid_patches)
        },
        "valid_patches": []
    }
    
    # Add volume paths to each patch entry
    for patch in valid_patches:
        vol_idx = patch["volume_idx"]
        patch_with_path = {
            "volume_path": str(train_data_paths[vol_idx]),
            "volume_index": vol_idx,
            "volume_name": patch["volume_name"],
            "start_position": patch["start_pos"],
            "patch_size": list(patch_size)
        }
        cache_data["valid_patches"].append(patch_with_path)
    
    # Save to JSON file
    with open(cache_file_path, 'w') as f:
        json.dump(cache_data, f, indent=2)
    
    print(f"Valid patches saved to: {cache_file_path}")
    return str(cache_file_path)

def load_cached_patches(train_data_paths: List,
                       label_paths: List,
                       patch_size: Tuple[int, int, int],
                       min_labeled_ratio: float,
                       bbox_threshold: float = 0.97,
                       downsample_level: int = 1,
                       cache_path: Optional[str] = None) -> Optional[List[Dict]]:
    """
    Load cached valid patches if they exist and match current configuration.
    
    Args:
        train_data_paths: List of training data paths
        label_paths: List of label paths
        patch_size: Tuple of patch dimensions
        min_labeled_ratio: Minimum labeled ratio threshold
        bbox_threshold: Bounding box threshold
        downsample_level: Downsample level used for patch finding
        cache_path: Optional path to load cache file from
        
    Returns:
        List of valid patch dictionaries if cache is valid, None otherwise
    """
    # Determine cache directory
    if cache_path is None:
        cache_dir = Path("patch_caches")
    else:
        cache_dir = Path(cache_path)
    
    # Generate cache filename
    cache_filename = generate_cache_filename(
        train_data_paths, label_paths, patch_size, 
        min_labeled_ratio, bbox_threshold, downsample_level
    )
    
    cache_file_path = cache_dir / cache_filename
    
    # Check if cache file exists
    if not cache_file_path.exists():
        return None
    
    try:
        # Load cache file
        with open(cache_file_path, 'r') as f:
            cache_data = json.load(f)
        
        # Validate cache metadata
        metadata = cache_data["metadata"]
        
        # Check if configuration matches
        if (
            [str(path) for path in train_data_paths] != metadata["train_data_paths"] or
            [str(path) for path in label_paths] != metadata["label_paths"] or
            list(patch_size) != metadata["patch_size"] or
            min_labeled_ratio != metadata["min_labeled_ratio"] or
            bbox_threshold != metadata["bbox_threshold"] or
            downsample_level != metadata.get("downsample_level", 1)
        ):
            print("Cache configuration mismatch - recomputing patches")
            return None
        
        # Convert cached patches back to expected format for base_dataset
        valid_patches = []
        for patch in cache_data["valid_patches"]:
            valid_patches.append({
                "volume_index": patch["volume_index"],
                "volume_name": patch["volume_name"],
                "position": patch["start_position"]
            })
        
        print(f"Loaded {len(valid_patches)} valid patches from cache: {cache_file_path}")
        return valid_patches
        
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error loading cache file {cache_file_path}: {e}")
        return None

def get_cache_info(train_data_paths: List,
                  label_paths: List,
                  patch_size: Tuple[int, int, int],
                  min_labeled_ratio: float,
                  bbox_threshold: float = 0.97,
                  cache_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Get information about the cache file if it exists.
    
    Args:
        train_data_paths: List of training data paths
        label_paths: List of label paths
        patch_size: Tuple of patch dimensions
        min_labeled_ratio: Minimum labeled ratio threshold
        bbox_threshold: Bounding box threshold
        cache_path: Optional path to cache file
        
    Returns:
        Dictionary with cache information or None if no cache exists
    """
    # Determine cache directory
    if cache_path is None:
        cache_dir = Path("patch_caches")
    else:
        cache_dir = Path(cache_path)
    
    # Generate cache filename
    cache_filename = generate_cache_filename(
        train_data_paths, label_paths, patch_size, 
        min_labeled_ratio, bbox_threshold
    )
    
    cache_file_path = cache_dir / cache_filename
    
    # Check if cache file exists
    if not cache_file_path.exists():
        return None
    
    try:
        # Load cache file
        with open(cache_file_path, 'r') as f:
            cache_data = json.load(f)
        
        return cache_data["metadata"]
        
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error reading cache file {cache_file_path}: {e}")
        return None
