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
                          valid_patch_find_resolution: int = 1,
                          valid_patch_value: Optional[float] = None,
                          bg_sampling_enabled: bool = False,
                          bg_to_fg_ratio: float = 0.5,
                          unlabeled_fg_enabled: bool = False,
                          unlabeled_fg_threshold: float = 0.05,
                          unlabeled_fg_bbox_threshold: float = 0.15) -> str:
    """
    Generate a unique cache filename based on dataset configuration.

    Args:
        train_data_paths: List of training data paths
        label_paths: List of label paths
        patch_size: Tuple of patch dimensions
        min_labeled_ratio: Minimum labeled ratio threshold
        bbox_threshold: Bounding box threshold
        valid_patch_find_resolution: Downsample level used for patch finding
        valid_patch_value: Optional specific label value for patch validation
        bg_sampling_enabled: Whether BG-only patch sampling is enabled
        bg_to_fg_ratio: Ratio of BG samples to FG samples
        unlabeled_fg_enabled: Whether unlabeled foreground detection is enabled
        unlabeled_fg_threshold: Minimum fraction of non-zero image voxels for unlabeled FG
        unlabeled_fg_bbox_threshold: Minimum bbox coverage for unlabeled FG

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
        f"valid_patch_find_resolution:{valid_patch_find_resolution}"
        f"valid_patch_value:{valid_patch_value}"
        f"bg_sampling_enabled:{bg_sampling_enabled}"
        f"bg_to_fg_ratio:{bg_to_fg_ratio}"
        f"unlabeled_fg_enabled:{unlabeled_fg_enabled}"
        f"unlabeled_fg_threshold:{unlabeled_fg_threshold}"
        f"unlabeled_fg_bbox_threshold:{unlabeled_fg_bbox_threshold}"
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
                      valid_patch_find_resolution: int = 1,
                      cache_path: Optional[str] = None,
                      valid_patch_value: Optional[float] = None,
                      bg_only_patches: Optional[List[Dict]] = None,
                      bg_sampling_enabled: bool = False,
                      bg_to_fg_ratio: float = 0.5,
                      unlabeled_fg_patches: Optional[List[Dict]] = None,
                      unlabeled_fg_enabled: bool = False,
                      unlabeled_fg_threshold: float = 0.05,
                      unlabeled_fg_bbox_threshold: float = 0.15) -> str:
    """
    Save valid patches to a JSON file with metadata.

    Args:
        valid_patches: List of valid patch dictionaries (foreground patches)
        train_data_paths: List of training data paths
        label_paths: List of label paths
        patch_size: Tuple of patch dimensions
        min_labeled_ratio: Minimum labeled ratio threshold
        bbox_threshold: Bounding box threshold
        valid_patch_find_resolution: Downsample level used for patch finding
        cache_path: Optional path to save cache file
        valid_patch_value: Optional specific label value for patch validation
        bg_only_patches: Optional list of background-only patch dictionaries
        bg_sampling_enabled: Whether BG-only patch sampling is enabled
        bg_to_fg_ratio: Ratio of BG samples to FG samples
        unlabeled_fg_patches: Optional list of unlabeled foreground patches (for semi-supervised)
        unlabeled_fg_enabled: Whether unlabeled foreground detection is enabled
        unlabeled_fg_threshold: Minimum fraction of non-zero image voxels for unlabeled FG
        unlabeled_fg_bbox_threshold: Minimum bbox coverage for unlabeled FG

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
        min_labeled_ratio, bbox_threshold, valid_patch_find_resolution, valid_patch_value,
        bg_sampling_enabled, bg_to_fg_ratio,
        unlabeled_fg_enabled, unlabeled_fg_threshold, unlabeled_fg_bbox_threshold
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
            "valid_patch_find_resolution": valid_patch_find_resolution,
            "valid_patch_value": valid_patch_value,
            "num_valid_patches": len(valid_patches),
            "bg_sampling_enabled": bg_sampling_enabled,
            "bg_to_fg_ratio": bg_to_fg_ratio,
            "num_bg_only_patches": len(bg_only_patches) if bg_only_patches else 0,
            "num_unlabeled_fg_patches": len(unlabeled_fg_patches) if unlabeled_fg_patches else 0,
            "unlabeled_fg_enabled": unlabeled_fg_enabled,
            "unlabeled_fg_threshold": unlabeled_fg_threshold,
            "unlabeled_fg_bbox_threshold": unlabeled_fg_bbox_threshold,
        },
        "valid_patches": [],
        "bg_only_patches": [],
        "unlabeled_fg_patches": [],
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

    # Add BG-only patches if present
    if bg_only_patches:
        for patch in bg_only_patches:
            vol_idx = patch["volume_idx"]
            patch_with_path = {
                "volume_path": str(train_data_paths[vol_idx]),
                "volume_index": vol_idx,
                "volume_name": patch["volume_name"],
                "start_position": patch["start_pos"],
                "patch_size": list(patch_size),
                "is_bg_only": True
            }
            cache_data["bg_only_patches"].append(patch_with_path)

    # Add unlabeled foreground patches if present (for semi-supervised learning)
    if unlabeled_fg_patches:
        for patch in unlabeled_fg_patches:
            vol_idx = patch["volume_idx"]
            patch_with_path = {
                "volume_path": str(train_data_paths[vol_idx]),
                "volume_index": vol_idx,
                "volume_name": patch["volume_name"],
                "start_position": patch["start_pos"],
                "patch_size": list(patch_size),
                "is_unlabeled_fg": True
            }
            cache_data["unlabeled_fg_patches"].append(patch_with_path)

    # Save to JSON file
    with open(cache_file_path, 'w') as f:
        json.dump(cache_data, f, indent=2)

    print(f"Valid patches saved to: {cache_file_path}")
    if bg_only_patches:
        print(f"  - {len(valid_patches)} FG patches, {len(bg_only_patches)} BG-only patches")
    if unlabeled_fg_patches:
        print(f"  - {len(unlabeled_fg_patches)} unlabeled foreground patches")
    return str(cache_file_path)

def load_cached_patches(train_data_paths: List,
                       label_paths: List,
                       patch_size: Tuple[int, int, int],
                       min_labeled_ratio: float,
                       bbox_threshold: float = 0.97,
                       valid_patch_find_resolution: int = 1,
                       cache_path: Optional[str] = None,
                       valid_patch_value: Optional[float] = None,
                       bg_sampling_enabled: bool = False,
                       bg_to_fg_ratio: float = 0.5,
                       unlabeled_fg_enabled: bool = False,
                       unlabeled_fg_threshold: float = 0.05,
                       unlabeled_fg_bbox_threshold: float = 0.15) -> Optional[Dict[str, List[Dict]]]:
    """
    Load cached valid patches if they exist and match current configuration.

    Args:
        train_data_paths: List of training data paths
        label_paths: List of label paths
        patch_size: Tuple of patch dimensions
        min_labeled_ratio: Minimum labeled ratio threshold
        bbox_threshold: Bounding box threshold
        valid_patch_find_resolution: Downsample level used for patch finding
        cache_path: Optional path to load cache file from
        valid_patch_value: Optional specific label value for patch validation
        bg_sampling_enabled: Whether BG-only patch sampling is enabled
        bg_to_fg_ratio: Ratio of BG samples to FG samples
        unlabeled_fg_enabled: Whether unlabeled foreground detection is enabled
        unlabeled_fg_threshold: Minimum fraction of non-zero image voxels for unlabeled FG
        unlabeled_fg_bbox_threshold: Minimum bbox coverage for unlabeled FG

    Returns:
        Dictionary with 'fg_patches', 'bg_patches', and 'unlabeled_fg_patches' lists if cache is valid, None otherwise
    """
    # Determine cache directory
    if cache_path is None:
        cache_dir = Path("patch_caches")
    else:
        cache_dir = Path(cache_path)

    # Generate cache filename
    cache_filename = generate_cache_filename(
        train_data_paths, label_paths, patch_size,
        min_labeled_ratio, bbox_threshold, valid_patch_find_resolution, valid_patch_value,
        bg_sampling_enabled, bg_to_fg_ratio,
        unlabeled_fg_enabled, unlabeled_fg_threshold, unlabeled_fg_bbox_threshold
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
            valid_patch_find_resolution != metadata.get("valid_patch_find_resolution", 1) or
            valid_patch_value != metadata.get("valid_patch_value") or
            bg_sampling_enabled != metadata.get("bg_sampling_enabled", False) or
            bg_to_fg_ratio != metadata.get("bg_to_fg_ratio", 0.5) or
            unlabeled_fg_enabled != metadata.get("unlabeled_fg_enabled", False) or
            unlabeled_fg_threshold != metadata.get("unlabeled_fg_threshold", 0.05) or
            unlabeled_fg_bbox_threshold != metadata.get("unlabeled_fg_bbox_threshold", 0.15)
        ):
            print("Cache configuration mismatch - recomputing patches")
            return None

        # Convert cached patches back to expected format for base_dataset
        fg_patches = []
        for patch in cache_data["valid_patches"]:
            fg_patches.append({
                "volume_index": patch["volume_index"],
                "volume_name": patch["volume_name"],
                "position": patch["start_position"],
                "is_bg_only": False
            })

        # Load BG-only patches if present
        bg_patches = []
        for patch in cache_data.get("bg_only_patches", []):
            bg_patches.append({
                "volume_index": patch["volume_index"],
                "volume_name": patch["volume_name"],
                "position": patch["start_position"],
                "is_bg_only": True
            })

        # Load unlabeled foreground patches if present (for semi-supervised)
        unlabeled_fg_patches = []
        for patch in cache_data.get("unlabeled_fg_patches", []):
            unlabeled_fg_patches.append({
                "volume_index": patch["volume_index"],
                "volume_name": patch["volume_name"],
                "position": patch["start_position"],
                "is_unlabeled_fg": True
            })

        print(f"Loaded {len(fg_patches)} FG patches from cache: {cache_file_path}")
        if bg_patches:
            print(f"Loaded {len(bg_patches)} BG-only patches from cache")
        if unlabeled_fg_patches:
            print(f"Loaded {len(unlabeled_fg_patches)} unlabeled foreground patches from cache")

        return {
            "fg_patches": fg_patches,
            "bg_patches": bg_patches,
            "unlabeled_fg_patches": unlabeled_fg_patches,
        }

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
