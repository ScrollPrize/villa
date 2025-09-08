"""
Distance transform auxiliary task implementation.

This module provides functionality for creating and computing distance transform
auxiliary tasks from source segmentation targets.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from scipy.ndimage import distance_transform_edt


def create_distance_transform_config(aux_task_name: str, aux_config: Dict[str, Any], 
                                   source_target_name: str) -> Dict[str, Any]:
    """
    Create distance transform auxiliary task configuration.
    
    Parameters
    ----------
    aux_task_name : str
        Name for the auxiliary task
    aux_config : dict
        Configuration for the auxiliary task from YAML
    source_target_name : str
        Name of the source target this auxiliary task depends on
        
    Returns
    -------
    dict
        Target configuration for the distance transform auxiliary task
    """
    target_config = {
        "out_channels": aux_config.get("out_channels", 1),
        "activation": "none", 
        "auxiliary_task": True,
        "task_type": "distance_transform",
        "source_target": source_target_name,
        "weight": aux_config.get("loss_weight", 1.0)
    }
    # Allow user to choose which distance to predict: 'signed' | 'inside' | 'outside'
    target_config["distance_type"] = str(aux_config.get("distance_type", "signed")).lower()
    # Prefer linear interpolation when downsampling targets for deep supervision
    target_config["ds_interpolation"] = aux_config.get("ds_interpolation", "linear")
    
    # Add losses configuration if present
    if "losses" in aux_config:
        target_config["losses"] = aux_config["losses"]
    
    return target_config


def compute_distance_transform(source_patch: np.ndarray, needs_sdt: bool = False, 
                             sdt_cache: Optional[Dict[str, np.ndarray]] = None,
                             cache_key_prefix: str = "") -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]:
    """
    Compute distance transform from source patch.
    
    Parameters
    ----------
    source_patch : np.ndarray
        Source segmentation patch of shape (C, ...) where C is channels
    needs_sdt : bool, optional
        Whether to compute signed distance transform (default: False)
    sdt_cache : dict, optional
        Cache for signed distance transforms to avoid recomputation
    cache_key_prefix : str, optional
        Prefix for cache keys
        
    Returns
    -------
    distance_patch : np.ndarray
        Distance transform patch of shape (C, ...) or (1, ...) if single channel output
    sdt_cache : dict or None
        Updated SDT cache if caching was used
    """
    distance_transforms = []
    if sdt_cache is None:
        sdt_cache = {}
    
    for c in range(source_patch.shape[0]):  # Iterate over channels
        channel_data = source_patch[c]
        cache_key = f"{cache_key_prefix}_sdt_{c}" if cache_key_prefix else f"sdt_{c}"
        
        # Create binary mask (foreground = 1)
        binary_mask = (channel_data > 0).astype(np.uint8)
        
        if cache_key in sdt_cache:
            # Reuse cached signed distance transform
            sdt = sdt_cache[cache_key]
            # For distance transform, we want the absolute distance inside the object
            # SDT is negative inside, positive outside
            distance_map = np.where(sdt < 0, -sdt, 0)
        else:
            if needs_sdt and np.any(binary_mask):
                # Compute signed distance transform
                inside_dist = distance_transform_edt(binary_mask)
                outside_dist = distance_transform_edt(1 - binary_mask)
                sdt = outside_dist - inside_dist
                sdt_cache[cache_key] = sdt
                # For distance transform, use the inside distance
                distance_map = inside_dist
            else:
                # Regular distance transform
                if np.any(binary_mask):
                    # Compute distance transform from foreground pixels
                    distance_map = distance_transform_edt(binary_mask)
                else:
                    # No foreground pixels, return zeros
                    distance_map = np.zeros_like(channel_data)
        
        distance_transforms.append(distance_map)
    
    # Stack distance transforms and ensure proper format
    if len(distance_transforms) == 1:
        # Single channel output
        distance_patch = distance_transforms[0][np.newaxis, ...]
    else:
        # Multi-channel output
        distance_patch = np.stack(distance_transforms, axis=0)
    
    # Ensure contiguous and proper dtype
    distance_patch = np.ascontiguousarray(distance_patch, dtype=np.float32)
    
    return distance_patch, sdt_cache
