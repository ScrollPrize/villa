"""
Surface normals auxiliary task implementation.

This module provides functionality for creating and computing surface normals
auxiliary tasks from source segmentation targets.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.filters import scharr


def create_surface_normals_config(aux_task_name: str, aux_config: Dict[str, Any], 
                                source_target_name: str) -> Dict[str, Any]:
    """
    Create surface normals auxiliary task configuration.
    
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
        Target configuration for the surface normals auxiliary task
    """
    target_config = {
        "out_channels": aux_config.get("out_channels", None),  # Will be determined by dimensionality
        "activation": "none", 
        "auxiliary_task": True,
        "task_type": "surface_normals",
        "source_target": source_target_name,
        "weight": aux_config.get("loss_weight", 1.0),
        "use_source_mask": True  # Use source mask for loss computation
    }
    # Prefer linear interpolation when downsampling targets for deep supervision
    target_config["ds_interpolation"] = aux_config.get("ds_interpolation", "linear")
    
    # Add losses configuration if present
    if "losses" in aux_config:
        target_config["losses"] = aux_config["losses"]
    
    return target_config


def compute_surface_normals_from_sdt(binary_mask: np.ndarray, is_2d: bool, 
                                   epsilon: float = 1e-6, 
                                   return_sdt: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Compute surface normals from a binary mask using signed distance transform.
    
    Parameters
    ----------
    binary_mask : np.ndarray
        Binary mask where 1 indicates foreground
    is_2d : bool
        Whether the data is 2D or 3D
    epsilon : float, optional
        Small value to avoid division by zero (default: 1e-6)
    return_sdt : bool, optional
        Whether to return the signed distance transform (default: False)
        
    Returns
    -------
    normals : np.ndarray
        Surface normals array of shape (2, H, W) for 2D or (3, D, H, W) for 3D
    sdt : np.ndarray or None
        Signed distance transform if return_sdt is True
    """
    # 1. Compute signed distance transform
    inside_dist = distance_transform_edt(binary_mask)
    outside_dist = distance_transform_edt(1 - binary_mask)
    sdt = outside_dist - inside_dist
    
    # 2. Compute gradients using Scharr filter
    if is_2d:
        # For 2D: gradients along x and y
        grad_x = scharr(sdt, axis=1)  # Gradient along x (width)
        grad_y = scharr(sdt, axis=0)  # Gradient along y (height)
        gradients = np.stack([grad_x, grad_y], axis=0)
    else:
        # For 3D: gradients along x, y, and z
        grad_x = scharr(sdt, axis=2)  # Gradient along x (width)
        grad_y = scharr(sdt, axis=1)  # Gradient along y (height)
        grad_z = scharr(sdt, axis=0)  # Gradient along z (depth)
        gradients = np.stack([grad_x, grad_y, grad_z], axis=0)
    
    # 3. Normalize gradients to unit vectors
    magnitude = np.sqrt(np.sum(gradients**2, axis=0, keepdims=True))
    magnitude = np.maximum(magnitude, epsilon)
    normals = gradients / magnitude
    
    # 4. Handle undefined regions (optional)
    # Where gradient magnitude is very small, normal direction is undefined
    undefined_mask = magnitude < epsilon
    normals = normals * (1 - undefined_mask)
    
    if return_sdt:
        return normals.astype(np.float32), sdt.astype(np.float32)
    else:
        return normals.astype(np.float32), None
