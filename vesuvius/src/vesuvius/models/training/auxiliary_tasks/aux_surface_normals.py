"""
Surface normals auxiliary task implementation.

This module provides functionality for creating and computing surface normals
auxiliary tasks from source segmentation targets.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np

import vc_reader.transforms as A


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
                                   return_sdt: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Compute surface normals from a binary mask using signed distance transform.
    
    Parameters
    ----------
    binary_mask : np.ndarray
        Binary mask where 1 indicates foreground
    is_2d : bool
        Whether the data is 2D or 3D
    return_sdt : bool, optional
        Whether to return the signed distance transform (default: False)
        
    Returns
    -------
    normals : np.ndarray
        Surface normals array of shape (2, H, W) for 2D or (3, D, H, W) for 3D
    sdt : np.ndarray or None
        Signed distance transform if return_sdt is True
    """
    # Signed distance (outside - inside) and its Scharr gradient over
    # max(|gradient|, 1e-6), as XYZ components.
    out = A.DistanceTransform(on="label", normals="_normals", components="xyz", implicit_background=True)(
        {"label": np.asarray(binary_mask)[None]}, targets={"label": A.Label()})
    normals, sdt = out["label_normals"], out["label_sdt"][0]

    if return_sdt:
        return normals.astype(np.float32), sdt.astype(np.float32)
    else:
        return normals.astype(np.float32), None
