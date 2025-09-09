"""
Structure tensor auxiliary task implementation.

Provides configuration and helpers for supervising structure tensor components
derived from a source segmentation target. Intended for regression losses with
masking over the foreground label region.
"""

from typing import Dict, Any, Tuple
import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt


def create_structure_tensor_config(aux_task_name: str, aux_config: Dict[str, Any],
                                   source_target_name: str) -> Dict[str, Any]:
    """
    Create structure tensor auxiliary task configuration.

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
        Target configuration for the structure tensor auxiliary task
    """
    target_config = {
        # Out channels depend on dimensionality: 3 (2D) or 6 (3D). Let auto-detect fill this.
        "out_channels": aux_config.get("out_channels", None),
        "activation": "none",
        "auxiliary_task": True,
        "task_type": "structure_tensor",
        "source_target": source_target_name,
        "weight": aux_config.get("loss_weight", 1.0),
        # Compute from: 'sdt' (signed distance transform) or 'binary' mask
        "compute_from": str(aux_config.get("compute_from", "sdt")).lower(),
        # Smoothing parameters for gradients and tensor averaging
        "grad_sigma": float(aux_config.get("grad_sigma", 1.0)),
        "tensor_sigma": float(aux_config.get("tensor_sigma", 1.5)),
        # For masked regression losses (e.g., MaskedMSELoss), we can encode background as ignore_index
        "ignore_index": aux_config.get("ignore_index", -100),
        # Prefer linear interpolation when downsampling for deep supervision
        "ds_interpolation": aux_config.get("ds_interpolation", "linear"),
    }

    # Add losses configuration if present
    if "losses" in aux_config:
        target_config["losses"] = aux_config["losses"]

    return target_config

def compute_structure_tensor(
    binary_mask: np.ndarray,
    is_2d: bool,
    *,
    compute_from: str = "sdt",
    grad_sigma: float = 1.0,
    tensor_sigma: float = 1.5,
    ignore_index: float | int | None = -100,
) -> np.ndarray:
    """
    Compute structure tensor components from a binary mask.

    Parameters
    ----------
    binary_mask : np.ndarray
        Binary mask where 1 indicates foreground (2D: HxW, 3D: DxHxW)
    is_2d : bool
        Whether data is 2D (True) or 3D (False)
    compute_from : str
        'sdt' (signed distance transform) or 'binary'
    grad_sigma : float
        Gaussian sigma for smoothing prior to gradient computation
    tensor_sigma : float
        Gaussian sigma for smoothing structure tensor components
    ignore_index : float | int | None
        Value assigned to background voxels for masked regression losses

    Returns
    -------
    np.ndarray
        Stacked structure tensor components as float32 array with shape
        (3, H, W) for 2D or (6, D, H, W) for 3D, masked with ignore_index on background when provided.
    """
    # Choose base scalar field to differentiate
    if compute_from.lower() == 'binary':
        base = binary_mask.astype(np.float32)
    else:
        # Signed distance: positive outside, negative inside
        inside = distance_transform_edt(binary_mask)
        outside = distance_transform_edt(1 - binary_mask)
        base = (outside - inside).astype(np.float32)

    # Smooth base before gradients
    base_s = gaussian_filter(base, sigma=grad_sigma) if grad_sigma and grad_sigma > 0 else base

    # Gradients
    if is_2d:
        gy, gx = np.gradient(base_s)
        Jxx = gx * gx
        Jxy = gx * gy
        Jyy = gy * gy
        comps = [Jxx, Jxy, Jyy]
    else:
        gz, gy, gx = np.gradient(base_s)
        Jxx = gx * gx
        Jxy = gx * gy
        Jxz = gx * gz
        Jyy = gy * gy
        Jyz = gy * gz
        Jzz = gz * gz
        comps = [Jxx, Jxy, Jxz, Jyy, Jyz, Jzz]

    # Smooth tensor components
    if tensor_sigma and tensor_sigma > 0:
        comps = [gaussian_filter(c, sigma=tensor_sigma) for c in comps]

    st = np.stack(comps, axis=0).astype(np.float32)

    # Mask background for masked regression
    if ignore_index is not None:
        bg = (binary_mask == 0)
        st[:, bg] = float(ignore_index)

    return np.ascontiguousarray(st, dtype=np.float32)
