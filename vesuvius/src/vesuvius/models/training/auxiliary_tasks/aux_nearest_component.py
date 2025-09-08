"""
Nearest-component vector+distance target generation.

For a binary source label, compute per-voxel:
- direction: unit vector pointing inside the nearest (correct) component,
             i.e., along the negative gradient of a signed distance transform
- distance:  absolute distance to the closest surface (same SDF magnitude)

Outputs shape: (C_dir + 1, H, W) or (C_dir + 1, D, H, W), where C_dir is 2 (2D) or 3 (3D).
"""

from typing import Dict, Any, Tuple
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter


def create_nearest_component_config(name: str, cfg: Dict[str, Any], source_target_name: str, is_2d: bool) -> Dict[str, Any]:
    """
    Create target config for nearest_component mode.
    """
    # out_channels = vector components + 1 distance channel
    out_ch = (2 if is_2d else 3) + 1
    target_config = {
        "out_channels": cfg.get("out_channels", out_ch),
        "activation": "none",
        "task_type": "nearest_component",
        "auxiliary_task": True,
        "source_target": source_target_name,
        "weight": cfg.get("loss_weight", 1.0),
        # Optional SDF smoothing for stability
        "sdf_sigma": float(cfg.get("sdf_sigma", 0.0)),
        # Prefer linear interpolation during DS downsampling
        "ds_interpolation": cfg.get("ds_interpolation", "linear"),
    }
    # Pass through any explicitly configured losses
    if "losses" in cfg:
        target_config["losses"] = cfg["losses"]
    return target_config


def compute_nearest_component(binary_mask: np.ndarray, is_2d: bool, *, sdf_sigma: float = 0.0) -> np.ndarray:
    """
    Compute per-voxel direction (unit vector pointing inside component) and distance to nearest surface.

    Returns array with shape (C_dir+1, H, W) for 2D or (C_dir+1, D, H, W) for 3D.
    """
    # Signed distance: positive outside, negative inside
    inside = distance_transform_edt(binary_mask)
    outside = distance_transform_edt(1 - binary_mask)
    sdf = (outside - inside).astype(np.float32)

    if sdf_sigma and sdf_sigma > 0:
        sdf = gaussian_filter(sdf, sigma=sdf_sigma)

    # Gradients
    if is_2d:
        gy, gx = np.gradient(sdf)
        # direction: -grad(sdf) normalized
        vx, vy = -gx, -gy
        # normalize
        eps = 1e-6
        nrm = np.sqrt(vx * vx + vy * vy) + eps
        vx /= nrm
        vy /= nrm
        dist = np.abs(sdf).astype(np.float32)
        out = np.stack([vx.astype(np.float32), vy.astype(np.float32), dist], axis=0)
    else:
        gz, gy, gx = np.gradient(sdf)
        vx, vy, vz = -gx, -gy, -gz
        eps = 1e-6
        nrm = np.sqrt(vx * vx + vy * vy + vz * vz) + eps
        vx /= nrm
        vy /= nrm
        vz /= nrm
        dist = np.abs(sdf).astype(np.float32)
        out = np.stack([vx.astype(np.float32), vy.astype(np.float32), vz.astype(np.float32), dist], axis=0)

    return np.ascontiguousarray(out)
