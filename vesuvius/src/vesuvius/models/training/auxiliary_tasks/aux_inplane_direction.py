"""
In-plane direction auxiliary task implementation.

Computes a tangent (in-plane) direction field from the structure tensor
derived from a source segmentation target. The output is a unit vector field
with 2 channels (2D) or 3 channels (3D), masked on the background.
"""

from typing import Dict, Any
import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt


def create_inplane_direction_config(aux_task_name: str, aux_config: Dict[str, Any],
                                    source_target_name: str) -> Dict[str, Any]:
    """
    Create in-plane direction auxiliary task configuration.

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
        Target configuration for the in-plane direction auxiliary task
    """
    target_config = {
        # Out channels depend on dimensionality: 2 (2D) or 3 (3D). Let auto-detect fill this.
        "out_channels": aux_config.get("out_channels", None),
        "activation": "none",
        "auxiliary_task": True,
        "task_type": "inplane_direction",
        "source_target": source_target_name,
        "weight": aux_config.get("loss_weight", 1.0),
        # Compute from: 'sdt' (signed distance transform) or 'binary' mask
        "compute_from": str(aux_config.get("compute_from", "sdt")).lower(),
        # Smoothing parameters for gradients and tensor averaging
        "grad_sigma": float(aux_config.get("grad_sigma", 1.0)),
        "tensor_sigma": float(aux_config.get("tensor_sigma", 1.5)),
        # Masking value for background
        "ignore_index": aux_config.get("ignore_index", -100),
        # Prefer linear interpolation during deep supervision downsampling
        "ds_interpolation": aux_config.get("ds_interpolation", "linear"),
    }

    if "losses" in aux_config:
        target_config["losses"] = aux_config["losses"]

    return target_config


def _eigh2x2(a11, a12, a22):
    """Eigen decomposition for symmetric 2x2 matrices.
    Returns eigenvalues (asc) and normalized eigenvectors as (2,2,...).
    """
    # trace and determinant
    tr = a11 + a22
    det = a11 * a22 - a12 * a12
    # eigenvalues
    tmp = np.sqrt(np.maximum(tr * tr / 4 - det, 0.0))
    l1 = tr / 2 - tmp
    l2 = tr / 2 + tmp
    # eigenvectors for l1
    # Solve (A - lI)v = 0; pick vector orthogonal to [a12, l1 - a11]
    v1x = np.where(np.abs(a12) > np.abs(l1 - a11), - (l1 - a11), -a12)
    v1y = np.where(np.abs(a12) > np.abs(l1 - a11), a12, (a11 - l1))
    norm = np.sqrt(v1x * v1x + v1y * v1y) + 1e-12
    v1x /= norm
    v1y /= norm
    # eigenvectors for l2 are orthogonal
    v2x = -v1y
    v2y = v1x
    return (l1, l2), np.stack([np.stack([v1x, v1y], axis=0), np.stack([v2x, v2y], axis=0)], axis=0)


def compute_inplane_direction(
    binary_mask: np.ndarray,
    is_2d: bool,
    *,
    compute_from: str = "sdt",
    grad_sigma: float = 1.0,
    tensor_sigma: float = 1.5,
    ignore_index: float | int | None = -100,
) -> np.ndarray:
    """
    Compute in-plane unit direction from the structure tensor.

    Algorithm
    ---------
    - Build scalar field (SDT or binary), smooth with grad_sigma
    - Compute gradients and structure tensor J = ∇u ∇u^T
    - Smooth tensor components with tensor_sigma
    - Eigendecompose J and take the eigenvector associated with the smallest eigenvalue
      (tangent/in-plane direction with highest coherence for sheet-like structures)

    Returns
    -------
    np.ndarray
        Unit vector field with shape (2, H, W) or (3, D, H, W); background masked to ignore_index if provided.
    """
    if compute_from.lower() == 'binary':
        base = binary_mask.astype(np.float32)
    else:
        inside = distance_transform_edt(binary_mask)
        outside = distance_transform_edt(1 - binary_mask)
        base = (outside - inside).astype(np.float32)

    u = gaussian_filter(base, sigma=grad_sigma) if grad_sigma and grad_sigma > 0 else base

    if is_2d:
        gy, gx = np.gradient(u)
        Jxx = gx * gx
        Jxy = gx * gy
        Jyy = gy * gy
        if tensor_sigma and tensor_sigma > 0:
            Jxx = gaussian_filter(Jxx, sigma=tensor_sigma)
            Jxy = gaussian_filter(Jxy, sigma=tensor_sigma)
            Jyy = gaussian_filter(Jyy, sigma=tensor_sigma)
        (l1, l2), vecs = _eigh2x2(Jxx, Jxy, Jyy)  # l1<=l2
        # eigenvector for smallest eigenvalue (index 0)
        v = vecs[0]  # shape (2, H, W)
    else:
        gz, gy, gx = np.gradient(u)
        Jxx = gx * gx
        Jxy = gx * gy
        Jxz = gx * gz
        Jyy = gy * gy
        Jyz = gy * gz
        Jzz = gz * gz
        if tensor_sigma and tensor_sigma > 0:
            Jxx = gaussian_filter(Jxx, sigma=tensor_sigma)
            Jxy = gaussian_filter(Jxy, sigma=tensor_sigma)
            Jxz = gaussian_filter(Jxz, sigma=tensor_sigma)
            Jyy = gaussian_filter(Jyy, sigma=tensor_sigma)
            Jyz = gaussian_filter(Jyz, sigma=tensor_sigma)
            Jzz = gaussian_filter(Jzz, sigma=tensor_sigma)
        # Assemble tensor per-voxel and eigendecompose
        # Shape broadcasting; we'll do a naive approach using numpy.linalg.eigh
        H = np.stack
        # Build symmetric matrices (D,H,W,3,3)
        J = np.stack([
            np.stack([Jxx, Jxy, Jxz], axis=-1),
            np.stack([Jxy, Jyy, Jyz], axis=-1),
            np.stack([Jxz, Jyz, Jzz], axis=-1)
        ], axis=-2)  # (..., 3, 3)
        # Move channels last to (...,3,3)
        # Already constructed as (...,3,3)
        # Compute eigenvalues/vectors
        w, vec = np.linalg.eigh(J)  # w ascending; vec (...,3,3); columns are eigenvectors
        # Smallest eigenvector is column 0
        v = np.moveaxis(vec[..., 0], -1, 0)  # (3, D, H, W)

    # Normalize and mask
    eps = 1e-6
    norm = np.sqrt(np.sum(v * v, axis=0, keepdims=True))
    v = v / np.maximum(norm, eps)

    if ignore_index is not None:
        bg = (binary_mask == 0)
        v[:, bg] = float(ignore_index)

    return np.ascontiguousarray(v.astype(np.float32))

