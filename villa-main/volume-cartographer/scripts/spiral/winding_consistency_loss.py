"""Winding consistency loss: L_winding_consistency (C2c).

Penalises deviations between the spiral model's predicted winding number
and the externally-computed psi field.  Unlike C2b (pre-screening), this
loss operates differentiably during optimisation: it samples the psi volume
at each track point's spiral-mapped position and encourages the spiral's
winding assignment to agree with psi(x).

Loss function:
    L = Huber(round(psi_sampled) - winding_index, delta=0.3)

where:
    psi_sampled = trilinear sample of psi volume at the track point's 3D position
    winding_index = floor(radius / dr_per_winding) = integer winding assigned by spiral
    delta = 0.3 (allows sub-winding fluctuations without penalty)

The loss fires only when:
    1. A psi volume is loaded (cfg['psi_volume_path'] is set)
    2. loss_weight_winding_consistency > 0
    3. The sampled psi value is valid (> 0)

Integration: called from the training step in fit_spiral.py, between the
track losses and the dense losses, as its own loss family.

Usage within fit_spiral.py:
    if self.psi_volume is not None and self.config['loss_weight_winding_consistency'] > 0:
        wc_loss = winding_consistency_loss(
            self.slice_to_spiral_transform, self.dr_per_winding,
            self.prepared_main_tracks, self.psi_volume, self.config,
        )
        backward_family({'winding_consistency': wc_loss * self.config['loss_weight_winding_consistency']})
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np


def load_psi_volume(zarr_path: str, device: torch.device) -> dict:
    """Load the psi winding field from an OME-Zarr into a GPU tensor.

    Returns a dict with 'tensor' (1, 1, Z, Y, X), 'resolution', and shape info.
    """
    import zarr

    store = zarr.open(str(zarr_path), mode="r")
    if "winding_position" in store:
        arr = np.asarray(store["winding_position"], dtype=np.float32)
    else:
        arr = np.asarray(store, dtype=np.float32)

    resolution = int(store.attrs.get("resolution_factor",
                     store.attrs.get("scaledown", 4)))

    t = torch.from_numpy(arr).to(device=device, dtype=torch.float32)
    # (Z, Y, X) -> (1, 1, Z, Y, X) for grid_sample
    t = t.unsqueeze(0).unsqueeze(0)

    return {
        "tensor": t,
        "resolution": resolution,
        "shape_zyx": arr.shape,
    }


def _sample_psi_at_points(
    points_zyx: torch.Tensor,
    psi_vol: dict,
) -> torch.Tensor:
    """Sample psi values at 3D points via trilinear interpolation.

    Args:
        points_zyx: (N, 3) tensor of full-resolution voxel coordinates [z, y, x].
        psi_vol: Dict from load_psi_volume with 'tensor', 'resolution', 'shape_zyx'.

    Returns:
        (N,) tensor of psi values. Out-of-bounds points get 0.
    """
    psi_t = psi_vol["tensor"]  # (1, 1, Z, Y, X)
    res = psi_vol["resolution"]
    Z, Y, X = psi_vol["shape_zyx"]

    # Convert full-res voxel coords to normalized grid coords [-1, 1]
    # psi_t is stored at 1/res resolution
    grid = points_zyx.clone().float()  # (N, 3) = [z, y, x]
    grid[:, 0] = grid[:, 0] / res / max(1, Z - 1) * 2 - 1  # z
    grid[:, 1] = grid[:, 1] / res / max(1, Y - 1) * 2 - 1  # y
    grid[:, 2] = grid[:, 2] / res / max(1, X - 1) * 2 - 1  # x

    # grid_sample expects (N, 1, 1, 1, 3) with last dim = (x, y, z)
    # But our grid is (z, y, x), so we need to reverse
    grid_xzy = grid[:, [2, 1, 0]]  # (N, 3) -> [x, y, z]
    grid_5d = grid_xzy.view(1, -1, 1, 1, 3)  # (1, N, 1, 1, 3)

    sampled = F.grid_sample(
        psi_t, grid_5d,
        mode='bilinear', padding_mode='zeros', align_corners=True,
    )
    # sampled: (1, 1, N, 1, 1) -> (N,)
    return sampled.view(-1)


def winding_consistency_loss(
    slice_to_spiral_transform,
    dr_per_winding: torch.Tensor,
    prepared_tracks: dict,
    psi_vol: dict,
    cfg: dict,
    n_sample: int = 4096,
) -> torch.Tensor:
    """Compute the winding consistency loss.

    Samples track points, queries psi(x), and penalises disagreement
    between the spiral's winding assignment and the psi field's winding.

    Args:
        slice_to_spiral_transform: The current slice->spiral coordinate transform.
        dr_per_winding: Scalar tensor, distance in radius units per winding.
        prepared_tracks: Dict with 'flat_zyx_gpu' (N, 3) on device.
        psi_vol: Dict from load_psi_volume.
        cfg: Config dict.
        n_sample: Number of points to sample per step.

    Returns:
        Scalar loss tensor.
    """
    device = dr_per_winding.device
    flat_zyx = prepared_tracks.get('flat_zyx_gpu')
    if flat_zyx is None:
        flat_zyx = prepared_tracks['flat_zyx_cpu'].to(device)

    # Subsample for efficiency
    n = flat_zyx.shape[0]
    if n > n_sample:
        idx = torch.randint(0, n, (n_sample,), device=device)
        points_zyx = flat_zyx[idx]
    else:
        points_zyx = flat_zyx

    # Sample psi at these points
    with torch.no_grad():
        psi_sampled = _sample_psi_at_points(points_zyx.detach(), psi_vol)

    # Mask: only use points where psi > 0 (valid winding region)
    valid = psi_sampled > 0.5
    if valid.sum() < 10:
        return torch.tensor(0.0, device=device, requires_grad=True)

    psi_valid = psi_sampled[valid]
    points_valid = points_zyx[valid]

    # Compute the spiral's winding assignment at these points
    # Transform from ZYX to spiral coordinates to get radius
    from sample_spiral import get_theta_and_radii
    theta, radii = get_theta_and_radii(
        slice_to_spiral_transform, points_valid)

    # Winding index from spiral: radius / dr_per_winding
    spiral_winding = radii / dr_per_winding.detach()

    # Psi winding: round to nearest integer
    psi_winding = torch.round(psi_valid)

    # Residual: difference between spiral assignment and psi assignment
    residual = spiral_winding - psi_winding

    # Huber loss with delta=0.3 (sub-winding fluctuations are OK)
    delta = cfg.get('winding_consistency_huber_delta', 0.3)
    loss = F.huber_loss(residual, torch.zeros_like(residual), delta=delta)

    return loss
