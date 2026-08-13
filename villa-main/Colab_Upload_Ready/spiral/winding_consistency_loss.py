"""Winding consistency loss: L_winding_consistency (C2c).

Penalises deviations between the spiral model's predicted winding number
and the externally-computed psi field.  Unlike C2b (pre-screening), this
loss operates differentiably during optimisation: it samples the psi volume
at each track point's spiral-mapped position and encourages the spiral's
winding assignment to agree with psi(x).

Loss function:
    L = Huber(shifted_radius / dr_per_winding - round(psi_sampled), delta)

where:
    psi_sampled = trilinear sample of psi volume at the track point's
                  scroll-space 3D position (via grid_sample)
    shifted_radius = the spiral's radius after theta correction
    dr_per_winding = radial spacing per winding in spiral space
    delta = 0.3 (allows sub-winding fluctuations without penalty)

The loss fires only when:
    1. A psi volume is loaded (cfg['psi_volume_path'] is set)
    2. loss_weight_winding_consistency > 0
    3. The sampled psi value is valid (> 0.5)

Integration: called from the training step in fit_spiral.py as its own
loss family.  Example:

    if self.psi_volume is not None and self.config['loss_weight_winding_consistency'] > 0:
        wc_loss = winding_consistency_loss(
            self.slice_to_spiral_transform, self.dr_per_winding,
            self.prepared_main_tracks, self.psi_volume, self.config,
        )
        backward_family({
            'winding_consistency': wc_loss * self.config['loss_weight_winding_consistency'],
        })
"""

from __future__ import annotations

import numpy as np
import torch  # type: ignore
import torch.nn.functional as F  # type: ignore

from sample_spiral import get_theta_and_radii


def load_psi_volume(zarr_path: str, device: torch.device) -> dict:
    """Load the psi winding field from an OME-Zarr into a GPU tensor.

    Returns a dict with:
        'tensor': (1, 1, Z, Y, X) float32 tensor on *device*
        'resolution': int, how many fullres voxels per zarr cell
        'shape_zyx': (Z, Y, X) tuple of the zarr array shape
    """
    import zarr  # type: ignore

    store = zarr.open(str(zarr_path), mode="r")
    if isinstance(store, zarr.Array):
        arr = np.asarray(store, dtype=np.float32)
    elif "winding_position" in store:
        arr = np.asarray(store["winding_position"], dtype=np.float32)
    else:
        arr = np.asarray(store, dtype=np.float32)

    resolution = int(store.attrs.get("resolution_factor",
                     store.attrs.get("scaledown", 4)))

    t = torch.from_numpy(arr).to(device=device, dtype=torch.float32)
    # (Z, Y, X) -> (1, 1, Z, Y, X) for F.grid_sample on a 3D volume
    t = t.unsqueeze(0).unsqueeze(0)

    return {
        "tensor": t,
        "resolution": resolution,
        "shape_zyx": arr.shape,  # tuple (Z, Y, X)
    }


def _sample_psi_at_points(
    points_zyx: torch.Tensor,
    psi_vol: dict,
) -> torch.Tensor:
    """Trilinearly sample the psi field at scroll-space ZYX positions.

    F.grid_sample in 3D mode:
        input:  (N_batch, C, D_in, H_in, W_in)   — here (1, 1, Z, Y, X)
        grid:   (N_batch, D_out, H_out, W_out, 3) — last dim is (x, y, z) normalised to [-1,1]
    We have N query points; we reshape them as (1, N, 1, 1, 3).

    Args:
        points_zyx: (N, 3) tensor, full-resolution voxel coords [z, y, x].
        psi_vol: dict from load_psi_volume.

    Returns:
        (N,) tensor of sampled psi values. OOB → 0.
    """
    psi_t = psi_vol["tensor"]   # (1, 1, Z, Y, X)
    res = psi_vol["resolution"]
    Z, Y, X = psi_vol["shape_zyx"]

    # Convert fullres voxel coords → normalised [-1,1] for the zarr grid.
    # The zarr is at 1/res resolution, so divide coords by res first.
    z_norm = points_zyx[:, 0].float() / res / max(1, Z - 1) * 2 - 1
    y_norm = points_zyx[:, 1].float() / res / max(1, Y - 1) * 2 - 1
    x_norm = points_zyx[:, 2].float() / res / max(1, X - 1) * 2 - 1

    # grid_sample 3D expects last dim = (x, y, z)  (not z, y, x!)
    grid = torch.stack([x_norm, y_norm, z_norm], dim=-1)  # (N, 3)
    # Reshape to (1, N, 1, 1, 3) — batch=1, D_out=N, H_out=1, W_out=1
    grid = grid.view(1, -1, 1, 1, 3)

    sampled = F.grid_sample(
        psi_t, grid,
        mode='bilinear', padding_mode='zeros', align_corners=True,
    )
    # sampled shape: (1, 1, N, 1, 1) → (N,)
    return sampled.view(-1)


def winding_consistency_loss(
    slice_to_spiral_transform,
    dr_per_winding: torch.Tensor,
    prepared_tracks: dict | None,
    psi_vol: dict,
    cfg,
    n_sample: int = 4096,
) -> torch.Tensor:
    """Compute the winding consistency loss.

    Steps:
        1. Sub-sample track points from prepared_tracks.
        2. Look up psi(x) at each point (detached — no grad through psi).
        3. Transform points to spiral space and compute shifted_radius via
           get_theta_and_radii (the *same* function that tracks.py uses).
        4. Winding index from spiral:  spiral_w = shifted_radius / dr.
        5. Winding index from psi:     psi_w    = round(psi_sampled).
        6. L = Huber(spiral_w − psi_w, delta=0.3).

    Args:
        slice_to_spiral_transform: Callable that maps (…, 3) scan-space ZYX
            to (…, 3) spiral-space ZYX.
        dr_per_winding: Scalar tensor — radial distance per winding.
        prepared_tracks: Dict with 'flat_zyx_cpu' key → (M, 3) on CPU.
            None → returns zero loss.
        psi_vol: Dict from load_psi_volume().
        cfg: Config dict-like (supports cfg['key'] and cfg.get('key', …)).
        n_sample: Max points to sample per step (default 4096).

    Returns:
        Scalar loss tensor (graph-connected through slice_to_spiral_transform
        and dr_per_winding, so backward_family can accumulate gradients).
    """
    device = dr_per_winding.device
    zero = torch.zeros([], device=device)

    if prepared_tracks is None:
        return zero

    # Get track points.  prepared_tracks stores them on CPU.
    flat_zyx_cpu = prepared_tracks.get('flat_zyx_cpu')
    if flat_zyx_cpu is None or flat_zyx_cpu.shape[0] == 0:
        return zero

    # Sub-sample for efficiency
    n = flat_zyx_cpu.shape[0]
    if n > n_sample:
        idx = torch.randint(0, n, (n_sample,))
        points_zyx = flat_zyx_cpu[idx].to(device)
    else:
        points_zyx = flat_zyx_cpu.to(device)

    # --- psi lookup (detached, no grad) ---
    with torch.no_grad():
        psi_sampled = _sample_psi_at_points(points_zyx, psi_vol)

    # Keep only points where psi is valid (> 0.5 means inside the scroll)
    valid = psi_sampled > 0.5
    n_valid = int(valid.sum().item())
    if n_valid < 10:
        return zero

    psi_valid = psi_sampled[valid]          # (V,)   detached
    points_valid = points_zyx[valid]        # (V, 3) has grad path via transform

    # --- spiral winding assignment (has grad) ---
    # 1. Transform scan-ZYX → spiral-ZYX
    spiral_zyx = slice_to_spiral_transform(points_valid)   # (V, 3)
    # 2. get_theta_and_radii expects the YX part (last two dims) and dr
    #    returns (theta, radius, shifted_radius)
    _theta, _radius, shifted_radius = get_theta_and_radii(
        spiral_zyx[..., 1:], dr_per_winding)

    # Spiral winding index: shifted_radius / dr_per_winding
    spiral_winding = shifted_radius / dr_per_winding    # (V,) — has grad

    # Psi winding index: round(psi) — detached integer target
    psi_winding = torch.round(psi_valid)                # (V,)

    # Residual
    residual = spiral_winding - psi_winding

    # Huber loss (smooth L1) with delta = 0.3 (sub-winding fluctuations OK)
    delta = float(cfg.get('winding_consistency_huber_delta', 0.3))
    loss = F.huber_loss(
        residual,
        torch.zeros_like(residual),
        reduction='mean',
        delta=delta,
    )

    return loss
