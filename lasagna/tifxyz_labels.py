"""Derive lasagna training channels from tifxyz surface masks on GPU.

All operations use PyTorch CUDA tensors. CuPy EDT is used for distance
transforms via zero-copy DLPack interop.

The 8 output channels are:
  cos (1ch), grad_mag (1ch), dir_z (2ch), dir_y (2ch), dir_x (2ch)
plus a validity mask indicating which voxels have reliable cos/grad_mag.

Reference algorithm: lasagna/labels_to_winding_volume.py lines 236-411.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


TAG = "[tifxyz_labels]"


# ---------------------------------------------------------------------------
# CuPy EDT interop
# ---------------------------------------------------------------------------

def edt_torch(mask: torch.Tensor) -> torch.Tensor:
    """Euclidean distance transform on a CUDA bool/uint8 tensor.

    Uses CuPy EDT via zero-copy DLPack round-trip. Input should be a 3D
    tensor on CUDA. Returns a float32 CUDA tensor of the same shape.
    """
    import cupy as cp
    from cupyx.scipy.ndimage import distance_transform_edt as cupy_edt

    cp_arr = cp.from_dlpack(mask.contiguous())
    dt = cupy_edt(cp_arr, float64_distances=False)
    return torch.from_dlpack(dt).float()


# ---------------------------------------------------------------------------
# Direction encoding
# ---------------------------------------------------------------------------

def encode_direction_channels(
    nx: torch.Tensor,
    ny: torch.Tensor,
    nz: torch.Tensor,
) -> torch.Tensor:
    """Double-angle direction encoding from 3D normal components.

    Returns (6, *spatial) tensor: [dir0_z, dir1_z, dir0_y, dir1_y, dir0_x, dir1_x].
    Each value in [0, 1].
    """
    eps = 1e-8
    inv_sqrt2 = 1.0 / math.sqrt(2.0)

    def _encode_dir(gx: torch.Tensor, gy: torch.Tensor):
        r2 = gx * gx + gy * gy + eps
        cos2t = (gx * gx - gy * gy) / r2
        sin2t = 2.0 * gx * gy / r2
        d0 = 0.5 + 0.5 * cos2t
        d1 = 0.5 + 0.5 * (cos2t - sin2t) * inv_sqrt2
        return d0, d1

    dir0_z, dir1_z = _encode_dir(nx, ny)   # Z-slices (XY plane)
    dir0_y, dir1_y = _encode_dir(nx, nz)   # Y-slices (XZ plane)
    dir0_x, dir1_x = _encode_dir(ny, nz)   # X-slices (YZ plane)
    return torch.stack([dir0_z, dir1_z, dir0_y, dir1_y, dir0_x, dir1_x], dim=0)


# ---------------------------------------------------------------------------
# Greedy chain ordering
# ---------------------------------------------------------------------------

def build_surface_chain(
    surface_masks: list[torch.Tensor],
    dts: list[torch.Tensor],
) -> list[int]:
    """Order N surfaces into a greedy nearest-neighbor chain.

    Starts with the most isolated surface (highest mean distance to all
    others), then greedily appends the closest unused surface.

    Args:
        surface_masks: list of N bool tensors (Z, Y, X)
        dts: list of N float tensors — EDT of complement (~mask)

    Returns:
        chain: list of N indices defining the chain order
    """
    N = len(surface_masks)
    if N <= 1:
        return list(range(N))

    # Pairwise average distances: avg_dist[i, j] = mean(dt_i[mask_j])
    avg_dist = torch.zeros(N, N, dtype=torch.float64)
    for i in range(N):
        dt_i = dts[i]
        for j in range(N):
            if i != j:
                mask_j = surface_masks[j]
                n_j = int(mask_j.sum().item())
                if n_j > 0:
                    avg_dist[i, j] = float(dt_i[mask_j].double().mean().item())

    # Total average distance per surface
    total_avg = torch.zeros(N, dtype=torch.float64)
    for i in range(N):
        vals = [avg_dist[i, j].item() for j in range(N) if i != j]
        total_avg[i] = sum(vals) / max(len(vals), 1)

    # Start with most isolated surface
    chain = [int(total_avg.argmax().item())]
    used = set(chain)

    # Greedily add closest unused
    for _ in range(N - 1):
        last = chain[-1]
        best_j = -1
        best_d = float("inf")
        for j in range(N):
            if j not in used and avg_dist[last, j].item() < best_d:
                best_d = avg_dist[last, j].item()
                best_j = j
        if best_j < 0:
            break
        chain.append(best_j)
        used.add(best_j)

    return chain


# ---------------------------------------------------------------------------
# DT gradient helpers
# ---------------------------------------------------------------------------

def _gradient_3d(vol: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute spatial gradient of a 3D volume via central differences.

    Returns (gz, gy, gx) each with the same shape as vol.
    Uses torch.gradient which handles boundary conditions with forward/backward diffs.
    """
    gz, gy, gx = torch.gradient(vol, dim=(0, 1, 2))
    return gz, gy, gx


def _dot_product_gradients(
    dt_a: torch.Tensor,
    dt_b: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute dot product of gradients of two distance transforms.

    If mask is provided, only computes within masked region (for memory
    efficiency on large volumes). Returns full-shape tensor.
    """
    dot = torch.zeros_like(dt_a)
    for ax in range(3):
        ga = torch.gradient(dt_a, dim=ax)[0]
        gb = torch.gradient(dt_b, dim=ax)[0]
        if mask is not None:
            dot[mask] += (ga[mask] * gb[mask])
        else:
            dot += ga * gb
        del ga, gb
    return dot


# ---------------------------------------------------------------------------
# Core: cos, grad_mag, validity derivation
# ---------------------------------------------------------------------------

def derive_cos_gradmag_validity(
    dts: list[torch.Tensor],
    surface_masks: list[torch.Tensor],
    chain: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Derive cos channel, grad_mag channel, and validity mask from chain-ordered DTs.

    Ports the algorithm from labels_to_winding_volume.py:307-411 to PyTorch.

    Args:
        dts: list of N float32 CUDA tensors — EDT of complement (~mask_k)
        surface_masks: list of N bool CUDA tensors
        chain: ordered surface indices from build_surface_chain()

    Returns:
        cos: (Z, Y, X) float32 — 0.5 + 0.5 * cos(pi * fractional_position)
        grad_mag: (Z, Y, X) float32 — 1 / inter-sheet spacing
        valid: (Z, Y, X) bool — True for voxels between first and last surfaces
    """
    N = len(chain)
    shape = dts[0].shape
    device = dts[0].device

    if N < 2:
        return (
            torch.full(shape, 0.5, device=device),
            torch.zeros(shape, device=device),
            torch.zeros(shape, dtype=torch.bool, device=device),
        )

    # -- Nearest surface per voxel -------------------------------------------
    # Stack DTs for chain surfaces and find nearest
    dt_stack = torch.stack([dts[chain[k]] for k in range(N)], dim=0)  # (N, Z, Y, X)
    dist_nearest, nearest_k = dt_stack.min(dim=0)  # both (Z, Y, X), nearest_k is index into chain

    # -- Build chain neighbor lookup -----------------------------------------
    # For each chain position k, find prev and next chain surface indices
    prev_chain_idx = torch.full((N,), -1, dtype=torch.long, device=device)
    next_chain_idx = torch.full((N,), -1, dtype=torch.long, device=device)
    for k in range(N):
        if k > 0:
            prev_chain_idx[k] = chain[k - 1]
        if k < N - 1:
            next_chain_idx[k] = chain[k + 1]

    # -- Chain-adjacent distances + dot-product side detection ---------------
    dist_prev = torch.full(shape, float("inf"), device=device)
    dist_next = torch.full(shape, float("inf"), device=device)
    dot_prev = torch.zeros(shape, device=device)

    for pos_in_chain in range(N):
        cc_idx = chain[pos_in_chain]
        is_nearest = nearest_k == pos_in_chain
        if not is_nearest.any():
            continue

        dt_near = dts[cc_idx]

        # Prev: distance + dot product of DT gradients
        if pos_in_chain > 0:
            prev_idx = chain[pos_in_chain - 1]
            dt_prev_cc = dts[prev_idx]
            dist_prev[is_nearest] = dt_prev_cc[is_nearest]

            # Dot product of gradients for side detection
            for ax in range(3):
                gn = torch.gradient(dt_near, dim=ax)[0]
                gp = torch.gradient(dt_prev_cc, dim=ax)[0]
                dot_prev[is_nearest] += gn[is_nearest] * gp[is_nearest]
                del gn, gp

        # Next: distance only
        if pos_in_chain < N - 1:
            next_idx = chain[pos_in_chain + 1]
            dist_next[is_nearest] = dts[next_idx][is_nearest]

    # -- Envelope mask (exterior voxels) ------------------------------------
    dt_first = dts[chain[0]]
    dt_last = dts[chain[-1]]
    dot_envelope = torch.zeros(shape, device=device)
    for ax in range(3):
        g1 = torch.gradient(dt_first, dim=ax)[0]
        g2 = torch.gradient(dt_last, dim=ax)[0]
        dot_envelope += g1 * g2
        del g1, g2

    on_any_surface = torch.zeros(shape, dtype=torch.bool, device=device)
    for k in range(N):
        on_any_surface |= surface_masks[chain[k]]
    outside_mask = (dot_envelope > 0) & (~on_any_surface)

    # -- Side detection: use_prev_side = dot_prev < 0 ----------------------
    use_prev_side = dot_prev < 0
    # Last chain surface has no next — force prev side
    is_nearest_last = nearest_k == (N - 1)
    use_prev_side[is_nearest_last] = True
    # need_prev == -1 (first surface) → dot_prev stays 0 → use_prev_side=False → next side ✓

    # -- Bracketing distances -----------------------------------------------
    d_lo = torch.where(use_prev_side, dist_prev, dist_nearest)
    d_hi = torch.where(use_prev_side, dist_nearest, dist_next)
    spacing = d_lo + d_hi

    # -- Cos and grad_mag ---------------------------------------------------
    # cos peaks at 1.0 on surfaces, dips to 0.0 midway
    frac = torch.clamp(d_lo / (spacing * 0.5 + 1e-6), 0.0, 1.0)
    cos = 0.5 + 0.5 * torch.cos(math.pi * frac)

    # grad_mag = inverse inter-sheet spacing
    grad_mag = 1.0 / (spacing + 1e-6)

    # -- Validity mask -------------------------------------------------------
    valid = ~outside_mask

    return cos, grad_mag, valid


# ---------------------------------------------------------------------------
# Top-level: compute all patch labels
# ---------------------------------------------------------------------------

def compute_patch_labels(
    surface_masks: list[torch.Tensor],
    direction_channels: torch.Tensor,
    normals_valid: torch.Tensor,
    device: torch.device = None,
) -> dict[str, torch.Tensor]:
    """Compute all 8 training channels + validity from surface masks and directions.

    This is the top-level function called from the training step. Inputs are
    CUDA tensors produced by the dataset (voxelized surface masks and splatted
    direction channels).

    Args:
        surface_masks: list of N bool tensors (Z, Y, X) — per-surface binary masks
        direction_channels: (6, Z, Y, X) float32 — pre-splatted direction values
        normals_valid: (Z, Y, X) bool — where direction splatting produced valid values
        device: target device (defaults to surface_masks[0].device)

    Returns:
        dict with keys:
            'targets': (8, Z, Y, X) float32 — [cos, grad_mag, dir_z(2), dir_y(2), dir_x(2)]
            'validity': (Z, Y, X) bool — voxels where cos/grad_mag are reliable
            'normals_valid': (Z, Y, X) bool — where directions are valid
    """
    N = len(surface_masks)
    if device is None and N > 0:
        device = surface_masks[0].device

    shape = surface_masks[0].shape if N > 0 else (1, 1, 1)

    if N < 2:
        # Not enough surfaces — cos/grad_mag are undefined
        targets = torch.zeros(8, *shape, device=device)
        # Fill direction channels if available
        if direction_channels is not None and direction_channels.numel() > 0:
            targets[2:8] = direction_channels
        return {
            "targets": targets,
            "validity": torch.zeros(shape, dtype=torch.bool, device=device),
            "normals_valid": normals_valid if normals_valid is not None else torch.zeros(shape, dtype=torch.bool, device=device),
        }

    # EDT of complement for each surface
    dts = []
    for mask in surface_masks:
        complement = ~mask
        dts.append(edt_torch(complement.to(torch.uint8)))

    # Chain ordering
    chain = build_surface_chain(surface_masks, dts)

    # Derive cos, grad_mag, validity
    cos, grad_mag, valid = derive_cos_gradmag_validity(dts, surface_masks, chain)

    # Assemble 8-channel targets
    targets = torch.zeros(8, *shape, device=device)
    targets[0] = cos
    targets[1] = grad_mag
    if direction_channels is not None and direction_channels.numel() > 0:
        targets[2:8] = direction_channels

    return {
        "targets": targets,
        "validity": valid,
        "normals_valid": normals_valid if normals_valid is not None else torch.zeros(shape, dtype=torch.bool, device=device),
    }
