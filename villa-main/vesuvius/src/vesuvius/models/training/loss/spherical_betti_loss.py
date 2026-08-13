"""
Spherical Betti Matching Loss for topological supervision via 2D plane slicing.

Samples 2D planes through 3D volumes at uniform orientations (Fibonacci sphere),
extracts slices, and applies Betti matching to compare H0/H1 topology between
predictions and ground truth.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .betti_losses import _load_betti_module

bm = _load_betti_module()


def _generate_fibonacci_directions(
    num_directions: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Fibonacci sphere sampling - optimal uniform coverage for any N.

    Uses the golden ratio to distribute points evenly on a unit sphere.
    Returns (3, num_directions) tensor of unit vectors.
    """
    indices = torch.arange(num_directions, device=device, dtype=dtype)
    phi = (1 + 5**0.5) / 2  # Golden ratio

    theta = 2 * math.pi * indices / phi
    z = 1 - 2 * (indices + 0.5) / num_directions
    r = torch.sqrt(1 - z**2)

    x = r * torch.cos(theta)
    y = r * torch.sin(theta)

    return torch.stack([x, y, z], dim=0)  # (3, num_directions)


def _axis_aligned_normals(device: torch.device, dtype: torch.dtype) -> Tensor:
    """Return the 3 axis-aligned plane normals: XY, XZ, YZ planes.

    Returns (3, 3) tensor where each column is a unit normal.
    - Column 0: Z-axis normal (XY plane slice)
    - Column 1: Y-axis normal (XZ plane slice)
    - Column 2: X-axis normal (YZ plane slice)
    """
    return torch.tensor([
        [0.0, 0.0, 1.0],  # XY plane (normal = Z)
        [0.0, 1.0, 0.0],  # XZ plane (normal = Y)
        [1.0, 0.0, 0.0],  # YZ plane (normal = X)
    ], device=device, dtype=dtype).T  # (3, 3)


def _orthonormal_basis(normal: Tensor) -> Tuple[Tensor, Tensor]:
    """Compute two orthonormal vectors perpendicular to the given normal.

    Uses Gram-Schmidt-like approach with a reference vector.

    Args:
        normal: (3,) unit vector

    Returns:
        u, v: (3,) unit vectors forming an orthonormal basis with normal
    """
    # Choose reference vector not parallel to normal
    if abs(normal[0].item()) < 0.9:
        ref = torch.tensor([1.0, 0.0, 0.0], device=normal.device, dtype=normal.dtype)
    else:
        ref = torch.tensor([0.0, 1.0, 0.0], device=normal.device, dtype=normal.dtype)

    # Gram-Schmidt
    u = ref - (ref @ normal) * normal
    u = u / u.norm().clamp_min(1e-8)

    v = torch.linalg.cross(normal, u)
    v = v / v.norm().clamp_min(1e-8)

    return u, v


def _slice_volume(
    volume: Tensor,
    normal: Tensor,
    slice_size: int,
) -> Tensor:
    """Extract 2D slice through volume center, perpendicular to normal.

    Uses grid_sample with trilinear interpolation.

    Args:
        volume: (B, D, H, W) tensor
        normal: (3,) unit vector - plane normal
        slice_size: Output size (slice_size x slice_size)

    Returns:
        (B, slice_size, slice_size) tensor
    """
    B, D, H, W = volume.shape
    device = volume.device
    dtype = volume.dtype

    # Get orthonormal basis for the slice plane
    u, v = _orthonormal_basis(normal)

    # Create 2D grid in [-1, 1] normalized slice coordinates
    lin = torch.linspace(-1, 1, slice_size, device=device, dtype=dtype)
    grid_s, grid_t = torch.meshgrid(lin, lin, indexing='ij')  # (slice_size, slice_size)

    # Map to 3D coordinates: p = center + s*u + t*v
    # grid_sample expects coordinates in (x, y, z) = (W, H, D) order and range [-1, 1]
    # Our u, v are in (D, H, W) order, so we need to be careful

    # Reshape for broadcasting: (slice_size, slice_size, 1)
    grid_s = grid_s.unsqueeze(-1)
    grid_t = grid_t.unsqueeze(-1)

    # Scale by volume extent (assuming isotropic for simplicity)
    # u, v are unit vectors, scale by normalized extent
    extent = torch.tensor([D, H, W], device=device, dtype=dtype)
    scale = extent.min() / 2.0  # Radius of largest inscribed sphere

    # 3D coordinates in volume space (normalized to [-1, 1])
    # u and v define directions in (D, H, W) space
    coords_3d = grid_s * u * scale / (extent / 2) + grid_t * v * scale / (extent / 2)
    # coords_3d shape: (slice_size, slice_size, 3) in (D, H, W) order

    # grid_sample expects (N, D_out, H_out, W_out, 3) with last dim as (x, y, z) = (W, H, D)
    # So we need to reverse the order
    grid = coords_3d[..., [2, 1, 0]]  # (slice_size, slice_size, 3) now in (W, H, D) order

    # Expand for batch and add depth dimension (we're sampling a 2D slice)
    # grid_sample for 3D expects (N, D_out, H_out, W_out, 3)
    # We want D_out=1 since it's a single slice
    grid = grid.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1, -1)  # (B, 1, slice_size, slice_size, 3)

    # Add channel dimension for grid_sample
    volume_5d = volume.unsqueeze(1)  # (B, 1, D, H, W)

    # Sample
    sampled = F.grid_sample(
        volume_5d,
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True,
    )  # (B, 1, 1, slice_size, slice_size)

    return sampled[:, 0, 0, :, :]  # (B, slice_size, slice_size)


def _slice_volume_axis_aligned(
    volume: Tensor,
    axis: int,
    slice_size: int,
) -> Tensor:
    """Extract axis-aligned 2D slice through volume center.

    No interpolation needed - just index the middle slice.

    Args:
        volume: (B, D, H, W) tensor
        axis: 0=XY (slice along D), 1=XZ (slice along H), 2=YZ (slice along W)
        slice_size: Output size (will resize if needed)

    Returns:
        (B, slice_size, slice_size) tensor
    """
    B, D, H, W = volume.shape

    if axis == 0:  # XY plane - slice along D (depth)
        idx = D // 2
        sliced = volume[:, idx, :, :]  # (B, H, W)
    elif axis == 1:  # XZ plane - slice along H
        idx = H // 2
        sliced = volume[:, :, idx, :]  # (B, D, W)
    else:  # YZ plane - slice along W
        idx = W // 2
        sliced = volume[:, :, :, idx]  # (B, D, H)

    # Resize if needed
    if sliced.shape[-2:] != (slice_size, slice_size):
        sliced = F.interpolate(
            sliced.unsqueeze(1),
            size=(slice_size, slice_size),
            mode='bilinear',
            align_corners=True,
        ).squeeze(1)

    return sliced


def _to_numpy(t: Tensor) -> np.ndarray:
    """Convert tensor to contiguous float64 numpy array."""
    return np.ascontiguousarray(t.detach().cpu().numpy().astype(np.float64))


def _filter_coords_by_mask(
    coords: np.ndarray,
    mask: Optional[Tensor],
    threshold: float = 0.5,
) -> np.ndarray:
    """Return boolean array indicating which coordinates are in valid (non-ignored) regions.

    Args:
        coords: (N, 2) array of 2D coordinates
        mask: (H, W) tensor where values >= threshold are valid, or None
        threshold: Threshold for considering a region valid (after interpolation)

    Returns:
        (N,) boolean array, True if coordinate is in valid region
    """
    if mask is None or coords.size == 0:
        return np.ones(coords.shape[0], dtype=bool)

    H, W = mask.shape
    mask_np = mask.detach().cpu().numpy()

    # Clamp coordinates to valid range
    row = np.clip(coords[:, 0].astype(np.int64), 0, H - 1)
    col = np.clip(coords[:, 1].astype(np.int64), 0, W - 1)

    # Check if coordinates are in valid regions
    return mask_np[row, col] >= threshold


def _compute_loss_from_result_with_mask(
    pred_field: Tensor,
    tgt_field: Tensor,
    res,
    mask: Optional[Tensor],
    *,
    include_unmatched_target: bool,
    push_to: str,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Compute loss from Betti matching result, filtering by mask.

    Excludes persistence pairs whose birth or death coordinates fall in ignored regions.
    """
    from .betti_losses import _tensor_values_at_coords, _stack_pairs, _loss_unmatched

    def _concat_and_filter(list_of_arrays, mask_tensor):
        """Concatenate arrays and return filtered coordinates + validity mask."""
        flat = []
        if list_of_arrays is not None:
            for a in list_of_arrays:
                if a is None:
                    continue
                if isinstance(a, (list, tuple)):
                    for b in a:
                        if isinstance(b, np.ndarray) and b.size > 0:
                            flat.append(b)
                elif isinstance(a, np.ndarray) and a.size > 0:
                    flat.append(a)

        if len(flat) == 0:
            return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=bool)

        coords = np.ascontiguousarray(np.concatenate(flat, axis=0))
        valid = _filter_coords_by_mask(coords, mask_tensor)
        return coords, valid

    # Get matched coordinates and filter
    pred_birth, pred_birth_valid = _concat_and_filter(res.input1_matched_birth_coordinates, mask)
    pred_death, pred_death_valid = _concat_and_filter(res.input1_matched_death_coordinates, mask)
    tgt_birth, tgt_birth_valid = _concat_and_filter(res.input2_matched_birth_coordinates, mask)
    tgt_death, tgt_death_valid = _concat_and_filter(res.input2_matched_death_coordinates, mask)

    # A matched pair is valid only if ALL four coordinates are in valid regions
    matched_valid = pred_birth_valid & pred_death_valid & tgt_birth_valid & tgt_death_valid

    # Filter matched coordinates
    pred_birth = pred_birth[matched_valid]
    pred_death = pred_death[matched_valid]
    tgt_birth = tgt_birth[matched_valid]
    tgt_death = tgt_death[matched_valid]

    # Compute matched loss
    if pred_birth.size == 0:
        loss_matched = pred_field.new_zeros(())
    else:
        pred_birth_vals = _tensor_values_at_coords(pred_field, pred_birth)
        pred_death_vals = _tensor_values_at_coords(pred_field, pred_death)
        tgt_birth_vals = _tensor_values_at_coords(tgt_field, tgt_birth)
        tgt_death_vals = _tensor_values_at_coords(tgt_field, tgt_death)
        pred_matched_pairs = _stack_pairs(pred_birth_vals, pred_death_vals)
        tgt_matched_pairs = _stack_pairs(tgt_birth_vals, tgt_death_vals)
        loss_matched = 2.0 * ((pred_matched_pairs - tgt_matched_pairs) ** 2).sum()

    # Get unmatched prediction coordinates and filter
    pred_unmatched_birth, ub_valid = _concat_and_filter(res.input1_unmatched_birth_coordinates, mask)
    pred_unmatched_death, ud_valid = _concat_and_filter(res.input1_unmatched_death_coordinates, mask)
    unmatched_pred_valid = ub_valid & ud_valid

    pred_unmatched_birth = pred_unmatched_birth[unmatched_pred_valid]
    pred_unmatched_death = pred_unmatched_death[unmatched_pred_valid]

    if pred_unmatched_birth.size == 0:
        loss_unmatched_pred = pred_field.new_zeros(())
    else:
        pred_unmatched_birth_vals = _tensor_values_at_coords(pred_field, pred_unmatched_birth)
        pred_unmatched_death_vals = _tensor_values_at_coords(pred_field, pred_unmatched_death)
        pred_unmatched_pairs = _stack_pairs(pred_unmatched_birth_vals, pred_unmatched_death_vals)
        loss_unmatched_pred = _loss_unmatched(pred_unmatched_pairs, push_to=push_to)

    total = loss_matched + loss_unmatched_pred

    loss_unmatched_tgt = pred_field.new_zeros(())
    if include_unmatched_target:
        tgt_unmatched_birth, vb_valid = _concat_and_filter(res.input2_unmatched_birth_coordinates, mask)
        tgt_unmatched_death, vd_valid = _concat_and_filter(res.input2_unmatched_death_coordinates, mask)
        unmatched_tgt_valid = vb_valid & vd_valid

        tgt_unmatched_birth = tgt_unmatched_birth[unmatched_tgt_valid]
        tgt_unmatched_death = tgt_unmatched_death[unmatched_tgt_valid]

        if tgt_unmatched_birth.size > 0:
            tgt_unmatched_birth_vals = _tensor_values_at_coords(tgt_field, tgt_unmatched_birth)
            tgt_unmatched_death_vals = _tensor_values_at_coords(tgt_field, tgt_unmatched_death)
            tgt_unmatched_pairs = _stack_pairs(tgt_unmatched_birth_vals, tgt_unmatched_death_vals)
            loss_unmatched_tgt = _loss_unmatched(tgt_unmatched_pairs, push_to=push_to)
            total = total + loss_unmatched_tgt

    aux = {
        "Betti matching loss (matched)": loss_matched.reshape(1).detach(),
        "Betti matching loss (unmatched prediction)": loss_unmatched_pred.reshape(1).detach(),
    }
    if include_unmatched_target:
        aux["Betti matching loss (unmatched target)"] = loss_unmatched_tgt.reshape(1).detach()

    return total.reshape(1), aux


class SphericalBettiLoss(nn.Module):
    """
    Betti matching loss on 2D slices sampled from spherical orientations.

    Samples planes through 3D volumes at uniform orientations (Fibonacci sphere
    plus axis-aligned), extracts 2D slices, and applies Betti matching to compare
    H0/H1 persistent homology between predictions and ground truth.

    Parameters:
        num_directions: Number of Fibonacci-sampled plane orientations (in addition
            to 3 axis-aligned slices). Total = 3 + num_directions.
        filtration: 'superlevel', 'sublevel', or 'bothlevel'
        include_unmatched_target: Whether to penalize unmatched target pairs
        push_unmatched_to: 'diagonal', 'one_zero', or 'death_death'
        ignore_label: Optional label value to ignore. Voxels with this label in
            the target will be masked out (set to 0 in both pred and target slices).
    """

    def __init__(
        self,
        num_directions: int = 32,
        filtration: str = 'superlevel',
        include_unmatched_target: bool = False,
        push_unmatched_to: str = 'diagonal',
        ignore_label: Optional[int] = None,
    ):
        super().__init__()
        assert filtration in ('superlevel', 'sublevel', 'bothlevel')
        assert push_unmatched_to in ('diagonal', 'one_zero', 'death_death')

        self.num_directions = num_directions
        self.filtration = filtration
        self.include_unmatched_target = include_unmatched_target
        self.push_unmatched_to = push_unmatched_to
        self.ignore_label = ignore_label

        # Cache for directions
        self._directions_cache: Dict[Tuple[int, torch.device, torch.dtype], Tensor] = {}

    def _get_directions(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Get cached Fibonacci directions + axis-aligned normals."""
        key = (self.num_directions, device, dtype)
        if key not in self._directions_cache:
            fib = _generate_fibonacci_directions(self.num_directions, device, dtype)
            axis = _axis_aligned_normals(device, dtype)
            self._directions_cache[key] = torch.cat([axis, fib], dim=1)  # (3, 3+num_directions)
        return self._directions_cache[key]

    def forward(
        self,
        input: Tensor,
        target: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Compute spherical Betti matching loss.

        Args:
            input: (B, C, D, H, W) logits. For C==2, uses softmax and foreground.
                   For C==1, applies sigmoid if needed.
            target: (B, 1, D, H, W) or (B, D, H, W) binary target

        Returns:
            loss: scalar tensor
            aux: dictionary of auxiliary losses
        """
        device = input.device
        batch_size = input.shape[0]
        num_channels = input.shape[1]

        if batch_size == 0:
            return input.new_zeros(()), {}

        # Get raw target for ignore mask before any processing
        if target.dim() == 5:
            raw_target = target[:, 0] if target.shape[1] == 1 else target.argmax(dim=1)
        else:
            raw_target = target  # (B, D, H, W)

        # Create ignore mask if needed: True = valid (not ignored), False = ignored
        if self.ignore_label is not None:
            valid_mask = (raw_target != self.ignore_label).float()  # (B, D, H, W)
        else:
            valid_mask = None

        # Extract foreground probabilities
        if num_channels == 2:
            probs = torch.softmax(input, dim=1)
            pred_fg = probs[:, 1]  # (B, D, H, W)
            tgt_fg = target[:, 1] if target.dim() == 5 and target.shape[1] == 2 else target.squeeze(1)
        else:
            pred_fg = torch.sigmoid(input[:, 0]) if not (input.min() >= 0 and input.max() <= 1) else input[:, 0]
            tgt_fg = target.squeeze(1) if target.dim() == 5 else target

        # Ensure float for interpolation
        pred_fg = pred_fg.float()
        tgt_fg = tgt_fg.float()

        B, D, H, W = pred_fg.shape
        slice_size = min(D, H, W)

        # Get all directions (axis-aligned + Fibonacci)
        directions = self._get_directions(device, torch.float32)
        num_total_dirs = directions.shape[1]

        # Collect all slices for batched bm.compute_matching call
        pred_slices: List[Tensor] = []
        tgt_slices: List[Tensor] = []
        mask_slices: List[Optional[Tensor]] = []

        # Process axis-aligned slices (first 3 directions)
        for axis in range(3):
            for b in range(batch_size):
                pred_slices.append(_slice_volume_axis_aligned(pred_fg[b:b+1], axis, slice_size)[0])
                tgt_slices.append(_slice_volume_axis_aligned(tgt_fg[b:b+1], axis, slice_size)[0])
                if valid_mask is not None:
                    mask_slices.append(_slice_volume_axis_aligned(valid_mask[b:b+1], axis, slice_size)[0])
                else:
                    mask_slices.append(None)

        # Process Fibonacci-sampled directions (remaining directions)
        for d in range(3, num_total_dirs):
            normal = directions[:, d]
            for b in range(batch_size):
                pred_slices.append(_slice_volume(pred_fg[b:b+1], normal, slice_size)[0])
                tgt_slices.append(_slice_volume(tgt_fg[b:b+1], normal, slice_size)[0])
                if valid_mask is not None:
                    mask_slices.append(_slice_volume(valid_mask[b:b+1], normal, slice_size)[0])
                else:
                    mask_slices.append(None)

        if len(pred_slices) == 0:
            return input.new_zeros(()), {}

        # Apply filtration transformation
        # For superlevel/bothlevel: invert values (high values appear first in filtration)
        # For sublevel: use original values
        if self.filtration in ('superlevel', 'bothlevel'):
            pred_proc = [1.0 - s for s in pred_slices]
            tgt_proc = [1.0 - s for s in tgt_slices]
        else:  # sublevel
            pred_proc = pred_slices
            tgt_proc = tgt_slices

        # Convert to numpy for batched bm.compute_matching call
        pred_np = [_to_numpy(s) for s in pred_proc]
        tgt_np = [_to_numpy(s) for s in tgt_proc]

        # Single batched call to compute_matching
        results = bm.compute_matching(
            pred_np,
            tgt_np,
            include_input1_unmatched_pairs=True,
            include_input2_unmatched_pairs=self.include_unmatched_target,
        )

        # Compute losses from results (with mask filtering)
        all_losses: List[Tensor] = []
        all_aux: List[Dict[str, Tensor]] = []

        for i, res in enumerate(results):
            loss_i, aux_i = _compute_loss_from_result_with_mask(
                pred_proc[i], tgt_proc[i], res, mask_slices[i],
                include_unmatched_target=self.include_unmatched_target,
                push_to=self.push_unmatched_to,
            )
            all_losses.append(loss_i)
            all_aux.append(aux_i)

        # Handle bothlevel: also compute sublevel and average
        if self.filtration == 'bothlevel':
            # Sublevel uses original (non-inverted) slices
            pred_np_sub = [_to_numpy(s) for s in pred_slices]
            tgt_np_sub = [_to_numpy(s) for s in tgt_slices]

            results_sub = bm.compute_matching(
                pred_np_sub,
                tgt_np_sub,
                include_input1_unmatched_pairs=True,
                include_input2_unmatched_pairs=self.include_unmatched_target,
            )

            for i, res in enumerate(results_sub):
                loss_i, aux_i = _compute_loss_from_result_with_mask(
                    pred_slices[i], tgt_slices[i], res, mask_slices[i],
                    include_unmatched_target=self.include_unmatched_target,
                    push_to=self.push_unmatched_to,
                )
                # Average with superlevel result
                all_losses[i] = 0.5 * (all_losses[i] + loss_i)
                for k in aux_i:
                    if k in all_aux[i]:
                        all_aux[i][k] = 0.5 * (all_aux[i][k] + aux_i[k])

        # Aggregate across all slices
        loss = torch.stack([l.reshape(()) for l in all_losses]).mean()

        # Aggregate auxiliary metrics
        aux_agg: Dict[str, Tensor] = {}
        if all_aux:
            all_keys = set().union(*(d.keys() for d in all_aux))
            for k in all_keys:
                values = [d[k] for d in all_aux if k in d]
                if values:
                    aux_agg[k] = torch.stack([v.reshape(()) for v in values]).mean()

        return loss, aux_agg
