"""Oriented per-point CT crops sampled on the GPU from an episode's volume block.

A block is a ``(B, 1, D, H, W)`` tensor of one pyramid level with its level-voxel ``z, y, x`` origin. Level
voxel ``i`` is centred on level-0 coordinate ``(i + 0.5) * 2**level - 0.5`` (mean-pooled pyramid). Crops are
laid out ``(normal, tangent, advance)`` and z-scored per crop, matching fiber_follow's crop normalization.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from vesuvius.neural_tracing.fiber_follow.data.ct_normalization import ZSCORE_EPSILON


def crop_offsets(crop, spacing: float, forward: float, device) -> torch.Tensor:
    """Local ``(t, a, n)`` offsets (Sn, St, Sa, 3) of the crop samples, ``spacing`` apart along every axis."""
    sn, st, sa = (int(c) for c in crop)
    axis = lambda count: (torch.arange(count, device=device, dtype=torch.float32)-(count-1)/2)*float(spacing)
    n, t, a = torch.meshgrid(axis(sn), axis(st), axis(sa)+forward, indexing='ij')
    return torch.stack([t, a, n], -1)


def crop_radius(crop, spacing: float, forward: float) -> float:
    """Largest distance of a crop sample from its point."""
    half = [(int(c)-1)/2*float(spacing) for c in crop]
    return math.sqrt(half[0]**2+half[1]**2+(half[2]+abs(forward))**2)


def view_radius(model) -> float:
    """Largest distance from a front point to any sample of its crop (the grid tokens' small crops along the curve
    stay closer)."""
    return crop_radius(model.crop, model.crop_spacing, model.crop_forward)


def to_block_grid(points: torch.Tensor, origin: torch.Tensor, level: int, size) -> torch.Tensor:
    """Level-0 ``xyz`` points (B, ..., 3) -> grid_sample coordinates of a block with ``zyx`` origin (B, 3)."""
    scale = float(1 << level)
    idx = (points+.5)/scale-.5-origin.flip(-1).to(points.dtype).view(origin.shape[0], *([1]*(points.ndim-2)), 3)
    return 2*idx/(_extent(tuple(int(v) for v in size), points.device, points.dtype)-1)-1


_EXTENTS: dict = {}


def _extent(size, device, dtype) -> torch.Tensor:
    """Block extent as a cached device tensor: creating it per call is a host-to-device copy that stalls the GPU
    queue, and dividing by Python scalars instead would round differently (CUDA multiplies by the reciprocal)."""
    key = (size, device, dtype)
    if key not in _EXTENTS:
        if len(_EXTENTS) > 4096:  # unpadded blocks (data.unpadded_blocks) bring new sizes every batch
            _EXTENTS.clear()
        _EXTENTS[key] = torch.tensor([size[2], size[1], size[0]], device=device, dtype=dtype)
    return _EXTENTS[key]


def sample_block(volume, origin: torch.Tensor, level: int, points: torch.Tensor) -> torch.Tensor:
    """Bilinear samples (B, C, d1, d2, d3) at level-0 points (B, d1, d2, d3, 3) of a batch of blocks: one
    (B, C, D, H, W) tensor, or a list of B (C, D, H, W) blocks of their own sizes (``data.unpadded_blocks``), each
    sampled through its own extent (same values; padding to a common size can cost far more memory than the blocks)."""
    if isinstance(volume, torch.Tensor):
        grid = to_block_grid(points, origin, level, volume.shape[2:])
        return F.grid_sample(volume, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    return torch.cat([F.grid_sample(v[None], to_block_grid(points[i:i+1], origin[i:i+1], level, v.shape[1:]),
                                    mode='bilinear', padding_mode='zeros', align_corners=True)
                      for i, v in enumerate(volume)])


def sample_crops(block, origin: torch.Tensor, level: int, pos: torch.Tensor, frames: torch.Tensor,
                 offsets: torch.Tensor) -> torch.Tensor:
    """CT crops (B, N, 1, Sn, St, Sa), z-scored per crop, for front_points at ``pos`` (B, N, 3) with frames
    (B, N, 3, 3); ``block`` as in ``sample_block``."""
    B, N = pos.shape[:2]
    sn, st, sa = offsets.shape[:3]
    world = (pos[:, :, None, None, None]+torch.einsum('bnij,stuj->bnstui', frames, offsets)).reshape(B, N*sn, st, sa, 3)
    ct = sample_block(block, origin, level, world).reshape(B, 1, N, sn, st, sa).transpose(1, 2)
    mean = ct.mean((-3, -2, -1), keepdim=True)
    std = ct.std((-3, -2, -1), keepdim=True, unbiased=False)
    return (ct-mean)/std.clamp_min(ZSCORE_EPSILON)


def sample_plane(block: torch.Tensor, origin: torch.Tensor, level: int, center: torch.Tensor, axes: torch.Tensor,
                 extent: tuple[float, float], spacing: float) -> torch.Tensor:
    """Raw intensities on a plane (H, W) of one block (1, D, H, W) through ``center`` (3,) spanned by ``axes`` (2, 3)."""
    h = int(extent[0]/spacing)
    w = int(extent[1]/spacing)
    ii = (torch.arange(h, device=block.device)-(h-1)/2)*spacing
    jj = (torch.arange(w, device=block.device)-(w-1)/2)*spacing
    pts = center+ii[:, None, None]*axes[0]+jj[None, :, None]*axes[1]
    grid = to_block_grid(pts[None, None], origin[None], level, block.shape[1:])
    return F.grid_sample(block[None], grid, mode='bilinear', padding_mode='zeros', align_corners=True)[0, 0, 0]
