"""Oriented crops sampled on the GPU from a CT block already on the device: the backend for tracers whose crop
positions come from the model inside a training step (the sheet tracer's rollouts). Crops read from a volume use
vc_reader (crops.py).

Same crop definition (``CropSpec``) and frame convention (columns ``u, v, f``) as the CPU sampler; crops are laid out
``(D, H, W)`` = forward (f), v, u, trilinear with zeros outside the block through vc_reader. A block
is a ``(B, 1, D, H, W)`` tensor of one pyramid level with its level-voxel ``z, y, x`` origin; level voxel ``i`` is
centred on level-0 coordinate ``(i + 0.5) * 2**level - 0.5`` (mean-pooled pyramid).
"""
from __future__ import annotations

import torch
from vc_reader.torch import sample_array

from vesuvius.neural_tracing.tracers.shared.crop_sampling.normalization import zscore_crops
from vesuvius.neural_tracing.tracers.shared.geometry import CropSpec


def crop_offsets(crop: CropSpec, device) -> torch.Tensor:
    """Local ``(u, v, f)`` offsets (D, H, W, 3) of the crop samples."""
    axis = lambda values: torch.as_tensor(values, dtype=torch.float32, device=device)
    f, v, u = torch.meshgrid(axis(crop.forward_coords), axis(crop.vertical_coords), axis(crop.lateral_coords),
                             indexing='ij')
    return torch.stack([u, v, f], -1)


def sample_block(volume, origin: torch.Tensor, level: int, points: torch.Tensor) -> torch.Tensor:
    """Trilinear samples (B, C, d1, d2, d3) at level-0 points (B, d1, d2, d3, 3) of a batch of blocks: one
    (B, C, D, H, W) tensor, or a list of B (C, D, H, W) blocks of their own sizes, each sampled through its own extent
    (same values; padding to a common size can cost far more memory than the blocks)."""
    def indices(p, o):
        scale = float(1 << level)
        return (p.flip(-1) + .5) / scale - .5 - o.to(p.dtype).view(o.shape[0], *([1]*(p.ndim-2)), 3)
    if isinstance(volume, torch.Tensor):
        return sample_array(volume, indices(points, origin))
    return torch.cat([sample_array(v[None], indices(points[i:i+1], origin[i:i+1]))
                      for i, v in enumerate(volume)])


def oriented_crops(block, origin: torch.Tensor, level: int, pos: torch.Tensor, frames: torch.Tensor,
                   offsets: torch.Tensor, zscore: bool = True) -> torch.Tensor:
    """Crops (B, N, 1, D, H, W), z-scored per crop (``zscore_crops``; raw values with ``zscore=False``), at ``pos``
    (B, N, 3) with frames (B, N, 3, 3) of columns ``u, v, f``; ``offsets`` from ``crop_offsets``, ``block`` as in
    ``sample_block``."""
    B, N = pos.shape[:2]
    d, h, w = offsets.shape[:3]
    world = (pos[:, :, None, None, None]+torch.einsum('bnij,dhwj->bndhwi', frames, offsets)).reshape(B, N*d, h, w, 3)
    crops = sample_block(block, origin, level, world).reshape(B, 1, N, d, h, w).transpose(1, 2)
    return zscore_crops(crops) if zscore else crops


def sample_plane(block: torch.Tensor, origin: torch.Tensor, level: int, center: torch.Tensor, axes: torch.Tensor,
                 extent: tuple[float, float], spacing: float) -> torch.Tensor:
    """Raw intensities on a plane (H, W) of one block (1, D, H, W) through ``center`` (3,) spanned by ``axes`` (2, 3)."""
    h = int(extent[0]/spacing)
    w = int(extent[1]/spacing)
    ii = (torch.arange(h, device=block.device)-(h-1)/2)*spacing
    jj = (torch.arange(w, device=block.device)-(w-1)/2)*spacing
    pts = center+ii[:, None, None]*axes[0]+jj[None, :, None]*axes[1]
    return sample_block(block[None], origin[None], level, pts[None, None])[0, 0, 0]
