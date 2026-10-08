"""Sheet crops through the shared GPU sampler (``shared/crop_sampling/gpu.py``).

A sheet crop counts samples along (normal, front tangent, advance) and is centred ``forward`` ahead of its point; as a
``CropSpec`` its u axis is the tangent, v the normal and f the advance. Sheet frames have columns (t, a, n); crops are
handed to the encoders in (normal, tangent, advance) order.
"""
from __future__ import annotations

import math

import torch

from vesuvius.neural_tracing.tracers.shared.crop_sampling import gpu
from vesuvius.neural_tracing.tracers.shared.geometry import CropSpec


def crop_spec(crop, spacing: float, forward: float) -> CropSpec:
    """The shared crop definition of a sheet crop of (normal, tangent, advance) sample counts."""
    sn, st, sa = (int(c) for c in crop)
    return CropSpec(depth=sa, width=st, height=sn, behind=(sa-1)/2-forward/spacing, spacing=float(spacing))


def crop_offsets(crop, spacing: float, forward: float, device) -> torch.Tensor:
    """Local ``(u, v, f)`` = (tangent, normal, advance) offsets (Sa, Sn, St, 3) of a sheet crop's samples."""
    return gpu.crop_offsets(crop_spec(crop, spacing, forward), device)


def crop_radius(crop, spacing: float, forward: float) -> float:
    """Largest distance of a crop sample from its point."""
    half = [(int(c)-1)/2*float(spacing) for c in crop]
    return math.sqrt(half[0]**2+half[1]**2+(half[2]+abs(forward))**2)


def view_radius(model) -> float:
    """Largest distance from a front point to any sample of its crop (the grid tokens' small crops along the curve
    stay closer)."""
    return crop_radius(model.crop, model.crop_spacing, model.crop_forward)


def sample_crops(block, origin: torch.Tensor, level: int, pos: torch.Tensor, frames: torch.Tensor,
                 offsets: torch.Tensor) -> torch.Tensor:
    """CT crops (B, N, 1, Sn, St, Sa), z-scored per crop, for front points at ``pos`` (B, N, 3) with sheet frames
    (B, N, 3, 3) of columns (t, a, n); ``block`` as in ``sample_block``."""
    crops = gpu.oriented_crops(block, origin, level, pos, frames[..., [0, 2, 1]], offsets)  # u = t, v = n, f = a
    return crops.permute(0, 1, 2, 4, 5, 3)  # (advance, normal, tangent) -> the encoders' (normal, tangent, advance)
