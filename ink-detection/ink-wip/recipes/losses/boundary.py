from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy import ndimage


def binary_mask_to_signed_distance_map(mask, *, dtype=np.float32):
    mask_np = np.asarray(mask, dtype=np.bool_)
    if mask_np.ndim != 2:
        raise ValueError(f"binary_mask_to_signed_distance_map expects a 2D mask, got shape={tuple(mask_np.shape)}")
    if not bool(mask_np.any()):
        return np.zeros(mask_np.shape, dtype=dtype)

    negmask = ~mask_np
    dist_out = ndimage.distance_transform_edt(negmask)
    dist_in = ndimage.distance_transform_edt(mask_np)
    signed_dist = dist_out * negmask - (dist_in - 1.0) * mask_np
    return np.asarray(signed_dist, dtype=dtype)


def compute_binary_boundary_loss(
    logits,
    dist_map,
    *,
    valid_mask=None,
    reduction_dims=(1, 2, 3),
):
    probs = torch.sigmoid(logits)
    dist_map = dist_map.to(device=probs.device, dtype=probs.dtype)
    if tuple(dist_map.shape) != tuple(probs.shape):
        raise ValueError(f"dist_map shape must match logits shape, got {tuple(dist_map.shape)} vs {tuple(probs.shape)}")

    if valid_mask is None:
        valid_mask = torch.ones_like(probs, dtype=probs.dtype)
    else:
        valid_mask = valid_mask.to(device=probs.device, dtype=probs.dtype)
        if tuple(valid_mask.shape) != tuple(probs.shape):
            raise ValueError(
                f"valid_mask shape must match logits shape, got {tuple(valid_mask.shape)} vs {tuple(probs.shape)}"
            )

    reduce_dims = tuple(int(dim) for dim in reduction_dims)
    denom = valid_mask.sum(dim=reduce_dims).clamp_min(1.0)
    return (probs * dist_map * valid_mask).sum(dim=reduce_dims) / denom


@dataclass(frozen=True)
class StitchBoundaryLoss:
    weight: float = 1.0

    @property
    def requires_boundary_dist_map(self) -> bool:
        return True

    def compute(self, stitched_logits, _stitched_targets, *, valid_mask, boundary_dist_map=None, **_kwargs):
        if boundary_dist_map is None:
            raise RuntimeError("boundary_dist_map is required when StitchBoundaryLoss is enabled")
        boundary_loss = compute_binary_boundary_loss(
            stitched_logits[None, None],
            boundary_dist_map[None, None],
            valid_mask=valid_mask[None, None],
            reduction_dims=(1, 2, 3),
        )[0]
        return {
            "loss": float(self.weight) * boundary_loss,
            "boundary_loss": boundary_loss,
        }
