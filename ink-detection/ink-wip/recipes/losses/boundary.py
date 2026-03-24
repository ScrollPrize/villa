from __future__ import annotations

from dataclasses import dataclass

import torch

from ink.recipes.stitch.artifact_primitives import binary_mask_to_signed_distance_map


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

    def compute(self, batch):
        boundary_dist_map = batch.boundary_dist_map
        if boundary_dist_map is None:
            raise RuntimeError("boundary_dist_map is required when StitchBoundaryLoss is enabled")
        boundary_loss = compute_binary_boundary_loss(
            batch.logits[None, None],
            boundary_dist_map[None, None],
            valid_mask=batch.valid_mask[None, None],
            reduction_dims=(1, 2, 3),
        )[0]
        return {
            "loss": float(self.weight) * boundary_loss,
            "boundary_loss": boundary_loss,
        }
