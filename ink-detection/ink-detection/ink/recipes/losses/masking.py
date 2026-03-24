from __future__ import annotations

import torch


def resolve_valid_mask(targets: torch.Tensor, valid_mask) -> torch.Tensor:
    if valid_mask is None:
        return torch.ones_like(targets, dtype=torch.float32)

    valid_mask = valid_mask.to(device=targets.device, dtype=torch.float32)
    if tuple(valid_mask.shape) != tuple(targets.shape):
        raise ValueError(
            f"valid_mask shape must match targets shape, got {tuple(valid_mask.shape)} vs {tuple(targets.shape)}"
        )
    return valid_mask


__all__ = ["resolve_valid_mask"]
