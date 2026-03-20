from __future__ import annotations

from dataclasses import dataclass

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


def compute_per_sample_dice_values(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    valid_mask=None,
    eps: float = 1e-7,
) -> torch.Tensor:
    targets = targets.float()
    valid_mask = resolve_valid_mask(targets, valid_mask)
    probs = torch.sigmoid(logits)
    batch_size = int(logits.shape[0])
    intersection = (probs * targets * valid_mask).reshape(batch_size, -1).sum(dim=1)
    union = (probs * valid_mask).reshape(batch_size, -1).sum(dim=1) + (targets * valid_mask).reshape(
        batch_size, -1
    ).sum(dim=1)
    return (2.0 * intersection + float(eps)) / (union + float(eps))


def compute_per_sample_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    valid_mask=None,
    eps: float = 1e-7,
) -> torch.Tensor:
    return 1.0 - compute_per_sample_dice_values(
        logits,
        targets,
        valid_mask=valid_mask,
        eps=eps,
    )


def compute_batch_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    valid_mask=None,
    eps: float = 1e-7,
) -> torch.Tensor:
    targets = targets.float()
    valid_mask = resolve_valid_mask(targets, valid_mask)
    probs = torch.sigmoid(logits)
    batch_intersection = (probs * targets * valid_mask).sum()
    batch_union = (probs * valid_mask).sum() + (targets * valid_mask).sum()
    batch_dice = (2.0 * batch_intersection + float(eps)) / (batch_union + float(eps))
    return 1.0 - batch_dice


@dataclass(frozen=True)
class DiceBatch:
    smooth_factor: float = 0.25
    soft_label_positive: float = 1.0
    soft_label_negative: float = 0.0
    eps: float = 1e-7

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor, *, valid_mask=None) -> torch.Tensor:
        return compute_batch_dice_loss(
            logits,
            targets,
            valid_mask=valid_mask,
            eps=self.eps,
        )

    def loss_values(self, logits: torch.Tensor, targets: torch.Tensor, *, valid_mask=None) -> torch.Tensor:
        return compute_per_sample_dice_loss(
            logits,
            targets,
            valid_mask=valid_mask,
            eps=self.eps,
        )


@dataclass(frozen=True)
class DicePerSample:
    smooth_factor: float = 0.25
    soft_label_positive: float = 1.0
    soft_label_negative: float = 0.0
    eps: float = 1e-7

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor, *, valid_mask=None) -> torch.Tensor:
        return compute_per_sample_dice_loss(
            logits,
            targets,
            valid_mask=valid_mask,
            eps=self.eps,
        )

    def loss_values(self, logits: torch.Tensor, targets: torch.Tensor, *, valid_mask=None) -> torch.Tensor:
        return self(logits, targets, valid_mask=valid_mask)
