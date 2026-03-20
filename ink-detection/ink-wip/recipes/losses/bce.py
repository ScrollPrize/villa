from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def build_bce_targets(
    targets,
    *,
    smooth_factor=0.25,
    soft_label_positive=1.0,
    soft_label_negative=0.0,
):
    targets = targets.float()
    soft_label_positive = float(soft_label_positive)
    soft_label_negative = float(soft_label_negative)
    soft_targets = targets * soft_label_positive + (1.0 - targets) * soft_label_negative
    smooth_factor = float(smooth_factor)
    if smooth_factor != 0.0:
        soft_targets = (1.0 - soft_targets) * smooth_factor + soft_targets * (1.0 - smooth_factor)
    return soft_targets


def resolve_valid_mask(targets: torch.Tensor, valid_mask) -> torch.Tensor:
    if valid_mask is None:
        return torch.ones_like(targets, dtype=torch.float32)

    valid_mask = valid_mask.to(device=targets.device, dtype=torch.float32)
    if tuple(valid_mask.shape) != tuple(targets.shape):
        raise ValueError(
            f"valid_mask shape must match targets shape, got {tuple(valid_mask.shape)} vs {tuple(targets.shape)}"
        )
    return valid_mask


def compute_per_sample_bce_values(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    valid_mask=None,
    smooth_factor: float = 0.25,
    soft_label_positive: float = 1.0,
    soft_label_negative: float = 0.0,
) -> torch.Tensor:
    targets = targets.float()
    valid_mask = resolve_valid_mask(targets, valid_mask)
    bce_targets = build_bce_targets(
        targets,
        smooth_factor=smooth_factor,
        soft_label_positive=soft_label_positive,
        soft_label_negative=soft_label_negative,
    )
    bce = F.binary_cross_entropy_with_logits(logits, bce_targets, reduction="none")

    batch_size = int(logits.shape[0])
    per_sample_valid = valid_mask.reshape(batch_size, -1).sum(dim=1).clamp_min(1.0)
    return (bce * valid_mask).reshape(batch_size, -1).sum(dim=1) / per_sample_valid


def compute_batch_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    valid_mask=None,
    smooth_factor: float = 0.25,
    soft_label_positive: float = 1.0,
    soft_label_negative: float = 0.0,
) -> torch.Tensor:
    targets = targets.float()
    valid_mask = resolve_valid_mask(targets, valid_mask)
    bce_targets = build_bce_targets(
        targets,
        smooth_factor=smooth_factor,
        soft_label_positive=soft_label_positive,
        soft_label_negative=soft_label_negative,
    )
    bce = F.binary_cross_entropy_with_logits(logits, bce_targets, reduction="none")
    batch_valid = valid_mask.sum().clamp_min(1.0)
    return (bce * valid_mask).sum() / batch_valid

@dataclass(frozen=True)
class BCEBatch:
    smooth_factor: float = 0.25
    soft_label_positive: float = 1.0
    soft_label_negative: float = 0.0
    eps: float = 1e-7

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor, *, valid_mask=None) -> torch.Tensor:
        return compute_batch_bce_loss(
            logits,
            targets,
            valid_mask=valid_mask,
            smooth_factor=self.smooth_factor,
            soft_label_positive=self.soft_label_positive,
            soft_label_negative=self.soft_label_negative,
        )

    def loss_values(self, logits: torch.Tensor, targets: torch.Tensor, *, valid_mask=None) -> torch.Tensor:
        return compute_per_sample_bce_values(
            logits,
            targets,
            valid_mask=valid_mask,
            smooth_factor=self.smooth_factor,
            soft_label_positive=self.soft_label_positive,
            soft_label_negative=self.soft_label_negative,
        )


@dataclass(frozen=True)
class BCEPerSample:
    smooth_factor: float = 0.25
    soft_label_positive: float = 1.0
    soft_label_negative: float = 0.0
    eps: float = 1e-7

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor, *, valid_mask=None) -> torch.Tensor:
        return compute_per_sample_bce_values(
            logits,
            targets,
            valid_mask=valid_mask,
            smooth_factor=self.smooth_factor,
            soft_label_positive=self.soft_label_positive,
            soft_label_negative=self.soft_label_negative,
        )

    def loss_values(self, logits: torch.Tensor, targets: torch.Tensor, *, valid_mask=None) -> torch.Tensor:
        return self(logits, targets, valid_mask=valid_mask)
