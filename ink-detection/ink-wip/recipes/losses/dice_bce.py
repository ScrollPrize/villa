from __future__ import annotations

from dataclasses import dataclass

import torch

from ink.recipes.losses.bce import compute_batch_bce_loss, compute_per_sample_bce_values
from ink.recipes.losses.dice import compute_batch_dice_loss, compute_per_sample_dice_values


def compute_dice_bce_batch_loss(recipe, logits: torch.Tensor, targets: torch.Tensor, *, valid_mask=None) -> torch.Tensor:
    batch_dice_loss = compute_batch_dice_loss(
        logits,
        targets,
        valid_mask=valid_mask,
        eps=recipe.eps,
    )
    batch_bce = compute_batch_bce_loss(
        logits,
        targets,
        valid_mask=valid_mask,
        smooth_factor=recipe.smooth_factor,
        soft_label_positive=recipe.soft_label_positive,
        soft_label_negative=recipe.soft_label_negative,
    )
    return 0.5 * batch_dice_loss + 0.5 * batch_bce


def compute_dice_bce_per_sample_loss(recipe, logits: torch.Tensor, targets: torch.Tensor, *, valid_mask=None) -> torch.Tensor:
    per_sample_dice = compute_per_sample_dice_values(
        logits,
        targets,
        valid_mask=valid_mask,
        eps=recipe.eps,
    )
    per_sample_bce = compute_per_sample_bce_values(
        logits,
        targets,
        valid_mask=valid_mask,
        smooth_factor=recipe.smooth_factor,
        soft_label_positive=recipe.soft_label_positive,
        soft_label_negative=recipe.soft_label_negative,
    )
    return 0.5 * (1.0 - per_sample_dice) + 0.5 * per_sample_bce


def _per_sample_terms(recipe, logits: torch.Tensor, targets: torch.Tensor, *, valid_mask=None) -> tuple[torch.Tensor, torch.Tensor]:
    per_sample_dice = compute_per_sample_dice_values(
        logits,
        targets,
        valid_mask=valid_mask,
        eps=recipe.eps,
    )
    per_sample_bce = compute_per_sample_bce_values(
        logits,
        targets,
        valid_mask=valid_mask,
        smooth_factor=recipe.smooth_factor,
        soft_label_positive=recipe.soft_label_positive,
        soft_label_negative=recipe.soft_label_negative,
    )
    return per_sample_dice, per_sample_bce


@dataclass(frozen=True)
class DiceBCEBatch:
    smooth_factor: float = 0.25
    soft_label_positive: float = 1.0
    soft_label_negative: float = 0.0
    eps: float = 1e-7

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor, *, valid_mask=None) -> torch.Tensor:
        return compute_dice_bce_batch_loss(self, logits, targets, valid_mask=valid_mask)

    def training_outputs(self, logits: torch.Tensor, targets: torch.Tensor, *, valid_mask=None) -> dict[str, torch.Tensor]:
        batch_dice_loss = compute_batch_dice_loss(
            logits,
            targets,
            valid_mask=valid_mask,
            eps=self.eps,
        )
        batch_bce = compute_batch_bce_loss(
            logits,
            targets,
            valid_mask=valid_mask,
            smooth_factor=self.smooth_factor,
            soft_label_positive=self.soft_label_positive,
            soft_label_negative=self.soft_label_negative,
        )
        return {
            "loss": 0.5 * batch_dice_loss + 0.5 * batch_bce,
            "metrics": {
                "dice_loss": batch_dice_loss,
                "bce_loss": batch_bce,
            },
        }

    def loss_values(self, logits: torch.Tensor, targets: torch.Tensor, *, valid_mask=None) -> torch.Tensor:
        per_sample_dice, per_sample_bce = _per_sample_terms(self, logits, targets, valid_mask=valid_mask)
        return 0.5 * (1.0 - per_sample_dice) + 0.5 * per_sample_bce

@dataclass(frozen=True)
class DiceBCEPerSample:
    smooth_factor: float = 0.25
    soft_label_positive: float = 1.0
    soft_label_negative: float = 0.0
    eps: float = 1e-7

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor, *, valid_mask=None) -> torch.Tensor:
        return compute_dice_bce_per_sample_loss(self, logits, targets, valid_mask=valid_mask)

    def training_outputs(self, logits: torch.Tensor, targets: torch.Tensor, *, valid_mask=None) -> dict[str, torch.Tensor]:
        per_sample_dice, per_sample_bce = _per_sample_terms(self, logits, targets, valid_mask=valid_mask)
        per_sample_dice_loss = 1.0 - per_sample_dice
        return {
            "loss": 0.5 * per_sample_dice_loss + 0.5 * per_sample_bce,
            "metrics": {
                "dice_loss": per_sample_dice_loss,
                "bce_loss": per_sample_bce,
            },
        }

    def loss_values(self, logits: torch.Tensor, targets: torch.Tensor, *, valid_mask=None) -> torch.Tensor:
        return compute_dice_bce_per_sample_loss(self, logits, targets, valid_mask=valid_mask)
