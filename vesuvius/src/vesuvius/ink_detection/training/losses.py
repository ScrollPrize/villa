"""Ink segmentation objectives and scalar weighting."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from vesuvius.ink_detection.config import InkConfig
from vesuvius.models.training.loss.nnunet_losses import DC_and_BCE_loss


class LabelSmoothedDCAndBCELoss(DC_and_BCE_loss):
    """Combine Dice and BCE with optional smoothing of BCE targets."""

    def __init__(self, *args, bce_label_smoothing: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0.0 <= bce_label_smoothing <= 1.0:
            raise ValueError(
                f"bce_label_smoothing must be between 0 and 1, got {bce_label_smoothing}"
            )
        self.bce_label_smoothing = float(bce_label_smoothing)

    def _smooth_bce_targets(self, target_regions: torch.Tensor) -> torch.Tensor:
        target_regions = target_regions.float()
        if self.bce_label_smoothing == 0.0:
            return target_regions
        return target_regions * (1.0 - self.bce_label_smoothing) + 0.5 * self.bce_label_smoothing

    def forward(
        self,
        net_output: torch.Tensor,
        target: torch.Tensor,
        loss_mask: torch.Tensor = None,
    ):
        if self.use_ignore_label:
            if target.dtype == torch.bool:
                mask = ~target[:, -1:]
            else:
                mask = (1 - target[:, -1:]).bool()
            target_regions = target[:, :-1]
        else:
            target_regions = target
            mask = loss_mask

        dc_loss = self.dc(net_output, target_regions, loss_mask=mask)
        smoothed_targets = self._smooth_bce_targets(target_regions)
        if mask is not None:
            denom = torch.clip(mask.sum(), min=1e-8)
            ce_loss = (self.ce(net_output, smoothed_targets) * mask).sum() / denom
        else:
            ce_loss = self.ce(net_output, smoothed_targets)
        return self.weight_ce * ce_loss + self.weight_dice * dc_loss


class WeightedLossTerm(nn.Module):
    """A named scalar-weighted objective and its metric label."""

    def __init__(
        self,
        name: str,
        weight: float,
        module: nn.Module,
        metric_name: str | None = None,
    ) -> None:
        super().__init__()
        self.name = str(name)
        self.weight = float(weight)
        self.module = module
        self.metric_name = str(metric_name or name)


def _sanitize_metric_name(name: str) -> str:
    cleaned = "".join(character.lower() if character.isalnum() else "_" for character in str(name))
    cleaned = cleaned.strip("_")
    return cleaned or "term"


def _split_loss_output(
    output: Any,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if isinstance(output, tuple):
        if len(output) != 2:
            raise ValueError(
                f"Expected (loss, aux_dict) tuple, got tuple of length {len(output)}"
            )
        loss_value, auxiliary = output
        if not isinstance(auxiliary, dict):
            raise TypeError(
                f"Expected aux metrics dict, got {type(auxiliary).__name__}"
            )
        return loss_value.reshape(()), auxiliary
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Expected loss tensor, got {type(output).__name__}")
    return output.reshape(()), {}


class CompositeLoss(nn.Module):
    """Sum weighted terms and expose their detached scalar metrics."""

    def __init__(self, terms: list[WeightedLossTerm]) -> None:
        super().__init__()
        if not terms:
            raise ValueError("CompositeLoss requires at least one term")
        self.terms = nn.ModuleList(terms)
        self.latest_metrics: dict[str, float] = {}

    def forward(
        self,
        net_output: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        total = net_output.new_zeros(())
        metrics: dict[str, float] = {}
        used_metric_names: set[str] = set()

        for index, term in enumerate(self.terms):
            raw_output = term.module(net_output, target)
            raw_loss, auxiliary = _split_loss_output(raw_output)
            weighted_loss = raw_loss * term.weight
            total = total + weighted_loss

            metric_name = _sanitize_metric_name(term.metric_name)
            if metric_name in used_metric_names:
                metric_name = f"{metric_name}_{index}"
            used_metric_names.add(metric_name)
            metrics[f"loss_terms/{metric_name}_raw"] = float(
                raw_loss.detach().item()
            )
            metrics[f"loss_terms/{metric_name}_weighted"] = float(
                weighted_loss.detach().item()
            )
            for key, value in auxiliary.items():
                auxiliary_key = _sanitize_metric_name(key)
                metrics[f"loss_aux/{metric_name}/{auxiliary_key}"] = float(
                    value.detach().item()
                )

        metrics["loss/total"] = float(total.detach().item())
        self.latest_metrics = metrics
        return total


def create_loss(config: InkConfig) -> CompositeLoss:
    """Build the ordered segmentation objective selected by InkConfig."""

    terms = []
    for term in config.loss.terms:
        module = LabelSmoothedDCAndBCELoss(
            bce_kwargs=dict(term.bce_kwargs),
            soft_dice_kwargs={
                "label_smoothing": term.dice_label_smoothing,
            },
            weight_dice=term.weight_dice,
            weight_ce=term.weight_ce,
            use_ignore_label=True,
            bce_label_smoothing=term.bce_label_smoothing,
        )
        terms.append(
            WeightedLossTerm(
                name=term.name,
                weight=term.weight,
                module=module,
                metric_name=term.metric_name,
            )
        )
    return CompositeLoss(terms)
