from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ink.recipes.losses.reporting import loss_values as resolve_loss_values
from ink.recipes.losses.reporting import resolve_train_output


@dataclass(frozen=True)
class LossTerm:
    loss: Any
    weight: float
    name: str


@dataclass(frozen=True)
class LossComposer:
    terms: tuple[LossTerm, ...]

    def __post_init__(self) -> None:
        assert self.terms
        names = [str(term.name) for term in self.terms]
        assert all(names)
        assert len(set(names)) == len(names)

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor, *, valid_mask=None) -> torch.Tensor:
        return self.training_outputs(logits, targets, valid_mask=valid_mask)["loss"]

    def training_outputs(self, logits: torch.Tensor, targets: torch.Tensor, *, valid_mask=None) -> dict[str, Any]:
        total_loss = None
        metrics: dict[str, torch.Tensor] = {}
        for term in self.terms:
            output = resolve_train_output(term.loss, logits, targets, valid_mask=valid_mask)
            weighted_loss = float(term.weight) * output.loss
            total_loss = weighted_loss if total_loss is None else total_loss + weighted_loss
            metrics[f"{term.name}_loss"] = output.loss
            for key, value in output.metrics.items():
                metrics[f"{term.name}/{key}"] = value
        assert total_loss is not None
        return {
            "loss": total_loss,
            "metrics": metrics,
        }

    def loss_values(self, logits: torch.Tensor, targets: torch.Tensor, *, valid_mask=None) -> torch.Tensor:
        total_values = None
        for term in self.terms:
            values = resolve_loss_values(term.loss, logits, targets, valid_mask=valid_mask)
            weighted_values = float(term.weight) * values
            total_values = weighted_values if total_values is None else total_values + weighted_values
        assert total_values is not None
        return total_values


__all__ = [
    "LossComposer",
    "LossTerm",
]
