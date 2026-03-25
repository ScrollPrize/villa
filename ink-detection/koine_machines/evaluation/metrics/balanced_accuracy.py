from __future__ import annotations

from dataclasses import dataclass

import torch

from koine_machines.evaluation.metrics.base import BaseMetric
from koine_machines.evaluation.metrics.confusion import Confusion, _resolve_metric_name


@dataclass(frozen=True, kw_only=True)
class BalancedAccuracy(BaseMetric):
    threshold: float = 0.5

    def __post_init__(self) -> None:
        object.__setattr__(self, "threshold", float(self.threshold))
        BaseMetric.__post_init__(self)

    def default_name(self) -> str:
        return _resolve_metric_name(
            explicit_name=None,
            base_name="BalancedAccuracy",
            threshold=float(self.threshold),
        )

    @staticmethod
    def _from_counts(counts) -> torch.Tensor:
        pos_denom = counts.tp + counts.fn
        neg_denom = counts.tn + counts.fp
        pos_recall = torch.where(pos_denom > 0.0, counts.tp / pos_denom, torch.full_like(pos_denom, torch.nan))
        neg_recall = torch.where(neg_denom > 0.0, counts.tn / neg_denom, torch.full_like(neg_denom, torch.nan))
        recalls = torch.stack((pos_recall, neg_recall))
        valid = ~torch.isnan(recalls)
        if bool(valid.any()):
            return recalls[valid].mean()
        return torch.zeros((), dtype=recalls.dtype, device=recalls.device)

    def _confusion_metric(self) -> Confusion:
        return Confusion(threshold=float(self.threshold))

    def compute_batch(self, batch) -> float:
        counts = self._confusion_metric().compute_batch(batch)
        return float(self._from_counts(counts).item())

    def compute_per_sample(self, batch) -> float:
        values = [
            float(self._from_counts(counts).item())
            for counts in self._confusion_metric().compute_per_sample(batch)
        ]
        if not values:
            return 0.0
        return sum(values) / float(len(values))
