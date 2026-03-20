from __future__ import annotations

from dataclasses import dataclass

import torch

from ink.recipes.metrics.confusion import (
    _ConfusionMetricState,
    _masked_confusion_counts,
    _resolve_metric_name,
    add_confusion_counts,
)
from ink.recipes.metrics.reports import MetricReport


def balanced_accuracy_from_counts(counts) -> torch.Tensor:
    pos_denom = counts.tp + counts.fn
    neg_denom = counts.tn + counts.fp
    pos_recall = torch.where(pos_denom > 0.0, counts.tp / pos_denom, torch.full_like(pos_denom, torch.nan))
    neg_recall = torch.where(neg_denom > 0.0, counts.tn / neg_denom, torch.full_like(neg_denom, torch.nan))
    recalls = torch.stack((pos_recall, neg_recall))
    valid = ~torch.isnan(recalls)
    if bool(valid.any()):
        return recalls[valid].mean()
    return torch.zeros((), dtype=recalls.dtype, device=recalls.device)


@dataclass(frozen=True, kw_only=True)
class BalancedAccuracy:
    threshold: float = 0.5
    name: str | None = None

    def __post_init__(self) -> None:
        threshold = float(self.threshold)
        explicit_name = None if self.name is None else str(self.name).strip()
        resolved_name = _resolve_metric_name(
            explicit_name=explicit_name,
            base_name="BalancedAccuracy",
            threshold=threshold,
        )
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "name", resolved_name)

    def metric_name(self) -> str:
        return str(self.name)

    def empty_state(self, *, n_groups: int | None = None) -> _ConfusionMetricState:
        del n_groups
        return _ConfusionMetricState()

    def update(self, state: _ConfusionMetricState, batch, *, shared=None) -> _ConfusionMetricState:
        del shared
        return _ConfusionMetricState(
            counts=add_confusion_counts(
                state.counts,
                _masked_confusion_counts(batch, threshold=float(self.threshold)),
            )
        )

    def finalize(self, state: _ConfusionMetricState) -> MetricReport:
        return MetricReport(
            summary={str(self.metric_name()): float(balanced_accuracy_from_counts(state.counts).item())}
        )
