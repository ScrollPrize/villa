from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch

from ink.recipes.metrics.batch import MetricBatch
from ink.recipes.metrics.reports import MetricReport


@dataclass(frozen=True)
class ConfusionCounts:
    tp: torch.Tensor
    fp: torch.Tensor
    fn: torch.Tensor
    tn: torch.Tensor


def zero_confusion_counts(*, device=None) -> ConfusionCounts:
    kwargs = {}
    if device is not None:
        kwargs["device"] = device
    return ConfusionCounts(
        tp=torch.zeros((), dtype=torch.float64, **kwargs),
        fp=torch.zeros((), dtype=torch.float64, **kwargs),
        fn=torch.zeros((), dtype=torch.float64, **kwargs),
        tn=torch.zeros((), dtype=torch.float64, **kwargs),
    )


def add_confusion_counts(left: ConfusionCounts, right: ConfusionCounts) -> ConfusionCounts:
    return ConfusionCounts(
        tp=left.tp + right.tp,
        fp=left.fp + right.fp,
        fn=left.fn + right.fn,
        tn=left.tn + right.tn,
    )


def confusion_counts(preds: torch.Tensor, targets: torch.Tensor) -> ConfusionCounts:
    preds = preds.bool()
    targets = targets.bool()
    return ConfusionCounts(
        tp=(preds & targets).sum(dtype=torch.float64),
        fp=(preds & ~targets).sum(dtype=torch.float64),
        fn=(~preds & targets).sum(dtype=torch.float64),
        tn=(~preds & ~targets).sum(dtype=torch.float64),
    )


def dice_from_counts(counts: ConfusionCounts) -> torch.Tensor:
    return (2.0 * counts.tp) / (2.0 * counts.tp + counts.fp + counts.fn + 1e-12)


def balanced_accuracy_from_counts(counts: ConfusionCounts) -> torch.Tensor:
    pos_denom = counts.tp + counts.fn
    neg_denom = counts.tn + counts.fp
    pos_recall = torch.where(pos_denom > 0.0, counts.tp / pos_denom, torch.full_like(pos_denom, torch.nan))
    neg_recall = torch.where(neg_denom > 0.0, counts.tn / neg_denom, torch.full_like(neg_denom, torch.nan))
    recalls = torch.stack((pos_recall, neg_recall))
    valid = ~torch.isnan(recalls)
    if bool(valid.any()):
        return recalls[valid].mean()
    return torch.zeros((), dtype=recalls.dtype, device=recalls.device)


def _masked_confusion_counts(batch: MetricBatch, *, threshold: float) -> ConfusionCounts:
    logits = batch.logits.detach()
    targets = batch.targets.detach()
    if tuple(logits.shape) != tuple(targets.shape):
        raise ValueError(f"logits/targets shape mismatch: {tuple(logits.shape)} vs {tuple(targets.shape)}")
    if batch.valid_mask is not None:
        valid_mask = batch.valid_mask.detach().bool()
        if tuple(valid_mask.shape) != tuple(targets.shape):
            raise ValueError(f"valid_mask shape mismatch: {tuple(valid_mask.shape)} vs {tuple(targets.shape)}")
        logits = logits[valid_mask]
        targets = targets[valid_mask]
    if int(targets.numel()) == 0:
        return zero_confusion_counts(device=batch.logits.device)
    preds = torch.sigmoid(logits).to(dtype=torch.float32) >= float(threshold)
    return confusion_counts(preds, targets.to(dtype=torch.float32) >= 0.5)


@dataclass(frozen=True)
class _ConfusionMetricState:
    counts: ConfusionCounts = field(default_factory=zero_confusion_counts)


@dataclass(frozen=True, kw_only=True)
class ConfusionMetric:
    name: str
    threshold: float = 0.5
    score_fn: object = dice_from_counts

    def build(self, *, data, runtime=None, stitch=None, logger=None, patch_loss=None) -> ConfusionMetric:
        del data, runtime, stitch, logger, patch_loss
        return self

    def empty_state(self, *, n_groups: int | None = None) -> _ConfusionMetricState:
        del n_groups
        return _ConfusionMetricState()

    def update(self, state: _ConfusionMetricState, batch: MetricBatch, *, shared=None) -> _ConfusionMetricState:
        del shared
        return _ConfusionMetricState(
            counts=add_confusion_counts(
                state.counts,
                _masked_confusion_counts(batch, threshold=float(self.threshold)),
            )
        )

    def finalize(self, state: _ConfusionMetricState) -> MetricReport:
        raw_value = self.score_fn(state.counts)
        value = float(raw_value.item()) if hasattr(raw_value, "item") else float(raw_value)
        return MetricReport(summary={str(self.name): value})


def _threshold_name_suffix(threshold: float) -> str:
    threshold_value = float(threshold)
    scaled_255 = threshold_value * 255.0
    rounded_255 = round(scaled_255)
    if math.isclose(scaled_255, float(rounded_255), rel_tol=0.0, abs_tol=1e-9):
        return f"thr_{int(rounded_255)}_255"
    text = f"{threshold_value:.6g}".replace("-", "neg_").replace(".", "_")
    return f"thr_{text}"


def _resolve_metric_name(*, explicit_name: str | None, base_name: str, threshold: float) -> str:
    if explicit_name is not None and str(explicit_name).strip():
        return str(explicit_name)
    if math.isclose(float(threshold), 0.5, rel_tol=0.0, abs_tol=1e-9):
        return str(base_name)
    return f"{base_name}_{_threshold_name_suffix(float(threshold))}"


@dataclass(frozen=True, kw_only=True)
class Dice:
    threshold: float = 0.5
    name: str | None = None

    def metric_name(self) -> str:
        return _resolve_metric_name(explicit_name=self.name, base_name="Dice", threshold=float(self.threshold))

    def build(self, *, data, runtime=None, stitch=None, logger=None, patch_loss=None) -> Dice:
        del data, runtime, stitch, logger, patch_loss
        return Dice(
            threshold=float(self.threshold),
            name=self.metric_name(),
        )

    def empty_state(self, *, n_groups: int | None = None) -> _ConfusionMetricState:
        del n_groups
        return _ConfusionMetricState()

    def update(self, state: _ConfusionMetricState, batch: MetricBatch, *, shared=None) -> _ConfusionMetricState:
        del shared
        return _ConfusionMetricState(
            counts=add_confusion_counts(
                state.counts,
                _masked_confusion_counts(batch, threshold=float(self.threshold)),
            )
        )

    def finalize(self, state: _ConfusionMetricState) -> MetricReport:
        return MetricReport(summary={str(self.metric_name()): float(dice_from_counts(state.counts).item())})


@dataclass(frozen=True, kw_only=True)
class BalancedAccuracy:
    threshold: float = 0.5
    name: str | None = None

    def metric_name(self) -> str:
        return _resolve_metric_name(
            explicit_name=self.name,
            base_name="BalancedAccuracy",
            threshold=float(self.threshold),
        )

    def build(self, *, data, runtime=None, stitch=None, logger=None, patch_loss=None) -> BalancedAccuracy:
        del data, runtime, stitch, logger, patch_loss
        return BalancedAccuracy(
            threshold=float(self.threshold),
            name=self.metric_name(),
        )

    def empty_state(self, *, n_groups: int | None = None) -> _ConfusionMetricState:
        del n_groups
        return _ConfusionMetricState()

    def update(self, state: _ConfusionMetricState, batch: MetricBatch, *, shared=None) -> _ConfusionMetricState:
        del shared
        return _ConfusionMetricState(
            counts=add_confusion_counts(
                state.counts,
                _masked_confusion_counts(batch, threshold=float(self.threshold)),
            )
        )

    def finalize(self, state: _ConfusionMetricState) -> MetricReport:
        return MetricReport(summary={str(self.metric_name()): float(balanced_accuracy_from_counts(state.counts).item())})
