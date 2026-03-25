from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch

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


def _masked_confusion_counts(batch, *, threshold: float) -> ConfusionCounts:
    # Expects a batch-like object with logits, require_targets(), and optional valid_mask.
    logits = batch.logits.detach()
    targets = batch.require_targets().detach()
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
