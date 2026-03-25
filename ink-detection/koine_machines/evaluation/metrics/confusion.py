from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from koine_machines.evaluation.metrics.base import BaseMetric


@dataclass(frozen=True)
class ConfusionCounts:
    tp: torch.Tensor
    fp: torch.Tensor
    fn: torch.Tensor
    tn: torch.Tensor


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
class Confusion(BaseMetric):
    threshold: float = 0.5

    def __post_init__(self) -> None:
        object.__setattr__(self, "threshold", float(self.threshold))
        BaseMetric.__post_init__(self)

    def default_name(self) -> str:
        return _resolve_metric_name(
            explicit_name=None,
            base_name="Confusion",
            threshold=float(self.threshold),
        )

    @staticmethod
    def zero_counts(*, device=None) -> ConfusionCounts:
        kwargs = {}
        if device is not None:
            kwargs["device"] = device
        return ConfusionCounts(
            tp=torch.zeros((), dtype=torch.float64, **kwargs),
            fp=torch.zeros((), dtype=torch.float64, **kwargs),
            fn=torch.zeros((), dtype=torch.float64, **kwargs),
            tn=torch.zeros((), dtype=torch.float64, **kwargs),
        )

    @staticmethod
    def add_counts(left: ConfusionCounts, right: ConfusionCounts) -> ConfusionCounts:
        return ConfusionCounts(
            tp=left.tp + right.tp,
            fp=left.fp + right.fp,
            fn=left.fn + right.fn,
            tn=left.tn + right.tn,
        )

    @staticmethod
    def _counts_from_predictions(preds: torch.Tensor, targets: torch.Tensor) -> ConfusionCounts:
        preds = preds.bool()
        targets = targets.bool()
        return ConfusionCounts(
            tp=(preds & targets).sum(dtype=torch.float64),
            fp=(preds & ~targets).sum(dtype=torch.float64),
            fn=(~preds & targets).sum(dtype=torch.float64),
            tn=(~preds & ~targets).sum(dtype=torch.float64),
        )

    def _counts_from_tensors(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
        device=None,
    ) -> ConfusionCounts:
        if tuple(logits.shape) != tuple(targets.shape):
            raise ValueError(f"logits/targets shape mismatch: {tuple(logits.shape)} vs {tuple(targets.shape)}")
        if valid_mask is not None:
            valid_mask = valid_mask.detach().bool()
            if tuple(valid_mask.shape) != tuple(targets.shape):
                raise ValueError(f"valid_mask shape mismatch: {tuple(valid_mask.shape)} vs {tuple(targets.shape)}")
            logits = logits[valid_mask]
            targets = targets[valid_mask]
        if int(targets.numel()) == 0:
            return self.zero_counts(device=device)
        preds = torch.sigmoid(logits).to(dtype=torch.float32) >= float(self.threshold)
        return self._counts_from_predictions(preds, targets.to(dtype=torch.float32) >= 0.5)

    def compute_batch(self, batch) -> ConfusionCounts:
        return self._counts_from_tensors(
            batch.logits.detach(),
            batch.require_targets().detach(),
            valid_mask=None if batch.valid_mask is None else batch.valid_mask.detach(),
            device=batch.logits.device,
        )

    def compute_per_sample(self, batch) -> list[ConfusionCounts]:
        logits = batch.logits.detach()
        targets = batch.require_targets().detach()
        valid_mask = None if batch.valid_mask is None else batch.valid_mask.detach()
        if logits.ndim < 3:
            return [
                self._counts_from_tensors(
                    logits,
                    targets,
                    valid_mask=valid_mask,
                    device=batch.logits.device,
                )
            ]
        return [
            self._counts_from_tensors(
                logits[index],
                targets[index],
                valid_mask=None if valid_mask is None else valid_mask[index],
                device=batch.logits.device,
            )
            for index in range(int(logits.shape[0]))
        ]
