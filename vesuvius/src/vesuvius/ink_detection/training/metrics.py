"""Binary confusion accumulation and balanced accuracy."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from vesuvius.ink_detection.types import ConfusionCounts, MetricBatch


def _threshold_suffix(threshold: float) -> str:
    scaled = float(threshold) * 255.0
    rounded = round(scaled)
    if math.isclose(scaled, float(rounded), rel_tol=0.0, abs_tol=1e-9):
        return f"thr_{int(rounded)}_255"
    text = f"{float(threshold):.6g}".replace("-", "neg_").replace(".", "_")
    return f"thr_{text}"


def _metric_name(base: str, threshold: float, explicit: str | None) -> str:
    if explicit is not None and str(explicit).strip():
        return str(explicit)
    if math.isclose(float(threshold), 0.5, rel_tol=0.0, abs_tol=1e-9):
        return base
    return f"{base}_{_threshold_suffix(threshold)}"


@dataclass(frozen=True, kw_only=True)
class Confusion:
    threshold: float = 0.5
    name: str | None = None
    per_sample: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "threshold", float(self.threshold))
        object.__setattr__(
            self, "name", _metric_name("Confusion", self.threshold, self.name)
        )
        object.__setattr__(self, "per_sample", bool(self.per_sample))

    def metric_name(self) -> str:
        return str(self.name)

    @staticmethod
    def zero_counts(*, device=None) -> ConfusionCounts:
        kwargs = {} if device is None else {"device": device}
        return ConfusionCounts(
            tp=torch.zeros((), dtype=torch.float64, **kwargs),
            fp=torch.zeros((), dtype=torch.float64, **kwargs),
            fn=torch.zeros((), dtype=torch.float64, **kwargs),
            tn=torch.zeros((), dtype=torch.float64, **kwargs),
        )

    @staticmethod
    def add_counts(
        left: ConfusionCounts, right: ConfusionCounts
    ) -> ConfusionCounts:
        return ConfusionCounts(
            tp=left.tp + right.tp,
            fp=left.fp + right.fp,
            fn=left.fn + right.fn,
            tn=left.tn + right.tn,
        )

    def _counts(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> ConfusionCounts:
        if logits.shape != targets.shape:
            raise ValueError(
                f"logits/targets shape mismatch: {tuple(logits.shape)} vs {tuple(targets.shape)}"
            )
        if valid_mask is not None:
            valid_mask = valid_mask.detach().bool()
            if valid_mask.shape != targets.shape:
                raise ValueError(
                    "valid_mask shape mismatch: "
                    f"{tuple(valid_mask.shape)} vs {tuple(targets.shape)}"
                )
            logits = logits[valid_mask]
            targets = targets[valid_mask]
        if targets.numel() == 0:
            return self.zero_counts(device=logits.device)
        predictions = torch.sigmoid(logits).to(torch.float32) >= self.threshold
        targets = targets.to(torch.float32) >= 0.5
        return ConfusionCounts(
            tp=(predictions & targets).sum(dtype=torch.float64),
            fp=(predictions & ~targets).sum(dtype=torch.float64),
            fn=(~predictions & targets).sum(dtype=torch.float64),
            tn=(~predictions & ~targets).sum(dtype=torch.float64),
        )

    def compute_batch(self, batch: MetricBatch) -> ConfusionCounts:
        return self._counts(
            batch.logits.detach(),
            batch.require_targets().detach(),
            None if batch.valid_mask is None else batch.valid_mask.detach(),
        )

    def compute_per_sample(self, batch: MetricBatch) -> list[ConfusionCounts]:
        logits = batch.logits.detach()
        targets = batch.require_targets().detach()
        valid = None if batch.valid_mask is None else batch.valid_mask.detach()
        if logits.ndim < 3:
            return [self._counts(logits, targets, valid)]
        return [
            self._counts(
                logits[index],
                targets[index],
                None if valid is None else valid[index],
            )
            for index in range(logits.shape[0])
        ]

    def compute(self, batch: MetricBatch):
        return self.compute_per_sample(batch) if self.per_sample else self.compute_batch(batch)


@dataclass(frozen=True, kw_only=True)
class BalancedAccuracy:
    threshold: float = 0.5
    name: str | None = None
    per_sample: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "threshold", float(self.threshold))
        object.__setattr__(
            self,
            "name",
            _metric_name("BalancedAccuracy", self.threshold, self.name),
        )
        object.__setattr__(self, "per_sample", bool(self.per_sample))

    def metric_name(self) -> str:
        return str(self.name)

    @staticmethod
    def _from_counts(counts: ConfusionCounts) -> torch.Tensor:
        positive_denominator = counts.tp + counts.fn
        negative_denominator = counts.tn + counts.fp
        positive_recall = torch.where(
            positive_denominator > 0,
            counts.tp / positive_denominator,
            torch.full_like(positive_denominator, torch.nan),
        )
        negative_recall = torch.where(
            negative_denominator > 0,
            counts.tn / negative_denominator,
            torch.full_like(negative_denominator, torch.nan),
        )
        recalls = torch.stack((positive_recall, negative_recall))
        valid = ~torch.isnan(recalls)
        if bool(valid.any()):
            return recalls[valid].mean()
        return torch.zeros((), dtype=recalls.dtype, device=recalls.device)

    def compute_batch(self, batch: MetricBatch) -> float:
        counts = Confusion(threshold=self.threshold).compute_batch(batch)
        return float(self._from_counts(counts).item())

    def compute_per_sample(self, batch: MetricBatch) -> float:
        values = [
            float(self._from_counts(counts).item())
            for counts in Confusion(threshold=self.threshold).compute_per_sample(batch)
        ]
        return 0.0 if not values else sum(values) / float(len(values))

    def compute(self, batch: MetricBatch) -> float:
        return self.compute_per_sample(batch) if self.per_sample else self.compute_batch(batch)
