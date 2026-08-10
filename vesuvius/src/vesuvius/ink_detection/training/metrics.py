"""Binary confusion accumulation and balanced accuracy."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vesuvius.ink_detection.types import ConfusionCounts, MetricBatch


@dataclass(frozen=True, kw_only=True)
class Confusion:
    threshold: float = 0.5

    def __post_init__(self) -> None:
        object.__setattr__(self, "threshold", float(self.threshold))

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

    def compute_batch(self, batch: MetricBatch) -> ConfusionCounts:
        logits = batch.logits.detach()
        targets = batch.require_targets().detach()
        valid_mask = None if batch.valid_mask is None else batch.valid_mask.detach()
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


class BalancedAccuracy:
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
