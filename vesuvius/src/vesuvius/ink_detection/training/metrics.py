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
        # Exact integer counts: int64 exists on every device, float64 does
        # not exist on MPS, and float32 stops being exact above 2**24 pixels.
        kwargs = {} if device is None else {"device": device}
        return ConfusionCounts(
            tp=torch.zeros((), dtype=torch.int64, **kwargs),
            fp=torch.zeros((), dtype=torch.int64, **kwargs),
            fn=torch.zeros((), dtype=torch.int64, **kwargs),
            tn=torch.zeros((), dtype=torch.int64, **kwargs),
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
            tp=(predictions & targets).sum(dtype=torch.int64),
            fp=(predictions & ~targets).sum(dtype=torch.int64),
            fn=(~predictions & targets).sum(dtype=torch.int64),
            tn=(~predictions & ~targets).sum(dtype=torch.int64),
        )


class BalancedAccuracy:
    @staticmethod
    def from_counts(counts: ConfusionCounts) -> torch.Tensor:
        # The ratio stays float64, computed on the CPU for devices without it.
        # Move first, then cast: torch 2.14 turns a fused MPS->CPU float64
        # .to(device="cpu", dtype=torch.float64) into zeros without an error.
        tp, fp, fn, tn = (
            value.detach().cpu().to(torch.float64)
            for value in (counts.tp, counts.fp, counts.fn, counts.tn)
        )
        positive_denominator = tp + fn
        negative_denominator = tn + fp
        positive_recall = torch.where(
            positive_denominator > 0,
            tp / positive_denominator,
            torch.full_like(positive_denominator, torch.nan),
        )
        negative_recall = torch.where(
            negative_denominator > 0,
            tn / negative_denominator,
            torch.full_like(negative_denominator, torch.nan),
        )
        recalls = torch.stack((positive_recall, negative_recall))
        valid = ~torch.isnan(recalls)
        if bool(valid.any()):
            return recalls[valid].mean()
        return torch.zeros((), dtype=recalls.dtype, device=recalls.device)
