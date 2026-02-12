from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch


def _safe_div(numer: torch.Tensor, denom: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return numer / (denom + eps)


@dataclass
class ConfusionCounts:
    tp: torch.Tensor
    fp: torch.Tensor
    tn: torch.Tensor
    fn: torch.Tensor


def confusion_counts(preds: torch.Tensor, targets: torch.Tensor) -> ConfusionCounts:
    """Compute TP/FP/TN/FN for boolean tensors (any shape).

    Returns float64 tensors for numeric stability.
    """
    preds = preds.bool()
    targets = targets.bool()
    tp = (preds & targets).sum(dtype=torch.float64)
    fp = (preds & ~targets).sum(dtype=torch.float64)
    tn = (~preds & ~targets).sum(dtype=torch.float64)
    fn = (~preds & targets).sum(dtype=torch.float64)
    return ConfusionCounts(tp=tp, fp=fp, tn=tn, fn=fn)


def recall_from_counts(c: ConfusionCounts) -> torch.Tensor:
    return _safe_div(c.tp, c.tp + c.fn)


def specificity_from_counts(c: ConfusionCounts) -> torch.Tensor:
    return _safe_div(c.tn, c.tn + c.fp)


def dice_from_counts(c: ConfusionCounts) -> torch.Tensor:
    # Dice/F1: 2TP / (2TP + FP + FN)
    return _safe_div(2.0 * c.tp, 2.0 * c.tp + c.fp + c.fn)


def fbeta_from_counts(c: ConfusionCounts, *, beta: float) -> torch.Tensor:
    # F_beta: (1+beta^2)TP / ((1+beta^2)TP + beta^2 FN + FP)
    beta2 = float(beta) ** 2
    numer = (1.0 + beta2) * c.tp
    denom = (1.0 + beta2) * c.tp + beta2 * c.fn + c.fp
    return _safe_div(numer, denom)


def mcc_from_counts(c: ConfusionCounts) -> torch.Tensor:
    # Matthews correlation coefficient.
    # (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
    numer = c.tp * c.tn - c.fp * c.fn
    denom = (c.tp + c.fp) * (c.tp + c.fn) * (c.tn + c.fp) * (c.tn + c.fn)
    denom = torch.sqrt(torch.clamp(denom, min=0.0)) + 1e-12
    return numer / denom


class StreamingBinarySegmentationMetrics:
    """Streaming (epoch-accumulated) pixel-level metrics for binary segmentation.

    Designed for validation-time evaluation where collecting all pixels in-memory
    may be too expensive.

    Notes:
    - Confusion-based metrics are computed at a fixed threshold.
    - best-F_beta is approximated via a fixed-width probability histogram.
    """

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        beta: float = 0.5,
        num_bins: int = 1000,
        device: Optional[torch.device] = None,
    ) -> None:
        if not (0.0 <= float(threshold) <= 1.0):
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        if float(beta) <= 0.0:
            raise ValueError(f"beta must be > 0, got {beta}")
        num_bins = int(num_bins)
        if num_bins < 2:
            raise ValueError(f"num_bins must be >= 2, got {num_bins}")

        self.threshold = float(threshold)
        self.beta = float(beta)
        self.num_bins = num_bins
        self._device = device

        self.reset(device=device)

    @property
    def device(self) -> torch.device:
        return self._device if self._device is not None else torch.device("cpu")

    def reset(self, *, device: Optional[torch.device] = None) -> None:
        if device is not None:
            self._device = device
        dev = self.device

        self._counts = ConfusionCounts(
            tp=torch.zeros((), device=dev, dtype=torch.float64),
            fp=torch.zeros((), device=dev, dtype=torch.float64),
            tn=torch.zeros((), device=dev, dtype=torch.float64),
            fn=torch.zeros((), device=dev, dtype=torch.float64),
        )

        self._pos_hist = torch.zeros(self.num_bins, device=dev, dtype=torch.float64)
        self._neg_hist = torch.zeros(self.num_bins, device=dev, dtype=torch.float64)

    def update(
        self,
        *,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Accumulate stats for a batch.

        Args:
            logits: model outputs (any shape).
            targets: binary targets in {0,1} (same shape as logits).
            mask: optional boolean mask selecting which pixels to include.
        """
        if logits.shape != targets.shape:
            raise ValueError(f"logits/targets shape mismatch: {tuple(logits.shape)} vs {tuple(targets.shape)}")

        logits = logits.detach()
        targets = targets.detach()

        if mask is not None:
            mask = mask.detach().bool()
            if mask.shape != targets.shape:
                raise ValueError(f"mask/targets shape mismatch: {tuple(mask.shape)} vs {tuple(targets.shape)}")
            logits = logits[mask]
            targets = targets[mask]

        if targets.numel() == 0:
            return

        targets_f = targets.to(dtype=torch.float32)
        targets_bool = targets_f >= 0.5

        # Thresholded confusion counts.
        probs = torch.sigmoid(logits).to(dtype=torch.float32)
        preds = probs >= float(self.threshold)
        batch_counts = confusion_counts(preds, targets_bool)
        self._counts.tp += batch_counts.tp
        self._counts.fp += batch_counts.fp
        self._counts.tn += batch_counts.tn
        self._counts.fn += batch_counts.fn

        # Histogram for approximate best-F_beta threshold sweep.
        # Bin index in [0, num_bins-1] for probs in [0, 1].
        idx = torch.clamp((probs * float(self.num_bins)).to(dtype=torch.int64), min=0, max=self.num_bins - 1)

        pos_idx = idx[targets_bool]
        neg_idx = idx[~targets_bool]
        if pos_idx.numel():
            self._pos_hist += torch.bincount(pos_idx, minlength=self.num_bins).to(dtype=torch.float64)
        if neg_idx.numel():
            self._neg_hist += torch.bincount(neg_idx, minlength=self.num_bins).to(dtype=torch.float64)

    def _curves(self) -> Dict[str, torch.Tensor]:
        total_pos = self._pos_hist.sum()

        # Curves are computed for thresholds descending from ~1 -> 0.
        tp = torch.cumsum(self._pos_hist.flip(0), dim=0)
        fp = torch.cumsum(self._neg_hist.flip(0), dim=0)

        # Handle degenerate cases.
        if float(total_pos.item()) <= 0.0:
            recall = torch.zeros_like(tp)
        else:
            recall = tp / total_pos

        # Threshold values that correspond to the cumulative curve points.
        thresholds = torch.arange(self.num_bins - 1, -1, -1, device=tp.device, dtype=torch.float64) / float(self.num_bins)

        return {
            "tp": tp,
            "fp": fp,
            "recall": recall,
            "thresholds": thresholds,
            "total_pos": total_pos,
        }

    def compute(self) -> Dict[str, torch.Tensor]:
        c = self._counts

        recall = recall_from_counts(c)
        specificity = specificity_from_counts(c)

        metrics: Dict[str, torch.Tensor] = {
            "specificity": specificity,
            "dice": dice_from_counts(c),
            "f_beta": fbeta_from_counts(c, beta=self.beta),
            "mcc": mcc_from_counts(c),
            "balanced_accuracy": 0.5 * (recall + specificity),
        }

        curves = self._curves()
        tp = curves["tp"]
        fp = curves["fp"]
        total_pos = curves["total_pos"]

        # Best F_beta over histogram thresholds (approx).
        if float(total_pos.item()) <= 0.0:
            metrics["best_f_beta"] = torch.tensor(float("nan"), device=self.device, dtype=torch.float64)
            metrics["best_f_beta_threshold"] = torch.tensor(float("nan"), device=self.device, dtype=torch.float64)
        else:
            fn = total_pos - tp
            beta2 = self.beta ** 2
            fbeta_curve = _safe_div((1.0 + beta2) * tp, (1.0 + beta2) * tp + beta2 * fn + fp)
            best_idx = int(torch.argmax(fbeta_curve).item()) if fbeta_curve.numel() else 0
            metrics["best_f_beta"] = fbeta_curve[best_idx]
            metrics["best_f_beta_threshold"] = curves["thresholds"][best_idx]

        return metrics
