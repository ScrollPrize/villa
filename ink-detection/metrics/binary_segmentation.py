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
    fn: torch.Tensor
    tn: torch.Tensor


def confusion_counts(preds: torch.Tensor, targets: torch.Tensor) -> ConfusionCounts:
    """Compute TP/FP/FN/TN for boolean tensors (any shape).

    Returns float64 tensors for numeric stability.
    """
    preds = preds.bool()
    targets = targets.bool()
    tp = (preds & targets).sum(dtype=torch.float64)
    fp = (preds & ~targets).sum(dtype=torch.float64)
    fn = (~preds & targets).sum(dtype=torch.float64)
    tn = (~preds & ~targets).sum(dtype=torch.float64)
    return ConfusionCounts(tp=tp, fp=fp, fn=fn, tn=tn)


def dice_from_counts(c: ConfusionCounts) -> torch.Tensor:
    # Dice/F1: 2TP / (2TP + FP + FN)
    return _safe_div(2.0 * c.tp, 2.0 * c.tp + c.fp + c.fn)


def _recall_or_nan(tp_like: torch.Tensor, fn_like: torch.Tensor) -> torch.Tensor:
    denom = tp_like + fn_like
    return torch.where(denom > 0.0, tp_like / denom, torch.full_like(denom, torch.nan))


def balanced_accuracy_from_counts(c: ConfusionCounts) -> torch.Tensor:
    """Balanced accuracy as mean recall across present classes.

    This matches the common definition used by sklearn:
    balanced_accuracy = (TPR + TNR) / 2 in the binary case, with absent-class
    recalls ignored.
    """
    pos_recall = _recall_or_nan(c.tp, c.fn)  # TPR / sensitivity
    neg_recall = _recall_or_nan(c.tn, c.fp)  # TNR / specificity
    recalls = torch.stack((pos_recall, neg_recall))
    valid = ~torch.isnan(recalls)
    if bool(valid.any()):
        return recalls[valid].mean()
    return torch.zeros((), device=recalls.device, dtype=recalls.dtype)


class StreamingBinarySegmentationMetrics:
    """Streaming (epoch-accumulated) pixel-level metrics for binary segmentation.

    Designed for validation-time evaluation where collecting all pixels in-memory
    may be too expensive.

    Notes:
    - Confusion-based metrics are computed at a fixed threshold.
    """

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        device: Optional[torch.device] = None,
    ) -> None:
        if not (0.0 <= float(threshold) <= 1.0):
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")

        self.threshold = float(threshold)
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
            fn=torch.zeros((), device=dev, dtype=torch.float64),
            tn=torch.zeros((), device=dev, dtype=torch.float64),
        )

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
        self._counts.fn += batch_counts.fn
        self._counts.tn += batch_counts.tn

    def compute(self) -> Dict[str, torch.Tensor]:
        c = self._counts

        return {
            "dice": dice_from_counts(c),
            "balanced_accuracy": balanced_accuracy_from_counts(c),
        }
