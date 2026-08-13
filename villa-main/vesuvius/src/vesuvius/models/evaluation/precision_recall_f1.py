import numpy as np
import torch
from typing import Dict

from .base_metric import BaseMetric


class PrecisionRecallF1Metric(BaseMetric):
    def __init__(self, num_classes: int = 2, ignore_index: int = 0, eps: float = 1e-8):
        super().__init__("precision_recall_f1")
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.eps = eps

    def compute(self, pred: torch.Tensor, gt: torch.Tensor, **kwargs) -> Dict[str, float]:
        # Convert bfloat16 early
        if pred.dtype == torch.bfloat16:
            pred = pred.float()
        if gt.dtype == torch.bfloat16:
            gt = gt.float()

        pred_np = pred.detach().cpu().numpy()
        gt_np = gt.detach().cpu().numpy()

        # Reduce predictions to label indices
        if pred_np.ndim == 5:  # (B, C, Z, Y, X)
            if pred_np.shape[1] > 1:
                pred_lbl = np.argmax(pred_np, axis=1).astype(np.int32)
            else:
                pred_lbl = (pred_np[:, 0] > 0.5).astype(np.int32)
        elif pred_np.ndim == 4:
            if pred_np.shape[1] <= 10:  # treat as channels
                if pred_np.shape[1] > 1:
                    pred_lbl = np.argmax(pred_np, axis=1).astype(np.int32)
                else:
                    pred_lbl = (pred_np[:, 0] > 0.5).astype(np.int32)
            else:
                pred_lbl = pred_np.astype(np.int32)
        else:
            raise ValueError(f"Unsupported pred shape for F1: {pred_np.shape}")

        if gt_np.ndim == 5:
            if gt_np.shape[1] == 1:
                gt_lbl = gt_np[:, 0].astype(np.int32)
            else:
                gt_lbl = np.argmax(gt_np, axis=1).astype(np.int32)
        elif gt_np.ndim == 4:
            if gt_np.shape[1] == 1:
                gt_lbl = gt_np[:, 0].astype(np.int32)
            elif gt_np.shape[1] <= 10:
                gt_lbl = np.argmax(gt_np, axis=1).astype(np.int32)
            else:
                gt_lbl = gt_np.astype(np.int32)
        elif gt_np.ndim == 3:
            gt_lbl = gt_np[np.newaxis, ...].astype(np.int32)
        else:
            raise ValueError(f"Unsupported gt shape for F1: {gt_np.shape}")

        batch_size = pred_lbl.shape[0]
        results: Dict[str, float] = {}

        # Initialize accumulators
        classes = list(range(self.num_classes))
        valid_classes = [c for c in classes if (self.ignore_index is None or c != self.ignore_index)]

        prec_sums = {c: 0.0 for c in valid_classes}
        rec_sums = {c: 0.0 for c in valid_classes}
        f1_sums = {c: 0.0 for c in valid_classes}

        for b in range(batch_size):
            pb = pred_lbl[b]
            gb = gt_lbl[b]
            for c in valid_classes:
                tp = float(np.sum((pb == c) & (gb == c)))
                fp = float(np.sum((pb == c) & (gb != c)))
                fn = float(np.sum((pb != c) & (gb == c)))

                prec = tp / (tp + fp + self.eps)
                rec = tp / (tp + fn + self.eps)
                f1 = 2.0 * prec * rec / (prec + rec + self.eps)

                prec_sums[c] += prec
                rec_sums[c] += rec
                f1_sums[c] += f1

        # Average over batch
        for c in valid_classes:
            results[f"precision_class_{c}"] = prec_sums[c] / batch_size
            results[f"recall_class_{c}"] = rec_sums[c] / batch_size
            results[f"f1_class_{c}"] = f1_sums[c] / batch_size

        # Macro means across classes
        if valid_classes:
            results["precision_mean"] = float(np.mean([results[f"precision_class_{c}"] for c in valid_classes]))
            results["recall_mean"] = float(np.mean([results[f"recall_class_{c}"] for c in valid_classes]))
            results["f1_mean"] = float(np.mean([results[f"f1_class_{c}"] for c in valid_classes]))

        return results
