import numpy as np
import torch
from typing import Dict
from skimage.morphology import skeletonize
from scipy.ndimage import convolve

from .base_metric import BaseMetric


class SkeletonBranchPointsMetric(BaseMetric):
    """
    Computes branch point count differences between prediction and ground truth.

    For each batch item and class, it:
    - Reduces channel predictions to class masks
      - If `num_classes == 2`: apply softmax over channels and threshold foreground at 0.5
      - If `num_classes > 2`: take argmax across channels and binarize per-class
    - For each z-slice, performs 2D skeletonization
    - Counts branch points (skeleton pixels with >2 8-neighbors)
    Aggregates counts across slices and averages over batch.
    """

    def __init__(self, num_classes: int = 2, ignore_index: int = None, threshold: float = 0.5):
        super().__init__("skeleton_branch_points")
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.threshold = threshold

    def compute(self, pred: torch.Tensor, gt: torch.Tensor, **kwargs) -> Dict[str, float]:
        # Convert BFloat16 to Float32 before numpy conversion
        if pred.dtype == torch.bfloat16:
            pred = pred.float()
        if gt.dtype == torch.bfloat16:
            gt = gt.float()

        pred_np = pred.detach().cpu().numpy()
        gt_np = gt.detach().cpu().numpy()

        # Reduce predictions to label maps or foreground probability (for 2-class)
        # Shapes handled: (B, C, Z, Y, X) or (B, C, H, W) or already label shaped
        if pred_np.ndim == 5:  # (B, C, Z, Y, X)
            if pred_np.shape[1] > 1:
                if self.num_classes == 2:
                    # Softmax along channel dim, take foreground prob (channel 1)
                    exps = np.exp(pred_np - np.max(pred_np, axis=1, keepdims=True))
                    softmax = exps / np.sum(exps, axis=1, keepdims=True)
                    pred_fg = softmax[:, 1]
                    pred_lbl = (pred_fg >= self.threshold).astype(np.int32)
                else:
                    pred_lbl = np.argmax(pred_np, axis=1).astype(np.int32)
            else:
                # Single-channel logits/probs for binary
                pred_lbl = (pred_np[:, 0] >= self.threshold).astype(np.int32)
        elif pred_np.ndim == 4:
            # Could be (B, C, H, W) or (B, Z, Y, X)
            if pred_np.shape[1] <= 10:  # likely channels
                if pred_np.shape[1] > 1:
                    if self.num_classes == 2:
                        exps = np.exp(pred_np - np.max(pred_np, axis=1, keepdims=True))
                        softmax = exps / np.sum(exps, axis=1, keepdims=True)
                        pred_fg = softmax[:, 1]
                        pred_lbl = (pred_fg >= self.threshold).astype(np.int32)
                    else:
                        pred_lbl = np.argmax(pred_np, axis=1).astype(np.int32)
                else:
                    pred_lbl = (pred_np[:, 0] >= self.threshold).astype(np.int32)
            else:
                # Already label volume (B, Z, Y, X)
                pred_lbl = pred_np.astype(np.int32)
        else:
            raise ValueError(f"Unsupported prediction shape for skeleton metric: {pred_np.shape}")

        # Prepare ground-truth labels
        if gt_np.ndim == 5:  # (B, C, Z, Y, X)
            if gt_np.shape[1] == 1:
                gt_lbl = gt_np[:, 0].astype(np.int32)
            else:
                gt_lbl = np.argmax(gt_np, axis=1).astype(np.int32)
        elif gt_np.ndim == 4:
            if gt_np.shape[1] == 1:  # (B,1,H,W)
                gt_lbl = gt_np[:, 0].astype(np.int32)
            elif gt_np.shape[1] <= 10:  # channels
                gt_lbl = np.argmax(gt_np, axis=1).astype(np.int32)
            else:
                gt_lbl = gt_np.astype(np.int32)
        elif gt_np.ndim == 3:
            gt_lbl = gt_np[np.newaxis, ...].astype(np.int32)
        else:
            raise ValueError(f"Unsupported ground truth shape for skeleton metric: {gt_np.shape}")

        # Ensure both are 4D (B, Z, Y, X). If 2D, add a singleton Z
        if pred_lbl.ndim == 3:
            pred_lbl = pred_lbl[:, np.newaxis, ...]
        if gt_lbl.ndim == 3:
            gt_lbl = gt_lbl[:, np.newaxis, ...]

        batch_size = pred_lbl.shape[0]

        # Neighbor-count kernel for 8-neighborhood in 2D (exclude center)
        neigh_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)

        results: Dict[str, float] = {}
        # Initialize accumulators
        classes = range(self.num_classes) if self.num_classes is not None else [1]
        for c in classes:
            if self.ignore_index is not None and c == self.ignore_index:
                continue
            results[f"branch_points_pred_class_{c}"] = 0.0
            results[f"branch_points_gt_class_{c}"] = 0.0
            results[f"branch_points_absdiff_class_{c}"] = 0.0

        total_pred = 0.0
        total_gt = 0.0

        for b in range(batch_size):
            for c in classes:
                if self.ignore_index is not None and c == self.ignore_index:
                    continue

                pred_mask = (pred_lbl[b] == c).astype(np.uint8)
                gt_mask = (gt_lbl[b] == c).astype(np.uint8)

                pred_bp = 0
                gt_bp = 0

                # Iterate over z-slices
                for z in range(pred_mask.shape[0]):
                    # Prediction slice
                    if pred_mask[z].any():
                        skel_pred = skeletonize(pred_mask[z].astype(bool))
                        neigh_pred = convolve(skel_pred.astype(np.uint8), neigh_kernel, mode='constant', cval=0)
                        pred_bp += int(((skel_pred.astype(np.uint8) == 1) & (neigh_pred >= 3)).sum())

                    # Ground truth slice
                    if gt_mask[z].any():
                        skel_gt = skeletonize(gt_mask[z].astype(bool))
                        neigh_gt = convolve(skel_gt.astype(np.uint8), neigh_kernel, mode='constant', cval=0)
                        gt_bp += int(((skel_gt.astype(np.uint8) == 1) & (neigh_gt >= 3)).sum())

                results[f"branch_points_pred_class_{c}"] += pred_bp
                results[f"branch_points_gt_class_{c}"] += gt_bp
                results[f"branch_points_absdiff_class_{c}"] += abs(pred_bp - gt_bp)
                total_pred += pred_bp
                total_gt += gt_bp

        # Average per batch
        valid_classes = [c for c in classes if (self.ignore_index is None or c != self.ignore_index)]
        for c in valid_classes:
            results[f"branch_points_pred_class_{c}"] /= batch_size
            results[f"branch_points_gt_class_{c}"] /= batch_size
            results[f"branch_points_absdiff_class_{c}"] /= batch_size

        results["branch_points_pred_total"] = total_pred / batch_size
        results["branch_points_gt_total"] = total_gt / batch_size
        results["branch_points_absdiff_total"] = abs(total_pred - total_gt) / batch_size

        return results

