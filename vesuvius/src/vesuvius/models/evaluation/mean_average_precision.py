from typing import Dict, List, Tuple
import numpy as np
import torch
import cc3d

from .base_metric import BaseMetric


def _label_components(mask: np.ndarray, connectivity: int) -> np.ndarray:
    # cc3d supports 2D and 3D arrays
    return cc3d.connected_components(mask.astype(np.uint8), connectivity=connectivity)


def _compute_iou_matrix(gt_cc: np.ndarray, pred_cc: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Build contingency table efficiently using bincount on combined ids
    gt_ids = gt_cc.ravel().astype(np.int64)
    pred_ids = pred_cc.ravel().astype(np.int64)
    gt_max = int(gt_ids.max())
    pred_max = int(pred_ids.max())

    if gt_max == 0 and pred_max == 0:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    base = pred_max + 1
    joint_ids = gt_ids * base + pred_ids
    counts = np.bincount(joint_ids)
    # Expand to full matrix (gt_max+1, pred_max+1)
    inter = np.zeros((gt_max + 1, pred_max + 1), dtype=np.int64)
    # Only fill indices that exist in counts
    nz = counts.nonzero()[0]
    inter_flat = inter.ravel()
    inter_flat[nz] = counts[nz]
    inter = inter_flat.reshape(inter.shape)

    # Area per component (exclude background later)
    gt_areas = inter.sum(axis=1)  # sum over pred ids
    pred_areas = inter.sum(axis=0)  # sum over gt ids

    # Remove background (0) rows/cols when computing IoU pairs
    if gt_max == 0 or pred_max == 0:
        iou = np.zeros((gt_max, pred_max), dtype=np.float32)
        return iou, gt_areas[1:].astype(np.float32), pred_areas[1:].astype(np.float32)

    inter_fg = inter[1:, 1:].astype(np.float32)
    gt_areas_fg = gt_areas[1:].astype(np.float32)
    pred_areas_fg = pred_areas[1:].astype(np.float32)
    # Broadcast union = area_gt + area_pred - intersection
    union = gt_areas_fg[:, None] + pred_areas_fg[None, :] - inter_fg
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = np.where(union > 0, inter_fg / union, 0.0)
    return iou, gt_areas_fg, pred_areas_fg


class MeanAveragePrecisionMetric(BaseMetric):
    """
    Instance-style mean Average Precision for segmentation via connected components.

    - For each class, label connected components in GT and prediction.
    - For thresholds T in {0.50, 0.55, ..., 0.95}, greedily match pairs with IoU >= T.
    - For each T, compute precision = TP / (TP + FP). mAP_class is the mean of precision(T) over thresholds.

    Note: predictions have no confidence scores; this yields a single operating point per T.
    """

    def __init__(self, num_classes: int = 2, connectivity: int = 26, ignore_index: int = 0):
        super().__init__("mean_average_precision")
        self.num_classes = num_classes
        self.connectivity = connectivity
        self.ignore_index = ignore_index
        # COCO-style thresholds 0.50:0.05:0.95
        self.thresholds: List[float] = [0.50 + i * 0.05 for i in range(10)]

    def compute(self, pred: torch.Tensor, gt: torch.Tensor, **kwargs) -> Dict[str, float]:
        # Convert to labels
        if pred.dtype == torch.bfloat16:
            pred = pred.float()
        if gt.dtype == torch.bfloat16:
            gt = gt.float()

        pred_np = pred.detach().cpu().numpy()
        gt_np = gt.detach().cpu().numpy()

        # Reduce predictions to label map
        if pred_np.ndim == 5:
            if pred_np.shape[1] > 1:
                pred_lbl = np.argmax(pred_np, axis=1).astype(np.int32)
            else:
                pred_lbl = (pred_np[:, 0] > 0.5).astype(np.int32)
        elif pred_np.ndim == 4:
            if pred_np.shape[1] <= 10:
                if pred_np.shape[1] > 1:
                    pred_lbl = np.argmax(pred_np, axis=1).astype(np.int32)
                else:
                    pred_lbl = (pred_np[:, 0] > 0.5).astype(np.int32)
            else:
                pred_lbl = pred_np.astype(np.int32)
        else:
            raise ValueError(f"Unsupported prediction shape for mAP: {pred_np.shape}")

        # Reduce gt to label map
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
            raise ValueError(f"Unsupported ground truth shape for mAP: {gt_np.shape}")

        batch_size = pred_lbl.shape[0]
        classes = [c for c in range(self.num_classes) if (self.ignore_index is None or c != self.ignore_index)]

        ap_sums = {c: 0.0 for c in classes}
        for b in range(batch_size):
            pb = pred_lbl[b]
            gb = gt_lbl[b]
            for c in classes:
                gt_mask = (gb == c)
                pred_mask = (pb == c)

                gt_cc = _label_components(gt_mask, self.connectivity)
                pred_cc = _label_components(pred_mask, self.connectivity)

                iou_mat, gt_areas, pred_areas = _compute_iou_matrix(gt_cc, pred_cc)
                n_gt = gt_areas.shape[0]
                n_pred = pred_areas.shape[0]

                if n_gt == 0 and n_pred == 0:
                    # Perfect empty case: precision defined as 1 across thresholds
                    ap = 1.0
                else:
                    precisions: List[float] = []
                    for thr in self.thresholds:
                        if iou_mat.size == 0:
                            tp = 0
                            fp = n_pred
                            # fn = n_gt
                        else:
                            # Greedy matching by IoU
                            gt_used = np.zeros(n_gt, dtype=bool)
                            pred_used = np.zeros(n_pred, dtype=bool)
                            # All pair indices with IoU >= thr
                            gt_idx, pred_idx = np.nonzero(iou_mat >= thr)
                            if gt_idx.size > 0:
                                scores = iou_mat[gt_idx, pred_idx]
                                order = np.argsort(-scores)
                                tp = 0
                                for k in order:
                                    gi = gt_idx[k]
                                    pi = pred_idx[k]
                                    if gt_used[gi] or pred_used[pi]:
                                        continue
                                    gt_used[gi] = True
                                    pred_used[pi] = True
                                    tp += 1
                            else:
                                tp = 0
                            fp = int(n_pred - tp)
                            # fn = int(n_gt - tp)
                        prec = tp / float(tp + fp) if (tp + fp) > 0 else (1.0 if n_gt == 0 else 0.0)
                        precisions.append(prec)
                    ap = float(np.mean(precisions)) if precisions else (1.0 if n_gt == 0 else 0.0)

                ap_sums[c] += ap

        results: Dict[str, float] = {}
        for c in classes:
            results[f"map_class_{c}"] = ap_sums[c] / batch_size
        results["map_mean"] = float(np.mean([results[f"map_class_{c}"] for c in classes])) if classes else 0.0
        return results
