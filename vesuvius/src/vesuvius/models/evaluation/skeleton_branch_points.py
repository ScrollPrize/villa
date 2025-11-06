import numpy as np
import torch
from typing import Dict, Optional, Tuple
from skimage.morphology import skeletonize
from scipy.ndimage import convolve

# Optional acceleration via OpenCV thinning if available
try:  # pragma: no cover - optional dependency
    import cv2  # type: ignore
    _HAS_CV2 = True
    _HAS_XIMGPROC = hasattr(cv2, 'ximgproc')
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore
    _HAS_CV2 = False
    _HAS_XIMGPROC = False

from .base_metric import BaseMetric


class SkeletonBranchPointsMetric(BaseMetric):
    """
    Computes branch point differences between prediction and ground truth,
    averaging branch-point counts over Z slices (per-class).

    For each batch item and class, it:
    - Reduces channel predictions to class masks
      - If `num_classes == 2`: apply softmax over channels and threshold foreground at 0.5
      - If `num_classes > 2`: take argmax across channels and binarize per-class
    - For each z-slice, performs 2D skeletonization
    - Counts branch points (skeleton pixels with >2 8-neighbors)
    Computes the mean count across Z slices and then averages over the batch.
    """

    def __init__(self, num_classes: int = 2, ignore_index: int = 0, threshold: float = 0.5):
        """
        Initialize the metric.

        By default, background (class 0) is ignored so skeletonization
        only runs on the foreground class(es).
        """
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
        # We lift it to 3D with a singleton z-dimension so we can convolve all slices at once
        neigh_kernel_2d = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
        neigh_kernel = neigh_kernel_2d[np.newaxis, :, :]

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

                pred_mask = (pred_lbl[b] == c)
                gt_mask = (gt_lbl[b] == c)

                # Fast exit if no foreground at all for either
                has_pred = bool(pred_mask.any())
                has_gt = bool(gt_mask.any())

                pred_bp = 0.0
                gt_bp = 0.0

                if has_pred:
                    skel_pred_u8 = self._skeletonize_stack_2d(pred_mask)
                    neigh_pred = convolve(skel_pred_u8, neigh_kernel, mode='constant', cval=0)
                    pred_bp_per_slice = ((skel_pred_u8 == 1) & (neigh_pred >= 3)).sum(axis=(1, 2))
                    # Average branch points across Z slices
                    pred_bp = float(pred_bp_per_slice.mean())

                if has_gt:
                    skel_gt_u8 = self._skeletonize_stack_2d(gt_mask)
                    neigh_gt = convolve(skel_gt_u8, neigh_kernel, mode='constant', cval=0)
                    gt_bp_per_slice = ((skel_gt_u8 == 1) & (neigh_gt >= 3)).sum(axis=(1, 2))
                    # Average branch points across Z slices
                    gt_bp = float(gt_bp_per_slice.mean())

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

    @staticmethod
    def _roi_bounds(mask2d: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Compute tight bounding box (ymin, ymax, xmin, xmax) for a 2D boolean mask.
        Returns None if the mask is empty. Pads by 1 pixel within bounds for stability.
        """
        ys, xs = np.nonzero(mask2d)
        if ys.size == 0:
            return None
        y0 = int(ys.min())
        y1 = int(ys.max()) + 1
        x0 = int(xs.min())
        x1 = int(xs.max()) + 1
        # 1px pad (clipped later by caller)
        return y0 - 1, y1 + 1, x0 - 1, x1 + 1

    def _skeletonize_slice(self, mask2d: np.ndarray) -> np.ndarray:
        """Skeletonize a single 2D slice (boolean array) efficiently with ROI cropping.
        Uses OpenCV ximgproc thinning if available; falls back to skimage.skeletonize.
        Returns uint8 array with values {0,1}.
        """
        if not mask2d.any():
            return mask2d.astype(np.uint8)

        H, W = mask2d.shape
        bounds = self._roi_bounds(mask2d)

        if bounds is None:
            return np.zeros((H, W), dtype=np.uint8)

        y0, y1, x0, x1 = bounds
        y0 = max(0, y0)
        x0 = max(0, x0)
        y1 = min(H, y1)
        x1 = min(W, x1)

        roi = mask2d[y0:y1, x0:x1]

        # Prefer OpenCV thinning for speed if available
        if _HAS_CV2 and _HAS_XIMGPROC:
            roi_u8 = (roi.astype(np.uint8)) * 255
            skel_roi_u8 = cv2.ximgproc.thinning(roi_u8)
            skel_roi = (skel_roi_u8 > 0).astype(np.uint8)
        else:
            # skimage expects boolean
            skel_roi = skeletonize(roi).astype(np.uint8)

        out = np.zeros((H, W), dtype=np.uint8)
        out[y0:y1, x0:x1] = skel_roi
        return out

    def _skeletonize_stack_2d(self, mask3d: np.ndarray) -> np.ndarray:
        """Apply 2D skeletonization slice-wise over a 3D boolean mask [Z, Y, X].
        Returns a uint8 array with shape [Z, Y, X]. Includes ROI cropping per-slice.
        """
        Z = mask3d.shape[0]
        # Pre-allocate output to avoid repeated allocations
        out = np.zeros_like(mask3d, dtype=np.uint8)
        for z in range(Z):
            m2d = mask3d[z]
            if m2d.any():
                out[z] = self._skeletonize_slice(m2d)
        return out
