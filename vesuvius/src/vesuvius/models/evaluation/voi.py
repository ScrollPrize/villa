import numpy as np
import cc3d
import torch
from typing import Dict, Optional, Tuple
from skimage.metrics import variation_of_information

from .base_metric import BaseMetric


class VOIMetric(BaseMetric):
    """
    Variation of Information (VOI) metric for 3D segmentation.

    Computes VOI between connected-component labelings derived from binary FG masks.
    Uses 26-connectivity for 3D connected components.

    Returns:
        voi_total: voi_split + voi_merge (lower is better)
        voi_split: H(GT | PR) - over-segmentation term
        voi_merge: H(PR | GT) - under-segmentation term
        voi_score: normalized to [0,1] (higher is better)
    """

    def __init__(
        self,
        connectivity: int = 26,
        use_union_mask: bool = True,
        alpha: float = 1.0,
        ignore_index: Optional[int] = None,
    ):
        super().__init__("voi")
        self.connectivity = connectivity
        self.use_union_mask = use_union_mask
        self.alpha = alpha
        self.ignore_index = ignore_index

    def compute(self, pred: torch.Tensor, gt: torch.Tensor, **kwargs) -> Dict[str, float]:
        mask = kwargs.get("mask")
        return compute_voi(
            pred=pred,
            gt=gt,
            connectivity=self.connectivity,
            use_union_mask=self.use_union_mask,
            alpha=self.alpha,
            ignore_index=self.ignore_index,
            mask=mask,
        )


def _bbox3d(mask: np.ndarray) -> Optional[Tuple[slice, slice, slice]]:
    """Compute 3D bounding box of binary mask."""
    assert mask.ndim == 3
    sls = []
    for ax in range(3):
        proj = mask.any(axis=tuple(i for i in range(3) if i != ax))
        idx = np.flatnonzero(proj)
        if idx.size == 0:
            return None
        sls.append(slice(int(idx[0]), int(idx[-1]) + 1))
    return (sls[0], sls[1], sls[2])


def _compute_voi_3d(
    pred_bin: np.ndarray,
    gt_bin: np.ndarray,
    connectivity: int,
    use_union_mask: bool,
    alpha: float,
) -> Dict[str, float]:
    """
    Compute VOI for a single 3D volume pair.
    Both inputs should be binary masks (bool or 0/1).
    """
    # Early out if nothing remains
    union = gt_bin | pred_bin
    if not union.any():
        return {
            "voi_total": 0.0,
            "voi_split": 0.0,
            "voi_merge": 0.0,
            "voi_score": 1.0,
        }

    # Crop to union bounding box for speed
    slc = _bbox3d(union)
    if slc is not None:
        crop_ratio = 0.7
        full_n = int(np.prod(gt_bin.shape, dtype=np.int64))
        bbox_n = int(
            (slc[0].stop - slc[0].start) *
            (slc[1].stop - slc[1].start) *
            (slc[2].stop - slc[2].start)
        )
        use_crop = (bbox_n <= int(crop_ratio * full_n))

        if use_crop:
            gt_use = gt_bin[slc]
            pred_use = pred_bin[slc]
        else:
            gt_use = gt_bin
            pred_use = pred_bin
    else:
        gt_use = gt_bin
        pred_use = pred_bin

    # Connected components (3D)
    gt_lab = cc3d.connected_components(gt_use.astype(np.uint8), connectivity=connectivity)
    pred_lab = cc3d.connected_components(pred_use.astype(np.uint8), connectivity=connectivity)

    if use_union_mask:
        m = (gt_lab > 0) | (pred_lab > 0)
        a = gt_lab[m]
        b = pred_lab[m]
    else:
        a = gt_lab.ravel()
        b = pred_lab.ravel()

    voi_split, voi_merge = variation_of_information(a, b)
    voi_total = float(voi_split + voi_merge)

    # Transform to [0,1] score (higher is better)
    voi_score = float(1.0 / (1.0 + alpha * voi_total))

    return {
        "voi_total": voi_total,
        "voi_split": float(voi_split),
        "voi_merge": float(voi_merge),
        "voi_score": voi_score,
    }


def compute_voi(
    pred: torch.Tensor,
    gt: torch.Tensor,
    connectivity: int = 26,
    use_union_mask: bool = True,
    alpha: float = 1.0,
    ignore_index: Optional[int] = None,
    mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Compute VOI metric for batched 3D predictions and ground truth.

    Args:
        pred: Prediction tensor
        gt: Ground truth tensor
        connectivity: 3D connectivity (6, 18, or 26)
        use_union_mask: Only count voxels where either pred or gt is foreground
        alpha: Strength parameter for score transform
        ignore_index: Label value to ignore
        mask: Optional binary mask for valid regions

    Returns:
        Dict with voi_total, voi_split, voi_merge, voi_score
    """
    # Convert BFloat16 to Float32 before numpy conversion
    if pred.dtype == torch.bfloat16:
        pred = pred.float()
    if gt.dtype == torch.bfloat16:
        gt = gt.float()

    pred_np = pred.detach().cpu().numpy()
    gt_np = gt.detach().cpu().numpy()
    mask_np: Optional[np.ndarray] = None

    if mask is not None:
        if isinstance(mask, torch.Tensor):
            mask_tensor = mask
            if mask_tensor.dtype == torch.bfloat16:
                mask_tensor = mask_tensor.float()
            mask_np = mask_tensor.detach().cpu().numpy().astype(bool)
        else:
            mask_np = np.asarray(mask).astype(bool)

    # Handle different input shapes for predictions
    if pred_np.ndim == 5:  # (batch, channels, depth, height, width)
        if pred_np.shape[1] > 1:  # Multi-channel, need argmax
            pred_np = np.argmax(pred_np, axis=1)
        else:  # Single channel, just squeeze
            pred_np = pred_np.squeeze(1)
    elif pred_np.ndim == 4:  # Could be (batch, depth, height, width) or (batch, channels, height, width)
        if pred_np.shape[1] <= 10:  # Likely channels dimension
            if pred_np.shape[1] > 1:
                pred_np = np.argmax(pred_np, axis=1)
            else:
                pred_np = pred_np.squeeze(1)

    # Handle different input shapes for ground truth
    if gt_np.ndim == 3:  # (depth, height, width)
        gt_np = gt_np[np.newaxis, ...]
        if mask_np is not None and mask_np.ndim == 3:
            mask_np = mask_np[np.newaxis, ...]
    elif gt_np.ndim == 5:  # (batch, channels, depth, height, width)
        if gt_np.shape[1] == 1:
            gt_np = gt_np.squeeze(1)
            if mask_np is not None and mask_np.ndim == 5 and mask_np.shape[1] == 1:
                mask_np = mask_np.squeeze(1)
        else:
            gt_np = np.argmax(gt_np, axis=1)
            if mask_np is not None and mask_np.ndim == 5 and mask_np.shape[1] == 1:
                mask_np = mask_np.squeeze(1)
    elif gt_np.ndim == 4:
        if gt_np.shape[1] == 1:
            gt_np = gt_np.squeeze(1)
            if mask_np is not None and mask_np.ndim == 4 and mask_np.shape[1] == 1:
                mask_np = mask_np.squeeze(1)

    if pred_np.ndim != gt_np.ndim:
        raise ValueError(
            f"Prediction and ground truth must have same number of dimensions after processing. "
            f"Got pred: {pred_np.shape}, gt: {gt_np.shape}"
        )

    batch_size = pred_np.shape[0]

    # Accumulate results across batch
    voi_totals = []
    voi_splits = []
    voi_merges = []
    voi_scores = []

    for i in range(batch_size):
        pred_vol = pred_np[i]
        gt_vol = gt_np[i]

        # Build valid mask
        valid = np.ones_like(gt_vol, dtype=bool)
        if mask_np is not None:
            valid &= mask_np[i] if mask_np.ndim == 4 else mask_np
        if ignore_index is not None:
            valid &= (gt_vol != ignore_index)

        # Create binary masks (foreground = class 1 for binary, or nonzero for multi-class)
        # Apply valid mask by setting invalid regions to 0
        pred_bin = ((pred_vol == 1) & valid) if pred_vol.max() <= 1 else ((pred_vol > 0) & valid)
        gt_bin = ((gt_vol == 1) & valid) if gt_vol.max() <= 1 else ((gt_vol > 0) & valid)

        result = _compute_voi_3d(
            pred_bin=pred_bin,
            gt_bin=gt_bin,
            connectivity=connectivity,
            use_union_mask=use_union_mask,
            alpha=alpha,
        )

        voi_totals.append(result["voi_total"])
        voi_splits.append(result["voi_split"])
        voi_merges.append(result["voi_merge"])
        voi_scores.append(result["voi_score"])

    return {
        "voi_total": float(np.mean(voi_totals)),
        "voi_split": float(np.mean(voi_splits)),
        "voi_merge": float(np.mean(voi_merges)),
        "voi_score": float(np.mean(voi_scores)),
    }
