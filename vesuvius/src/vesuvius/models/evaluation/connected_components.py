import numpy as np
import cc3d
import torch
from typing import Dict, Optional
from .base_metric import BaseMetric


class ConnectedComponentsMetric(BaseMetric):
    def __init__(self, num_classes: int = 2, connectivity: int = 26, ignore_index: Optional[int] = None):
        super().__init__("connected_components")
        self.num_classes = num_classes
        self.connectivity = connectivity
        self.ignore_index = ignore_index
    
    def compute(self, pred: torch.Tensor, gt: torch.Tensor, **kwargs) -> Dict[str, float]:
        return get_connected_components_difference(
            pred=pred,
            gt=gt,
            num_classes=self.num_classes,
            connectivity=self.connectivity,
            ignore_index=self.ignore_index,
            mask=kwargs.get("mask")
        )


def get_connected_components_difference(
                                         pred: torch.Tensor,
                                         gt: torch.Tensor,
                                         num_classes: int = 2,
                                         connectivity: int = 26,
                                         ignore_index: Optional[int] = None,
                                         mask: Optional[torch.Tensor] = None
                                         ) -> Dict[str, float]:
    def _normalize_mask(mask_arr: np.ndarray, target_shape: tuple) -> np.ndarray:
        if mask_arr.ndim == len(target_shape) + 1 and mask_arr.shape[1] == 1:
            mask_arr = np.squeeze(mask_arr, axis=1)
        if mask_arr.ndim == len(target_shape) - 1:
            mask_arr = np.expand_dims(mask_arr, axis=0)
        if mask_arr.shape != target_shape:
            try:
                mask_arr = np.broadcast_to(mask_arr, target_shape)
            except ValueError as exc:
                raise ValueError(
                    f"Mask shape {mask_arr.shape} is not compatible with target shape {target_shape}"
                ) from exc
        return mask_arr.astype(bool)

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
            mask_np = mask_tensor.detach().cpu().numpy()
        else:
            mask_np = np.asarray(mask)

    # Handle different input shapes for predictions
    if pred_np.ndim == 5:  # (batch, channels, depth, height, width)
        if pred_np.shape[1] > 1:  # Multi-channel, need argmax
            pred_np = np.argmax(pred_np, axis=1)
        else:  # Single channel, just squeeze
            pred_np = pred_np.squeeze(1)
    elif pred_np.ndim == 4:  # Could be (batch, depth, height, width) or (batch, channels, height, width)
        # Check if second dimension is channels (usually small) or spatial dimension
        if pred_np.shape[1] <= 10:  # Likely channels dimension
            if pred_np.shape[1] > 1:
                pred_np = np.argmax(pred_np, axis=1)
            else:
                pred_np = pred_np.squeeze(1)
        # Otherwise assume it's already (batch, depth, height, width)
    
    # Handle different input shapes for ground truth (and align mask)
    if gt_np.ndim == 3:  # (depth, height, width)
        gt_np = gt_np[np.newaxis, ...]  # Add batch dimension
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
    elif gt_np.ndim == 4:  # Could be (batch, depth, height, width) or (batch, 1, depth, height, width)
        if gt_np.shape[1] == 1:  # (batch, 1, height, width) for 2D or needs checking for 3D
            gt_np = gt_np.squeeze(1)
            if mask_np is not None and mask_np.ndim == 4 and mask_np.shape[1] == 1:
                mask_np = mask_np.squeeze(1)
        # Otherwise assume it's already (batch, depth, height, width)

    if pred_np.ndim != gt_np.ndim:
        raise ValueError(
            f"Prediction and ground truth must have same number of dimensions after processing. "
            f"Got pred: {pred_np.shape}, gt: {gt_np.shape}"
        )

    if mask_np is not None:
        mask_np = _normalize_mask(mask_np, gt_np.shape)

    batch_size = pred_np.shape[0]

    valid_mask = np.ones_like(gt_np, dtype=bool)
    if mask_np is not None:
        valid_mask &= mask_np
    if ignore_index is not None:
        valid_mask &= (gt_np != ignore_index)

    diff_per_class: Dict[str, float] = {}
    for c in range(num_classes):
        if ignore_index is not None and c == ignore_index:
            continue
        diff_per_class[f"connected_components_difference_class_{c}"] = 0.0

    total_gt_cc = 0
    total_pred_cc = 0

    for i in range(batch_size):
        current_valid = valid_mask[i]
        for c in range(num_classes):
            if ignore_index is not None and c == ignore_index:
                continue

            gt_mask = ((gt_np[i] == c) & current_valid).astype(np.uint8)
            pred_mask = ((pred_np[i] == c) & current_valid).astype(np.uint8)

            cc_gt = cc3d.connected_components(gt_mask, connectivity=connectivity)
            cc_pred = cc3d.connected_components(pred_mask, connectivity=connectivity)

            num_cc_gt = int(cc_gt.max())
            num_cc_pred = int(cc_pred.max())

            diff = abs(num_cc_pred - num_cc_gt)
            diff_per_class[f"connected_components_difference_class_{c}"] += diff

            total_gt_cc += num_cc_gt
            total_pred_cc += num_cc_pred

    for key in diff_per_class:
        diff_per_class[key] /= batch_size

    total_diff = abs(total_pred_cc - total_gt_cc) / batch_size
    diff_per_class["connected_components_difference_total"] = total_diff
    
    return diff_per_class
