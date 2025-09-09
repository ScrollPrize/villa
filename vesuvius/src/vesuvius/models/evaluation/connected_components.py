import numpy as np
import cc3d
import torch
from typing import Dict
from .base_metric import BaseMetric


class ConnectedComponentsMetric(BaseMetric):
    def __init__(self, num_classes: int = 2, connectivity: int = 26, ignore_index: int = 0):
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
            ignore_index=self.ignore_index
        )


def get_connected_components_difference(
                                         pred: torch.Tensor,
                                         gt: torch.Tensor,
                                         num_classes: int = 2,
                                         connectivity: int = 26,
                                         ignore_index: int = 0
                                         ) -> Dict[str, float]:
    # Convert BFloat16 to Float32 before numpy conversion
    if pred.dtype == torch.bfloat16:
        pred = pred.float()
    if gt.dtype == torch.bfloat16:
        gt = gt.float()
    
    pred_np = pred.detach().cpu().numpy()
    gt_np = gt.detach().cpu().numpy()

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
    
    # Handle different input shapes for ground truth
    if gt_np.ndim == 3:  # (depth, height, width)
        gt_np = gt_np[np.newaxis, ...]  # Add batch dimension
    elif gt_np.ndim == 5:  # (batch, channels, depth, height, width)
        if gt_np.shape[1] == 1:
            gt_np = gt_np.squeeze(1)
        else:
            gt_np = np.argmax(gt_np, axis=1)
    elif gt_np.ndim == 4:  # Could be (batch, depth, height, width) or (batch, 1, depth, height, width)
        if gt_np.shape[1] == 1:  # (batch, 1, height, width) for 2D or needs checking for 3D
            gt_np = gt_np.squeeze(1)
        # Otherwise assume it's already (batch, depth, height, width)

    batch_size = pred_np.shape[0]

    diff_per_class: Dict[str, float] = {}
    for c in range(num_classes):
        if ignore_index is not None and c == ignore_index:
            continue
        diff_per_class[f"connected_components_difference_class_{c}"] = 0.0

    total_gt_cc = 0
    total_pred_cc = 0

    for i in range(batch_size):
        for c in range(num_classes):
            if ignore_index is not None and c == ignore_index:
                continue

            gt_mask = (gt_np[i] == c).astype(np.uint8)
            pred_mask = (pred_np[i] == c).astype(np.uint8)

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

