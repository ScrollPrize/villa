import torch
import numpy as np
from typing import Dict
from .base_metric import BaseMetric


class IOUDiceMetric(BaseMetric):
    def __init__(self, num_classes: int = 2, ignore_index: int = None, smooth: float = 1e-6):
        super().__init__("iou_dice")
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth
    
    def compute(self, pred: torch.Tensor, gt: torch.Tensor, **kwargs) -> Dict[str, float]:
        return compute_iou_dice(
            pred=pred,
            gt=gt,
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            smooth=self.smooth
        )


def compute_iou_dice(pred: torch.Tensor, 
                     gt: torch.Tensor, 
                     num_classes: int = 2,
                     ignore_index: int = None,
                     smooth: float = 1e-6) -> Dict[str, float]:
    
    # Move to CPU and convert to numpy for processing
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
        # Otherwise assume it's already (batch, depth, height, width) or (batch, height, width)
    elif pred_np.ndim == 3 and pred_np.shape[0] <= 10:  # (channels, height, width)
        if pred_np.shape[0] > 1:
            pred_np = np.argmax(pred_np, axis=0)
        else:
            pred_np = pred_np.squeeze(0)
    
    # Handle different input shapes for ground truth
    if gt_np.ndim == 3:  # (depth, height, width) or (height, width)
        gt_np = gt_np[np.newaxis, ...]  # Add batch dimension
    elif gt_np.ndim == 5:  # (batch, channels, depth, height, width)
        if gt_np.shape[1] == 1:
            gt_np = gt_np.squeeze(1)
        else:
            gt_np = np.argmax(gt_np, axis=1)
    elif gt_np.ndim == 4:  # Could be (batch, depth, height, width) or (batch, channels, height, width)
        if gt_np.shape[1] == 1:  # (batch, 1, height, width) for 2D
            gt_np = gt_np.squeeze(1)
        elif gt_np.shape[1] <= 10:  # Likely channels
            gt_np = np.argmax(gt_np, axis=1)
        # Otherwise assume it's already (batch, depth, height, width)
    elif gt_np.ndim == 2:  # (height, width)
        gt_np = gt_np[np.newaxis, ...]  # Add batch dimension
    
    # Ensure both have same number of dimensions
    if pred_np.ndim != gt_np.ndim:
        raise ValueError(f"Prediction and ground truth must have same number of dimensions after processing. Got pred: {pred_np.shape}, gt: {gt_np.shape}")
    
    batch_size = pred_np.shape[0]
    results = {}
    
    # Initialize per-class metrics
    for c in range(num_classes):
        if ignore_index is not None and c == ignore_index:
            continue
        results[f"iou_class_{c}"] = 0.0
        results[f"dice_class_{c}"] = 0.0
    
    # Calculate metrics for each batch
    for b in range(batch_size):
        pred_batch = pred_np[b]
        gt_batch = gt_np[b]
        
        for c in range(num_classes):
            if ignore_index is not None and c == ignore_index:
                continue
            
            # Create binary masks for current class
            pred_mask = (pred_batch == c).astype(np.float32)
            gt_mask = (gt_batch == c).astype(np.float32)
            
            # Calculate intersection and union
            intersection = np.sum(pred_mask * gt_mask)
            pred_sum = np.sum(pred_mask)
            gt_sum = np.sum(gt_mask)
            union = pred_sum + gt_sum - intersection
            
            # Calculate IOU
            iou = (intersection + smooth) / (union + smooth)
            results[f"iou_class_{c}"] += iou
            
            # Calculate Dice coefficient
            dice = (2 * intersection + smooth) / (pred_sum + gt_sum + smooth)
            results[f"dice_class_{c}"] += dice
    
    # Average over batch
    for key in results:
        results[key] /= batch_size
    
    # Calculate mean metrics across classes
    valid_classes = [c for c in range(num_classes) if ignore_index is None or c != ignore_index]
    
    if valid_classes:
        mean_iou = np.mean([results[f"iou_class_{c}"] for c in valid_classes])
        mean_dice = np.mean([results[f"dice_class_{c}"] for c in valid_classes])
        results["mean_iou"] = mean_iou
        results["mean_dice"] = mean_dice
    
    return results