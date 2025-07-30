import torch
import numpy as np
from typing import Dict
from scipy.ndimage import distance_transform_edt
from .base_metric import BaseMetric


class HausdorffDistanceMetric(BaseMetric):
    def __init__(self, num_classes: int = 2, ignore_index: int = None, percentile: float = 95.0):
        super().__init__("hausdorff_distance")
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.percentile = percentile
    
    def compute(self, pred: torch.Tensor, gt: torch.Tensor, **kwargs) -> Dict[str, float]:
        return compute_hausdorff_distance(
            pred=pred,
            gt=gt,
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            percentile=self.percentile
        )


def compute_hausdorff_distance(pred: torch.Tensor, 
                               gt: torch.Tensor, 
                               num_classes: int = 2,
                               ignore_index: int = None,
                               percentile: float = 95.0) -> Dict[str, float]:
    
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
        results[f"hausdorff_distance_class_{c}"] = 0.0
        results[f"hausdorff_distance_{int(percentile)}_class_{c}"] = 0.0
    
    # Calculate metrics for each batch
    for b in range(batch_size):
        pred_batch = pred_np[b]
        gt_batch = gt_np[b]
        
        for c in range(num_classes):
            if ignore_index is not None and c == ignore_index:
                continue
            
            # Create binary masks for current class
            pred_mask = (pred_batch == c).astype(bool)
            gt_mask = (gt_batch == c).astype(bool)
            
            # Skip if either mask is empty
            if not np.any(pred_mask) or not np.any(gt_mask):
                # Assign a large distance if one mask is empty and the other is not
                if np.any(pred_mask) or np.any(gt_mask):
                    results[f"hausdorff_distance_class_{c}"] += np.inf
                    results[f"hausdorff_distance_{int(percentile)}_class_{c}"] += np.inf
                continue
            
            # Compute distance transforms
            pred_dist = distance_transform_edt(~pred_mask)
            gt_dist = distance_transform_edt(~gt_mask)
            
            # Get distances from pred to gt
            pred_to_gt_distances = pred_dist[gt_mask]
            
            # Get distances from gt to pred
            gt_to_pred_distances = gt_dist[pred_mask]
            
            # Compute standard Hausdorff distance (maximum of minimum distances)
            if len(pred_to_gt_distances) > 0 and len(gt_to_pred_distances) > 0:
                hausdorff = max(pred_to_gt_distances.max(), gt_to_pred_distances.max())
                results[f"hausdorff_distance_class_{c}"] += hausdorff
                
                # Compute percentile Hausdorff distance
                all_distances = np.concatenate([pred_to_gt_distances, gt_to_pred_distances])
                percentile_hausdorff = np.percentile(all_distances, percentile)
                results[f"hausdorff_distance_{int(percentile)}_class_{c}"] += percentile_hausdorff
    
    # Average over batch
    for key in results:
        if np.isinf(results[key]):
            # Keep infinity values as they are
            continue
        results[key] /= batch_size
    
    # Calculate mean metrics across classes
    valid_classes = [c for c in range(num_classes) if ignore_index is None or c != ignore_index]
    
    if valid_classes:
        # Calculate mean Hausdorff distance
        hausdorff_values = []
        percentile_hausdorff_values = []
        
        for c in valid_classes:
            hd_val = results[f"hausdorff_distance_class_{c}"]
            phd_val = results[f"hausdorff_distance_{int(percentile)}_class_{c}"]
            
            # Skip infinite values in mean calculation
            if not np.isinf(hd_val):
                hausdorff_values.append(hd_val)
            if not np.isinf(phd_val):
                percentile_hausdorff_values.append(phd_val)
        
        if hausdorff_values:
            results["mean_hausdorff_distance"] = np.mean(hausdorff_values)
        else:
            results["mean_hausdorff_distance"] = np.inf
            
        if percentile_hausdorff_values:
            results[f"mean_hausdorff_distance_{int(percentile)}"] = np.mean(percentile_hausdorff_values)
        else:
            results[f"mean_hausdorff_distance_{int(percentile)}"] = np.inf
    
    return results