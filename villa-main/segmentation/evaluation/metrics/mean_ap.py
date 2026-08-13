import numpy as np
from typing import Any, Dict, Union
from sklearn.metrics import average_precision_score

def compute(label: Any, prediction: Any, ignore_index: int = 0, average: str = 'macro',
           return_all: bool = True) -> Dict[str, float]:
    """
    Computes the Mean Average Precision for segmentation.
    
    Args:
        label (Any): Ground truth mask as a numpy array.
        prediction (Any): Predicted mask or probability map as a numpy array.
        ignore_index (int): Index to ignore (typically background class).
        average (str): Method for averaging: 'micro', 'macro', 'weighted', 'samples'.
        return_all (bool): Whether to return individual class AP values.
        
    Returns:
        Dict[str, float]: Dictionary containing mean_ap and individual class AP values.
    """
    # Ensure inputs are numpy arrays
    label_np = np.asarray(label)
    pred_np = np.asarray(prediction)
    
    # Save original shapes for reshaping later if needed
    orig_label_shape = label_np.shape
    
    # Flatten arrays for processing
    label_np = label_np.flatten()
    
    # Get unique classes (excluding background/ignore class)
    unique_classes = np.unique(label_np)
    unique_classes = unique_classes[unique_classes != ignore_index]
    
    if len(unique_classes) == 0:
        # No positive classes found
        return {"mean_ap": 0.0}
    
    result = {}
    aps = []
    
    # Handle different prediction formats
    is_multiclass_prediction = len(pred_np.shape) > len(orig_label_shape)
    
    # If prediction is already flattened, reshape it appropriately
    if is_multiclass_prediction and pred_np.ndim == 1:
        # Reshape prediction to have classes as first dimension
        num_classes = len(unique_classes) + (1 if ignore_index in np.unique(label_np) else 0)
        pred_np = pred_np.reshape(num_classes, -1)
    
    # For each class, compute AP
    for cls in unique_classes:
        y_true = (label_np == cls).astype(np.int32)
        
        if np.sum(y_true) == 0:
            # Skip classes with no positive samples
            continue
            
        # Get prediction values for this class
        if is_multiclass_prediction:
            # For multi-class prediction with class channels
            cls_idx = int(np.where(unique_classes == cls)[0][0])
            if cls_idx < pred_np.shape[0]:
                if pred_np.ndim > 1:
                    pred_cls = pred_np[cls_idx].flatten()
                else:
                    # If predictions are already flattened
                    pred_cls = pred_np
            else:
                # Skip if class index is out of bounds
                continue
        else:
            # For single-channel prediction
            # If prediction is binary or multiclass without class channels
            # we use the raw prediction values
            if pred_np.ndim > 1:
                pred_cls = pred_np.flatten()
            else:
                pred_cls = pred_np
            
            # For binary case with only two classes
            if len(unique_classes) == 1 and cls != 1:
                # Invert predictions for the negative class
                pred_cls = 1 - pred_cls
                
        # Compute AP for this class
        try:
            ap = average_precision_score(y_true, pred_cls)
            aps.append(ap)
            result[f"ap_class_{cls}"] = float(ap)
        except Exception as e:
            print(f"Error computing AP for class {cls}: {e}")
            result[f"ap_class_{cls}"] = 0.0
    
    # Compute mean AP across all classes
    if aps:
        result["mean_ap"] = float(np.mean(aps))
    else:
        result["mean_ap"] = 0.0
    
    return result