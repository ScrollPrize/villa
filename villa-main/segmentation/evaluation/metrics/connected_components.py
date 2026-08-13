import numpy as np
import cc3d
from typing import Any, Optional, Dict

def compute(
    label: Any,
    prediction: Any,
    num_classes: int,
    ignore_index: Optional[int] = None,
    connectivity: int = 26
) -> Dict[str, float]:
    """
    Computes the average absolute difference in connected component counts between the ground truth and predicted masks,
    averaged over the batch, for each class and overall.

    Parameters
    ----------
    label : Any
        Ground truth segmentation mask as a NumPy array with integer labels in [0, num_classes-1].
        Expected shape: [H, W, D] for a single volume or [N, H, W, D] for a batch.
    prediction : Any
        Predicted segmentation mask. Can be either:
          - A probability map with shape [N, num_classes, H, D] (will be converted to hard labels via argmax), or
          - A hard label map with shape [H, W, D] or [N, H, W, D].
    num_classes : int
        Total number of classes.
    ignore_index : Optional[int], default None
        Class index to ignore (e.g., background) in the computation.
    connectivity : int, default 26
        Connectivity used by cc3d.connected_components

    Returns
    -------
    Dict[str, float]
        Dictionary with keys "connected_components_class_X" (for each class X not ignored) containing the average absolute difference 
        in connected component counts, and a key "connected_components_total" for the overall difference.
    """
    # If prediction is a probability map, convert to hard labels.
    if prediction.ndim == 4:
        prediction = np.argmax(prediction, axis=1)
    
    # Ensure label and prediction have a batch dimension.
    # For 3D volumes, add a new axis so the shape becomes [1, H, W, D]
    if label.ndim == 3:
        label = label[np.newaxis, ...]
    elif label.ndim != 4:
        raise ValueError("Label array must have shape [H, W, D] or [N, H, W, D].")
        
    if prediction.ndim == 3:
        prediction = prediction[np.newaxis, ...]
    elif prediction.ndim != 3 and prediction.ndim != 4:
        raise ValueError("Prediction array must have shape [H, W, D] or [N, H, W, D] after conversion.")

    batch_size = label.shape[0]
    
    # Initialize per-class difference accumulator with a consistent key naming.
    diff_per_class: Dict[str, float] = {}
    for c in range(num_classes):
        if ignore_index is not None and c == ignore_index:
            continue
        diff_per_class[f"connected_components_difference_class_{c}"] = 0.0

    total_label_cc = 0
    total_pred_cc = 0

    # Loop over images in the batch.
    for i in range(batch_size):
        for c in range(num_classes):
            if ignore_index is not None and c == ignore_index:
                continue

            # Create binary masks for class c.
            gt_mask = (label[i] == c).astype(np.uint8)
            pred_mask = (prediction[i] == c).astype(np.uint8)

            # Compute connected components using the specified connectivity.
            cc_gt = cc3d.connected_components(gt_mask, connectivity=connectivity)
            cc_pred = cc3d.connected_components(pred_mask, connectivity=connectivity)

            num_cc_gt = int(cc_gt.max())
            num_cc_pred = int(cc_pred.max())

            diff = abs(num_cc_pred - num_cc_gt)
            diff_per_class[f"connected_components_difference_class_{c}"] += diff

            total_label_cc += num_cc_gt
            total_pred_cc += num_cc_pred

    # Average differences over the batch.
    for key in diff_per_class:
        diff_per_class[key] /= batch_size

    total_diff = abs(total_pred_cc - total_label_cc) / batch_size
    diff_per_class["connected_components_difference_total"] = total_diff

    return diff_per_class

# --- Example Main: Demonstrating Integration ---
if __name__ == "__main__":
    import torch

    # Simulation parameters.
    batch_size = 1
    height, width, depth = 256, 256, 64  # 3D volumes
    num_classes = 4
    ignore_index = 0

    # Simulated ground truth: hard label map with shape [batch_size, H, W, D].
    label_tensor = torch.randint(0, num_classes, (batch_size, height, width, depth))
    label_np = label_tensor.numpy()

    # Option 1: Soft predictions.
    # Simulate a probability map with shape [batch_size, num_classes, H, W, D].
    prediction_prob_tensor = torch.rand((batch_size, num_classes, height, width, depth))
    # Convert to hard labels via argmax.
    prediction_prob_hard = prediction_prob_tensor.argmax(dim=1).numpy()

    result_soft = compute(label_np, prediction_prob_hard, num_classes, ignore_index, connectivity=26)
    print("Connected components difference (soft predictions converted to hard labels):")
    for key, value in result_soft.items():
        print(f"{key}: {value:.4f}")

    # Option 2: Hard predictions.
    # Simulate a hard label map with shape [batch_size, H, W, D].
    prediction_hard_tensor = torch.randint(0, num_classes, (batch_size, height, width, depth))
    prediction_hard_np = prediction_hard_tensor.numpy()

    result_hard = compute(label_np, prediction_hard_np, num_classes, ignore_index, connectivity=26)
    print("\nConnected components difference (hard predictions):")
    for key, value in result_hard.items():
        print(f"{key}: {value:.4f}")
