import torch
from typing import Any, Optional

def compute(
    label: Any,
    prediction: Any,
    num_classes: int,
    ignore_index: Optional[int] = None,
    eps: float = 1e-12
) -> float:
    """
    Computes the average Dice coefficient for multiclass segmentation.

    Args:
        label (Any): Ground truth mask with integer class labels in [0, num_classes-1].
                     Expected shape is [N, H, W] or [H, W] for a single image.
        prediction (Any): Either a predicted probability map with shape [N, num_classes, H, W],
                          or a hard label map with shape [N, H, W] or [H, W].
                          For a probability map, the values are assumed to be soft scores.
        num_classes (int): Total number of classes.
        ignore_index (Optional[int], optional): Class index to ignore (e.g. background). Defaults to None.
        eps (float, optional): Small constant to avoid division by zero. Defaults to 1e-12.
        
    Returns:
        float: Average Dice coefficient over the selected classes.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert inputs to tensors on the proper device.
    if not isinstance(label, torch.Tensor):
        label = torch.tensor(label, device=device)
    else:
        label = label.to(device)
    
    if not isinstance(prediction, torch.Tensor):
        prediction = torch.tensor(prediction, device=device)
    else:
        prediction = prediction.to(device)
    
    # If single image inputs (2D), add a batch dimension.
    if label.dim() == 2:
        label = label.unsqueeze(0)  # shape becomes [1, H, W]
    if prediction.dim() == 2:
        prediction = prediction.unsqueeze(0)  # shape becomes [1, H, W]
    
    # Remove any extra channel dimension of size 1 in the ground truth.
    if label.dim() == 4 and label.shape[1] == 1:
        label = label.squeeze(1)
    
    # Validate ground truth: must have integer values in [0, num_classes-1]
    if label.min() < 0 or label.max() >= num_classes:
        raise ValueError(
            f"Ground truth label values must be in [0, {num_classes-1}], got min={label.min().item()} and max={label.max().item()}"
        )
    
    # Determine whether prediction is a probability map or a hard label map.
    # For a probability map, we expect shape [N, num_classes, H, W].
    if prediction.dim() == 4:
        # Assume prediction is a probability map (soft predictions)
        pred_soft = prediction.float()
    elif prediction.dim() == 3:
        # Assume prediction is already a hard label map. One-hot encode it.
        pred_soft = torch.nn.functional.one_hot(prediction.long(), num_classes=num_classes)
        # one_hot returns shape [N, H, W, num_classes] so permute it.
        pred_soft = pred_soft.permute(0, 3, 1, 2).float()
    else:
        raise ValueError("Prediction tensor has incompatible dimensions.")
    
    # One-hot encode the ground truth labels.
    label_one_hot = torch.nn.functional.one_hot(label.long(), num_classes=num_classes)
    # Permute to shape [N, num_classes, H, W]
    label_one_hot = label_one_hot.permute(0, 3, 1, 2).float()
    
    # Compute Dice per class.
    dice_scores = []
    for c in range(num_classes):
        if ignore_index is not None and c == ignore_index:
            continue
        
        pred_c = pred_soft[:, c, :, :]
        label_c = label_one_hot[:, c, :, :]
        
        # Compute the intersection and union terms.
        intersection = (pred_c * label_c).sum()
        denominator = pred_c.sum() + label_c.sum()
        dice_c = (2 * intersection + eps) / (denominator + eps)
        dice_scores.append(dice_c)
    
    # Return the average Dice score across classes (excluding any ignored ones).
    return torch.mean(torch.stack(dice_scores)).item()


# Example usage:
if __name__ == "__main__":
    # Create a simulated ground truth label for a single image with 4 classes (0,1,2,3).
    label = torch.randint(0, 4, (256, 256))
    
    # Option 1: Use a simulated probability map with shape [N, num_classes, H, W]
    prediction_prob = torch.rand((1, 4, 256, 256))
    dice_prob = compute(label, prediction_prob, num_classes=4, ignore_index=0)
    print(f"Dice using soft predictions: {dice_prob:.4f}")
    
    # Option 2: Use a simulated hard label prediction with shape [N, H, W]
    prediction_hard = torch.randint(0, 4, (256, 256))
    dice_hard = compute(label, prediction_hard, num_classes=4, ignore_index=0)
    print(f"Dice using hard labels: {dice_hard:.4f}")
