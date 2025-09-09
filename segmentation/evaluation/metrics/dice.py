import torch
from typing import Any

def compute(label: Any, prediction: Any, threshold: float = 0.5, eps: float = 1e-12) -> float:
    """
    Computes the Dice coefficient for binary segmentation using PyTorch, with GPU support.
    
    Args:
        label (Any): Ground truth mask. Can be a numpy array or a torch.Tensor.
        prediction (Any): Predicted mask or probability map. Can be a numpy array or a torch.Tensor.
        threshold (float): Threshold to binarize predictions if they are not already binary.
        
    Returns:
        float: Dice coefficient.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert inputs to torch tensors and move to the correct device
    if not isinstance(label, torch.Tensor):
        label = torch.tensor(label, device=device)
    else:
        label = label.to(device)

    if not isinstance(prediction, torch.Tensor):
        prediction = torch.tensor(prediction, device=device)
    else:
        prediction = prediction.to(device)

    # Convert tensors to float type
    label = label.float()
    
    # Binarize predictions using threshold
    prediction_bin = (prediction >= threshold).float()

    intersection = torch.sum(label * prediction_bin)
    dice = (2.0 * intersection) / (torch.sum(label) + torch.sum(prediction_bin) + eps)

    return dice.item()

# Example usage:
if __name__ == "__main__":
    label = torch.randint(0, 2, (1, 256, 256))  # Simulated binary ground truth
    prediction = torch.rand((1, 256, 256))  # Simulated probability map
    dice_score = compute(label, prediction)
    print(f"Dice Coefficient: {dice_score:.4f}")
