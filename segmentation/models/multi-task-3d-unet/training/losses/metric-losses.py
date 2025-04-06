# Giorgio Angelotti, 2025

import torch
import torch.nn as nn
import torch.nn.functional as F


class MetricAndNormalLoss(nn.Module):
    r"""
    Loss that computes:
    
    1. **Metric Determinant Loss:**  
       For each voxel, two 3D vectors (u and v) form a 2×2 metric tensor:
       
         g11 = <u, u>,   g12 = <u, v>,   g22 = <v, v>.
       
       With the prediction vectors normalized, g11 = g22 = 1 so that the determinant simplifies to:
       
         |g| = |1 - g12²|.
       
       An MSE loss is applied between the determinants of the prediction and the ground truth.
       
    2. **Normal Cosine Loss:**  
       The cross product n = u x v yields a normal vector.
       A cosine similarity loss (1 – mean cosine similarity) is computed between the predicted and target normals.
       
    The final loss is a weighted sum of these two components.
    
    Args:
        weight_metric (float): Weight for the metric determinant MSE loss.
        weight_normal (float): Weight for the normal cosine similarity loss.
        normalize_predictions (bool): If True, normalizes the prediction vectors (u and v)
                                      to unit length before further computations.
    """
    def __init__(self, weight_metric: float = 1.0, weight_normal: float = 10.0, 
                 normalize_predictions: bool = True):
        super(MetricAndNormalLoss, self).__init__()
        self.weight_metric = weight_metric
        self.weight_normal = weight_normal
        self.normalize_predictions = normalize_predictions

    def forward(self, pred: torch.Tensor, target: torch.Tensor, return_breakdown: bool = False):
        """
        Args:
            pred (torch.Tensor): Predicted tensor of shape [B, 6, Z, Y, X].
            target (torch.Tensor): Ground truth tensor of shape [B, 6, Z, Y, X].
            return_breakdown (bool): If True, returns a tuple (total_loss, breakdown_dict)
                                     where breakdown_dict contains individual loss components.
        
        Returns:
            torch.Tensor or tuple: Total loss if return_breakdown is False, otherwise
                                   a tuple (total_loss, breakdown_dict).
        """
        if pred.shape[1] != 6 or target.shape[1] != 6:
            raise ValueError("Both pred and target must have 6 channels (two 3D vectors).")
        
        B, _, Z, Y, X = pred.shape

        # Reshape into [B, 2, 3, Z, Y, X] where index 0: u and index 1: v
        pred = pred.view(B, 2, 3, Z, Y, X)
        target = target.view(B, 2, 3, Z, Y, X)
        
        # Extract u and v for both prediction and target
        u_pred, v_pred = pred[:, 0], pred[:, 1]   # shape: [B, 3, Z, Y, X]
        u_targ, v_targ = target[:, 0], target[:, 1]

        # Optionally normalize the predicted vectors along the channel dimension
        if self.normalize_predictions:
            u_pred = F.normalize(u_pred, dim=1, eps=1e-8)
            v_pred = F.normalize(v_pred, dim=1, eps=1e-8)

        total_loss = 0.0
        loss_metric = torch.tensor(0.0, device=pred.device)
        loss_normal = torch.tensor(0.0, device=pred.device)

        # ----- Metric Determinant Loss -----
        if self.weight_metric:
            # Compute the dot product u.v for prediction and target in batch
            g12_pred = (u_pred * v_pred).sum(dim=1)  # shape: [B, Z, Y, X]
            g12_targ = (u_targ * v_targ).sum(dim=1)
            
            # With normalized vectors: det = |1 - (u.v)^2|
            det_pred = torch.abs(1 - g12_pred**2)
            det_targ = torch.abs(1 - g12_targ**2)
            
            loss_metric = F.mse_loss(det_pred, det_targ)
            total_loss = total_loss + self.weight_metric * loss_metric

        # ----- Normal Cosine Loss -----
        if self.weight_normal:
            # Stack the corresponding u and v tensors to compute cross products in one operation.
            # The stacked shape is [B, 2, 3, Z, Y, X] where index 0 is for pred and index 1 for targ.
            us = torch.stack([u_pred, u_targ], dim=1)
            vs = torch.stack([v_pred, v_targ], dim=1)
            # Compute cross products along the vector dimension (dim=2)
            normals = torch.cross(us, vs, dim=2)
            normal_pred, normal_targ = normals[:, 0], normals[:, 1]
            
            # Compute cosine similarity along the channel dimension (3)
            cos_sim = F.cosine_similarity(normal_pred, normal_targ, dim=1, eps=1e-8)
            loss_normal = 1.0 - cos_sim.mean()
            total_loss = total_loss + self.weight_normal * loss_normal

        if return_breakdown:
            return total_loss, {"weighted_loss_metric": self.weight_metric * loss_metric, "weighted_loss_normal": self.weight_normal * loss_normal}
        else:
            return total_loss


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Define tensor dimensions: Batch size B, and spatial dimensions Z, Y, X
    B, Z, Y, X = 2, 4, 4, 4
    
    # Create random tensors for predictions and ground truth.
    # Each tensor has 6 channels (two 3D vectors: u and v)
    pred = torch.randn(B, 6, Z, Y, X, requires_grad=True)
    target = torch.randn(B, 6, Z, Y, X)
    
    # Instantiate the loss module.
    loss_fn = MetricAndNormalLoss(weight_metric=1.0, weight_normal=10.0, normalize_predictions=True)
    
    # Compute the loss with breakdown of individual terms.
    total_loss, loss_breakdown = loss_fn(pred, target, return_breakdown=True)
    
    print("Total Loss value:", total_loss.item())
    for term, value in loss_breakdown.items():
        print(f"{term}: {value.item()}")
    
    # Backward pass to test gradient computation.
    total_loss.backward()
    print("Gradient norm for predictions:", pred.grad.norm().item())
