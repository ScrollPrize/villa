"""AutoDiff implementation of LinChkDistLoss cost function for theseus."""

from typing import Optional, List, Tuple

import torch
import theseus as th


class LinChkDistLossAutoDiff(th.AutoDiffCostFunction):
    """
    A cost function that penalizes deviations from a target 2D location.
    
    This is an autodiff implementation of LinChkDistLoss using Theseus AutoDiffCostFunction.
    It computes the square root of the absolute difference between a 2D point and a target point,
    but only for positive differences.
    """
    def __init__(
        self,
        point: th.Point2,
        target: th.Point2,
        cost_weight: th.CostWeight,
        name: Optional[str] = None,
    ):
        """
        Initialize the LinChkDistLossAutoDiff cost function.
        
        Args:
            point: The 2D point (optimization variable)
            target: The target 2D point (auxiliary variable)
            cost_weight: Weight for this cost function
            name: Optional name for this cost function
        """
        self.point = point
        self.target = target
        
        # Define error function for autodiff computation
        def lin_chk_error_fn(optim_vars, aux_vars):
            """
            Error function for computing distance-based error.
            
            Args:
                optim_vars: List of optimization variable tensors [point]
                aux_vars: List of auxiliary variable tensors [target]
                
            Returns:
                Tensor with shape [batch_size, 2] containing the error values
            """
            # Extract tensors from variables
            point_tensor = optim_vars[0]  # Shape: [batch_size, 2]
            target_tensor = aux_vars[0]   # Shape: [batch_size, 2]
            
            # Make sure we're dealing with plain tensors, not Point2 objects
            if not isinstance(point_tensor, torch.Tensor):
                point_tensor = point_tensor.tensor
            if not isinstance(target_tensor, torch.Tensor):
                target_tensor = target_tensor.tensor
                
            # Calculate absolute differences
            a = torch.abs(point_tensor[:, 0:1] - target_tensor[:, 0:1])  # x component
            b = torch.abs(point_tensor[:, 1:2] - target_tensor[:, 1:2])  # y component
            
            # Initialize residuals
            batch_size = point_tensor.shape[0]
            residual = torch.zeros((batch_size, 2), device=point_tensor.device, dtype=point_tensor.dtype)
            
            # Use a differentiable approach to implement the conditional logic
            # Only apply sqrt when the absolute difference is positive
            mask_a = (a > 0).float()
            mask_b = (b > 0).float()
            
            # Use the mask to selectively apply sqrt
            # For where a > 0, use sqrt(a), otherwise use 0
            residual_a = mask_a * torch.sqrt(a + 1e-10)  # Add small epsilon for numerical stability
            residual_b = mask_b * torch.sqrt(b + 1e-10)
            
            # Combine into final residual
            residual = torch.cat([residual_a, residual_b], dim=1)
            
            return residual
        
        # Initialize the AutoDiffCostFunction with our variables and error function
        super().__init__(
            optim_vars=[point],
            aux_vars=[target],
            err_fn=lin_chk_error_fn,
            dim=2,
            cost_weight=cost_weight,
            name=name
        )
    
    def _copy_impl(self, new_name: Optional[str] = None) -> "LinChkDistLossAutoDiff":
        """
        Create a copy of this cost function.
        
        Args:
            new_name: Optional new name for the copy
            
        Returns:
            A new instance of this cost function
        """
        return LinChkDistLossAutoDiff(
            self.point.copy(),
            self.target.copy(),
            self.weight.copy(),
            name=new_name if new_name else self.name
        )