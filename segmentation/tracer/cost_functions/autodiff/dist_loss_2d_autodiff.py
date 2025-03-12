"""AutoDiff implementation of DistLoss2D cost function for theseus."""

from typing import Optional, List, Tuple

import torch
import theseus as th


class DistLoss2DAutoDiff(th.AutoDiffCostFunction):
    """
    A cost function that penalizes deviations from a target distance in 2D.
    
    This is an autodiff implementation of DistLoss2D using Theseus AutoDiffCostFunction.
    It automatically calculates the Jacobians using PyTorch's autograd.
    
    Coordinate Convention:
    - 2D points are in YX order [y, x] with 0=y, 1=x
    - Point tensor shapes should be [batch_size, 2] where last dimension is YX
    """
    def __init__(
        self,
        point_a: th.Point2,
        point_b: th.Point2,
        target_dist: float,
        cost_weight: th.CostWeight,
        name: Optional[str] = None,
    ):
        """
        Initialize the DistLoss2DAutoDiff cost function.
        
        Args:
            point_a: The first 2D point (optimization variable)
            point_b: The second 2D point (optimization variable)
            target_dist: The target distance between the points
            cost_weight: Weight for this cost function
            name: Optional name for this cost function
        """
        if target_dist == 0:
            raise ValueError("target_dist can't be zero for DistLoss2DAutoDiff")
            
        self.point_a = point_a
        self.point_b = point_b
        self.target_dist = target_dist
        
        # Define error function for autodiff computation
        def dist_error_fn(optim_vars, aux_vars):
            """
            Error function for computing 2D distance-based error.
            
            Args:
                optim_vars: List of optimization variable tensors [point_a, point_b]
                aux_vars: List of auxiliary variable tensors (empty in this case)
                
            Returns:
                Tensor with shape [batch_size, 1] containing the error values
            """
            # Extract tensors from optimization variables (Point2 objects)
            a_tensor = optim_vars[0].tensor  # Now a proper tensor of shape [batch_size, 2]
            b_tensor = optim_vars[1].tensor
            
            # Handle invalid points (-1, -1)
            # Check if all elements of each point are -1 (invalid point marker)
            a_invalid = torch.all(torch.isclose(a_tensor, torch.tensor(-1.0, device=a_tensor.device)), dim=1, keepdim=True)
            b_invalid = torch.all(torch.isclose(b_tensor, torch.tensor(-1.0, device=b_tensor.device)), dim=1, keepdim=True)
            invalid_mask = a_invalid | b_invalid
            
            # Calculate difference vector between points
            diff = a_tensor - b_tensor
            
            # Calculate squared distance and distance
            dist_squared = torch.sum(diff * diff, dim=1, keepdim=True)
            dist = torch.sqrt(dist_squared + 1e-10)  # Add epsilon for numerical stability
            
            # Create target distance tensor
            target_dist_tensor = torch.full_like(dist, self.target_dist)
            
            # Calculate error based on distance
            # Initialize error tensor
            error = torch.zeros_like(dist)
            
            # Zero or near-zero distance case
            zero_dist_mask = (dist <= 1e-6)
            error = torch.where(zero_dist_mask, dist_squared - 1.0, error)
            
            # Distance less than target_dist case
            close_dist_mask = (dist < self.target_dist) & ~zero_dist_mask
            error = torch.where(close_dist_mask, 
                               target_dist_tensor / (dist + 1e-2) - 1.0, 
                               error)
            
            # Distance greater than or equal to target_dist case
            far_dist_mask = ~(zero_dist_mask | close_dist_mask)
            error = torch.where(far_dist_mask, 
                               dist / target_dist_tensor - 1.0, 
                               error)
            
            # Zero out error for invalid points
            error = torch.where(invalid_mask, torch.zeros_like(error), error)
            
            return error
        
        # Initialize the AutoDiffCostFunction with our variables and error function
        super().__init__(
            optim_vars=[point_a, point_b],
            err_fn=dist_error_fn,
            dim=1,
            cost_weight=cost_weight,
            name=name
        )
    
    def _copy_impl(self, new_name: Optional[str] = None) -> "DistLoss2DAutoDiff":
        """
        Create a copy of this cost function.
        
        Args:
            new_name: Optional new name for the copy
            
        Returns:
            A new instance of this cost function
        """
        return DistLoss2DAutoDiff(
            self.point_a.copy(),
            self.point_b.copy(),
            self.target_dist,
            self.weight.copy(),
            name=new_name if new_name else self.name
        )