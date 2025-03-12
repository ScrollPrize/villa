"""AutoDiff implementation of DistLoss cost function for theseus."""

from typing import Optional, List, Tuple

import torch
import theseus as th


class DistLossAutoDiff(th.AutoDiffCostFunction):
    """
    A cost function that penalizes deviations from a target distance.
    
    This is an autodiff implementation of DistLoss using Theseus AutoDiffCostFunction.
    It automatically calculates the Jacobians using PyTorch's autograd.
    
    Important coordinate conventions:
    - All 3D points are in ZYX order [z, y, x]
    - When accessing point components, use indices: 0=z, 1=y, 2=x
    - This matches the convention used in other cost functions
    """
    def __init__(
        self,
        point_a: th.Point3,
        point_b: th.Point3,
        target_dist: float,
        cost_weight: th.CostWeight,
        name: Optional[str] = None,
    ):
        """
        Initialize the DistLossAutoDiff cost function.
        
        Args:
            point_a: The first 3D point (optimization variable)
            point_b: The second 3D point (optimization variable)
            target_dist: The target distance between the points
            cost_weight: Weight for this cost function
            name: Optional name for this cost function
        """
        self.point_a = point_a
        self.point_b = point_b
        self.target_dist = target_dist
        
        # Define error function for autodiff computation
        def dist_error_fn(optim_vars, aux_vars):
            """
            Error function for computing distance-based error.
            
            Args:
                optim_vars: List of optimization variable tensors [point_a, point_b]
                aux_vars: List of auxiliary variable tensors (empty in this case)
                
            Returns:
                Tensor with shape [batch_size, 1] containing the error values
            """
            # Extract tensors from optimization variables
            a_tensor = optim_vars[0]  # Shape: [batch_size, 3]
            b_tensor = optim_vars[1]  # Shape: [batch_size, 3]
            
            # Calculate difference vector between points
            diff = a_tensor - b_tensor
            
            # Calculate squared distance and distance
            # Make sure we're using the tensor data, not the variable itself
            diff_tensor = diff if isinstance(diff, torch.Tensor) else diff.tensor
            dist_squared = torch.sum(diff_tensor * diff_tensor, dim=1, keepdim=True)
            dist = torch.sqrt(dist_squared + 1e-10)  # Add epsilon for numerical stability
            
            # Create target distance tensor
            target_dist_tensor = torch.full_like(dist, self.target_dist)
            
            # Calculate error based on distance
            # Use a simple approach compatible with autograd
            close_mask = (dist < self.target_dist)
            zero_mask = (dist <= 1e-6)
            
            # Initialize error tensor
            error = torch.zeros_like(dist)
            
            # Compute error for different cases
            # Case 1: For zero or near-zero distances
            error = torch.where(zero_mask, dist_squared - 1.0, error)
            
            # Case 2: For distances less than target_dist (but not near zero)
            error = torch.where(close_mask & ~zero_mask, 
                                target_dist_tensor / dist - 1.0, 
                                error)
            
            # Case 3: For distances greater than or equal to target_dist
            error = torch.where(~close_mask & ~zero_mask, 
                                dist / target_dist_tensor - 1.0, 
                                error)
            
            # Handle invalid points (-1, -1, -1)
            # Make sure we're using tensor data
            a_tensor_data = a_tensor if isinstance(a_tensor, torch.Tensor) else a_tensor.tensor
            b_tensor_data = b_tensor if isinstance(b_tensor, torch.Tensor) else b_tensor.tensor
            
            a_invalid = torch.isclose(a_tensor_data, torch.full_like(a_tensor_data, -1.0)).all(dim=1, keepdim=True)
            b_invalid = torch.isclose(b_tensor_data, torch.full_like(b_tensor_data, -1.0)).all(dim=1, keepdim=True)
            invalid_mask = a_invalid | b_invalid
            
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
    
    def _copy_impl(self, new_name: Optional[str] = None) -> "DistLossAutoDiff":
        """
        Create a copy of this cost function.
        
        Args:
            new_name: Optional new name for the copy
            
        Returns:
            A new instance of this cost function
        """
        return DistLossAutoDiff(
            self.point_a.copy(),
            self.point_b.copy(),
            self.target_dist,
            self.weight.copy(),
            name=new_name if new_name else self.name
        )