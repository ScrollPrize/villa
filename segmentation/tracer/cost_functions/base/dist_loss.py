"""DistLoss cost function for theseus."""

from typing import List, Optional, Tuple

import torch
import theseus as th


class DistLoss(th.CostFunction):
    """
    A cost function that penalizes deviations from a target distance.
    
    This is a reimplementation of the DistLoss C++ cost function from 
    volume-cartographer. It penalizes the distance between two 3D points.
    
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
        Initialize the DistLoss cost function.
        
        Args:
            point_a: The first 3D point (optimization variable)
            point_b: The second 3D point (optimization variable)
            target_dist: The target distance between the points
            cost_weight: Weight for this cost function
            name: Optional name for this cost function
        """
        super().__init__(cost_weight, name=name)
        
        self.point_a = point_a
        self.point_b = point_b
        self.target_dist = target_dist
        
        # register optimization variables
        self.register_optim_vars(["point_a", "point_b"])
        
    def error(self) -> torch.Tensor:
        """
        Compute the error between the current distance and the target distance.
        
        Returns:
            The error tensor
        """
        batch_size = self.point_a.shape[0]
        
        # Handle invalid points (-1, -1, -1)
        a_tensor = self.point_a.tensor
        b_tensor = self.point_b.tensor
        
        a_invalid = torch.all(a_tensor == -1, dim=1, keepdim=True)
        b_invalid = torch.all(b_tensor == -1, dim=1, keepdim=True)
        
        # Calculate distance between points
        diff = a_tensor - b_tensor
        dist_squared = torch.sum(diff * diff, dim=1, keepdim=True)
        dist = torch.sqrt(dist_squared)
        
        # Calculate residual based on distance value
        # If dist <= 0, residual = w * (dist_squared - 1)
        # If dist < target_dist, residual = w * (target_dist/dist - 1)
        # Otherwise, residual = w * (dist/target_dist - 1)
        
        zero_dist_mask = (dist <= 0)
        close_dist_mask = (dist < self.target_dist) & ~zero_dist_mask
        far_dist_mask = ~(zero_dist_mask | close_dist_mask)
        
        residual = torch.zeros_like(dist)
        target_dist_tensor = torch.ones_like(dist) * self.target_dist
        
        # Handle different distance cases
        residual[zero_dist_mask] = dist_squared[zero_dist_mask] - 1
        residual[close_dist_mask] = target_dist_tensor[close_dist_mask] / dist[close_dist_mask] - 1
        residual[far_dist_mask] = dist[far_dist_mask] / target_dist_tensor[far_dist_mask] - 1
        
        # Mask out invalid points
        invalid_mask = a_invalid | b_invalid
        residual[invalid_mask] = 0.0
        
        return residual
        
    def dim(self) -> int:
        """
        Return the dimension of the error.
        
        Returns:
            The error dimension (1)
        """
        return 1
    
    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Compute the Jacobians of the error with respect to the variables.
        
        Returns:
            A tuple containing:
                - A list of Jacobian matrices, one for each optimization variable
                - The error tensor
        """
        batch_size = self.point_a.shape[0]
        
        # Get the current values of the optimization variables
        a_tensor = self.point_a.tensor
        b_tensor = self.point_b.tensor
        
        # Handle invalid points
        a_invalid = torch.all(a_tensor == -1, dim=1, keepdim=True)
        b_invalid = torch.all(b_tensor == -1, dim=1, keepdim=True)
        invalid_mask = a_invalid | b_invalid
        
        # Calculate differences and distances
        diff = a_tensor - b_tensor
        dist_squared = torch.sum(diff * diff, dim=1, keepdim=True)
        dist = torch.sqrt(dist_squared)
        
        # Initialize Jacobians for point_a and point_b
        jac_a = torch.zeros(batch_size, 1, 3, device=a_tensor.device, dtype=a_tensor.dtype)
        jac_b = torch.zeros(batch_size, 1, 3, device=b_tensor.device, dtype=b_tensor.dtype)
        
        # Different cases based on distance
        zero_dist_mask = (dist <= 0) & ~invalid_mask
        close_dist_mask = (dist < self.target_dist) & ~zero_dist_mask & ~invalid_mask
        # Points at exactly the target distance should have zero gradient
        at_target_mask = torch.isclose(dist, torch.tensor(self.target_dist, device=dist.device), atol=1e-5) & ~invalid_mask
        far_dist_mask = ~(zero_dist_mask | close_dist_mask | at_target_mask | invalid_mask)
        
        # For near-zero distance case
        if torch.any(zero_dist_mask):
            # d(diff^2)/d(point_a) = 2*diff
            for i in range(batch_size):
                if zero_dist_mask[i, 0]:
                    jac_a[i, 0, :] = 2 * diff[i]
                    jac_b[i, 0, :] = -2 * diff[i]
        
        # For distance < target_dist case
        if torch.any(close_dist_mask):
            for i in range(batch_size):
                if close_dist_mask[i, 0]:
                    # For close distances, we want points to move apart
                    # diff = a - b, so to move a away from b, we move in direction of -diff
                    # This matches the C++ implementation's sign direction
                    scale_close = self.target_dist / (dist[i, 0] ** 3)
                    jac_a[i, 0, :] = -scale_close * diff[i]  # Move point_a away from point_b
                    jac_b[i, 0, :] = scale_close * diff[i]   # Move point_b away from point_a
        
        # For distance >= target_dist case
        if torch.any(far_dist_mask):
            for i in range(batch_size):
                if far_dist_mask[i, 0]:
                    # For far distances, we want points to move closer
                    # diff = a - b, so to move a toward b, we move in direction of diff
                    # This matches the C++ implementation's sign direction
                    scale_far = 1.0 / (self.target_dist * dist[i, 0])
                    jac_a[i, 0, :] = scale_far * diff[i]  # Move point_a toward point_b
                    jac_b[i, 0, :] = -scale_far * diff[i] # Move point_b toward point_a
        
        # Calculate error
        error = self.error()
        
        return [jac_a, jac_b], error
    
    def _copy_impl(self, new_name: Optional[str] = None) -> "DistLoss":
        """
        Create a copy of this cost function.
        
        Args:
            new_name: Optional new name for the copy
            
        Returns:
            A new instance of this cost function
        """
        return DistLoss(
            self.point_a.copy(),
            self.point_b.copy(),
            self.target_dist,
            self.weight.copy(),
            name=new_name if new_name else self.name
        )


