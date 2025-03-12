"""StraightLoss2D cost function for theseus."""

from typing import List, Optional, Tuple

import torch
import theseus as th


class StraightLoss2D(th.CostFunction):
    """
    A cost function that penalizes deviations from a straight line in 2D.
    
    This is a reimplementation of the StraightLoss2D C++ cost function from 
    volume-cartographer. It attempts to keep three 2D points in a straight line
    by minimizing 1 - dot product of normalized direction vectors.
    
    Coordinate Convention:
    - 2D points are in YX order [y, x] with 0=y, 1=x
    - Point tensor shapes should be [batch_size, 2] where last dimension is YX
    """
    def __init__(
        self,
        point_a: th.Point2,
        point_b: th.Point2,
        point_c: th.Point2,
        cost_weight: th.CostWeight,
        name: Optional[str] = None,
    ):
        """
        Initialize the StraightLoss2D cost function.
        
        Args:
            point_a: The first 2D point (optimization variable)
            point_b: The middle 2D point (optimization variable)
            point_c: The third 2D point (optimization variable)
            cost_weight: Weight for this cost function
            name: Optional name for this cost function
        """
        super().__init__(cost_weight, name=name)
        
        self.point_a = point_a
        self.point_b = point_b
        self.point_c = point_c
        
        # register only point_b as an optimization variable, a and c are treated as fixed
        self.register_optim_vars(["point_b"])
        # register auxiliary variables
        self.register_aux_vars(["point_a", "point_c"])
        
    def error(self) -> torch.Tensor:
        """
        Compute the error between the current configuration and a straight line.
        
        Returns:
            The error tensor
        """
        batch_size = self.point_a.shape[0]
        
        # Get the current values of points
        a_tensor = self.point_a.tensor
        b_tensor = self.point_b.tensor
        c_tensor = self.point_c.tensor
        
        # Calculate direction vectors
        d1 = b_tensor - a_tensor  # b - a
        d2 = c_tensor - b_tensor  # c - b
        
        # Calculate lengths
        l1 = torch.sqrt(torch.sum(d1 * d1, dim=1, keepdim=True))
        l2 = torch.sqrt(torch.sum(d2 * d2, dim=1, keepdim=True))
        
        # Handle zero or near-zero length cases
        zero_length_mask = (l1 <= 0) | (l2 <= 0)
        
        # Prepare result tensor
        residual = torch.zeros(batch_size, 1, device=a_tensor.device, dtype=a_tensor.dtype)
        
        # Handle zero length case specially
        if torch.any(zero_length_mask):
            d1_squared = torch.sum(d1 * d1, dim=1, keepdim=True)
            d2_squared = torch.sum(d2 * d2, dim=1, keepdim=True)
            residual[zero_length_mask] = d1_squared[zero_length_mask] * d2_squared[zero_length_mask] - 1
        
        # For non-zero lengths, calculate normalized dot product
        non_zero_mask = ~zero_length_mask
        if torch.any(non_zero_mask):
            for i in range(batch_size):
                if non_zero_mask[i, 0]:
                    norm_dot = torch.sum(d1[i] * d2[i]) / (l1[i, 0] * l2[i, 0])
                    residual[i, 0] = 1.0 - torch.abs(norm_dot)
        
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
        
        # Get the current values of optimization variables
        a_tensor = self.point_a.tensor
        b_tensor = self.point_b.tensor
        c_tensor = self.point_c.tensor
        
        # Calculate direction vectors
        d1 = b_tensor - a_tensor  # b - a
        d2 = c_tensor - b_tensor  # c - b
        
        # Calculate lengths
        d1_squared = torch.sum(d1 * d1, dim=1, keepdim=True)
        d2_squared = torch.sum(d2 * d2, dim=1, keepdim=True)
        
        l1 = torch.sqrt(d1_squared)
        l2 = torch.sqrt(d2_squared)
        
        # Initialize Jacobian only for point_b which is the only optimization variable
        jac_b = torch.zeros(batch_size, 1, 2, device=b_tensor.device, dtype=b_tensor.dtype)
        
        # Handle zero or near-zero length cases
        zero_length_mask = (l1 <= 0) | (l2 <= 0)
        
        # For zero length case, compute d((d1^2 * d2^2) - 1)/d(points)
        if torch.any(zero_length_mask):
            for i in range(batch_size):
                if zero_length_mask[i, 0]:
                    # Jacobians for d(d1^2 * d2^2 - 1)/d(points)
                    d1_i = d1[i]
                    d2_i = d2[i]
                    
                    # For point_b: 2 * d2^2 * (b - a) + 2 * d1^2 * (c - b)
                    jac_b[i, 0, :] = 2 * d2_squared[i, 0] * d1_i + 2 * d1_squared[i, 0] * d2_i
        
        # For non-zero lengths, compute d(1-dot)/d(points)
        non_zero_mask = ~zero_length_mask
        if torch.any(non_zero_mask):
            # Calculate dot products individually
            dot = torch.zeros_like(l1)
            for i in range(batch_size):
                if non_zero_mask[i, 0]:
                    dot[i, 0] = torch.sum(d1[i] * d2[i])
            
            for i in range(batch_size):
                if non_zero_mask[i, 0]:
                    l1_i = l1[i, 0]
                    l2_i = l2[i, 0]
                    l1l2 = l1_i * l2_i
                    
                    if l1l2 < 1e-10:
                        # Skip if nearly zero to avoid numerical instability
                        continue
                        
                    d1_i = d1[i]
                    d2_i = d2[i]
                    
                    # For point b (affects both d1 and d2)
                    term_b1 = (d2_i / l1l2 - d1_i * dot[i, 0] / (l1_i ** 3 * l2_i))  # opposite of a
                    term_b2 = (d1_i / l1l2 - d2_i * dot[i, 0] / (l1_i * l2_i ** 3))  # opposite of c
                    jac_b[i, 0, :] = -(term_b1 + term_b2)
        
        # Calculate error
        error = self.error()
        
        # Return only the jacobian for point_b
        return [jac_b], error
    
    def _copy_impl(self, new_name: Optional[str] = None) -> "StraightLoss2D":
        """
        Create a copy of this cost function.
        
        Args:
            new_name: Optional new name for the copy
            
        Returns:
            A new instance of this cost function
        """
        return StraightLoss2D(
            self.point_a.copy(),
            self.point_b.copy(),
            self.point_c.copy(),
            self.weight.copy(),
            name=new_name if new_name else self.name
        )