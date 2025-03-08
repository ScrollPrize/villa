"""StraightLoss cost function for theseus."""

from typing import List, Optional, Tuple

import torch
import theseus as th


class StraightLoss(th.CostFunction):
    """
    A cost function that penalizes deviations from a straight line.
    
    This is a reimplementation of the StraightLoss C++ cost function from 
    volume-cartographer. It attempts to keep three 3D points in a straight line
    by minimizing 1 - dot product of normalized direction vectors.
    """
    def __init__(
        self,
        point_a: th.Point3,
        point_b: th.Point3,
        point_c: th.Point3,
        cost_weight: th.CostWeight,
        name: Optional[str] = None,
    ):
        """
        Initialize the StraightLoss cost function.
        
        Args:
            point_a: The first 3D point (optimization variable)
            point_b: The middle 3D point (optimization variable)
            point_c: The third 3D point (optimization variable)
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
        
        # Calculate direction vectors (matching C++ implementation)
        d1 = b_tensor - a_tensor  # b - a
        d2 = c_tensor - b_tensor  # c - b
        
        # Calculate lengths (matching C++ implementation)
        l1 = torch.sqrt(torch.sum(d1 * d1, dim=1, keepdim=True))
        l2 = torch.sqrt(torch.sum(d2 * d2, dim=1, keepdim=True))
        
        # Calculate dot product
        dot_product = torch.sum(d1 * d2, dim=1, keepdim=True)
        
        # Calculate normalized dot product (matching C++ implementation)
        # Add small epsilon to avoid division by zero
        normalized_dot = dot_product / (l1 * l2 + 1e-10)
        
        # Error is 1 - abs(dot product) (0 when perfectly aligned or anti-aligned, > 0 when not aligned)
        # This handles both positive and negative alignment correctly
        residual = 1.0 - torch.abs(normalized_dot)
        
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
        
        # Dot product
        dot = torch.sum(d1 * d2, dim=1, keepdim=True)
        normalized_dot = dot / (l1 * l2 + 1e-8)
        
        # Initialize Jacobian only for point_b which is the only optimization variable
        jac_b = torch.zeros(batch_size, 1, 3, device=b_tensor.device, dtype=b_tensor.dtype)
        
        # Compute Jacobian for point_b (derived using chain rule)
        for i in range(batch_size):
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
            jac_b[i, 0, :] = -(term_b1 + term_b2)  # Negative because b appears in both terms with opposite signs
        
        # Calculate error
        error = self.error()
        
        # Return only the jacobian for point_b
        return [jac_b], error
    
    def _copy_impl(self, new_name: Optional[str] = None) -> "StraightLoss":
        """
        Create a copy of this cost function.
        
        Args:
            new_name: Optional new name for the copy
            
        Returns:
            A new instance of this cost function
        """
        return StraightLoss(
            self.point_a.copy(),
            self.point_b.copy(),
            self.point_c.copy(),
            self.weight.copy(),
            name=new_name if new_name else self.name
        )