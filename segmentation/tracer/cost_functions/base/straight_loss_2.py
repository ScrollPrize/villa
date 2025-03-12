"""StraightLoss2 cost function for theseus."""

from typing import List, Optional, Tuple

import torch
import theseus as th


class StraightLoss2(th.CostFunction):
    """
    A cost function that penalizes deviations from a straight line using midpoint method.
    
    This is a reimplementation of the StraightLoss2 C++ cost function from 
    volume-cartographer. It computes the distance between the middle point and 
    the average of the two end points.
    
    Coordinate Convention:
    - 3D points are in ZYX order [z, y, x] with 0=z, 1=y, 2=x
    - Point tensor shapes should be [batch_size, 3] where last dimension is ZYX
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
        Initialize the StraightLoss2 cost function.
        
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
        
        # All points are optimization variables, allowing all points to move
        self.register_optim_vars(["point_a", "point_b", "point_c"])
        
    def error(self) -> torch.Tensor:
        """
        Compute the error between the middle point and the average of endpoints.
        
        Returns:
            The error tensor
        """
        # Get point tensors
        a_tensor = self.point_a.tensor
        b_tensor = self.point_b.tensor
        c_tensor = self.point_c.tensor
        
        # Calculate midpoint between a and c
        midpoint = (a_tensor + c_tensor) * 0.5
        
        # Error is the distance vector between b and midpoint
        residual = b_tensor - midpoint
        
        return residual
        
    def dim(self) -> int:
        """
        Return the dimension of the error.
        
        Returns:
            The error dimension (3)
        """
        return 3
    
    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Compute the Jacobians of the error with respect to the variables.
        
        Returns:
            A tuple containing:
                - A list of Jacobian matrices, one for each optimization variable
                - The error tensor
        """
        batch_size = self.point_a.shape[0]
        device = self.point_a.tensor.device
        dtype = self.point_a.tensor.dtype
        
        # For point_a (first point): d(b - (a+c)/2)/d(a) = -1/2 * I
        jac_a = torch.zeros(batch_size, 3, 3, device=device, dtype=dtype)
        # For each batch, set a scaled identity matrix
        for i in range(batch_size):
            jac_a[i] = -0.5 * torch.eye(3, device=device, dtype=dtype)
        
        # For point_b (middle point): d(b - (a+c)/2)/d(b) = I
        jac_b = torch.zeros(batch_size, 3, 3, device=device, dtype=dtype)
        # For each batch, set an identity matrix
        for i in range(batch_size):
            jac_b[i] = torch.eye(3, device=device, dtype=dtype)
        
        # For point_c (third point): d(b - (a+c)/2)/d(c) = -1/2 * I
        jac_c = torch.zeros(batch_size, 3, 3, device=device, dtype=dtype)
        # For each batch, set a scaled identity matrix
        for i in range(batch_size):
            jac_c[i] = -0.5 * torch.eye(3, device=device, dtype=dtype)
        
        # Calculate error
        error = self.error()
        
        # Return jacobians for all three points
        return [jac_a, jac_b, jac_c], error
    
    def _copy_impl(self, new_name: Optional[str] = None) -> "StraightLoss2":
        """
        Create a copy of this cost function.
        
        Args:
            new_name: Optional new name for the copy
            
        Returns:
            A new instance of this cost function
        """
        return StraightLoss2(
            self.point_a.copy(),
            self.point_b.copy(),
            self.point_c.copy(),
            self.weight.copy(),
            name=new_name if new_name else self.name
        )