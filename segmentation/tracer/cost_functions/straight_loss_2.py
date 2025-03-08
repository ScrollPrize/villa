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
        
        # register only point_b as an optimization variable, a and c are treated as fixed
        self.register_optim_vars(["point_b"])
        # register auxiliary variables
        self.register_aux_vars(["point_a", "point_c"])
        
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
        
        # For point_b (the only optimization variable): d(b - (a+c)/2)/d(b) = I
        jac_b = torch.zeros(batch_size, 3, 3, device=self.point_b.tensor.device, dtype=self.point_b.tensor.dtype)
        # For each batch, set an identity matrix
        for i in range(batch_size):
            jac_b[i] = torch.eye(3, device=self.point_b.tensor.device, dtype=self.point_b.tensor.dtype)
        
        # Calculate error
        error = self.error()
        
        # Return only the jacobian for point_b
        return [jac_b], error
    
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