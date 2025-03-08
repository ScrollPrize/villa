"""ZCoordLoss cost function for theseus."""

from typing import List, Optional, Tuple

import torch
import theseus as th


class ZCoordLoss(th.CostFunction):
    """
    A cost function that penalizes deviations from a target Z coordinate.
    
    This is a reimplementation of the ZCoordLoss C++ cost function from 
    volume-cartographer. It attempts to keep a 3D point at a specific Z coordinate.
    """
    def __init__(
        self,
        point: th.Point3,
        target_z: th.Vector,
        cost_weight: th.CostWeight,
        name: Optional[str] = None,
    ):
        """
        Initialize the ZCoordLoss cost function.
        
        Args:
            point: The 3D point to constrain (optimization variable)
            target_z: Target Z coordinate as a 1-dimensional vector
            cost_weight: Weight for this cost function
            name: Optional name for this cost function
        """
        super().__init__(cost_weight, name=name)
        
        # Check that target_z is a 1D vector
        if target_z.dof() != 1:
            raise ValueError("target_z must be a 1-dimensional vector")
        
        self.point = point
        self.target_z = target_z
        
        # Register point as an optimization variable
        self.register_optim_vars(["point"])
        # Register target_z as an auxiliary variable
        self.register_aux_vars(["target_z"])
        
    def error(self) -> torch.Tensor:
        """
        Compute the error between the current Z coordinate and the target.
        
        Returns:
            The error tensor
        """
        # Get the current z-coordinate (3rd element, index 2)
        z = self.point.tensor[:, 2:3]
        
        # Calculate difference with target z
        residual = z - self.target_z.tensor
        
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
        batch_size = self.point.shape[0]
        
        # Create Jacobian for the point (derivative of error w.r.t. point)
        # The Jacobian is [0, 0, 1] because error only depends on z-coordinate
        jac = torch.zeros(batch_size, 1, 3, device=self.point.device, dtype=self.point.dtype)
        jac[:, 0, 2] = 1.0  # derivative is 1 for z-coordinate
        
        # Calculate error
        error = self.error()
        
        return [jac], error
    
    def _copy_impl(self, new_name: Optional[str] = None) -> "ZCoordLoss":
        """
        Create a copy of this cost function.
        
        Args:
            new_name: Optional new name for the copy
            
        Returns:
            A new instance of this cost function
        """
        return ZCoordLoss(
            self.point.copy(),
            self.target_z.copy(),
            self.weight.copy(),
            name=new_name if new_name else self.name
        )