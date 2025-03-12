"""AutoDiff implementation of ZCoordLoss cost function for theseus."""

from typing import Optional, List, Tuple

import torch
import theseus as th


class ZCoordLossAutoDiff(th.AutoDiffCostFunction):
    """
    A cost function that penalizes deviations from a target Z coordinate.
    
    This is an autodiff implementation of ZCoordLoss using Theseus AutoDiffCostFunction.
    It automatically calculates the Jacobians using PyTorch's autograd.
    
    Important coordinate conventions:
    - All 3D points are in ZYX order [z, y, x]
    - When accessing point components, use indices: 0=z, 1=y, 2=x
    - This matches the convention used in other cost functions
    """
    def __init__(
        self,
        point: th.Point3,
        target_z: th.Vector,
        cost_weight: th.CostWeight,
        name: Optional[str] = None,
    ):
        """
        Initialize the ZCoordLossAutoDiff cost function.
        
        Args:
            point: The 3D point to constrain (optimization variable)
            target_z: Target Z coordinate as a 1-dimensional vector
            cost_weight: Weight for this cost function
            name: Optional name for this cost function
        """
        # Check that target_z is a 1D vector
        if target_z.dof() != 1:
            raise ValueError("target_z must be a 1-dimensional vector")
        
        self.point = point
        self.target_z = target_z
        
        # Define error function for autodiff computation
        def z_coord_error_fn(optim_vars, aux_vars):
            """
            Error function for computing z-coordinate based error.
            
            Args:
                optim_vars: List of optimization variable tensors [point]
                aux_vars: List of auxiliary variable tensors [target_z]
                
            Returns:
                Tensor with shape [batch_size, 1] containing the error values
            """
            # Extract tensors from optimization and auxiliary variables
            point_tensor = optim_vars[0]  # Shape: [batch_size, 3]
            target_z_tensor = aux_vars[0]  # Shape: [batch_size, 1]
            
            # Get the z-coordinate (1st element, index 0) in ZYX ordering
            z = point_tensor[:, 0:1]  # Shape: [batch_size, 1]
            
            # Calculate difference with target z
            # Make sure we're using the tensor data, not the variable itself
            target_z_data = target_z_tensor if isinstance(target_z_tensor, torch.Tensor) else target_z_tensor.tensor
            residual = z - target_z_data
            
            return residual
        
        # Initialize the AutoDiffCostFunction with our variables and error function
        super().__init__(
            optim_vars=[point],
            aux_vars=[target_z],
            err_fn=z_coord_error_fn,
            dim=1,
            cost_weight=cost_weight,
            name=name
        )
    
    def _copy_impl(self, new_name: Optional[str] = None) -> "ZCoordLossAutoDiff":
        """
        Create a copy of this cost function.
        
        Args:
            new_name: Optional new name for the copy
            
        Returns:
            A new instance of this cost function
        """
        return ZCoordLossAutoDiff(
            self.point.copy(),
            self.target_z.copy(),
            self.weight.copy(),
            name=new_name if new_name else self.name
        )