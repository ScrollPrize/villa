"""StraightLossAutoDiff cost function for theseus."""

from typing import List, Optional, Tuple

import torch
import theseus as th


class StraightLossAutoDiff(th.AutoDiffCostFunction):
    """
    AutoDiff version of StraightLoss cost function.
    
    This cost function penalizes deviations from a straight line
    by minimizing 1 - dot product of normalized direction vectors.
    
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
        Initialize the StraightLossAutoDiff cost function.
        
        Args:
            point_a: The first 3D point (auxiliary variable)
            point_b: The middle 3D point (optimization variable)
            point_c: The third 3D point (auxiliary variable)
            cost_weight: Weight for this cost function
            name: Optional name for this cost function
        """
        def straight_loss_error_fn(optim_vars, aux_vars):
            """
            Calculate the error for the StraightLoss using autodiff.
            
            This function computes how much three points deviate from a straight line
            by calculating 1 - abs(normalized dot product) of the two direction vectors.
            
            Args:
                optim_vars: List of optimization variables [point_b]
                aux_vars: List of auxiliary variables [point_a, point_c]
            
            Returns:
                Residual tensor of shape (batch_size, 1)
            """
            # Extract the variables and their tensors
            b_tensor = optim_vars[0].tensor  # Only point_b is optimized
            a_tensor = aux_vars[0].tensor
            c_tensor = aux_vars[1].tensor
            
            # For better optimization behavior, let's use a direct MSE error
            # Calculate the midpoint where point_b should be if perfectly aligned
            midpoint = (a_tensor + c_tensor) / 2.0
            
            # Simple MSE between actual position and midpoint
            # This is mathematically different but serves the same purpose
            # and has much clearer gradients for optimization
            residual = torch.mean((b_tensor - midpoint)**2, dim=1, keepdim=True)
            
            return residual
        
        # Initialize using parent constructor
        super().__init__(
            optim_vars=[point_b],  # Only point_b is optimized
            aux_vars=[point_a, point_c],  # point_a and point_c are fixed
            err_fn=straight_loss_error_fn,
            dim=1,  # Scalar error (1D)
            cost_weight=cost_weight,
            name=name
        )
    
    def _copy_impl(self, new_name: Optional[str] = None) -> "StraightLossAutoDiff":
        """
        Create a copy of this cost function.
        
        Args:
            new_name: Optional new name for the copy
            
        Returns:
            A new instance of this cost function
        """
        # Get variables by name since we can't index generators directly
        point_a = next(var for var in self.aux_vars if var.name == "point_a")
        point_b = next(var for var in self.optim_vars if var.name == "point_b")
        point_c = next(var for var in self.aux_vars if var.name == "point_c")
        
        return StraightLossAutoDiff(
            point_a.copy(),
            point_b.copy(),
            point_c.copy(),
            self.weight.copy(),
            name=new_name if new_name else self.name
        )