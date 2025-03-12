"""AutoDiff implementation of StraightLoss2 cost function for theseus."""

from typing import Optional, List, Tuple

import torch
import theseus as th


class StraightLoss2AutoDiff(th.AutoDiffCostFunction):
    """
    A cost function that penalizes deviations from a straight line using midpoint method.
    
    This is an autodiff implementation of StraightLoss2 using Theseus AutoDiffCostFunction.
    It computes the distance between the middle point and the average of the two end points.
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
        Initialize the StraightLoss2AutoDiff cost function.
        
        Args:
            point_a: The first 3D point (optimization variable)
            point_b: The middle 3D point (optimization variable)
            point_c: The third 3D point (optimization variable)
            cost_weight: Weight for this cost function
            name: Optional name for this cost function
        """
        self.point_a = point_a
        self.point_b = point_b
        self.point_c = point_c
        
        # Define error function for autodiff computation
        def straight_error_fn(optim_vars, aux_vars):
            """
            Error function for computing straightness-based error.
            
            Args:
                optim_vars: List of optimization variable tensors [point_a, point_b, point_c]
                aux_vars: List of auxiliary variable tensors (empty in this case)
                
            Returns:
                Tensor with shape [batch_size, 3] containing the error values
            """
            # Extract tensors from optimization variables
            # These are tensors, not Point3 objects
            a_tensor = optim_vars[0]  # Shape: [batch_size, 3]
            b_tensor = optim_vars[1]  # Shape: [batch_size, 3]
            c_tensor = optim_vars[2]  # Shape: [batch_size, 3]
            
            # Make sure we're dealing with plain tensors, not Point3 objects
            if not isinstance(a_tensor, torch.Tensor):
                a_tensor = a_tensor.tensor
            if not isinstance(b_tensor, torch.Tensor):
                b_tensor = b_tensor.tensor
            if not isinstance(c_tensor, torch.Tensor):
                c_tensor = c_tensor.tensor
                
            # Calculate midpoint between a and c
            midpoint = (a_tensor + c_tensor) * 0.5
            
            # Error is the distance vector between b and midpoint
            residual = b_tensor - midpoint
            
            return residual
        
        # Initialize the AutoDiffCostFunction with our variables and error function
        super().__init__(
            optim_vars=[point_a, point_b, point_c],
            err_fn=straight_error_fn,
            dim=3,
            cost_weight=cost_weight,
            name=name
        )
    
    def _copy_impl(self, new_name: Optional[str] = None) -> "StraightLoss2AutoDiff":
        """
        Create a copy of this cost function.
        
        Args:
            new_name: Optional new name for the copy
            
        Returns:
            A new instance of this cost function
        """
        return StraightLoss2AutoDiff(
            self.point_a.copy(),
            self.point_b.copy(),
            self.point_c.copy(),
            self.weight.copy(),
            name=new_name if new_name else self.name
        )