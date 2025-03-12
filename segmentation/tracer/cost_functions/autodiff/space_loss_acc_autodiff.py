"""AutoDiff implementation of SpaceLossAcc cost function for theseus."""

from typing import Optional, List, Tuple

import torch
import theseus as th

from tracer.core.interpolation import TrilinearInterpolatorAutoDiff


class SpaceLossAccAutoDiff(th.AutoDiffCostFunction):
    """
    A cost function that evaluates a 3D point in a volume and returns the interpolated value.
    
    This is an autodiff implementation of SpaceLossAcc using Theseus AutoDiffCostFunction.
    It automatically calculates the Jacobians using PyTorch's autograd and then applies
    the appropriate transformation for maximize mode if needed.
    
    Special behavior:
    - With positive weight and maximize=False: Minimizes the interpolated value (finds minimum in volume)
    - With positive weight and maximize=True: Maximizes the interpolated value (finds maximum in volume)
    
    Important coordinate conventions:
    - All 3D points are in ZYX order [z, y, x]
    - When accessing point components, use indices: 0=z, 1=y, 2=x
    - For interpolator evaluation, coordinates are passed as (z, y, x)
    - This matches the convention used in other cost functions
    """
    def __init__(
        self,
        point: th.Point3,          # 3D point to evaluate
        interpolator: TrilinearInterpolatorAutoDiff,  # Volume interpolator
        cost_weight: th.CostWeight,
        maximize: bool = False,    # If True, maximize the value instead of minimizing
        name: Optional[str] = None,
    ):
        """
        Initialize the SpaceLossAccAutoDiff cost function.
        
        Args:
            point: The 3D point to evaluate (optimization variable)
            interpolator: Trilinear interpolator for the 3D volume
            cost_weight: Weight for this cost function
            maximize: If True, maximize the value instead of minimizing
            name: Optional name for this cost function
        """
        self.point = point
        self.interpolator = interpolator
        self.maximize_mode = maximize
        
        print(f"SpaceLossAccAutoDiff initialized with maximize_mode={self.maximize_mode}")
        
        # Define error function for autodiff computation
        def space_error_fn(optim_vars, aux_vars):
            """
            Error function for computing space-based error.
            
            Args:
                optim_vars: List of optimization variable tensors [point]
                aux_vars: List of auxiliary variable tensors (empty in this case)
                
            Returns:
                Tensor with shape [batch_size, 1] containing the error values
            """
            # Extract point tensor from optimization variables
            points = optim_vars[0]  # Shape: [batch_size, 3]
            
            # Create input tensors for the interpolator in ZYX order
            # Note: Points tensor is in ZYX order [z, y, x]
            z = points[:, 0:1]  # Z coordinate (first element in ZYX)
            y = points[:, 1:2]  # Y coordinate (second element in ZYX)
            x = points[:, 2:3]  # X coordinate (third element in ZYX)
            
            # Sample the volume at the point location
            values = self.interpolator.evaluate(z, y, x)
            
            # IMPORTANT: We do NOT negate values here for maximize_mode
            # This matches the C++ implementation where the residual is just the
            # raw interpolated value. The maximize behavior is implemented
            # through the negative weights rather than inverting the error.
            # In Theseus, we'll handle this by manipulating the jacobians.
            
            return values
        
        # Initialize the AutoDiffCostFunction with our variables and error function
        super().__init__(
            optim_vars=[point],
            err_fn=space_error_fn,
            dim=1,
            cost_weight=cost_weight,
            name=name
        )
        
    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Compute the Jacobians of the error with respect to the variables.
        
        Override the default autodiff jacobians method to properly handle maximize mode.
        This ensures we have consistent behavior with the manual implementation.
        
        Returns:
            Tuple containing:
                - List of Jacobian matrices (one per optimization variable)
                - Error tensor
        """
        # Get default jacobians from parent class
        jacs, err = super().jacobians()
        
        # If in maximize mode, invert the jacobians (not the error)
        if self.maximize_mode:
            jacs = [-j for j in jacs]
            
        return jacs, err
    
    def _copy_impl(self, new_name: Optional[str] = None) -> "SpaceLossAccAutoDiff":
        """
        Create a copy of this cost function.
        
        Args:
            new_name: Optional new name for the copy
            
        Returns:
            A new instance of this cost function
        """
        return SpaceLossAccAutoDiff(
            self.point.copy(),
            self.interpolator,  # Interpolator is shared, not copied
            self.weight.copy(),
            maximize=self.maximize_mode,
            name=new_name if new_name else self.name
        )