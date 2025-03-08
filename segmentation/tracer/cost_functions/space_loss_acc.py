"""SpaceLossAcc cost function for theseus."""

from typing import List, Optional, Tuple

import torch
import theseus as th

from .trilinear_interpolator import TrilinearInterpolator


class SpaceLossAcc(th.CostFunction):
    """
    A cost function that evaluates a 3D point in a volume and returns the interpolated value.
    
    This is a reimplementation of the SpaceLossAcc C++ cost function from 
    volume-cartographer. It samples a 3D volume at the specified point location
    and uses the interpolated value as the error.
    
    Special behavior:
    - With positive weight: Minimizes the interpolated value (finds minimum in volume)
    - With negative weight: Maximizes the interpolated value (finds maximum in volume)
    """
    def __init__(
        self,
        point: th.Point3,          # 3D point to evaluate
        interpolator: TrilinearInterpolator,  # Volume interpolator
        cost_weight: th.CostWeight,
        maximize: bool = False,    # If True, maximize the value instead of minimizing
        name: Optional[str] = None,
    ):
        """
        Initialize the SpaceLossAcc cost function.
        
        Args:
            point: The 3D point to evaluate (optimization variable)
            interpolator: Trilinear interpolator for the 3D volume
            cost_weight: Weight for this cost function
            maximize: If True, maximize the value instead of minimizing
            name: Optional name for this cost function
        """
        super().__init__(cost_weight, name=name)
        
        self.point = point
        self.interpolator = interpolator
        self.maximize_mode = maximize
        
        print(f"SpaceLossAcc initialized with maximize_mode={self.maximize_mode}")
        
        # Register point as an optimization variable
        self.register_optim_vars(["point"])
        
    def error(self) -> torch.Tensor:
        """
        Compute the error by sampling the volume at the point location.
        
        Returns:
            The error tensor (sampled volume value)
        """
        batch_size = self.point.shape[0]
        
        # Get point coordinates - shape (batch_size, 3)
        points = self.point.tensor
        
        # Create input tensors for the interpolator
        z = points[:, 2:3]  # Shape (batch_size, 1)
        y = points[:, 1:2]  # Shape (batch_size, 1)
        x = points[:, 0:1]  # Shape (batch_size, 1) - Fix: this was incorrectly using entire point tensor
        
        # Sample the volume at the point location
        values = self.interpolator.evaluate(z, y, x)
        
        # Return the sampled values as the error
        return values
        
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
        
        This computes the gradient of the sampled volume with respect to the point coordinates.
        
        Special behavior:
        - With positive weight: Gradients point towards lower values
        - With negative weight: Gradients point towards higher values
        
        Returns:
            A tuple containing:
                - A list of Jacobian matrices, one for each optimization variable
                - The error tensor
        """
        batch_size = self.point.shape[0]
        
        # Get point coordinates - shape (batch_size, 3)
        points = self.point.tensor
        
        # Create input tensors for the interpolator
        z = points[:, 2:3]  # Shape (batch_size, 1)
        y = points[:, 1:2]  # Shape (batch_size, 1)
        x = points[:, 0:1]  # Shape (batch_size, 1) - Fix: this was incorrectly using entire point tensor
        
        # Sample the volume and compute gradients at the point location
        values, gradients = self.interpolator.evaluate_with_gradient(z, y, x)
        
        # Reshape gradients to (batch_size, 1, 3) for Jacobian format
        jac = gradients.reshape(batch_size, 1, 3)
        
        # If in maximize mode, invert gradient direction to climb uphill
        if self.maximize_mode:
            jac = -jac
        
        # Return jacobian and error
        return [jac], values
    
    def _copy_impl(self, new_name: Optional[str] = None) -> "SpaceLossAcc":
        """
        Create a copy of this cost function.
        
        Args:
            new_name: Optional new name for the copy
            
        Returns:
            A new instance of this cost function
        """
        return SpaceLossAcc(
            self.point.copy(),
            self.interpolator,  # Interpolator is shared, not copied
            self.weight.copy(),
            maximize=self.maximize_mode,
            name=new_name if new_name else self.name
        )