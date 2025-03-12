"""SpaceLossAcc cost function for theseus."""

from typing import List, Optional, Tuple
import os

import torch
import theseus as th

from tracer.core.interpolation import TrilinearInterpolator

# Check if gradient debugging is enabled via environment variable
GRADIENT_DEBUG_ENABLED = os.environ.get('GRADIENT_DEBUG', '0').lower() in ('1', 'true', 'yes', 'on')


class SpaceLossAcc(th.CostFunction):
    """
    A cost function that evaluates a 3D point in a volume and returns the interpolated value.
    
    This is a reimplementation of the SpaceLossAcc C++ cost function from 
    volume-cartographer. It samples a 3D volume at the specified point location
    and uses the interpolated value as the error.
    
    Special behavior:
    - With positive weight: Minimizes the interpolated value (finds minimum in volume)
    - With negative weight: Maximizes the interpolated value (finds maximum in volume)
    
    Important coordinate conventions:
    - All 3D points are in ZYX order [z, y, x]
    - When accessing point components, use indices: 0=z, 1=y, 2=x
    - For interpolator evaluation, coordinates are passed as (z, y, x)
    - This matches the convention used in other cost functions
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
        
        # Only print initialization message if debug is enabled
        if GRADIENT_DEBUG_ENABLED:
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
        
        # Create input tensors for the interpolator in ZYX order
        # Note: Points tensor is in ZYX order [z, y, x]
        z = points[:, 0:1]  # Z coordinate (first element in ZYX)
        y = points[:, 1:2]  # Y coordinate (second element in ZYX)
        x = points[:, 2:3]  # X coordinate (third element in ZYX)
        
        # Debug gradient tracking status (only if enabled)
        if GRADIENT_DEBUG_ENABLED:
            print(f"GRADIENT_DEBUG: SpaceLossAcc.error() - point.requires_grad={self.point.tensor.requires_grad}")
            print(f"GRADIENT_DEBUG: z,y,x requires_grad: {z.requires_grad}, {y.requires_grad}, {x.requires_grad}")
        
        # Sample the volume at the point location
        try:
            values = self.interpolator.evaluate(z, y, x)
            
            # Handle any invalid values to prevent optimization failures
            # This matches the C++ behavior which doesn't explicitly handle invalid values
            # but relies on the interpolator to do proper bounds checking
            if torch.isinf(values).any() or torch.isnan(values).any():
                # Set a default value for invalid points - 0.0 is a reasonable default
                # if we're maximizing intensity, that's the lowest value, if minimizing, that's low too
                default_value = 0.0
                
                # Create a mask for invalid values
                invalid_mask = torch.isinf(values) | torch.isnan(values)
                
                # Replace invalid values with default
                values = torch.where(invalid_mask, 
                                    torch.tensor(default_value, device=values.device, dtype=values.dtype),
                                    values)
                
                # Log without flooding output
                print(f"SpaceLossAcc detected and fixed invalid values")
        except Exception as e:
            # If interpolation completely fails, return a reasonable default
            print(f"SpaceLossAcc interpolation failed: {e}")
            # Return tensor of right shape with default value 0.0
            values = torch.zeros((batch_size, 1), dtype=torch.float32, device=self.point.tensor.device)
            # If we were maximizing, we want to avoid this solution, so add a negative flag
            if self.maximize_mode:
                values.fill_(-1.0)  # Discourage this solution if maximizing
        
        # Ensure gradient tracking is maintained (only if debug enabled)
        if GRADIENT_DEBUG_ENABLED:
            print(f"GRADIENT_DEBUG: values.requires_grad={values.requires_grad}")
        
        # With our PyTorch grid_sample implementation, this shouldn't happen anymore
        # but we keep it for robustness
        if not values.requires_grad and z.requires_grad and GRADIENT_DEBUG_ENABLED:
            print(f"GRADIENT_DEBUG ERROR: Interpolator is breaking gradient chain!")
        
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
        
        # Create input tensors for the interpolator in ZYX order
        # Note: Points tensor is in ZYX order [z, y, x]
        z = points[:, 0:1]  # Z coordinate (first element in ZYX)
        y = points[:, 1:2]  # Y coordinate (second element in ZYX)
        x = points[:, 2:3]  # X coordinate (third element in ZYX)
        
        # Debug gradient tracking status (only if enabled)
        if GRADIENT_DEBUG_ENABLED:
            print(f"GRADIENT_DEBUG: SpaceLossAcc.jacobians() - point.requires_grad={self.point.tensor.requires_grad}")
            print(f"GRADIENT_DEBUG: z,y,x requires_grad: {z.requires_grad}, {y.requires_grad}, {x.requires_grad}")
        
        try:
            # Sample the volume and compute gradients at the point location
            # Gradients will be returned in ZYX order [dz, dy, dx]
            values, gradients = self.interpolator.evaluate_with_gradient(z, y, x)
            
            # Handle any invalid values in the results
            if torch.isinf(values).any() or torch.isnan(values).any():
                # Set a default value for invalid points - 0.0 is a reasonable default
                default_value = 0.0
                
                # Create masks for invalid values
                invalid_mask = torch.isinf(values) | torch.isnan(values)
                
                # Replace invalid values with default
                values = torch.where(invalid_mask, 
                                  torch.tensor(default_value, device=values.device, dtype=values.dtype),
                                  values)
                
                # Create a zero gradient for invalid points
                # This keeps the point stationary if invalid
                zero_gradient = torch.zeros_like(gradients)
                
                # Fix any NaN or inf in gradients
                invalid_grad_mask = torch.isnan(gradients) | torch.isinf(gradients)
                if invalid_grad_mask.any():
                    gradients = torch.where(invalid_grad_mask, zero_gradient, gradients)
                
                # Log without flooding output
                print(f"SpaceLossAcc fixed invalid values/gradients in jacobians")
                
            # Verify gradient tracking is maintained (only if debug enabled)
            if GRADIENT_DEBUG_ENABLED:
                print(f"GRADIENT_DEBUG: values.requires_grad={values.requires_grad}")
            
            # Reshape gradients to (batch_size, 1, 3) for Jacobian format
            jac = gradients.reshape(batch_size, 1, 3)
            
            # If in maximize mode, invert gradient direction to climb uphill
            if self.maximize_mode:
                jac = -jac
                
        except Exception as e:
            # If interpolation completely fails, return a reasonable default
            print(f"SpaceLossAcc gradient calculation failed: {e}")
            
            # Create a default zero value and zero gradient
            values = torch.zeros((batch_size, 1), dtype=torch.float32, device=self.point.tensor.device)
            jac = torch.zeros((batch_size, 1, 3), dtype=torch.float32, device=self.point.tensor.device)
            
            # If maximizing, set a negative value
            if self.maximize_mode:
                values.fill_(-1.0)  # Discourage this solution if maximizing
        
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