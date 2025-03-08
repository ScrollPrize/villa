"""SpaceLineLossAcc cost function for theseus."""

from typing import List, Optional, Tuple

import torch
import theseus as th

from .trilinear_interpolator import TrilinearInterpolator


class SpaceLineLossAcc(th.CostFunction):
    """
    A cost function that evaluates multiple points along a line in a volume.
    
    This is a reimplementation of the SpaceLineLossAcc C++ cost function from 
    volume-cartographer. It samples a 3D volume at multiple points along a line
    defined by two endpoints, and aggregates the interpolated values.
    
    Special behavior:
    - With positive weight: Minimizes the interpolated values along the line (finds minimum path)
    - With negative weight: Maximizes the interpolated values along the line (finds maximum path)
    """
    def __init__(
        self,
        point_a: th.Point3,           # First endpoint of the line
        point_b: th.Point3,           # Second endpoint of the line
        interpolator: TrilinearInterpolator,  # Volume interpolator
        cost_weight: th.CostWeight,
        steps: int = 5,               # Number of steps along the line
        maximize: bool = False,       # If True, maximize values instead of minimizing
        name: Optional[str] = None,
    ):
        """
        Initialize the SpaceLineLossAcc cost function.
        
        Args:
            point_a: First endpoint of the line (optimization variable)
            point_b: Second endpoint of the line (optimization variable)
            interpolator: Trilinear interpolator for the 3D volume
            cost_weight: Weight for this cost function
            steps: Number of steps to sample along the line
            maximize: If True, maximize the values instead of minimizing
            name: Optional name for this cost function
        """
        super().__init__(cost_weight, name=name)
        
        self.point_a = point_a
        self.point_b = point_b
        self.interpolator = interpolator
        self.steps = max(2, steps)  # Ensure at least 2 steps
        self.maximize_mode = maximize
        
        # Register points as optimization variables
        self.register_optim_vars(["point_a", "point_b"])
        
    def _sample_line_points(self, point_a: torch.Tensor, point_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample points along the line between point_a and point_b.
        
        Args:
            point_a: First endpoint tensor of shape (batch_size, 3)
            point_b: Second endpoint tensor of shape (batch_size, 3)
            
        Returns:
            Tuple of (z, y, x) coordinates for sampled points
            Each has shape (batch_size, steps-1)
        """
        batch_size = point_a.shape[0]
        
        # Skip the first point (point_a) and sample steps-1 points along the line
        # This matches the C++ implementation which samples points 1/(steps) to (steps-1)/(steps) along the line
        fractions = torch.linspace(1.0 / self.steps, 1.0 - 1.0 / self.steps, self.steps - 1, device=point_a.device)
        
        # Pre-allocate tensors for sampled coordinates
        z_samples = torch.zeros(batch_size, self.steps - 1, device=point_a.device, dtype=point_a.dtype)
        y_samples = torch.zeros(batch_size, self.steps - 1, device=point_a.device, dtype=point_a.dtype)
        x_samples = torch.zeros(batch_size, self.steps - 1, device=point_a.device, dtype=point_a.dtype)
        
        # Sample points along the line for each batch element
        for b in range(batch_size):
            for i, f in enumerate(fractions):
                # Compute the linear interpolation: (1-f)*point_a + f*point_b
                z_samples[b, i] = (1.0 - f) * point_a[b, 2] + f * point_b[b, 2]
                y_samples[b, i] = (1.0 - f) * point_a[b, 1] + f * point_b[b, 1]
                x_samples[b, i] = (1.0 - f) * point_a[b, 0] + f * point_b[b, 0]
        
        return z_samples, y_samples, x_samples
        
    def error(self) -> torch.Tensor:
        """
        Compute the error by sampling the volume along the line.
        
        Returns:
            The error tensor (average of sampled values)
        """
        batch_size = self.point_a.shape[0]
        
        # Get point coordinates
        point_a = self.point_a.tensor  # Shape (batch_size, 3)
        point_b = self.point_b.tensor  # Shape (batch_size, 3)
        
        # Sample points along the line
        z_samples, y_samples, x_samples = self._sample_line_points(point_a, point_b)
        
        # Initialize sum of values
        sum_values = torch.zeros(batch_size, 1, device=point_a.device, dtype=point_a.dtype)
        
        # Sample the volume at each point along the line
        for i in range(self.steps - 1):
            z = z_samples[:, i:i+1]  # Shape (batch_size, 1)
            y = y_samples[:, i:i+1]  # Shape (batch_size, 1)
            x = x_samples[:, i:i+1]  # Shape (batch_size, 1)
            
            # Sample the volume at this point
            values = self.interpolator.evaluate(z, y, x)  # Shape (batch_size, 1)
            
            # Add to sum
            sum_values += values
        
        # Return the average value along the line
        return sum_values / (self.steps - 1)
        
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
        
        This computes the gradient of the sampled volume with respect to the endpoints.
        
        Special behavior:
        - With positive weight: Gradients point towards lower values
        - With negative weight: Gradients point towards higher values
        
        Returns:
            A tuple containing:
                - A list of Jacobian matrices, one for each optimization variable
                - The error tensor
        """
        batch_size = self.point_a.shape[0]
        
        # Get point coordinates
        point_a = self.point_a.tensor  # Shape (batch_size, 3)
        point_b = self.point_b.tensor  # Shape (batch_size, 3)
        
        # Sample points along the line
        z_samples, y_samples, x_samples = self._sample_line_points(point_a, point_b)
        
        # Initialize sum of values and gradients
        sum_values = torch.zeros(batch_size, 1, device=point_a.device, dtype=point_a.dtype)
        sum_gradients = torch.zeros(batch_size, 1, 3, device=point_a.device, dtype=point_a.dtype)
        
        # Sample the volume and compute gradients at each point along the line
        for i in range(self.steps - 1):
            z = z_samples[:, i:i+1]  # Shape (batch_size, 1)
            y = y_samples[:, i:i+1]  # Shape (batch_size, 1)
            x = x_samples[:, i:i+1]  # Shape (batch_size, 1)
            
            # Sample the volume at this point and get gradients
            values, gradients = self.interpolator.evaluate_with_gradient(z, y, x)
            
            # Add to sums
            sum_values += values
            sum_gradients += gradients.reshape(batch_size, 1, 3)
        
        # Calculate the average value and gradient
        avg_value = sum_values / (self.steps - 1)
        avg_gradient = sum_gradients / (self.steps - 1)
        
        # Compute Jacobians for the endpoints
        # The Jacobian represents how the average value changes as we move the endpoints
        # For a linear interpolation, points closer to the beginning of the line
        # are more influenced by point_a, and points closer to the end are more
        # influenced by point_b
        
        # Initialize Jacobians for point_a and point_b
        jac_point_a = torch.zeros(batch_size, 1, 3, device=point_a.device, dtype=point_a.dtype)
        jac_point_b = torch.zeros(batch_size, 1, 3, device=point_a.device, dtype=point_a.dtype)
        
        # For each sampled point, calculate its contribution to the Jacobians
        fractions = torch.linspace(1.0 / self.steps, 1.0 - 1.0 / self.steps, self.steps - 1, device=point_a.device)
        
        for i, f in enumerate(fractions):
            z = z_samples[:, i:i+1]
            y = y_samples[:, i:i+1]
            x = x_samples[:, i:i+1]
            
            # Get gradients at this point
            _, gradients = self.interpolator.evaluate_with_gradient(z, y, x)
            grad = gradients.reshape(batch_size, 1, 3)
            
            # Contribution to point_a is weighted by (1-f)
            jac_point_a += (1.0 - f) * grad
            
            # Contribution to point_b is weighted by f
            jac_point_b += f * grad
        
        # Normalize by number of steps
        jac_point_a /= (self.steps - 1)
        jac_point_b /= (self.steps - 1)
        
        # If in maximize mode, invert gradient direction to climb uphill
        if self.maximize_mode:
            jac_point_a = -jac_point_a
            jac_point_b = -jac_point_b
        
        # Return jacobians and error
        return [jac_point_a, jac_point_b], avg_value
    
    def _copy_impl(self, new_name: Optional[str] = None) -> "SpaceLineLossAcc":
        """
        Create a copy of this cost function.
        
        Args:
            new_name: Optional new name for the copy
            
        Returns:
            A new instance of this cost function
        """
        return SpaceLineLossAcc(
            self.point_a.copy(),
            self.point_b.copy(),
            self.interpolator,  # Interpolator is shared, not copied
            self.weight.copy(),
            steps=self.steps,
            maximize=self.maximize_mode,
            name=new_name if new_name else self.name
        )