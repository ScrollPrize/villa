"""AutoDiff implementation of SpaceLineLossAcc cost function for theseus."""

from typing import Optional, List, Tuple

import torch
import theseus as th

from tracer.core.interpolation import TrilinearInterpolator


class SpaceLineLossAccAutoDiff(th.AutoDiffCostFunction):
    """
    A cost function that evaluates multiple points along a line in a volume.
    
    This is an autodiff implementation of SpaceLineLossAcc using Theseus AutoDiffCostFunction.
    It samples a 3D volume at multiple points along a line defined by two endpoints, 
    and aggregates the interpolated values.
    
    Special behavior:
    - With positive weight: Minimizes the interpolated values along the line (finds minimum path)
    - With negative weight: Maximizes the interpolated values along the line (finds maximum path)
    """
    
    def _evaluate_autodiff_friendly(self, z, y, x):
        """
        Evaluate the volume at (z, y, x) points in a way that's compatible with autograd/vmap.
        
        This is a reimplementation of TrilinearInterpolator.evaluate() that avoids using .item(),
        which is not compatible with vmap in autograd.
        
        Args:
            z: Z coordinates tensor of shape (batch_size, num_points)
            y: Y coordinates tensor of shape (batch_size, num_points)
            x: X coordinates tensor of shape (batch_size, num_points)
            
        Returns:
            Values tensor of shape (batch_size, num_points)
        """
        # NOTE: It's important to understand that in our code, the coordinates are passed in as (z,y,x),
        # but for the point variables we store them as (x,y,z). This is why we need to swap the
        # order when computing the Jacobians and interpolating.
        # Get shapes
        batch_size = z.shape[0]
        num_points = 1 if len(z.shape) == 1 else z.shape[1]
        
        # Get volume tensor
        volume = self.interpolator.volume
        
        # Clamp coordinates to valid range
        z_clamped = torch.clamp(z, 0, volume.shape[1] - 1.001)
        y_clamped = torch.clamp(y, 0, volume.shape[2] - 1.001)
        x_clamped = torch.clamp(x, 0, volume.shape[3] - 1.001)
        
        # Get integer indices of the corner
        z0 = torch.floor(z_clamped).long()
        y0 = torch.floor(y_clamped).long()
        x0 = torch.floor(x_clamped).long()
        
        # Ensure corners are within bounds
        z0 = torch.clamp(z0, 0, volume.shape[1] - 2)
        y0 = torch.clamp(y0, 0, volume.shape[2] - 2)
        x0 = torch.clamp(x0, 0, volume.shape[3] - 2)
        
        # Compute fractional coordinates
        fz = z_clamped - z0.float()
        fy = y_clamped - y0.float()
        fx = x_clamped - x0.float()
        
        # Clamp fractional coordinates to [0, 1]
        fz = torch.clamp(fz, 0.0, 1.0)
        fy = torch.clamp(fy, 0.0, 1.0)
        fx = torch.clamp(fx, 0.0, 1.0)
        
        # Get values at the 8 corners of the cube
        # volume shape is (batch, z, y, x)
        # We use gather_nd-like indexing to get values for all batches
        vol_batch_idx = 0  # TrilinearInterpolator assumes batch dim 0
        
        # Create indices for the 8 corners
        c000 = volume[vol_batch_idx, z0, y0, x0]
        c001 = volume[vol_batch_idx, z0, y0, x0+1]
        c010 = volume[vol_batch_idx, z0, y0+1, x0]
        c011 = volume[vol_batch_idx, z0, y0+1, x0+1]
        c100 = volume[vol_batch_idx, z0+1, y0, x0]
        c101 = volume[vol_batch_idx, z0+1, y0, x0+1]
        c110 = volume[vol_batch_idx, z0+1, y0+1, x0]
        c111 = volume[vol_batch_idx, z0+1, y0+1, x0+1]
        
        # Perform trilinear interpolation
        c00 = c000 * (1 - fx) + c001 * fx
        c01 = c010 * (1 - fx) + c011 * fx
        c10 = c100 * (1 - fx) + c101 * fx
        c11 = c110 * (1 - fx) + c111 * fx
        
        c0 = c00 * (1 - fy) + c01 * fy
        c1 = c10 * (1 - fy) + c11 * fy
        
        c = c0 * (1 - fz) + c1 * fz
        
        # Reshape to match expected output format
        result = c.reshape(batch_size, -1)
        
        return result
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
        Initialize the SpaceLineLossAccAutoDiff cost function.
        
        Args:
            point_a: First endpoint of the line (optimization variable)
            point_b: Second endpoint of the line (optimization variable)
            interpolator: Trilinear interpolator for the 3D volume
            cost_weight: Weight for this cost function
            steps: Number of steps to sample along the line
            maximize: If True, maximize the values instead of minimizing
            name: Optional name for this cost function
        """
        self.point_a = point_a
        self.point_b = point_b
        self.interpolator = interpolator
        self.steps = max(2, steps)  # Ensure at least 2 steps
        self.maximize_mode = maximize
        
        # Define error function for autodiff computation
        def space_line_error_fn(optim_vars, aux_vars):
            """
            Error function for computing line-based error.
            
            Args:
                optim_vars: List of optimization variable tensors [point_a, point_b]
                aux_vars: List of auxiliary variable tensors (empty in this case)
                
            Returns:
                Tensor with shape [batch_size, 1] containing the average values
            """
            # Extract tensors from optimization variables
            point_a_tensor = optim_vars[0]  # Shape: [batch_size, 3]
            point_b_tensor = optim_vars[1]  # Shape: [batch_size, 3]
            
            # Make sure we're dealing with plain tensors, not Point3 objects
            if not isinstance(point_a_tensor, torch.Tensor):
                point_a_tensor = point_a_tensor.tensor
            if not isinstance(point_b_tensor, torch.Tensor):
                point_b_tensor = point_b_tensor.tensor
                
            batch_size = point_a_tensor.shape[0]
            
            # Sample points along the line and compute their values
            fractions = torch.linspace(1.0 / self.steps, 1.0 - 1.0 / self.steps, self.steps - 1, 
                                       device=point_a_tensor.device)
            
            # Instead of accumulating values in a loop, which can cause shape issues with autodiff,
            # we'll compute values for all fractions and then average them
            all_values = []
            
            for f in fractions:
                # Point tensors are stored as (x,y,z), but we need to pass (z,y,x) for interpolation
                # Coordinate 0 is x, 1 is y, 2 is z in point tensors 
                x = (1.0 - f) * point_a_tensor[:, 0:1] + f * point_b_tensor[:, 0:1]
                y = (1.0 - f) * point_a_tensor[:, 1:2] + f * point_b_tensor[:, 1:2]
                z = (1.0 - f) * point_a_tensor[:, 2:3] + f * point_b_tensor[:, 2:3]
                
                # Sample the volume at this point
                # We need to use a more autograd-friendly approach since the interpolator's evaluate
                # method uses .item() which doesn't work with vmap
                value = self._evaluate_autodiff_friendly(z, y, x)  # Shape (batch_size, 1)
                all_values.append(value)
            
            # Stack all values and compute mean along the first dimension
            if len(all_values) > 0:
                stacked_values = torch.stack(all_values, dim=1)  # Shape (batch_size, steps-1, 1)
                result = torch.mean(stacked_values, dim=1)       # Shape (batch_size, 1)
            else:
                # Fallback if no points were sampled
                result = torch.zeros((batch_size, 1), device=point_a_tensor.device, dtype=point_a_tensor.dtype)
            
            # Invert result if in maximize mode
            # For the autodiff version, we can directly invert the error instead of the Jacobians
            if self.maximize_mode:
                result = -result
                
            return result
        
        # Initialize the AutoDiffCostFunction with our variables and error function
        super().__init__(
            optim_vars=[point_a, point_b],
            err_fn=space_line_error_fn,
            dim=1,
            cost_weight=cost_weight,
            name=name
        )
    
    def _copy_impl(self, new_name: Optional[str] = None) -> "SpaceLineLossAccAutoDiff":
        """
        Create a copy of this cost function.
        
        Args:
            new_name: Optional new name for the copy
            
        Returns:
            A new instance of this cost function
        """
        return SpaceLineLossAccAutoDiff(
            self.point_a.copy(),
            self.point_b.copy(),
            self.interpolator,  # Interpolator is shared, not copied
            self.weight.copy(),
            steps=self.steps,
            maximize=self.maximize_mode,
            name=new_name if new_name else self.name
        )