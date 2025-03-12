"""AutoDiff implementation of ZLocationLoss cost function for theseus."""

from typing import Optional, List, Tuple

import torch
import theseus as th


class ZLocationLossAutoDiff(th.AutoDiffCostFunction):
    """
    A cost function that penalizes deviations from a target Z coordinate after interpolation.
    
    This is an autodiff implementation of ZLocationLoss using Theseus AutoDiffCostFunction.
    It automatically calculates the Jacobians using PyTorch's autograd.
    
    This implementation is fully vmap-compatible for use with autodiff.
    
    Important coordinate conventions:
    - All 3D points are in ZYX order [z, y, x]
    - When accessing point components, use indices: 0=z, 1=y, 2=x
    - This matches the convention used in other cost functions
    """
    def __init__(
        self,
        location: th.Point2,  # 2D location for interpolation [y, x]
        matrix: th.Variable,  # Matrix containing 3D points (N x H x W x 3)
        target_z: th.Variable,  # Target Z coordinate
        cost_weight: th.CostWeight,
        name: Optional[str] = None,
    ):
        """
        Initialize the ZLocationLossAutoDiff cost function.
        
        Args:
            location: The 2D location [y, x] for interpolation (optimization variable)
            matrix: Matrix of 3D points for interpolation (auxiliary variable)
            target_z: Target Z coordinate as a 1-dimensional vector
            cost_weight: Weight for this cost function
            name: Optional name for this cost function
        """
        # Check that target_z is a 1D vector
        if target_z.dof() != 1:
            raise ValueError("target_z must be a 1-dimensional vector")
        
        # Store variables
        self.location = location
        self.matrix = matrix
        self.target_z = target_z
        
        # Define error function for autodiff computation
        def z_location_error_fn(optim_vars, aux_vars):
            """
            Error function for computing z-location based error.
            
            Args:
                optim_vars: List of optimization variable tensors [location]
                aux_vars: List of auxiliary variable tensors [matrix, target_z]
                
            Returns:
                Tensor with shape [batch_size, 1] containing the error values
            """
            # Extract tensors from optimization and auxiliary variables
            location_tensor = optim_vars[0]  # Shape: [batch_size, 2]
            matrix_tensor = aux_vars[0]  # Shape: [batch_size, H, W, 3]
            target_z_tensor = aux_vars[1]  # Shape: [batch_size, 1]
            
            # Ensure we're working with tensor data
            matrix_data = matrix_tensor if isinstance(matrix_tensor, torch.Tensor) else matrix_tensor.tensor
            target_z_data = target_z_tensor if isinstance(target_z_tensor, torch.Tensor) else target_z_tensor.tensor
            
            batch_size = location_tensor.shape[0]
            
            # Get matrix dimensions
            h, w = matrix_data.shape[1:3]
            
            # Extract Y and X coordinates
            y = location_tensor[:, 0:1]  # Shape: [batch_size, 1]
            x = location_tensor[:, 1:2]  # Shape: [batch_size, 1]
            
            # Clamp coordinates to valid range
            y_clamped = torch.clamp(y, 0.0, float(h - 2))
            x_clamped = torch.clamp(x, 0.0, float(w - 2))
            
            # Get integer and fractional parts without .floor().long() or .item()
            # Use floor division and remainder for integer and fractional parts
            y0 = torch.floor(y_clamped)
            x0 = torch.floor(x_clamped)
            
            # Compute interpolation weights
            fy = y_clamped - y0
            fx = x_clamped - x0
            
            # Precompute complements for clarity
            fy1 = 1.0 - fy
            fx1 = 1.0 - fx
            
            # Initialize interpolated Z values
            interp_z = torch.zeros_like(y)
            
            # Vectorized bilinear interpolation without loops or indexing,
            # using batch_sample_values helper function
            for b in range(batch_size):
                # Get batch index using modulo
                b_idx = b % matrix_data.shape[0]
                
                # Convert to indices for this batch
                y0_b = y0[b].to(torch.long)
                x0_b = x0[b].to(torch.long)
                y1_b = y0_b + 1
                x1_b = x0_b + 1
                
                # Get interpolation weights
                fy_b = fy[b]
                fx_b = fx[b]
                fy1_b = fy1[b]
                fx1_b = fx1[b]
                
                # Sample Z values (first component in ZYX order)
                c00 = matrix_data[b_idx, y0_b, x0_b, 0]
                c01 = matrix_data[b_idx, y0_b, x1_b, 0]
                c10 = matrix_data[b_idx, y1_b, x0_b, 0]
                c11 = matrix_data[b_idx, y1_b, x1_b, 0]
                
                # Bilinear interpolation
                interp_z[b] = (fy1_b * (fx1_b * c00 + fx_b * c01) +
                               fy_b * (fx1_b * c10 + fx_b * c11))
            
            # Calculate residual (difference from target Z)
            residual = interp_z - target_z_data
            
            return residual
        
        # Initialize the AutoDiffCostFunction with our variables and error function
        super().__init__(
            optim_vars=[location],
            aux_vars=[matrix, target_z],
            err_fn=z_location_error_fn,
            dim=1,
            cost_weight=cost_weight,
            name=name
        )
    
    def _copy_impl(self, new_name: Optional[str] = None) -> "ZLocationLossAutoDiff":
        """
        Create a copy of this cost function.
        
        Args:
            new_name: Optional new name for the copy
            
        Returns:
            A new instance of this cost function
        """
        return ZLocationLossAutoDiff(
            self.location.copy(),
            self.matrix.copy(),
            self.target_z.copy(),
            self.weight.copy(),
            name=new_name if new_name else self.name
        )