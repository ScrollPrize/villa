"""ZLocationLoss cost function for theseus."""

from typing import List, Optional, Tuple
import math

import torch
import theseus as th


class ZLocationLoss(th.CostFunction):
    """
    A cost function that penalizes deviations from a target Z coordinate after interpolation.
    
    This is a reimplementation of the ZLocationLoss C++ cost function from 
    volume-cartographer. It interpolates a 3D point from a matrix at a given
    location and then enforces a specific Z coordinate.
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
        Initialize the ZLocationLoss cost function.
        
        Args:
            location: The 2D location [y, x] for interpolation (optimization variable)
            matrix: Matrix of 3D points for interpolation (auxiliary variable)
            target_z: Target Z coordinate as a 1-dimensional vector
            cost_weight: Weight for this cost function
            name: Optional name for this cost function
        """
        super().__init__(cost_weight, name=name)
        
        # Check that target_z is a 1D vector
        if target_z.dof() != 1:
            raise ValueError("target_z must be a 1-dimensional vector")
        
        self.location = location
        self.matrix = matrix
        self.target_z = target_z
        
        # Register location as an optimization variable
        self.register_optim_vars(["location"])
        # Register matrix and target_z as auxiliary variables
        self.register_aux_vars(["matrix", "target_z"])
        
    def error(self) -> torch.Tensor:
        """
        Compute the error between the interpolated Z coordinate and the target.
        
        Returns:
            The error tensor
        """
        batch_size = self.location.shape[0]
        
        # Initialize residual
        residual = torch.zeros((batch_size, 1), device=self.location.device, dtype=self.location.dtype)
        
        # Get matrix dimensions - assuming matrix is [batch, height, width, 3]
        matrix_tensor = self.matrix.tensor
        
        # Get location coordinates
        locations = self.location.tensor  # [batch, 2] - represents [y, x]
        
        # Iterate through batch
        for b in range(batch_size):
            # Check if location is valid
            y, x = locations[b, 0], locations[b, 1]
            
            # Check if location is within matrix bounds (with margin for interpolation)
            h, w = matrix_tensor.shape[1:3]
            if 0 <= y < h-1 and 0 <= x < w-1:
                # Get integer and fractional parts
                yi, xi = math.floor(y.item()), math.floor(x.item())
                fy, fx = y - yi, x - xi
                
                # Get the four surrounding points
                c00 = matrix_tensor[b, yi, xi]        # top-left
                c01 = matrix_tensor[b, yi, xi+1]      # top-right
                c10 = matrix_tensor[b, yi+1, xi]      # bottom-left
                c11 = matrix_tensor[b, yi+1, xi+1]    # bottom-right
                
                # Bilinear interpolation (same as interp_lin_2d in C++)
                c0 = (1-fx)*c00 + fx*c01  # Interpolate top row
                c1 = (1-fx)*c10 + fx*c11  # Interpolate bottom row
                p = (1-fy)*c0 + fy*c1     # Interpolate between rows
                
                # Extract z-coordinate (3rd element, index 2)
                z = p[2]
                
                # Calculate difference with target z
                residual[b, 0] = z - self.target_z.tensor[b, 0]
            
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
        batch_size = self.location.shape[0]
        
        # Initialize jacobian for location [batch, 1, 2]
        jac = torch.zeros(batch_size, 1, 2, device=self.location.device, dtype=self.location.dtype)
        
        # Get matrix and locations
        matrix_tensor = self.matrix.tensor
        locations = self.location.tensor
        
        # Iterate through batch
        for b in range(batch_size):
            # Get location coordinates
            y, x = locations[b, 0], locations[b, 1]
            
            # Check if location is within matrix bounds (with margin for interpolation)
            h, w = matrix_tensor.shape[1:3]
            if 0 <= y < h-1 and 0 <= x < w-1:
                # Get integer and fractional parts
                yi, xi = math.floor(y.item()), math.floor(x.item())
                fy, fx = y - yi, x - xi
                
                # Get the four surrounding points (z-coordinates only)
                c00 = matrix_tensor[b, yi, xi, 2]        # top-left
                c01 = matrix_tensor[b, yi, xi+1, 2]      # top-right
                c10 = matrix_tensor[b, yi+1, xi, 2]      # bottom-left
                c11 = matrix_tensor[b, yi+1, xi+1, 2]    # bottom-right
                
                # Compute derivatives of bilinear interpolation with respect to y and x
                # d/dy = d/dfy * dfy/dy = (c1 - c0) * 1
                # d/dx = d/dfx * dfx/dx = ((1-fy)*c01 + fy*c11 - (1-fy)*c00 - fy*c10) * 1
                
                # Derivatives of z w.r.t y and x
                dz_dy = (1-fx)*c10 + fx*c11 - (1-fx)*c00 - fx*c01  # c1 - c0
                dz_dx = (1-fy)*(c01 - c00) + fy*(c11 - c10)        # Linear interp of x-derivatives
                
                # Set jacobian values
                jac[b, 0, 0] = dz_dy  # dz/dy
                jac[b, 0, 1] = dz_dx  # dz/dx
                
        # Calculate error
        error = self.error()
        
        return [jac], error
    
    def _copy_impl(self, new_name: Optional[str] = None) -> "ZLocationLoss":
        """
        Create a copy of this cost function.
        
        Args:
            new_name: Optional new name for the copy
            
        Returns:
            A new instance of this cost function
        """
        return ZLocationLoss(
            self.location.copy(),
            self.matrix.copy(),
            self.target_z.copy(),
            self.weight.copy(),
            name=new_name if new_name else self.name
        )