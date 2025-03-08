"""SurfaceLossD cost function for theseus."""

from typing import List, Optional, Tuple
import math

import torch
import theseus as th


class SurfaceLossD(th.CostFunction):
    """
    A cost function that penalizes deviations from an interpolated 3D point.
    
    This is a reimplementation of the SurfaceLossD C++ cost function from 
    volume-cartographer. It interpolates a 3D point from a matrix at a given
    location and enforces the 3D point to match this interpolated value.
    """
    def __init__(
        self,
        point: th.Point3,          # 3D point to be constrained
        location: th.Point2,       # 2D location for interpolation [y, x]
        matrix: th.Variable,       # Matrix containing 3D points (N x H x W x 3)
        cost_weight: th.CostWeight,
        name: Optional[str] = None,
    ):
        """
        Initialize the SurfaceLossD cost function.
        
        Args:
            point: The 3D point to constrain (optimization variable)
            location: The 2D location [y, x] for interpolation (optimization variable)
            matrix: Matrix of 3D points for interpolation (auxiliary variable)
            cost_weight: Weight for this cost function
            name: Optional name for this cost function
        """
        super().__init__(cost_weight, name=name)
        
        self.point = point
        self.location = location
        self.matrix = matrix
        
        # Register point and location as optimization variables
        self.register_optim_vars(["point", "location"])
        # Register matrix as auxiliary variable
        self.register_aux_vars(["matrix"])
        
    def error(self) -> torch.Tensor:
        """
        Compute the error between the point and the interpolated value.
        
        Returns:
            The error tensor of dimension 3 (x, y, z)
        """
        batch_size = self.point.shape[0]
        
        # Initialize residual
        residual = torch.zeros((batch_size, 3), device=self.point.device, dtype=self.point.dtype)
        
        # Get matrix dimensions - assuming matrix is [batch, height, width, 3]
        matrix_tensor = self.matrix.tensor
        
        # Get location coordinates
        locations = self.location.tensor  # [batch, 2] - represents [y, x]
        
        # Get point coordinates
        points = self.point.tensor  # [batch, 3]
        
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
                interpolated_point = (1-fy)*c0 + fy*c1  # Interpolate between rows
                
                # Calculate difference between actual point and interpolated point
                # This matches the C++ implementation:
                # residual[0] = T(_w)*(v[0] - p[0]);
                # residual[1] = T(_w)*(v[1] - p[1]);
                # residual[2] = T(_w)*(v[2] - p[2]);
                residual[b] = interpolated_point - points[b]
            
        return residual
        
    def dim(self) -> int:
        """
        Return the dimension of the error.
        
        Returns:
            The error dimension (3 for x, y, z)
        """
        return 3
    
    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Compute the Jacobians of the error with respect to the variables.
        
        Returns:
            A tuple containing:
                - A list of Jacobian matrices, one for each optimization variable
                - The error tensor
        """
        batch_size = self.point.shape[0]
        
        # Initialize jacobians for point and location
        jac_point = torch.zeros(batch_size, 3, 3, device=self.point.device, dtype=self.point.dtype)
        jac_location = torch.zeros(batch_size, 3, 2, device=self.location.device, dtype=self.location.dtype)
        
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
                
                # Get the four surrounding points
                c00 = matrix_tensor[b, yi, xi]        # top-left
                c01 = matrix_tensor[b, yi, xi+1]      # top-right
                c10 = matrix_tensor[b, yi+1, xi]      # bottom-left
                c11 = matrix_tensor[b, yi+1, xi+1]    # bottom-right
                
                # Jacobian for the point - it's just -I (identity matrix with negative sign)
                # This is because residual = interpolated_point - point
                # so d(residual)/d(point) = -I
                jac_point[b] = -torch.eye(3, device=self.point.device, dtype=self.point.dtype)
                
                # Jacobian for the location - needs to be computed for each component
                # d(interpolated_point)/d(location) using bilinear interpolation derivatives
                
                # For y coordinate:
                # d/dy = c1 - c0 = (1-fx)*c10 + fx*c11 - ((1-fx)*c00 + fx*c01)
                dy_derivatives = (1-fx)*c10 + fx*c11 - ((1-fx)*c00 + fx*c01)
                
                # For x coordinate:
                # d/dx = d/dfx * dfx/dx = (c01 - c00)*(1-fy) + (c11 - c10)*fy
                dx_derivatives = (c01 - c00)*(1-fy) + (c11 - c10)*fy
                
                # Set jacobian values for location
                jac_location[b, :, 0] = dy_derivatives  # d(interp)/dy for all 3 dimensions
                jac_location[b, :, 1] = dx_derivatives  # d(interp)/dx for all 3 dimensions
                
        # Calculate error
        error = self.error()
        
        return [jac_point, jac_location], error
    
    def _copy_impl(self, new_name: Optional[str] = None) -> "SurfaceLossD":
        """
        Create a copy of this cost function.
        
        Args:
            new_name: Optional new name for the copy
            
        Returns:
            A new instance of this cost function
        """
        return SurfaceLossD(
            self.point.copy(),
            self.location.copy(),
            self.matrix.copy(),
            self.weight.copy(),
            name=new_name if new_name else self.name
        )