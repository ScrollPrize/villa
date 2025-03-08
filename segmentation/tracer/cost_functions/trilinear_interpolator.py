"""Trilinear interpolator for 3D volumes."""

from typing import List, Optional, Union, Tuple

import torch
import numpy as np


class TrilinearInterpolator:
    """
    A trilinear interpolator for 3D volumes.
    
    This provides similar functionality to CachedChunked3dInterpolator in the C++ code
    but without the chunking infrastructure. It focuses on accurate trilinear interpolation.
    """
    def __init__(self, volume: torch.Tensor):
        """
        Initialize the interpolator with a 3D volume.
        
        Args:
            volume: A 3D tensor of shape [depth, height, width] or 
                   a 4D tensor of shape [batch, depth, height, width]
        """
        if volume.dim() == 3:
            # Add batch dimension if not present
            self.volume = volume.unsqueeze(0)
        elif volume.dim() == 4:
            self.volume = volume
        else:
            raise ValueError(f"Volume must be 3D or 4D, got shape {volume.shape}")
        
        self.shape = list(self.volume.shape[1:])  # [depth, height, width]
        self.device = volume.device
        self.dtype = volume.dtype
        self.batch_size = self.volume.shape[0]
    
    def evaluate(self, z: torch.Tensor, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the volume at the specified coordinates using trilinear interpolation.
        
        This matches the Evaluate method from CachedChunked3dInterpolator.
        
        Args:
            z: Z coordinates (batch_size, ...)
            y: Y coordinates (batch_size, ...)
            x: X coordinates (batch_size, ...)
            
        Returns:
            Interpolated values at the specified coordinates (batch_size, ...)
        """
        batch_size = z.shape[0]
        assert batch_size <= self.batch_size, f"Batch size mismatch: {batch_size} > {self.batch_size}"
        
        # Get input shape to reshape output at the end
        input_shape = z.shape
        
        # Flatten coordinates for processing
        z_flat = z.reshape(batch_size, -1)
        y_flat = y.reshape(batch_size, -1)
        x_flat = x.reshape(batch_size, -1)
        num_points = z_flat.shape[1]
        
        # Initialize output
        output = torch.zeros(batch_size, num_points, device=self.device, dtype=self.dtype)
        
        # Process each point
        for b in range(batch_size):
            for i in range(num_points):
                # Get coordinates
                zi, yi, xi = z_flat[b, i].item(), y_flat[b, i].item(), x_flat[b, i].item()
                
                # Get integer indices of the corner
                z0 = int(np.floor(zi))
                y0 = int(np.floor(yi))
                x0 = int(np.floor(xi))
                
                # Ensure corner is within bounds (clamping)
                z0 = max(0, min(z0, self.shape[0] - 2))
                y0 = max(0, min(y0, self.shape[1] - 2))
                x0 = max(0, min(x0, self.shape[2] - 2))
                
                # Compute fractional coordinates
                fz = zi - z0
                fy = yi - y0
                fx = xi - x0
                
                # Clamp fractional coordinates to [0, 1]
                fz = max(0.0, min(1.0, fz))
                fy = max(0.0, min(1.0, fy))
                fx = max(0.0, min(1.0, fx))
                
                # Get values at the 8 corners of the cube
                c000 = self.volume[b, z0, y0, x0].item()
                c001 = self.volume[b, z0, y0, x0+1].item()
                c010 = self.volume[b, z0, y0+1, x0].item()
                c011 = self.volume[b, z0, y0+1, x0+1].item()
                c100 = self.volume[b, z0+1, y0, x0].item()
                c101 = self.volume[b, z0+1, y0, x0+1].item()
                c110 = self.volume[b, z0+1, y0+1, x0].item()
                c111 = self.volume[b, z0+1, y0+1, x0+1].item()
                
                # Perform trilinear interpolation
                c00 = c000 * (1 - fx) + c001 * fx
                c01 = c010 * (1 - fx) + c011 * fx
                c10 = c100 * (1 - fx) + c101 * fx
                c11 = c110 * (1 - fx) + c111 * fx
                
                c0 = c00 * (1 - fy) + c01 * fy
                c1 = c10 * (1 - fy) + c11 * fy
                
                value = c0 * (1 - fz) + c1 * fz
                
                output[b, i] = value
        
        # Reshape output to match input shape
        return output.reshape(input_shape)
        
    def evaluate_with_gradient(self, z: torch.Tensor, y: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate the volume and compute the gradient at the specified coordinates.
        
        Args:
            z: Z coordinates (batch_size, ...)
            y: Y coordinates (batch_size, ...)
            x: X coordinates (batch_size, ...)
            
        Returns:
            Tuple of (values, gradients) at the specified coordinates
            values: shape (batch_size, ...)
            gradients: shape (batch_size, ..., 3) - gradient in z, y, x order
        """
        batch_size = z.shape[0]
        assert batch_size <= self.batch_size, f"Batch size mismatch: {batch_size} > {self.batch_size}"
        
        # Get input shape to reshape output at the end
        input_shape = z.shape
        
        # Flatten coordinates for processing
        z_flat = z.reshape(batch_size, -1)
        y_flat = y.reshape(batch_size, -1)
        x_flat = x.reshape(batch_size, -1)
        num_points = z_flat.shape[1]
        
        # Initialize output
        output = torch.zeros(batch_size, num_points, device=self.device, dtype=self.dtype)
        gradient = torch.zeros(batch_size, num_points, 3, device=self.device, dtype=self.dtype)
        
        # Process each point
        for b in range(batch_size):
            for i in range(num_points):
                # Get coordinates
                zi, yi, xi = z_flat[b, i].item(), y_flat[b, i].item(), x_flat[b, i].item()
                
                # Get integer indices of the corner
                z0 = int(np.floor(zi))
                y0 = int(np.floor(yi))
                x0 = int(np.floor(xi))
                
                # Ensure corner is within bounds (clamping)
                z0 = max(0, min(z0, self.shape[0] - 2))
                y0 = max(0, min(y0, self.shape[1] - 2))
                x0 = max(0, min(x0, self.shape[2] - 2))
                
                # Compute fractional coordinates
                fz = zi - z0
                fy = yi - y0
                fx = xi - x0
                
                # Clamp fractional coordinates to [0, 1]
                fz = max(0.0, min(1.0, fz))
                fy = max(0.0, min(1.0, fy))
                fx = max(0.0, min(1.0, fx))
                
                # Get values at the 8 corners of the cube
                c000 = self.volume[b, z0, y0, x0].item()
                c001 = self.volume[b, z0, y0, x0+1].item()
                c010 = self.volume[b, z0, y0+1, x0].item()
                c011 = self.volume[b, z0, y0+1, x0+1].item()
                c100 = self.volume[b, z0+1, y0, x0].item()
                c101 = self.volume[b, z0+1, y0, x0+1].item()
                c110 = self.volume[b, z0+1, y0+1, x0].item()
                c111 = self.volume[b, z0+1, y0+1, x0+1].item()
                
                # Perform trilinear interpolation
                c00 = c000 * (1 - fx) + c001 * fx
                c01 = c010 * (1 - fx) + c011 * fx
                c10 = c100 * (1 - fx) + c101 * fx
                c11 = c110 * (1 - fx) + c111 * fx
                
                c0 = c00 * (1 - fy) + c01 * fy
                c1 = c10 * (1 - fy) + c11 * fy
                
                value = c0 * (1 - fz) + c1 * fz
                output[b, i] = value
                
                # Compute partial derivatives for gradient
                # Derivative with respect to x
                dc00_dx = c001 - c000
                dc01_dx = c011 - c010
                dc10_dx = c101 - c100
                dc11_dx = c111 - c110
                
                dc0_dx = dc00_dx * (1 - fy) + dc01_dx * fy
                dc1_dx = dc10_dx * (1 - fy) + dc11_dx * fy
                
                dx = dc0_dx * (1 - fz) + dc1_dx * fz
                
                # Derivative with respect to y
                dc0_dy = c01 - c00
                dc1_dy = c11 - c10
                
                dy = dc0_dy * (1 - fz) + dc1_dy * fz
                
                # Derivative with respect to z
                dz = c1 - c0
                
                # Store gradient (z, y, x order to match input order)
                gradient[b, i, 0] = dz
                gradient[b, i, 1] = dy
                gradient[b, i, 2] = dx
        
        # Reshape output to match input shape
        return output.reshape(input_shape), gradient.reshape(*input_shape, 3)
        
    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the volume at the specified points using trilinear interpolation.
        
        Args:
            points: Points of shape (batch_size, ..., 3) in order (z, y, x)
            
        Returns:
            Interpolated values at the specified points (batch_size, ...)
        """
        # Extract z, y, x coordinates
        z = points[..., 0]
        y = points[..., 1]
        x = points[..., 2]
        
        return self.evaluate(z, y, x)