"""AutoDiff implementation of AnchorLoss cost function for theseus."""

from typing import Optional, List, Tuple

import torch
import theseus as th
import torch.nn.functional as F

from tracer.core.interpolation import TrilinearInterpolator


class AnchorLossAutoDiff(th.AutoDiffCostFunction):
    """
    A cost function that anchors a 3D point based on a volume and distance.
    
    This is an autodiff implementation of AnchorLoss using Theseus AutoDiffCostFunction.
    It has two components:
    1. A volume term: It evaluates a 3D "anchor" point in the volume and uses that value
    2. A distance term: It calculates the distance between the 3D point and the anchor point
    
    The overall cost is a combination of these two terms.
    """
    def __init__(
        self,
        point: th.Point3,            # 3D point to anchor (optimization variable)
        anchor_point: th.Point3,     # 3D anchor point (optimization variable)  
        interpolator: TrilinearInterpolator,  # Volume interpolator
        cost_weight: th.CostWeight,
        name: Optional[str] = None,
    ):
        """
        Initialize the AnchorLossAutoDiff cost function.
        
        Args:
            point: The 3D point to anchor (optimization variable)
            anchor_point: The 3D anchor point (optimization variable)
            interpolator: Trilinear interpolator for the 3D volume
            cost_weight: Weight for this cost function
            name: Optional name for this cost function
        """
        self.point = point
        self.anchor_point = anchor_point
        
        # Store the volume data and make it accessible for autograd
        self.volume = interpolator.volume.clone() 
        
        # Define error function for autodiff computation
        def anchor_error_fn(optim_vars, aux_vars):
            """
            Error function for computing anchor-based error.
            
            Args:
                optim_vars: List of optimization variable tensors [point, anchor_point]
                aux_vars: List of auxiliary variable tensors (empty in this case)
                
            Returns:
                Tensor with shape [batch_size, 2] containing the error values
            """
            # Extract tensors from optimization variables
            points = optim_vars[0]  # Shape: [batch_size, 3]
            anchor_points = optim_vars[1]  # Shape: [batch_size, 3]
            
            # Make sure we're dealing with plain tensors, not Point3 objects
            if not isinstance(points, torch.Tensor):
                points = points.tensor
            if not isinstance(anchor_points, torch.Tensor):
                anchor_points = anchor_points.tensor
                
            # Workaround for Theseus batch handling issues:
            # Always use only the first element in the batch to avoid 
            # dimension mismatch errors in TheseusLayer.forward()
            batch_size = 1
            
            # Take only the first element for each tensor
            if points.shape[0] > 1:
                points = points[0:1, :]
            
            if anchor_points.shape[0] > 1:
                anchor_points = anchor_points[0:1, :]
            
            # Ensure both points and anchor_points have shape (batch_size, 3)
            if len(points.shape) > 2:
                points = points.reshape(batch_size, 3)
                
            if len(anchor_points.shape) > 2:
                anchor_points = anchor_points.reshape(batch_size, 3)
            
            # For autodiff, we need a vectorized, differentiable interpolation approach
            # First clamp the coordinates to valid range
            volume_shape = self.volume.shape
            
            # Clamp to valid range for interpolation (bounds of the volume)
            x = torch.clamp(anchor_points[:, 0], 0, volume_shape[2] - 1.001)
            y = torch.clamp(anchor_points[:, 1], 0, volume_shape[1] - 1.001)
            z = torch.clamp(anchor_points[:, 2], 0, volume_shape[0] - 1.001)
            
            # Get integer and fractional parts
            x0 = torch.floor(x).long()
            y0 = torch.floor(y).long()
            z0 = torch.floor(z).long()
            
            # Ensure the next index is within bounds
            x1 = (x0 + 1).clamp(max=volume_shape[2] - 1)
            y1 = (y0 + 1).clamp(max=volume_shape[1] - 1)
            z1 = (z0 + 1).clamp(max=volume_shape[0] - 1)
            
            # Compute the fractional part
            xd = (x - x0.float())
            yd = (y - y0.float())
            zd = (z - z0.float())
            
            # Create values tensor to hold the results for the batch
            values = torch.zeros((batch_size, 1), device=anchor_points.device, dtype=self.volume.dtype)
            
            # Sample the 8 corners of the voxel cube
            for b in range(batch_size):
                # Get the 8 corner values
                c000 = self.volume[z0[b], y0[b], x0[b]].float()
                c001 = self.volume[z0[b], y0[b], x1[b]].float()
                c010 = self.volume[z0[b], y1[b], x0[b]].float()
                c011 = self.volume[z0[b], y1[b], x1[b]].float()
                c100 = self.volume[z1[b], y0[b], x0[b]].float()
                c101 = self.volume[z1[b], y0[b], x1[b]].float()
                c110 = self.volume[z1[b], y1[b], x0[b]].float()
                c111 = self.volume[z1[b], y1[b], x1[b]].float()
                
                # Compute the weighted sum for trilinear interpolation
                c00 = c000 * (1 - xd[b]) + c001 * xd[b]
                c01 = c010 * (1 - xd[b]) + c011 * xd[b]
                c10 = c100 * (1 - xd[b]) + c101 * xd[b]
                c11 = c110 * (1 - xd[b]) + c111 * xd[b]
                
                c0 = c00 * (1 - yd[b]) + c01 * yd[b]
                c1 = c10 * (1 - yd[b]) + c11 * yd[b]
                
                values[b, 0] = c0 * (1 - zd[b]) + c1 * zd[b]
            
            # Calculate the first residual term: volume value term
            # Similar to C++ code: v = v - T(1); if (v < T(0)) v = T(0); residual[0] = T(_w)*v*v;
            values = values - 1.0
            values = torch.clamp(values, min=0.0)
            volume_term = values * values  # Square the value
            
            # Calculate the second residual term: distance term
            # Similar to C++ code: residual[1] = T(_w)*sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
            diffs = points - anchor_points  # Shape (batch_size, 3)
            distances = torch.norm(diffs, dim=1, keepdim=True)  # Shape (batch_size, 1)
            
            # Combine the two terms
            error_vector = torch.cat([volume_term, distances], dim=1)  # Shape (batch_size, 2)
            
            return error_vector
        
        # Initialize the AutoDiffCostFunction with our variables and error function
        super().__init__(
            optim_vars=[point, anchor_point],
            err_fn=anchor_error_fn,
            dim=2,
            cost_weight=cost_weight,
            name=name
        )
    
    def _copy_impl(self, new_name: Optional[str] = None) -> "AnchorLossAutoDiff":
        """
        Create a copy of this cost function.
        
        Args:
            new_name: Optional new name for the copy
            
        Returns:
            A new instance of this cost function
        """
        # Recreate the interpolator
        interpolator = TrilinearInterpolator(self.volume)
        
        return AnchorLossAutoDiff(
            self.point.copy(),
            self.anchor_point.copy(),
            interpolator,  # Interpolator is recreated
            self.weight.copy(),
            name=new_name if new_name else self.name
        )