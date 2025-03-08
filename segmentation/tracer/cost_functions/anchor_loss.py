"""AnchorLoss cost function for theseus."""

from typing import List, Optional, Tuple

import torch
import theseus as th

from .trilinear_interpolator import TrilinearInterpolator


class AnchorLoss(th.CostFunction):
    """
    A cost function that anchors a 3D point based on a volume and distance.
    
    This is a reimplementation of the AnchorLoss C++ cost function from 
    volume-cartographer. It has two components:
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
        Initialize the AnchorLoss cost function.
        
        Args:
            point: The 3D point to anchor (optimization variable)
            anchor_point: The 3D anchor point (optimization variable)
            interpolator: Trilinear interpolator for the 3D volume
            cost_weight: Weight for this cost function
            name: Optional name for this cost function
        """
        super().__init__(cost_weight, name=name)
        
        self.point = point
        self.anchor_point = anchor_point
        self.interpolator = interpolator
        
        # Register points as optimization variables
        self.register_optim_vars(["point", "anchor_point"])
        
    def error(self) -> torch.Tensor:
        """
        Compute the error, based on the volume value at the anchor point
        and the distance between point and anchor point.
        
        Returns:
            The error tensor (2 values per batch element)
        """
        batch_size = self.point.shape[0]
        
        # Get point coordinates
        points = self.point.tensor            # Shape (batch_size, 3)
        anchor_points = self.anchor_point.tensor  # Shape (batch_size, 3)
        
        # Create input tensors for the interpolator
        z = anchor_points[:, 2:3]  # Shape (batch_size, 1)
        y = anchor_points[:, 1:2]  # Shape (batch_size, 1)
        x = anchor_points[:, 0:1]  # Shape (batch_size, 1)
        
        # Sample the volume at the anchor point location
        values = self.interpolator.evaluate(z, y, x)  # Shape (batch_size, 1)
        
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
        
    def dim(self) -> int:
        """
        Return the dimension of the error.
        
        Returns:
            The error dimension (2)
        """
        return 2
    
    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Compute the Jacobians of the error with respect to the variables.
        
        Returns:
            A tuple containing:
                - A list of Jacobian matrices, one for each optimization variable
                - The error tensor
        """
        batch_size = self.point.shape[0]
        
        # Get point coordinates
        points = self.point.tensor            # Shape (batch_size, 3)
        anchor_points = self.anchor_point.tensor  # Shape (batch_size, 3)
        
        # Create input tensors for the interpolator
        z = anchor_points[:, 2:3]  # Shape (batch_size, 1)
        y = anchor_points[:, 1:2]  # Shape (batch_size, 1)
        x = anchor_points[:, 0:1]  # Shape (batch_size, 1)
        
        # Sample the volume and compute gradients at the anchor point location
        values, gradients = self.interpolator.evaluate_with_gradient(z, y, x)
        
        # Calculate the first residual term: volume value term
        values = values - 1.0
        value_mask = (values > 0.0).float()  # For clamping derivatives
        values = torch.clamp(values, min=0.0)
        volume_term = values * values  # Square the value
        
        # Calculate the second residual term: distance term
        diffs = points - anchor_points  # Shape (batch_size, 3)
        distances = torch.norm(diffs, dim=1, keepdim=True)  # Shape (batch_size, 1)
        
        # Create error vector
        error_vector = torch.cat([volume_term, distances], dim=1)  # Shape (batch_size, 2)
        
        # Calculate Jacobians
        # 1. Jacobian of the point variable (affects only the distance term)
        # The derivative of distance w.r.t. point is the normalized difference vector
        safe_distances = torch.clamp(distances, min=1e-10)  # Avoid division by zero
        normalized_diffs = diffs / safe_distances.unsqueeze(-1)  # Shape (batch_size, 3)
        
        # Jacobian for point - shape (batch_size, 2, 3)
        # First row is all zeros (volume term doesn't depend on point)
        # Second row is the normalized difference vector (derivative of distance)
        jac_point = torch.zeros(batch_size, 2, 3, device=points.device, dtype=points.dtype)
        jac_point[:, 1, :] = normalized_diffs
        
        # 2. Jacobian of the anchor_point variable (affects both terms)
        # For volume term: derivative is 2*value*gradient (chain rule for squared term)
        # Multiplied by mask to handle clamping
        vol_term_grad = 2.0 * values * value_mask * gradients  # Shape (batch_size, 1, 3)
        
        # For distance term: derivative is -normalized_diffs
        dist_term_grad = -normalized_diffs.unsqueeze(1)  # Shape (batch_size, 1, 3)
        
        # Combine the two parts of the Jacobian for anchor_point
        jac_anchor = torch.zeros(batch_size, 2, 3, device=points.device, dtype=points.dtype)
        jac_anchor[:, 0, :] = vol_term_grad.squeeze(1)
        jac_anchor[:, 1, :] = dist_term_grad.squeeze(1)
        
        # Return Jacobians (for point and anchor_point) and error
        return [jac_point, jac_anchor], error_vector
    
    def _copy_impl(self, new_name: Optional[str] = None) -> "AnchorLoss":
        """
        Create a copy of this cost function.
        
        Args:
            new_name: Optional new name for the copy
            
        Returns:
            A new instance of this cost function
        """
        return AnchorLoss(
            self.point.copy(),
            self.anchor_point.copy(),
            self.interpolator,  # Interpolator is shared, not copied
            self.weight.copy(),
            name=new_name if new_name else self.name
        )