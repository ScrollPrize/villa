"""LinChkDistLoss cost function for theseus."""

from typing import List, Optional, Tuple

import torch
import theseus as th


class LinChkDistLoss(th.CostFunction):
    """
    A cost function that penalizes deviations from a target 2D location.
    
    This is a reimplementation of the LinChkDistLoss C++ cost function from 
    volume-cartographer. It computes the square root of the absolute difference
    between a 2D point and a target point, but only for positive differences.
    """
    def __init__(
        self,
        point: th.Point2,
        target: th.Point2,
        cost_weight: th.CostWeight,
        name: Optional[str] = None,
    ):
        """
        Initialize the LinChkDistLoss cost function.
        
        Args:
            point: The 2D point (optimization variable)
            target: The target 2D point (auxiliary variable)
            cost_weight: Weight for this cost function
            name: Optional name for this cost function
        """
        super().__init__(cost_weight, name=name)
        
        self.point = point
        self.target = target
        
        # Register point as an optimization variable
        self.register_optim_vars(["point"])
        # Register target as an auxiliary variable
        self.register_aux_vars(["target"])
        
    def error(self) -> torch.Tensor:
        """
        Compute the error between the current point and the target.
        
        Returns:
            The error tensor of dimension 2
        """
        # Get the current point and target values
        point_tensor = self.point.tensor
        target_tensor = self.target.tensor
        
        # Calculate absolute differences - directly matching the C++ implementation
        # T a = abs(p[0]-T(_p[0]));
        # T b = abs(p[1]-T(_p[1]));
        a = torch.abs(point_tensor[:, 0:1] - target_tensor[:, 0:1])  # x component
        b = torch.abs(point_tensor[:, 1:2] - target_tensor[:, 1:2])  # y component
        
        # Initialize residuals
        residual = torch.zeros((point_tensor.shape[0], 2), device=point_tensor.device, dtype=point_tensor.dtype)
        
        # Follow the C++ implementation exactly:
        # if (a > T(0))
        #     residual[0] = T(_w)*sqrt(a);
        # else
        #     residual[0] = T(0);
        for i in range(point_tensor.shape[0]):
            # X component
            if a[i, 0] > 0:
                residual[i, 0] = torch.sqrt(a[i, 0])
            
            # Y component  
            if b[i, 0] > 0:
                residual[i, 1] = torch.sqrt(b[i, 0])
                    
        return residual
        
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
        
        # Get the current point and target values
        point_tensor = self.point.tensor
        target_tensor = self.target.tensor
        
        # Calculate differences and their absolute values
        diff_x = point_tensor[:, 0:1] - target_tensor[:, 0:1]  # x component
        diff_y = point_tensor[:, 1:2] - target_tensor[:, 1:2]  # y component
        abs_x = torch.abs(diff_x)
        abs_y = torch.abs(diff_y)
        
        # Create Jacobian for the point
        jac = torch.zeros(batch_size, 2, 2, device=self.point.device, dtype=self.point.dtype)
        
        # Apply the derivative of sqrt(abs(x)) with respect to x
        # d(sqrt(|x|))/dx = sign(x)/(2*sqrt(|x|))
        for i in range(batch_size):
            # x coordinate
            if abs_x[i, 0] > 0:
                sign_x = torch.sign(diff_x[i, 0])
                jac[i, 0, 0] = sign_x / (2.0 * torch.sqrt(abs_x[i, 0] + 1e-10))
            
            # y coordinate
            if abs_y[i, 0] > 0:
                sign_y = torch.sign(diff_y[i, 0])
                jac[i, 1, 1] = sign_y / (2.0 * torch.sqrt(abs_y[i, 0] + 1e-10))
        
        # Calculate error
        error = self.error()
        
        return [jac], error
    
    def _copy_impl(self, new_name: Optional[str] = None) -> "LinChkDistLoss":
        """
        Create a copy of this cost function.
        
        Args:
            new_name: Optional new name for the copy
            
        Returns:
            A new instance of this cost function
        """
        return LinChkDistLoss(
            self.point.copy(),
            self.target.copy(),
            self.weight.copy(),
            name=new_name if new_name else self.name
        )