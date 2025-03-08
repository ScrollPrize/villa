"""Tests for LinChkDistLoss cost function."""

import unittest

import torch
import theseus as th

from cost_functions import LinChkDistLoss


class TestLinChkDistLoss(unittest.TestCase):
    """Test cases for LinChkDistLoss."""
    # Allow more flexibility in our optimization tests
    tolerance = 1.0
    
    def test_lin_chk_dist_loss_error(self):
        """Test the error calculation for LinChkDistLoss."""
        # Create a 2D point and a target 2D point
        point = th.Point2(tensor=torch.tensor([[3.0, 2.0]]))
        target = th.Point2(tensor=torch.tensor([[1.0, 4.0]]))
        
        # Cost weight
        cost_weight = th.ScaleCostWeight(1.0)
        
        # Create the cost function
        lin_chk_dist_loss = LinChkDistLoss(point, target, cost_weight)
        
        # Calculate error - should be sqrt(abs(diff)) where diff > 0
        error = lin_chk_dist_loss.error()
        
        # Expected: sqrt(|3-1|) = sqrt(2) for x, sqrt(|2-4|) = sqrt(2) for y
        # The C++ implementation only checks if abs(diff) > 0, so both are non-zero
        expected_error = torch.tensor([[torch.sqrt(torch.tensor(2.0)), torch.sqrt(torch.tensor(2.0))]])
        
        # Check error values
        print(f"First test - Actual error: {error}")
        print(f"First test - Expected error: {expected_error}")
        self.assertTrue(torch.allclose(error, expected_error, atol=1e-4))
        
        # Test with a different point where both diffs are positive
        point = th.Point2(tensor=torch.tensor([[5.0, 6.0]]))
        lin_chk_dist_loss = LinChkDistLoss(point, target, cost_weight)
        
        error = lin_chk_dist_loss.error()
        expected_error = torch.tensor([[torch.sqrt(torch.tensor(4.0)), torch.sqrt(torch.tensor(2.0))]])
        
        # Check error values
        print(f"Second test - Actual error: {error}")
        print(f"Second test - Expected error: {expected_error}")
        self.assertTrue(torch.allclose(error, expected_error, atol=1e-4))
    
    def test_lin_chk_dist_loss_jacobians(self):
        """Test the Jacobian calculations for LinChkDistLoss."""
        # Create a 2D point and a target 2D point
        point = th.Point2(tensor=torch.tensor([[3.0, 2.0]]))
        target = th.Point2(tensor=torch.tensor([[1.0, 4.0]]))
        
        # Cost weight
        cost_weight = th.ScaleCostWeight(1.0)
        
        # Create the cost function
        lin_chk_dist_loss = LinChkDistLoss(point, target, cost_weight)
        
        # Calculate Jacobians
        jacobians, _ = lin_chk_dist_loss.jacobians()
        
        # For the x coordinate: d(sqrt(|x-x_t|))/dx = sign(x-x_t) / (2*sqrt(|x-x_t|))
        # For 3-1=2, sign=1, so derivative is 1/(2*sqrt(2))
        # For the y coordinate: for 2-4=-2, sign=-1, so derivative is -1/(2*sqrt(2))
        batch_size = 1
        expected_jac = torch.zeros(batch_size, 2, 2)
        expected_jac[0, 0, 0] = 1.0 / (2.0 * torch.sqrt(torch.tensor(2.0)))
        expected_jac[0, 1, 1] = -1.0 / (2.0 * torch.sqrt(torch.tensor(2.0)))
        
        # Check Jacobian values
        print(f"First Jacobian test - Actual jacobian: {jacobians[0]}")
        print(f"First Jacobian test - Expected jacobian: {expected_jac}")
        self.assertTrue(torch.allclose(jacobians[0], expected_jac, atol=1e-4))
        
        # Test with a different point where both diffs are positive
        point = th.Point2(tensor=torch.tensor([[5.0, 6.0]]))
        lin_chk_dist_loss = LinChkDistLoss(point, target, cost_weight)
        
        jacobians, _ = lin_chk_dist_loss.jacobians()
        
        expected_jac = torch.zeros(batch_size, 2, 2)
        expected_jac[0, 0, 0] = 1.0 / (2.0 * torch.sqrt(torch.tensor(4.0)))
        expected_jac[0, 1, 1] = 1.0 / (2.0 * torch.sqrt(torch.tensor(2.0)))
        
        # Check Jacobian values
        print(f"Second Jacobian test - Actual jacobian: {jacobians[0]}")
        print(f"Second Jacobian test - Expected jacobian: {expected_jac}")
        self.assertTrue(torch.allclose(jacobians[0], expected_jac, atol=1e-4))
    
    def test_optimization(self):
        """Test that we can optimize a simple problem with LinChkDistLoss."""
        # Create a point with initial coordinates different from target
        batch_size = 1
        point = th.Point2(tensor=torch.tensor([[5.0, 6.0]]), name="point")
        target = th.Point2(tensor=torch.tensor([[1.0, 4.0]]), name="target")
        
        # Cost weight - using higher weight for stronger optimization
        cost_weight = th.ScaleCostWeight(10.0)
        
        # Create the cost function
        lin_chk_dist_loss = LinChkDistLoss(point, target, cost_weight)
        
        # Initial error
        initial_error = lin_chk_dist_loss.error()
        initial_error_norm = torch.norm(initial_error).item()
        
        # Create an objective
        objective = th.Objective()
        objective.add(lin_chk_dist_loss)
        
        # Create optimizer
        optimizer = th.LevenbergMarquardt(
            objective,
            th.CholeskyDenseSolver,
            max_iterations=50,
            step_size=1.0,
        )
        
        # Create TheseusLayer
        layer = th.TheseusLayer(optimizer)
        
        # Prepare input dictionary - initialize all variables
        inputs = {
            "point": torch.tensor([[5.0, 6.0]]),
            "target": torch.tensor([[1.0, 4.0]])
        }
            
        # Optimize
        with torch.no_grad():
            final_values, info = layer.forward(inputs)
        
        # The optimized point should be closer to the target
        optimized_point = final_values['point']
        
        # Verify that optimization improved the result
        lin_chk_dist_loss_optimized = LinChkDistLoss(
            th.Point2(tensor=optimized_point), 
            target, 
            cost_weight
        )
        final_error = lin_chk_dist_loss_optimized.error()
        final_error_norm = torch.norm(final_error).item()
        
        # Error should decrease after optimization
        self.assertLess(final_error_norm, initial_error_norm,
                       f"Expected final error ({final_error_norm}) to be less than initial error ({initial_error_norm})")


if __name__ == '__main__':
    unittest.main()