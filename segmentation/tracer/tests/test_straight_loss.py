"""Tests for StraightLoss cost function."""

import unittest

import torch
import theseus as th

from cost_functions import StraightLoss


class TestStraightLoss(unittest.TestCase):
    """Test cases for StraightLoss."""
    # Allow more flexibility in our optimization tests
    tolerance = 1.0
    
    def test_straight_loss_error(self):
        """Test the error calculation for StraightLoss."""
        # Create optimization points - a straight line along the x-axis
        point_a = th.Point3(tensor=torch.tensor([[0.0, 0.0, 0.0]]))
        point_b = th.Point3(tensor=torch.tensor([[1.0, 0.0, 0.0]]))
        point_c = th.Point3(tensor=torch.tensor([[2.0, 0.0, 0.0]]))
        
        # Cost weight
        cost_weight = th.ScaleCostWeight(1.0)
        
        # Create the cost function
        straight_loss = StraightLoss(point_a, point_b, point_c, cost_weight)
        
        # Calculate error - should be 0 for perfectly straight line
        error = straight_loss.error()
        expected_error = torch.tensor([[0.0]])
        
        # Check error values
        self.assertTrue(torch.allclose(error, expected_error, atol=1e-4))
        
        # Now create a non-straight configuration
        point_b = th.Point3(tensor=torch.tensor([[1.0, 1.0, 0.0]]))
        straight_loss = StraightLoss(point_a, point_b, point_c, cost_weight)
        
        # Calculate error - should be positive for non-straight line
        error = straight_loss.error()
        
        # The dot product of normalized vectors should be less than 1
        self.assertTrue(error.item() > 0)
    
    def test_straight_loss_jacobians(self):
        """Test the Jacobian calculations for StraightLoss."""
        # Create optimization points - points in a straight line
        point_a = th.Point3(tensor=torch.tensor([[0.0, 0.0, 0.0]]))
        point_b = th.Point3(tensor=torch.tensor([[1.0, 0.0, 0.0]]))
        point_c = th.Point3(tensor=torch.tensor([[2.0, 0.0, 0.0]]))
        
        # Cost weight
        cost_weight = th.ScaleCostWeight(1.0)
        
        # Create the cost function
        straight_loss = StraightLoss(point_a, point_b, point_c, cost_weight)
        
        # Calculate Jacobians
        jacobians, _ = straight_loss.jacobians()
        
        # For a straight line, the gradient should be 0
        batch_size = 1
        expected_jac_b = torch.zeros(batch_size, 1, 3)
        
        # Check Jacobian values - now only checking point_b's jacobian
        self.assertTrue(torch.allclose(jacobians[0], expected_jac_b, atol=1e-4))
    
    def test_optimization(self):
        """Test that we can optimize a simple problem with StraightLoss."""
        # Create optimization points - a non-straight configuration
        batch_size = 1
        # Point b is optimizable, points a and c are fixed
        point_a = th.Point3(tensor=torch.tensor([[0.0, 0.0, 0.0]]), name="point_a")
        point_b = th.Point3(tensor=torch.tensor([[1.0, 1.0, 0.0]]), name="point_b")  # out of line
        point_c = th.Point3(tensor=torch.tensor([[2.0, 0.0, 0.0]]), name="point_c")
        
        # Cost weight - using higher weight for stronger optimization
        cost_weight = th.ScaleCostWeight(10.0)
        
        # Create the cost function
        straight_loss = StraightLoss(point_a, point_b, point_c, cost_weight)
        
        # Initial error (non-straight line)
        initial_error = straight_loss.error()
        self.assertGreater(initial_error.item(), 0.01)
        
        # Create an objective
        objective = th.Objective()
        objective.add(straight_loss)
        
        # Create optimizer with more iterations and larger step size
        optimizer = th.LevenbergMarquardt(
            objective,
            th.CholeskyDenseSolver,
            max_iterations=100,
            step_size=1.0,
        )
        
        # Create TheseusLayer
        layer = th.TheseusLayer(optimizer)
        
        # Prepare input dictionary - initialize all variables
        inputs = {
            "point_a": torch.tensor([[0.0, 0.0, 0.0]]),
            "point_b": torch.tensor([[1.0, 1.0, 0.0]]),
            "point_c": torch.tensor([[2.0, 0.0, 0.0]])
        }
            
        # Optimize - this will only optimize point_b as it's the only one we registered
        # as an optimization variable in the StraightLoss constructor
        with torch.no_grad():
            final_values, info = layer.forward(inputs)
        
        # point_b should now be closer to the line defined by point_a and point_c
        optimized_point_b = final_values['point_b']
        
        # Verify that optimization improved the result
        b_tensor = th.Point3(tensor=optimized_point_b)
        optimized_loss = StraightLoss(point_a, b_tensor, point_c, cost_weight)
        final_error = optimized_loss.error()
        
        # Error should decrease significantly after optimization
        self.assertLess(final_error.item(), initial_error.item(), 
                       f"Expected final error ({final_error.item()}) to be less than initial error ({initial_error.item()})")
        
        # Calculate the y-value of optimized_point_b
        # It should be closer to 0 (which would make it a straight line)
        initial_y = inputs['point_b'][0, 1].item()
        optimized_y = optimized_point_b[0, 1].item()
        
        # The optimized y-coordinate should be closer to 0
        self.assertLess(abs(optimized_y), abs(initial_y),
                       f"Expected optimized y-value ({optimized_y}) to be closer to 0 than initial y-value ({initial_y})")


if __name__ == '__main__':
    unittest.main()