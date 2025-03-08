"""Tests for StraightLoss2 cost function."""

import unittest

import torch
import theseus as th

from cost_functions import StraightLoss2


class TestStraightLoss2(unittest.TestCase):
    """Test cases for StraightLoss2."""
    
    def test_straight_loss2_error(self):
        """Test the error calculation for StraightLoss2."""
        # Create optimization points - a straight line along the x-axis
        point_a = th.Point3(tensor=torch.tensor([[0.0, 0.0, 0.0]]))
        point_b = th.Point3(tensor=torch.tensor([[1.0, 0.0, 0.0]]))
        point_c = th.Point3(tensor=torch.tensor([[2.0, 0.0, 0.0]]))
        
        # Cost weight
        cost_weight = th.ScaleCostWeight(1.0)
        
        # Create the cost function
        straight_loss = StraightLoss2(point_a, point_b, point_c, cost_weight)
        
        # Calculate error
        error = straight_loss.error()
        
        # The error should be the vector from point_b to the midpoint of point_a and point_c
        # Midpoint of [0,0,0] and [2,0,0] is [1,0,0], error is [1,0,0] - [1,0,0] = [0,0,0]
        expected_error = torch.tensor([[0.0, 0.0, 0.0]])
        
        # Check error values
        self.assertTrue(torch.allclose(error, expected_error, atol=1e-4))
        
        # Now create a non-straight configuration
        point_b = th.Point3(tensor=torch.tensor([[1.0, 1.0, 0.0]]))
        straight_loss = StraightLoss2(point_a, point_b, point_c, cost_weight)
        
        # Calculate error - should be non-zero for non-straight line
        error = straight_loss.error()
        
        # Error should be [0,1,0] (difference between [1,1,0] and midpoint [1,0,0])
        expected_error = torch.tensor([[0.0, 1.0, 0.0]])
        self.assertTrue(torch.allclose(error, expected_error, atol=1e-4))
    
    def test_straight_loss2_jacobians(self):
        """Test the Jacobian calculations for StraightLoss2."""
        # Create optimization points
        point_a = th.Point3(tensor=torch.tensor([[0.0, 0.0, 0.0]]))
        point_b = th.Point3(tensor=torch.tensor([[1.0, 0.0, 0.0]]))
        point_c = th.Point3(tensor=torch.tensor([[2.0, 0.0, 0.0]]))
        
        # Cost weight
        cost_weight = th.ScaleCostWeight(1.0)
        
        # Create the cost function
        straight_loss = StraightLoss2(point_a, point_b, point_c, cost_weight)
        
        # Calculate Jacobians
        jacobians, _ = straight_loss.jacobians()
        
        # Expected Jacobians
        batch_size = 1
        
        # Jacobian for point_b: Identity
        expected_jac_b = torch.zeros(batch_size, 3, 3)
        expected_jac_b[0] = torch.eye(3)
        
        # Check Jacobian values - only checking point_b's jacobian
        self.assertTrue(torch.allclose(jacobians[0], expected_jac_b, atol=1e-4))
    
    def test_optimization(self):
        """Test that we can optimize a simple problem with StraightLoss2."""
        # Create optimization points - a non-straight configuration
        batch_size = 1
        # Point b is optimizable, points a and c are fixed
        point_a = th.Point3(tensor=torch.tensor([[0.0, 0.0, 0.0]]), name="point_a")
        point_b = th.Point3(tensor=torch.tensor([[1.0, 1.0, 0.0]]), name="point_b")  # out of line
        point_c = th.Point3(tensor=torch.tensor([[2.0, 0.0, 0.0]]), name="point_c")
        
        # Calculate initial error - distance from midpoint
        midpoint = (point_a.tensor + point_c.tensor) * 0.5
        initial_distance = torch.norm(point_b.tensor - midpoint).item()
        
        # Cost weight - using higher weight for stronger optimization
        cost_weight = th.ScaleCostWeight(10.0)
        
        # Create the cost function
        straight_loss = StraightLoss2(point_a, point_b, point_c, cost_weight)
        
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
        # as an optimization variable in the StraightLoss2 constructor
        with torch.no_grad():
            final_values, info = layer.forward(inputs)
        
        # point_b should now be closer to the midpoint of a and c
        optimized_point_b = final_values['point_b']
        
        # Calculate final distance to midpoint
        final_distance = torch.norm(optimized_point_b - midpoint).item()
        
        # The distance to midpoint should decrease
        self.assertLess(final_distance, initial_distance,
                       f"Expected final distance to midpoint ({final_distance}) to be less than initial distance ({initial_distance})")
        
        # We expect the optimized y-coordinate to be closer to 0
        initial_y = inputs['point_b'][0, 1].item()
        optimized_y = optimized_point_b[0, 1].item()
        
        # The optimized y-coordinate should be closer to 0
        self.assertLess(abs(optimized_y), abs(initial_y),
                       f"Expected optimized y-value ({optimized_y}) to be closer to 0 than initial y-value ({initial_y})")


if __name__ == '__main__':
    unittest.main()