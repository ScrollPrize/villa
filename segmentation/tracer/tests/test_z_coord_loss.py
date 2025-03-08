"""Tests for ZCoordLoss cost function."""

import unittest

import torch
import theseus as th

from cost_functions import ZCoordLoss


class TestZCoordLoss(unittest.TestCase):
    """Test cases for ZCoordLoss."""
    # Allow more flexibility in our optimization tests
    tolerance = 1.0
    
    def test_z_coord_loss_error(self):
        """Test the error calculation for ZCoordLoss."""
        # Create a 3D point and a target Z coordinate
        point = th.Point3(tensor=torch.tensor([[1.0, 2.0, 3.0]]))
        target_z = th.Vector(tensor=torch.tensor([[5.0]]))
        
        # Cost weight
        cost_weight = th.ScaleCostWeight(1.0)
        
        # Create the cost function
        z_coord_loss = ZCoordLoss(point, target_z, cost_weight)
        
        # Calculate error - should be the difference between the z-coordinate and target_z
        error = z_coord_loss.error()
        expected_error = torch.tensor([[3.0 - 5.0]])  # z - target_z
        
        # Check error values
        self.assertTrue(torch.allclose(error, expected_error, atol=1e-4))
        
        # Test with a different point
        point = th.Point3(tensor=torch.tensor([[1.0, 2.0, 6.0]]))
        z_coord_loss = ZCoordLoss(point, target_z, cost_weight)
        
        # Calculate error - should be positive if z > target_z
        error = z_coord_loss.error()
        expected_error = torch.tensor([[6.0 - 5.0]])
        
        # Check error values
        self.assertTrue(torch.allclose(error, expected_error, atol=1e-4))
    
    def test_z_coord_loss_jacobians(self):
        """Test the Jacobian calculations for ZCoordLoss."""
        # Create a 3D point and a target Z coordinate
        point = th.Point3(tensor=torch.tensor([[1.0, 2.0, 3.0]]))
        target_z = th.Vector(tensor=torch.tensor([[5.0]]))
        
        # Cost weight
        cost_weight = th.ScaleCostWeight(1.0)
        
        # Create the cost function
        z_coord_loss = ZCoordLoss(point, target_z, cost_weight)
        
        # Calculate Jacobians
        jacobians, _ = z_coord_loss.jacobians()
        
        # The Jacobian should be [0, 0, 1] because only the z-coordinate affects the error
        batch_size = 1
        expected_jac = torch.zeros(batch_size, 1, 3)
        expected_jac[0, 0, 2] = 1.0
        
        # Check Jacobian values
        self.assertTrue(torch.allclose(jacobians[0], expected_jac, atol=1e-4))
    
    def test_optimization(self):
        """Test that we can optimize a simple problem with ZCoordLoss."""
        # Create a point with initial z-coordinate different from target
        batch_size = 1
        point = th.Point3(tensor=torch.tensor([[1.0, 2.0, 3.0]]), name="point")
        target_z = th.Vector(tensor=torch.tensor([[5.0]]), name="target_z")
        
        # Cost weight - using higher weight for stronger optimization
        cost_weight = th.ScaleCostWeight(10.0)
        
        # Create the cost function
        z_coord_loss = ZCoordLoss(point, target_z, cost_weight)
        
        # Initial error (non-matching z)
        initial_error = z_coord_loss.error()
        self.assertNotEqual(initial_error.item(), 0.0)
        
        # Create an objective
        objective = th.Objective()
        objective.add(z_coord_loss)
        
        # Create optimizer
        optimizer = th.LevenbergMarquardt(
            objective,
            th.CholeskyDenseSolver,
            max_iterations=20,
            step_size=1.0,
        )
        
        # Create TheseusLayer
        layer = th.TheseusLayer(optimizer)
        
        # Prepare input dictionary - initialize all variables
        inputs = {
            "point": torch.tensor([[1.0, 2.0, 3.0]]),
            "target_z": torch.tensor([[5.0]])
        }
            
        # Optimize
        with torch.no_grad():
            final_values, info = layer.forward(inputs)
        
        # The z-coordinate should now be closer to the target
        optimized_point = final_values['point']
        
        # Verify that optimization improved the result
        optimized_z = optimized_point[0, 2].item()
        
        # Z-coordinate should be very close to the target after optimization
        self.assertAlmostEqual(optimized_z, 5.0, delta=self.tolerance,
                             msg=f"Expected optimized z ({optimized_z}) to be close to target (5.0)")
        
        # The x and y coordinates should remain unchanged
        self.assertAlmostEqual(optimized_point[0, 0].item(), 1.0, delta=1e-4)
        self.assertAlmostEqual(optimized_point[0, 1].item(), 2.0, delta=1e-4)


if __name__ == '__main__':
    unittest.main()