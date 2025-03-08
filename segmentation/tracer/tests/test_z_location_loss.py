"""Tests for ZLocationLoss cost function."""

import unittest

import torch
import theseus as th

from cost_functions import ZLocationLoss


class TestZLocationLoss(unittest.TestCase):
    """Test cases for ZLocationLoss."""
    # Allow more flexibility in our optimization tests
    tolerance = 1.0
    
    def test_z_location_loss_error(self):
        """Test the error calculation for ZLocationLoss."""
        batch_size = 1
        
        # Create a simple 3D point matrix (batch x height x width x 3)
        # Each position in the matrix has a 3D point [x, y, z]
        # where z varies based on position
        height, width = 5, 5
        matrix = torch.zeros((batch_size, height, width, 3))
        
        # Fill the matrix with some test data
        # Here the z-value varies linearly from 0 to 4 along the y-axis
        for y in range(height):
            for x in range(width):
                matrix[0, y, x, 2] = float(y)  # z increases with y
        
        # Wrap in a Theseus Variable
        matrix_var = th.Variable(tensor=matrix)
        
        # Create a test location (y=1.5, x=2.0) - should interpolate to z=1.5
        location = th.Point2(tensor=torch.tensor([[1.5, 2.0]]))
        
        # Target z-coordinate
        target_z = th.Vector(tensor=torch.tensor([[3.0]]))
        
        # Cost weight
        cost_weight = th.ScaleCostWeight(1.0)
        
        # Create the cost function
        z_location_loss = ZLocationLoss(location, matrix_var, target_z, cost_weight)
        
        # Calculate error - should be the difference between interpolated z and target_z
        error = z_location_loss.error()
        
        # Expected: interpolated z = 1.5, target_z = 3.0, so error = 1.5 - 3.0 = -1.5
        expected_error = torch.tensor([[-1.5]])
        
        # Check error values
        self.assertTrue(torch.allclose(error, expected_error, atol=1e-4))
        
        # Test with a different location for interpolation
        location = th.Point2(tensor=torch.tensor([[2.5, 1.0]]))
        z_location_loss = ZLocationLoss(location, matrix_var, target_z, cost_weight)
        
        error = z_location_loss.error()
        # Expected: interpolated z = 2.5, target_z = 3.0, so error = 2.5 - 3.0 = -0.5
        expected_error = torch.tensor([[-0.5]])
        
        # Check error values
        self.assertTrue(torch.allclose(error, expected_error, atol=1e-4))
    
    def test_z_location_loss_jacobians(self):
        """Test the Jacobian calculations for ZLocationLoss."""
        batch_size = 1
        
        # Create a simple 3D point matrix with varying z-values
        height, width = 5, 5
        matrix = torch.zeros((batch_size, height, width, 3))
        
        # Fill the matrix - z increases with y and doesn't change with x
        for y in range(height):
            for x in range(width):
                matrix[0, y, x, 2] = float(y)
        
        # Wrap in a Theseus Variable
        matrix_var = th.Variable(tensor=matrix)
        
        # Create a test location (y=1.5, x=2.0)
        location = th.Point2(tensor=torch.tensor([[1.5, 2.0]]))
        
        # Target z-coordinate
        target_z = th.Vector(tensor=torch.tensor([[3.0]]))
        
        # Cost weight
        cost_weight = th.ScaleCostWeight(1.0)
        
        # Create the cost function
        z_location_loss = ZLocationLoss(location, matrix_var, target_z, cost_weight)
        
        # Calculate Jacobians
        jacobians, _ = z_location_loss.jacobians()
        
        # Since z only varies with y, the derivative dz/dy should be 1.0
        # and the derivative dz/dx should be 0.0
        expected_jac = torch.zeros(batch_size, 1, 2)
        expected_jac[0, 0, 0] = 1.0  # dz/dy = 1.0
        expected_jac[0, 0, 1] = 0.0  # dz/dx = 0.0
        
        # Check Jacobian values
        self.assertTrue(torch.allclose(jacobians[0], expected_jac, atol=1e-4))
        
        # Create a more complex matrix where z varies in both x and y directions
        matrix = torch.zeros((batch_size, height, width, 3))
        for y in range(height):
            for x in range(width):
                matrix[0, y, x, 2] = float(y) + 0.5 * float(x)
        
        # Update the matrix variable
        matrix_var = th.Variable(tensor=matrix)
        
        # Create a new cost function with the updated matrix
        z_location_loss = ZLocationLoss(location, matrix_var, target_z, cost_weight)
        
        # Calculate Jacobians
        jacobians, _ = z_location_loss.jacobians()
        
        # Now z varies with both y and x
        # dz/dy = 1.0
        # dz/dx = 0.5
        expected_jac = torch.zeros(batch_size, 1, 2)
        expected_jac[0, 0, 0] = 1.0  # dz/dy = 1.0
        expected_jac[0, 0, 1] = 0.5  # dz/dx = 0.5
        
        # Check Jacobian values
        self.assertTrue(torch.allclose(jacobians[0], expected_jac, atol=1e-4))
    
    def test_optimization(self):
        """Test that we can optimize a simple problem with ZLocationLoss."""
        batch_size = 1
        
        # Create a simple 3D point matrix
        height, width = 5, 5
        matrix = torch.zeros((batch_size, height, width, 3))
        
        # Fill the matrix - z varies with position
        for y in range(height):
            for x in range(width):
                matrix[0, y, x, 2] = float(y) + 0.5 * float(x)
        
        # Wrap in a Theseus Variable
        matrix_var = th.Variable(tensor=matrix, name="matrix")
        
        # Initial location
        location = th.Point2(tensor=torch.tensor([[1.0, 1.0]]), name="location")
        
        # Target z-coordinate - z=3.0
        target_z = th.Vector(tensor=torch.tensor([[3.0]]), name="target_z")
        
        # Cost weight - using higher weight for stronger optimization
        cost_weight = th.ScaleCostWeight(10.0)
        
        # Create the cost function
        z_location_loss = ZLocationLoss(location, matrix_var, target_z, cost_weight)
        
        # Initial error
        initial_error = z_location_loss.error()
        
        # Create an objective
        objective = th.Objective()
        objective.add(z_location_loss)
        
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
            "location": torch.tensor([[1.0, 1.0]]),
            "matrix": matrix,
            "target_z": torch.tensor([[3.0]])
        }
            
        # Optimize
        with torch.no_grad():
            final_values, info = layer.forward(inputs)
        
        # The location should now give a z-value closer to the target
        optimized_location = final_values['location']
        
        # Create a new loss function with the optimized location
        optimized_z_location_loss = ZLocationLoss(
            th.Point2(tensor=optimized_location),
            matrix_var,
            target_z,
            cost_weight
        )
        
        # Get the final error
        final_error = optimized_z_location_loss.error()
        
        # Error should decrease significantly after optimization
        self.assertLess(torch.abs(final_error).item(), torch.abs(initial_error).item(),
                       f"Expected final error magnitude ({torch.abs(final_error).item()}) to be less than initial error magnitude ({torch.abs(initial_error).item()})")
        
        # The final interpolated z should be close to the target z
        self.assertLess(torch.abs(final_error).item(), self.tolerance,
                       f"Expected final error magnitude ({torch.abs(final_error).item()}) to be less than tolerance ({self.tolerance})")


if __name__ == '__main__':
    unittest.main()