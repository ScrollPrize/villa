"""Tests for SurfaceLossD cost function."""

import unittest

import torch
import theseus as th

from cost_functions import SurfaceLossD


class TestSurfaceLossD(unittest.TestCase):
    """Test cases for SurfaceLossD."""
    # Allow more flexibility in our optimization tests
    tolerance = 1.0
    
    def test_surface_loss_d_error(self):
        """Test the error calculation for SurfaceLossD."""
        batch_size = 1
        
        # Create a simple 3D point matrix (batch x height x width x 3)
        # Each position in the matrix has a 3D point [x, y, z]
        height, width = 5, 5
        matrix = torch.zeros((batch_size, height, width, 3))
        
        # Fill the matrix with some test data
        # Here each element is a 3D point where:
        # x = column index
        # y = row index
        # z = row index + 0.5 * column index
        for y in range(height):
            for x in range(width):
                matrix[0, y, x, 0] = float(x)
                matrix[0, y, x, 1] = float(y)
                matrix[0, y, x, 2] = float(y) + 0.5 * float(x)
        
        # Wrap in a Theseus Variable
        matrix_var = th.Variable(tensor=matrix)
        
        # Create a test location (y=1.5, x=2.0)
        location = th.Point2(tensor=torch.tensor([[1.5, 2.0]]))
        
        # Point to constrain - different from the interpolated value
        point = th.Point3(tensor=torch.tensor([[0.0, 0.0, 0.0]]))
        
        # Cost weight
        cost_weight = th.ScaleCostWeight(1.0)
        
        # Create the cost function
        surface_loss_d = SurfaceLossD(point, location, matrix_var, cost_weight)
        
        # Calculate error - should be the difference between interpolated point and the point
        error = surface_loss_d.error()
        
        # Expected interpolated point at location (1.5, 2.0):
        # x = 2.0 (interpolated from x=2.0 in all rows)
        # y = 1.5 (interpolated from y=1.0 in row 1 and y=2.0 in row 2)
        # z = 1.5 + 0.5*2.0 = 2.5
        # So the error should be [2.0, 1.5, 2.5] - [0.0, 0.0, 0.0] = [2.0, 1.5, 2.5]
        expected_error = torch.tensor([[2.0, 1.5, 2.5]])
        
        # Check error values
        self.assertTrue(torch.allclose(error, expected_error, atol=1e-4),
                        f"Expected {expected_error}, got {error}")
        
        # Test with a different point
        point = th.Point3(tensor=torch.tensor([[1.0, 1.0, 1.0]]))
        surface_loss_d = SurfaceLossD(point, location, matrix_var, cost_weight)
        
        error = surface_loss_d.error()
        # Expected error: [2.0, 1.5, 2.5] - [1.0, 1.0, 1.0] = [1.0, 0.5, 1.5]
        expected_error = torch.tensor([[1.0, 0.5, 1.5]])
        
        # Check error values
        self.assertTrue(torch.allclose(error, expected_error, atol=1e-4),
                        f"Expected {expected_error}, got {error}")
    
    def test_surface_loss_d_jacobians(self):
        """Test the Jacobian calculations for SurfaceLossD."""
        batch_size = 1
        
        # Create a simple 3D point matrix with varying values
        height, width = 5, 5
        matrix = torch.zeros((batch_size, height, width, 3))
        
        # Fill the matrix with test data
        for y in range(height):
            for x in range(width):
                matrix[0, y, x, 0] = float(x)
                matrix[0, y, x, 1] = float(y)
                matrix[0, y, x, 2] = float(y) + 0.5 * float(x)
        
        # Wrap in a Theseus Variable
        matrix_var = th.Variable(tensor=matrix)
        
        # Create a test location (y=1.5, x=2.0)
        location = th.Point2(tensor=torch.tensor([[1.5, 2.0]]))
        
        # Point to constrain
        point = th.Point3(tensor=torch.tensor([[1.0, 1.0, 1.0]]))
        
        # Cost weight
        cost_weight = th.ScaleCostWeight(1.0)
        
        # Create the cost function
        surface_loss_d = SurfaceLossD(point, location, matrix_var, cost_weight)
        
        # Calculate Jacobians
        jacobians, _ = surface_loss_d.jacobians()
        
        # Jacobian for the point should be -I (negative identity matrix)
        expected_jac_point = -torch.eye(3).unsqueeze(0)  # [1, 3, 3]
        
        # Check point Jacobian
        self.assertTrue(torch.allclose(jacobians[0], expected_jac_point, atol=1e-4),
                        f"Expected {expected_jac_point}, got {jacobians[0]}")
        
        # Jacobian for the location:
        # For y coordinate (first column):
        # d(interp.x)/dy = 0 (x doesn't depend on y in our test matrix)
        # d(interp.y)/dy = 1 (y varies linearly with y)
        # d(interp.z)/dy = 1 (z increases by 1 for each unit in y)
        # For x coordinate (second column):
        # d(interp.x)/dx = 1 (x varies linearly with x)
        # d(interp.y)/dx = 0 (y doesn't depend on x in our test matrix)
        # d(interp.z)/dx = 0.5 (z increases by 0.5 for each unit in x)
        expected_jac_location = torch.tensor([[[0.0, 1.0],
                                              [1.0, 0.0],
                                              [1.0, 0.5]]])
        
        # Check location Jacobian
        self.assertTrue(torch.allclose(jacobians[1], expected_jac_location, atol=1e-4),
                        f"Expected {expected_jac_location}, got {jacobians[1]}")
    
    def test_optimization(self):
        """Test that we can optimize a simple problem with SurfaceLossD."""
        batch_size = 1
        
        # Create a simple 3D point matrix
        height, width = 5, 5
        matrix = torch.zeros((batch_size, height, width, 3))
        
        # Fill the matrix with test data
        for y in range(height):
            for x in range(width):
                matrix[0, y, x, 0] = float(x)
                matrix[0, y, x, 1] = float(y)
                matrix[0, y, x, 2] = float(y) + 0.5 * float(x)
        
        # Wrap in a Theseus Variable
        matrix_var = th.Variable(tensor=matrix, name="matrix")
        
        # Initial location
        location = th.Point2(tensor=torch.tensor([[1.5, 2.0]]), name="location")
        
        # Initial point - away from the surface
        point = th.Point3(tensor=torch.tensor([[0.0, 0.0, 0.0]]), name="point")
        
        # Cost weight - using higher weight for stronger optimization
        cost_weight = th.ScaleCostWeight(10.0)
        
        # Create the cost function
        surface_loss_d = SurfaceLossD(point, location, matrix_var, cost_weight)
        
        # Initial error
        initial_error = surface_loss_d.error()
        initial_error_norm = torch.norm(initial_error).item()
        
        # Create an objective
        objective = th.Objective()
        objective.add(surface_loss_d)
        
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
            "point": torch.tensor([[0.0, 0.0, 0.0]]),
            "location": torch.tensor([[1.5, 2.0]]),
            "matrix": matrix
        }
            
        # Optimize
        with torch.no_grad():
            final_values, info = layer.forward(inputs)
        
        # The point should now be closer to the interpolated surface point
        optimized_point = final_values['point']
        optimized_location = final_values['location']
        
        # Create a new loss function with the optimized values
        optimized_surface_loss_d = SurfaceLossD(
            th.Point3(tensor=optimized_point),
            th.Point2(tensor=optimized_location),
            matrix_var,
            cost_weight
        )
        
        # Get the final error
        final_error = optimized_surface_loss_d.error()
        final_error_norm = torch.norm(final_error).item()
        
        # Error should decrease significantly after optimization
        self.assertLess(final_error_norm, initial_error_norm,
                       f"Expected final error norm ({final_error_norm}) to be less than initial error norm ({initial_error_norm})")
        
        # The final error should be close to zero
        self.assertLess(final_error_norm, self.tolerance,
                       f"Expected final error norm ({final_error_norm}) to be less than tolerance ({self.tolerance})")


if __name__ == '__main__':
    unittest.main()