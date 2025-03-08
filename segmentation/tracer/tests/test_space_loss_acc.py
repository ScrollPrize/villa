"""Tests for SpaceLossAcc cost function."""

import unittest

import torch
import theseus as th

from cost_functions import SpaceLossAcc, TrilinearInterpolator


class TestSpaceLossAcc(unittest.TestCase):
    """Test cases for SpaceLossAcc."""
    # Allow more flexibility in our optimization tests
    tolerance = 1.0
    
    def test_space_loss_acc_error(self):
        """Test the error calculation for SpaceLossAcc."""
        batch_size = 1
        
        # Create a simple 3D volume with gradients
        volume = torch.zeros((4, 4, 4))
        for z in range(4):
            for y in range(4):
                for x in range(4):
                    volume[z, y, x] = float(z + y + x)
        
        # Create an interpolator
        interpolator = TrilinearInterpolator(volume)
        
        # Create a 3D point
        point = th.Point3(tensor=torch.tensor([[1.5, 1.5, 1.5]]))
        
        # Cost weight
        cost_weight = th.ScaleCostWeight(1.0)
        
        # Create the cost function
        space_loss = SpaceLossAcc(point, interpolator, cost_weight)
        
        # Calculate error - should be the interpolated value at the point
        error = space_loss.error()
        
        # Expected: interpolated value at (1.5, 1.5, 1.5) = 1.5 + 1.5 + 1.5 = 4.5
        expected_error = torch.tensor([[4.5]])
        
        # Check error values
        self.assertTrue(torch.allclose(error, expected_error, atol=1e-4))
        
        # Test with a different point
        point = th.Point3(tensor=torch.tensor([[2.5, 0.5, 1.0]]))
        space_loss = SpaceLossAcc(point, interpolator, cost_weight)
        
        error = space_loss.error()
        # Expected: interpolated value at (2.5, 0.5, 1.0) = 2.5 + 0.5 + 1.0 = 4.0
        expected_error = torch.tensor([[4.0]])
        
        # Check error values
        self.assertTrue(torch.allclose(error, expected_error, atol=1e-4))
    
    def test_space_loss_acc_jacobians(self):
        """Test the Jacobian calculations for SpaceLossAcc."""
        batch_size = 1
        
        # Create a simple 3D volume with different gradients along each axis
        volume = torch.zeros((4, 4, 4))
        for z in range(4):
            for y in range(4):
                for x in range(4):
                    volume[z, y, x] = float(1*z + 2*y + 3*x)  # Different weights for gradients
        
        # Create an interpolator
        interpolator = TrilinearInterpolator(volume)
        
        # Create a 3D point
        point = th.Point3(tensor=torch.tensor([[1.5, 1.5, 1.5]]))
        
        # Cost weight
        cost_weight = th.ScaleCostWeight(1.0)
        
        # Create the cost function
        space_loss = SpaceLossAcc(point, interpolator, cost_weight)
        
        # Calculate Jacobians
        jacobians, _ = space_loss.jacobians()
        
        # Expected gradients: dv/dz = 1, dv/dy = 2, dv/dx = 3
        expected_jac = torch.tensor([[[1.0, 2.0, 3.0]]])
        
        # Check Jacobian values
        self.assertTrue(torch.allclose(jacobians[0], expected_jac, atol=1e-4))
    
    def test_optimization(self):
        """Test that we can optimize a simple problem with SpaceLossAcc."""
        batch_size = 1
        
        # Create a simple 3D volume - with value INCREASING with distance from the origin (0,0,0)
        # This is different from our previous sphere to make it easier to see the optimization direction
        volume = torch.zeros((16, 16, 16))
        for z in range(16):
            for y in range(16):
                for x in range(16):
                    # Note: In our implementation [z, y, x] is the coordinate order
                    pos = torch.tensor([float(z), float(y), float(x)])
                    # Value increases with distance from origin
                    dist = torch.norm(pos)
                    volume[z, y, x] = dist  # Greater distance = greater value
        
        # Create an interpolator
        interpolator = TrilinearInterpolator(volume)
        
        # Initial point - closer to the origin
        # Note: Order is [z, y, x] - different from the volume creation loop
        point = th.Point3(tensor=torch.tensor([[4.0, 4.0, 4.0]]), name="point")
        
        # Goal: Move away from origin (maximize the distance = maximize the value)
        cost_weight = th.ScaleCostWeight(1.0)
        
        # Create the cost function with maximize=True to explicitly seek maximum values
        space_loss = SpaceLossAcc(point, interpolator, cost_weight, maximize=True)
        
        # Initial error
        initial_error = space_loss.error().item()
        
        # Prepare input dictionary
        inputs = {
            "point": torch.tensor([[4.0, 4.0, 4.0]])
        }
        
        # Get value and gradient at initial point
        z = inputs['point'][:, 2:3]
        y = inputs['point'][:, 1:2]
        x = inputs['point'][:, 0:1]
        initial_value, initial_gradient = interpolator.evaluate_with_gradient(z, y, x)
        print("Initial value at point:", initial_value.item())
        print("Initial gradient at point:", initial_gradient.squeeze())
        
        # Create an objective
        objective = th.Objective()
        objective.add(space_loss)
        
        # Create optimizer
        optimizer = th.LevenbergMarquardt(
            objective,
            th.CholeskyDenseSolver,
            max_iterations=50,
            step_size=1.0,
        )
        
        # Create TheseusLayer
        layer = th.TheseusLayer(optimizer)
        
        # Input dictionary was already prepared above
            
        # Print the initial point
        print("Initial point:", inputs['point'])
        
        # Optimize
        with torch.no_grad():
            final_values, info = layer.forward(inputs)
        
        # The point should move toward the center of the volume
        optimized_point = final_values['point']
        print("Optimized point:", optimized_point)
        
        # Create a new loss function with the optimized point
        optimized_space_loss = SpaceLossAcc(
            th.Point3(tensor=optimized_point),
            interpolator,
            cost_weight
        )
        
        # Get the final error
        final_error = optimized_space_loss.error().item()
        
        # Get value and gradient at optimized point
        z = optimized_point[:, 2:3]
        y = optimized_point[:, 1:2]
        x = optimized_point[:, 0:1]
        final_value, final_gradient = interpolator.evaluate_with_gradient(z, y, x)
        print("Final value at point:", final_value.item())
        print("Final gradient at point:", final_gradient.squeeze())
        
        print("Initial error (with weight):", initial_error)
        print("Final error (with weight):", final_error)
        
        # Since we're using maximize=True, the error should be HIGHER after optimization
        # This means the value at the optimized point has increased
        self.assertGreater(final_error, initial_error,
                         f"Expected final error ({final_error}) to be greater than initial error ({initial_error})")
        
        # For distance calculation from origin
        origin = torch.tensor([[0.0, 0.0, 0.0]])
        optimized_distance = torch.norm(optimized_point).item()
        initial_distance = torch.norm(torch.tensor([4.0, 4.0, 4.0])).item()
        
        print("Origin:", [0.0, 0.0, 0.0])
        print("Distance from initial point to origin:", initial_distance)
        print("Distance from optimized point to origin:", optimized_distance)
        
        # The optimized point should be further from the origin (higher value)
        self.assertGreater(optimized_distance, initial_distance,
                         f"Expected distance to origin ({optimized_distance}) to be greater than initial distance ({initial_distance})")


if __name__ == '__main__':
    unittest.main()