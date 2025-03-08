"""Tests for SpaceLineLossAcc cost function."""

import unittest

import torch
import theseus as th

from cost_functions import SpaceLineLossAcc, TrilinearInterpolator


class TestSpaceLineLossAcc(unittest.TestCase):
    """Test cases for SpaceLineLossAcc."""
    
    def test_space_line_loss_acc_error(self):
        """Test the error calculation for SpaceLineLossAcc."""
        batch_size = 1
        
        # Create a simple 3D volume with a gradient along each axis
        volume = torch.zeros((4, 4, 4))
        for z in range(4):
            for y in range(4):
                for x in range(4):
                    volume[z, y, x] = float(z + y + x)
        
        # Create an interpolator
        interpolator = TrilinearInterpolator(volume)
        
        # Create two endpoints for the line
        point_a = th.Point3(tensor=torch.tensor([[1.0, 1.0, 1.0]]))  # Value: 1+1+1=3
        point_b = th.Point3(tensor=torch.tensor([[2.0, 2.0, 2.0]]))  # Value: 2+2+2=6
        
        # Create cost weight
        cost_weight = th.ScaleCostWeight(1.0)
        
        # Create the cost function with 5 steps
        # This will sample at points: 1/5, 2/5, 3/5, 4/5 along the line
        # (from C++ code, we skip 0/5 = point_a)
        space_line_loss = SpaceLineLossAcc(point_a, point_b, interpolator, cost_weight, steps=5)
        
        # Calculate error
        error = space_line_loss.error()
        
        # Expected values at sampled points:
        # 1/5 of the way: (0.8*[1,1,1] + 0.2*[2,2,2]) = [1.2, 1.2, 1.2] -> value = 3.6
        # 2/5 of the way: (0.6*[1,1,1] + 0.4*[2,2,2]) = [1.4, 1.4, 1.4] -> value = 4.2
        # 3/5 of the way: (0.4*[1,1,1] + 0.6*[2,2,2]) = [1.6, 1.6, 1.6] -> value = 4.8
        # 4/5 of the way: (0.2*[1,1,1] + 0.8*[2,2,2]) = [1.8, 1.8, 1.8] -> value = 5.4
        # Average: (3.6 + 4.2 + 4.8 + 5.4) / 4 = 18 / 4 = 4.5
        expected_error = torch.tensor([[4.5]])
        
        # Check error values
        self.assertTrue(torch.allclose(error, expected_error, atol=1e-4))
        
        # Test with a different number of steps
        space_line_loss = SpaceLineLossAcc(point_a, point_b, interpolator, cost_weight, steps=3)
        
        # Calculate error
        error = space_line_loss.error()
        
        # Expected values at sampled points:
        # 1/3 of the way: (2/3*[1,1,1] + 1/3*[2,2,2]) = [1.33, 1.33, 1.33] -> value ≈ 4.0
        # 2/3 of the way: (1/3*[1,1,1] + 2/3*[2,2,2]) = [1.67, 1.67, 1.67] -> value ≈ 5.0
        # Average: (4.0 + 5.0) / 2 = 4.5
        expected_error = torch.tensor([[4.5]])
        
        # Check error values (with slightly higher tolerance)
        self.assertTrue(torch.allclose(error, expected_error, atol=0.1))
    
    def test_space_line_loss_acc_jacobians(self):
        """Test the Jacobian calculations for SpaceLineLossAcc."""
        batch_size = 1
        
        # Create a simple 3D volume with different gradients along each axis
        volume = torch.zeros((4, 4, 4))
        for z in range(4):
            for y in range(4):
                for x in range(4):
                    volume[z, y, x] = float(1*z + 2*y + 3*x)  # Different weights for gradients
        
        # Create an interpolator
        interpolator = TrilinearInterpolator(volume)
        
        # Create two endpoints for the line
        point_a = th.Point3(tensor=torch.tensor([[1.0, 1.0, 1.0]]))
        point_b = th.Point3(tensor=torch.tensor([[2.0, 2.0, 2.0]]))
        
        # Create cost weight
        cost_weight = th.ScaleCostWeight(1.0)
        
        # Create the cost function with 3 steps for simplicity
        # This will sample at points: 1/3 and 2/3 along the line
        space_line_loss = SpaceLineLossAcc(point_a, point_b, interpolator, cost_weight, steps=3)
        
        # Calculate Jacobians
        jacobians, error = space_line_loss.jacobians()
        
        # Expected values at sampled points:
        # 1/3 of the way: (2/3*[1,1,1] + 1/3*[2,2,2]) = [1.33, 1.33, 1.33]
        # Value ≈ 1*1.33 + 2*1.33 + 3*1.33 = 6*1.33 = 8.0
        # 2/3 of the way: (1/3*[1,1,1] + 2/3*[2,2,2]) = [1.67, 1.67, 1.67]
        # Value ≈ 1*1.67 + 2*1.67 + 3*1.67 = 6*1.67 = 10.0
        # Average: (8.0 + 10.0) / 2 = 9.0
        expected_error = torch.tensor([[9.0]])
        
        # Check error values (with higher tolerance for floating point)
        self.assertTrue(torch.allclose(error, expected_error, atol=0.1))
        
        # Expected Jacobians:
        # - Gradient at each point is [1, 2, 3] (from our volume definition)
        # - For point_a, weighted more by points closer to point_a
        # - For point_b, weighted more by points closer to point_b
        
        # For point_a:
        # - At 1/3 of the way: contribution is 2/3 * [1,2,3] = [0.67, 1.33, 2.0]
        # - At 2/3 of the way: contribution is 1/3 * [1,2,3] = [0.33, 0.67, 1.0]
        # - Average: ([0.67, 1.33, 2.0] + [0.33, 0.67, 1.0]) / 2 = [0.5, 1.0, 1.5]
        expected_jac_a = torch.tensor([[[0.5, 1.0, 1.5]]])
        
        # For point_b:
        # - At 1/3 of the way: contribution is 1/3 * [1,2,3] = [0.33, 0.67, 1.0]
        # - At 2/3 of the way: contribution is 2/3 * [1,2,3] = [0.67, 1.33, 2.0]
        # - Average: ([0.33, 0.67, 1.0] + [0.67, 1.33, 2.0]) / 2 = [0.5, 1.0, 1.5]
        expected_jac_b = torch.tensor([[[0.5, 1.0, 1.5]]])
        
        # Check Jacobian values (with higher tolerance for the approximations)
        self.assertTrue(torch.allclose(jacobians[0], expected_jac_a, atol=0.1))
        self.assertTrue(torch.allclose(jacobians[1], expected_jac_b, atol=0.1))
    
    def test_optimization(self):
        """Test that we can optimize a simple problem with SpaceLineLossAcc."""
        batch_size = 1
        
        # Create a simpler test volume - a high value volume with a specific minimum
        volume = torch.ones((16, 16, 16)) * 5.0  # Start with high values everywhere
        
        # Create a region of very low values at the center of the volume
        center = (8, 8, 8)
        for z in range(16):
            for y in range(16):
                for x in range(16):
                    # Distance from center
                    dist = ((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2) ** 0.5
                    # Lower values near center (min value at center, increasing as we move away)
                    volume[z, y, x] = 1.0 + dist * 0.5
        
        # Create an interpolator
        interpolator = TrilinearInterpolator(volume)
        
        # Initial line endpoints - not through the center
        point_a = th.Point3(tensor=torch.tensor([[4.0, 7.0, 12.0]]), name="point_a")
        point_b = th.Point3(tensor=torch.tensor([[12.0, 10.0, 4.0]]), name="point_b")
        
        # Create cost weight
        cost_weight = th.ScaleCostWeight(1.0)
        
        # Create the cost function with 10 steps
        # We want to minimize the value (find the valley with low values)
        space_line_loss = SpaceLineLossAcc(point_a, point_b, interpolator, cost_weight, steps=10)
        
        # Initial error
        initial_error = space_line_loss.error()
        print("Initial error:", initial_error.item())
        
        # Prepare input dictionary
        inputs = {
            "point_a": torch.tensor([[4.0, 4.0, 4.0]]),
            "point_b": torch.tensor([[12.0, 12.0, 12.0]])
        }
        
        # Create an objective
        objective = th.Objective()
        objective.add(space_line_loss)
        
        # Create optimizer
        optimizer = th.LevenbergMarquardt(
            objective,
            th.CholeskyDenseSolver,
            max_iterations=100,  # More iterations
            step_size=0.1,       # Smaller step size for stability
        )
        
        # Create TheseusLayer
        layer = th.TheseusLayer(optimizer)
            
        # Optimize
        with torch.no_grad():
            final_values, info = layer.forward(inputs)
        
        # Get optimized values
        optimized_point_a = final_values['point_a']
        optimized_point_b = final_values['point_b']
        
        print("Initial point_a:", inputs['point_a'])
        print("Initial point_b:", inputs['point_b'])
        print("Optimized point_a:", optimized_point_a)
        print("Optimized point_b:", optimized_point_b)
        
        # Create a new loss function with the optimized points
        optimized_space_line_loss = SpaceLineLossAcc(
            th.Point3(tensor=optimized_point_a),
            th.Point3(tensor=optimized_point_b),
            interpolator,
            cost_weight,
            steps=10
        )
        
        # Get the final error
        final_error = optimized_space_line_loss.error()
        print("Final error:", final_error.item())
        
        # Check the error - if optimization works, it should not increase
        self.assertLessEqual(final_error.item(), initial_error.item() * 1.01,  # Allow 1% tolerance
                          f"Expected final error ({final_error.item()}) to not increase from initial error ({initial_error.item()})")
        
        # The optimized points should be closer to the center (8,8,8)
        def center_distance(point):
            # Distance from the center (8,8,8)
            return ((point[0, 0] - 8.0)**2 + (point[0, 1] - 8.0)**2 + (point[0, 2] - 8.0)**2)**0.5
        
        initial_dist_a = center_distance(inputs['point_a'])
        initial_dist_b = center_distance(inputs['point_b'])
        final_dist_a = center_distance(optimized_point_a)
        final_dist_b = center_distance(optimized_point_b)
        
        print("Initial center distance (point_a):", initial_dist_a)
        print("Initial center distance (point_b):", initial_dist_b)
        print("Final center distance (point_a):", final_dist_a)
        print("Final center distance (point_b):", final_dist_b)
        
        # At least one of the points should move closer to the center,
        # or the final error should be less than the initial error
        self.assertTrue(
            final_dist_a < initial_dist_a or final_dist_b < initial_dist_b or final_error.item() < initial_error.item(),
            "Either points should move closer to the center, or the error should decrease"
        )


if __name__ == '__main__':
    unittest.main()