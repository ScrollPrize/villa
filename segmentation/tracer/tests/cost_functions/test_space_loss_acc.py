"""Tests for SpaceLossAcc and SpaceLossAccAutoDiff cost functions."""

import unittest
import numpy as np
import torch
import theseus as th

from cost_functions import SpaceLossAcc, TrilinearInterpolator
from cost_functions.autodiff.space_loss_acc_autodiff import SpaceLossAccAutoDiff
from tracer.core.interpolation import TrilinearInterpolatorAutoDiff
from tests.test_utils import TestCaseWithFullStackTrace


class TestSpaceLossAcc(TestCaseWithFullStackTrace):
    """Test cases for SpaceLossAcc and SpaceLossAccAutoDiff."""
    # Allow more flexibility in our optimization tests
    tolerance = 1.0
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple 3D test volume
        batch_size = 2
        self.volume_shape = (batch_size, 4, 4, 4)
        
        # Create a 3D volume with gradients
        self.test_volume = torch.zeros(self.volume_shape, dtype=torch.float32)
        
        # Fill with a pattern where value increases with distance from center
        for b in range(batch_size):
            for z in range(4):
                for y in range(4):
                    for x in range(4):
                        # Simple gradient: value = z + y + x
                        self.test_volume[b, z, y, x] = float(z + y + x)
        
        # Create a 3D volume with different gradients along each axis
        self.gradient_volume = torch.zeros(self.volume_shape, dtype=torch.float32)
        for b in range(batch_size):
            for z in range(4):
                for y in range(4):
                    for x in range(4):
                        # Different weights for gradients
                        self.gradient_volume[b, z, y, x] = float(1*z + 2*y + 3*x)
        
        # Create a volume with values increasing with distance from origin
        self.distance_volume = torch.zeros((batch_size, 16, 16, 16), dtype=torch.float32)
        for b in range(batch_size):
            for z in range(16):
                for y in range(16):
                    for x in range(16):
                        # Note: In our implementation [z, y, x] is the coordinate order
                        pos = torch.tensor([float(z), float(y), float(x)])
                        # Value increases with distance from origin
                        dist = torch.norm(pos)
                        self.distance_volume[b, z, y, x] = dist  # Greater distance = greater value
        
        # Create the interpolators
        self.interpolator = TrilinearInterpolator(self.test_volume)
        self.autodiff_interpolator = TrilinearInterpolatorAutoDiff(self.test_volume)
        
        self.gradient_interpolator = TrilinearInterpolator(self.gradient_volume)
        self.gradient_autodiff_interpolator = TrilinearInterpolatorAutoDiff(self.gradient_volume)
        
        self.distance_interpolator = TrilinearInterpolator(self.distance_volume)
        self.distance_autodiff_interpolator = TrilinearInterpolatorAutoDiff(self.distance_volume)
        
        # Create test points
        self.point_tensor = torch.tensor([
            [1.5, 1.5, 1.5],  # Batch item 0
            [2.5, 0.5, 1.0]   # Batch item 1
        ], dtype=torch.float32)
        self.point = th.Point3(tensor=self.point_tensor)
        
        # Create cost weight
        self.cost_weight = th.ScaleCostWeight(1.0)
    
    def create_cost_functions(self, point, interpolator, autodiff_interpolator, 
                             cost_weight, maximize=False):
        """Create both regular and autodiff versions of the cost function."""
        manual_cost = SpaceLossAcc(
            point, 
            interpolator, 
            cost_weight,
            maximize=maximize
        )
        autodiff_cost = SpaceLossAccAutoDiff(
            point, 
            autodiff_interpolator, 
            cost_weight,
            maximize=maximize
        )
        return manual_cost, autodiff_cost
    
    def test_interpolator_values_match(self):
        """Test that interpolator values match between regular and autodiff implementations."""
        # Create test inputs
        z = self.point_tensor[:, 0:1]
        y = self.point_tensor[:, 1:2]
        x = self.point_tensor[:, 2:3]
        
        # Compute interpolated values
        regular_values = self.interpolator.evaluate(z, y, x)
        autodiff_values = self.autodiff_interpolator.evaluate(z, y, x)
        
        # Check that values match
        self.assertTrue(torch.allclose(regular_values, autodiff_values, atol=1e-6))
    
    def test_error_values(self):
        """Test that the error values are correct and match between implementations."""
        # Create the cost functions
        manual_cost, autodiff_cost = self.create_cost_functions(
            self.point, self.interpolator, self.autodiff_interpolator, self.cost_weight
        )
        
        # Compute errors
        manual_error = manual_cost.error()
        autodiff_error = autodiff_cost.error()
        
        # Expected values: interpolated values at the test points
        # Point 1: (1.5, 1.5, 1.5) = 1.5 + 1.5 + 1.5 = 4.5
        # Point 2: (2.5, 0.5, 1.0) = 2.5 + 0.5 + 1.0 = 4.0
        expected_error = torch.tensor([[4.5], [4.0]])
        
        # Check manual error values
        self.assertTrue(torch.allclose(manual_error, expected_error, atol=1e-4),
                        f"Manual error {manual_error} doesn't match expected {expected_error}")
        
        # Check autodiff error values
        self.assertTrue(torch.allclose(autodiff_error, expected_error, atol=1e-4),
                        f"Autodiff error {autodiff_error} doesn't match expected {expected_error}")
        
        # Check that implementations match
        self.assertTrue(torch.allclose(manual_error, autodiff_error, atol=1e-6),
                        f"Manual error {manual_error} doesn't match autodiff error {autodiff_error}")
    
    def test_jacobians(self):
        """Test that the Jacobians have the expected values and match between implementations."""
        # Create the cost functions with gradient volume
        manual_cost, autodiff_cost = self.create_cost_functions(
            self.point, self.gradient_interpolator, self.gradient_autodiff_interpolator, 
            self.cost_weight
        )
        
        # Compute Jacobians
        manual_jacs, _ = manual_cost.jacobians()
        autodiff_jacs, _ = autodiff_cost.jacobians()
        
        # Expected gradients: dv/dz = 1, dv/dy = 2, dv/dx = 3
        expected_jac = torch.tensor([
            [[1.0, 2.0, 3.0]],  # For point 1
            [[1.0, 2.0, 3.0]]   # For point 2
        ])
        
        # Check manual Jacobian values
        self.assertTrue(torch.allclose(manual_jacs[0], expected_jac, atol=1e-4),
                        f"Manual Jacobian {manual_jacs[0]} doesn't match expected {expected_jac}")
        
        # Check autodiff Jacobian values
        self.assertTrue(torch.allclose(autodiff_jacs[0], expected_jac, atol=1e-4),
                        f"Autodiff Jacobian {autodiff_jacs[0]} doesn't match expected {expected_jac}")
        
        # Check that implementations match
        self.assertTrue(torch.allclose(manual_jacs[0], autodiff_jacs[0], atol=1e-6),
                        f"Manual Jacobian {manual_jacs[0]} doesn't match autodiff Jacobian {autodiff_jacs[0]}")
    
    def test_maximize_mode(self):
        """Test that maximize mode works correctly in both implementations."""
        # Create the cost functions with maximize=True
        manual_cost, autodiff_cost = self.create_cost_functions(
            self.point, self.interpolator, self.autodiff_interpolator, 
            self.cost_weight, maximize=True
        )
        
        # Compute errors
        manual_error = manual_cost.error()
        autodiff_error = autodiff_cost.error()
        
        # Now both implementations handle maximize mode the same way:
        # - errors are the raw interpolated values in both implementations
        # - Jacobians are negated in both implementations (tested separately)
        self.assertTrue(torch.allclose(manual_error, autodiff_error, atol=1e-6),
                        f"Manual error {manual_error} should match autodiff error {autodiff_error}")
        
        # Compute Jacobians
        manual_jacs, _ = manual_cost.jacobians()
        autodiff_jacs, _ = autodiff_cost.jacobians()
        
        # In maximize mode, both implementations should negate the gradient
        # so they should still match each other
        self.assertTrue(torch.allclose(manual_jacs[0], autodiff_jacs[0], atol=1e-6),
                        f"In maximize mode, manual Jacobian {manual_jacs[0]} should match autodiff Jacobian {autodiff_jacs[0]}")
    
    def test_optimization(self):
        """Test that we can optimize with both implementations and get similar results."""
        # Initial point - closer to the origin
        # Shape: [batch_size, 3], order: [z, y, x]
        init_value = torch.tensor([
            [4.0, 4.0, 4.0],  # Batch item 0
            [4.0, 4.0, 4.0]   # Batch item 1
        ], dtype=torch.float32)
        
        # Create two points for optimization (one for each implementation)
        point_manual = th.Point3(tensor=init_value.clone(), name="point_manual")
        point_autodiff = th.Point3(tensor=init_value.clone(), name="point_autodiff")
        
        # Create cost functions with maximize=True to seek maximum values
        manual_cost, autodiff_cost = self.create_cost_functions(
            point_manual, self.distance_interpolator, self.distance_autodiff_interpolator,
            self.cost_weight, maximize=True
        )
        autodiff_cost_for_manual_point = SpaceLossAccAutoDiff(
            point_autodiff, self.distance_autodiff_interpolator, self.cost_weight, maximize=True
        )
        
        # Create objectives
        objective_manual = th.Objective()
        objective_autodiff = th.Objective()
        
        # Add costs to objectives
        objective_manual.add(manual_cost)
        objective_autodiff.add(autodiff_cost_for_manual_point)
        
        # Create optimizers
        optimizer_manual = th.LevenbergMarquardt(
            objective_manual,
            th.CholeskyDenseSolver,
            max_iterations=50,
            step_size=1.0,
        )
        optimizer_autodiff = th.LevenbergMarquardt(
            objective_autodiff,
            th.CholeskyDenseSolver,
            max_iterations=50,
            step_size=1.0,
        )
        
        # Create TheseusLayers
        layer_manual = th.TheseusLayer(optimizer_manual)
        layer_autodiff = th.TheseusLayer(optimizer_autodiff)
        
        # Prepare input dictionaries
        inputs_manual = {"point_manual": init_value.clone()}
        inputs_autodiff = {"point_autodiff": init_value.clone()}
        
        # Run optimization
        with torch.no_grad():
            final_values_manual, info_manual = layer_manual.forward(inputs_manual)
            final_values_autodiff, info_autodiff = layer_autodiff.forward(inputs_autodiff)
        
        # Get optimized points
        optimized_point_manual = final_values_manual['point_manual']
        optimized_point_autodiff = final_values_autodiff['point_autodiff']
        
        # Calculate distances from origin for initial and optimized points
        initial_distance = torch.norm(init_value, dim=1)
        optimized_distance_manual = torch.norm(optimized_point_manual, dim=1)
        optimized_distance_autodiff = torch.norm(optimized_point_autodiff, dim=1)
        
        # Both implementations should move away from origin (increasing distance)
        # since we're in maximize mode and value increases with distance
        
        # Check manual implementation
        for i in range(init_value.shape[0]):
            self.assertGreater(
                optimized_distance_manual[i].item(), 
                initial_distance[i].item(),
                f"Manual: Expected distance to origin ({optimized_distance_manual[i].item()}) "
                f"to be greater than initial distance ({initial_distance[i].item()})"
            )
        
        # Note: We've observed that the autodiff implementation doesn't consistently
        # maximize in all test cases. In some datasets, especially with batch optimization,
        # some of the batch items may not maximize as expected, which is a known issue.
        # For our test, we'll check that at least one of the batch items improved.
        
        # Check that at least one point in the batch improved (moved away from origin)
        any_improved = False
        for i in range(init_value.shape[0]):
            if optimized_distance_autodiff[i].item() > initial_distance[i].item():
                any_improved = True
                break
                
        self.assertTrue(
            any_improved,
            f"AutoDiff: Expected at least one point to move away from origin. "
            f"Initial distances: {initial_distance}, Final distances: {optimized_distance_autodiff}"
        )
        
        # Now that we've fixed the autodiff implementation, both implementations should
        # move points in the same direction. Let's check that both implementations
        # successfully move away from the origin in maximize mode.
        manual_moved = torch.norm(optimized_point_manual - init_value, dim=1).max().item()
        autodiff_moved = torch.norm(optimized_point_autodiff - init_value, dim=1).max().item()
        
        print(f"Manual implementation moved by: {manual_moved}")
        print(f"Autodiff implementation moved by: {autodiff_moved}")
        
        # Check that both implementations moved away from the origin
        for i in range(init_value.shape[0]):
            # Manual implementation should move away from origin
            self.assertGreater(
                torch.norm(optimized_point_manual[i]).item(), 
                initial_distance[i].item(),
                f"Manual: Expected distance to origin ({torch.norm(optimized_point_manual[i]).item()}) "
                f"to be greater than initial distance ({initial_distance[i].item()})"
            )
            
            # Autodiff implementation should also move away from origin now
            self.assertGreater(
                torch.norm(optimized_point_autodiff[i]).item(), 
                initial_distance[i].item(),
                f"AutoDiff: Expected distance to origin ({torch.norm(optimized_point_autodiff[i]).item()}) "
                f"to be greater than initial distance ({initial_distance[i].item()})"
            )
        
        # Both implementations should be moving in the same direction, but may move
        # different distances due to different optimization paths. To check this,
        # we'll verify that the direction of movement is similar:
        manual_direction = (optimized_point_manual - init_value) / torch.norm(optimized_point_manual - init_value, dim=1, keepdim=True)
        autodiff_direction = (optimized_point_autodiff - init_value) / torch.norm(optimized_point_autodiff - init_value, dim=1, keepdim=True)
        
        # Check that the cosine similarity between directions is high (close to 1)
        # We'll check batch item by batch item
        for i in range(init_value.shape[0]):
            cosine_sim = torch.sum(manual_direction[i] * autodiff_direction[i])
            print(f"Cosine similarity for batch item {i}: {cosine_sim.item()}")
            self.assertGreater(
                cosine_sim.item(), 
                0.9,  # Cosine similarity > 0.9 means directions are very similar
                f"Direction vectors should be similar. Manual: {manual_direction[i]}, Autodiff: {autodiff_direction[i]}"
            )


if __name__ == '__main__':
    unittest.main()