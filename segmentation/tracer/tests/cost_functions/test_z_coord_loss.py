"""Tests for ZCoordLoss and ZCoordLossAutoDiff cost functions."""

import unittest

import torch
import theseus as th

from cost_functions.base.z_coord_loss import ZCoordLoss
from cost_functions.autodiff.z_coord_loss_autodiff import ZCoordLossAutoDiff
from tests.test_utils import TestCaseWithFullStackTrace


class TestZCoordLoss(TestCaseWithFullStackTrace):
    """Test cases for ZCoordLoss and ZCoordLossAutoDiff."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)  # For reproducibility
        
        # Create test points with consistent dtype
        self.dtype = torch.float32
        
        # Create a 3D point in ZYX ordering and a target Z coordinate
        # ZYX ordering means [z, y, x] so Z is at index 0
        self.point1 = th.Point3(tensor=torch.tensor([[3.0, 2.0, 1.0]], dtype=self.dtype))
        self.point2 = th.Point3(tensor=torch.tensor([[6.0, 2.0, 1.0]], dtype=self.dtype))
        
        # Point used in the original autodiff tests
        self.point_autodiff = th.Point3(tensor=torch.tensor([[2.0, 3.0, 4.0]], dtype=self.dtype))
        
        # Target Z coordinates
        self.target_z = th.Vector(tensor=torch.tensor([[5.0]], dtype=self.dtype), name="z_target")
        
        # Cost weights
        self.cost_weight = th.ScaleCostWeight(torch.tensor(1.0, dtype=self.dtype))
        self.strong_weight = th.ScaleCostWeight(torch.tensor(10.0, dtype=self.dtype))
        
        # Tolerance for optimization tests
        self.optimization_tolerance = 1.0
        # Tolerance for numerical comparisons
        self.comparison_tolerance = 1e-5
    
    def create_cost_functions(self, point, target_z, cost_weight):
        """Create both regular and autodiff versions of the cost function."""
        manual_cost = ZCoordLoss(
            point=point, 
            target_z=target_z, 
            cost_weight=cost_weight
        )
        
        autodiff_cost = ZCoordLossAutoDiff(
            point=point, 
            target_z=target_z, 
            cost_weight=cost_weight
        )
        
        return manual_cost, autodiff_cost
    
    def test_error_values(self):
        """Test that the error values are calculated correctly and match between implementations."""
        # Test with point1 (z=3.0, target=5.0)
        manual_cost, autodiff_cost = self.create_cost_functions(
            self.point1, self.target_z, self.cost_weight
        )
        
        manual_error = manual_cost.error()
        autodiff_error = autodiff_cost.error()
        
        # Expected error: z - target_z = 3.0 - 5.0 = -2.0
        expected_error = torch.tensor([[-2.0]], dtype=self.dtype)
        
        # Check error values
        self.assertTrue(torch.allclose(manual_error, expected_error, atol=self.comparison_tolerance),
                        f"Manual error {manual_error} doesn't match expected {expected_error}")
        self.assertTrue(torch.allclose(autodiff_error, expected_error, atol=self.comparison_tolerance),
                        f"Autodiff error {autodiff_error} doesn't match expected {expected_error}")
        
        # Verify manual and autodiff match
        self.assertTrue(torch.allclose(manual_error, autodiff_error, atol=self.comparison_tolerance),
                        f"Manual error {manual_error} doesn't match autodiff error {autodiff_error}")
        
        # Test with point2 (z=6.0, target=5.0)
        manual_cost, autodiff_cost = self.create_cost_functions(
            self.point2, self.target_z, self.cost_weight
        )
        
        manual_error = manual_cost.error()
        autodiff_error = autodiff_cost.error()
        
        # Expected error: z - target_z = 6.0 - 5.0 = 1.0
        expected_error = torch.tensor([[1.0]], dtype=self.dtype)
        
        # Check error values
        self.assertTrue(torch.allclose(manual_error, expected_error, atol=self.comparison_tolerance),
                        f"Manual error {manual_error} doesn't match expected {expected_error}")
        self.assertTrue(torch.allclose(autodiff_error, expected_error, atol=self.comparison_tolerance),
                        f"Autodiff error {autodiff_error} doesn't match expected {expected_error}")
        
        # Verify manual and autodiff match
        self.assertTrue(torch.allclose(manual_error, autodiff_error, atol=self.comparison_tolerance),
                        f"Manual error {manual_error} doesn't match autodiff error {autodiff_error}")
        
        # Test with original autodiff point (z=2.0, target=5.0)
        manual_cost, autodiff_cost = self.create_cost_functions(
            self.point_autodiff, self.target_z, self.cost_weight
        )
        
        manual_error = manual_cost.error()
        autodiff_error = autodiff_cost.error()
        
        # Expected error: z - target_z = 2.0 - 5.0 = -3.0
        expected_error = torch.tensor([[-3.0]], dtype=self.dtype)
        
        # Check error values
        self.assertTrue(torch.allclose(manual_error, expected_error, atol=self.comparison_tolerance),
                        f"Manual error {manual_error} doesn't match expected {expected_error}")
        self.assertTrue(torch.allclose(autodiff_error, expected_error, atol=self.comparison_tolerance),
                        f"Autodiff error {autodiff_error} doesn't match expected {expected_error}")
        
        # Verify manual and autodiff match
        self.assertTrue(torch.allclose(manual_error, autodiff_error, atol=self.comparison_tolerance),
                        f"Manual error {manual_error} doesn't match autodiff error {autodiff_error}")
    
    def test_jacobians(self):
        """Test that the Jacobians have the expected values and match between implementations."""
        # Create cost functions
        manual_cost, autodiff_cost = self.create_cost_functions(
            self.point1, self.target_z, self.cost_weight
        )
        
        # Get Jacobians
        manual_jacs, _ = manual_cost.jacobians()
        autodiff_jacs, _ = autodiff_cost.jacobians()
        
        # The Jacobian should be [1, 0, 0] in ZYX because only the z-coordinate affects the error
        batch_size = 1
        expected_jac = torch.zeros(batch_size, 1, 3, dtype=self.dtype)
        expected_jac[0, 0, 0] = 1.0  # z is index 0 in ZYX ordering
        
        # Check manual Jacobian values
        self.assertTrue(torch.allclose(manual_jacs[0], expected_jac, atol=self.comparison_tolerance),
                        f"Manual Jacobian {manual_jacs[0]} doesn't match expected {expected_jac}")
        
        # Check autodiff Jacobian values
        self.assertTrue(torch.allclose(autodiff_jacs[0], expected_jac, atol=self.comparison_tolerance),
                        f"Autodiff Jacobian {autodiff_jacs[0]} doesn't match expected {expected_jac}")
        
        # Verify manual and autodiff match
        self.assertTrue(torch.allclose(manual_jacs[0], autodiff_jacs[0], atol=self.comparison_tolerance),
                        f"Manual Jacobian {manual_jacs[0]} doesn't match autodiff Jacobian {autodiff_jacs[0]}")
    
    def test_optimization(self):
        """Test that we can optimize with both implementations and get similar results."""
        # Create optimization points with different names to avoid conflicts
        point_manual = th.Point3(
            tensor=torch.tensor([[3.0, 2.0, 1.0]], dtype=self.dtype), 
            name="point_manual"
        )
        
        point_autodiff = th.Point3(
            tensor=torch.tensor([[3.0, 2.0, 1.0]], dtype=self.dtype), 
            name="point_autodiff"
        )
        
        # Create named target vectors for each implementation
        target_z_manual = th.Vector(
            tensor=torch.tensor([[5.0]], dtype=self.dtype), 
            name="target_z_manual"
        )
        
        target_z_autodiff = th.Vector(
            tensor=torch.tensor([[5.0]], dtype=self.dtype), 
            name="target_z_autodiff"
        )
        
        # Create the cost functions
        manual_cost = ZCoordLoss(
            point=point_manual,
            target_z=target_z_manual,
            cost_weight=self.strong_weight
        )
        
        autodiff_cost = ZCoordLossAutoDiff(
            point=point_autodiff,
            target_z=target_z_autodiff,
            cost_weight=self.strong_weight
        )
        
        # Create objectives and add costs
        manual_objective = th.Objective()
        manual_objective.add(manual_cost)
        
        autodiff_objective = th.Objective()
        autodiff_objective.add(autodiff_cost)
        
        # Create optimizers
        manual_optimizer = th.LevenbergMarquardt(
            objective=manual_objective,
            max_iterations=20,
            step_size=1.0
        )
        
        autodiff_optimizer = th.LevenbergMarquardt(
            objective=autodiff_objective,
            max_iterations=20,
            step_size=1.0
        )
        
        # Create TheseusLayers
        manual_layer = th.TheseusLayer(manual_optimizer)
        autodiff_layer = th.TheseusLayer(autodiff_optimizer)
        
        # Prepare input dictionaries
        manual_inputs = {
            "point_manual": torch.tensor([[3.0, 2.0, 1.0]], dtype=self.dtype),
            "target_z_manual": torch.tensor([[5.0]], dtype=self.dtype)
        }
        
        autodiff_inputs = {
            "point_autodiff": torch.tensor([[3.0, 2.0, 1.0]], dtype=self.dtype),
            "target_z_autodiff": torch.tensor([[5.0]], dtype=self.dtype)
        }
        
        # Run optimization
        with torch.no_grad():
            manual_outputs, _ = manual_layer.forward(manual_inputs)
            autodiff_outputs, _ = autodiff_layer.forward(autodiff_inputs)
        
        # Get optimized points
        optimized_point_manual = manual_outputs["point_manual"]
        optimized_point_autodiff = autodiff_outputs["point_autodiff"]
        
        # Get optimized z coordinates
        manual_z = optimized_point_manual[0, 0].item()
        autodiff_z = optimized_point_autodiff[0, 0].item()
        
        # Print results for comparison
        print("Manual implementation:")
        print(f"  Initial z: 3.0, Target z: 5.0")
        print(f"  Optimized z: {manual_z}")
        print(f"  Optimized point: {optimized_point_manual.tolist()}")
        
        print("AutoDiff implementation:")
        print(f"  Initial z: 3.0, Target z: 5.0")
        print(f"  Optimized z: {autodiff_z}")
        print(f"  Optimized point: {optimized_point_autodiff.tolist()}")
        
        # Check that both implementations optimized z to reach the target
        self.assertAlmostEqual(manual_z, 5.0, delta=self.optimization_tolerance,
                              msg=f"Manual: Expected optimized z ({manual_z}) to be close to target (5.0)")
        self.assertAlmostEqual(autodiff_z, 5.0, delta=self.optimization_tolerance,
                              msg=f"AutoDiff: Expected optimized z ({autodiff_z}) to be close to target (5.0)")
        
        # Verify that y and x coordinates remained unchanged for both implementations
        # Manual
        self.assertAlmostEqual(optimized_point_manual[0, 1].item(), 2.0, delta=self.comparison_tolerance,
                              msg=f"Manual: y-coordinate should remain unchanged at 2.0")
        self.assertAlmostEqual(optimized_point_manual[0, 2].item(), 1.0, delta=self.comparison_tolerance,
                              msg=f"Manual: x-coordinate should remain unchanged at 1.0")
        
        # AutoDiff
        self.assertAlmostEqual(optimized_point_autodiff[0, 1].item(), 2.0, delta=self.comparison_tolerance,
                              msg=f"AutoDiff: y-coordinate should remain unchanged at 2.0")
        self.assertAlmostEqual(optimized_point_autodiff[0, 2].item(), 1.0, delta=self.comparison_tolerance, 
                              msg=f"AutoDiff: x-coordinate should remain unchanged at 1.0")
        
        # Both implementations should produce very similar results
        self.assertTrue(torch.allclose(optimized_point_manual, optimized_point_autodiff, atol=0.1),
                       f"Manual result {optimized_point_manual} should be close to autodiff result {optimized_point_autodiff}")


if __name__ == '__main__':
    unittest.main()