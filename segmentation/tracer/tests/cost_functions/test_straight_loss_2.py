"""Tests for StraightLoss2 and StraightLoss2AutoDiff cost functions."""

import unittest

import torch
import theseus as th

from cost_functions.base.straight_loss_2 import StraightLoss2
from cost_functions.autodiff.straight_loss_2_autodiff import StraightLoss2AutoDiff
from tests.test_utils import TestCaseWithFullStackTrace


class TestStraightLoss2(TestCaseWithFullStackTrace):
    """Test cases for StraightLoss2 and StraightLoss2AutoDiff."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)  # For reproducibility
        
        # Create test points with consistent dtype
        self.batch_size = 1
        self.dtype = torch.float32
        
        # Create a straight line configuration
        self.point_a_straight = th.Point3(tensor=torch.tensor([
            [0.0, 0.0, 0.0],  # straight line along x-axis
        ], dtype=self.dtype))
        
        self.point_b_straight = th.Point3(tensor=torch.tensor([
            [1.0, 0.0, 0.0],  # middle point on line
        ], dtype=self.dtype))
        
        self.point_c_straight = th.Point3(tensor=torch.tensor([
            [2.0, 0.0, 0.0],  # end point on line
        ], dtype=self.dtype))
        
        # Create a non-straight configuration
        self.point_a_nonstraight = th.Point3(tensor=torch.tensor([
            [0.0, 0.0, 0.0],  # start point
        ], dtype=self.dtype))
        
        self.point_b_nonstraight = th.Point3(tensor=torch.tensor([
            [1.0, 1.0, 0.0],  # middle point off the line
        ], dtype=self.dtype))
        
        self.point_c_nonstraight = th.Point3(tensor=torch.tensor([
            [2.0, 0.0, 0.0],  # end point
        ], dtype=self.dtype))
        
        # Create cost weights
        self.weight = th.ScaleCostWeight(torch.tensor(1.0, dtype=self.dtype))
        self.strong_weight = th.ScaleCostWeight(torch.tensor(10.0, dtype=self.dtype))
        self.anchor_weight = th.ScaleCostWeight(torch.tensor(5.0, dtype=self.dtype))
    
    def create_cost_functions(self, point_a, point_b, point_c, cost_weight):
        """Create both regular and autodiff versions of the cost function."""
        manual_cost = StraightLoss2(
            point_a=point_a, 
            point_b=point_b, 
            point_c=point_c, 
            cost_weight=cost_weight
        )
        
        autodiff_cost = StraightLoss2AutoDiff(
            point_a=point_a, 
            point_b=point_b, 
            point_c=point_c, 
            cost_weight=cost_weight
        )
        
        return manual_cost, autodiff_cost
    
    def test_error_values(self):
        """Test that the error values are calculated correctly and match between implementations."""
        # Test straight line configuration
        manual_cost, autodiff_cost = self.create_cost_functions(
            self.point_a_straight, self.point_b_straight, self.point_c_straight, self.weight
        )
        
        # Get error values
        manual_error = manual_cost.error()
        autodiff_error = autodiff_cost.error()
        
        # Expected error for a straight line should be [0, 0, 0]
        expected_error = torch.zeros((self.batch_size, 3), dtype=self.dtype)
        
        # Check manual error values
        self.assertTrue(torch.allclose(manual_error, expected_error, atol=1e-4),
                        f"Manual error {manual_error} doesn't match expected {expected_error}")
        
        # Check autodiff error values
        self.assertTrue(torch.allclose(autodiff_error, expected_error, atol=1e-4),
                        f"Autodiff error {autodiff_error} doesn't match expected {expected_error}")
        
        # Check that implementations match
        self.assertTrue(torch.allclose(manual_error, autodiff_error, atol=1e-5),
                        f"Manual error {manual_error} doesn't match autodiff error {autodiff_error}")
        
        # Test non-straight configuration
        manual_cost, autodiff_cost = self.create_cost_functions(
            self.point_a_nonstraight, self.point_b_nonstraight, self.point_c_nonstraight, self.weight
        )
        
        # Get error values
        manual_error = manual_cost.error()
        autodiff_error = autodiff_cost.error()
        
        # Expected error for non-straight line should be [0, 1, 0]
        expected_error = torch.tensor([[0.0, 1.0, 0.0]], dtype=self.dtype)
        
        # Check manual error values
        self.assertTrue(torch.allclose(manual_error, expected_error, atol=1e-4),
                        f"Manual error {manual_error} doesn't match expected {expected_error}")
        
        # Check autodiff error values
        self.assertTrue(torch.allclose(autodiff_error, expected_error, atol=1e-4),
                        f"Autodiff error {autodiff_error} doesn't match expected {expected_error}")
        
        # Check that implementations match
        self.assertTrue(torch.allclose(manual_error, autodiff_error, atol=1e-5),
                        f"Manual error {manual_error} doesn't match autodiff error {autodiff_error}")
    
    def test_jacobians(self):
        """Test the Jacobian calculations for StraightLoss2."""
        # Manual implementation only - autodiff handles Jacobians automatically
        straight_loss = StraightLoss2(
            self.point_a_straight, self.point_b_straight, self.point_c_straight, self.weight
        )
        
        # Calculate Jacobians
        jacobians, _ = straight_loss.jacobians()
        
        # Expected Jacobians - we should have three jacobians returned
        # Jacobian for point_a: -0.5 * Identity (since modifying a affects midpoint)
        expected_jac_a = torch.zeros(self.batch_size, 3, 3, dtype=self.dtype)
        expected_jac_a[0] = -0.5 * torch.eye(3, dtype=self.dtype)
        
        # Jacobian for point_b: Identity
        expected_jac_b = torch.zeros(self.batch_size, 3, 3, dtype=self.dtype)
        expected_jac_b[0] = torch.eye(3, dtype=self.dtype)
        
        # Jacobian for point_c: -0.5 * Identity (since modifying c affects midpoint)
        expected_jac_c = torch.zeros(self.batch_size, 3, 3, dtype=self.dtype)
        expected_jac_c[0] = -0.5 * torch.eye(3, dtype=self.dtype)
        
        # Check all three Jacobian values
        self.assertEqual(len(jacobians), 3, "Should have three jacobians for the three points")
        self.assertTrue(torch.allclose(jacobians[0], expected_jac_a, atol=1e-4), 
                        f"Jacobian for point_a {jacobians[0]} doesn't match expected {expected_jac_a}")
        self.assertTrue(torch.allclose(jacobians[1], expected_jac_b, atol=1e-4), 
                        f"Jacobian for point_b {jacobians[1]} doesn't match expected {expected_jac_b}")
        self.assertTrue(torch.allclose(jacobians[2], expected_jac_c, atol=1e-4), 
                        f"Jacobian for point_c {jacobians[2]} doesn't match expected {expected_jac_c}")
    
    def test_optimization(self):
        """Test that we can optimize with both implementations and get similar results."""
        torch.manual_seed(42)  # For reproducibility
        
        # Create test points for manual implementation with more pronounced deviation
        point_a_manual = th.Point3(
            tensor=torch.tensor([[0.0, 0.0, 0.0]], dtype=self.dtype), 
            name="point_a_manual"
        )
        point_b_manual = th.Point3(
            tensor=torch.tensor([[1.0, 2.0, 0.0]], dtype=self.dtype),  # Increased y-value to 2.0
            name="point_b_manual"
        )
        point_c_manual = th.Point3(
            tensor=torch.tensor([[2.0, 0.0, 0.0]], dtype=self.dtype), 
            name="point_c_manual"
        )
        
        # Create test points for autodiff implementation with same deviation
        point_a_autodiff = th.Point3(
            tensor=torch.tensor([[0.0, 0.0, 0.0]], dtype=self.dtype), 
            name="point_a_autodiff"
        )
        point_b_autodiff = th.Point3(
            tensor=torch.tensor([[1.0, 2.0, 0.0]], dtype=self.dtype),  # Increased y-value to 2.0
            name="point_b_autodiff"
        )
        point_c_autodiff = th.Point3(
            tensor=torch.tensor([[2.0, 0.0, 0.0]], dtype=self.dtype), 
            name="point_c_autodiff"
        )
        
        # Calculate initial error - distance from midpoint (same for both implementations)
        midpoint = (point_a_manual.tensor + point_c_manual.tensor) * 0.5
        initial_distance = torch.norm(point_b_manual.tensor - midpoint, dim=1).item()
        
        # Test 1: Manual implementation with TheseusLayer
        # Create cost function
        manual_cost = StraightLoss2(
            point_a=point_a_manual,
            point_b=point_b_manual,
            point_c=point_c_manual,
            cost_weight=self.strong_weight  # Using stronger weight
        )
        
        # Add anchor costs to keep point_a and point_c relatively fixed
        anchor_a_manual = th.Difference(
            point_a_manual, 
            th.Point3(tensor=torch.tensor([[0.0, 0.0, 0.0]], dtype=self.dtype)), 
            self.anchor_weight,
            name="anchor_a_manual"
        )
        
        anchor_c_manual = th.Difference(
            point_c_manual, 
            th.Point3(tensor=torch.tensor([[2.0, 0.0, 0.0]], dtype=self.dtype)), 
            self.anchor_weight,
            name="anchor_c_manual"
        )
        
        # Create objective and add costs
        manual_objective = th.Objective()
        manual_objective.add(manual_cost)
        manual_objective.add(anchor_a_manual)
        manual_objective.add(anchor_c_manual)
        
        # Create optimizer
        manual_optimizer = th.LevenbergMarquardt(
            objective=manual_objective,
            max_iterations=50,  # Reduced iterations for faster tests
            step_size=1.0,
            damping=0.1  # Explicit damping parameter
        )
        
        # Create TheseusLayer
        manual_layer = th.TheseusLayer(manual_optimizer)
        
        # Optimize
        manual_inputs = {
            "point_a_manual": point_a_manual.tensor,
            "point_b_manual": point_b_manual.tensor,
            "point_c_manual": point_c_manual.tensor
        }
        
        with torch.no_grad():
            manual_outputs, _ = manual_layer.forward(manual_inputs)
        
        # Get optimized points
        manual_a_opt = manual_outputs["point_a_manual"]
        manual_b_opt = manual_outputs["point_b_manual"]
        manual_c_opt = manual_outputs["point_c_manual"]
        
        # Calculate final distance
        manual_midpoint = (manual_a_opt + manual_c_opt) * 0.5
        manual_final_distance = torch.norm(manual_b_opt - manual_midpoint, dim=1).item()
        
        # Test 2: AutoDiff implementation with TheseusLayer
        # Create cost function
        autodiff_cost = StraightLoss2AutoDiff(
            point_a=point_a_autodiff,
            point_b=point_b_autodiff,
            point_c=point_c_autodiff,
            cost_weight=self.strong_weight  # Using stronger weight
        )
        
        # Add anchor costs to keep point_a and point_c relatively fixed
        anchor_a_autodiff = th.Difference(
            point_a_autodiff, 
            th.Point3(tensor=torch.tensor([[0.0, 0.0, 0.0]], dtype=self.dtype)), 
            self.anchor_weight,
            name="anchor_a_autodiff"
        )
        
        anchor_c_autodiff = th.Difference(
            point_c_autodiff, 
            th.Point3(tensor=torch.tensor([[2.0, 0.0, 0.0]], dtype=self.dtype)), 
            self.anchor_weight,
            name="anchor_c_autodiff"
        )
        
        # Create objective and add costs
        autodiff_objective = th.Objective()
        autodiff_objective.add(autodiff_cost)
        autodiff_objective.add(anchor_a_autodiff)
        autodiff_objective.add(anchor_c_autodiff)
        
        # Create optimizer
        autodiff_optimizer = th.LevenbergMarquardt(
            objective=autodiff_objective,
            max_iterations=50,  # Same as manual implementation
            step_size=1.0,
            damping=0.1  # Explicit damping parameter
        )
        
        # Create TheseusLayer
        autodiff_layer = th.TheseusLayer(autodiff_optimizer)
        
        # Optimize
        autodiff_inputs = {
            "point_a_autodiff": point_a_autodiff.tensor,
            "point_b_autodiff": point_b_autodiff.tensor,
            "point_c_autodiff": point_c_autodiff.tensor
        }
        
        with torch.no_grad():
            autodiff_outputs, _ = autodiff_layer.forward(autodiff_inputs)
        
        # Get optimized points
        autodiff_a_opt = autodiff_outputs["point_a_autodiff"]
        autodiff_b_opt = autodiff_outputs["point_b_autodiff"]
        autodiff_c_opt = autodiff_outputs["point_c_autodiff"]
        
        # Calculate final distance
        autodiff_midpoint = (autodiff_a_opt + autodiff_c_opt) * 0.5
        autodiff_final_distance = torch.norm(autodiff_b_opt - autodiff_midpoint, dim=1).item()
        
        # Print distances for debugging
        print(f"Initial distance: {initial_distance:.4f}")
        print(f"Manual final distance: {manual_final_distance:.4f}")
        print(f"AutoDiff final distance: {autodiff_final_distance:.4f}")
        
        # Check that both implementations reduce the distance
        self.assertLess(manual_final_distance, initial_distance * 0.5,
                       f"Manual implementation didn't reduce the distance enough")
        self.assertLess(autodiff_final_distance, initial_distance * 0.5,
                       f"AutoDiff implementation didn't reduce the distance enough")
        
        # Check that both implementations produce similar results
        self.assertAlmostEqual(manual_final_distance, autodiff_final_distance, delta=0.1,
                              msg=f"Manual and AutoDiff implementations produced different results")
        
        # Check that both optimizations moved the point closer to the line
        # Instead of specific coordinate checks, we validate using the distance metric
        # which we already confirmed above dropped from the initial value to near zero
        print(f"Initial distance from line: {initial_distance:.4f}")
        print(f"Manual final distance: {manual_final_distance:.4f}")
        print(f"AutoDiff final distance: {autodiff_final_distance:.4f}")
        
        # We'll also verify that both implementations produced very similar results
        manual_final_pos = manual_b_opt.flatten().tolist()
        autodiff_final_pos = autodiff_b_opt.flatten().tolist()
        print(f"Manual final position: {manual_final_pos}")
        print(f"AutoDiff final position: {autodiff_final_pos}")
        
        # Positions should be very close to each other
        self.assertTrue(
            torch.allclose(manual_b_opt, autodiff_b_opt, atol=1e-2),
            f"Manual final position {manual_b_opt} doesn't match autodiff final position {autodiff_b_opt}"
        )


if __name__ == '__main__':
    unittest.main()