"""Tests for ZLocationLoss and ZLocationLossAutoDiff cost functions."""

import unittest

import torch
import theseus as th

from cost_functions.base.z_location_loss import ZLocationLoss
from cost_functions.autodiff.z_location_loss_autodiff import ZLocationLossAutoDiff
from tests.test_utils import TestCaseWithFullStackTrace


class TestZLocationLoss(TestCaseWithFullStackTrace):
    """Test cases for ZLocationLoss and ZLocationLossAutoDiff."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)  # For reproducibility
        
        # Create test fixtures with consistent dtype
        self.dtype = torch.float32
        
        # Create different test matrices
        
        # 1. Simple linear matrix - z increases with y
        self.batch_size = 1
        self.height, self.width = 5, 5
        self.linear_matrix = torch.zeros((self.batch_size, self.height, self.width, 3), dtype=self.dtype)
        for y in range(self.height):
            for x in range(self.width):
                self.linear_matrix[0, y, x, 0] = float(y)  # z increases with y (z is at index 0 in ZYX)
        
        # 2. Bilinear matrix - z increases with both y and x
        self.bilinear_matrix = torch.zeros((self.batch_size, self.height, self.width, 3), dtype=self.dtype)
        for y in range(self.height):
            for x in range(self.width):
                self.bilinear_matrix[0, y, x, 0] = float(y) + 0.5 * float(x)  # z = y + 0.5x
        
        # 3. Distance-based matrix (like in autodiff test) - z is distance from center
        self.batch_size_multi = 2
        h, w = 4, 4
        self.distance_matrix = torch.zeros((self.batch_size_multi, h, w, 3), dtype=self.dtype)
        for b in range(self.batch_size_multi):
            for y in range(h):
                for x in range(w):
                    # Calculate distance from center (1.5, 1.5)
                    dy = y - 1.5
                    dx = x - 1.5
                    dist = (dy**2 + dx**2)**0.5
                    
                    # Set Z value (first element in ZYX order)
                    self.distance_matrix[b, y, x, 0] = dist
                    
                    # Set Y and X values for completeness
                    self.distance_matrix[b, y, x, 1] = float(y)
                    self.distance_matrix[b, y, x, 2] = float(x)
        
        # Create point matrix where distance is from (1,1)
        self.target_matrix = torch.zeros((1, h, w, 3), dtype=self.dtype)
        for y in range(h):
            for x in range(w):
                # Calculate distance from (1,1)
                dy = y - 1.0
                dx = x - 1.0
                dist = (dy**2 + dx**2)**0.5
                
                # Set Z value (first element in ZYX order)
                self.target_matrix[0, y, x, 0] = dist
        
        # Wrap matrices in Theseus Variables
        self.linear_matrix_var = th.Variable(tensor=self.linear_matrix.clone())
        self.bilinear_matrix_var = th.Variable(tensor=self.bilinear_matrix.clone())
        self.distance_matrix_var = th.Variable(tensor=self.distance_matrix.clone())
        self.target_matrix_var = th.Variable(tensor=self.target_matrix.clone(), name="test_grid")
        
        # Create test locations
        self.location1 = th.Point2(tensor=torch.tensor([[1.5, 2.0]], dtype=self.dtype))
        self.location2 = th.Point2(tensor=torch.tensor([[2.5, 1.0]], dtype=self.dtype))
        
        # Multi-batch locations
        self.location_multi = th.Point2(
            tensor=torch.tensor([[1.5, 1.5], [2.0, 1.0]], dtype=self.dtype)
        )
        
        # Target Z values
        self.target_z = th.Vector(tensor=torch.tensor([[3.0]], dtype=self.dtype))
        self.target_z_multi = th.Vector(
            tensor=torch.tensor([[3.0], [2.0]], dtype=self.dtype)
        )
        self.target_z_zero = th.Vector(tensor=torch.tensor([[0.0]], dtype=self.dtype), name="z_target_zero")
        
        # Cost weights
        self.cost_weight = th.ScaleCostWeight(torch.tensor(1.0, dtype=self.dtype))
        self.strong_weight = th.ScaleCostWeight(torch.tensor(10.0, dtype=self.dtype))
        
        # Tolerance values
        self.optimization_tolerance = 1.0
        self.comparison_tolerance = 1e-4
    
    def create_cost_functions(self, location, matrix, target_z, cost_weight):
        """Create both regular and autodiff versions of the cost function."""
        manual_cost = ZLocationLoss(
            location=location, 
            matrix=matrix, 
            target_z=target_z, 
            cost_weight=cost_weight
        )
        
        autodiff_cost = ZLocationLossAutoDiff(
            location=location, 
            matrix=matrix, 
            target_z=target_z, 
            cost_weight=cost_weight
        )
        
        return manual_cost, autodiff_cost
    
    def test_error_values(self):
        """Test that the error values are calculated correctly and match between implementations."""
        # Test with linear matrix and location1 - should interpolate to z=1.5
        manual_cost, autodiff_cost = self.create_cost_functions(
            self.location1, self.linear_matrix_var, self.target_z, self.cost_weight
        )
        
        manual_error = manual_cost.error()
        autodiff_error = autodiff_cost.error()
        
        # Expected: interpolated z = 1.5, target_z = 3.0, so error = 1.5 - 3.0 = -1.5
        expected_error = torch.tensor([[-1.5]], dtype=self.dtype)
        
        # Check error values
        self.assertTrue(torch.allclose(manual_error, expected_error, atol=self.comparison_tolerance),
                        f"Manual error {manual_error} doesn't match expected {expected_error}")
        self.assertTrue(torch.allclose(autodiff_error, expected_error, atol=self.comparison_tolerance),
                        f"Autodiff error {autodiff_error} doesn't match expected {expected_error}")
        
        # Check that implementations match
        self.assertTrue(torch.allclose(manual_error, autodiff_error, atol=self.comparison_tolerance),
                        f"Manual error {manual_error} doesn't match autodiff error {autodiff_error}")
        
        # Test with linear matrix and location2 - should interpolate to z=2.5
        manual_cost, autodiff_cost = self.create_cost_functions(
            self.location2, self.linear_matrix_var, self.target_z, self.cost_weight
        )
        
        manual_error = manual_cost.error()
        autodiff_error = autodiff_cost.error()
        
        # Expected: interpolated z = 2.5, target_z = 3.0, so error = 2.5 - 3.0 = -0.5
        expected_error = torch.tensor([[-0.5]], dtype=self.dtype)
        
        # Check error values
        self.assertTrue(torch.allclose(manual_error, expected_error, atol=self.comparison_tolerance),
                        f"Manual error {manual_error} doesn't match expected {expected_error}")
        self.assertTrue(torch.allclose(autodiff_error, expected_error, atol=self.comparison_tolerance),
                        f"Autodiff error {autodiff_error} doesn't match expected {expected_error}")
        
        # Check that implementations match
        self.assertTrue(torch.allclose(manual_error, autodiff_error, atol=self.comparison_tolerance),
                        f"Manual error {manual_error} doesn't match autodiff error {autodiff_error}")
        
        # Test with distance matrix and multi-batch locations
        # First create named variables to avoid warnings
        location_multi_named = th.Point2(
            tensor=self.location_multi.tensor.clone(),
            name="location_multi"
        )
        
        distance_matrix_named = th.Variable(
            tensor=self.distance_matrix_var.tensor.clone(),
            name="distance_matrix"
        )
        
        target_z_multi_named = th.Vector(
            tensor=self.target_z_multi.tensor.clone(),
            name="target_z_multi"
        )
        
        manual_cost, autodiff_cost = self.create_cost_functions(
            location_multi_named, distance_matrix_named, target_z_multi_named, self.cost_weight
        )
        
        manual_error = manual_cost.error()
        autodiff_error = autodiff_cost.error()
        
        # Check that implementations match (allowing for small differences in interpolation)
        self.assertTrue(torch.allclose(manual_error, autodiff_error, atol=0.01),
                        f"Manual error {manual_error} doesn't match autodiff error {autodiff_error}")
    
    def test_jacobians(self):
        """Test that the Jacobians have the expected values and match between implementations."""
        # Test with linear matrix - z only varies with y
        manual_cost, autodiff_cost = self.create_cost_functions(
            self.location1, self.linear_matrix_var, self.target_z, self.cost_weight
        )
        
        # Get Jacobians
        manual_jacs, _ = manual_cost.jacobians()
        autodiff_jacs, _ = autodiff_cost.jacobians()
        
        # Expected: dz/dy = 1.0, dz/dx = 0.0
        expected_jac = torch.zeros(self.batch_size, 1, 2, dtype=self.dtype)
        expected_jac[0, 0, 0] = 1.0  # dz/dy = 1.0
        expected_jac[0, 0, 1] = 0.0  # dz/dx = 0.0
        
        # Check Jacobian values
        self.assertTrue(torch.allclose(manual_jacs[0], expected_jac, atol=self.comparison_tolerance),
                        f"Manual Jacobian {manual_jacs[0]} doesn't match expected {expected_jac}")
        self.assertTrue(torch.allclose(autodiff_jacs[0], expected_jac, atol=self.comparison_tolerance),
                        f"Autodiff Jacobian {autodiff_jacs[0]} doesn't match expected {expected_jac}")
        
        # Check that implementations match
        self.assertTrue(torch.allclose(manual_jacs[0], autodiff_jacs[0], atol=self.comparison_tolerance),
                        f"Manual Jacobian {manual_jacs[0]} doesn't match autodiff Jacobian {autodiff_jacs[0]}")
        
        # Test with bilinear matrix - z varies with both y and x
        manual_cost, autodiff_cost = self.create_cost_functions(
            self.location1, self.bilinear_matrix_var, self.target_z, self.cost_weight
        )
        
        # Get Jacobians
        manual_jacs, _ = manual_cost.jacobians()
        autodiff_jacs, _ = autodiff_cost.jacobians()
        
        # Expected: dz/dy = 1.0, dz/dx = 0.5
        expected_jac = torch.zeros(self.batch_size, 1, 2, dtype=self.dtype)
        expected_jac[0, 0, 0] = 1.0   # dz/dy = 1.0
        expected_jac[0, 0, 1] = 0.5   # dz/dx = 0.5
        
        # Check Jacobian values
        self.assertTrue(torch.allclose(manual_jacs[0], expected_jac, atol=self.comparison_tolerance),
                        f"Manual Jacobian {manual_jacs[0]} doesn't match expected {expected_jac}")
        self.assertTrue(torch.allclose(autodiff_jacs[0], expected_jac, atol=self.comparison_tolerance),
                        f"Autodiff Jacobian {autodiff_jacs[0]} doesn't match expected {expected_jac}")
        
        # Check that implementations match
        self.assertTrue(torch.allclose(manual_jacs[0], autodiff_jacs[0], atol=self.comparison_tolerance),
                        f"Manual Jacobian {manual_jacs[0]} doesn't match autodiff Jacobian {autodiff_jacs[0]}")
    
    def test_optimization_bilinear(self):
        """Test that we can optimize a bilinear matrix problem with both implementations."""
        # Create optimization points with different names to avoid conflicts
        location_manual = th.Point2(
            tensor=torch.tensor([[1.0, 1.0]], dtype=self.dtype), 
            name="location_manual"
        )
        
        location_autodiff = th.Point2(
            tensor=torch.tensor([[1.0, 1.0]], dtype=self.dtype), 
            name="location_autodiff"
        )
        
        # Create named matrix variables
        matrix_manual = th.Variable(
            tensor=self.bilinear_matrix.clone(),
            name="matrix_manual"
        )
        
        matrix_autodiff = th.Variable(
            tensor=self.bilinear_matrix.clone(),
            name="matrix_autodiff"
        )
        
        # Create named target vectors
        target_z_manual = th.Vector(
            tensor=torch.tensor([[3.0]], dtype=self.dtype),
            name="target_z_manual"
        )
        
        target_z_autodiff = th.Vector(
            tensor=torch.tensor([[3.0]], dtype=self.dtype),
            name="target_z_autodiff"
        )
        
        # Create the cost functions
        manual_cost = ZLocationLoss(
            location=location_manual,
            matrix=matrix_manual,
            target_z=target_z_manual,
            cost_weight=self.strong_weight
        )
        
        autodiff_cost = ZLocationLossAutoDiff(
            location=location_autodiff,
            matrix=matrix_autodiff,
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
            max_iterations=50,
            step_size=1.0
        )
        
        autodiff_optimizer = th.LevenbergMarquardt(
            objective=autodiff_objective,
            max_iterations=50,
            step_size=1.0
        )
        
        # Create TheseusLayers
        manual_layer = th.TheseusLayer(manual_optimizer)
        autodiff_layer = th.TheseusLayer(autodiff_optimizer)
        
        # Prepare input dictionaries
        manual_inputs = {
            "location_manual": torch.tensor([[1.0, 1.0]], dtype=self.dtype),
            "matrix_manual": self.bilinear_matrix.clone(),
            "target_z_manual": torch.tensor([[3.0]], dtype=self.dtype)
        }
        
        autodiff_inputs = {
            "location_autodiff": torch.tensor([[1.0, 1.0]], dtype=self.dtype),
            "matrix_autodiff": self.bilinear_matrix.clone(),
            "target_z_autodiff": torch.tensor([[3.0]], dtype=self.dtype)
        }
        
        # Calculate initial errors
        initial_error_manual = manual_cost.error()
        initial_error_autodiff = autodiff_cost.error()
        
        # Run optimization
        with torch.no_grad():
            manual_outputs, _ = manual_layer.forward(manual_inputs)
            autodiff_outputs, _ = autodiff_layer.forward(autodiff_inputs)
        
        # Get optimized locations
        optimized_location_manual = manual_outputs["location_manual"]
        optimized_location_autodiff = autodiff_outputs["location_autodiff"]
        
        # Create new cost functions with optimized locations
        optimized_manual_cost = ZLocationLoss(
            th.Point2(tensor=optimized_location_manual),
            matrix_manual,
            target_z_manual,
            self.strong_weight
        )
        
        optimized_autodiff_cost = ZLocationLossAutoDiff(
            th.Point2(tensor=optimized_location_autodiff),
            matrix_autodiff,
            target_z_autodiff,
            self.strong_weight
        )
        
        # Get final errors
        final_error_manual = optimized_manual_cost.error()
        final_error_autodiff = optimized_autodiff_cost.error()
        
        # Print results for comparison
        print("Bilinear matrix optimization:")
        print(f"  Initial location: (1.0, 1.0), Target z: 3.0")
        print(f"  Manual optimized location: {optimized_location_manual.tolist()}")
        print(f"  Autodiff optimized location: {optimized_location_autodiff.tolist()}")
        print(f"  Manual initial/final error: {initial_error_manual.item():.4f}/{final_error_manual.item():.4f}")
        print(f"  Autodiff initial/final error: {initial_error_autodiff.item():.4f}/{final_error_autodiff.item():.4f}")
        
        # Verify that errors decreased
        self.assertLess(torch.abs(final_error_manual).item(), torch.abs(initial_error_manual).item(),
                       f"Manual: Expected final error magnitude to decrease")
        self.assertLess(torch.abs(final_error_autodiff).item(), torch.abs(initial_error_autodiff).item(),
                       f"Autodiff: Expected final error magnitude to decrease")
        
        # Verify final errors are small
        self.assertLess(torch.abs(final_error_manual).item(), self.optimization_tolerance,
                       f"Manual: Expected final error to be less than tolerance")
        self.assertLess(torch.abs(final_error_autodiff).item(), self.optimization_tolerance,
                       f"Autodiff: Expected final error to be less than tolerance")
        
        # Both implementations should produce similar optimized locations
        self.assertTrue(torch.allclose(optimized_location_manual, optimized_location_autodiff, atol=0.5),
                       f"Manual location {optimized_location_manual} should be close to autodiff location {optimized_location_autodiff}")
    
    def test_optimization_distance(self):
        """Test that we can optimize a distance matrix problem with AutoDiff."""
        # Create a location to optimize
        location = th.Point2(
            tensor=torch.tensor([[2.0, 2.0]], dtype=self.dtype),
            name="test_location"
        )
        
        # Create the autodiff cost function
        cost = ZLocationLossAutoDiff(
            location,
            self.target_matrix_var,
            self.target_z_zero,
            self.strong_weight
        )
        
        # Create an objective and add the cost
        objective = th.Objective()
        objective.add(cost)
        
        # Create the optimizer
        optimizer = th.LevenbergMarquardt(
            objective, 
            th.CholeskyDenseSolver,
            max_iterations=20
        )
        
        # Create TheseusLayer and run optimization
        layer = th.TheseusLayer(optimizer)
        inputs = {
            "test_location": torch.tensor([[2.0, 2.0]], dtype=self.dtype),
            "test_grid": self.target_matrix.clone(),
            # Make sure we use the name from the variable declaration above
            self.target_z_zero.name: torch.tensor([[0.0]], dtype=self.dtype)
        }
        
        # Calculate initial error
        initial_error = cost.error()
        
        # Run optimization
        result, _ = layer.forward(inputs)
        
        # Get optimized location
        optimized_location = result["test_location"]
        
        # Calculate final error
        final_cost = ZLocationLossAutoDiff(
            th.Point2(tensor=optimized_location),
            self.target_matrix_var,
            self.target_z_zero,
            self.strong_weight
        )
        final_error = final_cost.error()
        
        # Print results
        print("Distance matrix optimization:")
        print(f"  Initial location: {inputs['test_location']}")
        print(f"  Initial error: {initial_error}")
        print(f"  Optimized location: {optimized_location}")
        print(f"  Final error: {final_error}")
        print(f"  Target location: (1.0, 1.0)")
        
        # Verify that error decreased
        self.assertLess(
            torch.abs(final_error).item(),
            torch.abs(initial_error).item(),
            msg=f"Expected final error magnitude to decrease"
        )
        
        # Verify that location moved toward (1,1)
        self.assertLess(
            torch.norm(optimized_location - torch.tensor([[1.0, 1.0]], dtype=self.dtype)),
            torch.norm(inputs["test_location"] - torch.tensor([[1.0, 1.0]], dtype=self.dtype)),
            msg=f"Expected optimized location to be closer to target (1.0, 1.0)"
        )
        
        # Final location should be close to (1,1)
        self.assertTrue(
            torch.allclose(optimized_location, torch.tensor([[1.0, 1.0]], dtype=self.dtype), atol=0.1),
            msg=f"Expected optimized location to be close to (1.0, 1.0)"
        )


if __name__ == '__main__':
    unittest.main()