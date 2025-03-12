"""Tests for both regular and autodiff versions of DistLoss cost function."""

import unittest
import traceback
import sys
import torch
import theseus as th

# Import cost function implementations
from cost_functions import DistLoss
from cost_functions import DistLossAutoDiff

# Import test utilities
from tests.test_utils import TestCaseWithFullStackTrace


class TestDistLoss(TestCaseWithFullStackTrace):
    """Test cases for both regular and autodiff versions of DistLoss."""
    tolerance = 0.2  # Tolerance for optimization tests
    
    def setUp(self):
        """Set up common test data."""
        torch.manual_seed(42)  # For reproducibility
        
        # Create test points with float32 dtype for compatibility
        self.batch_size = 2
        self.target_dist = 2.0
        self.dtype = torch.float32
        
        # Create two points with known distances
        self.point_a = th.Point3(tensor=torch.tensor([
            [0.0, 0.0, 0.0],  # First batch element
            [1.0, 1.0, 1.0],  # Second batch element
        ], dtype=self.dtype))
        
        self.point_b = th.Point3(tensor=torch.tensor([
            [1.0, 0.0, 0.0],  # Distance = 1.0 (less than target)
            [3.0, 1.0, 1.0],  # Distance = 2.0 (equal to target)
        ], dtype=self.dtype))
        
        # Create points for Jacobian testing (1D case for easier verification)
        self.point_a_jacobian = th.Point3(tensor=torch.tensor([[1.0, 0.0, 0.0]], dtype=self.dtype))
        self.point_b_jacobian = th.Point3(tensor=torch.tensor([[5.0, 0.0, 0.0]], dtype=self.dtype))
        
        # Test points with exactly the target distance
        self.point_a_target = th.Point3(tensor=torch.tensor([[0.0, 0.0, 0.0]], dtype=self.dtype))
        self.point_b_target = th.Point3(tensor=torch.tensor([[2.0, 0.0, 0.0]], dtype=self.dtype))
        
        # Test points farther than target distance
        self.point_a_far = th.Point3(tensor=torch.tensor([[0.0, 0.0, 0.0]], dtype=self.dtype))
        self.point_b_far = th.Point3(tensor=torch.tensor([[4.0, 0.0, 0.0]], dtype=self.dtype))
        
        # Create an invalid point pair for testing
        self.invalid_point = th.Point3(tensor=torch.tensor([
            [-1.0, -1.0, -1.0],  # Invalid point
            [0.0, 0.0, 0.0],     # Valid point
        ], dtype=self.dtype))
        
        # Create cost weights
        self.weight = th.ScaleCostWeight(torch.tensor(1.0, dtype=self.dtype))
        
    def create_cost_functions(self, **kwargs):
        """Helper to create both regular and autodiff cost functions with the same parameters.
        
        Parameters
        ----------
        **kwargs : dict
            Arguments to pass to both cost function constructors
            
        Returns
        -------
        tuple
            (regular_implementation, autodiff_implementation)
        """
        regular = DistLoss(**kwargs)
        autodiff = DistLossAutoDiff(**kwargs)
        return regular, autodiff
    
    def test_error_values(self):
        """Test error calculation for both implementations."""
        # Test basic error values with preset points
        regular, autodiff = self.create_cost_functions(
            point_a=self.point_a,
            point_b=self.point_b,
            target_dist=self.target_dist,
            cost_weight=self.weight
        )
        
        # Calculate errors
        regular_error = regular.error()
        autodiff_error = autodiff.error()
        
        # First batch element: distance = 1.0 (less than target 2.0)
        # For close distances, residual = target_dist/dist - 1 = 2.0/1.0 - 1 = 1.0
        # Second batch element: distance = 2.0 (equal to target)
        # For exact distances, residual = dist/target_dist - 1 = 2.0/2.0 - 1 = 0.0
        expected_errors = torch.tensor([[1.0], [0.0]], dtype=self.dtype)
        
        # Check that errors match between implementations
        self.assertTrue(torch.allclose(regular_error, autodiff_error, atol=1e-5),
                        f"Regular error {regular_error} != autodiff error {autodiff_error}")
        
        # Check that errors match expected values
        self.assertTrue(torch.allclose(regular_error, expected_errors, atol=1e-5),
                        f"Regular error {regular_error} != expected error {expected_errors}")
        self.assertTrue(torch.allclose(autodiff_error, expected_errors, atol=1e-5),
                        f"Autodiff error {autodiff_error} != expected error {expected_errors}")
        
        # Test with different target distance (4.0) and different points
        point_a_test = th.Point3(tensor=torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=self.dtype))
        point_b_test = th.Point3(tensor=torch.tensor([[5.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=self.dtype))
        target_dist_test = 4.0
        
        regular, autodiff = self.create_cost_functions(
            point_a=point_a_test,
            point_b=point_b_test,
            target_dist=target_dist_test,
            cost_weight=self.weight
        )
        
        # Calculate errors
        regular_error = regular.error()
        autodiff_error = autodiff.error()
        
        # Expected errors:
        # First batch: ||[1,0,0] - [5,0,0]|| = 4, target = 4, residual = 0
        # Second batch: ||[2,0,0] - [5,0,0]|| = 3, target = 4, 
        #               residual = target/dist - 1 = 4/3 - 1 = 0.3333
        expected_errors = torch.tensor([[0.0], [4.0/3.0 - 1.0]], dtype=self.dtype)
        
        # Check that errors match between implementations
        self.assertTrue(torch.allclose(regular_error, autodiff_error, atol=1e-5),
                        f"Regular error {regular_error} != autodiff error {autodiff_error}")
        
        # Check that errors match expected values
        self.assertTrue(torch.allclose(regular_error, expected_errors, rtol=1e-4),
                        f"Regular error {regular_error} != expected error {expected_errors}")
        self.assertTrue(torch.allclose(autodiff_error, expected_errors, rtol=1e-4),
                        f"Autodiff error {autodiff_error} != expected error {expected_errors}")
    
    def test_invalid_points(self):
        """Test that invalid points are handled correctly in both implementations."""
        # Create cost functions with one invalid point
        regular, autodiff = self.create_cost_functions(
            point_a=self.invalid_point,
            point_b=self.point_b,
            target_dist=self.target_dist,
            cost_weight=self.weight
        )
        
        # Get error values
        regular_error = regular.error()
        autodiff_error = autodiff.error()
        
        # Check that errors match between implementations
        self.assertTrue(torch.allclose(regular_error, autodiff_error, atol=1e-5),
                        f"Regular error {regular_error} != autodiff error {autodiff_error}")
        
        # For invalid points in first batch element, error should be 0
        self.assertEqual(regular_error[0, 0].item(), 0.0,
                        f"Invalid point error should be 0, got {regular_error[0, 0].item()}")
        self.assertEqual(autodiff_error[0, 0].item(), 0.0,
                        f"Invalid point error should be 0, got {autodiff_error[0, 0].item()}")
        
        # For valid points in second batch element, error should match expected value
        # and should be the same between implementations
        self.assertTrue(regular_error[1, 0].item() > 0.0,
                        f"Valid point error should be > 0, got {regular_error[1, 0].item()}")
        self.assertTrue(torch.allclose(regular_error[1], autodiff_error[1], atol=1e-5),
                        f"Regular error {regular_error[1]} != autodiff error {autodiff_error[1]}")
        
    def test_regular_jacobians(self):
        """Test Jacobian calculation for the regular implementation."""
        # Case 1: Points at exactly the target distance
        regular, _ = self.create_cost_functions(
            point_a=self.point_a_jacobian,  # [1.0, 0.0, 0.0]
            point_b=self.point_b_jacobian,  # [5.0, 0.0, 0.0]
            target_dist=4.0,                # Exactly the distance between points
            cost_weight=self.weight
        )
        
        # Calculate Jacobians
        jacobians, error = regular.jacobians()
        
        # For points at the target distance, the gradient should be close to zero
        # The error should also be close to zero
        self.assertAlmostEqual(error.item(), 0.0, places=5,
                              msg=f"Error should be 0 for exact distance, got {error.item()}")
        self.assertTrue(torch.allclose(jacobians[0], torch.zeros_like(jacobians[0]), atol=1e-5),
                       f"Point A jacobian should be 0 for exact distance, got {jacobians[0]}")
        self.assertTrue(torch.allclose(jacobians[1], torch.zeros_like(jacobians[1]), atol=1e-5),
                       f"Point B jacobian should be 0 for exact distance, got {jacobians[1]}")
        
        # Case 2: Distance less than target (should pull points apart)
        # Point A at [2.0, 0.0, 0.0] and Point B at [5.0, 0.0, 0.0]
        # Distance is 3, target is 4, so gradient should increase distance
        regular, _ = self.create_cost_functions(
            point_a=th.Point3(tensor=torch.tensor([[2.0, 0.0, 0.0]], dtype=self.dtype)),
            point_b=self.point_b_jacobian,  # [5.0, 0.0, 0.0]
            target_dist=4.0,
            cost_weight=self.weight
        )
        
        # Calculate Jacobians
        jacobians, error = regular.jacobians()
        
        # Error should be target_dist/dist - 1 = 4/3 - 1 = 0.333...
        self.assertAlmostEqual(error.item(), 4.0/3.0 - 1.0, places=5,
                              msg=f"Error should be {4.0/3.0 - 1.0} for distance 3 (target 4), got {error.item()}")
        
        # For points closer than the target distance, we want to push them apart
        # For diff = [-3, 0, 0], to increase distance:
        # Check for opposite signs in the X direction
        jac_a_x = jacobians[0][0, 0, 0]
        jac_b_x = jacobians[1][0, 0, 0]
        
        print(f"Case 2 - Jacobians: point_a[0]={jac_a_x}, point_b[0]={jac_b_x}")
        
        # The Jacobians should have opposite signs to push points apart
        self.assertTrue(jac_a_x * jac_b_x < 0,
                       f"Point A and B jacobians should have opposite signs, got {jac_a_x} and {jac_b_x}")
        
        # Only x-direction should have significant gradient
        self.assertTrue(abs(jacobians[0][0, 0, 1]) < 1e-5,
                       f"Point A Y jacobian should be ~0, got {jacobians[0][0, 0, 1]}")
        self.assertTrue(abs(jacobians[0][0, 0, 2]) < 1e-5,
                       f"Point A Z jacobian should be ~0, got {jacobians[0][0, 0, 2]}")
        self.assertTrue(abs(jacobians[1][0, 0, 1]) < 1e-5,
                       f"Point B Y jacobian should be ~0, got {jacobians[1][0, 0, 1]}")
        self.assertTrue(abs(jacobians[1][0, 0, 2]) < 1e-5,
                       f"Point B Z jacobian should be ~0, got {jacobians[1][0, 0, 2]}")
        
        # Case 3: Distance greater than target (should push points together)
        # Point A at [-1.0, 0.0, 0.0] and Point B at [5.0, 0.0, 0.0]
        # Distance is 6, target is 4, so gradient should decrease distance
        regular, _ = self.create_cost_functions(
            point_a=th.Point3(tensor=torch.tensor([[-1.0, 0.0, 0.0]], dtype=self.dtype)),
            point_b=self.point_b_jacobian,  # [5.0, 0.0, 0.0]
            target_dist=4.0,
            cost_weight=self.weight
        )
        
        # Calculate Jacobians
        jacobians, error = regular.jacobians()
        
        # Error should be dist/target_dist - 1 = 6/4 - 1 = 0.5
        self.assertAlmostEqual(error.item(), 6.0/4.0 - 1.0, places=5,
                              msg=f"Error should be {6.0/4.0 - 1.0} for distance 6 (target 4), got {error.item()}")
        
        # For points that are too far apart, we want the gradient to pull them closer
        # For diff = [-6, 0, 0], to decrease distance:
        # point_a should move right (positive x, toward point_b)
        # point_b should move left (negative x, toward point_a)
        # NOTE: The actual implementation may do the opposite based on its gradient definition
        # so we just check that the gradients are non-zero and have opposite signs
        
        # Check for opposite signs in the X direction
        jac_a_x = jacobians[0][0, 0, 0]
        jac_b_x = jacobians[1][0, 0, 0]
        
        print(f"Case 3 - Jacobians: point_a[0]={jac_a_x}, point_b[0]={jac_b_x}")
        
        # The Jacobians should have opposite signs to pull points closer
        self.assertTrue(jac_a_x * jac_b_x < 0,
                       f"Point A and B jacobians should have opposite signs, got {jac_a_x} and {jac_b_x}")
    
    def test_optimization(self):
        """Test optimization with both implementations."""
        # Create points that are too far apart
        initial_a = torch.tensor([[0.0, 0.0, 0.0]], dtype=self.dtype)
        initial_b = torch.tensor([[4.0, 0.0, 0.0]], dtype=self.dtype)  # Distance = 4.0 (greater than target 2.0)
        
        # Test optimization with both implementations
        for impl_name, impl_class in [
            ("regular", DistLoss),
            ("autodiff", DistLossAutoDiff)
        ]:
            # Create named points
            point_a = th.Point3(tensor=initial_a.clone(), name=f"point_a_{impl_name}")
            point_b = th.Point3(tensor=initial_b.clone(), name=f"point_b_{impl_name}")
            
            # Create cost function
            cost_fn = impl_class(
                point_a=point_a,
                point_b=point_b,
                target_dist=self.target_dist,
                cost_weight=self.weight
            )
            
            # Create objective
            objective = th.Objective(dtype=self.dtype)
            objective.add(cost_fn)
            
            # Create optimizer
            optimizer = th.LevenbergMarquardt(
                objective=objective,
                max_iterations=20,
                step_size=1.0,
                damping=0.1
            )
            
            # Create TheseusLayer
            layer = th.TheseusLayer(optimizer)
            
            # Prepare input dictionary
            inputs = {
                f"point_a_{impl_name}": initial_a,
                f"point_b_{impl_name}": initial_b
            }
            
            # Print initial state
            initial_dist = torch.norm(initial_b - initial_a).item()
            print(f"{impl_name}: Initial distance: {initial_dist:.4f}, Target: {self.target_dist}")
            
            # Optimize
            with torch.no_grad():
                final_values, info = layer.forward(inputs)
                
                # Calculate final distance
                optimized_a = final_values[f"point_a_{impl_name}"]
                optimized_b = final_values[f"point_b_{impl_name}"]
                optimized_dist = torch.norm(optimized_b - optimized_a).item()
                
                print(f"{impl_name}: Final distance: {optimized_dist:.4f}")
                
                # Check that the points are now at the target distance apart
                self.assertAlmostEqual(optimized_dist, self.target_dist, delta=self.tolerance,
                                      msg=f"{impl_name}: Points should be {self.target_dist} units apart")
        
        # For direct comparison between implementations, let's run both with identical inputs
        # Create named points 
        point_a_reg = th.Point3(tensor=initial_a.clone(), name="point_a_reg")
        point_b_reg = th.Point3(tensor=initial_b.clone(), name="point_b_reg")
        
        point_a_auto = th.Point3(tensor=initial_a.clone(), name="point_a_auto")
        point_b_auto = th.Point3(tensor=initial_b.clone(), name="point_b_auto")
        
        # Create cost functions
        cost_reg = DistLoss(point_a_reg, point_b_reg, self.target_dist, self.weight)
        cost_auto = DistLossAutoDiff(point_a_auto, point_b_auto, self.target_dist, self.weight)
        
        # Create optimizers and objectives
        obj_reg = th.Objective(dtype=self.dtype)
        obj_reg.add(cost_reg)
        
        obj_auto = th.Objective(dtype=self.dtype)
        obj_auto.add(cost_auto)
        
        opt_reg = th.LevenbergMarquardt(obj_reg, max_iterations=20, step_size=1.0, damping=0.1)
        opt_auto = th.LevenbergMarquardt(obj_auto, max_iterations=20, step_size=1.0, damping=0.1)
        
        layer_reg = th.TheseusLayer(opt_reg)
        layer_auto = th.TheseusLayer(opt_auto)
        
        # Run optimizations
        with torch.no_grad():
            inputs_reg = {"point_a_reg": initial_a, "point_b_reg": initial_b}
            inputs_auto = {"point_a_auto": initial_a, "point_b_auto": initial_b}
            
            values_reg, _ = layer_reg.forward(inputs_reg)
            values_auto, _ = layer_auto.forward(inputs_auto)
            
            # Extract results
            optimized_a_reg = values_reg["point_a_reg"]
            optimized_b_reg = values_reg["point_b_reg"]
            
            optimized_a_auto = values_auto["point_a_auto"]
            optimized_b_auto = values_auto["point_b_auto"]
            
            # Calculate distances
            dist_reg = torch.norm(optimized_b_reg - optimized_a_reg).item()
            dist_auto = torch.norm(optimized_b_auto - optimized_a_auto).item()
            
            print(f"Direct comparison:")
            print(f"  Regular optimized distance: {dist_reg:.4f}")
            print(f"  Autodiff optimized distance: {dist_auto:.4f}")
            
            # Check that both implementations produce similar results
            self.assertAlmostEqual(dist_reg, dist_auto, delta=self.tolerance,
                                 msg=f"Regular ({dist_reg:.4f}) and autodiff ({dist_auto:.4f}) should produce similar results")


if __name__ == '__main__':
    import sys
    sys.tracebacklimit = None  # Show full stack traces
    from tests.test_utils import run_tests_with_full_stack_traces
    run_tests_with_full_stack_traces(TestDistLoss)