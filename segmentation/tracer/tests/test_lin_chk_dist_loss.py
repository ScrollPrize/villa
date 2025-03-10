"""Tests for both regular and autodiff versions of LinChkDistLoss cost function."""

import torch
import theseus as th

# Import cost function implementations
from cost_functions import LinChkDistLoss
from cost_functions import LinChkDistLossAutoDiff

# Import test utilities
from tests.test_utils import TestCaseWithFullStackTrace


class TestLinChkDistLoss(TestCaseWithFullStackTrace):
    """Test cases for both regular and autodiff versions of LinChkDistLoss."""
    
    # Allow more flexibility in our optimization tests
    tolerance = 1.0
    
    def setUp(self):
        """Set up common test data."""
        torch.manual_seed(42)  # For reproducibility
        
        # Create cost weight
        self.weight = th.ScaleCostWeight(1.0)
        
        # Standard test points and targets
        self.point1 = th.Point2(tensor=torch.tensor([[3.0, 2.0]]))
        self.target1 = th.Point2(tensor=torch.tensor([[1.0, 4.0]]))
        
        self.point2 = th.Point2(tensor=torch.tensor([[5.0, 6.0]]))
        self.target2 = th.Point2(tensor=torch.tensor([[1.0, 4.0]]))
    
    def create_cost_functions(self, point, target, cost_weight):
        """Helper to create both regular and autodiff cost functions with the same parameters.
        
        Parameters
        ----------
        point : th.Point2
            The point variable
        target : th.Point2
            The target point variable
        cost_weight : th.CostWeight
            Weight for the cost function
            
        Returns
        -------
        tuple
            (regular_implementation, autodiff_implementation)
        """
        regular = LinChkDistLoss(point, target, cost_weight)
        autodiff = LinChkDistLossAutoDiff(point, target, cost_weight)
        
        return regular, autodiff
    
    def test_error_values(self):
        """Test error calculation for both implementations."""
        # Test case 1: Point at (3,2) and target at (1,4)
        # Create both cost functions
        regular, autodiff = self.create_cost_functions(
            self.point1,
            self.target1,
            self.weight
        )
        
        # Calculate errors
        regular_error = regular.error()
        autodiff_error = autodiff.error()
        
        # Expected: sqrt(|3-1|) = sqrt(2) for x, sqrt(|2-4|) = sqrt(2) for y
        # The implementation only considers absolute differences > 0
        expected_error = torch.tensor([[torch.sqrt(torch.tensor(2.0)), torch.sqrt(torch.tensor(2.0))]])
        
        # Check that errors match between implementations
        self.assertTrue(torch.allclose(regular_error, autodiff_error, atol=1e-4),
                       f"Regular error {regular_error} != autodiff error {autodiff_error}")
        
        # Check that errors match expected values
        self.assertTrue(torch.allclose(regular_error, expected_error, atol=1e-4),
                       f"Regular error {regular_error} != expected error {expected_error}")
        self.assertTrue(torch.allclose(autodiff_error, expected_error, atol=1e-4),
                       f"Autodiff error {autodiff_error} != expected error {expected_error}")
        
        # Test case 2: Point at (5,6) and target at (1,4)
        # Create both cost functions
        regular, autodiff = self.create_cost_functions(
            self.point2,
            self.target2,
            self.weight
        )
        
        # Calculate errors
        regular_error = regular.error()
        autodiff_error = autodiff.error()
        
        # Expected: sqrt(|5-1|) = sqrt(4) = 2 for x, sqrt(|6-4|) = sqrt(2) for y
        expected_error = torch.tensor([[torch.sqrt(torch.tensor(4.0)), torch.sqrt(torch.tensor(2.0))]])
        
        # Check that errors match between implementations
        self.assertTrue(torch.allclose(regular_error, autodiff_error, atol=1e-4),
                       f"Regular error {regular_error} != autodiff error {autodiff_error}")
        
        # Check that errors match expected values
        self.assertTrue(torch.allclose(regular_error, expected_error, atol=1e-4),
                       f"Regular error {regular_error} != expected error {expected_error}")
        self.assertTrue(torch.allclose(autodiff_error, expected_error, atol=1e-4),
                       f"Autodiff error {autodiff_error} != expected error {expected_error}")
    
    def test_jacobians(self):
        """Test the Jacobian calculations for both implementations."""
        # Test case 1: Point at (3,2) and target at (1,4)
        # Create both cost functions
        regular, autodiff = self.create_cost_functions(
            self.point1,
            self.target1,
            self.weight
        )
        
        # Calculate Jacobians
        regular_jacobians, regular_error = regular.jacobians()
        autodiff_jacobians, autodiff_error = autodiff.jacobians()
        
        # Expected Jacobian:
        # For the x coordinate: d(sqrt(|x-x_t|))/dx = sign(x-x_t) / (2*sqrt(|x-x_t|))
        # For 3-1=2, sign=1, so derivative is 1/(2*sqrt(2))
        # For the y coordinate: for 2-4=-2, sign=-1, so derivative is -1/(2*sqrt(2))
        batch_size = 1
        expected_jac = torch.zeros(batch_size, 2, 2)
        expected_jac[0, 0, 0] = 1.0 / (2.0 * torch.sqrt(torch.tensor(2.0)))
        expected_jac[0, 1, 1] = -1.0 / (2.0 * torch.sqrt(torch.tensor(2.0)))
        
        # Check that jacobians are similar between implementations
        # Note: Autodiff may have small numerical differences, so we use a slightly higher tolerance
        self.assertTrue(torch.allclose(regular_jacobians[0], autodiff_jacobians[0], atol=1e-3),
                       f"Regular jacobian {regular_jacobians[0]} != autodiff jacobian {autodiff_jacobians[0]}")
        
        # Check that regular implementation matches expected values
        self.assertTrue(torch.allclose(regular_jacobians[0], expected_jac, atol=1e-4),
                       f"Regular jacobian {regular_jacobians[0]} != expected jacobian {expected_jac}")
        
        # Test case 2: Point at (5,6) and target at (1,4)
        # Create both cost functions
        regular, autodiff = self.create_cost_functions(
            self.point2,
            self.target2,
            self.weight
        )
        
        # Calculate Jacobians
        regular_jacobians, regular_error = regular.jacobians()
        autodiff_jacobians, autodiff_error = autodiff.jacobians()
        
        # Expected Jacobian:
        # For 5-1=4, sign=1, so derivative is 1/(2*sqrt(4)) = 1/4
        # For 6-4=2, sign=1, so derivative is 1/(2*sqrt(2))
        expected_jac = torch.zeros(batch_size, 2, 2)
        expected_jac[0, 0, 0] = 1.0 / (2.0 * torch.sqrt(torch.tensor(4.0)))
        expected_jac[0, 1, 1] = 1.0 / (2.0 * torch.sqrt(torch.tensor(2.0)))
        
        # Check that jacobians are similar between implementations
        self.assertTrue(torch.allclose(regular_jacobians[0], autodiff_jacobians[0], atol=1e-3),
                       f"Regular jacobian {regular_jacobians[0]} != autodiff jacobian {autodiff_jacobians[0]}")
        
        # Check that regular implementation matches expected values
        self.assertTrue(torch.allclose(regular_jacobians[0], expected_jac, atol=1e-4),
                       f"Regular jacobian {regular_jacobians[0]} != expected jacobian {expected_jac}")
    
    def test_optimization(self):
        """Test that both implementations optimize similarly."""
        # Create variables for both implementations with the same initial values
        point_initial = torch.tensor([[5.0, 6.0]])
        target_initial = torch.tensor([[1.0, 4.0]])
        
        # Test the regular implementation
        regular_result = self._test_implementation(LinChkDistLoss, point_initial, target_initial, "regular")
        
        # Test the autodiff implementation
        autodiff_result = self._test_implementation(LinChkDistLossAutoDiff, point_initial, target_initial, "autodiff")
        
        # Compare the optimized results from both implementations
        print(f"Regular implementation result: {regular_result}")
        print(f"AutoDiff implementation result: {autodiff_result}")
        
        # Both implementations should converge to similar results
        self.assertTrue(torch.allclose(regular_result, autodiff_result, atol=self.tolerance),
                        "Regular and AutoDiff implementations should converge to similar results")
    
    def _test_implementation(self, cost_function_class, point_initial, target_initial, impl_name):
        """Test optimization with the given implementation and return the optimized point.
        
        Args:
            cost_function_class: The cost function class to test (LinChkDistLoss or LinChkDistLossAutoDiff)
            point_initial: Initial point tensor
            target_initial: Target point tensor
            impl_name: Name of the implementation for logging
            
        Returns:
            The optimized point tensor
        """
        # Create variables with unique names to avoid conflicts
        point = th.Point2(tensor=point_initial.clone(), name=f"point_{impl_name}")
        target = th.Point2(tensor=target_initial.clone(), name=f"target_{impl_name}")
        
        # Cost weight - using higher weight for stronger optimization
        cost_weight = th.ScaleCostWeight(10.0)
        
        # Create cost function
        cost = cost_function_class(point, target, cost_weight)
        
        # Calculate initial error
        initial_error = cost.error()
        initial_error_norm = torch.norm(initial_error).item()
        
        # Create objective
        objective = th.Objective()
        objective.add(cost)
        
        # Create optimizer using Levenberg-Marquardt
        optimizer = th.LevenbergMarquardt(
            objective,
            th.CholeskyDenseSolver,
            max_iterations=50,
            step_size=1.0,
        )
        
        # Create TheseusLayer
        layer = th.TheseusLayer(optimizer)
        
        # Prepare input dictionary
        inputs = {
            f"point_{impl_name}": point_initial, 
            f"target_{impl_name}": target_initial
        }
        
        # Run optimization through TheseusLayer.forward
        with torch.no_grad():  # No need for gradients in this test
            solution, info = layer.forward(inputs)
        
        # Get optimized point
        optimized_point = solution[f"point_{impl_name}"]
        
        # Calculate final error with the optimized point
        point.update(optimized_point)  # Update the point variable with the solution
        final_error = cost.error()
        final_error_norm = torch.norm(final_error).item()
        
        # Print results
        print(f"{impl_name.capitalize()} implementation:")
        print(f"  Initial error: {initial_error_norm:.4f}")
        print(f"  Final error: {final_error_norm:.4f}")
        
        # Different Theseus versions might have different attribute names
        iterations = getattr(info, 'num_iterations', None)
        if iterations is None:
            # Try alternative attribute names
            iterations = getattr(info, 'iterations', 'unknown')
        print(f"  Iterations used: {iterations}")
        print(f"  Optimized point: {optimized_point}")
        
        # Error should decrease
        self.assertLess(final_error_norm, initial_error_norm,
                       f"{impl_name} implementation failed to reduce error")
        
        return optimized_point


if __name__ == '__main__':
    import sys
    sys.tracebacklimit = None  # Show full stack traces
    from tests.test_utils import run_tests_with_full_stack_traces
    run_tests_with_full_stack_traces(TestLinChkDistLoss)