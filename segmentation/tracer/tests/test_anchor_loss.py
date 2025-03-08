"""Tests for AnchorLoss cost function."""

import unittest

import torch
import theseus as th

from cost_functions import AnchorLoss, TrilinearInterpolator


class TestAnchorLoss(unittest.TestCase):
    """Test cases for AnchorLoss."""
    
    def test_anchor_loss_error(self):
        """Test the error calculation for AnchorLoss."""
        batch_size = 1
        
        # Create a simple 3D volume 
        volume = torch.zeros((4, 4, 4))
        for z in range(4):
            for y in range(4):
                for x in range(4):
                    # Volume values increase with distance from center (2,2,2)
                    center = torch.tensor([2.0, 2.0, 2.0])
                    pos = torch.tensor([float(z), float(y), float(x)])
                    dist = torch.norm(pos - center)
                    volume[z, y, x] = 1.0 + dist  # Values > 1.0 near edges, ~1.0 at center
        
        # Create an interpolator
        interpolator = TrilinearInterpolator(volume)
        
        # Create points
        point = th.Point3(tensor=torch.tensor([[1.0, 1.0, 1.0]]))  # Point to be anchored
        anchor_point = th.Point3(tensor=torch.tensor([[2.0, 2.0, 2.0]]))  # Anchor at volume center
        
        # Create cost weight
        cost_weight = th.ScaleCostWeight(1.0)
        
        # Create the cost function
        anchor_loss = AnchorLoss(point, anchor_point, interpolator, cost_weight)
        
        # Calculate error
        error = anchor_loss.error()
        
        # Expected errors:
        # - Volume term: The value at (2,2,2) should be ~1.0, so v-1 = 0, 
        #   then clamped to 0, then squared = 0
        # - Distance term: Distance from (1,1,1) to (2,2,2) = sqrt(3) ≈ 1.732
        expected_error = torch.tensor([[0.0, torch.sqrt(torch.tensor(3.0)).item()]])
        
        # Check error values
        self.assertTrue(torch.allclose(error, expected_error, atol=1e-4))
        
        # Test with a different anchor point (where volume value > 1)
        point = th.Point3(tensor=torch.tensor([[1.0, 1.0, 1.0]]))
        anchor_point = th.Point3(tensor=torch.tensor([[0.0, 0.0, 0.0]]))  # Corner of volume
        anchor_loss = AnchorLoss(point, anchor_point, interpolator, cost_weight)
        
        error = anchor_loss.error()
        
        # Expected errors:
        # - Volume term: The value at (0,0,0) should be ~1.0 + sqrt(12) ≈ 4.464, 
        #   so v-1 ≈ 3.464, then squared ≈ 12.0
        # - Distance term: Distance from (1,1,1) to (0,0,0) = sqrt(3) ≈ 1.732
        value_at_origin = 1.0 + torch.sqrt(torch.tensor(12.0)).item()
        vol_term = (value_at_origin - 1.0) ** 2
        expected_error = torch.tensor([[vol_term, torch.sqrt(torch.tensor(3.0)).item()]])
        
        # Check error values (with higher tolerance due to interpolation)
        self.assertTrue(torch.allclose(error, expected_error, atol=0.1))
    
    def test_anchor_loss_jacobians(self):
        """Test the Jacobian calculations for AnchorLoss."""
        batch_size = 1
        
        # Create a simple 3D volume with gradients in each dimension
        volume = torch.zeros((4, 4, 4))
        for z in range(4):
            for y in range(4):
                for x in range(4):
                    volume[z, y, x] = float(1*z + 2*y + 3*x + 4)  # Linear function with known gradients
        
        # Create an interpolator
        interpolator = TrilinearInterpolator(volume)
        
        # Create points - separate them to get non-zero distance term
        point = th.Point3(tensor=torch.tensor([[1.5, 1.5, 1.5]]))
        anchor_point = th.Point3(tensor=torch.tensor([[2.5, 2.5, 2.5]]))
        
        # Cost weight
        cost_weight = th.ScaleCostWeight(1.0)
        
        # Create the cost function
        anchor_loss = AnchorLoss(point, anchor_point, interpolator, cost_weight)
        
        # Calculate Jacobians
        jacobians, error = anchor_loss.jacobians()
        
        # Calculate expected error
        # Value at (2.5, 2.5, 2.5) = 1*2.5 + 2*2.5 + 3*2.5 + 4 = 2.5 + 5 + 7.5 + 4 = 19.0
        # Volume term = (19.0 - 1.0)^2 = 18^2 = 324
        # Distance term = sqrt((1.5-2.5)^2 + (1.5-2.5)^2 + (1.5-2.5)^2) = sqrt(3) ≈ 1.732
        value_at_anchor = 19.0  # = 1*2.5 + 2*2.5 + 3*2.5 + 4
        vol_term = (value_at_anchor - 1.0) ** 2
        dist_term = torch.sqrt(torch.tensor(3.0)).item()
        expected_error = torch.tensor([[vol_term, dist_term]])
        
        # Check error values (with higher tolerance for floating point)
        self.assertTrue(torch.allclose(error, expected_error, atol=0.1))
        
        # Expected Jacobians
        # 1. Point Jacobian: 
        #    - First row (volume term) is all zeros (point doesn't affect volume term)
        #    - Second row (distance term) is the normalized direction from point to anchor:
        #      [-1.0/sqrt(3), -1.0/sqrt(3), -1.0/sqrt(3)]
        norm_factor = 1.0 / torch.sqrt(torch.tensor(3.0)).item()
        expected_point_jac = torch.zeros(batch_size, 2, 3)
        expected_point_jac[:, 1, :] = norm_factor * torch.tensor([-1.0, -1.0, -1.0])
        
        # 2. Anchor Point Jacobian:
        #    - First row (volume term) is 2*(value-1)*[gradient]
        #      where gradient = [1, 2, 3] (from our volume definition)
        #    - Second row (distance term) is the normalized direction from anchor to point:
        #      [1.0/sqrt(3), 1.0/sqrt(3), 1.0/sqrt(3)]
        expected_anchor_jac = torch.zeros(batch_size, 2, 3)
        # Volume term gradient: 2*(19.0-1)*[1,2,3] = 36*[1,2,3] = [36, 72, 108]
        expected_anchor_jac[:, 0, :] = 2.0 * (value_at_anchor - 1.0) * torch.tensor([1.0, 2.0, 3.0])
        expected_anchor_jac[:, 1, :] = norm_factor * torch.tensor([1.0, 1.0, 1.0])
        
        # Check Jacobian values (with higher tolerance for floating point calculations)
        self.assertTrue(torch.allclose(jacobians[0], expected_point_jac, atol=0.1))
        self.assertTrue(torch.allclose(jacobians[1], expected_anchor_jac, atol=0.1))
    
    def test_optimization(self):
        """Test that we can optimize a simple problem with AnchorLoss."""
        batch_size = 1
        
        # Create a simple 3D volume where values are higher at the center
        volume = torch.zeros((16, 16, 16))
        center = torch.tensor([8.0, 8.0, 8.0])  # Center of the volume
        
        for z in range(16):
            for y in range(16):
                for x in range(16):
                    pos = torch.tensor([float(z), float(y), float(x)])
                    # Distance from center
                    dist = torch.norm(pos - center)
                    # Create a hill shape with peak of 3.0 at center
                    # Values will be > 1.0 near center, < 1.0 at edges
                    volume[z, y, x] = 3.0 * torch.exp(-0.05 * dist**2)
        
        # Create an interpolator
        interpolator = TrilinearInterpolator(volume)
        
        # Initial point - far from the center
        point = th.Point3(tensor=torch.tensor([[4.0, 4.0, 4.0]]), name="point")
        
        # Initial anchor point - somewhere else in the volume
        anchor_point = th.Point3(tensor=torch.tensor([[12.0, 12.0, 12.0]]), name="anchor")
        
        # Create cost weight
        cost_weight = th.ScaleCostWeight(1.0)
        
        # Create the cost function
        anchor_loss = AnchorLoss(point, anchor_point, interpolator, cost_weight)
        
        # Initial error
        initial_error = anchor_loss.error()
        
        # Prepare input dictionary
        inputs = {
            "point": torch.tensor([[4.0, 4.0, 4.0]]),
            "anchor": torch.tensor([[12.0, 12.0, 12.0]])
        }
        
        # Get initial distance
        initial_distance = torch.norm(inputs['point'] - inputs['anchor']).item()
        print("Initial distance between point and anchor:", initial_distance)
        
        # Create an objective
        objective = th.Objective()
        objective.add(anchor_loss)
        
        # Create optimizer
        optimizer = th.LevenbergMarquardt(
            objective,
            th.CholeskyDenseSolver,
            max_iterations=50,
            step_size=1.0,
        )
        
        # Create TheseusLayer
        layer = th.TheseusLayer(optimizer)
            
        # Optimize
        with torch.no_grad():
            final_values, info = layer.forward(inputs)
        
        # Get optimized values
        optimized_point = final_values['point']
        optimized_anchor = final_values['anchor']
        
        print("Initial point:", inputs['point'])
        print("Initial anchor:", inputs['anchor'])
        print("Optimized point:", optimized_point)
        print("Optimized anchor:", optimized_anchor)
        
        # Create a new loss function with the optimized points
        optimized_anchor_loss = AnchorLoss(
            th.Point3(tensor=optimized_point),
            th.Point3(tensor=optimized_anchor),
            interpolator,
            cost_weight
        )
        
        # Get the final error
        final_error = optimized_anchor_loss.error()
        print("Initial error:", initial_error)
        print("Final error:", final_error)
        
        # Calculate final distance
        final_distance = torch.norm(optimized_point - optimized_anchor).item()
        print("Final distance between point and anchor:", final_distance)
        
        # The anchor should move toward the center where values > 1.0
        # to minimize the volume term
        center_distance_initial = torch.norm(inputs['anchor'] - center).item()
        center_distance_final = torch.norm(optimized_anchor - center).item()
        print("Distance from initial anchor to center:", center_distance_initial)
        print("Distance from final anchor to center:", center_distance_final)
        
        # The point should move closer to the anchor to minimize the distance term
        # The total error should decrease
        self.assertLess(torch.sum(final_error).item(), torch.sum(initial_error).item(),
                         "Total error should decrease after optimization")
        
        # The anchor should be closer to the center after optimization
        self.assertLess(center_distance_final, center_distance_initial,
                         "Anchor should move closer to center where volume values are higher")
        
        # The point and anchor should be closer together after optimization
        self.assertLess(final_distance, initial_distance,
                         "Point and anchor should move closer together")


if __name__ == '__main__':
    unittest.main()