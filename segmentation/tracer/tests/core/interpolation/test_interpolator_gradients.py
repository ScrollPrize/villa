"""Test that gradients flow correctly through the TrilinearInterpolatorAutoDiff."""

import unittest
import torch
import numpy as np
from tracer.core.interpolation.trilinear_interpolator_autodiff import TrilinearInterpolatorAutoDiff

class TestInterpolatorGradients(unittest.TestCase):
    def setUp(self):
        # Create a simple test volume
        self.volume_size = 20
        self.test_volume = torch.zeros((self.volume_size, self.volume_size, self.volume_size), 
                                        dtype=torch.float32)
        
        # Fill with a simple gradient pattern
        for z in range(self.volume_size):
            for y in range(self.volume_size):
                for x in range(self.volume_size):
                    self.test_volume[z, y, x] = z * 0.1 + y * 0.05 + x * 0.01
        
        # Create the interpolator
        self.interpolator = TrilinearInterpolatorAutoDiff(self.test_volume)
    
    def test_gradient_flow(self):
        """Test that gradients flow correctly through the interpolator."""
        # Create input coordinates that require gradients
        coords = torch.tensor([
            [10.5, 10.5, 10.5],  # Center point
            [5.2, 7.3, 9.1],     # Random point
            [15.8, 12.4, 8.7]    # Another random point
        ], dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        
        # Extract coordinate components and enable gradients
        z = coords[:, :, 0].requires_grad_(True)
        y = coords[:, :, 1].requires_grad_(True)
        x = coords[:, :, 2].requires_grad_(True)
        
        # Evaluate the interpolator with gradient
        values, gradients = self.interpolator.evaluate_with_gradient(z, y, x)
        
        # Verify values have gradients enabled
        self.assertTrue(gradients.requires_grad, "Gradients tensor should require gradients")
        
        # Compute a loss based on the interpolated values and backpropagate
        loss = values.sum()
        loss.backward()
        
        # Verify gradients exist
        self.assertIsNotNone(z.grad, "Z coordinate should have gradients")
        self.assertIsNotNone(y.grad, "Y coordinate should have gradients")
        self.assertIsNotNone(x.grad, "X coordinate should have gradients")
        
        print(f"Z grad shape: {z.grad.shape}, values: {z.grad}")
        print(f"Y grad shape: {y.grad.shape}, values: {y.grad}")
        print(f"X grad shape: {x.grad.shape}, values: {x.grad}")
        
        # Verify second order gradients by computing a loss based on the gradient magnitudes
        if gradients.requires_grad:
            gradient_loss = gradients.norm()
            gradient_loss.backward()
            
            # Check if second-order gradients exist
            if z.grad.grad_fn is not None:
                print("Second-order gradients successfully computed!")
            else:
                print("Second-order gradients not available.")
        else:
            print("Gradients tensor does not require gradients, cannot compute second-order gradients.")

    def test_higher_order_gradient(self):
        """Test that higher-order gradients work through the interpolator."""
        # Create input coordinates that require gradients with create_graph=True
        z = torch.tensor([[10.5, 5.2, 15.8]], dtype=torch.float32, requires_grad=True)
        y = torch.tensor([[10.5, 7.3, 12.4]], dtype=torch.float32, requires_grad=True)
        x = torch.tensor([[10.5, 9.1, 8.7]], dtype=torch.float32, requires_grad=True)
        
        # Define a function that uses our interpolator
        def interp_fn(z_in, y_in, x_in):
            values, _ = self.interpolator.evaluate_with_gradient(z_in, y_in, x_in)
            return values
        
        # First-order gradients with respect to z
        values = interp_fn(z, y, x)
        dvalues_dz = torch.autograd.grad(
            outputs=values, 
            inputs=z, 
            grad_outputs=torch.ones_like(values),
            create_graph=True  # Need this for higher-order gradients
        )[0]
        
        print(f"First-order gradients wrt z: {dvalues_dz}")
        
        # Second-order gradients (d²values/dz²)
        d2values_dz2 = torch.autograd.grad(
            outputs=dvalues_dz,
            inputs=z,
            grad_outputs=torch.ones_like(dvalues_dz),
            retain_graph=True
        )[0]
        
        print(f"Second-order gradients wrt z: {d2values_dz2}")
        
        # The second derivatives should be close to zero for our linear gradient pattern
        self.assertTrue(torch.all(torch.abs(d2values_dz2) < 1e-3), 
                        "Second derivatives should be near zero for linear function")

if __name__ == '__main__':
    unittest.main()