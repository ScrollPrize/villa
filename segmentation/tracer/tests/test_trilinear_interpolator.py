"""Tests for TrilinearInterpolator."""

import unittest

import torch
import numpy as np

from cost_functions import TrilinearInterpolator


class TestTrilinearInterpolator(unittest.TestCase):
    """Test cases for TrilinearInterpolator."""
    
    def test_initialization(self):
        """Test initializing the interpolator with different volume shapes."""
        # Test with 3D volume
        volume_3d = torch.zeros((4, 4, 4))
        interpolator = TrilinearInterpolator(volume_3d)
        self.assertEqual(interpolator.shape, [4, 4, 4])
        self.assertEqual(interpolator.batch_size, 1)
        
        # Test with 4D volume (batch dimension)
        volume_4d = torch.zeros((2, 4, 4, 4))
        interpolator = TrilinearInterpolator(volume_4d)
        self.assertEqual(interpolator.shape, [4, 4, 4])
        self.assertEqual(interpolator.batch_size, 2)
        
        # Test with invalid shape
        with self.assertRaises(ValueError):
            TrilinearInterpolator(torch.zeros((4, 4)))
    
    def test_interpolation_constant_volume(self):
        """Test interpolation in a constant volume."""
        # Create a constant volume filled with ones
        volume = torch.ones((4, 4, 4))
        interpolator = TrilinearInterpolator(volume)
        
        # Test interpolation at various points
        z = torch.tensor([[1.5, 2.2]])
        y = torch.tensor([[1.5, 1.8]])
        x = torch.tensor([[1.5, 1.0]])
        
        values = interpolator.evaluate(z, y, x)
        
        # All values should be 1.0 in a constant volume
        expected = torch.ones((1, 2))
        self.assertTrue(torch.allclose(values, expected, atol=1e-5))
    
    def test_interpolation_gradient_volume(self):
        """Test interpolation in a volume with linear gradients."""
        # Create a 4x4x4 volume with gradients along each axis
        volume = torch.zeros((4, 4, 4))
        for z in range(4):
            for y in range(4):
                for x in range(4):
                    volume[z, y, x] = float(z + y + x)
        
        interpolator = TrilinearInterpolator(volume)
        
        # Test interpolation at integer points (should match exactly)
        z = torch.tensor([[1.0, 2.0]])
        y = torch.tensor([[1.0, 2.0]])
        x = torch.tensor([[1.0, 0.0]])
        
        values = interpolator.evaluate(z, y, x)
        
        expected = torch.tensor([[3.0, 4.0]])  # 1+1+1=3, 2+2+0=4
        self.assertTrue(torch.allclose(values, expected, atol=1e-5))
        
        # Test interpolation at non-integer points
        z = torch.tensor([[1.5, 2.5]])
        y = torch.tensor([[1.5, 1.0]])
        x = torch.tensor([[1.0, 0.5]])
        
        values = interpolator.evaluate(z, y, x)
        
        # At (1.5, 1.5, 1.0): Should be midway between (1,1,1)=3 and (2,2,1)=5, so 4.0
        # At (2.5, 1.0, 0.5): Should be an average of surrounding values
        expected = torch.tensor([[4.0, 4.0]])
        self.assertTrue(torch.allclose(values, expected, atol=1e-5))
    
    def test_gradient_calculation(self):
        """Test gradient calculation in the interpolator."""
        # Create a 4x4x4 volume with gradients along each axis
        volume = torch.zeros((4, 4, 4))
        for z in range(4):
            for y in range(4):
                for x in range(4):
                    volume[z, y, x] = float(z + 2*y + 3*x)  # Different weights for each axis
        
        interpolator = TrilinearInterpolator(volume)
        
        # Test at a midpoint location
        z = torch.tensor([[1.5]])
        y = torch.tensor([[1.5]])
        x = torch.tensor([[1.5]])
        
        values, gradients = interpolator.evaluate_with_gradient(z, y, x)
        
        # Expected gradients should match the coefficients in our volume
        # dv/dz = 1, dv/dy = 2, dv/dx = 3
        expected_gradients = torch.tensor([[[1.0, 2.0, 3.0]]])
        
        self.assertTrue(torch.allclose(gradients, expected_gradients, atol=1e-5))
        
    def test_interpolation_boundary_handling(self):
        """Test that boundary conditions are handled properly."""
        # Create a 4x4x4 volume with gradients along each axis
        volume = torch.zeros((4, 4, 4))
        for z in range(4):
            for y in range(4):
                for x in range(4):
                    volume[z, y, x] = float(z + y + x)
        
        interpolator = TrilinearInterpolator(volume)
        
        # Test interpolation with points slightly out of bounds
        z = torch.tensor([[-0.5, 4.5]])  # Outside of [0,3]
        y = torch.tensor([[0.5, 0.5]])
        x = torch.tensor([[0.5, 0.5]])
        
        values = interpolator.evaluate(z, y, x)
        
        # For z=-0.5, should clamp to z=0 and give 0+0.5+0.5 = 1.0
        # For z=4.5, should clamp to z=3 and give 3+0.5+0.5 = 4.0
        expected = torch.tensor([[1.0, 4.0]])
        self.assertTrue(torch.allclose(values, expected, atol=1e-5))


if __name__ == '__main__':
    unittest.main()