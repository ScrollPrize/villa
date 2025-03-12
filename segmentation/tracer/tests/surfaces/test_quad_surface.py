#!/usr/bin/env python3
"""Test program for QuadSurface implementation."""

import os
import numpy as np
import tempfile
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from surfaces.visualization import convert_zyx_to_xyz, visualize_surface_points
from pathlib import Path
import unittest

import sys
# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from surfaces.quad_surface import QuadSurface, load_quad_from_tifxyz

def create_test_surface():
    """Create a test surface for demonstration."""
    # Create a half-sphere surface
    size = 50
    points = np.zeros((size, size, 3), dtype=np.float32)
    
    # Generate coordinates
    for j in range(size):
        for i in range(size):
            # Normalize coordinates to [-1, 1]
            x = 2.0 * i / (size - 1) - 1.0
            y = 2.0 * j / (size - 1) - 1.0
            
            # Calculate radius from center
            r = np.sqrt(x*x + y*y)
            
            if r > 1.0:
                # Outside of unit circle, mark as invalid
                points[j, i] = [-1, -1, -1]
            else:
                # Inside unit circle, set z coordinate based on half-sphere equation
                z = np.sqrt(1.0 - r*r)
                
                # Scale and position the surface - Use ZYX ordering [z, y, x]
                points[j, i] = [5 + 5*z, 10 + 5*y, 10 + 5*x]
    
    return points


class TestQuadSurface(unittest.TestCase):
    """Test cases for QuadSurface."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_points = create_test_surface()
        self.surface = QuadSurface(self.test_points, (1.0, 1.0))
    
    def test_initialization(self):
        """Test that surface is initialized correctly."""
        self.assertEqual(self.surface._points.shape, self.test_points.shape)
        self.assertTrue(np.array_equal(self.surface._scale, np.array([1.0, 1.0], dtype=np.float32)))
        
    def test_pointer(self):
        """Test pointer creation and manipulation."""
        ptr = self.surface.pointer()
        self.assertTrue(np.array_equal(ptr.loc, np.zeros(3, dtype=np.float32)))
        
        # Test moving the pointer
        self.surface.move(ptr, (5, 10, 2))
        self.assertTrue(np.array_equal(ptr.loc, np.array([5, 10, 2], dtype=np.float32)))
        
    def test_coordinate_conversion(self):
        """Test coordinate conversion methods."""
        # Since internal coordinate system has the center at (0,0,0) in nominal coordinates,
        # we need to test that a point at (0,0,0) maps to the center of the grid
        ptr = self.surface.pointer()  # Default is (0,0,0)
        
        # Get raw location (internal coordinates)
        loc_raw = self.surface.loc_raw(ptr)
        
        # Should map to center of the grid
        center_x = self.test_points.shape[1] / 2
        center_y = self.test_points.shape[0] / 2
        
        self.assertAlmostEqual(loc_raw[0], center_x, delta=1.0)
        self.assertAlmostEqual(loc_raw[1], center_y, delta=1.0)
        
        # Now test that we can move to a valid point and get its coordinate
        # Find a valid point in the test surface
        valid_point = None
        for j in range(self.test_points.shape[0]):
            for i in range(self.test_points.shape[1]):
                if self.test_points[j, i, 0] != -1:
                    valid_point = (i, j)
                    break
            if valid_point:
                break
        
        # Move to this point
        if valid_point:
            # Position relative to center
            rel_x = valid_point[0] - center_x
            rel_y = valid_point[1] - center_y
            
            # Move pointer to this relative position
            self.surface.move(ptr, (rel_x, rel_y, 0))
            
            # Get coordinate
            coord = self.surface.coord(ptr)
            
            # Should be valid
            self.assertNotEqual(coord[0], -1)
        
    def test_normal_calculation(self):
        """Test surface normal calculation."""
        # Pick center of hemisphere
        ptr = self.surface.pointer()  # Already at center
        
        # Get normal at this location
        normal = self.surface.normal(ptr)
        
        # At center of hemisphere, normal should point approximately up
        self.assertTrue(normal[2] > 0.9)  # Z component should be close to 1
        self.assertTrue(np.allclose(np.linalg.norm(normal), 1.0))  # Should be unit length
        
    def test_point_to(self):
        """Test point-to-surface search."""
        ptr = self.surface.pointer()
        
        # Search for a point near the top of the hemisphere
        target = [10, 10, 12]  # Above the center
        dist = self.surface.point_to(ptr, target, 10.0, 100)
        
        # Should find a point near the top
        found_coord = self.surface.coord(ptr)
        
        # Distance from found point to target should equal returned distance
        actual_dist = np.linalg.norm(found_coord - target)
        self.assertAlmostEqual(dist, actual_dist, places=4)
        
        # Should find a point close to the center in X and Y
        self.assertTrue(np.abs(found_coord[0] - 10) < 1.0)
        self.assertTrue(np.abs(found_coord[1] - 10) < 1.0)
        
    def test_bounding_box(self):
        """Test bounding box calculation."""
        bbox = self.surface.bbox()
        
        # Check bounds against known properties of our test surface - using ZYX ordering
        self.assertTrue(bbox.low[0] >= 5)   # Min Z
        self.assertTrue(bbox.low[1] >= 5)   # Min Y
        self.assertTrue(bbox.low[2] >= 5)   # Min X
        self.assertTrue(bbox.high[0] <= 10)  # Max Z
        self.assertTrue(bbox.high[1] <= 15)  # Max Y
        self.assertTrue(bbox.high[2] <= 15)  # Max X
        
    def test_save_load(self):
        """Test saving and loading surface."""
        # Add metadata
        self.surface.meta = {
            "test_key": "test_value",
            "description": "Test surface"
        }
        
        # Create temporary directory for test
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save surface
            save_path = Path(tmpdir) / "test_surface"
            self.surface.save(save_path, "test_uuid")
            
            # Check that files were created
            self.assertTrue(os.path.exists(save_path / "x.tif"))
            self.assertTrue(os.path.exists(save_path / "y.tif"))
            self.assertTrue(os.path.exists(save_path / "z.tif"))
            self.assertTrue(os.path.exists(save_path / "meta.json"))
            
            # Load surface
            loaded_surf = load_quad_from_tifxyz(save_path)
            
            # Check metadata
            self.assertEqual(loaded_surf.meta["test_key"], "test_value")
            self.assertEqual(loaded_surf.meta["uuid"], "test_uuid")
            
            # Compare surfaces at specific locations
            for i in range(0, self.test_points.shape[1], 10):
                for j in range(0, self.test_points.shape[0], 10):
                    if self.test_points[j, i, 0] != -1:  # If valid point
                        # Get original and loaded coordinates
                        ptr_orig = self.surface.pointer()
                        self.surface.move(ptr_orig, (i - self.test_points.shape[1]/2, 
                                                  j - self.test_points.shape[0]/2, 0))
                        orig_coord = self.surface.coord(ptr_orig)
                        
                        ptr_loaded = loaded_surf.pointer()
                        loaded_surf.move(ptr_loaded, (i - self.test_points.shape[1]/2,
                                                   j - self.test_points.shape[0]/2, 0))
                        loaded_coord = loaded_surf.coord(ptr_loaded)
                        
                        # Should be almost equal, if both are valid
                        if orig_coord[0] != -1 and loaded_coord[0] != -1:
                            self.assertTrue(np.allclose(orig_coord, loaded_coord, atol=1e-2))


def visualize_surface(surf, filename_prefix='quad_surface'):
    """Visualize a surface using matplotlib."""
    # Extract valid points
    points = surf.raw_points()
    valid_points = []
    
    for j in range(points.shape[0]):
        for i in range(points.shape[1]):
            if points[j, i, 0] != -1:
                valid_points.append(points[j, i])
    
    valid_points = np.array(valid_points)
    
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points using our visualization utility that handles ZYX to XYZ conversion
    ax = visualize_surface_points(
        valid_points, 
        ax=ax, 
        color='b',
        title='QuadSurface Visualization'
    )
    
    plt.savefig(f'{filename_prefix}_visualization.png')
    
    # Plot a few normal vectors
    center_x = points.shape[1] / 2
    center_y = points.shape[0] / 2
    step = 10
    scale = 0.5
    
    # Collect points and normals for batch visualization
    valid_points = []
    normals = []
    
    for j in range(0, points.shape[0], step):
        for i in range(0, points.shape[1], step):
            if points[j, i, 0] != -1:
                p = points[j, i]
                
                ptr = surf.pointer()
                # Use proper ZYX ordering for the move
                surf.move(ptr, (0, j - center_y, i - center_x))
                n = surf.normal(ptr)
                
                valid_points.append(p)
                normals.append(n)
    
    # Convert to numpy arrays
    valid_points = np.array(valid_points)
    normals = np.array(normals)
    
    # Use our visualization helper for points with normals
    from surfaces.visualization import visualize_surface_with_normals
    visualize_surface_with_normals(
        valid_points, 
        normals, 
        scale=scale, 
        ax=ax
    )
    
    plt.savefig(f'{filename_prefix}_with_normals.png')
    return True


def interactive_test():
    """Run interactive tests with printed output."""
    print("Testing basic QuadSurface functionality...")
    
    # Create test surface
    points = create_test_surface()
    surf = QuadSurface(points, (1.0, 1.0))
    
    # Test bounding box
    bbox = surf.bbox()
    print(f"Bounding box: Low={bbox.low}, High={bbox.high}")
    
    # Test coordinate mapping
    ptr = surf.pointer()
    
    # Center of surface
    surf.move(ptr, (25, 25, 0))
    coord = surf.coord(ptr)
    normal = surf.normal(ptr)
    print(f"Center coordinate: {coord}")
    print(f"Center normal: {normal}")
    
    # Test point finding
    target = [10, 10, 10]
    print(f"Finding closest point to {target}...")
    dist = surf.point_to(ptr, target, 10.0, 100)
    print(f"Found point at distance {dist}")
    found_loc = surf.loc(ptr)
    found_coord = surf.coord(ptr)
    print(f"Found location: {found_loc}")
    print(f"Found coordinate: {found_coord}")
    
    # Test save/load
    print("\nTesting save/load functionality...")
    
    # Add metadata
    surf.meta = {
        "test_key": "test_value",
        "description": "Test surface"
    }
    
    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save surface
        save_path = Path(tmpdir) / "test_surface"
        surf.save(save_path, "test_uuid")
        print(f"Saved surface to {save_path}")
        
        # Check files
        files = os.listdir(save_path)
        print(f"Created files: {files}")
        
        # Load surface
        loaded_surf = load_quad_from_tifxyz(save_path)
        print(f"Loaded surface with metadata: {loaded_surf.meta}")
        
        # Compare surfaces
        original_bbox = surf.bbox()
        loaded_bbox = loaded_surf.bbox()
        print(f"Original bbox: Low={original_bbox.low}, High={original_bbox.high}")
        print(f"Loaded bbox: Low={loaded_bbox.low}, High={loaded_bbox.high}")
        
        # Compare points at specific location
        ptr = surf.pointer()
        surf.move(ptr, (25, 25, 0))
        orig_coord = surf.coord(ptr)
        
        ptr_loaded = loaded_surf.pointer()
        loaded_surf.move(ptr_loaded, (25, 25, 0))
        loaded_coord = loaded_surf.coord(ptr_loaded)
        
        print(f"Original coordinate at (25,25): {orig_coord}")
        print(f"Loaded coordinate at (25,25): {loaded_coord}")
        
        # Calculate error
        error = np.linalg.norm(orig_coord - loaded_coord)
        print(f"Load/save error: {error}")
        
        if error < 1e-3:
            print("Save/load test PASSED")
        else:
            print("Save/load test FAILED")
    
    # Create visualization
    print("\nCreating visualization...")
    visualize_surface(surf)
    print("Visualization saved to 'quad_surface_visualization.png' and 'quad_surface_with_normals.png'")
    
    print("\nAll interactive tests completed.")


if __name__ == "__main__":
    # First run automated unit tests
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    
    # Then run interactive tests with detailed output
    interactive_test()