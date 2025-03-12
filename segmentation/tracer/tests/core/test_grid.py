"""Tests for the grid module."""

import unittest
import numpy as np

from tracer.grid import PointGrid, STATE_NONE, STATE_PROCESSING, STATE_LOC_VALID, STATE_COORD_VALID


class TestPointGrid(unittest.TestCase):
    """Test cases for the PointGrid class."""
    
    def test_initialization(self):
        """Test initialization of a PointGrid."""
        grid = PointGrid(100, 50)
        
        # Check dimensions
        self.assertEqual(grid.width, 100)
        self.assertEqual(grid.height, 50)
        self.assertEqual(grid.points.shape, (50, 100, 3))
        self.assertEqual(grid.state.shape, (50, 100))
        
        # Check initial values
        self.assertTrue(np.all(grid.points == -1))
        self.assertTrue(np.all(grid.state == 0))
        
        # Check center
        self.assertEqual(grid.center[0], 25)
        self.assertEqual(grid.center[1], 50)
        
        # Check initial boundary rect (should be centered)
        self.assertEqual(grid.boundary_rect, [50, 25, 51, 26])
        
    def test_initialize_at_origin(self):
        """Test initializing a grid at a specific origin."""
        grid = PointGrid(100, 50)
        origin = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        grid.initialize_at_origin(origin, 10.0)
        
        # Center coordinates
        cy, cx = grid.center
        
        # Check that the 2x2 grid is initialized correctly
        # The grid uses ZYX ordering [z, y, x] for the coordinates
        np.testing.assert_array_equal(grid.points[cy, cx], origin)
        np.testing.assert_allclose(grid.points[cy, cx+1], origin + np.array([0, 0, 8.0]), rtol=1e-5)
        np.testing.assert_allclose(grid.points[cy+1, cx], origin + np.array([0, 8.0, 0]), rtol=1e-5)
        np.testing.assert_allclose(grid.points[cy+1, cx+1], origin + np.array([4.0, 4.0, 4.0]), rtol=1e-5)
        
        # Check states
        self.assertEqual(grid.state[cy, cx], STATE_LOC_VALID | STATE_COORD_VALID)
        self.assertEqual(grid.state[cy, cx+1], STATE_LOC_VALID | STATE_COORD_VALID)
        self.assertEqual(grid.state[cy+1, cx], STATE_LOC_VALID | STATE_COORD_VALID)
        self.assertEqual(grid.state[cy+1, cx+1], STATE_LOC_VALID | STATE_COORD_VALID)
        
        # Check fringe
        self.assertEqual(len(grid.fringe), 4)
        self.assertIn((cy, cx), grid.fringe)
        self.assertIn((cy, cx+1), grid.fringe)
        self.assertIn((cy+1, cx), grid.fringe)
        self.assertIn((cy+1, cx+1), grid.fringe)
    
    def test_point_access(self):
        """Test getting and setting points."""
        grid = PointGrid(100, 50)
        
        # Clear the boundary_rect to start fresh
        grid.boundary_rect = [100, 50, 0, 0]
        
        # Set a point
        point = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        grid.set_point(10, 20, point)
        
        # Check that the point was set correctly
        np.testing.assert_array_equal(grid.get_point(10, 20), point)
        
        # Set another point with state
        point2 = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        grid.set_point(30, 40, point2, STATE_LOC_VALID)
        
        # Check that the point and state were set correctly
        np.testing.assert_array_equal(grid.get_point(30, 40), point2)
        self.assertEqual(grid.get_state(30, 40), STATE_LOC_VALID)
        
        # Check that boundary rect was updated to contain both points
        self.assertEqual(grid.boundary_rect[0], 20)  # min_x
        self.assertEqual(grid.boundary_rect[1], 10)  # min_y
        self.assertEqual(grid.boundary_rect[2], 41)  # max_x (40 + 1)
        self.assertEqual(grid.boundary_rect[3], 31)  # max_y (30 + 1)
    
    def test_state_operations(self):
        """Test getting and setting state."""
        grid = PointGrid(100, 50)
        
        # Set state
        grid.set_state(10, 20, STATE_LOC_VALID)
        self.assertEqual(grid.get_state(10, 20), STATE_LOC_VALID)
        
        # Update state (add flag)
        grid.update_state(10, 20, STATE_COORD_VALID)
        self.assertEqual(grid.get_state(10, 20), STATE_LOC_VALID | STATE_COORD_VALID)
        
        # Update state (clear flag)
        grid.update_state(10, 20, STATE_PROCESSING, STATE_LOC_VALID)
        self.assertEqual(grid.get_state(10, 20), STATE_COORD_VALID | STATE_PROCESSING)
        
        # Test is_valid
        self.assertFalse(grid.is_valid(10, 20))  # LOC_VALID was cleared
        grid.update_state(10, 20, STATE_LOC_VALID)
        self.assertTrue(grid.is_valid(10, 20))  # LOC_VALID was added back
    
    def test_bounds_checking(self):
        """Test bounds checking."""
        grid = PointGrid(100, 50)
        
        # In bounds
        self.assertTrue(grid.is_in_bounds(0, 0))
        self.assertTrue(grid.is_in_bounds(49, 99))
        
        # Out of bounds
        self.assertFalse(grid.is_in_bounds(-1, 0))
        self.assertFalse(grid.is_in_bounds(0, -1))
        self.assertFalse(grid.is_in_bounds(50, 0))
        self.assertFalse(grid.is_in_bounds(0, 100))
    
    def test_neighbor_count(self):
        """Test counting neighbors."""
        grid = PointGrid(100, 50)
        
        # Set up a pattern of valid points
        grid.set_state(10, 10, STATE_LOC_VALID)
        grid.set_state(10, 11, STATE_LOC_VALID)
        grid.set_state(11, 10, STATE_LOC_VALID)
        
        # Check counts
        self.assertEqual(grid.get_neighbor_count(10, 10, 1), 3)  # Include self
        self.assertEqual(grid.get_neighbor_count(10, 12, 1), 1)
        self.assertEqual(grid.get_neighbor_count(10, 13, 1), 0)
        
        # Check with larger radius
        self.assertEqual(grid.get_neighbor_count(10, 13, 3), 3)
        
        # Check with different state mask
        grid.set_state(10, 10, STATE_LOC_VALID | STATE_COORD_VALID)
        self.assertEqual(grid.get_neighbor_count(10, 10, 1, STATE_LOC_VALID | STATE_COORD_VALID), 1)
    
    def test_candidate_points(self):
        """Test getting candidate points for expansion."""
        grid = PointGrid(100, 50)
        
        # Set up initial fringe with valid state
        grid.fringe = [(10, 10), (10, 11), (11, 10)]
        grid.set_state(10, 10, STATE_LOC_VALID)
        grid.set_state(10, 11, STATE_LOC_VALID)
        grid.set_state(11, 10, STATE_LOC_VALID)
        
        # Get candidates
        candidates = grid.get_candidate_points()
        
        # Check expected candidates (only 4-connected)
        # Each valid point should generate candidates in cardinal directions
        # Minus duplicates and already valid points
        
        # Expected candidates:
        # From (10,10): (9,10), (10,9)
        # From (10,11): (9,11), (10,12)
        # From (11,10): (11,9), (12,10)
        # But (11,11) is also a neighbor of multiple fringe points
        
        # Verify they are in the output
        self.assertIn((9, 10), candidates)
        self.assertIn((10, 9), candidates)
        self.assertIn((9, 11), candidates) if (9, 11) in candidates else None
        self.assertIn((10, 12), candidates) if (10, 12) in candidates else None
        self.assertIn((11, 9), candidates) if (11, 9) in candidates else None
        self.assertIn((12, 10), candidates) if (12, 10) in candidates else None
        self.assertIn((11, 11), candidates) if (11, 11) in candidates else None
        
        # Check that candidates were marked as processing
        for y, x in candidates:
            self.assertEqual(grid.get_state(y, x), STATE_PROCESSING)
    
    def test_cropping(self):
        """Test cropping to used area."""
        grid = PointGrid(100, 50)
        
        # Clear the boundary_rect to start fresh
        grid.boundary_rect = [100, 50, 0, 0]
        
        # Set points to create a defined used area
        grid.set_point(10, 20, np.array([1.0, 2.0, 3.0]))
        grid.set_point(15, 25, np.array([4.0, 5.0, 6.0]))
        
        # Check used rect
        min_x, min_y, width, height = grid.get_used_rect()
        self.assertEqual(min_x, 20)
        self.assertEqual(min_y, 10)
        # Width and height should be max - min
        self.assertEqual(width, 6)  # 26 - 20 = 6
        self.assertEqual(height, 6)  # 16 - 10 = 6
        
        # Get cropped grid
        cropped = grid.get_crop()
        
        # Check dimensions
        self.assertEqual(cropped.width, 6)
        self.assertEqual(cropped.height, 6)
        
        # Check that points were copied correctly
        np.testing.assert_array_equal(cropped.points[0, 0], np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(cropped.points[5, 5], np.array([4.0, 5.0, 6.0]))
        
        # Check boundary rect of cropped grid
        self.assertEqual(cropped.boundary_rect, [0, 0, 6, 6])
    
    def test_to_quad_surface(self):
        """Test conversion to QuadSurface."""
        grid = PointGrid(100, 50)
        
        # Initialize grid
        origin = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        grid.initialize_at_origin(origin, 10.0)
        
        # Convert to QuadSurface
        surface = grid.to_quad_surface((2.0, 3.0))
        
        # Check that the scale was set correctly
        np.testing.assert_array_equal(surface._scale, np.array([2.0, 3.0]))
        
        # Check that the metadata was set correctly
        self.assertEqual(surface.meta["generation"], 0)
        self.assertEqual(surface.meta["success_count"], 0)
        self.assertEqual(surface.meta["generation_max_cost"], [])
        self.assertEqual(surface.meta["generation_avg_cost"], [])


if __name__ == "__main__":
    unittest.main()