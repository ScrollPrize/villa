import unittest
import numpy as np
import torch
import os
import zarr
from pathlib import Path
from tracer.grid import PointGrid, STATE_LOC_VALID, STATE_COORD_VALID
from tracer.optimizer import SurfaceOptimizer
from tracer.fringe_expansion import FringeExpander
from tracer.core.interpolation import TrilinearInterpolator
from tests.volumes.test_coordinates import TEST_COORDINATES


class RealDataOptimizer(SurfaceOptimizer):
    """Optimizer that uses real zarr volume data for testing fringe expansion."""
    
    def __init__(self, grid, volume_path, resolution_level=0):
        """
        Initialize optimizer with real data.
        
        Args:
            grid: PointGrid to optimize
            volume_path: Path to zarr volume
            resolution_level: Resolution level to use
        """
        # Open the zarr array at the specified resolution level
        self.volume_path = Path(volume_path)
        self.resolution_level = str(resolution_level)
        zarr_volume = zarr.open(self.volume_path / self.resolution_level, mode='r')
        
        # Pass the zarr array directly to the parent class
        # The TrilinearInterpolator in the parent class supports zarr arrays
        super().__init__(grid, zarr_volume)
        
        # Track optimization calls
        self.optimize_calls = []
        self.last_loss_value = 0.1  # Initial loss value
    
    def optimize_points(self, candidate_points):
        """Implementation that uses real data interpolation."""
        self.optimize_calls.append(candidate_points)
        
        # For each candidate point, place it on a high-intensity region
        for y, x in candidate_points:
            # Get neighbor points
            neighbors = []
            for dy, dx in self.grid.neighbors:
                ny, nx = y + dy, x + dx
                if self.grid.is_valid(ny, nx):
                    neighbors.append(self.grid.get_point(ny, nx))
            
            if neighbors:
                # Use average of neighbors as starting point
                avg_pos = np.mean(neighbors, axis=0)
                
                # Find the highest intensity voxel in a small neighborhood around average
                best_pos = avg_pos.copy()
                best_value = 0.0
                
                # Search in a small neighborhood
                search_radius = 3
                for dz in range(-search_radius, search_radius+1):
                    for dy in range(-search_radius, search_radius+1):
                        for dx in range(-search_radius, search_radius+1):
                            # Calculate position
                            test_pos = avg_pos + np.array([dz, dy, dx])
                            
                            # Ensure within bounds
                            if (0 <= test_pos[0] < self.volume_shape[0] and
                                0 <= test_pos[1] < self.volume_shape[1] and
                                0 <= test_pos[2] < self.volume_shape[2]):
                                
                                # Sample intensity
                                intensity = self.sample_volume_at_point_3d(test_pos)
                                
                                # Update if better
                                if intensity > best_value:
                                    best_value = intensity
                                    best_pos = test_pos.copy()
                
                # Use the best position found
                self.grid.set_point(y, x, best_pos)
                
                # Update loss value based on best intensity
                self.last_loss_value = max(0.01, 1.0 - best_value)
            else:
                # If no valid neighbors, use one of our good test coordinates
                test_coord = TEST_COORDINATES[0]
                self.grid.set_point(y, x, np.array(test_coord, dtype=float))
    
    def get_last_loss(self):
        """Return the last computed loss value."""
        return self.last_loss_value
    
    def sample_volume_at_point_3d(self, point):
        """Sample volume at 3D point using the real data interpolator."""
        try:
            # Create input tensors
            z = torch.tensor([[float(point[0])]], dtype=torch.float32)
            y_val = torch.tensor([[float(point[1])]], dtype=torch.float32)
            x_val = torch.tensor([[float(point[2])]], dtype=torch.float32)
            
            # Call evaluate with proper tensor inputs
            value = self.interpolator.evaluate(z, y_val, x_val)
            return float(value.item())
        except Exception as e:
            print(f"ERROR: Error sampling at 3D point {point}: {e}")
            return 0.0


class TestFringeExpansion(unittest.TestCase):
    """Test fringe expansion implementation using real volume data."""
    
    def setUp(self):
        """Set up test environment with real data."""
        # Path to the real test volume
        self.volume_path = Path(__file__).parent / "volumes" / "s5_059_region_7300_3030_4555.zarr"
        
        # Check if the test volume exists
        if not os.path.exists(self.volume_path):
            self.skipTest(f"Test volume not found at {self.volume_path}")
        
        # Create a grid for testing - based on volume dimensions
        # For real volume we use a smaller grid to keep test runtime reasonable
        self.grid = PointGrid(100, 100)
        
        # Use one of our test coordinates as seed point
        seed_point = np.array(TEST_COORDINATES[0], dtype=float)
        self.grid.initialize_at_origin(seed_point, step_size=5.0)
        
        # Create optimizer with real data
        self.optimizer = RealDataOptimizer(self.grid, self.volume_path, resolution_level=1)
        
        # Create fringe expander with parameters tuned for real data
        self.expander = FringeExpander(
            grid=self.grid,
            optimizer=self.optimizer,
            reference_radius=2,  # Larger radius for real data
            max_reference_count=8,
            initial_reference_min=2,
            distance_threshold=3.0,  # Higher for real data
            intensity_threshold=0.05,  # Lower for real data
            max_optimization_tries=3,
            physical_fail_threshold=0.15,
            num_workers=2
        )
    
    def test_initial_state(self):
        """Test initial state of fringe expander with real data."""
        # Verify initial fringe size
        self.assertEqual(len(self.grid.fringe), 4, "Initial fringe should contain 4 seed points")
        
        # Verify all seed points are valid
        for y, x in self.grid.fringe:
            self.assertTrue(self.grid.is_valid(y, x), f"Seed point ({y}, {x}) should be valid")
    
    def test_collect_candidates(self):
        """Test candidate collection from fringe with real data."""
        # Collect candidates from initial fringe
        candidates = self.expander._collect_candidates_from_fringe()
        
        # Should find candidates around the seed points
        self.assertGreaterEqual(len(candidates), 4, "Should find at least 4 candidates")
        
        # Verify fringe is cleared
        self.assertEqual(len(self.grid.fringe), 0, "Fringe should be cleared after collecting candidates")
    
    def test_reference_counting(self):
        """Test reference point counting with real data."""
        # Create a simple grid with known pattern
        test_grid = PointGrid(10, 10)
        seed_coord = TEST_COORDINATES[1]  # Use second test coordinate
        test_grid.initialize_at_origin(np.array(seed_coord, dtype=float), step_size=2.0)
        
        # Override expander's grid
        old_grid = self.expander.grid
        self.expander.grid = test_grid
        
        # Count references for a point adjacent to the seed points
        y, x = test_grid.center
        y += 1  # One down from center
        ref_count, ref_points, best_ref = self.expander._count_reference_points(y, x)
        
        # Restore original grid
        self.expander.grid = old_grid
        
        # Should find at least 1 reference (the seed point)
        self.assertGreaterEqual(ref_count, 1, "Should find at least 1 reference point")
        self.assertIsNotNone(best_ref, "Should find a best reference point")
    
    def test_evaluate_candidate(self):
        """Test candidate evaluation with real data."""
        if len(self.grid.fringe) == 0:
            # If fringe is empty, collect some candidates first
            self.expander._collect_candidates_from_fringe()
            # And rebuild fringe with some valid points
            y, x = self.grid.center
            self.grid.fringe = [(y+1, x), (y, x+1), (y-1, x), (y, x-1)]
        
        # Get one of the fringe points for testing
        y, x = self.grid.fringe[0]
        
        # Get a candidate point adjacent to it
        cy, cx = y + 1, x + 1
        
        # Set candidate point in grid to have valid neighbors
        seed_coord = TEST_COORDINATES[0]
        self.grid.set_point(cy, cx, np.array(seed_coord, dtype=float))
        
        # Evaluate the candidate
        result = self.expander._evaluate_candidate(cy, cx)
        
        # This may succeed or fail depending on the real data, we just make sure it runs
        self.assertIsInstance(result, bool, "Candidate evaluation should return boolean")
    
    def test_one_generation_expansion(self):
        """Test expanding one generation with real data."""
        # Expand one generation
        new_points = self.expander.expand_one_generation()
        
        # Output results without strict assertions, as real data may give varied results
        print(f"Added {new_points} new points in first generation")
        print(f"Current fringe size: {len(self.grid.fringe)}")
        
        # We just test that it runs without errors
        self.assertIsInstance(new_points, int, "Should return number of new points")
    
    def test_multiple_generations_small(self):
        """Test expanding just a couple generations with real data."""
        # Expand two generations (limit for test speed)
        total_added = self.expander.expand_generations(2)
        
        # Output results without strict assertions
        print(f"Added {total_added} total points in 2 generations")
        print(f"Current fringe size: {len(self.grid.fringe)}")
        
        # Just test that it runs without errors
        self.assertIsInstance(total_added, int, "Should return number of total new points")
    
    def test_path_quality_evaluation(self):
        """Test path quality evaluation with real data."""
        # Use test coordinates
        start = np.array(TEST_COORDINATES[0], dtype=float)
        end = np.array(TEST_COORDINATES[1], dtype=float)
        
        # Calculate distance between points
        dist = np.linalg.norm(end - start)
        
        # Test the path quality (using relatively close points)
        if dist < 200:  # Only test if points are reasonably close
            quality = self.expander._evaluate_path_quality(start, end)
            self.assertIsInstance(quality, bool, "Path quality should be boolean")
        else:
            # Skip if points are too far apart
            print(f"Skipping path quality test - points too far apart: {dist} voxels")
            
        # Test with two points that are far apart (should be low quality)
        if len(TEST_COORDINATES) >= 10:
            distant_start = np.array(TEST_COORDINATES[0], dtype=float)
            distant_end = np.array(TEST_COORDINATES[9], dtype=float)  # Use 10th point
            
            # Test path quality between distant points
            quality = self.expander._evaluate_path_quality(distant_start, distant_end)
            
            # Path should be poor quality (crossing low intensity regions)
            print(f"Path quality between distant points: {quality}")
            # Don't assert the result as it depends on the specific points


if __name__ == '__main__':
    unittest.main()