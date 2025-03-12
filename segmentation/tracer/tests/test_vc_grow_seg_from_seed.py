#!/usr/bin/env python3
"""
Test comparing the Python and C++ implementations of vc_grow_seg_from_seed.

This test:
1. Runs both Python and C++ implementations on the same test volume
2. Uses the same configuration parameters
3. Compares the results to ensure consistency
4. Handles coordinate system differences (C++ uses XYZ, Python uses ZYX)
"""

import unittest
import os
import sys
import json
import shutil
import tempfile
import subprocess
import numpy as np
from pathlib import Path

from tests.volumes.test_coordinates import TEST_COORDINATES
import traceback

class TestVCGrowSegFromSeed(unittest.TestCase):
    """Test comparing Python and C++ implementations of vc_grow_seg_from_seed."""
    
    @classmethod
    def setUpClass(cls):
        """Set up the test environment."""
        # Volume path
        cls.volume_path = Path(__file__).parent / "volumes" / "s5_059_region_7300_3030_4555.zarr"
        
        # Verify volume exists
        if not cls.volume_path.exists():
            raise FileNotFoundError(f"Test volume not found: {cls.volume_path}")
        
        # Create temporary directory for output
        cls.temp_dir = Path(tempfile.mkdtemp())
        
        # Python implementation path
        cls.py_impl_path = Path(__file__).parent.parent / "vc_grow_seg_from_seed.py"
        
        # C++ implementation path
        cls.cpp_impl_path = Path(__file__).parent.parent / "volume-cartographer/build/bin/vc_grow_seg_from_seed"
        
        # Test if C++ implementation exists
        if not cls.cpp_impl_path.exists():
            raise FileNotFoundError(f"C++ implementation not found: {cls.cpp_impl_path}")
        
        # Default configuration
        cls.config = {
            "cache_size": 4294967296,  # 4GB
            "mode": "explicit_seed",
            "step_size": 10.0,
            "min_area_cm": 0.05,
            "tgt_overlap_count": 20,
            "search_effort": 10,
            "generations": 50,  # Reduced for faster tests
            "thread_limit": 8,
            "cache_root": str(cls.temp_dir / "cache")
        }
        
        # Create config file
        cls.config_path = cls.temp_dir / "test_config.json"
        with open(cls.config_path, 'w') as f:
            json.dump(cls.config, f, indent=2)
        
        # Create output directories
        cls.py_output_dir = cls.temp_dir / "python_output"
        cls.cpp_output_dir = cls.temp_dir / "cpp_output"
        os.makedirs(cls.py_output_dir, exist_ok=True)
        os.makedirs(cls.cpp_output_dir, exist_ok=True)
        
        # Create cache directory
        os.makedirs(cls.config["cache_root"], exist_ok=True)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up temporary directories."""
        try:
            shutil.rmtree(cls.temp_dir)
        except Exception as e:
            print(f"Error cleaning up: {e}")
    
    def run_python_implementation(self, seed_point_zyx):
        """
        Run the Python implementation with the given seed point.
        
        Args:
            seed_point_zyx: Seed point coordinates in ZYX order.
            
        Returns:
            Path to the output directory containing the generated surface.
        """
        # Convert ZYX to XYZ for command line arguments
        seed_x, seed_y, seed_z = seed_point_zyx[2], seed_point_zyx[1], seed_point_zyx[0]
        
        try:
            # Run Python implementation
            cmd = [
                sys.executable,
                str(self.py_impl_path),
                str(self.volume_path),
                str(self.py_output_dir),
                str(self.config_path),
                str(seed_x),
                str(seed_y),
                str(seed_z)
            ]
            
            result = subprocess.run(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                check=True,
                text=True
            )
            
            # Find the output directory (most recent auto_grown_* directory)
            output_dirs = sorted([
                d for d in os.listdir(self.py_output_dir) 
                if d.startswith("auto_grown_") and os.path.isdir(os.path.join(self.py_output_dir, d))
            ], reverse=True)
            
            if not output_dirs:
                print("Python implementation output:")
                print(result.stdout)
                print(result.stderr)
                raise FileNotFoundError("No output directory found after running Python implementation")
            
            return self.py_output_dir / output_dirs[0]
        
        except subprocess.CalledProcessError as e:
            print(f"Error running Python implementation: {e}")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
            raise
    
    def run_cpp_implementation(self, seed_point_zyx):
        """
        Run the C++ implementation with the given seed point.
        
        Args:
            seed_point_zyx: Seed point coordinates in ZYX order.
            
        Returns:
            Path to the output directory containing the generated surface.
        """
        # Convert ZYX to XYZ for C++ implementation
        # CRITICAL: C++ implementation expects XYZ order
        seed_x, seed_y, seed_z = seed_point_zyx[2], seed_point_zyx[1], seed_point_zyx[0]
        
        try:
            # Run C++ implementation
            cmd = [
                str(self.cpp_impl_path),
                str(self.volume_path),
                str(self.cpp_output_dir),
                str(self.config_path),
                str(seed_x),
                str(seed_y),
                str(seed_z)
            ]
            
            result = subprocess.run(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                check=True,
                text=True
            )
            
            # Find the output directory (most recent auto_grown_* directory)
            output_dirs = sorted([
                d for d in os.listdir(self.cpp_output_dir) 
                if d.startswith("auto_grown_") and os.path.isdir(os.path.join(self.cpp_output_dir, d))
            ], reverse=True)
            
            if not output_dirs:
                print("C++ implementation output:")
                print(result.stdout)
                print(result.stderr)
                raise FileNotFoundError("No output directory found after running C++ implementation")
            
            return self.cpp_output_dir / output_dirs[0]
            
        except subprocess.CalledProcessError as e:
            print(f"Error running C++ implementation: {e}")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
            raise
    
    def compare_surfaces(self, py_surface_dir, cpp_surface_dir):
        """
        Compare surfaces generated by Python and C++ implementations.
        
        Args:
            py_surface_dir: Path to Python output directory
            cpp_surface_dir: Path to C++ output directory
            
        Returns:
            dict: Comparison statistics
        """
        # Read Python surface
        py_meta_file = py_surface_dir / "meta.json"
        cpp_meta_file = cpp_surface_dir / "meta.json"
        
        self.assertTrue(py_meta_file.exists(), f"Python meta.json not found at {py_meta_file}")
        self.assertTrue(cpp_meta_file.exists(), f"C++ meta.json not found at {cpp_meta_file}")
        
        with open(py_meta_file, 'r') as f:
            py_meta = json.load(f)
        
        with open(cpp_meta_file, 'r') as f:
            cpp_meta = json.load(f)
        
        # Load x, y, z coordinates for both
        try:
            py_x = np.asarray(self.load_tif_file(py_surface_dir / "x.tif"))
            py_y = np.asarray(self.load_tif_file(py_surface_dir / "y.tif"))
            py_z = np.asarray(self.load_tif_file(py_surface_dir / "z.tif"))
            
            cpp_x = np.asarray(self.load_tif_file(cpp_surface_dir / "x.tif"))
            cpp_y = np.asarray(self.load_tif_file(cpp_surface_dir / "y.tif"))
            cpp_z = np.asarray(self.load_tif_file(cpp_surface_dir / "z.tif"))
        except Exception as e:
            print(f"Error loading TIF files: {e}")
            traceback.print_exc()
            raise
        
        # Get dimensions
        py_shape = py_x.shape
        cpp_shape = cpp_x.shape
        
        # Basic statistics
        stats = {
            "py_shape": py_shape,
            "cpp_shape": cpp_shape,
            "py_area_cm2": py_meta.get("area_cm2", 0),
            "cpp_area_cm2": cpp_meta.get("area_cm2", 0),
            "py_origin": py_meta.get("origin", [0, 0, 0]),
            "cpp_origin": cpp_meta.get("origin", [0, 0, 0]),
            "py_bbox": py_meta.get("bbox", {"min": [0, 0, 0], "max": [0, 0, 0]}),
            "cpp_bbox": cpp_meta.get("bbox", {"min": [0, 0, 0], "max": [0, 0, 0]})
        }
        
        # Calculate valid point counts
        py_valid = (py_x != -1).sum()
        cpp_valid = (cpp_x != -1).sum()
        
        stats["py_valid_points"] = py_valid
        stats["cpp_valid_points"] = cpp_valid
        stats["valid_point_ratio"] = cpp_valid / py_valid if py_valid > 0 else 0
        
        return stats

    def load_tif_file(self, file_path):
        """
        Load a TIFF file using PIL.
        
        Args:
            file_path: Path to the TIFF file
            
        Returns:
            numpy.ndarray: The loaded image data
        """
        try:
            from PIL import Image
            img = Image.open(file_path)
            return np.array(img)
        except ImportError:
            # Fallback to tifffile if PIL is not available
            import tifffile
            return tifffile.imread(file_path)
    
    def test_single_seed_point(self):
        """Test with a single seed point from TEST_COORDINATES."""
        # Select a seed point that tends to produce good results
        seed_point_zyx = TEST_COORDINATES[5]  # [253, 579, 848]
        
        print(f"Testing with seed point ZYX: {seed_point_zyx}")
        print(f"                       XYZ: {seed_point_zyx[2], seed_point_zyx[1], seed_point_zyx[0]}")
        
        # Run both implementations
        try:
            py_output_dir = self.run_python_implementation(seed_point_zyx)
            print(f"Python implementation output saved to: {py_output_dir}")
            
            cpp_output_dir = self.run_cpp_implementation(seed_point_zyx)
            print(f"C++ implementation output saved to: {cpp_output_dir}")
            
            # Compare the surfaces
            stats = self.compare_surfaces(py_output_dir, cpp_output_dir)
            
            # Print comparison statistics
            print("\nComparison Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            
            # Check that the results are reasonably similar
            # These thresholds can be adjusted based on expected differences
            self.assertGreater(stats["valid_point_ratio"], 0.7, 
                               "Valid point ratio too low, implementations differ significantly")
            
            # Check that the areas are similar (within 30%)
            py_area = stats["py_area_cm2"]
            cpp_area = stats["cpp_area_cm2"]
            area_ratio = min(py_area, cpp_area) / max(py_area, cpp_area) if max(py_area, cpp_area) > 0 else 0
            self.assertGreater(area_ratio, 0.7, "Surface areas differ significantly")
            
        except Exception as e:
            self.fail(f"Test failed with error: {e}")
            
    def test_multiple_seed_points(self):
        """Test with multiple seed points from TEST_COORDINATES."""
        # Select 3 seed points for more comprehensive testing
        # Skip if running in CI to save time
        if os.environ.get("CI") == "true":
            self.skipTest("Skipping multiple seed point test in CI environment")
            
        seed_points = [
            TEST_COORDINATES[0],  # [236, 171, 704]
            TEST_COORDINATES[5],  # [253, 579, 848]
            TEST_COORDINATES[10], # [547, 476, 252]
        ]
        
        results = []
        
        for i, seed_point_zyx in enumerate(seed_points):
            print(f"\nTest {i+1}/{len(seed_points)}:")
            print(f"Testing with seed point ZYX: {seed_point_zyx}")
            print(f"                       XYZ: {seed_point_zyx[2], seed_point_zyx[1], seed_point_zyx[0]}")
            
            try:
                # Run both implementations
                py_output_dir = self.run_python_implementation(seed_point_zyx)
                cpp_output_dir = self.run_cpp_implementation(seed_point_zyx)
                
                # Compare the surfaces
                stats = self.compare_surfaces(py_output_dir, cpp_output_dir)
                
                # Save results
                results.append({
                    "seed_point": seed_point_zyx,
                    "py_output": str(py_output_dir),
                    "cpp_output": str(cpp_output_dir),
                    "stats": stats
                })
                
                # Print comparison statistics
                print(f"Result for seed point {seed_point_zyx}:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                
            except Exception as e:
                print(f"Error with seed point {seed_point_zyx}: {e}")
                # Continue testing other seed points instead of failing immediately
                results.append({
                    "seed_point": seed_point_zyx,
                    "error": str(e)
                })
        
        # Check if at least 2 out of 3 tests succeeded
        success_count = sum(1 for r in results if "error" not in r)
        self.assertGreaterEqual(success_count, 2, 
                               f"Too many failures: {len(seed_points) - success_count} out of {len(seed_points)}")
        
        # For successful tests, verify similar results
        for result in results:
            if "error" in result:
                continue
                
            stats = result["stats"]
            # Check valid point ratio
            self.assertGreater(stats["valid_point_ratio"], 0.7, 
                              f"Valid point ratio too low for seed {result['seed_point']}")
            
            # Check area ratio
            py_area = stats["py_area_cm2"]
            cpp_area = stats["cpp_area_cm2"]
            area_ratio = min(py_area, cpp_area) / max(py_area, cpp_area) if max(py_area, cpp_area) > 0 else 0
            self.assertGreater(area_ratio, 0.7, 
                              f"Surface areas differ significantly for seed {result['seed_point']}")

if __name__ == "__main__":
    unittest.main()