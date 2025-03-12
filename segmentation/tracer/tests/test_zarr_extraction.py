#!/usr/bin/env python
"""
Test the zarr bounding box extraction functionality.
This test will create a sample zarr volume, extract a bounding box,
and verify the extraction was done correctly.
"""

import unittest
import numpy as np
import zarr
import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path to import extract_zarr_bounding_box
sys.path.append(str(Path(__file__).parent.parent))
from extract_zarr_bounding_box import extract_bounding_box

class TestZarrExtraction(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        self.input_zarr_path = Path(self.temp_dir) / "test_input.zarr"
        self.output_zarr_path = Path(self.temp_dir) / "test_output.zarr"
        
        # Create a sample zarr volume with multiple resolution levels
        self.create_test_zarr_volume()
    
    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.temp_dir)
    
    def create_test_zarr_volume(self):
        """Create a sample zarr volume with multiple resolution levels"""
        # Create base resolution (level 0) - a 100x100x100 volume with gradient pattern
        os.makedirs(self.input_zarr_path / "0", exist_ok=True)
        
        z, y, x = np.ogrid[0:100, 0:100, 0:100]
        # Create a pattern where each dimension has a different gradient
        data = (z/100.0 + y/200.0 + x/300.0).astype(np.float32)
        
        # Add some structural features (spheres) at various locations
        for center, radius in [
            ((30, 30, 30), 10),
            ((70, 70, 70), 15),
            ((30, 70, 50), 8)
        ]:
            cz, cy, cx = center
            sphere_mask = ((z-cz)**2 + (y-cy)**2 + (x-cx)**2) < radius**2
            data[sphere_mask] = 1.0
            
        # Create level 0
        array_0 = zarr.create(
            shape=data.shape,
            chunks=(20, 20, 20),
            dtype=np.float32,
            store=self.input_zarr_path / "0"
        )
        array_0[:] = data
        array_0.attrs['dimension_separator'] = '/'
        
        # Create level 1 (downsampled by 2)
        os.makedirs(self.input_zarr_path / "1", exist_ok=True)
        data_1 = data[::2, ::2, ::2]
        array_1 = zarr.create(
            shape=data_1.shape,
            chunks=(10, 10, 10),
            dtype=np.float32,
            store=self.input_zarr_path / "1"
        )
        array_1[:] = data_1
        array_1.attrs['dimension_separator'] = '/'
        
        # Create level 2 (downsampled by 4)
        os.makedirs(self.input_zarr_path / "2", exist_ok=True)
        data_2 = data[::4, ::4, ::4]
        array_2 = zarr.create(
            shape=data_2.shape,
            chunks=(5, 5, 5),
            dtype=np.float32,
            store=self.input_zarr_path / "2"
        )
        array_2[:] = data_2
        array_2.attrs['dimension_separator'] = '/'
        
        # Create metadata
        metadata = {
            "voxelsize": [1.0, 1.0, 1.0],
            "resolution_levels": 3,
            "created_for": "test_zarr_extraction"
        }
        
        with open(self.input_zarr_path / "meta.json", 'w') as f:
            import json
            json.dump(metadata, f)
    
    def test_extraction(self):
        """Test extracting a bounding box from the zarr volume"""
        # Define center and box size
        center_zyx = (50, 50, 50)
        box_size_zyx = (40, 40, 40)
        
        # Extract the bounding box
        extract_bounding_box(
            self.input_zarr_path,
            self.output_zarr_path,
            center_zyx,
            box_size_zyx
        )
        
        # Verify extraction results
        self.assertTrue(os.path.exists(self.output_zarr_path), 
                       "Output zarr directory not created")
        self.assertTrue(os.path.exists(self.output_zarr_path / "meta.json"), 
                       "Metadata file not created")
        
        # Check all resolution levels
        for level in ["0", "1", "2"]:
            level_path = self.output_zarr_path / level
            self.assertTrue(os.path.exists(level_path), 
                           f"Resolution level {level} not created")
            
            # Open the arrays
            input_array = zarr.open(self.input_zarr_path / level, mode='r')
            output_array = zarr.open(level_path, mode='r')
            
            # Verify dimensions
            scale = 2 ** int(level)
            expected_shape = tuple(s // scale for s in box_size_zyx)
            self.assertEqual(output_array.shape, expected_shape,
                            f"Wrong shape at level {level}: got {output_array.shape}, expected {expected_shape}")
            
            # Verify some data values from the original array were preserved
            # For level 0, we can do precise checking
            if level == "0":
                z_start = max(0, center_zyx[0] - box_size_zyx[0]//2)
                y_start = max(0, center_zyx[1] - box_size_zyx[1]//2)
                x_start = max(0, center_zyx[2] - box_size_zyx[2]//2)
                
                z_end = min(input_array.shape[0], z_start + box_size_zyx[0])
                y_end = min(input_array.shape[1], y_start + box_size_zyx[1])
                x_end = min(input_array.shape[2], x_start + box_size_zyx[2])
                
                expected_data = input_array[z_start:z_end, y_start:y_end, x_start:x_end]
                np.testing.assert_array_almost_equal(output_array[:], expected_data,
                                                   err_msg="Extracted data doesn't match original")
    
    def test_extraction_with_offset_center(self):
        """Test extraction with a center point near the edge of the volume"""
        # Define center near the edge and box size
        center_zyx = (20, 20, 20)
        box_size_zyx = (40, 40, 40)
        
        # Extract the bounding box
        extract_bounding_box(
            self.input_zarr_path,
            self.output_zarr_path,
            center_zyx,
            box_size_zyx
        )
        
        # Verify extraction results
        output_array = zarr.open(self.output_zarr_path / "0", mode='r')
        
        # Since the box is partially out of bounds, dimensions should be adjusted
        self.assertLessEqual(output_array.shape[0], box_size_zyx[0],
                           "Z dimension should be <= requested box size")
        self.assertLessEqual(output_array.shape[1], box_size_zyx[1],
                           "Y dimension should be <= requested box size")
        self.assertLessEqual(output_array.shape[2], box_size_zyx[2],
                           "X dimension should be <= requested box size")
        
        # Verify metadata contains extraction info
        import json
        with open(self.output_zarr_path / "meta.json", 'r') as f:
            metadata = json.load(f)
        
        self.assertIn("extraction_info", metadata,
                     "Metadata should contain extraction_info")
        self.assertEqual(metadata["source"]["center_zyx"], list(center_zyx),
                        "Center coordinates in metadata don't match input")
        
if __name__ == "__main__":
    unittest.main()