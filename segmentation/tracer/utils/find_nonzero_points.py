#!/usr/bin/env python
"""
Find non-zero points in a zarr volume for use as test coordinates.
"""

import numpy as np
import zarr
import random
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_nonzero_points(zarr_path, num_points=15, resolution_level="0", threshold=10, min_distance=50):
    """
    Find non-zero points in a zarr volume that are spaced at least min_distance apart.
    
    Parameters:
        zarr_path (str): Path to the zarr volume
        num_points (int): Number of points to find
        resolution_level (str): Resolution level to search in
        threshold (int): Minimum value to consider as non-zero
        min_distance (int): Minimum distance between points
        
    Returns:
        list: List of (z, y, x) coordinates of non-zero points
    """
    zarr_path = Path(zarr_path)
    level_path = zarr_path / resolution_level
    
    # Open the zarr array
    logger.info(f"Opening zarr array at {level_path}")
    zarr_array = zarr.open(level_path, mode='r')
    
    # Get the shape
    shape = zarr_array.shape
    logger.info(f"Array shape: {shape}")
    
    # Strategy: Sample random locations until we find enough non-zero points
    nonzero_points = []
    attempts = 0
    max_attempts = num_points * 500  # Limit the number of attempts
    
    # Block size for more efficient reading
    block_size = min(256, min(shape))
    
    def is_far_enough(point, points_list, min_dist):
        """Check if a point is far enough from all other points."""
        for existing_point in points_list:
            dist = sum((a-b)**2 for a, b in zip(point, existing_point))**0.5
            if dist < min_dist:
                return False
        return True
    
    # Sample from different regions of the volume
    region_size = shape[0] // 4  # Divide the volume into 4x4x4 regions
    regions_sampled = set()
    
    while len(nonzero_points) < num_points and attempts < max_attempts:
        # Choose a region that hasn't been sampled yet, if possible
        if len(regions_sampled) < 64:  # 4x4x4 = 64 regions
            region_z = random.randint(0, 3)
            region_y = random.randint(0, 3)
            region_x = random.randint(0, 3)
            region_key = (region_z, region_y, region_x)
            
            if region_key in regions_sampled and len(regions_sampled) < 64:
                continue
            
            regions_sampled.add(region_key)
            
            # Choose a random point within this region
            z_start = region_z * region_size + random.randint(0, region_size - block_size)
            y_start = region_y * region_size + random.randint(0, region_size - block_size)
            x_start = region_x * region_size + random.randint(0, region_size - block_size)
        else:
            # If all regions have been sampled, just choose random locations
            z_start = random.randint(0, shape[0] - block_size)
            y_start = random.randint(0, shape[1] - block_size)
            x_start = random.randint(0, shape[2] - block_size)
        
        # Read a block
        block = zarr_array[z_start:z_start+block_size, 
                          y_start:y_start+block_size, 
                          x_start:x_start+block_size]
        
        # Find non-zero points in this block
        z_indices, y_indices, x_indices = np.where(block > threshold)
        
        if len(z_indices) > 0:
            # Convert to absolute coordinates
            abs_z = z_indices + z_start
            abs_y = y_indices + y_start
            abs_x = x_indices + x_start
            
            # Randomize the order
            indices = list(range(len(z_indices)))
            random.shuffle(indices)
            
            for i in indices:
                point = (int(abs_z[i]), int(abs_y[i]), int(abs_x[i]))
                
                # Check if this point is far enough from all existing points
                if is_far_enough(point, nonzero_points, min_distance):
                    nonzero_points.append(point)
                    logger.info(f"Added point {point} (total: {len(nonzero_points)})")
                    if len(nonzero_points) >= num_points:
                        break
        
        attempts += 1
    
    logger.info(f"Found {len(nonzero_points)} non-zero points after {attempts} attempts")
    return nonzero_points

if __name__ == "__main__":
    zarr_path = "../tests/volumes/s5_059_region_7300_3030_4555.zarr"
    # Final parameters: 
    # - Lower threshold to find more structure
    # - Large minimum distance to ensure points are distributed
    # - Seed the random generator for reproducible results
    random.seed(42)
    np.random.seed(42)
    points = find_nonzero_points(zarr_path, num_points=15, threshold=10, min_distance=250)
    
    # Print the points in a format that can be used in tests
    print("\nNon-zero points (ZYX coordinates):")
    print("[\n    " + ",\n    ".join([f"({z}, {y}, {x})" for z, y, x in points]) + "\n]")
    
    # Also print in Python list format for easy copying
    print("\nPython list format:")
    print("[")
    for z, y, x in points:
        print(f"    [{z}, {y}, {x}],")
    print("]")