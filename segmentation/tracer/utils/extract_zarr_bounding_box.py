#!/usr/bin/env python
"""
Extract a bounding box from an OME-ZARR volume and save it as a new OME-ZARR
with the same resolution structure.

This script takes an input OME-ZARR volume, a center point in ZYX coordinates,
and extracts a bounding box of specified size around that point.
The extracted volume is saved as a new OME-ZARR maintaining the original
resolution pyramid structure.
"""

import argparse
import json
import shutil
import zarr
import numpy as np
from pathlib import Path
import os
import logging

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_bounding_box(input_zarr_path, output_zarr_path, center_zyx, box_size_zyx):
    """
    Extract a bounding box from an OME-ZARR volume and save it as a new OME-ZARR.
    
    Parameters:
        input_zarr_path (str): Path to the input OME-ZARR
        output_zarr_path (str): Path where the output OME-ZARR will be saved
        center_zyx (tuple): ZYX coordinates of the center point
        box_size_zyx (tuple): Size of the bounding box in ZYX order
    """
    input_path = Path(input_zarr_path)
    output_path = Path(output_zarr_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Load metadata from the input ZARR
    try:
        with open(input_path / "meta.json", 'r') as f:
            metadata = json.load(f)
            voxelsize = metadata.get("voxelsize", [1.0, 1.0, 1.0])
            logger.info(f"Loaded metadata with voxelsize: {voxelsize}")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Could not load metadata: {e}. Using default voxelsize [1.0, 1.0, 1.0]")
        metadata = {}
        voxelsize = [1.0, 1.0, 1.0]
    
    # Calculate the bounding box in each resolution level
    center_z, center_y, center_x = center_zyx
    box_size_z, box_size_y, box_size_x = box_size_zyx
    
    # Find all resolution levels
    resolution_levels = sorted([d for d in os.listdir(input_path) 
                               if d.isdigit() and os.path.isdir(input_path / d)])
    
    if not resolution_levels:
        logger.error(f"No resolution levels found in {input_path}")
        return
    
    logger.info(f"Found resolution levels: {resolution_levels}")
    
    # Process each resolution level
    for level in resolution_levels:
        level_path = input_path / level
        level_int = int(level)
        
        # Scale factors for this resolution level (assuming 2x downsampling per level)
        scale = 2 ** level_int
        
        # Calculate scaled center and box size
        scaled_center = (center_z // scale, center_y // scale, center_x // scale)
        scaled_box_size = (box_size_z // scale, box_size_y // scale, box_size_x // scale)
        
        # Calculate bounds
        half_z, half_y, half_x = [s // 2 for s in scaled_box_size]
        min_bounds = [
            max(0, sc - half) for sc, half in zip(scaled_center, [half_z, half_y, half_x])
        ]
        
        # Open the zarr array for this resolution level
        try:
            zarr_array = zarr.open(level_path, mode='r')
            
            # Get array shape
            array_shape = zarr_array.shape
            
            # Calculate max bounds (ensuring we don't go beyond array dimensions)
            max_bounds = [
                min(array_shape[i], min_bounds[i] + scaled_box_size[i])
                for i in range(3)
            ]
            
            # Recalculate actual box size (in case it was clipped)
            actual_box_size = [max_bounds[i] - min_bounds[i] for i in range(3)]
            
            logger.info(f"Level {level}: Extracting from {min_bounds} to {max_bounds}")
            
            # Extract the bounding box
            min_z, min_y, min_x = min_bounds
            max_z, max_y, max_x = max_bounds
            bbox_data = zarr_array[min_z:max_z, min_y:max_y, min_x:max_x]
            
            # Create output directory for this resolution level
            output_level_path = output_path / level
            os.makedirs(output_level_path, exist_ok=True)
            
            # Create a new zarr array with the same dtype and chunks as the original
            output_array = zarr.create(
                shape=bbox_data.shape,
                chunks=zarr_array.chunks,
                dtype=zarr_array.dtype,
                store=output_level_path,
                overwrite=True
            )
            
            # Copy array attributes
            for key, value in zarr_array.attrs.items():
                output_array.attrs[key] = value
            
            # Write the data
            output_array[:] = bbox_data
            
            logger.info(f"Level {level}: Saved array with shape {bbox_data.shape}")
            
        except Exception as e:
            logger.error(f"Error processing resolution level {level}: {e}")
    
    # Create updated metadata
    updated_metadata = metadata.copy()
    updated_metadata.update({
        "source": {
            "original_path": str(input_path),
            "center_zyx": center_zyx,
            "box_size_zyx": box_size_zyx
        },
        "extraction_info": {
            "min_bounds_zyx": min_bounds,
            "max_bounds_zyx": max_bounds,
            "actual_size_zyx": actual_box_size
        }
    })
    
    # Save metadata
    with open(output_path / "meta.json", 'w') as f:
        json.dump(updated_metadata, f, indent=2)
    
    logger.info(f"Extracted bounding box saved to {output_path}")
    return output_path

def main():
    """Parse command line arguments and run the extraction"""
    parser = argparse.ArgumentParser(description="Extract a bounding box from an OME-ZARR volume")
    parser.add_argument("input_zarr", help="Path to the input OME-ZARR")
    parser.add_argument("output_zarr", help="Path where the output OME-ZARR will be saved")
    parser.add_argument("--center", type=int, nargs=3, required=True,
                       help="ZYX coordinates of the center point")
    parser.add_argument("--size", type=int, nargs=3, required=True,
                       help="Size of the bounding box in ZYX order")
    parser.add_argument("--test-dir", action="store_true",
                       help="If set, automatically save to the tests/volumes directory")
    
    args = parser.parse_args()
    
    center_zyx = tuple(args.center)
    box_size_zyx = tuple(args.size)
    
    # If test-dir flag is set, save to the tests/volumes directory
    if args.test_dir:
        test_volumes_dir = Path(__file__).parent / "tests" / "volumes"
        os.makedirs(test_volumes_dir, exist_ok=True)
        
        # Create a name based on the input and center coordinates
        input_name = Path(args.input_zarr).stem
        output_name = f"{input_name}_box_z{center_zyx[0]}_y{center_zyx[1]}_x{center_zyx[2]}.zarr"
        output_zarr = test_volumes_dir / output_name
    else:
        output_zarr = args.output_zarr
    
    extract_bounding_box(args.input_zarr, output_zarr, center_zyx, box_size_zyx)

if __name__ == "__main__":
    main()