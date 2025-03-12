#!/usr/bin/env python
"""
Script to convert between tifxyz format and 3D TIFs.

This utility allows:
1. Converting a tifxyz directory (with x.tif, y.tif, z.tif) to a 3D TIF volume
2. Converting a test volume to a 3D TIF for visualization
3. Creating a combined visualization showing both the test volume and surface

Usage:
  python convert_tifxyz_to_3dtif.py --tifxyz_dir /path/to/tifxyz --output output.tif
  python convert_tifxyz_to_3dtif.py --test_volume test_volumes/hollow_cylinder.npy --output cylinder.tif
  python convert_tifxyz_to_3dtif.py --tifxyz_dir /path/to/tifxyz --test_volume test_volumes/hollow_cylinder.npy --output combined.tif
"""

import os
import argparse
import numpy as np
import tifffile
from pathlib import Path
from surfaces.quad_surface import load_quad_from_tifxyz

def convert_tifxyz_to_volume(tifxyz_dir, output_size=(64, 64, 64), padding=2):
    """
    Convert a tifxyz directory (surface representation) to a 3D volume.
    
    Args:
        tifxyz_dir: Path to the tifxyz directory
        output_size: Size of the output volume (z, y, x)
        padding: Number of voxels to add around surface points
        
    Returns:
        3D numpy array representing the surface as a volume
    """
    # Load the quad surface from tifxyz
    quad_surface = load_quad_from_tifxyz(tifxyz_dir)
    
    # Get the raw points (in ZYX order)
    points = quad_surface.raw_points()
    
    # Calculate bounding box
    bbox = quad_surface.bbox()
    
    # Determine volume dimensions
    if output_size is None:
        # Calculate dimensions from bounding box
        min_z, min_y, min_x = bbox.low
        max_z, max_y, max_x = bbox.high
        
        # Add padding
        min_z -= padding
        min_y -= padding
        min_x -= padding
        max_z += padding
        max_y += padding
        max_x += padding
        
        # Determine volume size
        z_size = int(max_z - min_z) + 1
        y_size = int(max_y - min_y) + 1
        x_size = int(max_x - min_x) + 1
        
        volume_shape = (z_size, y_size, x_size)
    else:
        volume_shape = output_size
    
    # Create empty volume
    volume = np.zeros(volume_shape, dtype=np.uint8)
    
    # Get valid points from the surface
    valid_mask = points[:, :, 0] != -1
    valid_points = points[valid_mask]
    
    # Normalize point coordinates to fit within the volume
    min_coords = np.min(valid_points, axis=0)
    max_coords = np.max(valid_points, axis=0)
    
    # Calculate scaling factors to fit points into volume
    z_scale = (volume_shape[0] - 2*padding) / max(1, (max_coords[0] - min_coords[0]))
    y_scale = (volume_shape[1] - 2*padding) / max(1, (max_coords[1] - min_coords[1]))
    x_scale = (volume_shape[2] - 2*padding) / max(1, (max_coords[2] - min_coords[2]))
    
    scale = min(z_scale, y_scale, x_scale)
    
    # Create volume from surface points
    for pt in valid_points:
        # Scale and offset points to fit in volume
        z = int((pt[0] - min_coords[0]) * scale) + padding
        y = int((pt[1] - min_coords[1]) * scale) + padding
        x = int((pt[2] - min_coords[2]) * scale) + padding
        
        # Ensure within bounds
        if 0 <= z < volume_shape[0] and 0 <= y < volume_shape[1] and 0 <= x < volume_shape[2]:
            # Set voxel and neighbors (to create a thicker surface)
            for dz in range(-padding, padding+1):
                for dy in range(-padding, padding+1):
                    for dx in range(-padding, padding+1):
                        zz, yy, xx = z+dz, y+dy, x+dx
                        if (0 <= zz < volume_shape[0] and 
                            0 <= yy < volume_shape[1] and 
                            0 <= xx < volume_shape[2]):
                            volume[zz, yy, xx] = 255
    
    return volume

def load_test_volume(volume_path):
    """
    Load a test volume from a numpy file.
    
    Args:
        volume_path: Path to the test volume file
        
    Returns:
        3D numpy array representing the test volume
    """
    # Load the volume from the numpy file
    volume = np.load(volume_path)
    
    # Ensure proper data type and range for visualization
    if volume.dtype != np.uint8:
        # Normalize to 0-255 range
        volume = ((volume - volume.min()) * (255.0 / max(1e-6, volume.max() - volume.min()))).astype(np.uint8)
    
    return volume

def create_combined_visualization(tifxyz_dir, test_volume_path, output_size=(128, 128, 128)):
    """
    Create a combined visualization of the test volume and surface.
    
    Args:
        tifxyz_dir: Path to the tifxyz directory
        test_volume_path: Path to the test volume file
        output_size: Size of the output volume (z, y, x)
        
    Returns:
        3D numpy array representing the combined visualization
    """
    # Load the test volume
    test_volume = load_test_volume(test_volume_path)
    
    # Create a volume from the surface
    surface_volume = convert_tifxyz_to_volume(tifxyz_dir, output_size)
    
    # Resize test volume to match surface volume if needed
    if test_volume.shape != surface_volume.shape:
        from scipy.ndimage import zoom
        factors = [
            surface_volume.shape[0] / test_volume.shape[0],
            surface_volume.shape[1] / test_volume.shape[1],
            surface_volume.shape[2] / test_volume.shape[2]
        ]
        test_volume = zoom(test_volume, factors, order=1)
    
    # Create RGB volume for visualization (R=test volume, G=surface, B=0)
    combined = np.zeros((*test_volume.shape, 3), dtype=np.uint8)
    combined[..., 0] = test_volume  # Red channel: test volume
    combined[..., 1] = surface_volume  # Green channel: surface
    
    return combined

def save_tiff_volume(volume, output_path):
    """
    Save a 3D volume as a multi-page TIFF file.
    
    Args:
        volume: 3D numpy array to save
        output_path: Path to save the TIFF file
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Save the volume as a multi-page TIFF
    tifffile.imwrite(output_path, volume)
    print(f"Saved 3D TIFF to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert between tifxyz and 3D TIFs")
    parser.add_argument("--tifxyz_dir", type=str, help="Path to tifxyz directory")
    parser.add_argument("--test_volume", type=str, help="Path to test volume file")
    parser.add_argument("--output", type=str, required=True, help="Output TIFF file path")
    parser.add_argument("--size", type=int, nargs=3, default=[128, 128, 128], 
                        help="Output volume size (z, y, x)")
    parser.add_argument("--padding", type=int, default=2, 
                        help="Padding around surface points")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.tifxyz_dir and not args.test_volume:
        parser.error("Either --tifxyz_dir or --test_volume must be provided")
    
    # Process based on input arguments
    if args.tifxyz_dir and args.test_volume:
        # Create combined visualization
        print(f"Creating combined visualization from {args.tifxyz_dir} and {args.test_volume}")
        volume = create_combined_visualization(args.tifxyz_dir, args.test_volume, tuple(args.size))
        save_tiff_volume(volume, args.output)
    
    elif args.tifxyz_dir:
        # Convert tifxyz to 3D TIF
        print(f"Converting tifxyz from {args.tifxyz_dir} to 3D TIF")
        volume = convert_tifxyz_to_volume(args.tifxyz_dir, tuple(args.size), args.padding)
        save_tiff_volume(volume, args.output)
    
    elif args.test_volume:
        # Convert test volume to 3D TIF
        print(f"Converting test volume from {args.test_volume} to 3D TIF")
        volume = load_test_volume(args.test_volume)
        save_tiff_volume(volume, args.output)

if __name__ == "__main__":
    main()