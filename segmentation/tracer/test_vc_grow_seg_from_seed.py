#!/usr/bin/env python3
"""
Test script for the improved vc_grow_seg_from_seed implementation.

This script tests the surface growing algorithm on a synthetic dataset 
(a hollow cylinder) and compares the performance of the new FringeExpander
with the legacy GrowthPriority approach.
"""

import os
import time
import argparse
import logging
import numpy as np
import zarr
import matplotlib.pyplot as plt
from pathlib import Path

# Debug control function
def set_debug_flags(general=True, optimizer=True, fringe=True, gradient=False):
    """Set environment variables for controlling debug output"""
    os.environ['DEBUG'] = '1' if general else '0'
    os.environ['OPTIMIZER_DEBUG'] = '1' if optimizer else '0'
    os.environ['FRINGE_DEBUG'] = '1' if fringe else '0'
    os.environ['GRADIENT_DEBUG'] = '1' if gradient else '0'
    if 'logger' in globals():
        logger.info(f"Debug flags set: general={general}, optimizer={optimizer}, fringe={fringe}, gradient={gradient}")

from tracer.grow import grow_surface_from_seed

# Configure logging to write to both console and file
log_file = "test_growth_log.txt"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("test_grow_seg")

# Enable all debug output by default
set_debug_flags(general=True, optimizer=True, fringe=True, gradient=False)

# Set up output directories
output_dir = Path("test_output")
vis_dir = output_dir / "visualizations"
os.makedirs(vis_dir, exist_ok=True)

def create_synthetic_volume():
    """Create a synthetic 3D volume for testing (hollow cylinder)."""
    # Check if the test volume already exists
    volume_path = Path("test_volumes/hollow_cylinder")
    
    # Force re-creation of the volume for testing
    if volume_path.exists():
        logger.info(f"Using existing test volume at {volume_path}")
        return str(volume_path)
    
    # Create directory
    os.makedirs(volume_path, exist_ok=True)
    
    # Create a synthetic hollow cylinder
    logger.info("Creating synthetic hollow cylinder volume...")
    z_size, y_size, x_size = 64, 64, 64
    data = np.zeros((z_size, y_size, x_size), dtype=np.float32)
    
    # Define center and radius
    center_z, center_y, center_x = z_size // 2, y_size // 2, x_size // 2
    
    # Parameters for the hollow cylinder
    outer_radius = 20  # Larger radius
    inner_radius = 16  # Thicker wall
    height = z_size - 10
    
    # Create indices for the whole volume
    z, y, x = np.ogrid[:z_size, :y_size, :x_size]
    
    # Calculate distance from center axis for each voxel
    dist_from_axis = np.sqrt((y - center_y)**2 + (x - center_x)**2)
    
    # Calculate distance from center point for each voxel (for a sphere)
    dist_from_center = np.sqrt((z - center_z)**2 + (y - center_y)**2 + (x - center_x)**2)
    
    # Create a hollow cylinder
    cylinder_mask = (
        (dist_from_axis <= outer_radius) & 
        (dist_from_axis >= inner_radius) & 
        (z >= (z_size - height) // 2) & 
        (z < (z_size + height) // 2)
    )
    data[cylinder_mask] = 1.0
    
    # Create a sphere at one end (z=10)
    end_center_z = 10
    sphere_radius = 10
    end_sphere_dist = np.sqrt((z - end_center_z)**2 + (y - center_y)**2 + (x - center_x)**2)
    end_sphere_mask = (end_sphere_dist <= sphere_radius)
    data[end_sphere_mask] = 1.0
    
    # Create a binary image (thresholded distance transform)
    # This creates a distance field that's 0 at the object boundaries and increases away from them
    from scipy import ndimage
    
    # First create binary volume (0s and 1s)
    binary_data = (data > 0.5).astype(np.int8)
    
    # Compute distance transform
    # distance to nearest zero (object boundary)
    dist_transform = ndimage.distance_transform_edt(binary_data)
    
    # Normalize to 0-1 range for easier testing
    max_dist = np.max(dist_transform)
    data = 1.0 - dist_transform / max_dist  # Invert so object boundary is 1.0
    
    # Save as zarr
    logger.info(f"Saving synthetic volume to {volume_path}")
    cylinder = zarr.open(str(volume_path), mode='w', shape=data.shape, dtype=np.float32)
    cylinder[:] = data
    
    # Save as numpy array for visualization
    np.save("test_volumes/hollow_cylinder.npy", data)
    
    return str(volume_path)

def visualize_result(surface, volume_data, title, filename):
    """Create a visualization of the surface and save it."""
    from mpl_toolkits.mplot3d import Axes3D
    
    # Extract surface points from the grid
    # Use the internal _points attribute which is a grid of 3D points
    points = surface._points
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Only plot valid points (where z coordinate is not -1, indicating an invalid point)
    valid_mask = points[:, :, 0] != -1
    valid_points = points[valid_mask]
    
    if len(valid_points) > 0:
        # Surface points are in ZYX order [z, y, x]
        # Plot as (x, y, z) for the 3D plot
        ax.scatter(valid_points[:, 2], valid_points[:, 1], valid_points[:, 0], 
                c='b', marker='.', alpha=0.5, label='Surface Points')
    else:
        logger.warning("No valid surface points to plot")
    
    # Show cylinder volume (just plot the center points of voxels where volume is 1)
    if volume_data is not None:
        z, y, x = np.where(volume_data > 0.5)
        
        # Randomly select a subset of volume points for visualization (for efficiency)
        # Display only a limited number of volume points to avoid clutter
        if len(z) > 0:
            indices = np.random.choice(len(z), min(1000, len(z)), replace=False)
            ax.scatter(x[indices], y[indices], z[indices], c='r', marker='.', alpha=0.2, label='Volume')
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Set title
    ax.set_title(title)
    
    # Add legend
    ax.legend()
    
    # Save the figure
    fig.tight_layout()
    plt.savefig(os.path.join(vis_dir, filename))
    logger.info(f"Saved visualization to {vis_dir}/{filename}")
    plt.close(fig)

def test_fringe_expander(volume_path, show_vis=True):
    """
    Test the FringeExpander implementation on the given volume.
    
    Args:
        volume_path: Path to the test volume
        show_vis: Whether to show visualizations interactively
    """
    logger.info("Testing FringeExpander implementation...")
    
    # Load volume data for visualization
    try:
        volume_data = np.load("test_volumes/hollow_cylinder.npy")
    except:
        volume_data = None
    
    # Seed point in the cylinder wall
    center_point = np.array([32, 32, 32], dtype=np.float32)
    
    # Run the FringeExpander
    logger.info("Starting surface growth with FringeExpander...")
    start_time = time.time()
    
    surface = grow_surface_from_seed(
        volume_path=volume_path,
        output_path=str(output_dir),
        seed_point=center_point,
        generations=30,
        step_size=2.0,
        intensity_threshold=0.5,
        batch_size=16,
        distance_threshold=1.5,
        physical_fail_threshold=5.0,  # Increase threshold to be more forgiving in tests
        max_reference_count=8,
        initial_reference_min=2,  # Lower initial reference minimum to make growth easier
        name_prefix="fringe_"
    )
    
    total_time = time.time() - start_time
    total_points = surface.meta.get('valid_points', 0)
    
    logger.info(f"Surface growth completed: {total_points} points in {total_time:.2f} seconds")
    
    # Create visualization
    visualize_result(surface, volume_data, 
                    f"Surface: {total_points} points in {total_time:.2f}s",
                    "surface_growth.png")
    
    # Plot statistics
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Points count
    ax.bar(['Surface Points'], [total_points], color='blue', alpha=0.7)
    ax.set_ylabel('Number of Surface Points')
    
    # Processing time
    ax2 = ax.twinx()
    ax2.bar(['Processing Time'], [total_time], color='red', alpha=0.5)
    ax2.set_ylabel('Time (seconds)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    ax.set_title('Surface Growth Performance')
    fig.tight_layout()
    
    # Save stats
    plt.savefig(os.path.join(vis_dir, "surface_stats.png"))
    logger.info(f"Saved performance visualization to {vis_dir}/surface_stats.png")
    
    if show_vis:
        plt.show()
    else:
        plt.close(fig)
    
    # Return statistics
    return {
        'points': total_points,
        'time': total_time,
    }

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test vc_grow_seg_from_seed implementation")
    parser.add_argument('--volume', help='Path to test volume (default: create synthetic)')
    parser.add_argument('--show', action='store_true', help='Show visualizations')
    
    # Debug options
    debug_group = parser.add_argument_group('debugging')
    debug_group.add_argument("--debug", action="store_true", help="Enable general debugging")
    debug_group.add_argument("--optimizer-debug", action="store_true", help="Enable optimizer debugging")
    debug_group.add_argument("--fringe-debug", action="store_true", help="Enable fringe debugging")
    debug_group.add_argument("--gradient-debug", action="store_true", help="Enable gradient debugging")
    debug_group.add_argument("--all-debug", action="store_true", help="Enable all debugging")
    debug_group.add_argument("--no-debug", action="store_true", help="Disable all debugging")
    
    args = parser.parse_args()
    
    # Configure debug settings from command line
    if args.no_debug:
        set_debug_flags(False, False, False, False)
    elif args.all_debug:
        set_debug_flags(True, True, True, True)
    else:
        # Individual debug flags
        general_debug = args.debug
        optimizer_debug = args.optimizer_debug
        fringe_debug = args.fringe_debug
        gradient_debug = args.gradient_debug
        
        # If any individual flags are set, use them
        if any([general_debug, optimizer_debug, fringe_debug, gradient_debug]):
            set_debug_flags(general_debug, optimizer_debug, fringe_debug, gradient_debug)
    
    # Create or use test volume
    volume_path = args.volume if args.volume else create_synthetic_volume()
    
    # Test the FringeExpander
    results = test_fringe_expander(volume_path, args.show)
    
    # Print summary
    print("\n=== Performance Summary ===")
    print(f"Surface Points: {results['points']}")
    print(f"Processing Time: {results['time']:.2f}s")
    print(f"Speed: {results['points']/results['time']:.2f} points/s")

if __name__ == "__main__":
    main()