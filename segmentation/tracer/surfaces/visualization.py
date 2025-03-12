"""
Visualization utilities for surface data.

This module provides utilities for visualizing 3D surface data, with automatic
conversion between internal ZYX coordinate ordering and XYZ ordering for visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, Tuple


def convert_zyx_to_xyz(points: np.ndarray) -> np.ndarray:
    """
    Convert points from ZYX order to XYZ order for visualization.
    
    Args:
        points: Points in ZYX order [z, y, x]
        
    Returns:
        Points in XYZ order [x, y, z]
    """
    if points.shape[-1] != 3:
        raise ValueError(f"Expected points with shape [..., 3], got {points.shape}")
    
    # Create output array with same shape
    xyz_points = np.zeros_like(points)
    
    # Reorder columns: [z, y, x] -> [x, y, z]
    xyz_points[..., 0] = points[..., 2]  # x = original x
    xyz_points[..., 1] = points[..., 1]  # y = original y
    xyz_points[..., 2] = points[..., 0]  # z = original z
    
    return xyz_points


def visualize_surface_points(points: np.ndarray, ax: Optional[plt.Axes] = None, 
                            color: str = 'b', show_axes_labels: bool = True,
                            title: Optional[str] = None) -> plt.Axes:
    """
    Visualize 3D surface points, automatically converting from ZYX to XYZ.
    
    Args:
        points: Points in ZYX order [z, y, x], shape can be (n, 3) or (h, w, 3)
        ax: Optional 3D axes to plot on
        color: Color for the points
        show_axes_labels: Whether to show X, Y, Z axis labels
        title: Optional title for the plot
        
    Returns:
        The matplotlib Axes used for plotting
    """
    # Ensure we're working with a 2D array of 3D points
    original_shape = points.shape
    if len(original_shape) > 2:
        points_2d = points.reshape(-1, 3)
    else:
        points_2d = points
    
    # Convert to XYZ ordering for visualization
    points_xyz = convert_zyx_to_xyz(points_2d)
    
    # Create axis if not provided
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    # Plot the points
    ax.scatter(points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2], c=color)
    
    # Add labels for clarity
    if show_axes_labels:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    
    if title:
        ax.set_title(title)
    
    # Add note about coordinate conversion
    ax.text2D(0.05, 0.95, "Note: Coordinates converted from ZYX to XYZ for visualization", 
              transform=ax.transAxes, fontsize=9)
    
    return ax


def visualize_surface_with_normals(points: np.ndarray, normals: np.ndarray,
                                  scale: float = 1.0, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Visualize 3D surface points with normal vectors, automatically converting from ZYX to XYZ.
    
    Args:
        points: Points in ZYX order [z, y, x], shape (n, 3)
        normals: Normal vectors in ZYX order [dz, dy, dx], shape (n, 3)
        scale: Scale factor for the normal vectors
        ax: Optional 3D axes to plot on
        
    Returns:
        The matplotlib Axes used for plotting
    """
    # Ensure we're working with proper shaped arrays
    if points.shape != normals.shape:
        raise ValueError(f"Points and normals must have the same shape. Got {points.shape} and {normals.shape}")
    
    # Convert to XYZ ordering for visualization
    points_xyz = convert_zyx_to_xyz(points)
    normals_xyz = convert_zyx_to_xyz(normals)
    
    # Create axis if not provided
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    # Plot the points
    ax.scatter(points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2], c='b')
    
    # Plot the normal vectors
    ax.quiver(points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2],
             normals_xyz[:, 0], normals_xyz[:, 1], normals_xyz[:, 2],
             length=scale, color='r')
    
    # Add labels for clarity
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Add note about coordinate conversion
    ax.text2D(0.05, 0.95, "Note: Coordinates converted from ZYX to XYZ for visualization", 
              transform=ax.transAxes, fontsize=9)
    
    return ax