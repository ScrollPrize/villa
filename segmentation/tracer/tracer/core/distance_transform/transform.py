"""
Implementation of distance transform for binary volumes.

This module provides the core functions for computing distance transforms,
based on the C++ implementation described in distance_transform.md.

The distance transform converts a binary volume into a continuous field
representing the distance to the nearest voxel that satisfies a threshold
condition (typically intensity >= 170).
"""

import torch
import numpy as np
from typing import Union, Tuple, Optional


def distance_transform(
    chunk: torch.Tensor, 
    steps: int = 15, 
    magic_value: float = -1.0
) -> torch.Tensor:
    """
    Compute the distance transform of a tensor.
    
    This is a vectorized implementation of the distance transform that iteratively
    propagates distances outward from object boundaries. The input is
    expected to have magic_value at non-object points and 0.0 at object
    boundaries.
    
    Args:
        chunk: Input tensor with values of 0.0 at object points and 
               magic_value at non-object points
        steps: Number of steps to propagate distances (controls max distance)
        magic_value: Value marking non-object points that need distances computed
        
    Returns:
        Tensor with distance values (0.0 at object points, increasing with distance)
    """
    # Create two buffers for iterative computation
    c1 = chunk.clone()
    c2 = torch.empty_like(chunk)
    
    # Create a mask for magic values
    magic_mask = (c1 == magic_value)
    
    # Iteratively propagate distances
    for n in range(steps // 2):
        # First iteration: c1 -> c2
        _dist_iteration_vectorized(c1, c2, magic_value)
        
        # Second iteration: c2 -> c1
        _dist_iteration_vectorized(c2, c1, magic_value)
    
    # Set any remaining magic values to max distance (steps)
    c1[c1 == magic_value] = float(steps)
    
    return c1


def _dist_iteration_vectorized(
    from_tensor: torch.Tensor,
    to_tensor: torch.Tensor,
    magic_value: float
) -> None:
    """
    Perform a single iteration of distance propagation using vectorized operations.
    
    This efficient implementation avoids explicit loops over the volume by using
    tensor operations to propagate distances.
    
    Args:
        from_tensor: Source tensor for this iteration
        to_tensor: Destination tensor for this iteration
        magic_value: Value marking non-object points
    """
    # Copy non-magic values directly - they remain unchanged
    to_tensor[:] = from_tensor[:]
    
    # Create mask for magic values
    magic_mask = (from_tensor == magic_value)
    
    if not magic_mask.any():
        # No magic values to update
        return
    
    # For each of the 6 directions, check neighbors and update distances
    # We'll use padding to handle boundaries efficiently
    padded = torch.nn.functional.pad(
        from_tensor, (1, 1, 1, 1, 1, 1), mode='constant', value=magic_value
    )
    
    # Check all 6 neighbors and find minimum valid distance
    neighbors = []
    
    # z-1 neighbor
    neighbors.append(padded[1:-1, 1:-1, 1:-1])
    
    # z+1 neighbor
    neighbors.append(padded[2:, 1:-1, 1:-1])
    
    # y-1 neighbor
    neighbors.append(padded[1:-1, 0:-2, 1:-1])
    
    # y+1 neighbor
    neighbors.append(padded[1:-1, 2:, 1:-1])
    
    # x-1 neighbor
    neighbors.append(padded[1:-1, 1:-1, 0:-2])
    
    # x+1 neighbor
    neighbors.append(padded[1:-1, 1:-1, 2:])
    
    # Start with all magic values
    min_dist = torch.full_like(from_tensor, magic_value)
    
    # For each neighbor, update minimum distance
    # Ignore magic values in neighbors
    for neighbor in neighbors:
        # Mask where neighbor is not magic
        valid_neighbor = (neighbor != magic_value)
        
        # Where neighbor is valid, take minimum of current min and neighbor+1
        min_dist = torch.where(
            valid_neighbor & (min_dist == magic_value),
            neighbor + 1,
            min_dist
        )
        
        min_dist = torch.where(
            valid_neighbor & (min_dist != magic_value) & (neighbor + 1 < min_dist),
            neighbor + 1,
            min_dist
        )
    
    # Update only magic cells in the output
    to_tensor[magic_mask] = min_dist[magic_mask]


class thresholded_distance:
    """
    Compute thresholded distance transform for binary volumes.
    
    This implements the thresholdedDistance struct from the C++ code,
    which applies the distance transform specifically to binary
    thresholded data.
    
    Attributes:
        BORDER: Border size around chunks
        CHUNK_SIZE: Size of each chunk
        FILL_V: Default fill value
        TH: Intensity threshold for binary segmentation
    """
    
    def __init__(self, threshold: float = 170.0, border: int = 16, chunk_size: int = 64):
        """
        Initialize the thresholded distance transform.
        
        Args:
            threshold: Intensity threshold for considering voxels as objects
                      (voxels >= threshold are objects, others are background)
            border: Border size around chunks
            chunk_size: Size of each chunk
        """
        self.BORDER = border
        self.CHUNK_SIZE = chunk_size
        self.FILL_V = 0
        self.TH = threshold
    
    def compute(self, large: torch.Tensor, offset_large: Optional[Tuple[int, int, int]] = None) -> torch.Tensor:
        """
        Compute thresholded distance transform on a volume chunk.
        
        Args:
            large: Input tensor (can be any shape, will be processed entirely)
            offset_large: Optional offset of the large chunk in the original volume
            
        Returns:
            Tensor with distance transform values, possibly cropped to remove border
        """
        # Get shape and create output tensor
        outer = torch.empty_like(large)
        
        # Get actual size from the tensor 
        shape = large.shape
        magic_value = -1.0
        
        # Create binary mask based on threshold
        # Important: In the C++ version, we use:
        # - Voxels with intensity >= TH are marked with 0.0 (they're inside the object)
        # - Voxels with intensity < TH are marked with magic_value (-1.0)
        mask = large < self.TH
        outer[mask] = magic_value  # Non-object points get magic value
        outer[~mask] = 0.0        # Object points get 0.0
        
        # Apply distance transform
        outer = distance_transform(outer, 15, magic_value)
        
        # If the input size is bigger than chunk_size + 2*border, crop it
        if all(s >= self.CHUNK_SIZE + 2 * self.BORDER for s in shape):
            low = int(self.BORDER)
            high = int(self.BORDER) + int(self.CHUNK_SIZE)
            if len(shape) == 3:
                small = outer[low:high, low:high, low:high]
            else:
                small = outer  # Handle different dimensionality
        else:
            # Otherwise, just return the full transform
            small = outer
        
        return small


def create_distance_field(
    volume: Union[torch.Tensor, np.ndarray],
    threshold: float = 170.0,
    steps: int = 15,
    chunk_size: int = 64,
    border: int = 16
) -> torch.Tensor:
    """
    Create a distance field from a binary volume.
    
    This is a high-level function that computes the distance transform
    for an entire volume without chunking. For large volumes, consider
    using the chunked version with ChunkedTensor.
    
    Args:
        volume: Input volume as torch tensor or numpy array
        threshold: Intensity threshold for object detection
        steps: Number of steps for distance propagation
        chunk_size: Size of chunks (for compatibility, not used)
        border: Border size (for compatibility, not used)
        
    Returns:
        Distance field tensor with same shape as input
    """
    # Convert numpy array to torch tensor if needed
    if isinstance(volume, np.ndarray):
        volume_tensor = torch.from_numpy(volume.astype(np.float32))
    else:
        volume_tensor = volume
    
    # Create binary mask based on threshold
    magic_value = -1.0
    outer = torch.empty_like(volume_tensor)
    mask = volume_tensor < threshold
    outer[mask] = magic_value
    outer[~mask] = 0.0
    
    # Apply distance transform
    return distance_transform(outer, steps, magic_value)