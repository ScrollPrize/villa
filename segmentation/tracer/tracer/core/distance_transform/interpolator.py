"""
Distance transform interpolator for efficient on-the-fly computation and caching.

This module provides the DistanceTransformInterpolator class, which computes and
caches distance transform chunks on demand, similar to the C++ implementation.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, Union, Any
from collections import OrderedDict

from tracer.core.distance_transform.transform import thresholded_distance, create_distance_field


class DistanceTransformInterpolator:
    """
    Interpolator for distance transform data with on-the-fly computation and caching.
    
    This class mimics the CachedChunked3dInterpolator<uint8_t, thresholdedDistance> from the C++ code.
    It computes the distance transform for chunks on demand and caches the results.
    
    Attributes:
        volume: Raw volume data
        threshold: Intensity threshold for object/background classification (default: 170.0)
        chunk_size: Size of each chunk (default: 64)
        border: Border size around chunks (default: 16)
        transformer: Distance transform processor
        cache: LRU cache for storing processed chunks
    """
    
    def __init__(
        self,
        volume: Union[np.ndarray, torch.Tensor, Any],
        threshold: float = 170.0,
        chunk_size: int = 64,
        border: int = 16,
        cache_size: int = 100
    ):
        """
        Initialize the distance transform interpolator.
        
        Args:
            volume: Raw volume data to process
            threshold: Intensity threshold for object/background classification
            chunk_size: Size of chunks for processing
            border: Border size around chunks
            cache_size: Maximum number of chunks to cache in memory
        """
        self.volume = volume
        self.threshold = threshold
        self.chunk_size = chunk_size
        self.border = border
        self.cache_size = cache_size
        
        # Create distance transform processor
        self.transformer = thresholded_distance(threshold=threshold, border=border, chunk_size=chunk_size)
        
        # Set up LRU cache for processed chunks
        self.cache = OrderedDict()
        
        # Get volume shape
        if hasattr(volume, 'shape'):
            self.shape = volume.shape
        else:
            # For unknown volume types, try to get shape
            try:
                self.shape = (volume.shape[0], volume.shape[1], volume.shape[2])
            except Exception as e:
                raise ValueError(f"Could not determine shape of volume: {e}")
        
        # Stats
        self.chunks_processed = 0
        
        print(f"Created DistanceTransformInterpolator for volume shape {self.shape}")
        print(f"Using threshold={threshold}, chunk_size={chunk_size}, border={border}")
    
    def sample(self, z: float, y: float, x: float) -> float:
        """
        Sample the distance transform at a 3D point using trilinear interpolation.
        
        Args:
            z, y, x: Coordinates to sample (in ZYX order)
            
        Returns:
            Distance value at the point (0.0 for object points, >0 for background)
        """
        # Check bounds
        if (z < 0 or z >= self.shape[0] - 1 or
            y < 0 or y >= self.shape[1] - 1 or
            x < 0 or x >= self.shape[2] - 1):
            return float('inf')  # Out of bounds
        
        # Get chunk indices
        z_idx = int(z // self.chunk_size)
        y_idx = int(y // self.chunk_size)
        x_idx = int(x // self.chunk_size)
        chunk_key = f"{z_idx}_{y_idx}_{x_idx}"
        
        # Check if chunk is already processed
        if chunk_key in self.cache:
            # Move to end (most recently used)
            chunk_data, offsets = self.cache.pop(chunk_key)
            self.cache[chunk_key] = (chunk_data, offsets)
        else:
            # Process chunk
            chunk_data, offsets = self._process_chunk(z_idx, y_idx, x_idx)
            
            # Store in cache
            self.cache[chunk_key] = (chunk_data, offsets)
            
            # Enforce cache size limit
            if len(self.cache) > self.cache_size:
                # Remove oldest (first) item
                self.cache.popitem(last=False)
        
        # Get local coordinates within chunk
        local_z = z - offsets[0]
        local_y = y - offsets[1]
        local_x = x - offsets[2]
        
        # Check if the point is within the chunk bounds (including safety check)
        if (local_z < 0 or local_z >= chunk_data.shape[0] - 1 or
            local_y < 0 or local_y >= chunk_data.shape[1] - 1 or
            local_x < 0 or local_x >= chunk_data.shape[2] - 1):
            # Instead of failing, try to get a nearby chunk that contains this point
            # Calculate which direction we should move in
            new_z_idx, new_y_idx, new_x_idx = z_idx, y_idx, x_idx
            
            if local_z < 0:
                new_z_idx = max(0, z_idx - 1)
            elif local_z >= chunk_data.shape[0] - 1:
                new_z_idx = min(self.shape[0] // self.chunk_size - 1, z_idx + 1)
                
            if local_y < 0:
                new_y_idx = max(0, y_idx - 1)
            elif local_y >= chunk_data.shape[1] - 1:
                new_y_idx = min(self.shape[1] // self.chunk_size - 1, y_idx + 1)
                
            if local_x < 0:
                new_x_idx = max(0, x_idx - 1)
            elif local_x >= chunk_data.shape[2] - 1:
                new_x_idx = min(self.shape[2] // self.chunk_size - 1, x_idx + 1)
            
            # If we found a different chunk to try
            if (new_z_idx != z_idx or new_y_idx != y_idx or new_x_idx != x_idx):
                try:
                    # Try sampling from the adjacent chunk
                    new_chunk_key = f"{new_z_idx}_{new_y_idx}_{new_x_idx}"
                    if new_chunk_key in self.cache:
                        new_chunk_data, new_offsets = self.cache[new_chunk_key]
                    else:
                        # Process the new chunk
                        new_chunk_data, new_offsets = self._process_chunk(new_z_idx, new_y_idx, new_x_idx)
                        self.cache[new_chunk_key] = (new_chunk_data, new_offsets)
                    
                    # Get new local coordinates
                    new_local_z = z - new_offsets[0]
                    new_local_y = y - new_offsets[1]
                    new_local_x = x - new_offsets[2]
                    
                    # Validate new coordinates
                    if (new_local_z >= 0 and new_local_z < new_chunk_data.shape[0] - 1 and
                        new_local_y >= 0 and new_local_y < new_chunk_data.shape[1] - 1 and
                        new_local_x >= 0 and new_local_x < new_chunk_data.shape[2] - 1):
                        
                        # Swap to the new chunk
                        chunk_data = new_chunk_data
                        local_z, local_y, local_x = new_local_z, new_local_y, new_local_x
                    else:
                        # Still out of bounds with new chunk
                        return float('inf')
                except Exception as e:
                    # Failed to get adjacent chunk
                    return float('inf')
            else:
                # No adjacent chunk to try
                return float('inf')
        
        # Use PyTorch's grid_sample for efficient trilinear interpolation
        try:
            # Make sure chunk data is a tensor
            if not isinstance(chunk_data, torch.Tensor):
                chunk_data = torch.tensor(chunk_data, dtype=torch.float32)
            
            # Add batch and channel dimensions if needed
            if len(chunk_data.shape) == 3:  # [z, y, x]
                chunk_data = chunk_data.unsqueeze(0).unsqueeze(0)  # [1, 1, z, y, x]
            
            # Print tensor shape for debugging
            print(f"DEBUG: chunk_data shape after reshaping: {chunk_data.shape}")
                
            # Convert local coordinates to the range [-1, 1] required by grid_sample
            # grid_sample expects coordinates in [z, y, x] order and [-1, 1] range
            # Use indices from the actual shape, not hardcoded
            if len(chunk_data.shape) != 5:
                raise ValueError(f"Expected 5D tensor, got shape {chunk_data.shape}")
                
            norm_z = 2.0 * (local_z / (chunk_data.shape[2] - 1)) - 1.0
            norm_y = 2.0 * (local_y / (chunk_data.shape[3] - 1)) - 1.0
            norm_x = 2.0 * (local_x / (chunk_data.shape[4] - 1)) - 1.0
            
            # Create the sampling grid in the correct shape for 5D input
            # For a 5D volume with shape [N, C, D, H, W], grid must be [N, D', H', W', 3]
            # For a single point sampling, we use [1, 1, 1, 1, 3]
            grid = torch.tensor([[[[[norm_z, norm_y, norm_x]]]]], dtype=torch.float32, device=chunk_data.device)
            
            # Sample using grid_sample with proper dimensions
            sampled = torch.nn.functional.grid_sample(
                chunk_data,        # [1, 1, D, H, W]
                grid,              # [1, 1, 1, 1, 3]
                mode='bilinear',   # bilinear is trilinear in 3D
                align_corners=True
            )
            
            # Extract the result (removes the extra dimensions)
            result = sampled.item()
            
        except Exception as e:
            print(f"ERROR using grid_sample: {e}")
            print(f"Falling back to nearest-neighbor sampling")
            
            # Fallback to nearest-neighbor sampling if grid_sample fails
            z0, y0, x0 = int(local_z), int(local_y), int(local_x)
            
            # Ensure indices are within bounds
            z0 = max(0, min(z0, chunk_data.shape[0] - 1))
            y0 = max(0, min(y0, chunk_data.shape[1] - 1)) 
            x0 = max(0, min(x0, chunk_data.shape[2] - 1))
            
            try:
                result = float(chunk_data[z0, y0, x0].item()) if torch.is_tensor(chunk_data[z0, y0, x0]) else float(chunk_data[z0, y0, x0])
            except Exception as e2:
                print(f"ERROR even with nearest-neighbor sampling: {e2}")
                return float('inf')
        return result
    
    def _process_chunk(self, z_idx: int, y_idx: int, x_idx: int) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        """
        Process a chunk of the volume with the distance transform.
        
        Args:
            z_idx, y_idx, x_idx: Chunk indices
            
        Returns:
            Tuple of (processed_chunk, offsets)
        """
        # Calculate chunk bounds with border
        z_start = max(0, z_idx * self.chunk_size - self.border)
        y_start = max(0, y_idx * self.chunk_size - self.border)
        x_start = max(0, x_idx * self.chunk_size - self.border)
        
        z_end = min(self.shape[0], (z_idx + 1) * self.chunk_size + self.border)
        y_end = min(self.shape[1], (y_idx + 1) * self.chunk_size + self.border)
        x_end = min(self.shape[2], (x_idx + 1) * self.chunk_size + self.border)
        
        # Extract chunk data
        if hasattr(self.volume, 'chunks'):  # Zarr array
            chunk_data = self.volume[z_start:z_end, y_start:y_end, x_start:x_end]
            if not isinstance(chunk_data, torch.Tensor):
                chunk_data = torch.from_numpy(np.array(chunk_data, dtype=np.float32))
        elif isinstance(self.volume, np.ndarray):
            chunk_data = torch.from_numpy(self.volume[z_start:z_end, y_start:y_end, x_start:x_end].astype(np.float32))
        elif isinstance(self.volume, torch.Tensor):
            chunk_data = self.volume[z_start:z_end, y_start:y_end, x_start:x_end].clone()
        else:
            raise ValueError(f"Unsupported volume type: {type(self.volume)}")
        
        # Process with distance transform
        try:
            processed = self.transformer.compute(chunk_data)
            # Increment counter
            self.chunks_processed += 1
        except Exception as e:
            print(f"ERROR in distance transform computation: {e}")
            # Fallback to direct threshold + distance field
            processed = create_distance_field(chunk_data, self.threshold)
            self.chunks_processed += 1
        
        # Return processed chunk and offsets
        return processed, (z_start, y_start, x_start)
    
    def get_stats(self) -> Dict:
        """
        Get interpolator statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "threshold": self.threshold,
            "chunk_size": self.chunk_size,
            "border": self.border,
            "cache_size": self.cache_size,
            "current_cache_usage": len(self.cache),
            "chunks_processed": self.chunks_processed
        }