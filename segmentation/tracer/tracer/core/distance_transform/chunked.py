"""
Chunked tensor implementation for efficient distance transform processing.

This module provides a ChunkedTensor class that breaks a large volume into
manageable chunks for processing with limited memory. It also includes
a ChunkCache for efficient storage and retrieval of processed chunks.
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, Union, Callable, Any
from collections import OrderedDict

from tracer.core.distance_transform.transform import thresholded_distance, create_distance_field
from tracer.core.interpolation.trilinear_interpolator_autodiff import TrilinearInterpolatorAutoDiff


class ChunkCache:
    """
    Cache for efficiently storing and retrieving volume chunks.
    
    This implements functionality similar to ChunkCache in the C++ code,
    managing a limited number of chunks in memory and optionally storing
    overflow to disk.
    
    Attributes:
        max_memory_chunks: Maximum number of chunks to keep in memory
        cache_dir: Directory for storing chunks on disk (if enabled)
        use_disk_cache: Whether to use disk caching for overflow
    """
    
    def __init__(
        self,
        max_memory_chunks: int = 100,
        cache_dir: Optional[str] = None,
        use_disk_cache: bool = False
    ):
        """
        Initialize the chunk cache.
        
        Args:
            max_memory_chunks: Maximum number of chunks to keep in memory
            cache_dir: Directory for storing chunks on disk (if enabled)
            use_disk_cache: Whether to use disk caching for overflow
        """
        self.max_memory_chunks = max_memory_chunks
        self.cache_dir = cache_dir
        self.use_disk_cache = use_disk_cache and cache_dir is not None
        
        # Initialize cache structures
        self.memory_cache = OrderedDict()  # LRU cache for chunks in memory
        self.disk_cache_index = set()  # Set of chunk indices stored on disk
        
        # Create cache directory if needed
        if self.use_disk_cache and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
    
    def get_chunk_key(self, indices: Tuple[int, int, int]) -> str:
        """
        Generate a unique string key for chunk indices.
        
        Args:
            indices: Tuple of (z, y, x) chunk indices
            
        Returns:
            String key for the chunk
        """
        return f"chunk_{indices[0]}_{indices[1]}_{indices[2]}"
    
    def get_disk_path(self, key: str) -> str:
        """
        Get the file path for a chunk on disk.
        
        Args:
            key: Chunk key
            
        Returns:
            Path to the chunk file on disk
        """
        return os.path.join(self.cache_dir, f"{key}.pt")
    
    def has_chunk(self, indices: Tuple[int, int, int]) -> bool:
        """
        Check if a chunk is cached (either in memory or on disk).
        
        Args:
            indices: Tuple of (z, y, x) chunk indices
            
        Returns:
            True if the chunk is cached, False otherwise
        """
        key = self.get_chunk_key(indices)
        return key in self.memory_cache or (
            self.use_disk_cache and key in self.disk_cache_index
        )
    
    def get_chunk(self, indices: Tuple[int, int, int]) -> Optional[torch.Tensor]:
        """
        Get a chunk from the cache.
        
        If the chunk is in memory, it will be returned directly.
        If it's on disk, it will be loaded into memory.
        
        Args:
            indices: Tuple of (z, y, x) chunk indices
            
        Returns:
            The cached chunk tensor, or None if not found
        """
        key = self.get_chunk_key(indices)
        
        # Check if chunk is in memory
        if key in self.memory_cache:
            # Move to end of OrderedDict (most recently used)
            chunk = self.memory_cache.pop(key)
            self.memory_cache[key] = chunk
            return chunk
        
        # Check if chunk is on disk
        if self.use_disk_cache and key in self.disk_cache_index:
            # Load from disk
            try:
                chunk_path = self.get_disk_path(key)
                chunk = torch.load(chunk_path)
                
                # Add to memory cache (may evict another chunk)
                self._add_to_memory_cache(key, chunk)
                
                return chunk
            except Exception as e:
                print(f"Error loading chunk from disk: {e}")
                # Remove from disk index if file doesn't exist or is corrupted
                self.disk_cache_index.remove(key)
                return None
        
        # Chunk not found
        return None
    
    def add_chunk(self, indices: Tuple[int, int, int], chunk: torch.Tensor) -> None:
        """
        Add a chunk to the cache.
        
        Args:
            indices: Tuple of (z, y, x) chunk indices
            chunk: Tensor to cache
        """
        key = self.get_chunk_key(indices)
        
        # Add to memory cache (may evict another chunk)
        self._add_to_memory_cache(key, chunk)
    
    def _add_to_memory_cache(self, key: str, chunk: torch.Tensor) -> None:
        """
        Add a chunk to the memory cache, evicting if necessary.
        
        Args:
            key: Chunk key
            chunk: Tensor to cache
        """
        # If memory cache is full, evict the least recently used chunk
        if len(self.memory_cache) >= self.max_memory_chunks:
            # Get oldest key (first item in OrderedDict)
            oldest_key, oldest_chunk = next(iter(self.memory_cache.items()))
            
            # Remove from memory cache
            self.memory_cache.pop(oldest_key)
            
            # Save to disk if disk caching is enabled
            if self.use_disk_cache:
                try:
                    chunk_path = self.get_disk_path(oldest_key)
                    torch.save(oldest_chunk, chunk_path)
                    self.disk_cache_index.add(oldest_key)
                except Exception as e:
                    print(f"Error saving chunk to disk: {e}")
        
        # Add to memory cache
        self.memory_cache[key] = chunk
    
    def clear(self) -> None:
        """Clear the cache (both memory and disk)."""
        self.memory_cache.clear()
        
        # Remove disk cache files if enabled
        if self.use_disk_cache:
            for key in self.disk_cache_index:
                try:
                    chunk_path = self.get_disk_path(key)
                    if os.path.exists(chunk_path):
                        os.remove(chunk_path)
                except Exception as e:
                    print(f"Error removing chunk file: {e}")
            
            self.disk_cache_index.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            "memory_chunks": len(self.memory_cache),
            "disk_chunks": len(self.disk_cache_index) if self.use_disk_cache else 0,
            "max_memory_chunks": self.max_memory_chunks,
            "use_disk_cache": self.use_disk_cache,
            "cache_dir": self.cache_dir if self.use_disk_cache else None
        }


class ChunkedTensor:
    """
    Tensor-like interface for chunked 3D data processing.
    
    This class provides a unified interface to chunked 3D data,
    either processing chunks on-demand or loading them from a cache.
    It's designed for efficient memory usage with large volumes.
    
    Attributes:
        compute: Processing function for chunks (e.g., thresholdedDistance)
        cache: ChunkCache for storing processed chunks
        chunk_size: Size of each chunk
        border: Border size around each chunk
    """
    
    def __init__(
        self,
        compute: Any,  # Function or class with compute method
        dataset: torch.Tensor,
        cache: Optional[ChunkCache] = None,
        cache_root: str = "",
        chunk_size: int = 64,
        border: int = 16,
        precomputed: bool = False
    ):
        """
        Initialize a chunked tensor.
        
        Args:
            compute: Function or class with compute method
            dataset: Source data tensor
            cache: Optional ChunkCache for storing processed chunks
            cache_root: Path for disk caching
            chunk_size: Size of each chunk
            border: Border size around each chunk
            precomputed: If True, dataset is already the processed result and won't be recomputed
        """
        self.compute = compute
        self.dataset = dataset
        self.precomputed = precomputed
        
        # Set chunk parameters
        self.chunk_size = chunk_size if chunk_size else compute.CHUNK_SIZE
        self.border = border if border else compute.BORDER
        
        # Initialize cache if not provided
        if cache is None:
            use_disk = len(cache_root) > 0
            self.cache = ChunkCache(
                max_memory_chunks=100,
                cache_dir=cache_root if use_disk else None,
                use_disk_cache=use_disk
            )
        else:
            self.cache = cache
        
        # Get dataset shape
        if hasattr(dataset, 'shape'):
            if len(dataset.shape) == 3:
                self.depth, self.height, self.width = dataset.shape
            elif len(dataset.shape) == 4:
                # Handle batch dimension
                _, self.depth, self.height, self.width = dataset.shape
            else:
                raise ValueError(f"Dataset must be 3D or 4D, got shape {dataset.shape}")
        else:
            raise ValueError("Dataset must have a shape attribute")
        
        # Calculate number of chunks in each dimension
        self.num_chunks_z = (self.depth + chunk_size - 1) // chunk_size
        self.num_chunks_y = (self.height + chunk_size - 1) // chunk_size
        self.num_chunks_x = (self.width + chunk_size - 1) // chunk_size
        
        print(f"DEBUG: ChunkedTensor initialized with {self.num_chunks_z}x{self.num_chunks_y}x{self.num_chunks_x} chunks, precomputed={precomputed}")
    
    def get_chunk_indices(self, z: int, y: int, x: int) -> Tuple[int, int, int]:
        """
        Get the chunk indices for a given voxel position.
        
        Args:
            z, y, x: Voxel coordinates
            
        Returns:
            Tuple of (chunk_z, chunk_y, chunk_x) indices
        """
        chunk_z = z // self.chunk_size
        chunk_y = y // self.chunk_size
        chunk_x = x // self.chunk_size
        
        # Clamp to valid chunk indices
        chunk_z = max(0, min(chunk_z, self.num_chunks_z - 1))
        chunk_y = max(0, min(chunk_y, self.num_chunks_y - 1))
        chunk_x = max(0, min(chunk_x, self.num_chunks_x - 1))
        
        return (chunk_z, chunk_y, chunk_x)
    
    def get_local_indices(self, z: int, y: int, x: int) -> Tuple[int, int, int]:
        """
        Get the local indices within a chunk for a given voxel position.
        
        Args:
            z, y, x: Voxel coordinates
            
        Returns:
            Tuple of (local_z, local_y, local_x) indices within the chunk
        """
        chunk_z, chunk_y, chunk_x = self.get_chunk_indices(z, y, x)
        
        local_z = z - chunk_z * self.chunk_size
        local_y = y - chunk_y * self.chunk_size
        local_x = x - chunk_x * self.chunk_size
        
        return (local_z, local_y, local_x)
    
    def get_chunk(self, chunk_z: int, chunk_y: int, chunk_x: int) -> torch.Tensor:
        """
        Get a processed chunk, either from cache or by computing it.
        
        Args:
            chunk_z, chunk_y, chunk_x: Chunk indices
            
        Returns:
            Processed chunk tensor
        """
        chunk_indices = (chunk_z, chunk_y, chunk_x)
        
        # Check if chunk is already cached
        cached_chunk = self.cache.get_chunk(chunk_indices)
        if cached_chunk is not None:
            return cached_chunk
            
        # If using precomputed data, just extract the appropriate chunk
        if self.precomputed:
            # Calculate chunk boundaries in the original data
            z_start = chunk_z * self.chunk_size
            y_start = chunk_y * self.chunk_size
            x_start = chunk_x * self.chunk_size
            
            z_end = min(z_start + self.chunk_size, self.depth)
            y_end = min(y_start + self.chunk_size, self.height)
            x_end = min(x_start + self.chunk_size, self.width)
            
            # Create a chunk-sized tensor
            chunk_shape = (self.chunk_size, self.chunk_size, self.chunk_size)
            chunk = torch.full(chunk_shape, float(self.compute.FILL_V), dtype=torch.float32)
            
            # Calculate valid region within chunk
            valid_z_size = z_end - z_start
            valid_y_size = y_end - y_start
            valid_x_size = x_end - x_start
            
            # Copy the data
            if len(self.dataset.shape) == 3:
                chunk[:valid_z_size, :valid_y_size, :valid_x_size] = \
                    self.dataset[z_start:z_end, y_start:y_end, x_start:x_end]
            else:
                # Use first batch if dataset has batch dimension
                chunk[:valid_z_size, :valid_y_size, :valid_x_size] = \
                    self.dataset[0, z_start:z_end, y_start:y_end, x_start:x_end]
            
            # Cache the chunk
            self.cache.add_chunk(chunk_indices, chunk)
            return chunk
        
        # Otherwise, compute the chunk from raw data
        # Calculate origin coordinates
        origin_z = chunk_z * self.chunk_size - self.border
        origin_y = chunk_y * self.chunk_size - self.border
        origin_x = chunk_x * self.chunk_size - self.border
        
        # Calculate extended size with borders
        ext_size = self.chunk_size + 2 * self.border
        
        # Create a tensor for the extended chunk (with border)
        extended_chunk = torch.full(
            (ext_size, ext_size, ext_size),
            float(self.compute.FILL_V),
            dtype=torch.float32
        )
        
        # Calculate valid ranges for source and destination
        # For source (the dataset)
        src_z_start = max(0, origin_z)
        src_y_start = max(0, origin_y)
        src_x_start = max(0, origin_x)
        
        src_z_end = min(self.depth, origin_z + ext_size)
        src_y_end = min(self.height, origin_y + ext_size)
        src_x_end = min(self.width, origin_x + ext_size)
        
        # For destination (the extended chunk)
        dst_z_start = max(0, -origin_z)
        dst_y_start = max(0, -origin_y)
        dst_x_start = max(0, -origin_x)
        
        # Calculate end ranges - will match the size of the valid region
        valid_z_size = src_z_end - src_z_start
        valid_y_size = src_y_end - src_y_start
        valid_x_size = src_x_end - src_x_start
        
        dst_z_end = dst_z_start + valid_z_size
        dst_y_end = dst_y_start + valid_y_size
        dst_x_end = dst_x_start + valid_x_size
        
        # Extract valid region from dataset in a single operation
        if len(self.dataset.shape) == 3:
            extended_chunk[dst_z_start:dst_z_end, dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                self.dataset[src_z_start:src_z_end, src_y_start:src_y_end, src_x_start:src_x_end]
        else:
            # Use first batch if dataset has batch dimension
            extended_chunk[dst_z_start:dst_z_end, dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                self.dataset[0, src_z_start:src_z_end, src_y_start:src_y_end, src_x_start:src_x_end]
        
        # Apply the compute function to get the processed chunk
        offset = (origin_z, origin_y, origin_x)
        processed_chunk = self.compute.compute(extended_chunk, offset)
        
        # Cache the processed chunk
        self.cache.add_chunk(chunk_indices, processed_chunk)
        
        return processed_chunk
    
    def __getitem__(self, coords: Tuple[int, int, int]) -> float:
        """
        Get the value at the specified coordinates.
        
        Args:
            coords: Tuple of (z, y, x) coordinates
            
        Returns:
            Value at the specified coordinates
        """
        z, y, x = coords
        
        # Get chunk indices and local indices
        chunk_indices = self.get_chunk_indices(z, y, x)
        local_z, local_y, local_x = self.get_local_indices(z, y, x)
        
        # Get the processed chunk
        chunk = self.get_chunk(*chunk_indices)
        
        # Return the value at the local coordinates
        return chunk[local_z, local_y, local_x].item()
    
    def get_interpolated_value(self, z: float, y: float, x: float) -> float:
        """
        Get an interpolated value at the specified floating-point coordinates.
        
        Args:
            z, y, x: Floating-point coordinates
            
        Returns:
            Interpolated value at the specified coordinates
        """
        # Determine which chunk contains this point
        chunk_z = int(z // self.chunk_size)
        chunk_y = int(y // self.chunk_size)
        chunk_x = int(x // self.chunk_size)
        
        # Clamp to valid chunk indices
        chunk_z = max(0, min(chunk_z, self.num_chunks_z - 1))
        chunk_y = max(0, min(chunk_y, self.num_chunks_y - 1))
        chunk_x = max(0, min(chunk_x, self.num_chunks_x - 1))
        
        # Get the chunk containing this point
        chunk = self.get_chunk(chunk_z, chunk_y, chunk_x)
        
        # Convert to local coordinates within the chunk
        local_z = z - chunk_z * self.chunk_size
        local_y = y - chunk_y * self.chunk_size
        local_x = x - chunk_x * self.chunk_size
        
        # For 3D grid_sample, we need a 5D grid of shape [N, D, H, W, 3]
        # where N is batch size, D, H, W are the output dimensions
        # and the last dimension contains the (x, y, z) normalized coordinates
        
        # Normalize coordinates to [-1, 1] range
        norm_x = 2.0 * local_x / (self.chunk_size - 1.0) - 1.0
        norm_y = 2.0 * local_y / (self.chunk_size - 1.0) - 1.0
        norm_z = 2.0 * local_z / (self.chunk_size - 1.0) - 1.0
        
        # Create the grid tensor with shape [1, 1, 1, 1, 3]
        # The coordinates are ordered [x, y, z] for grid_sample
        grid = torch.tensor([[[[[norm_x, norm_y, norm_z]]]]], 
                          dtype=torch.float32, device=chunk.device)
        
        # Add batch and channel dimensions to chunk for grid_sample
        # Shape becomes [1, 1, D, H, W]
        chunk_with_dims = chunk.unsqueeze(0).unsqueeze(0)
        
        # Use grid_sample for trilinear interpolation
        # Output shape will be [1, 1, 1, 1, 1]
        sampled = torch.nn.functional.grid_sample(
            chunk_with_dims,
            grid,
            mode='bilinear',  # 'bilinear' in 3D is actually trilinear
            align_corners=True
        )
        
        # Return the interpolated value
        return sampled.item()


class CachedChunked3dInterpolator:
    """
    Interpolator for efficient access to chunked 3D data.
    
    This class provides trilinear interpolation for chunked 3D tensors,
    similar to the CachedChunked3dInterpolator in the C++ code.
    
    Attributes:
        tensor: ChunkedTensor to interpolate
        use_cache: Whether to cache interpolation results
    """
    
    def __init__(
        self,
        tensor: ChunkedTensor,
        use_cache: bool = True,
        cache_size: int = 1000
    ):
        """
        Initialize the interpolator.
        
        Args:
            tensor: ChunkedTensor to interpolate
            use_cache: Whether to cache interpolation results
            cache_size: Maximum number of cached interpolation results
        """
        self.tensor = tensor
        self.use_cache = use_cache
        
        # Initialize cache if enabled
        if use_cache:
            self.cache_size = cache_size
            self.cache = OrderedDict()  # LRU cache
            self.cache_hits = 0
            self.cache_misses = 0
    
    def _get_cache_key(self, z: float, y: float, x: float) -> str:
        """
        Generate a cache key for the given coordinates.
        
        Args:
            z, y, x: Coordinates
            
        Returns:
            Cache key string
        """
        # Use string representation with limited precision
        return f"{z:.6f}_{y:.6f}_{x:.6f}"
    
    def evaluate(self, z: float, y: float, x: float) -> float:
        """
        Evaluate the interpolated value at the specified coordinates.
        
        Args:
            z, y, x: Coordinates
            
        Returns:
            Interpolated value at the specified coordinates
        """
        # Try cache first if enabled
        cache_key = None
        if self.use_cache:
            cache_key = self._get_cache_key(z, y, x)
            
            if cache_key in self.cache:
                # Cache hit
                self.cache_hits += 1
                cached_data = self.cache[cache_key]
                
                # Move to the end of OrderedDict (most recently used)
                self.cache.pop(cache_key)
                self.cache[cache_key] = cached_data
                
                # Check if entry has "value" key (for backward compatibility)
                if "value" in cached_data:
                    return cached_data["value"]
                else:
                    return cached_data
            else:
                # Cache miss
                self.cache_misses += 1
        
        # Get interpolated value
        value = self.tensor.get_interpolated_value(z, y, x)
        
        # Cache the result if enabled
        if self.use_cache and cache_key is not None:
            # We store value both as direct value and in dict with "value" key
            # for backward compatibility with evaluate_with_gradient
            entry = {"value": value, "gradient": None}
            self.cache[cache_key] = entry
            
            # Maintain cache size limit
            if len(self.cache) > self.cache_size:
                # Remove oldest (first) item in the ordered dict
                self.cache.popitem(last=False)
        
        return value
    
    def evaluate_with_gradient(self, z: float, y: float, x: float) -> Tuple[float, np.ndarray]:
        """
        Evaluate the interpolated value and gradient at the specified coordinates.
        Uses PyTorch autograd to compute the gradient.
        
        Args:
            z, y, x: Coordinates
            
        Returns:
            Tuple of (value, gradient) at the specified coordinates
        """
        # Try cache first if enabled
        if self.use_cache:
            cache_key = self._get_cache_key(z, y, x)
            
            if cache_key in self.cache:
                # Cache hit
                self.cache_hits += 1
                cached_data = self.cache[cache_key]
                
                # Move to the end of OrderedDict (most recently used)
                self.cache.pop(cache_key)
                self.cache[cache_key] = cached_data
                
                return cached_data["value"], cached_data["gradient"]
            else:
                # Cache miss
                self.cache_misses += 1
        
        # Determine which chunk contains this point
        chunk_z = int(z // self.tensor.chunk_size)
        chunk_y = int(y // self.tensor.chunk_size)
        chunk_x = int(x // self.tensor.chunk_size)
        
        # Clamp to valid chunk indices
        chunk_z = max(0, min(chunk_z, self.tensor.num_chunks_z - 1))
        chunk_y = max(0, min(chunk_y, self.tensor.num_chunks_y - 1))
        chunk_x = max(0, min(chunk_x, self.tensor.num_chunks_x - 1))
        
        # Get the chunk containing this point
        chunk = self.tensor.get_chunk(chunk_z, chunk_y, chunk_x)
        
        # Convert to local coordinates within the chunk
        local_z = z - chunk_z * self.tensor.chunk_size
        local_y = y - chunk_y * self.tensor.chunk_size
        local_x = x - chunk_x * self.tensor.chunk_size
        
        # Normalize coordinates to [-1, 1] range
        norm_x = 2.0 * local_x / (self.tensor.chunk_size - 1.0) - 1.0
        norm_y = 2.0 * local_y / (self.tensor.chunk_size - 1.0) - 1.0
        norm_z = 2.0 * local_z / (self.tensor.chunk_size - 1.0) - 1.0
        
        # Create the grid tensor with shape [1, 1, 1, 1, 3]
        # The coordinates are ordered [x, y, z] for grid_sample
        grid = torch.tensor([[[[[norm_x, norm_y, norm_z]]]]], 
                           dtype=torch.float32, device=chunk.device)
        
        # Add batch and channel dimensions to chunk
        chunk_with_dims = chunk.unsqueeze(0).unsqueeze(0)  # [1, 1, depth, height, width]
        
        # Use grid_sample for trilinear interpolation
        # align_corners=True to match our explicit interpolation at endpoints
        sampled = torch.nn.functional.grid_sample(
            chunk_with_dims, 
            grid,
            mode='bilinear',  # 'bilinear' in 3D is actually trilinear
            align_corners=True
        )
        
        # Get the interpolated value
        value = sampled.item()
        
        # Compute gradients using PyTorch autograd with finite differences
        # This is more reliable than trying to do backprop through grid_sample
        # with only a single point, especially since we need ZYX order gradients
        epsilon = 0.01
        
        # Z gradient
        z_plus = z + epsilon
        z_minus = z - epsilon
        local_z_plus = z_plus - chunk_z * self.tensor.chunk_size
        local_z_minus = z_minus - chunk_z * self.tensor.chunk_size
        
        # Ensure we're still in the same chunk
        if 0 <= local_z_plus < self.tensor.chunk_size and 0 <= local_z_minus < self.tensor.chunk_size:
            # Create normalized coordinates
            norm_z_plus = 2.0 * local_z_plus / (self.tensor.chunk_size - 1.0) - 1.0
            norm_z_minus = 2.0 * local_z_minus / (self.tensor.chunk_size - 1.0) - 1.0
            
            # Create the grid tensors
            grid_plus = torch.tensor([[[[[norm_x, norm_y, norm_z_plus]]]]], 
                                   dtype=torch.float32, device=chunk.device)
            
            grid_minus = torch.tensor([[[[[norm_x, norm_y, norm_z_minus]]]]], 
                                    dtype=torch.float32, device=chunk.device)
            
            # Compute samples
            sample_plus = torch.nn.functional.grid_sample(
                chunk_with_dims, grid_plus, mode='bilinear', align_corners=True
            ).item()
            
            sample_minus = torch.nn.functional.grid_sample(
                chunk_with_dims, grid_minus, mode='bilinear', align_corners=True
            ).item()
            
            dz = (sample_plus - sample_minus) / (2 * epsilon)
        else:
            # We crossed chunk boundaries, use the tensor's interpolation
            dz = (self.tensor.get_interpolated_value(z + epsilon, y, x) -
                  self.tensor.get_interpolated_value(z - epsilon, y, x)) / (2 * epsilon)
        
        # Y gradient (same approach)
        y_plus = y + epsilon
        y_minus = y - epsilon
        local_y_plus = y_plus - chunk_y * self.tensor.chunk_size
        local_y_minus = y_minus - chunk_y * self.tensor.chunk_size
        
        if 0 <= local_y_plus < self.tensor.chunk_size and 0 <= local_y_minus < self.tensor.chunk_size:
            # Create normalized coordinates
            norm_y_plus = 2.0 * local_y_plus / (self.tensor.chunk_size - 1.0) - 1.0
            norm_y_minus = 2.0 * local_y_minus / (self.tensor.chunk_size - 1.0) - 1.0
            
            # Create the grid tensors
            grid_plus = torch.tensor([[[[[norm_x, norm_y_plus, norm_z]]]]], 
                                   dtype=torch.float32, device=chunk.device)
            
            grid_minus = torch.tensor([[[[[norm_x, norm_y_minus, norm_z]]]]], 
                                    dtype=torch.float32, device=chunk.device)
            
            sample_plus = torch.nn.functional.grid_sample(
                chunk_with_dims, grid_plus, mode='bilinear', align_corners=True
            ).item()
            
            sample_minus = torch.nn.functional.grid_sample(
                chunk_with_dims, grid_minus, mode='bilinear', align_corners=True
            ).item()
            
            dy = (sample_plus - sample_minus) / (2 * epsilon)
        else:
            dy = (self.tensor.get_interpolated_value(z, y + epsilon, x) -
                  self.tensor.get_interpolated_value(z, y - epsilon, x)) / (2 * epsilon)
        
        # X gradient
        x_plus = x + epsilon
        x_minus = x - epsilon
        local_x_plus = x_plus - chunk_x * self.tensor.chunk_size
        local_x_minus = x_minus - chunk_x * self.tensor.chunk_size
        
        if 0 <= local_x_plus < self.tensor.chunk_size and 0 <= local_x_minus < self.tensor.chunk_size:
            # Create normalized coordinates
            norm_x_plus = 2.0 * local_x_plus / (self.tensor.chunk_size - 1.0) - 1.0
            norm_x_minus = 2.0 * local_x_minus / (self.tensor.chunk_size - 1.0) - 1.0
            
            # Create the grid tensors
            grid_plus = torch.tensor([[[[[norm_x_plus, norm_y, norm_z]]]]], 
                                   dtype=torch.float32, device=chunk.device)
            
            grid_minus = torch.tensor([[[[[norm_x_minus, norm_y, norm_z]]]]], 
                                    dtype=torch.float32, device=chunk.device)
            
            sample_plus = torch.nn.functional.grid_sample(
                chunk_with_dims, grid_plus, mode='bilinear', align_corners=True
            ).item()
            
            sample_minus = torch.nn.functional.grid_sample(
                chunk_with_dims, grid_minus, mode='bilinear', align_corners=True
            ).item()
            
            dx = (sample_plus - sample_minus) / (2 * epsilon)
        else:
            dx = (self.tensor.get_interpolated_value(z, y, x + epsilon) -
                  self.tensor.get_interpolated_value(z, y, x - epsilon)) / (2 * epsilon)
        
        # Create gradient array (z, y, x order)
        gradient = np.array([dz, dy, dx], dtype=np.float32)
        
        # Cache the results if enabled
        if self.use_cache:
            self.cache[cache_key] = {
                "value": value,
                "gradient": gradient
            }
            
            # Maintain cache size limit
            if len(self.cache) > self.cache_size:
                # Remove oldest (first) item in the ordered dict
                self.cache.popitem(last=False)
        
        return value, gradient
    
    def clear_cache(self) -> None:
        """Clear the cache if enabled."""
        if self.use_cache:
            self.cache.clear()
            self.cache_hits = 0
            self.cache_misses = 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics if cache is enabled.
        
        Returns:
            Dictionary with cache statistics
        """
        if not self.use_cache:
            return {"cache_enabled": False}
        
        total_accesses = self.cache_hits + self.cache_misses
        hit_ratio = self.cache_hits / total_accesses if total_accesses > 0 else 0.0
        
        return {
            "cache_size": len(self.cache),
            "max_size": self.cache_size,
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_ratio": hit_ratio,
            "cache_enabled": True
        }