"""
Chunked3D class for efficient access to chunked volumetric data.

This is a Python implementation of the Chunked3D C++ class from volume-cartographer.
It provides efficient access to chunked 3D data with caching.

Coordinate Convention:
- 3D coordinates are in ZYX order [z, y, x] with 0=z, 1=y, 2=x
- Volume data is indexed as volume[z, y, x]
- All access methods take parameters in ZYX order (z, y, x)
- Chunk coordinates are also ordered as (z_chunk, y_chunk, x_chunk)
"""

import os
import functools
import pathlib
from typing import Dict, Optional, Tuple, List, Any, Union, Callable

import numpy as np
import zarr
from zarr.storage import Store


class ChunkCache:
    """A simple cache for zarr chunks."""
    
    def __init__(self, max_size: int = 100):
        """
        Initialize the chunk cache.
        
        Args:
            max_size: Maximum number of chunks to store in the cache
        """
        self.cache: Dict[Tuple[str, Tuple[int, ...]], np.ndarray] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        
    def get(self, key: Tuple[str, Tuple[int, ...]], compute_func: Callable[[], np.ndarray]) -> np.ndarray:
        """
        Get a chunk from the cache or compute it.
        
        Args:
            key: Chunk key (dataset path, chunk indices)
            compute_func: Function to compute the chunk if not in cache
            
        Returns:
            The chunk data
        """
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        
        self.misses += 1
        chunk = compute_func()
        
        # Add to cache
        self.cache[key] = chunk
        
        # Limit cache size by removing oldest entries if needed
        if len(self.cache) > self.max_size:
            # Remove the first key (oldest entry in the OrderedDict)
            self.cache.pop(next(iter(self.cache)))
            
        return chunk
    
    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_ratio = self.hits / total if total > 0 else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_ratio": hit_ratio
        }


class Chunked3D:
    """
    Class for efficient access to chunked volumetric data.
    
    This class provides access to 3D data stored in zarr format with
    chunk-based caching for efficiency.
    """
    
    def __init__(
        self, 
        dataset: zarr.Array, 
        cache: Optional[ChunkCache] = None,
        fill_value: float = 0.0,
        cache_root: Optional[Union[str, pathlib.Path]] = None
    ):
        """
        Initialize a Chunked3D object.
        
        Args:
            dataset: Zarr array containing volumetric data
            cache: Optional chunk cache (created if not provided)
            fill_value: Value used for out-of-bounds access
            cache_root: Optional root directory for persistent cache
        """
        self.dataset = dataset
        self.cache = cache or ChunkCache()
        self.fill_value = fill_value
        self.cache_root = pathlib.Path(cache_root) if cache_root else None
        
        # Get dataset information
        self.shape = dataset.shape
        self.chunks = dataset.chunks
        self.ndim = dataset.ndim
        self.dtype = dataset.dtype
        
        # Compute number of chunks in each dimension
        self.num_chunks = tuple(
            (s + c - 1) // c for s, c in zip(self.shape, self.chunks)
        )
        
        # Verify is a 3D array
        if self.ndim != 3:
            raise ValueError(f"Expected 3D array, got {self.ndim}D")
            
    def __call__(self, z: int, y: int, x: int) -> float:
        """
        Access data at the specified 3D coordinates.
        
        Args:
            z: Z coordinate
            y: Y coordinate
            x: X coordinate
            
        Returns:
            Data value at the specified coordinates
        """
        # Check bounds
        if (
            z < 0 or z >= self.shape[0] or
            y < 0 or y >= self.shape[1] or
            x < 0 or x >= self.shape[2]
        ):
            return self.fill_value
            
        # Get chunk indices
        chunk_coords = self._get_chunk_coords(z, y, x)
        
        # Get chunk
        chunk = self.get_chunk(chunk_coords)
        
        # Get local coordinates within the chunk
        local_z = z % self.chunks[0]
        local_y = y % self.chunks[1]
        local_x = x % self.chunks[2]
        
        # Return value
        return float(chunk[local_z, local_y, local_x])
    
    def safe_at(self, z: int, y: int, x: int) -> float:
        """
        Thread-safe version of __call__.
        
        Args:
            z: Z coordinate
            y: Y coordinate
            x: X coordinate
            
        Returns:
            Data value at the specified coordinates
        """
        # For Python, we don't need a separate implementation since GIL 
        # already provides thread safety for the cache
        return self(z, y, x)
    
    def _get_chunk_coords(self, z: int, y: int, x: int) -> Tuple[int, int, int]:
        """
        Get chunk coordinates for the given position.
        
        Args:
            z: Z coordinate
            y: Y coordinate
            x: X coordinate
            
        Returns:
            Tuple of chunk indices (chunk_z, chunk_y, chunk_x)
        """
        chunk_z = z // self.chunks[0]
        chunk_y = y // self.chunks[1]
        chunk_x = x // self.chunks[2]
        return (chunk_z, chunk_y, chunk_x)
    
    def get_chunk(self, chunk_coords: Tuple[int, int, int]) -> np.ndarray:
        """
        Get a chunk of data.
        
        Args:
            chunk_coords: Chunk coordinates (chunk_z, chunk_y, chunk_x)
            
        Returns:
            Chunk data as numpy array
        """
        # Check if chunk is within bounds
        for i, c in enumerate(chunk_coords):
            if c < 0 or c >= self.num_chunks[i]:
                # Return a chunk filled with the fill value
                return np.full(self.chunks, self.fill_value, dtype=self.dtype)
        
        # Create a unique key for this chunk
        cache_key = (str(self.dataset.path), chunk_coords)
        
        # Get from cache or compute
        def compute_chunk():
            # Calculate the slice for this chunk
            z_slice = slice(
                chunk_coords[0] * self.chunks[0], 
                min((chunk_coords[0] + 1) * self.chunks[0], self.shape[0])
            )
            y_slice = slice(
                chunk_coords[1] * self.chunks[1], 
                min((chunk_coords[1] + 1) * self.chunks[1], self.shape[1])
            )
            x_slice = slice(
                chunk_coords[2] * self.chunks[2], 
                min((chunk_coords[2] + 1) * self.chunks[2], self.shape[2])
            )
            
            # Get the chunk data
            chunk = self.dataset[z_slice, y_slice, x_slice]
            
            # Pad if needed to full chunk size
            if chunk.shape != self.chunks:
                padded_chunk = np.full(self.chunks, self.fill_value, dtype=self.dtype)
                padded_chunk[:chunk.shape[0], :chunk.shape[1], :chunk.shape[2]] = chunk
                chunk = padded_chunk
                
            return chunk
        
        return self.cache.get(cache_key, compute_chunk)
    
    def get_area(
        self, 
        start_z: int, 
        start_y: int, 
        start_x: int, 
        size_z: int, 
        size_y: int, 
        size_x: int
    ) -> np.ndarray:
        """
        Get a 3D area of data.
        
        Args:
            start_z: Start Z coordinate
            start_y: Start Y coordinate  
            start_x: Start X coordinate
            size_z: Size in Z dimension
            size_y: Size in Y dimension
            size_x: Size in X dimension
            
        Returns:
            3D array of data
        """
        # Create output array
        output = np.full((size_z, size_y, size_x), self.fill_value, dtype=self.dtype)
        
        # Calculate bounds
        min_z = max(0, start_z)
        max_z = min(self.shape[0], start_z + size_z)
        min_y = max(0, start_y)
        max_y = min(self.shape[1], start_y + size_y)
        min_x = max(0, start_x)
        max_x = min(self.shape[2], start_x + size_x)
        
        # Check if any part is within bounds
        if min_z >= max_z or min_y >= max_y or min_x >= max_x:
            return output
            
        # Calculate chunk ranges
        chunk_min_z, chunk_min_y, chunk_min_x = self._get_chunk_coords(min_z, min_y, min_x)
        chunk_max_z, chunk_max_y, chunk_max_x = self._get_chunk_coords(max_z - 1, max_y - 1, max_x - 1)
        
        # Iterate over chunks
        for chunk_z in range(chunk_min_z, chunk_max_z + 1):
            for chunk_y in range(chunk_min_y, chunk_max_y + 1):
                for chunk_x in range(chunk_min_x, chunk_max_x + 1):
                    # Get chunk
                    chunk = self.get_chunk((chunk_z, chunk_y, chunk_x))
                    
                    # Calculate overlap between chunk and requested area
                    chunk_start_z = chunk_z * self.chunks[0]
                    chunk_start_y = chunk_y * self.chunks[1]
                    chunk_start_x = chunk_x * self.chunks[2]
                    
                    # Calculate intersection
                    isect_min_z = max(min_z, chunk_start_z)
                    isect_max_z = min(max_z, chunk_start_z + self.chunks[0])
                    isect_min_y = max(min_y, chunk_start_y)
                    isect_max_y = min(max_y, chunk_start_y + self.chunks[1])
                    isect_min_x = max(min_x, chunk_start_x)
                    isect_max_x = min(max_x, chunk_start_x + self.chunks[2])
                    
                    # Calculate local coordinates in chunk
                    local_min_z = isect_min_z - chunk_start_z
                    local_max_z = isect_max_z - chunk_start_z
                    local_min_y = isect_min_y - chunk_start_y
                    local_max_y = isect_max_y - chunk_start_y
                    local_min_x = isect_min_x - chunk_start_x
                    local_max_x = isect_max_x - chunk_start_x
                    
                    # Calculate destination coordinates in output
                    dest_min_z = isect_min_z - start_z
                    dest_max_z = isect_max_z - start_z
                    dest_min_y = isect_min_y - start_y
                    dest_max_y = isect_max_y - start_y
                    dest_min_x = isect_min_x - start_x
                    dest_max_x = isect_max_x - start_x
                    
                    # Copy data
                    output[
                        dest_min_z:dest_max_z,
                        dest_min_y:dest_max_y,
                        dest_min_x:dest_max_x
                    ] = chunk[
                        local_min_z:local_max_z,
                        local_min_y:local_max_y,
                        local_min_x:local_max_x
                    ]
                    
        return output
    
    def shape_info(self) -> List[int]:
        """
        Get the shape of the dataset.
        
        Returns:
            List containing dimensions [z, y, x]
        """
        return list(self.shape)


class Chunked3DAccessor:
    """
    Accessor class for Chunked3D that provides caching of the current chunk.
    
    This class provides a more efficient access pattern when multiple 
    consecutive accesses are within the same chunk.
    """
    
    def __init__(self, chunked_3d: Chunked3D):
        """
        Initialize a Chunked3DAccessor.
        
        Args:
            chunked_3d: The Chunked3D object to access
        """
        self.chunked_3d = chunked_3d
        self.current_chunk = None
        self.current_chunk_coords = (-1, -1, -1)
        
    def __call__(self, z: int, y: int, x: int) -> float:
        """
        Access data at the specified 3D coordinates.
        
        Args:
            z: Z coordinate
            y: Y coordinate
            x: X coordinate
            
        Returns:
            Data value at the specified coordinates
        """
        # Check bounds
        if (
            z < 0 or z >= self.chunked_3d.shape[0] or
            y < 0 or y >= self.chunked_3d.shape[1] or
            x < 0 or x >= self.chunked_3d.shape[2]
        ):
            return self.chunked_3d.fill_value
            
        # Get chunk coordinates
        chunk_coords = self.chunked_3d._get_chunk_coords(z, y, x)
        
        # Check if we need to load a new chunk
        if chunk_coords != self.current_chunk_coords:
            self.current_chunk = self.chunked_3d.get_chunk(chunk_coords)
            self.current_chunk_coords = chunk_coords
            
        # Get local coordinates within the chunk
        local_z = z % self.chunked_3d.chunks[0]
        local_y = y % self.chunked_3d.chunks[1]
        local_x = x % self.chunked_3d.chunks[2]
        
        # Return value
        return float(self.current_chunk[local_z, local_y, local_x])
    
    def safe_at(self, z: int, y: int, x: int) -> float:
        """
        Thread-safe version of __call__.
        
        Args:
            z: Z coordinate
            y: Y coordinate
            x: X coordinate
            
        Returns:
            Data value at the specified coordinates
        """
        # For Python, we don't need a separate implementation since GIL 
        # already provides thread safety
        return self(z, y, x)


def open_zarr_volume(
    path: Union[str, pathlib.Path], 
    cache_size: int = 100,
    fill_value: float = 0.0,
    cache_root: Optional[Union[str, pathlib.Path]] = None
) -> Chunked3D:
    """
    Open a zarr volume as a Chunked3D object.
    
    Args:
        path: Path to zarr file or directory
        cache_size: Size of chunk cache
        fill_value: Value used for out-of-bounds access
        cache_root: Optional root directory for persistent cache
        
    Returns:
        Chunked3D object for accessing the volume
    """
    path = pathlib.Path(path)
    
    # Open zarr array
    if path.is_dir():
        # Directory-based zarr
        z = zarr.open(str(path), mode='r')
    else:
        # File-based zarr (like .n5)
        store = zarr.N5Store(str(path)) if path.suffix == '.n5' else zarr.ZipStore(str(path))
        z = zarr.open(store, mode='r')
    
    # Handle nested arrays - get the first array we find
    dataset = None
    
    # Function to recursively find the first array
    def find_first_array(group):
        for name, obj in group.items():
            if isinstance(obj, zarr.Array) and obj.ndim == 3:
                return obj
            elif isinstance(obj, zarr.Group):
                found = find_first_array(obj)
                if found is not None:
                    return found
        return None
    
    # If z is a group, find the first 3D array
    if isinstance(z, zarr.Group):
        dataset = find_first_array(z)
        if dataset is None:
            raise ValueError(f"No 3D array found in {path}")
    elif isinstance(z, zarr.Array) and z.ndim == 3:
        dataset = z
    else:
        raise ValueError(f"Expected a zarr group or 3D array, got {type(z)}")
    
    # Create cache
    cache = ChunkCache(max_size=cache_size)
    
    # Create chunked 3D
    return Chunked3D(dataset, cache, fill_value, cache_root)