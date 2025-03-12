"""Tests for the Chunked3D class."""

import os
import tempfile
import shutil
import pathlib
import numpy as np
import pytest
import zarr

from surfaces.chunked_3d import ChunkCache, Chunked3D, Chunked3DAccessor, open_zarr_volume


class TestChunkCache:
    """Test cases for the ChunkCache class."""
    
    def test_initialization(self):
        """Test initialization of ChunkCache."""
        cache = ChunkCache(max_size=50)
        assert cache.max_size == 50
        assert len(cache.cache) == 0
        assert cache.hits == 0
        assert cache.misses == 0
        
    def test_get_and_cache(self):
        """Test get method with caching."""
        cache = ChunkCache(max_size=10)
        
        # Define a compute function that counts calls
        call_count = 0
        def compute_func():
            nonlocal call_count
            call_count += 1
            return np.ones((4, 4, 4))
            
        # First call - should compute and cache
        result1 = cache.get(("key1", (0, 0, 0)), compute_func)
        assert call_count == 1
        assert cache.hits == 0
        assert cache.misses == 1
        assert result1.shape == (4, 4, 4)
        assert np.all(result1 == 1)
        
        # Second call with same key - should use cached value
        result2 = cache.get(("key1", (0, 0, 0)), compute_func)
        assert call_count == 1  # Still 1, compute_func not called again
        assert cache.hits == 1
        assert cache.misses == 1
        assert result2 is result1  # Should be same object
        
        # Call with different key - should compute and cache
        result3 = cache.get(("key2", (1, 1, 1)), compute_func)
        assert call_count == 2
        assert cache.hits == 1
        assert cache.misses == 2
        assert result3 is not result1
        
    def test_cache_size_limit(self):
        """Test that cache respects size limit."""
        cache = ChunkCache(max_size=3)
        
        # Add 5 items (exceeding the limit)
        for i in range(5):
            key = (f"key{i}", (i, i, i))
            cache.get(key, lambda: np.ones((2, 2, 2)) * i)
        
        # Should only keep 3 items
        assert len(cache.cache) == 3
        
        # Should keep the most recent 3 items
        assert ("key2", (2, 2, 2)) in cache.cache
        assert ("key3", (3, 3, 3)) in cache.cache
        assert ("key4", (4, 4, 4)) in cache.cache
        
        # Earlier items should be evicted
        assert ("key0", (0, 0, 0)) not in cache.cache
        assert ("key1", (1, 1, 1)) not in cache.cache
        
    def test_clear(self):
        """Test clear method."""
        cache = ChunkCache(max_size=10)
        
        # Add some items
        for i in range(3):
            key = (f"key{i}", (i, i, i))
            cache.get(key, lambda: np.ones((2, 2, 2)))
        
        # Check initial state
        assert len(cache.cache) == 3
        assert cache.misses == 3
        
        # Clear cache
        cache.clear()
        
        # Check cleared state
        assert len(cache.cache) == 0
        assert cache.hits == 0
        assert cache.misses == 0
        
    def test_get_stats(self):
        """Test get_stats method."""
        cache = ChunkCache(max_size=10)
        
        # Add some items
        cache.get("key1", lambda: np.ones((2, 2, 2)))
        cache.get("key1", lambda: np.ones((2, 2, 2)))  # Hit
        cache.get("key2", lambda: np.ones((2, 2, 2)))
        
        # Get stats
        stats = cache.get_stats()
        
        # Check stats values
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert stats["size"] == 2
        assert stats["max_size"] == 10
        assert stats["hit_ratio"] == 1/3


class TestChunked3D:
    """Test cases for the Chunked3D class."""
    
    @pytest.fixture
    def zarr_file(self):
        """Create a temporary zarr file for testing."""
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Create a zarr array
        z = zarr.open(os.path.join(temp_dir, "test.zarr"), mode="w")
        data = np.arange(1000).reshape(10, 10, 10)
        z.create_dataset("data", data=data, chunks=(3, 3, 3))
        
        yield os.path.join(temp_dir, "test.zarr")
        
        # Clean up
        shutil.rmtree(temp_dir)
    
    def test_initialization(self, zarr_file):
        """Test initialization of Chunked3D."""
        # Open zarr array
        z = zarr.open(zarr_file, mode="r")
        dataset = z["data"]
        
        # Create Chunked3D
        chunked = Chunked3D(dataset)
        
        # Check attributes
        assert chunked.shape == (10, 10, 10)
        assert chunked.chunks == (3, 3, 3)
        assert chunked.ndim == 3
        assert chunked.num_chunks == (4, 4, 4)  # Ceiling division
        assert chunked.fill_value == 0.0
        
        # Test with different fill value
        chunked2 = Chunked3D(dataset, fill_value=-1.0)
        assert chunked2.fill_value == -1.0
        
    def test_get_chunk(self, zarr_file):
        """Test get_chunk method."""
        # Open zarr array
        z = zarr.open(zarr_file, mode="r")
        dataset = z["data"]
        
        # Create Chunked3D
        chunked = Chunked3D(dataset)
        
        # Get a chunk
        chunk = chunked.get_chunk((0, 0, 0))
        
        # Check chunk shape and values
        assert chunk.shape == (3, 3, 3)
        
        # Check specific values
        assert chunk[0, 0, 0] == dataset[0, 0, 0]
        assert chunk[2, 2, 2] == dataset[2, 2, 2]
        
        # Test out-of-bounds chunk
        out_chunk = chunked.get_chunk((10, 10, 10))
        assert out_chunk.shape == (3, 3, 3)
        assert np.all(out_chunk == chunked.fill_value)
        
    def test_call_operator(self, zarr_file):
        """Test __call__ method for direct access."""
        # Open zarr array
        z = zarr.open(zarr_file, mode="r")
        dataset = z["data"]
        
        # Create Chunked3D
        chunked = Chunked3D(dataset)
        
        # Check values using direct access
        assert chunked(0, 0, 0) == dataset[0, 0, 0]
        assert chunked(5, 5, 5) == dataset[5, 5, 5]
        assert chunked(9, 9, 9) == dataset[9, 9, 9]
        
        # Check out-of-bounds access
        assert chunked(-1, 0, 0) == chunked.fill_value
        assert chunked(0, -1, 0) == chunked.fill_value
        assert chunked(0, 0, -1) == chunked.fill_value
        assert chunked(10, 0, 0) == chunked.fill_value
        assert chunked(0, 10, 0) == chunked.fill_value
        assert chunked(0, 0, 10) == chunked.fill_value
        
    def test_get_area(self, zarr_file):
        """Test get_area method."""
        # Open zarr array
        z = zarr.open(zarr_file, mode="r")
        dataset = z["data"]
        data = np.array(dataset)  # Get full data as numpy array
        
        # Create Chunked3D
        chunked = Chunked3D(dataset)
        
        # Get a 2x2x2 area
        area = chunked.get_area(1, 1, 1, 2, 2, 2)
        expected = data[1:3, 1:3, 1:3]
        assert area.shape == (2, 2, 2)
        assert np.all(area == expected)
        
        # Test area that crosses chunk boundaries
        area2 = chunked.get_area(2, 2, 2, 3, 3, 3)
        expected2 = data[2:5, 2:5, 2:5]
        assert area2.shape == (3, 3, 3)
        assert np.all(area2 == expected2)
        
        # Test area that goes out of bounds
        area3 = chunked.get_area(8, 8, 8, 4, 4, 4)
        expected_shape = (4, 4, 4)
        assert area3.shape == expected_shape
        
        # Check inside portion matches original data
        valid_z = slice(0, 2)
        valid_y = slice(0, 2)
        valid_x = slice(0, 2)
        assert np.all(area3[valid_z, valid_y, valid_x] == data[8:10, 8:10, 8:10])
        
        # Check outside portion is filled with fill_value
        assert np.all(area3[2:, :, :] == chunked.fill_value)
        assert np.all(area3[:, 2:, :] == chunked.fill_value)
        assert np.all(area3[:, :, 2:] == chunked.fill_value)
        
    def test_shape_info(self, zarr_file):
        """Test shape_info method."""
        # Open zarr array
        z = zarr.open(zarr_file, mode="r")
        dataset = z["data"]
        
        # Create Chunked3D
        chunked = Chunked3D(dataset)
        
        # Check shape info
        assert chunked.shape_info() == [10, 10, 10]


class TestChunked3DAccessor:
    """Test cases for the Chunked3DAccessor class."""
    
    @pytest.fixture
    def zarr_file(self):
        """Create a temporary zarr file for testing."""
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Create a zarr array
        z = zarr.open(os.path.join(temp_dir, "test.zarr"), mode="w")
        data = np.arange(1000).reshape(10, 10, 10)
        z.create_dataset("data", data=data, chunks=(3, 3, 3))
        
        yield os.path.join(temp_dir, "test.zarr")
        
        # Clean up
        shutil.rmtree(temp_dir)
    
    def test_accessor(self, zarr_file):
        """Test Chunked3DAccessor."""
        # Open zarr array
        z = zarr.open(zarr_file, mode="r")
        dataset = z["data"]
        
        # Create Chunked3D
        chunked = Chunked3D(dataset)
        
        # Create accessor
        accessor = Chunked3DAccessor(chunked)
        
        # Check initial state
        assert accessor.current_chunk is None
        assert accessor.current_chunk_coords == (-1, -1, -1)
        
        # Access a point
        value1 = accessor(1, 1, 1)
        assert value1 == dataset[1, 1, 1]
        
        # Check that chunk was loaded
        assert accessor.current_chunk is not None
        assert accessor.current_chunk_coords == (0, 0, 0)
        
        # Access another point in the same chunk
        value2 = accessor(2, 2, 2)
        assert value2 == dataset[2, 2, 2]
        
        # Chunk should not have changed
        assert accessor.current_chunk_coords == (0, 0, 0)
        
        # Access a point in a different chunk
        value3 = accessor(4, 4, 4)
        assert value3 == dataset[4, 4, 4]
        
        # Chunk should have changed
        assert accessor.current_chunk_coords == (1, 1, 1)
        
        # Check out-of-bounds access
        assert accessor(-1, 0, 0) == chunked.fill_value
        assert accessor(10, 0, 0) == chunked.fill_value