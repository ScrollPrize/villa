"""
Tests for the distance transform implementation.

This module tests the core distance transform functions
and the chunked tensor implementation.
"""

import unittest
import torch
import numpy as np
import os
import shutil
import tempfile
from pathlib import Path

from tracer.core.distance_transform import (
    distance_transform,
    thresholded_distance,
    create_distance_field,
    ChunkCache,
    ChunkedTensor,
    CachedChunked3dInterpolator
)


class TestDistanceTransform(unittest.TestCase):
    """Test cases for the distance transform implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple 3D binary volume for testing
        size = 32
        self.volume = torch.zeros((size, size, size), dtype=torch.float32)
        
        # Create a simple sphere in the middle
        center = size // 2
        radius = size // 4
        
        for z in range(size):
            for y in range(size):
                for x in range(size):
                    # Distance from center
                    dist = np.sqrt((z - center)**2 + (y - center)**2 + (x - center)**2)
                    # Set voxels inside sphere to 255 (object), outside to 0 (background)
                    if dist <= radius:
                        self.volume[z, y, x] = 255.0
        
        # Create a temporary directory for chunk caching
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_distance_transform_basic(self):
        """Test basic distance transform functionality."""
        # Create binary mask with magic values
        mask = torch.zeros_like(self.volume)
        mask[self.volume < 170] = -1.0  # Non-object points get magic value
        
        # Apply distance transform
        result = distance_transform(mask, steps=10)
        
        # Verify result shape
        self.assertEqual(result.shape, self.volume.shape)
        
        # Verify that points inside the sphere (object) have distance 0
        center = self.volume.shape[0] // 2
        self.assertEqual(result[center, center, center].item(), 0.0)
        
        # Verify that points far from the sphere have distance greater than 0
        self.assertGreater(result[0, 0, 0].item(), 0.0)
    
    def test_thresholded_distance(self):
        """Test thresholded distance transform."""
        # Create compute object with smaller chunk size for testing
        compute = thresholded_distance(threshold=170.0, border=4, chunk_size=16)
        
        # Extract a chunk with border
        chunk_size = compute.CHUNK_SIZE
        border = compute.BORDER
        size = chunk_size + 2 * border
        
        # Center the chunk around the middle of the volume
        center = self.volume.shape[0] // 2
        start = center - size // 2
        end = start + size
        
        # Extract chunk with border
        chunk = self.volume[start:end, start:end, start:end]
        
        # Apply thresholded distance transform
        result = compute.compute(chunk)
        
        # Verify result shape
        self.assertEqual(result.shape, (chunk_size, chunk_size, chunk_size))
        
        # Verify that points inside the sphere have distance 0
        middle = chunk_size // 2
        self.assertEqual(result[middle, middle, middle].item(), 0.0)
        
        # Verify that points outside the sphere have positive distance
        self.assertGreater(result[0, 0, 0].item(), 0.0)
    
    def test_create_distance_field(self):
        """Test high-level distance field creation."""
        # Create distance field
        distance_field = create_distance_field(self.volume, threshold=170.0, steps=10)
        
        # Verify result shape
        self.assertEqual(distance_field.shape, self.volume.shape)
        
        # Verify that points inside the sphere (object) have distance 0
        center = self.volume.shape[0] // 2
        self.assertEqual(distance_field[center, center, center].item(), 0.0)
        
        # Verify that points far from the sphere have distance > 0
        self.assertGreater(distance_field[0, 0, 0].item(), 0.0)
    
    def test_chunk_cache(self):
        """Test chunk cache functionality."""
        # Create a cache
        cache = ChunkCache(max_memory_chunks=2, cache_dir=self.temp_dir, use_disk_cache=True)
        
        # Create some test chunks
        chunk1 = torch.ones((10, 10, 10))
        chunk2 = torch.zeros((10, 10, 10))
        chunk3 = torch.full((10, 10, 10), 2.0)
        
        # Add chunks to cache
        cache.add_chunk((0, 0, 0), chunk1)
        cache.add_chunk((0, 0, 1), chunk2)
        
        # Verify chunks are in cache
        self.assertTrue(cache.has_chunk((0, 0, 0)))
        self.assertTrue(cache.has_chunk((0, 0, 1)))
        
        # Get chunks from cache
        retrieved_chunk1 = cache.get_chunk((0, 0, 0))
        self.assertTrue(torch.all(retrieved_chunk1 == chunk1))
        
        # Add a third chunk that should evict the first one to disk
        cache.add_chunk((0, 1, 0), chunk3)
        
        # First chunk should now be on disk, not in memory
        self.assertTrue(cache.has_chunk((0, 0, 0)))
        
        # Get the first chunk back - should load from disk
        retrieved_chunk1_again = cache.get_chunk((0, 0, 0))
        self.assertTrue(torch.all(retrieved_chunk1_again == chunk1))
        
        # Get stats
        stats = cache.get_stats()
        self.assertEqual(stats["memory_chunks"], 2)
        self.assertGreaterEqual(stats["disk_chunks"], 1)
    
    def test_chunked_tensor(self):
        """Test chunked tensor functionality."""
        # Create a small volume that's exactly one chunk size
        compute = thresholded_distance(threshold=170.0)
        volume = torch.zeros((compute.CHUNK_SIZE, compute.CHUNK_SIZE, compute.CHUNK_SIZE))
        center = compute.CHUNK_SIZE // 2
        radius = compute.CHUNK_SIZE // 4
        
        # Create a sphere
        for z in range(compute.CHUNK_SIZE):
            for y in range(compute.CHUNK_SIZE):
                for x in range(compute.CHUNK_SIZE):
                    dist = np.sqrt((z - center)**2 + (y - center)**2 + (x - center)**2)
                    if dist <= radius:
                        volume[z, y, x] = 255.0
        
        # Process the volume first
        processed_volume = compute.compute(volume.clone())
        
        # Create chunked tensor with precomputed=True
        chunked = ChunkedTensor(compute, processed_volume, precomputed=True)
        
        # Test __getitem__
        # Point inside sphere should have distance 0
        value = chunked[center, center, center]
        self.assertEqual(value, 0.0)
        
        # Point outside sphere should have distance > 0
        value = chunked[0, 0, 0]
        self.assertGreater(value, 0.0)
        
        # Test interpolation
        value = chunked.get_interpolated_value(float(center), float(center), float(center))
        self.assertEqual(value, 0.0)
    
    def test_cached_interpolator(self):
        """Test cached chunked 3D interpolator."""
        # Create a small volume that's exactly one chunk size
        print("\n----- Starting test_cached_interpolator -----")
        compute = thresholded_distance(threshold=170.0, chunk_size=16, border=4)
        print(f"Created thresholded_distance with chunk_size={compute.CHUNK_SIZE}, border={compute.BORDER}, TH={compute.TH}")
        
        volume = torch.zeros((compute.CHUNK_SIZE, compute.CHUNK_SIZE, compute.CHUNK_SIZE))
        center = compute.CHUNK_SIZE // 2
        radius = compute.CHUNK_SIZE // 4
        print(f"Creating sphere with center={center}, radius={radius}")
        
        # Create a sphere
        for z in range(compute.CHUNK_SIZE):
            for y in range(compute.CHUNK_SIZE):
                for x in range(compute.CHUNK_SIZE):
                    dist = np.sqrt((z - center)**2 + (y - center)**2 + (x - center)**2)
                    if dist <= radius:
                        volume[z, y, x] = 255.0
        
        # Print information about the created volume
        print(f"Created volume with shape {volume.shape}")
        print(f"Volume min: {volume.min().item()}, max: {volume.max().item()}")
        print(f"Value at center: {volume[center, center, center].item()}")
        
        # Check values at different points in the volume (center, edge, outside)
        print(f"Value at [center, center, center] = {volume[center, center, center].item()}")
        print(f"Value at [0, 0, 0] = {volume[0, 0, 0].item()}")
        print(f"Value at [center+radius-1, center, center] = {volume[min(volume.shape[0]-1, center+radius-1), center, center].item()}")
        print(f"Value at [center+radius+1, center, center] = {volume[min(volume.shape[0]-1, center+radius+1), center, center].item()}")
        
        # Process the volume directly first to make a distance field
        print("\nProcessing volume through distance transform...")
        processed_volume = compute.compute(volume.clone())
        print(f"Processed volume shape: {processed_volume.shape}")
        print(f"Processed min: {processed_volume.min().item()}, max: {processed_volume.max().item()}")
        
        # Check values at different points in the processed volume
        print(f"Processed value at [center, center, center] = {processed_volume[center, center, center].item()}")
        print(f"Processed value at [0, 0, 0] = {processed_volume[0, 0, 0].item()}")
        print(f"Processed value at [center+radius-1, center, center] = {processed_volume[min(processed_volume.shape[0]-1, center+radius-1), center, center].item()}")
        print(f"Processed value at [center+radius+1, center, center] = {processed_volume[min(processed_volume.shape[0]-1, center+radius+1), center, center].item()}")
        
        # For the distance transform behavior after our fix:
        # - 0.0 should be for points where intensity >= threshold (inside object)
        # - Values > 0 are for points with intensity < threshold (outside object)
        center_point_value = processed_volume[center, center, center].item()
        outside_point_value = processed_volume[0, 0, 0].item()
        
        print(f"\nExpected behavior check:")
        if volume[center, center, center].item() >= compute.TH:
            print(f"Center is above threshold, should be 0.0 in processed volume (actual: {center_point_value})")
            expected_center_value = 0.0
        else:
            print(f"Center is below threshold, should have distance > 0 (actual: {center_point_value})")
            expected_center_value = center_point_value
                
        # Create chunked tensor and interpolator with precomputed=True
        print("\nCreating ChunkedTensor and Interpolator...")
        chunked = ChunkedTensor(compute, processed_volume, precomputed=True)
        interp = CachedChunked3dInterpolator(chunked)
        
        # Test evaluate at center
        print(f"\nEvaluating at center ({center}, {center}, {center})...")
        value = interp.evaluate(float(center), float(center), float(center))
        print(f"Interpolated value at center: {value}")
        
        # Check the actual point value directly against the processed volume
        print(f"Value directly from processed volume: {processed_volume[center, center, center].item()}")
        
        # Verify that the interpolator properly accesses the processed value
        self.assertEqual(value, center_point_value)
        
        # After our fix, we expect center to be 0.0 (inside object where intensity >= threshold)
        if volume[center, center, center].item() >= compute.TH:
            self.assertEqual(center_point_value, 0.0)
        
        # Test evaluate_with_gradient
        print("\nTesting evaluate_with_gradient...")
        interp.clear_cache()
        value, gradient = interp.evaluate_with_gradient(float(center), float(center), float(center))
        print(f"Value: {value}, Gradient: {gradient}")
        
        # Same value should be returned
        self.assertEqual(value, center_point_value)
        
        # Check that gradient is not None and has correct shape
        self.assertIsNotNone(gradient)
        self.assertIsInstance(gradient, np.ndarray)
        self.assertEqual(gradient.shape, (3,))
        
        # Test second evaluation for cache hit
        print("\nTesting cache hit...")
        value2 = interp.evaluate(float(center), float(center), float(center))
        print(f"Second evaluation value: {value2}")
        self.assertEqual(value2, value)
        
        # Check cache stats
        stats = interp.get_cache_stats()
        print(f"Cache stats: {stats}")
        self.assertEqual(stats["hits"], 1)
        
        # Test clear_cache
        interp.clear_cache()
        stats = interp.get_cache_stats()
        print(f"Cache stats after clear: {stats}")
        self.assertEqual(stats["hits"], 0)
        
        print("----- test_cached_interpolator completed -----")


if __name__ == '__main__':
    unittest.main()