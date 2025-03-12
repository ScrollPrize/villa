"""Tests for the SurfaceMeta class."""

import os
import json
import tempfile
import shutil
import pathlib
import numpy as np
import pytest

from surfaces.quad_surface import QuadSurface
from surfaces.surface_meta import SurfaceMeta


def create_test_surface(directory, name, points=None, scale=(1.0, 1.0)):
    """Create a test surface in the given directory."""
    if points is None:
        # Create a simple 5x5 surface with all valid points
        points = np.zeros((5, 5, 3), dtype=np.float32)
        for j in range(5):
            for i in range(5):
                points[j, i] = [i, j, 1.0]  # z=1 makes points valid
    
    surf = QuadSurface(points, scale)
    surf_dir = os.path.join(directory, name)
    os.makedirs(surf_dir, exist_ok=True)
    surf.save(surf_dir, name)
    return surf_dir


class TestSurfaceMeta:
    """Test cases for the SurfaceMeta class."""
    
    def setup_method(self):
        """Setup test data."""
        # Create a temporary directory for test surfaces
        self.temp_dir = tempfile.mkdtemp()
        
        # Create two test surfaces
        self.surf1_path = create_test_surface(self.temp_dir, "surface1")
        self.surf2_path = create_test_surface(self.temp_dir, "surface2")
        
    def teardown_method(self):
        """Clean up test data."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        """Test initialization of SurfaceMeta."""
        # Test with path
        meta1 = SurfaceMeta(self.surf1_path)
        assert meta1.path == pathlib.Path(self.surf1_path)
        assert meta1.meta == {}  # Empty metadata
        assert len(meta1.overlapping) == 0
        assert len(meta1.overlapping_str) == 0
        
        # Test with path and json_data
        test_data = {"key": "value"}
        meta2 = SurfaceMeta(self.surf1_path, test_data)
        assert meta2.path == pathlib.Path(self.surf1_path)
        assert meta2.meta == test_data
    
    def test_name(self):
        """Test name() method."""
        meta = SurfaceMeta(self.surf1_path)
        assert meta.name() == "surface1"
        
        # Test with no path
        meta = SurfaceMeta()
        assert meta.name() == ""
    
    def test_surf(self):
        """Test surf() method."""
        meta = SurfaceMeta(self.surf1_path)
        
        # Load the surface
        surf = meta.surf()
        assert surf is not None
        assert isinstance(surf, QuadSurface)
        
        # Check that it's cached
        assert meta._surf is surf
        assert meta.surf() is surf  # Same instance
    
    def test_write_overlapping(self):
        """Test write_overlapping() method."""
        meta1 = SurfaceMeta(self.surf1_path)
        meta2 = SurfaceMeta(self.surf2_path)
        
        # Write overlapping information
        meta1.write_overlapping(meta2)
        
        # Check sets
        assert meta2.name() in meta1.overlapping_str
        assert meta2 in meta1.overlapping
        
        # Check directory
        overlap_dir = os.path.join(self.surf1_path, "overlapping")
        assert os.path.exists(overlap_dir)
        assert os.path.exists(os.path.join(overlap_dir, "surface2"))
    
    def test_read_overlapping(self):
        """Test read_overlapping() method."""
        meta1 = SurfaceMeta(self.surf1_path)
        meta2 = SurfaceMeta(self.surf2_path)
        
        # Write overlapping information
        meta1.write_overlapping(meta2)
        
        # Create a new meta object to read the information
        meta1_new = SurfaceMeta(self.surf1_path)
        meta1_new.read_overlapping()
        
        # Check that it read the information
        assert meta2.name() in meta1_new.overlapping_str
    
    def test_update_meta(self):
        """Test update_meta() method."""
        meta = SurfaceMeta(self.surf1_path)
        
        # Ensure metadata exists in file
        with open(os.path.join(self.surf1_path, "meta.json"), 'r') as f:
            original_meta = json.load(f)
        
        # Update metadata
        update_data = {"test_key": "test_value"}
        meta.update_meta(update_data)
        
        # Check in-memory metadata
        assert meta.meta["test_key"] == "test_value"
        
        # Check file metadata
        with open(os.path.join(self.surf1_path, "meta.json"), 'r') as f:
            file_meta = json.load(f)
        assert file_meta["test_key"] == "test_value"
        
        # Check that original metadata is preserved
        for key, value in original_meta.items():
            assert key in file_meta
    
    def test_load_from_directory(self):
        """Test load_from_directory() method."""
        # Create a third surface with overlapping info
        surf3_path = create_test_surface(self.temp_dir, "surface3")
        meta1 = SurfaceMeta(self.surf1_path)
        meta3 = SurfaceMeta(surf3_path)
        meta1.write_overlapping(meta3)
        meta3.write_overlapping(meta1)
        
        # Load from directory
        surfaces = SurfaceMeta.load_from_directory(self.temp_dir)
        
        # Check loaded surfaces
        assert len(surfaces) == 3
        assert "surface1" in surfaces
        assert "surface2" in surfaces
        assert "surface3" in surfaces
        
        # Check overlapping connections
        assert surfaces["surface3"] in surfaces["surface1"].overlapping
        assert surfaces["surface1"] in surfaces["surface3"].overlapping
    
    def test_is_overlapping(self):
        """Test is_overlapping() method."""
        meta1 = SurfaceMeta(self.surf1_path)
        meta2 = SurfaceMeta(self.surf2_path)
        
        # Initially not overlapping
        assert not meta1.is_overlapping(meta2)
        assert not meta2.is_overlapping(meta1)
        
        # Set overlapping
        meta1.write_overlapping(meta2)
        
        # Check overlapping
        assert meta1.is_overlapping(meta2)
        assert not meta2.is_overlapping(meta1)  # One-way relationship
    
    def test_ensure_overlapping(self):
        """Test ensure_overlapping() method."""
        meta1 = SurfaceMeta(self.surf1_path)
        meta2 = SurfaceMeta(self.surf2_path)
        
        # Ensure overlapping
        meta1.ensure_overlapping(meta2)
        
        # Check bidirectional relationship
        assert meta1.is_overlapping(meta2)
        assert meta2.is_overlapping(meta1)