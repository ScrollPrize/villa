"""SurfaceMeta class for handling surface metadata and operations."""

import os
import json
import random
import pathlib
import numpy as np
from typing import Dict, List, Optional, Set, Union, Any, Tuple

from .quad_surface import QuadSurface, load_quad_from_tifxyz


def contains(surface: 'SurfaceMeta', point: np.ndarray, search_effort: int = 10) -> bool:
    """
    Check if a point is contained in a surface.
    
    This function checks if a point is close to any point on the surface.
    It uses random sampling of the surface to check for proximity.
    
    Args:
        surface: SurfaceMeta object
        point: 3D point to check (numpy array)
        search_effort: Number of random points to check (higher = more thorough)
        
    Returns:
        True if the point is contained in the surface
    """
    # Try to load the surface if not already loaded
    surf = surface.surf()
    if surf is None:
        return False
        
    # Get surface points
    points = surf.raw_points()
    h, w = points.shape[:2]
    
    # Get threshold based on surface scale
    threshold = 5.0  # Default threshold in voxels
    
    # Get bounding box
    bbox = surf.bbox()
    if bbox.low[0] == -1:
        return False  # Invalid bounding box
        
    # Quick check if point is within expanded bounding box
    margin = threshold * 2.0
    if (point[0] < bbox.low[0] - margin or point[0] > bbox.high[0] + margin or
        point[1] < bbox.low[1] - margin or point[1] > bbox.high[1] + margin or
        point[2] < bbox.low[2] - margin or point[2] > bbox.high[2] + margin):
        return False
        
    # Try random points for proximity check
    for _ in range(search_effort):
        y = random.randint(0, h-1)
        x = random.randint(0, w-1)
        
        if points[y, x, 0] != -1:  # Valid point
            # Calculate distance
            dist = np.linalg.norm(points[y, x] - point)
            if dist < threshold:
                return True
                
    # Create surface pointer and try more precise method
    ptr = surf.pointer()
    dist = surf.point_to(ptr, point, threshold * 2.0)
    
    return dist >= 0 and dist < threshold


def overlap(surface1: 'SurfaceMeta', surface2: 'SurfaceMeta', search_effort: int = 10) -> bool:
    """
    Check if two surfaces overlap.
    
    This function checks if any point from surface1 is contained in surface2 
    or vice versa.
    
    Args:
        surface1: First SurfaceMeta object
        surface2: Second SurfaceMeta object
        search_effort: Number of random points to check (higher = more thorough)
        
    Returns:
        True if the surfaces overlap
    """
    # Quick name-based check
    if surface1.is_overlapping(surface2) or surface2.is_overlapping(surface1):
        return True
        
    # Try to load surfaces if not already loaded
    surf1 = surface1.surf()
    surf2 = surface2.surf()
    if surf1 is None or surf2 is None:
        return False
        
    # Get bounding boxes
    bbox1 = surf1.bbox()
    bbox2 = surf2.bbox()
    if bbox1.low[0] == -1 or bbox2.low[0] == -1:
        return False  # Invalid bounding box
        
    # Quick check if bounding boxes overlap
    margin = 5.0  # Default margin in voxels
    if (bbox1.low[0] > bbox2.high[0] + margin or bbox1.high[0] < bbox2.low[0] - margin or
        bbox1.low[1] > bbox2.high[1] + margin or bbox1.high[1] < bbox2.low[1] - margin or
        bbox1.low[2] > bbox2.high[2] + margin or bbox1.high[2] < bbox2.low[2] - margin):
        return False
        
    # Get surface points
    points1 = surf1.raw_points()
    h1, w1 = points1.shape[:2]
    
    # Try random points from surface1 in surface2
    for _ in range(search_effort):
        y = random.randint(0, h1-1)
        x = random.randint(0, w1-1)
        
        if points1[y, x, 0] != -1:  # Valid point
            if contains(surface2, points1[y, x], search_effort):
                return True
                
    # Try the reverse: points from surface2 in surface1
    points2 = surf2.raw_points()
    h2, w2 = points2.shape[:2]
    
    for _ in range(search_effort):
        y = random.randint(0, h2-1)
        x = random.randint(0, w2-1)
        
        if points2[y, x, 0] != -1:  # Valid point
            if contains(surface1, points2[y, x], search_effort):
                return True
                
    return False


class SurfaceMeta:
    """
    Class for handling surface metadata and operations.
    
    SurfaceMeta is used to manage surface metadata, paths, and overlapping
    surface detection. It can load surfaces on demand and extract surface
    names from paths.
    """
    
    def __init__(self, path: Optional[Union[str, pathlib.Path]] = None, 
                 json_data: Optional[Dict[str, Any]] = None):
        """
        Initialize a SurfaceMeta object.
        
        Args:
            path: Path to the surface directory
            json_data: Optional metadata as a dictionary
        """
        self.path = pathlib.Path(path) if path else None
        self.meta = json_data or {}
        self.overlapping: Set[SurfaceMeta] = set()
        self.overlapping_str: Set[str] = set()
        self._surf = None
        
    def read_overlapping(self) -> None:
        """Read overlapping information from directory."""
        if not self.path:
            return
            
        # Check if overlapping directory exists
        overlap_dir = self.path / "overlapping"
        if not overlap_dir.exists():
            return
            
        # Read overlapping surface names
        for item in overlap_dir.iterdir():
            if item.is_file():
                self.overlapping_str.add(item.name)
                
    def surf(self) -> QuadSurface:
        """
        Return or load the surface.
        
        Returns:
            QuadSurface object
        """
        if self._surf is None and self.path:
            self._surf = load_quad_from_tifxyz(self.path)
        return self._surf
    
    def name(self) -> str:
        """
        Get surface name from path.
        
        Returns:
            Surface name
        """
        if not self.path:
            return ""
            
        return self.path.name
    
    def write_overlapping(self, other_meta: 'SurfaceMeta') -> None:
        """
        Write overlapping information for another surface.
        
        Args:
            other_meta: Other SurfaceMeta object
        """
        if not self.path or not other_meta.path:
            return
            
        # Create overlapping directory if it doesn't exist
        overlap_dir = self.path / "overlapping"
        os.makedirs(overlap_dir, exist_ok=True)
        
        # Create an empty file with the name of the other surface
        other_name = other_meta.name()
        if other_name:
            with open(overlap_dir / other_name, 'w') as f:
                pass
                
        # Add to overlapping sets
        self.overlapping.add(other_meta)
        self.overlapping_str.add(other_name)
        
    def update_meta(self, update_data: Dict[str, Any]) -> None:
        """
        Update metadata with new data.
        
        Args:
            update_data: New metadata to merge in
        """
        # Load existing metadata from file if available
        if self.path and (self.path / "meta.json").exists():
            try:
                with open(self.path / "meta.json", 'r') as f:
                    file_meta = json.load(f)
                    # Update in-memory metadata with file metadata first
                    self.meta.update(file_meta)
            except (json.JSONDecodeError, IOError):
                # If there's an error reading the file, continue with current metadata
                pass
                
        # Update with the new data
        self.meta.update(update_data)
        
        # If we have a loaded surface, update its metadata too
        if self._surf:
            self._surf.meta.update(update_data)
            
        # Save metadata if path exists
        if self.path:
            meta_path = self.path / "meta.json"
            with open(meta_path, 'w') as f:
                json.dump(self.meta, f, indent=4)
                
    @staticmethod
    def load_from_directory(directory: Union[str, pathlib.Path]) -> Dict[str, 'SurfaceMeta']:
        """
        Load all surfaces from a directory as SurfaceMeta objects.
        
        Args:
            directory: Directory containing surface folders
            
        Returns:
            Dictionary of surface name to SurfaceMeta object
        """
        directory = pathlib.Path(directory)
        result = {}
        
        # Find all directories with meta.json files
        for item in directory.iterdir():
            if item.is_dir() and (item / "meta.json").exists():
                meta = SurfaceMeta(item)
                # Load metadata
                with open(item / "meta.json", 'r') as f:
                    meta.meta = json.load(f)
                # Read overlapping information
                meta.read_overlapping()
                # Add to result
                result[meta.name()] = meta
                
        # Connect overlapping references
        for name, meta in result.items():
            for overlap_name in meta.overlapping_str:
                if overlap_name in result:
                    meta.overlapping.add(result[overlap_name])
                    
        return result
    
    def is_overlapping(self, other: 'SurfaceMeta') -> bool:
        """
        Check if this surface overlaps with another.
        
        Args:
            other: Other SurfaceMeta object
            
        Returns:
            True if surfaces overlap
        """
        if other.name() in self.overlapping_str:
            return True
            
        return False
    
    def ensure_overlapping(self, other: 'SurfaceMeta') -> None:
        """
        Ensure that overlapping information is recorded for both surfaces.
        
        Args:
            other: Other SurfaceMeta object
        """
        if not self.is_overlapping(other):
            self.write_overlapping(other)
            
        if not other.is_overlapping(self):
            other.write_overlapping(self)
            
    def set_surf(self, surface: QuadSurface) -> None:
        """
        Set the surface object.
        
        Args:
            surface: QuadSurface object
        """
        self._surf = surface