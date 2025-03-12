"""QuadSurface class for representing 3D surfaces as a grid of points."""

import os
import json
import numpy as np
import pathlib
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image


class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that can handle NumPy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class SurfacePointer:
    """Base class for surface pointers."""

    def clone(self):
        """Create a copy of this pointer."""
        raise NotImplementedError()


class TrivialSurfacePointer(SurfacePointer):
    """Simple surface pointer implementation."""

    def __init__(self, loc: np.ndarray):
        """Initialize pointer with location."""
        self.loc = np.array(loc, dtype=np.float32)

    def clone(self):
        """Create a copy of this pointer."""
        return TrivialSurfacePointer(self.loc.copy())


class Rect3D:
    """3D bounding box representation."""

    def __init__(self, low=None, high=None):
        """Initialize with low and high points."""
        self.low = np.array(low if low is not None else [0, 0, 0], dtype=np.float32)
        self.high = np.array(high if high is not None else [0, 0, 0], dtype=np.float32)


def expand_rect(rect: Rect3D, point: np.ndarray) -> Rect3D:
    """Expand rectangle to include the given point."""
    if point[0] == -1:  # Invalid point
        return rect
        
    new_rect = Rect3D()
    new_rect.low = np.minimum(rect.low, point)
    new_rect.high = np.maximum(rect.high, point)
    return new_rect


def at_int(points: np.ndarray, loc: np.ndarray) -> np.ndarray:
    """
    Interpolate point at the given location using bilinear interpolation.
    
    Args:
        points: Points grid [height, width, 3] where each point is in ZYX order
        loc: Location [y, x] on the grid - Follows YX ordering for 2D positions
        
    Returns:
        Interpolated 3D point in ZYX order (z, y, x)
    """
    x, y = loc[0], loc[1]  # Note: points array is indexed as [y, x], but stores ZYX coordinates
    ix, iy = int(x), int(y)
    fx, fy = x - ix, y - iy
    
    # Check bounds
    if ix < 0 or iy < 0 or ix >= points.shape[1]-1 or iy >= points.shape[0]-1:
        return np.array([-1, -1, -1], dtype=np.float32)
    
    # Get corner points (all points are in ZYX order)
    p00 = points[iy, ix]      # Point at (iy, ix) in ZYX order
    p01 = points[iy, ix+1]    # Point at (iy, ix+1) in ZYX order
    p10 = points[iy+1, ix]    # Point at (iy+1, ix) in ZYX order
    p11 = points[iy+1, ix+1]  # Point at (iy+1, ix+1) in ZYX order
    
    # Check if any point is invalid
    if p00[0] == -1 or p01[0] == -1 or p10[0] == -1 or p11[0] == -1:
        return np.array([-1, -1, -1], dtype=np.float32)
    
    # Interpolate (preserving ZYX ordering)
    p0 = (1 - fx) * p00 + fx * p01
    p1 = (1 - fx) * p10 + fx * p11
    return (1 - fy) * p0 + fy * p1


def grid_normal(points: np.ndarray, loc: np.ndarray) -> np.ndarray:
    """
    Calculate surface normal at the given location.
    
    Args:
        points: Points grid [height, width, 3] where each point is in ZYX order
        loc: Location [x, y] on the grid
        
    Returns:
        Surface normal vector in ZYX order
    """
    x, y = loc[0], loc[1]
    ix, iy = int(x), int(y)
    
    # Check bounds for surrounding points
    if ix < 1 or iy < 1 or ix >= points.shape[1]-1 or iy >= points.shape[0]-1:
        return np.array([0, 0, 1], dtype=np.float32)  # Default normal in ZYX order (z=0, y=0, x=1)
    
    # Get surrounding points for computing tangent vectors (all in ZYX order)
    dx = points[iy, ix+1] - points[iy, ix-1]  # Tangent in X direction
    dy = points[iy+1, ix] - points[iy-1, ix]  # Tangent in Y direction
    
    # Check if points are valid
    if dx[0] == -1 or dy[0] == -1:
        return np.array([0, 0, 1], dtype=np.float32)  # Default normal in ZYX order
    
    # Compute normal as cross product of tangent vectors
    # Since points are in ZYX order, the normal will also be in ZYX order
    normal = np.cross(dx, dy)
    norm = np.linalg.norm(normal)
    
    if norm > 1e-10:
        return normal / norm
    else:
        return np.array([0, 0, 1], dtype=np.float32)  # Default normal in ZYX order


def internal_loc(nominal: np.ndarray, internal: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """
    Convert from nominal to internal absolute coordinates.
    
    Args:
        nominal: Nominal coordinates in ZYX order
        internal: Internal relative coordinates in ZYX order
        scale: Scale factors [x, y]
        
    Returns:
        Internal absolute coordinates in ZYX order
    """
    # Note: nominal and internal are in ZYX order [z, y, x]
    # Scale applies to x and y coordinates (at indices 2 and 1 respectively)
    return internal + np.array([nominal[0], nominal[1] * scale[1], nominal[2] * scale[0]], dtype=np.float32)


def nominal_loc(nominal: np.ndarray, internal: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """
    Convert from internal to nominal coordinates.
    
    Args:
        nominal: Nominal coordinates in ZYX order
        internal: Internal relative coordinates in ZYX order
        scale: Scale factors [x, y]
        
    Returns:
        Nominal coordinates in ZYX order
    """
    # Note: nominal and internal are in ZYX order [z, y, x]
    # Scale applies to x and y coordinates (at indices 2 and 1 respectively)
    return nominal + np.array([internal[0], internal[1] / scale[1], internal[2] / scale[0]], dtype=np.float32)


def search_min_loc(points: np.ndarray, loc: np.ndarray, out: np.ndarray, 
                  target: np.ndarray, init_step: np.ndarray, min_step_x: float) -> float:
    """
    Find the closest point on the surface to the target point.
    
    Args:
        points: Points grid [height, width, 3] where each point is in ZYX order
        loc: Initial location (will be updated) [x, y] on the grid
        out: Output point (will be updated) [z, y, x] in ZYX order
        target: Target point [z, y, x] in ZYX order
        init_step: Initial step size [x, y]
        min_step_x: Minimum step size for x
        
    Returns:
        Distance to target
    """
    boundary = (1, 1, points.shape[1]-2, points.shape[0]-2)
    if loc[0] < boundary[0] or loc[1] < boundary[1] or loc[0] >= boundary[2] or loc[1] >= boundary[3]:
        out[:] = np.array([-1, -1, -1], dtype=np.float32)
        return -1
    
    # Initialize
    changed = True
    val = at_int(points, loc)
    out[:] = val
    best = np.sum((val - target)**2)  # Squared distance
    
    # Define search pattern
    search = np.array([
        [0, -1], [0, 1], [-1, -1], [-1, 0], 
        [-1, 1], [1, -1], [1, 0], [1, 1]
    ], dtype=np.float32)
    
    step = np.array(init_step, dtype=np.float32)
    
    # Iterative search
    while changed:
        changed = False
        
        for off in search:
            cand = loc + off * step
            
            # Skip if out of bounds
            if (cand[0] < boundary[0] or cand[1] < boundary[1] or 
                cand[0] >= boundary[2] or cand[1] >= boundary[3]):
                continue
            
            val = at_int(points, cand)
            if val[0] == -1:  # Invalid point
                continue
                
            res = np.sum((val - target)**2)  # Squared distance
            
            if res < best:
                changed = True
                best = res
                loc[:] = cand
                out[:] = val
        
        if changed:
            continue
        
        # Reduce step size
        step *= 0.5
        changed = True
        
        if step[0] < min_step_x:
            break
    
    return np.sqrt(best)  # Return distance


class QuadSurface:
    """
    Represents a 3D surface as a grid of 3D points.
    
    The QuadSurface supports operations like saving/loading, coordinate
    transformations, and bounding box calculations.
    
    Important coordinate conventions:
    - All 3D points are stored in ZYX order [z, y, x]
    - The grid is indexed as [y, x] for 2D access
    - All surface operations preserve the ZYX ordering
    - Default normals point in the +X direction [0, 0, 1] in ZYX coordinates
    """
    
    def __init__(self, points=None, scale=(1.0, 1.0)):
        """
        Initialize a QuadSurface with points and scale.
        
        Args:
            points: 2D grid of 3D points (numpy array or torch tensor) in ZYX order
                   Shape is [height, width, 3] where each point is [z, y, x]
            scale: Scale factors for x and y dimensions (default: (1.0, 1.0))
                  Used for coordinate transformations
        """
        if points is None:
            self._points = np.zeros((0, 0, 3), dtype=np.float32)
        elif isinstance(points, np.ndarray):
            self._points = points.copy().astype(np.float32)
        else:
            self._points = np.array(points, dtype=np.float32)
        
        self._scale = np.array(scale, dtype=np.float32)
        self._bounds = (0, 0, self._points.shape[1]-1, self._points.shape[0]-1)
        
        # Center is in the middle of the grid in nominal coordinates
        self._center = np.array([
            self._points.shape[1] / 2.0 / self._scale[0],
            self._points.shape[0] / 2.0 / self._scale[1],
            0
        ], dtype=np.float32)
        
        self._bbox = Rect3D([-1, -1, -1], [-1, -1, -1])
        self.meta = {}  # Metadata dictionary
        self.path = None  # Path where the surface is stored
        
    def get_points(self):
        """Get all points as a flattened array.
        
        Returns:
            Numpy array of all valid points, shape (N, 3)
        """
        # Flatten the grid of points and remove invalid points (-1 coordinate)
        valid_mask = self._points[:, :, 0] != -1
        return self._points[valid_mask]
        
    def width(self):
        """Get width of the surface.
        
        Returns:
            Width (number of grid columns)
        """
        return self._points.shape[1]
        
    def height(self):
        """Get height of the surface.
        
        Returns:
            Height (number of grid rows)
        """
        return self._points.shape[0]
        
    def pointer(self):
        """Return a new surface pointer."""
        return TrivialSurfacePointer(np.zeros(3, dtype=np.float32))
    
    def move(self, ptr, offset):
        """
        Move the pointer by the specified offset.
        
        Args:
            ptr: Surface pointer
            offset: Offset to move by [z, y, x] following ZYX coordinate ordering
        """
        offset = np.array(offset, dtype=np.float32)
        ptr.loc = ptr.loc + np.array([
            offset[0],  # Z
            offset[1] * self._scale[1],  # Y
            offset[2] * self._scale[0]   # X
        ], dtype=np.float32)
    
    def valid(self, ptr, offset=(0, 0, 0)):
        """
        Check if location is valid on the surface.
        
        Args:
            ptr: Surface pointer
            offset: Additional offset (default: [0, 0, 0])
            
        Returns:
            True if location is valid
        """
        p = internal_loc(np.array(offset, dtype=np.float32) + self._center, 
                         ptr.loc, self._scale)
        
        return (p[0] >= self._bounds[0] and p[1] >= self._bounds[1] and
                p[0] <= self._bounds[2] and p[1] <= self._bounds[3])
    
    def loc(self, ptr, offset=(0, 0, 0)):
        """
        Get nominal location from pointer and offset.
        
        Args:
            ptr: Surface pointer
            offset: Additional offset (default: [0, 0, 0])
            
        Returns:
            Nominal location
        """
        return nominal_loc(np.array(offset, dtype=np.float32), 
                          ptr.loc, self._scale)
    
    def loc_raw(self, ptr):
        """
        Get raw internal location from pointer.
        
        Args:
            ptr: Surface pointer
            
        Returns:
            Internal absolute location
        """
        return internal_loc(self._center, ptr.loc, self._scale)
    
    def coord(self, ptr, offset=(0, 0, 0)):
        """
        Convert surface location to 3D coordinate.
        
        Args:
            ptr: Surface pointer
            offset: Additional offset (default: [0, 0, 0])
            
        Returns:
            3D coordinate
        """
        p = internal_loc(np.array(offset, dtype=np.float32) + self._center, 
                        ptr.loc, self._scale)
        
        # Check bounds
        if (p[0] < 0 or p[1] < 0 or 
            p[0] >= self._points.shape[1]-1 or 
            p[1] >= self._points.shape[0]-1):
            return np.array([-1, -1, -1], dtype=np.float32)
        
        return at_int(self._points, p[:2])
    
    def normal(self, ptr, offset=(0, 0, 0)):
        """
        Calculate surface normal at the specified location.
        
        Args:
            ptr: Surface pointer
            offset: Additional offset (default: [0, 0, 0])
            
        Returns:
            Surface normal vector
        """
        p = internal_loc(np.array(offset, dtype=np.float32) + self._center, 
                        ptr.loc, self._scale)
        
        return grid_normal(self._points, p[:2])
    
    def point_to(self, ptr, target, threshold, max_iters=1000):
        """
        Find the closest point on the surface to the target point.
        
        Args:
            ptr: Surface pointer (will be updated)
            target: Target point [x, y, z]
            threshold: Distance threshold for acceptance
            max_iters: Maximum iterations
            
        Returns:
            Distance to target
        """
        target = np.array(target, dtype=np.float32)
        
        # Setup search steps
        step_small = np.array([
            max(1.0, self._scale[0]),
            max(1.0, self._scale[1])
        ], dtype=np.float32)
        
        min_mul = min(0.1 * self._points.shape[1] / self._scale[0],
                     0.1 * self._points.shape[0] / self._scale[1])
        step_large = np.array([
            min_mul * self._scale[0],
            min_mul * self._scale[1]
        ], dtype=np.float32)
        
        # Initial location from pointer
        loc = self.loc_raw(ptr)[:2].copy()
        out = np.zeros(3, dtype=np.float32)
        
        # First search with small steps
        dist = search_min_loc(self._points, loc, out, target, 
                             step_small, self._scale[0] * 0.1)
        
        # Check if good enough
        if dist >= 0 and dist < threshold:
            ptr.loc = np.array([
                loc[0], loc[1], 0
            ], dtype=np.float32) - np.array([
                self._center[0] * self._scale[0],
                self._center[1] * self._scale[1],
                0
            ], dtype=np.float32)
            return dist
        
        # Store best result so far
        min_loc = loc.copy()
        min_dist = dist
        if min_dist < 0:
            min_dist = 10 * (self._points.shape[1] / self._scale[0] + 
                            self._points.shape[0] / self._scale[1])
        
        # Try random locations
        r_full = 0
        for r in range(10 * max_iters):
            if r_full >= max_iters:
                break
                
            # Random location
            loc = np.array([
                1 + np.random.randint(self._points.shape[1] - 3),
                1 + np.random.randint(self._points.shape[0] - 3)
            ], dtype=np.float32)
            
            # Skip invalid points
            if self._points[int(loc[1]), int(loc[0])][0] == -1:
                continue
                
            r_full += 1
            
            # Search with large steps
            dist = search_min_loc(self._points, loc, out, target, 
                                 step_large, self._scale[0] * 0.1)
            
            # Check if good enough
            if dist >= 0 and dist < threshold:
                # Refine with small steps
                dist = search_min_loc(self._points, loc, out, target, 
                                     step_small, self._scale[0] * 0.1)
                
                ptr.loc = np.array([
                    loc[0], loc[1], 0
                ], dtype=np.float32) - np.array([
                    self._center[0] * self._scale[0],
                    self._center[1] * self._scale[1],
                    0
                ], dtype=np.float32)
                return dist
            elif dist >= 0 and dist < min_dist:
                min_loc = loc.copy()
                min_dist = dist
        
        # Return best result found
        ptr.loc = np.array([
            min_loc[0], min_loc[1], 0
        ], dtype=np.float32) - np.array([
            self._center[0] * self._scale[0],
            self._center[1] * self._scale[1],
            0
        ], dtype=np.float32)
        
        return min_dist
    
    def save(self, path, uuid=None):
        """
        Save the surface to the specified path.
        
        Args:
            path: Directory path to save the surface
            uuid: Optional UUID to use for the surface
        """
        # Convert path to pathlib.Path if it's a string
        if isinstance(path, str):
            path = pathlib.Path(path)
        
        self.path = path
        
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Split points into x, y, z components
        x = self._points[..., 0].astype(np.float32)
        y = self._points[..., 1].astype(np.float32)
        z = self._points[..., 2].astype(np.float32)
        
        # Save as TIFF files
        Image.fromarray(x).save(path / "x.tif")
        Image.fromarray(y).save(path / "y.tif")
        Image.fromarray(z).save(path / "z.tif")
        
        # Save metadata
        if uuid is None and isinstance(path, pathlib.Path):
            if path.name:
                uuid = path.name
            else:
                uuid = path.parent.name
        
        if not self.meta:
            self.meta = {}
        
        # Update metadata
        bbox = self.bbox()
        self.meta.update({
            "bbox": [
                [float(bbox.low[0]), float(bbox.low[1]), float(bbox.low[2])],
                [float(bbox.high[0]), float(bbox.high[1]), float(bbox.high[2])]
            ],
            "type": "seg",
            "uuid": uuid,
            "format": "tifxyz",
            "scale": [float(self._scale[0]), float(self._scale[1])]
        })
        
        # Write metadata to file
        with open(path / "meta.json.tmp", 'w') as f:
            json.dump(self.meta, f, indent=4, cls=NumpyJSONEncoder)
        
        # Rename to make creation atomic
        os.rename(path / "meta.json.tmp", path / "meta.json")
    
    def save_meta(self):
        """Save metadata to the surface path."""
        if not self.meta:
            raise RuntimeError("Can't save_meta() without metadata!")
        
        if not self.path:
            raise RuntimeError("No storage path for QuadSurface")
        
        path = self.path
        if isinstance(path, str):
            path = pathlib.Path(path)
        
        # Write metadata to file
        with open(path / "meta.json.tmp", 'w') as f:
            json.dump(self.meta, f, indent=4, cls=NumpyJSONEncoder)
        
        # Rename to make creation atomic
        os.rename(path / "meta.json.tmp", path / "meta.json")
    
    def bbox(self):
        """
        Calculate the 3D bounding box of the surface.
        
        Returns:
            Rect3D object representing the bounding box
        """
        if self._bbox.low[0] == -1:
            # Initialize with first valid point
            first_valid = None
            for j in range(self._points.shape[0]):
                for i in range(self._points.shape[1]):
                    if self._points[j, i, 0] != -1:
                        first_valid = self._points[j, i]
                        break
                if first_valid is not None:
                    break
            
            if first_valid is None:
                return self._bbox
            
            self._bbox = Rect3D(first_valid, first_valid)
            
            # Expand bounding box with all valid points
            for j in range(self._points.shape[0]):
                for i in range(self._points.shape[1]):
                    if self._points[j, i, 0] != -1:
                        self._bbox = expand_rect(self._bbox, self._points[j, i])
        
        return self._bbox
    
    def raw_points(self):
        """Return the raw points grid."""
        return self._points
    
    def set_raw_points(self, points):
        """Set the raw points grid."""
        self._points = np.array(points, dtype=np.float32)
        self._bounds = (0, 0, self._points.shape[1]-1, self._points.shape[0]-1)
        self._bbox = Rect3D([-1, -1, -1], [-1, -1, -1])  # Reset bounding box


def load_quad_from_tifxyz(path):
    """
    Load a QuadSurface from a tifxyz directory.
    
    Args:
        path: Path to the tifxyz directory
        
    Returns:
        QuadSurface object
    """
    if isinstance(path, str):
        path = pathlib.Path(path)
    
    # Load x, y, z component TIFF files
    from PIL import Image
    x = np.array(Image.open(path / "x.tif"), dtype=np.float32)
    y = np.array(Image.open(path / "y.tif"), dtype=np.float32)
    z = np.array(Image.open(path / "z.tif"), dtype=np.float32)
    
    # Combine into points array - tif files are in XYZ order, but we store as ZYX
    points = np.stack([z, y, x], axis=-1)  # Convert to ZYX ordering
    
    # Load metadata
    with open(path / "meta.json", 'r') as f:
        metadata = json.load(f)
    
    # Get scale
    scale = metadata.get("scale", [1.0, 1.0])
    
    # Mark invalid points
    for j in range(points.shape[0]):
        for i in range(points.shape[1]):
            if points[j, i, 2] <= 0:
                points[j, i] = [-1, -1, -1]
    
    # Handle mask if present
    mask_path = path / "mask.tif"
    if os.path.exists(mask_path):
        mask = np.array(Image.open(mask_path), dtype=np.uint8)
        # Resize mask if needed
        if mask.shape != points.shape[:2]:
            from PIL import Image
            mask = np.array(Image.fromarray(mask).resize(
                (points.shape[1], points.shape[0]), Image.NEAREST))
        
        # Apply mask
        for j in range(points.shape[0]):
            for i in range(points.shape[1]):
                if not mask[j, i]:
                    points[j, i] = [-1, -1, -1]
    
    # Create surface
    surf = QuadSurface(points, scale)
    surf.path = path
    surf.meta = metadata
    
    return surf