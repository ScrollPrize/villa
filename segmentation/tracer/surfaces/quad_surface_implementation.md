# QuadSurface Python Implementation Specification

This document outlines the specification for porting the C++ `QuadSurface` class to Python, including key functionality, methods, and expected behavior.

## Overview

The `QuadSurface` class is a fundamental data structure in Volume Cartographer that represents a 3D surface as a grid of 3D points. It serves as a primary output format of segmentation and surface tracing algorithms. In Python, we'll implement this class to match the C++ version's behavior while leveraging Numpy and PyTorch for efficient operations.

## Class Definition

```python
class QuadSurface:
    """
    Represents a 3D surface as a grid of 3D points.
    
    The QuadSurface supports operations like saving/loading, coordinate
    transformations, and bounding box calculations.
    """
```

## Core Properties

| Property | Type | Description |
|----------|------|-------------|
| `_points` | `np.ndarray` or `torch.Tensor` | 2D grid of 3D points (shape: [rows, cols, 3]) |
| `_scale` | `tuple` or `np.ndarray` | Scale factors for x and y dimensions (2-element tuple) |
| `_center` | `np.ndarray` | Center point of the surface (for internal/nominal coordinate conversion) |
| `_bounds` | `tuple` | Rectangle defining valid boundaries of the surface (x, y, width, height) |
| `_bbox` | `dict` | 3D bounding box (`{'low': [x,y,z], 'high': [x,y,z]}`) |
| `meta` | `dict` | Metadata associated with the surface |
| `path` | `str` or `pathlib.Path` | Path where the surface is stored |

## Key Methods

### Constructor

```python
def __init__(self, points=None, scale=(1.0, 1.0)):
    """
    Initialize a QuadSurface with points and scale.
    
    Args:
        points: 2D grid of 3D points (numpy array or torch tensor)
        scale: Scale factors for x and y dimensions (default: (1.0, 1.0))
    """
```

### Core Functionality

```python
def pointer(self):
    """Return a new surface pointer (equivalent to TrivialSurfacePointer)."""

def move(self, ptr, offset):
    """Move the pointer by the specified offset."""

def valid(self, ptr, offset=(0,0,0)):
    """Check if location is valid on the surface."""

def loc(self, ptr, offset=(0,0,0)):
    """Get nominal location from pointer and offset."""
    
def loc_raw(self, ptr):
    """Get raw internal location from pointer."""

def coord(self, ptr, offset=(0,0,0)):
    """Convert surface location to 3D coordinate."""

def normal(self, ptr, offset=(0,0,0)):
    """Calculate surface normal at the specified location."""

def point_to(self, ptr, target, threshold, max_iters=1000):
    """Find the closest point on the surface to the target point."""
```

### I/O Methods

```python
def save(self, path, uuid=None):
    """
    Save the surface to the specified path.
    
    Args:
        path: Directory path to save the surface
        uuid: Optional UUID to use for the surface
    """

def save_meta(self):
    """Save metadata to the surface path."""

def bbox(self):
    """Calculate the 3D bounding box of the surface."""

def raw_points(self):
    """Return the raw points grid."""

def set_raw_points(self, points):
    """Set the raw points grid."""
```

## Utility Functions

These functions should be implemented to support QuadSurface operations:

1. `internal_loc(nominal, internal, scale)`: Convert between nominal and internal coordinates
2. `nominal_loc(nominal, internal, scale)`: Convert between internal and nominal coordinates
3. `at_int(points, loc)`: Interpolate a 3D point at the specified location
4. `grid_normal(points, loc)`: Calculate surface normal at the specified location
5. `search_min_loc(points, loc, out, target, step, min_step)`: Find the closest point on the surface to a target

## Expected Behavior Verification

To verify the Python implementation matches the C++ version, we should test the following behaviors:

### 1. Coordinate System Transformations

The C++ QuadSurface operates with three coordinate systems:
- **Nominal coordinates**: Voxel volume coordinates
- **Internal relative coordinates**: Pointer coordinates where center is at (0,0)
- **Internal absolute coordinates**: Matrix coordinates where upper left corner is (0,0)

Verify that transformations between these coordinate systems match the C++ behavior:

```python
# Test surface with known matrix and transformations
test_points = np.zeros((10, 10, 3))
test_scale = (2.0, 2.0)
surface = QuadSurface(test_points, test_scale)

# Test pointer movement
ptr = surface.pointer()
surface.move(ptr, (1, 1, 0))

# Test coordinate conversion
coord = surface.coord(ptr)
loc = surface.loc(ptr)
raw_loc = surface.loc_raw(ptr)

# Expected: loc and coord should match C++ behavior with same inputs
```

### 2. Interpolation

Verify that point interpolation (at_int) behaves like the C++ version:

```python
# Create test surface with gradient values
test_points = create_gradient_surface(10, 10)
surface = QuadSurface(test_points, (1.0, 1.0))

# Interpolate at specific locations (integer and fractional)
int_point = surface.coord(surface.pointer(), (5, 5, 0))
frac_point = surface.coord(surface.pointer(), (5.5, 5.5, 0))

# Expected: point values should match bilinear interpolation in C++
```

### 3. Save and Load

Ensure that saving and loading produces identical files to the C++ version:

```python
# Create test surface
test_points = create_test_surface()
surface = QuadSurface(test_points, (1.0, 1.0))

# Add metadata
surface.meta = {
    "type": "seg",
    "uuid": "test_surface",
    "format": "tifxyz"
}

# Save surface
surface.save("/tmp/test_surface", "test_surface")

# Expected: 
# - x.tif, y.tif, z.tif files should be created
# - meta.json file should match C++ format
# - Points should be preserved accurately
```

### 4. Bounding Box Calculation

Verify that bounding box calculations match the C++ implementation:

```python
# Create test surface with known bounds
test_points = create_surface_with_bounds()
surface = QuadSurface(test_points, (1.0, 1.0))

# Calculate bounding box
bbox = surface.bbox()

# Expected: bbox should match C++ calculation for same input
```

### 5. Point Finding (point_to)

Verify that the `point_to` method finds the closest point on the surface to a target point:

```python
# Create test surface
test_points = create_test_surface()
surface = QuadSurface(test_points, (1.0, 1.0))

# Find point on surface closest to target
ptr = surface.pointer()
distance = surface.point_to(ptr, (10, 10, 10), 1.0, 1000)

# Expected: 
# - ptr should be updated to closest point
# - distance should match C++ calculation
```

## Implementation Strategy

1. Start with basic class structure and properties
2. Implement core utility functions (conversions, interpolation)
3. Add constructor and initialization logic
4. Implement coordinate system methods
5. Add I/O functionality
6. Implement bounding box calculation
7. Add point finding and interpolation logic

## Dependencies

- NumPy: For array operations
- PIL/Pillow: For image I/O (TIFF handling)
- PyTorch (optional): For GPU acceleration and integration with Theseus
- pathlib: For path handling
- json: For metadata handling

## Format Compatibility Notes

- The tifxyz format consists of three TIFF files (x.tif, y.tif, z.tif) representing the x, y, and z coordinates
- Invalid points are marked as (-1, -1, -1)
- The meta.json file contains various metadata including:
  - bbox: 3D bounding box
  - type: Surface type (usually "seg")
  - uuid: Unique identifier
  - format: "tifxyz"
  - scale: Scale factors (x, y)
  - Additional metadata depending on the source

## Performance Considerations

1. Use vectorized operations for coordinate transformations
2. Consider PyTorch tensors for GPU acceleration
3. Optimize interpolation for batch operations
4. Use efficient I/O methods for large surfaces
5. Consider memory mapping for very large surfaces