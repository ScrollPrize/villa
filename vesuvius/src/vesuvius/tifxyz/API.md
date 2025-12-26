# TifXYZ API Documentation

Python module for reading and writing tifxyz surface files from volume-cartographer.

## Installation

```python
from vesuvius.tifxyz import Tifxyz, read_tifxyz, write_tifxyz
```

## File Format

A tifxyz surface is a directory containing:

```
segment_name/
├── x.tif        # X coordinates (32-bit float)
├── y.tif        # Y coordinates (32-bit float)
├── z.tif        # Z coordinates (32-bit float)
├── meta.json    # Metadata (scale, bbox, uuid)
└── mask.tif     # Optional validity mask
```

## Scale and Resolution

Surfaces are stored at reduced resolution to save space. The `scale` parameter indicates what fraction of full resolution the grid represents:

- **scale = 0.05** means the grid is at 5% of full resolution (20x downsampled)
- **scale = 1.0** means the grid is at full resolution (no downsampling)

Key relationships:
- `nominal_size = stored_size / scale` (e.g., 4215 / 0.05 = 84,300)
- `zoom_factor = 1 / scale` (e.g., 1 / 0.05 = 20x to reach full res)

## Quick Start

```python
from vesuvius.tifxyz import read_tifxyz, write_tifxyz

# Read a surface
surface = read_tifxyz("/path/to/segment")

# Two ways to access coordinates:
print(surface.shape)         # (4215, 4373) - stored resolution
print(surface.nominal_size)  # (84300, 87460) - full resolution

# 1. Direct array access (stored resolution)
x = surface.x[100, 200]      # Fast, no interpolation
y = surface.y[100, 200]
z = surface.z[100, 200]

# 2. Lazy full-resolution access (interpolated on-demand)
x, y, z, valid = surface[2000, 4000]           # Single point
x, y, z, valid = surface[1000:1100, 2000:2100] # 100x100 tile
x, y, z, valid = surface.get_tile(1000, 2000, 100, 100)  # Same as above

# Write a surface
write_tifxyz("/path/to/output", surface, overwrite=True)
```

## Lazy vs Materialized Access

The key design principle: **full-resolution data is computed lazily, never stored**.

| Access Method | Resolution | Memory | Speed |
|--------------|------------|--------|-------|
| `surface.x[i,j]` | Stored (e.g., 4215×4373) | ~220 MB | Instant |
| `surface[i,j]` | Full (e.g., 84300×87460) | Per-tile only | ~0.1ms/point |
| `surface.get_tile(...)` | Full | Per-tile only | ~10ms/1000×1000 |

This means you can work with 88 GB of full-resolution data using only 220 MB of memory.

---

## Class: Tifxyz

Main class representing a tifxyz surface.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `x` | `ndarray[float32]` | X coordinates, shape (H, W) |
| `y` | `ndarray[float32]` | Y coordinates, shape (H, W) |
| `z` | `ndarray[float32]` | Z coordinates, shape (H, W) |
| `uuid` | `str` | Unique identifier |
| `scale` | `tuple[float, float]` | Grid scale (scale_y, scale_x) |
| `bbox` | `tuple` or `None` | Bounding box (x_min, y_min, z_min, x_max, y_max, z_max) |
| `mask` | `ndarray[bool]` or `None` | Validity mask |
| `path` | `Path` or `None` | Source path if loaded from disk |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `shape` | `tuple[int, int]` | Stored grid dimensions (height, width) |
| `nominal_size` | `tuple[int, int]` | Full resolution dimensions = shape / scale |
| `is_full_resolution` | `bool` | True if scale ~= 1.0 |
| `valid_mask` | `ndarray[bool]` | Validity mask (z > 0 and finite) |

### Constructor

```python
surface = Tifxyz(
    x=x_array,           # Required: X coordinates
    y=y_array,           # Required: Y coordinates
    z=z_array,           # Required: Z coordinates
    uuid="my-surface",   # Optional: identifier
    scale=(20.0, 20.0),  # Optional: grid scale
)
```

---

## Coordinate Access

### Lazy Full-Resolution Access (Recommended)

Access coordinates at full resolution using indexing. Data is interpolated on-demand.

```python
# Single point at full resolution
x, y, z, valid = surface[row, col]

# Tile at full resolution
x, y, z, valid = surface[1000:1100, 2000:2100]

# Using get_tile method
x, y, z, valid = surface.get_tile(row=1000, col=2000, height=100, width=100)
```

### Direct Array Access (Stored Resolution)

Access the stored arrays directly for maximum performance.

```python
# Single point at stored resolution
x, y, z = surface.x[row, col], surface.y[row, col], surface.z[row, col]

# Check validity first
if surface.valid_mask[row, col]:
    x, y, z = surface.x[row, col], surface.y[row, col], surface.z[row, col]

# All valid points
valid = surface.valid_mask
x_valid = surface.x[valid]
y_valid = surface.y[valid]
z_valid = surface.z[valid]
```

### get_point_at_grid

Get 3D coordinates at a stored grid location.

```python
x, y, z = surface.get_point_at_grid(grid_y, grid_x)
# Returns (-1, -1, -1) if invalid or out of bounds
```

---

## Coordinate Conversion

### grid_to_nominal

Convert stored grid indices to nominal (voxel) coordinates.

```python
nominal_y, nominal_x = surface.grid_to_nominal(grid_y, grid_x)
# nominal = grid * scale
```

### nominal_to_grid

Convert nominal coordinates to stored grid indices.

```python
grid_y, grid_x = surface.nominal_to_grid(nominal_y, nominal_x)
# grid = nominal / scale
```

### get_nominal_extent

Get the nominal coordinate extent of the grid.

```python
min_y, max_y, min_x, max_x = surface.get_nominal_extent()
```

---

## Materialized Upsampling

For cases where you need the full array in memory (not recommended for large surfaces).

### upsample

Upsample surface to a specific target scale.

```python
# Upsample to 50% of full resolution (10x zoom from 5%)
half_res = surface.upsample(target_scale=0.5, order=1)
print(half_res.shape)  # 10x larger than original
```

### to_full_resolution

Materialize the entire full-resolution surface. **Warning: May use tens of GB of memory.**

```python
# Creates ~88 GB array for a typical surface
full_res = surface.to_full_resolution()
```

### get_points_at_nominal

Interpolate 3D points at arbitrary nominal coordinates.

```python
x, y, z, valid = surface.get_points_at_nominal(
    nominal_y=np.array([100, 200, 300]),
    nominal_x=np.array([150, 250, 350]),
    order=1  # 0=nearest, 1=bilinear, 3=bicubic
)
```

---

## Normals

### get_normals

Compute surface normals lazily for a tile at full resolution.

```python
# Get normals for a 100x100 tile at full resolution
nx, ny, nz = surface.get_normals(row_start=1000, row_end=1100, col_start=2000, col_end=2100)
# Returns NaN for invalid/boundary points
# Shape: (100, 100) each
```

### compute_normals

Compute surface normals for the entire surface at stored resolution.
Used for whole-surface operations like analyzing normal direction.

```python
nx, ny, nz = surface.compute_normals()
# Returns NaN for invalid/boundary points
# Shape matches stored resolution, not full resolution
```

### compute_centroid

Compute centroid of all valid points.

```python
cx, cy, cz = surface.compute_centroid()
```

### analyze_normal_direction

Analyze whether normals point inward (toward centroid) or outward.

```python
analysis = surface.analyze_normal_direction()
# Or with pre-computed normals:
analysis = surface.analyze_normal_direction(normals=(nx, ny, nz))

print(analysis)
# {
#     'centroid': (x, y, z),
#     'direction': 'inward' | 'outward' | 'mixed',
#     'consistent': True | False,  # >95% same direction
#     'inward_fraction': 0.0-1.0,
#     'outward_fraction': 0.0-1.0,
#     'dominant_direction': 'inward' | 'outward',
#     'num_valid_normals': int
# }
```

### orient_normals

Orient all normals to point in a specified direction (flips individual normals as needed).

```python
nx, ny, nz = surface.compute_normals()

# Make all normals point outward
nx, ny, nz = surface.orient_normals((nx, ny, nz), direction='outward')

# Make all normals point inward
nx, ny, nz = surface.orient_normals((nx, ny, nz), direction='inward')
```

### flip_normals

Flip all normals (negate all components).

```python
nx, ny, nz = surface.flip_normals((nx, ny, nz))
# Equivalent to: -nx, -ny, -nz
```

### get_normals_pointing_outward / get_normals_pointing_inward

Convenience methods to compute and orient normals in one call.

```python
# All normals pointing away from centroid
nx, ny, nz = surface.get_normals_pointing_outward()

# All normals pointing toward centroid
nx, ny, nz = surface.get_normals_pointing_inward()
```

---

## Reading & Writing

### read_tifxyz

```python
surface = read_tifxyz(
    path,                   # Path to tifxyz directory
    load_mask=True,         # Load mask.tif if present
    validate=True,          # Validate data after loading
    full_resolution=False,  # If True, upsample to full resolution on load
)

# Load at full resolution directly
full_res = read_tifxyz("/path/to/segment", full_resolution=True)
```

### write_tifxyz

```python
write_tifxyz(
    path,                # Output directory path
    surface,             # Tifxyz object to write
    compression='lzw',   # TIFF compression
    tile_size=1024,      # TIFF tile size
    write_mask=True,     # Write mask.tif
    overwrite=False,     # Overwrite existing
)
```

### TifxyzReader

For more control over reading:

```python
from vesuvius.tifxyz import TifxyzReader

reader = TifxyzReader("/path/to/segment")

# Read just metadata
meta = reader.read_metadata()

# Read individual components
x = reader.read_coordinate('x')
y = reader.read_coordinate('y')
z = reader.read_coordinate('z')
mask = reader.read_mask()

# List extra channels
channels = reader.list_extra_channels()  # e.g., ['generations']
data = reader.read_extra_channel('generations')
```

---

## Invalid Points

Invalid points (holes in the surface) are indicated by:
- `z <= 0`
- Coordinates set to `(-1, -1, -1)`
- `valid_mask[y, x] == False`

```python
# Check single point
if surface.valid_mask[row, col]:
    # Point is valid

# Count valid points
num_valid = surface.valid_mask.sum()

# Get all valid coordinates
valid = surface.valid_mask
points = np.stack([surface.x[valid], surface.y[valid], surface.z[valid]], axis=1)
```
