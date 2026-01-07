# TifXYZ API Documentation

Python module for reading and writing tifxyz surface files from volume-cartographer.

## Installation

```python
from vesuvius.tifxyz import Tifxyz, read_tifxyz, write_tifxyz, list_tifxyz, load_folder
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

## Quick Start

```python
from vesuvius.tifxyz import read_tifxyz, write_tifxyz

# Read a surface (defaults to stored resolution)
surface = read_tifxyz("/path/to/segment")

# Get dimensions at current resolution
print(surface.shape)  # e.g., (4215, 4373) - stored resolution

# Access coordinates (direct array access at stored resolution)
x, y, z, valid = surface[100, 200]           # Single point
x, y, z, valid = surface[100:200, 200:300]   # 100x100 tile

# Switch to full resolution for interpolated access
surface.use_full_resolution()
print(surface.shape)  # e.g., (84300, 87460) - full resolution
x, y, z, valid = surface[2000:2100, 4000:4100]  # Interpolated

# Write a surface
write_tifxyz("/path/to/output", surface, overwrite=True)
```

---

## Class: Tifxyz

Main class representing a tifxyz surface.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `uuid` | `str` | Unique identifier |
| `bbox` | `tuple` or `None` | Bounding box (x_min, y_min, z_min, x_max, y_max, z_max) |
| `path` | `Path` or `None` | Source path if loaded from disk |
| `area` | `float` or `None` | Surface area if computed |
| `extra` | `dict` | Additional metadata fields |
| `volume` | `zarr.Array` or `None` | Associated volume (set via `volume_path` or manually) |
| `resolution` | `"stored"` or `"full"` | Current resolution mode (default: "stored") |
| `interp_method` | `str` | Interpolation method (default: "catmull_rom") |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `shape` | `tuple[int, int]` | Grid dimensions at current resolution |
| `valid_quad_mask` | `NDArray[bool]` | Mask of valid quads at stored resolution (H-1, W-1) |
| `valid_quad_indices` | `NDArray[int64]` | Indices of valid quads as (N, 2) array |
| `valid_vertex_mask` | `NDArray[bool]` | Mask of valid vertices at stored resolution |
| `quad_area` | `float` | Surface area computed from valid quad count |
| `quad_centers` | `NDArray[float32]` | Centers of quads at stored resolution (H-1, W-1, 3) |

---

## Resolution Mode

Surfaces can operate in two resolution modes:

- **`"stored"`** (default): Direct array access without interpolation. Fast and memory-efficient.
- **`"full"`**: Interpolated access at full resolution. Slower but provides sub-pixel accuracy.

### Switching Modes

```python
surface = read_tifxyz("/path/to/segment")

# Default: stored resolution
print(surface.resolution)  # "stored"
print(surface.shape)       # e.g., (4215, 4373) - stored dimensions

# Switch to full resolution
surface.use_full_resolution()
print(surface.shape)       # e.g., (84300, 87460) - full dimensions

# Switch back to stored
surface.use_stored_resolution()

# Or set directly
surface.resolution = "full"

# Chaining is supported
coords = surface.use_full_resolution().get_zyxs()
```

### Behavior by Mode

| Method | Stored Mode (default) | Full Mode |
|--------|----------------------|-----------|
| `shape` | Internal array dimensions | Computed full dimensions |
| `surface[i, j]` | Direct array access | Interpolated coordinates |
| `get_normals()` | Slice from cached normals | Compute at full resolution |
| `get_zyxs()` | No interpolation | Interpolated |

Methods that always use stored resolution (regardless of mode):
- `compute_normals()` - whole-surface cached computation
- `valid_quad_mask`, `quad_centers` - mesh operations
- `compute_centroid()` - internal computation

---

## Coordinate Access

Access coordinates at any position using indexing. Behavior depends on resolution mode.

```python
# Single point (at current resolution)
x, y, z, valid = surface[row, col]

# Tile/region
x, y, z, valid = surface[100:200, 200:300]

# For interpolated access, switch to full resolution
surface.use_full_resolution()
x, y, z, valid = surface[1000:1100, 2000:2100]  # Now interpolated
```

All access methods return a tuple of four arrays:
- `x`, `y`, `z`: Coordinate arrays (float32)
- `valid`: Boolean mask indicating valid points

### get_zyxs

Get coordinates stacked as a single array (useful for neural networks).

```python
# Get stacked coordinates (uses current resolution mode)
zyxs = surface.get_zyxs()  # shape: (H, W, 3), order: z, y, x

# Force specific resolution (overrides current mode)
zyxs = surface.get_zyxs(stored_resolution=True)   # Always stored
zyxs = surface.get_zyxs(stored_resolution=False)  # Always full/interpolated

# Invalid points have value -1
valid = (zyxs != -1).all(axis=-1)

# Get as torch tensor directly
zyxs_tensor = surface.get_zyxs(as_tensor=True)
```

### Interpolation Methods

The surface uses Catmull-Rom interpolation by default, which provides smooth curves that pass through control points. You can change the method:

```python
# Change default interpolation method
surface.interp_method = "linear"  # faster but less smooth
```

Available methods:
- `"catmull_rom"` (default): Smooth spline that passes through control points. Best quality.
- `"linear"`: Bilinear interpolation. Fast (via OpenCV).
- `"bspline"`: B-spline interpolation (via scipy). Smooth but approximating.

---

## Normals

### get_normals

Compute surface normals for a tile. In stored mode, slices from cached normals. In full mode, computes at full resolution.

```python
# Get normals for a tile (at current resolution)
nx, ny, nz = surface.get_normals(row_start=100, row_end=200, col_start=200, col_end=300)
# Returns NaN for invalid/boundary points
# Shape: (100, 100) each
```

### compute_normals

Compute surface normals for the entire surface (at internal resolution). Results are cached.

```python
nx, ny, nz = surface.compute_normals()
# Returns NaN for invalid/boundary points
# Subsequent calls return cached result
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
Computes normals automatically if not provided.

```python
# Make all normals point outward (default)
nx, ny, nz = surface.orient_normals('outward')

# Make all normals point inward
nx, ny, nz = surface.orient_normals('inward')

# Or pass pre-computed normals
nx, ny, nz = surface.orient_normals('outward', normals=(nx, ny, nz))
```

### flip_normals

Flip all normals (negate all components). Computes normals automatically if not provided.

```python
nx, ny, nz = surface.flip_normals()
# Equivalent to: -nx, -ny, -nz

# Or pass pre-computed normals
nx, ny, nz = surface.flip_normals(normals=(nx, ny, nz))
```

---

## Row Smoothing

### smooth_rows_catmull_rom

Apply 1D Catmull-Rom smoothing to each row independently. Returns array in same format as `get_zyxs()`.

For each row, collects valid points in column order (skipping invalid points), then applies 1D Catmull-Rom smoothing to the (x, y, z) coordinates independently.

```python
# Basic usage (uses current resolution mode)
smoothed = surface.smooth_rows_catmull_rom()
smoothed.shape  # (H, W, 3) - same as get_zyxs()

# Force stored resolution (fast, no interpolation)
smoothed = surface.smooth_rows_catmull_rom(stored_resolution=True)

# Force full resolution (interpolated)
smoothed = surface.smooth_rows_catmull_rom(stored_resolution=False)

# Invalid points still have value -1
valid = (smoothed != -1).all(axis=-1)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `stored_resolution` | `bool` or `None` | `None` | `True`: force stored resolution. `False`: force full resolution. `None`: use current `resolution` setting. |

**Returns:**

`NDArray[np.float32]` - Shape `(H, W, 3)` with smoothed coordinates in `[z, y, x]` order, same format as `get_zyxs()`. Invalid points have value -1. Rows with < 2 valid points are left unchanged.

**Notes:**
- Uses Catmull-Rom weights at t=0.5: `[-1/16, 9/16, 9/16, -1/16]`
- Edge points are handled by linearly extrapolating phantom control points beyond boundaries
- Invalid points (z ≤ 0 or not finite) are skipped during smoothing
- Valid points are replaced with their smoothed values

---

## Volume Association

Associate a surface with a zarr volume for coordinate-based sampling workflows.

### Setting the Volume

```python
# Option 1: Set volume at load time
surface = read_tifxyz("/path/to/segment", volume_path="/path/to/volume.zarr")

# Option 2: Set volume after loading
import zarr
surface.volume = zarr.open("/path/to/volume.zarr", mode="r")
```

### retarget

Rescale coordinates for a downsampled or upsampled volume. Returns a new Tifxyz instance.

```python
# Original surface at full resolution
surface = read_tifxyz("/path/to/segment", volume_path="/path/to/volume.zarr")

# Create surface for 2x downsampled volume
# Coordinates are divided by 2, volume is set to level "1" if OME-zarr
surface_2x = surface.retarget(2.0)

# Create surface for 4x downsampled volume
surface_4x = surface.retarget(4.0)

# For upsampled volume (factor < 1)
surface_half = surface.retarget(0.5)  # Coordinates multiplied by 2
```

When the volume is an OME-zarr group with multiple resolution levels (named "0", "1", "2", etc.), `retarget()` automatically selects the appropriate level based on the factor.

### Quad Properties

Properties for mesh-based operations at stored resolution:

```python
# Boolean mask of valid quads (all 4 corners valid)
mask = surface.valid_quad_mask  # shape: (H-1, W-1)

# Indices of valid quads
indices = surface.valid_quad_indices  # shape: (N, 2)

# Boolean mask of valid vertices
vertex_mask = surface.valid_vertex_mask  # shape: (H, W)

# Surface area from valid quad count
area = surface.quad_area

# Centers of each quad
centers = surface.quad_centers  # shape: (H-1, W-1, 3), invalid quads have -1
```

---

## Reading & Writing

### read_tifxyz

```python
surface = read_tifxyz(
    path,                   # Path to tifxyz directory
    load_mask=True,         # Load mask.tif if present
    validate=True,          # Validate data after loading
    volume_path=None,       # Optional path to OME-zarr volume
)

# Example with volume association
surface = read_tifxyz("/path/to/segment", volume_path="/path/to/volume.zarr")
print(surface.volume)  # zarr.Group or zarr.Array
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

## Discovery

Functions for finding and filtering tifxyz segments in a folder.

### list_tifxyz

Discover all tifxyz segments in a folder without loading coordinates. Returns lightweight `TifxyzInfo` objects for filtering before loading.

```python
from vesuvius.tifxyz import list_tifxyz

# Find all segments in a folder
segments = list_tifxyz("/path/to/segments")

# Filter by z-range (only segments whose bbox overlaps this range)
segments = list_tifxyz("/path/to/segments", z_range=(1000, 2000))

# Non-recursive (only immediate subdirectories)
segments = list_tifxyz("/path/to/segments", recursive=False)

# Work with results
for seg in segments:
    print(f"{seg.uuid}: z=[{seg.z_min}, {seg.z_max}]")
    surface = seg.load()  # Load full data when needed
```

### load_folder

Load all tifxyz segments in a folder.

```python
from vesuvius.tifxyz import load_folder

# Load all segments
for surface in load_folder("/path/to/segments", z_range=(1000, 2000)):
    print(f"{surface.uuid}: {surface.shape}")
```

### TifxyzInfo

Lightweight metadata container returned by `list_tifxyz`.

| Attribute | Type | Description |
|-----------|------|-------------|
| `path` | `Path` | Path to the tifxyz directory |
| `scale` | `tuple[float, float]` | Scale factors (scale_y, scale_x) |
| `bbox` | `tuple` or `None` | Bounding box (x_min, y_min, z_min, x_max, y_max, z_max) |
| `uuid` | `str` | Unique identifier |

| Property | Type | Description |
|----------|------|-------------|
| `z_min` | `float` or `None` | Minimum z from bbox |
| `z_max` | `float` or `None` | Maximum z from bbox |

| Method | Description |
|--------|-------------|
| `load(**kwargs)` | Load full `Tifxyz` object (kwargs passed to `read_tifxyz`) |

---

## Invalid Points

Invalid points (holes in the surface) are indicated by:
- `z <= 0`
- Coordinates set to `(-1, -1, -1)`
- The `valid` array returned from coordinate access is `False`

```python
# Access returns validity as 4th element
x, y, z, valid = surface[1000:1100, 2000:2100]

# Use valid mask to filter
x_valid = x[valid]
y_valid = y[valid]
z_valid = z[valid]

# Get all valid coordinates as point cloud
points = np.stack([x[valid], y[valid], z[valid]], axis=1)
```
