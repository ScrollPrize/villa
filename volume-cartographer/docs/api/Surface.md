# Surface API

The Surface classes provide an abstraction for working with 2D surfaces embedded in 3D space. These surfaces can represent scroll sheets, planes, or other geometric constructs within volumetric data.

## Base Class: `Surface`

`Surface` is the abstract base class that defines the common interface for all surface types.

```cpp
#include "vc/core/util/Surface.hpp"
```

### Virtual Methods

| Method | Description |
|--------|-------------|
| `pointer()` | Returns a central location point on the surface |
| `move(ptr, offset)` | Moves a pointer within the internal coordinate system |
| `valid(ptr, offset)` | Checks if the pointer location contains valid surface data |
| `loc(ptr, offset)` | Returns nominal pointer coordinates in output coordinates |
| `coord(ptr, offset)` | Returns the 3D world coordinate at the pointer location |
| `normal(ptr, offset)` | Returns the surface normal at the pointer location |
| `pointTo(ptr, coord, th, max_iters, ...)` | Finds the surface point closest to a target 3D coordinate |
| `gen(coords, normals, size, ptr, scale, offset)` | Generates coordinate and normal maps for rendering |

### Members

| Member | Type | Description |
|--------|------|-------------|
| `meta` | `nlohmann::json*` | Optional metadata associated with the surface |
| `path` | `std::filesystem::path` | File path for surfaces loaded from disk |
| `id` | `std::string` | Unique identifier for the surface |

---

## QuadSurface

A quad-based surface class that stores 3D coordinates on a regular 2D grid. This is the primary surface type used for scroll segmentations.

```cpp
#include "vc/core/util/QuadSurface.hpp"
```

### Constructors

```cpp
// Clone points into the surface
QuadSurface(const cv::Mat_<cv::Vec3f>& points, const cv::Vec2f& scale);

// Use points pointer directly (takes ownership)
QuadSurface(cv::Mat_<cv::Vec3f>* points, const cv::Vec2f& scale);
```

**Parameters:**
- `points`: A matrix of 3D coordinates. Invalid points are marked as `{-1, -1, -1}`.
- `scale`: The scale factor `{x_scale, y_scale}` mapping grid coordinates to nominal coordinates.

### Surface API Implementation

| Method | Description |
|--------|-------------|
| `pointer()` | Returns `{0, 0, 0}` (the center of the surface in relative coordinates) |
| `move(ptr, offset)` | Moves the pointer by offset scaled by the surface scale |
| `valid(ptr, offset)` | Returns true if the location contains valid (non-`-1`) data |
| `loc(ptr, offset)` | Converts internal pointer to nominal coordinates |
| `coord(ptr, offset)` | Returns interpolated 3D world coordinate at the location |
| `normal(ptr, offset)` | Computes surface normal via cross product of adjacent grid points |
| `pointTo(ptr, tgt, th, ...)` | Searches for the closest surface point to target coordinate |
| `gen(coords, normals, size, ptr, scale, offset)` | Generates coordinate/normal maps using affine warping |

### Additional Methods

| Method | Description |
|--------|-------------|
| `loc_raw(ptr)` | Returns internal absolute coordinates (upper-left at 0,0) |
| `gridNormal(row, col)` | Gets normal directly from grid indices (faster than `normal()`) |
| `size()` | Returns the surface size in nominal coordinates |
| `scale()` | Returns the scale factor `{x, y}` |
| `center()` | Returns the center point in internal coordinates |
| `bbox()` | Computes and caches the 3D bounding box of all valid points |

### Persistence

| Method | Description |
|--------|-------------|
| `save(path, uuid, force_overwrite)` | Saves to a directory with x.tif, y.tif, z.tif and meta.json |
| `save(path, force_overwrite)` | Saves using the directory name as uuid |
| `save_meta()` | Saves only the meta.json file |
| `saveOverwrite()` | Overwrites the surface at its current path |
| `saveSnapshot(maxBackups)` | Creates a backup in the volpkg's `backups/` directory |

### Channels

Surfaces can have additional data channels (e.g., masks, generation info) stored as TIFFs.

| Method | Description |
|--------|-------------|
| `setChannel(name, mat)` | Sets a named channel |
| `channel(name, flags)` | Retrieves a channel (loads from disk on demand) |
| `channelNames()` | Returns list of available channel names |
| `invalidateCache()` | Recomputes internal caches (bounds, center, bbox) |
| `invalidateMask()` | Removes the mask channel from memory and disk |

**Channel Flags:**
- `SURF_CHANNEL_NORESIZE` (1): Don't resize channel to match points matrix

### Raw Data Access

| Method | Description |
|--------|-------------|
| `rawPoints()` | Returns a copy of the points matrix |
| `rawPointsPtr()` | Returns a pointer to the internal points matrix |

### Grid Iteration

Range-based iteration over valid points and quads. Invalid points are marked as `{-1, -1, -1}`.

```cpp
// Iterate over all valid points
for (auto [row, col, point] : surf->validPoints()) {
    point += offset;  // Can modify (non-const version)
}

// Iterate over valid 2x2 quads (all 4 corners valid)
for (auto [row, col, p00, p01, p10, p11] : surf->validQuads()) {
    // p00 = (row, col), p01 = (row, col+1)
    // p10 = (row+1, col), p11 = (row+1, col+1)
}

// Works with STL algorithms
auto it = std::find_if(surf->validPoints().begin(), surf->validPoints().end(),
    [](auto& ref) { return ref.point[2] > 100.0f; });
```

| Method | Description |
|--------|-------------|
| `validPoints()` | Returns range of valid points (mutable or const) |
| `validQuads()` | Returns range of valid 2x2 quads |
| `isPointValid(row, col)` | Checks if a single point is valid |
| `isQuadValid(row, col)` | Checks if a 2x2 quad starting at (row, col) is valid |
| `countValidPoints()` | Returns count of valid points |
| `countValidQuads()` | Returns count of valid quads |

### Validity Mask

| Method | Description |
|--------|-------------|
| `validMask()` | Returns `cv::Mat_<uint8_t>` mask at native resolution (255=valid, 0=invalid) |
| `writeValidMask(const cv::Mat& img)` | Writes mask to `path/mask.tif`. If `img` provided, writes multi-layer TIFF |

### Loading

```cpp
QuadSurface* load_quad_from_tifxyz(const std::string& path, int flags = 0);
```

Loads a surface from a tifxyz directory containing:
- `x.tif`, `y.tif`, `z.tif` - coordinate bands
- `meta.json` - metadata with scale, uuid, bbox
- Optional: `mask.tif`, other channel TIFFs

**Load Flags:**
- `SURF_LOAD_IGNORE_MASK` (1): Don't apply the mask when loading

### Surface Operations

```cpp
// Returns points in A that are not within tolerance of any point in B
QuadSurface* surface_diff(QuadSurface* a, QuadSurface* b, float tolerance = 2.0);

// Combines points from both surfaces
QuadSurface* surface_union(QuadSurface* a, QuadSurface* b, float tolerance = 2.0);

// Returns only points that exist in both surfaces
QuadSurface* surface_intersection(QuadSurface* a, QuadSurface* b, float tolerance = 2.0);
```

### Utility Types

```cpp
struct Rect3D {
    cv::Vec3f low = {0, 0, 0};
    cv::Vec3f high = {0, 0, 0};
};

bool intersect(const Rect3D& a, const Rect3D& b);
Rect3D expand_rect(const Rect3D& a, const cv::Vec3f& p);
```

### CUDA Control

```cpp
// Enable/disable CUDA for space tracing operations
void set_space_tracing_use_cuda(bool enable);
```

---

## PlaneSurface

An infinite planar surface defined by an origin point and normal vector. Useful for slicing volumes and defining cutting planes.

```cpp
#include "vc/core/util/PlaneSurface.hpp"
```

### Constructors

```cpp
PlaneSurface();  // Default: origin at {0,0,0}, normal {0,0,1}
PlaneSurface(cv::Vec3f origin, cv::Vec3f normal);
```

### Surface API Implementation

| Method | Description |
|--------|-------------|
| `pointer()` | Returns `{0, 0, 0}` |
| `move(ptr, offset)` | Simple addition: `ptr += offset` |
| `valid(ptr, offset)` | Always returns `true` (infinite plane) |
| `loc(ptr, offset)` | Returns `ptr + offset` |
| `coord(ptr, offset)` | Generates the 3D coordinate on the plane |
| `normal(ptr, offset)` | Returns the plane's normal vector |
| `pointTo(...)` | Not implemented (calls `abort()`) |
| `gen(coords, normals, size, ptr, scale, offset)` | Generates a grid of coordinates on the plane |

### Plane-Specific Methods

| Method | Description |
|--------|-------------|
| `setNormal(normal)` | Sets and normalizes the plane normal, updates basis vectors |
| `setOrigin(origin)` | Sets the plane origin |
| `origin()` | Returns the current origin |
| `pointDist(wp)` | Returns the absolute distance from a point to the plane |
| `scalarp(point)` | Returns the signed distance from point to plane |
| `project(wp, render_scale, coord_scale)` | Projects a 3D point onto the plane's local 2D coordinate system |

### In-Plane Rotation

| Method | Description |
|--------|-------------|
| `setInPlaneRotation(radians)` | Rotates the in-plane basis vectors around the normal |
| `inPlaneRotation()` | Returns the current in-plane rotation angle |
| `basisX()` | Returns the X basis vector in the plane |
| `basisY()` | Returns the Y basis vector in the plane |

### Helper Function

```cpp
// Find location on surface that minimizes weighted distance to target points while staying on plane
float min_loc(
    const cv::Mat_<cv::Vec3f>& points,
    cv::Vec2f& loc,
    cv::Vec3f& out,
    const std::vector<cv::Vec3f>& tgts,
    const std::vector<float>& tds,
    PlaneSurface* plane,
    float init_step = 16.0,
    float min_step = 0.125
);
```

---

## Coordinate Systems

All surface classes use three coordinate systems:

1. **Nominal (World) Coordinates**: 3D voxel coordinates in the volume
2. **Internal Relative (Pointer) Coordinates**: Surface-local coordinates where the center is at `{0, 0, 0}`
3. **Internal Absolute (Grid) Coordinates**: For QuadSurface, grid indices where `{0, 0}` is the upper-left corner

The `ptr` parameter in most methods is in internal relative coordinates. Use `loc()` to convert to nominal coordinates and `coord()` to get the actual 3D world position.
