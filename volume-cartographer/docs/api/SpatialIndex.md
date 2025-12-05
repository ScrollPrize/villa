# Spatial Index API

The spatial index classes provide efficient spatial queries for 3D geometric data using R-tree data structures (via Boost.Geometry). These indexes accelerate operations like finding nearby points or locating which surface a 3D coordinate belongs to.

---

## SurfacePatchIndex

A spatial index for efficiently querying triangulated surface patches (QuadSurface objects). Useful for determining which surface contains a given 3D point, or for finding all triangles within a region.

```cpp
#include "vc/core/util/SurfacePatchIndex.hpp"
```

### Result Types

```cpp
struct LookupResult {
    QuadSurface* surface = nullptr;  // The surface containing the point
    cv::Vec3f ptr = {0, 0, 0};       // Surface-local pointer coordinates
    float distance = -1.0f;          // Distance from query point to surface
};

struct TriangleCandidate {
    QuadSurface* surface = nullptr;
    int i = 0;                       // Grid row index
    int j = 0;                       // Grid column index
    int triangleIndex = 0;           // 0 = (p00,p10,p01), 1 = (p10,p11,p01)
    std::array<cv::Vec3f, 3> world{};         // 3D world coordinates of vertices
    std::array<cv::Vec3f, 3> surfaceParams{}; // Surface pointer coordinates for vertices
};

struct TriangleSegment {
    QuadSurface* surface = nullptr;
    std::array<cv::Vec3f, 2> world{};        // 3D world coordinates of segment endpoints
    std::array<cv::Vec3f, 2> surfaceParams{}; // Surface pointer coordinates
};
```

### Building the Index

| Method | Description |
|--------|-------------|
| `rebuild(surfaces, bboxPadding)` | Build the index from a vector of `QuadSurface*`. Optional padding expands bounding boxes. |
| `clear()` | Remove all surfaces from the index |
| `empty()` | Returns `true` if the index contains no surfaces |

```cpp
SurfacePatchIndex index;
std::vector<QuadSurface*> surfaces = { surf1, surf2, surf3 };
index.rebuild(surfaces, 2.0f);  // 2 voxel padding on bounding boxes
```

### Point Location

| Method | Description |
|--------|-------------|
| `locate(worldPoint, tolerance, targetSurface)` | Find the closest surface point to a 3D coordinate |

**Parameters:**
- `worldPoint`: The 3D coordinate to locate
- `tolerance`: Maximum distance to consider a valid match
- `targetSurface`: Optional; if non-null, only search this specific surface

**Returns:** `std::optional<LookupResult>` - empty if no surface found within tolerance

```cpp
auto result = index.locate(cv::Vec3f{100, 200, 50}, 5.0f);
if (result) {
    QuadSurface* surf = result->surface;
    cv::Vec3f ptr = result->ptr;  // Use with surf->coord(ptr) to get exact position
    float dist = result->distance;
}
```

### Triangle Queries

Query all triangles within a 3D bounding box:

| Method | Description |
|--------|-------------|
| `queryTriangles(bounds, targetSurface, outCandidates)` | Find triangles in bounds, optionally filtered to one surface |
| `queryTriangles(bounds, targetSurfaces, outCandidates)` | Find triangles in bounds, filtered to a set of surfaces |
| `forEachTriangle(bounds, targetSurface, visitor)` | Iterate over triangles with a callback |
| `forEachTriangle(bounds, targetSurfaces, visitor)` | Iterate over triangles with a callback (set filter) |

```cpp
Rect3D bounds = {{50, 50, 50}, {150, 150, 150}};
std::vector<SurfacePatchIndex::TriangleCandidate> triangles;

// Get all triangles in the region
index.queryTriangles(bounds, nullptr, triangles);

// Get triangles from a specific surface only
index.queryTriangles(bounds, mySurface, triangles);

// Use a visitor pattern
index.forEachTriangle(bounds, nullptr, [](const auto& tri) {
    // Process each triangle
    cv::Vec3f v0 = tri.world[0];
    cv::Vec3f v1 = tri.world[1];
    cv::Vec3f v2 = tri.world[2];
});
```

### Triangle Clipping

Clip a triangle against a plane, returning the line segment where they intersect:

```cpp
static std::optional<TriangleSegment> clipTriangleToPlane(
    const TriangleCandidate& tri,
    const PlaneSurface& plane,
    float epsilon = 1e-4f);
```

Returns the segment where the triangle crosses the plane, or empty if no intersection.

### Incremental Updates

| Method | Description |
|--------|-------------|
| `updateSurface(surface)` | Reindex a surface after its geometry changed |
| `updateSurfaceRegion(surface, rowStart, rowEnd, colStart, colEnd)` | Reindex only a rectangular region |
| `removeSurface(surface)` | Remove a surface from the index |

```cpp
// After modifying surface geometry
index.updateSurface(surf);

// Or update just a region (more efficient for local changes)
index.updateSurfaceRegion(surf, 10, 20, 30, 40);

// Remove a surface
index.removeSurface(surf);
```

### Sampling Configuration

| Method | Description |
|--------|-------------|
| `setSamplingStride(stride)` | Set the grid sampling stride (default 1) |
| `samplingStride()` | Get the current sampling stride |

Higher stride values trade accuracy for performance by sampling fewer triangles.

---

## PointIndex

A spatial index for efficiently querying individual 3D points. Supports collection-based filtering and k-nearest neighbor queries.

```cpp
#include "vc/core/util/PointIndex.hpp"
```

### Result Type

```cpp
struct QueryResult {
    uint64_t id = 0;              // Point identifier
    uint64_t collectionId = 0;    // Collection the point belongs to
    cv::Vec3f position{0, 0, 0};  // 3D position
    float distanceSq = 0.0f;      // Squared distance from query point
};
```

### Building the Index

| Method | Description |
|--------|-------------|
| `clear()` | Remove all points from the index |
| `empty()` | Returns `true` if the index is empty |
| `size()` | Returns the total number of indexed points |

### Inserting Points

| Method | Description |
|--------|-------------|
| `insert(id, collectionId, position)` | Add a single point |
| `bulkInsert(points)` | Efficiently add multiple points using packing algorithm |
| `buildFromMat(points, collectionId)` | Build from a matrix of 3D points |

```cpp
PointIndex index;

// Single insertion
index.insert(1, 0, cv::Vec3f{10, 20, 30});
index.insert(2, 0, cv::Vec3f{15, 25, 35});

// Bulk insertion (more efficient for many points)
std::vector<std::tuple<uint64_t, uint64_t, cv::Vec3f>> points = {
    {1, 0, {10, 20, 30}},
    {2, 0, {15, 25, 35}},
    {3, 1, {100, 200, 300}},  // Different collection
};
index.bulkInsert(points);

// Build from OpenCV matrix
cv::Mat_<cv::Vec3f> coords(100, 100);
// ... fill coords ...
index.buildFromMat(coords, 0);  // All points in collection 0
```

### Modifying Points

| Method | Description |
|--------|-------------|
| `remove(id)` | Remove a point by its ID |
| `update(id, newPosition)` | Update a point's position; returns `false` if ID not found |

```cpp
index.update(1, cv::Vec3f{11, 21, 31});
index.remove(2);
```

### Nearest Neighbor Queries

| Method | Description |
|--------|-------------|
| `nearest(position, maxDistance)` | Find the globally nearest point |
| `nearestInCollection(position, collectionId, maxDistance)` | Find the nearest point in a specific collection |
| `kNearest(position, k, maxDistance)` | Find the k nearest points |
| `queryRadius(center, radius)` | Find all points within a radius |

**Note:** `maxDistance` defaults to infinity. The returned `distanceSq` is the **squared** distance.

```cpp
cv::Vec3f query{50, 50, 50};

// Find the single nearest point
auto result = index.nearest(query);
if (result) {
    float dist = std::sqrt(result->distanceSq);
}

// Find nearest in a specific collection
auto result = index.nearestInCollection(query, 0, 100.0f);

// Find 5 nearest neighbors
auto neighbors = index.kNearest(query, 5);
for (const auto& n : neighbors) {
    // n.id, n.position, n.distanceSq
}

// Find all points within radius 25
auto nearby = index.queryRadius(query, 25.0f);
```

---

## Implementation Notes

Both indexes use **Boost.Geometry R-trees**

