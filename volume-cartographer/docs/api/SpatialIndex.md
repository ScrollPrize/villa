# Resident spatial queries

Link `vc_object_store::vc_resident_geometry` for resident queries independently
of the optional relationship detectors. These queries do not write to a database.

## Nearest vertices

```cpp
#include <vc_object_store/geometry/VertexIndex.hpp>
vc::geometry::VertexIndex index;
index.rebuild(points); // cv::Mat_<cv::Vec3f>; optional row-major validity bytes
if (const auto hit = index.nearest({10, 20, 30}, 5.f)) {
    // hit->index is the original row-major vertex ID, including gaps.
    // hit->position is XYZ; hit->distanceSq is squared voxel distance.
}
```

Without explicit validity, the store's file convention excludes nonfinite
coordinates, any component equal to -1, and z <= 0. Explicit validity allows
finite negative coordinates and z=0, including transformed in-memory geometry.
The index owns its vertices; rebuild after edits. Equal-distance vertices use
the smaller original ID.

## Surface patches

```cpp
#include <vc_object_store/geometry/SurfaceIndex.hpp>
auto grid = std::make_shared<vc::object_links::SurfaceGrid>(points, cv::Vec2f(1,1));
vc::object_links::SurfaceIndex index;
index.rebuild({grid});
vc::object_links::SurfaceIndex::PointQuery query{{10,20,30}, 2.f, {}};
auto nearest = index.locate(query);
auto all = index.locateAll(query);
```

`SurfaceGrid` clones the coordinates and optionally accepts explicit validity.
Queries return the owning grid, distance, and `(column,row)` grid coordinates.
The filter supports one grid, an inclusion set, or an exclusion set. Tolerance
zero accepts exact hits. `locateAll` returns each matching surface once.

Each cell uses triangles `(top-left, top-right, bottom-left)` and
`(top-right, bottom-right, bottom-left)`. The shared bounds hierarchy prunes
candidate cells; triangle distance supplies the closest location. On a warped
cell these triangles differ from the reader's bilinear or bicubic surface patch.
A triangle hit's grid coordinates are not a promise that reader interpolation
at those coordinates yields the same XYZ point.

VC's `SurfaceQuerySet` associates immutable resident grids with editable
`QuadSurface` identities and converts returned grid coordinates into application
pointer coordinates. It exposes only nearest/all point queries and a single
surface filter. Rebuild it after a geometry edit; background queries retain the
snapshot they started with.
