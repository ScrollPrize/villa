#pragma once

#include "vc_object_store/objects/Geometry.hpp"
#include <array>
#include <optional>
#include <span>
#include <variant>

namespace vc::geometry {

using Point = std::array<double, 3>;
using PatchPoint = std::array<float, 3>;
using TrackPoint = std::array<int32_t, 3>;
using PointId = int64_t;

// Owned geometry, independent of identity, annotations, files and the database.
// Coordinates are source XYZ voxels (no implicit scaling or axis conversion).
// Public fields support construction/editing; validate() and encode() check them.
struct PointCollection {
    std::vector<PointId> ids;
    std::vector<Point> points;
    void validate() const;
    std::optional<size_t> indexOf(PointId id) const;
    std::optional<Point> point(PointId id) const;
};

struct Fiber {
    std::vector<Point> controlPoints, linePoints;
    std::vector<PointId> controlPointIds; // optional before first structural edit
    void validate() const;
    // Sampled line if present, otherwise the control polyline.
    double length() const;
};

struct Track {
    std::vector<TrackPoint> points;
    void validate() const;
    double length() const;
};

struct Patch {
    int64_t rows = 0, columns = 0;
    std::array<double, 2> scale{1, 1}; // column, row; grid steps per source voxel
    std::vector<PatchPoint> points;   // row-major; invalid vertices are (-1,-1,-1)
    void validate() const;
    const PatchPoint& vertex(int64_t row, int64_t column) const;
    bool valid(int64_t row, int64_t column) const;
};

using Geometry = std::variant<PointCollection, Fiber, Track, Patch>;

// Preserve scalar precision, point order and stable IDs. encode/decode are the
// bridge to the existing lossless array storage, not external file formats.
objects::GeometryData encode(const Geometry&);
Geometry decode(const objects::GeometryData&);

double polylineLength(std::span<const Point>);
double polylineLength(std::span<const TrackPoint>);
bool validPatchPoint(const PatchPoint&);

} // namespace vc::geometry
