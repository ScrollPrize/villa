#pragma once

#include "vc_object_store/objects/Geometry.hpp"
#include <array>
#include <optional>
#include <span>
#include <variant>

namespace vc::geometry {

using Point = std::array<double, 3>;
using SurfacePoint = std::array<float, 3>;
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
    std::vector<PointId> controlPointIds; // encode allocates IDs for a new value when omitted
    void validate() const;
    // Sampled line if present, otherwise the control polyline.
    double length() const;
};

struct Track {
    std::vector<TrackPoint> points;
    void validate() const;
    double length() const;
};

struct Surface {
    int64_t rows = 0, columns = 0;
    std::array<double, 2> scale{1, 1}; // column, row; grid steps per source voxel
    std::vector<SurfacePoint> points;   // row-major; vertices without coordinates are (-1,-1,-1)
    // The `valid` channel: empty, or one byte per vertex (row-major, nonzero = valid).
    std::vector<uint8_t> validity;
    // Other aligned channels, each a whole multiple of the grid per axis.
    std::vector<objects::GeometryArray> channels;
    void validate() const;
    const SurfacePoint& vertex(int64_t row, int64_t column) const;
    // Has coordinates and is not marked invalid by `validity`.
    bool valid(int64_t row, int64_t column) const;
};

// A scroll's umbilicus: the ordered control polyline of its centre, in source XYZ.
struct Umbilicus {
    std::vector<Point> points;
    void validate() const;
};

using Geometry = std::variant<PointCollection, Fiber, Track, Surface, Umbilicus>;

// Preserve scalar precision, point order and stable IDs. encode/decode are the
// bridge to the existing lossless array storage, not external file formats.
objects::GeometryData encode(const Geometry&);
Geometry decode(const objects::GeometryData&);

// Per-vertex validity (255 valid, 0 invalid) of a rows x columns grid from a
// row-major mask that is a whole multiple of it: a vertex is valid only when
// every mask value it covers is at least 255.
std::vector<uint8_t> validityFromMask(std::span<const double> mask, int64_t maskRows, int64_t maskColumns,
                                      int64_t rows, int64_t columns);
double polylineLength(std::span<const Point>);
double polylineLength(std::span<const TrackPoint>);
// A surface vertex has coordinates unless a component is -1 or non-finite;
// missing vertices are stored as the (-1,-1,-1) sentinel.
bool validSurfacePoint(const SurfacePoint&);
// tifxyz grids, and the in-memory grids of tools that read and write them,
// also mark missing vertices with z <= 0 (tracers leave extrapolations there).
// Their readers and writers convert with this rule at that boundary.
bool validTifxyzPoint(const SurfacePoint&);

} // namespace vc::geometry
