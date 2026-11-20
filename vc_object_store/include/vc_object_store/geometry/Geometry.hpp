#pragma once

#include "vc_object_store/objects/Geometry.hpp"
#include <array>
#include <cmath>
#include <concepts>
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

// Line tails. A manual fiber (two or more control points) is annotated only
// between its first and last control point. The line points the optimizer
// extrapolates past them (the open tails) are not fiber: every consumer ignores
// them - geometry, normals, crossings and constraints, link anchors, lengths,
// bounds and regions. A fiber with fewer than two controls (automatic traces)
// has no tails; its whole line counts. Read a fiber's line through
// annotatedLine/annotatedLineRange; the raw `linePoints` (with tails) are for
// editing, display of the extrapolation and lossless round trips only.
//
// The writer keeps every control as an exact line vertex. The annotated range
// runs over the line vertices nearest the controls, from the lowest to the
// highest such index, inclusive; control order does not matter (a live edit
// may hold its controls in any order) and non-finite controls do not count.
struct AnnotatedLineRange {
    size_t begin = 0, end = 0;  // [begin, end) into linePoints
    size_t size() const { return end - begin; }
};
AnnotatedLineRange annotatedLineRange(std::span<const Point> controlPoints, std::span<const Point> linePoints);
std::span<const Point> annotatedLine(std::span<const Point> controlPoints, std::span<const Point> linePoints);

struct Fiber {
    std::vector<Point> controlPoints;
    // Raw line including the extrapolated tails; consumers read annotatedLine().
    std::vector<Point> linePoints;
    std::vector<PointId> controlPointIds; // encode allocates IDs for a new value when omitted
    void validate() const;
    // The line without its tails (see "Line tails" above).
    std::span<const Point> annotatedLine() const { return geometry::annotatedLine(controlPoints, linePoints); }
    // Annotated line if present, otherwise the control polyline; never the tails.
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
// The store's one rule for surface vertices: a vertex has coordinates unless a
// component is -1 or non-finite, or z <= 0 (tracers leave extrapolations there).
// Stored surfaces hold missing vertices as the (-1,-1,-1) sentinel.
inline bool validSurfacePoint(const SurfacePoint& p)
{
    for (const auto v : p)
        if (!std::isfinite(v) || v == -1) return false;
    return p[2] > 0;
}
// The same for any three-component vector (cv::Vec3f, Eigen, ...).
template <class Vector>
    requires(!std::same_as<Vector, SurfacePoint>)
bool validSurfacePoint(const Vector& p)
{
    return validSurfacePoint(SurfacePoint{static_cast<float>(p[0]), static_cast<float>(p[1]), static_cast<float>(p[2])});
}

// The area of triangle abc (any three-component vectors), in double precision.
template <class Vector>
double triangleArea(const Vector& a, const Vector& b, const Vector& c)
{
    const double ux = static_cast<double>(b[0]) - static_cast<double>(a[0]);
    const double uy = static_cast<double>(b[1]) - static_cast<double>(a[1]);
    const double uz = static_cast<double>(b[2]) - static_cast<double>(a[2]);
    const double vx = static_cast<double>(c[0]) - static_cast<double>(a[0]);
    const double vy = static_cast<double>(c[1]) - static_cast<double>(a[1]);
    const double vz = static_cast<double>(c[2]) - static_cast<double>(a[2]);
    const double cx = uy * vz - uz * vy;
    const double cy = uz * vx - ux * vz;
    const double cz = ux * vy - uy * vx;
    return 0.5 * std::sqrt(cx * cx + cy * cy + cz * cz);
}

// The area of a surface grid in its vertices' units squared, added to `total`
// cell by cell in row-major order (so a grid read in row bands sums the same).
// Each cell is the triangles (p00,p10,p11) and (p00,p11,p01); a triangle counts
// when its three vertices have coordinates. `at(row, column)` is the vertex,
// `valid(row, column)` whether it has coordinates.
template <class At, class Valid>
double surfaceGridArea(int64_t rows, int64_t columns, At&& at, Valid&& valid, double total = 0)
{
    for (int64_t r = 0; r + 1 < rows; ++r)
        for (int64_t c = 0; c + 1 < columns; ++c) {
            const bool v00 = valid(r, c), v10 = valid(r, c + 1), v01 = valid(r + 1, c), v11 = valid(r + 1, c + 1);
            const double first = v00 && v10 && v11 ? triangleArea(at(r, c), at(r, c + 1), at(r + 1, c + 1)) : 0.0;
            const double second = v00 && v11 && v01 ? triangleArea(at(r, c), at(r + 1, c + 1), at(r + 1, c)) : 0.0;
            total += first + second;
        }
    return total;
}

} // namespace vc::geometry
