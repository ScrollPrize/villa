#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace vc::fibers
{
using Point = std::array<double, 3>;
struct Bounds {
    Point lower{}, upper{};
    bool operator==(const Bounds&) const = default;
};
struct ViewRegion {
    Point origin{}, u{1, 0, 0}, v{0, 1, 0}, n{0, 0, 1};
    Bounds box, local;
    bool operator==(const ViewRegion&) const = default;
    Point project(const Point& point) const;
    // Exact for any axes: true when every corner of other's box lies in this box.
    bool contains(const ViewRegion& other) const;
};
// The slab of a viewer's visible rectangle: its origin, right and down
// corners (volume voxels), +-depth along the normal. Viewer corners are
// float-rounded, so their axes are orthogonal only to ~1e-7; the axes are
// orthonormalized in double (the object store requires orthonormal axes) and
// local covers all four corners. box is left empty. nullopt when degenerate.
std::optional<ViewRegion> viewRegionFromCorners(const Point& origin, const Point& right, const Point& down,
                                                double depth);
bool clipSegment(const Point&, const Point&, const Bounds&, double&, double&);
} // namespace vc::fibers
