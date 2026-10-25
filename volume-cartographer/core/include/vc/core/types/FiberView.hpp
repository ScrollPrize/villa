#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <memory>
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
bool clipSegment(const Point&, const Point&, const Bounds&, double&, double&);
} // namespace vc::fibers
