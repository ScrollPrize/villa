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
enum class FamilyFilter { All, Horizontal, Vertical, None };
struct Bounds {
    Point lower{}, upper{};
    bool operator==(const Bounds&) const = default;
};
struct Summary {
    int64_t id{};
    std::string name, family;
    int64_t pointCount{};
    double length{};
    Bounds bounds;
};
struct Block {
    int64_t id{}, fiberId{}, firstSegment{};
    std::string family;
    std::vector<Point> points;
};
struct RegionPage {
    std::vector<Block> blocks;
    bool complete{true};
    size_t decodedBytes{};
};
struct ViewRegion {
    Point origin{}, u{1, 0, 0}, v{0, 1, 0}, n{0, 0, 1};
    Bounds box, local;
    bool operator==(const ViewRegion&) const = default;
    Point project(const Point& point) const;
    // Exact for slices sharing the same axes; conservatively false otherwise.
    bool contains(const ViewRegion& other) const;
};
bool clipSegment(const Point&, const Point&, const Bounds&, double&, double&);
} // namespace vc::fibers
