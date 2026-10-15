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
struct ViewCacheStats {
    uint64_t candidateQueries{}, candidateHits{}, blockReads{}, blockHits{};
    size_t retainedBytes{};
};

// Read-only view of an Automated Fiber Volume (.afv). Each instance owns one
// SQLite connection, so calls on an instance must be serialized, including
// across worker threads. metadata(), summary() and catalogByLength() never read
// geometry. Coordinates are native L0 XYZ doubles.
class FiberCollection
{
public:
    // viewCacheBytes bounds the cache reused by successive viewRegion() calls.
    explicit FiberCollection(const std::filesystem::path& path, size_t viewCacheBytes = 0);
    ~FiberCollection();
    FiberCollection(const FiberCollection&) = delete;
    FiberCollection& operator=(const FiberCollection&) = delete;

    std::string metadata(const std::string& key) const;
    Summary summary(int64_t fiberId) const;
    // Descending length, then ascending ID. Start with +infinity/0 and continue
    // from the last returned row. minLength is the complete polyline length in
    // native L0 voxels, inclusive.
    std::vector<Summary> catalogByLength(
        double beforeLength, int64_t afterId = 0, int limit = 200, double minLength = 0, FamilyFilter family = FamilyFilter::All) const;
    // Fibers crossing the oriented slice, as contiguous runs of their visible
    // segments with original coordinates and segment indices. maximum=0 returns
    // every crossing fiber; otherwise at most maximum fibers are chosen by stable
    // per-fiber priorities, with selected first when it crosses the slice.
    // complete=false reports that byteBudget stopped the query.
    RegionPage viewRegion(
        const ViewRegion& view,
        int minPoints = 2,
        size_t maximum = 0,
        int64_t selected = 0,
        size_t byteBudget = 8 * 1024 * 1024,
        const std::atomic_bool* cancelled = nullptr,
        double minLength = 0,
        FamilyFilter family = FamilyFilter::All) const;
    ViewCacheStats viewCacheStats() const;
    std::vector<Block> fiberBlocks(int64_t fiberId, int64_t firstSegment = 0, int limit = 64) const;
    Point point(int64_t fiberId, int64_t index) const;
    // Original annotation JSON without line_points.
    std::string annotation(int64_t fiberId) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Exact slab clipping of segment ab in double precision. On success,
// [first, last] is the parameter range of ab inside bounds.
bool clipSegment(const Point& a, const Point& b, const Bounds& bounds, double& first, double& last);
}  // namespace vc::fibers
