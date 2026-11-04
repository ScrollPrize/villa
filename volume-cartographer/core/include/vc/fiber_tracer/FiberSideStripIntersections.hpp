#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

namespace vc::fiber_tracer {

// A fiber's polyline, in the strip's volume voxels.
struct FiberPolyline {
    uint64_t id = 0;
    uint64_t generation = 1;
    std::vector<cv::Vec3d> points;
};

enum class FiberSideStripIntersectionSource {
    BranchLink,
    FiberSegment,
};

enum class FiberSideStripProgressPhase {
    BuildStripTriangles,
    BuildTriangleIndex,
    BranchLinks,
    FiberSegments,
    Deduplicate,
};

struct FiberSideStripLineQuery {
    uint64_t fiberId = 0;
    cv::Vec3d point{0.0, 0.0, 0.0};
    cv::Vec3d direction{0.0, 0.0, 1.0};
    cv::Vec3d connectorStart{std::numeric_limits<double>::quiet_NaN(),
                             std::numeric_limits<double>::quiet_NaN(),
                             std::numeric_limits<double>::quiet_NaN()};
};

struct FiberSideStripQueryOptions {
    cv::Mat_<cv::Vec3f> stripPoints;
    double deduplicateStripDistance = 1.0e-3;
    double aabbPadding = 1.0e-6;
    size_t maxResults = 64;
    size_t workerThreads = 0;
    std::vector<uint64_t> excludedFiberIds;
    std::vector<const FiberPolyline*> queryFibers;
    std::vector<FiberSideStripLineQuery> branchLinks;
};

struct FiberSideStripIntersection {
    uint64_t fiberId = 0;
    uint64_t generation = 1;
    int segmentIndex = -1;
    double arclength = 0.0;
    cv::Vec3d point{0.0, 0.0, 0.0};
    double stripRow = std::numeric_limits<double>::quiet_NaN();
    double stripCol = std::numeric_limits<double>::quiet_NaN();
    double distance = 0.0;
    FiberSideStripIntersectionSource source =
        FiberSideStripIntersectionSource::FiberSegment;
    cv::Vec3d connectorStart{std::numeric_limits<double>::quiet_NaN(),
                             std::numeric_limits<double>::quiet_NaN(),
                             std::numeric_limits<double>::quiet_NaN()};
    cv::Vec3d projectionTarget{std::numeric_limits<double>::quiet_NaN(),
                               std::numeric_limits<double>::quiet_NaN(),
                               std::numeric_limits<double>::quiet_NaN()};
    size_t branchLinkIndex = std::numeric_limits<size_t>::max();
};

using FiberSideStripCancelCallback = std::function<bool()>;
using FiberSideStripProgressCallback =
    std::function<void(FiberSideStripProgressPhase phase, size_t completed, size_t total)>;

// Where fiber segments and branch-link lines cross a strip surface's
// triangles (invalid strip vertices by the store's surface rule are skipped).
[[nodiscard]] std::vector<FiberSideStripIntersection> fiberSideStripIntersections(
    const FiberSideStripQueryOptions& options,
    FiberSideStripProgressCallback progressCallback = {},
    FiberSideStripCancelCallback cancelCallback = {});

} // namespace vc::fiber_tracer
