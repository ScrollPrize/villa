#pragma once

#include <opencv2/core.hpp>

#include <cstdint>
#include <vector>

// Seeds for bend surfaces: places where fibers of one family (horizontal or
// vertical) that lie close together run in clearly different directions. In a
// smoothly curved sheet nearby fibers of a family are near parallel; across a
// sharp bend they are not. Each seed is one such pair: its midpoint lies near
// the bend surface and its two directions are the fiber directions on either
// side. Directions are lines (sign-free), so a seed does not say which way
// round the bend turns.
namespace vc::fiber_tracer
{

struct FiberBendNode {
    cv::Vec3d point{0.0, 0.0, 0.0};
    cv::Vec3d tangent{0.0, 0.0, 0.0};  // unit; a zero tangent excludes the node
    std::int64_t fiber = 0;
    int family = 0;
};

struct FiberBendSeed {
    cv::Vec3d point{0.0, 0.0, 0.0};  // midpoint of the two nodes
    cv::Vec3d directionA{0.0, 0.0, 0.0};
    cv::Vec3d directionB{0.0, 0.0, 0.0};
    double angleDegrees = 0.0;  // between the two lines, in [0, 90]
    double distance = 0.0;      // between the two nodes
    int family = 0;
    std::int64_t fiberA = 0;
    std::int64_t fiberB = 0;
};

struct FiberBendSeedOptions {
    double radius = 100.0;          // largest node distance of a pair
    double minAngleDegrees = 45.0;  // smallest angle between the two lines
    unsigned threads = 0;           // 0: all hardware threads
};

// Every pair of nodes of the same family and different fibers within `radius`
// whose tangent lines differ by at least `minAngleDegrees`. Ordered by the
// first node's index, so the result does not depend on the thread count.
[[nodiscard]] std::vector<FiberBendSeed> fiberBendSeeds(const std::vector<FiberBendNode>& nodes,
                                                        const FiberBendSeedOptions& options = {});

}  // namespace vc::fiber_tracer
