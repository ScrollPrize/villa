#pragma once

#include "vc/fiber_tracer/FiberletCropTrace.hpp"

#include <cstddef>
#include <optional>
#include <span>
#include <vector>

namespace vc::fiber_tracer
{

struct FiberTraceSeedGeometry {
    double straightness = 0.0;
    double arcLengthBaseVoxels = 0.0;
    double centerDistanceBaseVoxels = 0.0;
    bool valid = false;
};

struct FiberTraceSeedGeometryReport {
    std::vector<FiberTraceSeedGeometry> traces;
    double primaryMinimumArcLengthBaseVoxels = 0.0;
};

[[nodiscard]] FiberTraceSeedGeometryReport measureFiberTraceSeedGeometry(
    const std::vector<FiberletCropTraceLine>& traces,
    const cv::Vec3d& cropMinimumBaseXYZ,
    const cv::Vec3d& cropMaximumBaseXYZ);

[[nodiscard]] std::optional<std::size_t> selectCentralStraightFiberTrace(
    const FiberTraceSeedGeometryReport& geometry,
    std::span<const unsigned char> eligible = {},
    bool requirePrimaryLength = true);

}  // namespace vc::fiber_tracer
