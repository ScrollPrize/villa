#pragma once
#include "vc/fiber_tracer/FiberReplayMetric.hpp"
#include <span>
#include <vector>

namespace vc::fiber_tracer {
// Independent placement diagnostic: nearest GT projection, flat end caps,
// shared normal/tangent ellipse. Input anchors must have unique stable IDs.
[[nodiscard]] nlohmann::json measureFiberAnchorCoverage(
    const std::vector<cv::Vec3d>& referenceBase,
    std::span<const cv::Vec3d> anchorsBase,
    const vc::lasagna::NormalSampler& normals, double normalWorkingToBaseScale,
    double normalRadiusBase, double tangentialFactor, double cellSideBase,
    std::span<const cv::Vec3d> fittedAxes = {});
}
