#pragma once

#include "vc/fiber_tracer/FiberletStorage.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core/types.hpp>

namespace vc::fiber_tracer
{

struct FiberletCropTraceLine {
    FiberletStorageKey seed;
    cv::Vec3d seedBaseXYZ{0.0, 0.0, 0.0};
    std::optional<cv::Vec3d> originalSeedBaseXYZ;
    float seedPresence = 0.0F;
    double totalMetricCost = 0.0;
    double pathLengthPredictionVoxels = 0.0;
    std::vector<cv::Vec3d> pointsBaseXYZ;
    std::vector<double> segmentMetricCosts;
    std::vector<double> segmentOriginalStepCosts;
    std::string negativeTermination;
    std::string positiveTermination;
    std::size_t negativeFiberlets = 0;
    std::size_t positiveFiberlets = 0;
};

} // namespace vc::fiber_tracer
