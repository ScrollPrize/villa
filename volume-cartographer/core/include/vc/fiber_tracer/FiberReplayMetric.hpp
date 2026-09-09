#pragma once

#include "vc/lasagna/LineModel.hpp"

#include <optional>

#include <nlohmann/json.hpp>
#include <opencv2/core/types.hpp>

namespace vc::fiber_tracer
{

inline constexpr double kFiberReplayTangentialThresholdFactor = 4.0;

struct FiberReplayThresholdMeasurement {
    double euclideanErrorBaseVoxels = 0.0;
    std::optional<double> normalErrorBaseVoxels;
    std::optional<double> tangentialErrorBaseVoxels;
    double thresholdErrorBaseVoxels = 0.0;
    double thresholdErrorRatio = 0.0;
    bool localNormalValid = false;
};

[[nodiscard]] FiberReplayThresholdMeasurement measureFiberReplayThreshold(
    const cv::Vec3d& evaluatorPointBase,
    const cv::Vec3d& matchedReferencePointBase,
    const vc::lasagna::NormalSampler& normalSampler,
    double normalWorkingToBaseScale,
    double normalThresholdBaseVoxels);

[[nodiscard]] bool fiberReplayThresholdExceeded(
    const FiberReplayThresholdMeasurement& measurement,
    double normalThresholdBaseVoxels);

[[nodiscard]] double fiberReplayTangentialThresholdBaseVoxels(
    double normalThresholdBaseVoxels);

void validateFiberReplayThresholdMeasurement(
    const FiberReplayThresholdMeasurement& measurement,
    double normalThresholdBaseVoxels);

[[nodiscard]] nlohmann::json fiberReplayThresholdDescriptorJson(
    double normalThresholdBaseVoxels);

[[nodiscard]] nlohmann::json fiberReplayThresholdMeasurementJson(
    const FiberReplayThresholdMeasurement& measurement);

[[nodiscard]] nlohmann::json fiberReplayOptionalThresholdMeasurementJson(
    const std::optional<FiberReplayThresholdMeasurement>& measurement,
    double normalThresholdBaseVoxels);

}  // namespace vc::fiber_tracer
