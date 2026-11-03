#pragma once

#include "vc/lasagna/LineModel.hpp"

#include <optional>

#include <nlohmann/json.hpp>
#include <opencv2/core/types.hpp>

namespace vc::fiber_tracer
{

inline constexpr double kFiberReplayTangentialThresholdFactor = 4.0;

// Distance travelled after first outside sample, reset by any inside sample.
struct FiberReplayExcursion {
    double graceBaseVoxels = 0;
    std::optional<double> outsideTravel;
    bool observe(bool outside, double stepBaseVoxels);
};

struct FiberReplayThresholdMeasurement {
    double euclideanErrorBaseVoxels = 0.0;
    std::optional<double> normalErrorBaseVoxels;
    std::optional<double> tangentialErrorBaseVoxels;
    double thresholdErrorBaseVoxels = 0.0;
    double thresholdErrorRatio = 0.0;
    bool localNormalValid = false;
    double tangentialFactor = kFiberReplayTangentialThresholdFactor;
};

[[nodiscard]] FiberReplayThresholdMeasurement measureFiberReplayThreshold(
    const cv::Vec3d& evaluatorPointBase,
    const cv::Vec3d& matchedReferencePointBase,
    const vc::lasagna::NormalSampler& normalSampler,
    double normalWorkingToBaseScale,
    double normalThresholdBaseVoxels,
    double tangentialFactor = kFiberReplayTangentialThresholdFactor);

[[nodiscard]] bool fiberReplayThresholdExceeded(
    const FiberReplayThresholdMeasurement& measurement,
    double normalThresholdBaseVoxels);

[[nodiscard]] double fiberReplayTangentialThresholdBaseVoxels(
    double normalThresholdBaseVoxels);

void validateFiberReplayThresholdMeasurement(
    const FiberReplayThresholdMeasurement& measurement,
    double normalThresholdBaseVoxels);

[[nodiscard]] nlohmann::json fiberReplayThresholdDescriptorJson(
    double normalThresholdBaseVoxels,
    double tangentialFactor = kFiberReplayTangentialThresholdFactor);

[[nodiscard]] nlohmann::json fiberReplayThresholdMeasurementJson(
    const FiberReplayThresholdMeasurement& measurement);

[[nodiscard]] nlohmann::json fiberReplayOptionalThresholdMeasurementJson(
    const std::optional<FiberReplayThresholdMeasurement>& measurement,
    double normalThresholdBaseVoxels);

}  // namespace vc::fiber_tracer
