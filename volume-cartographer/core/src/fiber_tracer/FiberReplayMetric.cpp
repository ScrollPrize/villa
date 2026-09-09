#include "vc/fiber_tracer/FiberReplayMetric.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace vc::fiber_tracer
{
namespace
{

constexpr double kEpsilon = 1.0e-12;

bool finiteVector(const cv::Vec3d& value)
{
    return std::isfinite(value[0]) && std::isfinite(value[1]) &&
           std::isfinite(value[2]);
}

bool nearlyEqual(double left, double right)
{
    return std::abs(left - right) <=
        1.0e-9 * std::max({1.0, std::abs(left), std::abs(right)});
}

double tangentialThreshold(double normalThreshold)
{
    if (!(normalThreshold >= 0.0) || !std::isfinite(normalThreshold) ||
        normalThreshold > std::numeric_limits<double>::max() /
                kFiberReplayTangentialThresholdFactor) {
        throw std::invalid_argument(
            "fiber replay normal threshold must be finite, non-negative, and have a finite tangential radius");
    }
    return normalThreshold * kFiberReplayTangentialThresholdFactor;
}

double thresholdRatio(double thresholdError, double normalThreshold)
{
    if (thresholdError == 0.0)
        return 0.0;
    if (normalThreshold > 0.0) {
        const double ratio = thresholdError / normalThreshold;
        return std::isfinite(ratio)
            ? ratio
            : std::numeric_limits<double>::max();
    }
    return std::numeric_limits<double>::max();
}

}  // namespace

FiberReplayThresholdMeasurement measureFiberReplayThreshold(
    const cv::Vec3d& evaluatorPointBase,
    const cv::Vec3d& matchedReferencePointBase,
    const vc::lasagna::NormalSampler& normalSampler,
    double normalWorkingToBaseScale,
    double normalThresholdBaseVoxels)
{
    (void)tangentialThreshold(normalThresholdBaseVoxels);
    if (!(normalWorkingToBaseScale > 0.0) ||
        !std::isfinite(normalWorkingToBaseScale) ||
        !finiteVector(evaluatorPointBase) ||
        !finiteVector(matchedReferencePointBase)) {
        throw std::invalid_argument(
            "fiber replay threshold coordinates and normal scale must be finite");
    }

    const cv::Vec3d delta = evaluatorPointBase - matchedReferencePointBase;
    const double euclideanSquared = std::max(0.0, delta.dot(delta));
    FiberReplayThresholdMeasurement result;
    result.euclideanErrorBaseVoxels = std::sqrt(euclideanSquared);

    const auto sampled = normalSampler.sampleNormal(
        matchedReferencePointBase * (1.0 / normalWorkingToBaseScale));
    const double normalLength = finiteVector(sampled.normal)
        ? cv::norm(sampled.normal)
        : 0.0;
    result.localNormalValid = sampled.valid &&
        std::isfinite(normalLength) && normalLength > kEpsilon;
    if (result.localNormalValid) {
        const cv::Vec3d normal = sampled.normal * (1.0 / normalLength);
        const double normalError = std::abs(delta.dot(normal));
        const double tangentialSquared =
            std::max(0.0, euclideanSquared - normalError * normalError);
        const double tangentialError = std::sqrt(tangentialSquared);
        result.normalErrorBaseVoxels = normalError;
        result.tangentialErrorBaseVoxels = tangentialError;
        result.thresholdErrorBaseVoxels = std::hypot(
            normalError,
            tangentialError / kFiberReplayTangentialThresholdFactor);
    } else {
        result.thresholdErrorBaseVoxels = result.euclideanErrorBaseVoxels;
    }
    result.thresholdErrorRatio = thresholdRatio(
        result.thresholdErrorBaseVoxels, normalThresholdBaseVoxels);
    validateFiberReplayThresholdMeasurement(result, normalThresholdBaseVoxels);
    return result;
}

bool fiberReplayThresholdExceeded(
    const FiberReplayThresholdMeasurement& measurement,
    double normalThresholdBaseVoxels)
{
    validateFiberReplayThresholdMeasurement(
        measurement, normalThresholdBaseVoxels);
    return measurement.thresholdErrorBaseVoxels > normalThresholdBaseVoxels;
}

double fiberReplayTangentialThresholdBaseVoxels(
    double normalThresholdBaseVoxels)
{
    return tangentialThreshold(normalThresholdBaseVoxels);
}

void validateFiberReplayThresholdMeasurement(
    const FiberReplayThresholdMeasurement& measurement,
    double normalThresholdBaseVoxels)
{
    (void)tangentialThreshold(normalThresholdBaseVoxels);
    const auto finiteNonNegative = [](double value) {
        return std::isfinite(value) && value >= 0.0;
    };
    if (!finiteNonNegative(measurement.euclideanErrorBaseVoxels) ||
        !finiteNonNegative(measurement.thresholdErrorBaseVoxels) ||
        !finiteNonNegative(measurement.thresholdErrorRatio)) {
        throw std::invalid_argument(
            "fiber replay threshold measurement is not finite and non-negative");
    }

    double expectedThresholdError = measurement.euclideanErrorBaseVoxels;
    if (measurement.localNormalValid) {
        if (!measurement.normalErrorBaseVoxels.has_value() ||
            !measurement.tangentialErrorBaseVoxels.has_value() ||
            !finiteNonNegative(*measurement.normalErrorBaseVoxels) ||
            !finiteNonNegative(*measurement.tangentialErrorBaseVoxels)) {
            throw std::invalid_argument(
                "valid replay threshold normal requires component errors");
        }
        const double componentMagnitude = std::hypot(
            *measurement.normalErrorBaseVoxels,
            *measurement.tangentialErrorBaseVoxels);
        if (!nearlyEqual(
                componentMagnitude,
                measurement.euclideanErrorBaseVoxels)) {
            throw std::invalid_argument(
                "replay threshold components do not reconstruct Euclidean error");
        }
        expectedThresholdError = std::hypot(
            *measurement.normalErrorBaseVoxels,
            *measurement.tangentialErrorBaseVoxels /
                kFiberReplayTangentialThresholdFactor);
    } else if (measurement.normalErrorBaseVoxels.has_value() ||
               measurement.tangentialErrorBaseVoxels.has_value()) {
        throw std::invalid_argument(
            "invalid replay threshold normal must not have component errors");
    }

    if (!nearlyEqual(
            expectedThresholdError,
            measurement.thresholdErrorBaseVoxels) ||
        !nearlyEqual(
            thresholdRatio(expectedThresholdError, normalThresholdBaseVoxels),
            measurement.thresholdErrorRatio)) {
        throw std::invalid_argument(
            "replay threshold error or ratio is inconsistent");
    }
}

nlohmann::json fiberReplayThresholdDescriptorJson(
    double normalThresholdBaseVoxels)
{
    return {
        {"shape", "lasagna_normal_ellipsoid"},
        {"normal_radius_base_voxels", normalThresholdBaseVoxels},
        {"tangential_factor", kFiberReplayTangentialThresholdFactor},
        {"tangential_radius_base_voxels",
         tangentialThreshold(normalThresholdBaseVoxels)},
        {"comparison", "threshold_error_strictly_greater"},
        {"invalid_normal_policy", "isotropic_euclidean"},
    };
}

nlohmann::json fiberReplayThresholdMeasurementJson(
    const FiberReplayThresholdMeasurement& measurement)
{
    return {
        {"euclidean_error_base_voxels",
         measurement.euclideanErrorBaseVoxels},
        {"normal_error_base_voxels",
         measurement.normalErrorBaseVoxels.has_value()
             ? nlohmann::json(*measurement.normalErrorBaseVoxels)
             : nlohmann::json(nullptr)},
        {"tangential_error_base_voxels",
         measurement.tangentialErrorBaseVoxels.has_value()
             ? nlohmann::json(*measurement.tangentialErrorBaseVoxels)
             : nlohmann::json(nullptr)},
        {"threshold_error_base_voxels",
         measurement.thresholdErrorBaseVoxels},
        {"threshold_error_ratio", measurement.thresholdErrorRatio},
        {"local_normal_valid", measurement.localNormalValid},
    };
}

nlohmann::json fiberReplayOptionalThresholdMeasurementJson(
    const std::optional<FiberReplayThresholdMeasurement>& measurement,
    double normalThresholdBaseVoxels)
{
    if (measurement.has_value()) {
        validateFiberReplayThresholdMeasurement(
            *measurement, normalThresholdBaseVoxels);
        return fiberReplayThresholdMeasurementJson(*measurement);
    }
    return {
        {"euclidean_error_base_voxels", nullptr},
        {"normal_error_base_voxels", nullptr},
        {"tangential_error_base_voxels", nullptr},
        {"threshold_error_base_voxels", nullptr},
        {"threshold_error_ratio", nullptr},
        {"local_normal_valid", nullptr},
    };
}

}  // namespace vc::fiber_tracer
