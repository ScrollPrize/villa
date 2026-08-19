#pragma once

#include "vc/fiber_tracer/FiberLocalScoring.hpp"

#include <algorithm>
#include <cmath>

namespace vc::fiber_tracer::detail
{

inline constexpr float kFiberLocalEpsilon = 1.0e-6f;

static inline float clampFiberLocalUnit(float value)
{
    if (!std::isfinite(value))
        return 0.0f;
    return std::clamp(value, -1.0f, 1.0f);
}

static inline float clampFiberLocalPositiveUnit(float value)
{
    if (!std::isfinite(value))
        return 0.0f;
    return std::clamp(value, 0.0f, 1.0f);
}

static inline cv::Vec3f normalizeFiberLocalOrZero(const cv::Vec3f& value)
{
    const float length = std::sqrt(value.dot(value));
    if (!(length > kFiberLocalEpsilon) || !std::isfinite(length))
        return {0.0f, 0.0f, 0.0f};
    return value / length;
}

static inline float fiberLocalAngleBetweenUnit(
    const cv::Vec3f& left, const cv::Vec3f& right)
{
    return std::acos(clampFiberLocalUnit(left.dot(right)));
}

static inline float fiberLocalExcessAngleSquared(float angle, float freeAngle)
{
    const float excess = std::max(0.0f, angle - freeAngle);
    return excess * excess;
}

static inline float fiberLocalAlignmentLossInline(
    float presence,
    const cv::Vec3f& previousStepDirection,
    const cv::Vec3f& candidateStepDirection,
    const cv::Vec3f& currentPredictionDirection,
    const cv::Vec3f& candidatePredictionDirection)
{
    float score = clampFiberLocalPositiveUnit(presence);
    score *= clampFiberLocalPositiveUnit(
        previousStepDirection.dot(candidateStepDirection));
    score *= clampFiberLocalPositiveUnit(
        previousStepDirection.dot(currentPredictionDirection));
    score *= clampFiberLocalPositiveUnit(
        previousStepDirection.dot(candidatePredictionDirection));
    score *= clampFiberLocalPositiveUnit(
        currentPredictionDirection.dot(candidateStepDirection));
    score *= clampFiberLocalPositiveUnit(
        currentPredictionDirection.dot(candidatePredictionDirection));
    score *= clampFiberLocalPositiveUnit(
        candidateStepDirection.dot(candidatePredictionDirection));
    return 1.0f - score;
}

static inline FiberLocalSmoothnessCost fiberLocalSmoothnessCostInline(
    const cv::Vec3f& previousStepDirection,
    const cv::Vec3f& candidateStepDirection,
    const cv::Vec3f& normal,
    bool normalValid,
    const FiberLocalSmoothnessConfig& config)
{
    FiberLocalSmoothnessCost cost;
    constexpr float epsilon2 = kFiberLocalEpsilon * kFiberLocalEpsilon;
    if (previousStepDirection.dot(previousStepDirection) <= epsilon2 ||
        candidateStepDirection.dot(candidateStepDirection) <= epsilon2) {
        return cost;
    }

    if (!normalValid || normal.dot(normal) <= epsilon2) {
        const float isotropicAngle = fiberLocalAngleBetweenUnit(
            previousStepDirection, candidateStepDirection);
        cost.isotropic = config.isotropicWeight *
                         fiberLocalExcessAngleSquared(
                             isotropicAngle, config.freeAngleRadians);
        cost.mode = FiberLocalSmoothnessMode::IsotropicFallback;
        return cost;
    }

    const float previousNormal = clampFiberLocalUnit(
        previousStepDirection.dot(normal));
    const float candidateNormal = clampFiberLocalUnit(
        candidateStepDirection.dot(normal));
    const cv::Vec3f previousTangent = normalizeFiberLocalOrZero(
        previousStepDirection - normal * previousNormal);
    const cv::Vec3f candidateTangent = normalizeFiberLocalOrZero(
        candidateStepDirection - normal * candidateNormal);
    const bool tangentValid =
        previousTangent.dot(previousTangent) > epsilon2 &&
        candidateTangent.dot(candidateTangent) > epsilon2;
    const float tangentAngle = tangentValid
        ? fiberLocalAngleBetweenUnit(previousTangent, candidateTangent)
        : fiberLocalAngleBetweenUnit(
              previousStepDirection, candidateStepDirection);
    const float normalAngle = std::abs(
        std::asin(candidateNormal) - std::asin(previousNormal));
    cost.tangent = config.tangentWeight * fiberLocalExcessAngleSquared(
        tangentAngle, config.freeAngleRadians);
    cost.normal = config.normalWeight * fiberLocalExcessAngleSquared(
        normalAngle, config.freeAngleRadians);
    cost.mode = FiberLocalSmoothnessMode::NormalAware;
    return cost;
}

static inline FiberLocalMetricCost fiberLocalMetricCostPreparedInline(
    const FiberLocalMetricSample* currentPrediction,
    const FiberLocalMetricSample& candidatePrediction,
    const cv::Vec3f& previousStepUnitDirection,
    float previousStepLength,
    const cv::Vec3f& candidateStepUnitDirection,
    float candidateStepLength,
    const cv::Vec3f& normal,
    bool normalValid,
    const FiberLocalMetricConfig& config)
{
    FiberLocalMetricCost cost;
    if (!candidatePrediction.valid) {
        cost.invalidPrediction = config.invalidPredictionCostPerVoxel *
                                 std::max(0.0f, candidateStepLength);
        return cost;
    }
    const cv::Vec3f& previous = previousStepUnitDirection;
    const cv::Vec3f& candidate = candidateStepUnitDirection;
    cv::Vec3f currentAxis = previous;
    if (currentPrediction != nullptr && currentPrediction->valid) {
        currentAxis = currentPrediction->direction;
        if (currentAxis.dot(previous) < 0.0f)
            currentAxis *= -1.0f;
    }
    cv::Vec3f candidateAxis = candidatePrediction.direction;
    if (candidateAxis.dot(candidate) < 0.0f)
        candidateAxis *= -1.0f;
    cost.alignment = fiberLocalAlignmentLossInline(
        candidatePrediction.presence, previous, candidate,
        currentAxis, candidateAxis) * std::max(0.0f, candidateStepLength);
    const auto smoothness = fiberLocalSmoothnessCostInline(
        previous, candidate, normal, normalValid, config.smoothness);
    const float effectiveLength = std::max(
        1.0f,
        (std::max(0.0f, previousStepLength) +
         std::max(0.0f, candidateStepLength)) * 0.5f);
    cost.isotropicSmoothness = smoothness.isotropic / effectiveLength;
    cost.tangentSmoothness = smoothness.tangent / effectiveLength;
    cost.normalSmoothness = smoothness.normal / effectiveLength;
    return cost;
}

}  // namespace vc::fiber_tracer::detail
