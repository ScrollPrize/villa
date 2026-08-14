#include "vc/fiber_tracer/FiberLocalScoring.hpp"

#include <algorithm>
#include <cmath>

namespace vc::fiber_tracer
{
namespace
{

constexpr float kEpsilon = 1.0e-6f;

float clampUnit(float value)
{
    if (!std::isfinite(value))
        return 0.0f;
    return std::clamp(value, -1.0f, 1.0f);
}

float clampPositiveUnit(float value)
{
    if (!std::isfinite(value))
        return 0.0f;
    return std::clamp(value, 0.0f, 1.0f);
}

cv::Vec3f normalizedOrZero(const cv::Vec3f& value)
{
    const float length = std::sqrt(value.dot(value));
    if (!(length > kEpsilon) || !std::isfinite(length))
        return {0.0f, 0.0f, 0.0f};
    return value / length;
}

float angleBetweenUnit(const cv::Vec3f& left, const cv::Vec3f& right)
{
    return std::acos(clampUnit(left.dot(right)));
}

float excessAngleSquared(float angle, float freeAngle)
{
    const float excess = std::max(0.0f, angle - freeAngle);
    return excess * excess;
}

}  // namespace

float fiberLocalAlignmentLoss(
    float presence,
    const cv::Vec3f& previousStepDirection,
    const cv::Vec3f& candidateStepDirection,
    const cv::Vec3f& currentPredictionDirection,
    const cv::Vec3f& candidatePredictionDirection)
{
    float score = clampPositiveUnit(presence);
    score *= clampPositiveUnit(previousStepDirection.dot(candidateStepDirection));
    score *= clampPositiveUnit(previousStepDirection.dot(currentPredictionDirection));
    score *= clampPositiveUnit(previousStepDirection.dot(candidatePredictionDirection));
    score *= clampPositiveUnit(currentPredictionDirection.dot(candidateStepDirection));
    score *= clampPositiveUnit(currentPredictionDirection.dot(candidatePredictionDirection));
    score *= clampPositiveUnit(candidateStepDirection.dot(candidatePredictionDirection));
    return 1.0f - score;
}

FiberLocalSmoothnessCost fiberLocalSmoothnessCost(
    const cv::Vec3f& previousStepDirection, const cv::Vec3f& candidateStepDirection, const cv::Vec3f& normal, bool normalValid, const FiberLocalSmoothnessConfig& config)
{
    FiberLocalSmoothnessCost cost;
    constexpr float epsilon2 = kEpsilon * kEpsilon;
    if (previousStepDirection.dot(previousStepDirection) <= epsilon2 || candidateStepDirection.dot(candidateStepDirection) <= epsilon2) {
        return cost;
    }

    const float isotropicAngle = angleBetweenUnit(previousStepDirection, candidateStepDirection);
    const float isotropic = config.isotropicWeight * excessAngleSquared(isotropicAngle, config.freeAngleRadians);
    if (!normalValid || normal.dot(normal) <= epsilon2) {
        cost.isotropic = isotropic;
        cost.mode = FiberLocalSmoothnessMode::IsotropicFallback;
        return cost;
    }

    const float previousNormal = clampUnit(previousStepDirection.dot(normal));
    const float candidateNormal = clampUnit(candidateStepDirection.dot(normal));
    const cv::Vec3f previousTangent = normalizedOrZero(previousStepDirection - normal * previousNormal);
    const cv::Vec3f candidateTangent = normalizedOrZero(candidateStepDirection - normal * candidateNormal);
    const bool tangentValid = previousTangent.dot(previousTangent) > epsilon2 && candidateTangent.dot(candidateTangent) > epsilon2;
    const float tangentAngle = tangentValid ? angleBetweenUnit(previousTangent, candidateTangent) : isotropicAngle;
    const float normalAngle = std::abs(std::asin(candidateNormal) - std::asin(previousNormal));
    cost.tangent = config.tangentWeight * excessAngleSquared(tangentAngle, config.freeAngleRadians);
    cost.normal = config.normalWeight * excessAngleSquared(normalAngle, config.freeAngleRadians);
    cost.mode = FiberLocalSmoothnessMode::NormalAware;
    return cost;
}

FiberLocalMetricCost fiberLocalMetricCost(
    const FiberLocalMetricSample* currentPrediction,
    const FiberLocalMetricSample& candidatePrediction,
    const cv::Vec3f& previousStepDirection,
    float previousStepLength,
    const cv::Vec3f& candidateStepDirection,
    float candidateStepLength,
    const cv::Vec3f& normal,
    bool normalValid,
    const FiberLocalMetricConfig& config)
{
    FiberLocalMetricCost cost;
    if (!candidatePrediction.valid) {
        cost.invalidPrediction = config.invalidPredictionCostPerVoxel * std::max(0.0f, candidateStepLength);
        return cost;
    }
    const cv::Vec3f previous = normalizedOrZero(previousStepDirection);
    const cv::Vec3f candidate = normalizedOrZero(candidateStepDirection);
    cv::Vec3f currentAxis = previous;
    if (currentPrediction != nullptr && currentPrediction->valid) {
        currentAxis = normalizedOrZero(currentPrediction->direction);
        if (currentAxis.dot(previous) < 0.0f)
            currentAxis *= -1.0f;
    }
    cv::Vec3f candidateAxis = normalizedOrZero(candidatePrediction.direction);
    if (candidateAxis.dot(candidate) < 0.0f)
        candidateAxis *= -1.0f;
    cost.alignment = fiberLocalAlignmentLoss(candidatePrediction.presence, previous, candidate, currentAxis, candidateAxis) *
                     std::max(0.0f, candidateStepLength);
    const auto smoothness = fiberLocalSmoothnessCost(previous, candidate, normal, normalValid, config.smoothness);
    const float effectiveLength = std::max(1.0f, (std::max(0.0f, previousStepLength) + std::max(0.0f, candidateStepLength)) * 0.5f);
    cost.isotropicSmoothness = smoothness.isotropic / effectiveLength;
    cost.tangentSmoothness = smoothness.tangent / effectiveLength;
    cost.normalSmoothness = smoothness.normal / effectiveLength;
    return cost;
}

}  // namespace vc::fiber_tracer
