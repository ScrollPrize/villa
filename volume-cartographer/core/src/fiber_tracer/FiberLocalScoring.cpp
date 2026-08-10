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

}  // namespace vc::fiber_tracer
