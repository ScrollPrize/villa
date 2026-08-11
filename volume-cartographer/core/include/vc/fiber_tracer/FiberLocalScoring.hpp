#pragma once

#include <opencv2/core/types.hpp>

namespace vc::fiber_tracer
{

enum class FiberLocalSmoothnessMode {
    None,
    NormalAware,
    IsotropicFallback,
};

struct FiberLocalSmoothnessConfig {
    float isotropicWeight = 2.0f;
    float normalWeight = 0.1f;
    float tangentWeight = 10.0f;
    float freeAngleRadians = 0.0f;
};

struct FiberLocalSmoothnessCost {
    float isotropic = 0.0f;
    float normal = 0.0f;
    float tangent = 0.0f;
    FiberLocalSmoothnessMode mode = FiberLocalSmoothnessMode::None;

    [[nodiscard]] float total() const noexcept { return isotropic + normal + tangent; }
};

[[nodiscard]] float fiberLocalAlignmentLoss(
    float presence,
    const cv::Vec3f& previousStepDirection,
    const cv::Vec3f& candidateStepDirection,
    const cv::Vec3f& currentPredictionDirection,
    const cv::Vec3f& candidatePredictionDirection);

[[nodiscard]] FiberLocalSmoothnessCost fiberLocalSmoothnessCost(
    const cv::Vec3f& previousStepDirection, const cv::Vec3f& candidateStepDirection, const cv::Vec3f& normal, bool normalValid, const FiberLocalSmoothnessConfig& config);

}  // namespace vc::fiber_tracer
