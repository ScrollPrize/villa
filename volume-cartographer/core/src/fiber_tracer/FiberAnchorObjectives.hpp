#pragma once

#include "vc/fiber_tracer/FiberAnchors.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include <opencv2/core/types.hpp>

namespace vc::fiber_tracer::detail {

struct CompensatedSum {
    void add(double value)
    {
        const double adjusted = value - correction;
        const double next = sum + adjusted;
        correction = (next - sum) - adjusted;
        sum = next;
    }

    double sum = 0.0;
    double correction = 0.0;
};

struct CompactFiberAnchorObservation {
    cv::Vec3f positionPredictionXYZ{0.0F, 0.0F, 0.0F};
    cv::Vec3f direction{0.0F, 0.0F, 0.0F};
    cv::Vec3f presenceGradientPredictionXYZ{0.0F, 0.0F, 0.0F};
    float presence = 0.0F;
    bool valid = false;
    bool presenceGradientValid = false;
};

struct FiberAnchorObjectiveComponent {
    cv::Vec3d axis{1.0, 0.0, 0.0};
    cv::Vec3d position{0.0, 0.0, 0.0};
};

struct FiberAnchorObjectiveConfig {
    double gaussianSigmaPredictionVoxels = 0.0;
    double gaussianCutoffSigmas = 0.0;
    double axialSupportHalfWidthPredictionVoxels = 0.0;
    double observationPresenceFloor = 0.0;
};

[[nodiscard]] double retainedSpatialObjectiveExpanded(
    std::span<const FiberAnchorObservation> observations,
    const std::array<FiberAnchorObjectiveComponent, 2>& components,
    size_t activeComponents,
    std::span<const uint8_t> assignments,
    std::span<const uint8_t> retainedInliers,
    const cv::Vec3d& pivot,
    const FiberAnchorObjectiveConfig& config);

[[nodiscard]] std::array<double, 2> retainedSpatialObjectivePairExpanded(
    std::span<const FiberAnchorObservation> observations,
    const std::array<FiberAnchorObjectiveComponent, 2>& first,
    const std::array<FiberAnchorObjectiveComponent, 2>& second,
    size_t activeComponents,
    std::span<const uint8_t> assignments,
    std::span<const uint8_t> retainedInliers,
    const cv::Vec3d& pivot,
    const FiberAnchorObjectiveConfig& config);

[[nodiscard]] double retainedSpatialObjectiveCompact(
    std::span<const CompactFiberAnchorObservation> observationStorage,
    std::span<const uint32_t> observationIndices,
    const std::array<FiberAnchorObjectiveComponent, 2>& components,
    size_t activeComponents,
    std::span<const uint8_t> assignments,
    std::span<const uint8_t> retainedInliers,
    const cv::Vec3d& pivot,
    const FiberAnchorObjectiveConfig& config);

[[nodiscard]] std::array<double, 2> retainedSpatialObjectivePairCompact(
    std::span<const CompactFiberAnchorObservation> observationStorage,
    std::span<const uint32_t> observationIndices,
    const std::array<FiberAnchorObjectiveComponent, 2>& first,
    const std::array<FiberAnchorObjectiveComponent, 2>& second,
    size_t activeComponents,
    std::span<const uint8_t> assignments,
    std::span<const uint8_t> retainedInliers,
    const cv::Vec3d& pivot,
    const FiberAnchorObjectiveConfig& config);

}  // namespace vc::fiber_tracer::detail
