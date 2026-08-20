#pragma once

#include "vc/fiber_tracer/FiberAnchors.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include <opencv2/core/types.hpp>

namespace vc::fiber_tracer::detail {

struct CompactFiberAnchorObservation {
    cv::Vec3f positionPredictionXYZ{0.0F, 0.0F, 0.0F};
    cv::Vec3f direction{0.0F, 0.0F, 0.0F};
    cv::Vec3f presenceGradientPredictionXYZ{0.0F, 0.0F, 0.0F};
    float presence = 0.0F;
    bool valid = false;
    bool presenceGradientValid = false;
    bool directionUsable = false;
};

struct CompactFiberAnchorProposalObservation {
    cv::Vec3f pivotOffsetPredictionXYZ{0.0F, 0.0F, 0.0F};
    cv::Vec3f direction{0.0F, 0.0F, 0.0F};
    float presence = 0.0F;
    uint32_t logicalIndex = 0;
};

static_assert(sizeof(CompactFiberAnchorProposalObservation) == 32);

struct FiberAnchorObjectiveComponent {
    cv::Vec3f axis{1.0F, 0.0F, 0.0F};
    cv::Vec3f position{0.0F, 0.0F, 0.0F};
};

struct FiberAnchorObjectiveConfig {
    float gaussianSigmaPredictionVoxels = 0.0F;
    float gaussianCutoffSigmas = 0.0F;
    float axialSupportHalfWidthPredictionVoxels = 0.0F;
    float observationPresenceFloor = 0.0F;
};

[[nodiscard]] float retainedSpatialObjectiveExpanded(
    std::span<const FiberAnchorObservation> observations,
    const std::array<FiberAnchorObjectiveComponent, 2>& components,
    size_t activeComponents,
    std::span<const uint8_t> assignments,
    std::span<const uint8_t> retainedInliers,
    const cv::Vec3f& pivot,
    const FiberAnchorObjectiveConfig& config);

[[nodiscard]] std::array<float, 2> retainedSpatialObjectivePairExpanded(
    std::span<const FiberAnchorObservation> observations,
    const std::array<FiberAnchorObjectiveComponent, 2>& first,
    const std::array<FiberAnchorObjectiveComponent, 2>& second,
    size_t activeComponents,
    std::span<const uint8_t> assignments,
    std::span<const uint8_t> retainedInliers,
    const cv::Vec3f& pivot,
    const FiberAnchorObjectiveConfig& config);

[[nodiscard]] float retainedSpatialObjectiveCompact(
    std::span<const CompactFiberAnchorObservation> observationStorage,
    std::span<const uint32_t> observationIndices,
    const std::array<FiberAnchorObjectiveComponent, 2>& components,
    size_t activeComponents,
    std::span<const uint8_t> assignments,
    std::span<const uint8_t> retainedInliers,
    const cv::Vec3f& pivot,
    const FiberAnchorObjectiveConfig& config);

[[nodiscard]] std::array<float, 2> retainedSpatialObjectivePairCompact(
    std::span<const CompactFiberAnchorObservation> observationStorage,
    std::span<const uint32_t> observationIndices,
    const std::array<FiberAnchorObjectiveComponent, 2>& first,
    const std::array<FiberAnchorObjectiveComponent, 2>& second,
    size_t activeComponents,
    std::span<const uint8_t> assignments,
    std::span<const uint8_t> retainedInliers,
    const cv::Vec3f& pivot,
    const FiberAnchorObjectiveConfig& config);

}  // namespace vc::fiber_tracer::detail
