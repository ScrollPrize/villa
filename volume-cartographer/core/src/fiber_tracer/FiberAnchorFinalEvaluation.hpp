#pragma once

#include "FiberAnchorObjectives.hpp"

namespace vc::fiber_tracer::detail {

struct FiberAnchorFinalEvaluation {
    std::array<float, 2> denominators{0.0F, 0.0F};
    std::array<float, 2> numerators{0.0F, 0.0F};
    std::array<float, 2> presenceMasses{0.0F, 0.0F};
    std::array<float, 2> alignedSupports{0.0F, 0.0F};
    std::array<float, 2> directionalCoherences{0.0F, 0.0F};
    std::array<size_t, 2> assignedCounts{0, 0};
    float objective = 0.0F;
};

[[nodiscard]] FiberAnchorFinalEvaluation finalAnchorEvaluationExpanded(
    std::span<const FiberAnchorObservation> observations,
    const std::array<FiberAnchorObjectiveComponent, 2>& components,
    size_t activeComponents,
    std::span<const uint8_t> assignments,
    std::span<const uint8_t> retainedInliers,
    const cv::Vec3d& pivot,
    const FiberAnchorObjectiveConfig& config);

[[nodiscard]] FiberAnchorFinalEvaluation finalAnchorEvaluationCompact(
    std::span<const CompactFiberAnchorObservation> observationStorage,
    std::span<const uint32_t> observationIndices,
    const std::array<FiberAnchorObjectiveComponent, 2>& components,
    size_t activeComponents,
    std::span<const uint8_t> assignments,
    std::span<const uint8_t> retainedInliers,
    const cv::Vec3d& pivot,
    const FiberAnchorObjectiveConfig& config);

}  // namespace vc::fiber_tracer::detail
