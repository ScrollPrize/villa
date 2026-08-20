#include "FiberAnchorFinalEvaluationKernels.hpp"

namespace vc::fiber_tracer::detail {
namespace {

using namespace final_evaluation_kernel;

template <typename Range>
[[nodiscard]] FiberAnchorFinalEvaluation evaluateFinal(
    const Range& observations,
    const std::array<FiberAnchorObjectiveComponent, 2>& inputComponents,
    size_t activeComponents,
    std::span<const uint8_t> assignments,
    std::span<const uint8_t> retainedInliers,
    const cv::Vec3f& inputPivot,
    const FiberAnchorObjectiveConfig& inputConfig)
{
    static_assert(std::is_same_v<typename Range::Scalar, float>);
    validateInputs(
        observations, activeComponents, assignments, retainedInliers);
    std::array<ObjectiveComponent<float>, 2> components;
    for (size_t component = 0; component < components.size(); ++component) {
        components[component].axis = inputComponents[component].axis;
        components[component].position = inputComponents[component].position;
    }
    const auto pivot = inputPivot;
    const auto config = floatConfig(inputConfig);
    FiberAnchorFinalEvaluation result;
    for (size_t logicalIndex = 0; logicalIndex < observations.size();
         ++logicalIndex) {
        const auto& observation = observations[logicalIndex];
        std::array<float, 2> gaussian{};
        for (size_t component = 0; component < activeComponents; ++component) {
            gaussian[component] = transverseGaussian<Range>(
                observation, components[component], pivot, config);
            result.denominators[component] += gaussian[component];
        }
        const uint8_t assigned = assignments[logicalIndex];
        if (!retainedInliers[logicalIndex] || assigned >= activeComponents)
            continue;
        cv::Vec3f direction;
        if (!usableDirection<Range>(observation, config.presenceFloor, direction))
            continue;
        const float presence = observation.presence;
        const float dot = direction.dot(components[assigned].axis);
        result.numerators[assigned] +=
            gaussian[assigned] * presence * dot * dot;
        result.presenceMasses[assigned] += gaussian[assigned] * presence;
        ++result.assignedCounts[assigned];
    }
    float numeratorTotal = 0.0F;
    float denominatorTotal = 0.0F;
    for (size_t component = 0; component < activeComponents; ++component) {
        numeratorTotal += result.numerators[component];
        denominatorTotal += result.denominators[component];
        if (result.denominators[component] > 0.0F) {
            result.alignedSupports[component] =
                result.numerators[component] / result.denominators[component];
        }
        if (result.presenceMasses[component] > 0.0F) {
            result.directionalCoherences[component] =
                result.numerators[component] / result.presenceMasses[component];
        }
    }
    if (denominatorTotal > 0.0F)
        result.objective = numeratorTotal / denominatorTotal;
    return result;
}

}  // namespace

FiberAnchorFinalEvaluation finalAnchorEvaluationExpanded(
    std::span<const FiberAnchorObservation> observations,
    const std::array<FiberAnchorObjectiveComponent, 2>& components,
    size_t activeComponents,
    std::span<const uint8_t> assignments,
    std::span<const uint8_t> retainedInliers,
    const cv::Vec3f& pivot,
    const FiberAnchorObjectiveConfig& config)
{
    return evaluateFinal(
        ExpandedFinalObservationRange{observations}, components,
        activeComponents, assignments, retainedInliers, pivot, config);
}

FiberAnchorFinalEvaluation finalAnchorEvaluationCompact(
    std::span<const CompactFiberAnchorObservation> observationStorage,
    std::span<const uint32_t> observationIndices,
    const std::array<FiberAnchorObjectiveComponent, 2>& components,
    size_t activeComponents,
    std::span<const uint8_t> assignments,
    std::span<const uint8_t> retainedInliers,
    const cv::Vec3f& pivot,
    const FiberAnchorObjectiveConfig& config)
{
    return evaluateFinal(
        CompactObservationRange{observationStorage, observationIndices},
        components, activeComponents, assignments, retainedInliers, pivot,
        config);
}

}  // namespace vc::fiber_tracer::detail
