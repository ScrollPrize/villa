#include "FiberAnchorObjectives.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <type_traits>

namespace vc::fiber_tracer::detail {
namespace {

constexpr double kMatrixEpsilon = 1.0e-15;

template <typename Scalar>
[[nodiscard]] bool finiteVector(const cv::Vec<Scalar, 3>& value)
{
    return std::isfinite(value[0]) && std::isfinite(value[1]) &&
           std::isfinite(value[2]);
}

template <typename Scalar>
[[nodiscard]] cv::Vec<Scalar, 3> converted(const cv::Vec3d& value)
{
    return {
        static_cast<Scalar>(value[0]),
        static_cast<Scalar>(value[1]),
        static_cast<Scalar>(value[2]),
    };
}

template <typename Scalar>
struct ObjectiveComponent {
    cv::Vec<Scalar, 3> axis;
    cv::Vec<Scalar, 3> position;
};

template <typename Scalar>
struct ObjectiveConfig {
    Scalar gaussianSigma;
    Scalar gaussianCutoff;
    Scalar axialSupportHalfWidth;
    Scalar presenceFloor;
};

template <typename Scalar>
[[nodiscard]] ObjectiveConfig<Scalar> converted(
    const FiberAnchorObjectiveConfig& config)
{
    return {
        static_cast<Scalar>(config.gaussianSigmaPredictionVoxels),
        static_cast<Scalar>(config.gaussianCutoffSigmas *
            config.gaussianSigmaPredictionVoxels),
        static_cast<Scalar>(config.axialSupportHalfWidthPredictionVoxels),
        static_cast<Scalar>(config.observationPresenceFloor),
    };
}

template <typename Scalar>
[[nodiscard]] std::array<ObjectiveComponent<Scalar>, 2> converted(
    const std::array<FiberAnchorObjectiveComponent, 2>& components)
{
    std::array<ObjectiveComponent<Scalar>, 2> result;
    for (size_t component = 0; component < result.size(); ++component) {
        result[component].axis = converted<Scalar>(components[component].axis);
        result[component].position =
            converted<Scalar>(components[component].position);
    }
    return result;
}

template <typename Scalar>
class ObjectiveSum;

template <>
class ObjectiveSum<double> {
public:
    void add(double value) { value_.add(value); }
    [[nodiscard]] double value() const { return value_.sum; }

private:
    CompensatedSum value_;
};

template <>
class ObjectiveSum<float> {
public:
    void add(float value) { value_ += value; }
    [[nodiscard]] float value() const { return value_; }

private:
    float value_ = 0.0F;
};

struct ExpandedObservationRange {
    using Observation = FiberAnchorObservation;
    using Scalar = double;
    static constexpr bool normalizeDirections = true;

    [[nodiscard]] size_t size() const { return observations.size(); }
    [[nodiscard]] const Observation& operator[](size_t logicalIndex) const
    {
        return observations[logicalIndex];
    }

    std::span<const Observation> observations;
};

struct CompactObservationRange {
    using Observation = CompactFiberAnchorObservation;
    using Scalar = float;
    static constexpr bool normalizeDirections = false;

    [[nodiscard]] size_t size() const { return indices.size(); }
    [[nodiscard]] const Observation& operator[](size_t logicalIndex) const
    {
        return storage[indices[logicalIndex]];
    }

    std::span<const Observation> storage;
    std::span<const uint32_t> indices;
};

template <typename Range>
void validateInputs(
    const Range& observations,
    size_t activeComponents,
    std::span<const uint8_t> assignments,
    std::span<const uint8_t> retainedInliers)
{
    if (activeComponents > 2)
        throw std::invalid_argument("fiber anchor objective has too many components");
    if (assignments.size() != observations.size() ||
        retainedInliers.size() != observations.size()) {
        throw std::invalid_argument("fiber anchor objective cardinality mismatch");
    }
    if constexpr (std::is_same_v<Range, CompactObservationRange>) {
        for (const uint32_t index : observations.indices) {
            if (index >= observations.storage.size())
                throw std::out_of_range("fiber anchor objective observation index");
        }
    }
}

template <typename Range>
[[nodiscard]] cv::Vec<typename Range::Scalar, 3> observationPosition(
    const typename Range::Observation& observation)
{
    using Scalar = typename Range::Scalar;
    return {
        static_cast<Scalar>(observation.positionPredictionXYZ[0]),
        static_cast<Scalar>(observation.positionPredictionXYZ[1]),
        static_cast<Scalar>(observation.positionPredictionXYZ[2]),
    };
}

template <typename Range>
[[nodiscard]] bool usableDirection(
    const typename Range::Observation& observation,
    typename Range::Scalar presenceFloor,
    cv::Vec<typename Range::Scalar, 3>& direction)
{
    using Scalar = typename Range::Scalar;
    const Scalar presence = static_cast<Scalar>(observation.presence);
    if (!observation.valid || !finiteVector(observation.direction) ||
        !std::isfinite(presence) || presence < presenceFloor ||
        presence < Scalar{0} || presence > Scalar{1}) {
        return false;
    }
    direction = {
        static_cast<Scalar>(observation.direction[0]),
        static_cast<Scalar>(observation.direction[1]),
        static_cast<Scalar>(observation.direction[2]),
    };
    Scalar norm2 = direction.dot(direction);
    if constexpr (Range::normalizeDirections) {
        const Scalar normalizationEpsilon = static_cast<Scalar>(
            kMatrixEpsilon * kMatrixEpsilon);
        if (!(norm2 > normalizationEpsilon) || !std::isfinite(norm2))
            return false;
        direction /= std::sqrt(norm2);
        norm2 = direction.dot(direction);
    }
    return norm2 > static_cast<Scalar>(kMatrixEpsilon);
}

template <typename Range>
[[nodiscard]] typename Range::Scalar transverseGaussian(
    const typename Range::Observation& observation,
    const ObjectiveComponent<typename Range::Scalar>& component,
    const cv::Vec<typename Range::Scalar, 3>& pivot,
    const ObjectiveConfig<typename Range::Scalar>& config)
{
    using Scalar = typename Range::Scalar;
    const auto position = observationPosition<Range>(observation);
    if (!finiteVector(position))
        return Scalar{0};
    const Scalar axial = (position - pivot).dot(component.axis);
    if (std::abs(axial) > config.axialSupportHalfWidth)
        return Scalar{0};
    const auto offset = position - component.position;
    const auto transverse =
        offset - component.axis * offset.dot(component.axis);
    const Scalar distanceSquared = transverse.dot(transverse);
    if (distanceSquared > config.gaussianCutoff * config.gaussianCutoff)
        return Scalar{0};
    return std::exp(-distanceSquared /
        (Scalar{2} * config.gaussianSigma * config.gaussianSigma));
}

template <size_t StateCount, typename Range>
[[nodiscard]] std::array<double, StateCount> evaluateObjectives(
    const Range& observations,
    const std::array<std::array<FiberAnchorObjectiveComponent, 2>, StateCount>&
        inputStates,
    size_t activeComponents,
    std::span<const uint8_t> assignments,
    std::span<const uint8_t> retainedInliers,
    const cv::Vec3d& inputPivot,
    const FiberAnchorObjectiveConfig& inputConfig)
{
    using Scalar = typename Range::Scalar;
    validateInputs(
        observations, activeComponents, assignments, retainedInliers);
    std::array<std::array<ObjectiveComponent<Scalar>, 2>, StateCount> states;
    for (size_t state = 0; state < StateCount; ++state)
        states[state] = converted<Scalar>(inputStates[state]);
    const auto pivot = converted<Scalar>(inputPivot);
    const auto config = converted<Scalar>(inputConfig);
    std::array<ObjectiveSum<Scalar>, StateCount> numerators;
    std::array<ObjectiveSum<Scalar>, StateCount> denominators;
    for (size_t logicalIndex = 0; logicalIndex < observations.size();
         ++logicalIndex) {
        const auto& observation = observations[logicalIndex];
        for (size_t component = 0; component < activeComponents; ++component) {
            for (size_t state = 0; state < StateCount; ++state) {
                denominators[state].add(transverseGaussian<Range>(
                    observation, states[state][component], pivot, config));
            }
        }
        const uint8_t component = assignments[logicalIndex];
        if (!retainedInliers[logicalIndex] || component >= activeComponents)
            continue;
        cv::Vec<Scalar, 3> direction;
        if (!usableDirection<Range>(observation, config.presenceFloor, direction))
            continue;
        const Scalar presence = static_cast<Scalar>(observation.presence);
        for (size_t state = 0; state < StateCount; ++state) {
            const Scalar gaussian = transverseGaussian<Range>(
                observation, states[state][component], pivot, config);
            const Scalar dot = direction.dot(states[state][component].axis);
            numerators[state].add(gaussian * presence * dot * dot);
        }
    }
    std::array<double, StateCount> result{};
    for (size_t state = 0; state < StateCount; ++state) {
        const Scalar denominator = denominators[state].value();
        if (denominator > Scalar{0}) {
            result[state] = static_cast<double>(
                numerators[state].value() / denominator);
        }
    }
    return result;
}

template <typename Range>
[[nodiscard]] double evaluateSingle(
    const Range& observations,
    const std::array<FiberAnchorObjectiveComponent, 2>& components,
    size_t activeComponents,
    std::span<const uint8_t> assignments,
    std::span<const uint8_t> retainedInliers,
    const cv::Vec3d& pivot,
    const FiberAnchorObjectiveConfig& config)
{
    const std::array<std::array<FiberAnchorObjectiveComponent, 2>, 1> states{
        components};
    return evaluateObjectives<1>(
        observations, states, activeComponents, assignments,
        retainedInliers, pivot, config)[0];
}

template <typename Range>
[[nodiscard]] std::array<double, 2> evaluatePair(
    const Range& observations,
    const std::array<FiberAnchorObjectiveComponent, 2>& first,
    const std::array<FiberAnchorObjectiveComponent, 2>& second,
    size_t activeComponents,
    std::span<const uint8_t> assignments,
    std::span<const uint8_t> retainedInliers,
    const cv::Vec3d& pivot,
    const FiberAnchorObjectiveConfig& config)
{
    return evaluateObjectives<2>(
        observations, std::array{first, second}, activeComponents, assignments,
        retainedInliers, pivot, config);
}

}  // namespace

double retainedSpatialObjectiveExpanded(
    std::span<const FiberAnchorObservation> observations,
    const std::array<FiberAnchorObjectiveComponent, 2>& components,
    size_t activeComponents,
    std::span<const uint8_t> assignments,
    std::span<const uint8_t> retainedInliers,
    const cv::Vec3d& pivot,
    const FiberAnchorObjectiveConfig& config)
{
    return evaluateSingle(
        ExpandedObservationRange{observations}, components, activeComponents,
        assignments, retainedInliers, pivot, config);
}

std::array<double, 2> retainedSpatialObjectivePairExpanded(
    std::span<const FiberAnchorObservation> observations,
    const std::array<FiberAnchorObjectiveComponent, 2>& first,
    const std::array<FiberAnchorObjectiveComponent, 2>& second,
    size_t activeComponents,
    std::span<const uint8_t> assignments,
    std::span<const uint8_t> retainedInliers,
    const cv::Vec3d& pivot,
    const FiberAnchorObjectiveConfig& config)
{
    return evaluatePair(
        ExpandedObservationRange{observations}, first, second,
        activeComponents, assignments, retainedInliers, pivot, config);
}

double retainedSpatialObjectiveCompact(
    std::span<const CompactFiberAnchorObservation> observationStorage,
    std::span<const uint32_t> observationIndices,
    const std::array<FiberAnchorObjectiveComponent, 2>& components,
    size_t activeComponents,
    std::span<const uint8_t> assignments,
    std::span<const uint8_t> retainedInliers,
    const cv::Vec3d& pivot,
    const FiberAnchorObjectiveConfig& config)
{
    return evaluateSingle(
        CompactObservationRange{observationStorage, observationIndices},
        components, activeComponents, assignments, retainedInliers, pivot,
        config);
}

std::array<double, 2> retainedSpatialObjectivePairCompact(
    std::span<const CompactFiberAnchorObservation> observationStorage,
    std::span<const uint32_t> observationIndices,
    const std::array<FiberAnchorObjectiveComponent, 2>& first,
    const std::array<FiberAnchorObjectiveComponent, 2>& second,
    size_t activeComponents,
    std::span<const uint8_t> assignments,
    std::span<const uint8_t> retainedInliers,
    const cv::Vec3d& pivot,
    const FiberAnchorObjectiveConfig& config)
{
    return evaluatePair(
        CompactObservationRange{observationStorage, observationIndices}, first,
        second, activeComponents, assignments, retainedInliers, pivot, config);
}

}  // namespace vc::fiber_tracer::detail
