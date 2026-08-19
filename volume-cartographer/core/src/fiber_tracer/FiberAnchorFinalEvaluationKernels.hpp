#pragma once

#include "FiberAnchorFinalEvaluation.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace vc::fiber_tracer::detail::final_evaluation_kernel {

constexpr double kMatrixEpsilon = 1.0e-15;

template <typename Scalar>
[[nodiscard]] bool finiteVector(const cv::Vec<Scalar, 3>& value)
{
    return std::isfinite(value[0]) && std::isfinite(value[1]) &&
           std::isfinite(value[2]);
}

[[nodiscard]] inline bool checkedFloat(double value, float& result)
{
    if (!std::isfinite(value) ||
        std::abs(value) > static_cast<double>(std::numeric_limits<float>::max())) {
        return false;
    }
    result = static_cast<float>(value);
    return true;
}

[[nodiscard]] inline cv::Vec3f checkedFloatVector(const cv::Vec3d& value)
{
    cv::Vec3f result;
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
        if (!checkedFloat(value[coordinate], result[coordinate])) {
            throw std::invalid_argument(
                "fiber anchor final state is not float representable");
        }
    }
    return result;
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

[[nodiscard]] inline ObjectiveConfig<float> checkedFloatConfig(
    const FiberAnchorObjectiveConfig& config)
{
    ObjectiveConfig<float> result;
    const double cutoff = config.gaussianCutoffSigmas *
        config.gaussianSigmaPredictionVoxels;
    if (!checkedFloat(config.gaussianSigmaPredictionVoxels,
            result.gaussianSigma) ||
        !checkedFloat(cutoff, result.gaussianCutoff) ||
        !checkedFloat(config.axialSupportHalfWidthPredictionVoxels,
            result.axialSupportHalfWidth) ||
        !checkedFloat(config.observationPresenceFloor, result.presenceFloor)) {
        throw std::invalid_argument(
            "fiber anchor final configuration is not float representable");
    }
    return result;
}

struct ExpandedFinalObservationRange {
    using Observation = FiberAnchorObservation;
    using Scalar = float;
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
        throw std::invalid_argument(
            "fiber anchor final evaluation has too many components");
    if (assignments.size() != observations.size() ||
        retainedInliers.size() != observations.size()) {
        throw std::invalid_argument(
            "fiber anchor final evaluation cardinality mismatch");
    }
    if constexpr (std::is_same_v<Range, CompactObservationRange>) {
        for (const uint32_t index : observations.indices) {
            if (index >= observations.storage.size()) {
                throw std::out_of_range(
                    "fiber anchor final evaluation observation index");
            }
        }
    }
}

template <typename Range>
[[nodiscard]] cv::Vec<typename Range::Scalar, 3> observationPosition(
    const typename Range::Observation& observation)
{
    using Scalar = typename Range::Scalar;
    if constexpr (std::is_same_v<Range, ExpandedFinalObservationRange>) {
        cv::Vec3f position;
        for (int coordinate = 0; coordinate < 3; ++coordinate) {
            if (!checkedFloat(
                    observation.positionPredictionXYZ[coordinate],
                    position[coordinate])) {
                return {
                    std::numeric_limits<float>::quiet_NaN(), 0.0F, 0.0F};
            }
        }
        return position;
    }
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
    Scalar presence;
    if constexpr (std::is_same_v<Range, ExpandedFinalObservationRange>) {
        if (!checkedFloat(observation.presence, presence))
            return false;
    } else {
        presence = static_cast<Scalar>(observation.presence);
    }
    if (!observation.valid || !finiteVector(observation.direction) ||
        !std::isfinite(presence) || presence < presenceFloor ||
        presence < Scalar{0} || presence > Scalar{1}) {
        return false;
    }
    if constexpr (std::is_same_v<Range, ExpandedFinalObservationRange>) {
        for (int coordinate = 0; coordinate < 3; ++coordinate) {
            if (!checkedFloat(
                    observation.direction[coordinate], direction[coordinate])) {
                return false;
            }
        }
    } else {
        direction = {
            static_cast<Scalar>(observation.direction[0]),
            static_cast<Scalar>(observation.direction[1]),
            static_cast<Scalar>(observation.direction[2]),
        };
    }
    Scalar norm2 = direction.dot(direction);
    if constexpr (Range::normalizeDirections) {
        const Scalar scale = std::max({
            std::abs(direction[0]), std::abs(direction[1]),
            std::abs(direction[2])});
        if (!(scale > static_cast<Scalar>(kMatrixEpsilon)) ||
            !std::isfinite(scale)) {
            return false;
        }
        direction /= scale;
        norm2 = direction.dot(direction);
        if (!(norm2 > Scalar{0}) || !std::isfinite(norm2))
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

}  // namespace vc::fiber_tracer::detail::final_evaluation_kernel
