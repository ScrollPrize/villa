#include "vc/fiber_tracer/FiberAnchors.hpp"
#include "vc/core/util/AtomicFile.hpp"
#include "vc/fiber_tracer/FiberAxisTensor.hpp"
#include "vc/fiber_tracer/PolylineGeometry.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <iomanip>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <utility>

namespace vc::fiber_tracer {
namespace {

constexpr double kMatrixEpsilon = 1.0e-15;
constexpr double kGeometryEpsilon = 1.0e-12;

double segmentAabbDistanceSquared(
    const cv::Vec3d& start,
    const cv::Vec3d& end,
    const cv::Vec3d& low,
    const cv::Vec3d& high)
{
    const cv::Vec3d delta = end - start;
    std::vector<double> breaks{0.0, 1.0};
    for (int axis = 0; axis < 3; ++axis) {
        if (std::abs(delta[axis]) <= kGeometryEpsilon)
            continue;
        for (const double bound : {low[axis], high[axis]}) {
            const double t = (bound - start[axis]) / delta[axis];
            if (t > 0.0 && t < 1.0)
                breaks.push_back(t);
        }
    }
    std::sort(breaks.begin(), breaks.end());
    breaks.erase(std::unique(breaks.begin(), breaks.end()), breaks.end());
    double best = std::numeric_limits<double>::infinity();
    const auto evaluate = [&](double t) {
        const cv::Vec3d point = start + delta * t;
        double squared = 0.0;
        for (int axis = 0; axis < 3; ++axis) {
            const double outside = point[axis] < low[axis]
                ? low[axis] - point[axis]
                : point[axis] > high[axis]
                    ? point[axis] - high[axis]
                    : 0.0;
            squared += outside * outside;
        }
        best = std::min(best, squared);
    };
    for (size_t interval = 0; interval + 1 < breaks.size(); ++interval) {
        const double begin = breaks[interval];
        const double finish = breaks[interval + 1];
        evaluate(begin);
        evaluate(finish);
        const double middle = 0.5 * (begin + finish);
        double quadratic = 0.0;
        double linear = 0.0;
        for (int axis = 0; axis < 3; ++axis) {
            const double point = start[axis] + delta[axis] * middle;
            double offset = 0.0;
            if (point < low[axis])
                offset = start[axis] - low[axis];
            else if (point > high[axis])
                offset = start[axis] - high[axis];
            else
                continue;
            quadratic += delta[axis] * delta[axis];
            linear += delta[axis] * offset;
        }
        if (quadratic > kGeometryEpsilon)
            evaluate(std::clamp(-linear / quadratic, begin, finish));
    }
    return best;
}

double interpolatedQuantile(const std::vector<double>& sorted, double quantile)
{
    if (sorted.empty())
        throw std::invalid_argument("anchor benchmark quantile population is empty");
    const double rank = quantile * static_cast<double>(sorted.size() - 1);
    const size_t lower = static_cast<size_t>(std::floor(rank));
    const size_t upper = static_cast<size_t>(std::ceil(rank));
    const double fraction = rank - static_cast<double>(lower);
    return sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction;
}

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

struct WeightedObservation {
    cv::Vec3d position{0.0, 0.0, 0.0};
    cv::Vec3d direction{0.0, 0.0, 0.0};
    double gaussian = 0.0;
    double weight = 0.0;
    size_t canonicalIndex = 0;
};

struct FitState {
    std::array<cv::Vec3d, 2> axes{};
    std::vector<uint8_t> assignments;
    double objectiveNumerator = -1.0;
    size_t iteration = 0;
};

constexpr uint8_t kUnassignedComponent = 2;

struct RefinedComponentState {
    cv::Vec3d axis{1.0, 0.0, 0.0};
    cv::Vec3d position{0.0, 0.0, 0.0};
};

struct RefinedEvaluation {
    std::vector<uint8_t> assignments;
    std::array<double, 2> denominators{0.0, 0.0};
    std::array<double, 2> numerators{0.0, 0.0};
    std::array<double, 2> presenceMasses{0.0, 0.0};
    std::array<size_t, 2> assignedCounts{0, 0};
    double objective = 0.0;
};

struct RefinedFitState {
    std::array<RefinedComponentState, 2> components;
    RefinedEvaluation evaluation;
    size_t activeComponents = 0;
    size_t acceptedIterations = 0;
};

struct PeakOwnerBounds {
    cv::Vec3d lower{0.0, 0.0, 0.0};
    cv::Vec3d upper{0.0, 0.0, 0.0};
};

constexpr size_t kNoDiagnosticId = std::numeric_limits<size_t>::max();

[[nodiscard]] bool finiteVector(const cv::Vec3d& value)
{
    return std::isfinite(value[0]) && std::isfinite(value[1]) &&
           std::isfinite(value[2]);
}

[[nodiscard]] cv::Vec3d normalized(const cv::Vec3d& value)
{
    const double norm2 = value.dot(value);
    if (!(norm2 > kMatrixEpsilon * kMatrixEpsilon) || !std::isfinite(norm2))
        return {0.0, 0.0, 0.0};
    return value / std::sqrt(norm2);
}

[[nodiscard]] std::array<cv::Vec3d, 2> transverseBasis(
    const cv::Vec3d& axis)
{
    size_t referenceIndex = 0;
    for (size_t index = 1; index < 3; ++index) {
        if (std::abs(axis[static_cast<int>(index)]) <
            std::abs(axis[static_cast<int>(referenceIndex)])) {
            referenceIndex = index;
        }
    }
    cv::Vec3d reference{0.0, 0.0, 0.0};
    reference[static_cast<int>(referenceIndex)] = 1.0;
    const cv::Vec3d first = normalized(
        reference - axis * reference.dot(axis));
    return {first, normalized(axis.cross(first))};
}

[[nodiscard]] bool insidePeakDomain(
    const cv::Vec3d& point,
    const cv::Vec3d& pivot,
    const PeakOwnerBounds& owner,
    double radius)
{
    const cv::Vec3d offset = point - pivot;
    if (offset.dot(offset) > radius * radius + 1.0e-12)
        return false;
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
        if (point[coordinate] < owner.lower[coordinate] ||
            point[coordinate] > owner.upper[coordinate]) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] cv::Matx33d weightedTensor(
    const std::vector<WeightedObservation>& observations,
    const std::vector<uint8_t>* assignments,
    uint8_t component)
{
    std::array<CompensatedSum, 9> sums;
    for (size_t index = 0; index < observations.size(); ++index) {
        if (assignments != nullptr && (*assignments)[index] != component)
            continue;
        const auto& observation = observations[index];
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                sums[static_cast<size_t>(row * 3 + column)].add(
                    observation.weight * observation.direction[row] *
                    observation.direction[column]);
            }
        }
    }
    cv::Matx33d tensor;
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column)
            tensor(row, column) = sums[static_cast<size_t>(row * 3 + column)].sum;
    }
    return tensor;
}

[[nodiscard]] std::vector<uint8_t> assignObservations(
    const std::vector<WeightedObservation>& observations,
    const std::array<cv::Vec3d, 2>& axes)
{
    std::vector<uint8_t> assignments(observations.size(), 0);
    for (size_t index = 0; index < observations.size(); ++index) {
        const double dot0 = observations[index].direction.dot(axes[0]);
        const double dot1 = observations[index].direction.dot(axes[1]);
        assignments[index] = dot0 * dot0 >= dot1 * dot1 ? 0 : 1;
    }
    return assignments;
}

[[nodiscard]] double objectiveNumerator(
    const std::vector<WeightedObservation>& observations,
    const std::array<cv::Vec3d, 2>& axes)
{
    CompensatedSum result;
    for (const auto& observation : observations) {
        const double dot0 = observation.direction.dot(axes[0]);
        const double dot1 = observation.direction.dot(axes[1]);
        result.add(observation.weight * std::max(dot0 * dot0, dot1 * dot1));
    }
    return result.sum;
}

[[nodiscard]] double projectiveUpdate(const cv::Vec3d& before, const cv::Vec3d& after)
{
    return 1.0 - std::clamp(std::abs(before.dot(after)), 0.0, 1.0);
}

[[nodiscard]] cv::Vec3d projectToConstraintPlane(
    const cv::Vec3d& point,
    const cv::Vec3d& pivot,
    const cv::Vec3d& axis)
{
    return point - axis * ((point - pivot).dot(axis));
}

[[nodiscard]] cv::Vec3d clampToWindow(
    const cv::Vec3d& point,
    const cv::Vec3d& pivot,
    const cv::Vec3d& axis,
    double radius,
    const cv::Vec3d& lower,
    const cv::Vec3d& upper)
{
    cv::Vec3d offset = projectToConstraintPlane(point, pivot, axis) - pivot;
    const double length = std::sqrt(std::max(0.0, offset.dot(offset)));
    if (length > radius)
        offset *= radius / length;
    double scale = 1.0;
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
        if (offset[coordinate] > 0.0) {
            scale = std::min(
                scale, (upper[coordinate] - pivot[coordinate]) /
                    offset[coordinate]);
        } else if (offset[coordinate] < 0.0) {
            scale = std::min(
                scale, (lower[coordinate] - pivot[coordinate]) /
                    offset[coordinate]);
        }
    }
    return pivot + offset * std::clamp(scale, 0.0, 1.0);
}

[[nodiscard]] double transverseGaussian(
    const FiberAnchorObservation& observation,
    const RefinedComponentState& component,
    const cv::Vec3d& pivot,
    const FiberAnchorConfig& config)
{
    if (!finiteVector(observation.positionPredictionXYZ))
        return 0.0;
    const double axial = (observation.positionPredictionXYZ - pivot).dot(component.axis);
    if (std::abs(axial) > config.axialSupportHalfWidthPredictionVoxels)
        return 0.0;
    const cv::Vec3d offset = observation.positionPredictionXYZ - component.position;
    const cv::Vec3d transverse = offset - component.axis * offset.dot(component.axis);
    const double distanceSquared = transverse.dot(transverse);
    const double cutoff = config.gaussianCutoffSigmas * config.gaussianSigmaPredictionVoxels;
    if (distanceSquared > cutoff * cutoff)
        return 0.0;
    return std::exp(-distanceSquared /
        (2.0 * config.gaussianSigmaPredictionVoxels *
         config.gaussianSigmaPredictionVoxels));
}

[[nodiscard]] RefinedEvaluation evaluateRefinedState(
    const std::vector<FiberAnchorObservation>& observations,
    const std::array<RefinedComponentState, 2>& components,
    size_t activeComponents,
    const cv::Vec3d& pivot,
    const FiberAnchorConfig& config)
{
    RefinedEvaluation evaluation;
    evaluation.assignments.assign(observations.size(), kUnassignedComponent);
    std::array<CompensatedSum, 2> denominators;
    std::array<CompensatedSum, 2> numerators;
    std::array<CompensatedSum, 2> presenceMasses;
    for (size_t index = 0; index < observations.size(); ++index) {
        const auto& observation = observations[index];
        std::array<double, 2> gaussian{0.0, 0.0};
        std::array<double, 2> evidence{0.0, 0.0};
        for (size_t component = 0; component < activeComponents; ++component) {
            gaussian[component] = transverseGaussian(
                observation, components[component], pivot, config);
            denominators[component].add(gaussian[component]);
            if (!(gaussian[component] > 0.0) || !observation.valid ||
                !finiteVector(observation.direction) ||
                !std::isfinite(observation.presence) ||
                observation.presence < config.observationPresenceFloor ||
                observation.presence < 0.0 || observation.presence > 1.0) {
                continue;
            }
            const cv::Vec3d direction = normalized(observation.direction);
            if (direction.dot(direction) <= kMatrixEpsilon)
                continue;
            const double dot = direction.dot(components[component].axis);
            evidence[component] = observation.presence * dot * dot;
        }
        uint8_t assigned = kUnassignedComponent;
        if (activeComponents == 1) {
            if (evidence[0] > 0.0)
                assigned = 0;
        } else if (evidence[0] > 0.0 || evidence[1] > 0.0) {
            assigned = evidence[0] >= evidence[1] ? 0 : 1;
        }
        evaluation.assignments[index] = assigned;
        if (assigned == kUnassignedComponent)
            continue;
        const auto& component = components[assigned];
        const cv::Vec3d direction = normalized(observation.direction);
        const double dot = direction.dot(component.axis);
        numerators[assigned].add(
            gaussian[assigned] * observation.presence * dot * dot);
        presenceMasses[assigned].add(
            gaussian[assigned] * observation.presence);
        ++evaluation.assignedCounts[assigned];
    }
    CompensatedSum numeratorTotal;
    CompensatedSum denominatorTotal;
    for (size_t component = 0; component < activeComponents; ++component) {
        evaluation.denominators[component] = denominators[component].sum;
        evaluation.numerators[component] = numerators[component].sum;
        evaluation.presenceMasses[component] = presenceMasses[component].sum;
        numeratorTotal.add(numerators[component].sum);
        denominatorTotal.add(denominators[component].sum);
    }
    evaluation.objective = denominatorTotal.sum > 0.0
        ? numeratorTotal.sum / denominatorTotal.sum
        : 0.0;
    return evaluation;
}

[[nodiscard]] RefinedFitState refineLocalComponents(
    const std::vector<FiberAnchorObservation>& observations,
    const cv::Vec3d& pivot,
    const std::array<cv::Vec3d, 2>& seedAxes,
    size_t activeComponents,
    const FiberAnchorConfig& config)
{
    RefinedFitState state;
    state.activeComponents = activeComponents;
    for (size_t component = 0; component < activeComponents; ++component) {
        state.components[component].axis = canonicalFiberAxis(seedAxes[component]);
        state.components[component].position = pivot;
    }
    cv::Vec3d lower{
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
    };
    cv::Vec3d upper{
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
    };
    for (const auto& observation : observations) {
        if (!finiteVector(observation.positionPredictionXYZ))
            continue;
        for (int coordinate = 0; coordinate < 3; ++coordinate) {
            lower[coordinate] = std::min(
                lower[coordinate], observation.positionPredictionXYZ[coordinate]);
            upper[coordinate] = std::max(
                upper[coordinate], observation.positionPredictionXYZ[coordinate]);
        }
    }
    state.evaluation = evaluateRefinedState(
        observations, state.components, activeComponents, pivot, config);

    for (int iteration = 0; iteration < config.maximumIterations; ++iteration) {
        auto proposed = state.components;
        for (size_t component = 0; component < activeComponents; ++component) {
            std::array<CompensatedSum, 9> tensorSums;
            for (size_t index = 0; index < observations.size(); ++index) {
                if (state.evaluation.assignments[index] != component)
                    continue;
                const auto& observation = observations[index];
                const double gaussian = transverseGaussian(
                    observation, state.components[component], pivot, config);
                const cv::Vec3d direction = normalized(observation.direction);
                const double weight = gaussian * observation.presence;
                for (int row = 0; row < 3; ++row) {
                    for (int column = 0; column < 3; ++column) {
                        tensorSums[static_cast<size_t>(row * 3 + column)].add(
                            weight * direction[row] * direction[column]);
                    }
                }
            }
            cv::Matx33d tensor;
            for (int row = 0; row < 3; ++row) {
                for (int column = 0; column < 3; ++column)
                    tensor(row, column) = tensorSums[static_cast<size_t>(row * 3 + column)].sum;
            }
            const FiberPrincipalAxis principal = principalFiberAxis(tensor);
            if (principal.unique)
                proposed[component].axis = principal.axis;

            const cv::Vec3d projectedCurrent = projectToConstraintPlane(
                state.components[component].position,
                pivot,
                proposed[component].axis);
            std::array<CompensatedSum, 3> centroid;
            CompensatedSum centroidMass;
            RefinedComponentState centered{
                proposed[component].axis,
                projectedCurrent,
            };
            for (size_t index = 0; index < observations.size(); ++index) {
                if (state.evaluation.assignments[index] != component)
                    continue;
                const auto& observation = observations[index];
                const double gaussian = transverseGaussian(
                    observation, centered, pivot, config);
                const cv::Vec3d direction = normalized(observation.direction);
                const double dot = direction.dot(centered.axis);
                const double weight = gaussian * observation.presence * dot * dot;
                centroidMass.add(weight);
                for (int axis = 0; axis < 3; ++axis)
                    centroid[axis].add(weight * observation.positionPredictionXYZ[axis]);
            }
            if (centroidMass.sum > 0.0) {
                const cv::Vec3d mean{
                    centroid[0].sum / centroidMass.sum,
                    centroid[1].sum / centroidMass.sum,
                    centroid[2].sum / centroidMass.sum,
                };
                proposed[component].position = clampToWindow(
                    mean, pivot, proposed[component].axis,
                    config.localWindowRadiusPredictionVoxels, lower, upper);
            } else {
                proposed[component].position = clampToWindow(
                    projectedCurrent, pivot, proposed[component].axis,
                    config.localWindowRadiusPredictionVoxels, lower, upper);
            }
        }

        bool accepted = false;
        RefinedFitState candidate = state;
        double acceptedDirectionUpdate = 0.0;
        double acceptedPositionUpdate = 0.0;
        for (int backtrack = 0; backtrack <= 8; ++backtrack) {
            const double fraction = std::ldexp(1.0, -backtrack);
            auto interpolated = state.components;
            for (size_t component = 0; component < activeComponents; ++component) {
                cv::Vec3d targetAxis = proposed[component].axis;
                if (state.components[component].axis.dot(targetAxis) < 0.0)
                    targetAxis *= -1.0;
                interpolated[component].axis = canonicalFiberAxis(normalized(
                    state.components[component].axis * (1.0 - fraction) +
                    targetAxis * fraction));
                const cv::Vec3d offset =
                    (state.components[component].position - pivot) * (1.0 - fraction) +
                    (proposed[component].position - pivot) * fraction;
                interpolated[component].position = clampToWindow(
                    pivot + offset, pivot, interpolated[component].axis,
                    config.localWindowRadiusPredictionVoxels, lower, upper);
            }
            RefinedEvaluation evaluation = evaluateRefinedState(
                observations, interpolated, activeComponents, pivot, config);
            const double tolerance = config.convergenceTolerance *
                std::max(1.0, std::abs(state.evaluation.objective));
            if (evaluation.objective <= state.evaluation.objective + tolerance)
                continue;
            candidate.components = interpolated;
            candidate.evaluation = std::move(evaluation);
            candidate.acceptedIterations = state.acceptedIterations + 1;
            for (size_t component = 0; component < activeComponents; ++component) {
                acceptedDirectionUpdate = std::max(
                    acceptedDirectionUpdate,
                    projectiveUpdate(
                        state.components[component].axis,
                        candidate.components[component].axis));
                const cv::Vec3d delta =
                    candidate.components[component].position -
                    state.components[component].position;
                acceptedPositionUpdate = std::max(
                    acceptedPositionUpdate,
                    std::sqrt(std::max(0.0, delta.dot(delta))));
            }
            accepted = true;
            break;
        }
        if (!accepted)
            break;
        const bool assignmentsUnchanged =
            candidate.evaluation.assignments == state.evaluation.assignments;
        state = std::move(candidate);
        if (assignmentsUnchanged &&
            acceptedDirectionUpdate <= config.convergenceTolerance &&
            acceptedPositionUpdate <=
                config.positionConvergenceTolerancePredictionVoxels) {
            break;
        }
    }
    return state;
}

[[nodiscard]] PeakOwnerBounds peakOwnerBounds(
    const std::vector<FiberAnchorObservation>& observations,
    const std::array<size_t, 3>& cellBeginZYX,
    const std::array<size_t, 3>& cellEndZYX)
{
    cv::Vec3d observedLower{
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
    };
    cv::Vec3d observedUpper{
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
    };
    for (const auto& observation : observations) {
        if (!finiteVector(observation.positionPredictionXYZ))
            continue;
        for (int coordinate = 0; coordinate < 3; ++coordinate) {
            observedLower[coordinate] = std::min(
                observedLower[coordinate],
                observation.positionPredictionXYZ[coordinate]);
            observedUpper[coordinate] = std::max(
                observedUpper[coordinate],
                observation.positionPredictionXYZ[coordinate]);
        }
    }

    PeakOwnerBounds owner;
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
        const size_t zyx = static_cast<size_t>(2 - coordinate);
        const double begin = static_cast<double>(cellBeginZYX[zyx]);
        const double end = static_cast<double>(cellEndZYX[zyx]);
        owner.lower[coordinate] = std::max(observedLower[coordinate], begin - 0.5);
        const double voronoiUpper = end - 0.5;
        if (observedUpper[coordinate] <= end - 1.0) {
            owner.upper[coordinate] = observedUpper[coordinate];
        } else {
            owner.upper[coordinate] = std::nextafter(
                std::min(observedUpper[coordinate], voronoiUpper),
                -std::numeric_limits<double>::infinity());
        }
    }
    return owner;
}

struct DirectionConditionedPeak {
    cv::Vec3d discrete{0.0, 0.0, 0.0};
    cv::Vec3d separable1d{0.0, 0.0, 0.0};
    cv::Vec3d joint2d{0.0, 0.0, 0.0};
};

[[nodiscard]] DirectionConditionedPeak findDirectionConditionedLocalPeak(
    const std::vector<FiberAnchorObservation>& observations,
    const cv::Vec3d& pivot,
    const PeakOwnerBounds& owner,
    const std::array<RefinedComponentState, 2>& components,
    size_t activeComponents,
    size_t selectedComponent,
    const FiberAnchorConfig& config)
{
    const cv::Vec3d axis = components[selectedComponent].axis;
    const auto basis = transverseBasis(axis);
    const double cutoff =
        config.gaussianCutoffSigmas * config.peakSigmaPredictionVoxels;
    const double axialCutoff =
        config.gaussianCutoffSigmas *
        config.peakAxialSigmaPredictionVoxels;
    const double unionRadius = config.localWindowRadiusPredictionVoxels + cutoff;
    const double invTwoTransverseSigma2 = 1.0 /
        (2.0 * config.peakSigmaPredictionVoxels *
         config.peakSigmaPredictionVoxels);
    const double invTwoAxialSigma2 = 1.0 /
        (2.0 * config.peakAxialSigmaPredictionVoxels *
         config.peakAxialSigmaPredictionVoxels);

    struct PeakObservation {
        cv::Vec3d position{0.0, 0.0, 0.0};
        double signal = 0.0;
        double directionAlignmentSquared = 0.0;
        cv::Vec3d presenceGradient{0.0, 0.0, 0.0};
        bool presenceGradientValid = false;
    };
    std::vector<PeakObservation> peakObservations;
    peakObservations.reserve(observations.size());
    for (const auto& observation : observations) {
        if (!finiteVector(observation.positionPredictionXYZ))
            continue;
        const cv::Vec3d pivotOffset =
            observation.positionPredictionXYZ - pivot;
        const double axial = pivotOffset.dot(axis);
        if (std::abs(axial) > axialCutoff)
            continue;
        const cv::Vec3d transverse = pivotOffset - axis * axial;
        if (transverse.dot(transverse) > unionRadius * unionRadius)
            continue;

        double signal = 0.0;
        double selectedAlignment = 0.0;
        if (observation.valid && finiteVector(observation.direction) &&
            std::isfinite(observation.presence) &&
            observation.presence >= config.observationPresenceFloor &&
            observation.presence >= 0.0 && observation.presence <= 1.0) {
            const cv::Vec3d direction = normalized(observation.direction);
            if (direction.dot(direction) > kMatrixEpsilon) {
                size_t assignment = 0;
                double bestAlignment = -1.0;
                for (size_t component = 0; component < activeComponents;
                     ++component) {
                    const double dot = direction.dot(components[component].axis);
                    const double alignment = dot * dot;
                    if (alignment > bestAlignment) {
                        bestAlignment = alignment;
                        assignment = component;
                    }
                }
                if (assignment == selectedComponent) {
                    signal = observation.presence * bestAlignment;
                    selectedAlignment = bestAlignment;
                }
            }
        }
        peakObservations.push_back({
            observation.positionPredictionXYZ,
            signal,
            selectedAlignment,
            observation.presenceGradientPredictionXYZ,
            observation.presenceGradientValid,
        });
    }

    const auto responseAt = [&](const cv::Vec3d& candidate) {
        CompensatedSum numerator;
        CompensatedSum denominator;
        CompensatedSum eligibleGradientWeight;
        CompensatedSum validGradientWeight;
        CompensatedSum inward;
        CompensatedSum outward;
        for (const auto& observation : peakObservations) {
            const cv::Vec3d offset = observation.position - candidate;
            const double axial = offset.dot(axis);
            if (std::abs(axial) > axialCutoff)
                continue;
            const cv::Vec3d transverse = offset - axis * axial;
            const double distanceSquared = transverse.dot(transverse);
            if (distanceSquared > cutoff * cutoff)
                continue;
            const double gaussian = std::exp(
                -distanceSquared * invTwoTransverseSigma2 -
                axial * axial * invTwoAxialSigma2);
            denominator.add(gaussian);
            numerator.add(gaussian * observation.signal);
            if (!(observation.directionAlignmentSquared > 0.0))
                continue;
            const double eligibleWeight =
                gaussian * observation.directionAlignmentSquared;
            eligibleGradientWeight.add(eligibleWeight);
            if (!observation.presenceGradientValid ||
                !finiteVector(observation.presenceGradient)) {
                continue;
            }
            const cv::Vec3d radial =
                (candidate - observation.position) -
                axis * (candidate - observation.position).dot(axis);
            const cv::Vec3d gradient = observation.presenceGradient -
                axis * observation.presenceGradient.dot(axis);
            const double radialNorm2 = radial.dot(radial);
            const double gradientNorm2 = gradient.dot(gradient);
            if (!(radialNorm2 > kMatrixEpsilon) ||
                !(gradientNorm2 > kMatrixEpsilon)) {
                continue;
            }
            validGradientWeight.add(eligibleWeight);
            const double cosine = std::clamp(
                gradient.dot(radial) /
                    std::sqrt(gradientNorm2 * radialNorm2),
                -1.0,
                1.0);
            const double vote = eligibleWeight *
                std::sqrt(gradientNorm2) *
                config.peakSigmaPredictionVoxels * cosine * cosine;
            if (cosine > 0.0)
                inward.add(vote);
            else if (cosine < 0.0)
                outward.add(vote);
        }
        const double presenceResponse = denominator.sum > 0.0
            ? numerator.sum / denominator.sum
            : 0.0;
        if (!(config.peakGradientWeight > 0.0) ||
            !(eligibleGradientWeight.sum > 0.0) ||
            !(validGradientWeight.sum > 0.0)) {
            return presenceResponse;
        }
        const double voteMass = inward.sum + outward.sum;
        if (!(voteMass > 0.0))
            return presenceResponse;
        const double coverage = std::clamp(
            validGradientWeight.sum / eligibleGradientWeight.sum, 0.0, 1.0);
        const double radialGradient =
            voteMass / validGradientWeight.sum;
        const double reliability = coverage * radialGradient /
            (radialGradient + config.peakGradientReliabilityScale);
        const double signedVote =
            (inward.sum - outward.sum) / voteMass;
        return presenceResponse +
            config.peakGradientWeight * reliability * signedVote;
    };

    using GridIndex = std::pair<int, int>;
    const int extent = static_cast<int>(std::floor(
        config.localWindowRadiusPredictionVoxels /
        config.peakGridStepPredictionVoxels));
    const auto pointAt = [&](const GridIndex& index) {
        return pivot + basis[0] *
                (static_cast<double>(index.first) *
                 config.peakGridStepPredictionVoxels) +
            basis[1] *
                (static_cast<double>(index.second) *
                 config.peakGridStepPredictionVoxels);
    };
    const auto feasible = [&](const GridIndex& index) {
        return std::abs(index.first) <= extent &&
            std::abs(index.second) <= extent &&
            insidePeakDomain(
                pointAt(index), pivot, owner,
                config.localWindowRadiusPredictionVoxels);
    };

    std::map<GridIndex, double> responseCache;
    const auto response = [&](const GridIndex& index) {
        const auto found = responseCache.find(index);
        if (found != responseCache.end())
            return found->second;
        return responseCache.emplace(index, responseAt(pointAt(index)))
            .first->second;
    };

    GridIndex current{0, 0};
    double nearestDistance = std::numeric_limits<double>::infinity();
    for (int first = -extent; first <= extent; ++first) {
        for (int second = -extent; second <= extent; ++second) {
            const GridIndex candidate{first, second};
            if (!feasible(candidate))
                continue;
            const cv::Vec3d delta =
                pointAt(candidate) - components[selectedComponent].position;
            const double distance = delta.dot(delta);
            if (distance < nearestDistance) {
                nearestDistance = distance;
                current = candidate;
            }
        }
    }

    constexpr std::array<GridIndex, 8> neighbors{{
        {-1, -1}, {-1, 0}, {-1, 1}, {0, -1},
        {0, 1}, {1, -1}, {1, 0}, {1, 1},
    }};
    while (true) {
        GridIndex best = current;
        double bestResponse = response(current);
        for (const auto& offset : neighbors) {
            const GridIndex candidate{
                current.first + offset.first,
                current.second + offset.second,
            };
            if (!feasible(candidate))
                continue;
            const double candidateResponse = response(candidate);
            if (candidateResponse > bestResponse ||
                (candidateResponse == bestResponse && candidate < best)) {
                best = candidate;
                bestResponse = candidateResponse;
            }
        }
        const double tolerance = 1.0e-12 *
            std::max(1.0, std::abs(response(current)));
        if (bestResponse <= response(current) + tolerance)
            break;
        current = best;
    }

    const cv::Vec3d discrete = pointAt(current);
    const double centerResponse = response(current);
    const double tolerance =
        1.0e-12 * std::max(1.0, std::abs(centerResponse));
    const auto acceptedPosition = [&](const cv::Vec3d& candidate) {
        return insidePeakDomain(
                   candidate, pivot, owner,
                   config.localWindowRadiusPredictionVoxels) &&
            responseAt(candidate) + tolerance >= centerResponse;
    };

    std::array<double, 2> separableOffsetGridSteps{0.0, 0.0};
    for (int dimension = 0; dimension < 2; ++dimension) {
        GridIndex lower = current;
        GridIndex upper = current;
        (dimension == 0 ? lower.first : lower.second) -= 1;
        (dimension == 0 ? upper.first : upper.second) += 1;
        if (!feasible(lower) || !feasible(upper))
            continue;
        const double lowerResponse = response(lower);
        const double upperResponse = response(upper);
        const double curvature =
            lowerResponse - 2.0 * centerResponse + upperResponse;
        if (!(curvature < 0.0) || !std::isfinite(curvature))
            continue;
        const double offset =
            0.5 * (lowerResponse - upperResponse) / curvature;
        if (std::isfinite(offset)) {
            separableOffsetGridSteps[static_cast<size_t>(dimension)] =
                std::clamp(offset, -0.5, 0.5);
        }
    }
    const cv::Vec3d separableCandidate =
        discrete + basis[0] *
                (separableOffsetGridSteps[0] *
                 config.peakGridStepPredictionVoxels) +
        basis[1] *
                (separableOffsetGridSteps[1] *
                 config.peakGridStepPredictionVoxels);
    const cv::Vec3d separable = acceptedPosition(separableCandidate)
        ? separableCandidate
        : discrete;

    std::array<std::array<double, 3>, 3> neighborhood{};
    bool completeNeighborhood = true;
    for (int first = -1; first <= 1; ++first) {
        for (int second = -1; second <= 1; ++second) {
            const GridIndex sample{
                current.first + first,
                current.second + second,
            };
            if (!feasible(sample)) {
                completeNeighborhood = false;
                continue;
            }
            const double value = response(sample);
            if (!std::isfinite(value)) {
                completeNeighborhood = false;
                continue;
            }
            neighborhood[static_cast<size_t>(first + 1)]
                        [static_cast<size_t>(second + 1)] = value;
        }
    }
    cv::Vec3d joint = discrete;
    if (completeNeighborhood) {
        const auto offset = fitFiberAnchorQuadraticPeak(neighborhood);
        if (offset.has_value()) {
            const cv::Vec3d candidate =
                discrete + basis[0] *
                        (offset->firstGridSteps *
                         config.peakGridStepPredictionVoxels) +
                basis[1] *
                        (offset->secondGridSteps *
                         config.peakGridStepPredictionVoxels);
            if (acceptedPosition(candidate))
                joint = candidate;
        }
    }
    return {discrete, separable, joint};
}

[[nodiscard]] bool betterState(const FitState& candidate, const FitState& current)
{
    return candidate.objectiveNumerator > current.objectiveNumerator;
}

[[nodiscard]] FitState refineSeedPair(
    const std::vector<WeightedObservation>& observations,
    std::array<cv::Vec3d, 2> axes,
    const FiberAnchorConfig& config)
{
    FitState best;
    best.objectiveNumerator = -1.0;
    std::vector<uint8_t> previousAssignments;
    std::vector<uint8_t> twoBackAssignments;
    for (int iteration = 0; iteration < config.maximumIterations; ++iteration) {
        auto assignments = assignObservations(observations, axes);
        std::array<cv::Vec3d, 2> updated = axes;
        for (uint8_t component = 0; component < 2; ++component) {
            const auto principal = principalFiberAxis(
                weightedTensor(observations, &assignments, component));
            if (principal.unique)
                updated[component] = principal.axis;
        }
        FitState state{
            updated,
            assignments,
            objectiveNumerator(observations, updated),
            static_cast<size_t>(iteration),
        };
        if (best.objectiveNumerator < 0.0 || betterState(state, best))
            best = state;

        const double update = std::max(
            projectiveUpdate(axes[0], updated[0]),
            projectiveUpdate(axes[1], updated[1]));
        const bool unchanged = !previousAssignments.empty() &&
            assignments == previousAssignments;
        const bool twoCycle = !twoBackAssignments.empty() &&
            assignments == twoBackAssignments;
        axes = updated;
        if ((unchanged && update <= config.convergenceTolerance) || twoCycle)
            break;
        twoBackAssignments = std::move(previousAssignments);
        previousAssignments = std::move(assignments);
    }
    best.assignments = assignObservations(observations, best.axes);
    best.objectiveNumerator = objectiveNumerator(observations, best.axes);
    return best;
}

[[nodiscard]] bool componentLess(
    const FiberAnchorComponent& left,
    const FiberAnchorComponent& right)
{
    if (left.retained != right.retained)
        return left.retained;
    if (!left.retained)
        return left.rejectionReason < right.rejectionReason;
    for (int axis = 0; axis < 3; ++axis) {
        if (left.anchor.positionPredictionXYZ[axis] !=
            right.anchor.positionPredictionXYZ[axis]) {
            return left.anchor.positionPredictionXYZ[axis] <
                right.anchor.positionPredictionXYZ[axis];
        }
    }
    for (int axis = 0; axis < 3; ++axis) {
        if (left.anchor.axisXYZ[axis] != right.anchor.axisXYZ[axis])
            return left.anchor.axisXYZ[axis] < right.anchor.axisXYZ[axis];
    }
    return false;
}

[[nodiscard]] size_t ceilDivide(size_t value, size_t divisor)
{
    return value / divisor + (value % divisor != 0 ? 1 : 0);
}

struct NmsCandidate {
    FiberCellAnchorResult* cell = nullptr;
    size_t componentIndex = 0;
};

[[nodiscard]] bool nmsRankedBefore(
    const NmsCandidate& left,
    const NmsCandidate& right)
{
    const auto& leftAnchor = left.cell->components[left.componentIndex].anchor;
    const auto& rightAnchor = right.cell->components[right.componentIndex].anchor;
    if (leftAnchor.alignedSupport != rightAnchor.alignedSupport)
        return leftAnchor.alignedSupport > rightAnchor.alignedSupport;
    if (leftAnchor.directionalCoherence != rightAnchor.directionalCoherence)
        return leftAnchor.directionalCoherence > rightAnchor.directionalCoherence;
    if (left.cell->cellZYX != right.cell->cellZYX)
        return left.cell->cellZYX < right.cell->cellZYX;
    return left.componentIndex < right.componentIndex;
}

[[nodiscard]] bool nmsNeighbors(
    const FiberAnchor& left,
    const FiberAnchor& right,
    const FiberAnchorConfig& config)
{
    double dot = std::clamp(left.axisXYZ.dot(right.axisXYZ), -1.0, 1.0);
    const double axialDot = std::abs(dot);
    const double minimumDot = std::cos(
        config.nmsMaximumAngleDegrees * std::acos(-1.0) / 180.0);
    if (axialDot < minimumDot)
        return false;
    cv::Vec3d alignedRight = right.axisXYZ;
    if (dot < 0.0)
        alignedRight *= -1.0;
    cv::Vec3d averageAxis = normalized(left.axisXYZ + alignedRight);
    if (averageAxis.dot(averageAxis) <= kMatrixEpsilon)
        averageAxis = left.axisXYZ;
    const cv::Vec3d delta = right.positionPredictionXYZ -
        left.positionPredictionXYZ;
    const double longitudinal = std::abs(delta.dot(averageAxis));
    const cv::Vec3d transverseVector =
        delta - averageAxis * delta.dot(averageAxis);
    const double transverse = std::sqrt(std::max(
        0.0, transverseVector.dot(transverseVector)));
    return longitudinal <= config.nmsLongitudinalRadiusPredictionVoxels &&
        transverse <= config.nmsTransverseRadiusPredictionVoxels;
}

void applyLocalMaximumNms(
    std::vector<FiberCellAnchorResult>& cells,
    const FiberAnchorConfig& config)
{
    std::vector<NmsCandidate> candidates;
    for (auto& cell : cells) {
        for (size_t component = 0; component < cell.components.size(); ++component) {
            if (cell.components[component].retained)
                candidates.push_back({&cell, component});
        }
    }
    if (candidates.empty())
        return;
    const double binSize = std::max(
        1.0e-12,
        std::hypot(
            config.nmsTransverseRadiusPredictionVoxels,
            config.nmsLongitudinalRadiusPredictionVoxels));
    using Bin = std::array<int64_t, 3>;
    std::map<Bin, std::vector<size_t>> bins;
    const auto binFor = [binSize](const cv::Vec3d& position) {
        return Bin{
            static_cast<int64_t>(std::floor(position[0] / binSize)),
            static_cast<int64_t>(std::floor(position[1] / binSize)),
            static_cast<int64_t>(std::floor(position[2] / binSize)),
        };
    };
    for (size_t index = 0; index < candidates.size(); ++index) {
        const auto& anchor = candidates[index].cell->components[
            candidates[index].componentIndex].anchor;
        bins[binFor(anchor.positionPredictionXYZ)].push_back(index);
    }
    std::vector<std::optional<size_t>> suppressors(candidates.size());
    for (size_t index = 0; index < candidates.size(); ++index) {
        const auto& candidate = candidates[index];
        const auto& anchor = candidate.cell->components[
            candidate.componentIndex].anchor;
        const Bin ownBin = binFor(anchor.positionPredictionXYZ);
        for (int dz = -1; dz <= 1 && !suppressors[index].has_value(); ++dz) {
            for (int dy = -1; dy <= 1 && !suppressors[index].has_value(); ++dy) {
                for (int dx = -1; dx <= 1 && !suppressors[index].has_value(); ++dx) {
                    const auto found = bins.find(Bin{
                        ownBin[0] + dx, ownBin[1] + dy, ownBin[2] + dz});
                    if (found == bins.end())
                        continue;
                    for (const size_t otherIndex : found->second) {
                        if (otherIndex == index)
                            continue;
                        const auto& other = candidates[otherIndex];
                        const auto& otherAnchor = other.cell->components[
                            other.componentIndex].anchor;
                        if (nmsRankedBefore(other, candidate) &&
                            nmsNeighbors(anchor, otherAnchor, config)) {
                            suppressors[index] = otherIndex;
                            break;
                        }
                    }
                }
            }
        }
    }
    for (size_t index = 0; index < candidates.size(); ++index) {
        if (!suppressors[index].has_value())
            continue;
        auto& component = candidates[index].cell->components[
            candidates[index].componentIndex];
        const auto& suppressor = candidates[*suppressors[index]];
        const auto& suppressorComponent = suppressor.cell->components[
            suppressor.componentIndex];
        component.nmsSuppressor = FiberAnchorDiagnosticSuppressor{
            suppressor.cell->cellZYX,
            suppressorComponent.diagnosticId,
            false,
            suppressorComponent.anchor.alignedSupport,
            suppressorComponent.anchor.directionalCoherence,
        };
        component.retained = false;
        component.rejectionReason = "nms_suppressed";
        --candidates[index].cell->retainedAnchorCount;
    }
}

} // namespace

std::optional<FiberAnchorQuadraticPeakOffset> fitFiberAnchorQuadraticPeak(
    const std::array<std::array<double, 3>, 3>& response)
{
    CompensatedSum sum;
    CompensatedSum firstMoment;
    CompensatedSum secondMoment;
    CompensatedSum firstSquaredMoment;
    CompensatedSum mixedMoment;
    CompensatedSum secondSquaredMoment;
    double responseScale = 1.0;
    for (int first = -1; first <= 1; ++first) {
        for (int second = -1; second <= 1; ++second) {
            const double value = response[static_cast<size_t>(first + 1)]
                                         [static_cast<size_t>(second + 1)];
            if (!std::isfinite(value))
                return std::nullopt;
            responseScale = std::max(responseScale, std::abs(value));
            sum.add(value);
            firstMoment.add(static_cast<double>(first) * value);
            secondMoment.add(static_cast<double>(second) * value);
            firstSquaredMoment.add(
                static_cast<double>(first * first) * value);
            mixedMoment.add(static_cast<double>(first * second) * value);
            secondSquaredMoment.add(
                static_cast<double>(second * second) * value);
        }
    }

    // Least-squares model a + bx + cy + dx^2 + exy + fy^2 on {-1,0,1}^2.
    const double gradientFirst = firstMoment.sum / 6.0;
    const double gradientSecond = secondMoment.sum / 6.0;
    const double hessianFirstFirst =
        firstSquaredMoment.sum - 2.0 * sum.sum / 3.0;
    const double hessianFirstSecond = mixedMoment.sum / 4.0;
    const double hessianSecondSecond =
        secondSquaredMoment.sum - 2.0 * sum.sum / 3.0;
    const std::array<double, 5> coefficients{
        gradientFirst,
        gradientSecond,
        hessianFirstFirst,
        hessianFirstSecond,
        hessianSecondSecond,
    };
    if (std::any_of(coefficients.begin(), coefficients.end(),
                    [](double value) { return !std::isfinite(value); })) {
        return std::nullopt;
    }

    constexpr double kRelativeCurvatureTolerance = 1.0e-12;
    const double curvatureTolerance =
        kRelativeCurvatureTolerance * responseScale;
    const double trace = hessianFirstFirst + hessianSecondSecond;
    const double discriminant = std::hypot(
        hessianFirstFirst - hessianSecondSecond,
        2.0 * hessianFirstSecond);
    const double largestEigenvalue = 0.5 * (trace + discriminant);
    const double determinant =
        hessianFirstFirst * hessianSecondSecond -
        hessianFirstSecond * hessianFirstSecond;
    if (!(largestEigenvalue < -curvatureTolerance) ||
        !(determinant > curvatureTolerance * curvatureTolerance) ||
        !std::isfinite(determinant)) {
        return std::nullopt;
    }

    const double first =
        (hessianFirstSecond * gradientSecond -
         hessianSecondSecond * gradientFirst) / determinant;
    const double second =
        (hessianFirstSecond * gradientFirst -
         hessianFirstFirst * gradientSecond) / determinant;
    if (!std::isfinite(first) || !std::isfinite(second) ||
        std::abs(first) > 0.5 || std::abs(second) > 0.5) {
        return std::nullopt;
    }
    return FiberAnchorQuadraticPeakOffset{first, second};
}

const char* fiberAnchorDiagnosticStageName(FiberAnchorDiagnosticStage stage)
{
    switch (stage) {
    case FiberAnchorDiagnosticStage::Initialized:
        return "initialized";
    case FiberAnchorDiagnosticStage::Refined:
        return "refined";
    case FiberAnchorDiagnosticStage::Support:
        return "support";
    case FiberAnchorDiagnosticStage::Selection:
        return "selection";
    case FiberAnchorDiagnosticStage::Nms:
        return "nms";
    case FiberAnchorDiagnosticStage::Count:
        break;
    }
    throw std::invalid_argument("invalid fiber anchor diagnostic stage");
}

void validateFiberAnchorConfig(const FiberAnchorConfig& config)
{
    if (config.cellSizePredictionVoxels < 2 || config.cellSizePredictionVoxels > 8)
        throw std::invalid_argument("fiber anchor cell size must be in [2, 8]");
    if (!(config.gaussianSigmaPredictionVoxels > 0.0) ||
        !std::isfinite(config.gaussianSigmaPredictionVoxels)) {
        throw std::invalid_argument("fiber anchor Gaussian sigma must be positive and finite");
    }
    if (!(config.peakSigmaPredictionVoxels > 0.0) ||
        !std::isfinite(config.peakSigmaPredictionVoxels)) {
        throw std::invalid_argument("fiber anchor peak sigma must be positive and finite");
    }
    if (!(config.peakGradientWeight >= 0.0) ||
        !std::isfinite(config.peakGradientWeight)) {
        throw std::invalid_argument(
            "fiber anchor peak gradient weight must be nonnegative and finite");
    }
    if (!(config.peakGradientReliabilityScale > 0.0) ||
        !std::isfinite(config.peakGradientReliabilityScale)) {
        throw std::invalid_argument(
            "fiber anchor peak gradient reliability scale must be positive and finite");
    }
    if (!(config.peakAxialSigmaPredictionVoxels > 0.0) ||
        !std::isfinite(config.peakAxialSigmaPredictionVoxels)) {
        throw std::invalid_argument(
            "fiber anchor peak axial sigma must be positive and finite");
    }
    if (!(config.peakGridStepPredictionVoxels > 0.0) ||
        !std::isfinite(config.peakGridStepPredictionVoxels)) {
        throw std::invalid_argument("fiber anchor peak grid step must be positive and finite");
    }
    if (!(config.gaussianCutoffSigmas > 0.0) ||
        !std::isfinite(config.gaussianCutoffSigmas)) {
        throw std::invalid_argument("fiber anchor Gaussian cutoff must be positive and finite");
    }
    if (!(config.localWindowRadiusPredictionVoxels > 0.0) ||
        !std::isfinite(config.localWindowRadiusPredictionVoxels)) {
        throw std::invalid_argument("fiber anchor local window must be positive and finite");
    }
    if (config.localWindowRadiusPredictionVoxels /
            config.peakGridStepPredictionVoxels >
        128.0) {
        throw std::invalid_argument(
            "fiber anchor peak grid radius must not exceed 128 steps");
    }
    if (!(config.axialSupportHalfWidthPredictionVoxels > 0.0) ||
        !std::isfinite(config.axialSupportHalfWidthPredictionVoxels)) {
        throw std::invalid_argument("fiber anchor axial support must be positive and finite");
    }
    if (!(config.positionConvergenceTolerancePredictionVoxels >= 0.0) ||
        !std::isfinite(config.positionConvergenceTolerancePredictionVoxels)) {
        throw std::invalid_argument("fiber anchor position tolerance must be finite and non-negative");
    }
    if (!(config.nmsMaximumAngleDegrees >= 0.0) ||
        !(config.nmsMaximumAngleDegrees <= 90.0) ||
        !std::isfinite(config.nmsMaximumAngleDegrees)) {
        throw std::invalid_argument("fiber anchor NMS angle must be in [0, 90]");
    }
    if (!(config.nmsTransverseRadiusPredictionVoxels >= 0.0) ||
        !std::isfinite(config.nmsTransverseRadiusPredictionVoxels)) {
        throw std::invalid_argument("fiber anchor NMS transverse radius must be finite and non-negative");
    }
    if (!(config.nmsLongitudinalRadiusPredictionVoxels >= 0.0) ||
        !std::isfinite(config.nmsLongitudinalRadiusPredictionVoxels)) {
        throw std::invalid_argument("fiber anchor NMS longitudinal radius must be finite and non-negative");
    }
    if (!(config.observationPresenceFloor >= 0.0) ||
        !(config.observationPresenceFloor <= 1.0) ||
        !std::isfinite(config.observationPresenceFloor)) {
        throw std::invalid_argument("fiber anchor observation presence floor must be in [0, 1]");
    }
    if (!(config.minimumAlignedSupport >= 0.0) ||
        !(config.minimumAlignedSupport <= 1.0) ||
        !std::isfinite(config.minimumAlignedSupport)) {
        throw std::invalid_argument("fiber anchor minimum aligned support must be in [0, 1]");
    }
    if (!(config.mergeMaximumAngleDegrees >= 0.0) ||
        !(config.mergeMaximumAngleDegrees <= 90.0) ||
        !std::isfinite(config.mergeMaximumAngleDegrees)) {
        throw std::invalid_argument("fiber anchor merge maximum angle must be in [0, 90]");
    }
    if (!(config.mergeMaximumAbsoluteObjectiveLoss >= 0.0) ||
        !(config.mergeMaximumAbsoluteObjectiveLoss <= 1.0) ||
        !std::isfinite(config.mergeMaximumAbsoluteObjectiveLoss) ||
        !(config.mergeMaximumRelativeObjectiveLoss >= 0.0) ||
        !(config.mergeMaximumRelativeObjectiveLoss <= 1.0) ||
        !std::isfinite(config.mergeMaximumRelativeObjectiveLoss)) {
        throw std::invalid_argument("fiber anchor merge objective losses must be in [0, 1]");
    }
    if (config.maximumSeedCount < 1 || config.maximumSeedCount > 64)
        throw std::invalid_argument("fiber anchor maximum seed count must be in [1, 64]");
    if (config.maximumIterations < 1)
        throw std::invalid_argument("fiber anchor maximum iterations must be positive");
    if (!(config.convergenceTolerance >= 0.0) ||
        !std::isfinite(config.convergenceTolerance)) {
        throw std::invalid_argument("fiber anchor convergence tolerance must be finite and non-negative");
    }
    if (config.maximumConcurrentSampleBytes == 0) {
        throw std::invalid_argument(
            "fiber anchor concurrent sample byte limit must be positive");
    }
    if (config.parallelThreads < 1)
        throw std::invalid_argument("fiber anchor thread count must be positive");
}

FiberAnchorCrop fiberAnchorCropFromBaseVoxels(
    const FiberAnchorCrop& baseCrop,
    double predictionToBaseScale)
{
    if (!(predictionToBaseScale > 0.0) || !std::isfinite(predictionToBaseScale))
        throw std::invalid_argument("fiber anchor base crop requires a positive prediction-to-base scale");

    FiberAnchorCrop predictionCrop;
    for (size_t axis = 0; axis < 3; ++axis) {
        const size_t origin = baseCrop.originXYZ[axis];
        const size_t extent = baseCrop.sizeXYZ[axis];
        if (extent == 0 || origin > std::numeric_limits<size_t>::max() - extent)
            throw std::invalid_argument("fiber anchor base crop must be non-empty and must not overflow");
        const auto snappedCeil = [&](size_t baseBoundary) {
            long double scaled = static_cast<long double>(baseBoundary) / predictionToBaseScale;
            const long double nearest = std::round(scaled);
            const long double tolerance = 1.0e-12L * std::max(1.0L, std::abs(scaled));
            if (std::abs(scaled - nearest) <= tolerance)
                scaled = nearest;
            return std::ceil(scaled);
        };
        const long double predictionBegin = snappedCeil(origin);
        const long double predictionEnd = snappedCeil(origin + extent);
        if (predictionBegin < 0.0L ||
            predictionEnd > static_cast<long double>(std::numeric_limits<size_t>::max())) {
            throw std::invalid_argument("fiber anchor base crop exceeds the prediction index range");
        }
        predictionCrop.originXYZ[axis] = static_cast<size_t>(predictionBegin);
        predictionCrop.sizeXYZ[axis] = static_cast<size_t>(predictionEnd - predictionBegin);
        if (predictionCrop.sizeXYZ[axis] == 0)
            throw std::invalid_argument("fiber anchor base crop contains no prediction-grid sample");
    }
    return predictionCrop;
}

void suppressFiberAnchorDuplicates(
    std::vector<FiberCellAnchorResult>& cells,
    const FiberAnchorConfig& config)
{
    validateFiberAnchorConfig(config);
    applyLocalMaximumNms(cells, config);
}

FiberCellAnchorResult fitFiberCellAnchors(
    const std::array<size_t, 3>& cellZYX,
    const std::array<size_t, 3>& cellBeginZYX,
    const std::array<size_t, 3>& cellEndZYX,
    const std::vector<FiberAnchorObservation>& input,
    const FiberAnchorConfig& config)
{
    validateFiberAnchorConfig(config);
    const size_t expected =
        (cellEndZYX[0] - cellBeginZYX[0]) *
        (cellEndZYX[1] - cellBeginZYX[1]) *
        (cellEndZYX[2] - cellBeginZYX[2]);
    const auto isOwned = [&cellBeginZYX, &cellEndZYX](const cv::Vec3d& position) {
        return position[0] >= static_cast<double>(cellBeginZYX[2]) &&
            position[0] < static_cast<double>(cellEndZYX[2]) &&
            position[1] >= static_cast<double>(cellBeginZYX[1]) &&
            position[1] < static_cast<double>(cellEndZYX[1]) &&
            position[2] >= static_cast<double>(cellBeginZYX[0]) &&
            position[2] < static_cast<double>(cellEndZYX[0]);
    };
    const size_t ownedCount = static_cast<size_t>(std::count_if(
        input.begin(), input.end(), [&](const auto& observation) {
            return finiteVector(observation.positionPredictionXYZ) &&
                isOwned(observation.positionPredictionXYZ);
        }));
    if (ownedCount != expected)
        throw std::invalid_argument("fiber anchor observations do not cover the owned cell voxels exactly once");

    FiberCellAnchorResult result;
    result.cellZYX = cellZYX;
    for (size_t componentIndex = 0;
         componentIndex < result.components.size(); ++componentIndex) {
        auto& component = result.components[componentIndex];
        component.anchor.cellZYX = cellZYX;
        auto& diagnostic = result.initializedDiagnostics[componentIndex];
        diagnostic.cellZYX = cellZYX;
        diagnostic.candidateId = componentIndex;
        diagnostic.transition.outcome = "rejected";
        diagnostic.transition.reason = "empty";
    }
    const cv::Vec3d center{
        (static_cast<double>(cellBeginZYX[2]) +
            static_cast<double>(cellEndZYX[2]) - 1.0) * 0.5,
        (static_cast<double>(cellBeginZYX[1]) +
            static_cast<double>(cellEndZYX[1]) - 1.0) * 0.5,
        (static_cast<double>(cellBeginZYX[0]) +
            static_cast<double>(cellEndZYX[0]) - 1.0) * 0.5,
    };
    const double invTwoSigma2 = 1.0 /
        (2.0 * config.gaussianSigmaPredictionVoxels *
         config.gaussianSigmaPredictionVoxels);
    CompensatedSum denominator;
    std::vector<WeightedObservation> observations;
    observations.reserve(input.size());
    for (size_t index = 0; index < input.size(); ++index) {
        const auto& candidate = input[index];
        if (!finiteVector(candidate.positionPredictionXYZ) ||
            !isOwned(candidate.positionPredictionXYZ)) {
            continue;
        }
        const cv::Vec3d delta = candidate.positionPredictionXYZ - center;
        const double gaussian = std::exp(-delta.dot(delta) * invTwoSigma2);
        denominator.add(gaussian);
        if (!candidate.valid ||
            !finiteVector(candidate.direction) || !std::isfinite(candidate.presence) ||
            candidate.presence < config.observationPresenceFloor ||
            candidate.presence < 0.0 || candidate.presence > 1.0) {
            continue;
        }
        const cv::Vec3d direction = normalized(candidate.direction);
        if (direction.dot(direction) <= kMatrixEpsilon)
            continue;
        observations.push_back({
            candidate.positionPredictionXYZ,
            direction,
            gaussian,
            gaussian * candidate.presence,
            index,
        });
    }
    if (observations.empty()) {
        for (auto& component : result.components)
            component.rejectionReason = "empty";
        return result;
    }

    const FiberPrincipalAxis global = principalFiberAxis(weightedTensor(observations, nullptr, 0));
    std::vector<cv::Vec3d> seeds;
    seeds.reserve(config.maximumSeedCount);
    if (global.unique) {
        seeds.push_back(global.axis);
    } else {
        size_t best = 0;
        for (size_t index = 1; index < observations.size(); ++index) {
            if (observations[index].weight > observations[best].weight)
                best = index;
        }
        seeds.push_back(canonicalFiberAxis(observations[best].direction));
    }
    std::vector<bool> selected(observations.size(), false);
    while (seeds.size() < config.maximumSeedCount) {
        size_t best = observations.size();
        double bestScore = 0.0;
        for (size_t index = 0; index < observations.size(); ++index) {
            if (selected[index])
                continue;
            double minimumDissimilarity = 1.0;
            for (const auto& seed : seeds) {
                const double dot = observations[index].direction.dot(seed);
                minimumDissimilarity = std::min(
                    minimumDissimilarity, std::max(0.0, 1.0 - dot * dot));
            }
            const double score = observations[index].weight * minimumDissimilarity;
            if (score > bestScore) {
                bestScore = score;
                best = index;
            }
        }
        if (best == observations.size() || !(bestScore > 0.0))
            break;
        selected[best] = true;
        seeds.push_back(canonicalFiberAxis(observations[best].direction));
    }

    FitState bestFit;
    bestFit.objectiveNumerator = -1.0;
    if (seeds.size() == 1) {
        bestFit = refineSeedPair(observations, {seeds[0], seeds[0]}, config);
    } else {
        for (size_t first = 0; first + 1 < seeds.size(); ++first) {
            for (size_t second = first + 1; second < seeds.size(); ++second) {
                const FitState fit = refineSeedPair(
                    observations, {seeds[first], seeds[second]}, config);
                if (bestFit.objectiveNumerator < 0.0 || betterState(fit, bestFit))
                    bestFit = fit;
            }
        }
    }
    const double seededObjective = denominator.sum > 0.0
        ? bestFit.objectiveNumerator / denominator.sum
        : 0.0;

    const std::array<FiberPrincipalAxis, 2> fittedComponents{
        principalFiberAxis(weightedTensor(observations, &bestFit.assignments, 0)),
        principalFiberAxis(weightedTensor(observations, &bestFit.assignments, 1)),
    };
    std::array<size_t, 2> fittedAssignedCounts{0, 0};
    for (const uint8_t assignment : bestFit.assignments) {
        if (assignment < fittedAssignedCounts.size())
            ++fittedAssignedCounts[assignment];
    }
    for (size_t componentIndex = 0;
         componentIndex < fittedComponents.size(); ++componentIndex) {
        auto& diagnostic = result.initializedDiagnostics[componentIndex];
        const auto& fitted = fittedComponents[componentIndex];
        diagnostic.metrics.assignedObservationCount =
            fittedAssignedCounts[componentIndex];
        if (fitted.unique) {
            FiberAnchor anchor;
            anchor.cellZYX = cellZYX;
            anchor.positionPredictionXYZ = center;
            anchor.axisXYZ = fitted.axis;
            diagnostic.anchor = anchor;
            diagnostic.metrics.objectiveContribution = denominator.sum > 0.0
                ? fitted.largestEigenvalue / denominator.sum
                : 0.0;
            diagnostic.transition.outcome = "continue";
            diagnostic.transition.reason.reset();
        } else {
            diagnostic.transition.reason = std::any_of(
                bestFit.assignments.begin(), bestFit.assignments.end(),
                [componentIndex](uint8_t assigned) {
                    return assigned == componentIndex;
                }) ? "degenerate" : "empty";
        }
    }
    bool mergeComponents = false;
    if (denominator.sum > 0.0 && global.unique &&
        fittedComponents[0].unique && fittedComponents[1].unique) {
        const double splitObjective =
            (fittedComponents[0].largestEigenvalue +
             fittedComponents[1].largestEigenvalue) / denominator.sum;
        const double jointObjective = global.largestEigenvalue / denominator.sum;
        const double objectiveLoss = std::max(0.0, splitObjective - jointObjective);
        const double allowedObjectiveLoss = std::max(
            config.mergeMaximumAbsoluteObjectiveLoss,
            config.mergeMaximumRelativeObjectiveLoss * jointObjective);
        const double axialDot = std::clamp(
            std::abs(fittedComponents[0].axis.dot(fittedComponents[1].axis)),
            0.0,
            1.0);
        const double angleDegrees =
            std::acos(axialDot) * 180.0 / std::acos(-1.0);
        mergeComponents =
            angleDegrees <= config.mergeMaximumAngleDegrees &&
            objectiveLoss <= allowedObjectiveLoss;
        result.mergeEvaluation = FiberAnchorMergeEvaluation{
            angleDegrees,
            jointObjective,
            splitObjective,
            objectiveLoss,
            allowedObjectiveLoss,
            mergeComponents,
        };
    }

    if (mergeComponents) {
        result.objective = result.mergeEvaluation->jointObjective;
    } else {
        result.objective = seededObjective;
    }

    std::array<cv::Vec3d, 2> seedAxes{};
    std::array<size_t, 2> diagnosticIds{kNoDiagnosticId, kNoDiagnosticId};
    size_t activeComponents = 0;
    if (mergeComponents) {
        seedAxes[activeComponents] = global.axis;
        diagnosticIds[activeComponents++] = 2;
        for (auto& diagnostic : result.initializedDiagnostics) {
            if (!diagnostic.anchor.has_value())
                continue;
            diagnostic.transition.outcome = "merged";
            diagnostic.transition.reason = "merged_same_direction";
            diagnostic.transition.successorId = 2;
        }
    } else {
        for (uint8_t componentIndex = 0; componentIndex < 2; ++componentIndex) {
            const auto& fitted = fittedComponents[componentIndex];
            if (fitted.unique) {
                seedAxes[activeComponents] = fitted.axis;
                diagnosticIds[activeComponents++] = componentIndex;
            } else {
                result.components[componentIndex].rejectionReason = std::any_of(
                    bestFit.assignments.begin(), bestFit.assignments.end(),
                    [componentIndex](uint8_t assigned) {
                        return assigned == componentIndex;
                    }) ? "degenerate" : "empty";
            }
        }
    }
    if (activeComponents == 0)
        return result;

    RefinedFitState refined = refineLocalComponents(
        input, center, seedAxes, activeComponents, config);
    const PeakOwnerBounds owner = peakOwnerBounds(
        input, cellBeginZYX, cellEndZYX);
    const auto broadRefinedComponents = refined.components;
    for (size_t componentIndex = 0; componentIndex < activeComponents;
         ++componentIndex) {
        const auto peak = findDirectionConditionedLocalPeak(
            input, center, owner, broadRefinedComponents,
            activeComponents, componentIndex, config);
        result.components[componentIndex]
            .discretePeakPositionPredictionXYZ = peak.discrete;
        result.components[componentIndex]
            .separablePeakPositionPredictionXYZ = peak.separable1d;
        result.components[componentIndex]
            .jointPeakPositionPredictionXYZ = peak.joint2d;
        refined.components[componentIndex].position = peak.separable1d;
    }
    refined.evaluation = evaluateRefinedState(
        input, refined.components, activeComponents, center, config);
    result.objective = refined.evaluation.objective;
    for (size_t componentIndex = 0; componentIndex < activeComponents; ++componentIndex) {
        auto& component = result.components[componentIndex];
        component.diagnosticId = diagnosticIds[componentIndex];
        component.diagnosticParentIds = mergeComponents
            ? std::vector<size_t>{0, 1}
            : std::vector<size_t>{diagnosticIds[componentIndex]};
        const double denominatorValue = refined.evaluation.denominators[componentIndex];
        const double numeratorValue = refined.evaluation.numerators[componentIndex];
        const double presenceMass = refined.evaluation.presenceMasses[componentIndex];
        component.assignedObservationCount =
            refined.evaluation.assignedCounts[componentIndex];
        component.anchor.axisXYZ =
            canonicalFiberAxis(refined.components[componentIndex].axis);
        component.anchor.positionPredictionXYZ =
            refined.components[componentIndex].position;
        component.anchor.alignedSupport = denominatorValue > 0.0
            ? numeratorValue / denominatorValue
            : 0.0;
        component.anchor.directionalCoherence = presenceMass > 0.0
            ? numeratorValue / presenceMass
            : 0.0;
        component.anchor.refinementScore = component.anchor.alignedSupport;
        component.anchor.refinementIterations = refined.acceptedIterations;
        if (component.assignedObservationCount == 0 || !(numeratorValue > 0.0)) {
            component.rejectionReason = "empty";
        } else if (component.anchor.alignedSupport < config.minimumAlignedSupport) {
            component.rejectionReason = "below_support";
        } else {
            component.retained = true;
            ++result.retainedAnchorCount;
        }
        component.retainedAfterSupport = component.retained;
    }
    if (mergeComponents)
        result.components[1].rejectionReason = "merged_same_direction";
    else if (activeComponents == 1 && result.components[1].rejectionReason.empty())
        result.components[1].rejectionReason = "empty";
    if (componentLess(result.components[1], result.components[0]))
        std::swap(result.components[0], result.components[1]);
    return result;
}

static FiberAnchorExtractionReport extractFiberAnchorsImpl(
    const FiberPredictionGridInfo& grid,
    const FiberAnchorConfig& config,
    const FiberStoredPredictionBatchSampler& sampler,
    std::optional<FiberAnchorCrop> crop,
    std::vector<std::array<size_t, 3>> explicitCells,
    const FiberAnchorRetainPredicate& retainPredicate,
    const FiberAnchorProgressCallback& progressCallback,
    bool refinedOnly = false)
{
    validateFiberAnchorConfig(config);
    if (!sampler)
        throw std::invalid_argument("fiber anchor extraction requires a prediction sampler");
    if (!(grid.predictionToBaseScale > 0.0) ||
        !std::isfinite(grid.predictionToBaseScale) ||
        std::any_of(grid.shapeZYX.begin(), grid.shapeZYX.end(),
                    [](size_t value) { return value == 0; })) {
        throw std::invalid_argument("fiber anchor extraction requires a valid prediction grid");
    }

    FiberAnchorExtractionReport report;
    report.grid = grid;
    report.config = config;
    const size_t cellSize = static_cast<size_t>(config.cellSizePredictionVoxels);
    const std::array<size_t, 3> totalCellsZYX{
        ceilDivide(grid.shapeZYX[0], cellSize),
        ceilDivide(grid.shapeZYX[1], cellSize),
        ceilDivide(grid.shapeZYX[2], cellSize),
    };
    const bool usesExplicitCells = !explicitCells.empty();
    if (usesExplicitCells) {
        std::sort(explicitCells.begin(), explicitCells.end());
        if (std::adjacent_find(explicitCells.begin(), explicitCells.end()) !=
            explicitCells.end()) {
            throw std::invalid_argument("fiber anchor cells must be unique");
        }
        report.selectedCellsZYX = explicitCells;
        report.selectedCellBeginZYX = explicitCells.front();
        report.selectedCellEndZYX = explicitCells.front();
        for (size_t axis = 0; axis < 3; ++axis)
            ++report.selectedCellEndZYX[axis];
        for (const auto& cell : explicitCells) {
            for (size_t axis = 0; axis < 3; ++axis) {
                if (cell[axis] >= totalCellsZYX[axis])
                    throw std::invalid_argument("fiber anchor cell lies outside the prediction grid");
                report.selectedCellBeginZYX[axis] =
                    std::min(report.selectedCellBeginZYX[axis], cell[axis]);
                report.selectedCellEndZYX[axis] =
                    std::max(report.selectedCellEndZYX[axis], cell[axis] + 1);
            }
        }
        const std::array<size_t, 3> beginZYX{
            report.selectedCellBeginZYX[0] * cellSize,
            report.selectedCellBeginZYX[1] * cellSize,
            report.selectedCellBeginZYX[2] * cellSize,
        };
        const std::array<size_t, 3> endZYX{
            std::min(grid.shapeZYX[0], report.selectedCellEndZYX[0] * cellSize),
            std::min(grid.shapeZYX[1], report.selectedCellEndZYX[1] * cellSize),
            std::min(grid.shapeZYX[2], report.selectedCellEndZYX[2] * cellSize),
        };
        report.selectedCrop = {
            {beginZYX[2], beginZYX[1], beginZYX[0]},
            {endZYX[2] - beginZYX[2], endZYX[1] - beginZYX[1],
             endZYX[0] - beginZYX[0]},
        };
    } else if (!crop.has_value()) {
        report.selectedCrop.originXYZ = {0, 0, 0};
        report.selectedCrop.sizeXYZ = {
            grid.shapeZYX[2], grid.shapeZYX[1], grid.shapeZYX[0]};
    } else {
        report.selectedCrop = *crop;
    }
    for (size_t xyz = 0; xyz < 3; ++xyz) {
        const size_t zyx = 2 - xyz;
        const size_t origin = report.selectedCrop.originXYZ[xyz];
        const size_t extent = report.selectedCrop.sizeXYZ[xyz];
        if (extent == 0 || origin > grid.shapeZYX[zyx] ||
            extent > grid.shapeZYX[zyx] - origin) {
            throw std::invalid_argument("fiber anchor crop must be non-empty and inside the prediction grid");
        }
    }

    const std::array<size_t, 3> cropBeginZYX = {
        report.selectedCrop.originXYZ[2],
        report.selectedCrop.originXYZ[1],
        report.selectedCrop.originXYZ[0],
    };
    const std::array<size_t, 3> cropEndZYX = {
        cropBeginZYX[0] + report.selectedCrop.sizeXYZ[2],
        cropBeginZYX[1] + report.selectedCrop.sizeXYZ[1],
        cropBeginZYX[2] + report.selectedCrop.sizeXYZ[0],
    };
    if (!usesExplicitCells) {
        for (size_t axis = 0; axis < 3; ++axis) {
            report.selectedCellBeginZYX[axis] = cropBeginZYX[axis] / cellSize;
            report.selectedCellEndZYX[axis] = ceilDivide(cropEndZYX[axis], cellSize);
        }
    }

    const auto checkedProduct = [](const std::array<size_t, 3>& size,
                                    const char* description) {
        size_t result = 1;
        for (const size_t extent : size) {
            if (extent != 0 && result > std::numeric_limits<size_t>::max() / extent)
                throw std::overflow_error(std::string(description) + " size overflows");
            result *= extent;
        }
        return result;
    };
    const double broadTransverseSupport =
        config.localWindowRadiusPredictionVoxels +
        config.gaussianCutoffSigmas * config.gaussianSigmaPredictionVoxels;
    const double broadSupportRadius = std::hypot(
        broadTransverseSupport,
        config.axialSupportHalfWidthPredictionVoxels);
    const double peakTransverseSupport =
        config.localWindowRadiusPredictionVoxels +
        config.gaussianCutoffSigmas * config.peakSigmaPredictionVoxels;
    const double peakSupportRadius = std::hypot(
        peakTransverseSupport,
        config.gaussianCutoffSigmas *
            config.peakAxialSigmaPredictionVoxels);
    const double maximumSupportRadius =
        std::max(broadSupportRadius, peakSupportRadius);
    const size_t sampleHalo =
        static_cast<size_t>(std::ceil(maximumSupportRadius)) +
        (config.peakGradientWeight > 0.0 ? 1 : 0);
    const std::set<std::array<size_t, 3>> explicitCellSet(
        explicitCells.begin(), explicitCells.end());
    const auto selectedCell = [&report, &explicitCellSet, usesExplicitCells](const std::array<size_t, 3>& cell) {
        if (usesExplicitCells)
            return explicitCellSet.contains(cell);
        for (size_t axis = 0; axis < 3; ++axis) {
            if (cell[axis] < report.selectedCellBeginZYX[axis] ||
                cell[axis] >= report.selectedCellEndZYX[axis]) {
                return false;
            }
        }
        return true;
    };

    const auto start = std::chrono::steady_clock::now();
    using CellIndex = std::array<size_t, 3>;
    std::vector<CellIndex> selectedCells = explicitCells;
    if (!usesExplicitCells) {
        for (size_t cz = report.selectedCellBeginZYX[0];
             cz < report.selectedCellEndZYX[0]; ++cz) {
            for (size_t cy = report.selectedCellBeginZYX[1];
                 cy < report.selectedCellEndZYX[1]; ++cy) {
                for (size_t cx = report.selectedCellBeginZYX[2];
                     cx < report.selectedCellEndZYX[2]; ++cx) {
                    selectedCells.push_back({cz, cy, cx});
                }
            }
        }
    }
    std::set<CellIndex> workCellSet(selectedCells.begin(), selectedCells.end());
    if (!refinedOnly) {
        const double nmsDistance = std::hypot(
            config.nmsTransverseRadiusPredictionVoxels,
            config.nmsLongitudinalRadiusPredictionVoxels);
        const double pivotReach =
            2.0 * config.localWindowRadiusPredictionVoxels + nmsDistance;
        const size_t cellRadius = static_cast<size_t>(std::ceil(
            pivotReach / static_cast<double>(cellSize))) + 1;
        const auto cellPivot = [&](const CellIndex& cell) {
            cv::Vec3d pivot;
            for (size_t axis = 0; axis < 3; ++axis) {
                const size_t begin = cell[axis] * cellSize;
                const size_t end = std::min(
                    begin + cellSize, grid.shapeZYX[axis]);
                pivot[static_cast<int>(2 - axis)] =
                    (static_cast<double>(begin) +
                     static_cast<double>(end) - 1.0) * 0.5;
            }
            return pivot;
        };
        for (const auto& selected : selectedCells) {
            const cv::Vec3d selectedPivot = cellPivot(selected);
            CellIndex begin{};
            CellIndex end{};
            for (size_t axis = 0; axis < 3; ++axis) {
                begin[axis] = selected[axis] > cellRadius
                    ? selected[axis] - cellRadius : 0;
                end[axis] = std::min(
                    totalCellsZYX[axis],
                    selected[axis] + std::min(
                        cellRadius + 1,
                        totalCellsZYX[axis] - selected[axis]));
            }
            for (size_t cz = begin[0]; cz < end[0]; ++cz) {
                for (size_t cy = begin[1]; cy < end[1]; ++cy) {
                    for (size_t cx = begin[2]; cx < end[2]; ++cx) {
                        const CellIndex candidate{cz, cy, cx};
                        const cv::Vec3d delta =
                            cellPivot(candidate) - selectedPivot;
                        if (delta.dot(delta) <=
                            pivotReach * pivotReach + 1.0e-12) {
                            workCellSet.insert(candidate);
                        }
                    }
                }
            }
        }
    }
    const std::vector<CellIndex> workCells(
        workCellSet.begin(), workCellSet.end());

    const auto processCells = [&](const std::vector<std::array<size_t, 3>>& requestedCells,
                                  bool tallySelectedDiagnostics,
                                  const char* phase) {
        const auto phaseStart = std::chrono::steady_clock::now();
        if (progressCallback)
            progressCallback({phase, 0, requestedCells.size(), 0.0});
        if (requestedCells.empty())
            return std::vector<FiberCellAnchorResult>{};

        const auto sampleBounds = [&](const CellIndex& cell) {
            const CellIndex begin{
                cell[0] * cellSize,
                cell[1] * cellSize,
                cell[2] * cellSize,
            };
            const CellIndex end{
                std::min(begin[0] + cellSize, grid.shapeZYX[0]),
                std::min(begin[1] + cellSize, grid.shapeZYX[1]),
                std::min(begin[2] + cellSize, grid.shapeZYX[2]),
            };
            std::array<size_t, 3> sampleBegin{};
            std::array<size_t, 3> sampleEnd{};
            for (size_t axis = 0; axis < 3; ++axis) {
                sampleBegin[axis] =
                    begin[axis] > sampleHalo ? begin[axis] - sampleHalo : 0;
                sampleEnd[axis] = std::min(
                    grid.shapeZYX[axis],
                    end[axis] +
                        std::min(sampleHalo, grid.shapeZYX[axis] - end[axis]));
            }
            return std::pair{sampleBegin, sampleEnd};
        };

        struct Tile {
            std::vector<size_t> cells;
            CellIndex sampleBegin{};
            CellIndex sampleEnd{};
            size_t estimatedBytes = 0;
        };
        constexpr size_t kTileCellsPerAxis = 4;
        std::map<CellIndex, std::vector<size_t>> tileCells;
        for (size_t index = 0; index < requestedCells.size(); ++index) {
            const auto& cell = requestedCells[index];
            tileCells[{cell[0] / kTileCellsPerAxis,
                       cell[1] / kTileCellsPerAxis,
                       cell[2] / kTileCellsPerAxis}].push_back(index);
        }
        const auto makeTile = [&](std::vector<size_t> cells) {
            Tile tile;
            tile.cells = std::move(cells);
            tile.sampleBegin = grid.shapeZYX;
            tile.sampleEnd = {0, 0, 0};
            size_t maximumCellSamples = 0;
            for (const size_t index : tile.cells) {
                const auto [begin, end] = sampleBounds(requestedCells[index]);
                std::array<size_t, 3> shape{};
                for (size_t axis = 0; axis < 3; ++axis) {
                    tile.sampleBegin[axis] =
                        std::min(tile.sampleBegin[axis], begin[axis]);
                    tile.sampleEnd[axis] =
                        std::max(tile.sampleEnd[axis], end[axis]);
                    shape[axis] = end[axis] - begin[axis];
                }
                maximumCellSamples = std::max(
                    maximumCellSamples,
                    checkedProduct(shape, "fiber anchor cell sample"));
            }
            const std::array<size_t, 3> tileShape{
                tile.sampleEnd[0] - tile.sampleBegin[0],
                tile.sampleEnd[1] - tile.sampleBegin[1],
                tile.sampleEnd[2] - tile.sampleBegin[2],
            };
            const size_t tileSamples =
                checkedProduct(tileShape, "fiber anchor tile sample");
            constexpr size_t denseBytes =
                sizeof(CellIndex) + sizeof(FiberStoredPredictionSample);
            tile.estimatedBytes = tileSamples >
                    std::numeric_limits<size_t>::max() / denseBytes
                ? std::numeric_limits<size_t>::max()
                : tileSamples * denseBytes;
            const size_t scratchBytes = maximumCellSamples >
                    std::numeric_limits<size_t>::max() /
                        sizeof(FiberAnchorObservation)
                ? std::numeric_limits<size_t>::max()
                : maximumCellSamples * sizeof(FiberAnchorObservation);
            if (tile.estimatedBytes >
                    std::numeric_limits<size_t>::max() - scratchBytes) {
                tile.estimatedBytes = std::numeric_limits<size_t>::max();
            } else {
                tile.estimatedBytes += scratchBytes;
            }
            return tile;
        };
        std::vector<Tile> tiles;
        for (auto& [key, cells] : tileCells) {
            (void)key;
            std::vector<std::vector<size_t>> pending;
            pending.push_back(std::move(cells));
            while (!pending.empty()) {
                auto current = std::move(pending.back());
                pending.pop_back();
                Tile tile = makeTile(std::move(current));
                if (tile.estimatedBytes <=
                    config.maximumConcurrentSampleBytes) {
                    tiles.push_back(std::move(tile));
                    continue;
                }
                if (tile.cells.size() == 1) {
                    throw std::runtime_error(
                        "fiber anchor cell sample exceeds the concurrent byte limit");
                }
                size_t splitAxis = 0;
                size_t largestSpan = 0;
                for (size_t axis = 0; axis < 3; ++axis) {
                    const auto [minimum, maximum] = std::minmax_element(
                        tile.cells.begin(), tile.cells.end(),
                        [&](size_t left, size_t right) {
                            return requestedCells[left][axis] <
                                requestedCells[right][axis];
                        });
                    const size_t span = requestedCells[*maximum][axis] -
                        requestedCells[*minimum][axis];
                    if (span > largestSpan) {
                        largestSpan = span;
                        splitAxis = axis;
                    }
                }
                std::stable_sort(tile.cells.begin(), tile.cells.end(),
                    [&](size_t left, size_t right) {
                        return requestedCells[left][splitAxis] <
                            requestedCells[right][splitAxis];
                    });
                const size_t middle = tile.cells.size() / 2;
                std::vector<size_t> left(
                    tile.cells.begin(), tile.cells.begin() + middle);
                std::vector<size_t> right(
                    tile.cells.begin() + middle, tile.cells.end());
                pending.push_back(std::move(right));
                pending.push_back(std::move(left));
            }
        }
        std::sort(tiles.begin(), tiles.end(),
            [](const Tile& left, const Tile& right) {
                return left.cells.front() < right.cells.front();
            });
        size_t maximumTileBytes = 0;
        for (const auto& tile : tiles)
            maximumTileBytes = std::max(maximumTileBytes, tile.estimatedBytes);
        const size_t memoryWorkers = std::max<size_t>(
            1, config.maximumConcurrentSampleBytes / maximumTileBytes);
        const size_t workerCount = std::min({
            tiles.size(),
            static_cast<size_t>(config.parallelThreads),
            memoryWorkers,
        });

        std::vector<std::optional<FiberCellAnchorResult>> jobResults(
            requestedCells.size());
        std::vector<std::exception_ptr> jobErrors(requestedCells.size());
        const auto processCell = [&]
            (const CellIndex& cellZYX,
             const Tile& tile,
             const std::vector<FiberStoredPredictionSample>& samples,
             const std::array<size_t, 3>& sampleShape) {
            const std::array<size_t, 3> begin{
                cellZYX[0] * cellSize,
                cellZYX[1] * cellSize,
                cellZYX[2] * cellSize,
            };
            const std::array<size_t, 3> end{
                std::min(begin[0] + cellSize, grid.shapeZYX[0]),
                std::min(begin[1] + cellSize, grid.shapeZYX[1]),
                std::min(begin[2] + cellSize, grid.shapeZYX[2]),
            };
            const cv::Vec3d pivot{
                (static_cast<double>(begin[2]) + static_cast<double>(end[2]) -
                 1.0) *
                    0.5,
                (static_cast<double>(begin[1]) + static_cast<double>(end[1]) -
                 1.0) *
                    0.5,
                (static_cast<double>(begin[0]) + static_cast<double>(end[0]) -
                 1.0) *
                    0.5,
            };
            const auto [cellSampleBegin, cellSampleEnd] = sampleBounds(cellZYX);
            const size_t plane = sampleShape[1] * sampleShape[2];
            const auto tileIndex = [&](size_t z, size_t y, size_t x) {
                return (z - tile.sampleBegin[0]) * plane +
                    (y - tile.sampleBegin[1]) * sampleShape[2] +
                    (x - tile.sampleBegin[2]);
            };
            const auto presenceGradient = [&](size_t z, size_t y, size_t x) {
                if (z == cellSampleBegin[0] || z + 1 >= cellSampleEnd[0] ||
                    y == cellSampleBegin[1] || y + 1 >= cellSampleEnd[1] ||
                    x == cellSampleBegin[2] || x + 1 >= cellSampleEnd[2]) {
                    return std::optional<cv::Vec3d>{};
                }
                constexpr std::array<double, 3> smooth{0.25, 0.5, 0.25};
                constexpr std::array<double, 3> derivative{-0.5, 0.0, 0.5};
                cv::Vec3d gradient{0.0, 0.0, 0.0};
                for (int dz = -1; dz <= 1; ++dz) {
                    for (int dy = -1; dy <= 1; ++dy) {
                        for (int dx = -1; dx <= 1; ++dx) {
                            const size_t index = tileIndex(z, y, x);
                            const size_t neighbor = static_cast<size_t>(
                                static_cast<std::ptrdiff_t>(index) +
                                static_cast<std::ptrdiff_t>(dz) *
                                    static_cast<std::ptrdiff_t>(plane) +
                                static_cast<std::ptrdiff_t>(dy) *
                                    static_cast<std::ptrdiff_t>(sampleShape[2]) +
                                dx);
                            const auto& sample = samples[neighbor];
                            if (!(sample.presenceValid || sample.valid) ||
                                !std::isfinite(sample.presence)) {
                                return std::optional<cv::Vec3d>{};
                            }
                            const double presence = sample.presence;
                            gradient[0] += presence * derivative[dx + 1] *
                                smooth[dy + 1] * smooth[dz + 1];
                            gradient[1] += presence * smooth[dx + 1] *
                                derivative[dy + 1] * smooth[dz + 1];
                            gradient[2] += presence * smooth[dx + 1] *
                                smooth[dy + 1] * derivative[dz + 1];
                        }
                    }
                }
                return std::optional<cv::Vec3d>{gradient};
            };
            std::vector<FiberAnchorObservation> cellObservations;
            const std::array<size_t, 3> cellSampleShape{
                cellSampleEnd[0] - cellSampleBegin[0],
                cellSampleEnd[1] - cellSampleBegin[1],
                cellSampleEnd[2] - cellSampleBegin[2],
            };
            cellObservations.reserve(checkedProduct(
                cellSampleShape, "fiber anchor observation"));
            for (size_t z = cellSampleBegin[0]; z < cellSampleEnd[0]; ++z) {
                for (size_t y = cellSampleBegin[1]; y < cellSampleEnd[1]; ++y) {
                    for (size_t x = cellSampleBegin[2]; x < cellSampleEnd[2]; ++x) {
                        const size_t index = tileIndex(z, y, x);
                        FiberAnchorObservation observation{
                            cv::Vec3d{
                                static_cast<double>(x),
                                static_cast<double>(y),
                                static_cast<double>(z),
                            },
                            samples[index].direction,
                            samples[index].presence,
                            samples[index].valid,
                        };
                        const cv::Vec3d delta =
                            observation.positionPredictionXYZ - pivot;
                        const auto& position =
                            observation.positionPredictionXYZ;
                        const bool owned =
                            position[0] >= static_cast<double>(begin[2]) &&
                            position[0] < static_cast<double>(end[2]) &&
                            position[1] >= static_cast<double>(begin[1]) &&
                            position[1] < static_cast<double>(end[1]) &&
                            position[2] >= static_cast<double>(begin[0]) &&
                            position[2] < static_cast<double>(end[0]);
                        if (owned ||
                            delta.dot(delta) <=
                                maximumSupportRadius * maximumSupportRadius +
                                    1.0e-12) {
                            if (config.peakGradientWeight > 0.0) {
                                const auto gradient = presenceGradient(z, y, x);
                                if (gradient.has_value()) {
                                    observation.presenceGradientPredictionXYZ =
                                        *gradient;
                                    observation.presenceGradientValid = true;
                                }
                            }
                            cellObservations.push_back(observation);
                        }
                    }
                }
            }
            return fitFiberCellAnchors(
                cellZYX, begin, end, cellObservations, config);
        };

        std::atomic<size_t> nextJob{0};
        std::atomic<size_t> completedJobs{0};
        std::mutex progressMutex;
        auto lastProgressTime = phaseStart;
        std::exception_ptr progressError;
        const auto worker = [&]() {
            while (true) {
                const size_t job = nextJob.fetch_add(1);
                if (job >= tiles.size())
                    break;
                try {
                    const Tile& tile = tiles[job];
                    const std::array<size_t, 3> sampleShape{
                        tile.sampleEnd[0] - tile.sampleBegin[0],
                        tile.sampleEnd[1] - tile.sampleBegin[1],
                        tile.sampleEnd[2] - tile.sampleBegin[2],
                    };
                    const size_t sampleCount = checkedProduct(
                        sampleShape, "fiber anchor tile sample");
                    std::vector<CellIndex> indices;
                    indices.reserve(sampleCount);
                    for (size_t z = tile.sampleBegin[0]; z < tile.sampleEnd[0]; ++z) {
                        for (size_t y = tile.sampleBegin[1]; y < tile.sampleEnd[1]; ++y) {
                            for (size_t x = tile.sampleBegin[2]; x < tile.sampleEnd[2]; ++x)
                                indices.push_back({z, y, x});
                        }
                    }
                    std::vector<FiberStoredPredictionSample> samples;
                    sampler(indices, 1, samples);
                    if (samples.size() != indices.size()) {
                        throw std::runtime_error(
                            "fiber stored prediction sampler returned the wrong sample count");
                    }
                    indices.clear();
                    indices.shrink_to_fit();
                    for (const size_t cellIndex : tile.cells) {
                        jobResults[cellIndex] = processCell(
                            requestedCells[cellIndex], tile, samples, sampleShape);
                        const size_t completed = completedJobs.fetch_add(1) + 1;
                        if (progressCallback) {
                            const auto now = std::chrono::steady_clock::now();
                            std::lock_guard lock(progressMutex);
                            if (!progressError &&
                                now - lastProgressTime >= std::chrono::seconds(1) &&
                                completed < requestedCells.size()) {
                                try {
                                    progressCallback({
                                        phase,
                                        completed,
                                        requestedCells.size(),
                                        std::chrono::duration<double>(
                                            now - phaseStart).count(),
                                    });
                                    lastProgressTime = now;
                                } catch (...) {
                                    progressError = std::current_exception();
                                }
                            }
                        }
                    }
                } catch (...) {
                    for (const size_t cellIndex : tiles[job].cells)
                        jobErrors[cellIndex] = std::current_exception();
                }
            }
        };
        std::vector<std::thread> workers;
        workers.reserve(workerCount);
        for (size_t workerIndex = 0; workerIndex < workerCount; ++workerIndex)
            workers.emplace_back(worker);
        for (auto& thread : workers)
            thread.join();

        for (const auto& error : jobErrors) {
            if (error)
                std::rethrow_exception(error);
        }
        if (progressError)
            std::rethrow_exception(progressError);
        if (progressCallback) {
            progressCallback({
                phase,
                requestedCells.size(),
                requestedCells.size(),
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - phaseStart).count(),
            });
        }

        std::vector<FiberCellAnchorResult> results;
        results.reserve(requestedCells.size());
        for (auto& result : jobResults) {
            FiberCellAnchorResult cell = std::move(*result);
            const bool tallyCell =
                tallySelectedDiagnostics && selectedCell(cell.cellZYX);
            if (tallyCell && retainPredicate) {
                for (auto& component : cell.components) {
                    if (!component.retained)
                        continue;
                    const FiberAnchorRetainEvaluation evaluation =
                        retainPredicate(component.anchor);
                    if (!evaluation.retained &&
                        (!evaluation.testedValue.has_value() ||
                         !evaluation.threshold.has_value())) {
                        throw std::invalid_argument(
                            "fiber anchor selection rejection requires tested value and threshold");
                    }
                    component.selectionTestedValue = evaluation.testedValue;
                    component.selectionThreshold = evaluation.threshold;
                    if (!evaluation.retained) {
                        component.retained = false;
                        component.rejectionReason = "outside_selection";
                        --cell.retainedAnchorCount;
                        ++report.diagnostics.outsideSelectionComponents;
                    }
                }
            }
            if (tallyCell) {
                for (auto& component : cell.components)
                    component.retainedAfterSelection = component.retained;
                if (cell.mergeEvaluation.has_value() &&
                    cell.mergeEvaluation->merged) {
                    ++report.diagnostics.mergedComponentPairs;
                }
                for (const auto& component : cell.components) {
                    if (component.rejectionReason == "empty")
                        ++report.diagnostics.emptyComponents;
                    else if (component.rejectionReason == "degenerate")
                        ++report.diagnostics.degenerateComponents;
                    else if (component.rejectionReason == "below_support")
                        ++report.diagnostics.belowSupportComponents;
                }
            }
            if (cell.retainedAnchorCount > 0 || tallyCell)
                results.push_back(std::move(cell));
        }
        return results;
    };

    std::vector<FiberCellAnchorResult> contextResults = processCells(
        workCells, true, refinedOnly ? "selected_cells" : "anchor_cells");

    report.diagnostics.totalCells = usesExplicitCells
        ? explicitCells.size()
        : checkedProduct({
              report.selectedCellEndZYX[0] - report.selectedCellBeginZYX[0],
              report.selectedCellEndZYX[1] - report.selectedCellBeginZYX[1],
              report.selectedCellEndZYX[2] - report.selectedCellBeginZYX[2],
          }, "selected fiber anchor cell");
    const auto makeComponentRecord = [](const FiberCellAnchorResult& cell,
                                        const FiberAnchorComponent& component) {
        FiberAnchorDiagnosticRecord record;
        record.cellZYX = cell.cellZYX;
        record.candidateId = component.diagnosticId;
        record.parentIds = component.diagnosticParentIds;
        record.anchor = component.anchor;
        record.discretePeakPositionPredictionXYZ =
            component.discretePeakPositionPredictionXYZ;
        record.separablePeakPositionPredictionXYZ =
            component.separablePeakPositionPredictionXYZ;
        record.jointPeakPositionPredictionXYZ =
            component.jointPeakPositionPredictionXYZ;
        record.metrics.assignedObservationCount =
            component.assignedObservationCount;
        record.metrics.alignedSupport = component.anchor.alignedSupport;
        record.metrics.directionalCoherence =
            component.anchor.directionalCoherence;
        record.metrics.refinementScore = component.anchor.refinementScore;
        record.metrics.refinementIterations =
            component.anchor.refinementIterations;
        return record;
    };
    for (const auto& cell : contextResults) {
        if (!selectedCell(cell.cellZYX))
            continue;
        for (const auto& initialized : cell.initializedDiagnostics) {
            report.diagnosticStages[static_cast<size_t>(
                FiberAnchorDiagnosticStage::Initialized)].push_back(initialized);
        }
        for (const auto& component : cell.components) {
            if (component.diagnosticId == kNoDiagnosticId)
                continue;
            FiberAnchorDiagnosticRecord refined =
                makeComponentRecord(cell, component);
            if (component.retainedAfterSupport) {
                refined.transition.outcome = "continue";
            } else {
                refined.transition.outcome = "rejected";
                refined.transition.reason = component.rejectionReason;
                if (component.rejectionReason == "below_support") {
                    refined.transition.testedValue =
                        component.anchor.alignedSupport;
                    refined.transition.threshold = config.minimumAlignedSupport;
                }
            }
            report.diagnosticStages[static_cast<size_t>(
                FiberAnchorDiagnosticStage::Refined)].push_back(
                    std::move(refined));
        }
    }
    for (auto& stage : report.diagnosticStages) {
        std::sort(stage.begin(), stage.end(), [](const auto& left, const auto& right) {
            return std::tie(left.cellZYX, left.candidateId) <
                std::tie(right.cellZYX, right.candidateId);
        });
    }
    if (refinedOnly) {
        report.elapsedSeconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        return report;
    }

    suppressFiberAnchorDuplicates(contextResults, config);
    for (auto& cell : contextResults) {
        if (!selectedCell(cell.cellZYX))
            continue;
        for (auto& component : cell.components) {
            if (component.nmsSuppressor.has_value()) {
                component.nmsSuppressor->externalContext =
                    !selectedCell(component.nmsSuppressor->cellZYX);
            }
        }
        for (const auto& component : cell.components) {
            if (component.rejectionReason == "nms_suppressed")
                ++report.diagnostics.nmsSuppressedComponents;
        }
        for (const auto& component : cell.components) {
            if (component.diagnosticId == kNoDiagnosticId)
                continue;
            if (!component.retainedAfterSupport)
                continue;
            FiberAnchorDiagnosticRecord support =
                makeComponentRecord(cell, component);
            if (component.retainedAfterSelection) {
                support.transition.outcome = "continue";
            } else {
                support.transition.outcome = "rejected";
                support.transition.reason = "outside_selection";
                support.transition.testedValue = component.selectionTestedValue;
                support.transition.threshold = component.selectionThreshold;
            }
            report.diagnosticStages[static_cast<size_t>(
                FiberAnchorDiagnosticStage::Support)].push_back(
                    std::move(support));

            if (!component.retainedAfterSelection)
                continue;
            FiberAnchorDiagnosticRecord selection =
                makeComponentRecord(cell, component);
            if (component.retained) {
                selection.transition.outcome = "continue";
            } else {
                selection.transition.outcome = "rejected";
                selection.transition.reason = "nms_suppressed";
                selection.transition.suppressor = component.nmsSuppressor;
            }
            report.diagnosticStages[static_cast<size_t>(
                FiberAnchorDiagnosticStage::Selection)].push_back(
                    std::move(selection));

            if (!component.retained)
                continue;
            FiberAnchorDiagnosticRecord nms =
                makeComponentRecord(cell, component);
            nms.transition.outcome = "final";
            report.diagnosticStages[static_cast<size_t>(
                FiberAnchorDiagnosticStage::Nms)].push_back(std::move(nms));
        }
        if (componentLess(cell.components[1], cell.components[0]))
            std::swap(cell.components[0], cell.components[1]);
        if (cell.retainedAnchorCount == 1) {
            ++report.diagnostics.oneAnchorCells;
        } else if (cell.retainedAnchorCount == 2) {
            ++report.diagnostics.twoAnchorCells;
        }
        if (cell.retainedAnchorCount > 0)
            report.nonEmptyCells.push_back(std::move(cell));
    }
    report.diagnostics.zeroAnchorCells =
        report.diagnostics.totalCells - report.diagnostics.oneAnchorCells -
        report.diagnostics.twoAnchorCells;
    for (auto& stage : report.diagnosticStages) {
        std::sort(stage.begin(), stage.end(), [](const auto& left, const auto& right) {
            return std::tie(left.cellZYX, left.candidateId) <
                std::tie(right.cellZYX, right.candidateId);
        });
    }
    report.elapsedSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    return report;
}

FiberAnchorExtractionReport extractFiberAnchors(
    const FiberPredictionGridInfo& grid,
    const FiberAnchorConfig& config,
    const FiberStoredPredictionBatchSampler& sampler,
    std::optional<FiberAnchorCrop> crop,
    const FiberAnchorProgressCallback& progressCallback)
{
    return extractFiberAnchorsImpl(
        grid, config, sampler, crop, {}, {}, progressCallback, false);
}

FiberAnchorExtractionReport extractFiberAnchorsForCells(
    const FiberPredictionGridInfo& grid,
    const FiberAnchorConfig& config,
    const FiberStoredPredictionBatchSampler& sampler,
    std::vector<std::array<size_t, 3>> cellsZYX,
    const FiberAnchorRetainPredicate& retainPredicate,
    const FiberAnchorProgressCallback& progressCallback)
{
    if (cellsZYX.empty())
        throw std::invalid_argument("fiber anchor explicit cell selection must not be empty");
    return extractFiberAnchorsImpl(
        grid,
        config,
        sampler,
        std::nullopt,
        std::move(cellsZYX),
        retainPredicate,
        progressCallback,
        false);
}

FiberAnchorExtractionReport extractRefinedFiberAnchorsForCells(
    const FiberPredictionGridInfo& grid,
    const FiberAnchorConfig& config,
    const FiberStoredPredictionBatchSampler& sampler,
    std::vector<std::array<size_t, 3>> cellsZYX,
    const FiberAnchorProgressCallback& progressCallback)
{
    if (cellsZYX.empty()) {
        throw std::invalid_argument(
            "fiber anchor explicit cell selection must not be empty");
    }
    return extractFiberAnchorsImpl(
        grid,
        config,
        sampler,
        std::nullopt,
        std::move(cellsZYX),
        {},
        progressCallback,
        true);
}

std::vector<std::array<size_t, 3>> fiberAnchorCellsNearPolyline(
    const std::vector<cv::Vec3d>& referenceLineBase,
    double radiusBaseVoxels,
    const FiberPredictionGridInfo& grid,
    int anchorCellSizePredictionVoxels)
{
    if (!(radiusBaseVoxels >= 0.0) || !std::isfinite(radiusBaseVoxels) ||
        !(grid.predictionToBaseScale > 0.0) ||
        !std::isfinite(grid.predictionToBaseScale) ||
        anchorCellSizePredictionVoxels < 1) {
        throw std::invalid_argument("fiber anchor polyline cell selection is invalid");
    }
    const auto reference = makePolylineArcGeometry(referenceLineBase);
    const size_t cellSize = static_cast<size_t>(anchorCellSizePredictionVoxels);
    const std::array<size_t, 3> cellShape{
        (grid.shapeZYX[0] + cellSize - 1) / cellSize,
        (grid.shapeZYX[1] + cellSize - 1) / cellSize,
        (grid.shapeZYX[2] + cellSize - 1) / cellSize,
    };
    const double scale = grid.predictionToBaseScale;
    const double radiusSquared = radiusBaseVoxels * radiusBaseVoxels;
    std::set<std::array<size_t, 3>> cells;
    for (size_t segment = 0; segment + 1 < reference.points.size(); ++segment) {
        const cv::Vec3d& start = reference.points[segment];
        const cv::Vec3d& endPoint = reference.points[segment + 1];
        std::array<size_t, 3> cellBeginZYX{};
        std::array<size_t, 3> cellEndZYX{};
        for (size_t axis = 0; axis < 3; ++axis) {
            const size_t xyz = 2 - axis;
            const double gridSize = static_cast<double>(grid.shapeZYX[axis]);
            const double lowerPrediction = std::clamp(
                (std::min(start[xyz], endPoint[xyz]) - radiusBaseVoxels) /
                    scale,
                0.0,
                gridSize);
            const double upperPrediction = std::clamp(
                (std::max(start[xyz], endPoint[xyz]) + radiusBaseVoxels) /
                    scale,
                0.0,
                gridSize);
            const size_t coarseBegin = static_cast<size_t>(std::floor(
                lowerPrediction / static_cast<double>(cellSize)));
            const size_t coarseEnd = static_cast<size_t>(std::ceil(
                upperPrediction / static_cast<double>(cellSize)));
            cellBeginZYX[axis] = coarseBegin > 0 ? coarseBegin - 1 : 0;
            cellEndZYX[axis] = std::min(cellShape[axis], coarseEnd + 1);
        }
        for (size_t cz = cellBeginZYX[0]; cz < cellEndZYX[0]; ++cz) {
            for (size_t cy = cellBeginZYX[1]; cy < cellEndZYX[1]; ++cy) {
                for (size_t cx = cellBeginZYX[2]; cx < cellEndZYX[2]; ++cx) {
                    const std::array<size_t, 3> begin{
                        cz * cellSize, cy * cellSize, cx * cellSize};
                    const std::array<size_t, 3> end{
                        std::min(grid.shapeZYX[0], begin[0] + cellSize),
                        std::min(grid.shapeZYX[1], begin[1] + cellSize),
                        std::min(grid.shapeZYX[2], begin[2] + cellSize),
                    };
                    const cv::Vec3d cellLow{
                        (static_cast<double>(begin[2]) - 0.5) * scale,
                        (static_cast<double>(begin[1]) - 0.5) * scale,
                        (static_cast<double>(begin[0]) - 0.5) * scale,
                    };
                    const cv::Vec3d cellHigh{
                        (static_cast<double>(end[2]) - 0.5) * scale,
                        (static_cast<double>(end[1]) - 0.5) * scale,
                        (static_cast<double>(end[0]) - 0.5) * scale,
                    };
                    if (segmentAabbDistanceSquared(
                            start, endPoint, cellLow, cellHigh) <=
                        radiusSquared + kGeometryEpsilon) {
                        cells.insert({cz, cy, cx});
                    }
                }
            }
        }
    }
    if (cells.empty())
        throw std::runtime_error("reference fiber selects no prediction cells");
    return {cells.begin(), cells.end()};
}

FiberAnchorBenchmarkReport benchmarkRefinedFiberAnchors(
    const FiberAnchorExtractionReport& anchors,
    const std::vector<cv::Vec3d>& referenceLineBase,
    const std::vector<double>& thresholdsBaseVoxels)
{
    if (!(anchors.grid.predictionToBaseScale > 0.0) ||
        !std::isfinite(anchors.grid.predictionToBaseScale) ||
        anchors.selectedCellsZYX.empty()) {
        throw std::invalid_argument("anchor benchmark extraction report is invalid");
    }
    if (thresholdsBaseVoxels.empty() ||
        std::any_of(
            thresholdsBaseVoxels.begin(), thresholdsBaseVoxels.end(),
            [](double value) { return !(value >= 0.0) || !std::isfinite(value); })) {
        throw std::invalid_argument("anchor benchmark thresholds are invalid");
    }
    if (!std::is_sorted(
            anchors.selectedCellsZYX.begin(), anchors.selectedCellsZYX.end()) ||
        std::adjacent_find(
            anchors.selectedCellsZYX.begin(), anchors.selectedCellsZYX.end()) !=
            anchors.selectedCellsZYX.end()) {
        throw std::invalid_argument("anchor benchmark cells are not canonical");
    }
    const auto reference = makePolylineArcGeometry(referenceLineBase);
    const auto& refined = anchors.diagnosticStages[static_cast<size_t>(
        FiberAnchorDiagnosticStage::Refined)];
    enum class BenchmarkPeakStage { Discrete, Separable1d, Joint2d };
    const auto measureStage = [&](BenchmarkPeakStage stage) {
        std::map<std::array<size_t, 3>, std::vector<double>> distancesByCell;
        for (const auto& cell : anchors.selectedCellsZYX)
            distancesByCell.emplace(cell, std::vector<double>{});

        std::vector<double> distances;
        for (const auto& record : refined) {
            if (!record.anchor.has_value())
                continue;
            const auto cell = distancesByCell.find(record.cellZYX);
            if (cell == distancesByCell.end()) {
                throw std::invalid_argument(
                    "refined anchor belongs to an unselected benchmark cell");
            }
            if (!record.discretePeakPositionPredictionXYZ.has_value() ||
                !record.separablePeakPositionPredictionXYZ.has_value() ||
                !record.jointPeakPositionPredictionXYZ.has_value()) {
                throw std::invalid_argument(
                    "refined anchor lacks matched peak benchmark provenance");
            }
            cv::Vec3d pointPrediction;
            switch (stage) {
            case BenchmarkPeakStage::Discrete:
                pointPrediction = *record.discretePeakPositionPredictionXYZ;
                break;
            case BenchmarkPeakStage::Separable1d:
                pointPrediction = *record.separablePeakPositionPredictionXYZ;
                break;
            case BenchmarkPeakStage::Joint2d:
                pointPrediction = *record.jointPeakPositionPredictionXYZ;
                break;
            }
            const double distance = distanceToPolylineArc(
                reference,
                pointPrediction * anchors.grid.predictionToBaseScale,
                0.0,
                reference.length());
            cell->second.push_back(distance);
            distances.push_back(distance);
        }

        FiberAnchorBenchmarkStageReport result;
        result.referenceCells = anchors.selectedCellsZYX.size();
        result.cellsWithRefinedAnchors = static_cast<size_t>(std::count_if(
            distancesByCell.begin(), distancesByCell.end(),
            [](const auto& value) { return !value.second.empty(); }));
        result.refinedAnchors = distances.size();
        std::sort(distances.begin(), distances.end());
        result.anchorDistancesBaseVoxels.count = distances.size();
        if (!distances.empty()) {
            result.anchorDistancesBaseVoxels.minimum = distances.front();
            result.anchorDistancesBaseVoxels.maximum = distances.back();
            result.anchorDistancesBaseVoxels.mean =
                std::accumulate(distances.begin(), distances.end(), 0.0) /
                static_cast<double>(distances.size());
            result.anchorDistancesBaseVoxels.median =
                interpolatedQuantile(distances, 0.5);
            result.anchorDistancesBaseVoxels.percentile95 =
                interpolatedQuantile(distances, 0.95);
        }
        for (const double threshold : thresholdsBaseVoxels) {
            FiberAnchorBenchmarkThreshold measured;
            measured.thresholdBaseVoxels = threshold;
            measured.anchorHits = static_cast<size_t>(std::upper_bound(
                distances.begin(), distances.end(),
                threshold + kGeometryEpsilon) - distances.begin());
            if (!distances.empty()) {
                measured.anchorHitRate =
                    static_cast<double>(measured.anchorHits) /
                    static_cast<double>(distances.size());
            }
            for (const auto& [cell, cellDistances] : distancesByCell) {
                (void)cell;
                if (!cellDistances.empty() &&
                    *std::min_element(
                        cellDistances.begin(), cellDistances.end()) <=
                        threshold + kGeometryEpsilon) {
                    ++measured.cellHits;
                }
            }
            measured.cellHitRate = static_cast<double>(measured.cellHits) /
                static_cast<double>(result.referenceCells);
            result.thresholds.push_back(measured);
        }
        return result;
    };

    return {
        measureStage(BenchmarkPeakStage::Discrete),
        measureStage(BenchmarkPeakStage::Separable1d),
        measureStage(BenchmarkPeakStage::Joint2d),
    };
}


nlohmann::json fiberAnchorReportJson(
    const FiberAnchorExtractionReport& report,
    const FiberAnchorArtifactInfo& artifact)
{
    if (artifact.sourceLocator.empty() || artifact.manifestContentHash.empty())
        throw std::invalid_argument("fiber anchor artifacts require source identity and manifest hash");
    nlohmann::json root = {
        {"format", "vc_fiberlet_anchors"},
        {"version", 1},
        {"source", {
            {"manifest", artifact.sourceLocator},
            {"manifest_content_hash", artifact.manifestContentHash},
        }},
        {"coordinates", {
            {"position_order", "XYZ"},
            {"cell_index_order", "ZYX"},
            {"position_space", "base_volume"},
            {"prediction_to_base_scale", report.grid.predictionToBaseScale},
            {"prediction_shape_zyx", report.grid.shapeZYX},
        }},
        {"selection", {
            {"prediction_interval_origin_base_xyz", {
                report.selectedCrop.originXYZ[0] * report.grid.predictionToBaseScale,
                report.selectedCrop.originXYZ[1] * report.grid.predictionToBaseScale,
                report.selectedCrop.originXYZ[2] * report.grid.predictionToBaseScale,
            }},
            {"prediction_interval_size_base_xyz", {
                report.selectedCrop.sizeXYZ[0] * report.grid.predictionToBaseScale,
                report.selectedCrop.sizeXYZ[1] * report.grid.predictionToBaseScale,
                report.selectedCrop.sizeXYZ[2] * report.grid.predictionToBaseScale,
            }},
            {"cell_begin_zyx", report.selectedCellBeginZYX},
            {"cell_end_zyx", report.selectedCellEndZYX},
        }},
        {"parameters", {
            {"cell_size_prediction_voxels", report.config.cellSizePredictionVoxels},
            {"gaussian_sigma_prediction_voxels", report.config.gaussianSigmaPredictionVoxels},
            {"peak_sigma_prediction_voxels", report.config.peakSigmaPredictionVoxels},
            {"peak_axial_sigma_prediction_voxels", report.config.peakAxialSigmaPredictionVoxels},
            {"peak_grid_step_prediction_voxels", report.config.peakGridStepPredictionVoxels},
            {"peak_gradient_weight", report.config.peakGradientWeight},
            {"peak_gradient_reliability_scale", report.config.peakGradientReliabilityScale},
            {"gaussian_cutoff_sigmas", report.config.gaussianCutoffSigmas},
            {"local_window_radius_prediction_voxels", report.config.localWindowRadiusPredictionVoxels},
            {"axial_support_half_width_prediction_voxels", report.config.axialSupportHalfWidthPredictionVoxels},
            {"position_convergence_tolerance_prediction_voxels", report.config.positionConvergenceTolerancePredictionVoxels},
            {"nms_maximum_angle_degrees", report.config.nmsMaximumAngleDegrees},
            {"nms_transverse_radius_prediction_voxels", report.config.nmsTransverseRadiusPredictionVoxels},
            {"nms_longitudinal_radius_prediction_voxels", report.config.nmsLongitudinalRadiusPredictionVoxels},
            {"observation_presence_floor", report.config.observationPresenceFloor},
            {"minimum_aligned_support", report.config.minimumAlignedSupport},
            {"merge_maximum_angle_degrees", report.config.mergeMaximumAngleDegrees},
            {"merge_maximum_absolute_objective_loss", report.config.mergeMaximumAbsoluteObjectiveLoss},
            {"merge_maximum_relative_objective_loss", report.config.mergeMaximumRelativeObjectiveLoss},
            {"maximum_seed_count", report.config.maximumSeedCount},
            {"maximum_iterations", report.config.maximumIterations},
            {"convergence_tolerance", report.config.convergenceTolerance},
        }},
        {"diagnostics", {
            {"total_cells", report.diagnostics.totalCells},
            {"zero_anchor_cells", report.diagnostics.zeroAnchorCells},
            {"one_anchor_cells", report.diagnostics.oneAnchorCells},
            {"two_anchor_cells", report.diagnostics.twoAnchorCells},
            {"empty_components", report.diagnostics.emptyComponents},
            {"degenerate_components", report.diagnostics.degenerateComponents},
            {"below_support_components", report.diagnostics.belowSupportComponents},
            {"merged_component_pairs", report.diagnostics.mergedComponentPairs},
            {"nms_suppressed_components", report.diagnostics.nmsSuppressedComponents},
            {"outside_selection_components", report.diagnostics.outsideSelectionComponents},
        }},
        {"cells", nlohmann::json::array()},
    };
    if (!report.selectedCellsZYX.empty()) {
        root["selection"]["cells_zyx"] = report.selectedCellsZYX;
    }
    if (artifact.baseVoxelSizeUm.has_value()) {
        root["coordinates"]["base_voxel_size_um"] = *artifact.baseVoxelSizeUm;
    }
    for (const auto& cell : report.nonEmptyCells) {
        nlohmann::json cellJson = {
            {"cell_zyx", cell.cellZYX},
            {"objective", cell.objective},
            {"components", nlohmann::json::array()},
        };
        if (cell.mergeEvaluation.has_value()) {
            const auto& merge = *cell.mergeEvaluation;
            cellJson["merge_evaluation"] = {
                {"angle_degrees", merge.angleDegrees},
                {"joint_objective", merge.jointObjective},
                {"split_objective", merge.splitObjective},
                {"objective_loss", merge.objectiveLoss},
                {"allowed_objective_loss", merge.allowedObjectiveLoss},
                {"merged", merge.merged},
            };
        }
        for (const auto& component : cell.components) {
            nlohmann::json componentJson = {
                {"retained", component.retained},
                {"assigned_observations", component.assignedObservationCount},
            };
            if (component.retained) {
                const auto& anchor = component.anchor;
                const cv::Vec3d positionBase =
                    anchor.positionPredictionXYZ * report.grid.predictionToBaseScale;
                componentJson["position_base_xyz"] = {
                    positionBase[0], positionBase[1], positionBase[2]};
                componentJson["axis_xyz"] = {
                    anchor.axisXYZ[0], anchor.axisXYZ[1], anchor.axisXYZ[2]};
                componentJson["aligned_support"] = anchor.alignedSupport;
                componentJson["directional_coherence"] =
                    anchor.directionalCoherence;
                componentJson["refinement_score"] = anchor.refinementScore;
                componentJson["refinement_iterations"] =
                    anchor.refinementIterations;
            } else {
                componentJson["reason"] = component.rejectionReason;
            }
            cellJson["components"].push_back(std::move(componentJson));
        }
        root["cells"].push_back(std::move(cellJson));
    }
    return root;
}

namespace
{

std::vector<std::array<size_t, 3>> selectedAnchorCells(
    const FiberAnchorExtractionReport& report)
{
    std::vector<std::array<size_t, 3>> selected = report.selectedCellsZYX;
    if (!selected.empty())
        return selected;
    for (size_t z = report.selectedCellBeginZYX[0];
         z < report.selectedCellEndZYX[0]; ++z) {
        for (size_t y = report.selectedCellBeginZYX[1];
             y < report.selectedCellEndZYX[1]; ++y) {
            for (size_t x = report.selectedCellBeginZYX[2];
                 x < report.selectedCellEndZYX[2]; ++x) {
                selected.push_back({z, y, x});
            }
        }
    }
    return selected;
}

nlohmann::json nullableSize(const std::optional<size_t>& value)
{
    return value.has_value() ? nlohmann::json(*value) : nlohmann::json(nullptr);
}

nlohmann::json nullableDouble(const std::optional<double>& value)
{
    return value.has_value() ? nlohmann::json(*value) : nlohmann::json(nullptr);
}

nlohmann::json anchorDiagnosticParameters(const FiberAnchorConfig& config)
{
    return {
        {"cell_size_prediction_voxels", config.cellSizePredictionVoxels},
        {"gaussian_sigma_prediction_voxels", config.gaussianSigmaPredictionVoxels},
        {"peak_sigma_prediction_voxels", config.peakSigmaPredictionVoxels},
        {"peak_axial_sigma_prediction_voxels", config.peakAxialSigmaPredictionVoxels},
        {"peak_grid_step_prediction_voxels", config.peakGridStepPredictionVoxels},
        {"peak_gradient_weight", config.peakGradientWeight},
        {"peak_gradient_reliability_scale", config.peakGradientReliabilityScale},
        {"gaussian_cutoff_sigmas", config.gaussianCutoffSigmas},
        {"local_window_radius_prediction_voxels", config.localWindowRadiusPredictionVoxels},
        {"axial_support_half_width_prediction_voxels", config.axialSupportHalfWidthPredictionVoxels},
        {"position_convergence_tolerance_prediction_voxels", config.positionConvergenceTolerancePredictionVoxels},
        {"nms_maximum_angle_degrees", config.nmsMaximumAngleDegrees},
        {"nms_transverse_radius_prediction_voxels", config.nmsTransverseRadiusPredictionVoxels},
        {"nms_longitudinal_radius_prediction_voxels", config.nmsLongitudinalRadiusPredictionVoxels},
        {"observation_presence_floor", config.observationPresenceFloor},
        {"minimum_aligned_support", config.minimumAlignedSupport},
        {"merge_maximum_angle_degrees", config.mergeMaximumAngleDegrees},
        {"merge_maximum_absolute_objective_loss", config.mergeMaximumAbsoluteObjectiveLoss},
        {"merge_maximum_relative_objective_loss", config.mergeMaximumRelativeObjectiveLoss},
        {"maximum_seed_count", config.maximumSeedCount},
        {"maximum_iterations", config.maximumIterations},
        {"convergence_tolerance", config.convergenceTolerance},
    };
}

std::string fiberAnchorReportObjForComponent(
    const FiberAnchorExtractionReport& report,
    const FiberAnchorArtifactInfo& artifact,
    std::optional<size_t> selectedComponent)
{
    if (!(artifact.glyphLengthBaseVoxels > 0.0) ||
        !std::isfinite(artifact.glyphLengthBaseVoxels)) {
        throw std::invalid_argument("fiber anchor OBJ glyph length must be positive and finite");
    }
    std::ostringstream output;
    output << std::setprecision(std::numeric_limits<double>::max_digits10);
    output << "# vc_fiberlet_anchors version 1\n";
    size_t vertex = 1;
    for (const auto& cell : report.nonEmptyCells) {
        for (size_t componentIndex = 0;
             componentIndex < cell.components.size(); ++componentIndex) {
            if (selectedComponent.has_value() &&
                *selectedComponent != componentIndex) {
                continue;
            }
            const auto& component = cell.components[componentIndex];
            if (!component.retained)
                continue;
            const auto& anchor = component.anchor;
            const cv::Vec3d center = anchor.positionPredictionXYZ *
                report.grid.predictionToBaseScale;
            const cv::Vec3d half = anchor.axisXYZ *
                (artifact.glyphLengthBaseVoxels * 0.5);
            const cv::Vec3d first = center - half;
            const cv::Vec3d second = center + half;
            output << "g cell_" << cell.cellZYX[0] << '_' << cell.cellZYX[1]
                   << '_' << cell.cellZYX[2] << "_anchor_" << componentIndex << '\n';
            output << "v " << first[0] << ' ' << first[1] << ' ' << first[2] << '\n';
            output << "v " << second[0] << ' ' << second[1] << ' ' << second[2] << '\n';
            output << "l " << vertex << ' ' << vertex + 1 << '\n';
            vertex += 2;
        }
    }
    return output.str();
}

}  // namespace

nlohmann::json fiberAnchorDiagnosticStageJson(
    const FiberAnchorExtractionReport& report,
    const FiberAnchorArtifactInfo& artifact,
    FiberAnchorDiagnosticStage stage)
{
    if (artifact.sourceLocator.empty() || artifact.manifestContentHash.empty())
        throw std::invalid_argument("fiber anchor diagnostics require source identity and manifest hash");
    if (!(artifact.glyphLengthBaseVoxels > 0.0) ||
        !std::isfinite(artifact.glyphLengthBaseVoxels)) {
        throw std::invalid_argument("fiber anchor diagnostic glyph length must be positive and finite");
    }
    const size_t stageIndex = static_cast<size_t>(stage);
    if (stageIndex >= kFiberAnchorDiagnosticStageCount)
        throw std::invalid_argument("invalid fiber anchor diagnostic stage");

    nlohmann::json root = {
        {"format", "vc_fiberlet_anchor_stage"},
        {"version", 1},
        {"stage", fiberAnchorDiagnosticStageName(stage)},
        {"source", {
            {"manifest", artifact.sourceLocator},
            {"manifest_content_hash", artifact.manifestContentHash},
        }},
        {"coordinates", {
            {"position_order", "XYZ"},
            {"cell_index_order", "ZYX"},
            {"position_space", "base_volume"},
            {"prediction_to_base_scale", report.grid.predictionToBaseScale},
            {"prediction_shape_zyx", report.grid.shapeZYX},
        }},
        {"selection", {{"cells_zyx", selectedAnchorCells(report)}}},
        {"parameters", anchorDiagnosticParameters(report.config)},
        {"glyph_length_base_voxels", artifact.glyphLengthBaseVoxels},
        {"summary", {
            {"record_count", 0},
            {"geometric_record_count", 0},
            {"outcomes", nlohmann::json::object()},
            {"reasons", nlohmann::json::object()},
        }},
        {"records", nlohmann::json::array()},
    };
    if (artifact.baseVoxelSizeUm.has_value())
        root["coordinates"]["base_voxel_size_um"] = *artifact.baseVoxelSizeUm;

    const auto& records = report.diagnosticStages[stageIndex];
    for (const auto& record : records) {
        nlohmann::json geometry = nullptr;
        if (record.anchor.has_value()) {
            const FiberAnchor& anchor = *record.anchor;
            const cv::Vec3d positionBase =
                anchor.positionPredictionXYZ * report.grid.predictionToBaseScale;
            geometry = {
                {"position_base_xyz", {positionBase[0], positionBase[1], positionBase[2]}},
                {"axis_xyz", {anchor.axisXYZ[0], anchor.axisXYZ[1], anchor.axisXYZ[2]}},
            };
        }
        nlohmann::json suppressor = nullptr;
        if (record.transition.suppressor.has_value()) {
            const auto& value = *record.transition.suppressor;
            suppressor = {
                {"cell_zyx", value.cellZYX},
                {"candidate_id", value.candidateId},
                {"external_context", value.externalContext},
                {"aligned_support", value.alignedSupport},
                {"directional_coherence", value.directionalCoherence},
            };
        }
        root["records"].push_back({
            {"cell_zyx", record.cellZYX},
            {"candidate_id", record.candidateId},
            {"parent_ids", record.parentIds},
            {"geometry", std::move(geometry)},
            {"metrics", {
                {"assigned_observations", nullableSize(record.metrics.assignedObservationCount)},
                {"objective_contribution", nullableDouble(record.metrics.objectiveContribution)},
                {"aligned_support", nullableDouble(record.metrics.alignedSupport)},
                {"directional_coherence", nullableDouble(record.metrics.directionalCoherence)},
                {"refinement_score", nullableDouble(record.metrics.refinementScore)},
                {"refinement_iterations", nullableSize(record.metrics.refinementIterations)},
            }},
            {"transition", {
                {"outcome", record.transition.outcome},
                {"reason", record.transition.reason.has_value()
                    ? nlohmann::json(*record.transition.reason) : nlohmann::json(nullptr)},
                {"successor_id", nullableSize(record.transition.successorId)},
                {"tested_value", nullableDouble(record.transition.testedValue)},
                {"threshold", nullableDouble(record.transition.threshold)},
                {"suppressor", std::move(suppressor)},
            }},
        });
        root["summary"]["record_count"] =
            root["summary"]["record_count"].get<size_t>() + 1;
        if (record.anchor.has_value()) {
            root["summary"]["geometric_record_count"] =
                root["summary"]["geometric_record_count"].get<size_t>() + 1;
        }
        auto& outcomes = root["summary"]["outcomes"];
        outcomes[record.transition.outcome] =
            outcomes.value(record.transition.outcome, size_t{0}) + 1;
        if (record.transition.reason.has_value()) {
            auto& reasons = root["summary"]["reasons"];
            reasons[*record.transition.reason] =
                reasons.value(*record.transition.reason, size_t{0}) + 1;
        }
    }
    return root;
}

std::string fiberAnchorReportObj(
    const FiberAnchorExtractionReport& report,
    const FiberAnchorArtifactInfo& artifact)
{
    return fiberAnchorReportObjForComponent(report, artifact, std::nullopt);
}

std::string fiberAnchorCellReportObj(const FiberAnchorExtractionReport& report)
{
    if (!(report.grid.predictionToBaseScale > 0.0) ||
        !std::isfinite(report.grid.predictionToBaseScale) ||
        report.config.cellSizePredictionVoxels < 1) {
        throw std::invalid_argument("fiber anchor cell OBJ metadata is invalid");
    }
    const std::vector<std::array<size_t, 3>> selectedCells =
        selectedAnchorCells(report);
    std::map<std::array<size_t, 3>, const FiberCellAnchorResult*> retainedByCell;
    for (const auto& cell : report.nonEmptyCells)
        retainedByCell.emplace(cell.cellZYX, &cell);

    const size_t cellSize = static_cast<size_t>(
        report.config.cellSizePredictionVoxels);
    std::ostringstream output;
    output << std::setprecision(std::numeric_limits<double>::max_digits10);
    output << "# vc_fiberlet_anchor_cells version 1\n";
    size_t vertex = 1;
    for (const auto& cellZYX : selectedCells) {
        std::array<size_t, 3> begin{};
        std::array<size_t, 3> end{};
        for (size_t axis = 0; axis < 3; ++axis) {
            begin[axis] = cellZYX[axis] * cellSize;
            end[axis] = std::min(
                report.grid.shapeZYX[axis], begin[axis] + cellSize);
            if (begin[axis] >= end[axis])
                throw std::invalid_argument("fiber anchor cell lies outside its grid");
        }
        const cv::Vec3d centerPredictionXYZ{
            0.5 * static_cast<double>(begin[2] + end[2] - 1),
            0.5 * static_cast<double>(begin[1] + end[1] - 1),
            0.5 * static_cast<double>(begin[0] + end[0] - 1),
        };
        const cv::Vec3d centerBase = centerPredictionXYZ *
            report.grid.predictionToBaseScale;
        const size_t centerVertex = vertex++;
        output << "g cell_" << cellZYX[0] << '_' << cellZYX[1] << '_'
               << cellZYX[2] << '\n';
        output << "v " << centerBase[0] << ' ' << centerBase[1] << ' '
               << centerBase[2] << '\n';
        output << "p " << centerVertex << '\n';
        const auto retained = retainedByCell.find(cellZYX);
        if (retained == retainedByCell.end())
            continue;
        for (const auto& component : retained->second->components) {
            if (!component.retained)
                continue;
            const cv::Vec3d anchorBase = component.anchor.positionPredictionXYZ *
                report.grid.predictionToBaseScale;
            const size_t anchorVertex = vertex++;
            output << "v " << anchorBase[0] << ' ' << anchorBase[1] << ' '
                   << anchorBase[2] << '\n';
            output << "l " << centerVertex << ' ' << anchorVertex << '\n';
        }
    }
    return output.str();
}

void writeFiberAnchorArtifacts(
    const std::filesystem::path& outputDirectory,
    const FiberAnchorExtractionReport& report,
    const FiberAnchorArtifactInfo& artifact)
{
    if (outputDirectory.empty())
        throw std::invalid_argument("fiber anchor output directory must not be empty");
    std::filesystem::create_directories(outputDirectory);
    vc::core::util::atomicWriteString(
        outputDirectory / "anchors.json",
        fiberAnchorReportJson(report, artifact).dump(2) + "\n");
    vc::core::util::atomicWriteString(
        outputDirectory / "anchors.obj",
        fiberAnchorReportObj(report, artifact));
    vc::core::util::atomicWriteString(
        outputDirectory / "anchors_0.obj",
        fiberAnchorReportObjForComponent(report, artifact, 0));
    vc::core::util::atomicWriteString(
        outputDirectory / "anchors_1.obj",
        fiberAnchorReportObjForComponent(report, artifact, 1));
    vc::core::util::atomicWriteString(
        outputDirectory / "anchor_cells.obj",
        fiberAnchorCellReportObj(report));
    std::filesystem::create_directories(outputDirectory / "stages");
    for (size_t index = 0; index < kFiberAnchorDiagnosticStageCount; ++index) {
        const auto stage = static_cast<FiberAnchorDiagnosticStage>(index);
        vc::core::util::atomicWriteString(
            outputDirectory / "stages" /
                (std::string(fiberAnchorDiagnosticStageName(stage)) + ".json"),
            fiberAnchorDiagnosticStageJson(report, artifact, stage).dump(2) + "\n");
    }
}

} // namespace vc::fiber_tracer
