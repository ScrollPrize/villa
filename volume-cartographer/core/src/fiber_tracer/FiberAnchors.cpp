#include "vc/fiber_tracer/FiberAnchors.hpp"
#include "vc/core/util/AtomicFile.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <iomanip>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace vc::fiber_tracer {
namespace {

constexpr double kMatrixEpsilon = 1.0e-15;

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

struct PrincipalAxis {
    cv::Vec3d axis{0.0, 0.0, 0.0};
    double largestEigenvalue = 0.0;
    double secondEigenvalue = 0.0;
    bool valid = false;
    bool unique = false;
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

[[nodiscard]] cv::Vec3d canonicalAxis(cv::Vec3d axis)
{
    axis = normalized(axis);
    size_t signAxis = 0;
    double largestAbsolute = std::abs(axis[0]);
    for (size_t index = 1; index < 3; ++index) {
        const double candidate = std::abs(axis[static_cast<int>(index)]);
        if (candidate > largestAbsolute) {
            largestAbsolute = candidate;
            signAxis = index;
        }
    }
    if (axis[static_cast<int>(signAxis)] < 0.0)
        axis *= -1.0;
    return axis;
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

[[nodiscard]] PrincipalAxis principalAxis(const cv::Matx33d& input)
{
    cv::Matx33d matrix = input;
    cv::Matx33d eigenvectors = cv::Matx33d::eye();
    constexpr std::array<std::pair<int, int>, 3> rotations = {
        std::pair{0, 1}, std::pair{0, 2}, std::pair{1, 2}};
    for (int sweep = 0; sweep < 32; ++sweep) {
        bool changed = false;
        for (const auto [p, q] : rotations) {
            const double app = matrix(p, p);
            const double aqq = matrix(q, q);
            const double apq = matrix(p, q);
            const double scale = std::max({1.0, std::abs(app), std::abs(aqq)});
            if (std::abs(apq) <= kMatrixEpsilon * scale)
                continue;
            changed = true;
            const double tau = (aqq - app) / (2.0 * apq);
            const double sign = tau >= 0.0 ? 1.0 : -1.0;
            const double tangent = sign /
                (std::abs(tau) + std::sqrt(1.0 + tau * tau));
            const double cosine = 1.0 / std::sqrt(1.0 + tangent * tangent);
            const double sine = tangent * cosine;
            for (int index = 0; index < 3; ++index) {
                if (index == p || index == q)
                    continue;
                const double aip = matrix(index, p);
                const double aiq = matrix(index, q);
                matrix(index, p) = matrix(p, index) = cosine * aip - sine * aiq;
                matrix(index, q) = matrix(q, index) = sine * aip + cosine * aiq;
            }
            matrix(p, p) = cosine * cosine * app -
                2.0 * sine * cosine * apq + sine * sine * aqq;
            matrix(q, q) = sine * sine * app +
                2.0 * sine * cosine * apq + cosine * cosine * aqq;
            matrix(p, q) = matrix(q, p) = 0.0;
            for (int row = 0; row < 3; ++row) {
                const double vip = eigenvectors(row, p);
                const double viq = eigenvectors(row, q);
                eigenvectors(row, p) = cosine * vip - sine * viq;
                eigenvectors(row, q) = sine * vip + cosine * viq;
            }
        }
        if (!changed)
            break;
    }

    std::array<int, 3> order{0, 1, 2};
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        return matrix(a, a) > matrix(b, b);
    });
    PrincipalAxis result;
    result.largestEigenvalue = matrix(order[0], order[0]);
    result.secondEigenvalue = matrix(order[1], order[1]);
    result.axis = canonicalAxis({
        eigenvectors(0, order[0]),
        eigenvectors(1, order[0]),
        eigenvectors(2, order[0]),
    });
    result.valid = result.largestEigenvalue > kMatrixEpsilon &&
        finiteVector(result.axis);
    const double gapTolerance = 1.0e-12 *
        std::max(1.0, std::abs(result.largestEigenvalue));
    result.unique = result.valid &&
        result.largestEigenvalue - result.secondEigenvalue > gapTolerance;
    return result;
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
        state.components[component].axis = canonicalAxis(seedAxes[component]);
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
            const PrincipalAxis principal = principalAxis(tensor);
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
                interpolated[component].axis = canonicalAxis(normalized(
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
            const auto principal = principalAxis(
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
        transverse <= config.localWindowRadiusPredictionVoxels;
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
            config.localWindowRadiusPredictionVoxels,
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
    if (!(config.gaussianCutoffSigmas > 0.0) ||
        !std::isfinite(config.gaussianCutoffSigmas)) {
        throw std::invalid_argument("fiber anchor Gaussian cutoff must be positive and finite");
    }
    if (!(config.localWindowRadiusPredictionVoxels > 0.0) ||
        !std::isfinite(config.localWindowRadiusPredictionVoxels)) {
        throw std::invalid_argument("fiber anchor local window must be positive and finite");
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
    if (config.processingBlockCellSide < 1 ||
        config.processingBlockCellSide > 64) {
        throw std::invalid_argument("fiber anchor processing block side must be in [1, 64]");
    }
    if (config.maximumSampleBlockBytes == 0)
        throw std::invalid_argument("fiber anchor sample block byte limit must be positive");
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

    const PrincipalAxis global = principalAxis(weightedTensor(observations, nullptr, 0));
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
        seeds.push_back(canonicalAxis(observations[best].direction));
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
        seeds.push_back(canonicalAxis(observations[best].direction));
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

    const std::array<PrincipalAxis, 2> fittedComponents{
        principalAxis(weightedTensor(observations, &bestFit.assignments, 0)),
        principalAxis(weightedTensor(observations, &bestFit.assignments, 1)),
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

    const RefinedFitState refined = refineLocalComponents(
        input, center, seedAxes, activeComponents, config);
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
            canonicalAxis(refined.components[componentIndex].axis);
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
    const FiberAnchorProgressCallback& progressCallback)
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
    const double maximumTransverseSupport =
        config.localWindowRadiusPredictionVoxels +
        config.gaussianCutoffSigmas * config.gaussianSigmaPredictionVoxels;
    const double maximumSupportRadius = std::hypot(
        maximumTransverseSupport,
        config.axialSupportHalfWidthPredictionVoxels);
    const size_t sampleHalo = static_cast<size_t>(std::ceil(maximumSupportRadius));
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
    const size_t blockSide = config.processingBlockCellSide;
    const auto processCells = [&](const std::vector<std::array<size_t, 3>>& requestedCells,
                                  bool tallySelectedDiagnostics,
                                  const char* phase) {
        using CellIndex = std::array<size_t, 3>;
        std::map<CellIndex, std::vector<CellIndex>> cellsByBlock;
        for (const auto& cell : requestedCells) {
            cellsByBlock[CellIndex{
                             cell[0] / blockSide,
                             cell[1] / blockSide,
                             cell[2] / blockSide,
                         }]
                .push_back(cell);
        }

        std::vector<FiberCellAnchorResult> results;
        const auto phaseStart = std::chrono::steady_clock::now();
        auto lastProgressTime = phaseStart;
        size_t completedCells = 0;
        if (progressCallback)
            progressCallback({phase, 0, requestedCells.size(), 0.0});
        for (const auto& [blockIndex, blockCells] : cellsByBlock) {
            const CellIndex blockCellBegin{
                blockIndex[0] * blockSide,
                blockIndex[1] * blockSide,
                blockIndex[2] * blockSide,
            };
            const CellIndex blockCellEnd{
                std::min(totalCellsZYX[0], blockCellBegin[0] + blockSide),
                std::min(totalCellsZYX[1], blockCellBegin[1] + blockSide),
                std::min(totalCellsZYX[2], blockCellBegin[2] + blockSide),
            };
            const CellIndex blockBegin{
                blockCellBegin[0] * cellSize,
                blockCellBegin[1] * cellSize,
                blockCellBegin[2] * cellSize,
            };
            const std::array<size_t, 3> blockEnd{
                std::min(blockCellEnd[0] * cellSize, grid.shapeZYX[0]),
                std::min(blockCellEnd[1] * cellSize, grid.shapeZYX[1]),
                std::min(blockCellEnd[2] * cellSize, grid.shapeZYX[2]),
            };
            std::array<size_t, 3> sampleBegin{};
            std::array<size_t, 3> sampleEnd{};
            for (size_t axis = 0; axis < 3; ++axis) {
                sampleBegin[axis] = blockBegin[axis] > sampleHalo ? blockBegin[axis] - sampleHalo : 0;
                sampleEnd[axis] = std::min(grid.shapeZYX[axis], blockEnd[axis] + std::min(sampleHalo, grid.shapeZYX[axis] - blockEnd[axis]));
            }
            const std::array<size_t, 3> sampleShape{
                sampleEnd[0] - sampleBegin[0],
                sampleEnd[1] - sampleBegin[1],
                sampleEnd[2] - sampleBegin[2],
            };
            std::vector<std::array<size_t, 3>> indices;
            const size_t sampleCount = checkedProduct(sampleShape, "fiber anchor sample block");
            constexpr size_t bytesPerSample = sizeof(std::array<size_t, 3>) + sizeof(FiberStoredPredictionSample) + sizeof(FiberAnchorObservation);
            if (sampleCount > config.maximumSampleBlockBytes / bytesPerSample) {
                throw std::runtime_error("fiber anchor sample block exceeds the configured byte limit");
            }
            indices.reserve(sampleCount);
            for (size_t z = sampleBegin[0]; z < sampleEnd[0]; ++z) {
                for (size_t y = sampleBegin[1]; y < sampleEnd[1]; ++y) {
                    for (size_t x = sampleBegin[2]; x < sampleEnd[2]; ++x)
                        indices.push_back({z, y, x});
                }
            }
            std::vector<FiberStoredPredictionSample> samples;
            sampler(indices, config.parallelThreads, samples);
            if (samples.size() != indices.size()) {
                throw std::runtime_error("fiber stored prediction sampler returned the wrong sample count");
            }
            std::vector<FiberAnchorObservation> observations;
            observations.reserve(samples.size());
            for (size_t index = 0; index < samples.size(); ++index) {
                observations.push_back({
                    cv::Vec3d{
                        static_cast<double>(indices[index][2]),
                        static_cast<double>(indices[index][1]),
                        static_cast<double>(indices[index][0]),
                    },
                    samples[index].direction,
                    samples[index].presence,
                    samples[index].valid,
                });
            }
            for (const auto& cellZYX : blockCells) {
                const size_t cz = cellZYX[0];
                const size_t cy = cellZYX[1];
                const size_t cx = cellZYX[2];
                const std::array<size_t, 3> begin{cz * cellSize, cy * cellSize, cx * cellSize};
                const std::array<size_t, 3> end{
                    std::min(begin[0] + cellSize, grid.shapeZYX[0]),
                    std::min(begin[1] + cellSize, grid.shapeZYX[1]),
                    std::min(begin[2] + cellSize, grid.shapeZYX[2]),
                };
                const cv::Vec3d pivot{
                    (static_cast<double>(begin[2]) + static_cast<double>(end[2]) - 1.0) * 0.5,
                    (static_cast<double>(begin[1]) + static_cast<double>(end[1]) - 1.0) * 0.5,
                    (static_cast<double>(begin[0]) + static_cast<double>(end[0]) - 1.0) * 0.5,
                };
                std::vector<FiberAnchorObservation> cellObservations;
                for (const auto& observation : observations) {
                    const cv::Vec3d delta = observation.positionPredictionXYZ - pivot;
                    const auto& position = observation.positionPredictionXYZ;
                    const bool owned =
                        position[0] >= static_cast<double>(begin[2]) && position[0] < static_cast<double>(end[2]) &&
                        position[1] >= static_cast<double>(begin[1]) && position[1] < static_cast<double>(end[1]) &&
                        position[2] >= static_cast<double>(begin[0]) && position[2] < static_cast<double>(end[0]);
                    if (owned || delta.dot(delta) <= maximumSupportRadius * maximumSupportRadius + 1.0e-12) {
                        cellObservations.push_back(observation);
                    }
                }
                FiberCellAnchorResult cell = fitFiberCellAnchors(cellZYX, begin, end, cellObservations, config);
                if (tallySelectedDiagnostics && retainPredicate) {
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
                if (tallySelectedDiagnostics) {
                    for (auto& component : cell.components)
                        component.retainedAfterSelection = component.retained;
                }
                if (tallySelectedDiagnostics) {
                    if (cell.mergeEvaluation.has_value() && cell.mergeEvaluation->merged) {
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
                if (cell.retainedAnchorCount > 0 || tallySelectedDiagnostics)
                    results.push_back(std::move(cell));
            }
            completedCells += blockCells.size();
            const auto now = std::chrono::steady_clock::now();
            if (progressCallback &&
                (completedCells == requestedCells.size() ||
                 now - lastProgressTime >= std::chrono::seconds(1))) {
                progressCallback({
                    phase,
                    completedCells,
                    requestedCells.size(),
                    std::chrono::duration<double>(now - phaseStart).count(),
                });
                lastProgressTime = now;
            }
        }
        return results;
    };

    std::vector<std::array<size_t, 3>> selectedCells = explicitCells;
    if (!usesExplicitCells) {
        for (size_t cz = report.selectedCellBeginZYX[0]; cz < report.selectedCellEndZYX[0]; ++cz) {
            for (size_t cy = report.selectedCellBeginZYX[1]; cy < report.selectedCellEndZYX[1]; ++cy) {
                for (size_t cx = report.selectedCellBeginZYX[2]; cx < report.selectedCellEndZYX[2]; ++cx) {
                    selectedCells.push_back({cz, cy, cx});
                }
            }
        }
    }
    std::vector<FiberCellAnchorResult> contextResults = processCells(
        selectedCells, true, "selected_cells");

    const double nmsDistance = std::hypot(config.localWindowRadiusPredictionVoxels, config.nmsLongitudinalRadiusPredictionVoxels);
    const double contextPivotDistance = config.localWindowRadiusPredictionVoxels + nmsDistance;
    const size_t contextCellRadius = static_cast<size_t>(std::ceil(contextPivotDistance / static_cast<double>(cellSize))) + 1;
    std::set<std::array<size_t, 3>> externalContextCells;
    for (const auto& cell : contextResults) {
        for (const auto& component : cell.components) {
            if (!component.retained)
                continue;
            const auto& position = component.anchor.positionPredictionXYZ;
            const std::array<double, 3> positionZYX{position[2], position[1], position[0]};
            std::array<size_t, 3> centerCell{};
            std::array<size_t, 3> beginCell{};
            std::array<size_t, 3> endCell{};
            for (size_t axis = 0; axis < 3; ++axis) {
                centerCell[axis] = std::min(totalCellsZYX[axis] - 1, static_cast<size_t>(positionZYX[axis]) / cellSize);
                beginCell[axis] = centerCell[axis] > contextCellRadius ? centerCell[axis] - contextCellRadius : 0;
                endCell[axis] =
                    std::min(totalCellsZYX[axis], centerCell[axis] + std::min(contextCellRadius + 1, totalCellsZYX[axis] - centerCell[axis]));
            }
            for (size_t cz = beginCell[0]; cz < endCell[0]; ++cz) {
                for (size_t cy = beginCell[1]; cy < endCell[1]; ++cy) {
                    for (size_t cx = beginCell[2]; cx < endCell[2]; ++cx) {
                        const std::array<size_t, 3> candidate{cz, cy, cx};
                        if (selectedCell(candidate))
                            continue;
                        const std::array<size_t, 3> candidateBegin{cz * cellSize, cy * cellSize, cx * cellSize};
                        const std::array<size_t, 3> candidateEnd{
                            std::min(candidateBegin[0] + cellSize, grid.shapeZYX[0]),
                            std::min(candidateBegin[1] + cellSize, grid.shapeZYX[1]),
                            std::min(candidateBegin[2] + cellSize, grid.shapeZYX[2]),
                        };
                        const cv::Vec3d pivot{
                            (static_cast<double>(candidateBegin[2]) + static_cast<double>(candidateEnd[2]) - 1.0) * 0.5,
                            (static_cast<double>(candidateBegin[1]) + static_cast<double>(candidateEnd[1]) - 1.0) * 0.5,
                            (static_cast<double>(candidateBegin[0]) + static_cast<double>(candidateEnd[0]) - 1.0) * 0.5,
                        };
                        const cv::Vec3d delta = pivot - position;
                        if (delta.dot(delta) <= contextPivotDistance * contextPivotDistance + 1.0e-12) {
                            externalContextCells.insert(candidate);
                        }
                    }
                }
            }
        }
    }
    const std::vector<std::array<size_t, 3>> externalCells(
        externalContextCells.begin(), externalContextCells.end());
    auto externalResults = processCells(externalCells, false, "nms_context");
    contextResults.insert(
        contextResults.end(),
        std::make_move_iterator(externalResults.begin()),
        std::make_move_iterator(externalResults.end()));
    std::sort(contextResults.begin(), contextResults.end(),
        [](const auto& left, const auto& right) {
            return left.cellZYX < right.cellZYX;
    });
    suppressFiberAnchorDuplicates(contextResults, config);
    report.diagnostics.totalCells = usesExplicitCells
        ? explicitCells.size()
        : checkedProduct({
              report.selectedCellEndZYX[0] - report.selectedCellBeginZYX[0],
              report.selectedCellEndZYX[1] - report.selectedCellBeginZYX[1],
              report.selectedCellEndZYX[2] - report.selectedCellBeginZYX[2],
          }, "selected fiber anchor cell");
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
        for (const auto& initialized : cell.initializedDiagnostics) {
            report.diagnosticStages[static_cast<size_t>(
                FiberAnchorDiagnosticStage::Initialized)].push_back(initialized);
        }
        const auto makeComponentRecord = [&](const FiberAnchorComponent& component) {
            FiberAnchorDiagnosticRecord record;
            record.cellZYX = cell.cellZYX;
            record.candidateId = component.diagnosticId;
            record.parentIds = component.diagnosticParentIds;
            record.anchor = component.anchor;
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
        for (const auto& component : cell.components) {
            if (component.diagnosticId == kNoDiagnosticId)
                continue;
            FiberAnchorDiagnosticRecord refined = makeComponentRecord(component);
            if (component.retainedAfterSupport) {
                refined.transition.outcome = "continue";
            } else {
                refined.transition.outcome = "rejected";
                refined.transition.reason = component.rejectionReason;
                if (component.rejectionReason == "below_support") {
                    refined.transition.testedValue = component.anchor.alignedSupport;
                    refined.transition.threshold = config.minimumAlignedSupport;
                }
            }
            report.diagnosticStages[static_cast<size_t>(
                FiberAnchorDiagnosticStage::Refined)].push_back(
                    std::move(refined));

            if (!component.retainedAfterSupport)
                continue;
            FiberAnchorDiagnosticRecord support = makeComponentRecord(component);
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
            FiberAnchorDiagnosticRecord selection = makeComponentRecord(component);
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
            FiberAnchorDiagnosticRecord nms = makeComponentRecord(component);
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
        grid, config, sampler, crop, {}, {}, progressCallback);
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
        progressCallback);
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
            {"gaussian_cutoff_sigmas", report.config.gaussianCutoffSigmas},
            {"local_window_radius_prediction_voxels", report.config.localWindowRadiusPredictionVoxels},
            {"axial_support_half_width_prediction_voxels", report.config.axialSupportHalfWidthPredictionVoxels},
            {"position_convergence_tolerance_prediction_voxels", report.config.positionConvergenceTolerancePredictionVoxels},
            {"nms_maximum_angle_degrees", report.config.nmsMaximumAngleDegrees},
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
        {"gaussian_cutoff_sigmas", config.gaussianCutoffSigmas},
        {"local_window_radius_prediction_voxels", config.localWindowRadiusPredictionVoxels},
        {"axial_support_half_width_prediction_voxels", config.axialSupportHalfWidthPredictionVoxels},
        {"position_convergence_tolerance_prediction_voxels", config.positionConvergenceTolerancePredictionVoxels},
        {"nms_maximum_angle_degrees", config.nmsMaximumAngleDegrees},
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
