#include "vc/fiber_tracer/FiberAnchors.hpp"
#include "vc/core/util/AtomicFile.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <iomanip>
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

} // namespace

void validateFiberAnchorConfig(const FiberAnchorConfig& config)
{
    if (config.cellSizePredictionVoxels < 2 || config.cellSizePredictionVoxels > 8)
        throw std::invalid_argument("fiber anchor cell size must be in [2, 8]");
    if (!(config.gaussianSigmaPredictionVoxels > 0.0) ||
        !std::isfinite(config.gaussianSigmaPredictionVoxels)) {
        throw std::invalid_argument("fiber anchor Gaussian sigma must be positive and finite");
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
    if (input.size() != expected)
        throw std::invalid_argument("fiber anchor cell observations do not cover the owned cell voxels");

    FiberCellAnchorResult result;
    result.cellZYX = cellZYX;
    for (auto& component : result.components)
        component.anchor.cellZYX = cellZYX;
    const double size = static_cast<double>(config.cellSizePredictionVoxels);
    const cv::Vec3d center{
        static_cast<double>(cellZYX[2] * config.cellSizePredictionVoxels) +
            (size - 1.0) * 0.5,
        static_cast<double>(cellZYX[1] * config.cellSizePredictionVoxels) +
            (size - 1.0) * 0.5,
        static_cast<double>(cellZYX[0] * config.cellSizePredictionVoxels) +
            (size - 1.0) * 0.5,
    };
    const double invTwoSigma2 = 1.0 /
        (2.0 * config.gaussianSigmaPredictionVoxels *
         config.gaussianSigmaPredictionVoxels);
    CompensatedSum denominator;
    std::vector<WeightedObservation> observations;
    observations.reserve(input.size());
    for (size_t index = 0; index < input.size(); ++index) {
        const auto& candidate = input[index];
        const cv::Vec3d delta = candidate.positionPredictionXYZ - center;
        const double gaussian = std::exp(-delta.dot(delta) * invTwoSigma2);
        denominator.add(gaussian);
        if (!candidate.valid || !finiteVector(candidate.positionPredictionXYZ) ||
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
    result.objective = denominator.sum > 0.0
        ? bestFit.objectiveNumerator / denominator.sum
        : 0.0;

    const std::array<PrincipalAxis, 2> fittedComponents{
        principalAxis(weightedTensor(observations, &bestFit.assignments, 0)),
        principalAxis(weightedTensor(observations, &bestFit.assignments, 1)),
    };
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

    const auto populateComponent = [&result, &observations, &config, &denominator](
                                       uint8_t componentIndex,
                                       const cv::Vec3d& axis,
                                       const std::vector<uint8_t>* assignments) {
        auto& component = result.components[componentIndex];
        component.anchor.axisXYZ = canonicalAxis(axis);
        CompensatedSum aligned;
        CompensatedSum presenceMass;
        std::array<CompensatedSum, 3> position;
        for (size_t index = 0; index < observations.size(); ++index) {
            if (assignments != nullptr && (*assignments)[index] != componentIndex)
                continue;
            ++component.assignedObservationCount;
            const auto& observation = observations[index];
            const double dot = observation.direction.dot(component.anchor.axisXYZ);
            const double alignedWeight = observation.weight * dot * dot;
            aligned.add(alignedWeight);
            presenceMass.add(observation.weight);
            for (int axis = 0; axis < 3; ++axis)
                position[axis].add(alignedWeight * observation.position[axis]);
        }
        if (component.assignedObservationCount == 0 || !(aligned.sum > 0.0)) {
            component.rejectionReason = "empty";
            return;
        }
        component.anchor.alignedSupport = aligned.sum / denominator.sum;
        component.anchor.directionalCoherence = aligned.sum / presenceMass.sum;
        component.anchor.positionPredictionXYZ = {
            position[0].sum / aligned.sum,
            position[1].sum / aligned.sum,
            position[2].sum / aligned.sum,
        };
        if (component.anchor.alignedSupport < config.minimumAlignedSupport) {
            component.rejectionReason = "below_support";
            return;
        }
        component.retained = true;
        ++result.retainedAnchorCount;
    };

    if (mergeComponents) {
        result.objective = result.mergeEvaluation->jointObjective;
        populateComponent(0, global.axis, nullptr);
        result.components[1].rejectionReason = "merged_same_direction";
    } else {
        for (uint8_t componentIndex = 0; componentIndex < 2; ++componentIndex) {
            const auto& fitted = fittedComponents[componentIndex];
            if (!fitted.unique) {
                result.components[componentIndex].rejectionReason = std::any_of(
                    bestFit.assignments.begin(), bestFit.assignments.end(),
                    [componentIndex](uint8_t assigned) {
                        return assigned == componentIndex;
                    }) ? "degenerate" : "empty";
                continue;
            }
            populateComponent(
                componentIndex, fitted.axis, &bestFit.assignments);
        }
    }
    if (componentLess(result.components[1], result.components[0]))
        std::swap(result.components[0], result.components[1]);
    return result;
}

FiberAnchorExtractionReport extractFiberAnchors(
    const FiberPredictionGridInfo& grid,
    const FiberAnchorConfig& config,
    const FiberStoredPredictionBatchSampler& sampler,
    std::optional<FiberAnchorCrop> crop)
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
    if (!crop.has_value()) {
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

    const size_t cellSize = static_cast<size_t>(config.cellSizePredictionVoxels);
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
    for (size_t axis = 0; axis < 3; ++axis) {
        report.selectedCellBeginZYX[axis] = cropBeginZYX[axis] / cellSize;
        report.selectedCellEndZYX[axis] = ceilDivide(cropEndZYX[axis], cellSize);
    }

    const auto start = std::chrono::steady_clock::now();
    for (size_t cz = report.selectedCellBeginZYX[0];
         cz < report.selectedCellEndZYX[0]; ++cz) {
        for (size_t cy = report.selectedCellBeginZYX[1];
             cy < report.selectedCellEndZYX[1]; ++cy) {
            for (size_t cx = report.selectedCellBeginZYX[2];
                 cx < report.selectedCellEndZYX[2]; ++cx) {
                const std::array<size_t, 3> cellZYX{cz, cy, cx};
                const std::array<size_t, 3> begin{
                    cz * cellSize, cy * cellSize, cx * cellSize};
                const std::array<size_t, 3> end{
                    std::min(begin[0] + cellSize, grid.shapeZYX[0]),
                    std::min(begin[1] + cellSize, grid.shapeZYX[1]),
                    std::min(begin[2] + cellSize, grid.shapeZYX[2]),
                };
                std::vector<std::array<size_t, 3>> indices;
                indices.reserve(
                    (end[0] - begin[0]) * (end[1] - begin[1]) *
                    (end[2] - begin[2]));
                for (size_t z = begin[0]; z < end[0]; ++z) {
                    for (size_t y = begin[1]; y < end[1]; ++y) {
                        for (size_t x = begin[2]; x < end[2]; ++x)
                            indices.push_back({z, y, x});
                    }
                }
                std::vector<FiberStoredPredictionSample> samples;
                sampler(indices, config.parallelThreads, samples);
                if (samples.size() != indices.size()) {
                    throw std::runtime_error(
                        "fiber stored prediction sampler returned the wrong sample count");
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
                FiberCellAnchorResult cell = fitFiberCellAnchors(
                    cellZYX, begin, end, observations, config);
                ++report.diagnostics.totalCells;
                if (cell.retainedAnchorCount == 0)
                    ++report.diagnostics.zeroAnchorCells;
                else if (cell.retainedAnchorCount == 1)
                    ++report.diagnostics.oneAnchorCells;
                else
                    ++report.diagnostics.twoAnchorCells;
                for (const auto& component : cell.components) {
                    if (component.rejectionReason == "empty")
                        ++report.diagnostics.emptyComponents;
                    else if (component.rejectionReason == "degenerate")
                        ++report.diagnostics.degenerateComponents;
                    else if (component.rejectionReason == "below_support")
                        ++report.diagnostics.belowSupportComponents;
                    else if (component.rejectionReason == "merged_same_direction")
                        ++report.diagnostics.mergedComponentPairs;
                }
                if (cell.retainedAnchorCount > 0)
                    report.nonEmptyCells.push_back(std::move(cell));
            }
        }
    }
    report.elapsedSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    return report;
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
        }},
        {"cells", nlohmann::json::array()},
    };
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

std::string fiberAnchorReportObj(
    const FiberAnchorExtractionReport& report,
    const FiberAnchorArtifactInfo& artifact)
{
    return fiberAnchorReportObjForComponent(report, artifact, std::nullopt);
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
}

} // namespace vc::fiber_tracer
