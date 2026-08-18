#include "vc/fiber_tracer/FiberAnchors.hpp"
#include "vc/core/util/AtomicFile.hpp"
#include "vc/fiber_tracer/FiberAxisTensor.hpp"
#include "vc/fiber_tracer/PolylineGeometry.hpp"
#include "vc/fiber_tracer/detail/FiberAnchorSupportStencil.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
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

double processCpuSeconds()
{
    const std::clock_t ticks = std::clock();
    return ticks == static_cast<std::clock_t>(-1)
        ? 0.0
        : static_cast<double>(ticks) / static_cast<double>(CLOCKS_PER_SEC);
}

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
    std::vector<uint8_t> retainedInliers;
    std::array<double, 2> denominators{0.0, 0.0};
    std::array<double, 2> numerators{0.0, 0.0};
    std::array<double, 2> presenceMasses{0.0, 0.0};
    std::array<size_t, 2> assignedCounts{0, 0};
    double objective = 0.0;
};

struct RefinedFitState {
    std::array<RefinedComponentState, 2> components;
    std::array<size_t, 2> componentIds{0, 1};
    std::array<size_t, 2> removedComponentIds{
        std::numeric_limits<size_t>::max(),
        std::numeric_limits<size_t>::max()};
    RefinedEvaluation evaluation;
    size_t activeComponents = 0;
    size_t removedComponentCount = 0;
    size_t acceptedIterations = 0;
};

constexpr size_t kRobustHistogramBins = 256;

[[nodiscard]] size_t robustHistogramBin(double residual)
{
    const double bounded = std::clamp(residual, 0.0, 1.0);
    return std::min(
        kRobustHistogramBins - 1,
        static_cast<size_t>(std::floor(
            bounded * static_cast<double>(kRobustHistogramBins))));
}

[[nodiscard]] double robustHistogramCenter(size_t bin)
{
    return (static_cast<double>(bin) + 0.5) /
        static_cast<double>(kRobustHistogramBins);
}

[[nodiscard]] size_t weightedHistogramQuantileBin(
    const std::array<double, kRobustHistogramBins>& histogram,
    double totalMass,
    double quantile)
{
    if (!(totalMass > 0.0))
        return kRobustHistogramBins - 1;
    const double target = std::clamp(quantile, 0.0, 1.0) * totalMass;
    double cumulative = 0.0;
    for (size_t bin = 0; bin < histogram.size(); ++bin) {
        cumulative += histogram[bin];
        if (cumulative >= target)
            return bin;
    }
    return kRobustHistogramBins - 1;
}

[[nodiscard]] double histogramMassAbove(
    const std::array<double, kRobustHistogramBins>& histogram,
    size_t cutoffBin)
{
    return std::accumulate(
        histogram.begin() + static_cast<std::ptrdiff_t>(cutoffBin + 1),
        histogram.end(), 0.0);
}

struct RobustHistogramCutoff {
    FiberAnchorRobustCutoff summary;
    size_t cutoffBin = kRobustHistogramBins - 1;
};

[[nodiscard]] RobustHistogramCutoff selectRobustHistogramCutoff(
    const std::array<double, kRobustHistogramBins>& residualHistogram,
    const std::array<double, kRobustHistogramBins>& deviationHistogram,
    double totalMass,
    double median,
    double maximumTrimMassFraction,
    double madMultiplier,
    double minimumAngleDegrees)
{
    RobustHistogramCutoff result;
    result.summary.totalMass = totalMass;
    result.summary.retainedMass = totalMass;
    if (!(totalMass > 0.0) || !(maximumTrimMassFraction > 0.0))
        return result;

    const double mad = robustHistogramCenter(weightedHistogramQuantileBin(
        deviationHistogram, totalMass, 0.5));
    const double floorRadians = minimumAngleDegrees * std::acos(-1.0) / 180.0;
    const double floorResidual = std::sin(floorRadians) * std::sin(floorRadians);
    result.cutoffBin = robustHistogramBin(std::clamp(
        std::max(median + madMultiplier * mad, floorResidual), 0.0, 1.0));
    result.summary.candidateTrimmedMass = histogramMassAbove(
        residualHistogram, result.cutoffBin);
    result.summary.detectedOutliers =
        result.summary.candidateTrimmedMass > 0.0;
    if (!result.summary.detectedOutliers)
        return result;

    const double maximumTrimmed = maximumTrimMassFraction * totalMass;
    if (result.summary.candidateTrimmedMass > maximumTrimmed) {
        result.cutoffBin = std::max(
            result.cutoffBin,
            weightedHistogramQuantileBin(
                residualHistogram, totalMass,
                1.0 - maximumTrimMassFraction));
    }
    result.summary.cutoffResidual =
        static_cast<double>(result.cutoffBin + 1) /
        static_cast<double>(kRobustHistogramBins);
    result.summary.trimmedMass = histogramMassAbove(
        residualHistogram, result.cutoffBin);
    result.summary.retainedMass = totalMass - result.summary.trimmedMass;
    return result;
}

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

[[nodiscard]] bool finiteVector(const cv::Vec3f& value)
{
    return std::isfinite(value[0]) && std::isfinite(value[1]) &&
           std::isfinite(value[2]);
}

struct CompactFiberAnchorObservation {
    cv::Vec3f positionPredictionXYZ{0.0F, 0.0F, 0.0F};
    cv::Vec3f direction{0.0F, 0.0F, 0.0F};
    cv::Vec3f presenceGradientPredictionXYZ{0.0F, 0.0F, 0.0F};
    float presence = 0.0F;
    bool valid = false;
    bool presenceGradientValid = false;
};

template <typename Observation>
[[nodiscard]] cv::Vec3d observationPosition(const Observation& observation)
{
    return cv::Vec3d{observation.positionPredictionXYZ};
}

template <typename Observation>
[[nodiscard]] double observationPresence(const Observation& observation)
{
    return static_cast<double>(observation.presence);
}

template <typename Observation>
[[nodiscard]] cv::Vec3d observationGradient(const Observation& observation)
{
    return cv::Vec3d{observation.presenceGradientPredictionXYZ};
}

template <typename Observation>
class IndexedObservationRange {
public:
    IndexedObservationRange(
        const std::vector<Observation>& observations,
        const std::vector<uint32_t>& indices,
        const std::vector<uint8_t>& gradientValidity)
        : observations_{observations},
          indices_{indices},
          gradientValidity_{gradientValidity}
    {
    }

    [[nodiscard]] size_t size() const { return indices_.size(); }

    [[nodiscard]] const Observation& operator[](size_t index) const
    {
        return observations_[indices_[index]];
    }

    [[nodiscard]] bool presenceGradientValid(size_t index) const
    {
        return gradientValidity_[index] != 0;
    }

private:
    const std::vector<Observation>& observations_;
    const std::vector<uint32_t>& indices_;
    const std::vector<uint8_t>& gradientValidity_;
};

template <typename Observation>
[[nodiscard]] bool observationGradientValid(
    const std::vector<Observation>& observations,
    size_t index)
{
    return observations[index].presenceGradientValid;
}

template <typename Observation>
[[nodiscard]] bool observationGradientValid(
    const IndexedObservationRange<Observation>& observations,
    size_t index)
{
    return observations.presenceGradientValid(index);
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

template <typename Observation>
[[nodiscard]] double transverseGaussian(
    const Observation& observation,
    const RefinedComponentState& component,
    const cv::Vec3d& pivot,
    const FiberAnchorConfig& config)
{
    const cv::Vec3d position = observationPosition(observation);
    if (!finiteVector(position))
        return 0.0;
    const double axial = (position - pivot).dot(component.axis);
    if (std::abs(axial) > config.axialSupportHalfWidthPredictionVoxels)
        return 0.0;
    const cv::Vec3d offset = position - component.position;
    const cv::Vec3d transverse = offset - component.axis * offset.dot(component.axis);
    const double distanceSquared = transverse.dot(transverse);
    const double cutoff = config.gaussianCutoffSigmas * config.gaussianSigmaPredictionVoxels;
    if (distanceSquared > cutoff * cutoff)
        return 0.0;
    return std::exp(-distanceSquared /
        (2.0 * config.gaussianSigmaPredictionVoxels *
         config.gaussianSigmaPredictionVoxels));
}

template <typename Observation>
[[nodiscard]] bool usableDirectionObservation(
    const Observation& observation,
    const FiberAnchorConfig& config,
    cv::Vec3d& direction)
{
    const double presence = observationPresence(observation);
    if (!observation.valid || !finiteVector(observation.direction) ||
        !std::isfinite(presence) ||
        presence < config.observationPresenceFloor ||
        presence < 0.0 || presence > 1.0) {
        return false;
    }
    direction = normalized(cv::Vec3d{observation.direction});
    return direction.dot(direction) > kMatrixEpsilon;
}

[[nodiscard]] bool usableDirectionObservation(
    const CompactFiberAnchorObservation& observation,
    const FiberAnchorConfig& config,
    cv::Vec3d& direction)
{
    const double presence = observationPresence(observation);
    if (!observation.valid || !finiteVector(observation.direction) ||
        !std::isfinite(presence) ||
        presence < config.observationPresenceFloor ||
        presence < 0.0 || presence > 1.0) {
        return false;
    }
    direction = cv::Vec3d{observation.direction};
    return direction.dot(direction) > kMatrixEpsilon;
}

struct RobustDirectionProposal {
    std::vector<uint8_t> assignments;
    std::vector<uint8_t> retainedInliers;
    std::array<cv::Vec3d, 2> axes{};
    std::array<bool, 2> unique{false, false};
};

template <typename ObservationRange>
[[nodiscard]] RobustDirectionProposal robustDirectionProposal(
    const ObservationRange& observations,
    const std::array<RefinedComponentState, 2>& components,
    size_t activeComponents,
    const cv::Vec3d& pivot,
    const FiberAnchorConfig& config,
    FiberAnchorFitProfile* profile,
    bool computeAxes)
{
    RobustDirectionProposal proposal;
    proposal.assignments.assign(observations.size(), kUnassignedComponent);
    proposal.retainedInliers.assign(observations.size(), 0);
    std::array<std::array<double, kRobustHistogramBins>, 2> residualHistograms{};
    using SymmetricTensor = std::array<CompensatedSum, 6>;
    using TensorHistogram = std::array<SymmetricTensor, kRobustHistogramBins>;
    std::optional<std::array<TensorHistogram, 2>> tensorHistograms;
    if (computeAxes)
        tensorHistograms.emplace();
    std::array<double, 2> totalMass{0.0, 0.0};

    for (size_t index = 0; index < observations.size(); ++index) {
        const auto& observation = observations[index];
        cv::Vec3d direction;
        if (!usableDirectionObservation(observation, config, direction))
            continue;
        std::array<double, 2> gaussian{0.0, 0.0};
        std::array<double, 2> alignment{0.0, 0.0};
        std::array<double, 2> score{0.0, 0.0};
        for (size_t component = 0; component < activeComponents; ++component) {
            gaussian[component] = transverseGaussian(
                observation, components[component], pivot, config);
            const double dot = direction.dot(components[component].axis);
            alignment[component] = dot * dot;
            score[component] = gaussian[component] *
                observationPresence(observation) *
                alignment[component];
        }
        uint8_t assigned = kUnassignedComponent;
        for (size_t component = 0; component < activeComponents; ++component) {
            if (score[component] > 0.0 &&
                (assigned == kUnassignedComponent ||
                 score[component] > score[assigned])) {
                assigned = static_cast<uint8_t>(component);
            }
        }
        proposal.assignments[index] = assigned;
        if (assigned == kUnassignedComponent)
            continue;
        const double mass = gaussian[assigned] *
            observationPresence(observation);
        const double residual = std::clamp(1.0 - alignment[assigned], 0.0, 1.0);
        const size_t residualBin = robustHistogramBin(residual);
        proposal.retainedInliers[index] = static_cast<uint8_t>(residualBin);
        residualHistograms[assigned][residualBin] += mass;
        totalMass[assigned] += mass;
        if (tensorHistograms.has_value()) {
            auto& tensor = (*tensorHistograms)[assigned][residualBin];
            tensor[0].add(mass * direction[0] * direction[0]);
            tensor[1].add(mass * direction[0] * direction[1]);
            tensor[2].add(mass * direction[0] * direction[2]);
            tensor[3].add(mass * direction[1] * direction[1]);
            tensor[4].add(mass * direction[1] * direction[2]);
            tensor[5].add(mass * direction[2] * direction[2]);
        }
    }
    if (profile != nullptr)
        profile->localTensorObservationVisits += observations.size();

    std::array<size_t, 2> cutoffBins{
        kRobustHistogramBins - 1, kRobustHistogramBins - 1};
    for (size_t component = 0; component < activeComponents; ++component) {
        if (!(totalMass[component] > 0.0) ||
            !(config.robustMaximumTrimMassFraction > 0.0)) {
            if (profile != nullptr)
                profile->robustRetainedMass += totalMass[component];
            continue;
        }
        const size_t medianBin = weightedHistogramQuantileBin(
            residualHistograms[component], totalMass[component], 0.5);
        const double median = robustHistogramCenter(medianBin);
        std::array<double, kRobustHistogramBins> deviationHistogram{};
        for (size_t residualBin = 0; residualBin < kRobustHistogramBins;
             ++residualBin) {
            deviationHistogram[robustHistogramBin(std::abs(
                robustHistogramCenter(residualBin) - median))] +=
                residualHistograms[component][residualBin];
        }
        const RobustHistogramCutoff cutoff = selectRobustHistogramCutoff(
            residualHistograms[component], deviationHistogram,
            totalMass[component], median,
            config.robustMaximumTrimMassFraction,
            config.robustMadMultiplier,
            config.robustMinimumAngleDegrees);
        if (!cutoff.summary.detectedOutliers) {
            if (profile != nullptr) {
                ++profile->robustComponentsWithoutOutliers;
                profile->robustRetainedMass += totalMass[component];
            }
            continue;
        }
        cutoffBins[component] = cutoff.cutoffBin;
        if (profile != nullptr) {
            profile->robustCandidateTrimmedMass +=
                cutoff.summary.candidateTrimmedMass;
            profile->robustTrimmedMass += cutoff.summary.trimmedMass;
            profile->robustRetainedMass += cutoff.summary.retainedMass;
            if (cutoff.summary.trimmedMass > 0.0)
                ++profile->robustTrimmedComponents;
        }
    }

    for (size_t index = 0; index < observations.size(); ++index) {
        const uint8_t assigned = proposal.assignments[index];
        proposal.retainedInliers[index] = static_cast<uint8_t>(
            assigned < activeComponents &&
            proposal.retainedInliers[index] <= cutoffBins[assigned]);
    }
    if (profile != nullptr)
        profile->localTensorObservationVisits += observations.size();
    if (!computeAxes)
        return proposal;
    for (size_t component = 0; component < activeComponents; ++component) {
        SymmetricTensor sums;
        for (size_t residualBin = 0;
             residualBin <= cutoffBins[component]; ++residualBin) {
            for (size_t entry = 0; entry < sums.size(); ++entry) {
                sums[entry].add(
                    (*tensorHistograms)[component][residualBin][entry].sum);
            }
        }
        const cv::Matx33d tensor{
            sums[0].sum, sums[1].sum, sums[2].sum,
            sums[1].sum, sums[3].sum, sums[4].sum,
            sums[2].sum, sums[4].sum, sums[5].sum,
        };
        const FiberPrincipalAxis principal = principalFiberAxis(tensor);
        proposal.unique[component] = principal.unique;
        if (principal.unique)
            proposal.axes[component] = principal.axis;
    }
    return proposal;
}

template <typename ObservationRange>
[[nodiscard]] double retainedSpatialObjective(
    const ObservationRange& observations,
    const std::array<RefinedComponentState, 2>& components,
    size_t activeComponents,
    const std::vector<uint8_t>& assignments,
    const std::vector<uint8_t>& retainedInliers,
    const cv::Vec3d& pivot,
    const FiberAnchorConfig& config)
{
    CompensatedSum numerator;
    CompensatedSum denominator;
    for (size_t index = 0; index < observations.size(); ++index) {
        for (size_t component = 0; component < activeComponents; ++component) {
            denominator.add(transverseGaussian(
                observations[index], components[component], pivot, config));
        }
        const uint8_t component = assignments[index];
        cv::Vec3d direction;
        if (!retainedInliers[index] || component >= activeComponents)
            continue;
        if (!usableDirectionObservation(observations[index], config, direction))
            continue;
        const double gaussian = transverseGaussian(
            observations[index], components[component], pivot, config);
        const double dot = direction.dot(components[component].axis);
        numerator.add(gaussian * observationPresence(observations[index]) *
            dot * dot);
    }
    return denominator.sum > 0.0 ? numerator.sum / denominator.sum : 0.0;
}

template <typename ObservationRange>
[[nodiscard]] std::array<double, 2> retainedSpatialObjectivePair(
    const ObservationRange& observations,
    const std::array<RefinedComponentState, 2>& first,
    const std::array<RefinedComponentState, 2>& second,
    size_t activeComponents,
    const std::vector<uint8_t>& assignments,
    const std::vector<uint8_t>& retainedInliers,
    const cv::Vec3d& pivot,
    const FiberAnchorConfig& config)
{
    std::array<CompensatedSum, 2> numerators;
    std::array<CompensatedSum, 2> denominators;
    for (size_t index = 0; index < observations.size(); ++index) {
        const auto& observation = observations[index];
        for (size_t component = 0; component < activeComponents; ++component) {
            denominators[0].add(transverseGaussian(
                observation, first[component], pivot, config));
            denominators[1].add(transverseGaussian(
                observation, second[component], pivot, config));
        }
        const uint8_t component = assignments[index];
        if (!retainedInliers[index] || component >= activeComponents)
            continue;
        cv::Vec3d direction;
        if (!usableDirectionObservation(observation, config, direction))
            continue;
        const double firstGaussian = transverseGaussian(
            observation, first[component], pivot, config);
        const double secondGaussian = transverseGaussian(
            observation, second[component], pivot, config);
        const double firstDot = direction.dot(first[component].axis);
        const double secondDot = direction.dot(second[component].axis);
        numerators[0].add(
            firstGaussian * observationPresence(observation) *
            firstDot * firstDot);
        numerators[1].add(
            secondGaussian * observationPresence(observation) *
            secondDot * secondDot);
    }
    return {
        denominators[0].sum > 0.0
            ? numerators[0].sum / denominators[0].sum
            : 0.0,
        denominators[1].sum > 0.0
            ? numerators[1].sum / denominators[1].sum
            : 0.0,
    };
}

template <typename ObservationRange>
[[nodiscard]] RefinedEvaluation evaluateFinalRefinedState(
    const ObservationRange& observations,
    const std::array<RefinedComponentState, 2>& components,
    size_t activeComponents,
    const std::vector<uint8_t>& assignments,
    const std::vector<uint8_t>& retainedInliers,
    const cv::Vec3d& pivot,
    const FiberAnchorConfig& config)
{
    RefinedEvaluation evaluation;
    evaluation.assignments = assignments;
    evaluation.retainedInliers = retainedInliers;
    std::array<CompensatedSum, 2> denominators;
    std::array<CompensatedSum, 2> numerators;
    std::array<CompensatedSum, 2> presenceMasses;
    for (size_t index = 0; index < observations.size(); ++index) {
        const auto& observation = observations[index];
        for (size_t component = 0; component < activeComponents; ++component) {
            denominators[component].add(transverseGaussian(
                observation, components[component], pivot, config));
        }
        const uint8_t assigned = assignments[index];
        if (!retainedInliers[index] || assigned >= activeComponents)
            continue;
        cv::Vec3d direction;
        if (!usableDirectionObservation(observation, config, direction))
            continue;
        const double gaussian = transverseGaussian(
            observation, components[assigned], pivot, config);
        const double dot = direction.dot(components[assigned].axis);
        numerators[assigned].add(
            gaussian * observationPresence(observation) * dot * dot);
        presenceMasses[assigned].add(
            gaussian * observationPresence(observation));
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

template <typename ObservationRange>
[[nodiscard]] RefinedFitState refineLocalComponents(
    const ObservationRange& observations,
    const cv::Vec3d& pivot,
    const std::array<cv::Vec3d, 2>& seedAxes,
    const std::array<size_t, 2>& seedComponentIds,
    size_t activeComponents,
    const FiberAnchorConfig& config,
    FiberAnchorFitProfile* profile)
{
    using RefinementClock = std::chrono::steady_clock;
    RefinedFitState state;
    state.activeComponents = activeComponents;
    for (size_t component = 0; component < activeComponents; ++component) {
        state.components[component].axis = canonicalFiberAxis(seedAxes[component]);
        state.components[component].position = pivot;
        state.componentIds[component] = seedComponentIds[component];
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
    for (size_t index = 0; index < observations.size(); ++index) {
        const cv::Vec3d position = observationPosition(observations[index]);
        if (!finiteVector(position))
            continue;
        for (int coordinate = 0; coordinate < 3; ++coordinate) {
            lower[coordinate] = std::min(
                lower[coordinate], position[coordinate]);
            upper[coordinate] = std::max(
                upper[coordinate], position[coordinate]);
        }
    }
    for (int iteration = 0; iteration < config.maximumIterations; ++iteration) {
        if (profile != nullptr) {
            ++profile->localRefinementAttempts;
        }
        const auto tensorStart = profile != nullptr
            ? RefinementClock::now()
            : RefinementClock::time_point{};
        RobustDirectionProposal robust = robustDirectionProposal(
            observations, state.components, state.activeComponents, pivot,
            config, profile, true);
        if (profile != nullptr) {
            profile->localTensorProposalWorkSeconds +=
                std::chrono::duration<double>(
                    RefinementClock::now() - tensorStart).count();
        }

        bool removedComponent = false;
        std::array<RefinedComponentState, 2> compactComponents{};
        std::array<size_t, 2> compactIds{kNoDiagnosticId, kNoDiagnosticId};
        size_t compactCount = 0;
        for (size_t component = 0; component < state.activeComponents; ++component) {
            if (!robust.unique[component]) {
                removedComponent = true;
                state.removedComponentIds[state.removedComponentCount++] =
                    state.componentIds[component];
                if (profile != nullptr)
                    ++profile->robustRemovedNonuniqueComponents;
                continue;
            }
            compactComponents[compactCount] = state.components[component];
            compactComponents[compactCount].axis = robust.axes[component];
            compactIds[compactCount] = state.componentIds[component];
            ++compactCount;
        }
        if (compactCount == 0) {
            state.activeComponents = 0;
            state.evaluation.assignments.assign(
                observations.size(), kUnassignedComponent);
            state.evaluation.retainedInliers.assign(observations.size(), 0);
            if (profile != nullptr) {
                profile->localRefinementAcceptedSteps +=
                    state.acceptedIterations;
            }
            return state;
        }
        if (removedComponent) {
            state.components = compactComponents;
            state.componentIds = compactIds;
            state.activeComponents = compactCount;
            --iteration;
            continue;
        }

        auto proposed = compactComponents;
        const auto centroidStart = profile != nullptr
            ? RefinementClock::now()
            : RefinementClock::time_point{};
        for (size_t component = 0; component < state.activeComponents; ++component) {
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
                if (robust.assignments[index] != component ||
                    !robust.retainedInliers[index]) {
                    continue;
                }
                const auto& observation = observations[index];
                const double gaussian = transverseGaussian(
                    observation, centered, pivot, config);
                cv::Vec3d direction;
                if (!usableDirectionObservation(
                        observation, config, direction)) {
                    continue;
                }
                const double dot = direction.dot(centered.axis);
                const double weight = gaussian *
                    observationPresence(observation) * dot * dot;
                centroidMass.add(weight);
                const cv::Vec3d position = observationPosition(observation);
                for (int axis = 0; axis < 3; ++axis) {
                    centroid[axis].add(
                        weight * position[axis]);
                }
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
        if (profile != nullptr) {
            profile->localCentroidObservationVisits +=
                state.activeComponents * observations.size();
            profile->localCentroidProposalWorkSeconds +=
                std::chrono::duration<double>(
                    RefinementClock::now() - centroidStart).count();
        }

        std::array<RefinedComponentState, 2> baseline = state.components;
        double acceptedDirectionUpdate = 0.0;
        double maximumTargetDisplacement = 0.0;
        for (size_t component = 0; component < state.activeComponents; ++component) {
            baseline[component].axis = proposed[component].axis;
            baseline[component].position = clampToWindow(
                projectToConstraintPlane(
                    state.components[component].position, pivot,
                    baseline[component].axis),
                pivot, baseline[component].axis,
                config.localWindowRadiusPredictionVoxels, lower, upper);
            acceptedDirectionUpdate = std::max(
                acceptedDirectionUpdate,
                projectiveUpdate(
                    state.components[component].axis,
                    baseline[component].axis));
            const cv::Vec3d displacement =
                proposed[component].position - baseline[component].position;
            maximumTargetDisplacement = std::max(
                maximumTargetDisplacement,
                std::sqrt(std::max(0.0, displacement.dot(displacement))));
        }
        const auto evaluationStart = profile != nullptr
            ? RefinementClock::now()
            : RefinementClock::time_point{};
        if (profile != nullptr)
            profile->refinedEvaluationObservationVisits += observations.size();
        auto acceptedComponents = baseline;
        double acceptedPositionUpdate = 0.0;
        const auto fractions = fiberAnchorSpatialBacktrackingFractions(
            maximumTargetDisplacement,
            config.peakGridStepPredictionVoxels);
        double baselineObjective = 0.0;
        for (size_t depth = 0; depth < fractions.size(); ++depth) {
            if (profile != nullptr) {
                ++profile->backtrackingEvaluations;
                ++profile->spatialCandidatesTested;
                ++profile->spatialCandidatesTestedByDepth[depth];
                profile->refinedEvaluationObservationVisits +=
                    observations.size();
            }
            auto candidate = baseline;
            for (size_t component = 0; component < state.activeComponents; ++component) {
                candidate[component].position = clampToWindow(
                    baseline[component].position +
                        (proposed[component].position - baseline[component].position) *
                            fractions[depth],
                    pivot, candidate[component].axis,
                    config.localWindowRadiusPredictionVoxels, lower, upper);
            }
            double objective = 0.0;
            if (depth == 0) {
                const auto objectives = retainedSpatialObjectivePair(
                    observations, baseline, candidate, state.activeComponents,
                    robust.assignments, robust.retainedInliers, pivot, config);
                baselineObjective = objectives[0];
                objective = objectives[1];
            } else {
                objective = retainedSpatialObjective(
                    observations, candidate, state.activeComponents,
                    robust.assignments, robust.retainedInliers, pivot, config);
            }
            const double tolerance = config.convergenceTolerance *
                std::max(1.0, std::abs(baselineObjective));
            if (objective <= baselineObjective + tolerance)
                continue;
            acceptedComponents = candidate;
            if (profile != nullptr)
                ++profile->spatialCandidatesAcceptedByDepth[depth];
            for (size_t component = 0; component < state.activeComponents; ++component) {
                const cv::Vec3d delta =
                    acceptedComponents[component].position -
                    baseline[component].position;
                acceptedPositionUpdate = std::max(
                    acceptedPositionUpdate,
                    std::sqrt(std::max(0.0, delta.dot(delta))));
            }
            break;
        }
        if (profile != nullptr) {
            profile->localStateEvaluationWorkSeconds +=
                std::chrono::duration<double>(
                    RefinementClock::now() - evaluationStart).count();
        }
        state.components = acceptedComponents;
        state.evaluation.assignments = robust.assignments;
        state.evaluation.retainedInliers = robust.retainedInliers;
        ++state.acceptedIterations;
        const double positionTolerance = std::max(
            config.positionConvergenceTolerancePredictionVoxels,
            config.peakGridStepPredictionVoxels);
        if (acceptedDirectionUpdate <= config.convergenceTolerance &&
            acceptedPositionUpdate <= positionTolerance) {
            break;
        }
        if (iteration + 1 == config.maximumIterations && profile != nullptr)
            ++profile->robustHardLimitHits;
    }
    if (state.activeComponents > 0) {
        const auto refreshStart = profile != nullptr
            ? RefinementClock::now()
            : RefinementClock::time_point{};
        const RobustDirectionProposal finalMembership = robustDirectionProposal(
            observations, state.components, state.activeComponents, pivot,
            config, profile, false);
        state.evaluation.assignments = finalMembership.assignments;
        state.evaluation.retainedInliers = finalMembership.retainedInliers;
        if (profile != nullptr) {
            profile->localTensorProposalWorkSeconds +=
                std::chrono::duration<double>(
                    RefinementClock::now() - refreshStart).count();
        }
    }
    if (profile != nullptr)
        profile->localRefinementAcceptedSteps += state.acceptedIterations;
    return state;
}

template <typename ObservationRange>
[[nodiscard]] PeakOwnerBounds peakOwnerBounds(
    const ObservationRange& observations,
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
    for (size_t index = 0; index < observations.size(); ++index) {
        const cv::Vec3d position = observationPosition(observations[index]);
        if (!finiteVector(position))
            continue;
        for (int coordinate = 0; coordinate < 3; ++coordinate) {
            observedLower[coordinate] = std::min(
                observedLower[coordinate],
                position[coordinate]);
            observedUpper[coordinate] = std::max(
                observedUpper[coordinate],
                position[coordinate]);
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

template <typename ObservationRange>
[[nodiscard]] DirectionConditionedPeak findDirectionConditionedLocalPeak(
    const ObservationRange& observations,
    const cv::Vec3d& pivot,
    const PeakOwnerBounds& owner,
    const std::array<RefinedComponentState, 2>& components,
    size_t selectedComponent,
    const std::vector<uint8_t>& assignments,
    const std::vector<uint8_t>& retainedInliers,
    const FiberAnchorConfig& config,
    FiberAnchorFitProfile* profile)
{
    const cv::Vec3d axis = components[selectedComponent].axis;
    const auto basis = transverseBasis(axis);
    const float cutoff = static_cast<float>(
        config.gaussianCutoffSigmas * config.peakSigmaPredictionVoxels);
    const double axialCutoff =
        config.gaussianCutoffSigmas *
        config.peakAxialSigmaPredictionVoxels;
    const double unionRadius = config.localWindowRadiusPredictionVoxels + cutoff;
    const float invTwoTransverseSigma2 = static_cast<float>(1.0 /
        (2.0 * config.peakSigmaPredictionVoxels *
         config.peakSigmaPredictionVoxels));
    const double invTwoAxialSigma2 = 1.0 /
        (2.0 * config.peakAxialSigmaPredictionVoxels *
         config.peakAxialSigmaPredictionVoxels);

    struct PeakObservation {
        float first = 0.0F;
        float second = 0.0F;
        float axialGaussian = 0.0F;
        float signal = 0.0F;
        float directionAlignmentSquared = 0.0F;
        float gradientFirst = 0.0F;
        float gradientSecond = 0.0F;
        float gradientNorm = 0.0F;
        bool presenceGradientValid = false;
    };
    std::vector<PeakObservation> peakObservations;
    peakObservations.reserve(observations.size());
    if (profile != nullptr) {
        ++profile->peakComponents;
        profile->peakPreparationObservationVisits += observations.size();
    }
    for (size_t observationIndex = 0;
         observationIndex < observations.size(); ++observationIndex) {
        const auto& observation = observations[observationIndex];
        const cv::Vec3d position = observationPosition(observation);
        if (!finiteVector(position))
            continue;
        const cv::Vec3d pivotOffset = position - pivot;
        const double axial = pivotOffset.dot(axis);
        if (std::abs(axial) > axialCutoff)
            continue;
        const cv::Vec3d transverse = pivotOffset - axis * axial;
        if (transverse.dot(transverse) > unionRadius * unionRadius)
            continue;
        const double first = transverse.dot(basis[0]);
        const double second = transverse.dot(basis[1]);

        cv::Vec3d direction;
        const bool usablePositive = usableDirectionObservation(
            observation, config, direction);
        const bool retainedForComponent =
            retainedInliers[observationIndex] &&
            assignments[observationIndex] == selectedComponent;
        double selectedAlignment = 0.0;
        double signal = 0.0;
        if (retainedForComponent && usablePositive) {
            const double dot = direction.dot(axis);
            selectedAlignment = dot * dot;
            signal = observationPresence(observation) * selectedAlignment;
        }
        PeakObservation peakObservation;
        peakObservation.first = static_cast<float>(first);
        peakObservation.second = static_cast<float>(second);
        peakObservation.axialGaussian = static_cast<float>(std::exp(
            -axial * axial * invTwoAxialSigma2));
        peakObservation.signal = static_cast<float>(signal);
        peakObservation.directionAlignmentSquared =
            static_cast<float>(selectedAlignment);
        if (observationGradientValid(observations, observationIndex) &&
            finiteVector(observation.presenceGradientPredictionXYZ)) {
            const cv::Vec3d gradient = observationGradient(observation);
            peakObservation.gradientFirst =
                static_cast<float>(gradient.dot(basis[0]));
            peakObservation.gradientSecond =
                static_cast<float>(gradient.dot(basis[1]));
            const double gradientNorm2 =
                peakObservation.gradientFirst * peakObservation.gradientFirst +
                peakObservation.gradientSecond * peakObservation.gradientSecond;
            if (gradientNorm2 > kMatrixEpsilon) {
                peakObservation.gradientNorm =
                    static_cast<float>(std::sqrt(gradientNorm2));
                peakObservation.presenceGradientValid = true;
            }
        }
        peakObservations.push_back(peakObservation);
    }

    const auto responseAt = [&](float candidateFirst, float candidateSecond, bool acceptance) {
        if (profile != nullptr) {
            if (acceptance)
                ++profile->peakAcceptanceResponses;
            else
                ++profile->peakComputedGridResponses;
            profile->peakResponseObservationVisits += peakObservations.size();
        }
        CompensatedSum numerator;
        CompensatedSum denominator;
        CompensatedSum eligibleGradientWeight;
        CompensatedSum validGradientWeight;
        CompensatedSum inward;
        CompensatedSum outward;
        for (const auto& observation : peakObservations) {
            const float radialFirst = candidateFirst - observation.first;
            const float radialSecond = candidateSecond - observation.second;
            const float distanceSquared =
                radialFirst * radialFirst + radialSecond * radialSecond;
            if (distanceSquared > cutoff * cutoff)
                continue;
            const float gaussian = observation.axialGaussian *
                std::exp(-distanceSquared * invTwoTransverseSigma2);
            denominator.add(gaussian);
            numerator.add(gaussian * observation.signal);
            if (!(observation.directionAlignmentSquared > 0.0))
                continue;
            const float eligibleWeight =
                gaussian * observation.directionAlignmentSquared;
            eligibleGradientWeight.add(eligibleWeight);
            if (!observation.presenceGradientValid) {
                continue;
            }
            if (!(distanceSquared > kMatrixEpsilon)) {
                continue;
            }
            validGradientWeight.add(eligibleWeight);
            const float cosine = std::clamp(
                (observation.gradientFirst * radialFirst +
                 observation.gradientSecond * radialSecond) /
                    (observation.gradientNorm * std::sqrt(distanceSquared)),
                -1.0F, 1.0F);
            const float vote = eligibleWeight * observation.gradientNorm *
                static_cast<float>(config.peakSigmaPredictionVoxels) *
                cosine * cosine;
            if (cosine > 0.0)
                inward.add(vote);
            else if (cosine < 0.0)
                outward.add(vote);
        }
        const double presenceResponse = denominator.sum > 0.0 ? numerator.sum / denominator.sum : 0.0;
        if (!(config.peakGradientWeight > 0.0) || !(eligibleGradientWeight.sum > 0.0) || !(validGradientWeight.sum > 0.0)) {
            return presenceResponse;
        }
        const double voteMass = inward.sum + outward.sum;
        if (!(voteMass > 0.0))
            return presenceResponse;
        const double coverage = std::clamp(validGradientWeight.sum / eligibleGradientWeight.sum, 0.0, 1.0);
        const double radialGradient = voteMass / validGradientWeight.sum;
        const double reliability = coverage * radialGradient / (radialGradient + config.peakGradientReliabilityScale);
        const double signedVote = (inward.sum - outward.sum) / voteMass;
        return presenceResponse + config.peakGradientWeight * reliability * signedVote;
    };

    using GridIndex = std::pair<int, int>;
    const int extent = static_cast<int>(std::floor(config.localWindowRadiusPredictionVoxels / config.peakGridStepPredictionVoxels));
    const auto pointAt = [&](const GridIndex& index) {
        return pivot + basis[0] * (static_cast<double>(index.first) * config.peakGridStepPredictionVoxels) +
               basis[1] * (static_cast<double>(index.second) * config.peakGridStepPredictionVoxels);
    };
    const auto responseAtIndex = [&](const GridIndex& index, bool acceptance) {
        return responseAt(
            static_cast<float>(index.first) *
                static_cast<float>(config.peakGridStepPredictionVoxels),
            static_cast<float>(index.second) *
                static_cast<float>(config.peakGridStepPredictionVoxels),
            acceptance);
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
        if (profile != nullptr)
            ++profile->peakGridResponseRequests;
        const auto found = responseCache.find(index);
        if (found != responseCache.end())
            return found->second;
        return responseCache.emplace(index, responseAtIndex(index, false))
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
        const cv::Vec3d offset = candidate - pivot;
        return insidePeakDomain(
                   candidate, pivot, owner,
                   config.localWindowRadiusPredictionVoxels) &&
            responseAt(
                static_cast<float>(offset.dot(basis[0])),
                static_cast<float>(offset.dot(basis[1])), true) +
                tolerance >= centerResponse;
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
    const FiberAnchorConfig& config,
    FiberAnchorFitProfile* profile)
{
    if (profile != nullptr)
        ++profile->seedPairs;
    FitState best;
    best.objectiveNumerator = -1.0;
    std::vector<uint8_t> previousAssignments;
    std::vector<uint8_t> twoBackAssignments;
    for (int iteration = 0; iteration < config.maximumIterations; ++iteration) {
        if (profile != nullptr) {
            ++profile->seedPairIterations;
            profile->seedAssignmentObservationVisits += observations.size();
            profile->seedTensorObservationVisits += 2 * observations.size();
            profile->seedObjectiveObservationVisits += observations.size();
        }
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
    if (profile != nullptr) {
        profile->seedAssignmentObservationVisits += observations.size();
        profile->seedObjectiveObservationVisits += observations.size();
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

void accumulateFitProfile(
    FiberAnchorFitProfile& total,
    const FiberAnchorFitProfile& value)
{
    total.invocations += value.invocations;
    total.nonemptyCells += value.nonemptyCells;
    total.weightedObservations += value.weightedObservations;
    total.seeds += value.seeds;
    total.seedGenerationObservationVisits +=
        value.seedGenerationObservationVisits;
    total.seedPairs += value.seedPairs;
    total.seedPairIterations += value.seedPairIterations;
    total.seedAssignmentObservationVisits +=
        value.seedAssignmentObservationVisits;
    total.seedTensorObservationVisits += value.seedTensorObservationVisits;
    total.seedObjectiveObservationVisits +=
        value.seedObjectiveObservationVisits;
    total.initializationObservationVisits +=
        value.initializationObservationVisits;
    total.localRefinementAttempts += value.localRefinementAttempts;
    total.localRefinementAcceptedSteps += value.localRefinementAcceptedSteps;
    total.backtrackingEvaluations += value.backtrackingEvaluations;
    total.robustComponentsWithoutOutliers +=
        value.robustComponentsWithoutOutliers;
    total.robustTrimmedComponents += value.robustTrimmedComponents;
    total.robustRemovedNonuniqueComponents +=
        value.robustRemovedNonuniqueComponents;
    total.robustHardLimitHits += value.robustHardLimitHits;
    total.spatialCandidatesTested += value.spatialCandidatesTested;
    for (size_t depth = 0;
         depth < total.spatialCandidatesTestedByDepth.size(); ++depth) {
        total.spatialCandidatesTestedByDepth[depth] +=
            value.spatialCandidatesTestedByDepth[depth];
        total.spatialCandidatesAcceptedByDepth[depth] +=
            value.spatialCandidatesAcceptedByDepth[depth];
    }
    total.robustCandidateTrimmedMass += value.robustCandidateTrimmedMass;
    total.robustTrimmedMass += value.robustTrimmedMass;
    total.robustRetainedMass += value.robustRetainedMass;
    total.localTensorObservationVisits += value.localTensorObservationVisits;
    total.localCentroidObservationVisits +=
        value.localCentroidObservationVisits;
    total.refinedEvaluationObservationVisits +=
        value.refinedEvaluationObservationVisits;
    total.peakComponents += value.peakComponents;
    total.peakPreparationObservationVisits +=
        value.peakPreparationObservationVisits;
    total.peakGridResponseRequests += value.peakGridResponseRequests;
    total.peakComputedGridResponses += value.peakComputedGridResponses;
    total.peakAcceptanceResponses += value.peakAcceptanceResponses;
    total.peakResponseObservationVisits +=
        value.peakResponseObservationVisits;
    total.finalEvaluationObservationVisits +=
        value.finalEvaluationObservationVisits;
    total.setupWorkSeconds += value.setupWorkSeconds;
    total.seedGenerationWorkSeconds += value.seedGenerationWorkSeconds;
    total.seedPairRefinementWorkSeconds +=
        value.seedPairRefinementWorkSeconds;
    total.initializationWorkSeconds += value.initializationWorkSeconds;
    total.localRefinementWorkSeconds += value.localRefinementWorkSeconds;
    total.localTensorProposalWorkSeconds +=
        value.localTensorProposalWorkSeconds;
    total.localCentroidProposalWorkSeconds +=
        value.localCentroidProposalWorkSeconds;
    total.localStateEvaluationWorkSeconds +=
        value.localStateEvaluationWorkSeconds;
    total.peakSearchWorkSeconds += value.peakSearchWorkSeconds;
    total.finalEvaluationWorkSeconds += value.finalEvaluationWorkSeconds;
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
                        if (other.cell == candidate.cell)
                            continue;
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

FiberAnchorRobustCutoff selectFiberAnchorRobustCutoff(
    const std::vector<FiberAnchorResidualSample>& samples,
    double maximumTrimMassFraction,
    double madMultiplier,
    double minimumAngleDegrees)
{
    if (!(maximumTrimMassFraction >= 0.0) ||
        maximumTrimMassFraction > 0.20 ||
        !std::isfinite(maximumTrimMassFraction) ||
        !(madMultiplier >= 0.0) || !std::isfinite(madMultiplier) ||
        !(minimumAngleDegrees >= 0.0) || minimumAngleDegrees > 90.0 ||
        !std::isfinite(minimumAngleDegrees)) {
        throw std::invalid_argument("invalid fiber anchor robust cutoff parameters");
    }
    FiberAnchorRobustCutoff result;
    std::array<double, kRobustHistogramBins> histogram{};
    for (const auto& sample : samples) {
        if (!std::isfinite(sample.residual) || !std::isfinite(sample.mass) ||
            !(sample.mass > 0.0)) {
            continue;
        }
        histogram[robustHistogramBin(sample.residual)] += sample.mass;
        result.totalMass += sample.mass;
    }
    result.retainedMass = result.totalMass;
    if (!(result.totalMass > 0.0) || !(maximumTrimMassFraction > 0.0))
        return result;

    const size_t medianBin = weightedHistogramQuantileBin(
        histogram, result.totalMass, 0.5);
    const double median = robustHistogramCenter(medianBin);
    std::array<double, kRobustHistogramBins> deviationHistogram{};
    for (const auto& sample : samples) {
        if (!std::isfinite(sample.residual) || !std::isfinite(sample.mass) ||
            !(sample.mass > 0.0)) {
            continue;
        }
        deviationHistogram[robustHistogramBin(std::abs(
            std::clamp(sample.residual, 0.0, 1.0) - median))] += sample.mass;
    }
    return selectRobustHistogramCutoff(
        histogram, deviationHistogram, result.totalMass, median,
        maximumTrimMassFraction, madMultiplier, minimumAngleDegrees).summary;
}

std::vector<double> fiberAnchorSpatialBacktrackingFractions(
    double maximumDisplacementPredictionVoxels,
    double targetStepPredictionVoxels,
    int maximumHalvings)
{
    if (!(maximumDisplacementPredictionVoxels >= 0.0) ||
        !std::isfinite(maximumDisplacementPredictionVoxels) ||
        !(targetStepPredictionVoxels > 0.0) ||
        !std::isfinite(targetStepPredictionVoxels) ||
        maximumHalvings < 0 || maximumHalvings > 8) {
        throw std::invalid_argument("invalid fiber anchor spatial backtracking parameters");
    }
    std::vector<double> fractions;
    fractions.reserve(static_cast<size_t>(maximumHalvings + 1));
    for (int depth = 0; depth <= maximumHalvings; ++depth) {
        const double fraction = std::ldexp(1.0, -depth);
        fractions.push_back(fraction);
        if (maximumDisplacementPredictionVoxels * fraction <=
            targetStepPredictionVoxels) {
            break;
        }
    }
    return fractions;
}

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
    if (!(config.robustMaximumTrimMassFraction >= 0.0) ||
        config.robustMaximumTrimMassFraction > 0.20 ||
        !std::isfinite(config.robustMaximumTrimMassFraction)) {
        throw std::invalid_argument(
            "fiber anchor robust maximum trim mass must be in [0, 0.20]");
    }
    if (!(config.robustMadMultiplier >= 0.0) ||
        !std::isfinite(config.robustMadMultiplier)) {
        throw std::invalid_argument(
            "fiber anchor robust MAD multiplier must be nonnegative and finite");
    }
    if (!(config.robustMinimumAngleDegrees >= 0.0) ||
        config.robustMinimumAngleDegrees > 90.0 ||
        !std::isfinite(config.robustMinimumAngleDegrees)) {
        throw std::invalid_argument(
            "fiber anchor robust minimum angle must be in [0, 90]");
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

template <typename ObservationRange>
FiberCellAnchorResult fitFiberCellAnchorsImpl(
    const std::array<size_t, 3>& cellZYX,
    const std::array<size_t, 3>& cellBeginZYX,
    const std::array<size_t, 3>& cellEndZYX,
    const ObservationRange& input,
    const FiberAnchorConfig& config,
    FiberAnchorFitProfile* profile)
{
    using FitClock = std::chrono::steady_clock;
    auto phaseStart = profile != nullptr
        ? FitClock::now()
        : FitClock::time_point{};
    if (profile != nullptr)
        ++profile->invocations;
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
    size_t ownedCount = 0;
    for (size_t index = 0; index < input.size(); ++index) {
        const cv::Vec3d position = observationPosition(input[index]);
        ownedCount += static_cast<size_t>(
            finiteVector(position) && isOwned(position));
    }
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
        const cv::Vec3d position = observationPosition(candidate);
        if (!finiteVector(position) || !isOwned(position)) {
            continue;
        }
        const cv::Vec3d delta = position - center;
        const double gaussian = std::exp(-delta.dot(delta) * invTwoSigma2);
        denominator.add(gaussian);
        cv::Vec3d direction;
        if (!usableDirectionObservation(candidate, config, direction)) {
            continue;
        }
        observations.push_back({
            position,
            direction,
            gaussian,
            gaussian * observationPresence(candidate),
            index,
        });
    }
    if (profile != nullptr) {
        profile->setupWorkSeconds += std::chrono::duration<double>(
            FitClock::now() - phaseStart).count();
        profile->weightedObservations += observations.size();
    }
    if (observations.empty()) {
        for (auto& component : result.components)
            component.rejectionReason = "empty";
        return result;
    }
    if (profile != nullptr) {
        ++profile->nonemptyCells;
        phaseStart = FitClock::now();
        profile->seedGenerationObservationVisits += observations.size();
    }

    const FiberPrincipalAxis global = principalFiberAxis(weightedTensor(observations, nullptr, 0));
    std::vector<cv::Vec3d> seeds;
    seeds.reserve(config.maximumSeedCount);
    if (global.unique) {
        seeds.push_back(global.axis);
    } else {
        if (profile != nullptr)
            profile->seedGenerationObservationVisits += observations.size();
        size_t best = 0;
        for (size_t index = 1; index < observations.size(); ++index) {
            if (observations[index].weight > observations[best].weight)
                best = index;
        }
        seeds.push_back(canonicalFiberAxis(observations[best].direction));
    }
    std::vector<bool> selected(observations.size(), false);
    while (seeds.size() < config.maximumSeedCount) {
        if (profile != nullptr)
            profile->seedGenerationObservationVisits += observations.size();
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
    if (profile != nullptr) {
        profile->seeds += seeds.size();
        profile->seedGenerationWorkSeconds += std::chrono::duration<double>(
            FitClock::now() - phaseStart).count();
        phaseStart = FitClock::now();
    }

    FitState bestFit;
    bestFit.objectiveNumerator = -1.0;
    if (seeds.size() == 1) {
        bestFit = refineSeedPair(
            observations, {seeds[0], seeds[0]}, config, profile);
    } else {
        for (size_t first = 0; first + 1 < seeds.size(); ++first) {
            for (size_t second = first + 1; second < seeds.size(); ++second) {
                const FitState fit = refineSeedPair(
                    observations, {seeds[first], seeds[second]}, config,
                    profile);
                if (bestFit.objectiveNumerator < 0.0 || betterState(fit, bestFit))
                    bestFit = fit;
            }
        }
    }
    if (profile != nullptr) {
        profile->seedPairRefinementWorkSeconds +=
            std::chrono::duration<double>(FitClock::now() - phaseStart).count();
        phaseStart = FitClock::now();
        profile->initializationObservationVisits +=
            2 * observations.size();
    }
    const double seededObjective = denominator.sum > 0.0
        ? bestFit.objectiveNumerator / denominator.sum
        : 0.0;

    const std::array<FiberPrincipalAxis, 2> fittedComponents{
        principalFiberAxis(weightedTensor(observations, &bestFit.assignments, 0)),
        principalFiberAxis(weightedTensor(observations, &bestFit.assignments, 1)),
    };
    std::array<size_t, 2> fittedAssignedCounts{0, 0};
    if (profile != nullptr)
        profile->initializationObservationVisits += bestFit.assignments.size();
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
                [profile, componentIndex](uint8_t assigned) {
                    if (profile != nullptr)
                        ++profile->initializationObservationVisits;
                    return assigned == componentIndex;
                }) ? "degenerate" : "empty";
        }
    }
    result.objective = seededObjective;

    std::array<cv::Vec3d, 2> seedAxes{};
    std::array<size_t, 2> diagnosticIds{kNoDiagnosticId, kNoDiagnosticId};
    size_t activeComponents = 0;
    for (uint8_t componentIndex = 0; componentIndex < 2; ++componentIndex) {
        const auto& fitted = fittedComponents[componentIndex];
        if (fitted.unique) {
            seedAxes[activeComponents] = fitted.axis;
            diagnosticIds[activeComponents++] = componentIndex;
        } else {
            result.components[componentIndex].rejectionReason = std::any_of(
                bestFit.assignments.begin(), bestFit.assignments.end(),
                [profile, componentIndex](uint8_t assigned) {
                    if (profile != nullptr)
                        ++profile->initializationObservationVisits;
                    return assigned == componentIndex;
                }) ? "degenerate" : "empty";
        }
    }
    if (activeComponents == 0) {
        if (profile != nullptr) {
            profile->initializationWorkSeconds +=
                std::chrono::duration<double>(FitClock::now() - phaseStart)
                    .count();
        }
        return result;
    }

    if (profile != nullptr) {
        profile->initializationWorkSeconds += std::chrono::duration<double>(
            FitClock::now() - phaseStart).count();
        phaseStart = FitClock::now();
    }

    RefinedFitState refined = refineLocalComponents(
        input, center, seedAxes, diagnosticIds, activeComponents, config,
        profile);
    for (size_t index = 0; index < refined.removedComponentCount; ++index) {
        const size_t diagnosticId = refined.removedComponentIds[index];
        if (diagnosticId >= result.initializedDiagnostics.size())
            continue;
        auto& diagnostic = result.initializedDiagnostics[diagnosticId];
        diagnostic.transition.outcome = "rejected";
        diagnostic.transition.reason = "degenerate";
    }
    if (profile != nullptr) {
        profile->localRefinementWorkSeconds += std::chrono::duration<double>(
            FitClock::now() - phaseStart).count();
        phaseStart = FitClock::now();
    }
    const PeakOwnerBounds owner = peakOwnerBounds(
        input, cellBeginZYX, cellEndZYX);
    const auto broadRefinedComponents = refined.components;
    for (size_t componentIndex = 0; componentIndex < refined.activeComponents;
         ++componentIndex) {
        const auto peak = findDirectionConditionedLocalPeak(
            input, center, owner, broadRefinedComponents,
            componentIndex, refined.evaluation.assignments,
            refined.evaluation.retainedInliers, config, profile);
        result.components[componentIndex]
            .discretePeakPositionPredictionXYZ = peak.discrete;
        result.components[componentIndex]
            .separablePeakPositionPredictionXYZ = peak.separable1d;
        result.components[componentIndex]
            .jointPeakPositionPredictionXYZ = peak.joint2d;
        refined.components[componentIndex].position = peak.separable1d;
    }
    if (profile != nullptr) {
        profile->peakSearchWorkSeconds += std::chrono::duration<double>(
            FitClock::now() - phaseStart).count();
        phaseStart = FitClock::now();
        profile->finalEvaluationObservationVisits += input.size();
    }
    refined.evaluation = evaluateFinalRefinedState(
        input, refined.components, refined.activeComponents,
        refined.evaluation.assignments,
        refined.evaluation.retainedInliers, center, config);
    result.objective = refined.evaluation.objective;
    for (size_t componentIndex = 0;
         componentIndex < refined.activeComponents; ++componentIndex) {
        auto& component = result.components[componentIndex];
        component.diagnosticId = refined.componentIds[componentIndex];
        component.diagnosticParentIds = {
            refined.componentIds[componentIndex]};
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
    size_t outputIndex = refined.activeComponents;
    for (size_t index = 0;
         index < refined.removedComponentCount &&
         outputIndex < result.components.size();
         ++index, ++outputIndex) {
        auto& component = result.components[outputIndex];
        component.diagnosticId = refined.removedComponentIds[index];
        component.diagnosticParentIds = {refined.removedComponentIds[index]};
        component.rejectionReason = "degenerate";
        component.removedDuringRobustRefinement = true;
    }
    while (outputIndex < result.components.size()) {
        result.components[outputIndex].rejectionReason = "empty";
        ++outputIndex;
    }
    if (componentLess(result.components[1], result.components[0]))
        std::swap(result.components[0], result.components[1]);
    if (profile != nullptr) {
        profile->finalEvaluationWorkSeconds += std::chrono::duration<double>(
            FitClock::now() - phaseStart).count();
    }
    return result;
}

FiberCellAnchorResult fitFiberCellAnchors(
    const std::array<size_t, 3>& cellZYX,
    const std::array<size_t, 3>& cellBeginZYX,
    const std::array<size_t, 3>& cellEndZYX,
    const std::vector<FiberAnchorObservation>& input,
    const FiberAnchorConfig& config,
    FiberAnchorFitProfile* profile)
{
    return fitFiberCellAnchorsImpl(
        cellZYX, cellBeginZYX, cellEndZYX, input, config, profile);
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
    const auto checkedMultiply = [](size_t left, size_t right,
                                     const char* description) {
        if (right != 0 &&
            left > std::numeric_limits<size_t>::max() / right) {
            throw std::overflow_error(
                std::string(description) + " size overflows");
        }
        return left * right;
    };
    const auto checkedAdd = [](size_t left, size_t right,
                                const char* description) {
        if (left > std::numeric_limits<size_t>::max() - right) {
            throw std::overflow_error(
                std::string(description) + " size overflows");
        }
        return left + right;
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
    const auto supportStencil = detail::buildFiberAnchorSupportStencil(
        cellSize, sampleHalo, maximumSupportRadius);
    const size_t supportStencilSize =
        detail::fiberAnchorSupportStencilSize(supportStencil);
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
    const double startCpuSeconds = processCpuSeconds();
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
    report.profile.selectedCells = selectedCells.size();
    report.profile.workCells = workCells.size();
    report.profile.contextCells = workCells.size() - selectedCells.size();
    report.profile.setupSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();

    const auto processCells = [&](const std::vector<std::array<size_t, 3>>& requestedCells,
                                  bool tallySelectedDiagnostics,
                                  const char* phase) {
        const auto phaseStart = std::chrono::steady_clock::now();
        if (progressCallback)
            progressCallback({phase, 0, requestedCells.size(), 0.0});
        if (requestedCells.empty())
            return std::vector<FiberCellAnchorResult>{};
        const auto tilePlanningStart = std::chrono::steady_clock::now();

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

        struct CachedPresenceGradient {
            cv::Vec3f value{0.0F, 0.0F, 0.0F};
            bool valid = false;
        };
        struct Tile {
            std::vector<size_t> cells;
            CellIndex sampleBegin{};
            CellIndex sampleEnd{};
            size_t estimatedBytes = 0;
        };
        constexpr size_t kTileCellsPerAxis = 6;
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
            const size_t denseBytes =
                sizeof(CellIndex) + sizeof(size_t) +
                sizeof(FiberStoredPredictionSample) +
                sizeof(CompactFiberAnchorObservation) +
                (config.peakGradientWeight > 0.0
                    ? sizeof(CachedPresenceGradient)
                    : 0);
            tile.estimatedBytes = tileSamples >
                    std::numeric_limits<size_t>::max() / denseBytes
                ? std::numeric_limits<size_t>::max()
                : tileSamples * denseBytes;
            constexpr size_t compactReferenceBytes =
                sizeof(uint32_t) + sizeof(uint8_t);
            const size_t scratchBytes = maximumCellSamples >
                    std::numeric_limits<size_t>::max() /
                        compactReferenceBytes
                ? std::numeric_limits<size_t>::max()
                : maximumCellSamples * compactReferenceBytes;
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
        const auto exactTileSampleUnion = [&]() {
            std::vector<size_t> zBoundaries;
            zBoundaries.reserve(2 * tiles.size());
            for (const auto& tile : tiles) {
                zBoundaries.push_back(tile.sampleBegin[0]);
                zBoundaries.push_back(tile.sampleEnd[0]);
            }
            std::sort(zBoundaries.begin(), zBoundaries.end());
            zBoundaries.erase(
                std::unique(zBoundaries.begin(), zBoundaries.end()),
                zBoundaries.end());

            size_t unionVoxels = 0;
            for (size_t zIndex = 0;
                 zIndex + 1 < zBoundaries.size(); ++zIndex) {
                const size_t zBegin = zBoundaries[zIndex];
                const size_t zEnd = zBoundaries[zIndex + 1];
                std::vector<const Tile*> zTiles;
                std::vector<size_t> yBoundaries;
                for (const auto& tile : tiles) {
                    if (tile.sampleBegin[0] > zBegin ||
                        tile.sampleEnd[0] < zEnd) {
                        continue;
                    }
                    zTiles.push_back(&tile);
                    yBoundaries.push_back(tile.sampleBegin[1]);
                    yBoundaries.push_back(tile.sampleEnd[1]);
                }
                std::sort(yBoundaries.begin(), yBoundaries.end());
                yBoundaries.erase(
                    std::unique(yBoundaries.begin(), yBoundaries.end()),
                    yBoundaries.end());

                size_t unionArea = 0;
                for (size_t yIndex = 0;
                     yIndex + 1 < yBoundaries.size(); ++yIndex) {
                    const size_t yBegin = yBoundaries[yIndex];
                    const size_t yEnd = yBoundaries[yIndex + 1];
                    std::vector<std::pair<size_t, size_t>> xIntervals;
                    for (const Tile* tile : zTiles) {
                        if (tile->sampleBegin[1] <= yBegin &&
                            tile->sampleEnd[1] >= yEnd) {
                            xIntervals.emplace_back(
                                tile->sampleBegin[2],
                                tile->sampleEnd[2]);
                        }
                    }
                    std::sort(xIntervals.begin(), xIntervals.end());
                    size_t coveredX = 0;
                    if (!xIntervals.empty()) {
                        size_t intervalBegin = xIntervals.front().first;
                        size_t intervalEnd = xIntervals.front().second;
                        for (size_t interval = 1;
                             interval < xIntervals.size(); ++interval) {
                            if (xIntervals[interval].first <= intervalEnd) {
                                intervalEnd = std::max(
                                    intervalEnd,
                                    xIntervals[interval].second);
                                continue;
                            }
                            coveredX += intervalEnd - intervalBegin;
                            intervalBegin = xIntervals[interval].first;
                            intervalEnd = xIntervals[interval].second;
                        }
                        coveredX += intervalEnd - intervalBegin;
                    }
                    const size_t yExtent = yEnd - yBegin;
                    if (coveredX != 0 &&
                        yExtent >
                            std::numeric_limits<size_t>::max() / coveredX) {
                        throw std::overflow_error(
                            "fiber anchor tile union area overflows");
                    }
                    const size_t stripArea = yExtent * coveredX;
                    if (unionArea >
                        std::numeric_limits<size_t>::max() - stripArea) {
                        throw std::overflow_error(
                            "fiber anchor tile union area overflows");
                    }
                    unionArea += stripArea;
                }
                const size_t zExtent = zEnd - zBegin;
                if (unionArea != 0 &&
                    zExtent >
                        std::numeric_limits<size_t>::max() / unionArea) {
                    throw std::overflow_error(
                        "fiber anchor tile union volume overflows");
                }
                const size_t slabVoxels = zExtent * unionArea;
                if (unionVoxels >
                    std::numeric_limits<size_t>::max() - slabVoxels) {
                    throw std::overflow_error(
                        "fiber anchor tile union volume overflows");
                }
                unionVoxels += slabVoxels;
            }
            return unionVoxels;
        };
        report.profile.uniqueTilePredictionVoxels += exactTileSampleUnion();

        const auto tileSampleCount = [&](const Tile& tile) {
            return checkedProduct({
                tile.sampleEnd[0] - tile.sampleBegin[0],
                tile.sampleEnd[1] - tile.sampleBegin[1],
                tile.sampleEnd[2] - tile.sampleBegin[2],
            }, "fiber anchor tile sample");
        };
        const auto overlapVoxels = [&](const Tile& left, const Tile& right) {
            std::array<size_t, 3> extent{};
            for (size_t axis = 0; axis < 3; ++axis) {
                const size_t begin = std::max(
                    left.sampleBegin[axis], right.sampleBegin[axis]);
                const size_t end = std::min(
                    left.sampleEnd[axis], right.sampleEnd[axis]);
                if (begin >= end)
                    return size_t{0};
                extent[axis] = end - begin;
            }
            return checkedProduct(extent, "fiber anchor tile overlap");
        };
        struct TileSamplingGroup {
            std::array<size_t, 2> tileIndices{};
            size_t tileCount = 0;
            size_t estimatedBytes = 0;
        };
        std::vector<TileSamplingGroup> samplingGroups;
        std::vector<uint8_t> grouped(tiles.size(), 0);
        for (size_t first = 0; first < tiles.size(); ++first) {
            if (grouped[first])
                continue;
            size_t bestSecond = tiles.size();
            size_t bestOverlap = 0;
            for (size_t second = first + 1; second < tiles.size(); ++second) {
                if (grouped[second])
                    continue;
                const size_t overlap =
                    overlapVoxels(tiles[first], tiles[second]);
                if (overlap > bestOverlap) {
                    bestOverlap = overlap;
                    bestSecond = second;
                }
            }
            TileSamplingGroup group;
            group.tileIndices[0] = first;
            group.tileCount = 1;
            group.estimatedBytes = tiles[first].estimatedBytes;
            grouped[first] = 1;
            if (bestSecond != tiles.size()) {
                group.tileIndices[1] = bestSecond;
                group.tileCount = 2;
                grouped[bestSecond] = 1;
                const size_t firstRawBytes =
                    checkedMultiply(
                        tileSampleCount(tiles[first]),
                        sizeof(FiberStoredPredictionSample),
                        "fiber anchor retained tile");
                const size_t secondRawBytes =
                    checkedMultiply(
                        tileSampleCount(tiles[bestSecond]),
                        sizeof(FiberStoredPredictionSample),
                        "fiber anchor retained tile");
                group.estimatedBytes = std::max(
                    checkedAdd(
                        tiles[first].estimatedBytes, secondRawBytes,
                        "fiber anchor tile sampling group"),
                    checkedAdd(
                        tiles[bestSecond].estimatedBytes, firstRawBytes,
                        "fiber anchor tile sampling group"));
            }
            samplingGroups.push_back(group);
        }
        size_t maximumGroupBytes = 0;
        for (const auto& group : samplingGroups) {
            maximumGroupBytes = std::max(
                maximumGroupBytes, group.estimatedBytes);
        }
        const size_t memoryWorkers = std::max<size_t>(
            1, config.maximumConcurrentSampleBytes / maximumGroupBytes);
        const size_t workerCount = std::min({
            samplingGroups.size(),
            static_cast<size_t>(config.parallelThreads),
            memoryWorkers,
        });
        report.profile.tiles += tiles.size();
        report.profile.samplingGroups += samplingGroups.size();
        report.profile.workers = std::max(report.profile.workers, workerCount);
        report.profile.tilePlanningSeconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - tilePlanningStart).count();

        std::vector<std::optional<FiberCellAnchorResult>> jobResults(
            requestedCells.size());
        std::vector<std::exception_ptr> jobErrors(requestedCells.size());
        struct WorkerProfile {
            size_t predictionSamplerCalls = 0;
            size_t submittedPredictionVoxels = 0;
            size_t reusedPredictionVoxels = 0;
            size_t candidateObservations = 0;
            size_t retainedObservations = 0;
            size_t supportStencilCells = 0;
            size_t clippedSupportCells = 0;
            size_t gradientAttempts = 0;
            size_t validGradients = 0;
            size_t gradientComputations = 0;
            size_t validGradientComputations = 0;
            size_t fitIterations = 0;
            double coordinateConstructionSeconds = 0.0;
            double predictionSamplingSeconds = 0.0;
            double gradientConstructionSeconds = 0.0;
            double observationConstructionSeconds = 0.0;
            double fittingSeconds = 0.0;
            FiberAnchorFitProfile fit;
        };
        std::vector<WorkerProfile> workerProfiles(workerCount);
        const auto processCell = [&]
            (const CellIndex& cellZYX,
             const Tile& tile,
             const std::vector<CompactFiberAnchorObservation>& observations,
             const std::array<size_t, 3>& sampleShape,
             std::vector<uint32_t>& cellObservationIndices,
             std::vector<uint8_t>& cellGradientValidity,
             WorkerProfile& workerProfile) {
            const auto observationStart = std::chrono::steady_clock::now();
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
            const std::array<size_t, 3> cellSampleShape{
                cellSampleEnd[0] - cellSampleBegin[0],
                cellSampleEnd[1] - cellSampleBegin[1],
                cellSampleEnd[2] - cellSampleBegin[2],
            };
            const size_t maximumObservations = checkedProduct(
                cellSampleShape, "fiber anchor observation");
            cellObservationIndices.clear();
            cellGradientValidity.clear();
            bool fullHalo = true;
            for (size_t axis = 0; axis < 3; ++axis) {
                fullHalo = fullHalo && end[axis] - begin[axis] == cellSize &&
                    begin[axis] >= sampleHalo &&
                    grid.shapeZYX[axis] - end[axis] >= sampleHalo;
            }
            if (fullHalo) {
                ++workerProfile.supportStencilCells;
                workerProfile.candidateObservations += maximumObservations;
                cellObservationIndices.reserve(supportStencilSize);
                cellGradientValidity.reserve(supportStencilSize);
                detail::visitFiberAnchorSupportStencilTileIndices(
                    supportStencil, cellSampleBegin, tile.sampleBegin,
                    sampleShape, [&](uint32_t compactIndex) {
                        cellObservationIndices.push_back(compactIndex);
                        bool gradientValid = false;
                        if (config.peakGradientWeight > 0.0) {
                            ++workerProfile.gradientAttempts;
                            gradientValid = observations[compactIndex]
                                .presenceGradientValid;
                            if (gradientValid)
                                ++workerProfile.validGradients;
                        }
                        cellGradientValidity.push_back(
                            static_cast<uint8_t>(gradientValid));
                    });
            } else {
                ++workerProfile.clippedSupportCells;
                cellObservationIndices.reserve(maximumObservations);
                cellGradientValidity.reserve(maximumObservations);
                for (size_t z = cellSampleBegin[0]; z < cellSampleEnd[0]; ++z) {
                    for (size_t y = cellSampleBegin[1]; y < cellSampleEnd[1]; ++y) {
                        for (size_t x = cellSampleBegin[2]; x < cellSampleEnd[2]; ++x) {
                            ++workerProfile.candidateObservations;
                            const size_t index = tileIndex(z, y, x);
                            const cv::Vec3d position{
                                static_cast<double>(x),
                                static_cast<double>(y),
                                static_cast<double>(z),
                            };
                            const cv::Vec3d delta = position - pivot;
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
                                cellObservationIndices.push_back(
                                    static_cast<uint32_t>(index));
                                bool gradientValid = false;
                                if (config.peakGradientWeight > 0.0) {
                                    ++workerProfile.gradientAttempts;
                                    const bool insideGradientHalo =
                                        z != cellSampleBegin[0] &&
                                        z + 1 < cellSampleEnd[0] &&
                                        y != cellSampleBegin[1] &&
                                        y + 1 < cellSampleEnd[1] &&
                                        x != cellSampleBegin[2] &&
                                        x + 1 < cellSampleEnd[2];
                                    gradientValid = insideGradientHalo &&
                                        observations[index]
                                            .presenceGradientValid;
                                    if (gradientValid)
                                        ++workerProfile.validGradients;
                                }
                                cellGradientValidity.push_back(
                                    static_cast<uint8_t>(gradientValid));
                            }
                        }
                    }
                }
            }
            workerProfile.retainedObservations +=
                cellObservationIndices.size();
            workerProfile.observationConstructionSeconds +=
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - observationStart).count();
            const auto fittingStart = std::chrono::steady_clock::now();
            const IndexedObservationRange observationRange{
                observations, cellObservationIndices,
                cellGradientValidity};
            auto result = fitFiberCellAnchorsImpl(
                cellZYX, begin, end, observationRange, config,
                &workerProfile.fit);
            workerProfile.fittingSeconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - fittingStart).count();
            for (const auto& component : result.components)
                workerProfile.fitIterations += component.anchor.refinementIterations;
            return result;
        };

        std::atomic<size_t> nextJob{0};
        std::atomic<size_t> completedJobs{0};
        std::mutex progressMutex;
        auto lastProgressTime = phaseStart;
        std::exception_ptr progressError;
        const auto worker = [&](size_t workerIndex) {
            auto& workerProfile = workerProfiles[workerIndex];
            while (true) {
                const size_t job = nextJob.fetch_add(1);
                if (job >= samplingGroups.size())
                    break;
                try {
                    const auto& group = samplingGroups[job];
                    const Tile* previousTile = nullptr;
                    std::vector<FiberStoredPredictionSample> previousSamples;
                    for (size_t groupIndex = 0;
                         groupIndex < group.tileCount; ++groupIndex) {
                    const Tile& tile = tiles[group.tileIndices[groupIndex]];
                    const std::array<size_t, 3> sampleShape{
                        tile.sampleEnd[0] - tile.sampleBegin[0],
                        tile.sampleEnd[1] - tile.sampleBegin[1],
                        tile.sampleEnd[2] - tile.sampleBegin[2],
                    };
                    const size_t sampleCount = checkedProduct(
                        sampleShape, "fiber anchor tile sample");
                    if (sampleCount >
                        static_cast<size_t>(
                            std::numeric_limits<uint32_t>::max())) {
                        throw std::runtime_error(
                            "fiber anchor tile exceeds compact index range");
                    }
                    const auto coordinateStart = std::chrono::steady_clock::now();
                    std::vector<CellIndex> indices;
                    indices.reserve(sampleCount);
                    std::vector<size_t> sampledOffsets;
                    sampledOffsets.reserve(sampleCount);
                    std::vector<FiberStoredPredictionSample> samples(sampleCount);
                    const size_t plane = sampleShape[1] * sampleShape[2];
                    for (size_t z = tile.sampleBegin[0]; z < tile.sampleEnd[0]; ++z) {
                        for (size_t y = tile.sampleBegin[1]; y < tile.sampleEnd[1]; ++y) {
                            for (size_t x = tile.sampleBegin[2]; x < tile.sampleEnd[2]; ++x) {
                                const size_t offset =
                                    (z - tile.sampleBegin[0]) * plane +
                                    (y - tile.sampleBegin[1]) * sampleShape[2] +
                                    (x - tile.sampleBegin[2]);
                                const bool reusable = previousTile != nullptr &&
                                    z >= previousTile->sampleBegin[0] &&
                                    z < previousTile->sampleEnd[0] &&
                                    y >= previousTile->sampleBegin[1] &&
                                    y < previousTile->sampleEnd[1] &&
                                    x >= previousTile->sampleBegin[2] &&
                                    x < previousTile->sampleEnd[2];
                                if (reusable) {
                                    const size_t previousWidth =
                                        previousTile->sampleEnd[2] -
                                        previousTile->sampleBegin[2];
                                    const size_t previousPlane =
                                        (previousTile->sampleEnd[1] -
                                         previousTile->sampleBegin[1]) *
                                        previousWidth;
                                    const size_t previousOffset =
                                        (z - previousTile->sampleBegin[0]) *
                                            previousPlane +
                                        (y - previousTile->sampleBegin[1]) *
                                            previousWidth +
                                        (x - previousTile->sampleBegin[2]);
                                    samples[offset] = previousSamples[previousOffset];
                                } else {
                                    indices.push_back({z, y, x});
                                    sampledOffsets.push_back(offset);
                                }
                            }
                        }
                    }
                    workerProfile.coordinateConstructionSeconds +=
                        std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - coordinateStart).count();
                    if (!indices.empty()) {
                        std::vector<FiberStoredPredictionSample> sampled;
                        const auto samplingStart =
                            std::chrono::steady_clock::now();
                        sampler(indices, 1, sampled);
                        workerProfile.predictionSamplingSeconds +=
                            std::chrono::duration<double>(
                                std::chrono::steady_clock::now() -
                                samplingStart).count();
                        ++workerProfile.predictionSamplerCalls;
                        if (sampled.size() != indices.size()) {
                            throw std::runtime_error(
                                "fiber stored prediction sampler returned the wrong sample count");
                        }
                        for (size_t index = 0; index < sampled.size(); ++index)
                            samples[sampledOffsets[index]] = sampled[index];
                    }
                    workerProfile.submittedPredictionVoxels += indices.size();
                    workerProfile.reusedPredictionVoxels +=
                        sampleCount - indices.size();
                    indices.clear();
                    indices.shrink_to_fit();
                    sampledOffsets.clear();
                    sampledOffsets.shrink_to_fit();
                    std::vector<CachedPresenceGradient> gradients;
                    if (config.peakGradientWeight > 0.0) {
                        const auto gradientStart = std::chrono::steady_clock::now();
                        gradients.resize(sampleCount);
                        constexpr std::array<double, 3> smooth{0.25, 0.5, 0.25};
                        constexpr std::array<double, 3> derivative{-0.5, 0.0, 0.5};
                        const size_t plane = sampleShape[1] * sampleShape[2];
                        const auto localIndex = [&](size_t z, size_t y, size_t x) {
                            return (z - tile.sampleBegin[0]) * plane +
                                (y - tile.sampleBegin[1]) * sampleShape[2] +
                                (x - tile.sampleBegin[2]);
                        };
                        for (size_t z = tile.sampleBegin[0] + 1;
                             z + 1 < tile.sampleEnd[0]; ++z) {
                            for (size_t y = tile.sampleBegin[1] + 1;
                                 y + 1 < tile.sampleEnd[1]; ++y) {
                                for (size_t x = tile.sampleBegin[2] + 1;
                                     x + 1 < tile.sampleEnd[2]; ++x) {
                                    ++workerProfile.gradientComputations;
                                    const size_t index = localIndex(z, y, x);
                                    auto& cached = gradients[index];
                                    bool valid = true;
                                    for (int dz = -1; dz <= 1 && valid; ++dz) {
                                        for (int dy = -1; dy <= 1 && valid; ++dy) {
                                            for (int dx = -1; dx <= 1; ++dx) {
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
                                                    valid = false;
                                                    break;
                                                }
                                                const double presence = sample.presence;
                                                cached.value[0] +=
                                                    presence * derivative[dx + 1] *
                                                    smooth[dy + 1] * smooth[dz + 1];
                                                cached.value[1] +=
                                                    presence * smooth[dx + 1] *
                                                    derivative[dy + 1] * smooth[dz + 1];
                                                cached.value[2] +=
                                                    presence * smooth[dx + 1] *
                                                    smooth[dy + 1] * derivative[dz + 1];
                                            }
                                        }
                                    }
                                    cached.valid = valid;
                                    if (valid)
                                        ++workerProfile.validGradientComputations;
                                }
                            }
                        }
                        const double elapsed = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - gradientStart).count();
                        workerProfile.gradientConstructionSeconds += elapsed;
                        workerProfile.observationConstructionSeconds += elapsed;
                    }
                    const auto compactStart = std::chrono::steady_clock::now();
                    std::vector<CompactFiberAnchorObservation> observations(
                        sampleCount);
                    for (size_t z = tile.sampleBegin[0];
                         z < tile.sampleEnd[0]; ++z) {
                        for (size_t y = tile.sampleBegin[1];
                             y < tile.sampleEnd[1]; ++y) {
                            for (size_t x = tile.sampleBegin[2];
                                 x < tile.sampleEnd[2]; ++x) {
                                const size_t index =
                                    (z - tile.sampleBegin[0]) * plane +
                                    (y - tile.sampleBegin[1]) * sampleShape[2] +
                                    (x - tile.sampleBegin[2]);
                                const auto& sample = samples[index];
                                auto& observation = observations[index];
                                observation.positionPredictionXYZ = {
                                    static_cast<float>(x),
                                    static_cast<float>(y),
                                    static_cast<float>(z),
                                };
                                const cv::Vec3d direction = normalized(
                                    sample.direction);
                                observation.direction = {
                                    static_cast<float>(direction[0]),
                                    static_cast<float>(direction[1]),
                                    static_cast<float>(direction[2]),
                                };
                                observation.presence =
                                    static_cast<float>(sample.presence);
                                observation.valid = sample.valid;
                                if (config.peakGradientWeight > 0.0 &&
                                    gradients[index].valid) {
                                    observation
                                        .presenceGradientPredictionXYZ =
                                        gradients[index].value;
                                    observation.presenceGradientValid = true;
                                }
                            }
                        }
                    }
                    workerProfile.observationConstructionSeconds +=
                        std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - compactStart)
                            .count();
                    std::vector<CachedPresenceGradient>().swap(gradients);
                    std::vector<uint32_t> cellObservationIndices;
                    std::vector<uint8_t> cellGradientValidity;
                    for (const size_t cellIndex : tile.cells) {
                        jobResults[cellIndex] = processCell(
                            requestedCells[cellIndex], tile, observations,
                            sampleShape, cellObservationIndices,
                            cellGradientValidity, workerProfile);
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
                    previousTile = &tile;
                    previousSamples = std::move(samples);
                    }
                } catch (...) {
                    for (size_t groupIndex = 0;
                         groupIndex < samplingGroups[job].tileCount;
                         ++groupIndex) {
                        const auto& tile = tiles[
                            samplingGroups[job].tileIndices[groupIndex]];
                        for (const size_t cellIndex : tile.cells)
                            jobErrors[cellIndex] = std::current_exception();
                    }
                }
            }
        };
        const auto cellProcessingStart = std::chrono::steady_clock::now();
        const double cellProcessingCpuStart = processCpuSeconds();
        std::vector<std::thread> workers;
        workers.reserve(workerCount);
        for (size_t workerIndex = 0; workerIndex < workerCount; ++workerIndex)
            workers.emplace_back(worker, workerIndex);
        for (auto& thread : workers)
            thread.join();
        report.profile.cellProcessingSeconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - cellProcessingStart).count();
        report.profile.cellProcessingCpuSeconds +=
            processCpuSeconds() - cellProcessingCpuStart;
        for (const auto& workerProfile : workerProfiles) {
            report.profile.predictionSamplerCalls +=
                workerProfile.predictionSamplerCalls;
            report.profile.submittedPredictionVoxels +=
                workerProfile.submittedPredictionVoxels;
            report.profile.reusedPredictionVoxels +=
                workerProfile.reusedPredictionVoxels;
            report.profile.candidateObservations +=
                workerProfile.candidateObservations;
            report.profile.retainedObservations +=
                workerProfile.retainedObservations;
            report.profile.supportStencilCells +=
                workerProfile.supportStencilCells;
            report.profile.clippedSupportCells +=
                workerProfile.clippedSupportCells;
            report.profile.gradientAttempts += workerProfile.gradientAttempts;
            report.profile.validGradients += workerProfile.validGradients;
            report.profile.gradientComputations +=
                workerProfile.gradientComputations;
            report.profile.validGradientComputations +=
                workerProfile.validGradientComputations;
            report.profile.fitIterations += workerProfile.fitIterations;
            report.profile.coordinateConstructionWorkSeconds +=
                workerProfile.coordinateConstructionSeconds;
            report.profile.predictionSamplingWorkSeconds +=
                workerProfile.predictionSamplingSeconds;
            report.profile.gradientConstructionWorkSeconds +=
                workerProfile.gradientConstructionSeconds;
            report.profile.observationConstructionWorkSeconds +=
                workerProfile.observationConstructionSeconds;
            report.profile.fittingWorkSeconds += workerProfile.fittingSeconds;
            accumulateFitProfile(report.profile.fit, workerProfile.fit);
        }

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

        const auto selectionStart = std::chrono::steady_clock::now();
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
                    ++report.profile.retainPredicateCalls;
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
        report.profile.selectionSeconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - selectionStart).count();
        return results;
    };

    std::vector<FiberCellAnchorResult> contextResults = processCells(
        workCells, true, refinedOnly ? "selected_cells" : "anchor_cells");

    const auto initialDiagnosticsStart = std::chrono::steady_clock::now();
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
        if (!component.removedDuringRobustRefinement)
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
    report.profile.initialDiagnosticsSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - initialDiagnosticsStart).count();
    if (refinedOnly) {
        report.elapsedSeconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        report.profile.elapsedCpuSeconds = processCpuSeconds() - startCpuSeconds;
        return report;
    }

    const auto duplicateSuppressionStart = std::chrono::steady_clock::now();
    suppressFiberAnchorDuplicates(contextResults, config);
    report.profile.duplicateSuppressionSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - duplicateSuppressionStart).count();
    const auto finalizationStart = std::chrono::steady_clock::now();
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
    report.profile.finalizationSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - finalizationStart).count();
    report.elapsedSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    report.profile.elapsedCpuSeconds = processCpuSeconds() - startCpuSeconds;
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
        {"version", 2},
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
            {"robust_maximum_trim_mass_fraction", report.config.robustMaximumTrimMassFraction},
            {"robust_mad_multiplier", report.config.robustMadMultiplier},
            {"robust_minimum_angle_degrees", report.config.robustMinimumAngleDegrees},
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
        {"robust_maximum_trim_mass_fraction", config.robustMaximumTrimMassFraction},
        {"robust_mad_multiplier", config.robustMadMultiplier},
        {"robust_minimum_angle_degrees", config.robustMinimumAngleDegrees},
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
