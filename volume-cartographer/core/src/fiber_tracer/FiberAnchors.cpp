#include "vc/fiber_tracer/FiberAnchors.hpp"
#include "vc/core/util/AtomicFile.hpp"
#include "vc/fiber_tracer/FiberAxisTensor.hpp"
#include "vc/fiber_tracer/PolylineGeometry.hpp"
#include "vc/fiber_tracer/detail/FiberAnchorPeakGrid.hpp"
#include "vc/fiber_tracer/detail/FiberAnchorSupportStencil.hpp"
#include "FiberAnchorObjectives.hpp"
#include "FiberAnchorFinalEvaluation.hpp"
#include "FiberFloatGeometry.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
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
#include <type_traits>
#include <utility>

namespace vc::fiber_tracer {
namespace {

constexpr float kMatrixEpsilon = 1.0e-12F;
constexpr float kGeometryEpsilon = 1.0e-6F;
constexpr float kFloatComparisonEpsilon =
    8.0F * std::numeric_limits<float>::epsilon();

double processCpuSeconds()
{
    const std::clock_t ticks = std::clock();
    return ticks == static_cast<std::clock_t>(-1)
        ? 0.0F
        : static_cast<double>(ticks) / static_cast<double>(CLOCKS_PER_SEC);
}

double segmentAabbDistanceSquared(
    const cv::Vec3d& start,
    const cv::Vec3d& end,
    const cv::Vec3d& low,
    const cv::Vec3d& high)
{
    const cv::Vec3d delta = end - start;
    std::vector<double> breaks{0.0F, 1.0F};
    for (int axis = 0; axis < 3; ++axis) {
        if (std::abs(delta[axis]) <= kGeometryEpsilon)
            continue;
        for (const double bound : {low[axis], high[axis]}) {
            const double t = (bound - start[axis]) / delta[axis];
            if (t > 0.0F && t < 1.0F)
                breaks.push_back(t);
        }
    }
    std::sort(breaks.begin(), breaks.end());
    breaks.erase(std::unique(breaks.begin(), breaks.end()), breaks.end());
    double best = std::numeric_limits<double>::infinity();
    const auto evaluate = [&](double t) {
        const cv::Vec3d point = start + delta * t;
        double squared = 0.0F;
        for (int axis = 0; axis < 3; ++axis) {
            const double outside = point[axis] < low[axis]
                ? low[axis] - point[axis]
                : point[axis] > high[axis]
                    ? point[axis] - high[axis]
                    : 0.0F;
            squared += outside * outside;
        }
        best = std::min(best, squared);
    };
    for (size_t interval = 0; interval + 1 < breaks.size(); ++interval) {
        const double begin = breaks[interval];
        const double finish = breaks[interval + 1];
        evaluate(begin);
        evaluate(finish);
        const double middle = 0.5F * (begin + finish);
        double quadratic = 0.0F;
        double linear = 0.0F;
        for (int axis = 0; axis < 3; ++axis) {
            const double point = start[axis] + delta[axis] * middle;
            double offset = 0.0F;
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

float interpolatedQuantile(const std::vector<float>& sorted, float quantile)
{
    if (sorted.empty())
        throw std::invalid_argument("anchor benchmark quantile population is empty");
    const float rank = quantile * static_cast<float>(sorted.size() - 1);
    const size_t lower = static_cast<size_t>(std::floor(rank));
    const size_t upper = static_cast<size_t>(std::ceil(rank));
    const float fraction = rank - static_cast<float>(lower);
    return sorted[lower] * (1.0F - fraction) + sorted[upper] * fraction;
}

using detail::CompactFiberAnchorObservation;

struct FloatSum {
    void add(float value) { sum += value; }
    float sum = 0.0F;
};

struct FloatStoredPredictionSample {
    cv::Vec3f direction{0.0F, 0.0F, 0.0F};
    float presence = 0.0F;
    bool valid = false;
    bool presenceValid = false;
};

struct WeightedObservation {
    cv::Vec3f position{0.0F, 0.0F, 0.0F};
    cv::Vec3f direction{0.0F, 0.0F, 0.0F};
    float gaussian = 0.0F;
    float weight = 0.0F;
    size_t canonicalIndex = 0;
};

struct FitState {
    std::array<cv::Vec3f, 2> axes{};
    std::vector<uint8_t> assignments;
    float objectiveNumerator = -1.0F;
    size_t iteration = 0;
};

constexpr uint8_t kUnassignedComponent = 2;

struct RefinedComponentState {
    cv::Vec3f axis{1.0F, 0.0F, 0.0F};
    cv::Vec3f position{0.0F, 0.0F, 0.0F};
};

struct RefinedEvaluation {
    std::vector<uint8_t> assignments;
    std::vector<uint8_t> retainedInliers;
    std::array<float, 2> denominators{0.0F, 0.0F};
    std::array<float, 2> numerators{0.0F, 0.0F};
    std::array<float, 2> presenceMasses{0.0F, 0.0F};
    std::array<float, 2> alignedSupports{0.0F, 0.0F};
    std::array<float, 2> directionalCoherences{0.0F, 0.0F};
    std::array<size_t, 2> assignedCounts{0, 0};
    float objective = 0.0F;
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

template <typename Scalar>
[[nodiscard]] size_t robustHistogramBin(Scalar residual)
{
    static_assert(std::is_floating_point_v<Scalar>);
    const Scalar bounded = std::clamp(residual, Scalar{0}, Scalar{1});
    return std::min(
        kRobustHistogramBins - 1,
        static_cast<size_t>(std::floor(
            bounded * static_cast<Scalar>(kRobustHistogramBins))));
}

[[nodiscard]] float robustHistogramCenter(size_t bin)
{
    return (static_cast<float>(bin) + 0.5F) /
        static_cast<float>(kRobustHistogramBins);
}

[[nodiscard]] size_t weightedHistogramQuantileBin(
    const std::array<float, kRobustHistogramBins>& histogram,
    float totalMass,
    float quantile)
{
    if (!(totalMass > 0.0F))
        return kRobustHistogramBins - 1;
    const float target = std::clamp(quantile, 0.0F, 1.0F) * totalMass;
    float cumulative = 0.0F;
    for (size_t bin = 0; bin < histogram.size(); ++bin) {
        cumulative += histogram[bin];
        if (cumulative >= target)
            return bin;
    }
    return kRobustHistogramBins - 1;
}

[[nodiscard]] float histogramMassAbove(
    const std::array<float, kRobustHistogramBins>& histogram,
    size_t cutoffBin)
{
    return std::accumulate(
        histogram.begin() + static_cast<std::ptrdiff_t>(cutoffBin + 1),
        histogram.end(), 0.0F);
}

struct RobustHistogramCutoff {
    FiberAnchorRobustCutoff summary;
    size_t cutoffBin = kRobustHistogramBins - 1;
};

[[nodiscard]] RobustHistogramCutoff selectRobustHistogramCutoff(
    const std::array<float, kRobustHistogramBins>& residualHistogram,
    const std::array<float, kRobustHistogramBins>& deviationHistogram,
    float totalMass,
    float median,
    float maximumTrimMassFraction,
    float madMultiplier,
    float minimumAngleDegrees)
{
    RobustHistogramCutoff result;
    result.summary.totalMass = totalMass;
    result.summary.retainedMass = totalMass;
    if (!(totalMass > 0.0F) || !(maximumTrimMassFraction > 0.0F))
        return result;

    const float mad = robustHistogramCenter(weightedHistogramQuantileBin(
        deviationHistogram, totalMass, 0.5F));
    const float floorRadians = minimumAngleDegrees * std::acos(-1.0F) / 180.0F;
    const float floorResidual = std::sin(floorRadians) * std::sin(floorRadians);
    result.cutoffBin = robustHistogramBin(std::clamp(
        std::max(median + madMultiplier * mad, floorResidual), 0.0F, 1.0F));
    result.summary.candidateTrimmedMass = histogramMassAbove(
        residualHistogram, result.cutoffBin);
    result.summary.detectedOutliers =
        result.summary.candidateTrimmedMass > 0.0F;
    if (!result.summary.detectedOutliers)
        return result;

    const float maximumTrimmed = maximumTrimMassFraction * totalMass;
    if (result.summary.candidateTrimmedMass > maximumTrimmed) {
        result.cutoffBin = std::max(
            result.cutoffBin,
            weightedHistogramQuantileBin(
                residualHistogram, totalMass,
                1.0F - maximumTrimMassFraction));
    }
    result.summary.cutoffResidual =
        static_cast<float>(result.cutoffBin + 1) /
        static_cast<float>(kRobustHistogramBins);
    result.summary.trimmedMass = histogramMassAbove(
        residualHistogram, result.cutoffBin);
    result.summary.retainedMass = totalMass - result.summary.trimmedMass;
    return result;
}

struct PeakOwnerBounds {
    cv::Vec3f lower{0.0F, 0.0F, 0.0F};
    cv::Vec3f upper{0.0F, 0.0F, 0.0F};
};

constexpr size_t kNoDiagnosticId = std::numeric_limits<size_t>::max();

[[nodiscard]] bool finiteVector(const cv::Vec3f& value)
{
    return std::isfinite(value[0]) && std::isfinite(value[1]) &&
           std::isfinite(value[2]);
}

[[nodiscard]] FloatStoredPredictionSample narrowStoredPredictionSample(
    const FiberStoredPredictionSample& input)
{
    FloatStoredPredictionSample output;
    output.valid = input.valid;
    output.presenceValid = input.presenceValid;
    if (std::isfinite(input.presence) &&
        input.presence >=
            static_cast<double>(std::numeric_limits<float>::lowest()) &&
        input.presence <=
            static_cast<double>(std::numeric_limits<float>::max())) {
        output.presence = static_cast<float>(input.presence);
    } else {
        output.presence = std::numeric_limits<float>::quiet_NaN();
    }

    const double scale = std::max({
        std::abs(input.direction[0]), std::abs(input.direction[1]),
        std::abs(input.direction[2])});
    if (!(scale > 0.0) || !std::isfinite(scale))
        return output;
    cv::Vec3d direction = input.direction / scale;
    const double norm2 = direction.dot(direction);
    if (!(norm2 > 0.0) || !std::isfinite(norm2))
        return output;
    direction /= std::sqrt(norm2);
    output.direction = {
        static_cast<float>(direction[0]),
        static_cast<float>(direction[1]),
        static_cast<float>(direction[2]),
    };
    return output;
}

template <typename Observation>
using FiberAnchorProposalScalar = std::conditional_t<
    std::is_same_v<std::remove_cvref_t<Observation>,
        CompactFiberAnchorObservation>,
    float,
    float>;

template <typename Scalar>
using FiberAnchorProposalVector = cv::Vec<Scalar, 3>;

template <typename Scalar>
struct FiberAnchorProposalComponent {
    FiberAnchorProposalVector<Scalar> axis{Scalar{1}, Scalar{0}, Scalar{0}};
    FiberAnchorProposalVector<Scalar> position{
        Scalar{0}, Scalar{0}, Scalar{0}};
};

template <typename Scalar, typename Observation>
[[nodiscard]] FiberAnchorProposalVector<Scalar> proposalObservationPosition(
    const Observation& observation)
{
    return {
        static_cast<Scalar>(observation.positionPredictionXYZ[0]),
        static_cast<Scalar>(observation.positionPredictionXYZ[1]),
        static_cast<Scalar>(observation.positionPredictionXYZ[2]),
    };
}

template <typename Scalar, typename Observation>
[[nodiscard]] bool usableProposalDirection(
    const Observation& observation,
    Scalar presenceFloor,
    FiberAnchorProposalVector<Scalar>& direction)
{
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
    if constexpr (!std::is_same_v<std::remove_cvref_t<Observation>,
                      CompactFiberAnchorObservation>) {
        const Scalar norm2 = direction.dot(direction);
        if (!(norm2 > static_cast<Scalar>(kMatrixEpsilon * kMatrixEpsilon)) ||
            !std::isfinite(norm2)) {
            direction = {Scalar{0}, Scalar{0}, Scalar{0}};
        } else {
            direction /= std::sqrt(norm2);
        }
    }
    return direction.dot(direction) > static_cast<Scalar>(kMatrixEpsilon);
}

template <typename Scalar, typename Observation>
[[nodiscard]] Scalar proposalTransverseGaussian(
    const Observation& observation,
    const FiberAnchorProposalComponent<Scalar>& component,
    const FiberAnchorProposalVector<Scalar>& pivot,
    Scalar axialSupportHalfWidth,
    Scalar gaussianCutoff,
    Scalar gaussianSigma)
{
    const auto position = proposalObservationPosition<Scalar>(observation);
    if (!finiteVector(position))
        return Scalar{0};
    const Scalar axial = (position - pivot).dot(component.axis);
    if (std::abs(axial) > axialSupportHalfWidth)
        return Scalar{0};
    const auto offset = position - component.position;
    const auto transverse =
        offset - component.axis * offset.dot(component.axis);
    const Scalar distanceSquared = transverse.dot(transverse);
    if (distanceSquared > gaussianCutoff * gaussianCutoff)
        return Scalar{0};
    return std::exp(-distanceSquared /
        (Scalar{2} * gaussianSigma * gaussianSigma));
}

template <typename Observation>
[[nodiscard]] cv::Vec3f observationPosition(const Observation& observation)
{
    return cv::Vec3f{observation.positionPredictionXYZ};
}

template <typename Observation>
[[nodiscard]] float observationPresence(const Observation& observation)
{
    return static_cast<float>(observation.presence);
}

template <typename Observation>
[[nodiscard]] cv::Vec3f observationGradient(const Observation& observation)
{
    return cv::Vec3f{observation.presenceGradientPredictionXYZ};
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

    [[nodiscard]] std::span<const Observation> observationStorage() const
    {
        return observations_;
    }

    [[nodiscard]] std::span<const uint32_t> observationIndices() const
    {
        return indices_;
    }

private:
    const std::vector<Observation>& observations_;
    const std::vector<uint32_t>& indices_;
    const std::vector<uint8_t>& gradientValidity_;
};

class DenseOwnedCompactObservationRange {
public:
    DenseOwnedCompactObservationRange(
        const std::vector<CompactFiberAnchorObservation>& observations,
        const std::array<size_t, 3>& tileBeginZYX,
        const std::array<size_t, 3>& tileShapeZYX,
        const std::array<size_t, 3>& cellBeginZYX,
        const std::array<size_t, 3>& cellEndZYX)
        : observations_{observations},
          layout_{detail::buildFiberAnchorOwnedCellTileLayout(
              observations.size(), tileBeginZYX, tileShapeZYX,
              cellBeginZYX, cellEndZYX)}
    {
    }

    [[nodiscard]] size_t size() const { return layout_.ownedSize; }

    template <typename Visitor>
    void visit(Visitor&& visitor) const
    {
        detail::visitFiberAnchorOwnedCellTileIndices(
            layout_, [&](size_t observationIndex, size_t canonicalIndex) {
                visitor(observations_[observationIndex], canonicalIndex);
            });
    }

private:
    const std::vector<CompactFiberAnchorObservation>& observations_;
    detail::FiberAnchorOwnedCellTileLayout layout_;
};

template <typename ObservationRange, typename Visitor>
void visitObservations(const ObservationRange& observations, Visitor&& visitor)
{
    for (size_t index = 0; index < observations.size(); ++index)
        visitor(observations[index], index);
}

template <typename Visitor>
void visitObservations(
    const DenseOwnedCompactObservationRange& observations,
    Visitor&& visitor)
{
    observations.visit(std::forward<Visitor>(visitor));
}

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

[[nodiscard]] cv::Vec3f normalized(const cv::Vec3f& value)
{
    const float norm2 = value.dot(value);
    if (!(norm2 > kMatrixEpsilon * kMatrixEpsilon) || !std::isfinite(norm2))
        return {0.0F, 0.0F, 0.0F};
    return value / std::sqrt(norm2);
}

[[nodiscard]] std::array<cv::Vec3f, 2> transverseBasis(
    const cv::Vec3f& axis)
{
    size_t referenceIndex = 0;
    for (size_t index = 1; index < 3; ++index) {
        if (std::abs(axis[static_cast<int>(index)]) <
            std::abs(axis[static_cast<int>(referenceIndex)])) {
            referenceIndex = index;
        }
    }
    cv::Vec3f reference{0.0F, 0.0F, 0.0F};
    reference[static_cast<int>(referenceIndex)] = 1.0F;
    const cv::Vec3f first = normalized(
        reference - axis * reference.dot(axis));
    return {first, normalized(axis.cross(first))};
}

[[nodiscard]] bool insidePeakDomain(
    const cv::Vec3f& point,
    const cv::Vec3f& pivot,
    const PeakOwnerBounds& owner,
    float radius)
{
    const cv::Vec3f offset = point - pivot;
    if (offset.dot(offset) > radius * radius + kGeometryEpsilon)
        return false;
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
        if (point[coordinate] < owner.lower[coordinate] ||
            point[coordinate] > owner.upper[coordinate]) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] cv::Matx33f weightedTensor(
    const std::vector<WeightedObservation>& observations,
    const std::vector<uint8_t>* assignments,
    uint8_t component)
{
    std::array<FloatSum, 9> sums;
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
    cv::Matx33f tensor;
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column)
            tensor(row, column) = sums[static_cast<size_t>(row * 3 + column)].sum;
    }
    return tensor;
}

[[nodiscard]] std::vector<uint8_t> assignObservations(
    const std::vector<WeightedObservation>& observations,
    const std::array<cv::Vec3f, 2>& axes)
{
    std::vector<uint8_t> assignments(observations.size(), 0);
    for (size_t index = 0; index < observations.size(); ++index) {
        const float dot0 = observations[index].direction.dot(axes[0]);
        const float dot1 = observations[index].direction.dot(axes[1]);
        assignments[index] = dot0 * dot0 >= dot1 * dot1 ? 0 : 1;
    }
    return assignments;
}

[[nodiscard]] float objectiveNumerator(
    const std::vector<WeightedObservation>& observations,
    const std::array<cv::Vec3f, 2>& axes)
{
    FloatSum result;
    for (const auto& observation : observations) {
        const float dot0 = observation.direction.dot(axes[0]);
        const float dot1 = observation.direction.dot(axes[1]);
        result.add(observation.weight * std::max(dot0 * dot0, dot1 * dot1));
    }
    return result.sum;
}

[[nodiscard]] float projectiveUpdate(const cv::Vec3f& before, const cv::Vec3f& after)
{
    return 1.0F - std::clamp(std::abs(before.dot(after)), 0.0F, 1.0F);
}

[[nodiscard]] cv::Vec3f projectToConstraintPlane(
    const cv::Vec3f& point,
    const cv::Vec3f& pivot,
    const cv::Vec3f& axis)
{
    return point - axis * ((point - pivot).dot(axis));
}

[[nodiscard]] cv::Vec3f clampToWindow(
    const cv::Vec3f& point,
    const cv::Vec3f& pivot,
    const cv::Vec3f& axis,
    float radius,
    const cv::Vec3f& lower,
    const cv::Vec3f& upper)
{
    cv::Vec3f offset = projectToConstraintPlane(point, pivot, axis) - pivot;
    const float length = std::sqrt(std::max(0.0F, offset.dot(offset)));
    if (length > radius)
        offset *= radius / length;
    float scale = 1.0F;
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
        if (offset[coordinate] > 0.0F) {
            scale = std::min(
                scale, (upper[coordinate] - pivot[coordinate]) /
                    offset[coordinate]);
        } else if (offset[coordinate] < 0.0F) {
            scale = std::min(
                scale, (lower[coordinate] - pivot[coordinate]) /
                    offset[coordinate]);
        }
    }
    return pivot + offset * std::clamp(scale, 0.0F, 1.0F);
}

template <typename Observation>
[[nodiscard]] float transverseGaussian(
    const Observation& observation,
    const RefinedComponentState& component,
    const cv::Vec3f& pivot,
    const FiberAnchorConfig& config)
{
    const cv::Vec3f position = observationPosition(observation);
    if (!finiteVector(position))
        return 0.0F;
    const float axial = (position - pivot).dot(component.axis);
    if (std::abs(axial) > config.axialSupportHalfWidthPredictionVoxels)
        return 0.0F;
    const cv::Vec3f offset = position - component.position;
    const cv::Vec3f transverse = offset - component.axis * offset.dot(component.axis);
    const float distanceSquared = transverse.dot(transverse);
    const float cutoff = config.gaussianCutoffSigmas * config.gaussianSigmaPredictionVoxels;
    if (distanceSquared > cutoff * cutoff)
        return 0.0F;
    return std::exp(-distanceSquared /
        (2.0F * config.gaussianSigmaPredictionVoxels *
         config.gaussianSigmaPredictionVoxels));
}

template <typename Observation>
[[nodiscard]] bool usableDirectionObservation(
    const Observation& observation,
    const FiberAnchorConfig& config,
    cv::Vec3f& direction)
{
    const float presence = observationPresence(observation);
    if (!observation.valid || !finiteVector(observation.direction) ||
        !std::isfinite(presence) ||
        presence < config.observationPresenceFloor ||
        presence < 0.0F || presence > 1.0F) {
        return false;
    }
    direction = normalized(cv::Vec3f{observation.direction});
    return direction.dot(direction) > kMatrixEpsilon;
}

[[nodiscard]] bool usableDirectionObservation(
    const CompactFiberAnchorObservation& observation,
    const FiberAnchorConfig& config,
    cv::Vec3f& direction)
{
    const float presence = observationPresence(observation);
    if (!observation.valid || !finiteVector(observation.direction) ||
        !std::isfinite(presence) ||
        presence < config.observationPresenceFloor ||
        presence < 0.0F || presence > 1.0F) {
        return false;
    }
    direction = cv::Vec3f{observation.direction};
    return direction.dot(direction) > kMatrixEpsilon;
}

struct RobustDirectionProposal {
    std::vector<uint8_t> assignments;
    std::vector<uint8_t> retainedInliers;
    std::array<cv::Vec3f, 2> axes{};
    std::array<bool, 2> unique{false, false};
};

template <typename ObservationRange>
[[nodiscard]] RobustDirectionProposal robustDirectionProposal(
    const ObservationRange& observations,
    const std::array<RefinedComponentState, 2>& components,
    size_t activeComponents,
    const cv::Vec3f& pivot,
    const FiberAnchorConfig& config,
    FiberAnchorFitProfile* profile,
    bool computeAxes)
{
    using Observation =
        std::remove_cvref_t<decltype(observations[size_t{0}])>;
    using Scalar = FiberAnchorProposalScalar<Observation>;
    constexpr bool compactFloatProposal = std::is_same_v<
        Observation, CompactFiberAnchorObservation>;
    using TensorEntry = std::conditional_t<
        compactFloatProposal, float, FloatSum>;
    using ScalarTensor = std::array<TensorEntry, 6>;
    using TensorHistogram = std::array<ScalarTensor, kRobustHistogramBins>;

    RobustDirectionProposal proposal;
    proposal.assignments.assign(observations.size(), kUnassignedComponent);
    proposal.retainedInliers.assign(observations.size(), 0);
    std::array<std::array<Scalar, kRobustHistogramBins>, 2>
        scalarResidualHistograms{};
    std::optional<std::array<TensorHistogram, 2>> tensorHistograms;
    if (computeAxes)
        tensorHistograms.emplace();
    std::array<float, 2> totalMass{0.0F, 0.0F};

    std::array<FiberAnchorProposalComponent<Scalar>, 2> scalarComponents{};
    for (size_t component = 0; component < activeComponents; ++component) {
        for (int coordinate = 0; coordinate < 3; ++coordinate) {
            scalarComponents[component].axis[coordinate] =
                static_cast<Scalar>(components[component].axis[coordinate]);
            scalarComponents[component].position[coordinate] =
                static_cast<Scalar>(components[component].position[coordinate]);
        }
    }
    const FiberAnchorProposalVector<Scalar> scalarPivot{
        static_cast<Scalar>(pivot[0]),
        static_cast<Scalar>(pivot[1]),
        static_cast<Scalar>(pivot[2]),
    };
    const Scalar presenceFloor =
        static_cast<Scalar>(config.observationPresenceFloor);
    const Scalar axialSupportHalfWidth =
        static_cast<Scalar>(config.axialSupportHalfWidthPredictionVoxels);
    const Scalar gaussianSigma =
        static_cast<Scalar>(config.gaussianSigmaPredictionVoxels);
    const Scalar gaussianCutoff = static_cast<Scalar>(
        config.gaussianCutoffSigmas *
        config.gaussianSigmaPredictionVoxels);

    for (size_t index = 0; index < observations.size(); ++index) {
        const auto& observation = observations[index];
        FiberAnchorProposalVector<Scalar> direction;
        if (!usableProposalDirection(
                observation, presenceFloor, direction)) {
            continue;
        }
        const Scalar presence = static_cast<Scalar>(observation.presence);
        std::array<Scalar, 2> gaussian{Scalar{0}, Scalar{0}};
        std::array<Scalar, 2> alignment{Scalar{0}, Scalar{0}};
        std::array<Scalar, 2> score{Scalar{0}, Scalar{0}};
        for (size_t component = 0; component < activeComponents; ++component) {
            gaussian[component] = proposalTransverseGaussian(
                observation, scalarComponents[component], scalarPivot,
                axialSupportHalfWidth, gaussianCutoff, gaussianSigma);
            const Scalar dot =
                direction.dot(scalarComponents[component].axis);
            alignment[component] = dot * dot;
            score[component] =
                gaussian[component] * presence * alignment[component];
        }
        uint8_t assigned = kUnassignedComponent;
        for (size_t component = 0; component < activeComponents; ++component) {
            if (score[component] > Scalar{0} &&
                (assigned == kUnassignedComponent ||
                 score[component] > score[assigned])) {
                assigned = static_cast<uint8_t>(component);
            }
        }
        proposal.assignments[index] = assigned;
        if (assigned == kUnassignedComponent)
            continue;
        const Scalar mass = gaussian[assigned] * presence;
        const Scalar residual = std::clamp(
            Scalar{1} - alignment[assigned], Scalar{0}, Scalar{1});
        const size_t residualBin = robustHistogramBin(residual);
        proposal.retainedInliers[index] = static_cast<uint8_t>(residualBin);
        scalarResidualHistograms[assigned][residualBin] += mass;
        if constexpr (!compactFloatProposal)
            totalMass[assigned] += static_cast<float>(mass);
        if (tensorHistograms.has_value()) {
            auto& tensor = (*tensorHistograms)[assigned][residualBin];
            const std::array<Scalar, 6> values{
                mass * direction[0] * direction[0],
                mass * direction[0] * direction[1],
                mass * direction[0] * direction[2],
                mass * direction[1] * direction[1],
                mass * direction[1] * direction[2],
                mass * direction[2] * direction[2],
            };
            for (size_t entry = 0; entry < tensor.size(); ++entry) {
                if constexpr (compactFloatProposal)
                    tensor[entry] += values[entry];
                else
                    tensor[entry].add(values[entry]);
            }
        }
    }
    if (profile != nullptr)
        profile->localTensorObservationVisits += observations.size();

    std::array<std::array<float, kRobustHistogramBins>, 2>
        residualHistograms{};
    for (size_t component = 0; component < activeComponents; ++component) {
        for (size_t residualBin = 0;
             residualBin < kRobustHistogramBins; ++residualBin) {
            const float mass = static_cast<float>(
                scalarResidualHistograms[component][residualBin]);
            residualHistograms[component][residualBin] = mass;
            if constexpr (compactFloatProposal)
                totalMass[component] += mass;
        }
    }

    std::array<size_t, 2> cutoffBins{
        kRobustHistogramBins - 1, kRobustHistogramBins - 1};
    for (size_t component = 0; component < activeComponents; ++component) {
        if (!(totalMass[component] > 0.0F) ||
            !(config.robustMaximumTrimMassFraction > 0.0F)) {
            if (profile != nullptr)
                profile->robustRetainedMass += totalMass[component];
            continue;
        }
        const size_t medianBin = weightedHistogramQuantileBin(
            residualHistograms[component], totalMass[component], 0.5F);
        const float median = robustHistogramCenter(medianBin);
        std::array<float, kRobustHistogramBins> deviationHistogram{};
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
            if (cutoff.summary.trimmedMass > 0.0F)
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
        std::array<float, 6> sums{};
        std::array<FloatSum, 6> compensatedSums{};
        for (size_t residualBin = 0;
             residualBin <= cutoffBins[component]; ++residualBin) {
            for (size_t entry = 0; entry < sums.size(); ++entry) {
                if constexpr (compactFloatProposal) {
                    sums[entry] += static_cast<float>(
                        (*tensorHistograms)[component][residualBin][entry]);
                } else {
                    compensatedSums[entry].add(
                        (*tensorHistograms)[component]
                            [residualBin][entry].sum);
                }
            }
        }
        if constexpr (!compactFloatProposal) {
            for (size_t entry = 0; entry < sums.size(); ++entry)
                sums[entry] = compensatedSums[entry].sum;
        }
        const cv::Matx33f tensor{
            sums[0], sums[1], sums[2],
            sums[1], sums[3], sums[4],
            sums[2], sums[4], sums[5],
        };
        const FiberPrincipalAxisF principal = principalFiberAxisF(tensor);
        proposal.unique[component] = principal.unique;
        if (principal.unique)
            proposal.axes[component] = principal.axis;
    }
    return proposal;
}

[[nodiscard]] detail::FiberAnchorObjectiveConfig objectiveConfig(
    const FiberAnchorConfig& config)
{
    return {
        config.gaussianSigmaPredictionVoxels,
        config.gaussianCutoffSigmas,
        config.axialSupportHalfWidthPredictionVoxels,
        config.observationPresenceFloor,
    };
}

[[nodiscard]] std::array<detail::FiberAnchorObjectiveComponent, 2>
objectiveComponents(const std::array<RefinedComponentState, 2>& components)
{
    std::array<detail::FiberAnchorObjectiveComponent, 2> result;
    for (size_t component = 0; component < result.size(); ++component) {
        result[component].axis = components[component].axis;
        result[component].position = components[component].position;
    }
    return result;
}

[[nodiscard]] float retainedSpatialObjective(
    const std::vector<FiberAnchorObservation>& observations,
    const std::array<RefinedComponentState, 2>& components,
    size_t activeComponents,
    const std::vector<uint8_t>& assignments,
    const std::vector<uint8_t>& retainedInliers,
    const cv::Vec3f& pivot,
    const FiberAnchorConfig& config)
{
    return detail::retainedSpatialObjectiveExpanded(
        observations, objectiveComponents(components), activeComponents,
        assignments, retainedInliers, pivot, objectiveConfig(config));
}

[[nodiscard]] std::array<float, 2> retainedSpatialObjectivePair(
    const std::vector<FiberAnchorObservation>& observations,
    const std::array<RefinedComponentState, 2>& first,
    const std::array<RefinedComponentState, 2>& second,
    size_t activeComponents,
    const std::vector<uint8_t>& assignments,
    const std::vector<uint8_t>& retainedInliers,
    const cv::Vec3f& pivot,
    const FiberAnchorConfig& config)
{
    return detail::retainedSpatialObjectivePairExpanded(
        observations, objectiveComponents(first), objectiveComponents(second),
        activeComponents, assignments, retainedInliers, pivot,
        objectiveConfig(config));
}

[[nodiscard]] float retainedSpatialObjective(
    const IndexedObservationRange<CompactFiberAnchorObservation>& observations,
    const std::array<RefinedComponentState, 2>& components,
    size_t activeComponents,
    const std::vector<uint8_t>& assignments,
    const std::vector<uint8_t>& retainedInliers,
    const cv::Vec3f& pivot,
    const FiberAnchorConfig& config)
{
    return detail::retainedSpatialObjectiveCompact(
        observations.observationStorage(), observations.observationIndices(),
        objectiveComponents(components), activeComponents, assignments,
        retainedInliers, pivot, objectiveConfig(config));
}

[[nodiscard]] std::array<float, 2> retainedSpatialObjectivePair(
    const IndexedObservationRange<CompactFiberAnchorObservation>& observations,
    const std::array<RefinedComponentState, 2>& first,
    const std::array<RefinedComponentState, 2>& second,
    size_t activeComponents,
    const std::vector<uint8_t>& assignments,
    const std::vector<uint8_t>& retainedInliers,
    const cv::Vec3f& pivot,
    const FiberAnchorConfig& config)
{
    return detail::retainedSpatialObjectivePairCompact(
        observations.observationStorage(), observations.observationIndices(),
        objectiveComponents(first), objectiveComponents(second),
        activeComponents, assignments, retainedInliers, pivot,
        objectiveConfig(config));
}

[[nodiscard]] RefinedEvaluation finalEvaluationResult(
    const detail::FiberAnchorFinalEvaluation& reduced,
    const std::vector<uint8_t>& assignments,
    const std::vector<uint8_t>& retainedInliers)
{
    RefinedEvaluation evaluation;
    evaluation.assignments = assignments;
    evaluation.retainedInliers = retainedInliers;
    for (size_t component = 0; component < 2; ++component) {
        evaluation.denominators[component] =
            static_cast<float>(reduced.denominators[component]);
        evaluation.numerators[component] =
            static_cast<float>(reduced.numerators[component]);
        evaluation.presenceMasses[component] =
            static_cast<float>(reduced.presenceMasses[component]);
        evaluation.alignedSupports[component] =
            static_cast<float>(reduced.alignedSupports[component]);
        evaluation.directionalCoherences[component] =
            static_cast<float>(reduced.directionalCoherences[component]);
        evaluation.assignedCounts[component] = reduced.assignedCounts[component];
    }
    evaluation.objective = static_cast<float>(reduced.objective);
    return evaluation;
}

[[nodiscard]] RefinedEvaluation evaluateFinalRefinedState(
    const std::vector<FiberAnchorObservation>& observations,
    const std::array<RefinedComponentState, 2>& components,
    size_t activeComponents,
    const std::vector<uint8_t>& assignments,
    const std::vector<uint8_t>& retainedInliers,
    const cv::Vec3f& pivot,
    const FiberAnchorConfig& config)
{
    return finalEvaluationResult(
        detail::finalAnchorEvaluationExpanded(
            observations, objectiveComponents(components), activeComponents,
            assignments, retainedInliers, pivot, objectiveConfig(config)),
        assignments, retainedInliers);
}

[[nodiscard]] RefinedEvaluation evaluateFinalRefinedState(
    const IndexedObservationRange<CompactFiberAnchorObservation>& observations,
    const std::array<RefinedComponentState, 2>& components,
    size_t activeComponents,
    const std::vector<uint8_t>& assignments,
    const std::vector<uint8_t>& retainedInliers,
    const cv::Vec3f& pivot,
    const FiberAnchorConfig& config)
{
    return finalEvaluationResult(
        detail::finalAnchorEvaluationCompact(
            observations.observationStorage(), observations.observationIndices(),
            objectiveComponents(components), activeComponents, assignments,
            retainedInliers, pivot, objectiveConfig(config)),
        assignments, retainedInliers);
}

template <typename ObservationRange>
[[nodiscard]] RefinedFitState refineLocalComponents(
    const ObservationRange& observations,
    const cv::Vec3f& pivot,
    const std::array<cv::Vec3f, 2>& seedAxes,
    const std::array<size_t, 2>& seedComponentIds,
    size_t activeComponents,
    const FiberAnchorConfig& config,
    FiberAnchorFitProfile* profile)
{
    using RefinementClock = std::chrono::steady_clock;
    RefinedFitState state;
    state.activeComponents = activeComponents;
    for (size_t component = 0; component < activeComponents; ++component) {
        state.components[component].axis = canonicalFiberAxisF(seedAxes[component]);
        state.components[component].position = pivot;
        state.componentIds[component] = seedComponentIds[component];
    }
    cv::Vec3f lower{
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(),
    };
    cv::Vec3f upper{
        -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
    };
    for (size_t index = 0; index < observations.size(); ++index) {
        const cv::Vec3f position = observationPosition(observations[index]);
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
            const cv::Vec3f projectedCurrent = projectToConstraintPlane(
                state.components[component].position,
                pivot,
                proposed[component].axis);
            std::array<FloatSum, 3> centroid;
            FloatSum centroidMass;
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
                const float gaussian = transverseGaussian(
                    observation, centered, pivot, config);
                cv::Vec3f direction;
                if (!usableDirectionObservation(
                        observation, config, direction)) {
                    continue;
                }
                const float dot = direction.dot(centered.axis);
                const float weight = gaussian *
                    observationPresence(observation) * dot * dot;
                centroidMass.add(weight);
                const cv::Vec3f position = observationPosition(observation);
                for (int axis = 0; axis < 3; ++axis) {
                    centroid[axis].add(
                        weight * position[axis]);
                }
            }
            if (centroidMass.sum > 0.0F) {
                const cv::Vec3f mean{
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
        float acceptedDirectionUpdate = 0.0F;
        float maximumTargetDisplacement = 0.0F;
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
            const cv::Vec3f displacement =
                proposed[component].position - baseline[component].position;
            maximumTargetDisplacement = std::max(
                maximumTargetDisplacement,
                std::sqrt(std::max(0.0F, displacement.dot(displacement))));
        }
        const auto evaluationStart = profile != nullptr
            ? RefinementClock::now()
            : RefinementClock::time_point{};
        if (profile != nullptr)
            profile->refinedEvaluationObservationVisits += observations.size();
        auto acceptedComponents = baseline;
        float acceptedPositionUpdate = 0.0F;
        const auto fractions = fiberAnchorSpatialBacktrackingFractions(
            maximumTargetDisplacement,
            config.peakGridStepPredictionVoxels);
        float baselineObjective = 0.0F;
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
            float objective = 0.0F;
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
            const float tolerance = config.convergenceTolerance *
                std::max(1.0F, std::abs(baselineObjective));
            if (objective <= baselineObjective + tolerance)
                continue;
            acceptedComponents = candidate;
            if (profile != nullptr)
                ++profile->spatialCandidatesAcceptedByDepth[depth];
            for (size_t component = 0; component < state.activeComponents; ++component) {
                const cv::Vec3f delta =
                    acceptedComponents[component].position -
                    baseline[component].position;
                acceptedPositionUpdate = std::max(
                    acceptedPositionUpdate,
                    std::sqrt(std::max(0.0F, delta.dot(delta))));
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
        const float positionTolerance = std::max(
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
    cv::Vec3f observedLower{
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(),
    };
    cv::Vec3f observedUpper{
        -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
    };
    for (size_t index = 0; index < observations.size(); ++index) {
        const cv::Vec3f position = observationPosition(observations[index]);
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
        const float begin = static_cast<float>(cellBeginZYX[zyx]);
        const float end = static_cast<float>(cellEndZYX[zyx]);
        owner.lower[coordinate] = std::max(observedLower[coordinate], begin - 0.5F);
        const float voronoiUpper = end - 0.5F;
        if (observedUpper[coordinate] <= end - 1.0F) {
            owner.upper[coordinate] = observedUpper[coordinate];
        } else {
            owner.upper[coordinate] = std::nextafter(
                std::min(observedUpper[coordinate], voronoiUpper),
                -std::numeric_limits<float>::infinity());
        }
    }
    return owner;
}

struct DirectionConditionedPeak {
    cv::Vec3f discrete{0.0F, 0.0F, 0.0F};
    cv::Vec3f separable1d{0.0F, 0.0F, 0.0F};
    cv::Vec3f joint2d{0.0F, 0.0F, 0.0F};
};

template <typename ObservationRange>
[[nodiscard]] DirectionConditionedPeak findDirectionConditionedLocalPeak(
    const ObservationRange& observations,
    const cv::Vec3f& pivot,
    const PeakOwnerBounds& owner,
    const std::array<RefinedComponentState, 2>& components,
    size_t selectedComponent,
    const std::vector<uint8_t>& assignments,
    const std::vector<uint8_t>& retainedInliers,
    const FiberAnchorConfig& config,
    FiberAnchorFitProfile* profile)
{
    const cv::Vec3f axis = components[selectedComponent].axis;
    const auto basis = transverseBasis(axis);
    const float cutoff = static_cast<float>(
        config.gaussianCutoffSigmas * config.peakSigmaPredictionVoxels);
    const float axialCutoff =
        config.gaussianCutoffSigmas *
        config.peakAxialSigmaPredictionVoxels;
    const float unionRadius = config.localWindowRadiusPredictionVoxels + cutoff;
    const float invTwoTransverseSigma2 = static_cast<float>(1.0F /
        (2.0F * config.peakSigmaPredictionVoxels *
         config.peakSigmaPredictionVoxels));
    const float invTwoAxialSigma2 = 1.0F /
        (2.0F * config.peakAxialSigmaPredictionVoxels *
         config.peakAxialSigmaPredictionVoxels);

    constexpr uint32_t kNoPeakEvidence = std::numeric_limits<uint32_t>::max();
    struct PeakResponseObservation {
        float first = 0.0F;
        float second = 0.0F;
        float axialGaussian = 0.0F;
        float signal = 0.0F;
    };
    struct PeakEvidenceObservation {
        float directionAlignmentSquared = 0.0F;
        float gradientFirst = 0.0F;
        float gradientSecond = 0.0F;
        float gradientNorm = 0.0F;
    };
    std::vector<PeakResponseObservation> responseObservations;
    std::vector<uint32_t> evidenceIndices;
    std::vector<PeakEvidenceObservation> evidenceObservations;
    responseObservations.reserve(observations.size());
    evidenceIndices.reserve(observations.size());
    evidenceObservations.reserve(observations.size() / 2);
    if (profile != nullptr) {
        ++profile->peakComponents;
        profile->peakPreparationObservationVisits += observations.size();
    }
    for (size_t observationIndex = 0;
         observationIndex < observations.size(); ++observationIndex) {
        const auto& observation = observations[observationIndex];
        const cv::Vec3f position = observationPosition(observation);
        if (!finiteVector(position))
            continue;
        const cv::Vec3f pivotOffset = position - pivot;
        const float axial = pivotOffset.dot(axis);
        if (std::abs(axial) > axialCutoff)
            continue;
        const cv::Vec3f transverse = pivotOffset - axis * axial;
        if (transverse.dot(transverse) > unionRadius * unionRadius)
            continue;
        const float first = transverse.dot(basis[0]);
        const float second = transverse.dot(basis[1]);

        cv::Vec3f direction;
        const bool usablePositive = usableDirectionObservation(
            observation, config, direction);
        const bool retainedForComponent =
            retainedInliers[observationIndex] &&
            assignments[observationIndex] == selectedComponent;
        float selectedAlignment = 0.0F;
        float signal = 0.0F;
        if (retainedForComponent && usablePositive) {
            const float dot = direction.dot(axis);
            selectedAlignment = dot * dot;
            signal = observationPresence(observation) * selectedAlignment;
        }
        PeakResponseObservation responseObservation;
        responseObservation.first = static_cast<float>(first);
        responseObservation.second = static_cast<float>(second);
        responseObservation.axialGaussian = static_cast<float>(std::exp(
            -axial * axial * invTwoAxialSigma2));
        responseObservation.signal = static_cast<float>(signal);
        uint32_t evidenceIndex = kNoPeakEvidence;
        const float alignment = static_cast<float>(selectedAlignment);
        if (alignment > 0.0F) {
            if (evidenceObservations.size() >= kNoPeakEvidence) {
                throw std::overflow_error(
                    "peak evidence observation index exceeds uint32 range");
            }
            evidenceIndex = static_cast<uint32_t>(evidenceObservations.size());
            PeakEvidenceObservation evidenceObservation;
            evidenceObservation.directionAlignmentSquared = alignment;
            if (observationGradientValid(observations, observationIndex) &&
                finiteVector(observation.presenceGradientPredictionXYZ)) {
                const cv::Vec3f gradient = observationGradient(observation);
                evidenceObservation.gradientFirst =
                    static_cast<float>(gradient.dot(basis[0]));
                evidenceObservation.gradientSecond =
                    static_cast<float>(gradient.dot(basis[1]));
                const float gradientNorm2 =
                    evidenceObservation.gradientFirst *
                        evidenceObservation.gradientFirst +
                    evidenceObservation.gradientSecond *
                        evidenceObservation.gradientSecond;
                if (gradientNorm2 > kMatrixEpsilon) {
                    evidenceObservation.gradientNorm =
                        static_cast<float>(std::sqrt(gradientNorm2));
                }
            }
            evidenceObservations.push_back(evidenceObservation);
        }
        responseObservations.push_back(responseObservation);
        evidenceIndices.push_back(evidenceIndex);
    }
    if (profile != nullptr) {
        profile->peakPreparedResponseObservations += responseObservations.size();
        profile->peakPreparedEvidenceObservations += evidenceObservations.size();
        profile->peakResponseObservationRecordBytes = std::max(
            profile->peakResponseObservationRecordBytes,
            sizeof(PeakResponseObservation));
        profile->peakEvidenceIndexRecordBytes = std::max(
            profile->peakEvidenceIndexRecordBytes, sizeof(uint32_t));
        profile->peakEvidenceObservationRecordBytes = std::max(
            profile->peakEvidenceObservationRecordBytes,
            sizeof(PeakEvidenceObservation));
        profile->peakMaximumObservationStorageBytes = std::max(
            profile->peakMaximumObservationStorageBytes,
            responseObservations.capacity() * sizeof(PeakResponseObservation) +
                evidenceIndices.capacity() * sizeof(uint32_t) +
                evidenceObservations.capacity() *
                    sizeof(PeakEvidenceObservation));
    }

    const auto responseAt = [&](float candidateFirst, float candidateSecond, bool acceptance) {
        if (profile != nullptr) {
            if (acceptance)
                ++profile->peakAcceptanceResponses;
            else
                ++profile->peakComputedGridResponses;
            profile->peakResponseObservationVisits += responseObservations.size();
        }
        size_t responseEvidenceVisits = 0;
        FloatSum numerator;
        FloatSum denominator;
        FloatSum eligibleGradientWeight;
        FloatSum validGradientWeight;
        FloatSum inward;
        FloatSum outward;
        for (size_t observationIndex = 0;
             observationIndex < responseObservations.size();
             ++observationIndex) {
            const auto& observation = responseObservations[observationIndex];
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
            const uint32_t evidenceIndex = evidenceIndices[observationIndex];
            if (evidenceIndex == kNoPeakEvidence)
                continue;
            const auto& evidence =
                evidenceObservations[evidenceIndex];
            ++responseEvidenceVisits;
            const float eligibleWeight =
                gaussian * evidence.directionAlignmentSquared;
            eligibleGradientWeight.add(eligibleWeight);
            if (!(evidence.gradientNorm > 0.0F)) {
                continue;
            }
            if (!(distanceSquared > kMatrixEpsilon)) {
                continue;
            }
            validGradientWeight.add(eligibleWeight);
            const float cosine = std::clamp(
                (evidence.gradientFirst * radialFirst +
                 evidence.gradientSecond * radialSecond) /
                    (evidence.gradientNorm * std::sqrt(distanceSquared)),
                -1.0F, 1.0F);
            const float vote = eligibleWeight * evidence.gradientNorm *
                static_cast<float>(config.peakSigmaPredictionVoxels) *
                cosine * cosine;
            if (cosine > 0.0F)
                inward.add(vote);
            else if (cosine < 0.0F)
                outward.add(vote);
        }
        if (profile != nullptr) {
            profile->peakResponseEvidenceObservationVisits +=
                responseEvidenceVisits;
        }
        const float presenceResponse = denominator.sum > 0.0F
            ? numerator.sum / denominator.sum
            : 0.0F;
        if (!(config.peakGradientWeight > 0.0F) || !(eligibleGradientWeight.sum > 0.0F) || !(validGradientWeight.sum > 0.0F)) {
            return presenceResponse;
        }
        const float voteMass = inward.sum + outward.sum;
        if (!(voteMass > 0.0F))
            return presenceResponse;
        const float coverage = std::clamp(validGradientWeight.sum / eligibleGradientWeight.sum, 0.0F, 1.0F);
        const float radialGradient = voteMass / validGradientWeight.sum;
        const float reliability = coverage * radialGradient / (radialGradient + config.peakGradientReliabilityScale);
        const float signedVote = (inward.sum - outward.sum) / voteMass;
        return presenceResponse + config.peakGradientWeight * reliability * signedVote;
    };

    using GridIndex = std::pair<int, int>;
    const int extent = static_cast<int>(std::floor(config.localWindowRadiusPredictionVoxels / config.peakGridStepPredictionVoxels));
    const detail::FiberAnchorPeakGridLayout peakGrid{extent};
    std::vector<cv::Vec3f> gridPoints(peakGrid.size());
    std::vector<uint8_t> feasibleGrid(peakGrid.size(), 0);
    for (int first = -extent; first <= extent; ++first) {
        for (int second = -extent; second <= extent; ++second) {
            const size_t slot = peakGrid.indexUnchecked(first, second);
            const cv::Vec3f point =
                pivot + basis[0] *
                    (static_cast<float>(first) *
                     config.peakGridStepPredictionVoxels) +
                basis[1] *
                    (static_cast<float>(second) *
                     config.peakGridStepPredictionVoxels);
            gridPoints[slot] = point;
            feasibleGrid[slot] = static_cast<uint8_t>(insidePeakDomain(
                point, pivot, owner,
                config.localWindowRadiusPredictionVoxels));
        }
    }
    const auto pointAt = [&](const GridIndex& index) -> const cv::Vec3f& {
        return gridPoints[peakGrid.indexUnchecked(index.first, index.second)];
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
        return peakGrid.contains(index.first, index.second) &&
            feasibleGrid[peakGrid.indexUnchecked(index.first, index.second)];
    };

    detail::FiberAnchorPeakResponseCache responseCache{peakGrid};
    const auto response = [&](const GridIndex& index) {
        if (profile != nullptr)
            ++profile->peakGridResponseRequests;
        return responseCache.getOrCompute(
            index.first, index.second,
            [&] { return responseAtIndex(index, false); });
    };

    GridIndex current{0, 0};
    float nearestDistance = std::numeric_limits<float>::infinity();
    for (int first = -extent; first <= extent; ++first) {
        for (int second = -extent; second <= extent; ++second) {
            const GridIndex candidate{first, second};
            if (!feasible(candidate))
                continue;
            const cv::Vec3f delta =
                pointAt(candidate) - components[selectedComponent].position;
            const float distance = delta.dot(delta);
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
        float bestResponse = response(current);
        for (const auto& offset : neighbors) {
            const GridIndex candidate{
                current.first + offset.first,
                current.second + offset.second,
            };
            if (!feasible(candidate))
                continue;
            const float candidateResponse = response(candidate);
            if (candidateResponse > bestResponse ||
                (candidateResponse == bestResponse && candidate < best)) {
                best = candidate;
                bestResponse = candidateResponse;
            }
        }
        const float tolerance = kFloatComparisonEpsilon *
            std::max(1.0F, std::abs(response(current)));
        if (bestResponse <= response(current) + tolerance)
            break;
        current = best;
    }

    const cv::Vec3f discrete = pointAt(current);
    const float centerResponse = response(current);
    const float tolerance =
        kFloatComparisonEpsilon * std::max(1.0F, std::abs(centerResponse));
    const auto acceptedPosition = [&](const cv::Vec3f& candidate) {
        const cv::Vec3f offset = candidate - pivot;
        return insidePeakDomain(
                   candidate, pivot, owner,
                   config.localWindowRadiusPredictionVoxels) &&
            responseAt(
                static_cast<float>(offset.dot(basis[0])),
                static_cast<float>(offset.dot(basis[1])), true) +
                tolerance >= centerResponse;
    };

    std::array<float, 2> separableOffsetGridSteps{0.0F, 0.0F};
    for (int dimension = 0; dimension < 2; ++dimension) {
        GridIndex lower = current;
        GridIndex upper = current;
        (dimension == 0 ? lower.first : lower.second) -= 1;
        (dimension == 0 ? upper.first : upper.second) += 1;
        if (!feasible(lower) || !feasible(upper))
            continue;
        const float lowerResponse = response(lower);
        const float upperResponse = response(upper);
        const float curvature =
            lowerResponse - 2.0F * centerResponse + upperResponse;
        if (!(curvature < 0.0F) || !std::isfinite(curvature))
            continue;
        const float offset =
            0.5F * (lowerResponse - upperResponse) / curvature;
        if (std::isfinite(offset)) {
            separableOffsetGridSteps[static_cast<size_t>(dimension)] =
                std::clamp(offset, -0.5F, 0.5F);
        }
    }
    const cv::Vec3f separableCandidate =
        discrete + basis[0] *
                (separableOffsetGridSteps[0] *
                 config.peakGridStepPredictionVoxels) +
        basis[1] *
                (separableOffsetGridSteps[1] *
                 config.peakGridStepPredictionVoxels);
    const cv::Vec3f separable = acceptedPosition(separableCandidate)
        ? separableCandidate
        : discrete;

    std::array<std::array<float, 3>, 3> neighborhood{};
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
            const float value = response(sample);
            if (!std::isfinite(value)) {
                completeNeighborhood = false;
                continue;
            }
            neighborhood[static_cast<size_t>(first + 1)]
                        [static_cast<size_t>(second + 1)] = value;
        }
    }
    cv::Vec3f joint = discrete;
    if (completeNeighborhood) {
        const auto offset = fitFiberAnchorQuadraticPeak(neighborhood);
        if (offset.has_value()) {
            const cv::Vec3f candidate =
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
    std::array<cv::Vec3f, 2> axes,
    const FiberAnchorConfig& config,
    FiberAnchorFitProfile* profile)
{
    if (profile != nullptr)
        ++profile->seedPairs;
    FitState best;
    best.objectiveNumerator = -1.0F;
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
        std::array<cv::Vec3f, 2> updated = axes;
        for (uint8_t component = 0; component < 2; ++component) {
            const auto principal = principalFiberAxisF(
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
        if (best.objectiveNumerator < 0.0F || betterState(state, best))
            best = state;

        const float update = std::max(
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
    total.ownedDiscoveryObservationVisits +=
        value.ownedDiscoveryObservationVisits;
    total.ownedInitializationObservationVisits +=
        value.ownedInitializationObservationVisits;
    total.avoidedOwnedSupportObservationVisits +=
        value.avoidedOwnedSupportObservationVisits;
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
    total.peakPreparedResponseObservations +=
        value.peakPreparedResponseObservations;
    total.peakPreparedEvidenceObservations +=
        value.peakPreparedEvidenceObservations;
    total.peakResponseObservationRecordBytes = std::max(
        total.peakResponseObservationRecordBytes,
        value.peakResponseObservationRecordBytes);
    total.peakEvidenceIndexRecordBytes = std::max(
        total.peakEvidenceIndexRecordBytes,
        value.peakEvidenceIndexRecordBytes);
    total.peakEvidenceObservationRecordBytes = std::max(
        total.peakEvidenceObservationRecordBytes,
        value.peakEvidenceObservationRecordBytes);
    total.peakMaximumObservationStorageBytes = std::max(
        total.peakMaximumObservationStorageBytes,
        value.peakMaximumObservationStorageBytes);
    total.peakGridResponseRequests += value.peakGridResponseRequests;
    total.peakComputedGridResponses += value.peakComputedGridResponses;
    total.peakAcceptanceResponses += value.peakAcceptanceResponses;
    total.peakResponseObservationVisits +=
        value.peakResponseObservationVisits;
    total.peakResponseEvidenceObservationVisits +=
        value.peakResponseEvidenceObservationVisits;
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
    float dot = std::clamp(left.axisXYZ.dot(right.axisXYZ), -1.0F, 1.0F);
    const float axialDot = std::abs(dot);
    const float minimumDot = std::cos(
        config.nmsMaximumAngleDegrees * std::acos(-1.0F) / 180.0F);
    if (axialDot < minimumDot)
        return false;
    cv::Vec3f alignedRight = right.axisXYZ;
    if (dot < 0.0F)
        alignedRight *= -1.0F;
    cv::Vec3f averageAxis = normalized(left.axisXYZ + alignedRight);
    if (averageAxis.dot(averageAxis) <= kMatrixEpsilon)
        averageAxis = left.axisXYZ;
    const cv::Vec3f delta = right.positionPredictionXYZ -
        left.positionPredictionXYZ;
    const float longitudinal = std::abs(delta.dot(averageAxis));
    const cv::Vec3f transverseVector =
        delta - averageAxis * delta.dot(averageAxis);
    const float transverse = std::sqrt(std::max(
        0.0F, transverseVector.dot(transverseVector)));
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
    const float binSize = std::max(
        kGeometryEpsilon,
        std::hypot(
            config.nmsTransverseRadiusPredictionVoxels,
            config.nmsLongitudinalRadiusPredictionVoxels));
    using Bin = std::array<int64_t, 3>;
    std::map<Bin, std::vector<size_t>> bins;
    const auto binFor = [binSize](const cv::Vec3f& position) {
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
    float maximumTrimMassFraction,
    float madMultiplier,
    float minimumAngleDegrees)
{
    if (!(maximumTrimMassFraction >= 0.0F) ||
        maximumTrimMassFraction > 0.20F ||
        !std::isfinite(maximumTrimMassFraction) ||
        !(madMultiplier >= 0.0F) || !std::isfinite(madMultiplier) ||
        !(minimumAngleDegrees >= 0.0F) || minimumAngleDegrees > 90.0F ||
        !std::isfinite(minimumAngleDegrees)) {
        throw std::invalid_argument("invalid fiber anchor robust cutoff parameters");
    }
    FiberAnchorRobustCutoff result;
    std::array<float, kRobustHistogramBins> histogram{};
    for (const auto& sample : samples) {
        if (!std::isfinite(sample.residual) || !std::isfinite(sample.mass) ||
            !(sample.mass > 0.0F)) {
            continue;
        }
        histogram[robustHistogramBin(sample.residual)] += sample.mass;
        result.totalMass += sample.mass;
    }
    result.retainedMass = result.totalMass;
    if (!(result.totalMass > 0.0F) || !(maximumTrimMassFraction > 0.0F))
        return result;

    const size_t medianBin = weightedHistogramQuantileBin(
        histogram, result.totalMass, 0.5F);
    const float median = robustHistogramCenter(medianBin);
    std::array<float, kRobustHistogramBins> deviationHistogram{};
    for (const auto& sample : samples) {
        if (!std::isfinite(sample.residual) || !std::isfinite(sample.mass) ||
            !(sample.mass > 0.0F)) {
            continue;
        }
        deviationHistogram[robustHistogramBin(std::abs(
            std::clamp(sample.residual, 0.0F, 1.0F) - median))] += sample.mass;
    }
    return selectRobustHistogramCutoff(
        histogram, deviationHistogram, result.totalMass, median,
        maximumTrimMassFraction, madMultiplier, minimumAngleDegrees).summary;
}

std::vector<float> fiberAnchorSpatialBacktrackingFractions(
    float maximumDisplacementPredictionVoxels,
    float targetStepPredictionVoxels,
    int maximumHalvings)
{
    if (!(maximumDisplacementPredictionVoxels >= 0.0F) ||
        !std::isfinite(maximumDisplacementPredictionVoxels) ||
        !(targetStepPredictionVoxels > 0.0F) ||
        !std::isfinite(targetStepPredictionVoxels) ||
        maximumHalvings < 0 || maximumHalvings > 8) {
        throw std::invalid_argument("invalid fiber anchor spatial backtracking parameters");
    }
    std::vector<float> fractions;
    fractions.reserve(static_cast<size_t>(maximumHalvings + 1));
    for (int depth = 0; depth <= maximumHalvings; ++depth) {
        const float fraction = std::ldexp(1.0F, -depth);
        fractions.push_back(fraction);
        if (maximumDisplacementPredictionVoxels * fraction <=
            targetStepPredictionVoxels) {
            break;
        }
    }
    return fractions;
}

std::optional<FiberAnchorQuadraticPeakOffset> fitFiberAnchorQuadraticPeak(
    const std::array<std::array<float, 3>, 3>& response)
{
    FloatSum sum;
    FloatSum firstMoment;
    FloatSum secondMoment;
    FloatSum firstSquaredMoment;
    FloatSum mixedMoment;
    FloatSum secondSquaredMoment;
    float responseScale = 1.0F;
    for (int first = -1; first <= 1; ++first) {
        for (int second = -1; second <= 1; ++second) {
            const float value = response[static_cast<size_t>(first + 1)]
                                         [static_cast<size_t>(second + 1)];
            if (!std::isfinite(value))
                return std::nullopt;
            responseScale = std::max(responseScale, std::abs(value));
            sum.add(value);
            firstMoment.add(static_cast<float>(first) * value);
            secondMoment.add(static_cast<float>(second) * value);
            firstSquaredMoment.add(
                static_cast<float>(first * first) * value);
            mixedMoment.add(static_cast<float>(first * second) * value);
            secondSquaredMoment.add(
                static_cast<float>(second * second) * value);
        }
    }

    // Least-squares model a + bx + cy + dx^2 + exy + fy^2 on {-1,0,1}^2.
    const float gradientFirst = firstMoment.sum / 6.0F;
    const float gradientSecond = secondMoment.sum / 6.0F;
    const float hessianFirstFirst =
        firstSquaredMoment.sum - 2.0F * sum.sum / 3.0F;
    const float hessianFirstSecond = mixedMoment.sum / 4.0F;
    const float hessianSecondSecond =
        secondSquaredMoment.sum - 2.0F * sum.sum / 3.0F;
    const std::array<float, 5> coefficients{
        gradientFirst,
        gradientSecond,
        hessianFirstFirst,
        hessianFirstSecond,
        hessianSecondSecond,
    };
    if (std::any_of(coefficients.begin(), coefficients.end(),
                    [](float value) { return !std::isfinite(value); })) {
        return std::nullopt;
    }

    constexpr float kRelativeCurvatureTolerance = 1.0e-6F;
    const float curvatureTolerance =
        kRelativeCurvatureTolerance * responseScale;
    const float trace = hessianFirstFirst + hessianSecondSecond;
    const float discriminant = std::hypot(
        hessianFirstFirst - hessianSecondSecond,
        2.0F * hessianFirstSecond);
    const float largestEigenvalue = 0.5F * (trace + discriminant);
    const float determinant =
        hessianFirstFirst * hessianSecondSecond -
        hessianFirstSecond * hessianFirstSecond;
    if (!(largestEigenvalue < -curvatureTolerance) ||
        !(determinant > curvatureTolerance * curvatureTolerance) ||
        !std::isfinite(determinant)) {
        return std::nullopt;
    }

    const float first =
        (hessianFirstSecond * gradientSecond -
         hessianSecondSecond * gradientFirst) / determinant;
    const float second =
        (hessianFirstSecond * gradientFirst -
         hessianFirstFirst * gradientSecond) / determinant;
    if (!std::isfinite(first) || !std::isfinite(second) ||
        std::abs(first) > 0.5F || std::abs(second) > 0.5F) {
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
    if (!(config.gaussianSigmaPredictionVoxels > 0.0F) ||
        !std::isfinite(config.gaussianSigmaPredictionVoxels)) {
        throw std::invalid_argument("fiber anchor Gaussian sigma must be positive and finite");
    }
    if (!(config.peakSigmaPredictionVoxels > 0.0F) ||
        !std::isfinite(config.peakSigmaPredictionVoxels)) {
        throw std::invalid_argument("fiber anchor peak sigma must be positive and finite");
    }
    if (!(config.peakGradientWeight >= 0.0F) ||
        !std::isfinite(config.peakGradientWeight)) {
        throw std::invalid_argument(
            "fiber anchor peak gradient weight must be nonnegative and finite");
    }
    if (!(config.peakGradientReliabilityScale > 0.0F) ||
        !std::isfinite(config.peakGradientReliabilityScale)) {
        throw std::invalid_argument(
            "fiber anchor peak gradient reliability scale must be positive and finite");
    }
    if (!(config.peakAxialSigmaPredictionVoxels > 0.0F) ||
        !std::isfinite(config.peakAxialSigmaPredictionVoxels)) {
        throw std::invalid_argument(
            "fiber anchor peak axial sigma must be positive and finite");
    }
    if (!(config.peakGridStepPredictionVoxels > 0.0F) ||
        !std::isfinite(config.peakGridStepPredictionVoxels)) {
        throw std::invalid_argument("fiber anchor peak grid step must be positive and finite");
    }
    if (!(config.gaussianCutoffSigmas > 0.0F) ||
        !std::isfinite(config.gaussianCutoffSigmas)) {
        throw std::invalid_argument("fiber anchor Gaussian cutoff must be positive and finite");
    }
    if (!(config.localWindowRadiusPredictionVoxels > 0.0F) ||
        !std::isfinite(config.localWindowRadiusPredictionVoxels)) {
        throw std::invalid_argument("fiber anchor local window must be positive and finite");
    }
    if (config.localWindowRadiusPredictionVoxels /
            config.peakGridStepPredictionVoxels >
        128.0F) {
        throw std::invalid_argument(
            "fiber anchor peak grid radius must not exceed 128 steps");
    }
    if (!(config.axialSupportHalfWidthPredictionVoxels > 0.0F) ||
        !std::isfinite(config.axialSupportHalfWidthPredictionVoxels)) {
        throw std::invalid_argument("fiber anchor axial support must be positive and finite");
    }
    if (!(config.positionConvergenceTolerancePredictionVoxels >= 0.0F) ||
        !std::isfinite(config.positionConvergenceTolerancePredictionVoxels)) {
        throw std::invalid_argument("fiber anchor position tolerance must be finite and non-negative");
    }
    if (!(config.nmsMaximumAngleDegrees >= 0.0F) ||
        !(config.nmsMaximumAngleDegrees <= 90.0F) ||
        !std::isfinite(config.nmsMaximumAngleDegrees)) {
        throw std::invalid_argument("fiber anchor NMS angle must be in [0, 90]");
    }
    if (!(config.nmsTransverseRadiusPredictionVoxels >= 0.0F) ||
        !std::isfinite(config.nmsTransverseRadiusPredictionVoxels)) {
        throw std::invalid_argument("fiber anchor NMS transverse radius must be finite and non-negative");
    }
    if (!(config.nmsLongitudinalRadiusPredictionVoxels >= 0.0F) ||
        !std::isfinite(config.nmsLongitudinalRadiusPredictionVoxels)) {
        throw std::invalid_argument("fiber anchor NMS longitudinal radius must be finite and non-negative");
    }
    if (!(config.observationPresenceFloor >= 0.0F) ||
        !(config.observationPresenceFloor <= 1.0F) ||
        !std::isfinite(config.observationPresenceFloor)) {
        throw std::invalid_argument("fiber anchor observation presence floor must be in [0, 1]");
    }
    if (!(config.minimumAlignedSupport >= 0.0F) ||
        !(config.minimumAlignedSupport <= 1.0F) ||
        !std::isfinite(config.minimumAlignedSupport)) {
        throw std::invalid_argument("fiber anchor minimum aligned support must be in [0, 1]");
    }
    if (!(config.robustMaximumTrimMassFraction >= 0.0F) ||
        config.robustMaximumTrimMassFraction > 0.20F ||
        !std::isfinite(config.robustMaximumTrimMassFraction)) {
        throw std::invalid_argument(
            "fiber anchor robust maximum trim mass must be in [0, 0.20F]");
    }
    if (!(config.robustMadMultiplier >= 0.0F) ||
        !std::isfinite(config.robustMadMultiplier)) {
        throw std::invalid_argument(
            "fiber anchor robust MAD multiplier must be nonnegative and finite");
    }
    if (!(config.robustMinimumAngleDegrees >= 0.0F) ||
        config.robustMinimumAngleDegrees > 90.0F ||
        !std::isfinite(config.robustMinimumAngleDegrees)) {
        throw std::invalid_argument(
            "fiber anchor robust minimum angle must be in [0, 90]");
    }
    if (!(config.mergeMaximumAngleDegrees >= 0.0F) ||
        !(config.mergeMaximumAngleDegrees <= 90.0F) ||
        !std::isfinite(config.mergeMaximumAngleDegrees)) {
        throw std::invalid_argument("fiber anchor merge maximum angle must be in [0, 90]");
    }
    if (!(config.mergeMaximumAbsoluteObjectiveLoss >= 0.0F) ||
        !(config.mergeMaximumAbsoluteObjectiveLoss <= 1.0F) ||
        !std::isfinite(config.mergeMaximumAbsoluteObjectiveLoss) ||
        !(config.mergeMaximumRelativeObjectiveLoss >= 0.0F) ||
        !(config.mergeMaximumRelativeObjectiveLoss <= 1.0F) ||
        !std::isfinite(config.mergeMaximumRelativeObjectiveLoss)) {
        throw std::invalid_argument("fiber anchor merge objective losses must be in [0, 1]");
    }
    if (config.maximumSeedCount < 1 || config.maximumSeedCount > 64)
        throw std::invalid_argument("fiber anchor maximum seed count must be in [1, 64]");
    if (config.maximumIterations < 1)
        throw std::invalid_argument("fiber anchor maximum iterations must be positive");
    if (!(config.convergenceTolerance >= 0.0F) ||
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
    if (!(predictionToBaseScale > 0.0F) || !std::isfinite(predictionToBaseScale))
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

template <typename ObservationRange, typename OwnedObservationRange>
FiberCellAnchorResult fitFiberCellAnchorsImpl(
    const std::array<size_t, 3>& cellZYX,
    const std::array<size_t, 3>& cellBeginZYX,
    const std::array<size_t, 3>& cellEndZYX,
    const ObservationRange& input,
    const OwnedObservationRange& ownedInput,
    bool validateOwnedCoverage,
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
    const auto isOwned = [&cellBeginZYX, &cellEndZYX](const cv::Vec3f& position) {
        return position[0] >= static_cast<float>(cellBeginZYX[2]) &&
            position[0] < static_cast<float>(cellEndZYX[2]) &&
            position[1] >= static_cast<float>(cellBeginZYX[1]) &&
            position[1] < static_cast<float>(cellEndZYX[1]) &&
            position[2] >= static_cast<float>(cellBeginZYX[0]) &&
            position[2] < static_cast<float>(cellEndZYX[0]);
    };
    if (validateOwnedCoverage) {
        size_t ownedCount = 0;
        for (size_t index = 0; index < input.size(); ++index) {
            const cv::Vec3f position = observationPosition(input[index]);
            ownedCount += static_cast<size_t>(
                finiteVector(position) && isOwned(position));
        }
        if (profile != nullptr)
            profile->ownedDiscoveryObservationVisits += input.size();
        if (ownedCount != expected)
            throw std::invalid_argument("fiber anchor observations do not cover the owned cell voxels exactly once");
    } else {
        if (ownedInput.size() != expected)
            throw std::logic_error("fiber anchor direct owned range has the wrong size");
        if (profile != nullptr) {
            profile->avoidedOwnedSupportObservationVisits +=
                2 * input.size() - ownedInput.size();
        }
    }

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
    const cv::Vec3f center{
        (static_cast<float>(cellBeginZYX[2]) +
            static_cast<float>(cellEndZYX[2]) - 1.0F) * 0.5F,
        (static_cast<float>(cellBeginZYX[1]) +
            static_cast<float>(cellEndZYX[1]) - 1.0F) * 0.5F,
        (static_cast<float>(cellBeginZYX[0]) +
            static_cast<float>(cellEndZYX[0]) - 1.0F) * 0.5F,
    };
    const float invTwoSigma2 = 1.0F /
        (2.0F * config.gaussianSigmaPredictionVoxels *
         config.gaussianSigmaPredictionVoxels);
    FloatSum denominator;
    std::vector<WeightedObservation> observations;
    observations.reserve(ownedInput.size());
    visitObservations(ownedInput, [&](const auto& candidate, size_t index) {
        const cv::Vec3f position = observationPosition(candidate);
        if (validateOwnedCoverage && (!finiteVector(position) || !isOwned(position)))
            return;
        const cv::Vec3f delta = position - center;
        const float gaussian = std::exp(-delta.dot(delta) * invTwoSigma2);
        denominator.add(gaussian);
        cv::Vec3f direction;
        if (!usableDirectionObservation(candidate, config, direction))
            return;
        observations.push_back({
            position,
            direction,
            gaussian,
            gaussian * observationPresence(candidate),
            index,
        });
    });
    if (profile != nullptr) {
        profile->ownedInitializationObservationVisits += ownedInput.size();
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

    const FiberPrincipalAxisF global = principalFiberAxisF(weightedTensor(observations, nullptr, 0));
    std::vector<cv::Vec3f> seeds;
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
        seeds.push_back(canonicalFiberAxisF(observations[best].direction));
    }
    std::vector<bool> selected(observations.size(), false);
    while (seeds.size() < config.maximumSeedCount) {
        if (profile != nullptr)
            profile->seedGenerationObservationVisits += observations.size();
        size_t best = observations.size();
        float bestScore = 0.0F;
        for (size_t index = 0; index < observations.size(); ++index) {
            if (selected[index])
                continue;
            float minimumDissimilarity = 1.0F;
            for (const auto& seed : seeds) {
                const float dot = observations[index].direction.dot(seed);
                minimumDissimilarity = std::min(
                    minimumDissimilarity, std::max(0.0F, 1.0F - dot * dot));
            }
            const float score = observations[index].weight * minimumDissimilarity;
            if (score > bestScore) {
                bestScore = score;
                best = index;
            }
        }
        if (best == observations.size() || !(bestScore > 0.0F))
            break;
        selected[best] = true;
        seeds.push_back(canonicalFiberAxisF(observations[best].direction));
    }
    if (profile != nullptr) {
        profile->seeds += seeds.size();
        profile->seedGenerationWorkSeconds += std::chrono::duration<double>(
            FitClock::now() - phaseStart).count();
        phaseStart = FitClock::now();
    }

    FitState bestFit;
    bestFit.objectiveNumerator = -1.0F;
    if (seeds.size() == 1) {
        bestFit = refineSeedPair(
            observations, {seeds[0], seeds[0]}, config, profile);
    } else {
        for (size_t first = 0; first + 1 < seeds.size(); ++first) {
            for (size_t second = first + 1; second < seeds.size(); ++second) {
                const FitState fit = refineSeedPair(
                    observations, {seeds[first], seeds[second]}, config,
                    profile);
                if (bestFit.objectiveNumerator < 0.0F || betterState(fit, bestFit))
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
    const float seededObjective = denominator.sum > 0.0F
        ? bestFit.objectiveNumerator / denominator.sum
        : 0.0F;

    const std::array<FiberPrincipalAxisF, 2> fittedComponents{
        principalFiberAxisF(weightedTensor(observations, &bestFit.assignments, 0)),
        principalFiberAxisF(weightedTensor(observations, &bestFit.assignments, 1)),
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
            diagnostic.metrics.objectiveContribution = denominator.sum > 0.0F
                ? fitted.largestEigenvalue / denominator.sum
                : 0.0F;
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

    std::array<cv::Vec3f, 2> seedAxes{};
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
        const float numeratorValue = refined.evaluation.numerators[componentIndex];
        component.assignedObservationCount =
            refined.evaluation.assignedCounts[componentIndex];
        component.anchor.axisXYZ =
            canonicalFiberAxisF(refined.components[componentIndex].axis);
        component.anchor.positionPredictionXYZ =
            refined.components[componentIndex].position;
        component.anchor.alignedSupport =
            refined.evaluation.alignedSupports[componentIndex];
        component.anchor.directionalCoherence =
            refined.evaluation.directionalCoherences[componentIndex];
        component.anchor.refinementScore = component.anchor.alignedSupport;
        component.anchor.refinementIterations = refined.acceptedIterations;
        if (component.assignedObservationCount == 0 || !(numeratorValue > 0.0F)) {
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
        cellZYX, cellBeginZYX, cellEndZYX, input, input,
        /*validateOwnedCoverage=*/true, config, profile);
}

static FiberAnchorExtractionReport extractFiberAnchorsImpl(
    const FiberPredictionGridInfo& grid,
    const FiberAnchorConfig& config,
    const FiberStoredPredictionBatchSampler& sampler,
    std::optional<FiberAnchorCrop> crop,
    std::vector<std::array<size_t, 3>> explicitCells,
    const FiberAnchorRetainPredicate& retainPredicate,
    const FiberAnchorProgressCallback& progressCallback,
    bool refinedOnly = false,
    bool retainDiagnostics = true)
{
    validateFiberAnchorConfig(config);
    if (!sampler)
        throw std::invalid_argument("fiber anchor extraction requires a prediction sampler");
    if (!(grid.predictionToBaseScale > 0.0F) ||
        !std::isfinite(grid.predictionToBaseScale) ||
        !detail::floatGridShapeExactlyRepresentable(grid.shapeZYX)) {
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
    const float broadTransverseSupport =
        config.localWindowRadiusPredictionVoxels +
        config.gaussianCutoffSigmas * config.gaussianSigmaPredictionVoxels;
    const float broadSupportRadius = std::hypot(
        broadTransverseSupport,
        config.axialSupportHalfWidthPredictionVoxels);
    const float peakTransverseSupport =
        config.localWindowRadiusPredictionVoxels +
        config.gaussianCutoffSigmas * config.peakSigmaPredictionVoxels;
    const float peakSupportRadius = std::hypot(
        peakTransverseSupport,
        config.gaussianCutoffSigmas *
            config.peakAxialSigmaPredictionVoxels);
    const float maximumSupportRadius =
        std::max(broadSupportRadius, peakSupportRadius);
    const size_t sampleHalo =
        static_cast<size_t>(std::ceil(maximumSupportRadius)) +
        (config.peakGradientWeight > 0.0F ? 1 : 0);
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
        const float nmsDistance = std::hypot(
            config.nmsTransverseRadiusPredictionVoxels,
            config.nmsLongitudinalRadiusPredictionVoxels);
        const float pivotReach =
            2.0F * config.localWindowRadiusPredictionVoxels + nmsDistance;
        const size_t cellRadius = static_cast<size_t>(std::ceil(
            pivotReach / static_cast<float>(cellSize))) + 1;
        const auto cellPivot = [&](const CellIndex& cell) {
            cv::Vec3f pivot;
            for (size_t axis = 0; axis < 3; ++axis) {
                const size_t begin = cell[axis] * cellSize;
                const size_t end = std::min(
                    begin + cellSize, grid.shapeZYX[axis]);
                pivot[static_cast<int>(2 - axis)] =
                    (static_cast<float>(begin) +
                     static_cast<float>(end) - 1.0F) * 0.5F;
            }
            return pivot;
        };
        for (const auto& selected : selectedCells) {
            const cv::Vec3f selectedPivot = cellPivot(selected);
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
                        const cv::Vec3f delta =
                            cellPivot(candidate) - selectedPivot;
                        if (delta.dot(delta) <=
                            pivotReach * pivotReach + kGeometryEpsilon) {
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
            progressCallback({phase, 0, requestedCells.size(), 0.0F});
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
                sizeof(FloatStoredPredictionSample) +
                sizeof(CompactFiberAnchorObservation) +
                (config.peakGradientWeight > 0.0F
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
            const size_t zBegin = std::min_element(
                tiles.begin(), tiles.end(), [](const auto& left, const auto& right) {
                    return left.sampleBegin[0] < right.sampleBegin[0];
                })->sampleBegin[0];
            const size_t zEnd = std::max_element(
                tiles.begin(), tiles.end(), [](const auto& left, const auto& right) {
                    return left.sampleEnd[0] < right.sampleEnd[0];
                })->sampleEnd[0];
            struct RowInterval {
                size_t y = 0;
                size_t xBegin = 0;
                size_t xEnd = 0;
            };
            std::vector<size_t> unionByZ(zEnd - zBegin, 0);
            std::vector<std::exception_ptr> errors(zEnd - zBegin);
            std::atomic<size_t> nextZ{0};
            const size_t workerCount = std::min(
                static_cast<size_t>(config.parallelThreads), zEnd - zBegin);
            const auto worker = [&]() {
                while (true) {
                    const size_t localZ = nextZ.fetch_add(1);
                    if (localZ >= unionByZ.size())
                        return;
                    try {
                        const size_t z = zBegin + localZ;
                        std::vector<RowInterval> intervals;
                        for (const auto& tile : tiles) {
                            if (z < tile.sampleBegin[0] ||
                                z >= tile.sampleEnd[0]) {
                                continue;
                            }
                            for (size_t y = tile.sampleBegin[1];
                                 y < tile.sampleEnd[1]; ++y) {
                                intervals.push_back({
                                    y, tile.sampleBegin[2], tile.sampleEnd[2]});
                            }
                        }
                        std::sort(intervals.begin(), intervals.end(),
                            [](const auto& left, const auto& right) {
                                return std::tie(left.y, left.xBegin, left.xEnd) <
                                    std::tie(right.y, right.xBegin, right.xEnd);
                            });
                        size_t covered = 0;
                        size_t mergedCount = 0;
                        for (const auto& interval : intervals) {
                            if (mergedCount != 0) {
                                auto& previous = intervals[mergedCount - 1];
                                if (previous.y == interval.y &&
                                    interval.xBegin <= previous.xEnd) {
                                    previous.xEnd = std::max(
                                        previous.xEnd, interval.xEnd);
                                    continue;
                                }
                            }
                            intervals[mergedCount++] = interval;
                        }
                        for (size_t index = 0; index < mergedCount; ++index) {
                            covered = checkedAdd(
                                covered,
                                intervals[index].xEnd - intervals[index].xBegin,
                                "fiber anchor tile union area");
                        }
                        unionByZ[localZ] = covered;
                    } catch (...) {
                        errors[localZ] = std::current_exception();
                    }
                }
            };
            std::vector<std::thread> workers;
            workers.reserve(workerCount);
            for (size_t workerIndex = 0; workerIndex < workerCount; ++workerIndex)
                workers.emplace_back(worker);
            for (auto& thread : workers)
                thread.join();
            for (const auto& error : errors) {
                if (error)
                    std::rethrow_exception(error);
            }
            size_t unionVoxels = 0;
            for (const size_t covered : unionByZ) {
                unionVoxels = checkedAdd(
                    unionVoxels, covered,
                    "fiber anchor tile union volume");
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
        struct SharedSampleInterval {
            size_t z = 0;
            size_t y = 0;
            size_t xBegin = 0;
            size_t xEnd = 0;
            size_t sampleOffset = 0;
        };
        struct SharedSampleRow {
            size_t z = 0;
            size_t y = 0;
            size_t intervalBegin = 0;
            size_t intervalEnd = 0;
        };
        struct TilePartition {
            size_t tileBegin = 0;
            size_t tileEnd = 0;
            size_t sharedUpperBytes = 0;
            size_t maximumTileBytes = 0;
        };
        constexpr size_t samplingScratchBytesPerVoxel =
            sizeof(CellIndex) + sizeof(FiberStoredPredictionSample);
        constexpr size_t kMinimumSamplingBatchVoxels = 4096;
        const size_t sharedTargetBytes =
            config.maximumConcurrentSampleBytes / 2;
        std::vector<TilePartition> partitions;
        size_t tileOccurrences = 0;
        size_t partitionBegin = 0;
        size_t partitionSharedUpperBytes = 0;
        size_t partitionMaximumTileBytes = 0;
        size_t partitionSampleUpperCount = 0;
        for (size_t tileIndex = 0; tileIndex < tiles.size(); ++tileIndex) {
            const auto& tile = tiles[tileIndex];
            const size_t tileSamples = tileSampleCount(tile);
            tileOccurrences = checkedAdd(
                tileOccurrences, tileSamples,
                "fiber anchor tile sample occurrences");
            const size_t tileSampleBytes = checkedMultiply(
                tileSamples, sizeof(FloatStoredPredictionSample),
                "fiber anchor shared tile sample upper bound");
            const size_t tileRows = checkedMultiply(
                tile.sampleEnd[0] - tile.sampleBegin[0],
                tile.sampleEnd[1] - tile.sampleBegin[1],
                "fiber anchor shared tile rows");
            const size_t tileMetadataBytes = checkedMultiply(
                tileRows,
                sizeof(SharedSampleInterval) + sizeof(SharedSampleRow),
                "fiber anchor shared row metadata upper bound");
            const size_t tileSharedUpperBytes = checkedAdd(
                tileSampleBytes, tileMetadataBytes,
                "fiber anchor shared tile upper bound");
            const size_t candidateSharedUpperBytes = checkedAdd(
                partitionSharedUpperBytes, tileSharedUpperBytes,
                "fiber anchor partition shared upper bound");
            const size_t candidateMaximumTileBytes = std::max(
                partitionMaximumTileBytes, tile.estimatedBytes);
            const size_t candidateSampleUpperCount = checkedAdd(
                partitionSampleUpperCount, tileSamples,
                "fiber anchor partition sample upper count");
            const size_t minimumSamplingScratchBytes = checkedMultiply(
                std::min(
                    candidateSampleUpperCount,
                    kMinimumSamplingBatchVoxels),
                samplingScratchBytesPerVoxel,
                "fiber anchor minimum sampling scratch");
            const size_t minimumLiveBytes = checkedAdd(
                checkedAdd(
                    candidateSharedUpperBytes, candidateMaximumTileBytes,
                    "fiber anchor partition minimum live bytes"),
                minimumSamplingScratchBytes,
                "fiber anchor partition minimum live bytes");
            const bool exceedsTarget = tileIndex != partitionBegin &&
                candidateSharedUpperBytes > sharedTargetBytes;
            const bool exceedsBudget = minimumLiveBytes >
                config.maximumConcurrentSampleBytes;
            if (exceedsTarget || exceedsBudget) {
                partitions.push_back({
                    partitionBegin, tileIndex, partitionSharedUpperBytes,
                    partitionMaximumTileBytes});
                partitionBegin = tileIndex;
                partitionSharedUpperBytes = tileSharedUpperBytes;
                partitionMaximumTileBytes = tile.estimatedBytes;
                partitionSampleUpperCount = tileSamples;
                const size_t singleSamplingScratchBytes = checkedMultiply(
                    std::min(tileSamples, kMinimumSamplingBatchVoxels),
                    samplingScratchBytesPerVoxel,
                    "fiber anchor minimum sampling scratch");
                const size_t singleMinimumLiveBytes = checkedAdd(
                    checkedAdd(
                        partitionSharedUpperBytes, partitionMaximumTileBytes,
                        "fiber anchor partition minimum live bytes"),
                    singleSamplingScratchBytes,
                    "fiber anchor partition minimum live bytes");
                if (singleMinimumLiveBytes >
                    config.maximumConcurrentSampleBytes) {
                    throw std::runtime_error(
                        "fiber anchor cell sample exceeds the concurrent byte limit");
                }
            } else {
                partitionSharedUpperBytes = candidateSharedUpperBytes;
                partitionMaximumTileBytes = candidateMaximumTileBytes;
                partitionSampleUpperCount = candidateSampleUpperCount;
            }
        }
        partitions.push_back({
            partitionBegin, tiles.size(), partitionSharedUpperBytes,
            partitionMaximumTileBytes});
        report.profile.tiles += tiles.size();
        report.profile.samplingPartitions += partitions.size();
        report.profile.tilePlanningSeconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - tilePlanningStart).count();

        std::vector<std::unique_ptr<FiberCellAnchorResult>> jobResults(
            requestedCells.size());
        std::vector<std::exception_ptr> jobErrors(requestedCells.size());
        report.profile.cellResultHandleBytes += checkedAdd(
            checkedMultiply(
                jobResults.capacity(),
                sizeof(std::unique_ptr<FiberCellAnchorResult>),
                "fiber anchor result handles"),
            checkedMultiply(
                jobErrors.capacity(), sizeof(std::exception_ptr),
                "fiber anchor error handles"),
            "fiber anchor cell-result handles");
        struct WorkerProfile {
            size_t candidateObservations = 0;
            size_t retainedObservations = 0;
            size_t supportStencilCells = 0;
            size_t clippedSupportCells = 0;
            size_t gradientAttempts = 0;
            size_t validGradients = 0;
            size_t gradientComputations = 0;
            size_t validGradientComputations = 0;
            size_t fitIterations = 0;
            size_t retainPredicateCalls = 0;
            size_t emptyComponents = 0;
            size_t degenerateComponents = 0;
            size_t belowSupportComponents = 0;
            size_t mergedComponentPairs = 0;
            size_t outsideSelectionComponents = 0;
            double coordinateConstructionSeconds = 0.0F;
            double predictionSamplingSeconds = 0.0F;
            double tileSampleCopySeconds = 0.0F;
            double gradientConstructionSeconds = 0.0F;
            double observationConstructionSeconds = 0.0F;
            double fittingSeconds = 0.0F;
            std::vector<double> tilePreparationDurations;
            std::vector<double> cellProcessingDurations;
            FiberAnchorFitProfile fit;
        };
        std::vector<WorkerProfile> workerProfiles(
            static_cast<size_t>(config.parallelThreads));
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
            const cv::Vec3f pivot{
                (static_cast<float>(begin[2]) + static_cast<float>(end[2]) -
                 1.0F) *
                    0.5F,
                (static_cast<float>(begin[1]) + static_cast<float>(end[1]) -
                 1.0F) *
                    0.5F,
                (static_cast<float>(begin[0]) + static_cast<float>(end[0]) -
                 1.0F) *
                    0.5F,
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
                        if (config.peakGradientWeight > 0.0F) {
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
                            const cv::Vec3f position{
                                static_cast<float>(x),
                                static_cast<float>(y),
                                static_cast<float>(z),
                            };
                            const cv::Vec3f delta = position - pivot;
                            const bool owned =
                                position[0] >= static_cast<float>(begin[2]) &&
                                position[0] < static_cast<float>(end[2]) &&
                                position[1] >= static_cast<float>(begin[1]) &&
                                position[1] < static_cast<float>(end[1]) &&
                                position[2] >= static_cast<float>(begin[0]) &&
                                position[2] < static_cast<float>(end[0]);
                            if (owned ||
                                delta.dot(delta) <=
                                    maximumSupportRadius * maximumSupportRadius +
                                        kGeometryEpsilon) {
                                cellObservationIndices.push_back(
                                    static_cast<uint32_t>(index));
                                bool gradientValid = false;
                                if (config.peakGradientWeight > 0.0F) {
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
            const auto ownedRangeStart = std::chrono::steady_clock::now();
            const DenseOwnedCompactObservationRange ownedObservationRange{
                observations, tile.sampleBegin, sampleShape, begin, end};
            workerProfile.fit.setupWorkSeconds +=
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - ownedRangeStart).count();
            auto result = fitFiberCellAnchorsImpl(
                cellZYX, begin, end, observationRange, ownedObservationRange,
                /*validateOwnedCoverage=*/false, config,
                &workerProfile.fit);
            workerProfile.fittingSeconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - fittingStart).count();
            for (const auto& component : result.components)
                workerProfile.fitIterations += component.anchor.refinementIterations;
            return result;
        };

        const auto prepareSparseCellResult = [&]
            (FiberCellAnchorResult& cell,
             bool tallyCell,
             WorkerProfile& workerProfile) {
            if (tallyCell && retainPredicate) {
                for (auto& component : cell.components) {
                    if (!component.retained)
                        continue;
                    ++workerProfile.retainPredicateCalls;
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
                        ++workerProfile.outsideSelectionComponents;
                    }
                }
            }
            if (tallyCell) {
                for (auto& component : cell.components)
                    component.retainedAfterSelection = component.retained;
                if (cell.mergeEvaluation.has_value() &&
                    cell.mergeEvaluation->merged) {
                    ++workerProfile.mergedComponentPairs;
                }
                for (const auto& component : cell.components) {
                    if (component.rejectionReason == "empty")
                        ++workerProfile.emptyComponents;
                    else if (component.rejectionReason == "degenerate")
                        ++workerProfile.degenerateComponents;
                    else if (component.rejectionReason == "below_support")
                        ++workerProfile.belowSupportComponents;
                }
            }
            cell.initializedDiagnostics = {};
            for (auto& component : cell.components) {
                component.discretePeakPositionPredictionXYZ.reset();
                component.separablePeakPositionPredictionXYZ.reset();
                component.jointPeakPositionPredictionXYZ.reset();
                std::vector<size_t>().swap(component.diagnosticParentIds);
            }
            return cell.retainedAnchorCount > 0;
        };

        std::atomic<size_t> completedJobs{0};
        std::mutex progressMutex;
        auto lastProgressTime = phaseStart;
        std::exception_ptr progressError;
        const auto reportCellCompleted = [&]() {
            const size_t completed = completedJobs.fetch_add(1) + 1;
            if (!progressCallback)
                return;
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
                        std::chrono::duration<double>(now - phaseStart).count(),
                    });
                    lastProgressTime = now;
                } catch (...) {
                    progressError = std::current_exception();
                }
            }
        };
        std::vector<double> partitionDurations;
        const auto cellProcessingStart = std::chrono::steady_clock::now();
        const double cellProcessingCpuStart = processCpuSeconds();
        for (const auto& partition : partitions) {
            const auto partitionStart = std::chrono::steady_clock::now();
            const auto intervalPreparationStart =
                std::chrono::steady_clock::now();
            const double intervalPreparationCpuStart = processCpuSeconds();
            size_t partitionZBegin = grid.shapeZYX[0];
            size_t partitionZEnd = 0;
            for (size_t tileIndex = partition.tileBegin;
                 tileIndex < partition.tileEnd; ++tileIndex) {
                const auto& tile = tiles[tileIndex];
                partitionZBegin = std::min(
                    partitionZBegin, tile.sampleBegin[0]);
                partitionZEnd = std::max(
                    partitionZEnd, tile.sampleEnd[0]);
            }
            const size_t zExtent = partitionZEnd - partitionZBegin;
            std::vector<size_t> additions(zExtent + 1, 0);
            std::vector<size_t> removals(zExtent + 1, 0);
            for (size_t tileIndex = partition.tileBegin;
                 tileIndex < partition.tileEnd; ++tileIndex) {
                const auto& tile = tiles[tileIndex];
                const size_t yExtent = tile.sampleEnd[1] - tile.sampleBegin[1];
                const size_t begin = tile.sampleBegin[0] - partitionZBegin;
                const size_t end = tile.sampleEnd[0] - partitionZBegin;
                additions[begin] = checkedAdd(
                    additions[begin], yExtent,
                    "fiber anchor shared interval additions");
                removals[end] = checkedAdd(
                    removals[end], yExtent,
                    "fiber anchor shared interval removals");
            }
            std::vector<size_t> rawIntervalsByZ(zExtent, 0);
            std::vector<size_t> rawIntervalOffsets(zExtent + 1, 0);
            size_t activeIntervals = 0;
            for (size_t localZ = 0; localZ < zExtent; ++localZ) {
                if (activeIntervals < removals[localZ]) {
                    throw std::logic_error(
                        "fiber anchor shared interval activity underflows");
                }
                activeIntervals -= removals[localZ];
                activeIntervals = checkedAdd(
                    activeIntervals, additions[localZ],
                    "fiber anchor active shared intervals");
                rawIntervalsByZ[localZ] = activeIntervals;
                rawIntervalOffsets[localZ + 1] = checkedAdd(
                    rawIntervalOffsets[localZ], activeIntervals,
                    "fiber anchor shared interval count");
            }
            std::vector<SharedSampleInterval> intervals(
                rawIntervalOffsets.back());
            const size_t rawIntervalBytes = checkedMultiply(
                intervals.capacity(), sizeof(SharedSampleInterval),
                "fiber anchor raw shared intervals");
            report.profile.maximumRawIntervalBytes = std::max(
                report.profile.maximumRawIntervalBytes, rawIntervalBytes);
            std::vector<size_t> mergedIntervalsByZ(zExtent, 0);
            std::vector<std::exception_ptr> intervalErrors(zExtent);
            std::atomic<size_t> nextIntervalZ{0};
            const size_t intervalWorkerCount = std::min(
                static_cast<size_t>(config.parallelThreads), zExtent);
            const auto intervalWorker = [&]() {
                while (true) {
                    const size_t localZ = nextIntervalZ.fetch_add(1);
                    if (localZ >= zExtent)
                        return;
                    try {
                        const size_t z = partitionZBegin + localZ;
                        const size_t rawBegin = rawIntervalOffsets[localZ];
                        const size_t rawEnd = rawIntervalOffsets[localZ + 1];
                        size_t output = rawBegin;
                        for (size_t tileIndex = partition.tileBegin;
                             tileIndex < partition.tileEnd; ++tileIndex) {
                            const auto& tile = tiles[tileIndex];
                            if (z < tile.sampleBegin[0] ||
                                z >= tile.sampleEnd[0]) {
                                continue;
                            }
                            for (size_t y = tile.sampleBegin[1];
                                 y < tile.sampleEnd[1]; ++y) {
                                intervals[output++] = {
                                    z, y, tile.sampleBegin[2],
                                    tile.sampleEnd[2], 0};
                            }
                        }
                        if (output != rawEnd) {
                            throw std::logic_error(
                                "fiber anchor shared interval count changed");
                        }
                        auto beginIt = intervals.begin() +
                            static_cast<std::ptrdiff_t>(rawBegin);
                        auto endIt = intervals.begin() +
                            static_cast<std::ptrdiff_t>(rawEnd);
                        std::sort(beginIt, endIt,
                            [](const auto& left, const auto& right) {
                                return std::tie(left.y, left.xBegin, left.xEnd) <
                                    std::tie(right.y, right.xBegin, right.xEnd);
                            });
                        size_t mergedCount = 0;
                        for (auto interval = beginIt; interval != endIt;
                             ++interval) {
                            if (mergedCount != 0) {
                                auto& previous = intervals[
                                    rawBegin + mergedCount - 1];
                                if (previous.y == interval->y &&
                                    interval->xBegin <= previous.xEnd) {
                                    previous.xEnd = std::max(
                                        previous.xEnd, interval->xEnd);
                                    continue;
                                }
                            }
                            intervals[rawBegin + mergedCount++] = *interval;
                        }
                        mergedIntervalsByZ[localZ] = mergedCount;
                    } catch (...) {
                        intervalErrors[localZ] = std::current_exception();
                    }
                }
            };
            std::vector<std::thread> intervalWorkers;
            intervalWorkers.reserve(intervalWorkerCount);
            for (size_t workerIndex = 0;
                 workerIndex < intervalWorkerCount; ++workerIndex) {
                intervalWorkers.emplace_back(intervalWorker);
            }
            for (auto& thread : intervalWorkers)
                thread.join();
            for (const auto& error : intervalErrors) {
                if (error)
                    std::rethrow_exception(error);
            }
            size_t mergedCount = 0;
            for (size_t localZ = 0; localZ < zExtent; ++localZ) {
                const size_t rawBegin = rawIntervalOffsets[localZ];
                const size_t count = mergedIntervalsByZ[localZ];
                for (size_t index = 0; index < count; ++index) {
                    intervals[mergedCount + index] =
                        intervals[rawBegin + index];
                }
                mergedCount += count;
            }
            intervals.resize(mergedCount);
            report.profile.intervalPreparationSeconds +=
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() -
                    intervalPreparationStart).count();
            report.profile.intervalPreparationCpuSeconds +=
                processCpuSeconds() - intervalPreparationCpuStart;
            std::vector<SharedSampleRow> rows;
            rows.reserve(intervals.size());
            size_t unionSamples = 0;
            for (size_t intervalIndex = 0;
                 intervalIndex < intervals.size(); ++intervalIndex) {
                auto& interval = intervals[intervalIndex];
                interval.sampleOffset = unionSamples;
                unionSamples = checkedAdd(
                    unionSamples, interval.xEnd - interval.xBegin,
                    "fiber anchor partition sample union");
                if (rows.empty() || rows.back().z != interval.z ||
                    rows.back().y != interval.y) {
                    rows.push_back({
                        interval.z, interval.y, intervalIndex,
                        intervalIndex + 1});
                } else {
                    rows.back().intervalEnd = intervalIndex + 1;
                }
            }
            const size_t intervalBytes = checkedMultiply(
                intervals.capacity(), sizeof(SharedSampleInterval),
                "fiber anchor shared intervals");
            const size_t rowBytes = checkedMultiply(
                rows.capacity(), sizeof(SharedSampleRow),
                "fiber anchor shared rows");
            const size_t sharedSampleBytes = checkedMultiply(
                unionSamples, sizeof(FloatStoredPredictionSample),
                "fiber anchor shared samples");
            const size_t sharedBaseBytes = checkedAdd(
                checkedAdd(intervalBytes, rowBytes,
                    "fiber anchor shared metadata"),
                sharedSampleBytes, "fiber anchor shared storage");
            const size_t minimumFitLiveBytes = checkedAdd(
                sharedBaseBytes, partition.maximumTileBytes,
                "fiber anchor partition fitting bytes");
            const size_t minimumSamplingLiveBytes = checkedAdd(
                sharedBaseBytes,
                checkedMultiply(
                    std::min(unionSamples, kMinimumSamplingBatchVoxels),
                    samplingScratchBytesPerVoxel,
                    "fiber anchor minimum sampling scratch"),
                "fiber anchor partition sampling bytes");
            if (minimumFitLiveBytes > config.maximumConcurrentSampleBytes ||
                minimumSamplingLiveBytes > config.maximumConcurrentSampleBytes) {
                throw std::runtime_error(
                    "fiber anchor cell sample exceeds the concurrent byte limit");
            }
            report.profile.maximumSharedSampleBytes = std::max(
                report.profile.maximumSharedSampleBytes, sharedBaseBytes);

            std::vector<FloatStoredPredictionSample> sharedSamples(unionSamples);
            const size_t minimumSamplingBatchVoxels = std::min(
                unionSamples, kMinimumSamplingBatchVoxels);
            const size_t maximumSamplingBatchCount =
                (unionSamples + minimumSamplingBatchVoxels - 1) /
                minimumSamplingBatchVoxels;
            size_t samplingWorkerCount = std::min(
                static_cast<size_t>(config.parallelThreads), unionSamples);
            const size_t maximumSamplingControlBytes = checkedAdd(
                checkedMultiply(
                    maximumSamplingBatchCount, sizeof(std::exception_ptr),
                    "fiber anchor sampling errors"),
                checkedMultiply(
                    samplingWorkerCount,
                    2 * sizeof(double) + sizeof(std::thread),
                    "fiber anchor sampling worker control"),
                "fiber anchor sampling control");
            if (maximumSamplingControlBytes >
                config.maximumConcurrentSampleBytes - sharedBaseBytes) {
                throw std::runtime_error(
                    "fiber anchor cell sample exceeds the concurrent byte limit");
            }
            const size_t samplingAvailableBytes =
                config.maximumConcurrentSampleBytes - sharedBaseBytes -
                maximumSamplingControlBytes;
            const size_t minimumSamplingScratchBytes = checkedMultiply(
                minimumSamplingBatchVoxels,
                samplingScratchBytesPerVoxel,
                "fiber anchor minimum sampling scratch");
            if (samplingAvailableBytes < minimumSamplingScratchBytes) {
                throw std::runtime_error(
                    "fiber anchor cell sample exceeds the concurrent byte limit");
            }
            while (samplingWorkerCount > 1 &&
                   samplingAvailableBytes / samplingWorkerCount <
                       minimumSamplingScratchBytes) {
                --samplingWorkerCount;
            }
            const size_t maximumBatchFromBudget =
                samplingAvailableBytes /
                (samplingWorkerCount * samplingScratchBytesPerVoxel);
            constexpr size_t kMaximumSamplingBatchVoxels = 65536;
            const size_t samplingBatchVoxels = std::max(
                minimumSamplingBatchVoxels, std::min({
                    kMaximumSamplingBatchVoxels,
                    maximumBatchFromBudget,
                    (unionSamples + samplingWorkerCount - 1) /
                        samplingWorkerCount,
                }));
            const size_t samplingBatchCount =
                (unionSamples + samplingBatchVoxels - 1) /
                samplingBatchVoxels;
            samplingWorkerCount = std::min(
                samplingWorkerCount, samplingBatchCount);
            const size_t samplingScratchBytes = checkedMultiply(
                checkedMultiply(
                    samplingWorkerCount, samplingBatchVoxels,
                    "fiber anchor concurrent sampling voxels"),
                samplingScratchBytesPerVoxel,
                "fiber anchor concurrent sampling scratch");
            report.profile.maximumAccountedLiveBytes = std::max(
                report.profile.maximumAccountedLiveBytes,
                checkedAdd(
                    checkedAdd(
                        sharedBaseBytes, maximumSamplingControlBytes,
                        "fiber anchor sampling live bytes"),
                    samplingScratchBytes,
                    "fiber anchor sampling live bytes"));
            report.profile.workers = std::max(
                report.profile.workers, samplingWorkerCount);
            report.profile.sharedSamplingBatches += samplingBatchCount;
            report.profile.maximumSamplingBatchVoxels = std::max(
                report.profile.maximumSamplingBatchVoxels,
                samplingBatchVoxels);
            std::vector<std::exception_ptr> samplingErrors(samplingBatchCount);
            std::vector<double> samplingCoordinateSeconds(samplingWorkerCount);
            std::vector<double> samplingWorkSeconds(samplingWorkerCount);
            std::atomic<size_t> nextSamplingBatch{0};
            const auto sharedSamplingStart = std::chrono::steady_clock::now();
            const double sharedSamplingCpuStart = processCpuSeconds();
            const auto sampleWorker = [&](size_t workerIndex) {
                while (true) {
                    const size_t batchIndex = nextSamplingBatch.fetch_add(1);
                    if (batchIndex >= samplingBatchCount)
                        return;
                    try {
                        const size_t sampleBegin =
                            batchIndex * samplingBatchVoxels;
                        const size_t sampleEnd = std::min(
                            unionSamples, sampleBegin + samplingBatchVoxels);
                        const auto coordinateStart =
                            std::chrono::steady_clock::now();
                        std::vector<CellIndex> indices;
                        indices.reserve(sampleEnd - sampleBegin);
                        auto intervalIt = std::lower_bound(
                            intervals.begin(), intervals.end(), sampleBegin,
                            [](const SharedSampleInterval& interval,
                               size_t offset) {
                                return interval.sampleOffset +
                                    (interval.xEnd - interval.xBegin) <= offset;
                            });
                        size_t offset = sampleBegin;
                        while (offset < sampleEnd) {
                            if (intervalIt == intervals.end()) {
                                throw std::logic_error(
                                    "fiber anchor shared sample offset is missing");
                            }
                            const size_t local = offset -
                                intervalIt->sampleOffset;
                            const size_t available =
                                intervalIt->xEnd - intervalIt->xBegin - local;
                            const size_t count = std::min(
                                available, sampleEnd - offset);
                            for (size_t index = 0; index < count; ++index) {
                                indices.push_back({
                                    intervalIt->z, intervalIt->y,
                                    intervalIt->xBegin + local + index});
                            }
                            offset += count;
                            ++intervalIt;
                        }
                        samplingCoordinateSeconds[workerIndex] +=
                            std::chrono::duration<double>(
                                std::chrono::steady_clock::now() -
                                coordinateStart).count();
                        std::vector<FiberStoredPredictionSample> sampled;
                        const auto samplingStart =
                            std::chrono::steady_clock::now();
                        sampler(indices, 1, sampled);
                        samplingWorkSeconds[workerIndex] +=
                            std::chrono::duration<double>(
                                std::chrono::steady_clock::now() -
                                samplingStart).count();
                        if (sampled.size() != indices.size()) {
                            throw std::runtime_error(
                                "fiber stored prediction sampler returned the wrong sample count");
                        }
                        for (size_t index = 0; index < sampled.size(); ++index) {
                            sharedSamples[sampleBegin + index] =
                                narrowStoredPredictionSample(sampled[index]);
                        }
                    } catch (...) {
                        samplingErrors[batchIndex] = std::current_exception();
                    }
                }
            };
            std::vector<std::thread> samplingWorkers;
            samplingWorkers.reserve(samplingWorkerCount);
            for (size_t workerIndex = 0;
                 workerIndex < samplingWorkerCount; ++workerIndex) {
                samplingWorkers.emplace_back(sampleWorker, workerIndex);
            }
            for (auto& thread : samplingWorkers)
                thread.join();
            report.profile.sharedSamplingSeconds +=
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() -
                    sharedSamplingStart).count();
            report.profile.sharedSamplingCpuSeconds +=
                processCpuSeconds() - sharedSamplingCpuStart;
            report.profile.coordinateConstructionWorkSeconds +=
                std::accumulate(
                    samplingCoordinateSeconds.begin(),
                    samplingCoordinateSeconds.end(), 0.0);
            report.profile.predictionSamplingWorkSeconds +=
                std::accumulate(
                    samplingWorkSeconds.begin(), samplingWorkSeconds.end(), 0.0);
            report.profile.predictionSamplerCalls += samplingBatchCount;
            report.profile.submittedPredictionVoxels += unionSamples;
            std::exception_ptr samplingError;
            for (const auto& error : samplingErrors) {
                if (error) {
                    samplingError = error;
                    break;
                }
            }
            if (samplingError) {
                for (size_t tileIndex = partition.tileBegin;
                     tileIndex < partition.tileEnd; ++tileIndex) {
                    for (const size_t cellIndex : tiles[tileIndex].cells) {
                        jobErrors[cellIndex] = samplingError;
                        reportCellCompleted();
                    }
                }
                partitionDurations.push_back(std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - partitionStart).count());
                continue;
            }

            const size_t partitionTileCount =
                partition.tileEnd - partition.tileBegin;
            struct ReadyTile {
                const Tile* tile = nullptr;
                const std::vector<CompactFiberAnchorObservation>* observations =
                    nullptr;
                std::array<size_t, 3> sampleShape{};
                size_t remainingCells = 0;
            };
            struct ReadyCellTask {
                ReadyTile* readyTile = nullptr;
                size_t cellIndex = 0;
            };
            size_t partitionCellCount = 0;
            for (size_t tileIndex = partition.tileBegin;
                 tileIndex < partition.tileEnd; ++tileIndex) {
                partitionCellCount = checkedAdd(
                    partitionCellCount, tiles[tileIndex].cells.size(),
                    "fiber anchor ready-cell queue");
            }
            const size_t readyQueueBytes = checkedMultiply(
                partitionCellCount, sizeof(ReadyCellTask),
                "fiber anchor ready-cell queue");
            const size_t fittingProfileBytes = checkedAdd(
                checkedMultiply(
                    partitionCellCount, sizeof(double),
                    "fiber anchor cell timing storage"),
                checkedMultiply(
                    partitionTileCount, sizeof(double),
                    "fiber anchor tile timing storage"),
                "fiber anchor fitting profile storage");
            const size_t fittingBaseBytes = checkedAdd(
                checkedAdd(
                    sharedBaseBytes, readyQueueBytes,
                    "fiber anchor fitting base bytes"),
                fittingProfileBytes,
                "fiber anchor fitting base bytes");
            if (fittingBaseBytes > config.maximumConcurrentSampleBytes ||
                partition.maximumTileBytes >
                    config.maximumConcurrentSampleBytes - fittingBaseBytes) {
                throw std::runtime_error(
                    "fiber anchor cell sample exceeds the concurrent byte limit");
            }
            const size_t fitAvailableBytes =
                config.maximumConcurrentSampleBytes - fittingBaseBytes;
            const size_t fitWorkerCount = std::min({
                partitionTileCount,
                static_cast<size_t>(config.parallelThreads),
                std::max<size_t>(
                    1, fitAvailableBytes / partition.maximumTileBytes),
            });
            report.profile.workers = std::max(
                report.profile.workers, fitWorkerCount);
            report.profile.maximumAccountedLiveBytes = std::max(
                report.profile.maximumAccountedLiveBytes,
                checkedAdd(
                    fittingBaseBytes,
                    checkedMultiply(
                        fitWorkerCount, partition.maximumTileBytes,
                        "fiber anchor fitting live bytes"),
                    "fiber anchor fitting live bytes"));

            std::mutex readyCellMutex;
            std::condition_variable readyCellCondition;
            std::vector<ReadyCellTask> readyCells;
            readyCells.reserve(partitionCellCount);
            size_t nextReadyCell = 0;
            std::atomic<size_t> nextTile{partition.tileBegin};
            size_t completedTiles = 0;
            const auto fitWorker = [&](size_t workerIndex) {
                auto& workerProfile = workerProfiles[workerIndex];
                std::vector<uint32_t> cellObservationIndices;
                std::vector<uint8_t> cellGradientValidity;
                const auto processReadyCell = [&](const ReadyCellTask& task) {
                    const auto cellStart = std::chrono::steady_clock::now();
                    try {
                        const auto& ready = *task.readyTile;
                        auto result = processCell(
                            requestedCells[task.cellIndex], *ready.tile,
                            *ready.observations, ready.sampleShape,
                            cellObservationIndices, cellGradientValidity,
                            workerProfile);
                        const bool retainResult = retainDiagnostics ||
                            prepareSparseCellResult(
                                result,
                                tallySelectedDiagnostics && selectedCell(
                                    result.cellZYX),
                                workerProfile);
                        if (retainResult) {
                            jobResults[task.cellIndex] =
                                std::make_unique<FiberCellAnchorResult>(
                                    std::move(result));
                        }
                    } catch (...) {
                        jobErrors[task.cellIndex] = std::current_exception();
                    }
                    const double duration = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - cellStart).count();
                    {
                        std::lock_guard lock(readyCellMutex);
                        --task.readyTile->remainingCells;
                    }
                    readyCellCondition.notify_all();
                    try {
                        workerProfile.cellProcessingDurations.push_back(duration);
                    } catch (...) {
                        if (!jobErrors[task.cellIndex])
                            jobErrors[task.cellIndex] = std::current_exception();
                    }
                    reportCellCompleted();
                };
                const auto helpUntil = [&](const ReadyTile* ownTile) {
                    while (true) {
                        ReadyCellTask task;
                        {
                            std::unique_lock lock(readyCellMutex);
                            readyCellCondition.wait(lock, [&]() {
                                return nextReadyCell < readyCells.size() ||
                                    (ownTile != nullptr
                                        ? ownTile->remainingCells == 0
                                        : completedTiles ==
                                            partitionTileCount);
                            });
                            if (nextReadyCell == readyCells.size())
                                return;
                            task = readyCells[nextReadyCell++];
                        }
                        processReadyCell(task);
                    }
                };
                while (true) {
                    const size_t tileIndex = nextTile.fetch_add(1);
                    if (tileIndex >= partition.tileEnd) {
                        helpUntil(nullptr);
                        break;
                    }
                    const Tile& tile = tiles[tileIndex];
                    try {
                    const auto tilePreparationStart =
                        std::chrono::steady_clock::now();
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
                    const auto copyStart = std::chrono::steady_clock::now();
                    std::vector<FloatStoredPredictionSample> samples(sampleCount);
                    const size_t plane = sampleShape[1] * sampleShape[2];
                    for (size_t z = tile.sampleBegin[0]; z < tile.sampleEnd[0]; ++z) {
                        for (size_t y = tile.sampleBegin[1]; y < tile.sampleEnd[1]; ++y) {
                            const auto rowIt = std::lower_bound(
                                rows.begin(), rows.end(), std::pair{z, y},
                                [](const SharedSampleRow& row,
                                   const std::pair<size_t, size_t>& key) {
                                    return std::pair{row.z, row.y} < key;
                                });
                            if (rowIt == rows.end() || rowIt->z != z ||
                                rowIt->y != y) {
                                throw std::logic_error(
                                    "fiber anchor shared sample row is missing");
                            }
                            const auto intervalIt = std::find_if(
                                intervals.begin() +
                                    static_cast<std::ptrdiff_t>(
                                        rowIt->intervalBegin),
                                intervals.begin() +
                                    static_cast<std::ptrdiff_t>(
                                        rowIt->intervalEnd),
                                [&](const SharedSampleInterval& interval) {
                                    return interval.xBegin <= tile.sampleBegin[2] &&
                                        interval.xEnd >= tile.sampleEnd[2];
                                });
                            if (intervalIt == intervals.begin() +
                                    static_cast<std::ptrdiff_t>(
                                        rowIt->intervalEnd)) {
                                throw std::logic_error(
                                    "fiber anchor shared sample interval is missing");
                            }
                            const size_t sourceOffset =
                                intervalIt->sampleOffset +
                                tile.sampleBegin[2] - intervalIt->xBegin;
                            const size_t destinationOffset =
                                (z - tile.sampleBegin[0]) * plane +
                                (y - tile.sampleBegin[1]) * sampleShape[2];
                            std::copy_n(
                                sharedSamples.begin() +
                                    static_cast<std::ptrdiff_t>(sourceOffset),
                                sampleShape[2],
                                samples.begin() +
                                    static_cast<std::ptrdiff_t>(
                                        destinationOffset));
                        }
                    }
                    workerProfile.tileSampleCopySeconds +=
                        std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - copyStart).count();
                    std::vector<CachedPresenceGradient> gradients;
                    if (config.peakGradientWeight > 0.0F) {
                        const auto gradientStart = std::chrono::steady_clock::now();
                        gradients.resize(sampleCount);
                        constexpr std::array<float, 3> smooth{0.25F, 0.5F, 0.25F};
                        constexpr std::array<float, 3> derivative{-0.5F, 0.0F, 0.5F};
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
                                                const float presence = sample.presence;
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
                                observation.direction = sample.direction;
                                observation.presence = sample.presence;
                                observation.valid = sample.valid;
                                if (config.peakGradientWeight > 0.0F &&
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
                    workerProfile.tilePreparationDurations.push_back(
                        std::chrono::duration<double>(
                            std::chrono::steady_clock::now() -
                            tilePreparationStart).count());
                    ReadyTile readyTile;
                    readyTile.tile = &tile;
                    readyTile.observations = &observations;
                    readyTile.sampleShape = sampleShape;
                    readyTile.remainingCells = tile.cells.size();
                    {
                        std::lock_guard lock(readyCellMutex);
                        for (const size_t cellIndex : tile.cells)
                            readyCells.push_back({&readyTile, cellIndex});
                    }
                    readyCellCondition.notify_all();
                    helpUntil(&readyTile);
                    } catch (...) {
                        const auto error = std::current_exception();
                        for (const size_t cellIndex : tile.cells) {
                            jobErrors[cellIndex] = error;
                            reportCellCompleted();
                        }
                    }
                    {
                        std::lock_guard lock(readyCellMutex);
                        ++completedTiles;
                    }
                    readyCellCondition.notify_all();
                }
            };
            std::vector<std::thread> fitWorkers;
            fitWorkers.reserve(fitWorkerCount);
            for (size_t workerIndex = 0;
                 workerIndex < fitWorkerCount; ++workerIndex) {
                fitWorkers.emplace_back(fitWorker, workerIndex);
            }
            for (auto& thread : fitWorkers)
                thread.join();
            partitionDurations.push_back(std::chrono::duration<double>(
                std::chrono::steady_clock::now() - partitionStart).count());
        }
        report.profile.cellProcessingSeconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - cellProcessingStart).count();
        report.profile.cellProcessingCpuSeconds +=
            processCpuSeconds() - cellProcessingCpuStart;
        report.profile.reusedPredictionVoxels +=
            tileOccurrences - report.profile.submittedPredictionVoxels;
        std::vector<double> tilePreparationDurations;
        std::vector<double> cellProcessingDurations;
        for (const auto& workerProfile : workerProfiles) {
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
            report.profile.retainPredicateCalls +=
                workerProfile.retainPredicateCalls;
            report.diagnostics.emptyComponents +=
                workerProfile.emptyComponents;
            report.diagnostics.degenerateComponents +=
                workerProfile.degenerateComponents;
            report.diagnostics.belowSupportComponents +=
                workerProfile.belowSupportComponents;
            report.diagnostics.mergedComponentPairs +=
                workerProfile.mergedComponentPairs;
            report.diagnostics.outsideSelectionComponents +=
                workerProfile.outsideSelectionComponents;
            report.profile.coordinateConstructionWorkSeconds +=
                workerProfile.coordinateConstructionSeconds;
            report.profile.predictionSamplingWorkSeconds +=
                workerProfile.predictionSamplingSeconds;
            report.profile.tileSampleCopyWorkSeconds +=
                workerProfile.tileSampleCopySeconds;
            report.profile.gradientConstructionWorkSeconds +=
                workerProfile.gradientConstructionSeconds;
            report.profile.observationConstructionWorkSeconds +=
                workerProfile.observationConstructionSeconds;
            report.profile.fittingWorkSeconds += workerProfile.fittingSeconds;
            tilePreparationDurations.insert(
                tilePreparationDurations.end(),
                workerProfile.tilePreparationDurations.begin(),
                workerProfile.tilePreparationDurations.end());
            cellProcessingDurations.insert(
                cellProcessingDurations.end(),
                workerProfile.cellProcessingDurations.begin(),
                workerProfile.cellProcessingDurations.end());
            accumulateFitProfile(report.profile.fit, workerProfile.fit);
        }
        const auto assignDurationQuantiles = [](
            std::vector<double>& durations,
            double& p50,
            double& p95,
            double& maximum) {
            if (durations.empty())
                return;
            std::sort(durations.begin(), durations.end());
            const auto percentile = [&](double quantile) {
                const size_t index = static_cast<size_t>(std::ceil(
                    quantile * static_cast<double>(durations.size()))) - 1;
                return durations[std::min(index, durations.size() - 1)];
            };
            p50 = percentile(0.50);
            p95 = percentile(0.95);
            maximum = durations.back();
        };
        assignDurationQuantiles(
            partitionDurations,
            report.profile.partitionP50Seconds,
            report.profile.partitionP95Seconds,
            report.profile.partitionMaximumSeconds);
        assignDurationQuantiles(
            tilePreparationDurations,
            report.profile.tilePreparationP50Seconds,
            report.profile.tilePreparationP95Seconds,
            report.profile.tilePreparationMaximumSeconds);
        assignDurationQuantiles(
            cellProcessingDurations,
            report.profile.cellProcessingP50Seconds,
            report.profile.cellProcessingP95Seconds,
            report.profile.cellProcessingMaximumSeconds);

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
        results.reserve(retainDiagnostics
            ? requestedCells.size()
            : std::min<size_t>(requestedCells.size(), 4096));
        for (auto& result : jobResults) {
            if (!result)
                continue;
            FiberCellAnchorResult cell = std::move(*result);
            result.reset();
            const bool tallyCell =
                tallySelectedDiagnostics && selectedCell(cell.cellZYX);
            if (retainDiagnostics && tallyCell && retainPredicate) {
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
            if (retainDiagnostics && tallyCell) {
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
            if (cell.retainedAnchorCount > 0 ||
                (retainDiagnostics && tallyCell)) {
                results.push_back(std::move(cell));
            }
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
    if (retainDiagnostics) {
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
            if (retainDiagnostics) {
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
    const FiberAnchorProgressCallback& progressCallback,
    bool retainDiagnostics)
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
        false,
        retainDiagnostics);
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
    if (!(radiusBaseVoxels >= 0.0F) || !std::isfinite(radiusBaseVoxels) ||
        !(grid.predictionToBaseScale > 0.0F) ||
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
                        (static_cast<double>(begin[2]) - 0.5F) * scale,
                        (static_cast<double>(begin[1]) - 0.5F) * scale,
                        (static_cast<double>(begin[0]) - 0.5F) * scale,
                    };
                    const cv::Vec3d cellHigh{
                        (static_cast<double>(end[2]) - 0.5F) * scale,
                        (static_cast<double>(end[1]) - 0.5F) * scale,
                        (static_cast<double>(end[0]) - 0.5F) * scale,
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

bool fiberAnchorCellIntersectsPolylineTube(
    const std::array<size_t, 3>& cellZYX,
    const std::vector<cv::Vec3d>& referenceLineBase,
    double radiusBaseVoxels,
    const FiberPredictionGridInfo& grid,
    int anchorCellSizePredictionVoxels)
{
    if (!(radiusBaseVoxels >= 0.0) || !std::isfinite(radiusBaseVoxels) ||
        !(grid.predictionToBaseScale > 0.0F) ||
        !std::isfinite(grid.predictionToBaseScale) ||
        anchorCellSizePredictionVoxels < 1 || referenceLineBase.size() < 2) {
        throw std::invalid_argument(
            "fiber anchor cell tube test configuration is invalid");
    }
    const auto cellSize = static_cast<size_t>(
        anchorCellSizePredictionVoxels);
    std::array<size_t, 3> begin{};
    std::array<size_t, 3> end{};
    for (size_t axis = 0; axis < 3; ++axis) {
        const size_t cellCount =
            (grid.shapeZYX[axis] + cellSize - 1) / cellSize;
        if (cellZYX[axis] >= cellCount)
            return false;
        begin[axis] = cellZYX[axis] * cellSize;
        end[axis] = std::min(grid.shapeZYX[axis], begin[axis] + cellSize);
    }
    const double scale = grid.predictionToBaseScale;
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
    const double radiusSquared = radiusBaseVoxels * radiusBaseVoxels;
    for (size_t index = 1; index < referenceLineBase.size(); ++index) {
        if (segmentAabbDistanceSquared(
                referenceLineBase[index - 1], referenceLineBase[index],
                cellLow, cellHigh) <= radiusSquared + kGeometryEpsilon) {
            return true;
        }
    }
    return false;
}

FiberAnchorBenchmarkReport benchmarkRefinedFiberAnchors(
    const FiberAnchorExtractionReport& anchors,
    const std::vector<cv::Vec3d>& referenceLineBase,
    const std::vector<float>& thresholdsBaseVoxels)
{
    if (!(anchors.grid.predictionToBaseScale > 0.0F) ||
        !std::isfinite(anchors.grid.predictionToBaseScale) ||
        anchors.selectedCellsZYX.empty()) {
        throw std::invalid_argument("anchor benchmark extraction report is invalid");
    }
    if (thresholdsBaseVoxels.empty() ||
        std::any_of(
            thresholdsBaseVoxels.begin(), thresholdsBaseVoxels.end(),
            [](float value) { return !(value >= 0.0F) || !std::isfinite(value); })) {
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
        std::map<std::array<size_t, 3>, std::vector<float>> distancesByCell;
        for (const auto& cell : anchors.selectedCellsZYX)
            distancesByCell.emplace(cell, std::vector<float>{});

        std::vector<float> distances;
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
            cv::Vec3f pointPrediction;
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
            const float distance = static_cast<float>(distanceToPolylineArc(
                reference,
                cv::Vec3d{pointPrediction[0], pointPrediction[1], pointPrediction[2]} *
                    anchors.grid.predictionToBaseScale,
                0.0F,
                reference.length()));
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
                std::accumulate(distances.begin(), distances.end(), 0.0F) /
                static_cast<float>(distances.size());
            result.anchorDistancesBaseVoxels.median =
                interpolatedQuantile(distances, 0.5F);
            result.anchorDistancesBaseVoxels.percentile95 =
                interpolatedQuantile(distances, 0.95F);
        }
        for (const float threshold : thresholdsBaseVoxels) {
            FiberAnchorBenchmarkThreshold measured;
            measured.thresholdBaseVoxels = threshold;
            measured.anchorHits = static_cast<size_t>(std::upper_bound(
                distances.begin(), distances.end(),
                threshold + kGeometryEpsilon) - distances.begin());
            if (!distances.empty()) {
                measured.anchorHitRate =
                    static_cast<float>(measured.anchorHits) /
                    static_cast<float>(distances.size());
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
            measured.cellHitRate = static_cast<float>(measured.cellHits) /
                static_cast<float>(result.referenceCells);
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
                detail::checkedScaleFloatIndex(report.selectedCrop.originXYZ[0], report.grid.predictionToBaseScale, "fiber anchor crop origin"),
                detail::checkedScaleFloatIndex(report.selectedCrop.originXYZ[1], report.grid.predictionToBaseScale, "fiber anchor crop origin"),
                detail::checkedScaleFloatIndex(report.selectedCrop.originXYZ[2], report.grid.predictionToBaseScale, "fiber anchor crop origin"),
            }},
            {"prediction_interval_size_base_xyz", {
                detail::checkedScaleFloatIndex(report.selectedCrop.sizeXYZ[0], report.grid.predictionToBaseScale, "fiber anchor crop size"),
                detail::checkedScaleFloatIndex(report.selectedCrop.sizeXYZ[1], report.grid.predictionToBaseScale, "fiber anchor crop size"),
                detail::checkedScaleFloatIndex(report.selectedCrop.sizeXYZ[2], report.grid.predictionToBaseScale, "fiber anchor crop size"),
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
                const cv::Vec3f positionBase = detail::checkedScaleFloatPosition(
                    anchor.positionPredictionXYZ,
                    report.grid.predictionToBaseScale,
                    "fiber anchor base position");
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

nlohmann::json nullableDouble(const std::optional<float>& value)
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
    if (!(artifact.glyphLengthBaseVoxels > 0.0F) ||
        !std::isfinite(artifact.glyphLengthBaseVoxels)) {
        throw std::invalid_argument("fiber anchor OBJ glyph length must be positive and finite");
    }
    std::ostringstream output;
    output << std::setprecision(std::numeric_limits<float>::max_digits10);
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
            const cv::Vec3f center = detail::checkedScaleFloatPosition(
                anchor.positionPredictionXYZ,
                report.grid.predictionToBaseScale,
                "fiber anchor OBJ base position");
            const cv::Vec3f half = anchor.axisXYZ *
                (artifact.glyphLengthBaseVoxels * 0.5F);
            const cv::Vec3f first = detail::checkedFloatPosition(
                center - half, "fiber anchor OBJ endpoint");
            const cv::Vec3f second = detail::checkedFloatPosition(
                center + half, "fiber anchor OBJ endpoint");
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
    if (!(artifact.glyphLengthBaseVoxels > 0.0F) ||
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
            const cv::Vec3f positionBase = detail::checkedScaleFloatPosition(
                anchor.positionPredictionXYZ,
                report.grid.predictionToBaseScale,
                "fiber anchor diagnostic base position");
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
    if (!(report.grid.predictionToBaseScale > 0.0F) ||
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
    output << std::setprecision(std::numeric_limits<float>::max_digits10);
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
        const cv::Vec3f centerPredictionXYZ{
            0.5F * static_cast<float>(begin[2] + end[2] - 1),
            0.5F * static_cast<float>(begin[1] + end[1] - 1),
            0.5F * static_cast<float>(begin[0] + end[0] - 1),
        };
        const cv::Vec3f centerBase = detail::checkedScaleFloatPosition(
            centerPredictionXYZ, report.grid.predictionToBaseScale,
            "fiber anchor cell base position");
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
            const cv::Vec3f anchorBase = detail::checkedScaleFloatPosition(
                component.anchor.positionPredictionXYZ,
                report.grid.predictionToBaseScale,
                "fiber anchor cell base position");
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
