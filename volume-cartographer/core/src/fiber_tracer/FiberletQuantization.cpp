#include "vc/fiber_tracer/FiberletQuantization.hpp"

#include "vc/fiber_tracer/FiberReplayMetric.hpp"
#include "vc/fiber_tracer/PolylineGeometry.hpp"
#include "vc/lasagna/ChannelSampler.hpp"

#include <boost/geometry.hpp>
#include <boost/geometry/index/rtree.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <numbers>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace vc::fiber_tracer
{
namespace
{

constexpr double kEpsilon = 1.0e-12;

struct SummaryBuilder {
    std::vector<double> values;

    FiberletQuantizationSummary finish() const
    {
        FiberletQuantizationSummary result;
        result.count = values.size();
        if (values.empty())
            return result;
        result.minimum = *std::min_element(values.begin(), values.end());
        result.maximum = *std::max_element(values.begin(), values.end());
        result.mean = std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
        auto ordered = values;
        std::sort(ordered.begin(), ordered.end());
        const size_t middle = ordered.size() / 2;
        result.median = ordered.size() % 2 == 0
            ? 0.5 * (ordered[middle - 1] + ordered[middle])
            : ordered[middle];
        return result;
    }
};

struct QuantizedAnchorKey {
    std::array<size_t, 3> cellZYX{0, 0, 0};
    uint8_t variant = 0;

    auto operator<=>(const QuantizedAnchorKey&) const = default;
};

struct QuantizedAnchor {
    QuantizedAnchorKey key;
    cv::Vec3f positionPredictionXYZ{0.0F, 0.0F, 0.0F};
    cv::Vec3f compactAxisXYZ{1.0F, 0.0F, 0.0F};
};

using DirectedCandidate = std::pair<size_t, bool>;

double length(const cv::Vec3f& value)
{
    return std::sqrt(static_cast<double>(value.dot(value)));
}

cv::Vec3f normalized(const cv::Vec3f& value)
{
    const double norm = length(value);
    if (!(norm > kEpsilon) || !std::isfinite(norm))
        throw std::invalid_argument("quantized fiberlet has a degenerate axis");
    return value * static_cast<float>(1.0 / norm);
}

double unorientedAngleDegrees(const cv::Vec3f& left, const cv::Vec3f& right)
{
    const cv::Vec3f a = normalized(left);
    const cv::Vec3f b = normalized(right);
    const double dot = std::clamp(std::abs(static_cast<double>(a.dot(b))), 0.0, 1.0);
    return std::acos(dot) * 180.0 / std::numbers::pi;
}

double pathLength(const std::vector<cv::Vec3f>& points)
{
    double result = 0.0;
    for (size_t index = 1; index < points.size(); ++index)
        result += length(points[index] - points[index - 1]);
    return result;
}

cv::Vec3f samplePathFraction(const std::vector<cv::Vec3f>& points, double fraction)
{
    if (points.empty())
        throw std::invalid_argument("cannot sample an empty fiberlet path");
    if (points.size() == 1)
        return points.front();
    const double total = pathLength(points);
    if (!(total > kEpsilon))
        throw std::invalid_argument("cannot sample a zero-length fiberlet path");
    const double target = std::clamp(fraction, 0.0, 1.0) * total;
    double traversed = 0.0;
    for (size_t index = 1; index < points.size(); ++index) {
        const double segment = length(points[index] - points[index - 1]);
        if (traversed + segment >= target || index + 1 == points.size()) {
            const double local = segment > kEpsilon ? (target - traversed) / segment : 0.0;
            return points[index - 1] +
                   (points[index] - points[index - 1]) * static_cast<float>(std::clamp(local, 0.0, 1.0));
        }
        traversed += segment;
    }
    return points.back();
}

DirectedCandidate directedCandidate(const FiberletGraph& graph, size_t arc)
{
    return {graph.edges.at(arc / 2).candidateIndex, arc % 2 == 0};
}

std::vector<size_t> successfulCandidateIndices(const FiberletPathReport& paths)
{
    std::vector<size_t> result;
    for (size_t index = 0; index < paths.candidates.size(); ++index) {
        if (paths.candidates[index].success)
            result.push_back(index);
    }
    return result;
}

std::vector<size_t> replayCandidates(const FiberletGraphReplayResult& replay)
{
    std::vector<size_t> result;
    for (const auto& segment : replay.segments)
        result.insert(result.end(), segment.candidateIndices.begin(), segment.candidateIndices.end());
    return result;
}

struct ReplayPolylineSet {
    using IndexPoint = boost::geometry::model::point<
        double, 3, boost::geometry::cs::cartesian>;
    using IndexBox = boost::geometry::model::box<IndexPoint>;
    using IndexValue = std::pair<IndexBox, size_t>;
    using Index = boost::geometry::index::rtree<
        IndexValue, boost::geometry::index::quadratic<16>>;
    struct Segment {
        cv::Vec3d first;
        cv::Vec3d second;
    };
    std::vector<PolylineArcGeometry> polylines;
    std::vector<cv::Vec3d> singletons;
    std::vector<Segment> segments;
    Index index;
};

ReplayPolylineSet::IndexPoint indexPoint(const cv::Vec3d& point)
{
    return {point[0], point[1], point[2]};
}

void finishReplayIndex(ReplayPolylineSet& lines)
{
    std::vector<ReplayPolylineSet::IndexValue> values;
    for (const auto& polyline : lines.polylines) {
        const auto& points = polyline.points;
        for (size_t index = 1; index < points.size(); ++index)
            lines.segments.push_back({points[index - 1], points[index]});
    }
    for (const auto& point : lines.singletons)
        lines.segments.push_back({point, point});
    values.reserve(lines.segments.size());
    for (size_t index = 0; index < lines.segments.size(); ++index) {
        const auto& segment = lines.segments[index];
        ReplayPolylineSet::IndexPoint minimum;
        ReplayPolylineSet::IndexPoint maximum;
        boost::geometry::set<0>(minimum, std::min(segment.first[0], segment.second[0]));
        boost::geometry::set<1>(minimum, std::min(segment.first[1], segment.second[1]));
        boost::geometry::set<2>(minimum, std::min(segment.first[2], segment.second[2]));
        boost::geometry::set<0>(maximum, std::max(segment.first[0], segment.second[0]));
        boost::geometry::set<1>(maximum, std::max(segment.first[1], segment.second[1]));
        boost::geometry::set<2>(maximum, std::max(segment.first[2], segment.second[2]));
        values.emplace_back(ReplayPolylineSet::IndexBox{minimum, maximum}, index);
    }
    lines.index = ReplayPolylineSet::Index(values.begin(), values.end());
}

ReplayPolylineSet replayPolylineSet(const FiberletGraphReplayResult& replay)
{
    ReplayPolylineSet result;
    for (const auto& segment : replay.segments) {
        std::vector<cv::Vec3d> points;
        points.reserve(segment.routePointsBaseXYZ.size());
        for (const auto& point : segment.routePointsBaseXYZ) {
            if (points.empty() || cv::norm(point - points.back()) > kEpsilon)
                points.push_back(point);
        }
        if (points.size() >= 2)
            result.polylines.push_back(makePolylineArcGeometry(points));
        else if (points.size() == 1)
            result.singletons.push_back(points.front());
    }
    finishReplayIndex(result);
    return result;
}

struct ClosestLinePoint {
    cv::Vec3d point{0.0, 0.0, 0.0};
    double distance = std::numeric_limits<double>::infinity();
};

ClosestLinePoint closestLinePoint(
    const ReplayPolylineSet& target,
    const cv::Vec3d& point)
{
    ClosestLinePoint best;
    if (target.segments.empty())
        return best;
    const auto query = indexPoint(point);
    size_t count = std::min<size_t>(16, target.segments.size());
    while (true) {
        std::vector<ReplayPolylineSet::IndexValue> nearest;
        nearest.reserve(count);
        target.index.query(
            boost::geometry::index::nearest(query, count),
            std::back_inserter(nearest));
        double maximumLowerBound = 0.0;
        for (const auto& [box, index] : nearest) {
            maximumLowerBound = std::max(maximumLowerBound,
                boost::geometry::distance(query, box));
            const auto& segment = target.segments[index];
            const cv::Vec3d delta = segment.second - segment.first;
            const double squared = delta.dot(delta);
            const double fraction = squared > kEpsilon
                ? std::clamp((point - segment.first).dot(delta) / squared,
                      0.0, 1.0)
                : 0.0;
            const cv::Vec3d closest = segment.first + fraction * delta;
            const double distance = cv::norm(point - closest);
            if (distance < best.distance)
                best = {closest, distance};
        }
        if (count == target.segments.size() ||
            best.distance <= maximumLowerBound + kEpsilon) {
            break;
        }
        count = std::min(target.segments.size(), count * 2);
    }
    return best;
}

struct ReplayLineDistance {
    bool available = false;
    size_t samples = 0;
    size_t invalidNormalSamples = 0;
    FiberletQuantizationSummary euclideanBaseVoxels;
    FiberletQuantizationSummary normalBaseVoxels;
    FiberletQuantizationSummary tangentialBaseVoxels;
};

size_t lineSampleCount(const ReplayPolylineSet& lines)
{
    size_t result = lines.singletons.size();
    for (const auto& polyline : lines.polylines) {
        result += std::max<size_t>(
            1, static_cast<size_t>(std::ceil(polyline.length()))) + 1;
    }
    return result;
}

struct LineDistanceProgress {
    const FiberletQuantizationProgressCallback& callback;
    std::string phase;
    std::string scenario;
    size_t total = 0;
    size_t completed = 0;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point last = start;

    void notify(bool force = false)
    {
        if (!callback)
            return;
        const auto now = std::chrono::steady_clock::now();
        if (!force && now - last < std::chrono::seconds(1))
            return;
        last = now;
        callback({
            phase,
            scenario,
            completed,
            total,
            std::chrono::duration<double>(now - start).count(),
        });
    }
};

void measureDirectedReplayLineDistance(
    SummaryBuilder& euclidean,
    SummaryBuilder& normal,
    SummaryBuilder& tangential,
    size_t& invalidNormalSamples,
    const ReplayPolylineSet& source,
    const ReplayPolylineSet& target,
    const vc::lasagna::NormalSampler& normalSampler,
    double normalWorkingToBaseScale,
    double normalThresholdBaseVoxels,
    bool normalAtTarget,
    LineDistanceProgress& progress)
{
    const auto measure = [&](const cv::Vec3d& point) {
        const auto closest = closestLinePoint(target, point);
        if (!std::isfinite(closest.distance))
            return;
        const auto components = measureFiberReplayThreshold(
            normalAtTarget ? point : closest.point,
            normalAtTarget ? closest.point : point,
            normalSampler, normalWorkingToBaseScale,
            normalThresholdBaseVoxels);
        euclidean.values.push_back(components.euclideanErrorBaseVoxels);
        if (components.localNormalValid) {
            normal.values.push_back(*components.normalErrorBaseVoxels);
            tangential.values.push_back(*components.tangentialErrorBaseVoxels);
        } else {
            ++invalidNormalSamples;
        }
        ++progress.completed;
        progress.notify();
    };
    for (const auto& polyline : source.polylines) {
        const size_t intervals = std::max<size_t>(
            1, static_cast<size_t>(std::ceil(polyline.length())));
        for (size_t index = 0; index <= intervals; ++index) {
            measure(samplePolylineArc(
                polyline,
                polyline.length() * static_cast<double>(index) /
                    static_cast<double>(intervals)).point);
        }
    }
    for (const auto& singleton : source.singletons)
        measure(singleton);
}

ReplayLineDistance symmetricReplayLineDistance(
    const ReplayPolylineSet& left,
    const ReplayPolylineSet& right,
    const vc::lasagna::NormalSampler& normalSampler,
    double normalWorkingToBaseScale,
    double normalThresholdBaseVoxels,
    bool rightIsReference,
    std::string phase,
    std::string scenario,
    const FiberletQuantizationProgressCallback& callback)
{
    const bool leftEmpty = left.polylines.empty() && left.singletons.empty();
    const bool rightEmpty = right.polylines.empty() && right.singletons.empty();
    ReplayLineDistance result;
    if (leftEmpty || rightEmpty)
        return result;
    SummaryBuilder euclidean;
    SummaryBuilder normal;
    SummaryBuilder tangential;
    LineDistanceProgress progress{
        callback,
        std::move(phase),
        std::move(scenario),
        lineSampleCount(left) + lineSampleCount(right),
    };
    progress.notify(true);
    result.available = true;
    measureDirectedReplayLineDistance(
        euclidean, normal, tangential, result.invalidNormalSamples,
        left, right, normalSampler, normalWorkingToBaseScale,
        normalThresholdBaseVoxels, true, progress);
    measureDirectedReplayLineDistance(
        euclidean, normal, tangential, result.invalidNormalSamples,
        right, left, normalSampler, normalWorkingToBaseScale,
        normalThresholdBaseVoxels, !rightIsReference, progress);
    progress.notify(true);
    result.samples = euclidean.values.size();
    result.euclideanBaseVoxels = euclidean.finish();
    result.normalBaseVoxels = normal.finish();
    result.tangentialBaseVoxels = tangential.finish();
    const auto removeRoundoff = [](FiberletQuantizationSummary& summary) {
        const auto clean = [](double value) {
            return value <= 1.0e-9 ? 0.0 : value;
        };
        summary.minimum = clean(summary.minimum);
        summary.mean = clean(summary.mean);
        summary.median = clean(summary.median);
        summary.maximum = clean(summary.maximum);
    };
    removeRoundoff(result.euclideanBaseVoxels);
    removeRoundoff(result.normalBaseVoxels);
    removeRoundoff(result.tangentialBaseVoxels);
    return result;
}

ReplayLineDistance directedReferenceLineDistance(
    const ReplayPolylineSet& replay,
    const ReplayPolylineSet& reference,
    const vc::lasagna::NormalSampler& normalSampler,
    double normalWorkingToBaseScale,
    double normalThresholdBaseVoxels,
    std::string scenario,
    const FiberletQuantizationProgressCallback& callback)
{
    const bool replayEmpty = replay.polylines.empty() && replay.singletons.empty();
    const bool referenceEmpty = reference.polylines.empty() && reference.singletons.empty();
    ReplayLineDistance result;
    if (replayEmpty || referenceEmpty)
        return result;
    SummaryBuilder euclidean;
    SummaryBuilder normal;
    SummaryBuilder tangential;
    LineDistanceProgress progress{
        callback,
        "reference_distance",
        std::move(scenario),
        lineSampleCount(replay),
    };
    progress.notify(true);
    result.available = true;
    measureDirectedReplayLineDistance(
        euclidean, normal, tangential, result.invalidNormalSamples,
        replay, reference, normalSampler, normalWorkingToBaseScale,
        normalThresholdBaseVoxels, true, progress);
    progress.notify(true);
    result.samples = euclidean.values.size();
    result.euclideanBaseVoxels = euclidean.finish();
    result.normalBaseVoxels = normal.finish();
    result.tangentialBaseVoxels = tangential.finish();
    return result;
}

ReplayPolylineSet referencePolylineSet(
    const std::vector<cv::Vec3d>& points,
    double beginArc,
    double endArc)
{
    ReplayPolylineSet result;
    auto selected = slicePolylineArc(
        makePolylineArcGeometry(points), beginArc, endArc);
    if (selected.size() >= 2)
        result.polylines.push_back(makePolylineArcGeometry(selected));
    else if (selected.size() == 1)
        result.singletons.push_back(selected.front());
    finishReplayIndex(result);
    return result;
}

double maximumLineDistance(const ReplayLineDistance& distance)
{
    return distance.euclideanBaseVoxels.maximum;
}

double maximumNormalLineDistance(const ReplayLineDistance& distance)
{
    return distance.normalBaseVoxels.maximum;
}

double maximumTangentialLineDistance(const ReplayLineDistance& distance)
{
    return distance.tangentialBaseVoxels.maximum;
}

uint64_t inversionCount(const std::vector<size_t>& baselineOrder, const std::vector<size_t>& measuredOrder)
{
    std::map<size_t, size_t> baselineRank;
    for (size_t rank = 0; rank < baselineOrder.size(); ++rank)
        baselineRank.emplace(baselineOrder[rank], rank);
    std::vector<uint64_t> tree(baselineOrder.size() + 1, 0);
    const auto add = [&](size_t rawIndex) {
        for (size_t index = rawIndex + 1; index < tree.size(); index += index & (~index + 1)) {
            ++tree[index];
        }
    };
    const auto prefix = [&](size_t count) {
        uint64_t result = 0;
        for (size_t index = count; index > 0; index &= index - 1)
            result += tree[index];
        return result;
    };
    uint64_t inversions = 0;
    for (size_t measuredRank = 0; measuredRank < measuredOrder.size(); ++measuredRank) {
        const size_t rank = baselineRank.at(measuredOrder[measuredRank]);
        inversions += static_cast<uint64_t>(measuredRank) - prefix(rank + 1);
        add(rank);
    }
    return inversions;
}

std::vector<size_t> orderedCosts(const FiberletPathReport& paths, const std::vector<size_t>& successful)
{
    std::vector<size_t> result = successful;
    std::sort(result.begin(), result.end(), [&](size_t left, size_t right) {
        return std::tuple{paths.candidates[left].cost.total(), left} < std::tuple{paths.candidates[right].cost.total(), right};
    });
    return result;
}

FiberletAnchorId graphAnchorId(const QuantizedAnchorKey& key)
{
    return {key.cellZYX, key.variant};
}

cv::Vec3f compactAxis(const cv::Vec3f& axis)
{
    return quantizeFiberletDirectionForEvaluation(axis);
}

struct QuantizedAnchorLayout {
    LoadedFiberAnchorArtifact loaded;
    std::map<FiberletAnchorId, QuantizedAnchor> anchors;
    FiberletQuantizationSummary positionErrors;
    FiberletQuantizationSummary axisErrors;
};

QuantizedAnchorLayout quantizeAnchors(
    const LoadedFiberAnchorArtifact& baselineAnchors,
    const FiberletPathReport& paths,
    const std::vector<size_t>& successful,
    double quantum,
    bool useCompactAxes,
    int chunkSide,
    FiberletQuantizationScenarioReport& report)
{
    const std::uint64_t bins =
        fiberletPositionBinCountForEvaluation(chunkSide, quantum);
    if (quantum > 0) {
        report.anchorPositionBits = bins <= 256 ? 8 : bins <= 65536 ? 16 : 0;
        if (report.anchorPositionBits == 0)
            throw std::invalid_argument("anchor chunk-local position exceeds uint16");
    }

    QuantizedAnchorLayout layout;
    layout.loaded = baselineAnchors;
    const double scale = paths.grid.predictionToBaseScale;
    std::map<std::array<std::uint32_t, 3>, std::size_t>
        quantizedPositionCounts;
    SummaryBuilder positionErrors;
    SummaryBuilder axisErrors;
    for (auto& cell : layout.loaded.report.nonEmptyCells) {
        for (size_t componentIndex = 0; componentIndex < cell.components.size(); ++componentIndex) {
            auto& component = cell.components[componentIndex];
            if (!component.retained)
                continue;
            const FiberletAnchorId id{cell.cellZYX, componentIndex};
            const cv::Vec3f sourcePosition = component.anchor.positionPredictionXYZ;
            const cv::Vec3f sourceAxis = component.anchor.axisXYZ;
            QuantizedAnchor anchor;
            anchor.key = {cell.cellZYX,
                static_cast<uint8_t>(componentIndex)};
            anchor.positionPredictionXYZ = sourcePosition;
            if (quantum > 0) {
                anchor.positionPredictionXYZ =
                    quantizeFiberletPositionForEvaluation(
                        sourcePosition, paths.grid, quantum);
                std::array<std::uint32_t, 3> quantizedPosition{};
                for (size_t axis = 0; axis < 3; ++axis) {
                    quantizedPosition[axis] = std::bit_cast<std::uint32_t>(
                        anchor.positionPredictionXYZ[static_cast<int>(axis)]);
                }
                ++quantizedPositionCounts[quantizedPosition];
                positionErrors.values.push_back(length(anchor.positionPredictionXYZ - sourcePosition) * scale);
            }
            anchor.compactAxisXYZ = compactAxis(sourceAxis);
            axisErrors.values.push_back(
                useCompactAxes ? unorientedAngleDegrees(sourceAxis, anchor.compactAxisXYZ) : 0.0);
            component.anchor.positionPredictionXYZ = anchor.positionPredictionXYZ;
            component.anchor.axisXYZ = useCompactAxes ? anchor.compactAxisXYZ : sourceAxis;
            if (!layout.anchors.emplace(id, anchor).second)
                throw std::invalid_argument("duplicate original fiberlet anchor identity");
            ++report.anchors;
            report.maximumVariants = std::max(
                report.maximumVariants, componentIndex + 1);
        }
    }
    for (const auto& [position, count] : quantizedPositionCounts) {
        (void)position;
        if (count > 1)
            ++report.coincidentPositionGroups;
    }

    if (quantum > 0) {
        int64_t maximumDelta = 0;
        for (const size_t index : successful) {
            const auto& candidate = paths.candidates[index];
            const auto& start = layout.anchors.at(candidate.start).key.cellZYX;
            const auto& target = layout.anchors.at(candidate.target).key.cellZYX;
            for (size_t axis = 0; axis < 3; ++axis)
                maximumDelta = std::max(maximumDelta,
                    std::abs(static_cast<int64_t>(target[axis]) -
                        static_cast<int64_t>(start[axis])));
        }
        report.anchorDeltaBits = maximumDelta <= 127 ? 8 : maximumDelta <= 32767 ? 16 : 0;
        if (report.anchorDeltaBits == 0)
            throw std::invalid_argument("fiberlet endpoint delta exceeds int16");
    }
    layout.positionErrors = positionErrors.finish();
    layout.axisErrors = axisErrors.finish();
    return layout;
}

void validateAndRemapCandidateAnchors(
    FiberletPathReport& measured,
    const FiberletPathReport& baseline,
    const QuantizedAnchorLayout& layout,
    bool remapKeys)
{
    if (measured.candidates.size() != baseline.candidates.size())
        throw std::invalid_argument("quantized-anchor DP changed the candidate population");
    for (size_t index = 0; index < measured.candidates.size(); ++index) {
        auto& candidate = measured.candidates[index];
        const auto& expected = baseline.candidates[index];
        if (candidate.start != expected.start || candidate.target != expected.target) {
            throw std::invalid_argument("quantized-anchor DP changed candidate identity or order");
        }
        if (!remapKeys)
            continue;
        const auto start = layout.anchors.find(candidate.start);
        const auto target = layout.anchors.find(candidate.target);
        if (start == layout.anchors.end() || target == layout.anchors.end())
            throw std::invalid_argument("quantized fiberlet endpoint is unresolved");
        candidate.start = graphAnchorId(start->second.key);
        candidate.target = graphAnchorId(target->second.key);
        if (candidate.start == candidate.target)
            throw std::invalid_argument("quantized fiberlet endpoints collapse to one anchor key");
    }
}

std::map<std::array<int64_t, 3>, std::vector<size_t>> costChunks(
    const FiberletPathReport& paths,
    const std::vector<size_t>& successful,
    int chunkSide)
{
    std::map<std::array<int64_t, 3>, std::vector<size_t>> chunks;
    for (const size_t index : successful) {
        std::array<int64_t, 3> chunk{};
        const auto& candidate = paths.candidates[index];
        const float scale = static_cast<float>(paths.grid.predictionToBaseScale);
        const cv::Vec3f first = candidate.startPositionPredictionXYZ * scale;
        const cv::Vec3f second = candidate.targetPositionPredictionXYZ * scale;
        cv::Vec3f owner = first;
        if (std::tuple{second[0], second[1], second[2]} <
            std::tuple{first[0], first[1], first[2]}) {
            owner = second;
        }
        for (size_t axis = 0; axis < 3; ++axis)
            chunk[axis] = static_cast<int64_t>(std::floor(owner[axis] / chunkSide));
        chunks[chunk].push_back(index);
    }
    return chunks;
}

void quantizeCosts(
    FiberletPathReport& measured,
    const FiberletPathReport& baseline,
    const std::vector<size_t>& successful,
    int chunkSide,
    int bits,
    FiberletCostQuantizationDomain domain,
    float costDensityMaximum)
{
    if (bits != 8 && bits != 16)
        return;
    if (domain == FiberletCostQuantizationDomain::SqrtPerPredictionVoxel) {
        for (const size_t index : successful) {
            const auto& candidate = baseline.candidates[index];
            const float decoded = quantizeFiberletCostForEvaluation(
                candidate.cost.total(),
                fiberletCandidatePathLength(candidate),
                0.0F, 1.0F, bits, domain, costDensityMaximum);
            measured.candidates[index].cost =
                {decoded, 0.0F, 0.0F, 0.0F, 0.0F};
        }
        return;
    }
    const auto chunks = costChunks(baseline, successful, chunkSide);
    for (const auto& [chunk, indices] : chunks) {
        (void)chunk;
        std::vector<float> costs;
        std::vector<float> lengths;
        costs.reserve(indices.size());
        lengths.reserve(indices.size());
        for (const size_t index : indices) {
            costs.push_back(baseline.candidates[index].cost.total());
            lengths.push_back(fiberletCandidatePathLength(baseline.candidates[index]));
        }
        const auto decoded = quantizeFiberletCostsForEvaluation(costs, lengths, bits, domain);
        for (size_t local = 0; local < indices.size(); ++local)
            measured.candidates[indices[local]].cost =
                {decoded[local], 0.0F, 0.0F, 0.0F, 0.0F};
    }
}

std::pair<uint64_t, uint64_t> chunkOrderingChanges(
    const FiberletPathReport& baseline,
    const FiberletPathReport& measured,
    const std::vector<size_t>& successful,
    int chunkSide,
    FiberletCostQuantizationDomain domain)
{
    uint64_t inversions = 0;
    uint64_t pairs = 0;
    if (domain == FiberletCostQuantizationDomain::SqrtPerPredictionVoxel) {
        const auto baselineOrder = orderedCosts(baseline, successful);
        const auto measuredOrder = orderedCosts(measured, successful);
        inversions = inversionCount(baselineOrder, measuredOrder);
        pairs = successful.size() < 2 ? 0 :
            static_cast<uint64_t>(successful.size()) *
                static_cast<uint64_t>(successful.size() - 1) / 2;
        return {inversions, pairs};
    }
    for (const auto& [chunk, indices] : costChunks(baseline, successful, chunkSide)) {
        (void)chunk;
        const auto baselineOrder = orderedCosts(baseline, indices);
        const auto measuredOrder = orderedCosts(measured, indices);
        inversions += inversionCount(baselineOrder, measuredOrder);
        pairs += indices.size() < 2 ? 0 :
            static_cast<uint64_t>(indices.size()) * static_cast<uint64_t>(indices.size() - 1) / 2;
    }
    return {inversions, pairs};
}

FiberletQuantizationScenarioReport compareScenario(
    const FiberletQuantizationScenario& scenario,
    const FiberletPathReport& baselinePaths,
    const FiberletGraph& baselineGraph,
    const FiberletGraphReplayResult& baselineReplay,
    const FiberletPathReport& measuredPaths,
    const FiberletGraph& measuredGraph,
    const FiberletGraphReplayResult& measuredReplay,
    const vc::lasagna::NormalSampler& normalSampler,
    double normalWorkingToBaseScale,
    double normalThresholdBaseVoxels,
    const ReplayPolylineSet& baselineLines,
    const ReplayPolylineSet& referenceLines,
    const ReplayLineDistance& baselineReferenceDistance,
    const FiberletQuantizationProgressCallback& progress,
    bool measureLineDistance = true)
{
    FiberletQuantizationScenarioReport result;
    result.scenario = scenario;
    result.valid = true;
    result.graphNodes = measuredGraph.nodes.size();
    result.graphEdges = measuredGraph.edges.size();
    result.graphTransitions = measuredGraph.transitions.size();
    const auto baselineSuccessful = successfulCandidateIndices(baselinePaths);
    const auto measuredSuccessful = successfulCandidateIndices(measuredPaths);
    const std::set<size_t> baselineSet(baselineSuccessful.begin(), baselineSuccessful.end());
    const std::set<size_t> measuredSet(measuredSuccessful.begin(), measuredSuccessful.end());
    std::vector<size_t> commonSuccessful;
    std::set_intersection(
        baselineSet.begin(), baselineSet.end(),
        measuredSet.begin(), measuredSet.end(),
        std::back_inserter(commonSuccessful));
    result.baselineSuccessfulFiberlets = baselineSuccessful.size();
    result.scenarioSuccessfulFiberlets = measuredSuccessful.size();
    result.commonSuccessfulFiberlets = commonSuccessful.size();
    result.addedSuccessfulFiberlets = measuredSuccessful.size() - commonSuccessful.size();
    result.removedSuccessfulFiberlets = baselineSuccessful.size() - commonSuccessful.size();
    SummaryBuilder pointErrors;
    SummaryBuilder lengthErrors;
    SummaryBuilder costAbsolute;
    SummaryBuilder costRelative;
    for (const size_t index : commonSuccessful) {
        const auto& baseline = baselinePaths.candidates[index];
        const auto& measured = measuredPaths.candidates[index];
        const size_t samples = std::max(baseline.pointsPredictionXYZ.size(), measured.pointsPredictionXYZ.size());
        for (size_t point = 0; point < samples; ++point) {
            const double fraction = samples > 1 ? static_cast<double>(point) / static_cast<double>(samples - 1) : 0.0;
            pointErrors.values.push_back(length(
                samplePathFraction(measured.pointsPredictionXYZ, fraction) -
                samplePathFraction(baseline.pointsPredictionXYZ, fraction)) *
                baselinePaths.grid.predictionToBaseScale);
        }
        lengthErrors.values.push_back(std::abs(pathLength(measured.pointsPredictionXYZ) - pathLength(baseline.pointsPredictionXYZ)));
        const double baselineCost = baseline.cost.total();
        const double error = std::abs(static_cast<double>(measured.cost.total()) - baselineCost);
        costAbsolute.values.push_back(error);
        costRelative.values.push_back(error / std::max(std::abs(baselineCost), kEpsilon));
    }
    result.pathPointErrorBaseVoxels = pointErrors.finish();
    result.pathLengthErrorPredictionVoxels = lengthErrors.finish();
    result.costAbsoluteError = costAbsolute.finish();
    result.costRelativeError = costRelative.finish();

    const auto baselineOrder = orderedCosts(baselinePaths, commonSuccessful);
    const auto measuredOrder = orderedCosts(measuredPaths, commonSuccessful);
    result.costOrderingInversions = inversionCount(baselineOrder, measuredOrder);
    result.costOrderingPairs =
        commonSuccessful.size() < 2 ? 0 : static_cast<uint64_t>(commonSuccessful.size()) * static_cast<uint64_t>(commonSuccessful.size() - 1) / 2;
    result.costTopK = std::min<size_t>(100, commonSuccessful.size());
    const std::set<size_t> baselineTop(baselineOrder.begin(), baselineOrder.begin() + result.costTopK);
    for (size_t index = 0; index < result.costTopK; ++index)
        result.costTopKAgreement += baselineTop.contains(measuredOrder[index]) ? 1U : 0U;

    using TransitionKey = std::pair<DirectedCandidate, DirectedCandidate>;
    std::map<TransitionKey, const FiberletGraphTransition*> baselineTransitions;
    for (const auto& transition : baselineGraph.transitions) {
        baselineTransitions.emplace(
            TransitionKey{
                directedCandidate(baselineGraph, transition.incomingArc),
                directedCandidate(baselineGraph, transition.outgoingArc)},
            &transition);
    }
    std::map<TransitionKey, const FiberletGraphTransition*> measuredTransitions;
    for (const auto& transition : measuredGraph.transitions) {
        measuredTransitions.emplace(
            TransitionKey{
                directedCandidate(measuredGraph, transition.incomingArc),
                directedCandidate(measuredGraph, transition.outgoingArc)},
            &transition);
    }
    SummaryBuilder joinAngles;
    SummaryBuilder joinCosts;
    for (const auto& [key, baseline] : baselineTransitions) {
        const auto found = measuredTransitions.find(key);
        if (found == measuredTransitions.end()) {
            ++result.removedTransitions;
            continue;
        }
        joinAngles.values.push_back(std::abs(static_cast<double>(found->second->angleDegrees) - baseline->angleDegrees));
        joinCosts.values.push_back(std::abs(static_cast<double>(found->second->cost.total()) - baseline->cost.total()));
    }
    for (const auto& [key, measured] : measuredTransitions) {
        (void)measured;
        result.addedTransitions += baselineTransitions.contains(key) ? 0U : 1U;
    }
    result.joinAngleErrorDegrees = joinAngles.finish();
    result.joinCostAbsoluteError = joinCosts.finish();

    result.baselineReplayFailures = baselineReplay.failures.size();
    result.replayFailures = measuredReplay.failures.size();
    result.replayFailureDelta =
        static_cast<int64_t>(result.replayFailures) -
        static_cast<int64_t>(result.baselineReplayFailures);
    const double replayLength = measuredReplay.referenceEndArcBase - measuredReplay.referenceBeginArcBase;
    result.replayCompletedFraction =
        replayLength > 0.0 ? (measuredReplay.completedReferenceArcBase - measuredReplay.referenceBeginArcBase) / replayLength : 1.0;
    const auto measuredCandidates = replayCandidates(measuredReplay);
    result.replaySelectedEdges = measuredCandidates.size();
    if (measureLineDistance) {
        const auto measuredLines = replayPolylineSet(measuredReplay);
        ReplayLineDistance lineDistance;
        if (scenario.name == "baseline") {
            lineDistance.available = true;
            lineDistance.samples = 2 * lineSampleCount(baselineLines);
        } else {
            lineDistance = symmetricReplayLineDistance(
                baselineLines, measuredLines, normalSampler,
                normalWorkingToBaseScale, normalThresholdBaseVoxels,
                false, "baseline_line_distance", scenario.name, progress);
        }
        const auto scenarioReferenceDistance = scenario.name == "baseline"
            ? baselineReferenceDistance
            : directedReferenceLineDistance(
                  measuredLines, referenceLines, normalSampler,
                  normalWorkingToBaseScale, normalThresholdBaseVoxels,
                  scenario.name, progress);
        result.lineDistanceAvailable = lineDistance.available;
        result.lineDistanceSamples = lineDistance.samples;
        result.lineDistanceInvalidNormalSamples = lineDistance.invalidNormalSamples;
        result.maximumLineDistanceBaseVoxels = maximumLineDistance(lineDistance);
        result.maximumLineNormalDistanceBaseVoxels = maximumNormalLineDistance(lineDistance);
        result.maximumLineTangentialDistanceBaseVoxels = maximumTangentialLineDistance(lineDistance);
        result.baselineReferenceInvalidNormalSamples =
            baselineReferenceDistance.invalidNormalSamples;
        result.baselineReferenceDistanceBaseVoxels =
            baselineReferenceDistance.euclideanBaseVoxels;
        result.baselineReferenceNormalDistanceBaseVoxels =
            baselineReferenceDistance.normalBaseVoxels;
        result.baselineReferenceTangentialDistanceBaseVoxels =
            baselineReferenceDistance.tangentialBaseVoxels;
        result.scenarioReferenceInvalidNormalSamples =
            scenarioReferenceDistance.invalidNormalSamples;
        result.scenarioReferenceDistanceBaseVoxels =
            scenarioReferenceDistance.euclideanBaseVoxels;
        result.scenarioReferenceNormalDistanceBaseVoxels =
            scenarioReferenceDistance.normalBaseVoxels;
        result.scenarioReferenceTangentialDistanceBaseVoxels =
            scenarioReferenceDistance.tangentialBaseVoxels;
    }
    return result;
}

}  // namespace

cv::Vec3f quantizeFiberletPositionForEvaluation(
    const cv::Vec3f& positionPredictionXYZ,
    const FiberPredictionGridInfo& grid,
    double quantumBaseVoxels)
{
    validateFiberletPositionQuantumForEvaluation(quantumBaseVoxels);
    if (quantumBaseVoxels == 0.0)
        return positionPredictionXYZ;
    if (!(grid.predictionToBaseScale > 0.0) ||
        !std::isfinite(grid.predictionToBaseScale)) {
        throw std::invalid_argument(
            "fiberlet evaluation position scale is invalid");
    }
    cv::Vec3f result;
    for (size_t xyz = 0; xyz < 3; ++xyz) {
        const double base = static_cast<double>(positionPredictionXYZ[xyz]) *
            grid.predictionToBaseScale;
        if (!(base >= 0.0) || !std::isfinite(base))
            throw std::invalid_argument(
                "fiberlet evaluation position is outside nonnegative coordinates");
        const double decodedBase = quantumBaseVoxels *
            std::floor(base / quantumBaseVoxels + 0.5);
        const size_t shapeAxis = grid.shapeZYX[2 - xyz];
        if (!(decodedBase < static_cast<double>(shapeAxis) *
                grid.predictionToBaseScale)) {
            throw std::invalid_argument(
                "fiberlet evaluation position leaves the prediction volume");
        }
        result[static_cast<int>(xyz)] = static_cast<float>(
            decodedBase / grid.predictionToBaseScale);
    }
    return result;
}

void validateFiberletPositionQuantumForEvaluation(double quantumBaseVoxels)
{
    if (quantumBaseVoxels == 0.0)
        return;
    if (!(quantumBaseVoxels > 0.0) || !std::isfinite(quantumBaseVoxels)) {
        throw std::invalid_argument(
            "fiberlet evaluation position quantum must be zero or positive and finite");
    }
}

std::uint64_t fiberletPositionBinCountForEvaluation(
    int chunkSideBaseVoxels,
    double quantumBaseVoxels)
{
    validateFiberletPositionQuantumForEvaluation(quantumBaseVoxels);
    if (chunkSideBaseVoxels <= 0)
        throw std::invalid_argument(
            "fiberlet evaluation position chunk side must be positive");
    if (quantumBaseVoxels == 0.0)
        return 0;
    const double bins =
        static_cast<double>(chunkSideBaseVoxels) / quantumBaseVoxels;
    const double roundedBins = std::round(bins);
    const double tolerance = 16.0 * std::numeric_limits<double>::epsilon() *
        std::max(1.0, std::abs(bins));
    if (!(roundedBins >= 1.0) || !std::isfinite(roundedBins) ||
        std::abs(bins - roundedBins) > tolerance ||
        roundedBins > static_cast<double>(
            std::numeric_limits<std::uint64_t>::max())) {
        throw std::invalid_argument(
            "fiberlet evaluation position quantum does not form an integral chunk grid");
    }
    return static_cast<std::uint64_t>(roundedBins);
}

cv::Vec3f quantizeFiberletDirectionForEvaluation(
    const cv::Vec3f& directionXYZ)
{
    const auto encoded = vc::lasagna::encodeCompactNormalToRaw(
        {directionXYZ[0], directionXYZ[1], directionXYZ[2]});
    if (!encoded.has_value())
        throw std::invalid_argument(
            "fiberlet evaluation direction is not compact-encodable");
    const cv::Vec3d decoded = vc::lasagna::decodeCompactNormalFromRaw(
        (*encoded)[0], (*encoded)[1]);
    return normalized({static_cast<float>(decoded[0]),
        static_cast<float>(decoded[1]), static_cast<float>(decoded[2])});
}

std::vector<float> quantizeFiberletCostsForEvaluation(
    std::span<const float> costs,
    int bits)
{
    if (bits != 8 && bits != 16)
        return {costs.begin(), costs.end()};
    if (costs.empty())
        return {};
    for (const float cost : costs) {
        if (!(cost >= 0.0F) || !std::isfinite(cost))
            throw std::invalid_argument(
                "fiberlet evaluation cost is invalid");
    }
    const auto [minimumIt, maximumIt] =
        std::minmax_element(costs.begin(), costs.end());
    const float minimum = *minimumIt;
    const float maximum = *maximumIt;
    std::vector<float> result;
    result.reserve(costs.size());
    for (const float cost : costs)
        result.push_back(quantizeFiberletCostForEvaluation(
            cost, 1.0F, minimum, maximum, bits,
            FiberletCostQuantizationDomain::RawTotal));
    return result;
}

std::string_view fiberletCostQuantizationDomainName(
    FiberletCostQuantizationDomain domain)
{
    switch (domain) {
    case FiberletCostQuantizationDomain::RawTotal:
        return "raw_total";
    case FiberletCostQuantizationDomain::SqrtPerPredictionVoxel:
        return "sqrt_per_prediction_voxel";
    }
    throw std::invalid_argument("unknown fiberlet cost quantization domain");
}

float fiberletCostQuantizationValueForEvaluation(
    float totalCost,
    float pathLengthPredictionVoxels,
    FiberletCostQuantizationDomain domain,
    float costDensityMaximum)
{
    if (!(totalCost >= 0.0F) || !std::isfinite(totalCost))
        throw std::invalid_argument("fiberlet evaluation cost is invalid");
    if (domain == FiberletCostQuantizationDomain::RawTotal)
        return totalCost;
    if (domain != FiberletCostQuantizationDomain::SqrtPerPredictionVoxel ||
        !(pathLengthPredictionVoxels > 0.0F) ||
        !std::isfinite(pathLengthPredictionVoxels) ||
        !(costDensityMaximum > 0.0F) || !std::isfinite(costDensityMaximum)) {
        throw std::invalid_argument("fiberlet evaluation cost density length is invalid");
    }
    const float density = totalCost / pathLengthPredictionVoxels;
    if (!(density >= 0.0F) || !std::isfinite(density))
        throw std::invalid_argument("fiberlet evaluation cost density is invalid");
    return std::sqrt(std::min(density / costDensityMaximum, 1.0F));
}

float quantizeFiberletCostForEvaluation(
    float totalCost,
    float pathLengthPredictionVoxels,
    float minimumValue,
    float maximumValue,
    int bits,
    FiberletCostQuantizationDomain domain,
    float costDensityMaximum)
{
    if (bits != 8 && bits != 16)
        return totalCost;
    const float value = fiberletCostQuantizationValueForEvaluation(
        totalCost, pathLengthPredictionVoxels, domain,
        costDensityMaximum);
    if (domain == FiberletCostQuantizationDomain::SqrtPerPredictionVoxel) {
        minimumValue = 0.0F;
        maximumValue = 1.0F;
    }
    if (!(minimumValue >= 0.0F) || !std::isfinite(minimumValue) ||
        maximumValue < minimumValue || !std::isfinite(maximumValue) ||
        value < minimumValue || value > maximumValue) {
        throw std::invalid_argument("fiberlet evaluation cost lies outside its affine range");
    }
    const std::uint32_t levels = bits == 8 ? 255U : 65535U;
    const float scale = maximumValue == minimumValue
        ? 0.0F
        : (maximumValue - minimumValue) / static_cast<float>(levels);
    std::uint32_t code = 0;
    if (value == maximumValue) {
        code = levels;
    } else if (scale > 0.0F) {
        const float raw = (value - minimumValue) / scale;
        if (!std::isfinite(raw) || raw < 0.0F ||
            raw > static_cast<float>(levels)) {
            throw std::invalid_argument("fiberlet evaluation cost lies outside its affine range");
        }
        code = static_cast<std::uint32_t>(std::floor(raw + 0.5F));
    }
    const float decodedValue = minimumValue + scale * static_cast<float>(code);
    const float decodedTotal = domain == FiberletCostQuantizationDomain::SqrtPerPredictionVoxel
        ? decodedValue * decodedValue * costDensityMaximum *
            pathLengthPredictionVoxels
        : decodedValue;
    if (!(decodedTotal >= 0.0F) || !std::isfinite(decodedTotal))
        throw std::invalid_argument("decoded fiberlet evaluation cost is invalid");
    return decodedTotal;
}

std::vector<float> quantizeFiberletCostsForEvaluation(
    std::span<const float> costs,
    std::span<const float> pathLengthsPredictionVoxels,
    int bits,
    FiberletCostQuantizationDomain domain,
    float costDensityMaximum)
{
    if (costs.size() != pathLengthsPredictionVoxels.size())
        throw std::invalid_argument("fiberlet evaluation cost and length counts differ");
    if (costs.empty())
        return {};
    std::vector<float> values;
    values.reserve(costs.size());
    for (size_t index = 0; index < costs.size(); ++index) {
        values.push_back(fiberletCostQuantizationValueForEvaluation(
            costs[index], pathLengthsPredictionVoxels[index], domain,
            costDensityMaximum));
    }
    const auto [minimum, maximum] = std::minmax_element(values.begin(), values.end());
    std::vector<float> result;
    result.reserve(costs.size());
    for (size_t index = 0; index < costs.size(); ++index) {
        result.push_back(quantizeFiberletCostForEvaluation(
            costs[index], pathLengthsPredictionVoxels[index], *minimum, *maximum,
            bits, domain, costDensityMaximum));
    }
    return result;
}

FiberletEvaluationQuantization defaultFiberletReplayQuantization(
    int storageChunkSideBaseVoxels)
{
    if (storageChunkSideBaseVoxels <= 0)
        throw std::invalid_argument(
            "fiberlet replay storage chunk side must be positive");
    return {
        0.0,
        true,
        16,
        storageChunkSideBaseVoxels,
        FiberletCostQuantizationDomain::SqrtPerPredictionVoxel,
        256.0F,
    };
}

FiberletEvaluationQuantization exactFiberletReplayQuantization(
    int storageChunkSideBaseVoxels)
{
    if (storageChunkSideBaseVoxels <= 0)
        throw std::invalid_argument(
            "fiberlet replay storage chunk side must be positive");
    return {
        0.0,
        false,
        0,
        storageChunkSideBaseVoxels,
        FiberletCostQuantizationDomain::RawTotal,
        0.0F,
    };
}

std::vector<FiberletQuantizationScenario> standardFiberletQuantizationScenarios()
{
    std::vector<FiberletQuantizationScenario> result{
        {"baseline", 0, false, 0},
        {"position_q1", 1, false, 0},
        {"position_q2", 2, false, 0},
        {"position_q4", 4, false, 0},
        {"compact_axis", 0, true, 0},
        {"compact_axis_cost_u8", 0, true, 8},
        {"compact_axis_cost_u16", 0, true, 16},
        {"compact_axis_cost_sqrt_u16_max256", 0, true, 16,
         FiberletCostQuantizationDomain::SqrtPerPredictionVoxel, 256.0F},
        {"position_q1_compact_axis", 1, true, 0},
        {"position_q2_compact_axis", 2, true, 0},
        {"position_q4_compact_axis", 4, true, 0},
        {"cost_u8", 0, false, 8},
        {"cost_u16", 0, false, 16},
    };
    for (const int quantum : {1, 2, 4}) {
        for (const int bits : {8, 16}) {
            if (quantum == 1 && bits == 8) {
                result.push_back({
                    "position_q1_8_compact_axis_cost_sqrt_u16_max256",
                    0.125,
                    true,
                    16,
                    FiberletCostQuantizationDomain::SqrtPerPredictionVoxel,
                    256.0F,
                });
            } else {
                result.push_back({
                    "combined_q" + std::to_string(quantum) + "_axis_cost_u" + std::to_string(bits),
                    static_cast<double>(quantum), true, bits});
            }
        }
    }
    return result;
}

FiberletGeometryCacheProfile fiberletGeometryCacheProfile(
    FiberletEvaluationQuantization replayQuantization)
{
    (void)fiberletPositionBinCountForEvaluation(
        replayQuantization.storageChunkSideBaseVoxels,
        replayQuantization.positionQuantumBaseVoxels);
    const bool quantizedGeometry =
        replayQuantization.positionQuantumBaseVoxels > 0 ||
        replayQuantization.compactDirections;
    return {
        .geometry = {
            replayQuantization.positionQuantumBaseVoxels,
            replayQuantization.compactDirections,
        },
        // Existing generated geometry caches were fingerprinted by the
        // combined u8 scenarios. This is an opaque namespace compatibility
        // tag; it is deliberately absent from FiberletGeometryQuantization.
        .compatibilityCostTagBits = quantizedGeometry ? 8 : 0,
        .storageChunkSideBaseVoxels =
            replayQuantization.storageChunkSideBaseVoxels,
    };
}

std::vector<FiberletQuantizationScenarioReport> benchmarkFiberletQuantization(
    const LoadedFiberAnchorArtifact& baselineAnchors,
    const FiberletPathReport& baselinePaths,
    const std::vector<cv::Vec3d>& referencePointsBaseXYZ,
    const vc::lasagna::NormalSampler& replayNormalSampler,
    double normalWorkingToBaseScale,
    const FiberletGraphReplayConfig& replayConfig,
    const FiberletQuantizedPathExtractor& pathExtractor,
    int storageChunkSideBaseVoxels,
    std::optional<std::string> selectedScenario,
    const FiberletQuantizationProgressCallback& progress)
{
    if (!pathExtractor)
        throw std::invalid_argument("fiberlet quantization requires a path extractor");
    if (baselineAnchors.report.grid.shapeZYX != baselinePaths.grid.shapeZYX ||
        baselineAnchors.report.grid.predictionToBaseScale != baselinePaths.grid.predictionToBaseScale) {
        throw std::invalid_argument("fiberlet quantization anchors do not match baseline paths");
    }
    const auto baselineSuccessful = successfulCandidateIndices(baselinePaths);
    const FiberletGraph baselineGraph = buildFiberletGraph(baselinePaths);
    const FiberletGraphReplayResult baselineReplay =
        traceFiberletGraphReplay(baselineGraph, referencePointsBaseXYZ, replayNormalSampler, normalWorkingToBaseScale, replayConfig);
    const ReplayPolylineSet baselineLines = replayPolylineSet(baselineReplay);
    const ReplayPolylineSet referenceLines = referencePolylineSet(
        referencePointsBaseXYZ,
        baselineReplay.referenceBeginArcBase,
        baselineReplay.referenceEndArcBase);
    const ReplayLineDistance baselineReferenceDistance =
        directedReferenceLineDistance(
            baselineLines,
            referenceLines,
            replayNormalSampler,
            normalWorkingToBaseScale,
            replayConfig.errorThresholdBaseVoxels,
            "baseline",
            progress);

    auto scenarios = standardFiberletQuantizationScenarios();
    if (selectedScenario.has_value()) {
        const auto found = std::find_if(
            scenarios.begin(), scenarios.end(), [&](const auto& scenario) {
                return scenario.name == *selectedScenario;
            });
        if (found == scenarios.end())
            throw std::invalid_argument(
                "unknown fiberlet quantization scenario: " + *selectedScenario);
        const auto selected = *found;
        scenarios.clear();
        scenarios.push_back(standardFiberletQuantizationScenarios().front());
        if (selected.name != "baseline")
            scenarios.push_back(selected);
    }

    struct GeometryRun {
        bool valid = false;
        std::string reason;
        FiberletQuantizationScenarioReport layout;
        QuantizedAnchorLayout anchors;
        FiberletPathReport paths;
        FiberletGraph graph;
        FiberletGraphReplayResult replay;
        double wallSeconds = 0.0;
    };
    using GeometryKey = std::pair<double, bool>;
    std::map<GeometryKey, GeometryRun> geometries;
    GeometryRun baselineGeometry;
    baselineGeometry.valid = true;
    baselineGeometry.anchors.loaded = baselineAnchors;
    baselineGeometry.paths = baselinePaths;
    baselineGeometry.graph = baselineGraph;
    baselineGeometry.replay = baselineReplay;
    baselineGeometry.layout.anchors = baselineGraph.nodes.size();
    geometries.emplace(GeometryKey{0, false}, std::move(baselineGeometry));

    const auto geometryName = [](const GeometryKey& key) {
        if (key == GeometryKey{0, false})
            return std::string{"baseline"};
        if (key == GeometryKey{0, true})
            return std::string{"compact_axis"};
        if (!key.second)
            return std::string{"position_q"} + std::to_string(key.first);
        std::ostringstream result;
        result << std::setprecision(17) << "position_q" << key.first
               << "_compact_axis";
        return result.str();
    };
    const auto geometry = [&](const GeometryKey& key) -> GeometryRun& {
        if (const auto found = geometries.find(key); found != geometries.end())
            return found->second;
        GeometryRun run;
        const auto start = std::chrono::steady_clock::now();
        try {
            run.anchors = quantizeAnchors(
                baselineAnchors,
                baselinePaths,
                baselineSuccessful,
                key.first,
                key.second,
                storageChunkSideBaseVoxels,
                run.layout);
            run.paths = pathExtractor(run.anchors.loaded);
            validateAndRemapCandidateAnchors(
                run.paths, baselinePaths, run.anchors, key.first > 0);
            run.graph = buildFiberletGraph(run.paths);
            run.replay = traceFiberletGraphReplay(
                run.graph,
                referencePointsBaseXYZ,
                replayNormalSampler,
                normalWorkingToBaseScale,
                replayConfig);
            run.valid = true;
        } catch (const std::exception& error) {
            run.reason = error.what();
        }
        run.wallSeconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        return geometries.emplace(key, std::move(run)).first->second;
    };

    std::vector<FiberletQuantizationScenarioReport> reports;
    for (const auto& scenario : scenarios) {
        const auto start = std::chrono::steady_clock::now();
        const GeometryKey geometryKey{
            scenario.positionQuantumBaseVoxels,
            scenario.compactAxes};
        auto& source = geometry(geometryKey);
        try {
            if (!source.valid)
                throw std::invalid_argument(source.reason);
            FiberletPathReport measured = source.paths;
            const auto successful = successfulCandidateIndices(source.paths);
            quantizeCosts(
                measured,
                source.paths,
                successful,
                storageChunkSideBaseVoxels,
                scenario.costBits,
                scenario.costDomain,
                scenario.costDensityMaximum);

            FiberletGraph graph = buildFiberletGraph(measured);
            const FiberletGraphReplayResult replay =
                traceFiberletGraphReplay(graph, referencePointsBaseXYZ, replayNormalSampler, normalWorkingToBaseScale, replayConfig);
            auto result = compareScenario(
                scenario,
                baselinePaths,
                baselineGraph,
                baselineReplay,
                measured,
                graph,
                replay,
                replayNormalSampler,
                normalWorkingToBaseScale,
                replayConfig.errorThresholdBaseVoxels,
                baselineLines,
                referenceLines,
                baselineReferenceDistance,
                progress);
            result.anchorPositionBits = source.layout.anchorPositionBits;
            result.anchorDeltaBits = source.layout.anchorDeltaBits;
            result.anchors = source.layout.anchors == 0 ? baselineGraph.nodes.size() : source.layout.anchors;
            result.coincidentPositionGroups = source.layout.coincidentPositionGroups;
            result.maximumVariants = source.layout.maximumVariants;
            result.positionErrorBaseVoxels = source.anchors.positionErrors;
            result.axisErrorDegrees = source.anchors.axisErrors;
            result.geometryDpWallSeconds = source.wallSeconds;
            result.geometryReferenceScenario = geometryName(geometryKey);
            const auto geometryComparison = compareScenario(
                scenario,
                source.paths,
                source.graph,
                source.replay,
                measured,
                graph,
                replay,
                replayNormalSampler,
                normalWorkingToBaseScale,
                replayConfig.errorThresholdBaseVoxels,
                baselineLines,
                referenceLines,
                baselineReferenceDistance,
                progress,
                false);
            result.geometryCostAbsoluteError = geometryComparison.costAbsoluteError;
            result.geometryCostRelativeError = geometryComparison.costRelativeError;
            result.geometryCostOrderingInversions = geometryComparison.costOrderingInversions;
            result.geometryCostOrderingPairs = geometryComparison.costOrderingPairs;
            result.geometryCostTopK = geometryComparison.costTopK;
            result.geometryCostTopKAgreement = geometryComparison.costTopKAgreement;
            if (scenario.costBits != 0) {
                const auto [inversions, pairs] = chunkOrderingChanges(
                    source.paths,
                    measured,
                    successful,
                    storageChunkSideBaseVoxels,
                    scenario.costDomain);
                result.chunkCostOrderingInversions = inversions;
                result.chunkCostOrderingPairs = pairs;
            }
            result.wallSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            reports.push_back(std::move(result));
        } catch (const std::exception& error) {
            FiberletQuantizationScenarioReport result = source.layout;
            result.scenario = scenario;
            result.reason = error.what();
            result.geometryDpWallSeconds = source.wallSeconds;
            result.wallSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            reports.push_back(std::move(result));
        }
    }
    return reports;
}

FiberletCachedReplayComparisonReport compareFiberletCachedReplays(
    const FiberletQuantizationScenario& scenario,
    const FiberletGraphReplayResult& baseline,
    const FiberletGraphReplayResult& measured,
    const std::vector<cv::Vec3d>& referencePointsBaseXYZ,
    const vc::lasagna::NormalSampler& normalSampler,
    double normalWorkingToBaseScale,
    double normalThresholdBaseVoxels,
    const FiberletQuantizationProgressCallback& progress)
{
    const auto completedFraction = [](const FiberletGraphReplayResult& replay) {
        const double length = replay.referenceEndArcBase -
            replay.referenceBeginArcBase;
        return length > 0.0
            ? std::clamp((replay.completedReferenceArcBase -
                  replay.referenceBeginArcBase) / length, 0.0, 1.0)
            : 1.0;
    };
    FiberletCachedReplayComparisonReport result;
    result.scenario = scenario;
    result.baselineFailures = baseline.failures.size();
    result.scenarioFailures = measured.failures.size();
    result.baselineCompletedFraction = completedFraction(baseline);
    result.scenarioCompletedFraction = completedFraction(measured);

    const auto baselineLines = replayPolylineSet(baseline);
    const auto measuredLines = replayPolylineSet(measured);
    const auto referenceLines = referencePolylineSet(
        referencePointsBaseXYZ,
        std::min(baseline.referenceBeginArcBase,
            measured.referenceBeginArcBase),
        std::max(baseline.referenceEndArcBase,
            measured.referenceEndArcBase));
    const auto lineDistance = symmetricReplayLineDistance(
        baselineLines, measuredLines, normalSampler,
        normalWorkingToBaseScale, normalThresholdBaseVoxels, false,
        "line_distance", scenario.name, progress);
    result.lineDistanceAvailable = lineDistance.available;
    result.lineDistanceSamples = lineDistance.samples;
    result.lineDistanceInvalidNormalSamples =
        lineDistance.invalidNormalSamples;
    result.lineDistanceBaseVoxels = lineDistance.euclideanBaseVoxels;
    result.lineNormalDistanceBaseVoxels = lineDistance.normalBaseVoxels;
    result.lineTangentialDistanceBaseVoxels =
        lineDistance.tangentialBaseVoxels;

    const auto baselineReference = directedReferenceLineDistance(
        baselineLines, referenceLines, normalSampler,
        normalWorkingToBaseScale, normalThresholdBaseVoxels,
        "baseline", progress);
    result.baselineReferenceInvalidNormalSamples =
        baselineReference.invalidNormalSamples;
    result.baselineReferenceDistanceBaseVoxels =
        baselineReference.euclideanBaseVoxels;
    result.baselineReferenceNormalDistanceBaseVoxels =
        baselineReference.normalBaseVoxels;
    result.baselineReferenceTangentialDistanceBaseVoxels =
        baselineReference.tangentialBaseVoxels;

    const auto measuredReference = directedReferenceLineDistance(
        measuredLines, referenceLines, normalSampler,
        normalWorkingToBaseScale, normalThresholdBaseVoxels,
        scenario.name, progress);
    result.scenarioReferenceInvalidNormalSamples =
        measuredReference.invalidNormalSamples;
    result.scenarioReferenceDistanceBaseVoxels =
        measuredReference.euclideanBaseVoxels;
    result.scenarioReferenceNormalDistanceBaseVoxels =
        measuredReference.normalBaseVoxels;
    result.scenarioReferenceTangentialDistanceBaseVoxels =
        measuredReference.tangentialBaseVoxels;
    return result;
}

}  // namespace vc::fiber_tracer
