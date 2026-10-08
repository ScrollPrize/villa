#include "vc/fiber_tracer/FiberSkeletonReplay.hpp"

#include "vc/core/io/PolylineObj.hpp"
#include "vc/core/util/AtomicFile.hpp"
#include "vc/fiber_tracer/FiberReplayMetric.hpp"

#include <algorithm>
#include <fstream>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <tuple>
#include <stdexcept>

namespace vc::fiber_tracer
{
namespace
{

constexpr double kEpsilon = 1.0e-9;

cv::Vec3d normalizedOrZero(const cv::Vec3d& value)
{
    const double length = cv::norm(value);
    if (!std::isfinite(length) || !(length > kEpsilon))
        return {0.0, 0.0, 0.0};
    return value / length;
}

void requireFinitePositive(double value, const char* name)
{
    if (!std::isfinite(value) || !(value > 0.0))
        throw std::invalid_argument(
            std::string("skeleton ") + name + " must be finite and positive");
}

void requireFiniteNonNegative(double value, const char* name)
{
    if (!std::isfinite(value) || value < 0.0)
        throw std::invalid_argument(
            std::string("skeleton ") + name +
            " must be finite and non-negative");
}

const char* nodeKindName(FiberSkeletonNodeKind kind)
{
    switch (kind) {
        case FiberSkeletonNodeKind::Terminal:
            return "terminal";
        case FiberSkeletonNodeKind::Junction:
            return "junction";
        default:
            return "other";
    }
}

nlohmann::json vectorJson(const cv::Vec3d& value)
{
    return nlohmann::json::array({value[0], value[1], value[2]});
}

nlohmann::json costJson(const FiberSkeletonCostBreakdown& cost)
{
    return {{"alignment", cost.alignment},
            {"normal_smoothness", cost.normalSmoothness},
            {"tangent_smoothness", cost.tangentSmoothness},
            {"isotropic_smoothness", cost.isotropicSmoothness},
            {"spacing", cost.spacing},
            {"total", cost.total()}};
}

// One traversable piece of the routable network: an unambiguous line or a
// selected connection through an ambiguous region.
struct RouteSegment {
    bool connection = false;
    std::size_t sourceId = 0;          // line id, or connection id
    std::size_t regionId = kFiberSkeletonNoIndex;
    std::array<std::size_t, 2> portIds{
        kFiberSkeletonNoIndex, kFiberSkeletonNoIndex};
    std::array<std::size_t, 2> candidates{0, 0};
    std::vector<cv::Vec3d> pointsBaseXYZ;
};

// Every line variant and every available connection, with each sub-port's
// own continuation: a fibre leaving a connection at a sub-port enters the
// line's variant for that sub-port, and a fibre leaving a line enters the
// region's connection for it. Nothing is chosen ahead of tracing.
struct RouteNetwork {
    std::vector<RouteSegment> segments;
    std::vector<double> segmentCost;     // the element's refined cost
    std::vector<double> segmentLength;   // base voxels
    using Key = std::tuple<std::size_t, std::size_t, std::size_t>;
    // (line, end, candidate) -> every variant with that candidate at that end.
    std::map<Key, std::vector<std::size_t>> lineOptions;
    // (region, port, candidate) -> every available connection from there.
    std::map<Key, std::vector<std::size_t>> regionOptions;
};

RouteNetwork buildRouteNetwork(const FiberSkeletonModel& model)
{
    RouteNetwork network;
    // The seam centre of a wide section is a routing target only when asked;
    // otherwise no variant or connection ending on it enters the network.
    const auto usable = [&](std::size_t port, std::size_t candidate) {
        if (model.config.routeThroughCentre || port == kFiberSkeletonNoIndex)
            return true;
        const auto& flags = model.topology.ports[port].candidateIsFibre;
        return candidate >= flags.size() || flags[candidate] != 0;
    };
    const auto add = [&](RouteSegment segment, double cost) {
        network.segmentLength.push_back(fiberPolylineLength(segment.pointsBaseXYZ));
        network.segmentCost.push_back(cost);
        network.segments.push_back(std::move(segment));
        return network.segments.size() - 1;
    };
    for (const auto& line : model.topology.lines) {
        for (const auto& variant : line.variants) {
            if (variant.pointsBaseXYZ.size() < 2)
                continue;
            if (!usable(line.portIds[0], variant.candidates[0]) ||
                !usable(line.portIds[1], variant.candidates[1])) {
                continue;
            }
            RouteSegment segment;
            segment.sourceId = line.id;
            segment.portIds = line.portIds;
            segment.candidates = variant.candidates;
            segment.pointsBaseXYZ = variant.pointsBaseXYZ;
            const auto index = add(std::move(segment), variant.cost.total());
            for (std::size_t end = 0; end < 2; ++end) {
                if (line.portIds[end] != kFiberSkeletonNoIndex) {
                    network.lineOptions[{line.id, end, variant.candidates[end]}]
                        .push_back(index);
                }
            }
        }
    }
    for (const auto& choice : model.choices) {
        if (choice.portLimitExceeded)
            continue;
        const auto& region = model.topology.regions[choice.regionId];
        for (const auto& connection : choice.connections) {
            if (!connection.available || connection.pointsBaseXYZ.size() < 2)
                continue;
            if (!usable(region.portIds[connection.portIds[0]], connection.candidates[0]) ||
                !usable(region.portIds[connection.portIds[1]], connection.candidates[1])) {
                continue;
            }
            RouteSegment segment;
            segment.connection = true;
            segment.sourceId = connection.id;
            segment.regionId = choice.regionId;
            segment.portIds = {region.portIds[connection.portIds[0]],
                               region.portIds[connection.portIds[1]]};
            segment.candidates = connection.candidates;
            segment.pointsBaseXYZ = connection.pointsBaseXYZ;
            const auto index = add(std::move(segment), connection.costTotal);
            for (std::size_t end = 0; end < 2; ++end) {
                network.regionOptions[{choice.regionId,
                                       network.segments[index].portIds[end],
                                       network.segments[index].candidates[end]}]
                    .push_back(index);
            }
        }
    }
    return network;
}

// The segments a fibre may continue on after leaving `segment` at the
// sub-port (`exitPort`, `exitCandidate`): out of a line into the region
// beyond it, out of a connection into the line beyond it, or across a shared
// midpoint port into the other region.
std::vector<std::size_t> continuationsFrom(
    const RouteNetwork& network, const FiberSkeletonModel& model,
    std::size_t segment, std::size_t exitPort, std::size_t exitCandidate)
{
    std::vector<std::size_t> options;
    if (exitPort == kFiberSkeletonNoIndex)
        return options;
    const auto& current = network.segments[segment];
    const auto& port = model.topology.ports[exitPort];
    const auto take = [&](const auto& table, const RouteNetwork::Key& key) {
        const auto found = table.find(key);
        if (found != table.end())
            options.insert(options.end(), found->second.begin(), found->second.end());
    };
    if (current.connection) {
        const std::size_t other = port.shared
            ? (port.regionIds[0] == current.regionId ? port.regionIds[1]
                                                     : port.regionIds[0])
            : kFiberSkeletonNoIndex;
        if (other != kFiberSkeletonNoIndex) {
            take(network.regionOptions, {other, exitPort, exitCandidate});
        } else {
            for (const auto& line : model.topology.lines) {
                for (std::size_t end = 0; end < 2; ++end) {
                    if (line.portIds[end] == exitPort)
                        take(network.lineOptions, {line.id, end, exitCandidate});
                }
            }
        }
    } else {
        take(network.regionOptions, {port.regionIds[0], exitPort, exitCandidate});
    }
    return options;
}

// Where a segment entered at (`enterPort`, `enterCandidate`) exits.
std::pair<std::size_t, std::size_t> exitOf(
    const RouteSegment& segment, std::size_t enterPort, std::size_t enterCandidate)
{
    const bool atStart =
        segment.portIds[0] == enterPort && segment.candidates[0] == enterCandidate;
    return atStart ? std::make_pair(segment.portIds[1], segment.candidates[1])
                   : std::make_pair(segment.portIds[0], segment.candidates[0]);
}

// Receding-horizon beam search over the sub-port graph. From the sub-port
// the trace is leaving through, every continuation is expanded until the
// route has covered `horizonBase` of arc, the element that crosses the
// horizon counted pro rata, so every live route is compared over the same
// length and cost decides between them. Routes that die before the horizon
// rank after every live one, longest first. `beamWidth` partial routes are
// kept per expansion. The first element of the best route is taken; the
// search runs again at the next sub-port, so a branch that dies inside the
// horizon is never entered while another continues.
struct RouteChoice {
    std::size_t segment = kFiberSkeletonNoIndex;
    bool continues = false;
    // Cost per unit length of the best route, for comparing seeds.
    double density = std::numeric_limits<double>::infinity();
    double length = 0.0;
};

RouteChoice chooseContinuation(
    const RouteNetwork& network, const FiberSkeletonModel& model,
    const std::vector<std::size_t>& firstOptions, std::size_t enterPort,
    std::size_t enterCandidate, const std::set<std::size_t>& visited,
    double horizonBase, double requiredBase, std::size_t beamWidth)
{
    // Elements are cheap to expand but a route could in principle wander
    // through many tiny ones; this only bounds the loop.
    constexpr int kMaxElements = 64;
    struct Partial {
        std::size_t first = kFiberSkeletonNoIndex;
        std::size_t last = kFiberSkeletonNoIndex;
        std::size_t exitPort = kFiberSkeletonNoIndex;
        std::size_t exitCandidate = 0;
        double cost = 0.0;
        double length = 0.0;
        int depth = 0;
        bool alive = true;      // can still be extended
        bool complete = false;  // reached the horizon
        std::vector<std::size_t> path;
    };
    // Adds an element, cutting it at the horizon so no route is credited
    // beyond it.
    const auto extend = [&](Partial& partial, std::size_t option) {
        const double length = network.segmentLength[option];
        const double cost = network.segmentCost[option];
        const double remaining = horizonBase - partial.length;
        if (length > remaining && length > 1e-9) {
            partial.cost += cost * (remaining / length);
            partial.length = horizonBase;
            partial.complete = true;
        } else {
            partial.cost += cost;
            partial.length += length;
            partial.complete = partial.length >= horizonBase;
        }
    };
    const auto density = [](const Partial& partial) {
        return partial.length > 1e-9 ? partial.cost / partial.length
                                     : std::numeric_limits<double>::infinity();
    };
    // A route that dies after covering what the reference still needs has
    // reached the target area; the crop holds no prediction beyond it, so
    // dying there is not a dead end.
    const auto settle = [&](Partial& partial) {
        if (!partial.alive && !partial.complete && partial.length >= requiredBase)
            partial.complete = true;
    };
    // Complete routes first, by cost per unit length (equal to cost over the
    // common horizon, and still comparable where the reference end cuts a
    // route short); then routes that died, longest first, then by density.
    const auto better = [&](const Partial& a, const Partial& b) {
        if (a.complete != b.complete)
            return a.complete;
        if (a.complete)
            return density(a) < density(b);
        if (a.length != b.length)
            return a.length > b.length;
        return density(a) < density(b);
    };

    std::vector<Partial> frontier;
    for (const auto option : firstOptions) {
        if (visited.count(option) != 0)
            continue;
        Partial partial;
        partial.first = option;
        partial.last = option;
        const auto exit = exitOf(network.segments[option], enterPort, enterCandidate);
        partial.exitPort = exit.first;
        partial.exitCandidate = exit.second;
        extend(partial, option);
        partial.depth = 1;
        partial.path = {option};
        partial.alive = partial.exitPort != kFiberSkeletonNoIndex && !partial.complete;
        settle(partial);
        frontier.push_back(std::move(partial));
    }
    std::vector<Partial> finished;
    for (int depth = 1; depth < kMaxElements && !frontier.empty(); ++depth) {
        std::vector<Partial> next;
        for (auto& partial : frontier) {
            if (!partial.alive) {
                finished.push_back(std::move(partial));
                continue;
            }
            const auto options = continuationsFrom(
                network, model, partial.last, partial.exitPort, partial.exitCandidate);
            bool extended = false;
            for (const auto option : options) {
                if (visited.count(option) != 0 ||
                    std::find(partial.path.begin(), partial.path.end(), option) !=
                        partial.path.end()) {
                    continue;
                }
                Partial grown = partial;
                grown.last = option;
                const auto exit = exitOf(network.segments[option],
                                         partial.exitPort, partial.exitCandidate);
                grown.exitPort = exit.first;
                grown.exitCandidate = exit.second;
                extend(grown, option);
                ++grown.depth;
                grown.path.push_back(option);
                grown.alive = grown.exitPort != kFiberSkeletonNoIndex && !grown.complete;
                settle(grown);
                next.push_back(std::move(grown));
                extended = true;
            }
            if (!extended) {
                partial.alive = false;
                settle(partial);
                finished.push_back(std::move(partial));
            }
        }
        std::sort(next.begin(), next.end(), better);
        if (next.size() > beamWidth)
            next.resize(beamWidth);
        frontier = std::move(next);
    }
    for (auto& partial : frontier)
        finished.push_back(std::move(partial));

    RouteChoice choice;
    if (finished.empty())
        return choice;
    const auto best = std::min_element(finished.begin(), finished.end(), better);
    choice.segment = best->first;
    choice.continues = best->complete;
    choice.density = density(*best);
    choice.length = best->length;
    return choice;
}

struct SeedLocation {
    std::size_t segment = kFiberSkeletonNoIndex;
    double arcBase = 0.0;
    bool forward = true;
    cv::Vec3d pointBaseXYZ{0.0, 0.0, 0.0};
    double distanceBase = std::numeric_limits<double>::infinity();
};

// Closest point of the routable network to the reference start, with the
// traversal direction that agrees with the reference tangent.
SeedLocation findSeed(
    const RouteNetwork& network, const cv::Vec3d& referencePoint,
    const cv::Vec3d& referenceTangent)
{
    SeedLocation best;
    for (std::size_t index = 0; index < network.segments.size(); ++index) {
        const auto& points = network.segments[index].pointsBaseXYZ;
        double arc = 0.0;
        for (std::size_t vertex = 1; vertex < points.size(); ++vertex) {
            const cv::Vec3d delta = points[vertex] - points[vertex - 1];
            const double lengthSquared = delta.dot(delta);
            const double segmentLength = std::sqrt(std::max(0.0, lengthSquared));
            const double t = lengthSquared > kEpsilon
                ? std::clamp(
                      (referencePoint - points[vertex - 1]).dot(delta) /
                          lengthSquared,
                      0.0, 1.0)
                : 0.0;
            const cv::Vec3d projected = points[vertex - 1] + delta * t;
            const double distance = cv::norm(referencePoint - projected);
            if (distance < best.distanceBase) {
                best.distanceBase = distance;
                best.segment = index;
                best.arcBase = arc + segmentLength * t;
                best.pointBaseXYZ = projected;
                best.forward =
                    normalizedOrZero(delta).dot(referenceTangent) >= 0.0;
            }
            arc += segmentLength;
        }
    }
    return best;
}

double pointToPolyline(
    const cv::Vec3d& point, const std::vector<cv::Vec3d>& polyline)
{
    if (polyline.size() < 2)
        return std::numeric_limits<double>::infinity();
    double best = std::numeric_limits<double>::infinity();
    for (std::size_t index = 1; index < polyline.size(); ++index) {
        const cv::Vec3d delta = polyline[index] - polyline[index - 1];
        const double lengthSquared = delta.dot(delta);
        const double t = lengthSquared > kEpsilon
            ? std::clamp(
                  (point - polyline[index - 1]).dot(delta) / lengthSquared,
                  0.0, 1.0)
            : 0.0;
        best = std::min(
            best, cv::norm(point - (polyline[index - 1] + delta * t)));
    }
    return best;
}

void quantiles(
    std::vector<double> values, double& median, double& p95, double& maximum)
{
    if (values.empty()) {
        median = p95 = maximum = 0.0;
        return;
    }
    std::sort(values.begin(), values.end());
    const auto pick = [&](double q) {
        const auto index = static_cast<std::size_t>(
            std::clamp(q * static_cast<double>(values.size() - 1), 0.0,
                       static_cast<double>(values.size() - 1)));
        return values[index];
    };
    median = pick(0.5);
    p95 = pick(0.95);
    maximum = values.back();
}

nlohmann::json gapJson(double median, double p95, double maximum)
{
    return {{"median", median}, {"p95", p95}, {"max", maximum}};
}

}  // namespace

FiberSkeletonReferenceCheck checkFiberSkeletonReference(
    const FiberSkeletonModel& model,
    const std::vector<cv::Vec3d>& referencePointsBase,
    FiberSkeletonFieldProvider& fields)
{
    if (referencePointsBase.size() < 2)
        throw std::invalid_argument("reference check needs a polyline");
    const auto begin = std::chrono::steady_clock::now();
    FiberSkeletonReferenceCheck check;

    FiberSkeletonRefineConfig config;
    config.weights = model.config.weights;
    config.solver = model.config.solver;
    const auto resampled =
        resampleFiberPolyline(referencePointsBase, model.config.stepBase);
    check.referenceCost =
        evaluateFiberSkeletonPolylineCost(resampled, fields, config);
    const auto refined = refineFixedEndpointPolyline(resampled, fields, config);
    check.refinedPointsBaseXYZ = refined.pointsBaseXYZ;
    check.refinedCost = refined.finalCost;
    check.referenceLengthBase = refined.initialLengthBase;
    check.refinedLengthBase = refined.lengthBase;
    check.iterations = refined.iterations;
    check.converged = refined.converged;

    std::vector<double> shift;
    shift.reserve(resampled.size());
    for (std::size_t index = 0; index < resampled.size(); ++index)
        shift.push_back(cv::norm(refined.pointsBaseXYZ[index] - resampled[index]));
    quantiles(shift, check.medianShiftBase, check.p95ShiftBase,
              check.maxShiftBase);

    const auto network = buildRouteNetwork(model);
    std::vector<double> networkGap;
    std::vector<double> graphGap;
    networkGap.reserve(refined.pointsBaseXYZ.size());
    graphGap.reserve(refined.pointsBaseXYZ.size());
    for (const auto& point : refined.pointsBaseXYZ) {
        double best = std::numeric_limits<double>::infinity();
        for (const auto& segment : network.segments)
            best = std::min(best, pointToPolyline(point, segment.pointsBaseXYZ));
        networkGap.push_back(best);
        double bestGraph = std::numeric_limits<double>::infinity();
        for (const auto& edge : model.graph.edges)
            bestGraph = std::min(bestGraph, pointToPolyline(point, edge.pointsBaseXYZ));
        graphGap.push_back(bestGraph);
    }
    quantiles(networkGap, check.medianNetworkGapBase, check.p95NetworkGapBase,
              check.maxNetworkGapBase);
    quantiles(graphGap, check.medianGraphGapBase, check.p95GraphGapBase,
              check.maxGraphGapBase);
    check.seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - begin)
            .count();
    return check;
}

FiberReplayTraceResult replayFiberSkeletonRefinedGeometry(
    const std::vector<std::vector<cv::Vec3d>>& refinedSegmentsBaseXYZ,
    const std::vector<cv::Vec3d>& refinedReferenceBaseXYZ,
    const vc::lasagna::NormalSampler& normalSampler,
    double normalWorkingToBaseScale, double nominalStepBase,
    double errorThresholdBaseVoxels, double excursionGraceBaseVoxels,
    double tangentialThresholdFactor, double matchRefineSteps)
{
    if (refinedReferenceBaseXYZ.size() < 2)
        throw std::invalid_argument("refined replay needs a reference");
    const auto reference = makePolylineArcGeometry(refinedReferenceBaseXYZ);
    std::size_t emitted = 0;
    const ReplaySegmentRunner runSegment =
        [&](const PolylineArcSample&, double, int maximumSteps,
            const ReplayPointObserver& observe,
            const FiberTraceProgressCallback&) {
            if (emitted >= refinedSegmentsBaseXYZ.size())
                return std::string("reference_end");
            const auto& points = refinedSegmentsBaseXYZ[emitted++];
            int step = 0;
            double travelled = 0.0;
            for (std::size_t index = 1; index < points.size(); ++index) {
                travelled += cv::norm(points[index] - points[index - 1]);
                if (++step > maximumSteps)
                    return std::string("step_limit");
                if (!observe(points[index], travelled, step))
                    return std::string("observer_stop");
            }
            return std::string("reference_end");
        };
    // A generous step budget: the refined geometry is already committed, this
    // only re-derives its reference matches.
    return traceReferenceReplayCore(
        reference, 0.0, reference.length(), nominalStepBase, 64.0,
        errorThresholdBaseVoxels, excursionGraceBaseVoxels,
        tangentialThresholdFactor, matchRefineSteps, normalSampler,
        normalWorkingToBaseScale, runSegment, {}, {});
}

FiberSkeletonTraceRefinement refineFiberSkeletonTrace(
    const FiberSkeletonModel& model, const FiberReplayTraceResult& replay,
    const std::vector<cv::Vec3d>& refinedReferenceBaseXYZ,
    FiberSkeletonFieldProvider& fields)
{
    if (refinedReferenceBaseXYZ.size() < 2)
        throw std::invalid_argument("trace refinement needs a reference");
    const auto begin = std::chrono::steady_clock::now();
    FiberSkeletonTraceRefinement refinement;

    FiberSkeletonRefineConfig config;
    config.weights = model.config.weights;
    config.solver = model.config.solver;

    std::vector<double> before;
    std::vector<double> after;
    for (const auto& segment : replay.segments) {
        if (segment.tracePointsBase.size() < 2)
            continue;
        for (const auto& point : segment.tracePointsBase)
            before.push_back(pointToPolyline(point, refinedReferenceBaseXYZ));
        const double length = fiberPolylineLength(segment.tracePointsBase);
        if (!(length > model.config.stepBase)) {
            refinement.segmentsBaseXYZ.push_back(segment.tracePointsBase);
            for (const auto& point : segment.tracePointsBase)
                after.push_back(pointToPolyline(point, refinedReferenceBaseXYZ));
            continue;
        }
        const auto resampled =
            resampleFiberPolyline(segment.tracePointsBase, model.config.stepBase);
        const auto initial =
            evaluateFiberSkeletonPolylineCost(resampled, fields, config);
        const auto solved = refineFixedEndpointPolyline(resampled, fields, config);
        refinement.initialCost.alignment += initial.alignment;
        refinement.initialCost.normalSmoothness += initial.normalSmoothness;
        refinement.initialCost.tangentSmoothness += initial.tangentSmoothness;
        refinement.initialCost.isotropicSmoothness += initial.isotropicSmoothness;
        refinement.initialCost.spacing += initial.spacing;
        refinement.finalCost.alignment += solved.finalCost.alignment;
        refinement.finalCost.normalSmoothness += solved.finalCost.normalSmoothness;
        refinement.finalCost.tangentSmoothness += solved.finalCost.tangentSmoothness;
        refinement.finalCost.isotropicSmoothness += solved.finalCost.isotropicSmoothness;
        refinement.finalCost.spacing += solved.finalCost.spacing;
        refinement.maxIterations =
            std::max(refinement.maxIterations, solved.iterations);
        if (!solved.converged)
            ++refinement.cappedSegments;
        for (const auto& point : solved.pointsBaseXYZ)
            after.push_back(pointToPolyline(point, refinedReferenceBaseXYZ));
        refinement.segmentsBaseXYZ.push_back(solved.pointsBaseXYZ);
    }
    quantiles(before, refinement.medianBeforeBase, refinement.p95BeforeBase,
              refinement.maxBeforeBase);
    quantiles(after, refinement.medianAfterBase, refinement.p95AfterBase,
              refinement.maxAfterBase);
    refinement.seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - begin)
            .count();
    return refinement;
}

double FiberSkeletonStageTimings::total() const noexcept
{
    return thresholdSeconds + thinningSeconds + graphSeconds +
        topologySeconds + lineRefineSeconds + choiceSeconds;
}

void validateFiberSkeletonTraceConfig(const FiberSkeletonTraceConfig& config)
{
    requireFinitePositive(config.tubeRadiusBase, "tube radius");
    requireFiniteNonNegative(config.cropMarginBase, "crop margin");
    if (config.thresholdU8 < 1 || config.thresholdU8 > 255)
        throw std::invalid_argument("skeleton threshold must be in [1, 255]");
    requireFiniteNonNegative(config.junctionMergeMaxBase, "junction merge");
    requireFiniteNonNegative(config.spurPruneMaxBase, "spur prune");
    requireFinitePositive(config.entryOffsetBase, "entry offset");
    requireFinitePositive(config.sectionRadiusBase, "section radius");
    requireFinitePositive(config.searchRadiusBase, "search radius");
    requireFinitePositive(config.lookaheadLengthBase, "lookahead length");
    if (config.beamWidth < 1)
        throw std::invalid_argument("beam width must be at least one");
    requireFiniteNonNegative(config.candidateMinOffsetBase, "candidate offset");
    requireFiniteNonNegative(
        config.candidateRefineRadiusBase, "candidate refine radius");
    requireFinitePositive(config.stepBase, "refinement step");
    requireFinitePositive(config.nominalStepBase, "nominal step");
    requireFiniteNonNegative(config.weights.normal, "normal weight");
    requireFiniteNonNegative(config.weights.tangent, "tangent weight");
    requireFiniteNonNegative(config.weights.isotropic, "isotropic weight");
    requireFiniteNonNegative(config.weights.spacing, "spacing weight");
    if (config.choice.maxPorts < 2)
        throw std::invalid_argument("skeleton port limit must be at least two");
    if (config.solver.maxIterations < 1)
        throw std::invalid_argument("skeleton solver iterations must be positive");
    requireFinitePositive(config.solver.functionTolerance, "function tolerance");
    if (config.workers < 1)
        throw std::invalid_argument("skeleton workers must be at least one");
    if (!(config.nominalStepBase >= config.stepBase))
        throw std::invalid_argument(
            "skeleton nominal step must not be shorter than the refinement step");
}

FiberSkeletonModel buildFiberSkeletonModel(
    FiberSkeletonFieldProvider& fields, const FiberSkeletonTraceConfig& config)
{
    validateFiberSkeletonTraceConfig(config);
    const auto& crop = fields.crop();
    using Clock = std::chrono::steady_clock;

    FiberSkeletonModel model;
    model.config = config;

    auto mark = Clock::now();
    auto mask = thresholdFiberSkeletonCrop(
        crop, config.thresholdU8, config.minComponentVoxels);
    const std::size_t thresholdVoxels = static_cast<std::size_t>(
        std::count(mask.begin(), mask.end(), static_cast<std::uint8_t>(1)));
    model.timings.thresholdSeconds =
        std::chrono::duration<double>(Clock::now() - mark).count();

    mark = Clock::now();
    thinFiberSkeletonMask(crop.shapeZYX, mask);
    model.timings.thinningSeconds =
        std::chrono::duration<double>(Clock::now() - mark).count();

    mark = Clock::now();
    model.graph = buildFiberSkeletonGraph(crop, mask);
    model.graph.statistics.thresholdVoxels = thresholdVoxels;
    reduceFiberSkeletonGraph(
        model.graph, config.junctionMergeMaxBase, config.spurPruneMaxBase);
    model.timings.graphSeconds =
        std::chrono::duration<double>(Clock::now() - mark).count();

    mark = Clock::now();
    FiberSkeletonSplitConfig splitConfig;
    splitConfig.entryOffsetBase = config.entryOffsetBase;
    splitConfig.sectionRadiusBase = config.sectionRadiusBase;
    splitConfig.thresholdU8 = config.thresholdU8;
    splitConfig.candidateMinOffsetBase = config.candidateMinOffsetBase;
    splitConfig.candidateRefineRadiusBase = config.candidateRefineRadiusBase;
    model.topology = buildFiberSkeletonTopology(
        model.graph, crop, splitConfig, crop.predictionToBaseScale);
    model.timings.topologySeconds =
        std::chrono::duration<double>(Clock::now() - mark).count();


    mark = Clock::now();
    FiberSkeletonPathSearchConfig search;
    search.corridorRadiusBase = config.searchRadiusBase;
    search.thresholdU8 = config.thresholdU8;
    refineFiberSkeletonLines(
        model.topology, fields, search, config.routeThroughCentre,
        config.weights, config.solver, config.stepBase);
    model.timings.lineRefineSeconds =
        std::chrono::duration<double>(Clock::now() - mark).count();

    mark = Clock::now();
    auto choiceConfig = config.choice;
    choiceConfig.search = search;
    choiceConfig.routeThroughCentre = config.routeThroughCentre;
    model.choices = solveFiberSkeletonChoices(
        model.topology, model.graph, fields, config.weights, config.solver,
        choiceConfig, config.stepBase);
    model.timings.choiceSeconds =
        std::chrono::duration<double>(Clock::now() - mark).count();

    model.portLimitRegions = 0;
    for (const auto& choice : model.choices) {
        if (choice.portLimitExceeded)
            ++model.portLimitRegions;
    }
    return model;
}

FiberSkeletonReplayResult traceFiberSkeletonReplay(
    const FiberSkeletonModel& model, const FiberSkeletonReplayRequest& request,
    const vc::lasagna::NormalSampler& normalSampler,
    double normalWorkingToBaseScale, const FiberTraceProgressCallback& progress,
    const FiberReplayFailureCallback& failureCallback)
{
    if (request.referencePointsBase.size() < 2)
        throw std::invalid_argument("skeleton replay reference is degenerate");
    requireFinitePositive(normalWorkingToBaseScale, "normal working scale");
    requireFiniteNonNegative(
        request.errorThresholdBaseVoxels, "replay error threshold");
    requireFiniteNonNegative(
        request.excursionGraceBaseVoxels, "replay excursion grace");
    requireFinitePositive(
        request.tangentialThresholdFactor, "tangential threshold factor");
    requireFinitePositive(request.maxStepFactor, "max step factor");

    const auto reference = makePolylineArcGeometry(request.referencePointsBase);
    const double startArc = request.referenceBeginArcBase.value_or(0.0);
    const double endArc =
        request.referenceEndArcBase.value_or(reference.length());
    if (!(endArc > startArc + kEpsilon))
        throw std::invalid_argument("skeleton replay has no forward extent");

    const auto network = buildRouteNetwork(model);
    const double stepBase = model.config.stepBase;
    const double nominalStepBase = model.config.nominalStepBase;

    FiberSkeletonReplayResult result;
    const auto begin = std::chrono::steady_clock::now();

    const ReplaySegmentRunner runSegment =
        [&](const PolylineArcSample& start, double segmentRemainingBase,
            int maximumSteps, const ReplayPointObserver& observe,
            const FiberTraceProgressCallback& segmentProgress) {
            FiberSkeletonSegmentReport report;
            report.startReferenceArcBase = start.arc;

            auto seed =
                findSeed(network, start.point, normalizedOrZero(start.tangent));
            // Several segments can lie within the seed tolerance of the
            // reference start: the variants of one line coincide near their
            // ends, and a connection stub may end right there. The seed is
            // chosen among all of them by the same lookahead the walk uses,
            // so a seed onto a route that dies is never taken while one that
            // continues is available; among continuing ones the cheapest,
            // then the nearest.
            if (seed.segment != kFiberSkeletonNoIndex) {
                const double tolerance = std::max(
                    2.0 * model.config.stepBase, request.errorThresholdBaseVoxels);
                struct SeedOption {
                    SeedLocation location;
                    RouteChoice route;
                };
                std::vector<SeedOption> options;
                for (std::size_t index = 0; index < network.segments.size(); ++index) {
                    const auto& segment = network.segments[index];
                    double arc = 0.0;
                    SeedLocation candidate;
                    for (std::size_t vertex = 1; vertex < segment.pointsBaseXYZ.size(); ++vertex) {
                        const cv::Vec3d delta = segment.pointsBaseXYZ[vertex] - segment.pointsBaseXYZ[vertex - 1];
                        const double lengthSquared = delta.dot(delta);
                        const double segmentLength = std::sqrt(std::max(0.0, lengthSquared));
                        const double t = lengthSquared > kEpsilon
                            ? std::clamp((start.point - segment.pointsBaseXYZ[vertex - 1]).dot(delta) / lengthSquared, 0.0, 1.0)
                            : 0.0;
                        const cv::Vec3d projected = segment.pointsBaseXYZ[vertex - 1] + delta * t;
                        const double distance = cv::norm(start.point - projected);
                        if (distance < candidate.distanceBase) {
                            candidate.distanceBase = distance;
                            candidate.segment = index;
                            candidate.arcBase = arc + segmentLength * t;
                            candidate.pointBaseXYZ = projected;
                            candidate.forward = normalizedOrZero(delta).dot(normalizedOrZero(start.tangent)) >= 0.0;
                        }
                        arc += segmentLength;
                    }
                    if (candidate.segment == kFiberSkeletonNoIndex ||
                        candidate.distanceBase > seed.distanceBase + tolerance) {
                        continue;
                    }
                    // The route from here onward: this segment, entered at the
                    // end behind the walker, and its lookahead.
                    const std::size_t entryEnd = candidate.forward ? 0 : 1;
                    SeedOption option;
                    option.location = candidate;
                    option.route = chooseContinuation(
                        network, model, {index}, segment.portIds[entryEnd],
                        segment.candidates[entryEnd], {},
                        model.config.lookaheadLengthBase, segmentRemainingBase,
                        model.config.beamWidth);
                    options.push_back(std::move(option));
                }
                const auto better = [](const SeedOption& a, const SeedOption& b) {
                    if (a.route.continues != b.route.continues)
                        return a.route.continues;
                    if (!a.route.continues && a.route.length != b.route.length)
                        return a.route.length > b.route.length;
                    if (std::abs(a.route.density - b.route.density) > 1e-9)
                        return a.route.density < b.route.density;
                    return a.location.distanceBase < b.location.distanceBase;
                };
                const auto best = std::min_element(options.begin(), options.end(), better);
                if (best != options.end())
                    seed = best->location;
            }
            if (seed.segment == kFiberSkeletonNoIndex) {
                ++result.unseededSegments;
                report.terminationReason = kFiberSkeletonNoSeed;
                result.segments.push_back(report);
                return std::string(kFiberSkeletonNoSeed);
            }
            // A seed outside the acceptance ellipsoid is not tracing: it is
            // reported and reset, never credited as covered reference.
            const auto seedMeasurement = measureFiberReplayThreshold(
                seed.pointBaseXYZ, start.point, normalSampler,
                normalWorkingToBaseScale, request.errorThresholdBaseVoxels,
                request.tangentialThresholdFactor);
            if (fiberReplayThresholdExceeded(
                    seedMeasurement, request.errorThresholdBaseVoxels)) {
                ++result.unseededSegments;
                report.terminationReason = kFiberSkeletonNoSeed;
                result.segments.push_back(report);
                return std::string(kFiberSkeletonNoSeed);
            }
            report.seeded = true;

            std::string reason = kFiberSkeletonTerminal;
            std::set<std::size_t> visited;
            std::size_t segmentIndex = seed.segment;
            double segmentArc = seed.arcBase;
            bool forward = seed.forward;
            int step = 0;
            cv::Vec3d lastPoint = seed.pointBaseXYZ;
            double routed = 0.0;
            bool running = true;

            while (running) {
                if (!visited.insert(segmentIndex).second) {
                    reason = kFiberSkeletonTubeBoundary;
                    break;
                }
                const auto& segment = network.segments[segmentIndex];
                if (segment.connection)
                    ++report.visitedRegions;
                else
                    ++report.visitedLines;
                const double segmentLength =
                    fiberPolylineLength(segment.pointsBaseXYZ);
                const double endOfSegment = forward ? segmentLength : 0.0;
                const double travel = std::abs(endOfSegment - segmentArc);

                const auto emitted = static_cast<std::size_t>(
                    std::max<double>(1.0, std::ceil(travel / stepBase)));
                for (std::size_t index = 1; index <= emitted; ++index) {
                    const double fraction =
                        static_cast<double>(index) /
                        static_cast<double>(emitted);
                    const double arc =
                        segmentArc + (endOfSegment - segmentArc) * fraction;
                    const cv::Vec3d point = fiberPolylinePointAtArc(
                        segment.pointsBaseXYZ, arc);
                    routed += cv::norm(point - lastPoint);
                    lastPoint = point;
                    ++step;
                    if (step > maximumSteps) {
                        reason = "step_limit";
                        running = false;
                        break;
                    }
                    if (!observe(point, routed, step)) {
                        reason = "observer_stop";
                        running = false;
                        break;
                    }
                }
                if (!running)
                    break;
                if (segmentProgress) {
                    FiberTraceProgress event;
                    event.step = step;
                    event.maxSteps = maximumSteps;
                    event.reason = "skeleton_route";
                    segmentProgress(event);
                }

                const std::size_t exitPort =
                    forward ? segment.portIds[1] : segment.portIds[0];
                const std::size_t exitCandidate =
                    forward ? segment.candidates[1] : segment.candidates[0];
                if (exitPort == kFiberSkeletonNoIndex) {
                    reason = kFiberSkeletonTerminal;
                    break;
                }
                const auto options = continuationsFrom(
                    network, model, segmentIndex, exitPort, exitCandidate);
                // A route that reaches what the reference still asks for
                // is complete even if the network ends right after it.
                const auto chosen = chooseContinuation(
                    network, model, options, exitPort, exitCandidate, visited,
                    model.config.lookaheadLengthBase,
                    std::max(0.0, segmentRemainingBase - routed),
                    model.config.beamWidth);
                const std::size_t nextSegment = chosen.segment;
                if (nextSegment == kFiberSkeletonNoIndex) {
                    // The sub-port has no continuation on the other side. A
                    // port with no line behind it, or whose region has one
                    // port, is where the skeleton ends; a region over the
                    // port limit is unresolved; anything else is a sub-port
                    // the network does not connect.
                    const auto& exitPortRef = model.topology.ports[exitPort];
                    const std::size_t region = exitPortRef.regionIds[0];
                    bool lineBehind = false;
                    for (const auto& line : model.topology.lines) {
                        if (line.portIds[0] == exitPort || line.portIds[1] == exitPort)
                            lineBehind = true;
                    }
                    const bool singlePortRegion = region != kFiberSkeletonNoIndex &&
                        model.topology.regions[region].portIds.size() <= 1;
                    if (region != kFiberSkeletonNoIndex &&
                        region < model.choices.size() &&
                        model.choices[region].portLimitExceeded) {
                        reason = kFiberSkeletonRegionUnresolved;
                    } else if ((segment.connection && !lineBehind && !exitPortRef.shared) ||
                               (!segment.connection && singlePortRegion)) {
                        reason = kFiberSkeletonTerminal;
                    } else {
                        reason = kFiberSkeletonUnmatchedPort;
                    }
                    break;
                }
                const auto& next = network.segments[nextSegment];
                const bool enterAtStart = next.portIds[0] == exitPort &&
                    next.candidates[0] == exitCandidate;
                segmentIndex = nextSegment;
                forward = enterAtStart;
                segmentArc = enterAtStart
                    ? 0.0
                    : fiberPolylineLength(next.pointsBaseXYZ);
            }

            report.terminationReason = reason;
            report.routedLengthBase = routed;
            result.segments.push_back(report);
            return reason;
        };

    // The core derives its step budget from the nominal step, while the
    // skeleton emits at the finer refinement step; the factor is scaled so the
    // budget still covers the whole directed run.
    const double stepFactor =
        request.maxStepFactor * std::max(1.0, nominalStepBase / stepBase);
    result.replay = traceReferenceReplayCore(
        reference, startArc, endArc, nominalStepBase, stepFactor,
        request.errorThresholdBaseVoxels, request.excursionGraceBaseVoxels,
        request.tangentialThresholdFactor, request.matchRefineSteps,
        normalSampler, normalWorkingToBaseScale, runSegment, progress,
        failureCallback);
    result.replaySeconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - begin)
            .count();
    return result;
}

nlohmann::json fiberSkeletonModelJson(
    const FiberSkeletonModel& model, const FiberSkeletonCrop& crop,
    const FiberSkeletonReplayResult* replay,
    const FiberSkeletonReferenceCheck* reference,
    const FiberSkeletonTraceRefinement* traceRefinement)
{
    const auto& config = model.config;
    nlohmann::json document;
    document["format"] = "vc_skeleton_trace";
    document["version"] = 1;
    document["config"] = {
        {"tube_radius_base", config.tubeRadiusBase},
        {"crop_margin_base", config.cropMarginBase},
        {"threshold_u8", config.thresholdU8},
        {"min_component_voxels", config.minComponentVoxels},
        {"junction_merge_max_base", config.junctionMergeMaxBase},
        {"spur_prune_max_base", config.spurPruneMaxBase},
        {"entry_offset_base", config.entryOffsetBase},
        {"section_radius_base", config.sectionRadiusBase},
        {"candidate_min_offset_base", config.candidateMinOffsetBase},
        {"candidate_refine_radius_base", config.candidateRefineRadiusBase},
        {"search_radius_base", config.searchRadiusBase},
        {"route_through_centre", config.routeThroughCentre},
        {"lookahead_length_base", config.lookaheadLengthBase},
        {"beam_width", config.beamWidth},
        {"dump_crop_presence", config.dumpCropPresence},
        {"render_line_variants", config.renderLineVariants},
        {"step_base", config.stepBase},
        {"nominal_step_base", config.nominalStepBase},
        {"weights",
         {{"normal", config.weights.normal},
          {"tangent", config.weights.tangent},
          {"isotropic", config.weights.isotropic},
          {"spacing", config.weights.spacing}}},
        {"choice",
         {{"max_ports", config.choice.maxPorts}}},
        {"solver",
         {{"max_iterations", config.solver.maxIterations},
          {"function_tolerance", config.solver.functionTolerance}}},
        {"workers", config.workers}};

    document["crop"] = {
        {"origin_zyx",
         nlohmann::json::array(
             {crop.originZYX[0], crop.originZYX[1], crop.originZYX[2]})},
        {"shape_zyx",
         nlohmann::json::array(
             {crop.shapeZYX[0], crop.shapeZYX[1], crop.shapeZYX[2]})},
        {"prediction_to_base_scale", crop.predictionToBaseScale},
        {"voxels", crop.voxelCount()},
        {"scoped_voxels", crop.scopedVoxels},
        {"missing_voxels", crop.missingVoxels}};

    std::size_t unavailableConnections = 0;
    for (const auto& choice : model.choices) {
        for (const auto& connection : choice.connections) {
            if (!connection.available)
                ++unavailableConnections;
        }
    }
    std::size_t cappedLines = 0;
    for (const auto& line : model.topology.lines) {
        if (line.refined && !line.converged)
            ++cappedLines;
    }
    std::size_t cappedConnections = 0;
    for (const auto& choice : model.choices) {
        for (const auto& connection : choice.connections) {
            if (connection.refined && !connection.converged)
                ++cappedConnections;
        }
    }
    const auto& statistics = model.graph.statistics;
    document["statistics"] = {
        {"threshold_voxels", statistics.thresholdVoxels},
        {"skeleton_voxels", statistics.skeletonVoxels},
        {"skeleton_components", statistics.skeletonComponents},
        {"raw_nodes", statistics.rawNodes},
        {"raw_edges", statistics.rawEdges},
        {"graph_nodes", statistics.graphNodes},
        {"graph_edges", statistics.graphEdges},
        {"graph_components", statistics.graphComponents},
        {"terminal_nodes", statistics.terminalNodes},
        {"junction_nodes", statistics.junctionNodes},
        {"cycle_rank", statistics.cycleRank},
        {"region_groups", statistics.regionGroups},
        {"grouped_junctions", statistics.groupedJunctions},
        {"pruned_spurs", statistics.prunedSpurs},
        {"suppressed_degree_two_nodes", statistics.suppressedDegreeTwoNodes},
        {"total_edge_length_base", statistics.totalEdgeLengthBase},
        {"regions", model.topology.regions.size()},
        {"ports", model.topology.ports.size()},
        {"lines", model.topology.lines.size()},
        {"searched_initials", model.topology.searchedInitials},
        {"unsearched_initials", model.topology.unsearchedInitials},
        {"shared_midpoint_ports", model.topology.sharedMidpointPorts},
        {"dropped_short_edges", model.topology.droppedShortEdges},
        {"unavailable_connections", unavailableConnections},
        {"port_limit_regions", model.portLimitRegions},
        {"line_refinements_capped", cappedLines},
        {"connection_refinements_capped", cappedConnections}};

    document["timings_seconds"] = {
        {"threshold", model.timings.thresholdSeconds},
        {"thinning", model.timings.thinningSeconds},
        {"graph", model.timings.graphSeconds},
        {"topology", model.timings.topologySeconds},
        {"line_refine", model.timings.lineRefineSeconds},
        {"choice", model.timings.choiceSeconds},
        {"total", model.timings.total()}};

    auto nodes = nlohmann::json::array();
    for (const auto& node : model.graph.nodes) {
        nodes.push_back(
            {{"id", node.id},
             {"position_base_xyz", vectorJson(node.positionBaseXYZ)},
             {"degree", node.degree},
             {"kind", nodeKindName(node.kind)},
             {"region_group", node.regionGroup}});
    }
    document["nodes"] = std::move(nodes);

    auto edges = nlohmann::json::array();
    for (const auto& edge : model.graph.edges) {
        edges.push_back(
            {{"id", edge.id},
             {"node_ids",
              nlohmann::json::array({edge.nodeIds[0], edge.nodeIds[1]})},
             {"length_base", edge.lengthBase},
             {"points", edge.pointsBaseXYZ.size()}});
    }
    document["edges"] = std::move(edges);

    auto ports = nlohmann::json::array();
    for (const auto& port : model.topology.ports) {
        auto candidates = nlohmann::json::array();
        for (const auto& candidate : port.candidatesBaseXYZ)
            candidates.push_back(vectorJson(candidate));
        ports.push_back(
            {{"id", port.id},
             {"edge_id", port.edgeId},
             {"region_ids",
              nlohmann::json::array({port.regionIds[0], port.regionIds[1]})},
             {"position_base_xyz", vectorJson(port.positionBaseXYZ)},
             {"initial_position_base_xyz",
              vectorJson(port.initialPositionBaseXYZ)},
             {"tangent_xyz", vectorJson(port.tangentXYZ)},
             {"node_ids",
              nlohmann::json::array({port.nodeIds[0], port.nodeIds[1]})},
             {"arc_from_node_base",
              nlohmann::json::array(
                  {port.arcFromNodeBase[0], port.arcFromNodeBase[1]})},
             {"shared", port.shared},
             {"section_width_base", port.sectionWidthBase},
             {"candidate_is_fibre", port.candidateIsFibre},
             {"candidates_base_xyz", candidates}});
    }
    document["ports"] = std::move(ports);

    auto lines = nlohmann::json::array();
    for (const auto& line : model.topology.lines) {
        auto variantCosts = nlohmann::json::array();
        for (const auto& variant : line.variants) {
            variantCosts.push_back(
                {{"candidates",
                  nlohmann::json::array(
                      {variant.candidates[0], variant.candidates[1]})},
                 {"total", variant.cost.total()}});
        }
        lines.push_back(
            {{"id", line.id},
             {"edge_id", line.edgeId},
             {"port_ids",
              nlohmann::json::array({line.portIds[0], line.portIds[1]})},
             {"initial_length_base", line.initialLengthBase},
             {"length_base", line.lengthBase},
             {"refined", line.refined},
             {"iterations", line.iterations},
             {"converged", line.converged},
             {"cost", costJson(line.cost)},
             {"variants", variantCosts},
             {"entry_variants",
              nlohmann::json::array({line.entryVariants[0], line.entryVariants[1]})}});
    }
    document["lines"] = std::move(lines);

    auto regions = nlohmann::json::array();
    for (const auto& region : model.topology.regions) {
        regions.push_back(
            {{"id", region.id},
             {"group", region.group},
             {"node_ids", region.nodeIds},
             {"interior_edge_ids", region.interiorEdgeIds},
             {"center_base_xyz", vectorJson(region.centerBaseXYZ)},
             {"port_ids", region.portIds},
             {"support_points", region.supportPointsBaseXYZ.size()}});
    }
    document["regions"] = std::move(regions);

    auto choices = nlohmann::json::array();
    for (const auto& choice : model.choices) {
        auto connections = nlohmann::json::array();
        for (const auto& connection : choice.connections) {
            connections.push_back(
                {{"id", connection.id},
                 {"region_port_indices",
                  nlohmann::json::array(
                      {connection.portIds[0], connection.portIds[1]})},
                 {"item_ids",
                  nlohmann::json::array(
                      {connection.itemIds[0], connection.itemIds[1]})},
                 {"candidates",
                  nlohmann::json::array(
                      {connection.candidates[0], connection.candidates[1]})},
                 {"length_base", connection.lengthBase},
                 {"available", connection.available},
                 {"searched_initial", connection.searchedInitial},
                 {"refined", connection.refined},
                 {"iterations", connection.iterations},
                 {"converged", connection.converged},
                 {"cost", costJson(connection.cost)},
                 {"cost_total", connection.costTotal}});
        }
        auto items = nlohmann::json::array();
        for (const auto& item : choice.items)
            items.push_back({{"slot", item.slot}, {"candidate", item.candidate}});
        choices.push_back({{"region_id", choice.regionId},
                           {"port_count", choice.portCount},
                           {"items", std::move(items)},
                           {"port_limit_exceeded", choice.portLimitExceeded},
                           {"entry_connections", choice.entryConnections},
                           {"solve_seconds", choice.solveSeconds},
                           {"connections", std::move(connections)}});
    }
    document["choices"] = std::move(choices);

    if (replay != nullptr) {
        auto segments = nlohmann::json::array();
        for (const auto& segment : replay->segments) {
            segments.push_back(
                {{"start_reference_arc_base", segment.startReferenceArcBase},
                 {"seeded", segment.seeded},
                 {"termination_reason", segment.terminationReason},
                 {"visited_lines", segment.visitedLines},
                 {"visited_regions", segment.visitedRegions},
                 {"routed_length_base", segment.routedLengthBase}});
        }
        document["replay"] = {
            {"segments", std::move(segments)},
            {"unseeded_segments", replay->unseededSegments},
            {"failures", replay->replay.failures.size()},
            {"completed_reference_arc_base",
             replay->replay.completedReferenceArcBase},
            {"reference_end_arc_base", replay->replay.referenceEndArcBase},
            {"replay_seconds", replay->replaySeconds}};
    }
    if (reference != nullptr) {
        document["reference_check"] = {
            {"reference_cost", costJson(reference->referenceCost)},
            {"refined_cost", costJson(reference->refinedCost)},
            {"reference_length_base", reference->referenceLengthBase},
            {"refined_length_base", reference->refinedLengthBase},
            {"refinement_shift_base",
             gapJson(reference->medianShiftBase, reference->p95ShiftBase,
                     reference->maxShiftBase)},
            {"refined_reference_to_route_network_base",
             gapJson(reference->medianNetworkGapBase,
                     reference->p95NetworkGapBase,
                     reference->maxNetworkGapBase)},
            {"refined_reference_to_skeleton_graph_base",
             gapJson(reference->medianGraphGapBase, reference->p95GraphGapBase,
                     reference->maxGraphGapBase)},
            {"iterations", reference->iterations},
            {"converged", reference->converged},
            {"seconds", reference->seconds}};
    }
    if (traceRefinement != nullptr) {
        document["trace_refinement"] = {
            {"initial_cost", costJson(traceRefinement->initialCost)},
            {"final_cost", costJson(traceRefinement->finalCost)},
            {"max_iterations", traceRefinement->maxIterations},
            {"capped_segments", traceRefinement->cappedSegments},
            {"trace_to_refined_reference_before_base",
             gapJson(traceRefinement->medianBeforeBase,
                     traceRefinement->p95BeforeBase,
                     traceRefinement->maxBeforeBase)},
            {"trace_to_refined_reference_after_base",
             gapJson(traceRefinement->medianAfterBase,
                     traceRefinement->p95AfterBase,
                     traceRefinement->maxAfterBase)},
            {"seconds", traceRefinement->seconds}};
    }
    return document;
}

void writeFiberSkeletonArtifacts(
    const std::filesystem::path& directory, const FiberSkeletonModel& model,
    const FiberSkeletonCrop& crop, const FiberSkeletonReplayResult* replay,
    const FiberSkeletonReferenceCheck* reference,
    const FiberSkeletonTraceRefinement* traceRefinement)
{
    std::filesystem::create_directories(directory);
    vc::core::util::atomicWriteString(
        directory / "skeleton.json",
        fiberSkeletonModelJson(model, crop, replay, reference, traceRefinement)
            .dump(1));
    if (traceRefinement != nullptr && !traceRefinement->segmentsBaseXYZ.empty()) {
        std::vector<vc::core::io::NamedPolyline> refined;
        for (std::size_t index = 0;
             index < traceRefinement->segmentsBaseXYZ.size(); ++index) {
            if (traceRefinement->segmentsBaseXYZ[index].size() >= 2) {
                refined.push_back({"refined_trace_segment_" + std::to_string(index),
                                   traceRefinement->segmentsBaseXYZ[index]});
            }
        }
        if (!refined.empty()) {
            vc::core::io::writePolylinesObj(
                refined, directory / "trace_refined.obj",
                "Committed trace re-solved full length with the skeleton cost");
        }
    }
    if (reference != nullptr && reference->refinedPointsBaseXYZ.size() >= 2) {
        vc::core::io::writePolylinesObj(
            {{"refined_reference", reference->refinedPointsBaseXYZ}},
            directory / "reference_refined.obj",
            "Reference refined with the skeleton cost, endpoints fixed");
    }
    if (model.config.dumpCropPresence && !crop.presence.empty()) {
        std::ofstream stream(directory / "crop_presence.u8",
                             std::ios::binary | std::ios::trunc);
        stream.write(reinterpret_cast<const char*>(crop.presence.data()),
                     static_cast<std::streamsize>(crop.presence.size()));
    }
    writeFiberSkeletonGraphObj(directory / "skeleton_graph.obj", model.graph);
    if (!model.config.renderLineVariants.empty()) {
        std::vector<vc::core::io::NamedPolyline> variants;
        for (const auto lineId : model.config.renderLineVariants) {
            if (lineId >= model.topology.lines.size())
                continue;
            const auto& line = model.topology.lines[lineId];
            for (std::size_t index = 0; index < line.variants.size(); ++index) {
                const auto& variant = line.variants[index];
                if (variant.pointsBaseXYZ.size() < 2)
                    continue;
                bool active = false;
                for (const auto& entries : line.entryVariants)
                    active = active || std::find(entries.begin(), entries.end(), index) != entries.end();
                variants.push_back(
                    {"line_" + std::to_string(lineId) + "_variant_" +
                         std::to_string(variant.candidates[0]) + "_" +
                         std::to_string(variant.candidates[1]) +
                         (active ? "_active" : "") + "_cost_" +
                         std::to_string(variant.cost.total()),
                     variant.pointsBaseXYZ});
            }
        }
        if (!variants.empty()) {
            vc::core::io::writePolylinesObj(
                variants, directory / "skeleton_line_variants.obj",
                "Every refined variant of the requested lines");
        }
    }

    std::vector<vc::core::io::NamedPolyline> routes;
    for (const auto& line : model.topology.lines) {
        std::set<std::size_t> used;
        for (const auto& entries : line.entryVariants)
            used.insert(entries.begin(), entries.end());
        used.erase(kFiberSkeletonNoIndex);
        for (const auto index : used) {
            const auto& variant = line.variants[index];
            if (variant.pointsBaseXYZ.size() >= 2) {
                routes.push_back(
                    {"line_" + std::to_string(line.id) + "_v" +
                         std::to_string(variant.candidates[0]) +
                         std::to_string(variant.candidates[1]),
                     variant.pointsBaseXYZ});
            }
        }
    }
    for (const auto& choice : model.choices) {
        std::set<std::size_t> used(choice.entryConnections.begin(),
                                   choice.entryConnections.end());
        used.erase(kFiberSkeletonNoIndex);
        for (const auto connectionId : used) {
            const auto& connection = choice.connections[connectionId];
            if (connection.pointsBaseXYZ.size() >= 2) {
                routes.push_back(
                    {"region_" + std::to_string(choice.regionId) + "_choice_" +
                         std::to_string(connectionId),
                     connection.pointsBaseXYZ});
            }
        }
    }
    if (!routes.empty()) {
        vc::core::io::writePolylinesObj(
            routes, directory / "skeleton_routes.obj",
            "VC3D skeleton routable network");
    }

    // Every connection the regions could offer, not only the selected ones,
    // so an oracle can be run over the whole candidate network.
    std::vector<vc::core::io::NamedPolyline> candidates;
    for (const auto& choice : model.choices) {
        const auto& region = model.topology.regions[choice.regionId];
        for (const auto& connection : choice.connections) {
            if (!connection.available || connection.pointsBaseXYZ.size() < 2)
                continue;
            candidates.push_back(
                {"region_" + std::to_string(choice.regionId) + "_p" +
                     std::to_string(region.portIds[connection.portIds[0]]) +
                     "_p" +
                     std::to_string(region.portIds[connection.portIds[1]]),
                 connection.pointsBaseXYZ});
        }
    }
    if (!candidates.empty()) {
        vc::core::io::writePolylinesObj(
            candidates, directory / "skeleton_candidates.obj",
            "Every available region connection, selected or not");
    }
}

}  // namespace vc::fiber_tracer
