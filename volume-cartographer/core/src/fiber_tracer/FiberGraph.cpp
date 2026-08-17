#include "vc/fiber_tracer/FiberGraph.hpp"

#include "vc/fiber_tracer/FiberLocalScoring.hpp"
#include "vc/fiber_tracer/PolylineGeometry.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace vc::fiber_tracer
{
namespace
{

constexpr double kEpsilon = 1.0e-12;
constexpr double kPi = 3.141592653589793238462643383279502884;

double length(const cv::Vec3d& value)
{
    return std::sqrt(value.dot(value));
}

cv::Vec3d normalized(const cv::Vec3d& value)
{
    const double norm = length(value);
    if (!(norm > kEpsilon) || !std::isfinite(norm))
        return {0.0, 0.0, 0.0};
    return value / norm;
}

double angleDegrees(const cv::Vec3d& first, const cv::Vec3d& second)
{
    const cv::Vec3d a = normalized(first);
    const cv::Vec3d b = normalized(second);
    return std::acos(std::clamp(a.dot(b), -1.0, 1.0)) * 180.0 / kPi;
}

size_t arcEdge(size_t arc)
{
    return arc / 2;
}
bool arcForward(size_t arc)
{
    return arc % 2 == 0;
}

size_t arcSource(const FiberletGraph& graph, size_t arc)
{
    const auto& edge = graph.edges.at(arcEdge(arc));
    return arcForward(arc) ? edge.startNode : edge.targetNode;
}

size_t arcTarget(const FiberletGraph& graph, size_t arc)
{
    const auto& edge = graph.edges.at(arcEdge(arc));
    return arcForward(arc) ? edge.targetNode : edge.startNode;
}

cv::Vec3d arcStartDirection(const FiberletGraph& graph, size_t arc)
{
    const auto& points = graph.edges.at(arcEdge(arc)).pointsBaseXYZ;
    if (arcForward(arc)) {
        for (size_t index = 1; index < points.size(); ++index) {
            const cv::Vec3d direction = normalized(points[index] - points[0]);
            if (length(direction) > kEpsilon)
                return direction;
        }
    } else {
        for (size_t index = points.size() - 1; index > 0; --index) {
            const cv::Vec3d direction = normalized(points[index - 1] - points.back());
            if (length(direction) > kEpsilon)
                return direction;
        }
    }
    throw std::invalid_argument("fiberlet graph arc has no start tangent");
}

cv::Vec3d arcEndDirection(const FiberletGraph& graph, size_t arc)
{
    const auto& points = graph.edges.at(arcEdge(arc)).pointsBaseXYZ;
    if (arcForward(arc)) {
        for (size_t index = points.size() - 1; index > 0; --index) {
            const cv::Vec3d direction = normalized(points.back() - points[index - 1]);
            if (length(direction) > kEpsilon)
                return direction;
        }
    } else {
        for (size_t index = 1; index < points.size(); ++index) {
            const cv::Vec3d direction = normalized(points[0] - points[index]);
            if (length(direction) > kEpsilon)
                return direction;
        }
    }
    throw std::invalid_argument("fiberlet graph arc has no end tangent");
}

double arcStartLength(const FiberletGraph& graph, size_t arc)
{
    const auto& points = graph.edges.at(arcEdge(arc)).pointsBaseXYZ;
    const double scale = graph.predictionToBaseScale;
    return arcForward(arc) ? length(points[1] - points[0]) / scale : length(points[points.size() - 2] - points.back()) / scale;
}

double arcEndLength(const FiberletGraph& graph, size_t arc)
{
    const auto& points = graph.edges.at(arcEdge(arc)).pointsBaseXYZ;
    const double scale = graph.predictionToBaseScale;
    return arcForward(arc) ? length(points.back() - points[points.size() - 2]) / scale : length(points[0] - points[1]) / scale;
}

std::vector<cv::Vec3d> orientedArcPoints(const FiberletGraph& graph, size_t arc)
{
    auto points = graph.edges.at(arcEdge(arc)).pointsBaseXYZ;
    if (!arcForward(arc))
        std::reverse(points.begin(), points.end());
    return points;
}

std::optional<size_t> transitionIndex(const FiberletGraph& graph, size_t incomingArc, size_t outgoingArc)
{
    const auto found =
        std::lower_bound(graph.transitions.begin(), graph.transitions.end(), std::pair{incomingArc, outgoingArc}, [](const auto& transition, const auto& key) {
            return std::pair{transition.incomingArc, transition.outgoingArc} < key;
        });
    if (found == graph.transitions.end() || found->incomingArc != incomingArc || found->outgoingArc != outgoingArc) {
        return std::nullopt;
    }
    return static_cast<size_t>(std::distance(graph.transitions.begin(), found));
}

struct RouteCandidate {
    std::vector<size_t> arcs;
    std::vector<size_t> transitions;
    std::set<size_t> visitedNodes;
    double loss = 0.0;
    double pathLength = 0.0;
};

double routeDensity(const RouteCandidate& route)
{
    return route.pathLength > kEpsilon ? route.loss / route.pathLength : std::numeric_limits<double>::infinity();
}

bool routeLess(const RouteCandidate& left, const RouteCandidate& right)
{
    return std::tuple{routeDensity(left), left.loss, left.arcs} < std::tuple{routeDensity(right), right.loss, right.arcs};
}

void pruneRoutes(std::vector<RouteCandidate>& routes, size_t beamWidth)
{
    std::sort(routes.begin(), routes.end(), routeLess);
    if (routes.size() > beamWidth)
        routes.resize(beamWidth);
}

std::optional<RouteCandidate> bestLookaheadRoute(
    const FiberletGraph& graph,
    size_t currentNode,
    const std::optional<size_t>& incomingArc,
    const std::set<size_t>& committedVisitedNodes,
    size_t beamWidth,
    size_t lookahead,
    const std::optional<cv::Vec3d>& initialDirection)
{
    std::vector<RouteCandidate> frontier;
    for (const size_t arc : graph.nodes.at(currentNode).outgoingArcs) {
        const auto join = incomingArc.has_value() ? transitionIndex(graph, *incomingArc, arc) : std::nullopt;
        if (incomingArc.has_value() && !join.has_value())
            continue;
        if (initialDirection.has_value() && !(angleDegrees(*initialDirection, arcStartDirection(graph, arc)) < graph.maximumJoinAngleDegrees)) {
            continue;
        }
        const size_t target = arcTarget(graph, arc);
        if (committedVisitedNodes.contains(target))
            continue;
        const auto& edge = graph.edges.at(arcEdge(arc));
        RouteCandidate candidate;
        candidate.arcs.push_back(arc);
        candidate.visitedNodes = committedVisitedNodes;
        candidate.visitedNodes.insert(target);
        candidate.loss = edge.cost.total();
        candidate.pathLength = edge.pathLengthPredictionVoxels;
        if (join.has_value()) {
            candidate.transitions.push_back(*join);
            candidate.loss += graph.transitions[*join].cost.total();
        }
        frontier.push_back(std::move(candidate));
    }
    if (frontier.empty())
        return std::nullopt;
    pruneRoutes(frontier, beamWidth);

    for (size_t depth = 1; depth < lookahead; ++depth) {
        std::vector<RouteCandidate> expanded;
        for (const auto& route : frontier) {
            const size_t tailArc = route.arcs.back();
            const size_t tailNode = arcTarget(graph, tailArc);
            for (const size_t arc : graph.nodes.at(tailNode).outgoingArcs) {
                const auto join = transitionIndex(graph, tailArc, arc);
                if (!join.has_value())
                    continue;
                const size_t target = arcTarget(graph, arc);
                if (route.visitedNodes.contains(target))
                    continue;
                RouteCandidate next = route;
                next.arcs.push_back(arc);
                next.transitions.push_back(*join);
                next.visitedNodes.insert(target);
                const auto& edge = graph.edges.at(arcEdge(arc));
                next.loss += edge.cost.total() + graph.transitions[*join].cost.total();
                next.pathLength += edge.pathLengthPredictionVoxels;
                expanded.push_back(std::move(next));
            }
        }
        if (expanded.empty())
            break;
        pruneRoutes(expanded, beamWidth);
        frontier = std::move(expanded);
    }
    return *std::min_element(frontier.begin(), frontier.end(), routeLess);
}

nlohmann::json anchorIdJson(const FiberletAnchorId& id)
{
    return {{"cell_zyx", id.cellZYX}, {"component", id.componentIndex}};
}

nlohmann::json pointJson(const cv::Vec3d& point)
{
    return nlohmann::json::array({point[0], point[1], point[2]});
}

nlohmann::json costJson(const FiberletPathCost& cost)
{
    return {
        {"invalid_prediction", cost.invalidPrediction},
        {"alignment", cost.alignment},
        {"isotropic_smoothness", cost.isotropicSmoothness},
        {"tangent_smoothness", cost.tangentSmoothness},
        {"normal_smoothness", cost.normalSmoothness},
        {"total", cost.total()},
    };
}

FiberLocalMetricSample localSample(const FiberStoredPredictionSample& sample)
{
    return {
        cv::Vec3f(sample.direction),
        static_cast<float>(sample.presence),
        sample.valid,
    };
}

FiberletPathCost pathCost(const FiberLocalMetricCost& local)
{
    FiberletPathCost cost;
    cost.invalidPrediction = local.invalidPrediction;
    cost.alignment = local.alignment;
    cost.isotropicSmoothness = local.isotropicSmoothness;
    cost.tangentSmoothness = local.tangentSmoothness;
    cost.normalSmoothness = local.normalSmoothness;
    return cost;
}

bool sameAxis(const cv::Vec3d& left, const cv::Vec3d& right)
{
    const cv::Vec3d a = normalized(left);
    const cv::Vec3d b = normalized(right);
    return length(a) > kEpsilon && length(b) > kEpsilon && std::abs(a.dot(b)) >= 1.0 - 1.0e-9;
}

}  // namespace

FiberletGraph buildFiberletGraph(const FiberletPathReport& paths, double maximumJoinAngleDegrees)
{
    if (!(paths.grid.predictionToBaseScale > 0.0) || !std::isfinite(paths.grid.predictionToBaseScale) || !(maximumJoinAngleDegrees >= 0.0) ||
        !(maximumJoinAngleDegrees <= 180.0) || !std::isfinite(maximumJoinAngleDegrees) || paths.anchorCellSizePredictionVoxels < 1) {
        throw std::invalid_argument("fiberlet graph configuration is invalid");
    }
    FiberletGraph graph;
    graph.predictionToBaseScale = paths.grid.predictionToBaseScale;
    graph.anchorCellSizePredictionVoxels = paths.anchorCellSizePredictionVoxels;
    graph.maximumJoinAngleDegrees = maximumJoinAngleDegrees;
    std::map<FiberletAnchorId, size_t> nodeByAnchor;
    const auto ensureNode =
        [&](const FiberletAnchorId& id, const cv::Vec3d& positionPrediction, const FiberStoredPredictionSample& prediction, const cv::Vec3d& normal, bool normalValid) {
            const auto [it, inserted] = nodeByAnchor.emplace(id, graph.nodes.size());
            const cv::Vec3d positionBase = positionPrediction * graph.predictionToBaseScale;
            if (inserted) {
                graph.nodes.push_back({id, positionBase, prediction, normal, normalValid, {}});
            } else if (length(graph.nodes[it->second].positionBaseXYZ - positionBase) > 1.0e-9) {
                throw std::invalid_argument("fiberlet graph anchor identity has inconsistent positions");
            } else {
                const auto& existing = graph.nodes[it->second];
                if (existing.prediction.valid != prediction.valid ||
                    (prediction.valid && (!sameAxis(existing.prediction.direction, prediction.direction) ||
                                          std::abs(existing.prediction.presence - prediction.presence) > 1.0e-9)) ||
                    existing.normalValid != normalValid || (normalValid && !sameAxis(existing.normalXYZ, normal))) {
                    throw std::invalid_argument("fiberlet graph anchor identity has inconsistent scoring samples");
                }
            }
            return it->second;
        };

    const auto visual = fiberletPathVisualMetrics(paths);
    std::map<size_t, FiberletPathVisualMetric> metricByCandidate;
    for (const auto& metric : visual.paths)
        metricByCandidate.emplace(metric.candidateIndex, metric);
    for (size_t candidateIndex = 0; candidateIndex < paths.candidates.size(); ++candidateIndex) {
        const auto& candidate = paths.candidates[candidateIndex];
        if (!candidate.success)
            continue;
        if (!candidate.scoreValid || candidate.pointsPredictionXYZ.size() < 2 || !(candidate.cost.total() >= 0.0) ||
            !std::isfinite(candidate.cost.total())) {
            throw std::invalid_argument("successful fiberlet is incomplete for graph construction");
        }
        const size_t start =
            ensureNode(candidate.start, candidate.startPositionPredictionXYZ, candidate.startPrediction, candidate.startNormalXYZ, candidate.startNormalValid);
        const size_t target = ensureNode(
            candidate.target, candidate.targetPositionPredictionXYZ, candidate.targetPrediction, candidate.targetNormalXYZ, candidate.targetNormalValid);
        FiberletGraphEdge edge;
        edge.candidateIndex = candidateIndex;
        edge.startNode = start;
        edge.targetNode = target;
        edge.cost = candidate.cost;
        const auto metric = metricByCandidate.find(candidateIndex);
        if (metric == metricByCandidate.end())
            throw std::invalid_argument("successful fiberlet has no visual metric");
        edge.pathLengthPredictionVoxels = metric->second.pathLengthPredictionVoxels;
        if (!(edge.pathLengthPredictionVoxels > kEpsilon))
            throw std::invalid_argument("successful fiberlet has zero graph length");
        edge.pointsBaseXYZ.reserve(candidate.pointsPredictionXYZ.size());
        for (const auto& point : candidate.pointsPredictionXYZ) {
            edge.pointsBaseXYZ.push_back(point * graph.predictionToBaseScale);
        }
        if (length(edge.pointsBaseXYZ.front() - graph.nodes[start].positionBaseXYZ) > 1.0e-9 ||
            length(edge.pointsBaseXYZ.back() - graph.nodes[target].positionBaseXYZ) > 1.0e-9) {
            throw std::invalid_argument("fiberlet graph path endpoints do not match its anchors");
        }
        const size_t edgeIndex = graph.edges.size();
        graph.edges.push_back(std::move(edge));
        graph.nodes[start].outgoingArcs.push_back(edgeIndex * 2);
        graph.nodes[target].outgoingArcs.push_back(edgeIndex * 2 + 1);
    }

    for (auto& node : graph.nodes)
        std::sort(node.outgoingArcs.begin(), node.outgoingArcs.end());
    for (size_t node = 0; node < graph.nodes.size(); ++node) {
        const auto& arcs = graph.nodes[node].outgoingArcs;
        for (const size_t incomingArc : arcs) {
            const size_t directedIncoming = incomingArc ^ 1U;
            const cv::Vec3d incomingDirection = arcEndDirection(graph, directedIncoming);
            for (const size_t outgoingArc : arcs) {
                if (arcEdge(directedIncoming) == arcEdge(outgoingArc))
                    continue;
                const double angle = angleDegrees(incomingDirection, arcStartDirection(graph, outgoingArc));
                if (angle < maximumJoinAngleDegrees && graph.nodes[node].prediction.valid) {
                    const double incomingLength = arcEndLength(graph, directedIncoming);
                    const double outgoingLength = arcStartLength(graph, outgoingArc);
                    const auto sample = localSample(graph.nodes[node].prediction);
                    const auto cost = fiberLocalMetricCost(
                        &sample,
                        sample,
                        cv::Vec3f(incomingDirection),
                        static_cast<float>(incomingLength),
                        cv::Vec3f(arcStartDirection(graph, outgoingArc)),
                        static_cast<float>(outgoingLength),
                        cv::Vec3f(graph.nodes[node].normalXYZ),
                        graph.nodes[node].normalValid,
                        FiberLocalMetricConfig{
                            static_cast<float>(paths.config.invalidPredictionCostPerVoxel),
                            FiberLocalSmoothnessConfig{
                                static_cast<float>(paths.config.smoothnessWeight),
                                static_cast<float>(paths.config.smoothnessNormalWeight),
                                static_cast<float>(paths.config.smoothnessTangentWeight),
                                static_cast<float>(paths.config.smoothnessFreeAngleDegrees * kPi / 180.0)}});
                    graph.transitions.push_back({directedIncoming, outgoingArc, angle, incomingLength, outgoingLength, pathCost(cost)});
                }
            }
        }
    }
    std::sort(graph.transitions.begin(), graph.transitions.end(), [](const auto& left, const auto& right) {
        return std::tuple{left.incomingArc, left.outgoingArc} < std::tuple{right.incomingArc, right.outgoingArc};
    });
    return graph;
}

FiberletGraphReplayResult traceFiberletGraphReplay(
    const FiberletGraph& graph,
    const std::vector<cv::Vec3d>& referencePointsBaseXYZ,
    const vc::lasagna::NormalSampler& normalSampler,
    double normalWorkingToBaseScale,
    const FiberletGraphReplayConfig& config,
    const FiberReplayFailureCallback& failureCallback)
{
    if (config.beamWidth < 1 || config.lookaheadEdges < 1 || !(config.errorThresholdBaseVoxels >= 0.0) ||
        !std::isfinite(config.errorThresholdBaseVoxels) || !(config.matchRefineSteps >= 0.0) || !std::isfinite(config.matchRefineSteps) ||
        !(config.minimumResetAdvanceBaseVoxels > 0.0) || !std::isfinite(config.minimumResetAdvanceBaseVoxels) ||
        !(config.referenceBeginArcBase >= 0.0) || !std::isfinite(config.referenceBeginArcBase) ||
        !(normalWorkingToBaseScale > 0.0) ||
        !std::isfinite(normalWorkingToBaseScale) ||
        (config.referenceEndArcBase.has_value() &&
         !std::isfinite(*config.referenceEndArcBase))) {
        throw std::invalid_argument("fiberlet graph replay configuration is invalid");
    }
    const auto reference = makePolylineArcGeometry(referencePointsBaseXYZ);
    if (config.referenceEndArcBase.has_value() &&
        *config.referenceEndArcBase > reference.length() + kEpsilon) {
        throw std::invalid_argument(
            "fiberlet graph replay reference end exceeds the reference");
    }
    const double referenceEndArcBase = config.referenceEndArcBase.has_value()
        ? *config.referenceEndArcBase
        : reference.length();
    if (config.referenceBeginArcBase >= referenceEndArcBase - kEpsilon)
        throw std::invalid_argument("fiberlet graph replay has no usable reference interval");
    FiberletGraphReplayResult result;
    result.referenceBeginArcBase = config.referenceBeginArcBase;
    result.referenceEndArcBase = referenceEndArcBase;
    result.completedReferenceArcBase = config.referenceBeginArcBase;
    const double intervalLength =
        referenceEndArcBase - config.referenceBeginArcBase;
    const size_t maximumSegments = static_cast<size_t>(std::ceil(intervalLength / config.minimumResetAdvanceBaseVoxels)) + 2;
    const double seedWindowBase =
        std::max(config.minimumResetAdvanceBaseVoxels, static_cast<double>(graph.anchorCellSizePredictionVoxels) * graph.predictionToBaseScale);
    const double seedBroadPhaseBase =
        fiberReplayTangentialThresholdBaseVoxels(
            config.errorThresholdBaseVoxels);
    std::set<size_t> consumedNodes;

    struct Seed {
        size_t node = 0;
        PolylineArcProjection projection;
        FiberReplayThresholdMeasurement thresholdMeasurement;
    };
    const auto selectSeed = [&](double resetArc) -> std::optional<Seed> {
        std::optional<Seed> selected;
        for (size_t node = 0; node < graph.nodes.size(); ++node) {
            if (consumedNodes.contains(node))
                continue;
            const auto projection = projectPointToPolylineArc(
                reference, graph.nodes[node].positionBaseXYZ, resetArc,
                referenceEndArcBase);
            if (projection.arc + kEpsilon < resetArc ||
                projection.distance > seedBroadPhaseBase)
                continue;
            const auto thresholdMeasurement = measureFiberReplayThreshold(
                graph.nodes[node].positionBaseXYZ, projection.point,
                normalSampler, normalWorkingToBaseScale,
                config.errorThresholdBaseVoxels);
            if (fiberReplayThresholdExceeded(
                    thresholdMeasurement,
                    config.errorThresholdBaseVoxels)) {
                continue;
            }
            const cv::Vec3d tangent = samplePolylineArc(reference, projection.arc).tangent;
            const bool aligned = std::any_of(graph.nodes[node].outgoingArcs.begin(), graph.nodes[node].outgoingArcs.end(), [&](size_t arc) {
                return angleDegrees(tangent, arcStartDirection(graph, arc)) < graph.maximumJoinAngleDegrees;
            });
            if (!aligned)
                continue;
            if (!selected.has_value() ||
                std::tuple{
                    projection.arc,
                    thresholdMeasurement.thresholdErrorRatio,
                    node} <
                    std::tuple{
                        selected->projection.arc,
                        selected->thresholdMeasurement.thresholdErrorRatio,
                        selected->node}) {
                selected = Seed{node, projection, thresholdMeasurement};
            }
        }
        return selected;
    };
    const auto appendFailure = [&](FiberReplayFailure event) {
        event.index = result.failures.size();
        event.referenceArcFraction = std::clamp(
            (event.referenceArcBase - result.referenceBeginArcBase) /
                intervalLength,
            0.0, 1.0);
        event.referencePointBase = samplePolylineArc(reference, event.referenceArcBase).point;
        result.failures.push_back(std::move(event));
        if (failureCallback)
            failureCallback(result.failures.back());
    };

    double resetArc = config.referenceBeginArcBase;
    for (size_t iteration = 0; iteration < maximumSegments &&
         resetArc < referenceEndArcBase - kEpsilon; ++iteration) {
        const auto seed = selectSeed(resetArc);
        if (!seed.has_value()) {
            FiberletGraphReplaySegment empty;
            empty.startReferenceArcBase = resetArc;
            empty.endReferenceArcBase = referenceEndArcBase;
            empty.terminationReason = "no_usable_seed_for_remaining_reference";
            const size_t segmentIndex = result.segments.size();
            result.segments.push_back(std::move(empty));
            appendFailure({
                0,
                segmentIndex,
                "no_usable_seed_for_remaining_reference",
                resetArc,
            });
            result.completedReferenceArcBase = referenceEndArcBase;
            break;
        }
        if (seed->projection.arc > resetArc + seedWindowBase + kEpsilon) {
            FiberletGraphReplaySegment gap;
            gap.startReferenceArcBase = resetArc;
            gap.endReferenceArcBase = seed->projection.arc;
            gap.terminationReason = "missing_seed_gap";
            const size_t segmentIndex = result.segments.size();
            result.segments.push_back(std::move(gap));
            appendFailure({0, segmentIndex, "missing_seed_gap", resetArc});
        }

        FiberletGraphReplaySegment segment;
        segment.startReferenceArcBase = seed->projection.arc;
        segment.endReferenceArcBase = seed->projection.arc;
        segment.routePointsBaseXYZ.push_back(graph.nodes[seed->node].positionBaseXYZ);
        segment.matches.push_back({
            0,
            seed->projection.arc,
            seed->projection.arc,
            seed->projection.point,
            resetArc,
            seed->projection.arc,
            seed->thresholdMeasurement,
        });
        size_t currentNode = seed->node;
        std::optional<size_t> incomingArc;
        std::set<size_t> visitedNodes{currentNode};
        double previousReferenceArc = seed->projection.arc;
        std::optional<FiberReplayFailure> distanceFailure;
        bool referenceExhausted =
            previousReferenceArc >= referenceEndArcBase - kEpsilon;
        bool terminalPartialEdge = false;

        while (!distanceFailure.has_value() &&
               previousReferenceArc < referenceEndArcBase - kEpsilon) {
            const cv::Vec3d startDirection = samplePolylineArc(reference, previousReferenceArc).tangent;
            const auto selected =
                bestLookaheadRoute(graph, currentNode, incomingArc, visitedNodes, config.beamWidth, config.lookaheadEdges, incomingArc.has_value() ? std::nullopt : std::make_optional(startDirection));
            if (!selected.has_value())
                break;
            const size_t arc = selected->arcs.front();
            const auto& edge = graph.edges.at(arcEdge(arc));
            const auto points = orientedArcPoints(graph, arc);
            for (size_t index = 1; index < points.size(); ++index) {
                const double stepBase = length(points[index] - segment.routePointsBaseXYZ.back());
                const auto forwardMatch = matchForwardPolylinePoint(
                    reference, points[index], previousReferenceArc, stepBase,
                    config.matchRefineSteps, referenceEndArcBase);
                const auto& match = forwardMatch.projection;
                const auto thresholdMeasurement = measureFiberReplayThreshold(
                    points[index], match.point, normalSampler,
                    normalWorkingToBaseScale,
                    config.errorThresholdBaseVoxels);
                segment.routePointsBaseXYZ.push_back(points[index]);
                segment.matches.push_back({
                    segment.routePointsBaseXYZ.size() - 1,
                    forwardMatch.predictedArc,
                    match.arc,
                    match.point,
                    previousReferenceArc,
                    forwardMatch.searchEndArc,
                    thresholdMeasurement,
                });
                previousReferenceArc = match.arc;
                segment.endReferenceArcBase = match.arc;
                if (!distanceFailure.has_value() &&
                    fiberReplayThresholdExceeded(
                        thresholdMeasurement,
                        config.errorThresholdBaseVoxels)) {
                    FiberReplayFailure event;
                    event.segmentIndex = result.segments.size();
                    event.reason = "distance_above_threshold";
                    event.referenceArcBase = match.arc;
                    event.evaluatorPointBase = points[index];
                    event.segmentPointIndex = segment.routePointsBaseXYZ.size() - 1;
                    event.candidateIndex = edge.candidateIndex;
                    event.arcIndex = arc;
                    event.candidatePathPointIndex = index;
                    event.thresholdMeasurement = thresholdMeasurement;
                    distanceFailure = std::move(event);
                }
                if (previousReferenceArc >= referenceEndArcBase - kEpsilon) {
                    referenceExhausted = true;
                    terminalPartialEdge = index + 1 < points.size();
                    break;
                }
            }
            segment.candidateIndices.push_back(edge.candidateIndex);
            segment.arcIndices.push_back(arc);
            segment.edgeCost += edge.cost;
            segment.totalLoss += edge.cost.total();
            if (incomingArc.has_value()) {
                const auto join = transitionIndex(graph, *incomingArc, arc);
                if (!join.has_value())
                    throw std::logic_error("selected fiberlet route has no graph transition");
                segment.transitionIndices.push_back(*join);
                segment.transitionCost += graph.transitions[*join].cost;
                segment.totalLoss += graph.transitions[*join].cost.total();
            }
            segment.pathLengthPredictionVoxels += edge.pathLengthPredictionVoxels;
            if (!terminalPartialEdge) {
                incomingArc = arc;
                currentNode = arcTarget(graph, arc);
                visitedNodes.insert(currentNode);
            }
        }
        segment.terminalPartialEdge = terminalPartialEdge;
        if (!terminalPartialEdge)
            segment.stopNodeIndex = currentNode;
        consumedNodes.insert(visitedNodes.begin(), visitedNodes.end());

        if (referenceExhausted && !distanceFailure.has_value()) {
            segment.endReferenceArcBase = referenceEndArcBase;
            segment.terminationReason = "reference_end";
            result.segments.push_back(std::move(segment));
            result.completedReferenceArcBase = referenceEndArcBase;
            break;
        }

        const double failureArc = distanceFailure.has_value() ? distanceFailure->referenceArcBase : previousReferenceArc;
        if (distanceFailure.has_value()) {
            segment.terminationReason = "distance_above_threshold";
        } else {
            segment.terminationReason = "graph_exhausted";
            FiberReplayFailure event;
            event.segmentIndex = result.segments.size();
            event.reason = "graph_exhausted";
            event.referenceArcBase = failureArc;
            if (!segment.routePointsBaseXYZ.empty()) {
                event.evaluatorPointBase = segment.routePointsBaseXYZ.back();
                event.segmentPointIndex = segment.routePointsBaseXYZ.size() - 1;
                if (!segment.matches.empty()) {
                    event.thresholdMeasurement =
                        segment.matches.back().thresholdMeasurement;
                }
            }
            distanceFailure = std::move(event);
        }
        result.segments.push_back(std::move(segment));
        appendFailure(std::move(*distanceFailure));
        resetArc = std::min(
            referenceEndArcBase,
            std::max(failureArc,
                result.segments.back().startReferenceArcBase +
                    config.minimumResetAdvanceBaseVoxels));
        if (!(resetArc > result.segments.back().startReferenceArcBase + kEpsilon))
            throw std::logic_error("fiberlet graph replay reset did not advance");
        result.completedReferenceArcBase = resetArc;
    }
    if (result.completedReferenceArcBase < referenceEndArcBase - kEpsilon)
        throw std::logic_error("fiberlet graph replay exceeded its deterministic reset bound");
    return result;
}

nlohmann::json fiberletGraphJson(const FiberletGraph& graph)
{
    nlohmann::json root = {
        {"format", "vc_fiberlet_graph"},
        {"version", 1},
        {"coordinates",
         {
             {"position_order", "XYZ"},
             {"position_space", "base_volume"},
             {"prediction_to_base_scale", graph.predictionToBaseScale},
             {"anchor_cell_size_prediction_voxels", graph.anchorCellSizePredictionVoxels},
         }},
        {"maximum_join_angle_degrees", graph.maximumJoinAngleDegrees},
        {"nodes", nlohmann::json::array()},
        {"edges", nlohmann::json::array()},
        {"transitions", nlohmann::json::array()},
    };
    for (const auto& node : graph.nodes) {
        root["nodes"].push_back({
            {"anchor", anchorIdJson(node.anchor)},
            {"position_base_xyz", pointJson(node.positionBaseXYZ)},
            {"outgoing_arcs", node.outgoingArcs},
            {"prediction_direction_xyz", pointJson(node.prediction.direction)},
            {"prediction_presence", node.prediction.presence},
            {"prediction_valid", node.prediction.valid},
            {"normal_xyz", pointJson(node.normalXYZ)},
            {"normal_valid", node.normalValid},
        });
    }
    for (size_t edgeIndex = 0; edgeIndex < graph.edges.size(); ++edgeIndex) {
        const auto& edge = graph.edges[edgeIndex];
        nlohmann::json points = nlohmann::json::array();
        for (const auto& point : edge.pointsBaseXYZ)
            points.push_back(pointJson(point));
        root["edges"].push_back({
            {"edge_id", edgeIndex},
            {"forward_arc", edgeIndex * 2},
            {"reverse_arc", edgeIndex * 2 + 1},
            {"candidate_index", edge.candidateIndex},
            {"start_node", edge.startNode},
            {"target_node", edge.targetNode},
            {"path_length_prediction_voxels", edge.pathLengthPredictionVoxels},
            {"cost", costJson(edge.cost)},
            {"points_base_xyz", std::move(points)},
        });
    }
    for (const auto& transition : graph.transitions) {
        root["transitions"].push_back({
            {"incoming_arc", transition.incomingArc},
            {"outgoing_arc", transition.outgoingArc},
            {"angle_degrees", transition.angleDegrees},
            {"incoming_length_prediction_voxels", transition.incomingLengthPredictionVoxels},
            {"outgoing_length_prediction_voxels", transition.outgoingLengthPredictionVoxels},
            {"cost", costJson(transition.cost)},
        });
    }
    return root;
}

std::string fiberletGraphReplayObj(const FiberletGraphReplayResult& replay)
{
    std::ostringstream output;
    output << "# vc_fiberlet_graph_replay version 2\n";
    output << std::setprecision(std::numeric_limits<double>::max_digits10);
    size_t vertexOffset = 0;
    for (size_t segmentIndex = 0; segmentIndex < replay.segments.size(); ++segmentIndex) {
        const auto& segment = replay.segments[segmentIndex];
        if (segment.routePointsBaseXYZ.empty())
            continue;
        output << "g segment_" << segmentIndex << '\n';
        for (const auto& point : segment.routePointsBaseXYZ)
            output << "v " << point[0] << ' ' << point[1] << ' ' << point[2] << '\n';
        output << "l";
        if (segment.routePointsBaseXYZ.size() == 1) {
            output << ' ' << vertexOffset + 1 << ' ' << vertexOffset + 1;
        } else {
            for (size_t index = 0; index < segment.routePointsBaseXYZ.size(); ++index)
                output << ' ' << vertexOffset + index + 1;
        }
        output << '\n';
        vertexOffset += segment.routePointsBaseXYZ.size();
    }
    return output.str();
}

nlohmann::json fiberletGraphReplayJson(const FiberletGraphReplayResult& replay, const FiberletGraphReplayConfig& config)
{
    nlohmann::json root = {
        {"format", "vc_fiberlet_graph_replay"},
        {"version", 2},
        {"coordinates",
         {
             {"position_order", "XYZ"},
             {"position_space", "base_volume"},
             {"distance_unit", "base_voxels"},
         }},
        {"config",
         {
             {"beam_width", config.beamWidth},
             {"lookahead_edges", config.lookaheadEdges},
             {"threshold", fiberReplayThresholdDescriptorJson(
                 config.errorThresholdBaseVoxels)},
             {"match_refine_steps", config.matchRefineSteps},
             {"minimum_reset_advance_base_voxels", config.minimumResetAdvanceBaseVoxels},
             {"reference_begin_arc_base", config.referenceBeginArcBase},
             {"reference_end_arc_base", replay.referenceEndArcBase},
         }},
        {"reference_begin_arc_base", replay.referenceBeginArcBase},
        {"reference_end_arc_base", replay.referenceEndArcBase},
        {"completed_reference_arc_base", replay.completedReferenceArcBase},
        {"segments", nlohmann::json::array()},
        {"failures", nlohmann::json::array()},
    };
    for (const auto& segment : replay.segments) {
        nlohmann::json points = nlohmann::json::array();
        for (const auto& point : segment.routePointsBaseXYZ)
            points.push_back(pointJson(point));
        nlohmann::json matches = nlohmann::json::array();
        for (const auto& match : segment.matches) {
            validateFiberReplayThresholdMeasurement(
                match.thresholdMeasurement,
                config.errorThresholdBaseVoxels);
            auto matchJson = fiberReplayThresholdMeasurementJson(
                match.thresholdMeasurement);
            matchJson.update({
                {"route_point_index", match.routePointIndex},
                {"predicted_reference_arc_base", match.predictedReferenceArcBase},
                {"matched_reference_arc_base", match.matchedReferenceArcBase},
                {"matched_reference_point_base_xyz", pointJson(match.matchedReferencePointBaseXYZ)},
                {"search_begin_arc_base", match.searchBeginArcBase},
                {"search_end_arc_base", match.searchEndArcBase},
            });
            matches.push_back(std::move(matchJson));
        }
        root["segments"].push_back({
            {"start_reference_arc_base", segment.startReferenceArcBase},
            {"end_reference_arc_base", segment.endReferenceArcBase},
            {"termination_reason", segment.terminationReason},
            {"route_points_base_xyz", std::move(points)},
            {"candidate_indices", segment.candidateIndices},
            {"arc_indices", segment.arcIndices},
            {"transition_indices", segment.transitionIndices},
            {"stop_node_index", segment.stopNodeIndex.has_value() ? nlohmann::json(*segment.stopNodeIndex) : nlohmann::json(nullptr)},
            {"terminal_partial_edge", segment.terminalPartialEdge},
            {"matches", std::move(matches)},
            {"total_loss", segment.totalLoss},
            {"edge_cost", costJson(segment.edgeCost)},
            {"transition_cost", costJson(segment.transitionCost)},
            {"path_length_prediction_voxels", segment.pathLengthPredictionVoxels},
            {"loss_per_prediction_voxel",
             segment.pathLengthPredictionVoxels > kEpsilon ? nlohmann::json(segment.totalLoss / segment.pathLengthPredictionVoxels) : nlohmann::json(nullptr)},
        });
    }
    for (const auto& failure : replay.failures) {
        auto failureJson = fiberReplayOptionalThresholdMeasurementJson(
            failure.thresholdMeasurement,
            config.errorThresholdBaseVoxels);
        failureJson.update({
            {"index", failure.index},
            {"segment_index", failure.segmentIndex},
            {"reason", failure.reason},
            {"reference_arc_base", failure.referenceArcBase},
            {"reference_arc_fraction", failure.referenceArcFraction},
            {"reference_point_base_xyz", pointJson(failure.referencePointBase)},
            {"evaluator_point_base_xyz",
             failure.evaluatorPointBase.has_value() ? nlohmann::json(pointJson(*failure.evaluatorPointBase)) : nlohmann::json(nullptr)},
            {"segment_point_index", failure.segmentPointIndex.has_value() ? nlohmann::json(*failure.segmentPointIndex) : nlohmann::json(nullptr)},
            {"candidate_index", failure.candidateIndex.has_value() ? nlohmann::json(*failure.candidateIndex) : nlohmann::json(nullptr)},
            {"arc_index", failure.arcIndex.has_value() ? nlohmann::json(*failure.arcIndex) : nlohmann::json(nullptr)},
            {"candidate_path_point_index",
             failure.candidatePathPointIndex.has_value() ? nlohmann::json(*failure.candidatePathPointIndex) : nlohmann::json(nullptr)},
        });
        root["failures"].push_back(std::move(failureJson));
    }
    return root;
}

}  // namespace vc::fiber_tracer
