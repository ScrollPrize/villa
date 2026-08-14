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

FiberletGraphReplayResult traceFiberletGraphReplay(const FiberletGraph& graph, const std::vector<cv::Vec3d>& referencePointsBaseXYZ, const FiberletGraphReplayConfig& config)
{
    if (config.beamWidth < 1 || config.lookaheadEdges < 1 || !(config.errorThresholdBaseVoxels >= 0.0) ||
        !std::isfinite(config.errorThresholdBaseVoxels) || !(config.matchRefineSteps >= 0.0) || !std::isfinite(config.matchRefineSteps) ||
        !(config.postrollDistanceBaseVoxels >= 0.0) || !std::isfinite(config.postrollDistanceBaseVoxels)) {
        throw std::invalid_argument("fiberlet graph replay configuration is invalid");
    }
    const auto reference = makePolylineArcGeometry(referencePointsBaseXYZ);
    FiberletGraphReplayResult result;
    result.requestedPostrollDistanceBaseVoxels = config.postrollDistanceBaseVoxels;
    const cv::Vec3d startDirection = normalized(referencePointsBaseXYZ[1] - referencePointsBaseXYZ[0]);

    std::optional<size_t> startNode;
    double startDistance = std::numeric_limits<double>::infinity();
    for (size_t node = 0; node < graph.nodes.size(); ++node) {
        bool aligned = false;
        for (const size_t arc : graph.nodes[node].outgoingArcs) {
            if (angleDegrees(startDirection, arcStartDirection(graph, arc)) < graph.maximumJoinAngleDegrees) {
                aligned = true;
                break;
            }
        }
        if (!aligned)
            continue;
        const double distance = length(graph.nodes[node].positionBaseXYZ - referencePointsBaseXYZ.front());
        if (distance < startDistance - kEpsilon || (std::abs(distance - startDistance) <= kEpsilon && (!startNode.has_value() || node < *startNode))) {
            startDistance = distance;
            startNode = node;
        }
    }
    if (!startNode.has_value() || startDistance > config.errorThresholdBaseVoxels) {
        result.status = FiberletGraphReplayStatus::NoUsableStart;
        result.reason = "no_aligned_anchor_within_threshold";
        return result;
    }

    result.routePointsBaseXYZ.push_back(graph.nodes[*startNode].positionBaseXYZ);
    const double startWindowBase = static_cast<double>(graph.anchorCellSizePredictionVoxels) * graph.predictionToBaseScale;
    const auto startMatch =
        projectPointToPolylineArc(reference, result.routePointsBaseXYZ.front(), 0.0, std::min(reference.length(), startWindowBase));
    result.matches.push_back({
        0,
        0.0,
        startMatch.arc,
        startMatch.point,
        0.0,
        std::min(reference.length(), startWindowBase),
        startMatch.distance,
    });
    if (startMatch.distance > config.errorThresholdBaseVoxels) {
        result.status = FiberletGraphReplayStatus::NoUsableStart;
        result.reason = "start_distance_above_threshold";
        return result;
    }

    size_t currentNode = *startNode;
    std::optional<size_t> incomingArc;
    std::set<size_t> visitedNodes{currentNode};
    double previousReferenceArc = startMatch.arc;
    bool failed = false;
    while (true) {
        if (!failed && previousReferenceArc >= reference.length() - kEpsilon) {
            result.status = FiberletGraphReplayStatus::ReferenceEnd;
            result.reason = "reference_end";
            result.stopNodeIndex = currentNode;
            return result;
        }
        const auto selected =
            bestLookaheadRoute(graph, currentNode, incomingArc, visitedNodes, config.beamWidth, config.lookaheadEdges, incomingArc.has_value() ? std::nullopt : std::make_optional(startDirection));
        if (!selected.has_value()) {
            result.status = failed ? FiberletGraphReplayStatus::FailureTruncated : FiberletGraphReplayStatus::GraphExhausted;
            if (failed) {
                result.reason = previousReferenceArc >= reference.length() - kEpsilon ? "postroll_comparison_interval_exhausted" : "postroll_graph_exhausted";
            } else {
                result.reason = "no_admissible_continuation";
            }
            result.stopNodeIndex = currentNode;
            return result;
        }
        const size_t arc = selected->arcs.front();
        const auto& edge = graph.edges.at(arcEdge(arc));
        const auto points = orientedArcPoints(graph, arc);
        for (size_t index = 1; index < points.size(); ++index) {
            const double stepBase = length(points[index] - result.routePointsBaseXYZ.back());
            const auto forwardMatch = matchForwardPolylinePoint(reference, points[index], previousReferenceArc, stepBase, config.matchRefineSteps);
            const auto& match = forwardMatch.projection;
            result.routePointsBaseXYZ.push_back(points[index]);
            result.matches.push_back({
                result.routePointsBaseXYZ.size() - 1,
                forwardMatch.predictedArc,
                match.arc,
                match.point,
                previousReferenceArc,
                forwardMatch.searchEndArc,
                match.distance,
            });
            previousReferenceArc = match.arc;
            if (failed) {
                result.completedPostrollDistanceBaseVoxels += stepBase;
            } else if (match.distance > config.errorThresholdBaseVoxels) {
                result.failureRoutePointIndex = result.routePointsBaseXYZ.size() - 1;
                result.failureReferenceArcBase = match.arc;
                failed = true;
                result.failureCandidateIndex = edge.candidateIndex;
                result.failureCandidatePathPointIndex = index;
                result.failureArcIndex = arc;
            }
        }
        result.candidateIndices.push_back(edge.candidateIndex);
        result.arcIndices.push_back(arc);
        result.edgeCost += edge.cost;
        result.totalLoss += edge.cost.total();
        if (incomingArc.has_value()) {
            const auto join = transitionIndex(graph, *incomingArc, arc);
            if (!join.has_value())
                throw std::logic_error("selected fiberlet route has no graph transition");
            result.transitionIndices.push_back(*join);
            result.transitionCost += graph.transitions[*join].cost;
            result.totalLoss += graph.transitions[*join].cost.total();
        }
        result.pathLengthPredictionVoxels += edge.pathLengthPredictionVoxels;
        incomingArc = arc;
        currentNode = arcTarget(graph, arc);
        visitedNodes.insert(currentNode);
        if (failed && result.completedPostrollDistanceBaseVoxels + kEpsilon >= result.requestedPostrollDistanceBaseVoxels) {
            result.status = FiberletGraphReplayStatus::FailureWithPostroll;
            result.reason = "postroll_distance_reached_at_anchor";
            result.stopNodeIndex = currentNode;
            return result;
        }
    }
}

const char* fiberletGraphReplayStatusName(FiberletGraphReplayStatus status) noexcept
{
    switch (status) {
        case FiberletGraphReplayStatus::FailureWithPostroll:
            return "failure_with_postroll";
        case FiberletGraphReplayStatus::FailureTruncated:
            return "failure_truncated";
        case FiberletGraphReplayStatus::ReferenceEnd:
            return "reference_end";
        case FiberletGraphReplayStatus::GraphExhausted:
            return "graph_exhausted";
        case FiberletGraphReplayStatus::NoUsableStart:
            return "no_usable_start";
    }
    return "unknown";
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
    output << "# vc_fiberlet_graph_replay version 1\n";
    output << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (const auto& point : replay.routePointsBaseXYZ)
        output << "v " << point[0] << ' ' << point[1] << ' ' << point[2] << '\n';
    if (!replay.routePointsBaseXYZ.empty()) {
        output << "l";
        if (replay.routePointsBaseXYZ.size() == 1)
            output << " 1 1";
        else
            for (size_t index = 0; index < replay.routePointsBaseXYZ.size(); ++index)
                output << ' ' << index + 1;
        output << '\n';
    }
    return output.str();
}

nlohmann::json fiberletGraphReplayJson(const FiberletGraphReplayResult& replay, const FiberletGraphReplayConfig& config)
{
    nlohmann::json points = nlohmann::json::array();
    for (const auto& point : replay.routePointsBaseXYZ)
        points.push_back(pointJson(point));
    nlohmann::json matches = nlohmann::json::array();
    for (const auto& match : replay.matches) {
        matches.push_back({
            {"route_point_index", match.routePointIndex},
            {"predicted_reference_arc_base", match.predictedReferenceArcBase},
            {"matched_reference_arc_base", match.matchedReferenceArcBase},
            {"matched_reference_point_base_xyz", pointJson(match.matchedReferencePointBaseXYZ)},
            {"search_begin_arc_base", match.searchBeginArcBase},
            {"search_end_arc_base", match.searchEndArcBase},
            {"error_base_voxels", match.errorBaseVoxels},
        });
    }
    nlohmann::json root = {
        {"format", "vc_fiberlet_graph_replay"},
        {"version", 1},
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
             {"error_threshold_base_voxels", config.errorThresholdBaseVoxels},
             {"match_refine_steps", config.matchRefineSteps},
             {"postroll_distance_base_voxels", config.postrollDistanceBaseVoxels},
         }},
        {"status", fiberletGraphReplayStatusName(replay.status)},
        {"reason", replay.reason},
        {"route_points_base_xyz", std::move(points)},
        {"candidate_indices", replay.candidateIndices},
        {"arc_indices", replay.arcIndices},
        {"transition_indices", replay.transitionIndices},
        {"failure_candidate_index", replay.failureCandidateIndex.has_value() ? nlohmann::json(*replay.failureCandidateIndex) : nlohmann::json(nullptr)},
        {"failure_candidate_path_point_index",
         replay.failureCandidatePathPointIndex.has_value() ? nlohmann::json(*replay.failureCandidatePathPointIndex) : nlohmann::json(nullptr)},
        {"failure_arc_index", replay.failureArcIndex.has_value() ? nlohmann::json(*replay.failureArcIndex) : nlohmann::json(nullptr)},
        {"stop_node_index", replay.stopNodeIndex.has_value() ? nlohmann::json(*replay.stopNodeIndex) : nlohmann::json(nullptr)},
        {"matches", std::move(matches)},
        {"failure_route_point_index", replay.failureRoutePointIndex.has_value() ? nlohmann::json(*replay.failureRoutePointIndex) : nlohmann::json(nullptr)},
        {"failure_reference_arc_base",
         replay.failureReferenceArcBase.has_value() ? nlohmann::json(*replay.failureReferenceArcBase) : nlohmann::json(nullptr)},
        {"postroll", nullptr},
        {"total_loss", replay.totalLoss},
        {"edge_cost", costJson(replay.edgeCost)},
        {"transition_cost", costJson(replay.transitionCost)},
        {"path_length_prediction_voxels", replay.pathLengthPredictionVoxels},
        {"loss_per_prediction_voxel",
         replay.pathLengthPredictionVoxels > kEpsilon ? nlohmann::json(replay.totalLoss / replay.pathLengthPredictionVoxels) : nlohmann::json(nullptr)},
    };
    const bool failed = replay.failureRoutePointIndex.has_value();
    if (failed) {
        const bool complete = replay.status == FiberletGraphReplayStatus::FailureWithPostroll;
        root["postroll"] = {
            {"requested_distance_base_voxels", replay.requestedPostrollDistanceBaseVoxels},
            {"completed_distance_base_voxels", replay.completedPostrollDistanceBaseVoxels},
            {"complete", complete},
            {"overshoot_base_voxels",
             complete ? std::max(0.0, replay.completedPostrollDistanceBaseVoxels - replay.requestedPostrollDistanceBaseVoxels) : 0.0},
            {"shortfall_base_voxels", complete ? 0.0 : std::max(0.0, replay.requestedPostrollDistanceBaseVoxels - replay.completedPostrollDistanceBaseVoxels)},
        };
    }
    return root;
}

}  // namespace vc::fiber_tracer
