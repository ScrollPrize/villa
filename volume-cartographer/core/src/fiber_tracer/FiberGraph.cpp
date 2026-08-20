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

constexpr float kFloatEpsilon = 1.0e-6F;
constexpr double kReplayEpsilon = 1.0e-12;
constexpr double kPi = 3.141592653589793238462643383279502884;

float length(const cv::Vec3f& value)
{
    return std::sqrt(value.dot(value));
}

bool finiteVector(const cv::Vec3f& value)
{
    return std::isfinite(value[0]) && std::isfinite(value[1]) &&
        std::isfinite(value[2]);
}

double length(const cv::Vec3d& value)
{
    return std::sqrt(value.dot(value));
}

cv::Vec3f normalized(const cv::Vec3f& value)
{
    const float norm = length(value);
    if (!(norm > kFloatEpsilon) || !std::isfinite(norm))
        return {0.0F, 0.0F, 0.0F};
    return value / norm;
}

float angleDegrees(const cv::Vec3f& first, const cv::Vec3f& second)
{
    const cv::Vec3f a = normalized(first);
    const cv::Vec3f b = normalized(second);
    return std::acos(std::clamp(a.dot(b), -1.0F, 1.0F)) *
        (180.0F / static_cast<float>(kPi));
}

double angleDegrees(const cv::Vec3d& first, const cv::Vec3f& second)
{
    const double firstNorm = length(first);
    const cv::Vec3d secondDouble(second);
    const double secondNorm = length(secondDouble);
    if (!(firstNorm > kReplayEpsilon) || !(secondNorm > kReplayEpsilon))
        return 180.0;
    return std::acos(std::clamp(
               first.dot(secondDouble) / (firstNorm * secondNorm), -1.0, 1.0)) *
        180.0 / kPi;
}

size_t arcEdge(size_t arc)
{
    return arc / 2;
}
bool arcForward(size_t arc)
{
    return arc % 2 == 0;
}

cv::Vec3f arcStartDirection(const FiberletGraph& graph, size_t arc)
{
    const auto& points = graph.edges.at(arcEdge(arc)).pointsBaseXYZ;
    if (arcForward(arc)) {
        for (size_t index = 1; index < points.size(); ++index) {
            const cv::Vec3f direction = normalized(points[index] - points[0]);
            if (length(direction) > kFloatEpsilon)
                return direction;
        }
    } else {
        for (size_t index = points.size() - 1; index > 0; --index) {
            const cv::Vec3f direction = normalized(points[index - 1] - points.back());
            if (length(direction) > kFloatEpsilon)
                return direction;
        }
    }
    throw std::invalid_argument("fiberlet graph arc has no start tangent");
}

cv::Vec3f arcEndDirection(const FiberletGraph& graph, size_t arc)
{
    const auto& points = graph.edges.at(arcEdge(arc)).pointsBaseXYZ;
    if (arcForward(arc)) {
        for (size_t index = points.size() - 1; index > 0; --index) {
            const cv::Vec3f direction = normalized(points.back() - points[index - 1]);
            if (length(direction) > kFloatEpsilon)
                return direction;
        }
    } else {
        for (size_t index = 1; index < points.size(); ++index) {
            const cv::Vec3f direction = normalized(points[0] - points[index]);
            if (length(direction) > kFloatEpsilon)
                return direction;
        }
    }
    throw std::invalid_argument("fiberlet graph arc has no end tangent");
}

float arcStartLength(const FiberletGraph& graph, size_t arc)
{
    const auto& points = graph.edges.at(arcEdge(arc)).pointsBaseXYZ;
    const float scale = graph.predictionToBaseScale;
    return arcForward(arc) ? length(points[1] - points[0]) / scale : length(points[points.size() - 2] - points.back()) / scale;
}

float arcEndLength(const FiberletGraph& graph, size_t arc)
{
    const auto& points = graph.edges.at(arcEdge(arc)).pointsBaseXYZ;
    const float scale = graph.predictionToBaseScale;
    return arcForward(arc) ? length(points.back() - points[points.size() - 2]) / scale : length(points[0] - points[1]) / scale;
}

std::vector<cv::Vec3d> orientedArcPoints(const FiberletGraph& graph, size_t arc)
{
    std::vector<cv::Vec3d> points;
    const auto& stored = graph.edges.at(arcEdge(arc)).pointsBaseXYZ;
    points.reserve(stored.size());
    for (const auto& point : stored)
        points.emplace_back(point);
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

FiberletStorageKey storageKey(const FiberletAnchorId& id)
{
    if (id.componentIndex > 1)
        throw std::invalid_argument("fiberlet graph anchor variant exceeds one");
    return {{
                static_cast<std::int64_t>(id.cellZYX[0]),
                static_cast<std::int64_t>(id.cellZYX[1]),
                static_cast<std::int64_t>(id.cellZYX[2])},
        static_cast<std::uint8_t>(id.componentIndex)};
}

const FiberletStorageKey& sourceAnchor(const DirectedFiberletStorageId& id)
{
    return id.reverse ? id.fiberlet.second : id.fiberlet.first;
}

const FiberletStorageKey& targetAnchor(const DirectedFiberletStorageId& id)
{
    return id.reverse ? id.fiberlet.first : id.fiberlet.second;
}

cv::Vec3f sourceArcStartDirection(const FiberletReplaySourceArc& arc)
{
    const auto direction = normalized(arc.startStepBaseXYZ);
    if (!(length(direction) > kFloatEpsilon))
        throw std::invalid_argument("fiberlet replay source arc has no start tangent");
    return direction;
}

struct SourceRouteCandidate {
    std::vector<DirectedFiberletStorageId> arcs;
    std::vector<FiberletReplaySourceTransition> transitions;
    std::set<FiberletStorageKey> visitedNodes;
    double loss = 0.0;
    double pathLength = 0.0;
};

double routeDensity(const SourceRouteCandidate& route)
{
    return route.pathLength > kReplayEpsilon
        ? route.loss / route.pathLength
        : std::numeric_limits<double>::infinity();
}

bool routeLess(const SourceRouteCandidate& left, const SourceRouteCandidate& right)
{
    return std::tuple{routeDensity(left), left.loss, left.arcs} <
        std::tuple{routeDensity(right), right.loss, right.arcs};
}

void pruneRoutes(std::vector<SourceRouteCandidate>& routes, size_t beamWidth)
{
    std::sort(routes.begin(), routes.end(), [](const auto& left, const auto& right) {
        return routeLess(left, right);
    });
    if (routes.size() > beamWidth)
        routes.resize(beamWidth);
}

std::optional<SourceRouteCandidate> bestLookaheadRoute(
    const FiberletReplayGraphSource& graph,
    const FiberletStorageKey& currentNode,
    const std::optional<DirectedFiberletStorageId>& incomingArc,
    const std::set<FiberletStorageKey>& committedVisitedNodes,
    size_t beamWidth,
    size_t lookahead,
    const std::optional<cv::Vec3d>& initialDirection)
{
    std::vector<SourceRouteCandidate> frontier;
    const auto incomingView = incomingArc.has_value()
        ? std::make_optional(graph.arc(*incomingArc))
        : std::nullopt;
    for (const auto& arcId : graph.outgoing(currentNode)) {
        const auto arc = graph.arc(arcId);
        const auto join = incomingView.has_value()
            ? graph.transition(*incomingView, arc)
            : std::nullopt;
        if (incomingArc.has_value() && !join.has_value())
            continue;
        if (initialDirection.has_value() &&
            !(angleDegrees(*initialDirection, sourceArcStartDirection(arc)) <
              graph.maximumJoinAngleDegrees()))
            continue;
        if (committedVisitedNodes.contains(arc.target))
            continue;
        SourceRouteCandidate candidate;
        candidate.arcs.push_back(arcId);
        candidate.visitedNodes = committedVisitedNodes;
        candidate.visitedNodes.insert(arc.target);
        candidate.loss = arc.cost.total();
        candidate.pathLength = arc.pathLengthPredictionVoxels;
        if (join.has_value()) {
            candidate.transitions.push_back(*join);
            candidate.loss += join->cost.total();
        }
        frontier.push_back(std::move(candidate));
    }
    if (frontier.empty())
        return std::nullopt;
    pruneRoutes(frontier, beamWidth);
    for (size_t depth = 1; depth < lookahead; ++depth) {
        std::vector<SourceRouteCandidate> expanded;
        for (const auto& route : frontier) {
            const auto& tailArc = route.arcs.back();
            const auto& tailNode = targetAnchor(tailArc);
            const auto tailView = graph.arc(tailArc);
            for (const auto& arcId : graph.outgoing(tailNode)) {
                const auto arc = graph.arc(arcId);
                const auto join = graph.transition(tailView, arc);
                if (!join.has_value())
                    continue;
                if (route.visitedNodes.contains(arc.target))
                    continue;
                SourceRouteCandidate next = route;
                next.arcs.push_back(arcId);
                next.transitions.push_back(*join);
                next.visitedNodes.insert(arc.target);
                next.loss += arc.cost.total() + join->cost.total();
                next.pathLength += arc.pathLengthPredictionVoxels;
                expanded.push_back(std::move(next));
            }
        }
        if (expanded.empty())
            break;
        pruneRoutes(expanded, beamWidth);
        frontier = std::move(expanded);
    }
    return *std::min_element(frontier.begin(), frontier.end(),
        [](const auto& left, const auto& right) {
            return routeLess(left, right);
        });
}

class EagerReplayGraphSource final : public FiberletReplayGraphSource {
public:
    explicit EagerReplayGraphSource(const FiberletGraph& graph) : graph_(graph)
    {
        for (size_t edgeIndex = 0; edgeIndex < graph_.edges.size(); ++edgeIndex) {
            const auto& edge = graph_.edges[edgeIndex];
            const auto start = storageKey(graph_.nodes.at(edge.startNode).anchor);
            const auto target = storageKey(graph_.nodes.at(edge.targetNode).anchor);
            FiberletStorageId id{std::min(start, target), std::max(start, target)};
            if (!edgeById_.emplace(id, edgeIndex).second)
                throw std::invalid_argument("fiberlet graph contains duplicate stable edge IDs");
            arcById_.emplace(DirectedFiberletStorageId{id, start != id.first}, edgeIndex * 2);
            arcById_.emplace(DirectedFiberletStorageId{id, target != id.first}, edgeIndex * 2 + 1);
        }
    }

    float predictionToBaseScale() const noexcept override { return graph_.predictionToBaseScale; }
    int anchorCellSizePredictionVoxels() const noexcept override { return graph_.anchorCellSizePredictionVoxels; }
    float maximumJoinAngleDegrees() const noexcept override { return graph_.maximumJoinAngleDegrees; }

    std::vector<FiberletReplaySourceAnchor> anchorsNearReference(
        const PolylineArcGeometry& reference,
        double beginArcBase,
        double endArcBase,
        double broadPhaseRadiusBaseVoxels) const override
    {
        std::vector<FiberletReplaySourceAnchor> result;
        for (const auto& node : graph_.nodes) {
            const auto projection = projectPointToPolylineArc(
                reference, cv::Vec3d(node.positionBaseXYZ), beginArcBase,
                endArcBase);
            if (projection.arc + kReplayEpsilon < beginArcBase ||
                projection.arc > endArcBase + kReplayEpsilon ||
                projection.distance > broadPhaseRadiusBaseVoxels)
                continue;
            result.push_back({storageKey(node.anchor), node.positionBaseXYZ});
        }
        std::sort(result.begin(), result.end(), [](const auto& left, const auto& right) {
            return left.id < right.id;
        });
        return result;
    }

    std::vector<DirectedFiberletStorageId> outgoing(
        const FiberletStorageKey& anchor) const override
    {
        const auto found = std::find_if(graph_.nodes.begin(), graph_.nodes.end(),
            [&](const auto& node) { return storageKey(node.anchor) == anchor; });
        if (found == graph_.nodes.end())
            throw std::out_of_range("fiberlet replay anchor is absent");
        std::vector<DirectedFiberletStorageId> result;
        result.reserve(found->outgoingArcs.size());
        for (const auto arc : found->outgoingArcs)
            result.push_back(stableArc(arc));
        std::sort(result.begin(), result.end());
        return result;
    }

    FiberletReplaySourceArc arc(const DirectedFiberletStorageId& id) const override
    {
        const auto found = arcById_.find(id);
        if (found == arcById_.end())
            throw std::out_of_range("fiberlet replay arc is absent");
        const auto numericArc = found->second;
        const auto& edge = graph_.edges.at(arcEdge(numericArc));
        const auto points = orientedArcPoints(graph_, numericArc);
        if (points.size() < 2)
            throw std::invalid_argument("fiberlet replay source arc is too short");
        FiberletReplaySourceArc result;
        result.id = id;
        result.source = sourceAnchor(id);
        result.target = targetAnchor(id);
        result.sourcePositionBaseXYZ = points.front();
        result.targetPositionBaseXYZ = points.back();
        result.startStepBaseXYZ = cv::Vec3f(points[1] - points[0]);
        result.endStepBaseXYZ = cv::Vec3f(points.back() - points[points.size() - 2]);
        result.pathLengthPredictionVoxels = edge.pathLengthPredictionVoxels;
        result.cost = edge.cost;
        result.diagnosticCandidateIndex = edge.candidateIndex;
        result.diagnosticArcIndex = numericArc;
        return result;
    }

    std::vector<cv::Vec3d> routePoints(
        const DirectedFiberletStorageId& id) const override
    {
        const auto found = arcById_.find(id);
        if (found == arcById_.end())
            throw std::out_of_range("fiberlet replay arc is absent");
        return orientedArcPoints(graph_, found->second);
    }

    std::optional<FiberletReplaySourceTransition> transition(
        const FiberletReplaySourceArc& incoming,
        const FiberletReplaySourceArc& outgoing) const override
    {
        const auto incomingFound = arcById_.find(incoming.id);
        const auto outgoingFound = arcById_.find(outgoing.id);
        if (incomingFound == arcById_.end() || outgoingFound == arcById_.end())
            throw std::out_of_range("fiberlet replay transition arc is absent");
        const auto index = transitionIndex(
            graph_, incomingFound->second, outgoingFound->second);
        if (!index.has_value())
            return std::nullopt;
        return FiberletReplaySourceTransition{
            incoming.id, outgoing.id, graph_.transitions[*index].cost, *index};
    }

private:
    DirectedFiberletStorageId stableArc(size_t numericArc) const
    {
        const auto& edge = graph_.edges.at(arcEdge(numericArc));
        const auto start = storageKey(graph_.nodes.at(edge.startNode).anchor);
        const auto target = storageKey(graph_.nodes.at(edge.targetNode).anchor);
        const FiberletStorageId id{std::min(start, target), std::max(start, target)};
        const auto& source = arcForward(numericArc) ? start : target;
        return {id, source != id.first};
    }

    const FiberletGraph& graph_;
    std::map<FiberletStorageId, size_t> edgeById_;
    std::map<DirectedFiberletStorageId, size_t> arcById_;
};

nlohmann::json anchorIdJson(const FiberletAnchorId& id)
{
    return {{"cell_zyx", id.cellZYX}, {"component", id.componentIndex}};
}

template <typename T>
nlohmann::json pointJson(const cv::Vec<T, 3>& point)
{
    return nlohmann::json::array({point[0], point[1], point[2]});
}

template <typename Cost>
nlohmann::json costJson(const Cost& cost)
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

FiberLocalMetricSample localSample(const FiberletPredictionSample& sample)
{
    return {
        sample.direction,
        sample.presence,
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

bool sameAxis(const cv::Vec3f& left, const cv::Vec3f& right)
{
    const cv::Vec3f a = normalized(left);
    const cv::Vec3f b = normalized(right);
    return length(a) > kFloatEpsilon && length(b) > kFloatEpsilon &&
        std::abs(a.dot(b)) >= 1.0F - 1.0e-5F;
}

}  // namespace

FiberletGraph buildFiberletGraph(const FiberletPathReport& paths, float maximumJoinAngleDegrees)
{
    const float predictionToBaseScale =
        static_cast<float>(paths.grid.predictionToBaseScale);
    if (!(paths.grid.predictionToBaseScale > 0.0) || !std::isfinite(paths.grid.predictionToBaseScale) || !(maximumJoinAngleDegrees >= 0.0) ||
        !(predictionToBaseScale > 0.0F) || !std::isfinite(predictionToBaseScale) ||
        !(maximumJoinAngleDegrees <= 180.0F) || !std::isfinite(maximumJoinAngleDegrees) || paths.anchorCellSizePredictionVoxels < 1) {
        throw std::invalid_argument("fiberlet graph configuration is invalid");
    }
    FiberletGraph graph;
    graph.predictionToBaseScale = predictionToBaseScale;
    graph.anchorCellSizePredictionVoxels = paths.anchorCellSizePredictionVoxels;
    graph.maximumJoinAngleDegrees = maximumJoinAngleDegrees;
    std::map<FiberletAnchorId, size_t> nodeByAnchor;
    const auto ensureNode =
        [&](const FiberletAnchorId& id, const cv::Vec3f& positionPrediction, const FiberletPredictionSample& prediction, const cv::Vec3f& normal, bool normalValid) {
            const auto [it, inserted] = nodeByAnchor.emplace(id, graph.nodes.size());
            const cv::Vec3f positionBase = positionPrediction * graph.predictionToBaseScale;
            if (!finiteVector(positionPrediction) || !finiteVector(positionBase))
                throw std::invalid_argument("fiberlet graph anchor position is not finite");
            if (inserted) {
                graph.nodes.push_back({id, positionBase, prediction, normal, normalValid, {}});
            } else if (length(graph.nodes[it->second].positionBaseXYZ - positionBase) > 1.0e-5F) {
                throw std::invalid_argument("fiberlet graph anchor identity has inconsistent positions");
            } else {
                const auto& existing = graph.nodes[it->second];
                if (existing.prediction.valid != prediction.valid ||
                    (prediction.valid && (!sameAxis(existing.prediction.direction, prediction.direction) ||
                                          std::abs(existing.prediction.presence - prediction.presence) > 1.0e-5F)) ||
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
        if (!candidate.scoreValid || candidate.pointsPredictionXYZ.size() < 2 || !(candidate.cost.total() >= 0.0F) ||
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
        if (!(edge.pathLengthPredictionVoxels > kFloatEpsilon))
            throw std::invalid_argument("successful fiberlet has zero graph length");
        edge.pointsBaseXYZ.reserve(candidate.pointsPredictionXYZ.size());
        for (const auto& point : candidate.pointsPredictionXYZ) {
            const cv::Vec3f pointBase = point * graph.predictionToBaseScale;
            if (!finiteVector(point) || !finiteVector(pointBase))
                throw std::invalid_argument("fiberlet graph path point is not finite");
            edge.pointsBaseXYZ.push_back(pointBase);
        }
        if (length(edge.pointsBaseXYZ.front() - graph.nodes[start].positionBaseXYZ) > 1.0e-5F ||
            length(edge.pointsBaseXYZ.back() - graph.nodes[target].positionBaseXYZ) > 1.0e-5F) {
            throw std::invalid_argument("fiberlet graph path endpoints do not match its anchors");
        }
        const size_t edgeIndex = graph.edges.size();
        graph.edges.push_back(std::move(edge));
        graph.nodes[start].outgoingArcs.push_back(edgeIndex * 2);
        graph.nodes[target].outgoingArcs.push_back(edgeIndex * 2 + 1);
    }

    for (auto& node : graph.nodes)
        std::sort(node.outgoingArcs.begin(), node.outgoingArcs.end());
    const float minimumJoinDot = std::cos(
        maximumJoinAngleDegrees * static_cast<float>(kPi / 180.0));
    for (size_t node = 0; node < graph.nodes.size(); ++node) {
        const auto& arcs = graph.nodes[node].outgoingArcs;
        for (const size_t incomingArc : arcs) {
            const size_t directedIncoming = incomingArc ^ 1U;
            const cv::Vec3f incomingDirection = arcEndDirection(graph, directedIncoming);
            for (const size_t outgoingArc : arcs) {
                if (arcEdge(directedIncoming) == arcEdge(outgoingArc))
                    continue;
                const cv::Vec3f outgoingDirection =
                    arcStartDirection(graph, outgoingArc);
                const float angle = angleDegrees(
                    incomingDirection, outgoingDirection);
                if (incomingDirection.dot(outgoingDirection) > minimumJoinDot &&
                    graph.nodes[node].prediction.valid) {
                    const float incomingLength = arcEndLength(graph, directedIncoming);
                    const float outgoingLength = arcStartLength(graph, outgoingArc);
                    const auto sample = localSample(graph.nodes[node].prediction);
                    const auto cost = fiberLocalMetricCost(
                        &sample,
                        sample,
                        incomingDirection,
                        incomingLength,
                        outgoingDirection,
                        outgoingLength,
                        graph.nodes[node].normalXYZ,
                        graph.nodes[node].normalValid,
                        FiberLocalMetricConfig{
                            paths.config.invalidPredictionCostPerVoxel,
                            FiberLocalSmoothnessConfig{
                                paths.config.smoothnessWeight,
                                paths.config.smoothnessNormalWeight,
                                paths.config.smoothnessTangentWeight,
                                paths.config.smoothnessFreeAngleDegrees *
                                    static_cast<float>(kPi / 180.0)}});
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
    const FiberReplayFailureCallback& failureCallback,
    const FiberletGraphReplayProgressCallback& progressCallback)
{
    const EagerReplayGraphSource source(graph);
    return traceFiberletGraphReplay(
        source, referencePointsBaseXYZ, normalSampler,
        normalWorkingToBaseScale, config, failureCallback, progressCallback);
}

FiberletGraphReplayResult traceFiberletGraphReplay(
    const FiberletReplayGraphSource& graph,
    const std::vector<cv::Vec3d>& referencePointsBaseXYZ,
    const vc::lasagna::NormalSampler& normalSampler,
    double normalWorkingToBaseScale,
    const FiberletGraphReplayConfig& config,
    const FiberReplayFailureCallback& failureCallback,
    const FiberletGraphReplayProgressCallback& progressCallback)
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
        *config.referenceEndArcBase > reference.length() + kReplayEpsilon) {
        throw std::invalid_argument(
            "fiberlet graph replay reference end exceeds the reference");
    }
    const double referenceEndArcBase = config.referenceEndArcBase.has_value()
        ? *config.referenceEndArcBase
        : reference.length();
    if (config.referenceBeginArcBase >= referenceEndArcBase - kReplayEpsilon)
        throw std::invalid_argument("fiberlet graph replay has no usable reference interval");
    FiberletGraphReplayResult result;
    result.referenceBeginArcBase = config.referenceBeginArcBase;
    result.referenceEndArcBase = referenceEndArcBase;
    result.completedReferenceArcBase = config.referenceBeginArcBase;
    const double intervalLength =
        referenceEndArcBase - config.referenceBeginArcBase;
    const size_t maximumSegments = static_cast<size_t>(std::ceil(intervalLength / config.minimumResetAdvanceBaseVoxels)) + 2;
    const double seedWindowBase =
        std::max(config.minimumResetAdvanceBaseVoxels,
            static_cast<double>(graph.anchorCellSizePredictionVoxels()) *
                graph.predictionToBaseScale());
    const double seedBroadPhaseBase =
        fiberReplayTangentialThresholdBaseVoxels(
            config.errorThresholdBaseVoxels);
    std::set<FiberletStorageKey> consumedNodes;
    std::map<FiberletStorageKey, size_t> nodeIndices;
    std::map<FiberletStorageId, size_t> candidateIndices;
    std::map<DirectedFiberletStorageId, size_t> arcIndices;
    std::map<std::pair<DirectedFiberletStorageId, DirectedFiberletStorageId>,
        size_t> transitionIndices;
    const auto stableIndex = []<typename Key>(std::map<Key, size_t>& indices,
                                 const Key& key) {
        const auto [found, inserted] = indices.emplace(key, indices.size());
        return found->second;
    };

    struct Seed {
        FiberletReplaySourceAnchor node;
        PolylineArcProjection projection;
        FiberReplayThresholdMeasurement thresholdMeasurement;
    };
    const auto selectSeed = [&](double resetArc) -> std::optional<Seed> {
        double scanBegin = resetArc;
        while (scanBegin < referenceEndArcBase - kReplayEpsilon) {
            const double scanEnd = std::min(
                referenceEndArcBase, scanBegin + seedWindowBase);
            std::optional<Seed> selected;
            for (const auto& node : graph.anchorsNearReference(
                     reference, scanBegin, scanEnd,
                     seedBroadPhaseBase)) {
                if (consumedNodes.contains(node.id))
                    continue;
                const auto projection = projectPointToPolylineArc(
                    reference, cv::Vec3d(node.positionBaseXYZ), resetArc,
                    referenceEndArcBase);
                if (projection.arc + kReplayEpsilon < scanBegin ||
                    projection.arc > scanEnd + kReplayEpsilon ||
                    projection.distance > seedBroadPhaseBase)
                    continue;
                const auto thresholdMeasurement = measureFiberReplayThreshold(
                    node.positionBaseXYZ, projection.point,
                    normalSampler, normalWorkingToBaseScale,
                    config.errorThresholdBaseVoxels);
                if (fiberReplayThresholdExceeded(
                        thresholdMeasurement,
                        config.errorThresholdBaseVoxels))
                    continue;
                const cv::Vec3d tangent =
                    samplePolylineArc(reference, projection.arc).tangent;
                bool aligned = false;
                for (const auto& arcId : graph.outgoing(node.id)) {
                    if (angleDegrees(
                            tangent,
                            sourceArcStartDirection(graph.arc(arcId))) <
                        graph.maximumJoinAngleDegrees()) {
                        aligned = true;
                        break;
                    }
                }
                if (!aligned)
                    continue;
                if (!selected.has_value() ||
                    std::tuple{projection.arc,
                        thresholdMeasurement.thresholdErrorRatio, node.id} <
                        std::tuple{selected->projection.arc,
                            selected->thresholdMeasurement.thresholdErrorRatio,
                            selected->node.id}) {
                    selected = Seed{node, projection, thresholdMeasurement};
                }
            }
            if (selected.has_value())
                return selected;
            if (scanEnd >= referenceEndArcBase - kReplayEpsilon)
                break;
            scanBegin = scanEnd;
        }
        return std::nullopt;
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
    const auto emitProgress = [&](size_t segmentIndex, double arcBase,
                                  const char* state) {
        if (!progressCallback)
            return;
        progressCallback({
            segmentIndex,
            arcBase,
            std::clamp(
                (arcBase - result.referenceBeginArcBase) / intervalLength,
                0.0, 1.0),
            state,
        });
    };

    double resetArc = config.referenceBeginArcBase;
    for (size_t iteration = 0; iteration < maximumSegments &&
         resetArc < referenceEndArcBase - kReplayEpsilon; ++iteration) {
        emitProgress(result.segments.size(), resetArc, "segment_start");
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
            emitProgress(
                result.segments.size() - 1, referenceEndArcBase, "completed");
            break;
        }
        if (seed->projection.arc > resetArc + seedWindowBase + kReplayEpsilon) {
            FiberletGraphReplaySegment gap;
            gap.startReferenceArcBase = resetArc;
            gap.endReferenceArcBase = seed->projection.arc;
            gap.terminationReason = "missing_seed_gap";
            const size_t segmentIndex = result.segments.size();
            result.segments.push_back(std::move(gap));
            appendFailure({0, segmentIndex, "missing_seed_gap", resetArc});
        }

        FiberletGraphReplaySegment segment;
        (void)stableIndex(nodeIndices, seed->node.id);
        segment.startReferenceArcBase = seed->projection.arc;
        segment.endReferenceArcBase = seed->projection.arc;
        segment.routePointsBaseXYZ.emplace_back(seed->node.positionBaseXYZ);
        segment.matches.push_back({
            0,
            seed->projection.arc,
            seed->projection.arc,
            seed->projection.point,
            resetArc,
            seed->projection.arc,
            seed->thresholdMeasurement,
        });
        FiberletStorageKey currentNode = seed->node.id;
        std::optional<DirectedFiberletStorageId> incomingArc;
        std::set<FiberletStorageKey> visitedNodes{currentNode};
        double previousReferenceArc = seed->projection.arc;
        std::optional<FiberReplayFailure> distanceFailure;
        bool referenceExhausted =
            previousReferenceArc >= referenceEndArcBase - kReplayEpsilon;
        bool terminalPartialEdge = false;

        while (!distanceFailure.has_value() &&
               previousReferenceArc < referenceEndArcBase - kReplayEpsilon) {
            const cv::Vec3d startDirection = samplePolylineArc(reference, previousReferenceArc).tangent;
            const auto selected =
                bestLookaheadRoute(graph, currentNode, incomingArc, visitedNodes, config.beamWidth, config.lookaheadEdges, incomingArc.has_value() ? std::nullopt : std::make_optional(startDirection));
            if (!selected.has_value())
                break;
            const auto arcId = selected->arcs.front();
            const auto edge = graph.arc(arcId);
            const auto points = graph.routePoints(arcId);
            if (points.size() < 2)
                throw std::logic_error("selected fiberlet route is too short");
            const size_t candidateIndex = edge.diagnosticCandidateIndex.value_or(
                stableIndex(candidateIndices, arcId.fiberlet));
            const size_t arcIndex = edge.diagnosticArcIndex.value_or(
                stableIndex(arcIndices, arcId));
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
                    event.candidateIndex = candidateIndex;
                    event.arcIndex = arcIndex;
                    event.candidatePathPointIndex = index;
                    event.thresholdMeasurement = thresholdMeasurement;
                    distanceFailure = std::move(event);
                }
                if (previousReferenceArc >= referenceEndArcBase - kReplayEpsilon) {
                    referenceExhausted = true;
                    terminalPartialEdge = index + 1 < points.size();
                    break;
                }
            }
            segment.candidateIndices.push_back(candidateIndex);
            segment.arcIndices.push_back(arcIndex);
            segment.edgeCost += edge.cost;
            segment.totalLoss += edge.cost.total();
            if (incomingArc.has_value()) {
                if (selected->transitions.empty())
                    throw std::logic_error("selected fiberlet route has no graph transition");
                const auto& join = selected->transitions.front();
                const size_t transitionIndexValue =
                    join.diagnosticTransitionIndex.value_or(stableIndex(
                        transitionIndices, std::pair{*incomingArc, arcId}));
                segment.transitionIndices.push_back(transitionIndexValue);
                segment.transitionCost += join.cost;
                segment.totalLoss += join.cost.total();
            }
            segment.pathLengthPredictionVoxels += edge.pathLengthPredictionVoxels;
            if (!terminalPartialEdge) {
                incomingArc = arcId;
                currentNode = edge.target;
                visitedNodes.insert(currentNode);
            }
            emitProgress(
                result.segments.size(), previousReferenceArc, "running");
        }
        segment.terminalPartialEdge = terminalPartialEdge;
        if (!terminalPartialEdge)
            segment.stopNodeIndex = stableIndex(nodeIndices, currentNode);
        consumedNodes.insert(visitedNodes.begin(), visitedNodes.end());

        if (referenceExhausted && !distanceFailure.has_value()) {
            segment.endReferenceArcBase = referenceEndArcBase;
            segment.terminationReason = "reference_end";
            result.segments.push_back(std::move(segment));
            result.completedReferenceArcBase = referenceEndArcBase;
            emitProgress(
                result.segments.size() - 1, referenceEndArcBase, "completed");
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
        if (!(resetArc > result.segments.back().startReferenceArcBase + kReplayEpsilon))
            throw std::logic_error("fiberlet graph replay reset did not advance");
        result.completedReferenceArcBase = resetArc;
        emitProgress(result.segments.size() - 1, resetArc, "restart");
    }
    if (result.completedReferenceArcBase < referenceEndArcBase - kReplayEpsilon)
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
             segment.pathLengthPredictionVoxels > kReplayEpsilon ? nlohmann::json(segment.totalLoss / segment.pathLengthPredictionVoxels) : nlohmann::json(nullptr)},
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
