#include "vc/fiber_tracer/FiberGraph.hpp"

#include "vc/fiber_tracer/FiberLocalScoring.hpp"
#include "vc/fiber_tracer/PolylineGeometry.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <future>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
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
    return std::isfinite(value[0]) && std::isfinite(value[1]) && std::isfinite(value[2]);
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
    return std::acos(std::clamp(a.dot(b), -1.0F, 1.0F)) * (180.0F / static_cast<float>(kPi));
}

double angleDegrees(const cv::Vec3d& first, const cv::Vec3f& second)
{
    const double firstNorm = length(first);
    const cv::Vec3d secondDouble(second);
    const double secondNorm = length(secondDouble);
    if (!(firstNorm > kReplayEpsilon) || !(secondNorm > kReplayEpsilon))
        return 180.0;
    return std::acos(std::clamp(first.dot(secondDouble) / (firstNorm * secondNorm), -1.0, 1.0)) * 180.0 / kPi;
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
    return {{static_cast<std::int64_t>(id.cellZYX[0]), static_cast<std::int64_t>(id.cellZYX[1]), static_cast<std::int64_t>(id.cellZYX[2])}, static_cast<std::uint8_t>(id.componentIndex)};
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

void addScaledCost(FiberletGraphReplayCost& target, const FiberletPathCost& source, double scale)
{
    if (!(scale >= 0.0) || !(scale <= 1.0) || !std::isfinite(scale))
        throw std::invalid_argument("fiberlet route cost scale is invalid");
    target.invalidPrediction += scale * static_cast<double>(source.invalidPrediction);
    target.alignment += scale * static_cast<double>(source.alignment);
    target.isotropicSmoothness += scale * static_cast<double>(source.isotropicSmoothness);
    target.tangentSmoothness += scale * static_cast<double>(source.tangentSmoothness);
    target.normalSmoothness += scale * static_cast<double>(source.normalSmoothness);
}

std::vector<cv::Vec3d> routePointPrefix(const std::vector<cv::Vec3d>& points, double fraction)
{
    if (points.size() < 2 || !(fraction >= 0.0) || !(fraction <= 1.0) || !std::isfinite(fraction)) {
        throw std::invalid_argument("fiberlet diagnostic route prefix is invalid");
    }
    if (fraction >= 1.0 - kReplayEpsilon)
        return points;
    double totalLength = 0.0;
    for (size_t index = 1; index < points.size(); ++index)
        totalLength += length(points[index] - points[index - 1]);
    if (!(totalLength > kReplayEpsilon))
        throw std::invalid_argument("fiberlet diagnostic route has zero length");
    const double targetLength = fraction * totalLength;
    std::vector<cv::Vec3d> result{points.front()};
    double covered = 0.0;
    for (size_t index = 1; index < points.size(); ++index) {
        const cv::Vec3d step = points[index] - points[index - 1];
        const double stepLength = length(step);
        if (covered + stepLength >= targetLength - kReplayEpsilon) {
            const double stepFraction = stepLength > kReplayEpsilon ? std::clamp((targetLength - covered) / stepLength, 0.0, 1.0) : 0.0;
            result.push_back(points[index - 1] + stepFraction * step);
            return result;
        }
        result.push_back(points[index]);
        covered += stepLength;
    }
    return points;
}

struct PersistentVisitedNodes {
    std::shared_ptr<const std::set<FiberletStorageKey>> base;
    std::shared_ptr<const PersistentVisitedNodes> parent;
    std::optional<FiberletStorageKey> added;
};

std::shared_ptr<const PersistentVisitedNodes> makeVisitedRoot(std::set<FiberletStorageKey> nodes)
{
    return std::make_shared<const PersistentVisitedNodes>(
        PersistentVisitedNodes{std::make_shared<const std::set<FiberletStorageKey>>(std::move(nodes)), nullptr, std::nullopt});
}

bool persistentVisitedContains(const std::shared_ptr<const PersistentVisitedNodes>& visited, const FiberletStorageKey& key)
{
    for (auto node = visited; node != nullptr; node = node->parent) {
        if (node->added.has_value() && *node->added == key)
            return true;
        if (node->base != nullptr && node->base->contains(key))
            return true;
    }
    return false;
}

std::shared_ptr<const PersistentVisitedNodes> persistentVisitedAdd(const std::shared_ptr<const PersistentVisitedNodes>& visited, FiberletStorageKey key)
{
    return std::make_shared<const PersistentVisitedNodes>(PersistentVisitedNodes{nullptr, visited, std::move(key)});
}

std::shared_ptr<const PersistentVisitedNodes> compactPersistentVisited(const std::shared_ptr<const PersistentVisitedNodes>& visited)
{
    std::set<FiberletStorageKey> nodes;
    std::vector<FiberletStorageKey> additions;
    for (auto node = visited; node != nullptr; node = node->parent) {
        if (node->added.has_value())
            additions.push_back(*node->added);
        if (node->base != nullptr) {
            nodes = *node->base;
            break;
        }
    }
    nodes.insert(additions.begin(), additions.end());
    return makeVisitedRoot(std::move(nodes));
}

struct PersistentRouteHistory {
    std::shared_ptr<const PersistentRouteHistory> parent;
    DirectedFiberletStorageId arc;
    DirectedFiberletStorageId logicalArc;
    std::optional<FiberletReplaySourceTransition> enteringTransition;
    std::shared_ptr<const PersistentVisitedNodes> visitedNodes;
    FiberletGraphReplayCost cumulativeEdgeCost;
    FiberletGraphReplayCost cumulativeTransitionCost;
    double cumulativePathLength = 0.0;
    size_t arcCount = 0;
};

struct PersistentRouteCandidate {
    FiberletStorageKey seed;
    std::shared_ptr<const PersistentRouteHistory> tail;
    std::shared_ptr<const PersistentVisitedNodes> visitedNodes;
    FiberletGraphReplayCost edgeCost;
    FiberletGraphReplayCost transitionCost;
    double pathLength = 0.0;
};

std::vector<const PersistentRouteHistory*> persistentRouteHistory(const PersistentRouteCandidate& route)
{
    std::vector<const PersistentRouteHistory*> result;
    if (route.tail != nullptr)
        result.reserve(route.tail->arcCount);
    for (auto node = route.tail; node != nullptr; node = node->parent)
        result.push_back(node.get());
    std::reverse(result.begin(), result.end());
    return result;
}

std::vector<DirectedFiberletStorageId> persistentRouteLogicalArcs(const PersistentRouteCandidate& route)
{
    std::vector<DirectedFiberletStorageId> result;
    for (const auto* node : persistentRouteHistory(route))
        result.push_back(node->logicalArc);
    return result;
}

double persistentRouteLoss(const PersistentRouteCandidate& route)
{
    return route.edgeCost.total() + route.transitionCost.total();
}

void validateReplayCost(const FiberletPathCost& cost, const char* description)
{
    const std::array values{
        cost.invalidPrediction,
        cost.alignment,
        cost.isotropicSmoothness,
        cost.tangentSmoothness,
        cost.normalSmoothness,
    };
    if (std::any_of(values.begin(), values.end(), [](float value) { return !(value >= 0.0F) || !std::isfinite(value); })) {
        throw std::invalid_argument(description);
    }
}

FiberletStorageKey persistentRouteAnchor(const PersistentRouteCandidate& route)
{
    return route.tail == nullptr ? route.seed : targetAnchor(route.tail->arc);
}

std::optional<DirectedFiberletStorageId> persistentRouteIncoming(const PersistentRouteCandidate& route)
{
    if (route.tail == nullptr)
        return std::nullopt;
    return route.tail->arc;
}

struct ExactPersistentStateBudget {
    std::atomic_size_t generated{0};
    size_t maximum = 0;

    void consume()
    {
        const size_t previous = generated.fetch_add(1, std::memory_order_relaxed);
        if (previous >= maximum) {
            throw std::runtime_error("persistent fiberlet beam exceeded the route-state limit");
        }
    }
};

std::vector<PersistentRouteCandidate> persistentRouteSuccessors(
    const FiberletReplayGraphSource& graph,
    const PersistentRouteCandidate& route,
    const std::optional<cv::Vec3d>& initialDirection,
    ExactPersistentStateBudget& stateBudget,
    size_t& generatedStates,
    size_t& rejectedStates)
{
    std::vector<PersistentRouteCandidate> result;
    const auto anchor = persistentRouteAnchor(route);
    const auto incomingId = persistentRouteIncoming(route);
    const auto incoming = incomingId.has_value() ? std::make_optional(graph.arc(*incomingId)) : std::nullopt;
    auto outgoing = graph.outgoing(anchor);
    std::sort(outgoing.begin(), outgoing.end());
    result.reserve(outgoing.size());
    for (const auto& outgoingId : outgoing) {
        const auto edge = graph.arc(outgoingId);
        if (persistentVisitedContains(route.visitedNodes, edge.target)) {
            ++rejectedStates;
            continue;
        }
        std::optional<FiberletReplaySourceTransition> transition;
        if (incoming.has_value()) {
            transition = graph.transition(*incoming, edge);
            if (!transition.has_value()) {
                ++rejectedStates;
                continue;
            }
        } else if (initialDirection.has_value() && !(angleDegrees(*initialDirection, sourceArcStartDirection(edge)) < graph.maximumJoinAngleDegrees())) {
            ++rejectedStates;
            continue;
        }
        stateBudget.consume();
        ++generatedStates;
        const double edgeLength = edge.pathLengthPredictionVoxels;
        if (!(edgeLength > kFloatEpsilon) || !std::isfinite(edgeLength))
            throw std::invalid_argument("persistent beam edge length is invalid");
        validateReplayCost(edge.cost, "persistent beam edge cost must be finite and nonnegative");
        if (transition.has_value()) {
            validateReplayCost(transition->cost, "persistent beam join cost must be finite and nonnegative");
        }
        PersistentRouteCandidate next = route;
        next.edgeCost += edge.cost;
        if (transition.has_value())
            next.transitionCost += transition->cost;
        next.pathLength += edgeLength;
        next.visitedNodes = persistentVisitedAdd(route.visitedNodes, edge.target);
        next.tail = std::make_shared<PersistentRouteHistory>(PersistentRouteHistory{
            route.tail,
            outgoingId,
            graph.logicalArcId(outgoingId),
            transition,
            next.visitedNodes,
            next.edgeCost,
            next.transitionCost,
            next.pathLength,
            route.tail == nullptr ? 1 : route.tail->arcCount + 1});
        result.push_back(std::move(next));
    }
    return result;
}

struct ExactPersistentRouteScore {
    FiberletGraphReplayCost edgeCost;
    FiberletGraphReplayCost transitionCost;
    double scoredLength = 0.0;

    [[nodiscard]] double total() const noexcept { return edgeCost.total() + transitionCost.total(); }
};

ExactPersistentRouteScore scorePersistentRouteAtHorizon(const FiberletReplayGraphSource& graph, const PersistentRouteCandidate& route, double horizonPredictionVoxels)
{
    if (!(horizonPredictionVoxels >= 0.0) || !std::isfinite(horizonPredictionVoxels) || route.pathLength < horizonPredictionVoxels - kReplayEpsilon) {
        throw std::invalid_argument("persistent route does not cover its scoring horizon");
    }
    ExactPersistentRouteScore score;
    if (horizonPredictionVoxels <= kReplayEpsilon) {
        score.scoredLength = horizonPredictionVoxels;
        return score;
    }
    auto crossing = route.tail;
    while (crossing != nullptr && crossing->parent != nullptr && crossing->parent->cumulativePathLength >= horizonPredictionVoxels - kReplayEpsilon) {
        crossing = crossing->parent;
    }
    if (crossing == nullptr || crossing->cumulativePathLength < horizonPredictionVoxels - kReplayEpsilon) {
        throw std::logic_error("persistent route scoring ended before horizon");
    }
    const double precedingLength = crossing->parent != nullptr ? crossing->parent->cumulativePathLength : 0.0;
    const auto edge = graph.arc(crossing->arc);
    const double edgeLength = edge.pathLengthPredictionVoxels;
    if (!(edgeLength > kFloatEpsilon) || !std::isfinite(edgeLength))
        throw std::invalid_argument("persistent beam edge length is invalid");
    validateReplayCost(edge.cost, "persistent beam edge cost must be finite and nonnegative");
    if (crossing->parent != nullptr) {
        score.edgeCost = crossing->parent->cumulativeEdgeCost;
        score.transitionCost = crossing->parent->cumulativeTransitionCost;
    }
    const double includedLength = horizonPredictionVoxels - precedingLength;
    if (!(includedLength > kReplayEpsilon) || includedLength > edgeLength + kReplayEpsilon) {
        throw std::logic_error("persistent route terminal scoring fraction is invalid");
    }
    if (crossing->enteringTransition.has_value()) {
        validateReplayCost(crossing->enteringTransition->cost, "persistent beam join cost must be finite and nonnegative");
        score.transitionCost += crossing->enteringTransition->cost;
    }
    addScaledCost(score.edgeCost, edge.cost, std::clamp(includedLength / edgeLength, 0.0, 1.0));
    score.scoredLength = horizonPredictionVoxels;
    return score;
}

PersistentRouteCandidate persistentRouteCommittedPrefix(const FiberletReplayGraphSource& graph, const PersistentRouteCandidate& route, double checkpointPredictionVoxels)
{
    (void)graph;
    if (!(checkpointPredictionVoxels >= 0.0) || !std::isfinite(checkpointPredictionVoxels) || route.pathLength < checkpointPredictionVoxels - kReplayEpsilon) {
        throw std::invalid_argument("persistent beam route does not cover the checkpoint");
    }
    PersistentRouteCandidate prefix;
    prefix.seed = route.seed;
    auto visited = makeVisitedRoot(std::set<FiberletStorageKey>{route.seed});
    prefix.visitedNodes = visited;
    if (checkpointPredictionVoxels <= kReplayEpsilon)
        return prefix;
    auto crossing = route.tail;
    while (crossing != nullptr && crossing->parent != nullptr && crossing->parent->cumulativePathLength >= checkpointPredictionVoxels - kReplayEpsilon) {
        crossing = crossing->parent;
    }
    if (crossing == nullptr || crossing->cumulativePathLength < checkpointPredictionVoxels - kReplayEpsilon) {
        throw std::logic_error("persistent beam checkpoint prefix was not found");
    }
    prefix.tail = crossing;
    prefix.edgeCost = crossing->cumulativeEdgeCost;
    prefix.transitionCost = crossing->cumulativeTransitionCost;
    prefix.pathLength = crossing->cumulativePathLength;
    prefix.visitedNodes = crossing->visitedNodes;
    return prefix;
}

std::vector<cv::Vec3d> persistentRoutePoints(const FiberletReplayGraphSource& graph, const PersistentRouteCandidate& route, const cv::Vec3d& seedPoint)
{
    std::vector<cv::Vec3d> result{seedPoint};
    for (const auto* node : persistentRouteHistory(route)) {
        auto points = graph.routePoints(node->arc);
        if (!points.empty() && !result.empty())
            points.erase(points.begin());
        result.insert(result.end(), points.begin(), points.end());
    }
    return result;
}

struct RankedPersistentPrefix {
    PersistentRouteCandidate prefix;
    PersistentRouteCandidate lookahead;
    std::vector<DirectedFiberletStorageId> prefixLogicalArcs;
    std::vector<DirectedFiberletStorageId> lookaheadLogicalArcs;
    FiberletGraphReplayCost scoredEdgeCost;
    FiberletGraphReplayCost scoredTransitionCost;
    double scoredPathLength = 0.0;
    double completePathLength = 0.0;
    double lossPerPredictionVoxel = 0.0;
    double totalLoss = 0.0;
};

RankedPersistentPrefix makeRankedPersistentPrefix(
    const FiberletReplayGraphSource& graph, PersistentRouteCandidate candidate, double checkpointPredictionVoxels, double scoringHorizonPredictionVoxels)
{
    RankedPersistentPrefix entry;
    const auto score = scorePersistentRouteAtHorizon(graph, candidate, scoringHorizonPredictionVoxels);
    entry.prefix = persistentRouteCommittedPrefix(graph, candidate, checkpointPredictionVoxels);
    entry.lookahead = std::move(candidate);
    entry.scoredEdgeCost = score.edgeCost;
    entry.scoredTransitionCost = score.transitionCost;
    entry.scoredPathLength = score.scoredLength;
    entry.completePathLength = entry.lookahead.pathLength;
    entry.totalLoss = score.total();
    entry.lossPerPredictionVoxel =
        entry.scoredPathLength > kReplayEpsilon ? entry.totalLoss / entry.scoredPathLength : std::numeric_limits<double>::infinity();
    return entry;
}

bool rankedPersistentPrefixLess(const RankedPersistentPrefix& left, const RankedPersistentPrefix& right)
{
    if (left.totalLoss != right.totalLoss)
        return left.totalLoss < right.totalLoss;
    const auto leftLookahead = persistentRouteLogicalArcs(left.lookahead);
    const auto rightLookahead = persistentRouteLogicalArcs(right.lookahead);
    if (leftLookahead != rightLookahead)
        return leftLookahead < rightLookahead;
    return persistentRouteLogicalArcs(left.prefix) < persistentRouteLogicalArcs(right.prefix);
}

void retainRankedPersistentPrefix(RankedPersistentPrefix entry, size_t beamWidth, std::vector<RankedPersistentPrefix>& ranked)
{
    const auto entryPrefixArcs = persistentRouteLogicalArcs(entry.prefix);
    const auto equivalent = std::find_if(ranked.begin(), ranked.end(), [&](const auto& existing) {
        return existing.prefix.seed == entry.prefix.seed && persistentRouteLogicalArcs(existing.prefix) == entryPrefixArcs;
    });
    if (equivalent != ranked.end()) {
        if (rankedPersistentPrefixLess(entry, *equivalent))
            *equivalent = std::move(entry);
    } else {
        ranked.push_back(std::move(entry));
    }
    std::sort(ranked.begin(), ranked.end(), rankedPersistentPrefixLess);
    if (ranked.size() > beamWidth)
        ranked.resize(beamWidth);
}

struct ExactPersistentSearchStats {
    size_t generated = 0;
    size_t expanded = 0;
    size_t completed = 0;
    size_t costPruned = 0;
    size_t rejected = 0;
    size_t dominated = 0;
};

struct PersistentQueueEntry {
    PersistentRouteCandidate route;
    double lowerBound = 0.0;
    size_t sequence = 0;
};

struct PersistentQueueGreater {
    bool operator()(const PersistentQueueEntry& left, const PersistentQueueEntry& right) const noexcept
    {
        return std::tie(left.lowerBound, left.sequence) > std::tie(right.lowerBound, right.sequence);
    }
};

class RelaxedPersistentCostToGo
{
public:
    explicit RelaxedPersistentCostToGo(const FiberletReplayGraphSource& graph) : graph_(graph) {}

    double lowerBound(const DirectedFiberletStorageId& incoming, double remainingPredictionVoxels)
    {
        if (!(remainingPredictionVoxels >= 0.0) || !std::isfinite(remainingPredictionVoxels)) {
            throw std::invalid_argument("persistent route remaining distance is invalid");
        }
        std::scoped_lock lock(mutex_);
        return lowerBoundUnlocked(incoming, remainingPredictionVoxels);
    }

private:
    double lowerBoundUnlocked(const DirectedFiberletStorageId& incoming, double remainingPredictionVoxels)
    {
        size_t bins = static_cast<size_t>(std::floor(remainingPredictionVoxels / kDistanceBin));
        while (bins > 0 && static_cast<double>(bins) * kDistanceBin > remainingPredictionVoxels) {
            --bins;
        }
        return solve(incoming, bins);
    }

    double solve(const DirectedFiberletStorageId& incomingId, size_t bins)
    {
        if (bins == 0)
            return 0.0;
        const auto key = std::pair{incomingId, bins};
        if (const auto found = memo_.find(key); found != memo_.end())
            return found->second;
        const auto incoming = graph_.arc(incomingId);
        const double targetDistance = static_cast<double>(bins) * kDistanceBin;
        double best = std::numeric_limits<double>::infinity();
        auto outgoingIds = graph_.outgoing(incoming.target);
        std::sort(outgoingIds.begin(), outgoingIds.end());
        for (const auto& outgoingId : outgoingIds) {
            const auto outgoing = graph_.arc(outgoingId);
            const auto transition = graph_.transition(incoming, outgoing);
            if (!transition.has_value())
                continue;
            validateReplayCost(transition->cost, "persistent beam join cost must be finite and nonnegative");
            validateReplayCost(outgoing.cost, "persistent beam edge cost must be finite and nonnegative");
            const double edgeLength = outgoing.pathLengthPredictionVoxels;
            if (!(edgeLength > kFloatEpsilon) || !std::isfinite(edgeLength)) {
                throw std::invalid_argument("persistent beam edge length is invalid");
            }
            const double joinCost = transition->cost.total();
            const double edgeCost = outgoing.cost.total();
            double candidate = joinCost;
            if (edgeLength >= targetDistance - kReplayEpsilon) {
                candidate += edgeCost * std::clamp(targetDistance / edgeLength, 0.0, 1.0);
            } else {
                candidate += edgeCost;
                candidate += lowerBoundUnlocked(outgoingId, targetDistance - edgeLength);
            }
            best = std::min(best, candidate);
        }
        memo_.emplace(key, best);
        return best;
    }

    static constexpr double kDistanceBin = 0.5;
    const FiberletReplayGraphSource& graph_;
    std::mutex mutex_;
    std::map<std::pair<DirectedFiberletStorageId, size_t>, double> memo_;
};

std::vector<PersistentRouteCandidate> persistentCheckpointPrefixes(
    const FiberletReplayGraphSource& graph,
    const std::vector<PersistentRouteCandidate>& initialRoutes,
    double checkpointPredictionVoxels,
    const cv::Vec3d& initialDirection,
    ExactPersistentStateBudget& stateBudget,
    ExactPersistentSearchStats& stats)
{
    if (!(checkpointPredictionVoxels >= 0.0) || !std::isfinite(checkpointPredictionVoxels)) {
        throw std::invalid_argument("persistent beam checkpoint is invalid");
    }
    std::vector<PersistentRouteCandidate> pending;
    std::vector<PersistentRouteCandidate> prefixes;
    for (const auto& initial : initialRoutes) {
        if (initial.pathLength >= checkpointPredictionVoxels - kReplayEpsilon)
            prefixes.push_back(initial);
        else
            pending.push_back(initial);
    }
    for (size_t index = 0; index < pending.size(); ++index) {
        auto route = std::move(pending[index]);
        ++stats.expanded;
        auto successors = persistentRouteSuccessors(
            graph, route, route.tail == nullptr ? std::make_optional(initialDirection) : std::nullopt, stateBudget, stats.generated, stats.rejected);
        for (auto successor : successors) {
            if (successor.pathLength >= checkpointPredictionVoxels - kReplayEpsilon)
                prefixes.push_back(std::move(successor));
            else
                pending.push_back(std::move(successor));
        }
    }
    return prefixes;
}

std::optional<RankedPersistentPrefix> exactPersistentPrefixContinuation(
    const FiberletReplayGraphSource& graph,
    const PersistentRouteCandidate& prefix,
    double scoringHorizonPredictionVoxels,
    double checkpointPredictionVoxels,
    RelaxedPersistentCostToGo& costToGo,
    ExactPersistentStateBudget& stateBudget,
    ExactPersistentSearchStats& stats)
{
    std::priority_queue<PersistentQueueEntry, std::vector<PersistentQueueEntry>, PersistentQueueGreater> pending;
    std::optional<RankedPersistentPrefix> best;
    size_t sequence = 0;
    const auto retainCompletion = [&](PersistentRouteCandidate candidate) {
        ++stats.completed;
        auto entry = makeRankedPersistentPrefix(graph, std::move(candidate), checkpointPredictionVoxels, scoringHorizonPredictionVoxels);
        if (!best.has_value() || rankedPersistentPrefixLess(entry, *best))
            best = std::move(entry);
    };
    const auto routeLowerBound = [&](const PersistentRouteCandidate& route) {
        const double accumulated = persistentRouteLoss(route);
        const auto incoming = persistentRouteIncoming(route);
        if (!incoming.has_value()) {
            throw std::logic_error("persistent lookahead prefix has no incoming fiberlet");
        }
        const double future = costToGo.lowerBound(*incoming, std::max(0.0, scoringHorizonPredictionVoxels - route.pathLength));
        return accumulated + future;
    };
    const auto enqueue = [&](PersistentRouteCandidate candidate, double lowerBound) {
        if (!(lowerBound >= 0.0) || std::isnan(lowerBound)) {
            throw std::invalid_argument("persistent route lower bound must be finite and nonnegative");
        }
        if (!std::isfinite(lowerBound)) {
            ++stats.costPruned;
            return;
        }
        pending.push(PersistentQueueEntry{std::move(candidate), lowerBound, sequence++});
    };
    if (prefix.pathLength >= scoringHorizonPredictionVoxels - kReplayEpsilon) {
        retainCompletion(prefix);
    } else {
        enqueue(prefix, routeLowerBound(prefix));
    }
    while (!pending.empty()) {
        if (best.has_value() && pending.top().lowerBound > best->totalLoss) {
            stats.costPruned += pending.size();
            break;
        }
        auto route = std::move(pending.top().route);
        pending.pop();
        ++stats.expanded;
        auto successors = persistentRouteSuccessors(graph, route, std::nullopt, stateBudget, stats.generated, stats.rejected);
        for (auto successor : successors) {
            if (successor.pathLength >= scoringHorizonPredictionVoxels - kReplayEpsilon) {
                retainCompletion(std::move(successor));
            } else {
                const double lowerBound = routeLowerBound(successor);
                if (best.has_value() && lowerBound > best->totalLoss) {
                    ++stats.costPruned;
                } else {
                    enqueue(std::move(successor), lowerBound);
                }
            }
        }
    }
    return best;
}

std::vector<RankedPersistentPrefix> exactPersistentRouteLookahead(
    const FiberletReplayGraphSource& graph,
    const std::vector<PersistentRouteCandidate>& initialRoutes,
    double scoringHorizonPredictionVoxels,
    double checkpointPredictionVoxels,
    const cv::Vec3d& initialDirection,
    size_t beamWidth,
    size_t expansionThreads,
    size_t maximumGeneratedStates,
    ExactPersistentSearchStats& stats)
{
    if (!(scoringHorizonPredictionVoxels > 0.0) || !std::isfinite(scoringHorizonPredictionVoxels) || !(checkpointPredictionVoxels >= 0.0) ||
        !std::isfinite(checkpointPredictionVoxels)) {
        throw std::invalid_argument("persistent exact lookahead horizon is invalid");
    }
    ExactPersistentStateBudget stateBudget;
    stateBudget.maximum = maximumGeneratedStates;
    auto prefixes = persistentCheckpointPrefixes(graph, initialRoutes, checkpointPredictionVoxels, initialDirection, stateBudget, stats);
    RelaxedPersistentCostToGo costToGo(graph);
    std::vector<std::optional<RankedPersistentPrefix>> completions(prefixes.size());
    std::vector<ExactPersistentSearchStats> prefixStats(prefixes.size());
    std::atomic_size_t nextPrefix{0};
    const size_t workerCount = std::min(std::max<size_t>(1, expansionThreads), prefixes.size());
    std::vector<std::future<void>> workers;
    workers.reserve(workerCount);
    for (size_t worker = 0; worker < workerCount; ++worker) {
        workers.push_back(std::async(std::launch::async, [&]() {
            while (true) {
                const size_t index = nextPrefix.fetch_add(1, std::memory_order_relaxed);
                if (index >= prefixes.size())
                    return;
                completions[index] =
                    exactPersistentPrefixContinuation(graph, prefixes[index], scoringHorizonPredictionVoxels, checkpointPredictionVoxels, costToGo, stateBudget, prefixStats[index]);
            }
        }));
    }
    for (auto& worker : workers)
        worker.get();
    stats.generated = stateBudget.generated.load(std::memory_order_relaxed);
    std::vector<RankedPersistentPrefix> ranked;
    ranked.reserve(beamWidth);
    for (size_t index = 0; index < completions.size(); ++index) {
        stats.expanded += prefixStats[index].expanded;
        stats.completed += prefixStats[index].completed;
        stats.costPruned += prefixStats[index].costPruned;
        stats.rejected += prefixStats[index].rejected;
        if (completions[index].has_value()) {
            retainRankedPersistentPrefix(std::move(*completions[index]), beamWidth, ranked);
        }
    }
    for (auto& entry : ranked) {
        entry.prefixLogicalArcs = persistentRouteLogicalArcs(entry.prefix);
        entry.lookaheadLogicalArcs = persistentRouteLogicalArcs(entry.lookahead);
    }
    return ranked;
}

struct BoundedPersistentRoute {
    PersistentRouteCandidate route;
    std::vector<DirectedFiberletStorageId> stablePrefixLogicalArcs;
};

struct BoundedRankedPersistentRoute {
    RankedPersistentPrefix ranked;
    std::vector<DirectedFiberletStorageId> stablePrefixLogicalArcs;
};

using StablePersistentPrefixKey = std::pair<FiberletStorageKey, std::vector<DirectedFiberletStorageId>>;

StablePersistentPrefixKey stablePersistentPrefixKey(const BoundedRankedPersistentRoute& route)
{
    return {route.ranked.lookahead.seed, route.stablePrefixLogicalArcs};
}

StablePersistentPrefixKey completePersistentRouteKey(const BoundedRankedPersistentRoute& route)
{
    return {route.ranked.lookahead.seed, route.ranked.lookaheadLogicalArcs};
}

bool boundedRankedPersistentRouteLess(const BoundedRankedPersistentRoute& left, const BoundedRankedPersistentRoute& right)
{
    if (rankedPersistentPrefixLess(left.ranked, right.ranked))
        return true;
    if (rankedPersistentPrefixLess(right.ranked, left.ranked))
        return false;
    return stablePersistentPrefixKey(left) < stablePersistentPrefixKey(right);
}

void retainLocalBoundedRoute(BoundedRankedPersistentRoute entry, size_t searchWidth, std::vector<BoundedRankedPersistentRoute>& retained)
{
    const auto key = completePersistentRouteKey(entry);
    const auto equivalent =
        std::find_if(retained.begin(), retained.end(), [&](const auto& existing) { return completePersistentRouteKey(existing) == key; });
    if (equivalent != retained.end()) {
        if (boundedRankedPersistentRouteLess(entry, *equivalent))
            *equivalent = std::move(entry);
    } else {
        retained.push_back(std::move(entry));
    }
    std::sort(retained.begin(), retained.end(), boundedRankedPersistentRouteLess);
    if (retained.size() > searchWidth)
        retained.resize(searchWidth);
}

void retainLocalBoundedPrefixRepresentative(BoundedRankedPersistentRoute entry, size_t searchWidth, std::vector<BoundedRankedPersistentRoute>& retained)
{
    const auto key = stablePersistentPrefixKey(entry);
    const auto equivalent =
        std::find_if(retained.begin(), retained.end(), [&](const auto& existing) { return stablePersistentPrefixKey(existing) == key; });
    if (equivalent != retained.end()) {
        if (boundedRankedPersistentRouteLess(entry, *equivalent))
            *equivalent = std::move(entry);
    } else {
        retained.push_back(std::move(entry));
    }
    std::sort(retained.begin(), retained.end(), boundedRankedPersistentRouteLess);
    if (retained.size() > searchWidth)
        retained.resize(searchWidth);
}

struct BoundedPersistentExpansionResult {
    std::vector<BoundedRankedPersistentRoute> prefixRepresentatives;
    std::vector<BoundedRankedPersistentRoute> globalCandidates;
    ExactPersistentSearchStats stats;
    std::optional<double> appliedLocalCompletionLossCutoffPerPredictionVoxel;
};

constexpr double kPersistentLabelDistanceBin = 0.5;

struct PersistentLabelKey {
    DirectedFiberletStorageId incoming;
    size_t remainingDistanceBin = 0;

    auto operator<=>(const PersistentLabelKey&) const = default;
};

struct PersistentLabelWinner {
    double accumulatedLoss = 0.0;
    PersistentRouteCandidate route;
};

PersistentLabelKey persistentLabelKey(const FiberletReplayGraphSource& graph, const PersistentRouteCandidate& route, double frontPredictionVoxels)
{
    const auto incoming = persistentRouteIncoming(route);
    if (!incoming.has_value())
        throw std::logic_error("bounded front label has no incoming fiberlet");
    const double frontOffset = std::abs(frontPredictionVoxels - route.pathLength);
    return {
        graph.logicalArcId(*incoming),
        static_cast<size_t>(std::floor(frontOffset / kPersistentLabelDistanceBin)),
    };
}

bool persistentLabelLess(const PersistentRouteCandidate& route, double loss, const PersistentLabelWinner& winner)
{
    if (loss != winner.accumulatedLoss)
        return loss < winner.accumulatedLoss;
    return persistentRouteLogicalArcs(route) < persistentRouteLogicalArcs(winner.route);
}

BoundedPersistentExpansionResult expandPersistentRouteToFront(
    const FiberletReplayGraphSource& graph,
    const BoundedPersistentRoute& initial,
    double frontPredictionVoxels,
    double frontBeginPredictionVoxels,
    double checkpointPredictionVoxels,
    double stablePrefixPredictionVoxels,
    const cv::Vec3d& initialDirection,
    size_t searchWidth,
    ExactPersistentStateBudget& stateBudget)
{
    BoundedPersistentExpansionResult result;
    std::priority_queue<PersistentQueueEntry, std::vector<PersistentQueueEntry>, PersistentQueueGreater> pending;
    std::map<PersistentLabelKey, PersistentLabelWinner> bestLabels;
    std::map<PersistentLabelKey, BoundedRankedPersistentRoute> bestCompletions;
    std::multiset<double> completionLosses;
    std::optional<double> completionCutoff;
    std::optional<double> appliedLocalCompletionCutoffPerPredictionVoxel;
    size_t sequence = 0;
    const auto routeLowerBound = [](const PersistentRouteCandidate& route) { return persistentRouteLoss(route); };
    const auto enqueue = [&](PersistentRouteCandidate route) {
        const auto key = persistentLabelKey(graph, route, frontPredictionVoxels);
        const double accumulatedLoss = persistentRouteLoss(route);
        const auto found = bestLabels.find(key);
        if (found != bestLabels.end() && !persistentLabelLess(route, accumulatedLoss, found->second)) {
            ++result.stats.dominated;
            return;
        }
        bestLabels.insert_or_assign(key, PersistentLabelWinner{accumulatedLoss, route});
        const double lowerBound = routeLowerBound(route);
        if (!(lowerBound >= 0.0) || std::isnan(lowerBound))
            throw std::invalid_argument("bounded front lower bound must be nonnegative");
        if (!std::isfinite(lowerBound)) {
            ++result.stats.costPruned;
            return;
        }
        pending.push({std::move(route), lowerBound, sequence++});
    };
    if (initial.route.pathLength >= frontPredictionVoxels - kReplayEpsilon) {
        pending.push({initial.route, persistentRouteLoss(initial.route), sequence++});
    } else {
        enqueue(initial.route);
    }
    while (!pending.empty()) {
        if (completionCutoff.has_value() && pending.top().lowerBound > *completionCutoff) {
            const double frontLength = frontPredictionVoxels - frontBeginPredictionVoxels;
            if (frontLength > kReplayEpsilon) {
                const double beginLoss = scorePersistentRouteAtHorizon(graph, initial.route, frontBeginPredictionVoxels).total();
                appliedLocalCompletionCutoffPerPredictionVoxel = std::max(0.0, (*completionCutoff - beginLoss) / frontLength);
            }
            result.stats.costPruned += pending.size();
            break;
        }
        auto route = std::move(pending.top().route);
        pending.pop();
        if (route.pathLength < frontPredictionVoxels - kReplayEpsilon) {
            const auto key = persistentLabelKey(graph, route, frontPredictionVoxels);
            const auto found = bestLabels.find(key);
            if (found == bestLabels.end() || found->second.route.tail != route.tail) {
                ++result.stats.dominated;
                continue;
            }
        }
        if (route.pathLength >= frontPredictionVoxels - kReplayEpsilon) {
            ++result.stats.completed;
            auto ranked = makeRankedPersistentPrefix(graph, std::move(route), checkpointPredictionVoxels, frontPredictionVoxels);
            ranked.prefixLogicalArcs = persistentRouteLogicalArcs(ranked.prefix);
            ranked.lookaheadLogicalArcs = persistentRouteLogicalArcs(ranked.lookahead);
            auto stablePrefix = initial.stablePrefixLogicalArcs;
            if (stablePrefix.empty()) {
                stablePrefix = persistentRouteLogicalArcs(persistentRouteCommittedPrefix(graph, ranked.lookahead, stablePrefixPredictionVoxels));
            }
            BoundedRankedPersistentRoute completed{std::move(ranked), std::move(stablePrefix)};
            const auto completionKey = persistentLabelKey(graph, completed.ranked.lookahead, frontPredictionVoxels);
            const auto found = bestCompletions.find(completionKey);
            if (found == bestCompletions.end() || boundedRankedPersistentRouteLess(completed, found->second)) {
                if (found != bestCompletions.end()) {
                    const auto previousLoss = completionLosses.find(found->second.ranked.totalLoss);
                    if (previousLoss == completionLosses.end())
                        throw std::logic_error("bounded front completion cutoff is inconsistent");
                    completionLosses.erase(previousLoss);
                }
                completionLosses.insert(completed.ranked.totalLoss);
                bestCompletions.insert_or_assign(completionKey, std::move(completed));
                if (completionLosses.size() >= searchWidth)
                    completionCutoff = *std::next(completionLosses.begin(), static_cast<std::ptrdiff_t>(searchWidth - 1));
                else
                    completionCutoff.reset();
            } else {
                ++result.stats.dominated;
            }
            continue;
        }
        ++result.stats.expanded;
        auto successors = persistentRouteSuccessors(
            graph,
            route,
            route.tail == nullptr ? std::make_optional(initialDirection) : std::nullopt,
            stateBudget,
            result.stats.generated,
            result.stats.rejected);
        for (auto& successor : successors) {
            if (successor.pathLength >= frontPredictionVoxels - kReplayEpsilon) {
                const auto score = scorePersistentRouteAtHorizon(graph, successor, frontPredictionVoxels);
                pending.push({std::move(successor), score.total(), sequence++});
            } else {
                enqueue(std::move(successor));
            }
        }
    }
    result.globalCandidates.reserve(std::min(searchWidth, bestCompletions.size()));
    for (const auto& [key, candidate] : bestCompletions) {
        (void)key;
        retainLocalBoundedRoute(candidate, searchWidth, result.globalCandidates);
    }
    for (const auto& candidate : result.globalCandidates)
        retainLocalBoundedPrefixRepresentative(candidate, searchWidth, result.prefixRepresentatives);
    result.appliedLocalCompletionLossCutoffPerPredictionVoxel = appliedLocalCompletionCutoffPerPredictionVoxel;
    return result;
}

std::vector<BoundedRankedPersistentRoute> selectBoundedFront(
    std::vector<BoundedRankedPersistentRoute> candidates, size_t retainedWidth, bool finalFront, FiberletGraphReplayPruneFront& diagnostics)
{
    std::sort(candidates.begin(), candidates.end(), boundedRankedPersistentRouteLess);
    std::vector<BoundedRankedPersistentRoute> unique;
    unique.reserve(candidates.size());
    std::set<StablePersistentPrefixKey> completeRoutes;
    for (auto& candidate : candidates) {
        if (completeRoutes.insert(completePersistentRouteKey(candidate)).second)
            unique.push_back(std::move(candidate));
    }

    std::vector<BoundedRankedPersistentRoute> selected;
    selected.reserve(std::min(retainedWidth, unique.size()));
    std::set<StablePersistentPrefixKey> selectedPrefixes;
    std::set<StablePersistentPrefixKey> selectedRoutes;
    for (auto& candidate : unique) {
        if (selected.size() >= retainedWidth)
            break;
        const StablePersistentPrefixKey prefixKey = finalFront ? StablePersistentPrefixKey{candidate.ranked.prefix.seed, candidate.ranked.prefixLogicalArcs}
                                                               : stablePersistentPrefixKey(candidate);
        if (!selectedPrefixes.insert(prefixKey).second)
            continue;
        selectedRoutes.insert(completePersistentRouteKey(candidate));
        selected.push_back(candidate);
        ++diagnostics.diversityProtectedCount;
    }
    std::set<StablePersistentPrefixKey> allPrefixes;
    for (const auto& candidate : unique) {
        allPrefixes.insert(
            finalFront ? StablePersistentPrefixKey{candidate.ranked.prefix.seed, candidate.ranked.prefixLogicalArcs}
                       : stablePersistentPrefixKey(candidate));
    }
    diagnostics.distinctPrefixCount = allPrefixes.size();
    if (!finalFront && selected.size() < retainedWidth) {
        for (auto& candidate : unique) {
            if (selected.size() >= retainedWidth)
                break;
            if (!selectedRoutes.insert(completePersistentRouteKey(candidate)).second)
                continue;
            selected.push_back(candidate);
            ++diagnostics.globalFillCount;
        }
    }
    std::sort(selected.begin(), selected.end(), boundedRankedPersistentRouteLess);
    diagnostics.retainedRouteCount = selected.size();
    diagnostics.prunedCandidateCount =
        diagnostics.completedCandidateCount > selected.size() ? diagnostics.completedCandidateCount - selected.size() : 0;
    diagnostics.searchWidthBound = selected.size() == retainedWidth && unique.size() > selected.size();
    return selected;
}

struct BoundedPersistentLookaheadResult {
    std::vector<RankedPersistentPrefix> ranked;
    std::vector<FiberletGraphReplayPruneFront> fronts;
    std::vector<DirectedFiberletStorageId> selectedPrefixLogicalArcs;
};

BoundedPersistentLookaheadResult boundedPersistentRouteLookahead(
    const FiberletReplayGraphSource& graph,
    const std::vector<PersistentRouteCandidate>& initialRoutes,
    double scoringHorizonPredictionVoxels,
    double rolloutBeginPredictionVoxels,
    double checkpointPredictionVoxels,
    double pruneDistancePredictionVoxels,
    const cv::Vec3d& initialDirection,
    size_t beamWidth,
    size_t searchWidth,
    size_t expansionThreads,
    size_t maximumGeneratedStates,
    ExactPersistentSearchStats& stats)
{
    if (!(scoringHorizonPredictionVoxels > 0.0) || !std::isfinite(scoringHorizonPredictionVoxels) || !(checkpointPredictionVoxels >= 0.0) ||
        !std::isfinite(checkpointPredictionVoxels) || !(pruneDistancePredictionVoxels > 0.0) ||
        !std::isfinite(pruneDistancePredictionVoxels) || searchWidth < beamWidth) {
        throw std::invalid_argument("persistent bounded lookahead configuration is invalid");
    }

    const double firstFront = std::min(scoringHorizonPredictionVoxels, checkpointPredictionVoxels);
    std::vector<double> fronts{firstFront};
    while (fronts.back() < scoringHorizonPredictionVoxels - kReplayEpsilon) {
        fronts.push_back(std::min(scoringHorizonPredictionVoxels, fronts.back() + pruneDistancePredictionVoxels));
    }

    ExactPersistentStateBudget stateBudget;
    stateBudget.maximum = maximumGeneratedStates;
    std::vector<BoundedPersistentRoute> active;
    active.reserve(initialRoutes.size());
    for (const auto& route : initialRoutes)
        active.push_back({route, {}});

    BoundedPersistentLookaheadResult result;
    for (size_t frontIndex = 0; frontIndex < fronts.size(); ++frontIndex) {
        FiberletGraphReplayPruneFront frontDiagnostics;
        frontDiagnostics.horizonPathLengthPredictionVoxels = fronts[frontIndex];
        frontDiagnostics.inputRouteCount = active.size();
        const bool finalFront = frontIndex + 1 == fronts.size();
        const double frontBeginPredictionVoxels = frontIndex == 0 ? rolloutBeginPredictionVoxels : fronts[frontIndex - 1];
        const size_t targetWidth = finalFront ? beamWidth : searchWidth;
        size_t localCandidateLimit = targetWidth;
        if (frontIndex > 0) {
            std::set<StablePersistentPrefixKey> activePrefixes;
            for (const auto& route : active)
                activePrefixes.insert({route.route.seed, route.stablePrefixLogicalArcs});
            const size_t globalFillSlots = targetWidth > activePrefixes.size() ? targetWidth - activePrefixes.size() : 0;
            localCandidateLimit = std::min(targetWidth, globalFillSlots + 1);
        }
        frontDiagnostics.localCandidateLimit = localCandidateLimit;
        if (active.size() == 1 && active.front().route.tail == nullptr && active.front().route.pathLength < fronts[frontIndex] - kReplayEpsilon) {
            ++frontDiagnostics.expandedStateCount;
            size_t rootGenerated = 0;
            size_t rootRejected = 0;
            auto successors =
                persistentRouteSuccessors(graph, active.front().route, active.front().route.tail == nullptr ? std::make_optional(initialDirection) : std::nullopt, stateBudget, rootGenerated, rootRejected);
            frontDiagnostics.generatedStateCount += rootGenerated;
            frontDiagnostics.rejectedStateCount += rootRejected;
            const auto stablePrefix = active.front().stablePrefixLogicalArcs;
            active.clear();
            active.reserve(successors.size());
            for (auto& successor : successors)
                active.push_back({std::move(successor), stablePrefix});
        }
        std::vector<BoundedPersistentExpansionResult> expanded(active.size());
        std::atomic_size_t nextInput{0};
        const size_t workerCount = std::min(std::max<size_t>(1, expansionThreads), active.size());
        std::vector<std::future<void>> workers;
        workers.reserve(workerCount);
        for (size_t worker = 0; worker < workerCount; ++worker) {
            workers.push_back(std::async(std::launch::async, [&]() {
                while (true) {
                    const size_t inputIndex = nextInput.fetch_add(1, std::memory_order_relaxed);
                    if (inputIndex >= active.size())
                        return;
                    expanded[inputIndex] =
                        expandPersistentRouteToFront(graph, active[inputIndex], fronts[frontIndex], frontBeginPredictionVoxels, checkpointPredictionVoxels, firstFront, initialDirection, localCandidateLimit, stateBudget);
                }
            }));
        }
        std::exception_ptr workerError;
        for (auto& worker : workers) {
            try {
                worker.get();
            } catch (...) {
                if (workerError == nullptr)
                    workerError = std::current_exception();
            }
        }
        if (workerError != nullptr)
            std::rethrow_exception(workerError);

        std::vector<BoundedRankedPersistentRoute> candidates;
        for (auto& expansion : expanded) {
            frontDiagnostics.generatedStateCount += expansion.stats.generated;
            frontDiagnostics.expandedStateCount += expansion.stats.expanded;
            frontDiagnostics.rejectedStateCount += expansion.stats.rejected;
            frontDiagnostics.dominatedStateCount += expansion.stats.dominated;
            frontDiagnostics.costPrunedStateCount += expansion.stats.costPruned;
            frontDiagnostics.completedCandidateCount += expansion.stats.completed;
            if (expansion.appliedLocalCompletionLossCutoffPerPredictionVoxel.has_value()) {
                if (!frontDiagnostics.minimumAppliedLocalCompletionLossCutoffPerPredictionVoxel.has_value())
                    frontDiagnostics.minimumAppliedLocalCompletionLossCutoffPerPredictionVoxel = expansion.appliedLocalCompletionLossCutoffPerPredictionVoxel;
                else
                    frontDiagnostics.minimumAppliedLocalCompletionLossCutoffPerPredictionVoxel =
                        std::min(*frontDiagnostics.minimumAppliedLocalCompletionLossCutoffPerPredictionVoxel, *expansion.appliedLocalCompletionLossCutoffPerPredictionVoxel);
            }
            candidates.insert(
                candidates.end(),
                std::make_move_iterator(expansion.prefixRepresentatives.begin()),
                std::make_move_iterator(expansion.prefixRepresentatives.end()));
            candidates.insert(
                candidates.end(),
                std::make_move_iterator(expansion.globalCandidates.begin()),
                std::make_move_iterator(expansion.globalCandidates.end()));
        }
        auto selected = selectBoundedFront(std::move(candidates), finalFront ? beamWidth : searchWidth, finalFront, frontDiagnostics);
        frontDiagnostics.cumulativeGeneratedStateCount = stateBudget.generated.load(std::memory_order_relaxed);
        result.fronts.push_back(frontDiagnostics);
        stats.expanded += frontDiagnostics.expandedStateCount;
        stats.completed += frontDiagnostics.completedCandidateCount;
        stats.rejected += frontDiagnostics.rejectedStateCount;
        stats.dominated += frontDiagnostics.dominatedStateCount;
        stats.costPruned += frontDiagnostics.costPrunedStateCount;

        if (finalFront) {
            result.ranked.reserve(selected.size());
            for (auto& candidate : selected)
                result.ranked.push_back(std::move(candidate.ranked));
            if (!selected.empty()) {
                result.selectedPrefixLogicalArcs = selected.front().stablePrefixLogicalArcs;
            }
        } else {
            active.clear();
            active.reserve(selected.size());
            for (auto& candidate : selected) {
                active.push_back({std::move(candidate.ranked.lookahead), std::move(candidate.stablePrefixLogicalArcs)});
            }
            if (active.empty())
                break;
        }
    }
    stats.generated = stateBudget.generated.load(std::memory_order_relaxed);
    return result;
}

class EagerReplayGraphSource final : public FiberletReplayGraphSource
{
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
        const PolylineArcGeometry& reference, double beginArcBase, double endArcBase, double broadPhaseRadiusBaseVoxels) const override
    {
        std::vector<FiberletReplaySourceAnchor> result;
        for (const auto& node : graph_.nodes) {
            const auto projection = projectPointToPolylineArc(reference, cv::Vec3d(node.positionBaseXYZ), beginArcBase, endArcBase);
            if (projection.arc + kReplayEpsilon < beginArcBase || projection.arc > endArcBase + kReplayEpsilon || projection.distance > broadPhaseRadiusBaseVoxels)
                continue;
            const auto id = storageKey(node.anchor);
            result.push_back({id, node.positionBaseXYZ});
        }
        std::sort(result.begin(), result.end(), [](const auto& left, const auto& right) { return left.id < right.id; });
        return result;
    }

    std::vector<DirectedFiberletStorageId> outgoing(const FiberletStorageKey& anchor) const override
    {
        const auto found =
            std::find_if(graph_.nodes.begin(), graph_.nodes.end(), [&](const auto& node) { return storageKey(node.anchor) == anchor; });
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

    std::vector<cv::Vec3d> routePoints(const DirectedFiberletStorageId& id) const override
    {
        const auto found = arcById_.find(id);
        if (found == arcById_.end())
            throw std::out_of_range("fiberlet replay arc is absent");
        return orientedArcPoints(graph_, found->second);
    }

    std::optional<FiberletReplaySourceTransition> transition(const FiberletReplaySourceArc& incoming, const FiberletReplaySourceArc& outgoing) const override
    {
        const auto incomingFound = arcById_.find(incoming.id);
        const auto outgoingFound = arcById_.find(outgoing.id);
        if (incomingFound == arcById_.end() || outgoingFound == arcById_.end())
            throw std::out_of_range("fiberlet replay transition arc is absent");
        const auto index = transitionIndex(graph_, incomingFound->second, outgoingFound->second);
        if (!index.has_value())
            return std::nullopt;
        return FiberletReplaySourceTransition{incoming.id, outgoing.id, graph_.transitions[*index].cost, *index};
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
    return length(a) > kFloatEpsilon && length(b) > kFloatEpsilon && std::abs(a.dot(b)) >= 1.0F - 1.0e-5F;
}

}  // namespace

FiberletGraph buildFiberletGraph(const FiberletPathReport& paths, float maximumJoinAngleDegrees)
{
    const float predictionToBaseScale = static_cast<float>(paths.grid.predictionToBaseScale);
    if (!(paths.grid.predictionToBaseScale > 0.0) || !std::isfinite(paths.grid.predictionToBaseScale) ||
        !(maximumJoinAngleDegrees >= 0.0) || !(predictionToBaseScale > 0.0F) || !std::isfinite(predictionToBaseScale) ||
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
    const float minimumJoinDot = std::cos(maximumJoinAngleDegrees * static_cast<float>(kPi / 180.0));
    for (size_t node = 0; node < graph.nodes.size(); ++node) {
        const auto& arcs = graph.nodes[node].outgoingArcs;
        for (const size_t incomingArc : arcs) {
            const size_t directedIncoming = incomingArc ^ 1U;
            const cv::Vec3f incomingDirection = arcEndDirection(graph, directedIncoming);
            for (const size_t outgoingArc : arcs) {
                if (arcEdge(directedIncoming) == arcEdge(outgoingArc))
                    continue;
                const cv::Vec3f outgoingDirection = arcStartDirection(graph, outgoingArc);
                const float angle = angleDegrees(incomingDirection, outgoingDirection);
                if (incomingDirection.dot(outgoingDirection) > minimumJoinDot && graph.nodes[node].prediction.valid) {
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
                                paths.config.smoothnessFreeAngleDegrees * static_cast<float>(kPi / 180.0)}});
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
    return traceFiberletGraphReplay(source, referencePointsBaseXYZ, normalSampler, normalWorkingToBaseScale, config, failureCallback, progressCallback);
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
    const double predictionToBaseScale = graph.predictionToBaseScale();
    if (!(predictionToBaseScale > 0.0) || !std::isfinite(predictionToBaseScale) || config.beamWidth < 1 || config.beamWidth > 16 ||
        config.expansionThreads < 1 || config.maximumGeneratedStatesPerIteration == 0 || !(config.beamStepDistanceBaseVoxels > 0.0) ||
        !std::isfinite(config.beamStepDistanceBaseVoxels) || !(config.lookaheadDistanceBaseVoxels > 0.0) ||
        !std::isfinite(config.lookaheadDistanceBaseVoxels) || (config.searchWidth != 0 && config.searchWidth < config.beamWidth) ||
        !(config.pruneDistanceBaseVoxels > 0.0) || !std::isfinite(config.pruneDistanceBaseVoxels) || !(config.errorThresholdBaseVoxels >= 0.0) ||
        !std::isfinite(config.errorThresholdBaseVoxels) || !(config.matchRefineSteps >= 0.0) || !std::isfinite(config.matchRefineSteps) ||
        !(config.minimumResetAdvanceBaseVoxels > 0.0) || !std::isfinite(config.minimumResetAdvanceBaseVoxels) ||
        !(config.referenceBeginArcBase >= 0.0) || !std::isfinite(config.referenceBeginArcBase) || !(normalWorkingToBaseScale > 0.0) ||
        !std::isfinite(normalWorkingToBaseScale) || (config.referenceEndArcBase.has_value() && !std::isfinite(*config.referenceEndArcBase))) {
        throw std::invalid_argument("fiberlet graph replay configuration is invalid");
    }
    const auto reference = makePolylineArcGeometry(referencePointsBaseXYZ);
    if (config.referenceEndArcBase.has_value() && *config.referenceEndArcBase > reference.length() + kReplayEpsilon) {
        throw std::invalid_argument("fiberlet graph replay reference end exceeds the reference");
    }
    const double referenceEndArcBase = config.referenceEndArcBase.has_value() ? *config.referenceEndArcBase : reference.length();
    if (config.referenceBeginArcBase >= referenceEndArcBase - kReplayEpsilon)
        throw std::invalid_argument("fiberlet graph replay has no usable reference interval");
    FiberletGraphReplayResult result;
    result.predictionToBaseScale = predictionToBaseScale;
    result.referenceBeginArcBase = config.referenceBeginArcBase;
    result.referenceEndArcBase = referenceEndArcBase;
    result.completedReferenceArcBase = config.referenceBeginArcBase;
    const double intervalLength = referenceEndArcBase - config.referenceBeginArcBase;
    const size_t maximumSegments = static_cast<size_t>(std::ceil(intervalLength / config.minimumResetAdvanceBaseVoxels)) + 2;
    const double seedWindowBase =
        std::max(config.minimumResetAdvanceBaseVoxels, static_cast<double>(graph.anchorCellSizePredictionVoxels()) * graph.predictionToBaseScale());
    const double seedBroadPhaseBase = fiberReplayTangentialThresholdBaseVoxels(config.errorThresholdBaseVoxels);
    std::set<FiberletStorageKey> consumedNodes;
    std::map<FiberletStorageKey, size_t> nodeIndices;
    std::map<FiberletStorageId, size_t> candidateIndices;
    std::map<DirectedFiberletStorageId, size_t> arcIndices;
    std::map<std::pair<DirectedFiberletStorageId, DirectedFiberletStorageId>, size_t> transitionIndices;
    const auto stableIndex = []<typename Key>(std::map<Key, size_t>& indices, const Key& key) {
        const auto [found, inserted] = indices.emplace(key, indices.size());
        return found->second;
    };

    struct Seed {
        FiberletReplaySourceAnchor node;
        PolylineArcProjection projection;
        FiberReplayThresholdMeasurement thresholdMeasurement;
    };
    const auto selectSeed = [&](double resetArc, std::optional<FiberletStorageKey> forcedKey) -> std::optional<Seed> {
        double scanBegin = resetArc;
        while (scanBegin < referenceEndArcBase - kReplayEpsilon) {
            const double scanEnd = std::min(referenceEndArcBase, scanBegin + seedWindowBase);
            std::optional<Seed> selected;
            for (const auto& node : graph.anchorsNearReference(reference, scanBegin, scanEnd, seedBroadPhaseBase)) {
                if (forcedKey.has_value() && node.id != *forcedKey)
                    continue;
                if (!forcedKey.has_value() && consumedNodes.contains(node.id))
                    continue;
                const auto projection = projectPointToPolylineArc(reference, cv::Vec3d(node.positionBaseXYZ), resetArc, referenceEndArcBase);
                if (projection.arc + kReplayEpsilon < scanBegin || projection.arc > scanEnd + kReplayEpsilon || projection.distance > seedBroadPhaseBase)
                    continue;
                const auto thresholdMeasurement =
                    measureFiberReplayThreshold(node.positionBaseXYZ, projection.point, normalSampler, normalWorkingToBaseScale, config.errorThresholdBaseVoxels);
                if (fiberReplayThresholdExceeded(thresholdMeasurement, config.errorThresholdBaseVoxels))
                    continue;
                const cv::Vec3d tangent = samplePolylineArc(reference, projection.arc).tangent;
                bool aligned = false;
                for (const auto& arcId : graph.outgoing(node.id)) {
                    if (angleDegrees(tangent, sourceArcStartDirection(graph.arc(arcId))) < graph.maximumJoinAngleDegrees()) {
                        aligned = true;
                        break;
                    }
                }
                if (!aligned)
                    continue;
                if (!selected.has_value() || std::tuple{projection.arc, thresholdMeasurement.thresholdErrorRatio, graph.logicalAnchorId(node.id)} <
                                                 std::tuple{
                                                     selected->projection.arc,
                                                     selected->thresholdMeasurement.thresholdErrorRatio,
                                                     graph.logicalAnchorId(selected->node.id)}) {
                    selected = Seed{node, projection, thresholdMeasurement};
                }
            }
            if (selected.has_value())
                return selected;
            if (forcedKey.has_value())
                break;
            if (scanEnd >= referenceEndArcBase - kReplayEpsilon)
                break;
            scanBegin = scanEnd;
        }
        return std::nullopt;
    };
    const auto appendFailure = [&](FiberReplayFailure event) {
        event.index = result.failures.size();
        event.referenceArcFraction = std::clamp((event.referenceArcBase - result.referenceBeginArcBase) / intervalLength, 0.0, 1.0);
        event.referencePointBase = samplePolylineArc(reference, event.referenceArcBase).point;
        result.failures.push_back(std::move(event));
        if (failureCallback)
            failureCallback(result.failures.back());
    };
    const auto emitProgress = [&](size_t segmentIndex,
                                  double arcBase,
                                  const char* state,
                                  std::optional<size_t> rolloutExpandedStateCount = {},
                                  std::optional<double> minimumAppliedLocalPruneLossCutoffPerPredictionVoxel = {}) {
        if (!progressCallback)
            return;
        progressCallback({
            segmentIndex,
            arcBase,
            std::clamp((arcBase - result.referenceBeginArcBase) / intervalLength, 0.0, 1.0),
            state,
            rolloutExpandedStateCount,
            minimumAppliedLocalPruneLossCutoffPerPredictionVoxel,
        });
    };

    double resetArc = config.referenceBeginArcBase;
    for (size_t iteration = 0; iteration < maximumSegments && resetArc < referenceEndArcBase - kReplayEpsilon; ++iteration) {
        emitProgress(result.segments.size(), resetArc, "segment_start");
        const auto seed = selectSeed(resetArc, iteration == 0 ? config.initialSeedKey : std::nullopt);
        if (iteration == 0 && config.initialSeedKey.has_value() && !seed.has_value()) {
            throw std::invalid_argument("fiberlet graph replay initial seed key is not usable in the focused interval");
        }
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
            emitProgress(result.segments.size() - 1, referenceEndArcBase, "completed");
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
        segment.seedKey = seed->node.id;
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
        const double maximumRouteLengthPredictionVoxels = (referenceEndArcBase - seed->projection.arc) / predictionToBaseScale;
        const double beamStepPredictionVoxels = config.beamStepDistanceBaseVoxels / predictionToBaseScale;
        const double lookaheadPredictionVoxels = config.lookaheadDistanceBaseVoxels / predictionToBaseScale;
        const double pruneDistancePredictionVoxels = config.pruneDistanceBaseVoxels / predictionToBaseScale;
        PersistentRouteCandidate initialBeam;
        initialBeam.seed = seed->node.id;
        initialBeam.visitedNodes = makeVisitedRoot(std::set<FiberletStorageKey>{seed->node.id});
        std::vector<PersistentRouteCandidate> beams{initialBeam};
        std::vector<FiberletGraphReplayDecision> decisions;
        PersistentRouteCandidate selectedRoute = initialBeam;
        std::optional<FiberReplayFailure> distanceFailure;
        bool referenceExhausted = maximumRouteLengthPredictionVoxels <= kReplayEpsilon;
        double previousReferenceArc = seed->projection.arc;
        std::set<FiberletStorageKey> selectedTraversedNodes{seed->node.id};
        const cv::Vec3d initialDirection = samplePolylineArc(reference, seed->projection.arc).tangent;
        const size_t maximumGeneratedStates = config.maximumGeneratedStatesPerIteration;

        const auto materializeSelected = [&](const PersistentRouteCandidate& route) {
            FiberletGraphReplaySegment built;
            std::set<FiberletStorageKey> traversedNodes{seed->node.id};
            built.seedKey = seed->node.id;
            built.startReferenceArcBase = seed->projection.arc;
            built.endReferenceArcBase = seed->projection.arc;
            built.routePointsBaseXYZ.emplace_back(seed->node.positionBaseXYZ);
            built.matches.push_back(
                {0, seed->projection.arc, seed->projection.arc, seed->projection.point, resetArc, seed->projection.arc, seed->thresholdMeasurement});
            double matchedArc = seed->projection.arc;
            std::optional<FiberReplayFailure> firstFailure;
            bool reachedEnd = false;
            const auto history = persistentRouteHistory(route);
            for (size_t arcOffset = 0; arcOffset < history.size(); ++arcOffset) {
                const auto* historyNode = history[arcOffset];
                const auto arcId = historyNode->arc;
                const auto edge = graph.arc(arcId);
                const auto fullPoints = graph.routePoints(arcId);
                const double remainingPathLength = std::max(0.0, maximumRouteLengthPredictionVoxels - built.pathLengthPredictionVoxels);
                const double includedArcFraction = std::clamp(remainingPathLength / static_cast<double>(edge.pathLengthPredictionVoxels), 0.0, 1.0);
                const auto points = routePointPrefix(fullPoints, includedArcFraction);
                if (points.size() < 2)
                    throw std::logic_error("persistent beam route is too short");
                double fullGeometryLengthBase = 0.0;
                for (size_t pointIndex = 1; pointIndex < fullPoints.size(); ++pointIndex) {
                    fullGeometryLengthBase += length(fullPoints[pointIndex] - fullPoints[pointIndex - 1]);
                }
                if (!(fullGeometryLengthBase > kReplayEpsilon))
                    throw std::logic_error("persistent beam route has zero geometry length");
                const size_t candidateIndex = edge.diagnosticCandidateIndex.value_or(stableIndex(candidateIndices, arcId.fiberlet));
                const size_t arcIndex = edge.diagnosticArcIndex.value_or(stableIndex(arcIndices, arcId));
                built.candidateIndices.push_back(candidateIndex);
                built.arcIndices.push_back(arcIndex);
                FiberletGraphReplayCommittedStep step;
                step.referenceBeginArcBase = matchedArc;
                double traversedGeometryLengthBase = 0.0;
                if (historyNode->enteringTransition.has_value()) {
                    const auto& join = *historyNode->enteringTransition;
                    step.transitionCost += join.cost;
                    built.transitionCost += join.cost;
                    built.transitionIndices.push_back(
                        join.diagnosticTransitionIndex.value_or(stableIndex(transitionIndices, std::pair{join.incoming, join.outgoing})));
                }
                for (size_t pointIndex = 1; pointIndex < points.size(); ++pointIndex) {
                    const double stepBase = length(points[pointIndex] - built.routePointsBaseXYZ.back());
                    traversedGeometryLengthBase += stepBase;
                    const auto forwardMatch =
                        matchForwardPolylinePoint(reference, points[pointIndex], matchedArc, stepBase, config.matchRefineSteps, referenceEndArcBase);
                    const auto thresholdMeasurement = measureFiberReplayThreshold(
                        points[pointIndex], forwardMatch.projection.point, normalSampler, normalWorkingToBaseScale, config.errorThresholdBaseVoxels);
                    built.routePointsBaseXYZ.push_back(points[pointIndex]);
                    built.matches.push_back({
                        built.routePointsBaseXYZ.size() - 1,
                        forwardMatch.predictedArc,
                        forwardMatch.projection.arc,
                        forwardMatch.projection.point,
                        matchedArc,
                        forwardMatch.searchEndArc,
                        thresholdMeasurement,
                    });
                    matchedArc = forwardMatch.projection.arc;
                    built.endReferenceArcBase = matchedArc;
                    if (!firstFailure.has_value() && fiberReplayThresholdExceeded(thresholdMeasurement, config.errorThresholdBaseVoxels)) {
                        FiberReplayFailure event;
                        event.segmentIndex = result.segments.size();
                        event.reason = "distance_above_threshold";
                        event.referenceArcBase = matchedArc;
                        event.evaluatorPointBase = points[pointIndex];
                        event.segmentPointIndex = built.routePointsBaseXYZ.size() - 1;
                        event.candidateIndex = candidateIndex;
                        event.arcIndex = arcIndex;
                        event.candidatePathPointIndex = pointIndex;
                        event.thresholdMeasurement = thresholdMeasurement;
                        firstFailure = std::move(event);
                    }
                    if (matchedArc >= referenceEndArcBase - kReplayEpsilon) {
                        reachedEnd = true;
                        break;
                    }
                }
                const double traversedFraction = std::clamp(traversedGeometryLengthBase / fullGeometryLengthBase, 0.0, includedArcFraction);
                addScaledCost(step.edgeCost, edge.cost, traversedFraction);
                step.pathLengthPredictionVoxels = traversedFraction * edge.pathLengthPredictionVoxels;
                built.edgeCost.invalidPrediction += step.edgeCost.invalidPrediction;
                built.edgeCost.alignment += step.edgeCost.alignment;
                built.edgeCost.isotropicSmoothness += step.edgeCost.isotropicSmoothness;
                built.edgeCost.tangentSmoothness += step.edgeCost.tangentSmoothness;
                built.edgeCost.normalSmoothness += step.edgeCost.normalSmoothness;
                built.pathLengthPredictionVoxels += step.pathLengthPredictionVoxels;
                if (traversedFraction >= 1.0 - kReplayEpsilon)
                    traversedNodes.insert(edge.target);
                step.referenceEndArcBase = matchedArc;
                built.committedSteps.push_back(std::move(step));
                if (firstFailure.has_value() || reachedEnd)
                    break;
                if (includedArcFraction < 1.0 - kReplayEpsilon) {
                    reachedEnd = true;
                    break;
                }
            }
            built.totalLoss = built.edgeCost.total() + built.transitionCost.total();
            built.terminalPartialEdge = !built.arcIndices.empty() && traversedNodes.size() <= built.arcIndices.size();
            if (!built.terminalPartialEdge && !built.arcIndices.empty())
                built.stopNodeIndex = stableIndex(nodeIndices, targetAnchor(history[built.arcIndices.size() - 1]->arc));
            return std::tuple{std::move(built), std::move(firstFailure), reachedEnd, std::move(traversedNodes)};
        };

        double checkpointPredictionVoxels = 0.0;
        for (size_t beamIteration = 0; !distanceFailure.has_value() && !referenceExhausted; ++beamIteration) {
            const double currentCheckpoint = checkpointPredictionVoxels;
            const double scoringHorizon = std::min(maximumRouteLengthPredictionVoxels, currentCheckpoint + lookaheadPredictionVoxels);
            const double nextCheckpoint = std::min(maximumRouteLengthPredictionVoxels, currentCheckpoint + beamStepPredictionVoxels);
            ExactPersistentSearchStats searchStats;
            std::vector<RankedPersistentPrefix> ranked;
            std::vector<FiberletGraphReplayPruneFront> pruneFronts;
            std::vector<DirectedFiberletStorageId> selectedPrefixLogicalArcs;
            if (config.searchWidth == 0) {
                ranked =
                    exactPersistentRouteLookahead(graph, beams, scoringHorizon, nextCheckpoint, initialDirection, config.beamWidth, config.expansionThreads, maximumGeneratedStates, searchStats);
                if (!ranked.empty()) {
                    selectedPrefixLogicalArcs = ranked.front().prefixLogicalArcs;
                }
            } else {
                auto bounded = boundedPersistentRouteLookahead(
                    graph,
                    beams,
                    scoringHorizon,
                    currentCheckpoint,
                    nextCheckpoint,
                    pruneDistancePredictionVoxels,
                    initialDirection,
                    config.beamWidth,
                    config.searchWidth,
                    config.expansionThreads,
                    maximumGeneratedStates,
                    searchStats);
                ranked = std::move(bounded.ranked);
                pruneFronts = std::move(bounded.fronts);
                selectedPrefixLogicalArcs = std::move(bounded.selectedPrefixLogicalArcs);
            }
            if (ranked.empty())
                break;
            const std::optional<size_t> rolloutExpandedStateCount =
                config.searchWidth == 0 ? std::nullopt : std::optional<size_t>{searchStats.expanded};
            const std::optional<double> minimumAppliedLocalPruneLossCutoffPerPredictionVoxel =
                config.searchWidth == 0 || pruneFronts.empty() ? std::nullopt : pruneFronts.back().minimumAppliedLocalCompletionLossCutoffPerPredictionVoxel;
            beams.clear();
            beams.reserve(ranked.size());
            for (const auto& entry : ranked) {
                auto prefix = entry.prefix;
                prefix.visitedNodes = compactPersistentVisited(prefix.visitedNodes);
                beams.push_back(std::move(prefix));
            }
            checkpointPredictionVoxels = nextCheckpoint;
            if (std::any_of(beams.begin(), beams.end(), [&](const auto& beam) {
                    return beam.pathLength + kReplayEpsilon < checkpointPredictionVoxels;
                })) {
                throw std::logic_error("persistent beam checkpoint exceeds a committed route");
            }
            selectedRoute = beams.front();

            auto [provisional, provisionalFailure, reachedEnd, provisionalTraversedNodes] = materializeSelected(selectedRoute);
            previousReferenceArc = provisional.endReferenceArcBase;
            if (config.recordDecisionDiagnostics) {
                FiberletGraphReplayDecision decision;
                decision.routePointIndex = provisional.routePointsBaseXYZ.empty() ? 0 : provisional.routePointsBaseXYZ.size() - 1;
                decision.referenceArcBase = previousReferenceArc;
                decision.checkpointPathLengthPredictionVoxels = currentCheckpoint;
                decision.nextCheckpointPathLengthPredictionVoxels = checkpointPredictionVoxels;
                decision.scoringHorizonPathLengthPredictionVoxels = scoringHorizon;
                decision.generatedStateCount = searchStats.generated;
                decision.expandedStateCount = searchStats.expanded;
                decision.evaluatedCandidateCount = searchStats.completed;
                decision.costPrunedStateCount = searchStats.costPruned;
                decision.rejectedStateCount = searchStats.rejected;
                decision.dominatedStateCount = searchStats.dominated;
                decision.retainedBeamCount = ranked.size();
                decision.searchMode = config.searchWidth == 0 ? "exact_cost_bounded" : "intermediate_pruned";
                decision.searchWidth = config.searchWidth;
                decision.pruneDistancePredictionVoxels = pruneDistancePredictionVoxels;
                decision.pruneFronts = std::move(pruneFronts);
                decision.selectedPrefixLogicalArcs = std::move(selectedPrefixLogicalArcs);
                decision.sourceKey = graph.logicalAnchorId(seed->node.id);
                decision.selectedRouteIndex = 0;
                for (const auto& entry : ranked) {
                    FiberletGraphReplayDecisionRoute route;
                    route.prefixLogicalArcs = entry.prefixLogicalArcs;
                    route.logicalArcs = entry.lookaheadLogicalArcs;
                    route.routePointsBaseXYZ = persistentRoutePoints(graph, entry.lookahead, cv::Vec3d(seed->node.positionBaseXYZ));
                    route.edgeCost = entry.scoredEdgeCost;
                    route.transitionCost = entry.scoredTransitionCost;
                    route.committedEdgeCost = entry.prefix.edgeCost;
                    route.committedTransitionCost = entry.prefix.transitionCost;
                    route.committedPathLengthPredictionVoxels = entry.prefix.pathLength;
                    route.pathLengthPredictionVoxels = entry.scoredPathLength;
                    route.completePathLengthPredictionVoxels = entry.completePathLength;
                    route.totalLoss = entry.totalLoss;
                    route.lossPerPredictionVoxel = entry.lossPerPredictionVoxel;
                    decision.routes.push_back(std::move(route));
                }
                decisions.push_back(std::move(decision));
            }
            if (provisionalFailure.has_value()) {
                segment = std::move(provisional);
                selectedTraversedNodes = std::move(provisionalTraversedNodes);
                distanceFailure = std::move(provisionalFailure);
            } else if (reachedEnd || checkpointPredictionVoxels >= maximumRouteLengthPredictionVoxels - kReplayEpsilon) {
                segment = std::move(provisional);
                selectedTraversedNodes = std::move(provisionalTraversedNodes);
                referenceExhausted = true;
            }
            emitProgress(result.segments.size(), previousReferenceArc, "running", rolloutExpandedStateCount, minimumAppliedLocalPruneLossCutoffPerPredictionVoxel);
        }
        if (segment.routePointsBaseXYZ.size() <= 1) {
            auto materialized = materializeSelected(selectedRoute);
            segment = std::move(std::get<0>(materialized));
            selectedTraversedNodes = std::move(std::get<3>(materialized));
            if (!distanceFailure.has_value())
                distanceFailure = std::move(std::get<1>(materialized));
        }
        segment.decisions = std::move(decisions);
        consumedNodes.insert(selectedTraversedNodes.begin(), selectedTraversedNodes.end());

        if (referenceExhausted && !distanceFailure.has_value()) {
            segment.endReferenceArcBase = referenceEndArcBase;
            segment.terminationReason = "reference_end";
            result.segments.push_back(std::move(segment));
            result.completedReferenceArcBase = referenceEndArcBase;
            emitProgress(result.segments.size() - 1, referenceEndArcBase, "completed");
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
                    event.thresholdMeasurement = segment.matches.back().thresholdMeasurement;
                }
            }
            distanceFailure = std::move(event);
        }
        result.segments.push_back(std::move(segment));
        appendFailure(std::move(*distanceFailure));
        resetArc = std::min(referenceEndArcBase, std::max(failureArc, result.segments.back().startReferenceArcBase + config.minimumResetAdvanceBaseVoxels));
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
    const auto storageKeyJson = [](const FiberletStorageKey& key) {
        return nlohmann::json::array({key.coordinateZYX[0], key.coordinateZYX[1], key.coordinateZYX[2], key.variant});
    };
    const auto arcIdJson = [&](const DirectedFiberletStorageId& arc) {
        return nlohmann::json::array({storageKeyJson(arc.fiberlet.first), storageKeyJson(arc.fiberlet.second), arc.reverse});
    };
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
             {"expansion_threads", config.expansionThreads},
             {"lookahead_mode", config.searchWidth == 0 ? "exact_cost_bounded" : "intermediate_pruned"},
             {"search_width", config.searchWidth},
             {"prune_distance_base_voxels", config.pruneDistanceBaseVoxels},
             {"prune_distance_prediction_voxels", config.pruneDistanceBaseVoxels / replay.predictionToBaseScale},
             {"beam_step_distance_base_voxels", config.beamStepDistanceBaseVoxels},
             {"beam_step_distance_prediction_voxels", config.beamStepDistanceBaseVoxels / replay.predictionToBaseScale},
             {"lookahead_distance_base_voxels", config.lookaheadDistanceBaseVoxels},
             {"lookahead_distance_prediction_voxels", config.lookaheadDistanceBaseVoxels / replay.predictionToBaseScale},
             {"maximum_generated_states_per_iteration", config.maximumGeneratedStatesPerIteration},
             {"threshold", fiberReplayThresholdDescriptorJson(config.errorThresholdBaseVoxels)},
             {"match_refine_steps", config.matchRefineSteps},
             {"minimum_reset_advance_base_voxels", config.minimumResetAdvanceBaseVoxels},
             {"reference_begin_arc_base", config.referenceBeginArcBase},
             {"reference_end_arc_base", replay.referenceEndArcBase},
             {"initial_seed_key", config.initialSeedKey.has_value() ? storageKeyJson(*config.initialSeedKey) : nlohmann::json(nullptr)},
             {"record_decision_diagnostics", config.recordDecisionDiagnostics},
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
            validateFiberReplayThresholdMeasurement(match.thresholdMeasurement, config.errorThresholdBaseVoxels);
            auto matchJson = fiberReplayThresholdMeasurementJson(match.thresholdMeasurement);
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
        nlohmann::json decisions = nlohmann::json::array();
        for (const auto& decision : segment.decisions) {
            nlohmann::json selectedPrefixLogicalArcs = nlohmann::json::array();
            for (const auto& arc : decision.selectedPrefixLogicalArcs)
                selectedPrefixLogicalArcs.push_back(arcIdJson(arc));
            nlohmann::json pruneFronts = nlohmann::json::array();
            for (const auto& front : decision.pruneFronts) {
                nlohmann::json frontJson = {
                    {"horizon_path_length_prediction_voxels", front.horizonPathLengthPredictionVoxels},
                    {"input_route_count", front.inputRouteCount},
                    {"local_candidate_limit", front.localCandidateLimit},
                    {"generated_state_count", front.generatedStateCount},
                    {"expanded_state_count", front.expandedStateCount},
                    {"rejected_state_count", front.rejectedStateCount},
                    {"dominated_state_count", front.dominatedStateCount},
                    {"cost_pruned_state_count", front.costPrunedStateCount},
                    {"completed_candidate_count", front.completedCandidateCount},
                    {"distinct_prefix_count", front.distinctPrefixCount},
                    {"diversity_protected_count", front.diversityProtectedCount},
                    {"global_fill_count", front.globalFillCount},
                    {"retained_route_count", front.retainedRouteCount},
                    {"pruned_candidate_count", front.prunedCandidateCount},
                    {"cumulative_generated_state_count", front.cumulativeGeneratedStateCount},
                    {"search_width_bound", front.searchWidthBound},
                };
                if (front.minimumAppliedLocalCompletionLossCutoffPerPredictionVoxel.has_value()) {
                    frontJson["minimum_applied_local_completion_loss_cutoff_per_prediction_voxel"] =
                        *front.minimumAppliedLocalCompletionLossCutoffPerPredictionVoxel;
                }
                pruneFronts.push_back(std::move(frontJson));
            }
            nlohmann::json routes = nlohmann::json::array();
            for (const auto& route : decision.routes) {
                nlohmann::json prefixLogicalArcs = nlohmann::json::array();
                for (const auto& arc : route.prefixLogicalArcs)
                    prefixLogicalArcs.push_back(arcIdJson(arc));
                nlohmann::json logicalArcs = nlohmann::json::array();
                for (const auto& arc : route.logicalArcs)
                    logicalArcs.push_back(arcIdJson(arc));
                nlohmann::json routePoints = nlohmann::json::array();
                for (const auto& point : route.routePointsBaseXYZ)
                    routePoints.push_back(pointJson(point));
                routes.push_back({
                    {"prefix_logical_arcs", std::move(prefixLogicalArcs)},
                    {"logical_arcs", std::move(logicalArcs)},
                    {"route_points_base_xyz", std::move(routePoints)},
                    {"edge_cost", costJson(route.edgeCost)},
                    {"transition_cost", costJson(route.transitionCost)},
                    {"committed_edge_cost", costJson(route.committedEdgeCost)},
                    {"committed_transition_cost", costJson(route.committedTransitionCost)},
                    {"committed_path_length_prediction_voxels", route.committedPathLengthPredictionVoxels},
                    {"path_length_prediction_voxels", route.pathLengthPredictionVoxels},
                    {"complete_path_length_prediction_voxels", route.completePathLengthPredictionVoxels},
                    {"total_loss", route.totalLoss},
                    {"loss_per_prediction_voxel", route.lossPerPredictionVoxel},
                });
            }
            decisions.push_back({
                {"route_point_index", decision.routePointIndex},
                {"reference_arc_base", decision.referenceArcBase},
                {"checkpoint_path_length_prediction_voxels", decision.checkpointPathLengthPredictionVoxels},
                {"next_checkpoint_path_length_prediction_voxels", decision.nextCheckpointPathLengthPredictionVoxels},
                {"scoring_horizon_path_length_prediction_voxels", decision.scoringHorizonPathLengthPredictionVoxels},
                {"generated_state_count", decision.generatedStateCount},
                {"expanded_state_count", decision.expandedStateCount},
                {"evaluated_candidate_count", decision.evaluatedCandidateCount},
                {"cost_pruned_state_count", decision.costPrunedStateCount},
                {"rejected_state_count", decision.rejectedStateCount},
                {"dominated_state_count", decision.dominatedStateCount},
                {"retained_beam_count", decision.retainedBeamCount},
                {"search_mode", decision.searchMode},
                {"search_width", decision.searchWidth},
                {"prune_distance_prediction_voxels", decision.pruneDistancePredictionVoxels},
                {"prune_fronts", std::move(pruneFronts)},
                {"selected_prefix_logical_arcs", std::move(selectedPrefixLogicalArcs)},
                {"source_key", storageKeyJson(decision.sourceKey)},
                {"incoming_logical_arc", decision.incomingLogicalArc.has_value() ? arcIdJson(*decision.incomingLogicalArc) : nlohmann::json(nullptr)},
                {"selected_route_index", decision.selectedRouteIndex.has_value() ? nlohmann::json(*decision.selectedRouteIndex) : nlohmann::json(nullptr)},
                {"routes", std::move(routes)},
            });
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
            {"seed_key", segment.seedKey.has_value() ? storageKeyJson(*segment.seedKey) : nlohmann::json(nullptr)},
            {"terminal_partial_edge", segment.terminalPartialEdge},
            {"matches", std::move(matches)},
            {"decisions", std::move(decisions)},
            {"total_loss", segment.totalLoss},
            {"edge_cost", costJson(segment.edgeCost)},
            {"transition_cost", costJson(segment.transitionCost)},
            {"path_length_prediction_voxels", segment.pathLengthPredictionVoxels},
            {"loss_per_prediction_voxel",
             segment.pathLengthPredictionVoxels > kReplayEpsilon ? nlohmann::json(segment.totalLoss / segment.pathLengthPredictionVoxels)
                                                                 : nlohmann::json(nullptr)},
        });
    }
    for (const auto& failure : replay.failures) {
        auto failureJson = fiberReplayOptionalThresholdMeasurementJson(failure.thresholdMeasurement, config.errorThresholdBaseVoxels);
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

std::vector<FiberletGraphReplayFailureWindow> fiberletGraphReplayFailureWindows(const FiberletGraphReplayResult& replay)
{
    std::vector<FiberletGraphReplayFailureWindow> result;
    result.reserve(replay.failures.size());
    for (const auto& failure : replay.failures) {
        if (failure.index != result.size())
            throw std::invalid_argument("fiberlet replay failures are not in stable index order");
        if (failure.segmentIndex >= replay.segments.size())
            throw std::invalid_argument("fiberlet replay failure segment index is out of range");
        const auto& segment = replay.segments[failure.segmentIndex];
        const double begin = !segment.matches.empty() ? segment.matches.front().searchBeginArcBase : segment.startReferenceArcBase;
        const double end = std::max(failure.referenceArcBase, segment.endReferenceArcBase);
        if (!std::isfinite(failure.referenceArcBase) || !std::isfinite(begin) || !std::isfinite(end) ||
            begin < replay.referenceBeginArcBase - kReplayEpsilon || end > replay.referenceEndArcBase + kReplayEpsilon ||
            failure.referenceArcBase < begin - kReplayEpsilon || end < failure.referenceArcBase - kReplayEpsilon || !(end > begin + kReplayEpsilon)) {
            throw std::invalid_argument("fiberlet replay failure window is inconsistent");
        }
        result.push_back({
            failure.index,
            failure.segmentIndex,
            failure.reason,
            failure.referenceArcBase,
            begin,
            end,
            segment.seedKey,
        });
    }
    return result;
}

}  // namespace vc::fiber_tracer
