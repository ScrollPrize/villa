#include "vc/fiber_tracer/FiberGraph.hpp"

#include "vc/fiber_tracer/FiberLocalScoring.hpp"
#include "vc/fiber_tracer/PolylineGeometry.hpp"

#include <utils/thread_pool.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <exception>
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

const char* replayCostModeName(FiberletGraphReplayCostMode mode)
{
    switch (mode) {
    case FiberletGraphReplayCostMode::Fiberlet:
        return "fiberlet";
    case FiberletGraphReplayCostMode::Stepped:
        return "stepped";
    }
    throw std::invalid_argument("fiberlet replay cost mode is invalid");
}

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

constexpr size_t kFiberletStorageKeyBits = 3 * 64 + 8;

bool storageKeyBit(const FiberletStorageKey& key, size_t bit)
{
    if (bit >= kFiberletStorageKeyBits)
        throw std::out_of_range("fiberlet storage key bit is out of range");
    if (bit < 3 * 64) {
        const size_t axis = bit / 64;
        const size_t shift = 63 - bit % 64;
        return (static_cast<std::uint64_t>(key.coordinateZYX[axis]) >> shift) & 1U;
    }
    return (static_cast<std::uint8_t>(key.variant) >> (7 - (bit - 3 * 64))) & 1U;
}

size_t firstDifferentStorageKeyBit(const FiberletStorageKey& left, const FiberletStorageKey& right)
{
    for (size_t bit = 0; bit < kFiberletStorageKeyBits; ++bit) {
        if (storageKeyBit(left, bit) != storageKeyBit(right, bit))
            return bit;
    }
    return kFiberletStorageKeyBits;
}

struct PersistentVisitedNodes {
    bool leaf = false;
    std::uint16_t branchBit = 0;
    FiberletStorageKey key;
    std::shared_ptr<const PersistentVisitedNodes> zero;
    std::shared_ptr<const PersistentVisitedNodes> one;
};

std::shared_ptr<const PersistentVisitedNodes> makeVisitedLeaf(FiberletStorageKey key)
{
    return std::make_shared<const PersistentVisitedNodes>(PersistentVisitedNodes{true, 0, std::move(key), nullptr, nullptr});
}

bool persistentVisitedContains(const std::shared_ptr<const PersistentVisitedNodes>& visited, const FiberletStorageKey& key)
{
    auto node = visited;
    while (node != nullptr && !node->leaf)
        node = storageKeyBit(key, node->branchBit) ? node->one : node->zero;
    return node != nullptr && node->key == key;
}

std::shared_ptr<const PersistentVisitedNodes> persistentVisitedAdd(const std::shared_ptr<const PersistentVisitedNodes>& visited, FiberletStorageKey key)
{
    if (visited == nullptr)
        return makeVisitedLeaf(std::move(key));
    auto leaf = visited;
    while (!leaf->leaf)
        leaf = storageKeyBit(key, leaf->branchBit) ? leaf->one : leaf->zero;
    if (leaf->key == key)
        return visited;
    const size_t differingBit = firstDifferentStorageKeyBit(key, leaf->key);
    if (differingBit >= kFiberletStorageKeyBits)
        throw std::logic_error("distinct fiberlet storage keys have identical bits");
    const auto inserted = makeVisitedLeaf(std::move(key));
    const auto rebuild = [&](const auto& self, const std::shared_ptr<const PersistentVisitedNodes>& node) -> std::shared_ptr<const PersistentVisitedNodes> {
        if (node->leaf || node->branchBit >= differingBit) {
            if (storageKeyBit(inserted->key, differingBit))
                return std::make_shared<const PersistentVisitedNodes>(
                    PersistentVisitedNodes{false, static_cast<std::uint16_t>(differingBit), {}, node, inserted});
            return std::make_shared<const PersistentVisitedNodes>(
                PersistentVisitedNodes{false, static_cast<std::uint16_t>(differingBit), {}, inserted, node});
        }
        if (storageKeyBit(inserted->key, node->branchBit)) {
            return std::make_shared<const PersistentVisitedNodes>(PersistentVisitedNodes{false, node->branchBit, {}, node->zero, self(self, node->one)});
        }
        return std::make_shared<const PersistentVisitedNodes>(PersistentVisitedNodes{false, node->branchBit, {}, self(self, node->zero), node->one});
    };
    return rebuild(rebuild, visited);
}

constexpr size_t kLogicalRouteAncestorLevels = std::numeric_limits<size_t>::digits;

struct PersistentLogicalRouteNode {
    std::shared_ptr<const PersistentLogicalRouteNode> parent;
    DirectedFiberletStorageId arc;
    size_t depth = 0;
    std::array<const PersistentLogicalRouteNode*, kLogicalRouteAncestorLevels> ancestors{};
};

struct PersistentLogicalRouteInternKey {
    const PersistentLogicalRouteNode* parent = nullptr;
    DirectedFiberletStorageId arc;
};

struct PersistentLogicalRouteInternKeyLess {
    bool operator()(const PersistentLogicalRouteInternKey& left, const PersistentLogicalRouteInternKey& right) const
    {
        if (left.parent != right.parent)
            return std::less<const PersistentLogicalRouteNode*>{}(left.parent, right.parent);
        return left.arc < right.arc;
    }
};

class PersistentLogicalRouteRegistry
{
public:
    PersistentLogicalRouteRegistry() : root_(std::make_shared<const PersistentLogicalRouteNode>()) {}

    [[nodiscard]] const std::shared_ptr<const PersistentLogicalRouteNode>& root() const noexcept { return root_; }

    [[nodiscard]] std::shared_ptr<const PersistentLogicalRouteNode> extend(const std::shared_ptr<const PersistentLogicalRouteNode>& parent, const DirectedFiberletStorageId& arc)
    {
        if (!parent)
            throw std::invalid_argument("persistent logical route parent is missing");
        const PersistentLogicalRouteInternKey key{parent.get(), arc};
        auto& shard = shards_[shardIndex(key)];
        std::lock_guard lock(shard.mutex);
        if (const auto found = shard.nodes.find(key); found != shard.nodes.end()) {
            if (auto existing = found->second.lock())
                return existing;
            auto node = makeNode(parent, arc);
            found->second = node;
            return node;
        }
        auto node = makeNode(parent, arc);
        shard.nodes.emplace(key, node);
        return node;
    }

    size_t pruneExpired(size_t maximumVisited = 4096)
    {
        size_t visited = 0;
        size_t completedShards = 0;
        while (visited < maximumVisited && completedShards < kShardCount) {
            auto& shard = shards_[pruneShard_];
            std::lock_guard lock(shard.mutex);
            auto current = shard.pruneCursor.has_value()
                ? *shard.pruneCursor
                : shard.nodes.begin();
            while (current != shard.nodes.end() &&
                   visited < maximumVisited) {
                ++visited;
                if (current->second.expired())
                    current = shard.nodes.erase(current);
                else
                    ++current;
            }
            if (current == shard.nodes.end()) {
                shard.pruneCursor.reset();
                pruneShard_ = (pruneShard_ + 1) % kShardCount;
                ++completedShards;
            } else {
                shard.pruneCursor = current;
            }
        }
        return visited;
    }

    [[nodiscard]] size_t internedCount()
    {
        size_t result = 0;
        for (auto& shard : shards_) {
            std::lock_guard lock(shard.mutex);
            result += shard.nodes.size();
        }
        return result;
    }

private:
    static std::shared_ptr<const PersistentLogicalRouteNode> makeNode(
        const std::shared_ptr<const PersistentLogicalRouteNode>& parent,
        const DirectedFiberletStorageId& arc)
    {
        auto node = std::make_shared<PersistentLogicalRouteNode>();
        node->parent = parent;
        node->arc = arc;
        node->depth = parent->depth + 1;
        node->ancestors[0] = parent.get();
        for (size_t level = 1; level < node->ancestors.size(); ++level) {
            const auto* previous = node->ancestors[level - 1];
            node->ancestors[level] = previous == nullptr
                ? nullptr
                : previous->ancestors[level - 1];
        }
        return node;
    }

    using NodeMap = std::map<
        PersistentLogicalRouteInternKey,
        std::weak_ptr<const PersistentLogicalRouteNode>,
        PersistentLogicalRouteInternKeyLess>;

    struct Shard {
        std::mutex mutex;
        NodeMap nodes;
        std::optional<NodeMap::iterator> pruneCursor;
    };

    static size_t shardIndex(const PersistentLogicalRouteInternKey& key)
    {
        size_t value = reinterpret_cast<std::uintptr_t>(key.parent) >> 4;
        for (const auto coordinate : key.arc.fiberlet.first.coordinateZYX)
            value ^= static_cast<size_t>(coordinate) + 0x9e3779b9U + (value << 6) + (value >> 2);
        value ^= static_cast<size_t>(key.arc.fiberlet.first.variant) + (value << 6) + (value >> 2);
        return value % kShardCount;
    }

    static constexpr size_t kShardCount = 32;
    std::shared_ptr<const PersistentLogicalRouteNode> root_;
    std::array<Shard, kShardCount> shards_;
    size_t pruneShard_ = 0;
};

const PersistentLogicalRouteNode* logicalRouteAncestorAtDepth(const PersistentLogicalRouteNode* node, size_t depth)
{
    if (node == nullptr || depth > node->depth)
        throw std::invalid_argument("persistent logical route ancestor depth is invalid");
    size_t distance = node->depth - depth;
    for (size_t level = 0; distance != 0; ++level) {
        if (distance & 1U)
            node = node->ancestors[level];
        distance >>= 1;
    }
    return node;
}

const PersistentLogicalRouteNode* logicalRouteLowestCommonAncestor(const PersistentLogicalRouteNode* left, const PersistentLogicalRouteNode* right)
{
    if (left == nullptr || right == nullptr)
        throw std::invalid_argument("persistent logical route is missing");
    const size_t commonDepth = std::min(left->depth, right->depth);
    left = logicalRouteAncestorAtDepth(left, commonDepth);
    right = logicalRouteAncestorAtDepth(right, commonDepth);
    if (left == right)
        return left;
    for (size_t level = kLogicalRouteAncestorLevels; level-- > 0;) {
        if (left->ancestors[level] != right->ancestors[level]) {
            left = left->ancestors[level];
            right = right->ancestors[level];
        }
    }
    return left->parent.get();
}

bool logicalRouteLess(const PersistentLogicalRouteNode* left, const PersistentLogicalRouteNode* right)
{
    if (left == right)
        return false;
    const auto* originalLeft = left;
    const auto* originalRight = right;
    const auto* common = logicalRouteLowestCommonAncestor(left, right);
    if (common == originalLeft)
        return true;
    if (common == originalRight)
        return false;
    const auto* leftChild = logicalRouteAncestorAtDepth(originalLeft, common->depth + 1);
    const auto* rightChild = logicalRouteAncestorAtDepth(originalRight, common->depth + 1);
    return leftChild->arc < rightChild->arc;
}

std::vector<DirectedFiberletStorageId> logicalRouteArcs(const PersistentLogicalRouteNode* route)
{
    if (route == nullptr)
        throw std::invalid_argument("persistent logical route is missing");
    std::vector<DirectedFiberletStorageId> result(route->depth);
    for (size_t offset = route->depth; offset > 0; --offset) {
        result[offset - 1] = route->arc;
        route = route->parent.get();
    }
    return result;
}

struct PersistentRouteHistory {
    std::shared_ptr<const PersistentRouteHistory> parent;
    DirectedFiberletStorageId arc;
    std::shared_ptr<const PersistentLogicalRouteNode> logicalRoute;
    std::optional<FiberletReplaySourceTransition> enteringTransition;
    std::shared_ptr<const PersistentVisitedNodes> visitedNodes;
    FiberletGraphReplayCost cumulativeEdgeCost;
    FiberletGraphReplayCost cumulativeTransitionCost;
    double cumulativePathLength = 0.0;
    size_t arcCount = 0;
};

struct PersistentRouteEvaluation {
    std::shared_ptr<const PersistentRouteEvaluation> parent;
    std::shared_ptr<const PersistentRouteHistory> history;
    std::vector<cv::Vec3d> appendedPointsBaseXYZ;
    std::vector<FiberletGraphReplayMatch> appendedMatches;
    std::optional<FiberletGraphReplayCommittedStep> committedStep;
    FiberletGraphReplayCost cumulativeEdgeCost;
    FiberletGraphReplayCost cumulativeTransitionCost;
    double cumulativePathLengthPredictionVoxels = 0.0;
    double matchedReferenceArcBase = 0.0;
    size_t routePointCount = 0;
    cv::Vec3d lastPointBaseXYZ{0.0, 0.0, 0.0};
    std::optional<FiberReplayFailure> failure;
    std::optional<size_t> failureCandidatePathPointIndex;
    bool reachedReferenceEnd = false;
    bool traversedFullEdge = false;
};

struct PersistentRouteCandidate {
    FiberletStorageKey seed;
    std::shared_ptr<const PersistentRouteHistory> tail;
    std::shared_ptr<const PersistentVisitedNodes> visitedNodes;
    std::shared_ptr<const PersistentLogicalRouteNode> logicalRoute;
    std::shared_ptr<const PersistentRouteEvaluation> evaluation;
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
    return logicalRouteArcs(route.logicalRoute.get());
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

void validateReplayCost(
    const FiberletGraphReplayCost& cost,
    const char* description)
{
    const std::array values{
        cost.invalidPrediction,
        cost.alignment,
        cost.isotropicSmoothness,
        cost.tangentSmoothness,
        cost.normalSmoothness,
    };
    if (std::any_of(values.begin(), values.end(), [](double value) {
            return !(value >= 0.0) || !std::isfinite(value);
        })) {
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

template <typename AccountGenerated>
std::vector<PersistentRouteCandidate> persistentRouteSuccessors(
    const FiberletReplayGraphSource& graph,
    PersistentLogicalRouteRegistry& logicalRoutes,
    const PersistentRouteCandidate& route,
    const std::optional<cv::Vec3d>& initialDirection,
    AccountGenerated&& accountGenerated,
    size_t& rejectedStates)
{
    std::vector<PersistentRouteCandidate> result;
    const auto anchor = persistentRouteAnchor(route);
    const auto incomingId = persistentRouteIncoming(route);
    const auto incoming = incomingId.has_value() ? std::make_optional(graph.arc(*incomingId)) : std::nullopt;
    const auto outgoing = graph.outgoingArcs(anchor);
    result.reserve(outgoing.size());
    for (size_t outgoingIndex = 0; outgoingIndex < outgoing.size();
         ++outgoingIndex) {
        const auto& edge = outgoing[outgoingIndex];
        const auto& outgoingId = edge.id;
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
        accountGenerated();
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
        const auto logicalArc = graph.logicalArcId(outgoingId);
        next.logicalRoute = logicalRoutes.extend(route.logicalRoute, logicalArc);
        next.tail = std::make_shared<PersistentRouteHistory>(PersistentRouteHistory{
            route.tail,
            outgoingId,
            next.logicalRoute,
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

std::vector<PersistentRouteCandidate> persistentRouteSuccessors(
    const FiberletReplayGraphSource& graph,
    PersistentLogicalRouteRegistry& logicalRoutes,
    const PersistentRouteCandidate& route,
    const std::optional<cv::Vec3d>& initialDirection,
    ExactPersistentStateBudget& stateBudget,
    size_t& generatedStates,
    size_t& rejectedStates)
{
    return persistentRouteSuccessors(
        graph,
        logicalRoutes,
        route,
        initialDirection,
        [&]() {
            stateBudget.consume();
            ++generatedStates;
        },
        rejectedStates);
}

struct ExactPersistentRouteScore {
    FiberletGraphReplayCost edgeCost;
    FiberletGraphReplayCost transitionCost;
    double scoredLength = 0.0;

    [[nodiscard]] double total() const noexcept { return edgeCost.total() + transitionCost.total(); }
};

ExactPersistentRouteScore scorePersistentRouteUnweightedAtHorizon(
    const FiberletReplayGraphSource& graph,
    const PersistentRouteCandidate& route,
    double horizonPredictionVoxels)
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

struct GeometricReplayCostConfig {
    FiberletGraphReplayCostMode mode = FiberletGraphReplayCostMode::Fiberlet;
    double predictionToBaseScale = 1.0;
    double weightPerBaseVoxel = 1.0;
    double delayPredictionVoxels = 0.0;
    double integrationStepPredictionVoxels = 1.0;
    double profileWeight = 1.0;
};

double geometricCostWeight(
    const GeometricReplayCostConfig& config,
    double distanceFromCheckpointPredictionVoxels);

double integrateArcCostProfile(
    const FiberletReplaySourceCostProfile& profile,
    double edgeLengthPredictionVoxels,
    double beginPredictionVoxels,
    double endPredictionVoxels,
    double edgeBeginFromCheckpointPredictionVoxels,
    double horizonFromCheckpointPredictionVoxels,
    const GeometricReplayCostConfig& config)
{
    if (!(beginPredictionVoxels >= 0.0) ||
        endPredictionVoxels < beginPredictionVoxels ||
        endPredictionVoxels > edgeLengthPredictionVoxels + kReplayEpsilon ||
        profile.segmentLengthsPredictionVoxels.empty() ||
        profile.segmentLengthsPredictionVoxels.size() !=
            profile.segmentCostDensities.size()) {
        throw std::invalid_argument("fiberlet cost profile interval is invalid");
    }
    double profileLength = 0.0;
    double profileCost = 0.0;
    for (size_t segment = 0; segment < profile.segmentLengthsPredictionVoxels.size(); ++segment) {
        const double segmentLength = profile.segmentLengthsPredictionVoxels[segment];
        const double density = profile.segmentCostDensities[segment];
        if (!(segmentLength > 0.0) || !std::isfinite(segmentLength) ||
            !(density >= 0.0) || !std::isfinite(density)) {
            throw std::invalid_argument("fiberlet cost profile is invalid");
        }
        profileLength += segmentLength;
        profileCost += density * segmentLength;
    }
    const double lengthTolerance =
        1.0e-4 * std::max(1.0, edgeLengthPredictionVoxels);
    if (std::abs(profileLength - edgeLengthPredictionVoxels) >
        lengthTolerance) {
        throw std::invalid_argument(
            "fiberlet cost profile length differs from its edge length");
    }
    const double averageDensity = profileCost / profileLength;

    double result = 0.0;
    size_t segment = 0;
    double segmentBegin = 0.0;
    while (segment + 1 < profile.segmentLengthsPredictionVoxels.size() &&
           segmentBegin + profile.segmentLengthsPredictionVoxels[segment] <=
               beginPredictionVoxels + kReplayEpsilon) {
        segmentBegin += profile.segmentLengthsPredictionVoxels[segment++];
    }
    double cursor = beginPredictionVoxels;
    while (cursor < endPredictionVoxels - kReplayEpsilon) {
        const double segmentEnd =
            segment + 1 == profile.segmentLengthsPredictionVoxels.size()
            ? edgeLengthPredictionVoxels
            : segmentBegin + profile.segmentLengthsPredictionVoxels[segment];
        const double relativeCursor =
            edgeBeginFromCheckpointPredictionVoxels + cursor;
        auto cellIndex = static_cast<std::int64_t>(std::floor(
            std::max(0.0, relativeCursor) /
            config.integrationStepPredictionVoxels));
        double cellBegin = static_cast<double>(cellIndex) *
            config.integrationStepPredictionVoxels;
        double cellEnd = std::min(
            horizonFromCheckpointPredictionVoxels,
            cellBegin + config.integrationStepPredictionVoxels);
        if (cellEnd <= relativeCursor + kReplayEpsilon &&
            cellEnd < horizonFromCheckpointPredictionVoxels -
                kReplayEpsilon) {
            ++cellIndex;
            cellBegin = static_cast<double>(cellIndex) *
                config.integrationStepPredictionVoxels;
            cellEnd = std::min(
                horizonFromCheckpointPredictionVoxels,
                cellBegin + config.integrationStepPredictionVoxels);
        }
        double weightRegionBegin = cellBegin;
        double weightRegionEnd = cellEnd;
        const double delay = config.delayPredictionVoxels;
        if (cellBegin < delay && delay < cellEnd) {
            if (relativeCursor < delay - kReplayEpsilon)
                weightRegionEnd = delay;
            else
                weightRegionBegin = delay;
        }
        const double partEnd = std::min({
            endPredictionVoxels,
            segmentEnd,
            cursor + (weightRegionEnd - relativeCursor)});
        if (!(partEnd > cursor + kReplayEpsilon)) {
            std::ostringstream message;
            message << std::setprecision(17)
                    << "fiberlet weighted integration did not advance: cursor="
                    << cursor << " end=" << endPredictionVoxels
                    << " segment_end=" << segmentEnd
                    << " relative_cursor=" << relativeCursor
                    << " cell_begin=" << cellBegin
                    << " cell_end=" << cellEnd
                    << " weight_region_begin=" << weightRegionBegin
                    << " weight_region_end=" << weightRegionEnd
                    << " horizon=" << horizonFromCheckpointPredictionVoxels
                    << " delay=" << delay;
            throw std::logic_error(message.str());
        }
        const double weightPosition =
            0.5 * (weightRegionBegin + weightRegionEnd);
        const double density =
            (1.0 - config.profileWeight) * averageDensity +
            config.profileWeight * profile.segmentCostDensities[segment];
        result += density * (partEnd - cursor) *
            geometricCostWeight(config, weightPosition);
        cursor = partEnd;
        if (cursor >= segmentEnd - kReplayEpsilon &&
            segment + 1 < profile.segmentLengthsPredictionVoxels.size()) {
            segmentBegin = segmentEnd;
            ++segment;
        }
    }
    return result;
}

double geometricCostWeight(
    const GeometricReplayCostConfig& config,
    double distanceFromCheckpointPredictionVoxels)
{
    if (config.weightPerBaseVoxel == 1.0)
        return 1.0;
    return std::pow(
        config.weightPerBaseVoxel,
        std::max(
            0.0,
            distanceFromCheckpointPredictionVoxels -
                config.delayPredictionVoxels) *
            config.predictionToBaseScale);
}

class DecisionCostProfileCache
{
public:
    explicit DecisionCostProfileCache(
        const FiberletReplayGraphSource& graph,
        size_t maximumProfiles = 4096)
        : graph_(graph)
        , maximumProfiles_(maximumProfiles)
    {
    }

    double integrate(
        const FiberletReplaySourceArc& edge,
        double beginPredictionVoxels,
        double endPredictionVoxels,
        double edgeBeginFromCheckpointPredictionVoxels,
        double horizonFromCheckpointPredictionVoxels,
        const GeometricReplayCostConfig& config)
    {
        const auto found = profiles_.find(edge.id);
        if (found != profiles_.end()) {
            return integrateArcCostProfile(
                found->second,
                edge.pathLengthPredictionVoxels,
                beginPredictionVoxels,
                endPredictionVoxels,
                edgeBeginFromCheckpointPredictionVoxels,
                horizonFromCheckpointPredictionVoxels,
                config);
        }
        auto profile = graph_.costProfile(edge.id);
        if (profiles_.size() < maximumProfiles_) {
            const auto [inserted, wasInserted] = profiles_.emplace(edge.id, std::move(profile));
            (void)wasInserted;
            return integrateArcCostProfile(
                inserted->second,
                edge.pathLengthPredictionVoxels,
                beginPredictionVoxels,
                endPredictionVoxels,
                edgeBeginFromCheckpointPredictionVoxels,
                horizonFromCheckpointPredictionVoxels,
                config);
        }
        return integrateArcCostProfile(
            profile,
            edge.pathLengthPredictionVoxels,
            beginPredictionVoxels,
            endPredictionVoxels,
            edgeBeginFromCheckpointPredictionVoxels,
            horizonFromCheckpointPredictionVoxels,
            config);
    }

private:
    const FiberletReplayGraphSource& graph_;
    size_t maximumProfiles_ = 0;
    std::map<DirectedFiberletStorageId, FiberletReplaySourceCostProfile> profiles_;
};

struct DecisionPersistentRouteScore {
    double prefixEdgeLoss = 0.0;
    double prefixTransitionLoss = 0.0;
    double forwardEdgeLoss = 0.0;
    double forwardTransitionLoss = 0.0;
    double scoredEndPredictionVoxels = 0.0;

    [[nodiscard]] double total() const noexcept
    {
        return prefixEdgeLoss + prefixTransitionLoss +
            forwardEdgeLoss + forwardTransitionLoss;
    }
};

void addEdgeToDecisionScore(
    DecisionCostProfileCache& profiles,
    const FiberletReplaySourceArc& edge,
    const std::optional<FiberletReplaySourceTransition>& enteringTransition,
    double edgeBeginPredictionVoxels,
    double checkpointPredictionVoxels,
    double horizonPredictionVoxels,
    const GeometricReplayCostConfig& config,
    DecisionPersistentRouteScore& score)
{
    const double edgeEndPredictionVoxels =
        edgeBeginPredictionVoxels + edge.pathLengthPredictionVoxels;
    if (edgeBeginPredictionVoxels >= horizonPredictionVoxels - kReplayEpsilon)
        return;
    validateReplayCost(edge.cost, "persistent beam edge cost must be finite and nonnegative");
    if (!(edge.pathLengthPredictionVoxels > kFloatEpsilon) ||
        !std::isfinite(edge.pathLengthPredictionVoxels)) {
        throw std::invalid_argument("persistent beam edge length is invalid");
    }

    const double prefixEnd = std::min({
        edgeEndPredictionVoxels,
        checkpointPredictionVoxels,
        horizonPredictionVoxels});
    if (prefixEnd > edgeBeginPredictionVoxels + kReplayEpsilon) {
        score.prefixEdgeLoss += edge.cost.total() * std::clamp(
            (prefixEnd - edgeBeginPredictionVoxels) /
                edge.pathLengthPredictionVoxels,
            0.0,
            1.0);
    }

    const double forwardBegin =
        std::max(edgeBeginPredictionVoxels, checkpointPredictionVoxels);
    const double forwardEnd =
        std::min(edgeEndPredictionVoxels, horizonPredictionVoxels);
    if (forwardEnd > forwardBegin + kReplayEpsilon) {
        if (config.mode == FiberletGraphReplayCostMode::Fiberlet) {
            score.forwardEdgeLoss += edge.cost.total() * std::clamp(
                (forwardEnd - forwardBegin) /
                    edge.pathLengthPredictionVoxels,
                0.0,
                1.0);
        } else {
            score.forwardEdgeLoss += profiles.integrate(
                edge,
                forwardBegin - edgeBeginPredictionVoxels,
                forwardEnd - edgeBeginPredictionVoxels,
                edgeBeginPredictionVoxels - checkpointPredictionVoxels,
                horizonPredictionVoxels - checkpointPredictionVoxels,
                config);
        }
    }

    if (enteringTransition.has_value() &&
        edgeBeginPredictionVoxels < horizonPredictionVoxels - kReplayEpsilon) {
        validateReplayCost(
            enteringTransition->cost,
            "persistent beam join cost must be finite and nonnegative");
        if (edgeBeginPredictionVoxels < checkpointPredictionVoxels - kReplayEpsilon) {
            score.prefixTransitionLoss += enteringTransition->cost.total();
        } else {
            const double weight =
                config.mode == FiberletGraphReplayCostMode::Fiberlet
                ? 1.0
                : geometricCostWeight(
                      config,
                      edgeBeginPredictionVoxels - checkpointPredictionVoxels);
            score.forwardTransitionLoss += enteringTransition->cost.total() * weight;
        }
    }
    score.scoredEndPredictionVoxels = std::min(
        horizonPredictionVoxels,
        std::max(score.scoredEndPredictionVoxels, edgeEndPredictionVoxels));
}

DecisionPersistentRouteScore scorePersistentRouteForDecision(
    const FiberletReplayGraphSource& graph,
    DecisionCostProfileCache& profiles,
    const PersistentRouteCandidate& route,
    double checkpointPredictionVoxels,
    double horizonPredictionVoxels,
    const GeometricReplayCostConfig& config,
    size_t* visitedHistoryNodes = nullptr)
{
    if (!(checkpointPredictionVoxels >= 0.0) ||
        !std::isfinite(checkpointPredictionVoxels) ||
        horizonPredictionVoxels < checkpointPredictionVoxels ||
        !std::isfinite(horizonPredictionVoxels) ||
        route.pathLength < checkpointPredictionVoxels - kReplayEpsilon ||
        !(config.predictionToBaseScale > 0.0) ||
        !std::isfinite(config.predictionToBaseScale)) {
        throw std::invalid_argument("persistent weighted route interval is invalid");
    }
    std::vector<const PersistentRouteHistory*> suffix;
    for (auto history = route.tail; history != nullptr;
         history = history->parent) {
        suffix.push_back(history.get());
        if (history->parent == nullptr ||
            history->parent->cumulativePathLength <
                checkpointPredictionVoxels - kReplayEpsilon) {
            break;
        }
    }
    std::reverse(suffix.begin(), suffix.end());
    if (visitedHistoryNodes != nullptr)
        *visitedHistoryNodes += suffix.size();

    DecisionPersistentRouteScore score;
    if (!suffix.empty() && suffix.front()->parent != nullptr) {
        const auto& prefix = *suffix.front()->parent;
        validateReplayCost(
            prefix.cumulativeEdgeCost,
            "persistent beam cumulative edge cost must be finite and nonnegative");
        validateReplayCost(
            prefix.cumulativeTransitionCost,
            "persistent beam cumulative join cost must be finite and nonnegative");
        score.prefixEdgeLoss = prefix.cumulativeEdgeCost.total();
        score.prefixTransitionLoss = prefix.cumulativeTransitionCost.total();
        score.scoredEndPredictionVoxels = prefix.cumulativePathLength;
    }
    for (const auto* history : suffix) {
        const double edgeBegin = history->parent != nullptr
            ? history->parent->cumulativePathLength
            : 0.0;
        addEdgeToDecisionScore(
            profiles,
            graph.arc(history->arc),
            history->enteringTransition,
            edgeBegin,
            checkpointPredictionVoxels,
            horizonPredictionVoxels,
            config,
            score);
        if (history->cumulativePathLength >= horizonPredictionVoxels - kReplayEpsilon)
            break;
    }
    score.scoredEndPredictionVoxels =
        std::min(route.pathLength, horizonPredictionVoxels);
    return score;
}

DecisionPersistentRouteScore extendDecisionRouteScore(
    const FiberletReplayGraphSource& graph,
    DecisionCostProfileCache& profiles,
    const PersistentRouteCandidate& successor,
    const DecisionPersistentRouteScore& parentScore,
    double checkpointPredictionVoxels,
    double horizonPredictionVoxels,
    const GeometricReplayCostConfig& config)
{
    if (successor.tail == nullptr)
        throw std::logic_error("persistent successor has no history");
    DecisionPersistentRouteScore score = parentScore;
    const double edgeBegin = successor.tail->parent != nullptr
        ? successor.tail->parent->cumulativePathLength
        : 0.0;
    addEdgeToDecisionScore(
        profiles,
        graph.arc(successor.tail->arc),
        successor.tail->enteringTransition,
        edgeBegin,
        checkpointPredictionVoxels,
        horizonPredictionVoxels,
        config,
        score);
    return score;
}

PersistentRouteCandidate persistentRouteCommittedPrefix(
    const FiberletReplayGraphSource& graph,
    const PersistentRouteCandidate& route,
    double checkpointPredictionVoxels,
    const std::shared_ptr<const PersistentLogicalRouteNode>& logicalRoot)
{
    (void)graph;
    if (!(checkpointPredictionVoxels >= 0.0) || !std::isfinite(checkpointPredictionVoxels) || route.pathLength < checkpointPredictionVoxels - kReplayEpsilon) {
        throw std::invalid_argument("persistent beam route does not cover the checkpoint");
    }
    PersistentRouteCandidate prefix;
    prefix.seed = route.seed;
    prefix.visitedNodes = persistentVisitedAdd(nullptr, route.seed);
    prefix.logicalRoute = logicalRoot;
    prefix.evaluation = route.evaluation;
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
    prefix.logicalRoute = crossing->logicalRoute;
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

std::vector<cv::Vec3d> persistentRoutePointsBetween(
    const FiberletReplayGraphSource& graph,
    const PersistentRouteCandidate& route,
    double beginPathLengthPredictionVoxels,
    double endPathLengthPredictionVoxels)
{
    if (!(beginPathLengthPredictionVoxels >= 0.0) ||
        !(endPathLengthPredictionVoxels >= beginPathLengthPredictionVoxels) ||
        !std::isfinite(beginPathLengthPredictionVoxels) ||
        !std::isfinite(endPathLengthPredictionVoxels)) {
        throw std::invalid_argument("persistent route geometry interval is invalid");
    }
    std::vector<const PersistentRouteHistory*> suffix;
    for (auto node = route.tail;
         node != nullptr &&
         node->cumulativePathLength > beginPathLengthPredictionVoxels +
             kReplayEpsilon;
         node = node->parent) {
        const double edgeBegin = node->parent != nullptr
            ? node->parent->cumulativePathLength
            : 0.0;
        if (edgeBegin < endPathLengthPredictionVoxels - kReplayEpsilon)
            suffix.push_back(node.get());
    }
    if (suffix.empty())
        return {};
    std::reverse(suffix.begin(), suffix.end());

    std::vector<cv::Vec3d> result;
    for (const auto* node : suffix) {
        const double edgeBegin = node->parent != nullptr
            ? node->parent->cumulativePathLength
            : 0.0;
        const double edgeEnd = node->cumulativePathLength;
        const double overlapBegin =
            std::max(beginPathLengthPredictionVoxels, edgeBegin);
        const double overlapEnd =
            std::min(endPathLengthPredictionVoxels, edgeEnd);
        if (!(overlapEnd > overlapBegin + kReplayEpsilon))
            continue;

        const auto edgeGeometry = makePolylineArcGeometry(
            graph.routePoints(node->arc));
        const double edgeLength = edgeEnd - edgeBegin;
        const double geometryBegin = edgeGeometry.length() *
            (overlapBegin - edgeBegin) / edgeLength;
        const double geometryEnd = edgeGeometry.length() *
            (overlapEnd - edgeBegin) / edgeLength;
        auto points = slicePolylineArc(
            edgeGeometry, geometryBegin, geometryEnd);
        if (!result.empty() && !points.empty() &&
            length(result.back() - points.front()) <= kReplayEpsilon) {
            points.erase(points.begin());
        }
        result.insert(result.end(), points.begin(), points.end());
    }
    return result;
}

struct RankedPersistentPrefix {
    PersistentRouteCandidate prefix;
    PersistentRouteCandidate lookahead;
    FiberletGraphReplayCost scoredEdgeCost;
    FiberletGraphReplayCost scoredTransitionCost;
    double weightedEdgeLoss = 0.0;
    double weightedTransitionLoss = 0.0;
    double scoredPathLength = 0.0;
    double completePathLength = 0.0;
    double lossPerPredictionVoxel = 0.0;
    double totalLoss = 0.0;
};

RankedPersistentPrefix makeRankedPersistentPrefix(
    const FiberletReplayGraphSource& graph,
    PersistentRouteCandidate candidate,
    const DecisionPersistentRouteScore& score,
    double checkpointPredictionVoxels,
    double scoringHorizonPredictionVoxels,
    const std::shared_ptr<const PersistentLogicalRouteNode>& logicalRoot)
{
    RankedPersistentPrefix entry;
    const auto diagnosticScore = scorePersistentRouteUnweightedAtHorizon(
        graph, candidate, scoringHorizonPredictionVoxels);
    entry.prefix = persistentRouteCommittedPrefix(
        graph, candidate, checkpointPredictionVoxels, logicalRoot);
    entry.lookahead = std::move(candidate);
    entry.scoredEdgeCost = diagnosticScore.edgeCost;
    entry.scoredTransitionCost = diagnosticScore.transitionCost;
    entry.weightedEdgeLoss = score.forwardEdgeLoss;
    entry.weightedTransitionLoss = score.forwardTransitionLoss;
    entry.scoredPathLength = scoringHorizonPredictionVoxels;
    entry.completePathLength = entry.lookahead.pathLength;
    entry.totalLoss = score.total();
    entry.lossPerPredictionVoxel = scoringHorizonPredictionVoxels > kReplayEpsilon
        ? entry.totalLoss / scoringHorizonPredictionVoxels
        : std::numeric_limits<double>::infinity();
    return entry;
}

bool rankedPersistentPrefixLess(const RankedPersistentPrefix& left, const RankedPersistentPrefix& right)
{
    if (left.totalLoss != right.totalLoss)
        return left.totalLoss < right.totalLoss;
    if (left.lookahead.logicalRoute != right.lookahead.logicalRoute)
        return logicalRouteLess(left.lookahead.logicalRoute.get(), right.lookahead.logicalRoute.get());
    return logicalRouteLess(left.prefix.logicalRoute.get(), right.prefix.logicalRoute.get());
}

struct ExactPersistentSearchStats {
    size_t generated = 0;
    size_t expanded = 0;
    size_t completed = 0;
    size_t costPruned = 0;
    size_t rejected = 0;
    size_t dominated = 0;
    size_t relaxedBoundStates = 0;
    size_t relaxedBoundHits = 0;
    size_t relaxedBoundZeroFallbacks = 0;
    size_t initializationHistoryNodeCount = 0;
    size_t logicalRouteInternCount = 0;
    size_t logicalRouteCleanupVisitedCount = 0;
};

struct PersistentQueueEntry {
    PersistentRouteCandidate route;
    DecisionPersistentRouteScore score;
    double lowerBound = 0.0;
    size_t sequence = 0;
};

struct PersistentQueueGreater {
    bool operator()(const PersistentQueueEntry& left, const PersistentQueueEntry& right) const noexcept
    {
        return std::tie(left.lowerBound, left.sequence) > std::tie(right.lowerBound, right.sequence);
    }
};

struct ExactPersistentQueueGreater {
    bool operator()(const PersistentQueueEntry& left, const PersistentQueueEntry& right) const
    {
        if (left.lowerBound != right.lowerBound)
            return left.lowerBound > right.lowerBound;
        if (left.route.seed != right.route.seed)
            return right.route.seed < left.route.seed;
        if (left.route.logicalRoute != right.route.logicalRoute)
            return logicalRouteLess(right.route.logicalRoute.get(), left.route.logicalRoute.get());
        return left.sequence > right.sequence;
    }
};

class WeightedRelaxedPersistentCostToGo
{
public:
    WeightedRelaxedPersistentCostToGo(
        const FiberletReplayGraphSource& graph,
        DecisionCostProfileCache& profiles,
        const GeometricReplayCostConfig& config,
        double horizonFromCheckpointPredictionVoxels,
        size_t maximumStates)
        : graph_(graph)
        , profiles_(profiles)
        , config_(config)
        , horizonFromCheckpointPredictionVoxels_(
              horizonFromCheckpointPredictionVoxels)
        , maximumStates_(maximumStates)
    {
    }

    double lowerBound(
        const DirectedFiberletStorageId& incoming,
        double remainingPredictionVoxels)
    {
        if (!(remainingPredictionVoxels >= 0.0) ||
            !std::isfinite(remainingPredictionVoxels)) {
            throw std::invalid_argument(
                "persistent route remaining distance is invalid");
        }
        return solve(incoming, distanceBins(remainingPredictionVoxels));
    }

    [[nodiscard]] size_t states() const noexcept { return memo_.size(); }
    [[nodiscard]] size_t hits() const noexcept { return hits_; }
    [[nodiscard]] size_t zeroFallbacks() const noexcept { return zeroFallbacks_; }

private:
    [[nodiscard]] static size_t distanceBins(double distance)
    {
        return static_cast<size_t>(std::floor(
            std::max(0.0, distance) / kDistanceBin));
    }

    double solve(const DirectedFiberletStorageId& incomingId, size_t bins)
    {
        if (bins == 0)
            return 0.0;
        const auto key = std::pair{incomingId, bins};
        if (const auto found = memo_.find(key); found != memo_.end()) {
            ++hits_;
            return found->second;
        }
        if (memo_.size() >= maximumStates_) {
            ++zeroFallbacks_;
            return 0.0;
        }

        const double targetDistance = static_cast<double>(bins) * kDistanceBin;
        const double relativeStart = std::max(
            0.0,
            horizonFromCheckpointPredictionVoxels_ - targetDistance);
        const auto incoming = graph_.arc(incomingId);
        double best = std::numeric_limits<double>::infinity();
        const auto outgoingArcs = graph_.outgoingArcs(incoming.target);
        for (size_t outgoingIndex = 0;
             outgoingIndex < outgoingArcs.size(); ++outgoingIndex) {
            const auto& outgoing = outgoingArcs[outgoingIndex];
            const auto transition = graph_.transition(incoming, outgoing);
            if (!transition.has_value())
                continue;
            validateReplayCost(
                transition->cost,
                "persistent beam join cost must be finite and nonnegative");
            if (!(outgoing.pathLengthPredictionVoxels > kFloatEpsilon) ||
                !std::isfinite(outgoing.pathLengthPredictionVoxels)) {
                throw std::invalid_argument(
                    "persistent beam edge length is invalid");
            }
            const double includedLength = std::min(
                targetDistance,
                static_cast<double>(outgoing.pathLengthPredictionVoxels));
            double candidate = 0.0;
            if (config_.mode == FiberletGraphReplayCostMode::Fiberlet) {
                candidate = transition->cost.total();
                validateReplayCost(
                    outgoing.cost,
                    "persistent beam edge cost must be finite and nonnegative");
                candidate += outgoing.cost.total() *
                    std::clamp(
                        includedLength / outgoing.pathLengthPredictionVoxels,
                        0.0,
                        1.0);
            } else {
                candidate = transition->cost.total() *
                    geometricCostWeight(config_, relativeStart);
                candidate += profiles_.integrate(
                    outgoing,
                    0.0,
                    includedLength,
                    relativeStart,
                    horizonFromCheckpointPredictionVoxels_,
                    config_);
            }
            if (outgoing.pathLengthPredictionVoxels <
                targetDistance - kReplayEpsilon) {
                const size_t continuationBins = std::min(
                    bins - 1,
                    distanceBins(
                        targetDistance -
                        outgoing.pathLengthPredictionVoxels));
                const double continuation = solve(
                    outgoing.id, continuationBins);
                candidate += continuation;
            }
            best = std::min(best, candidate);
        }
        memo_.emplace(key, best);
        return best;
    }

    static constexpr double kDistanceBin = 0.5;
    const FiberletReplayGraphSource& graph_;
    DecisionCostProfileCache& profiles_;
    const GeometricReplayCostConfig& config_;
    double horizonFromCheckpointPredictionVoxels_ = 0.0;
    size_t maximumStates_ = 0;
    std::map<std::pair<DirectedFiberletStorageId, size_t>, double> memo_;
    size_t hits_ = 0;
    size_t zeroFallbacks_ = 0;
};

struct RankedPersistentCompletion {
    PersistentRouteCandidate route;
    DecisionPersistentRouteScore decisionScore;
    FiberletGraphReplayCost scoredEdgeCost;
    FiberletGraphReplayCost scoredTransitionCost;
    double scoredPathLength = 0.0;
    double completePathLength = 0.0;
    double totalLoss = 0.0;
    size_t sequence = 0;
};

bool rankedPersistentCompletionLess(const RankedPersistentCompletion& left, const RankedPersistentCompletion& right)
{
    if (left.totalLoss != right.totalLoss)
        return left.totalLoss < right.totalLoss;
    if (left.route.seed != right.route.seed)
        return left.route.seed < right.route.seed;
    if (left.route.logicalRoute != right.route.logicalRoute)
        return logicalRouteLess(left.route.logicalRoute.get(), right.route.logicalRoute.get());
    return left.sequence < right.sequence;
}

void retainRankedPersistentCompletion(RankedPersistentCompletion entry, size_t beamWidth, std::vector<RankedPersistentCompletion>& ranked)
{
    const auto equivalent = std::find_if(ranked.begin(), ranked.end(), [&](const auto& existing) {
        return existing.route.seed == entry.route.seed && existing.route.logicalRoute == entry.route.logicalRoute;
    });
    if (equivalent != ranked.end()) {
        if (!rankedPersistentCompletionLess(entry, *equivalent))
            return;
        ranked.erase(equivalent);
    }

    const auto insertion = std::lower_bound(ranked.begin(), ranked.end(), entry, rankedPersistentCompletionLess);
    ranked.insert(insertion, std::move(entry));
    if (ranked.size() > beamWidth)
        ranked.pop_back();
}

std::vector<RankedPersistentPrefix> exactPersistentRouteLookahead(
    const FiberletReplayGraphSource& graph,
    PersistentLogicalRouteRegistry& logicalRoutes,
    const std::vector<PersistentRouteCandidate>& initialRoutes,
    double scoringBeginPredictionVoxels,
    double scoringHorizonPredictionVoxels,
    double checkpointPredictionVoxels,
    const cv::Vec3d& initialDirection,
    size_t beamWidth,
    utils::ThreadPool& expansionPool,
    size_t maximumGeneratedStates,
    const GeometricReplayCostConfig& geometricCost,
    ExactPersistentSearchStats& stats)
{
    if (!(scoringHorizonPredictionVoxels > 0.0) || !std::isfinite(scoringHorizonPredictionVoxels) || !(checkpointPredictionVoxels >= 0.0) ||
        !std::isfinite(checkpointPredictionVoxels)) {
        throw std::invalid_argument("persistent exact lookahead horizon is invalid");
    }
    constexpr size_t kExpansionBatchSize = 32;
    ExactPersistentStateBudget stateBudget;
    stateBudget.maximum = maximumGeneratedStates;
    DecisionCostProfileCache profiles(graph);
    WeightedRelaxedPersistentCostToGo costToGo(
        graph,
        profiles,
        geometricCost,
        scoringHorizonPredictionVoxels - scoringBeginPredictionVoxels,
        std::max<size_t>(1, std::min<size_t>(maximumGeneratedStates, 250'000)));
    std::priority_queue<PersistentQueueEntry, std::vector<PersistentQueueEntry>, ExactPersistentQueueGreater> pending;
    std::vector<RankedPersistentCompletion> completions;
    size_t queueSequence = 0;
    size_t completionSequence = 0;

    const auto completionCutoff = [&]() -> std::optional<double> {
        if (completions.size() < beamWidth)
            return std::nullopt;
        return completions.back().totalLoss;
    };
    const auto retainCompletion = [&](
        PersistentRouteCandidate candidate,
        DecisionPersistentRouteScore score) {
        ++stats.completed;
        const auto diagnosticScore = scorePersistentRouteUnweightedAtHorizon(
            graph, candidate, scoringHorizonPredictionVoxels);
        const double completePathLength = candidate.pathLength;
        retainRankedPersistentCompletion(
            RankedPersistentCompletion{
                std::move(candidate),
                score,
                diagnosticScore.edgeCost,
                diagnosticScore.transitionCost,
                scoringHorizonPredictionVoxels,
                completePathLength,
                score.total(),
                completionSequence++},
            beamWidth,
            completions);
    };
    const auto routeLowerBound = [&](
        const PersistentRouteCandidate& route,
        const DecisionPersistentRouteScore& score) {
        const auto incoming = persistentRouteIncoming(route);
        if (!incoming.has_value())
            return score.total();
        return score.total() + costToGo.lowerBound(
            *incoming,
            std::max(
                0.0,
                scoringHorizonPredictionVoxels - route.pathLength));
    };
    const auto enqueue = [&](
        PersistentRouteCandidate candidate,
        DecisionPersistentRouteScore score) {
        const double lowerBound = routeLowerBound(candidate, score);
        if (!(lowerBound >= 0.0) || std::isnan(lowerBound))
            throw std::invalid_argument("persistent route lower bound must be finite and nonnegative");
        const auto cutoff = completionCutoff();
        if (!std::isfinite(lowerBound) || (cutoff.has_value() && lowerBound > *cutoff)) {
            ++stats.costPruned;
            return;
        }
        pending.push(PersistentQueueEntry{
            std::move(candidate), std::move(score), lowerBound, queueSequence++});
    };

    for (const auto& route : initialRoutes) {
        auto score = scorePersistentRouteForDecision(
            graph,
            profiles,
            route,
            scoringBeginPredictionVoxels,
            scoringHorizonPredictionVoxels,
            geometricCost,
            &stats.initializationHistoryNodeCount);
        if (route.pathLength >= scoringHorizonPredictionVoxels - kReplayEpsilon)
            retainCompletion(route, std::move(score));
        else
            enqueue(route, std::move(score));
    }

    struct Expansion {
        std::vector<PersistentRouteCandidate> successors;
        size_t rejected = 0;
        std::exception_ptr error;
    };
    while (!pending.empty()) {
        const auto cutoff = completionCutoff();
        if (cutoff.has_value() && pending.top().lowerBound > *cutoff) {
            stats.costPruned += pending.size();
            break;
        }
        std::vector<PersistentQueueEntry> batch;
        batch.reserve(kExpansionBatchSize);
        while (batch.size() < kExpansionBatchSize && !pending.empty()) {
            if (cutoff.has_value() && pending.top().lowerBound > *cutoff)
                break;
            batch.push_back(pending.top());
            pending.pop();
        }
        std::vector<Expansion> expansions(batch.size());
        expansionPool.run_indexed_batch(batch.size(), [&](size_t index) {
            auto& expansion = expansions[index];
            try {
                expansion.successors = persistentRouteSuccessors(
                    graph,
                    logicalRoutes,
                    batch[index].route,
                    batch[index].route.tail == nullptr ? std::make_optional(initialDirection) : std::nullopt,
                    []() {},
                    expansion.rejected);
            } catch (...) {
                expansion.error = std::current_exception();
            }
        });
        for (const auto& expansion : expansions) {
            if (expansion.error)
                std::rethrow_exception(expansion.error);
        }
        stats.expanded += batch.size();
        for (size_t expansionIndex = 0;
             expansionIndex < expansions.size(); ++expansionIndex) {
            auto& expansion = expansions[expansionIndex];
            stats.rejected += expansion.rejected;
            for (auto& successor : expansion.successors) {
                stateBudget.consume();
                ++stats.generated;
                auto score = extendDecisionRouteScore(
                    graph,
                    profiles,
                    successor,
                    batch[expansionIndex].score,
                    scoringBeginPredictionVoxels,
                    scoringHorizonPredictionVoxels,
                    geometricCost);
                if (successor.pathLength >= scoringHorizonPredictionVoxels - kReplayEpsilon)
                    retainCompletion(std::move(successor), std::move(score));
                else
                    enqueue(std::move(successor), std::move(score));
            }
        }
    }

    std::vector<RankedPersistentPrefix> ranked;
    ranked.reserve(completions.size());
    for (auto& completion : completions) {
        RankedPersistentPrefix entry;
        entry.prefix = persistentRouteCommittedPrefix(graph, completion.route, checkpointPredictionVoxels, logicalRoutes.root());
        entry.lookahead = std::move(completion.route);
        entry.scoredEdgeCost = completion.scoredEdgeCost;
        entry.scoredTransitionCost = completion.scoredTransitionCost;
        entry.weightedEdgeLoss = completion.decisionScore.forwardEdgeLoss;
        entry.weightedTransitionLoss = completion.decisionScore.forwardTransitionLoss;
        entry.scoredPathLength = completion.scoredPathLength;
        entry.completePathLength = completion.completePathLength;
        entry.totalLoss = completion.totalLoss;
        entry.lossPerPredictionVoxel = entry.scoredPathLength > kReplayEpsilon
            ? entry.totalLoss / entry.scoredPathLength
            : std::numeric_limits<double>::infinity();
        ranked.push_back(std::move(entry));
    }
    stats.relaxedBoundStates += costToGo.states();
    stats.relaxedBoundHits += costToGo.hits();
    stats.relaxedBoundZeroFallbacks += costToGo.zeroFallbacks();
    return ranked;
}

struct BoundedPersistentRoute {
    PersistentRouteCandidate route;
    std::shared_ptr<const PersistentLogicalRouteNode> stablePrefixLogicalRoute;
};

struct BoundedRankedPersistentRoute {
    RankedPersistentPrefix ranked;
    std::shared_ptr<const PersistentLogicalRouteNode> stablePrefixLogicalRoute;
};

struct StablePersistentPrefixKey {
    FiberletStorageKey seed;
    std::shared_ptr<const PersistentLogicalRouteNode> logicalRoute;
};

struct StablePersistentPrefixKeyLess {
    bool operator()(const StablePersistentPrefixKey& left, const StablePersistentPrefixKey& right) const
    {
        if (left.seed != right.seed)
            return left.seed < right.seed;
        return logicalRouteLess(left.logicalRoute.get(), right.logicalRoute.get());
    }
};

bool operator==(const StablePersistentPrefixKey& left, const StablePersistentPrefixKey& right)
{
    return left.seed == right.seed && left.logicalRoute == right.logicalRoute;
}

StablePersistentPrefixKey stablePersistentPrefixKey(const BoundedRankedPersistentRoute& route)
{
    return {route.ranked.lookahead.seed, route.stablePrefixLogicalRoute};
}

StablePersistentPrefixKey completePersistentRouteKey(const BoundedRankedPersistentRoute& route)
{
    return {route.ranked.lookahead.seed, route.ranked.lookahead.logicalRoute};
}

bool boundedRankedPersistentRouteLess(const BoundedRankedPersistentRoute& left, const BoundedRankedPersistentRoute& right)
{
    if (rankedPersistentPrefixLess(left.ranked, right.ranked))
        return true;
    if (rankedPersistentPrefixLess(right.ranked, left.ranked))
        return false;
    return StablePersistentPrefixKeyLess{}(stablePersistentPrefixKey(left), stablePersistentPrefixKey(right));
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
    return logicalRouteLess(route.logicalRoute.get(), winner.route.logicalRoute.get());
}

BoundedPersistentExpansionResult expandPersistentRouteToFront(
    const FiberletReplayGraphSource& graph,
    PersistentLogicalRouteRegistry& logicalRoutes,
    const BoundedPersistentRoute& initial,
    double frontPredictionVoxels,
    double frontBeginPredictionVoxels,
    double scoringBeginPredictionVoxels,
    double checkpointPredictionVoxels,
    double stablePrefixPredictionVoxels,
    const cv::Vec3d& initialDirection,
    size_t searchWidth,
    const GeometricReplayCostConfig& geometricCost,
    ExactPersistentStateBudget& stateBudget)
{
    BoundedPersistentExpansionResult result;
    DecisionCostProfileCache profiles(graph);
    std::priority_queue<PersistentQueueEntry, std::vector<PersistentQueueEntry>, PersistentQueueGreater> pending;
    std::map<PersistentLabelKey, PersistentLabelWinner> bestLabels;
    std::map<PersistentLabelKey, BoundedRankedPersistentRoute> bestCompletions;
    std::multiset<double> completionLosses;
    std::optional<double> completionCutoff;
    std::optional<double> appliedLocalCompletionCutoffPerPredictionVoxel;
    size_t sequence = 0;
    const auto enqueue = [&](
        PersistentRouteCandidate route,
        DecisionPersistentRouteScore score) {
        const auto key = persistentLabelKey(graph, route, frontPredictionVoxels);
        const double accumulatedLoss = score.total();
        const auto found = bestLabels.find(key);
        if (found != bestLabels.end() && !persistentLabelLess(route, accumulatedLoss, found->second)) {
            ++result.stats.dominated;
            return;
        }
        bestLabels.insert_or_assign(key, PersistentLabelWinner{accumulatedLoss, route});
        const double lowerBound = accumulatedLoss;
        if (!(lowerBound >= 0.0) || std::isnan(lowerBound))
            throw std::invalid_argument("bounded front lower bound must be nonnegative");
        if (!std::isfinite(lowerBound)) {
            ++result.stats.costPruned;
            return;
        }
        pending.push({
            std::move(route), std::move(score), lowerBound, sequence++});
    };
    auto initialScore = scorePersistentRouteForDecision(
        graph,
        profiles,
        initial.route,
        scoringBeginPredictionVoxels,
        frontPredictionVoxels,
        geometricCost,
        &result.stats.initializationHistoryNodeCount);
    if (initial.route.pathLength >= frontPredictionVoxels - kReplayEpsilon) {
        pending.push({
            initial.route,
            initialScore,
            initialScore.total(),
            sequence++});
    } else {
        enqueue(initial.route, std::move(initialScore));
    }
    while (!pending.empty()) {
        if (completionCutoff.has_value() && pending.top().lowerBound > *completionCutoff) {
            const double frontLength = frontPredictionVoxels - frontBeginPredictionVoxels;
            if (frontLength > kReplayEpsilon) {
                const double beginLoss = scorePersistentRouteForDecision(
                    graph,
                    profiles,
                    initial.route,
                    scoringBeginPredictionVoxels,
                    frontBeginPredictionVoxels,
                    geometricCost,
                    &result.stats.initializationHistoryNodeCount).total();
                appliedLocalCompletionCutoffPerPredictionVoxel = std::max(0.0, (*completionCutoff - beginLoss) / frontLength);
            }
            result.stats.costPruned += pending.size();
            break;
        }
        auto queued = pending.top();
        pending.pop();
        auto route = std::move(queued.route);
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
            auto ranked = makeRankedPersistentPrefix(
                graph,
                std::move(route),
                queued.score,
                checkpointPredictionVoxels,
                frontPredictionVoxels,
                logicalRoutes.root());
            auto stablePrefix = initial.stablePrefixLogicalRoute;
            if (!stablePrefix) {
                stablePrefix = persistentRouteCommittedPrefix(graph, ranked.lookahead, stablePrefixPredictionVoxels, logicalRoutes.root()).logicalRoute;
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
            logicalRoutes,
            route,
            route.tail == nullptr ? std::make_optional(initialDirection) : std::nullopt,
            stateBudget,
            result.stats.generated,
            result.stats.rejected);
        for (auto& successor : successors) {
            auto score = extendDecisionRouteScore(
                graph,
                profiles,
                successor,
                queued.score,
                scoringBeginPredictionVoxels,
                frontPredictionVoxels,
                geometricCost);
            if (successor.pathLength >= frontPredictionVoxels - kReplayEpsilon) {
                const double totalLoss = score.total();
                pending.push({
                    std::move(successor),
                    std::move(score),
                    totalLoss,
                    sequence++});
            } else {
                enqueue(std::move(successor), std::move(score));
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
    std::set<StablePersistentPrefixKey, StablePersistentPrefixKeyLess> completeRoutes;
    for (auto& candidate : candidates) {
        if (completeRoutes.insert(completePersistentRouteKey(candidate)).second)
            unique.push_back(std::move(candidate));
    }

    std::vector<BoundedRankedPersistentRoute> selected;
    selected.reserve(std::min(retainedWidth, unique.size()));
    std::set<StablePersistentPrefixKey, StablePersistentPrefixKeyLess> selectedPrefixes;
    std::set<StablePersistentPrefixKey, StablePersistentPrefixKeyLess> selectedRoutes;
    for (auto& candidate : unique) {
        if (selected.size() >= retainedWidth)
            break;
        const StablePersistentPrefixKey prefixKey =
            finalFront ? StablePersistentPrefixKey{candidate.ranked.prefix.seed, candidate.ranked.prefix.logicalRoute}
                       : stablePersistentPrefixKey(candidate);
        if (!selectedPrefixes.insert(prefixKey).second)
            continue;
        selectedRoutes.insert(completePersistentRouteKey(candidate));
        selected.push_back(candidate);
        ++diagnostics.diversityProtectedCount;
    }
    std::set<StablePersistentPrefixKey, StablePersistentPrefixKeyLess> allPrefixes;
    for (const auto& candidate : unique) {
        allPrefixes.insert(
            finalFront ? StablePersistentPrefixKey{candidate.ranked.prefix.seed, candidate.ranked.prefix.logicalRoute}
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
    std::shared_ptr<const PersistentLogicalRouteNode> selectedPrefixLogicalRoute;
};

BoundedPersistentLookaheadResult boundedPersistentRouteLookahead(
    const FiberletReplayGraphSource& graph,
    PersistentLogicalRouteRegistry& logicalRoutes,
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
    const GeometricReplayCostConfig& geometricCost,
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
        active.push_back({route, nullptr});

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
            std::set<StablePersistentPrefixKey, StablePersistentPrefixKeyLess> activePrefixes;
            for (const auto& route : active)
                activePrefixes.insert({route.route.seed, route.stablePrefixLogicalRoute});
            const size_t globalFillSlots = targetWidth > activePrefixes.size() ? targetWidth - activePrefixes.size() : 0;
            localCandidateLimit = std::min(targetWidth, globalFillSlots + 1);
        }
        frontDiagnostics.localCandidateLimit = localCandidateLimit;
        if (active.size() == 1 && active.front().route.tail == nullptr && active.front().route.pathLength < fronts[frontIndex] - kReplayEpsilon) {
            ++frontDiagnostics.expandedStateCount;
            size_t rootGenerated = 0;
            size_t rootRejected = 0;
            auto successors =
                persistentRouteSuccessors(graph, logicalRoutes, active.front().route, active.front().route.tail == nullptr ? std::make_optional(initialDirection) : std::nullopt, stateBudget, rootGenerated, rootRejected);
            frontDiagnostics.generatedStateCount += rootGenerated;
            frontDiagnostics.rejectedStateCount += rootRejected;
            const auto stablePrefix = active.front().stablePrefixLogicalRoute;
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
                        expandPersistentRouteToFront(
                            graph,
                            logicalRoutes,
                            active[inputIndex],
                            fronts[frontIndex],
                            frontBeginPredictionVoxels,
                            rolloutBeginPredictionVoxels,
                            checkpointPredictionVoxels,
                            firstFront,
                            initialDirection,
                            localCandidateLimit,
                            geometricCost,
                            stateBudget);
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
            stats.initializationHistoryNodeCount +=
                expansion.stats.initializationHistoryNodeCount;
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
                result.selectedPrefixLogicalRoute = selected.front().stablePrefixLogicalRoute;
            }
        } else {
            active.clear();
            active.reserve(selected.size());
            for (auto& candidate : selected) {
                active.push_back({std::move(candidate.ranked.lookahead), std::move(candidate.stablePrefixLogicalRoute)});
            }
            if (active.empty())
                break;
        }
    }
    stats.generated = stateBudget.generated.load(std::memory_order_relaxed);
    return result;
}

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

FiberletReplayOutgoingArcView FiberletReplayGraphSource::outgoingArcs(
    const FiberletStorageKey& anchor) const
{
    const auto ids = outgoing(anchor);
    std::vector<FiberletReplaySourceArc> arcs;
    arcs.reserve(ids.size());
    for (const auto& id : ids)
        arcs.push_back(arc(id));
    std::sort(arcs.begin(), arcs.end(), [](const auto& left, const auto& right) {
        return left.id < right.id;
    });
    return FiberletReplayOutgoingArcView::owned(std::move(arcs));
}

FiberletReplayCostProfileView FiberletReplayGraphSource::costProfileView(
    const DirectedFiberletStorageId& id) const
{
    return FiberletReplayCostProfileView::owned(costProfile(id));
}

FiberletReplayRoutePointView FiberletReplayGraphSource::routePointView(
    const DirectedFiberletStorageId& id) const
{
    return FiberletReplayRoutePointView::owned(routePoints(id));
}

struct FiberletImmutableReplayGraphSource::Impl {
    struct AnchorData {
        FiberletReplaySourceAnchor anchor;
        std::size_t outgoingBegin = 0;
        std::size_t outgoingCount = 0;
    };

    float predictionToBaseScale = 1.0F;
    int anchorCellSizePredictionVoxels = 0;
    float maximumJoinAngleDegrees = 45.0F;
    std::vector<AnchorData> anchors;
    std::vector<FiberletReplaySourceArc> outgoing;
    std::vector<FiberletImmutableReplayEdge> edges;
    std::vector<size_t> transitionOffsets;
    std::vector<FiberletImmutableReplayTransition> transitions;
    std::vector<size_t> transitionDiagnosticIndices;

    [[nodiscard]] const AnchorData* findAnchor(
        const FiberletStorageKey& id) const
    {
        const auto found = std::lower_bound(
            anchors.begin(), anchors.end(), id,
            [](const AnchorData& value, const FiberletStorageKey& key) {
                return value.anchor.id < key;
            });
        return found != anchors.end() && found->anchor.id == id
            ? &*found
            : nullptr;
    }

    [[nodiscard]] const FiberletImmutableReplayEdge* findEdge(
        const FiberletStorageId& id) const
    {
        const auto found = std::lower_bound(
            edges.begin(), edges.end(), id,
            [](const FiberletImmutableReplayEdge& value,
               const FiberletStorageId& key) {
                return value.arc.id.fiberlet < key;
            });
        return found != edges.end() && found->arc.id.fiberlet == id
            ? &*found
            : nullptr;
    }

    [[nodiscard]] size_t arcIndex(const DirectedFiberletStorageId& id) const
    {
        const auto* found = findEdge(id.fiberlet);
        if (!found)
            throw std::out_of_range("fiberlet replay arc is absent");
        return static_cast<size_t>(found - edges.data()) * 2 +
            static_cast<size_t>(id.reverse);
    }
};

FiberletImmutableReplayGraphSource::FiberletImmutableReplayGraphSource(
    float predictionToBaseScale,
    int anchorCellSizePredictionVoxels,
    float maximumJoinAngleDegrees,
    std::vector<FiberletReplaySourceAnchor> anchors,
    std::vector<FiberletImmutableReplayEdge> edges,
    std::vector<FiberletReplaySourceTransition> transitions)
    : impl_(std::make_unique<Impl>())
{
    if (!(predictionToBaseScale > 0.0F) ||
        !std::isfinite(predictionToBaseScale) ||
        anchorCellSizePredictionVoxels < 1 ||
        !(maximumJoinAngleDegrees >= 0.0F) ||
        !(maximumJoinAngleDegrees <= 180.0F) ||
        !std::isfinite(maximumJoinAngleDegrees)) {
        throw std::invalid_argument(
            "immutable Fiberlet replay graph configuration is invalid");
    }
    impl_->predictionToBaseScale = predictionToBaseScale;
    impl_->anchorCellSizePredictionVoxels = anchorCellSizePredictionVoxels;
    impl_->maximumJoinAngleDegrees = maximumJoinAngleDegrees;
    std::sort(anchors.begin(), anchors.end(), [](const auto& left, const auto& right) {
        return left.id < right.id;
    });
    impl_->anchors.reserve(anchors.size());
    for (auto& anchor : anchors) {
        if (!impl_->anchors.empty() &&
            impl_->anchors.back().anchor.id == anchor.id) {
            throw std::invalid_argument(
                "immutable Fiberlet replay graph contains duplicate anchors");
        }
        impl_->anchors.push_back({std::move(anchor), 0, 0});
    }
    std::sort(edges.begin(), edges.end(), [](const auto& left, const auto& right) {
        return left.arc.id.fiberlet < right.arc.id.fiberlet;
    });
    if (impl_->anchors.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument(
            "immutable Fiberlet replay graph has too many anchors");
    }
    std::vector<std::array<std::uint32_t, 2>> edgeAnchorIndices(
        edges.size());
    std::vector<size_t> outgoingCounts(impl_->anchors.size(), 0);
    for (size_t edgeIndex = 0; edgeIndex < edges.size(); ++edgeIndex) {
        auto& edge = edges[edgeIndex];
        const auto id = edge.arc.id.fiberlet;
        if (edge.arc.id.reverse || edge.arc.source != id.first ||
            edge.arc.target != id.second ||
            edge.routePointsBaseXYZ.size() < 2 ||
            edge.costProfile.segmentLengthsPredictionVoxels.size() !=
                edge.costProfile.segmentCostDensities.size()) {
            throw std::invalid_argument(
                "immutable Fiberlet replay edge is not canonical");
        }
        const auto first = std::lower_bound(
            impl_->anchors.begin(), impl_->anchors.end(), id.first,
            [](const Impl::AnchorData& value, const FiberletStorageKey& key) {
                return value.anchor.id < key;
            });
        const auto second = std::lower_bound(
            impl_->anchors.begin(), impl_->anchors.end(), id.second,
            [](const Impl::AnchorData& value, const FiberletStorageKey& key) {
                return value.anchor.id < key;
            });
        if (first == impl_->anchors.end() || first->anchor.id != id.first ||
            second == impl_->anchors.end() || second->anchor.id != id.second) {
            throw std::invalid_argument(
                "immutable Fiberlet replay edge endpoint is absent");
        }
        if (edgeIndex != 0 &&
            edges[edgeIndex - 1].arc.id.fiberlet == id) {
            throw std::invalid_argument(
                "immutable Fiberlet replay graph contains duplicate edges");
        }
        const size_t firstIndex = static_cast<size_t>(
            first - impl_->anchors.begin());
        const size_t secondIndex = static_cast<size_t>(
            second - impl_->anchors.begin());
        edgeAnchorIndices[edgeIndex] = {
            static_cast<std::uint32_t>(firstIndex),
            static_cast<std::uint32_t>(secondIndex)};
        ++outgoingCounts[firstIndex];
        ++outgoingCounts[secondIndex];
        edge.arc.graphArcIndex = edgeIndex * 2;
    }
    size_t outgoingOffset = 0;
    for (size_t anchor = 0; anchor < impl_->anchors.size(); ++anchor) {
        impl_->anchors[anchor].outgoingBegin = outgoingOffset;
        impl_->anchors[anchor].outgoingCount = outgoingCounts[anchor];
        outgoingOffset += outgoingCounts[anchor];
    }
    impl_->outgoing.resize(outgoingOffset);
    std::vector<size_t> outgoingCursors(impl_->anchors.size(), 0);
    for (size_t anchor = 0; anchor < impl_->anchors.size(); ++anchor)
        outgoingCursors[anchor] = impl_->anchors[anchor].outgoingBegin;
    for (size_t edgeIndex = 0; edgeIndex < edges.size(); ++edgeIndex) {
        auto forward = edges[edgeIndex].arc;
        auto reverse = forward;
        reverse.id.reverse = true;
        reverse.graphArcIndex = edgeIndex * 2 + 1;
        std::swap(reverse.source, reverse.target);
        std::swap(
            reverse.sourcePositionBaseXYZ,
            reverse.targetPositionBaseXYZ);
        const auto start = reverse.startStepBaseXYZ;
        reverse.startStepBaseXYZ = -reverse.endStepBaseXYZ;
        reverse.endStepBaseXYZ = -start;
        const auto [firstIndex, secondIndex] = edgeAnchorIndices[edgeIndex];
        impl_->outgoing[outgoingCursors[firstIndex]++] = std::move(forward);
        impl_->outgoing[outgoingCursors[secondIndex]++] = std::move(reverse);
    }
    for (const auto& anchor : impl_->anchors) {
        auto begin = impl_->outgoing.begin() +
            static_cast<std::ptrdiff_t>(anchor.outgoingBegin);
        std::sort(
            begin,
            begin + static_cast<std::ptrdiff_t>(anchor.outgoingCount),
            [](const auto& left, const auto& right) {
                return left.id < right.id;
            });
    }
    impl_->edges = std::move(edges);
    struct IndexedTransition {
        std::uint32_t incoming = 0;
        std::uint32_t outgoing = 0;
        FiberletPathCost cost;
        std::optional<size_t> diagnosticIndex;
    };
    if (impl_->edges.size() * 2 >
        std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument(
            "immutable Fiberlet replay graph has too many directed arcs");
    }
    std::vector<IndexedTransition> indexedTransitions;
    indexedTransitions.reserve(transitions.size());
    for (const auto& transition : transitions) {
        validateReplayCost(
            transition.cost,
            "immutable Fiberlet replay transition cost is invalid");
        indexedTransitions.push_back({
            static_cast<std::uint32_t>(impl_->arcIndex(transition.incoming)),
            static_cast<std::uint32_t>(impl_->arcIndex(transition.outgoing)),
            transition.cost,
            transition.diagnosticTransitionIndex,
        });
    }
    std::sort(
        indexedTransitions.begin(), indexedTransitions.end(),
        [](const auto& left, const auto& right) {
            return std::pair{left.incoming, left.outgoing} <
                std::pair{right.incoming, right.outgoing};
        });
    impl_->transitionOffsets.assign(impl_->edges.size() * 2 + 1, 0);
    impl_->transitions.reserve(indexedTransitions.size());
    const size_t noDiagnostic = std::numeric_limits<size_t>::max();
    bool hasDiagnostics = false;
    std::optional<std::pair<std::uint32_t, std::uint32_t>> previousKey;
    for (const auto& transition : indexedTransitions) {
        const auto key = std::pair{transition.incoming, transition.outgoing};
        if (previousKey == key) {
            throw std::invalid_argument(
                "immutable Fiberlet replay graph contains duplicate transitions");
        }
        previousKey = key;
        ++impl_->transitionOffsets[transition.incoming + 1];
        impl_->transitions.push_back({transition.outgoing, transition.cost});
        impl_->transitionDiagnosticIndices.push_back(
            transition.diagnosticIndex.value_or(noDiagnostic));
        hasDiagnostics = hasDiagnostics || transition.diagnosticIndex.has_value();
    }
    std::partial_sum(
        impl_->transitionOffsets.begin(), impl_->transitionOffsets.end(),
        impl_->transitionOffsets.begin());
    if (!hasDiagnostics)
        impl_->transitionDiagnosticIndices.clear();
}

FiberletImmutableReplayGraphSource::FiberletImmutableReplayGraphSource(
    float predictionToBaseScale,
    int anchorCellSizePredictionVoxels,
    float maximumJoinAngleDegrees,
    std::vector<FiberletReplaySourceAnchor> anchors,
    std::vector<FiberletImmutableReplayEdge> edges,
    FiberletImmutableReplayTransitionCsr transitions)
    : FiberletImmutableReplayGraphSource(
          predictionToBaseScale,
          anchorCellSizePredictionVoxels,
          maximumJoinAngleDegrees,
          std::move(anchors),
          std::move(edges),
          std::vector<FiberletReplaySourceTransition>{})
{
    const size_t directedArcCount = impl_->edges.size() * 2;
    if (transitions.offsets.size() != directedArcCount + 1 ||
        transitions.offsets.front() != 0 ||
        transitions.offsets.back() != transitions.entries.size()) {
        throw std::invalid_argument(
            "immutable Fiberlet replay transition CSR shape is invalid");
    }
    for (size_t incoming = 0; incoming < directedArcCount; ++incoming) {
        const size_t begin = transitions.offsets[incoming];
        const size_t end = transitions.offsets[incoming + 1];
        if (begin > end || end > transitions.entries.size()) {
            throw std::invalid_argument(
                "immutable Fiberlet replay transition CSR offsets are invalid");
        }
        std::uint32_t previousOutgoing = 0;
        bool first = true;
        for (size_t index = begin; index < end; ++index) {
            const auto& transition = transitions.entries[index];
            if (transition.outgoingArc >= directedArcCount ||
                (!first && transition.outgoingArc <= previousOutgoing)) {
                throw std::invalid_argument(
                    "immutable Fiberlet replay transition CSR entries are invalid");
            }
            validateReplayCost(
                transition.cost,
                "immutable Fiberlet replay transition cost is invalid");
            previousOutgoing = transition.outgoingArc;
            first = false;
        }
    }
    impl_->transitionOffsets = std::move(transitions.offsets);
    impl_->transitions = std::move(transitions.entries);
    impl_->transitionDiagnosticIndices.clear();
}

FiberletImmutableReplayGraphSource::FiberletImmutableReplayGraphSource(
    const FiberletGraph& graph)
    : impl_(nullptr)
{
    std::vector<FiberletReplaySourceAnchor> anchors;
    anchors.reserve(graph.nodes.size());
    for (const auto& node : graph.nodes) {
        anchors.push_back({storageKey(node.anchor), node.positionBaseXYZ});
    }
    const auto stableArc = [&](std::size_t numericArc) {
        const auto& edge = graph.edges.at(arcEdge(numericArc));
        const auto start = storageKey(graph.nodes.at(edge.startNode).anchor);
        const auto target = storageKey(graph.nodes.at(edge.targetNode).anchor);
        const FiberletStorageId id{
            std::min(start, target), std::max(start, target)};
        const auto& source = arcForward(numericArc) ? start : target;
        return DirectedFiberletStorageId{id, source != id.first};
    };
    std::vector<FiberletImmutableReplayEdge> edges;
    edges.reserve(graph.edges.size());
    for (std::size_t index = 0; index < graph.edges.size(); ++index) {
        const auto& edge = graph.edges[index];
        const auto start = storageKey(graph.nodes.at(edge.startNode).anchor);
        const auto target = storageKey(graph.nodes.at(edge.targetNode).anchor);
        const FiberletStorageId id{
            std::min(start, target), std::max(start, target)};
        const std::size_t numericArc =
            index * 2 + static_cast<std::size_t>(start != id.first);
        auto points = orientedArcPoints(graph, numericArc);
        auto lengths = edge.segmentLengthsPredictionVoxels;
        auto densities = edge.segmentCostDensities;
        for (float& density : densities) {
            density = decodeFiberletStoredCostDensity(
                encodeFiberletStoredCostDensity(density));
        }
        if (start != id.first) {
            std::reverse(lengths.begin(), lengths.end());
            std::reverse(densities.begin(), densities.end());
        }
        FiberletReplaySourceArc arc;
        arc.id = {id, false};
        arc.source = id.first;
        arc.target = id.second;
        arc.sourcePositionBaseXYZ = points.front();
        arc.targetPositionBaseXYZ = points.back();
        arc.startStepBaseXYZ = cv::Vec3f(points[1] - points[0]);
        arc.endStepBaseXYZ =
            cv::Vec3f(points.back() - points[points.size() - 2]);
        arc.pathLengthPredictionVoxels = edge.pathLengthPredictionVoxels;
        arc.cost = edge.cost;
        arc.diagnosticCandidateIndex = edge.candidateIndex;
        arc.diagnosticArcIndex = numericArc;
        edges.push_back({
            std::move(arc), {std::move(lengths), std::move(densities)},
            std::move(points)});
    }
    std::vector<FiberletReplaySourceTransition> transitions;
    transitions.reserve(graph.transitions.size());
    for (std::size_t index = 0; index < graph.transitions.size(); ++index) {
        const auto& transition = graph.transitions[index];
        transitions.push_back({
            stableArc(transition.incomingArc),
            stableArc(transition.outgoingArc), transition.cost, index});
    }
    *this = FiberletImmutableReplayGraphSource(
        graph.predictionToBaseScale,
        graph.anchorCellSizePredictionVoxels,
        graph.maximumJoinAngleDegrees,
        std::move(anchors), std::move(edges), std::move(transitions));
}

FiberletImmutableReplayGraphSource::~FiberletImmutableReplayGraphSource() =
    default;
FiberletImmutableReplayGraphSource::FiberletImmutableReplayGraphSource(
    FiberletImmutableReplayGraphSource&&) noexcept = default;
FiberletImmutableReplayGraphSource&
FiberletImmutableReplayGraphSource::operator=(
    FiberletImmutableReplayGraphSource&&) noexcept = default;

float FiberletImmutableReplayGraphSource::predictionToBaseScale() const noexcept
{
    return impl_->predictionToBaseScale;
}

int FiberletImmutableReplayGraphSource::anchorCellSizePredictionVoxels() const
    noexcept
{
    return impl_->anchorCellSizePredictionVoxels;
}

float FiberletImmutableReplayGraphSource::maximumJoinAngleDegrees() const
    noexcept
{
    return impl_->maximumJoinAngleDegrees;
}

std::vector<FiberletReplaySourceAnchor>
FiberletImmutableReplayGraphSource::anchorsNearReference(
    const PolylineArcGeometry& reference,
    double beginArcBase,
    double endArcBase,
    double broadPhaseRadiusBaseVoxels) const
{
    std::vector<FiberletReplaySourceAnchor> result;
    for (const auto& stored : impl_->anchors) {
        const auto& anchor = stored.anchor;
        const auto projection = projectPointToPolylineArc(
            reference, anchor.positionBaseXYZ, beginArcBase, endArcBase);
        if (projection.arc + kReplayEpsilon < beginArcBase ||
            projection.arc > endArcBase + kReplayEpsilon ||
            projection.distance > broadPhaseRadiusBaseVoxels) {
            continue;
        }
        result.push_back(anchor);
    }
    return result;
}

std::vector<DirectedFiberletStorageId>
FiberletImmutableReplayGraphSource::outgoing(
    const FiberletStorageKey& anchor) const
{
    const auto view = outgoingArcs(anchor);
    std::vector<DirectedFiberletStorageId> result;
    result.reserve(view.size());
    for (std::size_t index = 0; index < view.size(); ++index)
        result.push_back(view[index].id);
    return result;
}

FiberletReplayOutgoingArcView
FiberletImmutableReplayGraphSource::outgoingArcs(
    const FiberletStorageKey& anchor) const
{
    const auto* found = impl_->findAnchor(anchor);
    if (!found)
        throw std::out_of_range("fiberlet replay anchor is absent");
    return {std::span<const FiberletReplaySourceArc>(impl_->outgoing)
                .subspan(found->outgoingBegin, found->outgoingCount)};
}

FiberletReplaySourceArc FiberletImmutableReplayGraphSource::arc(
    const DirectedFiberletStorageId& id) const
{
    const auto* found = impl_->findEdge(id.fiberlet);
    if (!found)
        throw std::out_of_range("fiberlet replay arc is absent");
    auto result = found->arc;
    result.id = id;
    result.graphArcIndex =
        static_cast<size_t>(found - impl_->edges.data()) * 2 +
        static_cast<size_t>(id.reverse);
    if (id.reverse) {
        std::swap(result.source, result.target);
        std::swap(
            result.sourcePositionBaseXYZ, result.targetPositionBaseXYZ);
        const auto start = result.startStepBaseXYZ;
        result.startStepBaseXYZ = -result.endStepBaseXYZ;
        result.endStepBaseXYZ = -start;
    }
    return result;
}

FiberletReplaySourceCostProfile
FiberletImmutableReplayGraphSource::costProfile(
    const DirectedFiberletStorageId& id) const
{
    const auto view = costProfileView(id);
    FiberletReplaySourceCostProfile result;
    result.segmentLengthsPredictionVoxels.reserve(
        view.segmentLengthsPredictionVoxels.size());
    result.segmentCostDensities.reserve(
        view.segmentCostDensities.size());
    for (std::size_t index = 0;
         index < view.segmentLengthsPredictionVoxels.size(); ++index) {
        result.segmentLengthsPredictionVoxels.push_back(
            view.segmentLengthsPredictionVoxels[index]);
        result.segmentCostDensities.push_back(
            view.segmentCostDensities[index]);
    }
    return result;
}

FiberletReplayCostProfileView
FiberletImmutableReplayGraphSource::costProfileView(
    const DirectedFiberletStorageId& id) const
{
    const auto* found = impl_->findEdge(id.fiberlet);
    if (!found) {
        throw std::out_of_range(
            "fiberlet replay cost-profile arc is absent");
    }
    return {found->costProfile.segmentLengthsPredictionVoxels,
            found->costProfile.segmentCostDensities, {}, id.reverse};
}

std::vector<cv::Vec3d> FiberletImmutableReplayGraphSource::routePoints(
    const DirectedFiberletStorageId& id) const
{
    const auto view = routePointView(id);
    std::vector<cv::Vec3d> result;
    result.reserve(view.size());
    for (std::size_t index = 0; index < view.size(); ++index)
        result.push_back(view[index]);
    return result;
}

FiberletReplayRoutePointView
FiberletImmutableReplayGraphSource::routePointView(
    const DirectedFiberletStorageId& id) const
{
    const auto* found = impl_->findEdge(id.fiberlet);
    if (!found)
        throw std::out_of_range("fiberlet replay route is absent");
    return {found->routePointsBaseXYZ, {}, id.reverse};
}

std::optional<FiberletReplaySourceTransition>
FiberletImmutableReplayGraphSource::transition(
    const FiberletReplaySourceArc& incoming,
    const FiberletReplaySourceArc& outgoing) const
{
    const auto resolveArcIndex = [&](const FiberletReplaySourceArc& arc) {
        if (arc.graphArcIndex < impl_->edges.size() * 2) {
            const auto& stored =
                impl_->edges[arc.graphArcIndex / 2].arc.id.fiberlet;
            if (stored == arc.id.fiberlet &&
                static_cast<bool>(arc.graphArcIndex % 2) == arc.id.reverse) {
                return arc.graphArcIndex;
            }
        }
        return impl_->arcIndex(arc.id);
    };
    const size_t incomingIndex = resolveArcIndex(incoming);
    const size_t outgoingIndex = resolveArcIndex(outgoing);
    const size_t begin = impl_->transitionOffsets.at(incomingIndex);
    const size_t end = impl_->transitionOffsets.at(incomingIndex + 1);
    const auto found = std::lower_bound(
        impl_->transitions.begin() + static_cast<std::ptrdiff_t>(begin),
        impl_->transitions.begin() + static_cast<std::ptrdiff_t>(end),
        outgoingIndex,
        [](const FiberletImmutableReplayTransition& value,
           size_t expected) {
            return value.outgoingArc < expected;
        });
    if (found == impl_->transitions.begin() +
                     static_cast<std::ptrdiff_t>(end) ||
        found->outgoingArc != outgoingIndex) {
        return std::nullopt;
    }
    std::optional<size_t> diagnosticIndex;
    if (!impl_->transitionDiagnosticIndices.empty()) {
        const size_t compactIndex = static_cast<size_t>(
            found - impl_->transitions.begin());
        const size_t stored =
            impl_->transitionDiagnosticIndices[compactIndex];
        if (stored != std::numeric_limits<size_t>::max())
            diagnosticIndex = stored;
    }
    return FiberletReplaySourceTransition{
        incoming.id, outgoing.id, found->cost, diagnosticIndex};
}

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
        if (candidate.segmentCosts.size() + 1 != candidate.pointsPredictionXYZ.size())
            throw std::invalid_argument("successful fiberlet segment costs differ from its graph geometry");
        edge.segmentLengthsPredictionVoxels.reserve(candidate.segmentCosts.size());
        edge.segmentCostDensities.reserve(candidate.segmentCosts.size());
        for (size_t segment = 0; segment < candidate.segmentCosts.size(); ++segment) {
            const float segmentLength = length(
                candidate.pointsPredictionXYZ[segment + 1] -
                candidate.pointsPredictionXYZ[segment]);
            if (!(segmentLength > kFloatEpsilon) || !std::isfinite(segmentLength))
                throw std::invalid_argument("successful fiberlet segment length is invalid");
            const float density = static_cast<float>(
                candidate.segmentCosts[segment].total() /
                segmentLength);
            if (!(density >= 0.0F) || !std::isfinite(density))
                throw std::invalid_argument("successful fiberlet segment cost density is invalid");
            edge.segmentLengthsPredictionVoxels.push_back(segmentLength);
            edge.segmentCostDensities.push_back(density);
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
    const FiberletImmutableReplayGraphSource source(graph);
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
    const bool validCostMode =
        config.costMode == FiberletGraphReplayCostMode::Fiberlet ||
        config.costMode == FiberletGraphReplayCostMode::Stepped;
    const bool inactiveSteppedSettingsChanged =
        config.costMode == FiberletGraphReplayCostMode::Fiberlet &&
        (config.geometricCostWeightPerBaseVoxel != 1.0 ||
         config.geometricCostDelayBaseVoxels != 0.0 ||
         config.costIntegrationStepBaseVoxels != 16.0 ||
         config.costProfileWeight != 1.0);
    if (!validCostMode)
        throw std::invalid_argument("fiberlet graph replay cost mode is invalid");
    if (inactiveSteppedSettingsChanged) {
        throw std::invalid_argument(
            "stepped replay cost settings require stepped cost mode");
    }
    if (!(predictionToBaseScale > 0.0) || !std::isfinite(predictionToBaseScale) || config.beamWidth < 1 ||
        config.expansionThreads < 1 || config.maximumGeneratedStatesPerIteration == 0 || !(config.beamStepDistanceBaseVoxels > 0.0) ||
        !std::isfinite(config.beamStepDistanceBaseVoxels) || !(config.lookaheadDistanceBaseVoxels > 0.0) ||
        !std::isfinite(config.lookaheadDistanceBaseVoxels) ||
        !(config.geometricCostWeightPerBaseVoxel > 0.0) ||
        config.geometricCostWeightPerBaseVoxel > 1.0 ||
        !std::isfinite(config.geometricCostWeightPerBaseVoxel) ||
        !(config.geometricCostDelayBaseVoxels >= 0.0) ||
        !std::isfinite(config.geometricCostDelayBaseVoxels) ||
        !(config.costIntegrationStepBaseVoxels > 0.0) ||
        !std::isfinite(config.costIntegrationStepBaseVoxels) ||
        !(config.costProfileWeight >= 0.0) ||
        config.costProfileWeight > 1.0 ||
        !std::isfinite(config.costProfileWeight) ||
        (config.searchWidth != 0 && config.searchWidth < config.beamWidth) ||
        !(config.pruneDistanceBaseVoxels > 0.0) || !std::isfinite(config.pruneDistanceBaseVoxels) || !(config.errorThresholdBaseVoxels >= 0.0) ||
        !std::isfinite(config.errorThresholdBaseVoxels) || !(config.matchRefineSteps >= 0.0) || !std::isfinite(config.matchRefineSteps) ||
        !(config.minimumResetAdvanceBaseVoxels > 0.0) || !std::isfinite(config.minimumResetAdvanceBaseVoxels) ||
        !(config.referenceBeginArcBase >= 0.0) || !std::isfinite(config.referenceBeginArcBase) || !(normalWorkingToBaseScale > 0.0) ||
        !std::isfinite(normalWorkingToBaseScale) || (config.referenceEndArcBase.has_value() && !std::isfinite(*config.referenceEndArcBase)) ||
        std::any_of(config.decisionDiagnosticReferenceArcWindowsBase.begin(), config.decisionDiagnosticReferenceArcWindowsBase.end(), [](const auto& window) {
            return !(window.first >= 0.0) || !(window.second >= window.first) || !std::isfinite(window.first) || !std::isfinite(window.second);
        })) {
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
    const auto selectSeed = [&](double resetArc, std::optional<FiberletStorageKey> forcedKey, bool firstWindowOnly) -> std::optional<Seed> {
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
            if (forcedKey.has_value() || firstWindowOnly)
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
    std::unique_ptr<utils::ThreadPool> exactExpansionPool;
    if (config.searchWidth == 0)
        exactExpansionPool =
            std::make_unique<utils::ThreadPool>(config.expansionThreads);

    double resetArc = config.referenceBeginArcBase;
    for (size_t iteration = 0; iteration < maximumSegments && resetArc < referenceEndArcBase - kReplayEpsilon; ++iteration) {
        emitProgress(result.segments.size(), resetArc, "segment_start");
        const auto seed = selectSeed(
            resetArc,
            iteration == 0 ? config.initialSeedKey : std::nullopt,
            iteration == 0 && config.requireInitialSeedInFirstWindow);
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
            result.completedReferenceArcBase = config.stopAtFirstFailure
                ? resetArc
                : referenceEndArcBase;
            emitProgress(
                result.segments.size() - 1,
                result.completedReferenceArcBase,
                config.stopAtFirstFailure ? "failed" : "completed");
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
            if (config.stopAtFirstFailure) {
                result.completedReferenceArcBase = resetArc;
                emitProgress(segmentIndex, resetArc, "failed");
                break;
            }
        }

        if (config.stopAtFirstFailure && !result.failures.empty())
            break;

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
        const GeometricReplayCostConfig geometricCost{
            config.costMode,
            predictionToBaseScale,
            config.geometricCostWeightPerBaseVoxel,
            config.geometricCostDelayBaseVoxels / predictionToBaseScale,
            config.costIntegrationStepBaseVoxels / predictionToBaseScale,
            config.costProfileWeight};
        PersistentLogicalRouteRegistry logicalRoutes;
        PersistentRouteCandidate initialBeam;
        initialBeam.seed = seed->node.id;
        initialBeam.visitedNodes = persistentVisitedAdd(nullptr, seed->node.id);
        initialBeam.logicalRoute = logicalRoutes.root();
        auto rootEvaluation = std::make_shared<PersistentRouteEvaluation>();
        rootEvaluation->appendedPointsBaseXYZ.emplace_back(seed->node.positionBaseXYZ);
        rootEvaluation->appendedMatches.push_back(
            {0, seed->projection.arc, seed->projection.arc, seed->projection.point, resetArc, seed->projection.arc, seed->thresholdMeasurement});
        rootEvaluation->matchedReferenceArcBase = seed->projection.arc;
        rootEvaluation->routePointCount = 1;
        rootEvaluation->lastPointBaseXYZ = cv::Vec3d(seed->node.positionBaseXYZ);
        initialBeam.evaluation = rootEvaluation;
        std::vector<PersistentRouteCandidate> beams{initialBeam};
        std::vector<FiberletGraphReplayDecision> decisions;
        PersistentRouteCandidate selectedRoute = initialBeam;
        std::optional<FiberReplayFailure> distanceFailure;
        bool referenceExhausted = maximumRouteLengthPredictionVoxels <= kReplayEpsilon;
        double previousReferenceArc = seed->projection.arc;
        std::set<FiberletStorageKey> selectedTraversedNodes{seed->node.id};
        const cv::Vec3d initialDirection = samplePolylineArc(reference, seed->projection.arc).tangent;
        const size_t maximumGeneratedStates = config.maximumGeneratedStatesPerIteration;

        const auto addReplayCost = [](FiberletGraphReplayCost& target, const FiberletGraphReplayCost& source) {
            target.invalidPrediction += source.invalidPrediction;
            target.alignment += source.alignment;
            target.isotropicSmoothness += source.isotropicSmoothness;
            target.tangentSmoothness += source.tangentSmoothness;
            target.normalSmoothness += source.normalSmoothness;
        };
        std::map<const PersistentRouteHistory*, std::shared_ptr<const PersistentRouteEvaluation>> evaluatedHistory;
        const auto evaluateRoute = [&](PersistentRouteCandidate& route) {
            auto evaluated = route.evaluation;
            if (!evaluated)
                throw std::logic_error("persistent beam evaluator continuation is missing");
            std::vector<std::shared_ptr<const PersistentRouteHistory>> missing;
            for (auto history = route.tail; history != nullptr; history = history->parent) {
                if (const auto found = evaluatedHistory.find(history.get()); found != evaluatedHistory.end()) {
                    evaluated = found->second;
                    break;
                }
                if (history == evaluated->history)
                    break;
                missing.push_back(history);
            }
            if (!missing.empty() && missing.back()->parent != evaluated->history)
                throw std::logic_error("persistent beam evaluator is not an ancestor of the route");
            std::reverse(missing.begin(), missing.end());
            for (const auto& historyNode : missing) {
                if (evaluated->failure.has_value() || evaluated->reachedReferenceEnd)
                    break;
                const auto arcId = historyNode->arc;
                const auto edge = graph.arc(arcId);
                const auto fullPoints = graph.routePoints(arcId);
                const double remainingPathLength = std::max(0.0, maximumRouteLengthPredictionVoxels - evaluated->cumulativePathLengthPredictionVoxels);
                const double includedArcFraction = std::clamp(remainingPathLength / static_cast<double>(edge.pathLengthPredictionVoxels), 0.0, 1.0);
                const auto points = routePointPrefix(fullPoints, includedArcFraction);
                if (points.size() < 2)
                    throw std::logic_error("persistent beam route is too short");
                double fullGeometryLengthBase = 0.0;
                for (size_t pointIndex = 1; pointIndex < fullPoints.size(); ++pointIndex)
                    fullGeometryLengthBase += length(fullPoints[pointIndex] - fullPoints[pointIndex - 1]);
                if (!(fullGeometryLengthBase > kReplayEpsilon))
                    throw std::logic_error("persistent beam route has zero geometry length");

                auto next = std::make_shared<PersistentRouteEvaluation>();
                next->parent = evaluated;
                next->history = historyNode;
                next->cumulativeEdgeCost = evaluated->cumulativeEdgeCost;
                next->cumulativeTransitionCost = evaluated->cumulativeTransitionCost;
                next->cumulativePathLengthPredictionVoxels = evaluated->cumulativePathLengthPredictionVoxels;
                next->matchedReferenceArcBase = evaluated->matchedReferenceArcBase;
                next->routePointCount = evaluated->routePointCount;
                next->lastPointBaseXYZ = evaluated->lastPointBaseXYZ;

                FiberletGraphReplayCommittedStep step;
                step.referenceBeginArcBase = next->matchedReferenceArcBase;
                if (historyNode->enteringTransition.has_value()) {
                    const auto& join = *historyNode->enteringTransition;
                    step.transitionCost += join.cost;
                    next->cumulativeTransitionCost += join.cost;
                }
                double traversedGeometryLengthBase = 0.0;
                for (size_t pointIndex = 1; pointIndex < points.size(); ++pointIndex) {
                    const double stepBase = length(points[pointIndex] - next->lastPointBaseXYZ);
                    traversedGeometryLengthBase += stepBase;
                    const auto forwardMatch =
                        matchForwardPolylinePoint(reference, points[pointIndex], next->matchedReferenceArcBase, stepBase, config.matchRefineSteps, referenceEndArcBase);
                    const auto thresholdMeasurement = measureFiberReplayThreshold(
                        points[pointIndex], forwardMatch.projection.point, normalSampler, normalWorkingToBaseScale, config.errorThresholdBaseVoxels);
                    next->appendedPointsBaseXYZ.push_back(points[pointIndex]);
                    next->appendedMatches.push_back({
                        next->routePointCount,
                        forwardMatch.predictedArc,
                        forwardMatch.projection.arc,
                        forwardMatch.projection.point,
                        next->matchedReferenceArcBase,
                        forwardMatch.searchEndArc,
                        thresholdMeasurement,
                    });
                    ++next->routePointCount;
                    next->lastPointBaseXYZ = points[pointIndex];
                    next->matchedReferenceArcBase = forwardMatch.projection.arc;
                    if (!next->failure.has_value() && fiberReplayThresholdExceeded(thresholdMeasurement, config.errorThresholdBaseVoxels)) {
                        FiberReplayFailure event;
                        event.segmentIndex = result.segments.size();
                        event.reason = "distance_above_threshold";
                        event.referenceArcBase = next->matchedReferenceArcBase;
                        event.evaluatorPointBase = points[pointIndex];
                        event.segmentPointIndex = next->routePointCount - 1;
                        event.thresholdMeasurement = thresholdMeasurement;
                        next->failure = std::move(event);
                        next->failureCandidatePathPointIndex = pointIndex;
                    }
                    if (next->matchedReferenceArcBase >= referenceEndArcBase - kReplayEpsilon) {
                        next->reachedReferenceEnd = true;
                        break;
                    }
                }
                const double traversedFraction = std::clamp(traversedGeometryLengthBase / fullGeometryLengthBase, 0.0, includedArcFraction);
                addScaledCost(step.edgeCost, edge.cost, traversedFraction);
                step.pathLengthPredictionVoxels = traversedFraction * edge.pathLengthPredictionVoxels;
                addReplayCost(next->cumulativeEdgeCost, step.edgeCost);
                next->cumulativePathLengthPredictionVoxels += step.pathLengthPredictionVoxels;
                next->traversedFullEdge = traversedFraction >= 1.0 - kReplayEpsilon;
                step.referenceEndArcBase = next->matchedReferenceArcBase;
                next->committedStep = std::move(step);
                if (includedArcFraction < 1.0 - kReplayEpsilon)
                    next->reachedReferenceEnd = true;
                evaluated = std::move(next);
                evaluatedHistory.insert_or_assign(historyNode.get(), evaluated);
            }
            route.evaluation = evaluated;
            return evaluated;
        };

        const auto materializeSelected = [&](const std::shared_ptr<const PersistentRouteEvaluation>& terminal) {
            if (!terminal)
                throw std::logic_error("persistent beam terminal evaluation is missing");
            FiberletGraphReplaySegment built;
            std::set<FiberletStorageKey> traversedNodes{seed->node.id};
            built.seedKey = seed->node.id;
            built.startReferenceArcBase = seed->projection.arc;
            built.endReferenceArcBase = terminal->matchedReferenceArcBase;
            std::vector<const PersistentRouteEvaluation*> contributions;
            for (auto current = terminal.get(); current != nullptr; current = current->parent.get())
                contributions.push_back(current);
            std::reverse(contributions.begin(), contributions.end());
            std::optional<FiberReplayFailure> failure = terminal->failure;
            for (const auto* contribution : contributions) {
                built.routePointsBaseXYZ
                    .insert(built.routePointsBaseXYZ.end(), contribution->appendedPointsBaseXYZ.begin(), contribution->appendedPointsBaseXYZ.end());
                built.matches.insert(built.matches.end(), contribution->appendedMatches.begin(), contribution->appendedMatches.end());
                if (contribution->history != nullptr) {
                    const auto& arcId = contribution->history->arc;
                    const auto edge = graph.arc(arcId);
                    const size_t candidateIndex = edge.diagnosticCandidateIndex.value_or(stableIndex(candidateIndices, arcId.fiberlet));
                    const size_t arcIndex = edge.diagnosticArcIndex.value_or(stableIndex(arcIndices, arcId));
                    built.candidateIndices.push_back(candidateIndex);
                    built.arcIndices.push_back(arcIndex);
                    if (contribution->history->enteringTransition.has_value()) {
                        const auto& join = *contribution->history->enteringTransition;
                        built.transitionIndices.push_back(
                            join.diagnosticTransitionIndex.value_or(stableIndex(transitionIndices, std::pair{join.incoming, join.outgoing})));
                    }
                    if (failure.has_value() && contribution->failure.has_value() && contribution->failure->segmentPointIndex == failure->segmentPointIndex) {
                        failure->candidateIndex = candidateIndex;
                        failure->arcIndex = arcIndex;
                        failure->candidatePathPointIndex = contribution->failureCandidatePathPointIndex;
                    }
                }
                if (contribution->committedStep.has_value())
                    built.committedSteps.push_back(*contribution->committedStep);
                if (contribution->history != nullptr && contribution->traversedFullEdge)
                    traversedNodes.insert(targetAnchor(contribution->history->arc));
            }
            built.edgeCost = terminal->cumulativeEdgeCost;
            built.transitionCost = terminal->cumulativeTransitionCost;
            built.pathLengthPredictionVoxels = terminal->cumulativePathLengthPredictionVoxels;
            built.totalLoss = built.edgeCost.total() + built.transitionCost.total();
            built.terminalPartialEdge = terminal->history != nullptr && !terminal->traversedFullEdge;
            if (!built.terminalPartialEdge && terminal->history != nullptr)
                built.stopNodeIndex = stableIndex(nodeIndices, targetAnchor(terminal->history->arc));
            return std::tuple{std::move(built), std::move(failure), terminal->reachedReferenceEnd, std::move(traversedNodes)};
        };
        std::set<const PersistentRouteHistory*> indexedSelectedHistory;
        const auto indexSelectedRouteSuffix = [&](const std::shared_ptr<const PersistentRouteEvaluation>& terminal) {
            std::vector<const PersistentRouteEvaluation*> unindexed;
            for (auto current = terminal.get(); current != nullptr && current->history != nullptr; current = current->parent.get()) {
                if (indexedSelectedHistory.contains(current->history.get()))
                    break;
                unindexed.push_back(current);
            }
            std::reverse(unindexed.begin(), unindexed.end());
            for (const auto* contribution : unindexed) {
                const auto& history = *contribution->history;
                const auto edge = graph.arc(history.arc);
                (void)edge.diagnosticCandidateIndex.value_or(stableIndex(candidateIndices, history.arc.fiberlet));
                (void)edge.diagnosticArcIndex.value_or(stableIndex(arcIndices, history.arc));
                if (history.enteringTransition.has_value()) {
                    const auto& join = *history.enteringTransition;
                    (void)join.diagnosticTransitionIndex.value_or(stableIndex(transitionIndices, std::pair{join.incoming, join.outgoing}));
                }
                indexedSelectedHistory.insert(contribution->history.get());
            }
            if (terminal->history != nullptr && terminal->traversedFullEdge)
                (void)stableIndex(nodeIndices, targetAnchor(terminal->history->arc));
        };

        double checkpointPredictionVoxels = 0.0;
        for (size_t beamIteration = 0; !distanceFailure.has_value() && !referenceExhausted; ++beamIteration) {
            const double currentCheckpoint = checkpointPredictionVoxels;
            const double scoringHorizon = std::min(maximumRouteLengthPredictionVoxels, currentCheckpoint + lookaheadPredictionVoxels);
            const double nextCheckpoint = std::min(maximumRouteLengthPredictionVoxels, currentCheckpoint + beamStepPredictionVoxels);
            ExactPersistentSearchStats searchStats;
            std::vector<RankedPersistentPrefix> ranked;
            std::vector<FiberletGraphReplayPruneFront> pruneFronts;
            std::shared_ptr<const PersistentLogicalRouteNode> selectedPrefixLogicalRoute;
            if (config.searchWidth == 0) {
                ranked = exactPersistentRouteLookahead(
                    graph,
                    logicalRoutes,
                    beams,
                    currentCheckpoint,
                    scoringHorizon,
                    nextCheckpoint,
                    initialDirection,
                    config.beamWidth,
                    *exactExpansionPool,
                    maximumGeneratedStates,
                    geometricCost,
                    searchStats);
                if (!ranked.empty()) {
                    selectedPrefixLogicalRoute = ranked.front().prefix.logicalRoute;
                }
            } else {
                auto bounded = boundedPersistentRouteLookahead(
                    graph,
                    logicalRoutes,
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
                    geometricCost,
                    searchStats);
                ranked = std::move(bounded.ranked);
                pruneFronts = std::move(bounded.fronts);
                selectedPrefixLogicalRoute = std::move(bounded.selectedPrefixLogicalRoute);
            }
            searchStats.logicalRouteInternCount =
                logicalRoutes.internedCount();
            if (ranked.empty())
                break;
            const std::optional<size_t> rolloutExpandedStateCount =
                config.searchWidth == 0 ? std::nullopt : std::optional<size_t>{searchStats.expanded};
            const std::optional<double> minimumAppliedLocalPruneLossCutoffPerPredictionVoxel =
                config.searchWidth == 0 || pruneFronts.empty() ? std::nullopt : pruneFronts.back().minimumAppliedLocalCompletionLossCutoffPerPredictionVoxel;
            beams.clear();
            beams.reserve(ranked.size());
            selectedRoute = ranked.front().prefix;
            for (const auto& entry : ranked)
                beams.push_back(config.searchWidth == 0 ? entry.lookahead : entry.prefix);
            searchStats.logicalRouteCleanupVisitedCount =
                logicalRoutes.pruneExpired();
            checkpointPredictionVoxels = nextCheckpoint;
            if (std::any_of(beams.begin(), beams.end(), [&](const auto& beam) {
                    return beam.pathLength + kReplayEpsilon < checkpointPredictionVoxels;
                })) {
                throw std::logic_error("persistent beam checkpoint exceeds a committed route");
            }
            evaluateRoute(selectedRoute);

            const auto& selectedEvaluation = selectedRoute.evaluation;
            indexSelectedRouteSuffix(selectedEvaluation);
            previousReferenceArc = selectedEvaluation->matchedReferenceArcBase;
            const bool inDecisionDiagnosticWindow = config.decisionDiagnosticReferenceArcWindowsBase.empty() ||
                std::any_of(
                    config.decisionDiagnosticReferenceArcWindowsBase.begin(),
                    config.decisionDiagnosticReferenceArcWindowsBase.end(),
                    [&](const auto& window) { return previousReferenceArc >= window.first && previousReferenceArc <= window.second; });
            if (config.recordDecisionDiagnostics && inDecisionDiagnosticWindow) {
                FiberletGraphReplayDecision decision;
                decision.routePointIndex = selectedEvaluation->routePointCount == 0 ? 0 : selectedEvaluation->routePointCount - 1;
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
                decision.relaxedBoundStateCount = searchStats.relaxedBoundStates;
                decision.relaxedBoundHitCount = searchStats.relaxedBoundHits;
                decision.relaxedBoundZeroFallbackCount =
                    searchStats.relaxedBoundZeroFallbacks;
                decision.initializationHistoryNodeCount =
                    searchStats.initializationHistoryNodeCount;
                decision.logicalRouteInternCount =
                    searchStats.logicalRouteInternCount;
                decision.logicalRouteCleanupVisitedCount =
                    searchStats.logicalRouteCleanupVisitedCount;
                decision.retainedBeamCount = ranked.size();
                decision.searchMode = config.searchWidth == 0 ? "exact_cost_bounded" : "intermediate_pruned";
                decision.searchWidth = config.searchWidth;
                decision.pruneDistancePredictionVoxels = pruneDistancePredictionVoxels;
                decision.pruneFronts = std::move(pruneFronts);
                decision.selectedPrefixLogicalArcs = logicalRouteArcs(selectedPrefixLogicalRoute.get());
                decision.sourceKey = graph.logicalAnchorId(seed->node.id);
                decision.selectedRouteIndex = 0;
                for (const auto& entry : ranked) {
                    FiberletGraphReplayDecisionRoute route;
                    route.prefixLogicalArcs = persistentRouteLogicalArcs(entry.prefix);
                    route.logicalArcs = persistentRouteLogicalArcs(entry.lookahead);
                    route.routePointsBaseXYZ = persistentRoutePointsBetween(
                        graph,
                        entry.lookahead,
                        currentCheckpoint,
                        entry.scoredPathLength);
                    route.edgeCost = entry.scoredEdgeCost;
                    route.transitionCost = entry.scoredTransitionCost;
                    route.committedEdgeCost = entry.prefix.edgeCost;
                    route.committedTransitionCost = entry.prefix.transitionCost;
                    route.committedPathLengthPredictionVoxels = entry.prefix.pathLength;
                    route.routePointsBeginPathLengthPredictionVoxels = currentCheckpoint;
                    route.pathLengthPredictionVoxels = entry.scoredPathLength;
                    route.completePathLengthPredictionVoxels = entry.completePathLength;
                    route.weightedEdgeLoss = entry.weightedEdgeLoss;
                    route.weightedTransitionLoss = entry.weightedTransitionLoss;
                    route.totalLoss = entry.totalLoss;
                    route.lossPerPredictionVoxel = entry.lossPerPredictionVoxel;
                    decision.routes.push_back(std::move(route));
                }
                decisions.push_back(std::move(decision));
            }
            if (selectedEvaluation->failure.has_value()) {
                auto materialized = materializeSelected(selectedEvaluation);
                segment = std::move(std::get<0>(materialized));
                distanceFailure = std::move(std::get<1>(materialized));
                selectedTraversedNodes = std::move(std::get<3>(materialized));
            } else if (selectedEvaluation->reachedReferenceEnd || checkpointPredictionVoxels >= maximumRouteLengthPredictionVoxels - kReplayEpsilon) {
                auto materialized = materializeSelected(selectedEvaluation);
                segment = std::move(std::get<0>(materialized));
                selectedTraversedNodes = std::move(std::get<3>(materialized));
                referenceExhausted = true;
            }
            emitProgress(result.segments.size(), previousReferenceArc, "running", rolloutExpandedStateCount, minimumAppliedLocalPruneLossCutoffPerPredictionVoxel);
        }
        if (segment.routePointsBaseXYZ.size() <= 1) {
            auto materialized = materializeSelected(evaluateRoute(selectedRoute));
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
        if (config.stopAtFirstFailure) {
            result.completedReferenceArcBase = failureArc;
            emitProgress(result.segments.size() - 1, failureArc, "failed");
            break;
        }
        resetArc = std::min(referenceEndArcBase, std::max(failureArc, result.segments.back().startReferenceArcBase + config.minimumResetAdvanceBaseVoxels));
        if (!(resetArc > result.segments.back().startReferenceArcBase + kReplayEpsilon))
            throw std::logic_error("fiberlet graph replay reset did not advance");
        result.completedReferenceArcBase = resetArc;
        emitProgress(result.segments.size() - 1, resetArc, "restart");
    }
    if (!config.stopAtFirstFailure && result.completedReferenceArcBase < referenceEndArcBase - kReplayEpsilon)
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
    nlohmann::json diagnosticWindows = nlohmann::json::array();
    for (const auto& [begin, end] : config.decisionDiagnosticReferenceArcWindowsBase)
        diagnosticWindows.push_back(nlohmann::json::array({begin, end}));
    nlohmann::json configJson = {
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
        {"cost_mode", replayCostModeName(config.costMode)},
        {"maximum_generated_states_per_iteration", config.maximumGeneratedStatesPerIteration},
        {"threshold", fiberReplayThresholdDescriptorJson(config.errorThresholdBaseVoxels)},
        {"match_refine_steps", config.matchRefineSteps},
        {"minimum_reset_advance_base_voxels", config.minimumResetAdvanceBaseVoxels},
        {"reference_begin_arc_base", config.referenceBeginArcBase},
        {"reference_end_arc_base", replay.referenceEndArcBase},
        {"initial_seed_key", config.initialSeedKey.has_value() ? storageKeyJson(*config.initialSeedKey) : nlohmann::json(nullptr)},
        {"require_initial_seed_in_first_window", config.requireInitialSeedInFirstWindow},
        {"stop_at_first_failure", config.stopAtFirstFailure},
        {"record_decision_diagnostics", config.recordDecisionDiagnostics},
        {"decision_diagnostic_reference_arc_windows_base", std::move(diagnosticWindows)},
    };
    if (config.costMode == FiberletGraphReplayCostMode::Stepped) {
        configJson.update({
            {"geometric_cost_weight_per_base_voxel", config.geometricCostWeightPerBaseVoxel},
            {"geometric_cost_delay_base_voxels", config.geometricCostDelayBaseVoxels},
            {"geometric_cost_delay_prediction_voxels", config.geometricCostDelayBaseVoxels / replay.predictionToBaseScale},
            {"cost_integration_step_base_voxels", config.costIntegrationStepBaseVoxels},
            {"cost_integration_step_prediction_voxels", config.costIntegrationStepBaseVoxels / replay.predictionToBaseScale},
            {"cost_profile_weight", config.costProfileWeight},
        });
    }
    nlohmann::json root = {
        {"format", "vc_fiberlet_graph_replay"},
        {"version", 3},
        {"coordinates",
         {
             {"position_order", "XYZ"},
             {"position_space", "base_volume"},
             {"distance_unit", "base_voxels"},
         }},
        {"config", std::move(configJson)},
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
                    {"route_points_begin_path_length_prediction_voxels", route.routePointsBeginPathLengthPredictionVoxels},
                    {"path_length_prediction_voxels", route.pathLengthPredictionVoxels},
                    {"complete_path_length_prediction_voxels", route.completePathLengthPredictionVoxels},
                    {"weighted_edge_loss", route.weightedEdgeLoss},
                    {"weighted_transition_loss", route.weightedTransitionLoss},
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
                {"relaxed_bound_state_count", decision.relaxedBoundStateCount},
                {"relaxed_bound_hit_count", decision.relaxedBoundHitCount},
                {"relaxed_bound_zero_fallback_count", decision.relaxedBoundZeroFallbackCount},
                {"initialization_history_node_count", decision.initializationHistoryNodeCount},
                {"logical_route_intern_count", decision.logicalRouteInternCount},
                {"logical_route_cleanup_visited_count", decision.logicalRouteCleanupVisitedCount},
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
        double end = std::max(failure.referenceArcBase, segment.endReferenceArcBase);
        if (!(end > begin + kReplayEpsilon) && !segment.matches.empty())
            end = std::min(replay.referenceEndArcBase, std::max(end, segment.matches.back().searchEndArcBase));
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
