#include "vc/fiber_tracer/FiberletCropTrace.hpp"

#include "vc/fiber_tracer/FiberReplayMetric.hpp"
#include "utils/thread_pool.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <exception>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <stdexcept>
#include <tuple>
#include <thread>

namespace vc::fiber_tracer
{
namespace
{

constexpr double kEpsilon = 1.0e-9;
constexpr double kPi = 3.14159265358979323846;

double length(const cv::Vec3d& value)
{
    return std::sqrt(std::max(0.0, value.dot(value)));
}

cv::Vec3d normalized(const cv::Vec3d& value)
{
    const double magnitude = length(value);
    return magnitude > kEpsilon ? value / magnitude : cv::Vec3d{};
}

bool inside(const cv::Vec3d& point, const FiberletCropTraceConfig& config)
{
    for (int axis = 0; axis < 3; ++axis) {
        if (!(point[axis] >= config.minimumBaseXYZ[axis] && point[axis] < config.maximumBaseXYZ[axis])) {
            return false;
        }
    }
    return true;
}

struct ClippedRoute {
    std::vector<cv::Vec3d> points;
    double retainedFraction = 1.0;
    bool exited = false;
};

ClippedRoute clipAtFirstExit(const std::vector<cv::Vec3d>& route, const FiberletCropTraceConfig& config)
{
    if (route.size() < 2 || !inside(route.front(), config))
        throw std::invalid_argument("Fiberlet crop route must start inside the crop");
    ClippedRoute result;
    result.points.push_back(route.front());
    double totalLength = 0.0;
    double retainedLength = 0.0;
    for (std::size_t index = 1; index < route.size(); ++index)
        totalLength += length(route[index] - route[index - 1]);
    if (!(totalLength > kEpsilon))
        throw std::invalid_argument("Fiberlet crop route has zero length");

    for (std::size_t index = 1; index < route.size(); ++index) {
        const cv::Vec3d start = route[index - 1];
        const cv::Vec3d finish = route[index];
        const cv::Vec3d delta = finish - start;
        const double segmentLength = length(delta);
        if (!(segmentLength > kEpsilon))
            continue;
        if (inside(finish, config)) {
            result.points.push_back(finish);
            retainedLength += segmentLength;
            continue;
        }
        double t = 1.0;
        for (int axis = 0; axis < 3; ++axis) {
            if (finish[axis] < config.minimumBaseXYZ[axis]) {
                t = std::min(t, (config.minimumBaseXYZ[axis] - start[axis]) / delta[axis]);
            } else if (finish[axis] >= config.maximumBaseXYZ[axis]) {
                t = std::min(t, (config.maximumBaseXYZ[axis] - start[axis]) / delta[axis]);
            }
        }
        t = std::clamp(t, 0.0, 1.0);
        result.points.push_back(start + delta * t);
        retainedLength += segmentLength * t;
        result.exited = true;
        break;
    }
    result.retainedFraction = std::clamp(retainedLength / totalLength, 0.0, 1.0);
    return result;
}

struct LookaheadState {
    FiberletStorageKey anchor;
    std::optional<FiberletReplaySourceArc> incoming;
    std::set<FiberletStorageKey> visited;
    std::vector<DirectedFiberletStorageId> arcs;
    double loss = 0.0;
    double lengthPrediction = 0.0;
};

struct LookaheadCompletion {
    DirectedFiberletStorageId first;
    std::vector<DirectedFiberletStorageId> arcs;
    double loss = 0.0;
    double lengthPrediction = 0.0;
};

bool completionLess(const LookaheadCompletion& left, const LookaheadCompletion& right)
{
    const double leftDensity = left.loss / left.lengthPrediction;
    const double rightDensity = right.loss / right.lengthPrediction;
    if (leftDensity != rightDensity)
        return leftDensity < rightDensity;
    return left.arcs < right.arcs;
}

std::optional<DirectedFiberletStorageId> selectLookaheadFirstArc(
    const FiberletReplayGraphSource& graph,
    const FiberletStorageKey& source,
    const std::optional<FiberletReplaySourceArc>& incoming,
    const std::set<FiberletStorageKey>& alreadyVisited,
    const std::optional<cv::Vec3d>& initialDirection,
    const std::optional<DirectedFiberletStorageId>& forcedFirst,
    const FiberletCropTraceConfig& config)
{
    const double horizonPrediction = config.lookaheadDistanceBaseVoxels / graph.predictionToBaseScale();
    std::vector<LookaheadState> frontier{{source, incoming, alreadyVisited, {}, 0.0, 0.0}};
    std::vector<LookaheadCompletion> completed;
    std::size_t generated = 0;
    const double minimumInitialDot = std::cos(graph.maximumJoinAngleDegrees() * kPi / 180.0);

    while (!frontier.empty() && generated < config.maximumGeneratedStatesPerStep) {
        std::vector<LookaheadState> next;
        for (auto& state : frontier) {
            bool expanded = false;
            auto outgoing = graph.outgoing(state.anchor);
            std::sort(outgoing.begin(), outgoing.end());
            for (const auto& id : outgoing) {
                if (state.arcs.empty() && forcedFirst.has_value() && id != *forcedFirst) {
                    continue;
                }
                const auto edge = graph.arc(id);
                if (state.visited.contains(edge.target))
                    continue;
                std::optional<FiberletReplaySourceTransition> join;
                if (state.incoming.has_value()) {
                    join = graph.transition(*state.incoming, edge);
                    if (!join.has_value())
                        continue;
                } else if (initialDirection.has_value()) {
                    const cv::Vec3d start = normalized(edge.startStepBaseXYZ);
                    if (!(start.dot(normalized(*initialDirection)) > minimumInitialDot)) {
                        continue;
                    }
                }
                const auto clipped = clipAtFirstExit(graph.routePoints(id), config);
                const double edgeLength = edge.pathLengthPredictionVoxels;
                if (!(edgeLength > kEpsilon))
                    continue;
                const double availableLength = edgeLength * clipped.retainedFraction;
                const double remaining = std::max(0.0, horizonPrediction - state.lengthPrediction);
                const double includedLength = std::min(availableLength, remaining);
                if (!(includedLength > kEpsilon))
                    continue;
                const double includedFraction = includedLength / edgeLength;
                LookaheadState candidate = state;
                candidate.anchor = edge.target;
                candidate.incoming = edge;
                candidate.visited.insert(edge.target);
                candidate.arcs.push_back(id);
                candidate.loss += edge.cost.total() * includedFraction;
                if (join.has_value())
                    candidate.loss += join->cost.total();
                candidate.lengthPrediction += includedLength;
                ++generated;
                expanded = true;

                const bool terminal = clipped.exited || candidate.lengthPrediction >= horizonPrediction - kEpsilon;
                if (terminal) {
                    completed.push_back({candidate.arcs.front(), candidate.arcs, candidate.loss, candidate.lengthPrediction});
                } else {
                    next.push_back(std::move(candidate));
                }
                if (generated >= config.maximumGeneratedStatesPerStep)
                    break;
            }
            if (!expanded && !state.arcs.empty()) {
                completed.push_back({state.arcs.front(), state.arcs, state.loss, state.lengthPrediction});
            }
            if (generated >= config.maximumGeneratedStatesPerStep)
                break;
        }
        if (next.size() > config.beamWidth * 64) {
            std::sort(next.begin(), next.end(), [](const auto& left, const auto& right) {
                const double leftDensity = left.loss / left.lengthPrediction;
                const double rightDensity = right.loss / right.lengthPrediction;
                if (leftDensity != rightDensity)
                    return leftDensity < rightDensity;
                return left.arcs < right.arcs;
            });
            next.resize(config.beamWidth * 64);
        }
        frontier = std::move(next);
    }
    for (const auto& state : frontier) {
        if (!state.arcs.empty()) {
            completed.push_back({state.arcs.front(), state.arcs, state.loss, state.lengthPrediction});
        }
    }
    if (completed.empty())
        return std::nullopt;
    std::sort(completed.begin(), completed.end(), completionLess);
    if (completed.size() > config.beamWidth)
        completed.resize(config.beamWidth);
    return completed.front().first;
}

struct SideTrace {
    std::vector<cv::Vec3d> points;
    std::string termination = "graph_exhausted";
    std::size_t fiberlets = 0;
};

SideTrace traceSide(
    const FiberletReplayGraphSource& graph,
    const FiberletStoredAnchor& seed,
    const cv::Vec3d& direction,
    const std::optional<DirectedFiberletStorageId>& forcedFirst,
    const FiberletCropTraceConfig& config)
{
    SideTrace result;
    const cv::Vec3d seedPoint(seed.positionPredictionXYZ * graph.predictionToBaseScale());
    result.points.push_back(seedPoint);
    FiberletStorageKey current = seed.key;
    std::optional<FiberletReplaySourceArc> incoming;
    std::set<FiberletStorageKey> visited{seed.key};
    for (std::size_t step = 0; step < config.maximumFiberletsPerSide; ++step) {
        const auto selected =
            selectLookaheadFirstArc(graph, current, incoming, visited, incoming.has_value() ? std::nullopt : std::make_optional(direction), incoming.has_value() ? std::nullopt : forcedFirst, config);
        if (!selected.has_value()) {
            result.termination = result.fiberlets == 0 ? "no_usable_edge" : "graph_exhausted";
            return result;
        }
        const auto edge = graph.arc(*selected);
        const auto clipped = clipAtFirstExit(graph.routePoints(*selected), config);
        result.points.insert(result.points.end(), std::next(clipped.points.begin()), clipped.points.end());
        ++result.fiberlets;
        if (clipped.exited) {
            result.termination = "crop_boundary";
            return result;
        }
        current = edge.target;
        incoming = edge;
        if (!visited.insert(current).second) {
            result.termination = "cycle_rejected";
            return result;
        }
    }
    result.termination = "fiberlet_limit";
    return result;
}

struct InitialPair {
    std::optional<DirectedFiberletStorageId> negative;
    std::optional<DirectedFiberletStorageId> positive;
};

struct TraceCandidate {
    FiberletCropTraceLine line;
    bool hasUsableEdge = false;
    bool bidirectional = false;
};

InitialPair selectInitialPair(const FiberletReplayGraphSource& graph, const FiberletStoredAnchor& seed)
{
    const cv::Vec3d axis = normalized(seed.fittedAxisXYZ);
    if (length(axis) <= kEpsilon)
        return {};
    const double minimumDot = std::cos(graph.maximumJoinAngleDegrees() * kPi / 180.0);
    std::vector<FiberletReplaySourceArc> negative;
    std::vector<FiberletReplaySourceArc> positive;
    for (const auto& id : graph.outgoing(seed.key)) {
        const auto edge = graph.arc(id);
        const cv::Vec3d direction = normalized(edge.startStepBaseXYZ);
        if (direction.dot(axis) > minimumDot)
            positive.push_back(edge);
        if (direction.dot(-axis) > minimumDot)
            negative.push_back(edge);
    }
    const auto edgeLess = [](const auto& left, const auto& right) {
        const double leftDensity = left.cost.total() / left.pathLengthPredictionVoxels;
        const double rightDensity = right.cost.total() / right.pathLengthPredictionVoxels;
        return std::tie(leftDensity, left.id) < std::tie(rightDensity, right.id);
    };
    std::sort(negative.begin(), negative.end(), edgeLess);
    std::sort(positive.begin(), positive.end(), edgeLess);

    struct RankedPair {
        DirectedFiberletStorageId negative;
        DirectedFiberletStorageId positive;
        double density = 0.0;
    };
    std::vector<RankedPair> pairs;
    for (const auto& neg : negative) {
        for (const auto& pos : positive) {
            if (neg.id.fiberlet == pos.id.fiberlet)
                continue;
            auto incomingId = neg.id;
            incomingId.reverse = !incomingId.reverse;
            const auto incoming = graph.arc(incomingId);
            const auto join = graph.transition(incoming, pos);
            if (!join.has_value())
                continue;
            const double routeLength = neg.pathLengthPredictionVoxels + pos.pathLengthPredictionVoxels;
            pairs.push_back({neg.id, pos.id, (neg.cost.total() + pos.cost.total() + join->cost.total()) / routeLength});
        }
    }
    if (!pairs.empty()) {
        std::sort(pairs.begin(), pairs.end(), [](const auto& left, const auto& right) {
            return std::tie(left.density, left.negative, left.positive) < std::tie(right.density, right.negative, right.positive);
        });
        return {pairs.front().negative, pairs.front().positive};
    }
    InitialPair result;
    if (!negative.empty())
        result.negative = negative.front().id;
    if (!positive.empty())
        result.positive = positive.front().id;
    if (result.negative.has_value() && result.positive.has_value() && result.negative->fiberlet == result.positive->fiberlet) {
        const auto neg = graph.arc(*result.negative);
        const auto pos = graph.arc(*result.positive);
        if (neg.cost.total() / neg.pathLengthPredictionVoxels <= pos.cost.total() / pos.pathLengthPredictionVoxels) {
            result.positive.reset();
        } else {
            result.negative.reset();
        }
    }
    return result;
}

TraceCandidate traceCandidate(
    const FiberletReplayGraphSource& graph,
    const FiberletStoredAnchor& seed,
    const FiberletCropTraceConfig& config)
{
    TraceCandidate result;
    result.line.seed = seed.key;
    result.line.seedPresence = seed.predictionPresence;
    const auto initial = selectInitialPair(graph, seed);
    if (!initial.negative.has_value() && !initial.positive.has_value())
        return result;
    result.hasUsableEdge = true;

    const cv::Vec3d axis = normalized(seed.fittedAxisXYZ);
    SideTrace negative;
    SideTrace positive;
    if (initial.negative.has_value()) {
        negative = traceSide(graph, seed, -axis, initial.negative, config);
    } else {
        negative.points = {cv::Vec3d(seed.positionPredictionXYZ * graph.predictionToBaseScale())};
        negative.termination = "no_usable_edge";
    }
    if (initial.positive.has_value()) {
        positive = traceSide(graph, seed, axis, initial.positive, config);
    } else {
        positive.points = {cv::Vec3d(seed.positionPredictionXYZ * graph.predictionToBaseScale())};
        positive.termination = "no_usable_edge";
    }

    result.line.negativeTermination = negative.termination;
    result.line.positiveTermination = positive.termination;
    result.line.negativeFiberlets = negative.fiberlets;
    result.line.positiveFiberlets = positive.fiberlets;
    result.line.pointsBaseXYZ.assign(negative.points.rbegin(), negative.points.rend());
    result.line.pointsBaseXYZ.insert(
        result.line.pointsBaseXYZ.end(),
        std::next(positive.points.begin()), positive.points.end());
    result.bidirectional = negative.fiberlets > 0 && positive.fiberlets > 0;
    return result;
}

using Bucket = std::array<std::int64_t, 3>;

Bucket bucketFor(const cv::Vec3d& point, double side)
{
    return {static_cast<std::int64_t>(std::floor(point[0] / side)), static_cast<std::int64_t>(std::floor(point[1] / side)), static_cast<std::int64_t>(std::floor(point[2] / side))};
}

std::size_t suppressCoveredAnchors(
    const std::vector<cv::Vec3d>& line,
    const std::vector<FiberletStoredAnchor>& anchors,
    std::vector<bool>& active,
    const std::map<Bucket, std::vector<std::size_t>>& index,
    const vc::lasagna::NormalSampler& normalSampler,
    double normalWorkingToBaseScale,
    const FiberletReplayGraphSource& graph,
    const FiberletCropTraceConfig& config)
{
    if (line.size() < 2)
        return 0;
    const double broadRadius = fiberReplayTangentialThresholdBaseVoxels(config.coverageNormalRadiusBaseVoxels);
    const double minimumDirectionDot = std::cos(config.coverageDirectionDegrees * kPi / 180.0);
    struct Projection {
        double distanceSquared = std::numeric_limits<double>::infinity();
        cv::Vec3d point{};
        cv::Vec3d tangent{};
    };
    std::map<std::size_t, Projection> projections;
    for (std::size_t segment = 1; segment < line.size(); ++segment) {
        const cv::Vec3d start = line[segment - 1];
        const cv::Vec3d finish = line[segment];
        const cv::Vec3d delta = finish - start;
        const double segmentLengthSquared = delta.dot(delta);
        if (!(segmentLengthSquared > kEpsilon))
            continue;
        const cv::Vec3d tangent = delta / std::sqrt(segmentLengthSquared);
        const Bucket low =
            bucketFor(cv::Vec3d{std::min(start[0], finish[0]) - broadRadius, std::min(start[1], finish[1]) - broadRadius, std::min(start[2], finish[2]) - broadRadius}, broadRadius);
        const Bucket high =
            bucketFor(cv::Vec3d{std::max(start[0], finish[0]) + broadRadius, std::max(start[1], finish[1]) + broadRadius, std::max(start[2], finish[2]) + broadRadius}, broadRadius);
        for (auto x = low[0]; x <= high[0]; ++x) {
            for (auto y = low[1]; y <= high[1]; ++y) {
                for (auto z = low[2]; z <= high[2]; ++z) {
                    const auto found = index.find({x, y, z});
                    if (found == index.end())
                        continue;
                    for (const auto anchorIndex : found->second) {
                        if (!active[anchorIndex])
                            continue;
                        const cv::Vec3d point(anchors[anchorIndex].positionPredictionXYZ * graph.predictionToBaseScale());
                        const double t = std::clamp((point - start).dot(delta) / segmentLengthSquared, 0.0, 1.0);
                        const cv::Vec3d projection = start + delta * t;
                        const double distanceSquared = (point - projection).dot(point - projection);
                        auto& best = projections[anchorIndex];
                        if (distanceSquared < best.distanceSquared) {
                            best = {distanceSquared, projection, tangent};
                        }
                    }
                }
            }
        }
    }

    std::size_t suppressed = 0;
    for (const auto& [anchorIndex, projection] : projections) {
        if (!active[anchorIndex])
            continue;
        const cv::Vec3d point(anchors[anchorIndex].positionPredictionXYZ * graph.predictionToBaseScale());
        const auto measurement =
            measureFiberReplayThreshold(point, projection.point, normalSampler, normalWorkingToBaseScale, config.coverageNormalRadiusBaseVoxels);
        if (fiberReplayThresholdExceeded(measurement, config.coverageNormalRadiusBaseVoxels)) {
            continue;
        }
        const cv::Vec3d axis = normalized(anchors[anchorIndex].fittedAxisXYZ);
        if (std::abs(axis.dot(projection.tangent)) + kEpsilon < minimumDirectionDot) {
            continue;
        }
        active[anchorIndex] = false;
        ++suppressed;
    }
    return suppressed;
}

}  // namespace

FiberletCropTraceResult traceFiberletCrop(
    const FiberletReplayGraphSource& graph,
    std::vector<FiberletStoredAnchor> anchors,
    const vc::lasagna::NormalSampler& normalSampler,
    double normalWorkingToBaseScale,
    const FiberletCropTraceConfig& config,
    const FiberletCropTraceProgress& progress)
{
    for (int axis = 0; axis < 3; ++axis) {
        if (!std::isfinite(config.minimumBaseXYZ[axis]) || !std::isfinite(config.maximumBaseXYZ[axis]) ||
            !(config.maximumBaseXYZ[axis] > config.minimumBaseXYZ[axis])) {
            throw std::invalid_argument("Fiberlet crop bounds must be finite and nonempty");
        }
    }
    if (!(config.lookaheadDistanceBaseVoxels > 0.0) || config.beamWidth == 0 || config.maximumGeneratedStatesPerStep == 0 ||
        config.maximumFiberletsPerSide == 0 || !(config.coverageNormalRadiusBaseVoxels > 0.0) ||
        !(config.coverageDirectionDegrees >= 0.0) || !(config.coverageDirectionDegrees <= 90.0) || !(normalWorkingToBaseScale > 0.0)) {
        throw std::invalid_argument("Fiberlet crop trace configuration is invalid");
    }
    std::sort(anchors.begin(), anchors.end(), [](const auto& left, const auto& right) {
        if (left.predictionPresence != right.predictionPresence)
            return left.predictionPresence > right.predictionPresence;
        return left.key < right.key;
    });

    FiberletCropTraceResult result;
    result.candidateAnchors = anchors.size();
    std::vector<bool> active(anchors.size(), true);
    const double bucketSide = fiberReplayTangentialThresholdBaseVoxels(config.coverageNormalRadiusBaseVoxels);
    std::map<Bucket, std::vector<std::size_t>> spatialIndex;
    for (std::size_t index = 0; index < anchors.size(); ++index) {
        const cv::Vec3d point(anchors[index].positionPredictionXYZ * graph.predictionToBaseScale());
        spatialIndex[bucketFor(point, bucketSide)].push_back(index);
    }

    const std::size_t requestedThreads = config.parallelThreads == 0
        ? std::max<std::size_t>(1, std::thread::hardware_concurrency())
        : config.parallelThreads;
    const std::size_t workerCount = graph.supportsConcurrentQueries()
        ? std::min(requestedThreads, std::max<std::size_t>(1, anchors.size()))
        : 1;
    std::optional<utils::ThreadPool> pool;
    if (workerCount > 1)
        pool.emplace(workerCount);

    std::size_t scan = 0;
    bool finished = false;
    while (!finished && scan < anchors.size()) {
        if ((config.maximumAttempts != 0 && result.attemptedAnchors >= config.maximumAttempts) ||
            (config.maximumFibers != 0 && result.lines.size() >= config.maximumFibers)) {
            break;
        }
        std::size_t batchCapacity = workerCount;
        if (config.maximumAttempts != 0) {
            batchCapacity = std::min(
                batchCapacity,
                config.maximumAttempts - result.attemptedAnchors);
        }
        std::vector<std::size_t> batch;
        batch.reserve(batchCapacity);
        while (scan < anchors.size() && batch.size() < batchCapacity) {
            if (active[scan])
                batch.push_back(scan);
            ++scan;
        }
        if (batch.empty())
            continue;

        std::vector<TraceCandidate> candidates(batch.size());
        std::vector<std::exception_ptr> failures(batch.size());
        std::vector<double> taskSeconds(batch.size());
        const auto compute = [&](std::size_t batchIndex) {
            const auto started = std::chrono::steady_clock::now();
            try {
                candidates[batchIndex] = traceCandidate(
                    graph, anchors[batch[batchIndex]], config);
            } catch (...) {
                failures[batchIndex] = std::current_exception();
            }
            taskSeconds[batchIndex] = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - started).count();
        };
        const auto batchStarted = std::chrono::steady_clock::now();
        const auto batchCpuStarted = std::clock();
        if (pool) {
            pool->run_indexed_batch(batch.size(), compute);
        } else {
            for (std::size_t batchIndex = 0; batchIndex < batch.size(); ++batchIndex)
                compute(batchIndex);
        }
        result.candidateBatchSeconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - batchStarted).count();
        result.candidateBatchCpuSeconds += static_cast<double>(
            std::clock() - batchCpuStarted) / CLOCKS_PER_SEC;
        result.candidateTaskSeconds += std::accumulate(
            taskSeconds.begin(), taskSeconds.end(), 0.0);
        result.maximumCandidateTaskSeconds = std::max(
            result.maximumCandidateTaskSeconds,
            *std::max_element(taskSeconds.begin(), taskSeconds.end()));
        result.computedCandidates += batch.size();
        for (const auto& failure : failures) {
            if (failure)
                std::rethrow_exception(failure);
        }

        const auto integrationStarted = std::chrono::steady_clock::now();
        for (std::size_t batchIndex = 0; batchIndex < batch.size(); ++batchIndex) {
            const std::size_t index = batch[batchIndex];
            if (!active[index]) {
                ++result.discardedCandidates;
                continue;
            }
            if ((config.maximumAttempts != 0 && result.attemptedAnchors >= config.maximumAttempts) ||
                (config.maximumFibers != 0 && result.lines.size() >= config.maximumFibers)) {
                finished = true;
                break;
            }
            active[index] = false;
            ++result.attemptedAnchors;
            auto& candidate = candidates[batchIndex];
            if (!candidate.hasUsableEdge || candidate.line.pointsBaseXYZ.size() < 2) {
                ++result.noEdgeAnchors;
                continue;
            }
            if (candidate.bidirectional)
                ++result.bidirectionalLines;
            else
                ++result.oneSidedLines;
            result.coveredAnchors += suppressCoveredAnchors(
                candidate.line.pointsBaseXYZ, anchors, active, spatialIndex,
                normalSampler, normalWorkingToBaseScale, graph, config);
            result.lines.push_back(std::move(candidate.line));
            if (progress) {
                const auto remaining = static_cast<std::size_t>(
                    std::count(active.begin(), active.end(), true));
                progress(result, remaining);
            }
        }
        result.integrationSeconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - integrationStarted).count();
    }
    return result;
}

}  // namespace vc::fiber_tracer
