#include "vc/fiber_tracer/FiberletCropTrace.hpp"

#include "vc/core/io/PolylineObj.hpp"
#include "vc/core/util/AtomicFile.hpp"
#include "vc/fiber_tracer/FiberAxisTensor.hpp"
#include "vc/fiber_tracer/FiberReplayMetric.hpp"
#include "vc/fiber_tracer/PolylineGeometry.hpp"
#include "utils/thread_pool.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <ctime>
#include <exception>
#include <iomanip>
#include <limits>
#include <locale>
#include <map>
#include <mutex>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <thread>

namespace vc::fiber_tracer
{
namespace
{

constexpr double kEpsilon = 1.0e-9;
constexpr double kPi = 3.14159265358979323846;
constexpr std::size_t kDirectionSeedCount = 8;
constexpr int kDirectionFitIterations = 32;
constexpr double kDirectionFitTolerance = 1.0e-12;
constexpr double kDirectionAxisSeparationEpsilon = 1.0e-8;
constexpr double kProbabilityRoundoffTolerance = 1.0e-12;

double length(const cv::Vec3d& value)
{
    return std::sqrt(std::max(0.0, value.dot(value)));
}

cv::Vec3d normalized(const cv::Vec3d& value)
{
    const double magnitude = length(value);
    return magnitude > kEpsilon ? value / magnitude : cv::Vec3d{};
}

bool finite(const cv::Vec3d& value)
{
    return std::isfinite(value[0]) && std::isfinite(value[1]) &&
        std::isfinite(value[2]);
}

struct DirectionStep {
    std::size_t line = 0;
    cv::Vec3d axis{0.0, 0.0, 0.0};
    double length = 0.0;
};

bool axisLess(const cv::Vec3d& left, const cv::Vec3d& right)
{
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
        if (left[coordinate] != right[coordinate])
            return left[coordinate] < right[coordinate];
    }
    return false;
}

std::array<double, 2> directionStepSupport(
    const cv::Vec3d& stepAxis,
    const cv::Vec3d& direction1,
    const cv::Vec3d& direction2)
{
    const double crossDot = std::clamp(direction1.dot(direction2), -1.0, 1.0);
    const double crossAlignment = crossDot * crossDot;
    const double separation = 1.0 - crossAlignment;
    const auto squaredAlignment = [&](const cv::Vec3d& direction) {
        const double dot = std::clamp(stepAxis.dot(direction), -1.0, 1.0);
        return dot * dot;
    };
    const double direction1Alignment = squaredAlignment(direction1);
    if (separation <= kDirectionAxisSeparationEpsilon)
        return {direction1Alignment, 0.0};
    const auto calibrated = [&](double alignment) {
        return std::clamp(
            (alignment - crossAlignment) / separation, 0.0, 1.0);
    };
    return {
        calibrated(direction1Alignment),
        calibrated(squaredAlignment(direction2)),
    };
}

std::size_t groupIndex(FiberDirectionGroup group)
{
    switch (group) {
    case FiberDirectionGroup::Direction1:
        return 0;
    case FiberDirectionGroup::Direction2:
        return 1;
    case FiberDirectionGroup::Mixed:
        return 2;
    }
    throw std::logic_error("invalid Fiber direction group");
}

std::string fiberName(const FiberletCropTraceLine& line, std::size_t index)
{
    std::ostringstream name;
    name.imbue(std::locale::classic());
    name << "fiber_" << std::setw(6) << std::setfill('0') << index
         << "_presence_" << std::fixed << std::setprecision(4)
         << line.seedPresence;
    return name.str();
}

struct ClippedRoute {
    double retainedFraction = 1.0;
    bool exited = false;
};

ClippedRoute clipAtFirstExit(
    const FiberletReplayRoutePointView& route,
    const cv::Vec3d& minimumBaseXYZ,
    const cv::Vec3d& maximumBaseXYZ,
    std::vector<cv::Vec3d>* appendedPoints = nullptr)
{
    if (route.size() < 2 ||
        !pointInHalfOpenBox(route.front(), minimumBaseXYZ, maximumBaseXYZ)) {
        throw std::invalid_argument("Fiberlet crop route must start inside the crop");
    }
    ClippedRoute result;
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
        if (pointInHalfOpenBox(finish, minimumBaseXYZ, maximumBaseXYZ)) {
            if (appendedPoints)
                appendedPoints->push_back(finish);
            retainedLength += segmentLength;
            continue;
        }
        const auto clipped = clipLineSegmentToHalfOpenBox(
            start, finish, minimumBaseXYZ, maximumBaseXYZ);
        if (!clipped) {
            // A minimum face belongs to the half-open crop. A segment that
            // starts on that face and points out has only a zero-length
            // intersection, which the general clipper intentionally rejects.
            if (pointInHalfOpenBox(start, minimumBaseXYZ, maximumBaseXYZ)) {
                result.exited = true;
                break;
            }
            throw std::logic_error("Fiberlet crop exit segment missed crop");
        }
        const double t = clipped->endFraction;
        if (appendedPoints)
            appendedPoints->push_back(clipped->finish);
        retainedLength += segmentLength * t;
        result.exited = true;
        break;
    }
    result.retainedFraction = std::clamp(retainedLength / totalLength, 0.0, 1.0);
    return result;
}

constexpr std::size_t kNoLookaheadRouteNode =
    std::numeric_limits<std::size_t>::max();

struct LookaheadRouteNode {
    FiberletStorageKey anchor;
    DirectedFiberletStorageId arcFromParent;
    std::size_t parent = kNoLookaheadRouteNode;
    double includedFraction = 0.0;
};

struct LookaheadState {
    std::size_t routeNode = kNoLookaheadRouteNode;
    std::optional<FiberletReplaySourceArc> incoming;
    DirectedFiberletStorageId first;
    std::size_t depth = 0;
    double loss = 0.0;
    double lengthPrediction = 0.0;
};

struct LookaheadCompletion {
    enum class Termination {
        Horizon,
        SearchBoxExit,
        GraphExhausted,
        StateLimit,
    };

    DirectedFiberletStorageId first;
    std::size_t routeNode = kNoLookaheadRouteNode;
    std::size_t depth = 0;
    double loss = 0.0;
    double lengthPrediction = 0.0;
    Termination termination = Termination::GraphExhausted;
};

struct LookaheadStatistics {
    std::size_t maximumRouteNodes = 0;
    std::size_t maximumRouteBytes = 0;
    std::size_t ambiguityDecisions = 0;
    std::size_t ambiguityRouteComparisons = 0;
    std::optional<double> minimumAmbiguityRelativeCostGap;
    double maximumAmbiguityThresholdRatio = 0.0;
};

struct LookaheadSearch {
    std::vector<LookaheadRouteNode> routes;
    std::vector<LookaheadCompletion> completed;
};

enum class InitialTraceSide {
    None,
    Negative,
    Positive,
};

bool rolloutContains(
    const std::vector<LookaheadRouteNode>& routes,
    std::size_t routeNode,
    const FiberletStorageKey& anchor)
{
    while (routeNode != kNoLookaheadRouteNode) {
        const auto& node = routes[routeNode];
        if (node.anchor == anchor)
            return true;
        routeNode = node.parent;
    }
    return false;
}

std::vector<DirectedFiberletStorageId> lookaheadArcs(
    const std::vector<LookaheadRouteNode>& routes,
    std::size_t routeNode,
    std::size_t depth)
{
    std::vector<DirectedFiberletStorageId> arcs;
    arcs.reserve(depth);
    while (routeNode != kNoLookaheadRouteNode &&
           routes[routeNode].parent != kNoLookaheadRouteNode) {
        arcs.push_back(routes[routeNode].arcFromParent);
        routeNode = routes[routeNode].parent;
    }
    std::reverse(arcs.begin(), arcs.end());
    return arcs;
}

const DirectedFiberletStorageId& lookaheadArcAt(
    const std::vector<LookaheadRouteNode>& routes,
    std::size_t routeNode,
    std::size_t depth,
    std::size_t index)
{
    for (std::size_t remaining = depth - index - 1; remaining > 0; --remaining)
        routeNode = routes[routeNode].parent;
    return routes[routeNode].arcFromParent;
}

bool completionLess(
    const std::vector<LookaheadRouteNode>& routes,
    const LookaheadCompletion& left,
    const LookaheadCompletion& right)
{
    const double leftDensity = left.loss / left.lengthPrediction;
    const double rightDensity = right.loss / right.lengthPrediction;
    if (leftDensity != rightDensity)
        return leftDensity < rightDensity;
    const std::size_t commonDepth = std::min(left.depth, right.depth);
    for (std::size_t index = 0; index < commonDepth; ++index) {
        const auto& leftArc = lookaheadArcAt(
            routes, left.routeNode, left.depth, index);
        const auto& rightArc = lookaheadArcAt(
            routes, right.routeNode, right.depth, index);
        if (leftArc != rightArc)
            return leftArc < rightArc;
    }
    return left.depth < right.depth;
}

LookaheadSearch collectLookaheadCompletions(
    const FiberletReplayGraphSource& graph,
    const FiberletStorageKey& source,
    const std::optional<FiberletReplaySourceArc>& incoming,
    const std::set<FiberletStorageKey>& alreadyVisited,
    const std::optional<cv::Vec3d>& initialDirection,
    const std::optional<DirectedFiberletStorageId>& forcedFirst,
    const std::optional<DirectedFiberletStorageId>& pairedFirst,
    InitialTraceSide initialSide,
    const FiberletCropTraceConfig& config,
    const FiberletCropTraceSearchBox& searchBox,
    LookaheadStatistics& statistics)
{
    const double horizonPrediction = config.lookaheadDistanceBaseVoxels / graph.predictionToBaseScale();
    std::vector<LookaheadRouteNode> routes{{source, {}, kNoLookaheadRouteNode, 0.0}};
    std::vector<LookaheadState> frontier{{0, incoming, {}, 0, 0.0, 0.0}};
    std::vector<LookaheadCompletion> completed;
    std::size_t generated = 0;
    const double minimumInitialDot = std::cos(graph.maximumJoinAngleDegrees() * kPi / 180.0);

    while (!frontier.empty() && generated < config.maximumGeneratedStatesPerStep) {
        std::vector<LookaheadState> next;
        for (const auto& state : frontier) {
            bool expanded = false;
            const auto stateAnchor = routes[state.routeNode].anchor;
            const auto outgoing = graph.outgoingArcs(stateAnchor);
            for (std::size_t outgoingIndex = 0;
                 outgoingIndex < outgoing.size(); ++outgoingIndex) {
                const auto& edge = outgoing[outgoingIndex];
                const auto& id = edge.id;
                if (state.depth == 0 && forcedFirst.has_value() && id != *forcedFirst) {
                    continue;
                }
                if (alreadyVisited.contains(edge.target) ||
                    rolloutContains(routes, state.routeNode, edge.target)) {
                    continue;
                }
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
                if (state.depth == 0 && pairedFirst.has_value()) {
                    std::optional<FiberletReplaySourceTransition> central;
                    if (initialSide == InitialTraceSide::Negative) {
                        auto incomingId = id;
                        incomingId.reverse = !incomingId.reverse;
                        central = graph.transition(
                            graph.arc(incomingId), graph.arc(*pairedFirst));
                    } else if (initialSide == InitialTraceSide::Positive) {
                        auto incomingId = *pairedFirst;
                        incomingId.reverse = !incomingId.reverse;
                        central = graph.transition(graph.arc(incomingId), edge);
                    }
                    if (initialSide == InitialTraceSide::None ||
                        !central.has_value()) {
                        continue;
                    }
                }
                const auto clipped = clipAtFirstExit(
                    graph.routePointView(id),
                    searchBox.minimumBaseXYZ,
                    searchBox.maximumBaseXYZ);
                const double edgeLength = edge.pathLengthPredictionVoxels;
                if (!(edgeLength > kEpsilon))
                    continue;
                const double availableLength = edgeLength * clipped.retainedFraction;
                const double remaining = std::max(0.0, horizonPrediction - state.lengthPrediction);
                const double includedLength = std::min(availableLength, remaining);
                if (!(includedLength > kEpsilon))
                    continue;
                const double includedFraction = includedLength / edgeLength;
                const std::size_t candidateRouteNode = routes.size();
                routes.push_back(
                    {edge.target, id, state.routeNode, includedFraction});
                LookaheadState candidate{
                    candidateRouteNode,
                    edge,
                    state.depth == 0 ? id : state.first,
                    state.depth + 1,
                    state.loss,
                    state.lengthPrediction,
                };
                candidate.loss += edge.cost.total() * includedFraction;
                if (join.has_value())
                    candidate.loss += join->cost.total();
                candidate.lengthPrediction += includedLength;
                ++generated;
                expanded = true;

                const bool reachedHorizon =
                    candidate.lengthPrediction >= horizonPrediction - kEpsilon;
                if (clipped.exited || reachedHorizon) {
                    completed.push_back(
                        {candidate.first, candidate.routeNode, candidate.depth,
                         candidate.loss, candidate.lengthPrediction,
                         reachedHorizon
                             ? LookaheadCompletion::Termination::Horizon
                             : LookaheadCompletion::Termination::SearchBoxExit});
                } else {
                    next.push_back(std::move(candidate));
                }
                if (generated >= config.maximumGeneratedStatesPerStep)
                    break;
            }
            if (!expanded && state.depth > 0) {
                completed.push_back(
                    {state.first, state.routeNode, state.depth, state.loss,
                     state.lengthPrediction,
                     LookaheadCompletion::Termination::GraphExhausted});
            }
            if (generated >= config.maximumGeneratedStatesPerStep)
                break;
        }
        if (next.size() > config.beamWidth * 64) {
            struct RankedState {
                LookaheadState state;
                std::vector<DirectedFiberletStorageId> arcs;
            };
            std::vector<RankedState> ranked;
            ranked.reserve(next.size());
            for (auto& state : next) {
                auto arcs = lookaheadArcs(
                    routes, state.routeNode, state.depth);
                ranked.push_back({std::move(state), std::move(arcs)});
            }
            std::sort(ranked.begin(), ranked.end(), [](const auto& left, const auto& right) {
                const double leftDensity = left.state.loss / left.state.lengthPrediction;
                const double rightDensity = right.state.loss / right.state.lengthPrediction;
                if (leftDensity != rightDensity)
                    return leftDensity < rightDensity;
                return left.arcs < right.arcs;
            });
            next.clear();
            next.reserve(config.beamWidth * 64);
            for (std::size_t index = 0; index < config.beamWidth * 64; ++index)
                next.push_back(std::move(ranked[index].state));
        }
        frontier = std::move(next);
    }
    for (const auto& state : frontier) {
        if (state.depth > 0) {
            completed.push_back(
                {state.first, state.routeNode, state.depth, state.loss,
                 state.lengthPrediction,
                 LookaheadCompletion::Termination::StateLimit});
        }
    }
    statistics.maximumRouteNodes = std::max(
        statistics.maximumRouteNodes, routes.size());
    statistics.maximumRouteBytes = std::max(
        statistics.maximumRouteBytes,
        routes.capacity() * sizeof(LookaheadRouteNode));
    return {std::move(routes), std::move(completed)};
}

const LookaheadCompletion* bestCompletion(const LookaheadSearch& search)
{
    if (search.completed.empty())
        return nullptr;
    return &*std::min_element(
        search.completed.begin(), search.completed.end(),
        [&](const auto& left, const auto& right) {
            return completionLess(search.routes, left, right);
        });
}

std::vector<cv::Vec3d> completionPoints(
    const FiberletReplayGraphSource& graph,
    const LookaheadSearch& search,
    const LookaheadCompletion& completion)
{
    std::vector<std::size_t> nodes;
    nodes.reserve(completion.depth);
    std::size_t node = completion.routeNode;
    while (node != kNoLookaheadRouteNode &&
           search.routes[node].parent != kNoLookaheadRouteNode) {
        nodes.push_back(node);
        node = search.routes[node].parent;
    }
    std::reverse(nodes.begin(), nodes.end());

    std::vector<cv::Vec3d> points;
    for (const auto routeNode : nodes) {
        const auto& step = search.routes[routeNode];
        const auto view = graph.routePointView(step.arcFromParent);
        std::vector<cv::Vec3d> edgePoints;
        edgePoints.reserve(view.size());
        for (std::size_t index = 0; index < view.size(); ++index)
            edgePoints.push_back(view[index]);
        const auto geometry = makePolylineArcGeometry(edgePoints);
        const auto clipped = slicePolylineArc(
            geometry, 0.0, geometry.length() * step.includedFraction);
        for (const auto& point : clipped) {
            if (points.empty() || length(point - points.back()) > kEpsilon)
                points.push_back(point);
        }
    }
    return points;
}

struct RouteSeparation {
    bool distinct = false;
    double maximumThresholdRatio = 0.0;
};

RouteSeparation directedRouteSeparation(
    const PolylineArcGeometry& evaluator,
    const PolylineArcGeometry& reference,
    const vc::lasagna::NormalSampler& normalSampler,
    double normalWorkingToBaseScale,
    double normalRadiusBaseVoxels)
{
    constexpr double kSampleStepBaseVoxels = 16.0;
    std::vector<double> arcs = evaluator.vertexArcs;
    for (double arc = kSampleStepBaseVoxels;
         arc < evaluator.length() - kEpsilon;
         arc += kSampleStepBaseVoxels) {
        arcs.push_back(arc);
    }
    std::sort(arcs.begin(), arcs.end());
    arcs.erase(
        std::unique(arcs.begin(), arcs.end(), [](double left, double right) {
            return std::abs(left - right) <= kEpsilon;
        }),
        arcs.end());

    RouteSeparation result;
    for (const double arc : arcs) {
        const auto sample = samplePolylineArc(evaluator, arc);
        const auto matched = projectPointToPolylineArc(
            reference, sample.point, 0.0, reference.length());
        const auto measurement = measureFiberReplayThreshold(
            sample.point, matched.point, normalSampler,
            normalWorkingToBaseScale, normalRadiusBaseVoxels);
        result.maximumThresholdRatio = std::max(
            result.maximumThresholdRatio, measurement.thresholdErrorRatio);
        result.distinct = result.distinct ||
            fiberReplayThresholdExceeded(measurement, normalRadiusBaseVoxels);
    }
    return result;
}

RouteSeparation routeSeparation(
    const std::vector<cv::Vec3d>& left,
    const std::vector<cv::Vec3d>& right,
    const vc::lasagna::NormalSampler& normalSampler,
    double normalWorkingToBaseScale,
    double normalRadiusBaseVoxels)
{
    const auto leftGeometry = makePolylineArcGeometry(left);
    const auto rightGeometry = makePolylineArcGeometry(right);
    const auto leftFromRight = directedRouteSeparation(
        leftGeometry, rightGeometry, normalSampler,
        normalWorkingToBaseScale, normalRadiusBaseVoxels);
    const auto rightFromLeft = directedRouteSeparation(
        rightGeometry, leftGeometry, normalSampler,
        normalWorkingToBaseScale, normalRadiusBaseVoxels);
    return {
        leftFromRight.distinct || rightFromLeft.distinct,
        std::max(leftFromRight.maximumThresholdRatio,
                 rightFromLeft.maximumThresholdRatio),
    };
}

struct LookaheadSelection {
    std::optional<DirectedFiberletStorageId> first;
    bool ambiguous = false;
};

LookaheadSelection selectLookaheadFirstArc(
    const FiberletReplayGraphSource& graph,
    const FiberletStorageKey& source,
    const std::optional<FiberletReplaySourceArc>& incoming,
    const std::set<FiberletStorageKey>& alreadyVisited,
    const std::optional<cv::Vec3d>& initialDirection,
    const std::optional<DirectedFiberletStorageId>& forcedFirst,
    const std::optional<DirectedFiberletStorageId>& pairedFirst,
    InitialTraceSide initialSide,
    const vc::lasagna::NormalSampler& normalSampler,
    double normalWorkingToBaseScale,
    const FiberletCropTraceConfig& config,
    const FiberletCropTraceSearchBox& searchBox,
    LookaheadStatistics& statistics)
{
    auto selectedSearch = collectLookaheadCompletions(
        graph, source, incoming, alreadyVisited, initialDirection,
        forcedFirst, pairedFirst, initialSide, config, searchBox, statistics);
    const auto* best = bestCompletion(selectedSearch);
    if (best == nullptr)
        return {};
    LookaheadSelection result{best->first, false};
    if (!config.ambiguityRelativeCostMargin.has_value() ||
        best->termination != LookaheadCompletion::Termination::Horizon) {
        return result;
    }

    std::optional<LookaheadSearch> alternativeSearch;
    const LookaheadSearch* candidates = &selectedSearch;
    if (forcedFirst.has_value()) {
        alternativeSearch = collectLookaheadCompletions(
            graph, source, incoming, alreadyVisited, initialDirection,
            std::nullopt, pairedFirst, initialSide, config, searchBox,
            statistics);
        candidates = &*alternativeSearch;
    }

    const double bestDensity = best->loss / best->lengthPrediction;
    std::map<DirectedFiberletStorageId, const LookaheadCompletion*>
        bestByFirstArc;
    for (const auto& candidate : candidates->completed) {
        if (candidate.termination !=
                LookaheadCompletion::Termination::Horizon ||
            candidate.first == best->first) {
            continue;
        }
        const auto [found, inserted] = bestByFirstArc.emplace(
            candidate.first, &candidate);
        if (!inserted && completionLess(
                candidates->routes, candidate, *found->second)) {
            found->second = &candidate;
        }
    }
    std::vector<const LookaheadCompletion*> ranked;
    ranked.reserve(bestByFirstArc.size());
    for (const auto& [first, candidate] : bestByFirstArc)
        ranked.push_back(candidate);
    std::sort(ranked.begin(), ranked.end(), [&](const auto* left, const auto* right) {
        return completionLess(candidates->routes, *left, *right);
    });
    const auto winnerPoints = completionPoints(graph, selectedSearch, *best);
    for (const auto* candidate : ranked) {
        const double candidateDensity =
            candidate->loss / candidate->lengthPrediction;
        const bool withinMargin = bestDensity == 0.0
            ? candidateDensity == 0.0
            : candidateDensity <= bestDensity *
                (1.0 + *config.ambiguityRelativeCostMargin);
        if (!withinMargin)
            break;
        ++statistics.ambiguityRouteComparisons;
        const auto candidatePoints = completionPoints(
            graph, *candidates, *candidate);
        const auto separation = routeSeparation(
            winnerPoints, candidatePoints, normalSampler,
            normalWorkingToBaseScale,
            config.ambiguityNormalRadiusBaseVoxels);
        statistics.maximumAmbiguityThresholdRatio = std::max(
            statistics.maximumAmbiguityThresholdRatio,
            separation.maximumThresholdRatio);
        if (!separation.distinct)
            continue;
        const double relativeGap = bestDensity == 0.0
            ? 0.0
            : (candidateDensity - bestDensity) / bestDensity;
        if (!statistics.minimumAmbiguityRelativeCostGap.has_value() ||
            relativeGap < *statistics.minimumAmbiguityRelativeCostGap) {
            statistics.minimumAmbiguityRelativeCostGap = relativeGap;
        }
        ++statistics.ambiguityDecisions;
        result.ambiguous = true;
        return result;
    }
    return result;
}

struct SideTrace {
    struct Boundary {
        FiberletStorageKey target;
        std::size_t pointCount = 0;
        double totalMetricCost = 0.0;
        double pathLengthPredictionVoxels = 0.0;
    };

    std::vector<cv::Vec3d> points;
    std::vector<Boundary> boundaries;
    std::string termination = "graph_exhausted";
    std::size_t fiberlets = 0;
  double totalMetricCost = 0.0;
  double pathLengthPredictionVoxels = 0.0;
};

SideTrace traceSide(
    const FiberletReplayGraphSource& graph,
    const FiberletStoredAnchor& seed,
    const cv::Vec3d& direction,
    const std::optional<DirectedFiberletStorageId>& forcedFirst,
    const std::optional<DirectedFiberletStorageId>& pairedFirst,
    InitialTraceSide initialSide,
    const vc::lasagna::NormalSampler& normalSampler,
    double normalWorkingToBaseScale,
    const FiberletCropTraceConfig& config,
    const FiberletCropTraceSearchBox& searchBox,
    LookaheadStatistics& statistics)
{
    SideTrace result;
    const cv::Vec3d seedPoint(seed.positionPredictionXYZ * graph.predictionToBaseScale());
    result.points.push_back(seedPoint);
    FiberletStorageKey current = seed.key;
    std::optional<FiberletReplaySourceArc> incoming;
    std::set<FiberletStorageKey> visited{seed.key};
    for (std::size_t step = 0; step < config.maximumFiberletsPerSide; ++step) {
        const auto selected = selectLookaheadFirstArc(
            graph, current, incoming, visited,
            incoming.has_value() ? std::nullopt
                                 : std::make_optional(direction),
            incoming.has_value() ? std::nullopt : forcedFirst,
            incoming.has_value() ? std::nullopt : pairedFirst,
            incoming.has_value() ? InitialTraceSide::None : initialSide,
            normalSampler, normalWorkingToBaseScale, config, searchBox,
            statistics);
        if (!selected.first.has_value()) {
            result.termination = result.fiberlets == 0 ? "no_usable_edge" : "graph_exhausted";
            return result;
        }
        if (selected.ambiguous) {
            result.termination = "ambiguous_route";
            return result;
        }
        const auto edge = graph.arc(*selected.first);
        if (incoming.has_value()) {
            const auto join = graph.transition(*incoming, edge);
            if (!join.has_value()) {
                throw std::logic_error(
                    "selected Fiberlet crop edge has no committed join");
            }
            result.totalMetricCost += join->cost.total();
        }
        const auto clipped = clipAtFirstExit(
            graph.routePointView(*selected.first),
            config.minimumBaseXYZ,
            config.maximumBaseXYZ,
            &result.points);
        result.totalMetricCost +=
            edge.cost.total() * clipped.retainedFraction;
        result.pathLengthPredictionVoxels +=
            edge.pathLengthPredictionVoxels * clipped.retainedFraction;
        ++result.fiberlets;
        if (clipped.exited) {
            result.termination = "crop_boundary";
            return result;
        }
        if (config.stopAtCoveredAnchors) {
            result.boundaries.push_back({
                edge.target,
                result.points.size(),
                result.totalMetricCost,
                result.pathLengthPredictionVoxels,
            });
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
    SideTrace negative;
    SideTrace positive;
    double centralMetricCost = 0.0;
    LookaheadStatistics lookahead;
    bool hasUsableEdge = false;
};

void truncateAtCoveredAnchor(
    SideTrace& side,
    const std::map<FiberletStorageKey, std::size_t>& anchorIndices,
    const std::vector<bool>& covered)
{
    for (std::size_t index = 0; index < side.boundaries.size(); ++index) {
        const auto found = anchorIndices.find(side.boundaries[index].target);
        if (found == anchorIndices.end() || !covered[found->second])
            continue;
        const auto boundary = side.boundaries[index];
        side.points.resize(boundary.pointCount);
        side.boundaries.resize(index + 1);
        side.fiberlets = index + 1;
        side.totalMetricCost = boundary.totalMetricCost;
        side.pathLengthPredictionVoxels =
            boundary.pathLengthPredictionVoxels;
        side.termination = "covered_anchor";
        return;
    }
}

FiberletCropTraceLine finalizeCandidate(TraceCandidate& candidate)
{
    auto& line = candidate.line;
    const auto& negative = candidate.negative;
    const auto& positive = candidate.positive;
    line.negativeTermination = negative.termination;
    line.positiveTermination = positive.termination;
    line.negativeFiberlets = negative.fiberlets;
    line.positiveFiberlets = positive.fiberlets;
    line.totalMetricCost =
        negative.totalMetricCost + positive.totalMetricCost;
    line.pathLengthPredictionVoxels =
        negative.pathLengthPredictionVoxels +
        positive.pathLengthPredictionVoxels;
    if (negative.fiberlets > 0 && positive.fiberlets > 0)
        line.totalMetricCost += candidate.centralMetricCost;
    line.pointsBaseXYZ.assign(
        negative.points.rbegin(), negative.points.rend());
    line.pointsBaseXYZ.insert(
        line.pointsBaseXYZ.end(), std::next(positive.points.begin()),
        positive.points.end());
    return std::move(line);
}

InitialPair selectInitialPair(const FiberletReplayGraphSource& graph, const FiberletStoredAnchor& seed)
{
    const cv::Vec3d axis = normalized(seed.fittedAxisXYZ);
    if (length(axis) <= kEpsilon)
        return {};
    const double minimumDot = std::cos(graph.maximumJoinAngleDegrees() * kPi / 180.0);
    std::vector<FiberletReplaySourceArc> negative;
    std::vector<FiberletReplaySourceArc> positive;
    const auto outgoing = graph.outgoingArcs(seed.key);
    for (std::size_t index = 0; index < outgoing.size(); ++index) {
        const auto& edge = outgoing[index];
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
    const vc::lasagna::NormalSampler& normalSampler,
    double normalWorkingToBaseScale,
    const FiberletCropTraceConfig& config,
    const FiberletCropTraceSearchBox& searchBox)
{
    TraceCandidate result;
    result.line.seed = seed.key;
    result.line.seedBaseXYZ = cv::Vec3d(
        seed.positionPredictionXYZ * graph.predictionToBaseScale());
    result.line.seedPresence = seed.predictionPresence;
    const auto initial = selectInitialPair(graph, seed);
    if (!initial.negative.has_value() && !initial.positive.has_value())
        return result;
    result.hasUsableEdge = true;

    const cv::Vec3d axis = normalized(seed.fittedAxisXYZ);
    if (initial.negative.has_value()) {
        result.negative = traceSide(
            graph, seed, -axis, initial.negative, initial.positive,
            InitialTraceSide::Negative, normalSampler,
            normalWorkingToBaseScale, config, searchBox, result.lookahead);
    } else {
        result.negative.points = {cv::Vec3d(seed.positionPredictionXYZ * graph.predictionToBaseScale())};
        result.negative.termination = "no_usable_edge";
    }
    if (initial.positive.has_value()) {
        result.positive = traceSide(
            graph, seed, axis, initial.positive, initial.negative,
            InitialTraceSide::Positive, normalSampler,
            normalWorkingToBaseScale, config, searchBox, result.lookahead);
    } else {
        result.positive.points = {cv::Vec3d(seed.positionPredictionXYZ * graph.predictionToBaseScale())};
        result.positive.termination = "no_usable_edge";
    }

    if (result.negative.fiberlets > 0 && result.positive.fiberlets > 0) {
        auto incomingId = *initial.negative;
        incomingId.reverse = !incomingId.reverse;
        const auto central = graph.transition(
            graph.arc(incomingId), graph.arc(*initial.positive));
        if (central.has_value())
            result.centralMetricCost = central->cost.total();
    }
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
    std::vector<bool>& covered,
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
                        if (covered[anchorIndex])
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
        if (covered[anchorIndex])
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
        covered[anchorIndex] = true;
        if (active[anchorIndex]) {
            active[anchorIndex] = false;
            ++suppressed;
        }
    }
    return suppressed;
}

}  // namespace

FiberletCropTraceSearchBox fiberletCropTraceSearchBox(
    const FiberletCropTraceConfig& config)
{
    if (!(config.lookaheadDistanceBaseVoxels > 0.0) ||
        !std::isfinite(config.lookaheadDistanceBaseVoxels)) {
        throw std::invalid_argument(
            "Fiberlet crop lookahead distance must be positive and finite");
    }
    FiberletCropTraceSearchBox result;
    for (int axis = 0; axis < 3; ++axis) {
        if (!std::isfinite(config.minimumBaseXYZ[axis]) ||
            !std::isfinite(config.maximumBaseXYZ[axis]) ||
            !(config.maximumBaseXYZ[axis] > config.minimumBaseXYZ[axis])) {
            throw std::invalid_argument(
                "Fiberlet crop bounds must be finite and nonempty");
        }
        result.minimumBaseXYZ[axis] =
            config.minimumBaseXYZ[axis] -
            config.lookaheadDistanceBaseVoxels;
        result.maximumBaseXYZ[axis] =
            config.maximumBaseXYZ[axis] +
            config.lookaheadDistanceBaseVoxels;
        if (!std::isfinite(result.minimumBaseXYZ[axis]) ||
            !std::isfinite(result.maximumBaseXYZ[axis])) {
            throw std::invalid_argument(
                "Fiberlet crop search bounds are not finite");
        }
    }
    return result;
}

FiberletCropTraceResult traceFiberletCrop(
    const FiberletReplayGraphSource& graph,
    std::vector<FiberletStoredAnchor> anchors,
    const vc::lasagna::NormalSampler& normalSampler,
    double normalWorkingToBaseScale,
    const FiberletCropTraceConfig& config,
    const FiberletCropTraceProgress& progress)
{
    const auto searchBox = fiberletCropTraceSearchBox(config);
    if (config.beamWidth == 0 || config.maximumGeneratedStatesPerStep == 0 ||
        config.maximumFiberletsPerSide == 0 || !(config.coverageNormalRadiusBaseVoxels > 0.0) ||
        !(config.coverageDirectionDegrees >= 0.0) || !(config.coverageDirectionDegrees <= 90.0) || !(normalWorkingToBaseScale > 0.0) ||
        !(config.ambiguityNormalRadiusBaseVoxels > 0.0) ||
        !std::isfinite(config.ambiguityNormalRadiusBaseVoxels) ||
        (config.ambiguityRelativeCostMargin &&
         (!std::isfinite(*config.ambiguityRelativeCostMargin) ||
          *config.ambiguityRelativeCostMargin < 0.0)) ||
        (config.maximumAcceptedCostDensity &&
         (!std::isfinite(*config.maximumAcceptedCostDensity) ||
          *config.maximumAcceptedCostDensity < 0.0))) {
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
    std::vector<bool> covered(anchors.size(), false);
    std::size_t activeAnchors = anchors.size();
    std::map<FiberletStorageKey, std::size_t> anchorIndices;
    const double bucketSide = fiberReplayTangentialThresholdBaseVoxels(config.coverageNormalRadiusBaseVoxels);
    std::map<Bucket, std::vector<std::size_t>> spatialIndex;
    for (std::size_t index = 0; index < anchors.size(); ++index) {
        if (!anchorIndices.emplace(anchors[index].key, index).second)
            throw std::invalid_argument("Fiberlet crop anchors contain a duplicate key");
        const cv::Vec3d point(anchors[index].positionPredictionXYZ * graph.predictionToBaseScale());
        spatialIndex[bucketFor(point, bucketSide)].push_back(index);
    }

    const std::size_t requestedThreads = config.parallelThreads == 0
        ? std::max<std::size_t>(1, std::thread::hardware_concurrency())
        : config.parallelThreads;
    const bool ambiguitySamplingIsConcurrent =
        !config.ambiguityRelativeCostMargin.has_value() ||
        normalSampler.supportsConcurrentSampling();
    const std::size_t workerCount = graph.supportsConcurrentQueries() &&
            ambiguitySamplingIsConcurrent
        ? std::min(requestedThreads, std::max<std::size_t>(1, anchors.size()))
        : 1;
    struct Completion {
        std::size_t anchorIndex = 0;
        TraceCandidate candidate;
        std::exception_ptr failure;
        double taskSeconds = 0.0;
    };
    struct CompletionQueue {
        std::mutex mutex;
        std::condition_variable ready;
        std::map<std::size_t, Completion> byTicket;
        std::size_t completed = 0;
        double taskSeconds = 0.0;
        double maximumTaskSeconds = 0.0;
        std::size_t maximumLookaheadRouteNodes = 0;
        std::size_t maximumLookaheadRouteBytes = 0;
        std::size_t ambiguityDecisions = 0;
        std::size_t ambiguityRouteComparisons = 0;
        std::optional<double> minimumAmbiguityRelativeCostGap;
        double maximumAmbiguityThresholdRatio = 0.0;
    } completions;

    std::optional<utils::ThreadPool> pool;
    if (workerCount > 1)
        pool.emplace(workerCount);
    const std::size_t speculationWindow = workerCount == 1
        ? 1
        : workerCount + std::max<std::size_t>(1, workerCount / 8);
    std::size_t scan = 0;
    std::size_t nextTicket = 0;
    std::size_t commitTicket = 0;
    bool stopped = false;
    std::exception_ptr orderedFailure;
    const auto schedulerStarted = std::chrono::steady_clock::now();
    const auto schedulerCpuStarted = std::clock();

    const auto limitsReached = [&] {
        return (config.maximumAttempts != 0 &&
                result.attemptedAnchors >= config.maximumAttempts) ||
            (config.maximumFibers != 0 &&
             result.lines.size() >= config.maximumFibers);
    };
    const auto submit = [&](std::size_t ticket, std::size_t anchorIndex) {
        const auto compute = [&, ticket, anchorIndex] {
            Completion completion;
            completion.anchorIndex = anchorIndex;
            const auto started = std::chrono::steady_clock::now();
            try {
                completion.candidate = traceCandidate(
                    graph, anchors[anchorIndex], normalSampler,
                    normalWorkingToBaseScale, config, searchBox);
            } catch (...) {
                completion.failure = std::current_exception();
            }
            completion.taskSeconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - started).count();
            {
                std::lock_guard lock(completions.mutex);
                ++completions.completed;
                completions.taskSeconds += completion.taskSeconds;
                completions.maximumTaskSeconds = std::max(
                    completions.maximumTaskSeconds,
                    completion.taskSeconds);
                completions.maximumLookaheadRouteNodes = std::max(
                    completions.maximumLookaheadRouteNodes,
                    completion.candidate.lookahead.maximumRouteNodes);
                completions.maximumLookaheadRouteBytes = std::max(
                    completions.maximumLookaheadRouteBytes,
                    completion.candidate.lookahead.maximumRouteBytes);
                completions.ambiguityDecisions +=
                    completion.candidate.lookahead.ambiguityDecisions;
                completions.ambiguityRouteComparisons +=
                    completion.candidate.lookahead.ambiguityRouteComparisons;
                const auto gap = completion.candidate.lookahead.
                    minimumAmbiguityRelativeCostGap;
                if (gap &&
                    (!completions.minimumAmbiguityRelativeCostGap ||
                     *gap < *completions.minimumAmbiguityRelativeCostGap)) {
                    completions.minimumAmbiguityRelativeCostGap = gap;
                }
                completions.maximumAmbiguityThresholdRatio = std::max(
                    completions.maximumAmbiguityThresholdRatio,
                    completion.candidate.lookahead.
                        maximumAmbiguityThresholdRatio);
                completions.byTicket.emplace(
                    ticket, std::move(completion));
            }
            completions.ready.notify_one();
        };
        if (pool)
            pool->enqueue(compute);
        else
            compute();
    };
    const auto fillWindow = [&] {
        while (!stopped && !limitsReached() &&
               nextTicket - commitTicket < speculationWindow) {
            if (config.maximumAttempts != 0) {
                const std::size_t outstanding = nextTicket - commitTicket;
                const std::size_t remaining =
                    config.maximumAttempts - result.attemptedAnchors;
                if (outstanding >= remaining)
                    break;
            }
            while (scan < anchors.size() && !active[scan])
                ++scan;
            if (scan == anchors.size())
                break;
            submit(nextTicket++, scan++);
        }
    };

    while (!stopped) {
        fillWindow();
        if (commitTicket == nextTicket) {
            if (scan == anchors.size() || limitsReached())
                break;
            continue;
        }

        Completion completion;
        {
            std::unique_lock lock(completions.mutex);
            completions.ready.wait(lock, [&] {
                return completions.byTicket.contains(commitTicket);
            });
            auto found = completions.byTicket.find(commitTicket);
            completion = std::move(found->second);
            completions.byTicket.erase(found);
        }
        ++commitTicket;
        if (completion.failure) {
            orderedFailure = completion.failure;
            stopped = true;
            break;
        }

        const auto integrationStarted = std::chrono::steady_clock::now();
        const std::size_t index = completion.anchorIndex;
        if (!active[index]) {
            ++result.discardedCandidates;
        } else if (!limitsReached()) {
            active[index] = false;
            --activeAnchors;
            ++result.attemptedAnchors;
            auto& candidate = completion.candidate;
            if (!candidate.hasUsableEdge) {
                ++result.noEdgeAnchors;
            } else {
                if (config.stopAtCoveredAnchors) {
                    truncateAtCoveredAnchor(
                        candidate.negative, anchorIndices, covered);
                    truncateAtCoveredAnchor(
                        candidate.positive, anchorIndices, covered);
                }
                const bool bidirectional = candidate.negative.fiberlets > 0 &&
                    candidate.positive.fiberlets > 0;
                auto line = finalizeCandidate(candidate);
                if (line.pointsBaseXYZ.size() < 2) {
                    ++result.noEdgeAnchors;
                } else if (config.maximumAcceptedCostDensity &&
                           line.totalMetricCost /
                                   line.pathLengthPredictionVoxels >
                               *config.maximumAcceptedCostDensity) {
                    ++result.qualityRejectedAnchors;
                } else {
                    result.coveredAnchorStops +=
                        candidate.negative.termination == "covered_anchor";
                    result.coveredAnchorStops +=
                        candidate.positive.termination == "covered_anchor";
                    result.acceptedAmbiguityStops +=
                        candidate.negative.termination == "ambiguous_route";
                    result.acceptedAmbiguityStops +=
                        candidate.positive.termination == "ambiguous_route";
                    if (bidirectional)
                        ++result.bidirectionalLines;
                    else
                        ++result.oneSidedLines;
                    covered[index] = true;
                    const std::size_t suppressed = suppressCoveredAnchors(
                        line.pointsBaseXYZ, anchors, active, covered,
                        spatialIndex, normalSampler, normalWorkingToBaseScale,
                        graph, config);
                    result.coveredAnchors += suppressed;
                    activeAnchors -= suppressed;
                    result.lines.push_back(std::move(line));
                }
            }
            if (progress)
                progress(result, activeAnchors);
        }
        result.integrationSeconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - integrationStarted).count();
        if (limitsReached())
            stopped = true;
    }
    if (pool)
        pool->wait_idle();
    {
        std::lock_guard lock(completions.mutex);
        result.computedCandidates = completions.completed;
        result.candidateTaskSeconds = completions.taskSeconds;
        result.maximumCandidateTaskSeconds =
            completions.maximumTaskSeconds;
        result.maximumLookaheadRouteNodes =
            completions.maximumLookaheadRouteNodes;
        result.maximumLookaheadRouteBytes =
            completions.maximumLookaheadRouteBytes;
        result.ambiguityDecisions = completions.ambiguityDecisions;
        result.ambiguityRouteComparisons =
            completions.ambiguityRouteComparisons;
        result.minimumAmbiguityRelativeCostGap =
            completions.minimumAmbiguityRelativeCostGap;
        result.maximumAmbiguityThresholdRatio =
            completions.maximumAmbiguityThresholdRatio;
    }
    result.candidateBatchSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - schedulerStarted).count();
    result.candidateBatchCpuSeconds = static_cast<double>(
        std::clock() - schedulerCpuStarted) / CLOCKS_PER_SEC;
    if (orderedFailure)
        std::rethrow_exception(orderedFailure);
    return result;
}

FiberDirectionClassification classifyFiberletCropDirections(
    const std::vector<FiberletCropTraceLine>& lines,
    double dominanceFraction)
{
    if (!(dominanceFraction > 0.5 && dominanceFraction <= 1.0) ||
        !std::isfinite(dominanceFraction)) {
        throw std::invalid_argument(
            "Fiber direction dominance fraction must be finite and in (0.5, 1]");
    }

    FiberDirectionClassification result;
    result.dominanceFraction = dominanceFraction;
    result.lines.resize(lines.size());
    std::vector<DirectionStep> steps;
    for (std::size_t lineIndex = 0; lineIndex < lines.size(); ++lineIndex) {
        const auto& points = lines[lineIndex].pointsBaseXYZ;
        for (const auto& point : points) {
            if (!finite(point)) {
                throw std::invalid_argument(
                    "Fiber direction classification requires finite points");
            }
        }
        for (std::size_t pointIndex = 1; pointIndex < points.size(); ++pointIndex) {
            const cv::Vec3d delta = points[pointIndex] - points[pointIndex - 1];
            const double stepLength = length(delta);
            if (!(stepLength > kEpsilon))
                continue;
            steps.push_back({lineIndex, delta / stepLength, stepLength});
            ++result.analyzedSteps;
            result.analyzedLengthBaseVoxels += stepLength;
        }
    }
    if (steps.empty()) {
        result.groupCounts[groupIndex(FiberDirectionGroup::Mixed)] = lines.size();
        return result;
    }

    cv::Matx33d globalTensor = cv::Matx33d::zeros();
    for (const auto& step : steps)
        globalTensor += fiberAxisTensor(step.axis, step.length);
    const auto global = principalFiberAxis(globalTensor);

    std::vector<cv::Vec3d> seeds;
    seeds.reserve(std::min(kDirectionSeedCount, steps.size()));
    if (global.unique) {
        seeds.push_back(global.axis);
    } else {
        std::size_t strongest = 0;
        for (std::size_t index = 1; index < steps.size(); ++index) {
            if (steps[index].length > steps[strongest].length)
                strongest = index;
        }
        seeds.push_back(canonicalFiberAxis(steps[strongest].axis));
    }
    while (seeds.size() < std::min(kDirectionSeedCount, steps.size())) {
        std::size_t best = steps.size();
        double bestScore = 0.0;
        for (std::size_t index = 0; index < steps.size(); ++index) {
            double dissimilarity = 1.0;
            for (const auto& seed : seeds) {
                const double dot = steps[index].axis.dot(seed);
                dissimilarity = std::min(
                    dissimilarity,
                    std::max(0.0, 1.0 - dot * dot));
            }
            const double score = steps[index].length * dissimilarity;
            if (score > bestScore) {
                best = index;
                bestScore = score;
            }
        }
        if (best == steps.size() || !(bestScore > 0.0))
            break;
        seeds.push_back(canonicalFiberAxis(steps[best].axis));
    }

    FiberAxisPairFit<double> bestFit;
    const auto refine = [&](const std::array<cv::Vec3d, 2>& initial) {
        return refineFiberAxisPair<double>(
            steps.size(), initial, kDirectionFitIterations,
            kDirectionFitTolerance,
            [&](std::size_t index) { return steps[index].axis; },
            [&](std::size_t index) { return steps[index].length; });
    };
    if (seeds.size() == 1) {
        bestFit = refine({seeds.front(), seeds.front()});
    } else {
        for (std::size_t first = 0; first + 1 < seeds.size(); ++first) {
            for (std::size_t second = first + 1; second < seeds.size(); ++second) {
                auto fit = refine({seeds[first], seeds[second]});
                if (bestFit.objective < 0.0 || fit.objective > bestFit.objective)
                    bestFit = std::move(fit);
            }
        }
    }

    bestFit.axes[0] = canonicalFiberAxis(bestFit.axes[0]);
    bestFit.axes[1] = canonicalFiberAxis(bestFit.axes[1]);
    std::array<double, 2> assignedLengths{0.0, 0.0};
    for (std::size_t index = 0; index < steps.size(); ++index)
        assignedLengths[bestFit.assignments[index]] += steps[index].length;
    const bool swapDirections =
        assignedLengths[1] > assignedLengths[0] ||
        (assignedLengths[1] == assignedLengths[0] &&
         axisLess(bestFit.axes[1], bestFit.axes[0]));
    if (swapDirections) {
        std::swap(bestFit.axes[0], bestFit.axes[1]);
        for (auto& assignment : bestFit.assignments)
            assignment = static_cast<std::uint8_t>(1 - assignment);
    }
    result.direction1BaseXYZ = bestFit.axes[0];
    result.direction2BaseXYZ = bestFit.axes[1];

    for (std::size_t index = 0; index < steps.size(); ++index) {
        auto& line = result.lines[steps[index].line];
        const auto support = directionStepSupport(
            steps[index].axis, bestFit.axes[0], bestFit.axes[1]);
        line.direction1SupportBaseVoxels += support[0] * steps[index].length;
        line.direction2SupportBaseVoxels += support[1] * steps[index].length;
        line.totalLengthBaseVoxels += steps[index].length;
    }
    for (auto& line : result.lines) {
        if (line.totalLengthBaseVoxels > kEpsilon) {
            const double direction1Fraction =
                line.direction1SupportBaseVoxels / line.totalLengthBaseVoxels;
            const double direction2Fraction =
                line.direction2SupportBaseVoxels / line.totalLengthBaseVoxels;
            if (direction1Fraction >= direction2Fraction &&
                direction1Fraction >= dominanceFraction) {
                line.group = FiberDirectionGroup::Direction1;
            } else if (direction2Fraction > direction1Fraction &&
                       direction2Fraction >= dominanceFraction) {
                line.group = FiberDirectionGroup::Direction2;
            }
        }
        ++result.groupCounts[groupIndex(line.group)];
    }
    return result;
}

std::vector<FiberDirectionAblationCandidate> rankMixedFiberDirections(
    const FiberDirectionClassification& classification)
{
    std::vector<FiberDirectionAblationCandidate> result;
    result.reserve(classification.groupCounts[groupIndex(
        FiberDirectionGroup::Mixed)]);
    for (std::size_t lineIndex = 0;
         lineIndex < classification.lines.size();
         ++lineIndex) {
        const auto& line = classification.lines[lineIndex];
        if (line.group != FiberDirectionGroup::Mixed)
            continue;
        if (!std::isfinite(line.direction1SupportBaseVoxels) ||
            !std::isfinite(line.direction2SupportBaseVoxels) ||
            !std::isfinite(line.totalLengthBaseVoxels) ||
            line.direction1SupportBaseVoxels < 0.0 ||
            line.direction2SupportBaseVoxels < 0.0 ||
            line.totalLengthBaseVoxels < 0.0) {
            throw std::invalid_argument(
                "Fiber direction ablation requires finite non-negative support");
        }
        const double confidence = line.totalLengthBaseVoxels > kEpsilon
            ? std::max(
                  line.direction1SupportBaseVoxels,
                  line.direction2SupportBaseVoxels) /
                line.totalLengthBaseVoxels
            : 0.0;
        result.push_back({lineIndex, confidence});
    }
    std::sort(
        result.begin(), result.end(),
        [](const auto& a, const auto& b) {
            if (a.confidence != b.confidence)
                return a.confidence > b.confidence;
            return a.lineIndex < b.lineIndex;
        });
    return result;
}

FiberDirectionObjPaths fiberDirectionObjPaths(
    const std::filesystem::path& allOutputPath)
{
    const auto sibling = [&](const std::string& suffix) {
        return allOutputPath.parent_path() /
            (allOutputPath.stem().string() + suffix +
             allOutputPath.extension().string());
    };
    return {
        allOutputPath,
        sibling("_dir1"),
        sibling("_dir2"),
        sibling("_mixed"),
        sibling("_anchors"),
        sibling("_dir1_anchors"),
        sibling("_dir2_anchors"),
        sibling("_mixed_anchors"),
    };
}

void writeFiberletCropDirectionObjs(
    const std::vector<FiberletCropTraceLine>& lines,
    const FiberDirectionClassification& classification,
    const std::filesystem::path& allOutputPath)
{
    if (classification.lines.size() != lines.size()) {
        throw std::invalid_argument(
            "Fiber direction classification does not match crop lines");
    }
    std::vector<vc::core::io::NamedPolyline> all;
    std::array<std::vector<vc::core::io::NamedPolyline>, 3> grouped;
    std::vector<vc::core::io::NamedPolyline> allAnchors;
    std::array<std::vector<vc::core::io::NamedPolyline>, 3> groupedAnchors;
    all.reserve(lines.size());
    allAnchors.reserve(lines.size());
    for (std::size_t index = 0; index < lines.size(); ++index) {
        const std::string name = fiberName(lines[index], index);
        vc::core::io::NamedPolyline line{
            name, lines[index].pointsBaseXYZ};
        vc::core::io::NamedPolyline anchor{
            name + "_anchor", {lines[index].seedBaseXYZ}};
        all.push_back(line);
        allAnchors.push_back(anchor);
        grouped[groupIndex(classification.lines[index].group)].push_back(
            std::move(line));
        groupedAnchors[groupIndex(classification.lines[index].group)].push_back(
            std::move(anchor));
    }
    const auto paths = fiberDirectionObjPaths(allOutputPath);
    vc::core::io::writePolylinesObj(all, paths.all, "VC3D Fiberlet crop traces");
    vc::core::io::writePolylinesObj(
        grouped[0], paths.direction1,
        "VC3D Fiberlet crop traces: direction 1 dominant");
    vc::core::io::writePolylinesObj(
        grouped[1], paths.direction2,
        "VC3D Fiberlet crop traces: direction 2 dominant");
    vc::core::io::writePolylinesObj(
        grouped[2], paths.mixed,
        "VC3D Fiberlet crop traces: mixed directions");
    vc::core::io::writePolylinesObj(
        allAnchors, paths.allAnchors, "VC3D Fiberlet crop trace seed anchors");
    vc::core::io::writePolylinesObj(
        groupedAnchors[0], paths.direction1Anchors,
        "VC3D Fiberlet crop trace seed anchors: direction 1 dominant");
    vc::core::io::writePolylinesObj(
        groupedAnchors[1], paths.direction2Anchors,
        "VC3D Fiberlet crop trace seed anchors: direction 2 dominant");
    vc::core::io::writePolylinesObj(
        groupedAnchors[2], paths.mixedAnchors,
        "VC3D Fiberlet crop trace seed anchors: mixed directions");
}

namespace {

struct RankedQualityLine {
    std::size_t index = 0;
    double density = 0.0;
};

std::vector<RankedQualityLine>
rankFiberletCropQuality(const std::vector<FiberletCropTraceLine>& lines) {
  std::vector<RankedQualityLine> ranked;
  ranked.reserve(lines.size());
  for (std::size_t index = 0; index < lines.size(); ++index) {
    const auto &line = lines[index];
    if (!(line.totalMetricCost >= 0.0) ||
        !std::isfinite(line.totalMetricCost) ||
        !(line.pathLengthPredictionVoxels > 0.0) ||
        !std::isfinite(line.pathLengthPredictionVoxels)) {
      throw std::invalid_argument(
          "Fiber quality requires finite nonnegative cost and positive length");
    }
    ranked.push_back({
        index,
        line.totalMetricCost / line.pathLengthPredictionVoxels,
    });
  }
  std::sort(ranked.begin(), ranked.end(),
            [](const auto &left, const auto &right) {
              return std::tie(left.density, left.index) <
                     std::tie(right.density, right.index);
            });
  return ranked;
}

}  // namespace

FiberQualityHistogram
classifyFiberletCropQuality(const std::vector<FiberletCropTraceLine> &lines) {
  const auto ranked = rankFiberletCropQuality(lines);

  FiberQualityHistogram result;
  for (std::size_t rank = 0; rank < ranked.size(); ++rank) {
    const std::size_t bin = std::min<std::size_t>(9, rank * 10 / ranked.size());
    result.bins[bin].lineIndices.push_back(ranked[rank].index);
  }
  for (auto &bin : result.bins) {
    if (bin.lineIndices.empty())
      continue;
    bin.minimumTotalMetricCost = std::numeric_limits<double>::infinity();
    bin.maximumTotalMetricCost = -std::numeric_limits<double>::infinity();
    bin.minimumCostDensity = std::numeric_limits<double>::infinity();
    bin.maximumCostDensity = -std::numeric_limits<double>::infinity();
    double totalCost = 0.0;
    double totalDensity = 0.0;
    for (const auto index : bin.lineIndices) {
      const auto &line = lines[index];
      const double density =
          line.totalMetricCost / line.pathLengthPredictionVoxels;
      bin.minimumTotalMetricCost =
          std::min(bin.minimumTotalMetricCost, line.totalMetricCost);
      bin.maximumTotalMetricCost =
          std::max(bin.maximumTotalMetricCost, line.totalMetricCost);
      bin.minimumCostDensity = std::min(bin.minimumCostDensity, density);
      bin.maximumCostDensity = std::max(bin.maximumCostDensity, density);
      totalCost += line.totalMetricCost;
      totalDensity += density;
    }
    const double count = static_cast<double>(bin.lineIndices.size());
    bin.meanTotalMetricCost = totalCost / count;
    bin.meanCostDensity = totalDensity / count;
  }
  return result;
}

FiberQualitySelection
selectFiberletCropQuality(
    const std::vector<FiberletCropTraceLine>& lines,
    double fraction) {
  if (!std::isfinite(fraction) || !(fraction > 0.0) || fraction > 1.0) {
    throw std::invalid_argument(
        "Fiber quality fraction must be finite and in (0, 1]");
  }
  FiberQualitySelection result;
  result.inputLines = lines.size();
  result.requestedFraction = fraction;
  if (lines.empty()) {
    result.effectiveFraction = 0.0;
    return result;
  }
  const auto ranked = rankFiberletCropQuality(lines);
  const std::size_t retained = std::min(
      lines.size(),
      std::max<std::size_t>(
          1,
          static_cast<std::size_t>(std::ceil(
              fraction * static_cast<double>(lines.size())))));
  result.lineIndices.reserve(retained);
  for (std::size_t rank = 0; rank < retained; ++rank)
    result.lineIndices.push_back(ranked[rank].index);
  result.maximumRetainedCostDensity = ranked[retained - 1].density;
  std::sort(result.lineIndices.begin(), result.lineIndices.end());
  result.effectiveFraction = static_cast<double>(retained) /
      static_cast<double>(lines.size());
  return result;
}

FiberQualitySelection selectFiberletCropQualityThreshold(
    const std::vector<FiberletCropTraceLine>& lines,
    double maximumCostDensity) {
  if (!std::isfinite(maximumCostDensity) || maximumCostDensity < 0.0) {
    throw std::invalid_argument(
        "Fiber quality threshold must be finite and nonnegative");
  }
  FiberQualitySelection result;
  result.inputLines = lines.size();
  result.requestedMaximumCostDensity = maximumCostDensity;
  const auto ranked = rankFiberletCropQuality(lines);
  result.lineIndices.reserve(lines.size());
  for (const auto& line : ranked) {
    if (line.density > maximumCostDensity)
      break;
    result.lineIndices.push_back(line.index);
    result.maximumRetainedCostDensity = line.density;
  }
  std::sort(result.lineIndices.begin(), result.lineIndices.end());
  result.effectiveFraction = lines.empty()
      ? 0.0
      : static_cast<double>(result.lineIndices.size()) /
          static_cast<double>(lines.size());
  return result;
}

FiberQualityObjPaths
fiberQualityObjPaths(const std::filesystem::path &allOutputPath) {
  FiberQualityObjPaths result;
  for (std::size_t bin = 0; bin < result.deciles.size(); ++bin) {
    std::ostringstream suffix;
    suffix << "_quality_" << std::setw(2) << std::setfill('0') << bin * 10
           << '_' << std::setw(2) << (bin + 1) * 10;
    result.deciles[bin] = allOutputPath.parent_path() /
                          (allOutputPath.stem().string() + suffix.str() +
                           allOutputPath.extension().string());
  }
  result.histogramCsv =
      allOutputPath.parent_path() /
      (allOutputPath.stem().string() + "_quality_histogram.csv");
  return result;
}

void writeFiberletCropQualityArtifacts(
    const std::vector<FiberletCropTraceLine> &lines,
    const FiberQualityHistogram &histogram,
    const std::filesystem::path &allOutputPath) {
  const auto paths = fiberQualityObjPaths(allOutputPath);
  std::ostringstream csv;
  csv.imbue(std::locale::classic());
  csv << std::setprecision(17)
      << "percentile_start,percentile_end,count,total_cost_min,total_cost_mean,"
         "total_cost_max,cost_density_min,cost_density_mean,cost_density_max\n";
  for (std::size_t binIndex = 0; binIndex < histogram.bins.size(); ++binIndex) {
    const auto &bin = histogram.bins[binIndex];
    std::vector<vc::core::io::NamedPolyline> polylines;
    polylines.reserve(bin.lineIndices.size());
    for (const auto lineIndex : bin.lineIndices) {
      if (lineIndex >= lines.size())
        throw std::invalid_argument(
            "Fiber quality histogram contains an invalid line index");
      polylines.push_back({
          fiberName(lines[lineIndex], lineIndex),
          lines[lineIndex].pointsBaseXYZ,
      });
    }
    vc::core::io::writePolylinesObj(
        polylines, paths.deciles[binIndex],
        "VC3D Fiberlet crop traces: quality rank decile");
    csv << binIndex * 10 << ',' << (binIndex + 1) * 10 << ','
        << bin.lineIndices.size();
    if (bin.lineIndices.empty()) {
      csv << ",,,,,,\n";
    } else {
      csv << ',' << bin.minimumTotalMetricCost << ',' << bin.meanTotalMetricCost
          << ',' << bin.maximumTotalMetricCost << ',' << bin.minimumCostDensity
          << ',' << bin.meanCostDensity << ',' << bin.maximumCostDensity
          << '\n';
    }
  }
  vc::core::util::atomicWriteString(paths.histogramCsv, csv.str());
}

FiberValueBands classifyFiberValues(std::span<const double> values)
{
    FiberValueBands result;
    std::array<double, 10> sums{};
    for (std::size_t index = 0; index < values.size(); ++index) {
        const double input = values[index];
        if (!std::isfinite(input) ||
            input < -kProbabilityRoundoffTolerance ||
            input > 1.0 + kProbabilityRoundoffTolerance) {
            std::ostringstream message;
            message << std::setprecision(17)
                    << "Fiber value band input " << index
                    << " must be finite and in [0, 1] up to roundoff; value="
                    << input;
            throw std::invalid_argument(message.str());
        }
        const double value = std::clamp(input, 0.0, 1.0);
        const std::size_t band = std::min<std::size_t>(
            9, static_cast<std::size_t>(std::floor(value * 10.0)));
        auto& current = result.bands[band];
        if (current.lineIndices.empty()) {
            current.minimumValue = value;
            current.maximumValue = value;
        } else {
            current.minimumValue = std::min(current.minimumValue, value);
            current.maximumValue = std::max(current.maximumValue, value);
        }
        current.lineIndices.push_back(index);
        sums[band] += value;
    }
    for (std::size_t band = 0; band < result.bands.size(); ++band) {
        auto& current = result.bands[band];
        if (!current.lineIndices.empty()) {
            current.meanValue = sums[band] /
                static_cast<double>(current.lineIndices.size());
        }
    }
    return result;
}

FiberValueBandObjPaths fiberValueBandObjPaths(
    const std::filesystem::path& outputBase)
{
    FiberValueBandObjPaths result;
    for (std::size_t band = 0; band < result.bands.size(); ++band) {
        result.bands[band] = outputBase.parent_path() /
            (outputBase.stem().string() + "_p" + std::to_string(band) +
             ".obj");
    }
    return result;
}

FiberValueBandObjPaths writeFiberletCropValueBandObjs(
    const std::vector<FiberletCropTraceLine>& lines,
    const FiberValueBands& bands,
    const std::filesystem::path& outputBase)
{
    const auto paths = fiberValueBandObjPaths(outputBase);
    std::vector<unsigned char> written(lines.size(), 0);
    for (std::size_t band = 0; band < bands.bands.size(); ++band) {
        std::vector<vc::core::io::NamedPolyline> polylines;
        polylines.reserve(bands.bands[band].lineIndices.size());
        for (const std::size_t index : bands.bands[band].lineIndices) {
            if (index >= lines.size() || written[index] != 0) {
                throw std::invalid_argument(
                    "Fiber value bands do not partition crop traces");
            }
            written[index] = 1;
            polylines.push_back({
                fiberName(lines[index], index), lines[index].pointsBaseXYZ});
        }
        vc::core::io::writePolylinesObj(
            polylines,
            paths.bands[band],
            "VC3D Fiberlet crop traces: post-filter H-value band");
    }
    if (std::find(written.begin(), written.end(), 0) != written.end()) {
        throw std::invalid_argument(
            "Fiber value bands do not partition crop traces");
    }
    return paths;
}

FiberTernaryStateObjPaths writeFiberletCropTernaryStateObjs(
    const std::vector<FiberletCropTraceLine>& lines,
    std::span<const FiberTernaryState> states,
    const std::filesystem::path& outputBase)
{
    if (states.size() != lines.size()) {
        throw std::invalid_argument(
            "Fiber ternary states do not match crop traces");
    }
    const auto sibling = [&] (const char* suffix) {
        return outputBase.parent_path() /
            (outputBase.stem().string() + suffix + ".obj");
    };
    const FiberTernaryStateObjPaths paths{
        sibling("_v"),
        sibling("_err"),
        sibling("_h"),
        sibling("_tie"),
    };
    std::array<std::vector<vc::core::io::NamedPolyline>, 4> grouped;
    for (std::size_t index = 0; index < lines.size(); ++index) {
        grouped[static_cast<std::size_t>(states[index])].push_back({
            fiberName(lines[index], index), lines[index].pointsBaseXYZ});
    }
    constexpr std::array<const char*, 4> comments{
        "VC3D Fiberlet crop traces: BP vertical argmax",
        "VC3D Fiberlet crop traces: BP error/Mixed argmax",
        "VC3D Fiberlet crop traces: BP horizontal argmax",
        "VC3D Fiberlet crop traces: BP exact argmax tie",
    };
    const std::array outputPaths{
        paths.vertical, paths.mixed, paths.horizontal, paths.tie};
    for (std::size_t group = 0; group < grouped.size(); ++group) {
        vc::core::io::writePolylinesObj(
            grouped[group], outputPaths[group], comments[group]);
    }
    return paths;
}

}  // namespace vc::fiber_tracer
