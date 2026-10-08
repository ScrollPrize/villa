#include "vc/fiber_tracer/FiberSkeletonChoice.hpp"

#include "vc/fiber_tracer/FiberLocalScoring.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <queue>
#include <set>
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

cv::Vec3f toFloat(const cv::Vec3d& value)
{
    return {static_cast<float>(value[0]), static_cast<float>(value[1]),
            static_cast<float>(value[2])};
}

double pointToPolylineDistance(
    const cv::Vec3d& point, const std::vector<cv::Vec3d>& polyline)
{
    if (polyline.empty())
        return std::numeric_limits<double>::infinity();
    double best = cv::norm(point - polyline.front());
    for (std::size_t index = 1; index < polyline.size(); ++index) {
        const cv::Vec3d delta = polyline[index] - polyline[index - 1];
        const double lengthSquared = delta.dot(delta);
        const double t = lengthSquared > kEpsilon
            ? std::clamp((point - polyline[index - 1]).dot(delta) /
                         lengthSquared, 0.0, 1.0)
            : 0.0;
        best = std::min(
            best, cv::norm(point - (polyline[index - 1] + delta * t)));
    }
    return best;
}

// An edge's polyline, oriented to start at `fromNode`.
std::vector<cv::Vec3d> orientedEdge(
    const FiberSkeletonEdge& edge, std::size_t fromNode)
{
    auto points = edge.pointsBaseXYZ;
    if (edge.nodeIds[0] != fromNode)
        std::reverse(points.begin(), points.end());
    return points;
}

// The stub from a port back to the region node its edge leaves from.
std::vector<cv::Vec3d> portToNode(
    const FiberSkeletonGraph& graph, const FiberSkeletonPort& port,
    std::size_t slot)
{
    const auto& edge = graph.edges[port.edgeId];
    const std::size_t node = port.nodeIds[slot];
    const double fromNode = port.arcFromNodeBase[slot];
    std::vector<cv::Vec3d> stub;
    if (edge.nodeIds[0] == node) {
        stub = fiberPolylineSlice(
            edge.pointsBaseXYZ, 0.0, std::min(fromNode, edge.lengthBase));
        std::reverse(stub.begin(), stub.end());
    } else {
        stub = fiberPolylineSlice(
            edge.pointsBaseXYZ,
            std::max(0.0, edge.lengthBase - fromNode), edge.lengthBase);
    }
    return stub;
}

void appendPolyline(
    std::vector<cv::Vec3d>& out, const std::vector<cv::Vec3d>& points)
{
    for (const auto& point : points) {
        if (out.empty() || cv::norm(out.back() - point) > kEpsilon)
            out.push_back(point);
    }
}

// The shared discrete metric summed along a polyline. This is the initial loss
// that selects a route through a region before any refinement.
double polylineDiscreteCost(
    const std::vector<cv::Vec3d>& points, FiberSkeletonFieldProvider& fields,
    const FiberLocalMetricConfig& config)
{
    if (points.size() < 3)
        return 0.0;
    std::vector<cv::Vec3d> references;
    references.reserve(points.size());
    for (std::size_t index = 0; index < points.size(); ++index) {
        const std::size_t back = index == 0 ? 0 : index - 1;
        const std::size_t front =
            index + 1 < points.size() ? index + 1 : points.size() - 1;
        cv::Vec3d reference = normalizedOrZero(points[front] - points[back]);
        if (reference.dot(reference) <= 0.0)
            reference = {1.0, 0.0, 0.0};
        references.push_back(reference);
    }
    std::vector<FiberSkeletonFieldSample> samples;
    sampleFiberSkeletonFields(fields, points, references, samples);

    double total = 0.0;
    for (std::size_t index = 1; index + 1 < points.size(); ++index) {
        const cv::Vec3d previous = points[index] - points[index - 1];
        const cv::Vec3d candidate = points[index + 1] - points[index];
        const double previousLength = cv::norm(previous);
        const double candidateLength = cv::norm(candidate);
        if (!(previousLength > kEpsilon) || !(candidateLength > kEpsilon))
            continue;
        FiberLocalMetricSample current;
        current.direction = toFloat(samples[index].direction);
        current.presence = static_cast<float>(samples[index].presence);
        current.valid = samples[index].valid;
        FiberLocalMetricSample next;
        next.direction = toFloat(samples[index + 1].direction);
        next.presence = static_cast<float>(samples[index + 1].presence);
        next.valid = samples[index + 1].valid;
        total += static_cast<double>(
            fiberLocalMetricCost(
                &current, next, toFloat(normalizedOrZero(previous)),
                static_cast<float>(previousLength),
                toFloat(normalizedOrZero(candidate)),
                static_cast<float>(candidateLength),
                toFloat(samples[index + 1].normal),
                samples[index + 1].normalValid, config)
                .total());
    }
    return total;
}

}  // namespace

std::vector<std::vector<std::size_t>> enumerateFiberSkeletonMatchings(
    std::size_t portCount)
{
    std::vector<std::vector<std::size_t>> matchings;
    std::vector<std::size_t> partner(portCount, kFiberSkeletonNoIndex);
    // Always resolve the lowest unassigned port first, so the enumeration is
    // deterministic and lexicographic in the partner table.
    const auto recurse = [&](auto&& self, std::size_t first) -> void {
        while (first < portCount && partner[first] != kFiberSkeletonNoIndex)
            ++first;
        if (first >= portCount) {
            matchings.push_back(partner);
            return;
        }
        self(self, first + 1);
        for (std::size_t second = first + 1; second < portCount; ++second) {
            if (partner[second] != kFiberSkeletonNoIndex)
                continue;
            partner[first] = second;
            partner[second] = first;
            self(self, first + 1);
            partner[first] = kFiberSkeletonNoIndex;
            partner[second] = kFiberSkeletonNoIndex;
        }
    };
    recurse(recurse, 0);
    return matchings;
}

std::vector<FiberSkeletonRegionChoice> solveFiberSkeletonChoices(
    const FiberSkeletonTopology& topology, const FiberSkeletonGraph& graph,
    FiberSkeletonFieldProvider& fields, const FiberSkeletonCostWeights& weights,
    const FiberSkeletonSolverConfig& solver,
    const FiberSkeletonChoiceConfig& choiceConfig, double stepBase)
{
    if (!(stepBase > 0.0) || !std::isfinite(stepBase))
        throw std::invalid_argument("choice step must be finite and positive");
    if (choiceConfig.maxPorts < 2)
        throw std::invalid_argument(
            "skeleton choice port limit must be at least two");

    FiberLocalMetricConfig metricConfig;
    metricConfig.smoothness.isotropicWeight =
        static_cast<float>(weights.isotropic);
    metricConfig.smoothness.normalWeight = static_cast<float>(weights.normal);
    metricConfig.smoothness.tangentWeight = static_cast<float>(weights.tangent);
    metricConfig.smoothness.freeAngleRadians = 0.0f;

    const auto candidateCount = [](const FiberSkeletonPort& port) {
        return std::max<std::size_t>(1, port.candidatesBaseXYZ.size());
    };
    const auto candidateAt = [](const FiberSkeletonPort& port, std::size_t index) {
        return port.candidatesBaseXYZ.empty() ? port.positionBaseXYZ
                                              : port.candidatesBaseXYZ[index];
    };

    std::vector<FiberSkeletonRegionChoice> results;
    results.reserve(topology.regions.size());

    for (const auto& region : topology.regions) {
        const auto start = std::chrono::steady_clock::now();
        FiberSkeletonRegionChoice choice;
        choice.regionId = region.id;
        choice.portCount = region.portIds.size();
        for (std::size_t slot = 0; slot < region.portIds.size(); ++slot) {
            const auto& port = topology.ports[region.portIds[slot]];
            for (std::size_t candidate = 0; candidate < candidateCount(port);
                 ++candidate) {
                choice.items.push_back({slot, candidate});
            }
        }
        if (choice.items.size() > choiceConfig.maxPorts) {
            choice.portLimitExceeded = true;
            choice.solveSeconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count();
            results.push_back(std::move(choice));
            continue;
        }

        // The region routes over its own nodes and interior edges, which are
        // ordinary graph elements, so a connection is a path in the graph.
        std::map<std::size_t, std::size_t> slotOfNode;
        for (std::size_t index = 0; index < region.nodeIds.size(); ++index)
            slotOfNode.emplace(region.nodeIds[index], index);
        std::vector<std::vector<std::pair<std::size_t, std::size_t>>> adjacency(
            region.nodeIds.size());
        std::map<std::size_t, double> edgeCost;
        for (const auto edgeId : region.interiorEdgeIds) {
            const auto& edge = graph.edges[edgeId];
            const auto first = slotOfNode.find(edge.nodeIds[0]);
            const auto second = slotOfNode.find(edge.nodeIds[1]);
            if (first == slotOfNode.end() || second == slotOfNode.end())
                continue;
            edgeCost.emplace(
                edgeId,
                polylineDiscreteCost(edge.pointsBaseXYZ, fields, metricConfig) +
                    edge.lengthBase * kEpsilon);
            adjacency[first->second].emplace_back(second->second, edgeId);
            if (first->second != second->second)
                adjacency[second->second].emplace_back(first->second, edgeId);
        }

        // Cheapest interior route between two of the region's nodes.
        const auto routeBetween = [&](std::size_t fromNode, std::size_t toNode) {
            std::vector<std::size_t> edges;
            const auto source = slotOfNode.find(fromNode);
            const auto target = slotOfNode.find(toNode);
            if (source == slotOfNode.end() || target == slotOfNode.end() ||
                source->second == target->second) {
                return edges;
            }
            const std::size_t count = region.nodeIds.size();
            std::vector<double> distance(
                count, std::numeric_limits<double>::infinity());
            std::vector<std::size_t> cameFromNode(count, kFiberSkeletonNoIndex);
            std::vector<std::size_t> cameFromEdge(count, kFiberSkeletonNoIndex);
            using Entry = std::pair<double, std::size_t>;
            std::priority_queue<Entry, std::vector<Entry>, std::greater<>> queue;
            distance[source->second] = 0.0;
            queue.emplace(0.0, source->second);
            while (!queue.empty()) {
                const auto [cost, slot] = queue.top();
                queue.pop();
                if (cost > distance[slot] + kEpsilon)
                    continue;
                if (slot == target->second)
                    break;
                for (const auto& [next, edgeId] : adjacency[slot]) {
                    const double step = cost + edgeCost.at(edgeId);
                    if (step + kEpsilon < distance[next]) {
                        distance[next] = step;
                        cameFromNode[next] = slot;
                        cameFromEdge[next] = edgeId;
                        queue.emplace(step, next);
                    }
                }
            }
            if (!std::isfinite(distance[target->second]))
                return edges;
            for (std::size_t slot = target->second; slot != source->second;) {
                edges.push_back(cameFromEdge[slot]);
                const std::size_t previous = cameFromNode[slot];
                if (previous == kFiberSkeletonNoIndex) {
                    edges.clear();
                    return edges;
                }
                slot = previous;
            }
            std::reverse(edges.begin(), edges.end());
            return edges;
        };

        const auto regionSlot = [&](const FiberSkeletonPort& port) {
            return port.regionIds[0] == region.id ? std::size_t{0}
                                                  : std::size_t{1};
        };

        // The route through the region between two ports, before any
        // candidate is applied; found once per port pair.
        struct PortPairRoute {
            bool reachable = false;
            std::vector<cv::Vec3d> points;
        };
        std::map<std::pair<std::size_t, std::size_t>, PortPairRoute> routes;
        const auto routeFor = [&](std::size_t first, std::size_t second)
            -> const PortPairRoute& {
            const auto key = std::make_pair(first, second);
            const auto found = routes.find(key);
            if (found != routes.end())
                return found->second;
            const auto& fromPort = topology.ports[region.portIds[first]];
            const auto& toPort = topology.ports[region.portIds[second]];
            const std::size_t fromSlot = regionSlot(fromPort);
            const std::size_t toSlot = regionSlot(toPort);
            const std::size_t fromNode = fromPort.nodeIds[fromSlot];
            const std::size_t toNode = toPort.nodeIds[toSlot];
            PortPairRoute route;
            appendPolyline(route.points, portToNode(graph, fromPort, fromSlot));
            route.reachable = fromNode == toNode;
            if (!route.reachable) {
                const auto path = routeBetween(fromNode, toNode);
                route.reachable = !path.empty();
                std::size_t current = fromNode;
                for (const auto edgeId : path) {
                    const auto& edge = graph.edges[edgeId];
                    appendPolyline(route.points, orientedEdge(edge, current));
                    current = edge.nodeIds[0] == current ? edge.nodeIds[1]
                                                         : edge.nodeIds[0];
                }
            }
            auto tail = portToNode(graph, toPort, toSlot);
            std::reverse(tail.begin(), tail.end());
            appendPolyline(route.points, tail);
            route.reachable = route.reachable && route.points.size() >= 2;
            return routes.emplace(key, std::move(route)).first->second;
        };

        // One connection per pair of items on different ports, each refined
        // between its own two candidate positions. All of them are kept.
        for (std::size_t first = 0; first < choice.items.size(); ++first) {
            for (std::size_t second = first + 1; second < choice.items.size();
                 ++second) {
                const auto& fromItem = choice.items[first];
                const auto& toItem = choice.items[second];
                if (fromItem.slot == toItem.slot)
                    continue;
                const auto& fromPort = topology.ports[region.portIds[fromItem.slot]];
                const auto& toPort = topology.ports[region.portIds[toItem.slot]];
                FiberSkeletonConnection connection;
                connection.id = choice.connections.size();
                connection.itemIds = {first, second};
                connection.portIds = {fromItem.slot, toItem.slot};
                connection.candidates = {fromItem.candidate, toItem.candidate};

                const auto& route = routeFor(fromItem.slot, toItem.slot);
                connection.available = route.reachable;
                if (!connection.available) {
                    // No route through the region's own graph: the pair is not
                    // connectable, and nothing is invented in its place.
                    choice.connections.push_back(std::move(connection));
                    continue;
                }

                FiberSkeletonRefineConfig refineConfig;
                refineConfig.weights = weights;
                refineConfig.solver = solver;
                const cv::Vec3d fromTangent =
                    normalizedOrZero(fromPort.tangentXYZ);
                const cv::Vec3d toTangent = normalizedOrZero(toPort.tangentXYZ);
                if (fromTangent.dot(fromTangent) > 0.0)
                    refineConfig.startTangent = -fromTangent;
                if (toTangent.dot(toTangent) > 0.0)
                    refineConfig.endTangent = toTangent;

                if (!fiberSkeletonSearchPath(
                        fields, choiceConfig.search, route.points,
                        candidateAt(fromPort, fromItem.candidate),
                        candidateAt(toPort, toItem.candidate),
                        connection.initialPointsBaseXYZ)) {
                    connection.initialPointsBaseXYZ = fiberSkeletonBlendEnds(
                        route.points, candidateAt(fromPort, fromItem.candidate),
                        candidateAt(toPort, toItem.candidate));
                    connection.searchedInitial = false;
                } else {
                    connection.searchedInitial = true;
                }
                const double initialLength =
                    fiberPolylineLength(connection.initialPointsBaseXYZ);
                if (initialLength > stepBase) {
                    const auto result = refineFixedEndpointPolyline(
                        resampleFiberPolyline(
                            connection.initialPointsBaseXYZ, stepBase),
                        fields, refineConfig);
                    connection.pointsBaseXYZ = result.pointsBaseXYZ;
                    connection.cost = result.finalCost;
                    connection.lengthBase = result.lengthBase;
                    connection.refined = true;
                    connection.iterations = result.iterations;
                    connection.converged = result.converged;
                } else {
                    connection.pointsBaseXYZ = connection.initialPointsBaseXYZ;
                    connection.cost = evaluateFiberSkeletonPolylineCost(
                        connection.pointsBaseXYZ, fields, refineConfig);
                    connection.lengthBase = initialLength;
                }
                connection.costTotal = connection.cost.total();
                choice.connections.push_back(std::move(connection));
            }
        }

        // Each item's cheapest connection: the route a fibre takes when it
        // arrives on that sub-port, from whichever side.
        choice.entryConnections.assign(choice.items.size(), kFiberSkeletonNoIndex);
        const auto isFibreItem = [&](std::size_t item) {
            const auto& entry = choice.items[item];
            const auto& port = topology.ports[region.portIds[entry.slot]];
            return entry.candidate >= port.candidateIsFibre.size() ||
                port.candidateIsFibre[entry.candidate] != 0;
        };
        for (const auto& connection : choice.connections) {
            if (!connection.available)
                continue;
            if (!choiceConfig.routeThroughCentre &&
                (!isFibreItem(connection.itemIds[0]) ||
                 !isFibreItem(connection.itemIds[1]))) {
                continue;
            }
            for (const auto item : connection.itemIds) {
                const auto current = choice.entryConnections[item];
                if (current == kFiberSkeletonNoIndex ||
                    connection.costTotal <
                        choice.connections[current].costTotal - kEpsilon) {
                    choice.entryConnections[item] = connection.id;
                }
            }
        }

        choice.solveSeconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        results.push_back(std::move(choice));
    }
    return results;
}

}  // namespace vc::fiber_tracer
