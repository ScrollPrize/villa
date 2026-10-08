#include "vc/fiber_tracer/FiberSkeleton.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

#include "vc/core/io/PolylineObj.hpp"
#include "vc/fiber_tracer/PolylineGeometry.hpp"

namespace vc::fiber_tracer
{
namespace
{

constexpr double kEpsilon = 1.0e-12;
constexpr std::size_t kNoIndex = std::numeric_limits<std::size_t>::max();

// Defensive bound on uniform resampling so that a tiny step cannot turn a long
// polyline into an unbounded allocation (or overflow the segment count).
constexpr double kMaxResampleSegments = 1.0e9;

// Length of the degenerate-free two-point marker written per graph node.
constexpr double kNodeMarkerLengthBase = 0.5;

// The 26 neighbor offsets as (dz, dy, dx), in the raster order used by the
// Python pair-connectivity reference (itertools.product((-1, 0, 1), repeat=3)).
constexpr std::array<std::array<int, 3>, 26> makeNeighborOffsets() noexcept
{
    std::array<std::array<int, 3>, 26> offsets{};
    std::size_t index = 0;
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (dz == 0 && dy == 0 && dx == 0)
                    continue;
                offsets[index++] = {dz, dy, dx};
            }
        }
    }
    return offsets;
}

constexpr auto kNeighborOffsets = makeNeighborOffsets();

// Foreground 26-neighborhood lookups on a dense 0/1 mask.
class MaskNeighborhood
{
public:
    MaskNeighborhood(
        const std::array<std::size_t, 3>& shapeZYX,
        const std::vector<std::uint8_t>& mask)
        : shape_(shapeZYX), mask_(mask)
    {
    }

    void collect(std::size_t flat, std::vector<std::size_t>& neighbors) const
    {
        neighbors.clear();
        const std::size_t sizeZ = shape_[0];
        const std::size_t sizeY = shape_[1];
        const std::size_t sizeX = shape_[2];
        const std::size_t x = flat % sizeX;
        const std::size_t y = (flat / sizeX) % sizeY;
        const std::size_t z = flat / (sizeX * sizeY);
        for (const auto& delta : kNeighborOffsets) {
            const std::ptrdiff_t nz =
                static_cast<std::ptrdiff_t>(z) + delta[0];
            const std::ptrdiff_t ny =
                static_cast<std::ptrdiff_t>(y) + delta[1];
            const std::ptrdiff_t nx =
                static_cast<std::ptrdiff_t>(x) + delta[2];
            if (nz < 0 || ny < 0 || nx < 0 ||
                nz >= static_cast<std::ptrdiff_t>(sizeZ) ||
                ny >= static_cast<std::ptrdiff_t>(sizeY) ||
                nx >= static_cast<std::ptrdiff_t>(sizeX)) {
                continue;
            }
            const std::size_t neighbor =
                (static_cast<std::size_t>(nz) * sizeY +
                 static_cast<std::size_t>(ny)) *
                    sizeX +
                static_cast<std::size_t>(nx);
            if (mask_[neighbor] != 0)
                neighbors.push_back(neighbor);
        }
    }

private:
    std::array<std::size_t, 3> shape_;
    const std::vector<std::uint8_t>& mask_;
};

std::pair<std::size_t, std::size_t> orderedPair(
    std::size_t left, std::size_t right)
{
    return left <= right ? std::pair{left, right} : std::pair{right, left};
}

const char* nodeKindName(FiberSkeletonNodeKind kind) noexcept
{
    switch (kind) {
        case FiberSkeletonNodeKind::Terminal:
            return "terminal";
        case FiberSkeletonNodeKind::Junction:
            return "junction";
        case FiberSkeletonNodeKind::Other:
            break;
    }
    return "other";
}

FiberSkeletonNodeKind nodeKindForDegree(std::size_t degree) noexcept
{
    if (degree == 1)
        return FiberSkeletonNodeKind::Terminal;
    if (degree >= 3)
        return FiberSkeletonNodeKind::Junction;
    return FiberSkeletonNodeKind::Other;
}

std::size_t findRoot(std::vector<std::size_t>& parent, std::size_t value)
{
    while (parent[value] != value) {
        parent[value] = parent[parent[value]];
        value = parent[value];
    }
    return value;
}

// Rebuilds `incidentEdges`, `degree` and `kind` of every live node from the
// live edges. A self-loop contributes its edge id twice.
void refreshIncidence(FiberSkeletonGraph& graph)
{
    for (auto& node : graph.nodes) {
        node.incidentEdges.clear();
        if (node.removed) {
            node.degree = 0;
            node.kind = FiberSkeletonNodeKind::Other;
        }
    }
    for (std::size_t edgeId = 0; edgeId < graph.edges.size(); ++edgeId) {
        const auto& edge = graph.edges[edgeId];
        if (edge.removed)
            continue;
        for (const std::size_t nodeId : edge.nodeIds) {
            if (nodeId >= graph.nodes.size() || graph.nodes[nodeId].removed) {
                throw std::invalid_argument(
                    "fiber skeleton edge references a missing node");
            }
            graph.nodes[nodeId].incidentEdges.push_back(edgeId);
        }
    }
    for (auto& node : graph.nodes) {
        if (node.removed)
            continue;
        node.degree = node.incidentEdges.size();
        node.kind = nodeKindForDegree(node.degree);
    }
}

// Connected components of the live node/edge graph.
std::size_t graphComponentCount(const FiberSkeletonGraph& graph)
{
    std::vector<std::size_t> parent(graph.nodes.size());
    for (std::size_t index = 0; index < parent.size(); ++index)
        parent[index] = index;
    for (const auto& edge : graph.edges) {
        if (edge.removed)
            continue;
        const std::size_t left = findRoot(parent, edge.nodeIds[0]);
        const std::size_t right = findRoot(parent, edge.nodeIds[1]);
        if (left != right)
            parent[std::max(left, right)] = std::min(left, right);
    }
    std::set<std::size_t> roots;
    for (std::size_t index = 0; index < graph.nodes.size(); ++index) {
        if (!graph.nodes[index].removed)
            roots.insert(findRoot(parent, index));
    }
    return roots.size();
}

// max(0, edges - nodes + components) over the live graph.
std::size_t structuralCycleRank(
    std::size_t nodeCount, std::size_t edgeCount, std::size_t componentCount)
{
    const std::ptrdiff_t rank = static_cast<std::ptrdiff_t>(edgeCount) -
        static_cast<std::ptrdiff_t>(nodeCount) +
        static_cast<std::ptrdiff_t>(componentCount);
    return rank > 0 ? static_cast<std::size_t>(rank) : 0;
}

// Recomputes the derived graph statistics. `extraCycles` carries the cycles of
// skeleton components that hold no node at all, which the node/edge graph
// cannot express.
void refreshGraphStatistics(FiberSkeletonGraph& graph, std::size_t extraCycles)
{
    refreshIncidence(graph);
    std::size_t nodeCount = 0;
    std::size_t terminalNodes = 0;
    std::size_t junctionNodes = 0;
    for (const auto& node : graph.nodes) {
        if (node.removed)
            continue;
        ++nodeCount;
        if (node.kind == FiberSkeletonNodeKind::Terminal)
            ++terminalNodes;
        else if (node.kind == FiberSkeletonNodeKind::Junction)
            ++junctionNodes;
    }
    std::size_t edgeCount = 0;
    double totalLength = 0.0;
    for (const auto& edge : graph.edges) {
        if (edge.removed)
            continue;
        ++edgeCount;
        totalLength += edge.lengthBase;
    }
    const std::size_t components = graphComponentCount(graph);
    graph.statistics.graphNodes = nodeCount;
    graph.statistics.graphEdges = edgeCount;
    graph.statistics.terminalNodes = terminalNodes;
    graph.statistics.junctionNodes = junctionNodes;
    graph.statistics.graphComponents = components;
    graph.statistics.cycleRank =
        structuralCycleRank(nodeCount, edgeCount, components) + extraCycles;
    graph.statistics.totalEdgeLengthBase = totalLength;
}

void reverseEdge(FiberSkeletonEdge& edge)
{
    std::swap(edge.nodeIds[0], edge.nodeIds[1]);
    std::reverse(edge.pointsBaseXYZ.begin(), edge.pointsBaseXYZ.end());
}

void orientEdgeEndingAt(FiberSkeletonEdge& edge, std::size_t nodeId)
{
    if (edge.nodeIds[1] == nodeId)
        return;
    if (edge.nodeIds[0] != nodeId) {
        throw std::invalid_argument(
            "fiber skeleton edge is not incident to the requested node");
    }
    reverseEdge(edge);
}

void orientEdgeStartingAt(FiberSkeletonEdge& edge, std::size_t nodeId)
{
    if (edge.nodeIds[0] == nodeId)
        return;
    if (edge.nodeIds[1] != nodeId) {
        throw std::invalid_argument(
            "fiber skeleton edge is not incident to the requested node");
    }
    reverseEdge(edge);
}

void pruneShortSpurs(FiberSkeletonGraph& graph, double spurPruneMaxBase)
{
    while (true) {
        refreshIncidence(graph);
        std::size_t chosenEdge = kNoIndex;
        std::size_t chosenTerminal = kNoIndex;
        for (std::size_t edgeId = 0; edgeId < graph.edges.size(); ++edgeId) {
            const auto& edge = graph.edges[edgeId];
            if (edge.removed)
                continue;
            const std::size_t first = edge.nodeIds[0];
            const std::size_t second = edge.nodeIds[1];
            if (first == second)
                continue;
            const std::size_t firstDegree = graph.nodes[first].degree;
            const std::size_t secondDegree = graph.nodes[second].degree;
            std::size_t terminal = kNoIndex;
            if (firstDegree == 1 && secondDegree >= 3)
                terminal = first;
            else if (secondDegree == 1 && firstDegree >= 3)
                terminal = second;
            else
                continue;
            if (!(edge.lengthBase <= spurPruneMaxBase))
                continue;
            chosenEdge = edgeId;
            chosenTerminal = terminal;
            break;
        }
        if (chosenEdge == kNoIndex)
            return;
        graph.edges[chosenEdge].removed = true;
        graph.nodes[chosenTerminal].removed = true;
        ++graph.statistics.prunedSpurs;
    }
}

// Joins the two edges of `nodeId` into the first one and drops both the node
// and the second edge. The node contributes exactly one polyline point, which
// both edges carry, so dropping it once from the concatenation keeps the joined
// polyline passing through every skeleton voxel exactly once.
void suppressDegreeTwoNode(FiberSkeletonGraph& graph, std::size_t nodeId)
{
    auto& node = graph.nodes[nodeId];
    const std::size_t firstId = node.incidentEdges[0];
    const std::size_t secondId = node.incidentEdges[1];
    orientEdgeEndingAt(graph.edges[firstId], nodeId);
    orientEdgeStartingAt(graph.edges[secondId], nodeId);
    auto& first = graph.edges[firstId];
    auto& second = graph.edges[secondId];
    if (first.pointsBaseXYZ.size() < 2 || second.pointsBaseXYZ.size() < 2) {
        throw std::invalid_argument(
            "fiber skeleton edge polyline requires at least two points");
    }

    first.nodeIds[1] = second.nodeIds[1];
    first.pointsBaseXYZ.insert(
        first.pointsBaseXYZ.end(), second.pointsBaseXYZ.begin() + 1,
        second.pointsBaseXYZ.end());
    first.lengthBase += second.lengthBase;

    second.removed = true;
    node.removed = true;
}

void suppressDegreeTwoNodes(FiberSkeletonGraph& graph)
{
    while (true) {
        refreshIncidence(graph);
        std::size_t chosen = kNoIndex;
        for (std::size_t nodeId = 0; nodeId < graph.nodes.size(); ++nodeId) {
            const auto& node = graph.nodes[nodeId];
            if (node.removed || node.incidentEdges.size() != 2)
                continue;
            if (node.incidentEdges[0] == node.incidentEdges[1])
                continue;
            chosen = nodeId;
            break;
        }
        if (chosen == kNoIndex)
            return;
        suppressDegreeTwoNode(graph, chosen);
        ++graph.statistics.suppressedDegreeTwoNodes;
    }
}

// Labels every junction node with the ambiguous region it belongs to. Junction
// nodes joined by an edge no longer than `junctionMergeMaxBase` share a group;
// every remaining junction is a region of its own. This is a labelling only: no
// node, edge or incidence is touched, so the topology stays intact. Requires
// up-to-date `degree` and `kind`.
void groupJunctionRegions(
    FiberSkeletonGraph& graph, double junctionMergeMaxBase)
{
    std::vector<std::size_t> parent(graph.nodes.size());
    for (std::size_t index = 0; index < parent.size(); ++index)
        parent[index] = index;
    for (const auto& edge : graph.edges) {
        if (edge.removed)
            continue;
        const std::size_t left = edge.nodeIds[0];
        const std::size_t right = edge.nodeIds[1];
        if (graph.nodes[left].degree < 3 || graph.nodes[right].degree < 3)
            continue;
        if (!(edge.lengthBase <= junctionMergeMaxBase))
            continue;
        const std::size_t leftRoot = findRoot(parent, left);
        const std::size_t rightRoot = findRoot(parent, right);
        if (leftRoot != rightRoot)
            parent[std::max(leftRoot, rightRoot)] = std::min(leftRoot, rightRoot);
    }

    for (auto& node : graph.nodes)
        node.regionGroup = kFiberSkeletonNoGroup;

    // Dense group ids in ascending node order, so the numbering is stable and
    // survives the identity-preserving renumbering of `compact()`.
    std::vector<std::size_t> groupOfRoot(
        graph.nodes.size(), kFiberSkeletonNoGroup);
    std::vector<std::size_t> groupSizes;
    for (std::size_t nodeId = 0; nodeId < graph.nodes.size(); ++nodeId) {
        auto& node = graph.nodes[nodeId];
        if (node.removed || node.kind != FiberSkeletonNodeKind::Junction)
            continue;
        const std::size_t root = findRoot(parent, nodeId);
        if (groupOfRoot[root] == kFiberSkeletonNoGroup) {
            groupOfRoot[root] = groupSizes.size();
            groupSizes.push_back(0);
        }
        node.regionGroup = groupOfRoot[root];
        ++groupSizes[node.regionGroup];
    }

    std::size_t groupedJunctions = 0;
    for (const std::size_t size : groupSizes) {
        if (size >= 2)
            groupedJunctions += size;
    }
    graph.statistics.regionGroups = groupSizes.size();
    graph.statistics.groupedJunctions = groupedJunctions;
}

}  // namespace

void FiberSkeletonGraph::compact()
{
    std::vector<std::size_t> nodeMap(nodes.size(), kNoIndex);
    std::size_t nextNode = 0;
    for (std::size_t index = 0; index < nodes.size(); ++index) {
        if (!nodes[index].removed)
            nodeMap[index] = nextNode++;
    }
    std::vector<std::size_t> edgeMap(edges.size(), kNoIndex);
    std::size_t nextEdge = 0;
    for (std::size_t index = 0; index < edges.size(); ++index) {
        if (!edges[index].removed)
            edgeMap[index] = nextEdge++;
    }

    std::vector<FiberSkeletonNode> keptNodes;
    keptNodes.reserve(nextNode);
    for (std::size_t index = 0; index < nodes.size(); ++index) {
        if (nodes[index].removed)
            continue;
        FiberSkeletonNode node = std::move(nodes[index]);
        node.id = nodeMap[index];
        std::vector<std::size_t> incident;
        incident.reserve(node.incidentEdges.size());
        for (const std::size_t edgeId : node.incidentEdges) {
            if (edgeId < edgeMap.size() && edgeMap[edgeId] != kNoIndex)
                incident.push_back(edgeMap[edgeId]);
        }
        node.incidentEdges = std::move(incident);
        keptNodes.push_back(std::move(node));
    }

    std::vector<FiberSkeletonEdge> keptEdges;
    keptEdges.reserve(nextEdge);
    for (std::size_t index = 0; index < edges.size(); ++index) {
        if (edges[index].removed)
            continue;
        FiberSkeletonEdge edge = std::move(edges[index]);
        edge.id = edgeMap[index];
        for (auto& nodeId : edge.nodeIds) {
            if (nodeId >= nodeMap.size() || nodeMap[nodeId] == kNoIndex) {
                throw std::invalid_argument(
                    "fiber skeleton edge references a removed node");
            }
            nodeId = nodeMap[nodeId];
        }
        keptEdges.push_back(std::move(edge));
    }

    nodes = std::move(keptNodes);
    edges = std::move(keptEdges);
}

FiberSkeletonGraph buildFiberSkeletonGraph(
    const FiberSkeletonCrop& crop, const std::vector<std::uint8_t>& skeleton)
{
    const std::size_t total = crop.voxelCount();
    if (skeleton.size() != total) {
        throw std::invalid_argument(
            "skeleton mask size does not match the crop shape");
    }

    FiberSkeletonGraph graph;
    std::vector<std::uint8_t> mask(total, 0);
    std::size_t skeletonVoxels = 0;
    for (std::size_t flat = 0; flat < total; ++flat) {
        if (skeleton[flat] != 0) {
            mask[flat] = 1;
            ++skeletonVoxels;
        }
    }
    graph.statistics.skeletonVoxels = skeletonVoxels;

    const MaskNeighborhood neighborhood(crop.shapeZYX, mask);
    std::vector<std::size_t> neighbors;
    std::vector<std::uint8_t> critical(total, 0);
    for (std::size_t flat = 0; flat < total; ++flat) {
        if (mask[flat] == 0)
            continue;
        neighborhood.collect(flat, neighbors);
        if (neighbors.size() != 2)
            critical[flat] = 1;
    }

    std::size_t skeletonComponents = 0;
    const auto componentLabels = labelFiberSkeletonComponents(
        crop.shapeZYX, mask, skeletonComponents);
    graph.statistics.skeletonComponents = skeletonComponents;

    std::vector<std::uint8_t> componentHasNode(skeletonComponents + 1, 0);
    for (std::size_t flat = 0; flat < total; ++flat) {
        if (critical[flat] != 0 && componentLabels[flat] > 0)
            componentHasNode[static_cast<std::size_t>(componentLabels[flat])] = 1;
    }
    std::size_t nodeFreeCycles = 0;
    for (std::size_t label = 1; label <= skeletonComponents; ++label) {
        if (componentHasNode[label] == 0)
            ++nodeFreeCycles;
    }

    // One node per non-degree-two voxel, numbered in raster order. Nothing is
    // contracted, so a junction cluster keeps its internal structure as a set
    // of nodes joined by empty-interior edges.
    std::vector<std::size_t> nodeOfVoxel(total, kNoIndex);
    for (std::size_t flat = 0; flat < total; ++flat) {
        if (critical[flat] == 0)
            continue;
        FiberSkeletonNode node;
        node.id = graph.nodes.size();
        node.positionBaseXYZ = crop.baseXYZ(flat);
        nodeOfVoxel[flat] = node.id;
        graph.nodes.push_back(std::move(node));
    }

    // Walk every link out of a node voxel once, keyed by the unordered voxel
    // pair it is entered through. A link is either a direct adjacency between
    // two node voxels (empty interior) or a maximal degree-two chain; a chain
    // that returns to its own node voxel becomes a self-loop.
    std::set<std::pair<std::size_t, std::size_t>> visited;
    std::vector<std::size_t> startNeighbors;
    std::vector<std::size_t> chainNeighbors;
    std::vector<std::size_t> interior;
    for (std::size_t start = 0; start < total; ++start) {
        if (critical[start] == 0)
            continue;
        const std::size_t startNode = nodeOfVoxel[start];
        neighborhood.collect(start, startNeighbors);
        for (const std::size_t first : startNeighbors) {
            if (!visited.insert(orderedPair(start, first)).second)
                continue;
            interior.clear();
            std::size_t previous = start;
            std::size_t current = first;
            while (nodeOfVoxel[current] == kNoIndex) {
                interior.push_back(current);
                neighborhood.collect(current, chainNeighbors);
                std::size_t subsequent = kNoIndex;
                std::size_t choices = 0;
                for (const std::size_t candidate : chainNeighbors) {
                    if (candidate == previous)
                        continue;
                    subsequent = candidate;
                    ++choices;
                }
                if (choices != 1) {
                    throw std::runtime_error(
                        "degree-two skeleton chain has inconsistent adjacency");
                }
                visited.insert(orderedPair(current, subsequent));
                previous = current;
                current = subsequent;
            }

            FiberSkeletonEdge edge;
            edge.id = graph.edges.size();
            edge.nodeIds = {startNode, nodeOfVoxel[current]};
            edge.pointsBaseXYZ.reserve(interior.size() + 2);
            edge.pointsBaseXYZ.push_back(
                graph.nodes[edge.nodeIds[0]].positionBaseXYZ);
            for (const std::size_t flat : interior)
                edge.pointsBaseXYZ.push_back(crop.baseXYZ(flat));
            edge.pointsBaseXYZ.push_back(
                graph.nodes[edge.nodeIds[1]].positionBaseXYZ);
            edge.lengthBase = fiberPolylineLength(edge.pointsBaseXYZ);
            graph.edges.push_back(std::move(edge));
        }
    }

    refreshGraphStatistics(graph, nodeFreeCycles);
    graph.statistics.rawNodes = graph.statistics.graphNodes;
    graph.statistics.rawEdges = graph.statistics.graphEdges;
    return graph;
}

void reduceFiberSkeletonGraph(
    FiberSkeletonGraph& graph, double junctionMergeMaxBase,
    double spurPruneMaxBase)
{
    if (!std::isfinite(junctionMergeMaxBase) || !std::isfinite(spurPruneMaxBase)) {
        throw std::invalid_argument(
            "fiber skeleton reduction limits must be finite");
    }

    refreshIncidence(graph);
    std::size_t liveNodes = 0;
    for (const auto& node : graph.nodes) {
        if (!node.removed)
            ++liveNodes;
    }
    std::size_t liveEdges = 0;
    for (const auto& edge : graph.edges) {
        if (!edge.removed)
            ++liveEdges;
    }
    // Cycles of skeleton components without any node are not representable in
    // the node/edge graph, so carry that part of the reported rank across the
    // reduction instead of dropping it.
    const std::size_t structural = structuralCycleRank(
        liveNodes, liveEdges, graphComponentCount(graph));
    const std::size_t carriedCycles = graph.statistics.cycleRank > structural
        ? graph.statistics.cycleRank - structural
        : 0;

    pruneShortSpurs(graph, spurPruneMaxBase);
    suppressDegreeTwoNodes(graph);

    refreshGraphStatistics(graph, carriedCycles);
    groupJunctionRegions(graph, junctionMergeMaxBase);
    graph.compact();
}

std::vector<cv::Vec3d> resampleFiberPolyline(
    const std::vector<cv::Vec3d>& pointsBaseXYZ, double stepBase)
{
    if (pointsBaseXYZ.size() < 2) {
        throw std::invalid_argument(
            "fiber polyline resampling requires at least two points");
    }
    if (!std::isfinite(stepBase) || !(stepBase > 0.0)) {
        throw std::invalid_argument(
            "fiber polyline resampling step must be finite and positive");
    }
    const auto geometry = makePolylineArcGeometry(pointsBaseXYZ);
    const double length = geometry.length();
    const double ratio = length / stepBase;
    if (!std::isfinite(ratio) || ratio > kMaxResampleSegments) {
        throw std::invalid_argument(
            "fiber polyline resampling step produces too many samples");
    }
    const auto segments = std::max<std::int64_t>(1, std::llround(ratio));
    std::vector<cv::Vec3d> resampled;
    resampled.reserve(static_cast<std::size_t>(segments) + 1);
    resampled.push_back(pointsBaseXYZ.front());
    for (std::int64_t index = 1; index < segments; ++index) {
        const double arc = length * (static_cast<double>(index) /
                                     static_cast<double>(segments));
        resampled.push_back(samplePolylineArc(geometry, arc).point);
    }
    resampled.push_back(pointsBaseXYZ.back());
    return resampled;
}

double fiberPolylineLength(const std::vector<cv::Vec3d>& pointsBaseXYZ)
{
    double length = 0.0;
    for (std::size_t index = 1; index < pointsBaseXYZ.size(); ++index) {
        const cv::Vec3d delta = pointsBaseXYZ[index] - pointsBaseXYZ[index - 1];
        length += std::sqrt(delta.dot(delta));
    }
    return length;
}

cv::Vec3d fiberPolylinePointAtArc(
    const std::vector<cv::Vec3d>& pointsBaseXYZ, double arcBase)
{
    if (pointsBaseXYZ.empty())
        throw std::invalid_argument("fiber polyline requires at least one point");
    if (!std::isfinite(arcBase))
        throw std::invalid_argument("fiber polyline arclength must be finite");
    if (pointsBaseXYZ.size() < 2 ||
        fiberPolylineLength(pointsBaseXYZ) <= kEpsilon) {
        return arcBase <= 0.0 ? pointsBaseXYZ.front() : pointsBaseXYZ.back();
    }
    const auto geometry = makePolylineArcGeometry(pointsBaseXYZ);
    if (arcBase <= 0.0)
        return pointsBaseXYZ.front();
    if (arcBase >= geometry.length())
        return pointsBaseXYZ.back();
    return samplePolylineArc(geometry, arcBase).point;
}

std::vector<cv::Vec3d> fiberPolylineSlice(
    const std::vector<cv::Vec3d>& pointsBaseXYZ, double beginArcBase,
    double endArcBase)
{
    if (pointsBaseXYZ.size() < 2) {
        throw std::invalid_argument(
            "fiber polyline slicing requires at least two points");
    }
    if (!std::isfinite(beginArcBase) || !std::isfinite(endArcBase) ||
        endArcBase < beginArcBase) {
        throw std::invalid_argument("fiber polyline slice interval is invalid");
    }
    const auto geometry = makePolylineArcGeometry(pointsBaseXYZ);
    const double length = geometry.length();
    const double begin = std::clamp(beginArcBase, 0.0, length);
    const double end = std::clamp(endArcBase, 0.0, length);
    auto slice = slicePolylineArc(geometry, begin, end);
    if (slice.empty())
        throw std::logic_error("fiber polyline slice produced no point");
    if (begin <= 0.0)
        slice.front() = pointsBaseXYZ.front();
    if (end >= length)
        slice.back() = pointsBaseXYZ.back();
    return slice;
}

void writeFiberSkeletonGraphObj(
    const std::filesystem::path& path, const FiberSkeletonGraph& graph)
{
    const std::filesystem::path parent = path.parent_path();
    if (!parent.empty())
        std::filesystem::create_directories(parent);

    std::vector<vc::core::io::NamedPolyline> edgeLines;
    edgeLines.reserve(graph.edges.size());
    for (const auto& edge : graph.edges) {
        if (edge.removed)
            continue;
        edgeLines.push_back(vc::core::io::NamedPolyline{
            "edge_" + std::to_string(edge.id) + "_n" +
                std::to_string(edge.nodeIds[0]) + "_n" +
                std::to_string(edge.nodeIds[1]),
            edge.pointsBaseXYZ});
    }
    vc::core::io::writePolylinesObj(
        edgeLines, path, "VC3D fiber skeleton edges");

    std::vector<vc::core::io::NamedPolyline> nodeMarkers;
    nodeMarkers.reserve(graph.nodes.size());
    for (const auto& node : graph.nodes) {
        if (node.removed)
            continue;
        const cv::Vec3d offset =
            node.positionBaseXYZ + cv::Vec3d{kNodeMarkerLengthBase, 0.0, 0.0};
        nodeMarkers.push_back(vc::core::io::NamedPolyline{
            "node_" + std::to_string(node.id) + "_" + nodeKindName(node.kind),
            std::vector<cv::Vec3d>{node.positionBaseXYZ, offset}});
    }
    const std::filesystem::path markerPath =
        parent / (path.stem().string() + "_markers.obj");
    vc::core::io::writePolylinesObj(
        nodeMarkers, markerPath, "VC3D fiber skeleton nodes");
}

}  // namespace vc::fiber_tracer
