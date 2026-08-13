#include "vc/lasagna/EclMaxflow.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace vc::lasagna {
namespace {

[[nodiscard]] int checkedInt(uint64_t value, const char* name)
{
    if (value > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string(name) + " exceeds int32 range");
    }
    return static_cast<int>(value);
}

void validateGraphAndTerminals(
    const MaxflowGraph& graph,
    std::span<const int> flow,
    int32_t sourceNode,
    int32_t sinkNode)
{
    const int nodes = checkedInt(graph.stats.graphNodes, "node count");
    const int edges = checkedInt(graph.stats.directedEdges, "edge count");
    if (nodes <= 0) {
        throw std::runtime_error("min-cut graph has no nodes");
    }
    if (sourceNode < 0 || sourceNode >= nodes || sinkNode < 0 || sinkNode >= nodes) {
        throw std::runtime_error("min-cut source/sink node is outside graph");
    }
    if (sourceNode == sinkNode) {
        throw std::runtime_error("min-cut source and sink nodes must differ");
    }
    if (graph.nindex.size() != static_cast<size_t>(nodes) + 1 ||
        graph.nlist.size() != static_cast<size_t>(edges) ||
        graph.capacity.size() != static_cast<size_t>(edges) ||
        flow.size() != static_cast<size_t>(edges)) {
        throw std::runtime_error("min-cut graph/flow arrays are inconsistent");
    }
}

} // namespace

EclTerminalExpansion expandTerminalRegionAcrossMinCutBoundaries(
    const MaxflowGraph& graph,
    int32_t seedNode,
    int32_t oppositeTerminalNode,
    int64_t maxFlow,
    EclTerminalSide side,
    uint64_t maxIterations,
    bool storeRegion)
{
    const int nodes = checkedInt(graph.stats.graphNodes, "node count");
    const int edges = checkedInt(graph.stats.directedEdges, "edge count");
    if (nodes <= 0) {
        throw std::runtime_error("terminal expansion graph has no nodes");
    }
    if (seedNode < 0 || seedNode >= nodes || oppositeTerminalNode < 0 || oppositeTerminalNode >= nodes) {
        throw std::runtime_error("terminal expansion source/sink node is outside graph");
    }
    if (seedNode == oppositeTerminalNode) {
        throw std::runtime_error("terminal expansion source and sink nodes must differ");
    }
    if (graph.nindex.size() != static_cast<size_t>(nodes) + 1 ||
        graph.nlist.size() != static_cast<size_t>(edges) ||
        graph.capacity.size() != static_cast<size_t>(edges)) {
        throw std::runtime_error("terminal expansion graph arrays are inconsistent");
    }

    EclTerminalExpansion expansion;
    expansion.valid = true;
    expansion.side = side;
    if (maxFlow <= 0) {
        expansion.regionNodes = 1;
        expansion.region = storeRegion ? std::vector<uint8_t>(static_cast<size_t>(nodes), 0)
                                       : std::vector<uint8_t>{};
        if (storeRegion) {
            expansion.region[static_cast<size_t>(seedNode)] = 1;
        }
        return expansion;
    }

    std::vector<uint8_t> region(static_cast<size_t>(nodes), 0);
    std::vector<uint8_t> inFrontier(static_cast<size_t>(nodes), 0);
    std::vector<int32_t> frontier;
    region[static_cast<size_t>(seedNode)] = 1;
    expansion.regionNodes = 1;

    const auto addFrontierNode = [&](int32_t node) {
        if (node == oppositeTerminalNode) {
            expansion.touchedOppositeTerminal = true;
            return;
        }
        if (region[static_cast<size_t>(node)] == 0 && inFrontier[static_cast<size_t>(node)] == 0) {
            inFrontier[static_cast<size_t>(node)] = 1;
            frontier.push_back(node);
        }
    };

    for (uint64_t iter = 0; iter <= maxIterations; ++iter) {
        frontier.clear();
        std::fill(inFrontier.begin(), inFrontier.end(), 0);
        int64_t boundaryCapacity = 0;

        for (int32_t src = 0; src < nodes; ++src) {
            for (uint64_t edge = graph.nindex[static_cast<size_t>(src)];
                 edge < graph.nindex[static_cast<size_t>(src) + 1];
                 ++edge) {
                const int32_t dst = graph.nlist[static_cast<size_t>(edge)];
                if (side == EclTerminalSide::Source) {
                    if (region[static_cast<size_t>(src)] != 0 &&
                        region[static_cast<size_t>(dst)] == 0) {
                        boundaryCapacity += graph.capacity[static_cast<size_t>(edge)];
                        addFrontierNode(dst);
                    }
                } else {
                    if (region[static_cast<size_t>(src)] == 0 &&
                        region[static_cast<size_t>(dst)] != 0) {
                        boundaryCapacity += graph.capacity[static_cast<size_t>(edge)];
                        addFrontierNode(src);
                    }
                }
            }
        }

        expansion.finalBoundaryCapacity = boundaryCapacity;
        expansion.finalBoundaryIsMinCut = boundaryCapacity == maxFlow;
        if (boundaryCapacity != maxFlow || frontier.empty() || expansion.touchedOppositeTerminal ||
            iter == maxIterations) {
            expansion.iterations = iter;
            break;
        }

        for (int32_t node : frontier) {
            region[static_cast<size_t>(node)] = 1;
        }
        expansion.regionNodes += frontier.size();
        expansion.absorbedNodes += frontier.size();
    }

    if (storeRegion) {
        expansion.region = std::move(region);
    }
    return expansion;
}

EclMaxflowMinCut computeMinCutFromFinalFlow(
    const MaxflowGraph& graph,
    std::span<const int> flow,
    int32_t sourceNode,
    int32_t sinkNode,
    bool storeSourceReachable,
    bool storeCutEdges)
{
    validateGraphAndTerminals(graph, flow, sourceNode, sinkNode);

    const int nodes = static_cast<int>(graph.stats.graphNodes);
    const int edges = static_cast<int>(graph.stats.directedEdges);

    std::vector<uint64_t> rnindex(static_cast<size_t>(nodes) + 1, 0);
    for (int32_t src = 0; src < nodes; ++src) {
        for (uint64_t edge = graph.nindex[static_cast<size_t>(src)];
             edge < graph.nindex[static_cast<size_t>(src) + 1];
             ++edge) {
            const int32_t dst = graph.nlist[static_cast<size_t>(edge)];
            ++rnindex[static_cast<size_t>(dst) + 1];
        }
    }
    for (size_t node = 1; node < rnindex.size(); ++node) {
        rnindex[node] += rnindex[node - 1];
    }
    std::vector<int32_t> rnlist(static_cast<size_t>(edges), 0);
    std::vector<int32_t> retoe(static_cast<size_t>(edges), 0);
    std::vector<uint64_t> cursor = rnindex;
    for (int32_t src = 0; src < nodes; ++src) {
        for (uint64_t edge = graph.nindex[static_cast<size_t>(src)];
             edge < graph.nindex[static_cast<size_t>(src) + 1];
             ++edge) {
            const int32_t dst = graph.nlist[static_cast<size_t>(edge)];
            const uint64_t reverseEdge = cursor[static_cast<size_t>(dst)]++;
            rnlist[static_cast<size_t>(reverseEdge)] = src;
            retoe[static_cast<size_t>(reverseEdge)] = static_cast<int32_t>(edge);
        }
    }

    std::vector<uint8_t> reachable(static_cast<size_t>(nodes), 0);
    std::deque<int32_t> queue;
    reachable[static_cast<size_t>(sourceNode)] = 1;
    queue.push_back(sourceNode);

    while (!queue.empty()) {
        const int32_t node = queue.front();
        queue.pop_front();

        for (uint64_t edge = graph.nindex[static_cast<size_t>(node)];
             edge < graph.nindex[static_cast<size_t>(node) + 1];
             ++edge) {
            const int32_t dst = graph.nlist[static_cast<size_t>(edge)];
            if (flow[static_cast<size_t>(edge)] < graph.capacity[static_cast<size_t>(edge)] &&
                reachable[static_cast<size_t>(dst)] == 0) {
                reachable[static_cast<size_t>(dst)] = 1;
                queue.push_back(dst);
            }
        }

        for (uint64_t reverseEdge = rnindex[static_cast<size_t>(node)];
             reverseEdge < rnindex[static_cast<size_t>(node) + 1];
             ++reverseEdge) {
            const int32_t src = rnlist[static_cast<size_t>(reverseEdge)];
            const int32_t edge = retoe[static_cast<size_t>(reverseEdge)];
            if (flow[static_cast<size_t>(edge)] > 0 && reachable[static_cast<size_t>(src)] == 0) {
                reachable[static_cast<size_t>(src)] = 1;
                queue.push_back(src);
            }
        }
    }

    EclMaxflowMinCut cut;
    cut.valid = true;
    for (uint8_t value : reachable) {
        cut.sourceReachableNodes += value != 0 ? 1U : 0U;
    }
    cut.sinkSideNodes = static_cast<uint64_t>(nodes) - cut.sourceReachableNodes;

    for (int32_t src = 0; src < nodes; ++src) {
        if (reachable[static_cast<size_t>(src)] == 0) {
            continue;
        }
        for (uint64_t edge = graph.nindex[static_cast<size_t>(src)];
             edge < graph.nindex[static_cast<size_t>(src) + 1];
             ++edge) {
            const int32_t dst = graph.nlist[static_cast<size_t>(edge)];
            if (reachable[static_cast<size_t>(dst)] == 0) {
                ++cut.cutDirectedEdges;
                cut.cutCapacity += graph.capacity[static_cast<size_t>(edge)];
                if (storeCutEdges) {
                    cut.cutEdges.push_back({
                        static_cast<int32_t>(edge),
                        src,
                        dst,
                        graph.capacity[static_cast<size_t>(edge)],
                    });
                }
            }
        }
    }

    if (storeSourceReachable) {
        cut.sourceReachable = std::move(reachable);
    }
    return cut;
}

} // namespace vc::lasagna
