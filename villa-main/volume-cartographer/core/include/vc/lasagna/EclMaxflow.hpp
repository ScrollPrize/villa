#pragma once

#include "vc/lasagna/MaxflowGraph.hpp"

#include <chrono>
#include <cstdint>
#include <span>
#include <vector>

namespace vc::lasagna {

struct EclMaxflowCutEdge {
    int32_t edge = -1;
    int32_t sourceSideNode = -1;
    int32_t sinkSideNode = -1;
    int32_t capacity = 0;
};

struct EclMaxflowMinCut {
    bool valid = false;
    uint64_t sourceReachableNodes = 0;
    uint64_t sinkSideNodes = 0;
    uint64_t cutDirectedEdges = 0;
    int64_t cutCapacity = 0;
    std::vector<uint8_t> sourceReachable;
    std::vector<EclMaxflowCutEdge> cutEdges;
};

enum class EclTerminalSide : uint8_t {
    Source,
    Sink
};

struct EclTerminalExpansion {
    bool valid = false;
    EclTerminalSide side = EclTerminalSide::Source;
    uint64_t iterations = 0;
    uint64_t regionNodes = 0;
    uint64_t absorbedNodes = 0;
    int64_t finalBoundaryCapacity = 0;
    bool finalBoundaryIsMinCut = false;
    bool touchedOppositeTerminal = false;
    std::vector<uint8_t> region;
};

struct EclMaxflowOptions {
    int runs = 1;
    bool computeMinCut = true;
    bool storeSourceReachable = false;
    bool storeCutEdges = false;
};

struct EclMaxflowResult {
    int64_t maxFlow = 0;
    double medianRuntimeSeconds = 0.0;
    double throughputGigaEdgesPerSecond = 0.0;
    int runs = 1;
    EclMaxflowMinCut minCut;
};

[[nodiscard]] bool eclMaxflowAvailable() noexcept;

[[nodiscard]] EclMaxflowMinCut computeMinCutFromFinalFlow(
    const MaxflowGraph& graph,
    std::span<const int> flow,
    int32_t sourceNode,
    int32_t sinkNode,
    bool storeSourceReachable = false,
    bool storeCutEdges = false);

[[nodiscard]] EclTerminalExpansion expandTerminalRegionAcrossMinCutBoundaries(
    const MaxflowGraph& graph,
    int32_t seedNode,
    int32_t oppositeTerminalNode,
    int64_t maxFlow,
    EclTerminalSide side,
    uint64_t maxIterations,
    bool storeRegion = false);

[[nodiscard]] EclMaxflowResult runEclMaxflow(
    const MaxflowGraph& graph,
    int32_t sourceNode,
    int32_t sinkNode,
    const EclMaxflowOptions& options);

[[nodiscard]] EclMaxflowResult runEclMaxflow(
    const MaxflowGraph& graph,
    int32_t sourceNode,
    int32_t sinkNode,
    int runs = 1);

} // namespace vc::lasagna
