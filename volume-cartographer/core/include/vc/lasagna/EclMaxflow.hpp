#pragma once

#include "vc/lasagna/MaxflowGraph.hpp"

#include <chrono>
#include <cstdint>
#include <vector>

namespace vc::lasagna {

struct EclMaxflowResult {
    int64_t maxFlow = 0;
    double medianRuntimeSeconds = 0.0;
    double throughputGigaEdgesPerSecond = 0.0;
    int runs = 1;
};

[[nodiscard]] bool eclMaxflowAvailable() noexcept;

[[nodiscard]] EclMaxflowResult runEclMaxflow(
    const MaxflowGraph& graph,
    int32_t sourceNode,
    int32_t sinkNode,
    int runs = 1);

} // namespace vc::lasagna
