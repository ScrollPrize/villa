#include "vc/lasagna/EclMaxflow.hpp"

#include <stdexcept>

namespace vc::lasagna {

bool eclMaxflowAvailable() noexcept
{
    return false;
}

EclMaxflowResult runEclMaxflow(
    const MaxflowGraph&,
    int32_t,
    int32_t,
    const EclMaxflowOptions&)
{
    throw std::runtime_error(
        "ECL-MaxFlow support was not built. Reconfigure volume-cartographer with "
        "-DVC_ENABLE_ECL_MAXFLOW=ON and a CUDA compiler.");
}

EclMaxflowResult runEclMaxflow(
    const MaxflowGraph& graph,
    int32_t sourceNode,
    int32_t sinkNode,
    int runs)
{
    EclMaxflowOptions options;
    options.runs = runs;
    return runEclMaxflow(graph, sourceNode, sinkNode, options);
}

} // namespace vc::lasagna
