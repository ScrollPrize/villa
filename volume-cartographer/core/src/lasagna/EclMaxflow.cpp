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
    int)
{
    throw std::runtime_error(
        "ECL-MaxFlow support was not built. Reconfigure volume-cartographer with "
        "-DVC_ENABLE_ECL_MAXFLOW=ON and a CUDA compiler.");
}

} // namespace vc::lasagna
