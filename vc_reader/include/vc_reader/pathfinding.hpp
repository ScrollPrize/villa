#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace vcr::pathfinding {
// Row-major float32 entered-pixel costs, eight neighbors without diagonal scaling.
// Nonnegative finite costs and +infinity (blocked) are accepted; NaN/negative costs,
// empty/oversized grids and out-of-range endpoints throw. The source costs zero.
// Blocked endpoints or a disconnected target return an empty path. Equal-cost paths
// prefer fewer steps, then the smaller predecessor index. Both endpoints are included.
std::vector<std::uint32_t> shortestPath(const float* costs, std::size_t rows, std::size_t cols,
                                      std::uint32_t source, std::uint32_t target);
#ifdef VC_READER_CUDA
namespace cuda {
// Host input/output; computes distances on the calling thread's CUDA device.
std::vector<std::uint32_t> shortestPath(const float* costs, std::size_t rows, std::size_t cols,
                                      std::uint32_t source, std::uint32_t target);
}
#endif
}
