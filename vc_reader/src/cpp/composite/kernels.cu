// CUDA reductions of sampled layer stacks: one thread per pixel.
#include "reducer.hpp"
#include "../cuda_buffer.hpp"

#include <algorithm>

namespace vcr::composite {
namespace {
__global__ void reducePixels(const float* values, std::size_t layers, std::size_t n, Options options, float* out)
{
    for (std::size_t p = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; p < n;
         p += std::size_t(gridDim.x) * blockDim.x) {
        detail::Reducer r(options);
        for (std::size_t k = 0; k < layers && !r.done; ++k)
            r.add(values[k * n + p]);
        out[p] = r.result();
    }
}
}  // namespace

namespace cuda {
void reduce(const float* values, std::size_t layers, std::size_t n, const Options& options, float* out)
{
    detail::checkLayers(layers, options);
    if (n == 0)
        return;
    constexpr unsigned threads = 128;
    const unsigned blocks = unsigned(std::min<std::size_t>((n + threads - 1) / threads, 65535));
    reducePixels<<<blocks, threads, 0, device::stream()>>>(values, layers, n, options, out);
    device::check(cudaGetLastError());
    device::complete();
}
}  // namespace cuda
}  // namespace vcr::composite
