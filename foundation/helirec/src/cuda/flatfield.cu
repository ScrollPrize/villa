#include "cuda_utils.cuh"
#include "helirec/kernels.hpp"

namespace helirec::cuda {

namespace {
__global__ void flatfield_kernel(const float* __restrict__ raw, const float* __restrict__ dark,
                                 const float* __restrict__ flat, float* __restrict__ out,
                                 std::size_t total, std::size_t n_pix, float threshold, bool take_log) {
    for (std::size_t k = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; k < total;
         k += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const std::size_t i = k % n_pix;
        float v = (raw[k] - dark[i]) / (flat[i] - dark[i]);
        v = fmaxf(v, threshold);  // fmaxf(NaN, t) == t, matching the CPU path
        out[k] = take_log ? -logf(v) : v;
    }
}
}  // namespace

void flatfield(const float* raw, const float* dark, const float* flat, float* out,
               std::size_t n_proj, std::size_t n_pix, const FlatFieldParams& p, void* stream) {
    const std::size_t total = n_proj * n_pix;
    if (!total) return;
    const unsigned block = 256;
    const unsigned grid = std::min<unsigned>(grid_1d(total, block), 65535u * 8u);
    flatfield_kernel<<<grid, block, 0, as_stream(stream)>>>(raw, dark, flat, out, total, n_pix,
                                                            p.threshold, p.take_log);
    HELIREC_CUDA_CHECK(cudaGetLastError());
}

}  // namespace helirec::cuda
