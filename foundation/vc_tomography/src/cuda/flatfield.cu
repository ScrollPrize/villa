#include "../common/flatfield_core.hpp"
#include "cuda_utils.cuh"
#include "vc_tomography/kernels.hpp"

namespace vc_tomography::cuda {

namespace {
__global__ void flatfield_kernel(const float* __restrict__ raw, const float* __restrict__ dark,
                                 const float* __restrict__ flat, float* __restrict__ out,
                                 std::size_t total, std::size_t n_pix, float threshold, bool take_log) {
    VC_TOMOGRAPHY_GRID_STRIDE(k, total) {
        const std::size_t i = k % n_pix;
        out[k] = flatfield_core::value(raw[k], dark[i], flat[i], threshold, take_log);
    }
}
}  // namespace

void flatfield(const float* raw, const float* dark, const float* flat, float* out,
               std::size_t n_proj, std::size_t n_pix, const FlatFieldParams& p, void* stream) {
    const std::size_t total = n_proj * n_pix;
    if (!total) return;
    flatfield_kernel<<<grid_for(total), 256, 0, as_stream(stream)>>>(raw, dark, flat, out, total, n_pix,
                                                            p.threshold, p.take_log);
    VC_TOMOGRAPHY_CUDA_CHECK(cudaGetLastError());
}

}  // namespace vc_tomography::cuda
