#include "cuda_utils.cuh"
#include "common/output_core.hpp"
#include "helirec/kernels.hpp"

namespace helirec::cuda {

namespace {
__global__ void to_uint8_kernel(const float* __restrict__ src, std::size_t src_pitch, std::uint8_t* __restrict__ out,
                                int ny, int nx, float lo, float range, int disk_n, int x0, int y0,
                                const std::uint8_t* __restrict__ mask, int mny, int mnx, int mfy, int mfx, int moy,
                                int mox) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;
    const std::size_t i = static_cast<std::size_t>(y) * nx + x;
    bool zero = disk_n > 0 && output_core::outside_disk(x0 + x, y0 + y, disk_n);
    if (mask && !zero)
        zero = !mask[static_cast<std::size_t>(output_core::mask_index(y, moy, mfy, mny)) * mnx +
                     output_core::mask_index(x, mox, mfx, mnx)];
    out[i] = zero ? std::uint8_t(0) : output_core::window_u8(src[static_cast<std::size_t>(y) * src_pitch + x], lo, range);
}
}  // namespace

void to_uint8(const float* src, std::size_t src_pitch, std::uint8_t* out, int ny, int nx, const Uint8Window& w,
              void* stream) {
    if (!(w.hi > w.lo)) throw Error("to_uint8: the window needs hi > lo");
    if (w.mask && (w.mask_ny < 1 || w.mask_nx < 1 || w.mask_fy < 1 || w.mask_fx < 1)) throw Error("to_uint8: empty mask");
    if (ny <= 0 || nx <= 0) return;
    const dim3 block(32, 8), grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    to_uint8_kernel<<<grid, block, 0, as_stream(stream)>>>(src, src_pitch, out, ny, nx, static_cast<float>(w.lo),
                                                           static_cast<float>(w.hi - w.lo), w.disk_n, w.x0, w.y0,
                                                           w.mask, w.mask_ny, w.mask_nx, w.mask_fy, w.mask_fx,
                                                           w.mask_oy, w.mask_ox);
    HELIREC_CUDA_CHECK(cudaGetLastError());
}

}  // namespace helirec::cuda
