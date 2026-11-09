#include <cmath>
#include <vector>

#include "common/interp_core.hpp"
#include "cuda_utils.cuh"
#include "helirec/kernels.hpp"

namespace helirec::cuda {

namespace {
constexpr int kTile = 16;           // 16x16 pixels per block
constexpr int kAngleChunk = 512;    // angles staged in shared memory per pass

template <int K>
__global__ void backproject_kernel(const float* __restrict__ sino, const float2* __restrict__ cs_sn,
                                   float* __restrict__ slice, int n_angles, int n_det, int ny, int nx,
                                   float rot_center, float scale, float c, int x0, int y0, bool clip,
                                   float r2, float us) {
    __shared__ float2 s_cs[kAngleChunk];
    const int ix = blockIdx.x * kTile + threadIdx.x;
    const int iy = blockIdx.y * kTile + threadIdx.y;
    const float x = static_cast<float>(ix + x0) - c;
    const float y = static_cast<float>(iy + y0) - c;
    const bool inside = ix < nx && iy < ny && (!clip || x * x + y * y <= r2);
    const int tid = threadIdx.y * kTile + threadIdx.x;
    float acc = 0.f;
    for (int a0 = 0; a0 < n_angles; a0 += kAngleChunk) {
        const int na = min(kAngleChunk, n_angles - a0);
        __syncthreads();
        for (int k = tid; k < na; k += kTile * kTile) s_cs[k] = cs_sn[a0 + k];
        __syncthreads();
        if (inside) {
            const float* row = sino + static_cast<size_t>(a0) * n_det;
            for (int k = 0; k < na; ++k, row += n_det)
                acc += interp_core::sample_row<K>(row, n_det, (x * s_cs[k].x + y * s_cs[k].y + rot_center) * us);
        }
    }
    if (ix < nx && iy < ny) slice[static_cast<size_t>(iy) * nx + ix] = inside ? acc * scale : 0.f;
}
}  // namespace

void backproject(const float* sino, const float* angles_host, float* slice, int n_angles,
                 int n_det, int ny, int nx, const BackprojectParams& p, void* stream) {
    const float c = p.center >= 0.f ? p.center : 0.5f * static_cast<float>(std::max(nx, ny) - 1);
    const float r = p.clip_radius >= 0.f ? p.clip_radius : c;
    std::vector<float2> cs(n_angles);
    for (int a = 0; a < n_angles; ++a)
        cs[a] = make_float2(std::cos(angles_host[a]), std::sin(angles_host[a]));
    cudaStream_t s = as_stream(stream);
    DeviceBuffer<float2> d_cs;
    d_cs.upload(cs.data(), cs.size(), s);
    dim3 block(kTile, kTile), grid((nx + kTile - 1) / kTile, (ny + kTile - 1) / kTile);
#define HELIREC_BP(K) backproject_kernel<K><<<grid, block, 0, s>>>(sino, d_cs.data(), slice, n_angles, n_det, ny, nx, \
                                           p.rot_center, p.scale, c, p.x0, p.y0, p.clip_outer_circle, r * r, p.u_scale)
    switch (p.interp) {
        case interp_core::Linear: HELIREC_BP(interp_core::Linear); break;
        case interp_core::Cubic: HELIREC_BP(interp_core::Cubic); break;
        case interp_core::Lanczos3: HELIREC_BP(interp_core::Lanczos3); break;
        case interp_core::BSpline3: HELIREC_BP(interp_core::BSpline3); break;
        case interp_core::BSpline5: HELIREC_BP(interp_core::BSpline5); break;
        default: throw Error("backproject: unknown interpolation");
    }
#undef HELIREC_BP
    HELIREC_CUDA_CHECK(cudaGetLastError());
    HELIREC_CUDA_CHECK(cudaStreamSynchronize(s));  // d_cs is freed on return
}

}  // namespace helirec::cuda
