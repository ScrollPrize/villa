#include <cmath>
#include <vector>

#include "common/interp_core.hpp"
#include "cuda_utils.cuh"
#include "helirec/kernels.hpp"

namespace helirec::cuda {

namespace {
constexpr int kTile = 16;           // 16x16 threads per block; a block takes a tile of 32x32 pixels, four per thread
static_assert(2 * kTile == kBackprojectTile, "a block computes one tile");
constexpr int kPix = 4;             // (kTile apart on each axis)
constexpr int kAngleChunk = 512;    // angles staged in shared memory per pass

template <int K>
__global__ void backproject_kernel(const float* __restrict__ sino, const float2* __restrict__ cs_sn,
                                   float* __restrict__ slice, int n_angles, int n_det, int ny, int nx,
                                   float rot_center, float scale, float c, int x0, int y0, bool clip,
                                   float r2, float us, const int32_t* __restrict__ tiles, int ntx) {
    __shared__ float2 s_cs[kAngleChunk];
    // the block's tile: (blockIdx.x, blockIdx.y), or entry blockIdx.x of the tile list
    const int bx = tiles ? tiles[blockIdx.x] % ntx : blockIdx.x, by = tiles ? tiles[blockIdx.x] / ntx : blockIdx.y;
    const int q0 = bx * kTile * 2 + threadIdx.x, r0 = by * kTile * 2 + threadIdx.y;
    float x[kPix], y[kPix], acc[kPix], umax[kPix];
    bool any = false;
#pragma unroll
    for (int j = 0; j < kPix; ++j) {
        const int ix = q0 + (j & 1) * kTile, iy = r0 + (j >> 1) * kTile;
        x[j] = static_cast<float>(ix + x0) - c;
        y[j] = static_cast<float>(iy + y0) - c;
        const bool inside = ix < nx && iy < ny && (!clip || x[j] * x[j] + y[j] * y[j] <= r2);
        any |= inside;
        acc[j] = 0.f;
        // detector positions (bins) within the kernel's reach: [reach_lo, umax)
        umax[j] = inside ? static_cast<float>(n_det) + interp_core::reach_hi<K>() : -1e30f;
    }
    const int tid = threadIdx.y * kTile + threadIdx.x;
    for (int a0 = 0; a0 < n_angles; a0 += kAngleChunk) {
        const int na = min(kAngleChunk, n_angles - a0);
        __syncthreads();
        for (int k = tid; k < na; k += kTile * kTile) s_cs[k] = cs_sn[a0 + k];
        __syncthreads();
        if (any) {
            const float* row = sino + static_cast<size_t>(a0) * n_det;
            for (int k = 0; k < na; ++k, row += n_det) {
                const float2 t = s_cs[k];
#pragma unroll
                for (int j = 0; j < kPix; ++j) {
                    // (the contraction nvcc applies to (x cos + y sin + rot) us, spelled out)
                    const float u = __fmul_rn(__fadd_rn(__fmaf_rn(x[j], t.x, __fmul_rn(y[j], t.y)), rot_center), us);
                    if (u >= interp_core::reach_lo<K>() && u < umax[j]) acc[j] += interp_core::sample_row_reach<K>(row, n_det, u);
                }
            }
        }
    }
#pragma unroll
    for (int j = 0; j < kPix; ++j) {
        const int ix = q0 + (j & 1) * kTile, iy = r0 + (j >> 1) * kTile;
        if (ix < nx && iy < ny) slice[static_cast<size_t>(iy) * nx + ix] = acc[j] * scale;
    }
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
    // (kept between calls: one slice after another on the calling thread, each finished before it returns)
    thread_local DeviceBuffer<float2> d_cs;
    thread_local DeviceBuffer<int32_t> d_tiles;
    d_cs.ensure(cs.size());
    HELIREC_CUDA_CHECK(cudaMemcpyAsync(d_cs.data(), cs.data(), cs.size() * sizeof(float2), cudaMemcpyHostToDevice, s));
    const int ntx = (nx + 2 * kTile - 1) / (2 * kTile), nty = (ny + 2 * kTile - 1) / (2 * kTile);
    dim3 block(kTile, kTile), grid(ntx, nty);
    const int32_t* tiles = nullptr;
    if (p.tiles) {   // the listed tiles alone; the others are 0
        HELIREC_CUDA_CHECK(cudaMemsetAsync(slice, 0, static_cast<size_t>(ny) * nx * sizeof(float), s));
        if (p.n_tiles <= 0) {
            HELIREC_CUDA_CHECK(cudaStreamSynchronize(s));
            return;
        }
        d_tiles.ensure(static_cast<size_t>(p.n_tiles));
        HELIREC_CUDA_CHECK(cudaMemcpyAsync(d_tiles.data(), p.tiles, p.n_tiles * sizeof(int32_t), cudaMemcpyHostToDevice, s));
        tiles = d_tiles.data();
        grid = dim3(p.n_tiles, 1);
    }
#define HELIREC_BP(K) backproject_kernel<K><<<grid, block, 0, s>>>(sino, d_cs.data(), slice, n_angles, n_det, ny, nx, \
                                           p.rot_center, p.scale, c, p.x0, p.y0, p.clip_outer_circle, r * r, p.u_scale, \
                                           tiles, ntx)
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
    HELIREC_CUDA_CHECK(cudaStreamSynchronize(s));  // the slice is complete on return
}

}  // namespace helirec::cuda
