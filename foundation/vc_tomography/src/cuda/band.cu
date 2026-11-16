#include <cub/device/device_select.cuh>
#include <thrust/iterator/counting_iterator.h>

#include "common/interp_core.hpp"
#include "cuda_utils.cuh"
#include "vc_tomography/band.hpp"

namespace vc_tomography::cuda {

namespace {
constexpr int kAngleChunk = 1024;  // angles staged in shared memory per pass

__device__ inline int lower(float v, int n) { return min(static_cast<int>(floorf(v)), n - 2); }

__global__ void mark_kernel(const float* y, const float* x, size_t a, size_t b, BandBox box, uint8_t* mask) {
    const size_t k = a + blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (k >= b) return;
    const int iy = lower(y[k], box.n_grid) - box.y0, ix = lower(x[k], box.n_grid) - box.x0;
    uint8_t* m = mask + static_cast<size_t>(iy) * box.w + ix;
    m[0] = 1; m[1] = 1; m[box.w] = 1; m[box.w + 1] = 1;
}

// fan: a diverging beam (BackprojectParams::fan, its n_fan columns on the device); null: a parallel beam
template <int K>
__global__ void backproject_list_kernel(const float* __restrict__ sino, const float2* __restrict__ cs_sn, int n_angles,
                                        int n_det, const int32_t* __restrict__ list, int count, BandBox box,
                                        float rot_center, float scale, float c, bool clip, float r2, float us,
                                        float* __restrict__ img, const float* __restrict__ fan, int n_fan) {
    __shared__ float2 s_cs[kAngleChunk];
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    int li = 0;
    float x = 0.f, y = 0.f;
    bool inside = false;
    if (k < count) {
        li = list[k];
        x = static_cast<float>(li % box.w + box.x0) - c;
        y = static_cast<float>(li / box.w + box.y0) - c;
        inside = !clip || x * x + y * y <= r2;
    }
    float acc = 0.f;
    for (int a0 = 0; a0 < n_angles; a0 += kAngleChunk) {
        const int na = min(kAngleChunk, n_angles - a0);
        __syncthreads();
        for (int t = threadIdx.x; t < na; t += blockDim.x) s_cs[t] = cs_sn[a0 + t];
        __syncthreads();
        if (inside) {
            const float* row = sino + static_cast<size_t>(a0) * n_det;
            if (fan) {
                for (int t = 0; t < na; ++t, row += n_det) {
                    const float at = x * s_cs[t].x + y * s_cs[t].y + rot_center;
                    acc += interp_core::sample_row<K>(row, n_det, (at + fan[interp_core::fan_column(at, n_fan)] * (x * s_cs[t].y - y * s_cs[t].x)) * us);
                }
            } else {
                for (int t = 0; t < na; ++t, row += n_det)
                    acc += interp_core::sample_row<K>(row, n_det, (x * s_cs[t].x + y * s_cs[t].y + rot_center) * us);
            }
        }
    }
    if (k < count) img[li] = inside ? acc * scale : 0.f;
}

__global__ void sample_kernel(const float* img_lo, const float* img_hi, const float* fz, const float* y,
                              const float* x, size_t a, size_t b, BandBox box, float* out) {
    const size_t k = a + blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (k >= b) return;
    const int iy = lower(y[k], box.n_grid), ix = lower(x[k], box.n_grid);
    const float wy = y[k] - static_cast<float>(iy), wx = x[k] - static_cast<float>(ix);
    const size_t o = static_cast<size_t>(iy - box.y0) * box.w + (ix - box.x0);
    const float l0 = img_lo[o] + wx * (img_lo[o + 1] - img_lo[o]);
    const float l1 = img_lo[o + box.w] + wx * (img_lo[o + box.w + 1] - img_lo[o + box.w]);
    const float h0 = img_hi[o] + wx * (img_hi[o + 1] - img_hi[o]);
    const float h1 = img_hi[o + box.w] + wx * (img_hi[o + box.w + 1] - img_hi[o + box.w]);
    const float lo = l0 + wy * (l1 - l0), hi = h0 + wy * (h1 - h0);
    out[k] = lo + fz[k] * (hi - lo);
}

struct Compactor {  // CUB scratch, reused
    DeviceBuffer<uint8_t> temp;
    DeviceBuffer<int> count;
};
Compactor& compactor() {
    thread_local Compactor c;
    return c;
}
}  // namespace

void band_mark(const float* y, const float* x, size_t a, size_t b, const BandBox& box, uint8_t* mask, void* stream) {
    if (b <= a) return;
    mark_kernel<<<grid_1d(b - a, 256), 256, 0, as_stream(stream)>>>(y, x, a, b, box, mask);
    VC_TOMOGRAPHY_CUDA_CHECK(cudaGetLastError());
}

size_t band_compact(const uint8_t* mask, size_t n, int32_t* list, void* stream) {
    cudaStream_t s = as_stream(stream);
    auto& C = compactor();
    C.count.ensure(1);
    thrust::counting_iterator<int32_t> it(0);
    size_t bytes = 0;
    VC_TOMOGRAPHY_CUDA_CHECK(cub::DeviceSelect::Flagged(nullptr, bytes, it, mask, list, C.count.data(), static_cast<int>(n), s));
    C.temp.ensure(bytes);
    VC_TOMOGRAPHY_CUDA_CHECK(cub::DeviceSelect::Flagged(C.temp.data(), bytes, it, mask, list, C.count.data(), static_cast<int>(n), s));
    int count = 0;
    VC_TOMOGRAPHY_CUDA_CHECK(cudaMemcpyAsync(&count, C.count.data(), sizeof(int), cudaMemcpyDeviceToHost, s));
    VC_TOMOGRAPHY_CUDA_CHECK(cudaStreamSynchronize(s));
    return static_cast<size_t>(count);
}

void backproject_list(const float* sino, const void* cs_sn, int n_angles, int n_det, const int32_t* list, size_t count,
                      const BandBox& box, const BackprojectParams& p, float* img, void* stream) {
    if (!count) return;
    const float c = p.center >= 0.f ? p.center : 0.5f * static_cast<float>(box.n_grid - 1);
    const float r = p.clip_radius >= 0.f ? p.clip_radius : c;
    // (kept between calls; the host table is the caller's and outlives the call)
    thread_local DeviceBuffer<float> d_fan;
    if (p.fan && p.n_fan < 1) throw Error("backproject_list: a fan without columns");
    if (p.fan) {
        d_fan.ensure(static_cast<size_t>(p.n_fan));
        VC_TOMOGRAPHY_CUDA_CHECK(cudaMemcpyAsync(d_fan.data(), p.fan, p.n_fan * sizeof(float), cudaMemcpyHostToDevice, as_stream(stream)));
    }
#define VC_TOMOGRAPHY_BPL(K) backproject_list_kernel<K><<<grid_1d(count, 128), 128, 0, as_stream(stream)>>>( \
        sino, static_cast<const float2*>(cs_sn), n_angles, n_det, list, static_cast<int>(count), box, p.rot_center, \
        p.scale, c, p.clip_outer_circle, r * r, p.u_scale, img, p.fan ? d_fan.data() : nullptr, p.n_fan)
    interp_core::dispatch_interp(p.interp, "backproject_list", [&](auto k) { VC_TOMOGRAPHY_BPL(decltype(k)::value); });
#undef VC_TOMOGRAPHY_BPL
    VC_TOMOGRAPHY_CUDA_CHECK(cudaGetLastError());
}

void band_sample(const float* img_lo, const float* img_hi, const float* fz, const float* y, const float* x, size_t a,
                 size_t b, const BandBox& box, float* out, void* stream) {
    if (b <= a) return;
    sample_kernel<<<grid_1d(b - a, 256), 256, 0, as_stream(stream)>>>(img_lo, img_hi, fz, y, x, a, b, box, out);
    VC_TOMOGRAPHY_CUDA_CHECK(cudaGetLastError());
}

}  // namespace vc_tomography::cuda
