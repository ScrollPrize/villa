// Eikonal phase retrieval, CUDA: the same passes as src/cpu/epr.cpp on device frames.
#include "common/epr_core.hpp"
#include "cuda_utils.cuh"
#include "vc_tomography/filters.hpp"

namespace vc_tomography::cuda {

namespace {
using epr_core::Coef;

// every kernel: a grid-stride loop over the pixels of nb frames (H, W)

__global__ void floor_kernel(float* __restrict__ a, size_t total, float eps) {
    VC_TOMOGRAPHY_GRID_STRIDE(i, total) a[i] = epr_core::floor_eps(a[i], eps);
}

__global__ void gradient_kernel(const float* __restrict__ I, float* __restrict__ gx, float* __restrict__ gy,
                                float* __restrict__ q, int nb, int H, int W, Coef c) {
    const size_t plane = static_cast<size_t>(H) * W, total = plane * nb;
    VC_TOMOGRAPHY_GRID_STRIDE(i, total) {
        const size_t base = (i / plane) * plane;
        const int x = static_cast<int>(i % W), y = static_cast<int>((i / W) % H);
        epr_core::gradient(I + base, H, W, y, x, c, gx[i], gy[i], q[i]);
    }
}

__global__ void divergence_kernel(const float* __restrict__ I, const float* __restrict__ gx,
                                  const float* __restrict__ gy, float* __restrict__ D, int nb, int H, int W) {
    const size_t plane = static_cast<size_t>(H) * W, total = plane * nb;
    VC_TOMOGRAPHY_GRID_STRIDE(i, total) {
        const size_t base = (i / plane) * plane;
        const int x = static_cast<int>(i % W), y = static_cast<int>((i / W) % H);
        D[i] = epr_core::flux_divergence(I + base, gx + base, gy + base, H, W, y, x);
    }
}

__global__ void residual_kernel(const float* __restrict__ det, const float* __restrict__ I,
                                const float* __restrict__ gx, const float* __restrict__ gy,
                                const float* __restrict__ q, const float* __restrict__ D, float* __restrict__ r,
                                int nb, int H, int W, Coef c) {
    const size_t plane = static_cast<size_t>(H) * W, total = plane * nb;
    VC_TOMOGRAPHY_GRID_STRIDE(i, total) {
        const size_t base = (i / plane) * plane;
        const int x = static_cast<int>(i % W), y = static_cast<int>((i / W) % H);
        r[i] = epr_core::floor_eps(det[i], c.eps) -
               epr_core::forward(I + base, gx + base, gy + base, q + base, D + base, H, W, y, x, c);
    }
}

__global__ void update_kernel(float* __restrict__ I, const float* __restrict__ r, size_t total, float relax, float eps) {
    VC_TOMOGRAPHY_GRID_STRIDE(i, total) I[i] = epr_core::update(I[i], r[i], relax, eps);
}

}  // namespace

void epr_frames(float* frames, int n_frames, int H, int W, const EprParams& p, FourierFilter& paganin, void* stream) {
    cudaStream_t s = as_stream(stream);
    const float A = static_cast<float>(p.shift_px2);
    const Coef c{A, 0.5f * A * A, static_cast<float>(p.shift_px2 * p.shift_px2 / (p.delta_beta * p.delta_beta)),
                 p.max_shift > 0.f ? p.max_shift / A : 0.f, p.eps};
    const size_t plane = static_cast<size_t>(H) * W;
    // frames per pass: six work planes each, about 768 MB in all
    const int batch = static_cast<int>(std::min<size_t>(std::max<size_t>((size_t(768) << 20) / (6 * plane * sizeof(float)), 1),
                                                        static_cast<size_t>(std::max(1, n_frames))));
    const size_t cap = static_cast<size_t>(batch) * plane;
    thread_local DeviceBuffer<float> I, gx, gy, q, D, r;
    I.ensure(cap);
    if (p.iters > 0) {
        gx.ensure(cap);
        gy.ensure(cap);
        q.ensure(cap);
        D.ensure(cap);
        r.ensure(cap);
    }
    for (int f0 = 0; f0 < n_frames; f0 += batch) {
        const int nb = std::min(batch, n_frames - f0);
        float* det = frames + static_cast<size_t>(f0) * plane;
        const size_t total = static_cast<size_t>(nb) * plane;
        const unsigned grid = grid_for(total);
        paganin.apply(det, I.data(), nb, stream);
        floor_kernel<<<grid, 256, 0, s>>>(I.data(), total, c.eps);
        for (int it = 0; it < p.iters; ++it) {
            gradient_kernel<<<grid, 256, 0, s>>>(I.data(), gx.data(), gy.data(), q.data(), nb, H, W, c);
            divergence_kernel<<<grid, 256, 0, s>>>(I.data(), gx.data(), gy.data(), D.data(), nb, H, W);
            residual_kernel<<<grid, 256, 0, s>>>(det, I.data(), gx.data(), gy.data(), q.data(), D.data(), r.data(), nb, H, W, c);
            paganin.apply(r.data(), r.data(), nb, stream);
            update_kernel<<<grid, 256, 0, s>>>(I.data(), r.data(), total, p.relax, c.eps);
        }
        VC_TOMOGRAPHY_CUDA_CHECK(cudaMemcpyAsync(det, I.data(), total * sizeof(float), cudaMemcpyDeviceToDevice, s));
        VC_TOMOGRAPHY_CUDA_CHECK(cudaGetLastError());
    }
}

}  // namespace vc_tomography::cuda
