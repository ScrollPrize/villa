// CUDA GHBP: kernels follow nabuxx ghbp_cuda.cu (backprojector_kernel,
// aggregator_kernel, clip_outer_circle_kernel) with the same arithmetic.
#include <cmath>

#include "common/ghbp_core.hpp"
#include "cuda_utils.cuh"
#include "helirec/ghbp.hpp"

namespace helirec::cuda {

namespace {

using namespace helirec::ghbp_core;

// threads: x over grid HEIGHT, y over WIDTH, z over grids (as nabuxx)
__global__ void level0_kernel(const float* __restrict__ setups, const float* __restrict__ T6, int rf, int gw,
                              int gh, int ng, float* __restrict__ grids, int n_bins, int n_angles, float scale,
                              const float* __restrict__ sino, int proj_offset) {
    const int a = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y * blockDim.y + threadIdx.y;
    const int gi = blockIdx.z * blockDim.z + threadIdx.z;
    if (a >= gh || b >= gw || gi >= ng) return;
    float Y, X;
    grid_point(T6 + gi * 6, a, b, Y, X);
    const float v = level0_value(setups, sino, n_bins, n_angles, gi * rf + proj_offset, rf, Y, X);
    grids[(static_cast<size_t>(gi) * gh + a) * gw + b] = hd::mul(scale, v);
}

__global__ void aggregate_kernel(int do_sum, const float* __restrict__ T6, const float* __restrict__ Tinv6, int red,
                                 int w, int h, int ng, float* __restrict__ out, int pw, int ph, int png,
                                 const float* __restrict__ prev) {
    const int a = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y * blockDim.y + threadIdx.y;
    const int gi = blockIdx.z * blockDim.z + threadIdx.z;
    if (a >= h || b >= w || gi >= ng) return;
    const float* T = T6 + gi * 6;
    if (hd::is_nan(T[0])) return;
    float Y, X;
    grid_point(T, a, b, Y, X);
    const float v = aggregate_value(Tinv6, prev, pw, ph, png, gi, red, Y, X);
    const size_t pos = (static_cast<size_t>(gi) * h + a) * w + b;
    out[pos] = do_sum ? hd::add(out[pos], v) : v;
}

__global__ void clip_kernel(float* img, int n) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || j >= n) return;
    if (!in_circle(i, j, n)) img[static_cast<size_t>(i) * n + j] = 0.f;
}

unsigned div_up(int a, int b) { return static_cast<unsigned>((a + b - 1) / b); }

}  // namespace

struct Ghbp::Impl {
    GhbpPlan plan;
    DeviceBuffer<float> setups;
    std::vector<DeviceBuffer<float>> T, Tinv;
    DeviceBuffer<float> ping, pong;
};

Ghbp::Ghbp(const GhbpPlan& plan) : impl_(std::make_unique<Impl>()) {
    auto& I = *impl_;
    I.plan = plan;
    I.setups.upload(plan.setups.data(), plan.setups.size());
    std::size_t max_elems = 0;
    for (std::size_t l = 0; l < plan.levels.size(); ++l) {
        const auto& L = plan.levels[l];
        I.T.emplace_back();
        I.Tinv.emplace_back();
        I.T.back().upload(L.T.data(), L.T.size());
        I.Tinv.back().upload(L.Tinv.data(), L.Tinv.size());
        if (l + 1 < plan.levels.size())
            max_elems = std::max(max_elems, static_cast<std::size_t>(L.g) * L.h * L.w);
    }
    I.ping.resize(max_elems);
    I.pong.resize(max_elems);
    HELIREC_CUDA_CHECK(cudaDeviceSynchronize());
}

Ghbp::~Ghbp() = default;

const GhbpPlan& Ghbp::plan() const { return impl_->plan; }

void Ghbp::run(const float* sino, float* out, void* stream) {
    auto& I = *impl_;
    const auto& P = I.plan.p;
    const auto& Ls = I.plan.levels;
    const int nL = static_cast<int>(Ls.size());
    cudaStream_t s = as_stream(stream);
    for (int leg = 0; leg < P.legs; ++leg) {
        {
            const auto& L = Ls[0];
            dim3 block(64, 4, 4), grid(div_up(L.h, 64), div_up(L.w, 4), div_up(L.g, 4));
            level0_kernel<<<grid, block, 0, s>>>(I.setups.data(), I.T[0].data() + 6 * leg * L.g, L.red, L.w, L.h,
                                                 L.g, I.ping.data(), I.plan.n_bins, I.plan.n_angles,
                                                 P.scale_factor, sino, leg * L.g * L.red);
            HELIREC_CUDA_CHECK(cudaGetLastError());
        }
        float* prev = I.ping.data();
        float* next = I.pong.data();
        for (int l = 1; l < nL; ++l) {
            const auto& L = Ls[l];
            const auto& Lp = Ls[l - 1];
            const bool last = l == nL - 1;
            float* dst = last ? out : next;
            const bool big = L.g >= 8;
            dim3 block(big ? 16 : 32, big ? 16 : 32, big ? 4 : 1);
            dim3 grid(div_up(L.h, block.x), div_up(L.w, block.y), div_up(L.g, block.z));
            aggregate_kernel<<<grid, block, 0, s>>>(last && leg > 0, I.T[l].data() + 6 * leg * L.g,
                                                    I.Tinv[l - 1].data() + 6 * leg * Lp.g, L.red, L.w, L.h, L.g,
                                                    dst, Lp.w, Lp.h, Lp.g, prev);
            HELIREC_CUDA_CHECK(cudaGetLastError());
            std::swap(prev, next);
        }
    }
    if (P.clip_outer_circle) {
        const int N = P.grid_size;
        dim3 block(16, 16), grid(div_up(N, 16), div_up(N, 16));
        clip_kernel<<<grid, block, 0, s>>>(out, N);
        HELIREC_CUDA_CHECK(cudaGetLastError());
    }
    HELIREC_CUDA_CHECK(cudaStreamSynchronize(s));
}

}  // namespace helirec::cuda
