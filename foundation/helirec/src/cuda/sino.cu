#include <cmath>
#include <vector>

#include "common/rebalance_core.hpp"
#include "common/sino_core.hpp"
#include "cuda_utils.cuh"

namespace helirec::cuda {

namespace {

unsigned grid_for(size_t n) { return static_cast<unsigned>(std::min<size_t>((n + 255) / 256, 65535u * 16u)); }

__global__ void minus_log_kernel(float* x, size_t n, float lo, float hi) {
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x)
        x[i] = sino_core::minus_log(x[i], lo, hi);
}

__global__ void multiply_kernel(float* x, const float* w, size_t n) {
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x)
        x[i] = __fmul_rn(x[i], w[i]);
}

__global__ void add_scaled_kernel(float* x, const float* y, float s, size_t n) {
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x)
        x[i] = __fadd_rn(x[i], __fmul_rn(s, y[i]));
}

__global__ void weight_difference_kernel(const float* x, const float* w, const float* w2, float* t, size_t n) {
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x)
        t[i] = __fmul_rn(x[i], __fsub_rn(w2[i], w[i]));
}

__global__ void multiply_add_kernel(float* x, const float* w, const float* t, size_t n) {
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x)
        x[i] = __fadd_rn(__fmul_rn(x[i], w[i]), t[i]);
}

__global__ void angular_fir_kernel(const float* __restrict__ in, float* __restrict__ out, int n_proj, int n_cols,
                                   const float* __restrict__ g, int radius, const float* __restrict__ support) {
    const size_t n = static_cast<size_t>(n_proj) * n_cols;
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x)
        out[i] = sino_core::angular_fir_at(in, support, n_proj, n_cols, static_cast<int>(i / n_cols), static_cast<int>(i % n_cols), g, radius);
}

// adj per row computed on the host-side formula but on device: one block per row
__global__ void ramp_adj_kernel(const float* sino, float* adj, RampGeometry g) {
    const int y = blockIdx.x;
    if (y >= g.n_proj || threadIdx.x != 0) return;
    if (g.R <= 0) { adj[y] = 0.f; return; }
    const int m = (y + g.n_proj / 2) % g.n_proj;
    const float* row = sino + static_cast<size_t>(y) * g.n_bins;
    const float* mrow = sino + static_cast<size_t>(m) * g.n_bins;
    if (!g.legacy) { adj[y] = sino_core::ramp_adj_exact(row, mrow, g); return; }
    const float sigma = __fdiv_rn(static_cast<float>(g.R), 3.0f);
    const float inv2s2 = __fdiv_rn(1.0f, __fmul_rn(__fmul_rn(2.0f, sigma), sigma));
    const float my_rot_f = g.rot_center, two_rots_f = __fmul_rn(2.0f, g.rot_center);
    const float mirror_center = __fsub_rn(two_rots_f, my_rot_f);
    const int i0 = max(0, g.my_rot - g.R), i1 = min(g.n_bins - 1, g.my_rot + g.R);
    float sa = 0, wa = 0, sb = 0, wb = 0;
    for (int ix = i0; ix <= i1; ++ix) {
        const float dA = __fsub_rn(static_cast<float>(ix), my_rot_f);
        const float wA = expf(-__fmul_rn(__fmul_rn(dA, dA), inv2s2));
        sa = __fadd_rn(sa, __fmul_rn(row[ix], wA));
        wa = __fadd_rn(wa, wA);
        const int xm = g.two_rots - ix;
        if (xm >= 0 && xm < g.n_bins) {
            const float dB = __fsub_rn(static_cast<float>(xm), mirror_center);
            const float wB = expf(-__fmul_rn(__fmul_rn(dB, dB), inv2s2));
            sb = __fadd_rn(sb, __fmul_rn(mrow[xm], wB));
            wb = __fadd_rn(wb, wB);
        }
    }
    adj[y] = (wa > 0 && wb > 0) ? __fsub_rn(__fdiv_rn(sb, wb), __fdiv_rn(sa, wa)) : 0.f;
}

__global__ void ramp_pad_kernel(const float* sino, const float* adj, float* padded, RampGeometry g) {
    const size_t total = static_cast<size_t>(g.n_proj) * g.npad;
    for (size_t k = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; k < total; k += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int x = static_cast<int>(k % g.npad), y = static_cast<int>(k / g.npad);
        const int m = (y + g.n_proj / 2) % g.n_proj;
        padded[k] = sino_core::ramp_pad_value(sino + static_cast<size_t>(y) * g.n_bins, sino + static_cast<size_t>(m) * g.n_bins,
                                              adj[y], x, g);
    }
}

__global__ void ramp_crop_kernel(const float* padded, float* out, RampGeometry g) {
    const size_t total = static_cast<size_t>(g.n_proj) * g.n_bins;
    for (size_t k = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; k < total; k += static_cast<size_t>(gridDim.x) * blockDim.x)
        out[k] = padded[(k / g.n_bins) * g.npad + k % g.n_bins];
}

// ---- rebalance
constexpr int kMaxContrib = 8;
struct DevContrib { int k0, k1; float alpha; int flags; };  // flags: 1 has_two, 2 odd

__global__ void nonempty_kernel(const float* W, size_t plane, size_t row_off, int n_cols, int n_proj, char* out) {
    const int i = blockIdx.x;
    if (i >= n_proj) return;
    const float* w = W + i * plane + row_off;
    __shared__ int any;
    if (threadIdx.x == 0) any = 0;
    __syncthreads();
    for (int c = threadIdx.x; c < n_cols; c += blockDim.x)
        if (w[c] != 0.0f) any = 1;
    __syncthreads();
    if (threadIdx.x == 0) out[i] = static_cast<char>(any);
}

__global__ void average_planes_kernel(const float* a, const float* b, float* out, size_t n) {
    for (size_t k = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; k < n; k += static_cast<size_t>(gridDim.x) * blockDim.x)
        out[k] = 0.5f * (a[k] + b[k]);
}

__device__ inline float mapped_dev(const float* row, int c, const int2* osrc, const float2* ow) {
    const int2 s = osrc[c];
    const float2 w = ow[c];
    float v = 0.f;
    if (s.x >= 0) v += w.x * row[s.x];
    if (s.y >= 0) v += w.y * row[s.y];
    return v;
}

// Wsrc: (n_proj, n_cols) row `row` of W; writes the balanced weights into W
__global__ void rebalance_row_kernel(const float* __restrict__ Wsrc, float* W, size_t plane, size_t row_off, int n_proj,
                                     int n_cols, const DevContrib* table, const int* count, const int2* osrc,
                                     const float2* ow) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y;
    if (c >= n_cols || i >= n_proj) return;
    const float wi = Wsrc[static_cast<size_t>(i) * n_cols + c];
    float B = wi;
    for (int t = 0; t < count[i]; ++t) {
        const DevContrib m = table[i * kMaxContrib + t];
        const float* r0 = Wsrc + static_cast<size_t>(m.k0) * n_cols;
        const float* r1 = (m.flags & 1) ? Wsrc + static_cast<size_t>(m.k1) * n_cols : nullptr;
        const float beta = 1.0f - m.alpha, alpha = m.alpha;
        if (!(m.flags & 2)) B += (m.flags & 1) ? beta * r0[c] + alpha * r1[c] : r0[c];
        else B += (m.flags & 1) ? beta * mapped_dev(r0, c, osrc, ow) + alpha * mapped_dev(r1, c, osrc, ow)
                                : mapped_dev(r0, c, osrc, ow);
    }
    W[i * plane + row_off + c] = wi == 0.0f ? 0.0f : wi / B;
}

}  // namespace

std::vector<std::pair<int, int>> rebalance_holes(const float* W, int n_proj, int n_rows, int n_cols, void* stream) {
    cudaStream_t s = as_stream(stream);
    const size_t plane = static_cast<size_t>(n_rows) * n_cols;
    DeviceBuffer<char> d_ne(n_proj);
    nonempty_kernel<<<n_proj, 256, 0, s>>>(W, plane, static_cast<size_t>(n_rows / 2) * n_cols, n_cols, n_proj, d_ne.data());
    HELIREC_CUDA_CHECK(cudaGetLastError());
    std::vector<char> nonempty(n_proj);
    d_ne.download(nonempty.data(), s);
    bool any = false;
    for (char v : nonempty) any |= v;
    if (!any) return {{-1, -1}};  // all empty: nothing to do (marker)
    return rebalance_core::holes_from(nonempty);
}

void fill_holes(float* X, const std::vector<std::pair<int, int>>& holes, int n_proj, int n_rows, int n_cols, void* stream) {
    if (holes.empty() || holes.front().first < 0) return;
    cudaStream_t s = as_stream(stream);
    const size_t plane = static_cast<size_t>(n_rows) * n_cols;
    DeviceBuffer<float> f(plane);
    for (auto [a, e] : holes) {
        const int left = (a + n_proj - 1) % n_proj, right = e % n_proj;
        average_planes_kernel<<<grid_for(plane), 256, 0, s>>>(X + left * plane, X + right * plane, f.data(), plane);
        HELIREC_CUDA_CHECK(cudaGetLastError());
        for (int idx = a; idx < e; ++idx)
            HELIREC_CUDA_CHECK(cudaMemcpyAsync(X + (idx % n_proj) * plane, f.data(), plane * sizeof(float), cudaMemcpyDeviceToDevice, s));
    }
    HELIREC_CUDA_CHECK(cudaStreamSynchronize(s));
}

void balance_weights(float* W, int n_proj, int n_rows, int n_cols, float rot_center, bool legacy, void* stream) {
    cudaStream_t s = as_stream(stream);
    const size_t plane = static_cast<size_t>(n_rows) * n_cols;
    const auto table = rebalance_core::contribution_table(n_proj, legacy);
    const auto omap = rebalance_core::odd_shift_map(n_cols, rot_center);
    std::vector<DevContrib> tab(static_cast<size_t>(n_proj) * kMaxContrib);
    std::vector<int> cnt(n_proj);
    for (int i = 0; i < n_proj; ++i) {
        if (table[i].size() > static_cast<size_t>(kMaxContrib)) throw Error("rebalance: too many contributions for the CUDA table");
        cnt[i] = static_cast<int>(table[i].size());
        for (size_t t = 0; t < table[i].size(); ++t) {
            const auto& m = table[i][t];
            tab[i * kMaxContrib + t] = {m.k0, m.k1, m.alpha, (m.has_two ? 1 : 0) | (m.odd ? 2 : 0)};
        }
    }
    std::vector<int2> osrc(n_cols);
    std::vector<float2> ow(n_cols);
    for (int c = 0; c < n_cols; ++c) {
        osrc[c] = make_int2(omap[c].src0, omap[c].src1);
        ow[c] = make_float2(omap[c].w0, omap[c].w1);
    }
    DeviceBuffer<DevContrib> d_tab;
    DeviceBuffer<int> d_cnt;
    DeviceBuffer<int2> d_osrc;
    DeviceBuffer<float2> d_ow;
    d_tab.upload(tab.data(), tab.size(), s);
    d_cnt.upload(cnt.data(), cnt.size(), s);
    d_osrc.upload(osrc.data(), osrc.size(), s);
    d_ow.upload(ow.data(), ow.size(), s);
    DeviceBuffer<float> Wsrc(static_cast<size_t>(n_proj) * n_cols);
    for (int row = 0; row < n_rows; ++row) {
        const size_t row_off = static_cast<size_t>(row) * n_cols;
        HELIREC_CUDA_CHECK(cudaMemcpy2DAsync(Wsrc.data(), n_cols * sizeof(float), W + row_off, plane * sizeof(float),
                                             n_cols * sizeof(float), n_proj, cudaMemcpyDeviceToDevice, s));
        dim3 grid((n_cols + 255) / 256, n_proj);
        rebalance_row_kernel<<<grid, 256, 0, s>>>(Wsrc.data(), W, plane, row_off, n_proj, n_cols, d_tab.data(),
                                                  d_cnt.data(), d_osrc.data(), d_ow.data());
        HELIREC_CUDA_CHECK(cudaGetLastError());
    }
    HELIREC_CUDA_CHECK(cudaStreamSynchronize(s));
}

int rebalance(float* R, float* W, int n_proj, int n_rows, int n_cols, float rot_center, bool legacy, void* stream) {
    const auto holes = rebalance_holes(W, n_proj, n_rows, n_cols, stream);
    if (!holes.empty() && holes.front().first < 0) return 0;  // all angles empty
    fill_holes(R, holes, n_proj, n_rows, n_cols, stream);
    fill_holes(W, holes, n_proj, n_rows, n_cols, stream);
    balance_weights(W, n_proj, n_rows, n_cols, rot_center, legacy, stream);
    return static_cast<int>(holes.size());
}

void minus_log(float* x, std::size_t n, float lo, float hi, void* stream) {
    minus_log_kernel<<<grid_for(n), 256, 0, as_stream(stream)>>>(x, n, lo, hi);
    HELIREC_CUDA_CHECK(cudaGetLastError());
}

void multiply(float* x, const float* w, std::size_t n, void* stream) {
    multiply_kernel<<<grid_for(n), 256, 0, as_stream(stream)>>>(x, w, n);
    HELIREC_CUDA_CHECK(cudaGetLastError());
}

void add_scaled(float* x, const float* y, float s, std::size_t n, void* stream) {
    add_scaled_kernel<<<grid_for(n), 256, 0, as_stream(stream)>>>(x, y, s, n);
    HELIREC_CUDA_CHECK(cudaGetLastError());
}

void weight_difference(const float* x, const float* w, const float* w2, float* t, std::size_t n, void* stream) {
    weight_difference_kernel<<<grid_for(n), 256, 0, as_stream(stream)>>>(x, w, w2, t, n);
    HELIREC_CUDA_CHECK(cudaGetLastError());
}

void multiply_add(float* x, const float* w, const float* t, std::size_t n, void* stream) {
    multiply_add_kernel<<<grid_for(n), 256, 0, as_stream(stream)>>>(x, w, t, n);
    HELIREC_CUDA_CHECK(cudaGetLastError());
}

void angular_fir(const float* in, float* out, int n_proj, int n_cols, const float* g, int radius, const float* support,
                 void* stream) {
    if (radius < 0 || 2 * radius + 1 > n_proj) throw Error("angular_fir: bad radius");
    cudaStream_t s = as_stream(stream);
    DeviceBuffer<float> d_g;
    d_g.upload(g, static_cast<std::size_t>(radius) + 1, s);
    angular_fir_kernel<<<grid_for(static_cast<size_t>(n_proj) * n_cols), 256, 0, s>>>(in, out, n_proj, n_cols, d_g.data(), radius, support);
    HELIREC_CUDA_CHECK(cudaGetLastError());
    HELIREC_CUDA_CHECK(cudaStreamSynchronize(s));  // d_g is freed on return
}

void ramp_pad(const float* sino, float* padded, const RampGeometry& g, void* stream) {
    cudaStream_t s = as_stream(stream);
    thread_local DeviceBuffer<float> adj;  // reused: no allocation / synchronisation per call
    adj.ensure(g.n_proj);
    ramp_adj_kernel<<<g.n_proj, 32, 0, s>>>(sino, adj.data(), g);
    HELIREC_CUDA_CHECK(cudaGetLastError());
    ramp_pad_kernel<<<grid_for(static_cast<size_t>(g.n_proj) * g.npad), 256, 0, s>>>(sino, adj.data(), padded, g);
    HELIREC_CUDA_CHECK(cudaGetLastError());
}

void ramp_crop(const float* padded, float* out, const RampGeometry& g, void* stream) {
    ramp_crop_kernel<<<grid_for(static_cast<size_t>(g.n_proj) * g.n_bins), 256, 0, as_stream(stream)>>>(padded, out, g);
    HELIREC_CUDA_CHECK(cudaGetLastError());
}

}  // namespace helirec::cuda
