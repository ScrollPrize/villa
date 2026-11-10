#include <algorithm>
#include <vector>

#include "common/accumulate_core.hpp"
#include "cuda_utils.cuh"
#include "helirec/accumulate.hpp"
#include "helirec/sino.hpp"

namespace helirec::cuda {

using namespace helirec::accum_core;

namespace {

// threads: x over merged columns, y over rows t, z over grid angles of the block
__global__ void accumulate_kernel(AccumScanView s, const int32_t* __restrict__ offsets, const int32_t* __restrict__ proj,
                                  const int8_t* __restrict__ role, int L, int H, int NX, AccumParams p,
                                  float* __restrict__ R, float* __restrict__ RW) {
    const int X = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    const int jl = blockIdx.z;
    if (X >= NX || t >= H || jl >= L) return;
    const size_t o = (static_cast<size_t>(jl) * H + t) * NX + X;
    float a = R[o], b = RW[o];
    for (int e = offsets[jl]; e < offsets[jl + 1]; ++e) contribute(s, proj[e], role[e], t, X, p, a, b);
    R[o] = a;
    RW[o] = b;
}

// the vertically interpolated values of the block's projections: threads x over detector columns, y over rows t, z
// over the projections (slots)
__global__ void column_values_kernel(AccumScanView s, const int32_t* __restrict__ slot_proj, int H, int stride,
                                     float* __restrict__ out) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (c >= s.det_cols || t >= H) return;
    out[(static_cast<size_t>(blockIdx.z) * H + t) * stride + c] = column_interp(s, slot_proj[blockIdx.z], t, c);
}

__global__ void lum_sums_kernel(const float* R, const float* RW, int L, int H, int NX, int cx, const float* gx,
                                const float* gz, int radius, float* A, float* B) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= L * H) return;
    window_sum(R, RW, H, NX, q / H, q % H, cx, gx, gz, radius, A[q], B[q]);
}

__global__ void lum_apply_kernel(float* R, const float* RW, const float* Wp, const float* Aha, const float* Bha,
                                 const float* Atot, const float* Btot, int LH, int NX, float rep) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int q = blockIdx.y + 65535 * blockIdx.z;  // rows of (L * H) may exceed gridDim.y's limit
    if (x >= NX || q >= LH) return;
    const float d = luminosity_diff(Aha[q], Bha[q], Atot[q], Btot[q]);
    const float dfin = __fdiv_rn(__fmul_rn(d, __fadd_rn(rep, 100.0f)), 200.0f);
    const size_t o = static_cast<size_t>(q) * NX + x;
    R[o] = __fadd_rn(R[o], __fsub_rn(__fmul_rn(d, Wp[o]), __fmul_rn(dfin, RW[o])));
}

__global__ void divide_kernel(float* R, const float* RW, size_t n) {
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x)
        R[i] = RW[i] > 0 ? __fdiv_rn(R[i], RW[i]) : 0.f;
}

__global__ void mend_kernel(float* R, const float* RW, int LH, int NX, float thr, int x0, int x1) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= LH) return;
    mend_row(R + static_cast<size_t>(q) * NX, RW + static_cast<size_t>(q) * NX, NX, thr, x0, x1);
}

__global__ void support_kernel(float* R, float* RW, int LH, int NX, float thr) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q < LH) measured_support(R + static_cast<size_t>(q) * NX, RW + static_cast<size_t>(q) * NX, NX, thr);
}
__global__ void fill_measured_kernel(float* R, const float* RW, int L, int H, int NX, float empty) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= L * H) return;
    const size_t o = static_cast<size_t>(q / H) * H * NX;
    fill_measured_row(R + o, RW + o, H, NX, q % H, empty);
}

template <class T>
DeviceBuffer<T> upload(const std::vector<T>& v, cudaStream_t s) {
    DeviceBuffer<T> d;
    d.upload(v.data(), v.size(), s);
    return d;
}

}  // namespace

// `v` into a device buffer kept between calls (grow-only)
template <class T>
const T* load(DeviceBuffer<T>& d, const std::vector<T>& v, cudaStream_t s) {
    d.ensure(std::max<std::size_t>(1, v.size()));
    if (!v.empty()) HELIREC_CUDA_CHECK(cudaMemcpyAsync(d.data(), v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice, s));
    return d.data();
}

void accumulate(const AccumScanView& view, const ContributorList& c, const AccumParams& p, float* R, float* RW,
                void* stream, const float* weight2, float* R2, float* RW2) {
    cudaStream_t st = as_stream(stream);
    const int L = c.g1 - c.g0;
    if (L <= 0 || p.H <= 0 || p.NX <= 0 || c.proj.empty()) return;
    // (work buffers of the calling thread, kept between blocks: every call finishes before it returns)
    thread_local DeviceBuffer<int32_t> d_off, d_proj, d_slot, d_slot_proj;
    thread_local DeviceBuffer<int8_t> d_role;
    thread_local DeviceBuffer<float> d_col;
    AccumScanView s = view;
    // the vertically interpolated values of the block's projections, once each (they are the same at every column
    // tap of every pixel a sample lands on); measured-only gridding interpolates per corner instead
    std::vector<int32_t> slot, slot_proj;
    if (!p.measured_only && !s.col_val) {
        contributor_slots(c, s.n_proj, slot, slot_proj);
        const std::size_t n_slots = slot_proj.size();
        const bool shift = s.x_exact && s.fract_x != 0.f;
        const int stride = shift ? shift_row_stride(s.det_cols) : s.det_cols;
        d_col.ensure(n_slots * p.H * stride);
        s.col_slot = load(d_slot, slot, st);
        dim3 block(128, 1, 1), grid((s.det_cols + 127) / 128, p.H, static_cast<unsigned>(n_slots));
        column_values_kernel<<<grid, block, 0, st>>>(s, load(d_slot_proj, slot_proj, st), p.H, stride, d_col.data());
        HELIREC_CUDA_CHECK(cudaGetLastError());
        if (shift) shift_rows(d_col.data(), static_cast<int>(n_slots) * p.H, s.det_cols, s.fract_x, st);
        s.col_val = d_col.data();
        s.col_rows = p.H;
        s.col_stride = stride;
        s.col_shifted = shift ? 1 : 0;
    }
    const int32_t* off = load(d_off, c.offsets, st);
    const int32_t* proj = load(d_proj, c.proj, st);
    const int8_t* role = load(d_role, c.role, st);
    dim3 block(128, 1, 1), grid((p.NX + 127) / 128, p.H, L);
    accumulate_kernel<<<grid, block, 0, st>>>(s, off, proj, role, L, p.H, p.NX, p, R, RW);
    HELIREC_CUDA_CHECK(cudaGetLastError());
    if (weight2) {   // the same contributions with the second weight map
        s.weight = weight2;
        accumulate_kernel<<<grid, block, 0, st>>>(s, off, proj, role, L, p.H, p.NX, p, R2, RW2);
        HELIREC_CUDA_CHECK(cudaGetLastError());
    }
    HELIREC_CUDA_CHECK(cudaStreamSynchronize(st));
}

void luminosity_sums(const float* R, const float* RW, int L, int H, int NX, int cx, const LuminosityKernel& k,
                     float* A, float* B, void* stream) {
    cudaStream_t st = as_stream(stream);
    auto gx = upload(k.gx, st);
    auto gz = upload(k.gz, st);
    lum_sums_kernel<<<(L * H + 127) / 128, 128, 0, st>>>(R, RW, L, H, NX, cx, gx.data(), gz.data(), k.radius, A, B);
    HELIREC_CUDA_CHECK(cudaGetLastError());
    HELIREC_CUDA_CHECK(cudaStreamSynchronize(st));
}

void luminosity_apply(float* R, const float* RW, const float* W_prev, const float* A_ha, const float* B_ha,
                      const float* A_tot, const float* B_tot, int L, int H, int NX, float rep, void* stream) {
    const int LH = L * H;
    dim3 grid((NX + 255) / 256, std::min(LH, 65535), (LH + 65534) / 65535);
    lum_apply_kernel<<<grid, 256, 0, as_stream(stream)>>>(R, RW, W_prev, A_ha, B_ha, A_tot, B_tot, L * H, NX, rep);
    HELIREC_CUDA_CHECK(cudaGetLastError());
}

void divide(float* R, const float* RW, std::size_t n, void* stream) {
    divide_kernel<<<static_cast<unsigned>(std::min<size_t>((n + 255) / 256, 65535u * 16u)), 256, 0, as_stream(stream)>>>(R, RW, n);
    HELIREC_CUDA_CHECK(cudaGetLastError());
}

void mend_null_weights(float* R, const float* RW, int L, int H, int NX, float thr, int x0, int x1, void* stream) {
    mend_kernel<<<(L * H + 127) / 128, 128, 0, as_stream(stream)>>>(R, RW, L * H, NX, thr, x0, x1);
    HELIREC_CUDA_CHECK(cudaGetLastError());
}

void prepare_measured(float* R, float* RW, int L, int H, int NX, float thr, bool fill, float empty, void* stream) {
    support_kernel<<<(L * H + 127) / 128, 128, 0, as_stream(stream)>>>(R, RW, L * H, NX, thr);
    HELIREC_CUDA_CHECK(cudaGetLastError());
    if (fill) {
        fill_measured_kernel<<<(L * H + 127) / 128, 128, 0, as_stream(stream)>>>(R, RW, L, H, NX, empty);
        HELIREC_CUDA_CHECK(cudaGetLastError());
    }
}

}  // namespace helirec::cuda
