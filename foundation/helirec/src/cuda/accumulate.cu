#include "common/accumulate_core.hpp"
#include "cuda_utils.cuh"
#include "helirec/accumulate.hpp"

namespace helirec::cuda {

using namespace helirec::accum_core;

namespace {

// threads: x over merged columns, y over rows t, z over grid angles of the block
__global__ void accumulate_kernel(AccumScanView s, const int32_t* __restrict__ offsets, const int32_t* __restrict__ proj,
                                  const int8_t* __restrict__ role, int L, int H, int NX, float non_zero, int border,
                                  float* __restrict__ R, float* __restrict__ RW) {
    const int X = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    const int jl = blockIdx.z;
    if (X >= NX || t >= H || jl >= L) return;
    const size_t o = (static_cast<size_t>(jl) * H + t) * NX + X;
    float a = R[o], b = RW[o];
    for (int e = offsets[jl]; e < offsets[jl + 1]; ++e) contribute(s, proj[e], role[e], t, X, NX, non_zero, border, a, b);
    R[o] = a;
    RW[o] = b;
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

template <class T>
DeviceBuffer<T> upload(const std::vector<T>& v, cudaStream_t s) {
    DeviceBuffer<T> d;
    d.upload(v.data(), v.size(), s);
    return d;
}

}  // namespace

void accumulate(const AccumScanView& s, const ContributorList& c, const AccumParams& p, float* R, float* RW,
                void* stream) {
    cudaStream_t st = as_stream(stream);
    const int L = c.g1 - c.g0;
    auto d_off = upload(c.offsets, st);
    auto d_proj = upload(c.proj, st);
    auto d_role = upload(c.role, st);
    dim3 block(128, 1, 1), grid((p.NX + 127) / 128, p.H, L);
    accumulate_kernel<<<grid, block, 0, st>>>(s, d_off.data(), d_proj.data(), d_role.data(), L, p.H, p.NX, p.non_zero,
                                              p.border, R, RW);
    HELIREC_CUDA_CHECK(cudaGetLastError());
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

}  // namespace helirec::cuda
