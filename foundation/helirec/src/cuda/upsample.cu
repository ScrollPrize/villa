#include <cufft.h>

#include "cuda_utils.cuh"
#include "common/interp_core.hpp"
#include "helirec/sino.hpp"

namespace helirec::cuda {

namespace {
__global__ void pad_rows_kernel(const float* in, float* out, int n_rows, int n, int L) {
    const size_t total = static_cast<size_t>(n_rows) * L;
    for (size_t k = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; k < total; k += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int i = static_cast<int>(k % L);
        out[k] = i < n ? in[(k / L) * n + i] : 0.f;
    }
}
__global__ void spread_spectrum_kernel(const cufftComplex* A, cufftComplex* B, int n_rows, int nc, int nco, int L) {
    const size_t total = static_cast<size_t>(n_rows) * nco;
    for (size_t k = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; k < total; k += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int j = static_cast<int>(k % nco);
        cufftComplex v = make_cuFloatComplex(0.f, 0.f);
        if (j < nc) {
            v = A[(k / nco) * nc + j];
            if (L % 2 == 0 && j == L / 2) { v.x *= 0.5f; v.y *= 0.5f; }
        }
        B[k] = v;
    }
}
__global__ void crop_scale_kernel(const float* in, float* out, int n_rows, int m, int Lo, float s) {
    const size_t total = static_cast<size_t>(n_rows) * m;
    for (size_t k = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; k < total; k += static_cast<size_t>(gridDim.x) * blockDim.x)
        out[k] = in[(k / m) * Lo + k % m] * s;
}
unsigned grid_for(size_t n) { return static_cast<unsigned>(std::min<size_t>((n + 255) / 256, 65535u * 16u)); }
void check(cufftResult r, const char* w) {
    if (r != CUFFT_SUCCESS) throw Error(std::string("cuFFT error in ") + w);
}

struct Upsampler {
    int n_rows = 0, L = 0, os = 0;
    cufftHandle fwd = 0, inv = 0;
    DeviceBuffer<float> a, b;
    DeviceBuffer<cufftComplex> A, B;
    ~Upsampler() { release(); }
    void release() {
        if (fwd) { cufftDestroy(fwd); cufftDestroy(inv); fwd = inv = 0; }
    }
    void ensure(int rows, int l, int o) {
        if (rows == n_rows && l == L && o == os) return;
        release();
        n_rows = rows; L = l; os = o;
        int nl = L, nlo = L * os;
        check(cufftPlanMany(&fwd, 1, &nl, nullptr, 1, 0, nullptr, 1, 0, CUFFT_R2C, rows), "plan R2C");
        check(cufftPlanMany(&inv, 1, &nlo, nullptr, 1, 0, nullptr, 1, 0, CUFFT_C2R, rows), "plan C2R");
        a.ensure(static_cast<size_t>(rows) * L);
        b.ensure(static_cast<size_t>(rows) * L * os);
        A.ensure(static_cast<size_t>(rows) * (L / 2 + 1));
        B.ensure(static_cast<size_t>(rows) * (L * os / 2 + 1));
    }
};
}  // namespace

__global__ void bspline_prefilter_kernel(const float* in, float* out, int n_rows, int n, int degree) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_rows) return;
    float* o = out + static_cast<size_t>(r) * n;
    const float* i = in + static_cast<size_t>(r) * n;
    if (o != i)
        for (int k = 0; k < n; ++k) o[k] = i[k];
    interp_core::bspline_prefilter_row(o, n, degree);
}

void bspline_prefilter(const float* in, float* out, int n_rows, int n, int degree, void* stream) {
    bspline_prefilter_kernel<<<(n_rows + 63) / 64, 64, 0, as_stream(stream)>>>(in, out, n_rows, n, degree);
    HELIREC_CUDA_CHECK(cudaGetLastError());
}

void upsample_rows(const float* in, float* out, int n_rows, int n, int os, void* stream) {
    cudaStream_t s = as_stream(stream);
    if (os == 1) {
        HELIREC_CUDA_CHECK(cudaMemcpyAsync(out, in, sizeof(float) * n_rows * n, cudaMemcpyDeviceToDevice, s));
        return;
    }
    thread_local Upsampler U;
    const int L = upsample_fft_length(n);
    U.ensure(n_rows, L, os);
    check(cufftSetStream(U.fwd, s), "stream");
    check(cufftSetStream(U.inv, s), "stream");
    pad_rows_kernel<<<grid_for(static_cast<size_t>(n_rows) * L), 256, 0, s>>>(in, U.a.data(), n_rows, n, L);
    check(cufftExecR2C(U.fwd, U.a.data(), U.A.data()), "R2C");
    spread_spectrum_kernel<<<grid_for(static_cast<size_t>(n_rows) * (L * os / 2 + 1)), 256, 0, s>>>(
        U.A.data(), U.B.data(), n_rows, L / 2 + 1, L * os / 2 + 1, L);
    check(cufftExecC2R(U.inv, U.B.data(), U.b.data()), "C2R");
    crop_scale_kernel<<<grid_for(static_cast<size_t>(n_rows) * n * os), 256, 0, s>>>(U.b.data(), out, n_rows, n * os, L * os,
                                                                                  1.f / static_cast<float>(L));
    HELIREC_CUDA_CHECK(cudaGetLastError());
}

}  // namespace helirec::cuda
