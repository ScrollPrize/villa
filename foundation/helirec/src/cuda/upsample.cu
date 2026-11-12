#include <cufft.h>

#include "cuda_utils.cuh"
#include "common/detector_map_core.hpp"
#include "common/interp_core.hpp"
#include "common/sino_core.hpp"
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

// rows (n_rows, S) holding n samples: the bridge to the transform's length L
__global__ void shift_bridge_kernel(float* rows, int n_rows, int n, int L, int S) {
    const int pad = L - n;
    const size_t total = static_cast<size_t>(n_rows) * pad;
    for (size_t q = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; q < total; q += static_cast<size_t>(gridDim.x) * blockDim.x) {
        float* a = rows + (q / pad) * S;
        const int i = n + static_cast<int>(q % pad);
        a[i] = sino_core::shift_bridge(a[n - 1], a[0], i, n, L);
    }
}
// shifts (n_shifts, or null): row r < n_shifts is delayed by shift + shifts[r]
__global__ void shift_phase_kernel(cufftComplex* A, int n_rows, int nc, int L, float shift, const float* shifts, int n_shifts) {
    const size_t total = static_cast<size_t>(n_rows) * nc;
    for (size_t q = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; q < total; q += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int r = static_cast<int>(q / nc);
        sino_core::shift_bin(A[q].x, A[q].y, static_cast<int>(q % nc), L, shifts && r < n_shifts ? shift + shifts[r] : shift);
    }
}

// In-place transforms of batches of `batch` rows (a plan is tied to its batch size: the rows are taken `batch` at a
// time, the last, incomplete batch through `tail`).
struct RowShifter {
    int L = 0, batch = 0;
    cufftHandle fwd = 0, inv = 0;
    DeviceBuffer<float> tail;
    ~RowShifter() { release(); }
    void release() {
        if (fwd) { cufftDestroy(fwd); cufftDestroy(inv); fwd = inv = 0; }
    }
    void ensure(int l, int b) {
        if (l == L && b == batch) return;
        release();
        L = l; batch = b;
        int n = L, real = 2 * (L / 2 + 1), cplx = L / 2 + 1;
        check(cufftPlanMany(&fwd, 1, &n, &real, 1, real, &cplx, 1, cplx, CUFFT_R2C, batch), "plan R2C (shift)");
        check(cufftPlanMany(&inv, 1, &n, &cplx, 1, cplx, &real, 1, real, CUFFT_C2R, batch), "plan C2R (shift)");
        tail.ensure(static_cast<size_t>(batch) * real);
        HELIREC_CUDA_CHECK(cudaMemset(tail.data(), 0, static_cast<size_t>(batch) * real * sizeof(float)));
    }
};
}  // namespace

void shift_rows(float* rows, int n_rows, int n, float shift, void* stream, const float* shifts) {
    if (n_rows <= 0 || n <= 1) return;
    cudaStream_t s = as_stream(stream);
    thread_local RowShifter U;
    const int L = upsample_fft_length(n), S = shift_row_stride(n), nc = L / 2 + 1;
    U.ensure(L, 4096);
    check(cufftSetStream(U.fwd, s), "stream");
    check(cufftSetStream(U.inv, s), "stream");
    shift_bridge_kernel<<<grid_for(static_cast<size_t>(n_rows) * (L - n)), 256, 0, s>>>(rows, n_rows, n, L, S);
    for (int r0 = 0; r0 < n_rows; r0 += U.batch) {
        const int nr = std::min(U.batch, n_rows - r0);
        float* a = rows + static_cast<size_t>(r0) * S;
        float* w = a;
        if (nr < U.batch) {   // (the rows of the buffer beyond nr are zero or earlier rows: finite either way)
            w = U.tail.data();
            HELIREC_CUDA_CHECK(cudaMemcpyAsync(w, a, static_cast<size_t>(nr) * S * sizeof(float), cudaMemcpyDeviceToDevice, s));
        }
        check(cufftExecR2C(U.fwd, w, reinterpret_cast<cufftComplex*>(w)), "R2C (shift)");
        const int np = w == a ? nr : U.batch;   // (every row of `tail`: its spare rows must stay bounded)
        shift_phase_kernel<<<grid_for(static_cast<size_t>(np) * nc), 256, 0, s>>>(reinterpret_cast<cufftComplex*>(w), np, nc, L, shift,
                                                                                    shifts ? shifts + r0 : nullptr, nr);
        check(cufftExecC2R(U.inv, reinterpret_cast<cufftComplex*>(w), w), "C2R (shift)");
        if (w != a)
            HELIREC_CUDA_CHECK(cudaMemcpyAsync(a, w, static_cast<size_t>(nr) * S * sizeof(float), cudaMemcpyDeviceToDevice, s));
    }
    HELIREC_CUDA_CHECK(cudaGetLastError());
}

namespace {
// the rows' transforms A (n_rows, nc) times `factor` (nc), laid in the nco bins of the oversampled transform
__global__ void resample_spread_kernel(const cufftComplex* A, cufftDoubleComplex* B, int n_rows, int nc, int nco, int L,
                                       const double* factor) {
    const size_t total = static_cast<size_t>(n_rows) * nco;
    for (size_t q = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; q < total; q += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int k = static_cast<int>(q % nco);
        cufftDoubleComplex v = make_cuDoubleComplex(0.0, 0.0);
        if (k < nc) {
            const cufftComplex a = A[(q / nco) * nc + k];   // (the Nyquist bin of a real row is real)
            v = make_cuDoubleComplex(hd::dmul(a.x, factor[k]), 2 * k == L ? 0.0 : hd::dmul(a.y, factor[k]));
        }
        B[q] = v;
    }
}
// g (n_rows, So): the oversampled rows (M samples each); rows (n_rows, S): sample j of row r <- its value at
// positions[r n + j], or at the position its detector map gives (row, geo). A thread takes B consecutive samples
// of a row (what a row's positions share is computed once for them).
__global__ void resample_gather_kernel(const double* g, float* rows, int n_rows, int n, int S, int So, int M, const double* positions,
                                       const ResampleRow* row, ResampleGeometry geo, const double* kernel) {
    constexpr int B = resample_block;
    const int strips = (n + B - 1) / B;
    const size_t total = static_cast<size_t>(n_rows) * strips;
    for (size_t q = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; q < total; q += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const size_t r = q / strips;
        const int j0 = static_cast<int>(q % strips) * B;
        double X[B], p[B], v[B];
        const int count = n - j0 < B ? n - j0 : B;
        if (positions) {
            for (int l = 0; l < B; ++l) p[l] = l < count ? positions[r * n + j0 + l] : 0.0;
        } else {
            const ResampleRow w = row[r];
            for (int l = 0; l < B; ++l) X[l] = static_cast<double>(j0 + l) - static_cast<double>(w.shift);
            map_core::source<B>(geo.map, w.y, w.row_slope, geo.slope_col, w.col_scale, geo.scale_col, w.row_scale, geo.scale_row, X, p, v);
        }
        sino_core::resample_values<B>(g + r * So, M, p, kernel, rows + r * S + j0, count);
    }
}

// the oversampled transform back, in batches as RowShifter's, and the kernel's factors
struct RowResampler {
    int L = 0, batch = 0;
    cufftHandle inv = 0;
    DeviceBuffer<double> grid, factor, kernel;
    ~RowResampler() { release(); }
    void release() {
        if (inv) { cufftDestroy(inv); inv = 0; }
    }
    void ensure(int n, int l, int b) {
        if (l == L && b == batch) return;
        release();
        L = l; batch = b;
        int m = resample_oversampling * L, cplx = m / 2 + 1, real = 2 * cplx;
        check(cufftPlanMany(&inv, 1, &m, &cplx, 1, cplx, &real, 1, real, CUFFT_Z2D, batch), "plan Z2D (resample)");
        grid.ensure(static_cast<size_t>(batch) * real);
        const std::vector<double> f = resample_deconvolution(n);
        factor.upload(f.data(), f.size());
        kernel.upload(resample_kernel().data(), resample_kernel().size());
        HELIREC_CUDA_CHECK(cudaStreamSynchronize(nullptr));
    }
};

void resample_rows_at(float* rows, int n_rows, int n, const double* positions, const ResampleRow* row, const ResampleGeometry& geo,
                      void* stream) {
    if (n_rows <= 0 || n <= 1) return;
    cudaStream_t s = as_stream(stream);
    thread_local RowShifter U;
    thread_local RowResampler V;
    const int L = upsample_fft_length(n), S = shift_row_stride(n), nc = L / 2 + 1;
    const int M = resample_oversampling * L, nco = M / 2 + 1, So = 2 * nco;
    U.ensure(L, 4096);
    V.ensure(n, L, U.batch);
    check(cufftSetStream(U.fwd, s), "stream");
    check(cufftSetStream(V.inv, s), "stream");
    shift_bridge_kernel<<<grid_for(static_cast<size_t>(n_rows) * (L - n)), 256, 0, s>>>(rows, n_rows, n, L, S);
    for (int r0 = 0; r0 < n_rows; r0 += U.batch) {
        const int nr = std::min(U.batch, n_rows - r0);
        float* a = rows + static_cast<size_t>(r0) * S;
        float* w = a;
        if (nr < U.batch) {   // (the spare rows of the buffer: zero, so that every transform stays finite)
            w = U.tail.data();
            HELIREC_CUDA_CHECK(cudaMemcpyAsync(w, a, static_cast<size_t>(nr) * S * sizeof(float), cudaMemcpyDeviceToDevice, s));
            HELIREC_CUDA_CHECK(cudaMemsetAsync(w + static_cast<size_t>(nr) * S, 0, static_cast<size_t>(U.batch - nr) * S * sizeof(float), s));
        }
        check(cufftExecR2C(U.fwd, w, reinterpret_cast<cufftComplex*>(w)), "R2C (resample)");
        cufftDoubleComplex* B = reinterpret_cast<cufftDoubleComplex*>(V.grid.data());
        resample_spread_kernel<<<grid_for(static_cast<size_t>(U.batch) * nco), 256, 0, s>>>(reinterpret_cast<cufftComplex*>(w), B, U.batch,
                                                                                         nc, nco, L, V.factor.data());
        check(cufftExecZ2D(V.inv, B, V.grid.data()), "Z2D (resample)");
        resample_gather_kernel<<<grid_for(static_cast<size_t>(nr) * ((n + resample_block - 1) / resample_block)), 256, 0, s>>>(
            V.grid.data(), a, nr, n, S, So, M, positions ? positions + static_cast<size_t>(r0) * n : nullptr, row ? row + r0 : nullptr, geo,
            V.kernel.data());
    }
    HELIREC_CUDA_CHECK(cudaGetLastError());
}
}  // namespace

void resample_rows(float* rows, int n_rows, int n, const double* positions, void* stream) {
    resample_rows_at(rows, n_rows, n, positions, nullptr, ResampleGeometry{}, stream);
}

void resample_rows(float* rows, int n_rows, int n, const ResampleRow* row, const ResampleGeometry& g, void* stream) {
    resample_rows_at(rows, n_rows, n, nullptr, row, g, stream);
}

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
