#include <cufft.h>

#include "cuda_utils.cuh"
#include "cufft_utils.cuh"
#include "common/detector_map_core.hpp"
#include "common/interp_core.hpp"
#include "common/sino_core.hpp"
#include "vc_tomography/sino.hpp"

namespace vc_tomography::cuda {

namespace {
__global__ void pad_rows_kernel(const float* in, float* out, int n_rows, int n, int L) {
    const size_t total = static_cast<size_t>(n_rows) * L;
    VC_TOMOGRAPHY_GRID_STRIDE(k, total) {
        const int i = static_cast<int>(k % L);
        out[k] = i < n ? in[(k / L) * n + i] : 0.f;
    }
}
__global__ void spread_spectrum_kernel(const cufftComplex* A, cufftComplex* B, int n_rows, int nc, int nco, int L) {
    const size_t total = static_cast<size_t>(n_rows) * nco;
    VC_TOMOGRAPHY_GRID_STRIDE(k, total) {
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
    VC_TOMOGRAPHY_GRID_STRIDE(k, total)
        out[k] = in[(k / m) * Lo + k % m] * s;
}
// The rows are transformed kUpsampleBatch at a time (a sinogram of tens of thousands of rows would otherwise hold
// its padded and oversampled copies and their spectra whole); a plan per batch size: the full batches' and the last
// one's. (The plans work in the thread's shared work area; the rows and their spectra are its shared buffers:
// cufft_utils.cuh)
constexpr int kUpsampleBatch = 4096;
struct Upsampler {
    int L = 0, os = 0;
    struct Plans { int rows = 0; CufftPlan fwd, inv; };
    std::vector<Plans> plans;
    Plans& plans_for(int rows, int l, int o) {
        if (l != L || o != os) plans.clear();
        L = l; os = o;
        for (auto& p : plans)
            if (p.rows == rows) return p;
        if (plans.size() >= 2) plans.erase(plans.begin());
        Plans p;
        p.rows = rows;
        int nl = L, nlo = L * os;
        p.fwd.make(1, &nl, CUFFT_R2C, rows, "plan R2C");
        p.inv.make(1, &nlo, CUFFT_C2R, rows, "plan C2R");
        plans.push_back(std::move(p));
        return plans.back();
    }
};

// rows (n_rows, S) holding n samples: the bridge to the transform's length L
__global__ void shift_bridge_kernel(float* rows, int n_rows, int n, int L, int S) {
    const int pad = L - n;
    const size_t total = static_cast<size_t>(n_rows) * pad;
    VC_TOMOGRAPHY_GRID_STRIDE(q, total) {
        float* a = rows + (q / pad) * S;
        const int i = n + static_cast<int>(q % pad);
        a[i] = sino_core::shift_bridge(a[n - 1], a[0], i, n, L);
    }
}
// shifts (n_shifts, or null): row r < n_shifts is delayed by shift + shifts[r]
__global__ void shift_phase_kernel(cufftComplex* A, int n_rows, int nc, int L, float shift, const float* shifts, int n_shifts) {
    const size_t total = static_cast<size_t>(n_rows) * nc;
    VC_TOMOGRAPHY_GRID_STRIDE(q, total) {
        const int r = static_cast<int>(q / nc);
        sino_core::shift_bin(A[q].x, A[q].y, static_cast<int>(q % nc), L, shifts && r < n_shifts ? shift + shifts[r] : shift);
    }
}

// In-place transforms of batches of `batch` rows (a plan is tied to its batch size: the rows are taken `batch` at a
// time, the last, incomplete batch through `tail`).
struct RowShifter {
    int L = 0, batch = 0;
    CufftPlan fwd, inv;
    DeviceBuffer<float> tail;
    void ensure(int l, int b) {
        if (l == L && b == batch) return;
        L = l; batch = b;
        int n = L, real = 2 * (L / 2 + 1), cplx = L / 2 + 1;
        fwd.make(1, &n, &real, 1, real, &cplx, 1, cplx, CUFFT_R2C, batch, "plan R2C (shift)");
        inv.make(1, &n, &cplx, 1, cplx, &real, 1, real, CUFFT_C2R, batch, "plan C2R (shift)");
        tail.ensure(static_cast<size_t>(batch) * real);
        VC_TOMOGRAPHY_CUDA_CHECK(cudaMemset(tail.data(), 0, static_cast<size_t>(batch) * real * sizeof(float)));
    }
};
}  // namespace

void shift_rows(float* rows, int n_rows, int n, float shift, void* stream, const float* shifts) {
    if (n_rows <= 0 || n <= 1) return;
    cudaStream_t s = as_stream(stream);
    thread_local RowShifter U;
    thread_local ThreadScratch scratch(U);
    const int L = upsample_fft_length(n), S = shift_row_stride(n), nc = L / 2 + 1;
    U.ensure(L, 4096);
    shift_bridge_kernel<<<grid_for(static_cast<size_t>(n_rows) * (L - n)), 256, 0, s>>>(rows, n_rows, n, L, S);
    for (int r0 = 0; r0 < n_rows; r0 += U.batch) {
        const int nr = std::min(U.batch, n_rows - r0);
        float* a = rows + static_cast<size_t>(r0) * S;
        float* w = a;
        if (nr < U.batch) {   // (the rows of the buffer beyond nr are zero or earlier rows: finite either way)
            w = U.tail.data();
            VC_TOMOGRAPHY_CUDA_CHECK(cudaMemcpyAsync(w, a, static_cast<size_t>(nr) * S * sizeof(float), cudaMemcpyDeviceToDevice, s));
        }
        cufft_check(cufftExecR2C(U.fwd.bind(s), w, reinterpret_cast<cufftComplex*>(w)), "R2C (shift)");
        const int np = w == a ? nr : U.batch;   // (every row of `tail`: its spare rows must stay bounded)
        shift_phase_kernel<<<grid_for(static_cast<size_t>(np) * nc), 256, 0, s>>>(reinterpret_cast<cufftComplex*>(w), np, nc, L, shift,
                                                                                    shifts ? shifts + r0 : nullptr, nr);
        cufft_check(cufftExecC2R(U.inv.bind(s), reinterpret_cast<cufftComplex*>(w), w), "C2R (shift)");
        if (w != a)
            VC_TOMOGRAPHY_CUDA_CHECK(cudaMemcpyAsync(a, w, static_cast<size_t>(nr) * S * sizeof(float), cudaMemcpyDeviceToDevice, s));
    }
    VC_TOMOGRAPHY_CUDA_CHECK(cudaGetLastError());
}

namespace {
// the rows' transforms A (n_rows, nc) times `factor` (nc), laid in the nco bins of the oversampled transform
__global__ void resample_spread_kernel(const cufftComplex* A, cufftDoubleComplex* B, int n_rows, int nc, int nco, int L,
                                       const double* factor) {
    const size_t total = static_cast<size_t>(n_rows) * nco;
    VC_TOMOGRAPHY_GRID_STRIDE(q, total) {
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
    VC_TOMOGRAPHY_GRID_STRIDE(q, total) {
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
    CufftPlan inv;
    DeviceBuffer<double> grid, factor, kernel;
    void ensure(int n, int l, int b) {
        if (l == L && b == batch) return;
        L = l; batch = b;
        int m = resample_oversampling * L, cplx = m / 2 + 1, real = 2 * cplx;
        inv.make(1, &m, &cplx, 1, cplx, &real, 1, real, CUFFT_Z2D, batch, "plan Z2D (resample)");
        grid.ensure(static_cast<size_t>(batch) * real);
        const std::vector<double> f = resample_deconvolution(n);
        factor.upload(f.data(), f.size());
        kernel.upload(resample_kernel().data(), resample_kernel().size());
        VC_TOMOGRAPHY_CUDA_CHECK(cudaStreamSynchronize(nullptr));
    }
};

void resample_rows_at(float* rows, int n_rows, int n, const double* positions, const ResampleRow* row, const ResampleGeometry& geo,
                      void* stream) {
    if (n_rows <= 0 || n <= 1) return;
    cudaStream_t s = as_stream(stream);
    thread_local RowShifter U;
    thread_local RowResampler V;
    thread_local ThreadScratch scratch(U, V);
    const int L = upsample_fft_length(n), S = shift_row_stride(n), nc = L / 2 + 1;
    const int M = resample_oversampling * L, nco = M / 2 + 1, So = 2 * nco;
    U.ensure(L, 4096);
    V.ensure(n, L, U.batch);
    shift_bridge_kernel<<<grid_for(static_cast<size_t>(n_rows) * (L - n)), 256, 0, s>>>(rows, n_rows, n, L, S);
    for (int r0 = 0; r0 < n_rows; r0 += U.batch) {
        const int nr = std::min(U.batch, n_rows - r0);
        float* a = rows + static_cast<size_t>(r0) * S;
        float* w = a;
        if (nr < U.batch) {   // (the spare rows of the buffer: zero, so that every transform stays finite)
            w = U.tail.data();
            VC_TOMOGRAPHY_CUDA_CHECK(cudaMemcpyAsync(w, a, static_cast<size_t>(nr) * S * sizeof(float), cudaMemcpyDeviceToDevice, s));
            VC_TOMOGRAPHY_CUDA_CHECK(cudaMemsetAsync(w + static_cast<size_t>(nr) * S, 0, static_cast<size_t>(U.batch - nr) * S * sizeof(float), s));
        }
        cufft_check(cufftExecR2C(U.fwd.bind(s), w, reinterpret_cast<cufftComplex*>(w)), "R2C (resample)");
        cufftDoubleComplex* B = reinterpret_cast<cufftDoubleComplex*>(V.grid.data());
        resample_spread_kernel<<<grid_for(static_cast<size_t>(U.batch) * nco), 256, 0, s>>>(reinterpret_cast<cufftComplex*>(w), B, U.batch,
                                                                                         nc, nco, L, V.factor.data());
        cufft_check(cufftExecZ2D(V.inv.bind(s), B, V.grid.data()), "Z2D (resample)");
        resample_gather_kernel<<<grid_for(static_cast<size_t>(nr) * ((n + resample_block - 1) / resample_block)), 256, 0, s>>>(
            V.grid.data(), a, nr, n, S, So, M, positions ? positions + static_cast<size_t>(r0) * n : nullptr, row ? row + r0 : nullptr, geo,
            V.kernel.data());
    }
    VC_TOMOGRAPHY_CUDA_CHECK(cudaGetLastError());
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
    VC_TOMOGRAPHY_CUDA_CHECK(cudaGetLastError());
}

void upsample_rows(const float* in, float* out, int n_rows, int n, int os, void* stream) {
    cudaStream_t s = as_stream(stream);
    if (os == 1) {
        VC_TOMOGRAPHY_CUDA_CHECK(cudaMemcpyAsync(out, in, sizeof(float) * n_rows * n, cudaMemcpyDeviceToDevice, s));
        return;
    }
    thread_local Upsampler U;
    thread_local ThreadScratch scratch(U);
    const int L = upsample_fft_length(n);
    FftScratch& scratch_fft = fft_scratch();
    for (int r0 = 0; r0 < n_rows; r0 += kUpsampleBatch) {
        const int nr = std::min(kUpsampleBatch, n_rows - r0);
        Upsampler::Plans& P = U.plans_for(nr, L, os);
        float* const a = grown(scratch_fft.real[0], static_cast<size_t>(nr) * L);
        float* const b = grown(scratch_fft.real[1], static_cast<size_t>(nr) * L * os);
        cufftComplex* const A = grown(scratch_fft.cplx[0], static_cast<size_t>(nr) * (L / 2 + 1));
        cufftComplex* const B = grown(scratch_fft.cplx[1], static_cast<size_t>(nr) * (L * os / 2 + 1));
        pad_rows_kernel<<<grid_for(static_cast<size_t>(nr) * L), 256, 0, s>>>(in + static_cast<size_t>(r0) * n, a, nr, n, L);
        cufft_check(cufftExecR2C(P.fwd.bind(s), a, A), "R2C");
        spread_spectrum_kernel<<<grid_for(static_cast<size_t>(nr) * (L * os / 2 + 1)), 256, 0, s>>>(A, B, nr, L / 2 + 1, L * os / 2 + 1, L);
        cufft_check(cufftExecC2R(P.inv.bind(s), B, b), "C2R");
        crop_scale_kernel<<<grid_for(static_cast<size_t>(nr) * n * os), 256, 0, s>>>(b, out + static_cast<size_t>(r0) * n * os, nr, n * os, L * os,
                                                                                  1.f / static_cast<float>(L));
    }
    VC_TOMOGRAPHY_CUDA_CHECK(cudaGetLastError());
}

}  // namespace vc_tomography::cuda
