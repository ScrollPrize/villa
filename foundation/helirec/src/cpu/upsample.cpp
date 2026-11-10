#include <fftw3.h>

#include <algorithm>
#include <complex>
#include <cstring>
#include <mutex>
#include <vector>

#include "common/interp_core.hpp"
#include "common/sino_core.hpp"
#include "helirec/filters.hpp"
#include "helirec/sino.hpp"

namespace helirec {

int upsample_fft_length(int n) { return next_smooth_power(n + 64); }
int shift_row_stride(int n) { return 2 * (upsample_fft_length(n) / 2 + 1); }

namespace cpu {

// band-limited (FFT zero-padding) upsampling of rows: out[r, i] = row r at position i / os, i < n * os
void upsample_rows(const float* in, float* out, int n_rows, int n, int os) {
    if (os == 1) { std::memcpy(out, in, sizeof(float) * n_rows * n); return; }
    const int L = upsample_fft_length(n), Lo = L * os, nc = L / 2 + 1, nco = Lo / 2 + 1;
    std::mutex& m = fftw_planner_mutex();
    fftwf_plan fwd, inv;
    std::vector<float> a(L), b(Lo);
    std::vector<std::complex<float>> A(nc), B(nco);
    {
        std::lock_guard<std::mutex> lock(m);
        fwd = fftwf_plan_dft_r2c_1d(L, a.data(), reinterpret_cast<fftwf_complex*>(A.data()), FFTW_ESTIMATE);
        inv = fftwf_plan_dft_c2r_1d(Lo, reinterpret_cast<fftwf_complex*>(B.data()), b.data(), FFTW_ESTIMATE);
    }
    for (int r = 0; r < n_rows; ++r) {
        std::fill(a.begin(), a.end(), 0.f);
        std::copy(in + static_cast<std::size_t>(r) * n, in + static_cast<std::size_t>(r + 1) * n, a.begin());
        fftwf_execute(fwd);
        std::fill(B.begin(), B.end(), std::complex<float>(0));
        for (int k = 0; k < nc; ++k) B[k] = A[k];
        if (L % 2 == 0) B[L / 2] *= 0.5f;  // split the Nyquist bin
        fftwf_execute(inv);
        const float s = 1.f / static_cast<float>(L);
        for (int i = 0; i < n * os; ++i) out[static_cast<std::size_t>(r) * n * os + i] = b[i] * s;
    }
    std::lock_guard<std::mutex> lock(m);
    fftwf_destroy_plan(fwd);
    fftwf_destroy_plan(inv);
}

// rows delayed by `shift` samples in place (sino.hpp): each row's transform in its own storage (r2c in place)
void shift_rows(float* rows, int n_rows, int n, float shift) {
    if (n_rows <= 0 || n <= 1) return;
    const int L = upsample_fft_length(n), S = shift_row_stride(n);
    std::mutex& m = fftw_planner_mutex();
    fftwf_plan fwd, inv;
    {
        std::lock_guard<std::mutex> lock(m);
        // (planned on the first row; FFTW_ESTIMATE leaves it untouched, FFTW_UNALIGNED lets every row use the plan)
        fwd = fftwf_plan_dft_r2c_1d(L, rows, reinterpret_cast<fftwf_complex*>(rows), FFTW_ESTIMATE | FFTW_UNALIGNED);
        inv = fftwf_plan_dft_c2r_1d(L, reinterpret_cast<fftwf_complex*>(rows), rows, FFTW_ESTIMATE | FFTW_UNALIGNED);
    }
#pragma omp parallel for schedule(static)
    for (int r = 0; r < n_rows; ++r) {
        float* a = rows + static_cast<std::size_t>(r) * S;
        const float last = a[n - 1], first = a[0];
        for (int i = n; i < L; ++i) a[i] = sino_core::shift_bridge(last, first, i, n, L);
        fftwf_execute_dft_r2c(fwd, a, reinterpret_cast<fftwf_complex*>(a));
        for (int k = 0; k <= L / 2; ++k) sino_core::shift_bin(a[2 * k], a[2 * k + 1], k, L, shift);
        fftwf_execute_dft_c2r(inv, reinterpret_cast<fftwf_complex*>(a), a);
    }
    std::lock_guard<std::mutex> lock(m);
    fftwf_destroy_plan(fwd);
    fftwf_destroy_plan(inv);
}

void bspline_prefilter(const float* in, float* out, int n_rows, int n, int degree) {
#pragma omp parallel for schedule(static)
    for (int r = 0; r < n_rows; ++r) {
        float* o = out + static_cast<std::size_t>(r) * n;
        if (o != in + static_cast<std::size_t>(r) * n) std::memcpy(o, in + static_cast<std::size_t>(r) * n, n * sizeof(float));
        interp_core::bspline_prefilter_row(o, n, degree);
    }
}

}  // namespace cpu
}  // namespace helirec
