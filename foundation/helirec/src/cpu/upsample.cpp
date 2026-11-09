#include <fftw3.h>

#include <algorithm>
#include <complex>
#include <cstring>
#include <mutex>
#include <vector>

#include "common/interp_core.hpp"
#include "helirec/filters.hpp"
#include "helirec/sino.hpp"

namespace helirec {

int upsample_fft_length(int n) { return next_smooth_power(n + 64); }

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
