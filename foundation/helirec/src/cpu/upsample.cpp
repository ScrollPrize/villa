#include <fftw3.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <map>
#include <mutex>
#include <vector>

#include "common/detector_map_core.hpp"
#include "common/interp_core.hpp"
#include "common/sino_core.hpp"
#include "helirec/filters.hpp"
#include "helirec/sino.hpp"

namespace helirec {

int upsample_fft_length(int n) { return next_smooth_power(n + 64); }
int shift_row_stride(int n) { return 2 * (upsample_fft_length(n) / 2 + 1); }

namespace {

// modified Bessel function of the first kind, order 0 (its series; x up to resample_beta)
double bessel_i0(double x) {
    double sum = 1, term = 1;
    for (int k = 1; k < 200 && term > 1e-18 * sum; ++k) {
        term *= (0.5 * x / k) * (0.5 * x / k);
        sum += term;
    }
    return sum;
}

// Gauss-Legendre nodes and weights on [-1, 1]
void gauss_legendre(int Q, std::vector<double>& node, std::vector<double>& weight) {
    node.resize(Q);
    weight.resize(Q);
    for (int q = 0; q < Q; ++q) {   // Newton's method on the Legendre polynomial of degree Q
        double x = std::cos(M_PI * (q + 0.75) / (Q + 0.5)), dp = 1;
        for (int iter = 0; iter < 100; ++iter) {
            double p0 = 1, p1 = x;
            for (int d = 2; d <= Q; ++d) {
                const double p2 = ((2 * d - 1) * x * p1 - (d - 1) * p0) / d;
                p0 = p1;
                p1 = p2;
            }
            dp = Q * (x * p1 - p0) / (x * x - 1);
            const double step = p1 / dp;
            x -= step;
            if (std::fabs(step) < 1e-16) break;
        }
        node[q] = x;
        weight[q] = 2 / ((1 - x * x) * dp * dp);
    }
}

// the weights of the taps at fraction f (sino.hpp), as sino_core::resample_values computes them
void kernel_weights(const std::vector<double>& c, double f, double* w) {
    const int H = resample_taps / 2, D = resample_degree, qe = D - D % 2, qo = D - 1 + D % 2;
    const double t = 2 * f - 1, s = t * t;
    for (int i = 0; i < H; ++i) {
        double even = c[static_cast<std::size_t>(qe) * H + i], odd = c[static_cast<std::size_t>(qo) * H + i];
        for (int q = qe - 2; q >= 0; q -= 2) even = even * s + c[static_cast<std::size_t>(q) * H + i];
        for (int q = qo - 2; q >= 1; q -= 2) odd = odd * s + c[static_cast<std::size_t>(q) * H + i];
        odd *= t;
        w[i] = even + odd;
        w[resample_taps - 1 - i] = even - odd;
    }
}

}  // namespace

const std::vector<double>& resample_kernel() {
    static const std::vector<double> kernel = [] {
        const int D = resample_degree, W = resample_taps, H = W / 2;
        std::vector<double> c(static_cast<std::size_t>(D + 1) * H, 0.0);
        // T_j(t) in powers of t (T_{j+1} = 2 t T_j - T_{j-1})
        std::vector<std::vector<double>> T(D + 1, std::vector<double>(D + 1, 0.0));
        T[0][0] = 1;
        if (D > 0) T[1][1] = 1;
        for (int j = 2; j <= D; ++j)
            for (int q = 0; q <= j; ++q) T[j][q] = (q > 0 ? 2 * T[j - 1][q - 1] : 0.0) - T[j - 2][q];
        const double i0_beta = bessel_i0(resample_beta);
        for (int i = 0; i < H; ++i) {
            // the window on tap i's interval, d = f - W / 2 + i, at the Chebyshev nodes of t = 2 f - 1 in [-1, 1]
            std::vector<double> value(D + 1);
            for (int k = 0; k <= D; ++k) {
                const double t = std::cos(M_PI * (k + 0.5) / (D + 1));
                const double z = (0.5 * (t + 1) - 0.5 * W + i) / (0.5 * W);
                value[k] = bessel_i0(resample_beta * std::sqrt(std::max(0.0, 1 - z * z))) / i0_beta;
            }
            for (int j = 0; j <= D; ++j) {   // its Chebyshev coefficients, then powers of t
                double a = 0;
                for (int k = 0; k <= D; ++k) a += value[k] * std::cos(M_PI * j * (k + 0.5) / (D + 1));
                a *= (j == 0 ? 1.0 : 2.0) / (D + 1);
                for (int q = 0; q <= j; ++q) c[static_cast<std::size_t>(q) * H + i] += a * T[j][q];
            }
        }
        return c;
    }();
    return kernel;
}

std::vector<double> resample_deconvolution(int n) {
    static std::mutex mutex;
    static std::map<int, std::vector<double>> made;   // by transform length
    const int L = upsample_fft_length(n), M = resample_oversampling * L, nc = L / 2 + 1, W = resample_taps;
    std::lock_guard<std::mutex> lock(mutex);
    auto it = made.find(L);
    if (it != made.end()) return it->second;
    // The kernel's transform at k cycles per M samples: the integral of its polynomials times cos(2 pi k d / M)
    // over each tap's interval (Gauss-Legendre nodes; the kernel is even).
    const int Q = 32;
    std::vector<double> node, weight;
    gauss_legendre(Q, node, weight);
    const std::vector<double>& c = resample_kernel();
    std::vector<double> at(static_cast<std::size_t>(Q) * W);
    for (int q = 0; q < Q; ++q) kernel_weights(c, 0.5 * (node[q] + 1), &at[static_cast<std::size_t>(q) * W]);
    std::vector<double> out(nc);
    for (int k = 0; k < nc; ++k) {
        double transform = 0;
        for (int q = 0; q < Q; ++q)
            for (int i = 0; i < W; ++i)
                transform += 0.5 * weight[q] * at[static_cast<std::size_t>(q) * W + i] *
                             std::cos(2 * M_PI * k * (0.5 * (node[q] + 1) - 0.5 * W + i) / M);
        out[k] = (2 * k == L ? 0.5 : 1.0) / (L * transform);
    }
    made[L] = out;
    return out;
}

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
void shift_rows(float* rows, int n_rows, int n, float shift, const float* shifts) {
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
        const float sh = shifts ? shift + shifts[r] : shift;
        for (int k = 0; k <= L / 2; ++k) sino_core::shift_bin(a[2 * k], a[2 * k + 1], k, L, sh);
        fftwf_execute_dft_c2r(inv, reinterpret_cast<fftwf_complex*>(a), a);
    }
    std::lock_guard<std::mutex> lock(m);
    fftwf_destroy_plan(fwd);
    fftwf_destroy_plan(inv);
}

namespace {

// rows read at positions(r, j0, count, p): the positions of samples [j0, j0 + count) of row r (sino.hpp)
template <class Positions>
void resample_rows_at(float* rows, int n_rows, int n, const Positions& positions) {
    if (n_rows <= 0 || n <= 1) return;
    const int L = upsample_fft_length(n), S = shift_row_stride(n), nc = L / 2 + 1;
    const int M = resample_oversampling * L, nco = M / 2 + 1;
    const std::vector<double> factor = resample_deconvolution(n);
    const double* kernel = resample_kernel().data();
    std::mutex& m = fftw_planner_mutex();
    fftwf_plan fwd;
    fftw_plan inv;
    {
        std::vector<std::complex<double>> spectrum(nco);
        std::vector<double> samples(M);
        std::lock_guard<std::mutex> lock(m);
        fwd = fftwf_plan_dft_r2c_1d(L, rows, reinterpret_cast<fftwf_complex*>(rows), FFTW_ESTIMATE | FFTW_UNALIGNED);
        inv = fftw_plan_dft_c2r_1d(M, reinterpret_cast<fftw_complex*>(spectrum.data()), samples.data(), FFTW_ESTIMATE | FFTW_UNALIGNED);
    }
#pragma omp parallel
    {
        std::vector<std::complex<double>> spectrum(nco);
        std::vector<double> samples(M);
        constexpr int B = resample_block;
        double p[B];
#pragma omp for schedule(static)
        for (int r = 0; r < n_rows; ++r) {
            float* a = rows + static_cast<std::size_t>(r) * S;
            const float last = a[n - 1], first = a[0];
            for (int i = n; i < L; ++i) a[i] = sino_core::shift_bridge(last, first, i, n, L);
            fftwf_execute_dft_r2c(fwd, a, reinterpret_cast<fftwf_complex*>(a));
            for (int k = 0; k < nc; ++k)   // (the Nyquist bin of a real row is real)
                spectrum[k] = std::complex<double>(a[2 * k] * factor[k], 2 * k == L ? 0.0 : a[2 * k + 1] * factor[k]);
            std::fill(spectrum.begin() + nc, spectrum.end(), std::complex<double>(0));
            fftw_execute_dft_c2r(inv, reinterpret_cast<fftw_complex*>(spectrum.data()), samples.data());
            for (int j0 = 0; j0 < n; j0 += B) {
                const int count = std::min(B, n - j0);
                positions(r, j0, count, p);
                sino_core::resample_values<B>(samples.data(), M, p, kernel, a + j0, count);
            }
        }
    }
    std::lock_guard<std::mutex> lock(m);
    fftwf_destroy_plan(fwd);
    fftw_destroy_plan(inv);
}

}  // namespace

void resample_rows(float* rows, int n_rows, int n, const double* positions) {
    constexpr int B = resample_block;
    resample_rows_at(rows, n_rows, n, [=](int r, int j0, int count, double* p) {
        std::copy_n(positions + static_cast<std::size_t>(r) * n + j0, count, p);
        std::fill(p + count, p + B, 0.0);
    });
}

void resample_rows(float* rows, int n_rows, int n, const ResampleRow* row, const ResampleGeometry& g) {
    constexpr int B = resample_block;
    resample_rows_at(rows, n_rows, n, [=, &g](int r, int j0, int, double* p) {
        const ResampleRow& w = row[r];
        double X[B], v[B];
        for (int q = 0; q < B; ++q) X[q] = static_cast<double>(j0 + q) - static_cast<double>(w.shift);
        map_core::source<B>(g.map, w.y, w.row_slope, g.slope_col, w.col_scale, g.scale_col, w.row_scale, g.scale_row, X, p, v);
    });
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
