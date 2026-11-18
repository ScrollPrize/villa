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
#include "vc_tomography/filters.hpp"
#include "vc_tomography/sino.hpp"
#include "cpu/fftw_plan.hpp"

namespace vc_tomography {

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
    std::vector<float> a(L), b(Lo);
    std::vector<std::complex<float>> A(nc), B(nco);
    const FftwfPlan fwd = make_fftw_plan([&] { return fftwf_plan_dft_r2c_1d(L, a.data(), reinterpret_cast<fftwf_complex*>(A.data()), FFTW_ESTIMATE); });
    const FftwfPlan inv = make_fftw_plan([&] { return fftwf_plan_dft_c2r_1d(Lo, reinterpret_cast<fftwf_complex*>(B.data()), b.data(), FFTW_ESTIMATE); });
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
}

// rows delayed by `shift` samples in place (sino.hpp): each row's transform in its own storage (r2c in place)
void shift_rows(float* rows, int n_rows, int n, float shift, const float* shifts) {
    if (n_rows <= 0 || n <= 1) return;
    const int L = upsample_fft_length(n), S = shift_row_stride(n);
    // (planned on the first row; FFTW_ESTIMATE leaves it untouched, FFTW_UNALIGNED lets every row use the plan)
    const FftwfPlan fwd = make_fftw_plan([&] { return fftwf_plan_dft_r2c_1d(L, rows, reinterpret_cast<fftwf_complex*>(rows), FFTW_ESTIMATE | FFTW_UNALIGNED); });
    const FftwfPlan inv = make_fftw_plan([&] { return fftwf_plan_dft_c2r_1d(L, reinterpret_cast<fftwf_complex*>(rows), rows, FFTW_ESTIMATE | FFTW_UNALIGNED); });
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
}

namespace {

// the oversampled row's transform back in the precision T (sino.hpp resample_real)
inline FftwfPlan inverse_plan(int m, std::complex<float>* spectrum, float* samples) {
    return make_fftw_plan([&] { return fftwf_plan_dft_c2r_1d(m, reinterpret_cast<fftwf_complex*>(spectrum), samples, FFTW_ESTIMATE | FFTW_UNALIGNED); });
}
inline FftwPlan inverse_plan(int m, std::complex<double>* spectrum, double* samples) {
    return make_fftw_plan([&] { return fftw_plan_dft_c2r_1d(m, reinterpret_cast<fftw_complex*>(spectrum), samples, FFTW_ESTIMATE | FFTW_UNALIGNED); });
}
inline void inverse(const FftwfPlan& plan, std::complex<float>* spectrum, float* samples) {
    fftwf_execute_dft_c2r(plan, reinterpret_cast<fftwf_complex*>(spectrum), samples);
}
inline void inverse(const FftwPlan& plan, std::complex<double>* spectrum, double* samples) {
    fftw_execute_dft_c2r(plan, reinterpret_cast<fftw_complex*>(spectrum), samples);
}

// rows read at positions(r, j0, count, p, below, rest): the positions p of samples [j0, j0 + count) of row r (sino.hpp;
// true), or where the kernel starts for them (below, rest: sino_core::resample_values_near; false)
template <class Positions>
void resample_rows_at(float* rows, int n_rows, int n, const Positions& positions) {
    using T = resample_real;
    if (n_rows <= 0 || n <= 1) return;
    const int L = upsample_fft_length(n), S = shift_row_stride(n), nc = L / 2 + 1;
    const int M = resample_oversampling * L, nco = M / 2 + 1;
    const std::vector<double> factor_d = resample_deconvolution(n);
    const std::vector<T> factor(factor_d.begin(), factor_d.end()), kernel_t(resample_kernel().begin(), resample_kernel().end());
    const T* kernel = kernel_t.data();
    const FftwfPlan fwd = make_fftw_plan([&] { return fftwf_plan_dft_r2c_1d(L, rows, reinterpret_cast<fftwf_complex*>(rows), FFTW_ESTIMATE | FFTW_UNALIGNED); });
    std::vector<std::complex<T>> plan_spectrum(nco);
    std::vector<T> plan_samples(M);
    const auto inv = inverse_plan(M, plan_spectrum.data(), plan_samples.data());
#pragma omp parallel
    {
        std::vector<std::complex<T>> spectrum(nco);
        std::vector<T> samples(M);
        constexpr int B = resample_block;
        double p[B];
        int below[B];
        float rest[B];
#pragma omp for schedule(static)
        for (int r = 0; r < n_rows; ++r) {
            float* a = rows + static_cast<std::size_t>(r) * S;
            const float last = a[n - 1], first = a[0];
            for (int i = n; i < L; ++i) a[i] = sino_core::shift_bridge(last, first, i, n, L);
            fftwf_execute_dft_r2c(fwd, a, reinterpret_cast<fftwf_complex*>(a));
            for (int k = 0; k < nc; ++k)   // (the Nyquist bin of a real row is real)
                spectrum[k] = std::complex<T>(static_cast<T>(a[2 * k]) * factor[k], 2 * k == L ? static_cast<T>(0) : static_cast<T>(a[2 * k + 1]) * factor[k]);
            std::fill(spectrum.begin() + nc, spectrum.end(), std::complex<T>(0));
            inverse(inv, spectrum.data(), samples.data());
            for (int j0 = 0; j0 < n; j0 += B) {
                const int count = std::min(B, n - j0);
                if (positions(r, j0, count, p, below, rest)) sino_core::resample_values<B, T>(samples.data(), M, p, kernel, a + j0, count);
                else sino_core::resample_values_near<B, T>(samples.data(), M, below, rest, kernel, a + j0, count);
            }
        }
    }
}

}  // namespace

void resample_rows(float* rows, int n_rows, int n, const double* positions) {
    constexpr int B = resample_block;
    resample_rows_at(rows, n_rows, n, [=](int r, int j0, int count, double* p, int*, float*) {
        std::copy_n(positions + static_cast<std::size_t>(r) * n + j0, count, p);
        std::fill(p + count, p + B, 0.0);
        return true;
    });
}

void resample_rows(float* rows, int n_rows, int n, const ResampleRow* row, const ResampleGeometry& g) {
    constexpr int B = resample_block;
    // the polynomial of each row's displacement along the row, where it has one (map_core::row_fit)
    static const map_core::RowFitTable table = map_core::row_fit_table();
    std::vector<double> fit(static_cast<std::size_t>(std::max(n_rows, 0)) * map_core::kRowFitSize);
    const auto terms = [&](int r) {
        const ResampleRow& w = row[r];
        return map_core::row_terms<double>(g.map, w.y, w.row_slope, g.slope_col, w.col_scale, g.scale_col, w.row_scale, g.scale_row);
    };
#pragma omp parallel for schedule(static)
    for (int r = 0; r < n_rows; ++r)
        map_core::row_fit(g.map, terms(r), table, static_cast<double>(row[r].shift), n, fit.data() + static_cast<std::size_t>(r) * map_core::kRowFitSize);
    resample_rows_at(rows, n_rows, n, [&](int r, int j0, int, double* p, int* below, float* rest) {
        const double* c = fit.data() + static_cast<std::size_t>(r) * map_core::kRowFitSize;
        if (c[0] == c[0]) {
            map_core::row_fit_taps<B>(c, static_cast<double>(row[r].shift), j0, n, resample_oversampling, resample_taps / 2, below, rest);
            return false;
        }
        double X[B], v[B], du[B], dv[B];
        for (int q = 0; q < B; ++q) X[q] = static_cast<double>(j0 + q) - static_cast<double>(row[r].shift);
        map_core::source<B>(g.map, terms(r), X, p, v, du, dv);
        return true;
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
}  // namespace vc_tomography
