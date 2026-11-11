#include "helirec/calibration/registration.hpp"

#include <fftw3.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <mutex>
#include <random>
#include <vector>

#include "helirec/common.hpp"
#include "helirec/fourier.hpp"  // fftw_planner_mutex(): the one lock around FFTW's planner

namespace helirec::calibration {

namespace {

using cfloat = std::complex<float>;

struct FftwFree {
    void operator()(void* p) const { fftwf_free(p); }
};

std::vector<float> taper_window(int n, double fraction) {
    std::vector<float> w(static_cast<std::size_t>(n), 1.0f);
    if (n < 4) return w;
    const int t = std::max(1, static_cast<int>(fraction * n));
    for (int i = 0; i < t && i < n / 2; ++i) {
        const float v = static_cast<float>(0.5 * (1.0 - std::cos(M_PI * (i + 0.5) / t)));
        w[i] = v;
        w[n - 1 - i] = v;
    }
    return w;
}

// signed frequency of bin k of an axis of length n, in cycles per sample
inline double frequency(int k, int n) { return static_cast<double>(k <= n / 2 ? k : k - n) / n; }

// the frame's weighted mean removed, then the taper: dst = (src - mean) * w0 * w1 * w2
void prepare(const float* src, float* dst, int n0, int n1, int n2, const std::vector<float>& w0, const std::vector<float>& w1,
             const std::vector<float>& w2) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n0; ++i) {
        const float* s = src + static_cast<std::size_t>(i) * n1 * n2;
        float* d = dst + static_cast<std::size_t>(i) * n1 * n2;
        double sum = 0, weight = 0;
        for (int j = 0; j < n1; ++j)
            for (int k = 0; k < n2; ++k) {
                const double w = static_cast<double>(w1[j]) * w2[k];
                sum += w * s[static_cast<std::size_t>(j) * n2 + k];
                weight += w;
            }
        const float mean = static_cast<float>(sum / weight);
        for (int j = 0; j < n1; ++j) {
            const float wj = w0[i] * w1[j];
            for (int k = 0; k < n2; ++k) d[static_cast<std::size_t>(j) * n2 + k] = (s[static_cast<std::size_t>(j) * n2 + k] - mean) * wj * w2[k];
        }
    }
}

// solve H x = -g for the active axes (H symmetric); false when H is not negative definite there
bool newton_step(const double H[3][3], const double g[3], const bool active[3], double x[3]) {
    int idx[3], m = 0;
    for (int j = 0; j < 3; ++j)
        if (active[j]) idx[m++] = j;
    x[0] = x[1] = x[2] = 0;
    if (!m) return true;
    // -H must be positive definite: Cholesky of A = -H, then A x = g
    double A[3][3], L[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}, y[3], sol[3];
    for (int r = 0; r < m; ++r)
        for (int c = 0; c < m; ++c) A[r][c] = -H[idx[r]][idx[c]];
    for (int r = 0; r < m; ++r) {
        for (int c = 0; c <= r; ++c) {
            double s = A[r][c];
            for (int k = 0; k < c; ++k) s -= L[r][k] * L[c][k];
            if (r == c) {
                if (s <= 0) return false;
                L[r][r] = std::sqrt(s);
            } else {
                L[r][c] = s / L[c][c];
            }
        }
    }
    for (int r = 0; r < m; ++r) {
        double s = g[idx[r]];
        for (int k = 0; k < r; ++k) s -= L[r][k] * y[k];
        y[r] = s / L[r][r];
    }
    for (int r = m - 1; r >= 0; --r) {
        double s = y[r];
        for (int k = r + 1; k < m; ++k) s -= L[k][r] * sol[k];
        sol[r] = s / L[r][r];
    }
    for (int r = 0; r < m; ++r) x[idx[r]] = sol[r];
    return true;
}

}  // namespace

Registration register_blocks(const float* a, const float* b, int n0, int n1, int n2, const RegistrationOptions& opt) {
    if (n0 < 1 || n1 < 1 || n2 < 2) throw Error("register_blocks: empty block");
    const int n[3] = {n0, n1, n2};
    const int nc = n2 / 2 + 1;
    const std::size_t N = static_cast<std::size_t>(n0) * n1 * n2, NC = static_cast<std::size_t>(n0) * n1 * nc;
    const bool active[3] = {n0 > 1, n1 > 1, n2 > 1};

    std::unique_ptr<float, FftwFree> ra(fftwf_alloc_real(N)), rb(fftwf_alloc_real(N));
    std::unique_ptr<fftwf_complex, FftwFree> ca(fftwf_alloc_complex(NC)), cb(fftwf_alloc_complex(NC));
    if (!ra || !rb || !ca || !cb) throw Error("register_blocks: out of memory");

    const std::vector<float> w0 = active[0] ? taper_window(n0, opt.taper) : std::vector<float>(1, 1.0f);
    const std::vector<float> w1 = taper_window(n1, opt.taper), w2 = taper_window(n2, opt.taper);
    prepare(a, ra.get(), n0, n1, n2, w0, w1, w2);
    prepare(b, rb.get(), n0, n1, n2, w0, w1, w2);

    fftwf_plan forward, inverse;
    {
        std::lock_guard<std::mutex> lock(fftw_planner_mutex());
        forward = fftwf_plan_dft_r2c_3d(n0, n1, n2, ra.get(), ca.get(), FFTW_ESTIMATE);
        inverse = fftwf_plan_dft_c2r_3d(n0, n1, n2, cb.get(), rb.get(), FFTW_ESTIMATE);
    }
    if (!forward || !inverse) throw Error("register_blocks: FFTW could not plan the transform");
#pragma omp parallel sections num_threads(2)
    {
#pragma omp section
        fftwf_execute_dft_r2c(forward, ra.get(), ca.get());
#pragma omp section
        fftwf_execute_dft_r2c(forward, rb.get(), cb.get());
    }

    // C = conj(A) B W, kept in ca; the energies of both blocks in the same band
    cfloat* A = reinterpret_cast<cfloat*>(ca.get());
    cfloat* B = reinterpret_cast<cfloat*>(cb.get());
    const double lo2 = 2 * opt.f_lo * opt.f_lo, hi2 = 2 * opt.f_hi * opt.f_hi;
    double ea = 0, eb = 0;
#pragma omp parallel for collapse(2) schedule(static) reduction(+ : ea, eb)
    for (int k0 = 0; k0 < n0; ++k0) {
        for (int k1 = 0; k1 < n1; ++k1) {
            const double f0 = frequency(k0, n0), f1 = frequency(k1, n1);
            double wq = 1.0;
            if (active[0]) {
                const double c = std::cos(M_PI * f0);
                wq = c * c;
                if (n0 % 2 == 0 && k0 == n0 / 2) wq = 0;
            }
            if (active[1] && n1 % 2 == 0 && k1 == n1 / 2) wq = 0;
            cfloat* pa = A + (static_cast<std::size_t>(k0) * n1 + k1) * nc;
            cfloat* pb = B + (static_cast<std::size_t>(k0) * n1 + k1) * nc;
            for (int k2 = 0; k2 < nc; ++k2) {
                const double f2 = frequency(k2, n2);
                const double rho2 = f1 * f1 + f2 * f2;
                double w = wq * (1.0 - std::exp(-rho2 / lo2)) * std::exp(-rho2 / hi2);
                if (n2 % 2 == 0 && k2 == n2 / 2) w = 0;
                // content that is constant along the rows or along the columns (stripes fixed to the detector or the beam)
                if (active[1] && (k2 <= opt.stripe_notch || std::min(k1, n1 - k1) <= opt.stripe_notch)) w = 0;
                const double mult = (k2 == 0) ? 1.0 : 2.0;
                ea += mult * w * std::norm(pa[k2]);
                eb += mult * w * std::norm(pb[k2]);
                const cfloat c = std::conj(pa[k2]) * pb[k2] * static_cast<float>(w);
                pa[k2] = c;
                pb[k2] = c;  // the inverse transform destroys its input
            }
        }
    }

    Registration out;
    if (!(ea > 0) || !(eb > 0)) {
        std::lock_guard<std::mutex> lock(fftw_planner_mutex());
        fftwf_destroy_plan(forward);
        fftwf_destroy_plan(inverse);
        return out;  // a block without content in the band
    }

    // integer maximum of the correlation within the search window
    fftwf_execute_dft_c2r(inverse, cb.get(), rb.get());
    {
        std::lock_guard<std::mutex> lock(fftw_planner_mutex());
        fftwf_destroy_plan(forward);
        fftwf_destroy_plan(inverse);
    }
    int s[3], best[3] = {0, 0, 0};
    for (int j = 0; j < 3; ++j) s[j] = active[j] ? std::min(opt.search[j], (n[j] - 1) / 2) : 0;
    float peak = -INFINITY;
    for (int d0 = -s[0]; d0 <= s[0]; ++d0)
        for (int d1 = -s[1]; d1 <= s[1]; ++d1) {
            const float* row = rb.get() + (static_cast<std::size_t>((d0 + n0) % n0) * n1 + (d1 + n1) % n1) * n2;
            for (int d2 = -s[2]; d2 <= s[2]; ++d2) {
                const float v = row[(d2 + n2) % n2];
                if (v > peak) peak = v, best[0] = d0, best[1] = d1, best[2] = d2;
            }
        }
    for (int j = 0; j < 3; ++j)
        if (active[j] && s[j] > 0 && std::abs(best[j]) == s[j]) out.at_edge = true;

    // continuous maximum: r(d) = sum_k Re[C_k exp(2 pi i f_k . d)]
    double d[3] = {static_cast<double>(best[0]), static_cast<double>(best[1]), static_cast<double>(best[2])};
    double r = 0;
    const auto evaluate = [&](const double at[3], double& value, double g[3], double H[3][3]) {
        std::vector<std::complex<double>> e0(n0), e1(n1), e2(nc);
        for (int k = 0; k < n0; ++k) e0[k] = std::polar(1.0, 2 * M_PI * frequency(k, n0) * at[0]);
        for (int k = 0; k < n1; ++k) e1[k] = std::polar(1.0, 2 * M_PI * frequency(k, n1) * at[1]);
        for (int k = 0; k < nc; ++k) e2[k] = std::polar(1.0, 2 * M_PI * frequency(k, n2) * at[2]);
        double v = 0, g0 = 0, g1 = 0, g2 = 0, h00 = 0, h01 = 0, h02 = 0, h11 = 0, h12 = 0, h22 = 0;
#pragma omp parallel for collapse(2) schedule(static) reduction(+ : v, g0, g1, g2, h00, h01, h02, h11, h12, h22)
        for (int k0 = 0; k0 < n0; ++k0) {
            for (int k1 = 0; k1 < n1; ++k1) {
                const double q0 = 2 * M_PI * frequency(k0, n0), q1 = 2 * M_PI * frequency(k1, n1);
                const std::complex<double> e01 = e0[k0] * e1[k1];
                const cfloat* pc = A + (static_cast<std::size_t>(k0) * n1 + k1) * nc;
                double sr = 0, si = 0, sr2 = 0, si2 = 0, sr22 = 0, si22 = 0;  // sums over k2 of t, q2 t, q2^2 t (Re, Im)
                for (int k2 = 0; k2 < nc; ++k2) {
                    const double mult = (k2 == 0) ? 1.0 : 2.0;
                    const double q2 = 2 * M_PI * frequency(k2, n2);
                    const std::complex<double> t = std::complex<double>(pc[k2].real(), pc[k2].imag()) * e2[k2];
                    sr += mult * t.real();
                    si += mult * t.imag();
                    sr2 += mult * q2 * t.real();
                    si2 += mult * q2 * t.imag();
                    sr22 += mult * q2 * q2 * t.real();
                    si22 += mult * q2 * q2 * t.imag();
                }
                // multiply the k2 sums by e01: Re[(sr + i si) e01] etc.
                const double re = sr * e01.real() - si * e01.imag(), im = sr * e01.imag() + si * e01.real();
                const double re2 = sr2 * e01.real() - si2 * e01.imag(), im2 = sr2 * e01.imag() + si2 * e01.real();
                const double re22 = sr22 * e01.real() - si22 * e01.imag();
                v += re;
                g0 -= q0 * im;
                g1 -= q1 * im;
                g2 -= im2;
                h00 -= q0 * q0 * re;
                h01 -= q0 * q1 * re;
                h11 -= q1 * q1 * re;
                h02 -= q0 * re2;
                h12 -= q1 * re2;
                h22 -= re22;
            }
        }
        value = v;
        g[0] = g0, g[1] = g1, g[2] = g2;
        H[0][0] = h00, H[0][1] = H[1][0] = h01, H[0][2] = H[2][0] = h02, H[1][1] = h11, H[1][2] = H[2][1] = h12, H[2][2] = h22;
    };
    double g[3], H[3][3];
    evaluate(d, r, g, H);
    for (int it = 0; it < 40; ++it) {
        double step[3];
        if (!newton_step(H, g, active, step)) break;  // not at a maximum: keep the integer estimate
        double largest = 0;
        for (int j = 0; j < 3; ++j) {
            step[j] = std::clamp(step[j], -0.5, 0.5);
            largest = std::max(largest, std::fabs(step[j]));
        }
        // take the step, halving it until the correlation does not decrease
        double trial[3], rt = 0, gt[3], Ht[3][3], scale = 1.0;
        bool accepted = false;
        for (int half = 0; half < 8; ++half, scale *= 0.5) {
            for (int j = 0; j < 3; ++j) trial[j] = d[j] + scale * step[j];
            evaluate(trial, rt, gt, Ht);
            if (rt >= r) {
                accepted = true;
                break;
            }
        }
        if (!accepted) break;
        for (int j = 0; j < 3; ++j) d[j] = trial[j], g[j] = gt[j];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) H[i][j] = Ht[i][j];
        r = rt;
        if (scale * largest < 1e-6) {
            out.converged = true;
            break;
        }
    }
    for (int j = 0; j < 3; ++j) out.shift[j] = active[j] ? d[j] : 0.0;
    out.correlation = r / std::sqrt(ea * eb);
    return out;
}

// ------------------------------------------------------------------------------------------------ self test

namespace {

// A field of Gaussian blobs, defined at every real position: samples of it at displaced grids are exact.
struct BlobField {
    std::vector<double> c0, c1, c2, amp;
    double s0, s1, s2;
    BlobField(std::mt19937& rng, int n, double lo0, double hi0, double lo1, double hi1, double lo2, double hi2, double s0_, double s1_, double s2_)
        : s0(s0_), s1(s1_), s2(s2_) {
        std::uniform_real_distribution<double> u(0, 1);
        std::normal_distribution<double> g(0, 1);
        for (int i = 0; i < n; ++i) {
            c0.push_back(lo0 + (hi0 - lo0) * u(rng));
            c1.push_back(lo1 + (hi1 - lo1) * u(rng));
            c2.push_back(lo2 + (hi2 - lo2) * u(rng));
            amp.push_back(g(rng));
        }
    }
    // out(i, j, k) = f(o0 + i, o1 + j, o2 + step2 * k)
    void render(float* out, int n0, int n1, int n2, double o0, double o1, double o2, double step2 = 1.0) const {
        std::fill(out, out + static_cast<std::size_t>(n0) * n1 * n2, 0.0f);
        std::vector<double> p0(n0), p1(n1), p2(n2);
        for (std::size_t m = 0; m < amp.size(); ++m) {
            const auto profile = [](std::vector<double>& p, int n, double origin, double step, double centre, double sigma, int& a, int& b) {
                a = n, b = -1;
                for (int i = 0; i < n; ++i) {
                    const double x = (origin + step * i - centre) / sigma;
                    p[i] = std::fabs(x) < 4.5 ? std::exp(-0.5 * x * x) : 0.0;
                    if (p[i] != 0.0) a = std::min(a, i), b = std::max(b, i);
                }
            };
            int a0, b0, a1, b1, a2, b2;
            if (n0 == 1) p0[0] = 1.0, a0 = b0 = 0;
            else profile(p0, n0, o0, 1.0, c0[m], s0, a0, b0);
            profile(p1, n1, o1, 1.0, c1[m], s1, a1, b1);
            profile(p2, n2, o2, step2, c2[m], s2, a2, b2);
            for (int i = a0; i <= b0; ++i)
                for (int j = a1; j <= b1; ++j) {
                    const double w = amp[m] * p0[i] * p1[j];
                    float* row = out + (static_cast<std::size_t>(i) * n1 + j) * n2;
                    for (int k = a2; k <= b2; ++k) row[k] += static_cast<float>(w * p2[k]);
                }
        }
    }
};

void add_noise(std::vector<float>& v, std::mt19937& rng, double sigma) {
    std::normal_distribution<float> g(0, static_cast<float>(sigma));
    for (float& x : v) x += g(rng);
}

}  // namespace

bool registration_selftest(bool verbose) {
    bool all = true;
    const double tolerance = 0.02;
    const auto report = [&](const char* name, const double got[3], const double want[3], double corr, bool flags_ok) {
        double worst = 0;
        for (int j = 0; j < 3; ++j) worst = std::max(worst, std::fabs(got[j] - want[j]));
        const bool ok = worst < tolerance && flags_ok;
        all &= ok;
        if (verbose)
            std::printf("%-44s %s  want (%+.3f %+.3f %+.3f) got (%+.4f %+.4f %+.4f) worst error %.4f, correlation %.3f\n", name, ok ? "ok  " : "FAIL",
                        want[0], want[1], want[2], got[0], got[1], got[2], worst, corr);
    };
    std::mt19937 rng(20251111);

    // 1, 2: three-dimensional displacements; b(x + d) = a(x)
    {
        const int n0 = 16, n1 = 96, n2 = 240;
        BlobField f(rng, 6000, -30, n0 + 30, -40, n1 + 40, -40, n2 + 40, 1.5, 2.5, 2.5);
        std::vector<float> a(static_cast<std::size_t>(n0) * n1 * n2), b(a.size());
        f.render(a.data(), n0, n1, n2, 0, 0, 0);
        add_noise(a, rng, 0.05);
        const double cases[3][3] = {{0.27, -0.41, 0.33}, {-1.30, 7.62, -11.20}, {0.0, 0.50, -0.25}};
        const char* names[3] = {"3D fractional displacement", "3D displacement of several samples", "3D half-sample displacement"};
        for (int c = 0; c < 3; ++c) {
            const double* d = cases[c];
            f.render(b.data(), n0, n1, n2, -d[0], -d[1], -d[2]);
            add_noise(b, rng, 0.05);
            const Registration r = register_blocks(a.data(), b.data(), n0, n1, n2);
            report(names[c], r.shift, d, r.correlation, r.converged && !r.at_edge);
        }
        // 3: the same with a level per frame and a smooth profile added to b (beam drift, flat mismatch)
        const double d[3] = {0.27, -0.41, 0.33};
        f.render(b.data(), n0, n1, n2, -d[0], -d[1], -d[2]);
        for (int i = 0; i < n0; ++i)
            for (int j = 0; j < n1; ++j)
                for (int k = 0; k < n2; ++k)
                    b[(static_cast<std::size_t>(i) * n1 + j) * n2 + k] += static_cast<float>(0.5 * std::sin(1.7 * i) + 2.0 * k / n2 + 1.0 * (j - n1 / 2.0) * (j - n1 / 2.0) / (n1 * n1));
        add_noise(b, rng, 0.05);
        const Registration r = register_blocks(a.data(), b.data(), n0, n1, n2);
        report("3D with frame levels and a smooth profile", r.shift, d, r.correlation, r.converged && !r.at_edge);
    }
    // 4: two dimensions (one frame)
    {
        const int n1 = 128, n2 = 256;
        BlobField f(rng, 3000, 0, 1, -40, n1 + 40, -40, n2 + 40, 1.0, 2.0, 2.0);
        std::vector<float> a(static_cast<std::size_t>(n1) * n2), b(a.size());
        const double d[3] = {0.0, 0.37, -0.62};
        f.render(a.data(), 1, n1, n2, 0, 0, 0);
        f.render(b.data(), 1, n1, n2, 0, -d[1], -d[2]);
        add_noise(a, rng, 0.05);
        add_noise(b, rng, 0.05);
        const Registration r = register_blocks(a.data(), b.data(), 1, n1, n2);
        report("2D fractional displacement", r.shift, d, r.correlation, r.converged && !r.at_edge);
    }
    // 5: a mirror axis. View A is f(column); the opposite view is f(2 c - column), a later frame and other rows.
    {
        const int n0 = 8, n1 = 64, n = 240, a0 = 40, b0 = 300;
        const double c = 290.37, dq = 0.133, dv = -0.2;
        BlobField f(rng, 5000, -30, n0 + 30, -40, n1 + 40, -40, 700, 1.5, 2.5, 2.5);
        std::vector<float> a(static_cast<std::size_t>(n0) * n1 * n), b(a.size()), flipped(a.size());
        f.render(a.data(), n0, n1, n, 0, 0, a0);
        // opposite view sampled on its own detector columns b0 .. b0 + n - 1, frames and rows displaced by (dq, dv)
        f.render(b.data(), n0, n1, n, -dq, -dv, 2 * c - b0, -1.0);
        for (std::size_t row = 0; row < static_cast<std::size_t>(n0) * n1; ++row)
            for (int k = 0; k < n; ++k) flipped[row * n + k] = b[row * n + (n - 1 - k)];
        add_noise(a, rng, 0.05);
        add_noise(flipped, rng, 0.05);
        const Registration r = register_blocks(a.data(), flipped.data(), n0, n1, n);
        const double got[3] = {r.shift[0], r.shift[1], mirror_axis(a0, b0, n, r.shift[2])};
        const double want[3] = {dq, dv, c};
        report("mirror axis from reversed columns", got, want, r.correlation, r.converged && !r.at_edge);
    }
    if (verbose) std::printf("registration self test: %s (tolerance %.2f samples)\n", all ? "passed" : "FAILED", tolerance);
    return all;
}

}  // namespace helirec::calibration
