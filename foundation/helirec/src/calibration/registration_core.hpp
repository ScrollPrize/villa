#pragma once
// What the CPU and the CUDA registration share: the geometry of a registration, its windows and band weights,
// and the interface of the array work (transforms, windows, sums over the spectrum) that each device implements.
// The estimator itself (the maximum search and the repeated displacement) is written once, in registration.cpp,
// on top of that interface. Plain C++17: this header is also compiled by nvcc.
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <memory>
#include <vector>

#include "helirec/calibration/registration.hpp"
#include "helirec/common.hpp"

namespace helirec::calibration {

// signed frequency of bin k of an axis of length n, in cycles per sample
inline double frequency(int k, int n) { return static_cast<double>(k <= n / 2 ? k : k - n) / n; }

// 1 on [lo, hi], raised-cosine ramps to 0 just outside the ends of the axis
inline std::vector<float> flat_taper(int L, int lo, int hi) {
    std::vector<float> w(static_cast<std::size_t>(L), 1.0f);
    lo = std::clamp(lo, 0, L - 1);
    hi = std::clamp(hi, lo, L - 1);
    for (int i = 0; i < lo; ++i) w[i] = static_cast<float>(0.5 * (1.0 - std::cos(M_PI * (i + 1) / (lo + 1))));
    for (int i = hi + 1; i < L; ++i) w[i] = static_cast<float>(0.5 * (1.0 - std::cos(M_PI * (L - i) / (L - hi))));
    return w;
}

// the analysis window of the inner block: a fraction of each end tapered
inline std::vector<float> taper_window(int n, double fraction) {
    if (n < 4) return std::vector<float>(static_cast<std::size_t>(n), 1.0f);
    const int t = std::clamp(static_cast<int>(fraction * n), 1, n / 2);
    return flat_taper(n, t, n - 1 - t);
}

// exp(2 pi i f d) per bin; the Nyquist bin of an even axis gets cos(pi d), which keeps a displaced real sequence real
inline std::vector<std::complex<double>> phasors(int L, int bins, double d) {
    std::vector<std::complex<double>> e(static_cast<std::size_t>(bins));
    for (int k = 0; k < bins; ++k) {
        if (L % 2 == 0 && k == L / 2) e[k] = std::cos(M_PI * d);
        else e[k] = std::polar(1.0, 2 * M_PI * frequency(k, L) * d);
    }
    return e;
}

// Sizes of one registration: block a is n, block b is L = n + 2 m; spectra keep nc and Lc bins along axis 2.
struct RegistrationGeometry {
    int n[3] = {0, 0, 0}, m[3] = {0, 0, 0}, L[3] = {0, 0, 0};
    int nc = 0, Lc = 0;
    bool active[3] = {false, false, false};
    std::size_t N = 0, NC = 0, NB = 0, NBC = 0;
    std::vector<float> w0, w1, w2;  // analysis window of the inner block

    RegistrationGeometry(const int n_[3], const int m_[3], const RegistrationOptions& opt) {
        for (int j = 0; j < 3; ++j) {
            n[j] = n_[j];
            m[j] = m_[j];
            L[j] = n_[j] + 2 * m_[j];
            active[j] = n_[j] > 1;
        }
        if (n[0] < 1 || n[1] < 1 || n[2] < 2 || m[0] < 0 || m[1] < 0 || m[2] < 0) throw Error("register_blocks: bad block size");
        nc = n[2] / 2 + 1;
        Lc = L[2] / 2 + 1;
        N = static_cast<std::size_t>(n[0]) * n[1] * n[2];
        NC = static_cast<std::size_t>(n[0]) * n[1] * nc;
        NB = static_cast<std::size_t>(L[0]) * L[1] * L[2];
        NBC = static_cast<std::size_t>(L[0]) * L[1] * Lc;
        w0 = active[0] ? taper_window(n[0], opt.taper) : std::vector<float>(1, 1.0f);
        w1 = taper_window(n[1], opt.taper);
        w2 = taper_window(n[2], opt.taper);
    }

    // tapers of the padded block: flat wherever a displaced inner block can reach. `largest_row_shift`: the
    // largest per-frame displacement along the rows that is removed first.
    void big_tapers(double largest_row_shift, std::vector<float>& t0, std::vector<float>& t1, std::vector<float>& t2) const {
        const int reach1 = 1 + static_cast<int>(std::ceil(largest_row_shift));
        t0 = m[0] > 0 ? flat_taper(L[0], m[0] - 1, m[0] + n[0]) : std::vector<float>(static_cast<std::size_t>(L[0]), 1.0f);
        t1 = m[1] > 0 ? flat_taper(L[1], m[1] - reach1, m[1] + n[1] - 1 + reach1) : std::vector<float>(static_cast<std::size_t>(L[1]), 1.0f);
        t2 = m[2] > 0 ? flat_taper(L[2], m[2] - 1, m[2] + n[2]) : std::vector<float>(static_cast<std::size_t>(L[2]), 1.0f);
    }

    // the band kept, per bin of the inner block's half spectrum: weight (n0, n1, nc)
    void band_weights(const RegistrationOptions& opt, float* weight) const {
        const double lo2 = 2 * opt.f_lo * opt.f_lo, hi2 = 2 * opt.f_hi * opt.f_hi;
#pragma omp parallel for collapse(2) schedule(static)
        for (int k0 = 0; k0 < n[0]; ++k0)
            for (int k1 = 0; k1 < n[1]; ++k1) {
                const double f0 = frequency(k0, n[0]), f1 = frequency(k1, n[1]);
                double wq = 1.0;
                if (active[0]) {
                    const double c = std::cos(M_PI * f0);
                    wq = c * c;
                    if (n[0] % 2 == 0 && k0 == n[0] / 2) wq = 0;
                }
                if (active[1] && n[1] % 2 == 0 && k1 == n[1] / 2) wq = 0;
                float* w = weight + (static_cast<std::size_t>(k0) * n[1] + k1) * nc;
                for (int k2 = 0; k2 < nc; ++k2) {
                    const double f2 = frequency(k2, n[2]);
                    const double rho2 = f1 * f1 + f2 * f2;
                    double v = wq * (1.0 - std::exp(-rho2 / lo2)) * std::exp(-rho2 / hi2);
                    if (n[2] % 2 == 0 && k2 == n[2] / 2) v = 0;
                    // content that is constant along the rows or along the columns: stripes fixed to the detector or the beam
                    if (active[1] && (k2 <= opt.stripe_notch || std::min(k1, n[1] - k1) <= opt.stripe_notch)) v = 0;
                    w[k2] = static_cast<float>(v);
                }
            }
    }
};

// The array work of a registration on one device. Blocks are host arrays; staging() gives host memory the device
// takes fastest (page-locked for CUDA): a caller that fills those and passes them to set_a / set_b saves a copy.
struct RegistrationBackend {
    virtual ~RegistrationBackend() = default;
    virtual float* staging_a() = 0;  // n samples
    virtual float* staging_b() = 0;  // L samples
    // block a: windowed, transformed, conjugated and weighted. Returns its energy in the band.
    virtual double set_a(const float* a) = 0;
    // block b: per frame the mean removed and the rows displaced by row_shift[q] (null: none), tapered, transformed
    virtual void set_b(const float* b, const double* row_shift) = 0;
    // b displaced by d (fractions of a sample), its inner block windowed like a, and the cross spectrum with a.
    // Returns the energy of the displaced b in the band.
    virtual double cross(const double d[3]) = 0;
    // integer maximum of the correlation of the last cross() within +- search; true if it lies on the border
    virtual bool peak(const int search[3], int best[3]) = 0;
    // r(x) = sum_k Re[C_k exp(2 pi i f_k . x)] of the last cross(), its gradient and Hessian
    virtual void correlation(const double at[3], double& value, double g[3], double H[3][3]) = 0;
};

// weights: the band weights of geom (band_weights), which outlive the backend
std::unique_ptr<RegistrationBackend> make_cpu_registration(const RegistrationGeometry& geom, const RegistrationOptions& opt, const float* weights);
#ifdef HELIREC_WITH_CUDA
std::unique_ptr<RegistrationBackend> make_cuda_registration(const RegistrationGeometry& geom, const RegistrationOptions& opt, const float* weights);
#endif

}  // namespace helirec::calibration
