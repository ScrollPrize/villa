// Block cross-spectra: what the CPU and CUDA implementations share (bins, window, registration, curves).
#include "helirec/block_spectra.hpp"

#include <algorithm>
#include <cmath>
#include <complex>

#include "common/block_spectra_core.hpp"
#include "calibration/correlation_peak.hpp"

namespace helirec {

namespace core = block_spectra_core;

int BlockSpectraSpec::n_shells() const { return static_cast<int>(std::ceil(shells * std::sqrt(3.0))); }

void validate(const BlockSpectraSpec& s) {
    if (s.n < 8 || s.n % 2) throw Error("block spectra: the block edge must be even and >= 8");
    if (s.oversample < 1 || s.oversample > 8) throw Error("block spectra: oversample must be 1..8");
    if (s.shells < 1 || s.shells > s.n / 2) throw Error("block spectra: shells must be 1..n/2");
    if (s.thetas < 1 || s.thetas > 90) throw Error("block spectra: thetas must be 1..90");
    if (s.align && !(s.align_band > 0.f && s.align_band <= 0.5f)) throw Error("block spectra: align_band must be in (0, 0.5]");
    if (s.align && (s.max_shift < 0 || s.max_shift >= s.n / 4)) throw Error("block spectra: max_shift must be in [0, n/4)");
}

BlockCorrelator::BlockCorrelator(const BlockSpectraSpec& s) : spec_(s) {
    validate(spec_);
    counts_ = block_spectra::bin_counts(spec_);
}

int resample_interp_from_name(const std::string& name) {
    if (name == "linear") return interp_core::Linear;
    if (name == "cubic") return interp_core::Cubic;
    if (name == "lanczos3") return interp_core::Lanczos3;
    throw Error("resample: interp must be linear, cubic or lanczos3, not '" + name + "'");
}

namespace block_spectra {

std::vector<double> theta_tan2(const BlockSpectraSpec& s) {
    std::vector<double> t(std::max(1, s.thetas - 1), 0.0);
    for (int j = 0; j + 1 < s.thetas; ++j) {
        const double v = std::tan((j + 1) * (M_PI / 2) / s.thetas);
        t[j] = v * v;
    }
    return t;
}

std::vector<double> bin_counts(const BlockSpectraSpec& s) {
    const int n = s.n, nc = n / 2 + 1;
    const std::vector<double> tan2 = theta_tan2(s);
    const double scale = 2.0 * s.shells / n;
    std::vector<double> c(s.n_bins(), 0.0);
    for (int iz = 0; iz < n; ++iz)
        for (int iy = 0; iy < n; ++iy)
            for (int ix = 0; ix < nc; ++ix) {
                const int kz = core::signed_index(iz, n), ky = core::signed_index(iy, n);
                const double w = core::half_weight(kz, ky, ix, n);
                if (w > 0) c[core::bin_of(kz, ky, ix, scale, tan2.data(), s.thetas)] += w;
            }
    return c;
}

std::vector<float> hann(int n, int oversample) {
    std::vector<float> w(static_cast<std::size_t>(n) * oversample);
    for (std::size_t j = 0; j < w.size(); ++j)
        w[j] = static_cast<float>(0.5 * (1.0 - std::cos(2 * M_PI * (static_cast<double>(j) / oversample + 0.5) / n)));
    return w;
}

int align_box(const BlockSpectraSpec& s) {
    return std::min(s.n / 2 - 1, static_cast<int>(std::ceil(s.align_band * s.n)));
}

void align(const BlockSpectraSpec& s, const float* box, const std::vector<double>& aa, const std::vector<double>& bb,
           std::array<double, 3>& shift, double& peak) {
    using cd = std::complex<double>;
    const int n = s.n, kb = align_box(s), side = 2 * kb + 1, nx = kb + 1, m = s.max_shift, ns = 2 * m + 1;
    const std::vector<double> tan2 = theta_tan2(s);
    const double scale = 2.0 * s.shells / n;
    // shells in the band, each weighed by 1 / sqrt(its two powers): the objective is the sum of their correlations
    const int band = std::min(s.n_shells(), std::max(1, static_cast<int>(std::lround(s.align_band * 2 * s.shells))));
    std::vector<double> g(s.n_shells(), 0.0);
    int used = 0;
    for (int sh = 0; sh < band; ++sh) {
        double pa = 0, pb = 0;
        for (int t = 0; t < s.thetas; ++t) {
            pa += aa[static_cast<std::size_t>(sh) * s.thetas + t];
            pb += bb[static_cast<std::size_t>(sh) * s.thetas + t];
        }
        if (pa > 0 && pb > 0) {
            g[sh] = 1.0 / std::sqrt(pa * pb);
            ++used;
        }
    }
    shift = {0, 0, 0};
    peak = 0;
    if (!used) return;
    // W = weight x cross-spectrum on the box (conjugate half counted twice)
    std::vector<cd> W(static_cast<std::size_t>(side) * side * nx);
    for (int a = 0; a < side; ++a)
        for (int b = 0; b < side; ++b)
            for (int c = 0; c < nx; ++c) {
                const int kz = a - kb, ky = b - kb;
                const std::size_t i = (static_cast<std::size_t>(a) * side + b) * nx + c;
                const int sh = core::bin_of(kz, ky, c, scale, tan2.data(), s.thetas) / s.thetas;
                const double w = (kz == 0 && ky == 0 && c == 0) ? 0.0 : g[sh] * (c == 0 ? 1.0 : 2.0);
                W[i] = w * cd(box[2 * i], box[2 * i + 1]);
            }
    // R(s) = Re sum W exp(2 pi i k.s / n) at the integer shifts of [-m, m]^3, one axis at a time
    std::vector<cd> ph(n);
    for (int j = 0; j < n; ++j) ph[j] = std::polar(1.0, 2 * M_PI * j / n);
    auto phase = [&](int k, int sh) { return ph[((k * sh) % n + n) % n]; };
    std::vector<cd> t1(static_cast<std::size_t>(side) * side * ns), t2(static_cast<std::size_t>(side) * ns * ns);
#pragma omp parallel for schedule(static)
    for (int a = 0; a < side; ++a)
        for (int b = 0; b < side; ++b)
            for (int sx = 0; sx < ns; ++sx) {
                cd acc = 0;
                const cd* row = W.data() + (static_cast<std::size_t>(a) * side + b) * nx;
                for (int c = 0; c < nx; ++c) acc += row[c] * phase(c, sx - m);
                t1[(static_cast<std::size_t>(a) * side + b) * ns + sx] = acc;
            }
#pragma omp parallel for schedule(static)
    for (int a = 0; a < side; ++a)
        for (int sy = 0; sy < ns; ++sy)
            for (int sx = 0; sx < ns; ++sx) {
                cd acc = 0;
                for (int b = 0; b < side; ++b) acc += t1[(static_cast<std::size_t>(a) * side + b) * ns + sx] * phase(b - kb, sy - m);
                t2[(static_cast<std::size_t>(a) * ns + sy) * ns + sx] = acc;
            }
    double best = -1e300;
    int bz = 0, by = 0, bx = 0;
    for (int sz = 0; sz < ns; ++sz)
        for (int sy = 0; sy < ns; ++sy)
            for (int sx = 0; sx < ns; ++sx) {
                cd acc = 0;
                for (int a = 0; a < side; ++a) acc += t2[(static_cast<std::size_t>(a) * ns + sy) * ns + sx] * phase(a - kb, sz - m);
                if (acc.real() > best) {
                    best = acc.real();
                    bz = sz - m; by = sy - m; bx = sx - m;
                }
            }
    // Ceres continuous peak: value, gradient and Hessian of R are sums over the box
    double p[3] = {static_cast<double>(bz), static_cast<double>(by), static_cast<double>(bx)};
    auto evaluate = [&](const double* q, double* grad, double* hess) {
        double v = 0, g0 = 0, g1 = 0, g2 = 0, h00 = 0, h01 = 0, h02 = 0, h11 = 0, h12 = 0, h22 = 0;
        const double w0 = 2 * M_PI / n;
#pragma omp parallel for schedule(static) reduction(+ : v, g0, g1, g2, h00, h01, h02, h11, h12, h22)
        for (int a = 0; a < side; ++a)
            for (int b = 0; b < side; ++b) {
                const double kz = (a - kb) * w0, ky = (b - kb) * w0;
                const cd* row = W.data() + (static_cast<std::size_t>(a) * side + b) * nx;
                for (int c = 0; c < nx; ++c) {
                    const double kx = c * w0;
                    const cd e = row[c] * std::polar(1.0, kz * q[0] + ky * q[1] + kx * q[2]);
                    v += e.real();
                    g0 -= kz * e.imag(); g1 -= ky * e.imag(); g2 -= kx * e.imag();
                    h00 -= kz * kz * e.real(); h01 -= kz * ky * e.real(); h02 -= kz * kx * e.real();
                    h11 -= ky * ky * e.real(); h12 -= ky * kx * e.real(); h22 -= kx * kx * e.real();
                }
            }
        grad[0] = g0; grad[1] = g1; grad[2] = g2;
        hess[0] = h00; hess[1] = h01; hess[2] = h02; hess[3] = h11; hess[4] = h12; hess[5] = h22;
        return v;
    };
    const bool active[3] = {true, true, true};
    const auto result = calibration::fitting::correlation_peak([&](const double* q, double& v, double* grad, double (*H)[3]) {
        double h[6];
        v = evaluate(q, grad, h);
        H[0][0] = h[0]; H[0][1] = H[1][0] = h[1]; H[0][2] = H[2][0] = h[2];
        H[1][1] = h[3]; H[1][2] = H[2][1] = h[4]; H[2][2] = h[5];
    }, active, p, used);
    if (!result.maximum) { peak = 0; return; }
    for (int a = 0; a < 3; ++a) shift[a] = -result.at[a];
    peak = result.value / used;
}

}  // namespace block_spectra

ShellCurves shell_curves(const BlockSpectraSpec& spec, const std::vector<double>& counts, const std::vector<double>& aa,
                         const std::vector<double>& bb, const std::vector<double>& ab, double theta_lo, double theta_hi) {
    const std::size_t nb = static_cast<std::size_t>(spec.n_bins());
    if (counts.size() != nb || aa.size() != nb || bb.size() != nb || ab.size() != nb)
        throw Error("shell_curves: tables must be (n_shells, thetas)");
    const int ns = spec.n_shells(), T = spec.thetas;
    const double width = 90.0 / T, eps = 1e-9;
    ShellCurves c;
    for (auto* v : {&c.frequency, &c.count, &c.correlation, &c.matched, &c.snr_split, &c.snr_reference, &c.filter, &c.power_a, &c.power_b}) v->assign(ns, 0.0);
    const double nan = std::nan("");
    for (int s = 0; s < ns; ++s) {
        double n = 0, a = 0, b = 0, x = 0, m = 0;
        for (int t = 0; t < T; ++t) {
            if (t * width < theta_lo - eps || (t + 1) * width > theta_hi + eps) continue;
            const std::size_t i = static_cast<std::size_t>(s) * T + t;
            n += counts[i]; a += aa[i]; b += bb[i]; x += ab[i];
            if (aa[i] > 0) m += ab[i] * std::fabs(ab[i]) / aa[i];
        }
        c.frequency[s] = (s + 0.5) * 0.5 / spec.shells;
        c.count[s] = n;
        const bool ok = n > 0 && a > 0 && b > 0;
        const double r = ok ? x / std::sqrt(a * b) : nan;
        c.correlation[s] = r;
        c.matched[s] = ok ? (m < 0 ? -1.0 : 1.0) * std::sqrt(std::fabs(m) / b) : nan;
        const double rp = std::max(r, 0.0);
        c.snr_split[s] = ok ? (rp < 1 ? rp / (1 - rp) : INFINITY) : nan;
        c.snr_reference[s] = ok ? (rp < 1 ? rp * rp / (1 - rp * rp) : INFINITY) : nan;
        c.filter[s] = ok ? x / a : nan;
        c.power_a[s] = n > 0 ? a / n : nan;
        c.power_b[s] = n > 0 ? b / n : nan;
    }
    return c;
}

}  // namespace helirec
