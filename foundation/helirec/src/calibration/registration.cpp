#include "helirec/calibration/registration.hpp"

#include <fftw3.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <memory>
#include <mutex>
#include <random>
#include <vector>

#include "calibration/registration_core.hpp"
#include "calibration/view_plan.hpp"
#include "helirec/common.hpp"
#include "helirec/fourier.hpp"  // fftw_planner_mutex(): the one lock around FFTW's planner

namespace helirec::calibration {

namespace {

using cfloat = std::complex<float>;
using cdouble = std::complex<double>;

struct FftwFree {
    void operator()(void* p) const { fftwf_free(p); }
};
using RealBuffer = std::unique_ptr<float, FftwFree>;
using ComplexBuffer = std::unique_ptr<fftwf_complex, FftwFree>;

RealBuffer alloc_real(std::size_t n) {
    RealBuffer p(fftwf_alloc_real(n));
    if (!p) throw Error("register_blocks: out of memory");
    return p;
}
ComplexBuffer alloc_complex(std::size_t n) {
    ComplexBuffer p(fftwf_alloc_complex(n));
    if (!p) throw Error("register_blocks: out of memory");
    return p;
}

void init_fftw_threads() {
    static std::once_flag once;
    std::call_once(once, [] { fftwf_init_threads(); });
}

// The array work on the CPU: FFTW transforms, OpenMP loops.
class CpuBackend final : public RegistrationBackend {
  public:
    CpuBackend(const RegistrationGeometry& g, const RegistrationOptions& opt, const float* weights) : g_(g), opt_(opt), weight_(weights) {
        r_ = alloc_real(g_.N);
        aw_ = alloc_complex(g_.NC);
        c_ = alloc_complex(g_.NC);
        rbig_ = alloc_real(g_.NB);
        cbig_ = alloc_complex(g_.NBC);
        cwork_ = alloc_complex(g_.NBC);
        const int threads = opt.threads > 0 ? opt.threads : std::min(8, omp_get_max_threads());
        init_fftw_threads();
        {
            std::lock_guard<std::mutex> lock(fftw_planner_mutex());
            const int* n = g_.n;
            const int* L = g_.L;
            fftwf_plan_with_nthreads(threads);
            forward_ = fftwf_plan_dft_r2c_3d(n[0], n[1], n[2], r_.get(), c_.get(), FFTW_ESTIMATE);
            inverse_ = fftwf_plan_dft_c2r_3d(n[0], n[1], n[2], c_.get(), r_.get(), FFTW_ESTIMATE);
            big_forward_ = fftwf_plan_dft_r2c_3d(L[0], L[1], L[2], rbig_.get(), cbig_.get(), FFTW_ESTIMATE);
            big_inverse_ = fftwf_plan_dft_c2r_3d(L[0], L[1], L[2], cwork_.get(), rbig_.get(), FFTW_ESTIMATE);
            fftwf_plan_with_nthreads(1);
            // one frame's rows, every column: transforms along axis 1 of a (L1, L2) frame
            rows_forward_ = fftwf_plan_many_dft_r2c(1, &L[1], L[2], rbig_.get(), nullptr, L[2], 1, cwork_.get(), nullptr, L[2], 1, FFTW_ESTIMATE | FFTW_UNALIGNED);
            rows_inverse_ = fftwf_plan_many_dft_c2r(1, &L[1], L[2], cwork_.get(), nullptr, L[2], 1, rbig_.get(), nullptr, L[2], 1, FFTW_ESTIMATE | FFTW_UNALIGNED);
        }
        if (!forward_ || !inverse_ || !big_forward_ || !big_inverse_ || !rows_forward_ || !rows_inverse_)
            throw Error("register_blocks: FFTW could not plan the transforms");
    }

    ~CpuBackend() override {
        std::lock_guard<std::mutex> lock(fftw_planner_mutex());
        for (fftwf_plan p : {forward_, inverse_, big_forward_, big_inverse_, rows_forward_, rows_inverse_})
            if (p) fftwf_destroy_plan(p);
    }

    float* staging_a() override {
        if (!stage_a_) stage_a_ = alloc_real(g_.N);
        return stage_a_.get();
    }
    float* staging_b() override {
        if (!stage_b_) stage_b_ = alloc_real(g_.NB);
        return stage_b_.get();
    }

    double set_a(const float* a) override {
        prepare(a, g_.n[1], g_.n[2]);
        fftwf_execute_dft_r2c(forward_, r_.get(), aw_.get());
        cfloat* A = reinterpret_cast<cfloat*>(aw_.get());
        const int nc = g_.nc;
        double ea = 0;
#pragma omp parallel for schedule(static) reduction(+ : ea)
        for (std::ptrdiff_t line = 0; line < static_cast<std::ptrdiff_t>(g_.n[0]) * g_.n[1]; ++line) {
            cfloat* pa = A + static_cast<std::size_t>(line) * nc;
            const float* w = weight_ + static_cast<std::size_t>(line) * nc;
            for (int k2 = 0; k2 < nc; ++k2) {
                ea += (k2 == 0 ? 1.0 : 2.0) * w[k2] * std::norm(pa[k2]);
                pa[k2] = std::conj(pa[k2]) * w[k2];
            }
        }
        return ea;
    }

    void set_b(const float* b, const double* row_shift) override {
        const int* L = g_.L;
        double largest = 0;
        if (row_shift)
            for (int q = 0; q < L[0]; ++q) largest = std::max(largest, std::fabs(row_shift[q]));
        std::vector<float> t0, t1, t2;
        g_.big_tapers(largest, t0, t1, t2);
        const std::size_t frame = static_cast<std::size_t>(L[1]) * L[2];
        float* big = rbig_.get();
        const int k1n = L[1] / 2 + 1;
        const bool displace = row_shift && largest > 0;
#pragma omp parallel
        {
            ComplexBuffer spec;
            if (displace) spec = alloc_complex(static_cast<std::size_t>(k1n) * L[2]);
#pragma omp for schedule(static)
            for (int q = 0; q < L[0]; ++q) {
                const float* src = b + static_cast<std::size_t>(q) * frame;
                float* dst = big + static_cast<std::size_t>(q) * frame;
                double sum = 0;
                for (std::size_t i = 0; i < frame; ++i) sum += src[i];
                const float mean = static_cast<float>(sum / static_cast<double>(frame));
                for (int v = 0; v < L[1]; ++v)
                    for (int u = 0; u < L[2]; ++u) dst[static_cast<std::size_t>(v) * L[2] + u] = (src[static_cast<std::size_t>(v) * L[2] + u] - mean) * t1[v];
                if (displace && row_shift[q] != 0.0) {
                    fftwf_execute_dft_r2c(rows_forward_, dst, spec.get());
                    cfloat* s = reinterpret_cast<cfloat*>(spec.get());
                    const std::vector<cdouble> e = phasors(L[1], k1n, row_shift[q]);
                    for (int k = 0; k < k1n; ++k) {
                        const cfloat f(static_cast<float>(e[k].real() / L[1]), static_cast<float>(e[k].imag() / L[1]));
                        for (int u = 0; u < L[2]; ++u) s[static_cast<std::size_t>(k) * L[2] + u] *= f;
                    }
                    fftwf_execute_dft_c2r(rows_inverse_, spec.get(), dst);
                }
                for (int v = 0; v < L[1]; ++v)
                    for (int u = 0; u < L[2]; ++u) dst[static_cast<std::size_t>(v) * L[2] + u] *= t0[q] * t2[u];
            }
        }
        fftwf_execute_dft_r2c(big_forward_, rbig_.get(), cbig_.get());
    }

    double cross(const double d[3]) override {
        const int* L = g_.L;
        const int Lc = g_.Lc, nc = g_.nc;
        const std::vector<cdouble> e0 = phasors(L[0], L[0], d[0]), e1 = phasors(L[1], L[1], d[1]), e2 = phasors(L[2], Lc, d[2]);
        const cfloat* src = reinterpret_cast<const cfloat*>(cbig_.get());
        cfloat* work = reinterpret_cast<cfloat*>(cwork_.get());
#pragma omp parallel for collapse(2) schedule(static)
        for (int k0 = 0; k0 < L[0]; ++k0)
            for (int k1 = 0; k1 < L[1]; ++k1) {
                const cdouble e01 = e0[k0] * e1[k1];
                const std::size_t base = (static_cast<std::size_t>(k0) * L[1] + k1) * Lc;
                for (int k2 = 0; k2 < Lc; ++k2) {
                    const cdouble e = e01 * e2[k2];
                    work[base + k2] = src[base + k2] * cfloat(static_cast<float>(e.real()), static_cast<float>(e.imag()));
                }
            }
        fftwf_execute_dft_c2r(big_inverse_, cwork_.get(), rbig_.get());
        prepare(rbig_.get() + (static_cast<std::size_t>(g_.m[0]) * L[1] + g_.m[1]) * L[2] + g_.m[2], L[1], L[2]);
        fftwf_execute_dft_r2c(forward_, r_.get(), c_.get());
        // C = conj(A) W B
        cfloat* C = reinterpret_cast<cfloat*>(c_.get());
        const cfloat* AW = reinterpret_cast<const cfloat*>(aw_.get());
        double eb = 0;
#pragma omp parallel for schedule(static) reduction(+ : eb)
        for (std::ptrdiff_t line = 0; line < static_cast<std::ptrdiff_t>(g_.n[0]) * g_.n[1]; ++line) {
            cfloat* pc = C + static_cast<std::size_t>(line) * nc;
            const cfloat* pa = AW + static_cast<std::size_t>(line) * nc;
            const float* w = weight_ + static_cast<std::size_t>(line) * nc;
            for (int k2 = 0; k2 < nc; ++k2) {
                eb += (k2 == 0 ? 1.0 : 2.0) * w[k2] * std::norm(pc[k2]);
                pc[k2] = pa[k2] * pc[k2];
            }
        }
        return eb;
    }

    bool peak(const int s[3], int best[3]) override {
        const int* n = g_.n;
        // the transform destroys its input: a copy of the cross spectrum
        if (!copy_) copy_ = alloc_complex(g_.NC);
        const cfloat* C = reinterpret_cast<const cfloat*>(c_.get());
        std::copy(C, C + g_.NC, reinterpret_cast<cfloat*>(copy_.get()));
        fftwf_execute_dft_c2r(inverse_, copy_.get(), r_.get());
        float top = -INFINITY;
        best[0] = best[1] = best[2] = 0;
        for (int d0 = -s[0]; d0 <= s[0]; ++d0)
            for (int d1 = -s[1]; d1 <= s[1]; ++d1) {
                const float* row = r_.get() + (static_cast<std::size_t>((d0 + n[0]) % n[0]) * n[1] + (d1 + n[1]) % n[1]) * n[2];
                for (int d2 = -s[2]; d2 <= s[2]; ++d2) {
                    const float v = row[(d2 + n[2]) % n[2]];
                    if (v > top) top = v, best[0] = d0, best[1] = d1, best[2] = d2;
                }
            }
        bool at_edge = false;
        for (int j = 0; j < 3; ++j)
            if (s[j] > 0 && std::abs(best[j]) == s[j]) at_edge = true;
        return at_edge;
    }

    void correlation(const double at[3], double& value, double g[3], double H[3][3]) override {
        const int n0 = g_.n[0], n1 = g_.n[1], n2 = g_.n[2], nc = g_.nc;
        std::vector<cdouble> e0(n0), e1(n1), e2(nc);
        for (int k = 0; k < n0; ++k) e0[k] = std::polar(1.0, 2 * M_PI * frequency(k, n0) * at[0]);
        for (int k = 0; k < n1; ++k) e1[k] = std::polar(1.0, 2 * M_PI * frequency(k, n1) * at[1]);
        for (int k = 0; k < nc; ++k) e2[k] = std::polar(1.0, 2 * M_PI * frequency(k, n2) * at[2]);
        const cfloat* C = reinterpret_cast<const cfloat*>(c_.get());
        double v = 0, g0 = 0, g1 = 0, g2 = 0, h00 = 0, h01 = 0, h02 = 0, h11 = 0, h12 = 0, h22 = 0;
#pragma omp parallel for collapse(2) schedule(static) reduction(+ : v, g0, g1, g2, h00, h01, h02, h11, h12, h22)
        for (int k0 = 0; k0 < n0; ++k0) {
            for (int k1 = 0; k1 < n1; ++k1) {
                const double q0 = 2 * M_PI * frequency(k0, n0), q1 = 2 * M_PI * frequency(k1, n1);
                const cdouble e01 = e0[k0] * e1[k1];
                const cfloat* pc = C + (static_cast<std::size_t>(k0) * n1 + k1) * nc;
                double sr = 0, si = 0, sr2 = 0, si2 = 0, sr22 = 0, si22 = 0;  // sums over k2 of t, q2 t, q2^2 t (Re, Im)
                for (int k2 = 0; k2 < nc; ++k2) {
                    const double mult = (k2 == 0) ? 1.0 : 2.0;
                    const double q2 = 2 * M_PI * frequency(k2, n2);
                    const cdouble t = cdouble(pc[k2].real(), pc[k2].imag()) * e2[k2];
                    sr += mult * t.real();
                    si += mult * t.imag();
                    sr2 += mult * q2 * t.real();
                    si2 += mult * q2 * t.imag();
                    sr22 += mult * q2 * q2 * t.real();
                    si22 += mult * q2 * q2 * t.imag();
                }
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
    }

  private:
    // the frame's weighted mean removed, then the window: r_ = (src - mean) w0 w1 w2. src has the given row and
    // column strides (the inner block inside a larger one).
    void prepare(const float* src, int stride1, int stride2) {
        const int n0 = g_.n[0], n1 = g_.n[1], n2 = g_.n[2];
        const std::vector<float>&w0 = g_.w0, &w1 = g_.w1, &w2 = g_.w2;
        float* dst_all = r_.get();
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n0; ++i) {
            const float* s = src + static_cast<std::size_t>(i) * stride1 * stride2;
            float* d = dst_all + static_cast<std::size_t>(i) * n1 * n2;
            double sum = 0, weight = 0;
            for (int j = 0; j < n1; ++j)
                for (int k = 0; k < n2; ++k) {
                    const double w = static_cast<double>(w1[j]) * w2[k];
                    sum += w * s[static_cast<std::size_t>(j) * stride2 + k];
                    weight += w;
                }
            const float mean = static_cast<float>(sum / weight);
            for (int j = 0; j < n1; ++j) {
                const float wj = w0[i] * w1[j];
                for (int k = 0; k < n2; ++k) d[static_cast<std::size_t>(j) * n2 + k] = (s[static_cast<std::size_t>(j) * stride2 + k] - mean) * wj * w2[k];
            }
        }
    }

    RegistrationGeometry g_;
    RegistrationOptions opt_;
    RealBuffer r_, rbig_, stage_a_, stage_b_;
    ComplexBuffer aw_, c_, copy_, cbig_, cwork_;
    const float* weight_;
    fftwf_plan forward_ = nullptr, inverse_ = nullptr, big_forward_ = nullptr, big_inverse_ = nullptr, rows_forward_ = nullptr, rows_inverse_ = nullptr;
};

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

struct Step {
    double delta[3] = {0, 0, 0};  // remaining displacement of the (displaced) b against a
    double correlation = 0;
    bool at_edge = false, at_maximum = false;
};

// b displaced by d, correlated with a: the continuous maximum of the correlation, by Newton steps from zero or
// (search) from its integer maximum
Step evaluate(RegistrationBackend& be, const RegistrationGeometry& geom, const RegistrationOptions& opt, double ea, const double d[3], bool search) {
    const double eb = be.cross(d);
    Step step;
    if (!(ea > 0) || !(eb > 0)) return step;  // a block without content in the band
    double at[3] = {0, 0, 0};
    if (search) {
        int s[3], best[3];
        for (int j = 0; j < 3; ++j) s[j] = geom.active[j] ? std::min(opt.search[j], (geom.n[j] - 1) / 2) : 0;
        step.at_edge = be.peak(s, best);
        for (int j = 0; j < 3; ++j) at[j] = best[j];
    }
    double r = 0, g[3], H[3][3];
    be.correlation(at, r, g, H);
    for (int it = 0; it < 40; ++it) {
        double x[3];
        if (!newton_step(H, g, geom.active, x)) break;  // not near a maximum
        double largest = 0;
        for (int j = 0; j < 3; ++j) {
            x[j] = std::clamp(x[j], -0.5, 0.5);
            largest = std::max(largest, std::fabs(x[j]));
        }
        if (largest < 1e-6) {
            step.at_maximum = true;
            break;
        }
        double trial[3], rt = 0, gt[3], Ht[3][3], scale = 1.0;
        bool accepted = false;
        for (int half = 0; half < 8; ++half, scale *= 0.5) {
            for (int j = 0; j < 3; ++j) trial[j] = at[j] + scale * x[j];
            be.correlation(trial, rt, gt, Ht);
            if (rt >= r) {
                accepted = true;
                break;
            }
        }
        if (!accepted) {
            // within the rounding of the sums the correlation no longer increases: a step this small is the maximum
            step.at_maximum = largest < 1e-3;
            break;
        }
        for (int j = 0; j < 3; ++j) at[j] = trial[j], g[j] = gt[j];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) H[i][j] = Ht[i][j];
        r = rt;
    }
    for (int j = 0; j < 3; ++j) step.delta[j] = geom.active[j] ? at[j] : 0.0;
    step.correlation = r / std::sqrt(ea * eb);
    return step;
}

}  // namespace

std::unique_ptr<RegistrationBackend> make_cpu_registration(const RegistrationGeometry& geom, const RegistrationOptions& opt, const float* weights) {
    return std::make_unique<CpuBackend>(geom, opt, weights);
}

struct Registrar::Impl {
    RegistrationGeometry geom;
    RegistrationOptions opt;
    Device device;
    std::vector<float> weights;  // the band kept, per bin of the inner block's half spectrum
    std::unique_ptr<RegistrationBackend> be;
    Impl(const int n[3], const int margin[3], const RegistrationOptions& o, Device dev) : geom(n, margin, o), opt(o), device(dev), weights(geom.NC) {
        geom.band_weights(opt, weights.data());
        if (dev == Device::CUDA) {
#ifdef HELIREC_WITH_CUDA
            be = make_cuda_registration(geom, opt, weights.data());
#else
            throw Error("register_blocks: built without CUDA");
#endif
        } else {
            be = make_cpu_registration(geom, opt, weights.data());
        }
    }
};

Registrar::Registrar(const int n[3], const int margin[3], const RegistrationOptions& opt, Device device)
    : impl_(std::make_unique<Impl>(n, margin, opt, device)) {}
Registrar::~Registrar() = default;

bool Registrar::fits(const int n[3], const int margin[3], Device device) const {
    for (int j = 0; j < 3; ++j)
        if (impl_->geom.n[j] != n[j] || impl_->geom.m[j] != margin[j]) return false;
    return impl_->device == device;
}
float* Registrar::staging_a() { return impl_->be->staging_a(); }
float* Registrar::staging_b() { return impl_->be->staging_b(); }

Registration Registrar::run(const float* a, const float* b, const double* row_shift) {
    RegistrationBackend& be = *impl_->be;
    const RegistrationGeometry& geom = impl_->geom;
    const RegistrationOptions& opt = impl_->opt;
    const auto can_displace = [&](int j) { return geom.active[j] && geom.m[j] >= 2; };
    const double ea = be.set_a(a);
    be.set_b(b, row_shift);
    Registration out;
    double d[3] = {0, 0, 0};
    Step s = evaluate(be, geom, opt, ea, d, true);
    out.evaluations = 1;
    out.correlation = s.correlation;
    out.at_edge = s.at_edge;
    for (int j = 0; j < 3; ++j) {
        out.shift[j] = s.delta[j];
        // a displacement of most of a sample or more: the caller cuts b again
        if (std::fabs(s.delta[j]) > 0.75) out.whole[j] = static_cast<int>(std::lround(s.delta[j]));
    }
    if (out.whole[0] || out.whole[1] || out.whole[2] || !s.at_maximum) return out;

    // The correlation of two blocks under one window prefers no displacement: its maximum understates what is
    // left. Displace b by the estimate and repeat until nothing is left; the understatement is learned per axis
    // from the last two steps (secant).
    double prev_d[3] = {0, 0, 0}, prev_delta[3] = {0, 0, 0}, gain[3] = {1, 1, 1};
    bool have_prev = false;
    for (; out.evaluations < opt.max_evaluations;) {
        double next[3];
        for (int j = 0; j < 3; ++j) {
            if (!can_displace(j)) {
                next[j] = 0;
                continue;
            }
            if (have_prev && std::fabs(d[j] - prev_d[j]) > 1e-4) {
                const double slope = (prev_delta[j] - s.delta[j]) / (d[j] - prev_d[j]);  // 1 for an unbiased step
                gain[j] = 1.0 / std::clamp(slope, 0.1, 1.5);
            }
            next[j] = d[j] + gain[j] * s.delta[j];
            // more than the padded block can be displaced by: the caller cuts b again
            if (std::fabs(next[j]) > 0.75) out.whole[j] = static_cast<int>(std::lround(next[j]));
        }
        if (out.whole[0] || out.whole[1] || out.whole[2]) {
            for (int j = 0; j < 3; ++j) out.shift[j] = can_displace(j) ? next[j] : s.delta[j];
            return out;
        }
        for (int j = 0; j < 3; ++j) prev_d[j] = d[j], prev_delta[j] = s.delta[j], d[j] = next[j];
        have_prev = true;
        s = evaluate(be, geom, opt, ea, d, false);
        ++out.evaluations;
        out.correlation = s.correlation;
        double left = 0;
        for (int j = 0; j < 3; ++j) {
            out.shift[j] = d[j] + s.delta[j];
            if (can_displace(j)) left = std::max(left, std::fabs(s.delta[j]));
        }
        if (!s.at_maximum) return out;
        if (left < opt.tolerance) {
            out.converged = true;
            break;
        }
    }
    return out;
}

Registration register_blocks(const float* a, const float* b, const int n[3], const int margin[3], const double* row_shift,
                             const RegistrationOptions& opt, Device device) {
    return Registrar(n, margin, opt, device).run(a, b, row_shift);
}

// ------------------------------------------------------------------------------------------------ self test

namespace {

// A field of Gaussian blobs, defined at every real position: samples of it on any grid are exact.
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
    // out(i, j, k) = f(x0[i], row0[i] + j, o2 + step2 * k); a field without extent along axis 0 if !along0
    void render(std::vector<float>& out, int n0, int n1, int n2, const std::vector<double>& x0, const std::vector<double>& row0, double o2,
                double step2 = 1.0, bool along0 = true) const {
        out.assign(static_cast<std::size_t>(n0) * n1 * n2, 0.0f);
#pragma omp parallel for schedule(dynamic, 1)
        for (int i = 0; i < n0; ++i) {
            std::vector<double> p1(n1), p2(n2);
            for (std::size_t m = 0; m < amp.size(); ++m) {
                double w0 = 1.0;
                if (along0) {
                    const double x = (x0[i] - c0[m]) / s0;
                    if (std::fabs(x) >= 4.5) continue;
                    w0 = std::exp(-0.5 * x * x);
                }
                const auto profile = [](std::vector<double>& p, int n, double origin, double step, double centre, double sigma, int& a, int& b) {
                    a = n, b = -1;
                    for (int k = 0; k < n; ++k) {
                        const double x = (origin + step * k - centre) / sigma;
                        p[k] = std::fabs(x) < 4.5 ? std::exp(-0.5 * x * x) : 0.0;
                        if (p[k] != 0.0) a = std::min(a, k), b = std::max(b, k);
                    }
                };
                int a1, b1, a2, b2;
                profile(p1, n1, row0[i], 1.0, c1[m], s1, a1, b1);
                if (b1 < a1) continue;
                profile(p2, n2, o2, step2, c2[m], s2, a2, b2);
                for (int j = a1; j <= b1; ++j) {
                    const double w = amp[m] * w0 * p1[j];
                    float* row = out.data() + (static_cast<std::size_t>(i) * n1 + j) * n2;
                    for (int k = a2; k <= b2; ++k) row[k] += static_cast<float>(w * p2[k]);
                }
            }
        }
    }
    // the plain case: out(i, j, k) = f(o0 + i, o1 + j, o2 + step2 k)
    void render(std::vector<float>& out, int n0, int n1, int n2, double o0, double o1, double o2, double step2 = 1.0, bool along0 = true) const {
        std::vector<double> x0(n0), row0(n0, o1);
        for (int i = 0; i < n0; ++i) x0[i] = o0 + i;
        render(out, n0, n1, n2, x0, row0, o2, step2, along0);
    }
};

void add_noise(std::vector<float>& v, std::mt19937& rng, double sigma) {
    std::normal_distribution<float> g(0, static_cast<float>(sigma));
    for (float& x : v) x += g(rng);
}

}  // namespace

bool registration_selftest(bool verbose, Device device) {
    bool all = true;
    const double tolerance = 0.01;
    const auto report = [&](const char* name, const double got[3], const double want[3], const Registration& r, bool flags_ok) {
        double worst = 0;
        for (int j = 0; j < 3; ++j) worst = std::max(worst, std::fabs(got[j] - want[j]));
        const bool ok = worst < tolerance && flags_ok;
        all &= ok;
        if (verbose)
            std::printf("%-52s %s  want (%+.3f %+.3f %+.3f) got (%+.4f %+.4f %+.4f) worst error %.4f, correlation %.3f, %d evaluations\n", name,
                        ok ? "ok  " : "FAIL", want[0], want[1], want[2], got[0], got[1], got[2], worst, r.correlation, r.evaluations);
    };
    std::mt19937 rng(20251111);
    double noise = 0.05;

    // b is cut again by whole samples until the registration reports none: what a caller does with real frames
    const auto with_recuts = [&](const std::vector<float>& a, const int n[3], const int m[3], const auto& render_b, double got[3]) {
        int extra[3] = {0, 0, 0};
        Registration r;
        std::vector<float> b;
        for (int pass = 0; pass < 6; ++pass) {
            render_b(extra, b);
            add_noise(b, rng, noise);
            r = register_blocks(a.data(), b.data(), n, m, nullptr, {}, device);
            for (int j = 0; j < 3; ++j) got[j] = extra[j] + r.shift[j];
            if (!r.whole[0] && !r.whole[1] && !r.whole[2]) break;
            for (int j = 0; j < 3; ++j) extra[j] += r.whole[j];
        }
        return r;
    };

    // 1 - 4: three-dimensional displacements, a(x) = b(x + margin + d). Fine content, then content with a long
    // correlation along the frames and the rows, where a windowed correlation is pulled towards zero the most.
    for (int kind = 0; kind < 2; ++kind) {
        const int n[3] = {16, 96, 240}, m[3] = {8, 12, 8};
        const int L[3] = {n[0] + 2 * m[0], n[1] + 2 * m[1], n[2] + 2 * m[2]};
        const double s0 = kind == 0 ? 1.5 : 5.0, s1 = kind == 0 ? 2.5 : 9.0;
        // the long-correlation case is there to show the absence of a pull towards zero: little noise, so that an
        // error of a hundredth of a sample is not noise
        noise = kind == 0 ? 0.05 : 0.01;
        BlobField f(rng, kind == 0 ? 8000 : 2500, -40, n[0] + 40, -60, n[1] + 60, -40, n[2] + 40, s0, s1, 2.5);
        std::vector<float> a;
        f.render(a, n[0], n[1], n[2], 0.0, 0.0, 0.0);
        add_noise(a, rng, noise);
        const double cases[3][3] = {{0.27, -0.41, 0.33}, {-1.30, 7.62, -11.20}, {0.45, 0.50, -0.25}};
        const char* names[2][3] = {{"3D fractional displacement", "3D displacement of several samples", "3D near half a sample"},
                                   {"3D fractional, long correlation", "3D several samples, long correlation", "3D near half, long correlation"}};
        for (int c = 0; c < 3; ++c) {
            const double* d = cases[c];
            double got[3];
            const Registration r = with_recuts(a, n, m, [&](const int extra[3], std::vector<float>& b) {
                f.render(b, L[0], L[1], L[2], -m[0] - d[0] + extra[0], -m[1] - d[1] + extra[1], -m[2] - d[2] + extra[2]);
            }, got);
            report(names[kind][c], got, d, r, r.converged && !r.at_edge);
        }
        if (kind == 0) {
            // a level per frame and a smooth profile added to b (beam drift, a flat that does not fit)
            const double d[3] = {0.27, -0.41, 0.33};
            std::vector<float> b;
            f.render(b, L[0], L[1], L[2], -m[0] - d[0], -m[1] - d[1], -m[2] - d[2]);
            for (int i = 0; i < L[0]; ++i)
                for (int j = 0; j < L[1]; ++j)
                    for (int k = 0; k < L[2]; ++k)
                        b[(static_cast<std::size_t>(i) * L[1] + j) * L[2] + k] += static_cast<float>(0.5 * std::sin(1.7 * i) + 2.0 * k / L[2] + 1.0 * (j - L[1] / 2.0) * (j - L[1] / 2.0) / (L[1] * L[1]));
            add_noise(b, rng, noise);
            const Registration r = register_blocks(a.data(), b.data(), n, m, nullptr, {}, device);
            report("3D with frame levels and a smooth profile", r.shift, d, r, r.converged && !r.at_edge);
        }
    }
    noise = 0.05;
    // 5: two dimensions (one frame)
    {
        const int n[3] = {1, 128, 256}, m[3] = {0, 8, 8};
        BlobField f(rng, 3000, 0, 1, -40, n[1] + 40, -40, n[2] + 40, 1.0, 2.0, 2.0);
        std::vector<float> a, b;
        const double d[3] = {0.0, 0.37, -0.62};
        f.render(a, 1, n[1], n[2], 0.0, 0.0, 0.0, 1.0, false);
        f.render(b, 1, n[1] + 2 * m[1], n[2] + 2 * m[2], 0.0, -m[1] - d[1], -m[2] - d[2], 1.0, false);
        add_noise(a, rng, noise);
        add_noise(b, rng, noise);
        const Registration r = register_blocks(a.data(), b.data(), n, m, nullptr, {}, device);
        report("2D fractional displacement", r.shift, d, r, r.converged && !r.at_edge);
    }
    // 6: a mirror axis. View A is f(column); the opposite view is f(2 c - column), a later frame and other rows.
    for (int crop = -1; crop < 3; ++crop) {
        const int n[3] = {8, 64, crop < 0 ? 240 : 128}, m[3] = {4, 8, 8};
        const int L[3] = {n[0] + 2 * m[0], n[1] + 2 * m[1], n[2] + 2 * m[2]};
        int a0 = 40, b0 = 302;
        if (crop >= 0) crop_column_origins(a0, b0, 240, n[2], crop * 0.5, true, 3);
        const double c = 290.37, dq = 0.133, dv = -0.2;
        BlobField f(rng, 9000, -40, n[0] + 40, -50, n[1] + 50, -60, 760, 1.5, 2.5, 2.5);
        std::vector<float> a, b, flipped;
        f.render(a, n[0], n[1], n[2], 0.0, 0.0, static_cast<double>(a0));
        // the opposite view on its own detector columns b0 - m .. b0 + n + m - 1, frames and rows displaced by (dq, dv)
        f.render(b, L[0], L[1], L[2], -m[0] - dq, -m[1] - dv, 2 * c - (b0 - m[2]), -1.0);
        flipped.resize(b.size());
        for (std::size_t row = 0; row < static_cast<std::size_t>(L[0]) * L[1]; ++row)
            for (int k = 0; k < L[2]; ++k) flipped[row * L[2] + k] = b[row * L[2] + (L[2] - 1 - k)];
        // Crops test coordinate bookkeeping on an exact field; the original full
        // window above also tests robustness to measurement noise.
        if (crop < 0) {
            add_noise(a, rng, noise);
            add_noise(flipped, rng, noise);
        }
        const Registration r = register_blocks(a.data(), flipped.data(), n, m, nullptr, {}, device);
        const double got[3] = {r.shift[0], r.shift[1], mirror_axis(a0, b0, n[2], r.shift[2])};
        const double want[3] = {dq, dv, c};
        report(crop < 0 ? "mirror axis from reversed columns" : "mirror axis from cropped corresponding rays",
               got, want, r, r.converged && !r.at_edge && !r.whole[2]);
    }
    // 7, 8: a stretch at rest against a helix, and a helix against a helix, cut and pre-aligned by a plan made
    // from a record whose heights are wrong. The registration must return the true displacement at equal angle.
    for (int kind = 0; kind < 2; ++kind) {
        const int n[3] = {16, 96, 240}, m[3] = {4, 12, 8};
        const int L[3] = {n[0] + 2 * m[0], n[1] + 2 * m[1], n[2] + 2 * m[2]};
        const double step = 0.045, travel_b = 0.1313, travel_a = kind == 0 ? 0.0 : 0.1313;
        const double height_error = 0.23, partner = kind == 0 ? 0.30 : -0.27, du = 0.4;
        const std::int64_t frames = 400, p0 = 100, J = 150;
        // true motion: A's frame p is at angle p (in frames); B's frame j is at angle j - J - partner, so the frame
        // of B at A's angle is p + J + partner
        Trajectory ta, tb_true, tb_stated;
        for (std::int64_t p = 0; p < frames; ++p) {
            ta.angle_deg.push_back(step * p);
            ta.rows.push_back(10.2 + travel_a * p);
            tb_true.angle_deg.push_back(step * (p - J - partner));
            tb_true.rows.push_back(31.7 + travel_b * p);
        }
        tb_stated = tb_true;
        for (double& v : tb_stated.rows) v -= height_error;  // the record puts B's content lower than it is
        BlockPlan plan;
        if (!plan_block(ta, tb_stated, p0, n[0], m[0], J, 0.0, plan)) {
            all = false;
            continue;
        }
        const int row_a = 40;
        BlobField f(rng, 9000, p0 - 40, p0 + n[0] + 40, -80, 260, -40, n[2] + 60, 1.5, 2.5, 2.5);
        // frame p of A on detector row v shows the sample at height v - rows_a[p]
        std::vector<double> x0(n[0]), r0(n[0]);
        for (int q = 0; q < n[0]; ++q) x0[q] = static_cast<double>(p0 + q), r0[q] = row_a - ta.rows[p0 + q];
        std::vector<float> a, b;
        f.render(a, n[0], n[1], n[2], x0, r0, 0.0);
        std::vector<double> xb(L[0]), rb(L[0]);
        for (int q = 0; q < L[0]; ++q) {
            const std::int64_t j = plan.j0 - m[0] + q;
            xb[q] = tb_true.angle_deg[j] / step;
            rb[q] = row_a + plan.row_offset - m[1] - tb_true.rows[j];
        }
        f.render(b, L[0], L[1], L[2], xb, rb, -m[2] - du);
        add_noise(a, rng, noise);
        add_noise(b, rng, noise);
        const Registration r = register_blocks(a.data(), b.data(), n, m, plan.row_shift.data(), {}, device);
        // the truth: B's content against A's at equal angle, averaged over the block
        double D_true = 0;
        for (int q = 0; q < n[0]; ++q) {
            const double j = static_cast<double>(p0 + q + J) + partner;
            D_true += (31.7 + travel_b * j) - ta.rows[p0 + q];
        }
        D_true /= n[0];
        const double got[3] = {r.shift[0], measured_D(plan, r.shift[0], r.shift[1]), r.shift[2]};
        const double want[3] = {partner, D_true, du};
        report(kind == 0 ? "rest against helix, stated heights off by 0.23" : "helix against helix, stated heights off by 0.23", got, want, r,
               r.converged && !r.at_edge && std::fabs(plan.stated_dq - partner) < 1e-6 && std::fabs(plan.stated_D - (D_true - height_error)) < 1e-6);
    }
    if (verbose) std::printf("registration self test on %s: %s (tolerance %.2f samples)\n", device_name(device), all ? "passed" : "FAILED", tolerance);
    return all;
}

}  // namespace helirec::calibration
