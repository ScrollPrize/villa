// CPU block cross-spectra (FFTW, multithreaded 3D transforms) and affine resampling.
#include <fftw3.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <mutex>
#include <vector>

#include "common/block_spectra_core.hpp"
#include "helirec/block_spectra.hpp"
#include "helirec/fourier.hpp"

namespace helirec::cpu {

namespace core = block_spectra_core;

namespace {

class FftwBlockCorrelator final : public BlockCorrelator {
  public:
    explicit FftwBlockCorrelator(const BlockSpectraSpec& s) : BlockCorrelator(s) {
        n_ = spec_.n;
        m_ = spec_.n * spec_.oversample;
        nc_ = n_ / 2 + 1;
        wa_ = block_spectra::hann(n_, 1);
        wb_ = block_spectra::hann(n_, spec_.oversample);
        tan2_ = block_spectra::theta_tan2(spec_);
        double s2 = 0;
        for (float w : wa_) s2 += static_cast<double>(w) * w;
        norm_ = 1.0 / (std::pow(static_cast<double>(n_), 3) * s2 * s2 * s2);
        real_ = fftwf_alloc_real(static_cast<std::size_t>(m_) * m_ * m_);
        A_ = fftwf_alloc_complex(static_cast<std::size_t>(n_) * n_ * nc_);
        B_ = fftwf_alloc_complex(static_cast<std::size_t>(m_) * m_ * (m_ / 2 + 1));
        if (!real_ || !A_ || !B_) throw Error("fftw allocation failed");
        std::lock_guard<std::mutex> lock(fftw_planner_mutex());
        static const bool threads = fftwf_init_threads() != 0;
        if (threads) fftwf_plan_with_nthreads(omp_get_max_threads());
        fwd_a_ = fftwf_plan_dft_r2c_3d(n_, n_, n_, real_, A_, FFTW_ESTIMATE);
        fwd_b_ = fftwf_plan_dft_r2c_3d(m_, m_, m_, real_, B_, FFTW_ESTIMATE);
        if (threads) fftwf_plan_with_nthreads(1);   // (every other plan of the library is single-threaded)
        if (!fwd_a_ || !fwd_b_) throw Error("fftw planning failed");
    }
    ~FftwBlockCorrelator() override {
        {
            std::lock_guard<std::mutex> lock(fftw_planner_mutex());
            if (fwd_a_) fftwf_destroy_plan(fwd_a_);
            if (fwd_b_) fftwf_destroy_plan(fwd_b_);
        }
        fftwf_free(real_);
        fftwf_free(A_);
        fftwf_free(B_);
    }

    void compute(const float* a, const float* b, BlockSpectra& out, const double* shift) override {
        window(a, n_, wa_, out.mean_a, out.var_a);
        fftwf_execute_dft_r2c(fwd_a_, real_, A_);
        window(b, m_, wb_, out.mean_b, out.var_b);
        fftwf_execute_dft_r2c(fwd_b_, real_, B_);
        crop();
        out.shift = {0, 0, 0};
        out.peak = 0;
        if (shift) out.shift = {shift[0], shift[1], shift[2]};
        table(out);
        if (spec_.align && !shift) {
            const int kb = block_spectra::align_box(spec_), side = 2 * kb + 1, nx = kb + 1;
            std::vector<float> box(static_cast<std::size_t>(side) * side * nx * 2);
#pragma omp parallel for schedule(static)
            for (int u = 0; u < side; ++u)
                for (int v = 0; v < side; ++v) {
                    const int iz = (u - kb + n_) % n_, iy = (v - kb + n_) % n_;
                    const fftwf_complex* pa = A_ + (static_cast<std::size_t>(iz) * n_ + iy) * nc_;
                    const fftwf_complex* pb = B_ + (static_cast<std::size_t>(iz) * n_ + iy) * nc_;
                    float* o = box.data() + (static_cast<std::size_t>(u) * side + v) * nx * 2;
                    for (int c = 0; c < nx; ++c) {   // A conj B, normalised as the tables
                        o[2 * c] = static_cast<float>(norm_ * (static_cast<double>(pa[c][0]) * pb[c][0] + static_cast<double>(pa[c][1]) * pb[c][1]));
                        o[2 * c + 1] = static_cast<float>(norm_ * (static_cast<double>(pa[c][1]) * pb[c][0] - static_cast<double>(pa[c][0]) * pb[c][1]));
                    }
                }
            block_spectra::align(spec_, box.data(), out.aa, out.bb, out.shift, out.peak);
            table(out);
        }
    }

  private:
    // real_ = (v - weighted mean) x window, for a block of edge e
    void window(const float* v, int e, const std::vector<float>& w, double& mean, double& var) {
        std::vector<double> s0(e, 0.0), s1(e, 0.0), s2(e, 0.0);
#pragma omp parallel for schedule(static)
        for (int z = 0; z < e; ++z) {
            double a0 = 0, a1 = 0, a2 = 0;
            for (int y = 0; y < e; ++y) {
                const float* row = v + (static_cast<std::size_t>(z) * e + y) * e;
                double r1 = 0, r2 = 0;
                for (int x = 0; x < e; ++x) {
                    const double t = row[x];
                    r1 += w[x] * t;
                    r2 += w[x] * t * t;
                }
                a1 += w[y] * r1;
                a2 += w[y] * r2;
                a0 += w[y];
            }
            s0[z] = w[z] * a0; s1[z] = w[z] * a1; s2[z] = w[z] * a2;
        }
        double t0 = 0, t1 = 0, t2 = 0, wx = 0;
        for (int x = 0; x < e; ++x) wx += w[x];
        for (int z = 0; z < e; ++z) { t0 += s0[z]; t1 += s1[z]; t2 += s2[z]; }
        t0 *= wx;
        mean = t1 / t0;
        var = t2 / t0 - mean * mean;
        const float mu = static_cast<float>(mean);
#pragma omp parallel for schedule(static)
        for (int z = 0; z < e; ++z)
            for (int y = 0; y < e; ++y) {
                const float* row = v + (static_cast<std::size_t>(z) * e + y) * e;
                float* o = real_ + (static_cast<std::size_t>(z) * e + y) * e;
                const float wzy = w[z] * w[y];
                for (int x = 0; x < e; ++x) o[x] = (row[x] - mu) * (wzy * w[x]);
            }
    }

    // B_: the frequencies of a's grid, in a's layout (n, n, n / 2 + 1), scaled to a's sampling density
    void crop() {
        const int f = spec_.oversample;
        if (f == 1) return;
        const float scale = 1.f / static_cast<float>(f * f * f);
        const std::size_t mc = static_cast<std::size_t>(m_ / 2 + 1);
        // in place: every destination element lies at or before its source, rows in increasing order
        for (int iz = 0; iz < n_; ++iz) {
            const int kz = core::signed_index(iz, n_), jz = kz < 0 ? kz + m_ : kz;
            for (int iy = 0; iy < n_; ++iy) {
                const int ky = core::signed_index(iy, n_), jy = ky < 0 ? ky + m_ : ky;
                const fftwf_complex* src = B_ + (static_cast<std::size_t>(jz) * m_ + jy) * mc;
                fftwf_complex* dst = B_ + (static_cast<std::size_t>(iz) * n_ + iy) * nc_;
                for (std::size_t c = 0; c < nc_; ++c) {
                    const float re = src[c][0] * scale, im = src[c][1] * scale;
                    dst[c][0] = re;
                    dst[c][1] = im;
                }
            }
        }
    }

    void table(BlockSpectra& out) const {
        const int nb = spec_.n_bins(), T = spec_.thetas;
        const double scale = 2.0 * spec_.shells / n_, w0 = 2 * M_PI / n_;
        const double dz = out.shift[0], dy = out.shift[1], dx = out.shift[2];
        const bool shifted = dz != 0 || dy != 0 || dx != 0;
        std::vector<double> part(static_cast<std::size_t>(n_) * nb * 3, 0.0);
#pragma omp parallel for schedule(dynamic, 4)
        for (int iz = 0; iz < n_; ++iz) {
            double* p = part.data() + static_cast<std::size_t>(iz) * nb * 3;
            const int kz = core::signed_index(iz, n_);
            for (int iy = 0; iy < n_; ++iy) {
                const int ky = core::signed_index(iy, n_);
                const fftwf_complex* pa = A_ + (static_cast<std::size_t>(iz) * n_ + iy) * nc_;
                const fftwf_complex* pb = B_ + (static_cast<std::size_t>(iz) * n_ + iy) * nc_;
                for (int c = 0; c < static_cast<int>(nc_); ++c) {
                    const double w = core::half_weight(kz, ky, c, n_);
                    if (w == 0) continue;
                    const double ar = pa[c][0], ai = pa[c][1], br = pb[c][0], bi = pb[c][1];
                    double re = ar * br + ai * bi;   // Re, Im of A conj B
                    if (shifted) {                   // b's content displaced by the shift: x exp(-2 pi i k.d / n)
                        const double im = ai * br - ar * bi, phi = -w0 * (kz * dz + ky * dy + c * dx);
                        re = re * std::cos(phi) - im * std::sin(phi);
                    }
                    double* q = p + static_cast<std::size_t>(core::bin_of(kz, ky, c, scale, tan2_.data(), T)) * 3;
                    q[0] += w * (ar * ar + ai * ai);
                    q[1] += w * (br * br + bi * bi);
                    q[2] += w * re;
                }
            }
        }
        out.aa.assign(nb, 0.0);
        out.bb.assign(nb, 0.0);
        out.ab.assign(nb, 0.0);
        for (int iz = 0; iz < n_; ++iz) {   // in a fixed order
            const double* p = part.data() + static_cast<std::size_t>(iz) * nb * 3;
            for (int i = 0; i < nb; ++i) {
                out.aa[i] += p[3 * i];
                out.bb[i] += p[3 * i + 1];
                out.ab[i] += p[3 * i + 2];
            }
        }
        for (int i = 0; i < nb; ++i) {
            out.aa[i] *= norm_;
            out.bb[i] *= norm_;
            out.ab[i] *= norm_;
        }
    }

    int n_ = 0, m_ = 0;
    std::size_t nc_ = 0;
    double norm_ = 1;
    std::vector<float> wa_, wb_;
    std::vector<double> tan2_;
    float* real_ = nullptr;
    fftwf_complex *A_ = nullptr, *B_ = nullptr;
    fftwf_plan fwd_a_ = nullptr, fwd_b_ = nullptr;
};

}  // namespace

std::unique_ptr<BlockCorrelator> make_block_correlator(const BlockSpectraSpec& spec) {
    return std::make_unique<FftwBlockCorrelator>(spec);
}

void resample_affine(const float* src, int sz, int sy, int sx, const double* m, float* out, int nz, int ny, int nx,
                     int interp) {
#pragma omp parallel for collapse(2) schedule(static)
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j) {
            float* o = out + (static_cast<std::size_t>(k) * ny + j) * nx;
            const double bz = m[0] * k + m[1] * j + m[3], by = m[4] * k + m[5] * j + m[7], bx = m[8] * k + m[9] * j + m[11];
            for (int i = 0; i < nx; ++i)
                o[i] = core::sample_volume(src, sz, sy, sx, bz + m[2] * i, by + m[6] * i, bx + m[10] * i, interp);
        }
}

}  // namespace helirec::cpu
