#include "helirec/calib.hpp"

#include <fftw3.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <mutex>

#include "helirec/fourier.hpp"

namespace helirec::calib {

namespace {

inline long reflect_index(long i, long n) {  // scipy 'reflect' (half-sample symmetric)
    if (n == 1) return 0;
    const long period = 2 * n;
    long r = i % period;
    if (r < 0) r += period;
    return r < n ? r : period - 1 - r;
}

}  // namespace

template <class T>
void gaussian_filter1d(const T* in, T* out, std::size_t n0, std::size_t n1, std::size_t n2, int axis, double sigma,
                       const std::string& mode, double truncate) {
    if (axis < 0 || axis > 2) throw Error("gaussian_filter1d: axis must be 0, 1 or 2");
    const bool nearest = mode == "nearest";
    if (!nearest && mode != "reflect") throw Error("gaussian_filter1d: mode must be 'reflect' or 'nearest'");
    const long r = static_cast<long>(truncate * sigma + 0.5);
    std::vector<double> k(2 * r + 1);
    double s = 0;
    for (long i = -r; i <= r; ++i) s += (k[i + r] = std::exp(-0.5 / (sigma * sigma) * double(i * i)));
    for (auto& v : k) v /= s;
    const std::size_t dims[3] = {n0, n1, n2};
    const long n = static_cast<long>(dims[axis]);
    const std::size_t stride = axis == 2 ? 1 : (axis == 1 ? n2 : n1 * n2);
    const std::size_t outer = axis == 0 ? 1 : n0, mid = axis == 1 ? 1 : (axis == 0 ? n1 * n2 : n1);
    const std::size_t inner = axis == 2 ? 1 : (axis == 1 ? n2 : 1);
    // iterate over all lines along `axis`
    const std::size_t n_lines = (n0 * n1 * n2) / static_cast<std::size_t>(n);
    (void)outer; (void)mid; (void)inner;
#pragma omp parallel
    {
        std::vector<double> line(n);
#pragma omp for schedule(static)
        for (std::ptrdiff_t li = 0; li < static_cast<std::ptrdiff_t>(n_lines); ++li) {
            // base offset of line li
            std::size_t base;
            if (axis == 2) base = static_cast<std::size_t>(li) * n2;
            else if (axis == 1) base = (static_cast<std::size_t>(li) / n2) * n1 * n2 + static_cast<std::size_t>(li) % n2;
            else base = static_cast<std::size_t>(li);
            for (long i = 0; i < n; ++i) line[i] = static_cast<double>(in[base + i * stride]);
            for (long i = 0; i < n; ++i) {
                double acc = 0;
                for (long j = -r; j <= r; ++j) {
                    long q = i + j;
                    q = nearest ? std::min(n - 1, std::max(0L, q)) : reflect_index(q, n);
                    acc += line[q] * k[j + r];
                }
                out[base + i * stride] = static_cast<T>(acc);
            }
        }
    }
}

template void gaussian_filter1d<float>(const float*, float*, std::size_t, std::size_t, std::size_t, int, double,
                                       const std::string&, double);
template void gaussian_filter1d<double>(const double*, double*, std::size_t, std::size_t, std::size_t, int, double,
                                        const std::string&, double);

std::vector<float> apodisation_mask(int Lx, float w_left, float w_right, float min_val, int sh_left, int sh_right) {
    std::vector<float> h(Lx);
    const float shl = static_cast<float>(sh_left), shr = static_cast<float>(sh_right);
    for (int x = 0; x < Lx; ++x) {
        float l = 1.0f, r = 1.0f;
        if (w_left > 0.0f) {
            const float m = w_left / 2.0f + shl;
            const float w = std::max(1.0f, w_left / 5.0f);
            const float d = static_cast<float>(x) - m;
            l = 0.5f * (1.0f + std::erf(d / w));
        }
        if (w_right > 0.0f) {
            const float m = w_right / 2.0f + shr;
            const float w = std::max(1.0f, w_right / 5.0f);
            const float d = static_cast<float>(x) - (static_cast<float>(Lx) - m);
            r = 0.5f * (1.0f - std::erf(d / w));
        }
        float v = l * r;
        if (min_val > 0.0f && v < min_val) v = min_val;
        h[x] = v;
    }
    return h;
}

std::vector<float> weight_border_v(int L, double tw, int crop_top, int crop_bottom) {
    const int m = static_cast<int>(std::nearbyint(tw / 2.0));  // python round: half to even
    const float w = static_cast<float>(tw / 5.0);
    std::vector<float> b(L);
    for (int z = 0; z < L; ++z) {
        const float ar = static_cast<float>(z - L + m + crop_bottom) / w;
        const float al = static_cast<float>(z - m - crop_top) / w;
        const float er = static_cast<float>(std::erf(static_cast<double>(ar)));
        const float el = static_cast<float>(std::erf(static_cast<double>(al)));
        const float rr = (1.0f - er) / 2.0f, rl = (1.0f + el) / 2.0f;
        b[z] = rr * rl;
    }
    return b;
}

// ---------------------------------------------------------------- detector blur from the noise of the flats
void tile_power_spectra(const float* img, std::size_t n, std::size_t rows, std::size_t cols, int tile, const int* ys,
                        int ny, const int* xs, int nx, double* power) {
    if (tile < 8 || tile % 2) throw Error("tile_power_spectra: an even tile of at least 8 pixels is needed");
    for (int i = 0; i < ny; ++i)
        if (ys[i] < 0 || static_cast<std::size_t>(ys[i]) + tile > rows) throw Error("tile_power_spectra: tile outside the rows");
    for (int i = 0; i < nx; ++i)
        if (xs[i] < 0 || static_cast<std::size_t>(xs[i]) + tile > cols) throw Error("tile_power_spectra: tile outside the columns");
    const int nc = tile / 2 + 1;
    const std::size_t per = static_cast<std::size_t>(tile) * nc;
    std::vector<float> win(tile);
    for (int i = 0; i < tile; ++i) win[i] = static_cast<float>(0.5 - 0.5 * std::cos(2.0 * M_PI * i / (tile - 1)));
    std::mutex& m = fftw_planner_mutex();
    fftwf_plan plan;
    {
        std::vector<float> a(static_cast<std::size_t>(tile) * tile);
        std::vector<std::complex<float>> A(per);
        std::lock_guard<std::mutex> lock(m);
        plan = fftwf_plan_dft_r2c_2d(tile, tile, a.data(), reinterpret_cast<fftwf_complex*>(A.data()),
                                     FFTW_ESTIMATE | FFTW_UNALIGNED);
    }
#pragma omp parallel
    {
        std::vector<float> a(static_cast<std::size_t>(tile) * tile);
        std::vector<std::complex<float>> A(per);
#pragma omp for schedule(dynamic, 1)
        for (int t = 0; t < ny * nx; ++t) {
            const int y0 = ys[t / nx], x0 = xs[t % nx];
            double* out = power + static_cast<std::size_t>(t) * per;
            for (std::size_t k = 0; k < n; ++k) {
                const float* f = img + k * rows * cols;
                for (int i = 0; i < tile; ++i) {
                    const float* src = f + (static_cast<std::size_t>(y0) + i) * cols + x0;
                    float* dst = a.data() + static_cast<std::size_t>(i) * tile;
                    for (int j = 0; j < tile; ++j) dst[j] = src[j] * win[i] * win[j];
                }
                fftwf_execute_dft_r2c(plan, a.data(), reinterpret_cast<fftwf_complex*>(A.data()));
                for (std::size_t q = 0; q < per; ++q) out[q] += std::norm(std::complex<double>(A[q]));
            }
        }
    }
    std::lock_guard<std::mutex> lock(m);
    fftwf_destroy_plan(plan);
}

namespace {

inline double sinc_pi(double a) {
    const double pa = M_PI * a;
    return std::fabs(pa) < 1e-9 ? 1.0 : std::sin(pa) / pa;
}

// the normalised noise spectra of the tiles and the model of detector_blur_field_fit
struct BlurFieldProblem {
    int tiles = 0, bins = 0;
    std::vector<double> Y;                 // (tiles, bins)
    std::vector<double> X, Yr;             // tile centres (units of 1000 px from the detector's centre)
    std::vector<double> FX, FY, wgt;       // per bin
    std::vector<char> low;
    mutable std::vector<double> M, num, den;

    void residual(const double* th, std::vector<double>& r) const {
        M.resize(static_cast<std::size_t>(tiles) * bins);
        num.assign(bins, 0.0);
        den.assign(bins, 0.0);
        for (int t = 0; t < tiles; ++t) {
            const double sx = th[0] * X[t] + th[1] * Yr[t] + th[4], sy = th[2] * X[t] + th[3] * Yr[t] + th[5];
            double* mt = &M[static_cast<std::size_t>(t) * bins];
            double lo = 0;
            int nlo = 0;
            for (int b = 0; b < bins; ++b) {
                const double s = sinc_pi(sx * FX[b] + sy * FY[b]);
                mt[b] = s * s;
                if (low[b]) { lo += mt[b]; ++nlo; }
            }
            lo /= nlo;
            const double* yt = &Y[static_cast<std::size_t>(t) * bins];
            for (int b = 0; b < bins; ++b) {
                mt[b] /= lo;
                num[b] += yt[b] * mt[b];
                den[b] += mt[b] * mt[b];
            }
        }
        r.resize(M.size());
        for (int t = 0; t < tiles; ++t)
            for (int b = 0; b < bins; ++b) {
                const std::size_t i = static_cast<std::size_t>(t) * bins + b;
                r[i] = (Y[i] - M[i] * num[b] / std::max(den[b], 1e-30)) * wgt[b];
            }
    }
};

inline double sum_squares(const std::vector<double>& r) {
    double s = 0;
    for (double v : r) s += v * v;
    return s;
}

// a x = b for a 6 x 6 system (partial pivoting); false: singular
bool solve6(std::array<std::array<double, 7>, 6> a, double* x) {
    for (int c = 0; c < 6; ++c) {
        int p = c;
        for (int r = c + 1; r < 6; ++r)
            if (std::fabs(a[r][c]) > std::fabs(a[p][c])) p = r;
        if (std::fabs(a[p][c]) < 1e-300) return false;
        std::swap(a[c], a[p]);
        for (int r = c + 1; r < 6; ++r) {
            const double f = a[r][c] / a[c][c];
            for (int k = c; k < 7; ++k) a[r][k] -= f * a[c][k];
        }
    }
    for (int c = 5; c >= 0; --c) {
        double v = a[c][6];
        for (int k = c + 1; k < 6; ++k) v -= a[c][k] * x[k];
        x[c] = v / a[c][c];
    }
    return true;
}

// Levenberg-Marquardt from `th` (forward-difference Jacobian); returns the sum of squares reached
double blur_levenberg(const BlurFieldProblem& p, std::array<double, 6>& th) {
    std::vector<double> r, r2, J;
    p.residual(th.data(), r);
    double cost = sum_squares(r), lam = 1e-2;
    const std::size_t n = r.size();
    const double h = 1e-4;
    for (int it = 0; it < 60; ++it) {
        J.resize(n * 6);
        for (int k = 0; k < 6; ++k) {
            auto t2 = th;
            t2[k] += h;
            p.residual(t2.data(), r2);
            for (std::size_t i = 0; i < n; ++i) J[i * 6 + k] = (r2[i] - r[i]) / h;
        }
        double g[6] = {0, 0, 0, 0, 0, 0}, H[6][6] = {};
        for (std::size_t i = 0; i < n; ++i) {
            const double* ji = &J[i * 6];
            for (int a = 0; a < 6; ++a) {
                g[a] += ji[a] * r[i];
                for (int b = a; b < 6; ++b) H[a][b] += ji[a] * ji[b];
            }
        }
        for (int a = 0; a < 6; ++a)
            for (int b = 0; b < a; ++b) H[a][b] = H[b][a];
        bool better = false;
        std::array<double, 6> cand{};
        double c2 = cost;
        for (int tries = 0; tries < 12 && !better; ++tries) {
            std::array<std::array<double, 7>, 6> sys;
            for (int a = 0; a < 6; ++a) {
                for (int b = 0; b < 6; ++b) sys[a][b] = H[a][b];
                sys[a][a] += lam * (H[a][a] + 1e-12);
                sys[a][6] = -g[a];
            }
            double step[6];
            if (solve6(sys, step)) {
                for (int a = 0; a < 6; ++a) cand[a] = th[a] + step[a];
                p.residual(cand.data(), r2);
                c2 = sum_squares(r2);
                better = c2 < cost;
            }
            if (!better) lam *= 4;
        }
        if (!better) break;
        const bool done = cost - c2 < 1e-9 * cost;
        th = cand;
        r.swap(r2);
        cost = c2;
        lam /= 3;
        if (done) break;
    }
    return cost;
}

}  // namespace

DetectorBlurField detector_blur_field_fit(const double* power, int ny, int nx, int tile, const int* ys, const int* xs,
                           std::size_t rows, std::size_t cols, double f_max) {
    if (ny < 1 || nx < 1 || ny * nx < 6) throw Error("detector_blur_field_fit: at least 6 tiles are needed");
    if (tile < 32 || tile % 2) throw Error("detector_blur_field_fit: an even tile of at least 32 pixels is needed");
    if (!(f_max > 0.15 && f_max <= 0.42)) throw Error("detector_blur_field_fit: f_max must be in (0.15, 0.42] cycles per pixel");
    const int nc = tile / 2 + 1, nbx = 8, nby = 16, nb = nbx * nby;
    const std::size_t per = static_cast<std::size_t>(tile) * nc;
    // frequency bins of the half plane fx > 0 (and fx = 0, fy > 0), and the frequencies of the white floor
    std::vector<int> bin_of(per, -1);
    std::vector<char> floor_at(per, 0);
    std::vector<double> nin(nb, 0.0), bfx(nb, 0.0), bfy(nb, 0.0);
    for (int i = 0; i < tile; ++i) {
        const double fy = static_cast<double>(i < tile / 2 ? i : i - tile) / tile;
        for (int j = 0; j < nc; ++j) {
            const double fx = static_cast<double>(j) / tile;
            const std::size_t q = static_cast<std::size_t>(i) * nc + j;
            floor_at[q] = fx > 0.42 && std::fabs(fy) > 0.42;
            if (!((j > 0 && j < tile / 2) || (j == 0 && fy > 0))) continue;
            const int b = std::min(static_cast<int>(fx * 16), nbx - 1) * nby + std::min(static_cast<int>((fy + 0.5) * 16), nby - 1);
            bin_of[q] = b;
            nin[b] += 1;
            bfx[b] += fx;
            bfy[b] += fy;
        }
    }
    BlurFieldProblem p;
    std::vector<int> sel;
    for (int b = 0; b < nb; ++b) {
        if (nin[b] == 0) continue;
        bfx[b] /= nin[b];
        bfy[b] /= nin[b];
        const double fr = std::hypot(bfx[b], bfy[b]);
        if (!(fr > 0.05 && fr < f_max)) continue;
        sel.push_back(b);
        p.FX.push_back(bfx[b]);
        p.FY.push_back(bfy[b]);
        p.wgt.push_back(std::sqrt(nin[b]));
        p.low.push_back(fr < 0.11);
    }
    p.bins = static_cast<int>(sel.size());
    p.tiles = ny * nx;
    if (std::count(p.low.begin(), p.low.end(), 1) == 0) throw Error("detector_blur_field_fit: no frequency bins below 0.11");
    p.Y.resize(static_cast<std::size_t>(p.tiles) * p.bins);
    std::vector<double> binned(nb);
    for (int t = 0; t < p.tiles; ++t) {
        const double* s = power + static_cast<std::size_t>(t) * per;
        double fl = 0;
        std::size_t nfl = 0;
        for (std::size_t q = 0; q < per; ++q)
            if (floor_at[q]) { fl += s[q]; ++nfl; }
        fl /= nfl;
        std::fill(binned.begin(), binned.end(), 0.0);
        for (std::size_t q = 0; q < per; ++q)
            if (bin_of[q] >= 0) binned[bin_of[q]] += s[q] - fl;
        double lo = 0;
        int nlo = 0;
        for (int k = 0; k < p.bins; ++k)
            if (p.low[k]) { lo += binned[sel[k]] / nin[sel[k]]; ++nlo; }
        lo /= nlo;
        if (!(lo > 0)) throw Error("detector_blur_field_fit: a tile has no noise above its floor at low frequencies");
        for (int k = 0; k < p.bins; ++k) p.Y[static_cast<std::size_t>(t) * p.bins + k] = binned[sel[k]] / nin[sel[k]] / lo;
        p.X.push_back((xs[t % nx] + 0.5 * tile - 0.5 * (static_cast<double>(cols) - 1)) / 1000.0);
        p.Yr.push_back((ys[t / nx] + 0.5 * tile - 0.5 * (static_cast<double>(rows) - 1)) / 1000.0);
    }
    DetectorBlurField out;
    std::vector<double> r;
    const double none[6] = {1e-9, 0, 0, 1e-9, 0, 0};
    p.residual(none, r);
    out.rms_none = std::sqrt(sum_squares(r) / r.size());
    const std::array<double, 6> starts[3] = {{2.0, 0, 0, 1.0, 0, 0}, {1.0, 0, 0, 1.0, 0.3, 0}, {3.0, 0.3, 0, 0.5, -0.3, 0.2}};
    double best = std::numeric_limits<double>::infinity();
    for (auto th : starts) {
        const double c = blur_levenberg(p, th);
        if (c < best) {
            best = c;
            const double sign = th[0] >= 0 ? 1.0 : -1.0;
            for (int k = 0; k < 6; ++k) out.field[k] = sign * th[k];
        }
    }
    out.rms_fit = std::sqrt(best / r.size());
    return out;
}

}  // namespace helirec::calib
