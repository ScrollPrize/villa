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

// ------------------------------------------------------------------ merits
namespace {

void gauss_clamped(const std::vector<float>& in, std::vector<float>& out, float sigma) {
    const int r = static_cast<int>(std::ceil(3 * sigma));
    std::vector<float> k(2 * r + 1);
    float s = 0.f;
    for (int i = -r; i <= r; ++i) {
        k[i + r] = std::exp(-0.5f * (i * i) / (sigma * sigma));
        s += k[i + r];
    }
    for (auto& v : k) v /= s;
    const int n = static_cast<int>(in.size());
    for (int i = 0; i < n; ++i) {
        float acc = 0.f;
        for (int j = -r; j <= r; ++j) acc += in[std::min(n - 1, std::max(0, i + j))] * k[j + r];
        out[i] = acc;
    }
}

inline float lerp(float a, float b, float t) { return a * (1.0f - t) + b * t; }

}  // namespace

void build_all_merits(const float* sig, const float* w, std::size_t L, std::size_t W, int ovl_min, int ovl_max, int ovs,
                      float sigma_wide, float sigma_narrow, float* merits, float* energies) {
    if (ovl_min > ovl_max) throw Error("build_all_merits: ovl_min > ovl_max");
    const int steps = (ovl_max - ovl_min) * ovs + 1;
#pragma omp parallel
    {
        std::vector<float> s0(W), s1(W), a(W), b(W), band0(W), band1(W), norm0(W), norm1(W), absv(W);
#pragma omp for schedule(dynamic)
        for (std::ptrdiff_t l = 0; l < static_cast<std::ptrdiff_t>(L); ++l) {
            for (int c = 0; c < 2; ++c) {
                const float* src = sig + (c * L + l) * W;
                std::vector<float>& sv = c == 0 ? s0 : s1;
                sv.assign(src, src + W);
                gauss_clamped(sv, a, sigma_wide);
                gauss_clamped(sv, b, sigma_narrow);
                std::vector<float>& band = c == 0 ? band0 : band1;
                for (std::size_t j = 0; j < W; ++j) {
                    band[j] = a[j] - b[j];
                    absv[j] = std::abs(sv[j]);
                }
                gauss_clamped(absv, c == 0 ? norm0 : norm1, sigma_narrow);
            }
            const float* w0 = w + (0 * L + l) * W;
            const float* w1 = w + (1 * L + l) * W;
            float* mrow = merits + static_cast<std::size_t>(l) * steps;
            for (int st = 0; st < steps; ++st) {
                const float overlap = ovl_min + st * (1.0f / static_cast<float>(ovs));
                double sd = 0.0, sw = 0.0;
                for (std::size_t ix = 0; ix < W; ++ix) {
                    float p = static_cast<float>(ix) + (overlap - W);
                    p = W - 1 - p;
                    if (p < 0.0f || p >= static_cast<float>(W - 1)) continue;
                    const int pi = static_cast<int>(std::floor(p));
                    const float fr = p - pi;
                    if (pi < 0 || pi + 1 >= static_cast<int>(W)) continue;
                    const float v1 = lerp(band1[pi], band1[pi + 1], fr);
                    const float ww = w0[ix] * lerp(w1[pi], w1[pi + 1], fr);
                    const float den = norm0[ix] + lerp(norm1[pi], norm1[pi + 1], fr);
                    float d = 0.f;
                    if (den != 0.0f) d = std::abs(band0[ix] - v1) / den;
                    sd += d * ww;
                    sw += ww;
                }
                mrow[st] = sw > 0.0 ? static_cast<float>(sd / sw) : std::numeric_limits<float>::infinity();
                energies[static_cast<std::size_t>(l) * steps + st] = static_cast<float>(sw);
            }
            // replace +inf by the nearest finite value (left preferred on ties)
            std::vector<float> copy(mrow, mrow + steps);
            for (int st = 0; st < steps; ++st) {
                if (std::isfinite(copy[st])) continue;
                int best = -1, bd = 1 << 30;
                for (int q = 0; q < steps; ++q)
                    if (std::isfinite(copy[q]) && std::abs(q - st) < bd) { bd = std::abs(q - st); best = q; }
                mrow[st] = best >= 0 ? copy[best] : 1e30f;
            }
        }
    }
}

// ------------------------------------------------------------------ correlator
double loss_integer(const double* img0, const double* w0, const double* img1, const double* w1, std::size_t n,
                    std::size_t Y_, std::size_t X_, int dy, int dx) {
    const long Y = static_cast<long>(Y_), X = static_cast<long>(X_);
    const long y0s = std::max(0, dy) + 1, y0e = std::min(Y, Y + dy) - 1, x0s = std::max(0, dx) + 1,
               x0e = std::min(X, X + dx) - 1;
    if (y0e <= y0s || x0e <= x0s) return std::numeric_limits<double>::infinity();
    double num = 0, den = 0;
    for (std::size_t k = 0; k < n; ++k)
        for (long y = y0s; y < y0e; ++y) {
            const std::size_t r0 = (k * Y_ + y) * X_, r1 = (k * Y_ + (y - dy)) * X_;
            for (long x = x0s; x < x0e; ++x) {
                const double ww = w0[r0 + x] * w1[r1 + x - dx];
                num += std::abs(img0[r0 + x] - img1[r1 + x - dx]) * ww;
                den += ww;
            }
        }
    return den > 0 ? num / den : std::numeric_limits<double>::infinity();
}

double loss_subpixel(const double* img0, const double* w0, const double* img1, const double* w1, std::size_t n,
                     std::size_t Y_, std::size_t X_, double dyF, double dxF) {
    const long Y = static_cast<long>(Y_), X = static_cast<long>(X_);
    const long ymin = std::max<long>(static_cast<long>(std::ceil(1 + dyF)), 0),
               ymax = std::min<long>(static_cast<long>(std::floor(Y - 2 + dyF)), Y - 1);
    const long xmin = std::max<long>(static_cast<long>(std::ceil(1 + dxF)), 0),
               xmax = std::min<long>(static_cast<long>(std::floor(X - 2 + dxF)), X - 1);
    if (ymax < ymin || xmax < xmin) return std::numeric_limits<double>::infinity();
    std::vector<long> x0(xmax - xmin + 1), x1(x0.size());
    std::vector<double> fx(x0.size());
    for (long x = xmin; x <= xmax; ++x) {
        const double xf = std::min<double>(std::max<double>(x - dxF, 0.0), X - 1);
        x0[x - xmin] = static_cast<long>(xf);
        x1[x - xmin] = x0[x - xmin] == X - 1 ? x0[x - xmin] : x0[x - xmin] + 1;
        fx[x - xmin] = xf - x0[x - xmin];
    }
    double num = 0, den = 0;
    for (std::size_t k = 0; k < n; ++k)
        for (long y = ymin; y <= ymax; ++y) {
            const double yf = std::min<double>(std::max<double>(y - dyF, 0.0), Y - 1);
            const long ya = static_cast<long>(yf), yb = ya == Y - 1 ? ya : ya + 1;
            const double fy = yf - ya;
            const double* a0 = img1 + (k * Y_ + ya) * X_;
            const double* a1 = img1 + (k * Y_ + yb) * X_;
            const double* b0 = w1 + (k * Y_ + ya) * X_;
            const double* b1 = w1 + (k * Y_ + yb) * X_;
            const std::size_t r0 = (k * Y_ + y) * X_;
            for (long x = xmin; x <= xmax; ++x) {
                const long i = x - xmin;
                const double c0 = a0[x0[i]] + (a0[x1[i]] - a0[x0[i]]) * fx[i];
                const double c1 = a1[x0[i]] + (a1[x1[i]] - a1[x0[i]]) * fx[i];
                const double v1 = c0 + (c1 - c0) * fy;
                const double d0 = b0[x0[i]] + (b0[x1[i]] - b0[x0[i]]) * fx[i];
                const double d1 = b1[x0[i]] + (b1[x1[i]] - b1[x0[i]]) * fx[i];
                const double ww1 = d0 + (d1 - d0) * fy;
                const double ww = w0[r0 + x] * ww1;
                num += std::abs(img0[r0 + x] - v1) * ww;
                den += ww;
            }
        }
    return den > 0 ? num / den : std::numeric_limits<double>::infinity();
}

OffsetResult find_offset(const double* img0, const double* w0, const double* img1, const double* w1, std::size_t n,
                         std::size_t Y, std::size_t X, int init_dy, int init_dx, int ry, int rx, double sub_range,
                         double sub_step) {
    OffsetResult r;
    const int ny = 2 * ry + 1, nx = 2 * rx + 1;
    r.loss_map.resize(static_cast<std::size_t>(ny) * nx);
#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int a = 0; a < ny; ++a)
        for (int b = 0; b < nx; ++b)
            r.loss_map[static_cast<std::size_t>(a) * nx + b] =
                loss_integer(img0, w0, img1, w1, n, Y, X, init_dy - ry + a, init_dx - rx + b);
    r.best_loss = std::numeric_limits<double>::infinity();
    for (int a = 0; a < ny; ++a)
        for (int b = 0; b < nx; ++b)
            if (r.loss_map[static_cast<std::size_t>(a) * nx + b] < r.best_loss) {
                r.best_loss = r.loss_map[static_cast<std::size_t>(a) * nx + b];
                r.best_dy = init_dy - ry + a;
                r.best_dx = init_dx - rx + b;
            }
    const int ns = static_cast<int>(std::lround(2 * sub_range / sub_step)) + 1;
    std::vector<double> sub(static_cast<std::size_t>(ns) * ns);
#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int a = 0; a < ns; ++a)
        for (int b = 0; b < ns; ++b)
            sub[static_cast<std::size_t>(a) * ns + b] =
                loss_subpixel(img0, w0, img1, w1, n, Y, X, r.best_dy - sub_range + a * sub_step,
                              r.best_dx - sub_range + b * sub_step);
    r.sub_loss = std::numeric_limits<double>::infinity();
    for (int a = 0; a < ns; ++a)
        for (int b = 0; b < ns; ++b)
            if (sub[static_cast<std::size_t>(a) * ns + b] < r.sub_loss) {
                r.sub_loss = sub[static_cast<std::size_t>(a) * ns + b];
                r.sub_dy = r.best_dy - sub_range + a * sub_step;
                r.sub_dx = r.best_dx - sub_range + b * sub_step;
            }
    return r;
}

RowShift mirror_row_shift(const float* a, const float* b, std::size_t rows, std::size_t cols, double axis,
                          int half_width, double expected, int radius, double sigma, int margin) {
    const long R = static_cast<long>(rows), n = 2L * half_width + 1;
    if (half_width < 1 || radius < 1 || margin < 0) throw Error("mirror_row_shift: bad half_width, radius or margin");
    if (!(axis - half_width >= 0 && axis + half_width <= static_cast<double>(cols) - 1))
        throw Error("mirror_row_shift: the mirrored columns leave the detector");
    // the two strips: a at axis + u, b at axis - u
    std::vector<float> sa(rows * n), sb(rows * n), ha(rows * n), hb(rows * n), tmp(rows * n);
#pragma omp parallel for schedule(static)
    for (long r = 0; r < R; ++r)
        for (long j = 0; j < n; ++j) {
            const double u = static_cast<double>(j - half_width);
            for (int k = 0; k < 2; ++k) {
                const double x = k == 0 ? axis + u : axis - u;
                const long i = std::min(static_cast<long>(std::floor(x)), static_cast<long>(cols) - 2);
                const float f = static_cast<float>(x - static_cast<double>(i));
                const float* row = (k == 0 ? a : b) + r * cols;
                (k == 0 ? sa : sb)[r * n + j] = row[i] * (1.f - f) + row[i + 1] * f;
            }
        }
    for (int k = 0; k < 2; ++k) {   // less the blur
        auto& s = k == 0 ? sa : sb;
        auto& h = k == 0 ? ha : hb;
        gaussian_filter1d<float>(s.data(), tmp.data(), 1, rows, n, 1, sigma, "reflect", 4.0);
        gaussian_filter1d<float>(tmp.data(), h.data(), 1, rows, n, 2, sigma, "reflect", 4.0);
        for (std::size_t i = 0; i < s.size(); ++i) h[i] = s[i] - h[i];
    }
    const long d0 = std::lround(expected), tries = 2L * radius + 1;
    std::vector<double> c(tries, -2.0);
#pragma omp parallel for schedule(dynamic)
    for (long t = 0; t < tries; ++t) {
        const long s = d0 - radius + t;
        // rows v of a against rows v + s of b, `margin` rows inside both
        const long v0 = std::max(0L, -s) + margin, v1 = std::min(R, R - s) - margin;
        if (v1 - v0 < 8) continue;
        double xy = 0, xx = 0, yy = 0;
        for (long v = v0; v < v1; ++v) {
            const float* x = ha.data() + v * n;
            const float* y = hb.data() + (v + s) * n;
            for (long j = margin; j < n - margin; ++j) {
                xy += static_cast<double>(x[j]) * y[j];
                xx += static_cast<double>(x[j]) * x[j];
                yy += static_cast<double>(y[j]) * y[j];
            }
        }
        if (xx > 0 && yy > 0) c[t] = xy / std::sqrt(xx * yy);
    }
    long best = 0;
    for (long t = 1; t < tries; ++t)
        if (c[t] > c[best]) best = t;
    RowShift out;
    out.correlation = c[best];
    out.on_edge = best == 0 || best == tries - 1 || c[best] <= -2.0;
    double sub = 0;
    if (!out.on_edge && c[best - 1] > -2.0 && c[best + 1] > -2.0) {
        const double den = c[best - 1] - 2 * c[best] + c[best + 1];
        if (den < 0) sub = 0.5 * (c[best - 1] - c[best + 1]) / den;
    }
    out.shift = static_cast<double>(d0 - radius + best) + sub;
    return out;
}

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

// ---------------------------------------------------------------- registration of two frames on tiles
std::vector<double> tile_shifts(const float* a, const float* b, std::size_t rows, std::size_t cols, int tile, int step,
                                int margin, int search) {
    if (tile < 16 || tile % 2 || step < 1 || margin < 0 || search < 1 || 2 * search + 2 > tile / 2)
        throw Error("tile_shifts: bad tile, step, margin or search");
    std::vector<int> ys, xs;
    for (int y = margin; y + tile + margin <= static_cast<int>(rows); y += step) ys.push_back(y);
    for (int x = margin; x + tile + margin <= static_cast<int>(cols); x += step) xs.push_back(x);
    const int ny = static_cast<int>(ys.size()), nx = static_cast<int>(xs.size()), nc = tile / 2 + 1;
    std::vector<double> out(static_cast<std::size_t>(ny) * nx * 5);
    if (out.empty()) return out;
    const std::size_t area = static_cast<std::size_t>(tile) * tile, per = static_cast<std::size_t>(tile) * nc;
    std::vector<float> win(tile);
    for (int i = 0; i < tile; ++i) win[i] = static_cast<float>(0.5 - 0.5 * std::cos(2.0 * M_PI * i / (tile - 1)));
    std::mutex& m = fftw_planner_mutex();
    fftwf_plan fwd, inv;
    {
        std::vector<float> t(area);
        std::vector<std::complex<float>> T(per);
        std::lock_guard<std::mutex> lock(m);
        fwd = fftwf_plan_dft_r2c_2d(tile, tile, t.data(), reinterpret_cast<fftwf_complex*>(T.data()), FFTW_ESTIMATE | FFTW_UNALIGNED);
        inv = fftwf_plan_dft_c2r_2d(tile, tile, reinterpret_cast<fftwf_complex*>(T.data()), t.data(), FFTW_ESTIMATE | FFTW_UNALIGNED);
    }
#pragma omp parallel
    {
        std::vector<float> ta(area), tb(area);
        std::vector<std::complex<float>> A(per), B(per);
        std::vector<std::complex<double>> ev(tile), eu(nc);
        auto load = [&](const float* img, int y0, int x0, std::vector<float>& t) {   // windowed tile less its mean; its energy
            double sw = 0, sv = 0;
            for (int i = 0; i < tile; ++i)
                for (int j = 0; j < tile; ++j) {
                    const double w = static_cast<double>(win[i]) * win[j];
                    sw += w;
                    sv += w * img[(static_cast<std::size_t>(y0) + i) * cols + x0 + j];
                }
            const float mean = static_cast<float>(sv / sw);
            double e = 0;
            for (int i = 0; i < tile; ++i)
                for (int j = 0; j < tile; ++j) {
                    const float v = (img[(static_cast<std::size_t>(y0) + i) * cols + x0 + j] - mean) * win[i] * win[j];
                    t[static_cast<std::size_t>(i) * tile + j] = v;
                    e += static_cast<double>(v) * v;
                }
            return e;
        };
#pragma omp for schedule(dynamic, 1)
        for (int q = 0; q < ny * nx; ++q) {
            const int y0 = ys[q / nx], x0 = xs[q % nx];
            double* o = &out[static_cast<std::size_t>(q) * 5];
            o[0] = y0 + 0.5 * (tile - 1);
            o[1] = x0 + 0.5 * (tile - 1);
            o[2] = o[3] = std::numeric_limits<double>::quiet_NaN();
            o[4] = 0;
            const double ea = load(a, y0, x0, ta), eb = load(b, y0, x0, tb);
            if (!(ea > 0) || !(eb > 0)) continue;
            fftwf_execute_dft_r2c(fwd, ta.data(), reinterpret_cast<fftwf_complex*>(A.data()));
            fftwf_execute_dft_r2c(fwd, tb.data(), reinterpret_cast<fftwf_complex*>(B.data()));
            for (std::size_t i = 0; i < per; ++i) A[i] = B[i] = std::conj(A[i]) * B[i];        // (the inverse overwrites A)
            fftwf_execute_dft_c2r(inv, reinterpret_cast<fftwf_complex*>(A.data()), ta.data());   // ta[d] = area x sum a(n) b(n + d)
            auto cc = [&](int dv, int du) { return ta[static_cast<std::size_t>((dv + tile) % tile) * tile + (du + tile) % tile]; };
            int bv = 0, bu = 0;
            float best = cc(0, 0);
            for (int dv = -search; dv <= search; ++dv)
                for (int du = -search; du <= search; ++du)
                    if (cc(dv, du) > best) { best = cc(dv, du); bv = dv; bu = du; }
            if (std::abs(bv) == search || std::abs(bu) == search) continue;
            // the maximum of the continuous correlation c(d) = sum_k C(k) exp(2 pi i k . d / tile) (the cross-spectrum
            // B holds), by Newton steps from the whole-pixel maximum: a parabola through the neighbours along each
            // axis is off by up to 0.1 pixel where the peak is elongated obliquely
            double dv = bv, du = bu, value = best;
            bool ok = true;
            for (int it = 0; it < 8 && ok; ++it) {
                for (int i = 0; i < tile; ++i) {
                    const double ph = 2.0 * M_PI * (i < tile / 2 ? i : i - tile) / tile * dv;
                    ev[i] = std::complex<double>(std::cos(ph), std::sin(ph));
                }
                for (int j = 0; j < nc; ++j) {
                    const double ph = 2.0 * M_PI * j / tile * du;
                    eu[j] = std::complex<double>(std::cos(ph), std::sin(ph));
                }
                double c = 0, gv = 0, gu = 0, hvv = 0, hvu = 0, huu = 0;
                for (int i = 0; i < tile; ++i) {
                    const double kv = 2.0 * M_PI * (i < tile / 2 ? i : i - tile) / tile;
                    for (int j = 0; j < nc; ++j) {
                        const double ku = 2.0 * M_PI * j / tile, mult = (j == 0 || j == tile / 2) ? 1.0 : 2.0;
                        const std::complex<double> z = std::complex<double>(B[static_cast<std::size_t>(i) * nc + j]) * ev[i] * eu[j];
                        const double re = mult * z.real(), im = mult * z.imag();
                        c += re;
                        gv -= kv * im;
                        gu -= ku * im;
                        hvv -= kv * kv * re;
                        hvu -= kv * ku * re;
                        huu -= ku * ku * re;
                    }
                }
                value = c;
                const double det = hvv * huu - hvu * hvu;
                if (!(hvv < 0) || !(det > 0)) { ok = false; break; }      // not a maximum
                const double sv = -(huu * gv - hvu * gu) / det, su = -(hvv * gu - hvu * gv) / det;
                dv += std::clamp(sv, -0.5, 0.5);
                du += std::clamp(su, -0.5, 0.5);
                if (std::fabs(dv - bv) > 1.0 || std::fabs(du - bu) > 1.0) { ok = false; break; }
                if (std::fabs(sv) < 1e-4 && std::fabs(su) < 1e-4) break;
            }
            if (!ok) continue;
            o[2] = dv;
            o[3] = du;
            o[4] = value / static_cast<double>(area) / std::sqrt(ea * eb);
        }
    }
    std::lock_guard<std::mutex> lock(m);
    fftwf_destroy_plan(fwd);
    fftwf_destroy_plan(inv);
    return out;
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
