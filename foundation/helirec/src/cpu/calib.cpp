#include "helirec/calib.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

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

}  // namespace helirec::calib
