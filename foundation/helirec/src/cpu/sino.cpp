#include <algorithm>
#include <cmath>
#include <cstring>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "common/rebalance_core.hpp"
#include "common/sino_core.hpp"
#include "helirec/filters.hpp"
#include "helirec/sino.hpp"

namespace helirec {

using namespace helirec::sino_core;

namespace {

long cround(double x) { return std::lround(x); }  // half away from zero (C lround)

// adj = Gb - Ga for row y (mirror row m): Gaussian-weighted means near the axis
float ramp_adj(const float* row, const float* mrow, const RampGeometry& g) {
    if (!g.legacy) return ramp_adj_exact(row, mrow, g);
    if (g.R <= 0) return 0.f;
    const float sigma = static_cast<float>(g.R) / 3.0f;
    const float inv2s2 = 1.0f / (2.0f * sigma * sigma);
    const float my_rot_f = g.rot_center, two_rots_f = 2.0f * g.rot_center;
    const float mirror_center = two_rots_f - my_rot_f;
    const int i0 = std::max(0, g.my_rot - g.R), i1 = std::min(g.n_bins - 1, g.my_rot + g.R);
    float sa = 0, wa = 0, sb = 0, wb = 0;
    for (int ix = i0; ix <= i1; ++ix) {
        const float dA = static_cast<float>(ix) - my_rot_f;
        const float wA = std::exp(-(dA * dA) * inv2s2);
        sa += row[ix] * wA;
        wa += wA;
        const int xm = g.two_rots - ix;
        if (xm >= 0 && xm < g.n_bins) {
            const float dB = static_cast<float>(xm) - mirror_center;
            const float wB = std::exp(-(dB * dB) * inv2s2);
            sb += mrow[xm] * wB;
            wb += wB;
        }
    }
    if (wa > 0 && wb > 0) return sb / wb - sa / wa;
    return 0.f;
}

}  // namespace

// ------------------------------------------------------------------ ramp geometry/filter
RampGeometry ramp_geometry(int n_bins, int n_proj, float rot_center, int pad_factor, bool legacy) {
    static const std::vector<long> powers = [] {
        std::set<long> p;
        for (int a = 2; a <= 15; ++a)
            for (int b = 0; b <= 9; ++b)
                for (int c = 0; c <= 6; ++c)
                    for (int d = 0; d <= 5; ++d)
                        p.insert((1L << a) * static_cast<long>(std::pow(3, b)) * static_cast<long>(std::pow(5, c)) *
                                 static_cast<long>(std::pow(7, d)));
        return std::vector<long>(p.begin(), p.end());
    }();
    auto it = std::lower_bound(powers.begin(), powers.end(), static_cast<long>(n_bins));
    const long base = it == powers.end() ? powers.back() : *it;
    RampGeometry g;
    g.n_bins = n_bins;
    g.n_proj = n_proj;
    g.rot_center = rot_center;
    g.npad = static_cast<int>(2 * base * std::max(1, pad_factor));
    g.pad_l = (g.npad - n_bins) / 2;
    g.two_rots = static_cast<int>(cround(static_cast<double>(2.0f * rot_center)));
    g.my_rot = static_cast<int>(cround(static_cast<double>(rot_center)));
    int R = g.two_rots > n_bins ? std::min(30, n_bins - 1 - std::max(g.my_rot, g.two_rots - g.my_rot))
                                : std::min(30, std::min(g.my_rot, g.two_rots - g.my_rot) - 1);
    g.R = std::max(R, 0);
    g.switch_pos = g.npad - g.pad_l;
    g.legacy = legacy;
    if (n_proj % 2 != 0) throw Error("ramp filter: odd number of grid angles is not supported");
    return g;
}

float ramp_scale(int n_proj, bool legacy) {
    return static_cast<float>(M_PI) * (legacy ? 1.f : 2.f) / static_cast<float>(n_proj);
}

std::vector<float> ramlak_half_spectrum(int npad, float scale) {
    std::vector<double> h(npad, 0.0);
    const int L2 = npad / 2 + 1;
    std::vector<float> hf(npad, 0.f);
    hf[0] = 0.25f;
    const float pi = static_cast<float>(M_PI);
    for (int n = 1; n <= L2 - 2; n += 2) hf[n] = -1.0f / (pi * pi * static_cast<float>(n) * static_cast<float>(n));
    for (int q = 0; q < npad - L2; ++q) hf[L2 + q] = hf[(L2 - 2) - q];
    // real part of the DFT (double), as numpy.fft.rfft(h.astype(float64)).real
    const int nh = npad / 2 + 1;
    std::vector<float> H(nh);
    for (int k = 0; k < nh; ++k) {
        double acc = 0;
        for (int n = 0; n < npad; ++n) {
            if (hf[n] == 0.f) continue;
            acc += static_cast<double>(hf[n]) * std::cos(2 * M_PI * static_cast<double>((static_cast<long>(k) * n) % npad) / npad);
        }
        float v = static_cast<float>(acc);
        if (v < 0 && std::abs(v) < 1e-7f) v = 0.f;
        H[k] = v;
    }
    for (auto& v : H) v *= scale;
    return H;
}

std::vector<std::string> ramp_filter_names() {
    return {"none", "shepp", "cosine", "hann", "hamming", "ramlak", "parzen", "butterworth"};
}

std::vector<float> ramp_filter_par(const std::string& text) {
    std::vector<float> out;
    std::string t = text;
    for (char& c : t)
        if (std::strchr(":/;,[]()", c)) c = ' ';
    std::istringstream is(t);
    float v;
    while (is >> v) out.push_back(v);
    return out;
}

float ramp_windowed(const std::string& name, float x, const std::vector<float>& p, float ramp) {
    const float pi = static_cast<float>(M_PI);
    const float p0 = p.empty() ? 0.5f : p[0], p1 = p.size() > 1 ? p[1] : 8.f;
    if (name == "none") return 1.f;
    if (name == "shepp") return x == 0.f ? ramp : ramp * (std::sin(pi * x) / (pi * x));
    if (name == "cosine") return ramp * std::cos(pi * x);
    if (name == "ramlak") return ramp;
    const bool cut = name == "hann" || name == "hamming" || name == "parzen";   // zero beyond the cutoff p0
    if (cut || name == "butterworth") {
        if (!(p0 > 0.f) || !std::isfinite(p0)) throw Error("ramp filter " + name + ": the cutoff (filter_par) must be > 0");
        if (cut && std::fabs(x) > p0) return 0.f;
        if (name == "hann") return ramp * 0.5f * (1.f + std::cos(pi * x / p0));
        if (name == "hamming") return ramp * (0.54f + 0.46f * std::cos(pi * x / p0));
        if (name == "parzen") return ramp * std::pow(1.f - std::fabs(x) / p0, 3.f);
        return ramp / (1.f + std::pow(x / p0, 2 * p1));
    }
    std::string list;
    for (const auto& n : ramp_filter_names()) list += (list.empty() ? "" : ", ") + n;
    throw Error("unknown ramp filter '" + name + "' (" + list + ")");
}

FourierFilterSpec ramp_filter_spec(const RampGeometry& g, float scale, const std::string& filter_name,
                                   const std::vector<float>& filter_par) {
    FourierFilterSpec s;
    s.ndim = 1;
    s.h = 1;
    s.w = g.npad;
    s.ph = 1;
    s.pw = g.npad;
    s.pad_top = 0;
    s.pad_left = 0;
    s.mode = PadMode::Zero;
    s.filter = ramlak_half_spectrum(g.npad, 1.f);
    for (std::size_t k = 0; k < s.filter.size(); ++k)
        s.filter[k] = ramp_windowed(filter_name, static_cast<float>(k) / g.npad, filter_par, s.filter[k]) * scale;
    return s;
}

FourierFilterSpec band_filter_spec(int ndim, int h, int w, double f1, double f2) {
    if (ndim != 1 && ndim != 2) throw Error("band_filter_spec: ndim must be 1 or 2");
    if (!(f1 >= 0 && f2 > f1 && f2 <= 0.75)) throw Error("band_filter_spec: need 0 <= f1 < f2 <= 0.75 cycles per pixel");
    FourierFilterSpec s;
    s.ndim = ndim;
    s.h = ndim == 2 ? h : 1;
    s.w = w;
    s.ph = ndim == 2 ? roundfft(h) : 1;      // (the frames carry their own margin rows)
    s.pw = roundfft(w + 64);
    s.pad_top = ndim == 2 ? (s.ph - h) / 2 : 0;
    s.pad_left = (s.pw - w) / 2;
    s.mode = PadMode::Edge;
    const int nc = s.pw / 2 + 1;
    s.filter.resize(static_cast<std::size_t>(s.ph) * nc);
    for (int i = 0; i < s.ph; ++i) {
        const double fy = ndim == 2 ? static_cast<double>(std::min(i, s.ph - i)) / s.ph : 0.0;
        for (int j = 0; j < nc; ++j) {
            const double fr = std::hypot(fy, static_cast<double>(j) / s.pw);
            const double v = fr <= f1 ? 0.0 : (fr >= f2 ? 1.0 : 0.5 - 0.5 * std::cos(M_PI * (fr - f1) / (f2 - f1)));
            s.filter[static_cast<std::size_t>(i) * nc + j] = static_cast<float>(v);
        }
    }
    return s;
}

std::vector<float> blur_response(const std::vector<double>& blur, double f, int rows, int cols) {
    if (blur.size() != 6) throw Error("sharpness_weighting: six numbers a11:a12:a21:a22:c1:c2 are needed");
    if (!(f > 0 && f <= 0.5)) throw Error("sharp_f must be in (0, 0.5] cycles per pixel");
    std::vector<float> out(static_cast<std::size_t>(rows) * cols);
    for (int r = 0; r < rows; ++r) {
        const double Y = (r - 0.5 * (rows - 1)) / 1000.0;
        for (int c = 0; c < cols; ++c) {
            const double X = (c - 0.5 * (cols - 1)) / 1000.0;
            const double a = (blur[0] * X + blur[1] * Y + blur[4]) * f;
            double v = 0.0;
            if (std::fabs(a) < 1.0) {
                const double pa = M_PI * a;
                v = std::fabs(pa) < 1e-9 ? 1.0 : std::sin(pa) / pa;
            }
            out[static_cast<std::size_t>(r) * cols + c] = static_cast<float>(std::max(v * v, 1e-6));
        }
    }
    return out;
}

std::vector<float> arc_deblur_taps(double arc, int radius) {
    if (!(arc > 0 && arc <= 1.5)) throw Error("exposure arc must be in (0, 1.5] grid angles");
    if (radius < 1 || radius > 16) throw Error("arc_deblur_taps: radius must be 1..16");
    // g[k] = integral over |f| <= 1/2 of cos(2 pi f k) / sinc(arc f) (midpoint rule), then unit sum
    const int n = 8192;
    std::vector<double> g(radius + 1, 0.0);
    for (int i = 0; i < n; ++i) {
        const double f = (i + 0.5) / n * 0.5, a = M_PI * arc * f, h = a / std::sin(a);
        for (int k = 0; k <= radius; ++k) g[k] += h * std::cos(2 * M_PI * f * k) / n;
    }
    double sum = g[0];
    for (int k = 1; k <= radius; ++k) sum += 2 * g[k];
    std::vector<float> out(radius + 1);
    for (int k = 0; k <= radius; ++k) out[k] = static_cast<float>(g[k] / sum);
    return out;
}

namespace cpu {

void minus_log(float* x, std::size_t n, float lo, float hi) {
#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) x[i] = sino_core::minus_log(x[i], lo, hi);
}

void multiply(float* x, const float* w, std::size_t n) {
#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) x[i] = x[i] * w[i];
}

void add_scaled(float* x, const float* y, float s, std::size_t n) {
#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) x[i] = x[i] + s * y[i];
}

void weight_difference(const float* x, const float* w, const float* w2, float* t, std::size_t n) {
#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) t[i] = x[i] * (w2[i] - w[i]);
}

void multiply_add(float* x, const float* w, const float* t, std::size_t n) {
#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) x[i] = x[i] * w[i] + t[i];
}

void angular_fir(const float* in, float* out, int n_proj, int n_cols, const float* g, int radius, const float* support) {
    if (radius < 0 || 2 * radius + 1 > n_proj) throw Error("angular_fir: bad radius");
#pragma omp parallel for schedule(static)
    for (int j = 0; j < n_proj; ++j)
        for (int x = 0; x < n_cols; ++x)
            out[static_cast<std::size_t>(j) * n_cols + x] = sino_core::angular_fir_at(in, support, n_proj, n_cols, j, x, g, radius);
}

void ramp_pad(const float* sino, float* padded, const RampGeometry& g, const int32_t* opposite) {
#pragma omp parallel for schedule(static)
    for (int y = 0; y < g.n_proj; ++y) {
        const int m = opposite ? opposite[y] : (y + g.n_proj / 2) % g.n_proj;
        const float* row = sino + static_cast<std::size_t>(y) * g.n_bins;
        const float* mrow = sino + static_cast<std::size_t>(m) * g.n_bins;
        const float adj = ramp_adj(row, mrow, g);
        float* out = padded + static_cast<std::size_t>(y) * g.npad;
        for (int x = 0; x < g.npad; ++x)
            out[x] = ramp_pad_value(row, mrow, adj, x, g);
    }
}

void ramp_crop(const float* padded, float* out, const RampGeometry& g) {
#pragma omp parallel for schedule(static)
    for (int y = 0; y < g.n_proj; ++y)
        std::memcpy(out + static_cast<std::size_t>(y) * g.n_bins, padded + static_cast<std::size_t>(y) * g.npad,
                    sizeof(float) * g.n_bins);
}

// ------------------------------------------------------------------ rebalance
namespace {
using rebalance_core::Contribution;
using rebalance_core::OddMap;
inline float mapped(const float* row, const OddMap& e) {
    float v = 0.f;
    if (e.src0 >= 0) v += e.w0 * row[e.src0];
    if (e.src1 >= 0) v += e.w1 * row[e.src1];
    return v;
}
void balance_with_table(float* W, int n_proj, int n_rows, int n_cols, const std::vector<std::vector<Contribution>>& table,
                        const std::vector<OddMap>& omap);
}  // namespace

int rebalance(float* R, float* W, int n_proj, int n_rows, int n_cols, float rot_center, bool legacy) {
    const std::size_t plane = static_cast<std::size_t>(n_rows) * n_cols;
    // ---- hole filling (mid-row emptiness test, cyclic merge)
    const int mid = n_rows / 2;
    std::vector<char> nonempty(n_proj, 0);
    for (int i = 0; i < n_proj; ++i) {
        const float* w = W + i * plane + static_cast<std::size_t>(mid) * n_cols;
        for (int c = 0; c < n_cols; ++c)
            if (w[c] != 0.0f) { nonempty[i] = 1; break; }
    }
    bool any = false;
    for (char v : nonempty) any |= v;
    if (!any) return 0;
    const auto holes = rebalance_core::holes_from(nonempty);
    std::vector<float> rf(plane), wf(plane);
    for (auto [s, e] : holes) {
        const int left = (s + n_proj - 1) % n_proj, right = e % n_proj;
        for (std::size_t k = 0; k < plane; ++k) {
            rf[k] = 0.5f * (R[left * plane + k] + R[right * plane + k]);
            wf[k] = 0.5f * (W[left * plane + k] + W[right * plane + k]);
        }
        for (int idx = s; idx < e; ++idx) {
            std::memcpy(R + (idx % n_proj) * plane, rf.data(), plane * sizeof(float));
            std::memcpy(W + (idx % n_proj) * plane, wf.data(), plane * sizeof(float));
        }
    }
    balance_with_table(W, n_proj, n_rows, n_cols, rebalance_core::contribution_table(n_proj, legacy),
                       rebalance_core::odd_shift_map(n_cols, rot_center));
    return static_cast<int>(holes.size());
}

namespace {
// W <- W_i / (W_i + the contributions table[i] names), row by row
void balance_with_table(float* W, int n_proj, int n_rows, int n_cols, const std::vector<std::vector<Contribution>>& table,
                        const std::vector<OddMap>& omap) {
    const std::size_t plane = static_cast<std::size_t>(n_rows) * n_cols;
    std::vector<float> Wsrc(static_cast<std::size_t>(n_proj) * n_cols);
    for (int row = 0; row < n_rows; ++row) {
        for (int i = 0; i < n_proj; ++i)
            std::memcpy(Wsrc.data() + static_cast<std::size_t>(i) * n_cols, W + i * plane + static_cast<std::size_t>(row) * n_cols,
                        n_cols * sizeof(float));
#pragma omp parallel
        {
            std::vector<float> B(n_cols);
#pragma omp for schedule(static)
            for (int i = 0; i < n_proj; ++i) {
                const float* Wi = Wsrc.data() + static_cast<std::size_t>(i) * n_cols;
                std::copy(Wi, Wi + n_cols, B.begin());
                for (const auto& m : table[i]) {
                    const float* r0 = Wsrc.data() + static_cast<std::size_t>(m.k0) * n_cols;
                    const float* r1 = m.has_two ? Wsrc.data() + static_cast<std::size_t>(m.k1) * n_cols : nullptr;
                    const float beta = 1.0f - m.alpha, alpha = m.alpha;
                    for (int c = 0; c < n_cols; ++c) {
                        if (!m.odd) B[c] += m.has_two ? beta * r0[c] + alpha * r1[c] : r0[c];
                        else B[c] += m.has_two ? beta * mapped(r0, omap[c]) + alpha * mapped(r1, omap[c]) : mapped(r0, omap[c]);
                    }
                }
                float* dst = W + i * plane + static_cast<std::size_t>(row) * n_cols;
                for (int c = 0; c < n_cols; ++c) dst[c] = Wi[c] == 0.0f ? 0.0f : Wi[c] / B[c];
            }
        }
    }
}

void check_turns(int n_proj, int turn) {
    if (turn < 2 || turn % 2 != 0) throw Error("rebalance_turns: the rows of a turn must be an even number");
    if (n_proj < turn) throw Error("rebalance_turns: fewer rows than a turn");
}
}  // namespace

// ------------------------------------------------------------------ rebalance, one row per view
TurnHoles turn_holes(const float* W, int n_proj, int n_rows, int n_cols, int turn) {
    check_turns(n_proj, turn);
    const std::size_t plane = static_cast<std::size_t>(n_rows) * n_cols;
    std::vector<float> peak(n_proj, 0.f);   // the largest weight of each row's mid-row
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n_proj; ++i) {
        const float* w = W + i * plane + static_cast<std::size_t>(n_rows / 2) * n_cols;
        for (int c = 0; c < n_cols; ++c)
            if (w[c] > peak[i]) peak[i] = w[c];
    }
    return rebalance_core::turn_holes_from(peak, turn);
}

void fill_turn_holes(float* X, const TurnHoles& holes, int n_rows, int n_cols) {
    const std::size_t plane = static_cast<std::size_t>(n_rows) * n_cols;
    for (std::size_t h = 0; h < holes.dst.size(); ++h) {
        float* dst = X + holes.dst[h] * plane;
        const float* a = X + holes.left[h] * plane;
        const float* b = X + holes.right[h] * plane;
#pragma omp parallel for schedule(static)
        for (std::ptrdiff_t k = 0; k < static_cast<std::ptrdiff_t>(plane); ++k) dst[k] = 0.5f * (a[k] + b[k]);
    }
}

void fill_from_turns(float* R, const float* W, int n_proj, int n_rows, int n_cols, int turn) {
    check_turns(n_proj, turn);
    const std::size_t plane = static_cast<std::size_t>(n_rows) * n_cols;
    // (only samples without weight are written and only samples with weight are read: no order matters)
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n_proj; ++i)
        for (std::size_t k = 0; k < plane; ++k) {
            const std::size_t o = i * plane + k;
            const int src = W[o] != 0.0f ? -1 : sino_core::turn_source(W, plane, k, n_proj, turn, i);
            if (src >= 0) R[o] = R[static_cast<std::size_t>(src) * plane + k];
        }
}

void balance_weights_turns(float* W, int n_proj, int n_rows, int n_cols, int turn, float rot_center) {
    check_turns(n_proj, turn);
    balance_with_table(W, n_proj, n_rows, n_cols, rebalance_core::contribution_table_turns(n_proj, turn),
                       rebalance_core::odd_shift_map(n_cols, rot_center));
}

int rebalance_turns(float* R, float* W, int n_proj, int n_rows, int n_cols, int turn, float rot_center) {
    const TurnHoles holes = turn_holes(W, n_proj, n_rows, n_cols, turn);
    if (holes.all_empty) return 0;
    fill_turn_holes(R, holes, n_rows, n_cols);
    fill_turn_holes(W, holes, n_rows, n_cols);
    fill_from_turns(R, W, n_proj, n_rows, n_cols, turn);
    balance_weights_turns(W, n_proj, n_rows, n_cols, turn, rot_center);
    return static_cast<int>(holes.dst.size());
}

}  // namespace cpu
}  // namespace helirec
