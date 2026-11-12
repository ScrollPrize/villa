#pragma once
// Per-element arithmetic of the sinogram stage, shared by CPU and CUDA.
// The reference runs these steps on the CPU without FMA: keep mul/add separate.
#include <cstddef>

#include "hd_math.hpp"
#include "helirec/sino.hpp"

namespace helirec::sino_core {

using namespace helirec::hd;

HR_HD float minus_log(float v, float lo, float hi) {
    v = v < lo ? lo : v;
    v = v > hi ? hi : v;
#if defined(__CUDA_ARCH__)
    return -logf(v);
#else
    return -std::log(v);
#endif
}

// one output of angular_fir (sino.hpp): taps in the order n = -radius .. radius
HR_HD float angular_fir_at(const float* in, const float* support, int n_proj, int n_cols, int j, int x, const float* g,
                           int radius) {
    float v = 0.f;
    for (int n = -radius; n <= radius; ++n) {
        int jj = j + n;
        if (jj < 0) jj += n_proj;
        if (jj >= n_proj) jj -= n_proj;
        const std::size_t o = static_cast<std::size_t>(jj) * n_cols + x;
        if (support && !(support[o] > 0.f)) return in[static_cast<std::size_t>(j) * n_cols + x];
        v = add(v, mul(g[n < 0 ? -n : n], in[o]));
    }
    return v;
}

// fill_from_turns (sino.hpp): the row whose sample k of the plane replaces that of row i (whose weight is 0): of the
// rows a whole number of turns away, the one with the largest weight there (the nearest of equal ones, i + d before
// i - d); -1: none has a weight
HR_HD int turn_source(const float* W, std::size_t plane, std::size_t k, int n_proj, int turn, int i) {
    int best = -1;
    float wbest = 0.f;
    for (int d = turn; i + d < n_proj || i - d >= 0; d += turn) {
        if (i + d < n_proj && W[static_cast<std::size_t>(i + d) * plane + k] > wbest) {
            best = i + d;
            wbest = W[static_cast<std::size_t>(best) * plane + k];
        }
        if (i - d >= 0 && W[static_cast<std::size_t>(i - d) * plane + k] > wbest) {
            best = i - d;
            wbest = W[static_cast<std::size_t>(best) * plane + k];
        }
    }
    return best;
}

// row sample at fractional position xf in [0, nx - 1] (linear)
HR_HD float lerp_row(const float* row, float xf, int nx) {
    const int i = static_cast<int>(floorf(xf));
    if (i >= nx - 1) return row[nx - 1];
    if (i < 0) return row[0];
    const float f = sub(xf, static_cast<float>(i));
    return add(row[i], mul(f, sub(row[i + 1], row[i])));
}

// offset matching of the mirrored continuation (exact reflection): Gaussian-weighted means of the row and of its
// opposite row at the reflected positions 2 rot - ix, over the R bins around the axis; adj = mirror - own
// (the legacy form, with integer reflection, is in the CPU / CUDA ramp_pad)
HR_HD float ramp_adj_exact(const float* row, const float* mrow, const RampGeometry& g) {
    if (g.R <= 0) return 0.f;
    const float sigma = static_cast<float>(g.R) / 3.0f;
    const float inv2s2 = 1.0f / mul(mul(2.0f, sigma), sigma);
    const float rot = g.rot_center, tr = mul(2.0f, g.rot_center);
    const int i0 = g.my_rot - g.R < 0 ? 0 : g.my_rot - g.R, i1 = g.my_rot + g.R > g.n_bins - 1 ? g.n_bins - 1 : g.my_rot + g.R;
    float sa = 0, wa = 0, sb = 0, wb = 0;
    for (int ix = i0; ix <= i1; ++ix) {
        const float dA = sub(static_cast<float>(ix), rot);
        const float w = expf(-mul(mul(dA, dA), inv2s2));   // the reflected position is as far from the axis
        sa = add(sa, mul(row[ix], w));
        wa = add(wa, w);
        const float xm = sub(tr, static_cast<float>(ix));
        if (xm >= 0.f && xm <= static_cast<float>(g.n_bins - 1)) {
            sb = add(sb, mul(lerp_row(mrow, xm, g.n_bins), w));
            wb = add(wb, w);
        }
    }
    return (wa > 0 && wb > 0) ? sub(sb / wb, sa / wa) : 0.f;
}

// one padded sample x of row `row` (mirror row `mrow`, offset adj) -- nabuxx pad_row_mirror_edges
HR_HD float ramp_pad_value(const float* row, const float* mrow, float adj, int x, const RampGeometry& g) {
    const int nx = g.n_bins;
    if (x < nx) return row[x];
    if (g.legacy) {   // the reference: reflection about the integer lround(2 rot_center)
        if (g.two_rots > nx) {
            const int xm = g.two_rots - x;
            if (xm >= 0) return sub(mrow[xm], adj);
            if (x < g.switch_pos) return sub(mrow[0], adj);
            return row[0];
        }
        const int xm = g.two_rots - (x - g.npad);
        if (xm <= nx - 1) return sub(mrow[xm], adj);
        if (x < g.switch_pos) return row[nx - 1];
        return sub(mrow[nx - 1], adj);
    }
    // the opposite ray of detector position x is at 2 rot_center - x (interpolated)
    const float tr = mul(2.0f, g.rot_center);
    if (tr > static_cast<float>(nx)) {
        const float xm = sub(tr, static_cast<float>(x));
        if (xm >= 0.f) return sub(lerp_row(mrow, xm, nx), adj);
        if (x < g.switch_pos) return sub(mrow[0], adj);
        return row[0];
    }
    const float xm = sub(tr, static_cast<float>(x - g.npad));
    if (xm <= static_cast<float>(nx - 1)) return sub(lerp_row(mrow, xm, nx), adj);
    if (x < g.switch_pos) return row[nx - 1];
    return sub(mrow[nx - 1], adj);
}

// ---------------------------------------------------------------- band-limited shift of rows (sino.hpp)
// sample i >= n of a row padded to L: the bridge from its last sample to its first
HR_HD float shift_bridge(float last, float first, int i, int n, int L) {
    const float t = static_cast<float>(i - n + 1) / static_cast<float>(L - n + 1);
    const float w = 0.5f - 0.5f * cosf(3.14159265358979f * t);
    return last + (first - last) * w;
}
// bin k of L (k <= L / 2) of a row delayed by `shift` samples, times 1 / L (the unnormalised inverse transform)
HR_HD void shift_bin(float& re, float& im, int k, int L, float shift) {
    const float s = 1.f / static_cast<float>(L);
    if (2 * k == L) {
        re *= cosf(3.14159265358979f * shift) * s;
        im = 0.f;
        return;
    }
    const float a = -6.28318530717959f * shift * static_cast<float>(k) / static_cast<float>(L);
    const float c = cosf(a) * s, d = sinf(a) * s;
    const float r = re * c - im * d;
    im = re * d + im * c;
    re = r;
}

// ---------------------------------------------------------------- band-limited resampling of rows (sino.hpp)
// The row's values at B positions p (samples of the row; any real numbers: the row is periodic) from g, its
// transform laid in M = resample_oversampling x L bins and transformed back (M samples): for each the sum of the
// resample_taps samples around resample_oversampling x p with the kernel's weights (kernel: resample_kernel()), in
// ascending order. The weights of the taps i and taps - 1 - i are the two signs of one polynomial: its even and odd
// parts in t are evaluated once, in powers of t^2. Out: the first `count` values (NaN for a position that is not
// finite, or too large to be reduced to the row's period).
template <int B>
HR_HD void resample_values(const double* g, int M, const double* p, const double* kernel, float* out, int count) {
    constexpr int H = resample_taps / 2, D = resample_degree;
    constexpr int qe = D - D % 2, qo = D - 1 + D % 2;   // the largest even and odd powers
    // (loops over the B positions, one operation each: the form the CPU's compilers vectorise)
    double first[B], t[B], s[B], even[B], odd[B], bad[B], w[resample_taps][B];
    HR_SIMD
    for (int l = 0; l < B; ++l) {
        double u = p[l] * static_cast<double>(resample_oversampling);   // (exact: a power of two)
        u -= static_cast<double>(M) * floor(u / static_cast<double>(M));   // (exact: fmod's result, or M - a rounding of it)
        const bool in_period = u >= 0.0 && u <= static_cast<double>(M);
        bad[l] = in_period ? 0.0 : static_cast<double>(NAN);
        u = in_period ? u : 0.0;
        const double lo = dsub(u, 0.5 * resample_taps);
        first[l] = ceil(lo);
        t[l] = dsub(dmul(2.0, dsub(first[l], lo)), 1.0);
        s[l] = dmul(t[l], t[l]);
    }
    for (int i = 0; i < H; ++i) {
        const double* c = kernel + i;   // (the coefficient of t^q of tap i: c[q * H])
        HR_SIMD
        for (int l = 0; l < B; ++l) even[l] = c[qe * H];
        for (int q = qe - 2; q >= 0; q -= 2)
            HR_SIMD
            for (int l = 0; l < B; ++l) even[l] = dmuladd(even[l], s[l], c[q * H]);
        HR_SIMD
        for (int l = 0; l < B; ++l) odd[l] = c[qo * H];
        for (int q = qo - 2; q >= 1; q -= 2)
            HR_SIMD
            for (int l = 0; l < B; ++l) odd[l] = dmuladd(odd[l], s[l], c[q * H]);
        HR_SIMD
        for (int l = 0; l < B; ++l) odd[l] = dmul(odd[l], t[l]);
        double* lower = w[i];
        double* upper = w[resample_taps - 1 - i];
        HR_SIMD
        for (int l = 0; l < B; ++l) lower[l] = dadd(even[l], odd[l]);
        HR_SIMD
        for (int l = 0; l < B; ++l) upper[l] = dsub(even[l], odd[l]);
    }
    for (int l = 0; l < count; ++l) {
        int m = static_cast<int>(first[l]);
        m = m < 0 ? m + M : m;
        double v = 0.0;
        for (int i = 0; i < resample_taps; ++i) {
            v = dmuladd(w[i][l], g[m], v);
            m = m + 1 == M ? 0 : m + 1;
        }
        out[l] = static_cast<float>(v + bad[l]);
    }
}

}  // namespace helirec::sino_core
