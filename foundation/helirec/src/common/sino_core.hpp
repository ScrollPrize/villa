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

}  // namespace helirec::sino_core
