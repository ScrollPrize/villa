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

}  // namespace helirec::sino_core
