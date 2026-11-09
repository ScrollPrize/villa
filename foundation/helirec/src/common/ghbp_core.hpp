#pragma once
// Per-pixel GHBP arithmetic, shared verbatim by the CPU and CUDA kernels.
// Operation order reproduces nabuxx ghbp_cuda.cu as compiled by nvcc (FMA contraction).
#include <cstddef>

#include "hd_math.hpp"

namespace helirec::ghbp_core {

using namespace helirec::hd;

// grid pixel (a = row along H, b = col along W) -> image coordinates (Y, X)
// nabuxx: T0*a + T1*b + T2
HR_HD void grid_point(const float* T, int a, int b, float& Y, float& X) {
    Y = add(fma(T[0], static_cast<float>(a), mul(T[1], static_cast<float>(b))), T[2]);
    X = add(fma(T[3], static_cast<float>(a), mul(T[4], static_cast<float>(b))), T[5]);
}

// Level 0: sum over rf projections of the linearly interpolated detector sample.
// S: (n_angles, 8) rows ph[4], pw[4]; dot4((0, Y, X, 1), r) = fma(X, r2, Y*r1) + r3
HR_HD float level0_value(const float* setups, const float* sino, int n_bins, int n_angles, int first_proj,
                         int rf, float Y, float X) {
    float val = 0.f;
    for (int k = 0; k < rf; ++k) {
        const int s = first_proj + k;
        if (s >= n_angles) continue;
        const float* S = setups + static_cast<std::size_t>(s) * 8;
        const float n = div(1.f, add(fma(X, S[6], mul(Y, S[5])), S[7]));
        const float h = mul(add(fma(X, S[2], mul(Y, S[1])), S[3]), n);
        const float fl = floor(h);
        const int i0 = static_cast<int>(fl), i1 = i0 + 1;
        const float f = sub(h, static_cast<float>(i0));
        const float* row = sino + static_cast<std::size_t>(s) * n_bins;
        if (i0 >= 0 && i0 < n_bins) val = fma(row[i0], sub(1.0f, f), val);
        if (i1 >= 0 && i1 < n_bins) val = fma(row[i1], f, val);
    }
    return val;
}

// clamp-to-edge bilinear sample of img (h rows, w cols) at column x, row y
HR_HD float bilinear_clamped(const float* img, int w, int h, float x, float y) {
    int ix0 = static_cast<int>(floor(x)), iy0 = static_cast<int>(floor(y));
    const float fx = sub(x, static_cast<float>(ix0)), fy = sub(y, static_cast<float>(iy0));
    int ix1 = ix0 + 1, iy1 = iy0 + 1;
    ix0 = imin(w - 1, imax(0, ix0));
    ix1 = imin(w - 1, imax(0, ix1));
    iy0 = imin(h - 1, imax(0, iy0));
    iy1 = imin(h - 1, imax(0, iy1));
    const float v00 = img[static_cast<std::size_t>(iy0) * w + ix0], v01 = img[static_cast<std::size_t>(iy0) * w + ix1];
    const float v10 = img[static_cast<std::size_t>(iy1) * w + ix0], v11 = img[static_cast<std::size_t>(iy1) * w + ix1];
    const float gx = sub(1.f, fx), gy = sub(1.f, fy);
    const float top = fma(v00, gx, mul(v01, fx));
    const float bot = fma(v10, gx, mul(v11, fx));
    return fma(top, gy, mul(bot, fy));
}

// Aggregation: sum of bilinear samples of the previous level's grids.
// prev grids are leg-local, indexed gi*red + k; Tinv rows map (Y, X, 1) -> (row, col)
HR_HD float aggregate_value(const float* Tinv_leg, const float* prev, int pw, int ph, int png, int gi, int red,
                            float Y, float X) {
    float val = 0.f;
    for (int k = 0; k < red; ++k) {
        const int pi = gi * red + k;
        if (pi >= png) continue;
        const float* Ti = Tinv_leg + pi * 6;
        if (is_nan(Ti[0])) break;
        const float px = add(fma(Ti[3], Y, mul(Ti[4], X)), Ti[5]);
        const float py = add(fma(Ti[0], Y, mul(Ti[1], X)), Ti[2]);
        val = add(val, bilinear_clamped(prev + static_cast<std::size_t>(pi) * ph * pw, pw, ph, px, py));
    }
    return val;
}

// nabuxx is_in_circle: (x-cx)^2 + (y-cy)^2 <= r2, r2 = (N/2)^2 (integer)
HR_HD bool in_circle(int i, int j, int N) {
    const float c = mul(static_cast<float>(N - 1), 0.5f);
    int r = N / 2;
    const float dx = sub(static_cast<float>(j), c), dy = sub(static_cast<float>(i), c);
    return fma(dx, dx, mul(dy, dy)) <= static_cast<float>(r * r);
}

}  // namespace helirec::ghbp_core
