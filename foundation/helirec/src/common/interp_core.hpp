#pragma once
// Interpolation of a sinogram row at a real detector position u (bins), shared by every exact backprojector
// (CPU, CUDA, full slice, band). Samples outside [0, n) are 0. Add a kernel: a new Interp value, a weight function
// and a case in sample_row / dispatch_interp.
//   Linear    2 taps (the classic pixel-driven backprojector)
//   Cubic     4 taps, Keys a = -0.5 (Catmull-Rom)
//   Lanczos3  6 taps, sinc(x) sinc(x/3), weights normalised to sum 1
//   BSpline3  4 taps of cubic B-spline coefficients (rows prefiltered once, bspline_prefilter_row): the cubic
//             spline through the samples; near-ideal passband, no ringing
//   BSpline5  6 taps, quintic B-spline (two prefilter poles); closer still to band-limited (sinc) interpolation
// Samples (and spline coefficients) outside [0, n) are 0: the row is zero-extended; the coefficients are exact for
// that extension inside [0, n), the taps beyond are dropped (affects u within 2-3 bins of the detector ends).
#include <cmath>

#include "hd_math.hpp"

namespace helirec::interp_core {

enum Interp : int { Linear = 0, Cubic = 1, Lanczos3 = 2, BSpline3 = 3, BSpline5 = 4 };

// rows of the exact backprojectors need a B-spline prefilter (spline degree, 0: none)
HR_HD int spline_degree(int interp) { return interp == BSpline3 ? 3 : (interp == BSpline5 ? 5 : 0); }

HR_HD float tap(const float* row, int n, int i) { return (i >= 0 && i < n) ? row[i] : 0.f; }

HR_HD float keys(float t) {  // a = -0.5
    t = fabsf(t);
    if (t < 1.f) return (1.5f * t - 2.5f) * t * t + 1.f;
    if (t < 2.f) return ((-0.5f * t + 2.5f) * t - 4.f) * t + 2.f;
    return 0.f;
}

HR_HD float lanczos3(float t) {
    if (fabsf(t) < 1e-6f) return 1.f;
    if (fabsf(t) >= 3.f) return 0.f;
    const float pt = 3.14159265358979f * t;
    return 3.f * sinf(pt) * sinf(pt / 3.f) / (pt * pt);
}

HR_HD float bspline3(float t) {
    t = fabsf(t);
    if (t < 1.f) return 2.f / 3.f - t * t + 0.5f * t * t * t;
    if (t < 2.f) {
        const float u = 2.f - t;
        return u * u * u / 6.f;
    }
    return 0.f;
}

HR_HD float bspline5(float t) {
    t = fabsf(t);
    if (t >= 3.f) return 0.f;
    const float a = 3.f - t, b = 2.f - t, c = 1.f - t;
    const float a5 = a * a * a * a * a, b5 = b * b * b * b * b, c5 = c * c * c * c * c;
    if (t >= 2.f) return a5 / 120.f;
    if (t >= 1.f) return (a5 - 6.f * b5) / 120.f;
    return (a5 - 6.f * b5 + 15.f * c5) / 120.f;
}

// in place: samples -> B-spline coefficients (degree 3 or 5), zero-extended boundary (Unser's recursive filters)
HR_HD void bspline_prefilter_row(float* c, int n, int degree) {
    if (n < 2) return;
    const float z3[1] = {-0.267949192431123f};                               // sqrt 3 - 2
    const float z5[2] = {-0.430575347099973f, -0.0430962882032647f};
    const float* zs = degree == 5 ? z5 : z3;
    const int np = degree == 5 ? 2 : 1;
    const float gain = degree == 5 ? 120.f : 6.f;
    for (int i = 0; i < n; ++i) c[i] *= gain;
    for (int q = 0; q < np; ++q) {
        const float z = zs[q];
        for (int i = 1; i < n; ++i) c[i] += z * c[i - 1];                    // causal (zero history)
        c[n - 1] = z / (z * z - 1.f) * c[n - 1];                             // anticausal start (zero tail)
        for (int i = n - 2; i >= 0; --i) c[i] = z * (c[i + 1] - c[i]);
    }
}

template <int K>
HR_HD float sample_row(const float* row, int n, float u) {
    const float fu = floorf(u);
    const int i0 = static_cast<int>(fu);
    const float w = u - fu;
    if (K == Linear) {
        const float v0 = tap(row, n, i0), v1 = tap(row, n, i0 + 1);
        return v0 + w * (v1 - v0);
    } else if (K == Cubic) {
        return tap(row, n, i0 - 1) * keys(w + 1.f) + tap(row, n, i0) * keys(w) + tap(row, n, i0 + 1) * keys(1.f - w) +
               tap(row, n, i0 + 2) * keys(2.f - w);
    } else if (K == BSpline3) {
        return tap(row, n, i0 - 1) * bspline3(w + 1.f) + tap(row, n, i0) * bspline3(w) +
               tap(row, n, i0 + 1) * bspline3(1.f - w) + tap(row, n, i0 + 2) * bspline3(2.f - w);
    } else if (K == BSpline5) {
        float acc = 0.f;
        for (int k = -2; k <= 3; ++k) acc += tap(row, n, i0 + k) * bspline5(w - static_cast<float>(k));
        return acc;
    } else {
        float acc = 0.f, ws = 0.f;
        for (int k = -2; k <= 3; ++k) {
            const float wk = lanczos3(w - static_cast<float>(k));
            acc += tap(row, n, i0 + k) * wk;
            ws += wk;
        }
        return acc / ws;
    }
}

}  // namespace helirec::interp_core
