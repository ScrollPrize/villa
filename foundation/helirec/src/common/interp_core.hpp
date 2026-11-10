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

// cubic B-spline, spelled out as nvcc contracts  2/3 - t t + 0.5 t t t  and without the division on CUDA
HR_HD float bspline3_inner(float t) {   // 0 <= t < 1
    return hd::muladd(t, hd::mul(t, hd::mul(t, 0.5f)), hd::muladd(-t, t, 2.f / 3.f));
}
HR_HD float bspline3_outer(float t) {   // 1 <= t <= 2 (2 - t is 0 or >= 1e-7: div6 is exact)
    const float u = 2.f - t;
    return hd::div6(hd::mul(u, hd::mul(u, u)));
}
HR_HD float bspline3(float t) {
    t = fabsf(t);
    if (t < 1.f) return bspline3_inner(t);
    if (t < 2.f) return bspline3_outer(t);
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

// taps of kernel K: row[i0 + tap_lo<K>() + t], t = 0 .. tap_count<K>() - 1, i0 = floor(u)
template <int K>
HR_HD int tap_count() { return K == Linear ? 2 : ((K == Cubic || K == BSpline3) ? 4 : 6); }
template <int K>
HR_HD int tap_lo() { return K == Linear ? 0 : ((K == Cubic || K == BSpline3) ? -1 : -2); }
// u in [reach_lo, n + reach_hi): some tap lies on the detector (elsewhere the sample is 0)
template <int K>
HR_HD float reach_lo() { return static_cast<float>(1 - tap_count<K>() - tap_lo<K>()); }
template <int K>
HR_HD float reach_hi() { return static_cast<float>(-tap_lo<K>()); }

// u within the kernel's reach of the detector (see reach_lo / reach_hi)
template <int K>
HR_HD float sample_row_reach(const float* row, int n, float u) {
    const float fu = floorf(u);
    const float w = u - fu;
    const int j = static_cast<int>(fu) + tap_lo<K>(), T = tap_count<K>();
    float v[6];
    if (j >= 0 && j + T <= n) {   // every tap on the detector
        for (int t = 0; t < T; ++t) v[t] = row[j + t];
    } else {
        for (int t = 0; t < T; ++t) v[t] = tap(row, n, j + t);
    }
    if (K == Linear) {
        return v[0] + w * (v[1] - v[0]);
    } else if (K == Cubic) {
        return v[0] * keys(w + 1.f) + v[1] * keys(w) + v[2] * keys(1.f - w) + v[3] * keys(2.f - w);
    } else if (K == BSpline3) {   // bspline3(w + 1), (w), (1 - w), (2 - w), each on the branch its argument takes
        const float t2 = 1.f - w;
        const float k0 = bspline3_outer(w + 1.f), k1 = bspline3_inner(w), k3 = bspline3_outer(2.f - w);
        const float k2 = t2 < 1.f ? bspline3_inner(t2) : 1.f / 6.f;
        return hd::muladd(v[3], k3, hd::muladd(v[2], k2, hd::muladd(v[1], k1, hd::mul(v[0], k0))));
    } else if (K == BSpline5) {
        float acc = 0.f;
        for (int k = -2; k <= 3; ++k) acc += v[k + 2] * bspline5(w - static_cast<float>(k));
        return acc;
    } else {
        float acc = 0.f, ws = 0.f;
        for (int k = -2; k <= 3; ++k) {
            const float wk = lanczos3(w - static_cast<float>(k));
            acc += v[k + 2] * wk;
            ws += wk;
        }
        return acc / ws;
    }
}

template <int K>
HR_HD float sample_row(const float* row, int n, float u) {
    return u >= reach_lo<K>() && u < static_cast<float>(n) + reach_hi<K>() ? sample_row_reach<K>(row, n, u) : 0.f;
}

}  // namespace helirec::interp_core
