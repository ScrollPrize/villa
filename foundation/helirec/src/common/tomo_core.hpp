#pragma once
// Ray-driven projector with exact chord lengths (Siddon-style) and the per-element arithmetic of the whole-slice
// methods (tomo.hpp), shared by the CPU and CUDA backends.
//
// Geometry (the exact backprojector's, kernels.hpp): pixel (ix, iy) is the unit square centred at
// X = ix - c, Y = iy - c; angle a; ray d is the line  X cos a + Y sin a + rot = d. The unknowns are the pixels whose
// centre lies in the reconstruction disk (radius R; the pipeline clears the outside anyway). The weight of pixel j in ray i
// is the length of the line inside the pixel. The forward
// projector walks each ray along its major axis; the backprojector gathers, for each pixel and angle, the <= 2 rays
// crossing it. Both evaluate the same per-step cell selection (chord_cells), including its half-open convention for
// rays exactly on a pixel boundary, so they are exact transposes.
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "hd_math.hpp"

namespace helirec::tomo_core {

// chord(tau): length of the line  X cos a + Y sin a = t  inside the unit square centred at the origin, tau = |t|:
// a trapezoid in tau with plateau 1 / max(|cos|, |sin|) (chord_k with the per-angle constants AngleC)
// Image domain: the box [ox, ox + W) x [oy, oy + H) of the global N x N grid (the whole grid, or an ROI box), stored
// row-major [iy - oy][ix - ox]; pixels whose centre lies outside the reconstruction disk (radius R) are not unknowns
// (clip = 1; clip = 0 when the box lies inside the disk).
struct Geom {
    int N = 0;            // global grid N x N
    int n_det = 0;
    float c = 0.f;        // pixel-grid centre ((N - 1) / 2)
    float rot = 0.f;      // detector position of the rotation axis
    float R = 1e30f;      // reconstruction disk radius
    int ox = 0, oy = 0, W = 0, H = 0;
    int clip = 1;
};

inline Geom box_geom(Geom g, int x0, int x1, int y0, int y1) {
    g.ox = x0; g.oy = y0; g.W = x1 - x0; g.H = y1 - y0;
    float m2 = 0.f;
    for (int cx : {x0, x1 - 1})
        for (int cy : {y0, y1 - 1}) {
            const float X = cx - g.c, Y = cy - g.c;
            m2 = X * X + Y * Y > m2 ? X * X + Y * Y : m2;
        }
    g.clip = g.R < 2.f * g.N && m2 > g.R * g.R;
    return g;
}

// per-angle constants of the chord function (computed once per angle)
struct AngleC {
    float ca, sa;         // cos, sin
    float inv_hi;         // plateau 1 / max(|cos|, |sin|)
    float inv_hilo;       // slope 1 / (|cos| |sin|)
    float edge, half;     // chord > 0 for tau < edge; plateau for tau <= half
    int walk_cols;        // |sin| >= |cos|: walk columns (X), minor axis Y
    float pad;
};

// projector models: Chord (the line integral through the pixel at the detector cell centre) or SF (separable
// footprint, Long-Fessler-Balter 2010: the same trapezoid integrated over the detector cell width)
enum Projector : int { Chord = 0, SF = 1 };

HR_HD AngleC angle_constants(float ca, float sa) {
    AngleC k;
    k.ca = ca;
    k.sa = sa;
    const float ac = fabsf(ca), as = fabsf(sa);
    const float hi = ac > as ? ac : as, lo = ac > as ? as : ac;
    k.inv_hi = 1.f / hi;
    k.edge = 0.5f * (hi + lo);
    k.half = lo < 1e-7f ? k.edge : 0.5f * (hi - lo);
    k.inv_hilo = lo < 1e-7f ? 1e30f : 1.f / (hi * lo);
    k.walk_cols = as >= ac;
    k.pad = 0.f;
    return k;
}

HR_HD float chord_k(float tau, const AngleC& k) {
    if (tau >= k.edge) return 0.f;
    if (tau <= k.half) return k.inv_hi;
    return (k.edge - tau) * k.inv_hilo;
}  // (lo ~ 0: half = edge, so the slope branch is never taken)

// integral of chord over [0, a], a >= 0 (half the trapezoid's area 1 is reached at a = edge)
HR_HD float chord_half_integral(float a, const AngleC& k) {
    if (a <= k.half) return a * k.inv_hi;
    if (a < k.edge) {
        const float e = k.edge - a;
        return 0.5f - 0.5f * e * e * k.inv_hilo;
    }
    return 0.5f;
}
// SF weight of a pixel whose centre projects at tau (bins) from the cell centre: integral of chord over the cell
HR_HD float sf_weight(float tau, const AngleC& k) {
    const float a = tau + 0.5f, b = tau - 0.5f;
    const float Ga = a >= 0.f ? chord_half_integral(a, k) : -chord_half_integral(-a, k);
    const float Gb = b >= 0.f ? chord_half_integral(b, k) : -chord_half_integral(-b, k);
    return Ga - Gb;
}

HR_HD bool in_disk(float X, float Y, const Geom& g) { return g.R >= 2.f * g.N || X * X + Y * Y <= g.R * g.R; }

// [lo, hi) of the major-axis indices whose pixels ray t0 may cross inside the box and the disk (empty: lo >= hi)
HR_HD void ray_span(const Geom& g, const AngleC& k, float t0, int& lo, int& hi) {
    lo = hi = 0;
    float L = 4.f * g.N;   // no disk: longer than any chord of the grid
    if (g.R < 2.f * g.N) {
        const float rr = g.R + 2.f, L2 = rr * rr - t0 * t0;
        if (L2 <= 0.f) return;
        L = sqrtf(L2);
    }
    // major coordinate (X when walking columns) of the disk chord: closest point t0 (ca, sa), direction (-sa, ca)
    const float pa = k.walk_cols ? k.ca : k.sa, pb = k.walk_cols ? k.sa : k.ca;   // major = t0 pa -+ L |pb|
    float mlo = t0 * pa - L * fabsf(pb), mhi = t0 * pa + L * fabsf(pb);
    // minor-axis box limits -> major interval: major = (t0 - minor pb) / pa
    const int m0 = k.walk_cols ? g.oy : g.ox, mn = k.walk_cols ? g.H : g.W;
    const float qlo = m0 - g.c - 2.f, qhi = m0 + mn - 1 - g.c + 2.f;   // (2: the SF support reaches 1.2 bins)
    if (fabsf(pa) > 1e-6f) {
        const float u = (t0 - qlo * pb) / pa, v = (t0 - qhi * pb) / pa;
        const float ulo = u < v ? u : v, uhi = u < v ? v : u;
        mlo = mlo > ulo ? mlo : ulo;
        mhi = mhi < uhi ? mhi : uhi;
    } else {
        const float q = t0 / pb;   // the ray runs along the major axis at minor coordinate q
        if (q < qlo || q > qhi) return;
    }
    const int M0 = k.walk_cols ? g.ox : g.oy, MN = k.walk_cols ? g.W : g.H;
    const int a = static_cast<int>(floorf(mlo + g.c)) - 1, b = static_cast<int>(ceilf(mhi + g.c)) + 2;
    lo = a < M0 ? M0 : a;
    hi = b > M0 + MN ? M0 + MN : b;
}

// forward: sum over the box pixels of chord x x[pixel] for ray (k, d); VAL: accumulate x (else only len2);
// L2: also the ray's norm: 1 sum of squared chords (|a_i|^2), 2 sum of chords (row sum). Walks the major axis M (X when walking columns); within one major step the ray
// covers a minor interval of width 2h <= 1, i.e. one or two pixels, and its length 1 / |pb| splits between them in
// proportion (the exact chord lengths, = chord_k). xT (optional): the box transposed ([ix - ox][iy - oy]), read when
// walking columns so neighbouring rays read neighbouring addresses. lane / nl: this thread takes steps
// s0 + lane, s0 + lane + nl, ... (a warp per ray).
// project_ray_steps: the walk over major indices [mlo, mhi) (step nl from mlo), counting only [s0, s1) (the ray's span):
// a warp walking a common range reads neighbouring addresses at each step.
// rd(index, cols) -> pixel value: index into the box image laid out by `cols` (columns walk: the transposed layout
// [ix - ox][iy - oy] when the reader has it (rd.transposed()), else [iy - oy][ix - ox])
struct BoxReader {
    const float *x, *xT;
    HR_HD bool transposed() const { return xT != nullptr; }
    HR_HD float operator()(std::size_t i, bool cols) const { return (cols && xT) ? xT[i] : x[i]; }
};

// Chord model, one major-axis step: ray t0 at major coordinate Mc covers the minor interval [yc - h, yc + h] (2h <= 1),
// i.e. cells i0 and i1 (i1 == i0: one cell), with chord lengths w0 + w1 = 1 / max(|cos|, |sin|) split in proportion.
// Cell i covers [i - c - 1/2, i - c + 1/2) (half-open: a ray on a boundary belongs to the upper cell). The forward
// projector and the backprojector both call this, so their matrices are identical, boundary rays included.
// Scalar-templated chord geometry is also used by the Fresnel wave sampler.
// Keep float aliases for the existing reconstruction kernels.
template<class T> struct ChordRayT { T t0i, pai, h, L, Lf; };
using ChordRay = ChordRayT<float>;
template<class T>
HR_HD ChordRayT<T> chord_ray_values(T ca, T sa, bool cols, T t0) {
    const T pa = cols ? ca : sa, pb = cols ? sa : ca;
    const T inv = T(1) / pb, pai = pa * inv, h = T(.5) * fabs(pai), L = T(1) / fabs(pb);
    return {t0 * inv, pai, h, L, h > 0 ? L / (T(2) * h) : T(0)};
}
HR_HD ChordRay chord_ray(const AngleC& k, float t0) {
    return chord_ray_values(k.ca, k.sa, bool(k.walk_cols), t0);
}
template<class T>
HR_HD void chord_cells(const ChordRayT<T>& r, T c, T Mc, int& i0, int& i1, T& w0, T& w1) {
    const T yc = r.t0i - Mc * r.pai, ymin = yc - r.h;
    i0 = static_cast<int>(floor(ymin + c + T(.5)));
    i1 = static_cast<int>(floor(yc + r.h + c + T(.5)));
    w0 = r.L; w1 = 0;
    if (i1 != i0) {
        w0 = (T(i1) - c - T(.5) - ymin) * r.Lf;
        w0 = w0 < 0 ? T(0) : (w0 > r.L ? r.L : w0);
        w1 = r.L - w0;
    }
}

template <bool VAL, int L2, int PJ, class RD>
HR_HD float project_ray_steps_rd(const RD& rd, const Geom& g, const AngleC& k, float t0, int s0, int s1, int mlo, int mhi,
                                 int nl, float* len2) {
    const bool cols = k.walk_cols;
    const float pa = cols ? k.ca : k.sa, pb = cols ? k.sa : k.ca;   // minor(M) = (t0 - M pa) / pb, |pb| >= |pa|
    const float inv = 1.f / pb;
    const float t0i = t0 * inv, pai = pa * inv;
    const float reach = (k.edge + 0.5f) * fabsf(inv);   // SF: minor half-range of the pixels touching the cell
    const ChordRay cr = chord_ray(k, t0);
    const int q0 = cols ? g.oy : g.ox, q1 = q0 + (cols ? g.H : g.W), m0 = cols ? g.ox : g.oy;
    const bool tr = cols && rd.transposed();
    const std::size_t ms = cols ? (tr ? g.H : 1) : g.W, qs = cols ? (tr ? 1 : g.W) : 1;
    float acc = 0.f, l2 = 0.f;
    for (int M = mlo; M < mhi; M += nl) {
        if (M < s0 || M >= s1) continue;
        const float Mc = static_cast<float>(M) - g.c;
        const float yc = t0i - Mc * pai;                 // the ray's minor coordinate at this step
        const std::size_t base = static_cast<std::size_t>(M - m0) * ms;
        if (PJ == SF) {
            int i0 = static_cast<int>(ceilf(yc + g.c - reach)), i1 = static_cast<int>(floorf(yc + g.c + reach));
            i0 = i0 < q0 ? q0 : i0;
            i1 = i1 >= q1 ? q1 - 1 : i1;
            for (int i = i0; i <= i1; ++i) {
                const float Q = static_cast<float>(i) - g.c;
                if (g.clip && !in_disk(Mc, Q, g)) continue;
                const float w = sf_weight(pb * (Q - yc), k);
                if (VAL) acc += w * rd(base + static_cast<std::size_t>(i - q0) * qs, cols);
                if (L2) l2 += L2 == 1 ? w * w : w;
            }
            continue;
        }
        int i0, i1;
        float w0, w1;
        chord_cells(cr, g.c, Mc, i0, i1, w0, w1);
        if (i0 >= q0 && i0 < q1 && (!g.clip || in_disk(Mc, static_cast<float>(i0) - g.c, g))) {
            if (VAL) acc += w0 * rd(base + static_cast<std::size_t>(i0 - q0) * qs, cols);
            if (L2) l2 += L2 == 1 ? w0 * w0 : w0;
        }
        if (i1 != i0 && i1 >= q0 && i1 < q1 && (!g.clip || in_disk(Mc, static_cast<float>(i1) - g.c, g))) {
            if (VAL) acc += w1 * rd(base + static_cast<std::size_t>(i1 - q0) * qs, cols);
            if (L2) l2 += L2 == 1 ? w1 * w1 : w1;
        }
    }
    if (L2) *len2 = l2;
    return acc;
}

template <bool VAL, int L2, int PJ = Chord>
HR_HD float project_ray_steps(const float* x, const float* xT, const Geom& g, const AngleC& k, float t0, int s0, int s1,
                              int mlo, int mhi, int nl, float* len2) {
    return project_ray_steps_rd<VAL, L2, PJ>(BoxReader{x, xT}, g, k, t0, s0, s1, mlo, mhi, nl, len2);
}

template <bool VAL, int L2, int PJ = Chord>
HR_HD float project_ray(const float* x, const float* xT, const Geom& g, const AngleC& k, int d, float* len2, int lane = 0,
                        int nl = 1) {
    const float t0 = static_cast<float>(d) - g.rot;
    int s0, s1;
    ray_span(g, k, t0, s0, s1);
    return project_ray_steps<VAL, L2, PJ>(x, xT, g, k, t0, s0, s1, s0 + lane, s1, nl, len2);
}

// whether ray (k, d) crosses the box (and the disk)
HR_HD bool ray_hits(const Geom& g, const AngleC& k, int d) {
    int s0, s1;
    ray_span(g, k, static_cast<float>(d) - g.rot, s0, s1);
    return s1 > s0;
}

// backward (adjoint) for one pixel (inside the disk) and one angle: the rays that may cross the pixel are bins
// floor(u0) and floor(u0) + 1 (chord support |u0 - d| <= edge <= 1 / sqrt 2); each one's weight for this pixel is the
// forward projector's (chord_cells at the pixel's major index)
HR_HD float chord_pixel_weight(const Geom& g, const AngleC& k, int d, float X, float Y) {
    const float Mf = k.walk_cols ? X : Y, Qf = k.walk_cols ? Y : X;
    const int M = static_cast<int>(floorf(Mf + g.c + 0.5f)), Q = static_cast<int>(floorf(Qf + g.c + 0.5f));
    int i0, i1;
    float w0, w1;
    chord_cells(chord_ray(k, static_cast<float>(d) - g.rot), g.c, static_cast<float>(M) - g.c, i0, i1, w0, w1);
    return Q == i0 ? w0 : (Q == i1 ? w1 : 0.f);
}
template <int PJ = Chord>
HR_HD float backproject_pixel(const float* yrow, const Geom& g, float X, float Y, const AngleC& k) {
    const float u0 = X * k.ca + Y * k.sa + g.rot;
    const float f = floorf(u0), fr = u0 - f;
    const int d = static_cast<int>(f);
    if (PJ == SF) {  // cells d - 1 .. d + 2 (the SF support is |u0 - cell| < edge + 1/2 <= 1.21)
        float acc = 0.f;
        for (int j = -1; j <= 2; ++j) {
            const int dj = d + j;
            if (dj >= 0 && dj < g.n_det) acc += yrow[dj] * sf_weight(fr - static_cast<float>(j), k);
        }
        return acc;
    }
    (void)fr;
    float acc = 0.f;
    if (d >= 0 && d < g.n_det) acc += chord_pixel_weight(g, k, d, X, Y) * yrow[d];
    if (d + 1 >= 0 && d + 1 < g.n_det) acc += chord_pixel_weight(g, k, d + 1, X, Y) * yrow[d + 1];
    return acc;
}

// ---------------------------------------------------------------- gridrec: PSWF tables
// SUM(coefs(k) * P(2k, x), k = 0..n/2), P(j, x) the Legendre polynomials
inline float legendre(int n, const float* coefs, float x) {
    float penult = 1.f, last = x, y = coefs[0];
    for (int j = 2; j <= n; ++j) {
        const float mxlast = -(x * last);
        const float cur = -(2 * mxlast + penult) + (penult + mxlast) / static_cast<float>(j);
        if (!(j & 1)) y += cur * coefs[j >> 1];
        penult = last;
        last = cur;
    }
    return y;
}

}  // namespace helirec::tomo_core
