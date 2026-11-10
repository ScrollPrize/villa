#pragma once
// Projector pairs of the whole-slice methods (tomo.hpp) and their per-element arithmetic, shared by the CPU and CUDA
// backends: Chord (exact chord lengths, Siddon-style) and SF (separable footprint).
//
// Geometry (the exact backprojector's, kernels.hpp): pixel (ix, iy) is the unit square centred at
// X = ix - c, Y = iy - c; angle a; ray d is the line  X cos a + Y sin a + rot = d. The unknowns are the pixels whose
// centre lies in the reconstruction disk (radius R; the pipeline clears the outside anyway).
// Chord: the weight of pixel j in ray i is the length of the line inside the pixel. The forward projector walks each
// ray along its major axis; the backprojector gathers, for each pixel and angle, the <= 2 rays crossing it. Both
// evaluate the same per-step cell selection (chord_cells), including its half-open convention for rays exactly on a
// pixel boundary, so they are exact transposes.
// SF: the weight is the chord trapezoid integrated over the detector cell (see "separable footprint" below).
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
    float inv_hilo;       // slope 1 / (|cos| |sin|) (0 for an axis-aligned angle: no slopes)
    float edge, half;     // chord > 0 for tau < edge; plateau for tau <= half
    int walk_cols;        // |sin| >= |cos|: walk columns (X), minor axis Y
    float hih;            // inv_hilo / 2: the footprint's tails are hih (edge - tau)^2
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
    k.inv_hilo = lo < 1e-7f ? 0.f : 1.f / (hi * lo);
    k.walk_cols = as >= ac;
    k.hih = 0.5f * k.inv_hilo;
    return k;
}

HR_HD float chord_k(float tau, const AngleC& k) {
    if (tau >= k.edge) return 0.f;
    if (tau <= k.half) return k.inv_hi;
    return (k.edge - tau) * k.inv_hilo;
}  // (lo ~ 0: half = edge, so the slope branch is never taken)

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
    const float qlo = m0 - g.c - 2.f, qhi = m0 + mn - 1 - g.c + 2.f;   // (2: the SF footprint reaches 1.21 px)
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

template <bool VAL, int L2, class RD>
HR_HD float project_ray_steps_rd(const RD& rd, const Geom& g, const AngleC& k, float t0, int s0, int s1, int mlo, int mhi,
                                 int nl, float* len2) {
    const bool cols = k.walk_cols;
    const ChordRay cr = chord_ray(k, t0);
    const int q0 = cols ? g.oy : g.ox, q1 = q0 + (cols ? g.H : g.W), m0 = cols ? g.ox : g.oy;
    const bool tr = cols && rd.transposed();
    const std::size_t ms = cols ? (tr ? g.H : 1) : g.W, qs = cols ? (tr ? 1 : g.W) : 1;
    float acc = 0.f, l2 = 0.f;
    for (int M = mlo; M < mhi; M += nl) {
        if (M < s0 || M >= s1) continue;
        const float Mc = static_cast<float>(M) - g.c;
        const std::size_t base = static_cast<std::size_t>(M - m0) * ms;
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

template <bool VAL, int L2>
HR_HD float project_ray_steps(const float* x, const float* xT, const Geom& g, const AngleC& k, float t0, int s0, int s1,
                              int mlo, int mhi, int nl, float* len2) {
    return project_ray_steps_rd<VAL, L2>(BoxReader{x, xT}, g, k, t0, s0, s1, mlo, mhi, nl, len2);
}

template <bool VAL, int L2>
HR_HD float project_ray(const float* x, const float* xT, const Geom& g, const AngleC& k, int d, float* len2, int lane = 0,
                        int nl = 1) {
    const float t0 = static_cast<float>(d) - g.rot;
    int s0, s1;
    ray_span(g, k, t0, s0, s1);
    return project_ray_steps<VAL, L2>(x, xT, g, k, t0, s0, s1, s0 + lane, s1, nl, len2);
}

// whether ray (k, d) crosses the box (and the disk)
HR_HD bool ray_hits(const Geom& g, const AngleC& k, int d) {
    int s0, s1;
    ray_span(g, k, static_cast<float>(d) - g.rot, s0, s1);
    return s1 > s0;
}

// ---------------------------------------------------------------- separable footprint (SF)
// A pixel's footprint on the detector is the chord trapezoid about its projected centre u (area 1); its weight in
// cell d is the footprint's integral over the cell [d - 1/2, d + 1/2]: F(d + 1/2 - u) - F(d - 1/2 - u), with F the
// footprint's cumulative integral (0 below -edge, 1 above +edge). Both projectors evaluate F at the cell boundaries,
// which neighbouring cells share:
//   backward  a pixel reaches four cells through three boundaries (backproject_pixel);
//   forward   one walk per boundary b accumulates the image below it; a cell is the difference of its two
//             boundaries. Per major-axis step the walk meets the pixels in the order of their projected centres
//             (pitch hi = max(|cos|, |sin|) bins); r is the first one not wholly below b, and r, r + 1 are the only
//             ones b cuts (2 edge <= 2 hi). With c0, c1 their parts below b, and r1 the first pixel not wholly
//             below the next boundary (r <= r1 <= r + 2),
//                 cell = sum_{r <= n < r1} x_n + (x c)(next boundary) - (x c)(b),
//             so the walk keeps two sums, A = sum x c and E = sum x ([n < r1] - c): cell d = A(d + 1) + E(d).
//             (r1 > r: the boundaries are 1 / hi >= 1 pixel apart.)
//             Every term is of the size of the pixel values (no differences of running totals), and r1 is what the
//             next boundary finds as its own r (exact fixed-point positions).
// The walks read copies of the image that are zero outside the box and the disk, with kSfPad zero pixels at both ends
// of each line (SfImage), so no step tests the domain.
constexpr int kSfPad = 8;   // > the 2 px slack of ray_span on either axis + the footprint's reach (2.3 px)

struct SfImage {
    const float* rows;   // [iy - oy][kSfPad + ix - ox], lines of W + 2 kSfPad: walks along rows (major axis Y)
    const float* cols;   // [ix - ox][kSfPad + iy - oy], lines of H + 2 kSfPad: walks along columns (major axis X)
};

// Boundary positions are kept in fixed point (2^-32 pixels): float32 coordinates resolve 1e-4 pixel at the edge of a
// 5000-pixel grid, and a cell is the difference of two boundaries, so their spacing has to be exact. v(beta, M) is the
// minor coordinate (pixels from the grid centre's pixel, along the detector's direction) up to which pixels lie wholly
// below boundary beta (detector position beta - 1/2) at major index M:  v = vb0 + beta inv - (2 M - (N - 1)) paih.
struct SfAngle {
    std::int64_t vb0, inv, paih;   // boundary 0 at the grid centre; boundary pitch 1 / hi; half the slope pa / hi
};
inline SfAngle sf_angle_constants(const Geom& g, const AngleC& k) {
    const double pa = k.walk_cols ? k.ca : k.sa, pb = k.walk_cols ? k.sa : k.ca;   // u = major pa + minor pb + rot
    const double hi = std::fabs(pb), sig = pb < 0 ? -1.0 : 1.0, one = 4294967296.0;
    const double c = 0.5 * (g.N - 1);
    SfAngle s;
    s.vb0 = std::llround(((-0.5 - static_cast<double>(g.rot) - static_cast<double>(k.edge)) / hi + sig * (c - std::floor(c))) * one);
    s.inv = std::llround(one / hi);
    s.paih = std::llround(0.5 * one * pa / hi);
    return s;
}

struct SfWalk {          // one boundary of one angle
    std::int64_t v0, dv; // v at major index 0; decrease per major step
    std::uint32_t pitch; // boundary pitch - 1 pixel (hi <= 1: the next boundary's r is r + 1, or r + 2 on a carry)
    float hi, his, lo, inv_hi, edge, hih;   // his = hi 2^-32
    int sig;             // +-1: the minor index increases / decreases with the detector coordinate
};
HR_HD SfWalk sf_walk(const Geom& g, const AngleC& k, const SfAngle& sa, int beta) {
    SfWalk w;
    w.v0 = sa.vb0 + beta * sa.inv + (g.N - 1) * sa.paih;
    w.dv = 2 * sa.paih;
    w.pitch = static_cast<std::uint32_t>(sa.inv);
    w.sig = (k.walk_cols ? k.sa : k.ca) < 0.f ? -1 : 1;
    w.hi = fabsf(k.walk_cols ? k.sa : k.ca);
    w.his = w.hi * 2.3283064365386963e-10f;
    w.lo = k.edge - k.half;
    w.inv_hi = k.inv_hi;
    w.edge = k.edge;
    w.hih = k.hih;
    return w;
}
// one major-axis step of the walk; p0: the line's pixel at minor index floor(c) (SfImage layout); V: v at this index;
// SIG: the walk's sig
template <int SIG>
HR_HD void sf_step(const SfWalk& w, const float* p0, std::int64_t V, float& A, float& E) {
    const std::uint32_t frac = static_cast<std::uint32_t>(V);            // v - floor(v)
    const int r = static_cast<int>(V >> 32) + 1;                          // first pixel not wholly below the boundary
    const bool two = static_cast<std::uint32_t>(frac + w.pitch) < frac;   // pixel r + 1 is wholly below the next one
    const float e0 = w.hi - static_cast<float>(frac) * w.his;   // from the boundary to the upper end of pixel r's footprint
    const float e1 = w.lo - e0;                 // > 0: the boundary cuts pixel r's upper slope and pixel r + 1's lower
    const float q = e0 * e0 * w.hih, l = (w.edge - e0) * w.inv_hi;
    const float c0 = e1 > 0.f ? 1.f - q : 0.5f + l;   // the part of pixel r below the boundary
    const float d0 = e1 > 0.f ? q : 0.5f - l;         // and above it
    const float c1 = e1 > 0.f ? e1 * e1 * w.hih : 0.f;
    const float* p = SIG > 0 ? p0 + r : p0 - r;
    const float x0 = p[0], x1 = p[SIG];
    A = x1 * c1 + (x0 * c0 + A);
    E = x1 * ((two ? 1.f : 0.f) - c1) + (x0 * d0 + E);
}
// the walk of one boundary over the major indices [mlo, mhi), counting only its own span [s0, s1) (a warp walks its
// boundaries' common range in step)
template <int SIG>
HR_HD void sf_walk_steps(const SfWalk& w, const float* line, std::size_t stride, int s0, int s1, int mlo, int mhi,
                         float& A, float& E) {
    const unsigned ns = s1 > s0 ? static_cast<unsigned>(s1 - s0) : 0u;   // (an empty span may have s1 < s0)
    std::int64_t V = w.v0 - mlo * w.dv;
    for (int M = mlo; M < mhi; ++M, V -= w.dv, line += stride)
        if (static_cast<unsigned>(M - s0) < ns) sf_step<SIG>(w, line, V, A, E);
}
HR_HD void sf_boundary_steps(const SfImage& im, const Geom& g, const AngleC& k, const SfWalk& w, int s0, int s1,
                             int mlo, int mhi, float& A, float& E) {
    if (mhi <= mlo) return;
    const bool cols = k.walk_cols;
    const int m0 = cols ? g.ox : g.oy, q0 = cols ? g.oy : g.ox;
    const std::size_t stride = static_cast<std::size_t>((cols ? g.H : g.W) + 2 * kSfPad);
    const float* line = (cols ? im.cols : im.rows) + (kSfPad - q0 + (g.N - 1) / 2) + static_cast<std::size_t>(mlo - m0) * stride;
    if (w.sig > 0) sf_walk_steps<1>(w, line, stride, s0, s1, mlo, mhi, A, E);
    else sf_walk_steps<-1>(w, line, stride, s0, s1, mlo, mhi, A, E);
}
HR_HD void sf_boundary(const SfImage& im, const Geom& g, const AngleC& k, const SfAngle& sa, int beta, float& A, float& E) {
    int s0, s1;
    ray_span(g, k, static_cast<float>(beta) - 0.5f - g.rot, s0, s1);
    A = E = 0.f;
    if (s1 > s0) sf_boundary_steps(im, g, k, sf_walk(g, k, sa, beta), s0, s1, s0, s1, A, E);
}
// value of pixel (ix, iy) of the box in the SfImage copies
HR_HD float sf_image_value(const float* x, const Geom& g, int ix, int iy) {
    const float X = g.ox + ix - g.c, Y = g.oy + iy - g.c;
    return (!g.clip || in_disk(X, Y, g)) ? x[static_cast<std::size_t>(iy) * g.W + ix] : 0.f;
}

// backward (adjoint) for one pixel (inside the disk) and one angle. Chord: the rays that may cross the pixel are bins
// floor(u0) and floor(u0) + 1 (chord support |u0 - d| <= edge <= 1 / sqrt 2); each one's weight for this pixel is the
// forward projector's (chord_cells at the pixel's major index). SF: cells d - 1 .. d + 2, d = floor(u0) (the
// footprint reaches edge + 1/2 <= 1.21 bins), through the boundaries at d - 1/2, d + 1/2, d + 3/2.
HR_HD float chord_pixel_weight(const Geom& g, const AngleC& k, int d, float X, float Y) {
    const float Mf = k.walk_cols ? X : Y, Qf = k.walk_cols ? Y : X;
    const int M = static_cast<int>(floorf(Mf + g.c + 0.5f)), Q = static_cast<int>(floorf(Qf + g.c + 0.5f));
    int i0, i1;
    float w0, w1;
    chord_cells(chord_ray(k, static_cast<float>(d) - g.rot), g.c, static_cast<float>(M) - g.c, i0, i1, w0, w1);
    return Q == i0 ? w0 : (Q == i1 ? w1 : 0.f);
}
// SF: with fr = u0 - d, the parts of the footprint below the three boundaries are t0 = F(-1/2 - fr) (a lower tail),
// c1 = F(1/2 - fr) and 1 - t2 = F(3/2 - fr) (an upper tail); tails are hih (edge - distance)^2, and
// F(s) = 1/2 + s / hi -+ hih (|s| - half)^2 beyond the plateau (|s| <= 1/2 <= edge).
// -> weights t0, w0, w1, t2 of cells d - 1 .. d + 2
HR_HD void sf_pixel_weights(float fr, const AngleC& k, float& t0, float& w0, float& w1, float& t2) {
    const float e5 = k.edge - 0.5f;
    const float a0 = fmaxf(e5 - fr, 0.f), a2 = fmaxf(e5 + (fr - 1.f), 0.f);
    t0 = a0 * a0 * k.hih;
    t2 = a2 * a2 * k.hih;
    const float s = 0.5f - fr, m = fmaxf(fabsf(s) - k.half, 0.f);
    const float q = m * m * k.hih;
    const float c1 = 0.5f + s * k.inv_hi - (s < 0.f ? -q : q);
    w0 = c1 - t0;
    w1 = (1.f - t2) - c1;
}
// u0 in [1, n_det - 2): the four cells are on the detector
HR_HD float sf_backproject_inner(const float* yrow, float u0, const AngleC& k, float acc) {
    const float f = floorf(u0);
    float t0, w0, w1, t2;
    sf_pixel_weights(u0 - f, k, t0, w0, w1, t2);
    const float* p = yrow + static_cast<int>(f);
    return acc + p[-1] * t0 + p[0] * w0 + p[1] * w1 + p[2] * t2;
}
// u0 anywhere
HR_HD float sf_backproject_edge(const float* yrow, int n_det, float u0, const AngleC& k, float acc) {
    if (!(u0 > -2.f && u0 < static_cast<float>(n_det + 1))) return acc;   // no cell within the footprint's reach
    const float f = floorf(u0);
    const int d = static_cast<int>(f);
    float t0, w0, w1, t2;
    sf_pixel_weights(u0 - f, k, t0, w0, w1, t2);
    if (d - 1 >= 0 && d - 1 < n_det) acc += yrow[d - 1] * t0;
    if (d >= 0 && d < n_det) acc += yrow[d] * w0;
    if (d + 1 >= 0 && d + 1 < n_det) acc += yrow[d + 1] * w1;
    if (d + 2 >= 0 && d + 2 < n_det) acc += yrow[d + 2] * t2;
    return acc;
}
template <int PJ = Chord>
HR_HD float backproject_pixel(const float* yrow, const Geom& g, float X, float Y, const AngleC& k) {
    const float u0 = X * k.ca + Y * k.sa + g.rot;
    if (PJ == SF)
        return u0 >= 1.f && u0 < static_cast<float>(g.n_det - 2) ? sf_backproject_inner(yrow, u0, k, 0.f)
                                                                : sf_backproject_edge(yrow, g.n_det, u0, k, 0.f);
    const int d = static_cast<int>(floorf(u0));
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
