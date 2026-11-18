#pragma once
// Inversion of a detector map (vc_tomography/detector_map.hpp), shared by CPU and CUDA: the same operations in the same
// order on both. In double precision the sums of the terms are contracted under CUDA (hd::dmuladd), so that the two
// backends' positions differ by roundings of double precision; in single precision nothing is contracted.
#include <cmath>

#include "hd_math.hpp"
#include "vc_tomography/detector_map.hpp"

namespace vc_tomography::map_core {

// the operations of either precision
HR_HD float o_add(float a, float b) { return hd::add(a, b); }
HR_HD float o_sub(float a, float b) { return hd::sub(a, b); }
HR_HD float o_mul(float a, float b) { return hd::mul(a, b); }
HR_HD float o_muladd(float a, float b, float c) { return hd::add(hd::mul(a, b), c); }
HR_HD double o_add(double a, double b) { return hd::dadd(a, b); }
HR_HD double o_sub(double a, double b) { return hd::dsub(a, b); }
HR_HD double o_mul(double a, double b) { return hd::dmul(a, b); }
HR_HD double o_muladd(double a, double b, double c) { return hd::dmuladd(a, b, c); }

// the largest powers of x and y among the terms
HR_HD void degrees(const DetectorMapView& m, int& dx, int& dy) {
    dx = dy = 0;
    for (int k = 0; k < m.n; ++k) {
        dx = m.i[k] > dx ? m.i[k] : dx;
        dy = m.j[k] > dy ? m.j[k] : dy;
    }
}

// what the positions of one row share, in the precision T
template <class T>
struct RowTerms {
    T y0, row_slope, slope_col, gc, gr, kc, kr;   // the linear part: p = (X + kc - Eu) gc, v = (y0 + row_slope (p - slope_col) + kr - Ev) gr
    T c_lo, c_hi, v_hi, c0, ics, v0, ivs;         // the area and the normalisation (x = (c - c0) ics: (c - c0) / cs within a rounding)
};
template <class T>
HR_HD RowTerms<T> row_terms(const DetectorMapView& m, T y0, T row_slope, T slope_col, T col_scale, T scale_col, T row_scale,
                            T scale_row) {
    RowTerms<T> r;
    r.y0 = y0;
    r.row_slope = row_slope;
    r.slope_col = slope_col;
    r.gc = static_cast<T>(1) / o_add(static_cast<T>(1), col_scale);
    r.gr = static_cast<T>(1) / o_add(static_cast<T>(1), row_scale);
    r.kc = o_mul(col_scale, scale_col);
    r.kr = o_mul(row_scale, scale_row);
    r.c_lo = -static_cast<T>(detector_map_col_margin);
    r.c_hi = static_cast<T>(m.det_cols - 1 + detector_map_col_margin);
    r.v_hi = static_cast<T>(m.det_rows - 1);
    r.c0 = static_cast<T>(m.c0);
    r.ics = static_cast<T>(1) / static_cast<T>(m.cs);
    r.v0 = static_cast<T>(m.v0);
    r.ivs = static_cast<T>(1) / static_cast<T>(m.vs);
    return r;
}

// (loops over the B positions, one operation each: the form the CPU's compilers vectorise)
// the positions (p, v) that the displacements (du, dv) give for the ideal columns X
template <class T, int B>
HR_HD void positions(const RowTerms<T>& r, const T* X, const T* du, const T* dv, T* p, T* v) {
    HR_SIMD
    for (int l = 0; l < B; ++l) p[l] = o_mul(o_sub(o_add(X[l], r.kc), du[l]), r.gc);
    HR_SIMD
    for (int l = 0; l < B; ++l)
        v[l] = o_mul(o_sub(o_add(o_add(r.y0, o_mul(r.row_slope, o_sub(p[l], r.slope_col))), r.kr), dv[l]), r.gr);
}
// the map (Eu, Ev) at the positions (p, v), held inside the detector; eu, ev: the terms' coefficients in T
template <class T, int B>
HR_HD void displacement(const DetectorMapView& m, const RowTerms<T>& r, const T* eu, const T* ev, int dx, int dy, const T* p,
                        const T* v, T* du, T* dv) {
    T x[B], y[B], mono[B], xp[detector_map_max_power + 1][B], yp[detector_map_max_power + 1][B];
    HR_SIMD
    for (int l = 0; l < B; ++l) {
        T c = p[l];
        c = c < r.c_lo ? r.c_lo : c;
        c = c > r.c_hi ? r.c_hi : c;
        x[l] = o_mul(o_sub(c, r.c0), r.ics);
    }
    HR_SIMD
    for (int l = 0; l < B; ++l) {
        T w = v[l];
        w = w < static_cast<T>(0) ? static_cast<T>(0) : w;
        w = w > r.v_hi ? r.v_hi : w;
        y[l] = o_mul(o_sub(w, r.v0), r.ivs);
    }
    HR_SIMD
    for (int l = 0; l < B; ++l) {
        xp[0][l] = yp[0][l] = static_cast<T>(1);
        xp[1][l] = x[l];
        yp[1][l] = y[l];
        du[l] = dv[l] = static_cast<T>(0);
    }
    for (int d = 2; d <= dx; ++d) {
        const T* lower = xp[d - 1];
        T* power = xp[d];
        HR_SIMD
        for (int l = 0; l < B; ++l) power[l] = o_mul(lower[l], x[l]);
    }
    for (int d = 2; d <= dy; ++d) {
        const T* lower = yp[d - 1];
        T* power = yp[d];
        HR_SIMD
        for (int l = 0; l < B; ++l) power[l] = o_mul(lower[l], y[l]);
    }
    for (int k = 0; k < m.n; ++k) {
        const T* xi = xp[m.i[k]];
        const T* yj = yp[m.j[k]];
        const T cu = eu[k], cv = ev[k];
        HR_SIMD
        for (int l = 0; l < B; ++l) mono[l] = o_mul(xi[l], yj[l]);
        HR_SIMD
        for (int l = 0; l < B; ++l) du[l] = o_muladd(cu, mono[l], du[l]);
        HR_SIMD
        for (int l = 0; l < B; ++l) dv[l] = o_muladd(cv, mono[l], dv[l]);
    }
}

// The detector positions (p, v) read for B ideal columns X of one row (a row of the gridded stack: its ideal row is
// y0 at column slope_col, y0 + row_slope (c - slope_col) at detector column c):
//   p + Eu(p, v) + col_scale (p - scale_col) = X,   v + Ev(p, v) + row_scale (v - scale_row) = y0 + row_slope (p - slope_col)
// Fixed point on (Eu, Ev), each pass solving the linear rest exactly; Eu, Ev evaluated inside the detector
// (detector_map.hpp). The passes are first made in single precision from zero (m.passes of them: what double
// precision would need from zero; single precision stops short at its own rounding, some 1e-6 pixels), then in
// double precision from there until one changes (Eu, Ev) by no more than m.settle: by then what is left is below
// detector_map_tolerance (a pass contracts by m's planned factor q, so that a change of d leaves at most
// d q / (1 - q)); at most m.passes + 1 of them, which reach the tolerance from any start within the map's bounds.
// B positions at a time, each with the same operations in the same order, the passes of all B ending together.
// further: that many passes more once the passes have settled (each takes what is left down by the planned factor
// again: for positions that are to be far inside the tolerance, row_fit's). The second form also gives the
// displacements (du, dv) = (Eu, Ev) at the positions.
template <int B>
HR_HD void source(const DetectorMapView& m, const RowTerms<double>& r, const double* X, double* p, double* v, double* du, double* dv,
                  int further = 0) {
    HR_SIMD
    for (int l = 0; l < B; ++l) du[l] = dv[l] = 0.0;
    if (m.n > 0 && m.passes > 0) {
        int dx = 0, dy = 0;
        degrees(m, dx, dy);
        {
            RowTerms<float> rf;
            rf.y0 = static_cast<float>(r.y0); rf.row_slope = static_cast<float>(r.row_slope); rf.slope_col = static_cast<float>(r.slope_col);
            rf.gc = static_cast<float>(r.gc); rf.gr = static_cast<float>(r.gr); rf.kc = static_cast<float>(r.kc); rf.kr = static_cast<float>(r.kr);
            rf.c_lo = static_cast<float>(r.c_lo); rf.c_hi = static_cast<float>(r.c_hi); rf.v_hi = static_cast<float>(r.v_hi);
            rf.c0 = static_cast<float>(r.c0); rf.ics = static_cast<float>(r.ics); rf.v0 = static_cast<float>(r.v0); rf.ivs = static_cast<float>(r.ivs);
            float eu[detector_map_max_terms], ev[detector_map_max_terms], Xf[B], pf[B], vf[B], uf[B], wf[B];
            for (int k = 0; k < m.n; ++k) {
                eu[k] = static_cast<float>(m.eu[k]);
                ev[k] = static_cast<float>(m.ev[k]);
            }
            HR_SIMD
            for (int l = 0; l < B; ++l) {
                Xf[l] = static_cast<float>(X[l]);
                uf[l] = wf[l] = 0.0f;
            }
            for (int pass = 0; pass < m.passes; ++pass) {
                positions<float, B>(rf, Xf, uf, wf, pf, vf);
                displacement<float, B>(m, rf, eu, ev, dx, dy, pf, vf, uf, wf);
            }
            HR_SIMD
            for (int l = 0; l < B; ++l) {
                du[l] = static_cast<double>(uf[l]);
                dv[l] = static_cast<double>(wf[l]);
            }
        }
        double nu[B], nv[B];
        int left = further;
        for (int pass = 0; pass <= m.passes + further; ++pass) {
            positions<double, B>(r, X, du, dv, p, v);
            displacement<double, B>(m, r, m.eu, m.ev, dx, dy, p, v, nu, nv);
            double change = 0.0;
            for (int l = 0; l < B; ++l) {
                const double a = fabs(hd::dsub(nu[l], du[l])), b = fabs(hd::dsub(nv[l], dv[l]));
                change = a > change ? a : change;
                change = b > change ? b : change;
                du[l] = nu[l];
                dv[l] = nv[l];
            }
            if (change <= m.settle && left-- <= 0) break;
        }
    }
    positions<double, B>(r, X, du, dv, p, v);
}
template <int B>
HR_HD void source(const DetectorMapView& m, double y0, double row_slope, double slope_col, double col_scale,
                  double scale_col, double row_scale, double scale_row, const double* X, double* p, double* v) {
    double du[B], dv[B];
    source<B>(m, row_terms<double>(m, y0, row_slope, slope_col, col_scale, scale_col, row_scale, scale_row), X, p, v, du, dv);
}

// The positions of a whole row of samples, X = j - shift for j = 0 .. n - 1, without solving the map at each: along
// a row the map's displacement Eu is a smooth function of j, so it is solved at the kRowFit Chebyshev nodes of the
// row and read elsewhere from the polynomial through them, wherever that polynomial is the displacement to
// kRowFitError pixels (far below detector_map_tolerance): checked at as many other points of the row, the row's
// ends among them, against the map solved there (at all these points one pass further than a sample's own solution
// takes, which leaves 1e-11 pixels where that leaves up to the tolerance). A row that fails the check (it crosses an edge of the area the
// map is held to, where Eu has a kink, or its map takes more terms) is solved sample by sample.
// row_fit: c[0 .. kRowFit) the coefficients of the powers of xi = (2 j - (n - 1)) / (n - 1), or c[0] not a number
// for a row to be solved sample by sample; then the row's kc and gc (RowTerms), kRowFitSize numbers in all.
// row_fit_taps: where the kernel of resample_rows starts for the samples j0 .. j0 + B - 1, from them.
constexpr int kRowFit = 16;
constexpr int kRowFitSize = kRowFit + 2;
constexpr double kRowFitError = 1e-9;
// the points of a row the map is solved at, as xi: the Chebyshev nodes, then as many between them with both ends;
// and the cosines that give the Chebyshev coefficients from the values at the nodes
struct RowFitTable {
    double xi[2 * kRowFit];
    double dct[kRowFit][kRowFit];
};
inline RowFitTable row_fit_table() {
    constexpr int N = kRowFit;
    constexpr double pi = 3.14159265358979323846;
    RowFitTable t;
    for (int k = 0; k < N; ++k) {
        t.xi[k] = std::cos(pi * (k + 0.5) / N);
        t.xi[N + k] = std::cos(pi * (k < N / 2 ? k : k + 1) / N);   // (1 first, -1 last)
        for (int j = 0; j < N; ++j) t.dct[j][k] = (j == 0 ? 1.0 : 2.0) / N * std::cos(pi * j * (k + 0.5) / N);
    }
    return t;
}
HR_HD void row_fit(const DetectorMapView& m, const RowTerms<double>& r, const RowFitTable& t, double shift, int n, double* c) {
    constexpr int N = kRowFit;
    const double half = 0.5 * static_cast<double>(n - 1);
    double X[2 * N], p[2 * N], v[2 * N], du[2 * N], dv[2 * N];
    for (int k = 0; k < 2 * N; ++k) X[k] = (t.xi[k] + 1.0) * half - shift;
    for (int k = 0; k < 2 * N; k += 8) source<8>(m, r, X + k, p + k, v + k, du + k, dv + k, 1);
    // the map's two holds leave the row smooth only if it stays on one side of each (the positions between the
    // points solved lie within a row and a column of them)
    bool smooth = n >= 4 * N;
    double v_lo = v[0], v_top = v[0], p_lo = p[0], p_top = p[0];
    for (int k = 1; k < 2 * N; ++k) {
        v_lo = v[k] < v_lo ? v[k] : v_lo;
        v_top = v[k] > v_top ? v[k] : v_top;
        p_lo = p[k] < p_lo ? p[k] : p_lo;
        p_top = p[k] > p_top ? p[k] : p_top;
    }
    smooth = smooth && (v_top <= -1.0 || v_lo >= r.v_hi + 1.0 || (v_lo >= 1.0 && v_top <= r.v_hi - 1.0));
    smooth = smooth && p_lo >= r.c_lo + 1.0 && p_top <= r.c_hi - 1.0;
    // the Chebyshev coefficients of the displacement at the nodes, then the powers of xi
    double T0[N], T1[N], T2[N];
    for (int q = 0; q < N; ++q) c[q] = T0[q] = T1[q] = 0.0;
    T0[0] = 1.0;
    T1[1] = 1.0;
    for (int j = 0; j < N; ++j) {
        double a = 0.0;
        for (int k = 0; k < N; ++k) a += du[k] * t.dct[j][k];
        if (j >= 2) {   // T_j = 2 xi T_(j - 1) - T_(j - 2), into T1
            for (int q = 0; q < N; ++q) T2[q] = (q > 0 ? 2.0 * T1[q - 1] : 0.0) - T0[q];
            for (int q = 0; q < N; ++q) {
                T0[q] = T1[q];
                T1[q] = T2[q];
            }
        }
        const double* T = j == 0 ? T0 : T1;
        for (int q = 0; q <= j; ++q) c[q] += a * T[q];
    }
    for (int k = N; k < 2 * N && smooth; ++k) {
        double value = c[N - 1];
        for (int q = N - 2; q >= 0; --q) value = value * t.xi[k] + c[q];
        smooth = fabs(value - du[k]) <= kRowFitError;
    }
    if (!smooth) c[0] = static_cast<double>(NAN);
    c[N] = r.kc;
    c[N + 1] = r.gc;
}
// Where the kernel of resample_rows starts for the samples j0 .. j0 + B - 1 of a row with a fit: its first tap lies
// at below[l] + rest[l] samples of the row oversampled `os` times (os x the position - half_taps), below a whole
// number and rest within 1 of 0 (sino_core::resample_values_near). The polynomial is expanded about the middle of the
// B samples: its value and slope there in double precision, the three terms after them in single precision, which
// holds them to 1e-12 pixels (over B samples they come to 1e-5 pixels; the term after those is below 1e-13). The
// position is never formed: it is the sample's own number less a small displacement, and only the displacement's
// part of the first tap needs more than whole numbers. Double precision is used once for the B samples, not for each.
template <int B>
HR_HD void row_fit_taps(const double* c, double shift, int j0, int n, int os, int half_taps, int* below, float* rest) {
    const double scale = 2.0 / static_cast<double>(n - 1), kc = c[kRowFit], gc = c[kRowFit + 1];
    const double mid = 0.5 * static_cast<double>(B - 1), jc = static_cast<double>(j0) + mid;
    const double xi = hd::dsub(hd::dmul(jc, scale), 1.0);
    const float xf = static_cast<float>(xi);
    // b_i: the polynomial's derivative of order i over i!, at xi (Horner's scheme, carried to the fourth)
    double b0 = c[kRowFit - 1], b1 = 0.0;
    float b2 = 0.0f, b3 = 0.0f, b4 = 0.0f;
    for (int q = kRowFit - 2; q >= 0; --q) {
        b4 = hd::fma(b4, xf, b3);
        b3 = hd::fma(b3, xf, b2);
        b2 = hd::fma(b2, xf, static_cast<float>(b1));
        b1 = hd::dmuladd(b1, xi, b0);
        b0 = hd::dmuladd(b0, xi, c[q]);
    }
    // the position of sample j is j - d(j): d = j (1 - gc) + (shift + Eu - kc) gc, from p = (j - shift + kc - Eu) gc;
    // at the middle sample, and its terms in t = j - the middle sample
    const double ng = hd::dsub(1.0, gc);
    const double d0 = hd::dmuladd(hd::dsub(hd::dadd(shift, b0), kc), gc, hd::dmul(jc, ng));
    const float hf = static_cast<float>(scale), gf = static_cast<float>(gc);
    const float d1 = static_cast<float>(hd::dmuladd(hd::dmul(b1, scale), gc, ng));
    const float d2 = b2 * hf * hf * gf, d3 = b3 * hf * hf * hf * gf, d4 = b4 * (hf * hf) * (hf * hf) * gf;
    // os x (j - d) - half_taps = [os j - half_taps + whole] + [part - os (d - d0)], os d0 = -(whole + part)
    const double a = hd::dmul(-static_cast<double>(os), d0), whole = floor(a + 0.5);
    const float part = static_cast<float>(hd::dsub(a, whole)), osf = static_cast<float>(os);
    const int base = os * j0 - half_taps + static_cast<int>(whole);
    HR_SIMD
    for (int l = 0; l < B; ++l) {
        const float t = static_cast<float>(l) - static_cast<float>(mid);
        const float moved = hd::fma(hd::fma(hd::fma(d4, t, d3), t, d2), t, d1) * t;   // d(j) - d0
        below[l] = base + os * l;
        rest[l] = hd::fma(-osf, moved, part);
    }
}

// The detector row read at the integer detector column c for the ideal row base + pos, relative to `base`: the v with
//   v + Ev(c, v) + row_scale (v - scale_row) = base + pos
// minus base. Single precision, as the gridding's other positions: v - base is a few rows, and the map changes by
// about 1e-3 rows per row, so that the rounding of v (6e-5 rows at row 1000) and of the terms leaves about 1e-7 rows.
// Without a scale and without terms the result is `pos` itself.
HR_HD float source_row(const DetectorMapView& m, float row_scale, float scale_row, int c, int base, float pos) {
    using namespace vc_tomography::hd;
    const float g = div(1.0f, add(1.0f, row_scale));
    const float lin = sub(pos, mul(row_scale, sub(static_cast<float>(base), scale_row)));
    float out = mul(lin, g);
    if (m.n == 0 || !m.ev_f) return out;
    const float c0 = static_cast<float>(m.c0), cs = static_cast<float>(m.cs);
    const float v0 = static_cast<float>(m.v0), vs = static_cast<float>(m.vs);
    const int c_lo = -detector_map_col_margin, c_hi = m.det_cols - 1 + detector_map_col_margin;
    const float v_hi = static_cast<float>(m.det_rows - 1);
    int dx = 0, dy = 0;
    degrees(m, dx, dy);
    float xp[detector_map_max_power + 1], yp[detector_map_max_power + 1];
    xp[0] = yp[0] = 1.0f;
    xp[1] = div(sub(static_cast<float>(c < c_lo ? c_lo : (c > c_hi ? c_hi : c)), c0), cs);
    for (int d = 2; d <= dx; ++d) xp[d] = mul(xp[d - 1], xp[1]);
    for (int pass = 0; pass < m.passes; ++pass) {
        const float v = add(static_cast<float>(base), out);
        yp[1] = div(sub(v < 0.0f ? 0.0f : (v > v_hi ? v_hi : v), v0), vs);
        for (int d = 2; d <= dy; ++d) yp[d] = mul(yp[d - 1], yp[1]);
        float dv = 0.0f;
        for (int k = 0; k < m.n; ++k) dv = add(dv, mul(m.ev_f[k], mul(xp[m.i[k]], yp[m.j[k]])));
        out = mul(sub(lin, dv), g);
    }
    return out;
}

}  // namespace vc_tomography::map_core
