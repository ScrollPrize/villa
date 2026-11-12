#pragma once
// Inversion of a detector map (helirec/detector_map.hpp), shared by CPU and CUDA: the same operations in the same
// order on both. In double precision the sums of the terms are contracted under CUDA (hd::dmuladd), so that the two
// backends' positions differ by roundings of double precision; in single precision nothing is contracted.
#include "hd_math.hpp"
#include "helirec/detector_map.hpp"

namespace helirec::map_core {

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
template <int B>
HR_HD void source(const DetectorMapView& m, double y0, double row_slope, double slope_col, double col_scale,
                  double scale_col, double row_scale, double scale_row, const double* X, double* p, double* v) {
    const RowTerms<double> r = row_terms<double>(m, y0, row_slope, slope_col, col_scale, scale_col, row_scale, scale_row);
    double du[B], dv[B];
    HR_SIMD
    for (int l = 0; l < B; ++l) du[l] = dv[l] = 0.0;
    if (m.n > 0 && m.passes > 0) {
        int dx = 0, dy = 0;
        degrees(m, dx, dy);
        {
            const RowTerms<float> rf = row_terms<float>(m, static_cast<float>(y0), static_cast<float>(row_slope),
                                                        static_cast<float>(slope_col), static_cast<float>(col_scale),
                                                        static_cast<float>(scale_col), static_cast<float>(row_scale),
                                                        static_cast<float>(scale_row));
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
        for (int pass = 0; pass <= m.passes; ++pass) {
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
            if (change <= m.settle) break;
        }
    }
    positions<double, B>(r, X, du, dv, p, v);
}

// The detector row read at the integer detector column c for the ideal row base + pos, relative to `base`: the v with
//   v + Ev(c, v) + row_scale (v - scale_row) = base + pos
// minus base. Single precision, as the gridding's other positions: v - base is a few rows, and the map changes by
// about 1e-3 rows per row, so that the rounding of v (6e-5 rows at row 1000) and of the terms leaves about 1e-7 rows.
// Without a scale and without terms the result is `pos` itself.
HR_HD float source_row(const DetectorMapView& m, float row_scale, float scale_row, int c, int base, float pos) {
    using namespace helirec::hd;
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

}  // namespace helirec::map_core
