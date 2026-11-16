// Detector map of the reconstruction: checks, the bounds its inversion is planned with, and host entry points.
#include "vc_tomography/detector_map.hpp"

#include <algorithm>
#include <cmath>
#include <string>

#include "common/detector_map_core.hpp"

namespace vc_tomography {

namespace {

double falling(int n, int k) {   // n (n - 1) .. (n - k + 1)
    double f = 1;
    for (int q = 0; q < k; ++q) f *= static_cast<double>(n - q);
    return f;
}

// d^a/dx^a d^b/dy^b of sum_k e[k] x^i[k] y^j[k] (normalised coordinates), at (x, y); crude: an upper bound of its
// magnitude over |x| <= x, |y| <= y (the sum of the terms' magnitudes)
double derivative(const DetectorMapView& m, const double* e, int a, int b, double x, double y, bool crude) {
    double s = 0;
    for (int k = 0; k < m.n; ++k) {
        if (m.i[k] < a || m.j[k] < b) continue;
        const double t = e[k] * falling(m.i[k], a) * std::pow(x, m.i[k] - a) * falling(m.j[k], b) * std::pow(y, m.j[k] - b);
        s += crude ? std::fabs(t) : t;
    }
    return s;
}

}  // namespace

DetectorMapPlan detector_map_plan(const DetectorMapView& m, double max_row_slope, double max_col_scale, double max_row_scale) {
    DetectorMapPlan plan;
    const double scale = std::max(std::fabs(max_col_scale), std::fabs(max_row_scale));
    if (!std::isfinite(scale) || !std::isfinite(max_row_slope)) throw Error("detector map: row_slope, col_scale and row_scale must be finite");
    if (scale >= 0.5) throw Error("col_scale / row_scale: a scale of 0.5 or more is not a correction of the detector's scale");
    if (m.n < 0 || m.n > detector_map_max_terms)
        throw Error("detector map: at most " + std::to_string(detector_map_max_terms) + " terms");
    if (m.n == 0) return plan;
    if (m.det_rows < 1 || m.det_cols < 1) throw Error("detector map: no detector");
    if (!std::isfinite(m.c0) || !std::isfinite(m.v0) || !std::isfinite(m.cs) || !std::isfinite(m.vs) || m.cs == 0 || m.vs == 0)
        throw Error("detector map: the normalisation (c0, cs, v0, vs) must be finite, the scales not zero");
    for (int k = 0; k < m.n; ++k) {
        if (m.i[k] < 0 || m.j[k] < 0 || m.i[k] > detector_map_max_power || m.j[k] > detector_map_max_power)
            throw Error("detector map: powers must be in [0, " + std::to_string(detector_map_max_power) + "]");
        if (!std::isfinite(m.eu[k]) || !std::isfinite(m.ev[k])) throw Error("detector map: coefficients must be finite");
    }
    // the area in normalised coordinates
    const double xa = (-detector_map_col_margin - m.c0) / m.cs, xb = (m.det_cols - 1 + detector_map_col_margin - m.c0) / m.cs;
    const double ya = (0 - m.v0) / m.vs, yb = (m.det_rows - 1 - m.v0) / m.vs;
    const double xm = std::max(std::fabs(xa), std::fabs(xb)), ym = std::max(std::fabs(ya), std::fabs(yb));
    const int N = 128;   // grid intervals each way
    const double hx = std::fabs(xb - xa) / N, hy = std::fabs(yb - ya) / N;
    // sup |d^a_x d^b_y E| over the area: the largest value at the grid's nodes, plus what the function can change
    // by within half a step of one (bounded by the terms' magnitudes)
    auto sup = [&](const double* e, int a, int b) {
        double node = 0;
        for (int q = 0; q <= N; ++q)
            for (int r = 0; r <= N; ++r)
                node = std::max(node, std::fabs(derivative(m, e, a, b, xa + (xb - xa) * q / N, ya + (yb - ya) * r / N, false)));
        return node + 0.5 * hx * derivative(m, e, a + 1, b, xm, ym, true) + 0.5 * hy * derivative(m, e, a, b + 1, xm, ym, true);
    };
    plan.max_eu = sup(m.eu, 0, 0);
    plan.max_ev = sup(m.ev, 0, 0);
    const double cs = std::fabs(m.cs), vs = std::fabs(m.vs);
    plan.gradient = std::max(sup(m.eu, 1, 0) / cs + sup(m.eu, 0, 1) / vs, sup(m.ev, 1, 0) / cs + sup(m.ev, 0, 1) / vs);
    // One pass maps (Eu, Ev) to the map at p = (X + .. - Eu) gc, v = (y0 + row_slope (p - ..) + .. - Ev) gr, with
    // gc = 1 / (1 + col_scale), gr = 1 / (1 + row_scale): a change of (Eu, Ev) by d moves p by at most g d and v by
    // at most g (1 + g |row_slope|) d, g = 1 / (1 - scale).
    const double g = 1.0 / (1.0 - scale), reach = g * (1.0 + g * std::fabs(max_row_slope));
    plan.contraction = plan.gradient * reach;
    const double start = reach * std::max(plan.max_eu, plan.max_ev);   // the error of (p, v) before the first pass
    if (!std::isfinite(start) || !std::isfinite(plan.contraction)) throw Error("detector map: not finite on the detector");
    double left = start;
    while (left > detector_map_tolerance) {
        if (plan.passes == detector_map_max_passes || !(plan.contraction < 1))
            throw Error("detector map: its gradient (at most " + std::to_string(plan.gradient) +
                        " pixels per pixel) is too steep for the fixed-point inversion");
        left *= plan.contraction;
        ++plan.passes;
    }
    // a pass that changes (Eu, Ev) by d leaves at most d q / (1 - q) of them, `reach` times that of the positions
    plan.settle = plan.contraction > 0 ? detector_map_tolerance * (1.0 - plan.contraction) / (plan.contraction * reach) : 1e300;
    return plan;
}

void detector_map_source(const DetectorMapView& map, const double* x, const double* y, std::size_t n, double row_slope,
                         double slope_col, double col_scale, double scale_col, double row_scale, double scale_row,
                         double* c, double* v) {
    for (std::size_t q = 0; q < n; ++q)
        map_core::source<1>(map, y[q], row_slope, slope_col, col_scale, scale_col, row_scale, scale_row, x + q, c + q, v + q);
}

void detector_map_source_row(const DetectorMapView& map, const int32_t* c, const int32_t* base, const float* pos,
                             std::size_t n, float row_scale, float scale_row, float* out) {
    for (std::size_t q = 0; q < n; ++q) out[q] = map_core::source_row(map, row_scale, scale_row, c[q], base[q], pos[q]);
}

namespace geometry {

bool Map::empty() const {
    for (double e : eu)
        if (e != 0.0) return false;
    for (double e : ev)
        if (e != 0.0) return false;
    return true;
}

void Map::basis(double c, double v, double* phi) const {
    const double x = (c - c0) / cs, y = (v - v0) / vs;
    // the powers of x and y by repeated multiplication, as the reconstruction forms them (detector_map_core.hpp):
    // a block's resampling asks for three displacements per sample, and std::pow was most of its time
    constexpr int top = 8;
    double xp[top + 1], yp[top + 1];
    xp[0] = yp[0] = 1.0;
    for (int d = 1; d <= top; ++d) xp[d] = xp[d - 1] * x, yp[d] = yp[d - 1] * y;
    for (std::size_t k = 0; k < i.size(); ++k)
        phi[k] = (i[k] >= 0 && i[k] <= top ? xp[i[k]] : std::pow(x, i[k])) * (j[k] >= 0 && j[k] <= top ? yp[j[k]] : std::pow(y, j[k]));
}

void Map::displacement(double c, double v, double& du, double& dv) const {
    double phi[detector_map_max_terms];
    if (i.size() > static_cast<std::size_t>(detector_map_max_terms)) throw Error("detector map: more than " + std::to_string(detector_map_max_terms) + " terms");
    basis(c, v, phi);
    du = dv = 0;
    for (std::size_t k = 0; k < i.size(); ++k) {
        du += eu[k] * phi[k];
        dv += ev[k] * phi[k];
    }
}

void Map::source(double x, double y, double& c, double& v) const {
    c = x, v = y;
    for (int pass = 0; pass < 3; ++pass) {  // the map changes by 1e-3 per pixel: three passes leave nothing
        double du, dv;
        displacement(c, v, du, dv);
        c = x - du, v = y - dv;
    }
}

}  // namespace geometry

}  // namespace vc_tomography
