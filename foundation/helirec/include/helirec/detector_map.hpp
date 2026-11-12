#pragma once
// A detector map for the reconstruction: where on the detector the ray of an ideal position is recorded.
//
// The convention is helirec::calibration::DetectorMap's (calibration/frames.hpp): detector pixel (c, v) looks along
// the ray at the ideal position (c + Eu(c, v), v + Ev(c, v)), with
//   Eu = sum_k eu[k] x^i[k] y^j[k],  Ev = sum_k ev[k] x^i[k] y^j[k],  x = (c - c0) / cs,  y = (v - v0) / vs.
// The value wanted at the ideal position (X, Y) is therefore read at the detector position (c, v) that solves
//   c + Eu(c, v) = X,   v + Ev(c, v) = Y
// (Eu = +2: column X is read at detector column X - 2; Ev = -1: row Y is read at detector row Y + 1).
//
// A projection may add a scale of its own to either equation (ScanTrajectory::col_scale, row_scale):
//   c + Eu(c, v) + col_scale (c - scale_col) = X,   v + Ev(c, v) + row_scale (v - scale_row) = Y.
//
// The polynomials are evaluated inside the detector only: at columns [-detector_map_col_margin, det_cols - 1 +
// detector_map_col_margin] and rows [0, det_rows - 1], an argument beyond being held at that edge (the rows
// gridded beyond the detector are copies of its edge rows; columns beyond it are the padding of the row transform).
// The scales are linear and are not held.
//
// The equations are solved by fixed-point passes on (Eu, Ev) from zero, each pass solving the rest, which is linear,
// exactly. detector_map_plan bounds the polynomials and their gradient over that area: a pass contracts the error
// by `contraction`, and `passes` of them leave less than detector_map_tolerance pixels (three or four for a map of
// 2 pixels that changes by 1e-3 pixels per pixel). The gridding's rows are solved so in single precision
// (accumulate.hpp: within 1e-6 rows); the columns of resample_rows (sino.hpp) in double precision, started from the
// single-precision passes and ended when a pass changes (Eu, Ev) by less than `settle`, which leaves less than the
// tolerance.
#include <cstdint>
#include <vector>

#include "helirec/common.hpp"

namespace helirec {

constexpr int detector_map_max_power = 8;       // largest power of x or y of a term
constexpr int detector_map_max_terms = 64;
constexpr int detector_map_col_margin = 32;     // columns beyond the detector over which the map is still evaluated
constexpr int detector_map_max_passes = 8;
// pixels left by the fixed point (in exact arithmetic): 1e-7 of the amplitude of a row's content at its Nyquist
// frequency (whose slope is pi x amplitude per pixel)
constexpr double detector_map_tolerance = 3e-8;

// Host- or device-resident view of a map (pointer space = backend). n = 0: no polynomial terms.
struct DetectorMapView {
    int n = 0;
    const int32_t* i = nullptr;     // (n) powers of x
    const int32_t* j = nullptr;     // (n) powers of y
    const double* eu = nullptr;     // (n) columns
    const double* ev = nullptr;     // (n) rows
    const float* ev_f = nullptr;    // (n) ev in single precision: the vertical position is solved in it
    double c0 = 0, cs = 1, v0 = 0, vs = 1;
    int det_rows = 0, det_cols = 0; // the area the polynomials are evaluated in
    int passes = 0;                 // fixed-point passes (detector_map_plan)
    double settle = 0;              // a pass that changes (Eu, Ev) by no more leaves less than the tolerance
};

struct DetectorMapPlan {
    int passes = 0;
    double max_eu = 0, max_ev = 0;  // bounds of |Eu|, |Ev| over the area (pixels)
    double gradient = 0;            // bound of max(|dEu/dc| + |dEu/dv|, |dEv/dc| + |dEv/dv|) over the area
    double contraction = 0;         // of one pass, with the scales and the roll
    double settle = 0;              // DetectorMapView::settle
};

// Checks a map (host pointers; n, powers, normalisation, finite coefficients, the area) and plans its inversion for
// projections whose |row_slope|, |col_scale| and |row_scale| are at most the given values. Throws Error for a map
// that the fixed point cannot invert to detector_map_tolerance in detector_map_max_passes passes, and for scales of
// 0.5 or more. The bounds are rigorous: values on a grid plus the largest change between its nodes.
DetectorMapPlan detector_map_plan(const DetectorMapView& map, double max_row_slope = 0, double max_col_scale = 0,
                                  double max_row_scale = 0);

// The detector positions (c, v) read for the ideal columns x of rows whose ideal row is y at column slope_col:
//   c + Eu(c, v) + col_scale (c - scale_col) = x,   v + Ev(c, v) + row_scale (v - scale_row) = y + row_slope (c - slope_col)
// in double precision, as resample_rows solves them (sino.hpp; within detector_map_tolerance: it ends the passes of
// eight positions together). `map` as planned (passes, settle). n positions.
void detector_map_source(const DetectorMapView& map, const double* x, const double* y, std::size_t n, double row_slope,
                         double slope_col, double col_scale, double scale_col, double row_scale, double scale_row,
                         double* c, double* v);
// The detector row read at the integer detector column c for the ideal row base + pos (single precision, as the
// gridding solves it: AccumScanView, accumulate.hpp), relative to `base`. n positions.
void detector_map_source_row(const DetectorMapView& map, const int32_t* c, const int32_t* base, const float* pos,
                             std::size_t n, float row_scale, float scale_row, float* out);

}  // namespace helirec
