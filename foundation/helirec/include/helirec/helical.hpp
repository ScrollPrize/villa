#pragma once
// Helical acquisition geometry: per-scan "span
// info" (which projections see which view height), the doable span, and the
// per-projection quantities needed to grid one chunk of slices.
//
// View height h: detector row, in the frame of projection 0 of the MAIN scan,
// onto which a slice projects. Slices are reconstructed for h in [lo, hi).
#include <cstdint>
#include <utility>
#include <vector>

#include "helirec/common.hpp"

namespace helirec {

struct ScanTrajectory {
    std::vector<double> angles_deg;         // physical, unwrapped (n)
    std::vector<double> z_translation_m;    // (n)
    std::vector<double> lateral_correction_m;  // (n) or empty (= 0)
    double pixel_size_m = 0;                // x pixel size used to convert translations
    int det_rows = 0, det_cols = 0;
    // placement in the merged (extended field of view) frame
    int shift_x = 0;           // integer column offset of this scan in the merged frame
    float x_readjustment = 0;  // shift_x - desired offset (e.g. -0.5)
    float z_readjustment = 0;  // vertical readjustment (pixels)
    std::size_t n() const { return angles_deg.size(); }
};

struct SpanInfo {
    std::vector<float> z_pix_per_proj;   // float32(z_pix[i] - z_pix[0])
    std::vector<float> x_pix_per_proj;   // float32(lateral_pix) + x_readjustment
    std::vector<int32_t> integer_shift_v;
    std::vector<float> fract_complement;
    int h_first = 0;                     // first height of the tables
    std::vector<int32_t> sunshine_start, sunshine_end;  // per height, -1 if not doable
    int doable_lo = 0, doable_hi = 0;    // first / last valid height (inclusive)
};

// float compare tolerance: angular_tolerance_steps (5 in the reference)
SpanInfo make_span_info(const ScanTrajectory& t, double angular_tolerance_steps);

// intersection over scans (same z direction assumed): [lo, hi) with hi EXCLUSIVE,
// i.e. heights lo..hi-1 can be reconstructed (the reference never uses the last valid height)
std::pair<int, int> doable_span(const std::vector<SpanInfo>& spans);

// projection index interval [first, last+1) whose union covers heights h0..h1 (inclusive)
std::pair<int, int> chunk_projection_span(const SpanInfo& s, int h0, int h1_inclusive);

// Everything the accumulator needs per projection of one scan for one chunk.
struct ChunkScan {
    int p0 = 0, p1 = 0;                  // projection range [p0, p1)
    std::vector<int32_t> row0;           // first raw row of the window (h_lo - margin - ishift)
    std::vector<float> fract_v;          // vertical interpolation fraction a
    std::vector<int32_t> j0;             // lower grid angle bin (unwrapped; j1 = (j0 + 1) % n_grid)
    std::vector<float> alpha;            // angular splat fraction (weight alpha to j0+1)
    std::vector<float> current_div;      // float32(I_p / 0.2): flat-field divisor
    std::vector<float> weight_factor;    // float32(I_p / 0.2): accumulation weight factor
    std::vector<float> flat_frac;        // f: flat_p = (1 - f) F0 + f F1
    std::vector<int32_t> flat_i0, flat_i1;  // flat series used
    int sx = 0, ixs = 0;                 // merged column of source column c: sx + ixs + c (+1 with fract_x)
    float fract_x = 0;
};

struct FlatSeriesInfo {
    std::vector<float> angles_deg;       // float32 unwrapped angle assigned to each flat series
};

struct FlatInterpolation { int i0=0, i1=0; float fraction=0; };
FlatInterpolation interpolate_flats(const FlatSeriesInfo&, float angle);

// h_lo..h_hi-1: slice heights of the chunk; margin: context rows either side (filters, luminosity window).
// Projections: those seeing the chunk's heights (the reference); full_context: every projection seeing a row of
// [h_lo - margin, h_hi + margin), so that the context rows do not depend on the chunking. padded_rows (with
// full_context): the gridding that pads rows beyond the detector from its last interpolable row needs that row
// inside the window.
ChunkScan prepare_chunk_scan(const ScanTrajectory& t, const SpanInfo& s, const std::vector<double>& control_A,
                             const FlatSeriesInfo& flats, int h_lo, int h_hi, int margin, int n_grid,
                             double nominal_current = 0.2, bool full_context = false, bool padded_rows = false);

// number of grid angles: round(360 / |mean step|) over the whole scan
int grid_angle_count(const std::vector<double>& angles_deg);

}  // namespace helirec
