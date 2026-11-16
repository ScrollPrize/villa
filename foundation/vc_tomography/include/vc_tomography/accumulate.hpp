#pragma once
// Gridded accumulation of one scan's projections onto the regular angular grid,
// written as a gather: every output
// pixel (j, t, X) sums its contributors in the reference's order, so the CPU
// and CUDA kernels are deterministic and match nabuxx project_radio.
//
//   R [j, t, X] += (v * A) * fx      RW[j, t, X] += (w * A) * fx
// v = interpolated flat-fielded intensity * interpolated weight, w = weight,
// A = angular splat (1 - alpha or alpha), fx = horizontal splat.
//
// A scan placed at a fractional column offset is, by the reference splat, averaged over two columns (at half a
// pixel: a [1/2, 1/2] box, response cos(pi nu)). With AccumScanView::x_taps > 0 its value is instead interpolated
// across x_taps columns (the weight stays the splat's), wherever all of them are measured columns; next to the
// detector edges and to unmeasured columns the splat is kept. A kernel of a few columns still loses the top of the
// band (lanczos3 at half a pixel: 0.82 at 0.36 cycles per pixel, 0.35 at 0.45, 0 at Nyquist). With
// AccumScanView::x_exact the rows of the block's projections are instead delayed by the offset exactly
// (shift_rows, sino.hpp) before they are gathered; gridding of measured pixels only (iterative data) keeps the
// kernel, unless the scan has a lateral registration or the lateral part of a detector map, which a kernel of fixed
// weights cannot apply: its rows are then read exactly as well (AccumScanView::col_wt).
//
// Vertically, a slice lies at a fractional detector row of each projection. The reference interpolates the two
// neighbouring rows linearly; as the fraction sweeps all values along a helix, that is a triangle kernel along z
// (sinc^2: 0.81 at half Nyquist, 0.41 at Nyquist). With AccumScanView::z_taps > 0 the value is interpolated across
// z_taps rows instead (the weight stays the linear one), wherever all of them are measured. Over the views of a
// voxel in a helix a kernel keeps the mean of its response over the fraction: lanczos3 (6 rows) 0.91 of the
// amplitude at 0.36 cycles per row, 0.82 at 0.40, 0.68 at 0.45; lanczos8 (16 rows) 1.00, 1.01, 0.90.
#include <cstdint>
#include <string>
#include <vector>

#include "vc_tomography/common.hpp"
#include "vc_tomography/detector_map.hpp"

namespace vc_tomography {

// Device- or host-resident views of one scan's chunk data (pointer space = backend).
struct AccumScanView {
    int n_proj = 0;          // projections in the chunk
    int det_rows = 0, det_cols = 0;
    int win_rows = 0;        // rows per projection window (H + 1 + 2 win_off)
    int win_off = 0;         // window rows before stack row 0 (context of the vertical kernel)
    const uint16_t* raw = nullptr;     // (n_proj, win_rows, det_cols), or (n_slots, ...) with raw_slot
    const int32_t* raw_slot = nullptr; // optional: projection k's window is raw[raw_slot[k]] (streamed subsets)
    const float* dark = nullptr;       // (det_rows, det_cols)
    const float* flats = nullptr;      // (n_flats, det_rows, det_cols): (flat - dark) / (I_flat / 0.2)
    const float* weight = nullptr;     // (det_rows, det_cols): weight map x horizontal apodisation
    // per projection
    const int32_t* row0 = nullptr;
    const float* fract_v = nullptr;
    const float* alpha = nullptr;
    const float* current_div = nullptr;
    const float* weight_factor = nullptr;
    const float* flat_frac = nullptr;
    const int32_t* flat_i0 = nullptr;
    const int32_t* flat_i1 = nullptr;
    // optional scintillator glare (sld.hpp), subtracted from (raw - dark) / current: maps (n_maps, glare_rows,
    // glare_cols) of the binned detector; projection k takes (1 - glare_frac) maps[glare_i0] + glare_frac maps[glare_i1]
    const float* glare = nullptr;
    int glare_rows = 0, glare_cols = 0;
    float glare_inv_bin = 0;
    const int32_t* glare_i0 = nullptr;
    const int32_t* glare_i1 = nullptr;
    const float* glare_frac = nullptr;
    const int32_t* glare_row0 = nullptr;  // optional first detector row of each full-resolution map strip
    // horizontal placement
    int sx = 0, ixs = 0;
    float fract_x = 0;
    // optional kernel for the fractional column offset (x_taps > 0, set_x_kernel): the value landing on a merged
    // column is sum_i x_w[i] value(c0 - x_taps / 2 + i), the source position being c0 - fract_x. 0: the reference's
    // two-column linear splat
    int x_taps = 0;
    float x_w[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    // the fractional column offset by an exact band-limited shift of the block's rows instead (set_x_kernel), where
    // accumulate computes them (col_val); the kernel above remains for the gridding of measured pixels only (of a
    // scan without a lateral registration or map: col_wt)
    int x_exact = 0;
    // optional kernel for the vertical position (z_taps > 0): the value at row t + fract_v[k] of projection k is
    // sum_i z_w[k * z_taps + i] value(t - (z_taps / 2 - 1) + i), wherever all those rows are on the detector with a
    // positive weight (else the linear interpolation of rows t, t + 1). The weight stays the linear one.
    int z_taps = 0;               // at most z_max_taps
    const float* z_w = nullptr;   // (n_proj, z_taps)
    // optional roll of the frames (ScanTrajectory::row_slope, per projection): the sample for stack row t at column c
    // is taken at row t + fract_v[k] + row_slope[k] (c - slope_col). The vertical kernel's weights then depend on
    // the column: they are read from z_lut, (z_lut_n + 1, z_taps) weights for the fractions i / z_lut_n
    // (z_kernel_lut). win_off must cover the kernel's context plus the largest row offset.
    const float* row_slope = nullptr;
    float slope_col = 0;
    const float* z_lut = nullptr;
    int z_lut_n = 0;
    // optional lateral registration of the frames (ScanTrajectory::col_offset, col_slope, per projection; both or
    // neither): the sample for column c of the row at detector row v is taken at column
    // c + col_offset[k] + col_slope[k] (v - slope_row). With x_exact the block's rows are delayed accordingly, each
    // by its own amount (accumulate; shift_rows); otherwise it is not applied.
    const float* col_offset = nullptr;
    const float* col_slope = nullptr;
    float slope_row = 0;
    // optional detector map of the scan (ScanTrajectory::map_*; detector_map.hpp) and scales of its projections
    // (ScanTrajectory::row_scale, col_scale; per projection, each or none): detector pixel (c, v) of projection k
    // looks at the ideal position (c + Eu(c, v) + col_scale[k] (c - scale_col), v + Ev(c, v) + row_scale[k] (v -
    // scale_row)), and the positions above (the slice's row with the roll, the column with the lateral
    // registration) are ideal ones.
    // map_rows: the vertical part is there (terms with ev, or row_scale). The sample for stack row t at detector
    // column c is then taken at the detector row v with v + Ev(c, v) + row_scale[k] (v - scale_row) = the row above.
    // The kernel's weights are read from z_lut as with a roll, here interpolated linearly between its fractions
    // (within 4e-7 of the kernel's weights). win_off must cover the largest row displacement as well.
    // map_cols: the lateral part is there (terms with eu, or col_scale). With x_exact the block's rows, which are
    // functions of the detector column, are read at the detector column p of every ideal column
    // (accumulate; resample_rows): p + Eu(p, v) + col_scale[k] (p - scale_col) = the column above, v the detector
    // row read at p; otherwise it is not applied (the pipeline refuses such a map).
    DetectorMapView map;
    const float* row_scale = nullptr;
    float scale_row = 0;
    const float* col_scale = nullptr;
    float scale_col = 0;
    int map_rows = 0;
    int map_cols = 0;
    // optional (accumulate sets it for the block in hand): the vertically interpolated values of the block's
    // projections, each computed once instead of at every column tap of every pixel it lands on. The value of
    // projection k at stack row t, column c is col_val[(col_slot[k] * col_rows + t) * det_cols + c].
    // col_stride: storage columns per row (det_cols, or shift_row_stride(det_cols)); col_shifted: the rows hold the
    // values delayed by fract_x, i.e. col_val[.. c] is the value at source position c - fract_x (with map_cols: at
    // the detector column of that ideal column).
    const float* col_val = nullptr;
    const int32_t* col_slot = nullptr;
    int col_rows = 0;
    int col_stride = 0;
    int col_shifted = 0;
    // Gridding of measured pixels only, of a scan with a lateral registration or the lateral part of a map (x_exact
    // and col_offset or map_cols; accumulate sets col_wt for the block in hand, with col_val and col_shifted): the
    // cached rows are the measured intensities of the detector's columns (measured_column: value over weight, the
    // -log of AccumParams::log_values taken after the read; an unmeasured column takes the nearest measured one's
    // of its row, so that the row has no step), read exactly as above. col_wt (col_val's layout, by detector column): each value's measured weight, zero for an unmeasured
    // column and within col_guard columns of one or of the row's ends. A sample counts where the two columns around
    // it have such a weight, with the splat's weight of them; elsewhere nothing is added (no splat of unshifted
    // values, which would lie where the map does not put them).
    // col_guard: the three columns either side that an exact read leans on most, plus the largest lateral
    // displacement of the scan (map, col_scale, registration), since the weights are read by the ideal column and the
    // values come from the detector column that looks at it.
    const float* col_wt = nullptr;
    int col_guard = 3;
    // Noise splitting (ReconConfig::noise_split_scale; 0: none): every raw sample u of the frames takes
    //   u + split_scale x sqrt(max(split_gain x (u - dark) + split_floor, 0)) x z,
    // z a standard normal number fixed by split_key (the seed and the scan), the projection's place in the scan
    // (split_proj0 + k) and the pixel: the same in every run, chunk and device (to the rounding of its logarithm and
    // cosine).
    // Noise that is not quite white (a scintillator spreads its quanta over neighbouring pixels): z is then a white
    // number plus up to two smooth fields, each of split_amp[i]^2 times the white number's variance and made of
    // standard normal numbers on a lattice of split_spacing[i] pixels of the frame, joined by cubic B-splines (power
    // spectrum sinc^8(f x spacing): near exp(-(f / f0)^2), f0 = 0.276 / spacing cycles per pixel); the sum times
    // split_norm, so that z keeps a variance of 1 (to the 8 % by which a lattice field's own varies over a cell).
    float split_scale = 0, split_gain = 0, split_floor = 0;
    float split_amp[2] = {0, 0}, split_spacing[2] = {0, 0}, split_norm = 1;
    // split_flat_frames (0: the flats are left as they are): the frames a flat is the mean of. A flat's noise is the
    // same in every projection of the scan (rings in a slice); with this each flat takes split_scale x its own
    // noise's size (a frame's at the flat's level, over the root of the frames) x numbers of its own as well, so
    // that the two runs do not share it either.
    float split_flat_frames = 0;
    std::uint32_t split_key = 0;
    int split_proj0 = 0;
};

// Kernel of the fractional column offset of a scan: `interp` linear (the reference splat; x_taps = 0), cubic (Keys,
// 4 columns) or lanczos3 (6 columns), weights normalised to sum 1. A zero offset needs none. `exact`: the exact shift
// (x_exact) wherever accumulate computes the block's rows, the kernel elsewhere.
void set_x_kernel(AccumScanView& v, int interp, bool exact = false);
// Vertical kernel: its id from the name (linear, cubic, lanczos3, lanczos8: interp_core::Interp), the rows of
// context it needs either side of the two linear rows (0 linear, 1 cubic, 2 lanczos3, 7 lanczos8), and its weights
// (n, taps) for the vertical fractions fract_v (taps = 2 + 2 context rows <= z_max_taps; empty for linear).
inline constexpr int z_max_taps = 16;
int z_kernel_from_name(const std::string& name);
const char* z_kernel_name(int interp);
int z_kernel_context(int interp);
std::vector<float> z_kernel_weights(const std::vector<float>& fract_v, int interp);
// the weights for the fractions i / n, i = 0 .. n: (n + 1, taps) (AccumScanView::z_lut; empty for linear)
std::vector<float> z_kernel_lut(int interp, int n);

// Contributors of each grid angle of a block [g0, g1): CSR over (j - g0), entries
// sorted by projection; role 0 = lower bin (weight 1 - alpha), 1 = upper bin (alpha).
struct ContributorList {
    int g0 = 0, g1 = 0;
    std::vector<int32_t> offsets;  // (g1 - g0 + 1)
    std::vector<int32_t> proj;     // local projection index
    std::vector<int8_t> role;
};
// upper_bin false (nearest-bin gridding: every alpha is 0, j0 is wrapped): only the role 0 entries.
ContributorList make_contributors(const int32_t* j0, int n_proj, int n_grid, int g0, int g1, bool upper_bin = true);
// The projections a block's contributors name, in ascending order (slot_proj), and each projection's place in that
// list (slot: n_proj entries, -1 for the others).
void contributor_slots(const ContributorList& c, int n_proj, std::vector<int32_t>& slot, std::vector<int32_t>& slot_proj);

struct AccumParams {
    int H = 0;              // rows of the gridded stack (slices + 2 margins)
    int NX = 0;             // merged width
    float non_zero = 1e-14f;
    int border = 100;       // border padding width of the reference
    bool measured_only = false;  // iterative data: exclude padding and invalid detector pixels
    bool log_values = false;     // -log each valid sample before gridding (no intensity processing)
    float log_clip_min = 1e-6f, log_clip_max = 10.f;
};

namespace cpu {
// adds this scan's contributions to R, RW (L = g1 - g0, H, NX). weight2 (a second weight map of the detector, like
// AccumScanView::weight): the same contributions with those weights are added to R2, RW2 as well (the block's rows
// are computed once).
void accumulate(const AccumScanView& s, const ContributorList& c, const AccumParams& p, float* R, float* RW,
                const float* weight2 = nullptr, float* R2 = nullptr, float* RW2 = nullptr);
}
#ifdef VC_TOMOGRAPHY_WITH_CUDA
namespace cuda {
struct DeviceContributors;  // opaque
void accumulate(const AccumScanView& s, const ContributorList& c, const AccumParams& p, float* R, float* RW,
                void* stream = nullptr, const float* weight2 = nullptr, float* R2 = nullptr, float* RW2 = nullptr);
}  // namespace cuda
#endif

// ---------------------------------------------------------------- luminosity
// Window sums around seam column cx (Gaussian sigma 6, radius 12, rows clipped to the stack):
//   A[j, t] = sum g R,  B[j, t] = sum g RW            (L, H)
struct LuminosityKernel {
    std::vector<float> gx, gz;  // 25 taps each (host expf, as the reference)
    int radius = 12;
};
LuminosityKernel luminosity_kernel();

namespace cpu {
void luminosity_sums(const float* R, const float* RW, int L, int H, int NX, int cx, const LuminosityKernel& k,
                     float* A, float* B);
// R += d * W_prev - (d * (rep + 100) / 200) * RW, with d = lum_TA - lum_HA computed from
// (A_ha, B_ha) and the TA sums A_tot - A_ha ... exactly as nabuxx (two contributions)
void luminosity_apply(float* R, const float* RW, const float* W_prev, const float* A_ha, const float* B_ha,
                      const float* A_tot, const float* B_tot, int L, int H, int NX, float repartition);
// R = RW > 0 ? R / RW : 0
void divide(float* R, const float* RW, std::size_t n);
// R += A (the sums of one row of scans added to those of the rows before)
void add_into(float* R, const float* A, std::size_t n);
// nabuxx mend_null_weights (row-wise nearest valid replacement) on the usable merged columns [x0, x1)
void mend_null_weights(float* R, const float* RW, int L, int H, int NX, float threshold, int x0, int x1);
// Remove unreliable measured support; optionally fill only unsupported values for filter context.
// Filling never changes RW, so these values cannot become equations in an iterative solver.
void prepare_measured(float* R, float* RW, int L, int H, int NX, float threshold, bool fill, float empty);
}  // namespace cpu
#ifdef VC_TOMOGRAPHY_WITH_CUDA
namespace cuda {
void luminosity_sums(const float* R, const float* RW, int L, int H, int NX, int cx, const LuminosityKernel& k,
                     float* A, float* B, void* stream = nullptr);
void luminosity_apply(float* R, const float* RW, const float* W_prev, const float* A_ha, const float* B_ha,
                      const float* A_tot, const float* B_tot, int L, int H, int NX, float repartition,
                      void* stream = nullptr);
void divide(float* R, const float* RW, std::size_t n, void* stream = nullptr);
void add_into(float* R, const float* A, std::size_t n, void* stream = nullptr);
void mend_null_weights(float* R, const float* RW, int L, int H, int NX, float threshold, int x0, int x1,
                       void* stream = nullptr);
void prepare_measured(float* R, float* RW, int L, int H, int NX, float threshold, bool fill, float empty,
                      void* stream = nullptr);
}  // namespace cuda
#endif

}  // namespace vc_tomography
