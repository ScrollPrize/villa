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
// detector edges and to unmeasured columns the splat is kept.
#include <cstdint>
#include <vector>

#include "helirec/common.hpp"

namespace helirec {

// Device- or host-resident views of one scan's chunk data (pointer space = backend).
struct AccumScanView {
    int n_proj = 0;          // projections in the chunk
    int det_rows = 0, det_cols = 0;
    int win_rows = 0;        // rows per projection window (H + 1)
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
    // horizontal placement
    int sx = 0, ixs = 0;
    float fract_x = 0;
    // optional kernel for the fractional column offset (x_taps > 0, set_x_kernel): the value landing on a merged
    // column is sum_i x_w[i] value(c0 - x_taps / 2 + i), the source position being c0 - fract_x. 0: the reference's
    // two-column linear splat
    int x_taps = 0;
    float x_w[8] = {0, 0, 0, 0, 0, 0, 0, 0};
};

// Kernel of the fractional column offset of a scan: `interp` linear (the reference splat; x_taps = 0), cubic (Keys,
// 4 columns) or lanczos3 (6 columns), weights normalised to sum 1. A zero offset needs none.
void set_x_kernel(AccumScanView& v, int interp);

// Contributors of each grid angle of a block [g0, g1): CSR over (j - g0), entries
// sorted by projection; role 0 = lower bin (weight 1 - alpha), 1 = upper bin (alpha).
struct ContributorList {
    int g0 = 0, g1 = 0;
    std::vector<int32_t> offsets;  // (g1 - g0 + 1)
    std::vector<int32_t> proj;     // local projection index
    std::vector<int8_t> role;
};
ContributorList make_contributors(const int32_t* j0, int n_proj, int n_grid, int g0, int g1);

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
// adds this scan's contributions to R, RW (L = g1 - g0, H, NX)
void accumulate(const AccumScanView& s, const ContributorList& c, const AccumParams& p, float* R, float* RW);
}
#ifdef HELIREC_WITH_CUDA
namespace cuda {
struct DeviceContributors;  // opaque
void accumulate(const AccumScanView& s, const ContributorList& c, const AccumParams& p, float* R, float* RW,
                void* stream = nullptr);
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
// nabuxx mend_null_weights (row-wise nearest valid replacement) on the usable merged columns [x0, x1)
void mend_null_weights(float* R, const float* RW, int L, int H, int NX, float threshold, int x0, int x1);
// Remove unreliable measured support; optionally fill only unsupported values for filter context.
// Filling never changes RW, so these values cannot become equations in an iterative solver.
void prepare_measured(float* R, float* RW, int L, int H, int NX, float threshold, bool fill, float empty);
}  // namespace cpu
#ifdef HELIREC_WITH_CUDA
namespace cuda {
void luminosity_sums(const float* R, const float* RW, int L, int H, int NX, int cx, const LuminosityKernel& k,
                     float* A, float* B, void* stream = nullptr);
void luminosity_apply(float* R, const float* RW, const float* W_prev, const float* A_ha, const float* B_ha,
                      const float* A_tot, const float* B_tot, int L, int H, int NX, float repartition,
                      void* stream = nullptr);
void divide(float* R, const float* RW, std::size_t n, void* stream = nullptr);
void mend_null_weights(float* R, const float* RW, int L, int H, int NX, float threshold, int x0, int x1,
                       void* stream = nullptr);
void prepare_measured(float* R, float* RW, int L, int H, int NX, float threshold, bool fill, float empty,
                      void* stream = nullptr);
}  // namespace cuda
#endif

}  // namespace helirec
