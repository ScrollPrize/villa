#pragma once
// GHBP: hierarchical (voxel-driven, multi-level) backprojection, a faithful
// port of nabuxx/ghbp.
//
// Angle convention: `angles` are the nabu angles A (radians). Output pixel
// [i][j] with c = (N-1)/2 receives, at full resolution, the detector sample at
//     h = rot_center + (j - c) cos A - (i - c) sin A
// GHBP approximates this by backprojecting groups of rf = ceil(n^(1/steps))
// angles onto coarse rotated grids (2x oversampled across the rays, coarse
// along them) and bilinearly aggregating those into the N x N slice. Result is
// a plain sum over angles times `scale_factor` (no d-theta factor).
#include <memory>
#include <vector>

#include "helirec/common.hpp"

namespace helirec {

struct GhbpParams {
    int grid_size = 0;              // N: output is N x N
    float rot_center = 0.f;         // detector bin of the rotation axis
    float scale_factor = 1.f;
    int reduction_steps = 2;
    int legs = 4;
    bool clip_outer_circle = false; // zero pixels with r > N/2 (integer N/2)
    double axis_source_meters = 1e9;  // fan-beam magnification only; 1e9 = parallel
    float voxel_size_microns = 1.f;   // only enters via axis_source_meters
    float whf_x = 1.f, whf_y = 1.f, fac = 1.f;
};

struct GhbpLevel {
    int h = 0, w = 0;      // grid shape
    int g = 0;             // grids per leg
    int red = 1;           // reduction factor into this level
    std::vector<float> T;     // (g * legs, 6): rows 0..1 of grid -> image transform (NaN padded)
    std::vector<float> Tinv;  // (g * legs, 6): rows 0..1 of the inverse
};

struct GhbpPlan {
    GhbpParams p;
    int n_angles = 0, n_bins = 0;
    std::vector<float> setups;     // (n_angles, 8): projection rows ph[4], pw[4]
    std::vector<GhbpLevel> levels; // level 0 .. final (final: h = w = N, g = 1)
};

// angles: nabu angles A (n_angles); axis_correction: optional per-angle shift (bins)
GhbpPlan make_ghbp_plan(const float* angles, int n_angles, int n_bins, const GhbpParams& p,
                        const float* axis_correction = nullptr);

namespace cpu {
// sino: (n_angles, n_bins) host; out: (N, N) host
void ghbp(const GhbpPlan& plan, const float* sino, float* out);
}  // namespace cpu

#ifdef HELIREC_WITH_CUDA
namespace cuda {
class Ghbp {
  public:
    explicit Ghbp(const GhbpPlan& plan);
    ~Ghbp();
    // device pointers; sino (n_angles, n_bins), out (N, N)
    void run(const float* sino, float* out, void* stream = nullptr);
    const GhbpPlan& plan() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace cuda
#endif

}  // namespace helirec
