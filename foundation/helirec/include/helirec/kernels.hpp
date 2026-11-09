#pragma once
// Kernel interfaces. One block per pipeline stage; the CPU and CUDA versions
// have identical semantics and are validated against each other in tests/.
#include <string>

#include "helirec/common.hpp"

namespace helirec {

// ------------------------------------------------------------ flat-field + log
// out[p, i] = -log( max( (raw[p, i] - dark[i]) / (flat[i] - dark[i]), threshold ) )
// raw: (n_proj, n_pix)  dark, flat: (n_pix)  out: (n_proj, n_pix)
// Set take_log=false to keep the normalized intensity.
struct FlatFieldParams {
    float threshold = 1e-4f;
    bool take_log = true;
};

namespace cpu {
void flatfield(const float* raw, const float* dark, const float* flat, float* out,
               std::size_t n_proj, std::size_t n_pix, const FlatFieldParams& p);
}  // namespace cpu

#ifdef HELIREC_WITH_CUDA
namespace cuda {
// device pointers
void flatfield(const float* raw, const float* dark, const float* flat, float* out,
               std::size_t n_proj, std::size_t n_pix, const FlatFieldParams& p,
               void* stream = nullptr);
}  // namespace cuda
#endif

// ------------------------------------------------------------ backprojection
// Exact pixel-driven parallel-beam backprojection with linear interpolation.
// Output pixel (iy, ix) sits at  x = ix + x0 - c,  y = iy + y0 - c  with
// c = center (default (n - 1) / 2 of an (n, n) grid; x0, y0 select an ROI).
// For each angle a:  u = x * cos(a) + y * sin(a) + rot_center   (detector px)
//   slice[iy, ix] += interp(sino[a, :], u * u_scale)     (0 outside the detector; linear by default)
// result *= scale. With clip_outer_circle, pixels with x^2 + y^2 > R^2 are 0,
// R = clip_radius (default: c).
// sino: (n_angles, n_det), angles: (n_angles) radians, slice: (ny, nx)
struct BackprojectParams {
    float rot_center = 0.f;
    float scale = 1.f;
    float center = -1.f;      // grid center c; <0 -> (max(nx, ny) - 1) / 2
    int x0 = 0, y0 = 0;       // ROI offset inside the (n, n) grid
    bool clip_outer_circle = false;
    float clip_radius = -1.f; // <0 -> c
    int interp = 0;           // row interpolation: 0 linear, 1 cubic (Keys), 2 lanczos3 (src/common/interp_core.hpp)
    float u_scale = 1.f;      // rows sampled u_scale times per detector bin (oversampled): index = u * u_scale
    float unit_gain = 1.f;    // pipeline output / attenuation (cm^-1): 1; legacy reference mode 1/2 (methods that
                              // compute attenuation themselves, e.g. iterative ones, multiply by it)
};
// interpolation name <-> BackprojectParams::interp
int interp_from_name(const std::string& name);

namespace cpu {
void backproject(const float* sino, const float* angles, float* slice, int n_angles, int n_det,
                 int ny, int nx, const BackprojectParams& p);
}  // namespace cpu

#ifdef HELIREC_WITH_CUDA
namespace cuda {
// device pointers (angles on the host: they are copied to the device internally)
void backproject(const float* sino, const float* angles_host, float* slice, int n_angles,
                 int n_det, int ny, int nx, const BackprojectParams& p, void* stream = nullptr);
}  // namespace cuda
#endif

}  // namespace helirec
