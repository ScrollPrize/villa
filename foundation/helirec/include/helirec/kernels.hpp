#pragma once
// Kernel interfaces. One block per pipeline stage; the CPU and CUDA versions
// have identical semantics and are validated against each other in tests/.
#include <cstdint>
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
// With `tiles`, only the pixels of the listed tiles are computed and the others are 0 (a caller that knows which
// pixels it keeps, e.g. under an output mask, skips the rest): the slice is cut in tiles of kBackprojectTile x
// kBackprojectTile pixels from its first pixel, tile ty * ceil(nx / kBackprojectTile) + tx.
constexpr int kBackprojectTile = 32;
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
    const std::int32_t* tiles = nullptr;   // (n_tiles) host: the tiles to compute, each once; null: the whole slice
    int n_tiles = 0;
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

// ------------------------------------------------------------ uint8 output
// out[y, x] = rint(clamp((src[y, x] - lo) / (hi - lo), 0, 1) * 255) in float32 (ties to even), NaN -> 0.
// With disk_n > 0 the rows are the ROI [x0, x0 + nx) x [y0, y0 + ny) of a (disk_n, disk_n) slice and the pixels
// outside its reconstruction disk, (X - c)^2 + (Y - c)^2 > (disk_n / 2)^2 with c = (disk_n - 1) / 2, are 0.
// With a mask the pixels whose mask element is 0 are 0: pixel (y, x) reads mask[(y + mask_oy) / mask_fy,
// (x + mask_ox) / mask_fx], indices clamped to the (mask_ny, mask_nx) mask, so a mask coarser than the slice serves.
// src: (ny, nx) rows at a pitch of src_pitch elements; out: (ny, nx) packed.
struct Uint8Window {
    // double: the kernels divide by float(hi - lo), the correctly rounded width of the window
    double lo = 0.0, hi = 1.0;
    int disk_n = 0, x0 = 0, y0 = 0;
    const std::uint8_t* mask = nullptr;   // (mask_ny, mask_nx) packed, on the kernel's device
    int mask_ny = 0, mask_nx = 0, mask_fy = 1, mask_fx = 1, mask_oy = 0, mask_ox = 0;
};

namespace cpu {
void to_uint8(const float* src, std::size_t src_pitch, std::uint8_t* out, int ny, int nx, const Uint8Window& w);
}  // namespace cpu

#ifdef HELIREC_WITH_CUDA
namespace cuda {
// device pointers
void to_uint8(const float* src, std::size_t src_pitch, std::uint8_t* out, int ny, int nx, const Uint8Window& w,
              void* stream = nullptr);
}  // namespace cuda
#endif

}  // namespace helirec
