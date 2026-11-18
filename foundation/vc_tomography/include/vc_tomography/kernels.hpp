#pragma once
// Kernel interfaces. One block per pipeline stage; the CPU and CUDA versions
// have identical semantics and are validated against each other in tests/.
#include <algorithm>
#include <cstdint>
#include <string>

#include "vc_tomography/common.hpp"

namespace vc_tomography {

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

#ifdef VC_TOMOGRAPHY_WITH_CUDA
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
// pixels it keeps, e.g. under an output mask, skips the rest). The tiles are those of the grid, which is cut in tiles
// of kBackprojectTile x kBackprojectTile pixels from its pixel (0, 0), not from the slice's first pixel: a pixel then
// lies at the same place of the same tile whatever part of the grid a slice covers, and has the same value there
// (the CUDA backprojector counts a pixel's positions from its tile's first pixel). BackprojectTiles: the tiles that
// hold pixels of a slice, tile ty * ntx + tx.
constexpr int kBackprojectTile = 32;
struct BackprojectTiles {
    int tx0 = 0, ty0 = 0;   // the grid's tile that holds the slice's first pixel
    int ntx = 0, nty = 0;   // the tiles the slice reaches along x and y
    int x0 = 0, y0 = 0, nx = 0, ny = 0;
    BackprojectTiles(int ny_, int nx_, int x0_, int y0_) : x0(x0_), y0(y0_), nx(nx_), ny(ny_) {
        const auto tile = [](int v) { return v >= 0 ? v / kBackprojectTile : -((-v + kBackprojectTile - 1) / kBackprojectTile); };
        tx0 = tile(x0);
        ty0 = tile(y0);
        ntx = nx > 0 ? tile(x0 + nx - 1) - tx0 + 1 : 0;
        nty = ny > 0 ? tile(y0 + ny - 1) - ty0 + 1 : 0;
    }
    int count() const { return ntx * nty; }
    // the slice's pixels [x_lo, x_hi) x [y_lo, y_hi) of its tile (tx, ty)
    int x_lo(int tx) const { return std::max(0, (tx0 + tx) * kBackprojectTile - x0); }
    int x_hi(int tx) const { return std::min(nx, (tx0 + tx + 1) * kBackprojectTile - x0); }
    int y_lo(int ty) const { return std::max(0, (ty0 + ty) * kBackprojectTile - y0); }
    int y_hi(int ty) const { return std::min(ny, (ty0 + ty + 1) * kBackprojectTile - y0); }
};
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
    // A beam that diverges (a source at a finite distance): the ray recorded at the column m of a row (m = u /
    // u_scale, the columns the rotation axis is counted in) is not the parallel beam's of the row's angle a but that
    // ray turned about its own point nearest the axis by fan[m] radians, towards a - fan[m]. A pixel (x, y) from the
    // axis, at t = x cos a + y sin a + rot_center, is then read at the column t + fan[t] (x sin a - y cos a): fan at
    // the nearest column to t, the ends' values beyond them. The filter along the rows is the parallel beam's: what
    // the exact fan-beam formula adds (a weight by the depth, the rows' cosine) differs from 1 by the fan itself and
    // is left out, as is the fan's square. (n_fan) host; null: a parallel beam.
    const float* fan = nullptr;
    int n_fan = 0;
};
// interpolation name <-> BackprojectParams::interp
int interp_from_name(const std::string& name);

namespace cpu {
void backproject(const float* sino, const float* angles, float* slice, int n_angles, int n_det,
                 int ny, int nx, const BackprojectParams& p);
}  // namespace cpu

#ifdef VC_TOMOGRAPHY_WITH_CUDA
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

#ifdef VC_TOMOGRAPHY_WITH_CUDA
namespace cuda {
// device pointers
void to_uint8(const float* src, std::size_t src_pitch, std::uint8_t* out, int ny, int nx, const Uint8Window& w,
              void* stream = nullptr);
}  // namespace cuda
#endif

}  // namespace vc_tomography
