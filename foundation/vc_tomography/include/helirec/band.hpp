#pragma once
// Band reconstruction for surface composites: the exact pixel-driven backprojector (kernels.hpp semantics)
// evaluated only at the slice pixels a set of sample points needs (their trilinear support), and the trilinear
// samples of those points between consecutive slices. No slice or volume is formed.
//
// Points (view height h, slice row y, slice column x) are given as zlo = lower slice (int), fz = h - zlo in
// [0, 1], y, x in full-slice pixels, sorted by zlo. A point reads slices zlo and zlo + 1 at pixels
// (iy + {0,1}, ix + {0,1}), iy = min(floor(y), N - 2), ix = min(floor(x), N - 2):
//     v = (1 - fz) * bilinear(slice zlo) + fz * bilinear(slice zlo + 1)
// i.e. the trilinear sample of the reconstructed volume (as a voxel grid) at that point.
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "helirec/common.hpp"
#include "helirec/kernels.hpp"

namespace helirec {

// pixel box of the slice holding every needed pixel; box images are (h, w), row-major
struct BandBox {
    int x0 = 0, y0 = 0, w = 0, h = 0;
    int n_grid = 0;  // N of the full (N, N) slice
};

namespace cpu {
// mask the 2x2 support of points [a, b)
void band_mark(const float* y, const float* x, std::size_t a, std::size_t b, const BandBox& box, uint8_t* mask);
// set mask pixels -> ordered list of box-linear indices; returns the count
std::size_t band_compact(const uint8_t* mask, std::size_t n, int32_t* list);
// exact backprojection of the listed box pixels (full-grid coordinates as BackprojectParams, x0/y0 ignored)
void backproject_list(const float* sino, const float* angles, int n_angles, int n_det, const int32_t* list,
                      std::size_t count, const BandBox& box, const BackprojectParams& p, float* img);
// trilinear samples of points [a, b) between box images of slices zlo and zlo + 1 -> out[a .. b)
void band_sample(const float* img_lo, const float* img_hi, const float* fz, const float* y, const float* x,
                 std::size_t a, std::size_t b, const BandBox& box, float* out);
}  // namespace cpu

#ifdef HELIREC_WITH_CUDA
namespace cuda {
// device pointers
void band_mark(const float* y, const float* x, std::size_t a, std::size_t b, const BandBox& box, uint8_t* mask,
               void* stream = nullptr);
// returns the count (synchronises); temp: scratch, grown as needed
std::size_t band_compact(const uint8_t* mask, std::size_t n, int32_t* list, void* stream = nullptr);
// cs_sn: device (n_angles) float2 (cos, sin)
void backproject_list(const float* sino, const void* cs_sn, int n_angles, int n_det, const int32_t* list,
                      std::size_t count, const BandBox& box, const BackprojectParams& p, float* img,
                      void* stream = nullptr);
void band_sample(const float* img_lo, const float* img_hi, const float* fz, const float* y, const float* x,
                 std::size_t a, std::size_t b, const BandBox& box, float* out, void* stream = nullptr);
}  // namespace cuda
#endif

// Holds the points, the per-slice state and the samples; fed slice by slice (in increasing, contiguous h)
// by SliceReconstructor::reconstruct_band.
class BandSampler {
  public:
    // n_sets: independent value sets over the same points (parameter sweeps; the pixel band is shared)
    BandSampler(Device device, const int32_t* zlo, const float* fz, const float* y, const float* x, std::size_t n,
                int grid_size, int n_sets = 1);
    ~BandSampler();
    BandSampler(const BandSampler&) = delete;
    BandSampler& operator=(const BandSampler&) = delete;

    std::size_t size() const;
    int n_sets() const;
    int h_first() const;      // first slice needed (min zlo)
    int h_last() const;       // last slice needed (max zlo + 1)
    const BandBox& box() const;
    Device device() const;
    // statistics
    std::size_t pixels_backprojected() const;
    std::size_t points_sampled() const;

    // one reconstructed slice of value set `set`: filtered sinogram (n_angles, n_det), on the sampler's device.
    // Each set must receive its slices in increasing, contiguous h.
    void slice(int h, const float* filtered, int n_angles, int n_det, const float* angles_host,
               const BackprojectParams& p, int set = 0);
    void values(float* out, int set = 0) const;  // host (size())
    void reset();                   // forget slices and samples (same points)

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace helirec
