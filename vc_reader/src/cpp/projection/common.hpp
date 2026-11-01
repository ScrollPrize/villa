#pragma once
// Surface samples projected into a voxel crop, shared by the CPU loop and the CUDA
// kernel. Points of the surface's quads (grid::point), offset through the band along
// the normal, each land in one voxel; a voxel chooses, among the samples it holds, the
// one whose surface point is nearest its centre, and reads the label at that sample's UV.
#include "../geometry/math.hpp"
#include <cmath>
#include <cstddef>
#include <cstdint>
#ifdef __CUDACC__
#define VCR_PROJECTION_HD __host__ __device__
#else
#define VCR_PROJECTION_HD
#endif
namespace vcr::projection {
using geometry::nearest;
// A crop's voxels and the label tile one target reads.
struct Raster {
    std::size_t count = 0;  // voxels
    int channels = 1, label_h = 0, label_w = 0;
    float uv_to_label[6]{};          // label pixel (x, y): the rows applied to (u, v, 1)
    std::int64_t label_origin[2]{};  // y, x of the label tile in its grid
};
// Voxel i takes the samples [offsets[i], offsets[i + 1]): their UV (two floats each), the
// distance from the voxel's centre to their surface point, and their id (ties). One thread
// owns each voxel, so selection and exact categorical conflict tests need no atomics.
template <class T>
VCR_PROJECTION_HD void raster(const Raster& g, const float* uv, const float* distance, const std::int64_t* ids, const std::int64_t* offsets, const T* label, T* out, bool* coverage,
                              bool* conflict, float* best, std::int64_t* winner, float* uvOut, bool* geometry, std::size_t i)
{
    for (auto pos = offsets[i]; pos < offsets[i + 1]; ++pos) {
        geometry[i] = true;
        const float x = uv[2 * pos], y = uv[2 * pos + 1], d = distance[pos];
        const float px = g.uv_to_label[0] * x + g.uv_to_label[1] * y + g.uv_to_label[2];
        const float py = g.uv_to_label[3] * x + g.uv_to_label[4] * y + g.uv_to_label[5];
        if (!std::isfinite(px) || !std::isfinite(py) || fabsf(px) > 0x1p30f || fabsf(py) > 0x1p30f) continue;
        const auto ix = nearest(px, g.label_origin[1]) - g.label_origin[1], iy = nearest(py, g.label_origin[0]) - g.label_origin[0];
        if (ix < 0 || iy < 0 || ix >= g.label_w || iy >= g.label_h) continue;
        const auto pixel = iy * g.label_w + ix;
        const bool choose = !coverage[i] || d < best[i] || (d == best[i] && ids[pos] < winner[i]);
        for (int ch = 0; ch < g.channels; ++ch) {
            const T value = label[std::size_t(ch) * g.label_h * g.label_w + pixel];
            if (coverage[i] && out[ch * g.count + i] != value) conflict[i] = true;
            if (choose) out[ch * g.count + i] = value;
        }
        if (choose) {
            best[i] = d;
            winner[i] = ids[pos];
            uvOut[2 * i] = x;
            uvOut[2 * i + 1] = y;
        }
        coverage[i] = true;
    }
}
}  // namespace vcr::projection
#undef VCR_PROJECTION_HD
