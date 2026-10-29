// One point's value through a caller's voxel lookup, shared by the CPU and CUDA raster
// samplers: Array::sample_nearest / sample_linear's definition (bounds, tie rule,
// interpolation on the axes a point is off the grid) for float32 coordinates relative
// to an integer origin.
#pragma once
#include "sampling_math.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>

#ifdef __CUDACC__
#define VCR_POINT_HD __host__ __device__
#else
#define VCR_POINT_HD
#endif

namespace vcr::sampling {

struct PointCell {
    using value_type = float;
    std::int64_t lo[3];
    float frac[3];
    std::size_t axes[3];
    std::size_t k = 0;
};

// The value at index-space point base + c (ZYX): nearest (order 0) or linear. Voxels
// outside the volume read as fill; non-finite linear points give NaN. False when a
// voxel's block is not available. A Lookup provides
//   shape[3], block[3] (edges of the blocks voxels are stored in), stride[3] (byte
//   steps inside a block), float fill, float load(const unsigned char*), and
//   const unsigned char* voxel(const std::int64_t* v, bool& available): the bytes of
//   voxel v (inside the volume), or null when its block holds nothing (fill) or is
//   not available (available = false).
template <class Lookup>
VCR_POINT_HD inline bool point(Lookup& L, const std::int64_t* base, const float* c, int order, float& value)
{
    if (order == 0) {
        std::int64_t v[3];
        for (int a = 0; a < 3; ++a) {
            if (!(c[a] > float(-1 - base[a]) && c[a] < float(L.shape[a] - base[a]) + 1.0f)) {
                value = L.fill;
                return true;
            }
            const float low = floorf(c[a]);
            v[a] = base[a] + std::int64_t(low) + nearest_rounds_up(c[a] - low);
            if (v[a] < 0 || v[a] >= L.shape[a]) {
                value = L.fill;
                return true;
            }
        }
        bool available = true;
        const unsigned char* at = L.voxel(v, available);
        if (!available)
            return false;
        value = at ? L.load(at) : L.fill;
        return true;
    }
    bool finite = true, outside = false;
    for (int a = 0; a < 3; ++a) {
        finite &= std::isfinite(c[a]);
        outside |= !(c[a] > float(-1 - base[a]) && c[a] < float(L.shape[a] - base[a]));
    }
    if (!finite || outside) {  // no neighbour inside the volume
        value = finite ? L.fill : NAN;
        return true;
    }
    PointCell cell;
    for (int a = 0; a < 3; ++a) {
        const float low = floorf(c[a]);
        cell.lo[a] = base[a] + std::int64_t(low);
        cell.frac[a] = c[a] - low;
        if (cell.frac[a] != 0.0f)
            cell.axes[cell.k++] = std::size_t(a);
    }
    const auto fill = [&](std::size_t) { return L.fill; };
    // Every neighbour inside the volume and in the near corner's block: read that block.
    bool whole = cell.lo[0] >= 0 && cell.lo[1] >= 0 && cell.lo[2] >= 0;
    for (std::size_t i = 0; i < cell.k; ++i) {
        const std::size_t a = cell.axes[i];
        whole &= cell.lo[a] + 1 < L.shape[a] && cell.lo[a] % L.block[a] + 1 < L.block[a];
    }
    if (whole) {
        bool available = true;
        const unsigned char* at = L.voxel(cell.lo, available);
        if (!available)
            return false;
        if (!at) {
            value = interpolate(cell, fill);
            return true;
        }
        const auto load = [&](const unsigned char* p) { return L.load(p); };
        if (cell.k == 3) {
            value = trilinear(cell.frac, at, L.stride, load);
            return true;
        }
        value = interpolate(cell, [&](std::size_t corner) {
            std::int64_t offset = 0;
            for (std::size_t i = 0; i < cell.k; ++i)
                if (corner >> i & 1)
                    offset += L.stride[cell.axes[i]];
            return L.load(at + offset);
        });
        return true;
    }
    bool available = true;
    value = interpolate(cell, [&](std::size_t corner) {
        std::int64_t v[3]{cell.lo[0], cell.lo[1], cell.lo[2]};
        for (std::size_t i = 0; i < cell.k; ++i)
            v[cell.axes[i]] += corner >> i & 1;
        if (v[0] < 0 || v[1] < 0 || v[2] < 0 || v[0] >= L.shape[0] || v[1] >= L.shape[1] || v[2] >= L.shape[2])
            return L.fill;
        bool here = true;
        const unsigned char* at = L.voxel(v, here);
        available &= here;
        return at ? L.load(at) : L.fill;
    });
    return available;
}

}  // namespace vcr::sampling

#undef VCR_POINT_HD
