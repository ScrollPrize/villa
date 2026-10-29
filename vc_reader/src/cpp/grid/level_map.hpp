// Where a level's indices sit for the points of one raster, shared by the CPU and CUDA
// samplers: float32 relative to an integer origin near the raster, so precision does
// not depend on where in the volume the raster lies.
#pragma once
#include <array>
#include <cmath>
#include <cstdint>

#ifdef __CUDACC__
#define VCR_LEVEL_MAP_HD __host__ __device__
#else
#define VCR_LEVEL_MAP_HD
#endif

namespace vcr::grid {

// index[a] = base[a] + scale[a] * (p[2 - a] - ref[2 - a]) + local[a] for an XYZ point p
// (indices ZYX).
struct LevelMap {
    float scale[3], ref[3], local[3];
    std::int64_t base[3];
};

// The map of index = scale * p + offset (per axis, ZYX from XYZ) about the XYZ point `at`.
inline LevelMap levelMap(const std::array<double, 3>& scale, const std::array<double, 3>& offset, const float* at)
{
    LevelMap m{};
    for (int a = 0; a < 3; ++a)
        m.ref[a] = std::isfinite(at[a]) ? at[a] : 0.0f;
    for (int a = 0; a < 3; ++a) {
        const double index = scale[a] * double(m.ref[2 - a]) + offset[a];
        m.base[a] = std::int64_t(std::floor(index));
        m.scale[a] = float(scale[a]);
        m.local[a] = float(index - double(m.base[a]));
    }
    return m;
}

// The indices of the XYZ point (x, y, z). Scalars, not an array: a vectorized load of a
// point just stored one coordinate at a time stalls on store forwarding.
VCR_LEVEL_MAP_HD inline void levelIndex(const LevelMap& m, float x, float y, float z, float* c)
{
    c[0] = m.scale[0] * (z - m.ref[2]) + m.local[0];
    c[1] = m.scale[1] * (y - m.ref[1]) + m.local[1];
    c[2] = m.scale[2] * (x - m.ref[0]) + m.local[2];
}

}  // namespace vcr::grid

#undef VCR_LEVEL_MAP_HD
