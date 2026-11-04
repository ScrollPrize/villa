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
// (indices ZYX), or with `general`, base[a] + sum_b linear[3a + b] * (p[b] - ref[b]) + local[a].
struct LevelMap {
    float scale[3], ref[3], local[3];
    std::int64_t base[3];
    bool general;
    float linear[9];
};

// The map of index = scale * p + offset (per axis, ZYX from XYZ), or of
// index = linear p + offset when `linear` is set, about the XYZ point `at`.
inline LevelMap levelMap(const std::array<double, 3>& scale, const std::array<double, 3>& offset,
                         const std::array<double, 9>* linear, const float* at)
{
    LevelMap m{};
    for (int a = 0; a < 3; ++a)
        m.ref[a] = std::isfinite(at[a]) ? at[a] : 0.0f;
    m.general = linear != nullptr;
    for (int a = 0; a < 3; ++a) {
        double index = offset[a];
        if (linear)
            for (int b = 0; b < 3; ++b) {
                index += (*linear)[std::size_t(3 * a + b)] * double(m.ref[b]);
                m.linear[3 * a + b] = float((*linear)[std::size_t(3 * a + b)]);
            }
        else
            index += scale[a] * double(m.ref[2 - a]);
        m.base[a] = std::int64_t(std::floor(index));
        m.scale[a] = float(scale[a]);
        m.local[a] = float(index - double(m.base[a]));
    }
    return m;
}

// The indices of the XYZ point (x, y, z). Scalars, not an array: a vectorized load of a
// point just stored one coordinate at a time stalls on store forwarding. With General
// false, the caller knows no map is general (the per-sample check costs ~5% on CPU).
template <bool General = true>
VCR_LEVEL_MAP_HD inline void levelIndex(const LevelMap& m, float x, float y, float z, float* c)
{
    if (General && m.general) {
        const float d[3]{x - m.ref[0], y - m.ref[1], z - m.ref[2]};
        for (int a = 0; a < 3; ++a)
            c[a] = m.linear[3 * a] * d[0] + m.linear[3 * a + 1] * d[1] + m.linear[3 * a + 2] * d[2] + m.local[a];
        return;
    }
    c[0] = m.scale[0] * (z - m.ref[2]) + m.local[0];
    c[1] = m.scale[1] * (y - m.ref[1]) + m.local[1];
    c[2] = m.scale[2] * (x - m.ref[0]) + m.local[2];
}

}  // namespace vcr::grid

#undef VCR_LEVEL_MAP_HD
