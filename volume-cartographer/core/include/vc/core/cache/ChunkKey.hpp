#pragma once

#include <cstddef>
#include <cstdint>
#include <utils/hash.hpp>

namespace vc::cache {

// Identifies a chunk in a multi-resolution volume pyramid.
// All indices use logical (z, y, x) order.
struct ChunkKey {
    int level = 0;  // pyramid level (0 = full res, higher = coarser)
    int iz = 0;     // chunk index along z (depth)
    int iy = 0;     // chunk index along y (height)
    int ix = 0;     // chunk index along x (width)

    bool operator==(const ChunkKey& o) const noexcept
    {
        return level == o.level && iz == o.iz && iy == o.iy && ix == o.ix;
    }

    bool operator!=(const ChunkKey& o) const noexcept { return !(*this == o); }

    // Return the equivalent key at a coarser pyramid level.
    // Each level halves spatial resolution, so chunk indices halve.
    [[nodiscard]] ChunkKey coarsen(int targetLevel) const noexcept
    {
        int shift = targetLevel - level;
        return {targetLevel, iz >> shift, iy >> shift, ix >> shift};
    }
};

struct ChunkKeyHash {
    size_t operator()(const ChunkKey& k) const noexcept
    {
        return utils::hash_combine_values(k.level, k.iz, k.iy, k.ix);
    }
};

}  // namespace vc::cache
