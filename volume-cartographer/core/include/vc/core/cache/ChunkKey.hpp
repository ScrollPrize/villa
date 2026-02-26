#pragma once

#include <cstddef>
#include <cstdint>

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
    // targetLevel must be >= level; if not, returns *this unchanged.
    [[nodiscard]] ChunkKey coarsen(int targetLevel) const noexcept
    {
        int shift = targetLevel - level;
        if (shift <= 0) return *this;
        return {targetLevel, iz >> shift, iy >> shift, ix >> shift};
    }
};

struct ChunkKeyHash {
    size_t operator()(const ChunkKey& k) const noexcept
    {
        // FNV-1a
        uint64_t h = 14695981039346656037ULL;
        h ^= static_cast<uint64_t>(k.level);
        h *= 1099511628211ULL;
        h ^= static_cast<uint64_t>(k.iz);
        h *= 1099511628211ULL;
        h ^= static_cast<uint64_t>(k.iy);
        h *= 1099511628211ULL;
        h ^= static_cast<uint64_t>(k.ix);
        h *= 1099511628211ULL;
        return static_cast<size_t>(h);
    }
};

}  // namespace vc::cache
