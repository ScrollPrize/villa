#pragma once

#include <cstddef>
#include <cstdint>
#include <utils/hash.hpp>

namespace vc::cache {

// Upper bound on pyramid levels (levels 0..kMaxLevels-1). Real pipelines
// use up to 6; the extra headroom costs nothing.
constexpr int kMaxLevels = 8;

// Identifies a chunk in a multi-resolution volume pyramid.
// All indices use logical (z, y, x) order.
struct ChunkKey {
    int level = 0;  // pyramid level (0 = full res, higher = coarser)
    int iz = 0;     // chunk index along z (depth)
    int iy = 0;     // chunk index along y (height)
    int ix = 0;     // chunk index along x (width)

    constexpr bool operator==(const ChunkKey& o) const noexcept
    {
        return level == o.level && iz == o.iz && iy == o.iy && ix == o.ix;
    }

    constexpr bool operator!=(const ChunkKey& o) const noexcept { return !(*this == o); }

    // Return the equivalent key at a coarser pyramid level.
    // Each level halves spatial resolution, so chunk indices halve.
    [[nodiscard]] constexpr ChunkKey coarsen(int targetLevel) const noexcept
    {
        if (targetLevel <= level) return *this;
        int shift = targetLevel - level;
        return {targetLevel, iz >> shift, iy >> shift, ix >> shift};
    }
};

struct ChunkKeyHash {
    size_t operator()(const ChunkKey& k) const noexcept
    {
        const uint64_t hi = (uint64_t(uint32_t(k.level)) << 40)
                          ^ (uint64_t(uint32_t(k.iz)) << 20)
                          ^  uint64_t(uint32_t(k.iy));
        const uint64_t lo = (uint64_t(uint32_t(k.iy)) << 32)
                          |  uint64_t(uint32_t(k.ix));
        return size_t(hi ^ (lo * 0x9E3779B97F4A7C15ULL));
    }
};

// Identifies a shard in a multi-resolution volume pyramid.
// For sharded datasets: maps to the shard grid (sz, sy, sx).
// For non-sharded datasets: maps 1:1 to chunk indices (same as ChunkKey coords).
struct ShardKey {
    int level = 0;
    int sz = 0, sy = 0, sx = 0;

    constexpr bool operator==(const ShardKey&) const noexcept = default;
    constexpr bool operator!=(const ShardKey& o) const noexcept { return !(*this == o); }
};

struct ShardKeyHash {
    size_t operator()(const ShardKey& k) const noexcept
    {
        const uint64_t hi = (uint64_t(uint32_t(k.level)) << 40)
                          ^ (uint64_t(uint32_t(k.sz)) << 20)
                          ^  uint64_t(uint32_t(k.sy));
        const uint64_t lo = (uint64_t(uint32_t(k.sy)) << 32)
                          |  uint64_t(uint32_t(k.sx));
        return size_t(hi ^ (lo * 0x9E3779B97F4A7C15ULL));
    }
};

}  // namespace vc::cache
