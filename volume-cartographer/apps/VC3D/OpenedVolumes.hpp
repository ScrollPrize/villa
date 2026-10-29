#pragma once

#include <vc_reader/zarr.hpp>

#include <cstddef>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

class Volume;

// The pyramid levels of volumes that VC3D's tools sample, each opened once and
// shared, holding their decoded chunks in one memory cache.
class OpenedVolumes {
public:
    struct Level {
        std::shared_ptr<vcr::zarr::OpenedArray> array;  // the opened physical level
        vcr::zarr::LevelPlacement working;              // placement of the volume's working level 0
    };

    explicit OpenedVolumes(std::size_t cacheBytes);

    // Working level `level` of `volume` (physical level level + baseScaleLevel()),
    // opened on first use. Map volume coordinates to the level's indices with
    // array->placement.fromLevel0(axis, working.toLevel0(axis, c)). Thread-safe.
    Level level(const Volume& volume, int level);

    struct CacheStats {
        std::size_t bytes = 0, capacity = 0;
    };
    // The decoded bytes the shared memory cache holds, and its limit.
    CacheStats cacheStats() const { return {cache_->nbytes(), cache_->max_bytes()}; }

private:
    std::shared_ptr<vcr::zarr::OpenedArray> open(const Volume& volume, int physical);

    std::mutex mutex_;
    std::shared_ptr<vcr::MemoryCache> cache_;
    std::map<std::pair<std::string, int>, std::shared_ptr<vcr::zarr::OpenedArray>> levels_;  // (location, level)
};
