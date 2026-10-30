#pragma once

#include <vc_reader/zarr.hpp>

#include <algorithm>
#include <cstddef>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

class Volume;

namespace vcr::reader_cuda { class Reader; }

// The pyramid levels of volumes that VC3D's tools sample, each opened once and
// shared, holding their decoded chunks in one memory cache.
class OpenedVolumes {
public:
    struct Level {
        std::shared_ptr<vcr::zarr::OpenedArray> array;  // the opened physical level
        vcr::zarr::LevelPlacement working;              // placement of the volume's working level 0
        // Samples the level on the GPU, when the render backend is CUDA and the
        // level is uint8; null otherwise.
        std::shared_ptr<vcr::reader_cuda::Reader> cuda;
    };

    // gpuCacheBytes > 0 samples frames on the GPU, keeping up to that many bytes of
    // each level resident (builds with CUDA, when a device is present).
    explicit OpenedVolumes(std::size_t cacheBytes, std::size_t gpuCacheBytes = 0);

    // Working level `level` of `volume` (physical level level + baseScaleLevel()),
    // opened on first use. Map volume coordinates to the level's indices with
    // array->placement.fromLevel0(axis, working.toLevel0(axis, c)). Thread-safe.
    Level level(const Volume& volume, int level);

    // What a volume's pyramid holds, from its metadata alone. Working levels: index
    // level + baseScaleLevel() of the physical pyramid; finer physical levels are not
    // listed. Read once per volume. Thread-safe; throws when the metadata cannot be read.
    struct Pyramid {
        std::vector<int> levels;  // the present working levels, ascending
        std::string dtype;        // NumPy typestr: |u1, <u2, <f4, ...

        int count() const { return levels.empty() ? 0 : levels.back() + 1; }
        bool has(int level) const { return std::binary_search(levels.begin(), levels.end(), level); }
    };
    const Pyramid& pyramid(const Volume& volume);

    struct CacheStats {
        std::size_t bytes = 0, capacity = 0;
    };
    // The decoded bytes the shared memory cache holds, and its limit.
    CacheStats cacheStats() const { return {cache_->nbytes(), cache_->max_bytes()}; }

private:
    std::shared_ptr<vcr::zarr::OpenedArray> open(const Volume& volume, int physical);

    std::mutex mutex_;
    std::shared_ptr<vcr::MemoryCache> cache_;
    std::size_t gpuCacheBytes_ = 0;
    std::map<std::pair<std::string, int>, std::shared_ptr<vcr::zarr::OpenedArray>> levels_;  // (location, level)
    std::map<const vcr::Array*, std::shared_ptr<vcr::reader_cuda::Reader>> readers_;
    std::map<std::string, Pyramid> pyramids_;  // by location
};
