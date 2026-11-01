#include "OpenedVolumes.hpp"

#include "vc/core/types/Volume.hpp"

#include <vc_reader/native/cuda_available.hpp>
#ifdef VC_READER_CUDA
#include <vc_reader/native/reader_cuda/reader.hpp>
#endif

OpenedVolumes::OpenedVolumes(std::size_t cacheBytes, std::size_t gpuCacheBytes)
    : cache_(std::make_shared<vcr::MemoryCache>(cacheBytes))
    , gpuCacheBytes_(vcr::cuda::available() ? gpuCacheBytes : 0)
{
}

std::shared_ptr<vcr::zarr::OpenedArray> OpenedVolumes::open(const Volume& volume, int physical)
{
    auto& opened = levels_[{volume.sourceLocation(), physical}];
    if (!opened) {
        auto options = volume.sourceOpenOptions();
        options.memoryCache = cache_;
        opened = std::make_shared<vcr::zarr::OpenedArray>(
            vcr::zarr::openPyramidLevel(volume.sourceLocation(), physical, options));
    }
    return opened;
}

OpenedVolumes::Level OpenedVolumes::level(const Volume& volume, int level)
{
    std::lock_guard lock(mutex_);
    const int base = volume.baseScaleLevel();
    auto opened = open(volume, base + level);
    std::shared_ptr<vcr::reader_cuda::Reader> reader;
#ifdef VC_READER_CUDA
    if (gpuCacheBytes_ && opened->meta.dtype == "|u1" && opened->meta.shape.size() == 3) {
        auto& shared = readers_[opened->array.get()];
        if (!shared)
            shared = std::make_shared<vcr::reader_cuda::Reader>(*opened->array, gpuCacheBytes_, 0);
        reader = shared;
    }
#endif
    return {opened, open(volume, base)->placement, std::move(reader)};
}

const OpenedVolumes::Pyramid& OpenedVolumes::pyramid(const Volume& volume)
{
    std::lock_guard lock(mutex_);
    // Two working bases of one source describe different working pyramids, so the
    // description is kept per base; the opened physical arrays are shared regardless.
    const std::pair<std::string, int> key{volume.sourceLocation(), volume.baseScaleLevel()};
    if (const auto found = pyramids_.find(key); found != pyramids_.end())
        return found->second;
    Pyramid result;
    for (const auto& level : vcr::zarr::pyramidLevels(key.first, volume.sourceOpenOptions())) {
        if (level.level >= key.second)
            result.levels.push_back(level.level - key.second);
        result.dtype = level.meta.dtype;
    }
    return pyramids_.emplace(key, std::move(result)).first->second;
}
