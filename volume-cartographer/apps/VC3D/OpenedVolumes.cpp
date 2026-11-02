#include "OpenedVolumes.hpp"

#include "vc/core/types/Volume.hpp"
#include "vc/core/util/CudaSampling.hpp"
#include "vc/core/util/Logging.hpp"

#include <vc_reader/array.hpp>
#include <vc_reader/cuda_available.hpp>

OpenedVolumes::OpenedVolumes(std::size_t cacheBytes, std::size_t gpuCacheBytes)
    : cache_(std::make_shared<vcr::MemoryCache>(cacheBytes))
    , gpuCacheBytes_(vcr::cuda::available() ? gpuCacheBytes : 0)
{
    if (gpuCacheBytes && !gpuCacheBytes_)
        Logger()->warn("GPU viewer sampling is on, but there is no CUDA device: sampling on the CPU");
    gpuPool_ = vc::cuda::makePool(gpuCacheBytes_);
}

OpenedVolumes::TransferStats OpenedVolumes::transferStats()
{
    std::lock_guard lock(mutex_);
    TransferStats result;
    for (const auto& [key, level] : levels_) {
        if (!level) continue;
        const auto& stats = level->array->stats();
        result.chunks += stats.remote_fetches.load();
        result.bytes += stats.remote_bytes.load();
    }
    return result;
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
    if (gpuCacheBytes_ && opened->meta.dtype == "|u1" && opened->meta.shape.size() == 3) {
        auto& shared = readers_[opened->array.get()];
        if (!shared)
            shared = vc::cuda::makeReader(*opened->array, opened->meta.type, gpuPool_);
        reader = shared;
    }
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
