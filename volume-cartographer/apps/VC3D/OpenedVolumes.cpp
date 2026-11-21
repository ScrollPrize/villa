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
        result.offlineMisses += stats.offline_misses.load();
    }
    return result;
}

std::vector<OpenedVolumes::OfflineMisses> OpenedVolumes::offlineMisses()
{
    std::lock_guard lock(mutex_);
    std::vector<OfflineMisses> result;
    for (const auto& [key, level] : levels_) {
        if (!level) continue;
        auto chunks = level->array->offline_misses();
        if (!chunks.empty())
            result.push_back({key.first, key.second, std::move(chunks)});
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
    const auto channel = opened->meta.shape.size() == 4
        ? std::optional<std::int64_t>(volume.channel().value_or(0)) : std::nullopt;
    std::shared_ptr<vcr::reader_cuda::Reader> reader;
    if (gpuCacheBytes_ && opened->meta.dtype == "|u1" &&
        (opened->meta.shape.size() == 3 || opened->meta.shape.size() == 4)) {
        auto& shared = readers_[{opened->array.get(), channel}];
        if (!shared)
            shared = vc::cuda::makeReader(*opened->array, opened->meta.type, gpuPool_, channel);
        reader = shared;
    }
    // Level zero is a coordinate frame, not necessarily a stored array: compact
    // predictions often start at /2. A nonzero working base is a retained level.
    const auto working = base == 0 ? vcr::zarr::LevelPlacement{} : open(volume, base)->placement;
    return {opened, working, std::move(reader), channel};
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
