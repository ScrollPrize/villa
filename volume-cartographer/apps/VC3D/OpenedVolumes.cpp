#include "OpenedVolumes.hpp"

#include "vc/core/types/Volume.hpp"

OpenedVolumes::OpenedVolumes(std::size_t cacheBytes) : cache_(std::make_shared<vcr::MemoryCache>(cacheBytes)) {}

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
    return {open(volume, base + level), open(volume, base)->placement};
}
