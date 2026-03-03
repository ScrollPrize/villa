#include "vc/core/cache/SimpleCacheFactory.hpp"

#include "vc/core/cache/ChunkSource.hpp"
#include "vc/core/cache/TieredChunkCache.hpp"
#include "vc/core/cache/VcDecompressor.hpp"
#include "vc/core/types/VcDataset.hpp"

namespace vc::cache {

std::unique_ptr<TieredChunkCache> createSimpleTieredCache(
    VcDataset* ds, size_t maxBytes, const std::filesystem::path& datasetPath)
{
    // Build single-level metadata from the dataset
    FileSystemChunkSource::LevelMeta lm;
    const auto& shape = ds->shape();
    const auto& chunks = ds->defaultChunkShape();
    lm.shape = {
        static_cast<int>(shape[0]),
        static_cast<int>(shape[1]),
        static_cast<int>(shape[2])};
    lm.chunkShape = {
        static_cast<int>(chunks[0]),
        static_cast<int>(chunks[1]),
        static_cast<int>(chunks[2])};

    std::string delimiter = ds->delimiter();

    // datasetPath should be the parent of the "0" directory (the zarr root)
    auto zarrRoot = datasetPath.parent_path();
    auto source = std::make_unique<FileSystemChunkSource>(
        zarrRoot, delimiter, std::vector{lm});

    auto decompress = makeVcDecompressor(ds);

    TieredChunkCache::Config config;
    config.hotMaxBytes = maxBytes;
    config.warmMaxBytes = 0;

    return std::make_unique<TieredChunkCache>(
        std::move(config),
        std::move(source),
        std::move(decompress),
        nullptr);  // no disk store
}

}  // namespace vc::cache
