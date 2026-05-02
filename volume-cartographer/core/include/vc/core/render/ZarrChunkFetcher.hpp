#pragma once

#include "vc/core/render/ChunkCache.hpp"
#include "vc/core/render/ChunkFetch.hpp"
#include "vc/core/render/IChunkedArray.hpp"
#include "vc/core/util/RemoteAuth.hpp"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace vc::render {

struct OpenedChunkedZarr {
    std::vector<IChunkedArray::LevelTransform> transforms;
    std::vector<std::array<int, 3>> shapes;
    std::vector<std::array<int, 3>> chunkShapes;
    std::vector<std::shared_ptr<IChunkFetcher>> fetchers;
    double fillValue = 0.0;
    ChunkDtype dtype = ChunkDtype::UInt8;
};

OpenedChunkedZarr openLocalZarrPyramid(const std::filesystem::path& root);
OpenedChunkedZarr openHttpZarrPyramid(const std::string& url);
OpenedChunkedZarr openHttpZarrPyramid(
    const std::string& url,
    const vc::HttpAuth& auth,
    int baseScaleLevel = 0);

std::unique_ptr<ChunkCache> createChunkCache(
    OpenedChunkedZarr opened,
    std::size_t decodedByteCapacity,
    std::size_t maxConcurrentReads = 16);

} // namespace vc::render
