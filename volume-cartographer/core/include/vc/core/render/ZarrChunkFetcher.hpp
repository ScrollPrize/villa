#pragma once

#include "vc/core/render/ChunkCache.hpp"
#include "vc/core/render/ChunkFetch.hpp"
#include "vc/core/render/IChunkedArray.hpp"
#include "vc/core/util/RemoteAuth.hpp"

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace vc::render {

struct OpenedChunkedZarr {
    std::vector<int> levelNumbers;
    std::vector<IChunkedArray::LevelTransform> transforms;
    std::vector<std::array<int, 3>> shapes;
    std::vector<std::array<int, 3>> chunkShapes;
    std::vector<std::array<int, 3>> storageChunkShapes;
    std::vector<std::shared_ptr<IChunkFetcher>> fetchers;
    std::vector<double> fillValues;
    double fillValue = 0.0;
    ChunkDtype dtype = ChunkDtype::UInt8;
    // True when the physical /0 OME coordinate transform is absent or an
    // identity scale with zero translation. This survives logical rebasing so
    // catalog prediction/source preflight can enforce prediction identity.
    bool physicalLevelZeroTransformIsIdentity = true;
};

OpenedChunkedZarr openLocalZarrPyramid(const std::filesystem::path& root);
OpenedChunkedZarr openHttpZarrPyramid(const std::string& url);
OpenedChunkedZarr openHttpZarrPyramid(
    const std::string& url,
    const vc::HttpAuth& auth,
    std::optional<int> baseScaleLevel = std::nullopt);

// Enforce the supported contiguous dyadic VC pyramid contract and make
// physical level baseScaleLevel logical level zero. Exposed for deterministic
// synthetic tests; remote nonzero opens call the same implementation.
OpenedChunkedZarr validateAndRebaseVcPyramid(
    OpenedChunkedZarr opened,
    int baseScaleLevel);

std::unique_ptr<ChunkCache> createChunkCache(
    OpenedChunkedZarr opened,
    std::size_t decodedByteCapacity,
    std::size_t maxConcurrentReads = 16);

} // namespace vc::render
