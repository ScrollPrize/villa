#pragma once

#include "vc/core/render/IChunkedArray.hpp"

namespace vc::render::detail {

// Internal bridge used by ROI implementation and physical channel adapters.
// Not installed/exported as a client-facing chunk-list read interface.
struct ChunkReadBatch {
    static bool enabled(const IChunkedArray& array) noexcept {
        return array.prefersBatchReads();
    }
    static std::vector<ChunkResult> read(IChunkedArray& array,
                                        const std::vector<ChunkKey>& keys,
                                        const ChunkRequestContext& request = {}) {
        return array.getChunksBlocking(keys, request);
    }
};

} // namespace vc::render::detail
