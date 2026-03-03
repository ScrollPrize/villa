#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>

namespace vc {
class VcDataset;
}

namespace vc::cache {

class TieredChunkCache;

// Factory for creating a single-dataset TieredChunkCache (for CLI tools,
// tracer, ChunkedTensor, etc. that only need one zarr level).
//
// Creates a 1-level cache backed by FileSystemChunkSource + VcDecompressor
// with hot = maxBytes, warm = 0 (no warm tier).
std::unique_ptr<TieredChunkCache> createSimpleTieredCache(
    VcDataset* ds, size_t maxBytes, const std::filesystem::path& datasetPath);

}  // namespace vc::cache
