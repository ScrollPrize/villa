#pragma once
// Arrays whose chunks are computed on demand from other data, e.g. a distance
// transform of a thresholded CT volume. A computed array is an ordinary vcr::Array:
// read and sample it like a stored one. The compute callback reads its inputs
// through vc_reader (typically region reads of other arrays). Computed chunks can
// persist in a decoded disk cache that later opens reuse.

#include "vc_reader/native/array.hpp"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>

namespace vcr::computed {

struct Options {
    Index shape, chunks;
    std::string dtype = "|u1";      // numpy typestr: |u1, <u2, <f4, ...
    nlohmann::json fillValue = 0;
    // Names everything the chunk values depend on: inputs, parameters and the
    // computation's version. Required with cacheDir; chunks persisted under another
    // identity are never read.
    std::string identity;
    // Persist computed chunks under cacheDir/<hash of identity> (uncompressed, or
    // lossless Delta3D with compressCache). Opening the same identity with another
    // shape, chunk shape, dtype or fill is an error.
    std::optional<std::filesystem::path> cacheDir;
    // Byte budget shared by everything under cacheDir (oldest chunks are evicted); unbounded when unset.
    std::optional<std::int64_t> cacheMaxBytes;
    bool compressCache = false;
    std::shared_ptr<MemoryCache> memoryCache;  // decoded chunks in memory; none when unset
    std::size_t threads = 0;  // region-read and sampling workers; 0: min(16, cores)
};

// A computed array. Each chunk is computed by one caller at a time, then kept in
// the memory cache and on disk when those are set.
std::unique_ptr<Array> open(const Options& options, ChunkCompute compute);

// The directory holding the persisted chunks of `options` (requires cacheDir).
std::filesystem::path cachePath(const Options& options);

}  // namespace vcr::computed
