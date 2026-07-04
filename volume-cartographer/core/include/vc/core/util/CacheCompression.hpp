#pragma once

#include <cstddef>
#include <optional>
#include <span>
#include <vector>

namespace vc {

// Lossless zstd compression for persistent chunk-cache payloads.
//
// Chunks cached from remote volumes with a raw/uncompressed source are stored
// as plain decoded bytes (".bin"). These helpers convert such payloads to and
// from single zstd frames (".zst" cache files). Level 3 keeps compression
// fast enough to run inline on the cache writer thread while decompression
// stays well above typical download bandwidth.
inline constexpr int kCacheCompressionLevel = 3;
inline constexpr const char* kCompressedCacheExtension = ".zst";

std::vector<std::byte> cacheCompress(std::span<const std::byte> input,
                                     int level = kCacheCompressionLevel);

// Decompresses a single zstd frame whose decoded size must equal
// expectedSize. Returns std::nullopt on any error or size mismatch
// (treated by callers as a corrupt cache entry).
std::optional<std::vector<std::byte>> cacheDecompress(
    std::span<const std::byte> input,
    std::size_t expectedSize);

} // namespace vc
