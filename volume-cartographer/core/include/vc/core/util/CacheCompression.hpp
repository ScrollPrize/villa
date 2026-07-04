#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <span>
#include <vector>

namespace vc {

// Lossless compression for persistent chunk-cache payloads and
// "vc_delta_zstd"-coded zarr chunks.
//
// Chunks cached from remote volumes with a raw/uncompressed source are stored
// as plain decoded bytes (".bin"). cacheCompress() converts such payloads to
// the self-describing "VCZ1" format (".zst" cache files): a small header
// followed by a zstd frame of the delta-zyx-filtered voxels. The filter
// stores each element as the difference from its predecessor along z, then y,
// then x (mod 2^8/2^16), which roughly halves the compressed size of scroll
// CT data compared to zstd on raw bytes while the filter itself runs at
// memory bandwidth. Level 3 keeps compression fast enough to run inline on
// the cache writer thread while decompression stays well above typical
// download bandwidth.
//
// VCZ1 layout (all integers little-endian):
//   0..3   magic 'V' 'C' 'Z' '1'
//   4      format version (1)
//   5      element size in bytes (1 or 2)
//   6..7   reserved (0)
//   8..19  chunk dims as three uint32: z, y, x (element counts)
//   20..   zstd frame of the filtered payload (z*y*x*elemSize bytes)
inline constexpr int kCacheCompressionLevel = 3;
inline constexpr const char* kCompressedCacheExtension = ".zst";

// Codec name used when a zarr array stores chunks in this format directly
// (v2 "compressor" id / v3 codec name).
inline constexpr const char* kDeltaZstdCodecName = "vc_delta_zstd";

// Compresses one decoded chunk of shapeZYX elements of elemSize bytes.
// input.size() must equal z*y*x*elemSize and elemSize must be 1 or 2;
// otherwise the payload is stored as a legacy plain zstd frame (no filter).
std::vector<std::byte> cacheCompress(std::span<const std::byte> input,
                                     std::array<int, 3> shapeZYX,
                                     std::size_t elemSize,
                                     int level = kCacheCompressionLevel);

// Decompresses a VCZ1 payload or a legacy plain zstd frame (the format
// written before the delta filter existed). The decoded size must equal
// expectedSize. Returns std::nullopt on any error or size mismatch
// (treated by callers as a corrupt cache entry).
std::optional<std::vector<std::byte>> cacheDecompress(
    std::span<const std::byte> input,
    std::size_t expectedSize);

} // namespace vc
