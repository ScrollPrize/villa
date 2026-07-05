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
// memory bandwidth. Level 1 both compresses better and runs ~2x faster than
// level 3 on this data: the delta residuals are noise-like, so the greedy
// match search at levels 2-12 emits statistically-coincidental short matches
// whose offsets cost more than huffman-coded literals; level 1 finds almost
// no matches and lands near the order-0 entropy of the residuals (verified
// per-chunk across regions and pyramid levels).
//
// Optionally the voxels are quantized before filtering (near-lossless mode):
// each value is snapped to the nearest multiple of quantBinWidth, so the
// per-voxel error is bounded by quantBinWidth/2 and masked zeros stay
// exactly zero. Width 3 (max error +-1) shrinks scroll CT chunks by a
// further ~20-25% over lossless. The width is recorded in the header so
// recompression can tell what an existing payload already carries;
// quantization is idempotent, so re-encoding at the same width is lossless.
//
// VCZ1 layout (all integers little-endian):
//   0..3   magic 'V' 'C' 'Z' '1'
//   4      format version (1)
//   5      element size in bytes (1 or 2)
//   6      quantization bin width (0 and 1 both mean lossless)
//   7      reserved (0)
//   8..19  chunk dims as three uint32: z, y, x (element counts)
//   20..   zstd frame of the filtered payload (z*y*x*elemSize bytes)
inline constexpr int kCacheCompressionLevel = 1;
inline constexpr const char* kCompressedCacheExtension = ".zst";

// Codec name used when a zarr array stores chunks in this format directly
// (v2 "compressor" id / v3 codec name).
inline constexpr const char* kDeltaZstdCodecName = "vc_delta_zstd";

// Near-lossless quantization bin widths offered in UIs. Width 1 is
// lossless; width 2k+1 bounds the per-voxel error by +-k.
inline constexpr int kCacheQuantLossless = 1;
inline constexpr int kCacheQuantMaxErr1 = 3;
inline constexpr int kCacheQuantMaxErr2 = 5;

// Compresses one decoded chunk of shapeZYX elements of elemSize bytes.
// quantBinWidth 1 is lossless; larger widths quantize first (see above).
// Throws std::invalid_argument unless input.size() equals z*y*x*elemSize
// with elemSize 1 or 2, and quantBinWidth is in [1, 255].
std::vector<std::byte> cacheCompress(std::span<const std::byte> input,
                                     std::array<int, 3> shapeZYX,
                                     std::size_t elemSize,
                                     int level = kCacheCompressionLevel,
                                     int quantBinWidth = kCacheQuantLossless);

// In-place near-lossless quantization as applied by cacheCompress: snaps
// each element to the nearest multiple of quantBinWidth (clamped to the
// dtype max; 0 stays 0). Width 1 is a no-op. Exposed so recompression
// tools can compute the expected decoded bytes for verification.
void cacheQuantize(std::span<std::byte> data,
                   std::size_t elemSize,
                   int quantBinWidth);

// Quantization bin width recorded in a VCZ1 payload (>= 1), or std::nullopt
// if input is not a VCZ1 payload. Payloads written before quantization
// existed report 1 (lossless).
std::optional<int> cacheQuantBinWidth(std::span<const std::byte> input);

// Decompresses a VCZ1 payload whose decoded size must equal expectedSize.
// Returns std::nullopt on any error or size mismatch (treated by callers
// as a corrupt cache entry).
std::optional<std::vector<std::byte>> cacheDecompress(
    std::span<const std::byte> input,
    std::size_t expectedSize);

} // namespace vc
