#pragma once

// c3d codec wrapper.  Thin shim around libc3d that mirrors the
// utils/video_codec.hpp API shape so BlockPipeline / recompress tools
// can dispatch on a common surface.
//
// c3d's chunk atom is fixed at 256^3 u8.  Every encoded chunk starts
// with the "C3DC" magic, so no extra wrapping header is needed.

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace utils {

struct C3dCodecParams {
    // Target compression ratio (> 1.0).  Rate control is a log-space
    // bisection on a single quantizer scalar; see libs/c3d/README.md.
    // Reference points on scroll CT: 10 -> ~46 dB PSNR, 50 -> ~40 dB,
    // 100 -> ~35 dB.  Default 50 is the sweet spot for the disk cache:
    // the visual-quality drop from 10 -> 50 is negligible on 8-bit
    // scroll data but the shards are ~5x smaller.
    float target_ratio = 50.0f;

    // Cube dimensions (canonical 256^3). Kept as fields so callers can
    // reuse the same params-struct idiom across codecs.
    int depth  = 256;  // Z
    int height = 256;  // Y
    int width  = 256;  // X

    // Skip libc3d's post-decode denoise blur (LOD 0 only).  Trades ~0.03 dB
    // PSNR at the default r=50 for ~14% of decode CPU.  Set true for GUI
    // tile rendering; leave false for archival / lossless-at-ratio reads.
    // Ignored by encode.
    bool skip_denoise = false;
};

[[nodiscard]] std::vector<std::byte> c3d_encode(
    std::span<const std::byte> raw, const C3dCodecParams& params);

[[nodiscard]] std::vector<std::byte> c3d_decode(
    std::span<const std::byte> compressed, std::size_t out_size,
    const C3dCodecParams& params);

// LOD-scalable decode.  lod ∈ [0, 5]; output is (256 >> lod)^3 u8.
// lod=0 is equivalent to c3d_decode.  Higher lod values decode a prefix
// of the chunk's subbands (see c3d chunk header lod_offsets), producing
// a coarser reconstruction.  Cheaper than full-decode when the caller
// doesn't need the finest detail.
[[nodiscard]] std::vector<std::byte> c3d_decode_lod(
    std::span<const std::byte> compressed, std::uint8_t lod);

// Magic sniff: buffer begins with "C3DC".
[[nodiscard]] bool is_c3d_compressed(std::span<const std::byte> data) noexcept;

// c3d chunks are always 256^3; returned as {Z, Y, X} for symmetry with
// video_header_dims().
[[nodiscard]] inline std::array<int, 3> c3d_header_dims(
    std::span<const std::byte>) noexcept { return {256, 256, 256}; }

}  // namespace utils
