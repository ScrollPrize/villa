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
    // Reference points on scroll CT: 10 -> ~46 dB PSNR, 100 -> ~35 dB.
    float target_ratio = 10.0f;

    // Cube dimensions.  Fixed at 256 for c3d; kept as fields so callers
    // can use the same params-struct idiom as VideoCodecParams.
    int depth  = 256;  // Z
    int height = 256;  // Y
    int width  = 256;  // X
};

[[nodiscard]] std::vector<std::byte> c3d_encode(
    std::span<const std::byte> raw, const C3dCodecParams& params);

[[nodiscard]] std::vector<std::byte> c3d_decode(
    std::span<const std::byte> compressed, std::size_t out_size,
    const C3dCodecParams& params);

// Magic sniff: buffer begins with "C3DC".
[[nodiscard]] bool is_c3d_compressed(std::span<const std::byte> data) noexcept;

// c3d chunks are always 256^3; returned as {Z, Y, X} for symmetry with
// video_header_dims().
[[nodiscard]] inline std::array<int, 3> c3d_header_dims(
    std::span<const std::byte>) noexcept { return {256, 256, 256}; }

}  // namespace utils
