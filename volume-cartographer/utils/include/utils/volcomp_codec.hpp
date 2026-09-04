#pragma once

// volcomp codec wrapper.  Thin shim around libs/volcomp (volcomp.h) that
// mirrors utils/c3d_codec.hpp so VcDataset / recompress tools dispatch on a
// common surface.
//
// volcomp's chunk atom is fixed at 128^3 u8 (2 MiB raw).  Every encoded chunk
// starts with the "VOLC" magic and records its quantiser step q, so no
// wrapping header is needed.  Decoding needs nothing but the bytes; q is only
// an encode parameter.

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace utils {

inline constexpr int kVolcompChunkSide = 128;
inline constexpr std::size_t kVolcompChunkBytes =
    static_cast<std::size_t>(kVolcompChunkSide) * kVolcompChunkSide * kVolcompChunkSide;

struct VolcompCodecParams {
    // Quantiser step in voxel units, 1..255.  Error percentiles scale with q
    // (P99 ≈ 2.5q on scroll CT); 8 is the archive default (≈ 40 dB PSNR,
    // ~40x), 4 is near-transparent, 16-32 suit coarse pyramid levels.
    float q = 8.0f;
};

// True when the codec is compiled in for this target and the CPU supports it
// (AVX2+FMA on x86-64).  Encode/decode throw when it is false.
[[nodiscard]] bool volcomp_available() noexcept;

[[nodiscard]] std::vector<std::byte> volcomp_encode(
    std::span<const std::byte> raw, const VolcompCodecParams& params);

[[nodiscard]] std::vector<std::byte> volcomp_decode(
    std::span<const std::byte> compressed, std::size_t out_size);

// Decode straight into a caller buffer of exactly kVolcompChunkBytes.
void volcomp_decode_into(std::span<const std::byte> compressed,
                         std::span<std::byte> out);

// Decode one 16^3 block (bz,by,bx in 0..7) into a 4096-byte buffer without
// decoding the rest of the chunk (touches <= 16 blocks of entropy data).
void volcomp_decode_block_into(std::span<const std::byte> compressed,
                               unsigned bz, unsigned by, unsigned bx,
                               std::span<std::byte> out);

// Magic sniff: buffer begins with "VOLC" and a supported version.
[[nodiscard]] bool is_volcomp_compressed(std::span<const std::byte> data) noexcept;

// q recorded in a chunk header (0 when not a volcomp chunk).
[[nodiscard]] float volcomp_chunk_q(std::span<const std::byte> data) noexcept;

// volcomp chunks are always 128^3; {Z, Y, X} for symmetry with c3d_header_dims().
[[nodiscard]] inline std::array<int, 3> volcomp_header_dims(
    std::span<const std::byte>) noexcept { return {128, 128, 128}; }

}  // namespace utils
