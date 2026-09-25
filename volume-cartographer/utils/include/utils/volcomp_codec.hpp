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
#include <optional>
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

// Always true: the codec is portable (AVX2 kernels picked at runtime on x86-64,
// plain C elsewhere).  Kept so call sites can stay defensive.
[[nodiscard]] bool volcomp_available() noexcept;
// "avx2" or "c": which kernel set this process uses.
[[nodiscard]] const char* volcomp_kernels() noexcept;

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

// Upstream library version ("1.3.0") and format revision (4 = surface
// chunks, header mode 6). volcomp_decode* read every mode of that revision.
[[nodiscard]] const char* volcomp_version() noexcept;
[[nodiscard]] unsigned volcomp_format_revision() noexcept;

// Surface chunk (mode 6): for thresholded probability fields (u8 = round(255 p)).
// A DCT base at step q (flat step law, q in [2, 255]; surface q 48 is about
// mode-0 q 16) plus a refinement that keeps (src >= thr) == (decoded >= thr)
// exact for every voxel. Decode with volcomp_decode* like any other chunk.
[[nodiscard]] std::vector<std::byte> volcomp_surface_encode(
    std::span<const std::byte> raw, float q, unsigned thr = 128);

struct VolcompSurfaceInfo {
    unsigned thr = 0;     // exact threshold
    unsigned margin = 0;  // refinement margin (0 = the base alone was exact)
};
// Threshold and margin of a surface chunk; nullopt for any other stream.
[[nodiscard]] std::optional<VolcompSurfaceInfo> volcomp_surface_info(
    std::span<const std::byte> data) noexcept;

// Opt-in decode-side deblocking (not part of the format, not used by any VC
// reader by default): decode, smooth voxels next to interior block faces with a
// Gaussian of `strength` voxels, then project each block back onto the
// quantisation cells it was decoded from. Lossy and surface chunks only (the
// latter keep their exact threshold); other chunks decode as volcomp_decode.
// zero_guard leaves exact zeros (masked air) and faces touching them alone.
void volcomp_decode_smooth_into(std::span<const std::byte> compressed,
                                std::span<std::byte> out, float strength,
                                bool zero_guard = true);

// volcomp chunks are always 128^3; {Z, Y, X} for symmetry with c3d_header_dims().
[[nodiscard]] inline std::array<int, 3> volcomp_header_dims(
    std::span<const std::byte>) noexcept { return {128, 128, 128}; }

}  // namespace utils
