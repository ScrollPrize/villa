#pragma once

// volcodec_ng codec wrapper.  Thin shim around libs/volcodec_ng
// (volcodec_ng_lossless.h) that mirrors utils/volcomp_codec.hpp, so VcDataset
// dispatches on the same surface.
//
// volcodec_ng is lossless: every voxel decodes exactly as it was stored.  Its
// chunk atom is fixed at 128^3 u8 (2 MiB raw), like volcomp's.  Chunks start
// with the "VCNG" magic and record their own mode, so neither encoding nor
// decoding takes a parameter.

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace utils {

inline constexpr int kVolcodecNgChunkSide = 128;
inline constexpr std::size_t kVolcodecNgChunkBytes =
    static_cast<std::size_t>(kVolcodecNgChunkSide) * kVolcodecNgChunkSide * kVolcodecNgChunkSide;

// "avx2", "neon" or "c": which kernel set this process uses (same bytes either way).
[[nodiscard]] const char* volcodec_ng_kernels() noexcept;
// Stream revision the encoder writes (the decoder reads 1 up to this).
[[nodiscard]] unsigned volcodec_ng_format_revision() noexcept;

[[nodiscard]] std::vector<std::byte> volcodec_ng_encode(std::span<const std::byte> raw);

[[nodiscard]] std::vector<std::byte> volcodec_ng_decode(
    std::span<const std::byte> compressed, std::size_t out_size);

// Decode straight into a caller buffer of exactly kVolcodecNgChunkBytes.
void volcodec_ng_decode_into(std::span<const std::byte> compressed,
                             std::span<std::byte> out);

// Magic sniff: buffer begins with "VCNG" and a revision this decoder reads.
[[nodiscard]] bool is_volcodec_ng_compressed(std::span<const std::byte> data) noexcept;

// Chunk mode recorded in the header: 2 raw, 3 constant, 6 lossless (5, an
// earlier lossless mode, is read too); -1 when not a volcodec_ng chunk.
[[nodiscard]] int volcodec_ng_chunk_mode(std::span<const std::byte> data) noexcept;

}  // namespace utils
