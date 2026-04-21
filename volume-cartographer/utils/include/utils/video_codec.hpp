#pragma once

// H.265 video codec compression for 3D cubic chunks.
//
// A 3D chunk of shape (Z, Y, X) is encoded as a Z-frame grayscale video
// sequence using H.265 (x265 encode, libde265 decode).
//
// Encoding: voxel values become the Y (luma) plane; monochrome I400 mode.
// Decoding: Y plane is extracted back to voxel values.

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace utils {

struct VideoCodecParams {
    // Quantization parameter (0-51). Lower = better quality, larger output.
    // 0 = lossless, 26 = default, 51 = worst quality.
    int qp = 26;

    // Chunk dimensions (Z frames of Y×X pixels). Must be set before encode.
    int depth = 0;   // Z
    int height = 0;  // Y
    int width = 0;   // X

    // Air-clamp threshold (0 = disabled).  Encode side snaps any voxel
    // v <= air_clamp to v = air_clamp before encoding (gives the codec a
    // flat constant region in the air/void instead of reconstruction noise).
    // Decode side zeroes any voxel v <= air_clamp post-decode, so end-users
    // see clean zero where the original had air noise.  Threshold is
    // recorded in the chunk header (v2) so decode is automatic.
    int air_clamp = 0;

    // Bit-shift (0..7, default 0 = off).  Opt-in ultra-compression at the
    // expense of signal quality: encode right-shifts input by shift_n,
    // decode left-shifts back (low shift_n bits become zero).  Recorded in
    // header (v3) so decode reconstructs the shifted range automatically.
    // Compounds with codec quantization error — only use when aggressive
    // compression matters more than fidelity.
    int shift_n = 0;
};

// Encode a 3D chunk as an H.265 bitstream.
// Input: raw uint8 voxel data in row-major (Z, Y, X) order.
// Returns: compressed bitstream bytes (VC3D header + H.265 NALUs).
[[nodiscard]] std::vector<std::byte> video_encode(
    std::span<const std::byte> raw, const VideoCodecParams& params);

// In-place variant: writes encoded bitstream into caller-owned `output`.
// Replaces `output`'s contents (clears then fills). Allows callers to
// reuse a thread-local buffer across chunks — eliminates the per-chunk
// std::vector allocation that was a steady source of swap pressure
// during streaming sessions.
void video_encode_into(
    std::span<const std::byte> raw,
    const VideoCodecParams& params,
    std::vector<std::byte>& output);

// Decode an H.265 bitstream back to a 3D chunk.
// Input: compressed bitstream bytes (VC3D header + H.265 NALUs).
// out_size: expected decompressed size (depth * height * width).
// Returns: raw uint8 voxel data in row-major (Z, Y, X) order.
[[nodiscard]] std::vector<std::byte> video_decode(
    std::span<const std::byte> compressed, std::size_t out_size,
    const VideoCodecParams& params);

// Zero-copy variant: write decoded voxels directly into `output`. Avoids
// the ~2 MiB interim allocation + zero-init + memcpy that the std::vector
// return form does on every call. Caller pre-sizes `output` to match the
// expected depth × height × width. Throws on header-size mismatch.
void video_decode_into(
    std::span<const std::byte> compressed,
    std::span<std::byte> output,
    const VideoCodecParams& params);

// Check if a compressed buffer has the VC3D video codec magic header.
[[nodiscard]] inline bool is_video_compressed(std::span<const std::byte> data) noexcept
{
    return data.size() >= 24 &&
        static_cast<char>(data[0]) == 'V' &&
        static_cast<char>(data[1]) == 'C' &&
        static_cast<char>(data[2]) == '3' &&
        static_cast<char>(data[3]) == 'D';
}

// Parse dimensions from a VC3D header. Returns {depth, height, width}.
[[nodiscard]] inline std::array<int, 3> video_header_dims(
    std::span<const std::byte> data) noexcept
{
    auto rd32 = [](const std::byte* p) -> int {
        return static_cast<int>(
            uint32_t(uint8_t(p[0])) | (uint32_t(uint8_t(p[1])) << 8) |
            (uint32_t(uint8_t(p[2])) << 16) | (uint32_t(uint8_t(p[3])) << 24));
    };
    if (data.size() < 24) return {0, 0, 0};
    return {rd32(data.data() + 8), rd32(data.data() + 12), rd32(data.data() + 16)};
}

}  // namespace utils
