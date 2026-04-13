#pragma once

// Video codec compression for 3D cubic chunks.
//
// A 3D chunk of shape (Z, Y, X) is encoded as a Z-frame grayscale video
// sequence using H.264 (OpenH264), with H.265 and AV1 support planned.
//
// Encoding: voxel values become the Y (luma) plane; U/V are set to 128.
// Decoding: Y plane is extracted back to voxel values; U/V are discarded.
//
// H.264 requires dimensions to be multiples of 2 (and ideally 16 for
// macroblock alignment). Chunks with odd dimensions are padded during
// encoding and cropped on decode.

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace utils {

enum class VideoCodecType {
    H264,
    H265,
    AV1,
    C3D,  // compress3d: 3D DCT + rANS, native 32³ blocks
};

struct VideoCodecParams {
    VideoCodecType type = VideoCodecType::H264;

    // Quantization parameter (0-51). Lower = better quality, larger output.
    // 0 = lossless (if codec supports it), 26 = default, 51 = worst quality.
    int qp = 26;

    // Chunk dimensions (Z frames of Y×X pixels). Must be set before encode.
    int depth = 0;   // Z
    int height = 0;  // Y
    int width = 0;   // X
};

// Encode a 3D chunk as a video bitstream.
// Input: raw uint8 voxel data in row-major (Z, Y, X) order.
// Returns: compressed bitstream bytes.
[[nodiscard]] std::vector<std::byte> video_encode(
    std::span<const std::byte> raw, const VideoCodecParams& params);

// Decode a video bitstream back to a 3D chunk.
// Input: compressed bitstream bytes.
// out_size: expected decompressed size (depth * height * width).
// Returns: raw uint8 voxel data in row-major (Z, Y, X) order.
[[nodiscard]] std::vector<std::byte> video_decode(
    std::span<const std::byte> compressed, std::size_t out_size,
    const VideoCodecParams& params);

// Check if a compressed buffer has the VC3D video codec magic header.
[[nodiscard]] inline bool is_video_compressed(std::span<const std::byte> data) noexcept
{
    if (data.size() >= 20 &&
        static_cast<char>(data[0]) == 'V' &&
        static_cast<char>(data[1]) == 'C' &&
        static_cast<char>(data[2]) == '3' &&
        static_cast<char>(data[3]) == 'D')
        return true;
    // Also check for C3D tiled container magic "C3T\x01"
    if (data.size() >= 16 &&
        static_cast<char>(data[0]) == 'C' &&
        static_cast<char>(data[1]) == '3' &&
        static_cast<char>(data[2]) == 'T' &&
        static_cast<uint8_t>(data[3]) == 0x01)
        return true;
    return false;
}

// Check if data is specifically a C3D tiled container.
[[nodiscard]] inline bool is_c3d_compressed(std::span<const std::byte> data) noexcept
{
    return data.size() >= 16 &&
           static_cast<char>(data[0]) == 'C' &&
           static_cast<char>(data[1]) == '3' &&
           static_cast<char>(data[2]) == 'T' &&
           static_cast<uint8_t>(data[3]) == 0x01;
}

// Parse dimensions from a VC3D or C3T header. Returns {depth, height, width}.
[[nodiscard]] inline std::array<int, 3> video_header_dims(
    std::span<const std::byte> data) noexcept
{
    auto rd16 = [](const std::byte* p) -> int {
        return static_cast<int>(
            uint16_t(uint8_t(p[0])) | (uint16_t(uint8_t(p[1])) << 8));
    };
    auto rd32 = [](const std::byte* p) -> int {
        return static_cast<int>(
            uint32_t(uint8_t(p[0])) | (uint32_t(uint8_t(p[1])) << 8) |
            (uint32_t(uint8_t(p[2])) << 16) | (uint32_t(uint8_t(p[3])) << 24));
    };
    // C3T container: dims are uint16 at offsets 6, 8, 10
    if (is_c3d_compressed(data)) {
        return {rd16(data.data() + 6), rd16(data.data() + 8), rd16(data.data() + 10)};
    }
    // VC3D header: dims are uint32 at offsets 8, 12, 16
    if (data.size() < 20) return {0, 0, 0};
    return {rd32(data.data() + 8), rd32(data.data() + 12), rd32(data.data() + 16)};
}

}  // namespace utils
