#pragma once

#include <vc_delta3d/codec.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace vc {

// Compatibility facade for Volume Cartographer's cache API. The codec
// implementation and wire-format contract live in ScrollPrize/vc-delta3d.
inline constexpr int kCacheQuantLossless =
    vc_delta3d::kQuantizationLossless;
inline constexpr int kCacheQuantMaxErr1 =
    vc_delta3d::kQuantizationMaxError1;
inline constexpr int kCacheQuantMaxErr2 =
    vc_delta3d::kQuantizationMaxError2;

// These are Volume Cartographer integration details rather than properties of
// the encoded stream itself.
inline constexpr const char* kCompressedCacheExtension = ".zst";
inline constexpr const char* kDelta3dCodecName = vc_delta3d::kCodecId;
inline constexpr const char* kVcz1CodecName = "vcz1";

namespace detail {

inline constexpr vc_delta3d::WireMagic kVcz1Magic{
    std::byte{'V'}, std::byte{'C'}, std::byte{'Z'}, std::byte{'1'}};

inline bool hasMagic(std::span<const std::byte> input,
                     const std::array<std::byte, 4>& magic)
{
    return input.size() >= magic.size() &&
           std::equal(magic.begin(), magic.end(), input.begin());
}

inline std::optional<std::array<std::byte, 21>> translatedVcz1Header(
    std::span<const std::byte> input)
{
    if (input.size() < 21)
        return std::nullopt;
    std::array<std::byte, 21> translated;
    std::copy_n(input.begin(), translated.size(), translated.begin());
    std::copy(vc_delta3d::kWireMagic.begin(), vc_delta3d::kWireMagic.end(),
              translated.begin());
    return translated;
}

} // namespace detail

inline constexpr bool isDelta3dCodecName(std::string_view name)
{
    return name == kDelta3dCodecName || name == kVcz1CodecName;
}

inline std::vector<std::byte> cacheCompress(
    std::span<const std::byte> input,
    std::array<int, 3> shapeZYX,
    std::size_t elemSize,
    int quantBinWidth = kCacheQuantLossless)
{
    return vc_delta3d::compress(input, shapeZYX, elemSize, quantBinWidth);
}

inline void cacheQuantize(std::span<std::byte> data,
                          std::size_t elemSize,
                          int quantBinWidth)
{
    vc_delta3d::quantize(data, elemSize, quantBinWidth);
}

inline std::optional<int> cacheQuantBinWidth(
    std::span<const std::byte> input)
{
    if (detail::hasMagic(input, detail::kVcz1Magic)) {
        const auto header = detail::translatedVcz1Header(input);
        return header ? vc_delta3d::quantization(*header) : std::nullopt;
    }
    return vc_delta3d::quantization(input);
}

inline std::optional<int> cacheDeltaMask(std::span<const std::byte> input)
{
    if (detail::hasMagic(input, detail::kVcz1Magic)) {
        const auto header = detail::translatedVcz1Header(input);
        return header ? vc_delta3d::deltaMask(*header) : std::nullopt;
    }
    return vc_delta3d::deltaMask(input);
}

inline std::optional<std::vector<std::byte>> cacheDecompress(
    std::span<const std::byte> input,
    std::size_t expectedSize)
{
    if (detail::hasMagic(input, detail::kVcz1Magic))
        return vc_delta3d::decompressWithMagic(
            input, expectedSize, detail::kVcz1Magic);
    return vc_delta3d::decompress(input, expectedSize);
}

inline bool cacheDecompressInto(std::span<const std::byte> input,
                                std::span<std::byte> output)
{
    if (detail::hasMagic(input, detail::kVcz1Magic))
        return vc_delta3d::decompressIntoWithMagic(
            input, output, detail::kVcz1Magic);
    return vc_delta3d::decompressInto(input, output);
}

} // namespace vc
