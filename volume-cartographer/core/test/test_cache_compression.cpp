// CacheCompression coverage: VCZ1 delta-zyx+zstd roundtrips and
// corrupt/mismatched payload handling.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/CacheCompression.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

#include <zstd.h>

namespace {

std::vector<std::byte> smoothVolume(std::array<int, 3> shape, std::size_t elemSize)
{
    // Correlated ramp + noise: representative of CT data and sensitive to
    // axis-order bugs in the filter (values differ along every axis).
    std::vector<std::byte> bytes(
        static_cast<std::size_t>(shape[0]) * shape[1] * shape[2] * elemSize);
    std::mt19937 rng(1234);
    std::uniform_int_distribution<int> noise(-3, 3);
    std::size_t i = 0;
    for (int z = 0; z < shape[0]; ++z)
        for (int y = 0; y < shape[1]; ++y)
            for (int x = 0; x < shape[2]; ++x) {
                const int v = 7 * z + 3 * y + 5 * x + noise(rng);
                if (elemSize == 1) {
                    bytes[i++] = static_cast<std::byte>(v & 0xFF);
                } else {
                    const auto value = static_cast<std::uint16_t>(v * 111);
                    std::memcpy(bytes.data() + i, &value, 2);
                    i += 2;
                }
            }
    return bytes;
}

std::span<const std::byte> asSpan(const std::vector<std::byte>& v)
{
    return {v.data(), v.size()};
}

} // namespace

TEST_CASE("VCZ1 roundtrip uint8")
{
    const std::array<int, 3> shape{8, 6, 10};
    const auto input = smoothVolume(shape, 1);
    const auto compressed = vc::cacheCompress(asSpan(input), shape, 1);

    REQUIRE(compressed.size() > 20);
    CHECK(static_cast<char>(compressed[0]) == 'V');
    CHECK(static_cast<char>(compressed[1]) == 'C');
    CHECK(static_cast<char>(compressed[2]) == 'Z');
    CHECK(static_cast<char>(compressed[3]) == '1');

    const auto decoded = vc::cacheDecompress(asSpan(compressed), input.size());
    REQUIRE(decoded.has_value());
    CHECK(*decoded == input);
}

TEST_CASE("VCZ1 roundtrip uint16")
{
    const std::array<int, 3> shape{5, 7, 9};
    const auto input = smoothVolume(shape, 2);
    const auto compressed = vc::cacheCompress(asSpan(input), shape, 2);
    const auto decoded = vc::cacheDecompress(asSpan(compressed), input.size());
    REQUIRE(decoded.has_value());
    CHECK(*decoded == input);
}

TEST_CASE("VCZ1 roundtrip single-voxel and flat shapes")
{
    for (const auto shape : {std::array<int, 3>{1, 1, 1},
                             std::array<int, 3>{1, 1, 16},
                             std::array<int, 3>{16, 1, 1},
                             std::array<int, 3>{1, 16, 1}}) {
        const auto input = smoothVolume(shape, 1);
        const auto compressed = vc::cacheCompress(asSpan(input), shape, 1);
        const auto decoded = vc::cacheDecompress(asSpan(compressed), input.size());
        REQUIRE(decoded.has_value());
        CHECK(*decoded == input);
    }
}

TEST_CASE("Filter reduces compressed size on correlated data")
{
    const std::array<int, 3> shape{32, 32, 32};
    const auto input = smoothVolume(shape, 1);
    const auto filtered = vc::cacheCompress(asSpan(input), shape, 1);

    std::vector<std::byte> unfiltered(ZSTD_compressBound(input.size()));
    const auto rc = ZSTD_compress(
        unfiltered.data(), unfiltered.size(), input.data(), input.size(), 3);
    REQUIRE_FALSE(ZSTD_isError(rc));
    CHECK(filtered.size() < rc);
}

TEST_CASE("Mismatched shape or element size throws")
{
    const std::array<int, 3> shape{4, 4, 4};
    const auto input = smoothVolume(shape, 1);
    CHECK_THROWS_AS(vc::cacheCompress(asSpan(input), {4, 4, 5}, 1),
                    std::invalid_argument);
    CHECK_THROWS_AS(vc::cacheCompress(asSpan(input), shape, 4),
                    std::invalid_argument);
    CHECK_THROWS_AS(vc::cacheCompress(asSpan(input), {0, 0, 0}, 1),
                    std::invalid_argument);
}

TEST_CASE("Plain zstd frames without a VCZ1 header are rejected")
{
    const std::array<int, 3> shape{4, 4, 4};
    const auto input = smoothVolume(shape, 1);
    std::vector<std::byte> frame(ZSTD_compressBound(input.size()));
    const auto rc = ZSTD_compress(
        frame.data(), frame.size(), input.data(), input.size(), 3);
    REQUIRE_FALSE(ZSTD_isError(rc));
    frame.resize(rc);
    CHECK_FALSE(vc::cacheDecompress(asSpan(frame), input.size()).has_value());
}

TEST_CASE("Near-lossless quantization bounds the per-voxel error")
{
    const std::array<int, 3> shape{8, 6, 10};
    for (const std::size_t elemSize : {std::size_t{1}, std::size_t{2}}) {
        const auto input = smoothVolume(shape, elemSize);
        for (const int width : {vc::kCacheQuantMaxErr1, vc::kCacheQuantMaxErr2}) {
            const auto compressed =
                vc::cacheCompress(asSpan(input), shape, elemSize,
                                  vc::kCacheCompressionLevel, width);
            CHECK(vc::cacheQuantBinWidth(asSpan(compressed)) == width);

            const auto decoded = vc::cacheDecompress(asSpan(compressed), input.size());
            REQUIRE(decoded.has_value());
            const int maxErr = width / 2;
            for (std::size_t i = 0; i < input.size(); i += elemSize) {
                int orig = 0, got = 0;
                std::memcpy(&orig, input.data() + i, elemSize);
                std::memcpy(&got, decoded->data() + i, elemSize);
                CHECK(std::abs(orig - got) <= maxErr);
                if (orig == 0)
                    CHECK(got == 0); // masked background must stay exact
            }

            // Idempotent: re-encoding the quantized voxels at the same
            // width must reproduce them exactly.
            const auto again =
                vc::cacheCompress(asSpan(*decoded), shape, elemSize,
                                  vc::kCacheCompressionLevel, width);
            const auto decodedAgain = vc::cacheDecompress(asSpan(again), input.size());
            REQUIRE(decodedAgain.has_value());
            CHECK(*decodedAgain == *decoded);
        }
    }
}

TEST_CASE("Quantization width is reported for all payload generations")
{
    const std::array<int, 3> shape{4, 4, 4};
    const auto input = smoothVolume(shape, 1);

    auto lossless = vc::cacheCompress(asSpan(input), shape, 1);
    CHECK(vc::cacheQuantBinWidth(asSpan(lossless)) == vc::kCacheQuantLossless);

    // Payloads written before quantization existed carry 0 in byte 6 and
    // must read back as lossless.
    lossless[6] = std::byte{0};
    CHECK(vc::cacheQuantBinWidth(asSpan(lossless)) == vc::kCacheQuantLossless);
    CHECK(vc::cacheDecompress(asSpan(lossless), input.size()).has_value());

    const std::vector<std::byte> garbage(64, std::byte{0xAB});
    CHECK_FALSE(vc::cacheQuantBinWidth(asSpan(garbage)).has_value());

    CHECK_THROWS_AS(
        vc::cacheCompress(asSpan(input), shape, 1, vc::kCacheCompressionLevel, 0),
        std::invalid_argument);
    CHECK_THROWS_AS(
        vc::cacheCompress(asSpan(input), shape, 1, vc::kCacheCompressionLevel, 256),
        std::invalid_argument);
}

TEST_CASE("Corrupt and mismatched payloads return nullopt")
{
    const std::array<int, 3> shape{4, 4, 4};
    const auto input = smoothVolume(shape, 1);
    const auto compressed = vc::cacheCompress(asSpan(input), shape, 1);

    // Wrong expected size (header dims disagree).
    CHECK_FALSE(vc::cacheDecompress(asSpan(compressed), input.size() + 1).has_value());

    // Truncated zstd frame.
    std::vector<std::byte> truncated(compressed.begin(),
                                     compressed.begin() + compressed.size() / 2);
    CHECK_FALSE(vc::cacheDecompress(asSpan(truncated), input.size()).has_value());

    // Garbage bytes (neither VCZ1 nor zstd).
    std::vector<std::byte> garbage(64, std::byte{0xAB});
    CHECK_FALSE(vc::cacheDecompress(asSpan(garbage), input.size()).has_value());
}
