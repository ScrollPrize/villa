#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "vc/core/util/CacheCompression.hpp"
#include <cstdlib>
#include <fstream>

TEST_CASE("Cache codec accepts preset frames and automatic fallback")
{
    for (const auto shape : {std::array{32, 32, 32}, std::array{32, 32, 64},
                             std::array{17, 19, 23}}) {
        for (const std::size_t elemSize : {1u, 2u}) {
            std::vector<std::byte> raw(shape[0] * shape[1] * shape[2] * elemSize);
            for (std::size_t i = 0; i < raw.size(); ++i)
                raw[i] = std::byte((i * 17 + i / 113) % 256);
            const auto encoded = vc::cacheCompress(raw, shape, elemSize);
            CHECK(vc::cacheQuantBinWidth(encoded) == 1);
            CHECK(vc::cacheDecompress(encoded, raw.size()) == raw);
            std::vector<std::byte> decoded(raw.size());
            REQUIRE(vc::cacheDecompressInto(encoded, decoded));
            CHECK(decoded == raw);
            if (elemSize == 2 || shape[0] == 17) {
                CHECK(encoded == vc_delta3d::compress(raw, shape, elemSize, 1,
                                                      vc_delta3d::Predictor::Delta));
            } else {
                CHECK_FALSE(vc::detail::hasMagic(encoded, vc_delta3d::kWireMagic));
            }
            const auto quantized = vc::cacheCompress(raw, shape, elemSize, 3);
            CHECK(vc::cacheQuantBinWidth(quantized) == 3);
            vc_delta3d::quantize(raw, elemSize, 3);
            CHECK(vc::cacheDecompress(quantized, raw.size()) == raw);
        }
    }
}

TEST_CASE("Cache codec still reads VCZ1 payloads")
{
    std::vector<std::byte> raw(32 * 32 * 32, std::byte{73});
    auto encoded = vc_delta3d::compress(raw, {32, 32, 32}, 1, 1,
                                       vc_delta3d::Predictor::Delta);
    std::copy(vc::detail::kVcz1Magic.begin(), vc::detail::kVcz1Magic.end(), encoded.begin());
    CHECK(vc::cacheQuantBinWidth(encoded) == 1);
    CHECK(vc::cacheDecompress(encoded, raw.size()) == raw);
}

TEST_CASE("Cache codec roundtrips an optional real 128 cubed uint8 chunk")
{
    const char* path = std::getenv("VC_DELTA3D_TEST_CHUNK");
    if (!path) return;
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    REQUIRE(stream.good());
    REQUIRE(stream.tellg() == 128 * 128 * 128);
    stream.seekg(0);
    std::vector<std::byte> raw(128 * 128 * 128);
    REQUIRE(bool(stream.read(reinterpret_cast<char*>(raw.data()), raw.size())));
    const auto encoded = vc::cacheCompress(raw, {128, 128, 128}, 1);
    const auto old = vc_delta3d::compress(raw, {128, 128, 128}, 1, 1,
                                         vc_delta3d::Predictor::Delta);
    REQUIRE(vc::cacheQuantBinWidth(encoded) == 1);
    CHECK(vc::cacheDecompress(encoded, raw.size()) == raw);
    CHECK(vc::cacheDecompress(old, raw.size()) == raw);
    std::cout << "Real chunk " << path << ": raw=" << raw.size()
              << " delta=" << old.size() << " fast=" << encoded.size() << '\n';
}
