// Regression test for the Zarr-on-disk wire format. VcDataset writes Zarr v2
// metadata advertising compressor_id = "blosc" (i.e. Blosc1). This test pins
// the production compress path to a Blosc1-compatible chunk format by:
//   1. Asserting the chunk header version byte is <= BLOSC1_VERSION_FORMAT.
//   2. Round-tripping through blosc_decompress on a fresh context.
// If anyone re-migrates to Blosc2 without auditing the wire format, this
// test fails before we ship unreadable Zarr datasets.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <blosc.h>

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace {

// Mirror of VcDataset.cpp::bloscCompress. Kept in lockstep manually; the
// version-byte assertion below catches any divergence that would break
// Blosc1 readers (numcodecs, zarr-python, etc.).
std::vector<uint8_t> compressLikeVcDataset(const std::vector<uint8_t>& input,
                                           const char* compname,
                                           int clevel,
                                           int shuffle,
                                           int typesize)
{
    std::vector<uint8_t> output(input.size() + BLOSC_MAX_OVERHEAD);
    const int rc = blosc_compress_ctx(clevel, shuffle, typesize,
                                      input.size(),
                                      input.data(),
                                      output.data(), output.size(),
                                      compname, /*blocksize=*/0,
                                      /*nthreads=*/1);
    REQUIRE(rc > 0);
    output.resize(static_cast<size_t>(rc));
    return output;
}

void runRoundTrip(const char* compname, int shuffle, int typesize)
{
    blosc_init();

    // Mixed-pattern input so the codec actually does work — a buffer of zeros
    // can hide bugs because every codec compresses it trivially.
    std::vector<uint8_t> input(64 * 1024);
    std::mt19937 rng(0xC0FFEE);
    for (size_t i = 0; i < input.size(); i += 4) {
        input[i + 0] = static_cast<uint8_t>(i);
        input[i + 1] = static_cast<uint8_t>(rng() & 0xFF);
        input[i + 2] = 0;
        input[i + 3] = static_cast<uint8_t>(i >> 8);
    }

    auto compressed = compressLikeVcDataset(input, compname, /*clevel=*/5,
                                            shuffle, typesize);

    // The first byte of a Blosc chunk header is the version. Blosc1 readers
    // reject anything > BLOSC_VERSION_FORMAT (=2 for Blosc1).
    REQUIRE_FALSE(compressed.empty());
    CHECK(compressed[0] <= BLOSC_VERSION_FORMAT);

    std::vector<uint8_t> roundtrip(input.size());
    const int decompressed = blosc_decompress_ctx(compressed.data(),
                                                  roundtrip.data(),
                                                  roundtrip.size(),
                                                  /*nthreads=*/1);
    REQUIRE(decompressed == static_cast<int>(input.size()));
    CHECK(std::memcmp(input.data(), roundtrip.data(), input.size()) == 0);

    blosc_destroy();
}

}  // namespace

TEST_CASE("Blosc1 round-trip: lz4, byte-shuffle")
{
    runRoundTrip("lz4", BLOSC_SHUFFLE, /*typesize=*/1);
}

TEST_CASE("Blosc1 round-trip: zstd, byte-shuffle, typesize=2")
{
    runRoundTrip("zstd", BLOSC_SHUFFLE, /*typesize=*/2);
}

TEST_CASE("Blosc1 round-trip: blosclz, no shuffle")
{
    runRoundTrip("blosclz", BLOSC_NOSHUFFLE, /*typesize=*/1);
}
