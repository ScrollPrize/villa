// Regression test for the Zarr-on-disk wire format. VcDataset writes Zarr v2
// metadata advertising compressor_id = "blosc" (i.e. Blosc1). Chunks produced
// by VcDataset must therefore remain Blosc1-readable.
//
// We compress via the *exact* production code path (vc::testing::
// bloscCompressForTest in VcDataset.cpp, which is a thin wrapper over the
// real bloscCompress) and assert:
//   1. The chunk header version byte is BLOSC_VERSION_FORMAT (Blosc1).
//   2. blosc_decompress_ctx round-trips the data byte-for-byte.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <blosc.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace vc::testing {
std::vector<std::byte> bloscCompressForTest(
    const void* src, std::size_t src_bytes,
    const char* compname, int clevel, int shuffle, int typesize);
}

namespace {

void runRoundTrip(const char* compname, int shuffle, int typesize)
{
    blosc_init();

    // Mixed-pattern input — a buffer of zeros compresses trivially under
    // every codec and would hide real bugs.
    std::vector<uint8_t> input(64 * 1024);
    std::mt19937 rng(0xC0FFEE);
    for (size_t i = 0; i < input.size(); i += 4) {
        input[i + 0] = static_cast<uint8_t>(i);
        input[i + 1] = static_cast<uint8_t>(rng() & 0xFF);
        input[i + 2] = 0;
        input[i + 3] = static_cast<uint8_t>(i >> 8);
    }

    auto compressed = vc::testing::bloscCompressForTest(
        input.data(), input.size(), compname, /*clevel=*/5, shuffle, typesize);

    // The first byte of a Blosc chunk header is the version. Blosc1 readers
    // reject anything > BLOSC_VERSION_FORMAT (=2 for Blosc1).
    REQUIRE_FALSE(compressed.empty());
    CHECK(static_cast<uint8_t>(compressed[0]) <= BLOSC_VERSION_FORMAT);

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

TEST_CASE("Blosc1 round-trip via production compress path: lz4, byte-shuffle")
{
    runRoundTrip("lz4", BLOSC_SHUFFLE, /*typesize=*/1);
}

TEST_CASE("Blosc1 round-trip via production compress path: zstd, typesize=2")
{
    runRoundTrip("zstd", BLOSC_SHUFFLE, /*typesize=*/2);
}

TEST_CASE("Blosc1 round-trip via production compress path: blosclz, no shuffle")
{
    runRoundTrip("blosclz", BLOSC_NOSHUFFLE, /*typesize=*/1);
}
