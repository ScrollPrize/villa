#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/types/VcDataset.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct TmpDir {
    fs::path path;
    TmpDir()
    {
        std::random_device rd;
        std::mt19937_64 rng(rd());
        path = fs::temp_directory_path() /
               ("vc_vcdataset_test_" + std::to_string(rng()));
        fs::create_directories(path);
    }
    ~TmpDir()
    {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
};

}

TEST_CASE("VcDataset: writeChunk + readChunk roundtrip uint8")
{
    TmpDir tmp;
    auto ds = vc::createZarrDataset(
        tmp.path, "0", {16, 16, 16}, {16, 16, 16},
        vc::VcDtype::uint8, "blosc", ".", 0);

    std::vector<uint8_t> input(16 * 16 * 16);
    for (size_t i = 0; i < input.size(); ++i) input[i] = uint8_t(i % 256);

    REQUIRE(ds->writeChunk(0, 0, 0, input.data(), input.size()));
    REQUIRE(ds->chunkExists(0, 0, 0));

    std::vector<uint8_t> output(16 * 16 * 16, 0);
    REQUIRE(ds->readChunk(0, 0, 0, output.data()));
    CHECK(output == input);
}

TEST_CASE("VcDataset: writeChunk + readChunk roundtrip uint16")
{
    TmpDir tmp;
    auto ds = vc::createZarrDataset(
        tmp.path, "0", {8, 8, 8}, {8, 8, 8},
        vc::VcDtype::uint16, "blosc", ".", 0);

    std::vector<uint16_t> input(8 * 8 * 8);
    for (size_t i = 0; i < input.size(); ++i) input[i] = uint16_t(i * 257);

    REQUIRE(ds->writeChunk(0, 0, 0, input.data(), input.size() * sizeof(uint16_t)));

    std::vector<uint16_t> output(8 * 8 * 8, 0);
    REQUIRE(ds->readChunk(0, 0, 0, output.data()));
    CHECK(output == input);
}

TEST_CASE("VcDataset: chunkExists / removeChunk")
{
    TmpDir tmp;
    auto ds = vc::createZarrDataset(
        tmp.path, "0", {16, 16, 16}, {16, 16, 16},
        vc::VcDtype::uint8, "blosc");

    CHECK_FALSE(ds->chunkExists(0, 0, 0));

    std::vector<uint8_t> input(16 * 16 * 16, 7);
    REQUIRE(ds->writeChunk(0, 0, 0, input.data(), input.size()));
    CHECK(ds->chunkExists(0, 0, 0));

    CHECK(ds->removeChunk(0, 0, 0));
    CHECK_FALSE(ds->chunkExists(0, 0, 0));
}

TEST_CASE("VcDataset: readChunkOrFill returns fill bytes for missing chunk")
{
    TmpDir tmp;
    auto ds = vc::createZarrDataset(
        tmp.path, "0", {16, 16, 16}, {16, 16, 16},
        vc::VcDtype::uint8, "blosc", ".", 17);

    std::vector<uint8_t> output(16 * 16 * 16, 0xFF);
    const bool existed = ds->readChunkOrFill(0, 0, 0, output.data());
    CHECK_FALSE(existed);
    for (auto v : output) CHECK(v == 17);
}

TEST_CASE("VcDataset: metadata accessors")
{
    TmpDir tmp;
    auto ds = vc::createZarrDataset(
        tmp.path, "0", {32, 16, 8}, {16, 8, 4},
        vc::VcDtype::uint8, "blosc");

    CHECK(ds->shape() == std::vector<size_t>{32, 16, 8});
    CHECK(ds->defaultChunkShape() == std::vector<size_t>{16, 8, 4});
    CHECK(ds->defaultChunkSize() == 16 * 8 * 4);
    CHECK(ds->getDtype() == vc::VcDtype::uint8);
    CHECK(ds->dtypeSize() == 1);
}

TEST_CASE("VcDataset: readRegion across chunk boundaries")
{
    TmpDir tmp;
    auto ds = vc::createZarrDataset(
        tmp.path, "0", {16, 16, 16}, {8, 8, 8},
        vc::VcDtype::uint8, "blosc", ".", 0);

    std::vector<uint8_t> chunk(8 * 8 * 8);
    for (int cz = 0; cz < 2; ++cz) {
        for (int cy = 0; cy < 2; ++cy) {
            for (int cx = 0; cx < 2; ++cx) {
                for (int z = 0; z < 8; ++z)
                    for (int y = 0; y < 8; ++y)
                        for (int x = 0; x < 8; ++x)
                            chunk[(z * 8 + y) * 8 + x] = uint8_t(
                                ((cz * 8 + z) + (cy * 8 + y) + (cx * 8 + x)) % 256);
                ds->writeChunk(cz, cy, cx, chunk.data(), chunk.size());
            }
        }
    }

    std::vector<uint8_t> region(6 * 6 * 6, 0);
    REQUIRE(ds->readRegion({5, 5, 5}, {6, 6, 6}, region.data()));

    for (int z = 0; z < 6; ++z)
        for (int y = 0; y < 6; ++y)
            for (int x = 0; x < 6; ++x)
                CHECK(region[(z * 6 + y) * 6 + x] == uint8_t((5 + z + 5 + y + 5 + x) % 256));
}

TEST_CASE("VcDataset: writeRegion across chunk boundaries roundtrips through readChunk")
{
    TmpDir tmp;
    auto ds = vc::createZarrDataset(
        tmp.path, "0", {16, 16, 16}, {8, 8, 8},
        vc::VcDtype::uint8, "blosc", ".", 0);

    std::vector<uint8_t> region(10 * 10 * 10);
    for (size_t i = 0; i < region.size(); ++i) region[i] = uint8_t((i * 31) % 256);

    REQUIRE(ds->writeRegion({3, 3, 3}, {10, 10, 10}, region.data()));

    std::vector<uint8_t> readBack(10 * 10 * 10, 0);
    REQUIRE(ds->readRegion({3, 3, 3}, {10, 10, 10}, readBack.data()));
    CHECK(readBack == region);
}

TEST_CASE("VcDataset: openZarrLevels finds sibling levels")
{
    TmpDir tmp;
    vc::createZarrDataset(tmp.path, "0", {16, 16, 16}, {16, 16, 16}, vc::VcDtype::uint8, "blosc");
    vc::createZarrDataset(tmp.path, "1", {8, 8, 8}, {8, 8, 8}, vc::VcDtype::uint8, "blosc");

    auto levels = vc::openZarrLevels(tmp.path);
    REQUIRE(levels.size() == 2);
    CHECK(levels[0]->shape() == std::vector<size_t>{16, 16, 16});
    CHECK(levels[1]->shape() == std::vector<size_t>{8, 8, 8});
}

TEST_CASE("VcDataset: zarr attributes round-trip")
{
    TmpDir tmp;
    fs::create_directories(tmp.path);

    utils::Json attrs;
    attrs["name"] = "test_volume";
    attrs["voxel_size"] = 7.91;
    attrs["dimensions"] = utils::Json::array();
    attrs["dimensions"].push_back(64);
    attrs["dimensions"].push_back(64);
    attrs["dimensions"].push_back(64);

    vc::writeZarrAttributes(tmp.path, attrs);

    auto loaded = vc::readZarrAttributes(tmp.path);
    CHECK(loaded["name"].get_string() == "test_volume");
    CHECK(loaded["voxel_size"].get_double() == doctest::Approx(7.91));
    REQUIRE(loaded["dimensions"].is_array());
    CHECK(loaded["dimensions"].size() == 3);
    CHECK(loaded["dimensions"][0].get_int() == 64);
}

TEST_CASE("VcDataset: zstd compressor roundtrip")
{
    TmpDir tmp;
    auto ds = vc::createZarrDataset(
        tmp.path, "0", {8, 8, 8}, {8, 8, 8},
        vc::VcDtype::uint8, "zstd", ".", 0);

    std::vector<uint8_t> input(8 * 8 * 8);
    for (size_t i = 0; i < input.size(); ++i) input[i] = uint8_t(i & 0x7F);
    REQUIRE(ds->writeChunk(0, 0, 0, input.data(), input.size()));

    std::vector<uint8_t> output(8 * 8 * 8, 0);
    REQUIRE(ds->readChunk(0, 0, 0, output.data()));
    CHECK(output == input);
}

TEST_CASE("VcDataset: no-compression roundtrip")
{
    TmpDir tmp;
    auto ds = vc::createZarrDataset(
        tmp.path, "0", {8, 8, 8}, {8, 8, 8},
        vc::VcDtype::uint8, "none", ".", 0);

    std::vector<uint8_t> input(8 * 8 * 8);
    for (size_t i = 0; i < input.size(); ++i) input[i] = uint8_t(i % 13);
    REQUIRE(ds->writeChunk(0, 0, 0, input.data(), input.size()));

    std::vector<uint8_t> output(8 * 8 * 8, 0);
    REQUIRE(ds->readChunk(0, 0, 0, output.data()));
    CHECK(output == input);
}

TEST_CASE("VcDataset: gzip compressor roundtrip")
{
    TmpDir tmp;
    auto ds = vc::createZarrDataset(
        tmp.path, "0", {8, 8, 8}, {8, 8, 8},
        vc::VcDtype::uint8, "gzip", ".", 0);
    std::vector<uint8_t> input(8 * 8 * 8);
    for (size_t i = 0; i < input.size(); ++i) input[i] = uint8_t(i & 0x3F);
    REQUIRE(ds->writeChunk(0, 0, 0, input.data(), input.size()));

    std::vector<uint8_t> output(8 * 8 * 8, 0);
    REQUIRE(ds->readChunk(0, 0, 0, output.data()));
    CHECK(output == input);
}

TEST_CASE("VcDataset: lz4 compressor roundtrip (when supported)")
{
    TmpDir tmp;
    auto ds = vc::createZarrDataset(
        tmp.path, "0", {8, 8, 8}, {8, 8, 8},
        vc::VcDtype::uint8, "lz4", ".", 0);
    std::vector<uint8_t> input(8 * 8 * 8, 0);
    for (size_t i = 0; i < input.size(); ++i) input[i] = uint8_t((i * 17) & 0x7F);

    if (!ds->writeChunk(0, 0, 0, input.data(), input.size())) {
        MESSAGE("lz4 writeChunk returned false — codec not available");
        return;
    }
    std::vector<uint8_t> output(8 * 8 * 8, 0);
    if (!ds->readChunk(0, 0, 0, output.data())) {
        MESSAGE("lz4 readChunk returned false — codec asymmetric");
        return;
    }
    CHECK(output == input);
}

TEST_CASE("VcDataset: writeChunk on out-of-bounds chunk index does not crash")
{
    TmpDir tmp;
    auto ds = vc::createZarrDataset(
        tmp.path, "0", {16, 16, 16}, {16, 16, 16},
        vc::VcDtype::uint8, "blosc");
    std::vector<uint8_t> input(16 * 16 * 16, 7);
    (void)ds->writeChunk(99, 0, 0, input.data(), input.size());
}

TEST_CASE("VcDataset: readRegion zero-size succeeds without writing")
{
    TmpDir tmp;
    auto ds = vc::createZarrDataset(
        tmp.path, "0", {16, 16, 16}, {8, 8, 8},
        vc::VcDtype::uint8, "blosc");
    std::vector<uint8_t> region;
    REQUIRE(ds->readRegion({0, 0, 0}, {0, 0, 0}, region.data()));
}

TEST_CASE("VcDataset: writeRegion zero-size succeeds")
{
    TmpDir tmp;
    auto ds = vc::createZarrDataset(
        tmp.path, "0", {16, 16, 16}, {8, 8, 8},
        vc::VcDtype::uint8, "blosc");
    REQUIRE(ds->writeRegion({0, 0, 0}, {0, 0, 0}, nullptr));
}

TEST_CASE("VcDataset: writeRegion of single voxel + readRegion roundtrip")
{
    TmpDir tmp;
    auto ds = vc::createZarrDataset(
        tmp.path, "0", {16, 16, 16}, {8, 8, 8},
        vc::VcDtype::uint8, "blosc", ".", 0);
    uint8_t in = 200;
    REQUIRE(ds->writeRegion({3, 5, 7}, {1, 1, 1}, &in));
    uint8_t out = 0;
    REQUIRE(ds->readRegion({3, 5, 7}, {1, 1, 1}, &out));
    CHECK(out == 200);
}

TEST_CASE("VcDataset: openZarrLevels skips non-numeric subdirs")
{
    TmpDir tmp;
    vc::createZarrDataset(tmp.path, "0", {16, 16, 16}, {16, 16, 16}, vc::VcDtype::uint8, "blosc");
    fs::create_directories(tmp.path / "metadata");
    fs::create_directories(tmp.path / "_thumbnails");

    auto levels = vc::openZarrLevels(tmp.path);
    REQUIRE(levels.size() == 1);
    CHECK(levels[0]->shape() == std::vector<size_t>{16, 16, 16});
}

TEST_CASE("VcDataset: writeChunk + reopen + readChunk via fresh VcDataset")
{
    TmpDir tmp;
    {
        auto ds = vc::createZarrDataset(
            tmp.path, "0", {16, 16, 16}, {16, 16, 16},
            vc::VcDtype::uint8, "blosc", ".", 0);
        std::vector<uint8_t> in(16 * 16 * 16);
        for (size_t i = 0; i < in.size(); ++i) in[i] = uint8_t((i * 31) % 256);
        REQUIRE(ds->writeChunk(0, 0, 0, in.data(), in.size()));
    }
    vc::VcDataset reopened(tmp.path / "0");
    std::vector<uint8_t> out(16 * 16 * 16, 0);
    REQUIRE(reopened.readChunk(0, 0, 0, out.data()));
    for (size_t i = 0; i < out.size(); ++i)
        CHECK(out[i] == uint8_t((i * 31) % 256));
}

TEST_CASE("VcDataset: readChunk on missing chunk returns false")
{
    TmpDir tmp;
    auto ds = vc::createZarrDataset(
        tmp.path, "0", {16, 16, 16}, {16, 16, 16},
        vc::VcDtype::uint8, "blosc");
    std::vector<uint8_t> output(16 * 16 * 16, 0xAB);
    CHECK_FALSE(ds->readChunk(0, 0, 0, output.data()));
}

TEST_CASE("VcDataset: defaultChunkSize matches product of chunk dims")
{
    TmpDir tmp;
    auto ds = vc::createZarrDataset(
        tmp.path, "0", {32, 16, 8}, {8, 4, 2},
        vc::VcDtype::uint16, "blosc");
    CHECK(ds->defaultChunkSize() == 8 * 4 * 2);
    CHECK(ds->dtypeSize() == 2);
}
