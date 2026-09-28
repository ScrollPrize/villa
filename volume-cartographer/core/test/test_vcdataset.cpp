// Coverage for core/src/VcDataset.cpp through the direct (non-Volume) API:
// createZarrDataset + writeChunk + readChunk + readRegion + openZarrLevels
// + readZarrAttributes / writeZarrAttributes.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/types/VcDataset.hpp"

#include "utils/Json.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

fs::path tmpDir(const std::string& tag)
{
    std::mt19937_64 rng(std::random_device{}());
    auto p = fs::temp_directory_path() /
             ("vc_vcds_" + tag + "_" + std::to_string(rng()));
    fs::create_directories(p);
    return p;
}

} // namespace

TEST_CASE("createZarrDataset: builds metadata + accessors")
{
    auto d = tmpDir("create");
    auto ds = vc::createZarrDataset(d, "arr",
        /*shape=*/{32, 16, 8}, /*chunks=*/{16, 8, 4},
        vc::VcDtype::uint8, /*compressor=*/"none");
    REQUIRE(ds);
    CHECK(ds->getDtype() == vc::VcDtype::uint8);
    CHECK(ds->dtypeSize() == 1);
    auto sh = ds->shape();
    REQUIRE(sh.size() == 3);
    CHECK(sh[0] == 32);
    CHECK(sh[1] == 16);
    CHECK(sh[2] == 8);
    auto cs = ds->defaultChunkShape();
    REQUIRE(cs.size() == 3);
    CHECK(cs[0] == 16);
    CHECK(ds->defaultChunkSize() == 16ULL * 8ULL * 4ULL);
    // .zarray written under the named subdir.
    CHECK(fs::exists(d / "arr" / ".zarray"));
    fs::remove_all(d);
}

TEST_CASE("VcDataset (uint16) round-trip via writeChunk / readChunk")
{
    auto d = tmpDir("u16");
    auto ds = vc::createZarrDataset(d, "arr",
        {8, 8, 8}, {8, 8, 8},
        vc::VcDtype::uint16, "none");
    REQUIRE(ds);
    CHECK(ds->getDtype() == vc::VcDtype::uint16);
    CHECK(ds->dtypeSize() == 2);

    std::vector<uint16_t> payload(8 * 8 * 8);
    for (size_t i = 0; i < payload.size(); ++i) payload[i] = static_cast<uint16_t>(i);
    REQUIRE(ds->writeChunk(0, 0, 0, payload.data(), payload.size() * sizeof(uint16_t)));
    CHECK(ds->chunkExists(0, 0, 0));

    std::vector<uint16_t> out(payload.size(), 0);
    CHECK(ds->readChunk(0, 0, 0, out.data()));
    CHECK(out[0] == payload[0]);
    CHECK(out[42] == payload[42]);
    fs::remove_all(d);
}

TEST_CASE("VcDataset: readChunkOrFill returns the fill value for an absent chunk")
{
    auto d = tmpDir("fill");
    auto ds = vc::createZarrDataset(d, "arr",
        {8, 8, 8}, {8, 8, 8},
        vc::VcDtype::uint8, "none", ".", /*fillValue=*/77);
    REQUIRE(ds);
    std::vector<uint8_t> out(ds->defaultChunkSize(), 0);
    CHECK_FALSE(ds->readChunkOrFill(0, 0, 0, out.data()));
    for (auto v : out) CHECK(int(v) == 77);
    fs::remove_all(d);
}

TEST_CASE("VcDataset: removeChunk after writeChunk")
{
    auto d = tmpDir("rm");
    auto ds = vc::createZarrDataset(d, "arr",
        {4, 4, 4}, {4, 4, 4},
        vc::VcDtype::uint8, "none");
    REQUIRE(ds);
    std::vector<uint8_t> p(ds->defaultChunkSize(), 0x55);
    ds->writeChunk(0, 0, 0, p.data(), p.size());
    CHECK(ds->chunkExists(0, 0, 0));
    CHECK(ds->removeChunk(0, 0, 0));
    CHECK_FALSE(ds->chunkExists(0, 0, 0));
    CHECK_FALSE(ds->removeChunk(0, 0, 0));
    fs::remove_all(d);
}

TEST_CASE("VcDataset: writeChunkSkipEmpty skips all-fill chunks and drops stale ones")
{
    auto d = tmpDir("skipempty");
    auto ds = vc::createZarrDataset(d, "arr",
        {4, 4, 4}, {4, 4, 4},
        vc::VcDtype::uint8, "none");
    REQUIRE(ds);

    // All-fill (zero) buffer: nothing written.
    std::vector<uint8_t> empty(ds->defaultChunkSize(), 0);
    CHECK_FALSE(ds->writeChunkSkipEmpty(0, 0, 0, empty.data(), empty.size()));
    CHECK_FALSE(ds->chunkExists(0, 0, 0));

    // Non-empty buffer: written.
    std::vector<uint8_t> data(ds->defaultChunkSize(), 0);
    data[3] = 0x42;
    CHECK(ds->writeChunkSkipEmpty(0, 0, 0, data.data(), data.size()));
    CHECK(ds->chunkExists(0, 0, 0));

    // Re-writing an all-fill buffer removes the stale chunk.
    CHECK_FALSE(ds->writeChunkSkipEmpty(0, 0, 0, empty.data(), empty.size()));
    CHECK_FALSE(ds->chunkExists(0, 0, 0));
    fs::remove_all(d);
}

TEST_CASE("createZarrDataset: blosc compressor metadata is numcodecs-compliant")
{
    auto d = tmpDir("blosc_meta");
    auto ds = vc::createZarrDataset(d, "arr",
        {8, 8, 8}, {8, 8, 8},
        vc::VcDtype::uint8, "blosc", "/", /*fillValue=*/0, /*compressionLevel=*/7);
    REQUIRE(ds);
    std::ifstream f(d / "arr" / ".zarray");
    std::string meta((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
    CHECK(meta.find("\"cname\"") != std::string::npos);
    CHECK(meta.find("\"shuffle\"") != std::string::npos);
    CHECK(meta.find("\"blocksize\"") != std::string::npos);
    CHECK(meta.find("\"clevel\": 7") != std::string::npos);
    CHECK(meta.find("\"dimension_separator\": \"/\"") != std::string::npos);
    f.close();
    ds.reset();
    fs::remove_all(d);
}

TEST_CASE("VcDataset: readRegion / writeRegion over a 2x2x2 chunk block")
{
    auto d = tmpDir("region");
    auto ds = vc::createZarrDataset(d, "arr",
        /*shape=*/{8, 8, 8}, /*chunks=*/{4, 4, 4},
        vc::VcDtype::uint8, "none");
    REQUIRE(ds);
    // Region covers all 8 chunks (2x2x2 grid).
    std::vector<uint8_t> in(8 * 8 * 8);
    for (size_t i = 0; i < in.size(); ++i) in[i] = static_cast<uint8_t>(i & 0xFF);
    CHECK(ds->writeRegion({0, 0, 0}, {8, 8, 8}, in.data()));
    std::vector<uint8_t> out(in.size(), 0);
    CHECK(ds->readRegion({0, 0, 0}, {8, 8, 8}, out.data()));
    CHECK(out[0] == in[0]);
    CHECK(out[in.size() - 1] == in[in.size() - 1]);
    fs::remove_all(d);
}

TEST_CASE("openZarrLevels: enumerates numerically-named subdirs with .zarray")
{
    auto d = tmpDir("levels");
    // Create three levels: 0 (32^3), 1 (16^3), 2 (8^3)
    vc::createZarrDataset(d, "0", {32, 32, 32}, {16, 16, 16}, vc::VcDtype::uint8, "none");
    vc::createZarrDataset(d, "1", {16, 16, 16}, {8, 8, 8}, vc::VcDtype::uint8, "none");
    vc::createZarrDataset(d, "2", {8, 8, 8}, {8, 8, 8}, vc::VcDtype::uint8, "none");
    auto levels = vc::openZarrLevels(d);
    CHECK(levels.size() == 3);
    CHECK(levels[0]->shape()[0] == 32);
    CHECK(levels[1]->shape()[0] == 16);
    CHECK(levels[2]->shape()[0] == 8);
    fs::remove_all(d);
}

TEST_CASE("readZarrAttributes / writeZarrAttributes round-trip")
{
    auto d = tmpDir("attrs");
    auto attrs = utils::Json::object();
    attrs["foo"] = "bar";
    attrs["n"] = 42;
    vc::writeZarrAttributes(d, attrs);
    CHECK(fs::exists(d / ".zattrs"));
    auto loaded = vc::readZarrAttributes(d);
    CHECK(loaded["foo"].get_string() == "bar");
    CHECK(loaded["n"].get_int64() == 42);
    fs::remove_all(d);
}

TEST_CASE("readZarrAttributes: missing .zattrs yields null or empty object")
{
    auto d = tmpDir("noattrs");
    auto j = vc::readZarrAttributes(d);
    // The impl may return null Json or {} — either is acceptable.
    if (!j.is_null()) CHECK(j.is_object());
    fs::remove_all(d);
}

TEST_CASE("VcDataset move-construct and move-assign")
{
    auto d = tmpDir("move");
    auto a = vc::createZarrDataset(d, "arr",
        {4, 4, 4}, {4, 4, 4}, vc::VcDtype::uint8, "none");
    REQUIRE(a);
    vc::VcDataset b(std::move(*a));
    CHECK(b.getDtype() == vc::VcDtype::uint8);
    vc::VcDataset c(std::move(b));
    CHECK(c.getDtype() == vc::VcDtype::uint8);
    fs::remove_all(d);
}

TEST_CASE("VcDataset: open existing zarr by path")
{
    auto d = tmpDir("reopen");
    {
        auto ds = vc::createZarrDataset(d, "arr",
            {8, 8, 8}, {8, 8, 8}, vc::VcDtype::uint8, "none");
        REQUIRE(ds);
    }
    vc::VcDataset ds(d / "arr");
    CHECK(ds.shape()[0] == 8);
    CHECK(ds.getDtype() == vc::VcDtype::uint8);
    CHECK(ds.path() == d / "arr");
    fs::remove_all(d);
}

TEST_CASE("VcDataset: zstd compressor path")
{
    auto d = tmpDir("zstd");
    auto ds = vc::createZarrDataset(d, "arr",
        {8, 8, 8}, {8, 8, 8},
        vc::VcDtype::uint8, "zstd");
    REQUIRE(ds);
    std::vector<uint8_t> payload(ds->defaultChunkSize(), 0xAB);
    if (ds->writeChunk(0, 0, 0, payload.data(), payload.size())) {
        // Roundtrip if compression support is linked.
        std::vector<uint8_t> out(payload.size(), 0);
        CHECK(ds->readChunk(0, 0, 0, out.data()));
        CHECK(out[0] == 0xAB);
    }
    fs::remove_all(d);
}

// A sharded array stores encoded inner chunks inside one shard object, so the
// write path has to run the codec itself (write_inner_chunk_to_shard stores
// what it is given verbatim). Round-tripping compressed data is what proves it.
TEST_CASE("createZarrDataset: sharded v3 round-trips through the shard index")
{
    auto d = tmpDir("shard");
    // Width deliberately does NOT divide by the chunk width: a real render is
    // 37860 px wide against 2048 px chunks. The shard rounds out to whole chunks
    // and overhangs the array, which is what keeps the inner-chunk grid addressable.
    const std::vector<size_t> shape{1, 256, 1000};
    const std::vector<size_t> chunks{1, 128, 256};   // inner
    const std::vector<size_t> shard{1, 128, 1024};   // one band row per shard, 4 chunks

    auto ds = vc::createZarrDataset(d, "arr", shape, chunks, vc::VcDtype::uint8,
                                    /*compressor=*/"zstd", /*dimensionSeparator=*/"/",
                                    /*fillValue=*/0, /*compressionLevel=*/3, shard);
    REQUIRE(ds);
    // chunks still describe the finest granularity, not the shard.
    CHECK(ds->defaultChunkShape() == chunks);
    CHECK(ds->defaultChunkSize() == 128u * 256u);
    // v3 array + v3 group metadata, and no v2 leftovers.
    CHECK(fs::exists(d / "arr" / "zarr.json"));
    CHECK(!fs::exists(d / "arr" / ".zarray"));
    CHECK(fs::exists(d / "zarr.json"));

    std::vector<uint8_t> in(ds->defaultChunkSize());
    for (size_t i = 0; i < in.size(); ++i) in[i] = uint8_t((i * 7 + 13) % 251);

    // Two inner chunks of the same shard -- including the LAST column, which a
    // shard sized to the ragged array width would have pushed into a phantom
    // second shard column -- plus one in the next shard row.
    CHECK(!ds->chunkExists(0, 0, 0));
    REQUIRE(ds->writeChunk(0, 0, 0, in.data(), in.size()));
    REQUIRE(ds->writeChunk(0, 0, 3, in.data(), in.size()));
    REQUIRE(ds->writeChunk(0, 1, 2, in.data(), in.size()));
    CHECK(ds->chunkExists(0, 0, 0));
    CHECK(ds->chunkExists(0, 0, 3));
    CHECK(!ds->chunkExists(0, 0, 1));

    for (auto idx : {std::array<size_t, 3>{0, 0, 0},
                     std::array<size_t, 3>{0, 0, 3},
                     std::array<size_t, 3>{0, 1, 2}}) {
        std::vector<uint8_t> out(in.size(), 0);
        REQUIRE(ds->readChunk(idx[0], idx[1], idx[2], out.data()));
        CHECK(std::memcmp(out.data(), in.data(), in.size()) == 0);
    }

    // The shard must be one object holding both inner chunks, and compression
    // must actually have run, so it is far smaller than the raw payloads.
    auto shardFile = d / "arr" / "c" / "0" / "0" / "0";
    REQUIRE(fs::exists(shardFile));
    CHECK(fs::file_size(shardFile) < 2 * in.size());

    // Clearing an inner chunk is an index edit, not an unlink.
    CHECK(ds->removeChunk(0, 0, 3));
    CHECK(!ds->chunkExists(0, 0, 3));
    CHECK(fs::exists(shardFile));
    CHECK(ds->chunkExists(0, 0, 0));

    fs::remove_all(d);
}

// A collapsed composite is a YX image. It is stored rank-2, but every caller
// in the render path addresses chunks as ZYX, so VcDataset presents it as a
// single z-plane and drops the leading index when it talks to the array.
TEST_CASE("VcDataset: rank-2 array is addressed as a single ZYX plane")
{
    auto d = tmpDir("rank2");
    auto ds = vc::createZarrDataset(d, "arr", /*shape=*/{256, 512},
                                    /*chunks=*/{128, 256}, vc::VcDtype::uint8,
                                    /*compressor=*/"none");
    REQUIRE(ds);

    // On disk it is two-dimensional...
    auto meta = utils::Json::parse_file(d / "arr" / ".zarray");
    CHECK(meta["shape"].size() == 2u);
    CHECK(meta["chunks"].size() == 2u);

    // ...but the ZYX view carries a leading extent of 1.
    CHECK(ds->shape() == std::vector<size_t>{1, 256, 512});
    CHECK(ds->defaultChunkShape() == std::vector<size_t>{1, 128, 256});
    CHECK(ds->defaultChunkSize() == 128u * 256u);

    std::vector<uint8_t> in(ds->defaultChunkSize());
    for (size_t i = 0; i < in.size(); ++i) in[i] = uint8_t(i % 255);
    REQUIRE(ds->writeChunk(0, 1, 1, in.data(), in.size()));
    CHECK(ds->chunkExists(0, 1, 1));

    std::vector<uint8_t> out(in.size(), 0);
    REQUIRE(ds->readChunk(0, 1, 1, out.data()));
    CHECK(std::memcmp(out.data(), in.data(), in.size()) == 0);

    // The chunk key has two components, not three.
    CHECK(fs::exists(d / "arr" / "1" / "1"));

    fs::remove_all(d);
}

// The inner-chunk grid within a shard is derived as shard/chunk per dimension,
// so a shard that is not a whole number of chunks addresses a shard column that
// the grid does not have. Refuse to write such a store rather than produce one
// only vc3d can read back.
TEST_CASE("createZarrDataset: rejects a shard that is not whole chunks")
{
    auto d = tmpDir("shard_ragged");
    CHECK_THROWS_AS(
        vc::createZarrDataset(d, "arr", /*shape=*/{1, 256, 1000},
                              /*chunks=*/{1, 128, 256}, vc::VcDtype::uint8,
                              "zstd", "/", 0, 3, /*shardShape=*/{1, 128, 1000}),
        std::runtime_error);
    fs::remove_all(d);
}
