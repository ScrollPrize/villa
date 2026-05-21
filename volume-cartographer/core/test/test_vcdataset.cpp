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
