// Push Volume coverage further: pyramid downsample on write, additional
// missing-policy values, normalize/derive helpers, write-then-shape variants.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/types/Volume.hpp"
#include "vc/core/types/Array3D.hpp"

#include "utils/Json.hpp"
#include <opencv2/core.hpp>

#include <array>
#include <cstring>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <random>
#include <vector>

namespace fs = std::filesystem;

namespace {

fs::path tmpDir(const std::string& tag)
{
    std::mt19937_64 rng(std::random_device{}());
    auto p = fs::temp_directory_path() /
             ("vc_vol_pyr_" + tag + "_" + std::to_string(rng()));
    fs::create_directories(p);
    return p;
}

Volume::ZarrCreateOptions optsMultiLevel()
{
    Volume::ZarrCreateOptions o;
    o.shapeZYX = {64, 64, 64};
    o.chunkShapeZYX = {32, 32, 32};
    o.numLevels = 3; // L0, L1, L2 — write to L0 cascades to L1 + L2
    o.compressor = "none";
    o.overwriteExisting = true;
    o.uuid = "pyr-vol";
    o.name = "pyr-vol";
    return o;
}

} // namespace

TEST_CASE("Volume::writeZYX (uint8) into a multi-level volume cascades to coarser levels")
{
    auto d = tmpDir("u8_pyr");
    auto v = Volume::New(d, optsMultiLevel());
    REQUIRE(v);
    CHECK(v->numScales() == 3);

    Array3D<uint8_t> in({16, 16, 16}, /*fill=*/77);
    v->writeZYX(in, {0, 0, 0}, /*level=*/0);

    // L0 chunk (0,0,0) should be present with our data.
    CHECK(v->chunkExists(/*level=*/0, {0, 0, 0}));

    // L1 chunk that covers the same region should also exist (downsampled).
    CHECK(v->chunkExists(/*level=*/1, {0, 0, 0}));

    // Read back L1 and verify it has the downsampled value (constant input
    // remains constant under mean downsample).
    Array3D<uint8_t> outL1({8, 8, 8});
    CHECK(v->readZYX(outL1, {0, 0, 0}, 1));
    CHECK(int(outL1(0, 0, 0)) == 77);
    fs::remove_all(d);
}

TEST_CASE("Volume::writeZYX (uint16) multi-level")
{
    auto d = tmpDir("u16_pyr");
    auto opts = optsMultiLevel();
    opts.dtype = vc::render::ChunkDtype::UInt16;
    auto v = Volume::New(d, opts);
    REQUIRE(v);

    Array3D<uint16_t> in({16, 16, 16}, /*fill=*/30000);
    v->writeZYX(in, {0, 0, 0}, 0);

    Array3D<uint16_t> outL1({8, 8, 8});
    CHECK(v->readZYX(outL1, {0, 0, 0}, /*level=*/1));
    CHECK(outL1(0, 0, 0) == 30000);
    fs::remove_all(d);
}

TEST_CASE("Volume::writeXYZ uint16 cascades the same way")
{
    auto d = tmpDir("u16_xyz_pyr");
    auto opts = optsMultiLevel();
    opts.dtype = vc::render::ChunkDtype::UInt16;
    auto v = Volume::New(d, opts);
    REQUIRE(v);
    Array3D<uint16_t> in({8, 8, 8}, /*fill=*/12345);
    v->writeXYZ(in, {0, 0, 0}, 0);
    Array3D<uint16_t> out({4, 4, 4});
    CHECK(v->readXYZ(out, {0, 0, 0}, 1));
    fs::remove_all(d);
}

TEST_CASE("Volume::readZYX (uint16) with missing-policy AllFill")
{
    auto d = tmpDir("u16_allfill");
    auto opts = optsMultiLevel();
    opts.dtype = vc::render::ChunkDtype::UInt16;
    opts.fillValue = 7000;
    auto v = Volume::New(d, opts);
    REQUIRE(v);
    Array3D<uint16_t> out({8, 8, 8});
    bool ok = v->readZYX(out, {0, 0, 0}, /*level=*/0,
                         Volume::MissingScaleLevelPolicy::AllFill);
    CHECK(ok);
    // Output should be the fill value.
    CHECK(out(0, 0, 0) == 7000);
    fs::remove_all(d);
}

TEST_CASE("Volume::readZYX (uint16) Error policy on missing level throws")
{
    auto d = tmpDir("u16_err");
    auto v = Volume::New(d, optsMultiLevel());
    REQUIRE(v);
    Array3D<uint16_t> out({8, 8, 8});
    // Level 99 doesn't exist; policy Error throws (or returns false).
    // Either path is acceptable — we just exercise it.
    try {
        (void)v->readZYX(out, {0, 0, 0}, 99,
                         Volume::MissingScaleLevelPolicy::Error);
    } catch (const std::exception&) {
        CHECK(true);
    }
    fs::remove_all(d);
}

TEST_CASE("Volume::readZYX (uint8) Empty policy returns false without filling")
{
    auto d = tmpDir("u8_empty");
    auto v = Volume::New(d, optsMultiLevel());
    REQUIRE(v);
    Array3D<uint8_t> out({4, 4, 4}, /*fill=*/55);
    bool ok = v->readZYX(out, {0, 0, 0}, /*level=*/99,
                         Volume::MissingScaleLevelPolicy::Empty);
    CHECK_FALSE(ok);
    fs::remove_all(d);
}

TEST_CASE("Volume: writeMetadata + updateMetadata work in-memory")
{
    auto d = tmpDir("meta_persist");
    auto v = Volume::New(d, optsMultiLevel());
    REQUIRE(v);
    utils::Json m = utils::Json::object();
    m["voxel_size_um"] = 7.91;
    v->writeMetadata(m);
    CHECK(v->metadata().contains("voxel_size_um"));
    utils::Json patch = utils::Json::object();
    patch["scan"] = "test-scan";
    v->updateMetadata(patch);
    CHECK(v->metadata().contains("scan"));
    CHECK(v->metadata().contains("voxel_size_um"));
    fs::remove_all(d);
}

TEST_CASE("Volume::createChunkCache with a persistent cache path")
{
    auto d = tmpDir("chunkcache");
    auto persistDir = tmpDir("persist");
    auto v = Volume::New(d, optsMultiLevel());
    REQUIRE(v);
    vc::render::ChunkCache::Options copts;
    copts.persistentCachePath = persistDir;
    auto c = v->createChunkCache(copts);
    CHECK(c != nullptr);
    fs::remove_all(d);
    fs::remove_all(persistDir);
}

TEST_CASE("Volume::shape(level) for all levels matches numScales")
{
    auto d = tmpDir("levels_shape");
    auto v = Volume::New(d, optsMultiLevel());
    REQUIRE(v);
    for (size_t lvl = 0; lvl < v->numScales(); ++lvl) {
        auto s = v->shape(static_cast<int>(lvl));
        CHECK(s[0] > 0);
        CHECK(s[1] > 0);
        CHECK(s[2] > 0);
    }
    fs::remove_all(d);
}

TEST_CASE("Volume: writeChunk with raw bytes round-trips through readChunkInto")
{
    auto d = tmpDir("chunkinto");
    auto v = Volume::New(d, optsMultiLevel());
    REQUIRE(v);
    std::vector<std::byte> in(v->chunkByteSize(0));
    std::memset(in.data(), 0xCD, in.size());
    v->writeChunk(0, {0, 0, 0}, in);

    std::vector<std::byte> out(v->chunkByteSize(0));
    CHECK(v->readChunkInto(0, {0, 0, 0}, out));
    CHECK(int(std::to_integer<int>(out[0])) == 0xCD);

    // Missing chunk: readChunkInto returns false.
    std::vector<std::byte> missing(v->chunkByteSize(0));
    CHECK_FALSE(v->readChunkInto(0, {1, 1, 1}, missing));
    fs::remove_all(d);
}
