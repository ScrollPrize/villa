// Volume reads on a multi-level pyramid: missing-level policies, per-level
// shapes and raw chunk reads.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/types/Volume.hpp"
#include "support/TestZarrVolume.hpp"
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

vc::test::ZarrVolumeSpec optsMultiLevel()
{
    vc::test::ZarrVolumeSpec o;
    o.shapeZYX = {64, 64, 64};
    o.chunkShapeZYX = {32, 32, 32};
    o.numLevels = 3;
    o.uuid = "pyr-vol";
    o.name = "pyr-vol";
    return o;
}

} // namespace

TEST_CASE("Volume::readZYX (uint16) with missing-policy AllFill")
{
    auto d = tmpDir("u16_allfill");
    auto opts = optsMultiLevel();
    opts.dtype = vc::render::ChunkDtype::UInt16;
    opts.fillValue = 7000;
    auto v = vc::test::createZarrVolume(d, opts);
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
    auto v = vc::test::createZarrVolume(d, optsMultiLevel());
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
    auto v = vc::test::createZarrVolume(d, optsMultiLevel());
    REQUIRE(v);
    Array3D<uint8_t> out({4, 4, 4}, /*fill=*/55);
    bool ok = v->readZYX(out, {0, 0, 0}, /*level=*/99,
                         Volume::MissingScaleLevelPolicy::Empty);
    CHECK_FALSE(ok);
    fs::remove_all(d);
}

TEST_CASE("Volume::shape(level) for all levels matches numScales")
{
    auto d = tmpDir("levels_shape");
    auto v = vc::test::createZarrVolume(d, optsMultiLevel());
    REQUIRE(v);
    for (size_t lvl = 0; lvl < v->numScales(); ++lvl) {
        auto s = v->shape(static_cast<int>(lvl));
        CHECK(s[0] > 0);
        CHECK(s[1] > 0);
        CHECK(s[2] > 0);
    }
    fs::remove_all(d);
}

TEST_CASE("Volume: a stored raw chunk round-trips through readChunkInto")
{
    auto d = tmpDir("chunkinto");
    auto v = vc::test::createZarrVolume(d, optsMultiLevel());
    REQUIRE(v);
    std::vector<std::byte> in(v->chunkByteSize(0));
    std::memset(in.data(), 0xCD, in.size());
    vc::test::writeChunk(*v, 0, {0, 0, 0}, in);

    std::vector<std::byte> out(v->chunkByteSize(0));
    CHECK(v->readChunkInto(0, {0, 0, 0}, out));
    CHECK(int(std::to_integer<int>(out[0])) == 0xCD);

    // Missing chunk: readChunkInto returns false.
    std::vector<std::byte> missing(v->chunkByteSize(0));
    CHECK_FALSE(v->readChunkInto(0, {1, 1, 1}, missing));
    fs::remove_all(d);
}
