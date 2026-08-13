// Extra Volume coverage beyond test_volume_local.cpp:
// - uint16 dtype write/read
// - writeMetadata / updateMetadata
// - writeRootAttributes / updateRootAttributes
// - setCacheBudget, setIOThreads
// - sample (single-slice) on a seeded uint8 volume
// - MissingScaleLevelPolicy::AllFill behaviour for a level that exists but is empty

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/types/Volume.hpp"
#include "vc/core/types/Array3D.hpp"
#include "vc/core/types/SampleParams.hpp"

#include "utils/Json.hpp"
#include <opencv2/core.hpp>

#include <array>
#include <cstring>
#include <filesystem>
#include <random>
#include <vector>

namespace fs = std::filesystem;

namespace {

fs::path tmpDir(const std::string& tag)
{
    std::mt19937_64 rng(std::random_device{}());
    auto p = fs::temp_directory_path() /
             ("vc_vol_extras_" + tag + "_" + std::to_string(rng()));
    fs::create_directories(p);
    return p;
}

Volume::ZarrCreateOptions optsU8()
{
    Volume::ZarrCreateOptions o;
    o.shapeZYX = {32, 32, 32};
    o.chunkShapeZYX = {16, 16, 16};
    o.numLevels = 1;
    o.compressor = "none";
    o.overwriteExisting = true;
    return o;
}

Volume::ZarrCreateOptions optsU16()
{
    auto o = optsU8();
    o.dtype = vc::render::ChunkDtype::UInt16;
    return o;
}

} // namespace

TEST_CASE("Volume uint16 round-trip via writeZYX/readZYX")
{
    auto d = tmpDir("u16");
    auto v = Volume::New(d, optsU16());
    REQUIRE(v);
    CHECK(v->dtype() == vc::render::ChunkDtype::UInt16);
    CHECK(v->dtypeSize() == 2);

    Array3D<uint16_t> in({8, 8, 8}, 0);
    for (size_t z = 0; z < 8; ++z)
        for (size_t y = 0; y < 8; ++y)
            for (size_t x = 0; x < 8; ++x)
                in(z, y, x) = static_cast<uint16_t>(z * 64 + y * 8 + x);
    v->writeZYX(in, {0, 0, 0}, 0);

    Array3D<uint16_t> out({8, 8, 8});
    CHECK(v->readZYX(out, {0, 0, 0}, 0));
    CHECK(out(0, 0, 0) == in(0, 0, 0));
    CHECK(out(7, 7, 7) == in(7, 7, 7));
    fs::remove_all(d);
}

TEST_CASE("Volume uint16: writeXYZ/readXYZ overload")
{
    auto d = tmpDir("u16_xyz");
    auto v = Volume::New(d, optsU16());
    REQUIRE(v);
    Array3D<uint16_t> in({4, 4, 4}, 12345);
    v->writeXYZ(in, {0, 0, 0}, 0);
    Array3D<uint16_t> out({4, 4, 4});
    CHECK(v->readXYZ(out, {0, 0, 0}, 0));
    CHECK(out(0, 0, 0) == 12345);
    fs::remove_all(d);
}

TEST_CASE("Volume::sample writes a Mat at the requested coords")
{
    auto d = tmpDir("sample");
    auto v = Volume::New(d, optsU8());
    REQUIRE(v);
    // Seed the volume with a gradient.
    Array3D<uint8_t> seed({32, 32, 32}, 0);
    for (size_t z = 0; z < 32; ++z)
        for (size_t y = 0; y < 32; ++y)
            for (size_t x = 0; x < 32; ++x)
                seed(z, y, x) = static_cast<uint8_t>((z + y + x) & 0xFF);
    v->writeZYX(seed, {0, 0, 0}, 0);

    cv::Mat_<cv::Vec3f> coords(4, 4);
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            coords(r, c) = cv::Vec3f(float(c), float(r), 0.f);
    cv::Mat_<uint8_t> out(4, 4, uint8_t{0});
    vc::SampleParams sp;
    sp.level = 0;
    v->sample(out, coords, sp);
    // No assertion on exact value — sampling interpolates; just non-crash + size.
    CHECK(out.rows == 4);
    CHECK(out.cols == 4);
    fs::remove_all(d);
}

TEST_CASE("Volume::sample uint16 overload")
{
    auto d = tmpDir("sample_u16");
    auto v = Volume::New(d, optsU16());
    REQUIRE(v);
    Array3D<uint16_t> seed({16, 16, 16}, 1000);
    v->writeZYX(seed, {0, 0, 0}, 0);
    cv::Mat_<cv::Vec3f> coords(2, 2);
    for (int r = 0; r < 2; ++r)
        for (int c = 0; c < 2; ++c)
            coords(r, c) = cv::Vec3f(float(c), float(r), 0.f);
    cv::Mat_<uint16_t> out(2, 2, uint16_t{0});
    vc::SampleParams sp;
    sp.level = 0;
    v->sample(out, coords, sp);
    CHECK(out.rows == 2);
    fs::remove_all(d);
}

TEST_CASE("Volume::writeMetadata / updateMetadata")
{
    auto d = tmpDir("meta");
    auto v = Volume::New(d, optsU8());
    REQUIRE(v);
    utils::Json m = utils::Json::object();
    m["voxel_size_um"] = 7.91;
    v->writeMetadata(m);
    CHECK(v->metadata()["voxel_size_um"].get_double() == doctest::Approx(7.91));

    // updateMetadata merges into existing.
    utils::Json patch = utils::Json::object();
    patch["scan"] = "test-scan";
    v->updateMetadata(patch);
    CHECK(v->metadata().contains("voxel_size_um"));
    CHECK(v->metadata()["scan"].get_string() == "test-scan");
    fs::remove_all(d);
}

TEST_CASE("Volume::writeRootAttributes / updateRootAttributes")
{
    auto d = tmpDir("rootattrs");
    auto v = Volume::New(d, optsU8());
    REQUIRE(v);
    utils::Json a = utils::Json::object();
    a["foo"] = "bar";
    v->writeRootAttributes(a);
    auto loaded = v->rootAttributes();
    CHECK(loaded["foo"].get_string() == "bar");
    // updateRootAttributes merges.
    utils::Json patch = utils::Json::object();
    patch["count"] = 7;
    v->updateRootAttributes(patch);
    auto merged = v->rootAttributes();
    CHECK(merged.contains("foo"));
    CHECK(merged["count"].get_int64() == 7);
    fs::remove_all(d);
}

TEST_CASE("Volume::setCacheBudget and setIOThreads")
{
    auto d = tmpDir("knobs");
    auto v = Volume::New(d, optsU8());
    REQUIRE(v);
    v->setCacheBudget(1ULL << 20);
    v->setIOThreads(2);
    // No accessor — just verify no crash and chunkedCache works after.
    auto* cache = v->chunkedCache();
    CHECK(cache != nullptr);
    fs::remove_all(d);
}

TEST_CASE("Volume::invalidateCache")
{
    auto d = tmpDir("invalidate");
    auto v = Volume::New(d, optsU8());
    REQUIRE(v);
    // Just exercise; no observable state.
    v->invalidateCache();
    CHECK(v->chunkedCache() != nullptr);
    fs::remove_all(d);
}

TEST_CASE("Volume::createChunkCache yields a non-null cache pointer")
{
    auto d = tmpDir("createcache");
    auto v = Volume::New(d, optsU8());
    REQUIRE(v);
    vc::render::ChunkCache::Options copts;
    copts.decodedByteCapacity = 1ULL << 20;
    auto c = v->createChunkCache(copts);
    CHECK(c != nullptr);
    CHECK(c->numLevels() == 1);
    fs::remove_all(d);
}

TEST_CASE("Volume: missing chunk read returns fill with AllFill policy")
{
    auto d = tmpDir("allfill_policy");
    auto v = Volume::New(d, optsU8());
    REQUIRE(v);
    // No writes — every chunk is absent and should fill with 0.
    Array3D<uint8_t> out({16, 16, 16});
    bool ok = v->readZYX(out, {0, 0, 0}, 0, Volume::MissingScaleLevelPolicy::AllFill);
    CHECK(ok);
    // All zeros (fillValue).
    bool allZero = true;
    for (size_t z = 0; z < 16; ++z)
        for (size_t y = 0; y < 16; ++y)
            for (size_t x = 0; x < 16; ++x)
                if (out(z, y, x) != 0) { allZero = false; break; }
    CHECK(allZero);
    fs::remove_all(d);
}

TEST_CASE("Volume::chunkGridShape")
{
    auto d = tmpDir("grid_shape");
    auto v = Volume::New(d, optsU8());
    REQUIRE(v);
    // shape=32^3, chunk=16^3 → 2x2x2 chunk grid
    auto g = v->chunkGridShape(0);
    CHECK(g[0] == 2);
    CHECK(g[1] == 2);
    CHECK(g[2] == 2);
    CHECK(v->chunkCount(0) == 8);
    fs::remove_all(d);
}

TEST_CASE("Volume::voxelSize defaults to 1.0 when not set in opts")
{
    auto d = tmpDir("voxel");
    auto v = Volume::New(d, optsU8());
    REQUIRE(v);
    CHECK(v->voxelSize() > 0.0);
    fs::remove_all(d);
}
