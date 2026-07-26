// Cover openLocalZarrPyramid + createChunkCache and the local-fetcher path.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/render/ZarrChunkFetcher.hpp"
#include "vc/core/render/ChunkCache.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VcDataset.hpp"

#include <filesystem>
#include <random>
#include <stdexcept>
#include <vector>

namespace fs = std::filesystem;
using vc::render::openLocalZarrPyramid;
using vc::render::OpenedChunkedZarr;
using vc::render::createChunkCache;

namespace {

fs::path tmpDir(const std::string& tag)
{
    std::mt19937_64 rng(std::random_device{}());
    auto p = fs::temp_directory_path() /
             ("vc_zcf_" + tag + "_" + std::to_string(rng()));
    fs::create_directories(p);
    return p;
}

fs::path makeLocalVolume(const fs::path& dir, size_t numLevels = 2,
                        vc::render::ChunkDtype dtype = vc::render::ChunkDtype::UInt8)
{
    Volume::ZarrCreateOptions opts;
    opts.shapeZYX = {64, 64, 64};
    opts.chunkShapeZYX = {32, 32, 32};
    opts.numLevels = numLevels;
    opts.compressor = "none";
    opts.overwriteExisting = true;
    opts.dtype = dtype;
    auto v = Volume::New(dir, opts);
    REQUIRE(v);
    return dir;
}

} // namespace

TEST_CASE("openLocalZarrPyramid: opens a multi-level local zarr")
{
    auto d = tmpDir("multi");
    makeLocalVolume(d, /*numLevels=*/3);
    auto opened = openLocalZarrPyramid(d);
    CHECK(opened.fetchers.size() >= 1);
    CHECK_FALSE(opened.shapes.empty());
    CHECK(opened.shapes[0][0] == 64);
    fs::remove_all(d);
}

TEST_CASE("openLocalZarrPyramid: single-level (no subdir) is also accepted")
{
    auto d = tmpDir("single");
    // Single-level: just a .zarray at the root, no /0/, /1/ subdirs.
    // Create via VcDataset directly to control this.
    auto ds = vc::createZarrDataset(d, "arr",
        /*shape=*/{32, 32, 32}, /*chunks=*/{32, 32, 32},
        vc::VcDtype::uint8, "none");
    REQUIRE(ds);
    // openLocalZarrPyramid looks for numeric subdirs first, then falls back
    // to opening `root` directly. Point at the array dir.
    auto opened = openLocalZarrPyramid(d / "arr");
    CHECK(opened.fetchers.size() >= 1);
    fs::remove_all(d);
}

TEST_CASE("openLocalZarrPyramid: uint16 volume")
{
    auto d = tmpDir("u16");
    makeLocalVolume(d, /*numLevels=*/2, vc::render::ChunkDtype::UInt16);
    auto opened = openLocalZarrPyramid(d);
    CHECK(opened.fetchers.size() >= 1);
    CHECK(opened.dtype == vc::render::ChunkDtype::UInt16);
    fs::remove_all(d);
}

TEST_CASE("openLocalZarrPyramid: missing dir throws")
{
    CHECK_THROWS(openLocalZarrPyramid("/__no__/__where__"));
}

TEST_CASE("createChunkCache wraps openLocalZarrPyramid result")
{
    auto d = tmpDir("cc_wrap");
    makeLocalVolume(d, 2);
    auto opened = openLocalZarrPyramid(d);
    auto cache = createChunkCache(std::move(opened),
        /*decodedByteCapacity=*/1ULL << 20);
    REQUIRE(cache);
    CHECK(cache->numLevels() >= 1);
    fs::remove_all(d);
}

TEST_CASE("validateAndRebaseVcPyramid maps physical level two to logical zero")
{
    auto d = tmpDir("rebase_two");
    makeLocalVolume(d, /*numLevels=*/6);
    auto physical = openLocalZarrPyramid(d);
    REQUIRE(physical.fetchers.size() == 6);
    const auto physicalLevelTwoFetcher = physical.fetchers[2];

    auto rebased = vc::render::validateAndRebaseVcPyramid(std::move(physical), 2);
    CHECK(rebased.fetchers.size() == 4);
    CHECK(rebased.shapes[0] == std::array<int, 3>{16, 16, 16});
    CHECK(rebased.chunkShapes.size() == 4);
    CHECK(rebased.storageChunkShapes.size() == 4);
    CHECK(rebased.levelNumbers == std::vector<int>{0, 1, 2, 3});
    CHECK(rebased.transforms[0].scaleFromLevel0 == std::array<double, 3>{1.0, 1.0, 1.0});
    CHECK(rebased.transforms[1].scaleFromLevel0 == std::array<double, 3>{0.5, 0.5, 0.5});
    CHECK(rebased.fetchers[0] == physicalLevelTwoFetcher);
    fs::remove_all(d);
}

TEST_CASE("validateAndRebaseVcPyramid accepts variable contiguous pyramid lengths")
{
    for (const size_t levels : {size_t{5}, size_t{7}}) {
        auto d = tmpDir("rebase_length");
        makeLocalVolume(d, levels);
        auto rebased = vc::render::validateAndRebaseVcPyramid(
            openLocalZarrPyramid(d), 2);
        CHECK(rebased.fetchers.size() == levels - 2);
        fs::remove_all(d);
    }
}

TEST_CASE("validateAndRebaseVcPyramid rejects gaps and invalid retained fill values")
{
    auto d = tmpDir("rebase_invalid");
    makeLocalVolume(d, /*numLevels=*/4);

    auto gap = openLocalZarrPyramid(d);
    gap.fetchers[1].reset();
    CHECK_THROWS_WITH_AS(
        vc::render::validateAndRebaseVcPyramid(std::move(gap), 2),
        doctest::Contains("gap at /1"), std::runtime_error);

    auto fills = openLocalZarrPyramid(d);
    fills.fillValues[3] = 1.0;
    CHECK_THROWS_WITH_AS(
        vc::render::validateAndRebaseVcPyramid(std::move(fills), 2),
        doctest::Contains("consistent fill value"), std::runtime_error);

    auto shape = openLocalZarrPyramid(d);
    shape.shapes[1][0] += shape.storageChunkShapes[1][0];
    CHECK_THROWS_WITH_AS(
        vc::render::validateAndRebaseVcPyramid(std::move(shape), 2),
        doctest::Contains("dyadic downscale"), std::runtime_error);
    fs::remove_all(d);
}

TEST_CASE("ZarrChunkFetcher fetches a present chunk from local")
{
    auto d = tmpDir("fetch_present");
    auto v = Volume::New(d, []() {
        Volume::ZarrCreateOptions o;
        o.shapeZYX = {32, 32, 32};
        o.chunkShapeZYX = {32, 32, 32};
        o.numLevels = 1;
        o.compressor = "none";
        o.overwriteExisting = true;
        return o;
    }());
    REQUIRE(v);
    Array3D<uint8_t> in({32, 32, 32}, /*fill=*/200);
    v->writeZYX(in, {0, 0, 0}, 0);

    auto opened = openLocalZarrPyramid(d);
    REQUIRE_FALSE(opened.fetchers.empty());
    vc::render::ChunkKey key{0, 0, 0, 0};
    auto r = opened.fetchers[0]->fetch(key);
    CHECK(r.status == vc::render::ChunkFetchStatus::Found);
    CHECK_FALSE(r.bytes.empty());
    fs::remove_all(d);
}

TEST_CASE("ZarrChunkFetcher fetches a Missing chunk")
{
    auto d = tmpDir("fetch_missing");
    auto v = Volume::New(d, []() {
        Volume::ZarrCreateOptions o;
        o.shapeZYX = {32, 32, 32};
        o.chunkShapeZYX = {32, 32, 32};
        o.numLevels = 1;
        o.compressor = "none";
        o.overwriteExisting = true;
        return o;
    }());
    REQUIRE(v);

    auto opened = openLocalZarrPyramid(d);
    REQUIRE_FALSE(opened.fetchers.empty());
    // No chunks written → first chunk is Missing.
    auto r = opened.fetchers[0]->fetch({0, 0, 0, 0});
    CHECK(r.status == vc::render::ChunkFetchStatus::Missing);
    fs::remove_all(d);
}
