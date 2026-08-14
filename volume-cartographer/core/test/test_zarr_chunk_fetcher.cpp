// Cover openLocalZarrPyramid + createChunkCache and the local-fetcher path.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/render/ZarrChunkFetcher.hpp"
#include "vc/core/render/ChunkCache.hpp"
#include "vc/core/render/ZarrDownloadBenchmark.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VcDataset.hpp"

#include <utils/zarr.hpp>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
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

TEST_CASE("remoteLevelKeysFromZattrs binds numeric dataset paths by value")
{
    // A scaledown export (lasagna/tiled_predict3d.py with first_level=2)
    // advertises only its coarse datasets; positional binding would register
    // the level-2 array as full resolution.
    auto d = tmpDir("zattrs_numeric");
    {
        std::ofstream f(d / ".zattrs");
        f << R"({"multiscales":[{"datasets":[{"path":"2"},{"path":"3"}]}]})";
    }
    auto store = std::make_shared<utils::FileSystemStore>(d);
    const auto keys = vc::render::remoteLevelKeysFromZattrs(store, 0);
    REQUIRE(keys.size() == 2);
    CHECK(keys[0] == std::pair<int, std::string>{2, "2"});
    CHECK(keys[1] == std::pair<int, std::string>{3, "3"});
    fs::remove_all(d);
}

TEST_CASE("remoteLevelKeysFromZattrs keeps positional binding for named datasets")
{
    auto d = tmpDir("zattrs_named");
    {
        std::ofstream f(d / ".zattrs");
        f << R"({"multiscales":[{"datasets":[{"path":"s0"},{"path":"s1"}]}]})";
    }
    auto store = std::make_shared<utils::FileSystemStore>(d);
    const auto keys = vc::render::remoteLevelKeysFromZattrs(store, 0);
    REQUIRE(keys.size() == 2);
    CHECK(keys[0] == std::pair<int, std::string>{0, "s0"});
    CHECK(keys[1] == std::pair<int, std::string>{1, "s1"});
    fs::remove_all(d);
}

TEST_CASE("remoteLevelKeysFromZattrs is unchanged for standard zero-based pyramids")
{
    auto d = tmpDir("zattrs_standard");
    {
        std::ofstream f(d / ".zattrs");
        f << R"({"multiscales":[{"datasets":[{"path":"0"},{"path":"1"},{"path":"2"}]}]})";
    }
    auto store = std::make_shared<utils::FileSystemStore>(d);
    const auto keys = vc::render::remoteLevelKeysFromZattrs(store, 0);
    REQUIRE(keys.size() == 3);
    for (int level = 0; level < 3; ++level) {
        CHECK(keys[static_cast<std::size_t>(level)] ==
              std::pair<int, std::string>{level, std::to_string(level)});
    }
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

TEST_CASE("Zarr download benchmark selects unique distributed logical chunks")
{
    const auto first = vc::render::selectZarrDownloadBenchmarkChunks(
        {17, 9, 5}, {4, 4, 4}, 2, 100, 41);
    const auto second = vc::render::selectZarrDownloadBenchmarkChunks(
        {17, 9, 5}, {4, 4, 4}, 2, 100, 41);
    REQUIRE(first.size() == 30);
    CHECK(first == second);

    std::unordered_set<vc::render::ChunkKey, vc::render::ChunkKeyHash> unique;
    for (const auto& key : first) {
        CHECK(key.level == 2);
        CHECK(key.iz >= 0);
        CHECK(key.iz < 5);
        CHECK(key.iy >= 0);
        CHECK(key.iy < 3);
        CHECK(key.ix >= 0);
        CHECK(key.ix < 2);
        unique.insert(key);
    }
    CHECK(unique.size() == first.size());
}

TEST_CASE("Zarr download benchmark uses encoded fetches and fixed admission")
{
    class EncodedFetcher final : public vc::render::IChunkFetcher {
    public:
        vc::render::ChunkFetchResult fetch(const vc::render::ChunkKey&) override
        {
            ++decodedCalls;
            return {};
        }

        vc::render::ChunkFetchResult fetchEncoded(
            const vc::render::ChunkKey&) override
        {
            ++calls;
            vc::render::ChunkFetchResult result;
            result.status = vc::render::ChunkFetchStatus::Found;
            result.bytes.resize(1024);
            return result;
        }

        std::atomic_size_t calls{0};
        std::atomic_size_t decodedCalls{0};
    };

    auto fetcher = std::make_shared<EncodedFetcher>();
    OpenedChunkedZarr opened;
    opened.shapes = {{64, 64, 64}};
    opened.chunkShapes = {{16, 16, 16}};
    opened.fetchers = {fetcher};

    vc::render::ZarrDownloadBenchmarkOptions options;
    options.chunkCount = 12;
    options.workers = 1;
    options.schedule = vc::render::ZarrDownloadSchedule::Fixed;
    std::vector<vc::render::ZarrDownloadProgress> progress;
    options.progressInterval = std::chrono::hours(1);
    options.progressCallback = [&](const auto& update) {
        progress.push_back(update);
    };
    const auto result = vc::render::runZarrDownloadBenchmark(opened, options);

    CHECK(fetcher->calls.load() == 12);
    CHECK(fetcher->decodedCalls.load() == 0);
    CHECK(result.requestedChunks == 12);
    CHECK(result.foundChunks == 12);
    CHECK(result.encodedBytes == 12 * 1024);
    CHECK(result.httpErrors == 0);
    CHECK(result.ioErrors == 0);
    CHECK(result.decodeErrors == 0);
    CHECK(result.sinkErrors == 0);
    CHECK(result.peakActive == 1);
    CHECK(result.finalTransferStats.admissionLimit == 1);
    CHECK_FALSE(result.finalTransferStats.adaptive);
    REQUIRE(progress.size() == 1);
    CHECK(progress[0].queuedChunks == 0);
    CHECK(progress[0].downloadingChunks == 0);
    CHECK(progress[0].completedChunks == 12);
    CHECK(progress[0].encodedBytes == 12 * 1024);
}

TEST_CASE("Zarr download benchmark can write encoded payloads to a temporary sink")
{
    class EncodedFetcher final : public vc::render::IChunkFetcher {
    public:
        vc::render::ChunkFetchResult fetch(const vc::render::ChunkKey&) override
        {
            return {};
        }

        vc::render::ChunkFetchResult fetchEncoded(
            const vc::render::ChunkKey&) override
        {
            vc::render::ChunkFetchResult result;
            result.status = vc::render::ChunkFetchStatus::Found;
            result.bytes.resize(17, std::byte{0x2a});
            return result;
        }
    };

    const auto output = tmpDir("download_benchmark_sink");
    OpenedChunkedZarr opened;
    opened.shapes = {{32, 32, 32}};
    opened.chunkShapes = {{16, 16, 16}};
    opened.fetchers = {std::make_shared<EncodedFetcher>()};

    vc::render::ZarrDownloadBenchmarkOptions options;
    options.chunkCount = 3;
    options.workers = 1;
    options.schedule = vc::render::ZarrDownloadSchedule::Fixed;
    options.outputDirectory = output;
    const auto result = vc::render::runZarrDownloadBenchmark(opened, options);

    CHECK(result.foundChunks == 3);
    CHECK(result.encodedBytes == 51);
    CHECK(result.sinkErrors == 0);
    std::size_t files = 0;
    for (const auto& entry : fs::directory_iterator(output)) {
        CHECK(entry.is_regular_file());
        CHECK(entry.file_size() == 17);
        ++files;
    }
    CHECK(files == 3);
    fs::remove_all(output);
}
