#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/render/ChunkCache.hpp"
#include "vc/core/render/ChunkedPlaneSampler.hpp"
#include "vc/core/render/ZarrChunkFetcher.hpp"
#include "vc/core/types/VcDataset.hpp"

#include <opencv2/core.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <random>
#include <thread>
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
               ("vc_render_test_" + std::to_string(rng()));
        fs::create_directories(path);
    }
    ~TmpDir()
    {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
};

constexpr int kVolDim = 32;
constexpr int kChunkDim = 16;
constexpr int kLevel1Dim = 16;

uint8_t voxelValue(int z, int y, int x) noexcept
{
    return static_cast<uint8_t>((z + y + x) % 256);
}

uint8_t voxelValueL1(int z, int y, int x) noexcept
{
    return static_cast<uint8_t>((z * 7 + y * 5 + x * 3) % 256);
}

fs::path makeFixtureZarr(const fs::path& root, const std::string& level = "0")
{
    fs::create_directories(root);
    const std::vector<size_t> shape = {kVolDim, kVolDim, kVolDim};
    const std::vector<size_t> chunks = {kChunkDim, kChunkDim, kChunkDim};

    auto ds = vc::createZarrDataset(
        root, level, shape, chunks, vc::VcDtype::uint8, "blosc", ".", 0);

    std::vector<uint8_t> chunkBuf(kChunkDim * kChunkDim * kChunkDim);
    const int nChunks = kVolDim / kChunkDim;
    for (int cz = 0; cz < nChunks; ++cz) {
        for (int cy = 0; cy < nChunks; ++cy) {
            for (int cx = 0; cx < nChunks; ++cx) {
                for (int z = 0; z < kChunkDim; ++z) {
                    for (int y = 0; y < kChunkDim; ++y) {
                        for (int x = 0; x < kChunkDim; ++x) {
                            const int gz = cz * kChunkDim + z;
                            const int gy = cy * kChunkDim + y;
                            const int gx = cx * kChunkDim + x;
                            chunkBuf[(z * kChunkDim + y) * kChunkDim + x] =
                                voxelValue(gz, gy, gx);
                        }
                    }
                }
                ds->writeChunk(static_cast<size_t>(cz),
                               static_cast<size_t>(cy),
                               static_cast<size_t>(cx),
                               chunkBuf.data(),
                               chunkBuf.size());
            }
        }
    }
    return root / level;
}

fs::path makeLevel1Zarr(const fs::path& root)
{
    const std::vector<size_t> shape = {kLevel1Dim, kLevel1Dim, kLevel1Dim};
    const std::vector<size_t> chunks = {kLevel1Dim, kLevel1Dim, kLevel1Dim};
    auto ds = vc::createZarrDataset(
        root, "1", shape, chunks, vc::VcDtype::uint8, "blosc", ".", 0);

    std::vector<uint8_t> chunkBuf(kLevel1Dim * kLevel1Dim * kLevel1Dim);
    for (int z = 0; z < kLevel1Dim; ++z) {
        for (int y = 0; y < kLevel1Dim; ++y) {
            for (int x = 0; x < kLevel1Dim; ++x) {
                chunkBuf[(z * kLevel1Dim + y) * kLevel1Dim + x] = voxelValueL1(z, y, x);
            }
        }
    }
    ds->writeChunk(0, 0, 0, chunkBuf.data(), chunkBuf.size());
    return root / "1";
}

fs::path makeSparseZarr(const fs::path& root)
{
    const std::vector<size_t> shape = {kVolDim, kVolDim, kVolDim};
    const std::vector<size_t> chunks = {kChunkDim, kChunkDim, kChunkDim};
    vc::createZarrDataset(
        root, "0", shape, chunks, vc::VcDtype::uint8, "blosc", ".", 42);
    return root / "0";
}

}

TEST_CASE("openLocalZarrPyramid: structure matches fixture")
{
    TmpDir tmp;
    makeFixtureZarr(tmp.path);

    auto opened = vc::render::openLocalZarrPyramid(tmp.path);

    REQUIRE(opened.fetchers.size() == 1);
    CHECK(opened.shapes[0] == std::array<int, 3>{kVolDim, kVolDim, kVolDim});
    CHECK(opened.chunkShapes[0] == std::array<int, 3>{kChunkDim, kChunkDim, kChunkDim});
    CHECK(opened.dtype == vc::render::ChunkDtype::UInt8);
    CHECK(opened.fillValue == doctest::Approx(0.0));
}

TEST_CASE("ZarrChunkFetcher: returns expected bytes for known chunks")
{
    TmpDir tmp;
    makeFixtureZarr(tmp.path);
    auto opened = vc::render::openLocalZarrPyramid(tmp.path);
    REQUIRE(opened.fetchers.size() == 1);

    auto check_chunk = [&](int cz, int cy, int cx) {
        auto result = opened.fetchers[0]->fetch({0, cz, cy, cx});
        REQUIRE(result.status == vc::render::ChunkFetchStatus::Found);
        REQUIRE(result.bytes.size() ==
                static_cast<std::size_t>(kChunkDim * kChunkDim * kChunkDim));
        for (int z = 0; z < kChunkDim; ++z) {
            for (int y = 0; y < kChunkDim; ++y) {
                for (int x = 0; x < kChunkDim; ++x) {
                    const auto idx = (z * kChunkDim + y) * kChunkDim + x;
                    const auto expected = voxelValue(
                        cz * kChunkDim + z, cy * kChunkDim + y, cx * kChunkDim + x);
                    REQUIRE(static_cast<uint8_t>(result.bytes[idx]) == expected);
                }
            }
        }
    };
    check_chunk(0, 0, 0);
    check_chunk(1, 1, 1);
    check_chunk(0, 1, 1);
}

TEST_CASE("ChunkCache::getChunkBlocking: returns decoded chunk")
{
    TmpDir tmp;
    makeFixtureZarr(tmp.path);
    auto cache = vc::render::createChunkCache(
        vc::render::openLocalZarrPyramid(tmp.path),
        64ULL * 1024 * 1024,
        4);

    auto result = cache->getChunkBlocking(0, 1, 1, 1);
    REQUIRE(result.status == vc::render::ChunkStatus::Data);
    REQUIRE(result.bytes);
    REQUIRE(result.bytes->size() ==
            static_cast<std::size_t>(kChunkDim * kChunkDim * kChunkDim));

    CHECK(static_cast<uint8_t>((*result.bytes)[0]) ==
          voxelValue(kChunkDim, kChunkDim, kChunkDim));
}

TEST_CASE("ChunkCache::tryGetChunk: miss queues, listener fires once data lands")
{
    TmpDir tmp;
    makeFixtureZarr(tmp.path);
    auto cache = vc::render::createChunkCache(
        vc::render::openLocalZarrPyramid(tmp.path),
        64ULL * 1024 * 1024,
        4);

    std::atomic<int> notifications{0};
    auto id = cache->addChunkReadyListener([&] { notifications.fetch_add(1); });

    auto first = cache->tryGetChunk(0, 0, 0, 0);
    CHECK((first.status == vc::render::ChunkStatus::MissQueued ||
           first.status == vc::render::ChunkStatus::Data));

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < deadline) {
        auto poll = cache->tryGetChunk(0, 0, 0, 0);
        if (poll.status == vc::render::ChunkStatus::Data) {
            REQUIRE(poll.bytes);
            CHECK(static_cast<uint8_t>((*poll.bytes)[0]) == voxelValue(0, 0, 0));
            CHECK(notifications.load() >= 1);
            cache->removeChunkReadyListener(id);
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    cache->removeChunkReadyListener(id);
    FAIL("chunk did not become Data within timeout");
}

TEST_CASE("ChunkedPlaneSampler::samplePlaneLevel: axis-aligned slice has gradient pixels")
{
    TmpDir tmp;
    makeFixtureZarr(tmp.path);
    auto cache = vc::render::createChunkCache(
        vc::render::openLocalZarrPyramid(tmp.path),
        64ULL * 1024 * 1024,
        4);

    cv::Mat_<uint8_t> out(kVolDim, kVolDim, uint8_t(0));
    cv::Mat_<uint8_t> coverage(kVolDim, kVolDim, uint8_t(0));

    const cv::Vec3f origin(0.0f, 0.0f, 10.0f);
    const cv::Vec3f vxStep(1.0f, 0.0f, 0.0f);
    const cv::Vec3f vyStep(0.0f, 1.0f, 0.0f);

    auto keys = vc::render::ChunkedPlaneSampler::collectPlaneDependencies(
        *cache, 0, origin, vxStep, vyStep, coverage);
    cache->prefetchChunks(keys, /*wait=*/true);

    auto stats = vc::render::ChunkedPlaneSampler::samplePlaneLevel(
        *cache, 0, origin, vxStep, vyStep, out, coverage);

    CHECK(stats.coveredPixels == kVolDim * kVolDim);

    for (int py = 0; py < kVolDim; ++py) {
        for (int px = 0; px < kVolDim; ++px) {
            REQUIRE(coverage(py, px) != 0);
            CHECK(out(py, px) == voxelValue(10, py, px));
        }
    }
}

TEST_CASE("Multi-level pyramid: levels and transforms")
{
    TmpDir tmp;
    makeFixtureZarr(tmp.path);
    makeLevel1Zarr(tmp.path);

    auto opened = vc::render::openLocalZarrPyramid(tmp.path);
    REQUIRE(opened.fetchers.size() == 2);
    CHECK(opened.shapes[0] == std::array<int, 3>{kVolDim, kVolDim, kVolDim});
    CHECK(opened.shapes[1] == std::array<int, 3>{kLevel1Dim, kLevel1Dim, kLevel1Dim});
    CHECK(opened.transforms[0].scaleFromLevel0[0] == doctest::Approx(1.0));
    CHECK(opened.transforms[1].scaleFromLevel0[0] == doctest::Approx(0.5));
}

TEST_CASE("ChunkedPlaneSampler: level-1 sampling reads downsampled data")
{
    TmpDir tmp;
    makeFixtureZarr(tmp.path);
    makeLevel1Zarr(tmp.path);
    auto cache = vc::render::createChunkCache(
        vc::render::openLocalZarrPyramid(tmp.path),
        64ULL * 1024 * 1024,
        4);

    cv::Mat_<uint8_t> out(kLevel1Dim, kLevel1Dim, uint8_t(0));
    cv::Mat_<uint8_t> coverage(kLevel1Dim, kLevel1Dim, uint8_t(0));

    const cv::Vec3f origin(0.0f, 0.0f, 6.0f);
    const cv::Vec3f vxStep(2.0f, 0.0f, 0.0f);
    const cv::Vec3f vyStep(0.0f, 2.0f, 0.0f);

    auto keys = vc::render::ChunkedPlaneSampler::collectPlaneDependencies(
        *cache, 1, origin, vxStep, vyStep, coverage);
    cache->prefetchChunks(keys, true);

    auto stats = vc::render::ChunkedPlaneSampler::samplePlaneLevel(
        *cache, 1, origin, vxStep, vyStep, out, coverage);

    CHECK(stats.coveredPixels == kLevel1Dim * kLevel1Dim);
    CHECK(out(0, 0) == voxelValueL1(3, 0, 0));
    CHECK(out(4, 5) == voxelValueL1(3, 4, 5));
}

TEST_CASE("Sparse zarr: missing chunk resolves to AllFill")
{
    TmpDir tmp;
    makeSparseZarr(tmp.path);
    auto cache = vc::render::createChunkCache(
        vc::render::openLocalZarrPyramid(tmp.path),
        64ULL * 1024 * 1024,
        4);

    auto result = cache->getChunkBlocking(0, 0, 0, 0);
    CHECK(result.status == vc::render::ChunkStatus::AllFill);
}

TEST_CASE("ChunkedPlaneSampler: sparse zarr produces fill-value pixels")
{
    TmpDir tmp;
    makeSparseZarr(tmp.path);
    auto cache = vc::render::createChunkCache(
        vc::render::openLocalZarrPyramid(tmp.path),
        64ULL * 1024 * 1024,
        4);

    cv::Mat_<uint8_t> out(kVolDim, kVolDim, uint8_t(99));
    cv::Mat_<uint8_t> coverage(kVolDim, kVolDim, uint8_t(0));

    const cv::Vec3f origin(0.0f, 0.0f, 5.0f);
    const cv::Vec3f vxStep(1.0f, 0.0f, 0.0f);
    const cv::Vec3f vyStep(0.0f, 1.0f, 0.0f);

    auto keys = vc::render::ChunkedPlaneSampler::collectPlaneDependencies(
        *cache, 0, origin, vxStep, vyStep, coverage);
    cache->prefetchChunks(keys, true);

    vc::render::ChunkedPlaneSampler::samplePlaneLevel(
        *cache, 0, origin, vxStep, vyStep, out, coverage);

    for (int py = 0; py < kVolDim; ++py) {
        for (int px = 0; px < kVolDim; ++px) {
            CHECK(out(py, px) == 42);
        }
    }
}

TEST_CASE("ChunkedPlaneSampler: out-of-bounds plane is fully filled with fill-value")
{
    TmpDir tmp;
    makeFixtureZarr(tmp.path);
    auto cache = vc::render::createChunkCache(
        vc::render::openLocalZarrPyramid(tmp.path),
        64ULL * 1024 * 1024,
        4);

    cv::Mat_<uint8_t> out(8, 8, uint8_t(99));
    cv::Mat_<uint8_t> coverage(8, 8, uint8_t(0));

    const cv::Vec3f origin(0.0f, 0.0f, 1000.0f);
    const cv::Vec3f vxStep(1.0f, 0.0f, 0.0f);
    const cv::Vec3f vyStep(0.0f, 1.0f, 0.0f);

    auto stats = vc::render::ChunkedPlaneSampler::samplePlaneLevel(
        *cache, 0, origin, vxStep, vyStep, out, coverage);

    CHECK(stats.coveredPixels == 8 * 8);
    for (int py = 0; py < 8; ++py)
        for (int px = 0; px < 8; ++px)
            CHECK(out(py, px) == 0);
}

TEST_CASE("ChunkedPlaneSampler::sampleCoordsLevel: per-pixel coords")
{
    TmpDir tmp;
    makeFixtureZarr(tmp.path);
    auto cache = vc::render::createChunkCache(
        vc::render::openLocalZarrPyramid(tmp.path),
        64ULL * 1024 * 1024,
        4);

    cv::Mat_<cv::Vec3f> coords(8, 8);
    for (int y = 0; y < 8; ++y)
        for (int x = 0; x < 8; ++x)
            coords(y, x) = cv::Vec3f(float(x), float(y), 5.0f);

    cv::Mat_<uint8_t> out(8, 8, uint8_t(0));
    cv::Mat_<uint8_t> coverage(8, 8, uint8_t(0));

    auto keys = vc::render::ChunkedPlaneSampler::collectCoordsDependencies(
        *cache, 0, coords, coverage);
    cache->prefetchChunks(keys, true);

    auto stats = vc::render::ChunkedPlaneSampler::sampleCoordsLevel(
        *cache, 0, coords, out, coverage);

    CHECK(stats.coveredPixels == 64);
    for (int py = 0; py < 8; ++py)
        for (int px = 0; px < 8; ++px)
            CHECK(out(py, px) == voxelValue(5, py, px));
}

TEST_CASE("ChunkCache: LRU evicts under capacity pressure")
{
    TmpDir tmp;
    makeFixtureZarr(tmp.path);
    auto opened = vc::render::openLocalZarrPyramid(tmp.path);
    std::vector<vc::render::ChunkCache::LevelInfo> levels;
    for (std::size_t i = 0; i < opened.shapes.size(); ++i)
        levels.push_back({opened.shapes[i], opened.chunkShapes[i], opened.transforms[i]});

    vc::render::ChunkCache::Options opts;
    opts.decodedByteCapacity = 8 * 1024;
    opts.maxConcurrentReads = 4;
    auto cache = std::make_shared<vc::render::ChunkCache>(
        levels, opened.fetchers, opened.fillValue, opened.dtype, opts);

    for (int cz = 0; cz < 2; ++cz)
        for (int cy = 0; cy < 2; ++cy)
            for (int cx = 0; cx < 2; ++cx)
                cache->getChunkBlocking(0, cz, cy, cx);

    auto stats = cache->stats();
    CHECK(stats.decodedByteCapacity == opts.decodedByteCapacity);
    CHECK(stats.decodedBytes <= opts.decodedByteCapacity * 4);
}

TEST_CASE("ChunkCache: persistent cache round-trips chunks across instances")
{
    TmpDir tmp, cacheTmp;
    makeFixtureZarr(tmp.path);

    vc::render::ChunkCache::Options opts;
    opts.decodedByteCapacity = 64ULL * 1024 * 1024;
    opts.maxConcurrentReads = 4;
    opts.persistentCachePath = cacheTmp.path;

    {
        auto opened = vc::render::openLocalZarrPyramid(tmp.path);
        std::vector<vc::render::ChunkCache::LevelInfo> levels;
        for (std::size_t i = 0; i < opened.shapes.size(); ++i)
            levels.push_back({opened.shapes[i], opened.chunkShapes[i], opened.transforms[i]});
        vc::render::ChunkCache cache(levels, opened.fetchers, opened.fillValue, opened.dtype, opts);
        REQUIRE(cache.getChunkBlocking(0, 0, 0, 0).status == vc::render::ChunkStatus::Data);
        REQUIRE(cache.getChunkBlocking(0, 1, 0, 1).status == vc::render::ChunkStatus::Data);
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }

    bool persistedFile = false;
    for (auto const& entry : fs::recursive_directory_iterator(cacheTmp.path))
        if (entry.is_regular_file()) { persistedFile = true; break; }
    CHECK(persistedFile);

    auto opened2 = vc::render::openLocalZarrPyramid(tmp.path);
    std::vector<vc::render::ChunkCache::LevelInfo> levels2;
    for (std::size_t i = 0; i < opened2.shapes.size(); ++i)
        levels2.push_back({opened2.shapes[i], opened2.chunkShapes[i], opened2.transforms[i]});
    vc::render::ChunkCache cache2(levels2, opened2.fetchers, opened2.fillValue, opened2.dtype, opts);

    auto r = cache2.getChunkBlocking(0, 0, 0, 0);
    REQUIRE(r.status == vc::render::ChunkStatus::Data);
    REQUIRE(r.bytes->size() == kChunkDim * kChunkDim * kChunkDim);
    CHECK(static_cast<uint8_t>((*r.bytes)[0]) == voxelValue(0, 0, 0));
}

TEST_CASE("ChunkCache: multiple listeners each fire and remove works")
{
    TmpDir tmp;
    makeFixtureZarr(tmp.path);
    auto cache = vc::render::createChunkCache(
        vc::render::openLocalZarrPyramid(tmp.path), 64ULL * 1024 * 1024, 4);

    std::atomic<int> a{0}, b{0};
    auto idA = cache->addChunkReadyListener([&] { a.fetch_add(1); });
    auto idB = cache->addChunkReadyListener([&] { b.fetch_add(1); });

    cache->tryGetChunk(0, 0, 0, 0);
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < deadline) {
        if (cache->tryGetChunk(0, 0, 0, 0).status == vc::render::ChunkStatus::Data) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    CHECK(a.load() >= 1);
    CHECK(b.load() >= 1);

    cache->removeChunkReadyListener(idA);
    int aBefore = a.load();
    cache->invalidate();
    cache->tryGetChunk(0, 1, 1, 1);
    deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < deadline) {
        if (cache->tryGetChunk(0, 1, 1, 1).status == vc::render::ChunkStatus::Data) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    CHECK(b.load() > 1);
    CHECK(a.load() == aBefore);
    cache->removeChunkReadyListener(idB);
}

TEST_CASE("ChunkedPlaneSampler::collectPlaneDependencies: returns chunks for the plane")
{
    TmpDir tmp;
    makeFixtureZarr(tmp.path);
    auto cache = vc::render::createChunkCache(
        vc::render::openLocalZarrPyramid(tmp.path), 64ULL * 1024 * 1024, 4);

    cv::Mat_<uint8_t> coverage(kVolDim, kVolDim, uint8_t(0));
    const cv::Vec3f origin(0.0f, 0.0f, 10.0f);
    const cv::Vec3f vxStep(1.0f, 0.0f, 0.0f);
    const cv::Vec3f vyStep(0.0f, 1.0f, 0.0f);

    auto keys = vc::render::ChunkedPlaneSampler::collectPlaneDependencies(
        *cache, 0, origin, vxStep, vyStep, coverage);

    REQUIRE(!keys.empty());
    for (const auto& k : keys) {
        CHECK(k.level == 0);
        CHECK(k.iz == 0);
    }
}

TEST_CASE("ChunkedPlaneSampler::requestPlaneDependencies: queues without filling output")
{
    TmpDir tmp;
    makeFixtureZarr(tmp.path);
    auto cache = vc::render::createChunkCache(
        vc::render::openLocalZarrPyramid(tmp.path), 64ULL * 1024 * 1024, 4);

    cv::Mat_<uint8_t> coverage(kVolDim, kVolDim, uint8_t(0));
    const cv::Vec3f origin(0.0f, 0.0f, 10.0f);
    const cv::Vec3f vxStep(1.0f, 0.0f, 0.0f);
    const cv::Vec3f vyStep(0.0f, 1.0f, 0.0f);

    auto stats = vc::render::ChunkedPlaneSampler::requestPlaneDependencies(
        *cache, 0, origin, vxStep, vyStep, coverage);

    CHECK(stats.requestedChunks > 0);
}

TEST_CASE("ChunkedPlaneSampler::samplePlaneCoarseToFine: fine pixels overwrite coarse ones")
{
    TmpDir tmp;
    makeFixtureZarr(tmp.path);
    makeLevel1Zarr(tmp.path);
    auto cache = vc::render::createChunkCache(
        vc::render::openLocalZarrPyramid(tmp.path), 64ULL * 1024 * 1024, 4);

    cv::Mat_<uint8_t> out(kVolDim, kVolDim, uint8_t(0));
    cv::Mat_<uint8_t> coverage(kVolDim, kVolDim, uint8_t(0));
    const cv::Vec3f origin(0.0f, 0.0f, 10.0f);
    const cv::Vec3f vxStep(1.0f, 0.0f, 0.0f);
    const cv::Vec3f vyStep(0.0f, 1.0f, 0.0f);

    auto k0 = vc::render::ChunkedPlaneSampler::collectPlaneDependencies(*cache, 0, origin, vxStep, vyStep, coverage);
    auto k1 = vc::render::ChunkedPlaneSampler::collectPlaneDependencies(*cache, 1, origin, vxStep, vyStep, coverage);
    cache->prefetchChunks(k0, true);
    cache->prefetchChunks(k1, true);

    auto stats = vc::render::ChunkedPlaneSampler::samplePlaneCoarseToFine(
        *cache, 0, origin, vxStep, vyStep, out, coverage);
    CHECK(stats.coveredPixels == kVolDim * kVolDim);

    for (int py = 0; py < kVolDim; ++py)
        for (int px = 0; px < kVolDim; ++px)
            CHECK(out(py, px) == voxelValue(10, py, px));
}

TEST_CASE("ChunkedPlaneSampler::sampleCoordsFineToCoarse: explicit coords with fallback")
{
    TmpDir tmp;
    makeFixtureZarr(tmp.path);
    auto cache = vc::render::createChunkCache(
        vc::render::openLocalZarrPyramid(tmp.path), 64ULL * 1024 * 1024, 4);

    cv::Mat_<cv::Vec3f> coords(8, 8);
    for (int y = 0; y < 8; ++y)
        for (int x = 0; x < 8; ++x)
            coords(y, x) = cv::Vec3f(float(x), float(y), 5.0f);

    cv::Mat_<uint8_t> out(8, 8, uint8_t(0));
    cv::Mat_<uint8_t> coverage(8, 8, uint8_t(0));

    auto keys = vc::render::ChunkedPlaneSampler::collectCoordsDependencies(*cache, 0, coords, coverage);
    cache->prefetchChunks(keys, true);

    auto stats = vc::render::ChunkedPlaneSampler::sampleCoordsFineToCoarse(
        *cache, 0, coords, out, coverage);

    CHECK(stats.coveredPixels == 64);
    for (int py = 0; py < 8; ++py)
        for (int px = 0; px < 8; ++px)
            CHECK(out(py, px) == voxelValue(5, py, px));
}

TEST_CASE("ChunkedPlaneSampler: tricubic sampling at non-integer coords produces values")
{
    TmpDir tmp;
    makeFixtureZarr(tmp.path);
    auto cache = vc::render::createChunkCache(
        vc::render::openLocalZarrPyramid(tmp.path), 64ULL * 1024 * 1024, 4);

    cv::Mat_<uint8_t> out(kVolDim, kVolDim, uint8_t(0));
    cv::Mat_<uint8_t> coverage(kVolDim, kVolDim, uint8_t(0));
    const cv::Vec3f origin(0.5f, 0.5f, 10.5f);
    const cv::Vec3f vxStep(1.0f, 0.0f, 0.0f);
    const cv::Vec3f vyStep(0.0f, 1.0f, 0.0f);

    vc::render::ChunkedPlaneSampler::Options opts(vc::Sampling::Trilinear, 32);
    auto keys = vc::render::ChunkedPlaneSampler::collectPlaneDependencies(
        *cache, 0, origin, vxStep, vyStep, coverage, opts);
    cache->prefetchChunks(keys, true);
    auto stats = vc::render::ChunkedPlaneSampler::samplePlaneLevel(
        *cache, 0, origin, vxStep, vyStep, out, coverage, opts);

    CHECK(stats.coveredPixels > 0);
    int nonzero = 0;
    for (int py = 0; py < kVolDim; ++py)
        for (int px = 0; px < kVolDim; ++px)
            if (out(py, px) != 0) ++nonzero;
    CHECK(nonzero > 0);
}

TEST_CASE("ChunkCache: stats and invalidate")
{
    TmpDir tmp;
    makeFixtureZarr(tmp.path);
    auto cache = vc::render::createChunkCache(
        vc::render::openLocalZarrPyramid(tmp.path),
        64ULL * 1024 * 1024,
        4);

    CHECK(cache->stats().decodedBytes == 0);

    auto result = cache->getChunkBlocking(0, 0, 0, 0);
    REQUIRE(result.status == vc::render::ChunkStatus::Data);
    const auto bytesAfterFetch = cache->stats().decodedBytes;
    CHECK(bytesAfterFetch >=
          static_cast<std::size_t>(kChunkDim * kChunkDim * kChunkDim));

    cache->invalidate();
    CHECK(cache->stats().decodedBytes == 0);

    auto result2 = cache->getChunkBlocking(0, 0, 0, 0);
    CHECK(result2.status == vc::render::ChunkStatus::Data);
}
