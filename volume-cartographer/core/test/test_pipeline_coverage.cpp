#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/render/ChunkCache.hpp"
#include "vc/core/render/ChunkedPlaneSampler.hpp"
#include "vc/core/render/ZarrChunkFetcher.hpp"
#include "vc/core/types/VcDataset.hpp"

#include <opencv2/core.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <future>
#include <memory>
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
               ("vc_pcov_test_" + std::to_string(rng()));
        fs::create_directories(path);
    }
    ~TmpDir()
    {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
};

uint8_t voxelValue(int z, int y, int x) noexcept
{
    return static_cast<uint8_t>((z + y + x) % 256);
}

constexpr int kVolDim = 32;
constexpr int kChunkDim = 16;

fs::path makeFixture(const fs::path& root)
{
    auto ds = vc::createZarrDataset(
        root, "0",
        {kVolDim, kVolDim, kVolDim},
        {kChunkDim, kChunkDim, kChunkDim},
        vc::VcDtype::uint8, "blosc", ".", 0);
    std::vector<uint8_t> chunkBuf(kChunkDim * kChunkDim * kChunkDim);
    for (int cz = 0; cz < kVolDim / kChunkDim; ++cz)
        for (int cy = 0; cy < kVolDim / kChunkDim; ++cy)
            for (int cx = 0; cx < kVolDim / kChunkDim; ++cx) {
                for (int z = 0; z < kChunkDim; ++z)
                    for (int y = 0; y < kChunkDim; ++y)
                        for (int x = 0; x < kChunkDim; ++x)
                            chunkBuf[(z * kChunkDim + y) * kChunkDim + x] =
                                voxelValue(cz * kChunkDim + z,
                                           cy * kChunkDim + y,
                                           cx * kChunkDim + x);
                ds->writeChunk(cz, cy, cx, chunkBuf.data(), chunkBuf.size());
            }
    return root / "0";
}

// Programmable fake fetcher for exercising ChunkCache error/missing paths.
class FakeFetcher final : public vc::render::IChunkFetcher {
public:
    vc::render::ChunkFetchStatus status = vc::render::ChunkFetchStatus::Found;
    int httpStatus = 0;
    std::string message;
    std::vector<std::byte> bytes;
    std::atomic<int> calls{0};

    vc::render::ChunkFetchResult fetch(const vc::render::ChunkKey&) override
    {
        calls.fetch_add(1);
        vc::render::ChunkFetchResult r;
        r.status = status;
        r.httpStatus = httpStatus;
        r.message = message;
        r.bytes = bytes;
        return r;
    }
};

std::shared_ptr<vc::render::ChunkCache>
makeFakeCache(const std::shared_ptr<FakeFetcher>& fetcher,
              std::array<int, 3> shape = {16, 16, 16},
              std::array<int, 3> chunk = {16, 16, 16})
{
    std::vector<vc::render::ChunkCache::LevelInfo> levels = {
        {shape, chunk, vc::render::IChunkedArray::LevelTransform{}}};
    std::vector<std::shared_ptr<vc::render::IChunkFetcher>> fetchers = {fetcher};
    return std::make_shared<vc::render::ChunkCache>(
        levels, fetchers, /*fillValue=*/0.0,
        vc::render::ChunkDtype::UInt8);
}

}

// ---------- ChunkCache constructor validation ----------------------------------

TEST_CASE("ChunkCache: constructor rejects empty levels")
{
    CHECK_THROWS(([] {
        vc::render::ChunkCache cache({}, {}, 0.0, vc::render::ChunkDtype::UInt8);
    })());
}

TEST_CASE("ChunkCache: constructor rejects level/fetcher count mismatch")
{
    std::vector<vc::render::ChunkCache::LevelInfo> levels = {
        {{16, 16, 16}, {16, 16, 16}, {}}};
    std::vector<std::shared_ptr<vc::render::IChunkFetcher>> fetchers;
    CHECK_THROWS(([&] {
        vc::render::ChunkCache cache(levels, fetchers, 0.0,
                                     vc::render::ChunkDtype::UInt8);
    })());
}

TEST_CASE("ChunkCache: constructor rejects null fetcher")
{
    std::vector<vc::render::ChunkCache::LevelInfo> levels = {
        {{16, 16, 16}, {16, 16, 16}, {}}};
    std::vector<std::shared_ptr<vc::render::IChunkFetcher>> fetchers = {nullptr};
    CHECK_THROWS(([&] {
        vc::render::ChunkCache cache(levels, fetchers, 0.0,
                                     vc::render::ChunkDtype::UInt8);
    })());
}

TEST_CASE("ChunkCache: constructor rejects negative shape dim")
{
    auto fetcher = std::make_shared<FakeFetcher>();
    std::vector<vc::render::ChunkCache::LevelInfo> levels = {
        {{-1, 16, 16}, {16, 16, 16}, {}}};
    std::vector<std::shared_ptr<vc::render::IChunkFetcher>> fetchers = {fetcher};
    CHECK_THROWS(([&] {
        vc::render::ChunkCache cache(levels, fetchers, 0.0,
                                     vc::render::ChunkDtype::UInt8);
    })());
}

TEST_CASE("ChunkCache: constructor rejects zero chunk dim")
{
    auto fetcher = std::make_shared<FakeFetcher>();
    std::vector<vc::render::ChunkCache::LevelInfo> levels = {
        {{16, 16, 16}, {16, 0, 16}, {}}};
    std::vector<std::shared_ptr<vc::render::IChunkFetcher>> fetchers = {fetcher};
    CHECK_THROWS(([&] {
        vc::render::ChunkCache cache(levels, fetchers, 0.0,
                                     vc::render::ChunkDtype::UInt8);
    })());
}

// ---------- ChunkCache metadata API + simple accessors -------------------------

TEST_CASE("ChunkCache: metadata accessors reflect construction args")
{
    auto fetcher = std::make_shared<FakeFetcher>();
    auto cache = makeFakeCache(fetcher, {32, 64, 128}, {16, 8, 4});

    CHECK(cache->numLevels() == 1);
    CHECK(cache->dtype() == vc::render::ChunkDtype::UInt8);
    CHECK(cache->fillValue() == doctest::Approx(0.0));
    CHECK(cache->shape(0) == std::array<int, 3>{32, 64, 128});
    CHECK(cache->chunkShape(0) == std::array<int, 3>{16, 8, 4});
    CHECK(cache->levelTransform(0).scaleFromLevel0[0] == doctest::Approx(1.0));
}

// ---------- Fake fetcher: error status propagation -----------------------------

TEST_CASE("ChunkCache: fetcher HttpError surfaces as Error status")
{
    auto fetcher = std::make_shared<FakeFetcher>();
    fetcher->status = vc::render::ChunkFetchStatus::HttpError;
    fetcher->httpStatus = 503;
    auto cache = makeFakeCache(fetcher);

    auto r = cache->getChunkBlocking(0, 0, 0, 0);
    CHECK(r.status == vc::render::ChunkStatus::Error);
    CHECK(r.error.find("503") != std::string::npos);
}

TEST_CASE("ChunkCache: fetcher IoError surfaces as Error status")
{
    auto fetcher = std::make_shared<FakeFetcher>();
    fetcher->status = vc::render::ChunkFetchStatus::IoError;
    auto cache = makeFakeCache(fetcher);

    auto r = cache->getChunkBlocking(0, 0, 0, 0);
    CHECK(r.status == vc::render::ChunkStatus::Error);
    CHECK_FALSE(r.error.empty());
}

TEST_CASE("ChunkCache: fetcher DecodeError surfaces as Error status")
{
    auto fetcher = std::make_shared<FakeFetcher>();
    fetcher->status = vc::render::ChunkFetchStatus::DecodeError;
    fetcher->message = "bad bytes";
    auto cache = makeFakeCache(fetcher);

    auto r = cache->getChunkBlocking(0, 0, 0, 0);
    CHECK(r.status == vc::render::ChunkStatus::Error);
    CHECK(r.error.find("bad bytes") != std::string::npos);
}

TEST_CASE("ChunkCache: fetcher Missing surfaces as AllFill status")
{
    auto fetcher = std::make_shared<FakeFetcher>();
    fetcher->status = vc::render::ChunkFetchStatus::Missing;
    auto cache = makeFakeCache(fetcher);

    auto r = cache->getChunkBlocking(0, 0, 0, 0);
    CHECK(r.status == vc::render::ChunkStatus::AllFill);
}

TEST_CASE("ChunkCache: same-key concurrent requests dedup at the fetcher")
{
    auto fetcher = std::make_shared<FakeFetcher>();
    fetcher->bytes.assign(16 * 16 * 16, std::byte{42});
    auto cache = makeFakeCache(fetcher);

    std::vector<std::future<vc::render::ChunkResult>> futures;
    for (int i = 0; i < 8; ++i)
        futures.push_back(std::async(std::launch::async,
            [&] { return cache->getChunkBlocking(0, 0, 0, 0); }));
    for (auto& f : futures) {
        auto r = f.get();
        CHECK(r.status == vc::render::ChunkStatus::Data);
    }
    CHECK(fetcher->calls.load() <= 2);
}

// ---------- Persistent cache: .empty marker round-trip ------------------------

TEST_CASE("ChunkCache: persistent .empty marker survives across cache instances")
{
    TmpDir tmp, cacheTmp;
    auto fetcher = std::make_shared<FakeFetcher>();
    fetcher->status = vc::render::ChunkFetchStatus::Missing;

    vc::render::ChunkCache::Options opts;
    opts.decodedByteCapacity = 1024 * 1024;
    opts.maxConcurrentReads = 2;
    opts.persistentCachePath = cacheTmp.path;

    {
        std::vector<vc::render::ChunkCache::LevelInfo> levels = {
            {{16, 16, 16}, {16, 16, 16}, {}}};
        std::vector<std::shared_ptr<vc::render::IChunkFetcher>> fetchers = {fetcher};
        vc::render::ChunkCache c(levels, fetchers, 0.0,
                                 vc::render::ChunkDtype::UInt8, opts);
        REQUIRE(c.getChunkBlocking(0, 0, 0, 0).status == vc::render::ChunkStatus::AllFill);
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }

    bool emptyMarkerExists = false;
    for (auto const& entry : fs::recursive_directory_iterator(cacheTmp.path))
        if (entry.is_regular_file() && entry.path().extension() == ".empty") {
            emptyMarkerExists = true;
            break;
        }
    CHECK(emptyMarkerExists);

    auto fetcher2 = std::make_shared<FakeFetcher>();
    fetcher2->status = vc::render::ChunkFetchStatus::HttpError;
    {
        std::vector<vc::render::ChunkCache::LevelInfo> levels = {
            {{16, 16, 16}, {16, 16, 16}, {}}};
        std::vector<std::shared_ptr<vc::render::IChunkFetcher>> fetchers = {fetcher2};
        vc::render::ChunkCache c2(levels, fetchers, 0.0,
                                  vc::render::ChunkDtype::UInt8, opts);
        auto r = c2.getChunkBlocking(0, 0, 0, 0);
        CHECK(r.status == vc::render::ChunkStatus::AllFill);
        CHECK(fetcher2->calls.load() == 0);
    }
}

// ---------- Coarse-to-fine paths ----------------------------------------------

TEST_CASE("ChunkedPlaneSampler::sampleCoordsCoarseToFine: walks levels coarsest to finest")
{
    TmpDir tmp;
    makeFixture(tmp.path);
    auto opened = vc::render::openLocalZarrPyramid(tmp.path);
    auto cache = vc::render::createChunkCache(std::move(opened),
                                               64ULL * 1024 * 1024, 4);

    cv::Mat_<cv::Vec3f> coords(8, 8);
    for (int y = 0; y < 8; ++y)
        for (int x = 0; x < 8; ++x)
            coords(y, x) = cv::Vec3f(float(x), float(y), 5.0f);

    cv::Mat_<uint8_t> out(8, 8, uint8_t(0));
    cv::Mat_<uint8_t> coverage(8, 8, uint8_t(0));

    auto keys = vc::render::ChunkedPlaneSampler::collectCoordsDependencies(
        *cache, 0, coords, coverage);
    cache->prefetchChunks(keys, true);

    auto stats = vc::render::ChunkedPlaneSampler::sampleCoordsCoarseToFine(
        *cache, 0, coords, out, coverage);

    CHECK(stats.coveredPixels >= 64);
    for (int y = 0; y < 8; ++y)
        for (int x = 0; x < 8; ++x)
            CHECK(out(y, x) == voxelValue(5, y, x));
}

// ---------- Parallel path (>= 128*128 px) -------------------------------------

TEST_CASE("ChunkedPlaneSampler: parallel sampling path on big output")
{
    TmpDir tmp;
    makeFixture(tmp.path);
    auto opened = vc::render::openLocalZarrPyramid(tmp.path);
    auto cache = vc::render::createChunkCache(std::move(opened),
                                               64ULL * 1024 * 1024, 4);

    constexpr int N = 200;
    cv::Mat_<uint8_t> out(N, N, uint8_t(0));
    cv::Mat_<uint8_t> coverage(N, N, uint8_t(0));
    const cv::Vec3f origin(0, 0, 5);
    const cv::Vec3f vxStep(0.16f, 0, 0);
    const cv::Vec3f vyStep(0, 0.16f, 0);

    auto keys = vc::render::ChunkedPlaneSampler::collectPlaneDependencies(
        *cache, 0, origin, vxStep, vyStep, coverage);
    cache->prefetchChunks(keys, true);

    auto stats = vc::render::ChunkedPlaneSampler::samplePlaneLevel(
        *cache, 0, origin, vxStep, vyStep, out, coverage);
    CHECK(stats.coveredPixels == N * N);
}

TEST_CASE("ChunkedPlaneSampler: parallel coords sampling on big output")
{
    TmpDir tmp;
    makeFixture(tmp.path);
    auto opened = vc::render::openLocalZarrPyramid(tmp.path);
    auto cache = vc::render::createChunkCache(std::move(opened),
                                               64ULL * 1024 * 1024, 4);

    constexpr int N = 200;
    cv::Mat_<cv::Vec3f> coords(N, N);
    for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x)
            coords(y, x) = cv::Vec3f(float(x) * 0.16f, float(y) * 0.16f, 5.0f);
    cv::Mat_<uint8_t> out(N, N, uint8_t(0));
    cv::Mat_<uint8_t> coverage(N, N, uint8_t(0));

    auto keys = vc::render::ChunkedPlaneSampler::collectCoordsDependencies(
        *cache, 0, coords, coverage);
    cache->prefetchChunks(keys, true);

    auto stats = vc::render::ChunkedPlaneSampler::sampleCoordsLevel(
        *cache, 0, coords, out, coverage);
    CHECK(stats.coveredPixels == N * N);
}

// ---------- ChunkedPlaneSampler: bad inputs ------------------------------------

TEST_CASE("ChunkedPlaneSampler: invalid level returns empty stats")
{
    TmpDir tmp;
    makeFixture(tmp.path);
    auto opened = vc::render::openLocalZarrPyramid(tmp.path);
    auto cache = vc::render::createChunkCache(std::move(opened),
                                               64ULL * 1024 * 1024, 4);

    cv::Mat_<uint8_t> out(8, 8, uint8_t(0));
    cv::Mat_<uint8_t> coverage(8, 8, uint8_t(0));
    auto stats = vc::render::ChunkedPlaneSampler::samplePlaneLevel(
        *cache, /*level=*/-1,
        cv::Vec3f(0, 0, 5), cv::Vec3f(1, 0, 0), cv::Vec3f(0, 1, 0),
        out, coverage);
    CHECK(stats.coveredPixels == 0);

    auto stats2 = vc::render::ChunkedPlaneSampler::samplePlaneLevel(
        *cache, /*level=*/99,
        cv::Vec3f(0, 0, 5), cv::Vec3f(1, 0, 0), cv::Vec3f(0, 1, 0),
        out, coverage);
    CHECK(stats2.coveredPixels == 0);
}

TEST_CASE("ChunkedPlaneSampler: empty coords matrix returns empty stats")
{
    TmpDir tmp;
    makeFixture(tmp.path);
    auto opened = vc::render::openLocalZarrPyramid(tmp.path);
    auto cache = vc::render::createChunkCache(std::move(opened),
                                               64ULL * 1024 * 1024, 4);

    cv::Mat_<cv::Vec3f> coords;
    cv::Mat_<uint8_t> out;
    cv::Mat_<uint8_t> coverage;
    auto stats = vc::render::ChunkedPlaneSampler::sampleCoordsLevel(
        *cache, 0, coords, out, coverage);
    CHECK(stats.coveredPixels == 0);
}

// ---------- VcDataset: re-open existing dataset --------------------------------

TEST_CASE("VcDataset: reopen path with existing .zarray")
{
    TmpDir tmp;
    auto ds1 = vc::createZarrDataset(
        tmp.path, "0", {16, 16, 16}, {16, 16, 16},
        vc::VcDtype::uint8, "blosc");
    std::vector<uint8_t> in(16 * 16 * 16, 7);
    REQUIRE(ds1->writeChunk(0, 0, 0, in.data(), in.size()));
    ds1.reset();

    vc::VcDataset reopened(tmp.path / "0");
    CHECK(reopened.shape() == std::vector<size_t>{16, 16, 16});
    std::vector<uint8_t> out(16 * 16 * 16, 0);
    REQUIRE(reopened.readChunk(0, 0, 0, out.data()));
    for (auto v : out) CHECK(v == 7);
}

TEST_CASE("VcDataset: zarr v2 dimension separator '/' (nested chunk paths)")
{
    TmpDir tmp;
    auto ds = vc::createZarrDataset(
        tmp.path, "0", {16, 16, 16}, {16, 16, 16},
        vc::VcDtype::uint8, "blosc", "/", 0);
    std::vector<uint8_t> in(16 * 16 * 16, 11);
    REQUIRE(ds->writeChunk(0, 0, 0, in.data(), in.size()));
    std::vector<uint8_t> out(16 * 16 * 16, 0);
    REQUIRE(ds->readChunk(0, 0, 0, out.data()));
    for (auto v : out) CHECK(v == 11);
}

// ---------- ZarrChunkFetcher edge -------------------------------------------------

TEST_CASE("openLocalZarrPyramid: directory with .zgroup but no levels falls back to root array")
{
    TmpDir tmp;
    auto ds = vc::createZarrDataset(
        tmp.path, "data", {16, 16, 16}, {16, 16, 16},
        vc::VcDtype::uint8, "blosc");
    std::vector<uint8_t> in(16 * 16 * 16, 9);
    REQUIRE(ds->writeChunk(0, 0, 0, in.data(), in.size()));

    // openLocalZarrPyramid skips non-numeric subdirs, so this fixture has no
    // numeric level dirs. With the "data" subdir as the array root, opening
    // the parent should not find a pyramid — opening the array dir directly
    // does work.
    auto opened = vc::render::openLocalZarrPyramid(tmp.path / "data");
    CHECK(opened.fetchers.size() == 1);
}

TEST_CASE("ChunkCache::prefetchChunks: invalid keys are skipped")
{
    TmpDir tmp;
    makeFixture(tmp.path);
    auto opened = vc::render::openLocalZarrPyramid(tmp.path);
    auto cache = vc::render::createChunkCache(std::move(opened),
                                               64ULL * 1024 * 1024, 4);

    std::vector<vc::render::ChunkKey> mixed = {
        {0, 0, 0, 0},
        {0, 99, 99, 99},
        {99, 0, 0, 0},
    };
    cache->prefetchChunks(mixed, /*wait=*/true);
    auto r = cache->getChunkBlocking(0, 0, 0, 0);
    CHECK(r.status == vc::render::ChunkStatus::Data);
}
