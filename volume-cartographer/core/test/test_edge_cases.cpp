#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/render/ChunkCache.hpp"
#include "vc/core/render/ChunkedPlaneSampler.hpp"
#include "vc/core/render/ZarrChunkFetcher.hpp"
#include "vc/core/types/VcDataset.hpp"
#include "utils/Json.hpp"

#include <opencv2/core.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <future>
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
               ("vc_edge_test_" + std::to_string(rng()));
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

void fillFixture(vc::VcDataset& ds,
                 int volZ, int volY, int volX,
                 int chunkZ, int chunkY, int chunkX)
{
    const int nz = (volZ + chunkZ - 1) / chunkZ;
    const int ny = (volY + chunkY - 1) / chunkY;
    const int nx = (volX + chunkX - 1) / chunkX;
    std::vector<uint8_t> buf(static_cast<size_t>(chunkZ) * chunkY * chunkX);
    for (int cz = 0; cz < nz; ++cz) {
        for (int cy = 0; cy < ny; ++cy) {
            for (int cx = 0; cx < nx; ++cx) {
                for (int z = 0; z < chunkZ; ++z) {
                    for (int y = 0; y < chunkY; ++y) {
                        for (int x = 0; x < chunkX; ++x) {
                            buf[(z * chunkY + y) * chunkX + x] =
                                voxelValue(cz * chunkZ + z, cy * chunkY + y, cx * chunkX + x);
                        }
                    }
                }
                ds.writeChunk(static_cast<size_t>(cz),
                              static_cast<size_t>(cy),
                              static_cast<size_t>(cx),
                              buf.data(), buf.size());
            }
        }
    }
}

fs::path makeFixture(const fs::path& root,
                     int volDim = 32, int chunkDim = 16,
                     std::int64_t fillValue = 0)
{
    auto ds = vc::createZarrDataset(
        root, "0",
        {static_cast<size_t>(volDim), static_cast<size_t>(volDim), static_cast<size_t>(volDim)},
        {static_cast<size_t>(chunkDim), static_cast<size_t>(chunkDim), static_cast<size_t>(chunkDim)},
        vc::VcDtype::uint8, "blosc", ".", fillValue);
    fillFixture(*ds, volDim, volDim, volDim, chunkDim, chunkDim, chunkDim);
    return root / "0";
}

std::shared_ptr<vc::render::ChunkCache>
makeCache(const fs::path& root, std::size_t cap = 64ULL * 1024 * 1024)
{
    auto opened = vc::render::openLocalZarrPyramid(root);
    return std::shared_ptr<vc::render::ChunkCache>(
        vc::render::createChunkCache(std::move(opened), cap, 4).release());
}

}

// ---------- VcDataset edge cases -----------------------------------------------

TEST_CASE("VcDataset: shape not divisible by chunk shape (partial trailing chunk)")
{
    TmpDir tmp;
    auto ds = vc::createZarrDataset(
        tmp.path, "0", {20, 20, 20}, {16, 16, 16},
        vc::VcDtype::uint8, "blosc", ".", 99);

    // The last chunk along each axis is partial — written chunk is a full 16^3
    // even though only 4 voxels along the trailing edge are "live".
    std::vector<uint8_t> chunkBuf(16 * 16 * 16, 7);
    REQUIRE(ds->writeChunk(0, 0, 0, chunkBuf.data(), chunkBuf.size()));
    REQUIRE(ds->writeChunk(1, 1, 1, chunkBuf.data(), chunkBuf.size()));

    // readRegion that crosses into the partial trailing chunk should not crash.
    std::vector<uint8_t> region(8 * 8 * 8, 0xAB);
    REQUIRE(ds->readRegion({12, 12, 12}, {8, 8, 8}, region.data()));
    int liveCount = 0, fillCount = 0;
    for (auto v : region) {
        if (v == 7) ++liveCount;
        else if (v == 99) ++fillCount;
    }
    CHECK((liveCount + fillCount) == int(region.size()));
    CHECK(liveCount > 0);
    CHECK(fillCount > 0);
}

TEST_CASE("VcDataset: readRegion entirely outside volume materializes fill")
{
    TmpDir tmp;
    auto ds = vc::createZarrDataset(
        tmp.path, "0", {16, 16, 16}, {16, 16, 16},
        vc::VcDtype::uint8, "blosc", ".", 55);

    std::vector<uint8_t> region(4 * 4 * 4, 0xAB);
    const bool ok = ds->readRegion({100, 100, 100}, {4, 4, 4}, region.data());
    CHECK(ok);
    for (auto v : region) CHECK(v == 55);
}

TEST_CASE("VcDataset: readChunkOrFill across all chunks of empty dataset")
{
    TmpDir tmp;
    auto ds = vc::createZarrDataset(
        tmp.path, "0", {16, 16, 16}, {8, 8, 8},
        vc::VcDtype::uint8, "blosc", ".", 33);

    std::vector<uint8_t> buf(8 * 8 * 8, 0);
    for (size_t cz = 0; cz < 2; ++cz)
        for (size_t cy = 0; cy < 2; ++cy)
            for (size_t cx = 0; cx < 2; ++cx) {
                std::fill(buf.begin(), buf.end(), 0xAB);
                CHECK_FALSE(ds->readChunkOrFill(cz, cy, cx, buf.data()));
                for (auto v : buf) CHECK(v == 33);
            }
}

// ---------- ChunkedPlaneSampler boundary tests ---------------------------------

TEST_CASE("ChunkedPlaneSampler: origin exactly on a chunk boundary samples cleanly")
{
    TmpDir tmp;
    makeFixture(tmp.path);
    auto cache = makeCache(tmp.path);

    cv::Mat_<uint8_t> out(8, 8, uint8_t(0));
    cv::Mat_<uint8_t> coverage(8, 8, uint8_t(0));

    // origin x=16 is exactly the boundary between chunk (0,0,0) and (0,0,1)
    const cv::Vec3f origin(16.0f, 16.0f, 16.0f);
    const cv::Vec3f vxStep(1.0f, 0.0f, 0.0f);
    const cv::Vec3f vyStep(0.0f, 1.0f, 0.0f);

    auto keys = vc::render::ChunkedPlaneSampler::collectPlaneDependencies(
        *cache, 0, origin, vxStep, vyStep, coverage);
    cache->prefetchChunks(keys, true);

    auto stats = vc::render::ChunkedPlaneSampler::samplePlaneLevel(
        *cache, 0, origin, vxStep, vyStep, out, coverage);
    CHECK(stats.coveredPixels == 64);
    CHECK(out(0, 0) == voxelValue(16, 16, 16));
}

TEST_CASE("ChunkedPlaneSampler: sampling at the last in-bounds voxel does not overflow")
{
    TmpDir tmp;
    makeFixture(tmp.path);
    auto cache = makeCache(tmp.path);

    cv::Mat_<uint8_t> out(1, 1, uint8_t(0));
    cv::Mat_<uint8_t> coverage(1, 1, uint8_t(0));
    const cv::Vec3f origin(31.0f, 31.0f, 31.0f);
    const cv::Vec3f vxStep(1.0f, 0.0f, 0.0f);
    const cv::Vec3f vyStep(0.0f, 1.0f, 0.0f);

    auto keys = vc::render::ChunkedPlaneSampler::collectPlaneDependencies(
        *cache, 0, origin, vxStep, vyStep, coverage);
    cache->prefetchChunks(keys, true);

    auto stats = vc::render::ChunkedPlaneSampler::samplePlaneLevel(
        *cache, 0, origin, vxStep, vyStep, out, coverage);
    CHECK(stats.coveredPixels == 1);
    CHECK(out(0, 0) == voxelValue(31, 31, 31));
}

TEST_CASE("ChunkedPlaneSampler: negative origin coordinates fill with fill-value")
{
    TmpDir tmp;
    makeFixture(tmp.path);
    auto cache = makeCache(tmp.path);

    cv::Mat_<uint8_t> out(8, 8, uint8_t(99));
    cv::Mat_<uint8_t> coverage(8, 8, uint8_t(0));
    const cv::Vec3f origin(-100.0f, -100.0f, -100.0f);
    const cv::Vec3f vxStep(1.0f, 0.0f, 0.0f);
    const cv::Vec3f vyStep(0.0f, 1.0f, 0.0f);

    auto stats = vc::render::ChunkedPlaneSampler::samplePlaneLevel(
        *cache, 0, origin, vxStep, vyStep, out, coverage);
    CHECK(stats.coveredPixels == 64);
    for (int y = 0; y < 8; ++y)
        for (int x = 0; x < 8; ++x)
            CHECK(out(y, x) == 0);
}

TEST_CASE("ChunkedPlaneSampler: degenerate zero-step plane samples the same voxel everywhere")
{
    TmpDir tmp;
    makeFixture(tmp.path);
    auto cache = makeCache(tmp.path);

    cv::Mat_<uint8_t> out(8, 8, uint8_t(0));
    cv::Mat_<uint8_t> coverage(8, 8, uint8_t(0));
    const cv::Vec3f origin(7.0f, 7.0f, 7.0f);
    const cv::Vec3f vxStep(0.0f, 0.0f, 0.0f);
    const cv::Vec3f vyStep(0.0f, 0.0f, 0.0f);

    auto keys = vc::render::ChunkedPlaneSampler::collectPlaneDependencies(
        *cache, 0, origin, vxStep, vyStep, coverage);
    cache->prefetchChunks(keys, true);

    auto stats = vc::render::ChunkedPlaneSampler::samplePlaneLevel(
        *cache, 0, origin, vxStep, vyStep, out, coverage);
    CHECK(stats.coveredPixels == 64);
    for (int y = 0; y < 8; ++y)
        for (int x = 0; x < 8; ++x)
            CHECK(out(y, x) == voxelValue(7, 7, 7));
}

TEST_CASE("ChunkedPlaneSampler: pre-covered pixels are not overwritten")
{
    TmpDir tmp;
    makeFixture(tmp.path);
    auto cache = makeCache(tmp.path);

    cv::Mat_<uint8_t> out(8, 8, uint8_t(0));
    cv::Mat_<uint8_t> coverage(8, 8, uint8_t(0));
    out(0, 0) = 123;
    coverage(0, 0) = 1;

    const cv::Vec3f origin(0.0f, 0.0f, 5.0f);
    const cv::Vec3f vxStep(1.0f, 0.0f, 0.0f);
    const cv::Vec3f vyStep(0.0f, 1.0f, 0.0f);

    auto keys = vc::render::ChunkedPlaneSampler::collectPlaneDependencies(
        *cache, 0, origin, vxStep, vyStep, coverage);
    cache->prefetchChunks(keys, true);
    vc::render::ChunkedPlaneSampler::samplePlaneLevel(
        *cache, 0, origin, vxStep, vyStep, out, coverage);

    CHECK(out(0, 0) == 123);
    CHECK(out(1, 0) == voxelValue(5, 1, 0));
    CHECK(out(0, 1) == voxelValue(5, 0, 1));
}

TEST_CASE("ChunkedPlaneSampler: large 256x256 output covers many tiles")
{
    TmpDir tmp;
    makeFixture(tmp.path);
    auto cache = makeCache(tmp.path);

    cv::Mat_<uint8_t> out(256, 256, uint8_t(0));
    cv::Mat_<uint8_t> coverage(256, 256, uint8_t(0));
    const cv::Vec3f origin(0.0f, 0.0f, 5.0f);
    const cv::Vec3f vxStep(0.125f, 0.0f, 0.0f);
    const cv::Vec3f vyStep(0.0f, 0.125f, 0.0f);

    auto keys = vc::render::ChunkedPlaneSampler::collectPlaneDependencies(
        *cache, 0, origin, vxStep, vyStep, coverage);
    cache->prefetchChunks(keys, true);
    auto stats = vc::render::ChunkedPlaneSampler::samplePlaneLevel(
        *cache, 0, origin, vxStep, vyStep, out, coverage);

    CHECK(stats.coveredPixels == 256 * 256);
    CHECK(out(0, 0) == voxelValue(5, 0, 0));
    const int clampedXY = std::min(31, int(255 * 0.125f + 0.5f));
    CHECK(out(255, 255) == voxelValue(5, clampedXY, clampedXY));
}

// ---------- ChunkCache concurrency / lifetime ----------------------------------

TEST_CASE("ChunkCache: concurrent getChunkBlocking on same key returns identical bytes")
{
    TmpDir tmp;
    makeFixture(tmp.path);
    auto cache = makeCache(tmp.path);

    constexpr int kThreads = 16;
    std::vector<std::future<vc::render::ChunkResult>> futures;
    for (int i = 0; i < kThreads; ++i) {
        futures.push_back(std::async(std::launch::async,
            [&] { return cache->getChunkBlocking(0, 0, 0, 0); }));
    }
    std::shared_ptr<const std::vector<std::byte>> first;
    for (auto& f : futures) {
        auto r = f.get();
        REQUIRE(r.status == vc::render::ChunkStatus::Data);
        REQUIRE(r.bytes);
        if (!first) first = r.bytes;
        CHECK(r.bytes->size() == first->size());
        CHECK(r.bytes.get() == first.get());
    }
}

TEST_CASE("ChunkCache: invalidate during in-flight async fetch does not crash")
{
    TmpDir tmp;
    makeFixture(tmp.path);
    auto cache = makeCache(tmp.path);

    std::atomic<int> notifications{0};
    auto id = cache->addChunkReadyListener([&] { notifications.fetch_add(1); });

    for (int cz = 0; cz < 2; ++cz)
        for (int cy = 0; cy < 2; ++cy)
            for (int cx = 0; cx < 2; ++cx)
                cache->tryGetChunk(0, cz, cy, cx);
    cache->invalidate();

    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(20));

    auto r = cache->getChunkBlocking(0, 0, 0, 0);
    CHECK(r.status == vc::render::ChunkStatus::Data);
    cache->removeChunkReadyListener(id);
}

TEST_CASE("ChunkCache: destroying the cache while fetches are in flight is safe")
{
    TmpDir tmp;
    makeFixture(tmp.path);
    {
        auto cache = makeCache(tmp.path);
        for (int cz = 0; cz < 2; ++cz)
            for (int cy = 0; cy < 2; ++cy)
                for (int cx = 0; cx < 2; ++cx)
                    cache->tryGetChunk(0, cz, cy, cx);
    }
    CHECK(true);
}

TEST_CASE("ChunkCache: removed listener does not fire on subsequent resolutions")
{
    TmpDir tmp;
    makeFixture(tmp.path);
    auto cache = makeCache(tmp.path);

    std::atomic<int> calls{0};
    auto id = cache->addChunkReadyListener([&] { calls.fetch_add(1); });

    cache->tryGetChunk(0, 0, 0, 0);
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (cache->tryGetChunk(0, 0, 0, 0).status != vc::render::ChunkStatus::Data &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    const int beforeRemove = calls.load();
    REQUIRE(beforeRemove >= 1);

    cache->removeChunkReadyListener(id);
    cache->invalidate();
    cache->tryGetChunk(0, 1, 1, 1);
    deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (cache->tryGetChunk(0, 1, 1, 1).status != vc::render::ChunkStatus::Data &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    CHECK(calls.load() == beforeRemove);
}

// ---------- Json edge cases ----------------------------------------------------

TEST_CASE("Json: get_int on a string throws")
{
    auto j = utils::Json("hello");
    CHECK_THROWS((void)j.get_int());
}

TEST_CASE("Json: get_string on a number throws")
{
    auto j = utils::Json(42);
    CHECK_THROWS((void)j.get_string());
}

TEST_CASE("Json: parse rejects malformed input")
{
    CHECK_THROWS(utils::Json::parse("{not_json"));
    CHECK_THROWS(utils::Json::parse("[1,2,"));
}

TEST_CASE("Json: deeply nested object survives parse + dump")
{
    std::string nested;
    for (int i = 0; i < 50; ++i) nested += "{\"a\":";
    nested += "1";
    for (int i = 0; i < 50; ++i) nested += "}";

    auto j = utils::Json::parse(nested);
    auto dumped = j.dump();
    auto j2 = utils::Json::parse(dumped);

    auto* p = &j2;
    for (int i = 0; i < 50; ++i) {
        REQUIRE(p->is_object());
        REQUIRE(p->contains("a"));
        p = &(*p)["a"];
    }
    CHECK(p->get_int() == 1);
}

TEST_CASE("Json: empty array and empty object dump correctly")
{
    CHECK(utils::Json::array().dump() == "[]");
    CHECK(utils::Json::object().dump() == "{}");
}

TEST_CASE("Json: integer overflow in parse still round-trips numerically")
{
    auto j = utils::Json::parse("9223372036854775807");
    CHECK(j.get_int64() == 9223372036854775807LL);
    auto j2 = utils::Json::parse(j.dump());
    CHECK(j2.get_int64() == 9223372036854775807LL);
}

TEST_CASE("Json: string with embedded special characters round-trips")
{
    auto original = utils::Json(std::string("line1\nline2\t\"quoted\"\\back"));
    auto reparsed = utils::Json::parse(original.dump());
    CHECK(reparsed.get_string() == original.get_string());
}

// ---------- More aggressive bug-hunting tests ----------------------------------

TEST_CASE("ChunkCache: getChunkBlocking with out-of-bounds chunk indices")
{
    TmpDir tmp;
    makeFixture(tmp.path);
    auto cache = makeCache(tmp.path);

    // Volume is 32^3 with 16^3 chunks → valid chunk indices are 0..1 per axis.
    auto r = cache->getChunkBlocking(0, 99, 0, 0);
    CHECK((r.status == vc::render::ChunkStatus::Error ||
           r.status == vc::render::ChunkStatus::AllFill));
}

TEST_CASE("ChunkCache: getChunkBlocking with negative-ish (very large) indices")
{
    TmpDir tmp;
    makeFixture(tmp.path);
    auto cache = makeCache(tmp.path);

    auto r = cache->getChunkBlocking(0, 0, -1, 0);
    CHECK((r.status == vc::render::ChunkStatus::Error ||
           r.status == vc::render::ChunkStatus::AllFill));
}

TEST_CASE("ChunkCache: getChunkBlocking with bad level returns non-Data status without crashing")
{
    TmpDir tmp;
    makeFixture(tmp.path);
    auto cache = makeCache(tmp.path);

    auto r = cache->getChunkBlocking(99, 0, 0, 0);
    CHECK(r.status != vc::render::ChunkStatus::Data);
}

TEST_CASE("ChunkedPlaneSampler: NaN coordinates do not propagate to output")
{
    TmpDir tmp;
    makeFixture(tmp.path);
    auto cache = makeCache(tmp.path);

    cv::Mat_<cv::Vec3f> coords(4, 4);
    for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 4; ++x)
            coords(y, x) = cv::Vec3f(float(x), float(y), 5.0f);
    coords(2, 2) = cv::Vec3f(std::numeric_limits<float>::quiet_NaN(),
                             0.0f, 0.0f);

    cv::Mat_<uint8_t> out(4, 4, uint8_t(0));
    cv::Mat_<uint8_t> coverage(4, 4, uint8_t(0));

    auto keys = vc::render::ChunkedPlaneSampler::collectCoordsDependencies(
        *cache, 0, coords, coverage);
    cache->prefetchChunks(keys, true);
    vc::render::ChunkedPlaneSampler::sampleCoordsLevel(
        *cache, 0, coords, out, coverage);

    for (int py = 0; py < 4; ++py)
        for (int px = 0; px < 4; ++px) {
            if (px == 2 && py == 2) continue;
            CHECK(out(py, px) == voxelValue(5, py, px));
        }
}

TEST_CASE("ChunkedPlaneSampler: infinite coordinates do not crash")
{
    TmpDir tmp;
    makeFixture(tmp.path);
    auto cache = makeCache(tmp.path);

    cv::Mat_<cv::Vec3f> coords(2, 2);
    coords(0, 0) = cv::Vec3f(std::numeric_limits<float>::infinity(), 0, 0);
    coords(0, 1) = cv::Vec3f(-std::numeric_limits<float>::infinity(), 0, 0);
    coords(1, 0) = cv::Vec3f(0, std::numeric_limits<float>::infinity(), 0);
    coords(1, 1) = cv::Vec3f(0, 0, std::numeric_limits<float>::infinity());

    cv::Mat_<uint8_t> out(2, 2, uint8_t(0));
    cv::Mat_<uint8_t> coverage(2, 2, uint8_t(0));

    vc::render::ChunkedPlaneSampler::sampleCoordsLevel(
        *cache, 0, coords, out, coverage);
    CHECK(true);
}

TEST_CASE("VcDataset: writeChunk rejects size mismatch")
{
    TmpDir tmp;
    auto ds = vc::createZarrDataset(
        tmp.path, "0", {16, 16, 16}, {16, 16, 16}, vc::VcDtype::uint8, "blosc");

    std::vector<uint8_t> tooSmall(8, 0);
    CHECK(ds->writeChunk(0, 0, 0, tooSmall.data(), tooSmall.size()) == false);
}

TEST_CASE("VcDataset: chunkExists is correctly false on fresh dataset")
{
    TmpDir tmp;
    auto ds = vc::createZarrDataset(
        tmp.path, "0", {16, 16, 16}, {16, 16, 16}, vc::VcDtype::uint8, "blosc");

    CHECK_FALSE(ds->chunkExists(0, 0, 0));
    CHECK_FALSE(ds->chunkExists(99, 99, 99));
}

TEST_CASE("VcDataset: removeChunk on non-existent chunk is a no-op (false)")
{
    TmpDir tmp;
    auto ds = vc::createZarrDataset(
        tmp.path, "0", {16, 16, 16}, {16, 16, 16}, vc::VcDtype::uint8, "blosc");

    CHECK_FALSE(ds->removeChunk(0, 0, 0));
}

TEST_CASE("Json: array out-of-bounds access via at() throws, operator[] is UB-safe")
{
    auto a = utils::Json::array();
    a.push_back(utils::Json(1));
    a.push_back(utils::Json(2));
    CHECK_THROWS((void)a.at(99));
}

TEST_CASE("Json: at() on object with missing key throws")
{
    auto j = utils::Json::object();
    j["a"] = 1;
    CHECK_THROWS((void)j.at(std::string("missing")));
}

TEST_CASE("Json: NaN and infinity float serialization")
{
    auto j = utils::Json(std::numeric_limits<double>::quiet_NaN());
    auto dumped = j.dump();
    INFO("NaN serialized as: " << dumped);
    CHECK(true);
}

TEST_CASE("ZarrChunkFetcher: invalid path throws on open")
{
    fs::path bogus = "/nonexistent/path/that/does/not/exist";
    CHECK_THROWS((void)vc::render::openLocalZarrPyramid(bogus));
}

TEST_CASE("ZarrChunkFetcher: empty directory throws on open")
{
    TmpDir tmp;
    CHECK_THROWS((void)vc::render::openLocalZarrPyramid(tmp.path));
}

TEST_CASE("ChunkCache: prefetchChunks with an empty key list is a no-op")
{
    TmpDir tmp;
    makeFixture(tmp.path);
    auto cache = makeCache(tmp.path);

    cache->prefetchChunks({}, /*wait=*/true);
    auto r = cache->getChunkBlocking(0, 0, 0, 0);
    CHECK(r.status == vc::render::ChunkStatus::Data);
}

TEST_CASE("ChunkedPlaneSampler: zero-area output (0x0) does not crash")
{
    TmpDir tmp;
    makeFixture(tmp.path);
    auto cache = makeCache(tmp.path);

    cv::Mat_<uint8_t> out;
    cv::Mat_<uint8_t> coverage;
    const cv::Vec3f origin(5, 5, 5);
    const cv::Vec3f vxStep(1, 0, 0);
    const cv::Vec3f vyStep(0, 1, 0);

    auto stats = vc::render::ChunkedPlaneSampler::samplePlaneLevel(
        *cache, 0, origin, vxStep, vyStep, out, coverage);
    CHECK(stats.coveredPixels == 0);
}

TEST_CASE("VcDataset: tiny 1x1x1 volume with 1x1x1 chunk")
{
    TmpDir tmp;
    auto ds = vc::createZarrDataset(
        tmp.path, "0", {1, 1, 1}, {1, 1, 1}, vc::VcDtype::uint8, "blosc", ".", 7);

    uint8_t in = 250;
    REQUIRE(ds->writeChunk(0, 0, 0, &in, 1));
    uint8_t out = 0;
    REQUIRE(ds->readChunk(0, 0, 0, &out));
    CHECK(out == 250);
}
