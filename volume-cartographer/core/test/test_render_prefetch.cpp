#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "../../apps/src/RenderPrefetch.hpp"
#include "vc/core/util/Compositing.hpp"
#include "vc/core/render/ChunkCache.hpp"

#include <atomic>
#include <cstring>
#include <limits>
#include <mutex>

namespace {
using Keys = std::unordered_set<vc::render::ChunkKey, vc::render::ChunkKeyHash>;

// Record actual blocking reads made by Slicing.cpp, independently of its
// opportunistic prefetch requests. Constant chunks avoid large fixture payloads.
class RecordingArray : public vc::render::IChunkedArray {
public:
    std::array<int, 3> dimensions{256, 256, 256};
    std::array<int, 3> chunks{128, 128, 128};
    Keys reads;
    std::atomic<int> prefetchCalls{0};
    int numLevels() const override { return 1; }
    std::array<int, 3> shape(int) const override { return dimensions; }
    std::array<int, 3> chunkShape(int) const override { return chunks; }
    vc::render::ChunkDtype dtype() const override { return vc::render::ChunkDtype::UInt8; }
    double fillValue() const override { return 0; }
    LevelTransform levelTransform(int) const override { return {}; }
    vc::render::ChunkResult tryGetChunk(int level, int z, int y, int x) override
    {
        std::lock_guard lock(mutex_);
        reads.insert({level, z, y, x});
        vc::render::ChunkResult result;
        result.status = vc::render::ChunkStatus::AllFill;
        result.shape = chunks;
        return result;
    }
    vc::render::ChunkResult getChunkBlocking(int level, int z, int y, int x) override
    { return tryGetChunk(level, z, y, x); }
    void prefetchChunks(const std::vector<vc::render::ChunkKey>&, bool, int) override
    { ++prefetchCalls; }
    ChunkReadyCallbackId addChunkReadyListener(ChunkReadyCallback) override { return 0; }
    void removeChunkReadyListener(ChunkReadyCallbackId) override {}
private:
    std::mutex mutex_;
};

Keys plan(RecordingArray& array, const cv::Mat_<cv::Vec3f>& points,
          const cv::Mat_<cv::Vec3f>& dirs, const std::vector<float>& offsets, bool composite)
{
    Keys keys;
    vc::render::prefetch::insertExactChunksForSamples(
        points, dirs, offsets, &array, 0,
        vc::render::prefetch::samplingForRender(composite), keys);
    return keys;
}
}

TEST_CASE("ordinary prefetch covers trilinear neighbours across a chunk boundary")
{
    RecordingArray array;
    cv::Mat_<cv::Vec3f> points(1, 1, cv::Vec3f{127.25f, 64.25f, 64.25f});
    cv::Mat_<cv::Vec3f> dirs(1, 1, cv::Vec3f{0, 0, 0});
    const auto keys = plan(array, points, dirs, {0}, false);
    std::vector<cv::Mat_<uint8_t>> output;
    readMultiSlice(output, &array, 0, points, dirs, {0});
    CHECK(keys.size() == 2);
    CHECK(keys == array.reads);
    array.reads.clear();
    sampleTileSlices(output, &array, 0, points, dirs, {0});
    CHECK(keys == array.reads);
}

TEST_CASE("planner and ordinary renderer round displaced positions in float")
{
    RecordingArray array;
    cv::Mat_<cv::Vec3f> points(1, 1, cv::Vec3f{127, 64, 64});
    cv::Mat_<cv::Vec3f> dirs(1, 1, cv::Vec3f{0.999999f, 0, 0});
    const auto keys = plan(array, points, dirs, {1}, false);
    std::vector<cv::Mat_<uint8_t>> output;
    readMultiSlice(output, &array, 0, points, dirs, {1});
    CHECK(keys.size() == 1);
    CHECK(keys == array.reads);
}

TEST_CASE("composite prefetch matches nearest rounding and layer offsets")
{
    RecordingArray array;
    array.dimensions = {4, 4, 4};
    array.chunks = {1, 1, 1};
    cv::Mat_<cv::Vec3f> points(1, 1, cv::Vec3f{std::nextafter(0.5f, 0.f), 1, 1});
    cv::Mat_<cv::Vec3f> dirs(1, 1, cv::Vec3f{0, 0, 1});
    const auto keys = plan(array, points, dirs, {-1, 0, 1}, true);
    cv::Mat_<uint8_t> output(1, 1, uint8_t{0});
    CompositeParams params;
    params.method = "mean";
    // Default composite sampling is deliberately used as an independent oracle.
    readCompositeFast(output, &array, 0, points, dirs, 1, -1, 1, params);
    CHECK(keys == array.reads);
    CHECK(keys.size() == 3);
}

TEST_CASE("ordinary planner matches reads around all chunk and volume faces")
{
    const float coords[] = {0, 0.25f, 127, 127.25f, 127.75f, 128,
                            std::nextafter(128.f, 0.f), 255.75f};
    RecordingArray array;
    for (int axis = 0; axis < 3; ++axis) {
        for (float value : coords) {
            cv::Vec3f point{127.25f, 127.25f, 127.25f};
            point[axis] = value;
            cv::Mat_<cv::Vec3f> points(1, 1, point);
            cv::Mat_<cv::Vec3f> dirs(1, 1, cv::Vec3f{0.25f, -0.5f, 1});
            const auto keys = plan(array, points, dirs, {-1, 0, 1}, false);
            array.reads.clear();
            std::vector<cv::Mat_<uint8_t>> output;
            readMultiSlice(output, &array, 0, points, dirs, {-1, 0, 1});
            CHECK(keys == array.reads);
        }
    }
}

TEST_CASE("nonfinite and outside samples produce no prefetch requests")
{
    RecordingArray array;
    for (float value : {-1.f, 256.f, std::numeric_limits<float>::max(),
                        std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::quiet_NaN()}) {
        cv::Mat_<cv::Vec3f> points(1, 1, cv::Vec3f{value, 64, 64});
        cv::Mat_<cv::Vec3f> dirs(1, 1, cv::Vec3f{0, 0, 0});
        CHECK(plan(array, points, dirs, {0}, false).empty());
    }
}

TEST_CASE("sparse samples in huge volumes do not allocate a volume sized bitmap")
{
    RecordingArray array;
    array.dimensions = {1000000000, 1000000000, 1000000000};
    array.chunks = {1, 1, 1};
    cv::Mat_<cv::Vec3f> points(1, 2);
    points(0, 0) = {1, 1, 1};
    points(0, 1) = {999999936.f, 999999936.f, 999999936.f};
    cv::Mat_<cv::Vec3f> dirs(1, 2, cv::Vec3f{0, 0, 0});
    const auto keys = plan(array, points, dirs, {0}, false);
    CHECK(keys.size() == 16);
    CHECK(keys.contains({0, 1, 1, 1}));
    CHECK(keys.contains({0, 999999937, 999999937, 999999937}));
}

TEST_CASE("prefetched view suppresses speculative calls without changing source defaults")
{
    RecordingArray array;
    cv::Mat_<cv::Vec3f> points(1, 1, cv::Vec3f{127.25f, 64.25f, 64.25f});
    cv::Mat_<cv::Vec3f> dirs(1, 1, cv::Vec3f{0, 0, 1});
    std::vector<cv::Mat_<uint8_t>> output;
    readMultiSlice(output, &array, 0, points, dirs, {0});
    CHECK(array.prefetchCalls.load() > 0);
    const int beforeView = array.prefetchCalls.load();
    {
        vc::render::prefetch::PrefetchedArrayView view(array);
        CHECK(view.shape(0) == array.shape(0));
        CHECK(view.chunkShape(0) == array.chunkShape(0));
        CHECK(view.dtype() == array.dtype());
        array.reads.clear();
        readMultiSlice(output, &view, 0, points, dirs, {0});
        CHECK(array.reads == plan(array, points, dirs, {0}, false));
        CHECK(array.prefetchCalls.load() == beforeView);
        array.reads.clear();
        cv::Mat_<uint8_t> composite(1, 1, uint8_t{0});
        CompositeParams params;
        readCompositeFast(composite, &view, 0, points, dirs, 1, 0, 0, params);
        CHECK(array.reads == plan(array, points, dirs, {0}, true));
        CHECK(array.prefetchCalls.load() == beforeView);
    }
    readMultiSlice(output, &array, 0, points, dirs, {0});
    CHECK(array.prefetchCalls.load() > beforeView);
}

namespace {
class CountingPayloadFetcher : public vc::render::IChunkFetcher {
public:
    bool fail = false;
    std::vector<std::byte> payload = std::vector<std::byte>(64, std::byte{17});
    std::atomic<int> calls{0};
    vc::render::ChunkFetchResult fetch(const vc::render::ChunkKey&) override
    {
        ++calls;
        vc::render::ChunkFetchResult result;
        if (fail) {
            result.status = vc::render::ChunkFetchStatus::HttpError;
            result.httpStatus = 503;
            result.message = "expected source error";
        } else {
            result.status = vc::render::ChunkFetchStatus::Found;
            result.bytes = payload;
        }
        return result;
    }
};
}

TEST_CASE("prefetched view still fetches sampled chunks evicted by a small RAM budget")
{
    auto fetcher = std::make_shared<CountingPayloadFetcher>();
    vc::render::ChunkCache::Options options;
    options.detectAllFillChunks = false;
    vc::render::ChunkCacheService::Options service;
    service.decodedByteCapacity = 64; // one chunk, while the sample needs eight
    service.fetchConcurrency.workerCapacity = 1;
    service.fetchConcurrency.maxConcurrentReads = 1;
    vc::render::ChunkCache cache({{{8, 8, 8}, {4, 4, 4}, {}}}, {fetcher}, 0,
                                vc::render::ChunkDtype::UInt8, options, service);
    cv::Mat_<cv::Vec3f> points(1, 1, cv::Vec3f{3.25f, 3.25f, 3.25f});
    cv::Mat_<cv::Vec3f> dirs(1, 1, cv::Vec3f{0, 0, 0});
    Keys keys;
    vc::render::prefetch::insertExactChunksForSamples(
        points, dirs, {0}, &cache, 0, vc::Sampling::Trilinear, keys);
    REQUIRE(keys.size() == 8);
    cache.prefetchChunks(std::vector<vc::render::ChunkKey>(keys.begin(), keys.end()), true);
    const int initialFetches = fetcher->calls.load();
    vc::render::prefetch::PrefetchedArrayView view(cache);
    std::vector<cv::Mat_<uint8_t>> output;
    readMultiSlice(output, &view, 0, points, dirs, {0});
    REQUIRE(output.size() == 1);
    CHECK(output[0](0, 0) == 17);
    CHECK(fetcher->calls.load() > initialFetches);
}

TEST_CASE("prefetched view preserves source errors instead of hiding failed reads")
{
    auto fetcher = std::make_shared<CountingPayloadFetcher>();
    fetcher->fail = true;
    vc::render::ChunkCache cache({{{4, 4, 4}, {4, 4, 4}, {}}}, {fetcher}, 0,
                                vc::render::ChunkDtype::UInt8);
    cache.prefetchChunks({{0, 0, 0, 0}}, true);
    vc::render::prefetch::PrefetchedArrayView view(cache);
    const auto result = view.getChunkBlocking(0, 0, 0, 0);
    CHECK(result.status == vc::render::ChunkStatus::Error);
    CHECK(result.error == "expected source error");
}

TEST_CASE("prefetched view preserves uint16 samples in band and tile readers")
{
    auto fetcher = std::make_shared<CountingPayloadFetcher>();
    const uint16_t value = 1300;
    const std::vector<uint16_t> samples(64, value);
    fetcher->payload.resize(samples.size() * sizeof(uint16_t));
    std::memcpy(fetcher->payload.data(), samples.data(), fetcher->payload.size());
    vc::render::ChunkCache cache({{{4, 4, 4}, {4, 4, 4}, {}}}, {fetcher}, 0,
                                vc::render::ChunkDtype::UInt16);
    cache.prefetchChunks({{0, 0, 0, 0}}, true);
    vc::render::prefetch::PrefetchedArrayView view(cache);
    CHECK(view.dtype() == vc::render::ChunkDtype::UInt16);
    cv::Mat_<cv::Vec3f> points(1, 1, cv::Vec3f{1.25f, 1.25f, 1.25f});
    cv::Mat_<cv::Vec3f> dirs(1, 1, cv::Vec3f{0, 0, 0});
    std::vector<cv::Mat_<uint16_t>> output;
    readMultiSlice(output, &view, 0, points, dirs, {0});
    CHECK(output.at(0)(0, 0) == value);
    sampleTileSlices(output, &view, 0, points, dirs, {0});
    CHECK(output.at(0)(0, 0) == value);
}
