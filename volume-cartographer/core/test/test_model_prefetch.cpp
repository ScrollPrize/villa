#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/lasagna/ModelPrefetch.hpp"
#include "vc/core/render/ChunkFetch.hpp"

#include <chrono>
#include <limits>
#include <mutex>
#include <set>
#include <thread>

namespace {
using namespace vc::render;
using namespace vc::lasagna;

class RecordingFetcher : public IChunkFetcher {
public:
    ChunkFetchResult fetch(const ChunkKey& key) override
    {
        std::lock_guard lock(mutex);
        keys.insert({key.iz, key.iy, key.ix});
        ChunkFetchResult result;
        result.status = fail ? ChunkFetchStatus::HttpError : ChunkFetchStatus::Found;
        result.bytes.assign(64, std::byte{17});
        return result;
    }
    std::mutex mutex;
    bool fail = false; // configured before requests start
    std::set<std::array<int, 3>> keys;
};

struct Source {
    std::shared_ptr<RecordingFetcher> fetcher = std::make_shared<RecordingFetcher>();
    std::shared_ptr<ChunkCache> cache;
    explicit Source(size_t capacity = 1024 * 1024)
    {
        ChunkCacheService::Options service;
        service.decodedByteCapacity = capacity;
        service.fetchConcurrency.maxConcurrentReads = 4;
        service.fetchConcurrency.adaptive = false;
        cache = std::make_shared<ChunkCache>(
            std::vector<ChunkCache::LevelInfo>{{{64, 64, 64}, {4, 4, 4}, {}}},
            std::vector<std::shared_ptr<IChunkFetcher>>{fetcher}, 0.0,
            ChunkDtype::UInt8, ChunkCache::Options{}, service);
    }
    ModelPrefetchSource source(double spacing = 1.0, bool remote = true) const
    { return {cache, spacing, remote}; }
};
}

TEST_CASE("model prefetch covers dense and sparse corridors identically")
{
    Source source;
    ModelPrefetchOptions options;
    options.radius = 0;
    ModelPrefetchPlan sparse({source.source()}, {{4, 8, 8}, {60, 8, 8}}, options);
    CHECK(sparse.report().planned == 64);
    std::vector<cv::Vec3d> dense;
    for (int i = 0; i <= 10000; ++i)
        dense.emplace_back(4.0 + 56.0 * i / 10000.0, 8, 8);
    ModelPrefetchPlan plan({source.source()}, dense, options);
    CHECK(plan.report().planned == sparse.report().planned);
    CHECK_FALSE(plan.report().truncated);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    bool done = false;
    while (!(done = plan.pump()) && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    REQUIRE(done);
    for (int z : {1, 2})
        for (int y : {1, 2})
            for (int x = 0; x < 16; ++x)
                REQUIRE(source.cache->getChunkBlocking(0, z, y, x).status == ChunkStatus::Data);
    std::lock_guard lock(source.fetcher->mutex);
    CHECK(source.fetcher->keys.size() == 64);
    CHECK(plan.report().submitted == 64);
}

TEST_CASE("model prefetch uses each channel's own coordinate scale")
{
    Source a, b;
    ModelPrefetchOptions options;
    options.radius = 0;
    ModelPrefetchPlan first({a.source()}, {{4, 8, 8}, {60, 8, 8}}, options);
    ModelPrefetchPlan scaled({b.source(2)}, {{8, 16, 16}, {120, 16, 16}}, options);
    CHECK(first.report().planned == scaled.report().planned);
    ModelPrefetchPlan both({a.source(), b.source(2)}, {{8, 16, 16}}, options);
    CHECK(both.report().planned == 16);
    CHECK(both.report().plannedBytes == 16 * 64);
}

TEST_CASE("model prefetch bounds planning and bytes independently")
{
    Source source;
    ModelPrefetchOptions options;
    options.maxRequests = 7;
    options.maxPlannedBytes = 3 * 64;
    ModelPrefetchPlan bytes({source.source()}, {{0, 0, 0}, {63, 63, 63}}, options);
    CHECK(bytes.report().planned <= 3);
    CHECK(bytes.report().plannedBytes <= options.maxPlannedBytes);
    CHECK(bytes.report().truncated);
    options.maxPlanningSteps = 1;
    ModelPrefetchPlan work({source.source()}, {{0, 0, 0}, {63, 63, 63}}, options);
    CHECK(work.report().planned == 0);
    CHECK(work.report().truncated);
}

TEST_CASE("model prefetch clips outside points and ignores invalid or local inputs")
{
    Source source;
    ModelPrefetchOptions options;
    options.radius = 0;
    ModelPrefetchPlan clipped({source.source()}, {{-100000, 8, 8}, {100000, 8, 8}}, options);
    CHECK(clipped.report().planned == 64);
    ModelPrefetchPlan outside({source.source()}, {{-100, -100, -100}}, options);
    CHECK(outside.report().planned == 0);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    ModelPrefetchPlan invalid({source.source()}, {{nan, 8, 8}}, options);
    CHECK(invalid.report().planned == 0);
    ModelPrefetchPlan local({source.source(1, false), source.source(0), {nullptr, 1, true}},
                            {{8, 8, 8}}, options);
    CHECK(local.report().planned == 0);
}

TEST_CASE("model prefetch cancellation issues no new requests")
{
    Source source;
    ModelPrefetchPlan plan({source.source()}, {{8, 8, 8}});
    REQUIRE(plan.report().planned > 0);
    std::atomic<bool> cancelled{true};
    CHECK(plan.pump(&cancelled));
    CHECK(plan.report().submitted == 0);
    CHECK(plan.pump());
    std::lock_guard lock(source.fetcher->mutex);
    CHECK(source.fetcher->keys.empty());
}

TEST_CASE("model prefetch skips permanent omissions without stalling other channels")
{
    bool tooSmall = false;
    SUBCASE("decoded budget cannot fit a chunk") { tooSmall = true; }
    SUBCASE("an earlier required read cached an error") { tooSmall = false; }
    Source unavailable(tooSmall ? 64 : 1024 * 1024), available;
    if (!tooSmall) {
        unavailable.fetcher->fail = true;
        REQUIRE(unavailable.cache->getChunkBlocking(0, 1, 1, 1).status == ChunkStatus::Error);
    }
    ModelPrefetchOptions options;
    options.radius = 0;
    ModelPrefetchPlan plan({unavailable.source(), available.source()}, {{8, 8, 8}}, options);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    bool done = false;
    while (!(done = plan.pump()) && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    REQUIRE(done);
    CHECK(plan.report().skipped >= (tooSmall ? 8 : 1));
    for (int z : {1, 2})
        for (int y : {1, 2})
            for (int x : {1, 2})
                CHECK(available.cache->getChunkBlocking(0, z, y, x).status == ChunkStatus::Data);
    std::lock_guard lock(available.fetcher->mutex);
    CHECK(available.fetcher->keys.size() == 8);
}
