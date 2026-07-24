// Coverage for core/src/render/ChunkCache.cpp.
//
// Drives the cache with a synthetic IChunkFetcher so we can deterministically
// exercise the hit/miss/in-flight/AllFill/error paths plus prefetch.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/render/ChunkCache.hpp"
#include "vc/core/render/ChunkFetch.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <filesystem>
#include <latch>
#include <memory>
#include <mutex>
#include <random>
#include <thread>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;
using vc::render::ChunkCache;
using vc::render::ChunkDtype;
using vc::render::DecodedChunkCacheBudget;
using vc::render::ChunkFetchResult;
using vc::render::ChunkFetchStatus;
using vc::render::ChunkKey;
using vc::render::ChunkResult;
using vc::render::ChunkStatus;
using vc::render::IChunkFetcher;

namespace {

class CountingFetcher : public IChunkFetcher {
public:
    ChunkFetchResult fetch(const ChunkKey& key) override
    {
        ++fetchCalls;
        std::lock_guard<std::mutex> lk(m_);
        auto it = canned_.find(key);
        if (it != canned_.end()) return it->second;
        ChunkFetchResult r;
        r.status = ChunkFetchStatus::Missing;
        return r;
    }

    void setCanned(const ChunkKey& k, ChunkFetchResult r)
    {
        std::lock_guard<std::mutex> lk(m_);
        canned_[k] = std::move(r);
    }

    std::atomic<int> fetchCalls{0};
private:
    std::mutex m_;
    std::unordered_map<ChunkKey, ChunkFetchResult, vc::render::ChunkKeyHash> canned_;
};

std::vector<std::byte> makeBytes(std::size_t n, std::byte v = std::byte{99})
{
    return std::vector<std::byte>(n, v);
}

std::shared_ptr<ChunkCache> makeCache(std::shared_ptr<CountingFetcher> f,
                                       std::array<int, 3> shape = {8, 8, 8},
                                       std::array<int, 3> chunkShape = {4, 4, 4})
{
    std::vector<ChunkCache::LevelInfo> levels = {
        {shape, chunkShape, {}},
    };
    ChunkCache::Options opts;
    opts.maxConcurrentReads = 4;
    opts.detectAllFillChunks = true;
    return std::make_shared<ChunkCache>(
        std::move(levels),
        std::vector<std::shared_ptr<vc::render::IChunkFetcher>>{f},
        /*fillValue=*/0.0,
        ChunkDtype::UInt8,
        opts);
}

ChunkResult waitForResolved(ChunkCache& c, int level, int iz, int iy, int ix,
                            std::chrono::milliseconds timeout = std::chrono::seconds{2})
{
    auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        auto r = c.tryGetChunk(level, iz, iy, ix);
        if (r.status != ChunkStatus::MissQueued) return r;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return c.tryGetChunk(level, iz, iy, ix);
}

} // namespace

TEST_CASE("ChunkCache basic IChunkedArray accessors")
{
    auto f = std::make_shared<CountingFetcher>();
    auto c = makeCache(f);
    CHECK(c->numLevels() == 1);
    CHECK(c->shape(0) == std::array<int, 3>{8, 8, 8});
    CHECK(c->chunkShape(0) == std::array<int, 3>{4, 4, 4});
    CHECK(c->dtype() == ChunkDtype::UInt8);
    CHECK(c->fillValue() == 0.0);
    auto lt = c->levelTransform(0);
    CHECK(lt.scaleFromLevel0[0] == doctest::Approx(1.0));
}

TEST_CASE("ChunkCache: out-of-range keys do not return Data; no fetcher hit")
{
    auto f = std::make_shared<CountingFetcher>();
    auto c = makeCache(f);
    auto r = c->tryGetChunk(0, /*iz=*/99, /*iy=*/0, /*ix=*/0);
    CHECK(r.status != ChunkStatus::Data);
    auto r2 = c->tryGetChunk(/*level=*/99, 0, 0, 0);
    CHECK(r2.status != ChunkStatus::Data);
    // Fetcher must not be called for out-of-range keys.
    CHECK(f->fetchCalls.load() == 0);
}

TEST_CASE("ChunkCache: first tryGetChunk queues; second returns the data")
{
    auto f = std::make_shared<CountingFetcher>();
    ChunkKey k{0, 0, 0, 0};
    ChunkFetchResult fr;
    fr.status = ChunkFetchStatus::Found;
    fr.bytes = makeBytes(4 * 4 * 4, std::byte{77});
    f->setCanned(k, fr);
    auto c = makeCache(f);

    auto first = c->tryGetChunk(0, 0, 0, 0);
    // The first call may resolve synchronously (small payload) or queue.
    CHECK((first.status == ChunkStatus::MissQueued || first.status == ChunkStatus::Data));

    auto resolved = waitForResolved(*c, 0, 0, 0, 0);
    REQUIRE(resolved.status == ChunkStatus::Data);
    REQUIRE(resolved.bytes);
    CHECK(resolved.bytes->size() == 4 * 4 * 4);
    CHECK(int(std::to_integer<int>((*resolved.bytes)[0])) == 77);

    // Second access hits cache; fetcher not called again.
    int callsAfter = f->fetchCalls.load();
    auto cached = c->tryGetChunk(0, 0, 0, 0);
    CHECK(cached.status == ChunkStatus::Data);
    CHECK(f->fetchCalls.load() == callsAfter);
}

TEST_CASE("ChunkCache: cache-only reads neither queue nor promote decoded chunks")
{
    auto f = std::make_shared<CountingFetcher>();
    for (int ix = 0; ix < 3; ++ix) {
        ChunkFetchResult result;
        result.status = ChunkFetchStatus::Found;
        result.bytes = makeBytes(64, std::byte{static_cast<unsigned char>(ix + 1)});
        f->setCanned({0, 0, 0, ix}, std::move(result));
    }

    std::vector<ChunkCache::LevelInfo> levels = {
        {{4, 4, 12}, {4, 4, 4}, {}},
    };
    ChunkCache::Options opts;
    opts.decodedByteCapacity = 1024;
    opts.decodedByteBudget = std::make_shared<DecodedChunkCacheBudget>(128);
    opts.maxConcurrentReads = 1;
    opts.detectAllFillChunks = false;
    ChunkCache cache(
        std::move(levels),
        std::vector<std::shared_ptr<vc::render::IChunkFetcher>>{f},
        0.0, ChunkDtype::UInt8, opts);

    CHECK(cache.getChunkIfCached(0, 0, 0, 0).status == ChunkStatus::MissQueued);
    CHECK(f->fetchCalls.load() == 0);
    REQUIRE(waitForResolved(cache, 0, 0, 0, 0).status == ChunkStatus::Data);
    REQUIRE(waitForResolved(cache, 0, 0, 0, 1).status == ChunkStatus::Data);

    // Looking at the oldest chunk as a fallback must not make it newer than
    // the target working set.
    CHECK(cache.getChunkIfCached(0, 0, 0, 0).status == ChunkStatus::Data);
    REQUIRE(waitForResolved(cache, 0, 0, 0, 2).status == ChunkStatus::Data);

    CHECK(cache.getChunkIfCached(0, 0, 0, 0).status == ChunkStatus::MissQueued);
    CHECK(cache.getChunkIfCached(0, 0, 0, 1).status == ChunkStatus::Data);
    CHECK(cache.getChunkIfCached(0, 0, 0, 2).status == ChunkStatus::Data);
}

TEST_CASE("ChunkCache: getChunkBlocking returns Data immediately for a found chunk")
{
    auto f = std::make_shared<CountingFetcher>();
    ChunkKey k{0, 0, 0, 0};
    ChunkFetchResult fr;
    fr.status = ChunkFetchStatus::Found;
    fr.bytes = makeBytes(4 * 4 * 4, std::byte{55});
    f->setCanned(k, fr);
    auto c = makeCache(f);
    auto r = c->getChunkBlocking(0, 0, 0, 0);
    REQUIRE(r.status == ChunkStatus::Data);
    CHECK(r.bytes->size() == 64);
}

TEST_CASE("ChunkCache: Missing fetch resolves to Missing status")
{
    auto f = std::make_shared<CountingFetcher>();
    // No canned -> Missing by default.
    auto c = makeCache(f);
    auto r = waitForResolved(*c, 0, 0, 0, 0);
    CHECK(r.status == ChunkStatus::Missing);
}

TEST_CASE("ChunkCache: all-zero data is detected as AllFill")
{
    auto f = std::make_shared<CountingFetcher>();
    ChunkFetchResult fr;
    fr.status = ChunkFetchStatus::Found;
    fr.bytes = makeBytes(4 * 4 * 4, std::byte{0}); // all == fillValue=0
    f->setCanned({0, 0, 0, 0}, fr);
    auto c = makeCache(f);
    auto r = waitForResolved(*c, 0, 0, 0, 0);
    CHECK(r.status == ChunkStatus::AllFill);
}

TEST_CASE("ChunkCache: HttpError/IoError surface as Error status")
{
    auto f = std::make_shared<CountingFetcher>();
    ChunkFetchResult fr;
    fr.status = ChunkFetchStatus::HttpError;
    fr.httpStatus = 500;
    fr.message = "server down";
    f->setCanned({0, 0, 0, 0}, fr);
    auto c = makeCache(f);
    auto r = waitForResolved(*c, 0, 0, 0, 0);
    CHECK(r.status == ChunkStatus::Error);
}

TEST_CASE("ChunkCache: prefetchChunks(wait=true) populates the cache")
{
    auto f = std::make_shared<CountingFetcher>();
    for (int iz : {0, 1}) {
        ChunkFetchResult fr;
        fr.status = ChunkFetchStatus::Found;
        fr.bytes = makeBytes(64, std::byte{42});
        f->setCanned({0, iz, 0, 0}, fr);
    }
    auto c = makeCache(f);
    std::vector<ChunkKey> keys = {{0, 0, 0, 0}, {0, 1, 0, 0}};
    c->prefetchChunks(keys, /*wait=*/true, /*priorityOffset=*/0);
    // Both should be resolved synchronously after wait=true returns.
    auto r0 = c->tryGetChunk(0, 0, 0, 0);
    auto r1 = c->tryGetChunk(0, 1, 0, 0);
    CHECK(r0.status == ChunkStatus::Data);
    CHECK(r1.status == ChunkStatus::Data);
}

TEST_CASE("ChunkCache: stats reflect decoded byte budget and activity")
{
    auto f = std::make_shared<CountingFetcher>();
    ChunkFetchResult fr;
    fr.status = ChunkFetchStatus::Found;
    fr.bytes = makeBytes(64, std::byte{1});
    f->setCanned({0, 0, 0, 0}, fr);
    auto c = makeCache(f);
    (void)waitForResolved(*c, 0, 0, 0, 0);
    auto s = c->stats();
    CHECK(s.decodedByteCapacity > 0);
    CHECK(s.decodedBytes >= 64);
    CHECK_FALSE(s.persistentCacheEnabled);
}

TEST_CASE("ChunkCache: invalidate clears decoded entries")
{
    auto f = std::make_shared<CountingFetcher>();
    ChunkFetchResult fr;
    fr.status = ChunkFetchStatus::Found;
    fr.bytes = makeBytes(64, std::byte{1});
    f->setCanned({0, 0, 0, 0}, fr);
    auto c = makeCache(f);
    (void)waitForResolved(*c, 0, 0, 0, 0);
    auto before = c->stats();
    CHECK(before.decodedBytes >= 64);
    c->invalidate();
    auto after = c->stats();
    CHECK(after.decodedBytes == 0);
    // Next access re-fetches.
    int calls_before = f->fetchCalls.load();
    (void)waitForResolved(*c, 0, 0, 0, 0);
    CHECK(f->fetchCalls.load() > calls_before);
}

TEST_CASE("ChunkCache: addChunkReadyListener/removeChunkReadyListener fires on resolve")
{
    auto f = std::make_shared<CountingFetcher>();
    ChunkFetchResult fr;
    fr.status = ChunkFetchStatus::Found;
    fr.bytes = makeBytes(64, std::byte{2});
    f->setCanned({0, 0, 0, 0}, fr);
    auto c = makeCache(f);

    std::atomic<int> fires{0};
    auto id = c->addChunkReadyListener([&]() { ++fires; });
    (void)waitForResolved(*c, 0, 0, 0, 0);
    // Allow a short tail for the callback to fire.
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    CHECK(fires.load() >= 1);
    c->removeChunkReadyListener(id);
    // Removing again is a no-op (just shouldn't crash).
    c->removeChunkReadyListener(id);
}

TEST_CASE("ChunkCache: persistent cache path round-trip")
{
    std::mt19937_64 rng(std::random_device{}());
    auto persistDir = fs::temp_directory_path() /
        ("vc_chunk_cache_persist_" + std::to_string(rng()));
    fs::create_directories(persistDir);

    auto f = std::make_shared<CountingFetcher>();
    ChunkFetchResult fr;
    fr.status = ChunkFetchStatus::Found;
    fr.bytes = makeBytes(64, std::byte{33});
    f->setCanned({0, 0, 0, 0}, fr);

    {
        std::vector<ChunkCache::LevelInfo> levels = {{{8,8,8}, {4,4,4}, {}}};
        ChunkCache::Options opts;
        opts.persistentCachePath = persistDir;
        ChunkCache c(std::move(levels),
                     std::vector<std::shared_ptr<vc::render::IChunkFetcher>>{f},
                     0.0, ChunkDtype::UInt8, opts);
        auto r = waitForResolved(c, 0, 0, 0, 0);
        CHECK(r.status == ChunkStatus::Data);
    }

    // New cache: should be able to read from persistent storage without
    // re-fetching. The fetcher could still be called once for the in-flight
    // path; just check we don't crash.
    {
        std::vector<ChunkCache::LevelInfo> levels = {{{8,8,8}, {4,4,4}, {}}};
        ChunkCache::Options opts;
        opts.persistentCachePath = persistDir;
        ChunkCache c(std::move(levels),
                     std::vector<std::shared_ptr<vc::render::IChunkFetcher>>{f},
                     0.0, ChunkDtype::UInt8, opts);
        auto r = waitForResolved(c, 0, 0, 0, 0);
        CHECK(r.status == ChunkStatus::Data);
    }

    fs::remove_all(persistDir);
}

TEST_CASE("ChunkCache: ctor without options uses defaults")
{
    auto f = std::make_shared<CountingFetcher>();
    std::vector<ChunkCache::LevelInfo> levels = {{{4,4,4}, {4,4,4}, {}}};
    ChunkCache c(std::move(levels),
                 std::vector<std::shared_ptr<vc::render::IChunkFetcher>>{f},
                 0.0, ChunkDtype::UInt8);
    CHECK(c.numLevels() == 1);
}

namespace {

std::shared_ptr<ChunkCache> makeTinyCapacityCache(
    std::shared_ptr<CountingFetcher> f)
{
    // 8x8x8 volume of 4x4x4 chunks: 8 chunks of 64 decoded bytes each.
    // Capacity of 128 bytes holds exactly two chunks.
    std::vector<ChunkCache::LevelInfo> levels = {{{8, 8, 8}, {4, 4, 4}, {}}};
    ChunkCache::Options opts;
    opts.maxConcurrentReads = 1;
    opts.detectAllFillChunks = true;
    opts.decodedByteCapacity = 128;
    return std::make_shared<ChunkCache>(
        std::move(levels),
        std::vector<std::shared_ptr<vc::render::IChunkFetcher>>{f},
        0.0, ChunkDtype::UInt8, opts);
}

std::shared_ptr<ChunkCache> makeSharedBudgetCache(
    std::shared_ptr<CountingFetcher> f,
    const std::shared_ptr<DecodedChunkCacheBudget>& budget)
{
    std::vector<ChunkCache::LevelInfo> levels = {{{8, 8, 8}, {4, 4, 4}, {}}};
    ChunkCache::Options opts;
    opts.maxConcurrentReads = 1;
    opts.detectAllFillChunks = true;
    opts.decodedByteCapacity = 1024;
    opts.decodedByteBudget = budget;
    return std::make_shared<ChunkCache>(
        std::move(levels),
        std::vector<std::shared_ptr<vc::render::IChunkFetcher>>{f},
        0.0, ChunkDtype::UInt8, opts);
}

void cannedDataChunks(CountingFetcher& f, int count)
{
    int i = 0;
    for (int iz : {0, 1})
        for (int iy : {0, 1})
            for (int ix : {0, 1}) {
                if (i++ >= count)
                    return;
                ChunkFetchResult fr;
                fr.status = ChunkFetchStatus::Found;
                fr.bytes = makeBytes(64, std::byte{99});
                f.setCanned({0, iz, iy, ix}, fr);
            }
}

} // namespace

TEST_CASE("ChunkCache: shared decoded budget is enforced across caches")
{
    auto budget = std::make_shared<DecodedChunkCacheBudget>(128);
    auto f1 = std::make_shared<CountingFetcher>();
    auto f2 = std::make_shared<CountingFetcher>();
    cannedDataChunks(*f1, 2);
    cannedDataChunks(*f2, 2);
    auto c1 = makeSharedBudgetCache(f1, budget);
    auto c2 = makeSharedBudgetCache(f2, budget);

    CHECK(waitForResolved(*c1, 0, 0, 0, 0).status == ChunkStatus::Data);
    CHECK(waitForResolved(*c2, 0, 0, 0, 0).status == ChunkStatus::Data);
    // Make c1's first chunk newer than c2's before adding a third chunk.
    CHECK(c1->tryGetChunk(0, 0, 0, 0).status == ChunkStatus::Data);
    CHECK(waitForResolved(*c2, 0, 0, 0, 1).status == ChunkStatus::Data);

    const auto stats = budget->stats();
    CHECK(stats.decodedBytes <= 128);
    CHECK(stats.maximumBytes == 128);
    CHECK(stats.cacheCount == 2);
    CHECK(c1->stats().decodedBytes == stats.decodedBytes);
    CHECK(c1->stats().decodedByteCapacity == 128);
    CHECK(c1->tryGetChunk(0, 0, 0, 0).status == ChunkStatus::Data);
    CHECK(c2->tryGetChunk(0, 0, 0, 0).status == ChunkStatus::MissQueued);
}

TEST_CASE("ChunkCache: invalidation releases shared decoded budget bytes")
{
    auto budget = std::make_shared<DecodedChunkCacheBudget>(128);
    auto f1 = std::make_shared<CountingFetcher>();
    auto f2 = std::make_shared<CountingFetcher>();
    cannedDataChunks(*f1, 1);
    cannedDataChunks(*f2, 1);
    auto c1 = makeSharedBudgetCache(f1, budget);
    auto c2 = makeSharedBudgetCache(f2, budget);

    CHECK(waitForResolved(*c1, 0, 0, 0, 0).status == ChunkStatus::Data);
    CHECK(waitForResolved(*c2, 0, 0, 0, 0).status == ChunkStatus::Data);
    CHECK(budget->stats().decodedBytes == 128);

    c1->invalidate();
    CHECK(budget->stats().decodedBytes == 64);
    c2->invalidate();
    CHECK(budget->stats().decodedBytes == 0);
}

TEST_CASE("ChunkCache: separate decoded budgets do not evict each other")
{
    auto normalBudget = std::make_shared<DecodedChunkCacheBudget>(64);
    auto overlayBudget = std::make_shared<DecodedChunkCacheBudget>(64);
    auto normalFetcher = std::make_shared<CountingFetcher>();
    auto overlayFetcher = std::make_shared<CountingFetcher>();
    cannedDataChunks(*normalFetcher, 1);
    cannedDataChunks(*overlayFetcher, 1);
    auto normal = makeSharedBudgetCache(normalFetcher, normalBudget);
    auto overlay = makeSharedBudgetCache(overlayFetcher, overlayBudget);

    CHECK(waitForResolved(*normal, 0, 0, 0, 0).status == ChunkStatus::Data);
    CHECK(waitForResolved(*overlay, 0, 0, 0, 0).status == ChunkStatus::Data);
    CHECK(normalBudget->stats().decodedBytes == 64);
    CHECK(overlayBudget->stats().decodedBytes == 64);
}

TEST_CASE("ChunkCache: recently touched entries never exceed capacity")
{
    auto f = std::make_shared<CountingFetcher>();
    cannedDataChunks(*f, 4);
    auto c = makeTinyCapacityCache(f);

    // Chunk order: (0,0,0), (0,0,1), (0,1,0), (0,1,1).
    CHECK(waitForResolved(*c, 0, 0, 0, 0).status == ChunkStatus::Data);
    CHECK(waitForResolved(*c, 0, 0, 0, 1).status == ChunkStatus::Data);
    CHECK(waitForResolved(*c, 0, 0, 1, 0).status == ChunkStatus::Data);
    CHECK(waitForResolved(*c, 0, 0, 1, 1).status == ChunkStatus::Data);

    // Strict LRU retains only the two most recent chunks.
    CHECK(c->stats().decodedBytes <= 128);
    CHECK(c->tryGetChunk(0, 0, 0, 0).status == ChunkStatus::MissQueued);
    CHECK(c->tryGetChunk(0, 0, 1, 0).status == ChunkStatus::Data);
    CHECK(c->tryGetChunk(0, 0, 1, 1).status == ChunkStatus::Data);
}

TEST_CASE("ChunkCache: every store enforces the configured byte capacity")
{
    auto f = std::make_shared<CountingFetcher>();
    cannedDataChunks(*f, 5);
    auto c = makeTinyCapacityCache(f);

    CHECK(waitForResolved(*c, 0, 0, 0, 0).status == ChunkStatus::Data);
    CHECK(waitForResolved(*c, 0, 0, 0, 1).status == ChunkStatus::Data);
    CHECK(waitForResolved(*c, 0, 0, 1, 0).status == ChunkStatus::Data);
    CHECK(waitForResolved(*c, 0, 0, 1, 1).status == ChunkStatus::Data);
    CHECK(waitForResolved(*c, 0, 1, 0, 0).status == ChunkStatus::Data);

    CHECK(c->stats().decodedBytes <= 128);
    CHECK(c->tryGetChunk(0, 0, 0, 0).status == ChunkStatus::MissQueued);
    CHECK(c->tryGetChunk(0, 1, 0, 0).status == ChunkStatus::Data);
}

TEST_CASE("ChunkCache: a new view priority epoch does not relax capacity")
{
    auto f = std::make_shared<CountingFetcher>();
    cannedDataChunks(*f, 5);
    auto c = makeTinyCapacityCache(f);

    c->beginViewRequest();
    CHECK(waitForResolved(*c, 0, 0, 0, 0).status == ChunkStatus::Data);
    CHECK(waitForResolved(*c, 0, 0, 0, 1).status == ChunkStatus::Data);
    CHECK(waitForResolved(*c, 0, 0, 1, 0).status == ChunkStatus::Data);
    CHECK(waitForResolved(*c, 0, 0, 1, 1).status == ChunkStatus::Data);
    CHECK(waitForResolved(*c, 0, 1, 0, 0).status == ChunkStatus::Data);

    // View epochs affect fetch priority only; decoded storage remains strict.
    CHECK(c->stats().decodedBytes <= 128);
    CHECK(c->tryGetChunk(0, 1, 0, 0).status == ChunkStatus::Data);
}

TEST_CASE("ChunkCache: entries from an earlier view epoch stay evictable")
{
    auto f = std::make_shared<CountingFetcher>();
    cannedDataChunks(*f, 4);
    auto c = makeTinyCapacityCache(f);

    CHECK(waitForResolved(*c, 0, 0, 0, 0).status == ChunkStatus::Data);
    CHECK(waitForResolved(*c, 0, 0, 0, 1).status == ChunkStatus::Data);

    // Advancing request priority does not change ordinary LRU eviction.
    c->beginViewRequest();
    CHECK(waitForResolved(*c, 0, 0, 1, 0).status == ChunkStatus::Data);
    CHECK(waitForResolved(*c, 0, 0, 1, 1).status == ChunkStatus::Data);

    CHECK(c->stats().decodedBytes <= 128);
    CHECK(c->tryGetChunk(0, 0, 1, 0).status == ChunkStatus::Data);
    CHECK(c->tryGetChunk(0, 0, 1, 1).status == ChunkStatus::Data);
}

TEST_CASE("ChunkCache: a large view working set never exceeds capacity")
{
    // 16x8x8 volume of 4x4x4 chunks: 16 chunks.
    auto f = std::make_shared<CountingFetcher>();
    for (int iz = 0; iz < 4; ++iz)
        for (int iy : {0, 1})
            for (int ix : {0, 1}) {
                ChunkFetchResult fr;
                fr.status = ChunkFetchStatus::Found;
                fr.bytes = makeBytes(64, std::byte{99});
                f->setCanned({0, iz, iy, ix}, fr);
            }
    std::vector<ChunkCache::LevelInfo> levels = {{{16, 8, 8}, {4, 4, 4}, {}}};
    ChunkCache::Options opts;
    opts.maxConcurrentReads = 1;
    opts.detectAllFillChunks = true;
    opts.decodedByteCapacity = 128;
    auto c = std::make_shared<ChunkCache>(
        std::move(levels),
        std::vector<std::shared_ptr<vc::render::IChunkFetcher>>{f},
        0.0, ChunkDtype::UInt8, opts);

    c->beginViewRequest();
    for (int i = 0; i < 9; ++i) {
        const int iz = i / 4;
        const int iy = (i / 2) % 2;
        const int ix = i % 2;
        CHECK(waitForResolved(*c, 0, iz, iy, ix).status == ChunkStatus::Data);
    }

    CHECK(c->stats().decodedBytes <= 128);
    CHECK(c->tryGetChunk(0, 0, 0, 0).status == ChunkStatus::MissQueued);
}

namespace {

// Records fetch order; the first fetch blocks until release() so later
// requests pile up in the priority queue behind it.
class BlockingOrderFetcher : public IChunkFetcher {
public:
    ChunkFetchResult fetch(const ChunkKey& key) override
    {
        bool first = false;
        {
            std::lock_guard<std::mutex> lk(m_);
            order_.push_back(key);
            first = order_.size() == 1;
        }
        if (first) {
            started_.count_down();
            std::unique_lock<std::mutex> lk(m_);
            cv_.wait(lk, [&] { return released_; });
        }
        ChunkFetchResult r;
        r.status = ChunkFetchStatus::Found;
        r.bytes = makeBytes(64, std::byte{7});
        return r;
    }

    void waitFirstStarted() { started_.wait(); }

    void release()
    {
        {
            std::lock_guard<std::mutex> lk(m_);
            released_ = true;
        }
        cv_.notify_all();
    }

    std::vector<ChunkKey> order()
    {
        std::lock_guard<std::mutex> lk(m_);
        return order_;
    }

private:
    std::mutex m_;
    std::condition_variable cv_;
    std::latch started_{1};
    bool released_ = false;
    std::vector<ChunkKey> order_;
};

} // namespace

TEST_CASE("ChunkCache: coarser levels are fetched before finer ones")
{
    auto f = std::make_shared<BlockingOrderFetcher>();
    std::vector<ChunkCache::LevelInfo> levels = {
        {{8, 8, 8}, {4, 4, 4}, {}},
        {{4, 4, 4}, {4, 4, 4}, {}},
    };
    ChunkCache::Options opts;
    opts.maxConcurrentReads = 1; // single worker => strict priority order
    opts.detectAllFillChunks = false;
    auto c = std::make_shared<ChunkCache>(
        std::move(levels),
        std::vector<std::shared_ptr<vc::render::IChunkFetcher>>{f, f},
        0.0, ChunkDtype::UInt8, opts);

    // Occupy the single worker, then queue a fine and a coarse chunk.
    (void)c->tryGetChunk(0, 0, 0, 0);
    f->waitFirstStarted();
    (void)c->tryGetChunk(0, 0, 0, 1); // fine (level 0)
    (void)c->tryGetChunk(1, 0, 0, 0); // coarse (level 1)
    f->release();

    CHECK(waitForResolved(*c, 0, 0, 0, 1).status == ChunkStatus::Data);
    CHECK(waitForResolved(*c, 1, 0, 0, 0).status == ChunkStatus::Data);

    const auto order = f->order();
    REQUIRE(order.size() == 3);
    CHECK(order[1].level == 1); // coarse chunk jumped the fine one
    CHECK(order[2].level == 0);
}

TEST_CASE("ChunkCache: disk-cached chunks resolve without touching the fetcher pool")
{
    std::mt19937_64 rng(std::random_device{}());
    const auto dir = fs::temp_directory_path() /
                     ("vc_chunk_probe_" + std::to_string(rng()));
    fs::create_directories(dir);

    ChunkFetchResult fr;
    fr.status = ChunkFetchStatus::Found;
    fr.bytes = makeBytes(64, std::byte{123});

    {
        // Warm the persistent cache.
        auto f = std::make_shared<CountingFetcher>();
        f->setCanned({0, 0, 0, 0}, fr);
        std::vector<ChunkCache::LevelInfo> levels = {{{4, 4, 4}, {4, 4, 4}, {}}};
        ChunkCache::Options opts;
        opts.persistentCachePath = dir;
        ChunkCache c(std::move(levels),
                     std::vector<std::shared_ptr<vc::render::IChunkFetcher>>{f},
                     0.0, ChunkDtype::UInt8, opts);
        CHECK(waitForResolved(c, 0, 0, 0, 0).status == ChunkStatus::Data);
        c.waitForPersistentWrites();
    }

    {
        // A fetcher that would only produce errors: the chunk must come
        // from the disk probe, never from the remote pool.
        auto f = std::make_shared<CountingFetcher>();
        ChunkFetchResult err;
        err.status = ChunkFetchStatus::HttpError;
        err.httpStatus = 500;
        f->setCanned({0, 0, 0, 0}, err);
        std::vector<ChunkCache::LevelInfo> levels = {{{4, 4, 4}, {4, 4, 4}, {}}};
        ChunkCache::Options opts;
        opts.persistentCachePath = dir;
        ChunkCache c(std::move(levels),
                     std::vector<std::shared_ptr<vc::render::IChunkFetcher>>{f},
                     0.0, ChunkDtype::UInt8, opts);
        auto r = waitForResolved(c, 0, 0, 0, 0);
        CHECK(r.status == ChunkStatus::Data);
        CHECK(f->fetchCalls.load() == 0);
    }

    fs::remove_all(dir);
}
