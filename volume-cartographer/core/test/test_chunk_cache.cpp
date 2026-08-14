// Coverage for core/src/render/ChunkCache.cpp.
//
// Drives the cache with a synthetic IChunkFetcher so we can deterministically
// exercise the hit/miss/in-flight/AllFill/error paths plus prefetch.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/render/ChunkCache.hpp"
#include "vc/core/render/ChunkFetch.hpp"
#include "vc/core/render/ChunkRequestScheduler.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <latch>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <span>
#include <thread>
#include <type_traits>
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
using vc::render::ChunkCacheService;
using vc::render::IChunkFetcher;
using vc::render::ChunkRequestScheduler;
using vc::render::ChunkRequestSelectionGate;
using vc::render::ChunkWorkPriority;

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

class BlockingFetcher : public IChunkFetcher {
public:
    ChunkFetchResult fetch(const ChunkKey&) override
    {
        ++fetchCalls;
        started_.count_down();
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [&] { return released_; });
        }
        finished_.count_down();
        ChunkFetchResult result;
        result.status = ChunkFetchStatus::Found;
        result.bytes = std::vector<std::byte>(64, std::byte{17});
        return result;
    }

    void waitStarted() { started_.wait(); }
    void waitFinished() { finished_.wait(); }
    void release()
    {
        {
            std::lock_guard lock(mutex_);
            released_ = true;
        }
        cv_.notify_all();
    }

    std::atomic<int> fetchCalls{0};

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    std::latch started_{1};
    std::latch finished_{1};
    bool released_ = false;
};

class BlockingEncodedFetcher : public IChunkFetcher {
public:
    ChunkFetchResult fetch(const ChunkKey& key) override
    {
        return decodeFetched(key, fetchEncoded(key));
    }

    ChunkFetchResult fetchEncoded(const ChunkKey&) override
    {
        std::call_once(startedOnce_, [this] { started_.count_down(); });
        {
            std::unique_lock lock(mutex_);
            cv_.wait(lock, [&] { return released_; });
        }
        ChunkFetchResult result;
        result.status = ChunkFetchStatus::Found;
        result.bytes = {std::byte{17}};
        return result;
    }

    ChunkFetchResult decodeFetched(
        const ChunkKey&, ChunkFetchResult) const override
    {
        ++decodeCalls;
        ChunkFetchResult result;
        result.status = ChunkFetchStatus::Found;
        result.bytes = std::vector<std::byte>(64, std::byte{29});
        return result;
    }

    void waitStarted() { started_.wait(); }

    void release()
    {
        {
            std::lock_guard lock(mutex_);
            released_ = true;
        }
        cv_.notify_all();
    }

    mutable std::atomic<int> decodeCalls{0};

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    std::once_flag startedOnce_;
    std::latch started_{1};
    bool released_ = false;
};

class ThrowingFetcher : public IChunkFetcher {
public:
    ChunkFetchResult fetch(const ChunkKey&) override
    {
        throw std::runtime_error("synthetic fetch failure");
    }
};

class SplitStageFetcher : public IChunkFetcher {
public:
    ~SplitStageFetcher() override { releasePersistentDecodes(); }

    ChunkFetchResult fetch(const ChunkKey& key) override
    {
        return decodeFetched(key, fetchEncoded(key));
    }

    ChunkFetchResult fetchEncoded(const ChunkKey&) override
    {
        {
            std::lock_guard lock(mutex_);
            remoteThread_ = std::this_thread::get_id();
            ++remoteCalls_;
        }
        cv_.notify_all();
        ChunkFetchResult result;
        result.status = ChunkFetchStatus::Found;
        result.bytes = {std::byte{77}};
        return result;
    }

    ChunkFetchResult decodeFetched(
        const ChunkKey&,
        ChunkFetchResult fetched) const override
    {
        {
            std::lock_guard lock(mutex_);
            remoteDecodeThread_ = std::this_thread::get_id();
            ++remoteDecodeCalls_;
        }
        cv_.notify_all();
        ChunkFetchResult result;
        if (fetched.status != ChunkFetchStatus::Found ||
            fetched.bytes != std::vector<std::byte>{std::byte{77}}) {
            result.status = ChunkFetchStatus::DecodeError;
            return result;
        }
        result.status = ChunkFetchStatus::Found;
        result.bytes = std::vector<std::byte>(64, std::byte{91});
        return result;
    }

    std::string persistentCacheExtension(const ChunkKey&) const override
    {
        return ".encoded";
    }

    ChunkFetchResult decodePersistentBytes(
        const ChunkKey& key,
        std::vector<std::byte>) const override
    {
        {
            std::unique_lock lock(mutex_);
            persistentDecodeOrder_.push_back(key);
            cv_.notify_all();
            cv_.wait(lock, [&] {
                return releaseAllPersistent_ || persistentDecodePermits_ > 0;
            });
            if (!releaseAllPersistent_)
                --persistentDecodePermits_;
        }
        ChunkFetchResult result;
        result.status = ChunkFetchStatus::Found;
        result.bytes = std::vector<std::byte>(64, std::byte{42});
        return result;
    }

    bool waitForRemote(std::chrono::milliseconds timeout) const
    {
        std::unique_lock lock(mutex_);
        return cv_.wait_for(lock, timeout, [&] { return remoteCalls_ > 0; });
    }

    bool waitForRemoteDecode(std::chrono::milliseconds timeout) const
    {
        std::unique_lock lock(mutex_);
        return cv_.wait_for(lock, timeout, [&] { return remoteDecodeCalls_ > 0; });
    }

    void releasePersistentDecodes() const
    {
        {
            std::lock_guard lock(mutex_);
            releaseAllPersistent_ = true;
        }
        cv_.notify_all();
    }

    void releasePersistentDecodes(int count) const
    {
        {
            std::lock_guard lock(mutex_);
            persistentDecodePermits_ += count;
        }
        cv_.notify_all();
    }

    bool waitForPersistentDecodes(
        std::size_t count,
        std::chrono::milliseconds timeout) const
    {
        std::unique_lock lock(mutex_);
        return cv_.wait_for(lock, timeout, [&] {
            return persistentDecodeOrder_.size() >= count;
        });
    }

    std::vector<ChunkKey> persistentDecodeOrder() const
    {
        std::lock_guard lock(mutex_);
        return persistentDecodeOrder_;
    }

    int remoteCalls() const
    {
        std::lock_guard lock(mutex_);
        return remoteCalls_;
    }

    bool remoteAndDecodeUsedDifferentThreads() const
    {
        std::lock_guard lock(mutex_);
        return remoteThread_ != std::thread::id{} &&
               remoteDecodeThread_ != std::thread::id{} &&
               remoteThread_ != remoteDecodeThread_;
    }

private:
    mutable std::mutex mutex_;
    mutable std::condition_variable cv_;
    mutable bool releaseAllPersistent_ = false;
    mutable int persistentDecodePermits_ = 0;
    mutable std::vector<ChunkKey> persistentDecodeOrder_;
    mutable int remoteCalls_ = 0;
    mutable int remoteDecodeCalls_ = 0;
    mutable std::thread::id remoteThread_;
    mutable std::thread::id remoteDecodeThread_;
};

std::vector<std::byte> makeBytes(std::size_t n, std::byte v = std::byte{99})
{
    return std::vector<std::byte>(n, v);
}

void writeTestBytes(const fs::path& path, std::span<const std::byte> bytes)
{
    fs::create_directories(path.parent_path());
    std::ofstream file(path, std::ios::binary);
    REQUIRE(file.good());
    file.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
    REQUIRE(file.good());
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

std::shared_ptr<ChunkCache> makeServiceCache(
    const std::shared_ptr<ChunkCacheService>& service,
    std::string identity,
    const std::shared_ptr<IChunkFetcher>& fetcher,
    std::size_t maxConcurrentReads = 4)
{
    std::vector<ChunkCache::LevelInfo> levels = {
        {{8, 8, 8}, {4, 4, 4}, {}},
    };
    ChunkCache::Options options;
    options.maxConcurrentReads = maxConcurrentReads;
    options.detectAllFillChunks = false;
    return std::make_shared<ChunkCache>(
        service, std::move(identity), std::move(levels),
        std::vector<std::shared_ptr<IChunkFetcher>>{fetcher},
        0.0, ChunkDtype::UInt8, std::move(options));
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

static_assert(std::is_trivially_copyable_v<vc::render::VolumeSourceId>);
static_assert(std::is_trivially_copyable_v<ChunkKey>);

TEST_CASE("ChunkCacheService interns source identity into a numeric hot key")
{
    auto service = std::make_shared<ChunkCacheService>(1024 * 1024);
    auto firstFetcher = std::make_shared<CountingFetcher>();
    auto secondFetcher = std::make_shared<CountingFetcher>();
    auto first = makeServiceCache(service, "local|/volume/a", firstFetcher);
    auto second = makeServiceCache(service, "local|/volume/a", secondFetcher);
    auto other = makeServiceCache(service, "local|/volume/b", secondFetcher);

    CHECK(first->sourceId());
    CHECK(first->sourceId() == second->sourceId());
    CHECK(first->sourceId() != other->sourceId());
    CHECK(service->sourceCount() == 2);
    CHECK(ChunkKey{0, 0, 0, 0, first->sourceId()} !=
          ChunkKey{0, 0, 0, 0, other->sourceId()});
}

TEST_CASE("ChunkCacheService rejects incompatible duplicate source metadata")
{
    auto service = std::make_shared<ChunkCacheService>(1024 * 1024);
    auto fetcher = std::make_shared<CountingFetcher>();
    auto first = makeServiceCache(service, "same-source", fetcher);
    std::vector<ChunkCache::LevelInfo> incompatibleLevels = {
        {{16, 8, 8}, {4, 4, 4}, {}},
    };
    ChunkCache::Options options;
    options.detectAllFillChunks = false;
    CHECK_THROWS_AS(
        ChunkCache(service, "same-source", std::move(incompatibleLevels),
                   std::vector<std::shared_ptr<IChunkFetcher>>{fetcher},
                   0.0, ChunkDtype::UInt8, std::move(options)),
        std::invalid_argument);
}

TEST_CASE("ChunkCacheService reuses a source across facade policy options")
{
    auto service = std::make_shared<ChunkCacheService>(1024 * 1024);
    auto fetcher = std::make_shared<CountingFetcher>();
    auto first = makeServiceCache(service, "policy-source", fetcher, 4);
    auto second = makeServiceCache(service, "policy-source", fetcher, 1);

    CHECK(first->sourceId() == second->sourceId());
    CHECK(service->sourceCount() == 1);
}

TEST_CASE("ChunkCacheService shares results and keeps sources warm")
{
    auto service = std::make_shared<ChunkCacheService>(1024 * 1024);
    auto fetcher = std::make_shared<CountingFetcher>();
    ChunkFetchResult result;
    result.status = ChunkFetchStatus::Found;
    result.bytes = makeBytes(64, std::byte{42});
    fetcher->setCanned({0, 0, 0, 0}, result);

    auto first = makeServiceCache(service, "remote|example/a|base=0", fetcher);
    REQUIRE(first->getChunkBlocking(0, 0, 0, 0).status == ChunkStatus::Data);
    CHECK(fetcher->fetchCalls.load() == 1);
    const auto sourceId = first->sourceId();
    first.reset();

    auto otherFetcher = std::make_shared<CountingFetcher>();
    ChunkFetchResult otherResult;
    otherResult.status = ChunkFetchStatus::Found;
    otherResult.bytes = makeBytes(64, std::byte{23});
    otherFetcher->setCanned({0, 0, 0, 0}, otherResult);
    auto other = makeServiceCache(service, "remote|example/b|base=0", otherFetcher);
    REQUIRE(other->getChunkBlocking(0, 0, 0, 0).status == ChunkStatus::Data);

    auto unusedFetcher = std::make_shared<CountingFetcher>();
    auto reacquired = makeServiceCache(
        service, "remote|example/a|base=0", unusedFetcher);
    CHECK(reacquired->sourceId() == sourceId);
    auto cached = reacquired->getChunkBlocking(0, 0, 0, 0);
    REQUIRE(cached.status == ChunkStatus::Data);
    CHECK(std::to_integer<int>((*cached.bytes)[0]) == 42);
    CHECK(fetcher->fetchCalls.load() == 1);
    CHECK(unusedFetcher->fetchCalls.load() == 0);
}

TEST_CASE("ChunkCacheService deduplicates an in-flight fetch across facades")
{
    auto service = std::make_shared<ChunkCacheService>(1024 * 1024);
    auto fetcher = std::make_shared<BlockingFetcher>();
    auto first = makeServiceCache(service, "shared-in-flight", fetcher);
    auto second = makeServiceCache(service, "shared-in-flight", fetcher);

    CHECK(first->tryGetChunk(0, 0, 0, 0).status == ChunkStatus::MissQueued);
    fetcher->waitStarted();
    CHECK(second->tryGetChunk(0, 0, 0, 0).status == ChunkStatus::MissQueued);
    CHECK(fetcher->fetchCalls.load() == 1);

    fetcher->release();
    CHECK(waitForResolved(*second, 0, 0, 0, 0).status == ChunkStatus::Data);
    CHECK(fetcher->fetchCalls.load() == 1);
}

TEST_CASE("ChunkCacheService rejects stale publication after source invalidation")
{
    auto service = std::make_shared<ChunkCacheService>(1024 * 1024);
    auto fetcher = std::make_shared<BlockingFetcher>();
    auto cache = makeServiceCache(service, "stale-source", fetcher);

    CHECK(cache->tryGetChunk(0, 0, 0, 0).status == ChunkStatus::MissQueued);
    fetcher->waitStarted();
    cache->invalidate();
    fetcher->release();
    fetcher->waitFinished();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    CHECK(cache->getChunkIfCached(0, 0, 0, 0).status ==
          ChunkStatus::MissQueued);
}

TEST_CASE("ChunkCacheService enforces one decoded budget across sources")
{
    auto service = std::make_shared<ChunkCacheService>(128);
    auto fetcherA = std::make_shared<CountingFetcher>();
    auto fetcherB = std::make_shared<CountingFetcher>();
    for (int ix : {0, 1}) {
        ChunkFetchResult result;
        result.status = ChunkFetchStatus::Found;
        result.bytes = makeBytes(
            64, std::byte{static_cast<unsigned char>(31 + ix)});
        fetcherA->setCanned({0, 0, 0, ix}, result);
        fetcherB->setCanned({0, 0, 0, ix}, result);
    }
    auto cacheA = makeServiceCache(service, "budget-a", fetcherA);
    auto cacheB = makeServiceCache(service, "budget-b", fetcherB);

    REQUIRE(cacheA->getChunkBlocking(0, 0, 0, 0).status == ChunkStatus::Data);
    REQUIRE(cacheB->getChunkBlocking(0, 0, 0, 0).status == ChunkStatus::Data);
    REQUIRE(cacheA->getChunkBlocking(0, 0, 0, 0).status == ChunkStatus::Data);
    REQUIRE(cacheB->getChunkBlocking(0, 0, 0, 1).status == ChunkStatus::Data);

    const auto stats = service->decodedByteBudget()->stats();
    CHECK(stats.decodedBytes == 128);
    CHECK(stats.cacheCount == 2);
    CHECK(cacheA->getChunkIfCached(0, 0, 0, 0).status == ChunkStatus::Data);
    CHECK(cacheB->getChunkIfCached(0, 0, 0, 0).status ==
          ChunkStatus::MissQueued);
    CHECK(cacheB->getChunkIfCached(0, 0, 0, 1).status == ChunkStatus::Data);
}

TEST_CASE("ChunkCacheService releases aggregate accounting on destruction")
{
    auto budget = std::make_shared<DecodedChunkCacheBudget>(1024 * 1024);
    {
        auto service =
            std::make_shared<ChunkCacheService>(1024 * 1024, budget);
        auto fetcher = std::make_shared<CountingFetcher>();
        ChunkFetchResult result;
        result.status = ChunkFetchStatus::Found;
        result.bytes = makeBytes(64, std::byte{7});
        fetcher->setCanned({0, 0, 0, 0}, result);
        auto cache = makeServiceCache(service, "temporary-source", fetcher);
        REQUIRE(cache->getChunkBlocking(0, 0, 0, 0).status ==
                ChunkStatus::Data);
        CHECK(budget->stats().decodedBytes == 64);
        cache.reset();
        CHECK(budget->stats().decodedBytes == 64);
    }
    CHECK(budget->stats().decodedBytes == 0);
}

TEST_CASE("ChunkCacheService facade destruction removes only its listeners")
{
    auto service = std::make_shared<ChunkCacheService>(1024 * 1024);
    auto fetcher = std::make_shared<CountingFetcher>();
    ChunkFetchResult result;
    result.status = ChunkFetchStatus::Found;
    result.bytes = makeBytes(64, std::byte{9});
    fetcher->setCanned({0, 0, 0, 0}, result);
    std::atomic<int> callbacks{0};
    {
        auto cache = makeServiceCache(service, "listener-source", fetcher);
        cache->addChunkReadyListener([&] { ++callbacks; });
    }
    auto reacquired = makeServiceCache(
        service, "listener-source", std::make_shared<CountingFetcher>());
    REQUIRE(reacquired->getChunkBlocking(0, 0, 0, 0).status ==
            ChunkStatus::Data);
    CHECK(callbacks.load() == 0);
}

TEST_CASE("ChunkCache reports source-qualified remote fetch start and stop")
{
    std::mt19937_64 rng(std::random_device{}());
    const auto dir = fs::temp_directory_path() /
                     ("vc_chunk_activity_" + std::to_string(rng()));
    fs::create_directories(dir);

    auto service = std::make_shared<ChunkCacheService>(1024 * 1024);
    auto fetcher = std::make_shared<BlockingFetcher>();
    std::vector<ChunkCache::LevelInfo> levels = {
        {{8, 8, 8}, {4, 4, 4}, {}},
    };
    ChunkCache::Options options;
    options.maxConcurrentReads = 1;
    options.detectAllFillChunks = false;
    options.persistentCachePath = dir;
    auto cache = std::make_shared<ChunkCache>(
        service, "remote-activity", std::move(levels),
        std::vector<std::shared_ptr<IChunkFetcher>>{fetcher},
        0.0, ChunkDtype::UInt8, std::move(options));

    std::mutex eventsMutex;
    std::vector<std::pair<ChunkKey, bool>> events;
    cache->addRemoteFetchActivityListener(
        [&](const ChunkKey& key, bool active) {
            std::lock_guard lock(eventsMutex);
            events.emplace_back(key, active);
        });

    CHECK(cache->tryGetChunk(0, 0, 0, 1).status == ChunkStatus::MissQueued);
    fetcher->waitStarted();
    {
        std::lock_guard lock(eventsMutex);
        REQUIRE(events.size() == 1);
        CHECK(events[0] == std::pair{
            ChunkKey{0, 0, 0, 1, cache->sourceId()}, true});
    }
    CHECK(cache->activeRemoteFetches() ==
          std::vector<ChunkKey>{{0, 0, 0, 1, cache->sourceId()}});

    fetcher->release();
    CHECK(waitForResolved(*cache, 0, 0, 0, 1).status == ChunkStatus::Data);
    {
        std::lock_guard lock(eventsMutex);
        REQUIRE(events.size() == 2);
        CHECK(events[1] == std::pair{
            ChunkKey{0, 0, 0, 1, cache->sourceId()}, false});
    }
    CHECK(cache->activeRemoteFetches().empty());
    cache->waitForPersistentWrites();
    fs::remove_all(dir);
}

TEST_CASE("ChunkCache clears remote activity after fetch exceptions and listener removal")
{
    std::mt19937_64 rng(std::random_device{}());
    const auto dir = fs::temp_directory_path() /
                     ("vc_chunk_activity_error_" + std::to_string(rng()));
    fs::create_directories(dir);

    auto service = std::make_shared<ChunkCacheService>(1024 * 1024);
    auto fetcher = std::make_shared<ThrowingFetcher>();
    std::vector<ChunkCache::LevelInfo> levels = {
        {{4, 4, 4}, {4, 4, 4}, {}},
    };
    ChunkCache::Options options;
    options.maxConcurrentReads = 1;
    options.detectAllFillChunks = false;
    options.persistentCachePath = dir;
    auto cache = std::make_shared<ChunkCache>(
        service, "remote-activity-error", std::move(levels),
        std::vector<std::shared_ptr<IChunkFetcher>>{fetcher},
        0.0, ChunkDtype::UInt8, std::move(options));

    std::atomic<int> callbacks{0};
    const auto listener = cache->addRemoteFetchActivityListener(
        [&](const ChunkKey&, bool) { ++callbacks; });
    cache->removeRemoteFetchActivityListener(listener);
    CHECK(waitForResolved(*cache, 0, 0, 0, 0).status == ChunkStatus::Error);
    CHECK(callbacks.load() == 0);
    CHECK(cache->activeRemoteFetches().empty());
    fs::remove_all(dir);
}

TEST_CASE("ChunkCache invalidation ends remote activity exactly once")
{
    std::mt19937_64 rng(std::random_device{}());
    const auto dir = fs::temp_directory_path() /
                     ("vc_chunk_activity_invalidate_" + std::to_string(rng()));
    fs::create_directories(dir);

    auto service = std::make_shared<ChunkCacheService>(1024 * 1024);
    auto fetcher = std::make_shared<BlockingFetcher>();
    std::vector<ChunkCache::LevelInfo> levels = {
        {{4, 4, 4}, {4, 4, 4}, {}},
    };
    ChunkCache::Options options;
    options.maxConcurrentReads = 1;
    options.detectAllFillChunks = false;
    options.persistentCachePath = dir;
    auto cache = std::make_shared<ChunkCache>(
        service, "remote-activity-invalidate", std::move(levels),
        std::vector<std::shared_ptr<IChunkFetcher>>{fetcher},
        0.0, ChunkDtype::UInt8, std::move(options));

    std::mutex eventsMutex;
    std::vector<bool> events;
    cache->addRemoteFetchActivityListener(
        [&](const ChunkKey&, bool active) {
            std::lock_guard lock(eventsMutex);
            events.push_back(active);
        });

    CHECK(cache->tryGetChunk(0, 0, 0, 0).status == ChunkStatus::MissQueued);
    fetcher->waitStarted();
    cache->invalidate();
    CHECK(cache->activeRemoteFetches().empty());
    {
        std::lock_guard lock(eventsMutex);
        CHECK(events == std::vector<bool>{true, false});
    }

    fetcher->release();
    fetcher->waitFinished();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    {
        std::lock_guard lock(eventsMutex);
        CHECK(events == std::vector<bool>{true, false});
    }
    fs::remove_all(dir);
}

TEST_CASE("ChunkCacheService isolates and invalidates sources independently")
{
    auto service = std::make_shared<ChunkCacheService>(1024 * 1024);
    auto fetcherA = std::make_shared<CountingFetcher>();
    auto fetcherB = std::make_shared<CountingFetcher>();
    ChunkFetchResult resultA;
    resultA.status = ChunkFetchStatus::Found;
    resultA.bytes = makeBytes(64, std::byte{11});
    ChunkFetchResult resultB;
    resultB.status = ChunkFetchStatus::Found;
    resultB.bytes = makeBytes(64, std::byte{22});
    fetcherA->setCanned({0, 0, 0, 0}, resultA);
    fetcherB->setCanned({0, 0, 0, 0}, resultB);
    auto cacheA = makeServiceCache(service, "source-a", fetcherA);
    auto cacheB = makeServiceCache(service, "source-b", fetcherB);

    auto a = cacheA->getChunkBlocking(0, 0, 0, 0);
    auto b = cacheB->getChunkBlocking(0, 0, 0, 0);
    REQUIRE(a.status == ChunkStatus::Data);
    REQUIRE(b.status == ChunkStatus::Data);
    CHECK(std::to_integer<int>((*a.bytes)[0]) == 11);
    CHECK(std::to_integer<int>((*b.bytes)[0]) == 22);

    cacheA->invalidate();
    CHECK(cacheA->getChunkIfCached(0, 0, 0, 0).status == ChunkStatus::MissQueued);
    CHECK(cacheB->getChunkIfCached(0, 0, 0, 0).status == ChunkStatus::Data);
}

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

    // Strict LRU retains only the two most recent chunks. Probe the expected
    // survivors BEFORE the evicted chunk: tryGetChunk on a missing entry
    // queues a background re-fetch, and its store evicts exactly the entries
    // asserted next (a race that flaked under parallel ctest load).
    CHECK(c->stats().decodedBytes <= 128);
    CHECK(c->tryGetChunk(0, 0, 1, 0).status == ChunkStatus::Data);
    CHECK(c->tryGetChunk(0, 0, 1, 1).status == ChunkStatus::Data);
    CHECK(c->tryGetChunk(0, 0, 0, 0).status == ChunkStatus::MissQueued);
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

struct SharedServiceFetchOrder {
    std::mutex mutex;
    std::condition_variable cv;
    std::latch firstStarted{1};
    bool released = false;
    std::vector<char> labels;
};

class LabeledServiceFetcher : public IChunkFetcher {
public:
    LabeledServiceFetcher(std::shared_ptr<SharedServiceFetchOrder> order,
                          char label, bool blockFirst)
        : order_(std::move(order)), label_(label), blockFirst_(blockFirst)
    {
    }

    ChunkFetchResult fetch(const ChunkKey&) override
    {
        {
            std::lock_guard lock(order_->mutex);
            order_->labels.push_back(label_);
        }
        if (blockFirst_) {
            order_->firstStarted.count_down();
            std::unique_lock lock(order_->mutex);
            order_->cv.wait(lock, [&] { return order_->released; });
            blockFirst_ = false;
        }
        ChunkFetchResult result;
        result.status = ChunkFetchStatus::Found;
        result.bytes = makeBytes(64, std::byte{41});
        return result;
    }

private:
    std::shared_ptr<SharedServiceFetchOrder> order_;
    char label_;
    bool blockFirst_;
};

} // namespace

TEST_CASE("ChunkCacheService prioritizes the active view across sources")
{
    auto service = std::make_shared<ChunkCacheService>(1024 * 1024);
    auto order = std::make_shared<SharedServiceFetchOrder>();
    auto fetcherB = std::make_shared<LabeledServiceFetcher>(order, 'B', false);
    auto fetcherA = std::make_shared<LabeledServiceFetcher>(order, 'A', true);
    auto cacheB = makeServiceCache(service, "priority-b", fetcherB, 1);
    auto cacheA = makeServiceCache(service, "priority-a", fetcherA, 1);

    CHECK(cacheA->tryGetChunk(0, 0, 0, 0).status == ChunkStatus::MissQueued);
    order->firstStarted.wait();
    CHECK(cacheA->tryGetChunk(0, 0, 0, 1).status == ChunkStatus::MissQueued);
    cacheB->replaceViewDemand({61, 1}, {0.0f, 0.0f}, {
        {{0, 0, 0, 0}, {0.0f, 0.0f}},
    });
    cacheB->updateViewFocus(61, {0.0f, 0.0f}, true);
    {
        std::lock_guard lock(order->mutex);
        order->released = true;
    }
    order->cv.notify_all();

    REQUIRE(waitForResolved(*cacheB, 0, 0, 0, 0).status == ChunkStatus::Data);
    REQUIRE(waitForResolved(*cacheA, 0, 0, 0, 1).status == ChunkStatus::Data);
    std::lock_guard lock(order->mutex);
    REQUIRE(order->labels.size() == 3);
    CHECK(order->labels == std::vector<char>{'A', 'B', 'A'});
}

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
    (void)c->tryGetChunk(0, 0, 0, 0); // duplicate demand counts once
    (void)c->tryGetChunk(0, 0, 0, 1); // fine (level 0)
    (void)c->tryGetChunk(1, 0, 0, 0); // coarse (level 1)
    const auto queued = c->stats().unresolvedFetchesByLevel;
    REQUIRE(queued.size() == 2);
    CHECK(queued[0] == 2);
    CHECK(queued[1] == 1);
    f->release();

    CHECK(waitForResolved(*c, 0, 0, 0, 1).status == ChunkStatus::Data);
    CHECK(waitForResolved(*c, 1, 0, 0, 0).status == ChunkStatus::Data);

    const auto order = f->order();
    REQUIRE(order.size() == 3);
    CHECK(order[1].level == 1); // coarse chunk jumped the fine one
    CHECK(order[2].level == 0);
    CHECK(c->stats().unresolvedFetchesByLevel ==
          std::vector<std::size_t>{0, 0});
}

TEST_CASE("ChunkCache: invalidation clears unresolved fetch counts")
{
    auto f = std::make_shared<BlockingOrderFetcher>();
    std::vector<ChunkCache::LevelInfo> levels = {
        {{4, 4, 8}, {4, 4, 4}, {}},
    };
    ChunkCache::Options opts;
    opts.maxConcurrentReads = 1;
    opts.detectAllFillChunks = false;
    auto c = std::make_shared<ChunkCache>(
        std::move(levels),
        std::vector<std::shared_ptr<vc::render::IChunkFetcher>>{f},
        0.0, ChunkDtype::UInt8, opts);

    (void)c->tryGetChunk(0, 0, 0, 0);
    f->waitFirstStarted();
    (void)c->tryGetChunk(0, 0, 0, 1);
    CHECK(c->stats().unresolvedFetchesByLevel ==
          std::vector<std::size_t>{2});

    c->invalidate();
    CHECK(c->stats().unresolvedFetchesByLevel ==
          std::vector<std::size_t>{0});
    f->release();
}

TEST_CASE("ChunkRequestScheduler reprioritizes pending keyed work")
{
    ChunkRequestScheduler scheduler(1);
    std::mutex mutex;
    std::condition_variable cv;
    bool release = false;
    std::latch started{1};
    std::vector<int> order;
    scheduler.submit(1, {}, 1, 0, [&] {
        started.count_down();
        std::unique_lock lock(mutex);
        cv.wait(lock, [&] { return release; });
        order.push_back(1);
    });
    started.wait();
    scheduler.submit(2, {}, 1, 0, [&] {
        std::lock_guard lock(mutex);
        order.push_back(2);
    });
    ChunkWorkPriority interactive;
    interactive.interactive = true;
    interactive.activeView = true;
    interactive.levelPriority = 0;
    interactive.distanceSquared = 4.0f;
    CHECK(scheduler.reprioritize(2, interactive));
    scheduler.submit(3, {}, 1, 0, [&] {
        std::lock_guard lock(mutex);
        order.push_back(3);
    });
    {
        std::lock_guard lock(mutex);
        release = true;
    }
    cv.notify_all();
    scheduler.waitIdle();
    CHECK(order == std::vector<int>{1, 2, 3});
}

TEST_CASE("ChunkRequestScheduler cancels pending keyed work only once")
{
    ChunkRequestScheduler scheduler(1);
    std::mutex mutex;
    std::condition_variable cv;
    bool release = false;
    std::latch started{1};
    std::vector<int> order;
    scheduler.submit(1, {}, 1, 0, [&] {
        started.count_down();
        std::unique_lock lock(mutex);
        cv.wait(lock, [&] { return release; });
        order.push_back(1);
    });
    started.wait();
    scheduler.submit(2, {}, 1, 0, [&] {
        std::lock_guard lock(mutex);
        order.push_back(2);
    });
    CHECK(scheduler.cancel(2));
    CHECK_FALSE(scheduler.cancel(2));
    CHECK_FALSE(scheduler.cancel(1));
    {
        std::lock_guard lock(mutex);
        release = true;
    }
    cv.notify_all();
    scheduler.waitIdle();
    CHECK(order == std::vector<int>{1});
}

TEST_CASE("ChunkRequestScheduler bounds interactive bursts")
{
    ChunkRequestScheduler scheduler(1, 3);
    std::mutex mutex;
    std::condition_variable cv;
    bool release = false;
    std::latch started{1};
    std::vector<int> order;
    scheduler.submit(1, {}, 1, 0, [&] {
        started.count_down();
        std::unique_lock lock(mutex);
        cv.wait(lock, [&] { return release; });
        order.push_back(1);
    });
    started.wait();
    ChunkWorkPriority gui;
    gui.interactive = true;
    gui.activeView = true;
    for (int id = 2; id <= 6; ++id) {
        scheduler.submit(std::uint64_t(id), gui, 1, 0, [&, id] {
            std::lock_guard lock(mutex);
            order.push_back(id);
        });
    }
    scheduler.submit(7, {}, 1, 0, [&] {
        std::lock_guard lock(mutex);
        order.push_back(7);
    });
    {
        std::lock_guard lock(mutex);
        release = true;
    }
    cv.notify_all();
    scheduler.waitIdle();
    REQUIRE(order.size() == 7);
    CHECK(order[4] == 7); // blocker, three GUI tasks, then background
}

TEST_CASE("ChunkRequestScheduler orders level then active view then distance")
{
    ChunkRequestScheduler scheduler(1);
    std::mutex mutex;
    std::condition_variable cv;
    bool release = false;
    std::latch started{1};
    std::vector<int> order;
    scheduler.submit(1, {}, 1, 0, [&] {
        started.count_down();
        std::unique_lock lock(mutex);
        cv.wait(lock, [&] { return release; });
        order.push_back(1);
    });
    started.wait();
    auto submit = [&](int id, bool active, int relativeLevel, float distance) {
        ChunkWorkPriority priority;
        priority.interactive = true;
        priority.activeView = active;
        priority.levelPriority = relativeLevel;
        priority.distanceSquared = distance;
        scheduler.submit(std::uint64_t(id), priority, 1, 0, [&, id] {
            std::lock_guard lock(mutex);
            order.push_back(id);
        });
    };
    submit(2, false, 2, 1.0f);   // inactive, coarse and near
    submit(3, true, 1, 1.0f);    // active, finer
    submit(4, true, 2, 100.0f);  // active, coarse and far
    submit(5, true, 2, 4.0f);    // active, coarse and near
    submit(6, true, 2, std::numeric_limits<float>::infinity());
    {
        std::lock_guard lock(mutex);
        release = true;
    }
    cv.notify_all();
    scheduler.waitIdle();
    CHECK(order == std::vector<int>{1, 5, 4, 6, 2, 3});
}

TEST_CASE("ChunkRequestScheduler estimates bandwidth before its admission window is full")
{
    using Clock = std::chrono::steady_clock;
    constexpr std::size_t chunkBytes = 2ULL * 1024ULL * 1024ULL;
    ChunkRequestScheduler::AdaptiveConcurrency adaptive;
    ChunkRequestScheduler scheduler(64, 7, {}, adaptive);

    const auto start = Clock::time_point{};
    scheduler.recordSuccessfulTransfer(
        chunkBytes, start, start + std::chrono::seconds(1));
    const auto stats = scheduler.transferStats();
    CHECK(stats.sampleCount == 1);
    CHECK(stats.bytesPerSecond == doctest::Approx(2.0 * 1024.0 * 1024.0));
    CHECK(stats.averageChunkBytes == doctest::Approx(double(chunkBytes)));
    CHECK(stats.admissionLimit == 2);
}

TEST_CASE("ChunkRequestScheduler probes upward with completion-paced admission")
{
    using Clock = std::chrono::steady_clock;
    constexpr std::size_t chunkBytes = 2ULL * 1024ULL * 1024ULL;
    ChunkRequestScheduler::AdaptiveConcurrency adaptive;
    adaptive.maximum = 16;
    adaptive.minimumEpochSeconds = 0.0;
    adaptive.maximumEpochSeconds = 60.0;
    adaptive.initialProbeMultiplier = 2;
    ChunkRequestScheduler scheduler(64, 7, {}, adaptive);
    auto cursor = Clock::time_point{};
    auto recordEpoch = [&](std::size_t concurrency,
                           double throughputMiB,
                           double latencySeconds) {
        const auto stats = scheduler.transferStats();
        REQUIRE(stats.admissionLimit == concurrency);
        REQUIRE(stats.targetAdmissionLimit == concurrency);
        const std::size_t count = std::max<std::size_t>(4, concurrency);
        const double windowSeconds =
            static_cast<double>(count * chunkBytes) /
            (throughputMiB * 1024.0 * 1024.0);
        REQUIRE(windowSeconds >= latencySeconds);
        const auto base = cursor + std::chrono::seconds(1);
        for (std::size_t sample = 0; sample < count; ++sample) {
            const double offset = count == 1
                ? 0.0
                : (windowSeconds - latencySeconds) *
                    static_cast<double>(sample) /
                    static_cast<double>(count - 1);
            const auto started = base + std::chrono::duration_cast<Clock::duration>(
                std::chrono::duration<double>(offset));
            scheduler.recordSuccessfulTransfer(
                chunkBytes,
                started,
                started + std::chrono::duration_cast<Clock::duration>(
                    std::chrono::duration<double>(latencySeconds)));
        }
        cursor = base + std::chrono::duration_cast<Clock::duration>(
            std::chrono::duration<double>(windowSeconds));
    };
    auto finishRamp = [&] {
        while (scheduler.transferStats().admissionLimit <
               scheduler.transferStats().targetAdmissionLimit) {
            const auto started = cursor + std::chrono::seconds(1);
            scheduler.recordSuccessfulTransfer(
                chunkBytes, started, started + std::chrono::milliseconds(10));
            cursor = started + std::chrono::milliseconds(10);
        }
    };
    auto selectDouble = [&](std::size_t concurrency) {
        recordEpoch(concurrency, 5.0 * concurrency, 0.10);
        const auto probing = scheduler.transferStats();
        CHECK(probing.targetAdmissionLimit == 2 * concurrency);
        CHECK(probing.admissionLimit == concurrency + 1);
        CHECK(probing.probing);
        finishRamp();
        recordEpoch(2 * concurrency, 9.0 * concurrency, 0.12);
        recordEpoch(concurrency, 5.0 * concurrency, 0.10);
        if (concurrency > adaptive.minimum) {
            recordEpoch(concurrency / 2, 3.0 * concurrency, 0.08);
            finishRamp();
            recordEpoch(concurrency, 5.0 * concurrency, 0.10);
        }
        finishRamp();
        CHECK(scheduler.transferStats().admissionLimit == 2 * concurrency);
    };

    selectDouble(2);
    selectDouble(4);
    selectDouble(8);
    const auto stats = scheduler.transferStats();
    CHECK(stats.admissionLimit == 16);
    CHECK(stats.targetAdmissionLimit == 16);
    CHECK(stats.longTermBytesPerSecond > 0.0);
}

TEST_CASE("ChunkRequestScheduler exploration cadence follows bandwidth stability")
{
    using Clock = std::chrono::steady_clock;
    constexpr std::size_t chunkBytes = 2ULL * 1024ULL * 1024ULL;
    ChunkRequestScheduler::AdaptiveConcurrency adaptive;
    adaptive.maximum = 4;
    adaptive.minimumEpochSeconds = 0.0;
    adaptive.maximumEpochSeconds = 60.0;
    adaptive.minimumStabilityObservationSeconds = 0.0;
    adaptive.initialProbeMultiplier = 2;
    adaptive.continuousSearchTurns = 1;
    ChunkRequestScheduler scheduler(4, 7, {}, adaptive);
    auto cursor = Clock::time_point{};
    auto recordEpoch = [&](std::size_t concurrency,
                           double throughputMiB,
                           double latencySeconds) {
        const std::size_t count = std::max<std::size_t>(4, concurrency);
        const double windowSeconds =
            static_cast<double>(count * chunkBytes) /
            (throughputMiB * 1024.0 * 1024.0);
        REQUIRE(windowSeconds >= latencySeconds);
        const auto base = cursor + std::chrono::seconds(1);
        for (std::size_t sample = 0; sample < count; ++sample) {
            const double offset = (windowSeconds - latencySeconds) *
                static_cast<double>(sample) / static_cast<double>(count - 1);
            const auto started = base + std::chrono::duration_cast<Clock::duration>(
                std::chrono::duration<double>(offset));
            scheduler.recordSuccessfulTransfer(
                chunkBytes,
                started,
                started + std::chrono::duration_cast<Clock::duration>(
                    std::chrono::duration<double>(latencySeconds)));
        }
        cursor = base + std::chrono::duration_cast<Clock::duration>(
            std::chrono::duration<double>(windowSeconds));
    };

    // The 4-worker probe has more throughput but excessive p90 latency, so C=2
    // is retained. This fixture disables the five-minute eligibility period to
    // exercise the stable/changed cadence calculation directly.
    recordEpoch(2, 10.0, 0.10);
    REQUIRE(scheduler.transferStats().admissionLimit == 3);
    const auto rampStarted = cursor + std::chrono::seconds(1);
    scheduler.recordSuccessfulTransfer(
        chunkBytes, rampStarted, rampStarted + std::chrono::milliseconds(10));
    cursor = rampStarted + std::chrono::milliseconds(10);
    recordEpoch(4, 12.0, 0.20);
    recordEpoch(2, 10.0, 0.10);
    auto stats = scheduler.transferStats();
    CHECK(stats.admissionLimit == 2);
    CHECK_FALSE(stats.probing);
    CHECK(stats.probeIntervalSeconds == doctest::Approx(300.0));

    recordEpoch(2, 10.0, 0.10);
    stats = scheduler.transferStats();
    CHECK(stats.probeIntervalSeconds == doctest::Approx(300.0));

    // A halving relative to the long-term EMA immediately shortens the next
    // exploration deadline to approximately one minute.
    recordEpoch(2, 5.0, 0.10);
    stats = scheduler.transferStats();
    CHECK(stats.probeIntervalSeconds == doctest::Approx(60.0));
    CHECK(stats.longTermBytesPerSecond > 5.0 * 1024.0 * 1024.0);
}

TEST_CASE("ChunkRequestScheduler requires saturated observation time for stability")
{
    using Clock = std::chrono::steady_clock;
    constexpr std::size_t chunkBytes = 2ULL * 1024ULL * 1024ULL;
    ChunkRequestScheduler::AdaptiveConcurrency adaptive;
    adaptive.minimum = 2;
    adaptive.maximum = 2;
    adaptive.minimumEpochSeconds = 0.0;
    adaptive.maximumEpochSeconds = 60.0;
    adaptive.continuousSearchTurns = 1;
    ChunkRequestScheduler scheduler(2, 7, {}, adaptive);
    const auto start = Clock::time_point{};
    for (int sample = 0; sample < 4; ++sample) {
        scheduler.recordSuccessfulTransfer(
            chunkBytes,
            start + std::chrono::milliseconds(sample * 200),
            start + std::chrono::milliseconds(sample * 200 + 100));
    }
    CHECK(scheduler.transferStats().probeIntervalSeconds ==
          doctest::Approx(60.0));
}

TEST_CASE("ChunkRequestScheduler retains saturated capacity when work drains")
{
    using Clock = std::chrono::steady_clock;
    constexpr std::size_t chunkBytes = 2ULL * 1024ULL * 1024ULL;
    ChunkRequestScheduler::AdaptiveConcurrency adaptive;
    adaptive.minimum = 2;
    adaptive.maximum = 2;
    adaptive.minimumEpochSeconds = 0.0;
    adaptive.maximumEpochSeconds = 60.0;
    adaptive.continuousSearchTurns = 1;
    ChunkRequestScheduler scheduler(2, 7, {}, adaptive);
    const auto start = Clock::time_point{};
    for (int sample = 0; sample < 4; ++sample) {
        scheduler.recordSuccessfulTransfer(
            chunkBytes,
            start + std::chrono::milliseconds(sample * 200),
            start + std::chrono::milliseconds(sample * 200 + 100));
    }
    const double saturatedBandwidth = scheduler.transferStats().bytesPerSecond;
    const double longTermBandwidth =
        scheduler.transferStats().longTermBytesPerSecond;

    scheduler.submit(1, {}, 0, 0, [&] {
        const auto underfilledStart = start + std::chrono::seconds(10);
        scheduler.recordSuccessfulTransfer(
            chunkBytes, underfilledStart,
            underfilledStart + std::chrono::seconds(4));
    });
    scheduler.waitIdle();

    CHECK(scheduler.transferStats().bytesPerSecond ==
          doctest::Approx(saturatedBandwidth));
    CHECK(scheduler.transferStats().longTermBytesPerSecond ==
          doctest::Approx(longTermBandwidth));
}

TEST_CASE("ChunkRequestScheduler starts continuous search with a fourfold probe")
{
    using Clock = std::chrono::steady_clock;
    constexpr std::size_t chunkBytes = 2ULL * 1024ULL * 1024ULL;
    ChunkRequestScheduler::AdaptiveConcurrency adaptive;
    adaptive.maximum = 16;
    adaptive.minimumEpochSeconds = 0.0;
    adaptive.maximumEpochSeconds = 60.0;
    ChunkRequestScheduler scheduler(16, 7, {}, adaptive);
    const auto start = Clock::time_point{};
    for (int sample = 0; sample < 4; ++sample) {
        scheduler.recordSuccessfulTransfer(
            chunkBytes,
            start + std::chrono::milliseconds(sample * 200),
            start + std::chrono::milliseconds(sample * 200 + 100));
    }
    const auto stats = scheduler.transferStats();
    CHECK(stats.admissionLimit == 3);
    CHECK(stats.targetAdmissionLimit == 8);
    CHECK(stats.probing);
}

TEST_CASE("ChunkRequestScheduler adaptive admission starts only its current limit")
{
    ChunkRequestScheduler::AdaptiveConcurrency adaptive;
    adaptive.maximum = 8;
    ChunkRequestScheduler scheduler(8, 7, {}, adaptive);
    std::mutex mutex;
    std::condition_variable cv;
    bool release = false;
    int started = 0;
    for (std::uint64_t id = 1; id <= 8; ++id) {
        scheduler.submit(id, {}, 1, 0, [&] {
            std::unique_lock lock(mutex);
            ++started;
            cv.notify_all();
            cv.wait(lock, [&] { return release; });
        });
    }
    {
        std::unique_lock lock(mutex);
        REQUIRE(cv.wait_for(lock, std::chrono::seconds(2), [&] {
            return started == 2;
        }));
        CHECK(scheduler.active() == 2);
        CHECK(scheduler.transferStats().admissionLimit == 2);
        release = true;
    }
    cv.notify_all();
    scheduler.waitIdle();
}

TEST_CASE("ChunkRequestScheduler fixed concurrency ignores transfer samples")
{
    using Clock = std::chrono::steady_clock;
    ChunkRequestScheduler scheduler(4);
    for (int sample = 0; sample < 16; ++sample) {
        scheduler.recordSuccessfulTransfer(
            2ULL * 1024ULL * 1024ULL, Clock::time_point{},
            Clock::time_point{} + std::chrono::milliseconds(320));
    }
    const auto stats = scheduler.transferStats();
    CHECK_FALSE(stats.adaptive);
    CHECK(stats.admissionLimit == 4);
    CHECK(stats.sampleCount == 16);
    CHECK(stats.bytesPerSecond ==
          doctest::Approx(100.0 * 1024.0 * 1024.0));
}

TEST_CASE("ChunkRequestScheduler publishes shared priority updates atomically")
{
    auto gate = std::make_shared<ChunkRequestSelectionGate>();
    ChunkRequestScheduler probeScheduler(1, 7, gate);
    ChunkRequestScheduler fetchScheduler(1, 7, gate);
    ChunkRequestScheduler decodeScheduler(1, 7, gate);
    std::mutex mutex;
    std::condition_variable cv;
    bool release = false;
    std::latch blockersStarted{3};
    std::latch blockersFinished{3};
    std::atomic_bool publishing{false};
    std::atomic_int selectedDuringPublication{0};

    auto submitBlocker = [&](ChunkRequestScheduler& scheduler, std::uint64_t id) {
        scheduler.submit(id, {}, 1, 0, [&] {
            blockersStarted.count_down();
            std::unique_lock lock(mutex);
            cv.wait(lock, [&] { return release; });
            lock.unlock();
            blockersFinished.count_down();
        });
    };
    submitBlocker(probeScheduler, 1);
    submitBlocker(fetchScheduler, 2);
    submitBlocker(decodeScheduler, 5);
    blockersStarted.wait();

    auto submitFollower = [&](ChunkRequestScheduler& scheduler, std::uint64_t id) {
        ChunkWorkPriority priority;
        priority.interactive = true;
        scheduler.submit(id, priority, 1, 0, [&] {
            if (publishing.load(std::memory_order_acquire))
                selectedDuringPublication.fetch_add(1, std::memory_order_relaxed);
        });
    };
    submitFollower(probeScheduler, 3);
    submitFollower(fetchScheduler, 4);
    submitFollower(decodeScheduler, 6);

    gate->publish([&] {
        publishing.store(true, std::memory_order_release);
        {
            std::lock_guard lock(mutex);
            release = true;
        }
        cv.notify_all();
        blockersFinished.wait();
        // Give both workers an opportunity to request their next task while
        // the shared publication remains deliberately incomplete.
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        CHECK(selectedDuringPublication.load(std::memory_order_relaxed) == 0);
        publishing.store(false, std::memory_order_release);
    });

    probeScheduler.waitIdle();
    fetchScheduler.waitIdle();
    decodeScheduler.waitIdle();
    CHECK(selectedDuringPublication.load(std::memory_order_relaxed) == 0);
}

TEST_CASE("ChunkCache view snapshots promote queued work and reject stale replacement")
{
    auto fetcher = std::make_shared<BlockingOrderFetcher>();
    std::vector<ChunkCache::LevelInfo> levels = {
        {{4, 4, 16}, {4, 4, 4}, {}},
    };
    ChunkCache::Options options;
    options.maxConcurrentReads = 1;
    options.detectAllFillChunks = false;
    auto cache = std::make_shared<ChunkCache>(
        std::move(levels),
        std::vector<std::shared_ptr<IChunkFetcher>>{fetcher},
        0.0, ChunkDtype::UInt8, options);

    (void)cache->tryGetChunk(0, 0, 0, 0);
    fetcher->waitFirstStarted();
    (void)cache->tryGetChunk(0, 0, 0, 1); // initially background
    (void)cache->tryGetChunk(0, 0, 0, 2); // remains background

    vc::render::ChunkRequestContext current{41, 2};
    cache->replaceViewDemand(current, {10.0f, 10.0f}, {
        {{0, 0, 0, 1}, {10.0f, 10.0f}},
    });
    cache->updateViewFocus(41, {10.0f, 10.0f}, true);
    cache->replaceViewDemand({41, 1}, {0.0f, 0.0f}, {
        {{0, 0, 0, 3}, {0.0f, 0.0f}},
    });

    fetcher->release();
    CHECK(waitForResolved(*cache, 0, 0, 0, 1).status == ChunkStatus::Data);
    CHECK(waitForResolved(*cache, 0, 0, 0, 2).status == ChunkStatus::Data);
    const auto order = fetcher->order();
    REQUIRE(order.size() == 3);
    CHECK(order[0].ix == 0);
    CHECK(order[1].ix == 1);
    CHECK(order[2].ix == 2);
    CHECK(cache->getChunkIfCached(0, 0, 0, 3).status == ChunkStatus::MissQueued);
}

TEST_CASE("ChunkCache superseded view demand cancels pending GUI work")
{
    auto fetcher = std::make_shared<BlockingOrderFetcher>();
    std::vector<ChunkCache::LevelInfo> levels = {
        {{4, 4, 24}, {4, 4, 4}, {}},
    };
    ChunkCache::Options options;
    options.maxConcurrentReads = 1;
    options.detectAllFillChunks = false;
    auto cache = std::make_shared<ChunkCache>(
        std::move(levels),
        std::vector<std::shared_ptr<IChunkFetcher>>{fetcher},
        0.0, ChunkDtype::UInt8, options);

    (void)cache->tryGetChunk(0, 0, 0, 0); // running background blocker
    fetcher->waitFirstStarted();
    cache->replaceViewDemand({81, 1}, {0.0f, 0.0f}, {
        {{0, 0, 0, 1}, {0.0f, 0.0f}},
        {{0, 0, 0, 2}, {1.0f, 0.0f}},
    });
    CHECK(cache->stats().unresolvedFetchesByLevel ==
          std::vector<std::size_t>{3});

    cache->replaceViewDemand({81, 2}, {0.0f, 0.0f}, {
        {{0, 0, 0, 3}, {0.0f, 0.0f}},
    });
    CHECK(cache->stats().unresolvedFetchesByLevel ==
          std::vector<std::size_t>{2});

    fetcher->release();
    CHECK(waitForResolved(*cache, 0, 0, 0, 3).status == ChunkStatus::Data);
    CHECK(fetcher->order() == std::vector<ChunkKey>{
        {0, 0, 0, 0}, {0, 0, 0, 3}});
}

TEST_CASE("ChunkCache closing a view cancels its pending GUI work")
{
    auto fetcher = std::make_shared<BlockingOrderFetcher>();
    std::vector<ChunkCache::LevelInfo> levels = {
        {{4, 4, 16}, {4, 4, 4}, {}},
    };
    ChunkCache::Options options;
    options.maxConcurrentReads = 1;
    options.detectAllFillChunks = false;
    auto cache = std::make_shared<ChunkCache>(
        std::move(levels),
        std::vector<std::shared_ptr<IChunkFetcher>>{fetcher},
        0.0, ChunkDtype::UInt8, options);

    (void)cache->tryGetChunk(0, 0, 0, 0);
    fetcher->waitFirstStarted();
    cache->replaceViewDemand({82, 1}, {0.0f, 0.0f}, {
        {{0, 0, 0, 1}, {0.0f, 0.0f}},
        {{0, 0, 0, 2}, {1.0f, 0.0f}},
    });
    cache->clearViewDemand(82, 1);
    CHECK(cache->stats().unresolvedFetchesByLevel ==
          std::vector<std::size_t>{1});
    (void)cache->tryGetChunk(0, 0, 0, 3, {82, 1}); // late closed-view fill
    CHECK(cache->stats().unresolvedFetchesByLevel ==
          std::vector<std::size_t>{1});
    fetcher->release();
    cache->getChunkBlocking(0, 0, 0, 0);
    CHECK(fetcher->order() == std::vector<ChunkKey>{{0, 0, 0, 0}});

    cache->replaceViewDemand({82, 2}, {0.0f, 0.0f}, {
        {{0, 0, 0, 3}, {0.0f, 0.0f}},
    });
    CHECK(waitForResolved(*cache, 0, 0, 0, 3).status == ChunkStatus::Data);
    CHECK(fetcher->order() == std::vector<ChunkKey>{
        {0, 0, 0, 0}, {0, 0, 0, 3}});
}

TEST_CASE("ChunkCache stale running download does not enter decode queue")
{
    auto fetcher = std::make_shared<BlockingEncodedFetcher>();
    std::vector<ChunkCache::LevelInfo> levels = {
        {{4, 4, 8}, {4, 4, 4}, {}},
    };
    ChunkCache::Options options;
    options.maxConcurrentReads = 1;
    options.detectAllFillChunks = false;
    auto cache = std::make_shared<ChunkCache>(
        std::move(levels),
        std::vector<std::shared_ptr<IChunkFetcher>>{fetcher},
        0.0, ChunkDtype::UInt8, options);

    cache->replaceViewDemand({85, 1}, {0.0f, 0.0f}, {
        {{0, 0, 0, 0}, {0.0f, 0.0f}},
    });
    fetcher->waitStarted();
    cache->replaceViewDemand({85, 2}, {0.0f, 0.0f}, {
        {{0, 0, 0, 1}, {0.0f, 0.0f}},
    });
    fetcher->release();
    CHECK(waitForResolved(*cache, 0, 0, 0, 1).status == ChunkStatus::Data);
    CHECK(fetcher->decodeCalls.load() == 1);
    CHECK(cache->stats().unresolvedFetchesByLevel ==
          std::vector<std::size_t>{0});
}

TEST_CASE("ChunkCache preserves pending work owned by another view or background")
{
    auto fetcher = std::make_shared<BlockingOrderFetcher>();
    std::vector<ChunkCache::LevelInfo> levels = {
        {{4, 4, 16}, {4, 4, 4}, {}},
    };
    ChunkCache::Options options;
    options.maxConcurrentReads = 1;
    options.detectAllFillChunks = false;
    auto cache = std::make_shared<ChunkCache>(
        std::move(levels),
        std::vector<std::shared_ptr<IChunkFetcher>>{fetcher},
        0.0, ChunkDtype::UInt8, options);

    (void)cache->tryGetChunk(0, 0, 0, 0);
    fetcher->waitFirstStarted();
    cache->replaceViewDemand({83, 1}, {0.0f, 0.0f}, {
        {{0, 0, 0, 1}, {0.0f, 0.0f}},
        {{0, 0, 0, 2}, {1.0f, 0.0f}},
    });
    cache->replaceViewDemand({84, 1}, {0.0f, 0.0f}, {
        {{0, 0, 0, 1}, {0.0f, 0.0f}},
    });
    (void)cache->tryGetChunk(0, 0, 0, 2); // explicit background owner

    cache->clearViewDemand(83, 1);
    CHECK(cache->stats().unresolvedFetchesByLevel ==
          std::vector<std::size_t>{3});
    cache->clearViewDemand(84, 1);
    CHECK(cache->stats().unresolvedFetchesByLevel ==
          std::vector<std::size_t>{2});

    fetcher->release();
    CHECK(waitForResolved(*cache, 0, 0, 0, 2).status == ChunkStatus::Data);
    CHECK(fetcher->order() == std::vector<ChunkKey>{
        {0, 0, 0, 0}, {0, 0, 0, 2}});
}

TEST_CASE("ChunkCache selects the coarsest view-relative demand first")
{
    auto fetcher = std::make_shared<BlockingOrderFetcher>();
    std::vector<ChunkCache::LevelInfo> levels(4);
    for (auto& level : levels) {
        level.shape = {4, 4, 12};
        level.chunkShape = {4, 4, 4};
    }
    ChunkCache::Options options;
    options.maxConcurrentReads = 1;
    options.detectAllFillChunks = false;
    auto cache = std::make_shared<ChunkCache>(
        std::move(levels),
        std::vector<std::shared_ptr<IChunkFetcher>>(4, fetcher),
        0.0, ChunkDtype::UInt8, options);

    (void)cache->tryGetChunk(0, 0, 0, 0);
    fetcher->waitFirstStarted();

    // Both chunks use absolute level 2. The active view sees the first as its
    // second fallback. Another view sees the shared second chunk as its third
    // fallback, so that chunk must be selected first despite being inactive.
    cache->replaceViewDemand({71, 1}, {0.0f, 0.0f}, {
        {{2, 0, 0, 0}, {0.0f, 0.0f}, 2},
    });
    cache->updateViewFocus(71, {0.0f, 0.0f}, true);
    cache->replaceViewDemand({72, 1}, {0.0f, 0.0f}, {
        {{2, 0, 0, 1}, {0.0f, 0.0f}, 0},
    });
    cache->replaceViewDemand({73, 1}, {0.0f, 0.0f}, {
        {{2, 0, 0, 1}, {0.0f, 0.0f}, 3},
    });

    fetcher->release();
    CHECK(waitForResolved(*cache, 2, 0, 0, 1).status == ChunkStatus::Data);
    CHECK(waitForResolved(*cache, 2, 0, 0, 0).status == ChunkStatus::Data);
    const auto order = fetcher->order();
    REQUIRE(order.size() == 3);
    CHECK(order[0] == ChunkKey{0, 0, 0, 0});
    CHECK(order[1] == ChunkKey{2, 0, 0, 1});
    CHECK(order[2] == ChunkKey{2, 0, 0, 0});
}

TEST_CASE("ChunkCache selects the terminal source level before ordinary fallbacks")
{
    auto fetcher = std::make_shared<BlockingOrderFetcher>();
    std::vector<ChunkCache::LevelInfo> levels(4);
    for (auto& level : levels) {
        level.shape = {4, 4, 12};
        level.chunkShape = {4, 4, 4};
    }
    ChunkCache::Options options;
    options.maxConcurrentReads = 1;
    options.detectAllFillChunks = false;
    auto cache = std::make_shared<ChunkCache>(
        std::move(levels),
        std::vector<std::shared_ptr<IChunkFetcher>>(4, fetcher),
        0.0, ChunkDtype::UInt8, options);

    (void)cache->tryGetChunk(0, 0, 0, 0);
    fetcher->waitFirstStarted();

    cache->replaceViewDemand({74, 1}, {0.0f, 0.0f}, {
        {{3, 0, 0, 0}, {0.0f, 0.0f}, 0},
    });
    cache->updateViewFocus(74, {0.0f, 0.0f}, true);
    cache->replaceViewDemand({75, 1}, {0.0f, 0.0f}, {
        {{3, 0, 0, 1}, {0.0f, 0.0f}, 3},
    });
    cache->replaceViewDemand({76, 1}, {0.0f, 0.0f}, {
        {{2, 0, 0, 2}, {0.0f, 0.0f}, 5},
    });

    fetcher->release();
    CHECK(waitForResolved(*cache, 3, 0, 0, 1).status == ChunkStatus::Data);
    CHECK(waitForResolved(*cache, 3, 0, 0, 0).status == ChunkStatus::Data);
    CHECK(waitForResolved(*cache, 2, 0, 0, 2).status == ChunkStatus::Data);
    const auto order = fetcher->order();
    REQUIRE(order.size() == 4);
    CHECK(order[0] == ChunkKey{0, 0, 0, 0});
    CHECK(order[1] == ChunkKey{3, 0, 0, 1});
    CHECK(order[2] == ChunkKey{3, 0, 0, 0});
    CHECK(order[3] == ChunkKey{2, 0, 0, 2});
}

TEST_CASE("ChunkCache rejects stale asynchronous GUI misses")
{
    auto fetcher = std::make_shared<BlockingOrderFetcher>();
    std::vector<ChunkCache::LevelInfo> levels = {
        {{4, 4, 20}, {4, 4, 4}, {}},
    };
    ChunkCache::Options options;
    options.maxConcurrentReads = 1;
    options.detectAllFillChunks = false;
    auto cache = std::make_shared<ChunkCache>(
        std::move(levels),
        std::vector<std::shared_ptr<IChunkFetcher>>{fetcher},
        0.0, ChunkDtype::UInt8, options);

    (void)cache->tryGetChunk(0, 0, 0, 0);
    fetcher->waitFirstStarted();
    (void)cache->tryGetChunk(0, 0, 0, 2); // background, queued first

    cache->replaceViewDemand({51, 2}, {0.0f, 0.0f}, {
        {{0, 0, 0, 1}, {0.0f, 0.0f}},
    });
    cache->updateViewFocus(51, {0.0f, 0.0f}, true);
    (void)cache->tryGetChunk(0, 0, 0, 3, {51, 1}); // stale async fill

    fetcher->release();
    CHECK(waitForResolved(*cache, 0, 0, 0, 1).status == ChunkStatus::Data);
    CHECK(waitForResolved(*cache, 0, 0, 0, 2).status == ChunkStatus::Data);
    CHECK(fetcher->order() == std::vector<ChunkKey>{
        {0, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 0, 2}});
    CHECK(cache->getChunkIfCached(0, 0, 0, 3).status == ChunkStatus::MissQueued);
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

TEST_CASE("ChunkCache classifies persistent misses while cached decodes are blocked")
{
    std::mt19937_64 rng(std::random_device{}());
    const auto dir = fs::temp_directory_path() /
                     ("vc_chunk_split_stages_" + std::to_string(rng()));
    fs::create_directories(dir);

    const auto encoded = std::vector<std::byte>{std::byte{11}};
    for (int ix = 0; ix < 16; ++ix) {
        writeTestBytes(
            dir / "level_0" / "0" / "0" /
                (std::to_string(ix) + ".encoded"),
            encoded);
    }

    auto fetcher = std::make_shared<SplitStageFetcher>();
    {
        std::vector<ChunkCache::LevelInfo> levels = {
            {{4, 4, 68}, {4, 4, 4}, {}},
        };
        ChunkCache::Options options;
        options.persistentCachePath = dir;
        options.maxConcurrentReads = 1;
        options.detectAllFillChunks = false;
        ChunkCache cache(
            std::move(levels),
            std::vector<std::shared_ptr<IChunkFetcher>>{fetcher},
            0.0, ChunkDtype::UInt8, options);

        for (int ix = 0; ix < 16; ++ix)
            (void)cache.tryGetChunk(0, 0, 0, ix);
        (void)cache.tryGetChunk(0, 0, 0, 16);

        // The old combined probe/decode pool blocked before classifying this
        // miss. The split 32-worker stat stage must admit its remote GET while
        // every CPU decode worker is occupied by cached data.
        const bool remoteStarted = fetcher->waitForRemote(std::chrono::seconds{2});
        fetcher->releasePersistentDecodes();
        CHECK(remoteStarted);
        CHECK(fetcher->remoteCalls() == 1);
        CHECK(waitForResolved(cache, 0, 0, 0, 16).status == ChunkStatus::Data);
        CHECK(fetcher->waitForRemoteDecode(std::chrono::seconds{2}));
        CHECK(fetcher->remoteAndDecodeUsedDifferentThreads());
    }

    fs::remove_all(dir);
}

TEST_CASE("ChunkCache reprioritizes pending decode work by view-relative level")
{
    std::mt19937_64 rng(std::random_device{}());
    const auto dir = fs::temp_directory_path() /
                     ("vc_chunk_decode_priority_" + std::to_string(rng()));
    const auto encoded = std::vector<std::byte>{std::byte{11}};
    for (int ix = 0; ix < 9; ++ix) {
        writeTestBytes(
            dir / "level_0" / "0" / "0" /
                (std::to_string(ix) + ".encoded"),
            encoded);
    }
    writeTestBytes(dir / "level_2" / "0" / "0" / "0.encoded", encoded);

    auto fetcher = std::make_shared<SplitStageFetcher>();
    {
        std::vector<ChunkCache::LevelInfo> levels(3);
        for (auto& level : levels) {
            level.shape = {4, 4, 36};
            level.chunkShape = {4, 4, 4};
        }
        ChunkCache::Options options;
        options.persistentCachePath = dir;
        options.maxConcurrentReads = 1;
        options.detectAllFillChunks = false;
        ChunkCache cache(
            std::move(levels),
            std::vector<std::shared_ptr<IChunkFetcher>>(3, fetcher),
            0.0, ChunkDtype::UInt8, options);

        for (int ix = 0; ix < 8; ++ix)
            (void)cache.tryGetChunk(0, 0, 0, ix);
        REQUIRE(fetcher->waitForPersistentDecodes(8, std::chrono::seconds{2}));

        cache.replaceViewDemand({91, 1}, {0.0f, 0.0f}, {
            {{0, 0, 0, 8}, {0.0f, 0.0f}, 0},
            {{2, 0, 0, 0}, {0.0f, 0.0f}, 2},
        });
        cache.updateViewFocus(91, {0.0f, 0.0f}, true);
        const auto deadline = std::chrono::steady_clock::now() +
            std::chrono::seconds{2};
        while (cache.stats().pendingDecodeTasks < 2 &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds{1});
        }
        REQUIRE(cache.stats().pendingDecodeTasks >= 2);
        fetcher->releasePersistentDecodes(1);
        REQUIRE(fetcher->waitForPersistentDecodes(9, std::chrono::seconds{2}));
        const auto order = fetcher->persistentDecodeOrder();
        REQUIRE(order.size() >= 9);
        CHECK(order[8] == ChunkKey{2, 0, 0, 0});
        fetcher->releasePersistentDecodes();
    }

    fs::remove_all(dir);
}

TEST_CASE("ChunkCache invalidation cancels pending decode work")
{
    std::mt19937_64 rng(std::random_device{}());
    const auto dir = fs::temp_directory_path() /
                     ("vc_chunk_decode_invalidate_" + std::to_string(rng()));
    const auto encoded = std::vector<std::byte>{std::byte{11}};
    for (int ix = 0; ix < 9; ++ix) {
        writeTestBytes(
            dir / "level_0" / "0" / "0" /
                (std::to_string(ix) + ".encoded"),
            encoded);
    }

    auto fetcher = std::make_shared<SplitStageFetcher>();
    {
        std::vector<ChunkCache::LevelInfo> levels = {
            {{4, 4, 36}, {4, 4, 4}, {}},
        };
        ChunkCache::Options options;
        options.persistentCachePath = dir;
        options.maxConcurrentReads = 1;
        options.detectAllFillChunks = false;
        ChunkCache cache(
            std::move(levels),
            std::vector<std::shared_ptr<IChunkFetcher>>{fetcher},
            0.0, ChunkDtype::UInt8, options);

        for (int ix = 0; ix < 8; ++ix)
            (void)cache.tryGetChunk(0, 0, 0, ix);
        REQUIRE(fetcher->waitForPersistentDecodes(8, std::chrono::seconds{2}));
        (void)cache.tryGetChunk(0, 0, 0, 8);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        cache.invalidate();
        fetcher->releasePersistentDecodes();
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        CHECK(fetcher->persistentDecodeOrder().size() == 8);
    }

    fs::remove_all(dir);
}
