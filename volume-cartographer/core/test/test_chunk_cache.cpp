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
#include <future>
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

class MultiBlockingFetcher : public IChunkFetcher {
public:
    ChunkFetchResult fetch(const ChunkKey& key) override
    {
        {
            std::lock_guard lock(mutex_);
            ++calls_[key];
            ++started_;
        }
        cv_.notify_all();
        {
            std::unique_lock lock(mutex_);
            cv_.wait(lock, [&] { return released_; });
        }
        ChunkFetchResult result;
        result.status = ChunkFetchStatus::Found;
        result.bytes = std::vector<std::byte>(64, std::byte{17});
        return result;
    }

    bool waitForStarted(std::size_t count,
                        std::chrono::milliseconds timeout)
    {
        std::unique_lock lock(mutex_);
        return cv_.wait_for(lock, timeout, [&] { return started_ >= count; });
    }

    int calls(const ChunkKey& key) const
    {
        std::lock_guard lock(mutex_);
        const auto found = calls_.find(key);
        return found == calls_.end() ? 0 : found->second;
    }

    void release()
    {
        {
            std::lock_guard lock(mutex_);
            released_ = true;
        }
        cv_.notify_all();
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::unordered_map<ChunkKey, int, vc::render::ChunkKeyHash> calls_;
    std::size_t started_ = 0;
    bool released_ = false;
};

// Scope guard for tests that park readers on a MultiBlockingFetcher: declare
// it AFTER the reader futures so it destructs first, releasing the fetcher
// before a fatal assertion's unwinding blocks in a future destructor.
// MultiBlockingFetcher::release() is idempotent.
struct FetcherReleaseGuard {
    explicit FetcherReleaseGuard(MultiBlockingFetcher& fetcher)
        : fetcher_(fetcher)
    {
    }
    ~FetcherReleaseGuard() { fetcher_.release(); }
    FetcherReleaseGuard(const FetcherReleaseGuard&) = delete;
    FetcherReleaseGuard& operator=(const FetcherReleaseGuard&) = delete;

private:
    MultiBlockingFetcher& fetcher_;
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

class PersistentSourceFetcher : public IChunkFetcher {
public:
    ChunkFetchResult fetch(const ChunkKey& key) override
    {
        return decodeFetched(key, fetchEncoded(key));
    }

    ChunkFetchResult fetchEncoded(const ChunkKey&) override
    {
        ++fetchCalls;
        {
            std::unique_lock lock(mutex_);
            ++started_;
            cv_.notify_all();
            cv_.wait(lock, [&] { return !blocking_ || released_; });
            return encoded_;
        }
    }

    ChunkFetchResult decodeFetched(
        const ChunkKey&, ChunkFetchResult fetched) const override
    {
        ++decodeCalls;
        if (fetched.status != ChunkFetchStatus::Found)
            return fetched;
        ChunkFetchResult decoded;
        decoded.status = ChunkFetchStatus::Found;
        decoded.bytes = makeDecoded(fetched.bytes);
        return decoded;
    }

    bool supportsSourcePayloadPersistence(const ChunkKey&) const override
    {
        return true;
    }

    ChunkFetchResult decodeSourcePayload(
        const ChunkKey&, std::vector<std::byte> bytes) const override
    {
        ++sourceDecodeCalls;
        ChunkFetchResult decoded;
        decoded.status = ChunkFetchStatus::Found;
        decoded.bytes = makeDecoded(bytes);
        return decoded;
    }

    void setEncoded(ChunkFetchResult result)
    {
        std::lock_guard lock(mutex_);
        encoded_ = std::move(result);
    }

    void block()
    {
        std::lock_guard lock(mutex_);
        blocking_ = true;
        released_ = false;
    }

    bool waitForStarted(int count, std::chrono::milliseconds timeout)
    {
        std::unique_lock lock(mutex_);
        return cv_.wait_for(lock, timeout, [&] { return started_ >= count; });
    }

    void release()
    {
        {
            std::lock_guard lock(mutex_);
            released_ = true;
        }
        cv_.notify_all();
    }

    std::atomic<int> fetchCalls{0};
    mutable std::atomic<int> decodeCalls{0};
    mutable std::atomic<int> sourceDecodeCalls{0};

private:
    static std::vector<std::byte> makeDecoded(
        const std::vector<std::byte>& encoded)
    {
        const auto value = encoded.empty() ? std::byte{0} : encoded.front();
        return std::vector<std::byte>(64, value);
    }

    std::mutex mutex_;
    std::condition_variable cv_;
    ChunkFetchResult encoded_{
        ChunkFetchStatus::Found, {std::byte{71}}};
    int started_ = 0;
    bool blocking_ = false;
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

ChunkCacheService::Options serviceOptions(
    std::size_t decodedByteCapacity = 1024 * 1024,
    std::size_t maxConcurrentReads = 16,
    bool adaptive = false)
{
    ChunkCacheService::Options options;
    options.decodedByteCapacity = decodedByteCapacity;
    options.fetchConcurrency.workerCapacity = std::max<std::size_t>(
        8, maxConcurrentReads);
    options.fetchConcurrency.maxConcurrentReads = maxConcurrentReads;
    options.fetchConcurrency.adaptive = adaptive;
    return options;
}

std::shared_ptr<ChunkCacheService> makeService(
    std::size_t decodedByteCapacity = 1024 * 1024,
    std::size_t maxConcurrentReads = 16,
    bool adaptive = false)
{
    return std::make_shared<ChunkCacheService>(
        serviceOptions(decodedByteCapacity, maxConcurrentReads, adaptive));
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

std::vector<std::byte> readTestBytes(const fs::path& path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    REQUIRE(file.good());
    const auto size = file.tellg();
    REQUIRE(size >= 0);
    std::vector<std::byte> bytes(static_cast<std::size_t>(size));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(bytes.data()), size);
    REQUIRE(file.good());
    return bytes;
}

std::shared_ptr<ChunkCache> makeCache(std::shared_ptr<CountingFetcher> f,
                                       std::array<int, 3> shape = {8, 8, 8},
                                       std::array<int, 3> chunkShape = {4, 4, 4},
                                       double fillValue = 0.0,
                                       ChunkDtype dtype = ChunkDtype::UInt8)
{
    std::vector<ChunkCache::LevelInfo> levels = {
        {shape, chunkShape, {}},
    };
    ChunkCache::Options opts;
    opts.detectAllFillChunks = true;
    auto cacheServiceOptions = serviceOptions(512ULL * 1024ULL * 1024ULL, 4);
    return std::make_shared<ChunkCache>(
        std::move(levels),
        std::vector<std::shared_ptr<vc::render::IChunkFetcher>>{f},
        fillValue,
        dtype,
        opts, std::move(cacheServiceOptions));
}

std::shared_ptr<ChunkCache> makeServiceCache(
    const std::shared_ptr<ChunkCacheService>& service,
    std::string identity,
    const std::shared_ptr<IChunkFetcher>& fetcher,
    std::size_t maxConcurrentReads = 0,
    bool adaptiveConcurrentReads = false)
{
    std::vector<ChunkCache::LevelInfo> levels = {
        {{8, 8, 8}, {4, 4, 4}, {}},
    };
    ChunkCache::Options options;
    options.detectAllFillChunks = false;
    if (maxConcurrentReads != 0) {
        service->configureFetchConcurrency(
            maxConcurrentReads, adaptiveConcurrentReads);
    }
    return service->acquireSource(
        std::move(identity), std::move(levels),
        std::vector<std::shared_ptr<IChunkFetcher>>{fetcher},
        0.0, ChunkDtype::UInt8, std::move(options));
}

std::shared_ptr<ChunkCache> makePersistentServiceCache(
    const std::shared_ptr<ChunkCacheService>& service,
    std::string identity,
    const std::shared_ptr<IChunkFetcher>& fetcher,
    const fs::path& persistentPath)
{
    std::vector<ChunkCache::LevelInfo> levels = {
        {{8, 8, 8}, {4, 4, 4}, {}},
    };
    ChunkCache::Options options;
    options.detectAllFillChunks = false;
    options.persistentCachePath = persistentPath;
    return service->acquireSource(
        std::move(identity), std::move(levels),
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
    auto service = makeService();
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

TEST_CASE("Delta3D cache mode permits sources without persistent caching")
{
    auto options = serviceOptions();
    options.persistentCacheEncoding =
        vc::render::PersistentCacheEncoding::Delta3dLossless;
    auto service = std::make_shared<ChunkCacheService>(std::move(options));
    auto fetcher = std::make_shared<CountingFetcher>();

    auto cache = makeServiceCache(
        service, "local-with-delta3d-process-mode", fetcher);

    REQUIRE(cache);
    CHECK(cache->persistentCacheLayout() ==
          vc::render::PersistentCacheLayout::Legacy);
}

TEST_CASE("ChunkCacheService carries adaptive download state into its shared scheduler")
{
    const ChunkCacheService::AdaptiveDownloadState initial{
        12, 48.0 * 1024.0 * 1024.0, 8, 6.0 * 1024.0 * 1024.0};
    auto options = serviceOptions(1024 * 1024, 16, true);
    options.initialAdaptiveDownloadState = initial;
    auto service = std::make_shared<ChunkCacheService>(std::move(options));
    auto fetcher = std::make_shared<CountingFetcher>();
    auto cache = makeServiceCache(
        service, "adaptive-restore", fetcher);
    REQUIRE(cache);

    const auto restored = service->adaptiveDownloadState();
    REQUIRE(restored);
    CHECK(restored->settledAdmissionLimit == 12);
    CHECK(restored->longTermBytesPerSecond ==
          doctest::Approx(48.0 * 1024.0 * 1024.0));
    CHECK(restored->maximumSaturatedParallelism == 8);
    CHECK(restored->saturatedBytesPerSecondPerWorker ==
          doctest::Approx(6.0 * 1024.0 * 1024.0));
}

TEST_CASE("ChunkCacheService rejects incompatible duplicate source metadata")
{
    auto service = makeService(1024 * 1024, 4);
    auto fetcher = std::make_shared<CountingFetcher>();
    auto first = makeServiceCache(service, "same-source", fetcher);
    std::vector<ChunkCache::LevelInfo> incompatibleLevels = {
        {{16, 8, 8}, {4, 4, 4}, {}},
    };
    ChunkCache::Options options;
    options.detectAllFillChunks = false;
    CHECK_THROWS_AS(
        service->acquireSource(
            "same-source", std::move(incompatibleLevels),
            std::vector<std::shared_ptr<IChunkFetcher>>{fetcher},
            0.0, ChunkDtype::UInt8, std::move(options)),
        std::invalid_argument);
    const auto concurrency = service->fetchConcurrency();
    CHECK(concurrency.maxConcurrentReads == 4);
    CHECK_FALSE(concurrency.adaptive);
}

TEST_CASE("ChunkCacheService source acquisition cannot change concurrency")
{
    auto service = makeService(1024 * 1024, 4);
    auto fetcher = std::make_shared<CountingFetcher>();
    auto first = makeServiceCache(service, "policy-source", fetcher);
    auto second = makeServiceCache(service, "policy-source", fetcher);

    CHECK(first->sourceId() == second->sourceId());
    CHECK(service->sourceCount() == 1);
    auto concurrency = service->fetchConcurrency();
    CHECK(concurrency.maxConcurrentReads == 4);
    CHECK_FALSE(concurrency.adaptive);

    service->configureFetchConcurrency(3, false);
    concurrency = service->fetchConcurrency();
    CHECK(concurrency.maxConcurrentReads == 3);
    CHECK_FALSE(concurrency.adaptive);

    service->configureFetchConcurrency(4, true);
    concurrency = service->fetchConcurrency();
    CHECK(concurrency.maxConcurrentReads == 4);
    CHECK(concurrency.adaptive);
}

TEST_CASE("ChunkCacheService increases admission without restarting work")
{
    auto service = makeService(1024 * 1024, 1);
    auto fetcher = std::make_shared<MultiBlockingFetcher>();
    auto cache = makeServiceCache(service, "scheduler-reconfigure", fetcher);
    std::atomic<int> callbacks{0};
    cache->addChunkReadyListener([&] { ++callbacks; });

    CHECK(cache->tryGetChunk(0, 0, 0, 0).status == ChunkStatus::MissQueued);
    CHECK(cache->tryGetChunk(0, 0, 0, 1).status == ChunkStatus::MissQueued);
    REQUIRE(fetcher->waitForStarted(1, std::chrono::seconds{2}));
    CHECK_FALSE(fetcher->waitForStarted(2, std::chrono::milliseconds{50}));

    service->configureFetchConcurrency(2, false);
    REQUIRE(fetcher->waitForStarted(2, std::chrono::seconds{2}));
    CHECK(fetcher->calls({0, 0, 0, 0}) == 1);
    CHECK(fetcher->calls({0, 0, 0, 1}) == 1);

    fetcher->release();
    CHECK(waitForResolved(*cache, 0, 0, 0, 0).status == ChunkStatus::Data);
    CHECK(waitForResolved(*cache, 0, 0, 0, 1).status == ChunkStatus::Data);
    std::this_thread::sleep_for(std::chrono::milliseconds{20});
    CHECK(callbacks.load() == 2);
    const auto concurrency = service->fetchConcurrency();
    CHECK(concurrency.maxConcurrentReads == 2);
    CHECK_FALSE(concurrency.adaptive);
}

TEST_CASE("ChunkCacheService shares results and keeps sources warm")
{
    auto service = makeService();
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

TEST_CASE("ChunkCacheService refreshes in-flight fetchers without stale publication")
{
    auto service = makeService();
    auto expired = std::make_shared<BlockingEncodedFetcher>();
    auto first = makeServiceCache(service, "credential-refresh", expired, 2);

    CHECK(first->tryGetChunk(0, 0, 0, 0).status == ChunkStatus::MissQueued);
    expired->waitStarted();

    auto refreshed = std::make_shared<CountingFetcher>();
    ChunkFetchResult replacement;
    replacement.status = ChunkFetchStatus::Found;
    replacement.bytes = makeBytes(64, std::byte{43});
    refreshed->setCanned({0, 0, 0, 0}, replacement);
    auto second = makeServiceCache(
        service, "credential-refresh", refreshed, 2);

    auto resolved = waitForResolved(*second, 0, 0, 0, 0);
    REQUIRE(resolved.status == ChunkStatus::Data);
    REQUIRE(resolved.bytes);
    CHECK(resolved.bytes->front() == std::byte{43});
    CHECK(first->sourceId() == second->sourceId());

    expired->release();
    std::this_thread::sleep_for(std::chrono::milliseconds{20});
    resolved = second->getChunkIfCached(0, 0, 0, 0);
    REQUIRE(resolved.status == ChunkStatus::Data);
    REQUIRE(resolved.bytes);
    CHECK(resolved.bytes->front() == std::byte{43});
    CHECK(expired->decodeCalls.load() == 0);
}

TEST_CASE("ChunkCacheService refresh preserves decoded chunks")
{
    auto service = makeService();
    auto original = std::make_shared<CountingFetcher>();
    ChunkFetchResult result;
    result.status = ChunkFetchStatus::Found;
    result.bytes = makeBytes(64, std::byte{57});
    original->setCanned({0, 0, 0, 0}, result);
    auto first = makeServiceCache(service, "warm-refresh", original);
    REQUIRE(first->getChunkBlocking(0, 0, 0, 0).status == ChunkStatus::Data);

    auto refreshed = std::make_shared<CountingFetcher>();
    auto second = makeServiceCache(service, "warm-refresh", refreshed);
    auto cached = second->getChunkIfCached(0, 0, 0, 0);
    REQUIRE(cached.status == ChunkStatus::Data);
    REQUIRE(cached.bytes);
    CHECK(cached.bytes->front() == std::byte{57});
    CHECK(refreshed->fetchCalls.load() == 0);
}

TEST_CASE("ChunkCacheService refresh retries retained source errors")
{
    auto service = makeService();
    auto expired = std::make_shared<CountingFetcher>();
    ChunkFetchResult denied;
    denied.status = ChunkFetchStatus::HttpError;
    denied.httpStatus = 403;
    expired->setCanned({0, 0, 0, 0}, denied);
    auto first = makeServiceCache(service, "error-refresh", expired);
    REQUIRE(first->getChunkBlocking(0, 0, 0, 0).status == ChunkStatus::Error);

    auto refreshed = std::make_shared<CountingFetcher>();
    ChunkFetchResult recovered;
    recovered.status = ChunkFetchStatus::Found;
    recovered.bytes = makeBytes(64, std::byte{67});
    refreshed->setCanned({0, 0, 0, 0}, recovered);
    auto second = makeServiceCache(service, "error-refresh", refreshed);
    const auto result = waitForResolved(*second, 0, 0, 0, 0);
    REQUIRE(result.status == ChunkStatus::Data);
    REQUIRE(result.bytes);
    CHECK(result.bytes->front() == std::byte{67});
}

TEST_CASE("ChunkCache source-scoped view clear preserves other sources")
{
    auto service = makeService();
    auto blocker = std::make_shared<BlockingFetcher>();
    auto baseFetcher = std::make_shared<CountingFetcher>();
    auto overlayFetcher = std::make_shared<CountingFetcher>();
    ChunkFetchResult result;
    result.status = ChunkFetchStatus::Found;
    result.bytes = makeBytes(64, std::byte{61});
    baseFetcher->setCanned({0, 0, 0, 0}, result);
    overlayFetcher->setCanned({0, 0, 0, 0}, result);
    auto blockerCache = makeServiceCache(service, "clear-blocker", blocker, 1);
    auto base = makeServiceCache(service, "clear-base", baseFetcher, 1);
    auto overlay = makeServiceCache(service, "clear-overlay", overlayFetcher, 1);

    (void)blockerCache->tryGetChunk(0, 0, 0, 0);
    blocker->waitStarted();
    base->replaceViewDemand({101, 1}, {0.0f, 0.0f}, {
        {{0, 0, 0, 0}, {0.0f, 0.0f}},
    });
    overlay->replaceViewDemand({101, 1}, {0.0f, 0.0f}, {
        {{0, 0, 0, 0}, {0.0f, 0.0f}},
    });

    overlay->clearSourceViewDemand(101, 1);
    CHECK(base->stats().unresolvedFetchesByLevel == std::vector<std::size_t>{1});
    CHECK(overlay->stats().unresolvedFetchesByLevel == std::vector<std::size_t>{0});
    (void)overlay->tryGetChunk(0, 0, 0, 1, {101, 1});
    CHECK(overlay->stats().unresolvedFetchesByLevel == std::vector<std::size_t>{0});

    blocker->release();
    REQUIRE(waitForResolved(*base, 0, 0, 0, 0).status == ChunkStatus::Data);
    overlay->replaceViewDemand({101, 2}, {0.0f, 0.0f}, {
        {{0, 0, 0, 0}, {0.0f, 0.0f}},
    });
    REQUIRE(waitForResolved(*overlay, 0, 0, 0, 0).status == ChunkStatus::Data);
}

TEST_CASE("ChunkCacheService deduplicates an in-flight fetch across handles")
{
    auto service = makeService();
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
    auto service = makeService();
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
    auto service = makeService(128);
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

TEST_CASE("ChunkCacheService capacity changes preserve sources and global LRU")
{
    auto service = makeService(128);
    auto fetcherA = std::make_shared<CountingFetcher>();
    auto fetcherB = std::make_shared<CountingFetcher>();
    ChunkFetchResult result;
    result.status = ChunkFetchStatus::Found;
    result.bytes = makeBytes(64, std::byte{41});
    fetcherA->setCanned({0, 0, 0, 0}, result);
    fetcherB->setCanned({0, 0, 0, 0}, result);
    auto cacheA = makeServiceCache(service, "resize-a", fetcherA);
    auto cacheB = makeServiceCache(service, "resize-b", fetcherB);
    const auto sourceA = cacheA->sourceId();
    const auto sourceB = cacheB->sourceId();

    REQUIRE(cacheA->getChunkBlocking(0, 0, 0, 0).status == ChunkStatus::Data);
    REQUIRE(cacheB->getChunkBlocking(0, 0, 0, 0).status == ChunkStatus::Data);
    service->configureDecodedByteCapacity(64);

    const auto stats = service->decodedByteBudget()->stats();
    CHECK(stats.maximumBytes == 64);
    CHECK(stats.decodedBytes == 64);
    CHECK(service->sourceCount() == 2);
    CHECK(cacheA->sourceId() == sourceA);
    CHECK(cacheB->sourceId() == sourceB);
    CHECK(cacheA->getChunkIfCached(0, 0, 0, 0).status ==
          ChunkStatus::MissQueued);
    CHECK(cacheB->getChunkIfCached(0, 0, 0, 0).status == ChunkStatus::Data);

    service->configureDecodedByteCapacity(256);
    CHECK(service->decodedByteBudget()->maximumBytes() == 256);
    CHECK(cacheB->getChunkIfCached(0, 0, 0, 0).status == ChunkStatus::Data);
}

TEST_CASE("ChunkCacheService capacity reduction preserves running and queued work")
{
    auto service = makeService(128, 1);
    auto fetcher = std::make_shared<MultiBlockingFetcher>();
    auto cache = makeServiceCache(service, "resize-in-flight", fetcher);
    const auto source = cache->sourceId();
    std::atomic<int> callbacks{0};
    cache->addChunkReadyListener([&] { ++callbacks; });

    CHECK(cache->tryGetChunk(0, 0, 0, 0).status == ChunkStatus::MissQueued);
    CHECK(cache->tryGetChunk(0, 0, 0, 1).status == ChunkStatus::MissQueued);
    REQUIRE(fetcher->waitForStarted(1, std::chrono::seconds{2}));
    CHECK_FALSE(fetcher->waitForStarted(2, std::chrono::milliseconds{50}));

    service->configureDecodedByteCapacity(64);
    CHECK(cache->sourceId() == source);
    CHECK(service->sourceCount() == 1);
    CHECK(cache->stats().unresolvedFetchesByLevel ==
          std::vector<std::size_t>{2});

    fetcher->release();
    REQUIRE(fetcher->waitForStarted(2, std::chrono::seconds{2}));
    for (int attempt = 0; attempt < 200 && callbacks.load() < 2; ++attempt)
        std::this_thread::sleep_for(std::chrono::milliseconds{5});
    CHECK(callbacks.load() == 2);
    CHECK(fetcher->calls({0, 0, 0, 0}) == 1);
    CHECK(fetcher->calls({0, 0, 0, 1}) == 1);
    CHECK(service->decodedByteBudget()->stats().decodedBytes == 64);
    CHECK(cache->stats().unresolvedFetchesByLevel ==
          std::vector<std::size_t>{0});
}

TEST_CASE("ChunkCacheService releases aggregate accounting on destruction")
{
    auto budget = std::make_shared<DecodedChunkCacheBudget>(1024 * 1024);
    {
        auto options = serviceOptions();
        options.decodedByteBudget = budget;
        auto service = std::make_shared<ChunkCacheService>(std::move(options));
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

TEST_CASE("ChunkCacheService handle destruction removes only its listeners")
{
    auto service = makeService();
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
    auto refreshedFetcher = std::make_shared<CountingFetcher>();
    refreshedFetcher->setCanned({0, 0, 0, 0}, result);
    auto reacquired = makeServiceCache(
        service, "listener-source", refreshedFetcher);
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

    auto service = makeService();
    auto fetcher = std::make_shared<BlockingFetcher>();
    std::vector<ChunkCache::LevelInfo> levels = {
        {{8, 8, 8}, {4, 4, 4}, {}},
    };
    ChunkCache::Options options;
    options.detectAllFillChunks = false;
    options.persistentCachePath = dir;
    auto cache = service->acquireSource(
        "remote-activity", std::move(levels),
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

    auto service = makeService();
    auto fetcher = std::make_shared<ThrowingFetcher>();
    std::vector<ChunkCache::LevelInfo> levels = {
        {{4, 4, 4}, {4, 4, 4}, {}},
    };
    ChunkCache::Options options;
    options.detectAllFillChunks = false;
    options.persistentCachePath = dir;
    auto cache = service->acquireSource(
        "remote-activity-error", std::move(levels),
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

    auto service = makeService();
    auto fetcher = std::make_shared<BlockingFetcher>();
    std::vector<ChunkCache::LevelInfo> levels = {
        {{4, 4, 4}, {4, 4, 4}, {}},
    };
    ChunkCache::Options options;
    options.detectAllFillChunks = false;
    options.persistentCachePath = dir;
    auto cache = service->acquireSource(
        "remote-activity-invalidate", std::move(levels),
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

TEST_CASE("ChunkCache invalidation stop events survive a throwing listener")
{
    // Invalidation-path listener isolation: a throwing listener must not
    // starve other listeners of their stop events or escape invalidate().
    // Listener iteration order is unordered, so pre-fix this failed only in
    // the thrower-first order (the per-key catch already stopped the escape);
    // post-fix it passes deterministically in both orders and guards the
    // per-listener contract going forward.
    std::mt19937_64 rng(std::random_device{}());
    const auto dir = fs::temp_directory_path() /
                     ("vc_chunk_activity_throwing_" + std::to_string(rng()));
    fs::create_directories(dir);

    auto service = makeService();
    auto fetcher = std::make_shared<BlockingFetcher>();
    std::vector<ChunkCache::LevelInfo> levels = {
        {{4, 4, 4}, {4, 4, 4}, {}},
    };
    ChunkCache::Options options;
    options.detectAllFillChunks = false;
    options.persistentCachePath = dir;
    auto cache = service->acquireSource(
        "remote-activity-throwing-listener", std::move(levels),
        std::vector<std::shared_ptr<IChunkFetcher>>{fetcher},
        0.0, ChunkDtype::UInt8, std::move(options));

    cache->addRemoteFetchActivityListener([&](const ChunkKey&, bool active) {
        if (!active)
            throw std::runtime_error("listener failure");
    });
    std::atomic<int> stopEvents{0};
    cache->addRemoteFetchActivityListener([&](const ChunkKey&, bool active) {
        if (!active)
            ++stopEvents;
    });

    CHECK(cache->tryGetChunk(0, 0, 0, 0).status == ChunkStatus::MissQueued);
    fetcher->waitStarted();
    CHECK_NOTHROW(cache->invalidate());
    CHECK(stopEvents.load() == 1);

    fetcher->release();
    fetcher->waitFinished();
    fs::remove_all(dir);
}

TEST_CASE("ChunkCacheService isolates and invalidates sources independently")
{
    auto service = makeService();
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
    opts.detectAllFillChunks = false;
    auto cacheServiceOptions = serviceOptions(1024, 1);
    cacheServiceOptions.decodedByteBudget =
        std::make_shared<DecodedChunkCacheBudget>(128);
    ChunkCache cache(
        std::move(levels),
        std::vector<std::shared_ptr<vc::render::IChunkFetcher>>{f},
        0.0, ChunkDtype::UInt8, opts, std::move(cacheServiceOptions));

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

TEST_CASE("ChunkCache: blocking read refetches when its entry is erased mid-wait")
{
    // Regression: a reader parked in getChunkBlocking could wake to find its
    // entry erased (decoded-budget eviction or invalidation beat it to the
    // lock after the fetch resolved) and surfaced a "chunk invalidated"
    // error instead of refetching the perfectly refetchable chunk.
    auto service = makeService();
    auto fetcher = std::make_shared<MultiBlockingFetcher>();
    auto cache = makeServiceCache(service, "blocking-refetch", fetcher);

    auto pending = std::async(std::launch::async, [&] {
        return cache->getChunkBlocking(0, 0, 0, 0);
    });
    // Declared after the future so it destructs first: a fatal assertion
    // below must release the fetcher before the future's destructor blocks
    // on the still-parked reader. release() is idempotent.
    FetcherReleaseGuard releaseGuard(*fetcher);
    REQUIRE(fetcher->waitForStarted(1, std::chrono::seconds{2}));

    // Erase the entry out from under the parked reader; the reader must
    // requeue the fetch under the new generation rather than fail.
    cache->invalidate();
    fetcher->release();

    REQUIRE(fetcher->waitForStarted(2, std::chrono::seconds{5}));
    REQUIRE(pending.wait_for(std::chrono::seconds{5}) ==
            std::future_status::ready);
    auto result = pending.get();
    CHECK(result.status == ChunkStatus::Data);
    REQUIRE(result.bytes);
    CHECK(result.bytes->size() == 64);
    CHECK(fetcher->calls({0, 0, 0, 0}) == 2);
}

TEST_CASE("ChunkCache: blocking read survives a sub-chunk decoded budget")
{
    // Regression: with less budget headroom than one chunk, the completion
    // path enforced the shared budget before notifying - evicting the
    // freshly decoded chunk out from under its parked reader every time, so
    // even a retrying reader failed. A parked blocking reader now protects
    // its key; eviction (and the budget's victim selection) must skip it.
    auto service = makeService(16);  // one 4x4x4 uint8 chunk is 64 bytes
    auto fetcher = std::make_shared<CountingFetcher>();
    ChunkKey k{0, 0, 0, 0};
    ChunkFetchResult fr;
    fr.status = ChunkFetchStatus::Found;
    fr.bytes = makeBytes(64, std::byte{55});
    fetcher->setCanned(k, fr);

    std::vector<ChunkCache::LevelInfo> levels = {{{8, 8, 8}, {4, 4, 4}, {}}};
    ChunkCache::Options options;
    options.detectAllFillChunks = false;
    options.decodedEvictionPreferSelf = true;  // the lasagna solve config
    auto cache = service->acquireSource(
        "sub-chunk-budget", std::move(levels),
        std::vector<std::shared_ptr<IChunkFetcher>>{fetcher},
        0.0, ChunkDtype::UInt8, std::move(options));

    auto result = cache->getChunkBlocking(0, 0, 0, 0);
    CHECK(result.status == ChunkStatus::Data);
    REQUIRE(result.bytes);
    CHECK(result.bytes->size() == 64);
    CHECK(fetcher->fetchCalls.load() == 1);
    // Releasing the pin re-enforces the budget, so the overshoot the pin
    // allowed does not persist.
    CHECK(service->decodedByteBudget()->stats().decodedBytes <= 16);
}

TEST_CASE("ChunkCache: blocking read survives zero metadata entry capacity")
{
    // Same shape as the budget case, but through enforceCapacityLocked: the
    // completion path enforces the entry-count capacity before notifying,
    // which erased the parked reader's freshly stored entry. Keys with
    // parked blocking readers must be skipped there too.
    auto service = makeService();
    auto fetcher = std::make_shared<CountingFetcher>();
    ChunkKey k{0, 0, 0, 0};
    ChunkFetchResult fr;
    fr.status = ChunkFetchStatus::Found;
    fr.bytes = makeBytes(64, std::byte{55});
    fetcher->setCanned(k, fr);

    std::vector<ChunkCache::LevelInfo> levels = {{{8, 8, 8}, {4, 4, 4}, {}}};
    ChunkCache::Options options;
    options.detectAllFillChunks = false;
    options.metadataEntryCapacity = 0;
    auto cache = service->acquireSource(
        "zero-entry-capacity", std::move(levels),
        std::vector<std::shared_ptr<IChunkFetcher>>{fetcher},
        0.0, ChunkDtype::UInt8, std::move(options));

    auto result = cache->getChunkBlocking(0, 0, 0, 0);
    CHECK(result.status == ChunkStatus::Data);
    REQUIRE(result.bytes);
    CHECK(result.bytes->size() == 64);
    CHECK(fetcher->fetchCalls.load() == 1);
    // The reader's exit re-enforces the entry-count capacity, so the entry
    // its registration protected is evicted once the reader has the result.
    CHECK(cache->stats().decodedBytes == 0);
}

TEST_CASE("ChunkCache: reader protection covers a successor it did not create")
{
    // The maintainer's multi-party scenario, single-successor form: a parked
    // blocking reader survives invalidation, and the successor entry is
    // created by ANOTHER party (here the main thread's tryGetChunk). The
    // reader's key registration must protect that successor through the
    // sub-chunk budget's completion-time enforcement even though the reader
    // never touched it. Contract under the fix: exactly two fetches (the
    // stale original and one shared successor - the emplace under the mutex
    // dedupes whoever comes second), reader gets Data.
    auto service = makeService(16);  // one 4x4x4 uint8 chunk is 64 bytes
    auto fetcher = std::make_shared<MultiBlockingFetcher>();
    auto cache = makeServiceCache(service, "successor-protection", fetcher);

    auto pending = std::async(std::launch::async, [&] {
        return cache->getChunkBlocking(0, 0, 0, 0);
    });
    FetcherReleaseGuard releaseGuard(*fetcher);
    REQUIRE(fetcher->waitForStarted(1, std::chrono::seconds{2}));

    cache->invalidate();
    // Either this queues the successor or the woken reader already did;
    // never both (mutex-serialized emplace on the same key).
    (void)cache->tryGetChunk(0, 0, 0, 0);
    fetcher->release();

    REQUIRE(pending.wait_for(std::chrono::seconds{5}) ==
            std::future_status::ready);
    auto result = pending.get();
    CHECK(result.status == ChunkStatus::Data);
    REQUIRE(result.bytes);
    CHECK(result.bytes->size() == 64);
    CHECK(fetcher->calls({0, 0, 0, 0}) == 2);
    CHECK(cache->stats().decodedBytes == 0);
}

TEST_CASE("ChunkCache: a reader exiting first cannot strip another's protection")
{
    // The maintainer's exact scenario: two readers on one key across an
    // invalidation. Reader A parks, the entry is invalidated, A requeues the
    // successor fetch; reader B joins while that fetch is in flight. Both
    // registrations are observed (stats().blockingReaders == 2) before the
    // fetch resolves, so whichever reader exits first, the other's key
    // registration keeps the resolved chunk alive under the sub-chunk
    // budget until it has read it. Exactly two fetches total.
    auto service = makeService(16);
    auto fetcher = std::make_shared<MultiBlockingFetcher>();
    auto cache = makeServiceCache(service, "two-reader-protection", fetcher);

    auto readerA = std::async(std::launch::async, [&] {
        return cache->getChunkBlocking(0, 0, 0, 0);
    });
    // Declared before the guard so the guard destructs first: a failing
    // REQUIRE below must release the fetcher before either future's
    // destructor blocks on a still-parked reader.
    std::future<ChunkResult> readerB;
    FetcherReleaseGuard releaseGuard(*fetcher);
    REQUIRE(fetcher->waitForStarted(1, std::chrono::seconds{2}));

    cache->invalidate();
    // Reader A wakes, warns, and requeues; wait until the successor fetch is
    // physically running (and parked in the fetcher).
    REQUIRE(fetcher->waitForStarted(2, std::chrono::seconds{5}));

    readerB = std::async(std::launch::async, [&] {
        return cache->getChunkBlocking(0, 0, 0, 0);
    });
    // B joins the in-flight successor; both registrations must be visible
    // before the fetch is allowed to resolve.
    bool bothRegistered = false;
    for (int i = 0; i < 5000 && !bothRegistered; ++i) {
        bothRegistered = cache->stats().blockingReaders == 2;
        if (!bothRegistered)
            std::this_thread::sleep_for(std::chrono::milliseconds{1});
    }
    REQUIRE(bothRegistered);
    fetcher->release();

    REQUIRE(readerA.wait_for(std::chrono::seconds{5}) ==
            std::future_status::ready);
    REQUIRE(readerB.wait_for(std::chrono::seconds{5}) ==
            std::future_status::ready);
    auto resultA = readerA.get();
    auto resultB = readerB.get();
    CHECK(resultA.status == ChunkStatus::Data);
    CHECK(resultB.status == ChunkStatus::Data);
    REQUIRE(resultA.bytes);
    REQUIRE(resultB.bytes);
    CHECK(fetcher->calls({0, 0, 0, 0}) == 2);
    // Both readers exited; the last exit re-enforced the budget.
    CHECK(cache->stats().decodedBytes == 0);
    CHECK(cache->stats().blockingReaders == 0);
}

TEST_CASE("ChunkCache: six blocking readers all survive an invalidation")
{
    // Integration stress for the same contract: arrival timing is
    // scheduler-dependent, so only the outcome is asserted - every reader
    // gets Data, none exhausts the retry limit.
    auto service = makeService(16);
    auto fetcher = std::make_shared<MultiBlockingFetcher>();
    auto cache = makeServiceCache(service, "six-reader-stress", fetcher);

    constexpr int kReaders = 6;
    std::vector<std::future<ChunkResult>> readers;
    readers.reserve(kReaders);
    // Guard before population: if a later std::async throws mid-loop, the
    // vector's unwinding must not block on futures parked in the fetcher.
    FetcherReleaseGuard releaseGuard(*fetcher);
    for (int i = 0; i < kReaders; ++i) {
        readers.push_back(std::async(std::launch::async, [&] {
            return cache->getChunkBlocking(0, 0, 0, 0);
        }));
    }
    REQUIRE(fetcher->waitForStarted(1, std::chrono::seconds{2}));

    cache->invalidate();
    fetcher->release();

    for (auto& reader : readers) {
        REQUIRE(reader.wait_for(std::chrono::seconds{10}) ==
                std::future_status::ready);
        auto result = reader.get();
        CHECK(result.status == ChunkStatus::Data);
        REQUIRE(result.bytes);
        CHECK(result.bytes->size() == 64);
    }
    CHECK(cache->stats().blockingReaders == 0);
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

TEST_CASE("ChunkCache: uint16 all-fill detection compares native byte pairs")
{
    constexpr std::uint16_t fill = 0x1234;
    std::vector<std::uint16_t> values(2 * 4 * 4 * 4, fill);
    values.back() = static_cast<std::uint16_t>(fill + 1);
    const auto bytes = std::as_bytes(std::span{values});

    auto f = std::make_shared<CountingFetcher>();
    ChunkFetchResult uniform;
    uniform.status = ChunkFetchStatus::Found;
    uniform.bytes.assign(bytes.begin(), bytes.begin() + 4 * 4 * 4 * 2);
    f->setCanned({0, 0, 0, 0}, std::move(uniform));

    ChunkFetchResult mixed;
    mixed.status = ChunkFetchStatus::Found;
    mixed.bytes.assign(bytes.begin() + 4 * 4 * 4 * 2, bytes.end());
    f->setCanned({0, 0, 0, 1}, std::move(mixed));

    auto c = makeCache(
        f, {4, 4, 8}, {4, 4, 4}, static_cast<double>(fill),
        ChunkDtype::UInt16);
    CHECK(waitForResolved(*c, 0, 0, 0, 0).status == ChunkStatus::AllFill);
    CHECK(waitForResolved(*c, 0, 0, 0, 1).status == ChunkStatus::Data);
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

TEST_CASE("ChunkCache maintenance persists exact source without decoding")
{
    std::mt19937_64 rng(std::random_device{}());
    const auto dir = fs::temp_directory_path() /
        ("vc_chunk_source_persist_" + std::to_string(rng()));
    fs::create_directories(dir);
    auto service = makeService();
    auto fetcher = std::make_shared<PersistentSourceFetcher>();
    auto cache = makePersistentServiceCache(
        service, "source-persist", fetcher, dir);

    const auto result = cache->persistChunkBlocking(
        0, 0, 0, 0, ChunkCache::PersistentRequestMode::Ensure);
    CHECK(result.status == ChunkCache::PersistentRequestStatus::Data);
    CHECK(fetcher->fetchCalls.load() == 1);
    CHECK(fetcher->decodeCalls.load() == 0);
    CHECK(fetcher->sourceDecodeCalls.load() == 0);
    CHECK(service->decodedByteBudget()->stats().decodedBytes == 0);

    const auto sourcePath = dir / "level_0" / "0" / "0" / "0.source";
    CHECK(readTestBytes(sourcePath) ==
          std::vector<std::byte>{std::byte{71}});

    const auto decoded = cache->getChunkBlocking(0, 0, 0, 0);
    REQUIRE(decoded.status == ChunkStatus::Data);
    REQUIRE(decoded.bytes);
    CHECK(decoded.bytes->front() == std::byte{71});
    CHECK(fetcher->fetchCalls.load() == 1);
    CHECK(fetcher->decodeCalls.load() == 0);
    CHECK(fetcher->sourceDecodeCalls.load() == 1);
    fs::remove_all(dir);
}

TEST_CASE("ChunkCache source refresh replaces only after a successful outcome")
{
    std::mt19937_64 rng(std::random_device{}());
    const auto dir = fs::temp_directory_path() /
        ("vc_chunk_source_refresh_" + std::to_string(rng()));
    const auto sourcePath = dir / "level_0" / "0" / "0" / "0.source";
    const auto emptyPath = dir / "level_0" / "0" / "0" / "0.empty";
    writeTestBytes(sourcePath, std::vector<std::byte>{std::byte{11}});

    auto service = makeService();
    auto fetcher = std::make_shared<PersistentSourceFetcher>();
    auto cache = makePersistentServiceCache(
        service, "source-refresh", fetcher, dir);

    ChunkFetchResult failed;
    failed.status = ChunkFetchStatus::HttpError;
    failed.httpStatus = 503;
    fetcher->setEncoded(failed);
    auto result = cache->persistChunkBlocking(
        0, 0, 0, 0, ChunkCache::PersistentRequestMode::Refresh);
    CHECK(result.status == ChunkCache::PersistentRequestStatus::Error);
    CHECK(readTestBytes(sourcePath) ==
          std::vector<std::byte>{std::byte{11}});

    ChunkFetchResult missing;
    missing.status = ChunkFetchStatus::Missing;
    fetcher->setEncoded(missing);
    result = cache->persistChunkBlocking(
        0, 0, 0, 0, ChunkCache::PersistentRequestMode::Refresh);
    CHECK(result.status == ChunkCache::PersistentRequestStatus::Missing);
    CHECK_FALSE(fs::exists(sourcePath));
    CHECK(fs::exists(emptyPath));

    ChunkFetchResult found;
    found.status = ChunkFetchStatus::Found;
    found.bytes = {std::byte{29}};
    fetcher->setEncoded(found);
    result = cache->persistChunkBlocking(
        0, 0, 0, 0, ChunkCache::PersistentRequestMode::Refresh);
    CHECK(result.status == ChunkCache::PersistentRequestStatus::Data);
    CHECK(readTestBytes(sourcePath) ==
          std::vector<std::byte>{std::byte{29}});
    CHECK_FALSE(fs::exists(emptyPath));
    CHECK(fetcher->decodeCalls.load() == 0);
    fs::remove_all(dir);
}

TEST_CASE("ChunkCache maintenance and decoded demand share one source transfer")
{
    std::mt19937_64 rng(std::random_device{}());
    const auto dir = fs::temp_directory_path() /
        ("vc_chunk_source_join_" + std::to_string(rng()));
    fs::create_directories(dir);
    auto service = makeService(1024 * 1024, 2);
    auto fetcher = std::make_shared<PersistentSourceFetcher>();
    fetcher->block();
    auto cache = makePersistentServiceCache(
        service, "source-join", fetcher, dir);

    auto persistence = std::async(std::launch::async, [&] {
        return cache->persistChunkBlocking(
            0, 0, 0, 0, ChunkCache::PersistentRequestMode::Refresh);
    });
    REQUIRE(fetcher->waitForStarted(1, std::chrono::seconds{2}));
    CHECK(cache->tryGetChunk(0, 0, 0, 0).status == ChunkStatus::MissQueued);
    std::this_thread::sleep_for(std::chrono::milliseconds{20});
    fetcher->release();

    CHECK(persistence.get().status ==
          ChunkCache::PersistentRequestStatus::Data);
    CHECK(waitForResolved(*cache, 0, 0, 0, 0).status == ChunkStatus::Data);
    CHECK(fetcher->fetchCalls.load() == 1);
    CHECK(fetcher->decodeCalls.load() == 1);
    fs::remove_all(dir);
}

TEST_CASE("ChunkCache maintenance survives replacement of joined view demand")
{
    std::mt19937_64 rng(std::random_device{}());
    const auto dir = fs::temp_directory_path() /
        ("vc_chunk_source_view_clear_" + std::to_string(rng()));
    fs::create_directories(dir);
    auto service = makeService(1024 * 1024, 1);
    auto fetcher = std::make_shared<PersistentSourceFetcher>();
    fetcher->block();
    auto cache = makePersistentServiceCache(
        service, "source-view-clear", fetcher, dir);

    auto persistence = std::async(std::launch::async, [&] {
        return cache->persistChunkBlocking(
            0, 0, 0, 0, ChunkCache::PersistentRequestMode::Refresh);
    });
    REQUIRE(fetcher->waitForStarted(1, std::chrono::seconds{2}));
    cache->replaceViewDemand({41, 1}, {0.0f, 0.0f}, {
        {{0, 0, 0, 0}, {0.0f, 0.0f}},
    });
    cache->replaceViewDemand({41, 2}, {0.0f, 0.0f}, {});
    fetcher->release();

    CHECK(persistence.get().status ==
          ChunkCache::PersistentRequestStatus::Data);
    CHECK(fetcher->fetchCalls.load() == 1);
    CHECK(fetcher->decodeCalls.load() == 0);
    CHECK(fs::exists(dir / "level_0" / "0" / "0" / "0.source"));
    fs::remove_all(dir);
}

TEST_CASE("ChunkCache maintenance joins a source transfer started by decoding")
{
    std::mt19937_64 rng(std::random_device{}());
    const auto dir = fs::temp_directory_path() /
        ("vc_chunk_decode_join_" + std::to_string(rng()));
    fs::create_directories(dir);
    auto service = makeService(1024 * 1024, 2);
    auto fetcher = std::make_shared<PersistentSourceFetcher>();
    fetcher->block();
    auto cache = makePersistentServiceCache(
        service, "decode-join", fetcher, dir);

    CHECK(cache->tryGetChunk(0, 0, 0, 0).status == ChunkStatus::MissQueued);
    REQUIRE(fetcher->waitForStarted(1, std::chrono::seconds{2}));
    auto persistence = std::async(std::launch::async, [&] {
        return cache->persistChunkBlocking(
            0, 0, 0, 0, ChunkCache::PersistentRequestMode::Refresh);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds{20});
    fetcher->release();

    CHECK(persistence.get().status ==
          ChunkCache::PersistentRequestStatus::Data);
    CHECK(waitForResolved(*cache, 0, 0, 0, 0).status == ChunkStatus::Data);
    CHECK(fetcher->fetchCalls.load() == 1);
    CHECK(fetcher->decodeCalls.load() == 1);
    CHECK(fs::exists(dir / "level_0" / "0" / "0" / "0.source"));
    fs::remove_all(dir);
}

TEST_CASE("ChunkCache maintenance restarts on source fetcher refresh")
{
    std::mt19937_64 rng(std::random_device{}());
    const auto dir = fs::temp_directory_path() /
        ("vc_chunk_maintenance_refresh_" + std::to_string(rng()));
    fs::create_directories(dir);
    auto service = makeService(1024 * 1024, 2);
    auto expired = std::make_shared<PersistentSourceFetcher>();
    expired->block();
    auto first = makePersistentServiceCache(
        service, "maintenance-refresh", expired, dir);

    auto persistence = std::async(std::launch::async, [&] {
        return first->persistChunkBlocking(
            0, 0, 0, 0, ChunkCache::PersistentRequestMode::Refresh);
    });
    REQUIRE(expired->waitForStarted(1, std::chrono::seconds{2}));

    auto refreshed = std::make_shared<PersistentSourceFetcher>();
    ChunkFetchResult replacement;
    replacement.status = ChunkFetchStatus::Found;
    replacement.bytes = {std::byte{88}};
    refreshed->setEncoded(replacement);
    auto second = makePersistentServiceCache(
        service, "maintenance-refresh", refreshed, dir);
    (void)second;

    REQUIRE(refreshed->waitForStarted(1, std::chrono::seconds{2}));
    CHECK(persistence.get().status ==
          ChunkCache::PersistentRequestStatus::Data);
    CHECK(readTestBytes(
              dir / "level_0" / "0" / "0" / "0.source") ==
          std::vector<std::byte>{std::byte{88}});
    expired->release();
    fs::remove_all(dir);
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
    opts.detectAllFillChunks = true;
    return std::make_shared<ChunkCache>(
        std::move(levels),
        std::vector<std::shared_ptr<vc::render::IChunkFetcher>>{f},
        0.0, ChunkDtype::UInt8, opts, serviceOptions(128, 1));
}

std::shared_ptr<ChunkCache> makeSharedBudgetCache(
    std::shared_ptr<CountingFetcher> f,
    const std::shared_ptr<DecodedChunkCacheBudget>& budget)
{
    std::vector<ChunkCache::LevelInfo> levels = {{{8, 8, 8}, {4, 4, 4}, {}}};
    ChunkCache::Options opts;
    opts.detectAllFillChunks = true;
    auto cacheServiceOptions = serviceOptions(1024, 1);
    cacheServiceOptions.decodedByteBudget = budget;
    return std::make_shared<ChunkCache>(
        std::move(levels),
        std::vector<std::shared_ptr<vc::render::IChunkFetcher>>{f},
        0.0, ChunkDtype::UInt8, opts, std::move(cacheServiceOptions));
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
    opts.detectAllFillChunks = true;
    auto c = std::make_shared<ChunkCache>(
        std::move(levels),
        std::vector<std::shared_ptr<vc::render::IChunkFetcher>>{f},
        0.0, ChunkDtype::UInt8, opts, serviceOptions(128, 1));

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

TEST_CASE("ChunkCache active-view marking defers queue resort until render publication")
{
    auto fetcher = std::make_shared<BlockingOrderFetcher>();
    std::vector<ChunkCache::LevelInfo> levels = {
        {{4, 4, 16}, {4, 4, 4}, {}},
    };
    ChunkCache::Options options;
    options.detectAllFillChunks = false;
    auto cache = std::make_shared<ChunkCache>(
        std::move(levels),
        std::vector<std::shared_ptr<IChunkFetcher>>{fetcher},
        0.0, ChunkDtype::UInt8, options, serviceOptions(1024 * 1024, 1));

    (void)cache->tryGetChunk(0, 0, 0, 0);
    fetcher->waitFirstStarted();
    cache->replaceViewDemand({201, 1}, {100.0f, 0.0f}, {
        {{0, 0, 0, 1}, {0.0f, 0.0f}},
    });
    cache->replaceViewDemand({202, 1}, {0.0f, 0.0f}, {
        {{0, 0, 0, 2}, {0.0f, 0.0f}},
    });

    cache->markViewActive(201);
    fetcher->release();
    REQUIRE(waitForResolved(*cache, 0, 0, 0, 2).status == ChunkStatus::Data);
    REQUIRE(waitForResolved(*cache, 0, 0, 0, 1).status == ChunkStatus::Data);
    CHECK(fetcher->order() == std::vector<ChunkKey>{
        {0, 0, 0, 0}, {0, 0, 0, 2}, {0, 0, 0, 1}});
}

TEST_CASE("ChunkCacheService prioritizes the active view across sources")
{
    auto service = makeService();
    auto order = std::make_shared<SharedServiceFetchOrder>();
    auto fetcherB = std::make_shared<LabeledServiceFetcher>(order, 'B', false);
    auto fetcherA = std::make_shared<LabeledServiceFetcher>(order, 'A', true);
    auto cacheB = makeServiceCache(service, "priority-b", fetcherB, 1);
    auto cacheA = makeServiceCache(service, "priority-a", fetcherA, 1);

    CHECK(cacheA->tryGetChunk(0, 0, 0, 0).status == ChunkStatus::MissQueued);
    order->firstStarted.wait();
    CHECK(cacheA->tryGetChunk(0, 0, 0, 1).status == ChunkStatus::MissQueued);
    cacheB->markViewActive(61);
    cacheB->replaceViewDemand({61, 1}, {0.0f, 0.0f}, {
        {{0, 0, 0, 0}, {0.0f, 0.0f}},
    });
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
    opts.detectAllFillChunks = false;
    auto c = std::make_shared<ChunkCache>(
        std::move(levels),
        std::vector<std::shared_ptr<vc::render::IChunkFetcher>>{f, f},
        0.0, ChunkDtype::UInt8, opts, serviceOptions(1024 * 1024, 1));

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
    opts.detectAllFillChunks = false;
    auto c = std::make_shared<ChunkCache>(
        std::move(levels),
        std::vector<std::shared_ptr<vc::render::IChunkFetcher>>{f},
        0.0, ChunkDtype::UInt8, opts, serviceOptions(1024 * 1024, 1));

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

TEST_CASE("ChunkRequestScheduler runs maintenance after interactive and background work")
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

    ChunkWorkPriority maintenance;
    maintenance.maintenance = true;
    scheduler.submit(2, maintenance, 1, 0, [&] {
        std::lock_guard lock(mutex);
        order.push_back(2);
    });
    scheduler.submit(3, {}, 1, 0, [&] {
        std::lock_guard lock(mutex);
        order.push_back(3);
    });
    ChunkWorkPriority interactive;
    interactive.interactive = true;
    scheduler.submit(4, interactive, 1, 0, [&] {
        std::lock_guard lock(mutex);
        order.push_back(4);
    });
    {
        std::lock_guard lock(mutex);
        release = true;
    }
    cv.notify_all();
    scheduler.waitIdle();
    CHECK(order == std::vector<int>{1, 4, 3, 2});
}

TEST_CASE("ChunkRequestScheduler reserves fetch admission from maintenance work")
{
    ChunkRequestScheduler scheduler(2, 7, {}, {}, {}, 1);
    ChunkWorkPriority maintenance;
    maintenance.maintenance = true;
    maintenance.reserveForegroundSlot = true;

    std::mutex mutex;
    std::condition_variable cv;
    bool releaseMaintenance = false;
    std::atomic<int> maintenanceStarted{0};
    std::latch firstMaintenanceStarted{1};
    auto maintenanceTask = [&] {
        if (maintenanceStarted.fetch_add(1) == 0)
            firstMaintenanceStarted.count_down();
        std::unique_lock lock(mutex);
        cv.wait(lock, [&] { return releaseMaintenance; });
    };

    scheduler.submit(1, maintenance, 1, 0, maintenanceTask);
    scheduler.submit(2, maintenance, 1, 0, maintenanceTask);
    firstMaintenanceStarted.wait();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    CHECK(maintenanceStarted.load() == 1);

    std::promise<void> interactiveStarted;
    auto interactiveFuture = interactiveStarted.get_future();
    ChunkWorkPriority interactive;
    interactive.interactive = true;
    scheduler.submit(3, interactive, 1, 0, [&] {
        interactiveStarted.set_value();
    });
    CHECK(interactiveFuture.wait_for(std::chrono::seconds(1)) ==
          std::future_status::ready);

    std::promise<void> refreshStarted;
    auto refreshFuture = refreshStarted.get_future();
    ChunkWorkPriority refresh;
    refresh.maintenance = true;
    scheduler.submit(4, refresh, 1, 0, [&] {
        refreshStarted.set_value();
    });
    CHECK(refreshFuture.wait_for(std::chrono::seconds(1)) ==
          std::future_status::ready);

    {
        std::lock_guard lock(mutex);
        releaseMaintenance = true;
    }
    cv.notify_all();
    scheduler.waitIdle();
    CHECK(maintenanceStarted.load() == 2);
}

TEST_CASE("ChunkRequestScheduler reservation preserves ordinary admission")
{
    ChunkRequestScheduler scheduler(2, 7, {}, {}, {}, 1);
    std::atomic<int> started{0};
    std::promise<void> bothStarted;
    auto bothStartedFuture = bothStarted.get_future();
    std::mutex mutex;
    std::condition_variable cv;
    bool release = false;
    auto task = [&] {
        if (++started == 2)
            bothStarted.set_value();
        std::unique_lock lock(mutex);
        cv.wait(lock, [&] { return release; });
    };

    scheduler.submit(1, {}, 1, 0, task);
    scheduler.submit(2, {}, 1, 0, task);
    CHECK(bothStartedFuture.wait_for(std::chrono::seconds(1)) ==
          std::future_status::ready);

    {
        std::lock_guard lock(mutex);
        release = true;
    }
    cv.notify_all();
    scheduler.waitIdle();
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

static void recordMeasuredTransfers(
    ChunkRequestScheduler& scheduler,
    std::size_t count,
    std::size_t bytesPerTransfer,
    std::chrono::steady_clock::time_point started,
    std::chrono::steady_clock::duration duration)
{
    std::vector<ChunkRequestScheduler::TransferMeasurement> transfers;
    transfers.reserve(count);
    for (std::size_t index = 0; index < count; ++index)
        transfers.emplace_back(scheduler.beginTransfer(started));
    const auto completed = started + duration;
    for (auto& transfer : transfers)
        transfer.recordBytes(bytesPerTransfer, completed);
    for (auto& transfer : transfers)
        transfer.finish(true, bytesPerTransfer, completed);
}

TEST_CASE("ChunkRequestScheduler bandwidth includes time to first byte")
{
    using Clock = std::chrono::steady_clock;
    constexpr std::size_t chunkBytes = 2ULL * 1024ULL * 1024ULL;
    ChunkRequestScheduler::AdaptiveConcurrency adaptive;
    ChunkRequestScheduler scheduler(64, 7, {}, adaptive);

    const auto start = Clock::now();
    auto transfer = scheduler.beginTransfer(start);
    transfer.recordBytes(chunkBytes, start + std::chrono::seconds(4));
    transfer.finish(true, chunkBytes, start + std::chrono::seconds(4));
    const auto stats = scheduler.transferStats();
    CHECK(stats.bytesPerSecond == doctest::Approx(0.5 * 1024.0 * 1024.0));
    CHECK(stats.admissionLimit == 2);
}

TEST_CASE("ChunkRequestScheduler aggregates streamed bytes without idle time")
{
    using Clock = std::chrono::steady_clock;
    constexpr std::size_t mib = 1024ULL * 1024ULL;
    ChunkRequestScheduler scheduler(2);
    const auto start = Clock::now();

    auto streamPair = [&](Clock::time_point pairStart) {
        auto first = scheduler.beginTransfer(pairStart);
        auto second = scheduler.beginTransfer(pairStart);
        first.recordBytes(mib, pairStart + std::chrono::seconds(1));
        second.recordBytes(mib, pairStart + std::chrono::seconds(1));
        first.recordBytes(mib, pairStart + std::chrono::seconds(1));
        second.recordBytes(mib, pairStart + std::chrono::seconds(1));
        first.finish(true, 2 * mib, pairStart + std::chrono::seconds(1));
        second.finish(true, 2 * mib, pairStart + std::chrono::seconds(1));
    };

    streamPair(start);
    CHECK(scheduler.transferStats().bytesPerSecond ==
          doctest::Approx(4.0 * double(mib)));

    // The hundred-second wall-clock gap is absent from the active-time axis.
    streamPair(start + std::chrono::seconds(101));
    CHECK(scheduler.transferStats().bytesPerSecond ==
          doctest::Approx(4.0 * double(mib)));
}

TEST_CASE("ChunkRequestScheduler streamed adaptive epochs require time and admission")
{
    using Clock = std::chrono::steady_clock;
    constexpr std::size_t mib = 1024ULL * 1024ULL;
    ChunkRequestScheduler::AdaptiveConcurrency adaptive;
    adaptive.maximum = 4;
    adaptive.minimumEpochSeconds = 5.0;
    adaptive.initialProbeMultiplier = 2;
    ChunkRequestScheduler scheduler(4, 7, {}, adaptive);
    const auto start = Clock::now();

    auto first = scheduler.beginTransfer(start);
    auto second = scheduler.beginTransfer(start);
    first.recordBytes(mib, start);
    second.recordBytes(mib, start);
    first.finish(true, mib, start + std::chrono::seconds(4));
    second.finish(true, mib, start + std::chrono::seconds(4));
    CHECK(scheduler.transferStats().admissionLimit == 2);
    CHECK_FALSE(scheduler.transferStats().probing);

    auto third = scheduler.beginTransfer(start + std::chrono::seconds(100));
    auto fourth = scheduler.beginTransfer(start + std::chrono::seconds(100));
    third.recordBytes(mib, start + std::chrono::seconds(100));
    fourth.recordBytes(mib, start + std::chrono::seconds(100));
    third.finish(true, mib, start + std::chrono::seconds(102));
    CHECK(scheduler.transferStats().admissionLimit == 3);
    CHECK(scheduler.transferStats().targetAdmissionLimit == 4);
    CHECK(scheduler.transferStats().probing);
    fourth.finish(true, mib, start + std::chrono::seconds(102));
}

TEST_CASE("ChunkRequestScheduler missing transfers do not prevent initial probe")
{
    using Clock = std::chrono::steady_clock;
    constexpr std::size_t mib = 1024ULL * 1024ULL;
    ChunkRequestScheduler::AdaptiveConcurrency adaptive;
    adaptive.maximum = 8;
    adaptive.minimumEpochSeconds = 5.0;
    adaptive.initialProbeMultiplier = 4;
    ChunkRequestScheduler scheduler(8, 7, {}, adaptive);
    const auto start = Clock::now();

    for (int second = 0; second < 5; ++second) {
        const auto intervalStart = start + std::chrono::seconds(second);
        const auto intervalEnd = intervalStart + std::chrono::seconds(1);
        auto found = scheduler.beginTransfer(intervalStart);
        auto missing = scheduler.beginTransfer(intervalStart);
        found.recordBytes(mib, intervalEnd);
        found.finish(true, mib, intervalEnd);
        missing.finish(false, 0, intervalEnd);
    }

    const auto stats = scheduler.transferStats();
    CHECK(stats.admissionLimit == 4);
    CHECK(stats.targetAdmissionLimit == 8);
    CHECK(stats.probing);
}

TEST_CASE("ChunkRequestScheduler restores capacity but restarts frequent probing")
{
    using Clock = std::chrono::steady_clock;
    constexpr std::size_t chunkBytes = 2ULL * 1024ULL * 1024ULL;
    ChunkRequestScheduler::AdaptiveConcurrency adaptive;
    adaptive.maximum = 16;
    adaptive.minimumEpochSeconds = 0.0;
    const ChunkRequestScheduler::AdaptiveState initial{
        12, 48.0 * 1024.0 * 1024.0, 8, 6.0 * 1024.0 * 1024.0};
    ChunkRequestScheduler scheduler(16, 7, {}, adaptive, initial);

    auto stats = scheduler.transferStats();
    CHECK(stats.admissionLimit == 12);
    CHECK(stats.targetAdmissionLimit == 12);
    CHECK(stats.longTermBytesPerSecond ==
          doctest::Approx(initial.longTermBytesPerSecond));
    CHECK(stats.probeIntervalSeconds ==
          doctest::Approx(adaptive.unstableProbeIntervalSeconds));
    CHECK_FALSE(stats.probing);

    const auto start = Clock::time_point{};
    recordMeasuredTransfers(
        scheduler, 12, chunkBytes, start, std::chrono::milliseconds(100));
    stats = scheduler.transferStats();
    CHECK(stats.admissionLimit == 13);
    CHECK(stats.targetAdmissionLimit == 16);
    CHECK(stats.probing);
}

TEST_CASE("ChunkRequestScheduler probes upward with completion-paced admission")
{
    using Clock = std::chrono::steady_clock;
    constexpr std::size_t chunkBytes = 2ULL * 1024ULL * 1024ULL;
    ChunkRequestScheduler::AdaptiveConcurrency adaptive;
    adaptive.maximum = 16;
    adaptive.minimumEpochSeconds = 0.0;
    adaptive.initialProbeMultiplier = 2;
    ChunkRequestScheduler scheduler(64, 7, {}, adaptive);
    auto cursor = Clock::time_point{};
    auto recordEpoch = [&](std::size_t concurrency,
                           double throughputMiB,
                           double latencySeconds) {
        const auto stats = scheduler.transferStats();
        REQUIRE(stats.admissionLimit == concurrency);
        REQUIRE(stats.targetAdmissionLimit == concurrency);
        const std::size_t count = concurrency;
        const double requestSeconds =
            static_cast<double>(chunkBytes * concurrency) /
            (throughputMiB * 1024.0 * 1024.0);
        const auto base = cursor + std::chrono::seconds(1);
        recordMeasuredTransfers(
            scheduler, count, chunkBytes, base,
            std::chrono::duration_cast<Clock::duration>(
                std::chrono::duration<double>(requestSeconds)));
        cursor = base + std::chrono::duration_cast<Clock::duration>(
            std::chrono::duration<double>(
                std::max(requestSeconds, latencySeconds)));
    };
    auto finishRamp = [&] {
        while (scheduler.transferStats().admissionLimit <
               scheduler.transferStats().targetAdmissionLimit) {
            const auto started = cursor + std::chrono::seconds(1);
            recordMeasuredTransfers(
                scheduler, 1, chunkBytes, started,
                std::chrono::milliseconds(10));
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
        CHECK(scheduler.transferStats().admissionLimit == 2 * concurrency);
        CHECK(scheduler.transferStats().targetAdmissionLimit == 2 * concurrency);
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
    adaptive.minimumStabilityObservationSeconds = 0.0;
    adaptive.initialProbeMultiplier = 2;
    adaptive.continuousSearchTurns = 1;
    ChunkRequestScheduler scheduler(4, 7, {}, adaptive);
    auto cursor = Clock::time_point{};
    auto recordEpoch = [&](std::size_t concurrency,
                           double throughputMiB,
                           double latencySeconds) {
        const std::size_t count = concurrency;
        const double requestSeconds =
            static_cast<double>(chunkBytes * concurrency) /
            (throughputMiB * 1024.0 * 1024.0);
        const auto base = cursor + std::chrono::seconds(1);
        recordMeasuredTransfers(
            scheduler, count, chunkBytes, base,
            std::chrono::duration_cast<Clock::duration>(
                std::chrono::duration<double>(requestSeconds)));
        cursor = base + std::chrono::duration_cast<Clock::duration>(
            std::chrono::duration<double>(
                std::max(requestSeconds, latencySeconds)));
    };

    // The 4-worker probe does not clear the required throughput gain, so C=2 is
    // retained. This fixture disables the five-minute eligibility period to
    // exercise the stable/changed cadence calculation directly.
    recordEpoch(2, 10.0, 0.10);
    REQUIRE(scheduler.transferStats().admissionLimit == 3);
    const auto rampStarted = cursor + std::chrono::seconds(1);
    recordMeasuredTransfers(
        scheduler, 1, chunkBytes, rampStarted, std::chrono::milliseconds(10));
    cursor = rampStarted + std::chrono::milliseconds(10);
    recordEpoch(4, 10.5, 0.20);
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

TEST_CASE("ChunkRequestScheduler does not veto throughput on request latency")
{
    using Clock = std::chrono::steady_clock;
    constexpr std::size_t chunkBytes = 2ULL * 1024ULL * 1024ULL;
    ChunkRequestScheduler::AdaptiveConcurrency adaptive;
    adaptive.maximum = 4;
    adaptive.minimumEpochSeconds = 0.0;
    adaptive.initialProbeMultiplier = 2;
    ChunkRequestScheduler scheduler(4, 7, {}, adaptive);
    auto cursor = Clock::time_point{};

    auto recordEpoch = [&](std::size_t concurrency, double throughputMiB) {
        REQUIRE(scheduler.transferStats().admissionLimit == concurrency);
        REQUIRE(scheduler.transferStats().targetAdmissionLimit == concurrency);
        const double requestSeconds =
            static_cast<double>(chunkBytes * concurrency) /
            (throughputMiB * 1024.0 * 1024.0);
        const auto started = cursor + std::chrono::seconds(1);
        const auto duration = std::chrono::duration_cast<Clock::duration>(
            std::chrono::duration<double>(requestSeconds));
        recordMeasuredTransfers(
            scheduler, concurrency, chunkBytes, started, duration);
        cursor = started + duration;
    };

    recordEpoch(2, 10.0);
    REQUIRE(scheduler.transferStats().targetAdmissionLimit == 4);
    const auto rampStarted = cursor + std::chrono::seconds(1);
    recordMeasuredTransfers(
        scheduler, 1, chunkBytes, rampStarted, std::chrono::milliseconds(10));
    cursor = rampStarted + std::chrono::milliseconds(10);

    // Aggregate throughput improves by 10%, while per-request latency grows by
    // about 82%. That latency increase is a normal consequence of concurrency
    // and must not veto the useful aggregate-bandwidth gain.
    recordEpoch(4, 11.0);
    CHECK(scheduler.transferStats().admissionLimit == 4);
    CHECK(scheduler.transferStats().targetAdmissionLimit == 4);
}

TEST_CASE("ChunkRequestScheduler delays downward probe after upward gain")
{
    using Clock = std::chrono::steady_clock;
    constexpr std::size_t chunkBytes = 2ULL * 1024ULL * 1024ULL;
    ChunkRequestScheduler::AdaptiveConcurrency adaptive;
    adaptive.maximum = 32;
    adaptive.minimumEpochSeconds = 0.0;
    adaptive.initialProbeMultiplier = 4;
    ChunkRequestScheduler scheduler(32, 7, {}, adaptive);
    auto cursor = Clock::time_point{};

    auto recordEpoch = [&](std::size_t concurrency, double throughputMiB) {
        REQUIRE(scheduler.transferStats().admissionLimit == concurrency);
        REQUIRE(scheduler.transferStats().targetAdmissionLimit == concurrency);
        const double requestSeconds =
            static_cast<double>(chunkBytes * concurrency) /
            (throughputMiB * 1024.0 * 1024.0);
        const auto started = cursor + std::chrono::seconds(1);
        const auto duration = std::chrono::duration_cast<Clock::duration>(
            std::chrono::duration<double>(requestSeconds));
        recordMeasuredTransfers(
            scheduler, concurrency, chunkBytes, started, duration);
        cursor = started + duration;
    };
    auto finishRamp = [&] {
        while (scheduler.transferStats().admissionLimit <
               scheduler.transferStats().targetAdmissionLimit) {
            const auto started = cursor + std::chrono::seconds(1);
            recordMeasuredTransfers(
                scheduler, 1, chunkBytes, started,
                std::chrono::milliseconds(10));
            cursor = started + std::chrono::milliseconds(10);
        }
    };

    recordEpoch(2, 2.0);
    REQUIRE(scheduler.transferStats().targetAdmissionLimit == 8);
    finishRamp();
    recordEpoch(8, 8.0);

    // The accepted C=8 probe remains installed. Its next settled epoch starts
    // another upward probe at C=32 rather than replaying C=2.
    CHECK(scheduler.transferStats().admissionLimit == 8);
    CHECK(scheduler.transferStats().targetAdmissionLimit == 8);
    recordEpoch(8, 8.0);
    CHECK(scheduler.transferStats().admissionLimit == 9);
    CHECK(scheduler.transferStats().targetAdmissionLimit == 32);
}

TEST_CASE("ChunkRequestScheduler requires saturated observation time for stability")
{
    using Clock = std::chrono::steady_clock;
    constexpr std::size_t chunkBytes = 2ULL * 1024ULL * 1024ULL;
    ChunkRequestScheduler::AdaptiveConcurrency adaptive;
    adaptive.minimum = 2;
    adaptive.maximum = 2;
    adaptive.minimumEpochSeconds = 0.0;
    adaptive.continuousSearchTurns = 1;
    ChunkRequestScheduler scheduler(2, 7, {}, adaptive);
    const auto start = Clock::time_point{};
    recordMeasuredTransfers(
        scheduler, 2, chunkBytes, start, std::chrono::milliseconds(100));
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
    adaptive.continuousSearchTurns = 1;
    ChunkRequestScheduler scheduler(2, 7, {}, adaptive);
    const auto start = Clock::time_point{};
    recordMeasuredTransfers(
        scheduler, 2, chunkBytes, start, std::chrono::milliseconds(100));
    const double longTermBandwidth =
        scheduler.transferStats().longTermBytesPerSecond;

    scheduler.submit(1, {}, 0, 0, [&] {
        const auto underfilledStart = start + std::chrono::seconds(10);
        recordMeasuredTransfers(
            scheduler, 1, chunkBytes, underfilledStart,
            std::chrono::seconds(4));
    });
    scheduler.waitIdle();

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
    ChunkRequestScheduler scheduler(16, 7, {}, adaptive);
    const auto start = Clock::time_point{};
    recordMeasuredTransfers(
        scheduler, 2, chunkBytes, start, std::chrono::milliseconds(100));
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

TEST_CASE("ChunkRequestScheduler fixed concurrency measures without adapting")
{
    using Clock = std::chrono::steady_clock;
    ChunkRequestScheduler scheduler(4);
    recordMeasuredTransfers(
        scheduler, 4, 2ULL * 1024ULL * 1024ULL, Clock::time_point{},
        std::chrono::milliseconds(320));
    const auto stats = scheduler.transferStats();
    CHECK_FALSE(stats.adaptive);
    CHECK(stats.admissionLimit == 4);
    CHECK(stats.bytesPerSecond ==
          doctest::Approx(25.0 * 1024.0 * 1024.0));
}

TEST_CASE("ChunkRequestScheduler decreases admission without cancelling work")
{
    ChunkRequestScheduler scheduler(3);
    std::mutex mutex;
    std::condition_variable cv;
    bool release = false;
    std::latch started{3};
    std::atomic<int> finished{0};
    std::atomic<int> followerSawFinished{-1};

    for (std::uint64_t id = 1; id <= 3; ++id) {
        scheduler.submit(id, {}, 1, 0, [&] {
            started.count_down();
            std::unique_lock lock(mutex);
            cv.wait(lock, [&] { return release; });
            lock.unlock();
            ++finished;
        });
    }
    started.wait();
    scheduler.submit(4, {}, 1, 0, [&] {
        followerSawFinished.store(finished.load());
    });

    scheduler.configureConcurrency(1);
    CHECK(scheduler.active() == 3);
    CHECK(scheduler.pending() == 1);
    {
        std::lock_guard lock(mutex);
        release = true;
    }
    cv.notify_all();
    scheduler.waitIdle();

    CHECK(finished.load() == 3);
    CHECK(followerSawFinished.load() == 3);
}

TEST_CASE("ChunkRequestScheduler changes fixed and adaptive modes in place")
{
    ChunkRequestScheduler scheduler(8);
    CHECK(scheduler.workerCapacity() == 8);
    scheduler.configureConcurrency(3);
    CHECK(scheduler.transferStats().admissionLimit == 3);
    CHECK_FALSE(scheduler.transferStats().adaptive);

    ChunkRequestScheduler::AdaptiveConcurrency adaptive;
    adaptive.minimum = 2;
    adaptive.maximum = 6;
    scheduler.configureConcurrency(6, adaptive);
    CHECK(scheduler.transferStats().adaptive);
    CHECK(scheduler.transferStats().admissionLimit == 3);

    scheduler.configureConcurrency(5);
    CHECK_FALSE(scheduler.transferStats().adaptive);
    CHECK(scheduler.transferStats().admissionLimit == 5);

    scheduler.configureConcurrency(6, adaptive);
    CHECK(scheduler.transferStats().adaptive);
    CHECK(scheduler.transferStats().admissionLimit == 3);
    scheduler.configureConcurrency(5);
    CHECK_THROWS_AS(scheduler.configureConcurrency(0), std::invalid_argument);
    CHECK_THROWS_AS(scheduler.configureConcurrency(9), std::invalid_argument);
}

TEST_CASE("ChunkCacheService rejects invalid fetch capacity and admission")
{
    auto options = serviceOptions(1024 * 1024, 1);
    options.fetchConcurrency.workerCapacity = 0;
    CHECK_THROWS_AS(
        std::make_shared<ChunkCacheService>(options), std::invalid_argument);

    options.fetchConcurrency.workerCapacity = 2;
    options.fetchConcurrency.maxConcurrentReads = 3;
    CHECK_THROWS_AS(
        std::make_shared<ChunkCacheService>(options), std::invalid_argument);

    auto service = makeService(1024 * 1024, 1);
    CHECK_THROWS_AS(
        service->configureFetchConcurrency(0, false), std::invalid_argument);
    CHECK_THROWS_AS(
        service->configureFetchConcurrency(9, false), std::invalid_argument);
    CHECK(service->fetchConcurrency().maxConcurrentReads == 1);
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

TEST_CASE("ChunkRequestScheduler releases completed tasks before becoming idle")
{
    struct BlockingDestructor {
        std::latch& started;
        std::latch& release;

        BlockingDestructor(std::latch& startedIn, std::latch& releaseIn)
            : started(startedIn)
            , release(releaseIn)
        {
        }

        ~BlockingDestructor()
        {
            started.count_down();
            release.wait();
        }
    };

    ChunkRequestScheduler scheduler(1);
    std::latch destructorStarted{1};
    std::latch releaseDestructor{1};
    auto captured = std::make_shared<BlockingDestructor>(
        destructorStarted, releaseDestructor);
    scheduler.submit(1, {}, 1, 0, [captured] {});
    captured.reset();

    std::atomic_bool idleReturned{false};
    std::jthread waiter([&] {
        scheduler.waitIdle();
        idleReturned.store(true, std::memory_order_release);
    });

    destructorStarted.wait();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    CHECK_FALSE(idleReturned.load(std::memory_order_acquire));
    releaseDestructor.count_down();
    waiter.join();
    CHECK(idleReturned.load(std::memory_order_acquire));
}

TEST_CASE("ChunkCache view snapshots promote queued work and reject stale replacement")
{
    auto fetcher = std::make_shared<BlockingOrderFetcher>();
    std::vector<ChunkCache::LevelInfo> levels = {
        {{4, 4, 16}, {4, 4, 4}, {}},
    };
    ChunkCache::Options options;
    options.detectAllFillChunks = false;
    auto cache = std::make_shared<ChunkCache>(
        std::move(levels),
        std::vector<std::shared_ptr<IChunkFetcher>>{fetcher},
        0.0, ChunkDtype::UInt8, options, serviceOptions(1024 * 1024, 1));

    (void)cache->tryGetChunk(0, 0, 0, 0);
    fetcher->waitFirstStarted();
    (void)cache->tryGetChunk(0, 0, 0, 1); // initially background
    (void)cache->tryGetChunk(0, 0, 0, 2); // remains background

    vc::render::ChunkRequestContext current{41, 2};
    cache->markViewActive(41);
    cache->replaceViewDemand(current, {10.0f, 10.0f}, {
        {{0, 0, 0, 1}, {10.0f, 10.0f}},
    });
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
    options.detectAllFillChunks = false;
    auto cache = std::make_shared<ChunkCache>(
        std::move(levels),
        std::vector<std::shared_ptr<IChunkFetcher>>{fetcher},
        0.0, ChunkDtype::UInt8, options, serviceOptions(1024 * 1024, 1));

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
    options.detectAllFillChunks = false;
    auto cache = std::make_shared<ChunkCache>(
        std::move(levels),
        std::vector<std::shared_ptr<IChunkFetcher>>{fetcher},
        0.0, ChunkDtype::UInt8, options, serviceOptions(1024 * 1024, 1));

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
    options.detectAllFillChunks = false;
    auto cache = std::make_shared<ChunkCache>(
        std::move(levels),
        std::vector<std::shared_ptr<IChunkFetcher>>{fetcher},
        0.0, ChunkDtype::UInt8, options, serviceOptions(1024 * 1024, 1));

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
    options.detectAllFillChunks = false;
    auto cache = std::make_shared<ChunkCache>(
        std::move(levels),
        std::vector<std::shared_ptr<IChunkFetcher>>{fetcher},
        0.0, ChunkDtype::UInt8, options, serviceOptions(1024 * 1024, 1));

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

TEST_CASE("ChunkCache view replacement preserves context-free prefetch")
{
    auto fetcher = std::make_shared<BlockingOrderFetcher>();
    std::vector<ChunkCache::LevelInfo> levels = {
        {{4, 4, 16}, {4, 4, 4}, {}},
    };
    ChunkCache::Options options;
    options.detectAllFillChunks = false;
    auto cache = std::make_shared<ChunkCache>(
        std::move(levels),
        std::vector<std::shared_ptr<IChunkFetcher>>{fetcher},
        0.0, ChunkDtype::UInt8, options, serviceOptions(1024 * 1024, 1));

    (void)cache->tryGetChunk(0, 0, 0, 0); // running background blocker
    fetcher->waitFirstStarted();
    cache->replaceViewDemand({86, 1}, {0.0f, 0.0f}, {
        {{0, 0, 0, 1}, {0.0f, 0.0f}},
    });
    cache->prefetchChunks(
        std::vector<ChunkKey>{{0, 0, 0, 1}},
        /*wait=*/false, /*priorityOffset=*/0);
    cache->replaceViewDemand({86, 2}, {0.0f, 0.0f}, {
        {{0, 0, 0, 2}, {0.0f, 0.0f}},
    });

    // Replacing view 86 removes its old slot from chunk 1, but the context-free
    // SurfaceCache-style prefetch remains as an independent background owner.
    CHECK(cache->stats().unresolvedFetchesByLevel ==
          std::vector<std::size_t>{3});
    fetcher->release();
    CHECK(waitForResolved(*cache, 0, 0, 0, 2).status == ChunkStatus::Data);
    CHECK(waitForResolved(*cache, 0, 0, 0, 1).status == ChunkStatus::Data);
    CHECK(fetcher->order() == std::vector<ChunkKey>{
        {0, 0, 0, 0}, {0, 0, 0, 2}, {0, 0, 0, 1}});
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
    options.detectAllFillChunks = false;
    auto cache = std::make_shared<ChunkCache>(
        std::move(levels),
        std::vector<std::shared_ptr<IChunkFetcher>>(4, fetcher),
        0.0, ChunkDtype::UInt8, options, serviceOptions(1024 * 1024, 1));

    (void)cache->tryGetChunk(0, 0, 0, 0);
    fetcher->waitFirstStarted();

    // Both chunks use absolute level 2. The active view sees the first as its
    // second fallback. Another view sees the shared second chunk as its third
    // fallback, so that chunk must be selected first despite being inactive.
    cache->markViewActive(71);
    cache->replaceViewDemand({71, 1}, {0.0f, 0.0f}, {
        {{2, 0, 0, 0}, {0.0f, 0.0f}, 2},
    });
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
    options.detectAllFillChunks = false;
    auto cache = std::make_shared<ChunkCache>(
        std::move(levels),
        std::vector<std::shared_ptr<IChunkFetcher>>(4, fetcher),
        0.0, ChunkDtype::UInt8, options, serviceOptions(1024 * 1024, 1));

    (void)cache->tryGetChunk(0, 0, 0, 0);
    fetcher->waitFirstStarted();

    cache->markViewActive(74);
    cache->replaceViewDemand({74, 1}, {0.0f, 0.0f}, {
        {{3, 0, 0, 0}, {0.0f, 0.0f}, 0},
    });
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
    options.detectAllFillChunks = false;
    auto cache = std::make_shared<ChunkCache>(
        std::move(levels),
        std::vector<std::shared_ptr<IChunkFetcher>>{fetcher},
        0.0, ChunkDtype::UInt8, options, serviceOptions(1024 * 1024, 1));

    (void)cache->tryGetChunk(0, 0, 0, 0);
    fetcher->waitFirstStarted();
    (void)cache->tryGetChunk(0, 0, 0, 2); // background, queued first

    cache->markViewActive(51);
    cache->replaceViewDemand({51, 2}, {0.0f, 0.0f}, {
        {{0, 0, 0, 1}, {0.0f, 0.0f}},
    });
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
        options.detectAllFillChunks = false;
        ChunkCache cache(
            std::move(levels),
            std::vector<std::shared_ptr<IChunkFetcher>>{fetcher},
            0.0, ChunkDtype::UInt8, options,
            serviceOptions(1024 * 1024, 1));

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
        options.detectAllFillChunks = false;
        ChunkCache cache(
            std::move(levels),
            std::vector<std::shared_ptr<IChunkFetcher>>(3, fetcher),
            0.0, ChunkDtype::UInt8, options,
            serviceOptions(1024 * 1024, 1));

        for (int ix = 0; ix < 8; ++ix)
            (void)cache.tryGetChunk(0, 0, 0, ix);
        REQUIRE(fetcher->waitForPersistentDecodes(8, std::chrono::seconds{2}));

        cache.markViewActive(91);
        cache.replaceViewDemand({91, 1}, {0.0f, 0.0f}, {
            {{0, 0, 0, 8}, {0.0f, 0.0f}, 0},
            {{2, 0, 0, 0}, {0.0f, 0.0f}, 2},
        });
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
        options.detectAllFillChunks = false;
        ChunkCache cache(
            std::move(levels),
            std::vector<std::shared_ptr<IChunkFetcher>>{fetcher},
            0.0, ChunkDtype::UInt8, options,
            serviceOptions(1024 * 1024, 1));

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
