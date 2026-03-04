#include "vc/core/cache/TieredChunkCache.hpp"
#include "vc/core/cache/CacheDebugLog.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>

namespace vc::cache {

// Process-wide shared decompression pool.  All TieredChunkCache instances
// share this pool so total thread count stays bounded regardless of how many
// volumes are loaded.
static std::shared_ptr<utils::ThreadPool> sharedDecompPool()
{
    static std::mutex mtx;
    static std::weak_ptr<utils::ThreadPool> weakPool;

    std::lock_guard lk(mtx);
    auto pool = weakPool.lock();
    if (!pool) {
        auto n = std::max<std::size_t>(2, std::thread::hardware_concurrency() / 2);
        pool = std::make_shared<utils::ThreadPool>(n);
        weakPool = pool;
    }
    return pool;
}

// Helper to build LRUCache config for the hot tier
static auto makeHotConfig(const TieredChunkCache::Config& cfg) {
    using HotCache = utils::LRUCache<ChunkKey, ChunkDataPtr, ChunkKeyHash>;
    typename HotCache::Config c;
    c.max_bytes = cfg.hotMaxBytes;
    c.evict_ratio = 15.0 / 16.0;
    c.promote_on_read = false;  // VC3D pattern: no LRU churn on reads
    c.size_fn = [](const ChunkDataPtr& p) -> std::size_t {
        return p ? p->totalBytes() : 0;
    };
    return c;
}

// Helper to build LRUCache config for the warm tier
static auto makeWarmConfig(const TieredChunkCache::Config& cfg) {
    using WarmCache = utils::LRUCache<ChunkKey, CompressedChunk, ChunkKeyHash>;
    typename WarmCache::Config c;
    c.max_bytes = cfg.warmMaxBytes;
    c.evict_ratio = 15.0 / 16.0;
    c.promote_on_read = false;
    c.size_fn = [](const CompressedChunk& ch) -> std::size_t {
        return ch.data.size();
    };
    return c;
}

TieredChunkCache::TieredChunkCache(
    Config config,
    std::unique_ptr<ChunkSource> source,
    DecompressFn decompress,
    std::shared_ptr<DiskStore> diskStore)
    : hotCache_(makeHotConfig(config))
    , warmCache_(makeWarmConfig(config))
    , config_(std::move(config))
    , diskStore_(std::move(diskStore))
    , source_(std::move(source))
    , decompress_(std::move(decompress))
    , ioPool_(config_.ioThreads)
    , decompPool_(sharedDecompPool())
{
    // Wire up the IO pool: fetch = check cold first, then ice
    ioPool_.setFetchFunc([this](const ChunkKey& key) -> std::vector<uint8_t> {
        using Clock = std::chrono::steady_clock;
        auto t0 = Clock::now();

        // Try cold (disk cache) first
        if (diskStore_) {
            auto diskData = diskStore_->get(config_.volumeId, key);
            if (diskData && !diskData->empty()) {
                auto n = statColdHits_.fetch_add(1, std::memory_order_relaxed);
                auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t0).count();
                if (n < 3)
                    std::fprintf(stderr, "[Cache] cold-hit #%lu lvl=%d (%d,%d,%d) bytes=%zu %ldms\n",
                                 n + 1, key.level, key.iz, key.iy, key.ix, diskData->size(), ms);
                if (auto* log = cacheDebugLog())
                    std::fprintf(log, "FETCH cold-hit lvl=%d (%d,%d,%d) bytes=%zu %ldms\n",
                                 key.level, key.iz, key.iy, key.ix, diskData->size(), ms);
                return std::move(*diskData);
            }
        }

        // Fetch from ice (remote/filesystem source)
        if (!source_) return {};

        auto t1 = Clock::now();
        std::vector<uint8_t> data;
        try {
            data = source_->fetch(key);
        } catch (const std::exception& e) {
            if (auto* log = cacheDebugLog())
                std::fprintf(log, "FETCH ice-error lvl=%d (%d,%d,%d) %s\n",
                             key.level, key.iz, key.iy, key.ix, e.what());
            throw;
        }
        auto t2 = Clock::now();
        auto fetchMs = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

        if (data.empty()) {
            if (auto* log = cacheDebugLog())
                std::fprintf(log, "FETCH ice-empty lvl=%d (%d,%d,%d) %ldms\n",
                             key.level, key.iz, key.iy, key.ix, fetchMs);
            return {};
        }

        auto n = statIceFetches_.fetch_add(1, std::memory_order_relaxed);
        if (n < 3)
            std::fprintf(stderr, "[Cache] ice-fetch #%lu lvl=%d (%d,%d,%d) bytes=%zu %ldms\n",
                         n + 1, key.level, key.iz, key.iy, key.ix, data.size(), fetchMs);

        // Store to cold (disk cache) for persistence
        if (diskStore_) {
            diskStore_->put(config_.volumeId, key, data);
        }
        auto totalMs = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t0).count();
        if (auto* log = cacheDebugLog())
            std::fprintf(log, "FETCH ice-ok lvl=%d (%d,%d,%d) bytes=%zu fetch=%ldms total=%ldms\n",
                         key.level, key.iz, key.iy, key.ix, data.size(), fetchMs, totalMs);

        return data;
    });

    // Wire up IO completion: compressed bytes → warm, then hand off
    // decompression to the dedicated decompression pool so I/O threads
    // stay free to fetch more chunks.
    ioPool_.setCompletionCallback(
        [this](const ChunkKey& key, std::vector<uint8_t>&& compressed) {
            if (compressed.empty()) {
                // Remember that this chunk doesn't exist so we never refetch
                {
                    std::unique_lock lock(negativeMutex_);
                    constexpr size_t maxNegative = 500000;
                    if (negativeCache_.size() >= maxNegative) {
                        // Evict ~25% of entries (arbitrary iteration order is fine)
                        size_t toEvict = maxNegative / 4;
                        auto it = negativeCache_.begin();
                        for (size_t i = 0; i < toEvict && it != negativeCache_.end(); i++) {
                            it = negativeCache_.erase(it);
                        }
                    }
                    negativeCache_.insert(key);
                }
                if (auto* log = cacheDebugLog())
                    std::fprintf(log, "COMPLETE empty lvl=%d (%d,%d,%d) [negative cached]\n",
                                 key.level, key.iz, key.iy, key.ix);
                return;
            }

            // Step 1: Store in warm tier (fast, stays on I/O thread)
            warmPut(key, std::move(compressed));

            // Steps 2-4: Decompress and promote on the decompression pool
            decompPool_->enqueue([this, key]() {
                using Clock = std::chrono::steady_clock;
                auto t0 = Clock::now();

                auto warmEntry = warmGet(key);
                if (warmEntry && decompress_) {
                    auto td0 = Clock::now();
                    auto data = decompress_(warmEntry->data, key);
                    auto decompMs = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - td0).count();
                    if (data) {
                        // Check if this key should be pinned
                        bool shouldPin = false;
                        {
                            std::lock_guard plock(pinnedKeysMutex_);
                            shouldPin = pendingPinKeys_.count(key) > 0;
                        }
                        size_t decompBytes = data->totalBytes();
                        hotPut(key, std::move(data), shouldPin);
                        auto totalMs = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t0).count();
                        if (auto* log = cacheDebugLog())
                            std::fprintf(log, "COMPLETE hot-put lvl=%d (%d,%d,%d) decompBytes=%zu decomp=%ldms total=%ldms\n",
                                         key.level, key.iz, key.iy, key.ix, decompBytes, decompMs, totalMs);
                    } else {
                        // Remove from warm to prevent repeated failed
                        // decompress attempts (hot loop).
                        warmCache_.remove(key);
                        if (auto* log = cacheDebugLog())
                            std::fprintf(log, "COMPLETE decompress-fail lvl=%d (%d,%d,%d) [removed from warm]\n",
                                         key.level, key.iz, key.iy, key.ix);
                    }
                }

                // Notify listeners (e.g., to trigger UI refresh).
                if (!chunkArrivedFlag_.exchange(true, std::memory_order_acq_rel)) {
                    std::lock_guard cbLock(callbackMutex_);
                    for (const auto& [id, cb] : chunkReadyListeners_) {
                        cb(key);
                    }
                }
            });
        });

    // Start IO workers after callbacks are set to avoid data races.
    ioPool_.start();

    loadNegativeCache();
}

TieredChunkCache::~TieredChunkCache()
{
    ioPool_.stop();
    // Wait for any in-flight decompression tasks from this cache before
    // tearing down.  The pool is shared, so we just wait for idle (which
    // covers our tasks since the IO pool — our only producer — is stopped).
    if (decompPool_) decompPool_->wait_idle();

    // Print final fetch stats summary
    auto cold = statColdHits_.load(std::memory_order_relaxed);
    auto ice = statIceFetches_.load(std::memory_order_relaxed);
    if (cold > 0 || ice > 0) {
        std::fprintf(stderr, "[Cache] session summary: coldHits=%lu iceFetches=%lu (%.0f%% from disk)\n",
                     cold, ice, (cold + ice) > 0 ? 100.0 * cold / (cold + ice) : 0.0);
    }
    decompPool_.reset();  // Release our reference (pool lives if others hold it)
    saveNegativeCache();
}

// =============================================================================
// Non-blocking reads
// =============================================================================

ChunkDataPtr TieredChunkCache::get(const ChunkKey& key)
{
    // Check hot tier
    auto hot = hotGet(key);
    if (hot) {
        statHotHits_.fetch_add(1, std::memory_order_relaxed);
        return hot;
    }

    // Check warm tier — decompress and promote to hot
    auto warm = warmGet(key);
    if (warm) {
        statWarmHits_.fetch_add(1, std::memory_order_relaxed);
        return promoteFromWarm(key, std::move(*warm));
    }

    statMisses_.fetch_add(1, std::memory_order_relaxed);
    return nullptr;
}

std::pair<ChunkDataPtr, int> TieredChunkCache::getBestAvailable(
    const ChunkKey& key)
{
    int maxLevel = source_ ? source_->numLevels() - 1 : 0;

    // Try the requested level first, then progressively coarser
    for (int lvl = key.level; lvl <= maxLevel; lvl++) {
        ChunkKey coarsened =
            (lvl == key.level) ? key : key.coarsen(lvl);

        auto data = get(coarsened);
        if (data) return {data, lvl};
    }

    // Nothing available at any level
    return {nullptr, -1};
}

// =============================================================================
// Blocking reads
// =============================================================================

ChunkDataPtr TieredChunkCache::getBlocking(const ChunkKey& key)
{
    // Known non-existent? Return immediately.
    {
        std::shared_lock lock(negativeMutex_);
        if (negativeCache_.count(key)) return nullptr;
    }

    // Check hot
    auto hot = hotGet(key);
    if (hot) {
        statHotHits_.fetch_add(1, std::memory_order_relaxed);
        return hot;
    }

    // Check warm
    {
        auto warm = warmGet(key);
        if (warm) {
            statWarmHits_.fetch_add(1, std::memory_order_relaxed);
            return promoteFromWarm(key, std::move(*warm));
        }
    }

    // Per-key lock to prevent duplicate loads
    auto diskLock = lockPool_.lock<ChunkKey, ChunkKeyHash>(key);

    // Re-check hot after acquiring lock (another thread may have loaded)
    hot = hotGet(key);
    if (hot) return hot;

    // Re-check warm after acquiring lock
    {
        auto warm = warmGet(key);
        if (warm) return promoteFromWarm(key, std::move(*warm));
    }

    // Full promotion chain: cold → warm → hot, or ice → cold → warm → hot
    auto data = loadFull(key);
    if (!data) {
        // Fetch failed — remember so we don't retry
        std::unique_lock lock(negativeMutex_);
        negativeCache_.insert(key);
    }
    return data;
}

// =============================================================================
// Async prefetch
// =============================================================================

void TieredChunkCache::prefetch(const ChunkKey& key)
{
    // Already in hot or warm? No-op.
    if (hotCache_.contains(key)) return;
    if (warmCache_.contains(key)) return;

    // Known non-existent? Don't waste an IO round-trip.
    {
        std::shared_lock lock(negativeMutex_);
        if (negativeCache_.count(key)) return;
    }

    ioPool_.submit(key);
}

void TieredChunkCache::prefetchRegion(
    int level, int iz0, int iy0, int ix0, int iz1, int iy1, int ix1)
{
    // Build all candidate keys
    int totalChecked = 0;
    std::vector<ChunkKey> allKeys;
    allKeys.reserve(static_cast<size_t>(iz1 - iz0 + 1) * (iy1 - iy0 + 1) * (ix1 - ix0 + 1));
    for (int iz = iz0; iz <= iz1; iz++) {
        for (int iy = iy0; iy <= iy1; iy++) {
            for (int ix = ix0; ix <= ix1; ix++) {
                totalChecked++;
                allKeys.push_back(ChunkKey{level, iz, iy, ix});
            }
        }
    }

    // Use LRUCache batch operations to filter efficiently
    auto missingHot = hotCache_.missing_keys(allKeys.begin(), allKeys.end());
    auto missingBoth = warmCache_.missing_keys(missingHot.begin(), missingHot.end());

    // Filter out negative-cached keys
    std::vector<ChunkKey> keys;
    if (!missingBoth.empty()) {
        keys.reserve(missingBoth.size());
        std::shared_lock lock(negativeMutex_);
        for (const auto& key : missingBoth) {
            if (!negativeCache_.count(key)) {
                keys.push_back(key);
            }
        }
    }
    if (auto* log = cacheDebugLog()) {
        static std::atomic<int> prefetchLogCount{0};
        int n = prefetchLogCount.fetch_add(1, std::memory_order_relaxed);
        if (n < 5) {
            std::fprintf(log, "prefetchRegion: level=%d range=(%d-%d,%d-%d,%d-%d) checked=%d toSubmit=%zu\n",
                         level, iz0, iz1, iy0, iy1, ix0, ix1, totalChecked, keys.size());
        }
    }
    if (!keys.empty()) {
        ioPool_.submit(keys);
    }
}

void TieredChunkCache::cancelPendingPrefetch()
{
    ioPool_.cancelPending();
}

void TieredChunkCache::setIOEpoch(uint64_t epoch)
{
    ioPool_.setCurrentEpoch(epoch);
}

// =============================================================================
// Pin level
// =============================================================================

void TieredChunkCache::pinLevel(
    int level,
    const std::array<int, 3>& gridDims,
    bool blocking)
{
    std::vector<ChunkKey> keys;
    for (int iz = 0; iz < gridDims[0]; iz++) {
        for (int iy = 0; iy < gridDims[1]; iy++) {
            for (int ix = 0; ix < gridDims[2]; ix++) {
                keys.push_back({level, iz, iy, ix});
            }
        }
    }

    if (blocking) {
        // Load all chunks synchronously and pin them
        int pinned = 0, failed = 0;
        for (auto& key : keys) {
            auto data = getBlocking(key);
            if (data) {
                // Re-insert as pinned
                hotPut(key, std::move(data), /*pinned=*/true);
                pinned++;
            } else {
                failed++;
            }
        }
        if (auto* log = cacheDebugLog())
            std::fprintf(log, "pinLevel: level=%d total=%zu pinned=%d failed=%d\n",
                         level, keys.size(), pinned, failed);
    } else {
        // Register keys as pending pin, then submit for background loading
        {
            std::lock_guard lock(pinnedKeysMutex_);
            for (auto& key : keys) {
                pendingPinKeys_.insert(key);
            }
        }
        for (auto& key : keys) {
            auto data = hotGet(key);
            if (!data) {
                ioPool_.submit(key);
            } else {
                // Already in hot — just mark as pinned
                hotPut(key, std::move(data), /*pinned=*/true);
            }
        }
    }
}

// =============================================================================
// Cache management
// =============================================================================

void TieredChunkCache::clearMemory()
{
    hotCache_.clear();
    warmCache_.clear();
}

void TieredChunkCache::clearAll()
{
    ioPool_.cancelPending();
    clearMemory();
    {
        std::unique_lock lock(negativeMutex_);
        negativeCache_.clear();
    }
    if (diskStore_) {
        diskStore_->clearVolume(config_.volumeId);
        // Remove persisted negative cache file
        std::error_code ec;
        std::filesystem::remove(
            diskStore_->root() / (config_.volumeId + ".negative"), ec);
    }
}

int TieredChunkCache::numLevels() const
{
    return source_ ? source_->numLevels() : 0;
}

std::array<int, 3> TieredChunkCache::chunkShape(int level) const
{
    return source_ ? source_->chunkShape(level) : std::array<int, 3>{0, 0, 0};
}

std::array<int, 3> TieredChunkCache::levelShape(int level) const
{
    return source_ ? source_->levelShape(level) : std::array<int, 3>{0, 0, 0};
}

void TieredChunkCache::setDataBounds(int minX, int maxX, int minY, int maxY, int minZ, int maxZ)
{
    std::lock_guard lock(dataBoundsMutex_);
    dataBoundsL0_ = {minX, maxX, minY, maxY, minZ, maxZ, true};
}

TieredChunkCache::DataBoundsL0 TieredChunkCache::dataBounds() const
{
    std::lock_guard lock(dataBoundsMutex_);
    return dataBoundsL0_;
}

bool TieredChunkCache::isNegativeCached(const ChunkKey& key) const
{
    std::shared_lock lock(negativeMutex_);
    return negativeCache_.count(key) > 0;
}

bool TieredChunkCache::areAllCachedInRegion(
    int level,
    int iz0, int iy0, int ix0,
    int iz1, int iy1, int ix1) const
{
    // Build keys for the region
    std::vector<ChunkKey> allKeys;
    allKeys.reserve(static_cast<size_t>(iz1 - iz0 + 1) * (iy1 - iy0 + 1) * (ix1 - ix0 + 1));
    for (int iz = iz0; iz <= iz1; iz++) {
        for (int iy = iy0; iy <= iy1; iy++) {
            for (int ix = ix0; ix <= ix1; ix++) {
                allKeys.push_back(ChunkKey{level, iz, iy, ix});
            }
        }
    }

    // Use LRUCache batch operations
    auto missingHot = hotCache_.missing_keys(allKeys.begin(), allKeys.end());
    if (missingHot.empty()) return true;

    auto missingBoth = warmCache_.missing_keys(missingHot.begin(), missingHot.end());
    if (missingBoth.empty()) return true;

    // Check negative cache for remaining misses
    std::shared_lock negLock(negativeMutex_);
    for (const auto& key : missingBoth) {
        if (negativeCache_.count(key) == 0) {
            return false;
        }
    }
    return true;
}

TieredChunkCache::ChunkReadyCallbackId
TieredChunkCache::addChunkReadyListener(ChunkReadyCallback cb)
{
    std::lock_guard lock(callbackMutex_);
    auto id = nextListenerId_.fetch_add(1, std::memory_order_relaxed);
    chunkReadyListeners_.emplace_back(id, std::move(cb));
    return id;
}

void TieredChunkCache::removeChunkReadyListener(ChunkReadyCallbackId id)
{
    std::lock_guard lock(callbackMutex_);
    auto it = std::remove_if(chunkReadyListeners_.begin(), chunkReadyListeners_.end(),
        [id](const auto& p) { return p.first == id; });
    chunkReadyListeners_.erase(it, chunkReadyListeners_.end());
}

void TieredChunkCache::clearChunkArrivedFlag()
{
    chunkArrivedFlag_.store(false, std::memory_order_release);
}

// =============================================================================
// Stats
// =============================================================================

auto TieredChunkCache::stats() const -> Stats
{
    Stats s;
    s.hotHits = statHotHits_.load(std::memory_order_relaxed);
    s.warmHits = statWarmHits_.load(std::memory_order_relaxed);
    s.coldHits = statColdHits_.load(std::memory_order_relaxed);
    s.iceFetches = statIceFetches_.load(std::memory_order_relaxed);
    s.misses = statMisses_.load(std::memory_order_relaxed);
    s.hotEvictions = hotCache_.evictions();
    s.warmEvictions = warmCache_.evictions();
    s.hotBytes = hotCache_.byte_size();
    s.warmBytes = warmCache_.byte_size();
    s.ioPending = ioPool_.pendingCount();
    return s;
}

// =============================================================================
// Hot tier — delegates to utils::LRUCache
// =============================================================================

ChunkDataPtr TieredChunkCache::hotGet(const ChunkKey& key)
{
    return hotCache_.get_or(key, nullptr);
}

void TieredChunkCache::hotPut(
    const ChunkKey& key, ChunkDataPtr data, bool pinned)
{
    if (pinned) {
        hotCache_.put_pinned(key, std::move(data));
    } else {
        hotCache_.put(key, std::move(data));
    }
}

// =============================================================================
// Warm tier — delegates to utils::LRUCache
// =============================================================================

std::optional<CompressedChunk> TieredChunkCache::warmGet(const ChunkKey& key)
{
    return warmCache_.get(key);
}

void TieredChunkCache::warmPut(
    const ChunkKey& key,
    std::vector<uint8_t> compressed)
{
    warmCache_.put(key, CompressedChunk{std::move(compressed)});
}

// =============================================================================
// Promotion helpers
// =============================================================================

ChunkDataPtr TieredChunkCache::promoteFromWarm(
    const ChunkKey& key, CompressedChunk warm)
{
    if (!decompress_) return nullptr;

    auto data = decompress_(warm.data, key);
    if (!data) return nullptr;

    hotPut(key, data);
    return data;
}

ChunkDataPtr TieredChunkCache::promoteFromCold(const ChunkKey& key)
{
    if (!diskStore_) return nullptr;

    auto compressed = diskStore_->get(config_.volumeId, key);
    if (!compressed || compressed->empty()) return nullptr;

    statColdHits_.fetch_add(1, std::memory_order_relaxed);

    // Keep a local copy for promotion before putting into warm cache,
    // since another thread could evict from warm between put and get.
    CompressedChunk localCopy{*compressed};

    // Store in warm
    warmPut(key, std::move(*compressed));

    // Decompress and promote to hot using our local copy
    return promoteFromWarm(key, std::move(localCopy));
}

ChunkDataPtr TieredChunkCache::promoteFromIce(const ChunkKey& key)
{
    if (!source_) return nullptr;

    auto compressed = source_->fetch(key);
    if (compressed.empty()) return nullptr;

    statIceFetches_.fetch_add(1, std::memory_order_relaxed);

    // Store to cold (disk cache)
    if (diskStore_) {
        diskStore_->put(config_.volumeId, key, compressed);
    }

    // Keep a local copy for promotion before putting into warm cache,
    // since another thread could evict from warm between put and get.
    CompressedChunk localCopy{compressed};

    // Store in warm
    warmPut(key, std::move(compressed));

    // Decompress and promote to hot using our local copy
    return promoteFromWarm(key, std::move(localCopy));
}

ChunkDataPtr TieredChunkCache::loadFull(const ChunkKey& key)
{
    // Try cold (disk cache)
    auto data = promoteFromCold(key);
    if (data) return data;

    // Try ice (remote/filesystem)
    return promoteFromIce(key);
}

// =============================================================================
// Negative cache persistence
// =============================================================================

void TieredChunkCache::loadNegativeCache()
{
    if (!diskStore_) return;

    auto path = diskStore_->root() / (config_.volumeId + ".negative");

    // Check file age — skip if older than 7 days
    {
        std::error_code ec;
        auto ftime = std::filesystem::last_write_time(path, ec);
        if (ec) return;  // file doesn't exist or can't stat

        auto sctp = std::chrono::file_clock::to_sys(ftime);
        auto age = std::chrono::system_clock::now() - sctp;
        constexpr auto maxAge = std::chrono::hours(7 * 24);
        if (age > maxAge) {
            if (auto* log = cacheDebugLog())
                std::fprintf(log, "Negative cache file older than 7 days, skipping load\n");
            std::filesystem::remove(path, ec);
            return;
        }
    }

    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return;

    int32_t level, iz, iy, ix;
    size_t count = 0;
    while (f.read(reinterpret_cast<char*>(&level), 4) &&
           f.read(reinterpret_cast<char*>(&iz), 4) &&
           f.read(reinterpret_cast<char*>(&iy), 4) &&
           f.read(reinterpret_cast<char*>(&ix), 4)) {
        negativeCache_.insert(ChunkKey{level, iz, iy, ix});
        count++;
    }
    if (count > 0) {
        if (auto* log = cacheDebugLog())
            std::fprintf(log, "Loaded %zu negative cache entries from disk\n", count);
    }
}

void TieredChunkCache::saveNegativeCache() const
{
    if (!diskStore_ || negativeCache_.empty()) return;

    auto path = diskStore_->root() / (config_.volumeId + ".negative");
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f.is_open()) return;

    std::shared_lock lock(negativeMutex_);
    for (const auto& key : negativeCache_) {
        int32_t level = key.level, iz = key.iz, iy = key.iy, ix = key.ix;
        f.write(reinterpret_cast<const char*>(&level), 4);
        f.write(reinterpret_cast<const char*>(&iz), 4);
        f.write(reinterpret_cast<const char*>(&iy), 4);
        f.write(reinterpret_cast<const char*>(&ix), 4);
    }
    if (auto* log = cacheDebugLog())
        std::fprintf(log, "Saved %zu negative cache entries to disk\n",
                     negativeCache_.size());
}

}  // namespace vc::cache
