#include "vc/core/cache/TieredChunkCache.hpp"
#include "vc/core/cache/CacheDebugLog.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>

namespace vc::cache {

// Convert ChunkKey to zarr chunk indices (z, y, x)
static std::array<size_t, 3> chunkIndices(const ChunkKey& key) {
    return {static_cast<size_t>(key.iz), static_cast<size_t>(key.iy), static_cast<size_t>(key.ix)};
}

// Read compressed chunk from ZarrArray (handles both sharded and unsharded)
static std::optional<std::vector<std::byte>> zarrReadChunk(
    utils::ZarrArray& zarr, const ChunkKey& key)
{
    auto idx = chunkIndices(key);
    if (zarr.is_sharded()) {
        return zarr.read_inner_chunk_from_shard(idx);
    }
    return zarr.read_chunk_raw(idx);
}

// Write compressed chunk to ZarrArray (handles both sharded and unsharded)
static void zarrWriteChunk(
    utils::ZarrArray& zarr, const ChunkKey& key,
    const uint8_t* data, size_t size)
{
    auto idx = chunkIndices(key);
    std::span<const std::byte> byteSpan(
        reinterpret_cast<const std::byte*>(data), size);
    if (zarr.is_sharded()) {
        zarr.write_inner_chunk_to_shard(idx, byteSpan);
    } else {
        zarr.write_chunk_raw(idx, byteSpan);
    }
}

// Check if chunk exists in ZarrArray
static bool zarrChunkExists(const utils::ZarrArray& zarr, const ChunkKey& key)
{
    auto idx = chunkIndices(key);
    if (zarr.is_sharded()) {
        return zarr.inner_chunk_exists(idx);
    }
    return zarr.read_chunk_raw(idx).has_value();
}

static bool isAllZero(const uint8_t* data, size_t size) noexcept
{
    const auto* p = reinterpret_cast<const uint64_t*>(data);
    size_t n8 = size / 8;
    for (size_t i = 0; i < n8; i++)
        if (p[i] != 0) return false;
    for (size_t i = n8 * 8; i < size; i++)
        if (data[i] != 0) return false;
    return true;
}

// Helper to build LRUCache config for the hot tier
static auto makeHotConfig(const TieredChunkCache::Config& cfg) {
    using HotCache = utils::LRUCache<ChunkKey, ChunkDataPtr, ChunkKeyHash>;
    typename HotCache::Config c;
    c.max_bytes = cfg.hotMaxBytes;
    c.evict_target = 15.0 / 16.0;
    c.promote_on_read = false;  // VC3D pattern: no LRU churn on reads
    c.size_fn = [](const ChunkDataPtr& p) -> std::size_t {
        return p ? p->totalBytes() : 0;
    };
    return c;
}

TieredChunkCache::TieredChunkCache(
    Config config,
    std::unique_ptr<ChunkSource> source,
    DecompressFn decompress,
    std::vector<std::shared_ptr<utils::ZarrArray>> diskLevels)
    : hotCache_(makeHotConfig(config))
    , config_(std::move(config))
    , diskLevels_(std::move(diskLevels))
    , source_(std::move(source))
    , decompress_(std::move(decompress))
    , ioPool_(config_.ioThreads)
{
    // Shared fetch function: check cold (disk) first, then ice (remote)
    IOPool::FetchFunc fetchFunc = [this](const ChunkKey& key) -> std::vector<uint8_t> {
        using Clock = std::chrono::steady_clock;
        auto t0 = Clock::now();

        // Try cold (disk cache) first
        auto* dz = (key.level < static_cast<int>(diskLevels_.size())) ? diskLevels_[key.level].get() : nullptr;
        if (dz) {
            auto diskData = zarrReadChunk(*dz, key);
            if (diskData && !diskData->empty()) {
                auto n = statColdHits_.fetch_add(1, std::memory_order_relaxed);
                auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t0).count();
                if (n < 10 || (n < 1000 && n % 100 == 0))
                    std::fprintf(stderr, "[Cache] cold-hit #%lu lvl=%d (%d,%d,%d) bytes=%zu %ldms\n",
                                 n + 1, key.level, key.iz, key.iy, key.ix, diskData->size(), ms);
                if (auto* log = cacheDebugLog())
                    std::fprintf(log, "FETCH cold-hit lvl=%d (%d,%d,%d) bytes=%zu %ldms\n",
                                 key.level, key.iz, key.iy, key.ix, diskData->size(), ms);
                // Convert std::byte → uint8_t
                auto& bytes = *diskData;
                return {reinterpret_cast<const uint8_t*>(bytes.data()),
                        reinterpret_cast<const uint8_t*>(bytes.data() + bytes.size())};
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

        // Detect all-zero chunks (zarr fill_value=0 for non-existent chunks).
        if (isAllZero(data.data(), data.size())) {
            if (auto* log = cacheDebugLog())
                std::fprintf(log, "FETCH ice-allzero lvl=%d (%d,%d,%d) bytes=%zu %ldms\n",
                             key.level, key.iz, key.iy, key.ix, data.size(), fetchMs);
            return {};
        }

        auto n = statIceFetches_.fetch_add(1, std::memory_order_relaxed);
        if (n < 10 || (n < 1000 && n % 100 == 0))
            std::fprintf(stderr, "[Cache] ice-fetch #%lu lvl=%d (%d,%d,%d) bytes=%zu %ldms\n",
                         n + 1, key.level, key.iz, key.iy, key.ix, data.size(), fetchMs);

        // Store to cold (disk cache) for persistence.
        auto* dzW = (key.level < static_cast<int>(diskLevels_.size())) ? diskLevels_[key.level].get() : nullptr;
        if (dzW) {
            zarrWriteChunk(*dzW, key, data.data(), data.size());
            statDiskWrites_.fetch_add(1, std::memory_order_relaxed);
        }

        auto totalMs = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t0).count();
        if (auto* log = cacheDebugLog())
            std::fprintf(log, "FETCH ice-ok lvl=%d (%d,%d,%d) bytes=%zu fetch=%ldms total=%ldms\n",
                         key.level, key.iz, key.iy, key.ix, data.size(), fetchMs, totalMs);

        return data;
    };

    // Shared completion callback: decompress inline on IO thread, then hotPut.
    IOPool::CompletionCallback completionCb =
        [this](const ChunkKey& key, std::vector<uint8_t>&& compressed) {
            if (compressed.empty()) {
                // Empty fetch result: negative cache (chunk doesn't exist)
                bloomAdd(key);
                {
                    std::lock_guard lock(negativeMutex_);
                    negativeCache_.insert(key);
                }
                if (auto* log = cacheDebugLog())
                    std::fprintf(log, "COMPLETE empty lvl=%d (%d,%d,%d) [negative cached]\n",
                                 key.level, key.iz, key.iy, key.ix);
                return;
            }

            // Decompress inline on the IO thread and put directly into hot tier
            if (decompress_) {
                using Clock = std::chrono::steady_clock;
                auto t0 = Clock::now();
                auto data = decompress_(compressed, key);
                auto decompMs = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t0).count();
                if (data) {
                    size_t decompBytes = data->totalBytes();
                    hotPut(key, std::move(data));
                    if (auto* log = cacheDebugLog())
                        std::fprintf(log, "COMPLETE hot-put lvl=%d (%d,%d,%d) decompBytes=%zu decomp=%ldms\n",
                                     key.level, key.iz, key.iy, key.ix, decompBytes, decompMs);
                } else {
                    if (auto* log = cacheDebugLog())
                        std::fprintf(log, "COMPLETE decompress-fail lvl=%d (%d,%d,%d)\n",
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
        };

    // Wire up IO pool with fetch and completion logic
    ioPool_.setFetchFunc(fetchFunc);
    ioPool_.setCompletionCallback(std::move(completionCb));

    // Start IO workers after callbacks are set to avoid data races.
    ioPool_.start();

    loadNegativeCache();
}

TieredChunkCache::~TieredChunkCache()
{
    ioPool_.stop();

    // Print final fetch stats summary
    auto cold = statColdHits_.load(std::memory_order_relaxed);
    auto ice = statIceFetches_.load(std::memory_order_relaxed);
    if (cold > 0 || ice > 0) {
        std::fprintf(stderr, "[Cache] session summary: coldHits=%lu iceFetches=%lu (%.0f%% from disk)\n",
                     cold, ice, (cold + ice) > 0 ? 100.0 * cold / (cold + ice) : 0.0);
    }
    saveNegativeCache();
}

// =============================================================================
// Bloom filter for negative cache (lock-free fast path)
// =============================================================================

void TieredChunkCache::bloomAdd(const ChunkKey& key) noexcept
{
    auto h = ChunkKeyHash{}(key);
    // Two hash functions derived from the single hash via golden ratio mixing
    auto h1 = h;
    auto h2 = h * 0x9E3779B97F4A7C15ULL;
    auto idx1 = h1 % kBloomBits;
    auto idx2 = h2 % kBloomBits;
    negativeBloom_[idx1 / 64].fetch_or(1ULL << (idx1 % 64), std::memory_order_relaxed);
    negativeBloom_[idx2 / 64].fetch_or(1ULL << (idx2 % 64), std::memory_order_relaxed);
}

bool TieredChunkCache::bloomMayContain(const ChunkKey& key) const noexcept
{
    auto h = ChunkKeyHash{}(key);
    auto h1 = h;
    auto h2 = h * 0x9E3779B97F4A7C15ULL;
    auto idx1 = h1 % kBloomBits;
    auto idx2 = h2 % kBloomBits;
    auto b1 = negativeBloom_[idx1 / 64].load(std::memory_order_relaxed) & (1ULL << (idx1 % 64));
    auto b2 = negativeBloom_[idx2 / 64].load(std::memory_order_relaxed) & (1ULL << (idx2 % 64));
    return b1 && b2;
}

void TieredChunkCache::bloomClear() noexcept
{
    for (auto& word : negativeBloom_)
        word.store(0, std::memory_order_relaxed);
}

// =============================================================================
// Non-blocking reads
// =============================================================================

ChunkDataPtr TieredChunkCache::get(const ChunkKey& key)
{
    // Direct access for coarsest level — no cache lookup
    auto coarse = getCoarse(key);
    if (coarse) return coarse;

    // Check hot tier
    auto hot = hotGet(key);
    if (hot) {
        statHotHits_.fetch_add(1, std::memory_order_relaxed);
        return hot;
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
    // Direct access for coarsest level — no cache lookup
    auto coarse = getCoarse(key);
    if (coarse) return coarse;

    // Fast path: hot cache hit. This is the common case during rendering —
    // just hash + shard + lock_guard + flat-map probe + return.
    auto hot = hotGet(key);
    if (hot) {
        statHotHits_.fetch_add(1, std::memory_order_relaxed);
        return hot;
    }

    // Known non-existent?
    if (isNegativeCached(key)) return nullptr;

    // Full promotion chain: cold → hot, or ice → cold → hot
    auto data = loadFull(key);
    if (!data) {
        bloomAdd(key);
        std::lock_guard lock(negativeMutex_);
        negativeCache_.insert(key);
    }
    return data;
}

// =============================================================================
// Async prefetch
// =============================================================================

void TieredChunkCache::prefetch(const ChunkKey& key)
{
    // Already ready for non-blocking use (or known sparse)? No-op.
    // Cold-disk-only chunks still need promotion, so they must be submitted.
    if (isReadyForNonBlockingRead(key)) return;

    ioPool_.submit(key);
}

void TieredChunkCache::prefetch(const std::vector<ChunkKey>& keys)
{
    if (keys.empty()) return;

    std::vector<ChunkKey> submitKeys;
    submitKeys.reserve(keys.size());
    for (const auto& key : keys) {
        if (!isReadyForNonBlockingRead(key)) {
            submitKeys.push_back(key);
        }
    }

    if (!submitKeys.empty()) {
        ioPool_.submit(submitKeys);
    }
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

    // Filter out negative-cached keys
    std::vector<ChunkKey> keys;
    if (!missingHot.empty()) {
        keys.reserve(missingHot.size());
        for (const auto& key : missingHot) {
            if (!isReadyForNonBlockingRead(key)) {
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
        ioPool_.submitBackground(keys);
    }
}

void TieredChunkCache::prefetchLevel(int level, const PrefetchProgressCb& progressCb)
{
    if (level < 0 || level >= numLevels()) return;

    auto shape = levelShape(level);
    auto chunks = chunkShape(level);
    int gridZ = (shape[0] + chunks[0] - 1) / chunks[0];
    int gridY = (shape[1] + chunks[1] - 1) / chunks[1];
    int gridX = (shape[2] + chunks[2] - 1) / chunks[2];
    int total = gridZ * gridY * gridX;

    if (auto* log = cacheDebugLog())
        std::fprintf(log, "prefetchLevel: level=%d grid=%dx%dx%d total=%d\n",
                     level, gridZ, gridY, gridX, total);

    std::vector<ChunkKey> keys;
    keys.reserve(total);
    int skipped = 0;
    for (int iz = 0; iz < gridZ; iz++)
        for (int iy = 0; iy < gridY; iy++)
            for (int ix = 0; ix < gridX; ix++) {
                ChunkKey key{level, iz, iy, ix};
                // Skip chunks already known to be empty (from propagateZeroChunks)
                if (isNegativeCached(key)) {
                    skipped++;
                } else {
                    keys.push_back(key);
                }
            }

    if (!keys.empty())
        ioPool_.submitBackground(keys);

    std::fprintf(stderr, "[Volume] Level %d: %d to fetch, %d skipped (known empty)\n",
                 level, static_cast<int>(keys.size()), skipped);

    if (progressCb)
        progressCb(total, total);
}

void TieredChunkCache::cancelPendingPrefetch()
{
    ioPool_.cancelPending();
}

// =============================================================================
// Coarse level — dedicated flat storage
// =============================================================================

void TieredChunkCache::loadCoarseLevel(int level)
{
    auto shape = levelShape(level);
    auto chunks = chunkShape(level);
    std::array<int, 3> grid = {
        (shape[0] + chunks[0] - 1) / chunks[0],
        (shape[1] + chunks[1] - 1) / chunks[1],
        (shape[2] + chunks[2] - 1) / chunks[2]
    };
    int total = grid[0] * grid[1] * grid[2];
    if (total > 10000) {
        std::fprintf(stderr, "[Cache] WARNING: coarse level has %d chunks, skipping flat storage\n", total);
        return;
    }
    coarseGrid_ = grid;
    coarseData_.resize(total);

    for (int iz = 0; iz < coarseGrid_[0]; iz++) {
        for (int iy = 0; iy < coarseGrid_[1]; iy++) {
            for (int ix = 0; ix < coarseGrid_[2]; ix++) {
                ChunkKey key{level, iz, iy, ix};
                auto data = getBlocking(key);
                int idx = iz * coarseGrid_[1] * coarseGrid_[2] + iy * coarseGrid_[2] + ix;
                coarseData_[idx] = std::move(data);
            }
        }
    }
    coarseLevel_ = level;
    std::fprintf(stderr, "[Cache] loadCoarseLevel: level=%d grid=%dx%dx%d (%d chunks)\n",
                 level, coarseGrid_[0], coarseGrid_[1], coarseGrid_[2], total);
}

ChunkDataPtr TieredChunkCache::getCoarse(const ChunkKey& key) const noexcept
{
    if (key.level != coarseLevel_ || coarseData_.empty()) return nullptr;
    if (key.iz < 0 || key.iz >= coarseGrid_[0] ||
        key.iy < 0 || key.iy >= coarseGrid_[1] ||
        key.ix < 0 || key.ix >= coarseGrid_[2]) return nullptr;
    int idx = key.iz * coarseGrid_[1] * coarseGrid_[2] + key.iy * coarseGrid_[2] + key.ix;
    return coarseData_[idx];
}

void TieredChunkCache::propagateZeroChunks(int coarseLevel)
{
    if (coarseLevel < 0 || coarseLevel >= numLevels()) return;

    // If coarse data was never loaded (e.g. loadCoarseLevel bailed due to
    // >10k chunks), getCoarse() returns nullptr for everything, which would
    // incorrectly mark ALL descendant chunks as zero.  Bail out early.
    if (coarseData_.empty() || coarseLevel != coarseLevel_) return;

    auto coarseShape = levelShape(coarseLevel);
    auto coarseChunks = chunkShape(coarseLevel);
    int cGridZ = (coarseShape[0] + coarseChunks[0] - 1) / coarseChunks[0];
    int cGridY = (coarseShape[1] + coarseChunks[1] - 1) / coarseChunks[1];
    int cGridX = (coarseShape[2] + coarseChunks[2] - 1) / coarseChunks[2];

    // Find all-zero chunks at the coarse level
    std::vector<ChunkKey> zeroChunks;
    for (int iz = 0; iz < cGridZ; iz++) {
        for (int iy = 0; iy < cGridY; iy++) {
            for (int ix = 0; ix < cGridX; ix++) {
                ChunkKey key{coarseLevel, iz, iy, ix};
                auto data = getCoarse(key);
                if (!data) {
                    // Null = chunk doesn't exist = all zeros (zarr fill_value=0)
                    zeroChunks.push_back(key);
                    continue;
                }

                // Check if chunk data is all zeros
                if (isAllZero(data->rawData(), data->totalBytes())) {
                    zeroChunks.push_back(key);
                }
            }
        }
    }

    if (zeroChunks.empty()) return;

    // For each zero chunk at the coarse level, compute all descendant chunk
    // keys at every finer level and add them to the negative cache.
    // Each coarse chunk covers a 2x region at the next finer level in each axis.
    int totalNegated = 0;
    {
        std::lock_guard lock(negativeMutex_);
        for (int lvl = coarseLevel - 1; lvl >= 0; lvl--) {
            int scale = 1 << (coarseLevel - lvl);  // 2, 4, 8, ...
            auto fineShape = levelShape(lvl);
            auto fineChunks = chunkShape(lvl);
            int fGridZ = (fineShape[0] + fineChunks[0] - 1) / fineChunks[0];
            int fGridY = (fineShape[1] + fineChunks[1] - 1) / fineChunks[1];
            int fGridX = (fineShape[2] + fineChunks[2] - 1) / fineChunks[2];

            for (const auto& zk : zeroChunks) {
                // This coarse chunk covers fine chunks [zk*scale .. (zk+1)*scale-1]
                int fz0 = zk.iz * scale, fz1 = std::min(fz0 + scale, fGridZ);
                int fy0 = zk.iy * scale, fy1 = std::min(fy0 + scale, fGridY);
                int fx0 = zk.ix * scale, fx1 = std::min(fx0 + scale, fGridX);

                for (int fz = fz0; fz < fz1; fz++) {
                    for (int fy = fy0; fy < fy1; fy++) {
                        for (int fx = fx0; fx < fx1; fx++) {
                            ChunkKey fineKey{lvl, fz, fy, fx};
                            // Don't overwrite chunks already cached or on disk
                            if (hotCache_.contains(fineKey))
                                continue;
                            auto* dzFine = (fineKey.level < static_cast<int>(diskLevels_.size())) ? diskLevels_[fineKey.level].get() : nullptr;
                            if (dzFine && zarrChunkExists(*dzFine, fineKey))
                                continue;
                            negativeCache_.insert(fineKey);
                            bloomAdd(fineKey);
                            totalNegated++;
                        }
                    }
                }
            }
        }
    }

    std::fprintf(stderr, "[Cache] propagateZeroChunks: %zu zero chunks at level %d → %d descendant keys negative-cached\n",
                 zeroChunks.size(), coarseLevel, totalNegated);
}

// =============================================================================
// Cache management
// =============================================================================

void TieredChunkCache::clearMemory()
{
    hotCache_.clear();
}

void TieredChunkCache::clearAll()
{
    ioPool_.cancelPending();
    clearMemory();
    bloomClear();
    {
        std::lock_guard lock(negativeMutex_);
        negativeCache_.clear();
    }
    if (!diskLevels_.empty()) {
        std::error_code ec;
        std::filesystem::remove(
            diskLevels_[0]->path().parent_path() / (config_.volumeId + ".negative"), ec);
    }
}

int TieredChunkCache::numLevels() const noexcept
{
    return source_ ? source_->numLevels() : 0;
}

std::array<int, 3> TieredChunkCache::chunkShape(int level) const noexcept
{
    return source_ ? source_->chunkShape(level) : std::array<int, 3>{0, 0, 0};
}

std::array<int, 3> TieredChunkCache::levelShape(int level) const noexcept
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
    // Bloom filter fast-reject: if bloom says no, definitely not cached.
    if (!bloomMayContain(key)) return false;
    std::lock_guard lock(negativeMutex_);
    return negativeCache_.count(key) > 0;
}

bool TieredChunkCache::areAllCachedInRegion(
    int level,
    int iz0, int iy0, int ix0,
    int iz1, int iy1, int ix1) const
{
    // Only count chunks in HOT cache (RAM) or negative-cached.
    // Disk-only chunks still require blocking reads — don't count them
    // as "cached" for best-effort level selection.
    for (int iz = iz0; iz <= iz1; iz++) {
        for (int iy = iy0; iy <= iy1; iy++) {
            for (int ix = ix0; ix <= ix1; ix++) {
                ChunkKey key{level, iz, iy, ix};
                if (!isReadyForNonBlockingRead(key)) return false;
            }
        }
    }
    return true;
}

size_t TieredChunkCache::countAvailable(const std::vector<ChunkKey>& keys) const
{
    size_t available = 0;
    for (const auto& key : keys) {
        if (isAvailableWithoutRemoteFetch(key)) {
            available++;
        }
    }
    return available;
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

void TieredChunkCache::clearChunkArrivedFlag() noexcept
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
    s.coldHits = statColdHits_.load(std::memory_order_relaxed);
    s.iceFetches = statIceFetches_.load(std::memory_order_relaxed);
    s.misses = statMisses_.load(std::memory_order_relaxed);
    s.hotEvictions = hotCache_.evictions();
    s.hotBytes = hotCache_.byte_size();
    s.ioPending = ioPool_.pendingCount();
    s.diskWrites = statDiskWrites_.load(std::memory_order_relaxed);
    {
        std::lock_guard lock(negativeMutex_);
        s.negativeCount = negativeCache_.size();
    }
    return s;
}

// =============================================================================
// Hot tier — delegates to utils::LRUCache
// =============================================================================

ChunkDataPtr TieredChunkCache::hotGet(const ChunkKey& key)
{
    return hotCache_.get_or(key, nullptr);
}

void TieredChunkCache::hotPut(const ChunkKey& key, ChunkDataPtr data)
{
    hotCache_.put(key, std::move(data));
}

// =============================================================================
// Promotion helpers
// =============================================================================

ChunkDataPtr TieredChunkCache::promoteFromCold(const ChunkKey& key)
{
    if (!decompress_) return nullptr;
    auto* dz = (key.level < static_cast<int>(diskLevels_.size())) ? diskLevels_[key.level].get() : nullptr;
    if (!dz) return nullptr;

    auto raw = zarrReadChunk(*dz, key);
    if (!raw || raw->empty()) return nullptr;

    statColdHits_.fetch_add(1, std::memory_order_relaxed);

    std::vector<uint8_t> compressed(
        reinterpret_cast<const uint8_t*>(raw->data()),
        reinterpret_cast<const uint8_t*>(raw->data() + raw->size()));
    auto data = decompress_(compressed, key);
    if (!data) return nullptr;
    // Move into cache; shared_ptr copy kept for return
    auto ret = data;
    hotPut(key, std::move(data));
    return ret;
}

ChunkDataPtr TieredChunkCache::promoteFromIce(const ChunkKey& key)
{
    if (!source_) return nullptr;

    auto compressed = source_->fetch(key);
    if (compressed.empty()) return nullptr;

    statIceFetches_.fetch_add(1, std::memory_order_relaxed);

    // Store to cold (disk cache) — per-level zarr
    auto* dzIce = (key.level < static_cast<int>(diskLevels_.size())) ? diskLevels_[key.level].get() : nullptr;
    if (dzIce) {
        zarrWriteChunk(*dzIce, key, compressed.data(), compressed.size());
        statDiskWrites_.fetch_add(1, std::memory_order_relaxed);
    }

    // Decompress and promote to hot; release compressed bytes immediately after
    if (!decompress_) return nullptr;
    auto data = decompress_(compressed, key);
    compressed.clear();
    compressed.shrink_to_fit();
    if (!data) return nullptr;
    // Move into cache; shared_ptr copy kept for return
    auto ret = data;
    hotPut(key, std::move(data));
    return ret;
}

ChunkDataPtr TieredChunkCache::loadFull(const ChunkKey& key)
{
    // Try cold (disk cache)
    auto data = promoteFromCold(key);
    if (data) return data;

    // Try ice (remote/filesystem)
    return promoteFromIce(key);
}

bool TieredChunkCache::isReadyForNonBlockingRead(const ChunkKey& key) const
{
    if (key.level == coarseLevel_ && !coarseData_.empty()) return true;
    if (hotCache_.contains(key)) return true;

    if (isNegativeCached(key)) return true;

    return false;
}

bool TieredChunkCache::isAvailableWithoutRemoteFetch(const ChunkKey& key) const
{
    if (isReadyForNonBlockingRead(key)) return true;

    // Check cold tier (local zarr): if in shard, no remote fetch needed.
    auto* dzAvail = (key.level < static_cast<int>(diskLevels_.size())) ? diskLevels_[key.level].get() : nullptr;
    if (dzAvail && zarrChunkExists(*dzAvail, key))
        return true;
    return false;
}

void TieredChunkCache::flushPersistentState()
{
    saveNegativeCache();
}

// =============================================================================
// Negative cache persistence
// =============================================================================

void TieredChunkCache::loadNegativeCache()
{
    if (diskLevels_.empty()) return;
    auto negRoot = diskLevels_[0]->path().parent_path();

    auto path = negRoot / (config_.volumeId + ".negative");
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return;

    int32_t level, iz, iy, ix;
    size_t count = 0;
    while (f.read(reinterpret_cast<char*>(&level), 4) &&
           f.read(reinterpret_cast<char*>(&iz), 4) &&
           f.read(reinterpret_cast<char*>(&iy), 4) &&
           f.read(reinterpret_cast<char*>(&ix), 4)) {
        ChunkKey k{level, iz, iy, ix};
        negativeCache_.insert(k);
        bloomAdd(k);
        count++;
    }
    if (count > 0) {
        if (auto* log = cacheDebugLog())
            std::fprintf(log, "Loaded %zu negative cache entries from disk\n", count);
    }
}

void TieredChunkCache::saveNegativeCache() const
{
    if (negativeCache_.empty()) return;
    if (diskLevels_.empty()) return;
    auto negRoot = diskLevels_[0]->path().parent_path();

    auto path = negRoot / (config_.volumeId + ".negative");
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f.is_open()) return;

    std::lock_guard lock(negativeMutex_);
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
