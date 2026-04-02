#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

#include "ChunkData.hpp"
#include "ChunkKey.hpp"
#include "ChunkSource.hpp"
#include "DiskStore.hpp"
#include "IOPool.hpp"
#include <utils/lru_cache.hpp>

namespace vc::cache {

// Multi-tiered chunk cache with three storage levels:
//
//   HOT   — decompressed in RAM, ready to sample (ChunkDataPtr)
//   COLD  — on local disk, persistent between runs (DiskStore)
//   ICE   — remote source, S3/HTTP/filesystem (ChunkSource)
//
// Promotion path:  ice → cold → hot  (decompression inline on IO thread)
// Eviction:        hot entries removed (cold still has on-disk copy).
//
// The coarsest pyramid level is stored in a dedicated flat array
// (coarseData_) that is always in memory — no cache lookup or eviction.
// This guarantees getBestAvailable() always returns data.
//
// Thread safety:
//   - get() / getBestAvailable(): safe from any thread
//   - getBlocking(): safe from any thread (may block)
//   - prefetch(): safe from any thread (non-blocking)
//   - Hot tier: sharded LRU (many readers, occasional writer)
class TieredChunkCache {
public:
    struct Config {
        size_t hotMaxBytes = 10ULL << 30;    // 10 GB
        std::string volumeId;                // for disk store keying
        int ioThreads = 8;
        size_t ioQueueSize = 10000000;  // max pending IO tasks
    };

    // source: where to fetch chunks (filesystem, HTTP, etc.)
    // decompress: converts compressed bytes → ChunkData
    // diskStore: shared disk cache (may be nullptr to disable cold tier)
    TieredChunkCache(
        Config config,
        std::unique_ptr<ChunkSource> source,
        DecompressFn decompress,
        std::shared_ptr<DiskStore> diskStore = nullptr);

    ~TieredChunkCache();

    TieredChunkCache(const TieredChunkCache&) = delete;
    TieredChunkCache& operator=(const TieredChunkCache&) = delete;

    // --- Non-blocking reads ---

    // Returns immediately. Returns nullptr on miss (hot cache miss).
    // Does NOT trigger a fetch. Use prefetch() to schedule background loading.
    [[nodiscard]] ChunkDataPtr get(const ChunkKey& key);

    // Returns the best available data, searching from the requested level
    // up to the coarsest. Returns {data, actualLevel}.
    // The coarsest level is always pinned hot, so this never returns nullptr
    // (after loadCoarseLevel() has been called).
    [[nodiscard]] std::pair<ChunkDataPtr, int> getBestAvailable(const ChunkKey& key);

    // --- Blocking reads ---

    // Blocks until the chunk is available (loads from cold/ice if needed).
    [[nodiscard]] ChunkDataPtr getBlocking(const ChunkKey& key);

    // --- Async prefetch ---

    // Schedule background fetch/promotion from cold/ice tier. No-op if the
    // chunk is already ready for non-blocking access.
    void prefetch(const ChunkKey& key);
    void prefetch(const std::vector<ChunkKey>& keys);

    // Prefetch all chunks needed for a region at a given level.
    void prefetchRegion(int level, int iz0, int iy0, int ix0,
                        int iz1, int iy1, int ix1);

    // Prefetch an entire pyramid level. Non-blocking — feeds chunks into
    // the IO queue in batches from a background thread so the queue doesn't
    // overflow. Calls progressCb(fetched, total) periodically from the
    // background thread if provided.
    using PrefetchProgressCb = std::function<void(int fetched, int total)>;
    void prefetchLevel(int level, const PrefetchProgressCb& progressCb = nullptr);

    // Scan the coarsest level for all-zero chunks and propagate zero-knowledge
    // to all finer levels by adding descendant keys to the negative cache.
    // Call after loadCoarseLevel(). Eliminates network requests for chunks
    // that are known-empty from the parent pyramid level.
    void propagateZeroChunks(int coarseLevel);

    // Cancel all pending (not in-flight) prefetch tasks.
    void cancelPendingPrefetch();

    // --- Cache management ---

    // Load the coarsest level into dedicated storage. Blocking.
    // After this, getCoarse() returns data directly without cache lookup.
    void loadCoarseLevel(int level);

    // Direct access to coarsest level data. Returns nullptr if not loaded
    // or out of range.
    [[nodiscard]] ChunkDataPtr getCoarse(const ChunkKey& key) const noexcept;

    // Clear hot tier. Does not touch cold/ice.
    void clearMemory();

    // Clear everything including disk cache for this volume.
    void clearAll();

    // Number of pyramid levels in the source.
    [[nodiscard]] int numLevels() const noexcept;

    // Chunk shape at a given level, in {z, y, x} order.
    [[nodiscard]] std::array<int, 3> chunkShape(int level) const noexcept;

    // Full dataset shape at a given level, in {z, y, x} order.
    [[nodiscard]] std::array<int, 3> levelShape(int level) const noexcept;

    // Persist negative-cache entries to disk without destroying the cache.
    // Call after bulk operations (e.g. level-5 priming) so reopening the same
    // remote volume doesn't re-download empty chunks.
    void flushPersistentState();

    // --- Logical data bounds (level-0 voxel coords, x/y/z order) ---
    // Physical volume bounds set from volume shape.
    // Used by CacheParams/ChunkSampler to skip chunks in zero-padded regions.
    struct DataBoundsL0 {
        int minX = 0, maxX = 0;
        int minY = 0, maxY = 0;
        int minZ = 0, maxZ = 0;
        bool valid = false;

        constexpr bool operator==(const DataBoundsL0&) const noexcept = default;
    };

    void setDataBounds(int minX, int maxX, int minY, int maxY, int minZ, int maxZ);
    [[nodiscard]] DataBoundsL0 dataBounds() const;

    // Check if a chunk is negative-cached (known to not exist on source).
    // In zarr format, missing chunks contain the fill value (zeros),
    // so callers should treat negative-cached chunks as available.
    [[nodiscard]] bool isNegativeCached(const ChunkKey& key) const;

    // Batch check: are ALL chunks in a region locally available (no remote
    // fetch needed)?  Includes hot tier, negative-cached chunks, AND
    // cold-disk-only entries.
    [[nodiscard]] bool areAllCachedInRegion(int level,
                              int iz0, int iy0, int ix0,
                              int iz1, int iy1, int ix1) const;

    // Count how many of the given keys are locally available (no remote
    // fetch needed): hot tier, negative-cached, or on disk.
    [[nodiscard]] size_t countAvailable(const std::vector<ChunkKey>& keys) const;

    // --- Notifications ---

    // Called from IOPool worker thread when a chunk becomes available.
    // Caller should bounce to main thread (e.g., via QMetaObject::invokeMethod).
    using ChunkReadyCallback = std::function<void(const ChunkKey&)>;
    using ChunkReadyCallbackId = uint64_t;

    // Register a chunk-ready listener. Returns an ID for removal.
    // Multiple listeners can coexist; all are called on each notification.
    [[nodiscard]] ChunkReadyCallbackId addChunkReadyListener(ChunkReadyCallback cb);
    void removeChunkReadyListener(ChunkReadyCallbackId id);

    // Clear the chunk-arrived debounce flag. Call this after processing
    // the chunk-ready callback to allow the next notification.
    void clearChunkArrivedFlag() noexcept;

    // --- Stats ---

    struct Stats {
        uint64_t hotHits = 0;
        uint64_t coldHits = 0;
        uint64_t iceFetches = 0;
        uint64_t misses = 0;       // non-blocking misses (hot miss)
        uint64_t hotEvictions = 0;
        size_t hotBytes = 0;
        size_t ioPending = 0;       // pending + in-flight IO tasks
        size_t diskFiles = 0;   // total chunk files on disk (all sessions)
        size_t diskBytes = 0;   // total bytes on disk
        uint64_t diskWrites = 0; // disk writes this session
        size_t negativeCount = 0; // chunks known to be empty/missing
    };

    [[nodiscard]] Stats stats() const;

private:
    // --- Hot tier (decompressed in RAM) ---
    // Sharded across 16 independent LRU caches to eliminate mutex
    // atomic contention on concurrent reads (12% CPU on aarch64).
    utils::ShardedLRUCache<ChunkKey, ChunkDataPtr, ChunkKeyHash> hotCache_;

    [[nodiscard]] ChunkDataPtr hotGet(const ChunkKey& key);
    void hotPut(const ChunkKey& key, ChunkDataPtr data);

    // --- Config (must be declared before ioPool_ for initialization order) ---
    Config config_;

    // --- Cold tier ---
    std::shared_ptr<DiskStore> diskStore_;

    // --- Ice tier ---
    std::unique_ptr<ChunkSource> source_;

    // --- Decompression ---
    DecompressFn decompress_;

    // --- I/O pool ---
    IOPool ioPool_;

    // --- Negative cache: chunks confirmed not to exist at the source ---
    // Prevents repeated S3 round-trips for out-of-bounds / sparse chunks.
    // Persisted to disk alongside the cold tier so it survives restarts.
    //
    // Fast path: atomic bloom filter rejects most lookups without taking the
    // mutex. The bloom filter has false positives but no false negatives,
    // so a negative bloom result means "definitely not in negative cache"
    // (i.e. the chunk might exist -- proceed to fetch).
    static constexpr size_t kBloomBits = 65536;  // 8 KB, ~1% FPR at 5000 entries
    std::array<std::atomic<uint64_t>, kBloomBits / 64> negativeBloom_{};
    void bloomAdd(const ChunkKey& key) noexcept;
    [[nodiscard]] bool bloomMayContain(const ChunkKey& key) const noexcept;
    void bloomClear() noexcept;
    mutable std::mutex negativeMutex_;
    std::unordered_set<ChunkKey, ChunkKeyHash> negativeCache_;
    void loadNegativeCache();
    void saveNegativeCache() const;

    // --- Dedicated storage for the coarsest pyramid level ---
    // Always in memory, no cache lookup needed.
    // Indexed by flattened (iz * gridY * gridX + iy * gridX + ix).
    int coarseLevel_ = -1;
    std::array<int, 3> coarseGrid_ = {0, 0, 0};
    std::vector<ChunkDataPtr> coarseData_;

    // --- Promotion helpers ---

    // Load from cold → hot. Returns the decompressed data.
    [[nodiscard]] ChunkDataPtr promoteFromCold(const ChunkKey& key);

    // Fetch from ice → cold → hot. Returns the decompressed data.
    [[nodiscard]] ChunkDataPtr promoteFromIce(const ChunkKey& key);

    // Full promotion chain (checks each tier in order).
    [[nodiscard]] ChunkDataPtr loadFull(const ChunkKey& key);

    // Ready for non-blocking access by get(): hot or negative-cached.
    [[nodiscard]] bool isReadyForNonBlockingRead(const ChunkKey& key) const;

    // Locally available without a remote fetch: hot, negative-cached,
    // OR present on disk (cold tier).
    [[nodiscard]] bool isAvailableWithoutRemoteFetch(const ChunkKey& key) const;

    mutable std::mutex callbackMutex_;
    std::vector<std::pair<ChunkReadyCallbackId, ChunkReadyCallback>> chunkReadyListeners_;
    std::atomic<ChunkReadyCallbackId> nextListenerId_{1};
    std::atomic<bool> chunkArrivedFlag_{false};

    // --- Logical data bounds ---
    mutable std::mutex dataBoundsMutex_;
    DataBoundsL0 dataBoundsL0_;

    // --- Stats ---
    mutable std::atomic<uint64_t> statHotHits_{0};
    std::atomic<uint64_t> statColdHits_{0};
    std::atomic<uint64_t> statIceFetches_{0};
    std::atomic<uint64_t> statDiskWrites_{0};
    mutable std::atomic<uint64_t> statMisses_{0};
};

}  // namespace vc::cache
