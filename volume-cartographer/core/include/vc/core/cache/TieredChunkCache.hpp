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
#include "IOPool.hpp"
#include <utils/lru_cache.hpp>
#include <utils/zarr.hpp>

namespace vc::cache {

// Multi-tiered chunk cache with three storage levels:
//
//   HOT   — decompressed in RAM, ready to sample (ChunkDataPtr)
//   COLD  — local zarr v3 sharded store on disk
//   ICE   — remote source, S3/HTTP/filesystem (ChunkSource)
//
// Promotion path:  ice → cold → hot  (decompression inline on IO thread)
// Eviction:        hot entries removed (cold still has on-disk copy).
class TieredChunkCache {
public:
    struct Config {
        size_t hotMaxBytes = 10ULL << 30;    // 10 GB
        std::string volumeId;                // for disk store keying
        int ioThreads = 8;
        size_t ioQueueSize = 10000000;
    };

    // diskZarr: local zarr v3 sharded array for cold tier (may be nullptr)
    TieredChunkCache(
        Config config,
        std::unique_ptr<ChunkSource> source,
        DecompressFn decompress,
        std::vector<std::shared_ptr<utils::ZarrArray>> diskLevels = {});

    ~TieredChunkCache();

    TieredChunkCache(const TieredChunkCache&) = delete;
    TieredChunkCache& operator=(const TieredChunkCache&) = delete;

    // --- Non-blocking reads ---
    [[nodiscard]] ChunkDataPtr get(const ChunkKey& key);
    [[nodiscard]] std::pair<ChunkDataPtr, int> getBestAvailable(const ChunkKey& key);

    // --- Blocking reads ---
    [[nodiscard]] ChunkDataPtr getBlocking(const ChunkKey& key);

    // --- Async prefetch (background queue) ---
    void prefetch(const ChunkKey& key);
    void prefetch(const std::vector<ChunkKey>& keys);
    void prefetchRegion(int level, int iz0, int iy0, int ix0,
                        int iz1, int iy1, int ix1);

    // --- Interactive fetch (high priority queue, for viewport chunks) ---
    void fetchInteractive(const std::vector<ChunkKey>& keys);

    using PrefetchProgressCb = std::function<void(int fetched, int total)>;
    void prefetchLevel(int level, const PrefetchProgressCb& progressCb = nullptr);

    // Bulk download whole shards directly to disk cache. Skips shards already on disk.
    void prefetchShardsLevel(int level, const PrefetchProgressCb& progressCb = nullptr);

    void propagateZeroChunks(int coarseLevel);
    void cancelPendingPrefetch();

    // --- Cache management ---
    void loadCoarseLevel(int level);
    [[nodiscard]] ChunkDataPtr getCoarse(const ChunkKey& key) const noexcept;
    void clearMemory();
    void clearAll();

    [[nodiscard]] int numLevels() const noexcept;
    [[nodiscard]] std::array<int, 3> chunkShape(int level) const noexcept;
    [[nodiscard]] std::array<int, 3> levelShape(int level) const noexcept;

    void flushPersistentState();

    // --- Logical data bounds ---
    struct DataBoundsL0 {
        int minX = 0, maxX = 0;
        int minY = 0, maxY = 0;
        int minZ = 0, maxZ = 0;
        bool valid = false;
        constexpr bool operator==(const DataBoundsL0&) const noexcept = default;
    };

    void setDataBounds(int minX, int maxX, int minY, int maxY, int minZ, int maxZ);
    [[nodiscard]] DataBoundsL0 dataBounds() const;

    [[nodiscard]] bool isNegativeCached(const ChunkKey& key) const;
    [[nodiscard]] bool areAllCachedInRegion(int level,
                              int iz0, int iy0, int ix0,
                              int iz1, int iy1, int ix1) const;
    [[nodiscard]] size_t countAvailable(const std::vector<ChunkKey>& keys) const;

    // --- Notifications ---
    using ChunkReadyCallback = std::function<void(const ChunkKey&)>;
    using ChunkReadyCallbackId = uint64_t;

    [[nodiscard]] ChunkReadyCallbackId addChunkReadyListener(ChunkReadyCallback cb);
    void removeChunkReadyListener(ChunkReadyCallbackId id);
    void clearChunkArrivedFlag() noexcept;

    // --- Stats ---
    struct Stats {
        uint64_t hotHits = 0;
        uint64_t coldHits = 0;
        uint64_t iceFetches = 0;
        uint64_t misses = 0;
        uint64_t hotEvictions = 0;
        size_t hotBytes = 0;
        size_t ioPending = 0;       // total (interactive + prefetch)
        size_t ioInteractive = 0;   // interactive queue only
        size_t ioPrefetch = 0;      // prefetch queue only
        uint64_t diskWrites = 0;
        size_t negativeCount = 0;
        size_t diskBytes = 0;    // total bytes on disk across all level shards
        size_t diskShards = 0;   // number of shard files on disk
        uint64_t totalSubmitted = 0;  // total chunks ever submitted to IO pool
        bool sharded = false;    // true if source uses sharded zarr
    };

    [[nodiscard]] Stats stats() const;
    [[nodiscard]] bool coarseLevelReady() const noexcept { return coarseLevel_ >= 0; }

private:
    utils::ShardedLRUCache<ChunkKey, ChunkDataPtr, ChunkKeyHash> hotCache_;

    [[nodiscard]] ChunkDataPtr hotGet(const ChunkKey& key);
    void hotPut(const ChunkKey& key, ChunkDataPtr data);

    Config config_;

    // --- Cold tier (per-level sharded zarr v3 arrays: path/0/, path/1/, ...) ---
    std::vector<std::shared_ptr<utils::ZarrArray>> diskLevels_;

    // --- Ice tier ---
    std::unique_ptr<ChunkSource> source_;

    // --- Decompression ---
    DecompressFn decompress_;

    // --- I/O pool ---
    IOPool ioPool_;

    // --- Negative cache ---
    static constexpr size_t kBloomBits = 65536;
    std::array<std::atomic<uint64_t>, kBloomBits / 64> negativeBloom_{};
    void bloomAdd(const ChunkKey& key) noexcept;
    [[nodiscard]] bool bloomMayContain(const ChunkKey& key) const noexcept;
    void bloomClear() noexcept;
    mutable std::mutex negativeMutex_;
    std::unordered_set<ChunkKey, ChunkKeyHash> negativeCache_;
    void loadNegativeCache();
    void saveNegativeCache() const;

    // --- Coarsest level storage ---
    int coarseLevel_ = -1;
    std::array<int, 3> coarseGrid_ = {0, 0, 0};
    std::vector<ChunkDataPtr> coarseData_;

    // --- Promotion helpers ---
    [[nodiscard]] ChunkDataPtr promoteFromCold(const ChunkKey& key);
    [[nodiscard]] ChunkDataPtr promoteFromIce(const ChunkKey& key);
    [[nodiscard]] ChunkDataPtr loadFull(const ChunkKey& key);
    [[nodiscard]] bool isReadyForNonBlockingRead(const ChunkKey& key) const;
    [[nodiscard]] bool isAvailableWithoutRemoteFetch(const ChunkKey& key) const;

    mutable std::mutex callbackMutex_;
    std::vector<std::pair<ChunkReadyCallbackId, ChunkReadyCallback>> chunkReadyListeners_;
    std::atomic<ChunkReadyCallbackId> nextListenerId_{1};
    std::atomic<bool> chunkArrivedFlag_{false};

    mutable std::mutex dataBoundsMutex_;
    DataBoundsL0 dataBoundsL0_;

    mutable std::atomic<uint64_t> statHotHits_{0};
    std::atomic<uint64_t> statColdHits_{0};
    std::atomic<uint64_t> statIceFetches_{0};
    std::atomic<uint64_t> statDiskWrites_{0};
    std::atomic<uint64_t> statTotalSubmitted_{0};
    mutable std::atomic<uint64_t> statMisses_{0};
};

}  // namespace vc::cache
