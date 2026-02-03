#pragma once

#include <xtensor/containers/xarray.hpp>

#include <atomic>
#include <memory>
#include <unordered_map>
#include <vector>
#include <cstdint>
#include <mutex>
#include <shared_mutex>

// Forward declaration
namespace z5 { class Dataset; }

/**
 * @brief Thread-safe chunk cache with shared_ptr lifetime management
 *
 * @tparam T Data type of cached chunks (uint8_t or uint16_t)
 *
 * Supports caching chunks from multiple z5::Dataset instances simultaneously.
 * Chunks are stored as shared_ptr so eviction removes from the cache but
 * doesn't free memory until all readers are done.
 *
 * Uses a hash map keyed by (dataset pointer, chunk z, chunk y, chunk x).
 * LRU eviction based on generation counter.
 */
template<typename T>
class ChunkCache
{
public:
    using ChunkPtr = std::shared_ptr<xt::xarray<T>>;

    explicit ChunkCache(size_t maxBytes = 0);
    ~ChunkCache();

    ChunkCache(const ChunkCache&) = delete;
    ChunkCache& operator=(const ChunkCache&) = delete;
    ChunkCache(ChunkCache&&) = delete;
    ChunkCache& operator=(ChunkCache&&) = delete;

    void setMaxBytes(size_t maxBytes);
    size_t cachedCount() const { return _cachedCount.load(std::memory_order_relaxed); }

    // Cache statistics
    struct Stats {
        uint64_t hits = 0;
        uint64_t misses = 0;
        uint64_t evictions = 0;
        uint64_t bytesRead = 0;
    };
    Stats stats() const;
    void resetStats();

    /**
     * @brief Get a chunk, loading from disk if needed.
     * Returns shared_ptr — caller holds the chunk alive even if evicted.
     */
    ChunkPtr get(z5::Dataset* ds, int iz, int iy, int ix);

    /**
     * @brief Check if chunk is cached without loading.
     */
    ChunkPtr getIfCached(z5::Dataset* ds, int iz, int iy, int ix) const;

    void prefetch(z5::Dataset* ds, int minIz, int minIy, int minIx, int maxIz, int maxIy, int maxIx);
    void clear();
    void flush();

private:
    struct ChunkKey {
        z5::Dataset* ds;
        int iz, iy, ix;
        bool operator==(const ChunkKey& o) const {
            return ds == o.ds && iz == o.iz && iy == o.iy && ix == o.ix;
        }
    };

    struct ChunkKeyHash {
        size_t operator()(const ChunkKey& k) const {
            size_t h = std::hash<const void*>()(k.ds);
            h ^= std::hash<int>()(k.iz) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int>()(k.iy) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int>()(k.ix) + 0x9e3779b9 + (h << 6) + (h >> 2);
            return h;
        }
    };

    struct CacheEntry {
        ChunkPtr chunk;
        size_t bytes;
        uint64_t lastAccess;
    };

    size_t _maxBytes = 0;
    std::atomic<size_t> _storedBytes{0};
    std::atomic<size_t> _cachedCount{0};
    std::atomic<uint64_t> _generation{0};

    mutable std::shared_mutex _mapMutex;
    std::unordered_map<ChunkKey, CacheEntry, ChunkKeyHash> _map;

    static constexpr int kLockPoolSize = 64;
    std::mutex _lockPool[kLockPoolSize];
    std::mutex _evictionMutex;

    size_t lockIndex(const ChunkKey& k) const { return ChunkKeyHash()(k) % kLockPoolSize; }

    ChunkPtr loadChunk(z5::Dataset* ds, int iz, int iy, int ix);
    void evictIfNeeded();

    // Stats counters
    mutable std::atomic<uint64_t> _hits{0};
    std::atomic<uint64_t> _misses{0};
    std::atomic<uint64_t> _evictions{0};
    std::atomic<uint64_t> _bytesRead{0};
};

extern template class ChunkCache<uint8_t>;
extern template class ChunkCache<uint16_t>;
