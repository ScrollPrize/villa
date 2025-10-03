#pragma once

#include <vector>
#include <memory>
#include <mutex>
#include <opencv2/core.hpp>
#include <atomic>
#include <chrono>

namespace vc::core::util {

class GridStore;
class LineSegList;

class LineSegListCache {
public:
    using CacheKey = std::pair<const void*, size_t>; // GridStore pointer and offset

    explicit LineSegListCache(size_t max_size_bytes = 1024 * 1024 * 1024); // 1 GB default

    std::shared_ptr<LineSegList> get(const CacheKey& key);
    void put(const CacheKey& key, std::shared_ptr<LineSegList> data);

private:
    struct CacheKeyHash {
        std::size_t operator()(const CacheKey& k) const {
            return std::hash<const void*>()(k.first) ^ (std::hash<size_t>()(k.second) << 1);
        }
    };

    struct CacheEntry {
        CacheKey key;
        std::weak_ptr<LineSegList> data;
        uint64_t generation;
        size_t size_bytes;
    };

    void evict();
    void check_print_stats();

    size_t max_size_bytes_;
    size_t current_size_bytes_ = 0;
    uint64_t generation_counter_ = 0;

    std::atomic<uint64_t> cache_hits_{0};
    std::atomic<uint64_t> cache_misses_{0};
    std::chrono::steady_clock::time_point last_stat_time_;

    std::vector<CacheEntry> entries_;
    std::unordered_map<CacheKey, size_t, CacheKeyHash> lookup_;
    
    std::mutex mutex_;
};

} // namespace vc::core::util
