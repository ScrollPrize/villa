#pragma once

#include "vc/core/util/xtensor_include.hpp"
#include XTENSORINCLUDE(containers, xarray.hpp)
#include <opencv2/core.hpp>
#include <vc/core/util/HashFunctions.hpp>

#include <shared_mutex>
#include <memory>
#include <unordered_map>
#include <string>

/**
 * @brief Thread-safe LRU cache for volume chunks
 *
 * @tparam T Data type of cached chunks (uint8_t or uint16_t)
 *
 * The cache uses a generation-based LRU eviction strategy. When the cache
 * is full, it removes the 10% oldest entries to amortize sorting costs.
 */
template<typename T>
class ChunkCache
{
public:
    /**
     * @brief Construct a new Chunk Cache object
     * @param size Maximum cache size in bytes
     */
    explicit ChunkCache(size_t size);

    ~ChunkCache();

    /**
     * @brief Get or create a group index for a dataset path
     * @param name Unique identifier for the group (e.g., dataset path + group name)
     * @return int Group index (used as high 16 bits of cache key)
     */
    int groupIdx(const std::string& name);

    /**
     * @brief Store a chunk in the cache
     * @param key Cache key (group_idx, z, y, x)
     * @param ar Chunk data (ownership transferred to cache)
     */
    void put(const cv::Vec4i& key, xt::xarray<T>* ar);

    /**
     * @brief Retrieve a chunk from the cache
     * @param key Cache key
     * @return std::shared_ptr<xt::xarray<T>> Cached chunk or nullptr if not found
     */
    std::shared_ptr<xt::xarray<T>> get(const cv::Vec4i& key);

    /**
     * @brief Check if a chunk exists in the cache
     * @param idx Cache key
     * @return true if chunk is cached
     */
    bool has(const cv::Vec4i& idx);

    /**
     * @brief Clear all cached data
     */
    void reset();

    std::shared_mutex mutex;

private:
    uint64_t _generation = 0;
    size_t _size = 0;
    size_t _stored = 0;
    std::unordered_map<cv::Vec4i, std::shared_ptr<xt::xarray<T>>, vec4i_hash> _store;
    std::unordered_map<cv::Vec4i, uint64_t, vec4i_hash> _gen_store;
    std::unordered_map<std::string, int> _group_store;
    std::shared_mutex _mutex;
};

// Explicit template instantiations (defined in ChunkCache.cpp)
extern template class ChunkCache<uint8_t>;
extern template class ChunkCache<uint16_t>;
