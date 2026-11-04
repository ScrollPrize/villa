// Byte-bounded LRU of decoded chunks, shared between arrays.
#pragma once

#include <cstddef>
#include <cstdint>
#include "lru.hpp"
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vcr {

using Chunk = std::vector<std::uint8_t>;
using ChunkPtr = std::shared_ptr<const Chunk>;  // nullptr: the chunk is not stored (reads as fill)

class MemoryCache {
public:
    explicit MemoryCache(std::size_t max_bytes) : entries_(max_bytes) {}

    // Found: (true, chunk or nullptr for "missing"). Not cached: (false, nullptr).
    std::pair<bool, ChunkPtr> get(const std::string& key)
    {
        std::lock_guard lock(mutex_);
        const auto* value = entries_.get(key);
        if (!value) { ++misses_; return {false, nullptr}; }
        ++hits_;
        return {true, *value};
    }

    void put(const std::string& key, ChunkPtr value)
    {
        const std::size_t cost = value ? value->size() : kMissingCost;
        std::lock_guard lock(mutex_);
        entries_.put(key, std::move(value), cost);
    }

    void clear()
    {
        std::lock_guard lock(mutex_);
        entries_.clear();
    }

    std::size_t max_bytes() const noexcept { return entries_.limit(); }
    std::size_t nbytes() { std::lock_guard lock(mutex_); return entries_.bytes(); }
    std::size_t size() { std::lock_guard lock(mutex_); return entries_.size(); }
    std::uint64_t hits() { std::lock_guard lock(mutex_); return hits_; }
    std::uint64_t misses() { std::lock_guard lock(mutex_); return misses_; }

private:
    static constexpr std::size_t kMissingCost = 64;
    Lru<ChunkPtr> entries_;
    std::uint64_t hits_ = 0, misses_ = 0;
    std::mutex mutex_;
};

}  // namespace vcr
