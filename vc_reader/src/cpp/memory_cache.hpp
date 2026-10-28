// Byte-bounded LRU of decoded chunks, shared between arrays.
#pragma once

#include <cstddef>
#include <cstdint>
#include <list>
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
    explicit MemoryCache(std::size_t max_bytes) : max_bytes_(max_bytes) {}

    // Found: (true, chunk or nullptr for "missing"). Not cached: (false, nullptr).
    std::pair<bool, ChunkPtr> get(const std::string& key)
    {
        std::lock_guard lock(mutex_);
        auto it = index_.find(key);
        if (it == index_.end()) {
            ++misses_;
            return {false, nullptr};
        }
        order_.splice(order_.begin(), order_, it->second);
        ++hits_;
        return {true, it->second->value};
    }

    void put(const std::string& key, ChunkPtr value)
    {
        const std::size_t cost = value ? value->size() : kMissingCost;
        if (cost > max_bytes_)
            return;
        std::lock_guard lock(mutex_);
        if (auto it = index_.find(key); it != index_.end()) {
            bytes_ -= it->second->cost;
            order_.erase(it->second);
            index_.erase(it);
        }
        order_.push_front(Entry{key, std::move(value), cost});
        index_.emplace(key, order_.begin());
        bytes_ += cost;
        while (bytes_ > max_bytes_ && !order_.empty()) {
            auto& last = order_.back();
            bytes_ -= last.cost;
            index_.erase(last.key);
            order_.pop_back();
        }
    }

    void clear()
    {
        std::lock_guard lock(mutex_);
        order_.clear();
        index_.clear();
        bytes_ = 0;
    }

    std::size_t max_bytes() const noexcept { return max_bytes_; }
    std::size_t nbytes() { std::lock_guard lock(mutex_); return bytes_; }
    std::size_t size() { std::lock_guard lock(mutex_); return index_.size(); }
    std::uint64_t hits() { std::lock_guard lock(mutex_); return hits_; }
    std::uint64_t misses() { std::lock_guard lock(mutex_); return misses_; }

private:
    static constexpr std::size_t kMissingCost = 64;
    struct Entry {
        std::string key;
        ChunkPtr value;
        std::size_t cost;
    };
    std::size_t max_bytes_;
    std::size_t bytes_ = 0;
    std::uint64_t hits_ = 0, misses_ = 0;
    std::list<Entry> order_;
    std::unordered_map<std::string, std::list<Entry>::iterator> index_;
    std::mutex mutex_;
};

}  // namespace vcr
