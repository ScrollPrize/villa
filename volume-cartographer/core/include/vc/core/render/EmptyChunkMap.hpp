#pragma once

#include "vc/core/render/ChunkFetch.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <unordered_map>

namespace vc::render {

// Source-local negative knowledge, protected by the owning cache mutex.
// Each lazily allocated 16^3 page uses two bits per chunk. No linearization
// of the whole volume (or allocation proportional to its bounding box).
class EmptyChunkMap {
public:
    enum class Value : uint8_t { Unknown = 0, Missing = 1, AllFill = 2 };
    Value get(const ChunkKey& key) const {
        const auto it = pages_.find(pageKey(key));
        if (it == pages_.end()) return Value::Unknown;
        const auto index = slot(key);
        return Value((it->second.words[index / 32] >> (2 * (index % 32))) & 3);
    }
    void set(const ChunkKey& key, Value value) {
        const auto page = pageKey(key);
        auto it = pages_.find(page);
        if (it == pages_.end()) {
            if (value == Value::Unknown) return;
            it = pages_.try_emplace(page).first;
        }
        const auto index = slot(key), shift = 2 * (index % 32);
        auto& word = it->second.words[index / 32];
        const auto old = Value((word >> shift) & 3);
        if (old == value) return;
        if (old == Value::Unknown) { ++known_; ++it->second.known; }
        if (value == Value::Unknown) { --known_; --it->second.known; }
        word = (word & ~(uint64_t(3) << shift)) | (uint64_t(value) << shift);
        if (!it->second.known) pages_.erase(it);
    }
    void clear() { pages_.clear(); known_ = 0; }
    size_t size() const { return known_; }
    size_t pageBytes() const { return pages_.size() * sizeof(Page); }
private:
    struct Page { std::array<uint64_t, 128> words{}; size_t known = 0; };
    static ChunkKey pageKey(const ChunkKey& k) {
        return {k.level, k.iz >> 4, k.iy >> 4, k.ix >> 4};
    }
    static size_t slot(const ChunkKey& k) {
        return (size_t(k.iz & 15) * 16 + (k.iy & 15)) * 16 + (k.ix & 15);
    }
    std::unordered_map<ChunkKey, Page, ChunkKeyHash> pages_;
    size_t known_ = 0;
};
} // namespace vc::render
