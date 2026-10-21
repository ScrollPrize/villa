#pragma once
#include "vc/core/render/IChunkedArray.hpp"
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace vc::render {
// Immutable ownership view, not a cache: pins keep payloads alive until all
// synchronous users finish, even if the regular cache evicts its own entries.
struct PinnedRegionMiss : std::runtime_error {
    explicit PinnedRegionMiss(const std::string& detail):std::runtime_error("sampling exceeded pinned chunk region: "+detail){}
};
class PinnedChunkRegion {
    friend class PinnedChunkLookup;
    struct Array {std::shared_ptr<IChunkedArray> source;std::map<std::array<int,4>,ChunkResult> chunks;};
    std::unordered_map<const IChunkedArray*,Array> arrays_;
    std::unordered_set<const void*> readyOwners_;
    bool sealed_=false;
public:
    void pin(std::shared_ptr<IChunkedArray>,const std::vector<ChunkKey>&);
    const ChunkResult& get(const IChunkedArray*,int,int,int,int) const;
    void markReady(const void* owner){if(sealed_)throw std::logic_error("sealed pinned region");readyOwners_.insert(owner);}
    // Call once after all channels are pinned, before sharing with seed jobs.
    void seal(){sealed_=true;}
    bool ready(const void* owner) const {return readyOwners_.contains(owner);}
    size_t bytes() const;
};
// Per-operation, per-array lookup. Region must be sealed and outlive this
// object and all borrowed results. No TLS, I/O, locks or shared_ptr increments.
// The tiny pointer window preserves locality across chunk boundaries.
class PinnedChunkLookup {
    const PinnedChunkRegion* region_;
    const IChunkedArray* source_;
    const PinnedChunkRegion::Array* array_;
    ChunkKey lastKey_{};
    const ChunkResult* last_ = nullptr;
    std::array<ChunkKey, 8> keys_{};
    std::array<const ChunkResult*, 8> results_{};
    size_t size_ = 0;
    size_t next_ = 0;
    const ChunkResult& resolve(const ChunkKey&);
public:
    PinnedChunkLookup(const PinnedChunkRegion&, const IChunkedArray*);
    const ChunkResult& get(const ChunkKey& key) {
        if (last_ && lastKey_ == key)
            return *last_;
        return resolve(key);
    }
};
}
