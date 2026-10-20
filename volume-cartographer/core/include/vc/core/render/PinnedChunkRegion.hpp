#pragma once
#include "vc/core/render/IChunkedArray.hpp"
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace vc::render {
// Immutable ownership view, not a cache: payloads stay owned by the regular
// cache AND these pins until all synchronous users have finished.
struct PinnedRegionMiss : std::runtime_error {
    explicit PinnedRegionMiss(const std::string& detail):std::runtime_error("sampling exceeded pinned chunk region: "+detail){}
};
class PinnedChunkRegion {
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
// Never span a coroutine yield or parallelFor: install on each synchronous job.
class ScopedPinnedChunkRegion {
    const PinnedChunkRegion* previous_;
public:
    explicit ScopedPinnedChunkRegion(const PinnedChunkRegion*);
    ~ScopedPinnedChunkRegion();
    ScopedPinnedChunkRegion(const ScopedPinnedChunkRegion&)=delete;
    ScopedPinnedChunkRegion& operator=(const ScopedPinnedChunkRegion&)=delete;
};
const PinnedChunkRegion* activePinnedChunkRegion();
const ChunkResult* borrowedPinnedChunk(const IChunkedArray*,int,int,int,int);
}
