#pragma once

#include "vc/core/render/IChunkedArray.hpp"
#include "vc/core/render/PinnedChunkRegion.hpp"
#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace vc::render {

// Short-lived shared payload references; storage and eviction remain owned by
// ChunkCacheService. Copy returned ChunkResult before another get may evict it.
struct LocalChunkCache {
    // ChunkResult owns a shared_ptr to decoded chunk bytes. Keeping every
    // result touched by a frame pins the whole working set even after the
    // shared ChunkCache evicts it. A small window preserves the hot consecutive
    // lookup fast path without allowing each render worker to become another
    // unbounded decoded-byte cache.
    static constexpr std::size_t kMaxPinnedChunks = 8;

    explicit LocalChunkCache(
        IChunkedArray& a, std::size_t expectedChunks = 0, bool queueMisses_ = true,
        ChunkRequestContext request_ = {}, bool blocking_ = false)
        : array(a)
        , queueMisses(queueMisses_)
        , blocking(blocking_)
        , request(request_)
    {
        if (expectedChunks > 0) {
            chunks.reserve(std::min(expectedChunks, kMaxPinnedChunks));
            requestedKeys.reserve(expectedChunks);
            errorKeys.reserve(expectedChunks);
        }
    }

    const ChunkResult& get(const ChunkKey& key, int& requested, int& errors)
    {
        if(const auto* pinned=borrowedPinnedChunk(&array,key.level,key.iz,key.iy,key.ix)) return *pinned;
        // Trilinear sampling reads 8 voxels per pixel and adjacent pixels share
        // chunks, so consecutive lookups overwhelmingly hit the same key. Skip
        // the hash-map probe in that case.
        if (lastResult && lastKey == key)
            return *lastResult;

        auto it = chunks.find(key);
        if (it == chunks.end()) {
            if (chunks.size() >= kMaxPinnedChunks) {
                lastResult = nullptr;
                chunks.clear();
            }
            ChunkResult result = blocking
                ? array.getChunkBlocking(key.level, key.iz, key.iy, key.ix)
                : queueMisses
                ? array.tryGetChunk(key.level, key.iz, key.iy, key.ix, request)
                : array.getChunkIfCached(key.level, key.iz, key.iy, key.ix);
            if (queueMisses && result.status == ChunkStatus::MissQueued &&
                requestedKeys.insert(key).second)
                ++requested;
            if (result.status == ChunkStatus::Error && errorKeys.insert(key).second)
                ++errors;
            it = chunks.emplace(key, std::move(result)).first;
        }

        lastKey = key;
        lastResult = &it->second;
        return it->second;
    }

    IChunkedArray& array;
    bool queueMisses = true;
    bool blocking = false;
    ChunkRequestContext request;
    std::unordered_map<ChunkKey, ChunkResult, ChunkKeyHash> chunks;
    std::unordered_set<ChunkKey, ChunkKeyHash> requestedKeys;
    std::unordered_set<ChunkKey, ChunkKeyHash> errorKeys;
    ChunkKey lastKey{};
    const ChunkResult* lastResult = nullptr;
};

} // namespace vc::render
