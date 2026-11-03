#pragma once

#include "vc/core/render/ChunkFetch.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace vc::render {

class ChunkChannelHints;
class IChunkedArray;
namespace detail { struct ChunkReadBatch; }

enum class ChunkStatus {
    MissQueued,
    Missing,
    AllFill,
    Data,
    Error
};

enum class ChunkDtype {
    UInt8,
    UInt16,
    // Variable-length application records, never dense tensor samples.
    Opaque,
    Float32 // Signed scalar channels; sampled through typed channel readers.
};

struct ChunkResult {
    ChunkStatus status = ChunkStatus::MissQueued;
    ChunkDtype dtype = ChunkDtype::UInt8;
    std::array<int, 3> shape{};
    std::shared_ptr<const std::vector<std::byte>> bytes;
    std::string error;
    std::shared_ptr<const DecodedChunkPayload> payload;
    // Zero-copy scalar view into a shared packed-channel decoded payload.
    std::size_t byteOffset = 0;
};

struct ChunkRequestContext {
    std::uint64_t viewId = 0;
    std::uint64_t viewVersion = 0;

    // View-independent interactive computation (e.g. tracing). Survives render
    // replacement and shares admission fairly with GUI work.
    bool foreground = false;
    // Optional run/read hint. Companion channels are prefetched/decoded into
    // their REGULAR caches. Subsequent ordinary reads need no special view.
    std::shared_ptr<const ChunkChannelHints> channelHints;

    [[nodiscard]] bool interactive() const noexcept { return viewId != 0; }
};

struct ChunkViewportSample {
    ChunkKey key;
    std::array<float, 2> viewportPosition{};
    // Offset from this view's requested level. Larger values are coarser
    // fallback levels and therefore have higher interactive priority.
    int relativeLevel = 0;
};

// Channels share a logical voxel grid, but storage chunk sizes may differ.
// Hints populate ordinary caches; they do not change values or cache identity.
class ChunkChannelHints final {
public:
    explicit ChunkChannelHints(std::vector<std::shared_ptr<IChunkedArray>> channels);
    void prefetch(const IChunkedArray& origin,const std::vector<ChunkKey>& keys,
                  int priority,const ChunkRequestContext& request) const;
private:
    std::vector<std::shared_ptr<IChunkedArray>> channels_;
};

class IChunkedArray {
public:
    using ChunkReadyCallbackId = std::uint64_t;

    struct LevelTransform {
        std::array<double, 3> scaleFromLevel0{1.0, 1.0, 1.0};
        std::array<double, 3> offsetFromLevel0{0.0, 0.0, 0.0};
    };

    using ChunkReadyCallback = std::function<void()>;

    virtual ~IChunkedArray() = default;
    virtual int numLevels() const = 0;
    virtual std::array<int, 3> shape(int level) const = 0;
    virtual std::array<int, 3> chunkShape(int level) const = 0;
    virtual ChunkDtype dtype() const = 0;
    virtual double fillValue() const = 0;
    virtual LevelTransform levelTransform(int level) const = 0;
    // Equal identities share one physical decoded payload (e.g. packed CZYX).
    virtual const void* physicalCacheIdentity() const noexcept {return this;}
protected:
    // Implementation hooks for ROI loading. Worker callers use Volume::readZYX;
    // chunk-list scheduling is deliberately not an additional public API.
    friend struct detail::ChunkReadBatch;
    virtual bool prefersBatchReads() const noexcept {return false;}
    virtual std::vector<ChunkResult> getChunksBlocking(const std::vector<ChunkKey>& keys,
                                                      const ChunkRequestContext& request = {})
    {
        std::vector<ChunkResult> results;results.reserve(keys.size());
        for(const auto& key:keys)results.push_back(getChunkBlocking(key.level,key.iz,key.iy,key.ix,request));
        return results;
    }

public:
    // Interactive viewers must use tryGetChunk() only. A miss queues I/O and
    // returns immediately; chunk-ready listeners are responsible for scheduling
    // a later repaint on the UI thread.
    virtual ChunkResult tryGetChunk(int level, int iz, int iy, int ix) = 0;
    virtual ChunkResult tryGetChunk(int level, int iz, int iy, int ix,
                                    const ChunkRequestContext& request)
    {
        if(request.channelHints)request.channelHints->prefetch(*this,{{level,iz,iy,ix}},0,request);
        return tryGetChunk(level, iz, iy, ix);
    }

    // Return a resolved chunk only when it is already in memory. This must not
    // queue a miss or promote a resident entry in the decoded-cache eviction
    // order. Implementations without a decoded cache may use the default.
    virtual ChunkResult getChunkIfCached(int level, int iz, int iy, int ix)
    {
        (void)iz;
        (void)iy;
        (void)ix;
        ChunkResult result;
        result.status = ChunkStatus::MissQueued;
        result.dtype = dtype();
        if (level >= 0 && level < numLevels())
            result.shape = chunkShape(level);
        return result;
    }

    // Blocking access is for CLI, batch, optimization, and prefetch callers.
    // Viewer rendering paths must not call this on the Qt/main thread.
    virtual ChunkResult getChunkBlocking(int level, int iz, int iy, int ix) = 0;
    virtual ChunkResult getChunkBlocking(int level, int iz, int iy, int ix,
                                         const ChunkRequestContext& request)
    {
        if(request.channelHints)request.channelHints->prefetch(*this,{{level,iz,iy,ix}},-1024,request);
        return getChunkBlocking(level, iz, iy, ix);
    }
    virtual void prefetchChunks(const std::vector<ChunkKey>& keys, bool wait, int priorityOffset = 0) = 0;
    virtual void prefetchChunks(const std::vector<ChunkKey>& keys,
                                bool wait,
                                int priorityOffset,
                                const ChunkRequestContext& request)
    {
        if(request.channelHints)request.channelHints->prefetch(*this,keys,priorityOffset,request);
        prefetchChunks(keys, wait, priorityOffset);
    }

    // Atomically replaces one view's located demand for this source. Building
    // and deduplicating `samples` happens before implementations take their
    // shared scheduler lock.
    virtual void replaceViewDemand(const ChunkRequestContext& request,
                                   const std::array<float, 2>& focus,
                                   std::vector<ChunkViewportSample> samples)
    {
        (void)request;
        (void)focus;
        (void)samples;
    }
    // Mark the most recently interacted view. This operation must remain O(1):
    // focus distances are captured and published by accepted render jobs.
    virtual void markViewActive(std::uint64_t viewId)
    {
        (void)viewId;
    }
    // Remove this source's demand for one view without closing that view in
    // other sources sharing the application-wide cache service.
    virtual void clearSourceViewDemand(std::uint64_t viewId,
                                       std::uint64_t viewVersion = 0)
    {
        (void)viewId;
        (void)viewVersion;
    }
    virtual void clearViewDemand(std::uint64_t viewId,
                                 std::uint64_t viewVersion = 0)
    {
        (void)viewId;
        (void)viewVersion;
    }
    virtual ChunkReadyCallbackId addChunkReadyListener(ChunkReadyCallback cb) = 0;
    virtual void removeChunkReadyListener(ChunkReadyCallbackId id) = 0;
};

} // namespace vc::render
