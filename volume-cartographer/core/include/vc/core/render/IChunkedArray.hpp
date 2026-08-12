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

enum class ChunkStatus {
    MissQueued,
    Missing,
    AllFill,
    Data,
    Error
};

enum class ChunkDtype {
    UInt8,
    UInt16
};

struct ChunkResult {
    ChunkStatus status = ChunkStatus::MissQueued;
    ChunkDtype dtype = ChunkDtype::UInt8;
    std::array<int, 3> shape{};
    std::shared_ptr<const std::vector<std::byte>> bytes;
    std::string error;
};

struct ChunkRequestContext {
    std::uint64_t viewId = 0;
    std::uint64_t viewVersion = 0;

    [[nodiscard]] bool interactive() const noexcept { return viewId != 0; }
};

struct ChunkViewportSample {
    ChunkKey key;
    std::array<float, 2> viewportPosition{};
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

    // Interactive viewers must use tryGetChunk() only. A miss queues I/O and
    // returns immediately; chunk-ready listeners are responsible for scheduling
    // a later repaint on the UI thread.
    virtual ChunkResult tryGetChunk(int level, int iz, int iy, int ix) = 0;
    virtual ChunkResult tryGetChunk(int level, int iz, int iy, int ix,
                                    const ChunkRequestContext& request)
    {
        (void)request;
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
    virtual void prefetchChunks(const std::vector<ChunkKey>& keys, bool wait, int priorityOffset = 0) = 0;
    virtual void prefetchChunks(const std::vector<ChunkKey>& keys,
                                bool wait,
                                int priorityOffset,
                                const ChunkRequestContext& request)
    {
        (void)request;
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
    virtual void updateViewFocus(std::uint64_t viewId,
                                 const std::array<float, 2>& focus,
                                 bool makeActive)
    {
        (void)viewId;
        (void)focus;
        (void)makeActive;
    }
    virtual void clearViewDemand(std::uint64_t viewId)
    {
        (void)viewId;
    }
    // Starts a newer interactive request. An exclusively-owned backing array
    // may discard unresolved work from the superseded view; resident chunks
    // are preserved.
    virtual void beginViewRequest(bool discardPending = false)
    {
        (void)discardPending;
    }

    virtual ChunkReadyCallbackId addChunkReadyListener(ChunkReadyCallback cb) = 0;
    virtual void removeChunkReadyListener(ChunkReadyCallbackId id) = 0;
};

} // namespace vc::render
