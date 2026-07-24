#pragma once

#include "vc/core/render/ChunkFetch.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
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

    virtual ChunkReadyCallbackId addChunkReadyListener(ChunkReadyCallback cb) = 0;
    virtual void removeChunkReadyListener(ChunkReadyCallbackId id) = 0;
};

} // namespace vc::render
