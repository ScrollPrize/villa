#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

#include "vc/core/util/MpscRing.hpp"
#include "ChunkKey.hpp"

namespace vc::cache {

// Per-frame state published atomically by the TickCoordinator.
// Readers load a raw const pointer via `currentFrame()` at the start of
// a render and hold it for the duration. All mutation happens on the
// coordinator's thread, between ticks, on the non-current buffer.
struct FrameState {
    std::uint64_t generation = 0;

    // Sorted, deduplicated list of chunks known to be all-zero. Published
    // from EmptyChunkNoted events. Readers binary-search it as a plain-
    // memory alternative to BlockPipeline::isEmptyChunk's atomic probe
    // loop. Only grows; a chunk becomes "empty" once and stays that way.
    std::vector<ChunkKey> emptyChunkKeys;

    // Drain counts from the tick that produced this frame.
    std::uint64_t chunksLandedThisTick = 0;
    std::uint64_t emptyChunksThisTick = 0;

    // Cumulative counts since process start. Useful for dashboards.
    std::uint64_t totalChunksLanded = 0;
    std::uint64_t totalEmptyChunks = 0;
};

// Single-writer, multi-reader state publisher. One dedicated std::jthread
// wakes every 16 ms, prepares the non-current FrameState buffer, then
// atomically swaps the `current_` pointer. Readers signal completion via
// `releaseFrame()` so the coordinator knows when the non-current buffer
// is safe to overwrite.
class TickCoordinator {
public:
    TickCoordinator();
    ~TickCoordinator();

    TickCoordinator(const TickCoordinator&) = delete;
    TickCoordinator& operator=(const TickCoordinator&) = delete;

    // Render entry: load the current FrameState. Valid until releaseFrame()
    // is called with the same pointer. Callers must release before their
    // next currentFrame() call.
    [[nodiscard]] const FrameState* currentFrame() const noexcept
    {
        return current_.load(std::memory_order_acquire);
    }

    // Signal that the caller is done reading `s`. `last_released_gen_` is
    // monotonic across all readers, so the coordinator recycles a buffer
    // once `last_released_gen_ >= buffer->generation` — i.e., some reader
    // has released a generation at least as new as the one the buffer
    // currently holds. Multi-reader safe because readers never un-release.
    void releaseFrame(const FrameState* s) noexcept
    {
        if (!s) return;
        // Monotonic: never step backwards if multiple viewers overlap.
        std::uint64_t prev = last_released_gen_.load(std::memory_order_relaxed);
        while (s->generation > prev) {
            if (last_released_gen_.compare_exchange_weak(
                    prev, s->generation, std::memory_order_release,
                    std::memory_order_relaxed)) {
                break;
            }
        }
    }

    [[nodiscard]] std::uint64_t generation() const noexcept
    {
        return gen_.load(std::memory_order_relaxed);
    }

    // Producer-side push. Non-blocking; drops silently on ring overflow
    // (tracked via `dropped*` counters). These are process-wide routes
    // via the global coordinator pointer set up in the constructor.
    static void notifyChunkLanded(const ChunkKey& k) noexcept;
    static void notifyEmptyChunkNoted(const ChunkKey& k) noexcept;

    // Convenience accessors for readers that don't have a direct handle
    // to the coordinator (e.g. BlockSampler, constructed deep inside
    // render code). Returns null if no coordinator is running.
    // `releaseFrameGlobal` is a no-op for null inputs so destructors
    // can call it unconditionally.
    static const FrameState* currentFrameGlobal() noexcept;
    static void releaseFrameGlobal(const FrameState* s) noexcept;

    [[nodiscard]] std::uint64_t droppedChunkLanded() const noexcept
    {
        return droppedChunkLanded_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] std::uint64_t droppedEmptyChunks() const noexcept
    {
        return droppedEmptyChunks_.load(std::memory_order_relaxed);
    }

private:
    void runLoop(std::stop_token stop) noexcept;

    std::array<FrameState, 2> frames_{};
    std::atomic<const FrameState*> current_{nullptr};
    std::atomic<std::uint64_t> last_released_gen_{0};
    std::atomic<std::uint64_t> gen_{0};

    // Producer events. Single ring per type keeps the implementation
    // simple; if CAS contention becomes visible in a profile we can
    // shard by BlockCache shard index.
    vc::util::MpscRing<ChunkKey, 16384> chunkLandedRing_;
    vc::util::MpscRing<ChunkKey, 4096>  emptyChunkRing_;

    // Full-ring drops. Should be zero in practice; non-zero values mean
    // the drain thread couldn't keep up or the rings are undersized.
    std::atomic<std::uint64_t> droppedChunkLanded_{0};
    std::atomic<std::uint64_t> droppedEmptyChunks_{0};

    // Cumulative counts, updated by the drain thread only.
    std::uint64_t totalChunksLanded_ = 0;
    std::uint64_t totalEmptyChunks_ = 0;

    // Master sorted list of empty chunks. Published (copied) into each
    // FrameState buffer at tick boundary. Grows over the session; shrinks
    // only on BlockPipeline::clearMemory, which we don't signal yet (so
    // the list is monotonic in practice).
    std::vector<ChunkKey> emptyChunkMaster_;

    // jthread declared last so its destructor runs first on teardown,
    // stopping the loop before member storage unwinds.
    std::jthread worker_;
};

}  // namespace vc::cache
