#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

#include "vc/core/util/MpscRing.hpp"
#include "BlockCache.hpp"  // Block, BlockKey, BlockPtr
#include "ChunkKey.hpp"

namespace vc::cache {

class BlockPipeline;  // fwd decl

// Per-viewer snapshot published by main-thread render handlers and
// consumed by the tick loop for slice scoping / prefetch coalescing.
// POD-only so we can seqlock-copy it between threads without a mutex.
struct ViewportSnapshot {
    bool active = false;
    int level = 0;
    BlockPipeline* pipeline = nullptr;
    // World-space (voxel coordinate) bounding box of what the viewer may
    // sample during its next render. Coarse; callers trade precision for
    // cheap compute. Meaningless when `active == false`.
    float minX = 0, minY = 0, minZ = 0;
    float maxX = 0, maxY = 0, maxZ = 0;
};

static constexpr std::size_t kMaxViewers = 16;

// Carried on chunkLandedRing_: identifies which pipeline the producer
// wrote to so the tick thread can resolve individual block pointers
// for slice population.
struct ChunkLandedEvent {
    ChunkKey key;
    BlockPipeline* pipeline;
    std::uint64_t _pad;  // 16 + 8 + 8 = 32 bytes, cache-friendly
};

// One slice entry. `packedKey` is the BlockSampler packKey of (bz,by,bx).
// Readers verify both `pipeline` and `level`; adaptive renders can have
// several pyramid levels resident for one pipeline, and block coordinates
// overlap across levels.
struct SliceEntry {
    std::uint64_t        packedKey;
    int                  level;
    const BlockPipeline* pipeline;
    const Block*         block;
};

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

    // Snapshots of every active viewer, copied from the seqlock slots at
    // the start of each tick. Stable during a render.
    std::array<ViewportSnapshot, kMaxViewers> viewports{};

    // Sorted-by-packedKey slice of recently-landed blocks for which the
    // producing pipeline is used by at least one active viewport (level
    // filter). Readers binary-search to short-circuit the atomic-heavy
    // BlockCache::get path. Stale entries are caught by the pipeline
    // mismatch check in the reader.
    std::vector<SliceEntry> slice;

    // Drain counts from the tick that produced this frame.
    std::uint64_t chunksLandedThisTick = 0;
    std::uint64_t emptyChunksThisTick = 0;
    std::uint64_t prefetchCallsThisTick = 0;

    // Cumulative counts since process start. Useful for dashboards.
    std::uint64_t totalChunksLanded = 0;
    std::uint64_t totalEmptyChunks = 0;
    std::uint64_t totalPrefetchCalls = 0;
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
    static void notifyChunkLanded(BlockPipeline* pipeline, const ChunkKey& k) noexcept;
    static void notifyEmptyChunkNoted(const ChunkKey& k) noexcept;

    // Convenience accessors for readers that don't have a direct handle
    // to the coordinator (e.g. BlockSampler, constructed deep inside
    // render code). Returns null if no coordinator is running.
    // `releaseFrameGlobal` is a no-op for null inputs so destructors
    // can call it unconditionally.
    static const FrameState* currentFrameGlobal() noexcept;
    static void releaseFrameGlobal(const FrameState* s) noexcept;

    // Viewport slot management. `acquireViewportSlot` returns an index in
    // [0, kMaxViewers) on success, -1 if the table is full. Release marks
    // the slot inactive. publishViewport copies into a per-slot seqlock;
    // the tick thread reads all slots each tick.
    static int  acquireViewportSlotGlobal() noexcept;
    static void releaseViewportSlotGlobal(int slotIdx) noexcept;
    static void publishViewportGlobal(int slotIdx,
                                      const ViewportSnapshot& s) noexcept;

    // Coalesced prefetch. Callers enqueue chunks to prefetch; the tick
    // thread groups them by (pipeline, targetLevel), dedups, and issues
    // one fetchInteractive call per group each tick. Falls back to an
    // immediate fetchInteractive if the coordinator is not running.
    static void enqueuePrefetchGlobal(BlockPipeline* pipeline,
                                      const std::vector<ChunkKey>& keys,
                                      int targetLevel) noexcept;

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
    vc::util::MpscRing<ChunkLandedEvent, 16384> chunkLandedRing_;
    vc::util::MpscRing<ChunkKey, 4096>          emptyChunkRing_;

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

    // Master slice ring-ish buffer. New entries appended; when it exceeds
    // kSliceMax, the front is dropped in bulk (erase from begin). Final
    // published vector is sorted + dedup'd into the FrameState.
    static constexpr std::size_t kSliceMax = 16384;
    std::vector<SliceEntry> sliceMaster_;

    // Per-viewer seqlock slots. `seq` is even at rest, odd while writing.
    // Cache-line aligned to avoid false sharing when several viewers
    // publish simultaneously on the main thread.
    struct alignas(64) ViewportSlot {
        std::atomic<std::uint64_t> seq{0};
        ViewportSnapshot snapshot{};
    };
    std::array<ViewportSlot, kMaxViewers> viewportSlots_{};
    std::atomic<std::uint32_t> viewportSlotAllocMask_{0};

    // Pending prefetch requests. Main threads append; tick thread drains
    // and dispatches. Short-held mutex is cheaper than an MPSC ring here
    // because producers post *vectors* of keys at once, not single items.
    struct PendingPrefetch {
        BlockPipeline* pipeline;
        int targetLevel;
        std::vector<ChunkKey> keys;
    };
    std::mutex prefetchMutex_;
    std::vector<PendingPrefetch> prefetchQueue_;
    std::uint64_t totalPrefetchCalls_ = 0;

    // jthread declared last so its destructor runs first on teardown,
    // stopping the loop before member storage unwinds.
    std::jthread worker_;
};

}  // namespace vc::cache
