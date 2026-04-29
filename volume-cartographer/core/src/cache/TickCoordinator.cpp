#include "vc/core/cache/TickCoordinator.hpp"

#include <algorithm>
#include <chrono>

#include "vc/core/cache/BlockPipeline.hpp"

namespace vc::cache {

namespace {

constexpr auto kTickInterval = std::chrono::milliseconds(16);

// Process-wide pointer used by the static notify* methods. Set on
// construction, cleared on destruction. Multiple coordinators in one
// process would be a bug; assert the first-writer-wins property via
// a simple atomic CAS.
std::atomic<TickCoordinator*> g_coordinator{nullptr};

}  // namespace

TickCoordinator::TickCoordinator()
{
    // Both buffers start at generation 0. The first publish advances to 1,
    // so a reader holding the initial buffer is trivially "behind" once
    // publishing begins.
    current_.store(&frames_[0], std::memory_order_release);

    TickCoordinator* expected = nullptr;
    g_coordinator.compare_exchange_strong(expected, this,
                                          std::memory_order_release,
                                          std::memory_order_relaxed);

    worker_ = std::jthread([this](std::stop_token stop) { runLoop(stop); });
}

TickCoordinator::~TickCoordinator()
{
    TickCoordinator* self = this;
    g_coordinator.compare_exchange_strong(self, nullptr,
                                          std::memory_order_release,
                                          std::memory_order_relaxed);
}

void TickCoordinator::notifyChunkLanded(BlockPipeline* pipeline,
                                        const ChunkKey& k) noexcept
{
    TickCoordinator* c = g_coordinator.load(std::memory_order_acquire);
    if (!c) return;
    ChunkLandedEvent e{k, pipeline, 0};
    if (!c->chunkLandedRing_.try_push(e)) {
        c->droppedChunkLanded_.fetch_add(1, std::memory_order_relaxed);
    }
}

void TickCoordinator::notifyEmptyChunkNoted(const ChunkKey& k) noexcept
{
    TickCoordinator* c = g_coordinator.load(std::memory_order_acquire);
    if (!c) return;
    if (!c->emptyChunkRing_.try_push(k)) {
        c->droppedEmptyChunks_.fetch_add(1, std::memory_order_relaxed);
    }
}

const FrameState* TickCoordinator::currentFrameGlobal() noexcept
{
    TickCoordinator* c = g_coordinator.load(std::memory_order_acquire);
    return c ? c->currentFrame() : nullptr;
}

void TickCoordinator::releaseFrameGlobal(const FrameState* s) noexcept
{
    if (!s) return;
    TickCoordinator* c = g_coordinator.load(std::memory_order_acquire);
    if (c) c->releaseFrame(s);
}

int TickCoordinator::acquireViewportSlotGlobal() noexcept
{
    TickCoordinator* c = g_coordinator.load(std::memory_order_acquire);
    if (!c) return -1;
    std::uint32_t mask = c->viewportSlotAllocMask_.load(std::memory_order_relaxed);
    for (;;) {
        // Find the lowest clear bit up to kMaxViewers.
        std::uint32_t free = ~mask & ((1u << kMaxViewers) - 1u);
        if (free == 0) return -1;
        const int idx = __builtin_ctz(free);
        const std::uint32_t bit = 1u << idx;
        if (c->viewportSlotAllocMask_.compare_exchange_weak(
                mask, mask | bit,
                std::memory_order_acq_rel, std::memory_order_relaxed)) {
            return idx;
        }
    }
}

void TickCoordinator::releaseViewportSlotGlobal(int slotIdx) noexcept
{
    TickCoordinator* c = g_coordinator.load(std::memory_order_acquire);
    if (!c || slotIdx < 0 || std::size_t(slotIdx) >= kMaxViewers) return;
    // Mark inactive via a zero-filled publish so tick-thread reads a clean
    // "no viewport" after release.
    ViewportSnapshot empty{};
    publishViewportGlobal(slotIdx, empty);
    c->viewportSlotAllocMask_.fetch_and(~(1u << slotIdx), std::memory_order_release);
}

void TickCoordinator::publishViewportGlobal(int slotIdx,
                                            const ViewportSnapshot& s) noexcept
{
    TickCoordinator* c = g_coordinator.load(std::memory_order_acquire);
    if (!c || slotIdx < 0 || std::size_t(slotIdx) >= kMaxViewers) return;
    auto& slot = c->viewportSlots_[slotIdx];
    // Seqlock write: bump to odd, copy payload, bump to even+2.
    const std::uint64_t prev = slot.seq.load(std::memory_order_relaxed);
    slot.seq.store(prev + 1, std::memory_order_release);
    slot.snapshot = s;
    slot.seq.store(prev + 2, std::memory_order_release);
}

void TickCoordinator::enqueuePrefetchGlobal(BlockPipeline* pipeline,
                                            const std::vector<ChunkKey>& keys,
                                            int targetLevel) noexcept
{
    if (!pipeline || keys.empty()) return;
    TickCoordinator* c = g_coordinator.load(std::memory_order_acquire);
    if (!c) {
        // No coordinator running (e.g. CLI tools) — fall back to direct.
        pipeline->fetchInteractive(keys, targetLevel);
        return;
    }
    PendingPrefetch p{pipeline, targetLevel, keys};
    std::lock_guard lk(c->prefetchMutex_);
    c->prefetchQueue_.push_back(std::move(p));
}

void TickCoordinator::runLoop(std::stop_token stop) noexcept
{
    auto next_tick = std::chrono::steady_clock::now() + kTickInterval;
    while (!stop.stop_requested()) {
        std::this_thread::sleep_until(next_tick);
        next_tick += kTickInterval;
        if (stop.stop_requested()) break;

        const FrameState* now = current_.load(std::memory_order_acquire);
        FrameState* next = (now == &frames_[0]) ? &frames_[1] : &frames_[0];

        // Clobber guard. `next` currently carries its previous-publish
        // generation; any reader that loaded `next` when it was last
        // current may still be reading it. It is safe to overwrite once
        // every such reader has released. Since releaseFrame is monotonic,
        // `last_released_gen >= next->generation` implies no reader still
        // holds `next`.
        const std::uint64_t nextOldGen = next->generation;
        if (nextOldGen > 0
            && last_released_gen_.load(std::memory_order_acquire) < nextOldGen) {
            continue;
        }

        // Gather the set of pyramid levels that any active viewport is
        // using right now. ChunkLanded events at levels outside this set
        // are ignored for slice population — keeps sliceMaster_ focused
        // on data the renderers are about to read.
        std::uint32_t activeLevelMask = 0;
        {
            const std::uint32_t allocPreview =
                viewportSlotAllocMask_.load(std::memory_order_acquire);
            for (std::size_t i = 0; i < kMaxViewers; ++i) {
                if (!(allocPreview & (1u << i))) continue;
                auto& slot = viewportSlots_[i];
                const std::uint64_t s1 = slot.seq.load(std::memory_order_acquire);
                if (s1 & 1u) continue;
                const ViewportSnapshot copy = slot.snapshot;
                const std::uint64_t s2 = slot.seq.load(std::memory_order_acquire);
                if (s1 != s2) continue;
                if (!copy.active) continue;
                if (copy.level >= 0 && copy.level < 32) {
                    activeLevelMask |= (1u << copy.level);
                }
            }
        }

        // Drain producer rings into master state.
        std::uint64_t chunksThisTick = 0;
        std::uint64_t emptiesThisTick = 0;
        ChunkLandedEvent ce;
        while (chunkLandedRing_.try_pop(ce)) {
            ++chunksThisTick;
            if (!ce.pipeline || ce.key.level < 0 || ce.key.level >= 32) continue;
            if (!(activeLevelMask & (1u << ce.key.level))) continue;
            // Resolve 512 blocks in the newly-landed canonical chunk.
            // Canonical chunks are 128^3 = 8 blocks per axis. blockAt
            // returns nullptr for blocks that weren't stored (e.g. the
            // zero-chunk shortcut); skip those.
            const int baseBz = ce.key.iz * 8;
            const int baseBy = ce.key.iy * 8;
            const int baseBx = ce.key.ix * 8;
            for (int dz = 0; dz < 8; ++dz) {
                for (int dy = 0; dy < 8; ++dy) {
                    for (int dx = 0; dx < 8; ++dx) {
                        const int bz = baseBz + dz;
                        const int by = baseBy + dy;
                        const int bx = baseBx + dx;
                        const BlockKey bk{ce.key.level, bz, by, bx};
                        const BlockPtr b = ce.pipeline->blockAt(bk);
                        if (!b) continue;
                        const std::uint64_t packed =
                            (std::uint64_t(std::uint32_t(bz)) << 42) |
                            (std::uint64_t(std::uint32_t(by)) << 21) |
                             std::uint64_t(std::uint32_t(bx));
                        sliceMaster_.push_back(SliceEntry{
                            packed, ce.key.level, ce.pipeline, b});
                    }
                }
            }
            // FIFO eviction: keep the most recent kSliceMax entries.
            if (sliceMaster_.size() > kSliceMax) {
                const std::size_t drop = sliceMaster_.size() - kSliceMax;
                sliceMaster_.erase(sliceMaster_.begin(),
                                   sliceMaster_.begin()
                                       + static_cast<std::ptrdiff_t>(drop));
            }
        }
        ChunkKey k;
        while (emptyChunkRing_.try_pop(k)) {
            ++emptiesThisTick;
            // Sorted insert; skip duplicates. Amortized O(log N) compare +
            // O(N) shift for the rare true-new entry. N is small in practice
            // (hundreds to thousands of empty chunks per volume).
            auto it = std::lower_bound(emptyChunkMaster_.begin(),
                                       emptyChunkMaster_.end(), k);
            if (it == emptyChunkMaster_.end() || *it != k) {
                emptyChunkMaster_.insert(it, k);
            }
        }
        totalChunksLanded_ += chunksThisTick;
        totalEmptyChunks_  += emptiesThisTick;

        // Gather all viewport snapshots via seqlock reads. This publishes
        // a point-in-time union that any consumer (slice scoping, prefetch
        // coalescing) can use without further atomics.
        std::array<ViewportSnapshot, kMaxViewers> viewports{};
        const std::uint32_t allocMask =
            viewportSlotAllocMask_.load(std::memory_order_acquire);
        for (std::size_t i = 0; i < kMaxViewers; ++i) {
            if (!(allocMask & (1u << i))) continue;
            auto& slot = viewportSlots_[i];
            // Seqlock read loop. Rare contention; bounded to <1ms worst case.
            for (int attempt = 0; attempt < 64; ++attempt) {
                const std::uint64_t s1 = slot.seq.load(std::memory_order_acquire);
                if (s1 & 1u) continue;  // writer in progress
                ViewportSnapshot copy = slot.snapshot;
                const std::uint64_t s2 = slot.seq.load(std::memory_order_acquire);
                if (s1 == s2) {
                    viewports[i] = copy;
                    break;
                }
            }
        }

        // Drain pending prefetch requests and dispatch one fetchInteractive
        // per (pipeline, targetLevel) group. Consolidates up to N viewers
        // of duplicate work into a single priority-queue rebuild per tick.
        std::vector<PendingPrefetch> drained;
        {
            std::lock_guard lk(prefetchMutex_);
            drained.swap(prefetchQueue_);
        }
        std::uint64_t prefetchCallsThisTick = 0;
        if (!drained.empty()) {
            // Sort so same-(pipeline,level) pairs are contiguous; then
            // merge keys per group, dedup, dispatch. Avoids a full
            // unordered_map when we typically have 1-2 groups.
            std::sort(drained.begin(), drained.end(),
                      [](const PendingPrefetch& a, const PendingPrefetch& b) {
                          if (a.pipeline != b.pipeline) return a.pipeline < b.pipeline;
                          return a.targetLevel < b.targetLevel;
                      });
            for (auto it = drained.begin(); it != drained.end();) {
                auto groupEnd = it + 1;
                while (groupEnd != drained.end()
                       && groupEnd->pipeline == it->pipeline
                       && groupEnd->targetLevel == it->targetLevel) {
                    ++groupEnd;
                }
                std::vector<ChunkKey> merged;
                std::size_t total = 0;
                for (auto p = it; p != groupEnd; ++p) total += p->keys.size();
                merged.reserve(total);
                for (auto p = it; p != groupEnd; ++p) {
                    merged.insert(merged.end(), p->keys.begin(), p->keys.end());
                }
                std::sort(merged.begin(), merged.end());
                merged.erase(std::unique(merged.begin(), merged.end()),
                             merged.end());
                if (!merged.empty()) {
                    it->pipeline->fetchInteractive(merged, it->targetLevel);
                    ++prefetchCallsThisTick;
                }
                it = groupEnd;
            }
        }
        totalPrefetchCalls_ += prefetchCallsThisTick;

        const std::uint64_t newGen = gen_.fetch_add(1, std::memory_order_relaxed) + 1;
        next->generation          = newGen;
        next->chunksLandedThisTick = chunksThisTick;
        next->emptyChunksThisTick  = emptiesThisTick;
        next->prefetchCallsThisTick = prefetchCallsThisTick;
        next->totalChunksLanded    = totalChunksLanded_;
        next->totalEmptyChunks     = totalEmptyChunks_;
        next->totalPrefetchCalls   = totalPrefetchCalls_;
        next->viewports            = viewports;
        // Republish the empties vector if the master changed this tick.
        // Vector assignment reuses capacity when possible; a full copy
        // of a few-thousand 16-byte entries is in the tens of µs.
        if (emptiesThisTick > 0 || next->emptyChunkKeys.size() != emptyChunkMaster_.size()) {
            next->emptyChunkKeys = emptyChunkMaster_;
        }
        // Rebuild published slice from master. Copy, then sort by packed
        // key so readers can binary_search. Dedup isn't strictly required;
        // readers verify pipeline and level because packed block coordinates
        // overlap across volumes and pyramid levels.
        if (chunksThisTick > 0 || next->slice.size() != sliceMaster_.size()) {
            next->slice = sliceMaster_;
            std::sort(next->slice.begin(), next->slice.end(),
                      [](const SliceEntry& a, const SliceEntry& b) {
                          if (a.packedKey != b.packedKey)
                              return a.packedKey < b.packedKey;
                          if (a.level != b.level)
                              return a.level < b.level;
                          return a.pipeline < b.pipeline;
                      });
        }
        current_.store(next, std::memory_order_release);
    }
}

}  // namespace vc::cache
