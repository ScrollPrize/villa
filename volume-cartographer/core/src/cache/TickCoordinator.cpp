#include "vc/core/cache/TickCoordinator.hpp"

#include <algorithm>
#include <chrono>

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

void TickCoordinator::notifyChunkLanded(const ChunkKey& k) noexcept
{
    TickCoordinator* c = g_coordinator.load(std::memory_order_acquire);
    if (!c) return;
    if (!c->chunkLandedRing_.try_push(k)) {
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

        // Drain producer rings into master state. Events are small POD
        // ChunkKeys. Chunk-landed events only contribute to counters for
        // now; empty-chunk events extend the sorted master vector.
        std::uint64_t chunksThisTick = 0;
        std::uint64_t emptiesThisTick = 0;
        ChunkKey k;
        while (chunkLandedRing_.try_pop(k)) {
            ++chunksThisTick;
        }
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

        const std::uint64_t newGen = gen_.fetch_add(1, std::memory_order_relaxed) + 1;
        next->generation          = newGen;
        next->chunksLandedThisTick = chunksThisTick;
        next->emptyChunksThisTick  = emptiesThisTick;
        next->totalChunksLanded    = totalChunksLanded_;
        next->totalEmptyChunks     = totalEmptyChunks_;
        // Republish the empties vector if the master changed this tick.
        // Vector assignment reuses capacity when possible; a full copy
        // of a few-thousand 16-byte entries is in the tens of µs.
        if (emptiesThisTick > 0 || next->emptyChunkKeys.size() != emptyChunkMaster_.size()) {
            next->emptyChunkKeys = emptyChunkMaster_;
        }
        current_.store(next, std::memory_order_release);
    }
}

}  // namespace vc::cache
