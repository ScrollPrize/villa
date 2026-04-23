#include "vc/core/cache/ChunkData.hpp"

#include <vector>

namespace vc::cache {

namespace {

// Per-thread pool of decoded-chunk buffers. Capped so we don't balloon
// memory on threads that process many chunks then go idle. 4 × ~2 MB
// per-thread ≈ 8 MB worst case per producer/consumer thread — fine given
// we already carry a 10 GB arena.
constexpr size_t kPoolMax = 4;

// Owning storage — uses default deleter so freeing at thread exit actually
// releases memory, vs. re-entering our custom deleter recursively.
std::vector<std::unique_ptr<ChunkData>>& pool()
{
    static thread_local std::vector<std::unique_ptr<ChunkData>> p;
    return p;
}

}  // namespace

void ChunkDataPoolDeleter::operator()(ChunkData* p) const noexcept
{
    if (!p) return;
    auto& pl = pool();
    if (pl.size() < kPoolMax) {
        // Reset scalar fields and `bytes` size (keeping capacity) so the
        // next acquire hands out a clean-looking ChunkData while the
        // underlying allocation stays warm in the per-thread heap.
        p->bytes.clear();
        p->shape = {0, 0, 0};
        p->elementSize = 1;
        p->isEmpty = false;
        pl.emplace_back(p);
    } else {
        delete p;
    }
}

ChunkDataPtr acquireChunkData() noexcept
{
    auto& pl = pool();
    if (!pl.empty()) {
        auto owned = std::move(pl.back());
        pl.pop_back();
        return ChunkDataPtr(owned.release());
    }
    return ChunkDataPtr(new ChunkData());
}

}  // namespace vc::cache
