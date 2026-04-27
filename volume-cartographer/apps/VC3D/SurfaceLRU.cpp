#include "SurfaceLRU.hpp"

#include "vc/core/util/QuadSurface.hpp"

#include <QThreadPool>

#include <atomic>
#include <mutex>
#include <unordered_set>

namespace {

// Dedupe concurrent async ensureLoaded() dispatches per surface — touch()
// is called every render tick for the active surface, so an unloaded
// surface would otherwise have a new QThreadPool task enqueued per frame
// until the load finishes. Not a correctness issue (ensureLoaded has its
// own mutex + double-check) but wasteful. The set holds raw pointers
// keyed by QuadSurface*; entries are removed when the async load
// returns, regardless of success or throw.
std::mutex g_asyncLoadMutex;
std::unordered_set<QuadSurface*> g_asyncLoadInFlight;

void asyncWarmSurface(std::shared_ptr<QuadSurface> surf)
{
    if (!surf || surf->isLoaded()) return;
    {
        std::lock_guard<std::mutex> lk(g_asyncLoadMutex);
        if (!g_asyncLoadInFlight.insert(surf.get()).second) return;
    }
    QThreadPool::globalInstance()->start([surf = std::move(surf)]() mutable {
        try { surf->ensureLoaded(); } catch (...) {}
        std::lock_guard<std::mutex> lk(g_asyncLoadMutex);
        g_asyncLoadInFlight.erase(surf.get());
    });
}

} // namespace

void SurfaceLRU::setMaxResident(std::size_t n)
{
    _maxResident = n;
    while (_order.size() > _maxResident) {
        evictOne();
    }
}

void SurfaceLRU::touch(const std::shared_ptr<QuadSurface>& surf)
{
    if (!surf) return;
    auto* raw = surf.get();
    auto it = _index.find(raw);
    if (it != _index.end()) {
        // Move to front.
        _order.splice(_order.begin(), _order, it->second);
    } else {
        _order.push_front(surf);
        _index[raw] = _order.begin();
        while (_order.size() > _maxResident) {
            evictOne();
        }
    }
    // Kick off the TIFF load on a worker if not already loaded. Keeps the
    // main thread (our caller) free for input processing — otherwise the
    // first rawPointsPtr() hit after eviction synchronously LZW-decodes
    // three ~170 MiB TIFFs on the main thread.
    asyncWarmSurface(surf);
}

void SurfaceLRU::pin(const std::shared_ptr<QuadSurface>& surf)
{
    if (!surf) return;
    _pinned.insert(surf.get());
    // Pinning implies recently used — bring to front so it's not the next
    // candidate when unpinned.
    touch(surf);
}

void SurfaceLRU::unpin(const std::shared_ptr<QuadSurface>& surf)
{
    if (!surf) return;
    _pinned.erase(surf.get());
}

void SurfaceLRU::forget(const std::shared_ptr<QuadSurface>& surf)
{
    if (!surf) return;
    auto* raw = surf.get();
    auto it = _index.find(raw);
    if (it == _index.end()) return;
    _order.erase(it->second);
    _index.erase(it);
    _pinned.erase(raw);
}

void SurfaceLRU::evictAll()
{
    auto it = _order.begin();
    while (it != _order.end()) {
        auto* raw = it->get();
        if (_pinned.count(raw)) { ++it; continue; }
        if (auto sp = *it) sp->unloadPoints();
        _index.erase(raw);
        it = _order.erase(it);
    }
}

void SurfaceLRU::evictOne()
{
    // Walk from the back (oldest) forward, skipping pinned entries.
    for (auto it = _order.rbegin(); it != _order.rend(); ++it) {
        auto* raw = it->get();
        if (_pinned.count(raw)) continue;
        auto sp = *it;
        if (sp) sp->unloadPoints();
        // Convert reverse_iterator to forward iterator for erase.
        auto fwd = std::next(it).base();
        _index.erase(raw);
        _order.erase(fwd);
        return;
    }
    // All pinned — nothing to evict.
}
