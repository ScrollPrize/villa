#include "SurfaceLRU.hpp"

#include "vc/core/util/QuadSurface.hpp"

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
        return;
    }
    _order.push_front(surf);
    _index[raw] = _order.begin();
    while (_order.size() > _maxResident) {
        evictOne();
    }
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
