#pragma once

#include <list>
#include <memory>
#include <unordered_map>
#include <unordered_set>

class QuadSurface;

// LRU cache that keeps a bounded number of QuadSurface point grids loaded.
// Surfaces that drop off the LRU have unloadPoints() called; ensureLoaded()
// will transparently re-read from disk on next access.
//
// Pinned surfaces (active segment, viewer-highlighted) are never evicted
// regardless of their LRU position.
class SurfaceLRU {
public:
    // 4 resident is enough for (a) the active segmentation, (b) whatever
    // the viewer was just looking at, (c-d) nearby neighbours the user may
    // click next. Higher values don't improve interactive feel but cost
    // ~170 MiB per slot (a segment's _points Mat). Pinned surfaces are
    // exempt from this cap.
    explicit SurfaceLRU(std::size_t maxResident = 4) : _maxResident(maxResident) {}

    void setMaxResident(std::size_t n);

    // Mark a surface as recently used. Inserts on first call. Evicts the
    // oldest unpinned surface when count exceeds maxResident.
    void touch(const std::shared_ptr<QuadSurface>& surf);

    // Pin/unpin a surface — pinned surfaces are exempt from eviction.
    void pin(const std::shared_ptr<QuadSurface>& surf);
    void unpin(const std::shared_ptr<QuadSurface>& surf);

    // Remove a surface from tracking entirely (does not unload).
    void forget(const std::shared_ptr<QuadSurface>& surf);

    // Drop all unpinned surfaces.
    void evictAll();

private:
    void evictOne();

    std::size_t _maxResident;
    // Most-recently-used at front, least at back.
    std::list<std::shared_ptr<QuadSurface>> _order;
    std::unordered_map<QuadSurface*, std::list<std::shared_ptr<QuadSurface>>::iterator> _index;
    std::unordered_set<QuadSurface*> _pinned;
};
