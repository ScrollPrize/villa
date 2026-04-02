#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <unordered_set>

#include "vc/core/render/CoreRenderPool.hpp"
#include "vc/core/render/TileGrid.hpp"
#include "vc/core/render/TiledViewerCamera.hpp"
#include "vc/core/render/TileTypes.hpp"

class Surface;
class Volume;

namespace vc::render {

// Platform-agnostic viewport renderer.  Orchestrates tile rendering: submits
// visible tiles to the background pool, drains completed results, and updates
// the tile grid.  Replaces TileRenderController for headless / non-Qt use.
//
// All public methods must be called from a single thread (the "main" thread).
// The tick() method should be called periodically (~60 Hz) to drain results
// and perform progressive refinement.
class ViewportRenderer {
public:
    // Construct with an internal (owned) CoreRenderPool.
    explicit ViewportRenderer(int numThreads = 2);

    // Construct with an external (non-owned) CoreRenderPool.
    // The caller must ensure the pool outlives this ViewportRenderer.
    explicit ViewportRenderer(CoreRenderPool* externalPool);

    ~ViewportRenderer();

    // Camera state changes (pan, zoom, slice offset).
    // Submits visible tiles to the background pool (skipping in-flight duplicates).
    void onCameraChanged(const TiledViewerCamera& camera,
                         const std::shared_ptr<Surface>& surface,
                         const std::shared_ptr<Volume>& volume,
                         const std::function<TileRenderParams(const WorldTileKey&)>& buildParams,
                         float vpLeft, float vpTop, float vpRight, float vpBottom);

    // Rendering parameters changed (window/level, colormap, etc.) -- re-renders everything.
    void onParamsChanged(const TiledViewerCamera& camera,
                         const std::shared_ptr<Surface>& surface,
                         const std::shared_ptr<Volume>& volume,
                         const std::function<TileRenderParams(const WorldTileKey&)>& buildParams,
                         float vpLeft, float vpTop, float vpRight, float vpBottom);

    // Deferred render: stores state and processes on next tick.
    // Rapid calls coalesce into a single onCameraChanged().
    void scheduleRender(const TiledViewerCamera& camera,
                        const std::shared_ptr<Surface>& surface,
                        const std::shared_ptr<Volume>& volume,
                        const std::function<TileRenderParams(const WorldTileKey&)>& buildParams,
                        float vpLeft, float vpTop, float vpRight, float vpBottom);

    // Cancel all in-flight renders.
    void cancelAll();

    // Clear cached render state so stale callbacks cannot re-trigger renders.
    void clearState();

    // Dirty flags set by the caller, processed each tick.
    void markChunkArrived();
    void markOverlaysDirty();

    // Must be called periodically (~60 Hz).  Returns true if more work is pending
    // (caller should keep ticking).
    bool tick();

    // Access
    TileGrid& tileGrid() { return _tileGrid; }
    const TileGrid& tileGrid() const { return _tileGrid; }
    CoreRenderPool& renderPool() { return *pool(); }
    const CoreRenderPool& renderPool() const { return *pool(); }
    int pendingCount() const;

    // Callbacks
    void setTileReadyCallback(std::function<void()> cb);
    void setSceneUpdatedCallback(std::function<void()> cb);
    void setOverlayCallback(std::function<void()> cb);

    // Per-result callback: called for each drained result with the raw pixels.
    // The callback receives ownership of the result (pixels are valid).
    // TileGrid metadata is updated BEFORE the callback; pixels are NOT stored in TileGrid.
    void setResultCallback(std::function<void(TileRenderResult&&)> cb) { _resultCallback = std::move(cb); }

    void setProgressiveEnabled(bool enabled) { _progressiveEnabled = enabled; }
    bool progressiveEnabled() const { return _progressiveEnabled; }

    // Controller identity (for shared pool routing)
    int controllerId() const { return _controllerId; }

    // Current epoch
    uint64_t epoch() const { return _currentEpoch->load(std::memory_order_relaxed); }

private:
    void drainResults();

    // Returns the active pool (owned or external).
    CoreRenderPool* pool() { return _externalPool ? _externalPool : _renderPool.get(); }
    const CoreRenderPool* pool() const { return _externalPool ? _externalPool : _renderPool.get(); }

    TileGrid _tileGrid;
    std::unique_ptr<CoreRenderPool> _renderPool;  // owned pool (null when using external)
    CoreRenderPool* _externalPool = nullptr;       // non-owned external pool (null when using owned)

    std::shared_ptr<std::atomic<uint64_t>> _currentEpoch = std::make_shared<std::atomic<uint64_t>>(0);
    int _controllerId;
    static inline std::atomic<int> _nextControllerId{0};

    // Last viewport rect (float edges)
    float _lastVpL = 0, _lastVpT = 0, _lastVpR = 0, _lastVpB = 0;

    bool _progressiveEnabled = true;
    int _desiredLevel = 0;

    // Last render state for refinement re-submission
    TiledViewerCamera _lastCamera;
    std::shared_ptr<Surface> _lastSurface;
    std::shared_ptr<Volume> _lastVolume;
    std::function<TileRenderParams(const WorldTileKey&)> _lastBuildParams;

    // Pending state for coalescing rapid viewport changes
    bool _pendingDirty = false;
    TiledViewerCamera _pendingCamera;
    std::shared_ptr<Surface> _pendingSurface;
    std::shared_ptr<Volume> _pendingVolume;
    std::function<TileRenderParams(const WorldTileKey&)> _pendingBuildParams;
    float _pendingVpL = 0, _pendingVpT = 0, _pendingVpR = 0, _pendingVpB = 0;

    // In-flight tile tracking to avoid duplicate submissions
    std::unordered_set<WorldTileKey, WorldTileKeyHash> _inFlightTiles;

    // Dirty flags — set from signal handlers on potentially different threads,
    // read on the main thread in tick().
    std::atomic<bool> _overlaysDirty{false};
    std::atomic<bool> _chunkArrived{false};

    // Callbacks
    std::function<void()> _sceneUpdatedCallback;
    std::function<void()> _overlayCallback;
    std::function<void(TileRenderResult&&)> _resultCallback;
};

} // namespace vc::render
