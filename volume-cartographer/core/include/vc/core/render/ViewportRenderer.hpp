#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>

#include "vc/core/render/CoreRenderPool.hpp"
#include "vc/core/render/TileGrid.hpp"
#include "vc/core/render/TiledViewerCamera.hpp"
#include "vc/core/render/TileTypes.hpp"

class Surface;
class Volume;

namespace vc::render {

// One frame per tick. Submit visible tiles, take results, done.
class ViewportRenderer final {
public:
    explicit ViewportRenderer(int numThreads = 2);
    explicit ViewportRenderer(CoreRenderPool* externalPool);
    ~ViewportRenderer() noexcept = default;

    // Camera changed: clear queue, submit visible tiles.
    void onCameraChanged(const TiledViewerCamera& camera,
                         const std::shared_ptr<Surface>& surface,
                         const std::shared_ptr<Volume>& volume,
                         const std::function<TileRenderParams(const WorldTileKey&)>& buildParams,
                         float vpLeft, float vpTop, float vpRight, float vpBottom);

    // Same but with pre-computed visible keys (bypasses grid coordinate system).
    void onCameraChangedDirect(const TiledViewerCamera& camera,
                               const std::shared_ptr<Surface>& surface,
                               const std::shared_ptr<Volume>& volume,
                               const std::function<TileRenderParams(const WorldTileKey&)>& buildParams,
                               const std::vector<WorldTileKey>& visibleKeys);

    // Params changed (window/level etc): same as camera changed.
    void onParamsChanged(const TiledViewerCamera& camera,
                         const std::shared_ptr<Surface>& surface,
                         const std::shared_ptr<Volume>& volume,
                         const std::function<TileRenderParams(const WorldTileKey&)>& buildParams,
                         float vpLeft, float vpTop, float vpRight, float vpBottom);

    // Deferred render: coalesces rapid changes, fires on next tick.
    void scheduleRender(const TiledViewerCamera& camera,
                        const std::shared_ptr<Surface>& surface,
                        const std::shared_ptr<Volume>& volume,
                        const std::function<TileRenderParams(const WorldTileKey&)>& buildParams,
                        float vpLeft, float vpTop, float vpRight, float vpBottom);

    // Cancel queued work.
    void cancelAll();

    // Clear cached state so stale callbacks can't re-trigger renders.
    void clearState();

    // A chunk was loaded into hot cache. Only triggers re-render if
    // the chunk is at a useful pyramid level for the current view.
    void markChunkArrived(int chunkLevel);
    void markOverlaysDirty();

    // One tick = take results + blit + maybe refine. Returns true if more work pending.
    bool tick();

    // Access
    [[nodiscard]] TileGrid& tileGrid() noexcept { return _tileGrid; }
    [[nodiscard]] const TileGrid& tileGrid() const noexcept { return _tileGrid; }
    [[nodiscard]] CoreRenderPool& renderPool() noexcept { return *pool(); }
    [[nodiscard]] const CoreRenderPool& renderPool() const noexcept { return *pool(); }

    // Callbacks
    void setTileReadyCallback(std::function<void()> cb);
    void setSceneUpdatedCallback(std::function<void()> cb);
    void setOverlayCallback(std::function<void()> cb);
    void setResultCallback(std::function<void(TileRenderResult&&)> cb) { _resultCallback = std::move(cb); }

    [[nodiscard]] int controllerId() const noexcept { return _controllerId; }
    [[nodiscard]] uint32_t currentBatch() const noexcept { return _batch; }

private:

    [[nodiscard]] CoreRenderPool* pool() noexcept { return _externalPool ? _externalPool : _renderPool.get(); }
    [[nodiscard]] const CoreRenderPool* pool() const noexcept { return _externalPool ? _externalPool : _renderPool.get(); }

    TileGrid _tileGrid;
    std::unique_ptr<CoreRenderPool> _renderPool;
    CoreRenderPool* _externalPool = nullptr;

    int _controllerId;
    uint32_t _batch = 0;
    static inline std::atomic<int> _nextControllerId{0};

    float _lastVpL = 0, _lastVpT = 0, _lastVpR = 0, _lastVpB = 0;

    int _desiredLevel = 0;
    TiledViewerCamera _lastCamera;
    std::shared_ptr<Surface> _lastSurface;
    std::shared_ptr<Volume> _lastVolume;
    std::function<TileRenderParams(const WorldTileKey&)> _lastBuildParams;
    std::vector<WorldTileKey> _lastVisibleKeys;

    std::atomic<bool> _overlaysDirty{false};
    std::atomic<bool> _chunkArrived{false};

    std::function<void()> _sceneUpdatedCallback;
    std::function<void()> _overlayCallback;
    std::function<void(TileRenderResult&&)> _resultCallback;
};

} // namespace vc::render
