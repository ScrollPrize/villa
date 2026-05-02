#pragma once

#include <QWidget>
#include <QPointF>
#include <QRectF>
#include <QColor>
#include <QImage>
#include <QTransform>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <memory>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <optional>
#include <unordered_map>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <QPainterPath>

#include "VolumeViewerBase.hpp"
#include "CVolumeViewerView.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/types/Sampling.hpp"
#include "vc/core/cache/BlockPipeline.hpp"
#include "vc/core/render/AdaptiveCamera.hpp"
#include "vc/core/util/SurfacePatchIndex.hpp"

class QGraphicsScene;
class QGraphicsItem;
class QLabel;
class QTimer;
struct POI;
class CState;
class ViewerManager;
class PlaneSurface;

// Stub TileScene — satisfies code that null-checks tileScene() then calls methods
#include "adaptive/TileScene.hpp"

// Backward-compat alias — all external code still uses CTiledVolumeViewer
#define CTiledVolumeViewer CAdaptiveVolumeViewer

// Adaptive per-pixel volume viewer. Renders a PlaneSurface via
// sampleAdaptiveARGB32 directly to a viewport-sized framebuffer,
// with composite post-process (CLAHE, raking light), intersections,
// and overlays.
class CAdaptiveVolumeViewer : public QWidget, public VolumeViewerBase
{
    Q_OBJECT

public:
    CAdaptiveVolumeViewer(CState* state, ViewerManager* manager,
                          QWidget* parent = nullptr);
    ~CAdaptiveVolumeViewer();

    // --- Data setup ---
    void setPointCollection(VCCollection* pc) { _pointCollection = pc; }
    void setSurface(const std::string& name) override;
    void setIntersects(const std::set<std::string>& names) override { _intersectTgts = names; renderIntersections(); }

    // --- Rendering ---
    void renderVisible(bool force = false) override;
    void renderIntersections() override;
    // Synchronous body of renderIntersections(). The public override just
    // schedules a coalesced debounce so rapid-fire UI setters (opacity /
    // thickness / surface-change / intersect-target slider) don't each
    // pay the full rtree + triangle-clip cost on the main thread.
    void renderIntersectionsNow();
    void invalidateVis() {}
    void invalidateIntersect(const std::string& = "") override;
    void centerOnVolumePoint(const cv::Vec3f& point, bool forceRender = false) override;
    void centerOnSurfacePoint(const cv::Vec2f& point, bool forceRender = false);

    // --- Accessors ---
    std::string surfName() const override { return _surfName; }
    std::shared_ptr<Volume> currentVolume() const override { return _volume; }
    vc::cache::BlockPipeline* chunkCachePtr() const override {
        return _volume ? _volume->tieredCache() : nullptr;
    }
    int datasetScaleIndex() const override { return _camera.dsScaleIdx; }
    float datasetScaleFactor() const override { return _camera.dsScale; }
    float getCurrentScale() const override { return _camera.scale; }
    float dsScale() const override { return _camera.dsScale; }
    float normalOffset() const override { return _camera.zOff; }
    Surface* currentSurface() const override;
    VCCollection* pointCollection() const override { return _pointCollection; }

    // --- Settings passthrough ---
    // Equality guards: Qt UI routinely fires value-changed signals even
    // when the user's action rounds to the same stored value (e.g. a
    // drag-release that reports the current spinbox value). Without the
    // guard we'd queue a real render for every no-op signal — the 16 ms
    // coalesce timer still fires, but it triggers a full frame for zero
    // visible change. Skipping identical settings keeps the pipeline
    // idle when nothing actually changed.
    void setCompositeRenderSettings(const CompositeRenderSettings& s) override {
        if (_compositeSettings == s) return;
        _compositeSettings = s;
        scheduleRender();
    }
    const CompositeRenderSettings& compositeRenderSettings() const override { return _compositeSettings; }
    bool isCompositeEnabled() const override { return _compositeSettings.enabled; }
    bool isPlaneCompositeEnabled() const override { return _compositeSettings.planeEnabled; }

    void setVolumeWindow(float low, float high) override;
    float volumeWindowLow() const { return _windowLow; }
    float volumeWindowHigh() const { return _windowHigh; }
    void setBaseColormap(const std::string& id) override {
        if (_baseColormapId == id) return;
        _baseColormapId = id;
        scheduleRender();
    }
    void setStretchValues(bool) { scheduleRender(); }

    // --- Display stubs ---
    void setResetViewOnSurfaceChange(bool v) override { _resetViewOnSurfaceChange = v; }
    void setShowDirectionHints(bool on) override { _showDirectionHints = on; emit overlaysUpdated(); }
    bool isShowDirectionHints() const override { return _showDirectionHints; }
    void setShowSurfaceNormals(bool on) override { _showSurfaceNormals = on; emit overlaysUpdated(); }
    bool isShowSurfaceNormals() const override { return _showSurfaceNormals; }
    float normalArrowLengthScale() const override { return _normalArrowLengthScale; }
    int normalMaxArrows() const override { return _normalMaxArrows; }
    void setNormalArrowLengthScale(float scale) { _normalArrowLengthScale = scale; emit overlaysUpdated(); }
    void setNormalMaxArrows(int maxArrows) { _normalMaxArrows = maxArrows; emit overlaysUpdated(); }

    // --- Overlay volume ---
    void setOverlayVolume(std::shared_ptr<Volume> volume) override;
    void setOverlayOpacity(float opacity) override;
    void setOverlayColormap(const std::string& colormapId) override;
    void setOverlayThreshold(float threshold) override;
    void setOverlayWindow(float low, float high) override;

    // --- Segmentation stubs ---
    void setSegmentationEditActive(bool) override {}
    void setSegmentationCursorMirroring(bool) override {}
    const ActiveSegmentationHandle& activeSegmentationHandle() const override {
        static ActiveSegmentationHandle h;
        return h;
    }

    // --- Interaction stubs ---
    uint64_t highlightedPointId() const override { return 0; }
    uint64_t selectedPointId() const override { return 0; }
    uint64_t selectedCollectionId() const override { return 0; }
    bool isPointDragActive() const override { return false; }
    const std::vector<ViewerOverlayControllerBase::PathPrimitive>& drawingPaths() const override {
        static std::vector<ViewerOverlayControllerBase::PathPrimitive> empty;
        return empty;
    }

    // --- Overlay management ---
    void setOverlayGroup(const std::string& key, const std::vector<QGraphicsItem*>& items) override;
    void clearOverlayGroup(const std::string& key) override;
    void clearAllOverlayGroups() override;

    // --- BBox ---
    auto selections() const -> std::vector<std::pair<QRectF, QColor>> override;
    std::optional<QRectF> activeBBoxSceneRect() const override;

    // --- Intersection ---
    float intersectionOpacity() const override { return _intersectionOpacity; }
    float intersectionThickness() const override { return _intersectionThickness; }
    int surfacePatchSamplingStride() const override { return _surfacePatchSamplingStride; }
    void setIntersectionOpacity(float v) override { _intersectionOpacity = v; renderIntersections(); }
    void setIntersectionThickness(float v) override { _intersectionThickness = v; renderIntersections(); }
    void setHighlightedSurfaceIds(const std::vector<std::string>& ids) override {
        _highlightedSurfaceIds.clear();
        for (const auto& id : ids) _highlightedSurfaceIds.insert(id);
        renderIntersections();
    }
    void setSurfacePatchSamplingStride(int s) override { _surfacePatchSamplingStride = s; }

    // --- Surface overlays ---
    bool surfaceOverlayEnabled() const override { return _surfaceOverlayEnabled; }
    const std::map<std::string, cv::Vec3b>& surfaceOverlays() const override { return _surfaceOverlays; }
    float surfaceOverlapThreshold() const override { return _surfaceOverlapThreshold; }
    void setSurfaceOverlayEnabled(bool enabled) {
        if (_surfaceOverlayEnabled == enabled) return;
        _surfaceOverlayEnabled = enabled;
        emit overlaysUpdated();
    }
    void setSurfaceOverlays(const std::map<std::string, cv::Vec3b>& overlays) {
        if (_surfaceOverlays == overlays) return;
        _surfaceOverlays = overlays;
        if (_surfaceOverlayEnabled) emit overlaysUpdated();
    }
    void setSurfaceOverlapThreshold(float threshold) {
        const float clamped = std::max(0.0f, threshold);
        if (std::abs(_surfaceOverlapThreshold - clamped) < 1e-6f) return;
        _surfaceOverlapThreshold = clamped;
        if (_surfaceOverlayEnabled) emit overlaysUpdated();
    }

    // --- Coordinate transforms ---
    QPointF volumeToScene(const cv::Vec3f& vol_point) override;
    cv::Vec3f sceneToVolume(const QPointF& scenePoint) const override;
    cv::Vec2f sceneToSurfaceCoords(const QPointF& scenePos) const override;
    void setLinkedCursorVolumePoint(const std::optional<cv::Vec3f>&) override {}
    QPointF surfaceCoordsToScene(float surfX, float surfY) const { return surfaceToScene(surfX, surfY); }
    bool sceneToVolumePN(cv::Vec3f& p, cv::Vec3f& n, const QPointF& scenePos) const;
    QPointF lastScenePosition() const override { return _lastScenePos; }

    // --- Surface offset ---
    void adjustSurfaceOffset(float dn) override;
    void resetSurfaceOffsets() override;

    // --- BBox tool ---
    void setBBoxMode(bool enabled);
    bool isBBoxMode() const { return _bboxMode; }
    QuadSurface* makeBBoxFilteredSurfaceFromSceneRect(const QRectF& sceneRect);
    void clearSelections();

    // --- Compat accessors ---
    CVolumeViewerView* fGraphicsView = nullptr;  // alias for _view, set in constructor

    // Returns nullptr — diagnostic viewer has no TileScene.
    // Code that null-checks this will gracefully skip (e.g. segmentation overlay).
    TileScene* tileScene() const { return nullptr; }

    // --- VolumeViewerBase interface ---
    CVolumeViewerView* graphicsView() const override { return _view; }
    QObject* asQObject() override { return this; }
    QMetaObject::Connection connectOverlaysUpdated(
        QObject* receiver, const std::function<void()>& callback) override {
        return connect(this, &CAdaptiveVolumeViewer::overlaysUpdated, receiver, callback);
    }

    void updateStatusLabel();
    void fitSurfaceInView() override;
    void requestRender();
    bool isWindowMinimized() const;
    bool eventFilter(QObject* watched, QEvent* event) override;

public slots:
    void OnVolumeChanged(std::shared_ptr<Volume> vol);
    void onSurfaceChanged(const std::string& name, const std::shared_ptr<Surface>& surf, bool isEditUpdate = false);
    void onPOIChanged(const std::string& name, POI* poi);
    void onSurfaceWillBeDeleted(const std::string& name, const std::shared_ptr<Surface>& surf);
    void onVolumeClosing();

    void onZoom(int steps, QPointF scenePoint, Qt::KeyboardModifiers modifiers);
    void onResized();
    void onPanStart(Qt::MouseButton, Qt::KeyboardModifiers);
    void onPanRelease(Qt::MouseButton, Qt::KeyboardModifiers);
    void onVolumeClicked(QPointF, Qt::MouseButton, Qt::KeyboardModifiers);
    void onMousePress(QPointF, Qt::MouseButton, Qt::KeyboardModifiers);
    void onMouseMove(QPointF, Qt::MouseButtons, Qt::KeyboardModifiers);
    void onMouseRelease(QPointF, Qt::MouseButton, Qt::KeyboardModifiers);
    void onKeyPress(int key, Qt::KeyboardModifiers modifiers);
    void onKeyRelease(int, Qt::KeyboardModifiers) {}
    void onScrolled() {}

    void onCollectionSelected(uint64_t) {}
    void onPointSelected(uint64_t) {}
    void onPathsChanged(const QList<ViewerOverlayControllerBase::PathPrimitive>&) {}
    void onDrawingModeActive(bool, float = 3.0f, bool = false) {}
    void adjustZoomByFactor(float factor) override;

signals:
    void sendVolumeClicked(cv::Vec3f vol_loc, cv::Vec3f normal, Surface* surf,
                           Qt::MouseButton buttons, Qt::KeyboardModifiers modifiers);
    void sendZSliceChanged(int z_value);
    void sendMousePressVolume(cv::Vec3f vol_loc, cv::Vec3f normal,
                              Qt::MouseButton button, Qt::KeyboardModifiers modifiers, QPointF scenePos);
    void sendMouseMoveVolume(cv::Vec3f vol_loc, Qt::MouseButtons buttons,
                             Qt::KeyboardModifiers modifiers, QPointF scenePos);
    void sendMouseReleaseVolume(cv::Vec3f vol_loc, Qt::MouseButton button,
                                Qt::KeyboardModifiers modifiers, QPointF scenePos);
    void sendCollectionSelected(uint64_t collectionId);
    void pointSelected(uint64_t pointId);
    void pointClicked(uint64_t pointId);
    void overlaysUpdated();
    void sendSegmentationRadiusWheel(int steps, QPointF scenePoint, cv::Vec3f worldPos);

private:
    void submitRender();
    void scheduleRender();
    void syncCameraTransform();
    Q_INVOKABLE void handleSurfaceAccessException(const QString& context, const QString& message);

    // Async render pipeline. submitRender() is called on the main thread
    // (Qt render timer). It snapshots what the worker needs, dispatches
    // renderIntoFramebuffer() to QThreadPool, and returns immediately.
    // The worker writes into _framebufferWork (double-buffer) so the paint
    // event can keep reading _framebuffer. When the worker finishes it
    // posts finishRenderOnMainThread() via QueuedConnection — that slot
    // swaps the buffers and performs the main-thread-only scene updates.
    //
    // RenderContext is the immutable snapshot of viewer state that the
    // worker reads. The body of renderIntoFramebuffer() must NEVER touch
    // `_camera` / `_compositeSettings` / `_windowLow` / etc. directly —
    // those are mutated on the main thread by input handlers and
    // settings setters; reading them on a worker thread without a lock
    // is a data race. Everything the worker needs from mutable state is
    // copied into RenderContext in submitRender() before dispatch.
    struct RenderContext {
        AdaptiveCamera camera;
        CompositeRenderSettings compositeSettings;
        vc::Sampling samplingMethod;
        bool interactive = false;
        float windowLow = 0.0f;
        float windowHigh = 255.0f;
        std::string baseColormapId;
        bool highlightDownscaled = false;
        cv::Vec3f zOffWorldDir{0.0f, 0.0f, 0.0f};
        // Strong references so the worker can't race against object
        // teardown: _surfWeak / _volume could otherwise be reset on main
        // mid-render. shared_ptr captures here keep them alive for the
        // duration of the render.
        std::shared_ptr<Surface> surf;
        std::shared_ptr<Volume> volume;
        std::shared_ptr<Volume> overlayVolume;
        float overlayOpacity = 0.0f;
        std::string overlayColormapId;
        float overlayWindowLow = 0.0f;
        float overlayWindowHigh = 255.0f;
    };
    void renderIntoFramebuffer(QImage& fb, const RenderContext& ctx);
    Q_INVOKABLE void finishRenderOnMainThread();

    // Framebuffer coordinate conversions
    QPointF surfaceToScene(float surfX, float surfY) const;
    cv::Vec2f sceneToSurface(const QPointF& scenePos) const;
    QRectF surfaceRectToSceneRect(const QRectF& surfRect) const;
    void updateFocusMarker(POI* poi = nullptr);

    void renderFlattenedIntersections(const std::shared_ptr<Surface>& surf);

    void panByF(float dx, float dy);
    void zoomStepsAt(int steps, const QPointF& scenePos);

    bool isAxisAlignedView() const;

    struct CameraSceneSnapshot {
        float camSurfX = 0.0f;
        float camSurfY = 0.0f;
        float camScale = 1.0f;
        float vpCx = 0.0f;
        float vpCy = 0.0f;
        QTransform sceneToView;
        QTransform viewToScene;
        bool valid = false;
    };
    CameraSceneSnapshot cameraSceneSnapshot() const;
    void warpIntersectionItemsFrom(const CameraSceneSnapshot& oldCam);

    // --- Qt widgets ---
    CVolumeViewerView* _view = nullptr;
    QGraphicsScene* _scene = nullptr;
    QLabel* _lbl = nullptr;
    QTimer* _renderTimer = nullptr;
    bool _renderPending = false;

    // Progressive rendering: during live interaction (pan drag, zoom wheel
    // events) we render at +1 pyramid level for faster frames. When the
    // user stops interacting, an idle timer fires and we kick a full-res
    // render to catch up. This trades a slightly-softer image during
    // motion for materially lower frame time, which in turn makes pans/
    // zooms feel smoother without touching the sample kernel.
    QTimer* _interactionIdleTimer = nullptr;
    bool _interactive = false;
    // True while panning/zooming, even when progressive rendering is off.
    // Used to defer intersection recompute while warped paths track motion.
    bool _navigationInteractionActive = false;
    void beginInteraction();

    // --- Framebuffer ---
    // _framebuffer — last committed frame, read by the view's paint path
    // (CVolumeViewerView::drawBackground holds a pointer to it).
    // _framebufferWork — scratch that the async worker writes into.
    // finishRenderOnMainThread() does std::swap on the two QImages (main
    // thread only, so no paint/swap race) to commit the new frame.
    QImage _framebuffer;
    QImage _framebufferWork;
    QImage _overlayFramebuffer;
    // Serialises render dispatch. Only one worker runs at a time; if
    // submitRender fires while busy, the next render is queued via
    // _renderPendingAfterWorker so we don't lose a frame's worth of
    // state change (slider drag, camera pan).
    std::atomic<bool> _renderWorkerBusy{false};
    std::atomic<bool> _renderWorkerFailed{false};
    bool _renderPendingAfterWorker = false;
    std::chrono::steady_clock::time_point _renderT0{};
    // Per-pixel pyramid-level tag (0 = desired, 1..5 = fallback depth).
    // Allocated lazily when "highlight downscaled chunks" is enabled.
    cv::Mat_<uint8_t> _levelBuffer;
    float _camSurfX = 0, _camSurfY = 0, _camScale = 1.0f;

    // --- Data ---
    std::shared_ptr<Volume> _volume;
    std::weak_ptr<Surface> _surfWeak;
    std::shared_ptr<Surface> _defaultSurface;
    std::string _surfName;
    CState* _state = nullptr;
    ViewerManager* _viewerManager = nullptr;
    VCCollection* _pointCollection = nullptr;

    // --- Camera ---
    AdaptiveCamera _camera;
    float _windowLow = 0.0f;
    float _windowHigh = 255.0f;
    std::string _baseColormapId;
    std::shared_ptr<Volume> _overlayVolume;
    float _overlayOpacity{0.5f};
    std::string _overlayColormapId;
    float _overlayWindowLow{0.0f};
    float _overlayWindowHigh{255.0f};
    bool _surfaceOverlayEnabled{false};
    std::map<std::string, cv::Vec3b> _surfaceOverlays;
    float _surfaceOverlapThreshold{5.0f};
    // Flattened-view z-scroll translation direction (unit vector in world
    // space). Captured at each shift+scroll from the surface normal under
    // the view center so the translation is rigid — plain zoom never
    // exposes curvature drift.
    cv::Vec3f _zOffWorldDir{0.0f, 0.0f, 0.0f};

    // LUT cache: rebuild only when inputs change.
    std::array<uint32_t, 256> _cachedLut{};
    float _cachedWindowLow = -1.f;
    float _cachedWindowHigh = -1.f;
    std::string _cachedColormapId;
    uint8_t _cachedIsoCutoff = 255;  // force first-build mismatch

    // Stretch post-pass: cached min/max reused next frame so static views
    // render in one pass. A camera change invalidates via _cachedStretchValid.
    bool _cachedStretchValid = false;
    int _cachedStretchLo = 0;
    int _cachedStretchHi = 255;
    CompositeRenderSettings _compositeSettings;
    bool _resetViewOnSurfaceChange = true;
    float _panSensitivity = 1.0f;
    float _zoomSensitivity = 1.0f;
    float _zScrollSensitivity = 1.0f;
    vc::Sampling _samplingMethod = vc::Sampling::Trilinear;
    bool _highlightDownscaled = false;
    bool _showDirectionHints = true;
    bool _showSurfaceNormals = false;
    float _normalArrowLengthScale = 1.0f;
    int _normalMaxArrows = 32;
    QString _lastStatusText;
    std::chrono::steady_clock::time_point _lastStatusUpdate{};
    // FPS tracking — ring buffer of recent submitRender() timestamps.
    // Status label reads the newest minus the oldest to get an averaged
    // fps number; single-frame interval is too noisy to display.
    // FPS ring stores per-frame render DURATIONS (seconds), not render
    // timestamps. The reported value is 1 / mean(duration) — i.e., the
    // theoretical framerate we would sustain at our measured render
    // cost. Decouples the readout from user-input pacing so an idle
    // viewer doesn't read 0 FPS and a held-button pan doesn't read the
    // timer interval as "60 fps".
    static constexpr int kFpsRingSize = 32;
    std::array<double, kFpsRingSize> _renderDurationsSec{};
    int _renderDurationHead = 0;
    int _renderDurationCount = 0;
    void recordRenderDuration(double seconds);
    float measuredFps() const;
    std::chrono::steady_clock::time_point _lastStretchScan{};
    cv::Ptr<cv::CLAHE> _claheCache;
    int _claheCacheTile = -1;
    double _claheCacheClip = -1.0;
    cv::Mat _rakingGx, _rakingGy;
    // Reused gray-domain buffer for CLAHE / raking post-pass. Allocated
    // once, resized on viewport change — avoids fbH×fbW alloc every frame.
    cv::Mat_<uint8_t> _grayBuf;
    std::array<uint32_t, 256> _deferredCmapLut{};
    std::string _deferredCmapId;
    bool _deferredCmapValid = false;

    // Cache for surf->gen() output. QuadSurface's gen does cv::warpAffine
    // + cv::remap under the hood (~13% of render time when it runs every
    // frame). Most frames are identical camera state — we can reuse the
    // coords/normals buffers when the cache key matches.
    cv::Mat_<cv::Vec3f> _genCoords;
    cv::Mat_<cv::Vec3f> _genNormals;
    int _genCacheFbW = 0;
    int _genCacheFbH = 0;
    float _genCacheScale = 0.0f;
    cv::Vec3f _genCacheOffset{0, 0, 0};
    bool _genCacheWantComposite = false;
    Surface* _genCacheSurfKey = nullptr;
    bool _genCacheDirty = true;
    float _genCacheZOff = 0.0f;
    cv::Vec3f _genCacheZOffDir{0.0f, 0.0f, 0.0f};
    int _genCachePrefetchLevel = -1;
    int _genCachePrefetchNumLevels = -1;

public:
    // Re-reads perf/interaction settings from disk into cached members.
    // Call after the user toggles values in the Viewer Controls panel;
    // submitRender() reads the cached members on every frame instead of
    // opening QSettings.
    void reloadPerfSettings();

private:

    // --- Content bounds for pan clamping ---
    float _contentMinU = 0, _contentMaxU = 0;
    float _contentMinV = 0, _contentMaxV = 0;

    // --- Pan state ---
    bool _isPanning = false;
    QPointF _lastPanSceneF;
    QPointF _lastScenePos;

    // --- Overlay groups (stored for VolumeViewerBase interface) ---
    std::unordered_map<std::string, std::vector<QGraphicsItem*>> _overlayGroups;
    QGraphicsItem* _focusMarker = nullptr;

    // --- BBox tool state ---
    bool _bboxMode = false;
    QPointF _bboxStart;
    std::optional<QRectF> _activeBBoxSurfRect;
    struct Selection {
        QRectF surfRect;
        QColor color;
    };
    std::vector<Selection> _selections;

    // --- Intersection overlay ---
    std::set<std::string> _intersectTgts;
    std::unordered_set<std::string> _highlightedSurfaceIds;
    std::vector<QGraphicsItem*> _intersectionItems;
    float _intersectionOpacity = 0.7f;
    float _intersectionThickness = 0.0f;
    int _surfacePatchSamplingStride = 2;
    std::unordered_map<std::string, size_t> _surfaceColorAssignments;
    size_t _nextColorIndex = 0;

    // Intersection cache fingerprint: skip the whole rebuild if nothing changed.
    struct IntersectFingerprint {
        int roiX = 0, roiY = 0, roiW = 0, roiH = 0;
        std::array<int, 3> planeOriginQ{};
        std::array<int, 3> planeNormalQ{};
        std::array<int, 3> planeBasisXQ{};
        std::array<int, 3> planeBasisYQ{};
        // Quantized to 0.001 to avoid spurious cache misses from
        // sub-slider float jitter.
        int opacityQ = -1;
        int thicknessQ = -1;
        size_t patchCount = 0;
        size_t surfaceCount = 0;
        size_t targetHash = 0;
        size_t targetGenerationHash = 0;
        size_t activeSegHash = 0;
        size_t highlightedSurfaceHash = 0;
        // Hash of the three seg xy/xz/yz plane poses for the flattened-
        // view path. 0 on the plane-view path (which uses the
        // plane{Origin,Normal,BasisX,BasisY}Q fields instead).
        size_t flattenedPlanesHash = 0;
        // Hash of everything surfaceToScene() consumes: _camSurfX/Y/Scale,
        // framebuffer size, and the QGraphicsView affine. The flattened-
        // view path emits scene coords directly from surface coords via
        // surfaceToScene(), so a pan/zoom with no other fingerprint field
        // changing must still force a rebuild — otherwise cached overlay
        // items stay at stale scene positions. Plane view path gets it
        // implicitly via roi{X,Y,W,H} (derived from _view->mapToScene)
        // so cameraHash stays 0 there.
        size_t cameraHash = 0;
        bool valid = false;
        bool operator==(const IntersectFingerprint&) const = default;
    };
    IntersectFingerprint _lastIntersectFp;
    // Coalescing flag. renderIntersections() (public, called from UI
    // setters) just sets this and reuses the existing _renderTimer
    // tick; the actual rtree + triangle-clip work runs once per tick
    // inside renderIntersectionsNow().
    bool _intersectionsDirty = false;

    // Flattened-intersection worker state. The heavy rtree walk + triangle
    // clip used to run on the main thread, blocking input processing on
    // surfaces with millions of patches. We now dispatch it to a thread
    // pool task; while a worker is in flight we skip scheduling another.
    // Cooked paths are handed back via a queued QMetaObject::invokeMethod
    // lambda and applied to the scene on the main thread.
    std::atomic<bool> _flattenedWorkerBusy{false};
    // Same for the plane-view intersection path (computePlaneIntersections
    // + grouping). Async pattern is identical — worker computes the raw
    // intersection segments, main thread builds QGraphicsPathItems.
    std::atomic<bool> _planeWorkerBusy{false};
    // Counts all background tasks that hold `this` (render worker,
    // intersection workers). The destructor spins on this reaching zero
    // before letting members go — otherwise a QThreadPool task still
    // running inside renderIntoFramebuffer() or an intersection compute
    // will dereference freed memory on app exit (observed as SIGSEGV
    // at shutdown after a warm-cache session).
    std::atomic<int> _backgroundWorkers{0};

    // --- Chunk-ready listener ---
    vc::cache::BlockPipeline::ChunkReadyCallbackId _chunkCbId = 0;
    vc::cache::BlockPipeline::ChunkReadyCallbackId _overlayChunkCbId = 0;
    bool _hadValidDataBounds = false;
    bool _dirtyWhileMinimized = false;

    // TickCoordinator viewport slot. Acquired lazily on first publish and
    // released in the destructor. -1 means "no slot" (either coordinator
    // missing, or allocation table was full at ctor time).
    int _tickViewportSlot = -1;
};
