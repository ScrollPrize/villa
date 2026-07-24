#pragma once

#include <QElapsedTimer>
#include <QImage>
#include <QPointF>
#include <QPointer>
#include <QStringList>
#include <QWidget>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <source_location>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <opencv2/core/mat.hpp>

#include "CVolumeViewerView.hpp"
#include "VolumeViewerBase.hpp"
#include "annotation_tools/SameWrapAnnotationTool.hpp"
#include "vc/core/render/ChunkedPlaneSampler.hpp"
#include "vc/core/render/IChunkedArray.hpp"
#include "vc/core/types/Sampling.hpp"
#include "vc/core/util/Compositing.hpp"
#include "vc/core/util/SurfacePatchIndex.hpp"

class CState;
class QEvent;
class QGraphicsEllipseItem;
class QGraphicsItem;
class QGraphicsPathItem;
class QGraphicsScene;
class QGraphicsSimpleTextItem;
class QTimer;
struct POI;
class PlaneSurface;
class Surface;
class ViewerManager;
class ViewerStatsBar;
class VCCollection;
class Volume;

namespace vc::render { class ChunkCache; }

class CChunkedVolumeViewer : public QWidget, public VolumeViewerBase
{
    Q_OBJECT

public:
    struct CameraState {
        float surfacePtrX = 0.0f;
        float surfacePtrY = 0.0f;
        float scale = 1.0f;
        float zOffset = 0.0f;
        cv::Vec3f zOffsetWorldDir{0, 0, 0};
    };
    struct SceneVolumeSample {
        cv::Vec3f position{0, 0, 0};
        cv::Vec3f normal{0, 0, 1};
        Surface* surface = nullptr;
    };
    using ShiftScrollOverride = std::function<bool(int, QPointF, Qt::KeyboardModifiers)>;

    CChunkedVolumeViewer(CState* state, ViewerManager* manager, QWidget* parent = nullptr);
    ~CChunkedVolumeViewer() override;

    void setPointCollection(VCCollection* pc) { _pointCollection = pc; }
    void setSurface(const std::string& name) override;
    void setIntersects(const std::set<std::string>& names) override;
    void renderVisible(
        bool force = false,
        const char* reason = "external caller",
        std::source_location caller = std::source_location::current()) override;
    void requestRender(
        const char* reason = "external caller",
        std::source_location caller = std::source_location::current()) override;
    void serviceRenderTick() override;
    void invalidateVis() override;
    void invalidateVisRegion(const std::string& name, const cv::Rect& changedCells) override;
    void centerOnVolumePoint(const cv::Vec3f& point, bool forceRender = false) override;
    void centerOnSurfacePoint(const cv::Vec2f& point, bool forceRender = false) override;
    void adjustSurfaceOffset(float delta) override;
    void resetSurfaceOffsets() override;
    void fitSurfaceInView() override;
    void resetViewForCurrentContent(bool forceRender = false) override;
    void notifyInteractiveViewChange(double motionPx);

    std::string surfName() const override { return _surfName; }
    std::shared_ptr<Volume> currentVolume() const override { return _volume; }
    float getCurrentScale() const override { return _scale; }
    static float clampCameraScale(float scale);
    float dsScale() const override { return _dsScale; }
    float normalOffset() const override { return _zOff; }
    CameraState cameraState() const;
    void applyCameraState(const CameraState& state, bool forceRender = true);
    void applyCameraStateForReplayRepaint(const CameraState& state);
    // Render-bench helpers: true when no render is running/queued/pending; count of
    // remote chunk fetches still outstanding. Used by replay to settle each frame.
    bool isRenderQuiescent() const;
    std::size_t chunkFetchesInFlight() const;
    int datasetScaleIndex() const override { return _dsScaleIdx; }
    float datasetScaleFactor() const override { return _dsScale; }
    Surface* currentSurface() const override;
    VCCollection* pointCollection() const override { return _pointCollection; }

    void setCompositeRenderSettings(const CompositeRenderSettings& s) override { if (_closing) return; _compositeSettings = s; submitRender("setCompositeRenderSettings"); }
    const CompositeRenderSettings& compositeRenderSettings() const override { return _compositeSettings; }
    bool isCompositeEnabled() const override { return _compositeSettings.enabled && !streamingCompositeUnsupported(); }
    bool isPlaneCompositeEnabled() const override { return _compositeSettings.planeEnabled && !streamingCompositeUnsupported(); }

    void setVolumeWindow(float low, float high) override;
    void setBaseColormap(const std::string& id) override { if (_closing) return; _baseColormapId = id; submitRender("setBaseColormap"); }
    void setStretchValues(bool) { if (_closing) return; submitRender("setStretchValues"); }
    void setResetViewOnSurfaceChange(bool v) override { _resetViewOnSurfaceChange = v; }
    void setPlaneIntersectionLinesVisible(bool visible) override;
    bool isPlaneIntersectionLinesVisible() const { return _planeIntersectionLinesVisible; }

    void setShowDirectionHints(bool on) override { if (_closing) return; _showDirectionHints = on; emit overlaysUpdated(); }
    bool isShowDirectionHints() const override { return _showDirectionHints; }
    void setShowSurfaceNormals(bool on) override { if (_closing) return; _showSurfaceNormals = on; emit overlaysUpdated(); }
    bool isShowSurfaceNormals() const override { return _showSurfaceNormals; }
    float normalArrowLengthScale() const override { return _normalArrowLengthScale; }
    int normalMaxArrows() const override { return _normalMaxArrows; }
    void setNormalArrowLengthScale(float scale) override { if (_closing) return; _normalArrowLengthScale = scale; emit overlaysUpdated(); }
    void setNormalMaxArrows(int maxArrows) override { if (_closing) return; _normalMaxArrows = maxArrows; emit overlaysUpdated(); }

    void setOverlayVolume(std::shared_ptr<Volume> volume) override;
    void setOverlayOpacity(float opacity) override;
    void setOverlayColormap(const std::string& colormapId) override;
    void setOverlaySamplingMethod(vc::Sampling method) override;
    void setOverlayThreshold(float threshold) override;
    void setOverlayWindow(float low, float high) override;
    void setOverlayMaxDisplayedResolution(int level) override;
    void setOverlayComposite(const OverlayCompositeSettings& settings) override;

    void setSegmentationEditActive(bool active) override { if (_closing) return; _segmentationEditActive = active; }
    void setSegmentationIntersectionDeferral(bool active) override;
    void setSegmentationCursorMirroring(bool enabled) override;
    const ActiveSegmentationHandle& activeSegmentationHandle() const override;

    uint64_t highlightedPointId() const override { return _highlightedPointId; }
    uint64_t selectedPointId() const override { return _selectedPointId; }
    uint64_t selectedCollectionId() const override { return _selectedCollectionId; }
    bool isPointDragActive() const override { return false; }
    bool isSameWrapAnnotationModeEnabled() const override { return _sameWrapAnnotation.enabled(); }
    double sameWrapAnnotationPolylineOpacity() const override { return _sameWrapAnnotationPolylineOpacity; }
    const std::vector<ViewerOverlayControllerBase::PathPrimitive>& drawingPaths() const override;

    void setOverlayGroup(const std::string& key, const std::vector<QGraphicsItem*>& items) override;
    void clearOverlayGroup(const std::string& key) override;
    void clearAllOverlayGroups() override;

    std::vector<std::pair<QRectF, QColor>> selections() const override;
    std::optional<QRectF> activeBBoxSceneRect() const override;
    void setBBoxMode(bool enabled) override;
    QuadSurface* makeBBoxFilteredSurfaceFromSceneRect(const QRectF& sceneRect) override;
    void clearSelections() override;

    void renderIntersections(
        const char* reason = "external caller",
        std::source_location caller = std::source_location::current()) override;
    void scheduleIntersectionRender(
        const char* reason = "external caller",
        std::source_location caller = std::source_location::current()) override;
    void invalidateIntersect(const std::string& = "") override;
    void invalidateIntersectRegion(const std::string& name, const cv::Rect& changedCells) override;
    float intersectionOpacity() const override { return _intersectionOpacity; }
    float intersectionThickness() const override { return _intersectionThickness; }
    int surfacePatchSamplingStride() const override { return _surfacePatchSamplingStride; }
    void setIntersectionOpacity(float v) override;
    void setIntersectionThickness(float v) override;
    void setHighlightedSurfaceIds(const std::vector<std::string>& ids) override;
    std::vector<std::string> highlightedSurfaceIds() const {
        return {_highlightedSurfaceIds.begin(), _highlightedSurfaceIds.end()};
    }
    void setSurfacePatchSamplingStride(int s) override;

    bool surfaceOverlayEnabled() const override { return _surfaceOverlayEnabled; }
    const std::map<std::string, cv::Vec3b>& surfaceOverlays() const override;
    float surfaceOverlapThreshold() const override { return _surfaceOverlapThreshold; }
    void setSurfaceOverlayEnabled(bool enabled) override { if (_closing) return; _surfaceOverlayEnabled = enabled; emit overlaysUpdated(); }
    void setSurfaceOverlays(const std::map<std::string, cv::Vec3b>& overlays) override { if (_closing) return; _surfaceOverlays = overlays; emit overlaysUpdated(); }
    void setSurfaceOverlapThreshold(float threshold) override { if (_closing) return; _surfaceOverlapThreshold = std::max(0.0f, threshold); emit overlaysUpdated(); }

    QPointF volumeToScene(const cv::Vec3f& volPoint) override;
    cv::Vec3f sceneToVolume(const QPointF& scenePoint) const override;
    [[nodiscard]] std::optional<SceneVolumeSample> sampleSceneVolume(const QPointF& scenePoint) const;
    cv::Vec2f sceneToSurfaceCoords(const QPointF& scenePos) const override;
    QPointF surfaceCoordsToScene(float surfX, float surfY) const override { return surfaceToScene(surfX, surfY); }
    void setLinkedCursorVolumePoint(const std::optional<cv::Vec3f>& point) override;
    QPointF lastScenePosition() const override { return _lastScenePos; }
    void setLineAnnotationPlacementPreviewEnabled(bool enabled);
    bool lineAnnotationPlacementPreviewEnabled() const { return _lineAnnotationPlacementPreviewEnabled; }
    bool lineAnnotationPlacementMarkerVisible() const;
    bool measurementSupported() const;
    void startMeasurementMode();
    bool isMeasurementActive() const;
    void clearMeasurement();
    void markSurfaceGeometryChanged();
    void setShiftScrollOverride(ShiftScrollOverride override) { _shiftScrollOverride = std::move(override); }

    CVolumeViewerView* graphicsView() const override { return _view; }
    QObject* asQObject() override { return this; }
    QMetaObject::Connection connectOverlaysUpdated(
        QObject* receiver, const std::function<void()>& callback) override {
        return connect(this, &CChunkedVolumeViewer::overlaysUpdated, receiver, callback);
    }

    void reloadPerfSettings() override;

protected:
    bool eventFilter(QObject* watched, QEvent* event) override;
    void showEvent(QShowEvent* event) override;

public slots:
    void OnVolumeChanged(std::shared_ptr<Volume> vol);
    void onSurfaceChanged(const std::string& name, const std::shared_ptr<Surface>& surf, bool isEditUpdate = false);
    void onSurfaceWillBeDeleted(const std::string& name, const std::shared_ptr<Surface>& surf);
    void onVolumeClosing();
    void onZoom(int steps, QPointF scenePoint, Qt::KeyboardModifiers modifiers);
    void onResized();
    void onCursorMove(QPointF scenePos);
    void onPanStart(Qt::MouseButton, Qt::KeyboardModifiers);
    void onPanRelease(Qt::MouseButton, Qt::KeyboardModifiers);
    void onVolumeClicked(QPointF, Qt::MouseButton, Qt::KeyboardModifiers);
    void onMousePress(QPointF, Qt::MouseButton, Qt::KeyboardModifiers);
    void onMouseMove(QPointF, Qt::MouseButtons, Qt::KeyboardModifiers);
    void onMouseRelease(QPointF, Qt::MouseButton, Qt::KeyboardModifiers);
    void onKeyPress(int key, Qt::KeyboardModifiers modifiers);
    void onKeyRelease(int key, Qt::KeyboardModifiers modifiers);
    void onScrolled() {}
    void onPathsChanged(const QList<ViewerOverlayControllerBase::PathPrimitive>& paths);
    void onCollectionSelected(uint64_t collectionId);
    void onPointSelected(uint64_t pointId);
    void setSameWrapAnnotationMode(bool enabled);
    void setSameWrapAnnotationSpacing(double spacingVx);
    void setSameWrapAnnotationPolylineOpacity(double opacity);
    void setSameWrapAnnotationMergeExisting(bool enabled);
    void setSameWrapAnnotationPathType(int pathType);
    void setSameWrapAnnotationFilterType(int filterType);
    void setSameWrapAnnotationFilterKernelSize(int kernelSize);
    bool hasSameWrapAnnotationPreview() const;
    void clearSameWrapAnnotationPreview();
    bool commitSameWrapAnnotationPreview();
    bool undoSameWrapAnnotation();
    void onDrawingModeActive(bool, float = 3.0f, bool = false) {}
    void onPOIChanged(const std::string& name, POI* poi);
    void adjustZoomByFactor(float factor) override;

signals:
    void sendVolumeClicked(cv::Vec3f volLoc, cv::Vec3f normal, Surface* surf,
                           Qt::MouseButton buttons, Qt::KeyboardModifiers modifiers);
    void sendZSliceChanged(int zValue);
    void sendMousePressVolume(cv::Vec3f volLoc, cv::Vec3f normal,
                              Qt::MouseButton button, Qt::KeyboardModifiers modifiers,
                              QPointF scenePos);
    void sendMouseMoveVolume(cv::Vec3f volLoc, Qt::MouseButtons buttons,
                             Qt::KeyboardModifiers modifiers, QPointF scenePos);
    void sendMouseReleaseVolume(cv::Vec3f volLoc, Qt::MouseButton button,
                                Qt::KeyboardModifiers modifiers, QPointF scenePos);
    void sendMouseDoubleClickVolume(cv::Vec3f volLoc, Qt::MouseButton button,
                                    Qt::KeyboardModifiers modifiers);
    void sendLineAnnotationSeedRequested(cv::Vec3f volLoc, QPointF scenePos);
    void sendCollectionSelected(uint64_t collectionId);
    void pointSelected(uint64_t pointId);
    void pointClicked(uint64_t pointId);
    void overlaysUpdated();
    void renderFrameSubmitted(std::uint64_t serial);
    void renderFrameCompleted(std::uint64_t serial, double workerElapsedMs);
    void sendSegmentationRadiusWheel(int steps, QPointF scenePoint, cv::Vec3f worldPos);
    void sharedCacheStatsChanged(const QStringList& items);

private:
    void quiesceForClose();
    void submitRender(
        const char* reason = "internal caller",
        std::source_location caller = std::source_location::current());
    void updateStatusLabel();
    void rebuildChunkArray();
    void syncCameraTransform();
    void requestDirectPaint();
    void resizeFramebuffer();
    void recalcPyramidLevel();
    void updateScalebarScale();   // push µm/scene-px to the view's scalebar overlay
    void panByF(float dx, float dy);
    void zoomStepsAt(int steps, const QPointF& scenePos);
    // Multiply the current scale by `factor` (clamped to [kMinScale, kMaxScale]),
    // keeping the scene point under `scenePos` fixed. The shared core of both the
    // discrete wheel zoom (zoomStepsAt) and the continuous adjustZoomByFactor.
    void zoomByFactorAt(float factor, const QPointF& scenePos);
    bool isAxisAlignedView() const;
    void ensureDefaultSurface();
    void updateContentBounds();
    QPointF surfaceToScene(float surfX, float surfY) const;
    cv::Vec2f sceneToSurface(const QPointF& scenePos) const;
    struct GeneratedSurfaceCache;
    struct PendingRenderJob {
        std::uint64_t requestId = 0;
        int fbW = 0;
        int fbH = 0;
        float surfacePtrX = 0.0f;
        float surfacePtrY = 0.0f;
        float scale = 1.0f;
        float zOff = 0.0f;
        cv::Vec3f zOffWorldDir{0, 0, 0};
        int startLevel = 0;
        int overlayStartLevel = 0;
        vc::Sampling samplingMethod = vc::Sampling::Trilinear;
        vc::Sampling overlaySamplingMethod = vc::Sampling::Nearest;
        CompositeRenderSettings compositeSettings;
        float windowLow = 0.0f;
        float windowHigh = 255.0f;
        std::string baseColormapId;
        std::shared_ptr<Surface> surf;
        std::string surfaceName;
        std::shared_ptr<vc::render::ChunkCache> chunkArray;
        std::shared_ptr<Volume> overlayVolume;
        std::shared_ptr<vc::render::ChunkCache> overlayChunkArray;
        float overlayOpacity = 0.0f;
        std::string overlayColormapId;
        float overlayWindowLow = 0.0f;
        float overlayWindowHigh = 255.0f;
        OverlayCompositeSettings overlayComposite;
        std::uint64_t chunkContentEpoch = 0;
        std::uint64_t surfaceGeometryEpoch = 0;
        std::shared_ptr<GeneratedSurfaceCache> genCache;
        bool genCacheDirty = false;
        std::string profileReason;
        std::string profileCaller;
        std::chrono::steady_clock::time_point submittedAt;
    };
    std::optional<PendingRenderJob> captureRenderJob(
        const char* reason,
        std::source_location caller,
        const std::shared_ptr<Surface>& surf,
        int fbW,
        int fbH,
        std::chrono::steady_clock::time_point submittedAt);
    void startRenderJob(PendingRenderJob job);
    void submitPendingRenderJobIfNeeded();
    void updateDisplayedFramebufferMapping();
    static bool renderJobsEquivalentForDisplay(const PendingRenderJob& a,
                                               const PendingRenderJob& b);
    // Everything of renderJobsEquivalentForDisplay except the chunk-content
    // epoch and gen-cache dirtiness: true when two jobs sample the same
    // pixels from the same source, so one frame's data can stand in for
    // missing chunks in the other.
    static bool renderJobsSameGeometry(const PendingRenderJob& a,
                                       const PendingRenderJob& b);
    struct RenderContext;
    struct RenderResult;
    static RenderResult renderFrame(RenderContext ctx);
    void finishRenderOnMainThread(std::shared_ptr<RenderResult> result);
    void maybeDegradeForCachePressure(const RenderResult& result);
    void markInteractiveMotion(double motionPx);
    int renderStartLevel(bool preferSurfaceResolution = false) const;
    int overlayRenderStartLevel(bool preferSurfaceResolution = false) const;
    bool streamingCompositeUnsupported() const;
    std::optional<cv::Vec3f> cursorVolumePosition(const QPointF& scenePos) const;
    void refreshCursorPositionAt(const QPointF& scenePos);
    void updateCursorCrosshair(const QPointF& scenePos);
    void updateLineAnnotationPlacementMarker(const QPointF& scenePos);
    void clearLineAnnotationPlacementMarker();
    bool handleMeasurementClick(const QPointF& scenePos, Qt::MouseButton button, Qt::KeyboardModifiers modifiers);
    void refreshMeasurementOverlay();
    void updateFocusMarker(POI* poi = nullptr);
    void refreshSameWrapAnnotationOverlay();
    std::optional<std::pair<uint64_t, uint64_t>> pointAtScenePosition(const QPointF& scenePos);
    void clearIntersectionItems();
    void updateIntersectionPreviewTransform();
    void renderFlattenedIntersections(const std::shared_ptr<Surface>& surf,
                                      const char* reason,
                                      std::source_location caller);
    QRectF surfaceRectToSceneRect(const QRectF& surfRect) const;

    CState* _state = nullptr;
    QPointer<ViewerManager> _viewerManager;
    VCCollection* _pointCollection = nullptr;
    CVolumeViewerView* _view = nullptr;
    QGraphicsScene* _scene = nullptr;
    ViewerStatsBar* _statsBar = nullptr;
    // No per-viewer timers. ViewerManager's global clock only services
    // intersection/status maintenance; render requests submit immediately.
    bool _closing = false;
    bool _intersectionPending = false;
    int  _statusRefreshTicks = 0;  // counts down to the periodic status refresh
    // A render was requested while this viewer was hidden (minimized MDI subwindow /
    // background tab). We skip the work and set this; showEvent re-renders so the
    // viewer catches up on whatever went stale while invisible.
    bool _renderStaleWhileHidden = false;
    bool _segmentationEditActive = false;
    bool _deferSegmentationIntersections = false;
    bool _deferredSegmentationIntersectionsDirty = false;
    bool _suppressNextSurfaceEditRender = false;
    std::string _pendingRenderReason;
    std::string _pendingRenderCaller;
    std::string _pendingIntersectionReason;
    std::string _pendingIntersectionCaller;

    std::shared_ptr<Volume> _volume;
    std::weak_ptr<Surface> _surfWeak;
    std::shared_ptr<Surface> _defaultSurface;
    std::string _surfName;
    std::shared_ptr<vc::render::ChunkCache> _chunkArray;
    vc::render::IChunkedArray::ChunkReadyCallbackId _chunkCbId = 0;

    QImage _framebuffer;
    std::atomic<bool> _renderWorkerBusy{false};
    std::optional<PendingRenderJob> _activeRenderJob;
    std::optional<PendingRenderJob> _pendingRenderJob;
    std::optional<PendingRenderJob> _displayedRenderJob;
    // Last presented render (sample values + coverage). The next render of
    // the same geometry reuses its pixels where chunks are missing so a
    // transient cache miss never blanks an already-shown region.
    std::shared_ptr<RenderResult> _lastRenderResult;
    bool _pendingRenderDirty = false;
    std::uint64_t _renderRequestSerial = 0;
    std::uint64_t _chunkContentEpoch = 0;
    std::uint64_t _surfaceGeometryEpoch = 0;
    std::uint64_t _renderSerial = 0;
    cv::Mat_<uint8_t> _values;
    cv::Mat_<uint8_t> _coverage;
    std::shared_ptr<GeneratedSurfaceCache> _genSurfaceCache;
    bool _genCacheDirty = true;

    float _surfacePtrX = 0.0f;
    float _surfacePtrY = 0.0f;
    float _scale = 1.0f;
    float _dsScale = 1.0f;
    int _dsScaleIdx = 0;
    // Extra pyramid levels applied on top of _dsScaleIdx when the shared
    // chunk cache reports view-protection stalls (live working set larger
    // than capacity). Reset when the zoom crosses a level boundary.
    int _cachePressureLevelBias = 0;
    float _zOff = 0.0f;
    float _camSurfX = 0.0f;
    float _camSurfY = 0.0f;
    float _camScale = 1.0f;
    cv::Vec3f _zOffWorldDir{0, 0, 0};

    float _windowLow = 0.0f;
    float _windowHigh = 255.0f;
    std::string _baseColormapId;
    std::shared_ptr<Volume> _overlayVolume;
    std::shared_ptr<vc::render::ChunkCache> _overlayChunkArray;
    vc::render::IChunkedArray::ChunkReadyCallbackId _overlayChunkCbId = 0;
    float _overlayOpacity = 0.5f;
    std::string _overlayColormapId;
    vc::Sampling _overlaySamplingMethod = vc::Sampling::Nearest;
    float _overlayWindowLow = 0.0f;
    float _overlayWindowHigh = 255.0f;
    int _overlayMaxDisplayedResolution = 0;
    OverlayCompositeSettings _overlayComposite;

    CompositeRenderSettings _compositeSettings;
    bool _resetViewOnSurfaceChange = true;
    bool _planeIntersectionLinesVisible = true;
    float _panSensitivity = 1.0f;
    float _zoomSensitivity = 1.0f;
    float _zScrollSensitivity = 1.0f;
    float _linkedCursorViewTolerance = 10.0f;
    vc::Sampling _samplingMethod = vc::Sampling::Trilinear;
    int _maxDisplayedResolution = 0;
    bool _showDirectionHints = true;
    bool _showSurfaceNormals = false;
    float _normalArrowLengthScale = 1.0f;
    int _normalMaxArrows = 32;
    bool _surfaceOverlayEnabled = false;
    bool _initializedFirstSegmentationSurface = false;
    std::map<std::string, cv::Vec3b> _surfaceOverlays;
    float _surfaceOverlapThreshold = 5.0f;
    float _intersectionOpacity = 0.7f;
    float _intersectionThickness = 0.0f;
    int _surfacePatchSamplingStride = 2;
    std::set<std::string> _intersectTgts;
    std::unordered_set<std::string> _highlightedSurfaceIds;

    // Resolved intersection targets, cached across render ticks. With very
    // large sessions (~100k visible surfaces) re-resolving every name via
    // CState per renderIntersections call costs tens of milliseconds on the
    // main thread even when the fingerprint would hit. Invalidated when the
    // target names / highlight set change and (via CState::surfacesVersion)
    // when the surface map itself changes.
    struct ResolvedIntersectTargets {
        std::unordered_set<SurfacePatchIndex::SurfacePtr> targets;
        std::size_t targetHash = 0;
        QuadSurface* activeSeg = nullptr;
        std::uint64_t surfacesVersion = 0;
        bool valid = false;
    };
    ResolvedIntersectTargets _resolvedIntersectTargets;

    std::vector<QGraphicsItem*> _intersectionItems;
    float _intersectionItemsCamSurfX = 0.0f;
    float _intersectionItemsCamSurfY = 0.0f;
    float _intersectionItemsCamScale = 1.0f;
    bool _intersectionItemsHaveCamera = false;
    std::unordered_map<std::string, size_t> _surfaceColorAssignments;
    size_t _nextColorIndex = 0;

    struct IntersectFingerprint {
        int roiX = 0, roiY = 0, roiW = 0, roiH = 0;
        std::array<int, 3> planeOriginQ{};
        std::array<int, 3> planeNormalQ{};
        std::array<int, 3> planeBasisXQ{};
        std::array<int, 3> planeBasisYQ{};
        int opacityQ = -1;
        int thicknessQ = -1;
        int indexSamplingStride = 0;
        size_t patchCount = 0;
        size_t surfaceCount = 0;
        size_t targetHash = 0;
        size_t targetGenerationHash = 0;
        size_t activeSegHash = 0;
        size_t highlightedSurfaceHash = 0;
        size_t flattenedPlanesHash = 0;
        size_t cameraHash = 0;
        bool valid = false;
        bool operator==(const IntersectFingerprint&) const = default;
    };
    IntersectFingerprint _lastIntersectFp;

    struct IntersectionGeometryCache {
        cv::Rect roi;
        std::array<int, 3> planeOriginQ{};
        std::array<int, 3> planeNormalQ{};
        std::array<int, 3> planeBasisXQ{};
        std::array<int, 3> planeBasisYQ{};
        int indexSamplingStride = 0;
        size_t patchCount = 0;
        size_t surfaceCount = 0;
        size_t targetHash = 0;
        size_t targetGenerationHash = 0;
        bool valid = false;
        std::unordered_map<SurfacePatchIndex::SurfacePtr,
                           std::vector<SurfacePatchIndex::TriangleSegment>> intersections;
    };
    IntersectionGeometryCache _intersectionGeometryCache;

    // Async plane-intersection compute. The r-tree query + triangle clip is heavy
    // (~12% of the GUI tick); on a geometry-cache miss it runs on a worker and the
    // QGraphicsItems are built on the main thread when the result lands. _intersectGen
    // rises on every input change; a result whose gen no longer matches is dropped.
    // One compute at a time; the current scene items stand as the preview meanwhile.
    std::uint64_t _intersectGen = 0;
    bool _intersectComputeInFlight = false;
    void finishPlaneIntersectionCompute(
        std::uint64_t gen, cv::Rect cacheRoi, IntersectFingerprint fp,
        std::shared_ptr<std::unordered_map<SurfacePatchIndex::SurfacePtr,
            std::vector<SurfacePatchIndex::TriangleSegment>>> result);

    struct FlattenedIntersectionLine {
        int planeIndex = 0;
        QPointF a;
        QPointF b;
    };

    struct FlattenedIntersectionCache {
        QuadSurface* surface = nullptr;
        size_t planesHash = 0;
        int indexSamplingStride = 0;
        uint64_t generation = 0;
        bool valid = false;
        std::unordered_map<std::uint64_t, std::vector<FlattenedIntersectionLine>> cellLines;
        std::unordered_map<std::uint64_t, std::vector<QGraphicsPathItem*>> tileItems;
    };
    FlattenedIntersectionCache _flattenedIntersectionCache;
    std::optional<cv::Rect> _flattenedIntersectionDirtyCells;

    float _contentMinU = 0.0f;
    float _contentMaxU = 0.0f;
    float _contentMinV = 0.0f;
    float _contentMaxV = 0.0f;
    bool _isPanning = false;
    bool _panSmoothingInitialized = false;
    float _smoothedPanDx = 0.0f;
    float _smoothedPanDy = 0.0f;
    QPointF _lastPanSceneF;
    QPointF _lastScenePos;
    std::optional<cv::Vec3f> _lastCursorVolumePos;
    ShiftScrollOverride _shiftScrollOverride;

    std::vector<ViewerOverlayControllerBase::PathPrimitive> _drawingPaths;
    std::unordered_map<std::string, std::vector<QGraphicsItem*>> _overlayGroups;
    QGraphicsItem* _cursorCrosshair = nullptr;
    QGraphicsEllipseItem* _lineAnnotationPlacementMarker = nullptr;
    bool _lineAnnotationPlacementPreviewEnabled = false;
    QGraphicsItem* _focusMarker = nullptr;
    bool _segmentationCursorMirroring = false;

    struct MeasurementPoint {
        cv::Vec2f surface{0.0f, 0.0f};
        cv::Vec3f volume{0.0f, 0.0f, 0.0f};
    };
    struct MeasurementState {
        bool active = false;
        std::optional<MeasurementPoint> first;
        std::optional<MeasurementPoint> second;
    };
    MeasurementState _measurement;

    uint64_t _highlightedPointId = 0;
    uint64_t _selectedCollectionId = 0;
    uint64_t _selectedPointId = 0;

    SameWrapAnnotationTool _sameWrapAnnotation;
    double _sameWrapAnnotationPolylineOpacity = 0.75;
    bool _sameWrapManualMergePressConsumed = false;
    bool _sameWrapManualPathDragActive = false;

    bool _bboxMode = false;
    QPointF _bboxStart;
    std::optional<QRectF> _activeBBoxSurfRect;
    struct Selection {
        QRectF surfRect;
        QColor color;
    };
    std::vector<Selection> _selections;
};
