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
#include <limits>
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
#include "VolumetricCompositor.hpp"
#include "ViewerSampling.hpp"
#include "annotation_tools/SameWrapAnnotationTool.hpp"
#include "vc/core/util/Compositing.hpp"
#include "vc/core/util/Rect3D.hpp"

class CState;
class CameraGizmoWidget;
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
class QuadSurface;
class ViewerManager;
class ViewerStatsBar;
class VCCollection;
class Volume;

class CChunkedVolumeViewer : public QWidget, public VolumeViewerBase
{
    Q_OBJECT

public:
    struct CameraState {
        float surfacePtrX = 0.0f;
        float surfacePtrY = 0.0f;
        float scale = 1.0f;
        float zOffset = 0.0f;
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
        const char* reason = "external caller",
        std::source_location caller = std::source_location::current()) override;
    void requestRender(
        const char* reason = "external caller",
        std::source_location caller = std::source_location::current()) override;
    void refreshStatus() override;
    void invalidateVis() override;
    void centerOnVolumePoint(const cv::Vec3f& point) override;
    void centerOnSurfacePoint(const cv::Vec2f& point) override;
    void adjustSurfaceOffset(float delta) override;
    void resetSurfaceOffsets() override;
    void fitSurfaceInView() override;
    void resetViewForCurrentContent() override;
    void notifyInteractiveViewChange();

    std::string surfName() const override { return _surfName; }
    std::shared_ptr<Volume> currentVolume() const override { return _volume; }
    float getCurrentScale() const override { return _scale; }
    static float clampCameraScale(float scale);
    float normalOffset() const override { return _zOff; }
    CameraState cameraState() const;
    void applyCameraState(const CameraState& state);
    // Pins the camera's surface Y: pans move only along X, zooms are anchored
    // on that row, and resets, centering and applied camera states land back
    // on it. Enforced in syncCameraTransform(), i.e. before any render is
    // submitted or repainted, so no frame is ever shown off the pinned row.
    void setPinnedSurfaceY(std::optional<float> surfaceY);
    Surface* currentSurface() const override;
    VCCollection* pointCollection() const override { return _pointCollection; }

    void setCompositeRenderSettings(const CompositeRenderSettings& s) override;
    const CompositeRenderSettings& compositeRenderSettings() const override { return _compositeSettings; }
    // Set by CWindow for the axis-aligned slice views: their volumetric-camera
    // azimuth is folded into the slice plane's basis by
    // AxisAlignedSliceController, so the render/mapping paths here must treat
    // the compositor azimuth as 0 (the gizmo still owns the value).
    void setVolumetricAzimuthInSurface(bool on) { _volumetricAzimuthInSurface = on; }
    bool isCompositeEnabled() const override { return _compositeSettings.enabled; }
    bool isPlaneCompositeEnabled() const override { return _compositeSettings.planeEnabled; }

    void setVolumeWindow(float low, float high) override;
    void setResetViewOnSurfaceChange(bool v) override { _resetViewOnSurfaceChange = v; }
    void setPlaneIntersectionLinesVisible(bool visible) override;
    bool planeIntersectionLinesVisible() const override { return _planeIntersectionLinesVisible; }

    void setShowSurfaceNormals(bool on) override { if (_closing) return; _showSurfaceNormals = on; emit overlaysUpdated(); }
    bool isShowSurfaceNormals() const override { return _showSurfaceNormals; }
    void setShowCoordinateFrame(bool show);
    float normalArrowLengthScale() const override { return _normalArrowLengthScale; }
    int normalMaxArrows() const override { return _normalMaxArrows; }
    void setNormalArrowLengthScale(float scale) override { if (_closing) return; _normalArrowLengthScale = scale; emit overlaysUpdated(); }
    void setNormalMaxArrows(int maxArrows) override { if (_closing) return; _normalMaxArrows = maxArrows; emit overlaysUpdated(); }

    void setOverlayVolume(std::shared_ptr<Volume> volume) override;
    void setOverlayOpacity(float opacity) override;
    void setOverlayColormap(const std::string& colormapId) override;
    void setOverlaySamplingMethod(int method) override;
    void setOverlayIntensityAsAlpha(bool enabled) override;
    void setOverlayWindow(float low, float high) override;
    void setOverlayMaxDisplayedResolution(int level) override;
    void setOverlayComposite(const OverlayCompositeSettings& settings) override;

    void setSegmentationCursorMirroring(bool enabled) override;

    std::optional<vc::PointRef> highlightedPoint() const override { return _highlightedPoint; }
    std::optional<vc::PointRef> selectedPoint() const override { return _selectedPoint; }
    bool isSameWrapAnnotationModeEnabled() const override { return _sameWrapAnnotation.enabled(); }
    double sameWrapAnnotationPolylineOpacity() const override { return _sameWrapAnnotationPolylineOpacity; }
    const std::vector<ViewerOverlayControllerBase::PathPrimitive>& drawingPaths() const override;

    void setOverlayGroup(const std::string& key, const std::vector<QGraphicsItem*>& items) override;
    // Moves every item registered under `key` by `delta` scene units; false
    // when no group is registered under that key. Overlay scene coordinates
    // are affine in the camera pointer at a fixed zoom, so a pan can shift a
    // group in place instead of rebuilding it.
    bool translateOverlayGroup(const std::string& key, const QPointF& delta);
    void clearOverlayGroup(const std::string& key) override;

    std::vector<std::pair<QRectF, QColor>> selections() const override;
    std::optional<QRectF> activeBBoxSceneRect() const override;
    void setBBoxMode(bool enabled) override;
    QuadSurface* makeBBoxFilteredSurfaceFromSceneRect(const QRectF& sceneRect) override;
    void clearSelections() override;

    const std::set<std::string>& intersects() const override { return _intersectTgts; }
    float intersectionOpacity() const override { return _intersectionOpacity; }
    float intersectionThickness() const override { return _intersectionThickness; }
    void setIntersectionOpacity(float v) override;
    void setIntersectionThickness(float v) override;

    bool surfaceOverlayEnabled() const override { return _surfaceOverlayEnabled; }
    const std::map<std::string, cv::Vec3b>& surfaceOverlays() const override;
    std::uint64_t surfaceOverlaysRevision() const override { return _surfaceOverlaysRevision; }
    float surfaceOverlapThreshold() const override { return _surfaceOverlapThreshold; }
    void setSurfaceOverlayEnabled(bool enabled) override
    {
        if (_closing || _surfaceOverlayEnabled == enabled) return;
        _surfaceOverlayEnabled = enabled;
        emit overlaysUpdated();
    }
    void setSurfaceOverlays(const std::map<std::string, cv::Vec3b>& overlays) override
    {
        if (_closing || _surfaceOverlays == overlays) return;
        _surfaceOverlays = overlays;
        ++_surfaceOverlaysRevision;
        emit overlaysUpdated();
    }

    std::optional<SurfaceProjection> projectVolumePoint(
        const cv::Vec3f& volPoint, float depthTolerance) const override;
    QPointF surfaceProjectionToScene(const SurfaceProjection& projection) const override;
    SurfaceProjectionContext surfaceProjectionContext() const override;
    QPointF volumeToScene(const cv::Vec3f& volPoint) override;
    cv::Vec3f sceneToVolume(const QPointF& scenePoint) const override;
    [[nodiscard]] std::optional<SceneVolumeSample> sampleSceneVolume(const QPointF& scenePoint) const;
    cv::Vec2f sceneToSurfaceCoords(const QPointF& scenePos) const override;
    QPointF surfaceCoordsToScene(float surfX, float surfY) const override { return surfaceToScene(surfX, surfY); }
    void setLinkedCursorVolumePoint(const std::optional<cv::Vec3f>& point) override;
    void setLocalCursorCrosshairSuppressed(bool suppressed) override;
    QPointF lastScenePosition() const override { return _lastScenePos; }
    void setLineAnnotationPlacementPreviewEnabled(bool enabled);
    bool lineAnnotationPlacementPreviewEnabled() const { return _lineAnnotationPlacementPreviewEnabled; }
    bool measurementSupported() const;
    void startMeasurementMode();
    bool isMeasurementActive() const;
    void clearMeasurement();
    void markSurfaceGeometryChanged();
    // Geometry-epoch introspection: lets callers tell whether the currently
    // DISPLAYED frame was rendered from the current surface geometry (stale
    // in-flight frames adopt with an older epoch).
    std::uint64_t surfaceGeometryEpoch() const { return _surfaceGeometryEpoch; }
    bool hasDisplayedRenderFrame() const { return _displayedRenderJob.has_value(); }
    std::uint64_t displayedSurfaceGeometryEpoch() const
    {
        return _displayedRenderJob ? _displayedRenderJob->surfaceGeometryEpoch : 0;
    }
    void setShiftScrollOverride(ShiftScrollOverride override) { _shiftScrollOverride = std::move(override); }
    // Accept linked-cursor points regardless of the global "Sync cursor across
    // views" toggle (used by the line annotation window's pane group, whose
    // mirroring is dialog-local).
    void setLinkedCursorAlwaysEnabled(bool enabled) { _linkedCursorAlwaysEnabled = enabled; }
    // Reject linked-cursor points outright, overriding both of the above. The
    // line annotation window needs this to keep mirrored crosses out of its
    // panes while the global toggle is on.
    void setLinkedCursorMirroringSuppressed(bool suppressed)
    {
        _linkedCursorMirroringSuppressed = suppressed;
    }

    CVolumeViewerView* graphicsView() const override { return _view; }
    QObject* asQObject() override { return this; }
    QMetaObject::Connection connectOverlaysUpdated(
        QObject* receiver, const std::function<void()>& callback) override {
        return connect(this, &CChunkedVolumeViewer::overlaysUpdated, receiver, callback);
    }
    // Re-runs every connectOverlaysUpdated callback: for a data change the
    // viewer itself cannot see (an overlay's source edited elsewhere).
    void refreshOverlays() { if (_closing) return; emit overlaysUpdated(); }
    void reloadPerfSettings() override;

protected:
    bool eventFilter(QObject* watched, QEvent* event) override;
    void showEvent(QShowEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;

public slots:
    void OnVolumeChanged(std::shared_ptr<Volume> vol);
    // Thin guard around onSurfaceChangedImpl; see the definition for why the
    // lazy surface load has to be contained before it reaches Qt.
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
    void onPathsChanged(const QList<ViewerOverlayControllerBase::PathPrimitive>& paths);
    void onCollectionSelected(uint64_t collectionId);
    void clearCollectionSelection();
    void onPointSelected(vc::PointRef point);
    void clearPointSelection();
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
    void pointSelected(vc::PointRef point);
    void pointClicked(vc::PointRef point);
    void overlaysUpdated();
    void renderFrameCompleted(std::uint64_t serial, double workerElapsedMs);
    void sendSegmentationRadiusWheel(int steps, QPointF scenePoint, cv::Vec3f worldPos);
    void sharedCacheStatsChanged(const QStringList& items);
    // Volumetric camera edited from inside the viewer (gizmo drag), so
    // external panels can refresh their yaw/pitch/perspective readouts.
    void compositeCameraChanged();

private:
    void quiesceForClose();
    void submitRender(
        const char* reason = "internal caller",
        std::source_location caller = std::source_location::current());
    void updateStatusLabel();
    void repositionStatsBarRight();
    void notifyNormalOffsetChanged();
    void setZOffset(float value);
    void openLevels();
    // Starts the background read of what the last frames could not sample from
    // the caches (see ViewerLoad), unless one is running.
    void startPendingLoad();
    void clearDisplayedFramebuffer();
    void syncCameraTransform();
    void requestDirectPaint();
    void resizeFramebuffer();
    void recalcPyramidLevel();
    void updateScalebarScale();   // push µm/scene-px to the view's scalebar overlay
    // Push the displayed plane's basis to the view's coordinate frame gizmo.
    void updateCoordinateFrame();
    void panByF(float dx, float dy);
    void zoomStepsAt(int steps, const QPointF& scenePos);
    // Multiply the current scale by `factor` (clamped to [kMinScale, kMaxScale]),
    // keeping the scene point under `scenePos` fixed. The shared core of both the
    // discrete wheel zoom (zoomStepsAt) and the continuous adjustZoomByFactor.
    void zoomByFactorAt(float factor, const QPointF& scenePos);
    bool isAxisAlignedView() const;
    void ensureDefaultSurface();
    void updateContentBounds();
    QPointF surfaceToScene(float surfX, float surfY, float wPx = 0.0f) const;
    cv::Vec2f sceneToSurface(const QPointF& scenePos) const;
    // Signed depth band along the surface normal that the view currently
    // displays; gates which volume points project onto a quad surface.
    void quadProjectDepthBand(float& depthLo, float& depthHi) const;
    struct PendingRenderJob {
        std::uint64_t requestId = 0;
        int fbW = 0;
        int fbH = 0;
        float surfacePtrX = 0.0f;
        float surfacePtrY = 0.0f;
        float scale = 1.0f;
        float zOff = 0.0f;
        int startLevel = 0;
        int overlayStartLevel = 0;
        int samplingMethod = 1;
        int overlaySamplingMethod = 0;
        CompositeRenderSettings compositeSettings;
        float windowLow = 0.0f;
        float windowHigh = 255.0f;
        std::shared_ptr<Surface> surf;
        std::string surfaceName;
        std::shared_ptr<const vc3d::ViewerLevels> levels;
        std::shared_ptr<Volume> overlayVolume;
        std::shared_ptr<const vc3d::ViewerLevels> overlayLevels;
        float overlayOpacity = 0.0f;
        std::string overlayColormapId;
        float overlayWindowLow = 0.0f;
        float overlayWindowHigh = 255.0f;
        bool overlayIntensityAsAlpha = false;
        OverlayCompositeSettings overlayComposite;
        std::uint64_t loadedDataEpoch = 0;
        std::uint64_t surfaceGeometryEpoch = 0;
        // Effective launch-time bounds. Null for disabled bounds and for view
        // types outside the supported plane/annotation scope.
        std::optional<Rect3D> focusBoundsBase;
        // The vertex grid of the QuadSurface shown (null for planes).
        std::shared_ptr<const vcr::grid::Surface> gridSurface;
        bool geometryDirty = false;
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
    // Everything of renderJobsEquivalentForDisplay except the loaded-data
    // epoch and gen-cache dirtiness: true when two jobs sample the same
    // pixels from the same source, so one frame's data can stand in for
    // missing chunks in the other.
    static bool renderJobsSameGeometry(const PendingRenderJob& a,
                                       const PendingRenderJob& b);
    struct RenderContext;
    struct RenderResult;
    static RenderResult renderFrame(RenderContext ctx);
    void finishRenderOnMainThread(std::shared_ptr<RenderResult> result);
    int renderStartLevel(bool preferSurfaceResolution = false) const;
    int overlayRenderStartLevel(bool preferSurfaceResolution = false) const;
    // Sync the volumetric camera gizmo's visibility/state with the current
    // composite settings and surface type.
    void updateCameraGizmo();
    // True when the volumetric composite is what renderFrame will draw for
    // the current surface (enabled + method volumetric + non-plane surface).
    bool volumetricCameraActive() const;
    // Screen-direction -> surface-UV-direction mapping of the volumetric
    // camera at w = 0 (identity when the mode is inactive):
    // M = Rz(-azimuth) * diag(1, 1/cos(tilt)). Pan/zoom deltas arrive in
    // screen space and must cross this to move the UV view center correctly.
    cv::Matx22f volumetricScreenToSurface() const;
    // Azimuth the compositor (and the w=0 screen<->surface mapping) should
    // apply: 0 when the slice-plane owner folds it into the plane basis
    // instead, the per-view camera azimuth otherwise.
    float volumetricEffectiveAzimuthDeg() const;
    // Exact w=0 screen<->surface mapping of the volumetric camera, including
    // perspective (a plane-to-screen homography; identity when the mode is
    // inactive). Both sides are in framebuffer pixels relative to the view
    // center / the surface pointer.
    // wPx = slab height above the (offset) surface, in screen pixels
    // (w layers * scale * wScale); 0 = the anchor plane the render pivots on.
    cv::Vec2f volumetricScreenPxToSurfacePx(const cv::Vec2f& screenRel, float wPx = 0.0f) const;
    cv::Vec2f volumetricSurfacePxToScreenPx(const cv::Vec2f& surfRel, float wPx = 0.0f) const;
    vc3d::volumetric::CameraParams volumetricPointMapCamera() const;
    float volumetricHalfSpan() const;
    std::optional<cv::Vec3f> cursorVolumePosition(const QPointF& scenePos) const;
    void refreshCursorPositionAt(const QPointF& scenePos);
    // projected=true draws the greyed-out variant used when a linked cursor
    // point lies off this pane's plane (e.g. after a normal-offset scroll).
    void updateCursorCrosshair(const QPointF& scenePos, bool projected = false);
    void updateLocalCursorCrosshair(const QPointF& scenePos);
    void updateLineAnnotationPlacementMarker(const QPointF& scenePos);
    void clearLineAnnotationPlacementMarker();
    bool handleMeasurementClick(const QPointF& scenePos, Qt::MouseButton button, Qt::KeyboardModifiers modifiers);
    void refreshMeasurementOverlay();
    void onSurfaceChangedImpl(const std::string& name, const std::shared_ptr<Surface>& surf, bool isEditUpdate);
    void updateFocusMarker(POI* poi = nullptr);
    void refreshSameWrapAnnotationOverlay();
    std::optional<std::pair<uint64_t, uint64_t>> pointAtScenePosition(const QPointF& scenePos);
    QRectF surfaceRectToSceneRect(const QRectF& surfRect) const;

    CState* _state = nullptr;
    QPointer<ViewerManager> _viewerManager;
    VCCollection* _pointCollection = nullptr;
    CVolumeViewerView* _view = nullptr;
    QGraphicsScene* _scene = nullptr;
    ViewerStatsBar* _statsBar = nullptr;
    ViewerStatsBar* _statsBarRight = nullptr;
    // Shared cache statistics are refreshed at most every 250 ms: updateStatusLabel
    // runs per mouse move.
    QElapsedTimer _sharedCacheStatsThrottle;
    QStringList _cachedSharedCacheItems;
    CameraGizmoWidget* _cameraGizmo = nullptr;
    // No per-viewer timers. ViewerManager's global clock only services
    // status maintenance; render requests submit immediately.
    bool _closing = false;
    // A render was requested while this viewer was hidden (minimized MDI subwindow /
    // background tab). We skip the work and set this; showEvent re-renders so the
    // viewer catches up on whatever went stale while invisible.
    bool _renderStaleWhileHidden = false;
    bool _suppressNextSurfaceEditRender = false;

    bool _volumetricAzimuthInSurface = false;
    // Last-seen frame of a displayed PlaneSurface: when the plane is rotated
    // in place (azimuth folding / up realignment) the world view center is
    // re-projected so the view spins about the screen center instead of the
    // plane origin.
    struct PlaneFrameSnapshot {
        bool valid = false;
        cv::Vec3f origin{0, 0, 0};
        cv::Vec3f normal{0, 0, 0};
        cv::Vec3f vx{0, 0, 0};
        cv::Vec3f vy{0, 0, 0};
    };
    PlaneFrameSnapshot _planeFrame;

    std::shared_ptr<Volume> _volume;
    std::weak_ptr<Surface> _surfWeak;
    std::shared_ptr<Surface> _defaultSurface;
    std::string _surfName;
    std::shared_ptr<const vc3d::ViewerLevels> _levels;
    // The background read of chunks frames lacked: one at a time; the newest
    // frame's request waits and replaces older waiting ones. `_loadSuperseded`
    // tells the running read that a newer request is waiting.
    bool _loadRunning = false;
    std::optional<vc3d::ViewerLoad> _pendingLoad;
    std::shared_ptr<std::atomic<bool>> _loadSuperseded;
    QString _loadError;

    QImage _framebuffer;
    std::atomic<bool> _renderWorkerBusy{false};
    std::optional<PendingRenderJob> _activeRenderJob;
    std::optional<PendingRenderJob> _pendingRenderJob;
    std::optional<PendingRenderJob> _displayedRenderJob;
    // Last presented render (sample values + coverage). The next render of
    // the same geometry reuses its pixels where chunks are missing so a
    // transient cache miss never blanks an already-shown region.
    std::shared_ptr<RenderResult> _lastRenderResult;
    std::uint64_t _renderRequestSerial = 0;
    std::uint64_t _loadedDataEpoch = 0;
    std::uint64_t _surfaceGeometryEpoch = 0;
    std::uint64_t _surfaceEditEpoch = 0;
    std::uint64_t _renderSerial = 0;
    // The shown QuadSurface's vertex grid for vcr::grid; dropped on every edit.
    std::shared_ptr<const vcr::grid::Surface> _gridSurface;
    // The frame's surface when it is `quad`, so cursor maths agrees with what is drawn.
    [[nodiscard]] std::shared_ptr<const vcr::grid::Surface> surfaceForPoints(const QuadSurface& quad) const;
    const QuadSurface* _gridSurfaceFor = nullptr;
    // The view's geometry changed: the next frame does not carry the last one's pixels.
    bool _geometryDirty = true;

    float _surfacePtrX = 0.0f;
    float _surfacePtrY = 0.0f;
    std::optional<float> _pinnedSurfacePtrY;
    float _scale = 1.0f;
    int _dsScaleIdx = 0;
    float _zOff = 0.0f;
    float _camSurfX = 0.0f;
    float _camSurfY = 0.0f;
    float _camScale = 1.0f;

    float _windowLow = 0.0f;
    float _windowHigh = 255.0f;
    std::shared_ptr<Volume> _overlayVolume;
    std::shared_ptr<const vc3d::ViewerLevels> _overlayLevels;
    float _overlayOpacity = 0.5f;
    std::string _overlayColormapId;
    int _overlaySamplingMethod = 0;
    float _overlayWindowLow = 0.0f;
    float _overlayWindowHigh = 255.0f;
    bool _overlayIntensityAsAlpha = false;
    int _overlayMaxDisplayedResolution = 0;
    OverlayCompositeSettings _overlayComposite;

    CompositeRenderSettings _compositeSettings;
    bool _resetViewOnSurfaceChange = true;
    bool _planeIntersectionLinesVisible = true;
    float _panSensitivity = 1.0f;
    float _zoomSensitivity = 1.0f;
    float _zScrollSensitivity = 1.0f;
    float _linkedCursorViewTolerance = 10.0f;
    int _samplingMethod = 1;
    int _maxDisplayedResolution = 0;
    bool _showSurfaceNormals = false;
    bool _showCoordinateFrame = true;
    float _normalArrowLengthScale = 1.0f;
    int _normalMaxArrows = 32;
    bool _surfaceOverlayEnabled = false;
    bool _initializedFirstSegmentationSurface = false;
    std::map<std::string, cv::Vec3b> _surfaceOverlays;
    std::uint64_t _surfaceOverlaysRevision = 0;
    float _surfaceOverlapThreshold = 5.0f;
    float _intersectionOpacity = 0.7f;
    float _intersectionThickness = 0.0f;
    std::set<std::string> _intersectTgts;

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
    // Cursor point mirrored in from another viewer; display-only fallback for
    // the status-bar position readout (never used for click/hit-test paths).
    std::optional<cv::Vec3f> _linkedCursorVolumePos;
    ShiftScrollOverride _shiftScrollOverride;

    std::vector<ViewerOverlayControllerBase::PathPrimitive> _drawingPaths;
    std::unordered_map<std::string, std::vector<QGraphicsItem*>> _overlayGroups;
    QGraphicsItem* _cursorCrosshair = nullptr;
    bool _cursorCrosshairProjected = false;
    bool _localCursorCrosshairSuppressed = false;
    QGraphicsEllipseItem* _lineAnnotationPlacementMarker = nullptr;
    bool _lineAnnotationPlacementPreviewEnabled = false;
    QGraphicsItem* _focusMarker = nullptr;
    bool _segmentationCursorMirroring = false;
    bool _linkedCursorAlwaysEnabled = false;
    bool _linkedCursorMirroringSuppressed = false;

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

    std::optional<vc::PointRef> _highlightedPoint;
    std::optional<vc::PointRef> _selectedPoint;

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
