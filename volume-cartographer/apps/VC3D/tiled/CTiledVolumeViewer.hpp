#pragma once

#include <QWidget>
#include <QPointF>
#include <QRectF>
#include <QColor>

#include <atomic>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>
#include <optional>

#include <QFutureWatcher>
#include <QPainterPath>

#include "VolumeViewerBase.hpp"
#include "vc/ui/VCCollection.hpp"
#include "CVolumeViewerView.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/cache/TieredChunkCache.hpp"
#include "vc/core/util/SurfacePatchIndex.hpp"
#include "vc/core/util/Slicing.hpp"

#include "tiled/TiledViewerCamera.hpp"
#include "tiled/TileScene.hpp"

class QGraphicsScene;
class QGraphicsItem;
class QGraphicsPixmapItem;
class QLabel;
class QTimer;
struct POI;
class CState;
class ViewerManager;
class PlaneSurface;

// Tiled volume viewer: fixed-size canvas composed of a grid of tile
// QGraphicsPixmapItems. Navigation updates tile *contents* rather than
// scrolling a large scene.
//
// Tiled volume viewer with async rendering, tile caching, progressive
// resolution, and retained-layer transitions.
class CTiledVolumeViewer : public QWidget, public VolumeViewerBase
{
    Q_OBJECT

public:
    CTiledVolumeViewer(CState* state, ViewerManager* manager,
                       QWidget* parent = nullptr);
    ~CTiledVolumeViewer();

    // --- Data setup ---
    void setPointCollection(VCCollection* point_collection);
    void setSurface(const std::string& name);
    void setIntersects(const std::set<std::string>& set);

    // --- Rendering ---
    void renderVisible(bool force = false);
    void renderIntersections() override;
    void invalidateVis();
    void invalidateIntersect(const std::string& name = "") override;

    // --- Accessors ---
    TileScene* tileScene() const { return _tileScene; }
    std::string surfName() const override { return _surfName; }
    std::shared_ptr<Volume> currentVolume() const override { return _volume; }
    vc::cache::TieredChunkCache* chunkCachePtr() const override {
        return _volume ? _volume->tieredCache() : nullptr;
    }
    int datasetScaleIndex() const override { return _camera.dsScaleIdx; }
    float datasetScaleFactor() const override { return _camera.dsScale; }
    float getCurrentScale() const override { return _camera.scale; }
    float dsScale() const override { return _camera.dsScale; }
    float normalOffset() const override { return _camera.zOff; }
    Surface* currentSurface() const override;
    VCCollection* pointCollection() const override { return _pointCollection; }
    uint64_t highlightedPointId() const override { return _highlightedPointId; }
    uint64_t selectedPointId() const override { return _selectedPointId; }
    uint64_t selectedCollectionId() const override { return _selectedCollectionId; }
    bool isPointDragActive() const override { return _draggedPointId != 0; }
    const std::vector<ViewerOverlayControllerBase::PathPrimitive>& drawingPaths() const override { return _paths; }

    // --- Composite settings ---
    void setCompositeRenderSettings(const CompositeRenderSettings& settings);
    const CompositeRenderSettings& compositeRenderSettings() const override { return _compositeSettings; }
    bool isCompositeEnabled() const override { return _compositeSettings.enabled; }
    bool isPlaneCompositeEnabled() const override { return _compositeSettings.planeEnabled; }
    int planeCompositeLayersFront() const { return _compositeSettings.planeLayersFront; }
    int planeCompositeLayersBehind() const { return _compositeSettings.planeLayersBehind; }
    bool postStretchValues() const { return _compositeSettings.postStretchValues; }
    bool postRemoveSmallComponents() const { return _compositeSettings.postRemoveSmallComponents; }
    int postMinComponentSize() const { return _compositeSettings.postMinComponentSize; }

    // --- Display settings ---
    void setResetViewOnSurfaceChange(bool reset);
    void setShowDirectionHints(bool on) { _showDirectionHints = on; updateAllOverlays(); }
    bool isShowDirectionHints() const override { return _showDirectionHints; }
    void setShowSurfaceNormals(bool on) { _showSurfaceNormals = on; updateAllOverlays(); }
    bool isShowSurfaceNormals() const override { return _showSurfaceNormals; }
    void setNormalArrowLengthScale(float scale) { _normalArrowLengthScale = scale; updateAllOverlays(); }
    float normalArrowLengthScale() const override { return _normalArrowLengthScale; }
    void setNormalMaxArrows(int maxArrows) { _normalMaxArrows = maxArrows; updateAllOverlays(); }
    int normalMaxArrows() const override { return _normalMaxArrows; }

    // --- Surface offset ---
    void adjustSurfaceOffset(float dn);
    void resetSurfaceOffsets();

    // --- Window/level ---
    void setVolumeWindow(float low, float high);
    float volumeWindowLow() const { return _baseWindowLow; }
    float volumeWindowHigh() const { return _baseWindowHigh; }
    void setBaseColormap(const std::string& colormapId);
    const std::string& baseColormap() const { return _baseColormapId; }
    void setStretchValues(bool enabled);
    bool stretchValues() const { return _stretchValues; }

    // --- Overlay volume ---
    void setOverlayVolume(std::shared_ptr<Volume> volume);
    std::shared_ptr<Volume> overlayVolume() const { return _overlayVolume; }
    void setOverlayOpacity(float opacity);
    float overlayOpacity() const { return _overlayOpacity; }
    void setOverlayColormap(const std::string& colormapId);
    const std::string& overlayColormap() const { return _overlayColormapId; }
    void setOverlayThreshold(float threshold);
    float overlayThreshold() const { return _overlayWindowLow; }
    void setOverlayWindow(float low, float high);
    float overlayWindowLow() const { return _overlayWindowLow; }
    float overlayWindowHigh() const { return _overlayWindowHigh; }

    // --- Segmentation ---
    void setSegmentationEditActive(bool active);
    void setSegmentationCursorMirroring(bool enabled) { _mirrorCursorToSegmentation = enabled; }
    bool segmentationCursorMirroringEnabled() const { return _mirrorCursorToSegmentation; }

    const ActiveSegmentationHandle& activeSegmentationHandle() const override;

    // --- Surface overlays ---
    void setSurfaceOverlayEnabled(bool enabled);
    bool surfaceOverlayEnabled() const override { return _surfaceOverlayEnabled; }
    void setSurfaceOverlays(const std::map<std::string, cv::Vec3b>& overlays);
    const std::map<std::string, cv::Vec3b>& surfaceOverlays() const override { return _surfaceOverlays; }
    void setSurfaceOverlapThreshold(float threshold);
    float surfaceOverlapThreshold() const override { return _surfaceOverlapThreshold; }

    // --- Intersection rendering ---
    void setIntersectionOpacity(float opacity);
    float intersectionOpacity() const override { return _intersectionOpacity; }
    void setIntersectionThickness(float thickness);
    float intersectionThickness() const override { return _intersectionThickness; }
    void setHighlightedSurfaceIds(const std::vector<std::string>& ids);
    void setSurfacePatchSamplingStride(int stride);
    int surfacePatchSamplingStride() const override { return _surfacePatchSamplingStride; }

    // --- Overlay group management ---
    void setOverlayGroup(const std::string& key, const std::vector<QGraphicsItem*>& items) override;
    void clearOverlayGroup(const std::string& key) override;
    void clearAllOverlayGroups() override;

    void updateAllOverlays();
    void updateStatusLabel();
    void fitSurfaceInView();

    bool isWindowMinimized() const;
    bool eventFilter(QObject* watched, QEvent* event) override;

    // --- Coordinate transforms ---
    // Transform from volume (world) coordinates to canvas scene coordinates
    QPointF volumeToScene(const cv::Vec3f& vol_point) override;
    // Transform from canvas scene coordinates to volume (world) coordinates
    cv::Vec3f sceneToVolume(const QPointF& scenePoint) const override;
    // Scene-to-volume coordinate conversion (returns position + normal)
    bool sceneToVolumePN(cv::Vec3f& p, cv::Vec3f& n, const QPointF& scenePos) const;
    // Transform from canvas scene coordinates to surface parameter coordinates
    cv::Vec2f sceneToSurfaceCoords(const QPointF& scenePos) const;
    QPointF lastScenePosition() const { return _lastScenePos; }

    // --- BBox tool ---
    void setBBoxMode(bool enabled);
    bool isBBoxMode() const { return _bboxMode; }
    QuadSurface* makeBBoxFilteredSurfaceFromSceneRect(const QRectF& sceneRect);
    auto selections() const -> std::vector<std::pair<QRectF, QColor>> override;
    std::optional<QRectF> activeBBoxSceneRect() const override { return _activeBBoxSceneRect; }
    void clearSelections();

    // --- Misc ---
    void onDrawingModeActive(bool active, float brushSize = 3.0f, bool isSquare = false);

    // --- VolumeViewerBase interface ---
    CVolumeViewerView* graphicsView() const override { return fGraphicsView; }
    QObject* asQObject() override { return this; }
    QMetaObject::Connection connectOverlaysUpdated(
        QObject* receiver, const std::function<void()>& callback) override
    {
        return connect(this, &CTiledVolumeViewer::overlaysUpdated, receiver, callback);
    }

    // Graphics view accessor (for overlay controllers)
    CVolumeViewerView* fGraphicsView;

public slots:
    void OnVolumeChanged(std::shared_ptr<Volume> vol);
    void onVolumeClicked(QPointF scene_loc, Qt::MouseButton buttons, Qt::KeyboardModifiers modifiers);
    void onPanRelease(Qt::MouseButton buttons, Qt::KeyboardModifiers modifiers);
    void onPanStart(Qt::MouseButton buttons, Qt::KeyboardModifiers modifiers);
    void onCollectionSelected(uint64_t collectionId);
    void onSurfaceChanged(const std::string& name, const std::shared_ptr<Surface>& surf, bool isEditUpdate = false);
    void onPOIChanged(const std::string& name, POI* poi);
    void onScrolled();
    void onResized();
    void onZoom(int steps, QPointF scene_point, Qt::KeyboardModifiers modifiers);
    void adjustZoomByFactor(float factor);
    void onCursorMove(QPointF);
    void onPathsChanged(const QList<ViewerOverlayControllerBase::PathPrimitive>& paths);
    void onPointSelected(uint64_t pointId);
    void onMousePress(QPointF scene_loc, Qt::MouseButton button, Qt::KeyboardModifiers modifiers);
    void onMouseMove(QPointF scene_loc, Qt::MouseButtons buttons, Qt::KeyboardModifiers modifiers);
    void onMouseRelease(QPointF scene_loc, Qt::MouseButton button, Qt::KeyboardModifiers modifiers);
    void onVolumeClosing();
    void onSurfaceWillBeDeleted(const std::string& name, const std::shared_ptr<Surface>& surf);
    void onKeyPress(int key, Qt::KeyboardModifiers modifiers);
    void onKeyRelease(int key, Qt::KeyboardModifiers modifiers);

signals:
    void sendVolumeClicked(cv::Vec3f vol_loc, cv::Vec3f normal, Surface* surf,
                           Qt::MouseButton buttons, Qt::KeyboardModifiers modifiers);
    void sendZSliceChanged(int z_value);
    void sendMousePressVolume(cv::Vec3f vol_loc, cv::Vec3f normal,
                              Qt::MouseButton button, Qt::KeyboardModifiers modifiers);
    void sendMouseMoveVolume(cv::Vec3f vol_loc, Qt::MouseButtons buttons,
                             Qt::KeyboardModifiers modifiers);
    void sendMouseReleaseVolume(cv::Vec3f vol_loc, Qt::MouseButton button,
                                Qt::KeyboardModifiers modifiers);
    void sendCollectionSelected(uint64_t collectionId);
    void pointSelected(uint64_t pointId);
    void pointClicked(uint64_t pointId);
    void overlaysUpdated();
    void sendSegmentationRadiusWheel(int steps, QPointF scenePoint, cv::Vec3f worldPos);

private:
    // onCursorMove implementation with pre-locked surface to avoid redundant locks
    void onCursorMoveImpl(QPointF scene_loc, const std::shared_ptr<Surface>& surf);

    // Core intersection rendering (bypasses throttle)
    void renderIntersectionsCore();

    // Async intersection computation completion handler
    void onIntersectionComputeFinished();

    // Camera-based navigation (replaces scrollbar-based)
    void panBy(int dx, int dy);
    void panByF(float dx, float dy);
    void zoomAt(float factor, const QPointF& widgetPos);
    void zoomStepsAt(int steps, const QPointF& scenePos);
    void setSliceOffset(float dz);

    void markActiveSegmentationDirty();

    // Mark overlays dirty on the render controller AND notify external listeners
    void invalidateOverlays();

    // Coalesced overlay update: batches multiple overlaysUpdated() emissions
    // into a single deferred emission on the next event loop iteration.
    void scheduleOverlayUpdate();

    // Recompute dynamic minimum scale so content never appears smaller than viewport
    void updateContentMinScale();


    // Returns true for axis-aligned viewer slots (xy/xz/yz plane, seg xz/yz)
    bool isAxisAlignedView() const;

    // Clamp lo/hi bounding box to volume dataBounds; returns false if empty after clamping
    bool clampToDataBounds(cv::Vec3f& lo, cv::Vec3f& hi) const;

    // Compute world-space bounding box for prefetching (PlaneSurface path).
    // Returns false if the bbox is empty after clamping to data bounds.
    bool computePlanePrefetchBBox(PlaneSurface* plane, const QRectF& prefetchRect,
                                  cv::Vec3f& lo, cv::Vec3f& hi) const;

    // Compute world-space bounding box for prefetching (QuadSurface / generic path).
    // Returns false if the bbox is empty after clamping to data bounds.
    bool computeQuadPrefetchBBox(const std::shared_ptr<Surface>& surf,
                                 const QRectF& prefetchRect,
                                 cv::Vec3f& lo, cv::Vec3f& hi) const;

    // Coalesce render requests to max one per 16ms per viewer
    void scheduleRender();

    // Render the full viewport directly to the framebuffer
    void submitRender();

    // Compute content extent in surface parameter space and resize framebuffer
    void rebuildContentGrid();

    // Called when volume is ready for rendering
    void onVolumeReady();

    // Called when data bounds become valid after async coarsest-level load
    void onDataBoundsReady();

    // Center the viewport on the current surfacePtr
    void centerViewport();

    // Get the current viewport rect in scene coordinates
    QRectF viewportSceneRect() const;

    // --- Widget components ---
    QGraphicsScene* _scene = nullptr;
    TileScene* _tileScene = nullptr;
    TiledViewerCamera _camera;
    // Content extent in surface coordinates (for pan clamping)
    float _fullContentMinU = 0, _fullContentMaxU = 0;
    float _fullContentMinV = 0, _fullContentMaxV = 0;

    // --- Data ---
    std::shared_ptr<Volume> _volume;
    std::weak_ptr<Surface> _surfWeak;
    std::shared_ptr<Surface> _defaultSurface;  // keeps alive for surfaceless remote volumes
    std::string _surfName;
    CState* _state = nullptr;
    ViewerManager* _viewerManager = nullptr;
    VCCollection* _pointCollection = nullptr;

    // --- Rendering state ---
    CompositeRenderSettings _compositeSettings;
    uint64_t _surfaceContentVersion = 0;
    float _baseWindowLow = 0.0f;
    float _baseWindowHigh = 255.0f;
    bool _stretchValues = false;
    std::string _baseColormapId;

    // --- Overlay volume ---
    std::shared_ptr<Volume> _overlayVolume;
    float _overlayOpacity = 0.5f;
    std::string _overlayColormapId;
    float _overlayWindowLow = 0.0f;
    float _overlayWindowHigh = 255.0f;

    // --- Surface overlays ---
    bool _surfaceOverlayEnabled = false;
    std::map<std::string, cv::Vec3b> _surfaceOverlays;
    float _surfaceOverlapThreshold = 5.0f;

    // --- Overlay state (graphics items, intersection + overlay group bookkeeping) ---
    struct OverlayState {
        // Overlay groups managed by overlay controllers
        std::unordered_map<std::string, std::vector<QGraphicsItem*>> groups;

        // Intersection items per surface name
        std::unordered_map<std::string, std::vector<QGraphicsItem*>> intersectItems;

        // Slice visualisation items (cleared on invalidateVis)
        std::vector<QGraphicsItem*> sliceVisItems;

        // Cursor crosshair and center-of-view marker
        QGraphicsItem* cursor = nullptr;
        QGraphicsItem* centerMarker = nullptr;
    };
    OverlayState _ov;

    // --- Intersection rendering ---
    float _intersectionOpacity = 1.0f;
    float _intersectionThickness = 0.0f;
    int _surfacePatchSamplingStride = 1;
    std::set<std::string> _intersectTgts = {"visible_segmentation"};
    std::unordered_set<std::string> _highlightedSurfaceIds;
    std::unordered_map<std::string, size_t> _surfaceColorAssignments;
    size_t _nextColorIndex = 0;

    // --- Point highlight cache (invalidated on camera change) ---
    std::unordered_map<uint64_t, QPointF> _pointSceneCache;

    // --- Interaction state ---
    uint64_t _highlightedPointId = 0;
    uint64_t _selectedPointId = 0;
    uint64_t _draggedPointId = 0;
    uint64_t _selectedCollectionId = 0;
    std::vector<ViewerOverlayControllerBase::PathPrimitive> _paths;
    bool _drawingModeActive = false;
    float _brushSize = 3.0f;
    bool _brushIsSquare = false;
    bool _resetViewOnSurfaceChange = true;
    bool _showDirectionHints = true;
    bool _showSurfaceNormals = false;
    float _normalArrowLengthScale = 1.0f;
    int _normalMaxArrows = 32;
    bool _segmentationEditActive = false;
    bool _mirrorCursorToSegmentation = false;
    QPointF _lastScenePos;

    // --- BBox tool ---
    bool _bboxMode = false;
    QPointF _bboxStart;
    std::optional<QRectF> _activeBBoxSceneRect;
    struct Selection { QRectF surfRect; QColor color; };
    std::vector<Selection> _selections;

    // --- Active segmentation handle ---
    mutable ActiveSegmentationHandle _activeSegHandle;
    mutable bool _activeSegHandleDirty = true;

    // --- Cached QuadSurface bounding box (for fitSurfaceInView) ---
    struct CachedSurfBBox {
        int colMin = 0, colMax = -1, rowMin = 0, rowMax = -1;
        const void* surfPtr = nullptr;  // identity check
    };
    mutable CachedSurfBBox _surfBBoxCache;

    // --- Status ---
    QLabel* _lbl = nullptr;
    QTimer* _renderTimer = nullptr;
    bool _renderPending = false;
    bool _dirtyWhileMinimized = false;
    bool _overlayUpdatePending = false;  // coalescing flag for scheduleOverlayUpdate()
    QTimer* _intersectionThrottleTimer = nullptr;  // coalesces renderIntersections calls
    bool _intersectionsDirty = false;

    // --- Intersection cache (skip recomputation when inputs unchanged) ---
    struct IntersectionCache {
        cv::Vec3f planeOrigin{0, 0, 0};
        cv::Vec3f planeNormal{0, 0, 0};
        cv::Rect planeRoi;
        std::unordered_set<SurfacePatchIndex::SurfacePtr> targets;
        float opacity = -1.0f;
        float thickness = -1.0f;
        float cameraScale = -1.0f;  // for adaptive sub-pixel culling
        bool valid = false;
    };
    IntersectionCache _intersectionCache;

    // --- Async intersection computation ---
    struct IntersectionPathEntry {
        QPainterPath path;
        QColor color;
        int zValue;
    };
    QFutureWatcher<std::vector<IntersectionPathEntry>> _intersectionFutureWatcher;
    bool _intersectionPending = false;   // true while background computation is running
    bool _intersectionRerunNeeded = false; // dirty flag: re-run after current finishes
    // Snapshot of cache key stored when async work is dispatched, applied on completion.
    IntersectionCache _pendingCacheKey;

    // --- Zoom limits ---
    float _contentMinScale = TiledViewerCamera::MIN_SCALE;  // dynamic minimum so content fills viewport
    float _navSpeed = 1.0f;  // navigation speed multiplier (zoom, pan, scroll)

    // --- Chunk-ready listener tracking ---
    vc::cache::TieredChunkCache::ChunkReadyCallbackId _chunkCbId = 0;
    bool _hadValidDataBounds = false;

    // --- Focus marker position (in surface coords) ---
    // Tracks where the focus point is on the surface, independent of camera pan.
    float _focusSurfacePos[2] = {0.0f, 0.0f};

    // --- Pan tracking ---
    // For tiled viewer, panning is tracked via delta signals from the view
    QPoint _lastPanPos;
    QPointF _lastPanSceneF;  // sub-pixel pan tracking in scene coords
    bool _isPanning = false;

    // Predictive prefetch state
    QPointF _lastPanScenePos;
    float _lastZOff = 0.0f;       // previous z offset for velocity-based z-prefetch
    float _zVelocity = 0.0f;      // last z-scroll direction (+/- or 0)

    // --- Visual zoom transform ---
    float _zoomBaseScale = 0.5f;       // camera.scale at start of current zoom

};
