#pragma once

#include <QWidget>
#include <QPointF>
#include <QRectF>
#include <QColor>
#include <QString>

#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>
#include <optional>

#include "VolumeViewerBase.hpp"
#include "vc/ui/VCCollection.hpp"
#include "CVolumeViewerView.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/cache/TieredChunkCache.hpp"
#include "vc/core/util/SurfacePatchIndex.hpp"
#include "vc/core/util/Slicing.hpp"

#include "tiled/TiledViewerCamera.hpp"
#include "tiled/TileScene.hpp"
#include "tiled/TileRenderer.hpp"
#include "tiled/TileRenderController.hpp"

class QGraphicsScene;
class QGraphicsItem;
class QGraphicsPixmapItem;
class QLabel;
struct POI;
class CState;
class ViewerManager;
class PlaneSurface;

// Tiled volume viewer: fixed-size canvas composed of a grid of tile
// QGraphicsPixmapItems. Navigation updates tile *contents* rather than
// scrolling a large scene.
//
// Phase 1: synchronous rendering. Later phases add async rendering,
// tile caching, and progressive resolution.
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
    void renderIntersections();
    void invalidateVis();
    void invalidateIntersect(const std::string& name = "");

    // --- Accessors ---
    std::string surfName() const { return _surfName; }
    std::shared_ptr<Volume> currentVolume() const { return _volume; }
    vc::cache::TieredChunkCache* chunkCachePtr() const override {
        return _volume ? _volume->tieredCache() : nullptr;
    }
    int datasetScaleIndex() const { return _camera.dsScaleIdx; }
    float datasetScaleFactor() const { return _camera.dsScale; }
    float getCurrentScale() const { return _camera.scale; }
    float dsScale() const { return _camera.dsScale; }
    float normalOffset() const { return _camera.zOff; }
    Surface* currentSurface() const;
    VCCollection* pointCollection() const { return _pointCollection; }
    uint64_t highlightedPointId() const { return _highlightedPointId; }
    uint64_t selectedPointId() const { return _selectedPointId; }
    uint64_t selectedCollectionId() const { return _selectedCollectionId; }
    bool isPointDragActive() const { return _draggedPointId != 0; }
    const std::vector<ViewerOverlayControllerBase::PathPrimitive>& drawingPaths() const { return _paths; }

    // --- Composite settings ---
    void setCompositeRenderSettings(const CompositeRenderSettings& settings);
    const CompositeRenderSettings& compositeRenderSettings() const { return _compositeSettings; }
    bool isCompositeEnabled() const { return _compositeSettings.enabled; }
    bool isPlaneCompositeEnabled() const { return _compositeSettings.planeEnabled; }
    int planeCompositeLayersFront() const { return _compositeSettings.planeLayersFront; }
    int planeCompositeLayersBehind() const { return _compositeSettings.planeLayersBehind; }
    bool postStretchValues() const { return _compositeSettings.postStretchValues; }
    bool postRemoveSmallComponents() const { return _compositeSettings.postRemoveSmallComponents; }
    int postMinComponentSize() const { return _compositeSettings.postMinComponentSize; }

    // --- Display settings ---
    void setResetViewOnSurfaceChange(bool reset);
    void setShowDirectionHints(bool on) { _showDirectionHints = on; updateAllOverlays(); }
    bool isShowDirectionHints() const { return _showDirectionHints; }
    void setShowSurfaceNormals(bool on) { _showSurfaceNormals = on; updateAllOverlays(); }
    bool isShowSurfaceNormals() const { return _showSurfaceNormals; }
    void setNormalArrowLengthScale(float scale) { _normalArrowLengthScale = scale; updateAllOverlays(); }
    float normalArrowLengthScale() const { return _normalArrowLengthScale; }
    void setNormalMaxArrows(int maxArrows) { _normalMaxArrows = maxArrows; updateAllOverlays(); }
    int normalMaxArrows() const { return _normalMaxArrows; }

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
    bool surfaceOverlayEnabled() const { return _surfaceOverlayEnabled; }
    void setSurfaceOverlays(const std::map<std::string, cv::Vec3b>& overlays);
    const std::map<std::string, cv::Vec3b>& surfaceOverlays() const { return _surfaceOverlays; }
    void setSurfaceOverlapThreshold(float threshold);
    float surfaceOverlapThreshold() const { return _surfaceOverlapThreshold; }

    // --- Intersection rendering ---
    void setIntersectionOpacity(float opacity);
    float intersectionOpacity() const { return _intersectionOpacity; }
    void setIntersectionThickness(float thickness);
    float intersectionThickness() const { return _intersectionThickness; }
    void setHighlightedSurfaceIds(const std::vector<std::string>& ids);
    void setSurfacePatchSamplingStride(int stride);
    int surfacePatchSamplingStride() const { return _surfacePatchSamplingStride; }

    // --- Overlay group management ---
    void setOverlayGroup(const std::string& key, const std::vector<QGraphicsItem*>& items);
    void clearOverlayGroup(const std::string& key);
    void clearAllOverlayGroups();

    void updateAllOverlays();
    void updateStatusLabel();
    void fitSurfaceInView();

    bool isWindowMinimized() const;
    bool eventFilter(QObject* watched, QEvent* event) override;

    // --- Coordinate transforms ---
    // Transform from volume (world) coordinates to canvas scene coordinates
    QPointF volumeToScene(const cv::Vec3f& vol_point);
    // Transform from canvas scene coordinates to volume (world) coordinates
    cv::Vec3f sceneToVolume(const QPointF& scenePoint) const;
    // Scene-to-volume coordinate conversion (returns position + normal)
    bool sceneToVolumePN(cv::Vec3f& p, cv::Vec3f& n, const QPointF& scenePos) const;
    QPointF lastScenePosition() const { return _lastScenePos; }

    // --- BBox tool ---
    void setBBoxMode(bool enabled);
    bool isBBoxMode() const { return _bboxMode; }
    QuadSurface* makeBBoxFilteredSurfaceFromSceneRect(const QRectF& sceneRect);
    auto selections() const -> std::vector<std::pair<QRectF, QColor>>;
    std::optional<QRectF> activeBBoxSceneRect() const { return _activeBBoxSceneRect; }
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
    void onSurfaceChanged(std::string name, std::shared_ptr<Surface> surf, bool isEditUpdate = false);
    void onPOIChanged(std::string name, POI* poi);
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
    void onSurfaceWillBeDeleted(std::string name, std::shared_ptr<Surface> surf);
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
    // Camera-based navigation (replaces scrollbar-based)
    void panBy(int dx, int dy);
    void zoomAt(float factor, const QPointF& widgetPos);
    void zoomStepsAt(int steps, const QPointF& scenePos);
    void setSliceOffset(float dz);

    // Build TileRenderParams for a given tile key
    TileRenderParams buildRenderParams(const WorldTileKey& wk) const;

    void markActiveSegmentationDirty();

    // Mark overlays dirty on the render controller AND notify external listeners
    void invalidateOverlays();

    // Coalesced overlay update: batches multiple overlaysUpdated() emissions
    // into a single deferred emission on the next event loop iteration.
    void scheduleOverlayUpdate();

    // Recompute and update the params hash on the render controller
    void updateParamsHash();

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

    // Submit all visible tiles to the render controller (async path)
    void submitRender();

    // Compute content extent in surface parameter space and rebuild the tile grid
    void rebuildContentGrid();

    // Called on the main thread when pinCoarsestLevel completes (background thread)
    void onPinComplete();

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
    TileRenderController* _renderController = nullptr;
    ContentBounds _contentBounds;

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
    bool _useFastInterpolation = false;

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
    bool _dirtyWhileMinimized = false;
    std::chrono::steady_clock::time_point _lastStatusUpdate;  // debounce status label
    bool _overlayUpdatePending = false;  // coalescing flag for scheduleOverlayUpdate()

    // --- Zoom limits ---
    float _contentMinScale = TiledViewerCamera::MIN_SCALE;  // dynamic minimum so content fills viewport

    // --- Remote volume pin progress ---
    std::atomic<int> _pinTotal{0};   // total chunks to pin at coarsest level
    int _pinReceived = 0;            // chunks received so far (main thread only)
    int _pinLevel = -1;              // pyramid level being pinned

    // --- Chunk-ready listener tracking ---
    vc::cache::TieredChunkCache::ChunkReadyCallbackId _chunkCbId = 0;
    bool _hadValidDataBounds = false;

    // --- Pan tracking ---
    // For tiled viewer, panning is tracked via delta signals from the view
    QPoint _lastPanPos;
    bool _isPanning = false;

    // Predictive prefetch state
    QPointF _lastPanScenePos;
};
