#pragma once

#include <QWidget>
#include <QPointF>
#include <QImage>

#include <memory>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <optional>
#include <unordered_map>

#include <opencv2/core.hpp>
#include <QFutureWatcher>
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

// Minimal adaptive volume viewer. Renders a PlaneSurface using
// samplePlaneAdaptiveARGB32 directly to a viewport-sized framebuffer.
// Diagnostic replacement for CTiledVolumeViewer — no overlays, no
// composite, no tiling, no prefetching.
class CAdaptiveVolumeViewer : public QWidget, public VolumeViewerBase
{
    Q_OBJECT

public:
    CAdaptiveVolumeViewer(CState* state, ViewerManager* manager,
                          QWidget* parent = nullptr);
    ~CAdaptiveVolumeViewer();

    // --- Data setup ---
    void setPointCollection(VCCollection* pc) { _pointCollection = pc; }
    void setSurface(const std::string& name);
    void setIntersects(const std::set<std::string>&) {}

    // --- Rendering ---
    void renderVisible(bool force = false);
    void renderIntersections() override {}
    void invalidateVis() {}
    void invalidateIntersect(const std::string& = "") override {}

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
    void setCompositeRenderSettings(const CompositeRenderSettings& s) {
        _compositeSettings = s;
        submitRender();
    }
    const CompositeRenderSettings& compositeRenderSettings() const override { return _compositeSettings; }
    bool isCompositeEnabled() const override { return _compositeSettings.enabled; }
    bool isPlaneCompositeEnabled() const override { return _compositeSettings.planeEnabled; }

    void setVolumeWindow(float low, float high);
    float volumeWindowLow() const { return _windowLow; }
    float volumeWindowHigh() const { return _windowHigh; }
    void setBaseColormap(const std::string& id) { _baseColormapId = id; submitRender(); }
    void setStretchValues(bool) { submitRender(); }

    // --- Display stubs ---
    void setResetViewOnSurfaceChange(bool v) { _resetViewOnSurfaceChange = v; }
    void setShowDirectionHints(bool) {}
    bool isShowDirectionHints() const override { return false; }
    void setShowSurfaceNormals(bool) {}
    bool isShowSurfaceNormals() const override { return false; }
    float normalArrowLengthScale() const override { return 1.0f; }
    int normalMaxArrows() const override { return 0; }
    void setNormalArrowLengthScale(float) {}
    void setNormalMaxArrows(int) {}

    // --- Overlay volume stubs ---
    void setOverlayVolume(std::shared_ptr<Volume>) {}
    void setOverlayOpacity(float) {}
    void setOverlayColormap(const std::string&) {}
    void setOverlayThreshold(float) {}
    void setOverlayWindow(float, float) {}

    // --- Segmentation stubs ---
    void setSegmentationEditActive(bool) {}
    void setSegmentationCursorMirroring(bool) {}
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

    // --- BBox stubs ---
    auto selections() const -> std::vector<std::pair<QRectF, QColor>> override { return {}; }
    std::optional<QRectF> activeBBoxSceneRect() const override { return std::nullopt; }

    // --- Intersection stubs ---
    float intersectionOpacity() const override { return 1.0f; }
    float intersectionThickness() const override { return 0.0f; }
    int surfacePatchSamplingStride() const override { return 1; }
    void setIntersectionOpacity(float) {}
    void setIntersectionThickness(float) {}
    void setHighlightedSurfaceIds(const std::vector<std::string>&) {}
    void setSurfacePatchSamplingStride(int) {}

    // --- Surface overlay stubs ---
    bool surfaceOverlayEnabled() const override { return false; }
    const std::map<std::string, cv::Vec3b>& surfaceOverlays() const override {
        static std::map<std::string, cv::Vec3b> empty;
        return empty;
    }
    float surfaceOverlapThreshold() const override { return 5.0f; }
    void setSurfaceOverlayEnabled(bool) {}
    void setSurfaceOverlays(const std::map<std::string, cv::Vec3b>&) {}
    void setSurfaceOverlapThreshold(float) {}

    // --- Coordinate transforms ---
    QPointF volumeToScene(const cv::Vec3f& vol_point) override;
    cv::Vec3f sceneToVolume(const QPointF& scenePoint) const override;
    cv::Vec2f sceneToSurfaceCoords(const QPointF& scenePos) const;
    bool sceneToVolumePN(cv::Vec3f& p, cv::Vec3f& n, const QPointF& scenePos) const;
    QPointF lastScenePosition() const { return _lastScenePos; }

    // --- Surface offset ---
    void adjustSurfaceOffset(float dn);
    void resetSurfaceOffsets();

    // --- BBox tool stubs ---
    void setBBoxMode(bool) {}
    bool isBBoxMode() const { return false; }
    QuadSurface* makeBBoxFilteredSurfaceFromSceneRect(const QRectF&) { return nullptr; }
    void clearSelections() {}

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
    void fitSurfaceInView();
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
    void onCursorMove(QPointF scenePos);
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
    void adjustZoomByFactor(float factor);

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
    void submitRender();
    void scheduleRender();

    // Framebuffer coordinate conversions
    QPointF surfaceToScene(float surfX, float surfY) const;
    cv::Vec2f sceneToSurface(const QPointF& scenePos) const;

    void panByF(float dx, float dy);
    void zoomStepsAt(int steps, const QPointF& scenePos);

    bool isAxisAlignedView() const;

    // --- Qt widgets ---
    CVolumeViewerView* _view = nullptr;
    QGraphicsScene* _scene = nullptr;
    QLabel* _lbl = nullptr;
    QTimer* _renderTimer = nullptr;
    bool _renderPending = false;

    // --- Framebuffer ---
    QImage _framebuffer;
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
    float _navSpeed = 1.0f;
    float _panSensitivity = 1.0f;
    float _zoomSensitivity = 1.0f;
    float _zScrollSensitivity = 1.0f;
    vc::Sampling _samplingMethod = vc::Sampling::Trilinear;

    // --- Content bounds for pan clamping ---
    float _contentMinU = 0, _contentMaxU = 0;
    float _contentMinV = 0, _contentMaxV = 0;

    // --- Pan state ---
    bool _isPanning = false;
    QPointF _lastPanSceneF;
    QPointF _lastScenePos;

    // --- Overlay groups (stored for VolumeViewerBase interface) ---
    std::unordered_map<std::string, std::vector<QGraphicsItem*>> _overlayGroups;

    // --- Chunk-ready listener ---
    vc::cache::BlockPipeline::ChunkReadyCallbackId _chunkCbId = 0;
    bool _hadValidDataBounds = false;
    bool _dirtyWhileMinimized = false;
};
