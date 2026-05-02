#pragma once

#include <QElapsedTimer>
#include <QImage>
#include <QPointF>
#include <QWidget>

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <opencv2/core.hpp>

#include "CVolumeViewerView.hpp"
#include "VolumeViewerBase.hpp"
#include "vc/core/render/ChunkedPlaneSampler.hpp"
#include "vc/core/render/IChunkedArray.hpp"
#include "vc/core/types/Sampling.hpp"
#include "vc/core/util/Compositing.hpp"
#include "vc/core/util/SurfacePatchIndex.hpp"

class CState;
class QGraphicsItem;
class QGraphicsScene;
class QLabel;
class QTimer;
struct POI;
class Surface;
class ViewerManager;
class VCCollection;
class Volume;

namespace vc::render { class ChunkCache; }

class CChunkedVolumeViewer : public QWidget, public VolumeViewerBase
{
    Q_OBJECT

public:
    CChunkedVolumeViewer(CState* state, ViewerManager* manager, QWidget* parent = nullptr);
    ~CChunkedVolumeViewer() override;

    void setPointCollection(VCCollection* pc) { _pointCollection = pc; }
    void setSurface(const std::string& name) override;
    void setIntersects(const std::set<std::string>& names) override { _intersectTgts = names; renderIntersections(); }
    void renderVisible(bool force = false) override;
    void requestRender() override { scheduleRender(); }
    void invalidateVis() override {}
    void centerOnVolumePoint(const cv::Vec3f& point, bool forceRender = false) override;
    void centerOnSurfacePoint(const cv::Vec2f& point, bool forceRender = false) override;
    void adjustSurfaceOffset(float delta) override;
    void resetSurfaceOffsets() override;
    void fitSurfaceInView() override;

    std::string surfName() const override { return _surfName; }
    std::shared_ptr<Volume> currentVolume() const override { return _volume; }
    vc::cache::BlockPipeline* chunkCachePtr() const override { return nullptr; }
    float getCurrentScale() const override { return _scale; }
    float dsScale() const override { return _dsScale; }
    float normalOffset() const override { return _zOff; }
    int datasetScaleIndex() const override { return _dsScaleIdx; }
    float datasetScaleFactor() const override { return _dsScale; }
    Surface* currentSurface() const override;
    VCCollection* pointCollection() const override { return _pointCollection; }

    void setCompositeRenderSettings(const CompositeRenderSettings& s) override { _compositeSettings = s; scheduleRender(); }
    const CompositeRenderSettings& compositeRenderSettings() const override { return _compositeSettings; }
    bool isCompositeEnabled() const override { return _compositeSettings.enabled && !streamingCompositeUnsupported(); }
    bool isPlaneCompositeEnabled() const override { return _compositeSettings.planeEnabled && !streamingCompositeUnsupported(); }

    void setVolumeWindow(float low, float high) override;
    void setBaseColormap(const std::string& id) override { _baseColormapId = id; scheduleRender(); }
    void setStretchValues(bool) { scheduleRender(); }
    void setResetViewOnSurfaceChange(bool v) override { _resetViewOnSurfaceChange = v; }

    void setShowDirectionHints(bool on) override { _showDirectionHints = on; emit overlaysUpdated(); }
    bool isShowDirectionHints() const override { return _showDirectionHints; }
    void setShowSurfaceNormals(bool on) override { _showSurfaceNormals = on; emit overlaysUpdated(); }
    bool isShowSurfaceNormals() const override { return _showSurfaceNormals; }
    float normalArrowLengthScale() const override { return _normalArrowLengthScale; }
    int normalMaxArrows() const override { return _normalMaxArrows; }
    void setNormalArrowLengthScale(float scale) override { _normalArrowLengthScale = scale; emit overlaysUpdated(); }
    void setNormalMaxArrows(int maxArrows) override { _normalMaxArrows = maxArrows; emit overlaysUpdated(); }

    void setOverlayVolume(std::shared_ptr<Volume> volume) override;
    void setOverlayOpacity(float opacity) override;
    void setOverlayColormap(const std::string& colormapId) override;
    void setOverlayThreshold(float threshold) override;
    void setOverlayWindow(float low, float high) override;

    void setSegmentationEditActive(bool) override {}
    void setSegmentationCursorMirroring(bool) override {}
    const ActiveSegmentationHandle& activeSegmentationHandle() const override;

    uint64_t highlightedPointId() const override { return 0; }
    uint64_t selectedPointId() const override { return 0; }
    uint64_t selectedCollectionId() const override { return 0; }
    bool isPointDragActive() const override { return false; }
    const std::vector<ViewerOverlayControllerBase::PathPrimitive>& drawingPaths() const override;

    void setOverlayGroup(const std::string& key, const std::vector<QGraphicsItem*>& items) override;
    void clearOverlayGroup(const std::string& key) override;
    void clearAllOverlayGroups() override;

    std::vector<std::pair<QRectF, QColor>> selections() const override;
    std::optional<QRectF> activeBBoxSceneRect() const override;
    void setBBoxMode(bool enabled) override;
    QuadSurface* makeBBoxFilteredSurfaceFromSceneRect(const QRectF& sceneRect) override;
    void clearSelections() override;

    void renderIntersections() override;
    void invalidateIntersect(const std::string& = "") override;
    float intersectionOpacity() const override { return _intersectionOpacity; }
    float intersectionThickness() const override { return _intersectionThickness; }
    int surfacePatchSamplingStride() const override { return _surfacePatchSamplingStride; }
    void setIntersectionOpacity(float v) override { _intersectionOpacity = v; renderIntersections(); }
    void setIntersectionThickness(float v) override { _intersectionThickness = v; renderIntersections(); }
    void setHighlightedSurfaceIds(const std::vector<std::string>& ids) override;
    void setSurfacePatchSamplingStride(int s) override { _surfacePatchSamplingStride = s; invalidateIntersect(); renderIntersections(); }

    bool surfaceOverlayEnabled() const override { return _surfaceOverlayEnabled; }
    const std::map<std::string, cv::Vec3b>& surfaceOverlays() const override;
    float surfaceOverlapThreshold() const override { return _surfaceOverlapThreshold; }
    void setSurfaceOverlayEnabled(bool enabled) override { _surfaceOverlayEnabled = enabled; emit overlaysUpdated(); }
    void setSurfaceOverlays(const std::map<std::string, cv::Vec3b>& overlays) override { _surfaceOverlays = overlays; emit overlaysUpdated(); }
    void setSurfaceOverlapThreshold(float threshold) override { _surfaceOverlapThreshold = std::max(0.0f, threshold); emit overlaysUpdated(); }

    QPointF volumeToScene(const cv::Vec3f& volPoint) override;
    cv::Vec3f sceneToVolume(const QPointF& scenePoint) const override;
    cv::Vec2f sceneToSurfaceCoords(const QPointF& scenePos) const override;
    QPointF surfaceCoordsToScene(float surfX, float surfY) const override { return surfaceToScene(surfX, surfY); }
    void setLinkedCursorVolumePoint(const std::optional<cv::Vec3f>& point) override;
    QPointF lastScenePosition() const override { return _lastScenePos; }

    CVolumeViewerView* graphicsView() const override { return _view; }
    QObject* asQObject() override { return this; }
    QMetaObject::Connection connectOverlaysUpdated(
        QObject* receiver, const std::function<void()>& callback) override {
        return connect(this, &CChunkedVolumeViewer::overlaysUpdated, receiver, callback);
    }

    void reloadPerfSettings() override;

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
    void onKeyRelease(int, Qt::KeyboardModifiers) {}
    void onScrolled() {}
    void onPathsChanged(const QList<ViewerOverlayControllerBase::PathPrimitive>& paths);
    void onCollectionSelected(uint64_t) {}
    void onPointSelected(uint64_t) {}
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
    void sendCollectionSelected(uint64_t collectionId);
    void pointSelected(uint64_t pointId);
    void pointClicked(uint64_t pointId);
    void overlaysUpdated();
    void sendSegmentationRadiusWheel(int steps, QPointF scenePoint, cv::Vec3f worldPos);

private:
    void scheduleRender();
    void submitRender();
    void updateStatusLabel();
    void rebuildChunkArray();
    void syncCameraTransform();
    bool renderInteractiveAxisAlignedSlicePreview();
    void updateInteractivePreviewFromStableFrame(float newSurfX, float newSurfY, float newScale);
    bool shouldRefreshInteractivePreview();
    void resizeFramebuffer();
    void recalcPyramidLevel();
    void panByF(float dx, float dy);
    void zoomStepsAt(int steps, const QPointF& scenePos);
    bool isAxisAlignedView() const;
    void ensureDefaultSurface();
    void updateContentBounds();
    QPointF surfaceToScene(float surfX, float surfY) const;
    cv::Vec2f sceneToSurface(const QPointF& scenePos) const;
    void renderOverlayVolumeForPlane(const cv::Vec3f& origin,
                                     const cv::Vec3f& vxStep,
                                     const cv::Vec3f& vyStep,
                                     int startLevel,
                                     const vc::render::ChunkedPlaneSampler::Options& options,
                                     cv::Mat_<uint8_t>& overlayValues,
                                     cv::Mat_<uint8_t>& overlayCoverage);
    void renderOverlayVolumeForCoords(const cv::Mat_<cv::Vec3f>& coords,
                                      int startLevel,
                                      const vc::render::ChunkedPlaneSampler::Options& options,
                                      cv::Mat_<uint8_t>& overlayValues,
                                      cv::Mat_<uint8_t>& overlayCoverage);
    void samplePlaneIntoValues(const cv::Vec3f& origin,
                               const cv::Vec3f& vxStep,
                               const cv::Vec3f& vyStep,
                               const cv::Vec3f& normal,
                               int startLevel,
                               const vc::render::ChunkedPlaneSampler::Options& options,
                               cv::Mat_<uint8_t>& values,
                               cv::Mat_<uint8_t>& coverage);
    void sampleCoordsIntoValues(const cv::Mat_<cv::Vec3f>& coords,
                                const cv::Mat_<cv::Vec3f>& normals,
                                int startLevel,
                                const vc::render::ChunkedPlaneSampler::Options& options,
                                cv::Mat_<uint8_t>& values,
                                cv::Mat_<uint8_t>& coverage);
    void markInteractiveMotion(double motionPx);
    int renderStartLevel(bool preferSurfaceResolution = false) const;
    int genericPreviewDownsampleFactor() const;
    bool streamingCompositeUnsupported() const;
    void updateFocusMarker(POI* poi = nullptr);
    void clearIntersectionItems();
    void updateIntersectionPreviewTransform();
    void renderFlattenedIntersections(const std::shared_ptr<Surface>& surf);
    QRectF surfaceRectToSceneRect(const QRectF& surfRect) const;

    CState* _state = nullptr;
    ViewerManager* _viewerManager = nullptr;
    VCCollection* _pointCollection = nullptr;
    CVolumeViewerView* _view = nullptr;
    QGraphicsScene* _scene = nullptr;
    QLabel* _lbl = nullptr;
    QTimer* _renderTimer = nullptr;
    QTimer* _settleRenderTimer = nullptr;
    bool _renderPending = false;
    bool _interactivePreview = false;
    QElapsedTimer _interactionClock;
    qint64 _lastInteractionMs = -1;
    qint64 _lastInteractivePreviewMs = -1;
    double _interactionSpeedPxPerSec = 0.0;

    std::shared_ptr<Volume> _volume;
    std::weak_ptr<Surface> _surfWeak;
    std::shared_ptr<Surface> _defaultSurface;
    std::string _surfName;
    std::unique_ptr<vc::render::ChunkCache> _chunkArray;
    vc::render::IChunkedArray::ChunkReadyCallbackId _chunkCbId = 0;

    QImage _framebuffer;
    QImage _stableFramebuffer;
    float _stableSurfX = 0.0f;
    float _stableSurfY = 0.0f;
    float _stableScale = 1.0f;
    bool _stableFramebufferValid = false;
    cv::Mat_<uint8_t> _values;
    cv::Mat_<uint8_t> _coverage;
    cv::Mat_<cv::Vec3f> _genCoords;
    cv::Mat_<cv::Vec3f> _genNormals;
    bool _genCacheDirty = true;
    Surface* _genCacheSurfKey = nullptr;
    int _genCacheFbW = 0;
    int _genCacheFbH = 0;
    float _genCacheScale = 0.0f;
    cv::Vec3f _genCacheOffset{0, 0, 0};
    float _genCacheZOff = 0.0f;
    cv::Vec3f _genCacheZOffDir{0, 0, 0};

    float _surfacePtrX = 0.0f;
    float _surfacePtrY = 0.0f;
    float _scale = 1.0f;
    float _dsScale = 1.0f;
    int _dsScaleIdx = 0;
    float _zOff = 0.0f;
    float _camSurfX = 0.0f;
    float _camSurfY = 0.0f;
    float _camScale = 1.0f;
    cv::Vec3f _zOffWorldDir{0, 0, 0};

    float _windowLow = 0.0f;
    float _windowHigh = 255.0f;
    std::string _baseColormapId;
    std::shared_ptr<Volume> _overlayVolume;
    std::unique_ptr<vc::render::ChunkCache> _overlayChunkArray;
    vc::render::IChunkedArray::ChunkReadyCallbackId _overlayChunkCbId = 0;
    float _overlayOpacity = 0.5f;
    std::string _overlayColormapId;
    float _overlayWindowLow = 0.0f;
    float _overlayWindowHigh = 255.0f;

    CompositeRenderSettings _compositeSettings;
    bool _resetViewOnSurfaceChange = true;
    float _panSensitivity = 1.0f;
    float _zoomSensitivity = 1.0f;
    float _zScrollSensitivity = 1.0f;
    vc::Sampling _samplingMethod = vc::Sampling::Trilinear;
    bool _showDirectionHints = true;
    bool _showSurfaceNormals = false;
    float _normalArrowLengthScale = 1.0f;
    int _normalMaxArrows = 32;
    bool _surfaceOverlayEnabled = false;
    std::map<std::string, cv::Vec3b> _surfaceOverlays;
    float _surfaceOverlapThreshold = 5.0f;
    float _intersectionOpacity = 0.7f;
    float _intersectionThickness = 0.0f;
    int _surfacePatchSamplingStride = 2;
    std::set<std::string> _intersectTgts;
    std::unordered_set<std::string> _highlightedSurfaceIds;
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

    std::vector<ViewerOverlayControllerBase::PathPrimitive> _drawingPaths;
    std::unordered_map<std::string, std::vector<QGraphicsItem*>> _overlayGroups;
    QGraphicsItem* _cursorCrosshair = nullptr;
    QGraphicsItem* _focusMarker = nullptr;

    bool _bboxMode = false;
    QPointF _bboxStart;
    std::optional<QRectF> _activeBBoxSurfRect;
    struct Selection {
        QRectF surfRect;
        QColor color;
    };
    std::vector<Selection> _selections;
};
