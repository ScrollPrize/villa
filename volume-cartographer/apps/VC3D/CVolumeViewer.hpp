#pragma once

#include <QWidget>
#include <QPointF>
#include <QRectF>
#include <QColor>
#include <QString>
#include <QList>
#include <QImage>

#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <optional>
#include "overlays/ViewerOverlayControllerBase.hpp"
#include "vc/ui/VCCollection.hpp"
#include "CSurfaceCollection.hpp"
#include "CVolumeViewerView.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/util/SurfacePatchIndex.hpp"
#include "vc/core/util/ChunkCache.hpp"
#include "vc/core/util/Slicing.hpp"

class QGraphicsScene;
class QGraphicsItem;
class QGraphicsPixmapItem;
class QLabel;
class QTimer;
class ViewerManager;


class CVolumeViewer : public QWidget
{
    Q_OBJECT

public:
    CVolumeViewer(CSurfaceCollection *col, ViewerManager* manager, QWidget* parent = 0);
    ~CVolumeViewer(void);

    void setCache(ChunkCache<uint8_t> *cache);
    void setPointCollection(VCCollection* point_collection);
    void setSurface(const std::string &name);
    void renderVisible(bool force = false);
    void renderIntersections();
    cv::Mat render_area(const cv::Rect &roi);
    cv::Mat_<uint8_t> render_composite(const cv::Rect &roi);
    cv::Mat_<uint8_t> render_composite_plane(const cv::Rect &roi, const cv::Mat_<cv::Vec3f> &coords, const cv::Vec3f &planeNormal);
    cv::Mat_<uint8_t> renderCompositeForSurface(std::shared_ptr<QuadSurface> surface, cv::Size outputSize);
    void invalidateVis();
    void invalidateIntersect(const std::string &name = "");
    
    void setIntersects(const std::set<std::string> &set);
    std::string surfName() const { return _surf_name; };
    void recalcScales();
    
    // Composite view methods
    void setCompositeEnabled(bool enabled);
    void setCompositeLayersInFront(int layers);
    void setCompositeLayersBehind(int layers);
    void setCompositeMethod(const std::string& method);
    void setCompositeAlphaMin(int value);
    void setCompositeAlphaMax(int value);
    void setCompositeAlphaThreshold(int value);
    void setCompositeMaterial(int value);
    void setCompositeReverseDirection(bool reverse);
    void setCompositeBLExtinction(float value);
    void setCompositeBLEmission(float value);
    void setCompositeBLAmbient(float value);
    void setLightingEnabled(bool enabled);
    void setLightAzimuth(float degrees);
    void setLightElevation(float degrees);
    void setLightDiffuse(float value);
    void setLightAmbient(float value);
    void setUseVolumeGradients(bool enabled);
    void setIsoCutoff(int value);
    void setResetViewOnSurfaceChange(bool reset);

    // Plane composite view methods (for XY/XZ/YZ plane viewers)
    void setPlaneCompositeEnabled(bool enabled);
    void setPlaneCompositeLayers(int front, int behind);
    bool isPlaneCompositeEnabled() const { return _plane_composite_enabled; }
    int planeCompositeLayersFront() const { return _plane_composite_layers_front; }
    int planeCompositeLayersBehind() const { return _plane_composite_layers_behind; }

    // Postprocessing settings
    void setPostStretchValues(bool enabled);
    bool postStretchValues() const { return _postStretchValues; }
    void setPostRemoveSmallComponents(bool enabled);
    bool postRemoveSmallComponents() const { return _postRemoveSmallComponents; }
    void setPostMinComponentSize(int size);
    int postMinComponentSize() const { return _postMinComponentSize; }
    bool isCompositeEnabled() const { return _composite_enabled; }
    std::shared_ptr<Volume> currentVolume() const { return volume; }
    ChunkCache<uint8_t>* chunkCachePtr() const { return cache; }
    int datasetScaleIndex() const { return _ds_sd_idx; }
    float datasetScaleFactor() const { return _ds_scale; }
    VCCollection* pointCollection() const { return _point_collection; }
    uint64_t highlightedPointId() const { return _highlighted_point_id; }
    uint64_t selectedPointId() const { return _selected_point_id; }
    uint64_t selectedCollectionId() const { return _selected_collection_id; }
    bool isPointDragActive() const { return _dragged_point_id != 0; }
    const std::vector<ViewerOverlayControllerBase::PathPrimitive>& drawingPaths() const { return _paths; }

    // Direction hints toggle
    void setShowDirectionHints(bool on) { _showDirectionHints = on; updateAllOverlays(); }
    bool isShowDirectionHints() const { return _showDirectionHints; }

    // Surface normals visualization toggle
    void setShowSurfaceNormals(bool on) { _showSurfaceNormals = on; updateAllOverlays(); }
    bool isShowSurfaceNormals() const { return _showSurfaceNormals; }

    // Surface normals arrow length scale (1.0 = default)
    void setNormalArrowLengthScale(float scale) { _normalArrowLengthScale = scale; updateAllOverlays(); }
    float normalArrowLengthScale() const { return _normalArrowLengthScale; }

    // Surface normals max arrows per axis
    void setNormalMaxArrows(int maxArrows) { _normalMaxArrows = maxArrows; updateAllOverlays(); }
    int normalMaxArrows() const { return _normalMaxArrows; }

    // Surface-relative offset controls (normal direction only)
    void adjustSurfaceOffset(float dn);
    void resetSurfaceOffsets();
    float normalOffset() const { return _z_off; }

    void updateStatusLabel();

    void setSegmentationEditActive(bool active);

    void fitSurfaceInView();
    void updateAllOverlays();
    
    // Generic overlay group management (ad-hoc helper for reuse)
    void setOverlayGroup(const std::string& key, const std::vector<QGraphicsItem*>& items);
    void clearOverlayGroup(const std::string& key);
    void clearAllOverlayGroups();

    // Get current scale for coordinate transformation
    float getCurrentScale() const { return _scale; }
    // Transform scene coordinates to volume coordinates
    cv::Vec3f sceneToVolume(const QPointF& scenePoint) const;
    QPointF volumePointToScene(const cv::Vec3f& vol_point) { return volumeToScene(vol_point); }
    // Get the last known scene position (for coordinate lookups)
    QPointF lastScenePosition() const { return _lastScenePos; }
    // Get the dataset scale factor for scene→surface coordinate conversion
    float dsScale() const { return _ds_scale; }
    Surface* currentSurface() const;

    // BBox drawing mode for segmentation view
    void setBBoxMode(bool enabled);
    bool isBBoxMode() const { return _bboxMode; }
    // Create a new QuadSurface with only points inside the given scene-rect
    QuadSurface* makeBBoxFilteredSurfaceFromSceneRect(const QRectF& sceneRect);
    // Current stored selections (scene-space rects with colors)
    auto selections() const -> std::vector<std::pair<QRectF, QColor>>;
    std::optional<QRectF> activeBBoxSceneRect() const { return _activeBBoxSceneRect; }
    void clearSelections();

    void setIntersectionOpacity(float opacity);
    float intersectionOpacity() const { return _intersectionOpacity; }
    void setIntersectionThickness(float thickness);
    float intersectionThickness() const { return _intersectionThickness; }
    void setHighlightedSurfaceIds(const std::vector<std::string>& ids);
    void setSurfacePatchSamplingStride(int stride);
    int surfacePatchSamplingStride() const { return _surfacePatchSamplingStride; }

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

    void setSegmentationCursorMirroring(bool enabled) { _mirrorCursorToSegmentation = enabled; }
    bool segmentationCursorMirroringEnabled() const { return _mirrorCursorToSegmentation; }

    void setVolumeWindow(float low, float high);
    float volumeWindowLow() const { return _baseWindowLow; }
    float volumeWindowHigh() const { return _baseWindowHigh; }

    struct ActiveSegmentationHandle {
        QuadSurface* surface{nullptr};
        std::string slotName;
        QColor accentColor;
        bool viewerIsSegmentationView{false};

        bool valid() const { return surface != nullptr; }
        explicit operator bool() const { return valid(); }
        void reset()
        {
            surface = nullptr;
            slotName.clear();
            accentColor = QColor();
            viewerIsSegmentationView = false;
        }
    };

    const ActiveSegmentationHandle& activeSegmentationHandle() const;

    void setBaseColormap(const std::string& colormapId);
    const std::string& baseColormap() const { return _baseColormapId; }
    void setStretchValues(bool enabled);
    bool stretchValues() const { return _stretchValues; }

    void setSurfaceOverlayEnabled(bool enabled);
    bool surfaceOverlayEnabled() const { return _surfaceOverlayEnabled; }
    void setSurfaceOverlay(const std::string& surfaceName);
    const std::string& surfaceOverlay() const { return _surfaceOverlayName; }
    void setSurfaceOverlapThreshold(float threshold);
    float surfaceOverlapThreshold() const { return _surfaceOverlapThreshold; }

    struct OverlayColormapEntry {
        QString label;
        std::string id;
    };
    static const std::vector<OverlayColormapEntry>& overlayColormapEntries();
    
    CVolumeViewerView* fGraphicsView;

public slots:
    void OnVolumeChanged(std::shared_ptr<Volume> vol);
    void onVolumeClicked(QPointF scene_loc,Qt::MouseButton buttons, Qt::KeyboardModifiers modifiers);
    void onPanRelease(Qt::MouseButton buttons, Qt::KeyboardModifiers modifiers);
    void onPanStart(Qt::MouseButton buttons, Qt::KeyboardModifiers modifiers);
    void onCollectionSelected(uint64_t collectionId);
    void onSurfaceChanged(std::string name, std::shared_ptr<Surface> surf, bool isEditUpdate = false);
    void onPOIChanged(std::string name, POI *poi);
    void onScrolled();
    void onResized();
    void onZoom(int steps, QPointF scene_point, Qt::KeyboardModifiers modifiers);
    void adjustZoomByFactor(float factor);  // Adjust zoom by multiplicative factor (e.g., 1.15 for +15%)
    void onCursorMove(QPointF);
    void onPathsChanged(const QList<ViewerOverlayControllerBase::PathPrimitive>& paths);
    void onPointSelected(uint64_t pointId);

    // Mouse event handlers for drawing (transform coordinates)
    void onMousePress(QPointF scene_loc, Qt::MouseButton button, Qt::KeyboardModifiers modifiers);
    void onMouseMove(QPointF scene_loc, Qt::MouseButtons buttons, Qt::KeyboardModifiers modifiers);
    void onMouseRelease(QPointF scene_loc, Qt::MouseButton button, Qt::KeyboardModifiers modifiers);
    void onVolumeClosing(); // Clear surface pointers when volume is closing
    void onSurfaceWillBeDeleted(std::string name, std::shared_ptr<Surface> surf); // Clear references before surface deletion
    void onKeyRelease(int key, Qt::KeyboardModifiers modifiers);
    void onDrawingModeActive(bool active, float brushSize = 3.0f, bool isSquare = false);

signals:
    void sendVolumeClicked(cv::Vec3f vol_loc, cv::Vec3f normal, Surface *surf, Qt::MouseButton buttons, Qt::KeyboardModifiers modifiers);
    void sendZSliceChanged(int z_value);
    
    // Mouse event signals with transformed volume coordinates
    void sendMousePressVolume(cv::Vec3f vol_loc, cv::Vec3f normal, Qt::MouseButton button, Qt::KeyboardModifiers modifiers);
    void sendMouseMoveVolume(cv::Vec3f vol_loc, Qt::MouseButtons buttons, Qt::KeyboardModifiers modifiers);
    void sendMouseReleaseVolume(cv::Vec3f vol_loc, Qt::MouseButton button, Qt::KeyboardModifiers modifiers);
    void sendCollectionSelected(uint64_t collectionId);
    void pointSelected(uint64_t pointId);
    void pointClicked(uint64_t pointId);
    void overlaysUpdated();
    void sendSegmentationRadiusWheel(int steps, QPointF scenePoint, cv::Vec3f worldPos);
    // (kept free for potential future signals)

protected:
    QPointF volumeToScene(const cv::Vec3f& vol_point);

protected:
    // widget components
    QGraphicsScene* fScene;

    // data
    bool fSkipImageFormatConv;

    QGraphicsPixmapItem* fBaseImageItem;

    std::shared_ptr<Volume> volume = nullptr;
    std::weak_ptr<Surface> _surf_weak;  // Non-owning reference to current surface
    cv::Vec3f _ptr = cv::Vec3f(0,0,0);
    cv::Vec2f _vis_center = {0,0};
    std::string _surf_name;
    
    ChunkCache<uint8_t> *cache = nullptr;
    QRect curr_img_area = {0,0,1000,1000};
    float _scale = 0.5;
    float _scene_scale = 1.0;
    float _ds_scale = 0.5;
    int _ds_sd_idx = 1;
    float _max_scale = 1;
    float _min_scale = 1;

    QLabel *_lbl = nullptr;

    float _z_off = 0.0;  // Offset along surface normal (perpendicular to surface)
    QPointF _lastScenePos;  // Last known scene position for grid coordinate lookups

    // Composite view settings (for segmentation/QuadSurface)
    bool _composite_enabled = false;
    int _composite_layers = 7;
    int _composite_layers_front = 8;
    int _composite_layers_behind = 0;
    std::string _composite_method = "max";
    int _composite_alpha_min = 170;
    int _composite_alpha_max = 220;
    int _composite_alpha_threshold = 9950;
    int _composite_material = 230;
    bool _composite_reverse_direction = false;
    float _composite_bl_extinction = 1.5f;
    float _composite_bl_emission = 1.5f;
    float _composite_bl_ambient = 0.1f;
    bool _lighting_enabled = false;
    float _light_azimuth = 45.0f;
    float _light_elevation = 45.0f;
    float _light_diffuse = 0.7f;
    float _light_ambient = 0.3f;
    bool _use_volume_gradients = false;
    int _iso_cutoff = 0;

    // Plane composite view settings (for XY/XZ/YZ plane viewers)
    // These share the same composite method/parameters as segmentation,
    // but have separate layer counts and enable flag
    bool _plane_composite_enabled = false;
    int _plane_composite_layers_front = 4;
    int _plane_composite_layers_behind = 4;
    
    QGraphicsItem *_center_marker = nullptr;
    QGraphicsItem *_cursor = nullptr;
    
    std::vector<QGraphicsItem*> slice_vis_items; 

    std::set<std::string> _intersect_tgts = {"visible_segmentation"};
    std::unordered_map<std::string, SurfacePatchIndex::SurfacePtr> _cachedIntersectSurfaces;
    std::unordered_map<std::string,std::vector<QGraphicsItem*>> _intersect_items;
    std::unordered_map<std::string, std::vector<IntersectionLine>> _cachedIntersectionLines;
    float _cachedIntersectionScale = 0.0f;  // Scale used when caching intersection lines
    // Reusable buffers to avoid per-frame allocations
    std::vector<SurfacePatchIndex::TriangleCandidate> _triangleCandidates;
    std::unordered_map<SurfacePatchIndex::SurfacePtr, std::vector<size_t>> _trianglesBySurface;
    bool _autoRefocusOnOffscreenIntersections = true;
    bool _hasLastPlaneOrigin = false;
    cv::Vec3f _lastPlaneOrigin = {0.0f, 0.0f, 0.0f};
    
    CSurfaceCollection *_surf_col = nullptr;
    ViewerManager* _viewerManager = nullptr;
    
    VCCollection* _point_collection = nullptr;

    float _intersectionOpacity{1.0f};
    float _intersectionThickness{0.0f};
    std::unordered_set<std::string> _highlightedSurfaceIds;

    // Persistent color assignments for intersection rendering (up to 500 surfaces)
    std::unordered_map<std::string, size_t> _surfaceColorAssignments;
    size_t _nextColorIndex{0};
    
    // Point interaction state
    uint64_t _highlighted_point_id = 0;
    uint64_t _selected_point_id = 0;
    uint64_t _dragged_point_id = 0;
    uint64_t _selected_collection_id = 0;
    
    std::vector<ViewerOverlayControllerBase::PathPrimitive> _paths;
    
    // Generic overlay groups; each key owns its items' lifetime
    std::unordered_map<std::string, std::vector<QGraphicsItem*>> _overlay_groups;
    
    // Drawing mode state
    bool _drawingModeActive = false;
    float _brushSize = 3.0f;
    bool _brushIsSquare = false;
    bool _resetViewOnSurfaceChange = true;
    bool _showDirectionHints = true;
    bool _showSurfaceNormals = false;
    float _normalArrowLengthScale = 1.0f;
    int _normalMaxArrows = 32;
    bool _segmentationEditActive = false;
    bool _suppressFocusRecentering = false;

    int _downscale_override = 0;  // 0=auto, 1=2x, 2=4x, 3=8x, 4=16x, 5=32x
    QTimer* _overlayUpdateTimer;

    // BBox tool state
    bool _bboxMode = false;
    QPointF _bboxStart;
    std::optional<QRectF> _activeBBoxSceneRect;
    struct Selection { QRectF surfRect; QColor color; };
    std::vector<Selection> _selections;

    bool _useFastInterpolation;

    std::shared_ptr<Volume> _overlayVolume;
    float _overlayOpacity{0.5f};
    std::string _overlayColormapId;
    float _overlayWindowLow{0.0f};
    float _overlayWindowHigh{255.0f};
    float _baseWindowLow{0.0f};
    float _baseWindowHigh{255.0f};
    bool _mirrorCursorToSegmentation{false};
    bool _overlayImageValid{false};
    QImage _overlayImage;

    int _surfacePatchSamplingStride{1};

    void markActiveSegmentationDirty();
    mutable ActiveSegmentationHandle _activeSegHandle;
    mutable bool _activeSegHandleDirty{true};


    std::string _baseColormapId;
    bool _stretchValues{false};
    bool _surfaceOverlayEnabled{false};
    std::string _surfaceOverlayName;
    float _surfaceOverlapThreshold{5.0f};

    // Postprocessing settings
    bool _postStretchValues{false};
    bool _postRemoveSmallComponents{false};
    int _postMinComponentSize{50};

    // Fast composite rendering cache - no mutex, specialized for composite
    FastCompositeCache _fastCompositeCache;

    // Cached normals for composite rendering - invalidated on surface/ptr change
    cv::Mat_<cv::Vec3f> _cachedNormals;
    cv::Mat_<cv::Vec3f> _cachedBaseCoords;
    cv::Mat_<cv::Vec3f> _coordsWorkBuffer;  // Reusable buffer for z_off-adjusted coords
    cv::Size _cachedNormalsSize;
    float _cachedNormalsScale{0.0f};
    cv::Vec3f _cachedNormalsPtr{0, 0, 0};
    float _cachedNormalsZOff{0.0f};
    std::weak_ptr<Surface> _cachedNormalsSurf;

    // Cached volume gradients for PBR lighting - separate surface tracking
    cv::Mat_<cv::Vec3f> _cachedNativeVolumeGradients;
    std::weak_ptr<Surface> _cachedGradientsSurf;

};  // class CVolumeViewer
