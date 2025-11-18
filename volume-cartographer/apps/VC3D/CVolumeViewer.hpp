#pragma once

#include <QtWidgets>

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

class QImage;
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
    cv::Mat_<uint8_t> renderCompositeForSurface(QuadSurface* surface, cv::Size outputSize);
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
    void setResetViewOnSurfaceChange(bool reset);
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
    void onSurfaceChanged(std::string name, Surface *surf);
    void onPOIChanged(std::string name, POI *poi);
    void onIntersectionChanged(std::string a, std::string b, Intersection *intersection);
    void onScrolled();
    void onResized();
    void onZoom(int steps, QPointF scene_point, Qt::KeyboardModifiers modifiers);
    void onCursorMove(QPointF);
    void onPathsChanged(const QList<ViewerOverlayControllerBase::PathPrimitive>& paths);
    void onPointSelected(uint64_t pointId);

    // Mouse event handlers for drawing (transform coordinates)
    void onMousePress(QPointF scene_loc, Qt::MouseButton button, Qt::KeyboardModifiers modifiers);
    void onMouseMove(QPointF scene_loc, Qt::MouseButtons buttons, Qt::KeyboardModifiers modifiers);
    void onMouseRelease(QPointF scene_loc, Qt::MouseButton button, Qt::KeyboardModifiers modifiers);
    void onVolumeClosing(); // Clear surface pointers when volume is closing
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
    Surface *_surf = nullptr;
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

    float _z_off = 0.0;
    
    // Composite view settings
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
    
    QGraphicsItem *_center_marker = nullptr;
    QGraphicsItem *_cursor = nullptr;
    
    std::vector<QGraphicsItem*> slice_vis_items; 

    std::set<std::string> _intersect_tgts = {"visible_segmentation"};
    std::unordered_map<std::string,std::vector<QGraphicsItem*>> _intersect_items;
    Intersection *_ignore_intersect_change = nullptr;
    bool _autoRefocusOnOffscreenIntersections = true;
    bool _hasLastPlaneOrigin = false;
    cv::Vec3f _lastPlaneOrigin = {0.0f, 0.0f, 0.0f};
    
    CSurfaceCollection *_surf_col = nullptr;
    ViewerManager* _viewerManager = nullptr;
    
    VCCollection* _point_collection = nullptr;

    float _intersectionOpacity{1.0f};
    float _intersectionThickness{0.0f};
    std::unordered_set<std::string> _highlightedSurfaceIds;
    
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
};  // class CVolumeViewer
