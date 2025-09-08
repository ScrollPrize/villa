#pragma once

#include <QtWidgets>

#include <set>
#include <unordered_map>
#include "PathData.hpp"
#include "vc/ui/VCCollection.hpp"
#include "COutlinedTextItem.hpp"
#include "CSurfaceCollection.hpp"
#include "CVolumeViewerView.hpp"
#include "vc/core/types/Volume.hpp"


class CVolumeViewer : public QWidget
{
    Q_OBJECT

public:
    CVolumeViewer(CSurfaceCollection *col, QWidget* parent = 0);
    ~CVolumeViewer(void);

    void setCache(ChunkCache *cache);
    void setPointCollection(VCCollection* point_collection);
    void setSurface(const std::string &name);
    void renderVisible(bool force = false);
    void renderIntersections();
    cv::Mat render_area(const cv::Rect &roi);
    void invalidateVis();
    void invalidateIntersect(const std::string &name = "");
    
    std::set<std::string> intersects();
    void setIntersects(const std::set<std::string> &set);
    std::string surfName() { return _surf_name; };
    void recalcScales();
    void renderPaths();
    
    // Composite view methods
    void setCompositeEnabled(bool enabled);
    void setCompositeLayers(int layers);
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

    // Direction hints toggle
    void setShowDirectionHints(bool on) { _showDirectionHints = on; updateAllOverlays(); }
    bool isShowDirectionHints() const { return _showDirectionHints; }

    void fitSurfaceInView();
    void updateAllOverlays();
    
    // Direction hints for vc_grow_seg_from_segments flip_x visualization
    void renderDirectionHints();
    void renderDirectionStepMarkers();

    // Generic overlay group management (ad-hoc helper for reuse)
    void setOverlayGroup(const std::string& key, const std::vector<QGraphicsItem*>& items);
    void clearOverlayGroup(const std::string& key);

    // Get current scale for coordinate transformation
    float getCurrentScale() const { return _scale; }
    // Transform scene coordinates to volume coordinates
    cv::Vec3f sceneToVolume(const QPointF& scenePoint) const;

    // BBox drawing mode for segmentation view
    void setBBoxMode(bool enabled);
    bool isBBoxMode() const { return _bboxMode; }
    // Create a new QuadSurface with only points inside the given scene-rect
    QuadSurface* makeBBoxFilteredSurfaceFromSceneRect(const QRectF& sceneRect);
    // Current stored selections (scene-space rects with colors)
    auto selections() const -> std::vector<std::pair<QRectF, QColor>>;
    void clearSelections();
    void updateSelectionGraphics();
    
    CVolumeViewerView* fGraphicsView;

public slots:
    void OnVolumeChanged(std::shared_ptr<Volume> vol);
    void onVolumeClicked(QPointF scene_loc,Qt::MouseButton buttons, Qt::KeyboardModifiers modifiers);
    void onPanRelease(Qt::MouseButton buttons, Qt::KeyboardModifiers modifiers);
    void onPanStart(Qt::MouseButton buttons, Qt::KeyboardModifiers modifiers);
    void onCollectionSelected(uint64_t collectionId);
    void onCollectionChanged(uint64_t collectionId);
    void onSurfaceChanged(std::string name, Surface *surf);
    void onPOIChanged(std::string name, POI *poi);
    void onIntersectionChanged(std::string a, std::string b, Intersection *intersection);
    void onScrolled();
    void onResized();
    void onZoom(int steps, QPointF scene_point, Qt::KeyboardModifiers modifiers);
    void onCursorMove(QPointF);
    void onPointAdded(const ColPoint& point);
    void onPointChanged(const ColPoint& point);
    void onPointRemoved(uint64_t pointId);
    void onPathsChanged(const QList<PathData>& paths);
    void onPointSelected(uint64_t pointId);

    // Mouse event handlers for drawing (transform coordinates)
    void onMousePress(QPointF scene_loc, Qt::MouseButton button, Qt::KeyboardModifiers modifiers);
    void onMouseMove(QPointF scene_loc, Qt::MouseButtons buttons, Qt::KeyboardModifiers modifiers);
    void onMouseRelease(QPointF scene_loc, Qt::MouseButton button, Qt::KeyboardModifiers modifiers);
    void onVolumeClosing(); // Clear surface pointers when volume is closing
    void onKeyRelease(int key, Qt::KeyboardModifiers modifiers);
    void onDrawingModeActive(bool active, float brushSize = 3.0f, bool isSquare = false);

signals:
    void SendSignalSliceShift(int shift, int axis);
    void SendSignalStatusMessageAvailable(QString text, int timeout);
    void sendVolumeClicked(cv::Vec3f vol_loc, cv::Vec3f normal, Surface *surf, Qt::MouseButton buttons, Qt::KeyboardModifiers modifiers);
    void sendShiftNormal(cv::Vec3f step);
    void sendZSliceChanged(int z_value);
    
    // Mouse event signals with transformed volume coordinates
    void sendMousePressVolume(cv::Vec3f vol_loc, cv::Vec3f normal, Qt::MouseButton button, Qt::KeyboardModifiers modifiers);
    void sendMouseMoveVolume(cv::Vec3f vol_loc, Qt::MouseButtons buttons, Qt::KeyboardModifiers modifiers);
    void sendMouseReleaseVolume(cv::Vec3f vol_loc, Qt::MouseButton button, Qt::KeyboardModifiers modifiers);
    void sendCollectionSelected(uint64_t collectionId);
    void pointSelected(uint64_t pointId);
    void pointClicked(uint64_t pointId);
    // (kept free for potential future signals)

protected:
    void ScaleImage(double nFactor);
    void CenterOn(const QPointF& point);
    QPointF volumeToScene(const cv::Vec3f& vol_point);
    void refreshPointPositions();
    void renderOrUpdatePoint(const ColPoint& point);

    void performDeferredUpdates();

protected:
    // widget components
    QGraphicsScene* fScene;

    // data
    QImage* fImgQImage;
    bool fSkipImageFormatConv;

    QGraphicsPixmapItem* fBaseImageItem;
    
    std::shared_ptr<Volume> volume = nullptr;
    Surface *_surf = nullptr;
    cv::Vec3f _ptr = cv::Vec3f(0,0,0);
    cv::Vec2f _vis_center = {0,0};
    std::string _surf_name;
    int axis = 0;
    int loc[3] = {0,0,0};
    
    ChunkCache *cache = nullptr;
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
    
    bool _slice_vis_valid = false;
    std::vector<QGraphicsItem*> slice_vis_items; 
    
    std::set<std::string> _intersect_tgts = {"visible_segmentation"};
    std::unordered_map<std::string,std::vector<QGraphicsItem*>> _intersect_items;
    Intersection *_ignore_intersect_change = nullptr;
    
    CSurfaceCollection *_surf_col = nullptr;
    
    VCCollection* _point_collection = nullptr;
    struct PointGraphics {
        QGraphicsEllipseItem* circle;
        COutlinedTextItem* text;
    };
    std::unordered_map<uint64_t, PointGraphics> _points_items;
    
    // Point interaction state
    uint64_t _highlighted_point_id = 0;
    uint64_t _selected_point_id = 0;
    uint64_t _dragged_point_id = 0;
    uint64_t _selected_collection_id = 0;
    uint64_t _current_shift_collection_id = 0;
    bool _new_shift_group_required = true;
    
    QList<PathData> _paths;
    std::vector<QGraphicsItem*> _path_items;
    
    // Generic overlay groups; each key owns its items' lifetime
    std::unordered_map<std::string, std::vector<QGraphicsItem*>> _overlay_groups;
    
    // Drawing mode state
    bool _drawingModeActive = false;
    float _brushSize = 3.0f;
    bool _brushIsSquare = false;
    bool _resetViewOnSurfaceChange = true;
    bool _showDirectionHints = true;

    int _downscale_override = 0;  // 0=auto, 1=2x, 2=4x, 3=8x, 4=16x, 5=32x
    QTimer* _overlayUpdateTimer;

    // BBox tool state
    bool _bboxMode = false;
    QPointF _bboxStart;
    QGraphicsRectItem* _bboxRectItem = nullptr;
    struct Selection { QRectF surfRect; QColor color; QGraphicsRectItem* item; };
    std::vector<Selection> _selections;

    bool _useFastInterpolation;


};  // class CVolumeViewer
