#include <algorithm>
#include <cmath>
#include <optional>
#include <cstdlib>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <QGuiApplication>
#include <QSettings>
#include <QVBoxLayout>
#include <QTimer>
#include <QLabel>
#include <QPainter>
#include <QPainterPath>
#include <QScopedValueRollback>
#include <QGraphicsView>
#include <QGraphicsScene>
#include <QGraphicsItem>
#include <QGraphicsPixmapItem>
#include <QGraphicsEllipseItem>

#include "CVolumeViewer.hpp"
#include "vc/ui/UDataManipulateUtils.hpp"

#include "VolumeViewerCmaps.hpp"
#include "VCSettings.hpp"
#include "ViewerManager.hpp"

#include "CVolumeViewerView.hpp"
#include "CSurfaceCollection.hpp"
#include "vc/ui/VCCollection.hpp"

#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/Geometry.hpp"

using qga = QGuiApplication;

using PathPrimitive = ViewerOverlayControllerBase::PathPrimitive;
using PathBrushShape = ViewerOverlayControllerBase::PathBrushShape;

constexpr double ZOOM_FACTOR = 1.05;
constexpr auto COLOR_CURSOR =  Qt::cyan;
constexpr float MIN_ZOOM = 0.03125f;
constexpr float MAX_ZOOM = 4.0f;

#define COLOR_SEG_YZ Qt::yellow
#define COLOR_SEG_XZ Qt::red
#define COLOR_SEG_XY QColor(255, 140, 0)

const CVolumeViewer::ActiveSegmentationHandle& CVolumeViewer::activeSegmentationHandle() const
{
    if (!_activeSegHandleDirty) {
        return _activeSegHandle;
    }

    ActiveSegmentationHandle handle;
    handle.slotName = "segmentation";
    handle.viewerIsSegmentationView = (_surf_name == "segmentation");
    handle.accentColor =
        (_surf_name == "seg yz"   ? COLOR_SEG_YZ
         : _surf_name == "seg xz" ? COLOR_SEG_XZ
                                   : COLOR_SEG_XY);
    if (_surf_col) {
        handle.surface = dynamic_cast<QuadSurface*>(_surf_col->surface(handle.slotName));
    }
    if (!handle.surface) {
        handle.slotName.clear();
    }

    _activeSegHandle = handle;
    _activeSegHandleDirty = false;
    return _activeSegHandle;
}

void CVolumeViewer::markActiveSegmentationDirty()
{
    _activeSegHandleDirty = true;
    _activeSegHandle.reset();
}

CVolumeViewer::CVolumeViewer(CSurfaceCollection *col, ViewerManager* manager, QWidget* parent)
    : QWidget(parent)
    , fGraphicsView(nullptr)
    , fBaseImageItem(nullptr)
    , _surf_col(col)
    , _viewerManager(manager)
    , _highlighted_point_id(0)
    , _selected_point_id(0)
    , _dragged_point_id(0)
{
    // Create graphics view
    fGraphicsView = new CVolumeViewerView(this);
    
    fGraphicsView->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
    fGraphicsView->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
    
    fGraphicsView->setTransformationAnchor(QGraphicsView::NoAnchor);
    
    fGraphicsView->setRenderHint(QPainter::Antialiasing);
    // setFocusProxy(fGraphicsView);
    connect(fGraphicsView, &CVolumeViewerView::sendScrolled, this, &CVolumeViewer::onScrolled);
    connect(fGraphicsView, &CVolumeViewerView::sendVolumeClicked, this, &CVolumeViewer::onVolumeClicked);
    connect(fGraphicsView, &CVolumeViewerView::sendZoom, this, &CVolumeViewer::onZoom);
    connect(fGraphicsView, &CVolumeViewerView::sendResized, this, &CVolumeViewer::onResized);
    connect(fGraphicsView, &CVolumeViewerView::sendCursorMove, this, &CVolumeViewer::onCursorMove);
    connect(fGraphicsView, &CVolumeViewerView::sendPanRelease, this, &CVolumeViewer::onPanRelease);
    connect(fGraphicsView, &CVolumeViewerView::sendPanStart, this, &CVolumeViewer::onPanStart);
    connect(fGraphicsView, &CVolumeViewerView::sendMousePress, this, &CVolumeViewer::onMousePress);
    connect(fGraphicsView, &CVolumeViewerView::sendMouseMove, this, &CVolumeViewer::onMouseMove);
    connect(fGraphicsView, &CVolumeViewerView::sendMouseRelease, this, &CVolumeViewer::onMouseRelease);
    connect(fGraphicsView, &CVolumeViewerView::sendKeyRelease, this, &CVolumeViewer::onKeyRelease);

    // Create graphics scene
    fScene = new QGraphicsScene({-2500,-2500,5000,5000}, this);

    // Set the scene
    fGraphicsView->setScene(fScene);

    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    // fCenterOnZoomEnabled = settings.value("viewer/center_on_zoom", false).toInt() != 0;
    // fScrollSpeed = settings.value("viewer/scroll_speed", false).toInt();
    fSkipImageFormatConv = settings.value("perf/chkSkipImageFormatConvExp", false).toBool();
    _downscale_override = settings.value("perf/downscale_override", 0).toInt();
    _useFastInterpolation = settings.value("perf/fast_interpolation", false).toBool();
    if (_useFastInterpolation) {
        std::cout << "using nearest neighbor interpolation" << std::endl;
    }
    QVBoxLayout* aWidgetLayout = new QVBoxLayout;
    aWidgetLayout->addWidget(fGraphicsView);

    setLayout(aWidgetLayout);

    _overlayUpdateTimer = new QTimer(this);
    _overlayUpdateTimer->setSingleShot(true);
    _overlayUpdateTimer->setInterval(50);
    connect(_overlayUpdateTimer, &QTimer::timeout, this, &CVolumeViewer::updateAllOverlays);

    _lbl = new QLabel(this);
    _lbl->setStyleSheet("QLabel { color : white; }");
    _lbl->move(10,5);
}

// Destructor
CVolumeViewer::~CVolumeViewer()
{
    delete fGraphicsView;
    delete fScene;
}

float round_scale(float scale)
{
    if (abs(scale-round(log2(scale))) < 0.02f)
        scale = pow(2,round(log2(scale)));
    // the most reduced OME zarr projection is 32x so make the min zoom out 1/32 = 0.03125
    if (scale < MIN_ZOOM) scale = MIN_ZOOM;
    if (scale > MAX_ZOOM) scale = MAX_ZOOM;
    return scale;
}

//get center of current visible area in scene coordinates
QPointF visible_center(QGraphicsView *view)
{
    QRectF bbox = view->mapToScene(view->viewport()->geometry()).boundingRect();
    return bbox.topLeft() + QPointF(bbox.width(),bbox.height())*0.5;
}


QPointF CVolumeViewer::volumeToScene(const cv::Vec3f& vol_point)
{
    PlaneSurface* plane = dynamic_cast<PlaneSurface*>(_surf);
    QuadSurface* quad = dynamic_cast<QuadSurface*>(_surf);
    cv::Vec3f p;

    if (plane) {
        p = plane->project(vol_point, 1.0, _scale);
    } else if (quad) {
        auto ptr = quad->pointer();
        _surf->pointTo(ptr, vol_point, 4.0, 100);
        p = _surf->loc(ptr) * _scale;
    }

    return QPointF(p[0], p[1]);
}

bool scene2vol(cv::Vec3f &p, cv::Vec3f &n, Surface *_surf, const std::string &_surf_name, CSurfaceCollection *_surf_col, const QPointF &scene_loc, const cv::Vec2f &_vis_center, float _ds_scale)
{
    // Safety check for null surface
    if (!_surf) {
        p = cv::Vec3f(0, 0, 0);
        n = cv::Vec3f(0, 0, 1);
        return false;
    }
    
    try {
        cv::Vec3f surf_loc = {static_cast<float>(scene_loc.x()/_ds_scale), static_cast<float>(scene_loc.y()/_ds_scale),0};
        
        auto ptr = _surf->pointer();
        
        n = _surf->normal(ptr, surf_loc);
        p = _surf->coord(ptr, surf_loc);
    } catch (const cv::Exception& e) {
        return false;
    }
    return true;
}

cv::Vec3f CVolumeViewer::sceneToVolume(const QPointF& scenePoint) const
{
    cv::Vec3f p, n;
    if (scene2vol(p, n,
                  const_cast<Surface*>(_surf),
                  _surf_name,
                  const_cast<CSurfaceCollection*>(_surf_col),
                  scenePoint,
                  _vis_center,
                  _scale)) {
        return p;
    }
    return {0.0f, 0.0f, 0.0f};
}

void CVolumeViewer::onCursorMove(QPointF scene_loc)
{
    if (!_surf || !_surf_col)
        return;

    cv::Vec3f p, n;
    if (!scene2vol(p, n, _surf, _surf_name, _surf_col, scene_loc, _vis_center, _scale)) {
        if (_cursor) _cursor->hide();
    } else {
        if (_cursor) {
            _cursor->show();
            PlaneSurface *plane = dynamic_cast<PlaneSurface*>(_surf);
            QuadSurface *quad = dynamic_cast<QuadSurface*>(_surf);
            if (plane) {
                const cv::Vec3f sp = plane->project(p, 1.0, _scale);
                _cursor->setPos(sp[0], sp[1]);
            } else if (quad) {
                // We already know the cursor's scene position when interacting with a quad,
                // so avoid re-running the expensive pointTo() search.
                _cursor->setPos(scene_loc);
            }
        }

        POI *cursor = _surf_col->poi("cursor");
        if (!cursor)
            cursor = new POI;
        cursor->p = p;
        cursor->n = n;
        cursor->src = _surf;
        _surf_col->setPOI("cursor", cursor);
    }

    if (_point_collection && _dragged_point_id == 0) {
        uint64_t old_highlighted_id = _highlighted_point_id;
        _highlighted_point_id = 0;

        const float highlight_dist_threshold = 10.0f;
        float min_dist_sq = highlight_dist_threshold * highlight_dist_threshold;

        const auto& collections = _point_collection->getAllCollections();
        for (const auto& col_pair : collections) {
            for (const auto& point_pair : col_pair.second.points) {
                QPointF point_scene_pos = volumeToScene(point_pair.second.p);
                QPointF diff = scene_loc - point_scene_pos;
                float dist_sq = QPointF::dotProduct(diff, diff);
                if (dist_sq < min_dist_sq) {
                    min_dist_sq = dist_sq;
                    _highlighted_point_id = point_pair.second.id;
                }
            }
        }

        if (old_highlighted_id != _highlighted_point_id) {
            emit overlaysUpdated();
        }
    }
}

void CVolumeViewer::recalcScales()
{
    float old_ds = _ds_scale;         // remember previous level
    // if (dynamic_cast<PlaneSurface*>(_surf))
    _min_scale = pow(2.0,1.-volume->numScales());
    // else
        // _min_scale = std::max(pow(2.0,1.-volume->numScales()), 0.5);
    
    /* -------- chooses _ds_scale/_ds_sd_idx -------- */
    if      (_scale >= _max_scale) { _ds_sd_idx = 0;                         }
    else if (_scale <  _min_scale) { _ds_sd_idx = volume->numScales()-1;     }
    else  { _ds_sd_idx = int(std::round(-std::log2(_scale))); }
    if (_downscale_override > 0) {
        _ds_sd_idx += _downscale_override;
        // Clamp to available scales
        _ds_sd_idx = std::min(_ds_sd_idx, (int)volume->numScales() - 1);
    }
    _ds_scale = std::pow(2.0f, -_ds_sd_idx);
    /* ---------------------------------------------------------------- */

    /* ---- refresh physical voxel size when pyramid level flips -- */
    if (volume && std::abs(_ds_scale - old_ds) > 1e-6f)
    {
        double vs = volume->voxelSize() / _ds_scale;   // µm per scene-unit
        fGraphicsView->setVoxelSize(vs, vs);           // keep scalebar honest
    }
}


void CVolumeViewer::onZoom(int steps, QPointF scene_loc, Qt::KeyboardModifiers modifiers)
{
    if (!_surf)
        return;

    if (_segmentationEditActive && (modifiers & Qt::ControlModifier)) {
        cv::Vec3f world = sceneToVolume(scene_loc);
        emit sendSegmentationRadiusWheel(steps, scene_loc, world);
        return;
    }

    for (auto& col : _intersect_items)
        for (auto& item : col.second)
            item->setVisible(false);

    bool handled = false;

    if (modifiers & Qt::ShiftModifier) {
        if (steps == 0) {
            return;
        }

        PlaneSurface* plane = dynamic_cast<PlaneSurface*>(_surf);
        int adjustedSteps = steps;

        if (_surf_name != "segmentation" && plane && _surf_col) {
            POI* focus = _surf_col->poi("focus");
            if (!focus) {
                focus = new POI;
                focus->p = plane->origin();
                focus->n = plane->normal(plane->pointer(), {});
            }

            cv::Vec3f normal = plane->normal(plane->pointer(), {});
            const double length = cv::norm(normal);
            if (length > 0.0) {
                normal *= static_cast<float>(1.0 / length);
            }

            cv::Vec3f newPosition = focus->p + normal * static_cast<float>(adjustedSteps);

            if (volume) {
                const float maxX = static_cast<float>(volume->sliceWidth() - 1);
                const float maxY = static_cast<float>(volume->sliceHeight() - 1);
                const float maxZ = static_cast<float>(volume->numSlices() - 1);

                newPosition[0] = std::clamp(newPosition[0], 0.0f, maxX);
                newPosition[1] = std::clamp(newPosition[1], 0.0f, maxY);
                newPosition[2] = std::clamp(newPosition[2], 0.0f, maxZ);
            }

            focus->p = newPosition;
            if (length > 0.0) {
                focus->n = normal;
            }
            focus->src = plane;

            {
                QScopedValueRollback<bool> focusGuard(_suppressFocusRecentering, true);
                _surf_col->setPOI("focus", focus);
            }
            handled = true;
        } else {
            if (_surf_name == "segmentation") {
                adjustedSteps = (steps > 0) ? 1 : -1;
            }

            _z_off += adjustedSteps;

            if (volume && plane) {
                float effective_z = plane->origin()[2] + _z_off;
                effective_z = std::max(0.0f, std::min(effective_z, static_cast<float>(volume->numSlices() - 1)));
                _z_off = effective_z - plane->origin()[2];
            }

            renderVisible(true);
            handled = true;
        }
    }

    if (!handled) {
        float zoom = pow(ZOOM_FACTOR, steps);
        _scale *= zoom;
        _scale = round_scale(_scale);
        // we should only zoom when we haven't hit the max / min, otherwise the zoom starts to pan center on the mouse
        if (_scale > MIN_ZOOM && _scale < MAX_ZOOM) {
            recalcScales();

            // The above scale is *not* part of Qt's scene-to-view transform, but part of the voxel-to-scene transform
            // implemented in PlaneSurface::project; it causes a zoom around the surface origin
            // Translations are represented in the Qt scene-to-view transform; these move the surface origin within the viewpoint
            // To zoom centered on the mouse, we adjust the scene-to-view translation appropriately
            // If the mouse were at the plane/surface origin, this adjustment should be zero
            // If the mouse were right of the plane origin, should translate to the left so that point ends up where it was
            fGraphicsView->translate(scene_loc.x() * (1 - zoom),
                                     scene_loc.y() * (1 - zoom));

            curr_img_area = {0,0,0,0};
            int max_size = 100000;
            fGraphicsView->setSceneRect(-max_size/2, -max_size/2, max_size, max_size);
        }
        renderVisible();
        emit overlaysUpdated();

        // Update center marker position after zoom for QuadSurface
        if (_center_marker && _center_marker->isVisible()) {
            if (auto* quad = dynamic_cast<QuadSurface*>(_surf)) {
                POI* focus = _surf_col->poi("focus");
                if (focus) {
                    auto ptr = quad->pointer();
                    float dist = quad->pointTo(ptr, focus->p, 4.0, 100);
                    if (dist < 4.0) {
                        cv::Vec3f sp = quad->loc(ptr) * _scale;
                        _center_marker->setPos(sp[0], sp[1]);
                    }
                }
            }
        }
    }

    _lbl->setText(QString("%1x %2").arg(_scale).arg(_z_off));

    _overlayUpdateTimer->stop();
    _overlayUpdateTimer->start();
}

void CVolumeViewer::OnVolumeChanged(std::shared_ptr<Volume> volume_)
{
    volume = volume_;
    
    // printf("sizes %d %d %d\n", volume_->sliceWidth(), volume_->sliceHeight(), volume_->numSlices());

    int max_size = 100000 ;//std::max(volume_->sliceWidth(), std::max(volume_->numSlices(), volume_->sliceHeight()))*_ds_scale + 512;
    // printf("max size %d\n", max_size);
    fGraphicsView->setSceneRect(-max_size/2,-max_size/2,max_size,max_size);
    
    if (volume->numScales() >= 2) {
        //FIXME currently hardcoded
        _max_scale = 0.5;
        _min_scale = pow(2.0,1.-volume->numScales());
    }
    else {
        //FIXME currently hardcoded
        _max_scale = 1.0;
        _min_scale = 1.0;
    }
    
    recalcScales();

    _lbl->setText(QString("%1x %2").arg(_scale).arg(_z_off));

    renderVisible(true);

    // ——— Scalebar: physical size per scene-unit, compensating for down-sampling ———
    // volume->voxelSize() is µm per original voxel;
    // each scene-unit is still one original voxel, but we read data at (_ds_scale) resolution,
    // so we scale the voxelSize by 1/_ds_scale.
    double vs = volume->voxelSize() / _ds_scale;
    fGraphicsView->setVoxelSize(vs, vs);
}

void CVolumeViewer::onVolumeClicked(QPointF scene_loc, Qt::MouseButton buttons, Qt::KeyboardModifiers modifiers)
{
    if (!_surf)
        return;

    // If a point was being dragged, don't do anything on release
    if (_dragged_point_id != 0) {
        return;
    }

    cv::Vec3f p, n;
    if (!scene2vol(p, n, _surf, _surf_name, _surf_col, scene_loc, _vis_center, _scale))
        return;

    if (buttons == Qt::LeftButton) {
        bool isShift = modifiers.testFlag(Qt::ShiftModifier);

        if (isShift && !_segmentationEditActive) {
            // If a collection is selected, add to it.
            if (_selected_collection_id != 0) {
                const auto& collections = _point_collection->getAllCollections();
                auto it = collections.find(_selected_collection_id);
                if (it != collections.end()) {
                    _point_collection->addPoint(it->second.name, p);
                }
            } else {
                // Otherwise, create a new collection.
                std::string new_name = _point_collection->generateNewCollectionName("col");
                auto new_point = _point_collection->addPoint(new_name, p);
                _selected_collection_id = new_point.collectionId;
                emit sendCollectionSelected(_selected_collection_id);
            }
        } else if (_highlighted_point_id != 0) {
            emit pointClicked(_highlighted_point_id);
        }
    }

    const auto& segmentation = activeSegmentationHandle();

    // Forward the click for focus
    if (dynamic_cast<PlaneSurface*>(_surf)) {
        sendVolumeClicked(p, n, _surf, buttons, modifiers);
    }
    else if (segmentation.viewerIsSegmentationView && segmentation.surface) {
        sendVolumeClicked(p, n, segmentation.surface, buttons, modifiers);
    }
    else {
        std::cout << "FIXME: onVolumeClicked()" << std::endl;
    }
}

void CVolumeViewer::setCache(ChunkCache<uint8_t> *cache_)
{
    cache = cache_;
}

void CVolumeViewer::setPointCollection(VCCollection* point_collection)
{
    _point_collection = point_collection;
    emit overlaysUpdated();
}

Surface* CVolumeViewer::currentSurface() const
{
    if (!_surf_col) {
        return _surf;
    }

    Surface* surface = _surf_col->surface(_surf_name);
    if (surface != _surf) {
        const_cast<CVolumeViewer*>(this)->_surf = surface;
    }

    return surface;
}

void CVolumeViewer::setSurface(const std::string &name)
{
    _surf_name = name;
    _surf = nullptr;
    markActiveSegmentationDirty();
    onSurfaceChanged(name, _surf_col->surface(name));
}


void CVolumeViewer::invalidateVis()
{
    for(auto &item : slice_vis_items) {
        fScene->removeItem(item);
        delete item;
    }
    slice_vis_items.resize(0);
}

void CVolumeViewer::setSegmentationEditActive(bool active)
{
    if (_segmentationEditActive == active) {
        return;
    }
    _segmentationEditActive = active;
    renderIntersections();
}


void CVolumeViewer::setHighlightedSurfaceIds(const std::vector<std::string>& ids)
{
    std::unordered_set<std::string> next(ids.begin(), ids.end());
    if (next == _highlightedSurfaceIds) {
        return;
    }
    _highlightedSurfaceIds = std::move(next);
    renderIntersections();
}

void CVolumeViewer::setSurfacePatchSamplingStride(int stride)
{
    stride = std::max(1, stride);
    if (_surfacePatchSamplingStride == stride) {
        return;
    }
    _surfacePatchSamplingStride = stride;
    renderIntersections();
}

void CVolumeViewer::setVolumeWindow(float low, float high)
{
    constexpr float kMaxValue = 255.0f;
    const float clampedLow = std::clamp(low, 0.0f, kMaxValue);
    float clampedHigh = std::clamp(high, 0.0f, kMaxValue);
    if (clampedHigh <= clampedLow) {
        clampedHigh = std::min(kMaxValue, clampedLow + 1.0f);
    }

    const bool unchanged = std::abs(clampedLow - _baseWindowLow) < 1e-6f &&
                           std::abs(clampedHigh - _baseWindowHigh) < 1e-6f;
    if (unchanged) {
        return;
    }

    _baseWindowLow = clampedLow;
    _baseWindowHigh = clampedHigh;

    if (volume) {
        renderVisible(true);
    }
}


void CVolumeViewer::fitSurfaceInView()
{
    if (!_surf || !fGraphicsView) {
        return;
    }

    Rect3D bbox;
    bool haveBounds = false;

    if (auto* quadSurf = dynamic_cast<QuadSurface*>(_surf)) {
        bbox = quadSurf->bbox();
        haveBounds = true;
    }

    if (!haveBounds) {
        // when we can't get bounds, just reset to a default view
        _scale = 1.0f;
        recalcScales();
        fGraphicsView->resetTransform();
        fGraphicsView->centerOn(0, 0);
        _lbl->setText(QString("%1x %2").arg(_scale).arg(_z_off));
        return;
    }

    // Calculate the actual dimensions of the bounding box
    float bboxWidth = bbox.high[0] - bbox.low[0];
    float bboxHeight = bbox.high[1] - bbox.low[1];

    if (bboxWidth <= 0 || bboxHeight <= 0) {
        return;
    }

    QSize viewportSize = fGraphicsView->viewport()->size();
    float viewportWidth = viewportSize.width();
    float viewportHeight = viewportSize.height();

    if (viewportWidth <= 0 || viewportHeight <= 0) {
        return;
    }

    // Calculate scale factor based on actual bbox dimensions
    float fit_factor = 0.8f;
    float required_scale_x = (viewportWidth * fit_factor) / bboxWidth;
    float required_scale_y = (viewportHeight * fit_factor) / bboxHeight;

    // Use the smaller scale to ensure the entire bbox fits
    float required_scale = std::min(required_scale_x, required_scale_y);

    _scale = required_scale;
    _scale = round_scale(_scale);
    recalcScales();

    fGraphicsView->resetTransform();
    fGraphicsView->centerOn(0, 0);

    _lbl->setText(QString("%1x %2").arg(_scale).arg(_z_off));
    curr_img_area = {0,0,0,0};
}


void CVolumeViewer::onSurfaceChanged(std::string name, Surface *surf, bool isEditUpdate)
{
    if (name == "segmentation" || name == _surf_name) {
        markActiveSegmentationDirty();
    }

    // Track whether we need to re-render intersections (debounce multiple triggers)
    bool needsIntersectionUpdate = false;

    // When active segmentation changes, force re-render of intersections
    // so the highlight colors update immediately (old segment loses highlight,
    // new segment gains it)
    // Skip if _intersect_tgts contains "segmentation" since it will be handled
    // by the intersection target logic below (avoids create-delete-create race
    // that can confuse Qt's scene invalidation)
    if (name == "segmentation" && !_intersect_tgts.count("segmentation")) {
        needsIntersectionUpdate = true;
    }

    if (_surf_name == name) {
        _surf = surf;
        if (!_surf) {
            clearAllOverlayGroups();
            fScene->clear();
            _intersect_items.clear();
            _cachedIntersectionLines.clear();
            slice_vis_items.clear();
            _paths.clear();
            emit overlaysUpdated();
            _cursor = nullptr;
            _center_marker = nullptr;
            fBaseImageItem = nullptr;
        }
        else {
            invalidateVis();
            if (!isEditUpdate) {
                _z_off = 0.0f;
            }
            if (name == "segmentation" && _resetViewOnSurfaceChange) {
                fitSurfaceInView();
            }
        }
    }

    if (name == _surf_name) {
        curr_img_area = {0,0,0,0};
        renderVisible(true); // Immediate render of slice
        // When the slice plane itself moves, re-render intersections since
        // the view_bbox will be at the new position
        needsIntersectionUpdate = true;
    }

    if (_intersect_tgts.count(name)) {
        invalidateIntersect(name);
        needsIntersectionUpdate = true;
    }

    // Single renderIntersections() call to avoid create-delete-create race
    if (needsIntersectionUpdate) {
        renderIntersections();
    }

    // Defer overlay updates
    _overlayUpdateTimer->stop();
    _overlayUpdateTimer->start();
}

QGraphicsItem *cursorItem(bool drawingMode = false, float brushSize = 3.0f, bool isSquare = false)
{
    if (drawingMode) {
        // Drawing mode cursor - shows brush shape and size
        QGraphicsItemGroup *group = new QGraphicsItemGroup();
        group->setZValue(10);
        
        QPen brushPen(QBrush(COLOR_CURSOR), 1.5);
        brushPen.setStyle(Qt::DashLine);
        
        // Draw brush shape
        if (isSquare) {
            float halfSize = brushSize / 2.0f;
            QGraphicsRectItem *rect = new QGraphicsRectItem(-halfSize, -halfSize, brushSize, brushSize);
            rect->setPen(brushPen);
            rect->setBrush(Qt::NoBrush);
            group->addToGroup(rect);
        } else {
            QGraphicsEllipseItem *circle = new QGraphicsEllipseItem(-brushSize/2, -brushSize/2, brushSize, brushSize);
            circle->setPen(brushPen);
            circle->setBrush(Qt::NoBrush);
            group->addToGroup(circle);
        }
        
        // Add small crosshair in center
        QPen centerPen(QBrush(COLOR_CURSOR), 1);
        QGraphicsLineItem *line = new QGraphicsLineItem(-2, 0, 2, 0);
        line->setPen(centerPen);
        group->addToGroup(line);
        line = new QGraphicsLineItem(0, -2, 0, 2);
        line->setPen(centerPen);
        group->addToGroup(line);
        
        return group;
    } else {
        // Regular cursor
        QPen pen(QBrush(COLOR_CURSOR), 2);
        QGraphicsLineItem *parent = new QGraphicsLineItem(-10, 0, -5, 0);
        parent->setZValue(10);
        parent->setPen(pen);
        QGraphicsLineItem *line = new QGraphicsLineItem(10, 0, 5, 0, parent);
        line->setPen(pen);
        line = new QGraphicsLineItem(0, -10, 0, -5, parent);
        line->setPen(pen);
        line = new QGraphicsLineItem(0, 10, 0, 5, parent);
        line->setPen(pen);
        
        return parent;
    }
}

QGraphicsItem *crossItem()
{
    QPen pen(QBrush(Qt::red), 1);
    QGraphicsLineItem *parent = new QGraphicsLineItem(-5, -5, 5, 5);
    parent->setZValue(10);
    parent->setPen(pen);
    QGraphicsLineItem *line = new QGraphicsLineItem(-5, 5, 5, -5, parent);
    line->setPen(pen);
    
    return parent;
}

//TODO make poi tracking optional and configurable
void CVolumeViewer::onPOIChanged(std::string name, POI *poi)
{    
    if (!poi || !_surf)
        return;
    
    if (name == "focus") {
        // Add safety check before dynamic_cast
        if (!_surf) {
            return;
        }
        
        if (auto* plane = dynamic_cast<PlaneSurface*>(_surf)) {
            if (!_suppressFocusRecentering) {
                fGraphicsView->centerOn(0, 0);
            }
            if (poi->p == plane->origin())
                return;
            
            plane->setOrigin(poi->p);
            emit overlaysUpdated();
            
            _surf_col->setSurface(_surf_name, plane);
        } else if (auto* quad = dynamic_cast<QuadSurface*>(_surf)) {
            auto ptr = quad->pointer();
            float dist = quad->pointTo(ptr, poi->p, 4.0, 100);
            
            if (dist < 4.0) {
                cv::Vec3f sp = quad->loc(ptr) * _scale;
                if (_center_marker) {
                    _center_marker->setPos(sp[0], sp[1]);
                    _center_marker->show();
                }
                fGraphicsView->centerOn(sp[0], sp[1]);
            } else {
                if (_center_marker) {
                    _center_marker->hide();
                }
            }

            renderVisible(true);
        }
    }
    else if (name == "cursor") {
        // Add safety check before dynamic_cast
        if (!_surf) {
            return;
        }

        if (_surf_name == "segmentation" && !_mirrorCursorToSegmentation) {
            if (!poi->src || poi->src != _surf) {
                return;
            }
        }

        PlaneSurface *slice_plane = dynamic_cast<PlaneSurface*>(_surf);
        const auto& segmentation = activeSegmentationHandle();
        QuadSurface *crop = segmentation.surface;
        
        cv::Vec3f sp;
        float dist = -1;
        if (slice_plane) {            
            dist = slice_plane->pointDist(poi->p);
            sp = slice_plane->project(poi->p, 1.0, _scale);
        }
        else if (segmentation.viewerIsSegmentationView && crop)
        {
            auto ptr = crop->pointer();
            dist = crop->pointTo(ptr, poi->p, 2.0);
            sp = crop->loc(ptr)*_scale ;//+ cv::Vec3f(_vis_center[0],_vis_center[1],0);
        }
        
        if (!_cursor) {
            _cursor = cursorItem(_drawingModeActive, _brushSize, _brushIsSquare);
            fScene->addItem(_cursor);
        }
        
        if (dist != -1) {
            if (dist < 20.0/_scale) {
                _cursor->setPos(sp[0], sp[1]);
                _cursor->setOpacity(1.0-dist*_scale/20.0);
            }
            else
                _cursor->setOpacity(0.0);
        }
    }
}



void CVolumeViewer::onPanStart(Qt::MouseButton buttons, Qt::KeyboardModifiers modifiers)
{
    renderVisible();

    _overlayUpdateTimer->stop();
    _overlayUpdateTimer->start();
}

void CVolumeViewer::onPanRelease(Qt::MouseButton buttons, Qt::KeyboardModifiers modifiers)
{
    renderVisible();

    _overlayUpdateTimer->stop();
    _overlayUpdateTimer->start();
}

void CVolumeViewer::onScrolled()
{
    // renderVisible();
}

void CVolumeViewer::onResized()
{
   renderVisible(true);
}

void CVolumeViewer::onPathsChanged(const QList<PathPrimitive>& paths)
{
    _paths.clear();
    _paths.reserve(paths.size());
    for (const auto& path : paths) {
        _paths.push_back(path);
    }
    emit overlaysUpdated();
}

void CVolumeViewer::onMousePress(QPointF scene_loc, Qt::MouseButton button, Qt::KeyboardModifiers modifiers)
{
    // BBox drawing consumes mouse events on segmentation view
    if (_bboxMode && _surf_name == "segmentation") {
        if (button == Qt::LeftButton) {
            // Convert to surface parameter coords (unscaled)
            cv::Vec3f p, n;
            if (!scene2vol(p, n, _surf, _surf_name, _surf_col, scene_loc, _vis_center, _scale)) return;
            auto* quad = dynamic_cast<QuadSurface*>(_surf);
            if (!quad) return;
            auto ptr = quad->pointer();
            quad->pointTo(ptr, p, 2.0f, 100);
            cv::Vec3f sp = quad->loc(ptr); // unscaled surface coords
            _bboxStart = QPointF(sp[0], sp[1]);
            QRectF r(QPointF(_bboxStart.x()*_scale, _bboxStart.y()*_scale), QPointF(_bboxStart.x()*_scale, _bboxStart.y()*_scale));
            _activeBBoxSceneRect = r.normalized();
            emit overlaysUpdated();
        }
        return; // consume in bbox mode
    }
    if (!_point_collection || !_surf) return;

    if (button == Qt::LeftButton) {
        if (_highlighted_point_id != 0 && !modifiers.testFlag(Qt::ControlModifier)) {
            emit pointClicked(_highlighted_point_id);
            _dragged_point_id = _highlighted_point_id;
            // Do not return, allow forwarding for other widgets
        }
    } else if (button == Qt::RightButton) {
        if (_highlighted_point_id != 0) {
            _point_collection->removePoint(_highlighted_point_id);
        }
    }

    // Forward for drawing widgets
    cv::Vec3f p, n;
    if (scene2vol(p, n, _surf, _surf_name, _surf_col, scene_loc, _vis_center, _scale)) {
        _lastScenePos = scene_loc;  // Track for grid coordinate lookups
        sendMousePressVolume(p, n, button, modifiers);
    }
}

void CVolumeViewer::onMouseMove(QPointF scene_loc, Qt::MouseButtons buttons, Qt::KeyboardModifiers modifiers)
{
    // BBox drawing consumes mouse events on segmentation view
    if (_bboxMode && _surf_name == "segmentation") {
        if (_activeBBoxSceneRect && (buttons & Qt::LeftButton)) {
            cv::Vec3f p, n;
            if (!scene2vol(p, n, _surf, _surf_name, _surf_col, scene_loc, _vis_center, _scale)) return;
            auto* quad = dynamic_cast<QuadSurface*>(_surf);
            if (!quad) return;
            auto ptr = quad->pointer();
            quad->pointTo(ptr, p, 2.0f, 100);
            cv::Vec3f sp = quad->loc(ptr); // unscaled
            QPointF cur(sp[0], sp[1]);
            QRectF r(QPointF(_bboxStart.x()*_scale, _bboxStart.y()*_scale), QPointF(cur.x()*_scale, cur.y()*_scale));
            _activeBBoxSceneRect = r.normalized();
            emit overlaysUpdated();
        }
        return; // consume in bbox mode
    }
    onCursorMove(scene_loc); // Keep highlighting up to date

    if ((buttons & Qt::LeftButton) && _dragged_point_id != 0) {
        cv::Vec3f p, n;
        if (scene2vol(p, n, _surf, _surf_name, _surf_col, scene_loc, _vis_center, _scale)) {
            if (auto point_opt = _point_collection->getPoint(_dragged_point_id)) {
                ColPoint updated_point = *point_opt;
                updated_point.p = p;
                _point_collection->updatePoint(updated_point);
            }
        }
    } else {
        if (!_surf) {
            return;
        }
        
        cv::Vec3f p, n;
        if (!scene2vol(p, n, _surf, _surf_name, _surf_col, scene_loc, _vis_center, _scale))
            return;

        _lastScenePos = scene_loc;  // Track for grid coordinate lookups
        emit sendMouseMoveVolume(p, buttons, modifiers);
    }
}

void CVolumeViewer::onMouseRelease(QPointF scene_loc, Qt::MouseButton button, Qt::KeyboardModifiers modifiers)
{
    // BBox drawing consumes mouse events on segmentation view
    if (_bboxMode && _surf_name == "segmentation") {
        if (button == Qt::LeftButton && _activeBBoxSceneRect) {
            // Determine final rect in surface parameter coords
            QRectF rScene = _activeBBoxSceneRect->normalized();
            QRectF rSurf(QPointF(rScene.left()/_scale, rScene.top()/_scale), QPointF(rScene.right()/_scale, rScene.bottom()/_scale));
            // Promote this rectangle into a persistent selection with unique color (stored unscaled)
            // Generate a distinct color using HSV cycling
            int idx = static_cast<int>(_selections.size());
            QColor col = QColor::fromHsv((idx * 53) % 360, 200, 255);
            _selections.push_back({rSurf, col});
            _activeBBoxSceneRect.reset();
            emit overlaysUpdated();
        }
        return; // consume in bbox mode
    }
    if (button == Qt::LeftButton && _dragged_point_id != 0) {
        _dragged_point_id = 0;
        // Re-run highlight logic
        onCursorMove(scene_loc);
    }

    // Forward for drawing widgets
    cv::Vec3f p, n;
    if (scene2vol(p, n, _surf, _surf_name, _surf_col, scene_loc, _vis_center, _scale)) {
        const auto& segmentation = activeSegmentationHandle();
        if (dynamic_cast<PlaneSurface*>(_surf)) {
            emit sendMouseReleaseVolume(p, button, modifiers);
        }
        else if (segmentation.viewerIsSegmentationView) {
            emit sendMouseReleaseVolume(p, button, modifiers);
        }
        else {
            std::cout << "FIXME: onMouseRelease()" << std::endl;
        }
    }
}

void CVolumeViewer::setBBoxMode(bool enabled)
{
    _bboxMode = enabled;
    if (!enabled && _activeBBoxSceneRect) {
        _activeBBoxSceneRect.reset();
        emit overlaysUpdated();
    }
}

QuadSurface* CVolumeViewer::makeBBoxFilteredSurfaceFromSceneRect(const QRectF& sceneRect)
{
    if (_surf_name != "segmentation") return nullptr;
    auto* quad = dynamic_cast<QuadSurface*>(_surf);
    if (!quad) return nullptr;

    const cv::Mat_<cv::Vec3f> src = quad->rawPoints();
    const int H = src.rows;
    const int W = src.cols;

    // Convert scene-space rect to surface-parameter rect (nominal units)
    QRectF rSurf(QPointF(sceneRect.left()/_scale,  sceneRect.top()/_scale),
                 QPointF(sceneRect.right()/_scale, sceneRect.bottom()/_scale));
    rSurf = rSurf.normalized();

    // Compute tight index bounds from surface-parameter rect
    const double cx = W * 0.5; // cols/2
    const double cy = H * 0.5; // rows/2
    const cv::Vec2f sc = quad->scale();
    int i0 = std::max(0,               (int)std::floor(cx + rSurf.left()   * sc[0]));
    int i1 = std::min(W - 1,           (int)std::ceil (cx + rSurf.right()  * sc[0]));
    int j0 = std::max(0,               (int)std::floor(cy + rSurf.top()    * sc[1]));
    int j1 = std::min(H - 1,           (int)std::ceil (cy + rSurf.bottom() * sc[1]));
    if (i0 > i1 || j0 > j1) return nullptr;

    const int outW = (i1 - i0 + 1);
    const int outH = (j1 - j0 + 1);
    cv::Mat_<cv::Vec3f> cropped(outH, outW, cv::Vec3f(-1.f, -1.f, -1.f));

    // Keep only points whose parameter coords fall inside rSurf (cheap, linear mapping)
    for (int j = j0; j <= j1; ++j) {
        for (int i = i0; i <= i1; ++i) {
            const cv::Vec3f& p = src(j, i);
            if (p[0] == -1.0f && p[1] == -1.0f && p[2] == -1.0f) continue;
            const double u = (i - cx) / sc[0];
            const double v = (j - cy) / sc[1];
            if (u >= rSurf.left() && u <= rSurf.right() && v >= rSurf.top() && v <= rSurf.bottom()) {
                cropped(j - j0, i - i0) = p;
            }
        }
    }

    // Remove spatial outliers, then trim to minimal grid again
    cv::Mat_<cv::Vec3f> cleaned = clean_surface_outliers(cropped);

    // Optional heuristic: tighten edges by requiring a minimum number of valid
    // points per border row/column to consider it part of the crop.
    auto countValidInCol = [&](int c) {
        int cnt = 0; for (int r = 0; r < cleaned.rows; ++r) if (cleaned(r,c)[0] != -1) ++cnt; return cnt; };
    auto countValidInRow = [&](int r) {
        int cnt = 0; for (int c = 0; c < cleaned.cols; ++c) if (cleaned(r,c)[0] != -1) ++cnt; return cnt; };
    int minValidCol = std::max(1, std::min(3, cleaned.rows));
    int minValidRow = std::max(1, std::min(3, cleaned.cols));

    int left = 0, right = cleaned.cols - 1, top = 0, bottom = cleaned.rows - 1;
    while (left <= right && countValidInCol(left) < minValidCol) ++left;
    while (right >= left && countValidInCol(right) < minValidCol) --right;
    while (top <= bottom && countValidInRow(top) < minValidRow) ++top;
    while (bottom >= top && countValidInRow(bottom) < minValidRow) --bottom;

    // Fallback to bounding any valid cell if heuristic removed everything
    if (left > right || top > bottom) {
        left = cleaned.cols; right = -1; top = cleaned.rows; bottom = -1;
        for (int j = 0; j < cleaned.rows; ++j)
            for (int i = 0; i < cleaned.cols; ++i)
                if (cleaned(j,i)[0] != -1) {
                    left = std::min(left, i); right = std::max(right, i);
                    top  = std::min(top,  j); bottom= std::max(bottom,j);
                }
        if (right < 0 || bottom < 0) return nullptr; // all removed
    }

    const int fW = (right - left + 1);
    const int fH = (bottom - top + 1);
    cv::Mat_<cv::Vec3f> finalPts(fH, fW, cv::Vec3f(-1.f, -1.f, -1.f));
    for (int j = top; j <= bottom; ++j)
        for (int i = left; i <= right; ++i)
            finalPts(j - top, i - left) = cleaned(j, i);

    auto* out = new QuadSurface(finalPts, quad->_scale);
    return out;
}

auto CVolumeViewer::selections() const -> std::vector<std::pair<QRectF, QColor>>
{
    std::vector<std::pair<QRectF, QColor>> out;
    out.reserve(_selections.size());
    for (const auto& s : _selections) {
        QRectF sceneRect(QPointF(s.surfRect.left()*_scale,  s.surfRect.top()*_scale),
                         QPointF(s.surfRect.right()*_scale, s.surfRect.bottom()*_scale));
        out.emplace_back(sceneRect.normalized(), s.color);
    }
    return out;
}

void CVolumeViewer::clearSelections()
{
    _selections.clear();
    emit overlaysUpdated();
}

void CVolumeViewer::setCompositeEnabled(bool enabled)
{
    if (_composite_enabled != enabled) {
        _composite_enabled = enabled;
        renderVisible(true);
        
        // Update status label
        QString status = QString("%1x %2").arg(_scale).arg(_z_off);
        if (_composite_enabled) {
            QString method = QString::fromStdString(_composite_method);
            method[0] = method[0].toUpper();
            status += QString(" | Composite: %1(%2)").arg(method).arg(_composite_layers);
        }
        _lbl->setText(status);
    }
}
void CVolumeViewer::setCompositeLayersInFront(int layers)
{
    if (layers >= 0 && layers <= 21 && layers != _composite_layers_front) {
        _composite_layers_front = layers;
        if (_composite_enabled) {
            renderVisible(true);
        }
    }
}

void CVolumeViewer::setCompositeLayersBehind(int layers)
{
    if (layers >= 0 && layers <= 21 && layers != _composite_layers_behind) {
        _composite_layers_behind = layers;
        if (_composite_enabled) {
            renderVisible(true);
        }
    }
}

void CVolumeViewer::setCompositeAlphaMin(int value)
{
    if (value >= 0 && value <= 255 && value != _composite_alpha_min) {
        _composite_alpha_min = value;
        if (_composite_enabled && _composite_method == "alpha") {
            renderVisible(true);
        }
    }
}

void CVolumeViewer::setCompositeAlphaMax(int value)
{
    if (value >= 0 && value <= 255 && value != _composite_alpha_max) {
        _composite_alpha_max = value;
        if (_composite_enabled && _composite_method == "alpha") {
            renderVisible(true);
        }
    }
}

void CVolumeViewer::setCompositeAlphaThreshold(int value)
{
    if (value >= 0 && value <= 10000 && value != _composite_alpha_threshold) {
        _composite_alpha_threshold = value;
        if (_composite_enabled && _composite_method == "alpha") {
            renderVisible(true);
        }
    }
}

void CVolumeViewer::setCompositeMaterial(int value)
{
    if (value >= 0 && value <= 255 && value != _composite_material) {
        _composite_material = value;
        if (_composite_enabled && _composite_method == "alpha") {
            renderVisible(true);
        }
    }
}

void CVolumeViewer::setCompositeReverseDirection(bool reverse)
{
    if (reverse != _composite_reverse_direction) {
        _composite_reverse_direction = reverse;
        if (_composite_enabled) {
            renderVisible(true);
        }
    }
}

void CVolumeViewer::setCompositeMethod(const std::string& method)
{
    if (method != _composite_method && (method == "max" || method == "mean" || method == "min" || method == "alpha")) {
        _composite_method = method;
        if (_composite_enabled) {
            renderVisible(true);
            
            // Update status label
            QString status = QString("%1x %2").arg(_scale).arg(_z_off);
            QString methodDisplay = QString::fromStdString(_composite_method);
            methodDisplay[0] = methodDisplay[0].toUpper();
            status += QString(" | Composite: %1(%2)").arg(methodDisplay).arg(_composite_layers);
            _lbl->setText(status);
        }
    }
}

void CVolumeViewer::onVolumeClosing()
{
    // Only clear segmentation-related surfaces, not persistent plane surfaces
    if (_surf_name == "segmentation") {
        onSurfaceChanged(_surf_name, nullptr);
    }
    // For plane surfaces (xy plane, xz plane, yz plane), just clear the scene
    // but keep the surface reference so it can render with the new volume
    else if (_surf_name == "xy plane" || _surf_name == "xz plane" || _surf_name == "yz plane") {
        if (fScene) {
            clearAllOverlayGroups();
            fScene->clear();
        }
        // Clear all item collections
        _intersect_items.clear();
        _cachedIntersectionLines.clear();
        slice_vis_items.clear();
        _paths.clear();
        emit overlaysUpdated();
        _cursor = nullptr;
        _center_marker = nullptr;
        fBaseImageItem = nullptr;
        // Note: We don't set _surf = nullptr here, so the surface remains available
    }
    else {
        // For other surface types (seg xz, seg yz), clear them
        onSurfaceChanged(_surf_name, nullptr);
    }
}

void CVolumeViewer::onDrawingModeActive(bool active, float brushSize, bool isSquare)
{
    _drawingModeActive = active;
    _brushSize = brushSize;
    _brushIsSquare = isSquare;
    
    // Update the cursor to reflect the drawing mode state
    if (_cursor) {
        fScene->removeItem(_cursor);
        delete _cursor;
        _cursor = nullptr;
    }
    
    // Force cursor update
    POI *cursor = _surf_col->poi("cursor");
    if (cursor) {
        onPOIChanged("cursor", cursor);
    }
}

void CVolumeViewer::onCollectionSelected(uint64_t collectionId)
{
    _selected_collection_id = collectionId;
    emit overlaysUpdated();
}

void CVolumeViewer::onKeyRelease(int /*key*/, Qt::KeyboardModifiers /*modifiers*/)
{
}

void CVolumeViewer::onPointSelected(uint64_t pointId)
{
    if (_selected_point_id == pointId) {
        return;
    }

    uint64_t old_selected_id = _selected_point_id;
    _selected_point_id = pointId;

    emit overlaysUpdated();
}

void CVolumeViewer::setResetViewOnSurfaceChange(bool reset)
{
    _resetViewOnSurfaceChange = reset;
}


// Draw two small arrows indicating growth direction candidates:
// red = flip_x=false (along +X)
// green = flip_x=true (opposite −X)
// Shown on segmentation and projected into slice views.
