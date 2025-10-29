#include "CVolumeViewer.hpp"
#include "vc/ui/UDataManipulateUtils.hpp"

#include "VolumeViewerCmaps.hpp"

#include <QGraphicsView>
#include <QGraphicsScene>

#include "CVolumeViewerView.hpp"
#include "CSurfaceCollection.hpp"
#include "vc/ui/VCCollection.hpp"

#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/Slicing.hpp"

#include <omp.h>

#include "OpChain.hpp"
#include "vc/core/util/Render.hpp"

#include <QPainter>
#include <QScopedValueRollback>

#include <cstdint>
#include <list>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <utility>

#include <opencv2/imgproc.hpp>

using qga = QGuiApplication;

using PathPrimitive = ViewerOverlayControllerBase::PathPrimitive;
using PathBrushShape = ViewerOverlayControllerBase::PathBrushShape;

#define BGND_RECT_MARGIN 8
#define DEFAULT_TEXT_COLOR QColor(255, 255, 120)
// More gentle zoom factor for smoother experience
#define ZOOM_FACTOR 1.05 // Reduced from 1.15 for even smoother zooming

#define COLOR_CURSOR Qt::cyan
#define COLOR_FOCUS QColor(50, 255, 215)
#define COLOR_SEG_YZ Qt::yellow
#define COLOR_SEG_XZ Qt::red
#define COLOR_SEG_XY QColor(255, 140, 0)

constexpr float MIN_ZOOM = 0.03125f;
constexpr float MAX_ZOOM = 4.0f;

#include <limits>
#include <algorithm>
#include <cmath>

namespace
{
constexpr size_t kAxisAlignedSliceCacheCapacity = 180;
constexpr float kScaleQuantization = 1000.0f;
constexpr float kZOffsetQuantization = 1000.0f;
constexpr float kDsScaleQuantization = 1000.0f;

inline int quantizeFloat(float value, float multiplier)
{
    return static_cast<int>(std::lround(value * multiplier));
}

inline bool planeIdForSurface(const std::string& name, uint8_t& outId)
{
    if (name == "seg xz") {
        outId = 0;
        return true;
    }
    if (name == "seg yz") {
        outId = 1;
        return true;
    }
    return false;
}

struct AxisAlignedSliceCacheKey
{
    uint8_t planeId = 0;
    uint16_t rotationKey = 0;
    int originX = 0;
    int originY = 0;
    int originZ = 0;
    int roiX = 0;
    int roiY = 0;
    int roiWidth = 0;
    int roiHeight = 0;
    int scaleMilli = 0;
    int dsScaleMilli = 0;
    int zOffsetMilli = 0;
    int dsIndex = 0;
    uintptr_t datasetPtr = 0;
    uint8_t fastInterpolation = 0;
    uint8_t baseWindowLow = 0;
    uint8_t baseWindowHigh = 0;

    bool operator==(const AxisAlignedSliceCacheKey& other) const noexcept
    {
        return planeId == other.planeId && rotationKey == other.rotationKey &&
               originX == other.originX && originY == other.originY && originZ == other.originZ &&
               roiX == other.roiX && roiY == other.roiY &&
               roiWidth == other.roiWidth && roiHeight == other.roiHeight &&
               scaleMilli == other.scaleMilli && dsScaleMilli == other.dsScaleMilli &&
               zOffsetMilli == other.zOffsetMilli && dsIndex == other.dsIndex &&
               datasetPtr == other.datasetPtr && fastInterpolation == other.fastInterpolation &&
               baseWindowLow == other.baseWindowLow && baseWindowHigh == other.baseWindowHigh;
    }
};

struct AxisAlignedSliceCacheKeyHasher
{
    std::size_t operator()(const AxisAlignedSliceCacheKey& key) const noexcept
    {
        std::size_t seed = 0;
        hashCombine(seed, key.planeId);
        hashCombine(seed, key.rotationKey);
        hashCombine(seed, key.originX);
        hashCombine(seed, key.originY);
        hashCombine(seed, key.originZ);
        hashCombine(seed, key.roiX);
        hashCombine(seed, key.roiY);
        hashCombine(seed, key.roiWidth);
        hashCombine(seed, key.roiHeight);
        hashCombine(seed, key.scaleMilli);
        hashCombine(seed, key.dsScaleMilli);
        hashCombine(seed, key.zOffsetMilli);
        hashCombine(seed, key.dsIndex);
        hashCombine(seed, key.datasetPtr);
        hashCombine(seed, key.fastInterpolation);
        hashCombine(seed, key.baseWindowLow);
        hashCombine(seed, key.baseWindowHigh);
        return seed;
    }

private:
    static void hashCombine(std::size_t& seed, std::size_t value) noexcept
    {
        seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
    }
};

class AxisAlignedSliceCache
{
public:
    explicit AxisAlignedSliceCache(size_t capacity)
        : _capacity(capacity)
    {
    }

    std::optional<cv::Mat> get(const AxisAlignedSliceCacheKey& key)
    {
        std::lock_guard<std::mutex> lock(_mutex);
        auto it = _entries.find(key);
        if (it == _entries.end()) {
            return std::nullopt;
        }
        _lru.splice(_lru.begin(), _lru, it->second.orderIt);
        return it->second.image.clone();
    }

    void put(const AxisAlignedSliceCacheKey& key, const cv::Mat& image)
    {
        std::lock_guard<std::mutex> lock(_mutex);
        auto it = _entries.find(key);
        if (it != _entries.end()) {
            it->second.image = image.clone();
            _lru.splice(_lru.begin(), _lru, it->second.orderIt);
            return;
        }

        if (_entries.size() >= _capacity && !_lru.empty()) {
            const AxisAlignedSliceCacheKey& evictKey = _lru.back();
            _entries.erase(evictKey);
            _lru.pop_back();
        }

        _lru.push_front(key);
        Entry entry;
        entry.image = image.clone();
        entry.orderIt = _lru.begin();
        _entries.emplace(_lru.front(), std::move(entry));
    }

private:
    struct Entry {
        cv::Mat image;
        std::list<AxisAlignedSliceCacheKey>::iterator orderIt;
    };

    size_t _capacity;
    std::list<AxisAlignedSliceCacheKey> _lru;
    std::unordered_map<AxisAlignedSliceCacheKey, Entry, AxisAlignedSliceCacheKeyHasher> _entries;
    std::mutex _mutex;
};

AxisAlignedSliceCache& axisAlignedSliceCache()
{
    static AxisAlignedSliceCache cache(kAxisAlignedSliceCacheCapacity);
    return cache;
}

} // namespace

// Helper: remove spatial outliers based on robust neighbor-distance stats
static cv::Mat_<cv::Vec3f> clean_surface_outliers(const cv::Mat_<cv::Vec3f>& points, float distance_threshold = 5.0f)
{
    cv::Mat_<cv::Vec3f> cleaned = points.clone();

    std::vector<float> all_neighbor_dists;
    all_neighbor_dists.reserve(points.rows * points.cols);

    // First pass: gather neighbor distances
    for (int j = 0; j < points.rows; ++j) {
        for (int i = 0; i < points.cols; ++i) {
            if (points(j, i)[0] == -1) continue;
            const cv::Vec3f center = points(j, i);
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    const int ny = j + dy;
                    const int nx = i + dx;
                    if (ny >= 0 && ny < points.rows && nx >= 0 && nx < points.cols) {
                        if (points(ny, nx)[0] != -1) {
                            const cv::Vec3f neighbor = points(ny, nx);
                            float dist = cv::norm(center - neighbor);
                            if (std::isfinite(dist) && dist > 0) {
                                all_neighbor_dists.push_back(dist);
                            }
                        }
                    }
                }
            }
        }
    }

    float median_dist = 0.0f;
    float mad = 0.0f;
    if (!all_neighbor_dists.empty()) {
        std::sort(all_neighbor_dists.begin(), all_neighbor_dists.end());
        median_dist = all_neighbor_dists[all_neighbor_dists.size() / 2];
        std::vector<float> abs_devs;
        abs_devs.reserve(all_neighbor_dists.size());
        for (float d : all_neighbor_dists) abs_devs.push_back(std::abs(d - median_dist));
        std::sort(abs_devs.begin(), abs_devs.end());
        mad = abs_devs[abs_devs.size() / 2];
    }
    const float threshold = median_dist + distance_threshold * (mad / 0.6745f);

    // Second pass: invalidate isolated/far points
    for (int j = 0; j < points.rows; ++j) {
        for (int i = 0; i < points.cols; ++i) {
            if (points(j, i)[0] == -1) continue;
            const cv::Vec3f center = points(j, i);
            float min_neighbor = std::numeric_limits<float>::infinity();
            int neighbor_count = 0;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    const int ny = j + dy;
                    const int nx = i + dx;
                    if (ny >= 0 && ny < points.rows && nx >= 0 && nx < points.cols) {
                        if (points(ny, nx)[0] != -1) {
                            float dist = cv::norm(center - points(ny, nx));
                            if (std::isfinite(dist)) {
                                min_neighbor = std::min(min_neighbor, dist);
                                neighbor_count++;
                            }
                        }
                    }
                }
            }
            if (neighbor_count == 0 || (min_neighbor > threshold && threshold > 0)) {
                cleaned(j, i) = cv::Vec3f(-1.f, -1.f, -1.f);
            }
        }
    }
    return cleaned;
}


CVolumeViewer::CVolumeViewer(CSurfaceCollection *col, QWidget* parent)
    : QWidget(parent)
    , fGraphicsView(nullptr)
    , fBaseImageItem(nullptr)
    , _surf_col(col)
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

    QSettings settings("VC.ini", QSettings::IniFormat);
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
CVolumeViewer::~CVolumeViewer(void)
{
    delete fGraphicsView;
    delete fScene;
}

void round_scale(float &scale)
{
    if (abs(scale-round(log2(scale))) < 0.02f)
        scale = pow(2,round(log2(scale)));
    // the most reduced OME zarr projection is 32x so make the min zoom out 1/32 = 0.03125
    if (scale < MIN_ZOOM) scale = MIN_ZOOM;
    if (scale > MAX_ZOOM) scale = MAX_ZOOM;
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
        cv::Vec3f surf_loc = {scene_loc.x()/_ds_scale, scene_loc.y()/_ds_scale,0};
        
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
            // Update cursor position visually without POI
            PlaneSurface *plane = dynamic_cast<PlaneSurface*>(_surf);
            QuadSurface *quad = dynamic_cast<QuadSurface*>(_surf);
            cv::Vec3f sp;

            if (plane) {
                sp = plane->project(p, 1.0, _scale);
            } else if (quad) {
                auto ptr = quad->pointer();
                _surf->pointTo(ptr, p, 4.0, 100);
                sp = _surf->loc(ptr) * _scale;
            }
            _cursor->setPos(sp[0], sp[1]);
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
        round_scale(_scale);
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

    // Forward the click for focus
    if (dynamic_cast<PlaneSurface*>(_surf))
        sendVolumeClicked(p, n, _surf, buttons, modifiers);
    else if (_surf_name == "segmentation")
        sendVolumeClicked(p, n, _surf_col->surface("segmentation"), buttons, modifiers);
    else
        std::cout << "FIXME: onVolumeClicked()" << std::endl;
}

void CVolumeViewer::setCache(ChunkCache *cache_)
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

void CVolumeViewer::invalidateIntersect(const std::string &name)
{
    if (!name.size() || name == _surf_name) {
        for(auto &pair : _intersect_items) {
            for(auto &item : pair.second) {
                fScene->removeItem(item);
                delete item;
            }
        }
        _intersect_items.clear();
    }
    else if (_intersect_items.count(name)) {
        for(auto &item : _intersect_items[name]) {
            fScene->removeItem(item);
            delete item;
        }
        _intersect_items.erase(name);
    }
}


void CVolumeViewer::onIntersectionChanged(std::string a, std::string b, Intersection *intersection)
{
    if (_ignore_intersect_change && intersection == _ignore_intersect_change)
        return;

    if (!_intersect_tgts.count(a) || !_intersect_tgts.count(b))
        return;

    //FIXME fix segmentation vs visible_segmentation naming and usage ..., think about dependency chain ..
    if (a == _surf_name || (_surf_name == "segmentation" && a == "visible_segmentation"))
        invalidateIntersect(b);
    else if (b == _surf_name || (_surf_name == "segmentation" && b == "visible_segmentation"))
        invalidateIntersect(a);
    
    renderIntersections();
}

void CVolumeViewer::setIntersects(const std::set<std::string> &set)
{
    _intersect_tgts = set;
    
    renderIntersections();
}

void CVolumeViewer::setIntersectionOpacity(float opacity)
{
    _intersectionOpacity = std::clamp(opacity, 0.0f, 1.0f);
    for (auto& pair : _intersect_items) {
        for (auto* item : pair.second) {
            if (item) {
                item->setOpacity(_intersectionOpacity);
            }
        }
    }
}

void CVolumeViewer::setOverlayVolume(std::shared_ptr<Volume> volume)
{
    if (_overlayVolume == volume) {
        return;
    }
    _overlayVolume = std::move(volume);

    renderVisible(true);
}

void CVolumeViewer::setOverlayOpacity(float opacity)
{
    float clamped = std::clamp(opacity, 0.0f, 1.0f);
    if (std::abs(clamped - _overlayOpacity) < 1e-6f) {
        return;
    }
    _overlayOpacity = clamped;
    if (_overlayVolume) {
        renderVisible(true);
    }
}

void CVolumeViewer::setOverlayColormap(const std::string& colormapId)
{
    if (_overlayColormapId == colormapId) {
        return;
    }
    _overlayColormapId = colormapId;
    if (_overlayVolume) {
        renderVisible(true);
    }
}

void CVolumeViewer::setOverlayThreshold(float threshold)
{
    setOverlayWindow(std::max(threshold, 0.0f), _overlayWindowHigh);
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

void CVolumeViewer::setOverlayWindow(float low, float high)
{
    constexpr float kMaxOverlayValue = 255.0f;
    const float clampedLow = std::clamp(low, 0.0f, kMaxOverlayValue);
    float clampedHigh = std::clamp(high, 0.0f, kMaxOverlayValue);
    if (clampedHigh <= clampedLow) {
        clampedHigh = std::min(kMaxOverlayValue, clampedLow + 1.0f);
    }

    const bool unchanged = std::abs(clampedLow - _overlayWindowLow) < 1e-6f &&
                           std::abs(clampedHigh - _overlayWindowHigh) < 1e-6f;
    if (unchanged) {
        return;
    }

    _overlayWindowLow = clampedLow;
    _overlayWindowHigh = clampedHigh;

    if (_overlayVolume) {
        renderVisible(true);
    }
}

const std::vector<CVolumeViewer::OverlayColormapEntry>& CVolumeViewer::overlayColormapEntries()
{
    static std::vector<OverlayColormapEntry> entries;
    static bool initialized = false;
    if (!initialized) {
        const auto& sharedEntries = volume_viewer_cmaps::entries();
        entries.reserve(sharedEntries.size());
        for (const auto& entry : sharedEntries) {
            entries.push_back({entry.label, entry.id});
        }
        initialized = true;
    }
    return entries;
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
    } else if (auto* opChain = dynamic_cast<OpChain*>(_surf)) {
        QuadSurface* src = opChain->src();
        if (src) {
            bbox = src->bbox();
            haveBounds = true;
        }
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
    round_scale(_scale);
    recalcScales();

    fGraphicsView->resetTransform();
    fGraphicsView->centerOn(0, 0);

    _lbl->setText(QString("%1x %2").arg(_scale).arg(_z_off));
    curr_img_area = {0,0,0,0};
}


void CVolumeViewer::onSurfaceChanged(std::string name, Surface *surf)
{
    if (_surf_name == name) {
        _surf = surf;
        if (!_surf) {
            clearAllOverlayGroups();
            fScene->clear();
            _intersect_items.clear();
            slice_vis_items.clear();
            _paths.clear();
            emit overlaysUpdated();
            _cursor = nullptr;
            _center_marker = nullptr;
            fBaseImageItem = nullptr;
        }
        else {
            invalidateVis();
            _z_off = 0.0f;
            if (name == "segmentation" && _resetViewOnSurfaceChange) {
                fitSurfaceInView();
            }
        }
    }

    if (name == _surf_name) {
        curr_img_area = {0,0,0,0};
        renderVisible(true); // Immediate render of slice
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
        
        PlaneSurface *slice_plane = dynamic_cast<PlaneSurface*>(_surf);
        // QuadSurface *crop = dynamic_cast<QuadSurface*>(_surf_col->surface("visible_segmentation"));
        QuadSurface *crop = dynamic_cast<QuadSurface*>(_surf_col->surface("segmentation"));
        
        cv::Vec3f sp;
        float dist = -1;
        if (slice_plane) {            
            dist = slice_plane->pointDist(poi->p);
            sp = slice_plane->project(poi->p, 1.0, _scale);
        }
        else if (_surf_name == "segmentation" && crop)
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

cv::Mat_<uint8_t> CVolumeViewer::render_composite(const cv::Rect &roi) {
    cv::Mat_<uint8_t> img;

    // Composite rendering for segmentation view
    cv::Mat_<float> accumulator;
    int count = 0;

    // Alpha composition state for each pixel
    cv::Mat_<float> alpha_accumulator;
    cv::Mat_<float> value_accumulator;

    // Alpha composition parameters using the new settings
    const float alpha_min = _composite_alpha_min / 255.0f;
    const float alpha_max = _composite_alpha_max / 255.0f;
    const float alpha_opacity = _composite_material / 255.0f;
    const float alpha_cutoff = _composite_alpha_threshold / 10000.0f;

    // Determine the z range based on front and behind layers
    int z_start = _composite_reverse_direction ? -_composite_layers_behind : -_composite_layers_front;
    int z_end = _composite_reverse_direction ? _composite_layers_front : _composite_layers_behind;

    for (int z = z_start; z <= z_end; z++) {
        cv::Mat_<cv::Vec3f> slice_coords;
        cv::Mat_<uint8_t> slice_img;

        cv::Vec2f roi_c = {roi.x+roi.width/2, roi.y + roi.height/2};
        _ptr = _surf->pointer();
        cv::Vec3f diff = {roi_c[0],roi_c[1],0};
        _surf->move(_ptr, diff/_scale);
        _vis_center = roi_c;
        float z_step = z * _ds_scale;  // Scale the step to maintain consistent physical distance
        _surf->gen(&slice_coords, nullptr, roi.size(), _ptr, _scale, {-roi.width/2, -roi.height/2, _z_off + z_step});

        readInterpolated3D(slice_img, volume->zarrDataset(_ds_sd_idx), slice_coords*_ds_scale, cache, _useFastInterpolation);

        // Convert to float for accumulation
        cv::Mat_<float> slice_float;
        slice_img.convertTo(slice_float, CV_32F);

        if (_composite_method == "alpha") {
            // Alpha composition algorithm
            if (alpha_accumulator.empty()) {
                alpha_accumulator = cv::Mat_<float>::zeros(slice_float.size());
                value_accumulator = cv::Mat_<float>::zeros(slice_float.size());
            }

            // Process each pixel
            for (int y = 0; y < slice_float.rows; y++) {
                for (int x = 0; x < slice_float.cols; x++) {
                    float pixel_value = slice_float(y, x);

                    // Normalize pixel value
                    float normalized_value = (pixel_value / 255.0f - alpha_min) / (alpha_max - alpha_min);
                    normalized_value = std::max(0.0f, std::min(1.0f, normalized_value)); // Clamp to [0,1]

                    // Skip empty areas (speed through)
                    if (normalized_value == 0.0f) {
                        continue;
                    }

                    float current_alpha = alpha_accumulator(y, x);

                    // Check alpha cutoff for early termination
                    if (current_alpha >= alpha_cutoff) {
                        continue;
                    }

                    // Calculate weight
                    float weight = (1.0f - current_alpha) * std::min(normalized_value * alpha_opacity, 1.0f);

                    // Accumulate
                    value_accumulator(y, x) += weight * normalized_value;
                    alpha_accumulator(y, x) += weight;
                }
            }
        } else {
            // Original composite methods
            if (accumulator.empty()) {
                accumulator = slice_float;
                if (_composite_method == "min") {
                    accumulator.setTo(255.0); // Initialize to max value for min operation
                    accumulator = cv::min(accumulator, slice_float);
                }
            } else {
                if (_composite_method == "max") {
                    accumulator = cv::max(accumulator, slice_float);
                } else if (_composite_method == "mean") {
                    accumulator += slice_float;
                    count++;
                } else if (_composite_method == "min") {
                    accumulator = cv::min(accumulator, slice_float);
                }
            }
        }
    }

    // Finalize alpha composition result
    if (_composite_method == "alpha") {
        accumulator = cv::Mat_<float>::zeros(value_accumulator.size());
        for (int y = 0; y < value_accumulator.rows; y++) {
            for (int x = 0; x < value_accumulator.cols; x++) {
                float final_value = value_accumulator(y, x) * 255.0f;
                accumulator(y, x) = std::max(0.0f, std::min(255.0f, final_value)); // Clamp to [0,255]
            }
        }
    }

    // Convert back to uint8
    if (_composite_method == "mean" && count > 0) {
        accumulator /= count;
    }
    accumulator.convertTo(img, CV_8U);
    return img;
}

cv::Mat_<uint8_t> CVolumeViewer::renderCompositeForSurface(QuadSurface* surface, cv::Size outputSize)
{
    if (!surface || !_composite_enabled || !volume) {
        return cv::Mat_<uint8_t>();
    }

    // Save current state
    float oldScale = _scale;
    cv::Vec2f oldVisCenter = _vis_center;
    Surface* oldSurf = _surf;
    float oldZOff = _z_off;
    cv::Vec3f oldPtr = _ptr;
    float oldDsScale = _ds_scale;
    int oldDsSdIdx = _ds_sd_idx;

    // Set up for surface rendering at 1:1 scale
    _surf = surface;
    _scale = 1.0f;
    _z_off = 0.0f;

    recalcScales();
    _ptr = _surf->pointer();
    cv::Rect roi(-outputSize.width/2, -outputSize.height/2,
                 outputSize.width, outputSize.height);

    _vis_center = cv::Vec2f(0, 0);

    cv::Mat_<uint8_t> result = render_composite(roi);

    _surf = oldSurf;
    _scale = oldScale;
    _vis_center = oldVisCenter;
    _z_off = oldZOff;
    _ptr = oldPtr;
    _ds_scale = oldDsScale;
    _ds_sd_idx = oldDsSdIdx;

    return result;
}


cv::Mat CVolumeViewer::render_area(const cv::Rect &roi)
{
    cv::Mat_<cv::Vec3f> coords;
    cv::Mat_<uint8_t> baseGray;
    const int baseWindowLowInt = static_cast<int>(std::clamp(_baseWindowLow, 0.0f, 255.0f));
    const int baseWindowHighInt = static_cast<int>(
        std::clamp(_baseWindowHigh, static_cast<float>(baseWindowLowInt + 1), 255.0f));
    const float baseWindowSpan = std::max(1.0f, static_cast<float>(baseWindowHighInt - baseWindowLowInt));

    _overlayImageValid = false;
    _overlayImage = QImage();

    const QRect roiRect(roi.x, roi.y, roi.width, roi.height);

    const bool useComposite = (_surf_name == "segmentation" && _composite_enabled &&
                               (_composite_layers_front > 0 || _composite_layers_behind > 0));

    cv::Mat baseColor;
    bool usedCache = false;
    AxisAlignedSliceCacheKey cacheKey{};
    bool cacheKeyValid = false;

    z5::Dataset* baseDataset = volume ? volume->zarrDataset(_ds_sd_idx) : nullptr;

    if (useComposite) {
        baseGray = render_composite(roi);
    } else {
        if (auto* plane = dynamic_cast<PlaneSurface*>(_surf)) {
            _surf->gen(&coords, nullptr, roi.size(), cv::Vec3f(0, 0, 0), _scale, {roi.x, roi.y, _z_off});

            uint8_t planeId = 0;
            if (plane->axisAlignedRotationKey() >= 0 && cache && baseDataset &&
                planeIdForSurface(_surf_name, planeId)) {
                cacheKey.planeId = planeId;
                cacheKey.rotationKey = static_cast<uint16_t>(plane->axisAlignedRotationKey());
                const cv::Vec3f origin = plane->origin();
                cacheKey.originX = static_cast<int>(std::lround(origin[0]));
                cacheKey.originY = static_cast<int>(std::lround(origin[1]));
                cacheKey.originZ = static_cast<int>(std::lround(origin[2]));
                cacheKey.roiX = roi.x;
                cacheKey.roiY = roi.y;
                cacheKey.roiWidth = roi.width;
                cacheKey.roiHeight = roi.height;
                cacheKey.scaleMilli = quantizeFloat(_scale, kScaleQuantization);
                cacheKey.dsScaleMilli = quantizeFloat(_ds_scale, kDsScaleQuantization);
                cacheKey.zOffsetMilli = quantizeFloat(_z_off, kZOffsetQuantization);
                cacheKey.dsIndex = _ds_sd_idx;
                cacheKey.datasetPtr = reinterpret_cast<uintptr_t>(baseDataset);
                cacheKey.fastInterpolation = _useFastInterpolation ? 1 : 0;
                cacheKey.baseWindowLow = static_cast<uint8_t>(baseWindowLowInt);
                cacheKey.baseWindowHigh = static_cast<uint8_t>(baseWindowHighInt);
                cacheKeyValid = true;

                if (auto cached = axisAlignedSliceCache().get(cacheKey)) {
                    baseColor = *cached;
                    usedCache = !baseColor.empty();
                }
            }
        } else {
            cv::Vec2f roi_c = {roi.x + roi.width / 2.0f, roi.y + roi.height / 2.0f};
            _ptr = _surf->pointer();
            cv::Vec3f diff = {roi_c[0], roi_c[1], 0};
            _surf->move(_ptr, diff / _scale);
            _vis_center = roi_c;
            _surf->gen(&coords, nullptr, roi.size(), _ptr, _scale, {-roi.width / 2.0f, -roi.height / 2.0f, _z_off});
        }

        if (!usedCache) {
            if (!baseDataset) {
                return cv::Mat();
            }
            readInterpolated3D(baseGray, baseDataset, coords * _ds_scale, cache, _useFastInterpolation);
        }
    }

    if (!usedCache && baseGray.empty()) {
        return cv::Mat();
    }

    if (!usedCache) {
        cv::Mat baseFloat;
        baseGray.convertTo(baseFloat, CV_32F);
        baseFloat -= static_cast<float>(baseWindowLowInt);
        baseFloat /= baseWindowSpan;
        cv::max(baseFloat, 0.0f, baseFloat);
        cv::min(baseFloat, 1.0f, baseFloat);
        baseFloat.convertTo(baseGray, CV_8U, 255.0f);

        if (baseGray.channels() == 1) {
            cv::cvtColor(baseGray, baseColor, cv::COLOR_GRAY2BGR);
        } else {
            baseColor = baseGray.clone();
        }

        if (cacheKeyValid && !baseColor.empty()) {
            axisAlignedSliceCache().put(cacheKey, baseColor);
        }
    }

    if (_overlayVolume && _overlayOpacity > 0.0f) {
        if (coords.empty()) {
            if (auto* plane = dynamic_cast<PlaneSurface*>(_surf)) {
                _surf->gen(&coords, nullptr, roi.size(), cv::Vec3f(0, 0, 0), _scale, {roi.x, roi.y, _z_off});
            } else {
                cv::Vec2f roi_c = {roi.x + roi.width / 2.0f, roi.y + roi.height / 2.0f};
                _ptr = _surf->pointer();
                cv::Vec3f diff = {roi_c[0], roi_c[1], 0};
                _surf->move(_ptr, diff / _scale);
                _vis_center = roi_c;
                _surf->gen(&coords, nullptr, roi.size(), _ptr, _scale, {-roi.width / 2.0f, -roi.height / 2.0f, _z_off});
            }
        }

        if (!coords.empty()) {
            int overlayIdx = 0;
            float overlayScale = 1.0f;
            if (_overlayVolume->numScales() > 0) {
                overlayIdx = std::min<int>(_ds_sd_idx, static_cast<int>(_overlayVolume->numScales()) - 1);
                overlayScale = std::pow(2.0f, -overlayIdx);
            }

            cv::Mat_<uint8_t> overlayValues;
            z5::Dataset* overlayDataset = _overlayVolume->zarrDataset(overlayIdx);
            readInterpolated3D(overlayValues, overlayDataset, coords * overlayScale, cache, /*nearest_neighbor=*/true);

            if (!overlayValues.empty()) {
                const int windowLow = static_cast<int>(std::clamp(_overlayWindowLow, 0.0f, 255.0f));
                const int windowHigh = static_cast<int>(std::clamp(_overlayWindowHigh, static_cast<float>(windowLow + 1), 255.0f));

                cv::Mat activeMask;
                cv::compare(overlayValues, windowLow, activeMask, cv::CmpTypes::CMP_GE);

                if (cv::countNonZero(activeMask) > 0) {
                    cv::Mat overlayScaled;
                    overlayValues.convertTo(overlayScaled, CV_32F);
                    overlayScaled -= static_cast<float>(windowLow);
                    overlayScaled.setTo(0.0f, overlayScaled < 0.0f);
                    const float windowSpan = std::max(1.0f, static_cast<float>(windowHigh - windowLow));
                    overlayScaled /= windowSpan;
                    cv::threshold(overlayScaled, overlayScaled, 1.0f, 1.0f, cv::THRESH_TRUNC);

                    cv::Mat overlayColorInput;
                    overlayScaled.convertTo(overlayColorInput, CV_8U, 255.0f);

                    const auto& spec = volume_viewer_cmaps::resolve(_overlayColormapId);
                    cv::Mat overlayColor = volume_viewer_cmaps::makeColors(overlayColorInput, spec);

                    if (!overlayColor.empty()) {
                        cv::Mat inactiveMask;
                        cv::bitwise_not(activeMask, inactiveMask);
                        overlayColor.setTo(cv::Scalar(0, 0, 0), inactiveMask);

                        cv::Mat overlayBGRA;
                        cv::cvtColor(overlayColor, overlayBGRA, cv::COLOR_BGR2BGRA);

                        std::vector<cv::Mat> channels;
                        cv::split(overlayBGRA, channels);
                        const uchar alphaValue = static_cast<uchar>(std::round(std::clamp(_overlayOpacity, 0.0f, 1.0f) * 255.0f));
                        channels[3].setTo(alphaValue, activeMask);
                        channels[3].setTo(0, inactiveMask);
                        cv::merge(channels, overlayBGRA);

                        cv::cvtColor(overlayBGRA, overlayBGRA, cv::COLOR_BGRA2RGBA);
                        QImage overlayImage(overlayBGRA.data, overlayBGRA.cols, overlayBGRA.rows, overlayBGRA.step, QImage::Format_RGBA8888);
                        _overlayImage = overlayImage.copy();
                        _overlayImageValid = true;
                    }
                }
            }
        }
    }

    return baseColor;
}

class LifeTime
{
public:
    LifeTime(std::string msg)
    {
        std::cout << msg << std::flush;
        start = std::chrono::high_resolution_clock::now();
    }
    ~LifeTime()
    {
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << " took " << std::chrono::duration<double>(end-start).count() << " s" << std::endl;
    }
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
};

void CVolumeViewer::renderVisible(bool force)
{
    if (_surf && _surf_col) {
        Surface* currentSurface = _surf_col->surface(_surf_name);
        if (!currentSurface) {
            // Surface was cleared (e.g. during volume reload) without a change signal
            // reaching this viewer yet; drop the dangling pointer before rendering.
            _surf = nullptr;
        }
    }

    if (!volume || !volume->zarrDataset() || !_surf)
        return;

    QRectF bbox = fGraphicsView->mapToScene(fGraphicsView->viewport()->geometry()).boundingRect();
    
    if (!force && QRectF(curr_img_area).contains(bbox))
        return;
    
    
    curr_img_area = {bbox.left(),bbox.top(), bbox.width(), bbox.height()};
    
    cv::Mat img = render_area({curr_img_area.x(), curr_img_area.y(), curr_img_area.width(), curr_img_area.height()});
    
    QImage qimg = Mat2QImage(img);
    if (_overlayImageValid && !_overlayImage.isNull()) {
        qimg = qimg.convertToFormat(QImage::Format_RGBA8888);
        QPainter painter(&qimg);
        painter.setCompositionMode(QPainter::CompositionMode_SourceOver);
        painter.drawImage(0, 0, _overlayImage);
    }

    QPixmap pixmap = QPixmap::fromImage(qimg, fSkipImageFormatConv ? Qt::NoFormatConversion : Qt::AutoColor);
 
    // Add the QPixmap to the scene as a QGraphicsPixmapItem
    if (!fBaseImageItem)
        fBaseImageItem = fScene->addPixmap(pixmap);
    else
        fBaseImageItem->setPixmap(pixmap);
    
    if (!_center_marker) {
        _center_marker = fScene->addEllipse({-10,-10,20,20}, QPen(COLOR_FOCUS, 3, Qt::DashDotLine, Qt::RoundCap, Qt::RoundJoin));
        _center_marker->setZValue(11);
    }

    _center_marker->setParentItem(fBaseImageItem);
    
    fBaseImageItem->setOffset(curr_img_area.topLeft());
}



void CVolumeViewer::renderIntersections()
{
    if (!volume || !volume->zarrDataset() || !_surf)
        return;
    
    std::vector<std::string> remove;
    for (auto &pair : _intersect_items)
        if (!_intersect_tgts.count(pair.first)) {
            for(auto &item : pair.second) {
                fScene->removeItem(item);
                delete item;
            }
            remove.push_back(pair.first);
        }
    for(auto key : remove)
        _intersect_items.erase(key);

    PlaneSurface *plane = dynamic_cast<PlaneSurface*>(_surf);

    
    if (plane) {
        cv::Rect plane_roi = {curr_img_area.x()/_scale, curr_img_area.y()/_scale, curr_img_area.width()/_scale, curr_img_area.height()/_scale};

        cv::Vec3f corner = plane->coord(cv::Vec3f(0,0,0), {plane_roi.x, plane_roi.y, 0.0});
        Rect3D view_bbox = {corner, corner};
        view_bbox = expand_rect(view_bbox, plane->coord(cv::Vec3f(0,0,0), {plane_roi.br().x, plane_roi.y, 0}));
        view_bbox = expand_rect(view_bbox, plane->coord(cv::Vec3f(0,0,0), {plane_roi.x, plane_roi.br().y, 0}));
        view_bbox = expand_rect(view_bbox, plane->coord(cv::Vec3f(0,0,0), {plane_roi.br().x, plane_roi.br().y, 0}));

        std::vector<std::string> intersect_cands;
        std::vector<std::string> intersect_tgts_v;

        for (auto key : _intersect_tgts)
            intersect_tgts_v.push_back(key);

#pragma omp parallel for
        for(int n=0;n<intersect_tgts_v.size();n++) {
            std::string key = intersect_tgts_v[n];
            bool haskey;
#pragma omp critical
            haskey = _intersect_items.count(key);
            if (!haskey && dynamic_cast<QuadSurface*>(_surf_col->surface(key))) {
                QuadSurface *segmentation = dynamic_cast<QuadSurface*>(_surf_col->surface(key));

                if (intersect(view_bbox, segmentation->bbox()))
#pragma omp critical
                    intersect_cands.push_back(key);
                else
#pragma omp critical
                    _intersect_items[key] = {};
            }
        }

        std::vector<std::vector<std::vector<cv::Vec3f>>> intersections(intersect_cands.size());

#pragma omp parallel for
        for(int n=0;n<intersect_cands.size();n++) {
            std::string key = intersect_cands[n];
            QuadSurface *segmentation = dynamic_cast<QuadSurface*>(_surf_col->surface(key));

            std::vector<std::vector<cv::Vec2f>> xy_seg_;
            if (key == "segmentation") {
                find_intersect_segments(intersections[n], xy_seg_, segmentation->rawPoints(), plane, plane_roi, 4/_scale, 1000);
            }
            else
                find_intersect_segments(intersections[n], xy_seg_, segmentation->rawPoints(), plane, plane_roi, 4/_scale);

        }

        std::hash<std::string> str_hasher;

        for(int n=0;n<intersect_cands.size();n++) {
            std::string key = intersect_cands[n];

            if (!intersections.size()) {
                _intersect_items[key] = {};
                continue;
            }

            size_t seed = str_hasher(key);
            srand(seed);

            int prim = rand() % 3;
            cv::Vec3i cvcol = {100 + rand() % 255, 100 + rand() % 255, 100 + rand() % 255};
            cvcol[prim] = 200 + rand() % 55;

            QColor col(cvcol[0],cvcol[1],cvcol[2]);
            float width = 2;
            int z_value = 5;

            if (key == "segmentation") {
                col =
                    (_surf_name == "seg yz"   ? COLOR_SEG_YZ
                     : _surf_name == "seg xz" ? COLOR_SEG_XZ
                                              : COLOR_SEG_XY);
                width = 3;
                z_value = 20;
            }


            QuadSurface *segmentation = dynamic_cast<QuadSurface*>(_surf_col->surface(intersect_cands[n]));
            std::vector<QGraphicsItem*> items;

            int len = 0;
            for (auto seg : intersections[n]) {
                QPainterPath path;

                bool first = true;
                cv::Vec3f last = {-1,-1,-1};
                for (auto wp : seg)
                {
                    len++;
                    cv::Vec3f p = plane->project(wp, 1.0, _scale);

                    if (last[0] != -1 && cv::norm(p-last) >= 8) {
                        auto item = fGraphicsView->scene()->addPath(path, QPen(col, width));
                        item->setZValue(z_value);
                        item->setOpacity(_intersectionOpacity);
                        items.push_back(item);
                        first = true;
                    }
                    last = p;

                    if (first)
                        path.moveTo(p[0],p[1]);
                    else
                        path.lineTo(p[0],p[1]);
                    first = false;
                }
                auto item = fGraphicsView->scene()->addPath(path, QPen(col, width));
                item->setZValue(z_value);
                item->setOpacity(_intersectionOpacity);
                items.push_back(item);
            }
            _intersect_items[key] = items;
            _ignore_intersect_change = new Intersection({intersections[n]});
            _surf_col->setIntersection(_surf_name, key, _ignore_intersect_change);
            _ignore_intersect_change = nullptr;
        }
    }
    else if (_surf_name == "segmentation" /*&& dynamic_cast<QuadSurface*>(_surf_col->surface("visible_segmentation"))*/) {
        // QuadSurface *crop = dynamic_cast<QuadSurface*>(_surf_col->surface("visible_segmentation"));

        //TODO make configurable, for now just show everything!
        std::vector<std::pair<std::string,std::string>> intersects = _surf_col->intersections("segmentation");
        for(auto pair : intersects) {
            std::string key = pair.first;
            if (key == "segmentation")
                key = pair.second;
            
            if (_intersect_items.count(key) || !_intersect_tgts.count(key))
                continue;
            
            std::unordered_map<cv::Vec3f,cv::Vec3f,vec3f_hash> location_cache;
            std::vector<cv::Vec3f> src_locations;

            for (auto seg : _surf_col->intersection(pair.first, pair.second)->lines)
                for (auto wp : seg)
                    src_locations.push_back(wp);
            
#pragma omp parallel
            {
                // SurfacePointer *ptr = crop->pointer();
                auto ptr = _surf->pointer();
#pragma omp for
                for (auto wp : src_locations) {
                    // float res = crop->pointTo(ptr, wp, 2.0, 100);
                    // cv::Vec3f p = crop->loc(ptr)*_ds_scale + cv::Vec3f(_vis_center[0],_vis_center[1],0);
                    float res = _surf->pointTo(ptr, wp, 2.0, 100);
                    cv::Vec3f p = _surf->loc(ptr)*_scale ;//+ cv::Vec3f(_vis_center[0],_vis_center[1],0);
                    //FIXME still happening?
                    if (res >= 2.0)
                        p = {-1,-1,-1};
                        // std::cout << "WARNING pointTo() high residual in renderIntersections()" << std::endl;
#pragma omp critical
                    location_cache[wp] = p;
                }
            }
            
            std::vector<QGraphicsItem*> items;
            for (auto seg : _surf_col->intersection(pair.first, pair.second)->lines) {
                QPainterPath path;
                
                bool first = true;
                cv::Vec3f last = {-1,-1,-1};
                for (auto wp : seg)
                {
                    cv::Vec3f p = location_cache[wp];
                    
                    if (p[0] == -1)
                        continue;

                    if (last[0] != -1 && cv::norm(p-last) >= 8) {
                        auto item = fGraphicsView->scene()->addPath(path, QPen(key == "seg yz" ? COLOR_SEG_YZ: COLOR_SEG_XZ, 2));
                        item->setZValue(5);
                        item->setOpacity(_intersectionOpacity);
                        items.push_back(item);
                        first = true;
                    }
                    last = p;

                    if (first)
                        path.moveTo(p[0],p[1]);
                    else
                        path.lineTo(p[0],p[1]);
                    first = false;
                }
                auto item = fGraphicsView->scene()->addPath(path, QPen(key == "seg yz" ? COLOR_SEG_YZ: COLOR_SEG_XZ, 2));
                item->setZValue(5);
                item->setOpacity(_intersectionOpacity);
                items.push_back(item);
            }
            _intersect_items[key] = items;
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
    // if (!dynamic_cast<OpChain*>(_surf) && !dynamic_cast<OpChain*>(_surf)->slow() && _min_scale == 1.0)
        // renderVisible();
    // if ((!dynamic_cast<OpChain*>(_surf) || !dynamic_cast<OpChain*>(_surf)->slow()) && _min_scale < 1.0)
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
        if (dynamic_cast<PlaneSurface*>(_surf))
            emit sendMouseReleaseVolume(p, button, modifiers);
        else if (_surf_name == "segmentation")
            emit sendMouseReleaseVolume(p, button, modifiers);
        else
            std::cout << "FIXME: onMouseRelease()" << std::endl;
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

void CVolumeViewer::updateAllOverlays()
{
    if (auto* plane = dynamic_cast<PlaneSurface*>(_surf)) {
        POI *poi = _surf_col->poi("focus");
        if (poi) {
            cv::Vec3f planeOrigin = plane->origin();
            // If plane origin differs from POI, update POI
            if (std::abs(poi->p[2] - planeOrigin[2]) > 0.01) {
                poi->p = planeOrigin;
                _surf_col->setPOI("focus", poi);  // NOW we do the expensive update
                emit sendZSliceChanged(static_cast<int>(poi->p[2]));
            }
        }
    }

    QPoint viewportPos = fGraphicsView->mapFromGlobal(QCursor::pos());
    QPointF scenePos = fGraphicsView->mapToScene(viewportPos);

    cv::Vec3f p, n;
    if (scene2vol(p, n, _surf, _surf_name, _surf_col, scenePos, _vis_center, _scale)) {
        POI *cursor = _surf_col->poi("cursor");
        if (!cursor)
            cursor = new POI;
        cursor->p = p;
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
                QPointF diff = scenePos - point_scene_pos;
                float dist_sq = QPointF::dotProduct(diff, diff);
                if (dist_sq < min_dist_sq) {
                    min_dist_sq = dist_sq;
                    _highlighted_point_id = point_pair.second.id;
                }
            }
        }

    }

    invalidateVis();
    invalidateIntersect();
    renderIntersections();

    emit overlaysUpdated();
}

void CVolumeViewer::setOverlayGroup(const std::string& key, const std::vector<QGraphicsItem*>& items)
{
    // Remove and delete existing items in the group
    clearOverlayGroup(key);
    _overlay_groups[key] = items;
}

// Visualize the 'step' parameter used by vc_grow_seg_from_segments by placing
// three small markers in either direction along the same direction arrows.

void CVolumeViewer::clearOverlayGroup(const std::string& key)
{
    auto it = _overlay_groups.find(key);
    if (it == _overlay_groups.end()) return;
    for (auto* item : it->second) {
        if (!item) continue;
        if (auto* scene = item->scene()) {
            scene->removeItem(item);
        } else if (fScene) {
            fScene->removeItem(item);
        }
        delete item;
    }
    _overlay_groups.erase(it);
}

void CVolumeViewer::clearAllOverlayGroups()
{
    if (_overlay_groups.empty()) {
        return;
    }

    for (auto& entry : _overlay_groups) {
        for (auto* item : entry.second) {
            if (!item) {
                continue;
            }
            if (auto* scene = item->scene()) {
                scene->removeItem(item);
            } else if (fScene) {
                fScene->removeItem(item);
            }
            delete item;
        }
    }
    _overlay_groups.clear();
}

// Draw two small arrows indicating growth direction candidates:
// red = flip_x=false (along +X)
// green = flip_x=true (opposite −X)
// Shown on segmentation and projected into slice views.
