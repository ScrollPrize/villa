#include "CTiledVolumeViewer.hpp"

#include "ViewerManager.hpp"
#include "VCSettings.hpp"
#include "VolumeViewerCmaps.hpp"
#include "../CState.hpp"
#include "../overlays/SegmentationOverlayController.hpp"
#include "vc/ui/VCCollection.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/render/PostProcess.hpp"
#include "vc/core/types/SampleParams.hpp"
#include <limits>

#include <QSettings>
#include <QTimer>
#include <QVBoxLayout>
#include <QLabel>
#include <QPainter>
#include <QGraphicsView>
#include <QGraphicsScene>
#include <QGraphicsItem>
#include <QGraphicsPathItem>
#include <QGraphicsPixmapItem>
#include <QGraphicsEllipseItem>
#include <QMdiSubWindow>
#include <QPainterPath>
#include <QWindowStateChangeEvent>
#include <QPointer>

#include <algorithm>
#include <array>
#include <cmath>

#include "vc/core/cache/TieredChunkCache.hpp"
#include "vc/core/render/PostProcess.hpp"
#include "vc/core/render/Colormaps.hpp"

namespace {

void blendOverlay(uint32_t* base, int baseStride,
                  const uint32_t* overlay, int overlayStride,
                  const cv::Mat_<uint8_t>& overlayMask,
                  float opacity, int rows, int cols)
{
    const float alpha = std::clamp(opacity, 0.0f, 1.0f);
    if (alpha <= 0.0f) return;
    const float ia = 1.0f - alpha;
    for (int y = 0; y < rows; ++y) {
        auto* dst = base + y * baseStride;
        const auto* src = overlay + y * overlayStride;
        const auto* mask = overlayMask.ptr<uint8_t>(y);
        for (int x = 0; x < cols; ++x) {
            if (mask[x] == 0) continue;
            const uint32_t d = dst[x], s = src[x];
            const auto r = static_cast<uint32_t>(static_cast<float>((s >> 16) & 0xFF) * alpha + static_cast<float>((d >> 16) & 0xFF) * ia);
            const auto g = static_cast<uint32_t>(static_cast<float>((s >> 8) & 0xFF) * alpha + static_cast<float>((d >> 8) & 0xFF) * ia);
            const auto b = static_cast<uint32_t>(static_cast<float>(s & 0xFF) * alpha + static_cast<float>(d & 0xFF) * ia);
            dst[x] = 0xFF000000u | (r << 16) | (g << 8) | b;
        }
    }
}

} // namespace

constexpr auto COLOR_CURSOR = Qt::cyan;
#define COLOR_FOCUS QColor(50, 255, 215)
#define COLOR_SEG_YZ Qt::yellow
#define COLOR_SEG_XZ Qt::red
#define COLOR_SEG_XY QColor(255, 140, 0)

static const QColor kIntersectionPalette[] = {
    // Vibrant saturated colors
    QColor(80, 180, 255),  // sky blue
    QColor(180, 80, 220),  // violet
    QColor(80, 220, 200),  // aqua/teal
    QColor(220, 80, 180),  // magenta
    QColor(80, 130, 255),  // medium blue
    QColor(160, 80, 255),  // purple
    QColor(80, 255, 220),  // cyan
    QColor(255, 80, 200),  // hot pink
    QColor(120, 220, 80),  // lime green
    QColor(80, 180, 120),  // spring green
    // Lighter/pastel variants
    QColor(150, 200, 255),  // light sky blue
    QColor(200, 150, 230),  // light violet
    QColor(150, 230, 210),  // light aqua
    QColor(230, 150, 200),  // light magenta
    QColor(150, 170, 255),  // light blue
    QColor(190, 150, 255),  // light purple
    QColor(150, 255, 230),  // light cyan
    QColor(255, 150, 210),  // light pink
    QColor(180, 240, 150),  // light lime
    QColor(150, 230, 170),  // light spring green
    // Deeper/darker variants
    QColor(50, 120, 200),  // deep blue
    QColor(140, 50, 180),  // deep violet
    QColor(50, 180, 160),  // deep teal
    QColor(180, 50, 140),  // deep magenta
    QColor(50, 90, 200),   // navy blue
    QColor(120, 50, 200),  // deep purple
    QColor(50, 200, 180),  // deep cyan
    QColor(200, 50, 160),  // deep pink
    QColor(80, 160, 60),   // forest green
    QColor(50, 140, 100),  // deep sea green
    // Extra variations with different saturation
    QColor(100, 160, 220),  // muted blue
    QColor(160, 100, 200),  // muted violet
    QColor(100, 200, 180),  // muted teal
    QColor(200, 100, 170),  // muted magenta
    QColor(120, 180, 240),  // soft blue
    QColor(180, 120, 220),  // soft purple
    QColor(120, 220, 200),  // soft cyan
    QColor(220, 120, 190),  // soft pink
    QColor(140, 200, 100),  // soft lime
    QColor(100, 180, 130),  // muted green
};

namespace
{
constexpr qreal kIntersectionZ = 18.0;
constexpr qreal kActiveSegZ = 20.0;
constexpr qreal kHighlightZ = 22.0;
constexpr auto kIntersectionItemsKey = "__plane_intersections__";

bool isFinitePoint(const QPointF& point)
{
    return std::isfinite(point.x()) && std::isfinite(point.y());
}

cv::Rect planeRoiFromSceneRect(TileScene* tileScene, const QRectF& sceneRect)
{
    if (!tileScene || !sceneRect.isValid()) {
        return {};
    }

    const std::array<QPointF, 4> corners = {
        sceneRect.topLeft(),
        sceneRect.topRight(),
        sceneRect.bottomLeft(),
        sceneRect.bottomRight(),
    };

    float minX = std::numeric_limits<float>::max();
    float minY = std::numeric_limits<float>::max();
    float maxX = std::numeric_limits<float>::lowest();
    float maxY = std::numeric_limits<float>::lowest();

    for (const auto& corner : corners) {
        const cv::Vec2f surfacePoint = tileScene->sceneToSurface(corner);
        minX = std::min(minX, surfacePoint[0]);
        minY = std::min(minY, surfacePoint[1]);
        maxX = std::max(maxX, surfacePoint[0]);
        maxY = std::max(maxY, surfacePoint[1]);
    }

    const int x = static_cast<int>(std::floor(minX));
    const int y = static_cast<int>(std::floor(minY));
    const int right = static_cast<int>(std::ceil(maxX));
    const int bottom = static_cast<int>(std::ceil(maxY));
    return {x, y, std::max(1, right - x), std::max(1, bottom - y)};
}

std::pair<int, int> surfaceParamToGrid(const QuadSurface* surface, const cv::Vec3f& param)
{
    if (!surface) {
        return {0, 0};
    }

    const cv::Vec3f center = surface->center();
    const cv::Vec2f scale = surface->scale();
    const int col = static_cast<int>(std::lround(param[0] + center[0] * scale[0]));
    const int row = static_cast<int>(std::lround(param[1] + center[1] * scale[1]));
    return {row, col};
}
}  // namespace

// ============================================================================
// Construction / destruction
// ============================================================================

CTiledVolumeViewer::CTiledVolumeViewer(CState* state,
                                       ViewerManager* manager,
                                       QWidget* parent)
    : QWidget(parent)
    , _state(state)
    , _viewerManager(manager)
{
    _compositeSettings.params.method = "max";
    _compositeSettings.params.alphaMin = 170 / 255.0f;
    _compositeSettings.params.alphaMax = 220 / 255.0f;
    _compositeSettings.params.alphaOpacity = 230 / 255.0f;
    _compositeSettings.params.alphaCutoff = 9950 / 10000.0f;

    // Create graphics view with scrollbars disabled
    fGraphicsView = new CVolumeViewerView(this);
    fGraphicsView->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    fGraphicsView->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    fGraphicsView->setTransformationAnchor(QGraphicsView::NoAnchor);
    fGraphicsView->setRenderHint(QPainter::Antialiasing, false);
    fGraphicsView->setScrollPanDisabled(true);
    // Software rendering — no OpenGL (eliminates ghosting from GL double-buffer)
    fGraphicsView->setViewportUpdateMode(QGraphicsView::FullViewportUpdate);

    // Connect signals from view
    connect(fGraphicsView, &CVolumeViewerView::sendScrolled, this, &CTiledVolumeViewer::onScrolled);
    connect(fGraphicsView, &CVolumeViewerView::sendVolumeClicked, this, &CTiledVolumeViewer::onVolumeClicked);
    connect(fGraphicsView, &CVolumeViewerView::sendZoom, this, &CTiledVolumeViewer::onZoom);
    connect(fGraphicsView, &CVolumeViewerView::sendResized, this, &CTiledVolumeViewer::onResized);
    connect(fGraphicsView, &CVolumeViewerView::sendCursorMove, this, &CTiledVolumeViewer::onCursorMove);
    connect(fGraphicsView, &CVolumeViewerView::sendPanRelease, this, &CTiledVolumeViewer::onPanRelease);
    connect(fGraphicsView, &CVolumeViewerView::sendPanStart, this, &CTiledVolumeViewer::onPanStart);
    connect(fGraphicsView, &CVolumeViewerView::sendMousePress, this, &CTiledVolumeViewer::onMousePress);
    connect(fGraphicsView, &CVolumeViewerView::sendMouseMove, this, &CTiledVolumeViewer::onMouseMove);
    connect(fGraphicsView, &CVolumeViewerView::sendMouseRelease, this, &CTiledVolumeViewer::onMouseRelease);
    connect(fGraphicsView, &CVolumeViewerView::sendKeyPress, this, &CTiledVolumeViewer::onKeyPress);
    connect(fGraphicsView, &CVolumeViewerView::sendKeyRelease, this, &CTiledVolumeViewer::onKeyRelease);

    // Create fixed-size scene
    _scene = new QGraphicsScene(this);
    // Disable BSP index — with hundreds of tile items at fixed positions,
    // the BSP tree overhead exceeds its benefit. NoIndex is faster for
    // scenes with frequent pixmap updates.
    _scene->setItemIndexMethod(QGraphicsScene::NoIndex);

    // Create tile scene manager
    _tileScene = new TileScene(_scene);

    // Set the scene on the view
    fGraphicsView->setScene(_scene);

    // Paint framebuffer directly in drawBackground, bypassing QGraphicsPixmapItem
    fGraphicsView->setDirectFramebuffer(&_tileScene->constFramebuffer());

    // Render throttle: coalesce all render requests into one per 16ms
    _renderTimer = new QTimer(this);
    _renderTimer->setSingleShot(true);
    _renderTimer->setInterval(16);
    connect(_renderTimer, &QTimer::timeout, this, [this]() {
        if (_renderPending) {
            _renderPending = false;
            submitRender();
            updateStatusLabel();
        }
    });

    // Read settings
    using namespace vc3d::settings;
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    _camera.downscaleOverride = settings.value(perf::DOWNSCALE_OVERRIDE, perf::DOWNSCALE_OVERRIDE_DEFAULT).toInt();
    _navSpeed = settings.value(viewer::NAV_SPEED, viewer::NAV_SPEED_DEFAULT).toFloat();
    if (_navSpeed <= 0.0f) _navSpeed = 1.0f;

    auto* layout = new QVBoxLayout;
    layout->addWidget(fGraphicsView);
    setLayout(layout);

    _lbl = new QLabel(this);
    _lbl->setStyleSheet("QLabel { color : #00FF00; background-color: rgba(0,0,0,128); padding: 2px 4px; }");
    _lbl->setMinimumWidth(300);
    _lbl->move(10, 5);

    // Throttle intersection recomputation during interaction
    _intersectionThrottleTimer = new QTimer(this);
    _intersectionThrottleTimer->setSingleShot(true);
    _intersectionThrottleTimer->setInterval(1000);
    connect(_intersectionThrottleTimer, &QTimer::timeout, this, [this]() {
        if (_intersectionsDirty) {
            _intersectionsDirty = false;
            renderIntersectionsCore();
        }
    });

    // Async intersection completion handler
    connect(&_intersectionFutureWatcher,
            &QFutureWatcher<std::vector<IntersectionPathEntry>>::finished,
            this, &CTiledVolumeViewer::onIntersectionComputeFinished);

}

CTiledVolumeViewer::~CTiledVolumeViewer()
{
    _intersectionFutureWatcher.disconnect();
    _intersectionFutureWatcher.cancel();
    _intersectionThrottleTimer->stop();

    if (_chunkCbId != 0 && _volume && _volume->tieredCache()) {
        _volume->tieredCache()->removeChunkReadyListener(_chunkCbId);
        _chunkCbId = 0;
    }
    delete _tileScene;
    // fGraphicsView and _scene are parented to this QWidget and will be
    // destroyed by Qt's parent-child ownership — do not delete them here.
}

// ============================================================================
// Data setup
// ============================================================================


void CTiledVolumeViewer::setPointCollection(VCCollection* pc)
{
    _pointCollection = pc;
    scheduleOverlayUpdate();
}

void CTiledVolumeViewer::setSurface(const std::string& name)
{
    _surfName = name;
    // Don't reset _surfWeak here — onSurfaceChanged() will update it
    markActiveSegmentationDirty();
    onSurfaceChanged(name, _state->surface(name));
}

void CTiledVolumeViewer::setIntersects(const std::set<std::string>& set)
{
    _intersectTgts = set;
    renderIntersections();
}

Surface* CTiledVolumeViewer::currentSurface() const
{
    if (!_state) {
        // NOTE: The returned raw pointer is only valid as long as the caller
        // (or another shared_ptr elsewhere) keeps the Surface alive.
        // _defaultSurface holds a shared_ptr that keeps the standalone
        // surface alive for the lifetime of this viewer, so this is safe
        // as long as callers don't stash the pointer across event loops.
        auto shared = _surfWeak.lock();
        return shared ? shared.get() : nullptr;
    }
    return _state->surfaceRaw(_surfName);
}

// ============================================================================
// Volume / surface change handlers
// ============================================================================

void CTiledVolumeViewer::OnVolumeChanged(std::shared_ptr<Volume> vol)
{
    // Remove old chunk-ready listener before switching volumes
    if (_chunkCbId != 0 && _volume && _volume->tieredCache()) {
        _volume->tieredCache()->removeChunkReadyListener(_chunkCbId);
        _chunkCbId = 0;
    }

    _volume = std::move(vol);
    _hadValidDataBounds = false;

    // Wire chunk-ready listener so arriving chunks trigger re-render
    if (_volume && _volume->numScales() >= 1) {
        auto* cache = _volume->tieredCache();
        QPointer<CTiledVolumeViewer> viewerGuard(this);
        _chunkCbId = cache->addChunkReadyListener(
            [viewerGuard, cache](const vc::cache::ChunkKey&) {
                QMetaObject::invokeMethod(qApp, [viewerGuard, cache]() {
                    if (viewerGuard) {
                        cache->clearChunkArrivedFlag();
                        viewerGuard->scheduleRender();
                    }
                }, Qt::QueuedConnection);
            });
    }

    onVolumeReady();
    updateStatusLabel();
}

void CTiledVolumeViewer::onSurfaceChanged(const std::string& name, const std::shared_ptr<Surface>& surf,
                                           bool isEditUpdate)
{
    if (name == "segmentation" || name == _surfName) {
        markActiveSegmentationDirty();
    }

    if (_surfName == name) {
        auto previousSurf = _surfWeak.lock();
        const bool isInPlaceQuadEditUpdate =
            isEditUpdate &&
            surf &&
            previousSurf &&
            previousSurf.get() == surf.get() &&
            dynamic_cast<QuadSurface*>(surf.get()) != nullptr;
        const bool isInPlacePlaneUpdate =
            surf &&
            previousSurf &&
            previousSurf.get() == surf.get() &&
            dynamic_cast<PlaneSurface*>(surf.get()) != nullptr;

        _surfWeak = surf;
        _surfBBoxCache = {};  // invalidate bounding box cache
        if (!surf) {
            _surfaceContentVersion = 0;

            clearAllOverlayGroups();
            _tileScene->sceneCleared();
            _ov.cursor = nullptr;
            _ov.centerMarker = nullptr;
            _scene->clear();
            _ov.intersectItems.clear();
            _ov.sliceVisItems.clear();
            _paths.clear();
            scheduleOverlayUpdate();
            // Grid will be rebuilt when new surface is set
        } else {
            invalidateVis();
            if (isInPlaceQuadEditUpdate) {
                ++_surfaceContentVersion;
    
            } else if (isInPlacePlaneUpdate) {
                _surfaceContentVersion = 0;
    
                updateContentMinScale();
                rebuildContentGrid();
                centerViewport();
            } else {
                _surfaceContentVersion = 0;
    
                if (!isEditUpdate) {
                    _camera.zOff = 0.0f;
                }

                updateContentMinScale();
                rebuildContentGrid();
                centerViewport();
                if (name == "segmentation" && _resetViewOnSurfaceChange) {
                    fitSurfaceInView();
                }
            }
        }
    }

    if (name == _surfName) {
        _camera.invalidate();
        submitRender();
    }

    if (name == "segmentation" || name == _surfName) {
        renderIntersections();
    }
}

void CTiledVolumeViewer::onVolumeClosing()
{
    if (_surfName == "segmentation") {
        onSurfaceChanged(_surfName, nullptr);
    } else if (isAxisAlignedView()) {
        clearAllOverlayGroups();
        _tileScene->sceneCleared();
        _ov.cursor = nullptr;
        _ov.centerMarker = nullptr;
        _scene->clear();
        _ov.intersectItems.clear();
        _ov.sliceVisItems.clear();
        _paths.clear();
        scheduleOverlayUpdate();
    } else {
        onSurfaceChanged(_surfName, nullptr);
    }
}

void CTiledVolumeViewer::onSurfaceWillBeDeleted(const std::string& /*name*/, const std::shared_ptr<Surface>& surf)
{
    auto current = _surfWeak.lock();
    if (current && current == surf) {
        _surfWeak.reset();
    }
}

// ============================================================================
// Zoom limits
// ============================================================================

void CTiledVolumeViewer::updateContentMinScale()
{
    if (!fGraphicsView) return;

    QSize vpSize = fGraphicsView->viewport()->size();
    float vpW = static_cast<float>(vpSize.width());
    float vpH = static_cast<float>(vpSize.height());
    if (vpW <= 0 || vpH <= 0) return;

    float contentW = 0, contentH = 0;

    if (_volume && isAxisAlignedView()) {
        auto [w, h, d] = _volume->shape();
        if (_surfName == "xy plane") {
            contentW = static_cast<float>(w);
            contentH = static_cast<float>(h);
        } else if (_surfName == "xz plane" || _surfName == "seg xz") {
            contentW = static_cast<float>(w);
            contentH = static_cast<float>(d);
        } else {  // yz plane / seg yz
            contentW = static_cast<float>(h);
            contentH = static_cast<float>(d);
        }
    } else {
        auto surf = _surfWeak.lock();
        if (auto* quadSurf = dynamic_cast<QuadSurface*>(surf.get())) {
            const cv::Mat_<cv::Vec3f>& pts = quadSurf->rawPoints();
            cv::Vec2f sc = quadSurf->scale();
            contentW = static_cast<float>(pts.cols) / sc[0];
            contentH = static_cast<float>(pts.rows) / sc[1];
        }
    }

    if (contentW <= 0 || contentH <= 0) {
        _contentMinScale = TiledViewerCamera::MIN_SCALE;
        return;
    }

    _contentMinScale = TiledViewerCamera::MIN_SCALE;
}

void CTiledVolumeViewer::rebuildContentGrid()
{
    if (!fGraphicsView) return;

    float contentMinX = 0, contentMinY = 0, contentMaxX = 0, contentMaxY = 0;

    auto surf = _surfWeak.lock();
    if (_volume && surf) {
        if (auto* plane = dynamic_cast<PlaneSurface*>(surf.get())) {
            // Project volume bounding box corners onto the plane
            auto [w, h, d] = _volume->shape();
            float x0 = 0, x1 = static_cast<float>(w), y0 = 0, y1 = static_cast<float>(h), z0 = 0, z1 = static_cast<float>(d);
            float corners[][3] = {
                {x0,y0,z0}, {x1,y0,z0}, {x0,y1,z0}, {x1,y1,z0},
                {x0,y0,z1}, {x1,y0,z1}, {x0,y1,z1}, {x1,y1,z1}
            };
            contentMinX = contentMinY = std::numeric_limits<float>::max();
            contentMaxX = contentMaxY = std::numeric_limits<float>::lowest();
            for (auto& c : corners) {
                cv::Vec3f proj = plane->project(cv::Vec3f(c[0], c[1], c[2]), 1.0, 1.0);
                contentMinX = std::min(contentMinX, proj[0]);
                contentMinY = std::min(contentMinY, proj[1]);
                contentMaxX = std::max(contentMaxX, proj[0]);
                contentMaxY = std::max(contentMaxY, proj[1]);
            }
        } else if (auto* quad = dynamic_cast<QuadSurface*>(surf.get())) {
            const cv::Mat_<cv::Vec3f>& pts = quad->rawPoints();
            cv::Vec2f sc = quad->scale();
            float halfW = (static_cast<float>(pts.cols) * 0.5f) / sc[0];
            float halfH = (static_cast<float>(pts.rows) * 0.5f) / sc[1];
            contentMinX = -halfW;
            contentMinY = -halfH;
            contentMaxX = halfW;
            contentMaxY = halfH;
        }
    }

    // Store content extent for pan clamping
    _fullContentMinU = contentMinX;
    _fullContentMaxU = contentMaxX;
    _fullContentMinV = contentMinY;
    _fullContentMaxV = contentMaxY;

    // Resize framebuffer to viewport size
    QSize vpSize = fGraphicsView->viewport()->size();
    _tileScene->rebuildGrid(vpSize.width(), vpSize.height());
}

void CTiledVolumeViewer::centerViewport()
{
    if (!fGraphicsView || !_tileScene) return;
    _tileScene->setCamera(_camera.surfacePtr[0], _camera.surfacePtr[1], _camera.scale);
    _tileScene->setCamZOff(_camera.zOff);
}

void CTiledVolumeViewer::onVolumeReady()
{
    if (!_volume) return;

    _hadValidDataBounds = false;

    // Create a default PlaneSurface so axis-aligned views render immediately
    if (!_surfWeak.lock() && _volume && isAxisAlignedView()) {
        auto shape = _volume->shape();
        cv::Vec3f center(static_cast<float>(shape[0]) * 0.5f, static_cast<float>(shape[1]) * 0.5f, static_cast<float>(shape[2]) * 0.5f);
        cv::Vec3f normal;
        if (_surfName == "xy plane") normal = cv::Vec3f(0, 0, 1);
        else if (_surfName == "xz plane" || _surfName == "seg xz") normal = cv::Vec3f(0, 1, 0);
        else normal = cv::Vec3f(1, 0, 0);
        auto defaultSurf = std::make_shared<PlaneSurface>(center, normal);
        _defaultSurface = defaultSurf;
        _surfWeak = defaultSurf;
    }

    _camera.recalcPyramidLevel(static_cast<int>(_volume->numScales()));
    updateContentMinScale();
    rebuildContentGrid();
    centerViewport();

    double vs = _volume->voxelSize() / static_cast<double>(_camera.dsScale);
    fGraphicsView->setVoxelSize(vs, vs);

    submitRender();
    renderIntersections();
    updateStatusLabel();
}

void CTiledVolumeViewer::onDataBoundsReady()
{
    if (!_volume) return;
    const auto& db = _volume->dataBounds();
    if (!db.valid) return;

    // Only the "xy plane" viewer directly re-centers its PlaneSurface and
    // updates the focus POI.  The POI change propagates through
    // AxisAlignedSliceController to seg xz / seg yz, which receive new
    // surfaces via onSurfaceChanged — no need to touch them here.
    if (_surfName == "xy plane") {
        auto surf = _surfWeak.lock();
        auto* plane = dynamic_cast<PlaneSurface*>(surf.get());
        float dbMinX = static_cast<float>(db.minX), dbMaxX = static_cast<float>(db.maxX);
        float dbMinY = static_cast<float>(db.minY), dbMaxY = static_cast<float>(db.maxY);
        float dbMinZ = static_cast<float>(db.minZ), dbMaxZ = static_cast<float>(db.maxZ);
        if (plane) {
            cv::Vec3f center((dbMinX + dbMaxX) * 0.5f,
                             (dbMinY + dbMaxY) * 0.5f,
                             (dbMinZ + dbMaxZ) * 0.5f);
            plane->setOrigin(center);
        }

        // Update focus POI — cascades to all axis-aligned viewers
        POI* focus = _state->poi("focus");
        if (focus) {
            cv::Vec3f center((dbMinX + dbMaxX) * 0.5f,
                             (dbMinY + dbMaxY) * 0.5f,
                             (dbMinZ + dbMaxZ) * 0.5f);
            focus->p = center;
            _state->setPOI("focus", focus);
        }
    }

    updateContentMinScale();
    rebuildContentGrid();
    centerViewport();
    _camera.invalidate();
    submitRender();
    renderIntersections();
}

QRectF CTiledVolumeViewer::viewportSceneRect() const
{
    if (!fGraphicsView) return QRectF();
    float vpW = static_cast<float>(fGraphicsView->viewport()->width());
    float vpH = static_cast<float>(fGraphicsView->viewport()->height());
    return QRectF(0, 0, vpW, vpH);
}

// ============================================================================
// Navigation
// ============================================================================

void CTiledVolumeViewer::panBy(int dx, int dy)
{
    panByF(static_cast<float>(dx), static_cast<float>(dy));
}

void CTiledVolumeViewer::panByF(float dx, float dy)
{
    const float invScale = _navSpeed / _camera.scale;
    _camera.surfacePtr[0] -= dx * invScale;
    _camera.surfacePtr[1] -= dy * invScale;

    // Clamp pan to full content bounds (not the windowed grid bounds)
    if (_fullContentMaxU > _fullContentMinU) {
        _camera.surfacePtr[0] = std::clamp(_camera.surfacePtr[0], _fullContentMinU, _fullContentMaxU);
        _camera.surfacePtr[1] = std::clamp(_camera.surfacePtr[1], _fullContentMinV, _fullContentMaxV);
    }

    _focusSurfacePos[0] = _camera.surfacePtr[0];
    _focusSurfacePos[1] = _camera.surfacePtr[1];

    // Clear z-velocity on pan so z-prefetch reverts to symmetric
    _zVelocity = 0.0f;

    centerViewport();
    submitRender();
}

void CTiledVolumeViewer::zoomAt(float factor, const QPointF& scenePos)
{
    // Convert continuous factor into discrete zoom-stop steps
    int steps = (factor > 1.0f) ? 1 : (factor < 1.0f) ? -1 : 0;
    if (steps == 0) return;
    zoomStepsAt(steps, scenePos);
}

void CTiledVolumeViewer::zoomStepsAt(int steps, const QPointF& scenePos)
{
    if (steps == 0) return;

    // ~5% zoom per wheel notch (standard for map/image viewers), scaled by navSpeed.
    float factor = std::pow(1.05f, static_cast<float>(steps) * _navSpeed);
    float newScale = std::clamp(_camera.scale * factor,
                                 _contentMinScale, TiledViewerCamera::MAX_SCALE);
    if (std::abs(newScale - _camera.scale) < _camera.scale * 1e-6f) return;

    // Zoom-at-point: the surface position under the cursor stays fixed.
    // scenePos is viewport-relative coordinates (not scene coords).
    // If cursor is outside viewport, zoom at center (dx=dy=0).
    float vpW = static_cast<float>(fGraphicsView->viewport()->width());
    float vpH = static_cast<float>(fGraphicsView->viewport()->height());
    float mx = static_cast<float>(scenePos.x());
    float my = static_cast<float>(scenePos.y());
    float dx = 0, dy = 0;
    if (mx >= 0 && mx < vpW && my >= 0 && my < vpH) {
        dx = mx - vpW * 0.5f;
        dy = my - vpH * 0.5f;
    }

    _camera.surfacePtr[0] += dx * (1.0f / _camera.scale - 1.0f / newScale);
    _camera.surfacePtr[1] += dy * (1.0f / _camera.scale - 1.0f / newScale);
    _camera.scale = newScale;

    if (_volume) {
        float oldDs = _camera.dsScale;
        _camera.recalcPyramidLevel(static_cast<int>(_volume->numScales()));
        if (std::abs(_camera.dsScale - oldDs) > 1e-6f) {
            double vs = _volume->voxelSize() / static_cast<double>(_camera.dsScale);
            fGraphicsView->setVoxelSize(vs, vs);
        }
    }

    fGraphicsView->resetTransform();
    rebuildContentGrid();
    centerViewport();
    _focusSurfacePos[0] = _camera.surfacePtr[0];
    _focusSurfacePos[1] = _camera.surfacePtr[1];

    _camera.invalidate();
    submitRender();
}

void CTiledVolumeViewer::setSliceOffset(float dz)
{
    float maxZ = 10000.0f;
    if (_volume) {
        auto [w, h, d] = _volume->shape();
        maxZ = static_cast<float>(std::max({w, h, d}));
    }
    _camera.zOff = std::clamp(_camera.zOff + dz, -maxZ, maxZ);
    _zVelocity = dz;
    _camera.invalidate();
    centerViewport();
    submitRender();
    updateStatusLabel();
}

void CTiledVolumeViewer::onZoom(int steps, QPointF scene_point, Qt::KeyboardModifiers modifiers)
{
    auto surf = _surfWeak.lock();
    if (!surf) return;

    if (_segmentationEditActive && (modifiers & Qt::ControlModifier)) {
        // scene_point is viewport-relative; convert to scene for segmentation
        QPointF sp = fGraphicsView->mapToScene(scene_point.toPoint());
        cv::Vec3f world = sceneToVolume(sp);
        emit sendSegmentationRadiusWheel(steps, sp, world);
        return;
    }

    if (modifiers & Qt::ShiftModifier) {
        if (steps == 0) return;

        PlaneSurface* plane = dynamic_cast<PlaneSurface*>(surf.get());
        // 1 voxel per notch at navSpeed=1, scaled by navSpeed
        float adjustedSteps = static_cast<float>(steps) * _navSpeed;

        if (_surfName != "segmentation" && plane && _state) {
            POI* focus = _state->poi("focus");
            if (!focus) {
                focus = new POI;
                focus->p = plane->origin();
                focus->n = plane->normal(cv::Vec3f(0, 0, 0), {});
            }

            cv::Vec3f normal = plane->normal(cv::Vec3f(0, 0, 0), {});
            const double length = cv::norm(normal);
            if (length > 0.0) normal *= static_cast<float>(1.0 / length);

            cv::Vec3f newPosition = focus->p + normal * static_cast<float>(adjustedSteps);

            if (_volume) {
                auto [w, h, d] = _volume->shape();
                float cx0 = 0, cy0 = 0, cz0 = 0;
                float cx1 = static_cast<float>(w - 1), cy1 = static_cast<float>(h - 1), cz1 = static_cast<float>(d - 1);
                newPosition[0] = std::clamp(newPosition[0], cx0, cx1);
                newPosition[1] = std::clamp(newPosition[1], cy0, cy1);
                newPosition[2] = std::clamp(newPosition[2], cz0, cz1);
            }

            focus->p = newPosition;
            if (length > 0.0) focus->n = normal;
            focus->surfaceId = _surfName;
            _state->setPOI("focus", focus);
        } else {
            setSliceOffset(static_cast<float>(adjustedSteps));
        }
    } else {
        // Zoom immediately, no momentum (causes ghosting)
        int zoomDir = (steps > 0) ? 1 : (steps < 0) ? -1 : 0;
        if (zoomDir != 0) {
            zoomStepsAt(zoomDir, scene_point);
        }
    }
}

void CTiledVolumeViewer::adjustZoomByFactor(float factor)
{
    auto surf = _surfWeak.lock();
    if (!surf) return;

    int steps = (factor > 1.0f) ? 1 : (factor < 1.0f) ? -1 : 0;
    if (steps == 0) return;

    // Zoom centered on camera position (avoids Qt's integer viewport rounding)
    QPointF sceneCenter = _tileScene->surfaceToScene(
        _camera.surfacePtr[0], _camera.surfacePtr[1]);
    zoomStepsAt(steps, sceneCenter);
}

void CTiledVolumeViewer::adjustSurfaceOffset(float dn)
{
    setSliceOffset(dn);
}

void CTiledVolumeViewer::resetSurfaceOffsets()
{
    _camera.zOff = 0.0f;
    _camera.invalidate();
    submitRender();
}

// ============================================================================
// Pan handling via CVolumeViewerView events
// ============================================================================

void CTiledVolumeViewer::onPanStart(Qt::MouseButton /*buttons*/, Qt::KeyboardModifiers /*modifiers*/)
{
    _isPanning = true;
    // The view handles pan tracking with _last_pan_position internally,
    // but since we disabled scrollbars, its scroll-based panning won't work.
    // We need to intercept the mouse move deltas instead.
    // Store current mouse pos for delta computation
    _lastPanPos = QCursor::pos();
    // Init float scene position for sub-pixel pan tracking
    QPoint gp = QCursor::pos();
    QPoint vp2 = fGraphicsView->viewport()->mapFromGlobal(gp);
    _lastPanSceneF = fGraphicsView->mapToScene(vp2);

    // Record viewport center in scene coords for predictive prefetch
    QRectF vp = viewportSceneRect();
    _lastPanScenePos = vp.center();
}

void CTiledVolumeViewer::onPanRelease(Qt::MouseButton /*buttons*/, Qt::KeyboardModifiers /*modifiers*/)
{
    _isPanning = false;
    _camera.invalidate();
    submitRender();
    renderIntersections();
    updateStatusLabel();
    scheduleOverlayUpdate();
}

void CTiledVolumeViewer::onScrolled()
{
    // In tiled mode, scrollbar-based scrolling is disabled.
    // Pan is handled via panBy() from mouse move deltas.
}

void CTiledVolumeViewer::onResized()
{
    updateContentMinScale();
    rebuildContentGrid();
    centerViewport();
    _camera.invalidate();
    submitRender();
    scheduleOverlayUpdate();
}

// ============================================================================
// Rendering
// ============================================================================

bool CTiledVolumeViewer::isAxisAlignedView() const
{
    return _surfName == "xy plane" || _surfName == "xz plane" || _surfName == "yz plane"
           || _surfName == "seg xz" || _surfName == "seg yz";
}

bool CTiledVolumeViewer::clampToDataBounds(cv::Vec3f& lo, cv::Vec3f& hi) const
{
    if (!_volume) return false;
    // Use full physical volume bounds for prefetch — data bounds are only
    // an approximation (derived from the coarsest pyramid level) and can
    // miss real data near boundaries.  The chunk-level skip in Slicing.cpp
    // still avoids I/O for truly empty regions.
    auto [w, h, d] = _volume->shape();
    lo[0] = std::max(lo[0], 0.f);
    lo[1] = std::max(lo[1], 0.f);
    lo[2] = std::max(lo[2], 0.f);
    hi[0] = std::min(hi[0], static_cast<float>(w - 1));
    hi[1] = std::min(hi[1], static_cast<float>(h - 1));
    hi[2] = std::min(hi[2], static_cast<float>(d - 1));
    return lo[0] <= hi[0] && lo[1] <= hi[1] && lo[2] <= hi[2];
}

void CTiledVolumeViewer::renderVisible(bool force)
{
    if (isWindowMinimized()) {
        _dirtyWhileMinimized = true;
        return;
    }
    if (force) {
        _camera.invalidate();
    }
    submitRender();
}


void CTiledVolumeViewer::scheduleRender()
{
    _renderPending = true;
    if (!_renderTimer->isActive())
        _renderTimer->start();
}

void CTiledVolumeViewer::submitRender()
{
    _pointSceneCache.clear();

    auto surf = _surfWeak.lock();
    if (!surf || !_volume || !_volume->zarrDataset()) return;

    uint32_t* fbBits = _tileScene->framebufferBits();
    int fbW = _tileScene->framebufferWidth();
    int fbH = _tileScene->framebufferHeight();
    int fbStride = _tileScene->framebufferStride();
    if (!fbBits || fbW <= 0 || fbH <= 0) goto prefetch;

    {
        auto* plane = dynamic_cast<PlaneSurface*>(surf.get());
        if (plane) {
            cv::Vec3f vx = plane->basisX();
            cv::Vec3f vy = plane->basisY();
            cv::Vec3f n = plane->normal(cv::Vec3f(0, 0, 0));

            float halfW = static_cast<float>(fbW) * 0.5f / _camera.scale;
            float halfH = static_cast<float>(fbH) * 0.5f / _camera.scale;

            cv::Vec3f origin = vx * (_camera.surfacePtr[0] - halfW)
                             + vy * (_camera.surfacePtr[1] - halfH)
                             + plane->origin() + n * _camera.zOff;

            std::array<uint32_t, 256> lut;
            vc::buildWindowLevelLut(lut, _baseWindowLow, _baseWindowHigh);

            vc::SampleParams sp;
            sp.level = _camera.dsScaleIdx;
            sp.method = vc::Sampling::Nearest;

            _volume->samplePlaneBestEffortARGB32(
                fbBits, fbStride, origin,
                vx / _camera.scale, vy / _camera.scale,
                fbW, fbH, sp, lut.data());
        }
    }

    // Save every xy plane frame so we can check if glitch is in pixel data or display
    if (_surfName == "xy plane") {
        static int fn = 0;
        if (fn >= 10 && fn < 30) {
            _tileScene->rawFramebuffer().save(QString("/tmp/fb_%1.png").arg(fn));
        }
        fn++;
    }

    _tileScene->markDirty();
    _tileScene->flush();
    fGraphicsView->viewport()->repaint();

prefetch:
    // Viewport-aware prefetch
    if (_volume && _volume->tieredCache()) {
        auto* planeForPrefetch = dynamic_cast<PlaneSurface*>(surf.get());
        cv::Vec3f lo, hi;
        int pfbW = _tileScene->framebufferWidth(), pfbH = _tileScene->framebufferHeight();
        bool ok = planeForPrefetch
            ? computePlanePrefetchBBox(planeForPrefetch, QRectF(0, 0, pfbW, pfbH), lo, hi)
            : computeQuadPrefetchBBox(surf, viewportSceneRect(), lo, hi);
        if (ok) {
            _volume->prefetchWorldBBox(lo, hi, _camera.dsScaleIdx);

            if (!_isPanning) {
                constexpr float PREFETCH_Z_BEHIND = 4.0f;
                constexpr float PREFETCH_Z_AHEAD = 24.0f;
                cv::Vec3f loZ = lo, hiZ = hi;
                if (_zVelocity > 0) {
                    loZ[2] -= PREFETCH_Z_BEHIND;
                    hiZ[2] += PREFETCH_Z_AHEAD;
                } else if (_zVelocity < 0) {
                    loZ[2] -= PREFETCH_Z_AHEAD;
                    hiZ[2] += PREFETCH_Z_BEHIND;
                } else {
                    loZ[2] -= 16.0f;
                    hiZ[2] += 16.0f;
                }
                _volume->prefetchWorldBBox(loZ, hiZ, _camera.dsScaleIdx);
            }
        }
    }
    _lastZOff = _camera.zOff;
}

bool CTiledVolumeViewer::computePlanePrefetchBBox(PlaneSurface* plane,
                                                   const QRectF& prefetchRect,
                                                   cv::Vec3f& lo, cv::Vec3f& hi) const
{
    const float invScale = 1.0f / _camera.scale;
    const float margin = 512.0f * invScale;

    cv::Vec2f vpTopLeft = _tileScene->sceneToSurface(QPointF(prefetchRect.left(), prefetchRect.top()));
    cv::Vec2f vpBotRight = _tileScene->sceneToSurface(QPointF(prefetchRect.right(), prefetchRect.bottom()));

    const float uMin = vpTopLeft[0] - margin;
    const float uMax = vpBotRight[0] + margin;
    const float vMin = vpTopLeft[1] - margin;
    const float vMax = vpBotRight[1] + margin;

    const cv::Vec3f o = plane->origin();
    const cv::Vec3f bx = plane->basisX();
    const cv::Vec3f by = plane->basisY();
    const cv::Vec3f n = plane->normal(cv::Vec3f(0, 0, 0));

    cv::Vec3f corners[4] = {
        o + bx * uMin + by * vMin + n * _camera.zOff,
        o + bx * uMax + by * vMin + n * _camera.zOff,
        o + bx * uMin + by * vMax + n * _camera.zOff,
        o + bx * uMax + by * vMax + n * _camera.zOff,
    };

    lo = cv::Vec3f(std::numeric_limits<float>::max(),
                   std::numeric_limits<float>::max(),
                   std::numeric_limits<float>::max());
    hi = cv::Vec3f(std::numeric_limits<float>::lowest(),
                   std::numeric_limits<float>::lowest(),
                   std::numeric_limits<float>::lowest());
    for (const auto& c : corners) {
        for (int i = 0; i < 3; i++) {
            lo[i] = std::min(lo[i], c[i]);
            hi[i] = std::max(hi[i], c[i]);
        }
    }
    return clampToDataBounds(lo, hi);
}

bool CTiledVolumeViewer::computeQuadPrefetchBBox(const std::shared_ptr<Surface>& surf,
                                                  const QRectF& prefetchRect,
                                                  cv::Vec3f& lo, cv::Vec3f& hi) const
{
    cv::Vec2f corners2d[4] = {
        _tileScene->sceneToSurface(QPointF(prefetchRect.left(), prefetchRect.top())),
        _tileScene->sceneToSurface(QPointF(prefetchRect.right(), prefetchRect.top())),
        _tileScene->sceneToSurface(QPointF(prefetchRect.left(), prefetchRect.bottom())),
        _tileScene->sceneToSurface(QPointF(prefetchRect.right(), prefetchRect.bottom())),
    };

    lo = cv::Vec3f(std::numeric_limits<float>::max(),
                   std::numeric_limits<float>::max(),
                   std::numeric_limits<float>::max());
    hi = cv::Vec3f(std::numeric_limits<float>::lowest(),
                   std::numeric_limits<float>::lowest(),
                   std::numeric_limits<float>::lowest());

    for (const auto& c2 : corners2d) {
        cv::Mat_<cv::Vec3f> pt;
        surf->gen(&pt, nullptr, cv::Size(1, 1), cv::Vec3f(0, 0, 0),
                  _camera.scale,
                  {c2[0] * _camera.scale, c2[1] * _camera.scale, _camera.zOff});
        if (pt.empty()) continue;
        const cv::Vec3f& v = pt(0, 0);
        for (int i = 0; i < 3; i++) {
            lo[i] = std::min(lo[i], v[i]);
            hi[i] = std::max(hi[i], v[i]);
        }
    }

    // Add margin for interpolation + scrolling
    float margin = 512.0f / _camera.scale;
    for (int i = 0; i < 3; i++) {
        lo[i] -= margin;
        hi[i] += margin;
    }

    return clampToDataBounds(lo, hi);
}


// ============================================================================
// Coordinate transforms
// ============================================================================

QPointF CTiledVolumeViewer::volumeToScene(const cv::Vec3f& vol_point)
{
    auto surf = _surfWeak.lock();
    auto* patchIndex = _viewerManager ? _viewerManager->surfacePatchIndex() : nullptr;
    return tiledVolumeToScene(surf.get(), _tileScene, patchIndex, vol_point);
}

cv::Vec3f CTiledVolumeViewer::sceneToVolume(const QPointF& scenePoint) const
{
    cv::Vec3f p, n;
    if (sceneToVolumePN(p, n, scenePoint)) {
        return p;
    }
    return {0.0f, 0.0f, 0.0f};
}

bool CTiledVolumeViewer::sceneToVolumePN(cv::Vec3f& p, cv::Vec3f& n,
                                          const QPointF& scenePos) const
{
    auto surf = _surfWeak.lock();
    return tiledSceneToVolume(surf.get(), _tileScene, scenePos, p, n);
}

cv::Vec2f CTiledVolumeViewer::sceneToSurfaceCoords(const QPointF& scenePos) const
{
    return _tileScene->sceneToSurface(scenePos);
}

// ============================================================================
// Mouse handlers
// ============================================================================

void CTiledVolumeViewer::onCursorMove(QPointF scene_loc)
{
    auto surf = _surfWeak.lock();
    onCursorMoveImpl(scene_loc, surf);
}

void CTiledVolumeViewer::onCursorMoveImpl(QPointF scene_loc,
                                           const std::shared_ptr<Surface>& surf)
{
    if (!surf || !_state) return;

    auto updateCursorPoi = [this](const QPointF& cursorScenePos) {
        cv::Vec3f p, n;
        if (!sceneToVolumePN(p, n, cursorScenePos)) {
            if (_ov.cursor) _ov.cursor->hide();
            return;
        }

        if (_ov.cursor) {
            _ov.cursor->show();
            _ov.cursor->setPos(cursorScenePos);
        }

        POI* cursor = _state->poi("cursor");
        if (!cursor) cursor = new POI;
        cursor->p = p;
        cursor->n = n;
        cursor->surfaceId = _surfName;
        _state->setPOI("cursor", cursor);
    };

    // Handle panning: if middle/right button is down, pan instead
    if (_isPanning) {
        // Use scene coordinates for sub-pixel precision
        QPointF currentScenePos = scene_loc;
        float dx = static_cast<float>(currentScenePos.x() - _lastPanSceneF.x());
        float dy = static_cast<float>(currentScenePos.y() - _lastPanSceneF.y());
        _lastPanSceneF = currentScenePos;
        _lastPanPos = QCursor::pos();  // keep for compatibility
        if (std::abs(dx) > 0.001f || std::abs(dy) > 0.001f) {
            // Scene coords are already in surface space — convert to pixels
            float pxDx = dx * _camera.scale;
            float pxDy = dy * _camera.scale;
            panByF(-pxDx, -pxDy);
        }

        const QPoint viewportPos = fGraphicsView->viewport()->mapFromGlobal(_lastPanPos);
        updateCursorPoi(fGraphicsView->mapToScene(viewportPos));
        return;
    }

    updateCursorPoi(scene_loc);

    // Point highlight logic
    if (_pointCollection && _draggedPointId == 0) {
        uint64_t oldHighlighted = _highlightedPointId;
        _highlightedPointId = 0;

        const auto& collections = _pointCollection->getAllCollections();
        if (!collections.empty()) {
            const float threshold = 10.0f;
            const float thresholdSq = threshold * threshold;
            // If we find a point within this radius, accept it immediately
            // without searching for the absolute closest.
            const float earlyOutSq = 3.0f * 3.0f;
            float minDistSq = thresholdSq;

            for (const auto& [colId, col] : collections) {
                for (const auto& [ptId, pt] : col.points) {
                    auto cacheIt = _pointSceneCache.find(pt.id);
                    QPointF ptScene;
                    if (cacheIt != _pointSceneCache.end()) {
                        ptScene = cacheIt->second;
                    } else {
                        ptScene = volumeToScene(pt.p);
                        _pointSceneCache[pt.id] = ptScene;
                    }
                    QPointF diff = scene_loc - ptScene;
                    float distSq = static_cast<float>(QPointF::dotProduct(diff, diff));
                    if (distSq < minDistSq) {
                        minDistSq = distSq;
                        _highlightedPointId = pt.id;
                        if (distSq < earlyOutSq)
                            goto highlight_done;
                    }
                }
            }
        }
        highlight_done:

        if (oldHighlighted != _highlightedPointId) {
            scheduleOverlayUpdate();
        }
    }
}

void CTiledVolumeViewer::onVolumeClicked(QPointF scene_loc, Qt::MouseButton buttons,
                                          Qt::KeyboardModifiers modifiers)
{
    auto surf = _surfWeak.lock();
    if (!surf) return;

    if (_draggedPointId != 0) return;

    cv::Vec3f p, n;
    if (!sceneToVolumePN(p, n, scene_loc)) return;

    if (buttons == Qt::LeftButton) {
        bool isShift = modifiers.testFlag(Qt::ShiftModifier);
        if (isShift && !_segmentationEditActive && _pointCollection) {
            if (_selectedCollectionId != 0) {
                const auto& collections = _pointCollection->getAllCollections();
                auto it = collections.find(_selectedCollectionId);
                if (it != collections.end()) {
                    _pointCollection->addPoint(it->second.name, p);
                }
            } else {
                std::string newName = _pointCollection->generateNewCollectionName("col");
                auto newPoint = _pointCollection->addPoint(newName, p);
                _selectedCollectionId = newPoint.collectionId;
                emit sendCollectionSelected(_selectedCollectionId);
            }
        } else if (_highlightedPointId != 0) {
            emit pointClicked(_highlightedPointId);
        }
    }

    const auto& segmentation = activeSegmentationHandle();
    if (dynamic_cast<PlaneSurface*>(surf.get())) {
        sendVolumeClicked(p, n, surf.get(), buttons, modifiers);
    } else if (segmentation.viewerIsSegmentationView && segmentation.surface) {
        sendVolumeClicked(p, n, segmentation.surface, buttons, modifiers);
    }
}

void CTiledVolumeViewer::onMousePress(QPointF scene_loc, Qt::MouseButton button,
                                       Qt::KeyboardModifiers modifiers)
{
    auto surf = _surfWeak.lock();
    if (!_pointCollection || !surf) return;

    if (button == Qt::LeftButton) {
        if (_highlightedPointId != 0 && !modifiers.testFlag(Qt::ControlModifier)) {
            emit pointClicked(_highlightedPointId);
            _draggedPointId = _highlightedPointId;
        }
    } else if (button == Qt::RightButton) {
        if (_highlightedPointId != 0) {
            _pointCollection->removePoint(_highlightedPointId);
        }
    }

    cv::Vec3f p, n;
    if (sceneToVolumePN(p, n, scene_loc)) {
        _lastScenePos = scene_loc;
        sendMousePressVolume(p, n, button, modifiers);
    }
}

void CTiledVolumeViewer::onMouseMove(QPointF scene_loc, Qt::MouseButtons buttons,
                                      Qt::KeyboardModifiers modifiers)
{
    auto surf = _surfWeak.lock();
    onCursorMoveImpl(scene_loc, surf);

    if ((buttons & Qt::LeftButton) && _draggedPointId != 0) {
        cv::Vec3f p, n;
        if (sceneToVolumePN(p, n, scene_loc)) {
            if (auto pointOpt = _pointCollection->getPoint(_draggedPointId)) {
                ColPoint updated = *pointOpt;
                updated.p = p;
                _pointCollection->updatePoint(updated);
            }
        }
    } else {
        if (!surf) return;
        cv::Vec3f p, n;
        if (!sceneToVolumePN(p, n, scene_loc)) return;
        _lastScenePos = scene_loc;
        emit sendMouseMoveVolume(p, buttons, modifiers);
    }
}

void CTiledVolumeViewer::onMouseRelease(QPointF scene_loc, Qt::MouseButton button,
                                         Qt::KeyboardModifiers modifiers)
{
    auto surf = _surfWeak.lock();
    if (button == Qt::LeftButton && _draggedPointId != 0) {
        _draggedPointId = 0;
        onCursorMoveImpl(scene_loc, surf);
    }

    cv::Vec3f p, n;
    if (sceneToVolumePN(p, n, scene_loc)) {
        const auto& segmentation = activeSegmentationHandle();
        if (dynamic_cast<PlaneSurface*>(surf.get())) {
            emit sendMouseReleaseVolume(p, button, modifiers);
        } else if (segmentation.viewerIsSegmentationView) {
            emit sendMouseReleaseVolume(p, button, modifiers);
        }
    }
}

void CTiledVolumeViewer::onKeyRelease(int /*key*/, Qt::KeyboardModifiers /*modifiers*/)
{
    // Arrow key pan moved to onKeyPress for immediate response
}

void CTiledVolumeViewer::onKeyPress(int key, Qt::KeyboardModifiers /*modifiers*/)
{
    constexpr int PAN_PX = 64;
    switch (key) {
    case Qt::Key_Left:  panBy( PAN_PX, 0); break;
    case Qt::Key_Right: panBy(-PAN_PX, 0); break;
    case Qt::Key_Up:    panBy(0,  PAN_PX); break;
    case Qt::Key_Down:  panBy(0, -PAN_PX); break;
    default: break;
    }
}

// ============================================================================
// POI handling
// ============================================================================

void CTiledVolumeViewer::onPOIChanged(const std::string& name, POI* poi)
{
    auto surf = _surfWeak.lock();
    if (!poi || !surf) return;

    if (name == "focus") {
        if (auto* plane = dynamic_cast<PlaneSurface*>(surf.get())) {
            const bool originChanged = (poi->p != plane->origin());
            if (originChanged) {
                plane->setOrigin(poi->p);
            }

            // Plane viewers should center on the new focus point itself,
            // not preserve the old pan offset from the previous plane origin.
            _camera.surfacePtr[0] = 0.0f;
            _camera.surfacePtr[1] = 0.0f;
            _focusSurfacePos[0] = 0.0f;
            _focusSurfacePos[1] = 0.0f;
            centerViewport();
            scheduleOverlayUpdate();
            if (originChanged) {
                _state->setSurface(_surfName, surf);
            } else {
                _camera.invalidate();
                submitRender();
                renderIntersections();
                updateStatusLabel();
            }
        } else if (auto* quad = dynamic_cast<QuadSurface*>(surf.get())) {
            cv::Vec3f ptr(0, 0, 0);
            auto* patchIndex = _viewerManager ? _viewerManager->surfacePatchIndex() : nullptr;
            float dist = quad->pointTo(ptr, poi->p, 4.0f, 100, patchIndex);
            if (dist < 4.0f) {
                // Center camera on the focus point
                cv::Vec3f loc = quad->loc(ptr);
                _camera.surfacePtr[0] = loc[0];
                _camera.surfacePtr[1] = loc[1];
                _focusSurfacePos[0] = loc[0];
                _focusSurfacePos[1] = loc[1];
                _camera.invalidate();
                submitRender();
            }
        }
    } else if (name == "cursor") {
        Surface* currentSurf = this->currentSurface();
        if (!currentSurf) return;

        if (_surfName == "segmentation" && !_mirrorCursorToSegmentation) {
            if (poi->surfaceId.empty() || poi->surfaceId != _surfName) return;
        }

        // Position cursor in canvas coords
        QPointF scenePos = volumeToScene(poi->p);
        if (!_ov.cursor) {
            // Create cursor item
            QPen pen(QBrush(COLOR_CURSOR), 2);
            QGraphicsLineItem* parent = new QGraphicsLineItem(-10, 0, -5, 0);
            parent->setZValue(10);
            parent->setPen(pen);
            auto* l1 = new QGraphicsLineItem(10, 0, 5, 0, parent); l1->setPen(pen);
            auto* l2 = new QGraphicsLineItem(0, -10, 0, -5, parent); l2->setPen(pen);
            auto* l3 = new QGraphicsLineItem(0, 10, 0, 5, parent); l3->setPen(pen);
            _ov.cursor = parent;
            _scene->addItem(_ov.cursor);
        }

        // Simple distance-based opacity
        PlaneSurface* slicePlane = dynamic_cast<PlaneSurface*>(currentSurf);
        if (slicePlane) {
            float dist = slicePlane->pointDist(poi->p);
            if (dist < 20.0f / _camera.scale) {
                _ov.cursor->setPos(scenePos);
                _ov.cursor->setOpacity(static_cast<qreal>(1.0f - dist * _camera.scale / 20.0f));
            } else {
                _ov.cursor->setOpacity(0.0);
            }
        } else {
            _ov.cursor->setPos(scenePos);
            _ov.cursor->setOpacity(1.0);
        }
    }
}

// ============================================================================
// Status / display
// ============================================================================

void CTiledVolumeViewer::updateStatusLabel()
{
    QString status = QString("%1x").arg(_camera.scale, 0, 'f', 2);

    // Desired pyramid level
    status += QString(" 1:%1").arg(1 << _camera.dsScaleIdx);

    status += QString(" z=%1").arg(_camera.zOff, 0, 'f', 1);

    if (_compositeSettings.enabled) {
        QString method = QString::fromStdString(_compositeSettings.params.method);
        if (!method.isEmpty())
            method[0] = method[0].toUpper();
        status += QString(" | %1(%2)").arg(method).arg(
            _compositeSettings.layersFront + _compositeSettings.layersBehind);
    }

    if (_volume && _volume->tieredCache()) {
        auto s = _volume->tieredCache()->stats();

        // Hit ratios
        uint64_t total = s.hotHits + s.coldHits + s.iceFetches + s.misses;
        if (total > 0) {
            auto pct = [&](uint64_t n) { return static_cast<int>(100 * n / total); };
            status += QString(" | H%1 D%2")
                .arg(pct(s.hotHits)).arg(pct(s.coldHits));
        }

        // RAM usage (hot tier)
        double hotGB = static_cast<double>(s.hotBytes) / (1024.0 * 1024.0 * 1024.0);
        status += QString(" | ram %1G").arg(hotGB, 0, 'f', 1);

        // Disk writes
        if (s.diskWrites > 0)
            status += QString(" | w%1").arg(s.diskWrites);

        // Negative cache
        if (s.negativeCount > 0)
            status += QString(" | neg %1").arg(s.negativeCount);

        // Queue & downloads
        if (s.ioPending > 0)
            status += QString(" | q%1").arg(s.ioPending);
        if (s.iceFetches > 0)
            status += QString(" dl%1").arg(s.iceFetches);
    }

    status += " [tiled]";

    _lbl->setText(status);
    _lbl->adjustSize();
}

void CTiledVolumeViewer::fitSurfaceInView()
{
    if (!fGraphicsView) return;

    auto surf = _surfWeak.lock();
    if (!surf || !dynamic_cast<QuadSurface*>(surf.get())) {
        // No surface (e.g. remote volume only) — reset to data center at scale 1
        _camera.scale = 1.0f;
        _camera.surfacePtr = cv::Vec3f(0, 0, 0);
        _camera.zOff = 0;
        if (_volume) _camera.recalcPyramidLevel(static_cast<int>(_volume->numScales()));
        _camera.invalidate();
        updateStatusLabel();
        fGraphicsView->resetTransform();
        rebuildContentGrid();
        centerViewport();
        submitRender();
        return;
    }

    auto* quadSurf = dynamic_cast<QuadSurface*>(surf.get());

    // Auto-crop: find bounding box of valid (non-sentinel) points.
    // Cache the result keyed on surface identity to avoid O(n²) scan every call.
    const cv::Mat_<cv::Vec3f>& pts = quadSurf->rawPoints();
    if (_surfBBoxCache.surfPtr != quadSurf || _surfBBoxCache.colMax < _surfBBoxCache.colMin) {
        int colMin = pts.cols, colMax = -1, rowMin = pts.rows, rowMax = -1;
        for (int j = 0; j < pts.rows; j++)
            for (int i = 0; i < pts.cols; i++)
                if (pts(j, i)[0] > -0.5f && std::isfinite(pts(j, i)[0])) {
                    colMin = std::min(colMin, i);
                    colMax = std::max(colMax, i);
                    rowMin = std::min(rowMin, j);
                    rowMax = std::max(rowMax, j);
                }
        _surfBBoxCache = {colMin, colMax, rowMin, rowMax, quadSurf};
    }
    int colMin = _surfBBoxCache.colMin, colMax = _surfBBoxCache.colMax;
    int rowMin = _surfBBoxCache.rowMin, rowMax = _surfBBoxCache.rowMax;

    if (colMax < colMin || rowMax < rowMin) {
        _camera.scale = 1.0f;
        _camera.surfacePtr = cv::Vec3f(0, 0, 0);
        if (_volume) _camera.recalcPyramidLevel(static_cast<int>(_volume->numScales()));
        updateStatusLabel();
        return;
    }

    // Valid region in grid pixels
    float validW = static_cast<float>(colMax - colMin + 1);
    float validH = static_cast<float>(rowMax - rowMin + 1);

    // Convert to surface parameter space (what the viewport uses)
    cv::Vec2f sc = quadSurf->scale();
    float validSurfW = validW / sc[0];
    float validSurfH = validH / sc[1];

    QSize vpSize = fGraphicsView->viewport()->size();
    float vpW = static_cast<float>(vpSize.width());
    float vpH = static_cast<float>(vpSize.height());
    if (vpW <= 0 || vpH <= 0) return;

    float fitFactor = 0.8f;
    float reqScaleX = (vpW * fitFactor) / validSurfW;
    float reqScaleY = (vpH * fitFactor) / validSurfH;
    _camera.scale = TiledViewerCamera::roundScale(std::min(reqScaleX, reqScaleY));

    // Center on the valid region.
    // surfacePtr is in grid units; (0,0,0) maps to the grid center (cols/2, rows/2).
    // To center on grid position (gx, gy): surfacePtr = (gx - cols/2, gy - rows/2, 0)
    float fColMin = static_cast<float>(colMin), fColMax = static_cast<float>(colMax);
    float fRowMin = static_cast<float>(rowMin), fRowMax = static_cast<float>(rowMax);
    float gridCenterX = (fColMin + fColMax) * 0.5f;
    float gridCenterY = (fRowMin + fRowMax) * 0.5f;
    _camera.surfacePtr[0] = gridCenterX - static_cast<float>(pts.cols) * 0.5f;
    _camera.surfacePtr[1] = gridCenterY - static_cast<float>(pts.rows) * 0.5f;
    _camera.surfacePtr[2] = 0;
    _focusSurfacePos[0] = _camera.surfacePtr[0];
    _focusSurfacePos[1] = _camera.surfacePtr[1];

    if (_volume) _camera.recalcPyramidLevel(static_cast<int>(_volume->numScales()));
    _camera.invalidate();
    updateStatusLabel();
    fGraphicsView->resetTransform();
    rebuildContentGrid();
    centerViewport();
    submitRender();
    renderIntersections();
}

bool CTiledVolumeViewer::isWindowMinimized() const
{
    auto* subWindow = qobject_cast<QMdiSubWindow*>(parentWidget());
    return subWindow && subWindow->isMinimized();
}

bool CTiledVolumeViewer::eventFilter(QObject* watched, QEvent* event)
{
    if (event->type() == QEvent::WindowStateChange) {
        auto* subWindow = qobject_cast<QMdiSubWindow*>(watched);
        if (subWindow && !subWindow->isMinimized()) {
            auto* stateEvent = static_cast<QWindowStateChangeEvent*>(event);
            if (stateEvent->oldState() & Qt::WindowMinimized) {
                if (_dirtyWhileMinimized) {
                    _dirtyWhileMinimized = false;
                    renderVisible(true);
                    updateAllOverlays();
                }
            }
        }
    }
    return QWidget::eventFilter(watched, event);
}

// ============================================================================
// Settings pass-through
// ============================================================================

void CTiledVolumeViewer::setCompositeRenderSettings(const CompositeRenderSettings& settings)
{
    if (_compositeSettings == settings) return;
    _compositeSettings = settings;
    _compositeSettings.params.updateLightDir();

    if (_volume) renderVisible(true);
    updateStatusLabel();
}

void CTiledVolumeViewer::setVolumeWindow(float low, float high)
{
    constexpr float kMax = 255.0f;
    float cLow = std::clamp(low, 0.0f, kMax);
    float cHigh = std::clamp(high, 0.0f, kMax);
    if (cHigh <= cLow) cHigh = std::min(kMax, cLow + 1.0f);
    if (std::abs(cLow - _baseWindowLow) < 1e-6f && std::abs(cHigh - _baseWindowHigh) < 1e-6f) return;
    _baseWindowLow = cLow;
    _baseWindowHigh = cHigh;

    if (_volume) renderVisible(true);
}

void CTiledVolumeViewer::setBaseColormap(const std::string& id)
{
    if (_baseColormapId == id) return;
    _baseColormapId = id;

    if (_volume) renderVisible(true);
}

void CTiledVolumeViewer::setStretchValues(bool enabled)
{
    if (_stretchValues == enabled) return;
    _stretchValues = enabled;

    if (_volume) renderVisible(true);
}

void CTiledVolumeViewer::setOverlayVolume(std::shared_ptr<Volume> vol)
{
    _overlayVolume = std::move(vol);

    if (_volume) renderVisible(true);
}

void CTiledVolumeViewer::setOverlayOpacity(float opacity)
{
    const float clamped = std::clamp(opacity, 0.0f, 1.0f);
    if (std::abs(clamped - _overlayOpacity) < 1e-6f) return;
    _overlayOpacity = clamped;

    if (_volume) renderVisible(true);
}

void CTiledVolumeViewer::setOverlayColormap(const std::string& id)
{
    if (_overlayColormapId == id) return;
    _overlayColormapId = id;

    if (_volume) renderVisible(true);
}
void CTiledVolumeViewer::setOverlayThreshold(float threshold) { setOverlayWindow(std::max(threshold, 0.0f), _overlayWindowHigh); }
void CTiledVolumeViewer::setOverlayWindow(float low, float high)
{
    const float clampedLow = std::clamp(low, 0.0f, 255.0f);
    float clampedHigh = std::clamp(high, 0.0f, 255.0f);
    if (clampedHigh <= clampedLow) {
        clampedHigh = std::min(255.0f, clampedLow + 1.0f);
    }
    if (std::abs(clampedLow - _overlayWindowLow) < 1e-6f &&
        std::abs(clampedHigh - _overlayWindowHigh) < 1e-6f) {
        return;
    }
    _overlayWindowLow = clampedLow;
    _overlayWindowHigh = clampedHigh;

    if (_volume) renderVisible(true);
}

void CTiledVolumeViewer::setResetViewOnSurfaceChange(bool reset) { _resetViewOnSurfaceChange = reset; }
void CTiledVolumeViewer::setSegmentationEditActive(bool active) { _segmentationEditActive = active; }

void CTiledVolumeViewer::setIntersectionOpacity(float opacity)
{
    _intersectionOpacity = std::clamp(opacity, 0.0f, 1.0f);
    renderIntersections();
}
void CTiledVolumeViewer::setIntersectionThickness(float thickness)
{
    _intersectionThickness = thickness;
    renderIntersections();
}
void CTiledVolumeViewer::setHighlightedSurfaceIds(const std::vector<std::string>& ids)
{
    _highlightedSurfaceIds = {ids.begin(), ids.end()};
    renderIntersections();
}
void CTiledVolumeViewer::setSurfacePatchSamplingStride(int stride)
{
    _surfacePatchSamplingStride = std::max(1, stride);
    renderIntersections();
}

void CTiledVolumeViewer::setSurfaceOverlayEnabled(bool enabled) { _surfaceOverlayEnabled = enabled; if (_volume) renderVisible(true); }
void CTiledVolumeViewer::setSurfaceOverlays(const std::map<std::string, cv::Vec3b>& overlays) { _surfaceOverlays = overlays; if (_volume && _surfaceOverlayEnabled) renderVisible(true); }
void CTiledVolumeViewer::setSurfaceOverlapThreshold(float threshold) { _surfaceOverlapThreshold = std::max(0.1f, threshold); if (_volume && _surfaceOverlayEnabled) renderVisible(true); }

// ============================================================================
// Overlay group management
// ============================================================================

void CTiledVolumeViewer::setOverlayGroup(const std::string& key, const std::vector<QGraphicsItem*>& items)
{
    clearOverlayGroup(key);
    _ov.groups[key] = items;
    // Items are already added to the scene by applyPrimitives().
    // Do NOT re-add here (matches CVolumeViewer behavior).
}

void CTiledVolumeViewer::clearOverlayGroup(const std::string& key)
{
    auto it = _ov.groups.find(key);
    if (it == _ov.groups.end()) return;
    for (auto* item : it->second) {
        if (!item) continue;
        if (item->scene()) {
            item->scene()->removeItem(item);
        }
        delete item;
    }
    _ov.groups.erase(it);
}

void CTiledVolumeViewer::clearAllOverlayGroups()
{
    for (auto& [key, items] : _ov.groups) {
        for (auto* item : items) {
            if (!item) continue;
            if (item->scene()) {
                item->scene()->removeItem(item);
            }
            delete item;
        }
    }
    _ov.groups.clear();
}

void CTiledVolumeViewer::invalidateOverlays()
{
    scheduleOverlayUpdate();
}

void CTiledVolumeViewer::scheduleOverlayUpdate()
{
    if (_overlayUpdatePending) return;
    _overlayUpdatePending = true;
    QMetaObject::invokeMethod(this, [this]() {
        _overlayUpdatePending = false;
        updateAllOverlays();
    }, Qt::QueuedConnection);
}

void CTiledVolumeViewer::updateAllOverlays()
{
    if (isWindowMinimized()) {
        _dirtyWhileMinimized = true;
        return;
    }

    invalidateVis();
    renderIntersections();
    emit overlaysUpdated();
}

// ============================================================================
// Intersection rendering
// ============================================================================

void CTiledVolumeViewer::renderIntersections()
{
    // During active interaction, throttle intersection recomputation to ~5Hz.
    // The full rebuild (invalidateIntersect + computePlaneIntersections + new
    // QGraphicsPathItems) is expensive and causes visible line pop-in.
    if (_isPanning && _intersectionThrottleTimer) {
        _intersectionsDirty = true;
        if (!_intersectionThrottleTimer->isActive()) {
            _intersectionThrottleTimer->start();
        }
        return;
    }

    renderIntersectionsCore();
}

void CTiledVolumeViewer::renderIntersectionsCore()
{
    // NOTE: old intersection items are kept visible until new ones are
    // built below.  invalidateIntersect() is called right before the new
    // items are installed, so lines never disappear mid-frame.

    auto surf = _surfWeak.lock();
    auto* plane = dynamic_cast<PlaneSurface*>(surf.get());
    if (!plane || !_state || !_tileScene || !_viewerManager) {
        invalidateIntersect();
        return;
    }

    auto* patchIndex = _viewerManager->surfacePatchIndex();
    if (!patchIndex || patchIndex->empty()) {
        invalidateIntersect();
        return;
    }

    const QRectF sceneRect = viewportSceneRect();
    if (!sceneRect.isValid()) {
        return;
    }

    std::unordered_set<SurfacePatchIndex::SurfacePtr> targets;
    auto addTarget = [&](const std::string& name) {
        auto quad = std::dynamic_pointer_cast<QuadSurface>(_state->surface(name));
        if (quad) {
            targets.insert(std::move(quad));
        }
    };

    for (const auto& name : _intersectTgts) {
        if (name == "visible_segmentation") {
            if (_highlightedSurfaceIds.empty()) {
                addTarget("segmentation");
            } else {
                for (const auto& id : _highlightedSurfaceIds) {
                    addTarget(id);
                }
            }
            continue;
        }
        addTarget(name);
    }

    if (targets.empty()) {
        invalidateIntersect();  // user removed all targets
        return;
    }

    const cv::Rect planeRoi = planeRoiFromSceneRect(_tileScene, sceneRect);
    if (planeRoi.width <= 0 || planeRoi.height <= 0) {
        return;
    }

    const auto intersections = patchIndex->computePlaneIntersections(*plane, planeRoi, targets);
    if (intersections.empty()) {
        return;
    }

    const auto& activeSeg = activeSegmentationHandle();
    const QColor activeSegColor = activeSeg.accentColor.isValid() ? activeSeg.accentColor
                                                                  : (_surfName == "seg yz"   ? QColor(COLOR_SEG_YZ)
                                                                     : _surfName == "seg xz" ? QColor(COLOR_SEG_XZ)
                                                                                             : QColor(COLOR_SEG_XY));
    auto activeSegShared = std::dynamic_pointer_cast<QuadSurface>(_state->surface("segmentation"));
    auto* segOverlay = _viewerManager->segmentationOverlay();
    const bool useApprovalMask = segOverlay && segOverlay->hasApprovalMaskData() && activeSegShared;

    // Group paths by (color, z-value) for efficient batching
    struct StyleKey {
        QRgb rgba;
        int z;
        bool operator==(const StyleKey& o) const { return rgba == o.rgba && z == o.z; }
    };
    struct StyleKeyHash {
        size_t operator()(const StyleKey& k) const { return std::hash<uint64_t>{}((uint64_t(k.rgba) << 32) | static_cast<uint64_t>(k.z)); }
    };
    std::unordered_map<StyleKey, QPainterPath, StyleKeyHash> groupedPaths;
    std::unordered_map<StyleKey, QColor, StyleKeyHash> groupedColors;

    for (const auto& [targetSurface, segments] : intersections) {
        if (!targetSurface || segments.empty()) {
            continue;
        }

        const bool isActiveSeg = activeSegShared && targetSurface == activeSegShared;
        const bool isHighlighted = _highlightedSurfaceIds.count(targetSurface->id) > 0;

        // Determine base color and z-value for this surface
        QColor baseColor;
        int zValue;
        if (isActiveSeg) {
            baseColor = activeSegColor;
            zValue = static_cast<int>(kActiveSegZ);
        } else if (isHighlighted) {
            baseColor = QColor(0, 220, 255);  // cyan
            zValue = static_cast<int>(kHighlightZ);
        } else {
            // Persistent palette color assignment
            const auto& surfId = targetSurface->id;
            size_t colorIndex;
            auto colorIt = _surfaceColorAssignments.find(surfId);
            if (colorIt != _surfaceColorAssignments.end()) {
                colorIndex = colorIt->second;
            } else if (_surfaceColorAssignments.size() < 500) {
                colorIndex = _nextColorIndex++;
                _surfaceColorAssignments[surfId] = colorIndex;
            } else {
                colorIndex = std::hash<std::string>{}(surfId);
            }
            baseColor = kIntersectionPalette[colorIndex % std::size(kIntersectionPalette)];
            zValue = static_cast<int>(kIntersectionZ);
        }

        for (const auto& segment : segments) {
            QPointF a = volumeToScene(segment.world[0]);
            QPointF b = volumeToScene(segment.world[1]);
            if (!isFinitePoint(a) || !isFinitePoint(b)) {
                continue;
            }

            QColor color = baseColor;
            float alpha = _intersectionOpacity;
            int segZ = zValue;

            if (useApprovalMask && isActiveSeg) {
                const cv::Vec3f midParam = (segment.surfaceParams[0] + segment.surfaceParams[1]) * 0.5f;
                const auto [row, col] = surfaceParamToGrid(targetSurface.get(), midParam);
                const QColor approvalColor = segOverlay->queryApprovalColor(row, col);
                if (approvalColor.isValid()) {
                    color = approvalColor;
                    alpha *= std::clamp(static_cast<float>(segOverlay->approvalMaskOpacity()) / 100.0f, 0.0f, 1.0f);
                    segZ += 5;
                }
            }

            color.setAlphaF(std::clamp(alpha, 0.0f, 1.0f));
            if (color.alpha() <= 0) {
                continue;
            }

            StyleKey key{color.rgba(), segZ};
            QPainterPath& path = groupedPaths[key];
            path.moveTo(a);
            path.lineTo(b);
            groupedColors[key] = color;
        }
    }

    std::vector<QGraphicsItem*> items;
    items.reserve(groupedPaths.size());
    for (const auto& [key, path] : groupedPaths) {
        if (path.isEmpty()) {
            continue;
        }
        auto* item = new QGraphicsPathItem(path);
        QPen pen(groupedColors[key]);
        pen.setWidthF(static_cast<qreal>(_intersectionThickness));
        pen.setCapStyle(Qt::RoundCap);
        pen.setJoinStyle(Qt::RoundJoin);
        item->setPen(pen);
        item->setBrush(Qt::NoBrush);
        item->setZValue(key.z);
        _scene->addItem(item);
        items.push_back(item);
    }

    // Remove old items and install new ones atomically — lines never disappear
    invalidateIntersect();
    if (!items.empty()) {
        _ov.intersectItems[kIntersectionItemsKey] = std::move(items);
        fGraphicsView->viewport()->repaint();
    }
}
void CTiledVolumeViewer::invalidateVis()
{
    for (auto* item : _ov.sliceVisItems) {
        _scene->removeItem(item);
        delete item;
    }
    _ov.sliceVisItems.clear();
}
void CTiledVolumeViewer::onIntersectionComputeFinished()
{
    // Stub — async intersection results handled by renderIntersections() directly
    _intersectionPending = false;
    if (_intersectionRerunNeeded) {
        _intersectionRerunNeeded = false;
        renderIntersections();
    }
}

void CTiledVolumeViewer::invalidateIntersect(const std::string& /*name*/)
{
    for (auto& [key, items] : _ov.intersectItems) {
        for (auto* item : items) {
            if (!item) {
                continue;
            }
            if (item->scene()) {
                item->scene()->removeItem(item);
            }
            delete item;
        }
    }
    _ov.intersectItems.clear();
}

void CTiledVolumeViewer::onPathsChanged(const QList<ViewerOverlayControllerBase::PathPrimitive>& paths)
{
    _paths.clear();
    _paths.reserve(static_cast<size_t>(paths.size()));
    for (const auto& path : paths) _paths.push_back(path);
    scheduleOverlayUpdate();
}

void CTiledVolumeViewer::onCollectionSelected(uint64_t id)
{
    _selectedCollectionId = id;
    scheduleOverlayUpdate();
}

void CTiledVolumeViewer::onPointSelected(uint64_t pointId)
{
    if (_selectedPointId == pointId) return;
    _selectedPointId = pointId;
    scheduleOverlayUpdate();
}

void CTiledVolumeViewer::onDrawingModeActive(bool active, float brushSize, bool isSquare)
{
    _drawingModeActive = active;
    _brushSize = brushSize;
    _brushIsSquare = isSquare;
    if (_ov.cursor) {
        _scene->removeItem(_ov.cursor);
        delete _ov.cursor;
        _ov.cursor = nullptr;
    }
    if (!_state) return;
    POI* cursor = _state->poi("cursor");
    if (cursor) onPOIChanged("cursor", cursor);
}

void CTiledVolumeViewer::markActiveSegmentationDirty()
{
    _activeSegHandleDirty = true;
    _activeSegHandle.reset();
}

const CTiledVolumeViewer::ActiveSegmentationHandle& CTiledVolumeViewer::activeSegmentationHandle() const
{
    if (!_activeSegHandleDirty) return _activeSegHandle;

    ActiveSegmentationHandle handle;
    handle.slotName = "segmentation";
    handle.viewerIsSegmentationView = (_surfName == "segmentation");
    handle.accentColor =
        (_surfName == "seg yz"   ? QColor(COLOR_SEG_YZ)
         : _surfName == "seg xz" ? QColor(COLOR_SEG_XZ)
                                  : QColor(COLOR_SEG_XY));
    if (_state) {
        auto surfHolder = _state->surface(handle.slotName);
        handle.surface = dynamic_cast<QuadSurface*>(surfHolder.get());
    }
    if (!handle.surface) handle.slotName.clear();

    _activeSegHandle = handle;
    _activeSegHandleDirty = false;
    return _activeSegHandle;
}

// BBox stubs
void CTiledVolumeViewer::setBBoxMode(bool enabled)
{
    _bboxMode = enabled;
    if (!enabled && _activeBBoxSceneRect) {
        _activeBBoxSceneRect.reset();
        scheduleOverlayUpdate();
    }
}

QuadSurface* CTiledVolumeViewer::makeBBoxFilteredSurfaceFromSceneRect(const QRectF& /*sceneRect*/)
{
    // TODO: port from CVolumeViewer (Phase 4)
    return nullptr;
}

auto CTiledVolumeViewer::selections() const -> std::vector<std::pair<QRectF, QColor>>
{
    std::vector<std::pair<QRectF, QColor>> out;
    out.reserve(_selections.size());
    for (const auto& s : _selections) {
        QPointF topLeft = _tileScene->surfaceToScene(static_cast<float>(s.surfRect.left()), static_cast<float>(s.surfRect.top()));
        QPointF botRight = _tileScene->surfaceToScene(static_cast<float>(s.surfRect.right()), static_cast<float>(s.surfRect.bottom()));
        QRectF sceneRect(topLeft, botRight);
        out.emplace_back(sceneRect.normalized(), s.color);
    }
    return out;
}

void CTiledVolumeViewer::clearSelections()
{
    _selections.clear();
    scheduleOverlayUpdate();
}
