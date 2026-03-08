#include "CTiledVolumeViewer.hpp"
#include <utils/hash.hpp>

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
#include <QScrollBar>
#include <QGuiApplication>
#include <QPointer>

#include <algorithm>
#include <array>
#include <cmath>

#include <QDebug>
#include "vc/core/cache/TieredChunkCache.hpp"

constexpr auto COLOR_CURSOR = Qt::cyan;
#define COLOR_FOCUS QColor(50, 255, 215)
#define COLOR_SEG_YZ Qt::yellow
#define COLOR_SEG_XZ Qt::red
#define COLOR_SEG_XY QColor(255, 140, 0)

namespace
{
constexpr qreal kIntersectionZ = 18.0;
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
    fGraphicsView->setRenderHint(QPainter::Antialiasing);
    fGraphicsView->setScrollPanDisabled(true);

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
    connect(fGraphicsView, &CVolumeViewerView::sendKeyRelease, this, &CTiledVolumeViewer::onKeyRelease);

    // Create fixed-size scene
    _scene = new QGraphicsScene(this);

    // Create tile scene manager
    _tileScene = new TileScene(_scene);

    // Create render controller (async tile rendering, shared pool from ViewerManager)
    _renderController = new TileRenderController(_tileScene, manager->renderPool(), this);

    // Set the scene on the view
    fGraphicsView->setScene(_scene);

    // Force viewport repaint when progressive refinement updates tiles.
    // QGraphicsPixmapItem::setPixmap() should auto-trigger this, but the
    // explicit signal guarantees the view refreshes when updates arrive
    // while the view is otherwise idle (no user interaction).
    connect(_renderController, &TileRenderController::sceneNeedsUpdate,
            this, [this]() { fGraphicsView->viewport()->update(); });

    // Read settings
    using namespace vc3d::settings;
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    _camera.downscaleOverride = settings.value(perf::DOWNSCALE_OVERRIDE, perf::DOWNSCALE_OVERRIDE_DEFAULT).toInt();
    _useFastInterpolation = settings.value(perf::FAST_INTERPOLATION, perf::FAST_INTERPOLATION_DEFAULT).toBool();

    auto* layout = new QVBoxLayout;
    layout->addWidget(fGraphicsView);
    setLayout(layout);

    // Wire callbacks into the unified tick on the render controller
    _renderController->setOverlayCallback([this]() { updateAllOverlays(); });
    _renderController->setZoomSettleCallback([this]() {
        fGraphicsView->resetTransform();
        rebuildContentGrid();
        centerViewport();
        _camera.invalidate();
        submitRender();
        renderIntersections();
    });

    _lbl = new QLabel(this);
    _lbl->setStyleSheet("QLabel { color : #00FF00; background-color: rgba(0,0,0,128); padding: 2px 4px; }");
    _lbl->setMinimumWidth(300);
    _lbl->move(10, 5);

    // Periodic status refresh so download/queue counters update even when idle
    auto* statusTimer = new QTimer(this);
    connect(statusTimer, &QTimer::timeout, this, &CTiledVolumeViewer::updateStatusLabel);
    statusTimer->start(250);

    _interactionSettleTimer = new QTimer(this);
    _interactionSettleTimer->setSingleShot(true);
    _interactionSettleTimer->setInterval(125);
    connect(_interactionSettleTimer, &QTimer::timeout, this, &CTiledVolumeViewer::settleInteractionRender);
}

CTiledVolumeViewer::~CTiledVolumeViewer()
{
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
    // Invalidate all caches and cancel in-flight work from the old volume
    _renderController->cancelAll();
    _renderController->clearState();
    _renderController->sliceCache().clear();
    _tileScene->resetMetadata();

    // Remove old chunk-ready listener before switching volumes
    if (_chunkCbId != 0 && _volume && _volume->tieredCache()) {
        _volume->tieredCache()->removeChunkReadyListener(_chunkCbId);
        _chunkCbId = 0;
    }

    _volume = vol;

    // Reset pin progress tracking
    _pinTotal = 0;
    _pinReceived = 0;
    _pinLevel = -1;
    _hadValidDataBounds = false;

    // Set up tiered chunk cache for progressive rendering
    if (_volume && _volume->numScales() >= 1) {
        // Wire chunk-ready listener so arriving chunks trigger refinement.
        auto* ctrl = _renderController;
        auto* cache = _volume->tieredCache();
        QPointer<CTiledVolumeViewer> viewerGuard(this);
        _chunkCbId = cache->addChunkReadyListener(
            [ctrl, viewerGuard, cache](const vc::cache::ChunkKey& key) {
                (void)key;
                QMetaObject::invokeMethod(ctrl, [ctrl, cache]() {
                    ctrl->markChunkArrived();
                    cache->clearChunkArrivedFlag();
                }, Qt::QueuedConnection);
                QMetaObject::invokeMethod(qApp, [viewerGuard]() {
                    if (viewerGuard) {
                        viewerGuard->updateStatusLabel();
                    }
                }, Qt::QueuedConnection);
            });
    }

    onPinComplete();
    updateStatusLabel();
}

void CTiledVolumeViewer::onSurfaceChanged(std::string name, std::shared_ptr<Surface> surf,
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
            updateParamsHash();
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
                updateParamsHash();
            } else if (isInPlacePlaneUpdate) {
                _surfaceContentVersion = 0;
                updateParamsHash();
                updateContentMinScale();
                rebuildContentGrid();
                centerViewport();
                _renderController->setAtomicNextEpochSwap(true);
            } else {
                _surfaceContentVersion = 0;
                updateParamsHash();
                if (!isEditUpdate) {
                    _camera.zOff = 0.0f;
                }

                // Full surface changes still need a full render reset.
                _renderController->cancelAll();
                _renderController->sliceCache().clear();
                _tileScene->clearAll();

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

    _renderController->markOverlaysDirty();
    if (name == "segmentation" || name == _surfName) {
        renderIntersections();
    }
}

void CTiledVolumeViewer::onVolumeClosing()
{
    _renderController->cancelAll();
    _renderController->clearState();

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
        _contentBounds = ContentBounds{};
    } else {
        onSurfaceChanged(_surfName, nullptr);
    }
}

void CTiledVolumeViewer::onSurfaceWillBeDeleted(std::string /*name*/, std::shared_ptr<Surface> surf)
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
    float vpW = vpSize.width();
    float vpH = vpSize.height();
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
            contentW = pts.cols / sc[0];
            contentH = pts.rows / sc[1];
        }
    }

    if (contentW <= 0 || contentH <= 0) {
        _contentMinScale = TiledViewerCamera::MIN_SCALE;
        return;
    }

    // Scale at which content just fills viewport (fit, not cover)
    float fitScale = std::min(vpW / contentW, vpH / contentH);
    _contentMinScale = std::max(fitScale, TiledViewerCamera::MIN_SCALE);
}

void CTiledVolumeViewer::beginInteractionRender()
{
    _interactionQualityActive = true;
    if (_interactionSettleTimer) {
        _interactionSettleTimer->start();
    }
}

void CTiledVolumeViewer::settleInteractionRender()
{
    if (_isPanning || !_interactionQualityActive) {
        return;
    }

    _interactionQualityActive = false;
    _camera.invalidate();
    submitRender();
    renderIntersections();
    updateStatusLabel();
}

bool CTiledVolumeViewer::interactionRenderActive() const
{
    return _isPanning || _interactionQualityActive;
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
            float x0 = 0, x1 = (float)w, y0 = 0, y1 = (float)h, z0 = 0, z1 = (float)d;
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
            float halfW = (pts.cols * 0.5f) / sc[0];
            float halfH = (pts.rows * 0.5f) / sc[1];
            contentMinX = -halfW;
            contentMinY = -halfH;
            contentMaxX = halfW;
            contentMaxY = halfH;
        }
    }

    // Compute content bounds
    ContentBounds bounds;
    bounds.scale = _camera.scale;
    bounds.worldTileSize = static_cast<float>(TileScene::TILE_PX) / _camera.scale;

    if (bounds.worldTileSize > 0 && contentMaxX > contentMinX && contentMaxY > contentMinY) {
        bounds.firstWorldCol = static_cast<int>(std::floor(contentMinX / bounds.worldTileSize));
        bounds.firstWorldRow = static_cast<int>(std::floor(contentMinY / bounds.worldTileSize));
        int lastCol = static_cast<int>(std::floor(contentMaxX / bounds.worldTileSize));
        int lastRow = static_cast<int>(std::floor(contentMaxY / bounds.worldTileSize));
        bounds.totalCols = lastCol - bounds.firstWorldCol + 1;
        bounds.totalRows = lastRow - bounds.firstWorldRow + 1;
    }

    _contentBounds = bounds;

    QSize vpSize = fGraphicsView->viewport()->size();
    _tileScene->rebuildGrid(bounds, vpSize.width(), vpSize.height());
}

void CTiledVolumeViewer::centerViewport()
{
    if (!fGraphicsView || !_tileScene) return;
    QPointF scenePos = _tileScene->surfaceToScene(_camera.surfacePtr[0], _camera.surfacePtr[1]);
    fGraphicsView->centerOn(scenePos);
}

void CTiledVolumeViewer::onPinComplete()
{
    if (!_volume) return;

    _hadValidDataBounds = false;

    // Create a default PlaneSurface immediately so remote volumes render
    // without waiting for a coarsest-level bounds scan.
    if (!_surfWeak.lock() && _volume && isAxisAlignedView()) {
        auto shape = _volume->shape();  // {width, height, slices} = {x, y, z}
        cv::Vec3f center(shape[0] * 0.5f, shape[1] * 0.5f, shape[2] * 0.5f);
        cv::Vec3f normal;
        if (_surfName == "xy plane") normal = cv::Vec3f(0, 0, 1);
        else if (_surfName == "xz plane" || _surfName == "seg xz") normal = cv::Vec3f(0, 1, 0);
        else normal = cv::Vec3f(1, 0, 0);  // yz plane / seg yz
        auto defaultSurf = std::make_shared<PlaneSurface>(center, normal);
        _defaultSurface = defaultSurf;
        _surfWeak = defaultSurf;
    }

    _camera.recalcPyramidLevel(_volume->numScales());
    updateContentMinScale();

    rebuildContentGrid();
    centerViewport();

    // Update scalebar
    double vs = _volume->voxelSize() / _camera.dsScale;
    fGraphicsView->setVoxelSize(vs, vs);

    submitRender();
    renderIntersections();
    updateStatusLabel();

    // Trigger background prefetch of pyramid levels if configured.
    // Setting value = how many of the coarsest levels to download.
    // E.g., with levels 0-5: setting=1 downloads 5/, setting=2 downloads 5/+4/, etc.
    // Level 5 (coarsest) is already pinned above, so we prefetch the rest.
    if (_volume && _volume->isRemote()) {
        using namespace vc3d::settings;
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        int prefetchCount = settings.value(perf::PREFETCH_LEVELS, perf::PREFETCH_LEVELS_DEFAULT).toInt();
        if (prefetchCount > 1) {
            int maxLevel = static_cast<int>(_volume->numScales()) - 1;
            // Coarsest (maxLevel) is already pinned; prefetch the next ones down
            int fromLevel = std::max(0, maxLevel - prefetchCount + 1);
            int toLevel = maxLevel - 1;
            if (toLevel >= fromLevel) {
                _volume->prefetchLevels(fromLevel, toLevel);
            }
        }
    }
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
        if (plane) {
            cv::Vec3f center((db.minX + db.maxX) * 0.5f,
                             (db.minY + db.maxY) * 0.5f,
                             (db.minZ + db.maxZ) * 0.5f);
            plane->setOrigin(center);
        }

        // Update focus POI — cascades to all axis-aligned viewers
        POI* focus = _state->poi("focus");
        if (focus) {
            cv::Vec3f center((db.minX + db.maxX) * 0.5f,
                             (db.minY + db.maxY) * 0.5f,
                             (db.minZ + db.maxZ) * 0.5f);
            focus->p = center;
            _state->setPOI("focus", focus);
        }
    }

    // Clear stale tiles and caches, then rebuild with new tight bounds
    _renderController->cancelAll();
    _renderController->sliceCache().clear();
    _tileScene->clearAll();
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
    return fGraphicsView->mapToScene(fGraphicsView->viewport()->rect()).boundingRect();
}

// ============================================================================
// Navigation
// ============================================================================

void CTiledVolumeViewer::panBy(int dx, int dy)
{
    const float invScale = 1.0f / _camera.scale;
    _camera.surfacePtr[0] -= dx * invScale;
    _camera.surfacePtr[1] -= dy * invScale;

    // Clamp pan to content bounds (derived from data bounds)
    if (_contentBounds.totalCols > 0 && _contentBounds.totalRows > 0) {
        float minU = _contentBounds.firstWorldCol * _contentBounds.worldTileSize;
        float maxU = (_contentBounds.firstWorldCol + _contentBounds.totalCols) * _contentBounds.worldTileSize;
        float minV = _contentBounds.firstWorldRow * _contentBounds.worldTileSize;
        float maxV = (_contentBounds.firstWorldRow + _contentBounds.totalRows) * _contentBounds.worldTileSize;
        _camera.surfacePtr[0] = std::clamp(_camera.surfacePtr[0], minU, maxU);
        _camera.surfacePtr[1] = std::clamp(_camera.surfacePtr[1], minV, maxV);
    }


    centerViewport();
    submitRender();
    scheduleOverlayUpdate();
    updateStatusLabel();
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
    const float newScale = std::max(
        TiledViewerCamera::stepScale(_camera.scale, steps), _contentMinScale);
    if (std::abs(newScale - _camera.scale) < 0.001f) {
        return;
    }

    // Convert scene position to viewport coordinates for zoom-at-point math
    QPointF vpPos = fGraphicsView->mapFromScene(scenePos);
    QSize vpSize = fGraphicsView->viewport()->size();
    const float vpCenterX = vpSize.width() * 0.5f;
    const float vpCenterY = vpSize.height() * 0.5f;
    const float dx = static_cast<float>(vpPos.x()) - vpCenterX;
    const float dy = static_cast<float>(vpPos.y()) - vpCenterY;

    _camera.surfacePtr[0] += dx * (1.0f / _camera.scale - 1.0f / newScale);
    _camera.surfacePtr[1] += dy * (1.0f / _camera.scale - 1.0f / newScale);
    _camera.scale = newScale;

    if (_volume) {
        float oldDs = _camera.dsScale;
        _camera.recalcPyramidLevel(_volume->numScales());
        if (std::abs(_camera.dsScale - oldDs) > 1e-6f) {
            double vs = _volume->voxelSize() / _camera.dsScale;
            fGraphicsView->setVoxelSize(vs, vs);
        }
    }

    // Every stop change is significant — always do a full re-render
    _renderController->cancelZoomSettle();
    fGraphicsView->resetTransform();
    rebuildContentGrid();
    centerViewport();
    _camera.invalidate();
    submitRender();
    renderIntersections();
}

void CTiledVolumeViewer::setSliceOffset(float dz)
{
    beginInteractionRender();
    _camera.zOff += dz;
    _camera.invalidate();
    submitRender();
}

void CTiledVolumeViewer::onZoom(int steps, QPointF scene_point, Qt::KeyboardModifiers modifiers)
{
    auto surf = _surfWeak.lock();
    if (!surf) return;

    if (_segmentationEditActive && (modifiers & Qt::ControlModifier)) {
        cv::Vec3f world = sceneToVolume(scene_point);
        emit sendSegmentationRadiusWheel(steps, scene_point, world);
        return;
    }

    if (modifiers & Qt::ShiftModifier) {
        if (steps == 0) return;

        PlaneSurface* plane = dynamic_cast<PlaneSurface*>(surf.get());
        int stepSize = _viewerManager ? _viewerManager->sliceStepSize() : 1;
        int adjustedSteps = steps * stepSize;

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
        // One zoom stop per scroll tick, regardless of scroll delta magnitude
        int zoomDir = (steps > 0) ? 1 : (steps < 0) ? -1 : 0;
        if (zoomDir != 0) {
            zoomStepsAt(zoomDir, scene_point);
        }
    }

    updateStatusLabel();
    invalidateOverlays();
}

void CTiledVolumeViewer::adjustZoomByFactor(float factor)
{
    auto surf = _surfWeak.lock();
    if (!surf) return;

    int steps = (factor > 1.0f) ? 1 : (factor < 1.0f) ? -1 : 0;
    if (steps == 0) return;

    // Zoom centered on viewport center
    QSize vpSize = fGraphicsView->viewport()->size();
    QPointF vpCenter(vpSize.width() * 0.5, vpSize.height() * 0.5);
    QPointF sceneCenter = fGraphicsView->mapToScene(vpCenter.toPoint());
    zoomStepsAt(steps, sceneCenter);

    updateStatusLabel();
    invalidateOverlays();
}

void CTiledVolumeViewer::adjustSurfaceOffset(float dn)
{
    setSliceOffset(dn);
    updateStatusLabel();
    invalidateOverlays();
}

void CTiledVolumeViewer::resetSurfaceOffsets()
{
    _camera.zOff = 0.0f;
    _camera.invalidate();
    submitRender();
    updateStatusLabel();
    invalidateOverlays();
}

// ============================================================================
// Pan handling via CVolumeViewerView events
// ============================================================================

void CTiledVolumeViewer::onPanStart(Qt::MouseButton /*buttons*/, Qt::KeyboardModifiers /*modifiers*/)
{
    _isPanning = true;
    if (_interactionSettleTimer) {
        _interactionSettleTimer->stop();
    }
    // The view handles pan tracking with _last_pan_position internally,
    // but since we disabled scrollbars, its scroll-based panning won't work.
    // We need to intercept the mouse move deltas instead.
    // Store current mouse pos for delta computation
    _lastPanPos = QCursor::pos();

    // Record viewport center in scene coords for predictive prefetch
    QRectF vp = viewportSceneRect();
    _lastPanScenePos = vp.center();
}

void CTiledVolumeViewer::onPanRelease(Qt::MouseButton /*buttons*/, Qt::KeyboardModifiers /*modifiers*/)
{
    _isPanning = false;
    if (!_interactionQualityActive) {
        _camera.invalidate();
        submitRender();
    }
    _renderController->markOverlaysDirty();
    renderIntersections();
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
    renderIntersections();
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

void CTiledVolumeViewer::submitRender()
{
    auto surf = _surfWeak.lock();
    if (!surf || !_volume || !_volume->zarrDataset()) {
        return;
    }

    if (_tileScene->cols() == 0 || _tileScene->rows() == 0) {
        return;
    }

    QRectF vpRect = viewportSceneRect();

    // Update center marker
    if (!_ov.centerMarker) {
        _ov.centerMarker = _scene->addEllipse({-10, -10, 20, 20},
            QPen(COLOR_FOCUS, 3, Qt::DashDotLine, Qt::RoundCap, Qt::RoundJoin));
        _ov.centerMarker->setZValue(11);
    }
    QPointF centerScene = _tileScene->surfaceToScene(_camera.surfacePtr[0], _camera.surfacePtr[1]);
    _ov.centerMarker->setPos(centerScene);

    // Submit to async render controller
    auto buildParams = [this](const WorldTileKey& wk) { return buildRenderParams(wk); };
    _renderController->scheduleRender(_camera, surf, _volume, buildParams, vpRect);

    // Compute prefetch viewport: extend in pan direction for predictive prefetch
    QRectF prefetchRect = vpRect;
    if (_isPanning) {
        QPointF currentCenter = vpRect.center();
        QPointF delta = currentCenter - _lastPanScenePos;
        _lastPanScenePos = currentCenter;

        // Extend viewport by 3 tile widths in the direction of pan movement
        // so prefetch stays ahead of rapid panning.
        constexpr int PREFETCH_TILES_AHEAD = 3;
        const float tileExtent = static_cast<float>(TileScene::TILE_PX) * PREFETCH_TILES_AHEAD;
        if (std::abs(delta.x()) > 0.5 || std::abs(delta.y()) > 0.5) {
            // Normalize delta and scale by tile extent
            double len = std::sqrt(delta.x() * delta.x() + delta.y() * delta.y());
            double nx = delta.x() / len;
            double ny = delta.y() / len;
            double extX = nx * tileExtent;
            double extY = ny * tileExtent;

            // Extend the rect in the direction of movement
            if (extX > 0) prefetchRect.setRight(prefetchRect.right() + extX);
            else           prefetchRect.setLeft(prefetchRect.left() + extX);
            if (extY > 0) prefetchRect.setBottom(prefetchRect.bottom() + extY);
            else           prefetchRect.setTop(prefetchRect.top() + extY);
        }
    }

    // Viewport-aware prefetch
    if (_volume->tieredCache()) {
        cv::Vec3f lo, hi;
        auto* plane = dynamic_cast<PlaneSurface*>(surf.get());
        bool ok = plane
            ? computePlanePrefetchBBox(plane, prefetchRect, lo, hi)
            : computeQuadPrefetchBBox(surf, prefetchRect, lo, hi);
        if (ok) {
            _volume->prefetchWorldBBox(lo, hi, _camera.dsScaleIdx);
        }
    }
}

bool CTiledVolumeViewer::computePlanePrefetchBBox(PlaneSurface* plane,
                                                   const QRectF& prefetchRect,
                                                   cv::Vec3f& lo, cv::Vec3f& hi) const
{
    const float invScale = 1.0f / _camera.scale;
    const float margin = TileScene::TILE_PX * invScale;

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
    float margin = TileScene::TILE_PX / _camera.scale;
    for (int i = 0; i < 3; i++) {
        lo[i] -= margin;
        hi[i] += margin;
    }

    return clampToDataBounds(lo, hi);
}

void CTiledVolumeViewer::updateParamsHash()
{
    auto h = utils::hash_combine_values(
        _surfaceContentVersion,
        _baseWindowLow, _baseWindowHigh, _stretchValues,
        _baseColormapId, _useFastInterpolation,
        // Composite settings — all fields that affect rendered output
        _compositeSettings.enabled,
        _compositeSettings.layersFront, _compositeSettings.layersBehind,
        _compositeSettings.reverseDirection,
        _compositeSettings.planeEnabled,
        _compositeSettings.planeLayersFront, _compositeSettings.planeLayersBehind,
        _compositeSettings.useVolumeGradients,
        // CompositeParams
        _compositeSettings.params.method,
        _compositeSettings.params.isoCutoff,
        _compositeSettings.params.alphaMin, _compositeSettings.params.alphaMax,
        _compositeSettings.params.alphaOpacity, _compositeSettings.params.alphaCutoff,
        _compositeSettings.params.blExtinction, _compositeSettings.params.blEmission,
        _compositeSettings.params.blAmbient,
        _compositeSettings.params.lightingEnabled,
        _compositeSettings.params.lightAzimuth, _compositeSettings.params.lightElevation,
        _compositeSettings.params.lightDiffuse, _compositeSettings.params.lightAmbient,
        // Postprocessing
        _compositeSettings.postStretchValues,
        _compositeSettings.postRemoveSmallComponents,
        _compositeSettings.postMinComponentSize);

    _renderController->setParamsHash(h);
}

TileRenderParams CTiledVolumeViewer::buildRenderParams(const WorldTileKey& wk) const
{
    TileRenderParams params;
    params.worldKey = wk;
    params.epoch = _camera.epoch;

    const bool interactionActive = interactionRenderActive();
    uint64_t cacheIdentity = _renderController ? _renderController->paramsHash() : 0;
    cacheIdentity = utils::hash_combine_values(cacheIdentity, interactionActive);
    params.cacheIdentity = cacheIdentity;

    // Surface parameter ROI from world tile coordinates
    params.surfaceROI.x = wk.worldCol * _contentBounds.worldTileSize;
    params.surfaceROI.y = wk.worldRow * _contentBounds.worldTileSize;
    params.surfaceROI.width = _contentBounds.worldTileSize;
    params.surfaceROI.height = _contentBounds.worldTileSize;

    params.tileW = TileScene::TILE_PX;
    params.tileH = TileScene::TILE_PX;

    params.scale = _camera.scale;
    params.dsScale = _camera.dsScale;
    params.dsScaleIdx = _camera.dsScaleIdx;
    params.zOff = _camera.zOff;

    if (interactionActive && _volume) {
        const int maxLevel = std::max(0, static_cast<int>(_volume->numScales()) - 1);
        params.dsScaleIdx = std::min(_camera.dsScaleIdx + 1, maxLevel);
        params.dsScale = std::pow(2.0f, -params.dsScaleIdx);
    }

    params.windowLow = _baseWindowLow;
    params.windowHigh = _baseWindowHigh;
    params.stretchValues = _stretchValues;
    params.colormapId = _baseColormapId;
    params.useFastInterpolation = _useFastInterpolation || interactionActive;
    params.compositeSettings = _compositeSettings;

    return params;
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

// ============================================================================
// Mouse handlers
// ============================================================================

void CTiledVolumeViewer::onCursorMove(QPointF scene_loc)
{
    auto surf = _surfWeak.lock();
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
        QPoint currentPos = QCursor::pos();
        QPoint delta = _lastPanPos - currentPos;
        _lastPanPos = currentPos;
        if (delta.x() != 0 || delta.y() != 0) {
            panBy(-delta.x(), -delta.y());
        }

        const QPoint viewportPos = fGraphicsView->viewport()->mapFromGlobal(currentPos);
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
                    QPointF ptScene = volumeToScene(pt.p);
                    QPointF diff = scene_loc - ptScene;
                    float distSq = QPointF::dotProduct(diff, diff);
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
    onCursorMove(scene_loc);

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
        onCursorMove(scene_loc);
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

void CTiledVolumeViewer::onKeyRelease(int key, Qt::KeyboardModifiers /*modifiers*/)
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

void CTiledVolumeViewer::onPOIChanged(std::string name, POI* poi)
{
    auto surf = _surfWeak.lock();
    if (!poi || !surf) return;

    if (name == "focus") {
        if (auto* plane = dynamic_cast<PlaneSurface*>(surf.get())) {
            beginInteractionRender();
            const bool originChanged = (poi->p != plane->origin());
            if (originChanged) {
                plane->setOrigin(poi->p);
            }

            // Plane viewers should center on the new focus point itself,
            // not preserve the old pan offset from the previous plane origin.
            _camera.surfacePtr[0] = 0.0f;
            _camera.surfacePtr[1] = 0.0f;
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
            float dist = quad->pointTo(ptr, poi->p, 4.0, 100, patchIndex);
            if (dist < 4.0) {
                // Center camera on the focus point
                cv::Vec3f loc = quad->loc(ptr);
                _camera.surfacePtr[0] = loc[0];
                _camera.surfacePtr[1] = loc[1];
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
                _ov.cursor->setOpacity(1.0 - dist * _camera.scale / 20.0);
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

    status += QString(" z=%1").arg(_camera.zOff, 0, 'f', 1);

    if (_compositeSettings.enabled) {
        QString method = QString::fromStdString(_compositeSettings.params.method);
        if (!method.isEmpty())
            method[0] = method[0].toUpper();
        status += QString(" | %1(%2)").arg(method).arg(
            _compositeSettings.layersFront + _compositeSettings.layersBehind);
    }

    // Cache stats
    {
        auto& sc = _renderController->sliceCache();
        size_t scTotal = sc.hits() + sc.misses();
        if (scTotal > 0) {
            int scPct = static_cast<int>(100 * sc.hits() / scTotal);
            status += QString(" | S%1").arg(scPct);
        }
    }
    if (_volume && _volume->tieredCache()) {
        auto s = _volume->tieredCache()->stats();

        // Hit ratios
        uint64_t total = s.hotHits + s.warmHits + s.coldHits + s.iceFetches + s.misses;
        if (total > 0) {
            auto pct = [&](uint64_t n) { return static_cast<int>(100 * n / total); };
            status += QString(" | H%1 W%2 D%3")
                .arg(pct(s.hotHits)).arg(pct(s.warmHits)).arg(pct(s.coldHits));
        }

        // RAM usage (hot + warm)
        double hotGB = s.hotBytes / (1024.0 * 1024.0 * 1024.0);
        double warmGB = s.warmBytes / (1024.0 * 1024.0 * 1024.0);
        status += QString(" | ram %1+%2G").arg(hotGB, 0, 'f', 1).arg(warmGB, 0, 'f', 1);

        // Disk cache
        if (s.diskFiles > 0) {
            double diskGB = s.diskBytes / (1024.0 * 1024.0 * 1024.0);
            status += QString(" | disk %1 (%2G)").arg(s.diskFiles).arg(diskGB, 0, 'f', 1);
        }

        // Negative cache
        if (s.negativeCount > 0)
            status += QString(" | neg %1").arg(s.negativeCount);

        // Queue & downloads
        if (s.ioPending > 0)
            status += QString(" | q%1").arg(s.ioPending);
        if (s.iceFetches > 0)
            status += QString(" dl%1").arg(s.iceFetches);
    } else if (_pinTotal > 0 && _pinReceived < _pinTotal) {
        status += QString(" | downloading %1/%2").arg(_pinReceived).arg(_pinTotal);
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
        if (_volume) _camera.recalcPyramidLevel(_volume->numScales());
        _camera.invalidate();
        updateStatusLabel();
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
                if (pts(j, i)[0] != -1 && std::isfinite(pts(j, i)[0])) {
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
        if (_volume) _camera.recalcPyramidLevel(_volume->numScales());
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
    float vpW = vpSize.width();
    float vpH = vpSize.height();
    if (vpW <= 0 || vpH <= 0) return;

    float fitFactor = 0.8f;
    float reqScaleX = (vpW * fitFactor) / validSurfW;
    float reqScaleY = (vpH * fitFactor) / validSurfH;
    _camera.scale = TiledViewerCamera::roundScale(std::min(reqScaleX, reqScaleY));

    // Center on the valid region.
    // surfacePtr is in grid units; (0,0,0) maps to the grid center (cols/2, rows/2).
    // To center on grid position (gx, gy): surfacePtr = (gx - cols/2, gy - rows/2, 0)
    float gridCenterX = (colMin + colMax) * 0.5f;
    float gridCenterY = (rowMin + rowMax) * 0.5f;
    _camera.surfacePtr[0] = gridCenterX - pts.cols * 0.5f;
    _camera.surfacePtr[1] = gridCenterY - pts.rows * 0.5f;
    _camera.surfacePtr[2] = 0;

    if (_volume) _camera.recalcPyramidLevel(_volume->numScales());
    _camera.invalidate();
    updateStatusLabel();
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
    updateParamsHash();
    // Params changed: clear cache and re-render
    auto surf = _surfWeak.lock();
    if (_volume && surf) {
        _camera.invalidate();
        _renderController->onParamsChanged(_camera, surf, _volume,
            [this](const WorldTileKey& wk) { return buildRenderParams(wk); },
            viewportSceneRect());
    }
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
    updateParamsHash();
    if (_volume) renderVisible(true);
}

void CTiledVolumeViewer::setBaseColormap(const std::string& id)
{
    if (_baseColormapId == id) return;
    _baseColormapId = id;
    updateParamsHash();
    if (_volume) renderVisible(true);
}

void CTiledVolumeViewer::setStretchValues(bool enabled)
{
    if (_stretchValues == enabled) return;
    _stretchValues = enabled;
    updateParamsHash();
    if (_volume) renderVisible(true);
}

void CTiledVolumeViewer::setOverlayVolume(std::shared_ptr<Volume> vol) { _overlayVolume = std::move(vol); }
void CTiledVolumeViewer::setOverlayOpacity(float opacity) { _overlayOpacity = std::clamp(opacity, 0.0f, 1.0f); }
void CTiledVolumeViewer::setOverlayColormap(const std::string& id) { _overlayColormapId = id; }
void CTiledVolumeViewer::setOverlayThreshold(float threshold) { setOverlayWindow(std::max(threshold, 0.0f), _overlayWindowHigh); }
void CTiledVolumeViewer::setOverlayWindow(float low, float high) { _overlayWindowLow = low; _overlayWindowHigh = high; }

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
    _renderController->markOverlaysDirty();
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
// Stubs for functionality ported in later phases
// ============================================================================

void CTiledVolumeViewer::renderIntersections()
{
    invalidateIntersect();

    auto surf = _surfWeak.lock();
    auto* plane = dynamic_cast<PlaneSurface*>(surf.get());
    if (!plane || !_state || !_tileScene || !_viewerManager) {
        return;
    }

    auto* patchIndex = _viewerManager->surfacePatchIndex();
    if (!patchIndex || patchIndex->empty()) {
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
    const QColor defaultColor = activeSeg.accentColor.isValid() ? activeSeg.accentColor : QColor(COLOR_SEG_XY);
    auto activeSegShared = std::dynamic_pointer_cast<QuadSurface>(_state->surface("segmentation"));
    auto* segOverlay = _viewerManager->segmentationOverlay();
    const bool useApprovalMask = segOverlay && segOverlay->hasApprovalMaskData() && activeSegShared;

    std::unordered_map<QRgb, QPainterPath> groupedPaths;
    std::unordered_map<QRgb, QColor> groupedColors;

    for (const auto& [targetSurface, segments] : intersections) {
        if (!targetSurface || segments.empty()) {
            continue;
        }

        QColor baseColor = (activeSeg.surface && targetSurface.get() == activeSeg.surface)
            ? defaultColor
            : (_highlightedSurfaceIds.count(targetSurface->id) > 0 ? QColor(Qt::cyan) : defaultColor);

        for (const auto& segment : segments) {
            QPointF a = volumeToScene(segment.world[0]);
            QPointF b = volumeToScene(segment.world[1]);
            if (!isFinitePoint(a) || !isFinitePoint(b)) {
                continue;
            }

            QColor color = baseColor;
            float alpha = _intersectionOpacity;

            if (useApprovalMask && targetSurface.get() == activeSegShared.get()) {
                const cv::Vec3f midParam = (segment.surfaceParams[0] + segment.surfaceParams[1]) * 0.5f;
                const auto [row, col] = surfaceParamToGrid(targetSurface.get(), midParam);
                const QColor approvalColor = segOverlay->queryApprovalColor(row, col);
                if (approvalColor.isValid()) {
                    color = approvalColor;
                    alpha *= std::clamp(segOverlay->approvalMaskOpacity() / 100.0f, 0.0f, 1.0f);
                }
            }

            color.setAlphaF(std::clamp(alpha, 0.0f, 1.0f));
            if (color.alpha() <= 0) {
                continue;
            }

            const QRgb key = color.rgba();
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
        pen.setWidthF(_intersectionThickness);
        pen.setCapStyle(Qt::RoundCap);
        pen.setJoinStyle(Qt::RoundJoin);
        item->setPen(pen);
        item->setBrush(Qt::NoBrush);
        item->setZValue(kIntersectionZ);
        _scene->addItem(item);
        items.push_back(item);
    }

    if (!items.empty()) {
        _ov.intersectItems[kIntersectionItemsKey] = std::move(items);
        fGraphicsView->viewport()->update();
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
    _paths.reserve(paths.size());
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
        QPointF topLeft = _tileScene->surfaceToScene(s.surfRect.left(), s.surfRect.top());
        QPointF botRight = _tileScene->surfaceToScene(s.surfRect.right(), s.surfRect.bottom());
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
