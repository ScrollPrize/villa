#include "CChunkedVolumeViewer.hpp"

#include <atomic>

#include "CState.hpp"
#include "OpenedVolumes.hpp"
#include "CameraGizmoWidget.hpp"
#include "VolumetricCompositor.hpp"
#include "elements/ViewerStatsBar.hpp"
#include "VCSettings.hpp"
#include "ViewerManager.hpp"
#include "vc/core/render/Colormaps.hpp"
#include "vc/core/render/PostProcess.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/Geometry.hpp"
#include "vc/core/util/Logging.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/ui/VCCollection.hpp"

#include <QCoreApplication>
#include <QCursor>
#include <QElapsedTimer>
#include <QEvent>
#include <QFrame>
#include <QShowEvent>
#include <QGraphicsEllipseItem>
#include <QGraphicsItem>
#include <QGraphicsPathItem>
#include <QGraphicsScene>
#include <QGraphicsSimpleTextItem>
#include <QMessageBox>
#include <QPainter>
#include <QPainterPath>
#include <QPointer>
#include <QSettings>
#include <QTimer>
#include <QVBoxLayout>
#include <QtConcurrent/QtConcurrentRun>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <source_location>
#include <unordered_map>


namespace {

constexpr float kMinScale = 0.002f;
constexpr float kMaxScale = 128.0f;
constexpr float kResolutionLodZoomBias = 0.5f;
constexpr float kSegmentationResolutionLodZoomBias = 1.0f;
constexpr int kSurfaceResolutionLevelBias = 1;
constexpr int kInitialSegmentationSurfaceLevel = 5;
constexpr float kPanSmoothingAlpha = 0.65f;

constexpr const char* kMeasurementOverlayKey = "segmentation_measurement";

constexpr float kFocusProjectionThreshold = 4.0f;

bool finiteVec2(const cv::Vec2f& v)
{
    return std::isfinite(v[0]) && std::isfinite(v[1]);
}

bool finiteVec3(const cv::Vec3f& v)
{
    return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]);
}

float distance3(const cv::Vec3f& a, const cv::Vec3f& b)
{
    const cv::Vec3f d = a - b;
    return std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
}

double volumeDistanceMicrometers(const cv::Vec3f& a, const cv::Vec3f& b, double voxelSizeUm)
{
    if (!(voxelSizeUm > 0.0)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const cv::Vec3d d(static_cast<double>(a[0]) - static_cast<double>(b[0]),
                      static_cast<double>(a[1]) - static_cast<double>(b[1]),
                      static_cast<double>(a[2]) - static_cast<double>(b[2]));
    return std::sqrt(d.dot(d)) * voxelSizeUm;
}

struct SafeFocusProjection {
    bool valid = false;
    const char* reason = "unchecked";
    float dist = -1.0f;
    cv::Vec3f ptr{NAN, NAN, NAN};
    cv::Vec3f loc{NAN, NAN, NAN};
    cv::Vec3f coord{NAN, NAN, NAN};
    cv::Vec2f grid{NAN, NAN};
    int row = -1;
    int col = -1;
    int nearestRow = -1;
    int nearestCol = -1;
};

SafeFocusProjection validateSafeFocusProjection(
    QuadSurface& surface,
    const cv::Vec3f& ptr,
    const cv::Vec3f* focusWorld,
    std::optional<float> pointToDistance = std::nullopt)
{
    SafeFocusProjection result;
    result.ptr = ptr;

    if (!finiteVec3(result.ptr)) {
        result.reason = "non_finite_ptr";
        return result;
    }

    result.loc = surface.loc(result.ptr);
    if (!finiteVec3(result.loc)) {
        result.reason = "non_finite_loc";
        return result;
    }

    if (pointToDistance) {
        result.dist = *pointToDistance;
        if (!std::isfinite(result.dist) ||
            result.dist < 0.0f ||
            result.dist >= kFocusProjectionThreshold) {
            result.reason = "distance_out_of_range";
            return result;
        }
    }

    result.coord = surface.coord(result.ptr);
    if (!vc::geometry::validSurfacePoint(result.coord)) {
        result.reason = "invalid_coord";
        return result;
    }

    if (!pointToDistance) {
        result.dist = focusWorld ? distance3(result.coord, *focusWorld) : 0.0f;
        if (!std::isfinite(result.dist) ||
            result.dist < 0.0f ||
            result.dist >= kFocusProjectionThreshold) {
            result.reason = "distance_out_of_range";
            return result;
        }
    }

    result.grid = surface.ptrToGrid(result.ptr);
    if (!finiteVec2(result.grid)) {
        result.reason = "non_finite_grid";
        return result;
    }

    const cv::Mat_<cv::Vec3f>* points = surface.rawPointsPtr();
    if (!points || points->cols < 2 || points->rows < 2) {
        result.reason = "missing_surface_points";
        return result;
    }
    if (result.grid[0] <= 0.0f || result.grid[1] <= 0.0f ||
        result.grid[0] >= static_cast<float>(points->cols - 1) ||
        result.grid[1] >= static_cast<float>(points->rows - 1)) {
        result.reason = "grid_on_or_outside_border";
        return result;
    }

    result.col = static_cast<int>(std::floor(result.grid[0]));
    result.row = static_cast<int>(std::floor(result.grid[1]));
    if (!surface.isQuadValid(result.row, result.col)) {
        result.reason = "invalid_interpolation_quad";
        return result;
    }

    result.nearestCol = static_cast<int>(std::lround(result.grid[0]));
    result.nearestRow = static_cast<int>(std::lround(result.grid[1]));
    if (!surface.isPointValid(result.nearestRow, result.nearestCol)) {
        result.reason = "invalid_nearest_point";
        return result;
    }
    const cv::Vec3f normal = surface.gridNormal(result.nearestRow, result.nearestCol);
    if (!finiteVec3(normal)) {
        result.reason = "non_finite_grid_normal";
        return result;
    }

    result.valid = true;
    result.reason = "valid";
    return result;
}

std::string profileCaller(const std::source_location& caller)
{
    return std::format("{}:{} {}", caller.file_name(), caller.line(), caller.function_name());
}

std::string profileRect(const cv::Rect& r)
{
    return std::format("[x={},y={},w={},h={}]", r.x, r.y, r.width, r.height);
}

std::string profileRectF(const QRectF& r)
{
    return std::format("[x={:.2f},y={:.2f},w={:.2f},h={:.2f}]",
                       r.x(), r.y(), r.width(), r.height());
}

class ProfileScope {
public:
    ProfileScope(const char* event,
                 const char* reason,
                 const std::source_location& caller)
        : event_(event ? event : "")
        , enabled_(ProfileLoggingEnabled())
    {
        if (!enabled_)
            return;
        reason_ = reason ? reason : "";
        caller_ = profileCaller(caller);
        timer_.start();
        Logger()->info("[vc3d-profile] {} begin reason='{}' caller='{}'",
                       event_, reason_, caller_);
    }

    ~ProfileScope()
    {
        finish();
    }

    bool enabled() const
    {
        return enabled_;
    }

    void setDetails(const char* details)
    {
        if (!enabled_)
            return;
        details_ = details ? details : "";
    }

    void setDetails(std::string details)
    {
        if (!enabled_)
            return;
        details_ = std::move(details);
    }

    void finish()
    {
        if (!enabled_ || finished_)
            return;
        finished_ = true;
        Logger()->info("[vc3d-profile] {} end elapsed_ms={} reason='{}' caller='{}'{}{}",
                       event_, timer_.elapsed(), reason_, caller_,
                       details_.empty() ? "" : " ", details_);
    }

private:
    const char* event_ = "";
    std::string reason_;
    std::string caller_;
    bool enabled_ = false;
    bool finished_ = false;
    QElapsedTimer timer_;
    std::string details_;
};

QString formatVec3(const cv::Vec3f& v)
{
    return QString("(%1, %2, %3)")
        .arg(v[0], 0, 'f', 1)
        .arg(v[1], 0, 'f', 1)
        .arg(v[2], 0, 'f', 1);
}

QString formatWholeVolumePosition(const cv::Vec3f& v)
{
    // v is in coordinate/UI order [x, y, z] = [width, height, slices]. Label each
    // component so the scroll Z (slice/depth) axis is unambiguous in the status bar.
    return QString("[X=%1, Y=%2, Z=%3]")
        .arg(v[0], 0, 'f', 0)
        .arg(v[1], 0, 'f', 0)
        .arg(v[2], 0, 'f', 0);
}

QString formatByteSize(std::size_t bytes)
{
    constexpr double kKiB = 1024.0;
    constexpr double kMiB = kKiB * 1024.0;
    constexpr double kGiB = kMiB * 1024.0;
    const double value = static_cast<double>(bytes);
    if (value >= kGiB)
        return QString("%1 GB").arg(value / kGiB, 0, 'f', 2);
    if (value >= kMiB)
        return QString("%1 MB").arg(value / kMiB, 0, 'f', 1);
    if (value >= kKiB)
        return QString("%1 KB").arg(value / kKiB, 0, 'f', 1);
    return QString("%1 B").arg(bytes);
}

// The view scale (pixels per working level-0 voxel) at which `level` of the opened
// levels is the source level: 0.75 pixels per voxel of that level. With no opened levels
// there is only a nominal level 0.
float scaleForRenderLevel(const vc3d::ViewerLevels* levels, int level, float lodZoomBias)
{
    const int count = levels ? levels->count() : 1;
    const int clamped = std::clamp(level, 0, std::max(0, count - 1));
    double extent = levels ? vc3d::maximumBaseVoxelExtent(*levels, clamped) : 1.0;
    if (!std::isfinite(extent) || extent <= 0.0)
        extent = 1.0;
    return std::clamp(0.75f / (static_cast<float>(extent) * lodZoomBias), kMinScale, kMaxScale);
}

float scaleForCoarsestRenderLevel(const vc3d::ViewerLevels* levels, float lodZoomBias)
{
    return scaleForRenderLevel(levels, (levels ? levels->count() : 1) - 1, lodZoomBias);
}

float fitScaleForExtent(float extentU, float extentV, int viewportW, int viewportH)
{
    constexpr float kFitPadding = 0.88f;
    if (!(extentU > 0.0f) || !(extentV > 0.0f) || viewportW <= 0 || viewportH <= 0) {
        return 1.0f;
    }

    const float scaleU = static_cast<float>(viewportW) / extentU;
    const float scaleV = static_cast<float>(viewportH) / extentV;
    return std::clamp(std::min(scaleU, scaleV) * kFitPadding, kMinScale, kMaxScale);
}

} // namespace

CChunkedVolumeViewer::CChunkedVolumeViewer(CState* state, ViewerManager* manager, QWidget* parent)
    : QWidget(parent)
    , _state(state)
    , _viewerManager(manager)
{
    _view = new CVolumeViewerView(this);
    _view->setFrameShape(QFrame::NoFrame);
    _view->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    _view->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    _view->setTransformationAnchor(QGraphicsView::NoAnchor);
    _view->setAlignment(Qt::AlignLeft | Qt::AlignTop);
    _view->setRenderHint(QPainter::Antialiasing, false);
    _view->setScrollPanDisabled(true);
    _view->setViewportUpdateMode(QGraphicsView::FullViewportUpdate);

    connect(_view, &CVolumeViewerView::sendVolumeClicked, this, &CChunkedVolumeViewer::onVolumeClicked);
    connect(_view, &CVolumeViewerView::sendZoom, this, &CChunkedVolumeViewer::onZoom);
    connect(_view, &CVolumeViewerView::sendResized, this, &CChunkedVolumeViewer::onResized);
    connect(_view, &CVolumeViewerView::sendCursorMove, this, &CChunkedVolumeViewer::onCursorMove);
    connect(_view, &CVolumeViewerView::sendPanRelease, this, &CChunkedVolumeViewer::onPanRelease);
    connect(_view, &CVolumeViewerView::sendPanStart, this, &CChunkedVolumeViewer::onPanStart);
    connect(_view, &CVolumeViewerView::sendMousePress, this, &CChunkedVolumeViewer::onMousePress);
    connect(_view, &CVolumeViewerView::sendMouseMove, this, &CChunkedVolumeViewer::onMouseMove);
    connect(_view, &CVolumeViewerView::sendMouseRelease, this, &CChunkedVolumeViewer::onMouseRelease);
    connect(_view, &CVolumeViewerView::sendMouseLeftView, this, [this]() {
        clearLineAnnotationPlacementMarker();
        if (_cursorCrosshair) {
            _cursorCrosshair->hide();
        }
        _lastCursorVolumePos.reset();
        updateStatusLabel();
        // Clear mirrored crosshairs in the other viewers too (nullopt
        // broadcasts are never gated), so they don't freeze at the last point.
        if (_viewerManager) {
            _viewerManager->broadcastLinkedCursor(this, std::nullopt);
        }
    });
    connect(_view, &CVolumeViewerView::sendMouseDoubleClick, this,
            [this](QPointF scenePos, Qt::MouseButton button, Qt::KeyboardModifiers modifiers) {
                const auto cursorPos = cursorVolumePosition(scenePos);
                cv::Vec3f volumePos;
                if (cursorPos) {
                    volumePos = *cursorPos;
                } else {
                    volumePos = sceneToVolume(scenePos);
                }
                emit sendMouseDoubleClickVolume(volumePos, button, modifiers);
            });
    connect(_view, &CVolumeViewerView::sendKeyPress, this, &CChunkedVolumeViewer::onKeyPress);
    connect(_view, &CVolumeViewerView::sendKeyRelease, this, &CChunkedVolumeViewer::onKeyRelease);

    _scene = new QGraphicsScene(this);
    _scene->setItemIndexMethod(QGraphicsScene::NoIndex);
    _view->setScene(_scene);
    _view->setDirectFramebuffer(&_framebuffer);

    // No per-viewer timers: render requests submit immediately. ViewerManager's
    // global clock still drains intersection requests and status refreshes.

    reloadPerfSettings();

    auto* layout = new QVBoxLayout;
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);
    layout->addWidget(_view);
    setLayout(layout);

    _statsBar = new ViewerStatsBar(this);
    _statsBar->move(10, 5);
    _statsBarRight = new ViewerStatsBar(this);
    _statsBarRight->setMinimumWidth(0);
    _statsBarRight->hide();

    _cameraGizmo = new CameraGizmoWidget(_view);
    _cameraGizmo->hide();
    connect(_cameraGizmo, &CameraGizmoWidget::cameraChanged, this,
            [this](float azimuthDeg, float tiltDeg, float perspective) {
                if (_closing)
                    return;
                _compositeSettings.params.camAzimuthDeg = azimuthDeg;
                _compositeSettings.params.camTiltDeg = tiltDeg;
                _compositeSettings.params.camPerspective = perspective;
                updateStatusLabel();
                emit compositeCameraChanged();
                submitRender("volumetric camera");
            });
}

CChunkedVolumeViewer::~CChunkedVolumeViewer()
{
    quiesceForClose();
}

void CChunkedVolumeViewer::showEvent(QShowEvent* event)
{
    QWidget::showEvent(event);
    // Render whatever went stale while this viewer was hidden.
    if (_renderStaleWhileHidden && !_closing) {
        _renderStaleWhileHidden = false;
        submitRender("shown after hidden");
    }
}

void CChunkedVolumeViewer::resizeEvent(QResizeEvent* event)
{
    QWidget::resizeEvent(event);
    repositionStatsBarRight();
}

void CChunkedVolumeViewer::repositionStatsBarRight()
{
    if (!_statsBarRight) {
        return;
    }
    _statsBarRight->move(std::max(0, width() - _statsBarRight->width() - 10), 5);
}

bool CChunkedVolumeViewer::eventFilter(QObject* watched, QEvent* event)
{
    if (event && event->type() == QEvent::Close && watched == parentWidget()) {
        quiesceForClose();
    }
    return QWidget::eventFilter(watched, event);
}

void CChunkedVolumeViewer::quiesceForClose()
{
    if (_closing) {
        return;
    }
    _closing = true;

    if (_viewerManager) {
        _viewerManager->unregisterViewer(this);
    }

    _activeRenderJob.reset();
    _pendingRenderJob.reset();
    _displayedRenderJob.reset();
    ++_renderSerial;

    // A running background read finishes its current part and stops.
    _pendingLoad.reset();
    if (_loadSuperseded)
        _loadSuperseded->store(true);
    _levels.reset();
    _overlayLevels.reset();
}

void CChunkedVolumeViewer::reloadPerfSettings()
{
    QSettings s(vc3d::settingsFilePath(), QSettings::IniFormat);
    using namespace vc3d::settings;
    _panSensitivity = std::max(0.01f, s.value(viewer::PAN_SENSITIVITY, viewer::PAN_SENSITIVITY_DEFAULT).toFloat());
    _zoomSensitivity = std::max(0.01f, s.value(viewer::ZOOM_SENSITIVITY, viewer::ZOOM_SENSITIVITY_DEFAULT).toFloat());
    _zScrollSensitivity = std::max(0.01f, s.value(viewer::ZSCROLL_SENSITIVITY, viewer::ZSCROLL_SENSITIVITY_DEFAULT).toFloat());
    _linkedCursorViewTolerance = std::max(
        0.0f,
        s.value(viewer::POINT_COLLECTION_VIEW_TOLERANCE,
                viewer::POINT_COLLECTION_VIEW_TOLERANCE_DEFAULT).toFloat());
    const int interpIdx = s.value(perf::INTERPOLATION_METHOD, perf::INTERPOLATION_METHOD_DEFAULT).toInt();
    const vc::Sampling sampling = static_cast<vc::Sampling>(std::clamp(interpIdx, 0, 1));
    _samplingMethod = sampling;
    _maxDisplayedResolution = std::clamp(
        s.value(viewer::MAX_DISPLAYED_RESOLUTION, viewer::MAX_DISPLAYED_RESOLUTION_DEFAULT).toInt(),
        0,
        5);
}

void CChunkedVolumeViewer::setSurface(const std::string& name)
{
    if (_closing) {
        return;
    }
    if (_measurement.active || _measurement.first || _measurement.second) {
        clearMeasurement();
    }
    _surfName = name;
    if (_state)
        onSurfaceChanged(name, _state->surface(name));
}

Surface* CChunkedVolumeViewer::currentSurface() const
{
    if (!_state) {
        auto shared = _surfWeak.lock();
        return shared ? shared.get() : nullptr;
    }
    return _state->surfaceRaw(_surfName);
}

CChunkedVolumeViewer::CameraState CChunkedVolumeViewer::cameraState() const
{
    CameraState state;
    state.surfacePtrX = _surfacePtrX;
    state.surfacePtrY = _surfacePtrY;
    state.scale = _scale;
    state.zOffset = _zOff;
    return state;
}

float CChunkedVolumeViewer::clampCameraScale(float scale)
{
    return std::clamp(scale, kMinScale, kMaxScale);
}

void CChunkedVolumeViewer::applyCameraState(const CameraState& state)
{
    if (_closing) {
        return;
    }
    _surfacePtrX = state.surfacePtrX;
    _surfacePtrY = state.surfacePtrY;
    _scale = state.scale;
    setZOffset(state.zOffset);
    recalcPyramidLevel();
    _geometryDirty = true;
    submitRender("annotation camera state applied");
    refreshMeasurementOverlay();
    emit overlaysUpdated();
}

void CChunkedVolumeViewer::setPinnedSurfaceY(std::optional<float> surfaceY)
{
    if (surfaceY && !std::isfinite(*surfaceY)) {
        surfaceY.reset();
    }
    _pinnedSurfacePtrY = surfaceY;
    if (_closing || !_pinnedSurfacePtrY || _surfacePtrY == *_pinnedSurfacePtrY) {
        return;
    }
    _geometryDirty = true;
    // submitRender's syncCameraTransform moves the camera onto the pin and
    // refreshes the measurement overlay.
    submitRender("camera pinned");
    emit overlaysUpdated();
}

void CChunkedVolumeViewer::openLevels()
{
    _levels.reset();
    _gridSurface.reset();  // its normals follow the levels' voxel spacing
    _lastRenderResult.reset();
    if (!_volume || !_state || !_state->openedVolumes())
        return;
    try {
        _levels = std::make_shared<const vc3d::ViewerLevels>(
            vc3d::openViewerLevels(*_state->openedVolumes(), *_volume));
    } catch (const std::exception& e) {
        if (_statsBar)
            _statsBar->setItems({QString("Volume unavailable: %1").arg(e.what())});
        _levels.reset();
    }
}

void CChunkedVolumeViewer::startPendingLoad()
{
    if (_closing || _loadRunning || !_pendingLoad)
        return;
    auto load = std::make_shared<const vc3d::ViewerLoad>(std::move(*_pendingLoad));
    _pendingLoad.reset();
    _loadRunning = true;
    _loadSuperseded = std::make_shared<std::atomic<bool>>(false);
    QPointer<CChunkedVolumeViewer> guard(this);
    (void)QtConcurrent::run([guard, load, superseded = _loadSuperseded]() {
        // Each part's arrival repaints; a newer waiting request stops this one.
        const auto loaded = [guard]() {
            QMetaObject::invokeMethod(qApp, [guard]() {
                if (!guard || guard->_closing)
                    return;
                ++guard->_loadedDataEpoch;
                guard->submitRender("volume data loaded");
            }, Qt::QueuedConnection);
        };
        QString error;
        try {
            vc3d::loadPending(*load, [&]() {
                loaded();
                return !superseded->load();
            });
        } catch (const std::exception& e) {
            error = QString::fromUtf8(e.what());
        }
        QMetaObject::invokeMethod(qApp, [guard, error]() {
            if (!guard)
                return;
            guard->_loadRunning = false;
            guard->_loadError = error;
            if (guard->_closing)
                return;
            if (error.isEmpty()) {
                guard->startPendingLoad();
                return;
            }
            // A failed read is retried by the next frame's request, no sooner than in a second.
            Logger()->warn("viewer: reading volume data failed: {}", error.toStdString());
            guard->updateStatusLabel();
            QTimer::singleShot(1000, guard, [guard]() {
                if (guard)
                    guard->startPendingLoad();
            });
        }, Qt::QueuedConnection);
    });
}

void CChunkedVolumeViewer::OnVolumeChanged(std::shared_ptr<Volume> vol)
{
    if (_closing) {
        return;
    }
    const bool hadVolume = static_cast<bool>(_volume);
    if (_surfWeak.lock() == _defaultSurface) {
        _surfWeak.reset();
        _defaultSurface.reset();
    }
    _geometryDirty = true;
    if (_cursorCrosshair)
        _cursorCrosshair->hide();
    clearLineAnnotationPlacementMarker();
    if (_focusMarker)
        _focusMarker->hide();

    _volume = std::move(vol);
    _pendingLoad.reset();
    openLevels();
    if (!_volume) {
        // No volume means submitRender() below will skip; drop the previous
        // volume's framebuffer so the view goes blank instead of showing it.
        clearDisplayedFramebuffer();
    }
    ensureDefaultSurface();
    if (_volume && isAxisAlignedView() && !hadVolume) {
        _scale = scaleForCoarsestRenderLevel(_levels.get(), kResolutionLodZoomBias);
    }
    recalcPyramidLevel();   // also pushes the scalebar µm/px (updateScalebarScale)
    updateContentBounds();
    resizeFramebuffer();
    submitRender("volume changed");
    updateStatusLabel();
}

void CChunkedVolumeViewer::invalidateVis()
{
    if (_closing) {
        return;
    }
    _geometryDirty = true;
    _gridSurface.reset();
}

void CChunkedVolumeViewer::onSurfaceChanged(const std::string& name,
                                            const std::shared_ptr<Surface>& surf,
                                            bool isEditUpdate)
{
    // Surface geometry is read lazily: the first access below reaches
    // QuadSurface::ensureLoaded(), which opens x/y/z.tif and throws when the
    // payload is missing or unreadable — a partial download, a segment deleted
    // by another process, an unmounted share. This slot is invoked straight
    // from a signal emission, and an exception must not be allowed to reach
    // Qt's dispatch frames: it cannot be caught above them (Qt installs no
    // handler, and on macOS unwinding through them terminates outright), so it
    // would take down the session over one bad segment. Contain it here, while
    // the throw still has only our own frames to unwind through.
    try {
        onSurfaceChangedImpl(name, surf, isEditUpdate);
    } catch (const std::exception& e) {
        Logger()->error("Surface '{}' could not be displayed: {}", name, e.what());
    }
}

void CChunkedVolumeViewer::onSurfaceChangedImpl(const std::string& name, const std::shared_ptr<Surface>& surf, bool isEditUpdate)
{
    if (_closing) {
        return;
    }
    const bool isCurrentSurface = (_surfName == name);
    const auto previousSurface = _surfWeak.lock();
    const bool isSameCurrentSurface = isCurrentSurface && previousSurface && previousSurface == surf;
    if (isCurrentSurface)
        _gridSurface.reset();  // edited or replaced: its grid is rebuilt for the next frame
    if (!isCurrentSurface) {
        return;
    }

    std::optional<cv::Vec3f> preservedViewCenter;
    if (!isEditUpdate && !_resetViewOnSurfaceChange && _surfName == "segmentation" &&
        surf && previousSurface && _view && !_framebuffer.isNull()) {
        const QPointF sceneCenter(static_cast<qreal>(_framebuffer.width()) * 0.5,
                                  static_cast<qreal>(_framebuffer.height()) * 0.5);
        preservedViewCenter = cursorVolumePosition(sceneCenter);
    }
    if (auto* newPlane = dynamic_cast<PlaneSurface*>(surf.get())) {
        // Plane rotated in place (azimuth folding / up realignment): keep the
        // world view center fixed so the view spins about the screen center,
        // not the plane origin. Origin or normal changes (slice scroll, focus
        // move, tilt) keep the existing follow-the-plane behavior. Applied
        // directly because these updates arrive as edit updates, which return
        // early below.
        if (_planeFrame.valid &&
            cv::norm(_planeFrame.origin - newPlane->origin()) < 1e-3 &&
            _planeFrame.normal.dot(newPlane->normal({0, 0, 0})) > 1.0f - 1e-6f &&
            (cv::norm(_planeFrame.vx - newPlane->basisX()) > 1e-6 ||
             cv::norm(_planeFrame.vy - newPlane->basisY()) > 1e-6)) {
            const cv::Vec3f center = _planeFrame.origin +
                                     _planeFrame.vx * _surfacePtrX +
                                     _planeFrame.vy * _surfacePtrY;
            const cv::Vec3f projected = newPlane->project(center, 1.0, 1.0);
            if (std::isfinite(projected[0]) && std::isfinite(projected[1])) {
                _surfacePtrX = projected[0];
                _surfacePtrY = projected[1];
            }
        }
        _planeFrame.valid = true;
        _planeFrame.origin = newPlane->origin();
        _planeFrame.normal = newPlane->normal({0, 0, 0});
        _planeFrame.vx = newPlane->basisX();
        _planeFrame.vy = newPlane->basisY();
    } else {
        _planeFrame.valid = false;
    }

    _surfWeak = surf;
    // The edit paths below can return without submitting a render, so they never
    // reach syncCameraTransform(); refresh the plane's frame here instead.
    updateCoordinateFrame();
    if (isSameCurrentSurface && isEditUpdate) {
        _geometryDirty = true;
        ++_surfaceEditEpoch;
        // A plane mutated in place (axis-aligned slice rotation/tilt, line
        // annotation cut planes) leaves nothing for the render-job identity in
        // renderJobsSameGeometry() to compare: same object, same camera. Without
        // a fresh epoch the submit below is discarded as a duplicate of the
        // displayed frame and the new orientation is never drawn.
        if (dynamic_cast<PlaneSurface*>(surf.get())) {
            ++_surfaceGeometryEpoch;
        }
        updateContentBounds();
        updateFocusMarker();
        if (_suppressNextSurfaceEditRender) {
            _suppressNextSurfaceEditRender = false;
        } else {
            submitRender("current surface edit update");
        }
        return;
    }

    _geometryDirty = true;
    if (!surf) {
        _measurement = {};
        // Forget the overlay groups before the scene deletes their items, so
        // no group ever names a freed item (translateOverlayGroup walks them).
        _overlayGroups.clear();
        _scene->clear();
        _cursorCrosshair = nullptr;
        _lineAnnotationPlacementMarker = nullptr;
        _focusMarker = nullptr;
        return;
    }
    updateContentBounds();
    const bool isSegmentationQuadSurface =
        _surfName == "segmentation" && dynamic_cast<QuadSurface*>(surf.get());

    auto setSegmentationPointerFromFocus = [&]() {
        auto* quad = dynamic_cast<QuadSurface*>(surf.get());
        auto* poi = _state ? _state->poi("focus") : nullptr;
        if (!quad || !poi) {
            return false;
        }
        if (!poi->surfaceId.empty() && !quad->id.empty() &&
            poi->surfaceId != quad->id && poi->surfaceId != _surfName) {
            return false;
        }

        if (poi->surfacePtr) {
            const SafeFocusProjection projected =
                validateSafeFocusProjection(*quad, *poi->surfacePtr, &poi->p);
            if (projected.valid) {
                _surfacePtrX = projected.loc[0];
                _surfacePtrY = projected.loc[1];
                return true;
            }
        }

        cv::Vec3f ptr = quad->pointer();
        const float dist = quad->pointTo(ptr, poi->p, kFocusProjectionThreshold, 100, nullptr);
        const SafeFocusProjection projected =
            validateSafeFocusProjection(*quad, ptr, &poi->p, dist);
        if (!projected.valid) {
            return false;
        }

        _surfacePtrX = projected.loc[0];
        _surfacePtrY = projected.loc[1];
        return true;
    };

    if (!isEditUpdate && isSegmentationQuadSurface && !_initializedFirstSegmentationSurface) {
        if (_resetViewOnSurfaceChange) {
            (void)setSegmentationPointerFromFocus();
        }
        setZOffset(0.0f);
        if (_resetViewOnSurfaceChange) {
            _scale = scaleForCoarsestRenderLevel(_levels.get(), kSegmentationResolutionLodZoomBias);
            recalcPyramidLevel();
        }
        _initializedFirstSegmentationSurface = true;
    } else if (!isEditUpdate && _resetViewOnSurfaceChange && isSegmentationQuadSurface) {
        (void)setSegmentationPointerFromFocus();
        setZOffset(0.0f);
        _scale = scaleForRenderLevel(_levels.get(), kInitialSegmentationSurfaceLevel + kSurfaceResolutionLevelBias,
                                     kResolutionLodZoomBias);
        recalcPyramidLevel();
    } else if (preservedViewCenter) {
        if (auto* plane = dynamic_cast<PlaneSurface*>(surf.get())) {
            const cv::Vec3f projected = plane->project(*preservedViewCenter, 1.0, 1.0);
            if (std::isfinite(projected[0]) && std::isfinite(projected[1])) {
                _surfacePtrX = projected[0];
                _surfacePtrY = projected[1];
            }
        } else if (auto* quad = dynamic_cast<QuadSurface*>(surf.get())) {
            if (const auto hit = _viewerManager ? _viewerManager->locate(*quad, *preservedViewCenter, 4.0f)
                                                : std::nullopt) {
                const cv::Vec3f loc = quad->loc(hit->ptr);
                if (std::isfinite(loc[0]) && std::isfinite(loc[1])) {
                    _surfacePtrX = loc[0];
                    _surfacePtrY = loc[1];
                }
            }
        }
    }

    // Safety net: whatever branch ran above, the (possibly preserved) pan can
    // land entirely off the new segment's extent — e.g. the user had zoomed
    // deep into a large segment and switched to a small one, leaving the
    // viewport in a region the new quad doesn't cover (all black). If the
    // viewport no longer intersects the segment's UV bounds, pan the view
    // center to the nearest point of those bounds. Zoom and the focus POI are
    // deliberately left untouched.
    if (!isEditUpdate && isSegmentationQuadSurface && _view) {
        auto* quad = dynamic_cast<QuadSurface*>(surf.get());
        try {
            quad->ensureLoaded();
            if (const cv::Mat_<cv::Vec3f>* points = quad->rawPointsPtr();
                points && !points->empty()) {
                // Surface-UV (nominal) bounds of the grid: gridToSurface maps
                // grid -> grid/scale - center.
                const cv::Vec3f center = quad->center();
                const cv::Vec2f gridScale = quad->scale();
                const float minU = -center[0];
                const float minV = -center[1];
                const float maxU = static_cast<float>(points->cols - 1) / gridScale[0] - center[0];
                const float maxV = static_cast<float>(points->rows - 1) / gridScale[1] - center[1];
                if (maxU > minU && maxV > minV) {
                    if (!std::isfinite(_surfacePtrX) || !std::isfinite(_surfacePtrY)) {
                        _surfacePtrX = (minU + maxU) * 0.5f;
                        _surfacePtrY = (minV + maxV) * 0.5f;
                    } else {
                        const QSize viewportSize = _view->viewport()->size();
                        const float scale = std::max(_scale, kMinScale);
                        const float halfW = 0.5f * static_cast<float>(viewportSize.width()) / scale;
                        const float halfH = 0.5f * static_cast<float>(viewportSize.height()) / scale;
                        const bool viewportTouchesSegment =
                            _surfacePtrX + halfW > minU && _surfacePtrX - halfW < maxU &&
                            _surfacePtrY + halfH > minV && _surfacePtrY - halfH < maxV;
                        if (!viewportTouchesSegment) {
                            _surfacePtrX = std::clamp(_surfacePtrX, minU, maxU);
                            _surfacePtrY = std::clamp(_surfacePtrY, minV, maxV);
                        }
                    }
                }
            }
        } catch (const std::exception& e) {
            qWarning() << "Could not keep segment visible after surface change:" << e.what();
        }
    }

    updateFocusMarker();
    submitRender("current surface changed");
}

void CChunkedVolumeViewer::onSurfaceWillBeDeleted(const std::string&, const std::shared_ptr<Surface>& surf)
{
    if (_closing) {
        return;
    }
    auto current = _surfWeak.lock();
    if (current && current == surf)
        _surfWeak.reset();
}

void CChunkedVolumeViewer::onVolumeClosing()
{
    if (_closing) {
        return;
    }
    _pendingLoad.reset();
    _levels.reset();
    _volume.reset();
    clearDisplayedFramebuffer();
    onSurfaceChanged(_surfName, nullptr);
}

void CChunkedVolumeViewer::clearDisplayedFramebuffer()
{
    _framebuffer = QImage();
    _displayedRenderJob.reset();
    _lastRenderResult.reset();
    if (_view && _view->viewport()) {
        _view->viewport()->update();
    }
}

void CChunkedVolumeViewer::onPOIChanged(const std::string& name, POI* poi)
{
    if (_closing) {
        return;
    }
    if (name != "focus" || !poi)
        return;
    if (property("vc_viewer_role").toString() == QStringLiteral("annotation")) {
        if (_focusMarker) {
            _focusMarker->hide();
        }
        return;
    }

    auto surf = _surfWeak.lock();
    if (auto* plane = dynamic_cast<PlaneSurface*>(surf.get())) {
        plane->setOrigin(poi->p);
        if (cv::norm(poi->n) > 0.5f)
            plane->setNormal(poi->n);
        updateContentBounds();
        _geometryDirty = true;
    }

    updateFocusMarker(poi);
    updateStatusLabel();
    emit overlaysUpdated();
    submitRender("focus POI changed");
}

void CChunkedVolumeViewer::ensureDefaultSurface()
{
    if (_surfWeak.lock() || !_volume || !isAxisAlignedView())
        return;
    const auto shape = _volume->shapeXyz();
    cv::Vec3f center(static_cast<float>(shape[0]) * 0.5f,
                     static_cast<float>(shape[1]) * 0.5f,
                     static_cast<float>(shape[2]) * 0.5f);
    cv::Vec3f normal;
    if (_surfName == "xy plane") normal = {0, 0, 1};
    else if (_surfName == "xz plane" || _surfName == "seg xz") normal = {0, 1, 0};
    else normal = {1, 0, 0};
    _defaultSurface = std::make_shared<PlaneSurface>(center, normal);
    _surfWeak = _defaultSurface;
}

bool CChunkedVolumeViewer::isAxisAlignedView() const
{
    return _surfName == "xy plane" || _surfName == "xz plane" ||
           _surfName == "yz plane" || _surfName == "seg xz" ||
           _surfName == "seg yz";
}

void CChunkedVolumeViewer::updateContentBounds()
{
    auto surf = _surfWeak.lock();
    if (!_volume || !surf)
        return;
    auto* plane = dynamic_cast<PlaneSurface*>(surf.get());
    if (!plane)
        return;

    const auto [w, h, d] = _volume->shapeXyz();
    const float corners[][3] = {
        {0, 0, 0}, {float(w), 0, 0}, {0, float(h), 0}, {float(w), float(h), 0},
        {0, 0, float(d)}, {float(w), 0, float(d)}, {0, float(h), float(d)}, {float(w), float(h), float(d)}
    };
    _contentMinU = _contentMinV = std::numeric_limits<float>::max();
    _contentMaxU = _contentMaxV = std::numeric_limits<float>::lowest();
    for (const auto& c : corners) {
        const cv::Vec3f proj = plane->project({c[0], c[1], c[2]}, 1.0, 1.0);
        _contentMinU = std::min(_contentMinU, proj[0]);
        _contentMinV = std::min(_contentMinV, proj[1]);
        _contentMaxU = std::max(_contentMaxU, proj[0]);
        _contentMaxV = std::max(_contentMaxV, proj[1]);
    }
}

void CChunkedVolumeViewer::recalcPyramidLevel()
{
    const float lodZoomBias = _surfName == "segmentation"
        ? kSegmentationResolutionLodZoomBias
        : kResolutionLodZoomBias;
    if (_levels) {
        _dsScaleIdx = vc3d::sourceLevelForView(*_levels, _scale, lodZoomBias);
    } else {
        _dsScaleIdx = 0;  // no opened levels: this frame has only a nominal level 0
    }
    updateScalebarScale();
}

void CChunkedVolumeViewer::updateScalebarScale()
{
    // Scene pixels are framebuffer pixels. `_scale` is the camera zoom in
    // framebuffer pixels per level-0 voxel-space unit; render LOD changes which
    // zarr level is sampled, not the physical size of the view.
    if (!_view || !_volume)
        return;
    if (!(_scale > 0.0f))
        return;
    // No positive voxel size in the volume metadata: fall back to voxel units so
    // the bar still tracks zoom (an early return here would freeze the scalebar
    // at the view's default µm/px forever).
    double voxel = _volume->voxelSize();
    const bool physical = voxel > 0.0;
    if (!physical)
        voxel = 1.0;
    const double unitsPerScenePx = voxel / static_cast<double>(_scale);
    _view->setVoxelSize(unitsPerScenePx, unitsPerScenePx, physical);
}

void CChunkedVolumeViewer::setShowCoordinateFrame(bool show)
{
    if (_closing || _showCoordinateFrame == show) {
        return;
    }
    _showCoordinateFrame = show;
    updateCoordinateFrame();
    requestDirectPaint();
}

void CChunkedVolumeViewer::updateCoordinateFrame()
{
    if (!_view) {
        return;
    }
    // Showing a world coordinates reference frame indicator in a quad surface
    // view doesn't make sense. We could show a uv coordinates indicator instead.
    PlaneSurface* plane = dynamic_cast<PlaneSurface*>(currentSurface());
    if (!_showCoordinateFrame || !plane) {
        _view->showCoordinateFrame(false);
        return;
    }

    _view->setCoordinateFrame(plane);
}

void CChunkedVolumeViewer::resizeFramebuffer()
{
    const QSize vpSize = _view->viewport()->size();
    const int w = std::max(1, vpSize.width());
    const int h = std::max(1, vpSize.height());
    if (_framebuffer.isNull() || _framebuffer.width() != w || _framebuffer.height() != h) {
        _framebuffer = QImage(w, h, QImage::Format_RGB32);
        _framebuffer.fill(QColor(64, 64, 64));
    }
    _scene->setSceneRect(0, 0, w, h);
}

void CChunkedVolumeViewer::requestRender(const char* reason, std::source_location caller)
{
    if (reason && std::string_view(reason) == "push/pull active viewer refresh") {
        _suppressNextSurfaceEditRender = true;
    }
    submitRender(reason, caller);
}

// Driven by ViewerManager's single global clock (~60Hz). Rendering is submitted
// immediately by view/data updates; this tick only drains non-render maintenance.
void CChunkedVolumeViewer::refreshStatus()
{
    if (_closing)
        return;
    updateStatusLabel();  // background download progress
}

void CChunkedVolumeViewer::syncCameraTransform()
{
    // Every render submission and repaint passes through here before the
    // camera is captured, so this is the one place the pinned row is
    // enforced: pans keep only their X component, zooms end up anchored on
    // the pinned row at the cursor's X, and resets / centering / applied
    // camera states land back on it.
    if (_pinnedSurfacePtrY) {
        _surfacePtrY = *_pinnedSurfacePtrY;
    }
    _camSurfX = _surfacePtrX;
    _camSurfY = _surfacePtrY;
    _camScale = _scale;
    updateCoordinateFrame();
    updateDisplayedFramebufferMapping();
    refreshMeasurementOverlay();
    updateFocusMarker();
    requestDirectPaint();
}

void CChunkedVolumeViewer::requestDirectPaint()
{
    if (_closing || !_view || !_view->viewport() || !_view->isVisible()) {
        return;
    }
    _view->viewport()->update();
}

int CChunkedVolumeViewer::renderStartLevel(bool preferSurfaceResolution) const
{
    if (!_levels || _levels->count() == 0)
        return 0;
    const float lodZoomBias = _surfName == "segmentation"
        ? kSegmentationResolutionLodZoomBias
        : kResolutionLodZoomBias;
    int level = vc3d::sourceLevelForView(*_levels, _scale, lodZoomBias);
    // The surface renderer resamples volume data straight into the
    // framebuffer and benefits from one finer source level.
    if (preferSurfaceResolution && level < _levels->count() - 1)
        level -= kSurfaceResolutionLevelBias;
    level = std::max(level, _maxDisplayedResolution);
    return std::clamp(level, 0, _levels->count() - 1);
}

int CChunkedVolumeViewer::overlayRenderStartLevel(bool preferSurfaceResolution) const
{
    if (!_overlayLevels || _overlayLevels->count() == 0)
        return 0;
    const float lodZoomBias = _surfName == "segmentation"
        ? kSegmentationResolutionLodZoomBias
        : kResolutionLodZoomBias;
    int level = vc3d::sourceLevelForView(*_overlayLevels, _scale, lodZoomBias);
    if (preferSurfaceResolution && level < _overlayLevels->count() - 1)
        level -= kSurfaceResolutionLevelBias;
    level = std::max(level, _overlayMaxDisplayedResolution);
    return std::clamp(level, 0, _overlayLevels->count() - 1);
}

void CChunkedVolumeViewer::setCompositeRenderSettings(const CompositeRenderSettings& s)
{
    if (_closing)
        return;
    // The slice-plane owner folds this view's azimuth into the plane itself,
    // so it must hear about anything that changes the effective camera —
    // including method/enable flips that implicitly zero it.
    const bool methodChanged =
        _compositeSettings.params.volumetric != s.params.volumetric ||
        _compositeSettings.params.options.method != s.params.options.method;
    const bool cameraChanged =
        _compositeSettings.params.camAzimuthDeg != s.params.camAzimuthDeg ||
        _compositeSettings.params.camTiltDeg != s.params.camTiltDeg ||
        _compositeSettings.params.camPerspective != s.params.camPerspective ||
        methodChanged ||
        _compositeSettings.planeEnabled != s.planeEnabled ||
        _compositeSettings.enabled != s.enabled;
    // Plane viewers draw the flattened viewer's slab bounds as dashed
    // intersection lines; refresh them when the slab changes.
    const bool slabChanged =
        _compositeSettings.enabled != s.enabled ||
        _compositeSettings.layersFront != s.layersFront ||
        _compositeSettings.layersBehind != s.layersBehind ||
        _compositeSettings.reverseDirection != s.reverseDirection ||
        methodChanged;
    _compositeSettings = s;
    updateCameraGizmo();
    if (cameraChanged)
        emit compositeCameraChanged();
    if (slabChanged)
        notifyNormalOffsetChanged();
    submitRender("setCompositeRenderSettings");
}

bool CChunkedVolumeViewer::volumetricCameraActive() const
{
    auto surf = _surfWeak.lock();
    if (!surf || !_compositeSettings.params.volumetric)
        return false;
    // Plane (slice) views gate on the per-plane composite toggle, the
    // flattened view on the main one — matching the render dispatch.
    return dynamic_cast<PlaneSurface*>(surf.get())
               ? _compositeSettings.planeEnabled
               : _compositeSettings.enabled;
}

float CChunkedVolumeViewer::volumetricEffectiveAzimuthDeg() const
{
    // Axis-aligned slice views fold the azimuth spin into the plane's basis
    // (AxisAlignedSliceController), so their screen x axis is +u already.
    if (_volumetricAzimuthInSurface)
        return 0.0f;
    return _compositeSettings.params.camAzimuthDeg;
}

cv::Matx22f CChunkedVolumeViewer::volumetricScreenToSurface() const
{
    if (!volumetricCameraActive())
        return cv::Matx22f::eye();
    constexpr float kDegToRad = 3.14159265358979323846f / 180.0f;
    const float az = volumetricEffectiveAzimuthDeg() * kDegToRad;
    const float tilt = std::clamp(_compositeSettings.params.camTiltDeg, 0.0f, 85.0f) * kDegToRad;
    const float ca = std::cos(az), sa = std::sin(az);
    const float invCt = 1.0f / std::cos(tilt);
    // Rz(-az) * diag(1, 1/cos(tilt)) — the w=0 row of the render's
    // screen->slab projection (see VolumetricCompositor::slabProjection).
    return {ca, sa * invCt,
            -sa, ca * invCt};
}

vc3d::volumetric::CameraParams CChunkedVolumeViewer::volumetricPointMapCamera() const
{
    vc3d::volumetric::CameraParams cam;
    cam.azimuthDeg = volumetricEffectiveAzimuthDeg();
    cam.tiltDeg = _compositeSettings.params.camTiltDeg;
    cam.perspective = _compositeSettings.params.camPerspective;
    return cam;
}

float CChunkedVolumeViewer::volumetricHalfSpan() const
{
    // Camera distance matching VolumetricCompositor::perspectiveCamera:
    // magnification 1 at the view center.
    return 0.5f * std::max(float(std::max(_framebuffer.width(),
                                          _framebuffer.height())), 1.0f);
}

cv::Vec2f CChunkedVolumeViewer::volumetricScreenPxToSurfacePx(const cv::Vec2f& s,
                                                              float wPx) const
{
    if (!volumetricCameraActive())
        return s;
    const auto q = vc3d::volumetric::screenToSlabPoint(
        volumetricPointMapCamera(), volumetricHalfSpan(), {s[0], s[1]}, wPx);
    return {q[0], q[1]};
}

cv::Vec2f CChunkedVolumeViewer::volumetricSurfacePxToScreenPx(const cv::Vec2f& q,
                                                              float wPx) const
{
    if (!volumetricCameraActive())
        return q;
    const auto s = vc3d::volumetric::slabPointToScreen(
        volumetricPointMapCamera(), volumetricHalfSpan(), {q[0], q[1]}, wPx);
    return {s[0], s[1]};
}

void CChunkedVolumeViewer::updateCameraGizmo()
{
    if (!_cameraGizmo)
        return;
    _cameraGizmo->setCamera(_compositeSettings.params.camAzimuthDeg,
                            _compositeSettings.params.camTiltDeg,
                            _compositeSettings.params.camPerspective);
    _cameraGizmo->setVisible(volumetricCameraActive());
}

struct CChunkedVolumeViewer::RenderContext {
    PendingRenderJob renderJob;
    std::uint64_t serial = 0;
    int fbW = 0;
    int fbH = 0;
    float surfacePtrX = 0.0f;
    float surfacePtrY = 0.0f;
    float scale = 1.0f;
    float zOff = 0.0f;
    int startLevel = 0;
    int overlayStartLevel = 0;
    vc::Sampling samplingMethod = vc::Sampling::Trilinear;
    vc::Sampling overlaySamplingMethod = vc::Sampling::Nearest;
    CompositeRenderSettings compositeSettings;
    float windowLow = 0.0f;
    float windowHigh = 255.0f;
    std::shared_ptr<Surface> surf;
    std::shared_ptr<const vc3d::ViewerLevels> levels;
    std::shared_ptr<Volume> overlayVolume;
    std::shared_ptr<const vc3d::ViewerLevels> overlayLevels;
    float overlayOpacity = 0.0f;
    std::string overlayColormapId;
    float overlayWindowLow = 0.0f;
    float overlayWindowHigh = 255.0f;
    bool overlayIntensityAsAlpha = false;
    OverlayCompositeSettings overlayComposite;
    std::shared_ptr<const vcr::grid::Surface> gridSurface;
    bool geometryDirty = false;
    std::shared_ptr<const RenderResult> prevResult;
    std::string profileReason;
    std::string profileCaller;
};

struct CChunkedVolumeViewer::RenderResult {
    std::uint64_t serial = 0;
    PendingRenderJob renderJob;
    QImage framebuffer;
    // Pre-LUT sample values and their coverage, kept so the next
    // same-geometry render can fill holes from this frame.
    cv::Mat_<uint8_t> values;
    cv::Mat_<uint8_t> coverage;
    // Volumetric composite output (already colormapped, RGB channel order).
    // Non-empty only for the "volumetric" method; carried forward like values.
    cv::Mat_<cv::Vec3b> colorValues;
    cv::Mat_<uint8_t> overlayValues;
    cv::Mat_<uint8_t> overlayCoverage;
    // What this frame could not sample from the caches.
    vc3d::ViewerLoad load;
    double renderFrameElapsedMs = 0.0;
    std::chrono::steady_clock::time_point submittedAt;
    std::chrono::steady_clock::time_point workerStartedAt;
    std::chrono::steady_clock::time_point workerFinishedAt;
};

CChunkedVolumeViewer::RenderResult CChunkedVolumeViewer::renderFrame(RenderContext ctx)
{
    QElapsedTimer renderTimer;
    renderTimer.start();
    // Sub-phase timing (only meaningful under --profile). sample = volume
    // sampling along the view, blit = colormap LUT + framebuffer write.
    const bool profilePhases = ProfileLoggingEnabled();
    qint64 phaseSampleMs = 0, phaseBlitMs = 0;
    QElapsedTimer phaseTimer;
    const bool overlayActive = ctx.overlayLevels && ctx.overlayVolume && ctx.overlayOpacity > 0.0f;
    if (ProfileLoggingEnabled()) {
        Logger()->info("[vc3d-profile] renderFrame begin reason='{}' caller='{}' serial={} surf='{}' size={}x{} level={} overlay_level={} overlay={} composite={} planeComposite={}",
                       ctx.profileReason, ctx.profileCaller, ctx.serial,
                       ctx.surf ? ctx.surf->id : std::string(""),
                       ctx.fbW, ctx.fbH, ctx.startLevel, ctx.overlayStartLevel,
                       overlayActive,
                       ctx.compositeSettings.enabled, ctx.compositeSettings.planeEnabled);
    }
    RenderResult result;
    result.serial = ctx.serial;
    result.renderJob = ctx.renderJob;
    result.submittedAt = ctx.renderJob.submittedAt;
    result.workerStartedAt = std::chrono::steady_clock::now();
    result.framebuffer = QImage(std::max(1, ctx.fbW), std::max(1, ctx.fbH), QImage::Format_RGB32);
    result.framebuffer.fill(QColor(64, 64, 64));
    auto finishRenderFrameProfile = [&]() {
        result.renderFrameElapsedMs = static_cast<double>(renderTimer.nsecsElapsed()) / 1000000.0;
        result.workerFinishedAt = std::chrono::steady_clock::now();
        if (!ProfileLoggingEnabled())
            return;
        Logger()->info("[vc3d-profile] renderFrame end elapsed_ms={} sample_ms={} blit_ms={} load_coarse={} load_target={} reason='{}' caller='{}' serial={} framebuffer={}x{}",
                       result.renderFrameElapsedMs, phaseSampleMs, phaseBlitMs,
                       result.load.coarse.size(), result.load.target.size(),
                       ctx.profileReason, ctx.profileCaller,
                       result.serial, result.framebuffer.width(), result.framebuffer.height());
    };

    if (!ctx.surf || !ctx.levels || ctx.fbW <= 0 || ctx.fbH <= 0) {
        finishRenderFrameProfile();
        return result;
    }
    cv::Mat_<uint8_t> values(ctx.fbH, ctx.fbW, uint8_t(0));
    cv::Mat_<uint8_t> coverage(ctx.fbH, ctx.fbW, uint8_t(0));
    cv::Mat_<cv::Vec3b> colorValues;
    cv::Mat_<uint8_t> overlayValues;
    cv::Mat_<uint8_t> overlayCoverage;
    cv::Mat_<uint8_t> focusBoundsInside;
    if (ctx.renderJob.focusBoundsBase) {
        focusBoundsInside = cv::Mat_<uint8_t>(ctx.fbH, ctx.fbW, uint8_t(0));
    }
    auto* plane = dynamic_cast<PlaneSurface*>(ctx.surf.get());
    const bool planeView = plane != nullptr;
    if (!planeView && !ctx.gridSurface) {
        finishRenderFrameProfile();
        return result;
    }

    const bool volumetricMethod = ctx.compositeSettings.params.volumetric;

    // Volumetric mode: composite a sampled layer stack front-to-back along
    // tilted view rays with a transfer function. Output is an already-
    // colormapped RGB buffer; the blit skips the scalar LUT for it. Shared by
    // the flattened-surface and plane (slice) paths — layer i sits at
    // w = zStart + i in slab space either way.
    auto volumetricCamera = [&]() {
        vc3d::volumetric::CameraParams cam;
        cam.azimuthDeg = ctx.compositeSettings.params.camAzimuthDeg;
        cam.tiltDeg = ctx.compositeSettings.params.camTiltDeg;
        cam.perspective = ctx.compositeSettings.params.camPerspective;
        return cam;
    };

    auto flattenedVolumetricMargins = [&](int outW, int outH) {
        const auto& cs = ctx.compositeSettings;
        const int front = std::max(0, cs.layersFront);
        const int behind = std::max(0, cs.layersBehind);
        const float wScale = std::max(cs.params.wScale, 0.01f);
        return vc3d::volumetric::computeSlabMargins(
            volumetricCamera(), front + behind + 1, -behind,
            ctx.scale * wScale, outW, outH);
    };

    auto compositeVolumetricStack = [&](const std::vector<cv::Mat_<uint8_t>>& layerValues,
                                        const std::vector<cv::Mat_<uint8_t>>& layerCoverage,
                                        int zStart,
                                        bool isFlattenedView,
                                        const vc3d::volumetric::SlabMargins& margins,
                                        cv::Mat_<cv::Vec3b>& colorDst,
                                        cv::Mat_<uint8_t>& cov) {
        const auto& cs = ctx.compositeSettings;
        const auto cam = volumetricCamera();

        std::array<uint32_t, 256> colorLut{};
        vc::buildWindowLevelLut(colorLut, ctx.windowLow, ctx.windowHigh);
        // Ray-segment length correction 1/|d_w| so tilted views don't look
        // artificially transparent (per-frame constant: parallel rays).
        const auto dir = vc3d::volumetric::viewDirection(cam);
        const float segLen = 1.0f / std::max(dir[2], 1e-3f);
        const auto opacityLut = vc3d::volumetric::buildOpacityLut(
            cs.params.options.alphaMin, cs.params.options.alphaMax, cs.params.options.alphaOpacity,
            cs.params.tfGamma, uint8_t(std::clamp(cs.params.options.isoCutoff, 0.0f, 255.0f)), segLen);

        // W scaling exaggerates the relief: the layer planes spread apart
        // along the normal before the rotated render (the in-plane content
        // and the sampling itself are untouched). Flattened view only —
        // a slice slab has no surface relief worth amplifying.
        const float wScale = isFlattenedView
            ? std::max(cs.params.wScale, 0.01f)
            : 1.0f;
        const float lightingStrength = isFlattenedView
            ? cs.params.lightingStrength
            : 0.0f;
        vc3d::volumetric::compositeVolumetric(layerValues, layerCoverage, cam,
                                              zStart, ctx.scale * wScale,
                                              colorLut, opacityLut,
                                              colorDst, cov, margins,
                                              lightingStrength);
    };

    // The frame's geometry, grown by `margins` pixels on each side: the plane
    // through the view, or the raster of the view over the surface's grid.
    auto frameGeometry = [&](const vc3d::volumetric::SlabMargins& margins) {
        vc3d::ViewerGeometry geometry;
        const int width = ctx.fbW + margins.left + margins.right;
        const int height = ctx.fbH + margins.top + margins.bottom;
        if (planeView) {
            const cv::Vec3f vx = plane->basisX();
            const cv::Vec3f vy = plane->basisY();
            const cv::Vec3f n = plane->normal({0, 0, 0});
            const float halfW = static_cast<float>(ctx.fbW) * 0.5f / ctx.scale;
            const float halfH = static_cast<float>(ctx.fbH) * 0.5f / ctx.scale;
            const cv::Vec3f vxStep = vx / ctx.scale;
            const cv::Vec3f vyStep = vy / ctx.scale;
            const cv::Vec3f origin = vx * (ctx.surfacePtrX - halfW) + vy * (ctx.surfacePtrY - halfH) +
                                     plane->origin() + n * ctx.zOff - vxStep * float(margins.left) -
                                     vyStep * float(margins.top);
            geometry.plane = vcr::grid::Plane{{origin[0], origin[1], origin[2]},
                                              {vxStep[0], vxStep[1], vxStep[2]},
                                              {vyStep[0], vyStep[1], vyStep[2]},
                                              {n[0], n[1], n[2]}, width, height};
            return geometry;
        }
        // QuadSurface::gen's view: surface UV (ptr - half the viewport) at `scale`
        // framebuffer pixels per surface unit, in grid units.
        auto* quad = static_cast<QuadSurface*>(ctx.surf.get());
        const double u0 = double(ctx.surfacePtrX) - (double(ctx.fbW) * 0.5 + margins.left) / double(ctx.scale);
        const double v0 = double(ctx.surfacePtrY) - (double(ctx.fbH) * 0.5 + margins.top) / double(ctx.scale);
        geometry.surface = ctx.gridSurface;
        geometry.raster = quad->gridRaster(u0, v0, double(ctx.scale), {width, height});
        return geometry;
    };
    const auto geometry = frameGeometry({});

    // Layer offsets along the normal: the plane's origin already carries zOff.
    const float zBase = planeView ? 0.0f : ctx.zOff;
    auto layerOffsets = [&](int front, int behind, float zStep) {
        std::vector<float> offsets;
        for (int i = -std::max(0, behind); i <= std::max(0, front); ++i)
            offsets.push_back(zBase + float(i) * zStep);
        return offsets;
    };
    // Coarser levels fill holes until one source chunk spans the viewport.
    vc3d::ViewerSample base{ctx.levels, ctx.startLevel,
                            vc3d::fallbackLevelCountForViewport(*ctx.levels, ctx.startLevel, ctx.fbW, ctx.fbH, ctx.scale),
                            ctx.samplingMethod, {zBase}, std::nullopt};
    const auto& cs = ctx.compositeSettings;
    const bool baseComposite = planeView ? cs.planeEnabled : cs.enabled;
    const int front = planeView ? cs.planeLayersFront : cs.layersFront;
    const int behind = planeView ? cs.planeLayersBehind : cs.layersBehind;
    const float zStep = (planeView ? cs.planeReverseDirection : cs.reverseDirection) ? -1.0f : 1.0f;

    if (profilePhases) phaseTimer.restart();
    // Pixels this frame could not sample from the caches are recorded in
    // result.load; the viewer reads them in the background and renders again.
    if (baseComposite && volumetricMethod) {
        // Tilted/perspective rays read the slab outside the screen window near
        // the edges: sample the layer stack over the exact extra border the
        // camera needs, then composite it to the screen.
        const int numLayers = std::max(0, front) + std::max(0, behind) + 1;
        const int zStart = -std::max(0, behind);
        const auto margins = planeView
            ? vc3d::volumetric::computeSlabMargins(volumetricCamera(), numLayers, zStart, ctx.scale, ctx.fbW, ctx.fbH)
            : flattenedVolumetricMargins(ctx.fbW, ctx.fbH);
        const auto stack = frameGeometry(margins);
        base.sampling = vc::Sampling::Nearest;
        base.offsets = layerOffsets(front, behind, zStep);
        const auto sampled = vc3d::sampleCached(stack, base, &result.load);
        std::vector<cv::Mat_<uint8_t>> layerValues, layerCoverage;
        for (int k = 0; k < numLayers; ++k) {
            layerValues.emplace_back(stack.height(), stack.width(), uint8_t(0));
            layerCoverage.emplace_back(stack.height(), stack.width(), uint8_t(0));
            if (!sampled.values.empty())
                vc3d::toDisplay(sampled, k, layerValues.back().ptr(), layerCoverage.back().ptr());
        }
        compositeVolumetricStack(layerValues, layerCoverage, zStart, !planeView, margins, colorValues, coverage);
    } else {
        if (baseComposite) {
            base.sampling = vc::Sampling::Nearest;
            base.offsets = layerOffsets(front, behind, zStep);
            base.composite = vc3d::compositeOptions(cs.params);
        }
        const auto sampled = vc3d::sampleCached(geometry, base, &result.load);
        if (!sampled.values.empty())
            vc3d::toDisplay(sampled, 0, values.ptr(), coverage.ptr());
    }
    if (overlayActive) {
        overlayValues = cv::Mat_<uint8_t>(ctx.fbH, ctx.fbW, uint8_t(0));
        overlayCoverage = cv::Mat_<uint8_t>(ctx.fbH, ctx.fbW, uint8_t(0));
        vc3d::ViewerSample overlay{ctx.overlayLevels, ctx.overlayStartLevel,
                                   vc3d::fallbackLevelCountForViewport(*ctx.overlayLevels, ctx.overlayStartLevel,
                                                                       ctx.fbW, ctx.fbH, ctx.scale),
                                   ctx.overlaySamplingMethod, {zBase}, std::nullopt};
        const auto& oc = ctx.overlayComposite;
        if (!planeView && oc.enabled && (oc.method == "max" || oc.method == "mean" || oc.method == "min") &&
            (oc.layersFront > 0 || oc.layersBehind > 0)) {
            // Keep "front" pointing the same physical direction as the primary composite.
            CompositeParams overlayParams;
            setCompositeMethod(overlayParams, oc.method);
            overlay.offsets = layerOffsets(oc.layersFront, oc.layersBehind, cs.reverseDirection ? -1.0f : 1.0f);
            overlay.composite = vc3d::compositeOptions(overlayParams);
        }
        const auto sampled = vc3d::sampleCached(geometry, overlay, &result.load);
        if (!sampled.values.empty())
            vc3d::toDisplay(sampled, 0, overlayValues.ptr(), overlayCoverage.ptr());
    }
    if (profilePhases) phaseSampleMs += phaseTimer.elapsed();

    if (!focusBoundsInside.empty()) {
        std::vector<float> xyz(std::size_t(ctx.fbW) * std::size_t(ctx.fbH) * 3);
        if (geometry.plane) {
            const auto& pl = *geometry.plane;
            for (int y = 0; y < ctx.fbH; ++y)
                for (int x = 0; x < ctx.fbW; ++x)
                    for (int a = 0; a < 3; ++a)
                        xyz[(std::size_t(y) * ctx.fbW + x) * 3 + a] =
                            pl.origin[a] + pl.dy[a] * float(y) + pl.dx[a] * float(x);
        } else {
            geometry.surface->resample(geometry.raster, xyz.data(), nullptr);
        }
        for (int y = 0; y < ctx.fbH; ++y) {
            auto* inside = focusBoundsInside.ptr<uint8_t>(y);
            for (int x = 0; x < ctx.fbW; ++x) {
                const float* p = &xyz[(std::size_t(y) * ctx.fbW + x) * 3];
                inside[x] = contains_point(*ctx.renderJob.focusBoundsBase, cv::Vec3f(p[0], p[1], p[2]));
            }
        }
    }

    // Carry forward the previous frame's pixels where this frame has no
    // data yet (still loading). Re-renders arrive as volume data loads, and
    // presenting a from-scratch frame that lacks a chunk flashes the region
    // blank until it lands. Sampling above always runs first, so refined data
    // still replaces carried pixels as soon as it exists.
    if (const RenderResult* prev = ctx.prevResult.get();
        prev && !ctx.geometryDirty &&
        renderJobsSameGeometry(ctx.renderJob, prev->renderJob)) {
        auto fillFromPrev = [](cv::Mat_<uint8_t>& dst, cv::Mat_<uint8_t>& cov,
                               const cv::Mat_<uint8_t>& prevDst,
                               const cv::Mat_<uint8_t>& prevCov) {
            if (dst.empty() || prevDst.empty() ||
                dst.size() != prevDst.size() || cov.size() != prevCov.size())
                return;
            for (int y = 0; y < dst.rows; ++y) {
                uint8_t* dstRow = dst.ptr<uint8_t>(y);
                uint8_t* covRow = cov.ptr<uint8_t>(y);
                const uint8_t* prevDstRow = prevDst.ptr<uint8_t>(y);
                const uint8_t* prevCovRow = prevCov.ptr<uint8_t>(y);
                for (int x = 0; x < dst.cols; ++x) {
                    if (!covRow[x] && prevCovRow[x]) {
                        dstRow[x] = prevDstRow[x];
                        covRow[x] = 1;
                    }
                }
            }
        };
        auto fillFromPrevColor = [](cv::Mat_<cv::Vec3b>& dst, cv::Mat_<uint8_t>& cov,
                                    const cv::Mat_<cv::Vec3b>& prevDst,
                                    const cv::Mat_<uint8_t>& prevCov) {
            if (dst.empty() || prevDst.empty() ||
                dst.size() != prevDst.size() || cov.size() != prevCov.size())
                return;
            for (int y = 0; y < dst.rows; ++y) {
                auto* dstRow = dst.ptr<cv::Vec3b>(y);
                uint8_t* covRow = cov.ptr<uint8_t>(y);
                const auto* prevDstRow = prevDst.ptr<cv::Vec3b>(y);
                const uint8_t* prevCovRow = prevCov.ptr<uint8_t>(y);
                for (int x = 0; x < dst.cols; ++x) {
                    if (!covRow[x] && prevCovRow[x]) {
                        dstRow[x] = prevDstRow[x];
                        covRow[x] = 1;
                    }
                }
            }
        };
        // Same geometry implies the same composite method (compositeSettings
        // compares ==), so exactly one of these applies per frame.
        if (!colorValues.empty())
            fillFromPrevColor(colorValues, coverage, prev->colorValues, prev->coverage);
        else
            fillFromPrev(values, coverage, prev->values, prev->coverage);
        fillFromPrev(overlayValues, overlayCoverage,
                     prev->overlayValues, prev->overlayCoverage);
    }
    result.values = values;
    result.coverage = coverage;
    result.colorValues = colorValues;
    result.overlayValues = overlayValues;
    result.overlayCoverage = overlayCoverage;
    if (profilePhases) phaseTimer.restart();
    std::array<uint32_t, 256> lut{};
    vc::buildWindowLevelLut(lut, ctx.windowLow, ctx.windowHigh);
    std::array<uint32_t, 256> overlayLut{};
    const bool hasOverlay = !overlayValues.empty() && !overlayCoverage.empty() &&
                            ctx.overlayOpacity > 0.0f;
    if (hasOverlay) {
        vc::buildOverlayLut(overlayLut, ctx.overlayWindowLow, ctx.overlayWindowHigh,
                            ctx.overlayColormapId, ctx.overlayIntensityAsAlpha);
    }
    auto* fbBits = reinterpret_cast<uint32_t*>(result.framebuffer.bits());
    const int fbStride = result.framebuffer.bytesPerLine() / 4;
    const uint32_t uncoveredPixel = planeView ? 0xFF404040u : 0xFF000000u;
    // The volumetric composite is already colormapped: blit its RGB buffer
    // straight into the framebuffer, skipping the scalar LUT stage. Overlay
    // blending on top is unchanged.
    const bool hasColorValues = !colorValues.empty();
    for (int y = 0; y < ctx.fbH; ++y) {
        auto* row = fbBits + size_t(y) * size_t(fbStride);
        const auto* src = values.ptr<uint8_t>(y);
        const auto* cov = coverage.ptr<uint8_t>(y);
        const auto* colorSrc = hasColorValues ? colorValues.ptr<cv::Vec3b>(y) : nullptr;
        const auto* overlaySrc = hasOverlay ? overlayValues.ptr<uint8_t>(y) : nullptr;
        const auto* overlayCov = hasOverlay ? overlayCoverage.ptr<uint8_t>(y) : nullptr;
        const auto* focusInside = !focusBoundsInside.empty()
            ? focusBoundsInside.ptr<uint8_t>(y)
            : nullptr;
        for (int x = 0; x < ctx.fbW; ++x) {
            uint32_t pixel = !cov[x] ? uncoveredPixel
                : hasColorValues
                    ? (0xFF000000u | (uint32_t(colorSrc[x][0]) << 16) |
                       (uint32_t(colorSrc[x][1]) << 8) | uint32_t(colorSrc[x][2]))
                    : lut[src[x]];
            if (hasOverlay && overlayCov[x]) {
                pixel = vc::blendOverlayArgb(pixel, overlayLut[overlaySrc[x]], ctx.overlayOpacity);
            }
            if (focusInside && !focusInside[x]) {
                pixel = (pixel & 0xFF000000u) |
                        ((pixel & 0x00FEFEFEu) >> 1);
            }
            row[x] = pixel;
        }
    }
    if (profilePhases) phaseBlitMs = phaseTimer.elapsed();
    finishRenderFrameProfile();
    return result;
}

std::optional<CChunkedVolumeViewer::PendingRenderJob> CChunkedVolumeViewer::captureRenderJob(
    const char* reason,
    std::source_location caller,
    const std::shared_ptr<Surface>& surf,
    int fbW,
    int fbH,
    std::chrono::steady_clock::time_point submittedAt)
{
    if (!surf || !_volume || !_levels || fbW <= 0 || fbH <= 0) {
        return std::nullopt;
    }

    PendingRenderJob job;
    job.requestId = ++_renderRequestSerial;
    job.fbW = fbW;
    job.fbH = fbH;
    job.surfacePtrX = _surfacePtrX;
    job.surfacePtrY = _surfacePtrY;
    job.scale = _scale;
    job.zOff = _zOff;
    const bool preferSurfaceResolution = dynamic_cast<PlaneSurface*>(surf.get()) == nullptr;
    job.startLevel = renderStartLevel(preferSurfaceResolution);
    job.overlayStartLevel = overlayRenderStartLevel(preferSurfaceResolution);
    job.samplingMethod = _samplingMethod;
    job.overlaySamplingMethod = _overlaySamplingMethod;
    job.compositeSettings = _compositeSettings;
    // Slice views carry the azimuth in the plane basis itself; the volumetric
    // compositor must not apply it a second time.
    if (_volumetricAzimuthInSurface)
        job.compositeSettings.params.camAzimuthDeg = 0.0f;
    job.windowLow = _windowLow;
    job.windowHigh = _windowHigh;
    job.surf = surf;
    job.surfaceName = _surfName;
    job.levels = _levels;
    job.overlayVolume = _overlayVolume;
    job.overlayLevels = _overlayLevels;
    job.overlayOpacity = _overlayOpacity;
    job.overlayColormapId = _overlayColormapId;
    job.overlayWindowLow = _overlayWindowLow;
    job.overlayWindowHigh = _overlayWindowHigh;
    job.overlayIntensityAsAlpha = _overlayIntensityAsAlpha;
    job.overlayComposite = _overlayComposite;
    job.loadedDataEpoch = _loadedDataEpoch;
    job.surfaceGeometryEpoch = _surfaceGeometryEpoch;
    // Focus bounds are in volume base coordinates for every view, including
    // segment QuadSurfaces. Mask the generated coordinates, not UVs;
    // the underlying surface and its cached tiles remain unchanged.
    if (_state) {
        job.focusBoundsBase = _state->activeFocusBounds();
    }
    // The surface's vertex grid, rebuilt after every edit of the surface.
    if (auto* quad = dynamic_cast<QuadSurface*>(surf.get())) {
        if (!_gridSurface || _gridSurfaceFor != quad)
            _gridSurface = vc3d::gridSurface(*quad, _levels ? _levels->vertexSpacing() : std::array<float, 3>{1, 1, 1});
        _gridSurfaceFor = quad;
    } else {
        _gridSurface.reset();
        _gridSurfaceFor = nullptr;
    }
    job.gridSurface = _gridSurface;
    job.geometryDirty = _geometryDirty;
    job.profileReason = reason ? reason : "";
    if (ProfileLoggingEnabled()) {
        job.profileCaller = profileCaller(caller);
    }
    job.submittedAt = submittedAt;

    return job;
}

bool CChunkedVolumeViewer::renderJobsEquivalentForDisplay(const PendingRenderJob& a,
                                                          const PendingRenderJob& b)
{
    const auto boundsEqual = [](const std::optional<Rect3D>& lhs,
                                const std::optional<Rect3D>& rhs) {
        if (lhs.has_value() != rhs.has_value()) {
            return false;
        }
        return !lhs || (lhs->low == rhs->low && lhs->high == rhs->high);
    };
    return renderJobsSameGeometry(a, b) &&
           a.loadedDataEpoch == b.loadedDataEpoch &&
           a.geometryDirty == b.geometryDirty &&
           boundsEqual(a.focusBoundsBase, b.focusBoundsBase);
}

bool CChunkedVolumeViewer::renderJobsSameGeometry(const PendingRenderJob& a,
                                                  const PendingRenderJob& b)
{
    auto vecEqual = [](const cv::Vec3f& lhs, const cv::Vec3f& rhs) {
        return lhs[0] == rhs[0] && lhs[1] == rhs[1] && lhs[2] == rhs[2];
    };
    return a.fbW == b.fbW &&
           a.fbH == b.fbH &&
           a.surfacePtrX == b.surfacePtrX &&
           a.surfacePtrY == b.surfacePtrY &&
           a.scale == b.scale &&
           a.zOff == b.zOff &&
           a.startLevel == b.startLevel &&
           a.overlayStartLevel == b.overlayStartLevel &&
           a.samplingMethod == b.samplingMethod &&
           a.overlaySamplingMethod == b.overlaySamplingMethod &&
           a.compositeSettings == b.compositeSettings &&
           a.windowLow == b.windowLow &&
           a.windowHigh == b.windowHigh &&
           a.surf.get() == b.surf.get() &&
           a.surfaceName == b.surfaceName &&
           a.levels.get() == b.levels.get() &&
           a.overlayVolume.get() == b.overlayVolume.get() &&
           a.overlayLevels.get() == b.overlayLevels.get() &&
           a.overlayOpacity == b.overlayOpacity &&
           a.overlayColormapId == b.overlayColormapId &&
           a.overlayWindowLow == b.overlayWindowLow &&
           a.overlayWindowHigh == b.overlayWindowHigh &&
           a.overlayIntensityAsAlpha == b.overlayIntensityAsAlpha &&
           a.overlayComposite == b.overlayComposite &&
           a.surfaceGeometryEpoch == b.surfaceGeometryEpoch;
}

void CChunkedVolumeViewer::updateDisplayedFramebufferMapping()
{
    if (!_view || !_displayedRenderJob || _displayedRenderJob->scale <= 0.0f) {
        if (_view) {
            _view->setDirectFramebufferMapping(1.0, QPointF(0.0, 0.0));
        }
        return;
    }

    const auto& job = *_displayedRenderJob;
    const qreal ratio = static_cast<qreal>(_scale) / static_cast<qreal>(job.scale);
    const qreal currentHalfW = static_cast<qreal>(_framebuffer.width()) * 0.5;
    const qreal currentHalfH = static_cast<qreal>(_framebuffer.height()) * 0.5;
    const qreal renderedHalfW = static_cast<qreal>(job.fbW) * 0.5;
    const qreal renderedHalfH = static_cast<qreal>(job.fbH) * 0.5;
    // The view-center delta lives in surface UV; the stale framebuffer slides
    // in screen space, so convert it back through the inverse camera mapping
    // (identity outside the volumetric mode).
    const cv::Vec2f surfDelta(job.surfacePtrX - _surfacePtrX,
                              job.surfacePtrY - _surfacePtrY);
    const cv::Vec2f screenDelta = volumetricScreenToSurface().inv() * surfDelta;
    const QPointF offset(
        currentHalfW + static_cast<qreal>(screenDelta[0]) * static_cast<qreal>(_scale) -
            renderedHalfW * ratio,
        currentHalfH + static_cast<qreal>(screenDelta[1]) * static_cast<qreal>(_scale) -
            renderedHalfH * ratio);
    _view->setDirectFramebufferMapping(ratio, offset);
}

void CChunkedVolumeViewer::startRenderJob(PendingRenderJob job)
{
    if (_closing || !job.levels) {
        return;
    }

    _renderWorkerBusy.store(true, std::memory_order_release);
    _activeRenderJob = job;

    RenderContext ctx;
    ctx.renderJob = job;
    ctx.serial = ++_renderSerial;
    ctx.fbW = job.fbW;
    ctx.fbH = job.fbH;
    ctx.surfacePtrX = job.surfacePtrX;
    ctx.surfacePtrY = job.surfacePtrY;
    ctx.scale = job.scale;
    ctx.zOff = job.zOff;
    ctx.startLevel = job.startLevel;
    ctx.overlayStartLevel = job.overlayStartLevel;
    ctx.samplingMethod = job.samplingMethod;
    ctx.overlaySamplingMethod = job.overlaySamplingMethod;
    ctx.compositeSettings = job.compositeSettings;
    ctx.windowLow = job.windowLow;
    ctx.windowHigh = job.windowHigh;
    ctx.surf = job.surf;
    ctx.levels = job.levels;
    ctx.overlayVolume = job.overlayVolume;
    ctx.overlayLevels = job.overlayLevels;
    ctx.overlayOpacity = job.overlayOpacity;
    ctx.overlayColormapId = job.overlayColormapId;
    ctx.overlayWindowLow = job.overlayWindowLow;
    ctx.overlayWindowHigh = job.overlayWindowHigh;
    ctx.overlayIntensityAsAlpha = job.overlayIntensityAsAlpha;
    ctx.overlayComposite = job.overlayComposite;
    ctx.gridSurface = job.gridSurface;
    ctx.geometryDirty = job.geometryDirty;
    ctx.prevResult = _lastRenderResult;
    ctx.profileReason = job.profileReason;
    ctx.profileCaller = job.profileCaller;
    _geometryDirty = false;

    QPointer<CChunkedVolumeViewer> guard(this);
    (void)QtConcurrent::run([guard, ctx = std::move(ctx)]() mutable {
        auto result = std::make_shared<RenderResult>(renderFrame(std::move(ctx)));
        QMetaObject::invokeMethod(qApp, [guard, result = std::move(result)]() mutable {
            if (guard) {
                guard->finishRenderOnMainThread(std::move(result));
            }
        }, Qt::QueuedConnection);
    });
}

void CChunkedVolumeViewer::submitPendingRenderJobIfNeeded()
{
    if (_closing || _renderWorkerBusy.load(std::memory_order_acquire)) {
        return;
    }
    if (!_pendingRenderJob) {
        return;
    }

    auto job = std::move(*_pendingRenderJob);
    _pendingRenderJob.reset();
    if (_displayedRenderJob &&
        renderJobsEquivalentForDisplay(job, *_displayedRenderJob)) {
        return;
    }
    startRenderJob(std::move(job));
}

void CChunkedVolumeViewer::submitRender(const char* reason, std::source_location caller)
{
    ProfileScope profile("submitRender", reason, caller);
    if (profile.enabled()) {
        profile.setDetails(std::format(
            "surf='{}' volume={} levels={} busy={} pending={}",
            _surfName, bool(_volume), bool(_levels),
            _renderWorkerBusy.load(std::memory_order_acquire),
            _pendingRenderJob.has_value()));
    }
    if (_closing) {
        profile.setDetails("action=skip closing");
        return;
    }
    syncCameraTransform();
    // Hidden viewer (minimized subwindow / background tab): skip the render.
    // Mark stale so showEvent renders it when it becomes visible.
    if (!isVisible()) {
        _renderStaleWhileHidden = true;
        profile.setDetails("action=skip hidden");
        return;
    }

    auto surf = _surfWeak.lock();
    if (!surf || !_volume || !_levels) {
        profile.setDetails("action=skip missing_input");
        return;
    }

    const bool busy = _renderWorkerBusy.load(std::memory_order_acquire);
    if (!busy) {
        resizeFramebuffer();
    }
    // Capture the job at the VIEWPORT size, never the displayed framebuffer's:
    // _framebuffer may hold a stale-size result adopted while the pane was
    // resizing (renders in flight across a layout change), and sizing new jobs
    // from it re-captures the stale size on every busy-time submit -- the pane
    // then ping-pongs between the two sizes for as long as renders keep coming
    // (visible as the image jittering up/down after a generated-view rebuild).
    const int fbW = _view && _view->viewport()
        ? std::max(1, _view->viewport()->width())
        : (!_framebuffer.isNull() ? _framebuffer.width() : 1);
    const int fbH = _view && _view->viewport()
        ? std::max(1, _view->viewport()->height())
        : (!_framebuffer.isNull() ? _framebuffer.height() : 1);
    auto job = captureRenderJob(reason, caller, surf, fbW, fbH, std::chrono::steady_clock::now());
    if (!job) {
        profile.setDetails("action=skip invalid_job");
        return;
    }

    if (busy) {
        if (_activeRenderJob &&
            renderJobsEquivalentForDisplay(*job, *_activeRenderJob)) {
            profile.setDetails("action=skip_duplicate_active");
            return;
        }
        if (_pendingRenderJob &&
            renderJobsEquivalentForDisplay(*job, *_pendingRenderJob)) {
            profile.setDetails("action=skip_duplicate_pending");
            return;
        }
        _pendingRenderJob = std::move(*job);
        updateStatusLabel();
        profile.setDetails("action=queued_latest_after_worker");
        return;
    }

    if (_displayedRenderJob &&
        renderJobsEquivalentForDisplay(*job, *_displayedRenderJob)) {
        profile.setDetails("action=skip_duplicate_displayed");
        return;
    }

    if (profile.enabled()) {
        profile.setDetails(std::format(
            "action=worker_start serial={} request={} size={}x{} level={}",
            _renderSerial + 1, job->requestId, job->fbW, job->fbH, job->startLevel));
    }
    startRenderJob(std::move(*job));
    updateStatusLabel();
}

void CChunkedVolumeViewer::finishRenderOnMainThread(std::shared_ptr<RenderResult> result)
{
    ProfileScope profile("finishRenderOnMainThread", "render worker finished",
                         std::source_location::current());
    if (profile.enabled()) {
        profile.setDetails(std::format(
            "result={} serial={} currentSerial={} pending={}",
            bool(result), result ? result->serial : 0, _renderSerial,
            _pendingRenderJob.has_value()));
    }
    _renderWorkerBusy.store(false, std::memory_order_release);
    _activeRenderJob.reset();
    if (_closing) {
        profile.setDetails("action=drop_closing");
        return;
    }
    if (!result) {
        submitPendingRenderJobIfNeeded();
        profile.setDetails("action=drop_empty_result");
        return;
    }

    _framebuffer = std::move(result->framebuffer);
    _displayedRenderJob = result->renderJob;
    _lastRenderResult = result;
    // Read what this frame lacked; it replaces an older waiting request.
    if (!result->load.empty()) {
        _pendingLoad = std::move(result->load);
        if (_loadSuperseded)
            _loadSuperseded->store(true);
        startPendingLoad();
    }
    syncCameraTransform();
    emit overlaysUpdated();
    _view->viewport()->update();
    emit renderFrameCompleted(result->serial, result->renderFrameElapsedMs);
    updateStatusLabel();
    if (_pendingRenderJob &&
        renderJobsEquivalentForDisplay(*_pendingRenderJob, result->renderJob)) {
        _pendingRenderJob.reset();
    }
    if (!_pendingRenderJob) {
        auto surf = _surfWeak.lock();
        if (surf && _volume && _levels) {
            // Viewport size, not _framebuffer: the framebuffer was just replaced
            // by this result, so sizing the staleness check from it would make a
            // stale-size result always look "equivalent" and never catch up.
            const int fbW = _view && _view->viewport()
                ? std::max(1, _view->viewport()->width())
                : (!_framebuffer.isNull() ? _framebuffer.width() : 1);
            const int fbH = _view && _view->viewport()
                ? std::max(1, _view->viewport()->height())
                : (!_framebuffer.isNull() ? _framebuffer.height() : 1);
            auto latest = captureRenderJob("catch up after stale presentation",
                                           std::source_location::current(),
                                           surf,
                                           fbW,
                                           fbH,
                                           std::chrono::steady_clock::now());
            if (latest &&
                !renderJobsEquivalentForDisplay(*latest, result->renderJob)) {
                _pendingRenderJob = std::move(*latest);
            }
        }
    }
    submitPendingRenderJobIfNeeded();

    if (profile.enabled()) {
        profile.setDetails(std::format(
            "action=display serial={} request={} worker_elapsed_ms={} framebuffer={}x{} pending={}",
            result->serial, result->renderJob.requestId, result->renderFrameElapsedMs,
            _framebuffer.width(), _framebuffer.height(), _pendingRenderJob.has_value()));
    }
}

void CChunkedVolumeViewer::renderVisible(const char* reason, std::source_location caller)
{
    ProfileScope profile("renderVisible", reason, caller);
    if (profile.enabled()) {
        profile.setDetails(std::format(
            "surf='{}' scale={:.4f} zOff={:.3f}",
            _surfName, _scale, _zOff));
    }
    if (_closing) {
        profile.setDetails("action=skip closing");
        return;
    }
    submitRender(reason ? reason : "renderVisible", caller);
    profile.setDetails("action=submit");
}

void CChunkedVolumeViewer::setVolumeWindow(float low, float high)
{
    if (_closing) {
        return;
    }
    const float clampedLow = std::clamp(low, 0.0f, 255.0f);
    float clampedHigh = std::clamp(high, 0.0f, 255.0f);
    if (clampedHigh <= clampedLow)
        clampedHigh = std::min(255.0f, clampedLow + 1.0f);
    if (std::abs(_windowLow - clampedLow) < 1e-6f &&
        std::abs(_windowHigh - clampedHigh) < 1e-6f)
        return;
    _windowLow = clampedLow;
    _windowHigh = clampedHigh;
    submitRender("volume window changed");
}

void CChunkedVolumeViewer::setOverlayVolume(std::shared_ptr<Volume> volume)
{
    if (_closing) {
        return;
    }
    _overlayVolume = std::move(volume);
    _overlayLevels.reset();
    if (_overlayVolume && _state && _state->openedVolumes()) {
        try {
            _overlayLevels = std::make_shared<const vc3d::ViewerLevels>(
                vc3d::openViewerLevels(*_state->openedVolumes(), *_overlayVolume));
        } catch (const std::exception& e) {
            Logger()->warn("overlay volume unavailable: {}", e.what());
        }
    }
    submitRender("overlay volume changed");
}

void CChunkedVolumeViewer::setOverlayOpacity(float opacity)
{
    if (_closing) {
        return;
    }
    const float nextOpacity = std::clamp(opacity, 0.0f, 1.0f);
    if (_overlayOpacity == nextOpacity)
        return;
    _overlayOpacity = nextOpacity;
    submitRender("overlay opacity changed");
}

void CChunkedVolumeViewer::setOverlayColormap(const std::string& colormapId)
{
    if (_closing) {
        return;
    }
    _overlayColormapId = colormapId;
    submitRender("overlay colormap changed");
}

void CChunkedVolumeViewer::setOverlaySamplingMethod(vc::Sampling method)
{
    if (_closing) {
        return;
    }

    const vc::Sampling sanitized = method == vc::Sampling::Trilinear
        ? vc::Sampling::Trilinear
        : vc::Sampling::Nearest;
    if (_overlaySamplingMethod == sanitized) {
        return;
    }

    _overlaySamplingMethod = sanitized;
    submitRender("overlay sampling changed");
}

void CChunkedVolumeViewer::setOverlayIntensityAsAlpha(bool enabled)
{
    if (_closing || _overlayIntensityAsAlpha == enabled) {
        return;
    }
    _overlayIntensityAsAlpha = enabled;
    submitRender("overlay alpha mode changed");
}

void CChunkedVolumeViewer::setOverlayWindow(float low, float high)
{
    if (_closing) {
        return;
    }
    _overlayWindowLow = std::clamp(low, 0.0f, 255.0f);
    _overlayWindowHigh = std::clamp(high, 0.0f, 255.0f);
    if (_overlayWindowHigh <= _overlayWindowLow) {
        _overlayWindowHigh = std::min(255.0f, _overlayWindowLow + 1.0f);
    }
    submitRender("overlay window changed");
}

void CChunkedVolumeViewer::setOverlayMaxDisplayedResolution(int level)
{
    if (_closing) {
        return;
    }

    const int clamped = std::clamp(level, 0, 5);
    if (_overlayMaxDisplayedResolution == clamped) {
        return;
    }

    _overlayMaxDisplayedResolution = clamped;
    submitRender("overlay max displayed resolution changed");
}

void CChunkedVolumeViewer::setOverlayComposite(const OverlayCompositeSettings& settings)
{
    if (_closing) {
        return;
    }

    if (_overlayComposite == settings) {
        return;
    }

    _overlayComposite = settings;
    submitRender("overlay composite changed");
}

void CChunkedVolumeViewer::panByF(float dx, float dy)
{
    const float invScale = _panSensitivity / _scale;
    // Pan deltas are screen-space; under the volumetric camera the UV view
    // center must move along the (rotated/foreshortened) screen directions.
    const cv::Matx22f m = volumetricScreenToSurface();
    const float sx = dx * invScale;
    const float sy = dy * invScale;
    _surfacePtrX -= m(0, 0) * sx + m(0, 1) * sy;
    _surfacePtrY -= m(1, 0) * sx + m(1, 1) * sy;
    if (_contentMaxU > _contentMinU) {
        _surfacePtrX = std::clamp(_surfacePtrX, _contentMinU, _contentMaxU);
        _surfacePtrY = std::clamp(_surfacePtrY, _contentMinV, _contentMaxV);
    }
    submitRender("pan");
    refreshSameWrapAnnotationOverlay();
    refreshMeasurementOverlay();
    emit overlaysUpdated();
}

void CChunkedVolumeViewer::zoomByFactorAt(float factor, const QPointF& scenePos)
{
    if (!std::isfinite(factor) || factor <= 0.0f)
        return;
    const float newScale = std::clamp(_scale * factor, kMinScale, kMaxScale);
    if (std::abs(newScale - _scale) < _scale * 1e-6f)
        return;
    const float vpW = static_cast<float>(_view->viewport()->width());
    const float vpH = static_cast<float>(_view->viewport()->height());
    const float mx = static_cast<float>(scenePos.x());
    const float my = static_cast<float>(scenePos.y());
    if (mx >= 0 && mx < vpW && my >= 0 && my < vpH) {
        const float dx = mx - vpW * 0.5f;
        const float dy = my - vpH * 0.5f;
        const float shift = 1.0f / _scale - 1.0f / newScale;
        // The zoom anchor offset is screen-space too (see panByF).
        const cv::Matx22f m = volumetricScreenToSurface();
        _surfacePtrX += (m(0, 0) * dx + m(0, 1) * dy) * shift;
        _surfacePtrY += (m(1, 0) * dx + m(1, 1) * dy) * shift;
    }
    _scale = newScale;
    recalcPyramidLevel();
    _geometryDirty = true;
    resizeFramebuffer();
    submitRender("zoom");
    refreshSameWrapAnnotationOverlay();
    refreshMeasurementOverlay();
    emit overlaysUpdated();
}

void CChunkedVolumeViewer::zoomStepsAt(int steps, const QPointF& scenePos)
{
    if (steps == 0)
        return;
    const float factor = std::pow(1.05f, static_cast<float>(steps) * _zoomSensitivity);
    zoomByFactorAt(factor, scenePos);
}

void CChunkedVolumeViewer::adjustZoomByFactor(float factor)
{
    if (!std::isfinite(factor) || factor <= 0.0f || factor == 1.0f)
        return;
    // Apply the true scale multiplier directly, centered on the viewport.
    const QPointF center(_view->viewport()->width() * 0.5,
                         _view->viewport()->height() * 0.5);
    zoomByFactorAt(factor, center);
}

void CChunkedVolumeViewer::notifyInteractiveViewChange()
{
    if (!_volume || !_levels)
        return;

    _geometryDirty = true;
    submitRender("interactive view change");
    emit overlaysUpdated();
}

void CChunkedVolumeViewer::adjustSurfaceOffset(float delta)
{
    float maxZ = 10000.0f;
    if (_volume) {
        const auto [w, h, d] = _volume->shapeXyz();
        maxZ = static_cast<float>(std::max({w, h, d}));
    }
    setZOffset(std::clamp(_zOff + delta, -maxZ, maxZ));
    _geometryDirty = true;
    submitRender("surface offset changed");
    updateStatusLabel();
}

void CChunkedVolumeViewer::resetSurfaceOffsets()
{
    _surfacePtrX = 0.0f;
    _surfacePtrY = 0.0f;
    setZOffset(0.0f);
    _geometryDirty = true;
    submitRender("surface offsets reset");
}

// Sets the normal offset, refreshing dependent overlays when it changes. Use
// this instead of assigning _zOff directly unless the notification must be
// suppressed (e.g. offscreen replay repaints).
void CChunkedVolumeViewer::setZOffset(float value)
{
    if (_zOff == value) {
        return;
    }
    _zOff = value;
    notifyNormalOffsetChanged();
    // Overlays filtered by distance to the displayed surface (fibers, points)
    // depend on the offset.
    emit overlaysUpdated();
}

// Plane viewers draw a dashed copy of the segmentation intersection displaced
// by the flattened viewer's normal offset; refresh them when that offset moves.
void CChunkedVolumeViewer::notifyNormalOffsetChanged()
{
    if (!_viewerManager || _viewerManager->segmentationViewer() != this)
        return;
    _viewerManager->forEachBaseViewer([this](VolumeViewerBase* v) {
        if (auto* viewer = dynamic_cast<CChunkedVolumeViewer*>(v); viewer && viewer != this)
            viewer->refreshOverlays();
    });
}

void CChunkedVolumeViewer::fitSurfaceInView()
{
    _surfacePtrX = 0.0f;
    _surfacePtrY = 0.0f;
    _scale = 0.5f;
    recalcPyramidLevel();
    _geometryDirty = true;
    submitRender("fit surface in view");
}

void CChunkedVolumeViewer::resetViewForCurrentContent()
{
    if (_closing || !_view) {
        return;
    }

    auto surf = _surfWeak.lock();
    if (!_volume || !surf) {
        return;
    }

    resizeFramebuffer();
    const QSize viewportSize = _view->viewport()->size();
    float minU = 0.0f;
    float maxU = 0.0f;
    float minV = 0.0f;
    float maxV = 0.0f;
    bool haveBounds = false;

    if (dynamic_cast<PlaneSurface*>(surf.get())) {
        updateContentBounds();
        if (_contentMaxU > _contentMinU && _contentMaxV > _contentMinV) {
            minU = _contentMinU;
            maxU = _contentMaxU;
            minV = _contentMinV;
            maxV = _contentMaxV;
            haveBounds = true;
        }
    } else if (auto* quad = dynamic_cast<QuadSurface*>(surf.get())) {
        try {
            quad->ensureLoaded();
            if (const cv::Mat_<cv::Vec3f>* points = quad->rawPointsPtr();
                points && !points->empty()) {
                // Surface-UV (nominal) bounds of the grid: gridToSurface maps
                // grid -> grid/scale - center.
                const cv::Vec3f center = quad->center();
                const cv::Vec2f gridScale = quad->scale();
                minU = -center[0];
                minV = -center[1];
                maxU = static_cast<float>(points->cols - 1) / gridScale[0] - center[0];
                maxV = static_cast<float>(points->rows - 1) / gridScale[1] - center[1];
                haveBounds = maxU > minU && maxV > minV;
            }
        } catch (const std::exception& e) {
            qWarning() << "Could not fit surface view:" << e.what();
        }
    }

    if (!haveBounds) {
        return;
    }

    _surfacePtrX = (minU + maxU) * 0.5f;
    _surfacePtrY = (minV + maxV) * 0.5f;
    _scale = fitScaleForExtent(maxU - minU,
                               maxV - minV,
                               viewportSize.width(),
                               viewportSize.height());
    setZOffset(0.0f);
    recalcPyramidLevel();
    _geometryDirty = true;
    updateFocusMarker();
    submitRender("reset view for current content");
    refreshSameWrapAnnotationOverlay();
    emit overlaysUpdated();
}

void CChunkedVolumeViewer::centerOnVolumePoint(const cv::Vec3f& point)
{
    auto surf = _surfWeak.lock();
    if (!surf)
        return;
    cv::Vec2f surfacePoint(0.0f, 0.0f);
    bool haveSurfacePoint = false;
    if (auto* plane = dynamic_cast<PlaneSurface*>(surf.get())) {
        const cv::Vec3f projected = plane->project(point, 1.0, 1.0);
        surfacePoint = {projected[0], projected[1]};
        haveSurfacePoint = true;
    } else if (auto* quad = dynamic_cast<QuadSurface*>(surf.get())) {
        if (const auto hit = _viewerManager ? _viewerManager->locate(*quad, point, 4.0f) : std::nullopt) {
            const cv::Vec3f loc = quad->loc(hit->ptr);
            surfacePoint = {loc[0], loc[1]};
            haveSurfacePoint = true;
        }
    }

    if (!haveSurfacePoint ||
        !std::isfinite(surfacePoint[0]) ||
        !std::isfinite(surfacePoint[1])) {
        return;
    }

    centerOnSurfacePoint(surfacePoint);
}

void CChunkedVolumeViewer::centerOnSurfacePoint(const cv::Vec2f& point)
{
    if (!std::isfinite(point[0]) || !std::isfinite(point[1]))
        return;

    _surfacePtrX = point[0];
    _surfacePtrY = point[1];
    _geometryDirty = true;
    submitRender("center on surface point");
    emit overlaysUpdated();
}

void CChunkedVolumeViewer::onZoom(int steps, QPointF scenePoint, Qt::KeyboardModifiers modifiers)
{
    auto surf = _surfWeak.lock();
    if (!surf)
        return;
    if (modifiers & Qt::ShiftModifier) {
        if (_shiftScrollOverride && _shiftScrollOverride(steps, scenePoint, modifiers)) {
            refreshCursorPositionAt(scenePoint);
            return;
        }
        if (auto* plane = dynamic_cast<PlaneSurface*>(surf.get())) {
            const cv::Vec3f normal = plane->normal({0, 0, 0});
            if (std::isfinite(normal[0]) && std::isfinite(normal[1]) &&
                std::isfinite(normal[2]) && cv::norm(normal) > 0.0f) {
                const float delta = static_cast<float>(steps) * _zScrollSensitivity;
                auto shiftedPlane = std::make_shared<PlaneSurface>(*plane);
                shiftedPlane->setOrigin(plane->origin() + normal * (delta + _zOff));
                setZOffset(0.0f);
                if (_state) {
                    _state->setSurface(_surfName, shiftedPlane, false, true);
                } else {
                    _defaultSurface = shiftedPlane;
                    _surfWeak = _defaultSurface;
                    updateContentBounds();
                    _geometryDirty = true;
                    submitRender("plane slice mouse wheel");
                }
            }
        } else {
            setZOffset(_zOff + static_cast<float>(steps) * _zScrollSensitivity);
            _geometryDirty = true;
            submitRender("z offset mouse wheel");
        }
        refreshCursorPositionAt(scenePoint);
    } else if (modifiers & Qt::ControlModifier) {
        emit sendSegmentationRadiusWheel(steps, scenePoint, sceneToVolume(scenePoint));
    } else {
        zoomStepsAt(steps > 0 ? 1 : (steps < 0 ? -1 : 0), scenePoint);
    }
}

void CChunkedVolumeViewer::onResized()
{
    if (_closing) {
        return;
    }
    resizeFramebuffer();
    _geometryDirty = true;
    submitRender("resized");
    refreshMeasurementOverlay();
    _view->viewport()->update();
}

void CChunkedVolumeViewer::onCursorMove(QPointF scenePos)
{
    _lastScenePos = scenePos;
    _lastCursorVolumePos = cursorVolumePosition(scenePos);
    updateLocalCursorCrosshair(scenePos);
    updateStatusLabel();
    if (_viewerManager) {
        _viewerManager->broadcastLinkedCursor(this, _lastCursorVolumePos);
    }
    if (!_isPanning)
        return;
    const float dx = static_cast<float>(scenePos.x() - _lastPanSceneF.x());
    const float dy = static_cast<float>(scenePos.y() - _lastPanSceneF.y());
    _lastPanSceneF = scenePos;
    if (std::abs(dx) > 0.001f || std::abs(dy) > 0.001f) {
        if (!_panSmoothingInitialized) {
            _smoothedPanDx = dx;
            _smoothedPanDy = dy;
            _panSmoothingInitialized = true;
        } else {
            _smoothedPanDx = kPanSmoothingAlpha * dx + (1.0f - kPanSmoothingAlpha) * _smoothedPanDx;
            _smoothedPanDy = kPanSmoothingAlpha * dy + (1.0f - kPanSmoothingAlpha) * _smoothedPanDy;
        }
        panByF(_smoothedPanDx, _smoothedPanDy);
    }
}

void CChunkedVolumeViewer::onPanStart(Qt::MouseButton, Qt::KeyboardModifiers)
{
    _isPanning = true;
    _panSmoothingInitialized = false;
    _smoothedPanDx = 0.0f;
    _smoothedPanDy = 0.0f;
    _lastPanSceneF = _view->mapToScene(_view->mapFromGlobal(QCursor::pos()));
}

void CChunkedVolumeViewer::onPanRelease(Qt::MouseButton, Qt::KeyboardModifiers)
{
    _isPanning = false;
    _panSmoothingInitialized = false;
    _smoothedPanDx = 0.0f;
    _smoothedPanDy = 0.0f;
    submitRender("pan released");
}

void CChunkedVolumeViewer::onVolumeClicked(QPointF scenePos, Qt::MouseButton button, Qt::KeyboardModifiers modifiers)
{
    if (_sameWrapAnnotation.enabled() && button == Qt::LeftButton && modifiers.testFlag(Qt::ShiftModifier)) {
        if (_sameWrapAnnotation.manualPathType()) {
            return;
        }
        const bool appendToPreview = _sameWrapAnnotation.hasPreview() &&
                                     !_sameWrapAnnotation.shiftReleasedSincePreview();
        if (_sameWrapAnnotation.generatePreview(
                _framebuffer,
                scenePos,
                appendToPreview,
                _scale,
                _pointCollection,
                [this](const QPointF& point) { return sceneToVolume(point); },
                [this](const cv::Vec3f& point) { return volumeToScene(point); },
                [this](const std::string& key, const std::vector<QGraphicsItem*>& items) {
                    setOverlayGroup(key, items);
                },
                [this](const std::string& key) { clearOverlayGroup(key); })) {
            return;
        }
    }

    if (button == Qt::LeftButton && modifiers == Qt::NoModifier) {
        if (const auto hit = pointAtScenePosition(scenePos)) {
            _selectedPoint = vc::PointRef{hit->first, hit->second};
            emit pointClicked(*_selectedPoint);
            emit overlaysUpdated();
            return;
        }
    }

    const auto sample = sampleSceneVolume(scenePos);
    if (!sample) {
        return;
    }
    emit sendVolumeClicked(sample->position, sample->normal, sample->surface, button, modifiers);
}

void CChunkedVolumeViewer::onCollectionSelected(uint64_t collectionId)
{
    if (_selectedPoint) {
        const auto point = _pointCollection ? _pointCollection->getPoint(*_selectedPoint)
                                            : std::optional<ColPoint>{};
        if (!point || point->collectionId != collectionId) {
            _selectedPoint.reset();
        }
    }
    emit overlaysUpdated();
}

void CChunkedVolumeViewer::clearCollectionSelection()
{
    _selectedPoint.reset();
    emit overlaysUpdated();
}

void CChunkedVolumeViewer::onPointSelected(vc::PointRef point)
{
    _selectedPoint = point;
    emit overlaysUpdated();
}

void CChunkedVolumeViewer::clearPointSelection()
{
    _selectedPoint.reset();
    emit overlaysUpdated();
}

void CChunkedVolumeViewer::setSameWrapAnnotationMode(bool enabled)
{
    _sameWrapManualPathDragActive = false;
    _sameWrapAnnotation.setEnabled(enabled);
    if (!enabled) {
        clearSameWrapAnnotationPreview();
    }
    emit overlaysUpdated();
}

void CChunkedVolumeViewer::setSameWrapAnnotationSpacing(double spacingVx)
{
    _sameWrapAnnotation.setSpacing(spacingVx);
}

void CChunkedVolumeViewer::setSameWrapAnnotationPolylineOpacity(double opacity)
{
    _sameWrapAnnotationPolylineOpacity = std::clamp(opacity, 0.0, 1.0);
    emit overlaysUpdated();
}

void CChunkedVolumeViewer::setSameWrapAnnotationMergeExisting(bool enabled)
{
    _sameWrapAnnotation.setMergeExistingAnnotations(enabled);
}

void CChunkedVolumeViewer::setSameWrapAnnotationPathType(int pathType)
{
    SameWrapAnnotationTool::PathType toolPathType = SameWrapAnnotationTool::PathType::ConnectedComponents;
    if (pathType == static_cast<int>(SameWrapAnnotationTool::PathType::ShortestPath)) {
        toolPathType = SameWrapAnnotationTool::PathType::ShortestPath;
    } else if (pathType == static_cast<int>(SameWrapAnnotationTool::PathType::Manual)) {
        toolPathType = SameWrapAnnotationTool::PathType::Manual;
    }
    _sameWrapManualPathDragActive = false;
    _sameWrapAnnotation.setPathType(toolPathType);
    clearSameWrapAnnotationPreview();
}

void CChunkedVolumeViewer::setSameWrapAnnotationFilterType(int filterType)
{
    SameWrapAnnotationTool::ImageFilterType toolFilterType = SameWrapAnnotationTool::ImageFilterType::None;
    if (filterType == static_cast<int>(SameWrapAnnotationTool::ImageFilterType::Median)) {
        toolFilterType = SameWrapAnnotationTool::ImageFilterType::Median;
    } else if (filterType == static_cast<int>(SameWrapAnnotationTool::ImageFilterType::Gaussian)) {
        toolFilterType = SameWrapAnnotationTool::ImageFilterType::Gaussian;
    }
    _sameWrapAnnotation.setImageFilterType(toolFilterType);
    clearSameWrapAnnotationPreview();
}

void CChunkedVolumeViewer::setSameWrapAnnotationFilterKernelSize(int kernelSize)
{
    _sameWrapAnnotation.setImageFilterKernelSize(kernelSize);
    clearSameWrapAnnotationPreview();
}

bool CChunkedVolumeViewer::hasSameWrapAnnotationPreview() const
{
    return _sameWrapAnnotation.hasPreview();
}

void CChunkedVolumeViewer::clearSameWrapAnnotationPreview()
{
    _sameWrapManualPathDragActive = false;
    _sameWrapAnnotation.clear([this](const std::string& key) { clearOverlayGroup(key); });
}

bool CChunkedVolumeViewer::commitSameWrapAnnotationPreview()
{
    return _sameWrapAnnotation.commit(
        _pointCollection,
        [this](const cv::Vec3f& point) { return volumeToScene(point); },
        QRectF(0.0, 0.0, _framebuffer.width(), _framebuffer.height()),
        [this](const std::string& key) { clearOverlayGroup(key); });
}

bool CChunkedVolumeViewer::undoSameWrapAnnotation()
{
    if (_sameWrapAnnotation.hasPreview()) {
        clearSameWrapAnnotationPreview();
        return true;
    }
    return _sameWrapAnnotation.undoLastCommit(_pointCollection);
}

void CChunkedVolumeViewer::refreshSameWrapAnnotationOverlay()
{
    if (!_sameWrapAnnotation.hasPreview()) {
        return;
    }

    _sameWrapAnnotation.refreshOverlay(
        [this](const cv::Vec3f& point) { return volumeToScene(point); },
        [this](const std::string& key, const std::vector<QGraphicsItem*>& items) {
            setOverlayGroup(key, items);
        },
        [this](const std::string& key) { clearOverlayGroup(key); });
}

void CChunkedVolumeViewer::onMousePress(QPointF scenePos, Qt::MouseButton button, Qt::KeyboardModifiers modifiers)
{
    _lastScenePos = scenePos;
    _lastCursorVolumePos = cursorVolumePosition(scenePos);
    updateLocalCursorCrosshair(scenePos);
    updateStatusLabel();
    if (_viewerManager) {
        _viewerManager->broadcastLinkedCursor(this, _lastCursorVolumePos);
    }
    _sameWrapManualMergePressConsumed = false;

    if (handleMeasurementClick(scenePos, button, modifiers)) {
        return;
    }

    if (_sameWrapAnnotation.enabled() && button == Qt::RightButton &&
        modifiers.testFlag(Qt::ShiftModifier)) {
        if (const auto hit = pointAtScenePosition(scenePos)) {
            if (_sameWrapAnnotation.manualMergePointClicked(
                    _pointCollection,
                    hit->first,
                    hit->second,
                    [this](const cv::Vec3f& point) { return volumeToScene(point); },
                    QRectF(0.0, 0.0, _framebuffer.width(), _framebuffer.height()),
                    [this](const SameWrapAnnotationTool::MixedDirectionMergeWarning& warning) {
                        const auto directionText = [this](const std::string& key) {
                            return key == "z" ? tr("mostly up/down in Z") : tr("mostly in XY");
                        };
                        const QMessageBox::StandardButton reply = QMessageBox::warning(
                            this,
                            tr("Same-wrap Merge"),
                            tr("You are about to merge two same-wrap point collections with different dominant directions:\n\n"
                               "%1: %2\n"
                               "%3: %4\n\n"
                               "Continue merging?")
                                .arg(QString::fromStdString(warning.firstCollectionName),
                                     directionText(warning.firstDirectionKey),
                                     QString::fromStdString(warning.secondCollectionName),
                                     directionText(warning.secondDirectionKey)),
                            QMessageBox::Yes | QMessageBox::No,
                            QMessageBox::No);
                        return reply == QMessageBox::Yes;
                    })) {
                const vc::PointRef hitRef{hit->first, hit->second};
                if (_pointCollection && _pointCollection->getPoint(hitRef)) {
                    _selectedPoint = hitRef;
                } else {
                    _selectedPoint.reset();
                }
                _sameWrapManualMergePressConsumed = true;
                if (_selectedPoint) {
                    emit pointClicked(*_selectedPoint);
                }
                emit overlaysUpdated();
                return;
            }
        }
    }

    if (_sameWrapAnnotation.enabled() && button == Qt::LeftButton &&
        modifiers.testFlag(Qt::ShiftModifier)) {
        if (_sameWrapAnnotation.manualPathType()) {
            const bool appendToPreview = _sameWrapAnnotation.hasPreview() &&
                                         !_sameWrapAnnotation.shiftReleasedSincePreview();
            _sameWrapManualPathDragActive = _sameWrapAnnotation.beginManualPreview(
                scenePos,
                appendToPreview,
                [this](const QPointF& point) { return sceneToVolume(point); },
                [this](const cv::Vec3f& point) { return volumeToScene(point); },
                [this](const std::string& key, const std::vector<QGraphicsItem*>& items) {
                    setOverlayGroup(key, items);
                },
                [this](const std::string& key) { clearOverlayGroup(key); });
        }
        return;
    }

    if (_bboxMode && _surfName == "segmentation" && button == Qt::LeftButton) {
        const cv::Vec2f sp = sceneToSurface(scenePos);
        _bboxStart = QPointF(sp[0], sp[1]);
        _activeBBoxSurfRect = QRectF(_bboxStart, QSizeF(0.0, 0.0));
        emit overlaysUpdated();
        return;
    }
    cv::Vec3f volumePos;
    if (_lastCursorVolumePos) {
        volumePos = *_lastCursorVolumePos;
    } else {
        volumePos = sceneToVolume(scenePos);
    }
    emit sendMousePressVolume(volumePos, {0, 0, 1}, button, modifiers, scenePos);
    if (_lineAnnotationPlacementPreviewEnabled && button == Qt::LeftButton &&
        modifiers == Qt::NoModifier &&
        std::isfinite(volumePos[0]) && std::isfinite(volumePos[1]) &&
        std::isfinite(volumePos[2])) {
        emit sendLineAnnotationSeedRequested(volumePos, scenePos);
    }
}

void CChunkedVolumeViewer::onMouseMove(QPointF scenePos, Qt::MouseButtons buttons, Qt::KeyboardModifiers modifiers)
{
    const bool reusedCursorSample = (_lastScenePos == scenePos && _lastCursorVolumePos.has_value());
    if (!reusedCursorSample) {
        _lastCursorVolumePos = cursorVolumePosition(scenePos);
    }
    _lastScenePos = scenePos;
    updateLocalCursorCrosshair(scenePos);
    updateStatusLabel();

    const std::optional<vc::PointRef> previousHighlight = _highlightedPoint;
    if (const auto hit = pointAtScenePosition(scenePos)) {
        _highlightedPoint = vc::PointRef{hit->first, hit->second};
    } else {
        _highlightedPoint.reset();
    }
    if (_highlightedPoint != previousHighlight) {
        emit overlaysUpdated();
    }

    if (_sameWrapManualPathDragActive && (buttons & Qt::LeftButton)) {
        _sameWrapAnnotation.appendManualPreview(
            scenePos,
            _scale,
            _pointCollection,
            [this](const QPointF& point) { return sceneToVolume(point); },
            [this](const cv::Vec3f& point) { return volumeToScene(point); },
            [this](const std::string& key, const std::vector<QGraphicsItem*>& items) {
                setOverlayGroup(key, items);
            },
            [this](const std::string& key) { clearOverlayGroup(key); });
        return;
    }
    if (_bboxMode && _activeBBoxSurfRect && (buttons & Qt::LeftButton)) {
        const cv::Vec2f sp = sceneToSurface(scenePos);
        _activeBBoxSurfRect = QRectF(_bboxStart, QPointF(sp[0], sp[1])).normalized();
        emit overlaysUpdated();
        return;
    }
    cv::Vec3f volumePos;
    if (_lastCursorVolumePos) {
        volumePos = *_lastCursorVolumePos;
    } else {
        volumePos = sceneToVolume(scenePos);
    }
    if (_lineAnnotationPlacementPreviewEnabled &&
        std::isfinite(volumePos[0]) && std::isfinite(volumePos[1]) &&
        std::isfinite(volumePos[2])) {
        updateLineAnnotationPlacementMarker(scenePos);
    } else {
        clearLineAnnotationPlacementMarker();
    }
    emit sendMouseMoveVolume(volumePos, buttons, modifiers, scenePos);
}

void CChunkedVolumeViewer::onMouseRelease(QPointF scenePos, Qt::MouseButton button, Qt::KeyboardModifiers modifiers)
{
    _lastScenePos = scenePos;
    _lastCursorVolumePos = cursorVolumePosition(scenePos);
    updateLocalCursorCrosshair(scenePos);
    updateStatusLabel();
    if (_viewerManager) {
        _viewerManager->broadcastLinkedCursor(this, _lastCursorVolumePos);
    }
    if (_sameWrapManualMergePressConsumed && button == Qt::RightButton) {
        _sameWrapManualMergePressConsumed = false;
        return;
    }
    if (_sameWrapManualPathDragActive && button == Qt::LeftButton) {
        _sameWrapAnnotation.appendManualPreview(
            scenePos,
            _scale,
            _pointCollection,
            [this](const QPointF& point) { return sceneToVolume(point); },
            [this](const cv::Vec3f& point) { return volumeToScene(point); },
            [this](const std::string& key, const std::vector<QGraphicsItem*>& items) {
                setOverlayGroup(key, items);
            },
            [this](const std::string& key) { clearOverlayGroup(key); });
        _sameWrapManualPathDragActive = false;
        return;
    }
    if (_bboxMode && _surfName == "segmentation" && button == Qt::LeftButton &&
        _activeBBoxSurfRect) {
        const cv::Vec2f sp = sceneToSurface(scenePos);
        const QRectF surfRect = QRectF(_bboxStart, QPointF(sp[0], sp[1])).normalized();
        const int idx = static_cast<int>(_selections.size());
        const QColor color = QColor::fromHsv((idx * 53) % 360, 200, 255);
        _selections.push_back({surfRect, color});
        _activeBBoxSurfRect.reset();
        emit overlaysUpdated();
        return;
    }
    cv::Vec3f volumePos;
    if (_lastCursorVolumePos) {
        volumePos = *_lastCursorVolumePos;
    } else {
        volumePos = sceneToVolume(scenePos);
    }
    emit sendMouseReleaseVolume(volumePos, button, modifiers, scenePos);
}

void CChunkedVolumeViewer::onPathsChanged(const QList<ViewerOverlayControllerBase::PathPrimitive>& paths)
{
    _drawingPaths.clear();
    _drawingPaths.reserve(static_cast<std::size_t>(paths.size()));
    for (const auto& path : paths) {
        _drawingPaths.push_back(path);
    }
    emit overlaysUpdated();
}

void CChunkedVolumeViewer::onKeyPress(int key, Qt::KeyboardModifiers)
{
    if (key == Qt::Key_Escape && isMeasurementActive()) {
        clearMeasurement();
        return;
    }

    constexpr float kPanPx = 64.0f;
    switch (key) {
        case Qt::Key_Left: panByF(kPanPx, 0); break;
        case Qt::Key_Right: panByF(-kPanPx, 0); break;
        case Qt::Key_Up: panByF(0, kPanPx); break;
        case Qt::Key_Down: panByF(0, -kPanPx); break;
        default: break;
    }
}

void CChunkedVolumeViewer::onKeyRelease(int key, Qt::KeyboardModifiers)
{
    if (key == Qt::Key_Shift) {
        _sameWrapAnnotation.noteShiftReleased();
    }
}

QPointF CChunkedVolumeViewer::surfaceToScene(float surfX, float surfY, float wPx) const
{
    // Keep the camera arithmetic in qreal. Overlay code derives an affine
    // surface-to-scene transform from this function; doing these operations in
    // float makes its one-unit basis lose precision far from surface origin,
    // and that error is then multiplied by the absolute surface coordinate.
    const qreal vpCx = static_cast<qreal>(_framebuffer.width()) * 0.5;
    const qreal vpCy = static_cast<qreal>(_framebuffer.height()) * 0.5;
    const qreal scale = static_cast<qreal>(_scale);
    const qreal vx = (static_cast<qreal>(surfX) - static_cast<qreal>(_surfacePtrX)) * scale;
    const qreal vy = (static_cast<qreal>(surfY) - static_cast<qreal>(_surfacePtrY)) * scale;
    if (!volumetricCameraActive())
        return QPointF(vx + vpCx, vy + vpCy);
    const cv::Vec2f s = volumetricSurfacePxToScreenPx(
        {static_cast<float>(vx), static_cast<float>(vy)}, wPx);
    return QPointF(s[0] + vpCx, s[1] + vpCy);
}

cv::Vec2f CChunkedVolumeViewer::sceneToSurface(const QPointF& scenePos) const
{
    if (_framebuffer.isNull() || _scale <= 0.0f)
        return {0, 0};
    const float vpCx = static_cast<float>(_framebuffer.width()) * 0.5f;
    const float vpCy = static_cast<float>(_framebuffer.height()) * 0.5f;
    const cv::Vec2f q = volumetricScreenPxToSurfacePx(
        {static_cast<float>(scenePos.x()) - vpCx,
         static_cast<float>(scenePos.y()) - vpCy});
    return {q[0] / _scale + _surfacePtrX, q[1] / _scale + _surfacePtrY};
}

QRectF CChunkedVolumeViewer::surfaceRectToSceneRect(const QRectF& surfRect) const
{
    // Bounding box of all four mapped corners: under the volumetric camera
    // the mapping can rotate, so two corners are not enough.
    const QPointF pts[4] = {
        surfaceToScene(static_cast<float>(surfRect.left()), static_cast<float>(surfRect.top())),
        surfaceToScene(static_cast<float>(surfRect.right()), static_cast<float>(surfRect.top())),
        surfaceToScene(static_cast<float>(surfRect.right()), static_cast<float>(surfRect.bottom())),
        surfaceToScene(static_cast<float>(surfRect.left()), static_cast<float>(surfRect.bottom()))};
    QRectF out(pts[0], pts[0]);
    for (int i = 1; i < 4; ++i) {
        out.setLeft(std::min(out.left(), pts[i].x()));
        out.setTop(std::min(out.top(), pts[i].y()));
        out.setRight(std::max(out.right(), pts[i].x()));
        out.setBottom(std::max(out.bottom(), pts[i].y()));
    }
    return out;
}

cv::Vec2f CChunkedVolumeViewer::sceneToSurfaceCoords(const QPointF& scenePos) const
{
    return sceneToSurface(scenePos);
}

// Match the points-overlay default view tolerance so points that are meant to
// render faded still project to their true location rather than collapsing.
static constexpr float kQuadProjectTolerance = 10.0f;

void CChunkedVolumeViewer::quadProjectDepthBand(float& depthLo, float& depthHi) const
{
    // The view shows content displaced by the normal offset — and, when
    // compositing, the whole slab. Points anywhere in that visible signed
    // depth band along the normal (e.g. the linked cursor from a slice
    // view) must still project; points outside it — including the mirror
    // side of the surface — must not.
    depthLo = _zOff;
    depthHi = _zOff;
    if (isCompositeEnabled()) {
        const float zStep = _compositeSettings.reverseDirection ? -1.0f : 1.0f;
        const float front = _zOff + float(std::max(0, _compositeSettings.layersFront)) * zStep;
        const float behind = _zOff - float(std::max(0, _compositeSettings.layersBehind)) * zStep;
        depthLo = std::min(front, behind);
        depthHi = std::max(front, behind);
    }
}

SurfaceProjectionContext
CChunkedVolumeViewer::surfaceProjectionContext() const
{
    SurfaceProjectionContext context;
    auto surf = _surfWeak.lock();
    context.surface = surf.get();
    context.surfaceEditEpoch = _surfaceEditEpoch;
    quadProjectDepthBand(context.depthLo, context.depthHi);
    if (auto* plane = dynamic_cast<PlaneSurface*>(surf.get())) {
        context.planeOrigin = plane->origin();
        context.planeBasisX = plane->basisX();
        context.planeBasisY = plane->basisY();
    }
    return context;
}

std::optional<SurfaceProjection>
CChunkedVolumeViewer::projectVolumePoint(const cv::Vec3f& volPoint,
                                         float depthTolerance) const
{
    auto surf = _surfWeak.lock();
    if (!surf)
        return std::nullopt;
    float depthLo = 0.0f;
    float depthHi = 0.0f;
    quadProjectDepthBand(depthLo, depthHi);
    if (auto* plane = dynamic_cast<PlaneSurface*>(surf.get())) {
        const cv::Vec3f proj = plane->project(volPoint, 1.0, 1.0);
        return SurfaceProjection{
            proj[0], proj[1], 0.0f, false,
            depthBandDistance(plane->scalarp(volPoint), depthLo, depthHi)};
    }
    if (auto* quad = dynamic_cast<QuadSurface*>(surf.get())) {
        const float tolerance =
            depthTolerance + std::max(std::abs(depthLo), std::abs(depthHi));
        const auto hit = _viewerManager ? _viewerManager->locate(*quad, volPoint, tolerance) : std::nullopt;
        if (!hit)
            return std::nullopt;
        const cv::Vec3f ptr = hit->ptr;
        const float dist = hit->distance;
        // Gate on the signed offset along the surface normal so only the
        // depth band the view actually displays accepts the point.
        float w = _zOff;
        float distance = dist;
        const cv::Vec3f surfCoord = quad->coord(ptr);
        const cv::Vec3f surfNormal = quad->normal(ptr);
        if (vc::geometry::validSurfacePoint(surfCoord) && finiteVec3(surfNormal)) {
            w = (volPoint - surfCoord).dot(surfNormal);
            if (w < depthLo - depthTolerance || w > depthHi + depthTolerance)
                return std::nullopt;
            distance = depthBandDistance(volPoint, surfCoord, surfNormal, depthLo, depthHi);
        }
        const cv::Vec3f loc = quad->loc(ptr);
        return SurfaceProjection{loc[0], loc[1], w, true, distance};
    }
    return std::nullopt;
}

QPointF CChunkedVolumeViewer::surfaceProjectionToScene(
    const SurfaceProjection& projection) const
{
    // Under the volumetric camera, points off the w=0 anchor plane (the
    // offset surface) render displaced by tilt/perspective; place the
    // marker with the render's own per-w mapping. wPx matches the render:
    // slab w in layers (zStep folds the reverse-direction sign back in)
    // times output scale times the wScale relief exaggeration.
    float wPx = 0.0f;
    if (projection.applyVolumetricW && volumetricCameraActive()) {
        const float zStep = _compositeSettings.reverseDirection ? -1.0f : 1.0f;
        const float wScale = std::max(_compositeSettings.params.wScale, 0.01f);
        wPx = (projection.w - _zOff) * zStep * _scale * wScale;
    }
    return surfaceToScene(projection.u, projection.v, wPx);
}

QPointF CChunkedVolumeViewer::volumeToScene(const cv::Vec3f& volPoint)
{
    if (const auto projection = projectVolumePoint(volPoint, kQuadProjectTolerance))
        return surfaceProjectionToScene(*projection);
    // Only on failure is it worth re-establishing why: a surface that is
    // absent, or neither a plane nor a quad, maps to a default-constructed
    // point, which callers treat differently from the NaN that means the
    // point simply missed the surface.
    auto surf = _surfWeak.lock();
    if (!surf ||
        (!dynamic_cast<PlaneSurface*>(surf.get()) && !dynamic_cast<QuadSurface*>(surf.get())))
        return {};
    return {NAN, NAN};
}

void CChunkedVolumeViewer::updateCursorCrosshair(const QPointF& scenePos, bool projected)
{
    if (!_scene || !std::isfinite(scenePos.x()) || !std::isfinite(scenePos.y()))
        return;

    const auto crosshairPen = [](bool greyedOut) {
        QPen pen(greyedOut ? QColor(160, 160, 160, 190) : QColor(50, 255, 215),
                 2.0, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin);
        pen.setCosmetic(true);
        return pen;
    };

    if (!_cursorCrosshair || !_cursorCrosshair->scene()) {
        QPainterPath path;
        constexpr qreal radius = 6.0;
        constexpr qreal arm = 14.0;
        constexpr qreal gap = 3.0;
        path.addEllipse(QPointF(0.0, 0.0), radius, radius);
        path.moveTo(-arm, 0.0);
        path.lineTo(-gap, 0.0);
        path.moveTo(gap, 0.0);
        path.lineTo(arm, 0.0);
        path.moveTo(0.0, -arm);
        path.lineTo(0.0, -gap);
        path.moveTo(0.0, gap);
        path.lineTo(0.0, arm);

        auto* marker = new QGraphicsPathItem(path);
        marker->setPen(crosshairPen(projected));
        marker->setBrush(Qt::NoBrush);
        marker->setZValue(120.0);
        marker->setAcceptedMouseButtons(Qt::NoButton);
        _scene->addItem(marker);
        _cursorCrosshair = marker;
        _cursorCrosshairProjected = projected;
    } else if (projected != _cursorCrosshairProjected) {
        static_cast<QGraphicsPathItem*>(_cursorCrosshair)->setPen(crosshairPen(projected));
        _cursorCrosshairProjected = projected;
    }

    _cursorCrosshair->setPos(scenePos);
    _cursorCrosshair->show();
}

void CChunkedVolumeViewer::updateLocalCursorCrosshair(const QPointF& scenePos)
{
    if (_localCursorCrosshairSuppressed) {
        if (_cursorCrosshair) _cursorCrosshair->hide();
        return;
    }
    updateCursorCrosshair(scenePos);
}

void CChunkedVolumeViewer::setLocalCursorCrosshairSuppressed(bool suppressed)
{
    if (_localCursorCrosshairSuppressed == suppressed) return;
    _localCursorCrosshairSuppressed = suppressed;
    if (suppressed) {
        if (_cursorCrosshair) _cursorCrosshair->hide();
    } else if (_lastCursorVolumePos
               && std::isfinite(_lastScenePos.x())
               && std::isfinite(_lastScenePos.y())) {
        updateCursorCrosshair(_lastScenePos);
    }
}

void CChunkedVolumeViewer::setSegmentationCursorMirroring(bool enabled)
{
    _segmentationCursorMirroring = enabled;
    if (!enabled && _cursorCrosshair) {
        _cursorCrosshair->hide();
    }
}

void CChunkedVolumeViewer::setLineAnnotationPlacementPreviewEnabled(bool enabled)
{
    _lineAnnotationPlacementPreviewEnabled = enabled;
    if (!enabled) {
        clearLineAnnotationPlacementMarker();
    }
}

void CChunkedVolumeViewer::markSurfaceGeometryChanged()
{
    if (_closing) {
        return;
    }
    ++_surfaceGeometryEpoch;
    _geometryDirty = true;
    _gridSurface.reset();
}

void CChunkedVolumeViewer::updateLineAnnotationPlacementMarker(const QPointF& scenePos)
{
    if (!_lineAnnotationPlacementPreviewEnabled || !_scene ||
        !std::isfinite(scenePos.x()) || !std::isfinite(scenePos.y())) {
        clearLineAnnotationPlacementMarker();
        return;
    }

    if (!_lineAnnotationPlacementMarker || !_lineAnnotationPlacementMarker->scene()) {
        auto* marker = new QGraphicsEllipseItem(-6.0, -6.0, 12.0, 12.0);
        QPen pen(QColor(Qt::yellow), 2.0);
        pen.setCosmetic(true);
        marker->setPen(pen);
        marker->setBrush(QBrush(QColor(255, 255, 0, 70)));
        marker->setZValue(135.0);
        marker->setAcceptedMouseButtons(Qt::NoButton);
        _scene->addItem(marker);
        _lineAnnotationPlacementMarker = marker;
    }

    _lineAnnotationPlacementMarker->setPos(scenePos);
    _lineAnnotationPlacementMarker->show();
}

void CChunkedVolumeViewer::clearLineAnnotationPlacementMarker()
{
    if (_lineAnnotationPlacementMarker) {
        _lineAnnotationPlacementMarker->hide();
    }
}

void CChunkedVolumeViewer::startMeasurementMode()
{
    if (!measurementSupported()) {
        return;
    }
    _measurement = {};
    _measurement.active = true;
    clearOverlayGroup(kMeasurementOverlayKey);
    if (_view) {
        _view->setFocus(Qt::MouseFocusReason);
        _view->setCursor(Qt::CrossCursor);
    }
}

bool CChunkedVolumeViewer::measurementSupported() const
{
    auto surf = _surfWeak.lock();
    if (!surf) {
        return false;
    }
    return _surfName == "segmentation" ||
           dynamic_cast<PlaneSurface*>(surf.get()) != nullptr;
}

bool CChunkedVolumeViewer::isMeasurementActive() const
{
    return _measurement.active;
}

void CChunkedVolumeViewer::clearMeasurement()
{
    _measurement = {};
    clearOverlayGroup(kMeasurementOverlayKey);
    if (_view) {
        _view->unsetCursor();
        _view->viewport()->update();
    }
}

bool CChunkedVolumeViewer::handleMeasurementClick(const QPointF& scenePos,
                                                  Qt::MouseButton button,
                                                  Qt::KeyboardModifiers modifiers)
{
    if (!_measurement.active || !measurementSupported()) {
        return false;
    }
    if (button != Qt::LeftButton || modifiers != Qt::NoModifier) {
        return true;
    }

    const cv::Vec3f volumePos = _lastCursorVolumePos ? *_lastCursorVolumePos : sceneToVolume(scenePos);
    if (!std::isfinite(volumePos[0]) ||
        !std::isfinite(volumePos[1]) ||
        !std::isfinite(volumePos[2])) {
        return true;
    }

    MeasurementPoint point;
    point.surface = sceneToSurface(scenePos);
    point.volume = volumePos;
    if (!_measurement.first || _measurement.second) {
        _measurement.first = point;
        _measurement.second.reset();
    } else {
        _measurement.second = point;
    }
    refreshMeasurementOverlay();
    return true;
}

void CChunkedVolumeViewer::refreshMeasurementOverlay()
{
    if (!_measurement.active && !_measurement.first && !_measurement.second) {
        clearOverlayGroup(kMeasurementOverlayKey);
        return;
    }

    std::vector<QGraphicsItem*> items;
    auto makeMarker = [](const QPointF& pos) {
        auto* marker = new QGraphicsEllipseItem(-4.0, -4.0, 8.0, 8.0);
        QPen pen(QColor(255, 230, 80), 1.5);
        pen.setCosmetic(true);
        marker->setPen(pen);
        marker->setBrush(QBrush(QColor(255, 230, 80, 90)));
        marker->setPos(pos);
        marker->setZValue(145.0);
        marker->setAcceptedMouseButtons(Qt::NoButton);
        return marker;
    };

    std::optional<QPointF> firstScene;
    std::optional<QPointF> secondScene;
    if (_measurement.first) {
        firstScene = surfaceToScene(_measurement.first->surface[0], _measurement.first->surface[1]);
        if (std::isfinite(firstScene->x()) && std::isfinite(firstScene->y())) {
            items.push_back(makeMarker(*firstScene));
        }
    }
    if (_measurement.second) {
        secondScene = surfaceToScene(_measurement.second->surface[0], _measurement.second->surface[1]);
        if (std::isfinite(secondScene->x()) && std::isfinite(secondScene->y())) {
            items.push_back(makeMarker(*secondScene));
        }
    }

    if (firstScene && secondScene &&
        std::isfinite(firstScene->x()) && std::isfinite(firstScene->y()) &&
        std::isfinite(secondScene->x()) && std::isfinite(secondScene->y())) {
        QPainterPath path;
        path.moveTo(*firstScene);
        path.lineTo(*secondScene);
        auto* line = new QGraphicsPathItem(path);
        QPen pen(QColor(255, 230, 80), 2.0, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin);
        pen.setCosmetic(true);
        line->setPen(pen);
        line->setBrush(Qt::NoBrush);
        line->setZValue(144.0);
        line->setAcceptedMouseButtons(Qt::NoButton);
        items.push_back(line);

        const double um = _volume
            ? volumeDistanceMicrometers(_measurement.first->volume,
                                        _measurement.second->volume,
                                        _volume->voxelSize())
            : std::numeric_limits<double>::quiet_NaN();
        if (std::isfinite(um)) {
            const QPointF mid = (*firstScene + *secondScene) * 0.5;
            const QPointF delta = *secondScene - *firstScene;
            const double len = std::hypot(delta.x(), delta.y());
            QPointF offset(0.0, -16.0);
            if (len > 1.0) {
                offset = QPointF(-delta.y() / len * 16.0, delta.x() / len * 16.0);
                if (offset.y() > 0.0) {
                    offset = -offset;
                }
            }
            const QString label = CVolumeViewerView::formatScaleBarLength(um).text;
            QFont font;
            font.setPointSize(12);
            font.setBold(true);

            auto* shadow = new QGraphicsSimpleTextItem(label);
            shadow->setFont(font);
            shadow->setBrush(QBrush(Qt::black));
            shadow->setPos(mid + offset + QPointF(1.0, 1.0));
            shadow->setZValue(145.0);
            shadow->setAcceptedMouseButtons(Qt::NoButton);
            items.push_back(shadow);

            auto* text = new QGraphicsSimpleTextItem(label);
            text->setFont(font);
            text->setBrush(QBrush(Qt::white));
            text->setPos(mid + offset);
            text->setZValue(146.0);
            text->setAcceptedMouseButtons(Qt::NoButton);
            items.push_back(text);
        }
    }

    setOverlayGroup(kMeasurementOverlayKey, items);
    if (_view) {
        _view->viewport()->update();
    }
}

std::optional<cv::Vec3f> CChunkedVolumeViewer::cursorVolumePosition(const QPointF& scenePos) const
{
    auto surf = _surfWeak.lock();
    if (!surf)
        return std::nullopt;

    cv::Vec3f p = sceneToVolume(scenePos);
    if (auto* plane = dynamic_cast<PlaneSurface*>(surf.get())) {
        p += plane->normal({0, 0, 0}) * _zOff;
    } else if (_zOff != 0.0f) {
        const cv::Vec2f sp = sceneToSurface(scenePos);
        const cv::Vec3f n = surf->normal({0, 0, 0}, {sp[0], sp[1], 0.0f});
        if (std::isfinite(n[0]) && std::isfinite(n[1]) && std::isfinite(n[2]))
            p += n * _zOff;
    }
    return p;
}

void CChunkedVolumeViewer::refreshCursorPositionAt(const QPointF& scenePos)
{
    _lastScenePos = scenePos;
    _lastCursorVolumePos = cursorVolumePosition(scenePos);
    updateLocalCursorCrosshair(scenePos);
    updateStatusLabel();
    if (_viewerManager) {
        _viewerManager->broadcastLinkedCursor(this, _lastCursorVolumePos);
    }
}

std::optional<std::pair<uint64_t, uint64_t>> CChunkedVolumeViewer::pointAtScenePosition(const QPointF& scenePos)
{
    if (!_pointCollection || !std::isfinite(scenePos.x()) || !std::isfinite(scenePos.y())) {
        return std::nullopt;
    }

    const auto& collections = _pointCollection->getAllCollections();
    if (collections.empty()) {
        return std::nullopt;
    }

    constexpr qreal kPointHitRadius = 4.0;
    constexpr qreal kPointHitRadiusSq = kPointHitRadius * kPointHitRadius;

    // volumeToScene() is expensive on a QuadSurface (a per-point pointTo search) and
    // this runs on every mouse-move for hover highlighting. Reject points that are
    // obviously too far before paying for the projection: the cursor's volume position
    // lies on the current surface, so a marker can only fall within kPointHitRadius
    // scene px if it is within ~kPointHitRadius/_scale surface units laterally, plus a
    // normal-direction margin covering the depth band that is actually drawn (markers
    // fade out past the view tolerance, whose generous ceiling we bound below).
    const std::optional<cv::Vec3f> cursorVol = cursorVolumePosition(scenePos);
    constexpr float kHoverDepthMarginVx = 256.0f;
    const float lateralBoundVx = static_cast<float>(kPointHitRadius) / std::max(_scale, 1e-3f);
    const float prefilterRadiusVx = lateralBoundVx + kHoverDepthMarginVx;
    const float prefilterRadiusSqVx = prefilterRadiusVx * prefilterRadiusVx;

    qreal bestDistSq = kPointHitRadiusSq;
    std::optional<std::pair<uint64_t, uint64_t>> bestHit;

    for (const auto& [collectionId, collection] : collections) {
        for (const auto& [pointId, point] : collection.points) {
            if (cursorVol) {
                const cv::Vec3f delta = point.p - *cursorVol;
                if (delta.dot(delta) > prefilterRadiusSqVx) {
                    continue;
                }
            }

            const QPointF pointScenePos = volumeToScene(point.p);
            if (!std::isfinite(pointScenePos.x()) || !std::isfinite(pointScenePos.y())) {
                continue;
            }

            const qreal dx = pointScenePos.x() - scenePos.x();
            const qreal dy = pointScenePos.y() - scenePos.y();
            const qreal distSq = dx * dx + dy * dy;
            if (distSq <= bestDistSq) {
                bestDistSq = distSq;
                bestHit = std::make_pair(collectionId, pointId);
            }
        }
    }

    return bestHit;
}

void CChunkedVolumeViewer::setLinkedCursorVolumePoint(const std::optional<cv::Vec3f>& point)
{
    auto hideCrosshair = [this]() {
        if (_cursorCrosshair) {
            _cursorCrosshair->hide();
        }
    };

    const bool accepted = !_linkedCursorMirroringSuppressed &&
        (_segmentationCursorMirroring || _linkedCursorAlwaysEnabled) && point.has_value();
    // Track the point for the status-bar position readout even when it doesn't
    // project onto this viewer's surface and the crosshair stays hidden. Only
    // refresh the (monolithic) status bar when the value changed and this
    // viewer actually displays the position — this runs per mouse move on
    // every mirrored viewer.
    const std::optional<cv::Vec3f> linkedPoint = accepted ? point : std::nullopt;
    if (linkedPoint != _linkedCursorVolumePos) {
        _linkedCursorVolumePos = linkedPoint;
        if (!property("vc_hide_status_position").toBool()) {
            updateStatusLabel();
        }
    }

    if (!accepted) {
        hideCrosshair();
        return;
    }

    QPointF scenePos;
    bool offPlane = false;
    if (auto* plane = dynamic_cast<PlaneSurface*>(currentSurface())) {
        // The flattened viewer's cursor point includes its normal offset, which
        // pushes it off-plane by up to that amount; widen the depth band so the
        // crosshair still lands on the dashed offset intersection line.
        float tolerance = _linkedCursorViewTolerance;
        if (auto* segViewer =
                _viewerManager ? _viewerManager->segmentationViewer() : nullptr;
            segViewer && segViewer != this) {
            tolerance += std::abs(segViewer->normalOffset());
        }
        const cv::Vec3f projected = plane->project(*point, 1.0f, 1.0f);
        if (!std::isfinite(projected[0]) ||
            !std::isfinite(projected[1]) ||
            !std::isfinite(projected[2])) {
            hideCrosshair();
            return;
        }
        // Beyond the depth band the point is not on this plane (typically after
        // a normal-offset scroll); show its in-plane projection greyed out
        // instead of hiding — the projection is already computed.
        offPlane = std::abs(projected[2] - _zOff) > tolerance;
        scenePos = surfaceToScene(projected[0], projected[1]);
    } else {
        scenePos = volumeToScene(*point);
    }

    if (!std::isfinite(scenePos.x()) || !std::isfinite(scenePos.y())) {
        hideCrosshair();
        return;
    }

    updateCursorCrosshair(scenePos, offPlane);
}

void CChunkedVolumeViewer::updateFocusMarker(POI* poi)
{
    if (!_scene)
        return;
    if (!poi && _state)
        poi = _state->poi("focus");
    if (!poi || !_surfWeak.lock()) {
        if (_focusMarker)
            _focusMarker->hide();
        return;
    }

    if (!_focusMarker || !_focusMarker->scene()) {
        auto* marker = new QGraphicsEllipseItem(-10.0, -10.0, 20.0, 20.0);
        QPen pen(QColor(50, 255, 215), 3.0, Qt::DashDotLine, Qt::RoundCap, Qt::RoundJoin);
        pen.setCosmetic(true);
        marker->setPen(pen);
        marker->setBrush(Qt::NoBrush);
        marker->setZValue(110.0);
        marker->setAcceptedMouseButtons(Qt::NoButton);
        _scene->addItem(marker);
        _focusMarker = marker;
    }

    const QPointF scenePos = volumeToScene(poi->p);
    if (!std::isfinite(scenePos.x()) || !std::isfinite(scenePos.y())) {
        _focusMarker->hide();
        return;
    }

    _focusMarker->setPos(scenePos);
    _focusMarker->show();
}

cv::Vec3f CChunkedVolumeViewer::sceneToVolume(const QPointF& scenePoint) const
{
    auto surf = _surfWeak.lock();
    if (!surf)
        return {0, 0, 0};
    const cv::Vec2f sp = sceneToSurface(scenePoint);
    return surf->coord({0, 0, 0}, {sp[0], sp[1], 0});
}

std::optional<CChunkedVolumeViewer::SceneVolumeSample> CChunkedVolumeViewer::sampleSceneVolume(
    const QPointF& scenePoint) const
{
    auto surf = _surfWeak.lock();
    if (!surf) {
        return std::nullopt;
    }

    const auto cursorPos = cursorVolumePosition(scenePoint);
    SceneVolumeSample sample;
    sample.position = cursorPos ? *cursorPos : sceneToVolume(scenePoint);
    sample.surface = surf.get();
    if (!std::isfinite(sample.position[0]) ||
        !std::isfinite(sample.position[1]) ||
        !std::isfinite(sample.position[2])) {
        return std::nullopt;
    }

    if (auto* plane = dynamic_cast<PlaneSurface*>(surf.get())) {
        sample.normal = plane->normal({0, 0, 0});
    } else {
        const cv::Vec2f sp = sceneToSurface(scenePoint);
        const cv::Vec3f surfaceNormal = surf->normal({0, 0, 0}, {sp[0], sp[1], 0.0f});
        if (std::isfinite(surfaceNormal[0]) &&
            std::isfinite(surfaceNormal[1]) &&
            std::isfinite(surfaceNormal[2]) &&
            cv::norm(surfaceNormal) > 0.0f) {
            sample.normal = surfaceNormal;
        }
    }
    if (!std::isfinite(sample.normal[0]) ||
        !std::isfinite(sample.normal[1]) ||
        !std::isfinite(sample.normal[2]) ||
        cv::norm(sample.normal) <= 0.0f) {
        sample.normal = {0, 0, 1};
    }
    return sample;
}

void CChunkedVolumeViewer::setOverlayGroup(const std::string& key, const std::vector<QGraphicsItem*>& items)
{
    clearOverlayGroup(key);
    _overlayGroups[key] = items;
    for (auto* item : items) {
        if (item && !item->scene())
            _scene->addItem(item);
    }
}

bool CChunkedVolumeViewer::translateOverlayGroup(const std::string& key, const QPointF& delta)
{
    const auto it = _overlayGroups.find(key);
    if (it == _overlayGroups.end())
        return false;
    if (delta.isNull())
        return true;
    for (auto* item : it->second) {
        if (item)
            item->moveBy(delta.x(), delta.y());
    }
    return true;
}

void CChunkedVolumeViewer::clearOverlayGroup(const std::string& key)
{
    auto it = _overlayGroups.find(key);
    if (it == _overlayGroups.end())
        return;
    // Detach before deleting: the map must never name a freed item
    // (translateOverlayGroup walks the registered items).
    const std::vector<QGraphicsItem*> items = std::move(it->second);
    _overlayGroups.erase(it);
    for (auto* item : items)
        delete item;
}

std::vector<std::pair<QRectF, QColor>> CChunkedVolumeViewer::selections() const
{
    std::vector<std::pair<QRectF, QColor>> out;
    out.reserve(_selections.size());
    for (const auto& selection : _selections) {
        out.emplace_back(surfaceRectToSceneRect(selection.surfRect), selection.color);
    }
    return out;
}

std::optional<QRectF> CChunkedVolumeViewer::activeBBoxSceneRect() const
{
    if (!_activeBBoxSurfRect)
        return std::nullopt;
    return surfaceRectToSceneRect(*_activeBBoxSurfRect);
}

void CChunkedVolumeViewer::setBBoxMode(bool enabled)
{
    _bboxMode = enabled;
    if (!enabled && _activeBBoxSurfRect) {
        _activeBBoxSurfRect.reset();
        emit overlaysUpdated();
    }
}

QuadSurface* CChunkedVolumeViewer::makeBBoxFilteredSurfaceFromSceneRect(const QRectF& sceneRect)
{
    if (_surfName != "segmentation")
        return nullptr;

    auto surf = _surfWeak.lock();
    auto* quad = dynamic_cast<QuadSurface*>(surf.get());
    if (!quad)
        return nullptr;

    const cv::Mat_<cv::Vec3f> src = quad->rawPoints();
    const int h = src.rows;
    const int w = src.cols;
    if (h <= 0 || w <= 0)
        return nullptr;

    const cv::Vec2f sp0 = sceneToSurface(sceneRect.topLeft());
    const cv::Vec2f sp1 = sceneToSurface(sceneRect.bottomRight());
    QRectF surfRect(QPointF(sp0[0], sp0[1]), QPointF(sp1[0], sp1[1]));
    surfRect = surfRect.normalized();

    const double cx = w * 0.5;
    const double cy = h * 0.5;
    const cv::Vec2f scale = quad->scale();
    if (scale[0] == 0.0f || scale[1] == 0.0f)
        return nullptr;

    const int i0 = std::max(0, static_cast<int>(std::floor(cx + surfRect.left() * scale[0])));
    const int i1 = std::min(w - 1, static_cast<int>(std::ceil(cx + surfRect.right() * scale[0])));
    const int j0 = std::max(0, static_cast<int>(std::floor(cy + surfRect.top() * scale[1])));
    const int j1 = std::min(h - 1, static_cast<int>(std::ceil(cy + surfRect.bottom() * scale[1])));
    if (i0 > i1 || j0 > j1)
        return nullptr;

    cv::Mat_<cv::Vec3f> cropped(j1 - j0 + 1, i1 - i0 + 1, cv::Vec3f(-1.0f, -1.0f, -1.0f));
    for (int j = j0; j <= j1; ++j) {
        for (int i = i0; i <= i1; ++i) {
            const cv::Vec3f& p = src(j, i);
            if (!vc::geometry::validSurfacePoint(p))
                continue;
            const double u = (i - cx) / scale[0];
            const double v = (j - cy) / scale[1];
            if (u >= surfRect.left() && u <= surfRect.right() &&
                v >= surfRect.top() && v <= surfRect.bottom()) {
                cropped(j - j0, i - i0) = p;
            }
        }
    }

    cv::Mat_<cv::Vec3f> cleaned = clean_surface_outliers(cropped);

    auto countValidInCol = [&](int c) {
        int count = 0;
        for (int r = 0; r < cleaned.rows; ++r) {
            if (vc::geometry::validSurfacePoint(cleaned(r, c)))
                ++count;
        }
        return count;
    };
    auto countValidInRow = [&](int r) {
        int count = 0;
        for (int c = 0; c < cleaned.cols; ++c) {
            if (vc::geometry::validSurfacePoint(cleaned(r, c)))
                ++count;
        }
        return count;
    };

    const int minValidCol = std::max(1, std::min(3, cleaned.rows));
    const int minValidRow = std::max(1, std::min(3, cleaned.cols));
    int left = 0;
    int right = cleaned.cols - 1;
    int top = 0;
    int bottom = cleaned.rows - 1;
    while (left <= right && countValidInCol(left) < minValidCol)
        ++left;
    while (right >= left && countValidInCol(right) < minValidCol)
        --right;
    while (top <= bottom && countValidInRow(top) < minValidRow)
        ++top;
    while (bottom >= top && countValidInRow(bottom) < minValidRow)
        --bottom;

    if (left > right || top > bottom) {
        left = cleaned.cols;
        right = -1;
        top = cleaned.rows;
        bottom = -1;
        for (int j = 0; j < cleaned.rows; ++j) {
            for (int i = 0; i < cleaned.cols; ++i) {
                if (vc::geometry::validSurfacePoint(cleaned(j, i))) {
                    left = std::min(left, i);
                    right = std::max(right, i);
                    top = std::min(top, j);
                    bottom = std::max(bottom, j);
                }
            }
        }
        if (right < 0 || bottom < 0)
            return nullptr;
    }

    cv::Mat_<cv::Vec3f> finalPts(bottom - top + 1, right - left + 1,
                                  cv::Vec3f(-1.0f, -1.0f, -1.0f));
    for (int j = top; j <= bottom; ++j) {
        for (int i = left; i <= right; ++i) {
            finalPts(j - top, i - left) = cleaned(j, i);
        }
    }

    return new QuadSurface(finalPts, quad->scale());
}

void CChunkedVolumeViewer::clearSelections()
{
    _selections.clear();
    emit overlaysUpdated();
}

void CChunkedVolumeViewer::setIntersects(const std::set<std::string>& names)
{
    if (_closing || _intersectTgts == names) {
        return;
    }
    _intersectTgts = names;
    emit overlaysUpdated();
}

void CChunkedVolumeViewer::setPlaneIntersectionLinesVisible(bool visible)
{
    if (_closing || _planeIntersectionLinesVisible == visible) {
        return;
    }
    _planeIntersectionLinesVisible = visible;
    emit overlaysUpdated();
}

void CChunkedVolumeViewer::setIntersectionOpacity(float v)
{
    if (_closing) {
        return;
    }
    const float clamped = std::clamp(v, 0.0f, 1.0f);
    if (std::abs(_intersectionOpacity - clamped) < 1e-6f) {
        return;
    }
    _intersectionOpacity = clamped;
    emit overlaysUpdated();
}

void CChunkedVolumeViewer::setIntersectionThickness(float v)
{
    if (_closing) {
        return;
    }
    const float clamped = std::max(0.0f, v);
    if (std::abs(_intersectionThickness - clamped) < 1e-6f) {
        return;
    }
    _intersectionThickness = clamped;
    emit overlaysUpdated();
}

const std::vector<ViewerOverlayControllerBase::PathPrimitive>& CChunkedVolumeViewer::drawingPaths() const
{
    return _drawingPaths;
}

const std::map<std::string, cv::Vec3b>& CChunkedVolumeViewer::surfaceOverlays() const
{
    return _surfaceOverlays;
}

void CChunkedVolumeViewer::updateStatusLabel()
{
    // Piggyback on the status refresh: it runs on every surface/settings
    // change, which is exactly when the gizmo's visibility can change.
    updateCameraGizmo();

    if (!_statsBar)
        return;

    QStringList items;
    QStringList sharedCacheItems;
    // The stats group (mip level, label, scale, framebuffer, composite) can be
    // routed to the right-anchored bar or dropped per viewer.
    QStringList statsItems;
    statsItems << QString("L%1").arg(_dsScaleIdx);
    if (const QString viewerLabel = property("vc_viewer_label").toString(); !viewerLabel.isEmpty())
        statsItems << viewerLabel;
    statsItems << QString("scale %1").arg(_scale, 0, 'f', 2);

    if (volumetricCameraActive()) {
        QString volumetric = QString("composite volumetric az %1 tilt %2")
                                 .arg(_compositeSettings.params.camAzimuthDeg, 0, 'f', 0)
                                 .arg(_compositeSettings.params.camTiltDeg, 0, 'f', 0);
        if (_compositeSettings.params.camPerspective > 0.0f)
            volumetric += QString(" persp %1").arg(_compositeSettings.params.camPerspective, 0, 'f', 2);
        // W scaling only applies to the flattened view's bendy slab.
        auto surf = _surfWeak.lock();
        const bool planeView = surf && dynamic_cast<PlaneSurface*>(surf.get()) != nullptr;
        if (!planeView && _compositeSettings.params.wScale != 1.0f)
            volumetric += QString(" w×%1").arg(_compositeSettings.params.wScale, 0, 'f', 1);
        statsItems << volumetric;
    } else if (_compositeSettings.enabled || _compositeSettings.planeEnabled) {
        statsItems << QString("composite %1").arg(QString::fromStdString(compositeMethodName(_compositeSettings.params)));
    }

    const bool showStats = !property("vc_hide_status_stats").toBool();
    const bool statsTopRight = property("vc_status_stats_top_right").toBool();
    if (showStats && !statsTopRight) {
        items << statsItems;
    }
    // Cache statistics are informational; refresh them at most every 250 ms
    // (see _sharedCacheStatsThrottle), not on every mouse move.
    constexpr qint64 kSharedCacheStatsRefreshMs = 250;
    if (!_sharedCacheStatsThrottle.isValid() ||
        _sharedCacheStatsThrottle.elapsed() >= kSharedCacheStatsRefreshMs) {
        _sharedCacheStatsThrottle.restart();
        QStringList freshItems;
        if (_state && _state->openedVolumes()) {
            const auto cache = _state->openedVolumes()->cacheStats();
            freshItems << QString("RAM %1/%2")
                .arg(formatByteSize(cache.bytes))
                .arg(formatByteSize(cache.capacity));
        }
        if (_loadRunning)
            freshItems << QStringLiteral("loading");
        if (!_loadError.isEmpty())
            freshItems << QStringLiteral("⚠ read failed: ") + _loadError;
        _cachedSharedCacheItems = std::move(freshItems);
    }
    sharedCacheItems = _cachedSharedCacheItems;

    auto surf = _surfWeak.lock();
    const auto& cursorPos =
        _lastCursorVolumePos ? _lastCursorVolumePos : _linkedCursorVolumePos;
    if (cursorPos && !property("vc_hide_status_position").toBool())
        items << formatWholeVolumePosition(*cursorPos);
    if (dynamic_cast<QuadSurface*>(surf.get())) {
        if (_zOff != 0.0f)
            items << QString("normal offset %1").arg(_zOff, 0, 'f', 1);
        if (_state && !property("vc_hide_status_poi").toBool()) {
            if (auto* poi = _state->poi("focus"))
                items << QString("POI %1").arg(formatVec3(poi->p));
        }
    } else if (property("vc_show_custom_normal_offset").toBool()) {
        const double offsetVx = property("vc_custom_normal_offset_vx").toDouble();
        if (offsetVx != 0.0)
            items << QString("normal offset %1").arg(offsetVx, 0, 'f', 1);
    }

    _statsBar->setItems(items);
    _statsBar->setVisible(!items.isEmpty());
    if (_statsBarRight) {
        const QStringList rightItems =
            (showStats && statsTopRight) ? statsItems : QStringList{};
        _statsBarRight->setItems(rightItems);
        _statsBarRight->setVisible(!rightItems.isEmpty());
        repositionStatsBarRight();
    }
    emit sharedCacheStatsChanged(sharedCacheItems);
}
