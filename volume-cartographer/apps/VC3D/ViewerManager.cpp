#include "ViewerManager.hpp"
#include "OpenDataSegmentCache.hpp"

#include "AxisAlignedSliceController.hpp"
#include "VCSettings.hpp"
#include "volume_viewers/VolumeViewerBase.hpp"
#include "volume_viewers/CChunkedVolumeViewer.hpp"
#include "volume_viewers/CVolumeViewerView.hpp"
#include "overlays/SegmentationOverlayController.hpp"
#include "overlays/PointsOverlayController.hpp"
#include "overlays/RawPointsOverlayController.hpp"
#include "overlays/PathsOverlayController.hpp"
#include "overlays/BBoxOverlayController.hpp"
#include "overlays/VectorOverlayController.hpp"
#include "overlays/VolumeOverlayController.hpp"
#include "overlays/InkDetectionOverlayController.hpp"
#include "segmentation/SegmentationModule.hpp"
#include "CState.hpp"
#include "vc/ui/VCCollection.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/AffineTransform.hpp"
#include "vc/core/util/Logging.hpp"

#include <QApplication>
#include <QCursor>
#include <QPointer>
#include <QMdiArea>
#include <QTimer>
#include <QThread>
#include <QMdiSubWindow>
#include <QSettings>
#include <QtConcurrent/QtConcurrent>
#include <QLoggingCategory>
#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <optional>
#include <string_view>
#include <unordered_set>
#include "utils/Json.hpp"
#include <opencv2/core.hpp>

#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/QuadSurface.hpp"

Q_LOGGING_CATEGORY(lcViewerManager, "vc.viewer.manager")

#define VC3D_DEBUG_QCINFO(category) if (!DebugLoggingEnabled()) {} else qCInfo(category)

namespace {

std::string coordinateSpaceTag(const VolumePkg& pkg, const std::string& volumeId)
{
    constexpr std::string_view prefix = "vc-open-data-coordinate-space:";
    for (const auto& tag : pkg.volumeTags(volumeId)) {
        if (tag.rfind(prefix, 0) == 0)
            return tag.substr(prefix.size());
    }
    return {};
}

QString compactViewerLabel(const std::string& surfaceName, const QString& title)
{
    if (surfaceName == "seg xz")
        return QStringLiteral("XZ");
    if (surfaceName == "seg yz")
        return QStringLiteral("YZ");
    if (surfaceName == "xy plane")
        return QStringLiteral("XY");
    if (title.startsWith(QStringLiteral("Surface ")))
        return title;
    return {};
}

bool isChunkedViewer(VolumeViewerBase* viewer)
{
    return viewer && qobject_cast<CChunkedVolumeViewer*>(viewer->asQObject());
}

bool isAnnotationViewer(VolumeViewerBase* viewer)
{
    return viewer &&
           viewer->asQObject() &&
           viewer->asQObject()->property("vc_viewer_role").toString() == QStringLiteral("annotation");
}

void centerViewerOnVolumePointForNavigation(VolumeViewerBase* viewer, const cv::Vec3f& position)
{
    if (!viewer) {
        return;
    }
    viewer->centerOnVolumePoint(position, !isChunkedViewer(viewer));
}

void centerViewerOnSurfacePointForNavigation(VolumeViewerBase* viewer, const cv::Vec2f& position)
{
    if (!viewer) {
        return;
    }
    viewer->centerOnSurfacePoint(position, !isChunkedViewer(viewer));
}

bool finiteVec3(const cv::Vec3f& v)
{
    return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]);
}

std::optional<cv::Vec3f> transformPoint(const cv::Vec3f& point, const cv::Matx44d& matrix)
{
    cv::Vec3d transformed;
    if (!vc::core::util::applyAffineTransform(cv::Vec3d(point), matrix, transformed)) {
        return std::nullopt;
    }
    const cv::Vec3f out(static_cast<float>(transformed[0]),
                        static_cast<float>(transformed[1]),
                        static_cast<float>(transformed[2]));
    return finiteVec3(out) ? std::optional<cv::Vec3f>(out) : std::nullopt;
}

cv::Vec3f clampToVolumeBounds(cv::Vec3f point, const std::shared_ptr<Volume>& volume)
{
    if (!volume) {
        return point;
    }
    const auto [w, h, d] = volume->shapeXyz();
    point[0] = std::clamp(point[0], 0.0f, static_cast<float>(std::max(1, w) - 1));
    point[1] = std::clamp(point[1], 0.0f, static_cast<float>(std::max(1, h) - 1));
    point[2] = std::clamp(point[2], 0.0f, static_cast<float>(std::max(1, d) - 1));
    return point;
}

// Uniform distance scale of an affine transform, or nullopt when the linear
// part is too anisotropic for a single scalar to be meaningful.
std::optional<double> relativeAffineDistanceScale(const cv::Matx44d& matrix)
{
    cv::Mat linear(3, 3, CV_64F);
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            linear.at<double>(row, col) = matrix(row, col);
        }
    }

    cv::SVD svd(linear, cv::SVD::NO_UV);
    if (svd.w.rows < 3) {
        return std::nullopt;
    }
    const double s0 = svd.w.at<double>(0, 0);
    const double s1 = svd.w.at<double>(1, 0);
    const double s2 = svd.w.at<double>(2, 0);
    if (!(std::isfinite(s0) && std::isfinite(s1) && std::isfinite(s2)) ||
        s0 <= 0.0 || s1 <= 0.0 || s2 <= 0.0) {
        return std::nullopt;
    }

    const double mean = (s0 + s1 + s2) / 3.0;
    const double maxDeviation =
        std::max({std::abs(s0 - mean), std::abs(s1 - mean), std::abs(s2 - mean)});
    const double relativeDeviation = maxDeviation / mean;
    if (!std::isfinite(relativeDeviation) || relativeDeviation > 0.02) {
        return std::nullopt;
    }
    return mean;
}

std::vector<ViewerManager*>& managerRegistry()
{
    static std::vector<ViewerManager*> registry;
    return registry;
}

}

ViewerManager::ViewerManager(CState* state,
                             VCCollection* points,
                             QObject* parent)
    : QObject(parent)
    , _state(state)
    , _points(points)
{
    managerRegistry().push_back(this);
    using namespace vc3d::settings;
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    const int savedOpacityPercent = settings.value(viewer::INTERSECTION_OPACITY, viewer::INTERSECTION_OPACITY_DEFAULT).toInt();
    const float normalized = static_cast<float>(savedOpacityPercent) / 100.0f;
    _intersectionOpacity = std::clamp(normalized, 0.0f, 1.0f);

    float storedBaseLow = settings.value(viewer::BASE_WINDOW_LOW, viewer::BASE_WINDOW_LOW_DEFAULT).toFloat();
    float storedBaseHigh = settings.value(viewer::BASE_WINDOW_HIGH, viewer::BASE_WINDOW_HIGH_DEFAULT).toFloat();
    // The window is a fixed 0-255 scale. Values persisted on a different scale (e.g. a
    // legacy 16-bit build) would otherwise clamp to (255, 255) and slam both slider
    // handles to the far right. Fall back to defaults if the stored pair is out of range
    // or collapsed so the control is usable again.
    if (storedBaseLow < 0.0f || storedBaseHigh > 255.0f || (storedBaseHigh - storedBaseLow) < 1.0f) {
        storedBaseLow = viewer::BASE_WINDOW_LOW_DEFAULT;
        storedBaseHigh = viewer::BASE_WINDOW_HIGH_DEFAULT;
    }
    _volumeWindowLow = std::clamp(storedBaseLow, 0.0f, 255.0f);
    const float minHigh = std::min(_volumeWindowLow + 1.0f, 255.0f);
    _volumeWindowHigh = std::clamp(storedBaseHigh, minHigh, 255.0f);

    const double storedZScroll = settings.value(viewer::ZSCROLL_SENSITIVITY,
                                                 viewer::ZSCROLL_SENSITIVITY_DEFAULT).toDouble();
    _zScrollSensitivity = std::clamp(storedZScroll, 0.1, 100.0);

    _surfacePatchSamplingStride = viewer::INTERSECTION_SAMPLING_STRIDE_DEFAULT;
    const float storedThickness = settings.value(viewer::INTERSECTION_THICKNESS, viewer::INTERSECTION_THICKNESS_DEFAULT).toFloat();
    _intersectionThickness = std::max(0.0f, storedThickness);
    _intersectionMaxSurfaces = viewer::INTERSECTION_MAX_SURFACES_DEFAULT;

    _surfacePatchIndexWatcher =
        new QFutureWatcher<std::shared_ptr<SurfacePatchIndex>>(this);
    connect(_surfacePatchIndexWatcher,
            &QFutureWatcher<std::shared_ptr<SurfacePatchIndex>>::finished,
            this,
            &ViewerManager::handleSurfacePatchIndexPrimeFinished);

    _surfacePatchIndexTaskWatcher =
        new QFutureWatcher<SurfacePatchIndexTaskResult>(this);
    connect(_surfacePatchIndexTaskWatcher,
            &QFutureWatcher<SurfacePatchIndexTaskResult>::finished,
            this,
            &ViewerManager::handleSurfacePatchIndexTaskFinished);

    if (_state) {
        connect(_state,
                &CState::surfaceChanged,
                this,
                &ViewerManager::handleSurfaceChanged);
        connect(_state,
                &CState::surfaceWillBeDeleted,
                this,
                &ViewerManager::handleSurfaceWillBeDeleted);
        connect(_state,
                &CState::poiChanged,
                this,
                &ViewerManager::handleFocusPoiChanged);
    }

    // The single maintenance clock for the whole app: ~60Hz, free-running. Render
    // requests submit immediately; this services deferred intersections and status.
    _globalClock = new QTimer(this);
    _globalClock->setInterval(16);
    connect(_globalClock, &QTimer::timeout, this, &ViewerManager::onGlobalTick);
    _globalClock->start();
}

ViewerManager::~ViewerManager()
{
    auto& registry = managerRegistry();
    registry.erase(std::remove(registry.begin(), registry.end(), this), registry.end());
}

const std::vector<ViewerManager*>& ViewerManager::allManagers()
{
    return managerRegistry();
}

void ViewerManager::onGlobalTick()
{
    for (auto* v : _baseViewers) {
        if (v)
            v->serviceRenderTick();
    }
}

VolumeViewerBase* ViewerManager::createViewer(const std::string& surfaceName,
                                              const QString& title,
                                              QMdiArea* mdiArea,
                                              ViewerRole role)
{
    if (!mdiArea || !_state) {
        return nullptr;
    }

    auto* chunkedViewer = new CChunkedVolumeViewer(_state, this, mdiArea);
    chunkedViewer->setProperty("vc_viewer_label", compactViewerLabel(surfaceName, title));
    QWidget* widget = chunkedViewer;

    auto* win = mdiArea->addSubWindow(widget);
    win->setWindowTitle(title);
    win->setWindowFlags(Qt::SubWindow | Qt::FramelessWindowHint);
    win->setAttribute(Qt::WA_DeleteOnClose);
    win->installEventFilter(widget);

    return initializeChunkedViewer(chunkedViewer, surfaceName, role);
}

VolumeViewerBase* ViewerManager::createViewerInWidget(const std::string& surfaceName,
                                                      QWidget* parent,
                                                      ViewerRole role)
{
    if (!parent || !_state) {
        return nullptr;
    }

    auto* chunkedViewer = new CChunkedVolumeViewer(_state, this, parent);
    return initializeChunkedViewer(chunkedViewer, surfaceName, role);
}

VolumeViewerBase* ViewerManager::initializeChunkedViewer(CChunkedVolumeViewer* chunkedViewer,
                                                         const std::string& surfaceName,
                                                         ViewerRole role)
{
    if (!chunkedViewer || !_state) {
        return nullptr;
    }

    auto* widget = chunkedViewer;
    VolumeViewerBase* baseViewer = chunkedViewer;
    chunkedViewer->setProperty("vc_viewer_role",
                               role == ViewerRole::Annotation
                                   ? QStringLiteral("annotation")
                                   : QStringLiteral("standard"));
    chunkedViewer->setPointCollection(_points);

    if (_state) {
        connect(_state, &CState::surfaceChanged, chunkedViewer, &CChunkedVolumeViewer::onSurfaceChanged);
        connect(_state, &CState::surfaceWillBeDeleted, chunkedViewer, &CChunkedVolumeViewer::onSurfaceWillBeDeleted);
        connect(_state, &CState::poiChanged, chunkedViewer, &CChunkedVolumeViewer::onPOIChanged);
        connect(_state, &CState::volumeChanged, chunkedViewer, &CChunkedVolumeViewer::OnVolumeChanged);
        connect(_state, &CState::volumeClosing, chunkedViewer, &CChunkedVolumeViewer::onVolumeClosing);
    }

    // Restore persisted viewer preferences
    {
        using namespace vc3d::settings;
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        bool showHints = settings.value(viewer::SHOW_DIRECTION_HINTS, viewer::SHOW_DIRECTION_HINTS_DEFAULT).toBool();
        baseViewer->setShowDirectionHints(showHints);
        bool showNormals = settings.value(viewer::SHOW_SURFACE_NORMALS, viewer::SHOW_SURFACE_NORMALS_DEFAULT).toBool();
        baseViewer->setShowSurfaceNormals(showNormals);
    }

    {
        using namespace vc3d::settings;
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        bool resetView = settings.value(viewer::RESET_VIEW_ON_SURFACE_CHANGE, viewer::RESET_VIEW_ON_SURFACE_CHANGE_DEFAULT).toBool();
        baseViewer->setResetViewOnSurfaceChange(resetView);
        _resetDefaults[baseViewer] = resetView;
        bool showPlaneIntersectionLines = settings.value(viewer::SHOW_PLANE_INTERSECTION_LINES,
                                                         viewer::SHOW_PLANE_INTERSECTION_LINES_DEFAULT).toBool();
        baseViewer->setPlaneIntersectionLinesVisible(showPlaneIntersectionLines);
    }

    baseViewer->setSurface(surfaceName);
    if (_state->currentVolume()) {
        chunkedViewer->OnVolumeChanged(_state->currentVolume());
    }
    baseViewer->setSegmentationEditActive(_segmentationEditActive);
    baseViewer->setSegmentationCursorMirroring(_mirrorCursorToSegmentation);

    if (role != ViewerRole::Annotation) {
        connect(chunkedViewer, &CChunkedVolumeViewer::sendVolumeClicked,
                this, &ViewerManager::handleVolumeClicked);
    }
    connect(chunkedViewer, &CChunkedVolumeViewer::sharedCacheStatsChanged,
            this, &ViewerManager::sharedCacheStatsChanged);

    if (auto* graphicsView = chunkedViewer->graphicsView()) {
        auto markActiveViewer = [this, baseViewer]() { _activeViewer = baseViewer; };
        connect(graphicsView, &CVolumeViewerView::sendMousePress, this, markActiveViewer);
        connect(graphicsView, &CVolumeViewerView::sendMouseDoubleClick, this, markActiveViewer);
        connect(graphicsView, &CVolumeViewerView::sendZoom, this, markActiveViewer);
        connect(graphicsView, &CVolumeViewerView::sendCursorMove, this, markActiveViewer);
    }

    _baseViewers.push_back(baseViewer);

    // Clean up when viewer is destroyed without an earlier close event.
    connect(widget, &QObject::destroyed, this, [this, baseViewer]() {
        unregisterViewer(baseViewer);
    });

    for (auto* overlay : _allOverlays) {
        overlay->attachViewer(baseViewer);
    }

    baseViewer->setIntersectionOpacity(_intersectionOpacity);
    baseViewer->setIntersectionThickness(_intersectionThickness);
    baseViewer->setSurfacePatchSamplingStride(_surfacePatchSamplingStride);
    baseViewer->setVolumeWindow(_volumeWindowLow, _volumeWindowHigh);
    baseViewer->setOverlayVolume(_overlayVolume);
    baseViewer->setOverlayOpacity(_overlayOpacity);
    baseViewer->setOverlayColormap(_overlayColormapId);
    baseViewer->setOverlaySamplingMethod(_overlaySamplingMethod);
    baseViewer->setOverlayWindow(_overlayWindowLow, _overlayWindowHigh);
    baseViewer->setOverlayMaxDisplayedResolution(_overlayMaxDisplayedResolution);
    baseViewer->setOverlayComposite(_overlayComposite);

    if (_segmentationModule && role != ViewerRole::Annotation) {
        _segmentationModule->attachViewer(baseViewer);
    }
    emit baseViewerCreated(baseViewer);
    return baseViewer;
}

void ViewerManager::unregisterViewer(VolumeViewerBase* viewer)
{
    if (!viewer) {
        return;
    }

    const auto viewerIt = std::find(_baseViewers.begin(), _baseViewers.end(), viewer);
    const bool knownViewer = viewerIt != _baseViewers.end() ||
                             _resetDefaults.find(viewer) != _resetDefaults.end();
    if (!knownViewer) {
        return;
    }

    emit baseViewerClosing(viewer);
    if (_segmentationModule) {
        _segmentationModule->detachViewer(viewer);
    }
    if (_activeViewer == viewer) {
        _activeViewer = nullptr;
    }
    _resetDefaults.erase(viewer);
    _baseViewers.erase(std::remove(_baseViewers.begin(), _baseViewers.end(), viewer), _baseViewers.end());
}

void ViewerManager::handleFocusPoiChanged(std::string name, POI* poi)
{
    if (name != "focus" || !poi) {
        return;
    }
    if (_slices) {
        _slices->applyOrientation();
    }
    if (!poi->suppressViewerRecenter) {
        const cv::Vec3f focusPosition = poi->p;
        QTimer::singleShot(0, this, [this, focusPosition]() {
            recenterPlaneViewersOn(focusPosition);
        });
    }
}

bool ViewerManager::resetFocusForVolumeChange(bool resetToCenter,
                                              const std::optional<cv::Vec3f>& overridePoint,
                                              const std::optional<cv::Vec3f>& overrideNormal)
{
    if (!_state || !_state->currentVolume()) {
        return false;
    }

    const auto [w, h, d] = _state->currentVolume()->shapeXyz();
    const cv::Vec3f hi(static_cast<float>(std::max(1, w) - 1),
                       static_cast<float>(std::max(1, h) - 1),
                       static_cast<float>(std::max(1, d) - 1));
    const auto clampToBounds = [&hi](cv::Vec3f p) {
        for (int axis = 0; axis < 3; ++axis) {
            p[axis] = std::clamp(p[axis], 0.0f, hi[axis]);
        }
        return p;
    };

    POI* poi = _state->poi("focus");
    const bool createdPoi = (poi == nullptr);
    if (!poi) {
        poi = new POI;
        poi->n = cv::Vec3f(0, 0, 1);
    }

    if (overridePoint) {
        poi->p = clampToBounds(*overridePoint);
        if (overrideNormal) {
            poi->n = *overrideNormal;
        }
    } else if (createdPoi || resetToCenter) {
        poi->p = hi * 0.5f;
    } else {
        poi->p = clampToBounds(poi->p);
    }
    poi->surfacePtr.reset();

    _state->setPOI("focus", poi);
    return true;
}

bool ViewerManager::centerFocusAt(const cv::Vec3f& position, const cv::Vec3f& normal, const std::string& sourceId)
{
    if (!_state) {
        return false;
    }

    POI* focus = _state->poi("focus");
    if (!focus) {
        focus = new POI;
    }

    focus->p = position;
    if (cv::norm(normal) > 0.0) {
        focus->n = normal;
    }
    if (!sourceId.empty()) {
        focus->surfaceId = sourceId;
    } else if (focus->surfaceId.empty()) {
        focus->surfaceId = "segmentation";
    }
    focus->surfacePtr.reset();

    focus->suppressTransientPlaneIntersections = true;
    _state->setPOI("focus", focus);
    recenterSegmentationViewerNear(position);

    // Get surface for orientation - look up by ID
    Surface* orientationSource = _state->surfaceRaw(focus->surfaceId);
    if (!orientationSource) {
        orientationSource = _state->surfaceRaw("segmentation");
    }
    if (_slices) {
        _slices->applyOrientation(orientationSource);
    }

    emit focusCenteredByUser(position);
    return true;
}

bool ViewerManager::centerFocusOnCursor()
{
    if (!_state) {
        return false;
    }

    const QPoint globalPos = QCursor::pos();
    auto tryCenterFromViewer = [&](VolumeViewerBase* viewer) -> bool {
        if (!viewer) {
            return false;
        }

        auto* viewerWidget = qobject_cast<QWidget*>(viewer->asQObject());
        if (viewerWidget && !viewerWidget->isVisible()) {
            return false;
        }

        auto* gv = viewer->graphicsView();
        auto* viewport = gv ? gv->viewport() : nullptr;
        if (!viewport) {
            return false;
        }

        const QPoint viewportPos = viewport->mapFromGlobal(globalPos);
        if (!viewport->rect().contains(viewportPos)) {
            return false;
        }

        const QPointF scenePos = gv->mapToScene(viewportPos);
        const cv::Vec3f p = viewer->sceneToVolume(scenePos);
        if (!finiteVec3(p)) {
            return false;
        }
        cv::Vec3f n(0, 0, 1);
        if (auto* plane = dynamic_cast<PlaneSurface*>(viewer->currentSurface())) {
            n = plane->normal(cv::Vec3f(0, 0, 0), {});
        }

        return centerFocusAt(p, n, viewer->surfName());
    };

    // Prefer the viewer actually under the mouse cursor: the active
    // window/subwindow can lag behind the hovered viewer, which would make the
    // focus jump use the wrong scene transform.
    if (QWidget* hoveredWidget = QApplication::widgetAt(globalPos)) {
        for (QWidget* widget = hoveredWidget; widget; widget = widget->parentWidget()) {
            if (auto* viewer = qobject_cast<CChunkedVolumeViewer*>(widget)) {
                if (tryCenterFromViewer(viewer)) {
                    return true;
                }
                break;
            }
        }
    }

    for (auto* viewer : _baseViewers) {
        if (tryCenterFromViewer(viewer)) {
            return true;
        }
    }

    // Fall back to the stored cursor POI when the mouse isn't over any viewport.
    POI* cursor = _state->poi("cursor");
    if (!cursor) {
        return false;
    }
    return centerFocusAt(cursor->p, cursor->n, cursor->surfaceId);
}

void ViewerManager::handleVolumeClicked(cv::Vec3f volLoc, cv::Vec3f normal, Surface* surf,
                                        Qt::MouseButton button, Qt::KeyboardModifiers modifiers)
{
    if (_volumeClickInterceptor && _volumeClickInterceptor(volLoc, normal, surf, button, modifiers)) {
        return;
    }

    if (modifiers & Qt::ShiftModifier) {
        // Reserved for point tools.
        return;
    }
    if (modifiers & Qt::ControlModifier) {
        std::string surfId;
        if (_state && surf) {
            surfId = _state->findSurfaceId(surf);
        }
        centerFocusAt(volLoc, normal, surfId);
    }
}

VolumeViewerBase* ViewerManager::activeViewer() const
{
    if (!_activeViewer) {
        return nullptr;
    }
    if (auto* activeObject = _activeViewer->asQObject()) {
        if (!activeObject->parent()) {
            return nullptr;
        }
    }
    return _activeViewer;
}

bool ViewerManager::recenterViewersOnCurrentFocus()
{
    if (!_state) {
        return false;
    }

    POI* focus = _state->poi("focus");
    if (!focus) {
        return false;
    }

    const cv::Vec3f position = focus->p;
    forEachBaseViewer([&position](VolumeViewerBase* viewer) {
        if (viewer && !isAnnotationViewer(viewer)) {
            centerViewerOnVolumePointForNavigation(viewer, position);
        }
    });

    return true;
}

void ViewerManager::recenterPlaneViewersOn(const cv::Vec3f& position)
{
    forEachBaseViewer([&position](VolumeViewerBase* viewer) {
        if (!viewer || isAnnotationViewer(viewer)) {
            return;
        }

        const std::string name = viewer->surfName();
        if (name == "xy plane" || name == "seg xz" || name == "seg yz") {
            centerViewerOnVolumePointForNavigation(viewer, position);
        }
    });
}

void ViewerManager::recenterSegmentationViewerNear(const cv::Vec3f& position)
{
    static constexpr float kMaxDistanceVoxels = 100.0f;

    auto* viewer = segmentationViewer();
    if (!viewer) {
        return;
    }

    auto activeSurface = _segmentationModule ? _segmentationModule->activeBaseSurfaceShared() : nullptr;
    if (!activeSurface) {
        activeSurface = std::dynamic_pointer_cast<QuadSurface>(_state ? _state->surface("segmentation") : nullptr);
    }
    if (!activeSurface) {
        return;
    }

    auto* patchIndex = surfacePatchIndex();
    if (!patchIndex || !patchIndex->containsSurface(activeSurface)) {
        return;
    }

    SurfacePatchIndex::PointQuery query;
    query.worldPoint = position;
    query.tolerance = kMaxDistanceVoxels;
    query.surfaces.only = activeSurface;
    auto hit = patchIndex->locate(query);
    if (hit && hit->distance <= kMaxDistanceVoxels) {
        const cv::Vec3f loc = activeSurface->loc(hit->ptr);
        centerViewerOnSurfacePointForNavigation(viewer, {loc[0], loc[1]});
    }
}

VolumeViewerBase* ViewerManager::segmentationViewer() const
{
    for (auto* viewer : _baseViewers) {
        if (viewer && viewer->surfName() == "segmentation") {
            return viewer;
        }
    }
    return nullptr;
}

void ViewerManager::setShowDirectionHints(bool show)
{
    forEachBaseViewer([show](VolumeViewerBase* viewer) {
        if (viewer) {
            viewer->setShowDirectionHints(show);
        }
    });
}

void ViewerManager::setShowSurfaceNormals(bool show)
{
    forEachBaseViewer([show](VolumeViewerBase* viewer) {
        if (viewer) {
            viewer->setShowSurfaceNormals(show);
        }
    });
}

void ViewerManager::setPlaneIntersectionLinesVisible(bool visible)
{
    forEachBaseViewer([visible](VolumeViewerBase* viewer) {
        if (viewer) {
            viewer->setPlaneIntersectionLinesVisible(visible);
        }
    });
}

void ViewerManager::setSurfaceOverlaysEnabled(bool enabled)
{
    forEachBaseViewer([enabled](VolumeViewerBase* viewer) {
        if (viewer) {
            viewer->setSurfaceOverlayEnabled(enabled);
        }
    });
}

void ViewerManager::setSurfaceOverlapThreshold(float threshold)
{
    forEachBaseViewer([threshold](VolumeViewerBase* viewer) {
        if (viewer) {
            viewer->setSurfaceOverlapThreshold(threshold);
        }
    });
}

void ViewerManager::setSurfaceOverlays(const std::map<std::string, cv::Vec3b>& overlays)
{
    forEachBaseViewer([&overlays](VolumeViewerBase* viewer) {
        if (viewer) {
            viewer->setSurfaceOverlays(overlays);
        }
    });
}

void ViewerManager::setResetViewOnSurfaceChangeDefault(bool enabled)
{
    const bool editingActive = _segmentationModule && _segmentationModule->editingEnabled();
    forEachBaseViewer([this, enabled, editingActive](VolumeViewerBase* viewer) {
        if (!viewer) {
            return;
        }
        setResetDefaultFor(viewer, enabled);
        if (editingActive && viewer->surfName() == "segmentation") {
            viewer->setResetViewOnSurfaceChange(false);
            return;
        }
        viewer->setResetViewOnSurfaceChange(enabled);
    });
}

void ViewerManager::setSegmentationResetViewSuppressed(bool suppressed)
{
    forEachBaseViewer([this, suppressed](VolumeViewerBase* viewer) {
        if (!viewer || viewer->surfName() != "segmentation") {
            return;
        }
        viewer->setResetViewOnSurfaceChange(suppressed ? false : resetDefaultFor(viewer));
    });
}

struct ViewerManager::ViewerNavigationSnapshot {
    struct Entry {
        QPointer<CChunkedVolumeViewer> viewer;
        CChunkedVolumeViewer::CameraState camera;
        cv::Vec3f center{0, 0, 0};
        bool hasCenter{false};
    };
    std::vector<Entry> entries;
};

std::shared_ptr<ViewerManager::ViewerNavigationSnapshot> ViewerManager::captureNavigation() const
{
    auto snapshot = std::make_shared<ViewerNavigationSnapshot>();
    forEachBaseViewer([&snapshot](VolumeViewerBase* baseViewer) {
        auto* viewer = dynamic_cast<CChunkedVolumeViewer*>(baseViewer);
        if (!viewer || !viewer->graphicsView()) {
            return;
        }
        ViewerNavigationSnapshot::Entry captured;
        captured.viewer = viewer;
        captured.camera = viewer->cameraState();
        const QSize viewportSize = viewer->graphicsView()->viewport()->size();
        const QPointF centerScene(
            static_cast<qreal>(std::max(1, viewportSize.width())) * 0.5,
            static_cast<qreal>(std::max(1, viewportSize.height())) * 0.5);
        if (const auto sample = viewer->sampleSceneVolume(centerScene)) {
            if (finiteVec3(sample->position)) {
                captured.center = sample->position;
                captured.hasCenter = true;
            }
        }
        snapshot->entries.push_back(std::move(captured));
    });
    return snapshot;
}

void ViewerManager::restoreNavigation(const std::shared_ptr<ViewerNavigationSnapshot>& snapshot,
                                      const cv::Matx44d& transform)
{
    if (!snapshot || !_state || !_state->currentVolume()) {
        return;
    }

    const auto navigationScale = relativeAffineDistanceScale(transform);
    for (const auto& captured : snapshot->entries) {
        CChunkedVolumeViewer* viewer = captured.viewer.data();
        if (!viewer || viewer->currentVolume() != _state->currentVolume()) {
            continue;
        }

        std::optional<cv::Vec3f> transformedCenter;
        if (captured.hasCenter) {
            transformedCenter = transformPoint(captured.center, transform);
        }
        if (!transformedCenter) {
            continue;
        }

        viewer->centerOnVolumePoint(
            clampToVolumeBounds(*transformedCenter, _state->currentVolume()),
            false);
        auto camera = viewer->cameraState();
        camera.scale = captured.camera.scale;
        if (navigationScale) {
            camera.scale = CChunkedVolumeViewer::clampCameraScale(
                static_cast<float>(static_cast<double>(captured.camera.scale) / *navigationScale));
        }
        camera.zOffset = navigationScale
            ? static_cast<float>(static_cast<double>(captured.camera.zOffset) *
                                 *navigationScale)
            : captured.camera.zOffset;
        camera.zOffsetWorldDir = captured.camera.zOffsetWorldDir;
        if (cv::norm(captured.camera.zOffsetWorldDir) > 0.0f) {
            const auto direction = vc::core::util::transformNormal(
                captured.camera.zOffsetWorldDir, transform);
            if (finiteVec3(direction)) {
                camera.zOffsetWorldDir = direction;
            }
        }
        viewer->applyCameraState(camera, false);
    }
}

void ViewerManager::switchVolume(std::shared_ptr<Volume> volume,
                                 const std::optional<cv::Matx44d>& navigationTransform)
{
    if (!_state) {
        return;
    }
    const bool hadVolume = static_cast<bool>(_state->currentVolume());

    std::optional<cv::Vec3f> transformedFocusPoint;
    std::optional<cv::Vec3f> transformedFocusNormal;
    std::shared_ptr<ViewerNavigationSnapshot> snapshot;
    if (navigationTransform) {
        if (POI* focus = _state->poi("focus"); focus && finiteVec3(focus->p)) {
            transformedFocusPoint = transformPoint(focus->p, *navigationTransform);
            if (cv::norm(focus->n) > 0.0f) {
                const cv::Vec3f normal =
                    vc::core::util::transformNormal(focus->n, *navigationTransform);
                if (finiteVec3(normal)) {
                    transformedFocusNormal = normal;
                }
            }
        }
        snapshot = captureNavigation();
    }

    // CState handles cache budget and volume ID resolution, and emits volumeChanged
    _state->setCurrentVolume(std::move(volume));

    resetFocusForVolumeChange(!hadVolume, transformedFocusPoint, transformedFocusNormal);

    if (snapshot && navigationTransform) {
        restoreNavigation(snapshot, *navigationTransform);
    }
}

void ViewerManager::registerOverlay(ViewerOverlayControllerBase* overlay)
{
    if (!overlay) {
        return;
    }
    if (std::find(_allOverlays.begin(), _allOverlays.end(), overlay) != _allOverlays.end()) {
        return;
    }
    _allOverlays.push_back(overlay);
    overlay->bindToViewerManager(this);
}

void ViewerManager::setSegmentationOverlay(SegmentationOverlayController* overlay)
{
    _segmentationOverlay = overlay;
    registerOverlay(overlay);
}

void ViewerManager::setSegmentationEditActive(bool active)
{
    _segmentationEditActive = active;
    forEachBaseViewer([active](VolumeViewerBase* v) { v->setSegmentationEditActive(active); });
}

void ViewerManager::setSegmentationModule(SegmentationModule* module)
{
    _segmentationModule = module;
    if (!_segmentationModule) {
        return;
    }

    forEachBaseViewer([this](VolumeViewerBase* v) { _segmentationModule->attachViewer(v); });
}

void ViewerManager::setPointsOverlay(PointsOverlayController* overlay)
{
    _pointsOverlay = overlay;
    registerOverlay(overlay);
}

void ViewerManager::setRawPointsOverlay(RawPointsOverlayController* overlay)
{
    _rawPointsOverlay = overlay;
    registerOverlay(overlay);
}

void ViewerManager::setPathsOverlay(PathsOverlayController* overlay)
{
    _pathsOverlay = overlay;
    registerOverlay(overlay);
}

void ViewerManager::setBBoxOverlay(BBoxOverlayController* overlay)
{
    _bboxOverlay = overlay;
    registerOverlay(overlay);
}

void ViewerManager::setVectorOverlay(VectorOverlayController* overlay)
{
    _vectorOverlay = overlay;
    registerOverlay(overlay);
}

void ViewerManager::setVolumeOverlay(VolumeOverlayController* overlay)
{
    _volumeOverlay = overlay;
    if (_volumeOverlay) {
        _volumeOverlay->syncWindowFromManager(_overlayWindowLow, _overlayWindowHigh);
    }
}

void ViewerManager::setInkDetectionOverlay(InkDetectionOverlayController* overlay)
{
    _inkDetectionOverlay = overlay;
    registerOverlay(overlay);
}

void ViewerManager::setIntersectionOpacity(float opacity)
{
    _intersectionOpacity = std::clamp(opacity, 0.0f, 1.0f);

    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue(vc3d::settings::viewer::INTERSECTION_OPACITY,
                      static_cast<int>(std::lround(_intersectionOpacity * 100.0f)));

    forEachBaseViewer([this](VolumeViewerBase* v) { v->setIntersectionOpacity(_intersectionOpacity); });
}

void ViewerManager::setIntersectionThickness(float thickness)
{
    const float clamped = std::clamp(thickness, 0.0f, 100.0f);
    if (std::abs(clamped - _intersectionThickness) < 1e-6f) {
        return;
    }
    _intersectionThickness = clamped;

    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue(vc3d::settings::viewer::INTERSECTION_THICKNESS, _intersectionThickness);

    forEachBaseViewer([this](VolumeViewerBase* v) { v->setIntersectionThickness(_intersectionThickness); });
}

void ViewerManager::setHighlightedSurfaceIds(const std::vector<std::string>& ids)
{
    forEachBaseViewer([&ids](VolumeViewerBase* v) { v->setHighlightedSurfaceIds(ids); });
}

void ViewerManager::setOverlayVolume(std::shared_ptr<Volume> volume, const std::string& volumeId)
{
    if (volume && _state && _state->vpkg()) {
        const auto baseSpace = coordinateSpaceTag(
            *_state->vpkg(), _state->currentVolumeId());
        const auto overlaySpace = coordinateSpaceTag(*_state->vpkg(), volumeId);
        if ((!baseSpace.empty() || !overlaySpace.empty()) &&
            (baseSpace.empty() || baseSpace != overlaySpace)) {
            Logger()->warn(
                "Rejected volume overlay '{}' because its explicit coordinate space does not match '{}'.",
                volumeId, _state->currentVolumeId());
            volume.reset();
        }
    }
    _overlayVolume = std::move(volume);
    _overlayVolumeId = _overlayVolume ? volumeId : std::string{};
    forEachBaseViewer([this](VolumeViewerBase* v) { v->setOverlayVolume(_overlayVolume); });

    emit overlayVolumeAvailabilityChanged(static_cast<bool>(_overlayVolume));
}

std::shared_ptr<Volume> ViewerManager::currentVolume() const
{
    return _state ? _state->currentVolume() : nullptr;
}

std::string ViewerManager::currentVolumeId() const
{
    return _state ? _state->currentVolumeId() : std::string{};
}

void ViewerManager::setOverlayOpacity(float opacity)
{
    _overlayOpacity = std::clamp(opacity, 0.0f, 1.0f);
    forEachBaseViewer([this](VolumeViewerBase* v) { v->setOverlayOpacity(_overlayOpacity); });
}

void ViewerManager::setOverlayColormap(const std::string& colormapId)
{
    _overlayColormapId = colormapId;
    forEachBaseViewer([this](VolumeViewerBase* v) { v->setOverlayColormap(_overlayColormapId); });
}

void ViewerManager::setOverlaySamplingMethod(vc::Sampling method)
{
    _overlaySamplingMethod = method == vc::Sampling::Trilinear
        ? vc::Sampling::Trilinear
        : vc::Sampling::Nearest;
    forEachBaseViewer([this](VolumeViewerBase* v) {
        v->setOverlaySamplingMethod(_overlaySamplingMethod);
    });
}

void ViewerManager::setOverlayThreshold(float threshold)
{
    setOverlayWindow(std::max(threshold, 0.0f), _overlayWindowHigh);
}

void ViewerManager::setOverlayWindow(float low, float high)
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

    if (_volumeOverlay) {
        _volumeOverlay->syncWindowFromManager(_overlayWindowLow, _overlayWindowHigh);
    }

    forEachBaseViewer([this](VolumeViewerBase* v) { v->setOverlayWindow(_overlayWindowLow, _overlayWindowHigh); });

    emit overlayWindowChanged(_overlayWindowLow, _overlayWindowHigh);
}

void ViewerManager::setOverlayMaxDisplayedResolution(int level)
{
    _overlayMaxDisplayedResolution = std::clamp(level, 0, 5);
    forEachBaseViewer([this](VolumeViewerBase* v) {
        v->setOverlayMaxDisplayedResolution(_overlayMaxDisplayedResolution);
    });
}

void ViewerManager::setOverlayComposite(const OverlayCompositeSettings& settings)
{
    OverlayCompositeSettings sanitized = settings;
    if (sanitized.method != "max" && sanitized.method != "mean" && sanitized.method != "min") {
        sanitized.method = "max";
    }
    sanitized.layersFront = std::clamp(sanitized.layersFront, 0, 64);
    sanitized.layersBehind = std::clamp(sanitized.layersBehind, 0, 64);

    _overlayComposite = sanitized;
    forEachBaseViewer([this](VolumeViewerBase* v) {
        v->setOverlayComposite(_overlayComposite);
    });
}

void ViewerManager::setVolumeWindow(float low, float high)
{
    constexpr float kMaxValue = 255.0f;
    const float clampedLow = std::clamp(low, 0.0f, kMaxValue);
    float clampedHigh = std::clamp(high, 0.0f, kMaxValue);
    if (clampedHigh <= clampedLow) {
        clampedHigh = std::min(kMaxValue, clampedLow + 1.0f);
    }

    const bool unchanged = std::abs(clampedLow - _volumeWindowLow) < 1e-6f &&
                           std::abs(clampedHigh - _volumeWindowHigh) < 1e-6f;
    if (unchanged) {
        return;
    }

    _volumeWindowLow = clampedLow;
    _volumeWindowHigh = clampedHigh;

    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue(vc3d::settings::viewer::BASE_WINDOW_LOW, _volumeWindowLow);
    settings.setValue(vc3d::settings::viewer::BASE_WINDOW_HIGH, _volumeWindowHigh);

    forEachBaseViewer([this](VolumeViewerBase* v) { v->setVolumeWindow(_volumeWindowLow, _volumeWindowHigh); });

    emit volumeWindowChanged(_volumeWindowLow, _volumeWindowHigh);
}

void ViewerManager::setSurfacePatchSamplingStride(int stride, bool userInitiated)
{
    stride = std::max(1, stride);
    if (_surfacePatchSamplingStride == stride) {
        return;
    }
    _surfacePatchSamplingStride = stride;

    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    if (userInitiated) {
        settings.setValue(vc3d::settings::viewer::INTERSECTION_SAMPLING_STRIDE, _surfacePatchSamplingStride);
        settings.setValue(vc3d::settings::viewer::INTERSECTION_SAMPLING_STRIDE_USER_SET, true);
    }

    if (_surfacePatchIndex.setSamplingStride(_surfacePatchSamplingStride)) {
        _surfacePatchIndexNeedsRebuild = true;
        _indexedSurfaceIds.clear();
        // Index was cleared — remove stale intersection lines immediately.
        // New lines will appear once the async rebuild completes.
        forEachBaseViewer([](VolumeViewerBase* v) { v->invalidateIntersect(); });
    }

    forEachBaseViewer([this](VolumeViewerBase* v) { v->setSurfacePatchSamplingStride(_surfacePatchSamplingStride); });

    emit samplingStrideChanged(_surfacePatchSamplingStride);
}

void ViewerManager::setIntersectionMaxSurfaces(int limit)
{
    limit = std::max(0, limit);
    if (_intersectionMaxSurfaces == limit) {
        return;
    }
    _intersectionMaxSurfaces = limit;

    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue(vc3d::settings::viewer::INTERSECTION_MAX_SURFACES, limit);
}

SurfacePatchIndex* ViewerManager::surfacePatchIndex()
{
    rebuildSurfacePatchIndexIfNeeded();
    if (_surfacePatchIndex.empty()) {
        return nullptr;
    }
    return &_surfacePatchIndex;
}

SurfacePatchIndex* ViewerManager::surfacePatchIndexIfReady()
{
    if (_surfacePatchIndex.empty()) {
        return nullptr;
    }
    return &_surfacePatchIndex;
}

SurfacePatchIndex* ViewerManager::activeSegmentationEditSurfacePatchIndex() const
{
    return _segmentationModule ? _segmentationModule->activeEditSurfacePatchIndex() : nullptr;
}

void ViewerManager::setSurfacePatchIndexCacheKey(const QString& key)
{
    if (key == _surfacePatchIndexCacheKey) {
        return;
    }
    const bool primeBusy = _surfacePatchIndexWatcher && _surfacePatchIndexWatcher->isRunning();
    const bool taskBusy = (_surfacePatchIndexTaskWatcher && _surfacePatchIndexTaskWatcher->isRunning()) ||
                          !_pendingSurfacePatchIndexTasks.empty();
    // Stash the live index under the outgoing key so returning to that folder
    // selection can skip the full rebuild. Only safe when the index is current
    // and no worker is reading or mutating it; otherwise it is simply dropped
    // and rebuilt on return like before.
    if (!_surfacePatchIndexCacheKey.isEmpty() && !_surfacePatchIndexNeedsRebuild &&
        !_surfacePatchIndex.empty() && !primeBusy && !taskBusy &&
        !_deferredIndexSwap && _indexReadsInFlight == 0) {
        CachedSurfacePatchIndex entry;
        entry.index = std::move(_surfacePatchIndex);
        entry.ids = std::move(_indexedSurfaceIds);
        _surfacePatchIndexCache.insert_or_assign(_surfacePatchIndexCacheKey, std::move(entry));
        _surfacePatchIndex = SurfacePatchIndex{};
        _surfacePatchIndex.setSamplingStride(_surfacePatchSamplingStride);
        VC3D_DEBUG_QCINFO(lcViewerManager) << "Stashed SurfacePatchIndex for" << _surfacePatchIndexCacheKey;
    }
    // The old folder's surfaces are about to be unbound; clearing the indexed
    // ids keeps handleSurfaceWillBeDeleted from queueing per-cell rtree
    // removals for surfaces the next prime replaces wholesale anyway.
    _indexedSurfaceIds.clear();
    _surfacePatchIndexNeedsRebuild = true;
    _surfacePatchIndexCacheKey = key;
    forEachBaseViewer([](VolumeViewerBase* v) { v->invalidateIntersect(); });
}

void ViewerManager::clearSurfacePatchIndexCache()
{
    _surfacePatchIndexCache.clear();
    _surfacePatchIndexCacheKey.clear();
}

void ViewerManager::invalidateSurfacePatchIndexCacheFor(const SurfacePatchIndex::SurfacePtr& surface)
{
    if (!surface) {
        return;
    }
    // The surface's geometry changed; any stashed index containing it would
    // pass the instance-identity check on reuse while holding stale patches.
    for (auto it = _surfacePatchIndexCache.begin(); it != _surfacePatchIndexCache.end();) {
        if (it->second.index.containsSurface(surface)) {
            it = _surfacePatchIndexCache.erase(it);
        } else {
            ++it;
        }
    }
}

void ViewerManager::refreshSurfacePatchIndex(const SurfacePatchIndex::SurfacePtr& surface)
{
    if (!surface) {
        return;
    }
    invalidateSurfacePatchIndexCacheFor(surface);
    // An intersection worker is reading the index now: defer the in-place mutation
    // (mark dirty -> rebuilt on the next query) instead of tearing the read.
    if (_indexReadsInFlight > 0) { _surfacePatchIndexNeedsRebuild = true; return; }
    const std::string surfId = surface->id;
    if (_surfacePatchIndexNeedsRebuild || _surfacePatchIndex.empty()) {
        _surfacePatchIndexNeedsRebuild = true;
        _indexedSurfaceIds.erase(surfId);
        VC3D_DEBUG_QCINFO(lcViewerManager) << "Deferred surface index refresh for" << surfId.c_str()
                                << "(global rebuild pending)";
        return;
    }

    if (_surfacePatchIndex.updateSurface(surface)) {
        _indexedSurfaceIds.insert(surfId);
        VC3D_DEBUG_QCINFO(lcViewerManager) << "Rebuilt SurfacePatchIndex entries for surface" << surfId.c_str();
        return;
    }

    _surfacePatchIndexNeedsRebuild = true;
    _indexedSurfaceIds.erase(surfId);
    VC3D_DEBUG_QCINFO(lcViewerManager) << "Failed to rebuild SurfacePatchIndex for surface" << surfId.c_str()
                            << "- marking index for rebuild";
}

void ViewerManager::refreshSurfacePatchIndex(const SurfacePatchIndex::SurfacePtr& surface, const cv::Rect& changedRegion)
{
    if (!surface) {
        return;
    }
    // Empty rect means no changes
    if (changedRegion.empty()) {
        VC3D_DEBUG_QCINFO(lcViewerManager) << "Skipped SurfacePatchIndex update (no changes)";
        return;
    }
    invalidateSurfacePatchIndexCacheFor(surface);
    if (_indexReadsInFlight > 0) { _surfacePatchIndexNeedsRebuild = true; return; }  // defer: worker reading

    const std::string surfId = surface->id;
    if (_surfacePatchIndexNeedsRebuild || _surfacePatchIndex.empty()) {
        _surfacePatchIndexNeedsRebuild = true;
        _indexedSurfaceIds.erase(surfId);
        VC3D_DEBUG_QCINFO(lcViewerManager) << "Deferred surface index refresh for" << surfId.c_str()
                                << "(global rebuild pending)";
        return;
    }

    // Use region-based update
    const int rowStart = changedRegion.y;
    const int rowEnd = changedRegion.y + changedRegion.height;
    const int colStart = changedRegion.x;
    const int colEnd = changedRegion.x + changedRegion.width;

    if (_surfacePatchIndex.updateSurfaceRegion(surface, rowStart, rowEnd, colStart, colEnd)) {
        _indexedSurfaceIds.insert(surfId);
        VC3D_DEBUG_QCINFO(lcViewerManager) << "Updated SurfacePatchIndex region for" << surfId.c_str()
                                << "rows" << rowStart << "-" << rowEnd
                                << "cols" << colStart << "-" << colEnd;
        return;
    }

    // Region update failed, fall back to full surface update
    refreshSurfacePatchIndex(surface);
}

void ViewerManager::primeSurfacePatchIndicesAsync()
{
    if (!_surfacePatchIndexWatcher) {
        return;
    }
    const bool hadRunningBuild = _surfacePatchIndexWatcher->isRunning();
    if (hadRunningBuild) {
        _surfacePatchIndexWatcher->cancel();
    }
    if (!_state) {
        return;
    }
    auto allSurfaces = _state->surfaces();
    std::vector<SurfacePatchIndex::SurfacePtr> quadSurfaces;
    std::vector<std::string> surfaceIds;
    quadSurfaces.reserve(allSurfaces.size());
    surfaceIds.reserve(allSurfaces.size());
    // Track seen surfaces to avoid duplicates (e.g., "segmentation" alias)
    std::unordered_set<SurfacePatchIndex::SurfacePtr> seenSurfaces;
    for (const auto& surface : allSurfaces) {
        if (auto quad = std::dynamic_pointer_cast<QuadSurface>(surface)) {
            if (vc3d::opendata::isOpenDataSegmentPlaceholder(quad->path)) {
                continue;
            }
            // Skip if we've already seen this surface (shared_ptr hash uses underlying pointer)
            if (seenSurfaces.insert(quad).second) {
                quadSurfaces.push_back(quad);
                surfaceIds.push_back(surface->id);
            }
        }
    }
    // Apply max surfaces limit
    if (_intersectionMaxSurfaces > 0 && quadSurfaces.size() > static_cast<size_t>(_intersectionMaxSurfaces)) {
        quadSurfaces.resize(_intersectionMaxSurfaces);
        surfaceIds.resize(_intersectionMaxSurfaces);
    }
    _pendingSurfacePatchIndexSurfaceIds = surfaceIds;
    if (quadSurfaces.empty()) {
        if (_indexReadsInFlight > 0) {                 // worker reading: defer the clear
            _surfacePatchIndexNeedsRebuild = true; return;
        }
        _surfacePatchIndex.clear();
        _indexedSurfaceIds.clear();
        _surfacePatchIndexNeedsRebuild = false;
        return;
    }

    // The live index already covers exactly this surface set at the current
    // stride (e.g. a reload that only changed folder colors): nothing to do.
    if (!hadRunningBuild && !_surfacePatchIndexNeedsRebuild && !_deferredIndexSwap &&
        !_surfacePatchIndex.empty() &&
        _surfacePatchIndex.samplingStride() == _surfacePatchSamplingStride &&
        _surfacePatchIndex.surfaceCount() == quadSurfaces.size() &&
        std::all_of(quadSurfaces.begin(), quadSurfaces.end(),
                    [this](const SurfacePatchIndex::SurfacePtr& quad) {
                        return _surfacePatchIndex.containsSurface(quad);
                    })) {
        _pendingSurfacePatchIndexSurfaceIds.clear();
        return;
    }

    // A folder-selection switch may have stashed a still-valid index for this
    // exact surface set — swap it back in instead of rebuilding.
    if (auto cacheIt = _surfacePatchIndexCache.find(_surfacePatchIndexCacheKey);
        cacheIt != _surfacePatchIndexCache.end()) {
        CachedSurfacePatchIndex entry = std::move(cacheIt->second);
        _surfacePatchIndexCache.erase(cacheIt);
        const bool matches =
            entry.index.samplingStride() == _surfacePatchSamplingStride &&
            entry.index.surfaceCount() == quadSurfaces.size() &&
            std::all_of(quadSurfaces.begin(), quadSurfaces.end(),
                        [&entry](const SurfacePatchIndex::SurfacePtr& quad) {
                            return entry.index.containsSurface(quad);
                        });
        if (matches) {
            _surfacesQueuedDuringRebuild.clear();
            _pendingSurfacePatchIndexSurfaceIds.clear();
            if (_indexReadsInFlight > 0) {
                // A worker is reading the live index; apply via the deferred
                // swap path once reads drain.
                _deferredIndexSwap = std::make_shared<SurfacePatchIndex>(std::move(entry.index));
                _deferredIndexSwapIds.assign(surfaceIds.begin(), surfaceIds.end());
                _surfacePatchIndexNeedsRebuild = false;
                return;
            }
            _surfacePatchIndex = std::move(entry.index);
            _indexedSurfaceIds = std::move(entry.ids);
            _surfacePatchIndexNeedsRebuild = false;
            VC3D_DEBUG_QCINFO(lcViewerManager) << "Reused stashed SurfacePatchIndex for"
                                    << _indexedSurfaceIds.size() << "surfaces"
                                    << "(" << _surfacePatchIndexCacheKey << ")";
            forEachBaseViewer([](VolumeViewerBase* v) {
                v->invalidateIntersect();
                v->renderIntersections("surface index cache hit");
            });
            return;
        }
        // Entry went stale (surfaces reloaded, deleted, or stride changed);
        // it was dropped above — fall through to a full rebuild.
    }

    // Clear rebuild flag since we're about to do an async build
    // (prevents rebuildSurfacePatchIndexIfNeeded from triggering a synchronous build)
    _surfacePatchIndexNeedsRebuild = false;

    // Clear any surfaces queued from a previous rebuild cycle
    _surfacesQueuedDuringRebuild.clear();

    // Build task captures shared_ptrs - surfaces stay alive throughout async operation
    const int stride = _surfacePatchSamplingStride;
    auto future = QtConcurrent::run([quadSurfaces, stride]() -> std::shared_ptr<SurfacePatchIndex> {
        try {
            auto index = std::make_shared<SurfacePatchIndex>();
            index->setSamplingStride(stride);
            index->rebuild(quadSurfaces);
            return index;
        } catch (const std::exception& e) {
            qCWarning(lcViewerManager) << "SurfacePatchIndex async rebuild failed:" << e.what();
        } catch (...) {
            qCWarning(lcViewerManager) << "SurfacePatchIndex async rebuild failed with an unknown exception";
        }
        return nullptr;
    });
    _surfacePatchIndexWatcher->setFuture(future);
}

void ViewerManager::rebuildSurfacePatchIndexIfNeeded()
{
    if (!_surfacePatchIndexNeedsRebuild) {
        return;
    }
    // Called from the render hot path via surfacePatchIndex(). Do not do
    // a synchronous rtree rebuild here — a 2K² surface takes seconds and
    // would freeze the GUI. primeSurfacePatchIndicesAsync() clears the
    // flag inside itself; readers will see the stale-but-valid index
    // until the worker swap completes.
    primeSurfacePatchIndicesAsync();
}

void ViewerManager::endIndexRead()
{
    if (_indexReadsInFlight > 0) --_indexReadsInFlight;
    if (_indexReadsInFlight > 0) return;               // other reads still in flight
    // Reads drained: apply a swap that was deferred while a worker was reading.
    if (_deferredIndexSwap) {
        _surfacePatchIndex = std::move(*_deferredIndexSwap);
        _deferredIndexSwap.reset();
        _surfacePatchIndexNeedsRebuild = false;
        _indexedSurfaceIds.clear();
        _indexedSurfaceIds.insert(_deferredIndexSwapIds.begin(), _deferredIndexSwapIds.end());
        _deferredIndexSwapIds.clear();
        forEachBaseViewer([](VolumeViewerBase* v) { v->renderIntersections("deferred index swap"); });
    }
    // Run any single-surface mutation task that was held while reads were in flight.
    startNextSurfacePatchIndexTask();
}

void ViewerManager::handleSurfacePatchIndexPrimeFinished()
{
    if (!_surfacePatchIndexWatcher) {
        return;
    }
    // A canceled build (superseded by a newer prime or a cache swap-in) may
    // still report finished; its result was discarded and must not be read.
    if (_surfacePatchIndexWatcher->future().isCanceled()) {
        return;
    }
    auto result = _surfacePatchIndexWatcher->future().result();
    if (!result) {
        _pendingSurfacePatchIndexSurfaceIds.clear();
        return;
    }
    // An intersection worker may be reading the live index. Swapping it now would
    // tear that read -> stash the rebuilt index; endIndexRead() applies it once reads
    // drain.
    if (_indexReadsInFlight > 0) {
        _deferredIndexSwap = std::move(result);
        _deferredIndexSwapIds = _pendingSurfacePatchIndexSurfaceIds;
        return;
    }
    _surfacePatchIndex = std::move(*result);
    _surfacePatchIndexNeedsRebuild = false;
    _indexedSurfaceIds.clear();
    _indexedSurfaceIds.insert(_pendingSurfacePatchIndexSurfaceIds.begin(),
                              _pendingSurfacePatchIndexSurfaceIds.end());

    auto queuedDuringRebuild = std::move(_surfacesQueuedDuringRebuild);
    _surfacesQueuedDuringRebuild.clear();

    VC3D_DEBUG_QCINFO(lcViewerManager) << "Asynchronously rebuilt SurfacePatchIndex for"
                            << _indexedSurfaceIds.size() << "surfaces"
                            << "at stride" << _surfacePatchSamplingStride;

    _pendingSurfacePatchIndexSurfaceIds.clear();

    // Surfaces added/removed while the full async rebuild was running do not
    // require another full rebuild. Apply just those deltas on the worker.
    if (queuedDuringRebuild.empty()) {
        forEachBaseViewer([](VolumeViewerBase* v) { v->renderIntersections(); });
    } else {
        forEachBaseViewer([](VolumeViewerBase* v) { v->invalidateIntersect(); });
        for (auto& task : queuedDuringRebuild) {
            queueSurfacePatchIndexTask(std::move(task));
        }
    }
}

void ViewerManager::queueSurfacePatchIndexTask(SurfacePatchIndexTask task)
{
    if (!task.surface) {
        return;
    }

    const QuadSurface* raw = task.surface.get();
    if (task.type == SurfacePatchIndexTaskType::Remove) {
        _indexedSurfaceIds.erase(task.id);
        _pendingSurfacePatchIndexTasks.erase(
            std::remove_if(_pendingSurfacePatchIndexTasks.begin(),
                           _pendingSurfacePatchIndexTasks.end(),
                           [raw](const SurfacePatchIndexTask& pending) {
                               return pending.surface.get() == raw;
                           }),
            _pendingSurfacePatchIndexTasks.end());
    } else {
        for (auto& pending : _pendingSurfacePatchIndexTasks) {
            if (pending.type == SurfacePatchIndexTaskType::Update &&
                pending.surface.get() == raw) {
                pending = std::move(task);
                startNextSurfacePatchIndexTask();
                return;
            }
        }
    }

    _pendingSurfacePatchIndexTasks.push_back(std::move(task));
    startNextSurfacePatchIndexTask();
}

void ViewerManager::startNextSurfacePatchIndexTask()
{
    if (!_surfacePatchIndexTaskWatcher ||
        _surfacePatchIndexTaskWatcher->isRunning() ||
        _pendingSurfacePatchIndexTasks.empty()) {
        return;
    }
    // This task MUTATES the live index on a worker. A plane-intersection read worker
    // may be reading it concurrently -> hold the task until reads drain (endIndexRead
    // re-runs us). Single-writer + single-reader exclusion, no lock on the index.
    if (_indexReadsInFlight > 0) {
        return;
    }

    SurfacePatchIndexTask task = std::move(_pendingSurfacePatchIndexTasks.front());
    _pendingSurfacePatchIndexTasks.erase(_pendingSurfacePatchIndexTasks.begin());

    auto* index = &_surfacePatchIndex;
    auto future = QtConcurrent::run([index, task = std::move(task)]() mutable -> SurfacePatchIndexTaskResult {
        SurfacePatchIndexTaskResult result;
        result.type = task.type;
        result.id = std::move(task.id);
        result.surface = std::move(task.surface);

        try {
            if (result.type == SurfacePatchIndexTaskType::Update) {
                result.success = index->updateSurface(result.surface);
            } else {
                result.success = index->removeSurface(result.surface);
            }
        } catch (const std::exception& e) {
            qCWarning(lcViewerManager) << "SurfacePatchIndex single-surface task failed:" << e.what();
        } catch (...) {
            qCWarning(lcViewerManager) << "SurfacePatchIndex single-surface task failed with an unknown exception";
        }

        return result;
    });
    _surfacePatchIndexTaskWatcher->setFuture(future);
}

void ViewerManager::handleSurfacePatchIndexTaskFinished()
{
    if (!_surfacePatchIndexTaskWatcher) {
        return;
    }

    const auto result = _surfacePatchIndexTaskWatcher->future().result();
    if (_surfacePatchIndexNeedsRebuild) {
        // A bulk surface replacement superseded this serialized delta while it
        // was running. The task mutated only the outgoing live index; rebuild
        // once from the final state and do not render the transient result.
        _indexedSurfaceIds.clear();
        primeSurfacePatchIndicesAsync();
        return;
    }

    if (result.success) {
        if (result.type == SurfacePatchIndexTaskType::Update) {
            _indexedSurfaceIds.insert(result.id);
            VC3D_DEBUG_QCINFO(lcViewerManager) << "Updated SurfacePatchIndex for surface"
                                               << result.id.c_str();
            forEachBaseViewer([](VolumeViewerBase* v) { v->renderIntersections(); });
        } else {
            _indexedSurfaceIds.erase(result.id);
            VC3D_DEBUG_QCINFO(lcViewerManager) << "Removed surface from SurfacePatchIndex"
                                               << result.id.c_str();
            forEachBaseViewer([](VolumeViewerBase* v) {
                v->invalidateIntersect();
                v->renderIntersections();
            });
        }
    } else if (result.type == SurfacePatchIndexTaskType::Update) {
        _indexedSurfaceIds.erase(result.id);
        _surfacePatchIndexNeedsRebuild = true;
        VC3D_DEBUG_QCINFO(lcViewerManager) << "Failed to update SurfacePatchIndex for surface"
                                           << result.id.c_str()
                                           << "- marking index for rebuild";
    }

    startNextSurfacePatchIndexTask();
}

bool ViewerManager::updateSurfacePatchIndexForSurface(const SurfacePatchIndex::SurfacePtr& quad, bool isEditUpdate)
{
    if (!quad) {
        return false;
    }

    const std::string surfId = quad->id;
    const bool alreadyIndexed = _surfacePatchIndex.containsSurface(quad);

    // Check if async rebuild is in progress
    const bool asyncRebuildInProgress = _surfacePatchIndexWatcher &&
                                        _surfacePatchIndexWatcher->isRunning();

    // During an active edit session, the mutable preview surface is tracked by
    // SegmentationEditManager's single-surface index. Do not churn the global
    // all-surfaces index for every preview update; it is refreshed when the
    // edit session is committed or closed.
    if (isEditUpdate) {
        if (alreadyIndexed) {
            _indexedSurfaceIds.insert(surfId);
        }
        return true;
    }

    if (_surfacePatchIndex.hasPendingUpdates(quad)) {
        // Pending cell updates mean real geometry changes; stashed indexes
        // containing this surface are stale.
        invalidateSurfacePatchIndexCacheFor(quad);
        const bool flushed = _surfacePatchIndex.flushPendingUpdates(quad);
        if (flushed) {
            _indexedSurfaceIds.insert(surfId);
        }
        _surfacePatchIndexNeedsRebuild = _surfacePatchIndexNeedsRebuild && !flushed;
        return flushed;
    }

    // Non-edit surfaceChanged signals are also used for UI alias/selection
    // updates, especially the "segmentation" alias. If the surface is already
    // present in the index and no cell updates are pending, there is no rtree
    // work to do.
    if (!isEditUpdate && alreadyIndexed) {
        _indexedSurfaceIds.insert(surfId);
        return true;
    }

    if (asyncRebuildInProgress) {
        // An async rebuild is already running. Apply this surface as a
        // single-surface delta after the rebuilt index is swapped in.
        _surfacesQueuedDuringRebuild.push_back(
            {SurfacePatchIndexTaskType::Update, surfId, quad});
        return true;
    }

    if (!_surfacePatchIndex.empty()) {
        queueSurfacePatchIndexTask(
            {SurfacePatchIndexTaskType::Update, surfId, quad});
        return true;
    }

    _surfacePatchIndexNeedsRebuild = true;
    return true;
}

void ViewerManager::handleSurfaceChanged(std::string name, std::shared_ptr<Surface> surf, bool isEditUpdate)
{
    if (name.empty()) {
        // Empty-name notifications represent a completed bulk mutation of the
        // surface catalog. Drop serialized deltas and rebuild once from the
        // final CState snapshot instead of replaying thousands of updates.
        _pendingSurfacePatchIndexTasks.clear();
        _surfacesQueuedDuringRebuild.clear();
        _pendingSurfacePatchIndexSurfaceIds.clear();
        _deferredIndexSwap.reset();
        _deferredIndexSwapIds.clear();
        _indexedSurfaceIds.clear();
        _surfacePatchIndexNeedsRebuild = true;
        forEachBaseViewer([](VolumeViewerBase* v) {
            v->invalidateIntersect();
        });

        // A single-surface task mutates the live index. Let it drain before
        // launching the replacement build; its completion handler will prime.
        const bool taskRunning =
            _surfacePatchIndexTaskWatcher && _surfacePatchIndexTaskWatcher->isRunning();
        if (!taskRunning) {
            primeSurfacePatchIndicesAsync();
        }
        return;
    }

    bool affectsSurfaceIndex = false;
    bool regionUpdated = false;

    if (auto quad = std::dynamic_pointer_cast<QuadSurface>(surf)) {
        affectsSurfaceIndex = true;
        if (updateSurfacePatchIndexForSurface(quad, isEditUpdate)) {
            regionUpdated = true;  // Signal that work was done (prevents marking index for rebuild)
        }
    } else if (!surf) {
        // Surface was removed - the handleSurfaceWillBeDeleted already cleaned up the index
        affectsSurfaceIndex = true;
        regionUpdated = true;  // Incremental removal already done - don't trigger full rebuild
        _indexedSurfaceIds.erase(name);
    }

    if (affectsSurfaceIndex) {
        _surfacePatchIndexNeedsRebuild = _surfacePatchIndexNeedsRebuild || !regionUpdated;
    }
}

void ViewerManager::handleSurfaceWillBeDeleted(std::string name, std::shared_ptr<Surface> surf)
{
    // Fast path on app shutdown: don't bother maintaining the rtree when
    // everything is about to be freed anyway. A single large surface would
    // otherwise cost ~11s of per-cell tree->remove() while the user stares
    // at a frozen window.
    if (_shuttingDown.load(std::memory_order_relaxed)) {
        return;
    }

    // Called BEFORE surface deletion - remove from R-tree index
    auto quad = std::dynamic_pointer_cast<QuadSurface>(surf);

    // Only process cleanup if we're deleting under the surface's actual ID.
    // Aliases like "segmentation" just point to surfaces that exist under their
    // own IDs - we don't want to remove from the index when an alias changes.
    const bool isDeletingByActualId = quad && (name == quad->id);

    if (isDeletingByActualId) {
        // Track whether this surface was ever actually indexed. If not,
        // the R-tree has nothing to remove — skip the whole mask walk.
        const bool wasIndexed = (_indexedSurfaceIds.find(name)
                                 != _indexedSurfaceIds.end());

        // Remove from indexed surface IDs
        _indexedSurfaceIds.erase(name);

        // Remove from queued-for-add IDs
        auto removeFromVector = [&name](std::vector<std::string>& vec) {
            vec.erase(std::remove(vec.begin(), vec.end(), name), vec.end());
        };
        removeFromVector(_pendingSurfacePatchIndexSurfaceIds);
        _surfacesQueuedDuringRebuild.erase(
            std::remove_if(_surfacesQueuedDuringRebuild.begin(),
                           _surfacesQueuedDuringRebuild.end(),
                           [&name](const SurfacePatchIndexTask& task) {
                               return task.id == name;
                           }),
            _surfacesQueuedDuringRebuild.end());

        // If an async rebuild is in progress, queue for removal from the new
        // index when it completes. Store the shared_ptr so the surface stays
        // alive for the R-tree removal even after CState drops it.
        bool asyncRebuildInProgress = _surfacePatchIndexWatcher &&
                                       _surfacePatchIndexWatcher->isRunning();
        if (asyncRebuildInProgress) {
            _surfacesQueuedDuringRebuild.push_back(
                {SurfacePatchIndexTaskType::Remove, name, quad});
        } else if (wasIndexed) {
            queueSurfacePatchIndexTask(
                {SurfacePatchIndexTaskType::Remove, name, quad});
        } else {
            VC3D_DEBUG_QCINFO(lcViewerManager) << "handleSurfaceWillBeDeleted:" << name.c_str()
                                    << "skipping removeSurface (never indexed)";
        }

        if (asyncRebuildInProgress || wasIndexed) {
            // Hide stale lines immediately; the async removal will update the
            // R-tree before intersections are rendered again.
            forEachBaseViewer([](VolumeViewerBase* v) {
                v->invalidateIntersect();
            });
        }
    }
}

bool ViewerManager::resetDefaultFor(VolumeViewerBase* viewer) const
{
    auto it = _resetDefaults.find(viewer);
    return it != _resetDefaults.end() ? it->second : true;
}

void ViewerManager::setResetDefaultFor(VolumeViewerBase* viewer, bool value)
{
    if (!viewer) {
        return;
    }
    _resetDefaults[viewer] = value;
}

void ViewerManager::setSegmentationCursorMirroring(bool enabled)
{
    _mirrorCursorToSegmentation = enabled;
    forEachBaseViewer([enabled](VolumeViewerBase* v) { v->setSegmentationCursorMirroring(enabled); });
    if (!enabled) {
        broadcastLinkedCursor(nullptr, std::nullopt);
    }
}

void ViewerManager::broadcastLinkedCursor(VolumeViewerBase* source,
                                          const std::optional<cv::Vec3f>& point)
{
    if (!_mirrorCursorToSegmentation && point.has_value()) {
        return;
    }
    forEachBaseViewer([source, &point](VolumeViewerBase* viewer) {
        if (viewer != source) {
            viewer->setLinkedCursorVolumePoint(point);
        }
    });
}

void ViewerManager::setZScrollSensitivity(double sensitivity)
{
    const double clamped = std::clamp(sensitivity, 0.1, 100.0);
    if (std::abs(_zScrollSensitivity - clamped) < 1e-9) {
        return;
    }
    _zScrollSensitivity = clamped;

    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue(vc3d::settings::viewer::ZSCROLL_SENSITIVITY, _zScrollSensitivity);

    forEachBaseViewer([](VolumeViewerBase* viewer) {
        if (viewer) {
            viewer->reloadPerfSettings();
        }
    });

    emit zScrollSensitivityChanged(_zScrollSensitivity);
}

void ViewerManager::forEachBaseViewer(const std::function<void(VolumeViewerBase*)>& fn) const
{
    if (!fn) {
        return;
    }
    for (auto* viewer : _baseViewers) {
        if (viewer) {
            fn(viewer);
        }
    }
}
