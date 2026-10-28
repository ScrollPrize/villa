#include "ViewerManager.hpp"
#include "OpenDataManifest.hpp"

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
#include "SurfaceCatalog.hpp"
#include "vc/core/types/VolumePkg.hpp"
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

std::optional<cv::Vec3f> transformPoint(const cv::Vec3f& point, const vcr::catalog::Affine& matrix)
{
    if (!finiteVec3(point)) {
        return std::nullopt;
    }
    const auto transformed = matrix.apply({point[0], point[1], point[2]});
    const cv::Vec3f out(static_cast<float>(transformed[0]),
                        static_cast<float>(transformed[1]),
                        static_cast<float>(transformed[2]));
    return finiteVec3(out) ? std::optional<cv::Vec3f>(out) : std::nullopt;
}

// A direction's normal through the transform, kept when it does not map.
cv::Vec3f transformNormal(const cv::Vec3f& normal, const vcr::catalog::Affine& matrix)
{
    if (!finiteVec3(normal)) {
        return normal;
    }
    const auto mapped = matrix.mapNormal({normal[0], normal[1], normal[2]});
    const cv::Vec3f out(static_cast<float>(mapped[0]), static_cast<float>(mapped[1]), static_cast<float>(mapped[2]));
    return finiteVec3(out) && cv::norm(out) > 0.0f ? out : normal;
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

    const float storedThickness = settings.value(viewer::INTERSECTION_THICKNESS, viewer::INTERSECTION_THICKNESS_DEFAULT).toFloat();
    _intersectionThickness = std::max(0.0f, storedThickness);

    // Derived surface tiles are workspace-local, while their decoded volume
    // chunks come from the application-wide cache service.
    applyViewerCacheSettings();

    if (_state) {
        connect(_state,
                &CState::volumeChanged,
                this,
                &ViewerManager::currentVolumeChanged);
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

void ViewerManager::applyViewerCacheSettings()
{
    using namespace vc3d::settings;
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    constexpr std::size_t mib = 1024ULL * 1024ULL;
    constexpr std::size_t gib = 1024ULL * mib;
    const auto surfaceGb = std::max<qlonglong>(
        0, settings.value(viewer_cache::SURFACE_CACHE_GB,
                          viewer_cache::SURFACE_CACHE_GB_DEFAULT).toLongLong());
    const auto overlayGb = std::max<qlonglong>(
        0, settings.value(viewer_cache::OVERLAY_SURFACE_CACHE_GB,
                          viewer_cache::OVERLAY_SURFACE_CACHE_GB_DEFAULT).toLongLong());

    setSurfaceCacheBudgets(std::size_t(surfaceGb) * gib, std::size_t(overlayGb) * gib);
}

void ViewerManager::setSurfaceCacheBudgets(std::size_t baseBytes, std::size_t overlayBytes)
{
    if (_surfaceCacheBudgetBytes == baseBytes &&
        _overlaySurfaceCacheBudgetBytes == overlayBytes) {
        return;
    }
    _surfaceCacheBudgetBytes = baseBytes;
    _overlaySurfaceCacheBudgetBytes = overlayBytes;
    forEachBaseViewer([baseBytes, overlayBytes](VolumeViewerBase* viewer) {
        if (viewer)
            viewer->setSurfaceCacheBudgets(baseBytes, overlayBytes);
    });
}

void ViewerManager::setPreferSurfaceTileFills(bool enabled)
{
    if (_preferSurfaceTileFills == enabled) {
        return;
    }
    _preferSurfaceTileFills = enabled;
    forEachBaseViewer([enabled](VolumeViewerBase* viewer) {
        if (viewer)
            viewer->setPreferSurfaceTileFills(enabled);
    });
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
        connect(_state, &CState::focusBoundsChanged, chunkedViewer,
                [chunkedViewer]() {
                    chunkedViewer->requestRender("focus bounds changed");
                });
    }

    // Restore persisted viewer preferences
    {
        using namespace vc3d::settings;
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        bool showNormals = settings.value(viewer::SHOW_SURFACE_NORMALS, viewer::SHOW_SURFACE_NORMALS_DEFAULT).toBool();
        baseViewer->setShowSurfaceNormals(showNormals);
    }
    chunkedViewer->setShowCoordinateFrame(_showCoordinateFrames);

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
    baseViewer->setVolumeWindow(_volumeWindowLow, _volumeWindowHigh);
    baseViewer->setOverlayVolume(_overlayVolume);
    baseViewer->setOverlayOpacity(_overlayOpacity);
    baseViewer->setOverlayColormap(_overlayColormapId);
    baseViewer->setOverlaySamplingMethod(_overlaySamplingMethod);
    baseViewer->setOverlayWindow(_overlayWindowLow, _overlayWindowHigh);
    baseViewer->setOverlayIntensityAsAlpha(_overlayIntensityAsAlpha);
    baseViewer->setOverlayMaxDisplayedResolution(_overlayMaxDisplayedResolution);
    baseViewer->setOverlayComposite(_overlayComposite);
    baseViewer->setSurfaceCacheBudgets(_surfaceCacheBudgetBytes,
                                       _overlaySurfaceCacheBudgetBytes);
    baseViewer->setPreferSurfaceTileFills(_preferSurfaceTileFills);

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

    if (_pendingLinkedCursorSource == viewer) {
        _pendingLinkedCursorSource = nullptr;
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

    if ((modifiers & Qt::ControlModifier) && (modifiers & Qt::ShiftModifier) &&
        button == Qt::LeftButton) {
        // Ctrl+Shift+click: like Ctrl+click, but if a patch lies under the
        // clicked point, activate it first so the flattened view navigates on
        // the right segment. Slice views only — there `surf` is the plane; the
        // flattened segmentation view reports the quad surface itself.
        if (!dynamic_cast<PlaneSurface*>(surf)) {
            return;
        }
        static constexpr float kPatchPickToleranceVoxels = 10.0f;
        if (_surfacePicker) {
            if (const auto id = _surfacePicker(volLoc, kPatchPickToleranceVoxels)) {
                emit surfaceActivationRequested(*id);
            }
        }
        std::string surfId;
        if (_state && surf) {
            surfId = _state->findSurfaceId(surf);
        }
        centerFocusAt(volLoc, normal, surfId);
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

    if (const auto hit = locate(*activeSurface, position, kMaxDistanceVoxels)) {
        const cv::Vec3f loc = activeSurface->loc(hit->ptr);
        centerViewerOnSurfacePointForNavigation(viewer, {loc[0], loc[1]});
    }
}

std::optional<ViewerManager::SurfaceLocation> ViewerManager::locate(const QuadSurface& surface, const cv::Vec3f& point,
                                                                    float tolerance) const
{
    return locate(surface, std::vector<cv::Vec3f>{point}, tolerance).front();
}

std::vector<std::optional<ViewerManager::SurfaceLocation>> ViewerManager::locate(const QuadSurface& surface,
                                                                                 const std::vector<cv::Vec3f>& points,
                                                                                 float tolerance) const
{
    std::vector<std::optional<SurfaceLocation>> result(points.size());
    const auto session = _state ? _state->objectStoreSession() : nullptr;
    if (points.empty() || !session || !_state->currentVolume() || !vc::objects::util::isUuid(surface.id)) return result;
    if (!std::isfinite(tolerance) || tolerance < 0) return result;
    std::vector<std::array<double, 3>> queried;
    std::vector<size_t> sources;
    for (size_t i = 0; i < points.size(); ++i)
        if (std::isfinite(points[i][0]) && std::isfinite(points[i][1]) && std::isfinite(points[i][2])) {
            queried.push_back({points[i][0], points[i][1], points[i][2]});
            sources.push_back(i);
        }
    const auto hits = SurfaceCatalog::nearest(*session, _state->surfaceSources(), _state->currentVolumeId(), {surface.id},
                                              queried, tolerance);
    for (size_t i = 0; i < queried.size(); ++i)
        if (!hits[i].empty()) {
            const auto& hit = hits[i].front();
            result[sources[i]] =
                SurfaceLocation{surface.gridToPtr({float(hit.grid[0]), float(hit.grid[1])}), float(hit.distance)};
        }
    return result;
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

void ViewerManager::setShowSurfaceNormals(bool show)
{
    forEachBaseViewer([show](VolumeViewerBase* viewer) {
        if (viewer) {
            viewer->setShowSurfaceNormals(show);
        }
    });
}

void ViewerManager::setShowCoordinateFrames(bool show)
{
    _showCoordinateFrames = show;
    forEachBaseViewer([show](VolumeViewerBase* viewer) {
        if (auto* chunkedViewer = dynamic_cast<CChunkedVolumeViewer*>(viewer)) {
            chunkedViewer->setShowCoordinateFrame(show);
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
                                      const vcr::catalog::Affine& transform)
{
    if (!snapshot || !_state || !_state->currentVolume()) {
        return;
    }

    // The camera keeps its on-screen scale through a near-similarity (column
    // lengths within about 2%); other transforms keep the camera as it was.
    const auto navigationScale = transform.similarityScale(0.04);
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
            const auto direction = transformNormal(
                captured.camera.zOffsetWorldDir, transform);
            if (finiteVec3(direction)) {
                camera.zOffsetWorldDir = direction;
            }
        }
        viewer->applyCameraState(camera, false);
    }
}

void ViewerManager::switchVolume(std::shared_ptr<Volume> volume,
                                 const std::optional<vcr::catalog::Affine>& navigationTransform)
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
                    transformNormal(focus->n, *navigationTransform);
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
    registerOverlay(overlay);
}

void ViewerManager::setRawPointsOverlay(RawPointsOverlayController* overlay)
{
    registerOverlay(overlay);
}

void ViewerManager::setPathsOverlay(PathsOverlayController* overlay)
{
    registerOverlay(overlay);
}

void ViewerManager::setBBoxOverlay(BBoxOverlayController* overlay)
{
    registerOverlay(overlay);
}

void ViewerManager::setVectorOverlay(VectorOverlayController* overlay)
{
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

void ViewerManager::setOverlayVolume(std::shared_ptr<Volume> volume, const std::string& volumeId)
{
    if (volume && _state && _state->vpkg()) {
        if (!_state->vpkg()->overlayCompatible(_state->currentVolumeId(), volumeId)) {
            Logger()->warn(
                "Rejected volume overlay '{}' because the catalog places it in another frame than '{}'.",
                volumeId, _state->currentVolumeId());
            volume.reset();
        }
    }
    _overlayVolume = std::move(volume);
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

void ViewerManager::setOverlayIntensityAsAlpha(bool enabled)
{
    _overlayIntensityAsAlpha = enabled;
    forEachBaseViewer([this](VolumeViewerBase* v) {
        v->setOverlayIntensityAsAlpha(_overlayIntensityAsAlpha);
    });
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
    _pendingLinkedCursorSource = source;
    _pendingLinkedCursorPoint = point;
    if (!point.has_value()) {
        // Clears (cursor left a viewer, mirroring turned off) apply at once and
        // supersede any point still waiting for the timer.
        flushLinkedCursor();
        return;
    }
    if (!_linkedCursorTimer) {
        _linkedCursorTimer = new QTimer(this);
        _linkedCursorTimer->setSingleShot(true);
        _linkedCursorTimer->setInterval(16);  // ~one global render tick
        connect(_linkedCursorTimer, &QTimer::timeout, this, [this]() { flushLinkedCursor(); });
    }
    if (!_linkedCursorTimer->isActive()) {
        _linkedCursorTimer->start();
    }
}

void ViewerManager::flushLinkedCursor()
{
    if (_linkedCursorTimer) {
        _linkedCursorTimer->stop();
    }
    auto* source = _pendingLinkedCursorSource;
    const auto point = _pendingLinkedCursorPoint;
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
