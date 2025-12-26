#include "PlaneSlicingOverlayController.hpp"

#include "../CVolumeViewer.hpp"
#include "../CSurfaceCollection.hpp"
#include "vc/core/util/Surface.hpp"

#include <QCursor>
#include <QGraphicsScene>

#include <algorithm>
#include <cmath>

#include "vc/core/util/PlaneSurface.hpp"

namespace
{
constexpr const char* kOverlayGroup = "plane_slicing_guides";
constexpr qreal kLineZ = 200.0;
constexpr qreal kHandleZ = 201.0;
constexpr qreal kHandleRadius = 10.0;
const QColor kXZColor(Qt::red);
const QColor kYZColor(Qt::green);
const QColor kHandleOutline(Qt::black);
constexpr Qt::PenStyle kLineStyle = Qt::DashLine;
constexpr float kHandleVolumeOffset = 200.0f;
constexpr float kMinDragDegrees = 0.25f;
} // namespace

PlaneSlicingOverlayController::PlaneSlicingOverlayController(CSurfaceCollection* surfaces,
                                                             QObject* parent)
    : ViewerOverlayControllerBase(kOverlayGroup, parent)
    , _surfaces(surfaces)
{
}

void PlaneSlicingOverlayController::setAxisAlignedEnabled(bool enabled)
{
    if (_axisAlignedEnabled == enabled) {
        return;
    }
    _axisAlignedEnabled = enabled;
    if (!_axisAlignedEnabled) {
        _activeDrag = {};
        for (auto& entry : _viewerStates) {
            if (entry.first && entry.first->fGraphicsView) {
                entry.first->fGraphicsView->setCursor(Qt::ArrowCursor);
            }
            removeInteractions(entry.first);
            clearOverlay(entry.first);
        }
    }
    refreshAll();
}

void PlaneSlicingOverlayController::setRotationSetter(std::function<void(const std::string&, float)> setter)
{
    _rotationSetter = std::move(setter);
}

void PlaneSlicingOverlayController::setRotationFinishedCallback(std::function<void()> callback)
{
    _rotationFinishedCallback = std::move(callback);
}

void PlaneSlicingOverlayController::setAxisAlignedOverlayOpacity(float opacity)
{
    float clamped = std::clamp(opacity, 0.0f, 1.0f);
    if (std::abs(_overlayOpacity - clamped) < 1e-4f) {
        return;
    }
    _overlayOpacity = clamped;
    refreshAll();
}

bool PlaneSlicingOverlayController::isOverlayEnabledFor(CVolumeViewer* viewer) const
{
    if (!_axisAlignedEnabled || !viewer) {
        return false;
    }
    return viewer->surfName() == "xy plane";
}

PlaneSlicingOverlayController::ViewerState& PlaneSlicingOverlayController::ensureViewerState(CVolumeViewer* viewer)
{
    return _viewerStates[viewer];
}

void PlaneSlicingOverlayController::clearViewerState(CVolumeViewer* viewer)
{
    auto it = _viewerStates.find(viewer);
    if (it == _viewerStates.end()) {
        return;
    }
    removeInteractions(viewer);
    _viewerStates.erase(it);
}

void PlaneSlicingOverlayController::installInteractions(CVolumeViewer* viewer, ViewerState& state)
{
    if (state.interactionsInstalled || !viewer) {
        return;
    }

    state.pressConn = QObject::connect(viewer, &CVolumeViewer::sendMousePressVolume,
                                       this, [this, viewer](cv::Vec3f volLoc, cv::Vec3f /*normal*/, Qt::MouseButton button, Qt::KeyboardModifiers modifiers) {
                                           handleMousePress(viewer, volLoc, button, modifiers);
                                       });
    state.moveConn = QObject::connect(viewer, &CVolumeViewer::sendMouseMoveVolume,
                                      this, [this, viewer](cv::Vec3f volLoc, Qt::MouseButtons buttons, Qt::KeyboardModifiers modifiers) {
                                          handleMouseMove(viewer, volLoc, buttons, modifiers);
                                      });
    state.releaseConn = QObject::connect(viewer, &CVolumeViewer::sendMouseReleaseVolume,
                                         this, [this, viewer](cv::Vec3f /*volLoc*/, Qt::MouseButton button, Qt::KeyboardModifiers modifiers) {
                                             handleMouseRelease(viewer, button, modifiers);
                                         });
    state.destroyedConn = QObject::connect(viewer, &QObject::destroyed,
                                           this, [this, viewer]() {
                                               clearViewerState(viewer);
                                           });
    state.interactionsInstalled = true;
}

void PlaneSlicingOverlayController::removeInteractions(CVolumeViewer* viewer)
{
    auto it = _viewerStates.find(viewer);
    if (it == _viewerStates.end()) {
        return;
    }

    ViewerState& state = it->second;
    if (!state.interactionsInstalled) {
        return;
    }

    QObject::disconnect(state.pressConn);
    QObject::disconnect(state.moveConn);
    QObject::disconnect(state.releaseConn);
    QObject::disconnect(state.destroyedConn);
    state.interactionsInstalled = false;
}

void PlaneSlicingOverlayController::updateViewerState(CVolumeViewer* viewer,
                                                      ViewerState& state,
                                                      const std::string& planeName,
                                                      const PlaneVisual& visual)
{
    state.planes[planeName] = visual;
}

void PlaneSlicingOverlayController::collectPrimitives(CVolumeViewer* viewer,
                                                      OverlayBuilder& builder)
{
    if (!viewer || !_surfaces) {
        return;
    }

    if (!_axisAlignedEnabled || viewer->surfName() != "xy plane") {
        removeInteractions(viewer);
        clearOverlay(viewer);
        return;
    }

    ViewerState& state = ensureViewerState(viewer);
    installInteractions(viewer, state);

    auto* focusPoi = _surfaces->poi("focus");
    if (!focusPoi) {
        clearViewerState(viewer);
        return;
    }

    const cv::Vec3f focus = focusPoi->p;

    const struct {
        const char* name;
        cv::Vec3f baseNormal;
        const QColor lineColor;
    } planeDefs[] = {
        {"seg xz", {0.0f, 1.0f, 0.0f}, kXZColor},
        {"seg yz", {1.0f, 0.0f, 0.0f}, kYZColor},
    };

    for (const auto& def : planeDefs) {
        auto planeHolder = _surfaces->surface(def.name);  // Keep surface alive during this iteration
        auto* plane = dynamic_cast<PlaneSurface*>(planeHolder.get());
        if (!plane) {
            continue;
        }

        cv::Vec3f normal = plane->normal({}, {});
        cv::Vec3f dir3D = normal.cross(cv::Vec3f(0.0f, 0.0f, 1.0f));
        if (cv::norm(dir3D) < 1e-5f) {
            continue;
        }
        cv::normalize(dir3D, dir3D);

        cv::Vec3f origin = focus;
        cv::Vec3f dirXY(dir3D[0], dir3D[1], 0.0f);
        if (cv::norm(dirXY) < 1e-5f) {
            continue;
        }
        cv::normalize(dirXY, dirXY);

        cv::Vec3f baseDir = def.baseNormal.cross(cv::Vec3f(0.0f, 0.0f, 1.0f));
        if (cv::norm(baseDir) < 1e-5f) {
            continue;
        }
        cv::normalize(baseDir, baseDir);
        float baseAngle = static_cast<float>(std::atan2(baseDir[1], baseDir[0]) * 180.0 / CV_PI);
        baseAngle = normalizeDegrees(baseAngle);

        const float span = 10000.0f;
        cv::Vec3f positivePoint = origin + dirXY * span;
        cv::Vec3f negativePoint = origin - dirXY * span;

        QPointF positiveScene = builder.viewer()->volumePointToScene(positivePoint);
        QPointF negativeScene = builder.viewer()->volumePointToScene(negativePoint);

        QColor lineColor = def.lineColor;
        lineColor.setAlphaF(_overlayOpacity);

        OverlayStyle lineStyle;
        lineStyle.penColor = lineColor;
        lineStyle.penWidth = 2.0;
        lineStyle.penStyle = kLineStyle;
        lineStyle.z = kLineZ;

        builder.addLineStrip({negativeScene, positiveScene}, false, lineStyle);

        cv::Vec3f handleOffset3D = dirXY * kHandleVolumeOffset;
        cv::Vec3f handlePositive = origin + handleOffset3D;
        cv::Vec3f handleNegative = origin - handleOffset3D;

        QPointF handlePositiveScene = builder.viewer()->volumePointToScene(handlePositive);
        QPointF handleNegativeScene = builder.viewer()->volumePointToScene(handleNegative);

        QColor handlePen = kHandleOutline;
        handlePen.setAlphaF(std::min(1.0f, _overlayOpacity + 0.25f));
        QColor handleBrush = def.lineColor;
        handleBrush.setAlphaF(_overlayOpacity);

        OverlayStyle handleStyle;
        handleStyle.penColor = handlePen;
        handleStyle.penWidth = 1.5;
        handleStyle.brushColor = handleBrush;
        handleStyle.z = kHandleZ;

        builder.addCircle(handlePositiveScene, kHandleRadius, true, handleStyle);
        builder.addCircle(handleNegativeScene, kHandleRadius, true, handleStyle);

        PlaneVisual visual;
        visual.origin = origin;
        visual.directionXY = dirXY;
        visual.handlePositiveVolume = handlePositive;
        visual.handleNegativeVolume = handleNegative;
        visual.handlePositiveScene = handlePositiveScene;
        visual.handleNegativeScene = handleNegativeScene;
        visual.baseAngleDegrees = baseAngle;

        updateViewerState(viewer, state, def.name, visual);
    }
}

static bool pointInsideHandle(const QPointF& scenePoint,
                              const QPointF& handleScene,
                              qreal radius)
{
    const qreal dx = scenePoint.x() - handleScene.x();
    const qreal dy = scenePoint.y() - handleScene.y();
    return (dx * dx + dy * dy) <= (radius * radius);
}

void PlaneSlicingOverlayController::handleMousePress(CVolumeViewer* viewer,
                                                     const cv::Vec3f& volumePoint,
                                                     Qt::MouseButton button,
                                                     Qt::KeyboardModifiers modifiers)
{
    Q_UNUSED(modifiers);
    if (!_axisAlignedEnabled || button != Qt::LeftButton || !viewer || viewer->surfName() != "xy plane") {
        return;
    }

    auto it = _viewerStates.find(viewer);
    if (it == _viewerStates.end()) {
        return;
    }

    ViewerState& state = it->second;
    QPointF scenePoint = viewer->volumePointToScene(volumePoint);

    for (auto& entry : state.planes) {
        const std::string& planeName = entry.first;
        PlaneVisual& visual = entry.second;

        bool onPositive = pointInsideHandle(scenePoint, visual.handlePositiveScene, kHandleRadius);
        bool onNegative = pointInsideHandle(scenePoint, visual.handleNegativeScene, kHandleRadius);
        if (onPositive || onNegative) {
            _activeDrag.viewer = viewer;
            _activeDrag.planeName = planeName;
            _activeDrag.positiveHandle = onPositive;
            viewer->fGraphicsView->setCursor(Qt::ClosedHandCursor);
            break;
        }
    }
}

void PlaneSlicingOverlayController::handleMouseMove(CVolumeViewer* viewer,
                                                    const cv::Vec3f& volumePoint,
                                                    Qt::MouseButtons buttons,
                                                    Qt::KeyboardModifiers modifiers)
{
    Q_UNUSED(modifiers);
    if (!_axisAlignedEnabled || !viewer || viewer->surfName() != "xy plane") {
        return;
    }

    auto it = _viewerStates.find(viewer);
    if (it == _viewerStates.end()) {
        return;
    }

    ViewerState& state = it->second;
    QPointF scenePoint = viewer->volumePointToScene(volumePoint);

    if (_activeDrag.viewer == viewer && !_activeDrag.planeName.empty()) {
        if (!(buttons & Qt::LeftButton)) {
            return;
        }

        auto planeIt = state.planes.find(_activeDrag.planeName);
        if (planeIt == state.planes.end()) {
            return;
        }

        auto* focusPoi = _surfaces ? _surfaces->poi("focus") : nullptr;
        if (!focusPoi || !_rotationSetter) {
            return;
        }

        const PlaneVisual& visual = planeIt->second;

        cv::Vec3f delta = volumePoint - focusPoi->p;
        if (!_activeDrag.positiveHandle) {
            delta *= -1.0f;
        }

        cv::Vec2f deltaXY(delta[0], delta[1]);
        float len = cv::norm(deltaXY);
        if (len < 1e-5f) {
            return;
        }
        deltaXY /= len;

        float angle = static_cast<float>(std::atan2(deltaXY[1], deltaXY[0]) * 180.0 / CV_PI);
        float candidate = normalizeDegrees(angle - visual.baseAngleDegrees);

        float currentAngle = 0.0f;
        auto planeSurfaceHolder = _surfaces->surface(_activeDrag.planeName);  // Keep surface alive
        if (auto* planeSurface = dynamic_cast<PlaneSurface*>(planeSurfaceHolder.get())) {
            cv::Vec3f currentNormal = planeSurface->normal({}, {});
            cv::Vec3f currentDir3D = currentNormal.cross(cv::Vec3f(0.0f, 0.0f, 1.0f));
            if (cv::norm(currentDir3D) > 1e-5f) {
                cv::normalize(currentDir3D, currentDir3D);
                cv::Vec3f currentDirXY(currentDir3D[0], currentDir3D[1], 0.0f);
                if (cv::norm(currentDirXY) > 1e-5f) {
                    cv::normalize(currentDirXY, currentDirXY);
                    currentAngle = normalizeDegrees(static_cast<float>(std::atan2(currentDirXY[1], currentDirXY[0]) * 180.0 / CV_PI) - visual.baseAngleDegrees);
                }
            }
        } else {
            currentAngle = normalizeDegrees(static_cast<float>(std::atan2(visual.directionXY[1], visual.directionXY[0]) * 180.0 / CV_PI) - visual.baseAngleDegrees);
        }
        if (std::abs(candidate - currentAngle) < kMinDragDegrees) {
            return;
        }

        _rotationSetter(_activeDrag.planeName, candidate);
        state.planes[_activeDrag.planeName].directionXY = cv::Vec3f(deltaXY[0], deltaXY[1], 0.0f);
        viewer->overlaysUpdated();
        return;
    }

    bool hoveringHandle = false;
    for (const auto& entry : state.planes) {
        const PlaneVisual& visual = entry.second;
        if (pointInsideHandle(scenePoint, visual.handlePositiveScene, kHandleRadius) ||
            pointInsideHandle(scenePoint, visual.handleNegativeScene, kHandleRadius)) {
            hoveringHandle = true;
            break;
        }
    }

    viewer->fGraphicsView->setCursor(hoveringHandle ? Qt::OpenHandCursor : Qt::ArrowCursor);
}

void PlaneSlicingOverlayController::handleMouseRelease(CVolumeViewer* viewer,
                                                       Qt::MouseButton button,
                                                       Qt::KeyboardModifiers modifiers)
{
    Q_UNUSED(modifiers);
    if (!_axisAlignedEnabled || !viewer || viewer->surfName() != "xy plane") {
        return;
    }

    if (button == Qt::LeftButton && _activeDrag.viewer == viewer) {
        const bool hadActiveDrag = !_activeDrag.planeName.empty();
        _activeDrag.viewer = nullptr;
        _activeDrag.planeName.clear();
        viewer->fGraphicsView->setCursor(Qt::ArrowCursor);
        if (hadActiveDrag && _rotationFinishedCallback) {
            _rotationFinishedCallback();
        }
    }
}

bool PlaneSlicingOverlayController::isScenePointNearRotationHandle(CVolumeViewer* viewer,
                                                                   const QPointF& scenePoint,
                                                                   qreal radiusScale) const
{
    if (!_axisAlignedEnabled || !viewer || radiusScale <= 0.0) {
        return false;
    }

    auto it = _viewerStates.find(viewer);
    if (it == _viewerStates.end()) {
        return false;
    }

    const qreal effectiveRadius = kHandleRadius * std::max<qreal>(radiusScale, 1.0);
    const ViewerState& state = it->second;
    for (const auto& entry : state.planes) {
        const PlaneVisual& visual = entry.second;
        if (pointInsideHandle(scenePoint, visual.handlePositiveScene, effectiveRadius) ||
            pointInsideHandle(scenePoint, visual.handleNegativeScene, effectiveRadius)) {
            return true;
        }
    }

    return false;
}

bool PlaneSlicingOverlayController::isVolumePointNearRotationHandle(CVolumeViewer* viewer,
                                                                    const cv::Vec3f& volumePoint,
                                                                    qreal radiusScale) const
{
    if (!viewer) {
        return false;
    }
    const QPointF scenePoint = viewer->volumePointToScene(volumePoint);
    return isScenePointNearRotationHandle(viewer, scenePoint, radiusScale);
}

float PlaneSlicingOverlayController::normalizeDegrees(float degrees)
{
    while (degrees > 180.0f) {
        degrees -= 360.0f;
    }
    while (degrees <= -180.0f) {
        degrees += 360.0f;
    }
    return degrees;
}
