#include "PlaneSlicingOverlayController.hpp"

#include <opencv2/core.hpp>  // cv::normalize

#include "../volume_viewers/CChunkedVolumeViewer.hpp"
#include "../volume_viewers/VolumeViewerBase.hpp"
#include "../CState.hpp"
#include "../ViewerManager.hpp"
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

PlaneSlicingOverlayController::PlaneSlicingOverlayController(CState* state,
                                                             QObject* parent)
    : ViewerOverlayControllerBase(kOverlayGroup, parent)
    , _state(state)
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
            if (entry.first && entry.first->graphicsView()) {
                entry.first->graphicsView()->setCursor(Qt::ArrowCursor);
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

bool PlaneSlicingOverlayController::isOverlayEnabledFor(VolumeViewerBase* viewer) const
{
    if (!_axisAlignedEnabled || !viewer) {
        return false;
    }
    // The guides render in the xy view only, but the xz/yz viewers must reach
    // collectPrimitives too: their rebuilds (scroll, composite changes) are
    // what propagates a refresh onto the xy viewers' guide lines.
    const std::string& name = viewer->surfName();
    return name == "xy plane" || name == "seg xz" || name == "seg yz";
}

PlaneSlicingOverlayController::ViewerState& PlaneSlicingOverlayController::ensureViewerState(VolumeViewerBase* viewer)
{
    return _viewerStates[viewer];
}

void PlaneSlicingOverlayController::clearViewerState(VolumeViewerBase* viewer)
{
    if (_activeDrag.viewer == viewer) {
        _activeDrag = {};
    }
    auto it = _viewerStates.find(viewer);
    if (it == _viewerStates.end()) {
        return;
    }
    removeInteractions(viewer);
    _viewerStates.erase(it);
}

void PlaneSlicingOverlayController::detachViewer(VolumeViewerBase* viewer)
{
    clearViewerState(viewer);
    ViewerOverlayControllerBase::detachViewer(viewer);
}

void PlaneSlicingOverlayController::installInteractions(VolumeViewerBase* viewer, ViewerState& state)
{
    if (state.interactionsInstalled || !viewer) {
        return;
    }

    QObject* viewerObject = viewer->asQObject();
    if (!viewerObject) {
        return;
    }

    if (auto* chunkedViewer = qobject_cast<CChunkedVolumeViewer*>(viewerObject)) {
        state.pressConn = QObject::connect(chunkedViewer, &CChunkedVolumeViewer::sendMousePressVolume,
                                           this, [this, viewer](cv::Vec3f volLoc, cv::Vec3f /*normal*/, Qt::MouseButton button, Qt::KeyboardModifiers modifiers, QPointF /*scenePos*/) {
                                               handleMousePress(viewer, volLoc, button, modifiers);
                                           });
        state.moveConn = QObject::connect(chunkedViewer, &CChunkedVolumeViewer::sendMouseMoveVolume,
                                          this, [this, viewer](cv::Vec3f volLoc, Qt::MouseButtons buttons, Qt::KeyboardModifiers modifiers, QPointF /*scenePos*/) {
                                              handleMouseMove(viewer, volLoc, buttons, modifiers);
                                          });
        state.releaseConn = QObject::connect(chunkedViewer, &CChunkedVolumeViewer::sendMouseReleaseVolume,
                                             this, [this, viewer](cv::Vec3f /*volLoc*/, Qt::MouseButton button, Qt::KeyboardModifiers modifiers, QPointF /*scenePos*/) {
                                                 handleMouseRelease(viewer, button, modifiers);
                                             });
    } else {
        return;
    }

    state.destroyedConn = QObject::connect(viewerObject, &QObject::destroyed,
                                           this, [this, viewer]() {
                                               clearViewerState(viewer);
                                           });
    state.interactionsInstalled = true;
}

void PlaneSlicingOverlayController::removeInteractions(VolumeViewerBase* viewer)
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

void PlaneSlicingOverlayController::updateViewerState(VolumeViewerBase* viewer,
                                                      ViewerState& state,
                                                      const std::string& planeName,
                                                      const PlaneVisual& visual)
{
    state.planes[planeName] = visual;
}

void PlaneSlicingOverlayController::collectPrimitives(VolumeViewerBase* viewer,
                                                      OverlayBuilder& builder)
{
    if (!viewer || !_state) {
        return;
    }

    if (!_axisAlignedEnabled || viewer->surfName() != "xy plane") {
        removeInteractions(viewer);
        clearOverlay(viewer);
        // The guide lines drawn in the xy view track the xz/yz viewers'
        // scroll offset and composite slab, which don't touch shared state
        // the xy viewer observes; propagate those viewers' refreshes.
        if (_axisAlignedEnabled &&
            (viewer->surfName() == "seg xz" || viewer->surfName() == "seg yz") &&
            manager()) {
            for (auto* other : manager()->baseViewers()) {
                if (other && other != viewer && other->surfName() == "xy plane") {
                    refreshViewer(other);
                }
            }
        }
        return;
    }

    ViewerState& state = ensureViewerState(viewer);
    installInteractions(viewer, state);

    auto* focusPoi = _state->poi("focus");
    if (!focusPoi) {
        clearViewerState(viewer);
        return;
    }

    const cv::Vec3f focus = focusPoi->p;

    // The xy viewer's displayed plane: guide lines mark where the xz/yz view
    // planes cross this plane, not where they cross the focus point.
    cv::Vec3f viewNormal(0.0f, 0.0f, 1.0f);
    cv::Vec3f viewOrigin = focus;
    {
        auto xyHolder = _state->surface("xy plane");
        if (auto* xyPlane = dynamic_cast<PlaneSurface*>(xyHolder.get())) {
            viewNormal = xyPlane->normal({}, {});
            if (cv::norm(viewNormal) < 1e-5f) {
                viewNormal = cv::Vec3f(0.0f, 0.0f, 1.0f);
            }
            cv::normalize(viewNormal, viewNormal);
            viewOrigin = xyPlane->origin() + viewNormal * viewer->normalOffset();
        }
    }

    const struct {
        const char* name;
        cv::Vec3f baseNormal;
        const QColor lineColor;
    } planeDefs[] = {
        {"seg xz", {0.0f, 1.0f, 0.0f}, kXZColor},
        {"seg yz", {1.0f, 0.0f, 0.0f}, kYZColor},
    };

    for (const auto& def : planeDefs) {
        auto planeHolder = _state->surface(def.name);  // Keep surface alive during this iteration
        auto* plane = dynamic_cast<PlaneSurface*>(planeHolder.get());
        if (!plane) {
            continue;
        }

        // The viewer displaying this plane contributes a scroll offset along
        // the normal and, when plane compositing is on, a slab around it.
        VolumeViewerBase* srcViewer = nullptr;
        if (manager()) {
            for (auto* candidate : manager()->baseViewers()) {
                if (candidate && candidate->surfName() == def.name) {
                    srcViewer = candidate;
                    break;
                }
            }
        }

        cv::Vec3f normal = plane->normal({}, {});
        if (cv::norm(normal) < 1e-5f) {
            continue;
        }
        cv::normalize(normal, normal);
        const cv::Vec3f planeOrigin =
            plane->origin() + normal * (srcViewer ? srcViewer->normalOffset() : 0.0f);

        cv::Vec3f dir3D = normal.cross(viewNormal);
        if (cv::norm(dir3D) < 1e-5f) {
            continue;
        }
        cv::normalize(dir3D, dir3D);

        // Line center: the point on both displayed planes nearest the focus.
        // Start from the focus dropped onto the view plane, then slide within
        // the view plane (perpendicular to the line) onto the sliced plane.
        cv::Vec3f origin = focus + viewNormal * (viewOrigin - focus).dot(viewNormal);
        const cv::Vec3f inViewPerp = viewNormal.cross(dir3D);
        const float perpDotNormal = inViewPerp.dot(normal);
        if (std::abs(perpDotNormal) > 1e-6f) {
            origin += inViewPerp * ((planeOrigin - origin).dot(normal) / perpDotNormal);
        }
        const cv::Vec3f dirXY = dir3D;

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

        QPointF positiveScene = builder.viewer()->volumeToScene(positivePoint);
        QPointF negativeScene = builder.viewer()->volumeToScene(negativePoint);

        QColor lineColor = def.lineColor;
        lineColor.setAlphaF(_overlayOpacity);

        OverlayStyle lineStyle;
        lineStyle.penColor = lineColor;
        lineStyle.penWidth = 2.0;
        lineStyle.penStyle = kLineStyle;
        lineStyle.z = kLineZ;

        builder.addLineStrip({negativeScene, positiveScene}, false, lineStyle);

        // When the plane's viewer composites a slab, mark its front/behind
        // extents with thinner dashed lines; volumetric compositing views the
        // slab from the front bound, so mark that side with triangles
        // pointing into the slab (mirrors the flattened-view slab bounds).
        if (srcViewer && srcViewer->isPlaneCompositeEnabled()) {
            const auto& cs = srcViewer->compositeRenderSettings();
            const float step = cs.planeReverseDirection ? -1.0f : 1.0f;
            const float slabFront = float(std::max(0, cs.planeLayersFront)) * step;
            const float slabBehind = -float(std::max(0, cs.planeLayersBehind)) * step;
            const bool volumetric = cs.params.method == "volumetric";

            OverlayStyle slabStyle = lineStyle;
            slabStyle.penWidth = 1.0;

            for (float slabOffset : {slabFront, slabBehind}) {
                if (std::abs(slabOffset) < 1e-3f) {
                    continue;
                }
                const cv::Vec3f slabOrigin = origin + normal * slabOffset;
                const QPointF a = builder.viewer()->volumeToScene(slabOrigin - dirXY * span);
                const QPointF b = builder.viewer()->volumeToScene(slabOrigin + dirXY * span);
                builder.addLineStrip({a, b}, false, slabStyle);
            }

            if (volumetric && std::abs(slabFront - slabBehind) > 1e-3f) {
                // Match the flattened-view slab markers: triangles at a fixed
                // scene-pixel spacing across the whole visible extent of the
                // line, not just around the handles.
                constexpr qreal kMarkerSpacingScenePx = 48.0;
                constexpr qreal kMarkerSizeScenePx = 8.0;
                OverlayStyle markerStyle;
                markerStyle.penColor = Qt::transparent;
                markerStyle.brushColor = lineColor;
                markerStyle.z = kLineZ;

                // The line is affine in the parameter s (voxels along dirXY):
                // clip it against the viewport in scene space, then convert
                // the pixel spacing back into a step in s.
                const QPointF sceneAt0 = builder.viewer()->volumeToScene(origin);
                const QPointF sceneAt1 = builder.viewer()->volumeToScene(origin + dirXY);
                const QPointF scenePerS = sceneAt1 - sceneAt0;
                const qreal pxPerS = std::hypot(scenePerS.x(), scenePerS.y());
                auto* gv = viewer->graphicsView();
                qreal sMin = -span;
                qreal sMax = span;
                bool visible = pxPerS > 1e-6 && gv;
                if (visible) {
                    // Liang-Barsky clip of sceneAt0 + s*scenePerS against the
                    // viewport rect.
                    const QRectF viewRect =
                        gv->mapToScene(gv->viewport()->rect()).boundingRect();
                    const qreal p[4] = {-scenePerS.x(), scenePerS.x(), -scenePerS.y(), scenePerS.y()};
                    const qreal q[4] = {sceneAt0.x() - viewRect.left(), viewRect.right() - sceneAt0.x(),
                                        sceneAt0.y() - viewRect.top(), viewRect.bottom() - sceneAt0.y()};
                    for (int e = 0; e < 4 && visible; ++e) {
                        if (std::abs(p[e]) < 1e-9) {
                            visible = q[e] >= 0;
                            continue;
                        }
                        const qreal t = q[e] / p[e];
                        if (p[e] < 0) sMin = std::max(sMin, t);
                        else sMax = std::min(sMax, t);
                    }
                    visible = visible && sMin <= sMax;
                }
                const qreal sStep = visible ? kMarkerSpacingScenePx / pxPerS : 1.0;
                // Anchor the grid at s=0 so markers don't crawl while panning.
                const qreal sStart = std::ceil(sMin / sStep) * sStep;
                for (qreal s = sStart; visible && s <= sMax; s += sStep) {
                    const cv::Vec3f base = origin + dirXY * float(s);
                    const QPointF from = builder.viewer()->volumeToScene(base + normal * slabFront);
                    const QPointF toward = builder.viewer()->volumeToScene(base + normal * slabBehind);
                    QPointF dir = toward - from;
                    const qreal dirLen = std::hypot(dir.x(), dir.y());
                    if (dirLen < 1e-6) {
                        continue;
                    }
                    dir /= dirLen;
                    const QPointF perp(-dir.y(), dir.x());
                    builder.addLineStrip({from + perp * (kMarkerSizeScenePx * 0.5),
                                          from - perp * (kMarkerSizeScenePx * 0.5),
                                          from + dir * kMarkerSizeScenePx},
                                         true, markerStyle);
                }
            }
        }

        cv::Vec3f handleOffset3D = dirXY * kHandleVolumeOffset;
        cv::Vec3f handlePositive = origin + handleOffset3D;
        cv::Vec3f handleNegative = origin - handleOffset3D;

        QPointF handlePositiveScene = builder.viewer()->volumeToScene(handlePositive);
        QPointF handleNegativeScene = builder.viewer()->volumeToScene(handleNegative);

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

void PlaneSlicingOverlayController::handleMousePress(VolumeViewerBase* viewer,
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
    QPointF scenePoint = viewer->volumeToScene(volumePoint);

    for (auto& entry : state.planes) {
        const std::string& planeName = entry.first;
        PlaneVisual& visual = entry.second;

        bool onPositive = pointInsideHandle(scenePoint, visual.handlePositiveScene, kHandleRadius);
        bool onNegative = pointInsideHandle(scenePoint, visual.handleNegativeScene, kHandleRadius);
        if (onPositive || onNegative) {
            _activeDrag.viewer = viewer;
            _activeDrag.planeName = planeName;
            _activeDrag.positiveHandle = onPositive;
            if (auto* gv = viewer->graphicsView()) gv->setCursor(Qt::ClosedHandCursor);
            break;
        }
    }
}

void PlaneSlicingOverlayController::handleMouseMove(VolumeViewerBase* viewer,
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
    QPointF scenePoint = viewer->volumeToScene(volumePoint);

    if (_activeDrag.viewer == viewer && !_activeDrag.planeName.empty()) {
        if (!(buttons & Qt::LeftButton)) {
            return;
        }

        auto planeIt = state.planes.find(_activeDrag.planeName);
        if (planeIt == state.planes.end()) {
            return;
        }

        if (!_state || !_rotationSetter) {
            return;
        }

        const PlaneVisual& visual = planeIt->second;

        // Rotate about the drawn line center (the plane's current position in
        // this view), which may sit away from the focus point.
        cv::Vec3f delta = volumePoint - visual.origin;
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
        auto planeSurfaceHolder = _state->surface(_activeDrag.planeName);  // Keep surface alive
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
        refreshViewer(viewer);
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

    if (auto* gv = viewer->graphicsView()) gv->setCursor(hoveringHandle ? Qt::OpenHandCursor : Qt::ArrowCursor);
}

void PlaneSlicingOverlayController::handleMouseRelease(VolumeViewerBase* viewer,
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
        if (auto* gv = viewer->graphicsView()) gv->setCursor(Qt::ArrowCursor);
        if (hadActiveDrag && _rotationFinishedCallback) {
            _rotationFinishedCallback();
        }
    }
}

bool PlaneSlicingOverlayController::isScenePointNearRotationHandle(VolumeViewerBase* viewer,
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

bool PlaneSlicingOverlayController::isVolumePointNearRotationHandle(VolumeViewerBase* viewer,
                                                                    const cv::Vec3f& volumePoint,
                                                                    qreal radiusScale) const
{
    if (!viewer) {
        return false;
    }
    const QPointF scenePoint = viewer->volumeToScene(volumePoint);
    return isScenePointNearRotationHandle(viewer, scenePoint, radiusScale);
}

float PlaneSlicingOverlayController::normalizeDegrees(float degrees)
{
    if (!std::isfinite(degrees)) {
        return 0.0f;
    }
    return std::remainder(degrees, 360.0f);
}
