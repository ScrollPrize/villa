#include "AxisAlignedSliceController.hpp"
#include "CState.hpp"
#include "ViewerManager.hpp"
#include "volume_viewers/VolumeViewerBase.hpp"
#include "volume_viewers/CChunkedVolumeViewer.hpp"
#include "overlays/PlaneSlicingOverlayController.hpp"
#include "volume_viewers/CVolumeViewerView.hpp"
#include "VCSettings.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include <QTimer>
#include <QCheckBox>
#include <QSpinBox>
#include <QSettings>
#include <QLoggingCategory>
#include <cmath>
#include <opencv2/core.hpp>

Q_LOGGING_CATEGORY(lcAxisSlices2, "vc.axis_aligned");

namespace
{
constexpr float kAxisRotationDegreesPerScenePixel = 0.25f;
constexpr float kMaxTiltDegrees = 45.0f;
constexpr float kEpsilon = 1e-6f;
constexpr float kDegToRad = static_cast<float>(CV_PI / 180.0);
constexpr int kRotationApplyDelayMs = 25;

cv::Vec3f rotateAroundZ(const cv::Vec3f& v, float radians)
{
    const float c = std::cos(radians);
    const float s = std::sin(radians);
    return {
        v[0] * c - v[1] * s,
        v[0] * s + v[1] * c,
        v[2]
    };
}

cv::Vec3f rotateAroundAxis(const cv::Vec3f& v, const cv::Vec3f& axis, float radians)
{
    const float axisMagnitude = cv::norm(axis);
    if (axisMagnitude <= kEpsilon) {
        return v;
    }
    const cv::Vec3f a = axis * (1.0f / axisMagnitude);
    const float c = std::cos(radians);
    const float s = std::sin(radians);
    return v * c + a.cross(v) * s + a * (a.dot(v) * (1.0f - c));
}

cv::Vec3f projectVectorOntoPlane(const cv::Vec3f& v, const cv::Vec3f& normal)
{
    const float dot = v.dot(normal);
    return v - normal * dot;
}

cv::Vec3f normalizeOrZero(const cv::Vec3f& v)
{
    const float magnitude = cv::norm(v);
    if (magnitude <= kEpsilon) {
        return cv::Vec3f(0.0f, 0.0f, 0.0f);
    }
    return v * (1.0f / magnitude);
}

float signedAngleBetween(const cv::Vec3f& from, const cv::Vec3f& to, const cv::Vec3f& axis)
{
    cv::Vec3f fromNorm = normalizeOrZero(from);
    cv::Vec3f toNorm = normalizeOrZero(to);
    if (cv::norm(fromNorm) <= kEpsilon || cv::norm(toNorm) <= kEpsilon) {
        return 0.0f;
    }

    float dot = fromNorm.dot(toNorm);
    dot = std::clamp(dot, -1.0f, 1.0f);
    cv::Vec3f cross = fromNorm.cross(toNorm);
    float angle = std::atan2(cv::norm(cross), dot);
    float sign = cross.dot(axis) >= 0.0f ? 1.0f : -1.0f;
    return angle * sign;
}
} // namespace

AxisAlignedSliceController::AxisAlignedSliceController(CState* state, QObject* parent)
    : QObject(parent)
    , _state(state)
{
    _rotationTimer = new QTimer(this);
    _rotationTimer->setSingleShot(true);
    _rotationTimer->setInterval(kRotationApplyDelayMs);
    connect(_rotationTimer, &QTimer::timeout, this, &AxisAlignedSliceController::processOrientationUpdate);
}

void AxisAlignedSliceController::setEnabled(bool enabled, QCheckBox* overlayCheckbox, QSpinBox* overlayOpacitySpin)
{
    _enabled = enabled;
    if (enabled) {
        _segXZRotationDeg = 0.0f;
        _segYZRotationDeg = 0.0f;
        _xyTilt = QPointF(0.0, 0.0);
        _segXZTilt = 0.0;
        _segYZTilt = 0.0;
    }
    _drags.clear();
    qCDebug(lcAxisSlices2) << "Axis-aligned slices" << (enabled ? "enabled" : "disabled");
    if (_planeSlicingOverlay) {
        bool overlaysVisible = !overlayCheckbox || overlayCheckbox->isChecked();
        _planeSlicingOverlay->setAxisAlignedEnabled(enabled && overlaysVisible);
    }
    if (overlayOpacitySpin) {
        overlayOpacitySpin->setEnabled(enabled && (!overlayCheckbox || overlayCheckbox->isChecked()));
    }
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue(vc3d::settings::viewer::USE_AXIS_ALIGNED_SLICES, enabled ? "1" : "0");
    updateSliceInteraction();
    applyOrientation();
}

void AxisAlignedSliceController::resetRotations()
{
    _segXZRotationDeg = 0.0f;
    _segYZRotationDeg = 0.0f;
    applyOrientation();
}

void AxisAlignedSliceController::resetTilt()
{
    _xyTilt = QPointF(0.0, 0.0);
    _segXZTilt = 0.0;
    _segYZTilt = 0.0;
    applyOrientation();
}

void AxisAlignedSliceController::resetAll()
{
    _segXZRotationDeg = 0.0f;
    _segYZRotationDeg = 0.0f;
    _xyTilt = QPointF(0.0, 0.0);
    _segXZTilt = 0.0;
    _segYZTilt = 0.0;
    _drags.clear();
    cancelOrientationTimer();
}

void AxisAlignedSliceController::onTiltHandleChanged(VolumeViewerBase* viewer, QPointF tilt)
{
    if (!_enabled || !viewer) {
        return;
    }

    const std::string surfaceName = viewer->surfName();
    if (surfaceName == "xy plane") {
        const double len = std::hypot(tilt.x(), tilt.y());
        if (len > 1.0) {
            tilt /= len;
        }
        _xyTilt = tilt;
    } else if (surfaceName == "seg xz") {
        _segXZTilt = std::clamp(tilt.x(), -1.0, 1.0);
    } else if (surfaceName == "seg yz") {
        _segYZTilt = std::clamp(tilt.y(), -1.0, 1.0);
    } else {
        return;
    }

    _pendingOrientationMotionPx = std::max(_pendingOrientationMotionPx, 96.0);
    scheduleOrientationUpdate();
    updateTiltHandles();
}

void AxisAlignedSliceController::onTiltHandleReset()
{
    resetTilt();
}

void AxisAlignedSliceController::onMousePress(VolumeViewerBase* viewer, const cv::Vec3f& volLoc, Qt::MouseButton button, Qt::KeyboardModifiers)
{
    if (!_enabled || button != Qt::MiddleButton || !viewer) {
        return;
    }

    const std::string surfaceName = viewer->surfName();
    if (surfaceName != "seg xz" && surfaceName != "seg yz") {
        return;
    }

    DragState& state = _drags[viewer];
    state.active = true;
    state.startScenePos = viewer->volumeToScene(volLoc);
    state.startRotationDegrees = currentRotationDegrees(surfaceName);
}

void AxisAlignedSliceController::onMouseMove(VolumeViewerBase* viewer, const cv::Vec3f& volLoc, Qt::MouseButtons buttons, Qt::KeyboardModifiers)
{
    if (!_enabled || !viewer || !(buttons & Qt::MiddleButton)) {
        return;
    }

    const std::string surfaceName = viewer->surfName();
    if (surfaceName != "seg xz" && surfaceName != "seg yz") {
        return;
    }

    auto it = _drags.find(viewer);
    if (it == _drags.end() || !it->second.active) {
        return;
    }

    DragState& state = it->second;
    QPointF currentScenePos = viewer->volumeToScene(volLoc);
    const float dragPixels = static_cast<float>(currentScenePos.y() - state.startScenePos.y());
    const float candidate = normalizeDegrees(state.startRotationDegrees - dragPixels * kAxisRotationDegreesPerScenePixel);
    const float currentRotation = currentRotationDegrees(surfaceName);

    if (std::abs(candidate - currentRotation) < 0.01f) {
        return;
    }

    setRotationDegrees(surfaceName, candidate);
    _pendingOrientationMotionPx = std::max(_pendingOrientationMotionPx, std::abs(double(dragPixels)));
    scheduleOrientationUpdate();
}

void AxisAlignedSliceController::onMouseRelease(VolumeViewerBase* viewer, Qt::MouseButton button, Qt::KeyboardModifiers)
{
    if (button != Qt::MiddleButton) {
        return;
    }

    auto it = _drags.find(viewer);
    if (it != _drags.end()) {
        it->second.active = false;
    }
    flushOrientationUpdate();
}

void AxisAlignedSliceController::applyOrientation(Surface* sourceOverride)
{
    if (!_state) {
        return;
    }

    cancelOrientationTimer();

    POI* focus = _state->poi("focus");
    cv::Vec3f origin = focus ? focus->p : cv::Vec3f(0, 0, 0);

    auto xyNormalFromTilt = [this]() {
        const double tiltLen = _enabled ? std::min(1.0, std::hypot(_xyTilt.x(), _xyTilt.y())) : 0.0;
        if (tiltLen <= kEpsilon) {
            return cv::Vec3f(0.0f, 0.0f, 1.0f);
        }

        const float angle = static_cast<float>(tiltLen) * kMaxTiltDegrees * kDegToRad;
        const cv::Vec3f axis = normalizeOrZero(cv::Vec3f(static_cast<float>(_xyTilt.y()),
                                                         static_cast<float>(-_xyTilt.x()),
                                                         0.0f));
        return normalizeOrZero(rotateAroundAxis({0.0f, 0.0f, 1.0f}, axis, angle));
    };

    // Helper to configure a plane with optional yaw rotation
    const auto configurePlane = [&](const std::string& planeName,
                                    const cv::Vec3f& baseNormal,
                                    float yawDeg = 0.0f,
                                    double tilt = 0.0) {
        auto planeShared = std::dynamic_pointer_cast<PlaneSurface>(_state->surface(planeName));
        if (!planeShared) {
            planeShared = std::make_shared<PlaneSurface>();
        }

        planeShared->setOrigin(origin);
        planeShared->setInPlaneRotation(0.0f);

        cv::Vec3f rotatedNormal = baseNormal;
        if (std::abs(yawDeg) > 0.001f) {
            const float radians = yawDeg * kDegToRad;
            rotatedNormal = rotateAroundZ(baseNormal, radians);
        }

        if (_enabled && std::abs(tilt) > kEpsilon) {
            const cv::Vec3f horizontalNormal(rotatedNormal[0], rotatedNormal[1], 0.0f);
            const cv::Vec3f tiltAxis = normalizeOrZero(horizontalNormal.cross({0.0f, 0.0f, 1.0f}));
            if (cv::norm(tiltAxis) > kEpsilon) {
                const float radians = static_cast<float>(tilt) * kMaxTiltDegrees * kDegToRad;
                rotatedNormal = normalizeOrZero(rotateAroundAxis(rotatedNormal, tiltAxis, radians));
            }
        }

        planeShared->setNormal(rotatedNormal);

        if (planeName == "xy plane") {
            const cv::Vec3f projectedRight = projectVectorOntoPlane({1.0f, 0.0f, 0.0f}, rotatedNormal);
            const cv::Vec3f desiredRight = normalizeOrZero(projectedRight);
            if (cv::norm(desiredRight) > kEpsilon) {
                const cv::Vec3f currentRight = planeShared->basisX();
                const float delta = signedAngleBetween(currentRight, desiredRight, rotatedNormal);
                if (std::abs(delta) > kEpsilon) {
                    planeShared->setInPlaneRotation(delta);
                }
            }
        } else {
            // Adjust in-plane rotation so "up" is aligned with volume Z when possible.
            const cv::Vec3f upAxis(0.0f, 0.0f, 1.0f);
            const cv::Vec3f projectedUp = projectVectorOntoPlane(upAxis, rotatedNormal);
            const cv::Vec3f desiredUp = normalizeOrZero(projectedUp);

            if (cv::norm(desiredUp) > kEpsilon) {
                const cv::Vec3f currentUp = planeShared->basisY();
                const float delta = signedAngleBetween(currentUp, desiredUp, rotatedNormal);
                if (std::abs(delta) > kEpsilon) {
                    planeShared->setInPlaneRotation(delta);
                }
            } else {
                planeShared->setInPlaneRotation(0.0f);
            }
        }

        _state->setSurface(planeName, planeShared);
        return planeShared;
    };

    // Always update the XY plane
    auto xyPlane = configurePlane("xy plane", xyNormalFromTilt());

    // Resolve segment so we can decide whether to fall back to canonical axes.
    QuadSurface* segment = nullptr;
    std::shared_ptr<Surface> segmentHolder;  // Keep surface alive during this scope
    if (sourceOverride) {
        segment = dynamic_cast<QuadSurface*>(sourceOverride);
    } else {
        segmentHolder = _state->surface("segmentation");
        segment = dynamic_cast<QuadSurface*>(segmentHolder.get());
    }

    const bool useCanonical = _enabled || !segment;

    if (useCanonical) {
        configurePlane("seg xz", {0.0f, 1.0f, 0.0f}, _segXZRotationDeg, _segXZTilt);
        configurePlane("seg yz", {1.0f, 0.0f, 0.0f}, _segYZRotationDeg, _segYZTilt);
    } else {
        auto segXZShared = std::dynamic_pointer_cast<PlaneSurface>(_state->surface("seg xz"));
        auto segYZShared = std::dynamic_pointer_cast<PlaneSurface>(_state->surface("seg yz"));

        if (!segXZShared) {
            segXZShared = std::make_shared<PlaneSurface>();
        }
        if (!segYZShared) {
            segYZShared = std::make_shared<PlaneSurface>();
        }

        cv::Vec3f ptr(0, 0, 0);
        auto* patchIndex = _viewerManager ? _viewerManager->surfacePatchIndexIfReady() : nullptr;
        segment->pointTo(ptr, origin, 1.0f, 1000, patchIndex);

        // Use the closest surface point as origin for the slicing planes,
        // not the raw focus point — ensures normals are true surface tangents
        cv::Vec3f surfOrigin = segment->coord(ptr, {0, 0, 0});
        segXZShared->setOrigin(surfOrigin);
        segYZShared->setOrigin(surfOrigin);

        cv::Vec3f xDir = segment->coord(ptr, {1, 0, 0});
        cv::Vec3f yDir = segment->coord(ptr, {0, 1, 0});
        segXZShared->setNormal(xDir - surfOrigin);
        segYZShared->setNormal(yDir - surfOrigin);
        segXZShared->setInPlaneRotation(0.0f);
        segYZShared->setInPlaneRotation(0.0f);

        _state->setSurface("seg xz", segXZShared);
        _state->setSurface("seg yz", segYZShared);
    }

    if (_planeSlicingOverlay) {
        _planeSlicingOverlay->refreshAll();
    }
    updateTiltHandles();
}

float AxisAlignedSliceController::normalizeDegrees(float degrees)
{
    if (!std::isfinite(degrees)) {
        return 0.0f;
    }
    return std::remainder(degrees, 360.0f);
}

float AxisAlignedSliceController::currentRotationDegrees(const std::string& surfaceName) const
{
    if (surfaceName == "seg xz") {
        return _segXZRotationDeg;
    }
    if (surfaceName == "seg yz") {
        return _segYZRotationDeg;
    }
    return 0.0f;
}

void AxisAlignedSliceController::setRotationDegrees(const std::string& surfaceName, float degrees)
{
    const float normalized = normalizeDegrees(degrees);
    if (surfaceName == "seg xz") {
        _segXZRotationDeg = normalized;
    } else if (surfaceName == "seg yz") {
        _segYZRotationDeg = normalized;
    }
}

void AxisAlignedSliceController::scheduleOrientationUpdate()
{
    if (!_enabled) {
        applyOrientation();
        return;
    }
    _orientationDirty = true;
    if (!_rotationTimer) {
        applyOrientation();
        return;
    }
    if (!_rotationTimer->isActive()) {
        _rotationTimer->start(kRotationApplyDelayMs);
    }
}

void AxisAlignedSliceController::flushOrientationUpdate()
{
    if (!_orientationDirty) {
        return;
    }
    const double motionPx = _pendingOrientationMotionPx;
    cancelOrientationTimer();
    applyOrientation();
    notifyInteractiveOrientationViewers(motionPx);
    _pendingOrientationMotionPx = 0.0;
}

void AxisAlignedSliceController::processOrientationUpdate()
{
    if (!_orientationDirty) {
        return;
    }
    const double motionPx = _pendingOrientationMotionPx;
    _orientationDirty = false;
    _pendingOrientationMotionPx = 0.0;
    applyOrientation();
    notifyInteractiveOrientationViewers(motionPx);
}

void AxisAlignedSliceController::cancelOrientationTimer()
{
    if (_rotationTimer && _rotationTimer->isActive()) {
        _rotationTimer->stop();
    }
    _orientationDirty = false;
}

void AxisAlignedSliceController::notifyInteractiveOrientationViewers(double motionPx)
{
    if (!_viewerManager) {
        return;
    }

    const double clampedMotionPx = std::max(0.0, motionPx);
    _viewerManager->forEachBaseViewer([clampedMotionPx](VolumeViewerBase* viewer) {
        if (!viewer) {
            return;
        }
        const std::string name = viewer->surfName();
        if (name == "xy plane" || name == "seg xz" || name == "seg yz") {
            if (auto* chunkedViewer = dynamic_cast<CChunkedVolumeViewer*>(viewer)) {
                chunkedViewer->notifyInteractiveViewChange(clampedMotionPx);
            }
        }
    });
}

void AxisAlignedSliceController::updateSliceInteraction()
{
    if (!_viewerManager) {
        return;
    }

    _viewerManager->forEachBaseViewer([this](VolumeViewerBase* viewer) {
        auto* graphicsView = viewer ? viewer->graphicsView() : nullptr;
        if (!viewer || !graphicsView) {
            return;
        }
        const std::string& name = viewer->surfName();
        if (name == "seg xz" || name == "seg yz") {
            graphicsView->setMiddleButtonPanEnabled(!_enabled);
            qCDebug(lcAxisSlices2) << "Middle-button pan set" << QString::fromStdString(name)
                                   << "enabled" << graphicsView->middleButtonPanEnabled();
        }
    });
    updateTiltHandles();
}

void AxisAlignedSliceController::updateTiltHandles()
{
    if (!_viewerManager) {
        return;
    }

    _viewerManager->forEachBaseViewer([this](VolumeViewerBase* viewer) {
        auto* graphicsView = viewer ? viewer->graphicsView() : nullptr;
        if (!viewer || !graphicsView) {
            return;
        }

        if (!graphicsView->property("vc_tilt_handle_bound").toBool()) {
            connect(graphicsView, &CVolumeViewerView::sendTiltHandleChanged,
                    this, [this, viewer](QPointF tilt) {
                        onTiltHandleChanged(viewer, tilt);
                    });
            connect(graphicsView, &CVolumeViewerView::sendTiltHandleReset,
                    this, [this]() {
                        onTiltHandleReset();
                    });
            graphicsView->setProperty("vc_tilt_handle_bound", true);
        }

        const std::string& name = viewer->surfName();
        auto mode = CVolumeViewerView::TiltHandleMode::Hidden;
        QPointF value = _xyTilt;
        if (_enabled && name == "xy plane") {
            mode = CVolumeViewerView::TiltHandleMode::Square;
        } else if (_enabled && name == "seg xz") {
            mode = CVolumeViewerView::TiltHandleMode::SemiCircleX;
            value = QPointF(_segXZTilt, 0.0);
        } else if (_enabled && name == "seg yz") {
            mode = CVolumeViewerView::TiltHandleMode::SemiCircleY;
            value = QPointF(0.0, _segYZTilt);
        }
        graphicsView->setTiltHandle(mode, value);
    });
}
