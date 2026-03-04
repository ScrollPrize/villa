#include "AxisAlignedSliceController.hpp"
#include "CState.hpp"
#include "ViewerManager.hpp"
#include "overlays/PlaneSlicingOverlayController.hpp"
#include "tiled/CTiledVolumeViewer.hpp"
#include "CVolumeViewerView.hpp"
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

void AxisAlignedSliceController::onMousePress(CTiledVolumeViewer* viewer, const cv::Vec3f& volLoc, Qt::MouseButton button, Qt::KeyboardModifiers)
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

void AxisAlignedSliceController::onMouseMove(CTiledVolumeViewer* viewer, const cv::Vec3f& volLoc, Qt::MouseButtons buttons, Qt::KeyboardModifiers)
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
    scheduleOrientationUpdate();
}

void AxisAlignedSliceController::onMouseRelease(CTiledVolumeViewer* viewer, Qt::MouseButton button, Qt::KeyboardModifiers)
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

    // Helper to configure a plane with optional yaw rotation
    const auto configurePlane = [&](const std::string& planeName,
                                    const cv::Vec3f& baseNormal,
                                    float yawDeg = 0.0f) {
        auto planeShared = std::dynamic_pointer_cast<PlaneSurface>(_state->surface(planeName));
        if (!planeShared) {
            planeShared = std::make_shared<PlaneSurface>();
        }

        planeShared->setOrigin(origin);
        planeShared->setInPlaneRotation(0.0f);

        // Apply yaw rotation if set
        cv::Vec3f rotatedNormal;
        if (std::abs(yawDeg) > 0.001f) {
            const float radians = yawDeg * kDegToRad;
            rotatedNormal = rotateAroundZ(baseNormal, radians);
        } else {
            rotatedNormal = baseNormal;
        }

        planeShared->setNormal(rotatedNormal);

        // Adjust in-plane rotation so "up" is aligned with volume Z when possible
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

        _state->setSurface(planeName, planeShared);
        return planeShared;
    };

    // Always update the XY plane
    auto xyPlane = configurePlane("xy plane", cv::Vec3f(0.0f, 0.0f, 1.0f));

    if (_enabled) {
        auto segXZShared = configurePlane("seg xz", cv::Vec3f(0.0f, 1.0f, 0.0f), _segXZRotationDeg);
        auto segYZShared = configurePlane("seg yz", cv::Vec3f(1.0f, 0.0f, 0.0f), _segYZRotationDeg);

        if (_planeSlicingOverlay) {
            _planeSlicingOverlay->refreshAll();
        }
        return;
    } else {
        QuadSurface* segment = nullptr;
        std::shared_ptr<Surface> segmentHolder;  // Keep surface alive during this scope
        if (sourceOverride) {
            segment = dynamic_cast<QuadSurface*>(sourceOverride);
        } else {
            segmentHolder = _state->surface("segmentation");
            segment = dynamic_cast<QuadSurface*>(segmentHolder.get());
        }
        if (!segment) {
            return;
        }

        auto segXZShared = std::dynamic_pointer_cast<PlaneSurface>(_state->surface("seg xz"));
        auto segYZShared = std::dynamic_pointer_cast<PlaneSurface>(_state->surface("seg yz"));

        if (!segXZShared) {
            segXZShared = std::make_shared<PlaneSurface>();
        }
        if (!segYZShared) {
            segYZShared = std::make_shared<PlaneSurface>();
        }

        segXZShared->setOrigin(origin);
        segYZShared->setOrigin(origin);

        cv::Vec3f ptr(0, 0, 0);
        auto* patchIndex = _viewerManager ? _viewerManager->surfacePatchIndex() : nullptr;
        segment->pointTo(ptr, origin, 1.0f, 1000, patchIndex);

        cv::Vec3f xDir = segment->coord(ptr, {1, 0, 0});
        cv::Vec3f yDir = segment->coord(ptr, {0, 1, 0});
        segXZShared->setNormal(xDir - origin);
        segYZShared->setNormal(yDir - origin);
        segXZShared->setInPlaneRotation(0.0f);
        segYZShared->setInPlaneRotation(0.0f);

        _state->setSurface("seg xz", segXZShared);
        _state->setSurface("seg yz", segYZShared);
        if (_planeSlicingOverlay) {
            _planeSlicingOverlay->refreshAll();
        }
        return;
    }
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
    cancelOrientationTimer();
    applyOrientation();
}

void AxisAlignedSliceController::processOrientationUpdate()
{
    if (!_orientationDirty) {
        return;
    }
    _orientationDirty = false;
    applyOrientation();
}

void AxisAlignedSliceController::cancelOrientationTimer()
{
    if (_rotationTimer && _rotationTimer->isActive()) {
        _rotationTimer->stop();
    }
    _orientationDirty = false;
}

void AxisAlignedSliceController::updateSliceInteraction()
{
    if (!_viewerManager) {
        return;
    }

    _viewerManager->forEachViewer([this](CTiledVolumeViewer* viewer) {
        if (!viewer || !viewer->fGraphicsView) {
            return;
        }
        const std::string& name = viewer->surfName();
        if (name == "seg xz" || name == "seg yz") {
            viewer->fGraphicsView->setMiddleButtonPanEnabled(!_enabled);
            qCDebug(lcAxisSlices2) << "Middle-button pan set" << QString::fromStdString(name)
                                   << "enabled" << viewer->fGraphicsView->middleButtonPanEnabled();
        }
    });
}
