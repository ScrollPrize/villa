#pragma once

#include <QObject>
#include <QPointF>
#include <opencv2/core/mat.hpp>
#include <string>
#include <unordered_map>

class QTimer;
class QCheckBox;
class QSpinBox;
class CState;
class ViewerManager;
class PlaneSlicingOverlayController;
class VolumeViewerBase;
class Surface;

class AxisAlignedSliceController : public QObject
{
    Q_OBJECT

public:
    explicit AxisAlignedSliceController(CState* state, QObject* parent = nullptr);

    void setViewerManager(ViewerManager* mgr);
    void setPlaneSlicingOverlay(PlaneSlicingOverlayController* overlay) { _planeSlicingOverlay = overlay; }

    bool isEnabled() const { return _enabled; }

    // Called when user toggles the checkbox
    // overlayCheckbox and overlayOpacitySpin are passed for UI state management.
    // persistSetting=false applies the mode without writing the global setting
    // (for workspaces that force a mode rather than reflect the user's choice).
    void setEnabled(bool enabled, QCheckBox* overlayCheckbox = nullptr, QSpinBox* overlayOpacitySpin = nullptr,
                    bool persistSetting = true);

    void resetRotations();
    void resetTilt();
    void resetAll();

    // Mouse event handlers for rotation dragging
    void onMousePress(VolumeViewerBase* viewer, const cv::Vec3f& volLoc, Qt::MouseButton button, Qt::KeyboardModifiers modifiers);
    void onMouseMove(VolumeViewerBase* viewer, const cv::Vec3f& volLoc, Qt::MouseButtons buttons, Qt::KeyboardModifiers modifiers);
    void onMouseRelease(VolumeViewerBase* viewer, Qt::MouseButton button, Qt::KeyboardModifiers modifiers);
    void onTiltHandleChanged(VolumeViewerBase* viewer, QPointF tilt);
    void onTiltHandleReset();

    // Apply slice plane orientations based on current state
    void applyOrientation(Surface* sourceOverride = nullptr);

    // Schedule/flush orientation updates (with debounce timer)
    void scheduleOrientationUpdate();
    void flushOrientationUpdate();
    void cancelOrientationTimer();

    // Update viewer middle-button pan interaction
    void updateSliceInteraction();

    float currentRotationDegrees(const std::string& surfaceName) const;
    void setRotationDegrees(const std::string& surfaceName, float degrees);

    // Volumetric-camera azimuth for a slice plane, read from the owning
    // viewer's per-view camera (0 when that viewer's volumetric mode is
    // inactive). applyOrientation folds it into the plane's in-plane
    // rotation, so the volumetric compositor only applies the tilt and every
    // plane-derived overlay/interaction sees the rotated view.
    float volumetricAzimuthDeg(const std::string& surfaceName) const;
    // Reconfigure the slice planes iff some view's volumetric azimuth differs
    // from what applyOrientation last folded in. Cheap no-op otherwise, so
    // it can hang off every per-view camera change.
    void syncVolumetricAzimuths();
    // Rotation-about-normal term that spins the plane's basis by azimuthDeg
    // in *screen* space (vx -> cos·vx − sin·vy), correcting for the basis
    // handedness so azimuth turns the same way on every plane — matching the
    // spin the volumetric compositor applies when it handles azimuth itself.
    static float azimuthInPlaneRotation(class PlaneSurface& plane, float azimuthDeg);

    static float normalizeDegrees(float degrees);

private:
    CState* _state;
    ViewerManager* _viewerManager{nullptr};
    PlaneSlicingOverlayController* _planeSlicingOverlay{nullptr};

    bool _enabled{false};
    float _segXZRotationDeg{0.0f};
    float _segYZRotationDeg{0.0f};
    QPointF _xyTilt{0.0, 0.0};
    double _segXZTilt{0.0};
    double _segYZTilt{0.0};

    struct DragState {
        bool active = false;
        QPointF startScenePos;
        float startRotationDegrees = 0.0f;
    };
    std::unordered_map<const VolumeViewerBase*, DragState> _drags;

    QTimer* _rotationTimer{nullptr};
    bool _orientationDirty{false};
    double _pendingOrientationMotionPx{0.0};
    // Azimuth last folded into each plane by applyOrientation, keyed by
    // surface name; lets syncVolumetricAzimuths skip no-op reconfigures.
    std::unordered_map<std::string, float> _appliedVolumetricAzimuthDeg;

    void processOrientationUpdate();
    void notifyInteractiveOrientationViewers(double motionPx);
    void updateTiltHandles();
    void attachViewer(VolumeViewerBase* baseViewer);
};
