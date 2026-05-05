#pragma once

#include <QObject>
#include <QPointF>
#include <opencv2/core.hpp>
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

    void setViewerManager(ViewerManager* mgr) { _viewerManager = mgr; }
    void setPlaneSlicingOverlay(PlaneSlicingOverlayController* overlay) { _planeSlicingOverlay = overlay; }

    bool isEnabled() const { return _enabled; }

    // Called when user toggles the checkbox
    // overlayCheckbox and overlayOpacitySpin are passed for UI state management
    void setEnabled(bool enabled, QCheckBox* overlayCheckbox = nullptr, QSpinBox* overlayOpacitySpin = nullptr);

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

    void processOrientationUpdate();
    void notifyInteractiveOrientationViewers(double motionPx);
    void updateTiltHandles();
};
