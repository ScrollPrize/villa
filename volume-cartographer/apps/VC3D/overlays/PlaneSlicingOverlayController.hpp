#pragma once

#include "ViewerOverlayControllerBase.hpp"

#include <QMetaObject>

#include <functional>
#include <string>
#include <unordered_map>

class CState;

class PlaneSlicingOverlayController : public ViewerOverlayControllerBase
{
    Q_OBJECT

public:
    explicit PlaneSlicingOverlayController(CState* state, QObject* parent = nullptr);

    void setAxisAlignedEnabled(bool enabled);
    void setRotationSetter(std::function<void(const std::string&, float)> setter);
    void setAxisAlignedOverlayOpacity(float opacity);
    void setRotationFinishedCallback(std::function<void()> callback);
    float axisAlignedOverlayOpacity() const { return _overlayOpacity; }

    bool isVolumePointNearRotationHandle(VolumeViewerBase* viewer,
                                         const cv::Vec3f& volumePoint,
                                         qreal radiusScale = 1.5) const;
    bool isScenePointNearRotationHandle(VolumeViewerBase* viewer,
                                        const QPointF& scenePoint,
                                        qreal radiusScale = 1.5) const;

protected:
    bool isOverlayEnabledFor(VolumeViewerBase* viewer) const override;
    void collectPrimitives(VolumeViewerBase* viewer, OverlayBuilder& builder) override;

private:
    struct PlaneVisual {
        cv::Vec3f origin;
        cv::Vec3f directionXY; // normalized 2D direction embedded in XY plane
        cv::Vec3f handlePositiveVolume;
        cv::Vec3f handleNegativeVolume;
        QPointF handlePositiveScene;
        QPointF handleNegativeScene;
        float baseAngleDegrees{0.0f};
    };

    struct ViewerState {
        std::unordered_map<std::string, PlaneVisual> planes;
        bool interactionsInstalled{false};
        QMetaObject::Connection pressConn;
        QMetaObject::Connection moveConn;
        QMetaObject::Connection releaseConn;
        QMetaObject::Connection destroyedConn;
    };

    void installInteractions(VolumeViewerBase* viewer, ViewerState& state);
    void removeInteractions(VolumeViewerBase* viewer);

    void handleMousePress(VolumeViewerBase* viewer,
                          const cv::Vec3f& volumePoint,
                          Qt::MouseButton button,
                          Qt::KeyboardModifiers modifiers);
    void handleMouseMove(VolumeViewerBase* viewer,
                         const cv::Vec3f& volumePoint,
                         Qt::MouseButtons buttons,
                         Qt::KeyboardModifiers modifiers);
    void handleMouseRelease(VolumeViewerBase* viewer,
                            Qt::MouseButton button,
                            Qt::KeyboardModifiers modifiers);

    ViewerState& ensureViewerState(VolumeViewerBase* viewer);
    void updateViewerState(VolumeViewerBase* viewer,
                           ViewerState& state,
                           const std::string& planeName,
                           const PlaneVisual& visual);
    void clearViewerState(VolumeViewerBase* viewer);

    static float normalizeDegrees(float degrees);

    CState* _state{nullptr};
    bool _axisAlignedEnabled{false};
    std::function<void(const std::string&, float)> _rotationSetter;
    std::function<void()> _rotationFinishedCallback;
    float _overlayOpacity{0.7f};

    mutable std::unordered_map<VolumeViewerBase*, ViewerState> _viewerStates;

    struct ActiveDragState {
        VolumeViewerBase* viewer{nullptr};
        std::string planeName;
        bool positiveHandle{true};
    } _activeDrag;
};
