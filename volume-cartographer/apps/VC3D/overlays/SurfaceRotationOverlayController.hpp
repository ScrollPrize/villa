#pragma once

#include <QMetaObject>
#include <QObject>
#include <QPointer>

#include <memory>
#include <string>
#include <vector>

class CState;
class QuadSurface;
class QGraphicsProxyWidget;
class ViewerManager;
class VolumeViewerBase;

class SurfaceRotationOverlayController : public QObject
{
public:
    explicit SurfaceRotationOverlayController(CState* state, QObject* parent = nullptr);
    ~SurfaceRotationOverlayController() override;

    void setViewerManager(ViewerManager* manager);
    void beginRotate();
    void cancelRotate();

private:
    struct ViewerEntry {
        VolumeViewerBase* viewer{nullptr};
        QGraphicsProxyWidget* proxy{nullptr};
        QMetaObject::Connection overlaysUpdatedConn;
        QMetaObject::Connection destroyedConn;
    };

    void attachViewer(VolumeViewerBase* viewer);
    void detachViewer(VolumeViewerBase* viewer);
    VolumeViewerBase* targetViewer() const;
    std::shared_ptr<QuadSurface> currentSourceSurface() const;
    void ensureWidgetForTarget();
    void clearWidgets();
    void positionWidget(ViewerEntry& entry) const;
    void setAngle(double angleDeg);
    void updatePreview();
    void applyRotation();
    static std::shared_ptr<QuadSurface> cloneSurface(const std::shared_ptr<QuadSurface>& surface);

    CState* _state{nullptr};
    ViewerManager* _viewerManager{nullptr};
    QMetaObject::Connection _viewerCreatedConn;
    QMetaObject::Connection _managerDestroyedConn;
    std::vector<ViewerEntry> _viewers;

    bool _rotateActive{false};
    double _angleDeg{0.0};
    std::shared_ptr<QuadSurface> _sourceSurface;
    std::shared_ptr<QuadSurface> _previewSurface;

    // Reentrancy + stale-completion guard for applyRotation's
    // QtConcurrent worker. _saveInFlight blocks a second Apply
    // while the first save is running. _saveSessionId increments
    // on every Apply dispatch and on cancelRotate, so a worker
    // whose captured session doesn't match the current one (the
    // user canceled or started a new rotation session) drops its
    // result instead of overwriting the live state.
    bool _saveInFlight{false};
    int _saveSessionId{0};
};
