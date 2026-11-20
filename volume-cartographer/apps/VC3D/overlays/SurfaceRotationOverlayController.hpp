#pragma once

#include <QMetaObject>
#include <QObject>
#include <QPointer>

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

class CState;
class QuadSurface;
class Surface;
class QGraphicsProxyWidget;
class ViewerManager;
class VolumeViewerBase;

class SurfaceRotationOverlayController : public QObject
{
public:
    explicit SurfaceRotationOverlayController(CState* state, QObject* parent = nullptr);
    ~SurfaceRotationOverlayController() override;

    void setViewerManager(ViewerManager* manager);
    // Publishes the transformed surface to the object store.
    void setSaver(std::function<bool(QuadSurface*)> saver) { _saver = std::move(saver); }
    // The stored segment's current revision (none when it is not stored). Apply saves only over
    // the revision the tool was opened on.
    void setRevisionReader(std::function<std::optional<std::string>(const std::string&)> reader)
    {
        _revisionReader = std::move(reader);
    }
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
    bool setHorizontalFlip(bool enabled);
    bool synchronizeTransformPersistenceCompatibility();
    void updatePreview();
    void applyRotation();
    // Make `surface` the active ("segmentation") surface.
    void showActive(const std::shared_ptr<QuadSurface>& surface);
    // End the preview: `surface` is the selected segment again.
    void endPreview(const std::shared_ptr<QuadSurface>& surface);
    // The segment the tool was opened on is still selected and shown as the tool left it.
    bool sourceSelected() const;
    // No newer revision of the segment was saved since the tool was opened.
    bool sourceCurrent() const;
    // End the preview without saving, leaving a newer selection or revision alone.
    void releaseSource();
    void closeRotate();
    static std::shared_ptr<QuadSurface> cloneSurface(const std::shared_ptr<QuadSurface>& surface);

    CState* _state{nullptr};
    std::function<bool(QuadSurface*)> _saver;
    std::function<std::optional<std::string>(const std::string&)> _revisionReader;
    ViewerManager* _viewerManager{nullptr};
    QMetaObject::Connection _viewerCreatedConn;
    QMetaObject::Connection _viewerClosingConn;
    QMetaObject::Connection _managerDestroyedConn;
    std::vector<ViewerEntry> _viewers;

    bool _rotateActive{false};
    double _angleDeg{0.0};
    bool _horizontalFlip{false};
    std::shared_ptr<QuadSurface> _sourceSurface;
    std::string _sourceSurfaceId;
    std::optional<std::string> _sourceRevision;
    std::shared_ptr<QuadSurface> _previewSurface;
    // What the tool put in the viewer last (the source or a preview).
    std::shared_ptr<Surface> _shownSurface;

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
