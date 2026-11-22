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
    // Publishes the transformed surface to the object store, only over the stored geometry
    // revision it was transformed from (the second argument).
    using Saver = std::function<bool(QuadSurface*, const std::optional<std::string>&)>;
    void setSaver(Saver saver) { _saver = std::move(saver); }
    // The geometry revision a surface holds: the one it was opened at or last saved as (none
    // when unknown). Read when Rotate opens, for the geometry it transforms.
    void setSourceRevisionReader(std::function<std::optional<std::string>(QuadSurface*)> reader)
    {
        _sourceRevisionReader = std::move(reader);
    }
    // The stored segment's current geometry revision (none when it is not stored or the store
    // did not answer). Apply saves only while it is the revision Rotate transforms.
    void setRevisionReader(std::function<std::optional<std::string>(const std::string&)> reader)
    {
        _revisionReader = std::move(reader);
    }
    // Opens the segment's stored revision again (selects it as the surface panel does).
    void setReopener(std::function<bool(const std::string&)> reopener) { _reopener = std::move(reopener); }
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
    // The stored geometry is known to be the one the tool transforms (an unknown revision never is).
    bool sourceCurrent() const;
    // A newer geometry revision of the segment is known to be stored.
    bool newerRevisionStored() const;
    // End the preview without saving, leaving a newer selection alone; a newer revision is reopened.
    void releaseSource();
    void closeRotate();
    static std::shared_ptr<QuadSurface> cloneSurface(const std::shared_ptr<QuadSurface>& surface);

    CState* _state{nullptr};
    Saver _saver;
    std::function<std::optional<std::string>(QuadSurface*)> _sourceRevisionReader;
    std::function<std::optional<std::string>(const std::string&)> _revisionReader;
    std::function<bool(const std::string&)> _reopener;
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
    // The geometry revision of _sourceSurface when the tool opened.
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
