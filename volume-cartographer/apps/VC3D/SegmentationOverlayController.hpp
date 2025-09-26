#pragma once

#include <QObject>
#include <QMetaObject>
#include <vector>
#include <optional>

#include <opencv2/core.hpp>

class CSurfaceCollection;
class CVolumeViewer;
class SegmentationEditManager;
class Surface;

class SegmentationOverlayController : public QObject
{
    Q_OBJECT

public:
    explicit SegmentationOverlayController(CSurfaceCollection* surfCollection, QObject* parent = nullptr);

    void attachViewer(CVolumeViewer* viewer);
    void detachViewer(CVolumeViewer* viewer);

    void setEditingEnabled(bool enabled);
    void setDownsample(int value);
    void setRadius(float radius);
    void refreshAll();
    void refreshViewer(CVolumeViewer* viewer);
    void setEditManager(SegmentationEditManager* manager) { _editManager = manager; }
    void setActiveHandle(std::optional<std::pair<int,int>> key, bool refresh = true);
    void setHoverHandle(std::optional<std::pair<int,int>> key, bool refresh = true);
    void setKeyboardHandle(std::optional<std::pair<int,int>> key, bool refresh = true);

private slots:
    void onSurfaceChanged(std::string name, Surface* surf);

private:
    void rebuildViewerOverlay(CVolumeViewer* viewer);

    struct ViewerState {
        CVolumeViewer* viewer{nullptr};
        QMetaObject::Connection overlayUpdateConn;
        QMetaObject::Connection destroyedConn;
    };

    CSurfaceCollection* _surfCollection;
    std::vector<ViewerState> _viewers;
    bool _editingEnabled{false};
    int _downsample{12};
    float _radius{10.0f};
    SegmentationEditManager* _editManager{nullptr};
    std::optional<std::pair<int,int>> _activeHandle;
    std::optional<std::pair<int,int>> _hoverHandle;
    std::optional<std::pair<int,int>> _keyboardHandle;
};
