#pragma once

#include <QObject>
#include <QPointer>

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>

class CChunkedVolumeViewer;
class CState;
class LineAnnotationDialog;
class Surface;
class ViewerManager;

class LineAnnotationController : public QObject
{
    Q_OBJECT

public:
    LineAnnotationController(CState* state,
                             ViewerManager* viewerManager,
                             QWidget* parentWidget,
                             QObject* parent = nullptr);

    bool canLaunchFromViewer(const CChunkedVolumeViewer* viewer) const;
    void launchFromViewer(CChunkedVolumeViewer* viewer, const QPointF& scenePoint);

private slots:
    void onSurfaceChanged(std::string name, std::shared_ptr<Surface> surf, bool isEditUpdate = false);

private:
    enum class SourceKind {
        Plane,
        Segmentation,
    };

    struct PaneRecord {
        int id = 0;
        SourceKind sourceKind = SourceKind::Plane;
        std::string surfaceName;
        QPointer<LineAnnotationDialog> dialog;
    };

    std::string nextSurfaceName();
    void cleanupSurfaceName(const std::string& surfaceName);

    CState* _state = nullptr;
    ViewerManager* _viewerManager = nullptr;
    QPointer<QWidget> _parentWidget;
    int _nextPaneId = 1;
    std::vector<PaneRecord> _panes;
};
