#pragma once

#include <QDialog>
#include <QPointer>

#include <memory>
#include <string>
#include <vector>

#include "volume_viewers/CChunkedVolumeViewer.hpp"

class CState;
class QMdiArea;
class QMdiSubWindow;
class ViewerManager;

class LineAnnotationDialog : public QDialog
{
    Q_OBJECT

public:
    struct Pane {
        std::string surfaceName;
        QPointer<CChunkedVolumeViewer> viewer;
        QPointer<QMdiSubWindow> subWindow;
    };

    explicit LineAnnotationDialog(ViewerManager* viewerManager, QWidget* parent = nullptr);

    CChunkedVolumeViewer* addPane(const std::string& surfaceName,
                                  const QString& title,
                                  const CChunkedVolumeViewer::CameraState& camera);
    const std::vector<Pane>& panes() const { return _panes; }

signals:
    void paneClosed(const std::string& surfaceName);

protected:
    void keyPressEvent(QKeyEvent* event) override;

private:
    ViewerManager* _viewerManager = nullptr;
    QMdiArea* _mdiArea = nullptr;
    std::vector<Pane> _panes;
};
