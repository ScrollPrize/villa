#include "LineAnnotationDialog.hpp"

#include "ViewerManager.hpp"

#include <QKeyEvent>
#include <QMdiArea>
#include <QMdiSubWindow>
#include <QVBoxLayout>

LineAnnotationDialog::LineAnnotationDialog(ViewerManager* viewerManager, QWidget* parent)
    : QDialog(parent)
    , _viewerManager(viewerManager)
{
    setWindowTitle(tr("Line Annotation"));
    setAttribute(Qt::WA_DeleteOnClose);
    resize(900, 700);

    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);

    _mdiArea = new QMdiArea(this);
    layout->addWidget(_mdiArea);
}

CChunkedVolumeViewer* LineAnnotationDialog::addPane(
    const std::string& surfaceName,
    const QString& title,
    const CChunkedVolumeViewer::CameraState& camera)
{
    if (!_viewerManager || !_mdiArea) {
        return nullptr;
    }

    auto* base = _viewerManager->createViewer(surfaceName,
                                             title,
                                             _mdiArea,
                                             ViewerManager::ViewerRole::Annotation);
    if (!base) {
        return nullptr;
    }

    auto* viewer = qobject_cast<CChunkedVolumeViewer*>(base->asQObject());
    if (!viewer) {
        return nullptr;
    }

    auto* subWindow = qobject_cast<QMdiSubWindow*>(viewer->parentWidget());
    if (subWindow) {
        subWindow->showMaximized();
        connect(subWindow, &QObject::destroyed, this, [this, surfaceName]() {
            emit paneClosed(surfaceName);
        });
    }

    viewer->applyCameraState(camera, false);
    _panes.push_back(Pane{surfaceName, viewer, subWindow});
    return viewer;
}

void LineAnnotationDialog::keyPressEvent(QKeyEvent* event)
{
    if (event->key() == Qt::Key_Escape ||
        (event->key() == Qt::Key_X && event->modifiers() == Qt::NoModifier)) {
        close();
        event->accept();
        return;
    }
    QDialog::keyPressEvent(event);
}
