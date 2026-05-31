#include "LineAnnotationDialog.hpp"

#include "ViewerManager.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <QBrush>
#include <QComboBox>
#include <QGraphicsEllipseItem>
#include <QKeyEvent>
#include <QHBoxLayout>
#include <QMdiArea>
#include <QPen>
#include <QMdiSubWindow>
#include <QPushButton>
#include <QTimer>
#include <QVBoxLayout>
#include <QWidget>

#include <algorithm>

namespace {

CChunkedVolumeViewer::CameraState generatedPaneCamera(CChunkedVolumeViewer* viewer,
                                                      const CChunkedVolumeViewer::CameraState& fallback)
{
    CChunkedVolumeViewer::CameraState camera = fallback;
    camera.surfacePtrX = 0.0f;
    camera.surfacePtrY = 0.0f;
    camera.zOffset = 0.0f;
    camera.zOffsetWorldDir = {0, 0, 0};

    auto* quad = viewer ? dynamic_cast<QuadSurface*>(viewer->currentSurface()) : nullptr;
    if (!quad) {
        return camera;
    }

    const cv::Size size = quad->size();
    if (size.width <= 0 || size.height <= 0) {
        return camera;
    }

    constexpr float kNominalGeneratedRowWidth = 900.0f;
    constexpr float kNominalGeneratedRowHeight = 260.0f;
    constexpr float kPadding = 0.85f;
    const float scaleX = kNominalGeneratedRowWidth / static_cast<float>(std::max(1, size.width));
    const float scaleY = kNominalGeneratedRowHeight / static_cast<float>(std::max(1, size.height));
    camera.scale = std::clamp(std::min(scaleX, scaleY) * kPadding, 0.5f, 16.0f);
    return camera;
}

} // namespace

LineAnnotationDialog::LineAnnotationDialog(ViewerManager* viewerManager, QWidget* parent)
    : QDialog(parent)
    , _viewerManager(viewerManager)
{
    setWindowTitle(tr("Line Annotation"));
    setAttribute(Qt::WA_DeleteOnClose);
    resize(900, 700);

    _layout = new QVBoxLayout(this);
    _layout->setContentsMargins(0, 0, 0, 0);
    _layout->setSpacing(0);

    auto* buttonRow = new QWidget(this);
    auto* buttonLayout = new QHBoxLayout(buttonRow);
    buttonLayout->setContentsMargins(6, 6, 6, 6);
    buttonLayout->setSpacing(6);
    _initialDirectionCombo = new QComboBox(buttonRow);
    _initialDirectionCombo->addItem(tr("sideways"), static_cast<int>(InitialDirectionMode::Sideways));
    _initialDirectionCombo->addItem(tr("z (in/out)"), static_cast<int>(InitialDirectionMode::ZInOut));
    _initialDirectionCombo->setCurrentIndex(1);
    buttonLayout->addWidget(_initialDirectionCombo);
    _showAsMeshButton = new QPushButton(tr("show as mesh"), buttonRow);
    _showAsMeshButton->setEnabled(false);
    buttonLayout->addWidget(_showAsMeshButton);
    buttonLayout->addStretch(1);
    _layout->addWidget(buttonRow, 0);
    connect(_showAsMeshButton, &QPushButton::clicked, this, [this]() {
        emit showAsMeshRequested();
    });

    _mdiArea = new QMdiArea(this);
    _layout->addWidget(_mdiArea);
}

LineAnnotationDialog::InitialDirectionMode LineAnnotationDialog::initialDirectionMode() const
{
    if (!_initialDirectionCombo) {
        return InitialDirectionMode::Sideways;
    }
    return static_cast<InitialDirectionMode>(_initialDirectionCombo->currentData().toInt());
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
            if (!_suppressPaneClosed) {
                emit paneClosed(surfaceName);
            }
        });
    }

    viewer->applyCameraState(camera, false);
    bindPaneInteractions(surfaceName, viewer, true);
    _panes.push_back(Pane{surfaceName, viewer, subWindow});
    return viewer;
}

bool LineAnnotationDialog::setGeneratedRows(
    const std::vector<std::vector<std::pair<std::string, QString>>>& rows,
    const CChunkedVolumeViewer::CameraState& camera,
    const std::map<std::string, cv::Vec3f>& pointMarkers)
{
    if (!_viewerManager || !_layout) {
        return false;
    }

    if (_showAsMeshButton) {
        _showAsMeshButton->setEnabled(false);
    }

    _suppressPaneClosed = true;
    if (_mdiArea) {
        _layout->removeWidget(_mdiArea);
        delete _mdiArea;
        _mdiArea = nullptr;
    }
    _suppressPaneClosed = false;
    _panes.clear();

    for (const auto& row : rows) {
        if (row.empty()) {
            continue;
        }

        auto* rowWidget = new QWidget(this);
        auto* rowLayout = new QHBoxLayout(rowWidget);
        rowLayout->setContentsMargins(0, 0, 0, 0);
        rowLayout->setSpacing(0);
        _layout->addWidget(rowWidget, 1);

        for (const auto& [surfaceName, title] : row) {
            auto* base = _viewerManager->createViewerInWidget(
                surfaceName,
                rowWidget,
                ViewerManager::ViewerRole::Annotation);
            if (!base) {
                return false;
            }
            auto* viewer = qobject_cast<CChunkedVolumeViewer*>(base->asQObject());
            if (!viewer) {
                return false;
            }
            viewer->setObjectName(title);
            viewer->applyCameraState(generatedPaneCamera(viewer, camera), false);
            bindPaneInteractions(surfaceName, viewer, false);
            rowLayout->addWidget(viewer, 1);
            _panes.push_back(Pane{surfaceName, viewer, {}});
            if (auto marker = pointMarkers.find(surfaceName); marker != pointMarkers.end()) {
                setLinePointMarker(surfaceName, viewer, marker->second);
            }
        }
    }
    const bool ok = !_panes.empty();
    if (_showAsMeshButton) {
        _showAsMeshButton->setEnabled(ok);
    }
    return ok;
}

void LineAnnotationDialog::bindPaneInteractions(const std::string& surfaceName,
                                                CChunkedVolumeViewer* viewer,
                                                bool seedPlacementEnabled)
{
    if (!viewer) {
        return;
    }

    viewer->setLineAnnotationPlacementPreviewEnabled(seedPlacementEnabled);
    if (!seedPlacementEnabled) {
        return;
    }
    connect(viewer,
            &CChunkedVolumeViewer::sendLineAnnotationSeedRequested,
            this,
            [this, surfaceName](cv::Vec3f volumePoint, QPointF scenePoint) {
                emit lineSeedRequested(surfaceName, volumePoint, scenePoint);
            });
}

void LineAnnotationDialog::setLinePointMarker(const std::string& surfaceName,
                                              CChunkedVolumeViewer* viewer,
                                              const cv::Vec3f& volumePoint)
{
    if (!viewer) {
        return;
    }

    const auto key = "line_annotation_point_" + surfaceName;
    QPointer<CChunkedVolumeViewer> viewerPtr(viewer);
    QTimer::singleShot(0, this, [viewerPtr, volumePoint, key]() {
        if (!viewerPtr) {
            return;
        }
        viewerPtr->renderVisible(true, "line annotation point marker");
        const QPointF scenePoint = viewerPtr->volumeToScene(volumePoint);
        if (!std::isfinite(scenePoint.x()) || !std::isfinite(scenePoint.y())) {
            return;
        }

        auto* marker = new QGraphicsEllipseItem(scenePoint.x() - 5.0,
                                                scenePoint.y() - 5.0,
                                                10.0,
                                                10.0);
        QPen pen(QColor(255, 230, 0), 2.0);
        pen.setCosmetic(true);
        marker->setPen(pen);
        marker->setBrush(QBrush(QColor(255, 230, 0, 180)));
        marker->setZValue(150.0);
        marker->setAcceptedMouseButtons(Qt::NoButton);

        viewerPtr->setOverlayGroup(key, {marker});
    });
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
