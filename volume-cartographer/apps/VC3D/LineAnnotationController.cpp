#include "LineAnnotationController.hpp"

#include "CState.hpp"
#include "LineAnnotationDialog.hpp"
#include "ViewerManager.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/Surface.hpp"
#include "volume_viewers/CChunkedVolumeViewer.hpp"

#include <QPointF>
#include <QWidget>

#include <algorithm>
#include <cmath>

LineAnnotationController::LineAnnotationController(CState* state,
                                                   ViewerManager* viewerManager,
                                                   QWidget* parentWidget,
                                                   QObject* parent)
    : QObject(parent)
    , _state(state)
    , _viewerManager(viewerManager)
    , _parentWidget(parentWidget)
{
    if (_state) {
        connect(_state,
                &CState::surfaceChanged,
                this,
                &LineAnnotationController::onSurfaceChanged);
    }
}

bool LineAnnotationController::canLaunchFromViewer(const CChunkedVolumeViewer* viewer) const
{
    if (!viewer || !_state || !_viewerManager) {
        return false;
    }
    auto* surface = viewer->currentSurface();
    if (dynamic_cast<PlaneSurface*>(surface)) {
        return true;
    }
    return viewer->surfName() == "segmentation" &&
           dynamic_cast<QuadSurface*>(surface) != nullptr;
}

void LineAnnotationController::launchFromViewer(CChunkedVolumeViewer* viewer, const QPointF& /*scenePoint*/)
{
    if (!canLaunchFromViewer(viewer)) {
        return;
    }

    auto surfaceName = nextSurfaceName();
    auto camera = viewer->cameraState();
    SourceKind sourceKind = SourceKind::Plane;

    if (auto* plane = dynamic_cast<PlaneSurface*>(viewer->currentSurface())) {
        auto clone = std::make_shared<PlaneSurface>(*plane);
        const cv::Vec3f normal = plane->normal({0, 0, 0});
        if (std::isfinite(normal[0]) && std::isfinite(normal[1]) &&
            std::isfinite(normal[2]) && cv::norm(normal) > 0.0f) {
            clone->setOrigin(plane->origin() + normal * viewer->normalOffset());
        }
        camera.zOffset = 0.0f;
        camera.zOffsetWorldDir = {0, 0, 0};
        _state->setSurface(surfaceName, clone);
    } else {
        sourceKind = SourceKind::Segmentation;
        _state->setSurface(surfaceName, _state->surface("segmentation"));
    }

    auto* dialog = new LineAnnotationDialog(_viewerManager, _parentWidget);
    if (!dialog->addPane(surfaceName, tr("Line Annotation Slice"), camera)) {
        dialog->deleteLater();
        _state->setSurface(surfaceName, nullptr);
        return;
    }
    dialog->showMaximized();
    dialog->raise();
    dialog->activateWindow();

    _panes.push_back(PaneRecord{_nextPaneId - 1, sourceKind, surfaceName, dialog});
    connect(dialog, &LineAnnotationDialog::paneClosed, this, [this](const std::string& name) {
        cleanupSurfaceName(name);
    });
    connect(dialog, &QObject::destroyed, this, [this, surfaceName]() {
        cleanupSurfaceName(surfaceName);
    });
}

void LineAnnotationController::onSurfaceChanged(std::string name,
                                                std::shared_ptr<Surface> surf,
                                                bool /*isEditUpdate*/)
{
    if (name != "segmentation" || !_state) {
        return;
    }
    for (const auto& pane : _panes) {
        if (pane.sourceKind == SourceKind::Segmentation) {
            _state->setSurface(pane.surfaceName, surf);
        }
    }
}

std::string LineAnnotationController::nextSurfaceName()
{
    return "line_annotation_slice_" + std::to_string(_nextPaneId++);
}

void LineAnnotationController::cleanupSurfaceName(const std::string& surfaceName)
{
    if (surfaceName.empty()) {
        return;
    }

    const auto before = _panes.size();
    _panes.erase(std::remove_if(_panes.begin(),
                                _panes.end(),
                                [&surfaceName](const PaneRecord& pane) {
                                    return pane.surfaceName == surfaceName;
                                }),
                 _panes.end());
    if (before == _panes.size()) {
        return;
    }

    if (_state) {
        _state->setSurface(surfaceName, nullptr);
    }
}
