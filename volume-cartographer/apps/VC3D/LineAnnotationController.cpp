#include "LineAnnotationController.hpp"

#include "CState.hpp"
#include "LineAnnotationDialog.hpp"
#include "ViewerManager.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/Logging.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/LasagnaNormalSampler.hpp"
#include "vc/lasagna/LineModel.hpp"
#include "vc/lasagna/LineOptimizer.hpp"
#include "vc/lasagna/LineViewBuilder.hpp"
#include "volume_viewers/CChunkedVolumeViewer.hpp"

#include <QFileDialog>
#include <QFutureWatcher>
#include <QMessageBox>
#include <QPointF>
#include <QtConcurrent/QtConcurrent>
#include <QWidget>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <utility>

namespace fs = std::filesystem;

struct LineAnnotationController::LineAnnotationSession {
    enum class TaskState {
        Idle,
        Running,
        Succeeded,
        Failed,
    };

    std::string surfaceName;
    std::string selectedDatasetLocation;
    fs::path selectedManifestPath;
    TaskState taskState = TaskState::Idle;
    cv::Vec3d seedPoint{0.0, 0.0, 0.0};
    std::string sourceAnnotationSurfaceName;
    vc::lasagna::LineOptimizationReport optimizationReport;
    vc::lasagna::LineModel optimizedLine;
    std::vector<std::string> generatedSurfaceNames;
    std::string error;
    QPointer<QFutureWatcher<OptimizationTaskResult>> watcher;
};

namespace {

void validateLasagnaManifest(const fs::path& manifestPath)
{
    vc::lasagna::LasagnaDataset dataset = vc::lasagna::LasagnaDataset::open(manifestPath);
    vc::lasagna::LasagnaNormalSampler sampler(dataset);
}

LineAnnotationController::OptimizationTaskResult optimizeLineFromManifest(
    fs::path manifestPath,
    cv::Vec3d seedPoint)
{
    LineAnnotationController::OptimizationTaskResult task;
    task.manifestPath = std::move(manifestPath);
    task.seedPoint = seedPoint;
    try {
        vc::lasagna::LasagnaDataset dataset =
            vc::lasagna::LasagnaDataset::open(task.manifestPath);
        vc::lasagna::LasagnaNormalSampler sampler(dataset);
        vc::lasagna::LineOptimizer optimizer(sampler);
        vc::lasagna::LineOptimizationConfig config;
        config.segmentsPerSide = 10;
        config.segmentLength = 50.0;
        config.samplesPerSegment = 4;
        task.result = optimizer.optimizeFromSeed(seedPoint, config);
        task.ok = true;
    } catch (const std::exception& ex) {
        task.ok = false;
        task.error = ex.what();
    } catch (...) {
        task.ok = false;
        task.error = "Unknown Lasagna line optimization error.";
    }
    return task;
}

} // namespace

LineAnnotationController::LineAnnotationController(CState* state,
                                                   ViewerManager* viewerManager,
                                                   QWidget* parentWidget,
                                                   QObject* parent)
    : QObject(parent)
    , _state(state)
    , _viewerManager(viewerManager)
    , _parentWidget(parentWidget)
    , _datasetPicker([this](QWidget* parent, const fs::path& startDir) {
        return pickDataset(parent, startDir);
    })
    , _optimizationTaskFactory([](fs::path manifestPath, cv::Vec3d seedPoint) {
        return optimizeLineFromManifest(std::move(manifestPath), seedPoint);
    })
{
    if (_state) {
        connect(_state,
                &CState::surfaceChanged,
                this,
                &LineAnnotationController::onSurfaceChanged);
    }
}

void LineAnnotationController::setDatasetPickerForTesting(DatasetPicker picker)
{
    _datasetPicker = std::move(picker);
}

void LineAnnotationController::setOptimizationTaskFactoryForTesting(OptimizationTaskFactory factory)
{
    _optimizationTaskFactory = std::move(factory);
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

    auto session = std::make_shared<LineAnnotationSession>();
    session->surfaceName = surfaceName;
    session->sourceAnnotationSurfaceName = surfaceName;

    _panes.push_back(PaneRecord{_nextPaneId - 1, sourceKind, surfaceName, dialog, session});
    connect(dialog, &LineAnnotationDialog::paneClosed, this, [this](const std::string& name) {
        cleanupSurfaceName(name);
    });
    connect(dialog,
            &LineAnnotationDialog::lineSeedRequested,
            this,
            [this](const std::string& name, cv::Vec3f volumePoint, QPointF) {
                handleLineSeed(name, volumePoint);
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

void LineAnnotationController::handleLineSeed(const std::string& surfaceName, cv::Vec3f volumePoint)
{
    auto* pane = paneForSurface(surfaceName);
    if (!pane || !pane->session) {
        return;
    }

    auto& session = *pane->session;
    if (session.taskState == LineAnnotationSession::TaskState::Running) {
        showError(tr("Line optimization is already running."));
        return;
    }

    if (!ensureDatasetForSession(session)) {
        return;
    }

    startOptimization(session, cv::Vec3d(volumePoint[0], volumePoint[1], volumePoint[2]));
}

bool LineAnnotationController::ensureDatasetForSession(LineAnnotationSession& session)
{
    if (!_state || !_state->vpkg()) {
        showError(tr("No volume package loaded."));
        return false;
    }

    auto vpkg = _state->vpkg();
    std::string selected = vpkg->selectedLasagnaDataset();
    fs::path manifestPath = vpkg->selectedLasagnaDatasetPath();

    if (selected.empty()) {
        const fs::path startDir = vpkg->path().empty()
            ? fs::path{}
            : vpkg->path().parent_path();
        auto picked = _datasetPicker ? _datasetPicker(_parentWidget, startDir)
                                     : std::optional<std::string>{};
        if (!picked || picked->empty()) {
            return false;
        }
        selected = *picked;
        manifestPath = vc::project::resolveLocalPath(selected, vpkg->path().parent_path());
        try {
            validateLasagnaManifest(manifestPath);
        } catch (const std::exception& ex) {
            showError(tr("Invalid Lasagna dataset: %1").arg(QString::fromStdString(ex.what())));
            return false;
        }
        vpkg->setSelectedLasagnaDataset(selected);
    } else {
        try {
            validateLasagnaManifest(manifestPath);
        } catch (const std::exception& ex) {
            showError(tr("Invalid selected Lasagna dataset: %1")
                          .arg(QString::fromStdString(ex.what())));
            return false;
        }
    }

    session.selectedDatasetLocation = selected;
    session.selectedManifestPath = manifestPath;
    return true;
}

void LineAnnotationController::startOptimization(LineAnnotationSession& session, cv::Vec3d seedPoint)
{
    session.taskState = LineAnnotationSession::TaskState::Running;
    session.seedPoint = seedPoint;
    session.error.clear();

    auto* watcher = new QFutureWatcher<OptimizationTaskResult>(this);
    session.watcher = watcher;
    const std::string surfaceName = session.surfaceName;
    connect(watcher,
            &QFutureWatcher<OptimizationTaskResult>::finished,
            this,
            [this, surfaceName, watcher]() {
                finishOptimization(surfaceName);
                watcher->deleteLater();
            });

    const auto manifestPath = session.selectedManifestPath;
    auto factory = _optimizationTaskFactory;
    watcher->setFuture(QtConcurrent::run([factory, manifestPath, seedPoint]() mutable {
        if (factory) {
            return factory(manifestPath, seedPoint);
        }
        return optimizeLineFromManifest(manifestPath, seedPoint);
    }));
}

void LineAnnotationController::finishOptimization(const std::string& surfaceName)
{
    auto* pane = paneForSurface(surfaceName);
    if (!pane || !pane->session || !pane->session->watcher) {
        return;
    }

    auto& session = *pane->session;
    auto* watcher = session.watcher.data();
    if (!watcher) {
        return;
    }

    OptimizationTaskResult task = watcher->result();
    session.watcher = nullptr;
    if (task.ok) {
        session.taskState = LineAnnotationSession::TaskState::Succeeded;
        session.seedPoint = task.seedPoint;
        session.selectedManifestPath = task.manifestPath;
        session.optimizationReport = task.result.report;
        session.optimizedLine = std::move(task.result.line);
        if (!materializeGeneratedViews(session)) {
            session.taskState = LineAnnotationSession::TaskState::Failed;
            return;
        }
        Logger()->info("Line annotation Lasagna optimization complete: points={} iterations={} initial_cost={} final_cost={} valid_normals={} invalid_normals={} converged={}",
                       session.optimizedLine.points.size(),
                       session.optimizationReport.iterations,
                       session.optimizationReport.initialCost,
                       session.optimizationReport.finalCost,
                       session.optimizationReport.validNormalSamples,
                       session.optimizationReport.invalidNormalSamples,
                       session.optimizationReport.converged);
        return;
    }

    session.taskState = LineAnnotationSession::TaskState::Failed;
    session.error = task.error;
    showError(tr("Lasagna line optimization failed: %1")
                  .arg(QString::fromStdString(task.error)));
}

bool LineAnnotationController::materializeGeneratedViews(LineAnnotationSession& session)
{
    if (!_state) {
        session.error = "No active application state.";
        showError(tr("Could not create line annotation views: no active application state."));
        return false;
    }

    vc::lasagna::LineViewSurfaces views;
    try {
        views = vc::lasagna::buildLineViewSurfaces(session.optimizedLine);
    } catch (const std::exception& ex) {
        session.error = ex.what();
        showError(tr("Could not create line annotation views: %1")
                      .arg(QString::fromStdString(session.error)));
        return false;
    }

    for (const auto& name : session.generatedSurfaceNames) {
        _state->setSurface(name, nullptr);
    }
    session.generatedSurfaceNames.clear();

    std::vector<std::vector<std::pair<std::string, QString>>> rows;
    rows.push_back({{"line-surface", tr("Line Surface")}});
    rows.push_back({{"line-side-slice", tr("Line Side Slice")}});

    _state->setSurface("line-surface", views.lineSurface);
    _state->setSurface("line-side-slice", views.lineSideSlice);
    session.generatedSurfaceNames.push_back("line-surface");
    session.generatedSurfaceNames.push_back("line-side-slice");

    std::vector<std::pair<std::string, QString>> zSliceRow;
    zSliceRow.reserve(views.lineZSlices.size());
    for (size_t i = 0; i < views.lineZSlices.size(); ++i) {
        std::ostringstream name;
        name << "line-z-slice-" << std::setw(3) << std::setfill('0') << i;
        const std::string surfaceName = name.str();
        _state->setSurface(surfaceName, views.lineZSlices[i]);
        session.generatedSurfaceNames.push_back(surfaceName);
        zSliceRow.push_back({surfaceName,
                             tr("Line Z Slice %1").arg(static_cast<int>(i))});
    }
    if (!zSliceRow.empty()) {
        rows.push_back(std::move(zSliceRow));
    }

    auto* pane = paneForSurface(session.surfaceName);
    if (!pane || !pane->dialog) {
        return true;
    }

    CChunkedVolumeViewer::CameraState camera;
    camera.scale = 1.0f;
    if (!pane->dialog->panes().empty() && pane->dialog->panes().front().viewer) {
        camera = pane->dialog->panes().front().viewer->cameraState();
        camera.zOffset = 0.0f;
        camera.zOffsetWorldDir = {0, 0, 0};
    }

    if (!pane->dialog->setGeneratedRows(rows, camera)) {
        for (const auto& name : session.generatedSurfaceNames) {
            _state->setSurface(name, nullptr);
        }
        session.generatedSurfaceNames.clear();
        session.error = "Failed to create generated annotation viewers.";
        showError(tr("Could not create generated line annotation viewers."));
        return false;
    }
    return true;
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
    std::vector<std::string> generatedSurfaceNames;
    for (const auto& pane : _panes) {
        if (pane.surfaceName == surfaceName && pane.session && pane.session->watcher) {
            auto* watcher = pane.session->watcher.data();
            disconnect(watcher, nullptr, this, nullptr);
            connect(watcher,
                    &QFutureWatcher<OptimizationTaskResult>::finished,
                    watcher,
                    &QObject::deleteLater);
            pane.session->watcher = nullptr;
        }
        if (pane.surfaceName == surfaceName && pane.session) {
            generatedSurfaceNames = pane.session->generatedSurfaceNames;
        }
    }
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
        for (const auto& name : generatedSurfaceNames) {
            _state->setSurface(name, nullptr);
        }
    }
}

LineAnnotationController::PaneRecord*
LineAnnotationController::paneForSurface(const std::string& surfaceName)
{
    auto it = std::find_if(_panes.begin(), _panes.end(), [&surfaceName](const PaneRecord& pane) {
        return pane.surfaceName == surfaceName;
    });
    return it == _panes.end() ? nullptr : &*it;
}

const LineAnnotationController::PaneRecord*
LineAnnotationController::paneForSurface(const std::string& surfaceName) const
{
    auto it = std::find_if(_panes.begin(), _panes.end(), [&surfaceName](const PaneRecord& pane) {
        return pane.surfaceName == surfaceName;
    });
    return it == _panes.end() ? nullptr : &*it;
}

std::optional<std::string> LineAnnotationController::pickDataset(
    QWidget* parent,
    const fs::path& startDir) const
{
    const QString picked = QFileDialog::getOpenFileName(
        parent,
        tr("Select Lasagna Dataset"),
        QString::fromStdString(startDir.string()),
        tr("Lasagna datasets (*.lasagna.json);;JSON files (*.json);;All files (*)"));
    if (picked.isEmpty()) {
        return std::nullopt;
    }
    return picked.toStdString();
}

LineAnnotationController::OptimizationTaskResult LineAnnotationController::runOptimizationTask(
    fs::path manifestPath,
    cv::Vec3d seedPoint) const
{
    if (_optimizationTaskFactory) {
        return _optimizationTaskFactory(std::move(manifestPath), seedPoint);
    }
    return optimizeLineFromManifest(std::move(manifestPath), seedPoint);
}

void LineAnnotationController::showError(const QString& message) const
{
    if (_parentWidget) {
        QMessageBox::warning(_parentWidget, tr("Line Annotation"), message);
    } else {
        Logger()->warn("Line Annotation: {}", message.toStdString());
    }
}
