#include "LineAnnotationController.hpp"

#include "CState.hpp"
#include "LineAnnotationDialog.hpp"
#include "SurfacePanelController.hpp"
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
#include <QDateTime>
#include <QStringList>
#include <QtConcurrent/QtConcurrent>
#include <QWidget>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <locale>
#include <sstream>
#include <utility>

#include <nlohmann/json.hpp>

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
    std::shared_ptr<vc::lasagna::LasagnaDataset> dataset;
    std::shared_ptr<vc::lasagna::LasagnaNormalSampler> normalSampler;
    TaskState taskState = TaskState::Idle;
    cv::Vec3d seedPoint{0.0, 0.0, 0.0};
    std::string sourceAnnotationSurfaceName;
    vc::lasagna::LineOptimizationReport optimizationReport;
    vc::lasagna::LineModel optimizedLine;
    std::vector<vc::lasagna::LineControlPoint> controlPoints;
    double focusedLinePosition = 0.0;
    std::optional<cv::Vec3d> focusedControlPoint;
    cv::Vec3d sourceSliceNormal{0.0, 0.0, 1.0};
    LineAnnotationController::InitialDirectionMode initialDirectionMode =
        LineAnnotationController::InitialDirectionMode::Sideways;
    std::vector<std::string> generatedSurfaceNames;
    std::string error;
    QPointer<QFutureWatcher<OptimizationTaskResult>> watcher;
    uint64_t fiberId = 0;
    bool suppressFiberSave = false;
};

namespace {

constexpr double kEpsilon = 1.0e-12;

bool finiteDirection(const cv::Vec3d& v)
{
    return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]) &&
           std::sqrt(v.dot(v)) > kEpsilon;
}

cv::Vec3d normalizedOrZero(const cv::Vec3d& v)
{
    if (!std::isfinite(v[0]) || !std::isfinite(v[1]) || !std::isfinite(v[2])) {
        return {0.0, 0.0, 0.0};
    }
    const double n = std::sqrt(v.dot(v));
    if (n <= kEpsilon) {
        return {0.0, 0.0, 0.0};
    }
    return v * (1.0 / n);
}

bool finitePoint(const cv::Vec3d& v)
{
    return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]);
}

nlohmann::json pointToJson(const cv::Vec3d& point)
{
    return nlohmann::json::array({point[0], point[1], point[2]});
}

cv::Vec3d pointFromJson(const nlohmann::json& value)
{
    if (!value.is_array() || value.size() != 3) {
        throw std::runtime_error("Point must be a [x, y, z] array");
    }
    cv::Vec3d point{
        value.at(0).get<double>(),
        value.at(1).get<double>(),
        value.at(2).get<double>(),
    };
    if (!finitePoint(point)) {
        throw std::runtime_error("Point contains non-finite coordinates");
    }
    return point;
}

cv::Vec3d initialTangentForMode(
    LineAnnotationController::InitialDirectionMode mode,
    const cv::Vec3d& sourceSliceNormal,
    const vc::lasagna::NormalSample& seedNormal)
{
    const cv::Vec3d sliceNormal = normalizedOrZero(sourceSliceNormal);
    const cv::Vec3d gtNormal = seedNormal.valid
        ? normalizedOrZero(seedNormal.normal)
        : cv::Vec3d{0.0, 0.0, 0.0};

    if (!finiteDirection(sliceNormal) || !finiteDirection(gtNormal)) {
        return {0.0, 0.0, 0.0};
    }

    if (mode == LineAnnotationController::InitialDirectionMode::ZInOut) {
        return normalizedOrZero(sliceNormal - gtNormal * sliceNormal.dot(gtNormal));
    }
    return normalizedOrZero(sliceNormal.cross(gtNormal));
}

void validateLasagnaManifest(const fs::path& manifestPath)
{
    vc::lasagna::LasagnaDataset dataset = vc::lasagna::LasagnaDataset::open(manifestPath);
    vc::lasagna::LasagnaNormalSampler sampler(dataset);
}

LineAnnotationController::OptimizationTaskResult optimizeLineWithSampler(
    fs::path manifestPath,
    std::vector<vc::lasagna::LineControlPoint> controlPoints,
    std::vector<cv::Vec3d> initialLinePoints,
    cv::Vec3d sourceSliceNormal,
    LineAnnotationController::InitialDirectionMode directionMode,
    const vc::lasagna::NormalSampler& sampler);

LineAnnotationController::OptimizationTaskResult optimizeLineFromManifest(
    fs::path manifestPath,
    std::vector<vc::lasagna::LineControlPoint> controlPoints,
    std::vector<cv::Vec3d> initialLinePoints,
    cv::Vec3d sourceSliceNormal,
    LineAnnotationController::InitialDirectionMode directionMode)
{
    vc::lasagna::LasagnaDataset dataset =
        vc::lasagna::LasagnaDataset::open(manifestPath);
    vc::lasagna::LasagnaNormalSampler sampler(dataset);
    return optimizeLineWithSampler(std::move(manifestPath),
                                   std::move(controlPoints),
                                   std::move(initialLinePoints),
                                   sourceSliceNormal,
                                   directionMode,
                                   sampler);
}

LineAnnotationController::OptimizationTaskResult optimizeLineWithSampler(
    fs::path manifestPath,
    std::vector<vc::lasagna::LineControlPoint> controlPoints,
    std::vector<cv::Vec3d> initialLinePoints,
    cv::Vec3d sourceSliceNormal,
    LineAnnotationController::InitialDirectionMode directionMode,
    const vc::lasagna::NormalSampler& sampler)
{
    LineAnnotationController::OptimizationTaskResult task;
    task.manifestPath = std::move(manifestPath);
    task.controlPoints = std::move(controlPoints);
    if (!task.controlPoints.empty()) {
        const auto seedIt = std::find_if(task.controlPoints.begin(),
                                         task.controlPoints.end(),
                                         [](const vc::lasagna::LineControlPoint& control) {
                                             return control.isSeed;
                                         });
        task.seedPoint = (seedIt == task.controlPoints.end()
            ? task.controlPoints.front()
            : *seedIt).volumePoint;
    }
    task.sourceSliceNormal = sourceSliceNormal;
    task.initialDirectionMode = directionMode;
    try {
        vc::lasagna::LineOptimizer optimizer(sampler);
        vc::lasagna::LineOptimizationConfig config;
        config.segmentsPerSide = 200;
        config.segmentLength = 32.0;
        config.straightnessWeight = 0.1;
        config.tangentStraightnessWeight = 5.0;
        config.normalStraightnessWeight = 0.005;
        config.samplesPerSegment = 1;
        config.maxIterations = 1000;
        config.differentiableNormalSampling = true;
        config.initialTangent = initialTangentForMode(
            directionMode,
            sourceSliceNormal,
            sampler.sampleNormal(task.seedPoint));
        config.useInitialTangent = finiteDirection(config.initialTangent);
        config.tangentGuideVector = normalizedOrZero(sourceSliceNormal);
        config.tangentGuideWeight = 1.0;
        config.tangentGuideMode = directionMode == LineAnnotationController::InitialDirectionMode::ZInOut
            ? vc::lasagna::LineOptimizationConfig::TangentGuideMode::ProjectVectorOntoTangentPlane
            : vc::lasagna::LineOptimizationConfig::TangentGuideMode::CrossVectorWithNormal;
        config.initialLinePoints = std::move(initialLinePoints);
        task.result = optimizer.optimizeFromControlPoints(task.controlPoints, config);
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
    , _optimizationTaskFactory([](fs::path manifestPath,
                                  std::vector<vc::lasagna::LineControlPoint> controlPoints,
                                  std::vector<cv::Vec3d> initialLinePoints,
                                  cv::Vec3d sourceSliceNormal,
                                  InitialDirectionMode directionMode) {
        return optimizeLineFromManifest(std::move(manifestPath),
                                        std::move(controlPoints),
                                        std::move(initialLinePoints),
                                        sourceSliceNormal,
                                        directionMode);
    })
{
    if (_state) {
        connect(_state,
                &CState::surfaceChanged,
                this,
                &LineAnnotationController::onSurfaceChanged);
        connect(_state,
                &CState::vpkgChanged,
                this,
                &LineAnnotationController::onVolumePackageChanged);
        if (_state->vpkg()) {
            loadFibersForCurrentPackage();
        }
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

void LineAnnotationController::setSurfacePanel(SurfacePanelController* panel)
{
    _surfacePanel = panel;
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
    std::shared_ptr<Surface> sourceSurface;
    cv::Vec3d sourceSliceNormal{
        camera.zOffsetWorldDir[0],
        camera.zOffsetWorldDir[1],
        camera.zOffsetWorldDir[2],
    };

    if (auto* plane = dynamic_cast<PlaneSurface*>(viewer->currentSurface())) {
        auto clone = std::make_shared<PlaneSurface>(*plane);
        const cv::Vec3f normal = plane->normal({0, 0, 0});
        sourceSliceNormal = {normal[0], normal[1], normal[2]};
        if (std::isfinite(normal[0]) && std::isfinite(normal[1]) &&
            std::isfinite(normal[2]) && cv::norm(normal) > 0.0f) {
            clone->setOrigin(plane->origin() + normal * viewer->normalOffset());
        }
        camera.zOffset = 0.0f;
        camera.zOffsetWorldDir = {0, 0, 0};
        sourceSurface = clone;
    } else {
        sourceKind = SourceKind::Segmentation;
        sourceSurface = _state->surface("segmentation");
    }

    auto session = std::make_shared<LineAnnotationSession>();
    launchSession(sourceKind,
                  surfaceName,
                  std::move(sourceSurface),
                  camera,
                  sourceSliceNormal,
                  std::move(session));
}

void LineAnnotationController::launchSession(LineAnnotationController::SourceKind sourceKind,
                                             const std::string& surfaceName,
                                             std::shared_ptr<Surface> sourceSurface,
                                             const CChunkedVolumeViewer::CameraState& camera,
                                             cv::Vec3d sourceSliceNormal,
                                             std::shared_ptr<LineAnnotationController::LineAnnotationSession> session)
{
    if (!_state || !session) {
        return;
    }

    _state->setSurface(surfaceName, std::move(sourceSurface));
    auto* dialog = new LineAnnotationDialog(_viewerManager, nullptr);
    if (!dialog->addPane(surfaceName, tr("Line Annotation Slice"), camera)) {
        dialog->deleteLater();
        _state->setSurface(surfaceName, nullptr);
        return;
    }
    dialog->showMaximized();
    dialog->raise();
    dialog->activateWindow();

    session->surfaceName = surfaceName;
    session->sourceAnnotationSurfaceName = surfaceName;
    session->sourceSliceNormal = finiteDirection(sourceSliceNormal)
        ? normalizedOrZero(sourceSliceNormal)
        : cv::Vec3d{0.0, 0.0, 1.0};

    _panes.push_back(PaneRecord{_nextPaneId - 1, sourceKind, surfaceName, dialog, session});
    connect(dialog, &LineAnnotationDialog::paneClosed, this, [this](const std::string& name) {
        cleanupSurfaceName(name);
    });
    connect(dialog,
            &LineAnnotationDialog::lineSeedRequested,
            this,
            [this, dialog](const std::string& name, cv::Vec3f volumePoint, QPointF) {
                InitialDirectionMode mode = InitialDirectionMode::Sideways;
                if (dialog) {
                    mode = dialog->initialDirectionMode() == LineAnnotationDialog::InitialDirectionMode::ZInOut
                        ? InitialDirectionMode::ZInOut
                        : InitialDirectionMode::Sideways;
                }
                handleLineSeed(name, volumePoint, mode);
            });
    connect(dialog,
            &LineAnnotationDialog::generatedControlPointRequested,
            this,
            [this](const std::string& name, cv::Vec3f volumePoint, double linePosition) {
                handleGeneratedControlPoint(name, volumePoint, linePosition);
            });
    connect(dialog, &LineAnnotationDialog::showAsMeshRequested, this, [this, surfaceName]() {
        handleShowAsMesh(surfaceName);
    });
    connect(dialog, &QObject::destroyed, this, [this, surfaceName]() {
        cleanupSurfaceName(surfaceName);
    });

    if (!session->optimizedLine.points.empty()) {
        session->taskState = LineAnnotationSession::TaskState::Succeeded;
        materializeGeneratedViews(*session);
    }
}

void LineAnnotationController::openFiber(uint64_t fiberId)
{
    auto it = std::find_if(_fibers.begin(), _fibers.end(), [fiberId](const StoredFiber& fiber) {
        return fiber.id == fiberId;
    });
    if (it == _fibers.end()) {
        showError(tr("Fiber %1 is not loaded.").arg(fiberId));
        return;
    }
    if (it->linePoints.empty()) {
        showError(tr("Fiber %1 has no line points.").arg(fiberId));
        return;
    }

    auto session = std::make_shared<LineAnnotationSession>();
    session->fiberId = it->id;
    session->focusedLinePosition = static_cast<double>(it->linePoints.size() / 2);
    session->focusedControlPoint = it->controlPoints.empty()
        ? std::optional<cv::Vec3d>{}
        : std::optional<cv::Vec3d>{it->controlPoints[it->controlPoints.size() / 2]};

    if (!ensureDatasetForSession(*session)) {
        return;
    }

    try {
        session->optimizedLine = lineModelFromPoints(it->linePoints, session->normalSampler.get());
    } catch (const std::exception& ex) {
        showError(tr("Could not reopen fiber %1: %2")
                      .arg(fiberId)
                      .arg(QString::fromStdString(ex.what())));
        return;
    }

    session->controlPoints.clear();
    session->controlPoints.reserve(it->controlPoints.size());
    double seedDistance = std::numeric_limits<double>::infinity();
    int seedControl = -1;
    for (size_t i = 0; i < it->controlPoints.size(); ++i) {
        const cv::Vec3d& controlPoint = it->controlPoints[i];
        int bestIndex = 0;
        double bestDistance = std::numeric_limits<double>::infinity();
        for (size_t lineIndex = 0; lineIndex < it->linePoints.size(); ++lineIndex) {
            const cv::Vec3d delta = it->linePoints[lineIndex] - controlPoint;
            const double distance = std::sqrt(delta.dot(delta));
            if (distance < bestDistance) {
                bestDistance = distance;
                bestIndex = static_cast<int>(lineIndex);
            }
        }
        vc::lasagna::LineControlPoint control;
        control.linePosition = static_cast<double>(bestIndex);
        control.volumePoint = controlPoint;
        control.optimizedIndex = bestIndex;
        session->controlPoints.push_back(control);

        const double centerDistance = std::abs(control.linePosition -
            static_cast<double>(it->linePoints.size() - 1) * 0.5);
        if (centerDistance < seedDistance) {
            seedDistance = centerDistance;
            seedControl = static_cast<int>(i);
        }
    }
    if (!session->controlPoints.empty()) {
        if (seedControl < 0) {
            seedControl = 0;
        }
        session->controlPoints[static_cast<size_t>(seedControl)].isSeed = true;
        session->seedPoint = session->controlPoints[static_cast<size_t>(seedControl)].volumePoint;
        session->optimizedLine.displayFrameAnchorIndex =
            session->controlPoints[static_cast<size_t>(seedControl)].optimizedIndex;
        session->focusedLinePosition =
            session->controlPoints[static_cast<size_t>(seedControl)].linePosition;
    }

    CChunkedVolumeViewer::CameraState camera;
    camera.scale = 1.0f;
    const cv::Vec3d origin = it->linePoints.empty()
        ? cv::Vec3d{0.0, 0.0, 0.0}
        : it->linePoints[it->linePoints.size() / 2];
    auto sourcePlane = std::make_shared<PlaneSurface>(
        cv::Vec3f{static_cast<float>(origin[0]),
                  static_cast<float>(origin[1]),
                  static_cast<float>(origin[2])},
        cv::Vec3f{0.0f, 0.0f, 1.0f});
    launchSession(SourceKind::Plane,
                  nextSurfaceName(),
                  sourcePlane,
                  camera,
                  {0.0, 0.0, 1.0},
                  std::move(session));
}

void LineAnnotationController::deleteFiber(uint64_t fiberId)
{
    const auto path = fiberPath(fiberId);
    std::error_code ec;
    fs::remove(path, ec);
    if (ec) {
        showError(tr("Could not delete fiber %1: %2")
                      .arg(fiberId)
                      .arg(QString::fromStdString(ec.message())));
        return;
    }

    _fibers.erase(std::remove_if(_fibers.begin(),
                                 _fibers.end(),
                                 [fiberId](const StoredFiber& fiber) {
                                     return fiber.id == fiberId;
                                 }),
                  _fibers.end());
    for (const auto& pane : _panes) {
        if (pane.session && pane.session->fiberId == fiberId) {
            pane.session->suppressFiberSave = true;
        }
    }
    emitFiberSummaries();
}

void LineAnnotationController::saveOpenFibers()
{
    for (const auto& pane : _panes) {
        if (!pane.session || pane.session->suppressFiberSave) {
            continue;
        }
        auto& session = *pane.session;
        if (session.taskState != LineAnnotationSession::TaskState::Succeeded ||
            session.optimizedLine.points.empty() ||
            session.controlPoints.empty()) {
            continue;
        }
        saveSessionAsFiber(session);
    }
}

std::vector<LineAnnotationController::FiberSummary> LineAnnotationController::fiberSummaries() const
{
    std::vector<FiberSummary> summaries;
    summaries.reserve(_fibers.size());
    for (const auto& fiber : _fibers) {
        summaries.push_back(FiberSummary{
            fiber.id,
            static_cast<int>(fiber.controlPoints.size()),
            static_cast<int>(fiber.linePoints.size()),
            lineLengthVx(fiber.linePoints),
        });
    }
    std::sort(summaries.begin(), summaries.end(), [](const FiberSummary& a, const FiberSummary& b) {
        return a.id < b.id;
    });
    return summaries;
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

void LineAnnotationController::onVolumePackageChanged(std::shared_ptr<VolumePkg> /*pkg*/)
{
    loadFibersForCurrentPackage();
}

void LineAnnotationController::handleLineSeed(const std::string& surfaceName,
                                              cv::Vec3f volumePoint,
                                              InitialDirectionMode directionMode)
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

    session.initialDirectionMode = directionMode;
    const cv::Vec3d seedPoint(volumePoint[0], volumePoint[1], volumePoint[2]);
    session.seedPoint = seedPoint;
    session.focusedLinePosition = 0.0;
    session.focusedControlPoint = seedPoint;
    session.controlPoints = {{0.0, seedPoint, true, -1}};
    startOptimization(session);
}

void LineAnnotationController::handleGeneratedControlPoint(const std::string& surfaceName,
                                                          cv::Vec3f volumePoint,
                                                          double linePosition)
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
    if (session.optimizedLine.points.empty() || session.controlPoints.empty()) {
        return;
    }
    if (!ensureDatasetForSession(session)) {
        return;
    }

    const double maxPosition = static_cast<double>(session.optimizedLine.points.size() - 1);
    linePosition = std::clamp(linePosition, 0.0, maxPosition);
    const cv::Vec3d clicked(volumePoint[0], volumePoint[1], volumePoint[2]);

    auto nearest = session.controlPoints.end();
    double nearestDistance = std::numeric_limits<double>::infinity();
    for (auto it = session.controlPoints.begin(); it != session.controlPoints.end(); ++it) {
        if (!std::isfinite(it->linePosition)) {
            continue;
        }
        const double distance = std::abs(it->linePosition - linePosition);
        if (distance < nearestDistance) {
            nearestDistance = distance;
            nearest = it;
        }
    }

    if (nearest != session.controlPoints.end() && nearestDistance <= 0.5) {
        nearest->volumePoint = clicked;
        nearest->optimizedIndex = -1;
        linePosition = nearest->linePosition;
        if (nearest->isSeed) {
            session.seedPoint = clicked;
        }
    } else {
        session.controlPoints.push_back({linePosition, clicked, false, -1});
    }

    session.focusedLinePosition = linePosition;
    session.focusedControlPoint = clicked;
    std::stable_sort(session.controlPoints.begin(),
                     session.controlPoints.end(),
                     [](const vc::lasagna::LineControlPoint& a,
                        const vc::lasagna::LineControlPoint& b) {
                         return a.linePosition < b.linePosition;
                     });
    startOptimization(session);
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
            auto dataset = std::make_shared<vc::lasagna::LasagnaDataset>(
                vc::lasagna::LasagnaDataset::open(manifestPath));
            auto sampler = std::make_shared<vc::lasagna::LasagnaNormalSampler>(*dataset);
            session.dataset = std::move(dataset);
            session.normalSampler = std::move(sampler);
        } catch (const std::exception& ex) {
            showError(tr("Invalid Lasagna dataset: %1").arg(QString::fromStdString(ex.what())));
            return false;
        }
        vpkg->setSelectedLasagnaDataset(selected);
    } else {
        if (!session.normalSampler || session.selectedManifestPath != manifestPath) {
            try {
                auto dataset = std::make_shared<vc::lasagna::LasagnaDataset>(
                    vc::lasagna::LasagnaDataset::open(manifestPath));
                auto sampler = std::make_shared<vc::lasagna::LasagnaNormalSampler>(*dataset);
                session.dataset = std::move(dataset);
                session.normalSampler = std::move(sampler);
            } catch (const std::exception& ex) {
                showError(tr("Invalid selected Lasagna dataset: %1")
                              .arg(QString::fromStdString(ex.what())));
                return false;
            }
        }
    }

    session.selectedDatasetLocation = selected;
    session.selectedManifestPath = manifestPath;
    return true;
}

void LineAnnotationController::startOptimization(LineAnnotationSession& session)
{
    if (session.controlPoints.empty()) {
        return;
    }
    session.taskState = LineAnnotationSession::TaskState::Running;
    session.error.clear();
    auto seedIt = std::find_if(session.controlPoints.begin(),
                               session.controlPoints.end(),
                               [](const vc::lasagna::LineControlPoint& control) {
                                   return control.isSeed;
                               });
    if (seedIt == session.controlPoints.end()) {
        session.controlPoints.front().isSeed = true;
        seedIt = session.controlPoints.begin();
    }
    session.seedPoint = seedIt->volumePoint;

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
    auto controlPoints = session.controlPoints;
    std::vector<cv::Vec3d> initialLinePoints;
    initialLinePoints.reserve(session.optimizedLine.points.size());
    for (const auto& point : session.optimizedLine.points) {
        initialLinePoints.push_back(point.position);
    }
    const cv::Vec3d sourceSliceNormal = session.sourceSliceNormal;
    const InitialDirectionMode directionMode = session.initialDirectionMode;
    auto dataset = session.dataset;
    auto normalSampler = session.normalSampler;
    watcher->setFuture(QtConcurrent::run([factory,
                                           manifestPath,
                                           controlPoints,
                                           initialLinePoints,
                                           sourceSliceNormal,
                                           directionMode,
                                           dataset,
                                           normalSampler]() mutable {
        if (factory) {
            return factory(manifestPath,
                           std::move(controlPoints),
                           std::move(initialLinePoints),
                           sourceSliceNormal,
                           directionMode);
        }
        if (normalSampler) {
            (void)dataset;
            return optimizeLineWithSampler(manifestPath,
                                           std::move(controlPoints),
                                           std::move(initialLinePoints),
                                           sourceSliceNormal,
                                           directionMode,
                                           *normalSampler);
        }
        return optimizeLineFromManifest(manifestPath,
                                        std::move(controlPoints),
                                        std::move(initialLinePoints),
                                        sourceSliceNormal,
                                        directionMode);
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
        session.controlPoints = std::move(task.controlPoints);
        for (auto& control : session.controlPoints) {
            double bestDistance = std::numeric_limits<double>::infinity();
            int bestIndex = -1;
            for (size_t i = 0; i < session.optimizedLine.points.size(); ++i) {
                const cv::Vec3d delta = session.optimizedLine.points[i].position - control.volumePoint;
                const double distance = std::sqrt(delta.dot(delta));
                if (distance < bestDistance) {
                    bestDistance = distance;
                    bestIndex = static_cast<int>(i);
                }
            }
            if (bestIndex >= 0) {
                control.optimizedIndex = bestIndex;
                control.linePosition = static_cast<double>(bestIndex);
                control.volumePoint = session.optimizedLine.points[static_cast<size_t>(bestIndex)].position;
                const bool matchesFocusedControl = session.focusedControlPoint.has_value() &&
                    std::sqrt((control.volumePoint - *session.focusedControlPoint).dot(
                        control.volumePoint - *session.focusedControlPoint)) <= 1.0e-6;
                if (std::abs(session.focusedLinePosition - control.linePosition) <= 0.5 ||
                    matchesFocusedControl) {
                    session.focusedLinePosition = control.linePosition;
                }
            }
            if (control.isSeed) {
                session.seedPoint = control.volumePoint;
            }
        }
        if (!materializeGeneratedViews(session)) {
            session.taskState = LineAnnotationSession::TaskState::Failed;
            return;
        }
        Logger()->info("Line annotation Lasagna optimization complete: seed=[{}, {}, {}] points={} iterations={} initial_cost={} final_cost={} valid_normals={} invalid_normals={} converged={}",
                       session.seedPoint[0],
                       session.seedPoint[1],
                       session.seedPoint[2],
                       session.optimizedLine.points.size(),
                       session.optimizationReport.iterations,
                       session.optimizationReport.initialCost,
                       session.optimizationReport.finalCost,
                       session.optimizationReport.validNormalSamples,
                       session.optimizationReport.invalidNormalSamples,
                       session.optimizationReport.converged);
        if (!session.optimizationReport.message.empty()) {
            Logger()->info("Line annotation Lasagna Ceres report:\n{}",
                           session.optimizationReport.message);
        }
        if (!session.optimizationReport.finalLosses.empty()) {
            std::ostringstream losses;
            losses.imbue(std::locale::classic());
            losses << std::scientific << std::setprecision(3);
            losses << "Line annotation Lasagna final loss breakdown:\n"
                   << "term                 n      weight    raw_cost weighted_cost\n";
            for (const auto& loss : session.optimizationReport.finalLosses) {
                losses << std::left << std::setw(18) << loss.name
                       << std::right << std::setw(6) << loss.residuals
                       << std::setw(12) << loss.weight
                       << std::setw(12) << loss.rawCost
                       << std::setw(14) << loss.weightedCost
                       << '\n';
            }
            Logger()->info("{}", losses.str());
        }
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

    _state->setSurface("line-surface", views.lineSurface);
    _state->setSurface("line-side-slice", views.lineSideSlice);
    session.generatedSurfaceNames.push_back("line-surface");
    session.generatedSurfaceNames.push_back("line-side-slice");

    std::vector<cv::Vec3f> linePoints;
    linePoints.reserve(session.optimizedLine.points.size());
    for (const auto& point : session.optimizedLine.points) {
        linePoints.push_back({static_cast<float>(point.position[0]),
                              static_cast<float>(point.position[1]),
                              static_cast<float>(point.position[2])});
    }

    const cv::Vec3f seedPoint{static_cast<float>(session.seedPoint[0]),
                              static_cast<float>(session.seedPoint[1]),
                              static_cast<float>(session.seedPoint[2])};

    LineAnnotationDialog::GeneratedViews generatedViews;
    generatedViews.lineSurfaceName = "line-surface";
    generatedViews.lineSurfaceTitle = tr("Line Surface");
    generatedViews.lineSideSliceName = "line-side-slice";
    generatedViews.lineSideSliceTitle = tr("Line Side Slice");
    generatedViews.linePoints = std::move(linePoints);
    generatedViews.lineUpVectors = views.lineUpVectors;
    generatedViews.seedPoint = seedPoint;
    generatedViews.seedLineIndex = static_cast<int>(session.optimizedLine.points.size() / 2);
    for (const auto& control : session.controlPoints) {
        LineAnnotationDialog::GeneratedOverlay::ControlPointMarker marker;
        marker.point = {static_cast<float>(control.volumePoint[0]),
                        static_cast<float>(control.volumePoint[1]),
                        static_cast<float>(control.volumePoint[2])};
        marker.linePosition = std::isfinite(control.linePosition)
            ? control.linePosition
            : static_cast<double>(control.optimizedIndex);
        marker.isSeed = control.isSeed;
        generatedViews.controlPoints.push_back(marker);
        if (control.isSeed && std::isfinite(marker.linePosition)) {
            generatedViews.seedLineIndex = static_cast<int>(std::llround(marker.linePosition));
        }
    }
    generatedViews.initialCenterIndex = static_cast<int>(std::llround(std::clamp(
        session.focusedLinePosition,
        0.0,
        static_cast<double>(std::max<size_t>(1, session.optimizedLine.points.size()) - 1))));

    generatedViews.currentCutName = "line-current-cut";
    generatedViews.currentCutSurface = std::make_shared<PlaneSurface>(
        seedPoint,
        cv::Vec3f{1.0f, 0.0f, 0.0f});
    _state->setSurface(generatedViews.currentCutName, generatedViews.currentCutSurface);
    session.generatedSurfaceNames.push_back(generatedViews.currentCutName);

    generatedViews.bottomCutSurfaces.reserve(7);
    for (int i = 0; i < 7; ++i) {
        const std::string surfaceName = "line-bottom-cut-" + std::to_string(i);
        auto plane = std::make_shared<PlaneSurface>(seedPoint, cv::Vec3f{1.0f, 0.0f, 0.0f});
        _state->setSurface(surfaceName, plane);
        session.generatedSurfaceNames.push_back(surfaceName);
        generatedViews.bottomCutSurfaces.push_back({surfaceName, std::move(plane)});
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

    if (!pane->dialog->setGeneratedLineViews(generatedViews, camera)) {
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

void LineAnnotationController::handleShowAsMesh(const std::string& surfaceName)
{
    auto* pane = paneForSurface(surfaceName);
    if (!pane || !pane->session) {
        return;
    }

    auto& session = *pane->session;
    if (session.taskState != LineAnnotationSession::TaskState::Succeeded) {
        showError(tr("Run line optimization before exporting generated meshes."));
        return;
    }

    try {
        const auto savedPaths = saveGeneratedQuadMeshes(session);
        if (savedPaths.empty()) {
            showError(tr("No generated line quad meshes are available to export."));
            return;
        }

        QStringList labels;
        labels.reserve(static_cast<int>(savedPaths.size()));
        for (const auto& path : savedPaths) {
            labels.push_back(QString::fromStdString(path.filename().string()));
        }
        QMessageBox::information(_parentWidget,
                                 tr("Line Annotation"),
                                 tr("Saved generated mesh surfaces in paths:\n%1")
                                     .arg(labels.join(QStringLiteral("\n"))));
    } catch (const std::exception& ex) {
        showError(tr("Could not save generated line meshes: %1")
                      .arg(QString::fromStdString(ex.what())));
    }
}

fs::path LineAnnotationController::resolveMeshExportPathsDir() const
{
    if (!_state || !_state->vpkg()) {
        throw std::runtime_error("No volume package loaded.");
    }

    auto vpkg = _state->vpkg();
    fs::path pathsDir = vpkg->outputSegmentsPath();
    const fs::path volpkgRoot = vpkg->path().empty()
        ? fs::path(vpkg->getVolpkgDirectory())
        : vpkg->path().parent_path();

    if (pathsDir.empty()) {
        if (volpkgRoot.empty()) {
            throw std::runtime_error("Volume package path is unavailable.");
        }
        pathsDir = volpkgRoot / "paths";
    }

    std::error_code ec;
    fs::create_directories(pathsDir, ec);
    if (ec) {
        throw std::runtime_error("Failed to create paths directory " +
                                 pathsDir.string() + ": " + ec.message());
    }

    bool hasEntry = false;
    const fs::path canonicalPaths = fs::weakly_canonical(pathsDir, ec);
    for (const auto& entryPath : vpkg->availableSegmentPaths()) {
        std::error_code entryEc;
        if (fs::weakly_canonical(entryPath, entryEc) == canonicalPaths && !entryEc) {
            hasEntry = true;
            break;
        }
    }
    if (!hasEntry && !volpkgRoot.empty() && pathsDir == volpkgRoot / "paths") {
        vpkg->addSegmentsEntry("paths");
    }

    return pathsDir;
}

fs::path LineAnnotationController::nextMeshExportPath(const fs::path& pathsDir,
                                                      const std::string& stem) const
{
    const std::string timestamp =
        QDateTime::currentDateTimeUtc().toString(QStringLiteral("yyyyMMdd_HHmmss")).toStdString();
    std::string base = "line_annotation_" + timestamp + "_" + stem;
    fs::path candidate = pathsDir / base;
    int suffix = 1;
    while (fs::exists(candidate)) {
        candidate = pathsDir / (base + "_" + std::to_string(suffix++));
    }
    return candidate;
}

std::vector<fs::path> LineAnnotationController::saveGeneratedQuadMeshes(LineAnnotationSession& session)
{
    if (!_state) {
        throw std::runtime_error("No active application state.");
    }

    const fs::path pathsDir = resolveMeshExportPathsDir();
    const std::vector<std::pair<std::string, std::string>> exports = {
        {"line-surface", "surface"},
        {"line-side-slice", "side_slice"},
    };

    std::vector<fs::path> savedPaths;
    for (const auto& [surfaceName, stem] : exports) {
        auto surface = std::dynamic_pointer_cast<QuadSurface>(_state->surface(surfaceName));
        if (!surface) {
            continue;
        }

        auto clone = std::make_shared<QuadSurface>(surface->rawPoints().clone(), surface->scale());
        clone->meta = surface->meta;

        const fs::path outputPath = nextMeshExportPath(pathsDir, stem);
        const std::string outputName = outputPath.filename().string();
        clone->save(outputPath.string(), outputName, false);

        if (_surfacePanel) {
            _surfacePanel->addSingleSegmentation(outputName);
        } else if (_state->vpkg()) {
            (void)_state->vpkg()->addSingleSegmentation(outputName);
        }
        savedPaths.push_back(outputPath);
    }

    if (!savedPaths.empty() && _state->vpkg()) {
        _state->emitSurfacesChanged();
    }

    Logger()->info("Line annotation saved {} generated mesh surface(s) to {}",
                   savedPaths.size(), pathsDir.string());
    return savedPaths;
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
            if (!pane.session->suppressFiberSave &&
                pane.session->taskState == LineAnnotationSession::TaskState::Succeeded &&
                !pane.session->optimizedLine.points.empty() &&
                !pane.session->controlPoints.empty()) {
                saveSessionAsFiber(*pane.session);
            }
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
        if (pane.surfaceName == surfaceName) {
            return true;
        }
        return pane.session &&
               std::find(pane.session->generatedSurfaceNames.begin(),
                         pane.session->generatedSurfaceNames.end(),
                         surfaceName) != pane.session->generatedSurfaceNames.end();
    });
    return it == _panes.end() ? nullptr : &*it;
}

const LineAnnotationController::PaneRecord*
LineAnnotationController::paneForSurface(const std::string& surfaceName) const
{
    auto it = std::find_if(_panes.begin(), _panes.end(), [&surfaceName](const PaneRecord& pane) {
        if (pane.surfaceName == surfaceName) {
            return true;
        }
        return pane.session &&
               std::find(pane.session->generatedSurfaceNames.begin(),
                         pane.session->generatedSurfaceNames.end(),
                         surfaceName) != pane.session->generatedSurfaceNames.end();
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
    std::vector<vc::lasagna::LineControlPoint> controlPoints,
    std::vector<cv::Vec3d> initialLinePoints,
    cv::Vec3d sourceSliceNormal,
    InitialDirectionMode directionMode) const
{
    if (_optimizationTaskFactory) {
        return _optimizationTaskFactory(std::move(manifestPath),
                                        std::move(controlPoints),
                                        std::move(initialLinePoints),
                                        sourceSliceNormal,
                                        directionMode);
    }
    return optimizeLineFromManifest(std::move(manifestPath),
                                    std::move(controlPoints),
                                    std::move(initialLinePoints),
                                    sourceSliceNormal,
                                    directionMode);
}

void LineAnnotationController::loadFibersForCurrentPackage()
{
    _fibers.clear();
    if (!_state || !_state->vpkg()) {
        emitFiberSummaries();
        return;
    }

    const fs::path dir = fibersDir();
    std::error_code ec;
    if (!fs::exists(dir, ec)) {
        emitFiberSummaries();
        return;
    }

    for (const auto& entry : fs::directory_iterator(dir, ec)) {
        if (ec) {
            break;
        }
        if (!entry.is_regular_file() || entry.path().extension() != ".json") {
            continue;
        }
        try {
            if (auto fiber = loadFiberFile(entry.path())) {
                _fibers.push_back(std::move(*fiber));
            }
        } catch (const std::exception& ex) {
            Logger()->warn("Skipping invalid VC3D fiber file {}: {}",
                           entry.path().string(),
                           ex.what());
        }
    }
    std::sort(_fibers.begin(), _fibers.end(), [](const StoredFiber& a, const StoredFiber& b) {
        return a.id < b.id;
    });
    emitFiberSummaries();
}

void LineAnnotationController::emitFiberSummaries()
{
    emit fibersChanged(fiberSummaries());
}

fs::path LineAnnotationController::fibersDir() const
{
    if (!_state || !_state->vpkg()) {
        return {};
    }
    const auto vpkg = _state->vpkg();
    const fs::path projectPath = vpkg->path();
    const fs::path root = projectPath.empty()
        ? fs::path(vpkg->getVolpkgDirectory())
        : projectPath.parent_path();
    return root / "fibers";
}

fs::path LineAnnotationController::fiberPath(uint64_t fiberId) const
{
    return fibersDir() / (std::to_string(fiberId) + ".json");
}

uint64_t LineAnnotationController::nextFiberId() const
{
    uint64_t id = 1;
    for (const auto& fiber : _fibers) {
        id = std::max(id, fiber.id + 1);
    }
    return id;
}

double LineAnnotationController::lineLengthVx(const std::vector<cv::Vec3d>& points)
{
    double length = 0.0;
    for (size_t i = 1; i < points.size(); ++i) {
        const cv::Vec3d delta = points[i] - points[i - 1];
        const double step = std::sqrt(delta.dot(delta));
        if (std::isfinite(step)) {
            length += step;
        }
    }
    return length;
}

vc::lasagna::LineModel LineAnnotationController::lineModelFromPoints(
    const std::vector<cv::Vec3d>& points,
    const vc::lasagna::NormalSampler* normalSampler)
{
    if (points.empty()) {
        throw std::runtime_error("Fiber has no line points");
    }
    if (!normalSampler) {
        throw std::runtime_error("No Lasagna normal sampler is available for this fiber");
    }

    std::vector<vc::lasagna::NormalSampleWithDerivative> samples;
    normalSampler->sampleNormalBatch(points, false, samples);
    if (samples.size() != points.size()) {
        throw std::runtime_error("Normal sampler returned the wrong number of samples");
    }

    vc::lasagna::LineModel model;
    model.points.reserve(points.size());
    int bestAnchor = -1;
    double bestAnchorDistance = std::numeric_limits<double>::infinity();
    const double center = static_cast<double>(points.size() - 1) * 0.5;
    for (size_t i = 0; i < points.size(); ++i) {
        vc::lasagna::LinePoint linePoint;
        linePoint.position = points[i];
        linePoint.sampledNormal = samples[i].sample;
        linePoint.sampledNormal.normal = normalizedOrZero(linePoint.sampledNormal.normal);
        linePoint.sampledNormal.valid =
            linePoint.sampledNormal.valid &&
            finiteDirection(linePoint.sampledNormal.normal);
        linePoint.valid = linePoint.sampledNormal.valid;
        if (linePoint.valid) {
            const double distance = std::abs(static_cast<double>(i) - center);
            if (distance < bestAnchorDistance) {
                bestAnchorDistance = distance;
                bestAnchor = static_cast<int>(i);
            }
        }
        model.points.push_back(std::move(linePoint));
    }
    if (bestAnchor < 0) {
        throw std::runtime_error("Fiber line points have no valid sampled normals");
    }
    model.displayFrameAnchorIndex = bestAnchor;
    return model;
}

void LineAnnotationController::saveSessionAsFiber(LineAnnotationSession& session)
{
    try {
        StoredFiber fiber;
        fiber.id = session.fiberId == 0 ? nextFiberId() : session.fiberId;
        fiber.controlPoints.reserve(session.controlPoints.size());
        auto controls = session.controlPoints;
        std::stable_sort(controls.begin(),
                         controls.end(),
                         [](const vc::lasagna::LineControlPoint& a,
                            const vc::lasagna::LineControlPoint& b) {
                             return a.linePosition < b.linePosition;
                         });
        for (const auto& control : controls) {
            fiber.controlPoints.push_back(control.volumePoint);
        }
        fiber.linePoints.reserve(session.optimizedLine.points.size());
        for (const auto& point : session.optimizedLine.points) {
            fiber.linePoints.push_back(point.position);
        }
        saveFiber(fiber);
        session.fiberId = fiber.id;

        auto it = std::find_if(_fibers.begin(), _fibers.end(), [&fiber](const StoredFiber& existing) {
            return existing.id == fiber.id;
        });
        if (it == _fibers.end()) {
            _fibers.push_back(std::move(fiber));
        } else {
            *it = std::move(fiber);
        }
        emitFiberSummaries();
    } catch (const std::exception& ex) {
        showError(tr("Could not save fiber: %1").arg(QString::fromStdString(ex.what())));
    }
}

void LineAnnotationController::saveFiber(const StoredFiber& fiber) const
{
    const fs::path dir = fibersDir();
    if (dir.empty()) {
        throw std::runtime_error("No volume package is loaded");
    }

    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) {
        throw std::runtime_error("Failed to create fibers directory " +
                                 dir.string() + ": " + ec.message());
    }

    nlohmann::json root = nlohmann::json::object();
    root["type"] = "vc3d_fiber";
    root["version"] = 1;
    root["control_points"] = nlohmann::json::array();
    root["line_points"] = nlohmann::json::array();
    for (const auto& point : fiber.controlPoints) {
        root["control_points"].push_back(pointToJson(point));
    }
    for (const auto& point : fiber.linePoints) {
        root["line_points"].push_back(pointToJson(point));
    }

    const fs::path finalPath = fiberPath(fiber.id);
    const fs::path tempPath = finalPath.string() + ".tmp";
    {
        std::ofstream out(tempPath);
        if (!out) {
            throw std::runtime_error("Failed to open " + tempPath.string());
        }
        out << root.dump(2) << '\n';
    }
    fs::rename(tempPath, finalPath, ec);
    if (ec) {
        fs::remove(tempPath);
        throw std::runtime_error("Failed to replace " + finalPath.string() + ": " + ec.message());
    }
}

std::optional<LineAnnotationController::StoredFiber> LineAnnotationController::loadFiberFile(
    const fs::path& path) const
{
    std::string stem = path.stem().string();
    if (stem.rfind("fiber_", 0) == 0) {
        stem = stem.substr(6);
    }
    if (stem.empty() || !std::all_of(stem.begin(), stem.end(), [](char ch) {
            return ch >= '0' && ch <= '9';
        })) {
        return std::nullopt;
    }

    StoredFiber fiber;
    fiber.id = std::stoull(stem);

    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Failed to open fiber file");
    }
    const nlohmann::json root = nlohmann::json::parse(in);
    const std::string type = root.value("type", std::string{});
    if (type != "vc3d_fiber") {
        return std::nullopt;
    }
    if (root.value("version", 0) != 1) {
        throw std::runtime_error("Unsupported vc3d_fiber version");
    }

    const auto& controls = root.at("control_points");
    const auto& linePoints = root.at("line_points");
    if (!controls.is_array() || !linePoints.is_array()) {
        throw std::runtime_error("control_points and line_points must be arrays");
    }

    fiber.controlPoints.reserve(controls.size());
    for (const auto& point : controls) {
        fiber.controlPoints.push_back(pointFromJson(point));
    }
    fiber.linePoints.reserve(linePoints.size());
    for (const auto& point : linePoints) {
        fiber.linePoints.push_back(pointFromJson(point));
    }
    return fiber;
}

void LineAnnotationController::showError(const QString& message) const
{
    if (_parentWidget) {
        QMessageBox::warning(_parentWidget, tr("Line Annotation"), message);
    } else {
        Logger()->warn("Line Annotation: {}", message.toStdString());
    }
}
